#!/usr/bin/env python

import copy
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import datasets
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _artifact_registry import infer_checkpoint_metadata, make_artifact_id, register_artifact

PROGRESS_LOG_EVERY_BATCHES = 100
PASSTHROUGH_FEATURE_KEYS = ("index", "episode_index", "task_index", "frame_index", "timestamp", "action")


@dataclass
class AnalysisLatentExportConfig:
    policy: PreTrainedConfig | None = None
    dataset_repo_id: str | None = None
    dataset_root: str | None = None
    episodes: list[int] | None = None
    output_dir: Path | None = None
    output_repo_id: str | None = None
    feature_prefix: str = "latent_labels"
    batch_size: int = 32
    num_workers: int = 8
    rename_map: dict[str, str] | None = None
    force: bool = False
    max_valid_samples: int | None = None

    def validate(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("Policy is not configured. Please specify a checkpoint with `--policy.path`.")

        cli_overrides = [
            arg
            for arg in (parser.get_cli_overrides("policy") or [])
            if not arg.startswith("--discover_packages_path=")
        ]
        self.policy = PreTrainedConfig.from_pretrained(
            policy_path,
            local_files_only=True,
            cli_overrides=cli_overrides,
        )
        self.policy.pretrained_path = Path(policy_path)

        if not self.dataset_repo_id:
            raise ValueError("Please specify `--dataset_repo_id`.")
        if self.output_dir is None:
            raise ValueError("Please specify `--output_dir`.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}.")
        if self.max_valid_samples is not None and self.max_valid_samples < 1:
            raise ValueError(f"max_valid_samples must be >= 1, got {self.max_valid_samples}.")
        if self.feature_prefix.startswith("observation."):
            raise ValueError(
                "feature_prefix must not use observation.*. "
                "Use a top-level namespace such as `latent_labels` or `lam_lapa`."
            )
        if self.output_repo_id is None:
            self.output_repo_id = f"{self.dataset_repo_id}_latent_analysis"

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        return
    if not force:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --force to overwrite it.")
    shutil.rmtree(output_dir)


def _get_required_method(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if callable(method):
        return method
    raise TypeError(
        f"Policy type {obj.config.type!r} does not implement `{method_name}()`. "
        "Install a latent-export-capable policy plugin."
    )


def _normalize_export_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise TypeError(f"prepare_latent_export() must return a dict, got {type(plan)}.")
    required_keys = {"delta_timestamps", "representations"}
    missing = required_keys.difference(plan)
    if missing:
        raise KeyError(f"prepare_latent_export() is missing keys: {sorted(missing)}")

    representations = {}
    for name, spec in plan["representations"].items():
        representations[name] = {
            "shape": tuple(int(dim) for dim in spec["shape"]),
            "dtype": np.dtype(spec["dtype"]),
            "invalid_fill_value": spec["invalid_fill_value"],
        }
    return {
        "delta_timestamps": plan["delta_timestamps"],
        "representations": representations,
    }


def _normalize_export_batch(batch_out: Any) -> dict[str, Any]:
    if not isinstance(batch_out, dict):
        raise TypeError(f"export_latent_labels() must return a dict, got {type(batch_out)}.")
    if "labels_by_name" not in batch_out or "valid_mask" not in batch_out:
        raise KeyError("export_latent_labels() must return `labels_by_name` and `valid_mask`.")

    valid_mask = batch_out["valid_mask"]
    if not torch.is_tensor(valid_mask):
        valid_mask = torch.as_tensor(valid_mask)

    labels_by_name = {}
    for name, labels in batch_out["labels_by_name"].items():
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        labels_by_name[name] = labels
    return {"labels_by_name": labels_by_name, "valid_mask": valid_mask}


def _to_numpy_column(values: Any) -> np.ndarray:
    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    array = values.detach().cpu().numpy()
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return array


def _expand_compact_labels(
    *,
    compact_batch: dict[str, Any],
    plan: dict[str, Any],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    valid_mask = compact_batch["valid_mask"]
    if valid_mask.ndim != 1 or valid_mask.shape[0] != batch_size:
        raise ValueError("export_latent_labels() must return `valid_mask` with shape [batch_size].")
    valid_mask_np = valid_mask.detach().cpu().numpy().astype(bool, copy=False)

    labels_by_name = compact_batch["labels_by_name"]
    expected_names = set(plan["representations"])
    actual_names = set(labels_by_name)
    if actual_names != expected_names:
        raise KeyError(
            f"export_latent_labels() returned label names {sorted(actual_names)}, expected {sorted(expected_names)}."
        )

    expanded = {}
    valid_count = int(valid_mask_np.sum())
    for name, spec in plan["representations"].items():
        labels_tensor = labels_by_name[name]
        if labels_tensor.shape[0] != valid_count:
            raise ValueError(
                f"export_latent_labels() returned {name!r} with a batch size that does not match the number of valid rows."
            )
        full = np.full(
            (batch_size, *spec["shape"]),
            spec["invalid_fill_value"],
            dtype=spec["dtype"],
        )
        if valid_count > 0:
            full[valid_mask_np] = labels_tensor.detach().cpu().numpy().astype(spec["dtype"], copy=False)
        expanded[name] = full
    return expanded, valid_mask_np


def _keep_count_through_episode(episode_index: np.ndarray, target_episode_index: int) -> int:
    return int(np.searchsorted(episode_index, target_episode_index, side="right"))


def _normalize_feature_spec(spec: dict[str, Any], fps: float | None = None) -> dict[str, Any]:
    out = copy.deepcopy(spec)
    out["shape"] = tuple(out["shape"])
    if fps is not None and "fps" not in out:
        out["fps"] = fps
    return out


def _build_output_features(
    *,
    source_info: dict[str, Any],
    plan: dict[str, Any],
    feature_prefix: str,
    passthrough_keys: list[str],
    fps: float,
) -> dict[str, dict[str, Any]]:
    features = {
        key: _normalize_feature_spec(source_info["features"][key])
        for key in passthrough_keys
    }
    for name, spec in plan["representations"].items():
        features[f"{feature_prefix}.{name}"] = {
            "dtype": spec["dtype"].name,
            "shape": tuple(spec["shape"]),
            "names": None,
            "fps": fps,
        }
    features[f"{feature_prefix}.valid"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
        "fps": fps,
    }
    return features


def _make_output_info(
    *,
    source_info: dict[str, Any],
    output_repo_id: str,
    output_features: dict[str, dict[str, Any]],
    total_frames: int,
    total_episodes: int,
    total_tasks: int,
) -> dict[str, Any]:
    return {
        "codebase_version": source_info.get("codebase_version", "v3.0"),
        "robot_type": source_info.get("robot_type"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": source_info.get("chunks_size", 1000),
        "fps": source_info["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": output_features,
        "data_files_size_in_mb": source_info.get("data_files_size_in_mb", 100),
        "repo_id": output_repo_id,
        "analysis_only": True,
    }


def _write_analysis_dataset_root(
    *,
    output_dir: Path,
    info: dict[str, Any],
    data_columns: dict[str, np.ndarray],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_info(info, output_dir)

    hf_features = get_hf_features_from_features(info["features"])
    dataset = datasets.Dataset.from_dict(data_columns, features=hf_features)

    data_path = output_dir / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(data_path))
    return data_path


def export_latent_analysis_dataset(cfg: AnalysisLatentExportConfig) -> None:
    cfg.validate()
    if cfg.policy is None:
        raise ValueError("Policy config was not loaded.")
    if cfg.output_dir is None:
        raise ValueError("output_dir was not configured.")
    if cfg.output_repo_id is None:
        raise ValueError("output_repo_id was not configured.")
    if cfg.dataset_repo_id is None:
        raise ValueError("dataset_repo_id was not configured.")

    output_dir = cfg.output_dir.resolve()
    _prepare_output_dir(output_dir, cfg.force)

    source_dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root)
    source_info = copy.deepcopy(source_dataset.meta.info)
    policy = make_policy(cfg.policy, ds_meta=source_dataset.meta, rename_map=cfg.rename_map)
    prepare_latent_export = _get_required_method(policy, "prepare_latent_export")
    export_latent_labels = _get_required_method(policy, "export_latent_labels")
    plan = _normalize_export_plan(prepare_latent_export(source_dataset.meta))

    label_dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
        episodes=cfg.episodes,
        delta_timestamps=plan["delta_timestamps"],
    )
    dataloader = torch.utils.data.DataLoader(
        label_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=str(cfg.policy.device).startswith("cuda"),
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    passthrough_keys = [key for key in PASSTHROUGH_FEATURE_KEYS if key in source_info["features"]]
    output_columns = {key: [] for key in passthrough_keys}
    for name in plan["representations"]:
        output_columns[f"{cfg.feature_prefix}.{name}"] = []
    output_columns[f"{cfg.feature_prefix}.valid"] = []

    logging.info(
        "Analysis-only latent export setup:\n%s",
        pformat(
            {
                "policy_type": cfg.policy.type,
                "policy_path": str(cfg.policy.pretrained_path),
                "dataset_repo_id": cfg.dataset_repo_id,
                "dataset_root": cfg.dataset_root,
                "episodes": cfg.episodes,
                "output_dir": str(output_dir),
                "feature_prefix": cfg.feature_prefix,
                "representation_names": list(plan["representations"]),
                "passthrough_keys": passthrough_keys,
                "batch_size": cfg.batch_size,
                "num_workers": cfg.num_workers,
                "rename_map": cfg.rename_map,
                "delta_timestamps": plan["delta_timestamps"],
                "max_valid_samples": cfg.max_valid_samples,
            }
        ),
    )

    total_written_valid = 0
    total_written_rows = 0
    stop_after_episode_index: int | None = None
    policy.eval()
    start_time = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            episode_index_batch = _to_numpy_column(batch["episode_index"]).astype(np.int64, copy=False)
            if stop_after_episode_index is not None and int(episode_index_batch[0]) > stop_after_episode_index:
                break

            compact_batch = _normalize_export_batch(export_latent_labels(batch))
            batch_size = int(_to_numpy_column(batch["index"]).shape[0])
            expanded_labels, valid_mask = _expand_compact_labels(
                compact_batch=compact_batch,
                plan=plan,
                batch_size=batch_size,
            )

            keep_count = batch_size
            if cfg.max_valid_samples is not None:
                if stop_after_episode_index is None:
                    cumulative_valid = total_written_valid + np.cumsum(valid_mask.astype(np.int64, copy=False))
                    if cumulative_valid.size and int(cumulative_valid[-1]) >= cfg.max_valid_samples:
                        threshold_row = int(np.searchsorted(cumulative_valid, cfg.max_valid_samples, side="left"))
                        stop_after_episode_index = int(episode_index_batch[threshold_row])
                if stop_after_episode_index is not None:
                    keep_count = _keep_count_through_episode(episode_index_batch, stop_after_episode_index)
                    if keep_count == 0:
                        break

            valid_prefix = valid_mask[:keep_count].astype(np.int64, copy=False)
            for key in passthrough_keys:
                values = _to_numpy_column(batch[key])[:keep_count]
                output_columns[key].append(values)
            for name, values in expanded_labels.items():
                output_columns[f"{cfg.feature_prefix}.{name}"].append(values[:keep_count])
            output_columns[f"{cfg.feature_prefix}.valid"].append(valid_prefix)

            total_written_rows += keep_count
            total_written_valid += int(valid_prefix.sum())

            if batch_idx % PROGRESS_LOG_EVERY_BATCHES == 0 or batch_idx == len(dataloader):
                elapsed_s = max(time.perf_counter() - start_time, 1e-9)
                processed_rows = min(batch_idx * cfg.batch_size, len(label_dataset))
                rate = processed_rows / elapsed_s
                logging.info(
                    "Progress: rows=%d/%d written_rows=%d valid=%d rate=%.1f rows/s",
                    processed_rows,
                    len(label_dataset),
                    total_written_rows,
                    total_written_valid,
                    rate,
                )

            if stop_after_episode_index is not None and keep_count < batch_size:
                break

    if total_written_rows == 0:
        raise ValueError("No rows were written to the analysis dataset.")

    consolidated = {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in output_columns.items()
        if chunks
    }
    order = np.argsort(consolidated["index"])
    consolidated = {name: values[order] for name, values in consolidated.items()}

    total_episodes = int(np.unique(consolidated["episode_index"]).shape[0])
    total_tasks = (
        int(np.unique(consolidated["task_index"]).shape[0])
        if "task_index" in consolidated
        else int(source_info.get("total_tasks", 0))
    )
    output_features = _build_output_features(
        source_info=source_info,
        plan=plan,
        feature_prefix=cfg.feature_prefix,
        passthrough_keys=passthrough_keys,
        fps=float(source_info["fps"]),
    )
    output_info = _make_output_info(
        source_info=source_info,
        output_repo_id=cfg.output_repo_id,
        output_features=output_features,
        total_frames=total_written_rows,
        total_episodes=total_episodes,
        total_tasks=total_tasks,
    )
    data_path = _write_analysis_dataset_root(
        output_dir=output_dir,
        info=output_info,
        data_columns=consolidated,
    )

    checkpoint_meta = infer_checkpoint_metadata(cfg.policy.pretrained_path)
    label_manifest = {
        "policy_type": cfg.policy.type,
        "policy_path": str(cfg.policy.pretrained_path),
        "source_dataset_repo_id": cfg.dataset_repo_id,
        "source_dataset_root": cfg.dataset_root,
        "episodes": cfg.episodes,
        "output_repo_id": cfg.output_repo_id,
        "output_dir": str(output_dir),
        "analysis_only": True,
        "feature_prefix": cfg.feature_prefix,
        "feature_names": {
            name: f"{cfg.feature_prefix}.{name}" for name in plan["representations"]
        },
        "valid_feature_name": f"{cfg.feature_prefix}.valid",
        "passthrough_feature_names": passthrough_keys,
        "delta_timestamps": plan["delta_timestamps"],
        "num_rows": total_written_rows,
        "num_valid_labels": total_written_valid,
        "data_file": str(data_path),
    }
    label_manifest_path = output_dir / "label_manifest.json"
    label_manifest_path.write_text(json.dumps(label_manifest, indent=2) + "\n")

    export_manifest_path = output_dir / "export_manifest.json"
    export_manifest = {
        "artifact_type": "latent_export",
        "export_kind": "analysis_only",
        "suite_name": "latent_export",
        "suite_version": "v1",
        "artifact_id": make_artifact_id(
            suite_name="latent_export",
            suite_version="v1",
            checkpoint_id=checkpoint_meta["source_checkpoint_id"],
            output_label=output_dir.name,
        ),
        **checkpoint_meta,
        "analysis_only": True,
        "script_path": str(Path(__file__).resolve()),
        "cli_args": list(sys.argv[1:]),
        "source_dataset_repo_id": cfg.dataset_repo_id,
        "source_dataset_root": cfg.dataset_root,
        "episodes": cfg.episodes,
        "output_repo_id": cfg.output_repo_id,
        "output_path": str(output_dir),
        "feature_prefix": cfg.feature_prefix,
        "feature_names": {
            name: f"{cfg.feature_prefix}.{name}" for name in plan["representations"]
        },
        "valid_feature_name": f"{cfg.feature_prefix}.valid",
        "passthrough_feature_names": passthrough_keys,
        "delta_timestamps": plan["delta_timestamps"],
        "num_rows": total_written_rows,
        "num_valid_labels": total_written_valid,
        "data_file": str(data_path),
        "label_manifest_path": str(label_manifest_path),
    }
    register_artifact(
        manifest_path=export_manifest_path,
        manifest=export_manifest,
        registry_candidates=[output_dir, cfg.policy.pretrained_path, cfg.dataset_root],
    )

    logging.info("Finished analysis-only export to %s", output_dir)
    logging.info("Data written to %s", data_path)
    logging.info("Manifests written to %s and %s", label_manifest_path, export_manifest_path)


@parser.wrap()
def main(cfg: AnalysisLatentExportConfig) -> None:
    register_third_party_plugins()
    init_logging()
    export_latent_analysis_dataset(cfg)


if __name__ == "__main__":
    main()
