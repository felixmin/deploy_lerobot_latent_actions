#!/usr/bin/env python

import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _artifact_registry import infer_checkpoint_metadata, make_artifact_id, register_artifact

PROGRESS_LOG_EVERY_BATCHES = 100


@dataclass
class LatentExportConfig:
    policy: PreTrainedConfig | None = None
    dataset_repo_id: str | None = None
    dataset_root: str | None = None
    episodes: list[int] | None = None
    output_dir: Path | None = None
    output_repo_id: str | None = None
    # Use a top-level namespace such as `latent_labels` or `lam_lapa`.
    # Avoid observation.* because dataset delta-timestamp expansion treats those
    # keys like normal observations and adds extra temporal axes.
    feature_prefix: str = "latent_labels"
    batch_size: int = 32
    num_workers: int = 8
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
            self.output_repo_id = f"{self.dataset_repo_id}_latent_labeled"

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


def _format_feature_values(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 0:
        return values
    if len(shape) == 1 and shape[0] == 1:
        return values.reshape(-1, 1)

    formatted = np.empty((values.shape[0],), dtype=object)
    for idx in range(values.shape[0]):
        formatted[idx] = values[idx]
    return formatted


def _compute_float_feature_stats(
    *,
    label_arrays: dict[str, np.ndarray],
    valid_supervision: np.ndarray,
    feature_infos: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    valid_rows = valid_supervision.reshape(-1).astype(bool, copy=False)
    if not valid_rows.any():
        return {}

    float_feature_data = {}
    float_feature_infos = {}
    for name, info in feature_infos.items():
        dtype = np.dtype(info["dtype"])
        if not np.issubdtype(dtype, np.floating):
            continue
        values = label_arrays[name][valid_rows]
        if values.shape[0] == 0:
            continue
        float_feature_data[name] = values
        float_feature_infos[name] = info

    if not float_feature_data:
        return {}

    return compute_episode_stats(
        episode_data=float_feature_data,
        features=float_feature_infos,
    )


def export_latent_dataset(cfg: LatentExportConfig) -> None:
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
    policy = make_policy(cfg.policy, ds_meta=source_dataset.meta)
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

    total_frames = int(source_dataset.meta.total_frames)
    label_arrays = {
        name: np.full(
            (total_frames, *spec["shape"]),
            spec["invalid_fill_value"],
            dtype=spec["dtype"],
        )
        for name, spec in plan["representations"].items()
    }
    valid_supervision = np.zeros((total_frames, 1), dtype=np.int64)
    feature_infos = {
        name: {"dtype": spec["dtype"].name, "shape": spec["shape"], "names": None}
        for name, spec in plan["representations"].items()
    }

    logging.info(
        "Label export setup:\n%s",
        pformat(
            {
                "policy_type": cfg.policy.type,
                "policy_path": str(cfg.policy.pretrained_path),
                "dataset_repo_id": cfg.dataset_repo_id,
                "dataset_root": cfg.dataset_root,
                "episodes": cfg.episodes,
                "feature_prefix": cfg.feature_prefix,
                "representation_names": list(plan["representations"]),
                "batch_size": cfg.batch_size,
                "num_workers": cfg.num_workers,
                "delta_timestamps": plan["delta_timestamps"],
            }
        ),
    )

    written = 0
    policy.eval()
    start_time = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            row_idx = batch["index"]
            if not torch.is_tensor(row_idx):
                row_idx = torch.as_tensor(row_idx)
            row_idx_np = row_idx.detach().cpu().numpy().astype(np.int64, copy=False)

            batch_out = _normalize_export_batch(export_latent_labels(batch))
            valid_mask = batch_out["valid_mask"]
            if valid_mask.ndim != 1 or valid_mask.shape[0] != row_idx.shape[0]:
                raise ValueError(
                    "export_latent_labels() must return `valid_mask` with shape [batch_size]."
                )

            valid_mask_np = valid_mask.detach().cpu().numpy().astype(bool, copy=False)
            valid_count = int(valid_mask_np.sum())

            if valid_count:
                valid_rows = row_idx_np[valid_mask_np]
                if cfg.max_valid_samples is not None:
                    remaining = cfg.max_valid_samples - written
                    if remaining <= 0:
                        break
                    if valid_count > remaining:
                        valid_rows = valid_rows[:remaining]
                        valid_count = remaining

                for name, labels_tensor in batch_out["labels_by_name"].items():
                    if labels_tensor.shape[0] != int(valid_mask_np.sum()):
                        raise ValueError(
                            f"export_latent_labels() returned {name!r} with a batch size that does not match the number of valid rows."
                        )
                    if cfg.max_valid_samples is not None and labels_tensor.shape[0] > valid_count:
                        labels_tensor = labels_tensor[:valid_count]
                    label_arrays[name][valid_rows] = labels_tensor.detach().cpu().numpy().astype(
                        plan["representations"][name]["dtype"],
                        copy=False,
                    )
                valid_supervision[valid_rows, 0] = 1
                written += valid_count

            if batch_idx % PROGRESS_LOG_EVERY_BATCHES == 0 or batch_idx == len(dataloader):
                elapsed_s = max(time.perf_counter() - start_time, 1e-9)
                processed_rows = min(batch_idx * cfg.batch_size, len(label_dataset))
                rate = processed_rows / elapsed_s
                logging.info(
                    "Progress: rows=%d/%d valid=%d rate=%.1f rows/s",
                    processed_rows,
                    len(label_dataset),
                    written,
                    rate,
                )

            if cfg.max_valid_samples is not None and written >= cfg.max_valid_samples:
                break

    logging.info(
        "Finalizing labeled dataset: output_dir=%s valid_labels=%d feature_names=%s",
        output_dir,
        int(valid_supervision.sum()),
        [f"{cfg.feature_prefix}.{name}" for name in plan["representations"]],
    )
    relabeled_dataset = add_features(
        dataset=source_dataset,
        features={
            **{
                f"{cfg.feature_prefix}.{name}": (
                    _format_feature_values(label_arrays[name], feature_infos[name]["shape"]),
                    feature_infos[name],
                )
                for name in plan["representations"]
            },
            f"{cfg.feature_prefix}.valid": (
                valid_supervision,
                {"dtype": "int64", "shape": (1,), "names": None},
            ),
        },
        output_dir=output_dir,
        repo_id=cfg.output_repo_id,
    )

    latent_stats = _compute_float_feature_stats(
        label_arrays=label_arrays,
        valid_supervision=valid_supervision,
        feature_infos=feature_infos,
    )
    if latent_stats:
        merged_stats = dict(relabeled_dataset.meta.stats or {})
        merged_stats.update(
            {f"{cfg.feature_prefix}.{name}": stats for name, stats in latent_stats.items()}
        )
        write_stats(merged_stats, relabeled_dataset.root)

    checkpoint_meta = infer_checkpoint_metadata(cfg.policy.pretrained_path)
    label_manifest = {
        "policy_type": cfg.policy.type,
        "policy_path": str(cfg.policy.pretrained_path),
        "source_dataset_repo_id": cfg.dataset_repo_id,
        "source_dataset_root": cfg.dataset_root,
        "episodes": cfg.episodes,
        "output_repo_id": cfg.output_repo_id,
        "output_dir": str(output_dir),
        "feature_prefix": cfg.feature_prefix,
        "feature_names": {
            name: f"{cfg.feature_prefix}.{name}" for name in plan["representations"]
        },
        "valid_feature_name": f"{cfg.feature_prefix}.valid",
        "stats_feature_names": [
            f"{cfg.feature_prefix}.{name}"
            for name, info in feature_infos.items()
            if np.issubdtype(np.dtype(info["dtype"]), np.floating)
        ],
        "delta_timestamps": plan["delta_timestamps"],
        "num_valid_labels": int(valid_supervision.sum()),
    }
    label_manifest_path = output_dir / "label_manifest.json"
    label_manifest_path.write_text(json.dumps(label_manifest, indent=2) + "\n")

    export_manifest_path = output_dir / "export_manifest.json"
    export_manifest = {
        "artifact_type": "latent_export",
        "export_kind": "full_labeled_dataset",
        "suite_name": "latent_export",
        "suite_version": "v1",
        "artifact_id": make_artifact_id(
            suite_name="latent_export",
            suite_version="v1",
            checkpoint_id=checkpoint_meta["source_checkpoint_id"],
            output_label=output_dir.name,
        ),
        **checkpoint_meta,
        "analysis_only": False,
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
        "stats_feature_names": [
            f"{cfg.feature_prefix}.{name}"
            for name, info in feature_infos.items()
            if np.issubdtype(np.dtype(info["dtype"]), np.floating)
        ],
        "delta_timestamps": plan["delta_timestamps"],
        "num_rows": int(source_dataset.meta.total_frames),
        "num_valid_labels": int(valid_supervision.sum()),
        "label_manifest_path": str(label_manifest_path),
    }
    register_artifact(
        manifest_path=export_manifest_path,
        manifest=export_manifest,
        registry_candidates=[output_dir, cfg.policy.pretrained_path, cfg.dataset_root],
    )

    logging.info("Finished export to %s", relabeled_dataset.root)
    logging.info("Manifests written to %s and %s", label_manifest_path, export_manifest_path)


@parser.wrap()
def main(cfg: LatentExportConfig) -> None:
    register_third_party_plugins()
    init_logging()
    export_latent_dataset(cfg)


if __name__ == "__main__":
    main()
