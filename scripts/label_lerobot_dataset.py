#!/usr/bin/env python

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


DEFAULT_LATENT_REPRESENTATION = "codebook_id_latents"
PROGRESS_LOG_EVERY_BATCHES = 100


@dataclass
class LatentExportConfig:
    policy: PreTrainedConfig | None = None
    dataset_repo_id: str | None = None
    dataset_root: str | None = None
    episodes: list[int] | None = None
    output_dir: Path | None = None
    output_repo_id: str | None = None
    latent_representation: str = DEFAULT_LATENT_REPRESENTATION
    feature_name: str = "latent_labels"
    valid_feature_name: str = "latent_supervised"
    batch_size: int = 32
    num_workers: int = 8
    force: bool = False
    max_valid_samples: int | None = None

    def validate(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("Policy is not configured. Please specify a checkpoint with `--policy.path`.")

        cli_overrides = parser.get_cli_overrides("policy")
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
        if self.feature_name == self.valid_feature_name:
            raise ValueError("feature_name and valid_feature_name must differ.")
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
    required_keys = {"delta_timestamps", "shape", "dtype", "invalid_fill_value"}
    missing = required_keys.difference(plan)
    if missing:
        raise KeyError(f"prepare_latent_export() is missing keys: {sorted(missing)}")

    shape = tuple(int(dim) for dim in plan["shape"])
    dtype = np.dtype(plan["dtype"])
    return {
        "delta_timestamps": plan["delta_timestamps"],
        "shape": shape,
        "dtype": dtype,
        "invalid_fill_value": plan["invalid_fill_value"],
    }


def _normalize_export_batch(batch_out: Any) -> dict[str, torch.Tensor]:
    if not isinstance(batch_out, dict):
        raise TypeError(f"export_latent_labels() must return a dict, got {type(batch_out)}.")
    if "labels" not in batch_out or "valid_mask" not in batch_out:
        raise KeyError("export_latent_labels() must return `labels` and `valid_mask`.")

    labels = batch_out["labels"]
    valid_mask = batch_out["valid_mask"]
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    if not torch.is_tensor(valid_mask):
        valid_mask = torch.as_tensor(valid_mask)
    return {"labels": labels, "valid_mask": valid_mask}


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

    register_third_party_plugins()

    source_dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root)
    policy = make_policy(cfg.policy, ds_meta=source_dataset.meta)
    prepare_latent_export = _get_required_method(policy, "prepare_latent_export")
    export_latent_labels = _get_required_method(policy, "export_latent_labels")

    plan = _normalize_export_plan(
        prepare_latent_export(source_dataset.meta, representation=cfg.latent_representation)
    )

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
    labels = np.full(
        (total_frames, *plan["shape"]),
        plan["invalid_fill_value"],
        dtype=plan["dtype"],
    )
    valid_supervision = np.zeros((total_frames, 1), dtype=np.int64)
    feature_info = {"dtype": plan["dtype"].name, "shape": plan["shape"], "names": None}

    logging.info(
        "Label export setup:\n%s",
        pformat(
            {
                "policy_type": cfg.policy.type,
                "policy_path": str(cfg.policy.pretrained_path),
                "dataset_repo_id": cfg.dataset_repo_id,
                "dataset_root": cfg.dataset_root,
                "episodes": cfg.episodes,
                "latent_representation": cfg.latent_representation,
                "feature_name": cfg.feature_name,
                "valid_feature_name": cfg.valid_feature_name,
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

            batch_out = _normalize_export_batch(
                export_latent_labels(batch, representation=cfg.latent_representation)
            )
            valid_mask = batch_out["valid_mask"]
            if valid_mask.ndim != 1 or valid_mask.shape[0] != row_idx.shape[0]:
                raise ValueError(
                    "export_latent_labels() must return `valid_mask` with shape [batch_size]."
                )

            valid_mask_np = valid_mask.detach().cpu().numpy().astype(bool, copy=False)
            valid_count = int(valid_mask_np.sum())

            if valid_count:
                labels_tensor = batch_out["labels"]
                if labels_tensor.shape[0] != valid_count:
                    raise ValueError(
                        "export_latent_labels() returned a labels batch that does not match the number of valid rows."
                    )

                valid_rows = row_idx_np[valid_mask_np]
                if cfg.max_valid_samples is not None:
                    remaining = cfg.max_valid_samples - written
                    if remaining <= 0:
                        break
                    if valid_count > remaining:
                        valid_rows = valid_rows[:remaining]
                        labels_tensor = labels_tensor[:remaining]
                        valid_count = remaining

                labels[valid_rows] = labels_tensor.detach().cpu().numpy().astype(plan["dtype"], copy=False)
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

    relabeled_dataset = add_features(
        dataset=source_dataset,
        features={
            cfg.feature_name: (labels, feature_info),
            cfg.valid_feature_name: (
                valid_supervision,
                {"dtype": "int64", "shape": (1,), "names": None},
            ),
        },
        output_dir=output_dir,
        repo_id=cfg.output_repo_id,
    )

    manifest = {
        "policy_type": cfg.policy.type,
        "policy_path": str(cfg.policy.pretrained_path),
        "source_dataset_repo_id": cfg.dataset_repo_id,
        "source_dataset_root": cfg.dataset_root,
        "episodes": cfg.episodes,
        "output_repo_id": cfg.output_repo_id,
        "output_dir": str(output_dir),
        "latent_representation": cfg.latent_representation,
        "feature_name": cfg.feature_name,
        "valid_feature_name": cfg.valid_feature_name,
        "delta_timestamps": plan["delta_timestamps"],
        "num_valid_labels": int(valid_supervision.sum()),
    }
    manifest_path = output_dir / "label_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logging.info("Finished export to %s", relabeled_dataset.root)
    logging.info("Manifest written to %s", manifest_path)


@parser.wrap()
def main(cfg: LatentExportConfig) -> None:
    init_logging()
    export_latent_dataset(cfg)


if __name__ == "__main__":
    main()
