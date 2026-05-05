#!/usr/bin/env python

"""Create a LeRobot dataset copy whose latent labels are summed future action windows.

This exporter writes one latent target per dataset row:

    <feature_prefix>.<representation_name> : [action_dim]
    <feature_prefix>.valid                 : scalar

`horizon_frames` controls the additive action window attached to each anchor row:

- latent at anchor `t` is `sum(action[t:t+horizon_frames])`
- if `t + horizon_frames` exceeds the episode length, the row is marked invalid

The resulting dataset can then be queried dynamically with delta timestamps to
assemble sparse latent plans such as `[0, 10, 20, 30, 40]` without relabeling.
"""

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
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _artifact_registry import infer_checkpoint_metadata, make_artifact_id, register_artifact

PROGRESS_LOG_EVERY_BATCHES = 100


@dataclass
class ActionAsLatentConfig:
    dataset_repo_id: str | None = None
    dataset_root: str | None = None
    episodes: list[int] | None = None
    output_dir: Path | None = None
    output_repo_id: str | None = None
    feature_prefix: str = "latent_labels"
    representation_name: str = "continuous_vector_latents"
    action_key: str = "action"
    horizon_frames: int = 10
    label_dtype: str = "float32"
    invalid_fill_value: float = float("nan")
    batch_size: int = 4096
    num_workers: int = 8
    force: bool = False
    max_valid_samples: int | None = None

    def validate(self) -> None:
        if not self.dataset_repo_id:
            raise ValueError("Please specify `--dataset_repo_id`.")
        if self.output_dir is None:
            raise ValueError("Please specify `--output_dir`.")
        if self.output_repo_id is None:
            suffix = f"action_as_latent_sum_h{self.horizon_frames}"
            self.output_repo_id = f"{self.dataset_repo_id}_{suffix}"
        if self.feature_prefix.startswith("observation."):
            raise ValueError(
                "feature_prefix must not use observation.*. "
                "Use a top-level namespace such as `latent_labels`."
            )
        if not self.representation_name:
            raise ValueError("representation_name must be non-empty.")
        if self.horizon_frames < 1:
            raise ValueError(f"horizon_frames must be >= 1, got {self.horizon_frames}.")
        np.dtype(self.label_dtype)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}.")
        if self.max_valid_samples is not None and self.max_valid_samples < 1:
            raise ValueError(f"max_valid_samples must be >= 1, got {self.max_valid_samples}.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return []


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        return
    if not force:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --force to overwrite it.")
    shutil.rmtree(output_dir)


def _format_feature_values(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 0:
        return values
    if len(shape) == 1:
        return values

    formatted = np.empty((values.shape[0],), dtype=object)
    for idx in range(values.shape[0]):
        formatted[idx] = values[idx]
    return formatted


def _infer_valid_shape() -> tuple[int, ...]:
    return (1,)


def _compute_float_feature_stats(
    *,
    label_array: np.ndarray,
    valid_supervision: np.ndarray,
) -> dict[str, np.ndarray]:
    if valid_supervision.ndim == 2 and int(valid_supervision.shape[1]) == 1:
        valid_supervision = valid_supervision.reshape(valid_supervision.shape[0])
    if label_array.ndim != 2:
        raise ValueError(
            "Expected latent labels with shape [N, A], "
            f"got {tuple(label_array.shape)}"
        )
    if valid_supervision.ndim != 1:
        raise ValueError(
            "Expected latent validity with shape [N], "
            f"got {tuple(valid_supervision.shape)}"
        )
    if int(label_array.shape[0]) != int(valid_supervision.shape[0]):
        raise ValueError(
            "Latent labels and validity mask must match on [N], "
            f"got labels={tuple(label_array.shape)} valid={tuple(valid_supervision.shape)}"
        )

    valid_rows = valid_supervision.astype(bool, copy=False)
    if not valid_rows.any():
        return {}

    values = label_array[valid_rows]
    values64 = values.astype(np.float64, copy=False)
    stats = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values64.mean(axis=0),
        "std": values64.std(axis=0, ddof=0),
        "count": np.array([int(values.shape[0])], dtype=np.int64),
    }
    for key, q in {
        "q01": 0.01,
        "q10": 0.10,
        "q50": 0.50,
        "q90": 0.90,
        "q99": 0.99,
    }.items():
        stats[key] = np.quantile(values64, q, axis=0)
    return stats


def _action_delta_timestamps(
    *,
    horizon_frames: int,
    fps: float,
) -> dict[str, list[float]]:
    stop = int(horizon_frames)
    return {"action": [frame_idx / float(fps) for frame_idx in range(stop)]}


def _infer_output_shape(
    *,
    action_dim: int,
) -> tuple[int, ...]:
    return (int(action_dim),)


def _reduce_action_window_batch(
    *,
    action_windows: torch.Tensor | np.ndarray,
    action_is_pad: torch.Tensor | np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(action_windows):
        action_windows = torch.as_tensor(action_windows)
    action_windows = action_windows.to(dtype=torch.float32)

    if action_windows.ndim != 3:
        raise ValueError(
            "Expected action windows with shape [B, H, A], "
            f"got {tuple(action_windows.shape)}"
        )

    batch_size = int(action_windows.shape[0])
    if action_is_pad is None:
        valid_rows = torch.ones(batch_size, dtype=torch.bool, device=action_windows.device)
    else:
        if not torch.is_tensor(action_is_pad):
            action_is_pad = torch.as_tensor(action_is_pad)
        action_is_pad = action_is_pad.to(device=action_windows.device, dtype=torch.bool)
        if action_is_pad.ndim != 2 or tuple(action_is_pad.shape[:2]) != tuple(action_windows.shape[:2]):
            raise ValueError(
                "Expected action_is_pad with shape [B, H] matching action windows, "
                f"got actions={tuple(action_windows.shape)} pads={tuple(action_is_pad.shape)}"
            )
        valid_rows = ~action_is_pad.any(dim=1)

    targets = action_windows.sum(dim=1)
    return targets, valid_rows


def export_action_as_latent_dataset(cfg: ActionAsLatentConfig) -> None:
    cfg.validate()
    if cfg.output_dir is None:
        raise ValueError("output_dir was not configured.")
    if cfg.output_repo_id is None:
        raise ValueError("output_repo_id was not configured.")
    if cfg.dataset_repo_id is None:
        raise ValueError("dataset_repo_id was not configured.")

    output_dir = cfg.output_dir.resolve()
    _prepare_output_dir(output_dir, cfg.force)

    source_dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root)
    action_feature = source_dataset.meta.features.get(cfg.action_key)
    if action_feature is None:
        raise KeyError(f"Dataset feature {cfg.action_key!r} is missing from source dataset metadata.")
    action_shape = tuple(int(dim) for dim in action_feature["shape"])
    if len(action_shape) != 1:
        raise ValueError(
            f"Expected scalar action vectors with shape [A] for {cfg.action_key!r}, got {action_shape}"
        )
    action_dim = int(action_shape[0])

    delta_timestamps = _action_delta_timestamps(
        horizon_frames=int(cfg.horizon_frames),
        fps=float(source_dataset.meta.fps),
    )
    label_dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
        episodes=cfg.episodes,
        delta_timestamps=delta_timestamps,
    )
    dataloader = torch.utils.data.DataLoader(
        label_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    output_shape = _infer_output_shape(
        action_dim=action_dim,
    )
    valid_shape = _infer_valid_shape()

    total_frames = int(source_dataset.meta.total_frames)
    label_array = np.full(
        (total_frames, *output_shape),
        cfg.invalid_fill_value,
        dtype=np.dtype(cfg.label_dtype),
    )
    valid_supervision = np.zeros((total_frames, *valid_shape), dtype=np.int64)
    feature_info = {"dtype": np.dtype(cfg.label_dtype).name, "shape": output_shape, "names": None}

    logging.info(
        "Action-as-latent export setup:\n%s",
        pformat(
            {
                "dataset_repo_id": cfg.dataset_repo_id,
                "dataset_root": cfg.dataset_root,
                "episodes": cfg.episodes,
                "feature_prefix": cfg.feature_prefix,
                "representation_name": cfg.representation_name,
                "action_key": cfg.action_key,
                "horizon_frames": cfg.horizon_frames,
                "batch_size": cfg.batch_size,
                "num_workers": cfg.num_workers,
                "output_shape": output_shape,
                "valid_shape": valid_shape,
                "delta_timestamps": delta_timestamps,
                "implementation": "streaming_action_window_sum",
            }
        ),
    )

    written = 0
    start_time = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            row_idx = batch["index"]
            if not torch.is_tensor(row_idx):
                row_idx = torch.as_tensor(row_idx)
            row_idx_np = row_idx.detach().cpu().numpy().astype(np.int64, copy=False)

            pad_key = f"{cfg.action_key}_is_pad"
            action_windows = batch[cfg.action_key]
            action_is_pad = batch.get(pad_key)
            targets, row_validity = _reduce_action_window_batch(
                action_windows=action_windows,
                action_is_pad=action_is_pad,
            )
            row_valid_mask = row_validity.detach().cpu().numpy().astype(bool, copy=False)
            row_valid_count = int(row_valid_mask.sum())

            if row_valid_count > 0:
                valid_rows = row_idx_np[row_valid_mask]
                if cfg.max_valid_samples is not None:
                    remaining = cfg.max_valid_samples - written
                    if remaining <= 0:
                        break
                    if row_valid_count > remaining:
                        valid_rows = valid_rows[:remaining]
                        row_valid_count = remaining

                valid_targets = targets[row_validity]
                if cfg.max_valid_samples is not None and valid_targets.shape[0] > row_valid_count:
                    valid_targets = valid_targets[:row_valid_count]

                label_array[valid_rows] = valid_targets.detach().cpu().numpy().astype(
                    np.dtype(cfg.label_dtype),
                    copy=False,
                )
                valid_supervision[valid_rows, 0] = 1
                written += row_valid_count

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
        "Finalizing labeled dataset: output_dir=%s valid_rows=%d feature_name=%s.%s",
        output_dir,
        int(valid_supervision.sum()),
        cfg.feature_prefix,
        cfg.representation_name,
    )
    relabeled_dataset = add_features(
        dataset=source_dataset,
        features={
            f"{cfg.feature_prefix}.{cfg.representation_name}": (
                _format_feature_values(label_array, output_shape),
                feature_info,
            ),
            f"{cfg.feature_prefix}.valid": (
                _format_feature_values(valid_supervision, valid_shape),
                {"dtype": "int64", "shape": valid_shape, "names": None},
            ),
        },
        output_dir=output_dir,
        repo_id=cfg.output_repo_id,
    )

    float_stats = _compute_float_feature_stats(
        label_array=label_array,
        valid_supervision=valid_supervision,
    )
    if float_stats:
        merged_stats = dict(relabeled_dataset.meta.stats or {})
        merged_stats[f"{cfg.feature_prefix}.{cfg.representation_name}"] = float_stats
        write_stats(merged_stats, relabeled_dataset.root)

    checkpoint_meta = infer_checkpoint_metadata(None)
    label_manifest = {
        "label_source_type": "dataset_action_vector",
        "source_dataset_repo_id": cfg.dataset_repo_id,
        "source_dataset_root": cfg.dataset_root,
        "episodes": cfg.episodes,
        "output_repo_id": cfg.output_repo_id,
        "output_dir": str(output_dir),
        "feature_prefix": cfg.feature_prefix,
        "feature_names": {
            cfg.representation_name: f"{cfg.feature_prefix}.{cfg.representation_name}",
        },
        "valid_feature_name": f"{cfg.feature_prefix}.valid",
        "stats_feature_names": [f"{cfg.feature_prefix}.{cfg.representation_name}"],
        "action_key": cfg.action_key,
        "horizon_frames": int(cfg.horizon_frames),
        "label_layout": "vector_per_frame",
        "delta_timestamps": delta_timestamps,
        "valid_shape": valid_shape,
        "num_valid_rows": int(valid_supervision.sum()),
        "num_valid_supervision_tokens": int(valid_supervision.sum()),
        "num_valid_labels": int(valid_supervision.sum()),
    }
    label_manifest_path = output_dir / "label_manifest.json"
    label_manifest_path.write_text(json.dumps(label_manifest, indent=2) + "\n")

    export_manifest_path = output_dir / "export_manifest.json"
    export_manifest = {
        "artifact_type": "latent_export",
        "export_kind": "full_labeled_dataset",
        "suite_name": "action_as_latent_export",
        "suite_version": "v1",
        "artifact_id": make_artifact_id(
            suite_name="action_as_latent_export",
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
            cfg.representation_name: f"{cfg.feature_prefix}.{cfg.representation_name}",
        },
        "valid_feature_name": f"{cfg.feature_prefix}.valid",
        "stats_feature_names": [f"{cfg.feature_prefix}.{cfg.representation_name}"],
        "action_key": cfg.action_key,
        "horizon_frames": int(cfg.horizon_frames),
        "label_layout": "vector_per_frame",
        "delta_timestamps": delta_timestamps,
        "num_rows": int(source_dataset.meta.total_frames),
        "valid_shape": valid_shape,
        "num_valid_rows": int(valid_supervision.sum()),
        "num_valid_supervision_tokens": int(valid_supervision.sum()),
        "num_valid_labels": int(valid_supervision.sum()),
        "label_manifest_path": str(label_manifest_path),
    }
    register_artifact(
        manifest_path=export_manifest_path,
        manifest=export_manifest,
        registry_candidates=[output_dir, cfg.dataset_root],
    )

    logging.info("Finished export to %s", relabeled_dataset.root)
    logging.info("Manifests written to %s and %s", label_manifest_path, export_manifest_path)


@parser.wrap()
def main(cfg: ActionAsLatentConfig) -> None:
    register_third_party_plugins()
    init_logging()
    export_action_as_latent_dataset(cfg)


if __name__ == "__main__":
    main()
