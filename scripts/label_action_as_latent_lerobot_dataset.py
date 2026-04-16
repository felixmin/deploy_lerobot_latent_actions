#!/usr/bin/env python

"""
Create a full LeRobot dataset copy whose latent labels are derived from future actions.

This is a control-data generator for downstream `latent_smolvla` experiments where
the "latent" target is built directly from the real action sequence instead of from
an upstream latent-action teacher.

The generated feature is written to:

    <feature_prefix>.<representation_name>

By default:

    latent_labels.continuous_vector_latents

Supported aggregation modes
---------------------------
- flatten_vector:
    Keep the whole future action chunk but flatten it into one vector of shape
    [horizon_frames * action_dim]. Example for LIBERO with 10 future frames:
    [70]. This is the recommended full-chunk export mode for large datasets
    because it avoids nested per-row Array2D/object serialization during the
    full-dataset rewrite.

- flatten:
    Keep the whole future action chunk as shape [horizon_frames, action_dim].
    Example for LIBERO with 10 future frames: [10, 7].

- sum:
    Sum all actions over the horizon and store a single [action_dim] vector.

- mean:
    Mean over the horizon and store a single [action_dim] vector.

- last:
    Use the final action in the horizon, shape [action_dim].

- sum_motion_gripper_final:
    Sum all motion dimensions over the horizon, and use the final gripper action
    as the last dimension. For 7D LIBERO actions this yields one [7] vector:
    sum(a[:, :6]), final(a[-1, 6]).

- sum_motion_gripper_mean:
    Sum all motion dimensions over the horizon, and use the mean gripper action
    as the last dimension.

Examples
--------
Full 1-second future chunk as the auxiliary latent target:

    conda run -n rlfv python label_action_as_latent_lerobot_dataset.py \
      --dataset_repo_id=HuggingFaceVLA/libero \
      --dataset_root=/mnt/data/workspace/runs_root/cache/huggingface/lerobot \
      --output_dir=/mnt/data/workspace/runs_root/runs_lerobot/latent_labels/libero_action_chunk10_flat \
      --output_repo_id=local/libero_action_chunk10_flat \
      --aggregation_mode=flatten_vector \
      --horizon_frames=10 \
      --force=true

Single "long-horizon action" control over one second:

    conda run -n rlfv python label_action_as_latent_lerobot_dataset.py \
      --dataset_repo_id=HuggingFaceVLA/libero \
      --dataset_root=/mnt/data/workspace/runs_root/cache/huggingface/lerobot \
      --output_dir=/mnt/data/workspace/runs_root/runs_lerobot/latent_labels/libero_action_chunk10_sum_motion_final \
      --output_repo_id=local/libero_action_chunk10_sum_motion_final \
      --aggregation_mode=sum_motion_gripper_final \
      --horizon_frames=10 \
      --force=true

Alignment note
--------------
`start_offset_frames` controls where the action window starts relative to the
anchor row:

- `start_offset_frames=0`:
    use actions at `[t, t+1, ..., t+horizon_frames-1]`
    This matches the way SmolVLA action chunks are built.

- `start_offset_frames=1`:
    use strictly-future actions at `[t+1, ..., t+horizon_frames]`
    This matches the usual "future 10" convention in some analysis scripts.
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

SUPPORTED_AGGREGATION_MODES = {
    "flatten_vector",
    "flatten",
    "sum",
    "mean",
    "last",
    "sum_motion_gripper_final",
    "sum_motion_gripper_mean",
}
SEQUENCE_AGGREGATION_MODES = {"flatten_vector", "flatten"}
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
    start_offset_frames: int = 0
    horizon_frames: int = 10
    aggregation_mode: str = "sum_motion_gripper_final"
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
            suffix = f"action_as_latent_{self.aggregation_mode}_h{self.horizon_frames}"
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
        if self.start_offset_frames < 0:
            raise ValueError(
                f"start_offset_frames must be >= 0, got {self.start_offset_frames}."
            )
        if self.aggregation_mode not in SUPPORTED_AGGREGATION_MODES:
            raise ValueError(
                f"aggregation_mode must be one of {sorted(SUPPORTED_AGGREGATION_MODES)}, "
                f"got {self.aggregation_mode!r}."
            )
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
    if len(shape) == 1 and shape[0] == 1:
        return values.reshape(-1, 1)

    formatted = np.empty((values.shape[0],), dtype=object)
    for idx in range(values.shape[0]):
        formatted[idx] = values[idx]
    return formatted


def _infer_valid_shape(
    *,
    aggregation_mode: str,
    horizon_frames: int,
) -> tuple[int, ...]:
    if aggregation_mode in SEQUENCE_AGGREGATION_MODES:
        return (int(horizon_frames),)
    return (1,)


def _compute_float_feature_stats(
    *,
    label_array: np.ndarray,
    valid_supervision: np.ndarray,
) -> dict[str, np.ndarray]:
    if valid_supervision.ndim == 2 and valid_supervision.shape[1] > 1:
        valid_steps = valid_supervision.astype(bool, copy=False)
        if label_array.ndim == 2:
            if label_array.shape[1] % valid_steps.shape[1] != 0:
                raise ValueError(
                    "Flat latent labels must be divisible by sequence-valid length, "
                    f"got labels={tuple(label_array.shape)} valid={tuple(valid_steps.shape)}"
                )
            step_dim = label_array.shape[1] // valid_steps.shape[1]
            values = label_array.reshape(label_array.shape[0], valid_steps.shape[1], step_dim)
        elif label_array.ndim == 3:
            if label_array.shape[1] != valid_steps.shape[1]:
                raise ValueError(
                    "Structured latent labels must match sequence-valid length, "
                    f"got labels={tuple(label_array.shape)} valid={tuple(valid_steps.shape)}"
                )
            values = label_array
        else:
            raise ValueError(
                "Sequence-valid supervision only supports label arrays with rank 2 or 3, "
                f"got rank={label_array.ndim}"
            )

        seq_len = int(values.shape[1])
        step_dim = int(values.shape[2])
        stats = {
            "min": np.full((seq_len, step_dim), np.nan, dtype=values.dtype),
            "max": np.full((seq_len, step_dim), np.nan, dtype=values.dtype),
            "mean": np.full((seq_len, step_dim), np.nan, dtype=np.float64),
            "std": np.full((seq_len, step_dim), np.nan, dtype=np.float64),
            "count": np.zeros((seq_len,), dtype=np.int64),
        }
        quantiles = {
            "q01": 0.01,
            "q10": 0.10,
            "q50": 0.50,
            "q90": 0.90,
            "q99": 0.99,
        }
        for key in quantiles:
            stats[key] = np.full((seq_len, step_dim), np.nan, dtype=np.float64)

        for step_idx in range(seq_len):
            step_keep = valid_steps[:, step_idx]
            if not step_keep.any():
                continue
            step_values = values[step_keep, step_idx, :]
            step_values64 = step_values.astype(np.float64, copy=False)
            stats["min"][step_idx] = step_values.min(axis=0)
            stats["max"][step_idx] = step_values.max(axis=0)
            stats["mean"][step_idx] = step_values64.mean(axis=0)
            stats["std"][step_idx] = step_values64.std(axis=0, ddof=0)
            stats["count"][step_idx] = int(step_values.shape[0])
            for key, q in quantiles.items():
                stats[key][step_idx] = np.quantile(step_values64, q, axis=0)
        return stats

    valid_rows = valid_supervision.reshape(-1).astype(bool, copy=False)
    if not valid_rows.any():
        return {}

    values = label_array[valid_rows]
    values64 = values.astype(np.float64, copy=False)
    stats = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values64.mean(axis=0),
        "std": values64.std(axis=0, ddof=0),
        "count": np.array([values.shape[0]], dtype=np.int64),
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
    start_offset_frames: int,
    horizon_frames: int,
    fps: float,
) -> dict[str, list[float]]:
    start = int(start_offset_frames)
    stop = start + int(horizon_frames)
    return {"action": [frame_idx / float(fps) for frame_idx in range(start, stop)]}


def _infer_output_shape(
    *,
    aggregation_mode: str,
    horizon_frames: int,
    action_dim: int,
) -> tuple[int, ...]:
    if aggregation_mode == "flatten_vector":
        return (int(horizon_frames) * int(action_dim),)
    if aggregation_mode == "flatten":
        return (int(horizon_frames), int(action_dim))
    return (int(action_dim),)


def _aggregate_actions(actions: torch.Tensor, aggregation_mode: str) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError(f"Expected actions with shape [B, H, A], got {tuple(actions.shape)}")

    if aggregation_mode == "flatten_vector":
        return actions.reshape(actions.shape[0], -1)
    if aggregation_mode == "flatten":
        return actions
    if aggregation_mode == "sum":
        return actions.sum(dim=1)
    if aggregation_mode == "mean":
        return actions.mean(dim=1)
    if aggregation_mode == "last":
        return actions[:, -1, :]

    if int(actions.shape[-1]) < 2:
        raise ValueError(
            f"Aggregation mode {aggregation_mode!r} requires at least 2 action dimensions, got {actions.shape[-1]}"
        )

    motion = actions[:, :, :-1].sum(dim=1)
    if aggregation_mode == "sum_motion_gripper_final":
        gripper = actions[:, -1, -1].unsqueeze(-1)
        return torch.cat([motion, gripper], dim=-1)
    if aggregation_mode == "sum_motion_gripper_mean":
        gripper = actions[:, :, -1].mean(dim=1, keepdim=True)
        return torch.cat([motion, gripper], dim=-1)

    raise ValueError(f"Unsupported aggregation_mode: {aggregation_mode!r}")


def _load_tabular_action_columns(dataset: LeRobotDataset, action_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hf_dataset = dataset.hf_dataset
    actions = np.stack(hf_dataset[action_key]).astype(np.float32, copy=False)
    row_indices = np.arange(actions.shape[0], dtype=np.int64)
    episode_indices = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    return row_indices, episode_indices, actions


def _iter_episode_segments(episode_indices: np.ndarray):
    if episode_indices.ndim != 1:
        raise ValueError(f"Expected episode_indices rank 1, got {episode_indices.ndim}")
    if episode_indices.size == 0:
        return
    boundaries = np.flatnonzero(np.diff(episode_indices) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [episode_indices.shape[0]]))
    for start, end in zip(starts, ends, strict=True):
        yield int(episode_indices[start]), int(start), int(end)


def _compute_valid_count(
    *,
    episode_length: int,
    start_offset_frames: int,
    horizon_frames: int,
) -> int:
    required_span = int(start_offset_frames) + int(horizon_frames)
    return int(episode_length) - required_span + 1


def _build_episode_batch_valid_mask(
    *,
    batch_start: int,
    batch_end: int,
    valid_count: int,
) -> np.ndarray:
    anchor_indices = np.arange(int(batch_start), int(batch_end), dtype=np.int64)
    return anchor_indices < int(valid_count)


def _build_action_windows(
    *,
    episode_actions: np.ndarray,
    anchor_indices: np.ndarray,
    start_offset_frames: int,
    horizon_frames: int,
) -> np.ndarray:
    return np.stack(
        [
            episode_actions[
                int(anchor) + int(start_offset_frames) : int(anchor) + int(start_offset_frames) + int(horizon_frames)
            ]
            for anchor in anchor_indices
        ],
        axis=0,
    )


def _build_padded_action_windows(
    *,
    episode_actions: np.ndarray,
    anchor_indices: np.ndarray,
    start_offset_frames: int,
    horizon_frames: int,
    invalid_fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = int(anchor_indices.shape[0])
    action_dim = int(episode_actions.shape[1])
    windows = np.full(
        (batch_size, int(horizon_frames), action_dim),
        invalid_fill_value,
        dtype=episode_actions.dtype,
    )
    valid_steps = np.zeros((batch_size, int(horizon_frames)), dtype=np.int64)
    episode_length = int(episode_actions.shape[0])

    for batch_idx, anchor in enumerate(anchor_indices.tolist()):
        start = int(anchor) + int(start_offset_frames)
        if start >= episode_length:
            continue
        stop = min(start + int(horizon_frames), episode_length)
        step_count = max(stop - start, 0)
        if step_count <= 0:
            continue
        windows[batch_idx, :step_count] = episode_actions[start:stop]
        valid_steps[batch_idx, :step_count] = 1
    return windows, valid_steps


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

    source_dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root, episodes=cfg.episodes)
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
        start_offset_frames=int(cfg.start_offset_frames),
        horizon_frames=int(cfg.horizon_frames),
        fps=float(source_dataset.meta.fps),
    )
    output_shape = _infer_output_shape(
        aggregation_mode=cfg.aggregation_mode,
        horizon_frames=int(cfg.horizon_frames),
        action_dim=action_dim,
    )
    valid_shape = _infer_valid_shape(
        aggregation_mode=cfg.aggregation_mode,
        horizon_frames=int(cfg.horizon_frames),
    )

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
                "start_offset_frames": cfg.start_offset_frames,
                "horizon_frames": cfg.horizon_frames,
                "aggregation_mode": cfg.aggregation_mode,
                "output_shape": output_shape,
                "valid_shape": valid_shape,
                "delta_timestamps": delta_timestamps,
                "implementation": "tabular_action_columns",
            }
        ),
    )

    written = 0
    processed_rows = 0
    processed_batches = 0
    start_time = time.perf_counter()
    row_indices, episode_indices, actions = _load_tabular_action_columns(source_dataset, cfg.action_key)
    stop_export = False

    for episode_id, start, end in _iter_episode_segments(episode_indices):
        episode_actions = actions[start:end]
        episode_rows = row_indices[start:end]
        episode_length = int(episode_actions.shape[0])
        valid_count = _compute_valid_count(
            episode_length=episode_length,
            start_offset_frames=int(cfg.start_offset_frames),
            horizon_frames=int(cfg.horizon_frames),
        )
        if cfg.aggregation_mode in SEQUENCE_AGGREGATION_MODES:
            if episode_length <= int(cfg.start_offset_frames):
                continue
        elif valid_count <= 0:
            continue

        for batch_start in range(0, episode_length, int(cfg.batch_size)):
            batch_end = min(batch_start + int(cfg.batch_size), episode_length)
            batch_rows = episode_rows[batch_start:batch_end]
            if cfg.aggregation_mode in SEQUENCE_AGGREGATION_MODES:
                local_anchor_indices = np.arange(batch_start, batch_end, dtype=np.int64)
                windows, step_validity = _build_padded_action_windows(
                    episode_actions=episode_actions,
                    anchor_indices=local_anchor_indices,
                    start_offset_frames=int(cfg.start_offset_frames),
                    horizon_frames=int(cfg.horizon_frames),
                    invalid_fill_value=float(cfg.invalid_fill_value),
                )
                row_valid_mask = step_validity.any(axis=1).astype(bool, copy=False)
                row_valid_count = int(row_valid_mask.sum())

                if row_valid_count > 0:
                    valid_positions = np.flatnonzero(row_valid_mask)
                    if cfg.max_valid_samples is not None:
                        remaining = cfg.max_valid_samples - written
                        if remaining <= 0:
                            stop_export = True
                            break
                        if row_valid_count > remaining:
                            valid_positions = valid_positions[:remaining]
                            row_valid_count = int(valid_positions.shape[0])

                    valid_rows = batch_rows[valid_positions]
                    aggregated = _aggregate_actions(
                        torch.from_numpy(windows[valid_positions]),
                        cfg.aggregation_mode,
                    ).detach().cpu().numpy().astype(np.dtype(cfg.label_dtype), copy=False)

                    label_array[valid_rows] = aggregated
                    valid_supervision[valid_rows] = step_validity[valid_positions]
                    written += row_valid_count
            else:
                batch_valid_mask = _build_episode_batch_valid_mask(
                    batch_start=batch_start,
                    batch_end=batch_end,
                    valid_count=valid_count,
                )
                batch_valid_count = int(batch_valid_mask.sum())

                if batch_valid_count > 0:
                    valid_positions = np.flatnonzero(batch_valid_mask)
                    if cfg.max_valid_samples is not None:
                        remaining = cfg.max_valid_samples - written
                        if remaining <= 0:
                            stop_export = True
                            break
                        if batch_valid_count > remaining:
                            valid_positions = valid_positions[:remaining]
                            batch_valid_count = int(valid_positions.shape[0])

                    valid_rows = batch_rows[valid_positions]
                    local_anchor_indices = np.arange(batch_start, batch_end, dtype=np.int64)[valid_positions]
                    windows = _build_action_windows(
                        episode_actions=episode_actions,
                        anchor_indices=local_anchor_indices,
                        start_offset_frames=int(cfg.start_offset_frames),
                        horizon_frames=int(cfg.horizon_frames),
                    )
                    aggregated = _aggregate_actions(
                        torch.from_numpy(windows),
                        cfg.aggregation_mode,
                    ).detach().cpu().numpy().astype(np.dtype(cfg.label_dtype), copy=False)

                    label_array[valid_rows] = aggregated
                    valid_supervision[valid_rows, 0] = 1
                    written += batch_valid_count

            processed_rows += int(batch_rows.shape[0])
            processed_batches += 1

            if processed_batches % PROGRESS_LOG_EVERY_BATCHES == 0 or processed_rows == total_frames:
                elapsed_s = max(time.perf_counter() - start_time, 1e-9)
                rate = processed_rows / elapsed_s
                logging.info(
                    "Progress: rows=%d/%d valid=%d rate=%.1f rows/s",
                    processed_rows,
                    total_frames,
                    written,
                    rate,
                )

            if cfg.max_valid_samples is not None and written >= cfg.max_valid_samples:
                stop_export = True
                break

        if stop_export:
            break

    logging.info(
        "Finalizing labeled dataset: output_dir=%s valid_labels=%d feature_name=%s.%s",
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
        "label_source_type": "dataset_action_aggregation",
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
        "start_offset_frames": int(cfg.start_offset_frames),
        "horizon_frames": int(cfg.horizon_frames),
        "aggregation_mode": cfg.aggregation_mode,
        "delta_timestamps": delta_timestamps,
        "valid_shape": valid_shape,
        "num_valid_rows": int(valid_supervision.reshape(total_frames, -1).any(axis=1).sum()),
        "num_valid_supervision_tokens": int(valid_supervision.sum()),
        "num_valid_labels": int(valid_supervision.reshape(total_frames, -1).any(axis=1).sum()),
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
        "start_offset_frames": int(cfg.start_offset_frames),
        "horizon_frames": int(cfg.horizon_frames),
        "aggregation_mode": cfg.aggregation_mode,
        "delta_timestamps": delta_timestamps,
        "num_rows": int(source_dataset.meta.total_frames),
        "valid_shape": valid_shape,
        "num_valid_rows": int(valid_supervision.reshape(total_frames, -1).any(axis=1).sum()),
        "num_valid_supervision_tokens": int(valid_supervision.sum()),
        "num_valid_labels": int(valid_supervision.reshape(total_frames, -1).any(axis=1).sum()),
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
