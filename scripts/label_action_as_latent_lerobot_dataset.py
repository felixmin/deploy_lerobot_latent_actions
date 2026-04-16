#!/usr/bin/env python

"""Create a LeRobot dataset copy whose latent labels are future action sequences.

This exporter writes one structured latent target per anchor row:

    <feature_prefix>.<representation_name> : [horizon_frames, action_dim]
    <feature_prefix>.valid                 : [horizon_frames]

The latent target keeps the full future action sequence. We do not collapse it to
`mean`, `last`, `sum`, or any other aggregate. Normalization stats are pooled over
all valid `(row, timestep)` pairs, so the saved stats have shape `[action_dim]` and
broadcast across the latent horizon exactly like standard action normalization.

`start_offset_frames` controls where the action window starts relative to the anchor row:

- `start_offset_frames=0`: use actions at `[t, t+1, ..., t+horizon_frames-1]`
- `start_offset_frames=1`: use actions at `[t+1, ..., t+horizon_frames]`
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
    start_offset_frames: int = 0
    horizon_frames: int = 10
    label_dtype: str = "float32"
    invalid_fill_value: float = float("nan")
    batch_size: int = 4096
    force: bool = False
    max_valid_samples: int | None = None

    def validate(self) -> None:
        if not self.dataset_repo_id:
            raise ValueError("Please specify `--dataset_repo_id`.")
        if self.output_dir is None:
            raise ValueError("Please specify `--output_dir`.")
        if self.output_repo_id is None:
            suffix = f"action_as_latent_sequence_h{self.horizon_frames}"
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
        np.dtype(self.label_dtype)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
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


def _infer_valid_shape(*, horizon_frames: int) -> tuple[int, ...]:
    return (int(horizon_frames),)


def _compute_float_feature_stats(
    *,
    label_array: np.ndarray,
    valid_supervision: np.ndarray,
) -> dict[str, np.ndarray]:
    if label_array.ndim != 3:
        raise ValueError(
            "Expected structured latent labels with shape [N, H, A], "
            f"got {tuple(label_array.shape)}"
        )
    if valid_supervision.ndim != 2:
        raise ValueError(
            "Expected structured latent validity with shape [N, H], "
            f"got {tuple(valid_supervision.shape)}"
        )
    if label_array.shape[:2] != valid_supervision.shape:
        raise ValueError(
            "Structured latent labels and validity mask must match on [N, H], "
            f"got labels={tuple(label_array.shape)} valid={tuple(valid_supervision.shape)}"
        )

    valid_steps = valid_supervision.astype(bool, copy=False)
    if not valid_steps.any():
        return {}

    values = label_array[valid_steps]
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
    start_offset_frames: int,
    horizon_frames: int,
    fps: float,
) -> dict[str, list[float]]:
    start = int(start_offset_frames)
    stop = start + int(horizon_frames)
    return {"action": [frame_idx / float(fps) for frame_idx in range(start, stop)]}


def _infer_output_shape(
    *,
    horizon_frames: int,
    action_dim: int,
) -> tuple[int, ...]:
    return (int(horizon_frames), int(action_dim))


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
        horizon_frames=int(cfg.horizon_frames),
        action_dim=action_dim,
    )
    valid_shape = _infer_valid_shape(
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
                "output_shape": output_shape,
                "valid_shape": valid_shape,
                "delta_timestamps": delta_timestamps,
                "implementation": "tabular_action_sequence",
            }
        ),
    )

    written = 0
    processed_rows = 0
    processed_batches = 0
    start_time = time.perf_counter()
    row_indices, episode_indices, actions = _load_tabular_action_columns(source_dataset, cfg.action_key)
    stop_export = False

    for _episode_id, start, end in _iter_episode_segments(episode_indices):
        episode_actions = actions[start:end]
        episode_rows = row_indices[start:end]
        episode_length = int(episode_actions.shape[0])
        if episode_length <= int(cfg.start_offset_frames):
            continue

        for batch_start in range(0, episode_length, int(cfg.batch_size)):
            batch_end = min(batch_start + int(cfg.batch_size), episode_length)
            batch_rows = episode_rows[batch_start:batch_end]
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
                label_array[valid_rows] = windows[valid_positions].astype(
                    np.dtype(cfg.label_dtype),
                    copy=False,
                )
                valid_supervision[valid_rows] = step_validity[valid_positions]
                written += row_valid_count

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
        "Finalizing labeled dataset: output_dir=%s valid_rows=%d feature_name=%s.%s",
        output_dir,
        int(valid_supervision.reshape(total_frames, -1).any(axis=1).sum()),
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
        "label_source_type": "dataset_action_sequence",
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
        "label_layout": "sequence",
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
        "label_layout": "sequence",
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
