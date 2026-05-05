#!/usr/bin/env python

"""
Create a full LeRobot dataset copy with synthetic latent labels.

This mirrors the full-dataset export style of `label_lerobot_dataset.py`, but
instead of calling a policy it generates labels from a simple inline
specification.

`label_spec` schema
-------------------
`label_spec` must decode to a dict with:

    {
      "base": {
        "mode": "constant" | "gaussian" | "uniform",
        ... mode-specific fields ...
      },
      "episode_groups": [
        {
          "episodes": [int, ...],
          "mode": "constant" | "gaussian" | "uniform",
          ... mode-specific fields ...
        },
        ...
      ]
    }

Supported modes:

- constant:
    {"mode": "constant", "constant_value": 0.0}
- gaussian:
    {"mode": "gaussian", "mean": 0.0, "std": 1.0}
- uniform:
    {"mode": "uniform", "low": -1.0, "high": 1.0}

Rows use the `base` mode by default. If a row's `episode_index` is listed in an
`episode_groups[*].episodes` override, that group's mode replaces the base mode.
Episode groups must not overlap.

`label_spec` examples
---------------------
All rows constant:

    --label_spec='{"base":{"mode":"constant","constant_value":0.0},"episode_groups":[]}'

Base Gaussian with episode override:

    --label_spec='{"base":{"mode":"gaussian","mean":0.09195,"std":0.85557},"episode_groups":[{"episodes":[15,16,17,18,19,20,30,31,32,33,34,35,36,37,38,39,40],"mode":"gaussian","mean":0.5,"std":0.01}]}'

YAML-style inline object also works:

    --label_spec='{base: {mode: gaussian, mean: 0.0, std: 0.001}, episode_groups: []}'

Uniform example:

    --label_spec='{"base":{"mode":"uniform","low":-0.5,"high":0.5},"episode_groups":[{"episodes":[0,1,2],"mode":"uniform","low":0.9,"high":1.1}]}'

`latent_shape` / representation shape
-------------------------------------
The latent tensor shape is configurable with `--label_shape`. For example:

    --label_shape='[4,32]'
    --label_shape='[1,128]'

The generated feature is written to:

    <feature_prefix>.<representation_name>

By default:

    latent_labels.continuous_vector_latents
"""

import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import yaml

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

SUPPORTED_MODES = {"constant", "gaussian", "uniform"}
PROGRESS_LOG_EVERY_BATCHES = 100


@dataclass
class SyntheticLatentLabelConfig:
    dataset_repo_id: str | None = None
    dataset_root: str | None = None
    episodes: list[int] | None = None
    output_dir: Path | None = None
    output_repo_id: str | None = None
    feature_prefix: str = "latent_labels"
    representation_name: str = "continuous_vector_latents"
    # Shape of the synthetic latent tensor written per row, for example [4, 32]
    # or [1, 128].
    label_shape: list[int] = field(default_factory=lambda: [4, 32])
    label_dtype: str = "float32"
    invalid_fill_value: float = float("nan")
    label_spec: str | None = None
    batch_size: int = 32768
    seed: int = 0
    force: bool = False
    max_valid_samples: int | None = None

    def validate(self) -> None:
        if not self.dataset_repo_id:
            raise ValueError("Please specify `--dataset_repo_id`.")
        if self.output_dir is None:
            raise ValueError("Please specify `--output_dir`.")
        if self.output_repo_id is None:
            self.output_repo_id = f"{self.dataset_repo_id}_synthetic_labeled"
        if self.feature_prefix.startswith("observation."):
            raise ValueError(
                "feature_prefix must not use observation.*. "
                "Use a top-level namespace such as `latent_labels` or `lam_lapa`."
            )
        if not self.representation_name:
            raise ValueError("representation_name must be non-empty.")
        if not self.label_shape or any(int(dim) < 1 for dim in self.label_shape):
            raise ValueError(f"label_shape must contain positive integers, got {self.label_shape}.")
        np.dtype(self.label_dtype)
        if self.label_spec is None:
            raise ValueError("Please specify `--label_spec`.")
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


def _sorted_unique_episode_list(episodes: Any) -> list[int]:
    if not isinstance(episodes, list) or len(episodes) == 0:
        raise ValueError("episode_groups[*].episodes must be a non-empty list of integers.")
    normalized = []
    for raw in episodes:
        value = int(raw)
        if value < 0:
            raise ValueError(f"Episode indices must be non-negative, got {value}.")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate episode index found in {episodes}.")
    return sorted(normalized)


def _validate_label_mode_object(obj: Any, *, require_episodes: bool) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError(f"Expected label mode object to be a dict, got {type(obj)}.")

    mode = str(obj.get("mode", "")).strip()
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode={mode!r}. Supported modes: {sorted(SUPPORTED_MODES)}.")

    allowed_keys = {"mode"}
    normalized: dict[str, Any] = {"mode": mode}

    if require_episodes:
        allowed_keys.add("episodes")
        normalized["episodes"] = _sorted_unique_episode_list(obj.get("episodes"))

    if mode == "constant":
        allowed_keys.add("constant_value")
        if "constant_value" not in obj:
            raise ValueError("constant mode requires `constant_value`.")
        normalized["constant_value"] = float(obj["constant_value"])
    elif mode == "gaussian":
        allowed_keys.update({"mean", "std"})
        if "mean" not in obj or "std" not in obj:
            raise ValueError("gaussian mode requires `mean` and `std`.")
        normalized["mean"] = float(obj["mean"])
        normalized["std"] = float(obj["std"])
        if normalized["std"] < 0:
            raise ValueError(f"gaussian std must be >= 0, got {normalized['std']}.")
    elif mode == "uniform":
        allowed_keys.update({"low", "high"})
        if "low" not in obj or "high" not in obj:
            raise ValueError("uniform mode requires `low` and `high`.")
        normalized["low"] = float(obj["low"])
        normalized["high"] = float(obj["high"])
        if normalized["high"] < normalized["low"]:
            raise ValueError(
                f"uniform range must satisfy high >= low, got low={normalized['low']} high={normalized['high']}."
            )

    extra_keys = sorted(set(obj).difference(allowed_keys))
    if extra_keys:
        raise ValueError(f"Unexpected keys for mode {mode!r}: {extra_keys}")

    return normalized


def parse_label_spec(raw_spec: str) -> dict[str, Any]:
    try:
        spec = json.loads(raw_spec)
    except json.JSONDecodeError:
        # Draccus CLI parsing can strip some quoting from inline objects, so
        # accept YAML-style mappings as a fallback for command-line usage.
        try:
            spec = yaml.safe_load(raw_spec)
        except yaml.YAMLError as exc:
            raise ValueError(f"label_spec must be valid JSON or YAML: {exc}") from exc

    if not isinstance(spec, dict):
        raise TypeError(f"label_spec must decode to a dict, got {type(spec)}.")
    if "base" not in spec:
        raise ValueError("label_spec must contain a `base` object.")

    allowed_keys = {"base", "episode_groups"}
    extra_keys = sorted(set(spec).difference(allowed_keys))
    if extra_keys:
        raise ValueError(f"Unexpected top-level label_spec keys: {extra_keys}")

    base = _validate_label_mode_object(spec["base"], require_episodes=False)

    raw_groups = spec.get("episode_groups", [])
    if not isinstance(raw_groups, list):
        raise TypeError("label_spec.episode_groups must be a list when provided.")

    groups = []
    seen_episodes: dict[int, int] = {}
    for group_idx, raw_group in enumerate(raw_groups):
        group = _validate_label_mode_object(raw_group, require_episodes=True)
        for episode in group["episodes"]:
            if episode in seen_episodes:
                raise ValueError(
                    f"Episode {episode} is assigned to multiple groups: "
                    f"{seen_episodes[episode]} and {group_idx}."
                )
            seen_episodes[episode] = group_idx
        groups.append(group)

    return {"base": base, "episode_groups": groups}


def _sample_mode_values(
    *,
    mode_spec: dict[str, Any],
    count: int,
    shape: tuple[int, ...],
    dtype: np.dtype,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode_spec["mode"] == "constant":
        return np.full((count, *shape), mode_spec["constant_value"], dtype=dtype)
    if mode_spec["mode"] == "gaussian":
        if not np.issubdtype(dtype, np.floating):
            raise ValueError(f"gaussian mode requires a floating dtype, got {dtype}.")
        values = rng.normal(loc=mode_spec["mean"], scale=mode_spec["std"], size=(count, *shape))
        return values.astype(dtype, copy=False)
    if mode_spec["mode"] == "uniform":
        if not np.issubdtype(dtype, np.floating):
            raise ValueError(f"uniform mode requires a floating dtype, got {dtype}.")
        values = rng.uniform(low=mode_spec["low"], high=mode_spec["high"], size=(count, *shape))
        return values.astype(dtype, copy=False)
    raise ValueError(f"Unsupported mode={mode_spec['mode']!r}.")


def generate_synthetic_labels(
    *,
    row_indices: np.ndarray,
    episode_indices: np.ndarray,
    label_shape: tuple[int, ...],
    label_dtype: np.dtype,
    label_spec: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    if row_indices.ndim != 1 or episode_indices.ndim != 1:
        raise ValueError("row_indices and episode_indices must be 1D arrays.")
    if row_indices.shape[0] != episode_indices.shape[0]:
        raise ValueError("row_indices and episode_indices must have the same length.")

    labels = _sample_mode_values(
        mode_spec=label_spec["base"],
        count=row_indices.shape[0],
        shape=label_shape,
        dtype=label_dtype,
        rng=rng,
    )

    for group in label_spec["episode_groups"]:
        group_mask = np.isin(episode_indices, np.asarray(group["episodes"], dtype=np.int64))
        group_count = int(group_mask.sum())
        if group_count == 0:
            continue
        labels[group_mask] = _sample_mode_values(
            mode_spec=group,
            count=group_count,
            shape=label_shape,
            dtype=label_dtype,
            rng=rng,
        )
    return labels


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

    quantiles = {
        "q01": 0.01,
        "q10": 0.10,
        "q50": 0.50,
        "q90": 0.90,
        "q99": 0.99,
    }
    float_feature_stats = {}
    for name, info in feature_infos.items():
        dtype = np.dtype(info["dtype"])
        if not np.issubdtype(dtype, np.floating):
            continue
        values = label_arrays[name][valid_rows]
        if values.shape[0] == 0:
            continue

        values64 = values.astype(np.float64, copy=False)
        stats = {
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "mean": values64.mean(axis=0),
            "std": values64.std(axis=0, ddof=0),
            "count": np.array([values.shape[0]], dtype=np.int64),
        }
        for key, q in quantiles.items():
            stats[key] = np.quantile(values64, q, axis=0)
        float_feature_stats[name] = stats

    return float_feature_stats


def _iter_index_episode_batches(
    dataset: LeRobotDataset,
    batch_size: int,
):
    # Iterate only the columns needed to synthesize labels. This avoids the very
    # expensive eager full-column materialization path on large datasets.
    index_dataset = dataset.hf_dataset.with_format(None).select_columns(["index", "episode_index"])
    for batch in index_dataset.iter(batch_size=batch_size):
        yield (
            np.asarray(batch["index"], dtype=np.int64),
            np.asarray(batch["episode_index"], dtype=np.int64),
        )


def export_synthetic_latent_dataset(cfg: SyntheticLatentLabelConfig) -> None:
    cfg.validate()
    if cfg.output_dir is None:
        raise ValueError("output_dir was not configured.")
    if cfg.output_repo_id is None:
        raise ValueError("output_repo_id was not configured.")
    if cfg.dataset_repo_id is None:
        raise ValueError("dataset_repo_id was not configured.")

    label_spec = parse_label_spec(cfg.label_spec or "")
    label_dtype = np.dtype(cfg.label_dtype)
    label_shape = tuple(int(dim) for dim in cfg.label_shape)
    rng = np.random.default_rng(int(cfg.seed))

    output_dir = cfg.output_dir.resolve()
    _prepare_output_dir(output_dir, cfg.force)

    source_dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root)
    processing_dataset = (
        source_dataset
        if cfg.episodes is None
        else LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root, episodes=cfg.episodes)
    )

    total_frames = int(source_dataset.meta.total_frames)
    processing_rows = len(processing_dataset)
    representation_full_name = f"{cfg.feature_prefix}.{cfg.representation_name}"
    label_array = np.full(
        (total_frames, *label_shape),
        cfg.invalid_fill_value,
        dtype=label_dtype,
    )
    valid_supervision = np.zeros((total_frames, 1), dtype=np.int64)

    feature_infos = {
        cfg.representation_name: {
            "dtype": label_dtype.name,
            "shape": label_shape,
            "names": None,
        }
    }

    logging.info(
        "Synthetic label export setup:\n%s",
        pformat(
            {
                "dataset_repo_id": cfg.dataset_repo_id,
                "dataset_root": cfg.dataset_root,
                "episodes": cfg.episodes,
                "output_dir": str(output_dir),
                "output_repo_id": cfg.output_repo_id,
                "feature_prefix": cfg.feature_prefix,
                "representation_name": cfg.representation_name,
                "label_shape": label_shape,
                "label_dtype": label_dtype.name,
                "batch_size": cfg.batch_size,
                "seed": int(cfg.seed),
                "max_valid_samples": cfg.max_valid_samples,
                "label_spec": label_spec,
            }
        ),
    )

    written = 0
    start_time = time.perf_counter()
    for batch_idx, (row_indices, episode_indices) in enumerate(
        _iter_index_episode_batches(processing_dataset, cfg.batch_size),
        start=1,
    ):
        if cfg.max_valid_samples is not None:
            remaining = cfg.max_valid_samples - written
            if remaining <= 0:
                break
            if row_indices.shape[0] > remaining:
                row_indices = row_indices[:remaining]
                episode_indices = episode_indices[:remaining]

        if row_indices.shape[0] == 0:
            continue

        labels = generate_synthetic_labels(
            row_indices=row_indices,
            episode_indices=episode_indices,
            label_shape=label_shape,
            label_dtype=label_dtype,
            label_spec=label_spec,
            rng=rng,
        )
        label_array[row_indices] = labels
        valid_supervision[row_indices, 0] = 1
        written += int(row_indices.shape[0])

        if batch_idx % PROGRESS_LOG_EVERY_BATCHES == 0 or written == processing_rows:
            elapsed_s = max(time.perf_counter() - start_time, 1e-9)
            rate = written / elapsed_s
            logging.info(
                "Progress: rows=%d/%d rate=%.1f rows/s",
                written,
                processing_rows,
                rate,
            )

        if cfg.max_valid_samples is not None and written >= cfg.max_valid_samples:
            break

    relabeled_dataset = add_features(
        dataset=source_dataset,
        features={
            representation_full_name: (
                _format_feature_values(label_array, label_shape),
                feature_infos[cfg.representation_name],
            ),
            f"{cfg.feature_prefix}.valid": (
                valid_supervision,
                {"dtype": "int64", "shape": (1,), "names": None},
            ),
        },
        output_dir=output_dir,
        repo_id=cfg.output_repo_id,
    )

    latent_stats = _compute_float_feature_stats(
        label_arrays={cfg.representation_name: label_array},
        valid_supervision=valid_supervision,
        feature_infos=feature_infos,
    )
    if latent_stats:
        merged_stats = dict(relabeled_dataset.meta.stats or {})
        merged_stats.update(
            {f"{cfg.feature_prefix}.{name}": stats for name, stats in latent_stats.items()}
        )
        write_stats(merged_stats, relabeled_dataset.root)

    checkpoint_meta = infer_checkpoint_metadata(None)
    label_manifest = {
        "labeling_kind": "synthetic",
        "source_dataset_repo_id": cfg.dataset_repo_id,
        "source_dataset_root": cfg.dataset_root,
        "episodes": cfg.episodes,
        "output_repo_id": cfg.output_repo_id,
        "output_dir": str(output_dir),
        "feature_prefix": cfg.feature_prefix,
        "feature_names": {cfg.representation_name: representation_full_name},
        "valid_feature_name": f"{cfg.feature_prefix}.valid",
        "stats_feature_names": [representation_full_name]
        if np.issubdtype(label_dtype, np.floating)
        else [],
        "representation_name": cfg.representation_name,
        "label_shape": list(label_shape),
        "label_dtype": label_dtype.name,
        "seed": int(cfg.seed),
        "label_spec": label_spec,
        "num_valid_labels": int(valid_supervision.sum()),
    }
    label_manifest_path = output_dir / "label_manifest.json"
    label_manifest_path.write_text(json.dumps(label_manifest, indent=2) + "\n")

    export_manifest_path = output_dir / "export_manifest.json"
    export_manifest = {
        "artifact_type": "synthetic_latent_export",
        "export_kind": "full_labeled_dataset",
        "suite_name": "synthetic_latent_export",
        "suite_version": "v1",
        "artifact_id": make_artifact_id(
            suite_name="synthetic_latent_export",
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
        "feature_names": {cfg.representation_name: representation_full_name},
        "valid_feature_name": f"{cfg.feature_prefix}.valid",
        "stats_feature_names": [representation_full_name]
        if np.issubdtype(label_dtype, np.floating)
        else [],
        "representation_name": cfg.representation_name,
        "label_shape": list(label_shape),
        "label_dtype": label_dtype.name,
        "seed": int(cfg.seed),
        "label_spec": label_spec,
        "num_rows": int(source_dataset.meta.total_frames),
        "num_valid_labels": int(valid_supervision.sum()),
        "label_manifest_path": str(label_manifest_path),
    }
    register_artifact(
        manifest_path=export_manifest_path,
        manifest=export_manifest,
        registry_candidates=[output_dir, cfg.dataset_root],
    )

    logging.info("Finished synthetic export to %s", relabeled_dataset.root)
    logging.info("Manifests written to %s and %s", label_manifest_path, export_manifest_path)


@parser.wrap()
def main(cfg: SyntheticLatentLabelConfig) -> None:
    register_third_party_plugins()
    init_logging()
    export_synthetic_latent_dataset(cfg)


if __name__ == "__main__":
    main()
