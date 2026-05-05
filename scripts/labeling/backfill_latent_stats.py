#!/usr/bin/env python

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.configs import parser
from lerobot.datasets.io_utils import write_stats
from lerobot.utils.utils import init_logging


@dataclass
class BackfillLatentStatsConfig:
    dataset_root: Path | None = None
    feature_prefix: str = "latent_labels"
    valid_feature_name: str | None = None

    def validate(self) -> None:
        if self.dataset_root is None:
            raise ValueError("Please specify `--dataset_root`.")
        self.dataset_root = self.dataset_root.resolve()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["dataset_root"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _series_to_array(series: pd.Series, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    values = series.to_numpy()
    if values.dtype == object:
        stacked = np.stack([np.asarray(v, dtype=dtype).reshape(shape) for v in values], axis=0)
    else:
        stacked = np.asarray(values, dtype=dtype)
        if stacked.shape[1:] != shape:
            stacked = stacked.reshape((-1, *shape))
    return stacked


def _series_to_valid_mask(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    if values.dtype == object:
        mask = np.stack([np.asarray(v).reshape(-1) for v in values], axis=0)
    else:
        mask = np.asarray(values)
    if mask.ndim > 1:
        mask = mask.reshape(mask.shape[0], -1)[:, 0]
    return mask.astype(bool, copy=False)


def _compute_float_feature_stats(
    *,
    values_by_name: dict[str, list[np.ndarray]],
    feature_infos: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    quantiles = {
        "q01": 0.01,
        "q10": 0.10,
        "q50": 0.50,
        "q90": 0.90,
        "q99": 0.99,
    }
    float_feature_stats = {}
    for name, value_chunks in values_by_name.items():
        if not value_chunks:
            continue

        info = feature_infos[name]
        dtype = np.dtype(info["dtype"])
        if not np.issubdtype(dtype, np.floating):
            continue

        values = np.concatenate(value_chunks, axis=0)
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


def _resolve_latent_feature_names(
    *,
    info: dict[str, Any],
    manifest: dict[str, Any] | None,
    feature_prefix: str,
    valid_feature_name: str | None,
) -> tuple[list[str], str]:
    features = info["features"]

    if manifest is not None:
        candidate_feature_names = list(manifest.get("feature_names", {}).values())
        resolved_valid = valid_feature_name or manifest.get("valid_feature_name")
    else:
        prefix = f"{feature_prefix}."
        candidate_feature_names = [key for key in features if key.startswith(prefix) and not key.endswith(".valid")]
        resolved_valid = valid_feature_name or f"{feature_prefix}.valid"

    float_feature_names = [
        key
        for key in candidate_feature_names
        if key in features and np.issubdtype(np.dtype(features[key]["dtype"]), np.floating)
    ]
    if not float_feature_names:
        raise ValueError("Did not find any floating latent features to backfill stats for.")
    if resolved_valid is None or resolved_valid not in features:
        raise ValueError(f"Could not resolve a valid-feature key, got {resolved_valid!r}.")

    return float_feature_names, resolved_valid


def backfill_latent_stats(cfg: BackfillLatentStatsConfig) -> None:
    cfg.validate()
    assert cfg.dataset_root is not None

    info_path = cfg.dataset_root / "meta" / "info.json"
    stats_path = cfg.dataset_root / "meta" / "stats.json"
    manifest_path = cfg.dataset_root / "label_manifest.json"

    info = _load_json(info_path)
    manifest = _load_json(manifest_path) if manifest_path.exists() else None

    float_feature_names, valid_feature_name = _resolve_latent_feature_names(
        info=info,
        manifest=manifest,
        feature_prefix=cfg.feature_prefix,
        valid_feature_name=cfg.valid_feature_name,
    )

    data_files = sorted((cfg.dataset_root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {cfg.dataset_root / 'data'}")

    feature_infos = {feature_name: info["features"][feature_name] for feature_name in float_feature_names}
    values_by_name = {feature_name: [] for feature_name in float_feature_names}
    for parquet_path in data_files:
        frame = pd.read_parquet(parquet_path, columns=[valid_feature_name, *float_feature_names])
        valid_mask = _series_to_valid_mask(frame[valid_feature_name])
        if not valid_mask.any():
            continue

        for feature_name in float_feature_names:
            feature_info = feature_infos[feature_name]
            dtype = np.dtype(feature_info["dtype"])
            shape = tuple(int(dim) for dim in feature_info["shape"])
            values = _series_to_array(frame[feature_name], shape=shape, dtype=dtype)[valid_mask]
            if values.shape[0] == 0:
                continue
            values_by_name[feature_name].append(values)

    aggregated_stats = _compute_float_feature_stats(
        values_by_name=values_by_name,
        feature_infos=feature_infos,
    )
    if not aggregated_stats:
        raise ValueError("No valid latent rows found; could not compute latent stats.")
    existing_stats = _load_json(stats_path) if stats_path.exists() else {}
    existing_stats.update(aggregated_stats)
    write_stats(existing_stats, cfg.dataset_root)

    if manifest is not None:
        manifest["stats_feature_names"] = sorted(float_feature_names)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logging.info("Updated latent stats for %s", float_feature_names)
    logging.info("Wrote merged stats to %s", stats_path)


@parser.wrap()
def main(cfg: BackfillLatentStatsConfig) -> None:
    init_logging()
    backfill_latent_stats(cfg)


if __name__ == "__main__":
    main()
