#!/usr/bin/env python

"""Probe latent labels against real actions.

This script intentionally keeps the latent analysis narrow: load an exported
latent-label dataset, derive current and future action targets, then fit ridge
and/or MLP probes on GPU with PyTorch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _artifact_registry import infer_checkpoint_metadata, load_export_manifest, make_artifact_id, register_artifact


@dataclass(frozen=True)
class ProbeSplit:
    train_rows: np.ndarray
    test_rows: np.ndarray
    val_rows: np.ndarray


def parse_hidden_dims(raw: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(
            "--probe-mlp-hidden-dims must be a comma-separated list of positive integers."
        )
    return dims


def parse_bool(raw: str) -> bool:
    return raw.lower() in {"1", "true", "yes", "on"}


def parse_future_target_config(raw: str) -> dict[str, Any]:
    value = raw.strip()
    if not value:
        raise argparse.ArgumentTypeError("--future-target-config must not be empty.")
    try:
        candidate = Path(value)
        payload = json.loads(candidate.read_text() if candidate.exists() else value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Failed to parse --future-target-config: {exc}") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("--future-target-config must decode to a JSON object.")
    return payload


def parse_probe_feature_sets(raw: str) -> set[str]:
    values = {part.strip() for part in raw.split(",") if part.strip()}
    if not values:
        raise argparse.ArgumentTypeError("--probe-feature-sets must not be empty.")
    allowed = {"ids_onehot", "codebook_vectors", "continuous"}
    if values == {"all"}:
        return allowed
    unknown = values.difference(allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown --probe-feature-sets entries: {sorted(unknown)}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPU ridge/MLP probes on latent-label exports.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root of the labeled dataset.")
    parser.add_argument("--feature-prefix", type=str, required=True, help="Feature prefix, e.g. latent_labels.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where outputs are written.")
    parser.add_argument("--future-frames", type=int, default=10, help="Future action horizon for target aggregation.")
    parser.add_argument(
        "--future-target-config",
        type=parse_future_target_config,
        default=None,
        help="Optional JSON object or JSON file describing future action target aggregation.",
    )
    parser.add_argument(
        "--probe-max-samples",
        type=int,
        default=0,
        help="Maximum valid rows for probes. Set to 0 to use all valid rows.",
    )
    parser.add_argument("--probe-test-size", type=float, default=0.2, help="Held-out test fraction.")
    parser.add_argument("--probe-val-size", type=float, default=0.1, help="Validation fraction from the train split.")
    parser.add_argument(
        "--probe-model",
        choices=("ridge", "mlp", "both"),
        default="both",
        help="Probe backend to run.",
    )
    parser.add_argument(
        "--probe-feature-sets",
        type=parse_probe_feature_sets,
        default={"continuous"},
        help="Comma-separated feature sets: continuous, codebook_vectors, ids_onehot, or all.",
    )
    parser.add_argument(
        "--probe-split",
        choices=("row", "episode"),
        default="episode",
        help="Whether held-out rows are random rows or full episodes.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="L2 regularization for ridge.")
    parser.add_argument("--probe-mlp-hidden-dims", type=parse_hidden_dims, default=(512, 256))
    parser.add_argument("--probe-mlp-alpha", type=float, default=1e-4, help="L2 regularization for MLP.")
    parser.add_argument(
        "--probe-mlp-max-iter",
        type=int,
        default=200,
        help="Maximum MLP epochs. Kept for compatibility with the old sklearn argument name.",
    )
    parser.add_argument("--probe-mlp-batch-size", type=int, default=8192)
    parser.add_argument("--probe-mlp-lr", type=float, default=1e-3)
    parser.add_argument("--probe-mlp-early-stopping", type=parse_bool, default=True)
    parser.add_argument("--probe-mlp-n-iter-no-change", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda", help="Torch device. Falls back to CPU if CUDA is unavailable.")
    parser.add_argument("--seed", type=int, default=0)

    # Legacy flags accepted as no-ops so older command templates do not fail.
    parser.add_argument("--top-k-sequences", type=int, default=50, help=argparse.SUPPRESS)
    parser.add_argument("--scatter-points", type=int, default=100000, help=argparse.SUPPRESS)
    parser.add_argument("--pca-fit-points", type=int, default=50000, help=argparse.SUPPRESS)
    parser.add_argument("--rounded-decimals", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--action-bins", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--bucket-kmeans-clusters", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--bucket-kmeans-fit-samples", type=int, default=50000, help=argparse.SUPPRESS)
    parser.add_argument("--bucket-top-k", type=int, default=20, help=argparse.SUPPRESS)
    parser.add_argument("--bucket-progress-bins", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--action-bucket-kmeans-clusters", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--action-bucket-kmeans-fit-samples", type=int, default=50000, help=argparse.SUPPRESS)
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_info(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / "meta" / "info.json").read_text())


def make_dataset(dataset_root: Path) -> ds.Dataset:
    return ds.dataset(dataset_root / "data", format="parquet")


def load_valid_counts(dataset: ds.Dataset, valid_col: str) -> dict[str, int]:
    table = dataset.to_table(columns=[valid_col])
    return {str(item["values"]): int(item["counts"]) for item in pc.value_counts(table[valid_col]).to_pylist()}


def load_ids(dataset: ds.Dataset, ids_col: str, valid_col: str) -> np.ndarray:
    table = dataset.to_table(columns=[ids_col], filter=ds.field(valid_col) == 1)
    ids = np.stack(table[ids_col].to_numpy(zero_copy_only=False)).astype(np.int64, copy=False)
    return ensure_2d_rows(ids)


def load_float_array(dataset: ds.Dataset, column_name: str, valid_col: str) -> np.ndarray:
    table = dataset.to_table(columns=[column_name], filter=ds.field(valid_col) == 1)
    obj = table[column_name].to_numpy(zero_copy_only=False)
    return np.stack([np.stack(row, axis=0) for row in obj], axis=0).astype(np.float32, copy=False)


def load_action_context(dataset: ds.Dataset, action_col: str, valid_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = dataset.to_table(columns=[action_col, valid_col, "episode_index"])
    actions = np.stack(table[action_col].to_numpy(zero_copy_only=False)).astype(np.float32, copy=False)
    valid = table[valid_col].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    episode_index = table["episode_index"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return actions, valid, episode_index


def ensure_2d_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D row array, got shape={arr.shape}.")
    return arr


def flatten_valid_latents(values: np.ndarray) -> np.ndarray:
    return values.reshape(values.shape[0], -1).astype(np.float32, copy=False)


def episode_ranges(episode_index: np.ndarray) -> list[tuple[int, int]]:
    split_points = np.flatnonzero(np.diff(episode_index)) + 1
    starts = np.concatenate(([0], split_points))
    ends = np.concatenate((split_points, [len(episode_index)]))
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def extract_valid_episode_index(valid: np.ndarray, episode_index: np.ndarray) -> np.ndarray:
    chunks = []
    for start, end in episode_ranges(episode_index):
        valid_ep = valid[start:end]
        n_valid = int(np.sum(valid_ep == 1))
        if not np.all(valid_ep[:n_valid] == 1) or not np.all(valid_ep[n_valid:] == 0):
            raise ValueError("Expected valid rows to be contiguous at the start of each episode.")
        if n_valid > 0:
            chunks.append(episode_index[start : start + n_valid])
    return np.concatenate(chunks, axis=0).astype(np.int64, copy=False) if chunks else np.empty((0,), dtype=np.int64)


def infer_episode_tail_counts(valid: np.ndarray, episode_index: np.ndarray) -> list[int]:
    return [int(np.sum(valid[start:end] == 0)) for start, end in episode_ranges(episode_index)]


def default_future_target_config(action_dim: int) -> dict[str, Any]:
    return {"name": "future_action_mean", "dims": [{"start": 0, "end": action_dim, "mode": "mean"}]}


def validate_future_target_config(config: dict[str, Any], action_dim: int) -> dict[str, Any]:
    if "name" not in config or not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("Future target config must define a non-empty string field `name`.")
    if config["name"] == "current_action":
        raise ValueError("Future target config name `current_action` is reserved.")
    raw_dims = config.get("dims")
    if not isinstance(raw_dims, list) or not raw_dims:
        raise ValueError("Future target config must contain a non-empty `dims` list.")

    coverage = np.zeros(action_dim, dtype=np.int64)
    normalized_dims = []
    for idx, raw_dim in enumerate(raw_dims):
        start = int(raw_dim["start"])
        end = int(raw_dim["end"])
        mode = str(raw_dim["mode"])
        if mode not in {"mean", "sum", "last", "compose_rotation"}:
            raise ValueError(f"Unsupported future target mode {mode!r} in dims[{idx}].")
        if start < 0 or end <= start or end > action_dim:
            raise ValueError(f"Invalid future target slice [{start}, {end}) for action_dim={action_dim}.")
        coverage[start:end] += 1
        normalized = {"start": start, "end": end, "mode": mode}
        if "scale" in raw_dim:
            normalized["scale"] = raw_dim["scale"]
        if mode == "compose_rotation":
            representation = str(raw_dim.get("representation", ""))
            if representation not in {"euler_delta", "rotvec_delta"}:
                raise ValueError("compose_rotation requires representation euler_delta or rotvec_delta.")
            if end - start != 3:
                raise ValueError("compose_rotation requires a 3D slice.")
            normalized["representation"] = representation
        normalized_dims.append(normalized)

    if not np.all(coverage == 1):
        raise ValueError("Future target config must cover each action dimension exactly once.")
    return {"name": config["name"].strip(), "dims": normalized_dims}


def _compose_rotation_window(deltas: np.ndarray, representation: str) -> np.ndarray:
    net = Rotation.identity()
    for delta in deltas:
        delta_rotation = (
            Rotation.from_euler("xyz", delta, degrees=False)
            if representation == "euler_delta"
            else Rotation.from_rotvec(delta)
        )
        net = delta_rotation * net
    return (
        net.as_euler("xyz", degrees=False).astype(np.float32, copy=False)
        if representation == "euler_delta"
        else net.as_rotvec().astype(np.float32, copy=False)
    )


def _aggregate_future_window(
    actions_ep: np.ndarray,
    n_valid: int,
    future_frames: int,
    target_config: dict[str, Any],
) -> np.ndarray:
    action_dim = actions_ep.shape[1]
    out = np.empty((n_valid, action_dim), dtype=np.float32)
    for dim_spec in target_config["dims"]:
        start = int(dim_spec["start"])
        end = int(dim_spec["end"])
        mode = str(dim_spec["mode"])
        block = actions_ep[:, start:end].astype(np.float32, copy=False)
        if "scale" in dim_spec:
            block = block * np.asarray(dim_spec["scale"], dtype=np.float32)

        if mode in {"mean", "sum"}:
            cumsum = np.vstack([np.zeros((1, end - start), dtype=np.float32), np.cumsum(block, axis=0)])
            future_sum = cumsum[1 + future_frames :] - cumsum[1:-future_frames]
            future_sum = future_sum[:n_valid]
            out[:, start:end] = future_sum / float(future_frames) if mode == "mean" else future_sum
        elif mode == "last":
            out[:, start:end] = block[future_frames : future_frames + n_valid]
        elif mode == "compose_rotation":
            windows = np.empty((n_valid, end - start), dtype=np.float32)
            for row in range(n_valid):
                windows[row] = _compose_rotation_window(
                    block[row + 1 : row + 1 + future_frames],
                    str(dim_spec["representation"]),
                )
            out[:, start:end] = windows
        else:
            raise ValueError(f"Unsupported future target mode during aggregation: {mode!r}")
    return out


def derive_action_targets(
    actions: np.ndarray,
    valid: np.ndarray,
    episode_index: np.ndarray,
    future_frames: int,
    future_target_config: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    if future_frames <= 0:
        raise ValueError("--future-frames must be positive.")
    action_dim = int(actions.shape[1])
    target_config = validate_future_target_config(
        default_future_target_config(action_dim) if future_target_config is None else future_target_config,
        action_dim=action_dim,
    )
    current_chunks = []
    future_chunks = []
    for start, end in episode_ranges(episode_index):
        actions_ep = actions[start:end]
        valid_ep = valid[start:end]
        n_valid = int(np.sum(valid_ep == 1))
        if not np.all(valid_ep[:n_valid] == 1) or not np.all(valid_ep[n_valid:] == 0):
            raise ValueError("Expected valid rows to be contiguous at the start of each episode.")
        if n_valid == 0:
            continue
        if actions_ep.shape[0] < n_valid + future_frames:
            raise ValueError(
                f"Episode length {actions_ep.shape[0]} is too short for n_valid={n_valid} and future_frames={future_frames}."
            )
        current_chunks.append(actions_ep[:n_valid])
        future_chunks.append(
            _aggregate_future_window(actions_ep, n_valid, future_frames, target_config)[:n_valid]
        )
    return {
        "current_action": np.concatenate(current_chunks, axis=0),
        str(target_config["name"]): np.concatenate(future_chunks, axis=0),
    }


def select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


def make_probe_split(
    valid_episode_index: np.ndarray,
    max_samples: int,
    test_size: float,
    val_size: float,
    seed: int,
    mode: str,
) -> ProbeSplit:
    if valid_episode_index.shape[0] < 3:
        raise ValueError("Action probes require at least three valid rows.")
    if max_samples < 0:
        raise ValueError("--probe-max-samples must be >= 0.")
    effective_max_samples = valid_episode_index.shape[0] if max_samples == 0 else max_samples
    rng = np.random.default_rng(seed)

    if mode == "row":
        sample_size = min(effective_max_samples, valid_episode_index.shape[0])
        sampled_rows = rng.choice(valid_episode_index.shape[0], size=sample_size, replace=False)
        train_val_rows, test_rows = train_test_split(
            sampled_rows,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        relative_val_size = val_size / max(1.0 - test_size, 1e-8)
        train_rows, val_rows = train_test_split(
            train_val_rows,
            test_size=relative_val_size,
            random_state=seed + 1,
            shuffle=True,
        )
        return ProbeSplit(np.sort(train_rows), np.sort(test_rows), np.sort(val_rows))

    if mode != "episode":
        raise ValueError(f"Unsupported probe split mode: {mode!r}")

    unique_eps, counts = np.unique(valid_episode_index, return_counts=True)
    if unique_eps.shape[0] < 3:
        raise ValueError("Episode-split probes require at least three valid episodes.")
    if effective_max_samples < valid_episode_index.shape[0]:
        order = rng.permutation(unique_eps.shape[0])
        selected_eps = []
        running_total = 0
        for order_idx in order:
            selected_eps.append(unique_eps[order_idx])
            running_total += int(counts[order_idx])
            if running_total >= effective_max_samples and len(selected_eps) >= 3:
                break
        selected_eps = np.asarray(selected_eps, dtype=np.int64)
    else:
        selected_eps = unique_eps

    train_val_eps, test_eps = train_test_split(selected_eps, test_size=test_size, random_state=seed, shuffle=True)
    relative_val_size = val_size / max(1.0 - test_size, 1e-8)
    train_eps, val_eps = train_test_split(
        train_val_eps,
        test_size=relative_val_size,
        random_state=seed + 1,
        shuffle=True,
    )
    train_rows = np.flatnonzero(np.isin(valid_episode_index, train_eps)).astype(np.int64, copy=False)
    test_rows = np.flatnonzero(np.isin(valid_episode_index, test_eps)).astype(np.int64, copy=False)
    val_rows = np.flatnonzero(np.isin(valid_episode_index, val_eps)).astype(np.int64, copy=False)
    if train_rows.shape[0] == 0 or test_rows.shape[0] == 0 or val_rows.shape[0] == 0:
        raise ValueError("Probe split produced an empty train, validation, or test set.")
    return ProbeSplit(np.sort(train_rows), np.sort(test_rows), np.sort(val_rows))


def make_dense_id_features(ids: np.ndarray, train_rows: np.ndarray, other_rows: list[np.ndarray]) -> list[np.ndarray]:
    """One-hot encode ID positions with train-fit categories.

    This is dense by design because torch linear/MLP probes operate on dense
    tensors. Use it for small discrete configurations, not huge seq1/cb4096 full
    datasets unless memory is acceptable.
    """
    train_ids = ids[train_rows]
    categories = [np.unique(train_ids[:, pos]) for pos in range(train_ids.shape[1])]
    offsets = np.cumsum([0] + [len(cat) for cat in categories[:-1]])
    total_dim = int(sum(len(cat) for cat in categories))
    encoded_arrays = []
    for rows in [train_rows, *other_rows]:
        encoded = np.zeros((rows.shape[0], total_dim), dtype=np.float32)
        values = ids[rows]
        for pos, cat in enumerate(categories):
            local = np.searchsorted(cat, values[:, pos])
            in_range = (local < len(cat)) & (cat[np.clip(local, 0, len(cat) - 1)] == values[:, pos])
            encoded[np.flatnonzero(in_range), offsets[pos] + local[in_range]] = 1.0
        encoded_arrays.append(encoded)
    return encoded_arrays


def standardize_from_train(
    train_values: np.ndarray,
    test_values: np.ndarray,
    val_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = train_values.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (
        ((train_values - mean) / std).astype(np.float32, copy=False),
        ((test_values - mean) / std).astype(np.float32, copy=False),
        ((val_values - mean) / std).astype(np.float32, copy=False),
        mean.squeeze(0),
        std.squeeze(0),
    )


def build_feature_arrays(
    *,
    feature_name: str,
    features: np.ndarray,
    split: ProbeSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_name == "ids_onehot":
        return tuple(make_dense_id_features(features, split.train_rows, [split.test_rows, split.val_rows]))  # type: ignore[return-value]
    train_x = features[split.train_rows]
    test_x = features[split.test_rows]
    val_x = features[split.val_rows]
    return standardize_from_train(train_x, test_x, val_x)[:3]


def r2_and_mse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true64 = y_true.astype(np.float64, copy=False)
    y_pred64 = y_pred.astype(np.float64, copy=False)
    residual = np.sum((y_true64 - y_pred64) ** 2, axis=0)
    centered = np.sum((y_true64 - y_true64.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - residual / np.maximum(centered, 1e-12)
    mse = np.mean((y_true64 - y_pred64) ** 2, axis=0)
    return r2.astype(np.float64, copy=False), mse.astype(np.float64, copy=False)


def score_predictions(
    *,
    probe_model: str,
    feature_name: str,
    target_name: str,
    split_mode: str,
    y_test: np.ndarray,
    prediction: np.ndarray,
    n_train: int,
    n_test: int,
    n_val: int,
    train_seconds: float,
    epochs: int | None,
) -> list[dict[str, Any]]:
    r2, mse = r2_and_mse(y_test, prediction)
    avg_r2 = float(np.mean(r2))
    avg_mse = float(np.mean(mse))
    return [
        {
            "probe_model": probe_model,
            "split_mode": split_mode,
            "feature_set": feature_name,
            "target": target_name,
            "action_dim": action_dim,
            "r2": float(r2_value),
            "avg_r2_for_target": avg_r2,
            "mse": float(mse_value),
            "avg_mse_for_target": avg_mse,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_val": int(n_val),
            "train_seconds": float(train_seconds),
            "epochs": None if epochs is None else int(epochs),
        }
        for action_dim, (r2_value, mse_value) in enumerate(zip(r2.tolist(), mse.tolist(), strict=True))
    ]


def fit_ridge_torch(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    device: torch.device,
) -> np.ndarray:
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    x_test_t = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    y_mean = y_train_t.mean(dim=0, keepdim=True)
    y_centered = y_train_t - y_mean
    gram = x_train_t.T @ x_train_t
    eye = torch.eye(gram.shape[0], dtype=torch.float32, device=device)
    weights = torch.linalg.solve(gram + float(alpha) * eye, x_train_t.T @ y_centered)
    prediction = x_test_t @ weights + y_mean
    return prediction.detach().cpu().numpy().astype(np.float32, copy=False)


class MLPProbe(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([torch.nn.Linear(prev_dim, hidden_dim), torch.nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_mlp_torch(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: tuple[int, ...],
    alpha: float,
    max_epochs: int,
    batch_size: int,
    lr: float,
    early_stopping: bool,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, int, float, float]:
    torch.manual_seed(seed)
    model = MLPProbe(x_train.shape[1], y_train.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=alpha)
    loss_fn = torch.nn.MSELoss()
    train_x = torch.as_tensor(x_train, dtype=torch.float32)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    val_x = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    test_x = torch.as_tensor(x_test, dtype=torch.float32, device=device)

    generator = torch.Generator()
    generator.manual_seed(seed)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        permutation = torch.randperm(train_x.shape[0], generator=generator)
        model.train()
        for start in range(0, train_x.shape[0], batch_size):
            batch_rows = permutation[start : start + batch_size]
            xb = train_x[batch_rows].to(device, non_blocking=True)
            yb = train_y[batch_rows].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
        if early_stopping and stale_epochs >= patience:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        prediction = model(test_x).detach().cpu().numpy().astype(np.float32, copy=False)
    train_seconds = time.time() - start_time
    return prediction, best_epoch, best_val, train_seconds


def summarize_probe_scores(probe_df: pd.DataFrame) -> pd.DataFrame:
    return (
        probe_df.groupby(["probe_model", "split_mode", "feature_set", "target"], as_index=False)
        .agg(
            mean_r2=("r2", "mean"),
            mean_mse=("mse", "mean"),
            n_train=("n_train", "first"),
            n_test=("n_test", "first"),
            n_val=("n_val", "first"),
            train_seconds=("train_seconds", "first"),
            epochs=("epochs", "first"),
        )
        .sort_values(["probe_model", "mean_r2", "mean_mse"], ascending=[True, False, True], ignore_index=True)
    )


def run_probes(
    *,
    feature_sets: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    split: ProbeSplit,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name, features in feature_sets.items():
        x_train, x_test, x_val = build_feature_arrays(feature_name=feature_name, features=features, split=split)
        for target_name, target_values in targets.items():
            y_train = target_values[split.train_rows].astype(np.float32, copy=False)
            y_test = target_values[split.test_rows].astype(np.float32, copy=False)
            y_val = target_values[split.val_rows].astype(np.float32, copy=False)

            if args.probe_model in {"ridge", "both"}:
                start_time = time.time()
                prediction = fit_ridge_torch(
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    alpha=args.ridge_alpha,
                    device=device,
                )
                rows.extend(
                    score_predictions(
                        probe_model="ridge",
                        feature_name=feature_name,
                        target_name=target_name,
                        split_mode=args.probe_split,
                        y_test=y_test,
                        prediction=prediction,
                        n_train=len(split.train_rows),
                        n_test=len(split.test_rows),
                        n_val=len(split.val_rows),
                        train_seconds=time.time() - start_time,
                        epochs=None,
                    )
                )

            if args.probe_model in {"mlp", "both"}:
                prediction, epochs, val_loss, train_seconds = fit_mlp_torch(
                    x_train=x_train,
                    x_test=x_test,
                    x_val=x_val,
                    y_train=y_train,
                    y_test=y_test,
                    y_val=y_val,
                    hidden_dims=args.probe_mlp_hidden_dims,
                    alpha=args.probe_mlp_alpha,
                    max_epochs=args.probe_mlp_max_iter,
                    batch_size=args.probe_mlp_batch_size,
                    lr=args.probe_mlp_lr,
                    early_stopping=args.probe_mlp_early_stopping,
                    patience=args.probe_mlp_n_iter_no_change,
                    seed=args.seed,
                    device=device,
                )
                model_rows = score_predictions(
                    probe_model="mlp",
                    feature_name=feature_name,
                    target_name=target_name,
                    split_mode=args.probe_split,
                    y_test=y_test,
                    prediction=prediction,
                    n_train=len(split.train_rows),
                    n_test=len(split.test_rows),
                    n_val=len(split.val_rows),
                    train_seconds=train_seconds,
                    epochs=epochs,
                )
                for row in model_rows:
                    row["best_val_mse"] = float(val_loss)
                rows.extend(model_rows)

    return pd.DataFrame(rows).sort_values(
        ["probe_model", "feature_set", "target", "action_dim"],
        ignore_index=True,
    )


def pick_feature_name(info: dict[str, Any], feature_prefix: str, *suffixes: str, required: bool = True) -> str | None:
    for suffix in suffixes:
        candidate = f"{feature_prefix}.{suffix}"
        if candidate in info["features"]:
            return candidate
    if required:
        raise KeyError(f"Missing feature for prefix {feature_prefix!r}; tried suffixes {list(suffixes)}.")
    return None


def value_summary(values: np.ndarray) -> dict[str, Any]:
    flat = values.reshape(-1).astype(np.float64, copy=False)
    return {
        "shape": list(values.shape),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "median": float(np.median(flat)),
        "max": float(np.max(flat)),
    }


def best_probe_rows(probe_summary_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for probe_model, model_df in probe_summary_df.groupby("probe_model", sort=True):
        best_r2 = model_df.sort_values(["mean_r2", "mean_mse"], ascending=[False, True]).iloc[0]
        best_mse = model_df.sort_values(["mean_mse", "mean_r2"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "probe_model": probe_model,
                "best_mean_r2": {
                    "feature_set": str(best_r2["feature_set"]),
                    "target": str(best_r2["target"]),
                    "value": float(best_r2["mean_r2"]),
                },
                "best_mean_mse": {
                    "feature_set": str(best_mse["feature_set"]),
                    "target": str(best_mse["target"]),
                    "value": float(best_mse["mean_mse"]),
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if not 0.0 < args.probe_test_size < 1.0:
        raise ValueError("--probe-test-size must be in (0, 1).")
    if not 0.0 < args.probe_val_size < 1.0:
        raise ValueError("--probe-val-size must be in (0, 1).")
    if args.probe_test_size + args.probe_val_size >= 1.0:
        raise ValueError("--probe-test-size + --probe-val-size must be < 1.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    export_manifest = load_export_manifest(dataset_root)
    checkpoint_meta = infer_checkpoint_metadata(
        None
        if export_manifest is None
        else export_manifest.get("source_checkpoint_path") or export_manifest.get("policy_path")
    )
    info = load_info(dataset_root)
    dataset = make_dataset(dataset_root)
    valid_col = f"{args.feature_prefix}.valid"
    continuous_col = pick_feature_name(info, args.feature_prefix, "continuous_vector_latents", "continuous")
    codebook_vectors_col = pick_feature_name(
        info,
        args.feature_prefix,
        "codebook_vector_latents",
        "codebook_vectors",
        required=False,
    )
    ids_col = pick_feature_name(info, args.feature_prefix, "codebook_id_latents", "codebook_ids", required=False)
    for column_name in [valid_col, "action", continuous_col]:
        if column_name not in info["features"]:
            raise KeyError(f"Missing feature in dataset metadata: {column_name}")

    valid_counts = load_valid_counts(dataset, valid_col)
    all_actions, all_valid, episode_index = load_action_context(dataset, "action", valid_col)
    action_targets = derive_action_targets(
        all_actions,
        all_valid,
        episode_index,
        args.future_frames,
        future_target_config=args.future_target_config,
    )
    future_target_names = [name for name in action_targets if name != "current_action"]
    if len(future_target_names) != 1:
        raise ValueError(f"Expected exactly one future target, got {future_target_names}.")
    future_target_name = future_target_names[0]
    valid_episode_index = extract_valid_episode_index(all_valid, episode_index)

    feature_sets: dict[str, np.ndarray] = {}
    continuous = load_float_array(dataset, continuous_col, valid_col)
    if "continuous" in args.probe_feature_sets:
        feature_sets["continuous"] = flatten_valid_latents(continuous)
    codebook_vectors = None
    if "codebook_vectors" in args.probe_feature_sets:
        if codebook_vectors_col is None:
            raise ValueError("Requested codebook_vectors, but this export does not contain codebook vector latents.")
        codebook_vectors = load_float_array(dataset, codebook_vectors_col, valid_col)
        feature_sets["codebook_vectors"] = flatten_valid_latents(codebook_vectors)
    ids = None
    if "ids_onehot" in args.probe_feature_sets:
        if ids_col is None:
            raise ValueError("Requested ids_onehot, but this export does not contain codebook ID latents.")
        ids = load_ids(dataset, ids_col, valid_col)
        feature_sets["ids_onehot"] = ids

    valid_frames = int(continuous.shape[0])
    if valid_episode_index.shape[0] != valid_frames:
        raise ValueError("Valid action targets and latent arrays disagree on row count.")
    for feature_name, values in feature_sets.items():
        if values.shape[0] != valid_frames:
            raise ValueError(f"{feature_name} has {values.shape[0]} rows but expected {valid_frames}.")
    for target_name, values in action_targets.items():
        if values.shape[0] != valid_frames:
            raise ValueError(f"{target_name} has {values.shape[0]} rows but expected {valid_frames}.")

    split = make_probe_split(
        valid_episode_index=valid_episode_index,
        max_samples=args.probe_max_samples,
        test_size=args.probe_test_size,
        val_size=args.probe_val_size,
        seed=args.seed,
        mode=args.probe_split,
    )
    probe_df = run_probes(feature_sets=feature_sets, targets=action_targets, split=split, args=args, device=device)
    probe_summary_df = summarize_probe_scores(probe_df)
    probe_df.to_csv(output_dir / "action_probe_scores.csv", index=False)
    probe_df.to_csv(output_dir / "action_probe_r2.csv", index=False)
    probe_summary_df.to_csv(output_dir / "action_probe_scores_summary.csv", index=False)
    probe_summary_df.to_csv(output_dir / "action_probe_r2_summary.csv", index=False)
    best_by_probe_model = best_probe_rows(probe_summary_df)

    target_config = validate_future_target_config(
        default_future_target_config(int(all_actions.shape[1])) if args.future_target_config is None else args.future_target_config,
        action_dim=int(all_actions.shape[1]),
    )
    tail_counts = infer_episode_tail_counts(all_valid, episode_index)
    summary = {
        "dataset_root": str(dataset_root),
        "feature_prefix": args.feature_prefix,
        "analysis_kind": "latent_action_probes",
        "future_frames": args.future_frames,
        "future_target_name": future_target_name,
        "future_target_config": target_config,
        "device": str(device),
        "total_frames": int(sum(valid_counts.values())),
        "valid_counts": valid_counts,
        "valid_frames": valid_frames,
        "invalid_frames": int(sum(v for k, v in valid_counts.items() if int(k) != 1)),
        "episode_invalid_tail_counts": {
            "min": int(np.min(tail_counts)),
            "median": float(np.median(tail_counts)),
            "max": int(np.max(tail_counts)),
            "unique_values": sorted({int(v) for v in tail_counts}),
        },
        "features": {
            "requested": sorted(args.probe_feature_sets),
            "available": {
                "continuous": continuous_col is not None,
                "codebook_vectors": codebook_vectors_col is not None,
                "ids_onehot": ids_col is not None,
            },
            "continuous": value_summary(continuous),
            "codebook_vectors": None if codebook_vectors is None else value_summary(codebook_vectors),
            "ids": None if ids is None else {"shape": list(ids.shape), "unique_rows": int(np.unique(ids, axis=0).shape[0])},
        },
        "split": {
            "mode": args.probe_split,
            "test_size": args.probe_test_size,
            "val_size": args.probe_val_size,
            "max_samples": args.probe_max_samples,
            "n_train": int(split.train_rows.shape[0]),
            "n_val": int(split.val_rows.shape[0]),
            "n_test": int(split.test_rows.shape[0]),
        },
        "action_probes": {
            "probe_model": args.probe_model,
            "probe_feature_sets": sorted(args.probe_feature_sets),
            "ridge_alpha": args.ridge_alpha,
            "mlp": {
                "hidden_dims": list(args.probe_mlp_hidden_dims),
                "alpha": args.probe_mlp_alpha,
                "max_epochs": args.probe_mlp_max_iter,
                "batch_size": args.probe_mlp_batch_size,
                "learning_rate": args.probe_mlp_lr,
                "early_stopping": args.probe_mlp_early_stopping,
                "patience": args.probe_mlp_n_iter_no_change,
            },
            "mean_scores_by_feature_and_target": probe_summary_df.to_dict(orient="records"),
            "best_by_probe_model": best_by_probe_model,
        },
        "artifacts": sorted(p.name for p in output_dir.iterdir()),
    }
    save_json(output_dir / "summary.json", summary)

    readme_lines = [
        "# Latent Action Probe Analysis",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Feature prefix: `{args.feature_prefix}`",
        f"- Future frames: `{args.future_frames}`",
        f"- Future target: `{future_target_name}`",
        f"- Device: `{device}`",
        f"- Valid frames: `{valid_frames}`",
        f"- Split: `{args.probe_split}`, train `{len(split.train_rows)}`, val `{len(split.val_rows)}`, test `{len(split.test_rows)}`",
        f"- Feature sets: `{', '.join(sorted(args.probe_feature_sets))}`",
        "",
        "## Best Probe Scores",
    ]
    for best in best_by_probe_model:
        readme_lines.append(
            f"- `{best['probe_model']}` best R^2: `{best['best_mean_r2']['feature_set']}` -> "
            f"`{best['best_mean_r2']['target']}` = `{best['best_mean_r2']['value']:.4f}`"
        )
        readme_lines.append(
            f"- `{best['probe_model']}` best MSE: `{best['best_mean_mse']['feature_set']}` -> "
            f"`{best['best_mean_mse']['target']}` = `{best['best_mean_mse']['value']:.6f}`"
        )
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines) + "\n")

    def probe_metric(feature_set: str, target: str, metric: str, model: str) -> float | None:
        rows = probe_summary_df[
            (probe_summary_df["probe_model"] == model)
            & (probe_summary_df["feature_set"] == feature_set)
            & (probe_summary_df["target"] == target)
        ]
        return None if rows.shape[0] == 0 else float(rows.iloc[0][metric])

    analysis_manifest = {
        "artifact_type": "latent_analysis",
        "analysis_kind": "latent_action_probes",
        "suite_name": "latent_action_probes",
        "suite_version": "gpu_v1",
        "artifact_id": make_artifact_id(
            suite_name="latent_action_probes",
            suite_version="gpu_v1",
            checkpoint_id=checkpoint_meta["source_checkpoint_id"],
            output_label=output_dir.name,
        ),
        **checkpoint_meta,
        "parent_export_artifact_id": None if export_manifest is None else export_manifest.get("artifact_id"),
        "parent_export_manifest_path": None if export_manifest is None else export_manifest.get("manifest_path"),
        "input_dataset_root": str(dataset_root),
        "input_dataset_repo_id": None if export_manifest is None else export_manifest.get("output_repo_id"),
        "script_path": str(Path(__file__).resolve()),
        "cli_args": list(sys.argv[1:]),
        "feature_prefix": args.feature_prefix,
        "output_path": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "readme_path": str(readme_path),
        "headline_metrics": {
            "probe_split": args.probe_split,
            "probe_model": args.probe_model,
            "probe_feature_sets": sorted(args.probe_feature_sets),
            "future_target_name": future_target_name,
            "continuous_current_ridge_mean_r2": probe_metric("continuous", "current_action", "mean_r2", "ridge"),
            "continuous_current_mlp_mean_r2": probe_metric("continuous", "current_action", "mean_r2", "mlp"),
            "continuous_future_ridge_mean_r2": probe_metric("continuous", future_target_name, "mean_r2", "ridge"),
            "continuous_future_mlp_mean_r2": probe_metric("continuous", future_target_name, "mean_r2", "mlp"),
        },
    }
    register_artifact(
        manifest_path=output_dir / "analysis_manifest.json",
        manifest=analysis_manifest,
        registry_candidates=[output_dir, dataset_root, checkpoint_meta["source_checkpoint_path"]],
    )


if __name__ == "__main__":
    main()
