#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mutual_info_score, normalized_mutual_info_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _artifact_registry import infer_checkpoint_metadata, load_export_manifest, make_artifact_id, register_artifact


def parse_hidden_dims(raw: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(
            "--probe-mlp-hidden-dims must be a comma-separated list of positive integers."
        )
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze latent feature distributions for a labeled LeRobot dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root of the labeled dataset.")
    parser.add_argument("--feature-prefix", type=str, required=True, help="Feature prefix, e.g. latent_labels.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where plots and tables will be written.")
    parser.add_argument(
        "--top-k-sequences",
        type=int,
        default=50,
        help="How many top codebook ID sequences to save and plot.",
    )
    parser.add_argument(
        "--scatter-points",
        type=int,
        default=100000,
        help="Maximum number of points to render in each PCA scatter plot.",
    )
    parser.add_argument(
        "--pca-fit-points",
        type=int,
        default=50000,
        help="Maximum number of rows used to fit each PCA projection.",
    )
    parser.add_argument(
        "--rounded-decimals",
        type=int,
        default=3,
        help="Decimal precision used for approximate uniqueness on float features.",
    )
    parser.add_argument(
        "--future-frames",
        type=int,
        default=10,
        help="Future horizon used for derived action summaries such as future_action_mean.",
    )
    parser.add_argument(
        "--action-bins",
        type=int,
        default=16,
        help="Quantile bins per action dimension for mutual information estimates.",
    )
    parser.add_argument(
        "--probe-max-samples",
        type=int,
        default=100000,
        help="Maximum number of valid rows to use in held-out action probes.",
    )
    parser.add_argument(
        "--probe-test-size",
        type=float,
        default=0.2,
        help="Held-out fraction for action probes.",
    )
    parser.add_argument(
        "--probe-model",
        choices=("ridge", "mlp", "both"),
        default="ridge",
        help="Probe backend used for action prediction.",
    )
    parser.add_argument(
        "--probe-split",
        choices=("row", "episode"),
        default="row",
        help="How to split data into train/test for the action probes.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization used for linear action probes.",
    )
    parser.add_argument(
        "--probe-mlp-hidden-dims",
        type=parse_hidden_dims,
        default=(512, 256),
        help="Comma-separated hidden sizes for the MLP action probe.",
    )
    parser.add_argument(
        "--probe-mlp-alpha",
        type=float,
        default=1e-4,
        help="L2 regularization strength for the MLP action probe.",
    )
    parser.add_argument(
        "--probe-mlp-max-iter",
        type=int,
        default=200,
        help="Maximum optimization steps for the MLP action probe.",
    )
    parser.add_argument(
        "--probe-mlp-early-stopping",
        type=lambda raw: raw.lower() in {"1", "true", "yes", "on"},
        default=True,
        help="Whether the MLP action probe should use validation-based early stopping.",
    )
    parser.add_argument(
        "--probe-mlp-n-iter-no-change",
        type=int,
        default=10,
        help="Patience used by the MLP action probe when early stopping is enabled.",
    )
    parser.add_argument(
        "--bucket-kmeans-clusters",
        type=int,
        default=128,
        help="Number of KMeans buckets used for continuous latents. Set to 0 to disable continuous bucketing.",
    )
    parser.add_argument(
        "--bucket-kmeans-fit-samples",
        type=int,
        default=50000,
        help="Maximum number of rows used to fit the continuous KMeans bucketing model.",
    )
    parser.add_argument(
        "--bucket-top-k",
        type=int,
        default=20,
        help="How many top buckets by count to keep in the JSON and README summaries.",
    )
    parser.add_argument(
        "--bucket-progress-bins",
        type=int,
        default=10,
        help="Number of normalized within-episode progress bins used for bucket context coverage.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    return json.loads(info_path.read_text())


def make_dataset(dataset_root: Path) -> ds.Dataset:
    return ds.dataset(dataset_root / "data", format="parquet")


def load_valid_counts(dataset: ds.Dataset, valid_col: str) -> dict[str, int]:
    table = dataset.to_table(columns=[valid_col])
    counts = {}
    for item in pc.value_counts(table[valid_col]).to_pylist():
        counts[str(item["values"])] = int(item["counts"])
    return counts


def load_ids(dataset: ds.Dataset, ids_col: str, valid_col: str) -> np.ndarray:
    table = dataset.to_table(columns=[ids_col], filter=ds.field(valid_col) == 1)
    obj = table[ids_col].to_numpy(zero_copy_only=False)
    ids = np.stack(obj).astype(np.int64, copy=False)
    return ensure_2d_rows(ids)


def load_float_array(dataset: ds.Dataset, column_name: str, valid_col: str) -> np.ndarray:
    table = dataset.to_table(columns=[column_name], filter=ds.field(valid_col) == 1)
    obj = table[column_name].to_numpy(zero_copy_only=False)
    return np.stack([np.stack(row, axis=0) for row in obj], axis=0).astype(np.float32, copy=False)


def load_action_context(
    dataset: ds.Dataset, action_col: str, valid_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = dataset.to_table(columns=[action_col, valid_col, "episode_index", "frame_index"])
    actions = np.stack(table[action_col].to_numpy(zero_copy_only=False)).astype(np.float32, copy=False)
    valid = table[valid_col].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    episode_index = table["episode_index"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    frame_index = table["frame_index"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return actions, valid, episode_index, frame_index


def ensure_2d_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D row array, got shape={arr.shape}.")
    return arr


def ensure_slot_tensor(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D slot tensor, got shape={arr.shape}.")
    return arr


def contiguous_row_view(arr_2d: np.ndarray) -> np.ndarray:
    arr_2d = ensure_2d_rows(arr_2d)
    arr_2d = np.ascontiguousarray(arr_2d)
    dtype = np.dtype((np.void, arr_2d.dtype.itemsize * arr_2d.shape[1]))
    return arr_2d.view(dtype).reshape(-1)


def unique_rows(arr_2d: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    row_view = contiguous_row_view(arr_2d)
    _, inverse, counts = np.unique(row_view, return_inverse=True, return_counts=True)
    return int(counts.shape[0]), inverse, counts


def summarize_numeric(values: np.ndarray) -> dict[str, float]:
    flat = values.reshape(-1).astype(np.float64, copy=False)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "p01": float(np.quantile(flat, 0.01)),
        "p05": float(np.quantile(flat, 0.05)),
        "median": float(np.median(flat)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
    }


def summarize_norms(values: np.ndarray) -> pd.DataFrame:
    values = ensure_slot_tensor(values)
    slot_norms = np.linalg.norm(values, axis=2)
    rows = []
    for slot_idx in range(slot_norms.shape[1]):
        slot_values = slot_norms[:, slot_idx].astype(np.float64, copy=False)
        rows.append(
            {
                "slot_index": slot_idx,
                "mean": float(np.mean(slot_values)),
                "std": float(np.std(slot_values)),
                "min": float(np.min(slot_values)),
                "p01": float(np.quantile(slot_values, 0.01)),
                "p05": float(np.quantile(slot_values, 0.05)),
                "median": float(np.median(slot_values)),
                "p95": float(np.quantile(slot_values, 0.95)),
                "p99": float(np.quantile(slot_values, 0.99)),
                "max": float(np.max(slot_values)),
            }
        )
    return pd.DataFrame(rows)


def format_sequence(seq: np.ndarray) -> str:
    return " ".join(str(int(v)) for v in seq.tolist())


def episode_ranges(episode_index: np.ndarray) -> list[tuple[int, int]]:
    split_points = np.flatnonzero(np.diff(episode_index)) + 1
    starts = np.concatenate(([0], split_points))
    ends = np.concatenate((split_points, [len(episode_index)]))
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def infer_episode_tail_counts(valid: np.ndarray, episode_index: np.ndarray) -> list[int]:
    tails = []
    for start, end in episode_ranges(episode_index):
        valid_ep = valid[start:end]
        tails.append(int(np.sum(valid_ep == 0)))
    return tails


def derive_action_targets(
    actions: np.ndarray,
    valid: np.ndarray,
    episode_index: np.ndarray,
    future_frames: int,
) -> dict[str, np.ndarray]:
    current_chunks = []
    future_mean_chunks = []

    for start, end in episode_ranges(episode_index):
        actions_ep = actions[start:end]
        valid_ep = valid[start:end]
        n_valid = int(np.sum(valid_ep == 1))
        if not np.all(valid_ep[:n_valid] == 1):
            raise ValueError("Expected valid rows to be contiguous at the start of each episode.")
        if not np.all(valid_ep[n_valid:] == 0):
            raise ValueError("Expected invalid rows to be contiguous at the end of each episode.")
        if n_valid == 0:
            continue
        if future_frames <= 0:
            raise ValueError("--future-frames must be positive.")
        if actions_ep.shape[0] < n_valid + future_frames:
            raise ValueError(
                f"Episode has length {actions_ep.shape[0]}, which is too short for n_valid={n_valid} and future_frames={future_frames}."
            )

        current_chunks.append(actions_ep[:n_valid])
        cumsum = np.vstack([np.zeros((1, actions_ep.shape[1]), dtype=np.float32), np.cumsum(actions_ep, axis=0)])
        future_sum = cumsum[1 + future_frames :] - cumsum[1:-future_frames]
        future_mean = (future_sum / float(future_frames)).astype(np.float32, copy=False)
        if future_mean.shape[0] < n_valid:
            raise ValueError(
                f"Derived future action target is too short for episode with n_valid={n_valid} and future_frames={future_frames}."
            )
        future_mean_chunks.append(future_mean[:n_valid])

    return {
        "current_action": np.concatenate(current_chunks, axis=0),
        "future_action_mean": np.concatenate(future_mean_chunks, axis=0),
    }


def extract_valid_episode_index(valid: np.ndarray, episode_index: np.ndarray) -> np.ndarray:
    chunks = []
    for start, end in episode_ranges(episode_index):
        valid_ep = valid[start:end]
        n_valid = int(np.sum(valid_ep == 1))
        if not np.all(valid_ep[:n_valid] == 1):
            raise ValueError("Expected valid rows to be contiguous at the start of each episode.")
        if not np.all(valid_ep[n_valid:] == 0):
            raise ValueError("Expected invalid rows to be contiguous at the end of each episode.")
        if n_valid == 0:
            continue
        chunks.append(episode_index[start : start + n_valid])
    if not chunks:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(chunks, axis=0).astype(np.int64, copy=False)


def extract_valid_scalar_context(values: np.ndarray, valid: np.ndarray, episode_index: np.ndarray) -> np.ndarray:
    chunks = []
    for start, end in episode_ranges(episode_index):
        valid_ep = valid[start:end]
        n_valid = int(np.sum(valid_ep == 1))
        if not np.all(valid_ep[:n_valid] == 1):
            raise ValueError("Expected valid rows to be contiguous at the start of each episode.")
        if not np.all(valid_ep[n_valid:] == 0):
            raise ValueError("Expected invalid rows to be contiguous at the end of each episode.")
        if n_valid == 0:
            continue
        chunks.append(values[start : start + n_valid])
    if not chunks:
        return np.empty((0,), dtype=values.dtype)
    return np.concatenate(chunks, axis=0)


def extract_valid_progress_context(valid: np.ndarray, episode_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    progress_idx_chunks = []
    episode_len_chunks = []
    for start, end in episode_ranges(episode_index):
        valid_ep = valid[start:end]
        n_valid = int(np.sum(valid_ep == 1))
        if not np.all(valid_ep[:n_valid] == 1):
            raise ValueError("Expected valid rows to be contiguous at the start of each episode.")
        if not np.all(valid_ep[n_valid:] == 0):
            raise ValueError("Expected invalid rows to be contiguous at the end of each episode.")
        if n_valid == 0:
            continue
        progress_idx_chunks.append(np.arange(n_valid, dtype=np.int64))
        episode_len_chunks.append(np.full(n_valid, n_valid, dtype=np.int64))
    if not progress_idx_chunks:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return (
        np.concatenate(progress_idx_chunks, axis=0).astype(np.int64, copy=False),
        np.concatenate(episode_len_chunks, axis=0).astype(np.int64, copy=False),
    )


def quantile_bin_1d(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.shape[0] <= 2:
        return np.zeros(values.shape[0], dtype=np.int64), edges
    binned = np.digitize(values, edges[1:-1], right=False).astype(np.int64, copy=False)
    return binned, edges


def quantile_bin_targets(targets: dict[str, np.ndarray], n_bins: int) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    binned_targets = {}
    bin_counts = {}
    for target_name, values in targets.items():
        binned = np.zeros(values.shape, dtype=np.int64)
        counts = []
        for dim in range(values.shape[1]):
            binned[:, dim], edges = quantile_bin_1d(values[:, dim], n_bins)
            counts.append(int(max(len(edges) - 1, 1)))
        binned_targets[target_name] = binned
        bin_counts[target_name] = counts
    return binned_targets, bin_counts


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def plot_valid_distribution(valid_counts: dict[str, int], output_path: Path) -> None:
    labels = sorted(valid_counts.keys(), key=int)
    values = [valid_counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#4c78a8", "#f58518"][: len(labels)])
    ax.set_title("Valid Flag Distribution")
    ax.set_xlabel("valid")
    ax.set_ylabel("frames")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_top_sequences(df: pd.DataFrame, output_path: Path) -> None:
    top = df.head(20).iloc[::-1]
    fig_height = max(5, 0.35 * len(top))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(top["sequence"], top["count"], color="#4c78a8")
    ax.set_title("Top Codebook ID Sequences")
    ax.set_xlabel("count")
    ax.set_ylabel("sequence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_id_position_counts(ids: np.ndarray, output_path: Path) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for pos in range(ids.shape[1]):
        ax = axes.flat[pos]
        values, counts = np.unique(ids[:, pos], return_counts=True)
        order = np.argsort(counts)[::-1]
        values = values[order]
        counts = counts[order]
        for value, count in zip(values.tolist(), counts.tolist(), strict=True):
            rows.append({"position": pos, "token_id": int(value), "count": int(count)})

        display_k = min(20, len(values))
        ax.bar([str(int(v)) for v in values[:display_k]], counts[:display_k], color="#f58518")
        ax.set_title(f"Position {pos} Top IDs")
        ax.set_xlabel("token id")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Codebook ID Usage by Position", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_value_histogram(values: np.ndarray, title: str, output_path: Path) -> None:
    flat = values.reshape(-1).astype(np.float64, copy=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flat, bins=120, color="#4c78a8", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_slot_norms(values: np.ndarray, title: str, output_path: Path) -> None:
    values = ensure_slot_tensor(values)
    norms = np.linalg.norm(values, axis=2)
    fig, ax = plt.subplots(figsize=(9, 5))
    for slot_idx in range(norms.shape[1]):
        ax.hist(norms[:, slot_idx], bins=100, alpha=0.45, label=f"slot {slot_idx}")
    ax.set_title(title)
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    fig_width = max(7, 1.1 * len(col_labels))
    fig_height = max(5, 0.5 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_pca_scatter(
    flat_values: np.ndarray,
    usage_counts: np.ndarray,
    title: str,
    output_path: Path,
    csv_path: Path,
    rng: np.random.Generator,
    fit_points: int,
    scatter_points: int,
    colorbar_label: str = "log10(ID sequence usage)",
) -> None:
    n_rows = flat_values.shape[0]
    fit_idx = rng.choice(n_rows, size=min(fit_points, n_rows), replace=False)
    plot_idx = rng.choice(n_rows, size=min(scatter_points, n_rows), replace=False)

    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    pca.fit(flat_values[fit_idx])
    coords = pca.transform(flat_values[plot_idx])
    plot_usage = usage_counts[plot_idx]
    log_usage = np.log10(plot_usage.astype(np.float64))

    sample_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "usage_count": plot_usage,
            "log10_usage_count": log_usage,
        }
    )
    sample_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=log_usage,
        s=7,
        cmap="viridis",
        alpha=0.7,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def compute_action_mutual_information(
    ids: np.ndarray,
    id_sequence_labels: np.ndarray,
    binned_targets: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    feature_map = {"id_sequence": id_sequence_labels}
    for pos in range(ids.shape[1]):
        feature_map[f"id_pos{pos}"] = ids[:, pos]

    for target_name, target_values in binned_targets.items():
        for action_dim in range(target_values.shape[1]):
            target = target_values[:, action_dim]
            for feature_name, feature_values in feature_map.items():
                rows.append(
                    {
                        "target": target_name,
                        "action_dim": action_dim,
                        "feature": feature_name,
                        "mi": float(mutual_info_score(feature_values, target)),
                        "nmi": float(normalized_mutual_info_score(feature_values, target)),
                    }
                )
    df = pd.DataFrame(rows)
    ranking = df.sort_values(["mi", "nmi"], ascending=[False, False]).reset_index(drop=True)
    return df, ranking


def group_rows_by_bucket(bucket_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bucket_index.ndim != 1:
        raise ValueError(f"Expected 1D bucket index array, got shape={bucket_index.shape}.")
    if bucket_index.shape[0] == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    order = np.argsort(bucket_index, kind="stable")
    sorted_bucket_index = bucket_index[order]
    split_points = np.flatnonzero(np.diff(sorted_bucket_index)) + 1
    starts = np.concatenate(([0], split_points)).astype(np.int64, copy=False)
    ends = np.concatenate((split_points, [sorted_bucket_index.shape[0]])).astype(np.int64, copy=False)
    bucket_ids = sorted_bucket_index[starts].astype(np.int64, copy=False)
    return order.astype(np.int64, copy=False), bucket_ids, starts, ends


def compute_bucket_action_statistics(
    *,
    bucket_index: np.ndarray,
    bucket_names: np.ndarray,
    target_values: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if bucket_index.shape[0] != target_values.shape[0]:
        raise ValueError("Bucket indices and action targets must have the same number of rows.")
    if bucket_names.ndim != 1:
        raise ValueError(f"Expected 1D bucket names array, got shape={bucket_names.shape}.")

    total_rows = int(bucket_index.shape[0])
    counts = np.bincount(bucket_index, minlength=int(bucket_names.shape[0]))
    order, active_bucket_ids, starts, ends = group_rows_by_bucket(bucket_index)
    sorted_targets = target_values[order].astype(np.float64, copy=False)
    global_var = np.var(target_values.astype(np.float64, copy=False), axis=0)
    within_var_numer = np.zeros(target_values.shape[1], dtype=np.float64)
    weighted_std_sum = np.zeros(target_values.shape[1], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for bucket_id, start, end in zip(active_bucket_ids.tolist(), starts.tolist(), ends.tolist(), strict=True):
        bucket_target = sorted_targets[start:end]
        count = int(end - start)
        fraction = float(count / total_rows)
        mean = np.mean(bucket_target, axis=0)
        std = np.std(bucket_target, axis=0)
        var = np.var(bucket_target, axis=0)
        within_var_numer += var * count
        weighted_std_sum += std * count

        row: dict[str, Any] = {
            "bucket_id": int(bucket_id),
            "bucket_label": str(bucket_names[bucket_id]),
            "count": count,
            "fraction": fraction,
        }
        for action_dim, value in enumerate(mean.tolist()):
            row[f"mean_a{action_dim}"] = float(value)
        for action_dim, value in enumerate(std.tolist()):
            row[f"std_a{action_dim}"] = float(value)
        rows.append(row)

    stats_df = pd.DataFrame(rows).sort_values(["count", "bucket_id"], ascending=[False, True], ignore_index=True)
    within_var = within_var_numer / float(max(total_rows, 1))
    variance_explained = np.where(global_var > 1e-12, 1.0 - (within_var / global_var), 0.0)
    weighted_mean_std = weighted_std_sum / float(max(total_rows, 1))

    summary = {
        "total_rows": total_rows,
        "total_buckets": int(bucket_names.shape[0]),
        "active_buckets": int(active_bucket_ids.shape[0]),
        "singleton_buckets": int(np.sum(counts == 1)),
        "max_bucket_usage": int(np.max(counts)) if counts.shape[0] > 0 else 0,
        "max_bucket_fraction": float(np.max(counts) / total_rows) if total_rows > 0 and counts.shape[0] > 0 else 0.0,
        "mean_variance_explained": float(np.mean(variance_explained)),
        "variance_explained_by_dim": [float(value) for value in variance_explained.tolist()],
        "mean_within_bucket_std": float(np.mean(weighted_mean_std)),
        "within_bucket_std_by_dim": [float(value) for value in weighted_mean_std.tolist()],
    }
    return stats_df, summary


def _normalized_progress_bins(
    progress_index: np.ndarray,
    episode_lengths: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    if progress_index.shape != episode_lengths.shape:
        raise ValueError("Progress indices and episode lengths must have the same shape.")
    if n_bins < 1:
        raise ValueError("Progress bins must be >= 1.")
    if progress_index.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    denom = np.maximum(episode_lengths.astype(np.float64, copy=False) - 1.0, 1.0)
    normalized = progress_index.astype(np.float64, copy=False) / denom
    bins = np.floor(normalized * float(n_bins)).astype(np.int64, copy=False)
    return np.clip(bins, 0, n_bins - 1)


def compute_bucket_context_statistics(
    *,
    bucket_index: np.ndarray,
    bucket_names: np.ndarray,
    valid_episode_index: np.ndarray,
    valid_frame_index: np.ndarray,
    valid_progress_index: np.ndarray,
    valid_episode_lengths: np.ndarray,
    progress_bins: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if bucket_index.shape[0] != valid_episode_index.shape[0]:
        raise ValueError("Bucket indices and episode indices must have the same number of rows.")
    if valid_frame_index.shape[0] != bucket_index.shape[0]:
        raise ValueError("Bucket indices and frame indices must have the same number of rows.")
    if valid_progress_index.shape[0] != bucket_index.shape[0]:
        raise ValueError("Bucket indices and progress indices must have the same number of rows.")
    if valid_episode_lengths.shape[0] != bucket_index.shape[0]:
        raise ValueError("Bucket indices and episode lengths must have the same number of rows.")
    if bucket_names.ndim != 1:
        raise ValueError(f"Expected 1D bucket names array, got shape={bucket_names.shape}.")

    total_rows = int(bucket_index.shape[0])
    if total_rows == 0:
        empty_df = pd.DataFrame(
            columns=[
                "bucket_id",
                "bucket_label",
                "count",
                "fraction",
                "unique_episodes",
                "episode_coverage",
                "max_episode_fraction",
                "episode_perplexity",
                "mean_run_length",
                "max_run_length",
                "adjacent_repeat_fraction",
                "progress_bin_coverage",
            ]
        )
        return empty_df, {
            "total_rows": 0,
            "total_buckets": int(bucket_names.shape[0]),
            "active_buckets": 0,
            "weighted_mean_episode_coverage": 0.0,
            "weighted_mean_max_episode_fraction": 0.0,
            "weighted_mean_episode_perplexity": 0.0,
            "weighted_mean_run_length": 0.0,
            "weighted_max_run_length": 0.0,
            "weighted_adjacent_repeat_fraction": 0.0,
            "weighted_progress_bin_coverage": 0.0,
            "bucket_episode_nmi": 0.0,
        }

    total_episodes = int(np.unique(valid_episode_index).shape[0])
    progress_bin_index = _normalized_progress_bins(valid_progress_index, valid_episode_lengths, n_bins=progress_bins)
    counts = np.bincount(bucket_index, minlength=int(bucket_names.shape[0]))
    bucket_episode_nmi = float(normalized_mutual_info_score(bucket_index, valid_episode_index))

    weighted_episode_coverage = 0.0
    weighted_max_episode_fraction = 0.0
    weighted_episode_perplexity = 0.0
    weighted_mean_run_length = 0.0
    weighted_max_run_length = 0.0
    weighted_adjacent_repeat_fraction = 0.0
    weighted_progress_bin_coverage = 0.0
    rows: list[dict[str, Any]] = []

    for bucket_id in np.flatnonzero(counts > 0).tolist():
        bucket_rows = np.flatnonzero(bucket_index == bucket_id)
        count = int(bucket_rows.shape[0])
        fraction = float(count / total_rows)
        bucket_eps = valid_episode_index[bucket_rows]
        bucket_frames = valid_frame_index[bucket_rows]
        bucket_progress_bins = progress_bin_index[bucket_rows]

        unique_eps, episode_counts = np.unique(bucket_eps, return_counts=True)
        probs = episode_counts.astype(np.float64, copy=False) / float(count)
        entropy = -np.sum(np.where(probs > 0.0, probs * np.log(probs), 0.0))
        episode_perplexity = float(np.exp(entropy))
        max_episode_fraction = float(np.max(probs))
        episode_coverage = float(unique_eps.shape[0] / max(total_episodes, 1))

        run_lengths = []
        run_start = 0
        for idx in range(1, count):
            same_episode = bucket_eps[idx] == bucket_eps[idx - 1]
            consecutive_frame = bucket_frames[idx] == bucket_frames[idx - 1] + 1
            if not (same_episode and consecutive_frame):
                run_lengths.append(idx - run_start)
                run_start = idx
        run_lengths.append(count - run_start)
        run_lengths_arr = np.asarray(run_lengths, dtype=np.int64)
        adjacent_repeat_fraction = float(np.sum(np.maximum(run_lengths_arr - 1, 0)) / float(max(count - 1, 1)))
        mean_run_length = float(np.mean(run_lengths_arr))
        max_run_length = int(np.max(run_lengths_arr))
        progress_bin_coverage = float(np.unique(bucket_progress_bins).shape[0] / float(max(progress_bins, 1)))

        weighted_episode_coverage += episode_coverage * count
        weighted_max_episode_fraction += max_episode_fraction * count
        weighted_episode_perplexity += episode_perplexity * count
        weighted_mean_run_length += mean_run_length * count
        weighted_max_run_length += float(max_run_length) * count
        weighted_adjacent_repeat_fraction += adjacent_repeat_fraction * count
        weighted_progress_bin_coverage += progress_bin_coverage * count

        rows.append(
            {
                "bucket_id": int(bucket_id),
                "bucket_label": str(bucket_names[bucket_id]),
                "count": count,
                "fraction": fraction,
                "unique_episodes": int(unique_eps.shape[0]),
                "episode_coverage": episode_coverage,
                "max_episode_fraction": max_episode_fraction,
                "episode_perplexity": episode_perplexity,
                "mean_run_length": mean_run_length,
                "max_run_length": max_run_length,
                "adjacent_repeat_fraction": adjacent_repeat_fraction,
                "progress_bin_coverage": progress_bin_coverage,
            }
        )

    stats_df = pd.DataFrame(rows).sort_values(
        ["count", "episode_coverage", "max_episode_fraction"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    summary = {
        "total_rows": total_rows,
        "total_buckets": int(bucket_names.shape[0]),
        "active_buckets": int(np.sum(counts > 0)),
        "weighted_mean_episode_coverage": float(weighted_episode_coverage / total_rows),
        "weighted_mean_max_episode_fraction": float(weighted_max_episode_fraction / total_rows),
        "weighted_mean_episode_perplexity": float(weighted_episode_perplexity / total_rows),
        "weighted_mean_run_length": float(weighted_mean_run_length / total_rows),
        "weighted_max_run_length": float(weighted_max_run_length / total_rows),
        "weighted_adjacent_repeat_fraction": float(weighted_adjacent_repeat_fraction / total_rows),
        "weighted_progress_bin_coverage": float(weighted_progress_bin_coverage / total_rows),
        "bucket_episode_nmi": bucket_episode_nmi,
        "progress_bins": int(progress_bins),
        "total_episodes": total_episodes,
    }
    return stats_df, summary


def make_discrete_bucket_spec(ids: np.ndarray, id_inverse: np.ndarray) -> dict[str, Any]:
    unique_labels, first_indices = np.unique(id_inverse, return_index=True)
    if unique_labels.shape[0] == 0:
        raise ValueError("Expected at least one discrete bucket.")
    if not np.array_equal(unique_labels, np.arange(unique_labels.shape[0], dtype=np.int64)):
        raise ValueError("Discrete ID inverse labels must be dense and zero-based.")

    bucket_names = np.asarray([format_sequence(seq) for seq in ids[first_indices]], dtype=object)
    counts = np.bincount(id_inverse, minlength=bucket_names.shape[0]).astype(np.int64, copy=False)
    return {
        "feature_set": "id_sequence",
        "bucket_kind": "discrete_sequence",
        "bucket_index": id_inverse.astype(np.int64, copy=False),
        "bucket_names": bucket_names,
        "bucket_counts": counts,
    }


def make_continuous_bucket_spec(
    continuous_flat: np.ndarray,
    *,
    n_clusters: int,
    fit_samples: int,
    seed: int,
) -> dict[str, Any]:
    if n_clusters < 1:
        raise ValueError("Continuous bucketing requires at least one cluster.")
    if fit_samples < 1:
        raise ValueError("Continuous bucketing requires at least one fit sample.")

    n_rows = continuous_flat.shape[0]
    effective_clusters = min(int(n_clusters), int(n_rows))
    rng = np.random.default_rng(seed)
    fit_size = min(int(fit_samples), int(n_rows))
    fit_idx = rng.choice(n_rows, size=fit_size, replace=False)

    scaler = StandardScaler()
    fit_values = scaler.fit_transform(continuous_flat[fit_idx]).astype(np.float32, copy=False)
    full_values = scaler.transform(continuous_flat).astype(np.float32, copy=False)

    batch_size = min(max(1024, 4 * effective_clusters), full_values.shape[0])
    model = MiniBatchKMeans(
        n_clusters=effective_clusters,
        random_state=seed,
        batch_size=batch_size,
        n_init=3,
        max_iter=100,
    )
    model.fit(fit_values)
    bucket_index = model.predict(full_values).astype(np.int64, copy=False)
    counts = np.bincount(bucket_index, minlength=effective_clusters).astype(np.int64, copy=False)
    bucket_names = np.asarray([f"cluster_{bucket_id}" for bucket_id in range(effective_clusters)], dtype=object)
    center_norms = np.linalg.norm(model.cluster_centers_.astype(np.float64, copy=False), axis=1)

    return {
        "feature_set": "continuous_kmeans",
        "bucket_kind": "kmeans",
        "bucket_index": bucket_index,
        "bucket_names": bucket_names,
        "bucket_counts": counts,
        "fit_rows": int(fit_size),
        "requested_clusters": int(n_clusters),
        "effective_clusters": int(effective_clusters),
        "cluster_center_l2_norm_mean": float(np.mean(center_norms)),
        "cluster_center_l2_norm_std": float(np.std(center_norms)),
    }


def run_action_bucket_analysis(
    *,
    bucket_specs: list[dict[str, Any]],
    action_targets: dict[str, np.ndarray],
    output_dir: Path,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    json_summary: dict[str, dict[str, Any]] = {}

    for bucket_spec in bucket_specs:
        feature_set = str(bucket_spec["feature_set"])
        bucket_index = np.asarray(bucket_spec["bucket_index"], dtype=np.int64)
        bucket_names = np.asarray(bucket_spec["bucket_names"], dtype=object)
        bucket_counts = np.asarray(bucket_spec["bucket_counts"], dtype=np.int64)
        feature_summary: dict[str, Any] = {
            "bucket_kind": str(bucket_spec["bucket_kind"]),
            "total_buckets": int(bucket_names.shape[0]),
            "active_buckets": int(np.sum(bucket_counts > 0)),
            "singleton_buckets": int(np.sum(bucket_counts == 1)),
            "max_bucket_usage": int(np.max(bucket_counts)) if bucket_counts.shape[0] > 0 else 0,
            "max_bucket_fraction": float(np.max(bucket_counts) / bucket_index.shape[0]) if bucket_index.shape[0] > 0 and bucket_counts.shape[0] > 0 else 0.0,
            "targets": {},
        }
        for key in ("fit_rows", "requested_clusters", "effective_clusters", "cluster_center_l2_norm_mean", "cluster_center_l2_norm_std"):
            if key in bucket_spec:
                feature_summary[key] = bucket_spec[key]

        for target_name, target_values in action_targets.items():
            stats_df, target_summary = compute_bucket_action_statistics(
                bucket_index=bucket_index,
                bucket_names=bucket_names,
                target_values=target_values,
            )
            stats_path = output_dir / f"action_buckets__{feature_set}__{target_name}.csv"
            stats_df.to_csv(stats_path, index=False)
            summary_rows.append(
                {
                    "feature_set": feature_set,
                    "bucket_kind": str(bucket_spec["bucket_kind"]),
                    "target": target_name,
                    "total_buckets": int(feature_summary["total_buckets"]),
                    "active_buckets": int(target_summary["active_buckets"]),
                    "singleton_buckets": int(target_summary["singleton_buckets"]),
                    "max_bucket_usage": int(target_summary["max_bucket_usage"]),
                    "max_bucket_fraction": float(target_summary["max_bucket_fraction"]),
                    "mean_variance_explained": float(target_summary["mean_variance_explained"]),
                    "mean_within_bucket_std": float(target_summary["mean_within_bucket_std"]),
                }
            )
            feature_summary["targets"][target_name] = {
                **target_summary,
                "artifact": stats_path.name,
                "top_rows": stats_df.head(top_k).to_dict(orient="records"),
            }
        json_summary[feature_set] = feature_summary

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_variance_explained", "max_bucket_fraction"],
        ascending=[False, False],
        ignore_index=True,
    )
    summary_df.to_csv(output_dir / "action_bucket_summary.csv", index=False)
    return summary_df, json_summary


def run_bucket_context_analysis(
    *,
    bucket_specs: list[dict[str, Any]],
    valid_episode_index: np.ndarray,
    valid_frame_index: np.ndarray,
    valid_progress_index: np.ndarray,
    valid_episode_lengths: np.ndarray,
    progress_bins: int,
    output_dir: Path,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    json_summary: dict[str, dict[str, Any]] = {}

    for bucket_spec in bucket_specs:
        feature_set = str(bucket_spec["feature_set"])
        bucket_index = np.asarray(bucket_spec["bucket_index"], dtype=np.int64)
        bucket_names = np.asarray(bucket_spec["bucket_names"], dtype=object)
        stats_df, feature_summary = compute_bucket_context_statistics(
            bucket_index=bucket_index,
            bucket_names=bucket_names,
            valid_episode_index=valid_episode_index,
            valid_frame_index=valid_frame_index,
            valid_progress_index=valid_progress_index,
            valid_episode_lengths=valid_episode_lengths,
            progress_bins=progress_bins,
        )
        stats_path = output_dir / f"bucket_context__{feature_set}.csv"
        stats_df.to_csv(stats_path, index=False)
        summary_rows.append(
            {
                "feature_set": feature_set,
                "bucket_kind": str(bucket_spec["bucket_kind"]),
                "total_buckets": int(feature_summary["total_buckets"]),
                "active_buckets": int(feature_summary["active_buckets"]),
                "bucket_episode_nmi": float(feature_summary["bucket_episode_nmi"]),
                "weighted_mean_episode_coverage": float(feature_summary["weighted_mean_episode_coverage"]),
                "weighted_mean_max_episode_fraction": float(feature_summary["weighted_mean_max_episode_fraction"]),
                "weighted_mean_episode_perplexity": float(feature_summary["weighted_mean_episode_perplexity"]),
                "weighted_mean_run_length": float(feature_summary["weighted_mean_run_length"]),
                "weighted_max_run_length": float(feature_summary["weighted_max_run_length"]),
                "weighted_adjacent_repeat_fraction": float(feature_summary["weighted_adjacent_repeat_fraction"]),
                "weighted_progress_bin_coverage": float(feature_summary["weighted_progress_bin_coverage"]),
            }
        )
        json_summary[feature_set] = {
            "bucket_kind": str(bucket_spec["bucket_kind"]),
            **feature_summary,
            "artifact": stats_path.name,
            "top_rows": stats_df.head(top_k).to_dict(orient="records"),
        }

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["weighted_mean_episode_coverage", "weighted_mean_max_episode_fraction"],
        ascending=[False, True],
        ignore_index=True,
    )
    summary_df.to_csv(output_dir / "bucket_context_summary.csv", index=False)
    return summary_df, json_summary


def build_probe_feature_sets(
    *,
    ids: np.ndarray | None,
    codebook_vectors_flat: np.ndarray | None,
    continuous_flat: np.ndarray | None,
) -> dict[str, np.ndarray]:
    feature_sets = {}
    if ids is not None:
        feature_sets["ids_onehot"] = ids
    if codebook_vectors_flat is not None:
        feature_sets["codebook_vectors"] = codebook_vectors_flat
    if continuous_flat is not None:
        feature_sets["continuous"] = continuous_flat
    if not feature_sets:
        raise ValueError("At least one feature set is required for action probes.")
    return feature_sets


def _select_episode_rows(
    valid_episode_index: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    unique_eps, counts = np.unique(valid_episode_index, return_counts=True)
    if unique_eps.shape[0] < 2:
        raise ValueError("Episode-split action probes require at least two valid episodes.")
    if max_samples >= valid_episode_index.shape[0]:
        return np.arange(valid_episode_index.shape[0], dtype=np.int64)

    order = rng.permutation(unique_eps.shape[0])
    selected_eps = []
    running_total = 0
    for order_idx in order:
        selected_eps.append(unique_eps[order_idx])
        running_total += int(counts[order_idx])
        if running_total >= max_samples and len(selected_eps) >= 2:
            break

    mask = np.isin(valid_episode_index, np.asarray(selected_eps, dtype=np.int64))
    selected_rows = np.flatnonzero(mask).astype(np.int64, copy=False)
    if selected_rows.shape[0] == 0:
        raise ValueError("Failed to select any rows for the episode-split probe.")
    return selected_rows


def make_probe_split(
    valid_episode_index: np.ndarray,
    max_samples: int,
    test_size: float,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if valid_episode_index.shape[0] < 2:
        raise ValueError("Action probes require at least two valid rows.")

    rng = np.random.default_rng(seed)
    if mode == "row":
        sample_size = min(max_samples, valid_episode_index.shape[0])
        sampled_rows = rng.choice(valid_episode_index.shape[0], size=sample_size, replace=False)
        train_rows, test_rows = train_test_split(sampled_rows, test_size=test_size, random_state=seed, shuffle=True)
        return (
            np.sort(np.asarray(train_rows, dtype=np.int64)),
            np.sort(np.asarray(test_rows, dtype=np.int64)),
        )

    if mode != "episode":
        raise ValueError(f"Unsupported probe split mode: {mode!r}")

    selected_rows = _select_episode_rows(valid_episode_index, max_samples=max_samples, rng=rng)
    selected_eps = np.unique(valid_episode_index[selected_rows])
    if selected_eps.shape[0] < 2:
        raise ValueError("Episode-split action probes require at least two valid episodes after sampling.")

    train_eps, test_eps = train_test_split(selected_eps, test_size=test_size, random_state=seed, shuffle=True)
    train_rows = selected_rows[np.isin(valid_episode_index[selected_rows], train_eps)]
    test_rows = selected_rows[np.isin(valid_episode_index[selected_rows], test_eps)]
    if train_rows.shape[0] == 0 or test_rows.shape[0] == 0:
        raise ValueError("Episode-split action probes produced an empty train or test split.")
    return np.sort(train_rows.astype(np.int64, copy=False)), np.sort(test_rows.astype(np.int64, copy=False))


def transform_probe_features(
    feature_name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    model_name: str,
) -> tuple[Any, Any]:
    if feature_name == "ids_onehot":
        encoder = make_one_hot_encoder()
        x_train_transformed = encoder.fit_transform(x_train)
        x_test_transformed = encoder.transform(x_test)
        if model_name == "mlp":
            x_train_transformed = x_train_transformed.toarray().astype(np.float32, copy=False)
            x_test_transformed = x_test_transformed.toarray().astype(np.float32, copy=False)
        return x_train_transformed, x_test_transformed

    scaler = StandardScaler()
    x_train_transformed = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    x_test_transformed = scaler.transform(x_test).astype(np.float32, copy=False)
    return x_train_transformed, x_test_transformed


def _score_probe_predictions(
    *,
    feature_name: str,
    target_name: str,
    y_test: np.ndarray,
    prediction: np.ndarray,
    probe_model: str,
    split_mode: str,
    n_train: int,
    n_test: int,
) -> list[dict[str, Any]]:
    raw_r2 = r2_score(y_test, prediction, multioutput="raw_values")
    avg_r2 = r2_score(y_test, prediction, multioutput="uniform_average")
    raw_mse = mean_squared_error(y_test, prediction, multioutput="raw_values")
    avg_mse = mean_squared_error(y_test, prediction, multioutput="uniform_average")
    rows = []
    for action_dim, (r2_value, mse_value) in enumerate(zip(raw_r2.tolist(), raw_mse.tolist(), strict=True)):
        rows.append(
            {
                "probe_model": probe_model,
                "split_mode": split_mode,
                "feature_set": feature_name,
                "target": target_name,
                "action_dim": action_dim,
                "r2": float(r2_value),
                "avg_r2_for_target": float(avg_r2),
                "mse": float(mse_value),
                "avg_mse_for_target": float(avg_mse),
                "n_train": int(n_train),
                "n_test": int(n_test),
            }
        )
    return rows


def fit_ridge_probe(
    feature_sets: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    ridge_alpha: float,
    split_mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name, features in feature_sets.items():
        x_train = features[train_rows]
        x_test = features[test_rows]
        x_train_transformed, x_test_transformed = transform_probe_features(
            feature_name,
            x_train,
            x_test,
            model_name="ridge",
        )
        model = Ridge(alpha=ridge_alpha)
        for target_name, target_values in targets.items():
            y_train = target_values[train_rows]
            y_test = target_values[test_rows]
            model.fit(x_train_transformed, y_train)
            prediction = model.predict(x_test_transformed)
            rows.extend(
                _score_probe_predictions(
                    feature_name=feature_name,
                    target_name=target_name,
                    y_test=y_test,
                    prediction=prediction,
                    probe_model="ridge",
                    split_mode=split_mode,
                    n_train=len(train_rows),
                    n_test=len(test_rows),
                )
            )
    return pd.DataFrame(rows)


def fit_mlp_probe(
    feature_sets: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    split_mode: str,
    *,
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    max_iter: int,
    early_stopping: bool,
    n_iter_no_change: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name, features in feature_sets.items():
        x_train = features[train_rows]
        x_test = features[test_rows]
        x_train_transformed, x_test_transformed = transform_probe_features(
            feature_name,
            x_train,
            x_test,
            model_name="mlp",
        )
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=1e-3,
            max_iter=max_iter,
            shuffle=True,
            random_state=seed,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=0.1,
        )
        for target_name, target_values in targets.items():
            y_train = target_values[train_rows]
            y_test = target_values[test_rows]
            model.fit(x_train_transformed, y_train)
            prediction = model.predict(x_test_transformed)
            rows.extend(
                _score_probe_predictions(
                    feature_name=feature_name,
                    target_name=target_name,
                    y_test=y_test,
                    prediction=np.asarray(prediction, dtype=np.float32),
                    probe_model="mlp",
                    split_mode=split_mode,
                    n_train=len(train_rows),
                    n_test=len(test_rows),
                )
            )
    return pd.DataFrame(rows)


def run_action_probes(
    *,
    ids: np.ndarray | None,
    codebook_vectors_flat: np.ndarray | None,
    continuous_flat: np.ndarray | None,
    valid_episode_index: np.ndarray,
    targets: dict[str, np.ndarray],
    max_samples: int,
    test_size: float,
    probe_model: str,
    split_mode: str,
    ridge_alpha: float,
    mlp_hidden_layer_sizes: tuple[int, ...],
    mlp_alpha: float,
    mlp_max_iter: int,
    mlp_early_stopping: bool,
    mlp_n_iter_no_change: int,
    seed: int,
) -> pd.DataFrame:
    train_rows, test_rows = make_probe_split(
        valid_episode_index,
        max_samples=max_samples,
        test_size=test_size,
        seed=seed,
        mode=split_mode,
    )
    feature_sets = build_probe_feature_sets(
        ids=ids,
        codebook_vectors_flat=codebook_vectors_flat,
        continuous_flat=continuous_flat,
    )

    frames = []
    if probe_model in {"ridge", "both"}:
        frames.append(
            fit_ridge_probe(
                feature_sets,
                targets,
                train_rows,
                test_rows,
                ridge_alpha=ridge_alpha,
                split_mode=split_mode,
            )
        )
    if probe_model in {"mlp", "both"}:
        frames.append(
            fit_mlp_probe(
                feature_sets,
                targets,
                train_rows,
                test_rows,
                split_mode=split_mode,
                hidden_layer_sizes=mlp_hidden_layer_sizes,
                alpha=mlp_alpha,
                max_iter=mlp_max_iter,
                early_stopping=mlp_early_stopping,
                n_iter_no_change=mlp_n_iter_no_change,
                seed=seed,
            )
        )
    if not frames:
        raise ValueError(f"Unsupported probe model choice: {probe_model!r}")

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["probe_model", "feature_set", "target", "action_dim"], ignore_index=True)
    )


def summarize_probe_scores(probe_df: pd.DataFrame) -> pd.DataFrame:
    return (
        probe_df.groupby(["probe_model", "split_mode", "feature_set", "target"], as_index=False)
        .agg(mean_r2=("r2", "mean"), mean_mse=("mse", "mean"))
        .sort_values(["probe_model", "mean_r2", "mean_mse"], ascending=[True, False, True], ignore_index=True)
    )


def plot_probe_heatmaps(probe_df: pd.DataFrame, output_dir: Path) -> list[str]:
    written = []
    groups = list(probe_df.groupby(["probe_model", "split_mode"], sort=True))
    for (probe_model, split_mode), group_df in groups:
        row_label_df = group_df.assign(row_label=lambda df: df["feature_set"] + " -> " + df["target"])
        r2_pivot = (
            row_label_df.pivot_table(index="row_label", columns="action_dim", values="r2", aggfunc="mean").sort_index()
        )
        mse_pivot = (
            row_label_df.pivot_table(index="row_label", columns="action_dim", values="mse", aggfunc="mean").sort_index()
        )
        r2_path = output_dir / f"action_probe_r2_heatmap__{probe_model}__{split_mode}.png"
        mse_path = output_dir / f"action_probe_mse_heatmap__{probe_model}__{split_mode}.png"
        plot_heatmap(
            r2_pivot.to_numpy(),
            r2_pivot.index.tolist(),
            [f"a{int(col)}" for col in r2_pivot.columns.tolist()],
            f"Held-Out Action Probe R^2 ({probe_model}, split={split_mode})",
            "R^2",
            r2_path,
        )
        plot_heatmap(
            mse_pivot.to_numpy(),
            mse_pivot.index.tolist(),
            [f"a{int(col)}" for col in mse_pivot.columns.tolist()],
            f"Held-Out Action Probe MSE ({probe_model}, split={split_mode})",
            "MSE",
            mse_path,
        )
        written.extend([r2_path.name, mse_path.name])

        if len(groups) == 1:
            legacy_r2 = output_dir / "action_probe_r2_heatmap.png"
            legacy_mse = output_dir / "action_probe_mse_heatmap.png"
            plot_heatmap(
                r2_pivot.to_numpy(),
                r2_pivot.index.tolist(),
                [f"a{int(col)}" for col in r2_pivot.columns.tolist()],
                "Held-Out Action Probe R^2",
                "R^2",
                legacy_r2,
            )
            plot_heatmap(
                mse_pivot.to_numpy(),
                mse_pivot.index.tolist(),
                [f"a{int(col)}" for col in mse_pivot.columns.tolist()],
                "Held-Out Action Probe MSE",
                "MSE",
                legacy_mse,
            )
            written.extend([legacy_r2.name, legacy_mse.name])
    return written


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    if args.bucket_kmeans_clusters < 0:
        raise ValueError("--bucket-kmeans-clusters must be >= 0.")
    if args.bucket_kmeans_fit_samples < 1:
        raise ValueError("--bucket-kmeans-fit-samples must be >= 1.")
    if args.bucket_top_k < 1:
        raise ValueError("--bucket-top-k must be >= 1.")
    if args.bucket_progress_bins < 1:
        raise ValueError("--bucket-progress-bins must be >= 1.")

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

    def pick_feature_name(*suffixes: str, required: bool = True) -> str | None:
        for suffix in suffixes:
            candidate = f"{args.feature_prefix}.{suffix}"
            if candidate in info["features"]:
                return candidate
        if not required:
            return None
        raise KeyError(
            f"Missing feature in dataset metadata for prefix {args.feature_prefix!r}. "
            f"Tried suffixes: {list(suffixes)}"
        )

    ids_col = pick_feature_name("codebook_id_latents", "codebook_ids", required=False)
    continuous_col = pick_feature_name("continuous_vector_latents", "continuous")
    codebook_vectors_col = pick_feature_name("codebook_vector_latents", "codebook_vectors", required=False)
    valid_col = f"{args.feature_prefix}.valid"
    action_col = "action"

    required_columns = [continuous_col, valid_col, action_col]
    for column_name in required_columns:
        if column_name not in info["features"]:
            raise KeyError(f"Missing feature in dataset metadata: {column_name}")

    valid_counts = load_valid_counts(dataset, valid_col)
    plot_valid_distribution(valid_counts, output_dir / "valid_distribution.png")

    all_actions, all_valid, episode_index, frame_index = load_action_context(dataset, action_col, valid_col)
    tail_counts = infer_episode_tail_counts(all_valid, episode_index)
    action_targets = derive_action_targets(all_actions, all_valid, episode_index, args.future_frames)
    valid_episode_index = extract_valid_episode_index(all_valid, episode_index)
    valid_frame_index = extract_valid_scalar_context(frame_index, all_valid, episode_index).astype(np.int64, copy=False)
    valid_progress_index, valid_episode_lengths = extract_valid_progress_context(all_valid, episode_index)

    continuous = load_float_array(dataset, continuous_col, valid_col)
    valid_frames = int(continuous.shape[0])
    if valid_episode_index.shape[0] != valid_frames:
        raise ValueError(
            "Valid-row episode index alignment failed: action targets and latent arrays disagree on row count."
        )
    if valid_frame_index.shape[0] != valid_frames or valid_progress_index.shape[0] != valid_frames:
        raise ValueError("Valid-row frame/progress alignment failed: context arrays disagree on row count.")

    ids = None
    id_inverse = None
    id_counts = None
    unique_id_seqs = None
    top_id_df = None
    position_df = None
    id_usage_per_row = np.ones(valid_frames, dtype=np.int64)
    if ids_col is not None:
        ids = load_ids(dataset, ids_col, valid_col)
        if ids.shape[0] != valid_frames:
            raise ValueError("Discrete ID latents and continuous latents disagree on the number of valid rows.")
        unique_id_seqs, id_inverse, id_counts = unique_rows(ids)
        id_usage_per_row = id_counts[id_inverse]

        unique_sequences, unique_sequence_inverse = np.unique(contiguous_row_view(ids), return_index=True)
        representative_ids = ids[unique_sequence_inverse]
        representative_counts = id_counts[np.arange(len(unique_sequences))]
        id_order = np.argsort(representative_counts)[::-1]
        top_id_df = pd.DataFrame(
            {
                "sequence": [format_sequence(seq) for seq in representative_ids[id_order]],
                "count": representative_counts[id_order],
                "fraction": representative_counts[id_order] / ids.shape[0],
            }
        )
        top_id_df.to_csv(output_dir / "codebook_id_sequence_counts.csv", index=False)
        plot_top_sequences(top_id_df.head(args.top_k_sequences), output_dir / "codebook_id_top_sequences.png")

        position_df = plot_id_position_counts(ids, output_dir / "codebook_id_position_counts.png")
        position_df.to_csv(output_dir / "codebook_id_position_counts.csv", index=False)

    binned_action_targets, action_bin_counts = quantile_bin_targets(action_targets, args.action_bins)
    action_mi_df = None
    action_mi_ranking_df = None
    if ids is not None and id_inverse is not None:
        action_mi_df, action_mi_ranking_df = compute_action_mutual_information(ids, id_inverse, binned_action_targets)
        action_mi_df.to_csv(output_dir / "id_action_mutual_information.csv", index=False)
        action_mi_ranking_df.to_csv(output_dir / "id_action_mutual_information_ranked.csv", index=False)

        mi_pivot = (
            action_mi_df.assign(row_label=lambda df: df["target"] + "_a" + df["action_dim"].astype(str))
            .pivot(index="row_label", columns="feature", values="mi")
            .sort_index()
        )
        nmi_pivot = (
            action_mi_df.assign(row_label=lambda df: df["target"] + "_a" + df["action_dim"].astype(str))
            .pivot(index="row_label", columns="feature", values="nmi")
            .sort_index()
        )
        plot_heatmap(
            mi_pivot.to_numpy(),
            mi_pivot.index.tolist(),
            mi_pivot.columns.tolist(),
            "Mutual Information Between ID Features and Action Targets",
            "MI",
            output_dir / "id_action_mutual_information_heatmap.png",
        )
        plot_heatmap(
            nmi_pivot.to_numpy(),
            nmi_pivot.index.tolist(),
            nmi_pivot.columns.tolist(),
            "Normalized Mutual Information Between ID Features and Action Targets",
            "NMI",
            output_dir / "id_action_normalized_mutual_information_heatmap.png",
        )

    continuous_flat = continuous.reshape(continuous.shape[0], -1)
    codebook_vectors = None
    codebook_vectors_flat = None
    if codebook_vectors_col is not None:
        codebook_vectors = load_float_array(dataset, codebook_vectors_col, valid_col)
        if codebook_vectors.shape[0] != valid_frames:
            raise ValueError("Codebook vectors and continuous latents disagree on the number of valid rows.")
        codebook_vectors_flat = codebook_vectors.reshape(codebook_vectors.shape[0], -1)

    continuous_exact_unique, _, continuous_counts = unique_rows(continuous_flat)
    continuous_rounded = np.round(continuous_flat, decimals=args.rounded_decimals)
    continuous_rounded_unique, _, continuous_rounded_counts = unique_rows(continuous_rounded)

    codebook_vectors_exact_unique = None
    codebook_vectors_counts = None
    codebook_vectors_rounded = None
    codebook_vectors_rounded_unique = None
    codebook_vectors_rounded_counts = None
    if codebook_vectors_flat is not None:
        codebook_vectors_exact_unique, _, codebook_vectors_counts = unique_rows(codebook_vectors_flat)
        codebook_vectors_rounded = np.round(codebook_vectors_flat, decimals=args.rounded_decimals)
        codebook_vectors_rounded_unique, _, codebook_vectors_rounded_counts = unique_rows(codebook_vectors_rounded)

    plot_value_histogram(continuous, "Continuous Latent Value Distribution", output_dir / "continuous_value_histogram.png")
    plot_slot_norms(continuous, "Continuous Latent L2 Norms by Slot", output_dir / "continuous_slot_norms.png")
    if codebook_vectors is not None:
        plot_value_histogram(
            codebook_vectors,
            "Codebook Vector Value Distribution",
            output_dir / "codebook_vectors_value_histogram.png",
        )
        plot_slot_norms(
            codebook_vectors,
            "Codebook Vector L2 Norms by Slot",
            output_dir / "codebook_vectors_slot_norms.png",
        )

    continuous_norms_df = summarize_norms(continuous)
    continuous_norms_df.to_csv(output_dir / "continuous_slot_norm_summary.csv", index=False)
    codebook_vectors_norms_df = None
    if codebook_vectors is not None:
        codebook_vectors_norms_df = summarize_norms(codebook_vectors)
        codebook_vectors_norms_df.to_csv(output_dir / "codebook_vectors_slot_norm_summary.csv", index=False)

    pca_colorbar_label = "log10(ID sequence usage)" if ids is not None else "constant (discrete IDs unavailable)"
    continuous_pca_title = (
        "Continuous Latents PCA Colored by ID Sequence Usage" if ids is not None else "Continuous Latents PCA"
    )

    make_pca_scatter(
        continuous_flat,
        id_usage_per_row,
        continuous_pca_title,
        output_dir / "continuous_pca_by_id_sequence_usage.png",
        output_dir / "continuous_pca_sample.csv",
        rng,
        args.pca_fit_points,
        args.scatter_points,
        colorbar_label=pca_colorbar_label,
    )
    if codebook_vectors_flat is not None:
        make_pca_scatter(
            codebook_vectors_flat,
            id_usage_per_row,
            "Codebook Vectors PCA Colored by ID Sequence Usage" if ids is not None else "Codebook Vectors PCA",
            output_dir / "codebook_vectors_pca_by_id_sequence_usage.png",
            output_dir / "codebook_vectors_pca_sample.csv",
            rng,
            args.pca_fit_points,
            args.scatter_points,
            colorbar_label=pca_colorbar_label,
        )

    bucket_specs = []
    if ids is not None and id_inverse is not None:
        bucket_specs.append(make_discrete_bucket_spec(ids, id_inverse))
    if args.bucket_kmeans_clusters > 0:
        bucket_specs.append(
            make_continuous_bucket_spec(
                continuous_flat,
                n_clusters=args.bucket_kmeans_clusters,
                fit_samples=args.bucket_kmeans_fit_samples,
                seed=args.seed,
            )
        )

    if bucket_specs:
        action_bucket_summary_df, action_bucket_summary = run_action_bucket_analysis(
            bucket_specs=bucket_specs,
            action_targets=action_targets,
            output_dir=output_dir,
            top_k=args.bucket_top_k,
        )
        bucket_context_summary_df, bucket_context_summary = run_bucket_context_analysis(
            bucket_specs=bucket_specs,
            valid_episode_index=valid_episode_index,
            valid_frame_index=valid_frame_index,
            valid_progress_index=valid_progress_index,
            valid_episode_lengths=valid_episode_lengths,
            progress_bins=args.bucket_progress_bins,
            output_dir=output_dir,
            top_k=args.bucket_top_k,
        )
    else:
        action_bucket_summary_df = pd.DataFrame(
            columns=[
                "feature_set",
                "bucket_kind",
                "target",
                "total_buckets",
                "active_buckets",
                "singleton_buckets",
                "max_bucket_usage",
                "max_bucket_fraction",
                "mean_variance_explained",
                "mean_within_bucket_std",
            ]
        )
        action_bucket_summary = {}
        action_bucket_summary_df.to_csv(output_dir / "action_bucket_summary.csv", index=False)
        bucket_context_summary_df = pd.DataFrame(
            columns=[
                "feature_set",
                "bucket_kind",
                "total_buckets",
                "active_buckets",
                "bucket_episode_nmi",
                "weighted_mean_episode_coverage",
                "weighted_mean_max_episode_fraction",
                "weighted_mean_episode_perplexity",
                "weighted_mean_run_length",
                "weighted_max_run_length",
                "weighted_adjacent_repeat_fraction",
                "weighted_progress_bin_coverage",
            ]
        )
        bucket_context_summary = {}
        bucket_context_summary_df.to_csv(output_dir / "bucket_context_summary.csv", index=False)

    probe_df = run_action_probes(
        ids=ids,
        codebook_vectors_flat=codebook_vectors_flat,
        continuous_flat=continuous_flat,
        valid_episode_index=valid_episode_index,
        targets=action_targets,
        max_samples=args.probe_max_samples,
        test_size=args.probe_test_size,
        probe_model=args.probe_model,
        split_mode=args.probe_split,
        ridge_alpha=args.ridge_alpha,
        mlp_hidden_layer_sizes=args.probe_mlp_hidden_dims,
        mlp_alpha=args.probe_mlp_alpha,
        mlp_max_iter=args.probe_mlp_max_iter,
        mlp_early_stopping=args.probe_mlp_early_stopping,
        mlp_n_iter_no_change=args.probe_mlp_n_iter_no_change,
        seed=args.seed,
    )
    probe_df.to_csv(output_dir / "action_probe_scores.csv", index=False)
    probe_df.to_csv(output_dir / "action_probe_r2.csv", index=False)

    probe_summary_df = summarize_probe_scores(probe_df)
    probe_summary_df.to_csv(output_dir / "action_probe_scores_summary.csv", index=False)
    probe_summary_df.to_csv(output_dir / "action_probe_r2_summary.csv", index=False)
    probe_heatmap_artifacts = plot_probe_heatmaps(probe_df, output_dir)

    best_by_probe_model = []
    for probe_model, model_df in probe_summary_df.groupby("probe_model", sort=True):
        best_r2_row = model_df.sort_values(["mean_r2", "mean_mse"], ascending=[False, True]).iloc[0]
        best_mse_row = model_df.sort_values(["mean_mse", "mean_r2"], ascending=[True, False]).iloc[0]
        best_by_probe_model.append(
            {
                "probe_model": probe_model,
                "split_mode": str(best_r2_row["split_mode"]),
                "best_mean_r2": {
                    "feature_set": str(best_r2_row["feature_set"]),
                    "target": str(best_r2_row["target"]),
                    "value": float(best_r2_row["mean_r2"]),
                },
                "best_mean_mse": {
                    "feature_set": str(best_mse_row["feature_set"]),
                    "target": str(best_mse_row["target"]),
                    "value": float(best_mse_row["mean_mse"]),
                },
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "feature_prefix": args.feature_prefix,
        "future_frames": args.future_frames,
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
        "id_sequences": None
        if ids is None or id_counts is None or unique_id_seqs is None
        else {
            "sequence_length": int(ids.shape[1]),
            "unique_sequences": unique_id_seqs,
            "singleton_sequences": int(np.sum(id_counts == 1)),
            "max_sequence_usage": int(np.max(id_counts)),
            "mean_sequence_usage": float(np.mean(id_counts)),
            "median_sequence_usage": float(np.median(id_counts)),
        },
        "continuous": {
            "shape_per_frame": list(continuous.shape[1:]),
            "flattened_dim": int(continuous_flat.shape[1]),
            "value_summary": summarize_numeric(continuous),
            "exact_unique_rows": continuous_exact_unique,
            "exact_singleton_rows": int(np.sum(continuous_counts == 1)),
            "exact_max_usage": int(np.max(continuous_counts)),
            "rounded_decimals": args.rounded_decimals,
            "rounded_unique_rows": continuous_rounded_unique,
            "rounded_singleton_rows": int(np.sum(continuous_rounded_counts == 1)),
            "rounded_max_usage": int(np.max(continuous_rounded_counts)),
        },
        "codebook_vectors": None
        if (
            codebook_vectors is None
            or codebook_vectors_flat is None
            or codebook_vectors_exact_unique is None
            or codebook_vectors_counts is None
            or codebook_vectors_rounded_unique is None
            or codebook_vectors_rounded_counts is None
        )
        else {
            "shape_per_frame": list(codebook_vectors.shape[1:]),
            "flattened_dim": int(codebook_vectors_flat.shape[1]),
            "value_summary": summarize_numeric(codebook_vectors),
            "exact_unique_rows": codebook_vectors_exact_unique,
            "exact_singleton_rows": int(np.sum(codebook_vectors_counts == 1)),
            "exact_max_usage": int(np.max(codebook_vectors_counts)),
            "rounded_decimals": args.rounded_decimals,
            "rounded_unique_rows": codebook_vectors_rounded_unique,
            "rounded_singleton_rows": int(np.sum(codebook_vectors_rounded_counts == 1)),
            "rounded_max_usage": int(np.max(codebook_vectors_rounded_counts)),
        },
        "action_targets": {
            target_name: {
                "shape": list(values.shape),
                "value_summary": summarize_numeric(values),
                "binned_dimensions": action_bin_counts[target_name],
            }
            for target_name, values in action_targets.items()
        },
        "action_mutual_information": None
        if action_mi_ranking_df is None
        else {
            "top_rows": action_mi_ranking_df.head(10).to_dict(orient="records"),
        },
        "action_probes": {
            "probe_model": args.probe_model,
            "split_mode": args.probe_split,
            "mean_scores_by_feature_and_target": probe_summary_df.to_dict(orient="records"),
            "best_by_probe_model": best_by_probe_model,
        },
        "action_buckets": {
            "summary_rows": action_bucket_summary_df.to_dict(orient="records"),
            "by_feature_set": action_bucket_summary,
        },
        "bucket_context": {
            "summary_rows": bucket_context_summary_df.to_dict(orient="records"),
            "by_feature_set": bucket_context_summary,
        },
        "artifacts": sorted(p.name for p in output_dir.iterdir()),
    }
    save_json(output_dir / "summary.json", summary)

    summary_lines = [
        "# Latent Feature Distribution Analysis",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Feature prefix: `{args.feature_prefix}`",
        f"- Future frames used for action summaries: `{args.future_frames}`",
        f"- Total frames: `{summary['total_frames']}`",
        f"- Valid frames: `{summary['valid_frames']}`",
        f"- Invalid frames: `{summary['invalid_frames']}`",
        "",
        "## ID Sequences",
    ]
    if summary["id_sequences"] is not None:
        summary_lines.extend(
            [
                f"- Unique sequences: `{summary['id_sequences']['unique_sequences']}`",
                f"- Singleton sequences: `{summary['id_sequences']['singleton_sequences']}`",
                f"- Max sequence usage: `{summary['id_sequences']['max_sequence_usage']}`",
                f"- Median sequence usage: `{summary['id_sequences']['median_sequence_usage']:.2f}`",
            ]
        )
    else:
        summary_lines.append("- Unavailable for this dataset because no discrete codebook ID latents were exported.")
    summary_lines.extend(
        [
            "",
            "## Continuous",
            f"- Exact unique rows: `{summary['continuous']['exact_unique_rows']}`",
            f"- Rounded unique rows ({args.rounded_decimals} decimals): `{summary['continuous']['rounded_unique_rows']}`",
            f"- Exact max usage: `{summary['continuous']['exact_max_usage']}`",
            "",
            "## Codebook Vectors",
        ]
    )
    if summary["codebook_vectors"] is not None:
        summary_lines.extend(
            [
                f"- Exact unique rows: `{summary['codebook_vectors']['exact_unique_rows']}`",
                f"- Rounded unique rows ({args.rounded_decimals} decimals): `{summary['codebook_vectors']['rounded_unique_rows']}`",
                f"- Exact max usage: `{summary['codebook_vectors']['exact_max_usage']}`",
            ]
        )
    else:
        summary_lines.append("- Unavailable for this dataset because no codebook vector latents were exported.")
    summary_lines.extend(
        [
            "",
            "## Action Mutual Information",
        ]
    )
    if action_mi_ranking_df is not None:
        top_mi_row = action_mi_ranking_df.iloc[0]
        summary_lines.append(
            f"- Top MI pair: `{top_mi_row['feature']}` vs `{top_mi_row['target']}` dim `{int(top_mi_row['action_dim'])}` with MI `{top_mi_row['mi']:.4f}` and NMI `{top_mi_row['nmi']:.4f}`"
        )
    else:
        summary_lines.append("- Unavailable for this dataset because no discrete ID latents were exported.")
    summary_lines.extend(
        [
            "",
            "## Action Probes",
            f"- Probe split mode: `{args.probe_split}`",
            f"- Probe backends requested: `{args.probe_model}`",
        ]
    )
    for best in best_by_probe_model:
        summary_lines.extend(
            [
                f"- Best mean held-out R^2 ({best['probe_model']}): `{best['best_mean_r2']['feature_set']}` -> `{best['best_mean_r2']['target']}` = `{best['best_mean_r2']['value']:.4f}`",
                f"- Best mean held-out MSE ({best['probe_model']}): `{best['best_mean_mse']['feature_set']}` -> `{best['best_mean_mse']['target']}` = `{best['best_mean_mse']['value']:.6f}`",
            ]
        )
    summary_lines.extend(["", "## Action Buckets"])
    if action_bucket_summary_df.shape[0] > 0:
        for row in action_bucket_summary_df.head(args.bucket_top_k).to_dict(orient="records"):
            summary_lines.append(
                f"- `{row['feature_set']}` -> `{row['target']}`: mean variance explained `{row['mean_variance_explained']:.4f}` across `{int(row['active_buckets'])}` active buckets"
            )
    else:
        summary_lines.append("- No action bucket analysis was run.")
    summary_lines.extend(["", "## Bucket Context"])
    if bucket_context_summary_df.shape[0] > 0:
        for row in bucket_context_summary_df.to_dict(orient="records"):
            summary_lines.append(
                f"- `{row['feature_set']}`: episode NMI `{row['bucket_episode_nmi']:.4f}`, weighted episode coverage `{row['weighted_mean_episode_coverage']:.4f}`, weighted max-episode fraction `{row['weighted_mean_max_episode_fraction']:.4f}`, weighted adjacent-repeat fraction `{row['weighted_adjacent_repeat_fraction']:.4f}`"
            )
    else:
        summary_lines.append("- No bucket context analysis was run.")
    summary_lines.extend(
        [
            "",
            "## Scatter Plot Coloring",
            "- PCA plots are colored by `log10(ID sequence usage)` for the corresponding frame."
            if ids is not None
            else "- PCA plots use a constant color scale because discrete ID sequence usage is unavailable.",
            "",
            "## Probe Heatmaps",
        ]
    )
    summary_lines.extend(f"- `{artifact}`" for artifact in sorted(probe_heatmap_artifacts))
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(summary_lines) + "\n")

    def probe_metric(feature_set: str, target: str, metric: str) -> float | None:
        rows = probe_summary_df[
            (probe_summary_df["feature_set"] == feature_set) & (probe_summary_df["target"] == target)
        ]
        if rows.shape[0] == 0:
            return None
        return float(rows.iloc[0][metric])

    def bucket_metric(feature_set: str, target: str, metric: str) -> float | None:
        rows = action_bucket_summary_df[
            (action_bucket_summary_df["feature_set"] == feature_set) & (action_bucket_summary_df["target"] == target)
        ]
        if rows.shape[0] == 0:
            return None
        return float(rows.iloc[0][metric])

    def context_metric(feature_set: str, metric: str) -> float | None:
        rows = bucket_context_summary_df[bucket_context_summary_df["feature_set"] == feature_set]
        if rows.shape[0] == 0:
            return None
        return float(rows.iloc[0][metric])

    analysis_manifest = {
        "artifact_type": "latent_analysis",
        "analysis_kind": "latent_core",
        "suite_name": "latent_core",
        "suite_version": "v1",
        "artifact_id": make_artifact_id(
            suite_name="latent_core",
            suite_version="v1",
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
            "continuous_current_mean_r2": probe_metric("continuous", "current_action", "mean_r2"),
            "continuous_current_mean_mse": probe_metric("continuous", "current_action", "mean_mse"),
            "continuous_future_mean_r2": probe_metric("continuous", "future_action_mean", "mean_r2"),
            "continuous_future_mean_mse": probe_metric("continuous", "future_action_mean", "mean_mse"),
            "id_sequence_current_mean_r2": probe_metric("id_sequence", "current_action", "mean_r2"),
            "id_sequence_current_mean_mse": probe_metric("id_sequence", "current_action", "mean_mse"),
            "id_sequence_future_mean_r2": probe_metric("id_sequence", "future_action_mean", "mean_r2"),
            "id_sequence_future_mean_mse": probe_metric("id_sequence", "future_action_mean", "mean_mse"),
            "continuous_kmeans_current_mean_variance_explained": bucket_metric(
                "continuous_kmeans", "current_action", "mean_variance_explained"
            ),
            "continuous_kmeans_future_mean_variance_explained": bucket_metric(
                "continuous_kmeans", "future_action_mean", "mean_variance_explained"
            ),
            "id_sequence_current_mean_variance_explained": bucket_metric(
                "id_sequence", "current_action", "mean_variance_explained"
            ),
            "id_sequence_future_mean_variance_explained": bucket_metric(
                "id_sequence", "future_action_mean", "mean_variance_explained"
            ),
            "continuous_kmeans_bucket_episode_nmi": context_metric("continuous_kmeans", "bucket_episode_nmi"),
            "continuous_kmeans_weighted_episode_coverage": context_metric(
                "continuous_kmeans", "weighted_mean_episode_coverage"
            ),
            "id_sequence_bucket_episode_nmi": context_metric("id_sequence", "bucket_episode_nmi"),
            "id_sequence_weighted_episode_coverage": context_metric(
                "id_sequence", "weighted_mean_episode_coverage"
            ),
            "top_mi": None if action_mi_ranking_df is None or action_mi_ranking_df.shape[0] == 0 else float(action_mi_ranking_df.iloc[0]["mi"]),
            "top_nmi": None if action_mi_ranking_df is None or action_mi_ranking_df.shape[0] == 0 else float(action_mi_ranking_df.iloc[0]["nmi"]),
        },
    }
    register_artifact(
        manifest_path=output_dir / "analysis_manifest.json",
        manifest=analysis_manifest,
        registry_candidates=[
            output_dir,
            dataset_root,
            checkpoint_meta["source_checkpoint_path"],
        ],
    )


if __name__ == "__main__":
    main()
