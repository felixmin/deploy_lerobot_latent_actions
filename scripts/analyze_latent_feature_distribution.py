#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze latent feature distributions for a labeled LeRobot dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root of the labeled dataset.")
    parser.add_argument("--feature-prefix", type=str, required=True, help="Feature prefix, e.g. lapa_lam_120000.")
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
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization used for linear action probes.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_info(dataset_root: Path) -> dict:
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
    return np.stack(obj).astype(np.int64, copy=False)


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


def contiguous_row_view(arr_2d: np.ndarray) -> np.ndarray:
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
        future_mean_chunks.append((future_sum / float(future_frames)).astype(np.float32, copy=False))

    return {
        "current_action": np.concatenate(current_chunks, axis=0),
        "future_action_mean": np.concatenate(future_mean_chunks, axis=0),
    }


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
        s=5,
        alpha=0.35,
        cmap="viridis",
        rasterized=True,
        linewidths=0,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("log10(ID sequence usage)")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
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


def run_action_probes(
    ids: np.ndarray,
    codebook_vectors_flat: np.ndarray,
    continuous_flat: np.ndarray,
    targets: dict[str, np.ndarray],
    max_samples: int,
    test_size: float,
    ridge_alpha: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample_size = min(max_samples, ids.shape[0])
    sampled_rows = rng.choice(ids.shape[0], size=sample_size, replace=False)
    train_rows, test_rows = train_test_split(sampled_rows, test_size=test_size, random_state=seed, shuffle=True)

    feature_sets = {
        "ids_onehot": ids,
        "codebook_vectors": codebook_vectors_flat,
        "continuous": continuous_flat,
    }

    rows = []
    for feature_name, features in feature_sets.items():
        x_train = features[train_rows]
        x_test = features[test_rows]

        if feature_name == "ids_onehot":
            encoder = make_one_hot_encoder()
            x_train_transformed = encoder.fit_transform(x_train)
            x_test_transformed = encoder.transform(x_test)
        else:
            scaler = StandardScaler()
            x_train_transformed = scaler.fit_transform(x_train)
            x_test_transformed = scaler.transform(x_test)

        model = Ridge(alpha=ridge_alpha)
        for target_name, target_values in targets.items():
            y_train = target_values[train_rows]
            y_test = target_values[test_rows]
            model.fit(x_train_transformed, y_train)
            prediction = model.predict(x_test_transformed)
            raw_r2 = r2_score(y_test, prediction, multioutput="raw_values")
            avg_r2 = r2_score(y_test, prediction, multioutput="uniform_average")
            for action_dim, r2_value in enumerate(raw_r2.tolist()):
                rows.append(
                    {
                        "feature_set": feature_name,
                        "target": target_name,
                        "action_dim": action_dim,
                        "r2": float(r2_value),
                        "avg_r2_for_target": float(avg_r2),
                        "n_train": int(len(train_rows)),
                        "n_test": int(len(test_rows)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = load_info(dataset_root)
    dataset = make_dataset(dataset_root)

    ids_col = f"{args.feature_prefix}.codebook_ids"
    continuous_col = f"{args.feature_prefix}.continuous"
    codebook_vectors_col = f"{args.feature_prefix}.codebook_vectors"
    valid_col = f"{args.feature_prefix}.valid"
    action_col = "action"

    for column_name in [ids_col, continuous_col, codebook_vectors_col, valid_col, action_col]:
        if column_name not in info["features"]:
            raise KeyError(f"Missing feature in dataset metadata: {column_name}")

    valid_counts = load_valid_counts(dataset, valid_col)
    plot_valid_distribution(valid_counts, output_dir / "valid_distribution.png")

    ids = load_ids(dataset, ids_col, valid_col)
    unique_id_seqs, id_inverse, id_counts = unique_rows(ids)
    id_usage_per_row = id_counts[id_inverse]

    unique_sequences, unique_sequence_inverse = np.unique(
        contiguous_row_view(ids), return_index=True
    )
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

    all_actions, all_valid, episode_index = load_action_context(dataset, action_col, valid_col)
    tail_counts = infer_episode_tail_counts(all_valid, episode_index)
    action_targets = derive_action_targets(all_actions, all_valid, episode_index, args.future_frames)
    binned_action_targets, action_bin_counts = quantile_bin_targets(action_targets, args.action_bins)

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

    continuous = load_float_array(dataset, continuous_col, valid_col)
    codebook_vectors = load_float_array(dataset, codebook_vectors_col, valid_col)
    continuous_flat = continuous.reshape(continuous.shape[0], -1)
    codebook_vectors_flat = codebook_vectors.reshape(codebook_vectors.shape[0], -1)

    continuous_exact_unique, continuous_inverse, continuous_counts = unique_rows(continuous_flat)
    continuous_rounded = np.round(continuous_flat, decimals=args.rounded_decimals)
    continuous_rounded_unique, _, continuous_rounded_counts = unique_rows(continuous_rounded)

    codebook_vectors_exact_unique, codebook_vectors_inverse, codebook_vectors_counts = unique_rows(codebook_vectors_flat)
    codebook_vectors_rounded = np.round(codebook_vectors_flat, decimals=args.rounded_decimals)
    codebook_vectors_rounded_unique, _, codebook_vectors_rounded_counts = unique_rows(codebook_vectors_rounded)

    plot_value_histogram(continuous, "Continuous Latent Value Distribution", output_dir / "continuous_value_histogram.png")
    plot_slot_norms(continuous, "Continuous Latent L2 Norms by Slot", output_dir / "continuous_slot_norms.png")
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
    codebook_vectors_norms_df = summarize_norms(codebook_vectors)
    codebook_vectors_norms_df.to_csv(output_dir / "codebook_vectors_slot_norm_summary.csv", index=False)

    make_pca_scatter(
        continuous_flat,
        id_usage_per_row,
        "Continuous Latents PCA Colored by ID Sequence Usage",
        output_dir / "continuous_pca_by_id_sequence_usage.png",
        output_dir / "continuous_pca_sample.csv",
        rng,
        args.pca_fit_points,
        args.scatter_points,
    )
    make_pca_scatter(
        codebook_vectors_flat,
        id_usage_per_row,
        "Codebook Vectors PCA Colored by ID Sequence Usage",
        output_dir / "codebook_vectors_pca_by_id_sequence_usage.png",
        output_dir / "codebook_vectors_pca_sample.csv",
        rng,
        args.pca_fit_points,
        args.scatter_points,
    )

    probe_df = run_action_probes(
        ids=ids,
        codebook_vectors_flat=codebook_vectors_flat,
        continuous_flat=continuous_flat,
        targets=action_targets,
        max_samples=args.probe_max_samples,
        test_size=args.probe_test_size,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
    )
    probe_df.to_csv(output_dir / "action_probe_r2.csv", index=False)
    probe_pivot = (
        probe_df.assign(row_label=lambda df: df["feature_set"] + " -> " + df["target"])
        .pivot_table(index="row_label", columns="action_dim", values="r2", aggfunc="mean")
        .sort_index()
    )
    plot_heatmap(
        probe_pivot.to_numpy(),
        probe_pivot.index.tolist(),
        [f"a{int(col)}" for col in probe_pivot.columns.tolist()],
        "Held-Out Action Probe R^2",
        "R^2",
        output_dir / "action_probe_r2_heatmap.png",
    )
    probe_summary_df = (
        probe_df.groupby(["feature_set", "target"], as_index=False)["r2"]
        .mean()
        .rename(columns={"r2": "mean_r2"})
        .sort_values("mean_r2", ascending=False)
    )
    probe_summary_df.to_csv(output_dir / "action_probe_r2_summary.csv", index=False)

    summary = {
        "dataset_root": str(dataset_root),
        "feature_prefix": args.feature_prefix,
        "future_frames": args.future_frames,
        "total_frames": int(sum(valid_counts.values())),
        "valid_counts": valid_counts,
        "valid_frames": int(ids.shape[0]),
        "invalid_frames": int(sum(v for k, v in valid_counts.items() if int(k) != 1)),
        "episode_invalid_tail_counts": {
            "min": int(np.min(tail_counts)),
            "median": float(np.median(tail_counts)),
            "max": int(np.max(tail_counts)),
            "unique_values": sorted({int(v) for v in tail_counts}),
        },
        "id_sequences": {
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
        "codebook_vectors": {
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
        "action_mutual_information": {
            "top_rows": action_mi_ranking_df.head(10).to_dict(orient="records"),
        },
        "action_probes": {
            "mean_r2_by_feature_and_target": probe_summary_df.to_dict(orient="records"),
        },
        "artifacts": sorted(p.name for p in output_dir.iterdir()),
    }
    save_json(output_dir / "summary.json", summary)

    top_mi_row = action_mi_ranking_df.iloc[0]
    best_probe_row = probe_summary_df.iloc[0]
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
        f"- Unique sequences: `{summary['id_sequences']['unique_sequences']}`",
        f"- Singleton sequences: `{summary['id_sequences']['singleton_sequences']}`",
        f"- Max sequence usage: `{summary['id_sequences']['max_sequence_usage']}`",
        f"- Median sequence usage: `{summary['id_sequences']['median_sequence_usage']:.2f}`",
        "",
        "## Continuous",
        f"- Exact unique rows: `{summary['continuous']['exact_unique_rows']}`",
        f"- Rounded unique rows ({args.rounded_decimals} decimals): `{summary['continuous']['rounded_unique_rows']}`",
        f"- Exact max usage: `{summary['continuous']['exact_max_usage']}`",
        "",
        "## Codebook Vectors",
        f"- Exact unique rows: `{summary['codebook_vectors']['exact_unique_rows']}`",
        f"- Rounded unique rows ({args.rounded_decimals} decimals): `{summary['codebook_vectors']['rounded_unique_rows']}`",
        f"- Exact max usage: `{summary['codebook_vectors']['exact_max_usage']}`",
        "",
        "## Action Mutual Information",
        f"- Top MI pair: `{top_mi_row['feature']}` vs `{top_mi_row['target']}` dim `{int(top_mi_row['action_dim'])}` with MI `{top_mi_row['mi']:.4f}` and NMI `{top_mi_row['nmi']:.4f}`",
        "",
        "## Action Probes",
        f"- Best mean held-out R^2: `{best_probe_row['feature_set']}` -> `{best_probe_row['target']}` = `{best_probe_row['mean_r2']:.4f}`",
        "",
        "## Scatter Plot Coloring",
        "- PCA plots are colored by `log10(ID sequence usage)` for the corresponding frame.",
    ]
    (output_dir / "README.md").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
