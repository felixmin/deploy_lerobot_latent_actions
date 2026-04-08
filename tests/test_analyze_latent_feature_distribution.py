import numpy as np
import pandas as pd

from conftest import load_script_module


analysis = load_script_module("analyze_latent_feature_distribution.py")


def test_extract_valid_episode_index_matches_valid_rows():
    valid = np.array([1, 1, 0, 1, 1, 1, 0], dtype=np.int64)
    episode_index = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    result = analysis.extract_valid_episode_index(valid, episode_index)

    assert result.tolist() == [0, 0, 1, 1, 1]


def test_extract_valid_progress_context_tracks_position_and_episode_length():
    valid = np.array([1, 1, 0, 1, 1, 1, 0], dtype=np.int64)
    episode_index = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    progress_index, episode_lengths = analysis.extract_valid_progress_context(valid, episode_index)

    assert progress_index.tolist() == [0, 1, 0, 1, 2]
    assert episode_lengths.tolist() == [2, 2, 3, 3, 3]


def test_derive_action_targets_truncates_to_valid_prefix():
    actions = np.arange(10, dtype=np.float32).reshape(5, 2)
    valid = np.array([1, 1, 0, 0, 0], dtype=np.int64)
    episode_index = np.zeros(5, dtype=np.int64)

    targets = analysis.derive_action_targets(actions, valid, episode_index, future_frames=2)

    assert targets["current_action"].shape == (2, 2)
    assert targets["future_action_mean"].shape == (2, 2)


def test_unique_rows_handles_single_slot_id_arrays():
    ids = np.array([7, 7, 3, 7, 3], dtype=np.int64)

    unique_count, inverse, counts = analysis.unique_rows(ids)

    assert unique_count == 2
    assert inverse.shape == (5,)
    assert sorted(counts.tolist()) == [2, 3]


def test_make_probe_split_episode_mode_keeps_episodes_disjoint():
    valid_episode_index = np.repeat(np.arange(6, dtype=np.int64), 8)

    train_rows, test_rows = analysis.make_probe_split(
        valid_episode_index,
        max_samples=32,
        test_size=0.25,
        seed=0,
        mode="episode",
    )

    train_eps = set(valid_episode_index[train_rows].tolist())
    test_eps = set(valid_episode_index[test_rows].tolist())
    assert train_eps
    assert test_eps
    assert train_eps.isdisjoint(test_eps)


def test_build_probe_feature_sets_accepts_continuous_only():
    continuous = np.random.default_rng(0).normal(size=(12, 32)).astype(np.float32)

    feature_sets = analysis.build_probe_feature_sets(
        ids=None,
        codebook_vectors_flat=None,
        continuous_flat=continuous,
    )

    assert set(feature_sets.keys()) == {"continuous"}
    assert feature_sets["continuous"].shape == (12, 32)


def test_summarize_norms_handles_single_slot_vectors():
    values = np.arange(24, dtype=np.float32).reshape(6, 4)

    summary_df = analysis.summarize_norms(values)

    assert summary_df["slot_index"].tolist() == [0]
    assert float(summary_df["mean"].iloc[0]) > 0.0


def test_make_discrete_bucket_spec_uses_full_id_sequences():
    ids = np.array([[1, 2], [1, 2], [3, 4], [3, 4], [9, 9]], dtype=np.int64)
    _, id_inverse, _ = analysis.unique_rows(ids)

    bucket_spec = analysis.make_discrete_bucket_spec(ids, id_inverse)

    assert bucket_spec["feature_set"] == "id_sequence"
    assert bucket_spec["bucket_kind"] == "discrete_sequence"
    assert bucket_spec["bucket_counts"].tolist() == [2, 2, 1]
    assert bucket_spec["bucket_names"].tolist() == ["1 2", "3 4", "9 9"]


def test_compute_bucket_action_statistics_recovers_perfect_bucket_means():
    bucket_index = np.array([0, 0, 1, 1], dtype=np.int64)
    bucket_names = np.array(["left", "right"], dtype=object)
    target_values = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )

    stats_df, summary = analysis.compute_bucket_action_statistics(
        bucket_index=bucket_index,
        bucket_names=bucket_names,
        target_values=target_values,
    )

    assert stats_df["count"].tolist() == [2, 2]
    assert stats_df["bucket_label"].tolist() == ["left", "right"]
    assert abs(float(summary["mean_variance_explained"]) - 1.0) < 1e-6
    assert abs(float(summary["mean_within_bucket_std"])) < 1e-6


def test_compute_bucket_context_statistics_distinguishes_reused_vs_episode_local_buckets():
    bucket_index = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    bucket_names = np.array(["reused", "episode_local"], dtype=object)
    valid_episode_index = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    valid_frame_index = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    valid_progress_index = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    valid_episode_lengths = np.array([4, 4, 4, 4, 4, 4, 4, 4], dtype=np.int64)

    stats_df, summary = analysis.compute_bucket_context_statistics(
        bucket_index=bucket_index,
        bucket_names=bucket_names,
        valid_episode_index=valid_episode_index,
        valid_frame_index=valid_frame_index,
        valid_progress_index=valid_progress_index,
        valid_episode_lengths=valid_episode_lengths,
        progress_bins=4,
    )

    reused = stats_df[stats_df["bucket_label"] == "reused"].iloc[0]
    local = stats_df[stats_df["bucket_label"] == "episode_local"].iloc[0]

    assert float(reused["episode_coverage"]) == 1.0
    assert float(local["episode_coverage"]) == 0.5
    assert abs(float(reused["max_episode_fraction"]) - (4.0 / 6.0)) < 1e-6
    assert float(local["max_episode_fraction"]) == 1.0
    assert float(reused["mean_run_length"]) == 3.0
    assert float(local["mean_run_length"]) == 2.0
    assert float(reused["progress_bin_coverage"]) == 1.0
    assert float(local["progress_bin_coverage"]) == 0.5
    assert 0.0 <= float(summary["bucket_episode_nmi"]) <= 1.0


def test_make_continuous_bucket_spec_assigns_two_clusters():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=-3.0, scale=0.1, size=(24, 4)).astype(np.float32)
    cluster_b = rng.normal(loc=3.0, scale=0.1, size=(24, 4)).astype(np.float32)
    continuous = np.concatenate([cluster_a, cluster_b], axis=0)

    bucket_spec = analysis.make_continuous_bucket_spec(
        continuous,
        n_clusters=2,
        fit_samples=48,
        seed=0,
    )

    counts = np.sort(bucket_spec["bucket_counts"])
    assert bucket_spec["feature_set"] == "continuous_kmeans"
    assert bucket_spec["bucket_kind"] == "kmeans"
    assert int(bucket_spec["effective_clusters"]) == 2
    assert counts.tolist() == [24, 24]


def test_fit_ridge_probe_returns_expected_schema():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 6)).astype(np.float32)
    y = np.stack(
        [
            2.0 * x[:, 0] - 0.5 * x[:, 1],
            1.5 * x[:, 2] + 0.25 * x[:, 3],
        ],
        axis=1,
    ).astype(np.float32)

    probe_df = analysis.fit_ridge_probe(
        {"continuous": x},
        {"current_action": y},
        train_rows=np.arange(0, 90, dtype=np.int64),
        test_rows=np.arange(90, 120, dtype=np.int64),
        ridge_alpha=1e-3,
        split_mode="row",
    )

    expected_columns = {
        "probe_model",
        "split_mode",
        "feature_set",
        "target",
        "action_dim",
        "r2",
        "avg_r2_for_target",
        "mse",
        "avg_mse_for_target",
        "n_train",
        "n_test",
    }
    assert expected_columns.issubset(set(probe_df.columns))
    assert set(probe_df["probe_model"]) == {"ridge"}
    assert float(probe_df["r2"].mean()) > 0.9


def test_fit_mlp_probe_returns_expected_schema():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(160, 8)).astype(np.float32)
    y = np.stack(
        [
            x[:, 0] * x[:, 1] + 0.2 * x[:, 2],
            np.tanh(x[:, 3] - x[:, 4]),
        ],
        axis=1,
    ).astype(np.float32)

    probe_df = analysis.fit_mlp_probe(
        {"continuous": x},
        {"future_action_mean": y},
        train_rows=np.arange(0, 120, dtype=np.int64),
        test_rows=np.arange(120, 160, dtype=np.int64),
        split_mode="row",
        hidden_layer_sizes=(32, 16),
        alpha=1e-4,
        max_iter=300,
        early_stopping=False,
        n_iter_no_change=10,
        seed=0,
    )

    assert set(probe_df["probe_model"]) == {"mlp"}
    assert set(probe_df["target"]) == {"future_action_mean"}
    assert float(probe_df["avg_mse_for_target"].iloc[0]) >= 0.0


def test_summarize_probe_scores_groups_by_model_and_split():
    probe_df = pd.DataFrame(
        [
            {"probe_model": "ridge", "split_mode": "row", "feature_set": "continuous", "target": "current_action", "action_dim": 0, "r2": 0.4, "mse": 0.2},
            {"probe_model": "ridge", "split_mode": "row", "feature_set": "continuous", "target": "current_action", "action_dim": 1, "r2": 0.6, "mse": 0.1},
            {"probe_model": "mlp", "split_mode": "episode", "feature_set": "continuous", "target": "current_action", "action_dim": 0, "r2": 0.5, "mse": 0.15},
        ]
    )

    summary_df = analysis.summarize_probe_scores(probe_df)

    assert {"probe_model", "split_mode", "feature_set", "target", "mean_r2", "mean_mse"} == set(summary_df.columns)
    ridge_row = summary_df[(summary_df["probe_model"] == "ridge") & (summary_df["split_mode"] == "row")].iloc[0]
    assert abs(float(ridge_row["mean_r2"]) - 0.5) < 1e-6
