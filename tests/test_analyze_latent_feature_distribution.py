import numpy as np
import pandas as pd

from conftest import load_script_module


analysis = load_script_module("analyze_latent_feature_distribution.py")


def test_extract_valid_episode_index_matches_valid_rows():
    valid = np.array([1, 1, 0, 1, 1, 1, 0], dtype=np.int64)
    episode_index = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    result = analysis.extract_valid_episode_index(valid, episode_index)

    assert result.tolist() == [0, 0, 1, 1, 1]


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
