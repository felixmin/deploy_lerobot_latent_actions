import numpy as np
import pandas as pd
import pytest

from conftest import load_script_module


spcfc = load_script_module("analyze_spcfc.py")


def test_expand_valid_rows_reconstructs_batch_shape():
    compact = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    valid_mask = np.array([True, False, True], dtype=bool)

    expanded = spcfc.expand_valid_rows(compact, valid_mask, full_shape=(2,), fill_value=0.0)

    assert expanded.shape == (3, 2)
    assert np.allclose(expanded[0], [1.0, 2.0])
    assert np.allclose(expanded[1], [0.0, 0.0])
    assert np.allclose(expanded[2], [3.0, 4.0])


def test_compute_spcfc_scores_matches_cosine_similarity():
    past = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    future = np.array([[1.0, 0.0], [1.0, -1.0]], dtype=np.float32)

    scores = spcfc.compute_spcfc_scores(past, future)

    assert np.allclose(scores[0], 1.0)
    assert np.allclose(scores[1], 0.0, atol=1e-6)


def test_iterate_aligned_batches_raises_on_mismatch():
    past_loader = [{"index": np.array([0, 1]), "episode_index": np.array([0, 0])}]
    future_loader = [{"index": np.array([0, 2]), "episode_index": np.array([0, 0])}]

    with pytest.raises(ValueError, match="index"):
        list(spcfc.iterate_aligned_batches(past_loader, future_loader))


def test_summarize_spcfc_returns_global_and_per_episode_tables():
    scores_df = pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            "episode_index": [0, 0, 1, 1],
            "latent_format": ["continuous", "continuous", "codebook_vectors", "codebook_vectors"],
            "spcfc": [0.1, 0.3, 0.5, 0.7],
        }
    )

    summary_df, per_episode_df = spcfc.summarize_spcfc(scores_df)

    assert {"latent_format", "count", "mean", "std", "min", "p01", "p05", "median", "p95", "p99", "max"} == set(
        summary_df.columns
    )
    assert {"latent_format", "episode_index", "count", "mean", "std", "min", "median", "max"} == set(
        per_episode_df.columns
    )
    continuous_row = summary_df[summary_df["latent_format"] == "continuous"].iloc[0]
    assert abs(float(continuous_row["mean"]) - 0.2) < 1e-6
