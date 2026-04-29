from pathlib import Path

import pandas as pd
import torch

from conftest import load_script_module


recon_analysis = load_script_module("analyze_latent_reconstructions.py")


def test_squeeze_single_frame_removes_time_dimension():
    frames = torch.arange(2 * 3 * 1 * 4 * 5, dtype=torch.float32).reshape(2, 3, 1, 4, 5)

    squeezed = recon_analysis.squeeze_single_frame(frames)

    assert tuple(squeezed.shape) == (2, 3, 4, 5)
    assert torch.equal(squeezed[:, :, 0, 0], frames[:, :, 0, 0, 0])


def test_compute_reconstruction_metrics_reports_zero_for_exact_match():
    target = torch.ones(2, 3, 1, 2, 2, dtype=torch.float32)
    reconstructions = {
        "ids": target.clone(),
        "continuous": target.clone() * 0.5,
    }

    rows = recon_analysis.compute_reconstruction_metrics(target, reconstructions)
    metrics_df = pd.DataFrame(rows)

    ids_rows = metrics_df[metrics_df["latent_format"] == "ids"]
    assert ids_rows.shape[0] == 2
    assert float(ids_rows["mse"].max()) == 0.0
    assert float(ids_rows["mae"].max()) == 0.0

    continuous_rows = metrics_df[metrics_df["latent_format"] == "continuous"]
    assert abs(float(continuous_rows.iloc[0]["mse"]) - 0.25) < 1e-8
    assert abs(float(continuous_rows.iloc[0]["mae"]) - 0.5) < 1e-8


def test_summarize_reconstruction_metrics_aggregates_per_format():
    metrics_df = pd.DataFrame(
        {
            "latent_format": ["ids", "ids", "continuous"],
            "mse": [0.0, 1.0, 0.25],
            "mae": [0.0, 1.0, 0.5],
            "psnr_db": [60.0, 0.0, 6.0],
        }
    )

    summary_df = recon_analysis.summarize_reconstruction_metrics(metrics_df)

    ids_row = summary_df[summary_df["latent_format"] == "ids"].iloc[0]
    assert abs(float(ids_row["mean_mse"]) - 0.5) < 1e-8
    assert abs(float(ids_row["median_mae"]) - 0.5) < 1e-8


def test_reservoir_update_keeps_capacity():
    rng = torch.Generator().manual_seed(0)
    np_rng = recon_analysis.np.random.default_rng(0)
    del rng
    reservoir: list[dict[str, int]] = []

    for seen_count in range(1, 11):
        recon_analysis.reservoir_update(
            reservoir,
            candidate={"value": seen_count},
            seen_count=seen_count,
            capacity=3,
            rng=np_rng,
        )

    assert len(reservoir) == 3
    assert all("value" in row for row in reservoir)


def test_render_reconstruction_grid_writes_png(tmp_path: Path):
    sample = {
        "index": 7,
        "episode_index": 1,
        "frame_index": 5,
        "first_frame": torch.zeros(3, 8, 8),
        "target_frame": torch.ones(3, 8, 8),
        "reconstructions": {
            "ids": torch.full((3, 8, 8), 0.25),
            "continuous": torch.full((3, 8, 8), 0.75),
        },
    }

    output_path = tmp_path / "grid.png"
    recon_analysis.render_reconstruction_grid([sample], ["ids", "continuous"], output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
