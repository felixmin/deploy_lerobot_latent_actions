import numpy as np
import torch

from conftest import load_script_module


action_label_export = load_script_module("label_action_as_latent_lerobot_dataset.py")


def test_validate_default_output_repo_id_uses_sequence_suffix():
    cfg = action_label_export.ActionAsLatentConfig(
        dataset_repo_id="local/libero",
        output_dir=action_label_export.Path("/tmp/out"),
        output_repo_id=None,
        horizon_frames=10,
    )

    cfg.validate()

    assert cfg.output_repo_id == "local/libero_action_as_latent_sequence_h10"


def test_infer_output_shape_preserves_structured_sequence():
    assert action_label_export._infer_output_shape(horizon_frames=10, action_dim=7) == (10, 7)


def test_infer_valid_shape_uses_per_step_mask():
    assert action_label_export._infer_valid_shape(horizon_frames=10) == (10,)


def test_action_delta_timestamps_respect_start_offset():
    assert action_label_export._action_delta_timestamps(
        start_offset_frames=0, horizon_frames=3, fps=10.0
    ) == {"action": [0.0, 0.1, 0.2]}
    assert action_label_export._action_delta_timestamps(
        start_offset_frames=1, horizon_frames=3, fps=10.0
    ) == {"action": [0.1, 0.2, 0.3]}


def test_build_padded_action_windows_emits_prefix_validity():
    actions = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2).numpy()
    windows, valid = action_label_export._build_padded_action_windows(
        episode_actions=actions,
        anchor_indices=torch.tensor([3, 5]).numpy(),
        start_offset_frames=0,
        horizon_frames=4,
        invalid_fill_value=-1.0,
    )
    assert windows.shape == (2, 4, 2)
    assert valid.tolist() == [[1, 1, 1, 0], [1, 0, 0, 0]]
    assert windows[0, 0].tolist() == [6.0, 7.0]
    assert windows[0, 2].tolist() == [10.0, 11.0]
    assert windows[0, 3].tolist() == [-1.0, -1.0]
    assert windows[1, 0].tolist() == [10.0, 11.0]
    assert windows[1, 1].tolist() == [-1.0, -1.0]


def test_compute_float_feature_stats_pools_across_all_valid_steps():
    label_array = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [-1.0, -1.0]],
            [[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
        ],
        dtype=np.float32,
    )
    valid_supervision = np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )

    stats = action_label_export._compute_float_feature_stats(
        label_array=label_array,
        valid_supervision=valid_supervision,
    )

    expected_values = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ],
        dtype=np.float64,
    )
    assert stats["count"].tolist() == [5]
    assert stats["mean"].shape == (2,)
    assert np.allclose(stats["min"], expected_values.min(axis=0))
    assert np.allclose(stats["max"], expected_values.max(axis=0))
    assert np.allclose(stats["mean"], expected_values.mean(axis=0))
    assert np.allclose(stats["std"], expected_values.std(axis=0, ddof=0))


def test_compute_float_feature_stats_returns_empty_without_valid_steps():
    label_array = np.zeros((2, 3, 4), dtype=np.float32)
    valid_supervision = np.zeros((2, 3), dtype=np.int64)

    assert (
        action_label_export._compute_float_feature_stats(
            label_array=label_array,
            valid_supervision=valid_supervision,
        )
        == {}
    )
