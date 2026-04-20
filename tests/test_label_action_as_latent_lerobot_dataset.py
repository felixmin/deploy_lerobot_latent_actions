import numpy as np
import torch

from conftest import load_script_module


action_label_export = load_script_module("label_action_as_latent_lerobot_dataset.py")


def test_validate_default_output_repo_id_uses_sum_suffix():
    cfg = action_label_export.ActionAsLatentConfig(
        dataset_repo_id="local/libero",
        output_dir=action_label_export.Path("/tmp/out"),
        output_repo_id=None,
        horizon_frames=10,
    )

    cfg.validate()

    assert cfg.output_repo_id == "local/libero_action_as_latent_sum_h10"


def test_infer_output_shape_uses_single_action_vector():
    assert action_label_export._infer_output_shape(action_dim=7) == (7,)


def test_infer_valid_shape_uses_singleton_validity_vector():
    assert action_label_export._infer_valid_shape() == (1,)


def test_action_delta_timestamps_respect_start_offset():
    assert action_label_export._action_delta_timestamps(
        horizon_frames=3, fps=10.0
    ) == {"action": [0.0, 0.1, 0.2]}


def test_reduce_action_window_batch_sums_windows_and_invalidates_padded_rows():
    actions = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2)
    action_is_pad = torch.tensor(
        [
            [False, False],
            [False, False],
            [False, True],
        ],
        dtype=torch.bool,
    )
    targets, valid = action_label_export._reduce_action_window_batch(
        action_windows=actions,
        action_is_pad=action_is_pad,
    )
    assert targets.shape == (3, 2)
    assert valid.tolist() == [1, 1, 0]
    assert targets[0].tolist() == [2.0, 4.0]
    assert targets[1].tolist() == [10.0, 12.0]


def test_reduce_action_window_batch_accepts_missing_padding_mask():
    actions = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    targets, valid = action_label_export._reduce_action_window_batch(
        action_windows=actions,
        action_is_pad=None,
    )
    assert valid.tolist() == [1, 1]
    assert targets.tolist() == [[6.0, 9.0], [24.0, 27.0]]


def test_compute_float_feature_stats_pools_across_valid_rows():
    label_array = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ],
        dtype=np.float32,
    )
    valid_supervision = np.array([1, 0, 1, 1, 1], dtype=np.int64)

    stats = action_label_export._compute_float_feature_stats(
        label_array=label_array,
        valid_supervision=valid_supervision,
    )

    expected_values = np.array(
        [
            [1.0, 10.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ],
        dtype=np.float64,
    )
    assert stats["count"].tolist() == [4]
    assert stats["mean"].shape == (2,)
    assert np.allclose(stats["min"], expected_values.min(axis=0))
    assert np.allclose(stats["max"], expected_values.max(axis=0))
    assert np.allclose(stats["mean"], expected_values.mean(axis=0))
    assert np.allclose(stats["std"], expected_values.std(axis=0, ddof=0))


def test_compute_float_feature_stats_returns_empty_without_valid_steps():
    label_array = np.zeros((2, 4), dtype=np.float32)
    valid_supervision = np.zeros((2,), dtype=np.int64)

    assert (
        action_label_export._compute_float_feature_stats(
            label_array=label_array,
            valid_supervision=valid_supervision,
        )
        == {}
    )


def test_compute_float_feature_stats_accepts_singleton_validity_vectors():
    label_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    valid_supervision = np.array([[1], [0]], dtype=np.int64)

    stats = action_label_export._compute_float_feature_stats(
        label_array=label_array,
        valid_supervision=valid_supervision,
    )

    assert stats["count"].tolist() == [1]
    assert stats["mean"].tolist() == [1.0, 2.0]
