import torch

from conftest import load_script_module


action_label_export = load_script_module("label_action_as_latent_lerobot_dataset.py")


def test_infer_output_shape_flatten():
    assert action_label_export._infer_output_shape(
        aggregation_mode="flatten", horizon_frames=10, action_dim=7
    ) == (10, 7)


def test_infer_output_shape_flatten_vector():
    assert action_label_export._infer_output_shape(
        aggregation_mode="flatten_vector", horizon_frames=10, action_dim=7
    ) == (70,)


def test_infer_valid_shape_uses_per_step_mask_for_sequence_modes():
    assert action_label_export._infer_valid_shape(
        aggregation_mode="flatten", horizon_frames=10
    ) == (10,)
    assert action_label_export._infer_valid_shape(
        aggregation_mode="flatten_vector", horizon_frames=10
    ) == (10,)
    assert action_label_export._infer_valid_shape(
        aggregation_mode="sum_motion_gripper_final", horizon_frames=10
    ) == (1,)


def test_action_delta_timestamps_respect_start_offset():
    assert action_label_export._action_delta_timestamps(
        start_offset_frames=0, horizon_frames=3, fps=10.0
    ) == {"action": [0.0, 0.1, 0.2]}
    assert action_label_export._action_delta_timestamps(
        start_offset_frames=1, horizon_frames=3, fps=10.0
    ) == {"action": [0.1, 0.2, 0.3]}


def test_compute_valid_count_matches_expected_tail():
    assert action_label_export._compute_valid_count(
        episode_length=20, start_offset_frames=0, horizon_frames=10
    ) == 11
    assert action_label_export._compute_valid_count(
        episode_length=20, start_offset_frames=1, horizon_frames=10
    ) == 10


def test_build_episode_batch_valid_mask_marks_tail_invalid():
    out = action_label_export._build_episode_batch_valid_mask(
        batch_start=8, batch_end=13, valid_count=11
    )
    assert out.tolist() == [True, True, True, False, False]


def test_infer_output_shape_single_vector_modes():
    for mode in ["sum", "mean", "last", "sum_motion_gripper_final", "sum_motion_gripper_mean"]:
        assert action_label_export._infer_output_shape(
            aggregation_mode=mode, horizon_frames=10, action_dim=7
        ) == (7,)


def test_aggregate_actions_flatten_preserves_structure():
    actions = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    out = action_label_export._aggregate_actions(actions, "flatten")
    assert out.shape == (2, 3, 4)
    assert torch.equal(out, actions)


def test_aggregate_actions_flatten_vector_flattens_structure():
    actions = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    out = action_label_export._aggregate_actions(actions, "flatten_vector")
    assert out.shape == (2, 12)
    assert torch.equal(out, actions.reshape(2, 12))


def test_aggregate_actions_sum():
    actions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        ]
    )
    out = action_label_export._aggregate_actions(actions, "sum")
    assert torch.allclose(out, torch.tensor([[9.0, 12.0]]))


def test_aggregate_actions_sum_motion_gripper_final():
    actions = torch.tensor(
        [
            [
                [1.0, 10.0, -1.0],
                [2.0, 20.0, 0.0],
                [3.0, 30.0, 1.0],
            ]
        ]
    )
    out = action_label_export._aggregate_actions(actions, "sum_motion_gripper_final")
    assert torch.allclose(out, torch.tensor([[6.0, 60.0, 1.0]]))


def test_aggregate_actions_sum_motion_gripper_mean():
    actions = torch.tensor(
        [
            [
                [1.0, 10.0, -1.0],
                [2.0, 20.0, 0.0],
                [3.0, 30.0, 1.0],
            ]
        ]
    )
    out = action_label_export._aggregate_actions(actions, "sum_motion_gripper_mean")
    assert torch.allclose(out, torch.tensor([[6.0, 60.0, 0.0]]))


def test_build_action_windows_respects_anchor_indices():
    actions = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2).numpy()
    out = action_label_export._build_action_windows(
        episode_actions=actions,
        anchor_indices=torch.tensor([0, 2]).numpy(),
        start_offset_frames=1,
        horizon_frames=2,
    )
    expected = torch.tensor(
        [
            [[2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0]],
        ]
    ).numpy()
    assert out.shape == (2, 2, 2)
    assert (out == expected).all()


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
