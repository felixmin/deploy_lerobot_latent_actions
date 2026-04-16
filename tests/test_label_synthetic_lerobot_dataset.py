import numpy as np
import pytest

from conftest import load_script_module


synthetic_label_export = load_script_module("label_synthetic_lerobot_dataset.py")


def test_parse_label_spec_accepts_base_and_episode_groups():
    spec = synthetic_label_export.parse_label_spec(
        '{"base":{"mode":"constant","constant_value":0.0},"episode_groups":[{"episodes":[1,3],"mode":"constant","constant_value":5.0}]}'
    )

    assert spec["base"] == {"mode": "constant", "constant_value": 0.0}
    assert spec["episode_groups"] == [
        {"episodes": [1, 3], "mode": "constant", "constant_value": 5.0}
    ]


def test_parse_label_spec_accepts_yaml_style_inline_object():
    spec = synthetic_label_export.parse_label_spec(
        "{base: {mode: gaussian, mean: 0.0, std: 1.0}, episode_groups: []}"
    )

    assert spec["base"] == {"mode": "gaussian", "mean": 0.0, "std": 1.0}
    assert spec["episode_groups"] == []


def test_parse_label_spec_rejects_overlapping_episode_groups():
    with pytest.raises(ValueError, match="assigned to multiple groups"):
        synthetic_label_export.parse_label_spec(
            '{"base":{"mode":"constant","constant_value":0.0},"episode_groups":[{"episodes":[1,2],"mode":"constant","constant_value":1.0},{"episodes":[2,3],"mode":"constant","constant_value":2.0}]}'
        )


def test_generate_synthetic_labels_applies_group_override():
    label_spec = synthetic_label_export.parse_label_spec(
        '{"base":{"mode":"constant","constant_value":0.0},"episode_groups":[{"episodes":[1],"mode":"constant","constant_value":7.0}]}'
    )
    row_indices = np.array([10, 11, 12, 13], dtype=np.int64)
    episode_indices = np.array([0, 1, 1, 2], dtype=np.int64)

    labels = synthetic_label_export.generate_synthetic_labels(
        row_indices=row_indices,
        episode_indices=episode_indices,
        label_shape=(2, 3),
        label_dtype=np.dtype("float32"),
        label_spec=label_spec,
        rng=np.random.default_rng(0),
    )

    assert labels.shape == (4, 2, 3)
    assert np.all(labels[0] == 0.0)
    assert np.all(labels[1] == 7.0)
    assert np.all(labels[2] == 7.0)
    assert np.all(labels[3] == 0.0)


def test_generate_synthetic_labels_gaussian_is_seeded_and_shaped():
    label_spec = synthetic_label_export.parse_label_spec(
        '{"base":{"mode":"gaussian","mean":1.5,"std":0.0},"episode_groups":[]}'
    )

    labels = synthetic_label_export.generate_synthetic_labels(
        row_indices=np.array([0, 1, 2], dtype=np.int64),
        episode_indices=np.array([0, 0, 1], dtype=np.int64),
        label_shape=(4,),
        label_dtype=np.dtype("float32"),
        label_spec=label_spec,
        rng=np.random.default_rng(123),
    )

    assert labels.shape == (3, 4)
    assert np.all(labels == np.float32(1.5))


def test_generate_synthetic_labels_uniform_respects_range_and_override():
    label_spec = synthetic_label_export.parse_label_spec(
        '{"base":{"mode":"uniform","low":-0.5,"high":0.5},"episode_groups":[{"episodes":[1],"mode":"uniform","low":0.9,"high":1.1}]}'
    )

    labels = synthetic_label_export.generate_synthetic_labels(
        row_indices=np.array([0, 1, 2, 3], dtype=np.int64),
        episode_indices=np.array([0, 1, 1, 2], dtype=np.int64),
        label_shape=(8,),
        label_dtype=np.dtype("float32"),
        label_spec=label_spec,
        rng=np.random.default_rng(123),
    )

    assert labels.shape == (4, 8)
    assert np.all(labels[0] >= np.float32(-0.5))
    assert np.all(labels[0] < np.float32(0.5))
    assert np.all(labels[1] >= np.float32(0.9))
    assert np.all(labels[1] < np.float32(1.1))
    assert np.all(labels[2] >= np.float32(0.9))
    assert np.all(labels[2] < np.float32(1.1))
    assert np.all(labels[3] >= np.float32(-0.5))
    assert np.all(labels[3] < np.float32(0.5))


def test_compute_float_feature_stats_uses_only_valid_rows():
    label_arrays = {
        "continuous_vector_latents": np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [100.0, 100.0, 100.0, 100.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        )
    }
    valid_supervision = np.array([[1], [0], [1]], dtype=np.int64)
    feature_infos = {
        "continuous_vector_latents": {"dtype": "float32", "shape": (4,), "names": None},
    }

    stats = synthetic_label_export._compute_float_feature_stats(
        label_arrays=label_arrays,
        valid_supervision=valid_supervision,
        feature_infos=feature_infos,
    )

    assert set(stats.keys()) == {"continuous_vector_latents"}
    expected_mean = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    assert np.allclose(stats["continuous_vector_latents"]["mean"], expected_mean)
    assert stats["continuous_vector_latents"]["count"].tolist() == [2]
