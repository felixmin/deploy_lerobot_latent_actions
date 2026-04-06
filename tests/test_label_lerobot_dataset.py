import numpy as np

from conftest import load_script_module


label_export = load_script_module("label_lerobot_dataset.py")


def test_compute_float_feature_stats_uses_only_valid_rows():
    label_arrays = {
        "continuous_vector_latents": np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [100.0, 100.0, 100.0, 100.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        ),
        "codebook_id_latents": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64),
    }
    valid_supervision = np.array([[1], [0], [1]], dtype=np.int64)
    feature_infos = {
        "continuous_vector_latents": {"dtype": "float32", "shape": (4,), "names": None},
        "codebook_id_latents": {"dtype": "int64", "shape": (2,), "names": None},
    }

    stats = label_export._compute_float_feature_stats(
        label_arrays=label_arrays,
        valid_supervision=valid_supervision,
        feature_infos=feature_infos,
    )

    assert set(stats.keys()) == {"continuous_vector_latents"}
    expected_mean = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    assert np.allclose(stats["continuous_vector_latents"]["mean"], expected_mean)
    assert stats["continuous_vector_latents"]["count"].tolist() == [2]
