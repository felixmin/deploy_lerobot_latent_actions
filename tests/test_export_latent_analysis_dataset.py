import numpy as np
import pyarrow.dataset as ds

from conftest import load_script_module


analysis_export = load_script_module("export_latent_analysis_dataset.py")


def test_write_analysis_dataset_root_is_analyzer_compatible(tmp_path):
    source_info = {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "fps": 10.0,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "features": {
            "action": {"dtype": "float32", "shape": [7], "names": ["a"] * 7, "fps": 10.0},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": 10.0},
            "index": {"dtype": "int64", "shape": [1], "names": None, "fps": 10.0},
        },
    }
    plan = {
        "representations": {
            "continuous_vector_latents": {
                "shape": (4,),
                "dtype": np.dtype("float32"),
                "invalid_fill_value": 0.0,
            }
        }
    }
    output_features = analysis_export._build_output_features(
        source_info=source_info,
        plan=plan,
        feature_prefix="latent_labels",
        passthrough_keys=["action", "episode_index", "index"],
        fps=10.0,
    )
    output_info = analysis_export._make_output_info(
        source_info=source_info,
        output_repo_id="toy/latent-analysis",
        output_features=output_features,
        total_frames=3,
        total_episodes=2,
        total_tasks=0,
    )
    data_columns = {
        "action": np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "episode_index": np.array([0, 0, 1], dtype=np.int64),
        "index": np.array([5, 6, 7], dtype=np.int64),
        "latent_labels.continuous_vector_latents": np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.6, 0.7, 0.8],
            ],
            dtype=np.float32,
        ),
        "latent_labels.valid": np.array([1, 0, 1], dtype=np.int64),
    }

    analysis_export._write_analysis_dataset_root(
        output_dir=tmp_path,
        info=output_info,
        data_columns=data_columns,
    )

    dataset = ds.dataset(tmp_path / "data", format="parquet")
    table = dataset.to_table(
        columns=[
            "action",
            "episode_index",
            "index",
            "latent_labels.continuous_vector_latents",
            "latent_labels.valid",
        ]
    )

    assert "latent_labels.continuous_vector_latents" in output_info["features"]
    assert table.num_rows == 3
    assert table["latent_labels.valid"].to_pylist() == [1, 0, 1]
    assert table["episode_index"].to_pylist() == [0, 0, 1]
    assert table["index"].to_pylist() == [5, 6, 7]

    actions = np.stack(table["action"].to_numpy(zero_copy_only=False)).astype(np.float32, copy=False)
    continuous_obj = table["latent_labels.continuous_vector_latents"].to_numpy(zero_copy_only=False)
    continuous = np.stack([np.asarray(row, dtype=np.float32) for row in continuous_obj], axis=0)
    valid_mask = table["latent_labels.valid"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

    assert actions.shape == (3, 7)
    assert continuous.shape == (3, 4)
    assert valid_mask.tolist() == [1, 0, 1]
    assert np.allclose(continuous[0], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    assert np.allclose(continuous[2], np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32))
