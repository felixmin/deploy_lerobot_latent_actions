import json
from pathlib import Path

from conftest import load_script_module


artifact_registry = load_script_module("_artifact_registry.py")


def test_infer_checkpoint_metadata_for_live_run_layout():
    meta = artifact_registry.infer_checkpoint_metadata(
        "/mnt/data/workspace/runs_root/runs/2026-04-04_02-16-17_lam_contrastive/lerobot/checkpoints/070000/pretrained_model"
    )

    assert meta["source_checkpoint_step"] == "070000"
    assert meta["source_run_name"] == "2026-04-04_02-16-17_lam_contrastive"
    assert meta["source_checkpoint_id"] == "2026-04-04_02-16-17_lam_contrastive__070000"


def test_infer_checkpoint_metadata_for_archive_layout():
    meta = artifact_registry.infer_checkpoint_metadata(
        "/mnt/data/workspace/runs_root/runs_lerobot/output_archive/2026-04-05/38573134_lam_plain_100k_bs256/checkpoints/010000/pretrained_model"
    )

    assert meta["source_checkpoint_step"] == "010000"
    assert meta["source_run_name"] == "38573134_lam_plain_100k_bs256"
    assert meta["source_checkpoint_id"] == "38573134_lam_plain_100k_bs256__010000"


def test_register_artifact_writes_manifest_and_global_registry(tmp_path):
    runs_root = tmp_path / "runs_root" / "runs_lerobot"
    output_dir = runs_root / "outputs" / "analysis" / "latent_core_v1" / "toy"
    manifest_path = output_dir / "analysis_manifest.json"

    payload = artifact_registry.register_artifact(
        manifest_path=manifest_path,
        manifest={
            "artifact_type": "latent_analysis",
            "suite_name": "latent_core",
            "suite_version": "v1",
            "artifact_id": "latent_core__v1__toy",
            "output_path": str(output_dir),
        },
        registry_candidates=[output_dir],
    )

    registry_path = runs_root / "analysis_registry" / "artifacts.jsonl"
    assert manifest_path.exists()
    assert registry_path.exists()

    manifest = json.loads(manifest_path.read_text())
    lines = registry_path.read_text().strip().splitlines()
    registry_row = json.loads(lines[-1])

    assert manifest["artifact_id"] == "latent_core__v1__toy"
    assert payload["registry_path"] == str(registry_path.resolve())
    assert registry_row["artifact_id"] == "latent_core__v1__toy"
    assert registry_row["manifest_path"] == str(manifest_path.resolve())
