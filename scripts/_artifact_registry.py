#!/usr/bin/env python

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import fcntl


ARTIFACT_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize_token(raw: str | None) -> str:
    if raw is None:
        return "unknown"
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    token = token.strip("._-")
    return token or "unknown"


def infer_checkpoint_metadata(policy_path: str | Path | None) -> dict[str, Any]:
    if policy_path is None:
        return {
            "source_checkpoint_id": None,
            "source_checkpoint_path": None,
            "source_checkpoint_dir": None,
            "source_checkpoint_step": None,
            "source_run_name": None,
            "source_run_path": None,
        }

    resolved = Path(policy_path).resolve()
    checkpoint_dir = resolved.parent if resolved.name == "pretrained_model" else resolved
    checkpoint_step = checkpoint_dir.name if checkpoint_dir.name.isdigit() else None

    run_dir: Path | None = None
    if checkpoint_dir.parent.name == "checkpoints":
        run_container = checkpoint_dir.parent.parent
        run_dir = run_container.parent if run_container.name == "lerobot" else run_container

    run_name = run_dir.name if run_dir is not None else None
    checkpoint_id_parts = []
    if run_name is not None:
        checkpoint_id_parts.append(sanitize_token(run_name))
    if checkpoint_step is not None:
        checkpoint_id_parts.append(sanitize_token(checkpoint_step))
    source_checkpoint_id = "__".join(checkpoint_id_parts) if checkpoint_id_parts else sanitize_token(checkpoint_dir.name)

    return {
        "source_checkpoint_id": source_checkpoint_id,
        "source_checkpoint_path": str(resolved),
        "source_checkpoint_dir": str(checkpoint_dir),
        "source_checkpoint_step": checkpoint_step,
        "source_run_name": run_name,
        "source_run_path": None if run_dir is None else str(run_dir),
    }


def make_artifact_id(
    *,
    suite_name: str,
    suite_version: str,
    checkpoint_id: str | None,
    output_label: str | None,
) -> str:
    parts = [sanitize_token(suite_name), sanitize_token(suite_version)]
    if checkpoint_id:
        parts.append(sanitize_token(checkpoint_id))
    if output_label:
        parts.append(sanitize_token(output_label))
    return "__".join(parts)


def find_runs_lerobot_root(*paths: str | Path | None) -> Path | None:
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path).resolve()
        candidates = (path, *path.parents)
        for candidate in candidates:
            if candidate.name == "runs_lerobot":
                return candidate
    return None


def resolve_registry_path(*paths: str | Path | None) -> Path | None:
    env_path = os.environ.get("LEROBOT_ANALYSIS_REGISTRY_PATH")
    if env_path:
        return Path(env_path).resolve()

    env_root = os.environ.get("LEROBOT_ANALYSIS_REGISTRY_ROOT")
    if env_root:
        return (Path(env_root).resolve() / "artifacts.jsonl").resolve()

    runs_lerobot_root = find_runs_lerobot_root(*paths)
    if runs_lerobot_root is None:
        return None
    return runs_lerobot_root / "analysis_registry" / "artifacts.jsonl"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(payload, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def register_artifact(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    registry_candidates: Sequence[str | Path | None],
) -> dict[str, Any]:
    payload = dict(manifest)
    payload.setdefault("artifact_schema_version", ARTIFACT_SCHEMA_VERSION)
    payload.setdefault("created_at", utc_now_iso())
    payload["manifest_path"] = str(manifest_path.resolve())

    registry_path = resolve_registry_path(manifest_path, *registry_candidates)
    payload["registry_path"] = None if registry_path is None else str(registry_path.resolve())

    write_json(manifest_path, payload)
    if registry_path is not None:
        append_jsonl(registry_path, payload)
    return payload


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_export_manifest(dataset_root: str | Path) -> dict[str, Any] | None:
    root = Path(dataset_root).resolve()
    for name in ("export_manifest.json", "label_manifest.json"):
        path = root / name
        if path.exists():
            payload = load_json(path)
            payload.setdefault("manifest_path", str(path))
            return payload
    return None
