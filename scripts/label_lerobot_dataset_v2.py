#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


LATENT_FORMAT_IDS = "ids"
LATENT_FORMAT_CONTINUOUS = "continuous"
LATENT_FORMAT_CODEBOOK_VECTORS = "codebook_vectors"
_PROGRESS_LOG_EVERY_BATCHES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-type", required=True, choices=["lapa_lam", "rlfv_lam"])
    parser.add_argument("--policy-path", required=True, help="Checkpoint dir for Policy.from_pretrained().")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root.")
    parser.add_argument(
        "--episodes",
        default=None,
        help="Optional JSON list of episode indices to keep, e.g. '[0,1]'.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output dataset directory for the relabeled dataset.",
    )
    parser.add_argument(
        "--output-repo-id",
        default=None,
        help="Repo id stored in the new dataset metadata. Defaults to '<input>_latent_labeled'.",
    )
    parser.add_argument("--camera-key", default=None, help="Override camera feature.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument(
        "--feature-name",
        default="latent_labels",
        help="Feature name for the generated labels.",
    )
    parser.add_argument(
        "--valid-feature-name",
        default="latent_supervised",
        help="Feature name for the generated supervision mask.",
    )
    parser.add_argument(
        "--plugin-src-path",
        action="append",
        default=[],
        help="Optional extra src path to prepend to PYTHONPATH-style resolution.",
    )
    parser.add_argument(
        "--latent-format",
        default=LATENT_FORMAT_IDS,
        choices=[
            LATENT_FORMAT_IDS,
            LATENT_FORMAT_CONTINUOUS,
            LATENT_FORMAT_CODEBOOK_VECTORS,
        ],
        help="Label representation to export.",
    )
    parser.add_argument(
        "--future-frames",
        type=int,
        default=None,
        help="LAM override for frame pairing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--max-valid-samples",
        type=int,
        default=None,
        help="Optional smoke-test cap on the number of valid samples to label. Remaining samples stay invalid.",
    )
    return parser.parse_args()


def _workspace_code_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_plugin_src(policy_type: str) -> Path:
    base = _workspace_code_dir()
    if policy_type == "lapa_lam":
        return base / "lerobot_policy_lapa_lam" / "src"
    if policy_type == "rlfv_lam":
        return base / "lerobot_policy_rlfv_lam" / "src"
    raise ValueError(f"Unsupported policy_type={policy_type!r}")


def _add_sys_path(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Required path does not exist: {resolved}")
    path_str = str(resolved)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _bootstrap_imports(policy_type: str, plugin_src_paths: list[str]) -> dict[str, Any]:
    workspace = _workspace_code_dir()
    _add_sys_path(workspace / "high-level-robot-planner" / "lerobot" / "src")
    _add_sys_path(_default_plugin_src(policy_type))
    for raw_path in plugin_src_paths:
        _add_sys_path(Path(raw_path))

    from lerobot.datasets.dataset_tools import add_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    modules: dict[str, Any] = {
        "LeRobotDataset": LeRobotDataset,
        "add_features": add_features,
    }

    if policy_type == "lapa_lam":
        from lerobot_policy_lapa_lam.modeling_lam import LAMPolicy

        modules["Policy"] = LAMPolicy
    elif policy_type == "rlfv_lam":
        from lerobot_policy_rlfv_lam.modeling_lam import RLFVLAMPolicy

        modules["Policy"] = RLFVLAMPolicy
    else:
        raise ValueError(f"Unsupported policy_type={policy_type!r}")

    return modules


def _resolve_output_repo_id(dataset_repo_id: str, output_repo_id: str | None) -> str:
    if output_repo_id:
        return output_repo_id
    return f"{dataset_repo_id}_latent_labeled"


def _parse_episodes(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise ValueError(f"--episodes must be a JSON list of ints, got {raw!r}")
    return value


def _resolve_device(requested_device: str | None, policy: Any) -> str:
    if requested_device:
        return requested_device
    config_device = getattr(policy.config, "device", None)
    if config_device:
        return str(config_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_camera_key(requested_camera_key: str | None, policy: Any, dataset: Any) -> str:
    if requested_camera_key:
        return requested_camera_key
    config_camera_key = getattr(policy.config, "camera_key", None)
    if config_camera_key:
        return str(config_camera_key)
    camera_keys = list(dataset.meta.camera_keys)
    if not camera_keys:
        raise ValueError("Dataset has no camera keys.")
    return str(camera_keys[0])


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        return
    if not force:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Pass --force to overwrite it."
        )
    shutil.rmtree(output_dir)


def _build_valid_pairs(dataset: Any, future_frames: int) -> tuple[np.ndarray, np.ndarray]:
    num_frames = int(dataset.num_frames)
    valid_mask = np.zeros(num_frames, dtype=np.int64)
    pair_targets = np.full(num_frames, -1, dtype=np.int64)

    episode_lengths_by_index = {
        int(ep_idx): int(length)
        for ep_idx, length in zip(dataset.meta.episodes["episode_index"], dataset.meta.episodes["length"], strict=True)
    }
    selected_episodes = (
        [int(ep_idx) for ep_idx in dataset.meta.episodes["episode_index"]]
        if dataset.episodes is None
        else sorted({int(ep_idx) for ep_idx in dataset.episodes})
    )

    offset = 0
    for episode_index in selected_episodes:
        episode_length = episode_lengths_by_index[episode_index]
        valid_count = max(episode_length - future_frames, 0)
        if valid_count > 0:
            starts = offset + np.arange(valid_count, dtype=np.int64)
            valid_mask[starts] = 1
            pair_targets[starts] = starts + future_frames
        offset += episode_length

    if offset != num_frames:
        raise ValueError(f"Episode-length reconstruction mismatch: expected {num_frames} frames, got {offset}")

    return valid_mask, pair_targets


@dataclass(frozen=True)
class LamLabelManifest:
    policy_type: str
    latent_format: str
    future_frames: int
    camera_key: str
    compatible_with_latent_smolvla: bool


class IndexedDeltaPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Any,
        *,
        camera_key: str,
        valid_indices: np.ndarray,
    ) -> None:
        self.dataset = dataset
        self.camera_key = str(camera_key)
        self.valid_indices = np.asarray(valid_indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.valid_indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row_idx = int(self.valid_indices[idx])
        item = self.dataset[row_idx]
        is_pad_key = f"{self.camera_key}_is_pad"
        if self.camera_key not in item:
            raise KeyError(f"Missing camera key {self.camera_key!r} in dataset item")
        if is_pad_key not in item:
            raise KeyError(f"Missing pad-mask key {is_pad_key!r} in dataset item")
        return {
            "row_idx": torch.tensor(row_idx, dtype=torch.int64),
            self.camera_key: item[self.camera_key],
            is_pad_key: item[is_pad_key],
        }


def _build_delta_timestamps(dataset: Any, camera_key: str, future_frames: int) -> dict[str, list[float]]:
    fps = float(dataset.fps)
    return {str(camera_key): [0.0, float(future_frames) / fps]}


def _get_latent_spec(policy: Any, latent_format: str) -> tuple[tuple[int, ...], np.dtype, float | int, str]:
    code_seq_len = int(policy.config.code_seq_len)
    quant_dim = int(policy.config.quant_dim)
    if latent_format == LATENT_FORMAT_IDS:
        return (code_seq_len,), np.dtype(np.int64), -100, "int64"
    if latent_format in {LATENT_FORMAT_CONTINUOUS, LATENT_FORMAT_CODEBOOK_VECTORS}:
        return (code_seq_len, quant_dim), np.dtype(np.float32), 0.0, "float32"
    raise ValueError(f"Unsupported latent_format={latent_format!r}")


def _make_dataloader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def _run_lam_labeling_dataloader(
    *,
    base_dataset: Any,
    pair_dataset: IndexedDeltaPairDataset,
    policy: Any,
    dataloader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    future_frames: int,
    camera_key: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    num_frames = int(base_dataset.num_frames)
    feature_shape, latent_dtype, invalid_fill, feature_dtype = _get_latent_spec(policy, args.latent_format)
    labels = np.full((num_frames, *feature_shape), invalid_fill, dtype=latent_dtype)
    valid_mask = np.zeros(num_frames, dtype=np.int64)

    total_valid = len(pair_dataset)
    total_batches = len(dataloader)
    logging.info(
        "LAM v2 labeling %d valid pairs out of %d frames with batch_size=%d num_workers=%d.",
        total_valid,
        num_frames,
        args.batch_size,
        args.num_workers,
    )

    start_time = time.perf_counter()
    processed = 0
    policy.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            row_idx = batch.pop("row_idx").cpu().numpy().astype(np.int64, copy=False)
            latents, pair_valid, _ = policy.extract_latents(batch, latent_format=args.latent_format)
            pair_valid_np = pair_valid.detach().cpu().numpy().astype(bool, copy=False)

            if pair_valid_np.any():
                valid_rows = row_idx[pair_valid_np]
                labels[valid_rows] = latents.detach().cpu().numpy().astype(latent_dtype, copy=False)
                valid_mask[valid_rows] = 1

            processed += int(row_idx.shape[0])
            if batch_idx % _PROGRESS_LOG_EVERY_BATCHES == 0 or batch_idx == total_batches:
                elapsed_s = max(time.perf_counter() - start_time, 1e-9)
                rate = processed / elapsed_s
                remaining = total_valid - processed
                eta_s = remaining / max(rate, 1e-9)
                logging.info(
                    "LAM v2 progress: %d/%d valid anchors (%.1f%%), batch %d/%d, %.1f anchors/s, ETA %.1f min",
                    processed,
                    total_valid,
                    100.0 * processed / max(total_valid, 1),
                    batch_idx,
                    total_batches,
                    rate,
                    eta_s / 60.0,
                )

    feature_info = {"dtype": feature_dtype, "shape": feature_shape, "names": None}
    manifest = LamLabelManifest(
        policy_type=args.policy_type,
        latent_format=args.latent_format,
        future_frames=future_frames,
        camera_key=camera_key,
        compatible_with_latent_smolvla=args.latent_format == LATENT_FORMAT_IDS,
    )
    return labels, valid_mask, feature_info, manifest.__dict__


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.info("Starting v2 labeling script.")
    logging.info("Loading source dataset metadata: repo_id=%s root=%s", args.dataset_repo_id, args.dataset_root)

    modules = _bootstrap_imports(args.policy_type, args.plugin_src_path)
    LeRobotDataset = modules["LeRobotDataset"]
    add_features = modules["add_features"]
    Policy = modules["Policy"]

    base_dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=_parse_episodes(args.episodes),
    )
    logging.info("Loading policy from %s", args.policy_path)
    policy = Policy.from_pretrained(args.policy_path, local_files_only=True)

    device = _resolve_device(args.device, policy)
    policy.to(device)
    policy.config.device = device
    camera_key = _resolve_camera_key(args.camera_key, policy, base_dataset)
    if hasattr(policy.config, "camera_key"):
        policy.config.camera_key = camera_key

    future_frames = args.future_frames or int(policy.config.future_frames)
    logging.info("Building valid anchor pairs with future_frames=%d and camera_key=%s", future_frames, camera_key)
    valid_mask_all, _ = _build_valid_pairs(base_dataset, future_frames=future_frames)
    valid_indices = np.flatnonzero(valid_mask_all)
    if args.max_valid_samples is not None:
        valid_indices = valid_indices[: args.max_valid_samples]
    logging.info("Resolved %d valid anchors.", len(valid_indices))

    delta_timestamps = _build_delta_timestamps(base_dataset, camera_key, future_frames)
    logging.info("Constructing pair dataset with delta_timestamps=%s", delta_timestamps)
    pair_dataset_base = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=_parse_episodes(args.episodes),
        delta_timestamps=delta_timestamps,
    )
    pair_dataset = IndexedDeltaPairDataset(
        pair_dataset_base,
        camera_key=camera_key,
        valid_indices=valid_indices,
    )
    dataloader = _make_dataloader(
        pair_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    logging.info(
        "DataLoader ready: batch_size=%d num_workers=%d total_batches=%d",
        args.batch_size,
        args.num_workers,
        len(dataloader),
    )

    labels, valid_mask, feature_info, manifest = _run_lam_labeling_dataloader(
        base_dataset=base_dataset,
        pair_dataset=pair_dataset,
        policy=policy,
        dataloader=dataloader,
        args=args,
        future_frames=future_frames,
        camera_key=camera_key,
    )

    valid_feature_info = {"dtype": "int64", "shape": (1,), "names": None}
    output_dir = Path(args.output_dir).resolve()
    _prepare_output_dir(output_dir, force=args.force)
    output_repo_id = _resolve_output_repo_id(args.dataset_repo_id, args.output_repo_id)

    logging.info(
        "Writing feature %s and mask %s into new dataset %s at %s.",
        args.feature_name,
        args.valid_feature_name,
        output_repo_id,
        output_dir,
    )
    relabeled_dataset = add_features(
        dataset=base_dataset,
        features={
            args.feature_name: (labels, feature_info),
            args.valid_feature_name: (valid_mask.reshape(-1, 1).astype(np.int64, copy=False), valid_feature_info),
        },
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    manifest_payload = {
        "source_dataset_repo_id": args.dataset_repo_id,
        "source_dataset_root": args.dataset_root,
        "episodes": _parse_episodes(args.episodes),
        "output_repo_id": output_repo_id,
        "output_dir": str(output_dir),
        "policy_path": args.policy_path,
        "feature_name": args.feature_name,
        "valid_feature_name": args.valid_feature_name,
        "implementation": "v2_dataloader",
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        **manifest,
    }
    manifest_path = output_dir / "label_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n")

    logging.info("Done.")
    logging.info("New dataset root: %s", relabeled_dataset.root)
    logging.info("New feature keys now include: %s", [args.feature_name, args.valid_feature_name])
    logging.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
