#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


LATENT_FORMAT_IDS = "ids"
LATENT_FORMAT_CONTINUOUS = "continuous"
LATENT_FORMAT_CODEBOOK_VECTORS = "codebook_vectors"
LATENT_FORMATS = (
    LATENT_FORMAT_IDS,
    LATENT_FORMAT_CONTINUOUS,
    LATENT_FORMAT_CODEBOOK_VECTORS,
)
FEATURE_SUFFIXES = {
    LATENT_FORMAT_IDS: "codebook_ids",
    LATENT_FORMAT_CONTINUOUS: "continuous",
    LATENT_FORMAT_CODEBOOK_VECTORS: "codebook_vectors",
}
PROGRESS_LOG_EVERY_BATCHES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-type", required=True, choices=["lapa_lam", "rlfv_lam"])
    parser.add_argument("--policy-path", required=True, help="Checkpoint dir for Policy.from_pretrained().")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root.")
    parser.add_argument("--episodes", default=None, help="Optional JSON list of episode indices to keep.")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--output-repo-id", default=None, help="Output dataset repo_id.")
    parser.add_argument(
        "--feature-prefix",
        required=True,
        help="Feature prefix; writes <prefix>.codebook_ids, <prefix>.continuous, <prefix>.codebook_vectors.",
    )
    parser.add_argument(
        "--valid-feature-name",
        default=None,
        help="Shared supervision mask feature name. Defaults to <feature-prefix>.valid.",
    )
    parser.add_argument("--camera-key", default=None, help="Override camera feature.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument("--future-frames", type=int, default=None, help="LAM override for frame pairing.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    parser.add_argument("--max-valid-samples", type=int, default=None, help="Optional smoke-test cap.")
    return parser.parse_args()


def _workspace_code_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_sys_path(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Required path does not exist: {resolved}")
    path_str = str(resolved)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _bootstrap_imports(policy_type: str) -> dict[str, Any]:
    workspace = _workspace_code_dir()
    _add_sys_path(workspace / "high-level-robot-planner" / "lerobot" / "src")

    if policy_type == "lapa_lam":
        _add_sys_path(workspace / "lerobot_policy_lapa_lam" / "src")
        from lerobot_policy_lapa_lam.modeling_lam import LAMPolicy as Policy
    else:
        _add_sys_path(workspace / "lerobot_policy_rlfv_lam" / "src")
        from lerobot_policy_rlfv_lam.modeling_lam import RLFVLAMPolicy as Policy

    from lerobot.datasets.dataset_tools import add_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return {
        "Policy": Policy,
        "add_features": add_features,
        "LeRobotDataset": LeRobotDataset,
    }


def _parse_episodes(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise ValueError(f"--episodes must be a JSON list of ints, got {raw!r}")
    return value


def _resolve_output_repo_id(dataset_repo_id: str, output_repo_id: str | None) -> str:
    return output_repo_id or f"{dataset_repo_id}_latent_labeled"


def _resolve_valid_feature_name(feature_prefix: str, valid_feature_name: str | None) -> str:
    return valid_feature_name or f"{feature_prefix}.valid"


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
        raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --force to overwrite it.")
    shutil.rmtree(output_dir)


def _build_valid_indices(dataset: Any, future_frames: int) -> np.ndarray:
    num_frames = int(dataset.num_frames)
    valid_mask = np.zeros(num_frames, dtype=bool)
    selected_episodes = None if dataset.episodes is None else {int(ep_idx) for ep_idx in dataset.episodes}

    offset = 0
    for episode_index, episode_length in zip(
        dataset.meta.episodes["episode_index"],
        dataset.meta.episodes["length"],
        strict=True,
    ):
        episode_index = int(episode_index)
        episode_length = int(episode_length)
        valid_count = max(episode_length - future_frames, 0)
        if selected_episodes is None or episode_index in selected_episodes:
            valid_mask[offset : offset + valid_count] = True
        offset += episode_length

    if offset != num_frames:
        raise ValueError(f"Episode-length reconstruction mismatch: expected {num_frames} frames, got {offset}")

    return np.flatnonzero(valid_mask)


def _build_delta_timestamps(dataset: Any, camera_key: str, future_frames: int) -> dict[str, list[float]]:
    return {camera_key: [0.0, float(future_frames) / float(dataset.fps)]}


class IndexedDeltaPairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Any, camera_key: str, valid_indices: np.ndarray):
        self.dataset = dataset
        self.camera_key = camera_key
        self.valid_indices = np.asarray(valid_indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.valid_indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row_idx = int(self.valid_indices[idx])
        item = self.dataset[row_idx]
        is_pad_key = f"{self.camera_key}_is_pad"
        return {
            "row_idx": torch.tensor(row_idx, dtype=torch.int64),
            self.camera_key: item[self.camera_key],
            is_pad_key: item[is_pad_key],
        }


def _make_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int, device: str) -> Any:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def _latent_spec(policy: Any, latent_format: str) -> tuple[tuple[int, ...], np.dtype, float | int, str]:
    code_seq_len = int(policy.config.code_seq_len)
    quant_dim = int(policy.config.quant_dim)
    if latent_format == LATENT_FORMAT_IDS:
        return (code_seq_len,), np.dtype(np.int64), -100, "int64"
    if latent_format in {LATENT_FORMAT_CONTINUOUS, LATENT_FORMAT_CODEBOOK_VECTORS}:
        return (code_seq_len, quant_dim), np.dtype(np.float32), 0.0, "float32"
    raise ValueError(f"Unsupported latent_format={latent_format!r}")


def _feature_name_map(feature_prefix: str) -> dict[str, str]:
    return {latent_format: f"{feature_prefix}.{FEATURE_SUFFIXES[latent_format]}" for latent_format in LATENT_FORMATS}


def _allocate_outputs(num_frames: int, policy: Any) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    outputs = {}
    feature_info = {}
    for latent_format in LATENT_FORMATS:
        feature_shape, latent_dtype, invalid_fill, feature_dtype = _latent_spec(policy, latent_format)
        outputs[latent_format] = np.full((num_frames, *feature_shape), invalid_fill, dtype=latent_dtype)
        feature_info[latent_format] = {"dtype": feature_dtype, "shape": feature_shape, "names": None}
    return outputs, feature_info


def _extract_all_latents(policy: Any, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    video, valid_pair, _ = policy._extract_frame_pair(batch)
    if not bool(valid_pair.any().item()):
        return {}, valid_pair
    valid_video = video[valid_pair]
    return {
        latent_format: policy.extract_latents_from_video(valid_video, latent_format=latent_format)
        for latent_format in LATENT_FORMATS
    }, valid_pair


def _run_labeling(
    base_dataset: Any,
    pair_dataset: IndexedDeltaPairDataset,
    dataloader: Any,
    policy: Any,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, dict[str, Any]]]:
    num_frames = int(base_dataset.num_frames)
    outputs, feature_info = _allocate_outputs(num_frames, policy)
    valid_mask = np.zeros(num_frames, dtype=np.int64)

    total_valid = len(pair_dataset)
    total_batches = len(dataloader)
    logging.info(
        "Labeling %d valid pairs out of %d frames with batch_size=%d num_workers=%d.",
        total_valid,
        num_frames,
        batch_size,
        num_workers,
    )

    start_time = time.perf_counter()
    processed = 0
    policy.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            row_idx = batch.pop("row_idx").cpu().numpy().astype(np.int64, copy=False)
            latents_by_format, pair_valid = _extract_all_latents(policy, batch)
            pair_valid_np = pair_valid.detach().cpu().numpy().astype(bool, copy=False)

            if pair_valid_np.any():
                valid_rows = row_idx[pair_valid_np]
                for latent_format, latents in latents_by_format.items():
                    target_dtype = outputs[latent_format].dtype
                    outputs[latent_format][valid_rows] = latents.detach().cpu().numpy().astype(target_dtype, copy=False)
                valid_mask[valid_rows] = 1

            processed += int(row_idx.shape[0])
            if batch_idx % PROGRESS_LOG_EVERY_BATCHES == 0 or batch_idx == total_batches:
                elapsed_s = max(time.perf_counter() - start_time, 1e-9)
                rate = processed / elapsed_s
                remaining = total_valid - processed
                eta_s = remaining / max(rate, 1e-9)
                logging.info(
                    "Progress: %d/%d valid anchors (%.1f%%), batch %d/%d, %.1f anchors/s, ETA %.1f min",
                    processed,
                    total_valid,
                    100.0 * processed / max(total_valid, 1),
                    batch_idx,
                    total_batches,
                    rate,
                    eta_s / 60.0,
                )

    return outputs, valid_mask, feature_info


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.info("Starting v3 labeling script.")

    modules = _bootstrap_imports(args.policy_type)
    Policy = modules["Policy"]
    add_features = modules["add_features"]
    LeRobotDataset = modules["LeRobotDataset"]

    episodes = _parse_episodes(args.episodes)
    base_dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=episodes)
    policy = Policy.from_pretrained(args.policy_path, local_files_only=True)

    device = _resolve_device(args.device, policy)
    policy.to(device)
    policy.config.device = device

    camera_key = _resolve_camera_key(args.camera_key, policy, base_dataset)
    policy.config.camera_key = camera_key
    future_frames = args.future_frames if args.future_frames is not None else int(policy.config.future_frames)

    valid_indices = _build_valid_indices(base_dataset, future_frames)
    if args.max_valid_samples is not None:
        valid_indices = valid_indices[: args.max_valid_samples]

    pair_dataset = IndexedDeltaPairDataset(
        LeRobotDataset(
            args.dataset_repo_id,
            root=args.dataset_root,
            episodes=episodes,
            delta_timestamps=_build_delta_timestamps(base_dataset, camera_key, future_frames),
        ),
        camera_key=camera_key,
        valid_indices=valid_indices,
    )
    dataloader = _make_dataloader(pair_dataset, args.batch_size, args.num_workers, device)

    outputs, valid_mask, feature_info = _run_labeling(
        base_dataset=base_dataset,
        pair_dataset=pair_dataset,
        dataloader=dataloader,
        policy=policy,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    output_dir = Path(args.output_dir).resolve()
    _prepare_output_dir(output_dir, args.force)
    output_repo_id = _resolve_output_repo_id(args.dataset_repo_id, args.output_repo_id)
    feature_names = _feature_name_map(args.feature_prefix)
    valid_feature_name = _resolve_valid_feature_name(args.feature_prefix, args.valid_feature_name)
    valid_feature_info = {"dtype": "int64", "shape": (1,), "names": None}

    features = {
        feature_names[latent_format]: (outputs[latent_format], feature_info[latent_format])
        for latent_format in LATENT_FORMATS
    }
    features[valid_feature_name] = (
        valid_mask.reshape(-1, 1).astype(np.int64, copy=False),
        valid_feature_info,
    )

    relabeled_dataset = add_features(
        dataset=base_dataset,
        features=features,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    manifest = {
        "implementation": "v3",
        "policy_type": args.policy_type,
        "policy_path": args.policy_path,
        "source_dataset_repo_id": args.dataset_repo_id,
        "source_dataset_root": args.dataset_root,
        "episodes": episodes,
        "output_repo_id": output_repo_id,
        "output_dir": str(output_dir),
        "camera_key": camera_key,
        "future_frames": future_frames,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "feature_prefix": args.feature_prefix,
        "feature_names": feature_names,
        "valid_feature_name": valid_feature_name,
        "latent_formats": list(LATENT_FORMATS),
        "latent_smolvla_feature_name": feature_names[LATENT_FORMAT_IDS],
    }
    manifest_path = output_dir / "label_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logging.info("Done.")
    logging.info("New dataset root: %s", relabeled_dataset.root)
    logging.info("Feature names: %s", {**feature_names, "valid": valid_feature_name})
    logging.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
