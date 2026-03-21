#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


LATENT_FORMAT_IDS = "ids"
LATENT_FORMAT_CONTINUOUS = "continuous"
LATENT_FORMAT_CODEBOOK_VECTORS = "codebook_vectors"
DISMO_FORMAT_MOTION = "motion_embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-type", required=True, choices=["lam", "dismo"])
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
            DISMO_FORMAT_MOTION,
        ],
        help="Label representation. For dismo, use motion_embeddings.",
    )
    parser.add_argument(
        "--future-frames",
        type=int,
        default=None,
        help="LAM override for frame pairing.",
    )
    parser.add_argument(
        "--dismo-lookahead",
        type=int,
        default=None,
        help="DISMO lookahead for motion_extractor.forward_sliding(). Defaults to config.max_delta_time.",
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
        help="Optional smoke-test cap on the number of valid samples/clips to label. Remaining samples stay invalid.",
    )
    return parser.parse_args()


def _workspace_code_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_plugin_src(policy_type: str) -> Path:
    base = _workspace_code_dir()
    if policy_type == "lam":
        return base / "lerobot_policy_lapa_lam" / "src"
    if policy_type == "dismo":
        return base / "lerobot_policy_dismo" / "src"
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

    if policy_type == "lam":
        from lerobot_policy_lam.modeling_lam import LAMPolicy

        modules["Policy"] = LAMPolicy
    elif policy_type == "dismo":
        from lerobot_policy_dismo.modeling_dismo import DisMoPolicy

        modules["Policy"] = DisMoPolicy
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


def _ensure_hwc_or_chw(image: object) -> torch.Tensor:
    tensor = torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dimensions, got shape {tuple(tensor.shape)}.")
    if tensor.shape[0] == 3:
        return tensor
    if tensor.shape[-1] == 3:
        return tensor.permute(2, 0, 1)
    raise ValueError(f"Unsupported image layout {tuple(tensor.shape)}.")


def _load_frame_pair(dataset: Any, first_idx: int, second_idx: int, camera_key: str) -> torch.Tensor:
    first_item = dataset[first_idx]
    second_item = dataset[second_idx]
    first_image = _ensure_hwc_or_chw(first_item[camera_key])
    second_image = _ensure_hwc_or_chw(second_item[camera_key])
    return torch.stack((first_image, second_image), dim=0)


def _build_valid_pairs(dataset: Any, future_frames: int) -> tuple[np.ndarray, np.ndarray]:
    dataset._ensure_hf_dataset_loaded()
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    num_frames = len(episode_indices)
    valid_mask = np.zeros(num_frames, dtype=np.int64)
    pair_targets = np.full(num_frames, -1, dtype=np.int64)
    for idx in range(num_frames - future_frames):
        target_idx = idx + future_frames
        if episode_indices[idx] != episode_indices[target_idx]:
            continue
        valid_mask[idx] = 1
        pair_targets[idx] = target_idx
    return valid_mask, pair_targets


def _extract_lam_latents(policy: Any, video: torch.Tensor, latent_format: str) -> torch.Tensor:
    return policy.extract_latents_from_video(video, latent_format=latent_format)


def _run_lam_labeling(dataset: Any, policy: Any, args: argparse.Namespace, camera_key: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    future_frames = args.future_frames or int(policy.config.future_frames)
    valid_mask, pair_targets = _build_valid_pairs(dataset, future_frames=future_frames)
    num_frames = int(valid_mask.shape[0])

    if args.latent_format == LATENT_FORMAT_IDS:
        feature_shape = (int(policy.config.code_seq_len),)
        latent_dtype = np.int64
        invalid_fill = -100
        feature_dtype = "int64"
    elif args.latent_format in {LATENT_FORMAT_CONTINUOUS, LATENT_FORMAT_CODEBOOK_VECTORS}:
        feature_shape = (int(policy.config.code_seq_len), int(policy.config.quant_dim))
        latent_dtype = np.float32
        invalid_fill = 0.0
        feature_dtype = "float32"
    else:
        raise ValueError(f"Unsupported latent_format={args.latent_format!r} for LAM.")

    labels = np.full((num_frames, *feature_shape), invalid_fill, dtype=latent_dtype)
    valid_indices = np.flatnonzero(valid_mask)
    if args.max_valid_samples is not None:
        valid_indices = valid_indices[: args.max_valid_samples]
    logging.info("LAM labeling %d valid pairs out of %d frames.", len(valid_indices), num_frames)

    if len(valid_indices) > 0:
        policy.eval()
        with torch.inference_mode():
            for start in range(0, len(valid_indices), args.batch_size):
                batch_indices = valid_indices[start : start + args.batch_size]
                frame_pairs = [
                    _load_frame_pair(dataset, int(frame_idx), int(pair_targets[frame_idx]), camera_key)
                    for frame_idx in batch_indices
                ]
                batch = {camera_key: torch.stack(frame_pairs, dim=0)}
                video, valid_pair, _ = policy._extract_frame_pair(batch)
                if not bool(valid_pair.all().item()):
                    raise RuntimeError("Unexpected invalid frame pair in LAM labeling batch.")
                latent_values = _extract_lam_latents(policy, video, args.latent_format)
                labels[batch_indices] = latent_values.detach().cpu().numpy().astype(latent_dtype, copy=False)

    feature_info = {"dtype": feature_dtype, "shape": feature_shape, "names": None}
    manifest = {
        "policy_type": "lam",
        "latent_format": args.latent_format,
        "future_frames": future_frames,
        "camera_key": camera_key,
        "compatible_with_latent_smolvla": args.latent_format == LATENT_FORMAT_IDS,
    }
    return labels, valid_mask, feature_info, manifest


def _build_valid_clip_starts(dataset: Any, clip_length: int) -> np.ndarray:
    dataset._ensure_hf_dataset_loaded()
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    num_frames = len(episode_indices)
    valid_mask = np.zeros(num_frames, dtype=np.int64)
    for idx in range(num_frames - clip_length + 1):
        if episode_indices[idx] == episode_indices[idx + clip_length - 1]:
            valid_mask[idx] = 1
    return valid_mask


def _load_clip(dataset: Any, start_idx: int, clip_length: int, camera_key: str) -> torch.Tensor:
    frames = [_ensure_hwc_or_chw(dataset[start_idx + offset][camera_key]) for offset in range(clip_length)]
    return torch.stack(frames, dim=0)


def _extract_dismo_motion(policy: Any, clip_batch: torch.Tensor, camera_key: str, lookahead: int) -> torch.Tensor:
    batch = {camera_key: clip_batch}
    clip_u8, valid_clip, _ = policy._extract_video_clip(batch)
    if not bool(valid_clip.all().item()):
        raise RuntimeError("Unexpected invalid clip in DISMO labeling batch.")
    clip_hwc = clip_u8.permute(0, 1, 3, 4, 2)
    motion_frames = policy._convert_frames(clip_hwc)
    motion_embeddings = policy.dismo.motion_extractor.forward_sliding(motion_frames, lookahead=lookahead)
    return motion_embeddings[:, 0]


def _run_dismo_labeling(dataset: Any, policy: Any, args: argparse.Namespace, camera_key: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    if args.latent_format != DISMO_FORMAT_MOTION:
        raise ValueError("DISMO currently supports only --latent-format motion_embeddings.")

    clip_length = int(policy.config.clip_length)
    lookahead = args.dismo_lookahead or int(policy.config.max_delta_time)
    if lookahead < 1 or lookahead >= clip_length:
        raise ValueError(f"dismo lookahead must satisfy 1 <= lookahead < clip_length, got {lookahead}.")

    valid_mask = _build_valid_clip_starts(dataset, clip_length=clip_length)
    num_frames = int(valid_mask.shape[0])
    embedding_dim = int(policy.config.motion_d_motion)
    labels = np.zeros((num_frames, embedding_dim), dtype=np.float32)
    valid_indices = np.flatnonzero(valid_mask)
    if args.max_valid_samples is not None:
        valid_indices = valid_indices[: args.max_valid_samples]
    logging.info("DISMO labeling %d valid clips out of %d frames.", len(valid_indices), num_frames)

    if len(valid_indices) > 0:
        policy.eval()
        with torch.inference_mode():
            for start in range(0, len(valid_indices), args.batch_size):
                batch_indices = valid_indices[start : start + args.batch_size]
                clips = [_load_clip(dataset, int(frame_idx), clip_length, camera_key) for frame_idx in batch_indices]
                clip_batch = torch.stack(clips, dim=0)
                embeddings = _extract_dismo_motion(policy, clip_batch, camera_key, lookahead=lookahead)
                labels[batch_indices] = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)

    feature_info = {"dtype": "float32", "shape": (embedding_dim,), "names": None}
    manifest = {
        "policy_type": "dismo",
        "latent_format": DISMO_FORMAT_MOTION,
        "clip_length": clip_length,
        "lookahead": lookahead,
        "camera_key": camera_key,
        "compatible_with_latent_smolvla": False,
    }
    return labels, valid_mask, feature_info, manifest


def _assert_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {output_dir}. Pass --force to allow reuse."
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    modules = _bootstrap_imports(args.policy_type, args.plugin_src_path)
    LeRobotDataset = modules["LeRobotDataset"]
    add_features = modules["add_features"]
    Policy = modules["Policy"]

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=_parse_episodes(args.episodes),
    )
    policy = Policy.from_pretrained(args.policy_path, local_files_only=True)

    device = _resolve_device(args.device, policy)
    policy.to(device)
    policy.config.device = device
    camera_key = _resolve_camera_key(args.camera_key, policy, dataset)
    if hasattr(policy.config, "camera_key"):
        policy.config.camera_key = camera_key

    if args.policy_type == "lam":
        labels, valid_mask, feature_info, manifest = _run_lam_labeling(dataset, policy, args, camera_key)
    elif args.policy_type == "dismo":
        labels, valid_mask, feature_info, manifest = _run_dismo_labeling(dataset, policy, args, camera_key)
    else:
        raise ValueError(f"Unsupported policy_type={args.policy_type!r}")

    valid_feature_info = {"dtype": "int64", "shape": (1,), "names": None}
    output_dir = Path(args.output_dir).resolve()
    _assert_output_dir(output_dir, force=args.force)
    output_repo_id = _resolve_output_repo_id(args.dataset_repo_id, args.output_repo_id)

    logging.info(
        "Writing feature %s and mask %s into new dataset %s at %s.",
        args.feature_name,
        args.valid_feature_name,
        output_repo_id,
        output_dir,
    )
    relabeled_dataset = add_features(
        dataset=dataset,
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
