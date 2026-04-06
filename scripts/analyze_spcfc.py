#!/usr/bin/env python

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _artifact_registry import infer_checkpoint_metadata, make_artifact_id, register_artifact

PROGRESS_LOG_EVERY_BATCHES = 50
SUPPORTED_LATENT_FORMATS = {"continuous", "codebook_vectors"}


def parse_episode_list(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    episodes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not episodes:
        raise argparse.ArgumentTypeError("--episodes must be a comma-separated list of integers.")
    return episodes


def parse_latent_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not formats:
        raise argparse.ArgumentTypeError("--latent-formats must not be empty.")
    invalid = sorted(set(formats).difference(SUPPORTED_LATENT_FORMATS))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported latent formats: {invalid}. Supported formats: {sorted(SUPPORTED_LATENT_FORMATS)}."
        )
    return formats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CoMo-style S-PCFC statistics for a latent-action checkpoint.")
    parser.add_argument("--policy-path", type=Path, required=True, help="Path to the latent-action checkpoint directory.")
    parser.add_argument("--dataset-repo-id", type=str, required=True, help="LeRobot dataset repo id to analyze.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional local dataset root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where S-PCFC plots and tables will be written.",
    )
    parser.add_argument(
        "--episodes",
        type=parse_episode_list,
        default=None,
        help="Optional comma-separated list of episode indices.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default=None,
        help="Optional camera key override. Defaults to the policy camera key or first image feature.",
    )
    parser.add_argument(
        "--offset-frames",
        type=int,
        default=10,
        help="Temporal offset in frames for past/current and future/current pairs.",
    )
    parser.add_argument(
        "--latent-formats",
        type=parse_latent_formats,
        default=("continuous", "codebook_vectors"),
        help="Comma-separated subset of: continuous,codebook_vectors",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used for encoding.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of jointly valid rows kept for the final statistics.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for optional subsampling.")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def summarize_1d(values: np.ndarray) -> dict[str, float]:
    if values.shape[0] == 0:
        raise ValueError("Cannot summarize an empty score array.")
    values = values.astype(np.float64, copy=False)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def plot_spcfc_distribution(scores: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores.astype(np.float64, copy=False), bins=120, color="#4c78a8", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def load_policy_for_analysis(policy_path: Path, dataset_meta: Any) -> tuple[Any, Any]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path), local_files_only=True)
    policy_cfg.pretrained_path = Path(policy_path)
    policy = make_policy(policy_cfg, ds_meta=dataset_meta)
    policy.eval()
    return policy_cfg, policy


def resolve_camera_key(policy: Any, camera_key_override: str | None) -> str:
    if camera_key_override:
        return camera_key_override
    camera_key = getattr(policy.config, "camera_key", None)
    if camera_key:
        return str(camera_key)
    image_features = getattr(policy.config, "image_features", None)
    if image_features:
        return str(next(iter(image_features)))
    raise ValueError("Unable to determine camera key from the policy config. Pass --camera-key explicitly.")


def make_pair_datasets(
    dataset_repo_id: str,
    dataset_root: str | None,
    episodes: list[int] | None,
    camera_key: str,
    fps: float,
    offset_frames: int,
) -> tuple[Any, Any, float]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    offset_seconds = float(offset_frames) / float(fps)
    past_dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps={camera_key: [-offset_seconds, 0.0]},
    )
    future_dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps={camera_key: [offset_seconds, 0.0]},
    )
    return past_dataset, future_dataset, offset_seconds


def make_dataloader(dataset: Any, batch_size: int, num_workers: int, device: Any) -> Any:
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def to_numpy_bool(values: Any) -> np.ndarray:
    import torch

    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    return values.detach().cpu().numpy().astype(bool, copy=False)


def to_numpy_int64(values: Any) -> np.ndarray:
    import torch

    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    return values.detach().cpu().numpy().astype(np.int64, copy=False)


def iterate_aligned_batches(past_loader: Any, future_loader: Any):
    if len(past_loader) != len(future_loader):
        raise ValueError("Past and future dataloaders do not have the same length.")

    for batch_idx, (past_batch, future_batch) in enumerate(zip(past_loader, future_loader, strict=True), start=1):
        past_index = to_numpy_int64(past_batch["index"])
        future_index = to_numpy_int64(future_batch["index"])
        if not np.array_equal(past_index, future_index):
            raise ValueError(f"Past and future batches differ at batch {batch_idx} on `index`.")

        past_episode_index = to_numpy_int64(past_batch["episode_index"])
        future_episode_index = to_numpy_int64(future_batch["episode_index"])
        if not np.array_equal(past_episode_index, future_episode_index):
            raise ValueError(f"Past and future batches differ at batch {batch_idx} on `episode_index`.")

        yield batch_idx, past_batch, future_batch, past_index, past_episode_index


def expand_valid_rows(
    compact_latents: np.ndarray,
    valid_mask: np.ndarray,
    full_shape: tuple[int, ...],
    fill_value: float,
) -> np.ndarray:
    if valid_mask.ndim != 1:
        raise ValueError(f"Expected `valid_mask` to be 1D, got shape {tuple(valid_mask.shape)}.")
    if compact_latents.shape[0] != int(valid_mask.sum()):
        raise ValueError("Compact latent rows do not match the number of valid rows.")

    expanded = np.full((valid_mask.shape[0], *full_shape), fill_value, dtype=compact_latents.dtype)
    if compact_latents.shape[0] > 0:
        expanded[valid_mask] = compact_latents
    return expanded


def extract_batch_latents(policy: Any, batch: dict[str, Any], latent_format: str) -> tuple[np.ndarray, np.ndarray, str]:
    import torch

    latents, valid_mask, camera_key = policy.extract_latents(batch, latent_format=latent_format)
    if not torch.is_tensor(latents):
        latents = torch.as_tensor(latents)
    valid_mask_np = to_numpy_bool(valid_mask)
    compact_latents = latents.detach().cpu().numpy()
    expanded = expand_valid_rows(
        compact_latents=compact_latents,
        valid_mask=valid_mask_np,
        full_shape=tuple(compact_latents.shape[1:]),
        fill_value=0.0,
    )
    return expanded, valid_mask_np, camera_key


def compute_spcfc_scores(past_latents: np.ndarray, future_latents: np.ndarray) -> np.ndarray:
    if past_latents.shape != future_latents.shape:
        raise ValueError(f"Shape mismatch: {past_latents.shape} vs {future_latents.shape}")

    past_flat = past_latents.reshape(past_latents.shape[0], -1).astype(np.float64, copy=False)
    future_flat = future_latents.reshape(future_latents.shape[0], -1).astype(np.float64, copy=False)
    past_norm = np.linalg.norm(past_flat, axis=1)
    future_norm = np.linalg.norm(future_flat, axis=1)
    denom = np.maximum(past_norm * future_norm, 1e-12)
    scores = np.sum(past_flat * future_flat, axis=1) / denom
    return np.clip(scores, -1.0, 1.0).astype(np.float32, copy=False)


def summarize_spcfc(scores_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for latent_format, group_df in scores_df.groupby("latent_format", sort=True):
        values = group_df["spcfc"].to_numpy()
        summary_rows.append(
            {
                "latent_format": latent_format,
                "count": int(values.shape[0]),
                **summarize_1d(values),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("latent_format", ignore_index=True)

    per_episode_df = (
        scores_df.groupby(["latent_format", "episode_index"], as_index=False)
        .agg(
            count=("spcfc", "size"),
            mean=("spcfc", "mean"),
            std=("spcfc", "std"),
            min=("spcfc", "min"),
            median=("spcfc", "median"),
            max=("spcfc", "max"),
        )
        .sort_values(["latent_format", "episode_index"], ignore_index=True)
    )
    return summary_df, per_episode_df


def analyze_spcfc(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=args.episodes)
    policy_cfg, policy = load_policy_for_analysis(args.policy_path.resolve(), source_dataset.meta)
    camera_key = resolve_camera_key(policy, args.camera_key)

    past_dataset, future_dataset, offset_seconds = make_pair_datasets(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        episodes=args.episodes,
        camera_key=camera_key,
        fps=source_dataset.meta.fps,
        offset_frames=args.offset_frames,
    )

    past_loader = make_dataloader(past_dataset, args.batch_size, args.num_workers, policy_cfg.device)
    future_loader = make_dataloader(future_dataset, args.batch_size, args.num_workers, policy_cfg.device)

    rng = np.random.default_rng(args.seed)
    score_chunks: dict[str, list[np.ndarray]] = {latent_format: [] for latent_format in args.latent_formats}
    row_index_chunks: list[np.ndarray] = []
    episode_index_chunks: list[np.ndarray] = []

    start_time = time.perf_counter()
    with np.errstate(invalid="ignore"):
        for batch_idx, past_batch, future_batch, row_idx_np, episode_idx_np in iterate_aligned_batches(past_loader, future_loader):
            joint_valid = None
            by_format: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            resolved_camera_key = None
            for latent_format in args.latent_formats:
                past_latents, past_valid, past_camera_key = extract_batch_latents(policy, past_batch, latent_format)
                future_latents, future_valid, future_camera_key = extract_batch_latents(policy, future_batch, latent_format)
                if past_camera_key != future_camera_key:
                    raise ValueError(
                        f"Camera key mismatch for latent format {latent_format!r}: {past_camera_key!r} vs {future_camera_key!r}."
                    )
                resolved_camera_key = past_camera_key
                this_joint_valid = past_valid & future_valid
                if joint_valid is None:
                    joint_valid = this_joint_valid
                elif not np.array_equal(joint_valid, this_joint_valid):
                    raise ValueError(f"Latent format {latent_format!r} produced a different joint-valid mask.")
                by_format[latent_format] = (past_latents, future_latents)

            if joint_valid is None or not bool(np.any(joint_valid)):
                continue

            row_index_chunks.append(row_idx_np[joint_valid])
            episode_index_chunks.append(episode_idx_np[joint_valid])
            for latent_format, (past_latents, future_latents) in by_format.items():
                score_chunks[latent_format].append(
                    compute_spcfc_scores(past_latents[joint_valid], future_latents[joint_valid])
                )

            if batch_idx % PROGRESS_LOG_EVERY_BATCHES == 0 or batch_idx == len(past_loader):
                elapsed_s = max(time.perf_counter() - start_time, 1e-9)
                processed_rows = min(batch_idx * args.batch_size, len(past_dataset))
                kept_rows = sum(chunk.shape[0] for chunk in row_index_chunks)
                logging.info(
                    "Progress: rows=%d/%d jointly_valid=%d rate=%.1f rows/s camera_key=%s",
                    processed_rows,
                    len(past_dataset),
                    kept_rows,
                    processed_rows / elapsed_s,
                    resolved_camera_key,
                )

    if not row_index_chunks:
        raise ValueError("No jointly valid rows were found for the requested offset.")

    base_df = pd.DataFrame(
        {
            "index": np.concatenate(row_index_chunks, axis=0),
            "episode_index": np.concatenate(episode_index_chunks, axis=0),
        }
    )
    score_arrays = {
        latent_format: np.concatenate(chunks, axis=0) for latent_format, chunks in score_chunks.items()
    }
    expected_rows = len(base_df)
    for latent_format, scores in score_arrays.items():
        if scores.shape[0] != expected_rows:
            raise ValueError(
                f"Latent format {latent_format!r} produced {scores.shape[0]} rows, expected {expected_rows}."
            )

    if args.max_samples is not None and expected_rows > args.max_samples:
        keep = np.sort(rng.choice(expected_rows, size=args.max_samples, replace=False))
        base_df = base_df.iloc[keep].reset_index(drop=True)
        score_arrays = {latent_format: scores[keep] for latent_format, scores in score_arrays.items()}

    scores_df = pd.concat(
        [
            base_df.assign(latent_format=latent_format, spcfc=scores)
            for latent_format, scores in score_arrays.items()
        ],
        ignore_index=True,
    )
    scores_df.to_parquet(output_dir / "spcfc_scores.parquet", index=False)

    summary_df, per_episode_df = summarize_spcfc(scores_df)
    summary_df.to_csv(output_dir / "spcfc_summary.csv", index=False)
    per_episode_df.to_csv(output_dir / "spcfc_by_episode.csv", index=False)

    plot_artifacts = []
    for latent_format, group_df in scores_df.groupby("latent_format", sort=True):
        output_path = output_dir / f"spcfc_distribution__{latent_format}.png"
        plot_spcfc_distribution(
            group_df["spcfc"].to_numpy(),
            title=f"S-PCFC Distribution ({latent_format})",
            output_path=output_path,
        )
        plot_artifacts.append(output_path.name)

    summary = {
        "policy_path": str(args.policy_path.resolve()),
        "policy_type": str(policy_cfg.type),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": args.dataset_root,
        "episodes": args.episodes,
        "camera_key": camera_key,
        "offset_frames": int(args.offset_frames),
        "offset_seconds": float(offset_seconds),
        "latent_formats": list(args.latent_formats),
        "sampled_rows": int(len(base_df)),
        "scores_by_latent_format": summary_df.to_dict(orient="records"),
        "artifacts": sorted(p.name for p in output_dir.iterdir()),
    }
    save_json(output_dir / "summary.json", summary)

    readme_lines = [
        "# S-PCFC Analysis",
        "",
        f"- Policy path: `{args.policy_path.resolve()}`",
        f"- Policy type: `{policy_cfg.type}`",
        f"- Dataset repo id: `{args.dataset_repo_id}`",
        f"- Camera key: `{camera_key}`",
        f"- Offset frames: `{args.offset_frames}`",
        f"- Offset seconds: `{offset_seconds:.6f}`",
        f"- Sampled jointly-valid rows: `{len(base_df)}`",
        "",
        "## Interpretation",
        "- S-PCFC is the cosine similarity between past-to-current and future-to-current latent motion embeddings.",
        "- Lower is better in the CoMo framing because it indicates less static/background redundancy.",
        "",
        "## Summary",
    ]
    for _, row in summary_df.iterrows():
        readme_lines.extend(
            [
                f"- `{row['latent_format']}`: mean=`{row['mean']:.4f}`, median=`{row['median']:.4f}`, p95=`{row['p95']:.4f}`, max=`{row['max']:.4f}`",
            ]
        )
    readme_lines.extend(["", "## Distribution Plots"])
    readme_lines.extend(f"- `{name}`" for name in plot_artifacts)
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines) + "\n")

    checkpoint_meta = infer_checkpoint_metadata(args.policy_path.resolve())
    summary_rows = summary_df.to_dict(orient="records")
    headline_metrics = {
        "offset_frames": int(args.offset_frames),
        "offset_seconds": float(offset_seconds),
    }
    for row in summary_rows:
        latent_format = str(row["latent_format"])
        headline_metrics[f"{latent_format}_mean"] = float(row["mean"])
        headline_metrics[f"{latent_format}_median"] = float(row["median"])
        headline_metrics[f"{latent_format}_p95"] = float(row["p95"])

    analysis_manifest = {
        "artifact_type": "latent_analysis",
        "analysis_kind": "spcfc",
        "suite_name": "spcfc",
        "suite_version": "v1",
        "artifact_id": make_artifact_id(
            suite_name="spcfc",
            suite_version="v1",
            checkpoint_id=checkpoint_meta["source_checkpoint_id"],
            output_label=output_dir.name,
        ),
        **checkpoint_meta,
        "parent_export_artifact_id": None,
        "parent_export_manifest_path": None,
        "input_dataset_root": args.dataset_root,
        "input_dataset_repo_id": args.dataset_repo_id,
        "script_path": str(Path(__file__).resolve()),
        "cli_args": list(sys.argv[1:]),
        "output_path": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "readme_path": str(readme_path),
        "headline_metrics": headline_metrics,
    }
    register_artifact(
        manifest_path=output_dir / "analysis_manifest.json",
        manifest=analysis_manifest,
        registry_candidates=[output_dir, args.policy_path, args.dataset_root],
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    analyze_spcfc(args)


if __name__ == "__main__":
    main()
