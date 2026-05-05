#!/usr/bin/env python

import json
import logging
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _artifact_registry import infer_checkpoint_metadata, make_artifact_id, register_artifact

PROGRESS_LOG_EVERY_BATCHES = 20
METADATA_KEYS = ("index", "episode_index", "task_index", "frame_index", "timestamp")
LATENT_FORMAT_TO_REPRESENTATION = {
    "ids": "codebook_id_latents",
    "continuous": "continuous_vector_latents",
    "codebook_vectors": "codebook_vector_latents",
}
LATENT_FORMAT_DISPLAY_NAMES = {
    "ids": "Reconstruction (IDs)",
    "continuous": "Reconstruction (Continuous)",
    "codebook_vectors": "Reconstruction (Codebook Vectors)",
}


@dataclass
class LatentReconstructionConfig:
    policy: PreTrainedConfig | None = None
    dataset_repo_id: str | None = None
    dataset_root: str | None = None
    episodes: list[int] | None = None
    output_dir: Path | None = None
    batch_size: int = 16
    num_workers: int = 8
    force: bool = False
    max_valid_samples: int | None = 128
    num_visualizations: int = 12
    latent_formats: list[str] | None = None
    seed: int = 0

    def validate(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("Policy is not configured. Please specify a checkpoint with `--policy.path`.")

        cli_overrides = [
            arg
            for arg in (parser.get_cli_overrides("policy") or [])
            if not arg.startswith("--discover_packages_path=")
        ]
        self.policy = PreTrainedConfig.from_pretrained(
            policy_path,
            local_files_only=True,
            cli_overrides=cli_overrides,
        )
        self.policy.pretrained_path = Path(policy_path)

        if not self.dataset_repo_id:
            raise ValueError("Please specify `--dataset_repo_id`.")
        if self.output_dir is None:
            raise ValueError("Please specify `--output_dir`.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}.")
        if self.max_valid_samples is not None and self.max_valid_samples < 1:
            raise ValueError(f"max_valid_samples must be >= 1, got {self.max_valid_samples}.")
        if self.num_visualizations < 1:
            raise ValueError(f"num_visualizations must be >= 1, got {self.num_visualizations}.")

        requested_formats = self.latent_formats or list(LATENT_FORMAT_TO_REPRESENTATION)
        invalid_formats = sorted(set(requested_formats).difference(LATENT_FORMAT_TO_REPRESENTATION))
        if invalid_formats:
            raise ValueError(
                f"Unsupported latent formats {invalid_formats}; expected a subset of {sorted(LATENT_FORMAT_TO_REPRESENTATION)}."
            )
        self.latent_formats = list(dict.fromkeys(requested_formats))

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        return
    if not force:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --force to overwrite it.")
    shutil.rmtree(output_dir)


def _get_required_method(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if callable(method):
        return method
    raise TypeError(
        f"Policy type {getattr(obj.config, 'type', '<unknown>')!r} does not implement `{method_name}()`."
    )


def _normalize_export_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise TypeError(f"prepare_latent_export() must return a dict, got {type(plan)}.")
    required_keys = {"delta_timestamps", "representations"}
    missing = required_keys.difference(plan)
    if missing:
        raise KeyError(f"prepare_latent_export() is missing keys: {sorted(missing)}")
    return {
        "delta_timestamps": plan["delta_timestamps"],
        "representations": dict(plan["representations"]),
    }


def _to_numpy_column(values: Any) -> np.ndarray:
    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    array = values.detach().cpu().numpy()
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return array


def squeeze_single_frame(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 5:
        raise ValueError(f"Expected frames with shape [B,C,1,H,W], got {tuple(frames.shape)}.")
    if frames.shape[2] != 1:
        raise ValueError(f"Expected exactly one frame in time dimension, got {tuple(frames.shape)}.")
    return frames[:, :, 0]


def compute_reconstruction_metrics(
    target_frame: torch.Tensor,
    reconstructions_by_format: dict[str, torch.Tensor],
) -> list[dict[str, float | str]]:
    target = squeeze_single_frame(target_frame).detach().cpu().to(torch.float32)
    rows: list[dict[str, float | str]] = []
    for latent_format, recon in reconstructions_by_format.items():
        recon_frame = squeeze_single_frame(recon).detach().cpu().to(torch.float32)
        mse = (recon_frame - target).square().mean(dim=(1, 2, 3)).numpy().astype(np.float64, copy=False)
        mae = (recon_frame - target).abs().mean(dim=(1, 2, 3)).numpy().astype(np.float64, copy=False)
        psnr = -10.0 * np.log10(np.maximum(mse, 1e-12))
        for idx in range(mse.shape[0]):
            rows.append(
                {
                    "latent_format": latent_format,
                    "mse": float(mse[idx]),
                    "mae": float(mae[idx]),
                    "psnr_db": float(psnr[idx]),
                }
            )
    return rows


def summarize_reconstruction_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for latent_format, group in metrics_df.groupby("latent_format", sort=False):
        mse = group["mse"].to_numpy(dtype=np.float64, copy=False)
        mae = group["mae"].to_numpy(dtype=np.float64, copy=False)
        psnr = group["psnr_db"].to_numpy(dtype=np.float64, copy=False)
        rows.append(
            {
                "latent_format": latent_format,
                "display_name": LATENT_FORMAT_DISPLAY_NAMES.get(latent_format, latent_format),
                "count": int(group.shape[0]),
                "mean_mse": float(np.mean(mse)),
                "median_mse": float(np.median(mse)),
                "mean_mae": float(np.mean(mae)),
                "median_mae": float(np.median(mae)),
                "mean_psnr_db": float(np.mean(psnr)),
                "median_psnr_db": float(np.median(psnr)),
            }
        )
    return pd.DataFrame(rows)


def _frame_to_image(frame: torch.Tensor) -> np.ndarray:
    image = frame.detach().cpu().to(torch.float32).permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def _sample_title(sample: dict[str, Any]) -> str:
    episode = sample.get("episode_index")
    frame = sample.get("frame_index")
    index = sample.get("index")
    parts = []
    if episode is not None:
        parts.append(f"ep {int(episode)}")
    if frame is not None:
        parts.append(f"frame {int(frame)}")
    if index is not None:
        parts.append(f"idx {int(index)}")
    return " | ".join(parts) if parts else "sample"


def render_reconstruction_grid(samples: list[dict[str, Any]], latent_formats: list[str], output_path: Path) -> None:
    if not samples:
        raise ValueError("No reconstruction samples were provided.")

    columns = ["First Frame", "Target Frame", *[LATENT_FORMAT_DISPLAY_NAMES[f] for f in latent_formats]]
    n_rows = len(samples)
    n_cols = len(columns)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, sample in enumerate(samples):
        panel_images = [
            sample["first_frame"],
            sample["target_frame"],
            *[sample["reconstructions"][latent_format] for latent_format in latent_formats],
        ]
        for col_idx, panel in enumerate(panel_images):
            ax = axes[row_idx, col_idx]
            ax.imshow(_frame_to_image(panel))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(columns[col_idx], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(_sample_title(sample), fontsize=9)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_sample_panel(sample: dict[str, Any], latent_formats: list[str], output_path: Path) -> None:
    render_reconstruction_grid([sample], latent_formats=latent_formats, output_path=output_path)


def reservoir_update(
    reservoir: list[dict[str, Any]],
    candidate: dict[str, Any],
    seen_count: int,
    capacity: int,
    rng: np.random.Generator,
) -> None:
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(candidate)
        return
    replace_idx = int(rng.integers(0, seen_count))
    if replace_idx < capacity:
        reservoir[replace_idx] = candidate


def build_readme(
    *,
    cfg: LatentReconstructionConfig,
    output_dir: Path,
    summary_df: pd.DataFrame,
    camera_key: str,
    analyzed_valid_rows: int,
    visualized_samples: int,
) -> Path:
    lines = [
        "# Latent Reconstruction Visualization",
        "",
        f"- Policy path: `{cfg.policy.pretrained_path}`",
        f"- Policy type: `{cfg.policy.type}`",
        f"- Dataset repo id: `{cfg.dataset_repo_id}`",
        f"- Dataset root: `{cfg.dataset_root}`",
        f"- Episodes: `{cfg.episodes}`",
        f"- Camera key: `{camera_key}`",
        f"- Analyzed valid rows: `{analyzed_valid_rows}`",
        f"- Visualized samples: `{visualized_samples}`",
        f"- Latent formats: `{cfg.latent_formats}`",
        "",
        "## Mean Reconstruction Metrics",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['latent_format']}`: mean MSE=`{row['mean_mse']:.6f}`, mean MAE=`{row['mean_mae']:.6f}`, mean PSNR=`{row['mean_psnr_db']:.2f} dB`"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "- `reconstruction_grid.png`: overview grid with first frame, target frame, and reconstructions for each latent format.",
            "- `sample_panels/`: individual sample panels for the visualized subset.",
            "- `reconstruction_metrics_by_format.csv`: aggregate reconstruction metrics.",
            "- `per_sample_reconstruction_metrics.csv.gz`: per-sample reconstruction metrics for all analyzed rows.",
            "- `visualized_samples.csv`: metadata for the visualized samples.",
        ]
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n")
    return readme_path


def analyze_latent_reconstructions(cfg: LatentReconstructionConfig) -> None:
    cfg.validate()
    if cfg.policy is None:
        raise ValueError("Policy config was not loaded.")
    if cfg.output_dir is None:
        raise ValueError("output_dir was not configured.")
    if cfg.dataset_repo_id is None:
        raise ValueError("dataset_repo_id was not configured.")
    if cfg.latent_formats is None:
        raise ValueError("latent_formats was not configured.")

    output_dir = cfg.output_dir.resolve()
    _prepare_output_dir(output_dir, cfg.force)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_panel_dir = output_dir / "sample_panels"
    sample_panel_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(cfg.seed))
    np_rng = np.random.default_rng(int(cfg.seed))

    source_dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root)
    policy = make_policy(cfg.policy, ds_meta=source_dataset.meta)
    policy.eval()

    prepare_latent_export = _get_required_method(policy, "prepare_latent_export")
    extract_frame_pair = _get_required_method(policy, "_extract_frame_pair")
    reconstruct_from_video = _get_required_method(policy, "reconstruct_from_video")
    plan = _normalize_export_plan(prepare_latent_export(source_dataset.meta))

    missing_representations = [
        LATENT_FORMAT_TO_REPRESENTATION[latent_format]
        for latent_format in cfg.latent_formats
        if LATENT_FORMAT_TO_REPRESENTATION[latent_format] not in plan["representations"]
    ]
    if missing_representations:
        raise ValueError(
            f"Policy export plan does not expose required latent representations: {sorted(missing_representations)}"
        )

    label_dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
        episodes=cfg.episodes,
        delta_timestamps=plan["delta_timestamps"],
    )
    dataloader = torch.utils.data.DataLoader(
        label_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=str(cfg.policy.device).startswith("cuda"),
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    logging.info(
        "Latent reconstruction visualization setup: policy=%s dataset=%s output=%s batch_size=%d max_valid_samples=%s formats=%s",
        cfg.policy.type,
        cfg.dataset_repo_id,
        output_dir,
        cfg.batch_size,
        cfg.max_valid_samples,
        cfg.latent_formats,
    )

    metrics_rows: list[dict[str, Any]] = []
    visualized_samples: list[dict[str, Any]] = []
    total_valid = 0
    total_batches = 0
    camera_key: str | None = None
    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            video, valid_mask, batch_camera_key = extract_frame_pair(batch)
            camera_key = batch_camera_key
            valid_mask_cpu = valid_mask.detach().cpu().numpy().astype(bool, copy=False)
            valid_count = int(valid_mask_cpu.sum())
            if valid_count == 0:
                continue

            if cfg.max_valid_samples is not None:
                remaining = int(cfg.max_valid_samples) - total_valid
                if remaining <= 0:
                    break
                if valid_count > remaining:
                    valid_indices = np.flatnonzero(valid_mask_cpu)[:remaining]
                    selected_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
                    selected_mask[torch.as_tensor(valid_indices, device=valid_mask.device)] = True
                    valid_mask = selected_mask
                    valid_mask_cpu = selected_mask.detach().cpu().numpy().astype(bool, copy=False)
                    valid_count = remaining

            valid_video = video[valid_mask]
            first_frame = valid_video[:, :, :1]
            target_frame = valid_video[:, :, 1:]

            reconstructions_by_format = {
                latent_format: reconstruct_from_video(
                    valid_video,
                    latent_format=latent_format,
                )
                for latent_format in cfg.latent_formats
            }

            batch_metadata = {
                key: _to_numpy_column(batch[key])[valid_mask_cpu]
                for key in METADATA_KEYS
                if key in batch
            }
            batch_metric_rows = compute_reconstruction_metrics(
                target_frame=target_frame,
                reconstructions_by_format=reconstructions_by_format,
            )
            for sample_idx in range(valid_count):
                sample_meta = {key: values[sample_idx] for key, values in batch_metadata.items()}
                for row_offset, latent_format in enumerate(cfg.latent_formats):
                    metric_row = batch_metric_rows[sample_idx + row_offset * valid_count]
                    metrics_rows.append(
                        {
                            **sample_meta,
                            "latent_format": latent_format,
                            "mse": metric_row["mse"],
                            "mae": metric_row["mae"],
                            "psnr_db": metric_row["psnr_db"],
                        }
                    )

                candidate = {
                    **sample_meta,
                    "first_frame": squeeze_single_frame(first_frame[sample_idx : sample_idx + 1])[0].detach().cpu(),
                    "target_frame": squeeze_single_frame(target_frame[sample_idx : sample_idx + 1])[0].detach().cpu(),
                    "reconstructions": {
                        latent_format: squeeze_single_frame(recon[sample_idx : sample_idx + 1])[0].detach().cpu()
                        for latent_format, recon in reconstructions_by_format.items()
                    },
                }
                reservoir_update(
                    visualized_samples,
                    candidate=candidate,
                    seen_count=total_valid + sample_idx + 1,
                    capacity=cfg.num_visualizations,
                    rng=np_rng,
                )

            total_valid += valid_count
            total_batches = batch_idx

            if batch_idx % PROGRESS_LOG_EVERY_BATCHES == 0 or batch_idx == len(dataloader):
                elapsed_s = max(time.perf_counter() - start_time, 1e-9)
                rate = total_valid / elapsed_s
                logging.info(
                    "Progress: batch=%d/%d valid_rows=%d rate=%.1f valid_rows/s",
                    batch_idx,
                    len(dataloader),
                    total_valid,
                    rate,
                )

            if cfg.max_valid_samples is not None and total_valid >= int(cfg.max_valid_samples):
                break

    if not metrics_rows:
        raise ValueError("No valid frame pairs were analyzed.")

    metrics_df = pd.DataFrame(metrics_rows)
    sort_columns = [column for column in ("index", "latent_format") if column in metrics_df.columns]
    if sort_columns:
        metrics_df = metrics_df.sort_values(sort_columns, ignore_index=True)
    summary_df = summarize_reconstruction_metrics(metrics_df)

    visualized_samples = sorted(
        visualized_samples,
        key=lambda sample: (
            int(sample.get("episode_index", 0)),
            int(sample.get("frame_index", 0)),
            int(sample.get("index", 0)),
        ),
    )

    render_reconstruction_grid(
        visualized_samples,
        latent_formats=cfg.latent_formats,
        output_path=output_dir / "reconstruction_grid.png",
    )

    visualized_rows: list[dict[str, Any]] = []
    for sample_num, sample in enumerate(visualized_samples, start=1):
        panel_name = f"sample_{sample_num:03d}_idx{int(sample.get('index', sample_num - 1)):06d}.png"
        render_sample_panel(sample, latent_formats=cfg.latent_formats, output_path=sample_panel_dir / panel_name)
        visualized_rows.append(
            {
                key: sample.get(key)
                for key in (*METADATA_KEYS, "index", "episode_index", "frame_index")
            }
            | {"panel_path": str((sample_panel_dir / panel_name).resolve())}
        )

    metrics_df.to_csv(output_dir / "per_sample_reconstruction_metrics.csv.gz", index=False)
    summary_df.to_csv(output_dir / "reconstruction_metrics_by_format.csv", index=False)
    pd.DataFrame(visualized_rows).to_csv(output_dir / "visualized_samples.csv", index=False)

    readme_path = build_readme(
        cfg=cfg,
        output_dir=output_dir,
        summary_df=summary_df,
        camera_key=camera_key or "<unknown>",
        analyzed_valid_rows=int(total_valid),
        visualized_samples=len(visualized_samples),
    )

    summary_payload = {
        "policy_path": str(cfg.policy.pretrained_path),
        "policy_type": str(cfg.policy.type),
        "dataset_repo_id": cfg.dataset_repo_id,
        "dataset_root": cfg.dataset_root,
        "episodes": cfg.episodes,
        "camera_key": camera_key,
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "max_valid_samples": None if cfg.max_valid_samples is None else int(cfg.max_valid_samples),
        "num_visualizations": int(cfg.num_visualizations),
        "latent_formats": list(cfg.latent_formats),
        "seed": int(cfg.seed),
        "analyzed_valid_rows": int(total_valid),
        "num_batches_processed": int(total_batches),
        "summary_by_format": summary_df.to_dict(orient="records"),
        "artifacts": sorted(p.name for p in output_dir.iterdir()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")

    checkpoint_meta = infer_checkpoint_metadata(cfg.policy.pretrained_path)
    headline_metrics = {}
    for _, row in summary_df.iterrows():
        prefix = str(row["latent_format"])
        headline_metrics[f"{prefix}_mean_mse"] = float(row["mean_mse"])
        headline_metrics[f"{prefix}_mean_psnr_db"] = float(row["mean_psnr_db"])

    analysis_manifest = {
        "artifact_type": "latent_analysis",
        "analysis_kind": "latent_reconstruction_visualization",
        "suite_name": "latent_reconstruction_visualization",
        "suite_version": "v1",
        "artifact_id": make_artifact_id(
            suite_name="latent_reconstruction_visualization",
            suite_version="v1",
            checkpoint_id=checkpoint_meta["source_checkpoint_id"],
            output_label=output_dir.name,
        ),
        **checkpoint_meta,
        "parent_export_artifact_id": None,
        "parent_export_manifest_path": None,
        "input_dataset_root": cfg.dataset_root,
        "input_dataset_repo_id": cfg.dataset_repo_id,
        "script_path": str(Path(__file__).resolve()),
        "cli_args": list(sys.argv[1:]),
        "output_path": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "readme_path": str(readme_path),
        "headline_metrics": {
            "analyzed_valid_rows": int(total_valid),
            "visualized_samples": len(visualized_samples),
            **headline_metrics,
        },
    }
    register_artifact(
        manifest_path=output_dir / "analysis_manifest.json",
        manifest=analysis_manifest,
        registry_candidates=[output_dir, cfg.policy.pretrained_path, cfg.dataset_root],
    )


@parser.wrap()
def main(cfg: LatentReconstructionConfig) -> None:
    register_third_party_plugins()
    init_logging()
    analyze_latent_reconstructions(cfg)


if __name__ == "__main__":
    main()
