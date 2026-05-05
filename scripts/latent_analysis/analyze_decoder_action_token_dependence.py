#!/usr/bin/env python

import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange

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

PROGRESS_LOG_EVERY_BATCHES = 50
METADATA_KEYS = ("index", "episode_index", "task_index", "frame_index", "timestamp")
CONDITION_TO_COLUMN = {
    "normal": "loss_normal",
    "hard_codebook": "loss_hard_codebook",
    "shuffled_action_tokens": "loss_shuffled_action_tokens",
    "zeroed_action_tokens": "loss_zeroed_action_tokens",
}
CONDITION_DESCRIPTIONS = {
    "normal": "Standard relaxed NSVQ reconstruction path.",
    "hard_codebook": "Hard nearest-codebook vectors decoded through the same decoder.",
    "shuffled_action_tokens": "Hard codebook action tokens batch-shuffled before decoding.",
    "zeroed_action_tokens": "Action tokens replaced with zeros before decoding.",
}


@dataclass
class DecoderTokenAblationConfig:
    policy: PreTrainedConfig | None = None
    dataset_repo_id: str | None = None
    dataset_root: str | None = None
    episodes: list[int] | None = None
    output_dir: Path | None = None
    batch_size: int = 32
    num_workers: int = 8
    force: bool = False
    max_valid_samples: int | None = None
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


def reshape_action_tokens(model: Any, tokens: torch.Tensor) -> torch.Tensor:
    action_h, action_w = model.action_shape
    return rearrange(tokens, "b (t h w) d -> b t h w d", h=action_h, w=action_w)


def _project_decoder_context(model: Any, first_frame: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "decoder_context_projection"):
        return model.decoder_context_projection(first_frame).detach()
    if hasattr(model, "pixel_projection"):
        return model.pixel_projection(first_frame).detach()
    raise TypeError("LAM model does not expose a known decoder context projection module.")


def compute_quantized_token_variants(
    model: Any,
    first_tokens_flat: torch.Tensor,
    last_tokens_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = first_tokens_flat.shape[0]
    first = model.vq.encode(first_tokens_flat.contiguous(), batch_size)
    last = model.vq.encode(last_tokens_flat.contiguous(), batch_size)
    delta = last - first

    distances = (
        torch.sum(delta**2, dim=1, keepdim=True)
        - 2 * torch.matmul(delta, model.vq.codebooks.t())
        + torch.sum(model.vq.codebooks.t() ** 2, dim=0, keepdim=True)
    )
    min_indices = torch.argmin(distances, dim=1)
    hard_quantized_input = model.vq.codebooks[min_indices]

    random_vector = torch.randn(delta.shape, device=delta.device, dtype=delta.dtype)
    norm_quantization_residual = (delta - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
    norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
    vq_error = (norm_quantization_residual / norm_random_vector + model.vq.eps) * random_vector
    relaxed_quantized_input = delta + vq_error

    relaxed_tokens = model.vq.decode(relaxed_quantized_input, batch_size)
    hard_tokens = model.vq.decode(hard_quantized_input, batch_size)
    return relaxed_tokens, hard_tokens, min_indices.reshape(batch_size, -1)


def decode_reconstruction_loss(
    model: Any,
    *,
    first_frame: torch.Tensor,
    last_frame: torch.Tensor,
    action_tokens: torch.Tensor,
) -> torch.Tensor:
    attn_bias = model.spatial_rel_pos_bias(model.grid_h, model.grid_w, device=first_frame.device)
    decoder_context = _project_decoder_context(model, first_frame)
    video_shape = tuple(decoder_context.shape[:-1])
    pixel_context = rearrange(decoder_context, "b t h w d -> (b t) (h w) d")
    action_context = rearrange(action_tokens, "b t h w d -> (b t) (h w) d")
    decoded = model.pixel_decoder(
        pixel_context,
        video_shape=video_shape,
        attn_bias=attn_bias,
        context=action_context,
    )
    decoded = rearrange(decoded, "(b t) (h w) d -> b t h w d", b=first_frame.shape[0], h=model.grid_h, w=model.grid_w)
    recon = model.pixel_to_pixels(decoded)
    return F.mse_loss(recon, last_frame, reduction="none").mean(dim=(1, 2, 3, 4))


def compute_condition_losses(model: Any, video: torch.Tensor, np_rng: np.random.Generator) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    video = model._normalize_video_input(video)
    first_frame = video[:, :, :1]
    last_frame = video[:, :, 1:]
    _, _, first_tokens_flat, last_tokens_flat = model._encode_frames(first_frame, last_frame)
    relaxed_tokens, hard_tokens, indices = compute_quantized_token_variants(model, first_tokens_flat, last_tokens_flat)

    normal_action_tokens = reshape_action_tokens(model, relaxed_tokens)
    hard_action_tokens = reshape_action_tokens(model, hard_tokens)
    if hard_action_tokens.shape[0] > 1:
        perm = torch.as_tensor(np_rng.permutation(hard_action_tokens.shape[0]), device=hard_action_tokens.device)
        shuffled_action_tokens = hard_action_tokens[perm]
    else:
        shuffled_action_tokens = hard_action_tokens
    zeroed_action_tokens = torch.zeros_like(hard_action_tokens)

    losses = {
        "normal": decode_reconstruction_loss(
            model,
            first_frame=first_frame,
            last_frame=last_frame,
            action_tokens=normal_action_tokens,
        ),
        "hard_codebook": decode_reconstruction_loss(
            model,
            first_frame=first_frame,
            last_frame=last_frame,
            action_tokens=hard_action_tokens,
        ),
        "shuffled_action_tokens": decode_reconstruction_loss(
            model,
            first_frame=first_frame,
            last_frame=last_frame,
            action_tokens=shuffled_action_tokens,
        ),
        "zeroed_action_tokens": decode_reconstruction_loss(
            model,
            first_frame=first_frame,
            last_frame=last_frame,
            action_tokens=zeroed_action_tokens,
        ),
    }
    return losses, indices


def _summarize_values(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64, copy=False)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def summarize_loss_table(loss_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    normal = loss_df[CONDITION_TO_COLUMN["normal"]].to_numpy(dtype=np.float64, copy=False)
    denom = np.maximum(normal, 1e-12)
    for condition, column in CONDITION_TO_COLUMN.items():
        values = loss_df[column].to_numpy(dtype=np.float64, copy=False)
        row = {
            "condition": condition,
            "description": CONDITION_DESCRIPTIONS[condition],
            "count": int(values.shape[0]),
            **_summarize_values(values),
        }
        summary_rows.append(row)

    comparison_rows: list[dict[str, Any]] = []
    for condition, column in CONDITION_TO_COLUMN.items():
        if condition == "normal":
            continue
        values = loss_df[column].to_numpy(dtype=np.float64, copy=False)
        delta = values - normal
        ratio = values / denom
        comparison_rows.append(
            {
                "condition": condition,
                "description": CONDITION_DESCRIPTIONS[condition],
                "count": int(values.shape[0]),
                "mean_delta_vs_normal": float(np.mean(delta)),
                "median_delta_vs_normal": float(np.median(delta)),
                "p95_delta_vs_normal": float(np.quantile(delta, 0.95)),
                "mean_ratio_vs_normal": float(np.mean(ratio)),
                "median_ratio_vs_normal": float(np.median(ratio)),
                "fraction_worse_than_normal": float(np.mean(delta > 0.0)),
                "fraction_better_than_normal": float(np.mean(delta < 0.0)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    return summary_df, comparison_df


def build_readme(
    *,
    cfg: DecoderTokenAblationConfig,
    output_dir: Path,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    camera_key: str,
    valid_rows: int,
    unique_id_sequences: int,
    latent_ablation: str,
) -> Path:
    lines = [
        "# Decoder Action Token Dependence",
        "",
        f"- Policy path: `{cfg.policy.pretrained_path}`",
        f"- Policy type: `{cfg.policy.type}`",
        f"- Dataset repo id: `{cfg.dataset_repo_id}`",
        f"- Dataset root: `{cfg.dataset_root}`",
        f"- Episodes: `{cfg.episodes}`",
        f"- Camera key: `{camera_key}`",
        f"- Valid rows analyzed: `{valid_rows}`",
        f"- Unique code sequences observed: `{unique_id_sequences}`",
        f"- Model latent_ablation setting: `{latent_ablation}`",
        "",
        "## Conditions",
    ]
    for condition in CONDITION_TO_COLUMN:
        lines.append(f"- `{condition}`: {CONDITION_DESCRIPTIONS[condition]}")

    lines.extend(["", "## Reconstruction Loss"])
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['condition']}`: mean=`{row['mean']:.6f}`, median=`{row['median']:.6f}`, p95=`{row['p95']:.6f}`, max=`{row['max']:.6f}`"
        )

    lines.extend(["", "## Delta Vs Normal"])
    for _, row in comparison_df.iterrows():
        lines.append(
            f"- `{row['condition']}`: mean delta=`{row['mean_delta_vs_normal']:.6f}`, mean ratio=`{row['mean_ratio_vs_normal']:.4f}`, fraction worse=`{row['fraction_worse_than_normal']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- If `shuffled_action_tokens` and `zeroed_action_tokens` stay close to `normal`, the decoder is likely ignoring action tokens.",
            "- If `hard_codebook` stays close to `normal`, the hard discrete codebook retains most of the useful decoder signal.",
            "- If `hard_codebook` is much worse than `normal`, the decoder relies on the relaxed/noisy NSVQ path more than the hard codebook vectors.",
        ]
    )

    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n")
    return readme_path


def analyze_decoder_action_token_dependence(cfg: DecoderTokenAblationConfig) -> None:
    cfg.validate()
    if cfg.policy is None:
        raise ValueError("Policy config was not loaded.")
    if cfg.output_dir is None:
        raise ValueError("output_dir was not configured.")
    if cfg.dataset_repo_id is None:
        raise ValueError("dataset_repo_id was not configured.")

    output_dir = cfg.output_dir.resolve()
    _prepare_output_dir(output_dir, cfg.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(cfg.seed))
    np_rng = np.random.default_rng(int(cfg.seed))

    source_dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root)
    policy = make_policy(cfg.policy, ds_meta=source_dataset.meta)
    policy.eval()

    prepare_latent_export = _get_required_method(policy, "prepare_latent_export")
    extract_frame_pair = _get_required_method(policy, "_extract_frame_pair")
    plan = _normalize_export_plan(prepare_latent_export(source_dataset.meta))

    model = getattr(policy, "lam", None)
    if model is None or not hasattr(model, "vq"):
        raise TypeError(f"Policy type {cfg.policy.type!r} does not expose a LAM model with a vector quantizer.")

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
        "Decoder token ablation setup: policy=%s dataset=%s output=%s batch_size=%d max_valid_samples=%s",
        cfg.policy.type,
        cfg.dataset_repo_id,
        output_dir,
        cfg.batch_size,
        cfg.max_valid_samples,
    )

    per_sample_rows: list[pd.DataFrame] = []
    id_rows: list[np.ndarray] = []
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
            losses_by_condition, indices = compute_condition_losses(model, valid_video, np_rng)

            batch_df = pd.DataFrame(
                {
                    key: _to_numpy_column(batch[key])[valid_mask_cpu]
                    for key in METADATA_KEYS
                    if key in batch
                }
            )
            for condition, column in CONDITION_TO_COLUMN.items():
                batch_df[column] = losses_by_condition[condition].detach().cpu().numpy().astype(np.float32, copy=False)
            per_sample_rows.append(batch_df)
            id_rows.append(indices.detach().cpu().numpy().astype(np.int64, copy=False))

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

    if not per_sample_rows:
        raise ValueError("No valid frame pairs were analyzed.")

    loss_df = pd.concat(per_sample_rows, ignore_index=True)
    loss_df = loss_df.sort_values("index", ignore_index=True)

    ids = np.concatenate(id_rows, axis=0)
    if ids.shape[0] != loss_df.shape[0]:
        raise ValueError("Mismatch between gathered losses and gathered codebook ids.")
    unique_id_sequences = int(np.unique(ids, axis=0).shape[0])
    loss_df["id_sequence"] = [" ".join(map(str, row.tolist())) for row in ids]

    summary_df, comparison_df = summarize_loss_table(loss_df)
    per_episode_df = (
        loss_df.groupby("episode_index", as_index=False)[list(CONDITION_TO_COLUMN.values())]
        .mean()
        .sort_values("episode_index", ignore_index=True)
        if "episode_index" in loss_df
        else pd.DataFrame()
    )

    per_sample_path = output_dir / "per_sample_reconstruction_losses.csv.gz"
    summary_path = output_dir / "reconstruction_loss_summary.csv"
    comparison_path = output_dir / "reconstruction_loss_vs_normal.csv"
    per_episode_path = output_dir / "reconstruction_loss_by_episode.csv"
    loss_df.to_csv(per_sample_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    if per_episode_df.shape[0] > 0:
        per_episode_df.to_csv(per_episode_path, index=False)

    readme_path = build_readme(
        cfg=cfg,
        output_dir=output_dir,
        summary_df=summary_df,
        comparison_df=comparison_df,
        camera_key=camera_key or "<unknown>",
        valid_rows=int(loss_df.shape[0]),
        unique_id_sequences=unique_id_sequences,
        latent_ablation=str(getattr(model, "latent_ablation", "unknown")),
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
        "seed": int(cfg.seed),
        "latent_ablation_setting": str(getattr(model, "latent_ablation", "unknown")),
        "valid_rows": int(loss_df.shape[0]),
        "num_batches_processed": int(total_batches),
        "unique_id_sequences": unique_id_sequences,
        "conditions": [
            {"condition": condition, "description": CONDITION_DESCRIPTIONS[condition]}
            for condition in CONDITION_TO_COLUMN
        ],
        "loss_summary": summary_df.to_dict(orient="records"),
        "vs_normal_summary": comparison_df.to_dict(orient="records"),
        "artifacts": sorted(p.name for p in output_dir.iterdir()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")

    checkpoint_meta = infer_checkpoint_metadata(cfg.policy.pretrained_path)

    def condition_metric(condition: str, column: str) -> float | None:
        rows = summary_df[summary_df["condition"] == condition]
        if rows.shape[0] == 0:
            return None
        return float(rows.iloc[0][column])

    def comparison_metric(condition: str, column: str) -> float | None:
        rows = comparison_df[comparison_df["condition"] == condition]
        if rows.shape[0] == 0:
            return None
        return float(rows.iloc[0][column])

    analysis_manifest = {
        "artifact_type": "latent_analysis",
        "analysis_kind": "decoder_token_ablation",
        "suite_name": "decoder_token_ablation",
        "suite_version": "v1",
        "artifact_id": make_artifact_id(
            suite_name="decoder_token_ablation",
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
            "valid_rows": int(loss_df.shape[0]),
            "unique_id_sequences": unique_id_sequences,
            "normal_mean_loss": condition_metric("normal", "mean"),
            "hard_codebook_mean_loss": condition_metric("hard_codebook", "mean"),
            "shuffled_action_tokens_mean_loss": condition_metric("shuffled_action_tokens", "mean"),
            "zeroed_action_tokens_mean_loss": condition_metric("zeroed_action_tokens", "mean"),
            "hard_codebook_mean_delta_vs_normal": comparison_metric("hard_codebook", "mean_delta_vs_normal"),
            "shuffled_action_tokens_mean_delta_vs_normal": comparison_metric(
                "shuffled_action_tokens", "mean_delta_vs_normal"
            ),
            "zeroed_action_tokens_mean_delta_vs_normal": comparison_metric(
                "zeroed_action_tokens", "mean_delta_vs_normal"
            ),
            "hard_codebook_mean_ratio_vs_normal": comparison_metric("hard_codebook", "mean_ratio_vs_normal"),
            "shuffled_action_tokens_mean_ratio_vs_normal": comparison_metric(
                "shuffled_action_tokens", "mean_ratio_vs_normal"
            ),
            "zeroed_action_tokens_mean_ratio_vs_normal": comparison_metric(
                "zeroed_action_tokens", "mean_ratio_vs_normal"
            ),
        },
    }
    register_artifact(
        manifest_path=output_dir / "analysis_manifest.json",
        manifest=analysis_manifest,
        registry_candidates=[output_dir, cfg.policy.pretrained_path, cfg.dataset_root],
    )


@parser.wrap()
def main(cfg: DecoderTokenAblationConfig) -> None:
    register_third_party_plugins()
    init_logging()
    analyze_decoder_action_token_dependence(cfg)


if __name__ == "__main__":
    main()
