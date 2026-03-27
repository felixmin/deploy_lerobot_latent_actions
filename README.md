# lerobot_latent_actions

Minimal offline latent-label export for LeRobot datasets.

The repo has one supported entrypoint:

```bash
python scripts/label_lerobot_dataset.py --help
```

## Design

- one script
- no numbered exporter versions
- no policy-type branches in the script
- policy defaults come from `--policy.path`
- any policy field can be overridden with `--policy.<field>=...`

The exporter expects the loaded policy to implement two methods:

- `prepare_latent_export(dataset_meta, *, representation)`
- `export_latent_labels(batch, *, representation)`

## Standard representations

Quantized latent policies should expose:

- `codebook_id_latents`
- `codebook_vector_latents`
- `continuous_vector_latents`

`continuous_vector_latents` is the generic continuous latent-vector name. For quantized policies it is the pre-quantization latent vector. Policies without a quantizer, such as DisMo, can still expose it for their continuous latent embeddings.

## Policy loading

The script uses normal LeRobot config loading and plugin discovery.

Make sure the intended LeRobot package is on `PYTHONPATH` or installed in the active environment. In this workspace that usually means:

```bash
PYTHONPATH=/mnt/data/workspace/code/lerobot/src
```

If the policy package is installed in the environment, `--policy.path` is enough.

If you are developing a local policy repo without installing it, make the package importable and tell the parser to load it:

```bash
PYTHONPATH=/mnt/data/workspace/code/lerobot/src:/path/to/lerobot_policy_lapa_lam/src \
python scripts/label_lerobot_dataset.py \
  --policy.path=/path/to/checkpoint/pretrained_model \
  --policy.discover_packages_path=lerobot_policy_lapa_lam \
  ...
```

## Example

```bash
PYTHONPATH=/mnt/data/workspace/code/lerobot/src:/mnt/data/workspace/code/lerobot_policy_lapa_lam/src \
HF_LEROBOT_HOME=/mnt/data/workspace/runs_root/cache/huggingface/lerobot \
python scripts/label_lerobot_dataset.py \
  --policy.path=/mnt/data/workspace/runs_root/runs_lerobot/outputs/train/2026-03-27/00-34-07_lapa_lam/checkpoints/120000/pretrained_model \
  --policy.discover_packages_path=lerobot_policy_lapa_lam \
  --dataset_repo_id=HuggingFaceVLA/libero \
  --dataset_root=/mnt/data/workspace/runs_root/cache/huggingface/lerobot/HuggingFaceVLA/libero \
  --output_dir=/mnt/data/workspace/runs_root/runs_lerobot/latent_labels/libero_lapa_lam_120000_ids \
  --latent_representation=codebook_id_latents \
  --policy.camera_key=observation.images.image \
  --policy.future_frames=10 \
  --batch_size=32 \
  --num_workers=8
```

This writes:

- `latent_labels`
- `latent_supervised`

For token-style supervision, use `codebook_id_latents`.
