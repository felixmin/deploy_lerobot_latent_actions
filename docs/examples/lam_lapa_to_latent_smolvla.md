# Worked Example: `lam_lapa` -> `latent_smolvla`

This is the concrete workflow that currently works in this workspace.

## Goal

1. train a latent-action model with `policy.type=lam_lapa`
2. export its latent labels onto a LIBERO dataset
3. train `policy.type=latent_smolvla` against those labels

The exporter itself stays generic. `lam_lapa` is just one concrete teacher policy.

## 1. Install The Policy Packages

```bash
pip install -e /mnt/data/workspace/code/lerobot_policy_lam_lapa
pip install -e /mnt/data/workspace/code/lerobot_policy_latent_smolvla
```

Or use `PYTHONPATH` during development:

```bash
export PYTHONPATH=/mnt/data/workspace/code/lerobot/src:/mnt/data/workspace/code/lerobot_policy_lam_lapa/src:/mnt/data/workspace/code/lerobot_policy_latent_smolvla/src
```

## 2. Train `lam_lapa`

Example:

```bash
lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.type=lam_lapa \
  --policy.push_to_hub=false \
  --policy.camera_key=observation.images.image \
  --policy.future_frames=10 \
  --batch_size=8 \
  --steps=200
```

After training, pick the checkpoint directory:
- `/path/to/run/checkpoints/000010/pretrained_model`

## 3. Label The Dataset

Use `latent_labels` as the feature prefix unless you have a strong reason to expose teacher-specific naming in the dataset schema.

```bash
PYTHONPATH=/mnt/data/workspace/code/lerobot/src:/mnt/data/workspace/code/lerobot_policy_lam_lapa/src \
HF_LEROBOT_HOME=/mnt/data/workspace/runs_root/cache/huggingface/lerobot \
python /mnt/data/workspace/code/lerobot_latent_actions/scripts/label_lerobot_dataset.py \
  --policy.path=/mnt/data/workspace/runs_root/runs_lerobot/outputs/train/2026-03-30/lam_lapa_10step_smoke2/checkpoints/000010/pretrained_model \
  --policy.discover_packages_path=lerobot_policy_lam_lapa \
  --dataset_repo_id=HuggingFaceVLA/libero \
  --dataset_root=/mnt/data/workspace/runs_root/cache/huggingface/lerobot/HuggingFaceVLA/libero \
  --output_dir=/mnt/data/workspace/runs_root/runs_lerobot/latent_labels/libero_lam_lapa_10step \
  --output_repo_id=rlfv/libero_lam_lapa_10step \
  --feature_prefix=latent_labels \
  --policy.camera_key=observation.images.image \
  --policy.future_frames=10 \
  --batch_size=32 \
  --num_workers=8
```

This writes:
- `latent_labels.codebook_id_latents`
- `latent_labels.codebook_vector_latents`
- `latent_labels.continuous_vector_latents`
- `latent_labels.valid`

## 4. Train `latent_smolvla`

Example command shape:

```bash
lerobot-train \
  --policy.type=latent_smolvla \
  --dataset.mix_path=/path/to/mixed_dataset.yaml \
  --dataset.mix_implementation=current \
  --policy.training_mode=multitask \
  --policy.latent_head_mode=vector_diffusion \
  --policy.latent_label_key=latent_labels.continuous_vector_latents \
  --policy.latent_valid_key=latent_labels.valid \
  --policy.latent_supervision_key=latent_supervision \
  --policy.action_supervision_key=action_supervision \
  --policy.latent_code_seq_len=4 \
  --policy.latent_vector_dim=128
```

## 5. Key Split

The split is:
- labeled dataset:
  - `latent_labels.continuous_vector_latents`
  - `latent_labels.valid`
- mixed dataset routing:
  - `latent_supervision`
  - `action_supervision`

## 6. Generalization

The same exporter workflow works for any other teacher policy, as long as that policy:
- loads from `--policy.path`
- implements `prepare_latent_export(dataset_meta)`
- implements `export_latent_labels(batch)`

See [Policy Interface](../policy_interface.md).
