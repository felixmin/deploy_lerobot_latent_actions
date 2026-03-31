# lerobot_latent_actions

Utilities for exporting latent labels onto LeRobot datasets.

The repo is intentionally general:
- it does not assume one specific latent-action model
- it does not assume one specific downstream policy
- the labeling script works with any installed policy plugin that follows the required export interface

The working example in this workspace is:
- latent-labeling policy: `lam_lapa`
- downstream policy: `latent_smolvla`

## What This Repo Covers

Typical workflow:

1. install LeRobot and one or more policy plugins
2. train or select a latent-action-model checkpoint
3. label a dataset with `scripts/label_lerobot_dataset.py`
4. train a downstream policy on those latent labels

Detailed guides:
- [Workflow](docs/workflow.md)
- [Feature Keys](docs/feature_keys.md)
- [Policy Interface](docs/policy_interface.md)
- [Worked Example: lam_lapa -> latent_smolvla](docs/examples/lam_lapa_to_latent_smolvla.md)

## Main Entrypoint

```bash
python scripts/label_lerobot_dataset.py --help
```

## Key Design Rules

- Use a top-level latent namespace such as `latent_labels` or `lam_lapa`.
- Do not write latent labels under `observation.*`.
- Feature provenance belongs in the manifest and dataset naming, not in the feature key itself.
- Downstream policies should reference the full dotted keys, e.g. `latent_labels.continuous_vector_latents` and `latent_labels.valid`.
- Keep latent validity separate from supervision routing:
  - labeled datasets provide `<prefix>.valid`
  - mixed datasets provide routing booleans such as `latent_supervision` and `action_supervision`

Why the `observation.*` rule matters:
- LeRobot expands every `observation.*` feature with observation delta timestamps.
- latent labels are supervision targets, not environment observations.
- placing them under `observation.*` can add unintended extra temporal axes.

For `latent_smolvla`, configured latent keys are preserved through preprocessing via complementary data, so the old `observation.latent.*` workaround is no longer needed.

## Standard Representation Names

Policies can expose any representations they want, but the standard names are:
- `codebook_id_latents`
- `codebook_vector_latents`
- `continuous_vector_latents`
- `valid`

With the default prefix `latent_labels`, the exported dataset fields become:
- `latent_labels.codebook_id_latents`
- `latent_labels.codebook_vector_latents`
- `latent_labels.continuous_vector_latents`
- `latent_labels.valid`

## Plugin Loading

The exporter uses normal LeRobot policy loading from `--policy.path`.

If the policy package is already installed in the environment, passing `--policy.path` is enough.

If you are developing a local policy repo without installing it, make it importable and tell LeRobot to discover it:

```bash
PYTHONPATH=/mnt/data/workspace/code/lerobot/src:/path/to/your_policy_package/src \
python scripts/label_lerobot_dataset.py \
  --policy.path=/path/to/checkpoint/pretrained_model \
  --policy.discover_packages_path=your_policy_package \
  ...
```

## Minimal Example

```bash
PYTHONPATH=/mnt/data/workspace/code/lerobot/src:/mnt/data/workspace/code/lerobot_policy_lam_lapa/src \
HF_LEROBOT_HOME=/mnt/data/workspace/runs_root/cache/huggingface/lerobot \
python scripts/label_lerobot_dataset.py \
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
