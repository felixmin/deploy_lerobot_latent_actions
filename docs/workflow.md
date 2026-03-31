# Workflow

This document describes the general latent-labeling workflow and then shows the working example used in this workspace.

## Overview

There are three roles:
- a latent-labeling policy that can read a LeRobot batch and emit latent targets
- a labeled dataset that stores those targets as extra features
- a downstream policy that consumes those features during training

In this workspace the concrete example is:
- teacher / labeler: `lam_lapa`
- student / downstream policy: `latent_smolvla`

## 1. Install LeRobot And Policy Plugins

You need:
- a LeRobot checkout or install
- a latent-labeling policy plugin
- optionally a downstream policy plugin

Example local development setup:

```bash
export PYTHONPATH=/mnt/data/workspace/code/lerobot/src:/mnt/data/workspace/code/lerobot_policy_lam_lapa/src:/mnt/data/workspace/code/lerobot_policy_latent_smolvla/src
```

Or install plugins into the active environment:

```bash
pip install -e /mnt/data/workspace/code/lerobot_policy_lam_lapa
pip install -e /mnt/data/workspace/code/lerobot_policy_latent_smolvla
```

## 2. Train Or Select A Latent-Action Checkpoint

`label_lerobot_dataset.py` does not train the latent-action model. It loads an existing checkpoint from `--policy.path`.

Requirements for that checkpoint:
- it must load through normal LeRobot `PreTrainedConfig.from_pretrained(...)`
- the policy package must be importable or installed
- the policy implementation must satisfy the export interface described in [Policy Interface](policy_interface.md)

Working example:
- train `lam_lapa`
- use the resulting `pretrained_model` directory as `--policy.path`

## 3. Label A Dataset

Run:

```bash
python scripts/label_lerobot_dataset.py \
  --policy.path=/path/to/checkpoint/pretrained_model \
  --policy.discover_packages_path=your_policy_package \
  --dataset_repo_id=HuggingFaceVLA/libero \
  --dataset_root=/path/to/libero \
  --output_dir=/path/to/output_dataset \
  --output_repo_id=your_namespace/your_labeled_dataset \
  --feature_prefix=latent_labels
```

Important points:
- `feature_prefix` should be a top-level namespace such as `latent_labels`
- do not use `observation.latent`
- the output dataset includes the original source data plus extra latent supervision features
- the exporter writes a `label_manifest.json` describing the origin and feature names

## 4. Inspect The Labeled Dataset

Check:
- `meta/info.json`
- `label_manifest.json`

You should see feature names like:
- `latent_labels.continuous_vector_latents`
- `latent_labels.valid`

If the policy exposes quantized representations, you may also see:
- `latent_labels.codebook_id_latents`
- `latent_labels.codebook_vector_latents`

## 5. Train A Downstream Policy

A downstream policy must know which feature keys to read.

For `latent_smolvla`, the typical settings are:

```bash
--policy.latent_label_key=latent_labels.continuous_vector_latents
--policy.latent_valid_key=latent_labels.valid
--policy.latent_supervision_key=latent_supervision
--policy.action_supervision_key=action_supervision
```

`latent_smolvla` preserves these configured keys through its preprocessing wrapper, so it can consume top-level latent supervision keys without needing the `observation.*` workaround.

## 6. Optional Mixed-Supervision Setup

If part of the dataset has action supervision and the rest is latent-only, use a mixed dataset config and gate losses with:
- `latent_valid_key`
- `latent_supervision_key`
- `action_supervision_key`

The labeled dataset should provide the latent validity mask:
- `latent_labels.valid`

The mixed dataset should provide the branch-routing booleans:
- `latent_supervision`
- `action_supervision`

This is the setup used in the `lam_lapa -> latent_smolvla` example.

## Worked Example Summary

1. train `lam_lapa`
2. export labels from its checkpoint onto LIBERO
3. use a top-level prefix such as `latent_labels`
4. train `latent_smolvla` with:
   - `latent_labels.continuous_vector_latents`
   - `latent_labels.valid`
   - `latent_supervision`
   - `action_supervision`

The detailed concrete commands are in [Worked Example: lam_lapa -> latent_smolvla](examples/lam_lapa_to_latent_smolvla.md).
