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
- [Analysis](docs/analysis.md)
- [Feature Keys](docs/feature_keys.md)
- [Policy Interface](docs/policy_interface.md)
- [Worked Example: lam_lapa -> latent_smolvla](docs/examples/lam_lapa_to_latent_smolvla.md)

## Main Entrypoint

```bash
python scripts/label_lerobot_dataset.py --help
```

Optional analysis entrypoints:

```bash
python scripts/analyze_latent_feature_distribution.py --help
python scripts/analyze_spcfc.py --help
```

## Key Design Rules

- Use `latent_labels` as the default top-level namespace.
- Do not write latent labels under `observation.*`.
- Feature provenance belongs in the manifest and dataset naming, not in the feature key itself.
- Downstream policies should reference the full dotted keys, e.g. `latent_labels.continuous_vector_latents` and `latent_labels.valid`.
- Keep latent validity separate from supervision routing:
  - labeled datasets provide `<prefix>.valid`
  - mixed datasets provide routing booleans such as `latent_supervision` and `action_supervision`

For most users, that means:
- exported latent targets:
  - `latent_labels.continuous_vector_latents`
  - `latent_labels.valid`
- mixed-dataset routing:
  - `latent_supervision`
  - `action_supervision`

Keep the root README short. Use the detailed docs for actual commands:
- [Workflow](docs/workflow.md)
- [Analysis](docs/analysis.md)
- [Feature Keys](docs/feature_keys.md)
- [Policy Interface](docs/policy_interface.md)
- [Worked Example: lam_lapa -> latent_smolvla](docs/examples/lam_lapa_to_latent_smolvla.md)
