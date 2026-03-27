# lerobot_latent_actions

Standalone utilities for generating offline latent-action labels for LeRobot datasets.

Current scope:

- relabel an existing LeRobot dataset with labels produced by an offline policy
- write those labels back as new dataset features using LeRobot `add_features`

Initial supported labelers:

- `lapa_lam` via `lerobot_policy_lapa_lam`
- `dismo` via `lerobot_policy_dismo`

The main entrypoint is:

```bash
python scripts/label_lerobot_dataset.py --help
```

For the newer DataLoader-based path:

```bash
python scripts/label_lerobot_dataset_v2.py --help
```

For the strict multi-format exporter:

```bash
python scripts/label_lerobot_dataset_v3.py --help
```

Notes:

- `lapa_lam --latent-format ids` is the directly compatible mode for `latent_smolvla`
- `label_lerobot_dataset_v3.py` writes all three LAM outputs at once:
  `prefix.codebook_ids`, `prefix.continuous`, `prefix.codebook_vectors`
- `dismo` currently exports continuous motion embeddings, which are useful for analysis or future consumers but are not directly compatible with the current cross-entropy latent-label path
