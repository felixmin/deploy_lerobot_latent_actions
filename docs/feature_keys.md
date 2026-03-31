# Feature Keys

## Recommended Naming Strategy

Use a top-level namespace for latent supervision.

Recommended default:
- `latent_labels.codebook_id_latents`
- `latent_labels.codebook_vector_latents`
- `latent_labels.continuous_vector_latents`
- `latent_labels.valid`

Alternative prefixes are fine if they are top-level, for example:
- `lam_lapa.continuous_vector_latents`
- `teacher_latents.continuous_vector_latents`

## What Not To Do

Do not store latent labels under `observation.*`, for example:
- `observation.latent.continuous_vector_latents`
- `observation.latent.valid`

Why not:
- LeRobot dataset construction treats every `observation.*` feature as an observation stream
- observation delta timestamps are applied automatically to those keys
- latent labels then pick up unintended extra temporal axes
- downstream losses may see the wrong rank or shape

## Why Top-Level Keys Work

Top-level latent keys are supervision targets, not observation inputs.

Policies that need them during preprocessing should preserve them explicitly.
For example, `latent_smolvla` preserves:
- `latent_label_key`
- `latent_valid_key`
- `latent_supervision_key`
- `action_supervision_key`

through complementary data before normalization/tokenization.

## Provenance

Feature keys should describe the type of supervision, not the producing experiment.

Prefer:
- `latent_labels.continuous_vector_latents`

Over:
- `lapa_lam_120000.continuous_vector_latents`
- `run42_teacher.continuous_vector_latents`

Put provenance in:
- `label_manifest.json`
- dataset naming
- checkpoint path
- experiment config

## Downstream Usage

A downstream policy should reference the full dotted key names explicitly.

Example for `latent_smolvla`:

```bash
--policy.latent_label_key=latent_labels.continuous_vector_latents \
--policy.latent_valid_key=latent_labels.valid \
--policy.latent_supervision_key=latent_supervision \
--policy.action_supervision_key=action_supervision
```

Here:
- `latent_labels.valid` means the latent target exists and is usable
- `latent_supervision` decides whether the latent branch should train on the sample
- `action_supervision` decides whether the action branch should train on the sample

## Standard Representation Names

The exporter is generic, but these representation names are the recommended standard:
- `codebook_id_latents`
- `codebook_vector_latents`
- `continuous_vector_latents`
- `valid`

If a policy does not expose some of them, that is fine. It should expose whichever representations it can produce consistently.
