# Analysis

`lerobot_latent_actions` provides two complementary analysis entrypoints:

- `scripts/analyze_latent_feature_distribution.py`
- `scripts/analyze_spcfc.py`

Use them for different questions.

## 1. Labeled-Dataset Analysis

`analyze_latent_feature_distribution.py` operates on an already labeled dataset and a feature prefix such as
`latent_labels`.

It computes:
- validity counts
- codebook ID usage statistics
- mutual information (`MI`) and normalized mutual information (`NMI`) between discrete IDs and binned action targets
- PCA scatter plots for continuous and codebook-vector latents
- held-out action probes from latent features to actions

Example:

```bash
python scripts/analyze_latent_feature_distribution.py \
  --dataset-root=/path/to/labeled_dataset \
  --feature-prefix=latent_labels \
  --output-dir=/path/to/analysis
```

### Action Probe Metrics

The action probe fits a small regressor from latent features to real robot actions.

Available probe backends:
- `ridge`
- `mlp`
- `both`

Available split modes:
- `row`
- `episode`

Important outputs:
- `action_probe_scores.csv`
- `action_probe_scores_summary.csv`
- `action_probe_r2_heatmap__<probe_model>__<split_mode>.png`
- `action_probe_mse_heatmap__<probe_model>__<split_mode>.png`

#### `R^2`

`R^2` measures how much variance in the real action is recoverable from the latent.

Range:
- `1.0`: perfect prediction
- `0.0`: same as predicting the mean action
- `< 0.0`: worse than the mean baseline

Interpretation:
- higher is better
- useful for ranking latent representations on the same dataset

#### `MSE`

`MSE` is the mean squared prediction error of the action probe.

Range:
- `0.0`: perfect prediction
- no fixed upper bound

Interpretation:
- lower is better
- absolute values depend on the action normalization and probe setup
- best used for comparisons within the same dataset and probe protocol

### `MI` and `NMI`

These are discrete-code diagnostics.

The script quantile-bins each action dimension and then measures the association between:
- the full ID sequence
- each ID position
- and the binned action targets

#### `MI`

Range:
- `0.0`: no dependence
- larger values mean stronger association
- no fixed upper bound

#### `NMI`

Range:
- typically `0.0` to `1.0`
- `0.0`: no association
- `1.0`: perfect deterministic association

Interpretation:
- useful for seeing whether the discrete codebook structure is action-relevant
- not a replacement for the action probe

## 2. Checkpoint-Based S-PCFC Analysis

`analyze_spcfc.py` computes CoMo-style S-PCFC from:
- a latent-action checkpoint
- the raw dataset

This script is separate because true S-PCFC cannot be derived from the forward-labeled dataset alone. It needs to
re-encode:
- `z(o_{t-n}, o_t)`
- `z(o_{t+n}, o_t)`

Example:

```bash
python scripts/analyze_spcfc.py \
  --policy-path=/path/to/checkpoint/pretrained_model \
  --dataset-repo-id=HuggingFaceVLA/libero \
  --dataset-root=/path/to/libero \
  --output-dir=/path/to/spcfc \
  --offset-frames=10
```

### `S-PCFC`

S-PCFC is the cosine similarity between:
- past-to-current latent motion
- future-to-current latent motion

Formula:

```text
S-PCFC(t) = cos(z(o_{t-n}, o_t), z(o_{t+n}, o_t))
```

Range:
- cosine similarity is in `[-1, 1]`
- in practice these latent comparisons are usually in `[0, 1]`

Interpretation:
- lower is better in the CoMo framing
- lower means less static/background redundancy and better motion fidelity

The script currently supports:
- `continuous`
- `codebook_vectors`

It intentionally does not compute S-PCFC on raw ID one-hot vectors.

## Choosing A Metric

Use `R^2` when:
- you want the clearest “how much action information is linearly recoverable?” metric

Use probe `MSE` when:
- you want a metric closer to the CoMo action-prediction setup

Use `MI` / `NMI` when:
- you want to inspect whether discrete code usage is related to actions at all

Use `S-PCFC` when:
- you want a motion-fidelity / static-redundancy diagnostic rather than a direct action-prediction metric

## Reading More

Code:
- `scripts/analyze_latent_feature_distribution.py`
- `scripts/analyze_spcfc.py`

Paper context:
- CoMo defines Action Prediction `MSE` and `S-PCFC` as complementary latent-motion diagnostics
- see the paper section usually titled `MSE and S-PCFC` for the exact metric motivation and formula
