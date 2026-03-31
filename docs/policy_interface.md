# Policy Interface

`label_lerobot_dataset.py` is policy-agnostic. It works with any installed policy that satisfies the export interface below.

## Policy Loading Requirements

The policy must:
- be loadable from `--policy.path` via normal LeRobot config loading
- be importable because the package is installed or discoverable through `--policy.discover_packages_path`
- implement the two export methods described below

The exporter loads the policy config first and then calls `make_policy(...)` to instantiate the policy.

## Required Methods

A compatible policy must implement:
- `prepare_latent_export(dataset_meta)`
- `export_latent_labels(batch)`

## `prepare_latent_export(dataset_meta)`

This method defines what the exporter should read and write.

It must return a dict with:
- `delta_timestamps`
- `representations`

### `delta_timestamps`

A mapping from dataset feature names to relative timestamps, used to build the LeRobot dataset loader.

Example:

```python
{
    "observation.images.image": [0.0, 1.0],
}
```

### `representations`

A mapping from representation name to metadata.

Example:

```python
{
    "continuous_vector_latents": {
        "shape": (4, 32),
        "dtype": "float32",
        "invalid_fill_value": 0.0,
    },
    "codebook_id_latents": {
        "shape": (4,),
        "dtype": "int64",
        "invalid_fill_value": -1,
    },
}
```

Each representation spec must provide:
- `shape`
- `dtype`
- `invalid_fill_value`

The exporter writes each representation as `<feature_prefix>.<name>`.

## `export_latent_labels(batch)`

This method runs on actual batches and must return a dict with:
- `labels_by_name`
- `valid_mask`

### `valid_mask`

`valid_mask` marks which rows in the incoming batch have valid supervision.

Requirements:
- tensor-like
- shape `[batch_size]`
- boolean or convertible to boolean

### `labels_by_name`

A mapping from representation name to a tensor-like value.

Requirements:
- keys must match the names declared in `prepare_latent_export(...)["representations"]`
- each tensor's first dimension must equal the number of valid rows in `valid_mask`
- remaining dimensions must match the declared representation shape

Example:

```python
{
    "labels_by_name": {
        "continuous_vector_latents": torch.randn(num_valid, 4, 32),
        "codebook_id_latents": torch.randint(0, 8, (num_valid, 4)),
    },
    "valid_mask": torch.tensor([True, False, True, True]),
}
```

In this example, every tensor in `labels_by_name` must have first dimension `3` because there are three valid rows.

## Standard Representation Names

The exporter does not require fixed names, but the recommended standard is:
- `codebook_id_latents`
- `codebook_vector_latents`
- `continuous_vector_latents`

The exporter also adds:
- `<feature_prefix>.valid`

That validity feature is not the same thing as per-sample branch routing. Routing masks such as
`latent_supervision` and `action_supervision` should come from the mixed dataset layer.

## Summary

A compatible policy plugin needs to do three things:
- load cleanly from a checkpoint path
- declare the latent export plan with `prepare_latent_export(...)`
- emit batched latent labels with `export_latent_labels(...)`

That is enough for `scripts/label_lerobot_dataset.py` to relabel a dataset without adding policy-specific branches to the exporter.
