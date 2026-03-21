# Refactor Plan: Generic Latent Label Export

## Goal

Turn `lerobot_latent_actions` into a generic latent-label export tool rather than a script with policy-specific `lam` / `dismo` branches.

The desired end state is:

1. the relabel script loads a policy through normal LeRobot plugin discovery
2. the policy exposes a standard latent-export interface
3. the script uses that interface to generate labels
4. the script writes labels back into a LeRobot dataset copy with `add_features(...)`

This keeps temporal logic and latent-format logic inside each labeling policy rather than in the generic relabel script.

---

## Current State

### What already works

- standalone relabel script exists:
  - [scripts/label_lerobot_dataset.py](/mnt/data/workspace/code/lerobot_latent_actions/scripts/label_lerobot_dataset.py)
- standalone latent policy exists:
  - [../lerobot_policy_latent_smolvla](/mnt/data/workspace/code/lerobot_policy_latent_smolvla)
- LA-PA now has a public latent export method:
  - [../lerobot_policy_lapa_lam/src/lerobot_policy_lam/modeling_lam.py](/mnt/data/workspace/code/lerobot_policy_lapa_lam/src/lerobot_policy_lam/modeling_lam.py)
- ID relabeling works end to end
- `latent_smolvla` CE smoke works end to end on relabeled LIBERO data

### What is still wrong architecturally

- the relabel script still branches on policy type (`lam`, `dismo`)
- the relabel script still owns policy-specific temporal logic
- the relabel script still knows too much about latent export internals

That means the current script is usable, but not the right long-term abstraction.

---

## Why Refactor

### 1. Policy-specific temporal logic should not live in the generic script

Right now the script manually does things like:

- build valid forward frame pairs for LA-PA
- implement custom lookahead logic for DISMO

That is brittle and does not scale.

The policy should decide:

- what temporal context it needs
- whether it uses past frames, future frames, or both
- how padding should be interpreted
- whether a sample is valid enough to produce a label

### 2. LeRobot already provides temporal querying

`LeRobotDataset` already supports policy-style temporal requests through `delta_timestamps` / delta indices.

Relevant code:

- [../high-level-robot-planner/lerobot/src/lerobot/datasets/factory.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/factory.py)
- [../high-level-robot-planner/lerobot/src/lerobot/datasets/lerobot_dataset.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/lerobot_dataset.py)

This means the relabel flow should reuse the dataset’s temporal retrieval instead of re-implementing it.

### 3. Policies should expose latent export as a public inference API

The generic relabel script should not know how to compute:

- codebook IDs
- codebook vectors
- continuous motion embeddings

Instead, each labeling policy should expose a public method for this.

---

## Target Design

## Generic relabel script

The script should:

1. load a policy by type using LeRobot/plugin discovery
2. load checkpoint weights from a local path
3. ask the policy what temporal context it needs
4. create a `LeRobotDataset` using that temporal request
5. iterate batches
6. call the policy’s latent-export method
7. write labels + valid mask back with `add_features(...)`

The script should **not**:

- know what a LAM is
- know what DISMO windows are
- know how to compute ids vs vectors internally

---

## Proposed interface

Add a small shared interface in this repo, for example:

- [src/lerobot_latent_actions/interfaces.py](/mnt/data/workspace/code/lerobot_latent_actions/src/lerobot_latent_actions/interfaces.py)

Suggested structures:

```python
@dataclass
class LatentLabelRequest:
    delta_timestamps: dict[str, list[float]] | None = None


@dataclass
class LatentLabelSpec:
    dtype: str
    shape: tuple[int, ...]
    invalid_fill_value: int | float


@dataclass
class LatentLabelBatch:
    labels: torch.Tensor
    valid_mask: torch.BoolTensor
```

And a policy-side contract:

```python
class SupportsLatentLabelExport(Protocol):
    def get_latent_label_request(self, dataset_meta) -> LatentLabelRequest: ...
    def get_latent_label_spec(self, latent_format: str) -> LatentLabelSpec: ...
    def export_latent_labels(
        self,
        batch: dict[str, Any],
        *,
        latent_format: str,
    ) -> LatentLabelBatch: ...
```

Duck typing is sufficient; a hard base class is not required.

---

## Policy responsibilities

Each labeling policy should own:

- temporal request
- latent format support
- valid-mask logic
- export implementation

### LA-PA

`LAMPolicy` should expose:

- `get_latent_label_request(...)`
- `get_latent_label_spec(...)`
- `export_latent_labels(...)`

This should wrap the already-added public latent extraction path in:

- [../lerobot_policy_lapa_lam/src/lerobot_policy_lam/modeling_lam.py](/mnt/data/workspace/code/lerobot_policy_lapa_lam/src/lerobot_policy_lam/modeling_lam.py)

### DISMO

`DisMoPolicy` should expose the same three methods.

This allows DISMO to define its own past/future requirements without the generic script knowing them.

---

## Script responsibilities

After refactor, the script should only orchestrate:

1. policy loading
2. dataset creation
3. batch iteration
4. writing features back

Suggested helper functions:

- `load_labeling_policy(policy_type, policy_path, device)`
- `build_label_dataset(repo_id, root, request, episodes=None)`
- `allocate_label_arrays(dataset_len, spec)`
- `run_label_export(policy, dataset, latent_format, batch_size, max_samples=None)`
- `write_labeled_dataset(dataset, labels, valid_mask, feature_name, valid_feature_name, output_dir, output_repo_id)`

---

## Dataset-writing rules

### Full-length arrays

The relabeler should always allocate full-length arrays for the source dataset:

- `labels.shape == (N, ...)`
- `valid_mask.shape == (N, 1)`

where `N = len(source_dataset)` in the source layout used for `add_features(...)`.

Unlabeled samples should be written as:

- invalid fill value for the label tensor
- `valid_mask = 0`

This matches the current `add_features(...)` implementation better than trying to rewrite filtered subsets.

### Default feature names

Prefer generic names by default:

- `latent_labels`
- `latent_supervised`

Source-specific names like:

- `lapa_ids`
- `lapa_vectors`

should remain optional CLI overrides.

---

## What stays the same

This refactor is mostly an abstraction cleanup.

It does **not** invalidate the currently working path:

- ID relabeling is already working
- CE latent training smoke is already working

So this refactor is mainly about making the system:

- policy-agnostic
- easier to extend
- better aligned with LeRobot temporal querying

---

## Known `add_features(...)` issues found during this work

### 1. Multi-dimensional feature write bug

Vector features exposed a pandas assignment bug in:

- [../high-level-robot-planner/lerobot/src/lerobot/datasets/dataset_tools.py](/mnt/data/workspace/code/high-level-robot-planner/lerobot/src/lerobot/datasets/dataset_tools.py)

This has been patched locally so multi-dimensional per-frame features are assigned row-wise.

### 2. Subset relabeling limitation

`add_features(...)` still rewrites using the source parquet layout.

That means using a filtered dataset object such as only episode `0` does **not** automatically make feature-array lengths match the original file layout.

So for now, the safest write-back path is:

- keep full-length feature arrays
- mark invalid / unlabeled rows with a validity mask

---

## Recommended implementation order

1. add shared latent-export interface types in this repo
2. move LA-PA fully onto that interface
3. refactor the script to remove `lam` / `dismo` branching
4. revalidate the working ID path
5. add DISMO implementation
6. finish vector-label dataset generation
7. run `latent_smolvla` smoke with `vector_diffusion`

---

## Bottom line

The current code is already useful, but too policy-specific.

The refactor should make:

- the script generic
- the policies responsible for their own latent export
- the dataset responsible for temporal retrieval
- the writer responsible only for storing labels and masks

That is the right long-term structure for latent-label generation.
