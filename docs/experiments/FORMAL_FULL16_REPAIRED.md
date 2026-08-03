# Formal full-context 16-run repaired matrix

This matrix prepares the next IEMOCAP clean-feature full-context batch. It is
configuration-only: the preparation command writes resolved YAML files and a
command list, but it does not import a training entrypoint, read a checkpoint,
or start an epoch.

## Matrix contract

- Matrix and experiment group: `formal_full16_repaired_seed42`
- Models: MMGCN, MultiDAG-CL Project Variant, DialogueGCN repaired, and
  GS-MCC Project Variant repaired
- Validation sessions: Ses01, Ses02, Ses03, and Ses04
- Fixed test session: Ses05
- Seed: 42
- Context: full-context only
- Checkpoint selection: validation Weighted-F1
- Test use for checkpoint, early-stopping, epoch, hyperparameter, or model
  selection: forbidden
- Feature entry: `clean_roberta_v1`
- Feature SHA256:
  `c604c557bc3fbb129ca02b2acd57166b669a89ef76ff0cea1e14f9cb206324bf`

The resulting expansion is `4 models × 4 validation sessions = 16 runs`.
Run IDs use `ff16r_<model>_full_context_val_<session>_s42`.

## Training signatures

DialogueGCN uses one repaired signature on every validation session:

```text
learning_rate=3e-4
max_epochs=60
early_stopping_min_epochs=30
early_stopping_patience=20
```

GS-MCC Project Variant uses one uniform delayed-stop signature on every
validation session:

```text
learning_rate=1e-5
max_epochs=250
early_stopping_min_epochs=90
early_stopping_patience=40
```

The matrix deliberately does not use per-session best-observed GS-MCC
settings and does not apply the 3e-5 diagnostic candidate to Ses01/Ses02.
MMGCN and MultiDAG-CL reference the existing `primary_seed42` full-context
base configs with no model, training, optimizer, or scheduler overrides.
The old Long32 matrix and base files remain unchanged; no duplicate base YAML
is created.

## Implementation identities

- MMGCN: unified/project implementation identity
- MultiDAG-CL: Project Variant, not author-official
- DialogueGCN: paper-aligned repaired, not author-official
- GS-MCC: GS-MCC Project Variant, not author-official

The source full-context bases retain their existing code-lineage metadata.
Each generated config also records the reporting identity above under
`formal_full16.implementation_identity` and explicitly sets
`formal_full16.author_official_reproduction: false`.

## Commands

Read-only structural and config validation:

```powershell
python scripts/experiments/prepare_formal_full16_repaired.py `
  --matrix configs/benchmarks/formal_rerun/iemocap_clean/full16_repaired_seed42.yaml `
  --mode check
```

Materialize one future batch without training:

```powershell
python scripts/experiments/prepare_formal_full16_repaired.py `
  --matrix configs/benchmarks/formal_rerun/iemocap_clean/full16_repaired_seed42.yaml `
  --mode prepare `
  --experiment-date 20260803 `
  --batch-id formal_full16_repaired_20260803_000000
```

Preparation creates the canonical experiment directories below
`outputs/<experiment_date>/formal_full16_repaired_seed42/`. The only files it
writes are:

```text
manifests/batches/<batch_id>/resolved_configs/<run_id>.yaml
manifests/batches/<batch_id>/commands.txt
```

An existing batch directory, an existing fixed run output, a duplicate run
ID, a duplicate output root, a non-16 command count, or test-selection leakage
causes preparation to fail before files are written. Existing Formal Long32
and repair-diagnostic groups use different experiment roots and are never
overwritten.

The generated commands are a handoff artifact only. Review them and perform
normal Git/remote validation before any later, separately authorized remote
training launch.
