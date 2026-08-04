# MultiDAG-CL paper reimplementation runtime

This runtime integrates the independent `paper_reimplementation` lineage with the
project's canonical IEMOCAP feature registry, session-holdout protocol, output
resolver, shared metrics, and dialogue batch contract. It is not an
`author_official` implementation and does not reproduce paper numbers on the
`project_fair` data track.

## Identity and context boundary

- Registry key: `multidag_cl_paper_reimplementation`.
- Primary profile: `paper_formula_behavior`.
- Data track: `project_fair` with `clean_roberta_v1`.
- Context label: `full_context`.
- DAG topology: causal and past-only.
- End-to-end path: noncausal because the frozen dialogue-axis text BiLSTM is
  bidirectional.
- Paper data track: blocked until the separately audited official assets are
  available.

The Stage-B2 model and curriculum math remain unchanged. The runtime executes the
Stage-B1 frozen `gradient_clip_norm=5.0` protocol immediately after backward and
immediately before every optimizer step. The resolved config and run manifest
record explicit global-norm clipping with norm type 2.0 and fail-closed non-finite
handling. Zero-step probes perform backward but do not clip.

## Configurations and modes

The canonical config family is
`configs/multidag_cl/paper_reimplementation/iemocap/full_context/clean_roberta_features/`:

- `project_fair.yaml`: future 30-epoch remote Project-Fair run; not executed in
  Stage B3.
- `synthetic_smoke.yaml`: CPU-only, one epoch, one batch, exactly one optimizer
  step, one validation pass, locked strict reload, and one fake-Test timing check.
- `real_batch_smoke.yaml`: CPU-only, one train forward/backward, zero optimizer
  steps, one validation batch, and zero real-Test access. It reports
  `BLOCKED_LOCAL_ASSET` when the pinned clean feature file is absent.

The thin entrypoint is
`scripts/models/multidag_cl/paper_reimplementation/train.py`. Its modes are
`check`, `synthetic-smoke`, `real-batch-smoke`, `train`, and `evaluate`.
`evaluate` requires a locked checkpoint. `check` validates the registry, feature
metadata, split, config, model, and optimizer without allocating a run or reading
the full dataset.

Example checks:

```powershell
conda run -n m3ed_mmgcn python scripts/models/multidag_cl/paper_reimplementation/train.py --mode check --config configs/multidag_cl/paper_reimplementation/iemocap/full_context/clean_roberta_features/project_fair.yaml --device cpu
```

```powershell
conda run -n m3ed_mmgcn python scripts/models/multidag_cl/paper_reimplementation/train.py --mode synthetic-smoke --config configs/multidag_cl/paper_reimplementation/iemocap/full_context/clean_roberta_features/synthetic_smoke.yaml --device cpu
```

## Selection and output contract

Curriculum membership is computed once from train gold labels/speakers, exported
to `curriculum_bucket_manifest.tsv`, and made visible from the one-based global
epoch. Validation and Test never enter difficulty or bucket membership.

Checkpoint selection uses unrounded Validation Weighted-F1, then lower Validation
loss, then earlier epoch. The best checkpoint is locked and strictly reloaded
before the Test gate opens. Test metrics have no selector API and are recorded with
`test_split_used_for_selection=false`.

Formal runs resolve to `outputs/<date>/<experiment_group>/runs/<run_id>/`. Stage-B3
smoke configs keep the repository-wide `output.root: outputs` template, but the
runtime requires and automatically applies `runtime.smoke_output_root`. Actual
smoke writes are restricted to
`tmp/assistant_work/paper_reproduction_stage_b3_runtime_correction/smoke_outputs/`
while retaining the same date/group/run hierarchy. Every run owns checkpoints,
logs, reports, predictions, manifests, and a resolved config. Fixed-path overwrite
is rejected by the common atomic allocator.

Training summaries distinguish `epoch_count`, `train_batch_count`, and
`optimizer_step_count`; the deprecated `train_batches` and `optimizer_steps`
aliases retain the corresponding real counts. Best-checkpoint locking preserves
the selected epoch's own global step. Resume validates the immutable resolved
config, identity, feature/split contract, curriculum membership, and checkpoint
hash before appending one atomic `resume_history` record. Evaluation-only mode
requires a locked checkpoint and reports that neither selection nor the training
manifest was mutated.
