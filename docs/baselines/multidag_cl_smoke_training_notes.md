# MultiDAG+CL smoke training notes

This is smoke training only. It is not a formal M3ED/IEMOCAP experiment.

## 1. Purpose

Verify that the project-side MultiDAG+CL port can run a tiny train -> validation
checkpoint selection -> checkpoint reload/evaluate loop on deterministic fake
dialogue data.

## 2. Files added

- `configs/smoke/train_multidag_cl_smoke.yaml`
- `scripts/baselines/train_multidag_cl_smoke.py`
- `docs/baselines/multidag_cl_smoke_training_notes.md`

## 3. Smoke config

The smoke config uses `MMGCN_SMOKE` fake dialogue batches with small feature
dimensions:

- text: 16
- audio: 12
- visual: 8
- hidden: 32
- classes: 4
- speakers: 2
- epochs: 2
- batch size: 2
- output root: `outputs/smoke/multidag_cl`

## 4. Training/evaluation protocol

The script builds train and validation dataloaders only. It uses the model's
masked cross-entropy loss and reports validation Accuracy, Weighted-F1,
Macro-F1, and UAR after each epoch.

## 5. Checkpoint selection

`best_model.pt` is selected by validation Weighted-F1 through
`training.select_best_by: val_weighted_f1`. The test split is not used for
checkpoint selection.

## 6. Reload evaluation

After training, the script reloads `best_model.pt` into a fresh model instance
and evaluates it once on the validation split. Results are saved to
`smoke_eval_reload_metrics.csv`.

## 7. Smoke run result

Local smoke run completed on 2026-07-07:

```text
command: conda run -n m3ed_mmgcn python scripts/baselines/train_multidag_cl_smoke.py --config configs/smoke/train_multidag_cl_smoke.yaml
output: outputs/smoke/multidag_cl/20260707_141430_multidag_cl_smoke
best checkpoint: best_model.pt
best epoch: 1
selection metric: val_weighted_f1
```

Validation metrics for the saved best checkpoint:

```text
val_acc=0.7143
val_weighted_f1=0.6429
val_macro_f1=0.6250
val_uar=0.7500
```

Reload evaluation from `best_model.pt` completed on the validation split and
matched the epoch-1 validation metrics. The test split was not used.

## 8. Remaining work before real M3ED/IEMOCAP training

1. Add a real-data adapter that maps current M3ED/IEMOCAP batches into the
   MultiDAG+CL smoke entry without changing existing dataset contracts.
2. Run a local real-data dry run on a tiny subset or one batch.
3. Define a remote V100 full-training protocol with validation-best checkpoint
   selection and the required metrics.
4. Revisit the exact official MultiDAG+CL predecessor rule before formal
   comparison.

## 9. Recommendation for next task

Next task: MultiDAG+CL real-data adapter or MultiDAG+CL M3ED local dry run.
