# MultiDAG+CL pipeline integration notes

## 1. Purpose

This note records the engineering integration that lets the project-side
`MultiDAGCLBaseline` use a MMGCN-like train -> checkpoint -> evaluate ->
analysis -> pipeline workflow.

This task does not run a formal IEMOCAP experiment matrix, does not implement
curriculum learning, does not add contrastive learning, and does not add
modality dropout.

## 2. Files added

- `scripts/baselines/train_multidag_cl.py`
- `scripts/baselines/evaluate_multidag_cl_checkpoint.py`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_smoke.yaml`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_full_smoke.yaml`
- `configs/pipeline/iemocap_multidag_cl_causal_smoke_pipeline.yaml`
- `configs/pipeline/iemocap_multidag_cl_full_smoke_pipeline.yaml`
- `docs/baselines/multidag_cl_pipeline_integration_notes.md`

## 3. Files modified

- `scripts/run_experiment_pipeline.py`

No changes were made to:

- `scripts/train_mmgcn.py`
- `scripts/evaluate_checkpoint.py`
- `models/baselines/mmgcn/`
- `models/baselines/sdt/`
- `datasets/`

The existing local modifications in `scripts/train_mmgcn.py` and
`scripts/evaluate_checkpoint.py` are pre-existing MMGCN smoke-support diffs and
were not touched in this task.

## 4. Training entry

New command:

```bash
python scripts/baselines/train_multidag_cl.py --config configs/baselines/multidag_cl/iemocap_multidag_cl_causal_smoke.yaml
```

The entry supports IEMOCAP official MMGCN-style feature pkl batches and builds
train/val dataloaders through the existing IEMOCAP adapter. It selects
`best_model.pt` only by validation Weighted-F1.

Supported tiny-run controls:

- `training.epochs`
- `training.batch_size`
- `training.lr`
- `training.weight_decay`
- `training.grad_clip`
- `training.max_train_batches`
- `training.max_val_batches`
- `training.select_best_by: val_weighted_f1`

## 5. Evaluation entry

New command:

```bash
python scripts/baselines/evaluate_multidag_cl_checkpoint.py \
  --checkpoint outputs/runs/<run_id>/checkpoints/best_model.pt \
  --split val
```

The evaluator reloads the checkpoint config, rebuilds `MultiDAGCLBaseline`,
loads `model_state_dict`, evaluates the requested split, and writes:

```text
logs/evaluations/<split>_best_model/
  metrics.csv
  predictions.csv
  confusion_matrix.csv
  per_class_recall.csv
```

`--split test` is supported only for reporting and is not used for checkpoint
selection.

## 6. Output directory compatibility

The training entry writes the MMGCN-compatible run layout:

```text
outputs/runs/<run_id>/
  checkpoints/
    best_model.pt
    last_model.pt
  logs/
    experiment_config.yaml
    epoch_metrics.csv
    val_predictions_best.csv
    confusion_matrix_best.csv
    per_class_recall_best.csv
    evaluations/
      val_best_model/
        metrics.csv
        predictions.csv
        confusion_matrix.csv
        per_class_recall.csv
  figures/
```

It also updates:

```text
outputs/latest_run.txt
```

with project-relative paths.

## 7. Analysis script compatibility

No analysis script changes were required. The existing scripts are already
CSV/run-directory driven, so the MultiDAG+CL train/eval entries align to their
expected files:

- `scripts/analyze/build_analysis_tables.py`
- `scripts/analyze/plot_single_run_training_curves.py`
- `scripts/analyze/plot_single_run_final_analysis.py`
- `scripts/analyze/plot_multi_run_training_curves.py`
- `scripts/analyze/plot_multi_run_final_analysis.py`

Multi-run analysis still requires a multi-run YAML whose `runs` list contains
the desired completed run IDs, matching the existing MMGCN workflow.

## 8. Pipeline YAMLs

Added:

- `configs/pipeline/iemocap_multidag_cl_causal_smoke_pipeline.yaml`
- `configs/pipeline/iemocap_multidag_cl_full_smoke_pipeline.yaml`

`scripts/run_experiment_pipeline.py` now registers:

- `MultiDAGCL` training -> `scripts/baselines/train_multidag_cl.py`
- `MultiDAGCL` evaluation -> `scripts/baselines/evaluate_multidag_cl_checkpoint.py`

The pipeline smoke YAMLs enable train, val/test checkpoint evaluation, analysis
tables, single-run training curves, and single-run final analysis. Evaluation is
capped by `evaluation.max_batches: 2` to keep the smoke workflow lightweight.

## 9. Smoke run results

Syntax check passed:

```bash
python -m py_compile scripts/baselines/train_multidag_cl.py scripts/baselines/evaluate_multidag_cl_checkpoint.py scripts/run_experiment_pipeline.py scripts/analyze/build_analysis_tables.py scripts/analyze/plot_single_run_training_curves.py scripts/analyze/plot_single_run_final_analysis.py scripts/analyze/plot_multi_run_training_curves.py scripts/analyze/plot_multi_run_final_analysis.py
```

The command was run with `PYTHONPYCACHEPREFIX=tmp/py_compile_cache` and needed
the managed Windows sandbox escalation because direct `.pyc` atomic rename was
denied.

IEMOCAP smoke training was not run locally because the feature file is missing:

```text
third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl
```

Therefore no new run directory, `epoch_metrics.csv`, or `metrics.csv` was
created in this task.

## 10. Known limitations

`graph.context_mode: causal` uses directed past context with the configured
`window_past`.

`graph.context_mode: full` is currently implemented as `past_all_causal`: a
large effective `window_past` lets each utterance read all previous utterances
and itself, but it does not read future utterances. This is not true offline
bidirectional/full context.

Formal IEMOCAP runs should be executed on the remote/data machine after this
code is committed and pushed.

## 11. Recommendation for next task

Next task: add the IEMOCAP causal/full/missing-modality experiment YAML matrix
after running the causal and full smoke pipelines on a machine with the official
IEMOCAP feature pkl.

Required answers:

1. MultiDAG+CL now has a formal train entry: yes,
   `scripts/baselines/train_multidag_cl.py`.
2. MultiDAG+CL now has a checkpoint evaluation entry: yes,
   `scripts/baselines/evaluate_multidag_cl_checkpoint.py`.
3. It can save a validation-best checkpoint: yes, selected by
   `val_weighted_f1`.
4. It can reload/evaluate checkpoint: yes, through the new evaluation entry;
   local execution was not run because IEMOCAP data is missing.
5. The output directory is aligned to MMGCN: yes, under
   `outputs/runs/<run_id>/checkpoints`, `logs`, and `figures`.
6. Analysis scripts can read MultiDAG+CL runs: yes, no analysis script changes
   were required.
7. Pipeline YAML can call the workflow: yes, through the two new pipeline YAMLs.
8. `scripts/train_mmgcn.py` / `scripts/evaluate_checkpoint.py` were modified:
   no, not in this task.
9. The full config is not true offline full; it is `past_all_causal`.
10. The project can proceed to the IEMOCAP causal/full/missing-modality YAML
    matrix task after smoke pipeline execution on a machine with the IEMOCAP
    feature pkl.
