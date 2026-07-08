# MultiDAG+CL formal YAML standardization notes

## 1. Purpose

This note records the standardization of formal MultiDAG+CL IEMOCAP YAMLs into
canonical nested directories. It is a configuration and documentation task
only: no model, dataset, MMGCN, or SDT code was changed.

## 2. Files added

Canonical formal training YAMLs were added under:

```text
configs/baselines/multidag_cl/iemocap/formal/
```

Canonical debug training YAMLs were added under:

```text
configs/baselines/multidag_cl/iemocap/debug/
```

Canonical formal/debug/missing pipeline YAMLs were added under:

```text
configs/pipeline/multidag_cl/iemocap/
```

Canonical multi-run analysis templates were added under:

```text
configs/analysis/multidag_cl/iemocap/
```

Run documentation was added:

```text
docs/baselines/multidag_cl_formal_run_plan.md
docs/baselines/multidag_cl_formal_yaml_standardization_notes.md
```

## 3. Files modified

No existing model, dataset, training, evaluation, or analysis script was
modified for this standardization. The old root-level YAMLs were kept in place
as legacy/compat entries.

## 4. Files kept as legacy

Legacy training YAMLs kept under `configs/baselines/multidag_cl/`:

```text
iemocap_multidag_cl_causal_w0.yaml
iemocap_multidag_cl_causal_w1.yaml
iemocap_multidag_cl_causal_w3.yaml
iemocap_multidag_cl_causal_w5.yaml
iemocap_multidag_cl_past_all_causal.yaml
iemocap_multidag_cl_causal_w5_quick.yaml
iemocap_multidag_cl_causal_smoke.yaml
iemocap_multidag_cl_full_smoke.yaml
iemocap_multidag_cl_causal_w5_text_only.yaml
iemocap_multidag_cl_causal_w5_audio_only.yaml
iemocap_multidag_cl_causal_w5_visual_only.yaml
iemocap_multidag_cl_causal_w5_text_audio.yaml
iemocap_multidag_cl_causal_w5_text_visual.yaml
iemocap_multidag_cl_causal_w5_audio_visual.yaml
```

Legacy pipeline YAMLs kept under `configs/pipeline/`:

```text
multidag_cl_iemocap_context_w0.yaml
multidag_cl_iemocap_context_w1.yaml
multidag_cl_iemocap_context_w3.yaml
multidag_cl_iemocap_context_w5.yaml
multidag_cl_iemocap_context_past_all.yaml
multidag_cl_iemocap_context_w5_quick.yaml
multidag_cl_iemocap_modality_text_only.yaml
multidag_cl_iemocap_modality_audio_only.yaml
multidag_cl_iemocap_modality_visual_only.yaml
multidag_cl_iemocap_modality_text_audio.yaml
multidag_cl_iemocap_modality_text_visual.yaml
multidag_cl_iemocap_modality_audio_visual.yaml
multidag_cl_iemocap_w5_missing_modalities.yaml
iemocap_multidag_cl_causal_smoke_pipeline.yaml
iemocap_multidag_cl_full_smoke_pipeline.yaml
```

Legacy analysis YAMLs kept under `configs/analysis/`:

```text
multidag_cl_iemocap_context_compare.yaml
multidag_cl_iemocap_modality_compare.yaml
```

The canonical formal run plan uses only the new nested paths.

## 5. Formal training YAMLs

Formal context-window training YAMLs:

```text
configs/baselines/multidag_cl/iemocap/formal/context_w0_tav.yaml
configs/baselines/multidag_cl/iemocap/formal/context_w1_tav.yaml
configs/baselines/multidag_cl/iemocap/formal/context_w3_tav.yaml
configs/baselines/multidag_cl/iemocap/formal/context_w5_tav.yaml
configs/baselines/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml
```

Formal modality-ablation training YAMLs:

```text
configs/baselines/multidag_cl/iemocap/formal/modality_w5_t.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_a.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_v.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_ta.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_tv.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_av.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_tav.yaml
```

All formal training YAMLs use `training.epochs: 30`,
`training.select_best_by: val_weighted_f1`, and no train/validation batch caps.

## 6. Formal pipeline YAMLs

Formal context-window pipeline YAMLs:

```text
configs/pipeline/multidag_cl/iemocap/formal/context_w0_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_w1_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_w3_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_w5_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml
```

Formal modality-ablation pipeline YAMLs:

```text
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_t.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_a.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_v.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_ta.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_tv.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_av.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_tav.yaml
```

All formal pipelines enable:

```text
evaluation.enabled: true
evaluation.splits: [val, test]
analysis_tables.enabled: true
single_run_analysis.training_curves: true
single_run_analysis.final_analysis: true
```

All formal pipelines set `evaluation.max_batches: null`.

## 7. Debug YAMLs

Debug training YAMLs:

```text
configs/baselines/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml
configs/baselines/multidag_cl/iemocap/debug/context_w5_tav_smoke.yaml
configs/baselines/multidag_cl/iemocap/debug/context_past_all_causal_tav_smoke.yaml
```

Debug pipeline YAMLs:

```text
configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml
configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_smoke.yaml
configs/pipeline/multidag_cl/iemocap/debug/context_past_all_causal_tav_smoke.yaml
```

Quick uses 5 epochs and no train/eval caps. Smoke uses 1 epoch, batch size 2,
and tiny caps.

## 8. Missing-modality pipeline

Canonical missing-modality pipeline:

```text
configs/pipeline/multidag_cl/iemocap/missing/missing_eval_from_context_w5_tav.yaml
```

Before running, fill:

```yaml
run_control:
  skip_train_use_run_id: "<formal_context_w5_tav_run_id>"
```

Missing-modality evaluation is test-time zeroing. It reuses the completed TAV
checkpoint and does not retrain a new model.

## 9. Multi-run analysis templates

Canonical templates:

```text
configs/analysis/multidag_cl/iemocap/context_compare.yaml
configs/analysis/multidag_cl/iemocap/modality_compare.yaml
configs/analysis/multidag_cl/iemocap/core_results_compare.yaml
```

Each template keeps `runs: []` until real run ids exist.

`core_results_compare.yaml` currently compares normal evaluation metrics only.
Missing-modality summary figures are generated by the missing-modality pipeline.

## 10. Figure-generation guarantee

Every canonical formal pipeline enables the full single-run analysis chain:

```text
analysis_tables.enabled: true
single_run_analysis.enabled: true
single_run_analysis.training_curves: true
single_run_analysis.final_analysis: true
```

The missing-modality pipeline enables:

```text
missing_modalities.enabled: true
missing_modalities.make_figures: true
```

## 11. Recommended run order

Recommended order for July 8, 2026:

1. `configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml`
2. `configs/pipeline/multidag_cl/iemocap/formal/context_w5_tav.yaml`
3. `configs/pipeline/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml`
4. `configs/pipeline/multidag_cl/iemocap/formal/context_w0_tav.yaml`
5. `configs/pipeline/multidag_cl/iemocap/missing/missing_eval_from_context_w5_tav.yaml`
6. `configs/pipeline/multidag_cl/iemocap/formal/context_w3_tav.yaml`
7. Modality-ablation pipelines after the core context/missing experiments

`context_w1_tav` can be filled later unless a denser context curve is needed
immediately.

## 12. Checks performed

YAML validation passed for:

```text
12 formal training YAMLs
3 debug training YAMLs
12 formal pipeline YAMLs
3 debug pipeline YAMLs
3 analysis templates
```

The validation checked that formal training YAMLs use the expected graph
settings, active modalities, 30 epochs, validation Weighted-F1 checkpoint
selection, and no train/validation caps. It also checked that formal pipelines
reference existing training YAMLs, evaluate `val` and `test`, enable
analysis-table and single-run figure stages, and have no formal `full` naming
or `context_mode: full` usage.

Syntax check passed:

```bash
PYTHONPYCACHEPREFIX=tmp/py_compile_cache python -m py_compile \
  scripts/run_experiment_pipeline.py \
  scripts/baselines/train_multidag_cl.py \
  scripts/baselines/evaluate_multidag_cl_checkpoint.py \
  scripts/experiments/evaluate_missing_modalities.py \
  scripts/analyze/plot_missing_modality_summary.py
```

The first Windows sandbox attempt hit the known `.pyc` atomic-rename permission
issue. The same command passed after rerunning with the required sandbox
escalation.

## 13. Remaining limitations

The new configs were not used to run formal training locally. Formal 30-epoch
training should run on the remote V100/data machine after review, commit, and
push.

The repository still retains old root-level compatibility YAMLs. Use the new
nested paths for formal run commands.

Direct answers:

1. Formal context-window YAMLs are the five `context_*_tav.yaml` files under the new formal training and pipeline directories.
2. Formal modality-ablation YAMLs are the seven `modality_w5_*.yaml` files under the new formal training and pipeline directories.
3. Quick/smoke YAMLs are the three `context_*_quick/smoke.yaml` files under the new debug directories.
4. The missing-modality pipeline is `configs/pipeline/multidag_cl/iemocap/missing/missing_eval_from_context_w5_tav.yaml`.
5. All canonical formal pipelines enable val/test evaluation.
6. All canonical formal pipelines enable `analysis_tables`.
7. All canonical formal pipelines enable `training_curves` and `final_analysis`.
8. No canonical formal file uses `full` naming.
9. The old `full` smoke meaning is standardized as `past_all_causal` in canonical debug/formal naming.
10. No model code was modified.
11. MMGCN, SDT, and datasets were not modified.
12. Next formal experiments are listed in the recommended run order above.
