# MultiDAG+CL formal IEMOCAP run plan

## 1. Purpose

This document is the canonical run plan for the formal MultiDAG+CL IEMOCAP
configuration set. It standardizes config names, separates formal/debug/missing
pipelines, and makes the single-run analysis chain part of every formal
train/evaluate pipeline.

These runs are for the remote/data machine. Do not run the formal 30-epoch
matrix on local Windows.

## 2. Naming convention

Canonical training YAMLs live under:

```text
configs/baselines/multidag_cl/iemocap/formal/
configs/baselines/multidag_cl/iemocap/debug/
```

Canonical pipeline YAMLs live under:

```text
configs/pipeline/multidag_cl/iemocap/formal/
configs/pipeline/multidag_cl/iemocap/debug/
configs/pipeline/multidag_cl/iemocap/missing/
```

Canonical analysis templates live under:

```text
configs/analysis/multidag_cl/iemocap/
```

Use `past_all_causal` for the all-past causal setting. Do not use `full` for
formal files. Any old root-level `full` smoke YAML is legacy only and is not
evidence of true offline bidirectional/full-context performance.

## 3. Formal configs to run

Context-window formal pipelines:

```text
configs/pipeline/multidag_cl/iemocap/formal/context_w0_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_w1_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_w3_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_w5_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml
```

Modality-ablation formal pipelines:

```text
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_t.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_a.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_v.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_ta.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_tv.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_av.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_tav.yaml
```

Every formal pipeline trains, evaluates `best_model.pt` on `val` and `test`,
builds analysis tables, and generates single-run training-curve and final
analysis figures.

## 4. Debug configs

Debug pipelines:

```text
configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml
configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_smoke.yaml
configs/pipeline/multidag_cl/iemocap/debug/context_past_all_causal_tav_smoke.yaml
```

The quick config uses 5 epochs without train/eval caps. Smoke configs use
1 epoch, batch size 2, and tiny train/eval caps. Smoke metrics are never formal
results.

## 5. Recommended remote command style

Run from the project root on the remote V100/data machine:

```bash
cd /home/zhiyuan/research/m3ed_mmgcn_clean

conda activate m3ed_mmgcn

PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/context_w5_tav.yaml
```

Use this activated-shell style for remote formal work. Do not use `conda run`
for the formal remote commands in this plan.

## 6. Today's priority order

Recommended order for July 8, 2026:

1. `configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml`
2. `configs/pipeline/multidag_cl/iemocap/formal/context_w5_tav.yaml`
3. `configs/pipeline/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml`
4. `configs/pipeline/multidag_cl/iemocap/formal/context_w0_tav.yaml`
5. `configs/pipeline/multidag_cl/iemocap/missing/missing_eval_from_context_w5_tav.yaml` after the `context_w5_tav` run_id is available
6. If time remains: `configs/pipeline/multidag_cl/iemocap/formal/context_w3_tav.yaml`
7. Queue modality-ablation runs after core context/missing experiments

`context_w1_tav` can be deferred unless you need a denser context-window curve
today. The first priority is to bracket current-only, standard w5, and
past-all-causal behavior, then fill the intermediate windows.

## 7. Context-window commands

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml
```

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/context_w5_tav.yaml
```

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml
```

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/context_w0_tav.yaml
```

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/context_w3_tav.yaml
```

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/context_w1_tav.yaml
```

## 8. Modality-ablation commands

These are from-scratch modality-ablation training runs, not test-time missing
modality evaluation.

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/modality_w5_tav.yaml
```

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/modality_w5_t.yaml
```

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/modality_w5_a.yaml
```

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/modality_w5_v.yaml
```

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/modality_w5_ta.yaml
```

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/modality_w5_tv.yaml
```

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/formal/modality_w5_av.yaml
```

## 9. Missing-modality command

Before running, fill this field with the completed formal `context_w5_tav`
run id:

```yaml
run_control:
  skip_train_use_run_id: "<context_w5_tav_run_id>"
```

Then run:

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiment_pipeline.py \
  --config configs/pipeline/multidag_cl/iemocap/missing/missing_eval_from_context_w5_tav.yaml
```

This is test-time zeroing of unavailable modalities. It does not retrain the
checkpoint.

## 10. Expected outputs for each single run

Each train/eval pipeline should produce:

```text
outputs/runs/<run_id>/
  checkpoints/best_model.pt
  checkpoints/last_model.pt
  logs/experiment_config.yaml
  logs/epoch_metrics.csv
  logs/evaluations/val_best_model/metrics.csv
  logs/evaluations/test_best_model/metrics.csv
  figures/training_curves/
  figures/final_analysis/
```

The missing-modality pipeline should add:

```text
outputs/runs/<context_w5_tav_run_id>/logs/missing_modalities/test_best_model/summary.csv
outputs/runs/<context_w5_tav_run_id>/logs/missing_modalities/test_best_model/metadata.json
outputs/runs/<context_w5_tav_run_id>/logs/missing_modalities/test_best_model/raw/<setting>/
outputs/runs/<context_w5_tav_run_id>/figures/missing_modalities/test_best_model/
```

## 11. How to fill multi-run analysis YAML after runs finish

Fill these templates after the formal run ids exist:

```text
configs/analysis/multidag_cl/iemocap/context_compare.yaml
configs/analysis/multidag_cl/iemocap/modality_compare.yaml
configs/analysis/multidag_cl/iemocap/core_results_compare.yaml
```

Example:

```yaml
runs:
  - run_id: 20260708_000000_iemocap_multidag_cl_context_w5_tav
    display_name: context_w5_tav
```

After filling the templates, rebuild analysis tables and run the multi-run
plotting scripts through a pipeline with the multi-run stage enabled, or call
the plotting scripts directly.

`core_results_compare.yaml` currently compares normal evaluation metrics only.
Missing-modality summary plots are produced by the missing-modality pipeline.

## 12. What not to claim from these experiments

Do not claim that `past_all_causal` is true offline bidirectional full context.
It sees the current utterance and all previous valid utterances, not future
utterances.

Do not use smoke metrics as formal conclusions.

Do not interpret test-time zeroing as from-scratch modality contribution.
Use the `modality_w5_*` formal runs for from-scratch modality ablation.

Do not use the test split to select `best_model.pt`. The training config
selects the checkpoint by validation Weighted-F1.
