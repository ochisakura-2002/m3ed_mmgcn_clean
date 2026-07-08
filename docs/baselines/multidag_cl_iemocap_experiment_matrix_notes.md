# MultiDAG+CL IEMOCAP experiment matrix notes

## 1. Purpose

This note records the IEMOCAP MultiDAG+CL experiment matrix for formal remote
runs. It prepares context-window, modality-ablation, and test-time
missing-modality workflows without changing the MultiDAGCL model structure,
without adding curriculum learning, and without adding modality-dropout
training.

The formal configs keep `training.epochs: 30`. A separate `causal_w5_quick`
5-epoch config is available for a short remote sanity pass before the formal
run.

## 2. Experiment groups

Context-window experiments answer how much causal history helps in an online
dialogue setting. They all train with text+audio+visual.

Modality-ablation experiments answer how from-scratch training behaves when
only a subset of modalities is available. These are separate checkpoints and
must not be mixed with test-time missing-modality robustness.

Missing-modality evaluation answers how a trained T+A+V checkpoint behaves when
some modalities are zeroed during evaluation.

## 3. Context-window configs

Training configs:

- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w0.yaml`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w1.yaml`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w3.yaml`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5.yaml`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_past_all_causal.yaml`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_quick.yaml`

Pipeline configs:

- `configs/pipeline/multidag_cl_iemocap_context_w0.yaml`
- `configs/pipeline/multidag_cl_iemocap_context_w1.yaml`
- `configs/pipeline/multidag_cl_iemocap_context_w3.yaml`
- `configs/pipeline/multidag_cl_iemocap_context_w5.yaml`
- `configs/pipeline/multidag_cl_iemocap_context_past_all.yaml`
- `configs/pipeline/multidag_cl_iemocap_context_w5_quick.yaml`

Window semantics:

- `w0`: current utterance only.
- `w1`: current utterance plus previous 1 valid utterance.
- `w3`: current utterance plus previous 3 valid utterances.
- `w5`: current utterance plus previous 5 valid utterances.
- `past_all_causal`: current utterance plus all previous valid utterances.

## 4. Modality-ablation configs

All modality-ablation configs use causal `window_past: 5` and train a fresh
checkpoint under the listed `model.active_modalities`.

Training configs:

- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_text_only.yaml`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_audio_only.yaml`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_visual_only.yaml`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_text_audio.yaml`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_text_visual.yaml`
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_audio_visual.yaml`

Pipeline configs:

- `configs/pipeline/multidag_cl_iemocap_modality_text_only.yaml`
- `configs/pipeline/multidag_cl_iemocap_modality_audio_only.yaml`
- `configs/pipeline/multidag_cl_iemocap_modality_visual_only.yaml`
- `configs/pipeline/multidag_cl_iemocap_modality_text_audio.yaml`
- `configs/pipeline/multidag_cl_iemocap_modality_text_visual.yaml`
- `configs/pipeline/multidag_cl_iemocap_modality_audio_visual.yaml`

## 5. Missing-modality evaluation

Pipeline config:

- `configs/pipeline/multidag_cl_iemocap_w5_missing_modalities.yaml`

Before running it, fill:

```yaml
run_control:
  skip_train_use_run_id: "<trained causal_w5 TAV run_id>"
```

Supported settings:

- `TAV`: text+audio+visual.
- `TA`: text+audio, so visual is missing.
- `TV`: text+visual, so audio is missing.
- `AV`: audio+visual, so text is missing.
- `T`: text only.
- `A`: audio only.
- `V`: visual only.
- `missing_text`: alias for `AV`.
- `missing_audio`: alias for `TV`.
- `missing_visual`: alias for `TA`.

For MultiDAG+CL checkpoints, missing modalities are implemented by zeroing the
corresponding feature tensors during evaluation. The checkpoint model structure
and saved `model.active_modalities` are not changed.

The missing-modality pipeline keeps
`missing_modalities.skip_if_not_full_train_modalities: true`, so it skips
checkpoints that were not trained with text+audio+visual.

## 6. Full vs past-all causal semantics

`past_all_causal` is not true offline bidirectional full context. It allows each
utterance to read itself and all valid previous utterances, but never future
utterances.

The older smoke config name `full` maps to this same past-all causal behavior.
Do not interpret any `full` smoke result as offline bidirectional full-context
performance.

## 7. How to run context-window experiments

Run from the project root on the remote/data machine:

```bash
python scripts/run_experiment_pipeline.py --config configs/pipeline/multidag_cl_iemocap_context_w5.yaml
```

Other context configs can be run by replacing the YAML path with:

```text
configs/pipeline/multidag_cl_iemocap_context_w0.yaml
configs/pipeline/multidag_cl_iemocap_context_w1.yaml
configs/pipeline/multidag_cl_iemocap_context_w3.yaml
configs/pipeline/multidag_cl_iemocap_context_past_all.yaml
configs/pipeline/multidag_cl_iemocap_context_w5_quick.yaml
```

The pipeline script uses the YAML field `execution.dry_run`; it does not expose
a separate `--dry-run` CLI option.

## 8. How to run modality-ablation experiments

Run one modality pipeline at a time from the project root:

```bash
python scripts/run_experiment_pipeline.py --config configs/pipeline/multidag_cl_iemocap_modality_text_only.yaml
```

Then repeat for:

```text
configs/pipeline/multidag_cl_iemocap_modality_audio_only.yaml
configs/pipeline/multidag_cl_iemocap_modality_visual_only.yaml
configs/pipeline/multidag_cl_iemocap_modality_text_audio.yaml
configs/pipeline/multidag_cl_iemocap_modality_text_visual.yaml
configs/pipeline/multidag_cl_iemocap_modality_audio_visual.yaml
```

These are from-scratch training runs. Do not compare them as if they were
test-time missing-modality settings from the same checkpoint.

## 9. How to run missing-modality evaluation after causal_w5 training

First run the full T+A+V causal w5 pipeline:

```bash
python scripts/run_experiment_pipeline.py --config configs/pipeline/multidag_cl_iemocap_context_w5.yaml
```

After it finishes, copy its `run_id` into:

```text
configs/pipeline/multidag_cl_iemocap_w5_missing_modalities.yaml
```

Then run:

```bash
python scripts/run_experiment_pipeline.py --config configs/pipeline/multidag_cl_iemocap_w5_missing_modalities.yaml
```

Expected missing-modality files:

```text
outputs/runs/<run_id>/logs/missing_modalities/test_best_model/summary.csv
outputs/runs/<run_id>/logs/missing_modalities/test_best_model/metadata.json
outputs/runs/<run_id>/logs/missing_modalities/test_best_model/raw/<setting>/
outputs/runs/<run_id>/figures/missing_modalities/test_best_model/
```

## 10. Expected outputs

Each train/eval pipeline writes the standard run layout:

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

The multi-run comparison templates are:

- `configs/analysis/multidag_cl_iemocap_context_compare.yaml`
- `configs/analysis/multidag_cl_iemocap_modality_compare.yaml`

They intentionally keep `runs: []` until formal run IDs are available.

## 11. What should not be interpreted from smoke runs

Smoke runs only confirm that code paths, checkpoint loading, evaluation, and
analysis output creation are wired correctly.

Do not use smoke metrics to claim model quality, superiority over another
baseline, robustness, modality contribution, or multi-seed stability.

## 12. Recommendation for today's run order

1. Run causal_w5 first.
2. Run past_all_causal second.
3. Run causal_w0 or causal_w3 third, depending on time.
4. Run missing-modality evaluation on the trained causal_w5 TAV run.
5. Only run single-modality training configs if time remains.
