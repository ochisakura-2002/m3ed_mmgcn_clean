# MultiDAG+CL observability and figure-generation notes

## 1. Purpose

This note records the engineering changes and audit for MultiDAG+CL training
observability and figure generation. This task did not change model structure,
datasets, MMGCN, or SDT, and it did not run a formal 30-epoch experiment.

The goal was to make long training runs easier to monitor and to clarify which
commands generate figures.

## 2. Files modified

Modified:

- `scripts/baselines/train_multidag_cl.py`

Added:

- `docs/baselines/multidag_cl_yaml_parameter_guide.md`
- `docs/baselines/multidag_cl_observability_and_figures_notes.md`

No changes were made to:

- `models/`
- `models/baselines/mmgcn/`
- `models/baselines/sdt/`
- `datasets/`
- MultiDAG+CL pipeline YAMLs

The pipeline YAMLs were audited and already had the required figure-generation
switches enabled where expected.

## 3. Training progress-bar changes

`scripts/baselines/train_multidag_cl.py` now shows a live train progress bar for
each epoch with:

- epoch number and total epochs in the description
- batch index and total batch count from tqdm
- current batch loss as `loss`
- running average train loss as `avg`
- current optimizer learning rate as `lr`
- valid utterance count as `valid`

The train progress bar now uses the cap from `training.max_train_batches` when
that field is set. For example, if the dataloader has 24 batches and
`max_train_batches: 2`, tqdm is built with total `2` and the iterator is limited
to 2 batches.

Validation inside each epoch now shows a live progress bar with:

- validation batch index and total batch count from tqdm
- current validation batch loss as `loss`
- running average validation loss as `avg`
- valid utterance count as `valid`

The validation progress bar also respects `training.max_val_batches`.

## 4. Epoch metric flushing

`logs/epoch_metrics.csv` is now overwritten after every completed epoch, not
only after all epochs finish.

The CSV still contains the existing required fields:

- `epoch`
- `train_loss`
- `val_loss`
- `val_acc`
- `val_weighted_f1`
- `val_macro_f1`
- `val_uar`

Checkpoint selection is unchanged. `best_model.pt` is still selected only by
validation Weighted-F1 through `training.select_best_by: val_weighted_f1`.
The test split is not used for checkpoint selection.

Each epoch now prints an immediate flushed summary:

```text
Epoch 03/30 | train_loss=... | val_loss=... | val_acc=... | val_weighted_f1=... | val_macro_f1=... | val_uar=...
```

When a new best checkpoint is saved, the script prints an immediate flushed
line:

```text
[BEST] epoch=03 val_weighted_f1=...
```

## 5. Figure-generation pipeline audit

Running only `scripts/baselines/train_multidag_cl.py` does not generate figures.
It writes the run directory, checkpoints, validation-best logs, and
`logs/epoch_metrics.csv`.

Figure generation is handled by `scripts/run_experiment_pipeline.py` analysis
stages:

- `analysis_tables.enabled: true` calls
  `scripts/analyze/build_analysis_tables.py` and writes master CSVs under
  `outputs/analysis_tables/`.
- `single_run_analysis.enabled: true` plus
  `single_run_analysis.training_curves: true` calls
  `scripts/analyze/plot_single_run_training_curves.py` and writes
  `figures/training_curves/`.
- `single_run_analysis.enabled: true` plus
  `single_run_analysis.final_analysis: true` calls
  `scripts/analyze/plot_single_run_final_analysis.py` and writes
  `figures/final_analysis/`.
- `missing_modalities.enabled: true` plus
  `missing_modalities.make_figures: true` calls
  `scripts/analyze/plot_missing_modality_summary.py` and writes
  `figures/missing_modalities/<stage>/`.

Audited context-window pipeline YAMLs:

| Pipeline YAML | Analysis tables | Training curves | Final analysis |
|---|---:|---:|---:|
| `configs/pipeline/multidag_cl_iemocap_context_w0.yaml` | yes | yes | yes |
| `configs/pipeline/multidag_cl_iemocap_context_w1.yaml` | yes | yes | yes |
| `configs/pipeline/multidag_cl_iemocap_context_w3.yaml` | yes | yes | yes |
| `configs/pipeline/multidag_cl_iemocap_context_w5.yaml` | yes | yes | yes |
| `configs/pipeline/multidag_cl_iemocap_context_past_all.yaml` | yes | yes | yes |
| `configs/pipeline/multidag_cl_iemocap_context_w5_quick.yaml` | yes | yes | yes |

Audited missing-modality pipeline:

| Pipeline YAML | `missing_modalities.enabled` | `missing_modalities.make_figures` |
|---|---:|---:|
| `configs/pipeline/multidag_cl_iemocap_w5_missing_modalities.yaml` | yes | yes |

No YAML fix was required.

## 6. Which commands generate figures

These commands generate figures if the required data/checkpoint files exist and
the pipeline stages complete:

```bash
python scripts/run_experiment_pipeline.py --config configs/pipeline/multidag_cl_iemocap_context_w5.yaml
```

The same is true for:

```text
configs/pipeline/multidag_cl_iemocap_context_w0.yaml
configs/pipeline/multidag_cl_iemocap_context_w1.yaml
configs/pipeline/multidag_cl_iemocap_context_w3.yaml
configs/pipeline/multidag_cl_iemocap_context_past_all.yaml
configs/pipeline/multidag_cl_iemocap_context_w5_quick.yaml
```

Expected single-run figure outputs:

```text
outputs/runs/<run_id>/figures/training_curves/
outputs/runs/<run_id>/figures/final_analysis/
```

Expected table outputs:

```text
outputs/analysis_tables/run_file_status.csv
outputs/analysis_tables/run_summary_master.csv
outputs/analysis_tables/epoch_metrics_master.csv
outputs/analysis_tables/evaluation_master.csv
outputs/analysis_tables/per_class_master.csv
```

After filling `run_control.skip_train_use_run_id` with a trained causal_w5 TAV
run id, this command generates missing-modality figures:

```bash
python scripts/run_experiment_pipeline.py --config configs/pipeline/multidag_cl_iemocap_w5_missing_modalities.yaml
```

Expected missing-modality outputs:

```text
outputs/runs/<run_id>/logs/missing_modalities/test_best_model/summary.csv
outputs/runs/<run_id>/logs/missing_modalities/test_best_model/metadata.json
outputs/runs/<run_id>/logs/missing_modalities/test_best_model/raw/<setting>/
outputs/runs/<run_id>/figures/missing_modalities/test_best_model/
```

## 7. Which commands do not generate figures

This command does not generate figures by itself:

```bash
python scripts/baselines/train_multidag_cl.py --config configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5.yaml
```

It generates training artifacts such as:

```text
outputs/runs/<run_id>/logs/epoch_metrics.csv
outputs/runs/<run_id>/checkpoints/best_model.pt
outputs/runs/<run_id>/checkpoints/last_model.pt
```

This command also does not generate figures by itself:

```bash
python scripts/baselines/evaluate_multidag_cl_checkpoint.py --checkpoint outputs/runs/<run_id>/checkpoints/best_model.pt --split test
```

It writes evaluation CSVs under:

```text
outputs/runs/<run_id>/logs/evaluations/test_best_model/
```

## 8. YAML parameter guide

A new student-facing guide was added:

```text
docs/baselines/multidag_cl_yaml_parameter_guide.md
```

It explains:

- baseline training YAML fields
- pipeline YAML fields
- missing-modality pipeline fields
- multi-run analysis YAML fields
- why only the pipeline generates figures
- why `execution.dry_run` is a YAML field, not a CLI flag
- why `evaluation.max_batches` is smoke-only
- why missing-modality evaluation is test-time zeroing, not retraining

## 9. Checks performed

Read and audited:

- `scripts/baselines/train_multidag_cl.py`
- `scripts/baselines/evaluate_multidag_cl_checkpoint.py`
- `scripts/run_experiment_pipeline.py`
- `scripts/analyze/plot_single_run_training_curves.py`
- `scripts/analyze/plot_single_run_final_analysis.py`
- `scripts/analyze/build_analysis_tables.py`
- `scripts/experiments/evaluate_missing_modalities.py`
- `scripts/analyze/plot_missing_modality_summary.py`
- MultiDAG+CL context, quick, missing-modality, and analysis YAMLs

Syntax check passed:

```bash
python -m py_compile scripts/baselines/train_multidag_cl.py scripts/baselines/evaluate_multidag_cl_checkpoint.py scripts/run_experiment_pipeline.py scripts/experiments/evaluate_missing_modalities.py scripts/analyze/plot_missing_modality_summary.py
```

The first sandboxed Windows run hit a `.pyc` atomic rename permission error
under `tmp/py_compile_cache`. The same command passed after running with the
required sandbox escalation.

No formal training, quick pipeline, or smoke pipeline was run in this task.

## 10. Remaining limitations

The improved progress bars were syntax-checked but not exercised against the
real IEMOCAP feature pkl in this local task.

Figure generation was audited from code and YAML. It was not rerun locally here
because that would require existing run outputs or the official feature pkl.

`graph.context_mode: full` in old smoke YAMLs still maps to past-all causal
semantics, not true offline bidirectional full context.

Multi-run analysis YAMLs intentionally keep `runs: []` until formal run ids are
available.

## 11. Recommendation for next run

After reviewing these changes, commit and push before using the remote V100
machine for formal training.

A cautious next command is the quick pipeline:

```bash
python scripts/run_experiment_pipeline.py --config configs/pipeline/multidag_cl_iemocap_context_w5_quick.yaml
```

If the quick pass is already trusted, the next formal command can be:

```bash
python scripts/run_experiment_pipeline.py --config configs/pipeline/multidag_cl_iemocap_context_w5.yaml
```

After the formal causal_w5 TAV run finishes, copy its run id into
`configs/pipeline/multidag_cl_iemocap_w5_missing_modalities.yaml`, then run:

```bash
python scripts/run_experiment_pipeline.py --config configs/pipeline/multidag_cl_iemocap_w5_missing_modalities.yaml
```

Required direct answers:

1. `train_multidag_cl.py` now shows real-time train batch progress: yes.
2. Validation evaluation now shows real-time progress during each epoch: yes.
3. `epoch_metrics.csv` refreshes after every epoch: yes.
4. Running only `train_multidag_cl.py` generates figures: no.
5. Running the pipeline can generate figures: yes, when the relevant YAML stages are enabled.
6. `context_w5`, `past_all`, `w0`, `w1`, `w3`, and `quick` pipelines enable figure stages: yes.
7. The missing-modality pipeline enables `make_figures`: yes.
8. Model code was modified: no.
9. MMGCN, SDT, and datasets were modified: no.
10. Next step can be commit/push and then formal remote causal_w5: yes, after review.
