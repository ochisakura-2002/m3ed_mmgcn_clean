# MultiDAG+CL stabilization multi-run analysis notes

## 1. Existing scripts checked

This pass checked the existing analysis entry points under `scripts/analyze/`
and the analysis configs under `configs/analysis/`.

Relevant existing scripts:

- `scripts/analyze/build_analysis_tables.py`: rebuilds global master tables
  from `outputs/runs/<run_id>/`.
- `scripts/analyze/plot_multi_run_training_curves.py`: plots selected epoch
  metrics from `outputs/analysis_tables/epoch_metrics_master.csv`.
- `scripts/analyze/plot_multi_run_final_analysis.py`: compares final
  evaluation metrics and per-class metrics from
  `outputs/analysis_tables/evaluation_master.csv` plus confusion matrices.
- `scripts/analyze/export_paper_multi_run_tables.py`: exports lightweight
  paper-facing result tables across run ids.
- `scripts/analyze/plot_missing_modality_summary.py`: plots one missing
  modality summary CSV for one run.
- `scripts/analyze/diagnose_multidag_cl_run.py`: diagnoses one MultiDAG+CL run
  from epoch/evaluation logs and writes optional diagnostic CSV/Markdown/plots.

The existing `plot_multi_run_training_curves.py` and
`plot_multi_run_final_analysis.py` remain useful for general-purpose
multi-run plots. They are not replaced.

## 2. Why a dedicated script was added

The 20260709 stabilization comparison needs a compact one-command artifact set:

- run summary table with config factors, best epoch, final losses, and test
  metrics;
- ranking by test Weighted-F1;
- deltas from the original w5 baseline;
- validation Weighted-F1 curves;
- train/validation loss overlay;
- best epoch, validation-test gap, and metric heatmap plots;
- a Markdown report with cautions about single-seed interpretation and
  validation-only checkpoint selection.

The existing multi-run scripts can cover part of this, but they depend on
global master tables and do not directly write the stabilization-specific
delta/gap/report artifacts. Therefore this pass added a narrow dedicated
script:

```text
scripts/analyze/plot_multidag_cl_stabilization_compare.py
```

It reads the same lightweight run logs directly and does not load checkpoints.

## 3. 20260709 config path

```text
configs/analysis/multidag_cl/iemocap/stabilization_20260709_compare.yaml
```

The config contains eight runs:

1. `original_w5` baseline.
2. `lr5e4`.
3. `lr3e4`.
4. `stable_candidate`.
5. `linear_encoder`.
6. `graph1`.
7. `dropout02`.
8. `dropout01`.

## 4. Run command

From the project root, after the run logs are available under
`outputs/runs/<run_id>/`:

```bash
python scripts/analyze/plot_multidag_cl_stabilization_compare.py \
  --config configs/analysis/multidag_cl/iemocap/stabilization_20260709_compare.yaml
```

This is an analysis-only command. It does not run training and does not read
checkpoint files.

## 5. Output directory

```text
outputs/analysis/multidag_cl/iemocap/stabilization_20260709/
```

Expected tables:

```text
tables/run_summary.csv
tables/run_ranking.csv
tables/delta_from_baseline.csv
```

Expected report:

```text
analysis_report.md
```

Expected figures:

```text
figures/final_test_metrics_bar.png
figures/final_test_metrics_bar.pdf
figures/test_weighted_f1_ranking.png
figures/test_weighted_f1_ranking.pdf
figures/delta_from_baseline.png
figures/delta_from_baseline.pdf
figures/val_weighted_f1_curves.png
figures/val_weighted_f1_curves.pdf
figures/loss_curves_overlay.png
figures/loss_curves_overlay.pdf
figures/best_epoch_bar.png
figures/best_epoch_bar.pdf
figures/val_test_gap.png
figures/val_test_gap.pdf
figures/metric_heatmap.png
figures/metric_heatmap.pdf
```

## 6. Figure interpretation

- `final_test_metrics_bar`: compares test Accuracy, Weighted-F1, Macro-F1, and
  UAR across all eight runs.
- `test_weighted_f1_ranking`: ranks runs by test Weighted-F1 and marks
  `original_w5`.
- `delta_from_baseline`: shows stabilization-run deltas relative to
  `original_w5`.
- `val_weighted_f1_curves`: shows validation Weighted-F1 by epoch.
- `loss_curves_overlay`: overlays train and validation loss curves for all
  runs.
- `best_epoch_bar`: shows which epoch each run selected by validation
  Weighted-F1.
- `val_test_gap`: plots `best_val_weighted_f1 - test_weighted_f1`.
- `metric_heatmap`: uses column-normalized colors and raw value annotations for
  core test metrics, validation Weighted-F1, and final validation loss.

These plots support stabilization diagnosis. They are not multi-seed
significance evidence by themselves.

## 7. Reusing for a new comparison

For a later stabilization or context comparison:

1. Copy `stabilization_20260709_compare.yaml` to a new config name.
2. Update `analysis_name`, `output_dir`, `baseline_run_id`, and the `runs` list.
3. Keep display names short enough for figure labels.
4. Run the same script with the new config path.

The script assumes each run has:

```text
outputs/runs/<run_id>/logs/experiment_config.yaml
outputs/runs/<run_id>/logs/epoch_metrics.csv
outputs/runs/<run_id>/logs/evaluations/val_best_model/metrics.csv
outputs/runs/<run_id>/logs/evaluations/test_best_model/metrics.csv
```

It will also read this optional file when present:

```text
outputs/runs/<run_id>/figures/diagnostics/multidag_cl_training_diagnosis.csv
```

Missing diagnostics do not fail the analysis. Missing required logs do fail the
analysis with a list of absent files.

## 8. Stable-candidate context comparison

For the post-stabilization H-R context sweep, use the general multi-run
training/final-analysis scripts with this template after all stable context run
ids exist:

```text
configs/analysis/multidag_cl/iemocap/stable_context_compare.yaml
```

Expected display names:

```text
stable_w0
stable_w3
stable_w5
stable_past_all_causal
```

This comparison is separate from `stabilization_20260709_compare.yaml`: the
20260709 file explains why the conservative stable-candidate setting was
chosen, while `stable_context_compare.yaml` compares context lengths under that
chosen setting.
