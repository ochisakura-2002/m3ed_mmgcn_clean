# MultiDAG+CL paper artifact export notes

## 1. Purpose

Each formal MultiDAG+CL experiment should produce more than raw logs. The
analysis layer now exports paper-facing tables and figures under:

```text
outputs/runs/<run_id>/paper_artifacts/
  tables/
  figures/
  cases/
```

These artifacts are derived from existing validation-best checkpoint logs and
figures. They do not change model code, train/val/test splits, training
protocol, or checkpoint selection.

## 2. Validation curves vs final test metrics

Training curves in `figures/training_curves/` are validation process metrics:

- validation loss
- validation accuracy
- validation weighted-F1
- validation macro-F1
- validation UAR

They show how validation performance changed by epoch. They are not test
curves.

Final test metrics come from evaluating `best_model.pt` on the test split:

```text
logs/evaluations/test_best_model/metrics.csv
```

`best_model.pt` is selected by validation weighted-F1. Test metrics are final
reporting metrics for that validation-best checkpoint. A high validation score
does not imply an equally high test score, and test metrics must not be used to
choose the best epoch.

## 3. Single-run paper artifacts

Run:

```bash
python scripts/analyze/export_paper_artifacts.py \
  --run-dir outputs/runs/<run_id> \
  --split test \
  --also-val
```

Key table outputs:

- `paper_artifacts/tables/single_run_summary.csv/md/tex`
- `paper_artifacts/tables/per_class_metrics_test.csv/md/tex`
- `paper_artifacts/tables/per_class_metrics_val.csv/md/tex` when `--also-val` is used
- `paper_artifacts/tables/val_vs_test_summary.csv/md/tex`

Key figure outputs:

- `paper_artifacts/figures/overall_test_metrics.png/pdf`
- `paper_artifacts/figures/confusion_matrix_raw_test.png/pdf`
- `paper_artifacts/figures/confusion_matrix_normalized_test.png/pdf`
- `paper_artifacts/figures/per_class_recall_test.png/pdf`
- `paper_artifacts/figures/val_vs_test_classification_metrics.png/pdf`
- `paper_artifacts/figures/val_vs_test_loss.png/pdf`
- `paper_artifacts/figures/val_weighted_f1_curve.png/pdf`
- `paper_artifacts/figures/val_loss_curve.png/pdf`

The `val_vs_test_summary` table records the validation-best epoch, the
validation metrics at that epoch, final test metrics for the validation-best
checkpoint, and validation-minus-test gaps.

## 4. Split diagnostics

Run once per dataset/config:

```bash
python scripts/analyze/diagnose_iemocap_splits.py \
  --config configs/baselines/multidag_cl/iemocap/formal/context_w5_tav.yaml \
  --output-dir outputs/paper_artifacts/split_diagnostics/iemocap
```

Outputs include:

- `split_statistics.csv/md/tex`
- `label_distribution_by_split.csv/md/tex`
- `label_distribution_by_split.png/pdf`
- `utterance_count_by_split.png/pdf`
- `dialogue_length_distribution.png/pdf`
- `split_diagnostics_notes.md`

The train split has many more dialogues than test, so seeing more train batches
than test batches is expected. With the current IEMOCAP setup, the validation
split is derived from official train dialogues, while test comes from the
official test dialogues. A validation-test gap can come from the small
validation set, label distribution differences, dialogue length differences, or
a harder official test split.

Use split diagnostics together with per-class recall and confusion matrices.
Do not switch to test-based checkpoint selection.

## 5. Multi-run paper tables

Run after several completed runs exist:

```bash
python scripts/analyze/export_paper_multi_run_tables.py \
  --run-ids <run1> <run2> <run3> \
  --output-dir outputs/paper_artifacts/tables
```

The first version writes:

- `main_results.csv/md/tex`
- `context_window_results.csv/md/tex` when context/window fields are available
- `modality_ablation_results.csv/md/tex` when active modality fields are available

It writes one row per run. Mean/std aggregation is intentionally left for a
future pass.

## 6. Recommended workflow after each formal run

1. Run the train/eval/analysis pipeline.
2. Run `export_paper_artifacts.py` for the completed run, or enable the
   `paper_artifacts` stage in the pipeline YAML.
3. Run `diagnose_iemocap_splits.py` once per dataset/config.
4. After multiple runs, run `export_paper_multi_run_tables.py`.

The formal w5 pipeline enables:

```yaml
paper_artifacts:
  enabled: true
  split: test
  also_val: true
  output_subdir: paper_artifacts
```

The quick pipeline includes the same section but keeps it disabled so quick
results are not mistaken for formal paper results.

## 7. What not to do

- Do not use test metrics to choose the best epoch.
- Do not report quick or smoke results as formal results.
- Do not mix from-scratch modality ablation with test-time missing-modality
  evaluation.
- Do not confuse validation curves with final test metrics.
