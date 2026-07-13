# Causal benchmark review tool notes

## Purpose

`scripts/analyze/summarize_causal_benchmark_runs.py` audits the four MMGCN and four MultiDAG-inspired causal benchmark runs listed in `configs/analysis/causal_benchmark_8run_review.yaml`. It also compares the earlier duplicate MultiDAG Ses01 run with the configured formal Ses01 candidate.

The tool only reads existing run artifacts. It does not train models, evaluate checkpoints, create placeholder runs, or infer missing experiment values.

## Remote command

Run from the repository root after the real run directories have been produced:

```bash
python scripts/analyze/summarize_causal_benchmark_runs.py \
  --config configs/analysis/causal_benchmark_8run_review.yaml
```

Optional path overrides are available for a different remote layout:

```bash
python scripts/analyze/summarize_causal_benchmark_runs.py \
  --config configs/analysis/causal_benchmark_8run_review.yaml \
  --runs-root outputs/runs \
  --output-dir outputs/analysis/causal_benchmark_8run_review \
  --report-path docs/baselines/causal_benchmark_8run_review.md \
  --strict
```

All paths may be relative to the repository root. No local or remote absolute path is embedded in the configuration.

## Inputs

For every configured formal run and the duplicate Ses01 run, the tool checks or reads:

- `run_metadata.json`
- `logs/experiment_config.yaml`
- `logs/epoch_metrics.csv`
- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- supported JSON, CSV, YAML, and YML files recursively under `logs/evaluations/`

Evaluation artifacts are ranked using their split metadata, path, artifact role, and checkpoint identifier. Metric values are never used to select an evaluation file. A Test source is confirmed only when the chosen artifact identifies the Test split and `best_model.pt`, the run has that checkpoint, and the saved run configuration selects checkpoints by the configured validation metric.

Both current training implementations update the best checkpoint with strict improvement (`>` for `val_weighted_f1`). Consequently, an exact tie is resolved to the earliest tied epoch. The generated report records this policy and its code source.

## Outputs

The default output directory contains:

```text
outputs/analysis/causal_benchmark_8run_review/
  selected_runs.csv
  run_completeness.csv
  protocol_consistency.csv
  best_validation_results.csv
  test_results.csv
  four_session_statistics.csv
  training_stability.csv
  duplicate_ses01_comparison.csv
  source_files.txt
```

The Markdown audit is written to:

```text
docs/baselines/causal_benchmark_8run_review.md
```

CSV numeric values are written with at least six decimal places. Missing or unverifiable facts remain empty or are explicitly marked `UNCONFIRMED`; they are never replaced with zero. Aggregate rows report their effective sample count `n` and use the configured sample-standard-deviation `ddof`.

## Failure behavior

If any configured run directory is absent, the command exits before creating the output directory or Markdown report and prints every missing run path. This is the expected local behavior when the remote `outputs/runs` data is unavailable.

Default mode can still generate a factual report when a present run has an incomplete or unrecognized artifact; affected facts are marked `UNCONFIRMED`. `--strict` writes the audit outputs and then returns a non-zero exit code when a formal run is incomplete, its best validation epoch cannot be established, its Test result cannot be tied to the validation-selected checkpoint, a required protocol check fails, or the configured newer duplicate target is incomplete.
