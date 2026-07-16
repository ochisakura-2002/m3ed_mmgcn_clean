# Original MERC focused-fix handoff

## Delivered protocol

The suite now has three mutually exclusive `experiment_track` values:

- `legacy_official_split_safe_selection`: original trainVid pool, original
  testVid/Ses05 Test, dialogue Validation drawn only from trainVid, and
  `paper_adjacent_not_exact` comparability.
- `legacy_fivefold_fair_comparison`: legacy features with Ses01--Ses05 outer
  folds and inner dialogue Validation.
- `clean_roberta_fivefold_fair_comparison`: Clean RoBERTa v1 with the same fair
  five-fold scheme.

Analysis produces separate per-track summaries and rankings. Only eligible
models in the first track can receive a paper-gap diagnostic.

## Model fidelity

- MMGCN original-reproduction configs enable residual propagation.
- DialogueGCN uses explicit paper-formula graph layers and a numerical fixture;
  real-data fidelity remains unconfirmed.
- MultiDAG+CL retains its architecture and now tests curriculum behavior on
  realistic dialogue labels/speakers, including split isolation.
- GS-MCC uses option B: `project_paper_oriented_gsmcc`, status
  `PROJECT_VARIANT_NOT_PAPER_REPRODUCTION`, excluded from paper ranking.

## Remote execution sequence

1. Run `python -m pytest -q`,
   `python scripts/dev/validate_config_tree.py`, and `git diff --check`.
2. Run the four legacy smoke configs, then the four clean smoke configs. Smoke
   epoch train/Validation remain capped at one batch; reloaded final Validation
   and final Test are full-split evaluations.
3. Run stage `legacy_paper_adjacent_screening` for diagnostic evidence only.
4. Run stage `clean_screening` for all four candidates. These jobs use seed 42,
   outer Test Ses05, Clean RoBERTa v1, and no batch cap.
5. Run `python scripts/analyze/analyze_original_merc_results.py`. Top-two
   selection stays pending until all four clean screening results exist and
   never reads Test metrics as selection evidence.
6. Review generated top-two jobs, then execute stage `top2` on the remote V100.
7. Run `legacy_folds` only when the legacy-feature fair-comparison table is
   needed; run `clean_folds`/resolved top-two jobs for final baseline evidence.
8. Re-run the canonical analysis entrypoint and keep rankings separated by
   experiment track.

## Output and resume contract

Every pipeline resolves one legal `YYYYMMDD` launch date using this priority:
CLI `--experiment-date`, config `output.experiment_date`, environment
`MERC_EXPERIMENT_DATE`, local machine date. All child runs inherit the frozen
value, including jobs that finish after midnight. New runs live under
`outputs/<date>/runs`; the pipeline owns a unique directory under
`outputs/<date>/manifests` and never relies on a shared global
`latest_run.txt`.

Use `--experiment-date 20260716` on the pipeline CLI or export
`MERC_EXPERIMENT_DATE=20260716`. Resume always writes into the run containing
the supplied checkpoint rather than a new date. Automatic analysis discovers
new dated runs first and legacy `outputs/runs` results second. Explicit
`--runs-root`, `--run-dir`, checkpoint, and `--output-dir` paths are obeyed
unchanged. `outputs/environment`, `outputs/reference`, and `outputs/cache`
remain global static directories.

```bash
find outputs/20260716/runs -maxdepth 1 -type d
find outputs/20260716/logs -type f
find outputs/20260716/analysis -maxdepth 2 -type f
python scripts/analyze/analyze_original_merc_results.py --runs-root outputs/runs --output-dir <legacy-analysis-dir>
```

No real-data result is claimed until both pinned feature PKLs pass SHA256
verification. Do not commit generated reports, checkpoints, PKLs, or placeholder
CSVs.
