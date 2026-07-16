# Original MERC remote execution handoff

This document is the command-oriented companion to
`docs/experiments/original_merc_handoff.md`.

## Required inputs

```text
third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl
data/processed/iemocap/IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.pkl
```

The expected SHA256 values are stored in every config and verified before model
construction. No current local result is a real-data performance claim.

## Date-organized outputs

New Original MERC runs use `outputs/<YYYYMMDD>/runs/<run_id>/`. Pipeline logs,
analysis, reports, manifests, audits, and review artifacts use the corresponding
dated categories. Resolve order is `--experiment-date`,
`output.experiment_date`, `MERC_EXPERIMENT_DATE`, then the launch machine's
local date. The pipeline freezes that value once and passes it to every job, so
an overnight smoke/screening/fold batch remains under its launch date.

Resume from `last_model.pt` continues in the checkpoint's existing run
directory even on another day. Explicit checkpoint, run, and output paths are
never relocated. Old `outputs/runs/...` and former Original MERC trees remain
readable and must not be moved or renamed. `outputs/environment`,
`outputs/reference`, and `outputs/cache` are the only global static exceptions.

## Ordered commands

```bash
python -m pytest -q
python scripts/dev/validate_config_tree.py
git diff --check

python scripts/baselines/run_original_merc_pipeline.py --stage smoke
python scripts/baselines/run_original_merc_pipeline.py --stage smoke --execute --experiment-date 20260716

python scripts/baselines/run_original_merc_pipeline.py --stage legacy_paper_adjacent_screening
python scripts/baselines/run_original_merc_pipeline.py --stage legacy_paper_adjacent_screening --execute

python scripts/baselines/run_original_merc_pipeline.py --stage clean_screening
python scripts/baselines/run_original_merc_pipeline.py --stage clean_screening --execute

python scripts/analyze/analyze_original_merc_results.py

python scripts/baselines/run_original_merc_pipeline.py --stage top2
python scripts/baselines/run_original_merc_pipeline.py --stage top2 --execute

python scripts/analyze/analyze_original_merc_results.py
```

The environment-variable equivalent is:

```bash
MERC_EXPERIMENT_DATE=20260716 python scripts/baselines/run_original_merc_pipeline.py --stage clean_screening --execute
```

Inspect or analyze one launch day with:

```bash
find outputs/20260716/runs -maxdepth 1 -type d
find outputs/20260716/logs -type f
find outputs/20260716/analysis -maxdepth 2 -type f
python scripts/analyze/analyze_original_merc_results.py --runs-root outputs/runs --output-dir <explicit-old-analysis-dir>
```

The manifest interleaves smoke configs; if strict ordering is desired, run the
four `_legacy.yaml` files first and their four `_clean.yaml` counterparts
second. Formal training and multi-seed evaluation belong on the remote V100.

## Audit cautions

- Checkpoint and top-two selection use Validation only. Test is final reporting.
- The original-split diagnostic, legacy five-fold comparison, and clean
  five-fold comparison must never share one ranking.
- `project_paper_oriented_gsmcc` is an engineering variant, not evidence of
  reproducing GS-MCC.
- DialogueGCN has a formula-level graph fixture, but real-data performance is
  still unconfirmed.
- Causal implementations and configs are out of scope and remain unchanged.

Suggested commit message (not executed):

```text
fix: align original MERC reproductions and selection protocol
```
