# Legacy Analysis Imports After Script Phase 3B1

## Audit scope and method

The audit parsed Python source with `ast` and counted one legacy analysis edge for each `import` or `from` statement targeting `scripts.analyze`, one of its descendants, or the former `scripts.analysis.analyze_original_merc_results` compatibility module.

The source set contained all 223 Python files currently tracked in the Git index plus 20 Phase 3B1 package/wrapper/test files created in the working tree and intended for tracking, for 243 parsed files total. The working-tree candidates were included so newly created compatibility code could not escape the gate merely because no commit or broad `git add` was allowed. All files parsed successfully.

The audit excluded `outputs/`, `data/`, `third_party/`, `tmp/`, its own two output files, and the two protected untracked group-meeting files. The protected files were not opened or parsed:

- `scripts/analyze/export_group_meeting_baseline_report.py`
- `tests/analyze/test_group_meeting_baseline_report.py`

## AST results

| Category | Python files | Parsed | Files with legacy analysis imports | Legacy edges | Status |
|---|---:|---:|---:|---:|---|
| `canonical_analysis` | 19 | 19 | 0 | 0 | PASS |
| `compatibility_wrapper` | 15 | 15 | 0 | 0 | PASS |
| `deferred_script` | 5 | 5 | 0 | 0 | Deferred to Phase 3B2 |
| `other_tracked_python` | 180 | 180 | 0 | 0 | PASS |
| `test` | 24 | 24 | 5 | 6 | Compatibility imports retained |
| `group_meeting_protected` | 2 | 0 | 0 | 0 | Untracked and not parsed |

`CANONICAL_ANALYSIS_LEGACY_IMPORTS=0`.

The six remaining AST edges are test-only references. Four exercise migrated wrapper compatibility, while two continue to import the feature/data scripts deliberately deferred to Phase 3B2. No production canonical analysis module imports a legacy analysis wrapper.

## Compatibility wrappers

There are 15 thin compatibility entrypoints: 13 wrappers created at the old tracked locations of moved implementations plus two pre-existing aliases redirected directly to the final canonical Original-MERC module. Their source imports only `scripts._load_compat_module`; there is no wrapper chain, duplicated argparse, duplicated metric logic, or reverse canonical dependency.

## Deferred analysis files

Five source-confirmed feature/data or diagnose files remain in place for Phase 3B2:

- `scripts/analyze/audit_iemocap_feature_pkl.py`
- `scripts/analyze/diagnose_iemocap_splits.py`
- `scripts/analyze/diagnose_loss_stability.py`
- `scripts/analyze/diagnose_multidag_cl_run.py`
- `scripts/analyze/probe_iemocap_text_features.py`

## Legacy CLI text references

An independent UTF-8 text scan found 139 occurrences of the old slash-form CLI path across 32 files, excluding this audit's two outputs:

| Category | Files | References | Disposition |
|---|---:|---:|---|
| `documentation_reference` | 29 | 116 | Historical audits/handoffs and explicit compatibility documentation retained |
| `compatibility_wrapper` | 1 | 7 | `scripts/workflows/run_pipeline.py` targets retained wrappers so unchanged configs keep working |
| `deferred_script` | 1 | 1 | Deferred script's self-usage example retained until Phase 3B2 |
| `test` | 1 | 15 | Dedicated compatibility mapping retained |
| Total | 32 | 139 | Audited |

These text references are not Python dependency edges. Historical documents were not rewritten, the Phase 3A workflow file was outside the allowed modification scope, and the compatibility targets remain valid. New canonical source docstrings use canonical paths.

## Removal gate

The Phase 3B1 wrappers must remain until Phase 3B2 and config migration are complete, old CLI consumers have moved to canonical paths, whole-project tests pass, and a separate removal task retires the wrappers and their dedicated compatibility tests together.
