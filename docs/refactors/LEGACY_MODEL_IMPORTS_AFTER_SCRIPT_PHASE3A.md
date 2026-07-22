# Legacy Model Imports After Script Phase 3A

## Audit scope and method

The audit parses Python source with `ast` and counts one legacy model edge for
each `import` or `from` statement whose module is `models.baselines` or one of
its descendants. It covers the Phase 3A canonical execution tree, every old
execution-path compatibility wrapper, all deferred scripts, and tests. Package
`__init__.py` files supporting the canonical execution tree are included in the
canonical category.

The two local group-meeting files were classified separately and were neither
opened by the audit nor modified or staged:

- `scripts/analyze/export_group_meeting_baseline_report.py`
- `tests/analyze/test_group_meeting_baseline_report.py`

## Results

| Category | Python files | Parsed | Files with legacy imports | Legacy import edges | Status |
|---|---:|---:|---:|---:|---|
| `canonical_execution` | 33 | 33 | 0 | 0 | PASS |
| `compatibility_wrapper` | 19 | 19 | 0 | 0 | PASS |
| `deferred_script` | 47 | 47 | 10 | 12 | Deferred to Phase 3B |
| `test` | 23 | 23 | 13 | 27 | Compatibility coverage retained |
| `group_meeting_protected` | 2 | 0 | 0 | 0 | Untracked and not parsed |

All 122 audited files parsed successfully; the two protected files were
deliberately not parsed. Canonical execution scripts and old CLI wrappers have
zero direct legacy model imports. The wrappers import only canonical script
modules and alias the legacy module name to the canonical module so existing
imports and monkeypatch-based tests preserve their behavior.

## Deferred script imports

The 12 remaining deferred-script edges are outside the Phase 3A execution
layout scope:

| File | Edges | Legacy targets |
|---|---:|---|
| `scripts/analyze/audit_model_causality.py` | 2 | `models.baselines.causal_baseline_registry` |
| `scripts/baselines/debug_causal_dialoguegcn_forward.py` | 1 | `models.baselines.dialoguegcn` |
| `scripts/baselines/debug_causal_gsmcc_forward.py` | 1 | `models.baselines.gsmcc` |
| `scripts/baselines/debug_multidag_cl_forward.py` | 1 | `models.baselines.multidag_cl` |
| `scripts/baselines/debug_multidag_cl_real_batch.py` | 1 | `models.baselines.multidag_cl` |
| `scripts/baselines/debug_new_causal_graph_real_batch.py` | 1 | `models.baselines.causal_baseline_registry` |
| `scripts/baselines/debug_sdt_forward.py` | 1 | `models.baselines.sdt` |
| `scripts/debug/debug_mmgcn_forward.py` | 1 | `models.baselines.mmgcn.mm_gcn` |
| `scripts/debug/debug_simple_mlp_step.py` | 1 | `models.baselines.simple_mlp` |
| `scripts/dev/diagnose_gsmcc_numerics.py` | 2 | `models.baselines.original_repro`, `models.baselines.original_repro.gsmcc.model` |

These files remain in place for the explicitly deferred analysis/debug/data
script refactor. They do not serve as Phase 3A canonical execution entrypoints.

## Test imports

Tests retain 27 legacy model import edges across 13 files. Ten of those edges
belong to `tests/test_model_layout_compatibility.py`, whose purpose is to prove
old/canonical model symbol identity. The remaining compatibility imports keep
coverage of the existing model wrappers and are not dependencies of canonical
execution code.

## Independent checks

The AST result is cross-checked with text search over Python imports. Broader
matches for runtime keys such as `original_repro_*` are not counted as Python
dependency edges and must not be mechanically renamed. The detailed edge rows
and category summaries are in
`LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3A.csv`.
