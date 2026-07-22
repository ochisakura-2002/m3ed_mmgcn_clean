# Legacy Model Imports After Model Refactor

## Audit scope and method

This audit covers every Git-tracked Python file under `scripts/` and `tests/`. It explicitly excludes the protected local group-meeting files:

- `scripts/analyze/export_group_meeting_baseline_report.py`
- `tests/analyze/test_group_meeting_baseline_report.py`

The file list came from `git ls-files -- scripts tests`; Python then filtered `.py` files and parsed each file with `ast`. One dependency edge means one AST `import` or `from ... import ...` statement targeting one module. A multi-symbol `from` statement therefore remains one edge. The detailed rows are in `LEGACY_MODEL_IMPORTS_AFTER_MODEL_REFACTOR.csv`.

## Summary

| Scope | Tracked Python files with legacy imports | Legacy import edges |
|---|---:|---:|
| `scripts/` | 21 | 26 |
| `tests/` | 13 | 27 |
| Total | 34 | 53 |

- Tracked Python files parsed: 88.
- AST parse errors: 0.
- AST audit status: PASS.
- CSV source-file status: every recorded source is Git tracked; the protected untracked files are absent.
- Canonical `models/` cross-check: 42 Python files outside `models/baselines/` parsed, with 0 reverse imports from `models.baselines`.

## Why these imports are currently allowed

The completed task was deliberately limited to the model layout. Production scripts and existing tests were not migrated, and the old `models/baselines/` modules were retained as thin identity-preserving re-export wrappers for exactly this transition. Keeping these imports temporarily preserves existing entry points and tests while the model-only refactor is reviewed; it does not mean the old paths remain valid for new code.

Configuration values such as `original_repro_mmgcn` are runtime-protocol compatibility identifiers, not Python import paths or proof of author-official provenance. They are outside this import audit and must not be mechanically deleted or renamed.

## Next canonical import migration

The scripts refactor should move production, debug, analysis and evaluation imports to these canonical targets:

| Legacy target | Canonical target family |
|---|---|
| `models.baselines.causal_baseline_registry` | `models.registry.causal` |
| `models.baselines.causal_graph_common` | `models.common.causal_graph` |
| `models.baselines.dialoguegcn` | `models.dialoguegcn.unified` |
| `models.baselines.gsmcc` | `models.gsmcc.project_variant.causal` |
| `models.baselines.mmgcn.mm_gcn` | `models.mmgcn.unified.mm_gcn` |
| `models.baselines.multidag_cl` | `models.multidag_cl.unified` |
| `models.baselines.original_repro` and `.registry` | `models.registry.paper_aligned` and the relevant model's `paper_aligned` package |
| `models.baselines.original_repro.dialoguegcn` | `models.dialoguegcn.paper_aligned` |
| `models.baselines.original_repro.gsmcc.model` | `models.gsmcc.project_variant.full_context.model` |
| `models.baselines.original_repro.multidag_cl` | `models.multidag_cl.paper_aligned` |
| `models.baselines.sdt` | `models.experimental.sdt` |
| `models.baselines.simple_mlp` | `models.simple_mlp.model` |

Non-compatibility tests should migrate with the production paths they exercise. `tests/test_model_layout_compatibility.py` intentionally imports both old and canonical symbols and should remain until the wrapper-removal gate.

## Wrapper removal gate

Compatibility wrappers can be removed only in a separate follow-up after:

1. Production scripts and non-compatibility tests use canonical imports.
2. Whole-model pickle/module-path and checkpoint compatibility risks have an explicit disposition.
3. At least one migration cycle and the complete pytest/config/diff gates pass.
4. A repeated tracked-file AST audit finds no legacy consumers except the dedicated compatibility test.
5. The compatibility test and wrappers are then retired or replaced together, without conflating runtime keys such as `original_repro_*` with imports.

## `git grep` cross-check

The required independent command was run:

```text
git grep -n -E "models\.baselines|original_repro|causal_baseline_registry" -- "*.py"
```

Every AST-recorded import statement was present in the text results. The broader text expression also matched non-import material: runtime/config keys, function uses, dataset module names, and compatibility-wrapper internals. Those extra text matches explain why raw `git grep` output is not itself the import count; they do not contradict the AST result.

The previously reported manual `NO_MATCHES` result is rejected. It arose from the local command environment/search-variable handling and did not reflect the repository's actual imports.
