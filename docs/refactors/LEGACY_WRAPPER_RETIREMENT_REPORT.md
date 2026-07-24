# Legacy Wrapper Retirement Report

## Scope and preconditions

This task retired the one-migration-cycle model and script compatibility layer
before formal long training. It ran on branch `refactor/model-first-layout` from
HEAD `adebf052e640146c3f1c43db95df9247d511c1ec`, with local/remote ahead-behind
`0/0`, a clean tracked worktree and index, exactly 183 tracked YAML files, and
only the two protected untracked group-meeting files.

No training, commit, push, dependency installation, data modification,
checkpoint operation, or formal-output creation was performed.

## Inventory method

The audit scanned every tracked Python file under `models/` and `scripts/` and
classified source by AST structure, imports, module docstrings, forwarding
behavior, prior refactor mappings, and consumer references. File length was not
used as a wrapper criterion.

Pre-retirement classification:

| Classification | Models | Scripts | Total |
|---|---:|---:|---:|
| canonical implementation | 22 | 65 | 87 |
| canonical package/init | 20 | 31 | 51 |
| confirmed legacy wrapper | 33 | 64 | 97 |
| legacy package init used only by wrappers | 1 | 0 | 1 |
| ambiguous | 0 | 0 | 0 |
| total tracked Python scanned | 76 | 160 | 236 |

The 33 model wrappers are the report-confirmed re-export modules/packages from
the model-first layout migration. The script set includes 63 transparent
`_load_compat_module` aliases from Phases 3A/3B1/3B2 and
`scripts/workflows/paper_aligned/run_reproduction.py`, whose source only
imported and called the final canonical pipeline `main()` without independent
behavior. The latter meets the retirement task's explicit thin-main-forwarder
criterion.

The machine-readable pre-deletion inventory is
`LEGACY_WRAPPER_RETIREMENT_INVENTORY.csv`. Its reference counts are a
pre-retirement snapshot. Per-wrapper counts can overlap for hierarchical
package imports; the repository-wide unique active-reference count is 245.

## Deletion result

- Confirmed model wrappers deleted with `git rm`: 33.
- Confirmed script wrappers deleted with `git rm`: 64.
- Legacy package `__init__.py` files deleted: 12. This count overlaps 11
  re-export wrapper rows and includes one package marker that existed only for
  the retired wrapper tree.
- Wrapper-only compatibility tests deleted: 4.
- Canonical implementation files deleted: 0.
- Canonical package/init files deleted: 0.
- Ambiguous files: 0.

Ignored bytecode caches left by the retired imports were removed only from the
exact retired directories so those directories disappeared from the actual
working tree. The protected untracked group-meeting script keeps its
`scripts/analyze/` directory by design and was not modified.

No replacement wrapper, redirect, symlink, duplicate entrypoint, or copied
implementation was introduced. `scripts/__init__.py` is now a canonical package
initializer and no longer contains the compatibility loader.

## Active reference retirement

The pre-retirement audit found 245 unique active reference lines outside the
wrapper files and frozen historical records. They were either updated to their
final canonical target or removed with the dedicated compatibility tests.

Surviving active consumers were updated in:

- production workflow and analysis code;
- canonical dev/directory checks;
- tests for original-MERC runtime, causal runtime, clean features, launchers,
  output paths, analysis, and command generation;
- active operating documents, protocols, model inventory, smoke instructions,
  implementation specification, and persistent project context.

Frozen historical migration reports, before snapshots, move/reference CSVs,
and historical notes continue to record before paths as history. They are not
recognized as active imports, commands, or entrypoints.

Final active audits:

- `ACTIVE_LEGACY_MODEL_IMPORTS=0`
- `ACTIVE_LEGACY_SCRIPT_REFERENCES=0`
- `ACTIVE_LEGACY_ENTRYPOINTS=0`
- `ACTIVE_OLD_CONFIG_REFERENCES=0`

## Tests and behavior gates

The four wrapper-availability test files were replaced by
`tests/test_legacy_wrapper_retirement.py`. The new gate verifies:

- canonical model, registry, runtime, workflow, and CLI imports;
- retired paths are absent and selected old module paths are not importable;
- all inventory paths are absent;
- seven representative canonical CLIs start through `--help`;
- canonical registry symbols resolve;
- MMGCN strict state-dict loading keeps the same key protocol;
- training/evaluation command registries and the formal 16-run command plan
  remain canonical;
- configs do not reference retired scripts;
- tracked model/script Python contains neither the compatibility loader nor a
  definition-free thin `main()` forwarder.

No canonical model implementation file changed. Model forward behavior,
mathematics, graph construction, causal masks, loss, optimizer behavior,
training controls, data split, checkpoint selection, runtime/checkpoint schema,
and config semantics are unchanged.

## Config and dry-run gates

Final config gates:

- config tree validation: pass;
- Phase 4A strict migration-plan audit: pass, tracked YAML 183 and candidate
  collisions 0;
- Config Batch 1--7 strict audits: pass with moved counts
  `16/17/17/10/13/37/73`;
- tracked YAML count: 183;
- active old config references: 0.

Six read-only/no-training construction dry-runs passed:

1. MMGCN unified, full context;
2. MMGCN unified, causal context;
3. MultiDAG-inspired + CL unified, causal-context synthetic smoke config;
4. DialogueGCN unified, causal context;
5. GS-MCC Project Variant, causal context;
6. GS-MCC Project Variant, full context.

Each dry-run only parsed configuration, validated the applicable canonical
contract, and constructed the model. Epochs started: 0. Outputs created: 0.

## Full regression

The direct system Python command could not collect model tests because that
interpreter does not contain PyTorch. The repository's required
`m3ed_mmgcn` conda environment was therefore used. A sandboxed full run first
encountered only the repository's known permission errors while collecting
existing `tmp/pytest_*` directories. Re-running the same tracked suite with
normal local permissions and explicitly ignoring the protected untracked
`tests/analyze` directory passed:

`309 passed, 3 skipped, 14 warnings`

The reduced count versus the previous 436-test gate is expected: four
wrapper-availability suites were removed and replaced with canonical-only
retirement coverage.

## Final state

- `LEGACY_MODEL_WRAPPERS=RETIRED`
- `LEGACY_SCRIPT_WRAPPERS=RETIRED`
- `CANONICAL_ONLY_LAYOUT=COMPLETED`
- `FORMAL_LONG_TRAINING_STARTED=NO`
- `MODEL_CODE_CHANGED=NO`
- `MODEL_MATH_CHANGED=NO`
- `TRAINING_BEHAVIOR_CHANGED=NO`
- `CONFIG_YAML_COUNT=183`
- `READY_TO_COMMIT_WRAPPER_RETIREMENT=YES`
- `READY_FOR_LONG_TRAINING_CONFIG_PREPARATION=YES`

Old model import paths and old script commands are no longer supported. Future
code, configs, tests, documentation, and commands must use canonical paths.
