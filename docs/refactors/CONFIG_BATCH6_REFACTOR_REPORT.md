# Config Batch 6 Refactor Report

## Scope

Batch 6 migrated 37 YAML files using the exact frozen Phase 4A plan: 2 cross-model benchmarks and 35 model-scoped ablations. The migration plan and classification inventory were not edited.

## Layout-role result

- `cross_model_benchmark`: 2
- `model_scoped_ablation`: 35
- `causal_unified`: 1
- `original_merc`: 1
- `NOT_APPLICABLE`: 35 (all model-scoped ablations)

Smoke remains an orthogonal classification attribute; the single smoke config remains one of the 35 model-scoped ablations.

## Integrity and protocol

All 37 YAML files are byte-identical and semantic-identical before and after `git mv`. Model membership, model order, provenance, ablation variables, controlled variables, modality settings, split/seed/metric fields, and runtime budgets are unchanged. No `author_official` lineage was introduced; GS-MCC benchmark members remain `project_variant`. Author-official MultiDAG+CL and GS-MCC reproduction work has not started.

## References

- Active references updated: 59
- Batch 7 YAML references updated in place: 27
- Frozen historical references retained: 109
- Active old-path references remaining: 0

## Artifacts

The before/after snapshots, move ledger, semantic diff, reference audit, and cross-model membership audit are stored beside this report. Batch 7 YAML files were not moved.

## Gates

- Batch 6 audit tests: `79 passed`
- Phase 4A plan audit tests: `10 passed`
- Full tracked pytest, excluding the protected untracked group-meeting test: `430 passed, 3 skipped, 14 warnings`
- Config tree validation: pass
- Phase 4A strict plan audit: pass
- Batch 1--6 strict migration audits: pass
- Safe 37-config dry-run: pass; no epoch or formal training started
- `git diff --check`: pass
