# Config Batch 2 refactor report

## Status

`CONFIG_BATCH2_POST_MOVE_RECOVERY_STATUS=PASS`

`RECOVERY_BASELINE_SOURCE=GIT_HEAD`

`PREVIEW_STATUS=NOT_APPLICABLE_POST_MOVE`

Config Batch 2 migrated the 17 Phase 4A `migration_batch=Batch 2` MMGCN YAML files to their planned canonical paths. The final set contains 13 `unified` configs and 4 `paper_aligned` configs. `paper_aligned` remains a project lineage label and does not mean `author_official`.

## Recovery history

The first Batch 2 attempt was blocked because the audit tool treated `source_config` as an unapproved key. A later execution moved all 17 YAML files before completing the intended pre-move preview, leaving a correct migration with an incomplete audit trail.

This recovery did not roll back, repeat, or replace the migration. It reconstructed every before state from `git show HEAD:<old_path>` at Git HEAD `3e9cafbbae409b08c4f1607dae93072648a80990`, then compared that baseline with the current `candidate_new_path`. The before snapshot records `snapshot_source=git_head`; the after snapshot records `snapshot_source=working_tree`. SHA comparisons use Git canonical text bytes so Windows `core.autocrlf` materialization is not misclassified as YAML content change.

## Semantic audit

- Planned/moved YAML: 17 / 17
- Old paths remaining: 0
- Candidate paths present: 17
- Tracked YAML total: 183
- Git-canonical byte and recursive-semantic identical: 13
- YAML content changed: 4
- `source_config` paths updated: 4
- Unapproved semantic changes: 0

The four changed clean-RoBERTa session configs modify only `provenance.source_config`. Each before value is a Batch 2 old path, each after value is that row's exact planned candidate, every target exists under `configs/`, and model, implementation, dataset, context mode, feature set, and purpose identities remain consistent. Arbitrary, absolute, empty, cross-row, and cross-batch targets are rejected.

## Preview and actual-state accounting

Because the move had already completed, a real pre-move preview was not replayed. The audit records `PREVIEW_STATUS=NOT_APPLICABLE_POST_MOVE` and uses Git HEAD as the preview baseline. Unit tests independently verify that preview is read-only and that preview changes, actual file changes, approved actual changes, and unapproved actual changes use separate counters.

## Reference audit

The reference audit records 90 updated active references, 20 retained historical references, and 0 active old references. Regression execution found 24 additional composed/dynamic path consumers that a contiguous-string scan could not see; launcher, config-validator, and test consumers were updated and added to the audit.

Two YAML-to-YAML references remain only because the Phase 4A reference graph marks them frozen historical references while their Batch 1/Batch 7 source YAML files are outside the Batch 2 edit scope. Historical documentation and migration records retain old paths by design. Production code, active tests, active documentation, and non-historical YAML consumers retain no Batch 2 old path.

## Validation

- Batch migration audit tests: `26 passed`
- Phase 4A plan audit tests: `10 passed`
- MMGCN/config/pipeline/registry/checkpoint regressions: `266 passed, 3 skipped`
- Full pytest: `385 passed, 3 skipped`
- Config tree validation: PASS
- Phase 4A strict plan audit: PASS (`tracked_yaml=183`, existing Batch 7 `candidate_collisions=1`)
- Batch 1 strict regression audit: PASS
- Batch 2 strict audit: PASS
- Git whitespace check: PASS

The no-training dry-run parsed all 17 YAML files, resolved four `source_config` values, verified 17 canonical entrypoints, generated 17 CLI commands, resolved 17 model implementations/registries, and parsed 17 output/checkpoint fields. The native paper-aligned `--dry-run` was not invoked because it verifies the complete formal feature asset before entering its dry-run branch; equivalent config normalization, runtime schema, registry, and command-generation checks were used without loading formal features.

No model code, model mathematics, training behavior, hyperparameters, feature assets, formal outputs, or checkpoints changed. No formal or smoke training was started. No commit or push was performed.

## Artifacts

- `docs/refactors/CONFIG_BATCH2_MOVES.csv`
- `docs/refactors/CONFIG_BATCH2_BEFORE_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH2_AFTER_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH2_SEMANTIC_DIFF.csv`
- `docs/refactors/CONFIG_BATCH2_REFERENCE_AUDIT.csv`
- `scripts/dev/audit_config_batch_migration.py`
- `tests/dev/test_audit_config_batch_migration.py`

The next configuration migration stage is Config Batch 3. Batch 7's existing candidate collision remains unresolved and is not part of this batch.
