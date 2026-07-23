# Config Layout Refactor Batch 1 Report

## 1. Git precheck

- Repository root: verified with `git rev-parse --show-toplevel`; no checkout
  absolute path is persisted in configuration or code.
- Branch: `refactor/model-first-layout`
- Base HEAD: `870ac7dd5d58b6e0a1f4012cf10335db836b2f1c`
- Upstream: `origin/refactor/model-first-layout`
- Ahead/behind before migration: `0/0`
- The recent history contains the model, execution, analysis, support, and Config
  Phase 4A commits.
- Tracked and staged state was clean before this task.
- The only untracked files were the two protected group-meeting files.
- No commit or push was performed.

## 2. Baseline gates

- Pre-migration full pytest: `357 passed, 3 skipped, 30 warnings`.
- Pre-migration config validation: PASS.
- Pre-migration `git diff --check`: PASS.
- Phase 4A strict plan audit:
  `CONFIG_MIGRATION_PLAN_AUDIT=PASS tracked_yaml=183 candidate_collisions=1`.
- The first sandboxed pytest collection was blocked only by existing
  `tmp/pytest_*` and pytest cache permissions. The same command passed under
  normal local permissions.

## 3. Batch 1 exact inventory

The only source of migration membership was
`docs/refactors/CONFIG_MIGRATION_PLAN_PHASE4A.csv`, filtered by
`migration_batch=Batch 1`.

- Planned rows: 16.
- Unique old paths: 16.
- Unique candidate paths: 16.
- Missing or untracked old paths before move: 0.
- Candidate paths already present before move: 0.
- `manual_review != NO`: 0.
- `collision_status != CLEAR`: 0.
- `author_official`: 0.
- Historical output snapshots: 0.
- Batch 2--7 rows included: 0.

## 4. Before SHA and semantic snapshot

`docs/refactors/CONFIG_BATCH1_BEFORE_SNAPSHOT.json` records the SHA256,
top-level keys, full recursively parsed YAML structure, canonical entrypoint,
model/dataset fields, split/seed/epoch/batch-size fields, feature paths,
output roots, config references, and script references for all 16 YAML files.
All 16 files parsed successfully.

| Old path | Canonical path | Pre-move SHA256 |
|---|---|---|
| `configs/analysis/four_model_causal_audit_synthetic.yaml` | `configs/benchmarks/causal_unified/analysis/four_model_audit_synthetic.yaml` | `ed55a6033ee1575869d25fc687e964fbd47848b0740f900bfb5eeddea1bade69` |
| `configs/data/iemocap_feature_sets.yaml` | `configs/_shared/data/iemocap/feature_sets.yaml` | `a4fa15c479e056986d3b54a695424c592f7c9634cbd4e157423fa08c8b573212` |
| `configs/smoke/causal_dialoguegcn_iemocap_synthetic.yaml` | `configs/dialoguegcn/unified/synthetic/causal_context/synthetic/audit_fixture.yaml` | `6cbe3f9851d98c55a9cd951c04802e975cde4ec3e9cebcb39d5ee2c4e6f376af` |
| `configs/smoke/causal_gsmcc_iemocap_synthetic.yaml` | `configs/gsmcc/project_variant/synthetic/causal_context/synthetic/audit_fixture.yaml` | `41a05c942ad0fa2325120b830a4a664303c41e53af87f53514e98cc8d2d92ed9` |
| `configs/smoke/original_repro/dialoguegcn_clean.yaml` | `configs/dialoguegcn/paper_aligned/iemocap/full_context/clean_roberta_features/smoke.yaml` | `96801061aa80552d575576ce895742cf483cc6dde4741115c5f53de32f124605` |
| `configs/smoke/original_repro/dialoguegcn_legacy.yaml` | `configs/dialoguegcn/paper_aligned/iemocap/full_context/legacy_mmgcn_features/smoke.yaml` | `8776cac7f7d00a15460be70693eb750bc57fdc0ad4d5e26dbcb24d8f15218f58` |
| `configs/smoke/original_repro/mmgcn_clean.yaml` | `configs/mmgcn/paper_aligned/iemocap/full_context/clean_roberta_features/smoke.yaml` | `681db08fdfda7e3962f6ae176d0f42874ccf74a240f784f1a2b6c3317d0be81c` |
| `configs/smoke/original_repro/mmgcn_legacy.yaml` | `configs/mmgcn/paper_aligned/iemocap/full_context/legacy_mmgcn_features/smoke.yaml` | `7bd92742c15ac0ed519e035e993cd3c08e38320dd29367ce76c1378d76171f7d` |
| `configs/smoke/original_repro/multidag_cl_clean.yaml` | `configs/multidag_cl/paper_aligned/iemocap/full_context/clean_roberta_features/smoke.yaml` | `cb0f73cdd0af5e40e5decf927412b8fe07eefb94d487beef98d8f46bd7e75c23` |
| `configs/smoke/original_repro/multidag_cl_legacy.yaml` | `configs/multidag_cl/paper_aligned/iemocap/full_context/legacy_mmgcn_features/smoke.yaml` | `c833701fafa4153c567a5c6de580905f7a7bc489d48c507e72f29e4b135717ef` |
| `configs/smoke/train_causal_dialoguegcn_end_to_end.yaml` | `configs/dialoguegcn/unified/synthetic/causal_context/synthetic/smoke_end_to_end.yaml` | `333d39c052d0707ee1564511592a55f8b634d31af5b5f21570e8f8b9326b0386` |
| `configs/smoke/train_causal_dialoguegcn_iemocap_real_2epoch.yaml` | `configs/dialoguegcn/unified/iemocap/causal_context/legacy_mmgcn_features/smoke_real_2epoch.yaml` | `120c86a0508bc21725b7087132afc236a5c1ae19dbf64826c0993cdcc53f3369` |
| `configs/smoke/train_causal_gsmcc_end_to_end.yaml` | `configs/gsmcc/project_variant/synthetic/causal_context/synthetic/smoke_end_to_end.yaml` | `f7a1c2e994cd76efabe9309fb84e0824f845fb24561e3dea52217d97260f5e94` |
| `configs/smoke/train_mmgcn_smoke.yaml` | `configs/mmgcn/unified/synthetic/full_context/synthetic/smoke.yaml` | `890e71fcffd20277275bbc063fe8bafb391895a2abb3aed336e45e2ac44ea030` |
| `configs/smoke/train_multidag_cl_smoke.yaml` | `configs/multidag_cl/unified/synthetic/causal_context/synthetic/smoke.yaml` | `cf542c9478623730380a673cfc7732363b3982febaa81e2b48121be203f670e2` |
| `configs/train_simple_mlp_m3ed.yaml` | `configs/simple_mlp/unified/m3ed/full_context/m3ed_features/development.yaml` | `4c18a538448ed8d4872a1ed2391ea8ea667215bb511aee078802f2b3a395ade3` |

## 5. Git moves

All 16 mappings were executed with `git mv`. Only parent directories required
by actual YAML targets were created. No `.gitkeep` files were added.

- `BATCH1_OLD_YAML_REMAINING=0`
- `BATCH1_NEW_YAML_PRESENT=16`
- YAML wrapper/redirect/symlink count: 0.
- New/old dual-truth count: 0.

## 6. Canonical paths

The canonical paths are exactly the `new_path` column of
`docs/refactors/CONFIG_BATCH1_MOVES.csv`; the table above is the human-readable
equivalent. No path was inferred from its old filename outside the Phase 4A
plan.

## 7. YAML content changes

- Byte-changing YAML edits: 0.
- Approved changed keys: none.
- Feature asset path changes: 0.
- Hyperparameter, split, seed, checkpoint, modality, or metric changes: 0.
- Training behavior changes: 0.

## 8. Active references

`docs/refactors/CONFIG_BATCH1_REFERENCE_AUDIT.csv` records 49 reference rows:

- Updated active reference rows: 28.
- Retained historical reference rows: 20.
- Allowed active directory-pattern rows not naming a Batch 1 old file: 1.
- Exact Batch 1 old-file references left in active code/tests/YAML/docs: 0.

The strict Batch audit discovered one Phase 4A graph omission:
`utils/iemocap_features.py` constructed the old feature-registry path. It was
updated to `configs/_shared/data/iemocap/feature_sets.yaml` and recorded as a
reference-graph gap. `scripts/dev/validate_config_tree.py` had the same
constructed registry path and a dynamic Original-MERC smoke directory rule;
both were updated.

## 9. Historical references

Twenty frozen historical-document edges retain their original commands and
paths. Each is marked `historical_reference=YES`, `updated=NO`, and
`remaining_reference_allowed=YES` in the reference audit. Historical output
snapshots were not scanned or edited.

## 10. Entrypoint audit

- Executable Batch 1 configs with an existing canonical entrypoint: 15.
- Declarative shared-data config with no single entrypoint: 1, correctly marked
  not applicable.
- Missing canonical entrypoints: 0.
- Entrypoint rewrites inside YAML: 0.
- SimpleMLP remains mapped to the SimpleMLP trainer.
- GS-MCC remains `project_variant`; no `author_official` path was created.

## 11. Old-path audit

The strict Batch audit scans tracked/staged text while excluding protected
roots. It found no exact Batch 1 old YAML path in production code, tests,
active YAML, or active docs. Remaining exact old paths occur only in historical
docs and migration/audit records.

## 12. Tracked YAML count

- Before migration: 183.
- After migration: 183.
- Missing YAML: 0.
- Byte-identical duplicate YAML introduced by Batch 1: 0.
- Batch 2--7 YAML moved: 0.
- Manual-review pair moved: 0.

## 13. Semantic diff

`docs/refactors/CONFIG_BATCH1_AFTER_SNAPSHOT.json` contains 16 parsed entries.
`docs/refactors/CONFIG_BATCH1_SEMANTIC_DIFF.csv` contains 16 PASS rows:

- `byte_identical=YES`: 16.
- `semantic_identical=YES`: 16.
- Non-approved semantic changes: 0.
- `SEMANTIC_AUDIT_STATUS=PASS`.

## 14. Targeted tests

- Batch migration audit synthetic fixtures: `10 passed`.
- Phase 4A plan audit fixtures after adding executed-move coverage:
  `10 passed`.
- Combined Batch audit, plan audit, feature-registry, causal synthetic,
  Original-MERC analysis, and model-contract consumers: `70 passed`.
- An initial combined run produced `69 passed, 1 failed` only because the new
  test omitted an imported constant; the import was corrected and the same
  suite then passed.

## 15. Full pytest

- Before migration: `357 passed, 3 skipped, 30 warnings`.
- After migration: `369 passed, 3 skipped, 30 warnings`.
- No model/training assertion failed.

## 16. Config validation

`scripts/dev/validate_config_tree.py` was updated for the canonical shared-data
registry and the split Original-MERC smoke layout. Validation result: PASS.

## 17. Phase 4A plan audit

The read-only plan audit now accepts exactly one tracked truth per mapping:
either the unmigrated `old_path` or the executed `candidate_new_path`. It still
rejects old/new dual truth, missing mappings, invalid provenance, and
unresolved unmarked collisions.

Final result:

`CONFIG_MIGRATION_PLAN_AUDIT=PASS tracked_yaml=183 candidate_collisions=1`

The remaining collision is unchanged and belongs to Batch 7.

## 18. Batch 1 audit

New read-only tool:
`scripts/dev/audit_config_batch_migration.py`.

It validates plan/move cardinality, old/new existence and tracking, SHA and
semantic records, approved path-key changes, active/historical references,
later-batch preservation, total YAML count, duplicate copies, provenance,
manual-review preservation, and group-meeting index protection.

Final result:

`CONFIG_BATCH_MIGRATION_AUDIT=PASS batch=1 moved=16`

## 19. Safe dry-run

- All 16 canonical YAML files were parsed and schema-checked.
- Six relevant CLI entrypoints passed `--help`.
- Original-MERC pipeline planning ran without `--execute` and generated eight
  smoke jobs, including all six moved canonical paper-aligned smoke configs.
- The paper-aligned trainer's native `--dry-run` was intentionally not used
  because it validates and loads real feature assets before model
  construction. No formal or smoke CLI training command was launched.
- `DRY_RUN_STATUS=PASS`.

## 20. Protected paths

- No model files changed.
- No training or evaluation algorithm changed.
- No tracked or production asset under `data/`, `outputs/`, `third_party/`,
  repository `tmp/`, checkpoint, cache, or historical output snapshots was
  edited. Pytest retained its existing ignored temporary-fixture behavior;
  those paths were not read as production evidence or manually altered.
- No formal training was started.

## 21. Group-meeting files

- `scripts/analyze/export_group_meeting_baseline_report.py`: untracked,
  untouched, not staged.
- `tests/analyze/test_group_meeting_baseline_report.py`: untracked,
  untouched, not staged.
- `GROUP_MEETING_FILES_CHANGED=NO`
- `GROUP_MEETING_FILES_STAGED=NO`

## 22. Later batches

Config Batch 2--7 YAML files remain at their planned old paths. The two
manual-review missing-modality rows and their collision remain unchanged.
Batch 2 has not started.

## 23. Persistent context

`AGENTS.md`, `docs/PROJECT_CONTEXT.md`, and `docs/project_map.md` were updated
incrementally to record:

- `CONFIG_LAYOUT_REFACTOR=IN_PROGRESS`
- `CONFIG_BATCH_1=COMPLETED`
- `CONFIG_BATCH_2..7=NOT_STARTED`
- Batch 1 canonical paths and invalid old paths
- no YAML wrappers or dual truth
- Batch 2 as the only next migration phase
- continued group-meeting file protection

## 24. Rollback boundary

Rollback is limited to Config Batch 1: reverse only the 16 recorded mappings,
restore only the active-reference edits listed in the reference audit, and
remove only the Batch 1 audit artifacts/tool/test introduced here. Do not
touch Batch 2--7 YAML, protected roots, script/model compatibility wrappers, or
the group-meeting files. Do not use `git clean` or `git reset --hard`.

## 25. Status

- `CONFIG_BATCH1_STATUS=PASS`
- `BATCH1_PLANNED_YAML=16`
- `BATCH1_YAML_MOVED=16`
- `BATCH1_OLD_YAML_REMAINING=0`
- `BATCH1_NEW_YAML_PRESENT=16`
- `BATCH1_YAML_CONTENT_CHANGED=0`
- `BATCH1_UNAPPROVED_SEMANTIC_CHANGES=0`
- `ACTIVE_REFERENCES_UPDATED=28`
- `ACTIVE_OLD_REFERENCES_REMAINING=0`
- `HISTORICAL_OLD_REFERENCES_RETAINED=20`
- `TRACKED_YAML_COUNT=183`
- `CONFIG_VALIDATION_STATUS=PASS`
- `SEMANTIC_AUDIT_STATUS=PASS`
- `DRY_RUN_STATUS=PASS`
- `MODEL_CODE_CHANGED=NO`
- `TRAINING_BEHAVIOR_CHANGED=NO`
- `FORMAL_TRAINING_STARTED=0`
- `GROUP_MEETING_FILES_CHANGED=NO`
- `GROUP_MEETING_FILES_STAGED=NO`
- `COMMIT_CREATED=NO`
- `PUSH_PERFORMED=NO`
- `READY_TO_COMMIT_CONFIG_BATCH1=YES`
- `READY_FOR_CONFIG_BATCH2=YES`
