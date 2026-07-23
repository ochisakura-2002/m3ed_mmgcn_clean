# Config Batch 3 refactor report

## 1. Git precheck

Batch 3 started on branch `refactor/model-first-layout` at Git HEAD
`753eae191558c4ccad5c69e24a8b960c20691784`. The branch was synchronized
with `origin/refactor/model-first-layout` (`0/0`). The tracked worktree was
clean; the only untracked files were the two protected local group-meeting
files. No commit or push was performed.

## 2. Baseline gates

Before any move, config validation, the strict Phase 4A migration-plan audit,
`git diff --check`, and both audit test modules passed. The batch-audit tests
reported `26 passed` at the pre-move baseline; the plan-audit tests reported
`10 passed`. The first sandboxed pytest attempt could not create its Windows
temporary directory, so the same command was rerun outside that filesystem
restriction without changing repository state.

## 3. Exact Batch 3 inventory

`docs/refactors/CONFIG_MIGRATION_PLAN_PHASE4A.csv` contains exactly 17 rows
with `migration_batch=Batch 3`. All 17 are `multidag_cl`, all have
`manual_review=NO` and `collision_status=CLEAR`, and none is a pipeline,
manual-review collision, M3ED config, or `author_official` config.

- Unified: 13
- Paper-aligned: 4
- IEMOCAP: 17
- Planned/moved: 17 / 17

The exact row-by-row mapping is recorded in
`docs/refactors/CONFIG_BATCH3_MOVES.csv`.

## 4. Preview

The read-only pre-move preview passed and left the working tree unchanged.

- `PREVIEW_PATH_CHANGES_REQUIRED=4`
- `ACTUAL_YAML_CONTENT_MODIFICATIONS=0`
- `ACTUAL_UNAPPROVED_SEMANTIC_CHANGES=0`
- Candidate collisions in Batch 3: 0
- Manual-review rows in Batch 3: 0
- Expected changed key: `source_config`

## 5. Before snapshot

`docs/refactors/CONFIG_BATCH3_BEFORE_SNAPSHOT.json` was generated while all
17 old paths still existed and all 17 new paths were absent. It records
`snapshot_source=working_tree_pre_move`, raw SHA256, recursive parsed YAML,
top-level keys, model/implementation/dataset/context identity, feature and
split fields, sessions, seeds, optimization fields, curriculum, graph,
modality, checkpoint/output fields, and config/script references. All 17 YAML
files parsed successfully.

## 6. MultiDAG unified

The 13 unified YAML files now live under
`configs/multidag_cl/unified/iemocap/causal_context/`. They preserve the
`MultiDAGCL` consumer key, causal/full-past or windowed context settings,
legacy MMGCN versus clean RoBERTa feature lineage, split rules, modality
settings, and smoke/formal purpose.

## 7. MultiDAG paper-aligned

The four paper-aligned YAML files now live under
`configs/multidag_cl/paper_aligned/iemocap/full_context/`. They preserve the
`original_repro_multidag_cl` registry key and their legacy/clean screening and
fivefold-base roles.

## 8. Provenance constraints

`paper_aligned` remains a project lineage label and does not mean
`author_official`. The current project MultiDAG implementation is not an
author-official complete reproduction. No filename containing words such as
`original`, `official`, or `paper` was used as provenance evidence, and no
config was reclassified between unified and paper-aligned.

## 9. Git move inventory

All 17 plan rows were executed with `git mv`. The mapping consists of:

- Five unified legacy causal-benchmark YAML files.
- Five unified clean-RoBERTa YAML files.
- Three unified legacy debug YAML files.
- Four paper-aligned Original-MERC YAML files.

No YAML was copied, merged, deleted, wrapped, redirected, or symlinked.
All 17 old paths are absent, all 17 canonical paths are present, and the
tracked YAML total remains 183.

## 10. YAML content changes

Thirteen migrated YAML files are byte-identical and semantically identical.
Four clean-RoBERTa session YAML files changed only
`provenance.source_config`, from their invalidated Batch 3 old path to the
exact same-batch canonical session path. No model, dataset, feature asset,
split, session, seed, optimizer, curriculum, graph, context, modality,
checkpoint, run-name, experiment-ID, or output-schema field changed.

## 11. Pipeline and YAML reference updates

Twenty-seven active YAML-to-YAML references were updated, including 12
pipeline training-config references. Batch 7 pipeline YAML files were not
moved; only their internal Batch 3 config references changed. Manifest,
benchmark analysis, experiment matrix, and the four migrated
`source_config` references now resolve to canonical paths.

## 12. Active references

`docs/refactors/CONFIG_BATCH3_REFERENCE_AUDIT.csv` records 52 updated active
references:

- Config/YAML consumers: 27
- Script consumers: 17
- Test consumers: 8
- Reference types: 27 `yaml_reference`, 20 `dynamic_path_consumer`,
  4 `test_fixture`, and 1 `python_string`

The dynamic-path rows cover command construction in the benchmark launcher,
config-tree validator mappings, and test matrices that the original static
reference graph could not express.

## 13. Historical references

Sixteen old-path occurrences remain in four historical documents under
`docs/baselines/`. Every row is recorded with
`historical_reference=YES`, `requires_update=NO`, `updated=NO`, and
`remaining_reference_allowed=YES`. No historical occurrence is used by an
active consumer.

## 14. Entrypoint audit

All 17 configs retain their intended canonical consumer:

- Unified training: `scripts/models/multidag_cl/unified/train.py`
- Paper-aligned training: `scripts/workflows/paper_aligned/train.py`

CLI help also passed for unified training, evaluation and smoke, the
paper-aligned trainer, the generic pipeline runner, and unified checkpoint
evaluation.

## 15. Registry audit

All 13 unified configs resolve the `MultiDAGCL` consumer key. All four
paper-aligned configs resolve `original_repro_multidag_cl` through
`models.registry.paper_aligned.MODEL_REGISTRY`. No registry or model code
changed.

## 16. Curriculum audit

Before/after snapshot comparison found no change to curriculum-learning
enablement, schedule, pacing, or related parameters. The auditor has negative
tests that reject curriculum drift.

## 17. Graph and context audit

Before/after snapshot comparison found no change to causal flags, context
mode, context windows, graph layers, graph parameters, or modality settings.
The auditor has negative tests for causal, context-window, and graph drift.

## 18. Old-path audit

The strict Batch 3 audit and an independent fixed-string rescan found:

- Active old references remaining: 0
- Historical old references retained: 16
- Old YAML files remaining: 0
- New YAML files present: 17

The scan covered `configs/`, `scripts/`, `tests/`, active docs, `AGENTS.md`,
and `README.md`, while excluding generated outputs/data/vendor/temp trees and
the two protected group-meeting files.

## 19. Semantic diff

`docs/refactors/CONFIG_BATCH3_AFTER_SNAPSHOT.json` and
`docs/refactors/CONFIG_BATCH3_SEMANTIC_DIFF.csv` contain 17 passing rows:

- Byte- and semantic-identical: 13
- Approved `source_config`-only changes: 4
- Unapproved semantic changes: 0

## 20. Targeted tests

The 24-file MultiDAG/config/pipeline/registry/checkpoint-related suite passed:
`369 passed, 3 skipped`. The final batch-audit module passed
`40 passed`; the migration-plan module passed `10 passed`.

## 21. Full pytest

The repository-wide gate passed: `399 passed, 3 skipped`. The warnings are
dependency deprecations and do not indicate migration failures.

## 22. Config validation

`scripts/dev/validate_config_tree.py` passed after its MultiDAG dynamic path
tables were updated to the canonical Batch 3 paths. The strict Phase 4A plan
audit also passed with `tracked_yaml=183` and the one known Batch 7 candidate
collision group.

## 23. Batch 1 regression

The strict Batch 1 audit passed with all 16 migrated YAML files intact. The
auditor now permits only later-batch active reference rewrites that are
explicitly recorded in a later reference-audit artifact; unrelated semantic
drift still fails. Positive and negative tests cover this boundary.

## 24. Batch 2 regression

The strict Batch 2 audit passed with all 17 MMGCN YAML files intact, including
its four previously approved `source_config` changes.

## 25. Batch 3 audit

The strict Batch 3 audit passed with 17 moved configs, the required 13/4
lineage split, four approved path-only changes, 52 updated active references,
16 marked historical references, zero active old references, zero unapproved
semantic changes, and 183 tracked YAML files.

## 26. Safe dry-run

The no-training dry-run parsed all 17 YAML files, checked their relative
config references, generated 17 commands, resolved unified and paper-aligned
model keys, parsed feature dimensions/class counts/output/checkpoint fields,
and ran six CLI help paths. It did not load full feature assets, initialize a
formal run, create outputs, or execute an epoch.

## 27. Protected paths

The two local group-meeting files were not modified, moved, deleted, staged,
or used as production-config evidence. No changes were made under `models/`,
`data/`, `outputs/`, `third_party/`, or `tmp/`.

## 28. Batch 4 not started

`CONFIG_BATCH_4=NOT_STARTED`. No Batch 4--7 YAML was moved, and the remaining
candidate collision group is still reserved for Batch 7 resolution. Official
MultiDAG and GS-MCC reproduction work remains not started.

## 29. Rollback boundary

The reviewable rollback boundary is the 17 `git mv` pairs, the four approved
`source_config` rewrites, active reference updates, Batch 3 audit artifacts,
auditor/validator tests, and persistent-context updates in this worktree.
There is no commit to revert and no remote state to roll back. Reverting
should not touch prior Batch 1/2 artifacts, later-batch YAML, model code,
formal outputs, data assets, or the protected group-meeting files.
