# Config Batch 4 refactor report

## 1. Git precheck

Batch 4 started on branch `refactor/model-first-layout` at Git HEAD
`0e1a62a84fb464632990552d8cb2cfd32ea5b055`. The branch was synchronized
with `origin/refactor/model-first-layout` (`0/0`). The tracked worktree and
index were clean; the only untracked files were the two protected local
group-meeting files. No commit or push was performed.

## 2. Baseline gates

Before any move, config validation, the strict Phase 4A migration-plan audit,
`git diff --check`, and both audit test modules passed. The batch-audit tests
reported `40 passed` at the pre-move baseline; the plan-audit tests reported
`10 passed`. The first sandboxed batch-audit run could not create its Windows
pytest temporary directory, so the same command was rerun with normal local
filesystem permissions without changing repository state.

## 3. Exact Batch 4 inventory

`docs/refactors/CONFIG_MIGRATION_PLAN_PHASE4A.csv` contains exactly 10 rows
with `migration_batch=Batch 4`. All 10 are DialogueGCN and IEMOCAP configs;
all have `manual_review=NO` and `collision_status=CLEAR`. None is a pipeline
YAML, M3ED config, collision row, or `author_official` config.

- Unified: 6
- Paper-aligned: 4
- Planned/moved: 10 / 10
- Unique old/new paths: 10 / 10

The exact row-by-row mapping is recorded in
`docs/refactors/CONFIG_BATCH4_MOVES.csv`.

## 4. Preview

The read-only pre-move preview passed and left the working tree unchanged.

- `PREVIEW_PATH_CHANGES_REQUIRED=0`
- `ACTUAL_FILE_CHANGES=0`
- `ACTUAL_YAML_CONTENT_MODIFICATIONS=0`
- `ACTUAL_UNAPPROVED_SEMANTIC_CHANGES=0`
- Candidate collisions in Batch 4: 0
- Manual-review rows in Batch 4: 0

## 5. Before snapshot

`docs/refactors/CONFIG_BATCH4_BEFORE_SNAPSHOT.json` was generated while all
10 old paths still existed and all 10 new paths were absent. It records
`snapshot_source=working_tree_pre_move`, the immutable starting Git HEAD, raw
SHA256, recursive parsed YAML, top-level keys, model/implementation/dataset/
context identity, feature and split fields, sessions, seeds, optimization,
graph/relation/speaker/edge, loss/modality, checkpoint/output, and config/
script reference fields. All 10 YAML files parsed successfully.

## 6. DialogueGCN provenance

Provenance was determined from YAML contents, registry consumers, entrypoints,
and runtime behavior—not from words such as `original` or `official` in an
old filename. Six configs retain the unified `causal_dialoguegcn` lineage;
four retain the project `paper_aligned` `original_repro_dialoguegcn` lineage.
`paper_aligned` does not mean `author_official`, and this batch is not an
author-official DialogueGCN reproduction.

## 7. DialogueGCN unified

The six unified YAML files now live under
`configs/dialoguegcn/unified/iemocap/causal_context/`:

- Five legacy-MMGCN-feature causal benchmark configs:
  `val_official_prefix` and `val_ses01`--`val_ses04`.
- One clean-RoBERTa-feature `smoke_real_2epoch` config.

They preserve model keys, IEMOCAP data identity, feature lineage, session
splits, causal context, graph window, relation/speaker settings, modality,
optimizer, checkpoint, output, and smoke/formal purpose.

## 8. DialogueGCN paper-aligned

The four paper-aligned YAML files now live under
`configs/dialoguegcn/paper_aligned/iemocap/full_context/`:

- Legacy MMGCN features: `screening` and `fivefold_base`.
- Clean RoBERTa features: `dialoguegcn_clean` and `fivefold_base`.

They preserve the `original_repro_dialoguegcn` registry key and their
Original-MERC screening/fivefold-base roles. They remain project
paper-aligned configs, not author-official configs.

## 9. Git move inventory

All 10 plan rows were executed with `git mv`. No YAML was copied, merged,
deleted, wrapped, redirected, or symlinked. All 10 old paths are absent, all
10 canonical paths are present and tracked, and the tracked YAML total
remains 183. No Batch 5--7 YAML was moved.

## 10. YAML content changes

All 10 migrated YAML files are byte-identical and semantically identical.
No model, implementation, dataset, feature asset, split, session, seed,
epoch, batch size, optimizer, learning rate, dropout, graph, relation,
speaker, edge, causal/context, loss, modality, checkpoint, run-name,
experiment-ID, or output-schema field changed. There are no approved changed
keys because no YAML content update was required.

## 11. Pipeline and YAML reference updates

Twelve active YAML-to-YAML references were updated. Five are
`configs/pipeline/dialoguegcn/iemocap/causal_benchmark/` training-config
references. The pipeline YAML files themselves were not moved; only their
internal Batch 4 train-config paths changed. The Original-MERC pipeline
manifest and two analysis/audit YAML consumers also resolve to canonical
paths.

## 12. Active references

`docs/refactors/CONFIG_BATCH4_REFERENCE_AUDIT.csv` records 30 updated active
references across YAML, scripts, tests, documentation, validator tables,
and the formal benchmark launcher. It includes 18 exact static occurrences
and 12 dynamic-path consumer entries that cannot be represented completely
by the frozen Phase 4A static reference graph.

- Active references updated: 30
- Pipeline references updated: 5
- Active old references remaining: 0

## 13. Historical references

No historical old-path occurrence was retained for Batch 4. Three references
that the older frozen graph classified as historical were active audit/test
consumers in the current tree and were therefore updated. The reference audit
records `HISTORICAL_OLD_REFERENCES_RETAINED=0`.

## 14. Entrypoint audit

All 10 configs retain their intended canonical consumers:

- Unified causal training: `scripts/workflows/causal_graph/train.py`
- Paper-aligned training: `scripts/workflows/paper_aligned/train.py`
- Generic pipeline: `scripts/workflows/run_pipeline.py`
- Unified checkpoint evaluation: `scripts/evaluation/unified_checkpoint.py`

CLI help passed for all four entrypoints. The command-generation dry-run did
not execute training.

## 15. Registry audit

All six unified configs resolve `causal_dialoguegcn` through
`models/registry/causal.py`. All four paper-aligned configs resolve
`original_repro_dialoguegcn` through `models/registry/paper_aligned.py`.
No registry or model source file changed.

## 16. Graph, relation, and speaker audit

Before/after snapshot comparison found no change to graph windows, graph
layers, graph parameters, relation types, speaker modeling, edge settings,
or modality topology. The Batch 4 auditor has negative tests that reject
graph-window, relation, and speaker-setting drift.

## 17. Context and causal audit

The six unified configs remain `causal_context`; the four paper-aligned
configs remain `full_context`. No causal flag, context window, session split,
or future-information boundary changed. Negative tests reject causal/context
drift.

## 18. Old-path audit

The strict Batch 4 audit and an independent fixed-string rescan found:

- Active old references remaining: 0
- Historical old references retained: 0
- Old YAML files remaining: 0
- New YAML files present: 10

The scan covered active configs, scripts, tests, documentation, `AGENTS.md`,
and `README.md`, while excluding generated outputs/data/vendor/temp trees,
Batch audit artifacts that intentionally record mappings, and the two
protected group-meeting files.

## 19. Semantic diff

`docs/refactors/CONFIG_BATCH4_AFTER_SNAPSHOT.json` and
`docs/refactors/CONFIG_BATCH4_SEMANTIC_DIFF.csv` contain 10 passing rows:

- Byte-identical: 10
- Semantic-identical: 10
- YAML content changes: 0
- Unapproved semantic changes: 0

## 20. Targeted tests

The final audit/plan/benchmark focused suite passed `64 passed`. The
16-file DialogueGCN/config/pipeline/registry/checkpoint-related suite passed
`181 passed, 3 skipped`. The Batch 4 extension expanded the batch-audit
module from 40 to 52 tests, including positive placement/lineage coverage and
negative model, provenance, author-official, graph, relation, speaker,
causal, context, prior-batch, and later-batch checks.

## 21. Full pytest

The final repository-wide tracked gate in the existing `m3ed_mmgcn` Conda
environment passed: `403 passed, 3 skipped, 14 warnings`. The protected
untracked group-meeting test was explicitly excluded. The warnings are
dependency deprecations and do not indicate migration failures.

## 22. Config validation

`scripts/dev/validate_config_tree.py` passed after its DialogueGCN dynamic
path tables were updated to the canonical Batch 4 paths. The strict Phase 4A
plan audit also passed with `tracked_yaml=183` and the one known Batch 7
candidate collision group.

## 23. Batch 1--3 regression

Strict Batch 1, Batch 2, and Batch 3 audits passed with all prior YAML moves,
approved Batch 2/3 `source_config` changes, reference audits, and semantic
boundaries intact. The reusable auditor now validates prior snapshots against
their immutable recorded starting HEAD instead of assuming the current HEAD
is the earlier batch's starting commit.

## 24. Batch 4 audit

The strict Batch 4 audit passed with 10 moved configs, the required 6/4
lineage split, zero YAML content changes, 30 updated active references, five
pipeline references, zero historical old references, zero active old
references, zero unapproved semantic changes, and 183 tracked YAML files.

## 25. Safe dry-run

The no-training dry-run parsed all 10 YAML files, generated 10 commands,
resolved all 10 registry keys, checked graph/relation/speaker settings,
parsed output/checkpoint fields, and resolved all five pipeline references.
It did not load full feature assets, initialize a formal run, create outputs,
execute an epoch, or start training.

## 26. Protected paths

The two local group-meeting files were not read as production evidence,
modified, moved, deleted, staged, or uploaded. No changes were made under
`models/`, `data/`, `outputs/`, `third_party/`, or `tmp/`. Model code and
training behavior were not changed.

## 27. Batch 5 not started

`CONFIG_BATCH_5=NOT_STARTED`. No Batch 5--7 YAML was moved. The remaining
candidate collision group is still reserved for Batch 7 resolution. Official
MultiDAG+CL and GS-MCC reproduction work remains not started, and no final
baseline has been selected.

## 28. Rollback boundary

The reviewable rollback boundary is the 10 `git mv` pairs, 30 active
reference updates, Batch 4 audit artifacts, validator/auditor/tests, and
persistent-context updates in this worktree. There is no commit to revert
and no remote state to roll back. Reverting must not touch prior Batch 1--3
artifacts, Batch 5--7 YAML, model code, formal outputs, data assets, or the
protected group-meeting files.
