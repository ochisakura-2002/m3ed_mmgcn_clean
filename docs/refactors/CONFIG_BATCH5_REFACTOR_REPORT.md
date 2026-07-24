# Config Batch 5 refactor report

## 1. Git precheck

Batch 5 started on branch `refactor/model-first-layout` at Git HEAD
`2403a801b48307f214f2fdd6768d206e40c453e9`. The branch was synchronized
with `origin/refactor/model-first-layout` (`0/0`). The tracked worktree and
index were clean; the only untracked files were the two protected local
group-meeting files. No commit or push was performed.

## 2. Baseline gates

Before any move, config validation, the strict Phase 4A migration-plan audit,
`git diff --check`, and both migration-audit test modules passed. The two
audit modules reported `62 passed` at the pre-move baseline. The
repository-wide tracked
baseline passed `403 passed, 3 skipped, 14 warnings`. The protected untracked
group-meeting test was explicitly excluded.

## 3. Exact Batch 5 inventory

`docs/refactors/CONFIG_MIGRATION_PLAN_PHASE4A.csv` contains exactly 13 rows
with `migration_batch=Batch 5`. All 13 are GS-MCC, `project_variant`, and
IEMOCAP configs. All have `manual_review=NO` and
`collision_status=CLEAR`; none belongs to Batch 6 or Batch 7.

- Causal context: 7
- Full context: 6
- Planned/moved: 13 / 13
- Unique old/new paths: 13 / 13
- Author-official configs: 0

The exact row-by-row mapping is recorded in
`docs/refactors/CONFIG_BATCH5_MOVES.csv`.

## 4. Preview

The read-only pre-move preview passed and left the working tree unchanged.

- `PREVIEW_PATH_CHANGES_REQUIRED=0`
- `ACTUAL_FILE_CHANGES=0`
- `ACTUAL_YAML_CONTENT_MODIFICATIONS=0`
- `ACTUAL_UNAPPROVED_SEMANTIC_CHANGES=0`
- Candidate collisions in Batch 5: 0
- Manual-review rows in Batch 5: 0

## 5. Before snapshot

`docs/refactors/CONFIG_BATCH5_BEFORE_SNAPSHOT.json` was generated while all
13 old paths still existed and all 13 new paths were absent. It records
`snapshot_source=working_tree_pre_move`, the immutable starting Git HEAD,
raw SHA256, recursive parsed YAML, model and implementation identity,
dataset, feature lineage, context, split, training, graph, loss, modality,
checkpoint, output, and reference fields. All 13 YAML files parsed
successfully.

## 6. GS-MCC provenance

Provenance was determined from YAML content, registry consumers, entrypoints,
and runtime behavior rather than words such as `original` or `official` in
old filenames. All 13 configs remain `project_variant`. The seven causal
configs use `causal_gsmcc_inspired`; the six full-context configs use
`project_paper_oriented_gsmcc`, whose registry provenance is
`PROJECT_VARIANT_NOT_PAPER_REPRODUCTION` with
`paper_reproduction_eligible=false`.

`OFFICIAL_GSMCC_REPRODUCTION=NOT_STARTED`.

## 7. Causal-context configs

The seven causal configs now live under
`configs/gsmcc/project_variant/iemocap/causal_context/`:

- Legacy MMGCN features: `val_official_prefix`, `val_ses01` through
  `val_ses04`, and `smoke_real_2epoch`.
- Clean RoBERTa features: `smoke_real_2epoch`.

They preserve the causal graph contract, model key, feature identity,
validation split, graph window, fusion, modality, optimizer, output, and
smoke/formal purpose.

## 8. Full-context configs

The six full-context configs now live under
`configs/gsmcc/project_variant/iemocap/full_context/`:

- Legacy MMGCN features: `screening`, `fivefold_base`, and `smoke`.
- Clean RoBERTa features: `gsmcc_clean`, `fivefold_base`, and `smoke`.

They preserve the project-variant registry key, noncausal offline
full-context grade, Original-MERC screening/fivefold/smoke roles, feature
identity, loss, modality, optimizer, and output fields.

## 9. Git move inventory

All 13 plan rows were executed with `git mv`. No YAML was copied, merged,
wrapped, redirected, or symlinked. All 13 old paths are absent, all 13
canonical paths are present and tracked, and the tracked YAML total remains
183. No Batch 6 or Batch 7 YAML was moved.

## 10. YAML content changes

All 13 migrated YAML files are byte-identical and semantically identical.
No model, implementation, dataset, feature asset, split, session, seed,
epoch, batch size, optimizer, learning rate, dropout, graph, fusion,
causal/context, loss, modality, checkpoint, run-name, experiment-ID, or
output-schema field changed. There are no approved changed keys because no
YAML content update was required.

## 11. Pipeline and YAML reference updates

Fourteen active YAML-to-YAML references were updated. Five are
`configs/pipeline/gsmcc/iemocap/causal_benchmark/` training-config
references. The pipeline YAML files themselves were not moved; only their
internal Batch 5 train-config paths changed. The Original-MERC pipeline
manifest and two analysis/audit YAML consumers also resolve to canonical
paths.

## 12. Active references

`docs/refactors/CONFIG_BATCH5_REFERENCE_AUDIT.csv` records 46 updated active
references across YAML, scripts, tests, documentation, validator tables,
analysis scope classification, and benchmark launchers. It includes 26 exact
static occurrences and 20 dynamic-path consumer entries.

- Active references updated: 46
- Pipeline references updated: 5
- Active old references remaining: 0

## 13. Historical references

No historical old-path occurrence was retained for Batch 5. Three
references that the older frozen Phase 4A graph classified as historical
were active audit/test consumers in the current tree and were updated. The
reference audit records `HISTORICAL_OLD_REFERENCES_RETAINED=0`.

## 14. Entrypoint audit

All 13 configs retain their intended canonical consumers:

- Causal training: `scripts/workflows/causal_graph/train.py`
- Full-context project-variant training:
  `scripts/workflows/paper_aligned/train.py`
- Generic pipeline: `scripts/workflows/run_pipeline.py`
- Unified checkpoint evaluation:
  `scripts/evaluation/unified_checkpoint.py`

CLI help passed for all four entrypoints. Command generation did not execute
training.

## 15. Registry audit

All seven causal configs resolve `causal_gsmcc_inspired` through
`models/registry/causal.py`. All six full-context configs resolve
`project_paper_oriented_gsmcc` through
`models/registry/paper_aligned.py`. The latter explicitly remains
`PROJECT_VARIANT_NOT_PAPER_REPRODUCTION`. No registry or model source file
changed.

## 16. Graph and fusion audit

Before/after snapshot comparison found no change to graph windows, graph
layers, graph parameters, fusion type, modality topology, filter steps, or
speaker settings. The Batch 5 auditor has negative tests that reject graph,
fusion, and modality drift.

## 17. Modality and missing-modality audit

Before/after comparison found no change to text/audio/visual feature
dimensions, active modality sets, modality encoder settings, or
missing-modality controls. Batch 5 contains no candidate-collision
missing-modality pipeline move, and the auditor rejects modality or
missing-modality drift.

## 18. Context and causal audit

The seven causal configs remain `causal_context` with
`graph.context_mode=causal` and `window_future=0`. The six full-context
configs remain `full_context` with
`causal_grade=noncausal_offline_full_context`. Negative tests reject
causal/full-context placement or contract drift.

## 19. Old-path audit

The strict Batch 5 audit and an independent fixed-string rescan found:

- Active old references remaining: 0
- Historical old references retained: 0
- Old YAML files remaining: 0
- New YAML files present: 13

The scan covered active configs, scripts, tests, documentation, `AGENTS.md`,
and `README.md`, while excluding generated output/data/vendor/temp trees,
Batch audit artifacts that intentionally record mappings, and the two
protected group-meeting files.

## 20. Semantic diff

`docs/refactors/CONFIG_BATCH5_AFTER_SNAPSHOT.json` and
`docs/refactors/CONFIG_BATCH5_SEMANTIC_DIFF.csv` contain 13 passing rows:

- Byte-identical: 13
- Semantic-identical: 13
- YAML content changes: 0
- Unapproved semantic changes: 0

## 21. Targeted tests

The final migration-audit test modules passed `79 passed`. The GS-MCC, causal,
Original-MERC, pipeline, registry, checkpoint, launcher, migration-plan, and
batch-audit regression set passed `219 passed, 3 skipped`. The Batch 5
extension adds positive lineage/context placement coverage and negative
model, provenance, author-official, graph, fusion, modality, causal,
full-context, prior-batch, and later-batch checks.

## 22. Full pytest

The final repository-wide tracked gate in the existing `m3ed_mmgcn` Conda
environment passed `420 passed, 3 skipped, 14 warnings`. The protected
untracked group-meeting test was explicitly excluded. The warnings are
dependency deprecations and do not indicate migration failures.

## 23. Config validation

`scripts/dev/validate_config_tree.py` passed after its GS-MCC dynamic path
tables were updated to the canonical Batch 5 paths. The strict Phase 4A plan
audit also passed with `tracked_yaml=183` and the one known Batch 7 candidate
collision group.

## 24. Batch 1--4 regression

Strict Batch 1, Batch 2, Batch 3, and Batch 4 audits passed with all prior
YAML moves, approved Batch 2/3 `source_config` changes, reference audits, and
semantic boundaries intact.

## 25. Batch 5 audit

The strict Batch 5 audit passed with 13 moved configs, the required 7/6
context split, zero YAML content changes, 46 updated active references, five
pipeline references, zero historical old references, zero active old
references, zero unapproved semantic changes, and 183 tracked YAML files.

## 26. Safe dry-run

The no-training dry-run parsed all 13 YAML files, generated 13 commands,
resolved all 13 registry keys, checked graph/fusion/context/modality fields,
parsed 13 output specs and five checkpoint specs, resolved all five pipeline
references, and resolved all seven GS-MCC Original-MERC manifest references.
It did not load feature assets, initialize a formal run, create outputs,
execute an epoch, or start training.

## 27. Protected paths

The two local group-meeting files were not read as production evidence,
modified, moved, deleted, staged, or uploaded. No changes were made under
`models/`, `data/`, `outputs/`, `third_party/`, or `tmp/`. Model code and
training behavior were not changed.

## 28. Batch 6 not started

`CONFIG_BATCH_6=NOT_STARTED`. No Batch 6 or Batch 7 YAML was moved. The
remaining candidate collision group is still reserved for Batch 7
resolution. Official MultiDAG+CL and GS-MCC reproduction work remains not
started, and no final baseline has been selected.

## 29. Rollback boundary

The reviewable rollback boundary is the 13 `git mv` pairs, 46 active
reference updates, Batch 5 audit artifacts, validator/auditor/tests, and
persistent-context updates in this worktree. There is no commit to revert
and no remote state to roll back. Reverting must not touch prior Batch 1--4
artifacts, Batch 6--7 YAML, model code, formal outputs, data assets, or the
protected group-meeting files.
