# Config Batch 7 Refactor Report

## Scope

Batch 7 migrated all 73 planned YAML files on branch
`refactor/model-first-layout` from Git HEAD
`10a31f79f9136d2109f8f3309364f1ebe65afe77`. No training, commit, or push was
performed.

## Collision resolution

The two missing-modality pipelines were retained as separate configurations.

- `missing_eval_from_context_w5_tav.yaml` is tied to the ordinary
  `context_w5_tav` source pipeline and source training config.
- `missing_eval_from_stable_context_w5_tav.yaml` is tied to
  `context_w5_tav_stable_candidate`.

Their source training configs differ in the stable profile, including dropout,
graph depth, and learning rate. The Phase 4A plan and classification rows now
assign two unique canonical paths under
`configs/benchmarks/ablations/missing_modality/pipelines/multidag_cl/legacy_mmgcn_features/`.
The migration plan has zero manual-review rows and zero candidate collisions.

## Migration result

- Planned YAML: 73
- YAML moved with `git mv`: 73
- Old YAML remaining: 0
- New YAML present: 73
- Tracked YAML total: 183
- Byte/content unchanged YAML: 60
- YAML with required repository-internal config-path rewrites: 13
- Unapproved semantic changes: 0

No old YAML wrappers, redirects, symlinks, or duplicate configuration truths
were introduced.

## References

The reference audit contains 101 updated active references and 204 retained
frozen historical references. Exact references and dynamic directory/template
consumers were updated. Active old references remaining: 0.

Two previously active Batch 2 MMGCN references discovered by the Batch 1--7
regression audit were also updated to their existing canonical target and
recorded in the Batch 7 reference audit.

## Artifacts

- `CONFIG_BATCH7_BEFORE_SNAPSHOT.json`
- `CONFIG_BATCH7_AFTER_SNAPSHOT.json`
- `CONFIG_BATCH7_MOVES.csv`
- `CONFIG_BATCH7_SEMANTIC_DIFF.csv`
- `CONFIG_BATCH7_REFERENCE_AUDIT.csv`
- `CONFIG_BATCH7_COLLISION_REVIEW.csv`

## Gates

- Config tree validation: pass
- Phase 4A strict migration-plan audit: pass
- Batch 1--7 strict migration audits: pass
- Targeted migration, pipeline, launcher, and analysis tests:
  `150 passed`
- Full tracked pytest, explicitly ignoring the protected local
  `tests/analyze` directory: `436 passed, 3 skipped, 14 warnings`
- Safe 73-config dry-run: pass
  - 73 YAML files parsed
  - 73 commands generated
  - 85 explicit config references resolved
  - 6 canonical entrypoint help paths passed
  - training started: 0
- Git whitespace checks: pass
- Protected group-meeting files changed: no
- Protected group-meeting files staged: no

## Status

`CONFIG_BATCH_7=COMPLETED`

`CONFIG_LAYOUT_REFACTOR=COMPLETED`

Author-official MultiDAG+CL and GS-MCC reproductions remain not started, and no
final paper baseline has been selected.
