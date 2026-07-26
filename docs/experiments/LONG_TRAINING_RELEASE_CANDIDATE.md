# IEMOCAP Clean long-training release candidate

Status: `CURRENT_REMOTE_OLD_LAYOUT_BATCH_RUNNING; FUTURE_LAYOUT_READY`

This package prepares configuration only. It does not start training, evaluate
checkpoints, create feature placeholders, weaken SHA256 validation, commit, or
push.

## Scope and counts

- Dataset: IEMOCAP.
- Feature track: frozen Clean RoBERTa v1.
- Experiment track: `clean_roberta_session_holdout_fair_comparison`.
- Validation split strategy: `session_holdout`.
- Protocol version: `long_training_session_holdout_v1`.
- Base model-context configs: 8.
- Primary seed: 42.
- Primary runs: 32.
- Optional three-seed runs: 96 (`enabled: false`, seeds 13/42/77).
- Primary full-context runs: 16.
- Primary causal-context runs: 16.
- Primary runs per model family: 8.

The primary matrix follows the repository's formal fixed-test Session protocol:
Ses01, Ses02, Ses03, and Ses04 are the four Validation Sessions, while official
Ses05 remains the Test Session. Every model expands both full-context and
causal-context over the same four Validation Sessions. The 32 primary runs
therefore form 16 exact context pair keys and the optional three-seed matrix
forms 48 pair keys. Each key contains exactly one full-context member and one
causal-context member.

The experiment track is intentionally distinct from
`clean_roberta_fivefold_fair_comparison`. The long-training matrix fixes Ses05
as Test and rotates Ses01--Ses04 as whole-session Validation holdouts; it does
not run the five outer Test folds represented by the older track.

The pair key is model family, the shared `unified_clean` protocol lineage,
feature set, Validation Session, Test Session, and seed. The separate
`implementation_lineage` and provenance fields remain unchanged and continue
to disclose that the full-context and causal-context runtime implementations
are not falsely claimed to be identical or author-official.

## Parameter sources

No test result was used to select a hyperparameter. Model, optimizer, learning
rate, batch size, dropout, weight decay, gradient clipping, scheduler, epoch
budget, early stopping, and model-specific parameters are inherited from the
listed canonical source. The long-training base owns the matrix-specific
experiment-track, split-strategy, Validation-Session, and fixed-Test-Session
metadata instead of inheriting the parameter source's screening protocol.

| Model/context | Parameter source | Epochs | Selection |
|---|---|---:|---|
| MMGCN full | `configs/mmgcn/paper_aligned/iemocap/full_context/clean_roberta_features/mmgcn_clean.yaml` | 20 | `val_weighted_f1` |
| MMGCN causal | `configs/mmgcn/unified/iemocap/causal_context/clean_roberta_features/val_ses01.yaml` | 30 | `val_weighted_f1` |
| MultiDAG+CL full | `configs/multidag_cl/paper_aligned/iemocap/full_context/clean_roberta_features/multidag_cl_clean.yaml` | 30 | `val_weighted_f1` |
| MultiDAG+CL causal | `configs/multidag_cl/unified/iemocap/causal_context/clean_roberta_features/val_ses01.yaml` | 30 | `val_weighted_f1` |
| DialogueGCN full | `configs/dialoguegcn/paper_aligned/iemocap/full_context/clean_roberta_features/dialoguegcn_clean.yaml` | 60 | `val_weighted_f1` |
| DialogueGCN causal | `configs/dialoguegcn/unified/iemocap/causal_context/legacy_mmgcn_features/val_ses01.yaml` | 30 | `val_weighted_f1` |
| GS-MCC Project Variant full | `configs/gsmcc/project_variant/iemocap/full_context/clean_roberta_features/gsmcc_clean.yaml` | 250 | `val_weighted_f1` |
| GS-MCC Project Variant causal | `configs/gsmcc/project_variant/iemocap/causal_context/legacy_mmgcn_features/val_ses01.yaml` | 30 | `val_weighted_f1` |

The repository has no non-smoke Clean causal Session YAML for DialogueGCN or
GS-MCC. Their long-training bases therefore preserve every training and model
parameter from the corresponding formal causal `val_ses01.yaml`, while only
the feature declaration and text dimensions are adapted from the already
validated Clean smoke config and `clean_roberta_v1` registry entry. This is a
feature-track adaptation, not test-driven tuning.

The 20260722 batch's ignored runtime `resolved_configs/` directory is not
present in this local checkout. The canonical launcher shows that its
resolution step changes only device, output root, experiment date, and the
causal pipeline's train-config pointer; therefore canonical formal source
configs are the parameter authority here.

## 20260725 failed launch and protocol repair

Remote launcher batch
`outputs/launcher_logs/formal_long32_20260725_213652` stopped on its first run,
`ltp_mmgcn_full_context_val_ses01_s42`, before any epoch started. The
full-context base still declared
`clean_roberta_fivefold_fair_comparison`/`outer_session_stratified`, while the
matrix changed only `dataset.val_split_strategy` to `session_holdout`.
`scripts/runtime/paper_aligned.py` correctly rejected that mixed protocol.

All eight bases now declare the shared
`clean_roberta_session_holdout_fair_comparison` track with
`session_holdout` and top-level
`protocol_version: long_training_session_holdout_v1`. The strict paper-aligned
track policy includes this exact mapping, and its dataloader forwards
`dataset.val_session_id`. The prepare command also rejects any base whose
protocol version, track, split strategy, or comparability metadata differs from
the matrix protocol.

The regression gate materializes 32 resolved configs in a temporary directory,
dispatches each one through the config-only validator used by its actual
entrypoint, and requires 32/32 passes before accepting the matrix. It also
requires 32 unique commands, 32 unique run IDs, 32 unique output roots, 16
context pair keys, and zero unpaired members or test-selection leakage.

## Output ownership

Future primary preparations use experiment group
`formal_long32_primary_seed42`; the disabled expansion uses
`formal_long32_multiseed`. A preparation freezes one `experiment_date` and
requires one unique `batch_id`. Run roots are
`outputs/<date>/<group>/runs/<run_id>`, launcher logs are under
`logs/launcher/<batch_id>`, and all resolved configs, commands, matrix
snapshots, commit provenance, and preparation metadata are under
`manifests/batches/<batch_id>`. Review, reports, and analysis have matching
batch roots. The exact policy is documented in
`docs/experiments/OUTPUT_DIRECTORY_LAYOUT.md`.

The formal 32-run remote batch reported as currently running uses the old
layout at commit `d8a7701` or the remote machine's actual HEAD. It is distinct
from the historical failed launch above and is not migrated, restarted,
stopped, or asked to pull this local change.

## Provenance and context boundaries

- `paper_aligned` does not mean `author_official`.
- Current MultiDAG+CL full-context remains a project paper-aligned
  implementation, not the complete author-official reproduction.
- Both GS-MCC implementations remain
  `PROJECT_VARIANT_NOT_PAPER_REPRODUCTION`.
- Full-context configs retain their noncausal future-context behavior.
- Causal configs require `graph.context_mode: causal` and
  `graph.window_future: 0`; full nodal attention and bidirectional recurrent
  paths remain forbidden by the causal registry.
- GS-MCC full-context retains `angular_similarity_eps: 1.0e-7`, contrastive
  settings, gradient clipping, AdamW settings, and runtime finite checks.
- GS-MCC causal retains zero consistency/complementarity auxiliary weights and
  the strict causal runtime contract.

## Selection and assets

Every base and matrix declares validation-only checkpoint selection by
`val_weighted_f1`; test selection leakage is zero. Only `best_model.pt`
selected on validation may be evaluated on test. The matrix checker also
rejects unpaired context members, duplicate pair members, Session mismatches,
selection leakage, and output collisions before printing any command.

The local Clean feature file is intentionally absent. Config-only review checks
the registry key, relative path, pinned SHA256, dimensions (768/1582/342), six
labels, YAML syntax, model registry, entrypoint resolution, matrix expansion,
and command generation. The allowed local status is
`FEATURE_REGISTRY_STATUS=REMOTE_ASSET_REQUIRED`. Real feature loading,
checksum verification against bytes, dataloader construction, and asset-level
dry-run remain `REMOTE_DRY_RUN_REQUIRED`.

## Release boundary

The current old-layout remote batch continues untouched. After it is complete,
future batches may regenerate a new canonical manifest through normal Git
handoff; this document does not authorize changing the running workspace. The
optional three-seed matrix remains disabled, and
`LONG_TRAINING_COMMANDS.sh` only checks/materializes configs and prints
commands; it never executes training.
