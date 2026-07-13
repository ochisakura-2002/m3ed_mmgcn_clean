# New causal graph baselines — Task 2 report

Date: 2026-07-13

## Status boundary

This task integrates **Project causal GS-MCC-inspired** and **Project causal
DialogueGCN** into the project benchmark code. It does not claim an exact
official reproduction. Local checks use deterministic synthetic dialogues;
there is no local official IEMOCAP PKL, so all real-batch and performance
claims remain **UNCONFIRMED**.

## Required answers

1. **Formal train/evaluate entries:** both new models use
   `scripts/baselines/train_new_causal_graph_baseline.py` and
   `scripts/baselines/evaluate_new_causal_graph_checkpoint.py`. A read-only
   real-batch entry is available at
   `scripts/baselines/debug_new_causal_graph_real_batch.py`.
2. **Synthetic end-to-end training:** yes. Both models completed two epochs,
   validation, validation Weighted-F1 best selection, best/last checkpoint
   writing, best reload, and final validation/Test artifact writing.
   Synthetic metric values are diagnostics, not performance results.
3. **Checkpoint reload:** passed for both command-line smoke runs and the
   automated exact-logit reload test.
4. **Session-Holdout:** yes. Each model has `Ses01` through `Ses04` validation
   folds while official `Ses05` remains Test. `official_prefix` keeps its
   existing prefix-split semantics.
5. **Training YAML count:** 10 formal YAMLs: five GS-MCC-inspired and five
   DialogueGCN. Four additional smoke YAMLs cover synthetic end-to-end and
   remote two-epoch real-data checks.
6. **Pipeline YAML count:** 10, five per model.
7. **Pipeline support:** yes. Exact registry keys route both canonical model
   names to the shared new train/evaluate entries. The latest run's real
   `run_id` and `best_model.pt` are used.
8. **Four-model causal audit:** yes. The existing audit now supports all four
   models, multiple cutoffs, three future perturbations, joint T/A/V future
   randomization, history-squared-logit gradients, prefix/full checks, and
   future/padding/cross-dialogue edge checks. Model-specific low/high and
   context/graph diagnostics are also recorded.
9. **MMGCN/MultiDAG regression:** the focused old/new causal suite passed. The
   final full-suite count is recorded in the verification section below.
10. **New dependencies:** none.
11. **Remote real-batch audit:** the code path and protocol are complete, but
    execution against the real PKL is **UNCONFIRMED**.
12. **Two-epoch real-data smoke:** dedicated `outputs/dev` configurations are
    present for both models; actual remote execution is **UNCONFIRMED**.
13. **Formal 30-epoch four-fold execution:** code, four Session-Holdout YAMLs,
    and pipeline YAMLs are present. Formal execution must remain gated on the
    PKL hash, config validation, real-batch dry-runs, unified real-batch audit,
    and two-epoch real-data smoke. It was not run locally.
14. **Remaining UNCONFIRMED items:** real PKL loading on the remote checkout;
    real feature dimensions/content; upstream feature-extractor online
    causality; real-batch strict audit; two-epoch real-data stability; formal
    optimization/performance; official GS-MCC FGO/contrastive equivalence.
15. **Next remote order:** sync commit, activate environment, verify PKL SHA,
    validate configs, run each real-batch dry-run, run the four-model real-batch
    audit, run both two-epoch real-data smokes, explicitly reload/evaluate their
    best checkpoints, inspect artifacts, and only then decide whether to start
    the formal four folds.

## Configuration-to-constructor mapping

- GS-MCC `model.modality_encoder_type` is recorded in run metadata as the
  effective `context_encoder_type`; its graph operator is recorded as
  `causal_directed_polynomial_filter`. `official_fgo_reproduced=false`.
- DialogueGCN `context_encoder_type=causal_gru` maps directly to the stepwise
  forward-only GRU. Relation IDs use
  `source_speaker * num_speakers + target_speaker`, giving four relations for
  two speakers. Full nodal attention and future temporal relations are off.

## Selection and artifact contract

Every epoch performs Train then Validation. Best is updated only when
`current_val_weighted_f1 > best_val_weighted_f1`; exact ties retain the earlier
epoch. Test is not constructed or evaluated until training ends and the saved
best checkpoint has been rebuilt from its embedded configuration.

Each final split directory contains `metrics.csv`, `predictions.csv`,
`confusion_matrix.csv`, and `per_class_recall.csv`. Each run also contains
`epoch_metrics.csv`, `run_metadata.json`, `best_model.pt`, and `last_model.pt`.

## Verification record

```text
py_compile: PASS
config validation: PASS
GS-MCC synthetic end-to-end/reload: PASS
DialogueGCN synthetic end-to-end/reload: PASS
four-model synthetic audit at 1e-6: PASS
focused old/new causal pytest: 42 passed
full pytest: 60 passed
real IEMOCAP result: UNCONFIRMED
new dependency: NO
```
