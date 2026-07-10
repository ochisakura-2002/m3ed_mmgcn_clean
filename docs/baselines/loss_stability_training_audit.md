# Loss stability training audit

## 1. Files audited

- `scripts/baselines/train_multidag_cl.py`
- `scripts/baselines/evaluate_multidag_cl_checkpoint.py`
- `scripts/run_experiment_pipeline.py`
- `scripts/train_mmgcn.py`
- `scripts/evaluate_checkpoint.py`
- `scripts/analyze/diagnose_multidag_cl_run.py`
- `scripts/analyze/plot_multidag_cl_stabilization_compare.py`
- `scripts/dev/validate_config_tree.py`
- `configs/baselines/multidag_cl/iemocap/formal/context_w5_tav.yaml`
- `configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_stable_candidate.yaml`
- `configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_stable_candidate.yaml`
- `configs/pipeline/mmgcn_pipeline.yaml`
- `docs/baselines/multidag_cl_baseline_stabilization_notes.md`
- `docs/baselines/multidag_cl_multirun_stabilization_analysis_notes.md`
- `docs/baselines/configs_cleanup_audit.md`

## 2. Current MultiDAG+CL epoch metrics

Before this pass, `scripts/baselines/train_multidag_cl.py` wrote these fields
to `logs/epoch_metrics.csv`:

- `epoch`
- `train_loss`
- `val_loss`
- `val_acc`
- `val_weighted_f1`
- `val_macro_f1`
- `val_uar`
- `train_acc`
- `train_weighted_f1`
- `train_macro_f1`
- `train_uar`
- `lr`
- `grad_norm`

This pass keeps those fields and adds scheduler/early-stopping observability
fields:

- `lr_after_scheduler`
- `scheduler_type`
- `scheduler_monitor`
- `scheduler_monitor_value`
- `stopped_early`
- `early_stop_epoch`
- `early_stop_monitor`
- `early_stop_monitor_value`

## 3. Audit answers

1. Current loss and metric fields: listed above.
2. LR was already recorded as `lr`.
3. Gradient norm was already recorded as `grad_norm`.
4. Scheduler support was not present before this pass. This pass adds
   disabled-by-default `none`, `reduce_on_plateau`, `cosine`, and `step`
   support.
5. Early stopping was not present before this pass. This pass adds
   disabled-by-default early stopping.
6. Label smoothing was not present before this pass. This pass adds optional
   masked CE label smoothing through `training.loss.label_smoothing`.
7. Class weights were not present before this pass. This pass adds optional
   `training.loss.class_weights: balanced`, computed from train labels and
   saved to `logs/class_weights.json`.
8. AdamW `betas` and `eps` were not configurable before this pass. This pass
   exposes them under `training.optimizer` while keeping AdamW as the only
   supported optimizer.
9. A separate auxiliary `val_loss` checkpoint while keeping `best_model.pt`
   selected by validation Weighted-F1 is still not implemented. The default
   `best_model.pt` remains selected by `val_weighted_f1`; an explicit
   `training.select_best_by` can change `best_model.pt`, but the new
   loss-stability YAMLs do not do that.
10. The MMGCN pipeline remains a separate validation/comparison path for this
    round. No MMGCN training logic, model logic, or dataset logic was modified.

## 4. Likely training factors behind unstable loss

- The stable candidate still used a fixed LR without any decay, so validation
  loss could rebound after early good epochs.
- Checkpoint selection used validation Weighted-F1, which is appropriate for
  reporting but can disagree with validation loss behavior.
- Plain CE can be sensitive when class distribution is uneven; optional label
  smoothing and balanced class weights now make that testable without changing
  the model.
- AdamW defaults were implicit, so optimizer numerical settings were not
  recorded or tunable in YAML.
- Early stopping was absent, so late validation loss degradation was only
  observed after the run finished.

## 5. Scope decision

This pass only changes training/evaluation mechanics and analysis/config
support. It does not add a new paper module, does not change
`MultiDAGCLBaseline`, does not change MMGCN, does not change datasets, and does
not use the test split for checkpoint selection.
