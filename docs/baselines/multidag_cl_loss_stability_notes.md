# MultiDAG+CL loss stability notes

## 1. Motivation

The current priority is to make the baseline train like a credible model before
adding new MERC modules. A noisy or non-decreasing training loss makes later
accuracy comparisons hard to interpret, even if one run lands on a decent test
number.

## 2. Why loss convergence is prioritized before accuracy

Accuracy and Weighted-F1 can improve from checkpoint luck, class imbalance, or
validation/test variance. A smoother decreasing train loss is a more basic
sanity signal: the optimizer, loss, LR scale, graph depth, and regularization
are at least cooperating.

If loss becomes more stable but accuracy does not immediately rise, that is
still useful baseline work. It makes later model changes easier to judge.

## 3. Current symptoms

The existing stable candidate is conservative but still uses a fixed LR and no
early stopping. Validation loss can degrade after early epochs while checkpoint
selection continues to follow validation Weighted-F1. That can be legitimate,
but it needs explicit diagnostics instead of being hand-waved.

## 4. Added training-stability options

All options are disabled by default unless YAML explicitly enables them.

- Scheduler: `none`, `reduce_on_plateau`, `cosine`, and `step`.
- Early stopping: optional monitor/mode/patience/min-delta stopping.
- Label smoothing: optional masked CE smoothing through
  `training.loss.label_smoothing`.
- Class weights: optional `training.loss.class_weights: balanced`, computed
  from train labels and saved to `logs/class_weights.json`.
- Optimizer parameters: optional AdamW `betas` and `eps`.

Evaluation loss uses the same optional label smoothing/class-weight settings as
training when those fields are present in the checkpoint config.

## 5. Loss-stability configs

New training configs live under:

```text
configs/baselines/multidag_cl/iemocap/loss_stability/
```

New pipeline configs live under:

```text
configs/pipeline/multidag_cl/iemocap/loss_stability/
```

These configs are training-stability experiments, not new model contributions.
They preserve the stable-candidate structure: causal GRU, one graph layer,
dropout `0.2`, causal window 5, TAV, 30 epochs, batch size 8, no batch caps, and
default `select_best_by: val_weighted_f1`.

## 6. Recommended run order

1. `configs/pipeline/multidag_cl/iemocap/loss_stability/context_w5_tav_stable_lr3e4_plateau_valloss.yaml`
2. `configs/pipeline/multidag_cl/iemocap/loss_stability/context_w5_tav_stable_lr3e4_cosine.yaml`
3. `configs/pipeline/multidag_cl/iemocap/loss_stability/context_w5_tav_stable_lr3e4_label_smoothing005.yaml`
4. `configs/pipeline/multidag_cl/iemocap/loss_stability/context_w5_tav_stable_lr5e4_plateau_valloss.yaml`
5. `configs/pipeline/multidag_cl/iemocap/loss_stability/context_w5_tav_stable_lr3e4_earlystop_valloss.yaml`

Early stopping itself does not make loss decrease more; it only prevents late
epochs from worsening the run once the monitored validation loss stops
improving.

## 7. How to diagnose results

After a run exists under `outputs/runs/<run_id>/`, run:

```bash
python scripts/analyze/diagnose_loss_stability.py --run-id <run_id>
```

For multiple runs:

```bash
python scripts/analyze/diagnose_loss_stability.py \
  --run-id RUN1 \
  --run-id RUN2 \
  --output-dir outputs/analysis/loss_stability_compare
```

The diagnostic script reads `logs/epoch_metrics.csv` only. It writes Markdown,
CSV, PNG, and PDF artifacts and labels each run as one of:

- `stable_decreasing`
- `decreasing_but_overfit`
- `unstable`
- `not_decreasing`
- `insufficient_data`

## 8. What not to conclude from one seed

One seed can show that a training setup is visibly broken or promising. It
cannot prove the baseline is stable across random seeds, and it cannot prove a
method improves test performance. Test accuracy and test Weighted-F1 are
reference metrics here, not checkpoint-selection criteria.

Validation-loss scheduling is not test tuning. It uses validation behavior only
and leaves test evaluation for final reporting.

## 9. Next decision rules

- If lr3e4 plus plateau gives smoother train loss and avoids early validation
  loss explosion, prefer it as the next baseline candidate.
- If cosine is smoother than plateau without hurting validation Weighted-F1,
  keep it in the multi-seed set.
- If label smoothing stabilizes loss but hurts Weighted-F1 badly, keep it as an
  analysis finding rather than a default.
- If early stopping merely truncates overfit without improving the loss trend,
  report it as a guardrail, not a training-quality fix.
- If none of these stabilize train loss, inspect data split behavior, class
  distribution, and model implementation differences before adding new modules.
