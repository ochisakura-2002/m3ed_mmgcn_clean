# MultiDAG+CL baseline stabilization notes

## 1. Problem observed

The formal IEMOCAP `context_w5_tav` run is unstable: test accuracy is around
58, roughly two points below the MMGCN comparison and far below the
paper-reported number around 68. The 30-epoch trend appears worse than MMGCN,
with validation loss and validation metrics starting to degrade after roughly
epoch 5.

Before adding any new modules, the project needs to determine whether the
current MultiDAG+CL-inspired baseline is overfitting, failing to optimize, or
being hurt by one specific stabilization-sensitive choice such as encoder type,
learning rate, graph depth, or dropout.

## 2. Why paper accuracy is not a direct target

The current implementation is MultiDAG+CL-inspired, not exact official
reproduction. It keeps the project batch contract, separate T/A/V feature
tensors, device-neutral code, validation-best checkpoint selection, and the
H-R causal online setting.

The paper number is therefore useful context, but it is not a direct target for
this code path unless the official setting is reproduced more literally,
including the official predecessor rule, official feature flow, and curriculum
or contrastive-learning details.

## 3. Known differences from official MultiDAG+CL

Known differences include:

- The implementation is not a line-by-line official reproduction.
- CL/curriculum learning is not implemented yet.
- The model consumes separate text, audio, and visual tensors instead of one
  concatenated official-style feature tensor.
- `past_all_causal` is not true offline full context; it reads only the current
  and previous valid utterances.
- The H-R route prioritizes causal online setting, not future-looking offline
  full context.
- Graph attention and update logic are a maintainable project implementation,
  not a direct copy of official `DAGERC_fushion`.

## 4. Diagnostic interpretation

Use `scripts/analyze/diagnose_multidag_cl_run.py` on a completed run before
changing model structure. It reads `epoch_metrics.csv`, best validation/test
evaluation CSVs, and `experiment_config.yaml`, then writes diagnosis outputs
under `figures/diagnostics/`.

Stabilization should first determine whether the problem is overfitting or
optimization failure:

- Overfitting-like: train loss improves while validation loss or validation
  Weighted-F1 degrades after the validation-best epoch.
- Optimization-failure-like: train loss barely improves, metrics stay weak, or
  gradients/LR suggest the model is not making progress.
- Mixed or unclear: one run is not decisive; use the stabilization set below.

Do not use the test split for checkpoint selection. Test metrics are final
reporting metrics for the validation-best checkpoint only.

## 5. Stabilization experiment set

New stabilization configs live under:

```text
configs/baselines/multidag_cl/iemocap/stabilize/
configs/pipeline/multidag_cl/iemocap/stabilize/
```

The set isolates one likely cause at a time:

- `context_w5_tav_linear_encoder.yaml`: replace causal GRU with linear encoder.
- `context_w5_tav_lr5e4.yaml`: lower LR to `0.0005`.
- `context_w5_tav_lr3e4.yaml`: lower LR to `0.0003`.
- `context_w5_tav_graph1.yaml`: reduce graph layers to 1.
- `context_w5_tav_dropout02.yaml`: reduce dropout to `0.2`.
- `context_w5_tav_dropout01.yaml`: reduce dropout to `0.1`.
- `context_w5_tav_stable_candidate.yaml`: conservative combination using
  dropout `0.2`, one graph layer, causal GRU, LR `0.0005`, weight decay
  `0.0001`, and grad clipping `1.0`.

All stabilize runs keep `epochs: 30`, `batch_size: 8`,
`select_best_by: val_weighted_f1`, and no train/validation/evaluation batch
caps.

Scheduler and early stopping are not enabled in this round. A later small
training-script change can add disabled-by-default support for
`ReduceLROnPlateau` on `val_weighted_f1` or `val_loss`, plus early stopping
with patience 5, after the current baseline diagnosis is reviewed.

## 6. Recommended run order

Recommended first three stabilization experiments:

1. `configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_lr5e4.yaml`
2. `configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_graph1.yaml`
3. `configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_stable_candidate.yaml`

Then run `context_w5_tav_linear_encoder.yaml` if the first three do not clearly
improve the trend, or if the diagnosis points specifically at causal GRU
instability.

## 7. How this supports the H-R causal dialogue route

The H-R causal route needs a stable online baseline before new modules are
meaningful. The stabilization set keeps causal constraints intact: no future
utterances, validation-only checkpoint selection, and no offline full-context
claim.

If the baseline becomes stable under lower LR, shallower graph depth, or a
more conservative encoder setting, later H-R modules can be compared against a
cleaner causal baseline instead of chasing noise from an unstable training
configuration.

## 8. What not to do yet

Do not add new modules before the baseline is stable.

Do not implement curriculum learning or contrastive learning as a stabilization
shortcut.

Do not change the `MultiDAGCLBaseline` main structure before the run diagnosis
identifies a concrete failure mode.

Do not modify MMGCN, SDT, datasets, data files, checkpoints, or historical
outputs for this stabilization task.

Do not call `past_all_causal` true offline full context.

## 9. Next decision rule

Use the diagnostic CSV/Markdown and the first stabilization runs to decide:

- If `stable_candidate` improves validation/test metrics and training trends,
  use it as the baseline for later module experiments.
- If `linear_encoder` beats `causal_gru`, report causal GRU instability and
  reconsider the encoder choice.
- If `graph1` beats the two-layer graph setting, reduce graph depth for the H-R
  baseline.
- If lower LR improves trends, update the formal default LR.
- If none of the stabilize configs improves the trajectory, stop and inspect
  data splits, label distribution, class-level errors, and implementation
  differences before adding modules.
