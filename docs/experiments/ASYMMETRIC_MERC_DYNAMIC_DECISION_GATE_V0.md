# TA-Preserving Dynamic Decision Gate v0

WORKSTREAM = ASYMMETRIC_MERC. Implementation base: branch
`integration/asymmetric-merc-v0`, HEAD `8cb93a0cf6acfd3fa6086698f5a95bf0aeac0a48`.
This task implements the last preapproved structural change on the current visual
reliability route. It does not run real IEMOCAP, access Test, commit or push.

The motivating observation is that fixed decision fusion B0 improved common-Dev
Weighted-F1 over TA-v0, whereas B1 Concat-Linear and B2 feature-level VRG-v0 were
negative. This module preserves the frozen experts' decisions and learns only
an utterance-specific visual correction weight. MMGCN remains a classic
reproduction anchor, not a selected final paper baseline.

## Files and reuse

- `models/asymmetric_merc/dynamic_decision_gate.py`: dynamic and constant models.
- `scripts/models/asymmetric_merc/train_dynamic_decision_gate.py`: one-seed entrypoint.
- `configs/asymmetric_merc/dynamic_decision_gate/seed100.yaml`: frozen protocol and switch.
- `scripts/analysis/models/asymmetric_merc/dynamic_decision_report.py`: post-selection report.
- `scripts/analysis/models/asymmetric_merc/weighted_fusion.py`: existing reader extended
  with explicit Train mode and Q2-2 reference score columns; its default Dev
  probability behavior is preserved. No additional prediction parser is introduced.
- `tests/models/asymmetric_merc/test_dynamic_decision_gate.py`: synthetic verification.
- This document.

Exact-ID inner joins, label/probability validation, metrics and fixed-alpha scores
come from `weighted_fusion.py`. Quadrants and gate group statistics come from
`vrg_report.py`. CSV/JSON serialization comes from `fusion_report.py`.
No TA/S2D/MultiDAG expert implementation, checkpoint or frozen feature cache is
loaded or imported by the new entrypoint.

## Model contract

Both inputs are frozen `[B,6]` probabilities in class order
`hap,sad,neu,ang,exc,fru`. They are read as float64, detached, and clamped from
below at `1e-12`; legal probability inputs are not renormalized.

```text
q = concat(P_TA, P_V, abs(P_TA-P_V), P_TA*P_V)      [B,24]
h = ReLU(Linear(24,32)(q))                         [B,32]
g = sigmoid(Linear(32,1)(h))                       [B,1]
s = (1-g)*log(P_TA) + g*log(P_V)                   [B,6]
loss = CrossEntropy(s, true_label)
```

There are exactly **833 trainable parameters**: `24*32+32+32+1`.
There is no learned classifier after fusion, dropout, attention, vector gate,
class-wise gate, MoE, Transformer, or additional confidence network. Labels, IDs,
correctness, quadrants, dialogue IDs and hidden representations are not model inputs.

The final layer's weights are zero; its bias is
`log(0.12/0.88) = -1.9924301646902063`. Thus epoch0 uses `g=0.12` everywhere,
preserving B0 scores `0.88*log(P_TA)+0.12*log(P_V)`. The standalone
`ConstantDecisionGate` has zero parameters and never trains. YAML
`dynamic_decision_gate.enabled: false` selects this sanity control and constructs
no optimizer. Existing expert and B1/B2 checkpoint schemas are unchanged; the
new gate has its own checkpoint schema.

## Input and reproduction gate

Explicit inputs, with no discovery or globbing:

| Expert | Train | Dev |
| --- | --- | --- |
| TA-v0 seed100 epoch18 | `train_best_predictions.tsv` | `validation_best_predictions.tsv` |
| Frozen Linear-B seed100 | `predictions_train.csv` | `predictions_dev.csv` |

The reader accepts the existing `predicted_label` / `pred_label` aliases,
`utterance_id`, `true_label`, and `prob_0..5`; Train metadata must say `train` and
Dev metadata must say `dev`, `val`, or `validation`. When split metadata is absent,
the explicit CLI argument declares it. IDs are kept verbatim, sorted and joined
exactly. Duplicates, invalid probabilities, argmax disagreements and conflicting
labels fail. Formal common counts must be Train **4818**, Dev **978**. Train/Dev
IDs must not overlap across either expert, including rows excluded by the join.

The tracked Q2-2 YAML contains only six-decimal metrics. It cannot establish a
`1e-10` reproduction contract. Formal mode therefore requires both existing Q2-2
artifacts via explicit paths:

- `--b0-predictions`: `fusion/seed100_predictions.csv`; the shared reader's
  `score_columns=True` mode reads its existing `score_0..5` and stored predictions.
  Softmax is used only to represent these reference scores in the common reader;
  base probability inputs are unchanged.
- `--b0-metrics`: `fusion/seed100_metrics.json`, with `alpha: 0.88`, matching
  `common_rows`, `synthetic: false`, and full-precision `fusion` values for
  `accuracy`, `weighted_f1`, `macro_f1`, and `uar`.

Before optimizer construction or any training step, `epoch0_reproduction.json`
must pass all of:

1. Formal Dev count is 978; reference IDs and labels equal the full common subset.
2. Epoch0 predictions exactly equal the independent shared fixed-alpha formula
   and frozen Q2-2 predictions.
3. All four epoch0 metrics equal fixed-alpha metrics and the frozen artifact
   within absolute tolerance **1e-10**. Reference metrics must also match metrics
   recomputed from reference predictions. Rounded targets receive no relaxed tolerance.
4. The untrained constant control's scores exactly equal epoch0 scores; initial
   gates are 0.12 within `1e-15`, with population std at most `1e-15`.

Failure records `ABORT_BEFORE_TRAINING`; no optimizer or best checkpoint exists.
Formula score differences are recorded separately, since NumPy/PyTorch log
implementations may differ at the last bit. Prediction identity remains exact.
Synthetic mode uses small artificial common subsets and may omit the frozen
references; this is explicitly recorded as `synthetic_formula_only`. It never
claims reproduction of the real B0 result or a scientific signal.

The actual frozen artifacts have not been read in this task. The real-data
epoch0 contract must pass at the start of a separately authorized formal run.

## Training and selection

The frozen protocol is seed100, AdamW, lr `1e-3`, weight decay `1e-2`, batch size
128, maximum 100 epochs, patience 10; Train shuffle is true and Dev shuffle false.
The sole loss is CrossEntropy applied to the unnormalized fusion scores as
logits. There is no scheduler, class weighting, label smoothing, sweep,
cross-fitting, K-fold or multi-seed launcher.

Epoch0 records Dev metrics/loss and gate mean/std and is saved as the initial best
checkpoint. Only `new_dev_weighted_f1 > best_dev_weighted_f1` replaces it. Equal
W-F1 preserves the earlier epoch regardless of loss. Ten consecutive epochs
without strict improvement stop training. The selected checkpoint is reloaded
and evaluated, and its Dev W-F1 is checked against the epoch0 floor. Synthetic
smoke runs may explicitly limit training to 1--3 epochs without changing the
frozen YAML protocol.

## Methodological boundary

This experiment evaluates whether a very low-capacity dynamic decision rule can
improve a frozen fixed-weight fusion baseline. Base-expert Train predictions are
in-sample rather than out-of-fold, so a positive result is treated as a
development signal pending reviewer-facing confirmation if required.

The manifest records `base_train_predictions: in_sample`,
`base_prediction_training_status: {TA: in_sample, V: in_sample}`, and
`cross_fitted: false`. This is low-capacity exploratory stacking / dynamic
decision fusion. Dev labels are used only for checkpoint selection and evaluation.

## Artifacts and mechanism schema

The run directory contains `best_model.pt`, `epoch_metrics.csv`,
`best_dev_predictions.csv`, `dev_gates.tsv`, `best_dev_metrics.json`,
`gate_mechanism.json`, `gate_mechanism.md`, `resolved_config.yaml`,
`run_manifest.json`, and `epoch0_reproduction.json`. Existing run directories are
never overwritten. Under `outputs/`, only
`outputs/<YYYYMMDD>/<experiment_group>/runs/<run_id>` is accepted. Synthetic
tests write only below the ignored workspace temporary directory.

The manifest includes Git HEAD/status, input and code hashes, runtime versions,
source declarations, joins, probability clipping counts, fixed criteria, dtype,
parameter count and Test isolation fields. Inputs are checked for changes during
the run. Source identities remain caller declarations; CSV contents alone do
not prove which checkpoint generated them.

`dev_gates.tsv` fields are `utterance_id,true_label,ta_pred,v_pred,dynamic_pred,
visual_gate,quadrant`, using frozen base predictions:
Q11=both correct, Q10=TA correct/V wrong, Q01=TA wrong/V correct, Q00=both wrong.
Mechanism JSON/Markdown includes:

- `all`: count, mean, population std, min, max.
- `quadrants.Q11/Q10/Q01/Q00` and `true_classes.<class>`: count, mean, std;
  absent groups have null statistics.
- `delta_rescue_harm = mean(g|Q01)-mean(g|Q10)`.
- `mean_abs_deviation_from_0_12`, `fraction_gate_gt_0_12`, `fraction_gate_lt_0_12`.
- Full-precision B0/selected metrics, `delta_weighted_f1`, fixed criteria, verdict,
  in-sample limitation and evidence mode.

Mechanism quantities are computed only after checkpoint selection and never
control optimization, patience or selection. Scientific criteria are frozen:

| Verdict | Condition | Next action |
| --- | --- | --- |
| STRONG_POSITIVE | delta W-F1 >= 0.005 and direction > 0 | Separate seeds100--104 task and necessary confirmation |
| PERFORMANCE_ONLY | delta W-F1 >= 0.005 and direction <= 0 | No reliability claim; GPT reviews seed expansion |
| WEAK | 0 < delta W-F1 < 0.005 | No automatic five-seed expansion |
| NEGATIVE | delta W-F1 <= 0 | End route; no v2 gate, attention, router or MoE |

Missing Q01/Q10 cannot establish a direction and is reported as undetermined for
an otherwise large improvement. Synthetic output always says `SYNTHETIC_ONLY`.

## Test isolation and verification

The CLI exposes only four base prediction paths, two B0 reference paths, config,
output directory, Dev-only evaluation, device and synthetic smoke options.
Every explicit path, including its resolved target, is checked for obvious
`test`, `testing`, and `validation_test` tokens before any input file is read or
hashed. Other split values fail before IO. Only CSV/TSV expert inputs are accepted;
there are no hidden-feature, feature-cache or source-checkpoint inputs.

Synthetic verification covers A--V from the request: exact join and permutation
invariance; duplicate/label/probability failures; shapes, 833 parameters,
initialization, clamp without renormalization, fixed-alpha/constant reproduction;
finite CrossEntropy backward and gate gradients; strict epoch0 checkpoint
selection and patience; frozen quadrants and mechanism delta; pre-IO isolation;
and successful end-to-end operation using probabilities alone. It also checks
reference damage, the `1e-10` boundary, protocol rejection, output collision,
disabled control and checkpoint reload.

```powershell
conda run --no-capture-output -n m3ed_mmgcn python -X utf8 -m pytest -q -p no:cacheprovider tests/models/asymmetric_merc/test_dynamic_decision_gate.py tests/models/asymmetric_merc/test_q2_2_weighted_fusion.py tests/models/asymmetric_merc/test_frozen_tav.py
```

Symlink-target validation is implemented. Its creation-based test may skip on
Windows without symlink permission; ordinary pre-IO path/split tests still run.
No real-data performance result is produced by this task.

Local verification on 2026-11-12: **166 passed, 1 skipped** across the three
Asymmetric MERC test modules above. The skip is the OS-protected symlink creation
case. `py_compile` passed for all four changed/new production Python files and
the new test file; the standalone entrypoint's `--help` passed. `git diff --check`
passed. No real predictions, Test files or expert checkpoints were opened.
