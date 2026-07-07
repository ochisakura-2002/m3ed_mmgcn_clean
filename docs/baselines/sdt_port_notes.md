# SDT project port notes

## 1. Purpose

This note documents the narrow project-side SDT baseline port added for a
device-neutral fake-forward smoke test. The task does not add formal SDT
training, does not connect SDT to `scripts/train_mmgcn.py`, and does not modify
the current MMGCN or dataset flow.

## 2. Files added

- `models/baselines/sdt/__init__.py`
- `models/baselines/sdt/sdt_model.py`
- `scripts/baselines/debug_sdt_forward.py`
- `docs/baselines/sdt_port_notes.md`

## 3. Interface

`SDTBaseline` accepts the current project batch-first dialogue contract:

```text
text_features:   [B, T, D_text]
audio_features:  [B, T, D_audio]
visual_features: [B, T, D_visual]
attention_mask:  [B, T]
speaker_ids_int: [B, T]
labels:          [B, T], optional
```

The forward output is a dictionary with:

```text
logits:          [B, T, num_classes]
loss:            masked loss or None
aux_losses:      CE/distillation loss components when labels are provided
features:        fused contextual states
modality_logits: text/audio/visual auxiliary logits
```

Padding positions are excluded from CE and distillation losses through
`attention_mask`.

## 4. Differences from official SDT

- The port is batch-first throughout. The official dataloader/model path uses
  time-first modality tensors before internal permutation.
- The port uses `nn.Linear` projections instead of official 1D convolution
  projections. For per-utterance feature projection this is the same shape
  role, but it is implemented in a project-native style.
- The port keeps the official SDT idea of intra/cross-modal transformer context,
  unimodal gated branches, multimodal gated fusion, and modality auxiliary
  predictions.
- The self-distillation loss is implemented as masked KL from each modality
  prediction to the detached fused prediction. This is a maintainable project
  approximation, not a line-by-line reproduction of official training code.
- The official SDT training script is intentionally not copied because it uses
  test-best selection and does not match the project validation/checkpoint
  protocol.

## 5. Device-neutral fixes

- The port contains no `.cuda()` calls.
- Speaker padding indices are created with `torch.full_like`/`torch.where` on
  the input tensor device.
- Sinusoidal position encodings are constructed on the input feature device and
  dtype.
- The fake-forward script runs on CPU by default and does not require GPU
  availability.

## 6. Fake-forward smoke results

Requested syntax commands:

```powershell
python -m py_compile models/baselines/sdt/__init__.py
python -m py_compile models/baselines/sdt/sdt_model.py
python -m py_compile scripts/baselines/debug_sdt_forward.py
```

Result: passed after rerunning outside the sandbox because the managed Windows
sandbox denied Python's atomic `.pyc` rename step with `WinError 5`.

The bare system command below did not run the model because system Python 3.13
does not have `torch` installed:

```powershell
python scripts/baselines/debug_sdt_forward.py
```

Project-environment smoke command:

```powershell
conda run -n m3ed_mmgcn python scripts/baselines/debug_sdt_forward.py
```

Result:

```text
[M3ED fake] logits shape: torch.Size([2, 4, 7])
[M3ED fake] loss finite: True
[M3ED fake] backward ok: True
[IEMOCAP fake] logits shape: torch.Size([2, 5, 6])
[IEMOCAP fake] loss finite: True
[IEMOCAP fake] backward ok: True
```

## 7. Remaining work before real training

1. Add an SDT-specific training entry or a small baseline model factory.
2. Add YAML configuration for SDT dimensions, hidden size, dropout,
   self-distillation, and loss weights.
3. Save validation-best checkpoints; never select by test results.
4. Report Weighted-F1, Macro-F1, UAR, and Accuracy.
5. Add checkpoint reload/evaluate support for SDT.
6. Add local smoke training before any real-data run.
7. Define a remote V100 full-training protocol with run IDs, seed records, and
   git commit hashes.

## 8. Recommendation for next task

Add a dedicated SDT local smoke training entry and YAML config, still separate
from the current MMGCN training path.
