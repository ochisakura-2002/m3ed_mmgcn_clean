# MultiDAG+CL project port notes

## 1. Purpose

This note documents the narrow project-side MultiDAG+CL baseline port added for
a device-neutral fake-forward smoke test. The task does not add formal
MultiDAG+CL training, does not connect MultiDAG+CL to `scripts/train_mmgcn.py`,
and does not modify the current MMGCN, SDT, dataset, training, or evaluation
flow.

## 2. Files added

- `models/baselines/multidag_cl/__init__.py`
- `models/baselines/multidag_cl/multidag_cl_model.py`
- `scripts/baselines/debug_multidag_cl_forward.py`
- `docs/baselines/multidag_cl_port_notes.md`

## 3. Interface

`MultiDAGCLBaseline` accepts the current project batch-first dialogue contract:

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
logits:                 [B, T, num_classes]
loss:                   masked CE loss or None
aux_losses:             CE/total loss entries when labels are provided
adjacency:              [B, T, T] directed past adjacency
same_speaker_mask:      [B, T, T]
different_speaker_mask: [B, T, T]
features:               contextual graph features
```

Padding positions are excluded from CE loss through `attention_mask`.

## 4. Differences from official MultiDAG+CL

- The port keeps text, audio, and visual tensors separate at the project
  boundary, then concatenates projected active modalities at one controlled
  fusion point.
- The official code reads JSON feature files and builds one concatenated
  feature tensor in `dataset.py`; this port consumes the existing project batch
  fields directly.
- The port keeps the official direction of feature-level DAG dialogue modeling,
  same/different speaker relations, and directed past context.
- The graph layer is a maintainable project implementation of speaker-aware
  causal DAG attention plus GRU updates, not a line-by-line copy of official
  `DAGERC_fushion`.
- The port does not implement official curriculum learning.
- The official training and evaluation scripts are intentionally not copied
  because they are not device-neutral and do not match the project
  validation-best checkpoint protocol.

## 5. Device-neutral fixes

- The port contains no `.cuda()` calls.
- All generated masks, adjacency matrices, speaker ids, and zero tensors are
  created on the input tensor device.
- The fake-forward script runs on CPU by default and does not require GPU
  availability.
- The graph code does not assume fixed batch size or fixed dialogue length.

## 6. Directed past adjacency

The helper `build_directed_past_adjacency` returns `[B, T, T]`, where
`adj[b, i, j] = 1` means utterance `i` may read utterance `j`.

Rules implemented:

- Only valid utterance rows and columns from `attention_mask` can be nonzero.
- Future edges are disallowed with `j <= i`.
- Valid self-loops are included.
- `window_past = 0` keeps only self-loops.
- `window_past > 0` keeps the current utterance plus at most the previous
  `window_past` valid utterances.
- Same/different speaker masks are built separately and consumed by the graph
  layer.

This is intentionally simpler than the official `get_adj_v1`, which stops after
a configured number of same-speaker predecessors and includes the intervening
history. That official detail should be revisited before real training.

## 7. Fake-forward smoke results

Requested syntax commands:

```powershell
python -m py_compile models/baselines/multidag_cl/__init__.py
python -m py_compile models/baselines/multidag_cl/multidag_cl_model.py
python -m py_compile scripts/baselines/debug_multidag_cl_forward.py
```

Result: passed after rerunning `python -m py_compile` outside the managed
Windows sandbox because the sandbox denied Python's atomic `.pyc` rename step
with `WinError 5`.

The bare system command below did not run the model because system Python does
not have `torch` installed:

```powershell
python scripts/baselines/debug_multidag_cl_forward.py
```

Project-environment smoke command:

```powershell
conda run -n m3ed_mmgcn python scripts/baselines/debug_multidag_cl_forward.py
```

Result:

```text
[M3ED fake] logits shape: torch.Size([2, 4, 7])
[M3ED fake] loss finite: True
[M3ED fake] backward ok: True
[M3ED fake] future leakage: False
[IEMOCAP fake] logits shape: torch.Size([2, 5, 6])
[IEMOCAP fake] loss finite: True
[IEMOCAP fake] backward ok: True
[IEMOCAP fake] future leakage: False
```

## 8. Remaining work before real training

1. Add a MultiDAG+CL-specific training entry or a small baseline model factory.
2. Add YAML configuration for dimensions, hidden size, dropout, active
   modalities, `window_past`, and graph-layer count.
3. Save validation-best checkpoints; never select by test results.
4. Report Weighted-F1, Macro-F1, UAR, and Accuracy.
5. Add checkpoint reload/evaluate support for MultiDAG+CL.
6. Add local smoke training before any real-data run.
7. Define a remote V100 full-training protocol with run IDs, seed records, and
   git commit hashes.
8. Decide whether to implement official curriculum learning.
9. Decide whether to add target-user mask, causal window, or modality robustness
   experiment protocols.
10. Revisit the exact official `get_adj_v1` predecessor rule before formal
    comparison.

## 9. Recommendation for next task

Add a dedicated MultiDAG+CL local smoke training entry and YAML config, still
separate from the current MMGCN training path.
