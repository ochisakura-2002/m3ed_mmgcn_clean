# MultiDAG+CL encoder audit notes

Date: 2026-07-08

## 1. Purpose

This note records the narrow model-port refinement for the project-side
`MultiDAGCLBaseline`. The task updates the three modality input encoders to a
configurable causal encoder, audits the existing attention and graph update
path, and keeps MultiDAG+CL separate from MMGCN.

This is not a formal IEMOCAP or M3ED experiment. It does not implement
curriculum learning and does not connect MultiDAG+CL to the formal training
pipeline.

## 2. Files changed

Changed:

- `models/baselines/multidag_cl/__init__.py`
- `models/baselines/multidag_cl/multidag_cl_model.py`
- `scripts/baselines/train_multidag_cl_smoke.py`
- `scripts/baselines/debug_multidag_cl_real_batch.py`
- `configs/smoke/train_multidag_cl_smoke.yaml`

Added:

- `docs/baselines/multidag_cl_encoder_audit_notes.md`

Not changed:

- `scripts/train_mmgcn.py`
- `scripts/evaluate_checkpoint.py`
- `datasets/`
- `models/baselines/mmgcn/`
- `models/baselines/sdt/`

## 3. Previous encoder issue

The previous project port encoded each modality with only:

```text
Linear(input_dim -> hidden_dim)
ReLU
Dropout
```

That was useful for a minimal fake-forward port, but it was too shallow for a
dialogue emotion baseline that should model causal utterance history before
graph propagation.

## 4. New causal modality encoder

`CausalModalityEncoder` was added inside
`models/baselines/multidag_cl/multidag_cl_model.py`.

Default configuration:

```text
modality_encoder_type="causal_gru"
modality_encoder_layers=1
```

Default structure:

```text
Linear(input_dim -> hidden_dim, bias=False)
LayerNorm(hidden_dim)
GELU
Dropout
Unidirectional GRU(hidden_dim -> hidden_dim, batch_first=True)
LayerNorm(hidden_dim)
Dropout
```

The GRU is explicitly unidirectional and uses `batch_first=True`. No BiGRU or
BiLSTM is used.

The baseline-equivalent ablation path is retained:

```text
modality_encoder_type="linear"
```

That path uses `Linear + ReLU + Dropout` and masks padded positions.

## 5. Causal guarantee

The default encoder is causal because each modality uses a unidirectional GRU.
At utterance `i`, the GRU state can depend only on utterances `<= i` in the
same sequence.

Padding is handled through `attention_mask`:

- encoder inputs are zeroed at padded positions;
- sequence lengths are derived from `attention_mask`;
- `pack_padded_sequence` skips trailing padding during GRU execution;
- encoder outputs are masked again before fusion.

This matches the online dialogue setting better than BiLSTM or BiGRU, because
bidirectional recurrent encoders would read future utterances. A causal GRU is
also lighter than a causal Transformer and is a stable first choice for
feature-level IEMOCAP/M3ED baselines.

## 6. Attention / graph update audit

Audit result: no graph or attention rewrite was needed in this task.

Checked points:

- Directed past adjacency still allows only `j <= i` and valid utterance pairs.
- `window_past` is non-negative and constrains valid previous utterances.
- `same_speaker_mask` and `different_speaker_mask` are built from valid pairs
  under `attention_mask`.
- Graph attention uses the adjacency row as its mask before normalization.
- The graph update uses learnable query/key/value projections, relation bias,
  `GRUCell`, LayerNorm, and a feed-forward block.
- Logits remain `[B, T, C]`.
- Masked cross entropy uses only positions where `attention_mask` is true and
  labels are non-negative.
- Fake backward checks produced finite gradients after the encoder change.

The implementation is not a line-by-line copy of official MultiDAG+CL, but it
preserves the required project behavior for this round: causal, speaker-aware,
directed past graph, learnable parameters, and masked training objective.

## 7. Compatibility with fake-forward and smoke training

Syntax check:

```text
python -m py_compile models/baselines/multidag_cl/__init__.py models/baselines/multidag_cl/multidag_cl_model.py scripts/baselines/debug_multidag_cl_forward.py scripts/baselines/train_multidag_cl_smoke.py scripts/baselines/debug_multidag_cl_real_batch.py
```

Result: passed after rerunning outside the managed Windows sandbox because the
first attempt hit a `.pyc` atomic rename permission error.

Fake-forward:

```text
conda run -n m3ed_mmgcn python scripts/baselines/debug_multidag_cl_forward.py
```

Result: passed.

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

Fake smoke training:

```text
conda run -n m3ed_mmgcn python scripts/baselines/train_multidag_cl_smoke.py --config configs/smoke/train_multidag_cl_smoke.yaml
```

Result: passed.

```text
output: outputs/smoke/multidag_cl/20260708_103657_multidag_cl_smoke
best epoch: 1
selection metric: val_weighted_f1
reload val: acc=0.5714, weighted_f1=0.4190, macro_f1=0.3667, uar=0.5000
```

Smoke metrics are only a code-path check, not evidence of model quality.

Additional ablation sanity check:

```text
modality_encoder_type="linear"
```

Result: a small forward/loss/backward check passed.

## 8. Real-batch dry run status

Attempted command:

```text
conda run -n m3ed_mmgcn python -B scripts/baselines/debug_multidag_cl_real_batch.py --config configs/train_mmgcn_iemocap_official.yaml --dataset IEMOCAP --split train --batch-size-override 2 --device config
```

Result: not completed locally because the configured IEMOCAP feature file is
missing:

```text
third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl
```

The script did not reach model forward/loss/backward for real data. This is a
local data-availability issue, not a model-contract failure.

## 9. Remaining differences from official MultiDAG+CL

- Curriculum learning is still not implemented.
- The project port consumes separate `text_features`, `audio_features`, and
  `visual_features`; official code consumes one concatenated feature tensor.
- The graph layer remains a maintainable project implementation rather than a
  line-by-line copy of official `DAGERC_fushion`.
- The exact official predecessor rule from `get_adj_v1` is still not copied.
- Official nodal attention variants are not implemented.
- The project port is device-neutral and contains no `.cuda()` calls.

## 10. Recommendation for next task

Next recommended task: run the real-data one-batch dry run on a machine where
the configured IEMOCAP or M3ED feature files are available. If that passes,
continue with a separate MultiDAG+CL train/eval/pipeline integration task while
keeping validation-best checkpoint selection and the required metric set.

Explicit answers:

1. Is the encoder no longer only Linear + ReLU? Yes. The default path is now a
   causal GRU encoder with LayerNorm, GELU, Dropout, and masking.
2. Is the default encoder causal? Yes.
3. Does it use BiGRU or BiLSTM? No.
4. Was attention / graph update modified? No.
5. Why not modify it? The audit found the current graph path already satisfies
   causal directed adjacency, speaker-aware masks, learnable updates, masked
   CE, and fake forward/backward checks.
6. Did fake forward pass? Yes, in the `m3ed_mmgcn` conda environment.
7. Did fake smoke training pass? Yes.
8. Did this affect MMGCN / SDT / datasets / `train_mmgcn.py`? No.
