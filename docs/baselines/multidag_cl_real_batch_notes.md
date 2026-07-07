# MultiDAG+CL real-batch dry run notes

This is a one-batch dry run only. It is not a formal training experiment.

## 1. Purpose

Verify whether one real M3ED or IEMOCAP dialogue batch can enter the
project-side `MultiDAGCLBaseline`, run forward with labels, compute masked CE
loss, run `loss.backward()`, and pass basic tensor-contract checks.

This task does not add formal MultiDAG+CL training, does not save checkpoints,
does not use the test split, and does not connect MultiDAG+CL to
`scripts/train_mmgcn.py`.

## 2. Files added

- `scripts/baselines/debug_multidag_cl_real_batch.py`
- `docs/baselines/multidag_cl_real_batch_notes.md`

## 3. Dataset/config used

Primary command attempted:

```powershell
conda run -n m3ed_mmgcn python -B scripts\baselines\debug_multidag_cl_real_batch.py --config configs\train_mmgcn_m3ed.yaml --dataset M3ED --split train --batch-size-override 2
```

Fallback command attempted:

```powershell
conda run -n m3ed_mmgcn python -B scripts\baselines\debug_multidag_cl_real_batch.py --config configs\train_mmgcn_iemocap_official.yaml --dataset IEMOCAP --split train --batch-size-override 2
```

The script only allows `train` or `val`; `test` is intentionally unavailable.

## 4. Batch fields and shapes

No real batch was loaded in this local checkout because the configured feature
files are missing.

When data is available, the script prints these fields:

```text
text_features
audio_features
visual_features
labels
attention_mask
speaker_ids_int
lengths
```

## 5. Forward/loss/backward result

`python -m py_compile scripts\baselines\debug_multidag_cl_real_batch.py` passed.

M3ED forward/loss/backward was not reached because the M3ED feature pkl is not
available locally.

IEMOCAP forward/loss/backward was not reached because the IEMOCAP official
feature pkl is not available locally.

## 6. Future leakage check

The script checks that `adjacency` has shape `[B, T, T]` and that no edge points
from an utterance to a future utterance. The check was not reached in this local
run because no real batch was loaded.

## 7. Failure reason, if any

M3ED real-data dry run was not completed because data file is missing:

```text
data/processed/M3ED/features/Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed.pkl
```

IEMOCAP real-data dry run was not completed because data file is missing:

```text
third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl
```

The failures are data-availability failures, not model-contract failures.

## 8. Remaining work before real training

1. Run the same one-batch dry run on a machine with the configured M3ED feature
   pkl and metadata files available.
2. If M3ED succeeds, record the printed tensor shapes, logits shape, loss
   finite result, backward result, valid count, and future leakage result.
3. Add a tiny-subset local smoke training entry for MultiDAG+CL real data before
   adding any formal real-data training entry.
4. Keep validation-best checkpoint selection for later real training; do not use
   test split for checkpoint selection.

## 9. Recommendation for next task

Next task: run MultiDAG+CL M3ED tiny-subset local smoke training on a machine
where the real M3ED feature files are available, then define the remote V100
protocol for full MultiDAG+CL training.
