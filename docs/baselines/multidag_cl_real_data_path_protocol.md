# MultiDAG+CL real-data path and dry-run protocol

## 1. Purpose

This document records the data-path preconditions and one-batch dry-run
protocol for the project-side `MultiDAGCLBaseline` on real M3ED or IEMOCAP
batches.

The dry run is only meant to check whether one existing real-data dataloader can
produce a batch that enters MultiDAG+CL, computes masked CE loss, runs
`loss.backward()`, and passes tensor-contract checks. It is not formal training,
does not save checkpoints, and does not use the test split.

## 2. Current failure summary

The latest local M3ED command reached the script entrypoint and config loading,
then stopped before dataloader construction completed:

```text
M3ED real-data dry run was not completed because data file is missing: data/processed/M3ED/features/Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed.pkl
```

This is a data-availability failure. It does not show a MultiDAG+CL forward,
loss, backward, or tensor-contract failure because no real batch was loaded.

## 3. Config paths checked

Checked from the project root on local Windows.

| Dataset | Config | Key | Path | Exists locally | Note |
|---|---|---|---|---|---|
| M3ED | `configs/mmgcn/unified/m3ed/full_context/m3ed_features/skeleton.yaml` | `dataset.feature_pkl_path` | `data/processed/M3ED/features/Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed.pkl` | No | Required before M3ED dataloader construction. |
| M3ED | `configs/mmgcn/unified/m3ed/full_context/m3ed_features/skeleton.yaml` | `dataset.metadata_path` | `data/metadata/m3ed_metadata.csv` | No | Required for dialogue metadata and split/sample alignment. |
| M3ED | `configs/mmgcn/unified/m3ed/full_context/m3ed_features/skeleton.yaml` | `dataset.label_mapping_path` | `data/metadata/m3ed_label_mapping.csv` | No | Required for label mapping. |
| IEMOCAP | `configs/mmgcn/unified/iemocap/full_context/legacy_mmgcn_features/val_official_prefix.yaml` | `dataset.feature_pkl_path` | `third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl` | No | Required before IEMOCAP dataloader construction. |

## 4. Required files before real one-batch dry run

For M3ED, the machine running the dry run must have these files at the relative
paths declared by `configs/mmgcn/unified/m3ed/full_context/m3ed_features/skeleton.yaml`:

```text
data/processed/M3ED/features/Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed.pkl
data/metadata/m3ed_metadata.csv
data/metadata/m3ed_label_mapping.csv
```

For IEMOCAP, the machine running the dry run must have this file at the relative
path declared by `configs/mmgcn/unified/iemocap/full_context/legacy_mmgcn_features/val_official_prefix.yaml`:

```text
third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl
```

If a local or remote machine stores data elsewhere, prefer a local-only debug
config or an environment-specific launch wrapper. Do not hardcode personal
absolute paths into the formal project configs.

## 5. Local Windows dry-run command

Run from the project root. Use the project environment, keep the command to one
batch, and keep the split to `train` or `val`.

```powershell
conda run -n m3ed_mmgcn python -B scripts\diagnostics\models\multidag_cl\debug_multidag_cl_real_batch.py --config configs\mmgcn\unified\m3ed\full_context\m3ed_features\skeleton.yaml --dataset M3ED --split train --batch-size-override 2 --device cpu
```

Optional IEMOCAP path check and one-batch dry run:

```powershell
conda run -n m3ed_mmgcn python -B scripts\diagnostics\models\multidag_cl\debug_multidag_cl_real_batch.py --config configs\mmgcn\unified\iemocap\full_context\legacy_mmgcn_features\val_official_prefix.yaml --dataset IEMOCAP --split train --batch-size-override 2 --device cpu
```

Do not run a full epoch or formal training on local Windows for this task.

## 6. Remote V100 dry-run command

After syncing the code to the remote V100 machine and confirming the data files
exist at the config-declared relative paths, run from the project root:

```bash
conda run -n m3ed_mmgcn python -B scripts/diagnostics/models/multidag_cl/debug_multidag_cl_real_batch.py --config configs/mmgcn/unified/m3ed/full_context/m3ed_features/skeleton.yaml --dataset M3ED --split train --batch-size-override 2 --device config
```

Optional IEMOCAP fallback:

```bash
conda run -n m3ed_mmgcn python -B scripts/diagnostics/models/multidag_cl/debug_multidag_cl_real_batch.py --config configs/mmgcn/unified/iemocap/full_context/legacy_mmgcn_features/val_official_prefix.yaml --dataset IEMOCAP --split train --batch-size-override 2 --device config
```

Record the remote git commit hash before running:

```bash
git rev-parse HEAD
```

## 7. Expected successful output

A successful dry run should print:

```text
MultiDAG+CL real-data one-batch dry run
Config: <config path>
Dataset: M3ED or IEMOCAP
Split: train or val
Batch size override: 2
Device: cpu or cuda
Core tensor shapes:
  text_features: shape=[B, T, D_text]
  audio_features: shape=[B, T, D_audio]
  visual_features: shape=[B, T, D_visual]
  labels: shape=[B, T]
  attention_mask: shape=[B, T]
  speaker_ids_int: shape=[B, T]
  lengths: shape=[B]
Dry-run checks:
  logits shape: (B, T, num_classes)
  loss: <finite value>
  loss finite: True
  backward ok: True
  adjacency shape: (B, T, T)
  future leakage: False
  valid_count: <positive integer>
M3ED real-data one-batch dry run passed.
```

For IEMOCAP, the final line should use `IEMOCAP` instead of `M3ED`.

When reporting a successful run, record the config path, dataset, split, device,
batch size, printed core tensor shapes, logits shape, loss value, `loss finite`,
`backward ok`, adjacency shape, `future leakage`, `valid_count`, and git commit
hash if run remotely.

## 8. What to do if files are missing

If the script exits with a message like `data file is missing`, do not create
fake files, do not copy large feature pkl files into Git, and do not change the
formal config paths just to satisfy one machine.

Instead:

1. Confirm the file exists on the intended data machine.
2. If data exists under a different local root, create a separate local debug
   config in a later task and keep it out of formal experiment configs unless it
   uses portable relative paths.
3. If using the remote V100 machine, sync the code first, then run the same
   one-batch command where the real data already exists.
4. Record the exact missing relative path and the script exit reason in the
   run notes.

If the files exist but the script reports a contract failure, record the full
error message plus the printed batch keys and shapes. That would be a real
adapter/model-contract issue and should be handled in a separate code task.

## 9. What not to commit

Do not commit:

```text
data/
outputs/
tmp/
checkpoint/
checkpoints/
logs/
third_party/MMGCN_official/IEMOCAP_features/*.pkl
*.pkl
*.pickle
*.pt
*.pth
*.ckpt
```

Also do not commit personal absolute paths, copied real feature files, remote
server paths, or full training outputs.

## 10. Recommendation for next task

Choose one of these paths:

1. Put the required data files at the existing relative paths on local Windows,
   then rerun the M3ED one-batch dry run.
2. Push/sync the current code to the remote V100 machine and rerun the M3ED
   one-batch dry run where the real M3ED files already exist.
3. Create a local debug config that points to the real data location using
   portable relative paths or an environment-specific convention, then rerun the
   one-batch dry run.

Only after a real one-batch dry run passes should the project move to a
tiny-subset real-data smoke training protocol for MultiDAG+CL.
