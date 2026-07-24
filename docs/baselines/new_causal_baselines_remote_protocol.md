# New causal graph baselines — remote IEMOCAP protocol

This protocol validates code and real-data compatibility before any formal
30-epoch run. Do not treat synthetic values or two-epoch smoke values as model
performance.

## 1. Synchronize and identify the revision

```bash
git pull --ff-only
git status --short
git rev-parse HEAD
```

The working tree should contain only intentional remote-local files. Do not
continue from an unknown dirty model/config state.

## 2. Activate the existing environment

```bash
conda activate m3ed_mmgcn
python --version
python -c "import torch, yaml, pandas, numpy, sklearn; print(torch.__version__)"
```

Do not install PyG, `torch-scatter`, or `torch-sparse`.

## 3. Verify the official feature file

```bash
test -f third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl
sha256sum third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl
```

Required SHA256:

```text
ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3
```

Stop on a mismatch; do not edit the PKL or silently change the configured hash.

## 4. Validate the configuration tree

```bash
python scripts/dev/validate_config_tree.py
```

## 5. Run one real batch per new model

```bash
python scripts/baselines/debug_new_causal_graph_real_batch.py \
  --config configs/gsmcc/project_variant/iemocap/causal_context/legacy_mmgcn_features/val_official_prefix.yaml
```

```bash
python scripts/baselines/debug_new_causal_graph_real_batch.py \
  --config configs/dialoguegcn/unified/iemocap/causal_context/legacy_mmgcn_features/val_official_prefix.yaml
```

Return dialogue IDs, lengths, three modality shapes, logits/representation
shapes, finite-gradient status, loss components, edge/relation counts, and
padding-loss exclusion status.

## 6. Run the unified real-batch causal audit

```bash
python scripts/analyze/run_four_model_causal_audit.py \
  --config configs/analysis/four_model_causal_audit.yaml \
  --strict
```

Keep all four per-model directories and the summary/report, including partial
outputs if one model fails. A strict failure blocks formal training.

## 7. Run two-epoch real-data smoke training

```bash
python scripts/baselines/train_new_causal_graph_baseline.py \
  --config configs/gsmcc/project_variant/iemocap/causal_context/legacy_mmgcn_features/smoke_real_2epoch.yaml
GS_RUN_ID=$(awk -F= '$1=="run_id" {print $2}' outputs/dev/latest_run.txt)
```

```bash
python scripts/baselines/evaluate_new_causal_graph_checkpoint.py \
  --checkpoint "outputs/dev/${GS_RUN_ID}/checkpoints/best_model.pt" \
  --split val
python scripts/baselines/evaluate_new_causal_graph_checkpoint.py \
  --checkpoint "outputs/dev/${GS_RUN_ID}/checkpoints/best_model.pt" \
  --split test
```

```bash
python scripts/baselines/train_new_causal_graph_baseline.py \
  --config configs/dialoguegcn/unified/iemocap/causal_context/legacy_mmgcn_features/smoke_real_2epoch.yaml
DIALOGUEGCN_RUN_ID=$(awk -F= '$1=="run_id" {print $2}' outputs/dev/latest_run.txt)
```

```bash
python scripts/baselines/evaluate_new_causal_graph_checkpoint.py \
  --checkpoint "outputs/dev/${DIALOGUEGCN_RUN_ID}/checkpoints/best_model.pt" \
  --split val
python scripts/baselines/evaluate_new_causal_graph_checkpoint.py \
  --checkpoint "outputs/dev/${DIALOGUEGCN_RUN_ID}/checkpoints/best_model.pt" \
  --split test
```

## 8. Check run contents

```bash
find "outputs/dev/${GS_RUN_ID}" -maxdepth 4 -type f | sort
find "outputs/dev/${DIALOGUEGCN_RUN_ID}" -maxdepth 4 -type f | sort
```

For each run confirm best/last checkpoints, embedded configuration reload,
`run_metadata.json`, `epoch_metrics.csv`, and four final artifacts for both Val
and Test. Confirm checkpoint selection says validation Weighted-F1 and
`test_split_used_for_selection=false`.

## 9. Return this report before formal training

Return:

1. commit hash and environment versions;
2. PKL SHA256;
3. config-validation result;
4. both real-batch dry-run outputs;
5. four-model audit summary/report;
6. both two-epoch run IDs and best epochs;
7. checkpoint reload/evaluation status;
8. artifact file listings;
9. any warning/error and every remaining `UNCONFIRMED` item.

Do not automatically start the 30-epoch Session-Holdout folds. Start them only
after explicit review of this gate report; Test must never select a checkpoint,
seed, configuration, or model.
