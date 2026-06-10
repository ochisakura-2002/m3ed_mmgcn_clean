# Active Modalities Design Note

## 1. Purpose

This document records the planned design for adding optional modality activation and missing-modality evaluation to the current MMGCN pipeline.

The goal is to support single-modal, bi-modal, and tri-modal experiments without changing the dataset interface, data files, training loop semantics, or metric computation.

The default behavior must remain exactly equivalent to the current full tri-modal setting:

```text
text + audio + visual
```

If no modality-specific configuration is provided, the model should behave as the existing full-modality MMGCN baseline.

---

## 2. Current Project Context

Current project root:

```text
/home/zhiyuan/research/m3ed_mmgcn_clean
```

Current environment:

```text
Conda env: m3ed_mmgcn
Python: /home/zhiyuan/anaconda3/envs/m3ed_mmgcn/bin/python
GPU: V100 16G
```

Currently supported datasets:

```text
M3ED
IEMOCAP official MMGCN features
```

Currently supported models:

```text
SimpleMLP
MMGCN
```

The current MMGCN implementation is an official-aligned dialogue-level MMGCN adapter. The class name is still `M3EDMMGCN` for historical reasons, but the model is already used by both M3ED and IEMOCAP through a unified dialogue-batch interface.

---

## 3. Unified Batch Interface

Both M3ED and IEMOCAP dataloaders return the same batch structure:

```python
{
    "text_features":   [B, T, D_text],
    "audio_features":  [B, T, D_audio],
    "visual_features": [B, T, D_visual],
    "labels":          [B, T],
    "attention_mask":  [B, T],
    "lengths":         [B],
    "speaker_ids_int": [B, T],
    "dialogue_ids":    [...]
}
```

Dataset-specific feature dimensions:

```text
M3ED:
    text_dim   = 768
    audio_dim  = 1024
    visual_dim = 342
    num_classes = 7

IEMOCAP:
    text_dim   = 100
    audio_dim  = 1582
    visual_dim = 342
    num_classes = 6
```

The active-modality mechanism should depend only on modality names:

```text
text
audio
visual
```

It should not depend on dataset name, raw feature dimension, label space, or split strategy.

---

## 4. Design Principle

The dataset layer must always return complete tri-modal features.

Do not modify:

```text
datasets/
collators/
data/*.pkl
```

The dataset is responsible for reading real data. The model is responsible for deciding which modalities participate in computation.

This prevents modality-ablation experiments from changing the sample structure, label structure, or valid utterance mask.

---

## 5. New Configuration Field

Add an optional configuration field:

```yaml
modality:
  active_modalities:
    - text
    - audio
    - visual
```

If the `modality` field is absent, the default must be:

```yaml
modality:
  active_modalities:
    - text
    - audio
    - visual
```

Examples:

Text-only:

```yaml
modality:
  active_modalities:
    - text
```

Text + audio:

```yaml
modality:
  active_modalities:
    - text
    - audio
```

Text + visual:

```yaml
modality:
  active_modalities:
    - text
    - visual
```

Full tri-modal:

```yaml
modality:
  active_modalities:
    - text
    - audio
    - visual
```

---

## 6. Files to Modify

Required modifications:

```text
models/baselines/mmgcn/mm_gcn.py
models/baselines/mmgcn/dense_graph.py
scripts/train_mmgcn.py
scripts/evaluate_checkpoint.py
```

Recommended later modification:

```text
scripts/analyze/build_analysis_tables.py
```

Possible new files later:

```text
configs/modality_ablation/
scripts/experiments/generate_m3ed_modality_configs.py
scripts/evaluate_checkpoint_missing_modalities.py
```

---

## 7. Model-Level Changes

### 7.1 `mm_gcn.py`

The `M3EDMMGCN` class should add an optional argument:

```python
active_modalities=None
```

Default:

```python
("text", "audio", "visual")
```

The model should include a helper function to normalize modality names:

```python
_normalize_active_modalities(active_modalities)
```

Expected behavior:

```text
Input:  ["audio", "text"]
Output: ("text", "audio")
```

The function should:

```text
1. Convert names to lowercase
2. Validate that every name is one of text/audio/visual
3. Reject empty modality lists
4. Return modalities in fixed order: text, audio, visual
```

The model should also support a forward-level override:

```python
forward(..., active_modalities=None)
```

This enables two experiment types:

```text
1. Training from scratch with selected modalities
2. Evaluating a tri-modal checkpoint under test-time missing-modality settings
```

---

## 8. Missing-Modality Semantics

Inactive modalities should not merely have their input features set to zero.

Incorrect implementation:

```python
audio_features = torch.zeros_like(audio_features)
```

This is insufficient because the audio branch may still receive information from:

```text
audio_fc bias
modal embedding
graph nodes
residual connection
GCN h0 injection
classifier bias
```

Correct implementation:

```text
For inactive modality:
    projected hidden representation = zero
    modal embedding is not added
    speaker embedding is not added
    graph edges related to that modality are removed
    post-GCN hidden representation is forced to zero
    final fusion block for that modality remains zero
```

For example, if `audio` is inactive:

```text
audio_hidden = 0
audio-related adjacency rows = 0
audio-related adjacency columns = 0
audio_out = 0
```

The final classifier input dimension should remain unchanged:

```text
[B, T, 3H]
```

Inactive modality blocks are represented as zero vectors.

---

## 9. Graph-Level Changes

### 9.1 `dense_graph.py`

The graph construction function should accept:

```python
active_modalities=("text", "audio", "visual")
```

Full tri-modal behavior should remain equivalent to the current implementation.

For missing modalities, the graph should keep the fixed three-modality node layout:

```text
text nodes
audio nodes
visual nodes
```

But inactive modality nodes should be isolated:

```text
hidden = 0
adjacency row = 0
adjacency column = 0
```

This design keeps tensor shape stable and avoids changing the classifier input size.

### 9.2 Degree Normalization Safety

When inactive nodes have no edges, their degree may become zero.

The graph normalization step must avoid division by zero:

```python
degree = degree.clamp(min=eps)
```

The normalized adjacency of inactive nodes should remain zero, and no NaN values should appear.

---

## 10. Metric Compatibility

The metric computation does not need to change.

Current metrics are computed at the utterance level:

```text
logits: [B, T, num_classes]
labels: [B, T]
valid_mask: labels != IGNORE_INDEX
```

Modality missingness does not remove utterances. It only removes information channels.

Therefore, the following must remain unchanged:

```text
labels
attention_mask
lengths
valid_mask
loss computation
accuracy computation
macro-F1 computation
weighted-F1 computation
UAR computation
```

The model still outputs:

```text
logits: [B, T, num_classes]
```

Thus the existing training and evaluation metrics remain compatible.

---

## 11. Training Script Changes

### 11.1 `scripts/train_mmgcn.py`

In `build_model(config)`, read:

```python
modality_config = config.get("modality", {})
active_modalities = modality_config.get(
    "active_modalities",
    ["text", "audio", "visual"],
)
```

Pass it into the model:

```python
model = M3EDMMGCN(
    ...,
    active_modalities=active_modalities,
)
```

Old configs without a `modality` field must still train as full tri-modal MMGCN.

---

## 12. Evaluation Script Changes

### 12.1 `scripts/evaluate_checkpoint.py`

Evaluation reconstructs the model from `checkpoint["config"]`.

Therefore, it must also read:

```python
modality_config = config.get("modality", {})
active_modalities = modality_config.get(
    "active_modalities",
    ["text", "audio", "visual"],
)
```

Then pass it into the model:

```python
model = M3EDMMGCN(
    ...,
    active_modalities=active_modalities,
)
```

Old checkpoints should remain loadable with:

```python
strict=True
```

This requires that `active_modalities` is stored only as a Python attribute and does not introduce new learnable parameters or buffers.

---

## 13. Checkpoint Compatibility

The `state_dict` structure should not change.

Do not add new learnable parameters for this feature.

Do not add new registered buffers for this feature.

The checkpoint continues to save:

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config,
    ...
}
```

For new modality-ablation runs, `config` will include:

```yaml
modality:
  active_modalities:
    - text
    - audio
```

For old checkpoints, this field may be absent, and evaluation should default to full tri-modal mode.

---

## 14. Pipeline Compatibility

The pipeline does not need to understand modality internals.

Existing pipeline logic remains:

```text
pipeline yaml
    -> train_config_path
        -> train_mmgcn.py
            -> model config
                -> MMGCN
```

To run modality-ablation experiments, only `train_config_path` needs to point to a config that includes the `modality.active_modalities` field.

Example:

```yaml
train:
  enabled: true
  train_config_path: configs/modality_ablation/m3ed_mmgcn_TA.yaml
```

---

## 15. Regression Test Requirement

Before modifying model code, save the current full tri-modal test result.

After modifying model code, evaluate the same checkpoint again.

The full tri-modal result should remain unchanged.

Recommended commands:

```bash
cat outputs/latest_run.txt

RUN_DIR=$(grep '^run_dir=' outputs/latest_run.txt | cut -d= -f2-)

python scripts/evaluate_checkpoint.py \
  --checkpoint "$RUN_DIR/checkpoints/best_model.pt" \
  --split test

mkdir -p "$RUN_DIR/logs/evaluations_regression_baseline"

cp -r "$RUN_DIR/logs/evaluations/test_best_model" \
  "$RUN_DIR/logs/evaluations_regression_baseline/test_best_model_before_active_modalities"
```

After code modification:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint "$RUN_DIR/checkpoints/best_model.pt" \
  --split test
```

Compare:

```bash
diff \
  "$RUN_DIR/logs/evaluations_regression_baseline/test_best_model_before_active_modalities/metrics.csv" \
  "$RUN_DIR/logs/evaluations/test_best_model/metrics.csv"

diff \
  "$RUN_DIR/logs/evaluations_regression_baseline/test_best_model_before_active_modalities/predictions.csv" \
  "$RUN_DIR/logs/evaluations/test_best_model/predictions.csv"
```

Expected result:

```text
Full tri-modal metrics should remain unchanged.
Full tri-modal predictions should remain unchanged.
```

If the full tri-modal result changes, the modification has polluted the baseline path and must be fixed before running modality-ablation experiments.

---

## 16. Planned Experiment Types

### 16.1 Modality Contribution Analysis

Train separate models from scratch:

```text
T
A
V
T+A
T+V
A+V
T+A+V
```

This answers:

```text
How much information does each modality or modality combination provide when trained under the same setting?
```

### 16.2 Missing-Modality Robustness Analysis

Train a full tri-modal model:

```text
T+A+V
```

Evaluate the same checkpoint under different active-modality settings:

```text
T+A+V
T+A
T+V
A+V
T
A
V
```

This answers:

```text
How robust is a tri-modal model when one or more modalities are unavailable at test time?
```

These two experiment types must not be mixed in interpretation.

---

## 17. Analysis Table Extension

Later, `scripts/analyze/build_analysis_tables.py` should record:

```text
active_modalities
num_active_modalities
modality_setting
```

Example mapping:

```text
text                  -> T
audio                 -> A
visual                -> V
text+audio            -> TA
text+visual           -> TV
audio+visual          -> AV
text+audio+visual     -> TAV
```

This prevents modality-ablation runs from becoming indistinguishable in the analysis table.

---

## 18. Current Backup

Before starting the active-modality modification, the following files were backed up:

```text
_backup_before_active_modalities_20260609_193318/
  dense_graph.py
  evaluate_checkpoint.py
  mm_gcn.py
  train_mmgcn.py
```

These backups preserve the pre-modification MMGCN training and evaluation code.

---

## 19. Summary

This modification should be implemented as a backward-compatible extension.

Expected guarantees:

```text
1. Old configs without modality fields still run as full tri-modal MMGCN.
2. Old checkpoints still load with strict=True.
3. Dataset and collate logic remain unchanged.
4. Loss and metric computation remain unchanged.
5. Full tri-modal evaluation remains identical before and after the change.
6. Single-modal and bi-modal experiments are controlled only by config.
7. IEMOCAP remains compatible because the mechanism depends on modality names, not dataset-specific logic.
```

The key scientific purpose is to separate modality contribution analysis from missing-modality robustness analysis while preserving the existing M3ED and IEMOCAP training/evaluation pipeline.

