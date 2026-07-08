# SDT isolated smoke review

Date: 2026-07-07

This is an isolated feasibility and smoke review for the official SDT baseline.
It does not integrate SDT into this project, does not modify the current MMGCN
training or evaluation flow, and does not change any real data, checkpoint, or
output directory.

## Scope

- Official repository: https://github.com/butterfliesss/SDT
- Reviewed commit: `60af9218231843939da4ddd82fd5e201adeec63a`
- Local isolated clone: `tmp/sdt_isolated_review/SDT`
- Main project files changed by this task: this report only
- Main project files intentionally not changed:
  - `scripts/train_mmgcn.py`
  - `scripts/evaluate_checkpoint.py`
  - `models/baselines/mmgcn/`
  - `datasets/`
  - `configs/train_mmgcn_m3ed.yaml`

## Source files reviewed

SDT files:

- `README.md`
- `requirements.txt`
- `exec_iemocap.sh`
- `exec_meld.sh`
- `dataloader.py`
- `model.py`
- `train.py`

Local context files:

- `AGENTS.md`
- `docs/project_map.md`
- `docs/codex_workflow.md`
- `docs/experiment_protocol.md`
- `docs/module_implementation_spec.md`
- `docs/smoke_test_protocol.md`
- `docs/git_workflow.md`
- `docs/baselines/baseline_rescreen_report.md`
- `configs/train_mmgcn_m3ed.yaml`
- `configs/smoke/train_mmgcn_smoke.yaml`
- `datasets/smoke/mmgcn_smoke_dataset.py`
- `datasets/collators/m3ed_collate.py`

## Repository status

The official SDT repository is small and contains one model file, one dataloader
file, one train script, two shell run scripts, and a short requirements file.
The README points users to preprocessed IEMOCAP/MELD feature files that should
be placed under `data/` in the SDT repo. The isolated review did not download
those datasets.

Official requirements:

```text
torch==1.4.0
numpy==1.19.2
pandas==1.1.5
scikit-learn==0.24.2
```

Current local conda environment used for smoke checks:

```text
torch 2.6.0+cpu
torch.cuda.is_available() == False
```

## SDT data contract

`dataloader.py` expects preprocessed pickle files:

- IEMOCAP: `data/iemocap_multimodal_features.pkl`
- MELD: `data/meld_multimodal_features.pkl`

Each item returned by the official dataloader is:

```text
text, visual, audio, qmask, umask, label, vid
```

The official collate function pads the first four fields with time-first layout
and pads `umask` and `label` with batch-first layout:

```text
textf: [T, B, D_text]
visuf: [T, B, D_visual]
acouf: [T, B, D_audio]
qmask: [T, B, n_speakers] before train.py permutes it
umask: [B, T]
label: [B, T]
```

`train.py` then permutes `qmask` to `[B, T, n_speakers]` before calling the
model.

This differs from the current project contract, which is batch-first:

```text
text_features:   [B, T, D_text]
audio_features:  [B, T, D_audio]
visual_features: [B, T, D_visual]
labels:          [B, T]
attention_mask:  [B, T]
speaker_ids_int: [B, T]
```

The adaptation is straightforward but must be explicit:

1. Transpose project feature tensors from `[B, T, D]` to `[T, B, D]`.
2. Pass visual before audio to match SDT forward order.
3. Convert `speaker_ids_int` to one-hot `qmask` with a padding speaker index.
4. Convert `attention_mask` to float or bool `umask` accepted by SDT losses.
5. Keep labels batch-first and masked by `attention_mask`.

## Model contract and dimensions

`Transformer_Based_Model` is configurable at construction time:

```text
dataset, temp, D_text, D_visual, D_audio, n_head,
n_classes, hidden_dim, n_speakers, dropout
```

The official train script hardcodes:

```text
D_text = 1024
D_visual = 342
D_audio = 1582 for IEMOCAP, 300 for MELD
n_classes = 6 for IEMOCAP, 7 for MELD
n_speakers = 2 for IEMOCAP, 9 for MELD
hidden_dim = 1024 by default
n_head = 8 by default
```

For this project's M3ED configuration, the model constructor can accept:

```text
D_text = 768
D_audio = 1024
D_visual = 342
n_classes = 7
```

The official `train.py` cannot be used as-is for M3ED because those dimensions
and paths are hardcoded there.

## Smoke checks performed

### Commit check

Command:

```powershell
git ls-remote https://github.com/butterfliesss/SDT.git HEAD
```

Result:

```text
60af9218231843939da4ddd82fd5e201adeec63a HEAD
```

### Isolated clone

Command:

```powershell
git clone --depth 1 https://github.com/butterfliesss/SDT.git tmp\sdt_isolated_review\SDT
```

Result: clone succeeded.

### Syntax parse

`py_compile` attempted to write `.pyc` files but Windows denied the atomic
rename step in the isolated clone/cache directory. To avoid modifying the
third-party source directory further, a read-only `ast.parse` syntax check was
used instead.

Command:

```powershell
conda run -n m3ed_mmgcn python -c "import ast, pathlib; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8')) for f in ['dataloader.py','model.py','train.py']]; print('syntax_parse_ok')"
```

Result:

```text
syntax_parse_ok
```

### Parameter count

Command instantiated the official model in the isolated clone without moving it
to CUDA.

Results:

```text
IEMOCAP-default 79687704 trainable parameters
M3ED-projected   5362204 trainable parameters
smoke-small         5692 trainable parameters
```

Where:

- `IEMOCAP-default`: `D_text=1024`, `D_visual=342`, `D_audio=1582`,
  `hidden_dim=1024`, `n_head=8`, `n_classes=6`, `n_speakers=2`.
- `M3ED-projected`: `D_text=768`, `D_visual=342`, `D_audio=1024`,
  `hidden_dim=256`, `n_head=8`, `n_classes=7`, `n_speakers=2`.
- `smoke-small`: tiny fake dimensions used only for local forward smoke.

### Original CPU fake forward

Original SDT code failed on CPU:

```text
AssertionError: Torch not compiled with CUDA enabled
```

Failure location:

```text
model.py:282
spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
```

This is a device-hardcoding issue. It triggers even for equal-length fake
dialogues because the code still constructs a CUDA tensor for the padding slice.

### Shape-only fake forward with runtime device shim

To separate device hardcoding from tensor-shape compatibility, a runtime-only
identity shim was used:

```python
torch.Tensor.cuda = lambda self, *a, **k: self
```

No SDT source file was edited. With the shim, a tiny fake batch completed
forward, SDT self-distillation losses, and backward.

Output shapes:

```text
[(2, 4, 3), (2, 4, 3), (2, 4, 3), (2, 4, 3), (2, 4, 3),
 (2, 4, 3), (2, 4, 3), (2, 4, 3), (2, 4, 3)]
```

Loss status:

```text
loss_finite True
```

Interpretation: the SDT model shape path is compatible with a fake dialogue
batch after a minimal device-neutral fix. The original code is not CPU-smokeable
without that fix.

## Training-loop risks

The official `train.py` is not suitable for this project's experiment protocol
without changes:

1. It calls both IEMOCAP and MELD loaders with `valid=0.0`.
2. It still evaluates a `valid_loader`, but that loader is empty and returns
   `nan` metrics.
3. It selects the best epoch using `test_fscore`, not validation metrics.
4. It does not save validation-selected checkpoints.
5. It reports weighted F1 and accuracy, but not Macro-F1 or UAR.
6. It writes `record_YYYY_M_D.pk` in the working directory.
7. The shell scripts launch 10 repeated runs in the background and redirect logs
   to `sdt_iemocap.txt` or `sdt_meld.txt`, which is not aligned with the current
   run directory convention.

For this project, SDT should reuse the project's validation-best checkpoint
selection and metric reporting rules rather than the official training loop.

## Compatibility verdict

SDT remains a viable next baseline candidate, but not as a drop-in script.

Positive:

- Official code is available and small.
- The model consumes utterance-level text/audio/visual features.
- The model constructor can accept M3ED feature dimensions.
- A fake forward/loss/backward path succeeds after removing device hardcoding.
- No PyTorch Geometric or raw-video dependency was found.

Risks:

- Official dependencies are old, especially `torch==1.4.0`.
- Official code assumes CUDA in `model.py`.
- Official train script uses test-best selection and no real validation split.
- Official dataloader depends on specific IEMOCAP/MELD pickle schemas.
- The official input layout is time-first for modality features, while this
  project uses batch-first.
- Full official hidden size has about 79.7M trainable parameters, so V100 16G is
  likely feasible but should still be checked with real M3ED dialogue lengths.

## Recommended next step

Do not connect SDT to `scripts/train_mmgcn.py`.

For a future integration task, add a separate, narrow SDT baseline path:

1. Create `models/baselines/sdt/` with a device-neutral port of the model.
2. Add a dedicated config such as `configs/baselines/sdt_m3ed_smoke.yaml`.
3. Add a dedicated training/evaluation entry or small model factory, preserving
   the current `[B, T, D]` collate contract.
4. Keep active modality and self-distillation weights YAML-controlled.
5. Add smoke checks for:
   - CPU fake forward/loss/backward,
   - validation-best checkpoint save,
   - checkpoint reload/evaluate,
   - required metrics: Weighted-F1, Macro-F1, UAR, Accuracy.

Until that exists, SDT should be treated as "source-level smoke passed with a
known device fix required", not as an integrated project baseline.
