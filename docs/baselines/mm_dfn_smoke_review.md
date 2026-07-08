# MM-DFN isolated smoke review

Date: 2026-07-07

## 1. Task scope

This is an isolated source and smoke review for the official MM-DFN repository.
It does not integrate MM-DFN into the project, does not modify the current
MMGCN training or evaluation flow, and does not change datasets, checkpoints,
or formal experiment outputs.

Planned project-side change for this task: this report only.

MM-DFN was reviewed as a candidate answer to:

```text
Can MM-DFN serve as the main development baseline for this project?
```

Short answer: Maybe.

It is conceptually well aligned with dynamic multimodal graph fusion, but the
official repository is not suitable as a drop-in training path in the current
project environment. A project-native, device-neutral port would be required
before treating it as a formal baseline.

## 2. Local project constraints read

Relevant local context read:

| File | Result | Relevant note |
|---|---|---|
| `AGENTS.md` | Read | Current main baseline is `MMGCN`; no full repo scan; do not commit. |
| `docs/project_map.md` | Read | Current project map, unified batch contract, train/eval entry points. |
| `docs/codex_workflow.md` | Read | Local Windows is for editing and lightweight checks only. |
| `docs/experiment_protocol.md` | Read | Formal results need validation-selected checkpoint and Weighted-F1, Macro-F1, UAR, Accuracy. |
| `docs/module_implementation_spec.md` | Read | New modules must preserve the `[B, T, D]` contract and have baseline-equivalent paths. |
| `docs/smoke_test_protocol.md` | Read | Fake forward/import checks are smoke checks only, not model-effect evidence. |
| `docs/git_workflow.md` | Read | Do not commit automatically; avoid data/checkpoint/cache artifacts. |
| `docs/baselines/baseline_rescreen_report.md` | Read | MM-DFN was previously identified as a priority baseline candidate with old dependency risk. |
| `docs/baselines/sdt_port_notes.md` | Read | SDT is an isolated candidate port, not connected to formal training. |
| `docs/baselines/sdt_isolated_smoke_review.md` | Read | Extra SDT status context; not modified. |
| `docs/baselines/sdt_smoke_review.md` | Missing | The requested quick file does not exist in this checkout. |

Current project batch contract:

```text
text_features   [B, T, D_text]
audio_features  [B, T, D_audio]
visual_features [B, T, D_visual]
labels          [B, T]
attention_mask  [B, T]
speaker_ids_int [B, T]
```

Relevant current dimensions:

```text
M3ED:    text=768, audio=1024, visual=342, classes=7, speakers=2
IEMOCAP: text=100, audio=1582, visual=342, classes=6, speakers=2
```

## 3. SDT frozen status

`docs/baselines/sdt_port_notes.md` already documents SDT as a narrow
device-neutral fake-forward port. No SDT source or documentation was changed in
this task.

SDT current status:

- isolated candidate port
- fake-forward passed
- not part of official training pipeline
- not the main graph baseline
- keep as future non-graph comparison or backup
- no further SDT work in this task

## 4. Repository snapshot

| Item | Result |
|---|---|
| Repo URL | `https://github.com/zerohd4869/MM-DFN` |
| Commit hash | `da970366069247e05de3b9298f1e1bbc5c77a187` |
| README exists | Yes: `README.md` |
| requirements exists | Yes: `requirements.txt` |
| Train entry | `code/run_train_erc.py`; shell wrappers in `script/run_train_ie.sh` and `script/run_train_me.sh` |
| Dataset entry | `code/dataloader.py` |
| Model entry | `code/model.py`, `code/model_mm.py`, `code/model_GCN.py`, `code/model_fusion.py` |
| IEMOCAP support | Yes |
| MELD support | Yes |

The repository was shallow-cloned into `tmp/mm_dfn_isolated_review/MM-DFN`.
The upstream repository tracks two pkl feature files under `data/`; they were
pulled by the shallow clone but were not read. They were removed from the local
`tmp` review clone immediately after size inspection, leaving only source and
documentation files for this review.

## 5. Expected input format

The official dataloader reads pickle files rather than the current project
dataset objects.

IEMOCAP pickle schema in `code/dataloader.py`:

```text
videoIDs, videoSpeakers, videoLabels, videoText,
videoAudio, videoVisual, videoSentence, trainVid, testVid
```

MELD uses the same leading fields plus one extra trailing field.

Each IEMOCAP/MELD sample returns:

```text
text, visual, audio, qmask, umask, label, vid
```

The collate function pads modality tensors and `qmask` with `pad_sequence`
in time-first layout:

```text
textf: [T, B, D_text]
visuf: [T, B, D_visual]
acouf: [T, B, D_audio]
qmask: [T, B, n_speakers]
umask: [B, T]
label: [B, T]
```

The graph model computes `lengths` from `umask`, flattens valid utterances, and
returns valid-utterance logits:

```text
log_prob: [N_valid, num_classes]
```

Official dimensional assumptions in `code/run_train_erc.py`:

```text
IEMOCAP: text=100, audio=1582, visual=342, classes=6, speakers=2
MELD:    text=600, audio=300,  visual=342, classes=7, speakers=9
```

The default graph path in the provided shell scripts uses `--modals avl` and
`--graph_type GDF`, which routes into the multimodal dynamic fusion graph code.

## 6. Gap to current project batch contract

MM-DFN is close to the project shape, but not identical.

Required adapter from current project batches:

1. Transpose features from `[B, T, D]` to `[T, B, D]`.
2. Preserve modality order carefully: the official dataloader returns
   text, visual, audio, while the project contract is text, audio, visual.
3. Convert `speaker_ids_int [B, T]` to one-hot `qmask [T, B, n_speakers]`.
4. Convert `attention_mask [B, T]` to `umask [B, T]`.
5. Compute `lengths` from `attention_mask`.
6. Flatten labels to valid utterances for official-style loss, or repad logits
   back to `[B, T, C]` for project-style metrics/evaluation.
7. Replace hardcoded IEMOCAP/MELD dimensions with YAML-configured dimensions for
   M3ED and IEMOCAP official features.

The official training scripts should not be used as-is because they set
repository-local absolute-style paths such as `WORK_DIR="/MM-DFN"`, write to
their own `outputs/` and `logs/` conventions, and do not follow this project's
validation-best checkpoint protocol.

## 7. Dependency and GPU risk

README-listed core requirements:

```text
python 3.6.10
torch 1.4.0
torch-geometric 1.4.3
torch-scatter 2.0.4
scikit-learn 0.21.2
CUDA 10.1
```

The full `requirements.txt` is a broad environment snapshot with many unrelated
packages. It should not be installed directly into this project.

Dependency risks:

- `code/model.py` imports `torch_geometric.nn.RGCNConv` and `GraphConv`.
- The current local environment has `torch 2.6.0+cpu` and no
  `torch_geometric`.
- The code contains many direct `.cuda()` calls, including inside graph
  construction and fusion helper code.
- The core GDF submodule hits a PyTorch 2.6 advanced-indexing incompatibility
  in `MM_GCN.create_big_adj` during a shape-only fake-forward attempt.

GPU risk on V100 16G:

- Feature-level MM-DFN should be feasible in principle on V100 16G after a port.
- The dense multimodal adjacency in the GDF path scales roughly with
  `(num_modalities * valid_utterances)^2`, so long M3ED dialogues need a
  lightweight memory smoke before formal runs.
- The larger risk is not raw GPU memory; it is dependency, device, and old-code
  compatibility.

## 8. Fake forward / import check

Checks performed:

| Check | Result |
|---|---|
| `git ls-remote` HEAD | Passed: `da970366069247e05de3b9298f1e1bbc5c77a187` |
| Shallow clone | Passed into `tmp/mm_dfn_isolated_review/MM-DFN` |
| Syntax parse | Passed for `dataloader.py`, `loss.py`, `model.py`, `model_GCN.py`, `model_mm.py`, `model_fusion.py`, `run_train_erc.py` |
| Full `model.py` import | Failed: `ModuleNotFoundError: No module named 'torch_geometric'` |
| Core `MM_GCN` shape-only fake forward | Failed under current Torch after a runtime-only `.cuda()` identity shim; error occurs in `create_big_adj` advanced indexing |

Interpretation:

The source files are syntactically parseable, but the official model path is not
importable in the current environment without PyTorch Geometric. A limited core
GDF fake forward also exposed PyTorch-version fragility. Therefore, this review
does not claim official MM-DFN forward compatibility.

## 9. Fit as main development baseline

Verdict: Maybe.

Positive fit:

- MM-DFN is feature-level MERC, not raw-video or MLLM-based.
- It supports IEMOCAP and MELD, the same common baseline datasets used by many
  MERC papers.
- It has a dynamic multimodal graph-fusion path, which is closer to the project
  research direction than a plain transformer or MLP baseline.
- It already uses speaker information through `qmask` and speaker-aware graph
  logic.
- It can be mapped to the current `[B, T, D]` contract with a contained adapter
  or project-native port.

Negative fit:

- The official training stack is old and not reproducible in the current
  environment without dependency work.
- The official scripts use `valid_rate=0.0` in the provided shell wrappers; when
  no validation split is used, the training loop selects by test performance.
  That violates the project experiment protocol.
- Outputs and metrics do not match the current run-directory and checkpoint
  conventions.
- Current code is not device-neutral and not PyTorch-2.x clean.

Conclusion:

MM-DFN should not replace MMGCN as the current deployed baseline yet. It is a
strong candidate for a second graph-family baseline only after a small,
device-neutral project port and fake-forward smoke test pass.

## 10. Fit for future modules

| Future module | Fit | Note |
|---|---|---|
| target-user mask | Maybe | Speaker `qmask` is already central, but target-user evaluation needs project-side masking and padded-output handling. |
| causal window | Maybe | The graph code has `window_past` and `window_future`, but default recurrent/context paths are not strictly causal without port changes. |
| modality dropout | Yes | `modals` already supports modality subsets, and pre-graph feature masking can be added cleanly in a project-native port. |
| confidence gate | Maybe | Dynamic fusion is a natural insertion point, but reliability scores must be made explicit and YAML-controlled. |
| unimodal auxiliary heads | Yes | Audio, visual, and text branches exist before fusion, so auxiliary heads are structurally natural. |

## 11. Relation to MMGCN and newer baselines

MMGCN is still suitable as the classic graph baseline in this project.

MM-DFN is suitable as a dynamic fusion graph baseline candidate, but only after
porting. It should be treated as an MMGCN-family successor rather than a
drop-in replacement for the current MMGCN training path.

The two can jointly support paper experiments:

- MMGCN: stable classic graph baseline and current main experimental anchor.
- MM-DFN: dynamic graph/fusion comparison after a project-native smoke path is
  available.
- SDT: future non-graph comparison or backup, already frozen in this task.

Compared with newer candidates, MM-DFN is older but better aligned with the
current feature-level graph research line than many raw-video, MLLM, or
preprocessing-heavy methods. Its biggest weakness is engineering age, not
method relevance.

## 12. Recommendation

Main development baseline answer: Maybe.

Recommended integration style: C.

```text
C. Project-side device-neutral port + current unified batch contract
```

Why C instead of running the official scripts:

1. It preserves the project `[B, T, D]` batch contract.
2. It avoids installing the old official environment snapshot.
3. It removes hardcoded `.cuda()` behavior and absolute-style script paths.
4. It lets the project keep validation-selected checkpoints and required
   metrics: Weighted-F1, Macro-F1, UAR, Accuracy.
5. It makes M3ED dimensions YAML-configured instead of hardcoded around
   IEMOCAP/MELD.

Top 3 risks:

1. Old dependency stack and PyG: official `model.py` cannot import in the
   current environment because `torch_geometric` is missing, and the README
   targets Torch 1.4/CUDA 10.1.
2. Device and PyTorch-version fragility: many `.cuda()` calls and at least one
   PyTorch 2.6 indexing incompatibility were observed in the GDF path.
3. Protocol mismatch: official scripts use old path/output conventions and can
   select by test performance when `valid_rate=0.0`, which conflicts with the
   project's validation-best checkpoint rule.

## 13. Next action

Recommended next action:

```text
MM-DFN device-neutral port
```

Scope for that future task should be narrow:

1. Add a project-native MM-DFN model wrapper under a clearly separated baseline
   path.
2. Keep all dimensions and optional modules YAML-controlled.
3. Accept the current project batch directly:

```text
text_features, audio_features, visual_features, attention_mask, speaker_ids_int
```

4. Return padded logits `[B, T, C]` and masked loss.
5. Add CPU fake forward/loss/backward before any training entry.
6. Do not connect it to `scripts/train_mmgcn.py` until the isolated smoke path
   is stable.

No blocker was found for the general "local edit + remote V100 training"
workflow, but MM-DFN should not be sent to remote training until its dependency,
device, and validation-protocol issues are removed by a project-native port.
