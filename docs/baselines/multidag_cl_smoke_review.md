# MultiDAG+CL isolated smoke review

Date: 2026-07-07

## 1. Task scope

This is an isolated source and smoke review for the official MultiDAG+CL
repository. It answers whether MultiDAG+CL is a better fit than MM-DFN as this
project's main development baseline or near-term causal/DAG graph baseline.

This task did not integrate MultiDAG+CL into `models/baselines/`, did not modify
the current MMGCN training/evaluation flow, and did not change datasets,
checkpoints, or formal experiment outputs.

Planned project-side change for this task: this report only.

Short answer:

```text
MultiDAG+CL: Maybe
```

It is more attractive than MM-DFN as a near-term causal/DAG graph comparison
because it is pure PyTorch at the graph core, batch-first, speaker-aware, and
does not require PyTorch Geometric. It is not ready as a drop-in main baseline
because the official scripts have dependency, device, data-path, modality, and
checkpoint-selection issues.

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
| `docs/baselines/baseline_rescreen_report.md` | Read | MultiDAG+CL was previously identified as a priority baseline candidate for causal/window-oriented graph work. |
| `docs/baselines/sdt_port_notes.md` | Read | SDT is an isolated candidate port, not connected to formal training. |
| `docs/baselines/mm_dfn_smoke_review.md` | Read | MM-DFN is method-aligned but blocked by old dependency/device/protocol risks. |

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
M3ED:                     text=768, audio=1024, visual=342, classes=7, speakers=2
IEMOCAP official adapter: text=100, audio=1582, visual=342, classes=6, speakers=2
Target hardware:          V100 16G
```

## 3. Why MM-DFN is not immediately ported

The previous MM-DFN review concluded:

```text
MM-DFN: Maybe
```

MM-DFN is conceptually aligned with dynamic multimodal graph fusion, but the
official repository is not a safe immediate port target:

1. It targets old Python/Torch/CUDA/PyG versions.
2. The current environment lacks `torch_geometric`.
3. The code contains many direct `.cuda()` calls.
4. A PyTorch 2.6 indexing incompatibility was observed in the GDF fake-forward
   path.
5. The official training protocol can select by test performance, which violates
   this project's validation-best checkpoint rule.

Therefore this task reviews MultiDAG+CL as a potentially lower-risk graph and
causal-history candidate, not as an MM-DFN port continuation.

## 4. Repository snapshot

| Item | Result |
|---|---|
| Repo URL | `https://github.com/vanntc711/MultiDAG-CL` |
| Commit hash | `59a75877065a91bf9388fbb564607fe79717fd4f` |
| README exists | Yes: `README.md` |
| requirements exists | No `requirements.txt`; versions are listed only in README |
| Train entry | `run.py` |
| Dataset entry | `dataset.py`, `dataloader.py` |
| Model entry | `model.py`, `model_utils.py` |
| IEMOCAP support | Yes, documented and wired through `dataset_name='IEMOCAP'` |
| MELD support | Yes, documented and accepted by code paths |

The repository was shallow-cloned into:

```text
tmp/multidag_cl_isolated_review/MultiDAG-CL
```

No official data files were downloaded.

README-listed environment:

```text
Python 3.6
PyTorch 1.6.0
Transformers 3.5.1
CUDA 10.1
```

No PyTorch Geometric dependency was found in the reviewed source files.

## 5. Expected input format

The official code reads JSON feature files rather than the current project
dataset objects:

```text
../data/<dataset_name>/<split>_data_roberta_mm.json.feature
```

It also reads vocab files:

```text
../data/<dataset_name>/speaker_vocab.pkl
../data/<dataset_name>/label_vocab.pkl
```

Each dialogue is a list of utterance dictionaries. The reviewed code expects:

```text
u['text']
u['label']
u['speaker']
u['cls']
```

The default multimodal feature construction is manual concatenation in
`dataset.py`:

```text
features.append(u['cls'][0] + u['cls'][1] + u['cls'][2])
```

The README gives IEMOCAP all-modality dimensions as:

```text
2948 = 1024 text + 1582 audio + 342 visual
```

The collate output is already batch-first:

```text
features        [B, N, D_concat]
labels          [B, N], padded with -1
adj             [B, N, N]
s_mask          [B, N, N]
s_mask_onehot   [B, N, N, 2]
lengths         [B]
speakers        [B, N], padded with -1
utterances      list
```

`adj[:, i, :]` represents directed predecessors for utterance `i`. The active
adjacency builder, `get_adj_v1`, links each utterance to previous utterances and
stops after `windowp` same-speaker predecessors. The parser has `windowf`, but
the reviewed active collate path does not use future edges.

## 6. Gap to current project batch contract

MultiDAG+CL is closer to the current project contract than MM-DFN in tensor
layout, because it already uses batch-first dialogue tensors. The main gap is
that it expects one concatenated feature tensor rather than three separate
modalities.

Required adapter from current project batches:

1. Keep `text_features`, `audio_features`, and `visual_features` as separate
   inputs at the project boundary.
2. Optionally apply modality dropout or reliability gates before concatenation.
3. Concatenate selected modalities into `[B, T, D_concat]`.
4. Convert `speaker_ids_int [B, T]` into `speakers [B, T]`, `s_mask [B, T, T]`,
   and `s_mask_onehot [B, T, T, 2]`.
5. Build a directed past adjacency from `attention_mask [B, T]` and
   `speaker_ids_int [B, T]`.
6. Convert `attention_mask` to `lengths [B]`.
7. Return logits as `[B, T, C]` and use masked loss with labels padded or masked
   at invalid positions.
8. Configure dimensions from YAML rather than hardcoding README values.

Important dimension issue:

```text
Official MultiDAG+CL IEMOCAP README text dimension: 1024
Current project IEMOCAP official adapter text dimension: 100
```

This is manageable through a project-native `emb_dim`, but it prevents copying
the official command lines unchanged.

## 7. Dependency and GPU risk

Dependency risks:

1. No `requirements.txt` exists.
2. README targets Python 3.6, PyTorch 1.6.0, Transformers 3.5.1, and CUDA 10.1.
3. The current local project environment has `torch 2.6.0+cpu` but no
   `transformers`.
4. `model.py`, `run.py`, `evaluate.py`, and `dataloader.py` import
   `transformers` at module import time.
5. `run.py` and `evaluate.py` set `CUDA_VISIBLE_DEVICES='0'`.
6. `run.py`, `evaluate.py`, and `trainer.py` use direct `.cuda()` calls.

GPU risk on V100 16G:

1. The graph core is feature-level and should be feasible on V100 16G.
2. No PyG dependency was found, reducing the largest MM-DFN-style environment
   risk.
3. The active DAG computation is sequential over utterances and uses dense
   `[B, T, T]` adjacency/masks, so long M3ED dialogues still need a small memory
   smoke before formal remote training.
4. The larger risk is not V100 memory; it is official-script reproducibility and
   protocol mismatch.

## 8. Fake forward / import check

Checks performed:

| Check | Result |
|---|---|
| Shallow clone | Passed into `tmp/multidag_cl_isolated_review/MultiDAG-CL` |
| Commit hash | `59a75877065a91bf9388fbb564607fe79717fd4f` |
| Syntax compile | Passed for `run.py`, `evaluate.py`, `dataloader.py`, `dataset.py`, `model.py`, `model_utils.py`, `trainer.py`, `cl.py`, `utils.py` after rerunning outside the sandbox because Windows denied the sandboxed `.pyc` atomic rename |
| Official `model.py` import | Failed in current env: `ModuleNotFoundError: No module named 'transformers'` |
| Limited DAG core fake forward | Passed with an in-memory `transformers` stub, random concatenated features, random labels, CPU torch 2.6 |

Limited fake-forward result:

```text
limited_fake_forward_logits (2, 5, 6)
limited_fake_forward_loss_finite True
```

Interpretation:

The unmodified official import does not pass in the current environment because
`transformers` is missing. The limited fake forward does not claim the official
repository is directly runnable; it only shows that the feature-level
`DAGERC_fushion` core can execute a random forward/loss/backward under the
current torch version when the unused BERT imports are bypassed.

## 9. Fit as main development baseline

Verdict: Maybe.

Positive fit:

1. It directly models directed past dialogue structure, which aligns with the
   causal-history/window research direction.
2. It is speaker-aware through speaker ids, same/different speaker masks, and
   directed predecessor adjacency.
3. It is feature-level, not raw-video, MLLM, or end-to-end audiovisual training.
4. It supports IEMOCAP and MELD.
5. It already uses batch-first tensors and padded labels, which is close to the
   current project batch contract.
6. It does not require PyTorch Geometric.
7. The core DAG model passed a limited CPU fake forward with random tensors.

Negative fit:

1. Official scripts are not device-neutral.
2. Official import requires `transformers`, which is absent in the current
   environment.
3. Official data paths are relative to `../data/<dataset>` and do not match the
   current project dataset adapters.
4. Modality choice is done by manually editing `dataset.py`.
5. The official training loop saves `best_model` when `test_fscore` improves,
   not when validation improves.
6. It reports weighted F1/accuracy but not the full project metric set:
   Weighted-F1, Macro-F1, UAR, Accuracy.

Conclusion:

MultiDAG+CL is more suitable than MM-DFN as a near-term causal/DAG graph baseline
candidate. It should not replace MMGCN as the current main baseline until a
project-native, device-neutral port preserves the unified batch contract and
validation-best checkpoint protocol.

## 10. Fit for future modules

| Future module | Fit | Note |
|---|---|---|
| target-user mask | Maybe | Speaker ids and directed adjacency are useful, but target-user masking must be added explicitly at loss/evaluation or adjacency construction time. |
| causal window | Yes | The active graph is directed to previous utterances and has a `windowp` control. A port should keep `windowf=0` for causal settings. |
| modality dropout | Yes | The current project can apply dropout before concatenation. Official code needs a port because it manually concatenates modalities in `dataset.py`. |
| confidence gate | Maybe | A reliability gate is natural before concatenation, but official MultiDAG+CL has no explicit modality reliability branch. |
| unimodal auxiliary heads | Maybe | Possible in a project port if unimodal projections are retained before concat; not present in the official model. |

Overall missing-modality/reliability fit: Maybe. MultiDAG+CL can host these
modules, but only if the port keeps separate modality tensors until the last
controlled fusion point.

## 11. Relation to MMGCN, MM-DFN, and newer baselines

Required relationship statement:

```text
MMGCN: classic graph baseline
MM-DFN: dynamic fusion graph candidate, but engineering risk is high
MultiDAG+CL: suitable as a causal/DAG near-term graph baseline candidate after a device-neutral port
```

Compared with MMGCN:

1. MMGCN remains the stable current anchor.
2. MultiDAG+CL gives a cleaner causal-history/DAG narrative than dense
   multimodal graph propagation.
3. MultiDAG+CL should be added as a separate baseline path, not by mutating
   MMGCN internals.

Compared with MM-DFN:

1. MultiDAG+CL has lower dependency risk because no PyG dependency was found.
2. MultiDAG+CL's official tensor layout is closer to `[B, T, D]`.
3. MM-DFN is stronger for dynamic multimodal fusion, but its official code is
   riskier in the current environment.
4. MultiDAG+CL is better aligned with causal-history/window and speaker-aware
   directed relation experiments.

Compared with SDT and newer non-graph baselines:

1. SDT remains useful as a non-graph transformer/self-distillation comparison.
2. MultiDAG+CL is the better candidate when the paper story needs directed
   graph/history structure.
3. Reliability and missing-modality modules should be project-owned, not copied
   from official scripts.

## 12. Recommendation

Main development baseline answer:

```text
Maybe
```

Recommended integration style:

```text
C. Project-side device-neutral port + current unified batch contract
```

Why C instead of running the official scripts:

1. It preserves the current project `[B, T, D]` batch contract.
2. It avoids changing `datasets/` or the existing MMGCN collate path.
3. It removes direct `.cuda()` calls and `CUDA_VISIBLE_DEVICES` hardcoding.
4. It avoids depending on the old README environment as the project runtime.
5. It keeps modality tensors separate long enough to support dropout, confidence
   gating, and auxiliary heads.
6. It enforces validation-best checkpoint selection instead of test-best model
   saving.
7. It can report the required project metrics: Weighted-F1, Macro-F1, UAR, and
   Accuracy.

Top 3 risks:

1. Protocol risk: official `run.py` saves the best model by `test_fscore`,
   which conflicts with validation-best checkpoint selection.
2. Dependency/device risk: no `requirements.txt`, old README environment,
   missing local `transformers`, `CUDA_VISIBLE_DEVICES='0'`, and direct
   `.cuda()` calls.
3. Modality/data risk: official data uses concatenated JSON features under
   `../data`, manual modality edits in `dataset.py`, and text dimensions that do
   not match the current IEMOCAP official adapter.

## 13. Next action

Recommended next action:

```text
MultiDAG+CL device-neutral port
```

Narrow scope for that future task:

1. Add an isolated project-native MultiDAG+CL model wrapper under a separated
   baseline path.
2. Accept the current batch fields directly:

```text
text_features, audio_features, visual_features, attention_mask, speaker_ids_int
```

3. Keep all dimensions, active modalities, `windowp`, dropout, and optional
   future module switches YAML-controlled.
4. Build directed speaker-aware adjacency inside the wrapper or a small helper.
5. Return padded logits `[B, T, C]` and masked loss.
6. Add CPU fake forward/loss/backward before any local smoke training entry.
7. Add a local smoke training entry only after the isolated fake-forward path is
   stable.

No blocker was found for the general "local edit + remote V100 training"
workflow. The blocker is specific to the official MultiDAG+CL scripts: they
should not be used for formal remote training until the device, path, modality,
and checkpoint-selection issues are removed by a project-native port.
