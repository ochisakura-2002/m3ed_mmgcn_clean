# MERC baseline rescreen report

Date: 2026-07-07

This report is a research and engineering feasibility screen only. It does not integrate a new baseline, modify the current MMGCN training/evaluation flow, or inspect any local GS-MCC code.

## Scope and constraints

- Primary project setting: feature-level multimodal emotion recognition in conversation on `M3ED`, with `MMGCN` as the current deployed baseline.
- Target hardware: V100 16G for formal training; local Windows is for editing, smoke checks, CSV analysis, and lightweight inspection only.
- Preferred input contract:

```text
text_features   [B, T, D_text]
audio_features  [B, T, D_audio]
visual_features [B, T, D_visual]
labels          [B, T]
attention_mask  [B, T]
speaker_ids_int [B, T]
```

- Excluded as default choices: MLLM-based MERC, end-to-end video models, large generative/diffusion recovery models, raw-video-heavy methods, and methods whose code is not verifiably released.
- Online verification was performed on 2026-07-07 for public paper pages and GitHub repositories. When runnable code could not be verified, the candidate is not treated as deployable.

## Local project context

Quick context files read:

| File | Status | Relevant note |
|---|---|---|
| `AGENTS.md` | Exists | Project entry rules, no full scan, report-only task constraints, known smoke changes. |
| `docs/project_map.md` | Exists | Current project map, entry scripts, dataset/model responsibilities. |
| `docs/codex_workflow.md` | Exists | Local/remote workflow and no-commit rule. |
| `docs/experiment_protocol.md` | Exists | Metrics, missing-modality vs from-scratch ablation distinction, checkpoint selection rules. |
| `docs/module_implementation_spec.md` | Exists | Batch shape contract, MMGCN internal shape expectations, YAML switch expectations for future modules. |
| `docs/smoke_test_protocol.md` | Exists | Local smoke-test boundaries and fake-dialogue smoke policy. |
| `docs/git_workflow.md` | Exists | Git safety rules and data/checkpoint exclusion rules. |

Git snapshot requested by task:

```text
branch: main
HEAD:   0670c2a
status:
 M scripts/evaluate_checkpoint.py
 M scripts/train_mmgcn.py
?? AGENTS.md
?? configs/smoke/
?? datasets/smoke/
?? docs/baselines/
?? docs/codex_workflow.md
?? docs/experiment_protocol.md
?? docs/git_workflow.md
?? docs/module_implementation_spec.md
?? docs/project_map.md
?? docs/smoke_test_protocol.md
```

Current `git diff --stat` before writing this report showed only:

```text
scripts/evaluate_checkpoint.py | 16 ++++++++++++++--
scripts/train_mmgcn.py         | 15 +++++++++++++--
```

These two modified scripts match the known local smoke-test changes described in `AGENTS.md`; this report does not revert, commit, or repair them.

### Current model entry points

- Training entry: `scripts/train_mmgcn.py`.
- Checkpoint evaluation entry: `scripts/evaluate_checkpoint.py`.
- Main MMGCN implementation: `models/baselines/mmgcn/mm_gcn.py`.
- Current simple sanity baseline: `models/baselines/simple_mlp.py`.
- Current train script supports `dataset.name` values `M3ED`, `IEMOCAP`, and `MMGCN_SMOKE`.
- Current evaluation script supports `SimpleMLP` and `MMGCN`, and has an evaluation-time `--active-modalities` override for MMGCN.

### Current data interface

- M3ED data enters through `datasets/m3ed/torch_dataset.py` plus `datasets/collators/m3ed_collate.py`.
- IEMOCAP official MMGCN-style features enter through `datasets/iemocap/official_feature_dataset.py`.
- The collate output is padded dialogue-level feature tensors with `attention_mask`, `lengths`, and `speaker_ids_int`.
- Current M3ED default config uses feature dimensions `text=768`, `audio=1024`, `visual=342`; IEMOCAP official adapter documents `text=100`, `audio=1582`, `visual=342`.

### Key constraints for new baseline integration

1. The baseline should consume utterance-level pre-extracted features, not raw videos or raw audio.
2. It should accept or be cheaply adapted to padded `[B, T, D]` dialogue batches.
3. It should preserve a baseline-equivalent path when disabled or when run with simple full-modality settings.
4. It should not force changes to `datasets/`, the current MMGCN collate contract, or existing checkpoint compatibility.
5. It should make validation-selected checkpoint evaluation possible with `Weighted-F1`, `Macro-F1`, `UAR`, and `Accuracy`.
6. It should be feasible on V100 16G without MLLM, end-to-end video encoders, large diffusion models, or hard-to-reproduce preprocessing.

### Interface risks found

- Many public MERC repos use dialogue-flattened graph batches or concatenated feature vectors instead of the current padded `[B, T, D]` contract. This is manageable, but it means a small adapter layer is needed.
- Several graph candidates depend on PyTorch Geometric and older CUDA/PyTorch versions. This is more of a dependency/reproducibility risk than a pure GPU-memory risk.
- Several recent papers claim code availability, but the linked repository is either unreleased, very new, data-link-dependent, or incomplete.
- Label spaces differ across M3ED, IEMOCAP, and MELD; any new baseline smoke test must first verify label mapping and metrics before comparing results.
- The current project has no generic model registry for arbitrary baselines. A future integration should add a separate, narrow training entry or a model factory rather than bending `MMGCN` internals.

## Candidate screen

| Candidate | Year/Venue | Task | Code URL | Official? | Dataset Support | Input Format | GPU Risk on V100 16G | Adaptation Cost | Authority/Reliability | Fit to Our Project | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SDT | 2024 TMM | Transformer-based MERC with self-distillation | https://github.com/butterfliesss/SDT | Yes | IEMOCAP, MELD | Preprocessed utterance-level multimodal features | Medium | Medium | Strong | Best balance of newer method, official runnable code, IEMOCAP/MELD support, and feature-level deployment | Priority baseline |
| MM-DFN | 2022 ICASSP | Multimodal ERC/MERC, dynamic graph fusion | https://github.com/zerohd4869/MM-DFN | Yes | IEMOCAP, MELD | Utterance-level text/audio/visual features, DialogueRNN/MMGCN-style processed features | Low | Medium | Strong | Very close to current feature-level setup; good MMGCN replacement family, though older than SDT | Priority baseline |
| MultiDAG+CL | 2024 LREC-COLING | Multimodal ERC with directed acyclic graph and curriculum learning | https://github.com/vanntc711/MultiDAG-CL | Yes | IEMOCAP, MELD | Concatenated multimodal utterance features; README documents text/audio/visual dimensions | Low | Medium | Moderate | Good causal/window-oriented direction; DAG structure is aligned with future causal/online analysis | Priority baseline |
| COGMEN | 2022 NAACL | Contextualized GNN-based multimodal emotion recognition | https://github.com/Exploration-Lab/COGMEN | Yes | IEMOCAP, MOSEI | Concatenated utterance-level audio/text/video features plus speaker graph | Medium | Medium | Strong | Solid graph baseline and clear code, but dataset support is less aligned with MELD/M3ED and PyG is required | Backup baseline |
| DnR | 2026 WACV/arXiv | Plug-and-play divide/refine representation framework for MERC | https://github.com/mattam301/DnR-WACV2026 | Yes | IEMOCAP, MELD, MOSI, MOSEI, UR-FUNNY, MUSTARD | Feature-level text/audio/visual with supported backbones including MMGCN and MM-DFN | Medium | High | Moderate | Interesting for reliability/missing-modality style work, but it is a plug-in framework rather than a clean standalone baseline | Backup baseline |
| VEGA | 2025 ACM MM | CLIP visual-anchor MERC | https://github.com/dkollias/VEGA | Yes | Paper: IEMOCAP, MELD; README data path emphasizes IEMOCAP | Multimodal features plus CLIP visual anchor features/assets | Medium | High | Moderate | Relevant for reliability/visual grounding, but CLIP anchors and extra assets violate the simplest feature-level baseline goal | Related work only |
| DGDA-Net | 2026 arXiv | Cross-scenario graph domain adaptation for MERC | https://github.com/Xudmm1239439/DGDA-Net | Yes | IEMOCAP, MELD | Preprocessed features with pretraining/domain-adaptation flow | Medium | High | Weak | Useful for future target-user/domain-shift setting, but too new, domain-adaptation-specific, and rough as the next default baseline | Related work only |
| SSLCL | 2023 arXiv | Model-agnostic supervised contrastive learning for ERC | https://github.com/TaoShi1998/SSLCL | Partly | IEMOCAP, MELD | Loss/framework over sample features and labels, not a complete MERC model pipeline | Low | Medium | Weak | Could become an auxiliary loss later, but it is not a deployable standalone baseline | Related work only |
| GS-MCC | 2024 arXiv | Graph-spectrum MERC consistency/complementarity learning | Not verified in this pass | Unknown | Paper says two benchmark datasets; exact deployable support not verified here | Sliding-window multimodal interaction graph, code not locally present | Medium | High | Unknown | Interesting graph-spectrum direction, but this round must not assume local or complete code | Exclude for now |
| CMATH | 2024 arXiv | Cross-modality augmented transformer with hierarchical variational distillation | https://github.com/cjw-MER/CMATH | Claimed, but repo only says code will be released | IEMOCAP, MELD | Cross-modal transformer/reconstruction/distillation; no runnable code found | Medium | High | Weak | Good idea but not deployable now because repository is effectively unreleased | Exclude for now |
| HAUCL | 2024 arXiv | Hypergraph autoencoder and contrastive learning for MERC | Paper links `https://github.com/yzjred/-HAUCL`, but it was unavailable in this pass | Claimed, not verified | IEMOCAP, MELD | Dynamic hypergraph + contrastive learning; code URL could not be opened/verified | Medium | High | Unknown | Recent and relevant, but code availability is not reliable enough for default baseline selection | Exclude for now |
| GraphSmile | 2024 arXiv | Graph structure plus sentiment dynamics for MERC/MSAC | No public code verified in this pass | Unknown | Multiple benchmarks, exact runnable support not verified here | Graph-based multimodal dialogue features plus auxiliary sentiment dynamics | Medium | High | Moderate paper, Unknown code | Strong related method, but no verified deployable code in this pass | Exclude for now |

## Candidate rationale

### Why SDT is first priority

SDT is the best default candidate in this screen because it is newer than MMGCN/MM-DFN, accepted by TMM, avoids raw-video or MLLM dependencies, and has an official repository with `requirements.txt`, `train.py`, `dataloader.py`, and IEMOCAP/MELD shell entries. It is less graph-centered than MMGCN, which makes it useful as a genuinely different fusion/context baseline. The main risks are training-loop changes for self-distillation and possible memory sensitivity on long dialogues, but feature-level MERC should still be feasible on V100 16G after a fake-batch forward smoke test.

### Why MM-DFN is second priority

MM-DFN is the closest deployment-shaped replacement family for MMGCN. Its official repository includes `requirements.txt`, train scripts for IEMOCAP/MELD, processed-feature assumptions similar to DialogueRNN/MMGCN, and a method that directly targets redundant graph fusion. The reason it is not first is age and dependency risk: it lists Python 3.6, Torch 1.4, CUDA 10.1, and older PyG packages. For our project, the safer path is to port the model logic into the current environment after a source-level smoke review, not to force the full old environment into this repo.

### Why MultiDAG+CL is third priority

MultiDAG+CL is attractive because DAG-based dialogue modeling maps well to future causal/window/online settings. Its README explicitly shows multimodal feature dimensions and IEMOCAP/MELD runs. The repository is smaller and less mature than SDT/MM-DFN, and the README suggests manual modality changes inside `dataset.py`, so it should be smoke-tested after SDT and MM-DFN.

### Why COGMEN is backup rather than priority

COGMEN is a strong official NAACL 2022 graph baseline and has clear train/eval files. It is useful if we want a speaker-aware graph model as a replacement family. It is not first priority because it targets IEMOCAP/MOSEI rather than IEMOCAP/MELD in the paper/repo flow, and it relies on PyG plus Comet-style experiment infrastructure.

### Why DnR is backup rather than priority

DnR is promising for reliability and representation analysis because it decomposes modality-specific, redundant, and synergistic information and supports MMGCN/MM-DFN backbones. It is not a simple new baseline, though: it adds pretraining/refinement stages and is very new. It is better treated as a second-phase robustness/reliability framework after a main baseline is chosen.

## GS-MCC status

1. The local checkout currently does not contain a downloaded GS-MCC `third_party` repository.
2. This report did not read any local GS-MCC code.
3. This report does not default-select GS-MCC as the main new baseline.
4. GS-MCC is evaluated only from external paper-level information in this round; no conclusion is based on nonexistent local code.
5. If GS-MCC is reconsidered later, it needs an independent smoke-test investigation covering:
   - official repository address,
   - exact commit hash,
   - whether README modules match the paper modules,
   - whether IEMOCAP and MELD are supported by runnable scripts,
   - whether V100 16G is sufficient,
   - whether its graph-spectrum inputs can be adapted to the current `[B, T, D]` batch contract without changing existing MMGCN data flow.

## Recommended baseline shortlist

1. SDT
   - Reason: newer TMM transformer/self-distillation baseline; feature-level and not graph-only, giving a useful alternative to MMGCN-family graph models while staying inside V100-scale constraints.
   - Risk: self-distillation adds training-loop changes; transformer context may be more memory sensitive on long dialogues.
   - Next smoke test: verify official repo commit, inspect dataloader expected pickle schema, and map one fake `[B, T, D_text/audio/visual]` batch into the model forward without touching current MMGCN code.

2. MM-DFN
   - Reason: closest deployable successor to MMGCN; feature-level MERC, official code, IEMOCAP/MELD support, and dynamic fusion directly addresses MMGCN's older dense graph fusion.
   - Risk: old Torch/CUDA/PyG stack; code should be reviewed and ported rather than installed blindly.
   - Next smoke test: clone or inspect the official repo in an isolated review location, record commit hash, verify README/train scripts/requirements, then run a CPU-only fake-feature forward or minimal source-level import check before any integration.

3. MultiDAG+CL
   - Reason: DAG formulation aligns with future causal/window/online experiments and is relatively lightweight.
   - Risk: smaller repository and README suggests manual modality edits; needs careful reproducibility check.
   - Next smoke test: inspect code path for all-three-modality IEMOCAP training, identify the exact point where concatenated features are built, and test whether a wrapper can convert current padded batches into its expected dialogue graph input.

## Source links used

Checked online on 2026-07-07. These links are source evidence only; no third-party code was downloaded into this project during this task.

- MM-DFN paper: https://arxiv.org/abs/2203.02385
- MM-DFN code: https://github.com/zerohd4869/MM-DFN
- SDT paper: https://arxiv.org/abs/2310.20494
- SDT code: https://github.com/butterfliesss/SDT
- MultiDAG+CL paper: https://arxiv.org/abs/2402.17269
- MultiDAG+CL code: https://github.com/vanntc711/MultiDAG-CL
- COGMEN paper: https://arxiv.org/abs/2205.02455
- COGMEN code: https://github.com/Exploration-Lab/COGMEN
- DnR paper: https://arxiv.org/abs/2601.14274
- DnR code: https://github.com/mattam301/DnR-WACV2026
- VEGA paper: https://arxiv.org/abs/2508.06564
- VEGA code: https://github.com/dkollias/VEGA
- DGDA-Net paper: https://arxiv.org/abs/2603.26840
- DGDA-Net code: https://github.com/Xudmm1239439/DGDA-Net
- SSLCL paper: https://arxiv.org/abs/2310.16676
- SSLCL code: https://github.com/TaoShi1998/SSLCL
- GS-MCC paper: https://arxiv.org/abs/2404.17862
- CMATH paper: https://arxiv.org/abs/2411.10060
- CMATH code placeholder: https://github.com/cjw-MER/CMATH
- HAUCL paper: https://arxiv.org/abs/2408.00970
- GraphSmile paper: https://arxiv.org/abs/2407.21536

## Summary counts

- Candidate total: 12
- Priority baseline: 3
- Backup baseline: 2
- Related work only: 3
- Exclude for now: 4
