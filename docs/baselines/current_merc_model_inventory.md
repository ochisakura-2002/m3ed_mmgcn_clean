# Current MERC model inventory

Audit date: 2026-07-13

## Counting rule

A model family is counted only when a project implementation exists under
`models/`. A model is counted as runnable end-to-end only when project code has
a train -> checkpoint -> evaluate path. Synthetic-only forward ports are listed
but are not counted as runnable experiment baselines.

Current counts:

- Project model implementations: **6** (`MMGCN`, project MultiDAG-inspired,
  Project causal GS-MCC-inspired, Project causal DialogueGCN, `SDT`,
  `SimpleMLP`).
- Train/checkpoint/evaluate code paths: **5** (`MMGCN`, project
  MultiDAG-inspired, Project causal GS-MCC-inspired, Project causal
  DialogueGCN, `SimpleMLP`).
- Synthetic-only candidates: **1** (`SDT`).
- Current `third_party/`: absent in this checkout.

## Inventory

| Model / variant | Project implementation | Train entry | Eval entry | Existing configs | Dataset / modalities | Formal result evidence | Causal parameter and actual destination | Bidirectional RNN | Future edges | Full-dialogue attention | Time-axis global pooling | Test selects checkpoint | Current runnable state | Preliminary causal grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MMGCN default/full | `models/baselines/mmgcn/mm_gcn.py::M3EDMMGCN`, `dense_graph.py` | `scripts/train_mmgcn.py` | `scripts/evaluate_checkpoint.py` | `configs/train_mmgcn_iemocap_official.yaml`, M3ED and pipeline YAMLs | M3ED, IEMOCAP, synthetic smoke; T/A/V combinations | **Unconfirmed**: no local formal run artifact; old notes only mention an approximate comparison | `graph.context_mode/window_*` -> train/eval builder -> model -> `build_official_like_multimodal_adjacency`; formal IEMOCAP uses `full` | No | Yes in `full` | No | No | No; validation monitor only | Code path complete; real PKL absent locally | `noncausal` |
| MMGCN causal | Same implementation, explicit causal graph branch | Same | Same | M3ED causal skeleton plus 5 new IEMOCAP causal benchmark YAMLs | M3ED/IEMOCAP-shaped T/A/V | No formal IEMOCAP causal result yet | Same parameter chain; `context_mode=causal` rejects `source_time > target_time` before normalization | No | No | No | No | No | Synthetic audit passed; real-batch audit pending remote PKL | `strict_model_causal` |
| Project MultiDAG-inspired (`MultiDAGCLBaseline`) | `models/baselines/multidag_cl/multidag_cl_model.py` | `scripts/baselines/train_multidag_cl.py` | `scripts/baselines/evaluate_multidag_cl_checkpoint.py` | IEMOCAP debug/formal/stabilize/loss-stability plus 5 causal benchmark YAMLs | Formal entry: IEMOCAP; synthetic M3ED/IEMOCAP-shaped; all 7 T/A/V subsets | Documents report prior context-w5 results, but no local output artifact exists to re-verify exact metrics | `graph.context_mode/window_past` -> effective history window; `model.modality_encoder_type` -> causal GRU/linear; no standalone official `causal` API | No; GRU has `bidirectional=False` | No; adjacency is `j <= i` | No | No | No; validation selector only | Train/eval/pipeline path complete; synthetic audit passed | `strict_model_causal` |
| Project causal GS-MCC-inspired (`CausalGSMCCInspiredBaseline`) | `models/baselines/gsmcc/causal_gsmcc_model.py`, `causal_spectral_ops.py`, `causal_gsmcc_losses.py` | `scripts/baselines/train_new_causal_graph_baseline.py` | `scripts/baselines/evaluate_new_causal_graph_checkpoint.py` | 5 formal IEMOCAP causal benchmark YAMLs plus synthetic/real-data smoke YAMLs | Formal entry: IEMOCAP T/A/V; deterministic synthetic dialogue smoke | **UNCONFIRMED**: no real IEMOCAP performance run; synthetic values are not results | Validated config requires `context_mode=causal`, `window_future=0`, non-bidirectional encoder; `window_past` -> shared `adj[target,source]` -> directed polynomial filter | No; optional GRU has `bidirectional=False` | No; source time is always `<=` target time | No | No | No; best checkpoint uses validation Weighted-F1 and Test runs only after reload | Train/checkpoint/evaluate code path complete; synthetic end-to-end smoke passed; real-batch audit and formal IEMOCAP performance pending | `strict_model_causal` on synthetic audit; real batch and upstream features unverified |
| Project causal DialogueGCN (`CausalDialogueGCNBaseline`) | `models/baselines/dialoguegcn/causal_dialoguegcn_model.py`, `causal_dialoguegcn_graph.py`, `relational_graph_conv.py` | `scripts/baselines/train_new_causal_graph_baseline.py` | `scripts/baselines/evaluate_new_causal_graph_checkpoint.py` | 5 formal IEMOCAP causal benchmark YAMLs plus synthetic/real-data smoke YAMLs | Formal entry: IEMOCAP T/A/V; deterministic synthetic dialogue smoke | **UNCONFIRMED**: no real IEMOCAP performance run; synthetic values are not results | Validated config requires `context_mode=causal`, `window_future=0`, stepwise forward-only context encoding, speaker-pair graph, and mask-first edge attention | No; default GRU has `bidirectional=False` | No; no future-direction relation exists | No; nodal attention disabled | No | No; best checkpoint uses validation Weighted-F1 and Test runs only after reload | Train/checkpoint/evaluate code path complete; synthetic end-to-end smoke passed; real-batch audit and formal IEMOCAP performance pending | `strict_model_causal` on synthetic audit; real batch and upstream features unverified |
| SDT project port | `models/baselines/sdt/sdt_model.py::SDTBaseline` | None | None | None | Synthetic M3ED/IEMOCAP-shaped TAV | None | No causal parameter | No BiRNN found | N/A | Yes: MultiheadAttention has padding mask but no causal mask | No separate global pool required; full attention already leaks | N/A | Fake forward only | `noncausal` |
| SimpleMLP | `models/baselines/simple_mlp.py::M3EDConcatMLP` | `scripts/train_simple_mlp.py` | Shared `scripts/evaluate_checkpoint.py` | `configs/train_simple_mlp_m3ed.yaml` | M3ED TAV | No formal result; config describes a debug/sanity baseline | No causal parameter; classifier is independent per utterance | No | No graph | No | No | No; validation monitor | Code path complete | `strict_model_causal` |
| Official MultiDAG | No implementation in current checkout | None | None | None | Unknown in project | None | Unable to inspect locally | Unconfirmed | Unconfirmed | Optional behavior only known from external source, not a project implementation | Unconfirmed | Original external workflow not adopted | Not runnable | `unable_to_determine` |
| Official MultiDAG+CL | No implementation in current checkout; project model is not equivalent | None | None | None | Unknown in project | None | Unable to inspect locally | Unconfirmed | Unconfirmed | External variants may include global nodal attention | Unconfirmed | External original workflow uses test F-score; prohibited here | Not runnable | `unable_to_determine` |
| MM-DFN | No implementation; isolated review document only | None | None | None | N/A | Import/core forward previously reported blocked; no project result | No project causal path | Unconfirmed | Unconfirmed | Unconfirmed | Unconfirmed | External protocol concerns are not a project checkpoint path | Not runnable | `unable_to_determine` |
| DialogueRNN | No project implementation | None | None | None | N/A | None | N/A | Unconfirmed | N/A | Unconfirmed | Unconfirmed | N/A | Not runnable | `unable_to_determine` |
| Official DialogueGCN (external) | No official implementation copied into project; isolated project-derived candidate is listed above | None | None | None | Original root IEMOCAP path is text-only | None | Official audit finds bidirectional context, past/future edges, and 8 relations | Yes | Yes | Optional full-dialogue nodal attention | No separate pooling | Root script reports max Test F1 across epochs | Not runnable as official source in project | `noncausal` by source audit |
| DAG-ERC | No independent project implementation | None | None | None | N/A | None | N/A | Unconfirmed | Unconfirmed | Unconfirmed | Unconfirmed | N/A | Not runnable | `unable_to_determine` |
| Independent GRU/LSTM baseline | None; MultiDAG's internal GRU is not a separate baseline | None | None | None | N/A | None | N/A | N/A | N/A | N/A | N/A | N/A | Not runnable | `unable_to_determine` |

## Important naming boundaries

1. `MultiDAGCLBaseline` is kept as a compatibility class/config name. Reports
   must call it **project MultiDAG-inspired**, because it uses a project-specific
   modality encoder, window rule, graph layer, and CE-only training.
2. There is no official MultiDAG or official MultiDAG+CL source tree in the
   current checkout.
3. The absence of `third_party/` also means the configured IEMOCAP PKL cannot be
   inspected or used for a local real-batch audit in this checkout.
4. Every PKL-consuming causal model retains the feature status
   `utterance_level_but_extractor_not_fully_verified`.
5. `CausalGSMCCInspiredBaseline` must be called **Project causal GS-MCC-inspired**.
   Its directed polynomial filter and auxiliary losses are project mechanisms,
   not an official GS-MCC FGO/contrastive reproduction.
6. `CausalDialogueGCNBaseline` must be called **Project causal DialogueGCN** and
   described as a DialogueGCN-derived project implementation because its T/A/V
   adapter and pure-PyTorch RGCN are project code.
