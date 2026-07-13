# DialogueGCN official source audit

Audit date: 2026-07-13

## Evidence and scope

- Paper: [DialogueGCN, EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1015/), DOI `10.18653/v1/D19-1015`.
- Author-team repository: [declare-lab/conv-emotion](https://github.com/declare-lab/conv-emotion).
- Audited commit: `6128ca20e9c736605cce7e99d5d95db0356c35f5`.
- Audited directories: `DialogueGCN/` and `DialogueGCN-mianzhang/`.
- Local read-only audit copy: `tmp/source_audit/conv-emotion` (ignored by Git).

The paper's reusable core is an utterance graph whose relations encode speaker pairs and temporal direction, learned edge attention, relational graph convolution, a second graph convolution, and optional nodal attention. The released implementations are offline full-dialogue models, not streaming causal models.

## Original `DialogueGCN/` path, organized by class/def

| File | class / def | Input tensor(s) | Output tensor(s) | Main computation | Time mixing / future utterance | Bidirectional sequence encoder | Full-dialogue attention or normalization | Future labels | Test selects best | Direct migration suitability |
|---|---|---|---|---|---|---|---|---|---|---|
| `model.py` | `MaskedEdgeAttention.forward` | sequential features `[T,B,2D_e]`, lengths, edge pairs | edge weights `[B,T,T]` | Linear score, full-time softmax, `1e-10`/`1` edge mask, re-normalization | Future neighbors are legal when configured; not exact mask-first | No | Yes, pre-mask time softmax | No | No | Replace with legal-neighbor masked softmax |
| same | `edge_perms` | length, past/future windows | `(j,item)` pairs | Windowed past/current/future edges | Defaults in IEMOCAP script are past 10 and future 10 | N/A | No | No | No | Window meaning reusable; orientation must be redefined |
| same | `batch_graphify` | encoded utterances, speakers, lengths, windows | flat nodes, PyG edge index/weight/type | Disconnected dialogue batching and relation construction | Contains both temporal directions | Upstream | Uses attention computed over the full tensor | No | No | Not direct: PyG and edge convention |
| same | `DialogueGCNModel.__init__` | base encoder and graph dimensions | modules | DialogRNN forward/reverse pair, or BiLSTM/BiGRU, edge attention, RGCN+GraphConv | Stores `window_future` | **Yes** for default LSTM/GRU; explicit reverse DialogRNN otherwise | Optional nodal attention | No | No | Requires replacement |
| same | `DialogueGCNModel._reverse_seq` | sequence and mask | reversed valid prefixes | Reverses each dialogue for backward DialogRNN | Explicit future-to-current context | Implements backward path | No | No | No | Prohibited in causal version |
| same | `DialogueGCNModel.forward` | `U [T,B,D_m]`, speaker one-hot, mask, lengths | flat log probabilities and graph tensors | Bidirectional context, graphify, graph network, classifier | Future enters through encoder and edges | **Yes** in standard choices | Optional downstream nodal attention | No | No | Not direct |
| same | `GraphNetwork.forward` | nodes, PyG edges/relations/weights | node log probabilities | `RGCNConv` then `GraphConv`, concatenate input and graph output | Both layers consume the supplied future edges | No | Optional nodal classifier | No | No | Paper core retained only in reimplemented directed form |
| same | `attentive_node_features` | flat graph features, lengths, utterance mask | `[T,B,H]` attentive features | For every target `t`, attends over all dialogue nodes with padding mask only | **Yes**, reads `t+1...T` | No | **Yes**, full-dialogue nodal attention | No | No | Disabled in first project version |
| same | `classify_node_features` | graph features and nodal flag | log probabilities | Nodal-attention or same-node classifier | Noncausal only when nodal path enabled; graph features are already noncausal | No | Optional full-dialogue attention | No | No | Same-time classifier only is portable |
| `dataloader.py` | `IEMOCAPDataset.__getitem__` | PKL dialogue | text/visual/audio/speaker/mask/labels | Loads all three stored modalities | No transformation-level guard | No | No | No constructed future labels | No | Data schema informative; not copied |
| `train_IEMOCAP.py` | `train_or_eval_graph_model` | data loader | loss/metrics/edge dumps | Calls graph model with **text only** (`model(textf,...)`) | Model is offline | Upstream | Batch-complete | Same-position labels | Caller does | Shows original IEMOCAP model is not T/A/V fused |
| same | main epoch loop | train/empty-valid/test loaders | printed metrics | `valid=0.0`, evaluates Test every epoch, reports `max(all_fscore)` | N/A | N/A | N/A | N/A | **Yes, reported maximum is Test F1 across epochs** | Prohibited by project protocol |

## `DialogueGCN-mianzhang/` maintained reimplementation

| File | class / def | Input / output | Time and protocol finding | Migration finding |
|---|---|---|---|---|
| `dgcn/model/SeqContext.py` | `SeqContext.__init__/forward` | padded text features -> contextual utterances | Both LSTM and GRU are `bidirectional=True` | Must be replaced by a forward-only encoder |
| `dgcn/model/functions.py` | `edge_perms` | length/windows -> edge pairs | Same past/future window semantics and both temporal directions | Causal project keeps only `source<=target` |
| same | `batch_graphify` | contextual features/speakers -> PyG graph | Uses `2*S^2` relations; same pair orientation issue | Replace with `adj[target,source]` and `S^2` relations |
| `dgcn/model/EdgeAtt.py` | `EdgeAtt.forward` | legal neighbor features -> edge weights | Extracts a legal neighbor set before softmax, unlike the older root implementation; future neighbors still exist when `wf>0` | Mask-first pattern is useful, implementation still depends on offline edges/PyG |
| `dgcn/model/GCN.py` | `GCN.forward` | PyG graph -> graph features | RGCN then GraphConv over both directions | Reimplement in pure PyTorch without reverse edges |
| `dgcn/model/DialogueGCN.py` | `DialogueGCN.forward` | context features and batch metadata -> classifier input | Calls bidirectional context/graph path | Architectural organization is informative, not copied line-for-line |
| `dgcn/model/Classifier.py` | `MaskedEmotionAtt.forward` | graph features -> per-node attention result | Softmax spans a dialogue mask without a causal triangle | Full nodal attention must stay disabled |
| `dgcn/Coach.py` | `Coach.train` | train/dev/test datasets | Selects `best_state` by **dev F1**, although it logs Test every epoch | Selection policy is acceptable; per-epoch Test logging is not adopted |
| `preprocess.py` | `split` | official PKL IDs -> train/dev/test lists | Creates a dev split from training IDs and retains the test list | Current project Session-Holdout logic is not modified or replaced |

## Required focused findings

### Encoder and modalities

- Root `DialogueGCN/train_IEMOCAP.py` defaults to a 2-layer bidirectional LSTM with `D_m=100`.
- DialogRNN mode separately runs forward and reversed sequences; GRU mode is also bidirectional.
- The dataloader exposes text, audio, and visual features, but the graph training function calls the model with `textf` only. Thus the original root IEMOCAP graph model is text-only in the executed path.
- A project T/A/V adapter is necessarily a project extension, not an official multimodal DialogueGCN reproduction.

### Edges, direction, and relations

- `window_past=10`, `window_future=10` are the root IEMOCAP defaults. `-1` means unbounded on the corresponding side.
- Edge pairs are authored as `(j,item)` and treated by attention as target/neighbour, but written unchanged into PyG `edge_index`, whose standard convention is source/target. Removing future neighbors without transposing this convention would still leak later `j` into earlier `item`.
- Relations are `(source speaker, target speaker, temporal direction)`, totaling `2*S^2` or 8 for IEMOCAP's two speakers.
- Once only past/current sources are legal, the future direction bit is removed. The project relation count becomes `S^2` or 4, with self-loops using their ordinary same-speaker pair relation.
- `RGCNConv` and `GraphConv` consume the provided directed edges. No evidence supports automatically mapping future relations into past relations in a causal port.

### Edge and nodal attention

The older root edge attention computes full-time softmax before applying its graph mask. The Mianzhang reimplementation improves this by selecting a legal neighbor list before softmax, but its graph still includes future neighbors by default. Root and Mianzhang nodal attention both span the complete valid dialogue rather than `0...t`; the first causal project version therefore disables nodal attention and classifies the same-time graph representation.

### Training selection and dependencies

- Root IEMOCAP training uses no effective validation split, reads Test every epoch, and reports the maximum Test F1.
- Mianzhang selects the checkpoint by dev F1 but still reports Test each epoch.
- The root implementation depends on an old PyTorch/PyG stack and uses deprecated call signatures. Exact compatibility with the current Python 3.9 environment is **UNCONFIRMED** and no new PyG dependency is introduced.
- Upstream text/audio/visual extractor causality and current real-PKL compatibility remain **UNCONFIRMED** locally.

## Verdict

**Official DialogueGCN is not model-level strict causal.** Demonstrated future paths are the bidirectional or explicitly reversed sequential encoder, future graph neighbors, both temporal relation directions, and optional full-dialogue nodal attention. The root training protocol also selects its reported epoch from Test F1.

The project causal version replaces those paths while retaining the paper's speaker-pair relational graph, learned edge attention, relational message passing, residual node representation, and utterance classification. Because it adds a project T/A/V adapter and a pure-PyTorch RGCN, it is a **DialogueGCN-derived project implementation**, named **Project causal DialogueGCN**, not a line-by-line official reproduction.
