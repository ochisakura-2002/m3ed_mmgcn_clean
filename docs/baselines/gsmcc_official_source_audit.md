# GS-MCC official source audit

Audit date: 2026-07-13

## Evidence and scope

- Paper: [AAAI-25 GS-MCC](https://ojs.aaai.org/index.php/AAAI/article/view/33242), DOI `10.1609/aaai.v39i11.33242`.
- Author repository: [FuchenZhang/GS-MCC](https://github.com/FuchenZhang/GS-MCC).
- Audited commit: `38c4038a7738f9bf7b3132c3e99a126e1cf1f28d`.
- Local read-only audit copy: `tmp/source_audit/GS-MCC` (ignored by Git).
- Executed entry inferred from imports: `train_fourier.py` imports `DialogueGCNModel` and losses from `FourierGNNmodel.py`.

The paper describes a sliding-window multimodal interaction graph, Fourier graph operators (FGO), low/high-frequency collaboration, and contrastive learning. The repository README calls itself official, but several paper-to-code correspondences are not explicit in the executed source. Those points are marked **UNCONFIRMED** rather than reconstructed from intent.

## Active model path, organized by class/def

| File | class / def | Input tensor(s) | Output tensor(s) | Main computation | Time mixing / future utterance | Bidirectional sequence encoder | Full-dialogue attention or normalization | Future labels | Test selects best | Direct migration suitability |
|---|---|---|---|---|---|---|---|---|---|---|
| `FourierGNNmodel.py` | `MaskedKLDivLoss.forward` | `log_pred [N,C]`, probability target `[N,C]`, mask | scalar | Masked `KLDivLoss` | Inputs are produced by the full model; no independent causal guard | N/A | No time operation inside this def | No class labels; target distribution can contain future context | No | Only the generic masked reduction is portable; this is not an InfoNCE/sample contrastive loss |
| same | `MaskedEdgeAttention.forward` | `M [T,B,H]`, lengths, edge pairs | scores `[B,T,T]` | Linear scores, `softmax(dim=0)` over the full time axis, then an edge mask of `1`/`1e-10`, division by masked sum | Default edge set includes future neighbors; the `1e-10` pre-mask path also prevents an exact mask-first guarantee | No | Yes: softmax is computed before the legal-edge mask | No | No | Replace with mask-before-softmax |
| same | `edge_perms` | length, `window_past`, `window_future` | list of `(j,item)` pairs | Sliding temporal window | `window_future=10` by default; `-1` means unbounded on that side | N/A | No | No | No | Window semantics are reusable, edge orientation is not |
| same | `batch_graphify` | encoded feature tensor, speaker mask, lengths, windows | flat nodes, PyG `edge_index`, edge weights/types | Builds per-dialogue edges and offsets nodes into a batched disconnected graph | Explicit future pairs when `window_future>0` | Upstream | Attention was already computed on the complete padded time tensor | No | No | Not direct: PyG dependency and edge-orientation ambiguity |
| same | `FGN.tokenEmb` | `x [B,N]` | `[B,N,D]` | Elementwise learned embedding | No time awareness | No | No | No | No | Mathematical behavior can be studied, but it is not a causal graph primitive |
| same | `FGN.fourierGC` | complex FFT coefficients | complex coefficients | Three learned complex-valued diagonal transforms and soft shrinkage | Operates on the dimension passed by `FGN.forward`; no adjacency input | No | Global over the transformed axis | No | No | Not copied; no explicit low/high causal branches |
| same | `FGN.forward` | in the active graph path, `x [num_nodes, hidden]` | `[num_nodes, hidden]` | `rfft(x, dim=1)`, learned complex transform, `irfft`, projection MLP | The FFT axis is the hidden-feature axis, not the dialogue-node axis in the active call | No | Full hidden-axis FFT; no graph Laplacian/eigendecomposition | No | No | Not an official-FGO-equivalent causal operator |
| same | `GraphNetwork.forward` | flat nodes, `edge_index`, `edge_norm`, `edge_type` | graph features | PyG `RGCNConv`, then `FGN`, then concatenate residual | Future paths enter through edges and upstream BiLSTM; `edge_norm` is not passed to `RGCNConv` in this file | No | FGN spans each node's hidden axis | No | No | Replace with pure-PyTorch directed polynomial filters |
| same | `DialogueGCNModel.__init__` | dimensions and graph options | modules | Three modality projections, three 2-layer BiLSTMs, three graph networks, gated fusion | `window_future` retained | **Yes**, all three LSTMs set `bidirectional=True` | Optional nodal flag is stored | No | No | Not direct |
| same | `DialogueGCNModel.forward` | text/visual/audio, speaker mask, utterance mask, lengths | modal/fused log probabilities plus graph tensors | Project modalities, BiLSTM encode, build three modality graphs, graph encode separately, gated fusion | Future enters each current representation through BiLSTM and graph edges | **Yes** | The active `GraphNetwork(return_feature=True)` returns before its nodal-attention classifier, so nodal attention is dormant in this path | No | No | Requires replacement, not patching |
| `train_fourier.py` | `train_or_eval_graph_model` | batches from loader | loss/accuracy/F1/predictions | Fused CE + three modal CEs + three KL terms toward fused probabilities | Loss consumes outputs already containing future context | Upstream | Batch-complete evaluation | Labels are same-position labels; no future-label construction found | Caller does | Training loss is informative, but not copied as claimed contrastive sampling |
| same | main epoch loop | train/valid/test loaders | printed metrics and best labels/preds | Runs Train, Validation, and Test every epoch | N/A | N/A | N/A | N/A | **Yes: `best_fscore` is updated from `test_fscore` and final result uses `max(all_fscore)`** | Prohibited by project protocol |

## Supporting-file register

| File | class / def | Inputs / outputs | Audit result |
|---|---|---|---|
| `dataloader.py` | `IEMOCAPDataset.__getitem__` | Stored text/visual/audio/speaker/label -> per-dialogue tensors | Uses the PKL's fixed train/test lists. It does not create future labels, but extractor causality is **UNCONFIRMED**. |
| same | `IEMOCAPDataset.collate_fn` | variable dialogues -> padded tensors | Standard padding; no cross-dialogue edges are created here. |
| `train_fourier.py` | `get_train_valid_sampler` | training dataset -> random train/valid samplers | Holds out 10% of the provided training list. |
| `augs.py` | `aug_random_mask` | node features -> masked features | Not imported by `train_fourier.py`; active-use status is **UNCONFIRMED**. |
| same | `aug_random_edge` | sparse adjacency -> augmented adjacency | Removes/adds edges symmetrically, which would violate directed causality if used. Not imported by the active entry. |
| same | `aug_drop_node`, `aug_subgraph`, `delete_row_col` | features/adjacency -> reduced graph | Structural augmentation helpers; not imported by the active entry. |
| `models.py` | `FGN`, `GraphNetwork`, `DialogueGCNModel` and graph helpers | Older/parallel model definitions | `train_fourier.py` does not import this file. It repeats bidirectional encoders and windowed future graph logic; it is not treated as the executed authority. |

## Required focused findings

### Encoders and feature dimensions

- The executed model creates `linear_t`, `linear_a`, and `linear_v`, followed by three 2-layer LSTMs with `bidirectional=True`.
- Default IEMOCAP dimensions in `train_fourier.py` are text `1024` (`textCNN`), audio `1582` (`IS10`), and visual `342` (`denseface`). The project's current unified IEMOCAP interface uses text `100`, audio `1582`, visual `342`; therefore the text interface is not shape-compatible.
- Argument names for audio/visual are swapped across the call and model signature, then paired with correspondingly named projections. The runtime intent is inferable, but a clean modality-name mapping is **UNCONFIRMED** without executing the unavailable official feature file.

### Edge direction and relation semantics

`edge_perms` creates `(j,item)` where surrounding attention code treats `j` as the target and `item` as its neighbor/source. `batch_graphify` writes this pair unchanged to `edge_index`. PyG's default message-passing convention is `edge_index[0]=source`, `edge_index[1]=target`. Consequently, simply setting `window_future=0` would reverse the intended causal edge: a later `j` would message an earlier `item`. The project version therefore defines its own unambiguous `adj[target,source]` tensor instead of porting these indices.

Relations use `2 * num_speakers**2`: source/target speaker pair plus a temporal-direction bit. For two speakers this gives 8 relations. A strictly past-to-current project graph needs only the 4 speaker-pair relations; retaining a future-direction relation would be semantically empty.

### Graph topology and normalization

- The active model constructs three separate single-modality utterance graphs. It is not a single joint `3T` node graph.
- No explicit `A + A^T` appears, but symmetric past/future windows create both temporal directions for most in-window node pairs.
- Edge attention performs a full-time softmax before masking and re-normalization. This must be replaced by mask-first softmax.
- `GraphNetwork.forward` ignores its `edge_norm` argument when calling `RGCNConv` in the audited GS-MCC file.
- Exact relation-degree normalization inside the external `torch-geometric==1.7.2` implementation is **UNCONFIRMED** because that dependency is not vendored. It cannot restore causality when the provided edge list already contains future paths.

### Fourier operator and low/high paths

The audited `FGN` receives the RGCN output and calls `torch.fft.rfft` over dimension 1 of a `[num_nodes, hidden]` tensor. It receives neither adjacency nor a Laplacian and performs no graph eigendecomposition. The source therefore does not expose which adjacency/Laplacian defines an FGO; in this active call, none is passed. The code also returns one transformed representation rather than explicit low/high graph-frequency branches. Exact correspondence between this implementation and the paper's FGO/low-high equations is **UNCONFIRMED**.

### Nodal attention

Generic `attentive_node_features` iterates over every dialogue time and applies `MatchingAttention` over the complete padded dialogue using only a padding mask, so it is noncausal if activated. In the executed GS-MCC class, graph networks use `return_feature=True` and return before this classifier, so the nodal-attention flag does not affect the active graph feature path.

### “Contrastive” objective

The executed loss contains fused/modal classification losses and KL divergence from each modality distribution to the fused distribution. No positive/negative sample builder, queue, InfoNCE denominator, or cross-time pair construction was found. Therefore:

- there is no demonstrated future-label path in the KL loss itself;
- every distribution still contains future input through the model;
- how the paper's contrastive terminology maps to the released KL implementation is **UNCONFIRMED**;
- inventing an InfoNCE loss and calling it official GS-MCC would be unsupported.

## Verdict

**Official GS-MCC is not model-level strict causal.** Demonstrated future paths are:

1. three bidirectional LSTMs;
2. default `window_future=10` graph neighbors;
3. edge-pair orientation that cannot be made causal by only zeroing `window_future`;
4. softmax before legal-edge masking;
5. test evaluation every epoch and best result selected by Test F1 (protocol leakage, separate from model inference causality).

Unconfirmed items are the precise paper-to-code FGO/low-high mapping, exact paper contrastive sample construction, external PyG normalization internals, upstream feature-extractor causality, and successful execution with the absent official feature file.

Strict causalization must replace the BiLSTMs, define edge direction explicitly, remove future relations, mask before softmax, avoid reverse-edge/symmetric operators, use a causal high/low approximation, and use validation-only checkpoint selection. Those changes are material. The resulting project class must be called **Project causal GS-MCC-inspired**, not Official GS-MCC or an exact reproduction.
