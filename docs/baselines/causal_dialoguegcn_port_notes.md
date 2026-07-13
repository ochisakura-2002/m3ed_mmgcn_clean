# Project causal DialogueGCN port notes

Status: synthetic model-level causal candidate; not integrated into formal training.

## Naming boundary

The implemented class is `CausalDialogueGCNBaseline`. Reports must call it **Project causal DialogueGCN** and describe it as a **DialogueGCN-derived project implementation**. It uses a project T/A/V adapter and pure-PyTorch relational convolution, not an official line-by-line port.

## Official core and causal risk

DialogueGCN's paper core is a sequential utterance encoder, speaker/temporal relation graph, edge attention, relational graph convolution, graph convolution, and utterance classification with optional nodal attention. The audited code reads future utterances through bidirectional or explicitly reversed context encoders, future graph edges, a future temporal-relation family, and optional full-dialogue nodal attention. See `dialoguegcn_official_source_audit.md`.

## Original and project relation definitions

The original implementation allocates `2*S^2` relations:

```text
(source speaker, target speaker, past/future temporal direction)
```

For two speakers this is 8 relations. The strict-causal graph has no future direction, so the project uses exactly `S^2` relations:

```text
relation_id = source_speaker * S + target_speaker
```

For two speakers:

```text
0: source_0 -> target_0
1: source_0 -> target_1
2: source_1 -> target_0
3: source_1 -> target_1
```

Self-loops use the ordinary same-speaker relation; no extra self-loop relation is allocated.

## Multimodal adapter

The original root IEMOCAP graph training path passes only its 100-dimensional text feature to DialogueGCN even though the dataloader loads audio and visual arrays. This project must support the unified T/A/V interface, so it adds:

```text
T, A, V -> independent Linear + GELU
        -> concat (default) or learned modality gate
        -> Linear / weighted sum
        -> LayerNorm + GELU + Dropout
        -> utterance representation
```

This is a same-time adapter and cannot introduce future information. It is a project extension.

## Causal context encoder

Default `context_encoder_type=causal_gru` uses `nn.GRU(..., bidirectional=False)` with packed valid prefixes. `linear` is also supported. BiLSTM, BiGRU, an unmasked Transformer, and complete-time pooling are not available.

## Causal graph and edge attention

For valid target `i` and source `j`:

```text
A[b,i,j] = 1[0 <= i-j <= window_past]
```

`window_past=None` means all history; `0` means current/self only. Padding has no edges and each batch item is a separate graph.

With contextual representations `c_i`, learned query/key projections produce:

```text
e_ij = <W_q c_i, W_k c_j> / sqrt(d)
alpha_ij = exp(e_ij) / sum_{r: A[i,r]=1} exp(e_ir)
```

The mask is applied before softmax. Invalid future/padding positions receive exactly zero and do not enter the denominator.

## Pure-PyTorch relational convolution

For relation `r`, layer input `h_j`, and legal attention `alpha_ij`:

```text
m_i = sum_r sum_{j: relation(i,j)=r} alpha_ij W_r h_j
z_i = LayerNorm(
  Dropout(GELU(m_i + W_self h_i)) + W_res h_i
)
```

Each relation owns one weight tensor. Layers reuse the same strictly causal adjacency and relation IDs. There is no graph symmetrization, reverse-edge insertion, future relation, or future-to-past remapping. One or more layers are supported.

## Nodal attention

The official optional nodal attention reads the complete valid dialogue. Task 1 disables it and classifies the same-time graph representation. Config accepts only `none`; `full` and every other value raise. A future causal nodal-attention implementation would require a separate lower-triangular mask and is not needed for this stage.

## Input and output

```text
text_features:    [B,T,D_t]
audio_features:   [B,T,D_a]
visual_features:  [B,T,D_v]
attention_mask:   [B,T]
lengths:          [B]
speaker_ids_int:  [B,T]
```

Default output is `logits [B,T,C]`. `return_aux=True` returns adjacency, `edge_type`/`relation_ids`, the exact relation mapping, edge attention, context representation, graph representation, and logits.

## Configuration fields

Validated fields include all modality/hidden/class dimensions, `dropout`, `window_past`, `context_encoder_type`, `num_graph_layers`, `num_speakers`, `fusion_type`, and `context_mode`.

`context_mode` must equal `causal`. `window_future>0`, `bidirectional=True`, and `nodal_attention=full` raise. The synthetic IEMOCAP interface config is `configs/smoke/causal_dialoguegcn_iemocap_synthetic.yaml`.

## Evidence and remaining limits

The IEMOCAP-shaped synthetic script completed forward, masked CE, backward, all-parameter finite-gradient checking, padding-loss exclusion, relation reporting, and legal-neighbor edge-attention normalization. Dynamic tests passed future T/A/V and joint perturbation, prefix/full equivalence, future gradients, relation orientation, two graph layers, forward-only GRU, and deterministic evaluation at `1e-6` (also passing `1e-5`).

Still **UNCONFIRMED**: real-PKL audit, upstream extractor causality, formal training behavior, Session-Holdout integration, checkpoint/evaluation compatibility, and benchmark performance.
