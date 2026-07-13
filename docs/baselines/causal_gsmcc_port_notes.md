# Project causal GS-MCC-inspired port notes

Status: synthetic model-level causal candidate; not integrated into formal training.

## Naming boundary

The implemented class is `CausalGSMCCInspiredBaseline`. Reports must call it **Project causal GS-MCC-inspired**. It is not Official GS-MCC, an exact GS-MCC reproduction, or an official causal implementation.

## Official mechanism and causal risk

The paper presents sliding-window multimodal graphs, graph-frequency high/low information, collaborative learning, and emotion classification. The audited repository implements three bidirectional modality encoders, three separate window graphs, RGCN followed by a hidden-axis FFT module, gated fusion, modal/fused classification, and KL consistency. Its default future window, bidirectional LSTMs, edge orientation, pre-mask softmax, and Test-F1 selection are incompatible with the project causal contract. See `gsmcc_official_source_audit.md`.

## Retained ideas

1. Separate text/audio/visual projections.
2. A multimodal graph representation.
3. Repeated high/low transformations intended to separate smooth and residual signals.
4. Collaboration of high- and low-frequency representations before classification.
5. Optional consistency/complementarity auxiliary objectives with default weight zero.

## Removed or replaced mechanisms

| Official/released path | Project causal replacement |
|---|---|
| Three BiLSTMs | Position-independent projection by default; optional forward-only GRU |
| Three separate modality graphs with past/future edges | One utterance-major `3T` graph with history/current edges only |
| Ambiguous PyG edge pair convention | Dense batched `adj[target,source]` |
| PyG RGCN plus hidden-axis FFT | Pure-PyTorch directed polynomial high/low filter |
| Full-time softmax before edge mask | No learned graph attention in v1; row normalization only over legal sources |
| Paper/code contrastive mapping unclear | Explicitly project-local same-time auxiliary proxies, default disabled |
| Test-F1 result selection | No train/eval integration in Task 1; future integration must use Validation only |

## Causal multimodal graph

For dialogue `b`, utterance time `i`, and target/source modalities `m,n`, nodes use utterance-major order `(i,m)`. With `w=window_past` and `w=+infinity` when the value is `None`:

```text
A[b,(i,m),(j,n)] = 1[
  i and j are valid
  and 0 <= i-j <= w
]
```

All same-time T/A/V directions are legal because `j=i`. Every historical same- and cross-modal edge is legal inside the window. Padding nodes have no incoming or outgoing edges. Each batch item has its own adjacency, so cross-dialogue edges are not representable.

The row-normalized shift is:

```text
S[b,p,q] = A[b,p,q] / sum_r A[b,p,r]
```

Rows are targets and columns are sources. No column with `source_time > target_time` enters the denominator.

## Causal directed high/low filter

For graph-layer input `H` and filter depth `K`:

```text
L_0 = H
L_k = S L_(k-1),                 k=1...K
Delta_k = L_(k-1) - L_k
Low_raw = L_K
High_raw = (1/K) sum_k Delta_k
```

The learned residual branches are:

```text
Low  = LayerNorm(H + Dropout(GELU(W_low  Low_raw  + b_low)))
High = LayerNorm(H + Dropout(GELU(W_high High_raw + b_high)))
H_next = (Low + High) / 2
```

Every power of lower-triangular `S` remains lower triangular; stacking graph/filter layers cannot create a future path. There is no symmetrization, reverse edge, full-graph eigendecomposition, or future-dependent degree normalization.

This operator is deliberately named a **causal directed polynomial graph filter**. It is not called the official Fourier graph operator. The audited released `FGN` does not take adjacency/Laplacian and FFTs the hidden axis in its active call, so exact official-FGO equivalence is **UNCONFIRMED**.

## Fusion

The default is:

```text
[Low_utterance ; High_utterance]
-> Linear
-> LayerNorm
-> classifier
```

`Low_utterance` and `High_utterance` are same-time means over the three modality nodes. Configured alternatives are `gate` and `sum`; all are same-time operations.

## Auxiliary losses

`compute_causal_gsmcc_loss` returns:

```text
classification_loss
consistency_loss
complementarity_loss
total_loss
```

- Classification is masked utterance-level cross entropy.
- Consistency is the mean same-utterance pairwise cosine distance among the three low-frequency modality representations.
- Complementarity is the mean absolute same-utterance pairwise cosine similarity among high-frequency modality representations, encouraging non-redundancy.
- Neither auxiliary term reads labels or pairs an utterance with future time positions.

These two auxiliary objectives are transparent **project proxies**, not a claimed migration of the paper's contrastive loss. The released active loss uses modal/fused CE and KL distributions but exposes no positive/negative sampling. Default weights are both `0.0`; Task 1 only establishes a runnable loss path.

## Input and output

```text
text_features:    [B,T,D_t]
audio_features:   [B,T,D_a]
visual_features:  [B,T,D_v]
attention_mask:   [B,T]
lengths:          [B]
speaker_ids_int:  [B,T]
```

Default output is `logits [B,T,C]`. `return_aux=True` additionally returns utterance low/high/fused representations, per-modality low/high representations, boolean and normalized adjacency, node times, edge counts, and valid-node counts. `speaker_ids_int` is validated for interface compatibility; v1 does not make GS-MCC graph weights speaker-aware.

## Configuration fields

Required dimensions are `text_dim`, `audio_dim`, `visual_dim`, `hidden_dim`, and `num_classes`. Validated structural fields are `dropout`, `window_past`, `num_filter_steps`, `num_graph_layers`, `modality_encoder_type`, `fusion_type`, the three loss weights, and `context_mode`.

`context_mode` must equal `causal`. `window_future>0`, `bidirectional=True`, and `nodal_attention=full` raise exceptions. The synthetic IEMOCAP interface config is `configs/smoke/causal_gsmcc_iemocap_synthetic.yaml`.

## Evidence and remaining limits

The IEMOCAP-shaped synthetic script completed forward, masked loss, backward, all-parameter finite-gradient checking, and padding-loss exclusion. Dynamic tests passed future T/A/V and joint perturbation, prefix/full equivalence, future gradients, low/high invariance, adjacency direction, normalization, and multi-layer propagation at `1e-6` (also passing `1e-5`).

Still **UNCONFIRMED**: real-PKL shape/content audit, upstream extractor causality, formal optimization stability, checkpoint compatibility, performance, and exact paper FGO/contrastive correspondence.
