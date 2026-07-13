# MultiDAG causal audit

## Identity and verdict

The project implementation is
`models/baselines/multidag_cl/multidag_cl_model.py::MultiDAGCLBaseline`.
It is a project-native **MultiDAG-inspired causal baseline**, not an official
MultiDAG or MultiDAG+CL reproduction. The compatibility name `MultiDAGCL` is
retained in code and checkpoints.

The project path is `model_level_strict_causal`. Its end-to-end status remains
`end_to_end_causal_unverified_features`.

## Project-native implementation audit

| Stage | Path / class / def | Input -> output | Future information |
|---|---|---|---|
| Modality encoder | `multidag_cl_model.py::CausalModalityEncoder` | each `[B,T,D]` -> `[B,T,H]` | Default GRU has `bidirectional=False`; the retained `linear` branch is position-wise |
| Directed graph | `build_directed_past_adjacency` | mask/speakers/window -> `[B,T,T]` | `adj[b,i,j]` requires `j <= i`, validity, and the history window |
| Speaker relation | `build_speaker_relation_masks` | speakers -> same/different relation matrices | Full relation matrices exist, but they do not add graph edges |
| Graph attention | `_masked_normalized_softmax` | scores/mask -> weights | Mask is applied with `masked_fill` before softmax and re-normalized |
| Graph update | `_SpeakerAwareDAGLayer.forward` | `[B,T,H]` -> `[B,T,H]` | Iterates in time order and slices sources to `:index+1` |
| Layer stack | `MultiDAGCLBaseline.forward` | fused features -> graph states | Each layer receives causal states; no future state can be created in an earlier layer |
| Classifier | same `forward` | same-time layer states -> `[B,T,C]` | No temporal pooling or full-sequence attention |

The speaker matrices include future pairs as metadata, but the layer receives
only the current row prefix and an already-causal adjacency. Future speaker
identity therefore cannot affect a current attention weight.

The train/eval builders resolve `graph.context_mode` and `window_past`, then pass
the effective history window and `modality_encoder_type` to the model. Existing
`context_mode: full` names in this model family are mapped to `past_all_causal`;
they mean unlimited past, not bidirectional context.

## Dynamic audit

Local synthetic result (`B=2`, `T=5`, cutoff `t=2`, eval mode):

| Check | Maximum |
|---|---:|
| Future zero perturbation | 0 |
| Future random noise | 0 |
| Future cross-sample shuffle | 0 |
| Prefix/full | 0 in the standardized audit run |
| Future text gradient | 0 |
| Future audio gradient | 0 |
| Future visual gradient | 0 |
| Future adjacency violations | 0 |

The standardized `causal_gru` benchmark passes at `1e-6` and `1e-5`. Unit tests
also cover the retained `linear` encoder branch.

## Project implementation vs official models

### Project MultiDAG-inspired

- Independently encodes T/A/V with a causal GRU before fusion.
- Uses a simple most-recent-`W` utterance window and includes self-loops.
- Uses the project's speaker-aware attention/GRUCell/FFN graph layer.
- Optimizes masked cross-entropy only; no curriculum component is implemented.
- Selects checkpoints using validation metrics.

### Official MultiDAG

The external official repository is not present in this checkout. The official
source describes a `DAGERC_fushion` backbone with sequential DAG updates. Its
window semantics count same-speaker predecessors and include intervening
utterances, which differs from the project window. Official source references:
[dataset graph construction](https://github.com/vanntc711/MultiDAG-CL/blob/main/dataset.py),
[model backbone](https://github.com/vanntc711/MultiDAG-CL/blob/main/model.py).

### Official MultiDAG+CL

Official MultiDAG+CL uses the same backbone with curriculum-learning training;
`CL` does not mean the contrastive loss in the current project name. The project
does not implement this curriculum and must not be reported as an exact official
reproduction. See the [official README](https://github.com/vanntc711/MultiDAG-CL).

The official optional global nodal attention is noncausal. Official training
also selects a best model from test F-score in its original workflow; that
selection protocol is prohibited for this benchmark and is not reused here.

## Repairs

No project MultiDAG-inspired model repair was required. Standardization changes
only add automated evidence, explicit benchmark YAMLs, metadata, and accurate
naming in reports.

