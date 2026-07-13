# MMGCN causal audit

## Verdict

The explicit `graph.context_mode: causal` path is
`model_level_strict_causal`. The default/formal IEMOCAP `full` path is
`model_level_noncausal` and remains unchanged for official compatibility.

The end-to-end status is `end_to_end_causal_unverified_features` because the
upstream extraction of the utterance-level PKL features has not been fully
audited for online causality.

## Actual implementation and parameter chain

| Stage | Path / symbol | Input -> output | Future information |
|---|---|---|---|
| Training config | `configs/train_mmgcn_m3ed_causal.yaml`, `graph` | context/window values | Sets `context_mode: causal`; this M3ED config is marked `skeleton_only`, not a formal IEMOCAP result |
| Train builder | `scripts/train_mmgcn.py::build_model` | YAML -> `M3EDMMGCN` | Passes `context_mode`, `window_past`, and `window_future` into the model |
| Eval builder | `scripts/evaluate_checkpoint.py::build_model` | checkpoint config -> model | Reconstructs the same graph settings |
| Model | `models/baselines/mmgcn/mm_gcn.py::M3EDMMGCN` | `[B,T,D_*]` -> `[B,T,C]` | The causal value reaches adjacency construction in `forward` |
| Graph | `models/baselines/mmgcn/dense_graph.py::build_official_like_multimodal_adjacency` | compact modality nodes -> `[3N,3N]` | Causal mode filters future sources before normalization |

`scripts/train_mmgcn.py::save_checkpoint` saves the complete config; evaluation
therefore does not silently fall back from a causal checkpoint to the full graph.

## Static path audit

### Per-modality encoders

`M3EDMMGCN._project_modalities` applies a separate `Linear` projection at every
utterance. Speaker and modality embeddings are added per position. There is no
RNN, BiRNN, sequence attention, or temporal pooling in this stage.

Input:

`text/audio/visual [B,T,D]`, optional `speaker_ids_int [B,T]`

Output:

`audio_h`, `visual_h`, `text_h`, each `[B,T,H]`

Future path: none in this stage.

### Same-modality graph edges

`dense_graph.py::context_allowed(target_time, source_time, ...)` returns false
when `context_mode == "causal"` and `source_time > target_time`. The adjacency
write uses `adj[target_node, source_node]`, so this direction is not reversed.

Although a full pairwise similarity matrix is calculated, each entry depends
only on its target/source feature pair. Only allowed entries are copied into the
adjacency. Degree normalization runs after masking, so a causal target's degree
does not depend on a future row entry.

Future path: none in causal mode; present in `full` mode.

### Cross-modality edges

Cross-modality edges connect audio/visual/text nodes for the same utterance only.
They have `source_time == target_time` and are permitted by the contract.

### Graph propagation

`GraphConvolution.forward` computes `hi = adj @ input_features`. A target row
therefore reads only its permitted source columns. Repeated lower-triangular
message-passing layers remain lower-triangular. Dropout, Linear, ReLU, residual,
and `h0` injection are node-wise operations and do not mix time steps.

### Classifier

The classifier concatenates the three propagated nodes belonging to the same
utterance and applies `final_fc`. It performs no dialogue-level pooling.

### Padding authority

`lengths` is the authoritative validity input for compact MMGCN nodes;
`attention_mask` is accepted but not used in node construction. Dynamic prefix
tests must therefore truncate both consistently. This is not a future leak for
well-formed project batches, but inconsistent external inputs could otherwise be
misinterpreted.

## Dynamic audit

Command shape:

```powershell
conda run -n m3ed_mmgcn python scripts/analyze/audit_model_causality.py --config <causal-yaml> --mode synthetic --output-dir outputs/dev/causal_audit/mmgcn
```

Local synthetic result (`B=2`, `T=5`, cutoff `t=2`, eval mode):

| Check | Maximum |
|---|---:|
| Future zero perturbation | 0 |
| Future random noise | 0 |
| Future cross-sample shuffle | 0 |
| Prefix/full | `5.21540641784668e-08` |
| Future text gradient | 0 |
| Future audio gradient | 0 |
| Future visual gradient | 0 |
| Future/cross-dialogue adjacency violations | 0 |

The causal path passes at `1e-6` and `1e-5`. A `full`-context negative-control
unit test changes the current logit after future perturbation, confirming that
the test can detect leakage.

## Noncausal compatibility path

`configs/train_mmgcn_iemocap_official.yaml` intentionally uses
`context_mode: full` and `window_future: null`. It constructs future same-modality
edges and is `model_level_noncausal`. This configuration was not changed.

## Repairs

No MMGCN model repair was required. The new IEMOCAP causal benchmark is kept in
separate YAML files and selects the existing explicit causal branch.
