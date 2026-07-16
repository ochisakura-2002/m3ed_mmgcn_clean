# MMGCN paper-code-project mapping

| Paper module / formula | Official code | Project code | Equivalent? | Known difference and reason | Performance impact | Evidence |
|---|---|---|---|---|---|---|
| Modality encoders, Eq. (2) | `model.py::Model`, modality setup | `original_repro/mmgcn/model.py::text_context_encoder`, `audio_encoder`, `visual_encoder` | Yes for effective path | Paper describes speaker concatenation with all modalities; released path injects it into language. Released effective path retained and documented. | Possible | STRUCTURE_PASS; performance unconfirmed |
| Speaker embedding, Eq. (3) | `model_mm.py::MM_GCN` | `speaker_embeddings` | Partial | Addition follows released tensor dimensions rather than literal paper concatenation. | Possible | paper/code mismatch documented |
| Three-node-per-utterance graph, Eqs. (4)-(5) | `model_mm.py::MM_GCN.create_big_adj` | `build_official_like_multimodal_adjacency` | Yes | Full same-modal graph and aligned cross-modal edges; angular similarity and gamma retained. | Low expected | mechanism test / code audit |
| Renormalized adjacency, Eq. (6) | `model_mm.py` adjacency normalization | shared dense graph builder | Yes | Numerical epsilon is added for zero-degree safety. | Negligible | synthetic finite checks |
| Deep GCNII, Eq. (7) | `model_GCN.py::GCNII`, `model_mm.py::MM_GCN` | `GCNIIBackbone` | Yes at equation level | Released paths disagree on depth; YAML pins paper-oriented four layers. | Likely | `paper_code_mismatch_resolved` by exposed depth |
| GCNII residual path | official constructor/run path enables residual propagation | `use_residual` | Yes | Original-reproduction smoke, screening, and five-fold base configs set `true`; unrelated causal configs are untouched. | Material | default/config contract test |
| Fusion/classifier, Eqs. (8)-(9) | `model_mm.py::MM_GCN.forward` | graph modality concat + `classifier` | Yes | Padding is scattered back into unified `[B,T,C]` output. | None for valid nodes | forward/backward test |
| Weighted CE + L2 | `train.py`, `run.sh` | masked weighted CE + optimizer weight decay | Yes | L2 is optimizer weight decay rather than an explicit loss term. | Low | config and checkpoint audit |

Current status: `STRUCTURE_PASS`, `TRAINING_PASS` on synthetic round-trip,
`UNCONFIRMED` on real IEMOCAP until pinned PKLs are available.
