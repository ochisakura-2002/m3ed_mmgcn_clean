# DialogueGCN paper-code-project mapping

| Paper module / formula | Official code | Project code | Current fidelity | Evidence / remaining difference |
|---|---|---|---|---|
| Bidirectional sequential context encoder | GRU/LSTM base-model paths | `context_encoder` | Adapted | Formal IEMOCAP default is bidirectional LSTM. |
| Directed past/future graph and `2*M^2` relations | `edge_perms`, `batch_graphify` | `build_dialoguegcn_graph`, `dialoguegcn_relation_id` | Formula aligned | Window and speaker/direction fixture. |
| Edge attention | `MaskedEdgeAttention` | `_EdgeAttention` | Paper-formula aligned adaptation | Bilinear incoming-edge softmax; no claim of binary identity with the legacy helper. |
| First relation-aware convolution | `GraphNetwork.conv1` / RGCN | `DialogueGCNRelationalGraphNetwork` relation weights and `root` | Paper-formula aligned | Relation-specific transforms, per-relation neighbor normalization, edge-attention weighting, and independent root transform are explicit. |
| Second graph convolution | `GraphNetwork.conv2` | `second_neighbor`, `second_root` | Paper-formula aligned | Explicit neighbor sum and independent self transform. |
| Context residual and nodal attention | `classify_node_features`, `MatchingAttention` | context/graph concatenation and `_NodalAttention` | Adapted | Full-dialogue nodal attention is intentionally noncausal. |
| Weighted CE and L2 | `train_IEMOCAP.py` | masked class-weighted CE and optimizer weight decay | Adapted | Test-every-epoch behavior is removed. |

The paper-adjacent IEMOCAP defaults are LSTM, dropout 0.4, learning rate
0.0003, batch size 32, L2/weight decay 0.0, class weighting enabled, and nodal
attention enabled.

Current status is `paper_equation_aligned_official_code_adapted`. A fixed
three-node synthetic graph test assigns nodes, relations, attention, and layer
weights and checks the exact hand-computed two-layer output. This is stronger
than a shape/finite check, but real IEMOCAP performance and numerical identity
with the legacy PyG stack remain unconfirmed. Do not describe the entire model
as official-code-identical.
