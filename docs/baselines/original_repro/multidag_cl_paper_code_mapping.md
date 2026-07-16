# MultiDAG+CL paper-code-project mapping

| Paper module / formula | Official code | Project code | Equivalent? | Known difference and reason | Performance impact | Evidence |
|---|---|---|---|---|---|---|
| Modality encoding/fusion, Eqs. (1)-(2) | `model.py::DAGERC_fushion` | `OriginalReproMultiDAGCL` encoders and `fused_input` | Yes | Unified batch-major padding replaces repository JSON batches. | Low | shape/backward test |
| DAG attention, Eq. (3) | `model.py::DAGERC_fushion.forward` | `_DAGLayer.attention` | Yes | Stable masked softmax handles empty predecessor rows. | Negligible | finite/padding test |
| Relation messages, Eq. (4) | same class | `_DAGLayer.same_speaker`, `different_speaker` | Yes | Relation IDs are a boolean same-speaker tensor. | None expected | predecessor/relation test |
| Node/context GRUs, Eqs. (5)-(7) | same class | `_DAGLayer.node_gru`, `context_gru`; state sum | Yes | Pure PyTorch batch-major translation. | Low | forward/backward test |
| Released predecessor selection | `dataset.py::get_adj_v1` | `build_get_adj_v1` | Yes | Scans backward until kth same-speaker predecessor; not a fixed utterance radius. | Material if changed | exact adjacency fixture |
| Difficulty, Eq. (8) | `cl.py::Dialog.measure_difficulty` | `dialogue_difficulty_from_sequences` | Yes | No class-frequency term is invented because paper equation/code contain none. | Material | exact scalar fixture |
| Baby-step scheduler, Algorithm 1 | `dataset.py`, `dataloader.py`, `trainer.py` | `curriculum_baby_step_indices`, runtime subset loader | Yes | IEMOCAP uses five buckets per paper Table 4. | Material | epoch visibility fixture |
| CE and classifier | `trainer.py`, `model.py` | structured classification loss/classifier | Yes | Test-per-epoch selection is removed to prevent leakage. | Protocol, not architecture | checkpoint audit |

Current status: `STRUCTURE_PASS`, `TRAINING_PASS` on synthetic round-trip,
`UNCONFIRMED` on real IEMOCAP.
