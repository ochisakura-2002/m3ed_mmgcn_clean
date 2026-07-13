# Causal repairs changelog

## 2026-07-13 baseline audit

### Model repairs

No model computation repair was necessary.

- MMGCN already had an explicit `context_mode: causal` branch whose mask is
  applied before graph normalization and message passing.
- The project MultiDAG-inspired model already used a unidirectional modality
  encoder, directed-past adjacency, pre-softmax masking, and chronological graph
  updates.

The existing MMGCN `full` compatibility path was deliberately left unchanged.

### Protocol and verification additions

| Addition | Why it is causal-safe | Effect on noncausal configs | Test |
|---|---|---|---|
| Future perturbation/prefix/gradient audit tool | Runs the real forward path and exposes every measured difference | Read-only; no model behavior change | Synthetic MMGCN and MultiDAG audit |
| MMGCN invariance tests | Checks graph direction plus output/gradient invariance | Includes `full` negative control; does not rewrite it | `tests/test_mmgcn_causal_invariance.py` |
| MultiDAG invariance tests | Checks encoder, graph, output, and gradient paths | Retained linear branch remains available | `tests/test_multidag_causal_invariance.py` |
| Session-holdout validation | Prevents train/validation session and speaker overlap without touching test | `official_prefix` and `random` remain supported | `tests/test_iemocap_session_holdout.py` |
| Separate causal benchmark YAMLs | Select existing causal branches explicitly | Existing YAML defaults are unchanged | `scripts/dev/validate_config_tree.py` |

