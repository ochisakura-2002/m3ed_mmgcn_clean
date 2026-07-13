# New causal graph baselines — Task 1 report

Date: 2026-07-13

## Executive status

Two isolated candidates now exist:

1. **Project causal GS-MCC-inspired** (`CausalGSMCCInspiredBaseline`).
2. **Project causal DialogueGCN** (`CausalDialogueGCNBaseline`).

Both complete synthetic IEMOCAP-interface forward/loss/backward and pass the required model-level dynamic causal tests. Neither is connected to formal data loading, training, checkpoint evaluation, Session-Holdout, or the four-model benchmark.

## Required questions

### 1. Is official GS-MCC strict causal?

No. The released executed path has three bidirectional LSTMs, a default positive future window, an edge orientation that cannot be causalized by only setting the future window to zero, and softmax before exact edge masking. Its training loop also selects its reported best result from Test F1.

### 2. Is official DialogueGCN strict causal?

No. The released root and maintained variants use bidirectional or explicitly forward/reverse context encoding, past and future graph edges, two temporal relation directions, and optional full-dialogue nodal attention. Root IEMOCAP reporting takes the maximum Test F1 across epochs.

### 3. What are the future-information paths?

| Family | Demonstrated paths |
|---|---|
| GS-MCC | BiLSTM text/audio/visual encoders; future-window graph neighbors; PyG pair-orientation mismatch under a naive causal window; pre-mask full-time softmax; offline features with extractor causality **UNCONFIRMED** |
| DialogueGCN | BiLSTM/BiGRU or reversed DialogRNN; future-window graph neighbors; future temporal relations; optional full-dialogue nodal attention; offline features with extractor causality **UNCONFIRMED** |

Test-driven epoch/result selection is a protocol leak for both root workflows, not a tensor-path inference leak.

### 4. Which project files were added?

- Shared graph utilities under `models/baselines/causal_graph_common/`.
- GS-MCC-inspired model, directed spectral approximation, and losses under `models/baselines/gsmcc/`.
- DialogueGCN model, graph/attention, and pure-PyTorch RGCN under `models/baselines/dialoguegcn/`.
- Two synthetic YAMLs, two independent debug scripts, three test modules, two source audits, two port notes, and this report.

No official repository source was copied into `models/`; the temporary audit clones remain under ignored `tmp/source_audit/`.

### 5. Did both models complete forward/loss/backward?

Yes, with `B=2`, `T=6`, unequal lengths `[6,5]`, padding, speaker IDs 0/1, and IEMOCAP interface dimensions text/audio/visual `100/1582/342`, `C=6`.

- GS-MCC-inspired: logits `(2,6,6)`, low/high `(2,6,16)`, 34,342 parameters, 288 edges, masked classification plus auxiliary components, all gradients finite.
- DialogueGCN: logits/context/graph `(2,6,6)/(2,6,16)/(2,6,16)`, 38,182 parameters, 32 edges, all 4 relations active, normalized legal edge attention, all gradients finite.

Loss values are random synthetic diagnostics and are not experimental results.

### 6. Did future perturbation tests pass?

Yes. Independent future text, audio, and visual replacement by zeros, random noise, and cross-sample shuffles, plus joint T/A/V randomization, leave all valid logits at `<=t` unchanged within `1e-6` for both models.

### 7. Did prefix/full pass?

Yes. Current logits match between full dialogue and consistently truncated prefix. GS low/high and DialogueGCN context/graph current representations also match.

### 8. Did future gradient pass?

Yes. Current logits have exactly zero synthetic gradients to valid future T/A/V input positions.

### 9. Were dependencies added?

No. The implementation uses existing PyTorch and PyYAML only. It does not add `torch-geometric`, `torch-scatter`, or `torch-sparse`.

### 10. Was existing model behavior modified?

No. MMGCN, project MultiDAG-inspired, their tests/results, formal YAMLs, dataset split logic, train/eval scripts, and pipeline were not modified.

### 11. Can the GS-MCC project version be called official GS-MCC?

No. Its directed polynomial high/low filter replaces the released hidden-axis FFT/PyG path, and its auxiliary objectives are explicit project proxies. The required name is **Project causal GS-MCC-inspired**.

### 12. Which DialogueGCN paper cores are retained?

Speaker-pair relations, learned edge attention, relational message passing, residual node representation, sequential context before graph propagation, and utterance classification are retained. Temporal future relations and full nodal attention are removed; a project T/A/V adapter and pure-PyTorch convolution are added.

### 13. Can the models enter a real-batch audit?

Yes, as isolated candidates. They are ready for a read-only real-batch tensor/shape/causality audit. They are not formal IEMOCAP baselines, benchmark-ready, or approved for training.

### 14. What must Task 2 integrate?

Task 2 must explicitly extend, without altering current semantics:

1. a shared IEMOCAP Session-Holdout batch adapter/factory;
2. isolated train entries or a deliberate unified baseline registry;
3. validation-only checkpoint selection and checkpoint metadata;
4. checkpoint evaluation with the same split contract;
5. four-model benchmark configuration/runner and audit hooks;
6. real-batch future perturbation, prefix/full where feasible, and gradient audit;
7. metric/report code for Accuracy, Weighted-F1, Macro-F1, and UAR.

Likely touched files include new baseline train/eval entries plus carefully reviewed extensions to the dataset factory, pipeline registry, and benchmark YAML. Existing `scripts/train_mmgcn.py`, `scripts/evaluate_checkpoint.py`, `scripts/baselines/train_multidag_cl.py`, and `scripts/run_experiment_pipeline.py` were intentionally untouched in Task 1.

### 15. Current blockers

- The formal IEMOCAP PKL is absent locally, so real feature dimensions/content and batch behavior were not executed.
- Upstream feature extractors are not verified online-causal.
- Official GS-MCC FGO and contrastive paper-to-code mappings remain **UNCONFIRMED** from the released active source.
- Old official PyG dependency compatibility is **UNCONFIRMED** and deliberately irrelevant to the pure-PyTorch project ports.
- Neither candidate has formal optimization, checkpoint, Session-Holdout, or performance evidence.

## Verification record

```text
py_compile: PASS
GS synthetic forward/loss/backward: PASS
DialogueGCN synthetic forward/loss/backward: PASS
pytest: 23 passed
future perturbation: PASS at 1e-6 and 1e-5
prefix/full: PASS at 1e-6 and 1e-5
future gradient: PASS (exact zero at tested future positions)
new dependency: NO
formal training executed: NO
```
