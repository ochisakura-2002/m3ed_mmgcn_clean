# Causal baseline readiness report

Audit date: 2026-07-13

## Executive verdict

The repository now has two verified model-level causal contextual benchmarks:

- MMGCN with the explicit causal dense-graph branch.
- The project-native MultiDAG-inspired model with causal GRU and directed-past
  graph updates.

Both pass synthetic future perturbation, prefix/full, future-gradient, and
adjacency tests at `1e-6`. Neither may be called a fully verified end-to-end
online causal system because the upstream PKL feature extractors remain
unverified.

No formal training or real-PKL evaluation was run.

## Required answers

1. **How many current MERC model implementations?** Four: MMGCN,
   project MultiDAG-inspired, SDT, and SimpleMLP.
2. **How many have a train/checkpoint/evaluate code path?** Three: MMGCN,
   project MultiDAG-inspired, and SimpleMLP. SDT is fake-forward only.
3. **Which models have causal parameters?** MMGCN has graph
   `context_mode/window_past/window_future`; MultiDAG-inspired has graph context
   and window plus `modality_encoder_type`. SDT and SimpleMLP do not expose a
   causal switch.
4. **Which causal parameters actually work?** MMGCN graph values pass through
   both train/eval builders into adjacency construction. MultiDAG graph/window
   and encoder values pass through its builders into the model. Both chains are
   statically and dynamically verified.
5. **Did MMGCN pass static audit?** Yes, for `context_mode: causal`. Its
   `full` path is intentionally noncausal.
6. **Did MMGCN pass dynamic future invariance?** Yes in synthetic mode:
   zero/noise/shuffle differences 0, prefix/full
   `5.21540641784668e-08`, future T/A/V gradients 0, adjacency violations 0.
7. **Did MultiDAG-inspired pass static audit?** Yes. Its encoder is
   unidirectional, adjacency is lower triangular, mask precedes softmax, graph
   updates are chronological, and the classifier has no future pooling.
8. **Did MultiDAG-inspired pass dynamic future invariance?** Yes in synthetic
   mode: all reported differences/gradients and adjacency violations are 0.
9. **Were future-information paths found?** Yes outside the causal candidates:
   MMGCN `full` builds future same-modality edges, and SDT uses full-dialogue
   attention without a causal mask. The official external implementations are
   absent locally and cannot receive a project model verdict.
10. **Which future paths were repaired?** None in the two candidate models;
    their explicit causal computations were already correct. Verification and
    separate benchmark selection were added instead of changing model math.
11. **Was any existing noncausal behavior changed?** No. In particular,
    `configs/train_mmgcn_iemocap_official.yaml` remains `full` and old
    `official_prefix`/`random` split behavior remains available.
12. **Is Session-Holdout implemented?** Yes for `Ses01`–`Ses04`, preserving
    trainVid/testVid order and keeping Ses05 test unchanged. Four synthetic PKL
    tests pass for dialogue/session/session-qualified-speaker disjointness and
    legacy split regression. Real official PKL counts/hash could not be locally
    rechecked because `third_party/` is absent.
13. **How many benchmark configs exist?** Ten training configurations: five
    MMGCN and five MultiDAG-inspired (`official_prefix`, `val_ses01`...
    `val_ses04`). Each has a matching pipeline YAML, for 20 new YAML files total.
14. **Can MMGCN be labelled model-level strict causal?** Yes, only for the
    explicit causal YAMLs and model path tested here.
15. **Can MultiDAG be labelled model-level strict causal?** The **project
    MultiDAG-inspired** implementation can. Official MultiDAG and official
    MultiDAG+CL remain `unable_to_determine` in this checkout.
16. **What feature causality remains unverified?** Whether the text, audio, and
    visual feature extractors used cross-utterance/future dialogue context;
    whether preprocessing normalization used future utterances; and whether the
    configured PKL exactly matches the supplied SHA on the remote machine.
17. **What external model type is the best next target?** A history-only DAG-ERC
    style model with no global nodal attention and a validation-only checkpoint
    adapter. It is structurally compatible with the causal contract and provides
    a useful independent comparison to dense MMGCN and the project
    MultiDAG-inspired path. Dependency/source availability must be audited first.

## Session-holdout protocol

New dataset fields:

```yaml
dataset:
  val_split_strategy: session_holdout
  val_session_id: Ses01  # Ses01, Ses02, Ses03, or Ses04
```

The implementation filters official `trainVid` in its original order. The held
session becomes validation; the other three sessions become training; official
`testVid` remains unchanged. `parse_iemocap_session_id` is strict and refuses to
guess an assignment for malformed identifiers.

## Benchmark and selection standard

All ten training configs use:

- the same configured IEMOCAP PKL path and declared SHA256;
- the same six-label mapping;
- validation Weighted-F1 for best checkpoint selection;
- no train/validation/evaluation batch cap;
- full validation and test evaluation;
- Accuracy, Weighted-F1, Macro-F1, UAR, loss, confusion matrix, and per-class
  outputs already provided by the train/eval paths;
- no test metric in scheduling, early stopping, or checkpoint selection.

Each formal run now writes `run_metadata.json` with effective model causality,
feature identity, split protocol, checkpoint metric, seed, and feature dimensions.

## Local verification summary

| Check | Result |
|---|---|
| `py_compile` for requested split/audit/validator/model/entry files | Passed |
| Requested causal pytest files | Test modules are present; project conda lacks `pytest`, and cross-environment pytest loading is incompatible. The same 12 test functions passed directly under the conda PyTorch environment. |
| Session-holdout unit tests | 4 passed |
| Run metadata pytest | 6 passed with system pytest |
| Config tree validation | Passed |
| MMGCN benchmark synthetic audit | Passed at `1e-6` and `1e-5` |
| MultiDAG benchmark synthetic audit | Passed at `1e-6` and `1e-5` |
| Real IEMOCAP batch audit | Not run: configured `third_party/` and PKL are absent locally |
| Formal training | Not run, by task constraint |

## Remaining boundary

The current readiness claim is limited to code and model-level information flow.
Before remote formal runs, verify the PKL SHA256, run `real_batch` causal audits
for both models, record split manifests, and retain validation-selected
checkpoints without inspecting test results during selection.

