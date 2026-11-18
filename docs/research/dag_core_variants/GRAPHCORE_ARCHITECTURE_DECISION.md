# GraphCore architecture decision

WORKSTREAM = MULTIDAG. Base HEAD: `daf0957a757044a4c00cb0827ce847add7caf1d1`.
The identity guard passed in the required repository and branch. Existing dirty
files and deleted documents are preserved; deleted entry documents were read
from HEAD without restoration. Paper mapping was written before source code.

## Frozen boundary

Parent: `ta_v0_t1024_ab1_diagnostic`. User-supplied Dev reference W-F1 is
0.6935718114 (not re-evaluated). Input T1024+A1582, packed UniLSTM 100,
audio projection/ReLU 100, concat/fusion to h0=300, four 300D stages, 1500D
preclassifier and original six-class classifier remain fixed. The parent is the
reference for this experiment, not a claim about the project's final baseline.

Use `TAV0` inheritance for initialization and stable encoders/fusion/classifier.
The enabled variant replaces its graph stack with one unified GraphCore.
The new forward duplicates only the explicit input validation/boundary wiring,
not encoder/classifier math or training mechanics. An OFF model delegates
directly to the original forward with the original state_dict identity and
initialization order. No baseline file is edited.

Contract: `core(h0, adjacency, relations, attention_mask) -> tuple(h1,h2,h3,h4)`.
All tensors are [B,T,300], padding exactly zero; canonical order is the tensor's
time dimension. Adjacency is bool [B,receiver,source], strictly lower triangular,
and must exclude padding. Relations are complete, disjoint same/different masks.
Core validation rejects invalid edges rather than silently repairing them.

Implementation locations (new files only):

- `models/multidag_cl/graph_core_variants/`: common contract, Track, Coop, DCN,
  frozen graph config and a single TA wrapper.
- `scripts/runtime/multidag_cl_graph_core_variants/`: config validation and shared
  formal/synthetic orchestration. Subclass original TrainDevRuntime solely for
  checkpoint identity; use original optimizer, loader, masked loss, clipping,
  selector, checkpointing, fit loop and Dev exports through imports.
- `scripts/models/multidag_cl/graph_core_variants/train.py`: one config-selected
  entry for the three independent models, not three copied trainers.
- `configs/multidag_cl/{track_t1024_v0,coop_t1024_v0,dcn_t1024_v0}/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev.yaml`.
- `tests/models/multidag_cl/graph_core_variants/`: contracts, equation checks,
  causal and prefix guards, runtime/config/identity/No-Test regression.

## Configuration and checkpoint decisions

Each formal YAML has the exact baseline model/data/runtime/training/checkpoint
sections, with new `identity` and `graph_core` sections and its own output group.
The small core defaults are made explicit in resolved configs and manifests.
No optimizer/loss/dropout/early-stop/seed/budget override is accepted. The OFF
switch is a synthetic ablation path; formal variant runs require enabled=true.

The model keeps a real frozen TAV0Config for dataset/collate compatibility.
Variant identity and graph parameters live separately, appear in model extra
state and structured checkpoint identity, and reject cross-variant or baseline
loads before copying parameters, even with strict=false. The underlying training
checkpoint schema is replaced with a distinct variant schema by the new runtime.

Train/Dev mechanics remain seed100, 30 epochs, batch16, AdamW lr5e-4,
betas(.9,.999), eps1e-6, weight_decay0, gradient clip5, no scheduler, no class
weights, no label smoothing, no curriculum, no early stopping. Selection is Dev
W-F1 descending, Dev loss ascending, epoch ascending. NO Test factory access,
loader, evaluation, metric or prediction is permitted. Synthetic checks create
only Train/Dev datasets and run on CPU.

## Output and audit

Use the existing output resolver and fresh atomic run-directory creation:
`outputs/<DATE>/multidag_<variant>_t1024_formal/runs/<run_id>`.
Run ID includes model/variant, seed100, timezone timestamp, and hash of resolved
config and data identity. Explicit output base/date options retain this layout.
Collisions fail before any optimizer step. Formal use requires an explicit flag
and all three data paths. No formal run is executed as part of this task.

Manifest fields include model_name, model_lineage, graph_core_variant,
parent_baseline, source_paper, dimensions/layer count, parameter_count, git
identity, seed, training_config, data_identity, full core config and zero Test
counters. `graph_topology_changed=false` means the input node/edge support is
unchanged; closure and implicit self terms are propagation operators, not new
stored DAG edges. `future_edges_added=false` is verified by causal tests.

## Acceptance evidence

Require all original ta_v0 tests plus new variant tests. Tests must check exact
unchanged module initialization and OFF state_dict/logits, four-depth outputs,
gradients, variable lengths/padding, arbitrary speaker-ID relabeling, invalid
graphs, future perturbation at u4/u5, truncated-prefix equivalence, dense-source
formula agreement for PDCN and Track, manifest/checkpoint identities, formal
protocol mutations, output collision and Test rejection before factory calls.
The final implementation report records actual outcomes; no training-quality
conclusion follows from passing these engineering gates.
