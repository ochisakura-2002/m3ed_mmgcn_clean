# TAV Full Audit v1 — source audit

WORKSTREAM = MULTIDAG; inspected HEAD `8ee7d966ce85cd26f66c0a2e152d79c642b101a8`.
The workstream guard passed. Existing dirty files and deleted documentation are
outside this task. Deleted project entry documents were read from HEAD only.

## Reuse decisions

| Concern | Inspected source | Decision |
| --- | --- | --- |
| T1024 true TA reference | `models/multidag_cl/ta_v0/`, `scripts/runtime/multidag_cl_ta_v0/`, `configs/multidag_cl/ta_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev_t1024_ab1.yaml` | Leave unchanged. TA uses 300D modality encoders and a learned 600→300 fusion; it is a separate lineage. |
| Historical causal TAV | `models/multidag_cl/paper_reimplementation/{encoders,model,config}.py`; `configs/multidag_cl/paper_reimplementation/iemocap/full_context/paper_data_reproduction/causal_text_encoder_diagnostic/modality_ablation/cl7_unilstm_tav.yaml` | Reuse PaperFormulaModalityEncoder: A ReLU(1582→100), V ReLU(342→100), T linear(1024→100)→packed UniLSTM(100); A,V,T concatenation=300. Reuse historical classifier and four MultiDAGLayer modules. No BiLSTM. |
| Encoded modality ablation | Same historical encoder and seven historical modality YAMLs | Full encoders retained. Apply the same deterministic encoded zero-mask, with multiplication by zero so inactive branches retain zero gradients. Inference mask is a forward argument, never checkpoint identity mutation. |
| GraphCore variants | `models/multidag_cl/graph_core_variants/{common,coop,track,dcn}.py`, corresponding runtime | Import CoopCore, TrackCore, DCNCore directly. Their strict-past DAG contract and four 300D outputs are unchanged. No copied core implementations. |
| CL7 | `models/multidag_cl/paper_reimplementation/curriculum/`, `scripts/runtime/multidag_cl_paper_reimplementation/curriculum.py`; historical CL7 UniLSTM YAML | Reuse CurriculumRuntime, scorer, balanced_stable_contiguous K=7, official_one_bucket_per_epoch. Difficulty=(within-speaker emotion shifts + speaker count)/(utterance count + speaker count). Train labels only; stable original-index ties. One additional bucket per global epoch, saturated at seven. This is the project's historically used CL7 profile, not a claim of author-official reproduction. |
| Window semantics | historical `context_w{0,1,3,5}_tav.yaml`, `context_past_all_causal_tav.yaml`, unified adjacency builder and `models/common/causal_graph/graph_builders.py` | Reuse build_causal_utterance_adjacency: previous K utterances, None=all past, 0=no past. self_loop=False satisfies existing GraphCore/MultiDAGLayer contract; current utterance enters through h0/node update. This is distinct from the paper reimplementation/TA-v0 same-speaker-count window. Frontend UniLSTM history is retained at every graph window, including w0. |
| Train/Dev protocol | TA-v0 training runtime and checkpoint selector | Reuse seed, masked CE, metrics, selector, atomic checkpoint helpers, completed-epoch resume mechanics. Adapt TAV batch/identity/CL loader only. All formal runs remain seed100/30 epochs/batch16/AdamW/clip5. |
| Canonical identity | portable TA dataset schema, manifests and canonical order | New TAV adapter validates identity without reading Test targets or indexing Test feature arrays. Test target objects are copied opaquely by the builder from the current package. No Test factory. |
| Asset alignment | asymmetric package builder; historical official-feature schema/import/conversion tools | Require explicit canonical T1024 package, canonical identity manifests, historical A1582/V342 feature source and an explicit audited audio-fingerprint bridge. Revalidate every fingerprint and one-to-one match; never guess positional alignment. Historical labels are never consumed. Historical split metadata is used only to detect conflicts, never as output authority. |
| Output | `utils/output_paths.py`, GraphCore formal runtime | Use canonical date/group/runs resolver; fixed experiment IDs, atomic exclusive creation. Resume stays in the original run. Batch evaluation/analysis uses canonical batch directories. |

## Boundaries and readiness

No real asset paths were supplied and none are discovered by scanning protected
directories. This task delivers a builder and synthetic validation, not a claimed
7433-row real TAV file. The bridge must be supplied explicitly on the asset host;
the checkout does not contain a reusable fingerprint bridge file. Missing,
duplicate, ambiguous or cross-split mapping blocks asset construction.

Test IDs/split membership and feature bytes may be carried through during asset
construction; Test label values are not indexed, compared, printed, remapped or
used for metrics. Runtime only materializes Train/Dev labels. SHA256 hashes cover
opaque files, not Test metrics. TAV-architecture/TA-active is explicitly distinct
from the earlier true-TA GraphCore screening lineage. No historical TA code or
configuration will be overwritten. No formal matrix is run locally.
