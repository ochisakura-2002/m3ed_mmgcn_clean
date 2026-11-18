# TAV Full Audit v1 implementation report

Date: 2026-11-18. WORKSTREAM = MULTIDAG.
Inspected HEAD: `8ee7d966ce85cd26f66c0a2e152d79c642b101a8`.
Workstream guard: PASS. No commit, push, branch switch, stash, reset or clean.

```text
CANONICAL_TAV_ASSET_READY = BUILDER_READY
TAV_BASELINE_READY = YES
GRAPHCORE_TAV_READY = YES
CL_OFF_READY = YES
CL7_READY = YES
TRAIN_MODALITY_CONFIGS = 14
GRAPHCORE_ADDITIONAL_CONFIGS = 6
CONTEXT_ADDITIONAL_CONFIGS = 4
TOTAL_FORMAL_TRAININGS = 24
MISSING_MODALITY_EVALS = 56
TEST_ACCESSED = NO
BASELINE_TA_MODIFIED = NO
READY_FOR_GPT_REVIEW = YES
```

YES means implemented and validated with synthetic Train/Dev fixtures. It does
not mean that real assets were built, that remote GPU training ran, or that any
model achieved a particular score. Real-asset construction remains a separate
asset-host step using explicit inputs. No real dataset or checkpoint was opened.

## Implementation and lineage

The [source audit](TAV_FULL_AUDIT_SOURCE_AUDIT.md) was written before model changes.
The new implementation lives in `models/multidag_cl/tav_full_audit_v1/` and imports
the historical PaperFormulaModalityEncoder, MultiDAGLayer, classifier, and the
existing CoopCore/TrackCore/DCNCore classes. Their source files are unchanged.
This is a project causal TAV lineage, not an author-official reproduction claim.

T1024→linear100→packed UniLSTM100, A1582→linear100→ReLU,
V342→linear100→ReLU. Concatenation order is A,V,T. Encoded deterministic masks
retain all encoders and the same 300D h0. Four 300D graph states plus h0 give the
same 1500D classifier input for all seven active-modality choices. Inference masks
are Dev-only forward arguments and never rewrite checkpoint identity.

| Core | Parameter count (all seven masks keep the complete architecture) |
| --- | ---: |
| Original | 5,975,110 |
| Coop | 5,979,926 |
| Track | 1,638,306 |
| DCN | 1,278,306 |

All seven modality-training architectures are Original, with exactly 5,975,110
parameters and identical tensor shapes/initialization under the same seed.
TAV-architecture/TA-active does not reuse or reinterpret true-TA screening runs.

`past_all_causal` uses every strict-past graph predecessor. `wK` uses the previous
K utterances and w0 uses no past graph edges. The existing common causal graph
builder is called directly with self_loop=False. Current-utterance computation
remains in h0/node update; the UniLSTM retains its causal history even for w0.
Thus this audits graph context, not a memoryless text encoder. The historical
same-speaker-count window used in TA-v0 is a different topology. Causality tests
cover the model boundary on supplied utterance features; upstream feature
extraction locality must remain part of the real asset's provenance audit.

CL7 directly reuses the existing CurriculumRuntime/scorer/partitioner/schedule:
within-speaker emotion shifts, balanced stable contiguous seven buckets, and one
additional bucket per global epoch. Only Train labels determine difficulty.
CL_OFF exposes all buckets from epoch 1; both modes use the same deterministic
loader implementation and seed policy. Bucket membership and its SHA256 are
persisted and checked on resume. No new curriculum or auxiliary loss was added.

## Asset builder and isolation

Builder: `scripts/data/build/multidag_cl/build_tav_t1024_v1.py`.
Adapter: `datasets/iemocap/tav_full_audit_dataset.py`.

Inputs are explicit: current canonical T1024 TA PKL, current model-input CSV,
current provenance/order CSV, historical nine-item PKL, its exact split JSON,
and the audited fingerprint-bridge CSV. No asset discovery scans are performed.
The historical split JSON uses the existing `project_pkl_sha256` and
`train_dialogue_ids`, `dev_dialogue_ids`, `test_dialogue_ids` contract.

Bridge columns:

```text
utterance_id,historical_dialogue_id,historical_row_index,audio_sha256
```

Row indices are zero-based. Fingerprints are SHA256 of contiguous little-endian
float32 A1582 model-input bytes. Historical float64 JSON-derived arrays are cast
to those exact model-input bytes before matching. Every bridge row is checked
against both source arrays; duplicate fingerprints or reused rows block the
build. Required counts are 7433 utterances/151 dialogues, with Train 4830/100,
Dev 980/20 and Test identity metadata 1623/31. Matching must be 7433/0/0/0 for
matched/unmatched/ambiguous/split_conflict.

Only historical PKL indices 0, 4, 5 supply row metadata/A/V. Historical targets
and vocabularies are never consumed. The current package owns IDs, speakers,
labels and split membership. Its T/A arrays and target objects are retained.
Test target objects are passed through opaquely during construction: no indexing,
length checks, comparisons, conversions, remapping or reporting of their values.
The generated model manifest leaves Test target columns blank. Test identities
and split counts are validated; runtime never indexes Test feature arrays or
creates a Test dataset. Opaque whole-file hashing/unpickling is distinct from
consuming Test target values. Synthetic poison objects verify this boundary.

The builder refuses an existing destination and creates:

```text
iemocap_asym_tav_t1024_v1.pkl
model_input_manifest_tav_t1024_v1.csv
tav_t1024_provenance_manifest_v1.json
```

Provenance includes source hashes, T/A/V dimensions, mapping method, current
label/split authority, row/order digest, output PKL/CSV SHA256 and builder SHA256.
Runtime verifies the complete sidecar, hashes, exact canonical counts and
Train/Dev labels. A partial construction is not a certified package. No real
asset, manifest or fabricated 7433-row feature file was generated on Windows.
The required real fingerprint bridge is external and was not present among the
explicit task inputs; the builder will not guess one from row positions.

## Matrices, execution and outputs

Independent configs and frozen CSVs:

```text
configs/multidag_cl/tav_full_audit_v1/
  baseline/                 2 YAML
  graph_core/               6 YAML
  modality_train/          12 YAML
  context/                  4 YAML
  TAV_FULL_AUDIT_MATRIX.csv             24 unique training rows
  TAV_MISSING_MODALITY_EVAL_MATRIX.csv   56 unique Dev rows
```

The baseline rows participate in both modality×CL and graph×CL comparisons and
are trained once. All rows use seed100, 30 epochs, batch16, AdamW lr5e-4,
betas(0.9,0.999), eps1e-6, weight_decay0, global clip5, no scheduler/class weights/
label smoothing/early stopping. Best selection is Dev Weighted-F1 descending,
Dev loss ascending, then earlier epoch. Test remains HOLD.

The frontend source config is pinned by SHA256 after UTF-8/LF normalization, so
Windows CRLF checkout and Linux LF checkout have the same design identity.
All 24 configs and both matrices are validated against the frozen design before
data access. Existing TA/historical configs were not edited or moved.

Executor: `scripts/workflows/benchmarks/run_tav_full_audit_v1.py`.
It defaults to a no-data/no-write plan, then executes phases 1–6 in the requested
order when explicitly invoked on the remote CUDA host. The whole matrix is fixed:
no scores affect future membership, no automatic multi-seed expansion. All
existing run paths are preflighted before the first epoch. Only completed,
non-synthetic, exact-config/data, 30-epoch runs with an intact selected checkpoint
can be skipped. Failed/incomplete/mismatched runs stop the batch.

Runs use `outputs/<DATE>/<group>/runs/<experiment_id>`. Groups include TAV, core,
CL, active modalities and context. Creation is exclusive; concurrent/resume
execution uses an exclusive run lock. Explicit completed-epoch resume retains
the original run directory, optimizer/RNG, selector and historical best. It can
also finish Dev export after a final-epoch export failure without retraining.
Checkpoint/log/manifest disagreement fails closed; no checkpoint guessing occurs.

Batch state/logs/Dev masks use the shared resolver under the
`tav_full_audit_v1` group with batch IDs. Phase 6 reloads each of eight selected
full-TAV checkpoints and performs seven Dev masks, zero training steps, with a
single unchanged checkpoint SHA per seven evaluations. A failed batch can be
invoked with a new batch ID after explicit run repair/resume; existing complete
training jobs are safely skipped. Existing evaluation artifacts are not overwritten.

Summarizer: `scripts/analysis/models/multidag_cl/summarize_tav_full_audit_v1.py`.
It reads explicit run manifests, `validation_best_metrics.json`, selected
checkpoint hashes and exact batch-owned Dev mask files. It never searches Test
reports or loads checkpoint target tensors. It writes six requested CSVs under
canonical batch analysis directories; incomplete rows retain blank metrics.
Table row counts are 24/8/20/14/5/56 for full/graph/CL/train-modality/context/
missing-modality respectively. Accuracy, Weighted-F1, Macro-F1, UAR, loss,
best_epoch and parameter_count are included.

## Validation

- Joint targeted regression: **533 passed in 39.89s**, including new TAV tests,
  original TA-v0/GraphCore, portable TA dataset and historical CL7/UniLSTM/masks.
- Final focused TAV regression after the last resume/path and fixture-cleanup
  checks: **78 passed in 17.42s**. This includes builder file roundtrip/hashes, six result tables,
  final-export recovery, moved-run refusal and CRLF/LF source-hash equivalence.
- Syntax validation passed for all 19 new Python files. Protected original
  TA/core/runtime/config paths have zero tracked or untracked source changes.
- All 24 YAMLs and exact 24/56 matrices passed config-only validation; fixed-order
  no-data plan passed. Original CPU synthetic two-epoch smoke, selected-best
  reload and Dev evaluation passed. The parameter count is 5,975,110.
- Coverage includes all four core identities and 300×5→1500 contract, future
  perturbation and prefix invariance, all seven deterministic masks, same
  parameter initialization, visual reorder mapping and fail-closed conflicts,
  poisoned Test targets/features, CL schedule identity, data/config collision,
  failed-run status, checkpoint identity, and bit-exact completed-epoch resume.
- Local Python is the existing `m3ed_mmgcn` environment, torch 2.6.0+cpu. No
  dependencies were installed. Python 3.9 private-directory ACL behavior required
  an in-process test-only mkdir mode compatibility shim and a short pytest
  basetemp path; neither production TA code nor environment packages were edited.
- No full repository pytest claim is made. This task ran synthetic targeted
  checks only. Formal GPU trainings started: 0. Real Dev evaluations: 0.

## Commands for review and subsequent remote use

All commands run from the repository root. Set the variables to explicit trusted
asset-host inputs; they are not auto-discovered or supplied by this task.

No-data review:

```bash
python scripts/workflows/benchmarks/run_tav_full_audit_v1.py
python scripts/models/multidag_cl/tav_full_audit_v1/train.py \
  --config configs/multidag_cl/tav_full_audit_v1/baseline/tav_original_cloff_tav_past_all_causal_s100.yaml \
  --synthetic-smoke
```

The smoke stores only tiny synthetic artifacts under an isolated `tmp/tav_smoke_*`
directory. It cannot consume real paths or use CUDA. A normal Linux regression:

```bash
python -m pytest tests/models/multidag_cl/tav_full_audit_v1 \
  tests/models/multidag_cl/graph_core_variants tests/models/multidag_cl/ta_v0 \
  tests/data/test_ta_v0_dataset.py -q -p no:cacheprovider
```

Asset construction on the asset host (not run in this task):

```bash
python scripts/data/build/multidag_cl/build_tav_t1024_v1.py \
  --canonical-asset "$CANONICAL_TA_T1024" \
  --model-manifest "$CANONICAL_MODEL_MANIFEST" \
  --order-manifest "$CANONICAL_ORDER_MANIFEST" \
  --historical-asset "$HISTORICAL_TAV_PKL" \
  --historical-split-manifest "$HISTORICAL_SPLIT_JSON" \
  --fingerprint-bridge "$AUDITED_AUDIO_BRIDGE" \
  --destination "$NEW_TAV_DIRECTORY"
```

Later remote execution, after real asset certification and review (not run):

```bash
python scripts/workflows/benchmarks/run_tav_full_audit_v1.py --execute \
  --asset-path "$TAV_ASSET" --model-manifest-path "$TAV_MODEL_MANIFEST" \
  --provenance-manifest-path "$TAV_PROVENANCE_JSON" \
  --batch-id "$TAV_BATCH_ID" --experiment-date "$TAV_DATE" --device cuda
```

Explicit repair/resume for an incomplete run (not an automatic launcher action):

```bash
python scripts/models/multidag_cl/tav_full_audit_v1/train.py --formal-train-dev \
  --config "$TAV_CONFIG" --asset-path "$TAV_ASSET" \
  --model-manifest-path "$TAV_MODEL_MANIFEST" \
  --provenance-manifest-path "$TAV_PROVENANCE_JSON" \
  --resume "$ORIGINAL_RUN/checkpoints/last_model.pt" --device cuda
```

Summary after the remote batch, or a partial-status summary (not run on real data):

```bash
python scripts/analysis/models/multidag_cl/summarize_tav_full_audit_v1.py \
  --batch-manifest "outputs/$TAV_DATE/tav_full_audit_v1/manifests/batches/$TAV_BATCH_ID/batch_manifest.json"
```

## Worktree boundary

All task source/config/docs/tests are new files. Pre-existing dirty `.gitignore`,
deleted historical documents and unrelated untracked files were preserved.
No existing TA baseline/core/runtime/config source was changed. Synthetic test
artifacts are ignored under `tmp/`; no real `outputs/`, data, weights or remote
workspaces were modified. `git diff --stat` shows the pre-existing tracked
changes; new task files remain untracked and are listed separately because no
staging or commit was requested.

The test fixture removes only its own verified, freshly allocated synthetic
directory after use. Earlier task-owned test artifacts were cleaned during
delivery; the independent CLI smoke artifact is retained. Existing user temp
files and earlier user cleanup deletions were not restored or removed.

Task inventory: 47 new files: 19 Python files, 24 YAML files, two matrix CSVs,
and two Markdown reports. No task-owned tracked file was modified.
