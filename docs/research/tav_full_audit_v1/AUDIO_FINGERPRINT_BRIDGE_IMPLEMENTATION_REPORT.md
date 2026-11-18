# Audio Fingerprint Bridge v1 implementation report

Date: 2026-11-18. WORKSTREAM = MULTIDAG.
Inspected HEAD: `4835de28c65cf0ba2ed8b16f9c8466617d3a8258`.
Branch: `feat/multidag-cl-paper-reimplementation`. Workstream guard: PASS.

```ini
BRIDGE_EXPORTER_READY = YES
SYNTHETIC_TESTS = PASS
EXISTING_TAV_CODE_MODIFIED = NO
READY_FOR_REMOTE_BRIDGE_BUILD = YES
```

These statuses certify the exporter and synthetic validation. The real bridge
and real TAV package were not generated or certified on Windows. No real input
PKL, Test target, Test metric, checkpoint, remote session or training was opened.
No commit, push, branch switch, remote change or stash was performed.

## Audit before implementation

Searches covered the existing data builders, diagnostics, datasets, relevant
tests and reports, with Git HEAD searches also covering locally deleted tracked
documentation. Git history was searched for fingerprint, alignment, T1024 and
AB1 work. Protected data, outputs, third_party and historical temporary assets
were not scanned. Existing unrelated dirty/deleted/untracked files were preserved.

| Existing source | Finding and reuse |
| --- | --- |
| `datasets/iemocap/tav_full_audit_dataset.py::fingerprint` | Reused directly, without changing its source. Frozen SHA256 of C-order little-endian float32 A1582 bytes; historical float64 is converted before hashing. |
| `scripts/data/build/multidag_cl/build_tav_t1024_v1.py` and `tests/models/multidag_cl/tav_full_audit_v1/test_asset.py` | Existing bridge consumer checks unique exact hashes, zero-based historical rows and cross-split conflicts. Exporter emits its required four columns unchanged. A new test passes the exported bridge to this unchanged consumer. |
| `scripts/data/build/multidag_cl/convert_official_json_to_project_pkl.py` and official-feature pipeline tests | Historical nine-item layout: item 0 contains dialogue row identities, item 4 contains A1582. Conversion intentionally does not force float32, explaining historical float64. Split JSON is bound to `project_pkl_sha256`. |
| `scripts/data/build/build_iemocap_asymmetric_dataset.py::recover_dialogue_splits` | Existing speaker-token mapping recovers dialogue/split identities only. It cannot supply an utterance audio bridge. Its split overlap, coverage and source-binding rules informed the exporter checks. |
| `datasets/iemocap/ta_v0_dataset.py`, portable-adapter report/tests | Reused `UID`, `integer`, `unique_rows`, `read_columns`, schema and assertion helpers. Exact join uses current `utterance_id`, TA dialogue/row, real dialogue, speaker and split. Provenance supplies positive unique per-dialogue canonical order. Label columns are excluded. |
| `tests/models/multidag_cl/ta_v0/test_t1024_ab1.py`, commit `daf0957` | AB1 freezes text dimension 1024 and canonical audio 1582. No reusable T1024 reconstruction or 7433-row fingerprint-export tool was found in the inspected source/history. Earlier real alignment evidence is not claimed as a locally rerun audit. |
| TAV Full Audit source/implementation reports | Confirm the same float32 fingerprint definition, fixed 7433/151 universe, current split authority and opaque Test targets. |
| `scripts/runtime/multidag_cl_ta_v0/formal.py` | Reused streaming `sha256_file` and `git_identity`; no training function is called. |

## Delivered files and contract

Only three files are added:

- `scripts/data/build/multidag_cl/build_audio_fingerprint_bridge_v1.py`
- `tests/data/build/multidag_cl/test_audio_fingerprint_bridge_v1.py`
- This report.

All six CLI paths are required: `--canonical-asset`, `--model-manifest`,
`--order-manifest`, `--historical-asset`, `--historical-split-manifest`, `--output`.
There is no asset discovery, count override, overwrite option or production
synthetic mode. Small fixtures use the in-memory count hook only.

Canonical A arrays must be finite float32 with shape `[N,1582]`; text metadata
must declare T1024. The exporter does not access text feature arrays. Historical
A must have shape `[N,1582]`, be finite float32/float64, and remain finite after
float32 conversion. No scaling, normalization, approximate matching, rounding
tolerance or positional fallback is applied. Historical IDs need not resemble
canonical IDs; only their dialogue keys and row counts locate audio rows.

Historical PKL access is limited to items 0 and 4. Historical labels, vocabulary,
speaker tokens, V/T arrays and embedded train/test lists are not indexed. Current
labels, vocabulary and label columns are also not indexed, including Train/Dev.
Test IDs, membership and audio bytes participate solely in the required identity
bridge. No Test dataset loader or evaluation is constructed.

Current PKL splits are authoritative and must agree with both current manifests.
Historical split JSON is used only for source binding, coverage and conflict
audits; it never sets or repairs a canonical split. The row order follows the
current model-manifest CSV, matching the existing builder, independently of PKL
dictionary insertion order and order-manifest row order. Canonical order values
are validated without renumbering.

## Fail-closed acceptance and publication

Production requires exactly 7433 canonical and historical rows, 151 dialogues,
Train 4830/100, Dev 980/20, and Test identity/audio 1623/31. All historical rows
must be used once, with 7433 exact matches. Reject any duplicate current or
historical fingerprint, multiple historical candidates, ambiguous or unmatched
row, reused historical location, unused historical row, or split conflict.
Duplicate counters count excess rows beyond unique fingerprints; ambiguity
counts current utterances with multiple historical candidates.

CSV columns, in order:

```text
utterance_id,historical_dialogue_id,historical_row_index,audio_sha256,canonical_split,historical_split
```

Historical row indices are zero-based. CSV bytes use UTF-8 without BOM and LF
line endings. The audit path is derived from the explicit output CSV path, e.g.
`audio_fingerprint_bridge_t1024_v1.audit.json`. It records all five source file
SHA256 values, split/row/dialogue counts, match and failure counters, algorithm,
bridge SHA256, builder/exporter/shared-helper SHA256, UTC timestamp, Git HEAD and
dirty state, plus zero label/loader/evaluation inspection counters. Source file
and source code hashes are rechecked immediately before publication.

Validation errors create neither final CSV nor audit. Both byte streams are
staged beside the requested output, flushed and fsynced, then published with
exclusive atomic hard links: audit first, CSV last. Existing paths are never
overwritten. Ordinary publication failures remove only this invocation's own
links and staging files. A process crash can leave an orphan audit/staging file;
it cannot expose a partial CSV. Such an audit alone is not a usable bridge. The
destination filesystem must support hard links; unsupported publication fails.
Audit timestamps and Git dirty state can differ across runs; identical inputs
and model-manifest row order yield identical bridge bytes and bridge SHA256.

## Validation

Final scoped regression: **97 passed** (37 new exporter cases plus 60 existing
TAV asset / T1024 AB1 cases). All inputs are small synthetic fixtures; no fake
7433-row bridge is generated.

New coverage includes perfect mapping, actual float64 rounding into float32,
both duplicate sources, unmatched/ambiguous/reused-row rejection, cross-split
conflicts, historical/current label isolation via throwing sentinels, deterministic
ordering, SHA256 stability with independent `struct.pack` bytes and fixed digest,
noncontiguous/big-endian inputs, finite/dimension/count/identity/order guards,
unchanged builder compatibility, CLI wiring, provenance roundtrip, input/code
mutation, exclusive destinations and publication failure rollback.

The first new-test run found an incorrect expected constant in the SHA256 test;
it was corrected against the independent packed-byte digest. An unadjusted
joint run had 95 passes and two existing AB1 fixture setup errors because the
system pytest temporary directory is inaccessible under this Windows ACL.
Rerunning with a transient pytest plugin that creates `tmp_path` using ordinary
workspace `Path.mkdir()` produced 97 passes; no existing test or assertion was
changed. Conda output capture also required UTF-8/live output. The successful
joint rerun used the existing environment's interpreter directly because Conda
does not support a multiline `python -c` argument. No dependency was installed.

Final CLI `--help`, both new Python AST parses and `git diff --check` passed.
SHA256 comparison of 56 pre-existing protected model/runtime/config/builder/
dataset/report files found zero changes. The complete tracked diff SHA256 stayed
`dd4fb8c9f374199ea332bfd67cac949c311189d67a1ba7f47f0380d421ec8555`,
confirming preservation of all pre-existing tracked changes. The normal
`git diff --stat` still shows only those earlier 135 changed/deleted tracked
files; the three task files are new and remain untracked, without staging.

Portable regression command (normal Linux temp permissions):

```bash
python -m pytest tests/data/build/multidag_cl/test_audio_fingerprint_bridge_v1.py \
  tests/models/multidag_cl/tav_full_audit_v1/test_asset.py \
  tests/models/multidag_cl/ta_v0/test_t1024_ab1.py -q -p no:cacheprovider
```

## Ubuntu build handoff — not executed here

Run from the registered MULTIDAG repository root after making the reviewed
exporter available there. These relative paths express the user-supplied asset
locations; verify them explicitly on that host. The guard must pass before work.

```bash
python scripts/dev/check_workstream.py multidag

ASSET_ROOT=../data/IEMOCAP/asymmetric_merc_v1
HISTORICAL_ROOT=data/processed/iemocap/multidag_cl_paper_data_reproduction

python scripts/data/build/multidag_cl/build_audio_fingerprint_bridge_v1.py \
  --canonical-asset "$ASSET_ROOT/ta/iemocap_asym_ta_t1024_ab1_v1.pkl" \
  --model-manifest "$ASSET_ROOT/metadata/model_input_manifest_t1024_ab1_v1.csv" \
  --order-manifest "$ASSET_ROOT/metadata/iemocap_asym_manifest_v1.csv" \
  --historical-asset "$HISTORICAL_ROOT/IEMOCAP_features_multidag_cl_official.pkl" \
  --historical-split-manifest "$HISTORICAL_ROOT/official_split_manifest.json" \
  --output "$ASSET_ROOT/metadata/audio_fingerprint_bridge_t1024_v1.csv"
```

Accept the remote result only when all fixed counts pass, the audit is complete,
its bridge SHA256 matches the actual CSV, and all error counters are zero.
The existing TAV builder can then consume that CSV through its unchanged
`--fingerprint-bridge` argument. Real TAV construction and training remain later
tasks. This Windows task does not change TAV models, GraphCore, CL, training
configs, 24/56 matrices or existing builder semantics.
