# IEMOCAP split protocol audit

## Scope

This audit follows the split path used by the active MultiDAG+CL configuration:

1. `configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_stable_candidate.yaml`
2. `scripts/baselines/train_multidag_cl.py`
3. `datasets/iemocap/__init__.py`
4. `datasets/iemocap/official_feature_dataset.py`
5. `scripts/analyze/diagnose_iemocap_splits.py`

The audit and diagnostic extension do not change split membership, dataset return values, dataloader behavior, model code, training code, or YAML parameters.

## Active call chain

- The YAML sets `system.seed: 42`, `dataset.valid_ratio: 0.1`, and `dataset.val_split_strategy: official_prefix` (`context_w5_tav_stable_candidate.yaml:6,14-15`).
- The training entry imports `build_iemocap_dataloader` from `datasets.iemocap` (`scripts/baselines/train_multidag_cl.py:30`). Its `build_dataloader` function passes the configured feature path, validation ratio, strategy, and seed to the IEMOCAP builder (`scripts/baselines/train_multidag_cl.py:417-438`). Train and validation loaders are constructed through this function (`scripts/baselines/train_multidag_cl.py:1177-1178`).
- The diagnostic script directly constructs `IEMOCAPOfficialFeatureDataset` with the same four values (`scripts/analyze/diagnose_iemocap_splits.py`, `build_dataset`). It does not implement an independent split.

## Audit answers

### 1. Where do the official train/test dialogue IDs come from?

They are read from the feature pickle configured at:

```text
third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl
```

`IEMOCAPOfficialFeatureDataset` loads a 9-item pickle and assigns its last two entries to `self.trainVid` and `self.testVid` (`datasets/iemocap/official_feature_dataset.py:104-123`). `_build_split_ids` copies these objects in their stored order (`datasets/iemocap/official_feature_dataset.py:160-162`).

The project does not reconstruct official membership by scanning filenames, parsing sessions, or applying a new official-split rule. The pickle's `trainVid` and `testVid` are the source of truth.

### 2. How is validation constructed from official train?

The adapter computes:

```python
n_val = int(valid_ratio * len(trainVid))
```

For the active `official_prefix` strategy it then uses:

```python
val_ids = trainVid[:n_val]
train_ids = trainVid[n_val:]
test_ids = testVid
```

The implementation is at `datasets/iemocap/official_feature_dataset.py:160-196`.

With `valid_ratio=0.1` and the reported official-train pool of 120 dialogues, `n_val=int(0.1 * 120)=12`; the remaining 108 dialogues form the realized training split. The observed 108/12 split is therefore consistent with this exact prefix construction.

### 3. What seed does the validation split use?

The configured seed is **42**. Both the training and diagnostic paths pass it to the dataset.

However, the active `official_prefix` branch does not shuffle and never reads `self.seed`. Therefore:

```text
configured seed: 42
effective validation-membership seed: not used / N/A
```

The seed affects validation membership only if `val_split_strategy` is `random`, whose separate branch uses `random.Random(self.seed)` (`datasets/iemocap/official_feature_dataset.py:183-188`).

### 4. Is this random, ordered, or session-aware?

It is an **ordered dialogue-prefix split** in the sequence stored in `trainVid`.

- It is not a random dialogue split.
- It is not a session-aware split.
- It is not a speaker-aware split.
- A sampler or training-loader shuffle may change iteration order, but it does not change validation membership.

### 5. Can dialogue IDs be parsed into session and speaker?

For a canonical IEMOCAP dialogue ID such as `Ses01F_impro01`, the session can be parsed reliably as `Ses01` using the anchored form:

```text
^(Ses0[1-5])[FM]_
```

The `F` or `M` immediately following `SesNN` must not be treated as the dialogue's only speaker. IEMOCAP dialogues are dyadic, and the complete participating-speaker set cannot be enumerated reliably from that single dialogue-ID marker alone.

The feature pickle provides per-utterance roles in `videoSpeakers[dialogue_id]` and utterance IDs in `videoIDs[dialogue_id]`. The extended diagnostic derives a session-qualified identity such as `Ses01F` or `Ses01M` only when all of the following hold:

1. The dialogue session parses with the strict anchored pattern.
2. The speaker, utterance-ID, and label sequences have the same length.
3. Every raw `videoSpeakers` value is `F` or `M`.
4. Every utterance ID belongs to the dialogue and ends in `_F###` or `_M###`.
5. The utterance-ID suffix agrees with the raw `videoSpeakers` value.

If any check fails, the relevant session/speaker field is left empty and the report records the failure count. The diagnostic does not infer real names or any cross-session identity relationship.

The adapter's model-facing `speaker_ids_int` values (`M -> 0`, `F -> 1`) are local role IDs, not globally unique people. Using raw `M/F` or `0/1` for cross-split overlap would create false overlap between different sessions.

### 6. Can train/validation/test share a session?

**Yes, the implementation permits it.** No split branch groups, filters, or asserts IDs by session.

- Train and validation can share a session when dialogues from that session occur on both sides of the prefix boundary.
- Train/test and validation/test session disjointness depends entirely on the contents of the pickle's `trainVid` and `testVid`.
- The code does not independently guarantee session disjointness for any pair.

Actual overlap must be computed from the realized dialogue IDs. It must not be inferred only from the intended meaning of “official train/test.”

### 7. Can train/validation/test share a speaker?

**Yes, the implementation permits it.** It does not group or filter dialogue IDs by a session-qualified speaker identity.

If train and validation share a session, they may also contain the same session actors (`SesNNF` and/or `SesNNM`). Whether either shares a speaker with test again depends on the realized IDs in the pickle.

Shared session or speaker identity is not automatically called data leakage when dialogue IDs are disjoint. It is nevertheless important for interpretation: validation on already represented sessions or speakers can be more familiar than testing on unseen sessions or speakers, and can therefore contribute to an optimistic validation-test gap.

## Dialogue-ID disjointness guarantee

For unique `trainVid` values, the prefix and suffix slices are disjoint by construction. The adapter does not assert that `trainVid` contains no duplicate IDs, and it does not assert that `trainVid` and `testVid` are disjoint. The extended diagnostic therefore calculates all three realized set intersections instead of reporting a theoretical zero.

## Local evidence limitation

At audit time, the configured feature pickle is absent from this checkout. Consequently, this document can establish the exact code path and what overlap the implementation allows, but it cannot truthfully claim the realized session or speaker intersections.

When the pickle is available, run:

```bash
python scripts/analyze/diagnose_iemocap_splits.py \
  --config configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_stable_candidate.yaml \
  --output-dir outputs/analysis/iemocap_split_diagnostics/stable_w5_extended
```

The resulting `split_overlap_summary.csv` and `split_protocol_report.md` are the authoritative realized-overlap diagnostics. Parsing coverage in the report must be checked before interpreting a zero session/speaker overlap as genuine disjointness.
