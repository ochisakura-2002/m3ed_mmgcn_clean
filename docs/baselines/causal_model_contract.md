# MERC strict causal model contract

Contract version: `1.0`

## Scope

For a dialogue `u_1, u_2, ..., u_T`, the prediction for utterance `u_t` may
depend only on `u_1, ..., u_t`. It must be invariant to every permitted change
of `u_{t+1}, ..., u_T` when the model is in evaluation mode.

This contract separates model causality from feature-extractor causality. A
model can satisfy this document while consuming precomputed utterance features
whose upstream extraction process is not known to be online-causal.

## Allowed operations

1. Batch complete dialogues and compute losses for all valid time steps at once.
2. Use the current utterance's text, audio, and visual features.
3. Use current and historical speaker identities.
4. Use a unidirectional GRU, LSTM, or RNN.
5. Use attention whose mask is lower triangular and is applied before softmax.
6. Use graph edges `adj[target=i, source=j]` only when `j <= i`.
7. Apply normalization independently to each utterance, such as LayerNorm over
   its feature dimension.
8. Use padded tensors when padded positions cannot enter a valid prediction.

## Prohibited operations

1. `bidirectional=True`, BiGRU, BiLSTM, or equivalent reverse recurrence.
2. An edge from a future source node `j > i` into target node `i`.
3. Full-dialogue self-attention without an effective causal mask.
4. Mean/max/attention pooling over the complete time axis followed by writing
   the pooled representation back into per-utterance predictions.
5. Gates, normalization statistics, or graph weights for `u_t` that depend on a
   future utterance.
6. Inputs for `u_t` constructed using future labels.
7. Test metrics used to select an epoch, checkpoint, hyperparameter, seed, or
   validation session.
8. Merely zeroing future values while still exposing their positions to an
   unmasked normalization, attention, recurrent encoder, or graph computation.

## Model-level verdict

- `model_level_strict_causal`: static review and the required dynamic tests pass.
- `model_level_noncausal`: at least one demonstrated future-information path
  exists, or a future perturbation changes a current/history logit.
- `unable_to_determine`: the implementation or required evidence is unavailable.

Inventory documents may use these staging labels before the final verdict:

- `strict_model_causal`
- `causal_candidate_needs_dynamic_test`
- `noncausal`
- `unable_to_determine`

## End-to-end verdict

Models consuming the current MMGCN-style PKL features must use:

`end_to_end_causal_unverified_features`

or the metadata value:

`utterance_level_but_extractor_not_fully_verified`

Passing model tests must not be reported as a fully verified real-time causal
system until each upstream text/audio/visual extractor is independently shown
not to use future dialogue context.

## Required dynamic evidence

With `model.eval()` and dropout disabled, choose a valid cutoff `t` that leaves
at least one valid future utterance.

1. Future perturbation invariance: independently replace future T/A/V features
   with zero, random noise, and cross-sample shuffles. Compare all logits at
   positions `<= t`.
2. Prefix/full equivalence: compare the full-dialogue logit at `t` with the final
   logit from the prefix `u_1...u_t`, with lengths, masks, speakers, and padding
   truncated consistently.
3. Future gradient: differentiate a current logit with respect to future T/A/V
   inputs. Every maximum future gradient must be zero within tolerance.
4. Structural check: a causal adjacency must contain no `source_time >
   target_time` edge and no cross-dialogue edge.

The default pass tolerance is `1e-6`; reports must also expose the result at
`1e-5`. Differences must never be rounded away or hidden.

## Training and selection protocol

The best checkpoint is selected using validation `Weighted-F1`. Test is run only
after checkpoint selection and cannot drive a scheduler or early-stopping rule.
The benchmark also records Accuracy, Weighted-F1, Macro-F1, UAR, and loss, plus
confusion matrices and per-class metrics.

