# GraphCore paper mechanism audit

WORKSTREAM = MULTIDAG. Audit date: 2026-11-18. These are project adaptations,
not author-official reproductions. No MERC performance claim is made.

## Sources actually inspected

Local originals are in `tmp/assistant_work/dag_core_papers/` (ignored assets).
Text extraction used the existing bundled pypdf; method pages were rendered and
visually inspected. No package was installed. Main methods and relevant appendices
were read before model implementation.

| Local PDF | SHA-256 | Method evidence |
| --- | --- | --- |
| `pei24a.pdf` | `3416fddcf186dc1ffe2ed8fe4f9cb8758e2306f499142658746aca8b829b835c` | pp. 3-5, Definition 3.1, Eqs. 1-5; p. 13 Algorithms 1-2 |
| `finkelshtein24a.pdf` | `dd68a5f806f2815c5c5ad4360a4bd3787a863576aa32ea555ba43226e372851a` | pp. 2-5, Section 4 Eqs. 1-2; Appendix E.1, p. 19 |
| `Rey 等 - 2025 - Directed Acyclic Graph Convolutional Networks.pdf` | `270e62f2662e1a934fb0e727186b5b7068beecb1ab3a1831e462e91eec3b4671` | pp. 3-7, Sections III-V, Eqs. 3-7, 13-18 |
| `fesser26b.pdf` | `8d6d62f9dba6733af986e8c57a5c7046b204e09c679f11c4b80634183f7b4959` | pp. 4-6, Definition 2, Eqs. 4-7, Propositions 6-7 |

Primary source links: [MTGCN](https://proceedings.mlr.press/v235/pei24a.html),
[CO-GNN](https://proceedings.mlr.press/v235/finkelshtein24a.html),
[DCN 2025 preprint](https://arxiv.org/abs/2506.12218v1),
[Dune](https://proceedings.mlr.press/v306/fesser26b.html).
The local Rey 2025 PDF is the implementation reference, not a later revision.

## MultiDAG-Track

- SOURCE_MECHANISM: MTGCN maintains category-specific message tensors on
  isomorphic copies of the same graph. Loading uses node-track affiliations F;
  each track propagates independently with normalized adjacency and initial
  residual alpha*M0 (Eq. 2), without per-round propagation weights/nonlinearities.
  Acquisition selects affiliated tracks and applies W_Z (Eq. 3). Learned F uses
  auxiliary GCN representations, category prototypes and labeled/pseudo-labeled
  representatives (Eqs. 4-5). Multi-stage retraining appears in Algorithm 2.
- OUR_ADAPTATION: two persistent **speaker-origin** tracks, not emotion tracks.
  Track 0 belongs to the first observed speaker; track 1 to the other speaker.
  Every node/edge remains present in both propagation graphs. Load each h0 only
  into its speaker's track. Use receiver-row-normalized (A+I), rather than
  symmetric degree normalization, to avoid future out-degree information.
  Four rounds use `M_l=P M_(l-1)+0.1 M0`. At each receiver, acquire the current
  speaker's and other speaker's messages through a bias-free 600->300 projection.
  Readouts never feed back into the persistent tracks. Track swaps under speaker
  relabeling therefore preserve outputs. Same/other is relative at acquisition.
- NOT_TRANSFERRED: emotion labels/prototypes, auxiliary GCN, pseudo labels,
  affiliation regularizer, multi-stage training, undirected propagation, claimed
  emotion-semantic separation guarantees. Speaker separation does not establish
  emotion homophily or prove that oversmoothing exists in this baseline.
- CAUSAL_RISK: symmetric normalization or full-dialogue speaker statistics could
  leak future structure. Only past edges, the current speaker and first speaker
  identity are used. More than two speaker groups is rejected explicitly.
- INPUT_CONTRACT: float32 h0 [B,T,300], unchanged bool receiver/source DAG,
  named same/different speaker relations, contiguous valid-prefix mask.
- OUTPUT_CONTRACT: four 300D readouts. Original output is final acquired Z;
  adapter acquires each of four meaningful propagation depths, retaining all
  depth information for the frozen classifier. Four projections add 720,000
  trainable parameters inside the core; no external head is added.
- PARAMETER_IMPACT: two parameter-free persistent tracks plus four 600x300
  readout matrices; all original DAG parameters are removed from the variant.
- IMPLEMENTATION_COMPLEXITY: O(B*T^2*H*4 + B*T*H^2*4), two message channels;
  no graph copies on disk, no extra dependency.

## MultiDAG-Coop

- SOURCE_MECHANISM: CO-GNN predicts probabilities for STANDARD (listen and
  broadcast), LISTEN, BROADCAST, ISOLATE. An action GNN sees node/neighborhood
  states (Eq. 1); straight-through Gumbel-softmax samples actions. The environment
  GNN aggregates only broadcasting neighbors when the receiver listens (Eq. 2).
  Appendix E.1 describes learned inverse temperature; the paper shares its
  action network across layers in the discussed efficient formulation.
- OUR_ADAPTATION: deterministic differentiable four-action distribution
  `softmax(Linear(H,4))` at each layer. Sender marginal is p(S)+p(B); receiver
  marginal is p(S)+p(L). Multiply the **already normalized** baseline attention
  weight by both marginals, without renormalizing. The sender uses completed
  current-layer state, the receiver its previous-layer state, following the
  original MultiDAG sequential schedule. Keep baseline relation value transforms
  and dual-GRU updates. A small per-layer local policy replaces the action GNN;
  policy weights are not shared across depths in this adaptation.
- NOT_TRANSFERRED: sampling, Gumbel noise, learned temperature, discrete edge
  removal, action-GNN look-ahead, random-policy expressivity claims and shared
  policy weights. Multiplying marginals is a soft communication surrogate, not
  exact expectation of the nonlinear discrete CO-GNN update.
- CAUSAL_RISK: policies must never inspect descendants. Receiver policies are
  local; sender states have already completed in the past-only DAG. Receiver
  factors must not be renormalized away, including on singleton neighborhoods.
- INPUT_CONTRACT: the same h0/DAG/relations/mask contract.
- OUTPUT_CONTRACT: four sequential 300D states. Source already has depth-wise
  states; the adapter exposes four, with no representation-dimension adapter.
- PARAMETER_IMPACT: original graph-layer parameters plus 4*(300*4+4)=4,816
  policy parameters. No loss, encoder, classifier or dropout change.
- IMPLEMENTATION_COMPLEXITY: baseline sequential complexity plus local policies.

## MultiDAG-DCN (prefix PDCN)

- SOURCE_MECHANISM: for weighted DAG adjacency A, `W=(I-A)^-1` (Eq. 4) is
  weighted transitive closure. D_k selects k and its ancestors. Causal shift
  `S_k=W D_k W^-1` (Eq. 6) keeps common-ancestor contributions. Columns of W
  form a simultaneous eigenbasis, with binary eigenvalues from D_k. DCN stacks
  `ReLU(sum_k S_k X Theta_lk)` (Eq. 14); its general parameters scale with N.
  PDCN applies a shared MLP to each `S_k X` and sums branches (Eqs. 17-18).
- OUR_ADAPTATION: choose PDCN's shared four-layer, bias-free 300->300 ReLU MLP.
  Keep exact causal shifts, not adjacency-power convolutions. Use positive,
  past-row-normalized edge weights on precisely the baseline edge support.
  At receiver i, evaluate PDCN on its observed induced prefix [0,i] and read its
  last row. Equivalently, sum branches k<=i only. This is **prefix PDCN**, not
  full-dialogue PDCN. Reachability is computed separately with boolean algebra,
  never inferred from floating point path magnitudes. W and W^-1 do not depend
  on features or future out-degrees. No branches are sampled.
- NOT_TRANSFERRED: general DCN's node-index-specific filters (variable N would
  require parameter allocation/padding conventions); full-dialogue PDCN branch
  collection, graph regression head, original task/loss/training protocol.
- CAUSAL_RISK: although each S_k is lower triangular, summing future-indexed
  branches can make a prefix depend on how many future nodes exist. Prefix
  evaluation removes that structural leak. Both future-feature perturbation and
  prefix-extension invariance must pass. The conversation adjacency has a
  preceding-node chain, so its reachability is typically a total order; this
  limits the diversity of shifts and is an expressivity caveat, not a violation
  of the source assumptions. Summing branches can grow norms with length.
- INPUT_CONTRACT: same variable-length h0/DAG/relations/mask. Speaker metadata
  is validated but the source PDCN operator is speaker-agnostic after graph build.
- OUTPUT_CONTRACT: source PDCN returns final summed Z; adapter additionally
  sums each of its four MLP depth states to supply h1..h4. Intermediate readouts
  cost no extra trainable parameters and do not feed back into branches.
- PARAMETER_IMPACT: 4*300*300=360,000 core parameters, independent of dialogue
  length. No trainable adjacency, closure, branch-index coefficient or head.
- IMPLEMENTATION_COMPLEXITY: dense triangular solve O(B*T^3), O(B*T^2*H*4)
  intermediate branch storage/work plus O(B*T^2*H^2*4) MLP work. Designed for
  dialogue-scale DAGs; no formal speed claim. Core execution uses float32.

DCN and PDCN are not interchangeable: PDCN is selected specifically for shared
parameters across different-length dialogues. Prefix-induced evaluation retains
the exact PDCN formula on each observed graph, so no structural incompatibility
requires blocking this variant. Equation-level and prefix-equivalence tests are
required; merely passing shape tests is insufficient evidence.

## Dune

- SOURCE_MECHANISM: skew-Hermitian projection involving A and A-dagger,
  followed by a complex matrix exponential; see Definition 2, Eqs. 4-7.
- OUR_ADAPTATION: none this round; compatibility analysis only.
- NOT_TRANSFERRED: full Dune convolution, transformer/positional-encoding path,
  complex classifier and training model.
- CAUSAL_RISK: bidirectional propagation conflicts with past-only full-graph
  inference. Directedness alone does not establish causal compatibility.
- INPUT_CONTRACT: source node features and directed adjacency/edge features.
- OUTPUT_CONTRACT: source complex features, projected to real for prediction;
  no executable MultiDAG adapter is introduced.
- PARAMETER_IMPACT: zero in this task.
- IMPLEMENTATION_COMPLEXITY: deferred; see DUNE_COMPATIBILITY_ANALYSIS.md.
