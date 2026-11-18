# Dune compatibility analysis

DUNE_ANALYSIS = COMPLETE. Analysis only; no Dune model or dependency installed.
The local source, hash and inspected sections are recorded in
[paper mapping](PAPER_TO_MULTIDAG_MAPPING.md). Primary source:
[Fesser et al., ICML 2026](https://proceedings.mlr.press/v306/fesser26b.html).

## Mechanism and theorem scope

Definition 2 (p. 4) constructs a skew-Hermitian generator
`Pi(A)=i*alpha/2*(A+A^dagger)+beta/2*(A-A^dagger)`, with alpha=beta=1/2 in
the experiments, and uses `exp(t*Pi(A))*X*W`. Edge features may supply complex
scalar edge weights. Taylor approximation implements the exponential.
The projection includes both edge directions even though direction is encoded.

The strongest no-oversmoothing statement (Proposition 6, p. 5) assumes no
nonlinearities and unitary feature matrices W; the exact gradient norm argument
has the same qualification. It is not a guarantee for arbitrary ReLU, GRU,
real-component readout or unconstrained feature matrices. The directed energy
definition assumes positive in/out degrees; a causal DAG's roots/sinks do not
automatically satisfy that hypothesis. Therefore those theorem statements
cannot be copied directly onto the current MultiDAG.

## Six decisions for this project

1. **Depth:** the pathology can occur in shallow networks depending on operators,
   but the paper's motivation and >100-layer experiments concern accumulated
   contraction/expansion. Four layers alone neither prove nor disprove a problem.
   Sequential MultiDAG also propagates through a dialogue inside a single layer,
   so effective path length differs from a synchronous four-hop GNN.
2. **Current evidence:** no h0..h4 similarity, energy, rank or Jacobian evidence
   was supplied for this strong baseline. There is insufficient mechanism
   evidence to prioritize a Dune training variant now. No diagnostic or MERC
   performance result is fabricated from paper results.
3. **Dependencies:** the mathematical core can be prototyped using existing
   PyTorch complex tensors, matrix_exp or sparse Taylor products; an additional
   package is not mathematically necessary. Adopting the complete author
   benchmark code may require its own graph/transformer dependency stack and
   would need a separate environment audit. No such stack is adopted here.
4. **Causal boundary:** full-graph Dune is incompatible. Its A-dagger terms send
   future-source messages backwards on a conversational DAG. Already the first
   Taylor term has this issue. Masking after exp is not a proof of unitarity;
   discarding the upper triangle generally destroys its norm guarantees. More
   generally, a square lower-triangular unitary matrix is diagonal, so a
   nontrivial strictly past-only square unitary mixing operator cannot simply
   be obtained by imposing a triangular mask. A separate prefix-by-prefix
   architecture might read only the current node of each prefix, but differs
   from one full-graph Dune pass and needs fresh mathematical/engineering work.
5. **Future insertion:** if separately authorized and justified, replace the
   GraphCore propagation operator between h0 and depth readouts, preserving
   encoder/fusion/classifier contracts. Do not add a transformer, complex head
   or positional encoding to the current single-factor experiment.
6. **Diagnosis first:** yes. On Train/Dev only, compare h0..h4 pairwise cosine,
   centered effective rank, degree-safe edge-difference energy normalized by
   feature norm, activation norms and per-depth gradient/Jacobian sensitivity
   to distant history, stratified by dialogue length and speaker relation.
   Distinguish oversmoothing from oversquashing and ordinary overfitting.
   DAG-safe diagnostic definitions must be declared; do not apply the paper's
   strongly-connected normalized energy blindly. Future diagnostics require a
   separate task and do not authorize reading Test.

## Contract record

- SOURCE_MECHANISM: directed unitary convolution described above.
- OUR_ADAPTATION: analysis only.
- NOT_TRANSFERRED: executable Dune, hybrid transformer and positional encodings.
- CAUSAL_RISK: full-graph reverse propagation; triangular masking loses guarantees.
- INPUT_CONTRACT: source graph/edge features and real or complex node features.
- OUTPUT_CONTRACT: source complex hidden features, often real-component readout;
  no new output contract is installed in MERC.
- PARAMETER_IMPACT: zero in this repository task.
- IMPLEMENTATION_COMPLEXITY: dense exponential O(T^3), or truncated sparse
  products with approximation-error checks; any prefix approach multiplies
  execution cost and needs a separate design.
