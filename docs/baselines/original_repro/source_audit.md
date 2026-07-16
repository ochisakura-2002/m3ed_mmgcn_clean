# Source audit

Audit date: 2026-07-15. Repositories were inspected at pinned commits; the same
provenance is embedded by the model registry.

| Baseline | Paper | Repository / commit | Effective selection finding | Project status |
|---|---|---|---|---|
| MMGCN | ACL-IJCNLP 2021, `2021.acl-long.440` | `hujingwen6666/MMGCN` / `85732d984f70c1c84dd47c81aa97f1271397b899` | released loop observes Test | `official_code_adapted` |
| MultiDAG+CL | LREC-COLING 2024, `2024.lrec-main.380` | `vanntc711/MultiDAG-CL` / `59a75877065a91bf9388fbb564607fe79717fd4f` | released training path observes Test and has a loader mismatch | `official_code_adapted_with_protocol_repairs` |
| GS-MCC | AAAI 2025, DOI `10.1609/aaai.v39i23.33242` | `FuchenZhang/GS-MCC` / `38c4038a7738f9bf7b3132c3e99a126e1cf1f28d` | effective trainer observes Test; paper and effective objective differ materially | `PROJECT_VARIANT_NOT_PAPER_REPRODUCTION` |
| DialogueGCN | EMNLP-IJCNLP 2019, `D19-1015` | `declare-lab/conv-emotion` / `6128ca20e9c736605cce7e99d5d95db0356c35f5` | official loop observes Test each epoch | `paper_equation_aligned_official_code_adapted` |

## MMGCN

The effective IEMOCAP path uses a text projection and bidirectional LSTM, linear
audio/visual encoders, three modality node sets, full same-modality edges, and
aligned cross-modal edges. Released paths disagree on graph depth; the project
keeps depth configurable and pins four for the original-reproduction configs.
The official residual propagation path is enabled in legacy/clean smoke,
screening, and five-fold base configs. Existing causal MMGCN behavior is not
changed.

## MultiDAG+CL

Released `get_adj_v1` scans predecessors until the configured count of
same-speaker predecessors is reached. Difficulty is
`(same-speaker emotion shifts + speaker count) / (utterance count + speaker count)`;
neither the printed equation nor audited implementation has a class-frequency
term. IEMOCAP uses five curriculum buckets. The project removes Test from the
training/ordering path and tests the scheduler with realistic dialogue labels
and speakers.

## GS-MCC

The paper describes a multimodal interaction graph, explicit low/high-frequency
operators, cross-frequency contrastive construction, and a total loss. The
released effective Fourier and training paths do not provide a trustworthy
one-to-one executable reference for all of those mechanisms. A complete
migration was not reliable within this repair, so option B is used. The project
implementation is named `project_paper_oriented_gsmcc`, is excluded from paper
gap/reproduction ranking, and may only be reported as an engineering variant.

## DialogueGCN

The project retains bidirectional context encoding, past/future edges,
`2*M^2` speaker/direction relations, nodal attention, and class weighting. The
first graph layer now applies relation-specific transforms, per-relation
neighbor normalization, edge-attention weights, and an independent root
transform. The second graph layer has explicit neighbor propagation and an
independent self transform. A hand-computed three-node numerical fixture checks
both layers. Paper-adjacent configs use LSTM, dropout 0.4, learning rate 0.0003,
batch size 32, zero L2, class weights, nodal attention, and windows of 10.

## Selection-risk repair

`legacy_official_split_safe_selection` preserves original trainVid/testVid and
creates Validation only within trainVid. `legacy_fivefold_fair_comparison` and
`clean_roberta_fivefold_fair_comparison` use outer-session Test folds plus inner
dialogue Validation. Test is excluded from checkpoint, epoch, hyperparameter,
curriculum, and top-two selection. Track and comparability metadata are embedded
in configs, checkpoints, metric rows, and reports so the three rankings cannot
be mixed.
