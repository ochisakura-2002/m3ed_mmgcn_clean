# GS-MCC paper-code-project mapping

Decision: **option B**. A reliable migration of the complete official/paper
mechanism is outside this focused repair. The implementation is therefore
registered as `project_paper_oriented_gsmcc` with status
`PROJECT_VARIANT_NOT_PAPER_REPRODUCTION`.

| Mechanism | Paper / released code finding | Project implementation | Fidelity decision |
|---|---|---|---|
| Multimodal interaction graph | The paper and released effective graph path are not a clean one-to-one specification. | `build_sliding_multimodal_graph` is a project interpretation. | Engineering comparison only. |
| Fourier Graph Operator | Released `FourierGNNmodel.py` applies FFT to its effective feature tensor; it does not expose the paper's complete explicit low/high construction. | `FourierGraphOperator` is a simplified auditable operator. | Not an official migration. |
| Low/high branches | Paper description and released effective path materially differ. | Paired low/high project branches are retained for experiments. | Not paper-faithful evidence. |
| Contrastive positives/negatives | Released trainer objectives do not provide a trustworthy executable reference for the printed construction. | Project cross-frequency contrastive loss is retained. | No paper-gap eligibility. |
| Total loss | Paper and released training path differ materially. | Project classification plus optional contrastive loss. | Cannot support a successful reproduction claim. |

The historical aliases `gsmcc` and `gs-mcc` resolve to the project key only for
CLI compatibility. Configs, registry metadata, checkpoints, analysis rows, and
reports must preserve the project key and fidelity status. Paper reference
values may be shown as bibliography context, but the model is excluded from
paper reproduction ranking and gap calculations.
