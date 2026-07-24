# Original MERC comparison suite

This suite keeps the four offline/full-context candidates isolated from the
project's causal models. MMGCN, MultiDAG+CL, and DialogueGCN are maintained as
original-reproduction adapters. The GS-MCC candidate is deliberately named
`project_paper_oriented_gsmcc` and carries
`PROJECT_VARIANT_NOT_PAPER_REPRODUCTION`; it is an engineering comparison only.

## Experiment tracks

| Track | Features | Test split | Validation | Permitted interpretation |
|---|---|---|---|---|
| `legacy_official_split_safe_selection` | released MMGCN PKL | original `testVid` (Ses05) | dialogue split made only inside original `trainVid` | paper-adjacent diagnostic; `protocol_comparability: paper_adjacent_not_exact` |
| `legacy_fivefold_fair_comparison` | released MMGCN PKL | Ses01--Ses05 outer folds | dialogue split inside the other four sessions | fair comparison under legacy features; never a strict paper reproduction gap |
| `clean_roberta_fivefold_fair_comparison` | Clean RoBERTa v1 | Ses01--Ses05 outer folds | dialogue split inside the other four sessions | final baseline and module comparison track |

Analysis groups and ranks by `experiment_track`. Paper-gap columns are emitted
only for eligible models on `legacy_official_split_safe_selection`; the two
five-fold tracks cannot enter that ranking.

## Selection contract

The four clean screening jobs use Clean RoBERTa v1, seed 42, outer Test Ses05,
dialogue-level Validation from Ses01--Ses04, and no batch cap. Top-two selection
requires all four clean screening results and uses, in order:

1. clean screening Validation Weighted-F1;
2. training stability;
3. reproduction credibility;
4. a clear module insertion point;
5. legacy paper-adjacent Validation as auxiliary evidence only.

Test metrics are prohibited from checkpoint, epoch, hyperparameter, and top-two
selection.

## Entrypoints

Validate configuration wiring:

```powershell
conda run -n m3ed_mmgcn python scripts/dev/validate_config_tree.py
```

Inspect a stage without launching training:

```powershell
conda run -n m3ed_mmgcn python scripts/workflows/paper_aligned/run_pipeline.py --stage smoke
```

Train one verified smoke after placing the pinned feature PKL:

```powershell
conda run -n m3ed_mmgcn python scripts/workflows/paper_aligned/train.py `
  --config configs/mmgcn/paper_aligned/iemocap/full_context/legacy_mmgcn_features/smoke.yaml
```

Evaluate a saved checkpoint:

```powershell
conda run -n m3ed_mmgcn python scripts/workflows/paper_aligned/evaluate.py `
  --checkpoint <run>/checkpoints/best_model.pt --split test
```

Aggregate completed outputs through the canonical analysis directory:

```powershell
conda run -n m3ed_mmgcn python scripts/analysis/paper_aligned/analyze_original_merc_results.py
```

Only the canonical analysis entrypoint above is supported. Generated reports
are runtime artifacts and must not be committed without traceable real training
inputs.

## Current fidelity boundary

- MMGCN formal original-reproduction configs enable the official residual path.
- MultiDAG+CL retains the audited architecture and has real-dialogue curriculum
  contract tests; real PKL training remains unconfirmed.
- DialogueGCN's two graph layers are paper-formula aligned and covered by a
  hand-computed numerical fixture. Dense PyTorch adaptation and real IEMOCAP
  performance remain to be validated, so no official-code numerical identity
  is claimed.
- GS-MCC follows option B. It cannot produce a paper reproduction gap or be
  cited as a successful GS-MCC reproduction.
