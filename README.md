# MERC Baseline Research

This repository studies multimodal emotion recognition in conversation (MERC).
The current paper track reproduces and compares Original/noncausal MERC baselines
before introducing new research modules.

For Codex and project architecture context, see:

- `AGENTS.md`
- `docs/PROJECT_CONTEXT.md`

## Current stage

The active stage is formal legacy/Clean screening of four candidates:

- MMGCN
- MultiDAG+CL, where CL means Curriculum Learning
- DialogueGCN
- `project_paper_oriented_gsmcc`, a project variant rather than a faithful GS-MCC reproduction

Legacy results provide paper-adjacent reproduction diagnostics. Clean Validation
provides the fair evidence used to select the Top-2 models; Test is not used for
model selection.

Causal MERC and M3ED routes remain in the repository for long-term online and
cross-dataset research, but they are not the current paper's main screening track.

## Research route

1. Complete the four-model legacy/Clean formal screening.
2. Select Top-2 using Clean Validation evidence.
3. Run Top-2 five-fold experiments.
4. Run the final baseline with multiple random seeds.
5. Add the selected innovation module to the stable baseline.
6. Run ablations and controlled comparison experiments.
7. Preserve causal MERC and M3ED as long-term research routes.

## Project structure

- `configs/`: YAML configuration files.
- `data/raw/`: raw M3ED data.
- `data/processed/`: processed features or cached data.
- `data/metadata/`: metadata files such as splits and labels.
- `datasets/`: dataset and collate code.
- `models/`: Original/noncausal, causal, and engineering baseline implementations.
- `scripts/`: executable scripts.
- `utils/`: shared utility functions.
- `outputs/`: experiment outputs.
- `third_party/`: external baseline code and pretrained models.
- `docs/`: project notes and reading summaries.

## Development principle

Do not add innovation modules before the Original MERC baseline screening is stable.

Use Validation for checkpoint and model selection; reserve Test for final reporting.
Each change should pass local gates before entering the remote training workflow.
