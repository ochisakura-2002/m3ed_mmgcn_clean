# m3ed_mmgcn_clean

This project is the multimodal emotion recognition baseline project for M3ED + MMGCN.

## Current stage

Project skeleton stage.

The current goal is not to add new modules, but to first reproduce a stable MMGCN baseline on M3ED.

## Research route

1. Build project skeleton.
2. Inspect M3ED data format.
3. Inspect MMGCN baseline code.
4. Prepare M3ED metadata and processed features.
5. Reproduce MMGCN baseline.
6. Build train / evaluate / logging pipeline.
7. Add experiment run management.
8. Add confidence-aware modality gating.
9. Add modality dropout training.
10. Evaluate robustness under missing or unreliable modalities.

## Project structure

- `configs/`: YAML configuration files.
- `data/raw/`: raw M3ED data.
- `data/processed/`: processed features or cached data.
- `data/metadata/`: metadata files such as splits and labels.
- `datasets/`: dataset and collate code.
- `models/`: MMGCN model and modality encoders.
- `scripts/`: executable scripts.
- `utils/`: shared utility functions.
- `outputs/`: experiment outputs.
- `third_party/`: external baseline code and pretrained models.
- `docs/`: project notes and reading summaries.

## Development principle

Do not add innovation modules before the MMGCN baseline is reproducible.

Each step should be tested locally before being integrated into the full training pipeline.
