# MultiDAG-CL Paper Reimplementation Project Fair matrix

## Scope and evidence boundary

The formal model name is **MultiDAG-CL Paper Reimplementation**. This matrix uses the
**Project Fair Comparison Track** with the project `clean_roberta_v1` feature registry.
It is a project-protocol comparison and is not a reproduction of the paper-reported
numbers or an `author_official` implementation claim.

The four configurations hold Ses05 fixed as Test, hold seed 100 fixed, and rotate the
complete Validation session through Ses01--Ses04. Validation Weighted-F1 selects the
best checkpoint. Test does not participate in checkpoint, epoch, hyperparameter, or
model selection, and it is evaluated once after the selected checkpoint is locked.

## Configuration matrix

| Validation session | Configuration | Run name |
| --- | --- | --- |
| Ses01 | `project_fair.yaml` | `multidag_cl_paper_reimplementation_project_fair_val_ses01_seed100` |
| Ses02 | `project_fair_val_ses02_seed100.yaml` | `multidag_cl_paper_reimplementation_project_fair_val_ses02_seed100` |
| Ses03 | `project_fair_val_ses03_seed100.yaml` | `multidag_cl_paper_reimplementation_project_fair_val_ses03_seed100` |
| Ses04 | `project_fair_val_ses04_seed100.yaml` | `multidag_cl_paper_reimplementation_project_fair_val_ses04_seed100` |

All paths are relative to:

```text
configs/multidag_cl/paper_reimplementation/iemocap/full_context/clean_roberta_features/
```

Across the four YAML files, the only permitted differences are:

```text
run_name
dataset.validation_session
```

Every other value remains identical, including the registry key, feature identity and
SHA256, Ses05 Test split, seed 100, 30 epochs, optimizer, curriculum, model structure,
global-norm gradient clipping, checkpoint selection rules, and experiment group.

## Ses01 run identity

The reviewed canonical Ses01 run is:

```text
multidag_cl_paper_reimplementation_project_fair_val_ses01_seed100_20260805_202548_dd2b86
```

The accidental duplicate is:

```text
multidag_cl_paper_reimplementation_project_fair_val_ses01_seed100_20260805_202504_c4c666
```

Their relationship is:

```text
DETERMINISTIC_NUMERICAL_REPEAT_WITH_RUN_METADATA_DIFFERENCES
```

The duplicate is not a second seed and must not be counted as a second statistical
sample.

## Local validation

Each configuration can be checked locally without allocating a run, reading Test, or
executing an optimizer step:

```powershell
python scripts/models/multidag_cl/paper_reimplementation/train.py --mode check --config <CONFIG> --device cpu
```

The matrix regression test performs a deep comparison after removing the two allowed
difference fields. This prevents silent drift in learning rate, dropout, curriculum,
graph settings, encoder settings, selection protocol, or other nested configuration
values.
