# MultiDAG+CL YAML parameter guide

This guide explains the MultiDAG+CL experiment YAMLs as experiment controls,
not as a copy of the config files. It covers four YAML families:

1. Baseline training YAML.
2. Pipeline YAML.
3. Missing-modality pipeline YAML.
4. Multi-run analysis YAML.

## 1. Baseline Training YAML

Baseline training YAMLs are consumed by:

```bash
python scripts/baselines/train_multidag_cl.py --config <training_yaml>
```

They define one training run and one checkpoint family under
`outputs/runs/<run_id>/`.

### System

| Field | Meaning | If increased or broadened | If decreased or narrowed | Common values | Current project reason |
|---|---|---|---|---|---|
| `system.seed` | Random seed for split construction, model initialization, and dataloader behavior where applicable. | A larger number is just a different random seed; it does not mean a stronger setting. | A smaller number is also just a different seed. | `42`, or a planned seed list for multi-seed runs. | Current configs use `42` for a stable first formal pass. |
| `system.device` | Device requested by the script. | Moving from `cpu` to `cuda` makes training much faster if CUDA is available. | Moving from `cuda` to `cpu` is safer locally but much slower. | `cuda` on remote V100, `cpu` for local syntax/debug checks. | Formal training is intended for the remote GPU machine. |

### Dataset

| Field | Meaning | If increased or broadened | If decreased or narrowed | Common values | Current project reason |
|---|---|---|---|---|---|
| `dataset.name` | Dataset adapter name. | Switching to another dataset would require a supported adapter and matching feature format. | Keeping only `IEMOCAP` keeps this script on the official MMGCN-style feature path. | `IEMOCAP`. | MultiDAG+CL integration is currently built around IEMOCAP official features. |
| `dataset.feature_pkl_path` | Relative path to the official feature pkl. | A different path can point to another feature export. | A missing or wrong path stops training before it starts. | `third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl`. | Keeps configs portable without hard-coding local absolute paths. |
| `dataset.num_classes` | Number of emotion classes. | More classes changes classifier output size and metrics labels. | Fewer classes changes the task definition. | `6` for IEMOCAP official setup. | Matches the six-label IEMOCAP feature files. |
| `dataset.valid_ratio` | Validation fraction when a validation split is derived. | Larger validation split gives more validation examples but fewer training examples. | Smaller validation split gives more training examples but noisier validation. | `0.1`. | Mirrors existing project convention for official-prefix validation. |
| `dataset.val_split_strategy` | Rule used by the IEMOCAP dataloader to choose validation dialogues. | Broader/custom strategies can change which dialogues are held out. | Stricter strategies improve reproducibility if fixed. | `official_prefix`. | Keeps validation deterministic and aligned with the current adapter. |
| `dataset.label_list` | Human-readable class names by label id. | Adding names only makes sense if `num_classes` also grows. | Removing names breaks the required one-name-per-class mapping. | `[Happy, Sad, Neutral, Angry, Excited, Frustrated]`. | These names match the six IEMOCAP labels used in output CSVs and figures. |

### Model

| Field | Meaning | If increased or broadened | If decreased or narrowed | Common values | Current project reason |
|---|---|---|---|---|---|
| `model.name` | Model registry name used by training and pipeline scripts. | Switching names changes the training entry selected by the pipeline. | Keeping `MultiDAGCL` keeps this workflow isolated from MMGCN and SDT. | `MultiDAGCL`. | This task is scoped to the project MultiDAG+CL baseline. |
| `model.text_feature_dim` | Text feature dimension expected in the pkl. | Larger values require matching feature tensors. | Smaller values require matching feature tensors. | `100`. | Matches the official IEMOCAP text feature shape. |
| `model.audio_feature_dim` | Audio feature dimension expected in the pkl. | Larger values require matching feature tensors. | Smaller values require matching feature tensors. | `1582`. | Matches the official IEMOCAP audio feature shape. |
| `model.visual_feature_dim` | Visual feature dimension expected in the pkl. | Larger values require matching feature tensors. | Smaller values require matching feature tensors. | `342`. | Matches the official IEMOCAP visual feature shape. |
| `model.hidden_dim` | Shared hidden size inside the baseline. | Larger can increase capacity and memory/time cost. | Smaller can reduce overfitting risk and speed up training but may underfit. | `128`. | A moderate baseline value for first formal runs. |
| `model.num_classes` | Classifier output size. | More outputs require more labels and matching dataset config. | Fewer outputs change the task. | `6`. | Must match `dataset.num_classes`. |
| `model.num_speakers` | Number of speaker ids expected by the dialogue features. | More speakers only makes sense for datasets with more speaker ids. | Fewer speakers can collapse speaker identities incorrectly. | `2`. | IEMOCAP dialogue features are treated as two-speaker conversations. |
| `model.dropout` | Dropout probability. | Higher dropout regularizes more but can underfit. | Lower dropout preserves more signal but can overfit. | `0.1` to `0.5`; current formal configs use `0.3`. | `0.3` is a conservative regularization choice for 30-epoch runs. |
| `model.active_modalities` | Modalities used while training this checkpoint. | Broader lists, such as `["text","audio","visual"]`, train a full TAV model. | Narrower lists, such as `["text"]`, train from-scratch modality ablations. | TAV, T, A, V, TA, TV, AV. | Context-window configs use TAV; modality-ablation configs train separate checkpoints. |
| `model.num_graph_layers` | Number of graph propagation layers. | More layers can mix longer graph information but cost more memory and may oversmooth. | Fewer layers are faster and simpler but may underuse graph context. | `1` to `3`; current configs use `2`. | Keeps the baseline small while still using graph propagation. |
| `model.modality_encoder_type` | Per-modality encoder before graph reasoning. | `causal_gru` is a stronger temporal encoder than a simple projection. | `linear` is lighter and closer to a feature projection baseline. | `causal_gru`, `linear`. | Current formal configs use `causal_gru` to keep temporal encoding causal. |
| `model.modality_encoder_layers` | Number of recurrent/projection layers in the modality encoder. | More layers can increase capacity and cost. | Fewer layers are cheaper and easier to train. | `1`. | Current configs use one layer to keep the baseline controlled. |

### Graph

| Field | Meaning | If increased or broadened | If decreased or narrowed | Common values | Current project reason |
|---|---|---|---|---|---|
| `graph.context_mode` | Semantic mode for dialogue context. | `past_all_causal` broadens context to all previous utterances. | `causal` with a finite window restricts context to recent history. | `causal`, `past_all_causal`. | Formal context comparison uses causal windows plus a past-all causal condition. |
| `graph.window_past` | Number of previous valid utterances visible to each utterance in causal mode. | Larger windows use more history and cost more graph work. | Smaller windows focus on local context and are cheaper. | `0`, `1`, `3`, `5`, `null`. | The experiment matrix compares `0/1/3/5/null` to study causal history length. |

Special cases:

- `window_past: 0` means current utterance only.
- `window_past: 1` means current utterance plus the previous valid utterance.
- `window_past: 3` and `5` add wider causal history.
- `window_past: null` with `context_mode: past_all_causal` maps to an internal
  large effective window, so each utterance sees itself and all previous valid
  utterances.
- Removed legacy smoke configs used `context_mode: full` as a past-all causal
  alias. Canonical configs use `past_all_causal` directly and do not claim true
  offline bidirectional full context.

### Training

| Field | Meaning | If increased or broadened | If decreased or narrowed | Common values | Current project reason |
|---|---|---|---|---|---|
| `training.epochs` | Number of training epochs. | More epochs allow more optimization but take longer and may overfit. | Fewer epochs are faster but may stop before convergence. | `30` formal, `5` quick, `1` smoke. | Formal configs use `30`; `causal_w5_quick` uses `5` as a short remote sanity pass. |
| `training.batch_size` | Dialogues per training batch. | Larger batches can improve throughput but use more memory. | Smaller batches reduce memory pressure but may train noisier. | `8` formal, `2` smoke. | `8` is a practical GPU baseline; smoke uses `2` to stay tiny. |
| `training.lr` | AdamW learning rate. | Higher can train faster but may become unstable. | Lower is more stable but can underfit within fixed epochs. | `0.001`. | Keeps the first formal matrix simple and consistent. |
| `training.weight_decay` | AdamW L2-style regularization. | Higher regularizes more but may suppress useful weights. | Lower regularizes less and can overfit. | `0.0001` formal, `0.0` smoke. | Formal configs use mild regularization; smoke avoids extra confounds. |
| `training.grad_clip` | Maximum gradient norm when positive. | Higher clips less often. | Lower clips more aggressively and may slow learning. | `1.0`. | Keeps training stable without changing model structure. |
| `training.select_best_by` | Metric used to select `best_model.pt`. | Broader choices would allow other monitor metrics. | Current strict value prevents accidental test-based selection. | `val_weighted_f1`. | This integration only allows validation Weighted-F1 checkpoint selection. |
| `training.max_train_batches` | Optional cap on train batches per epoch. | Larger caps approach full training. | Smaller caps make a smoke run faster and less representative. | `null` formal, small integer for smoke. | Formal training should use `null`; caps are only for smoke/debug. |
| `training.max_val_batches` | Optional cap on validation batches per epoch. | Larger caps approach full validation. | Smaller caps make validation faster but noisier. | `null` formal, small integer for smoke. | Formal validation should use `null`; caps are only for smoke/debug. |

### Output

| Field | Meaning | If increased or broadened | If decreased or narrowed | Common values | Current project reason |
|---|---|---|---|---|---|
| `output.run_root` | Root directory for run folders. | A broader/shared output root collects many experiments. | A narrower temporary root is useful for smoke-only artifacts. | `outputs/runs`. | Formal run layout matches existing analysis scripts. |
| `output.experiment_name` | Suffix used in the generated run id. | More specific names make later analysis easier. | Shorter names are easier to type but less descriptive. | `iemocap_multidag_cl_causal_w5`. | Names encode dataset, baseline, context, and modality condition. |

## 2. Pipeline YAML

Pipeline YAMLs are consumed by:

```bash
python scripts/run_experiment_pipeline.py --config <pipeline_yaml>
```

The pipeline is the piece that wires training, evaluation, tables, and figures
together. Running only `train_multidag_cl.py` writes checkpoints and training
logs, but it does not call the plotting scripts.

| Field | Meaning | Practical effect | Common values | Current project reason |
|---|---|---|---|---|
| `project.pipeline_name` | Human-readable pipeline name printed in logs. | Helps identify which workflow ran. | Context, modality, or missing-modality names. | Names mirror YAML filenames. |
| `dataset.name` | Dataset expected by the pipeline. | Checked against the training YAML to prevent mismatches. | `IEMOCAP`. | Keeps formal MultiDAG+CL runs on one dataset family. |
| `model.name` | Model registry key for train/evaluate script selection. | `MultiDAGCL` routes to `scripts/baselines/train_multidag_cl.py` and `evaluate_multidag_cl_checkpoint.py`. | `MultiDAGCL`. | Avoids touching MMGCN or SDT entries. |
| `execution.dry_run` | YAML-controlled dry-run switch. | If true, commands are printed but not executed. | `false` for real runs, `true` for command audit. | This is a YAML field, not a `--dry-run` CLI argument. |
| `train.enabled` | Whether the pipeline trains a new run. | If true, pipeline launches the train script. | `true` for context/modality pipelines, `false` for missing-modality reuse. | Missing-modality evaluation reuses an existing TAV checkpoint. |
| `train.train_config_path` | Training YAML to launch. | Determines model settings, output name, and training budget. | A path under `configs/baselines/multidag_cl/`. | Keeps training details out of the pipeline scheduler. |
| `evaluation.enabled` | Whether to evaluate a checkpoint after training or from an existing run. | If true, writes `logs/evaluations/<split>_<checkpoint>/`. | `true` for formal train/eval pipelines. | Final analysis needs `test_best_model` outputs. |
| `evaluation.checkpoint_name` | Checkpoint file under `checkpoints/`. | Usually selects `best_model.pt`. | `best_model.pt`. | Evaluation reports the validation-selected checkpoint. |
| `evaluation.splits` | Splits to evaluate. | More splits produce more evaluation folders. | `[val, test]` formal. | Val is useful for audit; test is for final reporting. |
| `evaluation.max_batches` | Optional evaluation cap. | Larger or `null` evaluates more data; smaller is smoke-only. | Omitted or `null` formal, small integer smoke. | Formal test evaluation should not be capped. |
| `analysis_tables.enabled` | Whether to rebuild master CSV tables. | If true, writes under `outputs/analysis_tables/`. | `true` for formal context pipelines. | Lets completed runs appear in summary tables. |
| `single_run_analysis.enabled` | Master switch for one-run figure generation. | If false, no single-run training/final figures are produced. | `true` for formal context pipelines. | This is why pipeline runs can produce figures. |
| `single_run_analysis.training_curves` | Calls `plot_single_run_training_curves.py`. | Writes `figures/training_curves/`. | `true`. | Uses `logs/epoch_metrics.csv`. |
| `single_run_analysis.final_analysis` | Calls `plot_single_run_final_analysis.py`. | Writes `figures/final_analysis/`. | `true`. | Uses `logs/evaluations/test_best_model/`. |
| `missing_modalities.enabled` | Enables test-time missing-modality stage. | Calls the missing-modality evaluator when true. | `false` in context pipelines, `true` in missing-modality pipeline. | Keeps robustness evaluation separate from training. |
| `multi_run_training_curves.enabled` | Enables cross-run training-curve plots. | Requires a multi-run analysis YAML with completed run ids. | `false` until formal runs exist. | Avoids hard-coding run ids before they are created. |
| `multi_run_training_curves.config_path` | Config for cross-run training plots. | Used only when the stage is enabled. | `configs/benchmarks/ablations/analysis/multidag_cl/context_compare.yaml`. | Template is ready but `runs` is empty. |
| `multi_run_final_analysis.enabled` | Enables cross-run final-metric plots. | Requires completed run ids and test evaluations. | `false` until formal runs exist. | Keeps context and modality comparisons separate. |
| `multi_run_final_analysis.config_path` | Config for cross-run final plots. | Used only when the stage is enabled. | Context or modality comparison YAML. | Lets report figures be built after formal runs finish. |

Why the pipeline generates figures:

- Training writes `logs/epoch_metrics.csv`.
- Evaluation writes `logs/evaluations/test_best_model/*`.
- `single_run_analysis.training_curves: true` calls the training-curve plotter.
- `single_run_analysis.final_analysis: true` calls the final-analysis plotter.

Why only running the train script does not necessarily generate figures:

- `train_multidag_cl.py` is intentionally a training entry.
- It writes checkpoints and logs, not report figures.
- Plotting belongs to the pipeline analysis stage.

## 3. Missing-Modality Pipeline YAML

Missing-modality YAMLs reuse a trained full TAV checkpoint. They do not retrain
the model.

| Field | Meaning | Practical effect | Common values | Current project reason |
|---|---|---|---|---|
| `run_control.skip_train_use_run_id` | Existing run id to evaluate when `train.enabled: false`. | Must point to a completed run under `outputs/runs/`. | A trained causal_w5 TAV run id. | Missing-modality evaluation must use the already-trained full model. |
| `missing_modalities.checkpoint_name` | Checkpoint to evaluate. | Usually `best_model.pt`. | `best_model.pt`. | Keeps evaluation tied to validation-selected checkpoint. |
| `missing_modalities.split` | Dataset split for robustness evaluation. | `test` produces final reporting numbers. | `test`. | Robustness is evaluated after checkpoint selection. |
| `missing_modalities.settings` | Test-time active modality settings. | More settings produce more raw eval folders and rows in summary. | `TAV`, `TA`, `TV`, `AV`, `T`, `A`, `V`. | Covers full, two-modality, and one-modality availability. |
| `missing_modalities.output_subdir` | Folder under `logs/` for missing-modality outputs. | Changes where summary/raw outputs are stored. | `missing_modalities`. | Keeps robustness outputs separate from normal evaluation. |
| `missing_modalities.make_figures` | Whether to plot the missing-modality summary. | If true, calls `plot_missing_modality_summary.py`. | `true`. | The formal missing-modality pipeline should produce figures automatically. |
| `missing_modalities.skip_if_not_full_train_modalities` | Guard against evaluating a non-TAV checkpoint as missing-modality robustness. | If true, skips checkpoints not trained with text+audio+visual. | `true`. | Prevents mixing from-scratch modality ablation with test-time missingness. |

Setting meanings:

- `TAV`: text, audio, and visual active.
- `TA`: text and audio active, so visual is missing.
- `TV`: text and visual active, so audio is missing.
- `AV`: audio and visual active, so text is missing.
- `T`: text only.
- `A`: audio only.
- `V`: visual only.
- `missing_text`: alias for `AV`.
- `missing_audio`: alias for `TV`.
- `missing_visual`: alias for `TA`.

Important interpretation rule:

Missing-modality evaluation is test-time zeroing. The evaluator zeros unavailable
feature tensors before forward. It does not retrain the checkpoint, and it does
not change the model structure saved in the checkpoint.

## 4. Multi-Run Analysis YAML

Multi-run analysis YAMLs are consumed by the cross-run plotting scripts after
formal runs finish. They are templates until real run ids exist.

| Field | Meaning | If expanded | If reduced | Common values | Current project reason |
|---|---|---|---|---|---|
| `runs` | List of completed runs to compare. Each entry normally has `run_id` and `display_name`. | More runs create broader comparisons but can make plots crowded. | Fewer runs make plots simpler but answer narrower questions. | Empty while waiting for formal run ids, then context or modality groups. | The repo keeps `runs: []` so it does not reference nonexistent runs. |

Usage guidance:

- Fill `runs` only after formal runs finish.
- Do not pre-hard-code run ids that do not exist.
- Keep context-window comparison and modality-ablation comparison in separate
  YAML files.
- Context comparison should compare `w0/w1/w3/w5/past_all_causal`.
- Modality comparison should compare from-scratch T, A, V, TA, TV, AV, and TAV
  runs only when those checkpoints were trained separately.
