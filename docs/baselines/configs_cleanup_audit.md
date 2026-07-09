# Configs cleanup audit

## 1. Purpose

This audit records the cleanup of project YAML configs after the MultiDAG+CL
stabilization stage. The goal was to keep canonical, paper-relevant, and
future-experiment configs while removing duplicate root-level legacy
MultiDAG+CL YAMLs that made `configs/` hard to navigate.

No model code, dataset code, training script logic, evaluation script logic,
data files, checkpoints, outputs, logs, or caches were modified.

## 2. Current config tree before cleanup

The pre-cleanup YAML inventory was:

```text
configs/analysis/multi_run_final_analysis.yaml
configs/analysis/multi_run_training_curves.yaml
configs/analysis/multidag_cl/iemocap/context_compare.yaml
configs/analysis/multidag_cl/iemocap/core_results_compare.yaml
configs/analysis/multidag_cl/iemocap/modality_compare.yaml
configs/analysis/multidag_cl/iemocap/stabilization_20260709_compare.yaml
configs/analysis/multidag_cl_iemocap_context_compare.yaml
configs/analysis/multidag_cl_iemocap_modality_compare.yaml
configs/baselines/multidag_cl/iemocap/debug/context_past_all_causal_tav_smoke.yaml
configs/baselines/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml
configs/baselines/multidag_cl/iemocap/debug/context_w5_tav_smoke.yaml
configs/baselines/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml
configs/baselines/multidag_cl/iemocap/formal/context_w0_tav.yaml
configs/baselines/multidag_cl/iemocap/formal/context_w1_tav.yaml
configs/baselines/multidag_cl/iemocap/formal/context_w3_tav.yaml
configs/baselines/multidag_cl/iemocap/formal/context_w5_tav.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_a.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_av.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_t.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_ta.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_tav.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_tv.yaml
configs/baselines/multidag_cl/iemocap/formal/modality_w5_v.yaml
configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_dropout01.yaml
configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_dropout02.yaml
configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_graph1.yaml
configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_linear_encoder.yaml
configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_lr3e4.yaml
configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_lr5e4.yaml
configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_stable_candidate.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_smoke.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w0.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w1.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w3.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_audio_only.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_audio_visual.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_quick.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_text_audio.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_text_only.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_text_visual.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_visual_only.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_full_smoke.yaml
configs/baselines/multidag_cl/iemocap_multidag_cl_past_all_causal.yaml
configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_A.yaml
configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_AV.yaml
configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_T.yaml
configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_TA.yaml
configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_TAV.yaml
configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_TV.yaml
configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_V.yaml
configs/modality_ablation/train_mmgcn_iemocap_official_TA_smoke.yaml
configs/pipeline/iemocap_multidag_cl_causal_smoke_pipeline.yaml
configs/pipeline/iemocap_multidag_cl_full_smoke_pipeline.yaml
configs/pipeline/mmgcn_pipeline.yaml
configs/pipeline/mmgcn_pipeline_analyze.yaml
configs/pipeline/mmgcn_pipeline_m3ed.yaml
configs/pipeline/multidag_cl/iemocap/debug/context_past_all_causal_tav_smoke.yaml
configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml
configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_smoke.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_w0_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_w1_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_w3_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/context_w5_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_a.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_av.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_t.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_ta.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_tav.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_tv.yaml
configs/pipeline/multidag_cl/iemocap/formal/modality_w5_v.yaml
configs/pipeline/multidag_cl/iemocap/missing/missing_eval_from_context_w5_tav.yaml
configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_dropout01.yaml
configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_dropout02.yaml
configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_graph1.yaml
configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_linear_encoder.yaml
configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_lr3e4.yaml
configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_lr5e4.yaml
configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_stable_candidate.yaml
configs/pipeline/multidag_cl_iemocap_context_past_all.yaml
configs/pipeline/multidag_cl_iemocap_context_w0.yaml
configs/pipeline/multidag_cl_iemocap_context_w1.yaml
configs/pipeline/multidag_cl_iemocap_context_w3.yaml
configs/pipeline/multidag_cl_iemocap_context_w5.yaml
configs/pipeline/multidag_cl_iemocap_context_w5_quick.yaml
configs/pipeline/multidag_cl_iemocap_modality_audio_only.yaml
configs/pipeline/multidag_cl_iemocap_modality_audio_visual.yaml
configs/pipeline/multidag_cl_iemocap_modality_text_audio.yaml
configs/pipeline/multidag_cl_iemocap_modality_text_only.yaml
configs/pipeline/multidag_cl_iemocap_modality_text_visual.yaml
configs/pipeline/multidag_cl_iemocap_modality_visual_only.yaml
configs/pipeline/multidag_cl_iemocap_w5_missing_modalities.yaml
configs/smoke/train_mmgcn_smoke.yaml
configs/smoke/train_multidag_cl_smoke.yaml
configs/train_mmgcn_iemocap_official.yaml
configs/train_mmgcn_m3ed.yaml
configs/train_mmgcn_m3ed_causal.yaml
configs/train_simple_mlp_m3ed.yaml
```

## 3. Config categories found

- MultiDAG+CL IEMOCAP canonical training configs: `debug/`, `formal/`, and `stabilize/`.
- MultiDAG+CL IEMOCAP canonical pipeline configs: `debug/`, `formal/`, `stabilize/`, and `missing/`.
- MultiDAG+CL IEMOCAP analysis configs: context, modality, core-result, stabilization, and stable-context comparisons.
- Duplicate legacy MultiDAG+CL root-level YAMLs under `configs/baselines/multidag_cl/`, `configs/pipeline/`, and `configs/analysis/`.
- Existing MMGCN/SimpleMLP/root smoke configs outside this cleanup scope.
- Existing MMGCN modality-ablation configs outside this cleanup scope.

## 4. YAMLs kept

Kept MultiDAG+CL canonical training YAMLs:

- `configs/baselines/multidag_cl/iemocap/debug/context_past_all_causal_tav_smoke.yaml` - minimal debug smoke, canonical replacement for old full smoke.
- `configs/baselines/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml` - short debug run without formal caps.
- `configs/baselines/multidag_cl/iemocap/debug/context_w5_tav_smoke.yaml` - minimal debug smoke for causal w5.
- `configs/baselines/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml` - formal all-past causal context condition.
- `configs/baselines/multidag_cl/iemocap/formal/context_w0_tav.yaml` - formal current-utterance context condition.
- `configs/baselines/multidag_cl/iemocap/formal/context_w1_tav.yaml` - formal causal window 1 condition.
- `configs/baselines/multidag_cl/iemocap/formal/context_w3_tav.yaml` - formal causal window 3 condition.
- `configs/baselines/multidag_cl/iemocap/formal/context_w5_tav.yaml` - formal causal window 5 condition.
- `configs/baselines/multidag_cl/iemocap/formal/modality_w5_a.yaml` - formal from-scratch audio-only modality condition.
- `configs/baselines/multidag_cl/iemocap/formal/modality_w5_av.yaml` - formal from-scratch audio+visual modality condition.
- `configs/baselines/multidag_cl/iemocap/formal/modality_w5_t.yaml` - formal from-scratch text-only modality condition.
- `configs/baselines/multidag_cl/iemocap/formal/modality_w5_ta.yaml` - formal from-scratch text+audio modality condition.
- `configs/baselines/multidag_cl/iemocap/formal/modality_w5_tav.yaml` - formal full TAV modality condition.
- `configs/baselines/multidag_cl/iemocap/formal/modality_w5_tv.yaml` - formal from-scratch text+visual modality condition.
- `configs/baselines/multidag_cl/iemocap/formal/modality_w5_v.yaml` - formal from-scratch visual-only modality condition.
- `configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_dropout01.yaml` - completed 20260709 stabilization ablation.
- `configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_dropout02.yaml` - completed 20260709 stabilization ablation.
- `configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_graph1.yaml` - completed 20260709 stabilization ablation.
- `configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_linear_encoder.yaml` - completed 20260709 stabilization ablation.
- `configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_lr3e4.yaml` - completed 20260709 stabilization ablation.
- `configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_lr5e4.yaml` - completed 20260709 stabilization ablation.
- `configs/baselines/multidag_cl/iemocap/stabilize/context_w5_tav_stable_candidate.yaml` - best single-seed stable candidate from 20260709.

Kept MultiDAG+CL canonical pipeline YAMLs:

- `configs/pipeline/multidag_cl/iemocap/debug/context_past_all_causal_tav_smoke.yaml` - canonical debug smoke pipeline.
- `configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml` - canonical quick pipeline.
- `configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_smoke.yaml` - canonical debug smoke pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml` - formal context pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/context_w0_tav.yaml` - formal context pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/context_w1_tav.yaml` - formal context pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/context_w3_tav.yaml` - formal context pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/context_w5_tav.yaml` - formal context pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_a.yaml` - formal modality pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_av.yaml` - formal modality pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_t.yaml` - formal modality pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_ta.yaml` - formal modality pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_tav.yaml` - formal modality pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_tv.yaml` - formal modality pipeline.
- `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_v.yaml` - formal modality pipeline.
- `configs/pipeline/multidag_cl/iemocap/missing/missing_eval_from_context_w5_tav.yaml` - test-time missing-modality evaluation for the original formal w5 TAV checkpoint.
- `configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_dropout01.yaml` - completed stabilization pipeline.
- `configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_dropout02.yaml` - completed stabilization pipeline.
- `configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_graph1.yaml` - completed stabilization pipeline.
- `configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_linear_encoder.yaml` - completed stabilization pipeline.
- `configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_lr3e4.yaml` - completed stabilization pipeline.
- `configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_lr5e4.yaml` - completed stabilization pipeline.
- `configs/pipeline/multidag_cl/iemocap/stabilize/context_w5_tav_stable_candidate.yaml` - completed stable-candidate pipeline.

Kept analysis YAMLs:

- `configs/analysis/multi_run_final_analysis.yaml` - generic multi-run final-analysis template outside the MultiDAG cleanup scope.
- `configs/analysis/multi_run_training_curves.yaml` - generic multi-run training-curve template outside the MultiDAG cleanup scope.
- `configs/analysis/multidag_cl/iemocap/context_compare.yaml` - canonical formal context comparison template.
- `configs/analysis/multidag_cl/iemocap/core_results_compare.yaml` - canonical core results comparison template.
- `configs/analysis/multidag_cl/iemocap/modality_compare.yaml` - canonical formal modality comparison template.
- `configs/analysis/multidag_cl/iemocap/stabilization_20260709_compare.yaml` - completed stabilization comparison config.

Kept non-MultiDAG or local smoke YAMLs:

- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_A.yaml` - MMGCN modality-ablation config outside this cleanup scope.
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_AV.yaml` - MMGCN modality-ablation config outside this cleanup scope.
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_T.yaml` - MMGCN modality-ablation config outside this cleanup scope.
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_TA.yaml` - MMGCN modality-ablation config outside this cleanup scope.
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_TAV.yaml` - MMGCN modality-ablation config outside this cleanup scope.
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_TV.yaml` - MMGCN modality-ablation config outside this cleanup scope.
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_V.yaml` - MMGCN modality-ablation config outside this cleanup scope.
- `configs/modality_ablation/train_mmgcn_iemocap_official_TA_smoke.yaml` - MMGCN/IEMOCAP smoke ablation config outside this cleanup scope.
- `configs/pipeline/mmgcn_pipeline.yaml` - existing MMGCN pipeline config left in place to avoid an unrelated MMGCN path migration.
- `configs/pipeline/mmgcn_pipeline_analyze.yaml` - existing MMGCN analysis pipeline config left in place.
- `configs/pipeline/mmgcn_pipeline_m3ed.yaml` - existing MMGCN M3ED pipeline config left in place.
- `configs/smoke/train_mmgcn_smoke.yaml` - local MMGCN fake-data smoke config.
- `configs/smoke/train_multidag_cl_smoke.yaml` - local fake-data MultiDAG+CL smoke config still used by `scripts/baselines/train_multidag_cl_smoke.py`.
- `configs/train_mmgcn_iemocap_official.yaml` - existing MMGCN formal config left in place.
- `configs/train_mmgcn_m3ed.yaml` - existing MMGCN formal config left in place.
- `configs/train_mmgcn_m3ed_causal.yaml` - existing MMGCN causal config left in place.
- `configs/train_simple_mlp_m3ed.yaml` - existing SimpleMLP config left in place.

## 5. YAMLs moved or renamed

No YAML file was directly moved with `git mv`. The prior canonicalization had
already created nested replacements for the legacy root-level MultiDAG+CL
YAMLs. This cleanup deleted the duplicate legacy paths and documented their
canonical replacements.

New YAMLs added in this pass:

- `configs/baselines/multidag_cl/iemocap/stabilize/context_w0_tav_stable_candidate.yaml` - stable-candidate H-R current-utterance context.
- `configs/baselines/multidag_cl/iemocap/stabilize/context_w3_tav_stable_candidate.yaml` - stable-candidate H-R causal window 3 context.
- `configs/baselines/multidag_cl/iemocap/stabilize/context_past_all_causal_tav_stable_candidate.yaml` - stable-candidate H-R all-past causal context.
- `configs/pipeline/multidag_cl/iemocap/stabilize/context_w0_tav_stable_candidate.yaml` - pipeline for stable current-utterance context.
- `configs/pipeline/multidag_cl/iemocap/stabilize/context_w3_tav_stable_candidate.yaml` - pipeline for stable causal window 3 context.
- `configs/pipeline/multidag_cl/iemocap/stabilize/context_past_all_causal_tav_stable_candidate.yaml` - pipeline for stable all-past causal context.
- `configs/pipeline/multidag_cl/iemocap/missing/missing_eval_from_stable_context_w5_tav.yaml` - missing-modality evaluation template for the stable w5 checkpoint.
- `configs/analysis/multidag_cl/iemocap/stable_context_compare.yaml` - stable-candidate context comparison template.

## 6. YAMLs deleted

Deleted legacy training YAMLs:

- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_smoke.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/debug/context_w5_tav_smoke.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_full_smoke.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/debug/context_past_all_causal_tav_smoke.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w0.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/formal/context_w0_tav.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w1.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/formal/context_w1_tav.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w3.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/formal/context_w3_tav.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/formal/context_w5_tav.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_past_all_causal.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_quick.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_text_only.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/formal/modality_w5_t.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_audio_only.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/formal/modality_w5_a.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_visual_only.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/formal/modality_w5_v.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_text_audio.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/formal/modality_w5_ta.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_text_visual.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/formal/modality_w5_tv.yaml`.
- `configs/baselines/multidag_cl/iemocap_multidag_cl_causal_w5_audio_visual.yaml` - replaced by `configs/baselines/multidag_cl/iemocap/formal/modality_w5_av.yaml`.

Deleted legacy pipeline YAMLs:

- `configs/pipeline/iemocap_multidag_cl_causal_smoke_pipeline.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_smoke.yaml`.
- `configs/pipeline/iemocap_multidag_cl_full_smoke_pipeline.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/debug/context_past_all_causal_tav_smoke.yaml`.
- `configs/pipeline/multidag_cl_iemocap_context_w0.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/formal/context_w0_tav.yaml`.
- `configs/pipeline/multidag_cl_iemocap_context_w1.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/formal/context_w1_tav.yaml`.
- `configs/pipeline/multidag_cl_iemocap_context_w3.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/formal/context_w3_tav.yaml`.
- `configs/pipeline/multidag_cl_iemocap_context_w5.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/formal/context_w5_tav.yaml`.
- `configs/pipeline/multidag_cl_iemocap_context_past_all.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/formal/context_past_all_causal_tav.yaml`.
- `configs/pipeline/multidag_cl_iemocap_context_w5_quick.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/debug/context_w5_tav_quick.yaml`.
- `configs/pipeline/multidag_cl_iemocap_modality_text_only.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_t.yaml`.
- `configs/pipeline/multidag_cl_iemocap_modality_audio_only.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_a.yaml`.
- `configs/pipeline/multidag_cl_iemocap_modality_visual_only.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_v.yaml`.
- `configs/pipeline/multidag_cl_iemocap_modality_text_audio.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_ta.yaml`.
- `configs/pipeline/multidag_cl_iemocap_modality_text_visual.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_tv.yaml`.
- `configs/pipeline/multidag_cl_iemocap_modality_audio_visual.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/formal/modality_w5_av.yaml`.
- `configs/pipeline/multidag_cl_iemocap_w5_missing_modalities.yaml` - replaced by `configs/pipeline/multidag_cl/iemocap/missing/missing_eval_from_context_w5_tav.yaml`.

Deleted legacy analysis YAMLs:

- `configs/analysis/multidag_cl_iemocap_context_compare.yaml` - replaced by `configs/analysis/multidag_cl/iemocap/context_compare.yaml`.
- `configs/analysis/multidag_cl_iemocap_modality_compare.yaml` - replaced by `configs/analysis/multidag_cl/iemocap/modality_compare.yaml`.

## 7. Legacy YAMLs removed and why

The removed YAMLs were duplicate root-level compatibility entries. They were
removed because canonical nested YAMLs already existed, active docs/scripts were
updated to the canonical paths, and leaving both forms created ambiguity around
which config should be used for formal, debug, missing-modality, and analysis
runs.

## 8. Canonical config tree after cleanup

```text
configs/
  analysis/
    multidag_cl/
      iemocap/
        context_compare.yaml
        core_results_compare.yaml
        modality_compare.yaml
        stabilization_20260709_compare.yaml
        stable_context_compare.yaml
    multi_run_final_analysis.yaml
    multi_run_training_curves.yaml
  baselines/
    multidag_cl/
      iemocap/
        debug/
          context_past_all_causal_tav_smoke.yaml
          context_w5_tav_quick.yaml
          context_w5_tav_smoke.yaml
        formal/
          context_past_all_causal_tav.yaml
          context_w0_tav.yaml
          context_w1_tav.yaml
          context_w3_tav.yaml
          context_w5_tav.yaml
          modality_w5_a.yaml
          modality_w5_av.yaml
          modality_w5_t.yaml
          modality_w5_ta.yaml
          modality_w5_tav.yaml
          modality_w5_tv.yaml
          modality_w5_v.yaml
        stabilize/
          context_past_all_causal_tav_stable_candidate.yaml
          context_w0_tav_stable_candidate.yaml
          context_w3_tav_stable_candidate.yaml
          context_w5_tav_dropout01.yaml
          context_w5_tav_dropout02.yaml
          context_w5_tav_graph1.yaml
          context_w5_tav_linear_encoder.yaml
          context_w5_tav_lr3e4.yaml
          context_w5_tav_lr5e4.yaml
          context_w5_tav_stable_candidate.yaml
  pipeline/
    multidag_cl/
      iemocap/
        debug/
          context_past_all_causal_tav_smoke.yaml
          context_w5_tav_quick.yaml
          context_w5_tav_smoke.yaml
        formal/
          context_past_all_causal_tav.yaml
          context_w0_tav.yaml
          context_w1_tav.yaml
          context_w3_tav.yaml
          context_w5_tav.yaml
          modality_w5_a.yaml
          modality_w5_av.yaml
          modality_w5_t.yaml
          modality_w5_ta.yaml
          modality_w5_tav.yaml
          modality_w5_tv.yaml
          modality_w5_v.yaml
        missing/
          missing_eval_from_context_w5_tav.yaml
          missing_eval_from_stable_context_w5_tav.yaml
        stabilize/
          context_past_all_causal_tav_stable_candidate.yaml
          context_w0_tav_stable_candidate.yaml
          context_w3_tav_stable_candidate.yaml
          context_w5_tav_dropout01.yaml
          context_w5_tav_dropout02.yaml
          context_w5_tav_graph1.yaml
          context_w5_tav_linear_encoder.yaml
          context_w5_tav_lr3e4.yaml
          context_w5_tav_lr5e4.yaml
          context_w5_tav_stable_candidate.yaml
```

MMGCN, SimpleMLP, generic analysis, and fake-data smoke YAMLs remain in their
existing locations to avoid mixing this MultiDAG+CL cleanup with an unrelated
MMGCN path migration.

## 9. References updated

Updated active config references in:

- `docs/baselines/multidag_cl_formal_run_plan.md`
- `docs/baselines/multidag_cl_formal_yaml_standardization_notes.md`
- `docs/baselines/multidag_cl_baseline_stabilization_notes.md`
- `docs/baselines/multidag_cl_multirun_stabilization_analysis_notes.md`
- `docs/baselines/multidag_cl_yaml_parameter_guide.md`
- `docs/baselines/multidag_cl_observability_and_figures_notes.md`
- `docs/baselines/multidag_cl_iemocap_experiment_matrix_notes.md`
- `docs/baselines/multidag_cl_pipeline_integration_notes.md`
- `docs/baselines/multidag_cl_paper_artifacts_notes.md`
- `scripts/analyze/diagnose_iemocap_splits.py`

## 10. Checks performed

Syntax checks passed:

```bash
python -m py_compile scripts/run_experiment_pipeline.py scripts/baselines/train_multidag_cl.py scripts/baselines/evaluate_multidag_cl_checkpoint.py scripts/analyze/plot_multidag_cl_stabilization_compare.py
python -m py_compile scripts/dev/validate_config_tree.py
```

The first sandboxed Windows attempts hit the known `.pyc` atomic rename access
issue. Both syntax checks passed after rerunning with permission to write the
Python cache files.

Config validation passed:

```bash
python scripts/dev/validate_config_tree.py
```

The validator checks:

- all formal/stabilize pipeline YAMLs point to existing training YAMLs;
- all formal/stabilize pipelines enable evaluation, analysis tables, single-run analysis, training curves, and final analysis;
- all formal/stabilize training YAMLs have no train/validation batch caps;
- no canonical formal YAML uses `full` naming or `context_mode: full`;
- stable-candidate configs use the required encoder, graph depth, dropout, LR, weight decay, grad clipping, and context settings;
- duplicate legacy root-level MultiDAG+CL YAMLs are gone.

## 11. Remote recovery note

Because the remote project's `configs/` directory was deleted, restore it after
reviewing, committing, and pushing this cleanup from local:

```bash
git fetch origin
git switch <branch>
git pull --ff-only
git status --short
```

Do not manually recreate remote YAML files. Let Git restore the reviewed local
tree.

## 12. MMGCN config restoration

After the cleanup, the MMGCN training and pipeline YAMLs were rechecked because
they were reported as missing from the cleaned config tree. Git history and the
current checkout were checked for the original MMGCN paths before making any
config changes.

Required MMGCN training YAMLs confirmed at their original paths:

- `configs/train_mmgcn_iemocap_official.yaml`
- `configs/train_mmgcn_m3ed.yaml`
- `configs/train_mmgcn_m3ed_causal.yaml`

Required MMGCN pipeline YAMLs confirmed at their original paths:

- `configs/pipeline/mmgcn_pipeline.yaml`
- `configs/pipeline/mmgcn_pipeline_analyze.yaml`
- `configs/pipeline/mmgcn_pipeline_m3ed.yaml`

Required MMGCN smoke and modality-ablation YAMLs confirmed at their original
paths:

- `configs/smoke/train_mmgcn_smoke.yaml`
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_A.yaml`
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_AV.yaml`
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_T.yaml`
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_TA.yaml`
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_TAV.yaml`
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_TV.yaml`
- `configs/modality_ablation/m3ed/mmgcn/train_mmgcn_m3ed_V.yaml`
- `configs/modality_ablation/train_mmgcn_iemocap_official_TA_smoke.yaml`

No MMGCN YAML path migration was performed. The original root-level training
paths and `configs/pipeline/mmgcn_pipeline*.yaml` paths were preserved so
existing scripts and docs continue to resolve them.

The MultiDAG+CL canonical config tree was not modified in this follow-up. No
model code, dataset code, training logic, evaluation logic, or pipeline logic was
modified.
