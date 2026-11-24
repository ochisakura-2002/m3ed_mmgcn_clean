# H1 Bounded Selective History Attenuation Fast-Kill

2026-11-24 · `WORKSTREAM = MULTIDAG` · `feat/multidag-cl-paper-reimplementation`

`H_SUBGROUP_PROVENANCE = PASS`。本地已核验正式恢复 ZIP，两份副本的 SHA-256 均为 `519740deb419683d978c1eadca5879855d8818e9fb3c4bdc34e628e91a229d3e`。恢复批次为 `history_top25_seed100_20261124_212530`，manifest SHA-256 为 `e1c42f8b359614b12c135a0fad28ed0b3c419f7a427ea866f1b7bd4838419732`。此前本地 Round 1 的 `PENDING` 占位表因此退出 H1 输入路径。

| 冻结项 | 验证值 |
| --- | ---: |
| Dev 样本 / 历史 trace | 980 / 5,880 |
| H-top25 样本 / F0 错误 | 245 / 104 |
| U-top25 样本 / F0 错误 | 245 / 102 |
| 全 Dev F0 错误 | 287 |
| SHIFT / STABLE / NO_HISTORY | 273 / 667 / 40 |
| U_ONLY / H_ONLY / U_INTERSECT_H / NEITHER | 101 / 101 / 144 / 634 |

入口校验 ZIP 与解包目录逐字节相同、manifest 及每个文件 SHA-256、同一 F0 checkpoint、Dev 键与 gold、归档 logits、认证投影及 Test HOLD。H-top25、SHIFT/STABLE/NO_HISTORY 和 U/H 都直接读取冻结表，保持原名单。历史条件逐样本 trace 用于核验来源，分组只进入终点解释。

## H1 插入点与协议

- `TrackCore` 在 `row_operator(..., self_terms=True)` 完成后调用一次 H1，可选钩子在四层共享结果。`FusionTAV` 透传该钩子，从同一 forward 取得 `[B,T,300]` 的 `h0.detach()`。只收集严格 `j<i` 的合法边，余弦 `eps=1e-8`，随后 clip 到 `[-1,1]`。同、异说话人分别用一个原始标量，前向 clamp 到 `[0,0.5]`，初始均为 0。
- H1 用 `1−lambda*d` 乘原先归一化边贡献。原 self、残差、邻接、前端、融合和分类器继续使用 F0 的实现。F0 `state_dict` schema 保持原样，两个 H1 标量由外层持有。
- 正式入口复用 F0 checkpoint 工厂与 Train/Dev loader，固定 batch size 来自 F0（16）。在全 Dev 历史 tape 上检查 CTRL、显式旁路和零初始化的精确相等。F0 持续 eval 并冻结，训练只优化两个标量。每轮使用固定 seed 的 Train shuffle 和原 masked CE，Adam `lr=0.003`、`betas=(0.9,0.999)`、`weight_decay=0`，epoch 20 是唯一 PRIMARY 终点。
- 运行报告包含指标、迁移计数、类别混淆、冻结分层、边分布、校准和对话配对 bootstrap。每条合法边统计一次，四层共享同一衰减因子。边分布描述实际施加的系数，实验结论限于该衰减对冻结 F0 的效果。

## 本地验收

独立 HEAD 归档与候选归档的合成 F0 `h0…h4`、logits、mask、邻接逐字节相同，`state_dict` 键、形状和 extra state 一致。候选中也分别验证了显式 bypass 与零 lambda。CUDA 上的真实 F0 replay、正式优化和最终 Dev 指标由 Ubuntu 入口执行。本地本轮正式训练 0 次，Test 未触及。[测试报告](HISTORY_FASTKILL_TEST_REPORT.md) · [依赖闭包](HISTORY_FASTKILL_DEPENDENCY_CLOSURE_REPORT.md) · [Ubuntu 运行手册](HISTORY_FASTKILL_RUNBOOK.md)

## REQUIRED_MODIFIED_FILES

- `models/multidag_cl/fusion/model.py`
- `models/multidag_cl/graph_core_variants/track.py`
- `models/multidag_cl/history_attenuation.py`
- `scripts/analysis/models/multidag_cl/history_fastkill_provenance.py`
- `scripts/analysis/models/multidag_cl/history_fastkill_runtime.py`
- `scripts/analysis/models/multidag_cl/history_fastkill_metrics.py`
- `scripts/analysis/models/multidag_cl/run_history_fastkill.py`
- `configs/multidag_cl/tav_full_audit_v1/history_fastkill_seed100.yaml`
- `tests/models/multidag_cl/tav_full_audit_v1/test_history_attenuation.py`
- `docs/research/post_failure_method_design_001/HISTORY_FASTKILL_IMPLEMENTATION_REPORT.md`
- `docs/research/post_failure_method_design_001/HISTORY_FASTKILL_TEST_REPORT.md`
- `docs/research/post_failure_method_design_001/HISTORY_FASTKILL_RUNBOOK.md`
- `docs/research/post_failure_method_design_001/HISTORY_FASTKILL_DEPENDENCY_CLOSURE_REPORT.md`

## UNRELATED_DIRTY_FILES

以下 66 条来自任务开始时的 `git status --porcelain=v1 --untracked-files=all`。这次没有更改这些路径。

```text
 M .gitignore
 M docs/research/post_fusion/PARALLEL_FASTKILL_HANDOFF.md
 M docs/research/post_fusion/PARALLEL_FASTKILL_IMPLEMENTATION_REPORT.md
 M docs/research/post_fusion/PARALLEL_FASTKILL_RUNBOOK.md
 M docs/research/post_fusion/PARALLEL_FASTKILL_TEST_REPORT.md
 M docs/research/post_fusion/POST_FUSION_PARALLEL_FASTKILL_001_DESIGN.md
 M scripts/analysis/models/multidag_cl/export_fastkill_bundle.py
 M scripts/runtime/multidag_cl_fastkill/runtime.py
 M tests/models/multidag_cl/tav_full_audit_v1/test_post_fusion_fastkill.py
?? configs/multidag_cl/fusion_candidates/prototype_smoke.yaml
?? datasets/asymmetric_iemocap.py
?? docs/IEMOCAP_ASYMMETRIC_MERC_PROJECT_SOURCE_20261021_v1.2.md
?? docs/MERC_PATHS_PROJECT_SOURCE_20261022.md
?? docs/analysis/fusion/CT_MULTIDAG_FUSION_FINAL_SUMMARY_20261123.md
?? docs/analysis/fusion/FUSION_ARCHIVE_AUDIT_20261123.md
?? docs/fusion/CURRENT_FUSION_ARCHITECTURE_AUDIT_20261119.md
?? docs/fusion/FUSION_HYPOTHESIS_AND_EXPERIMENT_DESIGN_V1_20261119.md
?? docs/reproduction/MULTIDAG_CL_FEATURE_REPRODUCTION_STAGE_C1.md
?? docs/research/fusion/CT_MULTIDAG_FUSION_DESIGN_STUDY_20261121.md
?? docs/research/post_failure_method_design_001/01_FAILURE_MECHANISM_INTERPRETATION.md
?? docs/research/post_failure_method_design_001/02_RECENT_2024_2026_LITERATURE_MATRIX.csv
?? docs/research/post_failure_method_design_001/03_CORE_PAPER_DEEP_REVIEW.md
?? docs/research/post_failure_method_design_001/04_CANDIDATE_METHODS.md
?? docs/research/post_failure_method_design_001/05_METHOD_COMPARISON_MATRIX.csv
?? docs/research/post_failure_method_design_001/06_PRIMARY_METHOD_DESIGN.md
?? docs/research/post_failure_method_design_001/07_PRIMARY_METHOD_ARCHITECTURE.md
?? docs/research/post_failure_method_design_001/08_FASTKILL_EXPERIMENT_PLAN.md
?? docs/research/post_failure_method_design_001/09_RISKS_AND_FAILURE_MODES.md
?? docs/research/post_failure_method_design_001/10_FINAL_RECOMMENDATION.md
?? docs/research/post_failure_method_design_001/papers/L01_UAF.pdf
?? docs/research/post_failure_method_design_001/papers/L02_SRERC.pdf
?? docs/research/post_failure_method_design_001/papers/L03_ERCThinker.pdf
?? docs/research/post_failure_method_design_001/papers/L04_DCLF.pdf
?? docs/research/post_failure_method_design_001/papers/L05_HiRoC.pdf
?? docs/research/post_failure_method_design_001/papers/L06_ECERC.pdf
?? docs/research/post_failure_method_design_001/papers/L07_EmoTrans.pdf
?? docs/research/post_failure_method_design_001/papers/L08_Handling_Ambiguity.pdf
?? docs/research/post_failure_method_design_001/papers/L09_Decoupled_Neutral.pdf
?? docs/research/post_failure_method_design_001/papers/L10_SER_Uncertainty.pdf
?? docs/research/post_fusion/NEXT_ROUTE_CANDIDATE_MATRIX.csv
?? docs/research/post_fusion/ORIGINAL_MULTIDAG_FUTURE_WORK_TRACE.md
?? docs/research/post_fusion/POST_FUSION_ROUTE_AUDIT_001.md
?? docs/research/post_fusion/TOP2_ROUTE_RECOMMENDATION.md
?? docs/research/post_fusion/b1_signal_review_001/B1_NEW_ERRORS.csv
?? docs/research/post_fusion/b1_signal_review_001/B1_PAIRWISE_ERROR_MIGRATION.csv
?? docs/research/post_fusion/b1_signal_review_001/B1_PER_CLASS_EFFECT.csv
?? docs/research/post_fusion/b1_signal_review_001/B1_PROTOTYPE_GEOMETRY_REVIEW.csv
?? docs/research/post_fusion/b1_signal_review_001/B1_RECOVERED_ERRORS.csv
?? docs/research/post_fusion/b1_signal_review_001/B1_R_4SEED_RUNBOOK.md
?? docs/research/post_fusion/b1_signal_review_001/B1_R_CONFIRMATION_PROTOCOL.md
?? docs/research/post_fusion/b1_signal_review_001/B1_TRAINING_TRAJECTORY_REVIEW.csv
?? docs/research/post_fusion/b1_signal_review_001/FASTKILL_ROUTE_DECISION_FIX_REPORT.md
?? docs/research/post_fusion/b1_signal_review_001/HISTORICAL_F0_CONTROL_IDENTITY_AUDIT.md
?? docs/research/post_fusion/b1_signal_review_001/POST_FUSION_B1_SIGNAL_REVIEW_001.md
?? docs/research/visual_representation_audit/VA0_V342_S2D_PROVENANCE_AUDIT_20261120.md
?? docs/research/visual_representation_audit/VA1D_COMMON_POPULATION_VISUAL_PROBE_20261120.md
?? docs/research/visual_representation_audit/VA2D_COMMON_POPULATION_CT_MULTIDAG_VISUAL_REPLACEMENT_20261120.md
?? models/experimental/ct_multidag_fusion.py
?? scripts/analysis/models/multidag_cl/plot_overfitting_diagnostics.py
?? scripts/diagnostics/experiments/run_va1d_visual_representation_probe.py
?? scripts/diagnostics/experiments/run_va2d_visual_replacement.py
?? scripts/diagnostics/experiments/va2d_inputs.py
?? scripts/diagnostics/models/multidag_cl/smoke_ct_multidag_fusion.py
?? tests/analysis/models/multidag_cl/test_plot_overfitting_diagnostics.py
?? tests/data/build/multidag_cl/test_official_feature_pipeline.py
?? tests/test_iemocap_canonical_manifest.py
```
