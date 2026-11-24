# Pairwise Fast-Kill 依赖闭包审计

审计日期为 2026-11-24。工作流为 MULTIDAG，分支为 `feat/multidag-cl-paper-reimplementation`，工作流门禁通过。基线为 `cf36b54627cd5d7f594b48004c2b4ad4bbd585fb`。

## 根因

最初的 Fast-Kill 提交包含调用方，依赖的辅助函数和 F0 捕获接口分批进入 Git。分别归档三个历史提交，在归档目录中复现了各次错误。

| 历史提交 | 归档复现 | 修复已进入 |
| --- | --- | --- |
| `d03ae19` | Fast-Kill 导入 `logit_probabilities` 失败 | `ac0f558` |
| `ac0f558` | Fast-Kill 导入 `ranked_candidates` 失败 | `e0fc919` |
| `e0fc919` | `forward_f0() got an unexpected keyword argument 'capture_states'` | `cf36b54` |

`cf36b54` 已包含 `FrozenSplit.layer_states`、两个 `capture_states=False` 默认值和 `forward_f0(..., inspect_model=None)`。捕获从同一次 eval/inference forward 输出按有效行提取第 0、4 层。Dev 的 batch tape、人口顺序与标签、归档 logits 容差、argmax 一致性和 checkpoint SHA 校验仍由原生产路径执行。冻结 F0、分类器和 Fast-Kill 科学设计保持当前 HEAD 的实现。

## 闭包范围与结果

从 `scripts/analysis/models/multidag_cl/run_pairwise_fastkill.py` 递归解析工作树和 HEAD 的项目内绝对、相对及函数内导入，纳入包初始化文件，并检索动态导入调用。另以四个验收测试及 F0 测试的 `conftest.py` 为根检查测试依赖。

两边均得到 96 个运行时 Python 文件，另有 9 个测试侧文件。对路径全集运行 `git status --short -- <paths>`、`git diff -- <paths>` 和 `git diff HEAD -- <paths>`。修改前均无差异，修改后唯一 Python 差异为回归测试。入口默认使用的 `configs/multidag_cl/tav_full_audit_v1/pairwise_fastkill_seed100.yaml` 也已跟踪且无差异。

<details>
<summary>96 个运行时文件</summary>

```text
datasets/__init__.py
datasets/iemocap/__init__.py
datasets/iemocap/official_feature_dataset.py
datasets/iemocap/original_repro_split.py
datasets/iemocap/split_utils.py
datasets/iemocap/ta_v0_dataset.py
datasets/iemocap/tav_full_audit_dataset.py
models/__init__.py
models/common/__init__.py
models/common/causal_graph/__init__.py
models/common/causal_graph/graph_builders.py
models/common/causal_graph/masked_ops.py
models/common/causal_graph/relation_utils.py
models/multidag_cl/__init__.py
models/multidag_cl/fusion/__init__.py
models/multidag_cl/fusion/base.py
models/multidag_cl/fusion/experts.py
models/multidag_cl/fusion/matching.py
models/multidag_cl/fusion/model.py
models/multidag_cl/fusion/registry.py
models/multidag_cl/graph_core_variants/__init__.py
models/multidag_cl/graph_core_variants/common.py
models/multidag_cl/graph_core_variants/config.py
models/multidag_cl/graph_core_variants/coop.py
models/multidag_cl/graph_core_variants/dcn.py
models/multidag_cl/graph_core_variants/model.py
models/multidag_cl/graph_core_variants/track.py
models/multidag_cl/pairwise_refinement.py
models/multidag_cl/paper_reimplementation/__init__.py
models/multidag_cl/paper_reimplementation/attention.py
models/multidag_cl/paper_reimplementation/config.py
models/multidag_cl/paper_reimplementation/contracts.py
models/multidag_cl/paper_reimplementation/curriculum/__init__.py
models/multidag_cl/paper_reimplementation/curriculum/buckets.py
models/multidag_cl/paper_reimplementation/curriculum/difficulty.py
models/multidag_cl/paper_reimplementation/curriculum/schedule.py
models/multidag_cl/paper_reimplementation/dag_layer.py
models/multidag_cl/paper_reimplementation/encoders.py
models/multidag_cl/paper_reimplementation/graph.py
models/multidag_cl/paper_reimplementation/model.py
models/multidag_cl/ta_v0/__init__.py
models/multidag_cl/ta_v0/causal_history_gate.py
models/multidag_cl/ta_v0/config.py
models/multidag_cl/ta_v0/encoders.py
models/multidag_cl/ta_v0/model.py
models/multidag_cl/tav_full_audit_v1/__init__.py
models/multidag_cl/tav_full_audit_v1/model.py
models/registry/__init__.py
models/registry/paper_reimplementation.py
scripts/__init__.py
scripts/analysis/__init__.py
scripts/analysis/models/__init__.py
scripts/analysis/models/multidag_cl/__init__.py
scripts/analysis/models/multidag_cl/export_fusion_topology_bundle.py
scripts/analysis/models/multidag_cl/failure_audit_inference.py
scripts/analysis/models/multidag_cl/failure_audit_metrics.py
scripts/analysis/models/multidag_cl/pairwise_fastkill_metrics.py
scripts/analysis/models/multidag_cl/pairwise_fastkill_runtime.py
scripts/analysis/models/multidag_cl/pairwise_preflight_inputs.py
scripts/analysis/models/multidag_cl/pairwise_preflight_metrics.py
scripts/analysis/models/multidag_cl/run_pairwise_fastkill.py
scripts/analysis/models/multidag_cl/run_pairwise_preflight.py
scripts/analysis/models/multidag_cl/summarize_tav_full_audit_v1.py
scripts/dev/__init__.py
scripts/dev/check_workstream.py
scripts/models/__init__.py
scripts/models/multidag_cl/__init__.py
scripts/models/multidag_cl/ta_v0/export_dev.py
scripts/models/multidag_cl/ta_v0/train.py
scripts/runtime/__init__.py
scripts/runtime/multidag_cl_fusion/__init__.py
scripts/runtime/multidag_cl_fusion/runtime.py
scripts/runtime/multidag_cl_fusion/spec.py
scripts/runtime/multidag_cl_paper_reimplementation/__init__.py
scripts/runtime/multidag_cl_paper_reimplementation/adapter.py
scripts/runtime/multidag_cl_paper_reimplementation/checkpoint.py
scripts/runtime/multidag_cl_paper_reimplementation/curriculum.py
scripts/runtime/multidag_cl_paper_reimplementation/diagnostic_artifacts.py
scripts/runtime/multidag_cl_paper_reimplementation/diagnostics.py
scripts/runtime/multidag_cl_paper_reimplementation/evaluation.py
scripts/runtime/multidag_cl_paper_reimplementation/manifest.py
scripts/runtime/multidag_cl_paper_reimplementation/optimizer.py
scripts/runtime/multidag_cl_paper_reimplementation/trainer.py
scripts/runtime/multidag_cl_paper_reimplementation/validation.py
scripts/runtime/multidag_cl_ta_v0/__init__.py
scripts/runtime/multidag_cl_ta_v0/formal.py
scripts/runtime/multidag_cl_ta_v0/training.py
scripts/runtime/multidag_cl_tav_full_audit_v1/__init__.py
scripts/runtime/multidag_cl_tav_full_audit_v1/confirm.py
scripts/runtime/multidag_cl_tav_full_audit_v1/runtime.py
scripts/runtime/multidag_cl_tav_full_audit_v1/spec.py
utils/__init__.py
utils/iemocap_features.py
utils/metrics.py
utils/output_paths.py
utils/run_metadata.py
```

</details>

<details>
<summary>9 个额外的测试侧文件</summary>

```text
scripts/data/build/multidag_cl/__init__.py
scripts/data/build/multidag_cl/build_tav_t1024_v1.py
scripts/data/build/multidag_cl/build_train_dev_projection.py
scripts/data/build/multidag_cl/schema.py
tests/data/build/multidag_cl/test_train_dev_projection.py
tests/models/multidag_cl/tav_full_audit_v1/conftest.py
tests/models/multidag_cl/tav_full_audit_v1/test_failure_audit.py
tests/models/multidag_cl/tav_full_audit_v1/test_pairwise_fastkill.py
tests/models/multidag_cl/tav_full_audit_v1/test_pairwise_preflight.py
```

</details>

## 归档验证

现有 HEAD 直接通过 `git archive HEAD` 导出，`--help` 退出码为 0，四个指定测试模块共 73 项通过。

候选以相同 HEAD 为基底，在独立临时 Git 库的索引中只纳入下方两条路径。将该树绑定到临时库的 HEAD 后，用 `git archive HEAD` 导出到另一目录。项目工作区的 HEAD 和索引保持原状。Python 隔离进程核对了 99 个已加载项目模块的来源路径，全部位于归档目录。

| 候选检查 | 结果 |
| --- | --- |
| `python -B scripts/analysis/models/multidag_cl/run_pairwise_fastkill.py --help` | `CLI_EXIT = 0` |
| `test_pairwise_fastkill.py`、`test_pairwise_preflight.py`、`test_failure_audit.py`、`test_train_dev_projection.py` | 75 passed，`PYTEST_EXIT = 0` |
| 合成 Fast-Kill 执行 | epoch 20 head、一次 Dev 评分、轻量归档与无效输入路径通过 |
| 真实 F0 工厂加载合成 checkpoint | Train/Dev 捕获前后 logits、argmax 和样本顺序一致 |
| 状态与冻结校验 | 每个 split 仅返回 `h0`、`h4`，与模型对应层的原始有效行完全相同，均为 finite `float32 [N_valid, 300]`。模型为 eval、参数冻结、`grad=None`，参数值及 checkpoint SHA 相同 |

验证使用 Windows、Python 3.9.25 和 PyTorch 2.6.0 CPU。Windows 沙箱拒绝访问以 `0700` 创建的临时目录，归档外的 pytest 启动器仅在本次可丢弃的测试临时目录内让目录继承父目录 ACL。四个测试模块保持归档里的版本。真实 IEMOCAP Test 数据和远端正式运行未参与验证。

## REQUIRED_MODIFIED_FILES

- `tests/models/multidag_cl/tav_full_audit_v1/test_pairwise_fastkill.py`
- `docs/research/post_failure_method_design_001/PAIRWISE_FASTKILL_DEPENDENCY_CLOSURE_REPORT.md`

## UNRELATED_DIRTY_FILES

以下为审计开始时已存在的无关工作树项目。三个点名文件均在 Fast-Kill 闭包之外。

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

## 验收状态

```text
PAIRWISE_FASTKILL_DEPENDENCY_CLOSURE = PASS
CLEAN_HEAD_CLI = PASS
CLEAN_HEAD_TESTS = 75 passed
F0_CAPTURE_IDENTITY = PASS
F0_CORE_MODIFIED = NO
FORMAL_TRAIN_RUNS = 0
TEST_TOUCHED = NO
READY_FOR_GIT_COMMIT = YES
```
