# Script Analysis Layout Refactor Report

## 1. PRECHECK / 执行前 Git 状态

- Repo root：`E:/MERC/m3ed_mmgcn_clean`
- Branch：`refactor/model-first-layout`
- HEAD：`46af77a64b1a7347c46904a9710692a31576bf2f`
- Upstream：`origin/refactor/model-first-layout`，ahead/behind 为 `0/0`
- 最近两个重构 commit：`46af77a refactor: organize execution scripts and workflows`、`1988757 refactor: organize model implementations by provenance`
- Tracked working tree 与 staged 区在执行前干净。
- 仅有的 untracked 文件为受保护组会工具：
  - `scripts/analyze/export_group_meeting_baseline_report.py`
  - `tests/analyze/test_group_meeting_baseline_report.py`
- 两个受保护文件均未读取为生产代码依据、未修改、未 staged。

## 2. CONTEXT_READ

执行前按指定顺序读取：

1. `AGENTS.md`
2. `docs/PROJECT_CONTEXT.md`
3. `docs/project_map.md`
4. `docs/refactors/MODEL_LAYOUT_REFACTOR_REPORT.md`
5. `docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_MODEL_REFACTOR.md`
6. `docs/refactors/SCRIPT_EXECUTION_LAYOUT_REFACTOR_REPORT.md`
7. `docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3A.md`
8. `docs/codex_workflow.md`
9. `docs/git_workflow.md`
10. `docs/experiment_protocol.md`

没有创建另一套持久项目上下文。

## 3. BASELINE_GATES / 执行前门禁

- 受限沙箱内首次完整 pytest 只因既有 `tmp/pytest_*` 与 `.pytest_cache` 权限在收集阶段失败，没有测试断言失败；未删除或修改这些目录。
- 正常本地权限下同一完整 pytest：`239 passed, 3 skipped, 30 warnings`。
- `scripts/dev/validate_config_tree.py`：`Config validation passed.`
- `git diff --check`：PASS。
- 组会文件 staged 状态：NO。

## 4. TRACKED_ANALYSIS_INVENTORY 与职责分类

执行前 `git ls-files scripts/analyze scripts/analysis` 找到 20 个 tracked Python 文件。分类依据为 module docstring、argparse、main、imports、输入输出格式、模型/轨道耦合、调用关系、路径推导与写入行为，不依据文件名猜测。

| 执行前路径 | 源码确认的职责 | Phase 3B1 分类 |
|---|---|---|
| `scripts/analysis/analyze_original_merc_results.py` | 指向旧实现的 15 行兼容入口 | compatibility wrapper，改为直达最终 canonical |
| `scripts/analyze/analyze_original_merc_results.py` | Original MERC 三轨聚合、数值审计、论文 gap 与报告 | `paper_aligned` |
| `scripts/analyze/audit_iemocap_feature_pkl.py` | 比较 legacy/candidate feature PKL | deferred feature/data |
| `scripts/analyze/audit_model_causality.py` | 对多个 causal 模型执行未来扰动、梯度与边审计 | `causal` |
| `scripts/analyze/build_analysis_tables.py` | 跨模型构建 run/epoch/evaluation/per-class master tables | `common` |
| `scripts/analyze/diagnose_iemocap_splits.py` | 构造 dataset 并诊断 split/protocol | deferred diagnose |
| `scripts/analyze/diagnose_loss_stability.py` | 诊断 epoch loss 稳定性 | deferred diagnose |
| `scripts/analyze/diagnose_multidag_cl_run.py` | 诊断单个 MultiDAG+CL run | deferred diagnose |
| `scripts/analyze/export_original_merc_reproduction_report.py` | 指向 Original-MERC 分析的 15 行兼容别名 | compatibility wrapper，改为直达最终 canonical |
| `scripts/analyze/export_paper_artifacts.py` | 模型无关的单 run 论文表格/图像导出 | `common` |
| `scripts/analyze/export_paper_multi_run_tables.py` | 模型无关的多 run 论文表格导出 | `common` |
| `scripts/analyze/plot_missing_modality_summary.py` | 从通用 summary CSV 绘制缺失模态图 | `common` |
| `scripts/analyze/plot_multi_run_final_analysis.py` | 跨模型多 run 最终指标与 per-class 图表 | `common` |
| `scripts/analyze/plot_multi_run_training_curves.py` | 跨模型多 run epoch 曲线 | `common` |
| `scripts/analyze/plot_multidag_cl_stabilization_compare.py` | MultiDAG+CL 稳定化 run 专属比较 | `models/multidag_cl` |
| `scripts/analyze/plot_single_run_final_analysis.py` | 模型无关的单 run 最终评估图表 | `common` |
| `scripts/analyze/plot_single_run_training_curves.py` | 模型无关的单 run epoch 曲线 | `common` |
| `scripts/analyze/probe_iemocap_text_features.py` | 读取 feature PKL 并执行 text feature probe | deferred feature/data |
| `scripts/analyze/run_four_model_causal_audit.py` | 编排四模型 causal audit | `causal` |
| `scripts/analyze/summarize_causal_benchmark_runs.py` | 只读审计并汇总 causal 8-run benchmark | `causal` |

## 5. SCRIPT_MOVES / 所有 git mv

所有真实实现使用 `git mv`；没有复制实现后再删除。

| 旧实现路径 | 新 canonical 路径 |
|---|---|
| `scripts/analyze/build_analysis_tables.py` | `scripts/analysis/common/build_analysis_tables.py` |
| `scripts/analyze/export_paper_artifacts.py` | `scripts/analysis/common/export_paper_artifacts.py` |
| `scripts/analyze/export_paper_multi_run_tables.py` | `scripts/analysis/common/export_paper_multi_run_tables.py` |
| `scripts/analyze/plot_missing_modality_summary.py` | `scripts/analysis/common/plot_missing_modality_summary.py` |
| `scripts/analyze/plot_multi_run_final_analysis.py` | `scripts/analysis/common/plot_multi_run_final_analysis.py` |
| `scripts/analyze/plot_multi_run_training_curves.py` | `scripts/analysis/common/plot_multi_run_training_curves.py` |
| `scripts/analyze/plot_single_run_final_analysis.py` | `scripts/analysis/common/plot_single_run_final_analysis.py` |
| `scripts/analyze/plot_single_run_training_curves.py` | `scripts/analysis/common/plot_single_run_training_curves.py` |
| `scripts/analyze/audit_model_causality.py` | `scripts/analysis/causal/audit_model_causality.py` |
| `scripts/analyze/run_four_model_causal_audit.py` | `scripts/analysis/causal/run_four_model_causal_audit.py` |
| `scripts/analyze/summarize_causal_benchmark_runs.py` | `scripts/analysis/causal/summarize_causal_benchmark_runs.py` |
| `scripts/analyze/analyze_original_merc_results.py` | `scripts/analysis/paper_aligned/analyze_original_merc_results.py` |
| `scripts/analyze/plot_multidag_cl_stabilization_compare.py` | `scripts/analysis/models/multidag_cl/plot_stabilization_compare.py` |

合计移动 13 个实现文件。只创建有实际实现的 package 目录和 `__init__.py`，没有 `.gitkeep` 或空 `author_official` 目录。

## 6. CANONICAL_ANALYSIS_PATHS

- 通用分析：`scripts/analysis/common/`
- causal 分析：`scripts/analysis/causal/`
- Original MERC / paper-aligned 分析：`scripts/analysis/paper_aligned/`
- 模型专属分析：`scripts/analysis/models/multidag_cl/`

## 7. COMPATIBILITY_WRAPPERS

13 个被迁移的旧 tracked 路径均创建薄 module-alias wrapper。另有两个既有兼容入口改为直接指向最终 canonical module：

- `scripts/analysis/analyze_original_merc_results.py`
- `scripts/analyze/export_original_merc_reproduction_report.py`

因此当前兼容入口总数为 15，其中本阶段新建 13。wrapper 只调用 `scripts._load_compat_module`，不复制分析实现、不定义 argparse、不重算指标、不改变输出。import 时旧模块名与 canonical module 指向同一对象；CLI 执行时直接调用 canonical `main()` 并传播退出码。没有使用 `runpy`，也没有 wrapper 链或反向依赖。

## 8. DEFERRED_FILES

以下 5 个 tracked 文件职责属于 feature/data 或 diagnose，保持原位等待 Phase 3B2：

- `scripts/analyze/audit_iemocap_feature_pkl.py`
- `scripts/analyze/diagnose_iemocap_splits.py`
- `scripts/analyze/diagnose_loss_stability.py`
- `scripts/analyze/diagnose_multidag_cl_run.py`
- `scripts/analyze/probe_iemocap_text_features.py`

没有移动 debug、diagnose、inspect、prepare、features、maintenance 或 dev 目录。

## 9. CANONICAL_IMPORTS

canonical analysis 内部 import 已切换为：

- `models.registry.causal`
- `scripts.models.mmgcn.unified.train`
- `scripts.models.multidag_cl.unified.train`
- `scripts.runtime.causal_graph`
- `scripts.runtime.paper_aligned`
- canonical analysis 之间使用 `scripts.analysis...`

canonical analysis 中 `models.baselines`、`scripts.analyze`、旧 execution wrapper import 均为 0。两个原脚本在 Python 3.8 下因 `str | None` 注解缺少延迟注解 import 而无法 import；仅增加 `from __future__ import annotations`，不改变分析行为。

## 10. CLI_COMPATIBILITY

`tests/test_analysis_layout_compatibility.py` 对 13 组迁移前/后入口执行 `--help`：

- 返回码全部为 0。
- 每组关键 option 一致。
- help 运行 cwd 与 matplotlib cache 均位于 pytest 临时目录。
- 未读取正式 outputs、未生成报告、未启动训练。

专项测试整体结果：`31 passed`。

## 11. SYNTHETIC_ANALYSIS_TESTS

使用 `tmp_path` 创建包含 `setting,loss,acc,uar,macro_f1,weighted_f1` 的最小 missing-modality summary CSV，分别调用旧入口和 canonical 入口。验证：

- 返回码相同且为 0。
- 输出文件名集合相同。
- `summary_sorted.csv` schema、行顺序和数值内容相同。
- PNG/PDF 均存在且非空。
- PNG 尺寸与 DPI 元数据相同。

没有读取正式 outputs。

## 12. PATH_RESOLUTION

- `scripts/analysis/{common,causal,paper_aligned}/` 的 repo root 调整为相对新层级的 `parents[3]`。
- `scripts/analysis/models/multidag_cl/` 调整为 `parents[4]`。
- 兼容测试导入所有带 root 常量的 canonical module，均解析到当前仓库根。
- 显式 input/output/config/run/manifest 路径语义未改；日期化输出 helper 继续来自 `utils.output_paths`。
- 未引入本地或远程绝对路径。

## 13. OUTPUT_SCHEMA_COMPATIBILITY

本阶段未修改 Accuracy、Weighted-F1、Macro-F1、UAR、confusion matrix、per-class 指标、mean/std、checkpoint 选择、validation ranking、test 使用规则、label 顺序、prediction schema、finite/probability 校验、CSV/JSON 字段、图形数据、DPI、语言、字体回退或论文表格数值。

`OUTPUT_SCHEMA_CHANGED=NO`，`METRIC_IMPLEMENTATION_CHANGED=NO`。

## 14. IMPORT_AUDIT

最终 AST 与旧 CLI 文本引用审计见：

- `docs/refactors/LEGACY_ANALYSIS_IMPORTS_AFTER_PHASE3B1.csv`
- `docs/refactors/LEGACY_ANALYSIS_IMPORTS_AFTER_PHASE3B1.md`

- `CANONICAL_ANALYSIS_LEGACY_IMPORTS=0`
- tests 中旧 analysis import edges：`6`
- compatibility wrappers：`15`
- deferred analysis files：`5`
- 旧 CLI path 文本引用：`139`

两份本地组会文件未进入 tracked AST 清单、未解析。

## 15. TARGETED_TESTS

- 新 analysis layout compatibility：`31 passed`
- 直接使用旧 analysis import 的既有 tests：`41 passed`
- 合计 targeted：`72 passed`

## 16. FULL_PYTEST 与 FULL_GATES

- 完整 pytest：`270 passed, 3 skipped, 30 warnings`
- Config validation：`Config validation passed.`
- `git diff --check`：PASS
- canonical legacy analysis import gate：PASS（0 edges）
- CLI compatibility：PASS
- synthetic analysis：PASS

## 17. PERSISTENT_CONTEXT

已增量更新：

- `AGENTS.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/project_map.md`

只把 Model Layout、Script Phase 3A 和 Script Phase 3B1 写为 completed；Phase 3B2、config layout、作者官方 MultiDAG+CL/GS-MCC 接入和最终 baseline 选择仍明确为未开始/未完成。

## 18. PROTECTED_PATHS

- `models/`、`configs/`、`datasets/`、`utils/`：0 个变更。
- `scripts/models/`、`scripts/runtime/`、`scripts/evaluation/`、`scripts/workflows/`：0 个变更。
- `data/`、`third_party/`、`outputs/`、`tmp/`：未移动或修改。
- 正式训练：未启动。
- 两份组会文件：unchanged、untracked、not staged。

## 19. NEXT_PHASE

下一阶段为 Script Phase 3B2：处理 debug、diagnose、feature/data、inspect、maintenance 与 dev/support scripts。Phase 3B2 和完整门禁完成后，才规划 `configs/` 重构。不得提前删除 Phase 3A/3B1 compatibility wrapper，也不得提前接入或宣称完成作者官方 MultiDAG+CL/GS-MCC。

## 20. 回滚方法

本阶段没有 commit。回滚前先再次确认两份组会文件仍为 untracked/not staged。不得使用 `git clean` 或 `git reset --hard`；只对本报告列出的 tracked paths 使用 `git restore --staged` / `git restore`，并逐项人工移除本阶段新增的 wrapper、package、测试和文档。也可保留 HEAD `46af77a64b1a7347c46904a9710692a31576bf2f` 作为 Phase 3B1 前边界。

## 21. 状态摘要

- `SCRIPT_ANALYSIS_LAYOUT_REFACTOR_STATUS=PASS`
- `FILES_MOVED=13`
- `COMPATIBILITY_WRAPPERS_CREATED=13`
- `DEFERRED_ANALYSIS_FILES=5`
- `CANONICAL_ANALYSIS_LEGACY_IMPORTS=0`
- `CLI_COMPATIBILITY_STATUS=PASS`
- `SYNTHETIC_ANALYSIS_STATUS=PASS`
- `OUTPUT_SCHEMA_CHANGED=NO`
- `METRIC_IMPLEMENTATION_CHANGED=NO`
- `CONFIGS_MOVED=0`
- `MODELS_CHANGED=NO`
- `FORMAL_TRAINING_STARTED=0`
- `GROUP_MEETING_FILES_CHANGED=NO`
- `GROUP_MEETING_FILES_STAGED=NO`
- `COMMIT_CREATED=NO`
- `PUSH_PERFORMED=NO`
- `READY_FOR_SCRIPT_PHASE3B2=YES`
