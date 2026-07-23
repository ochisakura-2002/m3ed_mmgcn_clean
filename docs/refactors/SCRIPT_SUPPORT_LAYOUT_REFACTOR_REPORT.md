# Script Support Layout Refactor Report

## 1. PRECHECK / 执行前 Git 状态

- Repo root：`E:/MERC/m3ed_mmgcn_clean`
- Branch：`refactor/model-first-layout`
- HEAD：`63dfe55fca2e3883af7736aeb081423b147b3d26`
- Upstream：`origin/refactor/model-first-layout`，ahead/behind=`0/0`
- 最近三项重构 commit：
  - `63dfe55 refactor: organize analysis scripts by responsibility`
  - `46af77a refactor: organize execution scripts and workflows`
  - `1988757 refactor: organize model implementations by provenance`
- 执行前 tracked working tree 与 staged 区干净。
- 仅有两份允许的 untracked 组会文件；均未修改、未 staged。

## 2. CONTEXT_READ

执行前按任务指定顺序读取：

1. `AGENTS.md`
2. `docs/PROJECT_CONTEXT.md`
3. `docs/project_map.md`
4. `docs/refactors/MODEL_LAYOUT_REFACTOR_REPORT.md`
5. `docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_MODEL_REFACTOR.md`
6. `docs/refactors/SCRIPT_EXECUTION_LAYOUT_REFACTOR_REPORT.md`
7. `docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3A.md`
8. `docs/refactors/SCRIPT_ANALYSIS_LAYOUT_REFACTOR_REPORT.md`
9. `docs/refactors/LEGACY_ANALYSIS_IMPORTS_AFTER_PHASE3B1.md`
10. `docs/codex_workflow.md`
11. `docs/git_workflow.md`
12. `docs/experiment_protocol.md`

没有建立竞争性的持久上下文。

## 3. BASELINE_GATES / 执行前门禁

- 受限沙箱内完整 pytest 因既有 `tmp/pytest_*` 与 `.pytest_cache` 权限在
  collection 阶段失败，没有测试断言失败，也没有修改或删除这些目录。
- 正常本地权限下同一完整 pytest：`270 passed, 3 skipped, 30 warnings`。
- `scripts/dev/validate_config_tree.py`：`Config validation passed.`
- `git diff --check`：PASS。

## 4. SUPPORT_SCRIPT_INVENTORY

按 `git ls-files` 盘点了以下范围：

- `scripts/debug/`：8
- `scripts/diagnose/`：1
- `scripts/features/`：1
- `scripts/prepare/`：1
- `scripts/inspect/`：5
- `scripts/maintenance/`：3
- `scripts/dev/`：2
- Phase 3B1 deferred `scripts/analyze/`：5
- Phase 3A deferred baseline debug：6

合计 32 个候选 Python 文件。inventory 没有包含两份 untracked 组会文件。

## 5. CLASSIFICATION

分类依据为 module docstring、`main`、argparse、imports、输入输出、数据写入、
PASS/FAIL/异常语义、模型/数据集耦合、tests/docs/subprocess 引用、repo-root
推导与旧 model import，而不是文件名。

| Canonical scope | 文件数 |
|---|---:|
| `data/build` | 1 |
| `data/prepare` | 1 |
| `data/inspect` | 3 |
| `diagnostics/data` | 9 |
| `diagnostics/models` | 11 |
| `diagnostics/experiments` | 3 |
| `maintenance` | 2 |
| `dev` | 2 |

机器可读结果：
`docs/refactors/SCRIPT_SUPPORT_CLASSIFICATION_PHASE3B2.csv`。

全部 32 个候选的 confidence 为 `high`；`MANUAL_REVIEW_FILES=0`。

## 6. SCRIPT_MOVES

所有真实实现均使用 `git mv`，没有复制实现后删除：

| 旧实现路径 | 新 canonical 路径 |
|---|---|
| `scripts/analyze/audit_iemocap_feature_pkl.py` | `scripts/diagnostics/data/audit_iemocap_feature_pkl.py` |
| `scripts/analyze/diagnose_iemocap_splits.py` | `scripts/diagnostics/data/diagnose_iemocap_splits.py` |
| `scripts/analyze/diagnose_loss_stability.py` | `scripts/diagnostics/experiments/diagnose_loss_stability.py` |
| `scripts/analyze/diagnose_multidag_cl_run.py` | `scripts/diagnostics/experiments/diagnose_multidag_cl_run.py` |
| `scripts/analyze/probe_iemocap_text_features.py` | `scripts/diagnostics/data/probe_iemocap_text_features.py` |
| `scripts/baselines/debug_causal_dialoguegcn_forward.py` | `scripts/diagnostics/models/dialoguegcn/debug_causal_dialoguegcn_forward.py` |
| `scripts/baselines/debug_causal_gsmcc_forward.py` | `scripts/diagnostics/models/gsmcc/debug_causal_gsmcc_forward.py` |
| `scripts/baselines/debug_multidag_cl_forward.py` | `scripts/diagnostics/models/multidag_cl/debug_multidag_cl_forward.py` |
| `scripts/baselines/debug_multidag_cl_real_batch.py` | `scripts/diagnostics/models/multidag_cl/debug_multidag_cl_real_batch.py` |
| `scripts/baselines/debug_new_causal_graph_real_batch.py` | `scripts/diagnostics/models/debug_new_causal_graph_real_batch.py` |
| `scripts/baselines/debug_sdt_forward.py` | `scripts/diagnostics/models/sdt/debug_sdt_forward.py` |
| `scripts/debug/debug_dialogue_feature_dataset.py` | `scripts/diagnostics/data/debug_dialogue_feature_dataset.py` |
| `scripts/debug/debug_iemocap_dataloader.py` | `scripts/diagnostics/data/debug_iemocap_dataloader.py` |
| `scripts/debug/debug_io.py` | `scripts/diagnostics/experiments/debug_io.py` |
| `scripts/debug/debug_m3ed_dataloader.py` | `scripts/diagnostics/data/debug_m3ed_dataloader.py` |
| `scripts/debug/debug_m3ed_dataset.py` | `scripts/diagnostics/data/debug_m3ed_dataset.py` |
| `scripts/debug/debug_m3ed_feature_dataset.py` | `scripts/diagnostics/data/debug_m3ed_feature_dataset.py` |
| `scripts/debug/debug_mmgcn_forward.py` | `scripts/diagnostics/models/mmgcn/debug_mmgcn_forward.py` |
| `scripts/debug/debug_simple_mlp_step.py` | `scripts/diagnostics/models/simple_mlp/debug_simple_mlp_step.py` |
| `scripts/dev/diagnose_gsmcc_numerics.py` | `scripts/diagnostics/models/gsmcc/diagnose_gsmcc_numerics.py` |
| `scripts/diagnose/diagnose_m3ed_label_alignment.py` | `scripts/diagnostics/data/diagnose_m3ed_label_alignment.py` |
| `scripts/features/build_iemocap_clean_text_features.py` | `scripts/data/build/build_iemocap_clean_text_features.py` |
| `scripts/inspect/extract_mmgcn_core_blocks.py` | `scripts/diagnostics/models/mmgcn/extract_mmgcn_core_blocks.py` |
| `scripts/inspect/inspect_m3ed_feature_files.py` | `scripts/data/inspect/inspect_m3ed_feature_files.py` |
| `scripts/inspect/inspect_m3ed_official.py` | `scripts/data/inspect/inspect_m3ed_official.py` |
| `scripts/inspect/inspect_mmgcn_feature_pkl.py` | `scripts/data/inspect/inspect_mmgcn_feature_pkl.py` |
| `scripts/inspect/inspect_mmgcn_official_model.py` | `scripts/diagnostics/models/mmgcn/inspect_mmgcn_official_model.py` |
| `scripts/maintenance/check_env.py` | `scripts/dev/check_env.py` |
| `scripts/prepare/prepare_m3ed_metadata.py` | `scripts/data/prepare/prepare_m3ed_metadata.py` |

`FILES_MOVED=29`。

## 7. CANONICAL_SUPPORT_PATHS

- `scripts/data/{build,prepare,inspect}/`
- `scripts/diagnostics/{data,models,experiments}/`
- `scripts/maintenance/`
- `scripts/dev/`

只创建包含实际实现或 package metadata 的目录；没有 `.gitkeep`、空模型目录或
`author_official` 占位目录。

## 8. COMPATIBILITY_WRAPPERS

29 个旧 tracked 路径全部保留一跳 module-alias wrapper。wrapper 只：

1. 从旧固定层级解析 repo root；
2. import `scripts._load_compat_module`；
3. 委派最终 canonical module；
4. CLI 模式调用 canonical `main()` 并传播退出码。

wrapper 不定义函数、class、argparse、诊断/构建算法或 model import；canonical
实现不反向 import 旧 wrapper。`COMPATIBILITY_WRAPPERS_CREATED=29`。

## 9. UNCHANGED_DEV_AND_MAINTENANCE

源码职责已正确，因此保持原位：

- `scripts/dev/validate_config_tree.py`
- `scripts/maintenance/collect_evaluation_summary.py`
- `scripts/maintenance/rebuild_experiment_summary.py`

仅为 `scripts/dev/` 与 `scripts/maintenance/` 增加 package `__init__.py`。

## 10. CANONICAL MODEL IMPORT UPDATES

Phase 3B2 canonical implementation 将旧 import 切换到：

- `models.registry.{causal,paper_aligned}`
- `models.common.causal_graph`
- `models.{dialoguegcn,mmgcn,multidag_cl}.unified`
- `models.multidag_cl.paper_aligned`
- `models.gsmcc.project_variant.{causal,full_context}`
- `models.experimental.sdt`
- `models.simple_mlp.model`
- `scripts.runtime.{causal_graph,paper_aligned}`

没有修改 `models/`、模型数学、forward、loss、阈值或 state-dict schema。

## 11. LEGACY MODEL IMPORT AST AUDIT

AST 覆盖 Git-tracked `scripts/`、`tests/` 以及本阶段 working-tree
wrapper/package/test，共 188 个 Python 文件，解析错误为 0。独立 `git grep`
交叉检查覆盖全部 AST import 行。

| Scope | Legacy model import edges |
|---|---:|
| Canonical execution | 0 |
| Canonical analysis | 0 |
| Canonical support | 0 |
| Compatibility wrappers | 0 |
| Deferred production scripts | 0 |
| Tests | 10 |

剩余 10 条全部位于专用 `tests/test_model_layout_compatibility.py`。详细结果：

- `docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3B2.csv`
- `docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3B2.md`

## 12. PATH_RESOLUTION

移动后按真实层级更新 repo-root 推导：

- `scripts/data/*`、`scripts/diagnostics/{data,experiments}/` 与 generic
  `scripts/diagnostics/models/`：`parents[3]`
- `scripts/diagnostics/models/<model>/`：`parents[4]`
- `scripts/dev/check_env.py`：同层级迁移，继续 `parents[2]`

29 个 canonical module 的 import/path test 验证所有 `PROJECT_ROOT` 均解析为
当前仓库根。没有硬编码本地或远程绝对路径。

## 13. CLI COMPATIBILITY

对 13 组带 CLI 的迁移脚本分别运行旧/新 `--help`：

- 返回码全部为 0；
- 关键 option 一致；
- matplotlib/cache 目录使用 pytest 临时目录；
- 未读取正式数据、未启动训练、未执行维护动作。

无 argparse 的危险工具仅做 import/委派/静态检查。

## 14. SYNTHETIC DATA/BUILD TESTS

使用 pytest `tmp_path` 与 fake tokenizer/model 构建最小 nine-item IEMOCAP
feature PKL。旧/新入口：

- 输出文件名相同；
- pickle 字节内容相同；
- metadata schema 与内容（除时间戳）相同；
- 输入 SHA 保持不变；
- 所有输出只写 pytest 临时目录。

`SYNTHETIC_DATA_STATUS=PASS`。

## 15. DIAGNOSTIC TESTS

使用 synthetic nine-item PKL 验证 feature audit：

- PASS 情况 old/new summary、CSV/Markdown/JSON 输出一致；
- NaN FAIL 情况 old/new strict 返回码均为 1；
- failed check 与输出目录内容一致。

真实 batch、正式数据和正式 checkpoint 诊断没有执行。
`DIAGNOSTIC_STATUS=PASS`。

## 16. INSPECT TESTS

对迁移后的 feature-PKL inspect 入口使用 synthetic pickle：

- old/new stdout 一致；
- 输入 SHA 前后不变；
- 未写正式 data 或 outputs。

`INSPECT_STATUS=PASS`。

## 17. MAINTENANCE SAFETY

- `collect_evaluation_summary.py --help`：PASS。
- `rebuild_experiment_summary.py --help`：PASS。
- `check_env.py` 已分类为 read-only dev gate。
- `prepare_m3ed_metadata.py`、`debug_io.py`、真实 batch/model 诊断与 summary
  rebuild 只做 import、CLI help、mock/静态检查，未执行真实动作。

`MAINTENANCE_SAFETY_STATUS=PASS`。

## 18. TARGETED TESTS

- Support layout compatibility：`78 passed`
- 所有 canonical model import 更新涉及的既有 tests：
  `103 passed, 3 skipped`
- AST parse：45 个 Phase 3B2 support/test 文件，0 error

## 19. FULL PYTEST

- 执行前：`270 passed, 3 skipped, 30 warnings`
- 执行后：`348 passed, 3 skipped, 30 warnings`
- 新增 support compatibility tests：78

完整 pytest 未启动正式训练，也未修改正式数据或 outputs。

## 20. CONFIG VALIDATION 与 DIFF CHECK

- `scripts/dev/validate_config_tree.py`：`Config validation passed.`
- `git diff --check`：PASS。
- `configs/` 内容与路径未修改、未移动。

## 21. PROTECTED PATHS

无变更：

- `models/`
- `configs/`
- `scripts/models/`
- `scripts/runtime/`
- `scripts/evaluation/`
- `scripts/workflows/`
- `scripts/analysis/`
- `datasets/`
- `utils/`
- `data/`
- `third_party/`
- `outputs/`
- `tmp/`

没有正式训练、网络下载、依赖安装或历史资产清理。

## 22. GROUP MEETING FILES

- `scripts/analyze/export_group_meeting_baseline_report.py`
- `tests/analyze/test_group_meeting_baseline_report.py`

两文件保持 untracked、unchanged、not staged；没有用于决定生产结构，也没有
修改 `.gitignore` 或 `.git/info/exclude`。

## 23. PERSISTENT CONTEXT

已增量更新：

- `AGENTS.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/project_map.md`

只把 model layout 与 Script Phase 3A/3B1/3B2 写为 completed；config layout、
作者官方 MultiDAG+CL/GS-MCC 与最终 baseline 选择仍为未开始/未完成。

## 24. ALL SCRIPT LAYOUT STATUS

Model layout、execution scripts、analysis scripts 与 support scripts 均已建立
canonical tree 并通过完整门禁。旧 model/script wrapper 继续保留，删除必须
另立任务。

`ALL_SCRIPT_LAYOUT_REFACTOR_STATUS=PASS`。

## 25. NEXT PHASE

下一阶段是 model-first config layout refactor。`configs/` 当前仍使用旧结构，
其旧 script path 继续由 compatibility wrapper 支持。config 门禁完成前不接入
或宣称完成作者官方 MultiDAG+CL/GS-MCC。

## 26. ROLLBACK

本阶段没有 commit。回滚前先确认两份组会文件仍为 untracked/not staged。
不得使用 `git clean` 或 `git reset --hard`；只对本报告列出的 task paths
逐项撤销 staged rename 与 working-tree 修改，并人工确认本阶段新增 wrapper、
package、test 与文档。HEAD
`63dfe55fca2e3883af7736aeb081423b147b3d26` 是 Phase 3B2 前边界。

## 27. STATUS SUMMARY

- `SCRIPT_SUPPORT_LAYOUT_REFACTOR_STATUS=PASS`
- `FILES_MOVED=29`
- `COMPATIBILITY_WRAPPERS_CREATED=29`
- `MANUAL_REVIEW_FILES=0`
- `CANONICAL_SUPPORT_LEGACY_MODEL_IMPORTS=0`
- `TRACKED_PRODUCTION_SCRIPT_LEGACY_MODEL_IMPORTS=0`
- `CLI_COMPATIBILITY_STATUS=PASS`
- `PATH_RESOLUTION_STATUS=PASS`
- `SYNTHETIC_DATA_STATUS=PASS`
- `DIAGNOSTIC_STATUS=PASS`
- `INSPECT_STATUS=PASS`
- `MAINTENANCE_SAFETY_STATUS=PASS`
- `DATA_BEHAVIOR_CHANGED=NO`
- `DIAGNOSTIC_RULES_CHANGED=NO`
- `OUTPUT_SCHEMA_CHANGED=NO`
- `CONFIGS_MOVED=0`
- `MODELS_CHANGED=NO`
- `FORMAL_TRAINING_STARTED=0`
- `GROUP_MEETING_FILES_CHANGED=NO`
- `GROUP_MEETING_FILES_STAGED=NO`
- `COMMIT_CREATED=NO`
- `PUSH_PERFORMED=NO`
- `ALL_SCRIPT_LAYOUT_REFACTOR_STATUS=PASS`
- `READY_FOR_CONFIG_REFACTOR=YES`
