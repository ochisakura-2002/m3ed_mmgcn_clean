# Script Execution Layout Refactor Report

## 1. PRECHECK / 执行前 Git 状态

- Repo root：`E:/MERC/m3ed_mmgcn_clean`
- Branch：`refactor/model-first-layout`
- HEAD：`1988757fffccfd50532ed4a20d15fedb017eb639`
- HEAD commit：`refactor: organize model implementations by provenance`
- Tracked working tree：干净。
- Staged changes：无。
- 仅有的 untracked 文件：
  - `scripts/analyze/export_group_meeting_baseline_report.py`
  - `tests/analyze/test_group_meeting_baseline_report.py`
- 模型目录重构已作为独立 commit 存在，当前分支与预期一致。

## 2. BASELINE_GATES / 执行前门禁

- 受限沙箱内首次完整 pytest 因既有 `tmp/pytest_*` 和
  `.pytest_cache` 权限在收集阶段失败；未发生测试断言失败。
- 在正常本地权限下重跑同一命令：`187 passed, 3 skipped, 30 warnings`。
- `scripts/dev/validate_config_tree.py`：`Config validation passed.`
- `git diff --check`：PASS。
- 两份组会文件保持 untracked / not staged。

## 3. SCRIPT_MOVES / 所有 git mv

源码职责核验后共执行 19 个 `git mv`：

| 旧路径 | Canonical 路径 |
|---|---|
| `scripts/train_mmgcn.py` | `scripts/models/mmgcn/unified/train.py` |
| `scripts/train_simple_mlp.py` | `scripts/models/simple_mlp/train.py` |
| `scripts/evaluate_checkpoint.py` | `scripts/evaluation/unified_checkpoint.py` |
| `scripts/train.py` | `scripts/workflows/train_from_config.py` |
| `scripts/run_experiment_pipeline.py` | `scripts/workflows/run_pipeline.py` |
| `scripts/baselines/train_multidag_cl.py` | `scripts/models/multidag_cl/unified/train.py` |
| `scripts/baselines/evaluate_multidag_cl_checkpoint.py` | `scripts/models/multidag_cl/unified/evaluate.py` |
| `scripts/baselines/train_multidag_cl_smoke.py` | `scripts/models/multidag_cl/unified/smoke.py` |
| `scripts/baselines/new_causal_graph_runtime.py` | `scripts/runtime/causal_graph.py` |
| `scripts/baselines/train_new_causal_graph_baseline.py` | `scripts/workflows/causal_graph/train.py` |
| `scripts/baselines/evaluate_new_causal_graph_checkpoint.py` | `scripts/workflows/causal_graph/evaluate.py` |
| `scripts/baselines/original_merc_runtime.py` | `scripts/runtime/paper_aligned.py` |
| `scripts/baselines/train_original_merc_baseline.py` | `scripts/workflows/paper_aligned/train.py` |
| `scripts/baselines/evaluate_original_merc_checkpoint.py` | `scripts/workflows/paper_aligned/evaluate.py` |
| `scripts/baselines/run_original_merc_pipeline.py` | `scripts/workflows/paper_aligned/run_pipeline.py` |
| `scripts/run_original_merc_reproduction_pipeline.py` | `scripts/workflows/paper_aligned/run_reproduction.py` |
| `scripts/experiments/run_causal8_original8_formal.py` | `scripts/workflows/benchmarks/run_causal8_original8.py` |
| `scripts/experiments/evaluate_missing_modalities.py` | `scripts/workflows/ablations/evaluate_missing_modalities.py` |
| `scripts/experiments/generate_modality_ablation_configs.py` | `scripts/workflows/ablations/generate_modality_configs.py` |

没有创建空 `author_official` scripts，也没有创建无实现内容的目录。

## 4. CANONICAL_ENTRYPOINTS

- 模型入口：`scripts/models/{mmgcn,multidag_cl,simple_mlp}/`
- 跨模型 runtime：`scripts/runtime/{causal_graph,paper_aligned}.py`
- 统一 checkpoint 评估：`scripts/evaluation/unified_checkpoint.py`
- 通用 workflow：`scripts/workflows/{train_from_config,run_pipeline}.py`
- causal workflow：`scripts/workflows/causal_graph/`
- paper-aligned workflow：`scripts/workflows/paper_aligned/`
- 正式 benchmark：`scripts/workflows/benchmarks/run_causal8_original8.py`
- 缺失模态/消融：`scripts/workflows/ablations/`

`paper_aligned` 仍表示项目内论文结构对齐，不是 `author_official`；当前
MultiDAG 不是作者官方完整复现，GS-MCC 仍为 `project_variant`。

## 5. COMPATIBILITY_WRAPPERS

19 个旧路径全部保留薄 wrapper。wrapper 不含训练/评估逻辑、不 import
模型、不定义 argparse 或默认参数。`scripts._load_compat_module` 将导入的旧
模块名透明别名到 canonical module，因此既有 `from old import function` 和
monkeypatch 测试仍操作同一 canonical module；作为 CLI 直接执行时，wrapper
调用 canonical `main()` 并用 `SystemExit` 传播返回码。

两个 runtime wrapper 没有 CLI `main`，只做模块别名；其直接执行仍只加载
runtime 定义并以 0 退出。canonical scripts 不反向 import 旧 wrapper，未形成
循环依赖。没有使用 `runpy` 实现 wrapper；通用 pipeline 内原有的子进程
`runpy.run_path` bootstrap 语义保持不变。

## 6. CLI_COMPATIBILITY

迁移前后对六个关键旧入口执行 `--help`，返回码、字节数和 SHA-256 完全一致：

| 旧 CLI | Return | SHA-256 |
|---|---:|---|
| `scripts/train_mmgcn.py` | 0 | `c22b396aaad97867a95cc5c16bc141078cf51b62a5b381e78c43bc27053696dc` |
| `scripts/baselines/train_multidag_cl.py` | 0 | `b06573396a3c10395ecc7424fe721094f914a0dc6eaaa03b99025c3070cc494d` |
| `scripts/baselines/train_new_causal_graph_baseline.py` | 0 | `97a5a0da1e396da26978eb1f168998e399ff9e9e94fa4ce181d3737e9836ba43` |
| `scripts/baselines/train_original_merc_baseline.py` | 0 | `0d8aa0071a6ad52c98b2d8b1526cbe83cddd47ec05d1aa6891628ff2debce2ad` |
| `scripts/evaluate_checkpoint.py` | 0 | `e87361d4cf33b69eb902e53cc4a309e68fcc20441093474ffc95383e6d7380ba` |
| `scripts/experiments/run_causal8_original8_formal.py` | 0 | `dc70984a99a7dae010ffa566a0742f82fe63b66c0310b95ac342c9ac4140076a` |

新 canonical 入口的 `--help` 也全部返回 0，并保留同一组关键 option。专项测试
还以替换 canonical `main` 的方式证明旧 wrapper 委派执行，不触发正式训练。

## 7. MODEL_IMPORT_AUDIT

AST 审计结果：

| Category | Files | Files with legacy imports | Legacy edges |
|---|---:|---:|---:|
| `canonical_execution` | 33 | 0 | 0 |
| `compatibility_wrapper` | 19 | 0 | 0 |
| `deferred_script` | 47 | 10 | 12 |
| `test` | 23 | 13 | 27 |
| `group_meeting_protected` | 2 | 0 | 0 |

canonical imports 已切换到 `models.registry.causal`、
`models.registry.paper_aligned`、`models.mmgcn.unified`、
`models.multidag_cl.unified`、`models.multidag_cl.paper_aligned`、
`models.gsmcc.project_variant.causal` 和 `models.simple_mlp.model`。122 个非保护
Python 文件全部 AST parse 成功。详细结果见：

- `docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3A.csv`
- `docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3A.md`

## 8. DEFERRED LEGACY IMPORTS

10 个 Phase 3B deferred scripts 保留 12 条 legacy model import，位于 analysis、
debug 和 dev 范围；13 个 tests 保留 27 条，其中 10 条属于专用 model layout
compatibility test。它们没有被机械迁移，也没有与 runtime key
`original_repro_*` 混淆。

## 9. PATH_RESOLUTION

canonical scripts 不再使用易随目录深度失效的固定 `parents[n]` 解析 repo
root，而是沿 `Path(__file__).resolve().parents` 查找同时含 `AGENTS.md` 与
`scripts/` 的明确仓库标记。旧 wrapper 因路径本身固定，只用自身旧层级解析
root。配置与输出仍相对 repo root 解析，没有写入本地或远程绝对路径。

`tests/test_script_layout_compatibility.py` 验证 canonical `PROJECT_ROOT`、所有
活跃 train/evaluate targets 和 16-run entrypoints 均从仓库根解析为真实文件。

通用 pipeline 中既有 `GS_MCC -> scripts/train_gsmcc.py` registry 项没有对应
tracked 文件，也没有任何已交付 YAML 使用 `model.name: GS_MCC`。本阶段没有
可证明等价的实现可迁移，因此保留这个 inactive 的既有占位路由，不将它错误
映射到 causal GS-MCC project variant；所有实际配置可达路由均通过解析门禁。

## 10. SUBPROCESS_COMPATIBILITY

canonical workflow 的硬编码 target 已更新为：

- MMGCN：`scripts/models/mmgcn/unified/train.py`
- SimpleMLP：`scripts/models/simple_mlp/train.py`
- MultiDAG+CL：`scripts/models/multidag_cl/unified/{train,evaluate}.py`
- causal graph：`scripts/workflows/causal_graph/{train,evaluate}.py`
- unified evaluation：`scripts/evaluation/unified_checkpoint.py`
- missing modality：`scripts/workflows/ablations/evaluate_missing_modalities.py`
- formal causal pipeline：`scripts/workflows/run_pipeline.py`
- formal paper-aligned train：`scripts/workflows/paper_aligned/train.py`

子进程 cwd、project-root bootstrap、stdout/stderr 合并、run-info 协议和返回码
处理均未改变。

## 11. CONFIG_COMPATIBILITY

- `configs/` 移动数：0。
- YAML 内容与 schema：未改。
- 旧 YAML 中现有旧 script path 继续由 wrapper 支持。
- `original_repro_*` runtime key、checkpoint metadata、run ID、manifest、metrics
  和 prediction schema 均未改名。
- Config tree validation：PASS。

## 12. DRY_RUN

正式 16-run launcher 原本没有 `--dry-run`，本阶段没有新增会改变其正式行为的
选项，也没有启动正式实验。使用现有 synthetic fixture 调用 `build_run_plan` 和
`prepare_runs`，验证恰好生成 16 个 run、日期冻结、config path 不变、canonical
command target 存在，并且只写 pytest 临时目录。paper-aligned 的既有
`dry_run_only=True` runtime test 继续通过。

## 13. TARGETED_TESTS

- Script layout compatibility：`52 passed`。
- causal/paper-aligned runtime、checkpoint reload、formal launcher、output path 与
  clean-feature 专项：`93 passed, 3 skipped`。
- 受限沙箱内专项首次运行的 31 个 setup error 全部来自系统 pytest 临时目录
  PermissionError；正常本地权限下同一命令通过。

## 14. FULL_PYTEST

- 执行前：`187 passed, 3 skipped, 30 warnings`。
- 执行后：`239 passed, 3 skipped, 30 warnings`。
- 新增测试数：52。
- 完整 pytest 未启动正式训练，也未写入正式 `outputs/`。

## 15. FULL_GATES

- 完整 pytest：PASS。
- Config validation：PASS。
- `git diff --check`：PASS。
- legacy module alias identity：PASS。
- 六组旧 CLI byte-identical `--help`：PASS。
- canonical model import AST gate：PASS（0 edges）。

## 16. PROTECTED_PATHS

- `configs/`：0 个变更，0 个移动。
- `models/`、`datasets/`、`utils/`：0 个变更。
- `data/`、`third_party/`、`outputs/`、`tmp/`：未修改或移动。
- tests 只新增 `tests/test_script_layout_compatibility.py`，并对两个既有 path
  assertion 做 canonical target 的最小更新。
- 组会文件保持 untracked、not staged、unchanged。
- 模型数学、forward、loss、graph、causal mask、curriculum、seed、optimizer、
  batch size、epochs、checkpoint 选择与 state-dict schema 均未改变。

## 17. 未迁移 scripts

以下范围完整留给 Phase 3B：

- `scripts/analyze/`
- `scripts/analysis/`
- `scripts/debug/`
- `scripts/diagnose/`
- `scripts/features/`
- `scripts/prepare/`
- `scripts/inspect/`
- `scripts/maintenance/`
- `scripts/dev/`

本阶段没有移动其中任何 tracked 文件。

## 18. PERSISTENT_CONTEXT

已增量更新：

- `AGENTS.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/project_map.md`

记录了 canonical execution tree、compatibility paths、测试、legacy import
counts、当前 branch/HEAD，以及下一阶段为 Script Phase 3B。没有把计划中的
Phase 3B、config 重构或 author-official reproduction 写成已完成。

## 19. GIT_DIFF 与回滚

本阶段未 commit、未 push。`git mv` 在 index 中记录 19 个 rename；canonical
内容修改、旧 wrapper、package `__init__`、测试和文档留在工作区供审查。最终
交付同时展示 `git diff --stat`、`git diff --cached --stat` 与完整
`git status --short --untracked-files=all`，以避免 untracked wrapper 被普通 diff
遗漏。

回滚时先确认两份组会文件仍为 untracked/not staged。不得使用 `git clean` 或
`git reset --hard`；只对本报告列出的任务路径逐项使用 `git restore --staged`
和 `git restore`，并人工移除本任务新增文件。也可直接保留 HEAD
`1988757fffccfd50532ed4a20d15fedb017eb639` 作为重构前边界。

## 20. NEXT_PHASE / 状态摘要

下一阶段为 Script Phase 3B：迁移 analysis/debug/data/maintenance scripts 并清理
其 12 条 legacy model imports。Phase 3B 完整门禁通过后才规划 `configs/`
重构；旧 script/model wrappers 的删除必须另立任务。

- `SCRIPT_EXECUTION_LAYOUT_REFACTOR_STATUS=PASS`
- `FILES_MOVED=19`
- `COMPATIBILITY_WRAPPERS_CREATED=19`
- `CANONICAL_EXECUTION_LEGACY_MODEL_IMPORTS=0`
- `DEFERRED_SCRIPT_LEGACY_MODEL_IMPORTS=12`
- `CLI_COMPATIBILITY_STATUS=PASS`
- `PATH_RESOLUTION_STATUS=PASS`
- `CONFIGS_MOVED=0`
- `MODELS_CHANGED=NO`
- `FORMAL_TRAINING_STARTED=0`
- `GROUP_MEETING_FILES_STAGED=NO`
- `COMMIT_CREATED=NO`
- `PUSH_PERFORMED=NO`
- `READY_FOR_SCRIPT_PHASE3B=YES`
