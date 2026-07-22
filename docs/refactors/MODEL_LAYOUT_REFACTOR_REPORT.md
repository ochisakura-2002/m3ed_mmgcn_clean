# Model Layout Refactor Report

## 1. 执行前 Git 状态

- Repo root：`E:/MERC/m3ed_mmgcn_clean`
- Source branch：`feat/date-organized-outputs`
- Working branch：`refactor/model-first-layout`
- HEAD：`e09e8864e3cb4fa364c185f6ca1eb342f971283b`
- Tracked working tree：干净。
- 受保护 untracked：`scripts/analyze/export_group_meeting_baseline_report.py`、`tests/analyze/test_group_meeting_baseline_report.py`；均未修改、未 staged。
- `docs/PROJECT_CONTEXT.md` 在执行前不存在于 HEAD；本阶段按任务要求首次建立为唯一持久项目上下文。

## 2. 执行前门禁

- 完整 pytest：`176 passed, 3 skipped, 30 warnings`。
- Config validation：`Config validation passed.`
- `git diff --check`：PASS。

受限沙箱中的首次完整 pytest 因无法读取既有 `tmp/pytest_*` 目录而在收集阶段报 PermissionError；同一命令在具有正常本地文件权限的环境中通过。未删除或修改 `tmp/`。

## 3. Git moves

所有实现均用 `git mv` 迁移，没有复制实现后再删除。

| 旧路径 | 新 canonical 路径 | 文件数 | 模型/职责 | Implementation type |
|---|---|---:|---|---|
| `models/baselines/causal_graph_common/` | `models/common/causal_graph/` | 4 | 公共 causal 图工具 | common |
| `models/baselines/causal_baseline_registry.py` | `models/registry/causal.py` | 1 | causal registry | registry |
| `models/baselines/original_repro/common.py` | `models/common/paper_aligned.py` | 1 | 论文对齐公共工具 | paper_aligned common |
| `models/baselines/original_repro/registry.py` | `models/registry/paper_aligned.py` | 1 | 论文对齐 registry | registry |
| `models/baselines/mmgcn/` | `models/mmgcn/unified/` | 3 | MMGCN | unified |
| `models/baselines/original_repro/mmgcn/` | `models/mmgcn/paper_aligned/` | 2 | MMGCN | paper_aligned |
| `models/baselines/multidag_cl/` | `models/multidag_cl/unified/` | 2 | MultiDAG+CL | unified |
| `models/baselines/original_repro/multidag_cl/` | `models/multidag_cl/paper_aligned/` | 3 | MultiDAG+CL | paper_aligned |
| `models/baselines/dialoguegcn/` | `models/dialoguegcn/unified/` | 4 | DialogueGCN | unified |
| `models/baselines/original_repro/dialoguegcn/` | `models/dialoguegcn/paper_aligned/` | 2 | DialogueGCN | paper_aligned |
| `models/baselines/gsmcc/` | `models/gsmcc/project_variant/causal/` | 4 | GS-MCC Project Variant | project_variant, causal_context |
| `models/baselines/original_repro/gsmcc/` | `models/gsmcc/project_variant/full_context/` | 2 | GS-MCC Project Variant | project_variant, full_context |
| `models/baselines/simple_mlp.py` | `models/simple_mlp/model.py` | 1 | SimpleMLP | sanity baseline |
| `models/baselines/sdt/` | `models/experimental/sdt/` | 2 | SDT | experimental |

合计移动 32 个已跟踪文件。MMGCN 的 `mm_gcn.py` 未在本阶段改名，以降低整模型 pickle/module-path 风险。

## 4. Canonical model paths

- `models/common/causal_graph/`
- `models/common/paper_aligned.py`
- `models/registry/causal.py`
- `models/registry/paper_aligned.py`
- `models/mmgcn/{unified,paper_aligned}/`
- `models/multidag_cl/{unified,paper_aligned}/`
- `models/dialoguegcn/{unified,paper_aligned}/`
- `models/gsmcc/project_variant/{causal,full_context}/`
- `models/simple_mlp/model.py`
- `models/experimental/sdt/`

没有创建空 `author_official` 目录或 `.gitkeep`。作者官方 MultiDAG+CL 和 GS-MCC 仍未接入。

## 5. Compatibility wrappers

以下 32 个旧文件位置新建为薄 re-export wrapper；另将既有 `models/baselines/original_repro/__init__.py` 转为 canonical re-export。所有符号直接引用 canonical 对象，不包含 forward、loss 或第二套 registry。

| 旧 wrapper | Canonical target | 主要导出符号 |
|---|---|---|
| `models/baselines/causal_graph_common/__init__.py` | `models.common.causal_graph` | 8 个图/掩码工具 |
| `models/baselines/causal_graph_common/graph_builders.py` | `models.common.causal_graph.graph_builders` | adjacency 构建、normalize、causal assert |
| `models/baselines/causal_graph_common/masked_ops.py` | `models.common.causal_graph.masked_ops` | `masked_softmax` |
| `models/baselines/causal_graph_common/relation_utils.py` | `models.common.causal_graph.relation_utils` | `build_speaker_pair_relation_ids` |
| `models/baselines/causal_baseline_registry.py` | `models.registry.causal` | 两个名称常量及 4 个 registry 函数 |
| `models/baselines/mmgcn/__init__.py` | `models.mmgcn.unified` | `M3EDMMGCN` |
| `models/baselines/mmgcn/mm_gcn.py` | `models.mmgcn.unified.mm_gcn` | `GraphConvolution`, `GCNIIBackbone`, `M3EDMMGCN` |
| `models/baselines/mmgcn/dense_graph.py` | `models.mmgcn.unified.dense_graph` | 11 个 dense graph 工具 |
| `models/baselines/multidag_cl/__init__.py` | `models.multidag_cl.unified` | encoder、model、两个 graph helper |
| `models/baselines/multidag_cl/multidag_cl_model.py` | `models.multidag_cl.unified.multidag_cl_model` | encoder、model、两个 graph helper |
| `models/baselines/dialoguegcn/__init__.py` | `models.dialoguegcn.unified` | config、model |
| `models/baselines/dialoguegcn/causal_dialoguegcn_graph.py` | `models.dialoguegcn.unified.causal_dialoguegcn_graph` | graph builder、edge attention |
| `models/baselines/dialoguegcn/causal_dialoguegcn_model.py` | `models.dialoguegcn.unified.causal_dialoguegcn_model` | config、model |
| `models/baselines/dialoguegcn/relational_graph_conv.py` | `models.dialoguegcn.unified.relational_graph_conv` | `RelationalGraphConv` |
| `models/baselines/gsmcc/__init__.py` | `models.gsmcc.project_variant.causal` | config、model、loss |
| `models/baselines/gsmcc/causal_gsmcc_losses.py` | `models.gsmcc.project_variant.causal.causal_gsmcc_losses` | `compute_causal_gsmcc_loss` |
| `models/baselines/gsmcc/causal_gsmcc_model.py` | `models.gsmcc.project_variant.causal.causal_gsmcc_model` | config、model |
| `models/baselines/gsmcc/causal_spectral_ops.py` | `models.gsmcc.project_variant.causal.causal_spectral_ops` | filter、layer |
| `models/baselines/original_repro/common.py` | `models.common.paper_aligned` | 10 个公共常量/工具 |
| `models/baselines/original_repro/registry.py` | `models.registry.paper_aligned` | registry、keys、provenance、4 个函数 |
| `models/baselines/original_repro/mmgcn/__init__.py` | `models.mmgcn.paper_aligned` | `OriginalReproMMGCN` |
| `models/baselines/original_repro/mmgcn/model.py` | `models.mmgcn.paper_aligned.model` | `OriginalReproMMGCN` |
| `models/baselines/original_repro/multidag_cl/__init__.py` | `models.multidag_cl.paper_aligned` | model、adjacency、3 个 Curriculum Learning helper |
| `models/baselines/original_repro/multidag_cl/curriculum.py` | `models.multidag_cl.paper_aligned.curriculum` | 3 个 Curriculum Learning helper |
| `models/baselines/original_repro/multidag_cl/model.py` | `models.multidag_cl.paper_aligned.model` | model、adjacency |
| `models/baselines/original_repro/dialoguegcn/__init__.py` | `models.dialoguegcn.paper_aligned` | network、model、两个 graph helper |
| `models/baselines/original_repro/dialoguegcn/model.py` | `models.dialoguegcn.paper_aligned.model` | network、model、两个 graph helper |
| `models/baselines/original_repro/gsmcc/__init__.py` | `models.gsmcc.project_variant.full_context` | `ProjectPaperOrientedGSMCC` |
| `models/baselines/original_repro/gsmcc/model.py` | `models.gsmcc.project_variant.full_context.model` | model、Fourier operator、3 个 helper/loss |
| `models/baselines/simple_mlp.py` | `models.simple_mlp.model` | `M3EDConcatMLP` |
| `models/baselines/sdt/__init__.py` | `models.experimental.sdt` | `SDTBaseline` |
| `models/baselines/sdt/sdt_model.py` | `models.experimental.sdt.sdt_model` | `SDTBaseline` |
| `models/baselines/original_repro/__init__.py`（既有文件转为 wrapper） | canonical paper-aligned packages + `models.registry.paper_aligned` | 4 个模型类及 registry API |

## 6. Internal import updates

只修改了因目录变化必须调整的 import：

- causal graph consumer 改为 `models.common.causal_graph`。
- causal registry 改为 `models.dialoguegcn.unified` 与 `models.gsmcc.project_variant.causal`。
- paper-aligned registry 改为各模型 canonical package 与 `models.common.paper_aligned`。
- MMGCN paper-aligned 改为 `models.mmgcn.unified` 与 `models.common.paper_aligned`。
- MultiDAG+CL、DialogueGCN、GS-MCC full-context 的共享工具改为 `models.common.paper_aligned`。
- canonical `models/` 中 `models.baselines` 静态引用为 0，没有 wrapper 循环依赖。

## 7. Import identity validation

`tests/test_model_layout_compatibility.py` 验证以下旧/新符号使用 `is` 相等：

- MMGCN unified 与 paper_aligned。
- MultiDAG+CL unified 与 paper_aligned。
- DialogueGCN unified 与 paper_aligned。
- GS-MCC causal 与 full-context project variants。
- causal registry build function。
- paper-aligned registry build function。

专项结果：`11 passed`（10 组 identity + 1 个 state-dict strict-load gate）。

## 8. Behavior and checkpoint gates

模型相关 targeted suite 包含既有 model contracts、numerical validity、mechanisms、causal invariance、synthetic training、checkpoint reload 和新增兼容测试：

- Import / instantiate：PASS。
- Synthetic forward：PASS。
- Loss finite：PASS。
- Backward：PASS。
- Gradients finite：PASS。
- Causal invariance / future-edge restrictions：PASS。
- Targeted pytest：`93 passed, 3 skipped`。
- MMGCN 旧/新构造的 state-dict key 顺序一致；同一 state dict 在 canonical 实例上 `strict=True` 加载通过。
- 既有 causal checkpoint reload 与 original MERC runtime checkpoint tests：PASS。

没有读取或修改 `outputs/` 中的正式 checkpoint，也没有启动正式训练。

## 9. Full gates

- 迁移后完整 pytest：`187 passed, 3 skipped, 30 warnings`。
- 迁移后 config validation：`Config validation passed.`
- 最终工作区与 staged `git diff --check`：PASS。

## 10. Protected paths

- `configs/`：0 个变更，0 个移动。
- 生产 `scripts/`：0 个变更，0 个移动。
- 既有 `tests/`：0 个修改或移动；仅新增 `tests/test_model_layout_compatibility.py`。
- `datasets/`、`utils/`、`data/`、`third_party/`、`outputs/`、`tmp/`：0 个变更。
- 组会文件保持 untracked / not staged；没有修改 `.gitignore`。

## 11. 未迁移内容与下一阶段

未迁移：`configs/`、生产 `scripts/`、既有 tests、datasets、utils、数据与输出目录。现有生产脚本仍通过 `models.baselines...` wrapper 运行。

下一阶段为 `scripts` 目录模型专属入口与公共 runtime/workflow 重构。建议顺序：

1. 将训练、评估、debug 和 analysis 脚本的模型 import 切到 canonical registry/package。
2. 固定 checkpoint/config schema 后抽取共享 model construction/runtime。
3. 抽取 pipeline workflow，保持现有 CLI 入口为兼容 wrapper。
4. 完整门禁通过后，另立任务评估旧 model wrapper 的删除时点。

主要风险是 whole-model pickle module path、脚本内隐式旧 registry 依赖、旧名称 `original_repro_*` 的配置兼容；不得在脚本迁移中顺便改模型数学或 configs schema。

## 12. Legacy import verification correction

此前人工收集的 `NO_MATCHES` 结果不可靠，不能作为旧 import 已清零的证据。根因是本地命令环境或搜索变量处理问题，而不是仓库中确实不存在引用。

本次使用 Git tracked 文件清单加 Python AST 重新解析 `scripts/` 与 `tests/`，并用以下独立文本搜索交叉核验：

```text
git grep -n -E "models\.baselines|original_repro|causal_baseline_registry" -- "*.py"
```

AST 共成功解析 88 个 tracked Python 文件，0 个解析错误。按一个 AST import 语句指向一个模块计一条依赖边，真实结果为：

- `scripts/`：21 个文件，26 条旧 import 边。
- `tests/`：13 个文件，27 条旧 import 边。
- 合计：34 个文件，53 条旧 import 边。

详细结果见 `docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_MODEL_REFACTOR.csv`，方法与迁移边界见同目录 Markdown 摘要。`git grep` 覆盖了所有 AST import 行，同时还命中了配置 key、函数名、dataset 模块名和 wrapper 内文本，因此不能直接把全部文本命中计为 import。

生产 scripts 尚未迁移，继续依赖兼容 wrapper 是模型目录阶段的预期行为。另对 `models/baselines/` 之外 42 个 canonical model Python 文件执行 AST 核验，反向依赖旧 wrapper 的 import 为 0。因此本次更正不影响模型目录重构 `PASS` 结论；这些生产脚本和非兼容性测试引用由下一阶段 scripts 重构迁移，专用 compatibility test 保留到 wrapper 删除门禁。

## 13. 回滚方式

本任务没有 commit。回滚前先再次核对组会 untracked 文件并保持它们不被操作；不得使用 `git clean` 或 `git reset --hard`。可仅对本任务列出的 tracked paths 使用 `git restore --staged` / `git restore`，再逐项人工确认并移除本任务新建的 wrapper、package、测试和文档。也可保留源分支 `feat/date-organized-outputs` 不变，将工作分支作为隔离回滚边界。

## 14. 状态摘要

- `MODEL_LAYOUT_REFACTOR_STATUS=PASS`
- `FILES_MOVED=32`
- `COMPATIBILITY_WRAPPERS_CREATED=32`
- `MODEL_BEHAVIOR_CHANGED=NO`
- `STATE_DICT_SCHEMA_CHANGED=NO`
- `CONFIGS_MOVED=0`
- `PRODUCTION_SCRIPTS_MOVED=0`
- `FORMAL_TRAINING_STARTED=0`
- `GROUP_MEETING_FILES_STAGED=NO`
- `COMMIT_CREATED=NO`
- `PUSH_PERFORMED=NO`
- `READY_FOR_SCRIPT_REFACTOR=YES`
- `SCRIPT_LAYOUT_REFACTOR_STATUS=NOT_STARTED`
- `CONFIG_LAYOUT_REFACTOR_STATUS=NOT_STARTED`
- `OFFICIAL_MULTIDAG_REPRODUCTION=NOT_STARTED`
- `OFFICIAL_GSMCC_REPRODUCTION=NOT_STARTED`
- `FINAL_BASELINE_SELECTED=NO`
- `LEGACY_IMPORT_AST_STATUS=PASS`
- `LEGACY_IMPORT_SCRIPT_FILES=21`
- `LEGACY_IMPORT_SCRIPT_EDGES=26`
- `LEGACY_IMPORT_TEST_FILES=13`
- `LEGACY_IMPORT_TEST_EDGES=27`
