# MERC 持久项目上下文

本文档是跨 Codex 对话的长期项目记忆。新任务仍先读取 `AGENTS.md`，再读取本文档，并按任务类型补读 `docs/project_map.md` 与相关规范。

## 项目目标与当前阶段

当前研究任务是多模态对话情感识别（MERC），长期目标与机器人实时交互中的 causal MERC 有关。项目当前不是围绕单一 `M3ED + MMGCN` 组合开发，而是在多模型实验平台上建立作者官方复现、统一规程实现、causalization 和最终 baseline 决策的可审计链路。

当前已经完成三类实验基础：

1. 统一 Original / full-context 轨道。
2. 统一 causal 轨道。
3. Legacy 与 Clean 特征轨道。

近期 baseline 决策和正式实验主要围绕 IEMOCAP；M3ED 仍是长期数据资产和机器人代理场景的扩展方向。MMGCN Legacy 复现较稳定，是经典锚点，但尚未确定为最终论文 baseline。

`MODEL_LAYOUT_REFACTOR=COMPLETED`

`SCRIPT_EXECUTION_LAYOUT_REFACTOR=COMPLETED`

`SCRIPT_ANALYSIS_LAYOUT_REFACTOR=COMPLETED`

`SCRIPT_SUPPORT_LAYOUT_REFACTOR=NOT_STARTED`

`CONFIG_LAYOUT_REFACTOR=NOT_STARTED`

`OFFICIAL_MULTIDAG_REPRODUCTION=NOT_STARTED`

`OFFICIAL_GSMCC_REPRODUCTION=NOT_STARTED`

`FINAL_BASELINE_SELECTED=NO`

模型实现目录的模型优先重构、Phase 3A 生产执行脚本重构和 Phase 3B1 analysis layout 已经完成。模型训练/评估入口、跨模型 runtime、workflow、正式 benchmark launcher、模态缺失 workflow 与结果分析实现均已有 canonical script tree；旧执行和旧分析路径保留 compatibility wrapper。debug、diagnose、feature/data、inspect、maintenance 与 dev/support scripts 尚未完成 Phase 3B2，`configs/` 也尚未迁移。下一阶段是 Script Phase 3B2，之后才重构 `configs/`；全部门禁完成前不部署作者官方 MultiDAG+CL 或 GS-MCC。

## 模型路线与命名边界

- `unified`：项目统一输入、训练和评价接口下的实现。
- `paper_aligned`：项目内依据论文结构实现，不代表作者官方源码。
- `project_variant`：借用论文思想但与作者官方实现存在明确差异。
- `experimental`：尚未进入正式 baseline 池的实验模型。
- 当前 GS-MCC 两套实现的规范显示名均为 `GS-MCC Project Variant`。
- MultiDAG+CL 中的 CL 表示 Curriculum Learning。
- 上下文模式使用 `full_context` 与 `causal_context`；causal 信息边界不等于 `author_official`。
- 作者官方 MultiDAG+CL 与 GS-MCC 尚未接入，不得创建空的 `author_official` 目录暗示已有实现。

## 当前模型与 baseline 状态

当前实验平台包括：

1. MMGCN。
2. MultiDAG-inspired + Curriculum Learning。
3. DialogueGCN。
4. GS-MCC Project Variant。
5. Causal-MMGCN。
6. Causal-MultiDAG-inspired。

当前项目 MultiDAG 是统一或 `paper_aligned` 实现，不是作者官方完整复现。GS-MCC 的 causal 与 full-context 两套实现都是 `project_variant`，也不是作者官方完整复现。MMGCN Legacy 只作为复现较稳定的经典锚点；最终 baseline 必须等官方复现、统一协议表现、causalization 降幅、复现可信度和可修改性完成量化审计后决定。

## Canonical model paths

- 公共 causal 图：`models/common/causal_graph/`
- paper-aligned 公共工具：`models/common/paper_aligned.py`
- causal registry：`models/registry/causal.py`
- paper-aligned registry：`models/registry/paper_aligned.py`
- MMGCN unified：`models/mmgcn/unified/`
- MMGCN paper-aligned：`models/mmgcn/paper_aligned/`
- MultiDAG+CL unified：`models/multidag_cl/unified/`
- MultiDAG+CL paper-aligned：`models/multidag_cl/paper_aligned/`
- DialogueGCN unified：`models/dialoguegcn/unified/`
- DialogueGCN paper-aligned：`models/dialoguegcn/paper_aligned/`
- GS-MCC causal project variant：`models/gsmcc/project_variant/causal/`
- GS-MCC full-context project variant：`models/gsmcc/project_variant/full_context/`
- SimpleMLP：`models/simple_mlp/model.py`
- SDT experimental：`models/experimental/sdt/`

MMGCN unified 的实现文件继续命名为 `mm_gcn.py`，本阶段未改为 `model.py`，以降低整模型 pickle/module-path 兼容风险。

## Compatibility paths

以下旧路径至少保留一个迁移周期，只能作为薄 re-export wrapper：

- `models/baselines/causal_graph_common/`
- `models/baselines/causal_baseline_registry.py`
- `models/baselines/{mmgcn,multidag_cl,dialoguegcn,gsmcc}/`
- `models/baselines/original_repro/` 及其模型子包、`common.py`、`registry.py`
- `models/baselines/simple_mlp.py`
- `models/baselines/sdt/`

canonical execution scripts 已直接使用模型 canonical path，不再依赖 `models.baselines...`。既有模型兼容测试与部分 Phase 3B deferred scripts 暂时保留旧 model wrapper import；新代码必须直接使用 canonical path。

## Model registries

- 新 causal baseline 构造与名称规范化：`models.registry.causal`
- paper-aligned / project-variant MERC 模型构造与 provenance：`models.registry.paper_aligned`
- 旧 registry 模块只重导出同一函数和对象，不维护第二套 registry。

## 数据、特征与正式实验

- 近期正式实验与 baseline 决策主要使用 IEMOCAP。
- M3ED 是长期数据资产，并服务于后续机器人代理与 causal MERC 场景扩展。
- 特征比较同时保留 Legacy 与 Clean 轨道，不得跨轨道混写结果。
- 统一 batch 形状与模型插入约束见 `docs/module_implementation_spec.md`。
- Original MERC 三轨协议、缺失模态与从头训练消融边界见 `docs/experiment_protocol.md`。
- validation 选择 best checkpoint，test 不参与 checkpoint、epoch、超参数或模型选择。
- 正式结果至少报告 Weighted-F1、Macro-F1、UAR、Accuracy。

当前正式实验批次为 `causal8_original8_formal_20260722_012904`：包含 8 个 causal run 和 8 个 Original run，共 16 个正式实验；全部 `PASS`，数值状态均为 `FINITE`。该批次证明统一运行链路已完成并通过数值门禁，不等于作者官方复现完成，也不等于最终 baseline 已选定。

## Canonical runtime entries 与输出规则

Phase 3A canonical execution entries 为：

- MMGCN 训练：`scripts/models/mmgcn/unified/train.py`
- MultiDAG+CL 训练/评估/smoke：`scripts/models/multidag_cl/unified/{train,evaluate,smoke}.py`
- SimpleMLP 训练：`scripts/models/simple_mlp/train.py`
- 统一 checkpoint 评估：`scripts/evaluation/unified_checkpoint.py`
- causal 与 paper-aligned runtime：`scripts/runtime/{causal_graph,paper_aligned}.py`
- 通用 pipeline：`scripts/workflows/run_pipeline.py`
- causal workflow：`scripts/workflows/causal_graph/{train,evaluate}.py`
- paper-aligned workflow：`scripts/workflows/paper_aligned/`
- 正式 16-run benchmark：`scripts/workflows/benchmarks/run_causal8_original8.py`
- 模态缺失/消融 workflow：`scripts/workflows/ablations/`

Phase 3B1 canonical analysis paths 为：

- 通用结果表格、绘图与论文产物：`scripts/analysis/common/`
- causal 模型与 benchmark 分析：`scripts/analysis/causal/`
- Original MERC / paper-aligned 分析：`scripts/analysis/paper_aligned/`
- 模型专属分析：`scripts/analysis/models/<model>/`

迁移前的 tracked `scripts/analyze/` 分析入口继续作为 module-alias compatibility wrapper；`scripts/analysis/analyze_original_merc_results.py` 与 `scripts/analyze/export_original_merc_reproduction_report.py` 也直接委派最终 canonical Original-MERC module。新代码不得 import 这些旧 analysis wrapper。

迁移前路径继续作为 compatibility wrapper，可保持既有 CLI 与尚未迁移的 YAML 可运行；新代码不得 import 旧 script wrapper。

正式实验输出按启动日期写入 `outputs/<YYYYMMDD>/`；本地 smoke 使用 `tmp/`。显式 checkpoint/run/output 路径优先，旧输出只读兼容，不移动、不改名。不要提交数据、feature pkl、checkpoint、cache 或大体积输出。

## 已完成工作

- 从源分支 `feat/date-organized-outputs` 的 `e09e8864e3cb4fa364c185f6ca1eb342f971283b` 创建工作分支 `refactor/model-first-layout`。
- 通过 14 组 `git mv` 将 32 个已跟踪模型文件迁移到 canonical 路径。
- 在旧路径创建 32 个薄 wrapper，并将 `models/baselines/original_repro/__init__.py` 转为兼容入口。
- canonical 实现的内部 import 已切换到 `models.common`、`models.registry` 和各模型 canonical package。
- 新增 `tests/test_model_layout_compatibility.py`，验证 10 组核心新旧符号身份及 MMGCN strict state-dict 加载。
- 模型数学、forward 契约、默认超参数和 state-dict schema 未改变。
- Phase 3A 使用 19 个 `git mv` 建立 canonical execution tree，并在全部旧路径创建 19 个透明 module-alias wrapper。
- canonical scripts 的模型 import 已迁移到 `models.registry`、`models.common` 与模型 canonical package；canonical `models.baselines` import 为 0。
- pipeline、missing-modality workflow 与正式 16-run launcher 的内部 subprocess target 已指向 canonical scripts；`configs/` 内容与 runtime/checkpoint key 未改。
- 新增 `tests/test_script_layout_compatibility.py` 与 Phase 3A legacy import CSV/Markdown 审计。
- Phase 3B1 使用 13 个 `git mv` 将真实分析实现迁入 `scripts/analysis/`，并在 13 个旧 tracked 路径建立薄 module-alias wrapper；另将 2 个既有分析别名直接指向最终 canonical module。
- canonical analysis 的旧 model/script/analysis wrapper import 为 0；5 个 feature/data 或 diagnose 文件保留原位，等待 Phase 3B2。
- 新增 `tests/test_analysis_layout_compatibility.py`，覆盖新旧 import identity、CLI help、repo-root 解析和 synthetic missing-modality 输出等价。

## 最近门禁结果

执行日期：2026-07-23，本地 Windows，conda 环境 `m3ed_mmgcn`。

- 迁移前完整 pytest：`176 passed, 3 skipped`。
- 迁移前 config validation：通过。
- 新旧 import 与 state-dict 专项：`11 passed`。
- 模型相关 synthetic / numerical / checkpoint 测试：`93 passed, 3 skipped`。
- 迁移后完整 pytest：`187 passed, 3 skipped`。
- 迁移后 config validation：通过。
- canonical `models/` 内 `models.baselines` 静态引用：0。
- 最终工作区与 staged `git diff --check`：通过。
- Script Phase 3A 执行前完整 pytest：`187 passed, 3 skipped`。
- Script layout compatibility：`52 passed`。
- Script runtime/checkpoint/formal-launcher/output-path 专项：`93 passed, 3 skipped`。
- Script Phase 3A 完整 pytest：`239 passed, 3 skipped`。
- canonical execution legacy model imports：0；compatibility wrapper legacy model imports：0。
- deferred scripts：10 个文件、12 条 legacy model import；tests：13 个文件、27 条 legacy model import。
- Phase 3A 基于分支 `refactor/model-first-layout`、HEAD `1988757fffccfd50532ed4a20d15fedb017eb639`；本阶段未 commit、未 push。
- Script Phase 3B1 执行前完整 pytest：`239 passed, 3 skipped`；config validation 与 diff check 通过。
- Analysis layout compatibility：`31 passed`；既有 analysis consumer 专项：`41 passed`。
- Script Phase 3B1 执行后完整 pytest：`270 passed, 3 skipped`；config validation 与 diff check 通过。
- canonical analysis legacy imports：0；compatibility wrappers：15；deferred analysis files：5；tests 中 legacy analysis import：5 个文件、6 条边；旧 CLI path 文本引用：139。
- Phase 3B1 基于分支 `refactor/model-first-layout`、HEAD `46af77a64b1a7347c46904a9710692a31576bf2f`；本阶段未 commit、未 push。

完整 pytest 在受限沙箱内会因既有 `tmp/pytest_*` 与系统 pytest 临时目录权限而无法收集；在具有正常本地文件权限的同一环境中通过。这不是模型断言失败，且本任务未删除或修改这些目录。

## 已知问题与保护边界

- Phase 3A production execution 与 Phase 3B1 analysis scripts 已迁移；debug、diagnose、feature/data、inspect、maintenance 与 dev/support scripts 尚未迁移，是 Phase 3B2 范围。
- 旧 script wrapper 与旧 model wrapper 至少保留一个迁移周期；删除前必须完成 Phase 3B、configs 迁移和独立删除门禁。
- `scripts/analyze/export_group_meeting_baseline_report.py` 和 `tests/analyze/test_group_meeting_baseline_report.py` 是本地 untracked 组会文件，必须保持 not staged，不修改、不上传。
- 不扫描或修改 `outputs/`、`data/`、`third_party/`、`tmp/`；不启动本地正式训练。

已知论文叙事风险：

1. 不得将 `paper_aligned` 写成 `author_official`，也不得把当前 MultiDAG 或 GS-MCC 项目实现写成作者官方完整复现。
2. 配置中的 `original_repro_*` 等 key 是运行协议兼容标识，不足以证明实现 provenance。
3. 不得因 MMGCN Legacy 复现稳定就提前声明其为最终论文 baseline。
4. full-context 与 causal 比较必须控制除未来信息相关内部结构以外的变量；test 结果不得用于模型选择。
5. 当前统一规程结果、未来作者官方结果及其 causalized 结果必须分轨报告，不得混写成同一复现结论。

## 下一阶段工程顺序

模型目录重构、Script Phase 3A 和 Script Phase 3B1 已完成。下一阶段为 Script Phase 3B2，只处理 debug、diagnose、feature/data、inspect、maintenance 与 dev/support scripts 及其 legacy imports；通过完整门禁后再重构 `configs/`。目录重构全部完成并通过门禁后，才部署作者官方 MultiDAG+CL 和 GS-MCC。

Phase 3B2 不得删除 Phase 3A/3B1 wrapper、改变 runtime/checkpoint/config schema，或把 `paper_aligned` 误写为 `author_official`。Phase 3B1 结果见 `docs/refactors/SCRIPT_ANALYSIS_LAYOUT_REFACTOR_REPORT.md` 与 `docs/refactors/LEGACY_ANALYSIS_IMPORTS_AFTER_PHASE3B1.*`。

## 下一阶段研究路线

1. 使用作者官方源码、官方特征和作者规程复现 MultiDAG+CL。
2. 使用作者官方源码、官方特征和作者规程复现 GS-MCC。
3. 在官方 full-context 实现基础上，仅 causal 化未来信息相关的内部结构，保留其他变量不变。
4. 全量审计作者官方 full-context、作者官方 causalized、当前统一 full-context、当前统一 causal 四条轨道。
5. 根据 causal 化下降、统一协议表现、复现可信度和可修改性决定最终 baseline。
6. 再决定论文主线是通用 MERC 还是 causal MERC，并进入模块实验。
