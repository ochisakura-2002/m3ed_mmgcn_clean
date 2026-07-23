# Codex 项目上下文入口

本仓库是一个多模型多模态对话情感识别（MERC）实验平台，长期目标与机器人实时交互中的 causal MERC 有关。当前阶段正在建立作者官方复现、统一实验规程、causalization 和最终 baseline 决策；`MMGCN` 是经典锚点，但尚未确定为最终论文 baseline。后续 Codex 对话进入本仓库时，先按本文档建立上下文，不要默认全量扫描仓库。

## 任务开始规则

1. 任何任务开始前，先读取 `docs/project_map.md`。
2. 任务涉及 Git、本地/远程同步、实验执行时，同时读取 `docs/codex_workflow.md` 和 `docs/git_workflow.md`。
3. 任务涉及实验设计、缺失模态、模态消融、结果分析时，同时读取 `docs/experiment_protocol.md`。
4. 任务涉及模型结构或 `MMGCN` 模块修改时，同时读取 `docs/module_implementation_spec.md`。
5. 不要全量扫描仓库，除非 `docs/project_map.md` 不足以定位任务文件。
6. 修改前列出计划查看和修改的文件。
7. 修改后显示 `git diff --stat`。
8. 除非用户明确要求，不要执行 `git commit`。

## 路径与数据规则

1. 不要硬编码本地或远程绝对路径。
2. 配置、脚本和文档中优先使用相对路径。
3. 不要随意修改 `third_party/`、checkpoints、原始数据、大体积 `outputs`、cache 文件。
4. 不要修改 `data/`、`checkpoint/`、`checkpoints/`、`logs/`、`outputs/`，除非用户明确要求并说明用途。
5. 本地 Windows 主要用于代码编辑、轻量 smoke test、CSV 分析、图表生成；不要在本地跑正式训练。
6. 正式训练、多 seed 实验和 checkpoint 评估主要在远程 V100 服务器上执行。

## 修改原则

1. 修改必须小而可审查。
2. 每次优先只解决一个问题。
3. 新模块必须提供 YAML 开关，便于消融。
4. 新模块必须保留 baseline-equivalent 路径，也就是关闭模块后行为应尽量等价于原 baseline。
5. 优先添加 smoke test 或最小可运行检查，再讨论模型效果优化。
6. 不要直接安装依赖；如需依赖，建议加入 `requirements.txt` 或 `environment.yml`，但当前 checkout 未找到这两个文件。

## 当前模型目录与阶段

1. 模型实现已迁移到模型优先的 canonical 路径：`models/<model>/<lineage>/`；公共代码位于 `models/common/`，registry 位于 `models/registry/`。
2. `models/baselines/` 与 `models/baselines/original_repro/` 只保留一个迁移周期的兼容 re-export wrapper；新代码禁止 import 这些旧路径。
3. `paper_aligned` 表示项目内按论文结构实现，不等于 `author_official`；当前 MultiDAG+CL 项目实现不是作者官方完整复现，GS-MCC 两套实现均为 `project_variant`。
4. `MMGCN` 是当前经典复现锚点，不得表述为已经选定的最终 baseline；SDT 位于 `models/experimental/sdt/`，仍不进入正式 baseline 排名。
5. 目录重构不得改变模型数学行为、forward 契约或 `state_dict` schema。
6. 模型目录与 scripts layout 重构均已完成；Config Phase 4A audit 与 correction 已完成，183 个 tracked YAML 均在分类和迁移计划中；9 个 smoke flag 已按内容修正，当前剩余 2 行 manual review 和 1 个 candidate collision group，`configs/` 尚未移动或修改。
7. 作者官方 MultiDAG+CL 与 GS-MCC 必须等全部 config migration batch 与门禁完成后再部署；当前不要提前移动 `configs/`。
8. 不要为定位模型代码扫描 `outputs/`、`data/`、`third_party/` 或 `tmp/`。

## 当前脚本目录与阶段

1. 生产执行脚本的 canonical 路径位于 `scripts/models/`、`scripts/runtime/`、`scripts/evaluation/` 和 `scripts/workflows/`。
2. 迁移前的训练、评估、pipeline 与实验 launcher 路径只保留 compatibility wrapper；新代码禁止依赖这些旧 wrapper。
3. `configs/` 尚未迁移，旧 YAML 中的 script path 继续由 compatibility wrapper 支持。
4. Phase 3B1 analysis layout 已完成；canonical 分析实现位于 `scripts/analysis/{common,causal,paper_aligned,models}/`。
5. 被迁移的 tracked `scripts/analyze/` 入口只保留 compatibility wrapper；新代码禁止 import 这些旧 analysis wrapper。
6. Phase 3B2 support layout 已完成；canonical 支持脚本位于 `scripts/data/{build,prepare,inspect}/`、`scripts/diagnostics/{data,models,experiments}/`、`scripts/maintenance/` 与 `scripts/dev/`。
7. `data/build` 生成数据或特征资产，`data/prepare` 做训练前数据准备，`data/inspect` 只读查看数据；`diagnostics` 判定数据、模型或实验异常；`maintenance` 可更新仓库/汇总资产，`dev` 只放静态验证与开发门禁。
8. 迁移前的 debug、diagnose、features、prepare、inspect 及 deferred analyze/support 路径只保留 compatibility wrapper；新代码禁止 import 这些旧 support wrapper。
9. 所有 tracked 生产 scripts 已切换到 canonical model import；旧 model import 只允许保留在专用兼容性测试中。
10. `configs/` 尚未迁移；Config Phase 4A audit/correction 已完成，但 `CONFIG_LAYOUT_REFACTOR` 与 Config Batch 1--7 均未开始。
11. `scripts/analyze/export_group_meeting_baseline_report.py` 与 `tests/analyze/test_group_meeting_baseline_report.py` 继续作为本地 untracked 组会文件保护，不读取其内容作为生产依据、不修改、不 staged。
12. 配置 provenance 必须依据 YAML 内容、registry 与真实 consumer，不得根据旧文件名中的 `original`、`official` 等字样猜测。
13. YAML 迁移不保留双份真相，也不创建旧 YAML wrapper；发生候选路径碰撞时必须先人工消解。
14. 后续配置迁移严格按 `docs/refactors/CONFIG_MIGRATION_PLAN_PHASE4A.csv` 的 batch 执行，每次任务只允许一个 batch。
15. 下一阶段仅为 Config Batch 1，且只能在 Phase 4A correction 门禁通过后开始；剩余 1 个 candidate collision group 属于 Batch 7，不属于 Batch 1，必须在 Batch 7 前决定合并或通过独立任务补足机器可读 source-run 语义。
