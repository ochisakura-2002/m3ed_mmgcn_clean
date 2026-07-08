# Codex 项目上下文入口

本仓库是一个多模态对话情感识别项目，当前主开发 baseline 是 `MMGCN`。后续 Codex 对话进入本仓库时，先按本文档建立上下文，不要默认全量扫描仓库。

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

## 当前已知状态提示

最近本地 smoke test 引入了以下工作区变化：

1. 已修改：`scripts/train_mmgcn.py`
2. 已修改：`scripts/evaluate_checkpoint.py`
3. 未跟踪：`configs/smoke/`
4. 未跟踪：`datasets/smoke/`

这些变化用于让 `MMGCN` 在无真实 M3ED 数据时跑通最小 train -> checkpoint -> evaluate 闭环。后续任务不要误判为用户无关改动。
