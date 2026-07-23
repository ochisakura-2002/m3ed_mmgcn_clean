# Codex 使用工作流

本文档说明后续应该如何使用 Codex 处理本项目任务。

## 新任务如何开始

每个新 Codex 任务建议从以下步骤开始：

1. 读取 `AGENTS.md`。
2. 读取 `docs/project_map.md`。
3. 执行 `git status --short`。
4. 执行 `git diff --stat`。
5. 明确当前任务要查看哪些文件、要修改哪些文件。
6. 如果工作区已有修改，先判断是否与当前任务相关，不要直接覆盖。

推荐对 Codex 的开场要求：

```text
先读取 AGENTS.md 和 docs/project_map.md。
先执行 git status --short 和 git diff --stat。
修改前列出计划查看和修改的文件。
每次只解决一个问题。
修改后展示 git diff --stat。
不要 commit。
```

## 不同任务读取哪些文档

通用代码任务：

1. `AGENTS.md`
2. `docs/project_map.md`

Git、本地/远程同步、实验执行：

1. `docs/codex_workflow.md`
2. `docs/git_workflow.md`

实验设计、缺失模态、模态消融、结果分析：

1. `docs/experiment_protocol.md`
2. `docs/project_map.md`

模型结构或 `MMGCN` 模块修改：

1. `docs/module_implementation_spec.md`
2. `docs/experiment_protocol.md`
3. `models/baselines/mmgcn/mm_gcn.py`
4. `models/baselines/mmgcn/dense_graph.py`

本地 smoke test：

1. `docs/smoke_test_protocol.md`
2. `configs/mmgcn/unified/synthetic/full_context/synthetic/smoke.yaml`
3. `datasets/smoke/mmgcn_smoke_dataset.py`

## 本地 Windows 工作流

本地 Windows 适合做：

1. 代码编辑。
2. 语法检查。
3. import 检查。
4. 配置解析。
5. 小型 fake dataset smoke test。
6. CSV 表格分析。
7. 图表生成。
8. Git diff 审查。

本地 Windows 不适合做：

1. 正式 M3ED 训练。
2. 多 seed 长训练。
3. 大 checkpoint 批量评估。
4. 依赖真实大体积 feature pkl 的完整实验。

本地命令优先使用：

```powershell
conda run -n m3ed_mmgcn python <script> <args>
```

不要假设当前 shell 已经 `conda activate m3ed_mmgcn`。

## 远程 V100 工作流

远程 V100 适合做：

1. 正式训练。
2. 多 seed 训练。
3. checkpoint 评估。
4. 缺失模态评估。
5. 模态消融训练。

远程执行前，应先确认本地需要的代码已经提交并 push 到 GitHub。不要手工在远程临时改代码跑正式实验，除非这些修改之后会同步回仓库。

## 什么修改必须 push 后再远程训练

以下修改必须先 commit 并 push，再去远程 V100 训练：

1. `models/` 下的模型结构修改。
2. `datasets/` 下的数据读取或 collate 修改。
3. `scripts/train_mmgcn.py`、`scripts/evaluate_checkpoint.py`、`scripts/run_experiment_pipeline.py` 修改。
4. `configs/` 下正式实验配置修改。
5. `utils/metrics.py` 或指标计算逻辑修改。
6. 任何影响 checkpoint 可复现性的代码修改。

以下修改通常不需要远程训练前 push，但仍建议纳入版本管理：

1. 文档。
2. 本地 smoke 配置。
3. 分析脚本的小修。

## 绝对不要提交到 Git 的文件

不要提交：

1. `data/`
2. `outputs/`
3. `tmp/`
4. `checkpoint/`
5. `checkpoints/`
6. `logs/`
7. `*.pkl`
8. `*.pickle`
9. `*.npz`
10. `*.npy`
11. `*.h5`
12. `*.hdf5`
13. `*.pt`
14. `*.pth`
15. `*.ckpt`
16. `__pycache__/`
17. `.venv/`
18. `venv/`
19. `.vscode/`
20. `.idea/`

如果某个实验产物需要归档，优先提交小型 CSV 摘要、表格、配置和文档说明，不提交大文件。

## 如何要求 Codex 工作

建议使用这种格式：

```text
本次只解决一个问题：<问题>
先查看 git status --short 和 git diff --stat。
修改前说明计划查看和修改的文件。
不要扫描 outputs、data、checkpoint。
不要运行长训练。
不要 commit。
修改后给出运行命令和 git diff --stat。
```

模型改动任务建议再补充：

```text
必须提供 YAML 开关。
关闭模块后应尽量等价于原 baseline。
优先添加 smoke test。
不要先优化模型效果。
```

实验任务建议再补充：

```text
先区分测试时缺失模态和从头训练模态消融。
报告 Weighted-F1、Macro-F1、UAR、Accuracy。
validation 选 best checkpoint，test 只评估该 checkpoint。
```
