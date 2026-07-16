# Codex 仓库规则

本文件是 Codex 进入 MERC 仓库时的简短操作入口。
详细、持久的项目上下文位于 `docs/PROJECT_CONTEXT.md`。
源码、正式配置和运行产物始终是最终事实来源。

## 任务启动

1. 先阅读本文件。
2. 根据任务阅读 `docs/PROJECT_CONTEXT.md` 的相关章节。
3. 运行 `git status -sb` 和 `git branch --show-current`。
4. 修改前列出计划查看、修改或删除的文件。
5. 优先使用项目上下文列出的 canonical 路径。
6. 只对任务相关目录执行有目标的 `rg`、文件打开和测试。
7. 不要默认递归扫描整个仓库。
8. `docs/PROJECT_CONTEXT.md` 不足时，才检查冲突涉及的具体实现。

任务涉及 Git、远程同步或实验执行时，同时阅读：

- `docs/codex_workflow.md`
- `docs/git_workflow.md`

任务涉及实验设计、数据划分、缺失模态或结果分析时，同时阅读：

- `docs/experiment_protocol.md`

任务涉及模型结构或 MMGCN 模块修改时，同时阅读：

- `docs/module_implementation_spec.md`

## 默认扫描边界

除非任务明确涉及，下列目录默认不扫描：

- `outputs/`
- `third_party/`
- `data/processed/`
- `tmp/source_audit/`
- `.git/`

不要为了“熟悉项目”执行无目的全仓库扫描。
不要读取 PKL、checkpoint 或大体积结果文件来建立一般上下文。
需要核实时，使用明确路径、窄范围 `rg` 和最小文件集合。

## 保护规则

未经用户明确批准，不得：

- 修改数据划分或 Session holdout 规则；
- 修改 legacy/Clean 特征路径或 SHA256；
- 改变 Validation、Test 或 checkpoint selection 协议；
- 将 Test 指标用于模型选择；
- 修改模型 canonical 名称或复现忠实度声明；
- 把项目 GS-MCC 变体写成官方 GS-MCC 复现；
- 删除 causal MERC 路线、配置、审计或测试；
- 删除 Original MERC 四模型路线；
- 删除历史结果、best checkpoint、metrics 或 predictions；
- 修改远程正在训练的工作区；
- 提交本地临时文件、缓存、PKL、checkpoint 或正式 outputs；
- 随意修改 `third_party/`、`data/processed/` 或 `outputs/`；
- 硬编码本地或远程绝对路径到源码和正式配置。

不得把普通兼容别名当作新的 canonical 实现。
不得移动历史 outputs；读取逻辑必须继续兼容旧结构。

## 修改原则

- 修改应小而可审查，每次优先解决一个问题。
- 保留用户已有且与任务无关的工作区改动。
- 新模型模块应提供 YAML 开关，便于消融。
- 新模块关闭后应尽量保持 baseline-equivalent 行为。
- 优先补充 smoke test 或最小回归，再讨论效果优化。
- 不直接安装依赖；确有需要时先说明并更新环境声明。
- 不在本地 Windows 运行正式训练或多 seed 实验。
- 正式训练和 checkpoint 评估主要在远程 V100 环境执行。
- 除非用户明确要求，不执行 `git commit` 或 `git push`。

## 质量门禁

代码或配置改动完成后，标准门禁为：

```text
python -m pytest -q
python scripts/dev/validate_config_tree.py
git diff --check
```

在本地应通过 Conda 环境 `m3ed_mmgcn` 运行这些命令。
若任务范围很小，可先跑定向测试，但交付前仍应说明完整门禁状态。
文档或代码修改后显示 `git diff --stat`。
测试历史记录不是当前 PASS 的替代品。

## 工作流

```text
本地 Codex 修改
→ 本地测试
→ 人工审查
→ Git commit/push
→ 远程 pull
→ 远程训练
→ 自动整理结果包
→ 人工分析
```

远程正在训练时，不 pull、不编辑、不切换其工作区。
正式远程实验使用 tmux，并通过 review bundle 回传结果供本地分析。

## 项目上下文维护

只有下列重大变化需要同步更新 `docs/PROJECT_CONTEXT.md`：

- 新增或删除模型；
- 正式训练或评估入口变化；
- 数据或特征版本变化；
- 数据划分变化；
- 实验协议变化；
- 输出目录规则变化；
- 最终 baseline 选择；
- 重大复现结论；
- 论文主线变化。

普通 bug fix、格式调整、小型测试和临时调试不更新项目总结。
`docs/project_map.md` 仅为历史兼容跳转页，不是第二份项目上下文。
不要创建并行的 PROJECT_OVERVIEW、PROJECT_STATE、PROJECT_INDEX、
PROJECT_SUMMARY 或 CODEBASE_MAP 文档。

## 上下文冲突

若 `docs/PROJECT_CONTEXT.md` 与源码、正式配置或产物明显冲突：

1. 不自动相信任一方；
2. 只检查冲突涉及的具体文件；
3. 在任务结果中明确指出冲突与证据；
4. 修复代码或文档后同步更新项目总结；
5. 不借机扩大扫描或修改范围。
