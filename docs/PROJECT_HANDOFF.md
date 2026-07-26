# MERC 项目阶段交接总结

> 本文件用于跨对话交接。  
> 只记录会影响项目理解、代码状态、实验协议、结果解释和后续决策的信息。  
> 不记录临时命令、一次性作图流程、临时压缩包、聊天过程和已失效操作。  
> 每个结论必须区分为：**已证实事实、较高置信度判断、待实验验证假设**。  
> 后续每次阶段总结都应继续遵守这一准则，并在原有内容上增量更新，而不是重新堆叠过程记录。

---

## 1. 项目目标

本项目围绕多模态对话情感识别（Multimodal Emotion Recognition in Conversation, MERC）开展研究，重点关注：

1. 面向双人对话和人机交互场景的情感识别；
2. 构建满足在线部署要求的 causal MERC，即预测当前话语时不使用未来话语；
3. 在可复现 baseline 上完成模块改造、消融、对比和论文实验；
4. 最终从多个 baseline 中选择一个效果、稳定性、可解释性和可维护性较好的主干模型，作为后续创新工作的基础。

当前项目同时保留 full-context 与 causal-context 两类实现，但二者目前不能直接解释为严格的“未来信息有无”单变量因果对照，因为部分模型在训练入口、模态、隐藏维度、学习率、损失和训练轮数上仍存在差异。

---

## 2. 项目路径与运行环境

### 本地项目

```text
E:\MERC\m3ed_mmgcn_clean
```

### 远程项目

```text
/home/zhiyuan/research/m3ed_mmgcn_clean
```

### Conda 环境

```text
m3ed_mmgcn
```

### 当前分支

```text
refactor/model-first-layout
```

### 当前 Git 状态

```text
工作区：clean
本地相对远程：ahead 1
当前本地提交：5c95d2857f63b6afaf20cc26e321a3736a03af33
该提交尚未 push
```

该提交用于清理本地专用组会和临时分析工具，不涉及模型、训练配置或实验协议。

---

## 3. 关键项目提交

### 正式 32-run 训练对应提交

```text
d8a7701b62339a6908a39de5d800cb816174ec90
```

该提交对应当前已完成的主要正式批次，包含 4 个模型、4 个验证 Session、2 种上下文模式，共 32 个运行。

### 本地清理提交

```text
5c95d2857f63b6afaf20cc26e321a3736a03af33
```

该提交完成：

- 删除本地专用组会作图脚本；
- 删除本地专用组会测试；
- 删除未提交的 long32 作图源码和相关测试、说明；
- 保留所有正式训练结果和已生成图表；
- 更新 `.gitignore`、`AGENTS.md` 和 `docs/PROJECT_CONTEXT.md`；
- 工作区恢复为 clean。

---

## 4. 当前代码结构与训练入口

当前 4 个主要模型族及入口为：

### DialogueGCN

```text
scripts/workflows/paper_aligned/train.py
```

### MMGCN

```text
scripts/models/mmgcn/unified/train.py
```

### MultiDAG-CL

```text
scripts/models/multidag_cl/unified/train.py
```

### GS-MCC Project Variant

```text
scripts/workflows/causal_graph/train.py
```

这些入口目前已经纳入统一实验管理和输出路径体系。

---

## 5. 输出目录规范

项目已经完成输出目录重构，canonical 结构为：

```text
outputs/<YYYYMMDD>/<experiment_group>/
├── runs/
├── logs/
├── manifests/
├── review/
├── reports/
└── analysis/
```

例如正式 32-run 主批次使用：

```text
formal_long32_primary_seed42
```

后续所有正式实验、修复实验和论文复现实验都应使用该结构，不再写回旧目录：

```text
outputs/long_training/
outputs/launcher_logs/
```

旧结果仍保留用于审计和对照，但不作为新实验的写入位置。

---

## 6. 当前正式实验协议

### 数据集

当前主要正式实验使用 IEMOCAP。

### Session 设计

```text
Validation Session：Ses01、Ses02、Ses03、Ses04
Test Session：Ses05
```

### Seed

```text
42
```

### 上下文模式

```text
full-context
causal-context
```

### 模型数量与总运行数

```text
4 models × 4 validation sessions × 2 context modes = 32 runs
```

### Checkpoint 选择

```text
基于 validation weighted F1 选择 best checkpoint
Test 不参与 checkpoint 选择
```

### 已证实事实

- 32/32 个正式 run 完成；
- 无运行级失败；
- 未发现 Traceback、OOM、NaN；
- 32 个 best checkpoint 和 32 个 last checkpoint 均存在；
- 所有 best epoch 与最大 validation weighted F1 对齐；
- 所有运行使用同一特征文件哈希；
- Test 未参与模型选择。

---

## 7. 正式 32-run 的工程状态

### 已证实事实

- 训练链路完整；
- 运行配置均可解析；
- 输出和 checkpoint 可追踪；
- 正式批次可用于后续异常诊断与 baseline 比较；
- 最近一次完整测试结果为：

```text
318 passed, 3 skipped
```

### 当前证据缺口

共 8 个 run 缺少统一的 test 结果证据：

```text
causal MMGCN：4 个
causal MultiDAG-CL：4 个
```

其中：

- causal MMGCN 缺少标准化 validation/test/run_summary 证据；
- causal MultiDAG-CL 缺少标准化 test/run_summary 证据。

这些缺口后续需要通过独立 evaluation-only 阶段补齐，不能把缺失值补零、插值，或用 validation 指标代替 test 指标。

---

## 8. 当前各模型的主要结果与问题

### 8.1 MMGCN

#### 已证实事实

full-context 组整体较稳定，四个 Validation Session 的 test weighted F1 均可用。

大致组均值：

```text
full-context：
validation weighted F1 ≈ 0.4053
test weighted F1 ≈ 0.4758
```

causal-context 组当前已有 validation 指标，但缺少统一 test 证据。

#### 当前定位

MMGCN 是当前较稳定的工程 baseline 之一，但其 causal 结果证据需要补齐。当前不把 full 与 causal 的差异解释为严格因果效应。

---

### 8.2 MultiDAG-CL

#### 已证实事实

当前项目版本的 MultiDAG-CL 表现较好。

大致组均值：

```text
full-context：
validation weighted F1 ≈ 0.5428
test weighted F1 ≈ 0.5817

causal-context：
validation weighted F1 ≈ 0.6020
test 证据缺失
```

#### 当前定位

当前版本可作为高分 baseline 候选，但尚未完成严格论文级复现。现有实现需要与论文和官方代码进一步核对，区分：

```text
author_official
paper_reproduction
paper_aligned
project_variant
```

---

### 8.3 DialogueGCN full-context

#### 已证实异常

异常 run：

```text
ltp_dialoguegcn_full_context_val_ses04_s42
```

主要表现：

- best epoch 约为 1；
- 约第 11 轮早停；
- validation weighted F1 约为 0.0497；
- test weighted F1 约为 0.0905；
- validation 和 test 几乎全部预测为 Neutral；
- predicted class count = 1；
- dominant class ratio 接近 1；
- 输出概率接近六分类均匀分布；
- logit 对不同样本的区分度极低。

#### 较高置信度判断

该问题不像“模型高置信度偏向 Neutral”，更像是模型尚未形成有效区分能力，输出整体接近常量。

#### 待验证假设

最强假设是：

> DialogueGCN full Ses04 在出现延迟学习转折之前，就被 patience 较短的 early stopping 终止。

支持线索：

- Ses03 在相近前期同样学习缓慢；
- Ses03 在第 11 轮后才开始明显改善；
- Ses04 恰好在类似时点停止。

该假设尚未被诊断训练正式证明。仍需排除：

- Session 特定数据问题；
- 标签映射问题；
- 类别权重问题；
- 特征异常；
- 梯度未更新；
- optimizer 参数遗漏；
- mask、图节点或边构建异常。

---

### 8.4 GS-MCC Project Variant full-context

#### 已证实异常

Ses01 和 Ses02 表现明显偏低：

```text
Ses01：
best epoch ≈ 4
stop epoch ≈ 24
validation weighted F1 ≈ 0.175
test weighted F1 ≈ 0.166

Ses02：
best epoch ≈ 6
stop epoch ≈ 26
validation weighted F1 ≈ 0.166
test weighted F1 ≈ 0.206
```

输出特征：

- 预测类别覆盖不足；
- softmax 分布接近均匀；
- logit 区分度较低。

相对正常的 Ses03、Ses04 学习较慢：

```text
Ses03：
约第 47 轮后进入 validation weighted F1 > 0.2 区域
best epoch 约为 175

Ses04：
约第 37 轮后进入 validation weighted F1 > 0.2 区域
best epoch 约为 205
```

#### 较高置信度判断

GS-MCC full 当前存在明显的慢学习和 Session 敏感性。

#### 待验证假设

最强假设是：

> learning rate 约 1e-5 导致有效学习阶段较迟，而 patience 约 20 的 early stopping 与该慢学习过程不匹配。

但该假设不是唯一可能原因。仍需排除：

- loss 组合或 loss 权重错误；
- scheduler 进一步降低学习率；
- 参数组学习率不一致；
- 参数未加入 optimizer；
- 对比损失或辅助损失量级异常；
- 梯度或参数更新过小；
- Session 特定特征或数据异常。

---

## 9. full-context 与 causal-context 的解释边界

当前 full-context 和 causal-context 之间并非严格单变量控制实验。

二者可能同时存在以下差异：

- 训练入口；
- 模型实现；
- 模态组合；
- 隐藏维度；
- batch size；
- learning rate；
- epoch 数；
- loss 组成；
- 上下文图构建方式。

因此目前可以做：

```text
descriptive comparison
```

但不能直接声称：

```text
causal effect of removing future context
```

后续如需论证未来信息的作用，必须建立严格 paired configuration，只改变是否允许未来上下文，其他变量保持一致。

---

## 10. 当前项目已完成的主要修改

### 已完成

1. 建立 4 个 baseline 的统一训练与实验管理链路；
2. 建立 32-run 正式矩阵；
3. 修复 track 与 split protocol 不一致问题；
4. 增加严格 runtime validation；
5. 完成 32-run 正式训练；
6. 完成 canonical output 路径重构；
7. 保留 legacy read compatibility；
8. 清理本地专用组会和临时分析源码；
9. 保留正式训练结果、图表、表格和审计产物；
10. 建立 `AGENTS.md` 和 `docs/PROJECT_CONTEXT.md` 作为持久项目记忆。

### 未完成

1. DialogueGCN full 异常修复；
2. GS-MCC full 异常修复；
3. 修复后正式受影响 run 重跑；
4. causal MMGCN test 证据补齐；
5. causal MultiDAG-CL test 证据补齐；
6. GS-MCC 论文级复现；
7. MultiDAG-CL 论文级复现；
8. full/causal 严格单变量控制实验；
9. 多 Seed 稳定性实验；
10. 最终 baseline 选择。

---

## 11. 下一阶段任务顺序

后续必须按顺序进行。

### 阶段 2：修复 GS-MCC full 与 DialogueGCN full

顺序：

1. 静态诊断；
2. 增加公共训练诊断；
3. 增加 early stopping 的 `min_epochs` 支持；
4. 准备不超过 12 个小规模诊断候选；
5. 本地测试；
6. Git 提交并同步远程；
7. 远程运行小规模诊断矩阵；
8. 分析根因和候选效果；
9. 只重跑受影响的正式 full-context runs；
10. 保留旧结果并生成替换映射；
11. 更新正式分析数据包；
12. 重新生成表格、图和分析结果。

修复阶段不得直接覆盖旧正式结果。

### 阶段 3：GS-MCC 与 MultiDAG-CL 论文级复现

顺序：

1. 核对论文；
2. 核对作者官方代码；
3. 核对数据、特征、划分、指标和超参数；
4. 区分官方实现、论文复现、论文对齐和项目变体；
5. 分别构建 GS-MCC 与 MultiDAG-CL 的复现入口；
6. 按论文标准训练；
7. 评估与论文结果差距；
8. 形成最终 baseline 候选。

### 最终目标

完成 baseline 选取，并为后续模块创新、消融和论文实验建立稳定主干。

---

## 12. 后续执行原则

### 12.1 未知信息不得猜测

遇到未知的路径、文件名、配置字段、训练入口、输出结构、官方代码、数据维度、checkpoint 或日志字段时，优先给出只读导出命令，由用户提供结果后再制定方案。

### 12.2 每个 Codex 任务使用新对话

默认规则：

```text
一个明确任务 = 一个新的 Codex 对话
```

只有同一任务中的小修正可以保留在原对话。

### 12.3 Codex 优先读取持久文件

每个新 Codex 对话优先读取：

```text
AGENTS.md
docs/PROJECT_CONTEXT.md
docs/PROJECT_HANDOFF.md
```

并只扫描本任务相关目录，避免重复全仓库扫描。

### 12.4 Codex 任务前后必须解释

任务发布前说明：

- 要解决的问题；
- 修改范围；
- 禁止事项；
- 成功标准。

任务完成后解释：

- 改了什么；
- 关键报告字段含义；
- 哪些是事实；
- 哪些仍是假设；
- 是否可以提交；
- 是否可以训练。

### 12.5 Git 命令必须完整成块

每次需要 Git 时，分别提供：

1. 本地检查、暂存、提交和 push 的完整命令；
2. 远程检查、pull 和 commit 对齐的完整命令；
3. 预期输出和异常处理说明。

### 12.6 训练命令按任务块组织

训练相关命令分为：

1. 启动块；
2. 监控块；
3. 收集块。

每一块应尽量是一套可连续执行的完整命令，不拆成大量零散操作。

### 12.7 阶段结束必须标定

每个阶段结束后必须给出：

- 阶段完成状态；
- Git 状态；
- 测试状态；
- 训练状态；
- 输出位置；
- 已确认问题；
- 未确认问题；
- 下一阶段进入条件。

未经用户确认，不自动进入下一阶段。

---

## 13. 当前阶段标定

```text
STAGE_1_HANDOFF_STATUS=PASS
PROJECT_BRANCH=refactor/model-first-layout
LOCAL_HEAD=5c95d2857f63b6afaf20cc26e321a3736a03af33
REMOTE_SYNC_STATUS=LOCAL_AHEAD_1_NOT_PUSHED
WORKTREE_STATUS=CLEAN
FORMAL_32RUN_STATUS=32_COMPLETED
FORMAL_TRAINING_COMMIT=d8a7701b62339a6908a39de5d800cb816174ec90
DIALOGUEGCN_FULL_REPAIR_STATUS=NOT_STARTED
GSMCC_FULL_REPAIR_STATUS=NOT_STARTED
GSMCC_PAPER_REPRODUCTION_STATUS=NOT_STARTED
MULTIDAG_CL_PAPER_REPRODUCTION_STATUS=NOT_STARTED
READY_FOR_STAGE_2=YES
```

---

## 14. 新对话最小读取顺序

新对话接续项目时，按以下顺序读取：

```text
1. AGENTS.md
2. docs/PROJECT_CONTEXT.md
3. docs/PROJECT_HANDOFF.md
4. 当前 git status
5. 与本阶段直接相关的配置、入口和结果目录
```

除非上述文件无法解释当前问题，否则不要重新扫描整个仓库。
