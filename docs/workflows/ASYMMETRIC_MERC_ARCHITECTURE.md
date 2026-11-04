# Asymmetric MERC 架构规则

本文冻结 workstream 的组件归属、组合方式与阶段顺序，是架构约束，不代表各阶段已经实现或通过实验验证。
本轮治理任务不修改 MultiDAG、S2D 的数学行为、forward 契约或 checkpoint/state-dict schema。

## 组件归属

| Namespace | 定位 |
| --- | --- |
| `models/multidag_cl/` | reusable TA component：TA fusion、causal MultiDAG history、TA contextual representation |
| `models/s2d/` | reusable V component：S2D Visual Expert，提供视觉表示 `h_v` |
| `models/asymmetric_merc/` | composition / late fusion / confidence / router |

## 当前目标架构

```text
T + A
  |
  v
TA fusion
  |
  v
causal MultiDAG history
  |
  v
TA contextual representation -----\
                                  > TA/V late fusion -> K=6
S2D Visual Expert -> h_v ---------/
```

TA 分支负责对话历史建模，视觉分支提供 `h_v`，在 TA contextual representation 之后进行 TA/V late fusion，输出六类情感预测。
V0 不把 V 放回 DAG，不在本轮额外定义或改变两个组件的数学行为。

## 阶段顺序

| 阶段 | 范围 |
| --- | --- |
| V0 | TA contextual + S2D `h_v` simple late fusion |
| V1 | `A_raw`-guided visual frame sampling |
| V2 | TA confidence + V confidence |
| V3 | asymmetric dynamic router |

每个阶段作为独立实现任务推进；新增模型模块继续遵守既有 YAML 开关、baseline-equivalent 路径和最小检查要求。
本规则不提前确定 late fusion 公式、置信度计算、router 算法或训练规程。

## 复用与集成规则

- 不允许复制一份 MultiDAG 源码进入 `models/asymmetric_merc/`。
- 不允许复制一份 S2D 源码进入 `models/asymmetric_merc/`。
- integration 通过 canonical import 使用现有 component，组合与适配代码归 `models/asymmetric_merc/`。
- 组件更新先在所属 workstream 完成审查，再通过 reviewed commit 的 cherry-pick 或 clean milestone merge 进入 integration；不得复制源文件代替 Git 集成。
- 集成前检查当前 worktree 身份、dirty 状态、提交依赖与改动范围，不覆盖历史本地文件。

工作区边界见 [Git Workstream Policy](GIT_WORKSTREAM_POLICY.md)，共享外部资产见 [Active Workstreams](../handoffs/ACTIVE_WORKSTREAMS.md)。
