# 实验协议

本文档记录当前 `MMGCN` 相关实验逻辑，避免后续混淆实验解释。

## 当前主线

当前主线是复现和分析 `M3ED + MMGCN` baseline。`SimpleMLP` 作为 sanity baseline 和对照模型保留。

项目支持：

1. `M3ED`
2. `IEMOCAP official MMGCN-style features`

当前实验应优先保证训练、评估、分析闭环可复现，再讨论新模块或效果优化。

## 两类模态实验不能混淆

### 测试时缺失模态评估

测试时缺失模态评估指：

1. 先训练一个完整三模态模型。
2. 在 evaluation 阶段屏蔽某些模态。
3. 使用同一个 checkpoint 比较不同测试条件。

它回答的问题是：完整三模态模型在测试时遇到缺失或不可用模态时有多鲁棒。

它不能回答的问题是：某个模态本身对任务贡献有多大。

### 从头训练模态消融

从头训练模态消融指：

1. 每个模态条件单独训练模型。
2. 每个条件都有自己的 checkpoint。
3. 在相同训练设置和 seed 下比较最终测试结果。

它更适合回答：在同样训练预算和模型设置下，不同模态组合能提供多少信息。

## 当前测试时缺失模态条件

当前测试时缺失模态条件固定为：

1. `TAV`
2. `TA`
3. `TV`
4. `AV`
5. `T`
6. `A`
7. `V`

含义：

1. `T` = text
2. `A` = audio
3. `V` = visual

不要把测试时屏蔽结果解释成模态本身贡献。比如 `TAV checkpoint` 在测试时只开放 `T` 的结果，表示三模态训练模型在只剩 text 可用时的鲁棒性，不表示 text-only 从头训练模型的真实能力。

## 配对 seed 分析规则

多 seed 分析必须使用配对规则：

1. 在每个 seed 内先计算差值：`TAV - 当前条件`。
2. 得到每个 seed 的 paired drop。
3. 再跨 seed 汇总 mean、std、置信区间或其他统计量。

不要先跨 seed 平均 `TAV` 和当前条件，再相减。那样会破坏配对关系。

## 当前下一步计划实验

下一步计划是从头训练模态消融，优先顺序：

1. `TAV`
2. `T`
3. `TA`
4. `TV`

原因：

1. `TAV` 是完整 baseline。
2. `T` 用于确认 text-only 能力。
3. `TA` 和 `TV` 用于初步比较 audio 与 visual 作为附加模态的贡献。

后续可扩展到：

1. `A`
2. `V`
3. `AV`

## 必须报告的指标

每个正式实验至少报告：

1. `Weighted-F1`
2. `Macro-F1`
3. `UAR`
4. `Accuracy`

指标解释应同时关注类别不均衡问题。只看 `Accuracy` 不足以说明模型在少数类上的表现。

## checkpoint 选择规则

训练阶段：

1. 使用 validation split 选择 best checkpoint。
2. 监控指标以配置中的 `logging.monitor_metric` 为准。
3. 当前常见选择是 `val_uar`。

测试阶段：

1. 只评估 validation 选出的 best checkpoint。
2. 不允许根据 test 指标选择 checkpoint。
3. test 结果只用于最终报告或对照分析。

## 配置记录规则

每次正式实验必须保留：

1. 使用的 YAML 配置。
2. git commit hash。
3. seed。
4. 数据路径说明，但不要写死个人绝对路径。
5. 输出 run id。
6. checkpoint 名称。
7. evaluation split。

如果本地和远程路径不同，文档中只记录相对项目路径或环境变量式描述。

## 结果解释边界

允许说：

1. 某条件下 `MMGCN` 的 `Weighted-F1`、`Macro-F1`、`UAR`、`Accuracy` 是多少。
2. 测试时缺失模态使完整三模态 checkpoint 的性能下降多少。
3. 从头训练的不同模态组合在相同 seed 下相差多少。

不要说：

1. 测试时屏蔽某模态的结果等于该模态本身贡献。
2. 单 seed 结果证明方法稳定有效。
3. test 上最好的 checkpoint 是最终模型选择依据。
4. 没有 smoke 和复现记录时声称实验闭环可靠。
