# 模块实现规格模板

本文档用于未来添加或修改模型模块时约束实现范围，尤其是 `MMGCN` 相关模块。

## 研究动机

新增模块前必须先写清楚研究动机：

1. 它要解决哪个多模态情感识别问题。
2. 它针对哪类失败模式。
3. 它和现有 `MMGCN` baseline 的关系是什么。
4. 它是否影响训练、评估、缺失模态分析或模态消融解释。

不要只因为“可能提升效果”就直接加入模块。

## 目标失败模式

每个新模块至少对应一个明确失败模式，例如：

1. 少数类 recall 过低。
2. 测试时某模态缺失导致性能大幅下降。
3. dense graph 引入过多跨 utterance 噪声。
4. audio 或 visual 特征质量不稳定。
5. speaker 信息使用不足。
6. 模态间冲突没有被抑制。

如果无法描述失败模式，先做分析，不要改模型。

## 模块输入

必须说明输入来自哪里：

1. `text_features`
2. `audio_features`
3. `visual_features`
4. `lengths`
5. `attention_mask`
6. `speaker_ids_int`
7. projected hidden features
8. graph node features
9. adjacency
10. utterance-level fused features

不要改变 dataset 和 collate 的统一 batch 接口，除非任务明确要求并提供迁移方案。

## 模块输出

必须说明输出用于哪里：

1. 替换某个模态 hidden。
2. 修改 graph adjacency。
3. 生成 modality gate。
4. 生成 utterance-level fused features。
5. 生成辅助 loss。
6. 只生成分析日志。

输出必须有明确张量形状。

## 张量形状约定

当前统一 batch 接口：

```text
text_features:   [B, T, D_text]
audio_features:  [B, T, D_audio]
visual_features: [B, T, D_visual]
labels:          [B, T]
attention_mask:  [B, T]
lengths:         [B]
speaker_ids_int: [B, T]
```

`MMGCN` 内部常用形状：

```text
projected modality hidden: [B, T, H]
valid utterance nodes:     [N, H]
graph nodes:               [3N, H]
adjacency:                 [3N, 3N]
utterance features:        [N, 3H]
logits:                    [B, T, C]
```

新增模块必须明确是否处理 padded position。默认只对有效 utterance 生效，padding 由 `lengths` 或 `attention_mask` 控制。

## 插入 MMGCN 的具体位置

常见插入位置：

1. 原始特征投影前：在 `text_features/audio_features/visual_features` 上处理。
2. 投影后：在 `audio_h/visual_h/text_h` 上处理。
3. graph 构建前：影响 `audio_nodes/visual_nodes/text_nodes`。
4. graph 构建时：影响 `dense_graph.py` 中 adjacency。
5. graph 传播后：影响 `audio_out/visual_out/text_out`。
6. final classifier 前：影响 `[audio_out, visual_out, text_out]` 拼接特征。

涉及 `MMGCN` 的实现前，优先查看：

1. `models/baselines/mmgcn/mm_gcn.py`
2. `models/baselines/mmgcn/dense_graph.py`
3. `scripts/train_mmgcn.py`
4. `scripts/evaluate_checkpoint.py`
5. `configs/train_mmgcn_m3ed.yaml`

## YAML 配置开关

新增模块必须提供 YAML 开关，例如：

```yaml
module_name:
  enabled: false
  hidden_dim: 128
  dropout: 0.1
```

默认值必须尽量保持原 baseline 行为。旧配置没有该字段时，也应能正常运行。

## baseline-equivalent 路径

新增模块必须提供 baseline-equivalent 路径：

1. `enabled: false` 时，行为应尽量等价于原 baseline。
2. 不应改变 `state_dict`，除非必须。
3. 如果改变 `state_dict`，必须说明旧 checkpoint 是否还能加载。
4. 如果引入新参数，必须说明初始化方式。
5. 如果引入随机过程，必须受 seed 控制。

关闭模块后的 smoke test 应优先通过。

## 是否引入新损失

新增 loss 前必须说明：

1. loss 名称。
2. 输入张量。
3. 标量权重。
4. YAML 开关。
5. 默认权重。
6. `enabled: false` 或权重为 0 时是否完全等价 baseline。
7. 是否影响 checkpoint 选择指标。

默认不要引入新 loss。先完成结构闭环和最小测试。

## 最小测试

模块实现后至少做：

1. `py_compile` 语法检查。
2. fake dialogue smoke train。
3. fake dialogue checkpoint evaluate。
4. `enabled: false` 路径 smoke。
5. `enabled: true` 路径 smoke。
6. 如果涉及 active modalities，至少检查 `TAV` 和一个非完整条件。

本地只跑小型 smoke，不跑正式训练。

## 必要消融

正式实验前至少规划：

1. baseline `MMGCN`。
2. 新模块关闭。
3. 新模块开启。
4. 关键超参数消融。
5. 如果与模态相关，区分测试时缺失模态评估和从头训练模态消融。

结果报告必须包含：

1. `Weighted-F1`
2. `Macro-F1`
3. `UAR`
4. `Accuracy`

## Codex 允许修改哪些文件

模型模块任务通常允许修改：

1. `models/baselines/mmgcn/mm_gcn.py`
2. `models/baselines/mmgcn/dense_graph.py`
3. `configs/` 下相关 YAML。
4. `scripts/train_mmgcn.py`，仅限配置读取和参数传递。
5. `scripts/evaluate_checkpoint.py`，仅限配置读取和参数传递。
6. `docs/` 下相关设计和结果记录。
7. `datasets/smoke/` 或 smoke 配置，仅限最小测试需要。

修改前必须列出具体文件。

## Codex 禁止修改哪些文件

除非用户明确要求，禁止修改：

1. `data/`
2. `outputs/`
3. `tmp/`
4. `checkpoint/`
5. `checkpoints/`
6. `logs/`
7. `third_party/`
8. 真实 feature pkl。
9. 真实 checkpoint。
10. 原始数据。
11. 大体积缓存。

不要为了让测试通过而改真实数据或历史实验输出。

## 如何记录实验结果

每次正式实验记录：

1. 实验名称。
2. git commit hash。
3. YAML 配置路径。
4. dataset。
5. seed。
6. active modalities。
7. checkpoint 选择规则。
8. validation 指标。
9. test 指标。
10. 是否为测试时缺失模态评估。
11. 是否为从头训练模态消融。
12. 输出 run id。

建议把小型摘要写入 `docs/` 或小型 CSV，不提交 checkpoint 和大体积输出。
