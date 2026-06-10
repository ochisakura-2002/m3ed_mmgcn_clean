# M3ED + MMGCN Baseline Progress Record

Date: 2026-05-26

## 1. Current stage

本阶段完成了 M3ED 多模态特征接入、SimpleMLP sanity baseline、官方 MMGCN 核心逻辑解读、第一版纯 PyTorch MMGCN 实现、MMGCN 训练闭环、统一 checkpoint 评估脚本和 baseline 结果汇总。

当前项目已经具备以下完整链路：

```text
M3ED feature pkl
    -> M3EDTorchDataset / DataLoader
    -> SimpleMLP or MMGCN
    -> train / val
    -> best checkpoint
    -> val / test evaluation
    -> evaluation_summary.csv
```

## 2. Key code and output locations

| Type | Path | Purpose |
| --- | --- | --- |
| MMGCN model | `models/baselines/mmgcn/mm_gcn.py` | 第一版 M3ED-MMGCN 模型实现 |
| Dense graph | `models/baselines/mmgcn/dense_graph.py` | 构造多模态 dense adjacency |
| MMGCN training | `scripts/train_mmgcn.py` | 训练、验证、保存 checkpoint |
| Checkpoint evaluation | `scripts/evaluate_checkpoint.py` | 统一评估 val/test split |
| Evaluation summary | `outputs/evaluation_summary.csv` | 汇总所有 checkpoint 评估结果 |
| Baseline report | `outputs/report_figures/baseline_results/` | 指标表、类别召回表、结果小结 |

## 3. Metric comparison

### Validation split

| model | split | acc | uar | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| MMGCN | val | 0.5608 | 0.3237 | 0.3340 | 0.5239 |
| MMGCN | val | 0.5448 | 0.2905 | 0.2885 | 0.4872 |
| SimpleMLP | val | 0.5431 | 0.3742 | 0.3924 | 0.5306 |

### Test split

| model | split | acc | uar | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| MMGCN | test | 0.5154 | 0.2889 | 0.2935 | 0.4808 |
| MMGCN | test | 0.5068 | 0.2541 | 0.2400 | 0.4468 |
| SimpleMLP | test | 0.5173 | 0.3398 | 0.3534 | 0.5035 |

## 4. Test per-class recall

| label_name | MMGCN | SimpleMLP |
| --- | --- | --- |
| Happy | 0.1550 | 0.3128 |
| Neutral | 0.7596 | 0.7229 |
| Sad | 0.2650 | 0.2956 |
| Disgust | 0.0390 | 0.1514 |
| Anger | 0.6372 | 0.5231 |
| Fear | 0.0000 | 0.0154 |
| Surprise | 0.0447 | 0.3574 |

## 5. Current interpretation

第一版 MMGCN 已完成 M3ED 适配和训练评估闭环，但在当前设置下，test UAR 和 macro-F1 低于 SimpleMLP。

从 per-class recall 看，MMGCN 更偏向 Neutral 和 Anger 等多数类，但对 Happy、Disgust、Fear、Surprise 等类别表现较弱。这说明当前 dense multimodal graph 结构在该配置下可能加剧类别不均衡问题，尚不能作为优于简单三模态拼接模型的证据。

## 6. Evidence boundary

当前结果只能证明第一版 MMGCN 复现链路可运行，不能证明 MMGCN 在 M3ED 上优于简单融合模型。如果在论文或组会中描述，应避免使用“MMGCN 显著提升性能”之类结论。

更稳妥的表述是：

> 第一版 MMGCN 已完成 M3ED 适配和完整训练评估，但在当前设置下，其 test UAR 和 macro-F1 低于 SimpleMLP，提示后续需要进一步分析图结构、类别不均衡和上下文建模方式。

## 7. Next experiment priorities

后续实验优先级如下：

1. `causal-context MMGCN`: 将 `context_mode` 从 `full` 改为 `causal`，模拟机器人实时对话中不能使用未来上下文的情形。
2. `window-limited MMGCN`: 设置有限 `window_past/window_future`，检查全连接 dense graph 是否引入过多噪声。
3. `class-weighted loss`: 针对 Fear、Disgust、Surprise 等少数类召回过低的问题，引入类别权重。
4. `multiple random seeds`: 当前只有单次 seed，不能判断结果稳定性。
5. `feature pkl comparison`: 比较不同 M3ED 官方特征组合对结果的影响。

## 8. Current risk assessment

当前最大风险不是代码不能跑，而是复杂模型没有带来相对于简单模型的有效提升。如果后续没有改善 UAR 和 macro-F1，MMGCN 只能作为工程复现 baseline，不能作为论文方法贡献的核心证据。
