# GS-MCC 项目变体数值稳定性报告

日期：2026-07-20  
模型：`project_paper_oriented_gsmcc`  
忠实度：`PROJECT_VARIANT_NOT_PAPER_REPRODUCTION`

## 故障与历史证据

已知无效运行是：

- Legacy：`original_gsmcc_legacy_screening_20260716_151823_9ab9c9`
- Clean：`project_gsmcc_clean_screening_20260716_175031_53d594`

两次运行都从 epoch 1 开始记录 NaN train loss，Test loss、probability 和 confidence 也为 NaN。Legacy best checkpoint 的 436 个 tensor 中有 327 个非有限，2,105,827 个元素中有 2,105,718 个非有限；Clean 对应为 436/327 和 2,306,227/2,306,118。非有限 tensor 比例均为 75%，非有限元素比例分别约为 99.9948% 和 99.9953%。

全 NaN logits 经 softmax 后仍是全 NaN。该历史执行路径的 argmax 落到索引 0，因此所有 utterance 被写成 label 0 / Happy；这不是模型学到的分类结果。Legacy 与 Clean prediction CSV 完全相同，是两次运行坍缩到同一非有限输出路径的证据，不能作为跨特征鲁棒性的证据。

上述历史统计来自任务提供的已确认审计结果。当前本地工作区不包含这两个 run 目录，因此本次没有改写或删除历史产物。

## 第一个非有限阶段与根因

旧实现先计算

```python
cosine = F.cosine_similarity(left, right).clamp(-1.0, 1.0)
similarity = 1.0 - torch.acos(cosine) / math.pi
```

同模态滑窗包含 target 自身，所以每个有效节点的自环都经过 `angular_similarity(x, x)`。forward 的 similarity 有限，但 `acos` 在 `+1` 和 `-1` 的导数边界发散。最小 float32 复现中，forward 有限而 `angular_similarity(x, x).sum().backward()` 产生 NaN gradient。因此第一个可证明的非有限阶段是 angular adjacency 的 backward，而不是 FFT forward、classification loss 或 contrastive loss。

修复前最小复现由回归测试保留：闭区间裁剪后的 `acos(1)` forward 有限、backward 非有限。修复后相同、近似相同、相反、正交和零向量的 forward/backward 均有限；正交 similarity 约为 0.5，输出保持在 `[0, 1]`。

## 数学修复

非自环边采用：

```text
c = cosine_similarity(x_i, x_j)
c_safe = clamp(c, -1 + eps, 1 - eps)
s_ij = 1 - acos(c_safe) / pi
```

`eps` 由 `model.angular_similarity_eps` 配置，当前 Original MERC GS-MCC 配置均为 `1e-7`。这保持 angular similarity 对 cosine 的单调性和 `[0, 1]` 语义，同时避开不可微端点。

自环直接赋固定权重 `1.0`，完全不计算 `acos(x_i, x_i)`。整个 adjacency 没有 detach；非自环边仍能把梯度传回上下文编码器。

图归一化现在显式要求 raw adjacency 有限、有效节点 degree 有限且严格大于 0、inverse square root degree 有限、normalized adjacency 有限。padding 节点仍隔离，只给 padding degree 一个用于 `rsqrt` 的中性值 1；没有用末端 `nan_to_num` 掩盖错误。

## FFT 与 contrastive 定位

诊断脚本记录每层 low/high frequency input、FFT 实部/虚部/幅值、可学习 complex gain、复数乘积、inverse FFT、branch output 和梯度。Synthetic 两 batch 诊断中这些量全部有限，没有观察到 FFT 幅值爆炸，因此没有 clamp 频域输出。

同一 synthetic batch 分别运行 contrastive on/off 后，normalized embedding、similarity matrix、positive/negative logits、logsumexp、raw/weighted loss 和梯度均有限。contrastive objective 没有修改；本修复只改变 angular boundary 与 self-loop 构造。

## 诊断脚本

新增 `scripts/dev/diagnose_gsmcc_numerics.py`。脚本只读取 train split，默认检查两个 batch，并可在同一配置上运行 contrastive on/off。输出位置是：

```text
outputs/<date>/audits/gsmcc_numerics/<diagnosis_id>/
```

输出包括 `tensor_finiteness.csv`、`gradient_finiteness.csv`、`parameter_finiteness_before.csv`、`parameter_finiteness_after.csv`、`loss_components.json`、`adjacency_statistics.json` 和 `diagnosis_summary.json`。

当前本机缺少两份正式配置引用的 PKL：

```text
third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl
data/processed/iemocap/IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.pkl
```

因此 Legacy/Clean 真实两 batch 诊断和真实 2-epoch smoke 都在 feature SHA 前置校验处停止，没有生成伪造的真实数据结论。对应测试会在 PKL 存在的环境执行，否则明确 skip。

## Fail-fast、checkpoint 与运行状态

Original MERC 公共训练循环逐 batch 检查 logits、classification loss、每个 auxiliary loss、total loss、backward 后梯度、gradient clipping 后梯度和 optimizer step 后参数。首个非有限值立即抛出 `FloatingPointError` 子类，错误包含 model、epoch、batch、stage、tensor/parameter、各 loss、learning rate 和 AMP 状态。

best/last checkpoint 保存前扫描所有浮点和复数 tensor；非有限 checkpoint 不会落盘。reload 同时验证文件结构与全部 checkpoint tensor。成功 summary 写出：

- `numeric_status: FINITE`
- `checkpoint_numeric_validation: passed`
- `checkpoint_nonfinite_tensor_count: 0`
- `checkpoint_nonfinite_element_count: 0`
- `checkpoint_parameters_finite: true`
- `final_metrics_finite: true`

只有这些数值条件、prediction count 和正常返回同时成立，run/pipeline 才能标为 `PASS`；否则 pipeline 写 `NUMERICALLY_INVALID`。

## Invalid-run 分析过滤

分析入口先审计 run，再生成任何 aggregate、ranking、baseline selection、Top-2 或 paper gap。以下任一条件都会隔离 run：非 `FINITE` 状态、checkpoint 参数非有限、metrics 中 NaN/Inf、prediction confidence/probability 中空值或 NaN/Inf、epoch 必填 loss 为空或非有限。

审计只读取 `run_dir/logs/evaluations/...`。invalid run 单独写入 `invalid_runs.csv`，不会进入 `aggregate_metrics.csv`。如果 Clean screening 候选中存在 invalid run，`top2_selection.yaml` 写 `status: pending_invalid_run_repair`。回归 fixture 覆盖了旧 pipeline 错写 `PASS`、空 loss、NaN probability、非有限 checkpoint、aggregate 隔离和 Top-2 阻塞。

## 本地闭环与补跑范围

在真实 PKL 不可用的条件下，使用临时、未跟踪 synthetic 配置分别按 Legacy 100 维和 Clean 768 维文本输入运行了两套闭环。两套都完成 2 epochs、每 epoch 2 个 train batches、真实 backward/optimizer step、Validation/Test、best/last save 与 reload，并得到 `numeric_status=FINITE`、checkpoint 非有限 tensor/element 为 0。

本修复没有改变 MMGCN、MultiDAG+CL、DialogueGCN 或 causal baseline 的数学实现，也没有改变数据划分、PKL、feature SHA、Test 协议、Clean Validation 规则或日期输出语义。需要补跑的正式 screening 仅是 GS-MCC Legacy 与 Clean。具备正式补跑数据的远程环境应先运行两套 diagnosis 和两套 2-epoch real smoke；全部有限后再启动正式补跑并重新分析 Top-2。
