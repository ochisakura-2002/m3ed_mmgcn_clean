# Fast-Kill Windows 本地验收

2026-11-23 · PyTorch 2.6.0 CPU · 正式训练 0 次 · Test HOLD

| 验收项 | 结果 | 验证对象 |
| --- | --- | --- |
| 冻结身份和三个 YAML | PASS | canonical F0 config、源码 guard、seed100、双开拒绝、Windows 正式入口拒绝 |
| CTRL equivalence | PASS | 对照冻结 runtime，初始 state/RNG、两次更新的 logits、概率、loss、梯度范数、最终参数和 Dev 预测均精确相等 |
| B1 训练与推理 | PASS | loss 有限，原型梯度有限且非零，原型收到更新，Dev 绕过辅助目标，weights-only schema 及推理参数计数相同 |
| A1 分数和有效权重 | PASS | 论文式相似度分支、同 speaker 转移、padding 无权重、后缀扰动、epoch 8 起恢复原 CE |
| 三臂批次和目标 | PASS | synthetic 全流程 17 个 Train 对话、每轮 2 步、8 轮 16 步，batch tape/RNG、每轮有效目标数完全相同 |
| Test 隔离 | PASS | dataset、loader、forward、geometry 和评估均拒绝 Test，完全配对归档的 Test HOLD counters 为 0 |
| Analysis Bundle | PASS | 包中预测、归档逐文件 SHA、选点、原型记录、300D NPZ、A0 文件及 A1 权重通过 synthetic 端到端检查 |

```powershell
conda run -n m3ed_mmgcn python -B -m pytest -p no:cacheprovider -q tests/models/multidag_cl/tav_full_audit_v1
```

回归目录共 129 项通过，其中本次新增 8 项。Config-only 确认三臂各有 210 次预期更新、144,900 个预期 Train 目标和零 Test counters。

真实 Train/Dev 的 210 步和 144,900 个目标是执行前的契约，Ubuntu 产物会逐轮检查。合成验收无真实候选效应或真实 CUDA 结论。完整 B0 和真实 A0 所需认证资产及冻结权重由 Ubuntu 侧提供。
