# A0 Prefix WES 零训练诊断

2026-11-23 · `A0_DIAGNOSTIC = INCONCLUSIVE`（真实 Train）· `OOF_DIFFICULTY_VALIDATION = NOT_AVAILABLE`

本地可用资产包括历史归档的 Train epoch 汇总和整段 DMF 分桶记录。认证的 `model_input_manifest_tav_t1024_v1.csv` 与一致的 provenance JSON 未找到本地副本，现有历史记录不足以重建逐位置 Train gold/speaker 序列。无已知的合格 Train-only OOF logits、loss 或 error 文件。

CPU 零训练样例覆盖 7 对话、20 话语、13 种逐对话后缀切点。每个切点的后缀标签全部轮转，检查固定前缀的两种分数和第 1–8 轮权重逐值相等。得到 `PREFIX_SUFFIX_INVARIANCE = PASS`。独立集成验收使用 17 个 synthetic 对话，于第 8 轮检查权重恢复为 1。真实 Train 的后缀审计由 Ubuntu launcher 在任何正式更新前执行。

以下为本地零训练样例的逐目标统计，数据规模仅用于检查导出契约。

| 分数 | 均值 | 最小 | 最大 | 长度 Pearson |
| --- | ---: | ---: | ---: | ---: |
| 完整对话原 DMF | 0.597 | 0.500 | 0.667 | 0.994 |
| 等权前缀 | 0.537 | 0.500 | 0.667 | 0.473 |
| WES 前缀 | 0.511 | 0.480 | 0.642 | 0.133 |

本地数据保存在 `tmp/post_fusion_fastkill_001_local_diagnostics_20261123/a0_synthetic/`。`--a0` 只需认证 manifest 和 provenance。正式 launcher 对经校验的 Train dataset 重新打分，使用该次输出的逐位置结果。导出每对话终点及每目标 CSV、长度 Pearson／Spearman、固定分位边界下的 label × speaker 分布、三个评分的早期有效 class exposure，以及后缀测试清单。原 DMF 和等权前缀的曝光计算是零训练反事实。 [Preflight 命令](PARALLEL_FASTKILL_RUNBOOK.md)
