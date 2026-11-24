# H1 Fast-Kill 测试报告

2026-11-24 · Windows · Python 3.9.25 · PyTorch 2.6.0 CPU

| 验收 | 结果 |
| --- | --- |
| H1 专项 | 28 passed |
| 归档内定向回归 | 202 passed，含 H1 28 项 |
| Clean archive CLI `--help` | PASS |
| 正式恢复 ZIP 复核 | PASS，解包内容、manifest SHA-256、成员和预注册计数一致 |
| 原 HEAD 与候选归档的合成 F0 | 逐字节相同 |
| `F0_BYPASS_IDENTITY` / `LAMBDA_ZERO_IDENTITY` | PASS / PASS（合成 F0） |
| `CAUSAL_INVARIANCE` | PASS（使用正 lambda） |
| 正式 IEMOCAP 训练 / Test | 0 次 / HOLD |

两参数模块的 clamp 边界 Adam 检查从 raw=0 执行一次有效更新，两个有效 lambda 均约 `0.003`。关模块与零 lambda 分别验证了 logits、预测、有效 mask、邻接和层状态。激活分支验证了合法历史边范围、同异说话人关系、冻结 F0、`grad=None`、相似度 detach、未来扰动、前缀截断、padding 和单 utterance。跨形状的前缀比较使用 `rtol=1e-5`、`atol=1e-6` 及 exact argmax/mask。等形状 identity 比较使用 exact tensor equality。

合成 CLI 运行函数完整执行 20 个训练 epoch，仅在终点完成一次 PRIMARY Dev 前向和评分。测试核对确定性 Train tape、H-top25 245/104、SHIFT 的 273/667/40、U/H 完整划分、五项 HOLD、预注册决策边界及输出 ZIP 白名单。合成结果的两个参数留在归档目录外的结果目录。

归档回归包含以下 10 个测试模块。

- `tav_full_audit_v1/test_history_attenuation.py`
- `tav_full_audit_v1/test_history_top25_recovery.py`
- `tav_full_audit_v1/test_pairwise_preflight.py`
- `tav_full_audit_v1/test_pairwise_fastkill.py`
- `tav_full_audit_v1/test_failure_audit.py`
- `tav_full_audit_v1/test_model.py`
- `tav_full_audit_v1/test_fusion.py`
- `tav_full_audit_v1/test_fusion_topology.py`
- `tests/models/multidag_cl/graph_core_variants/test_model.py`
- `tests/data/build/multidag_cl/test_train_dev_projection.py`

正式恢复 ZIP 是已有的 Dev-only 轻量文件。F0 的真实 checkpoint 和 IEMOCAP Train/Dev 特征由未来 Ubuntu 正式入口校验。
