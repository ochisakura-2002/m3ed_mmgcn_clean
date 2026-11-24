# CT-MultiDAG 认证 Train/Dev 投影实现报告

`WORKSTREAM = MULTIDAG` · 2026-11-24 · Windows 合成验收

## 源身份与输出契约

Builder 读取冻结 canonical TAV 的 pickle、model manifest 和 provenance。通过源文件 SHA-256、canonical manifest 的 `identity_sha`、split 计数、对齐状态及 HOLD 证据，组装与正式 TAV runtime 一致的完整 `data_identity`。正式 Train 和 Dev 分别固定为 100 段（4,830 条）与 20 段（980 条）。源资产校验失败时返回 `INVALID`。

| 文件 | 说明 |
| --- | --- |
| `train_dev_tav.pkl` | 保留 canonical schema、dataset version、feature spec、label vocab，以及 Train/Dev dialogue 的 T/A/V、utterance ID、speaker、标签和标签 ID。保留原 split 内 dialogue 顺序、dialogue 字典插入顺序及原始行顺序。 |
| `model_input_manifest.csv` | 保留 Train/Dev 原行与所有原列，字段值逐项透传，包括额外列。 |
| `train_dev_provenance.json` | 使用 `ct_multidag_train_dev_projection_v1` 与现有文件名、SHA-256、顺序 SHA-256 和 HOLD 字段。 |

PyTorch 的现有 Preflight projection validator 在发布前实际重新打开暂存包，校验 Train/Dev IDs、原始行索引、speaker、标签、标签 ID、A1582 指纹和 T/A/V shape/dtype。Builder 复核三个源文件 SHA-256 后预留新目录，最后发布 provenance。已有目标目录返回 `REFUSE`。普通发布失败会撤销本次已发布的文件，源目录保留原文件。

## Test 隔离

可信的全 split pickle 仍需整体反序列化。Builder 只检查 split 成员与 dialogue 字典的键，从 Train/Dev 成员提取记录，Test dialogue 的字段从未被取值。全量原 CSV 的身份 SHA 只用 `IDENTITY_FIELDS`，Test 分支只取这些元数据。投影包无 Test split、Test dialogue 和 Test manifest 行，五个 Test HOLD counters 均为 0。

本轮保持 Preflight 的三个模块、正式模型及训练路径原样。Windows 未运行正式数据 projection、F0 前向或 Pairwise Preflight。正式包与历史 F0 run manifest 的 `data_identity` 相等性由 Ubuntu 用户按 Runbook 校验。[Runbook](TRAIN_DEV_PROJECTION_RUNBOOK.md) · [测试报告](TRAIN_DEV_PROJECTION_TEST_REPORT.md)
