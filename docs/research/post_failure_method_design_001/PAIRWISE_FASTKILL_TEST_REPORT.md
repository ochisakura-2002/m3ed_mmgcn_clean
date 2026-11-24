# CT-MultiDAG Pairwise Fast-Kill · Windows 验收

2026-11-24 · MULTIDAG · Python 3.9.25 · PyTorch 2.6.0 CPU · pytest 8.4.2

定向合成回归 **120 passed**，覆盖新增 Fast-Kill、既有 Pairwise Preflight、Failure Audit、Train/Dev 投影、TAVFullAudit 和 Fusion。正式 CLI 的 `--help` 可用，运行入口没有学习率、hidden size、门阈值、seed 或 epoch 选择参数。

| 验收项 | 结果 |
| --- | --- |
| Head | 参数精确为 9,856。Xavier uniform、零偏置、全零类对向量、15 个固定索引、双向反号和严格浮点上下界。 |
| 初始化与旁路 | 初始化和关闭态预测与 F0 一致。门外、padding、零修正、零间隔及写回浮点平局逐字节透传。 |
| 写回 | 核对 0.3 + (−0.5) 的翻转、第三类紧邻或等于原第二名的情况、另外四类逐值保持。 |
| Gate、监督与梯度 | 覆盖同分按 class ID、Train 顺序统计量、阈值等号、严格间隔、gold 独立性、S 与 padding 排除。微型优化下降，W/c/V 有梯度，F0 无梯度。 |
| 冻结前向与因果 | 用合成 dialogue 跑实际 FusionTAV F0 结构，核对 h0/h4 缓存与常规 F0 logits；验证截断、未来扰动、换序、padding，能拒绝篡改后的探针结果。 |
| 固定终点 | 通过替身身份和微型合成缓存走完整入口。恰好训练 20 轮，PRIMARY Dev 只评分一次。仅 epoch 20 写独立 head 工件，可严格重载。 |
| 评分与判定 | 覆盖 GO / STOP / REVIEW_SIGNAL 边界、非法运行优先、30 个有向类对、错→错与 R/D 的区别、校准边界及不等长对话配对 bootstrap。 |
| 身份和 Test HOLD | 损坏身份、人口、非有限状态、冻结门控、因果或旁路失败能拒绝。带非零 HOLD 声明的输入被拒绝，当前五项 Test 操作计数仍为 0。 |
| 产物 | 临时合成目录验证 CSV、JSON、报告、checkpoint SHA-256、回传 ZIP 完整性和同 batch 冲突拒绝。ZIP 内只有轻量报告。 |

在已启用的 `m3ed_mmgcn` 环境复跑。受管 Windows 沙箱对 pytest 的临时目录触发 ACL 错误，最终运行使用批准的测试命令与仓库内唯一临时目录。

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$fastkillQa = Join-Path 'tmp' ('pytest_pairwise_fastkill_' + [guid]::NewGuid().ToString('N'))
python -B -m pytest -q --tb=short -p no:cacheprovider --basetemp $fastkillQa `
  tests/models/multidag_cl/tav_full_audit_v1/test_pairwise_fastkill.py `
  tests/models/multidag_cl/tav_full_audit_v1/test_pairwise_preflight.py `
  tests/models/multidag_cl/tav_full_audit_v1/test_failure_audit.py `
  tests/data/build/multidag_cl/test_train_dev_projection.py `
  tests/models/multidag_cl/tav_full_audit_v1/test_model.py `
  tests/models/multidag_cl/tav_full_audit_v1/test_fusion.py
python -B scripts/analysis/models/multidag_cl/run_pairwise_fastkill.py --help
```

此次验收使用随机权重的 F0 结构、合成特征和替身身份资产。正式 checkpoint、认证 IEMOCAP 投影和 Ubuntu CUDA 结果由用户按 Runbook 在正式机器上校验。[Ubuntu Runbook](PAIRWISE_FASTKILL_RUNBOOK.md)

```text
WORKSTREAM_CHECK = PASS
PAIRWISE_HEAD_IMPLEMENTED = YES
PAIRWISE_HEAD_PARAMETERS = 9856
SEED100_FASTKILL_IMPLEMENTED = YES
PAIRWISE_FASTKILL_TESTS = PASS
F0_CORE_MODIFIED = NO
FORMAL_TRAIN_RUNS = 0
TEST_TOUCHED = NO
READY_FOR_UBUNTU_FASTKILL = YES
HANDOFF_READY = YES
```
