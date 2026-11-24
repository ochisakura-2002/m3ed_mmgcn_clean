# CT-MultiDAG seed100 零训练 Preflight Windows 验收

2026-11-24 · `m3ed_mmgcn` · PyTorch 2.6.0 CPU · pytest 8.4.2

定向回归 **24 passed**，包括新 Preflight 的 20 项和已有 Failure Audit 的 4 项。CLI `--help` 可用。给正式入口传入不存在的哨兵路径时，Windows 在读输入或创建输出前返回 2。

| 验收项 | 结果与验证内容 |
| --- | --- |
| MAD / 排序 | 均匀分布接近 1、近 one-hot 接近 0、值域 `[0,1]`。相同 logits 按类 ID 排序，真类 rank 保持完整。 |
| Train τ / Gate | `ceil(0.75N)` 顺序统计量，无插值。覆盖低 MAD、等阈值、平分、`m=1`、`m>1`、小于 1 和 padding。 |
| Oracle / 评分 | 逐行检查只改 gate 内真类为 rank-2 的错误。四项指标与项目 `classification_report` 一致，检验非翻转样本逐值保留及 Q4 净恢复。 |
| 支持度 | 检查 15 个无序类对、30 个方向、六类与各群真类排名，缺失分层有显式状态。 |
| 状态与门槛 | 两条等号边界允许 PASS，单项不足即 STOP。独立 synthetic PASS 与 STOP 路径引导同一报告流程。身份拒绝返回 INVALID，且测试拦截任何后续 dataset/forward 调用。 |
| 冻结身份 | 检查读取范围只有 seed100。即使 matched CTRL 的指标相同，checkpoint SHA 不同仍被拒绝。正式 CLI 的完整历史工件校验留待 Ubuntu。 |
| Test HOLD | 真实 `FusionTAV` 在合成结构化 Train/Dev 包上以 CPU inference mode 前向。记录只创建两个 dataset 和两个 loader，拒绝 Test split 请求时未触发第三个 loader。模型 forward 不接收 labels，无梯度。五项非零或缺失计数逐项拒绝，错误范围或哈希的包在进入 pickle loader 前拒绝。 |
| Round 1 | 用反序合成 trace 验证按键对齐，损坏 SHA 后 Q4 回落历史稳定排序。合成历史变体检验 U/H 两个 245 样本预算和捕获计数，不一致时拒绝分层。 |
| 回传包 | 合成 PASS/STOP 均生成 ZIP，逐文件核对清单、SHA-256、ZIP 完整性和重名拒绝。外部权重文件未被纳入。 |

可在 Windows 的已激活项目环境用独立临时目录复跑。

```powershell
$preflightQa = Join-Path 'tmp' ('pytest_pairwise_preflight_' + [guid]::NewGuid().ToString('N'))
python -B -m pytest -q --tb=short -p no:cacheprovider --basetemp $preflightQa `
  tests/models/multidag_cl/tav_full_audit_v1/test_pairwise_preflight.py `
  tests/models/multidag_cl/tav_full_audit_v1/test_failure_audit.py
```

受管沙箱的默认临时目录及 pytest `0700` 临时目录触发 Windows ACL 错误。最终验收通过批准的本地测试命令在独立的 `tmp` 目录运行。输入只使用合成数组与替身身份工件。真实 seed100 F0、认证数据包及 CUDA 正式路径没有在 Windows 执行，尚无真实 Oracle PASS/STOP。正式训练 0 次，Test 保持 HOLD。
