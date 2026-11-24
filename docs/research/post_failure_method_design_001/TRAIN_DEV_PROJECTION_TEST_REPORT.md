# CT-MultiDAG 认证 Train/Dev 投影 Windows 验收

2026-11-24 · `m3ed_mmgcn` · PyTorch 2.6.0 CPU

定向回归 **36 passed**，包括 16 项新 Builder 测试和 20 项既有 Preflight 测试。CLI `--help` 确认四个正式参数可用。

| 验收项 | 检查内容 |
| --- | --- |
| Train/Dev 原样投影 | 断言 dialogue 字典与 split 列表原顺序、utterance ID、speaker、标签及标签 ID 一致。逐字节比较 T/A/V，包括有符号零，同时核对 dtype 与 shape。 |
| CSV 保真 | 源列和源行顺序、额外列、带引号和多行字段、前导零均保留。Test 行及其私有哨兵值不进入输出。 |
| Test 隔离 | 合成 Test dialogue 只要被取字段或迭代即失败。Spy 确认现有 validator 只构造 Train/Dev dataset，证书五项 HOLD counters 为 0。 |
| 认证与消费 | 用共享 `identity_sha` 重算源及投影顺序哈希，检查三个源文件 SHA-256 与产物两份 SHA-256。用独立构造的历史 `data_identity` 实际读取已发布的三文件。 |
| 拒绝路径 | 覆盖错误源 provenance、HOLD 违规、计数不符、错 shape、错 label、错 speaker、错行索引、源目录写入和已存在目标。 |
| 中途失败 | 模拟源文件在验证后变化和发布第二个文件失败，最终目录均没有出现。 |

复跑使用项目环境及唯一临时目录。

```powershell
$projectionQa = Join-Path 'tmp' ('pytest_train_dev_projection_' + [guid]::NewGuid().ToString('N'))
python -B -m pytest -q --tb=short -p no:cacheprovider --basetemp $projectionQa `
  tests/data/build/multidag_cl/test_train_dev_projection.py `
  tests/models/multidag_cl/tav_full_audit_v1/test_pairwise_preflight.py
python -B scripts/data/build/multidag_cl/build_train_dev_projection.py --help
```

验收使用合成文件，未打开正式 canonical 资产、执行真实 F0 前向或生成正式投影。正式 Ubuntu 验证留待手动交接。
