# PIL_SCREEN_001 本地测试报告

2026-11-28 · Windows `MULTIDAG` · 既有 Conda 环境 `m3ed_mmgcn`，Python 3.9.25、Torch 2.6.0+cpu。

## 执行结果

| 检查 | 实测 |
|---|---|
| 工作流和语法 | guard PASS，新增 5 个 Python 文件 `py_compile` 通过，D 的旧源码门禁通过。 |
| 配方、批次及平台门禁 | `plan` 解析原 F0 seed100/CL_OFF/TAV YAML 和四臂顺序，显示每臂 210 次更新。完整 30 轮合成采样带与原 DataLoader 迭代一致，每轮为 16×6 + 4，篡改单步 SHA 即拒绝。Windows 正式 `preflight` 在数据读取前以非零状态退出。 |
| 参数、RNG、OFF 等价 | 模型参数实测 1,638,306 / 1,698,306 / 1,639,807。四臂共享参数和首步前 RNG 逐项一致。固定形状的 float32 logits、概率、表示、mask、邻接与原 F0 数值精确一致。OFF 单步目标、clip 前范数、更新权重和 RNG 与原入口一致。 |
| C01 | 首发、连续同人、交错、ID 重命名、padding、对话隔离、前缀截断与追加、同形状未来扰动、显式缺失两端 A/V、训练期缺失拒绝。不同形状使用同 dtype F0 的前缀差值校准，并以 8 个 float32 epsilon 为下限。两个模式在非零 U 下输出有差异。 |
| C07 | Train-only 固定置乱、类内边际、错 split/ID/SHA、缺失与重复 ID 拒绝、空前驱集、BCE 分母、零共享首步辅助梯度、两端 gather 反向及 padding 无梯度。扰动 Dev 标签保留 logits、训练 tape 和 RNG。 |
| 训练与恢复 | 在合成对话上运行 OFF 和四个候选。中断后从已提交 epoch 恢复 C07，Best/E30 预测、配对带、模型和辅助头与连续运行一致。C01 按臂 strict reload，C07 由原类 strict load 部署权重。 |
| 分析及归档 | 以合成的 20 个对话、980 个话语复核逐话语导出、10,000 次成对重采样、CSV 回读、裁决、10 套 PNG/SVG、SHA 清单和归档。模拟已链接归档但尚未发布目录的中断，恢复后路径和 SHA 一致。 |

PIL 专项 18 项。固定源码后，包含专项及原 D、TAV runtime、Fusion 和 Fusion 五 seed 入口的完整定向回归为 **111 passed**，16 条弃用警告来自当前 Matplotlib/Pyparsing。最初使用 pytest 默认临时目录时遇到 Windows ACL，改用仓库现有 `artifact_dir`。一次回归中途编辑 PIL 入口触发了 D 测试的源码 SHA 门禁，固定源码复跑后通过。

```powershell
conda activate m3ed_mmgcn
python -B -m pytest -q -p no:cacheprovider --tb=short tests/models/multidag_cl/tav_full_audit_v1/test_pil_screen.py tests/models/multidag_cl/tav_full_audit_v1/test_original_dev_alignment.py tests/models/multidag_cl/tav_full_audit_v1/test_runtime.py tests/models/multidag_cl/tav_full_audit_v1/test_fusion.py tests/models/multidag_cl/tav_full_audit_v1/test_fusion_final_5seed.py
```

## 实际进度

- 正式轨迹 0，Test 访问 0，真实 PKL 加载 0。
- D0 历史身份核验、V100 环境一致性、真实 Train null 分布和正式结果仍待远端。
- C01/C07 性能裁决为待运行。合成数据产生的指标仅验证分析代码。正式统一分析、封包和图表状态以远端批次回执为准。
