# Original Dev Alignment 实现与测试记录

## 兼容认证

Ubuntu 以已审核 Stage A 回执的 SHA-256 为信任入口。回执验证后逐一核验 D0 运行证据和预算锁，再复核 F0 投影、训练配置和 D0 快照。Stage B 的 source inventory 按相对路径并集比较，新增、移除或改动均需精确匹配策略。经审查的四项改动覆盖编排、环境身份、失败恢复、表完整性和作图，旧字节基准为 Git `25271ae`，新字节取部署文件。详见 [源码兼容策略](../../../scripts/workflows/benchmarks/original_dev_source_compatibility.json) 与 [修改文件 SHA-256](MODIFIED_FILES_SHA256.csv)。

正式归档的 `stage_a_source_compatibility.json` 记录原 Stage A SHA-256、兼容策略 SHA-256、两侧完整文件清单的规范 SHA-256 和逐项旧、新值。环境门禁逐字段严格比较，包含完整的 `torch_build`，历史 MKL Build 差异依旧会拒绝。历史回执已有的环境字段继续原样认证。新回执另外记录进程导入顺序和 Ubuntu 数学库路径。模型、训练循环、优化器、数据、损失和评价原语沿用冻结源码。

修改文件清单以文件原始字节计算 SHA-256。Git 基线对应 Ubuntu 的 LF 字节，两个已有未跟踪文档使用任务开始时的本地字节。哈希清单本身不参与自哈希。

## 本地验证

| 检查 | 结果 |
| --- | --- |
| 工作区与冻结源码 | MULTIDAG guard 通过，冻结 F0 路径相对 `aa4fa63` 无变化。 |
| 启动与环境 | 嵌套 JSON、tuple/list、NumPy scalar、真实 MKL/数值差异及字段缺失；独立 CPU 子进程验证相同启动身份，并验证旧绘图导入路径会额外加载 Matplotlib。 |
| 认证与恢复 | 允许清单逐字节验证，未知依赖和错误 SHA 会失败。环境错误发生在 D1 和正式分析目录创建前。旧回执按原字节进入归档，模拟完整 D1 的渲染失败与部分发布后重试，验证权重哈希和训练次数计数。 |
| CPU 合成全链 | 原训练入口执行两组正式形状的 30 轮和 210 步，保存并回放 E13 第 6 批的 U*=90，校验 Best、E30、四轮平均、两路 U*、八端点表、配对统计、13 对 PNG/SVG、图索引和归档成员/哈希。 |
| 边界 | 验证 D3 缺失时六个有效端点和 12 对图、空分组、无定义 ROC/PR 表头、NaN 拒绝和无 GUI Agg。核验 Noto Sans SC 中文绘图和缺字体时的英文回退。 |
| 执行手册 | 七段命令通过 Bash `-n`，本地 `plan` 输出冻结 recipe。 |
| 定向回归 | 183 passed，用时 219.24 秒。范围涵盖原 Dev、TAV runtime、fusion 和历史 G0–G4。16 条 warning 来自本地 Matplotlib/pyparsing。Windows 测试进程仅为本次独立 scratch 目录及其子目录继承工作区权限。 |

CPU 路径使用合成 Train/Dev 数据。正式 GPU 训练和 Test 访问均为 0。本地工作区找不到正式 Stage A 产物或远程失败残留。Ubuntu 预检须复核真实 CUDA/MKL、审批源清单、980 条 Dev 重放、文件权限和历史失败状态，成功后才可运行 D1。Linux 动态库证据随预检返回。

## 状态

以下 PASS 表示本地验证通过，READY 表示已提供工程入口或认证实现。正式 Ubuntu 预检结果待验证。

```text
STAGE_A_PRESERVED = YES
ENVIRONMENT_CANONICALIZATION = PASS
STAGE_B_INITIALIZATION = PASS
STAGE_A_SOURCE_COMPATIBILITY = READY
FAILED_ATTEMPT_RECOVERY = READY
END_TO_END_SYNTHETIC_SMOKE = PASS
STAGE_B_PREFLIGHT_ENTRY = READY
STAGE_B_EXECUTION_ENTRY = READY
PLOT_AND_PACKAGE = READY
TEST = HOLD
READY_FOR_GPT_REVIEW = YES
```
