# PIL_SCREEN_001 本地测试报告

## 2026-11-28 · 预检兼容修复验收

既有 Conda `m3ed_mmgcn` · Python 3.9.25 · Torch 2.6.0+cpu · NumPy 2.0.1 · PyYAML 6.0.3 · pytest 8.4.2 · scikit-learn 1.6.1 · Matplotlib 3.9.4。复用既有 `artifact_dir`。该环境缺 coverage，逐项记录真实调用入口与覆盖边界。

| 检查 | 实测 |
|---|---|
| 历史回归 | 新增 `test_pil_history_contract.py`，37 passed。 |
| 完整定向套件 | 原 PIL、历史 D、TAV runtime、Fusion 和 Fusion 五 seed 的 111 项，加上 37 项历史回归。当前工作区与短路径隔离候选树均为 **148 passed，0 failed，0 skipped**，各有 16 条既有 Matplotlib/Pyparsing 弃用警告。 |
| 旧故障重放 | 在隔离树恢复 `b6a6938` 的原 `read_history_archive` 函数。使用同一夹具、环境和冻结 D 依赖，正例在 `d.CLASSIFICATION` 处复现 `AttributeError`，pytest 退出 1。恢复候选函数后同用例通过。 |
| 静态及导入 | PIL 的四个 Python 文件及新测试 `py_compile`，29 个 `d.*` 使用点及 102 个直接导入绑定核验。116 个原 provenance collector 本地文件哈希，独立记录在轻量包。 |
| CLI | `--help` 和 `plan` 通过，显示每臂 210 步。缺参数、非法 action、Windows CUDA 平台门均退出 2。CPU 门使用不存在的投影路径，在读取前终止。 |
| RUNBOOK | 7 个 Bash 块经 `bash -n`，3 个 PowerShell 块经 PowerShell Parser。本地仅解析手册里的同步与启动命令。 |

```powershell
$pilTests = 'tests/models/multidag_cl/tav_full_audit_v1'
conda run --no-capture-output -n m3ed_mmgcn python -B -m pytest -q -p no:cacheprovider --tb=short "$pilTests/test_pil_history_contract.py" "$pilTests/test_pil_screen.py" "$pilTests/test_original_dev_alignment.py" "$pilTests/test_runtime.py" "$pilTests/test_fusion.py" "$pilTests/test_fusion_final_5seed.py"
```

隔离评估从固定 HEAD 使用 `git -c core.autocrlf=false archive` 导出，只叠加交付文件。Windows Git 的默认换行转换会影响原文件 SHA，隔离进程以提交字节为准。只读比较 Git 差异、未跟踪路径和实际源码身份。首次完整隔离运行有 147 项通过，Fusion 五 seed 用例的 PyTorch 临时 checkpoint 路径达 266 字符，文件无法打开。换用较短的隔离根后复跑固定源码。主工作区的旧缓存目录产生 ACL 警告，定向测试沿用已存在的测试临时目录。

### 历史数据合同与测试边界

任务书记载 GPT 从真实 Stage B 包打开 63 个 tar 成员。120 条 Fit/Dev 轨迹含 CE 60 条，D0 TSV 30 条，8 个预测端点共 7,840 行。CE 的四个分类指标、Dev hard CE 和 Fit 实际更新目标对齐到 D0 TSV 后逐轮精确一致。Windows 定点检查任务书所在目录、历史报告引用及已知仓库路径，未定位独立 Stage A/B 包，也未取得完整 D0/F0 run。真实工件本地状态 `REMOTE_REQUIRED`。

夹具按任务书三份表的有序表头和内部文件名生成。20 段合成 Dev 对话含 980 条 ID，另有 120 条轨迹、30 条 TSV、8×980 条预测，验证 Fit/Dev 的 4830/980 计数。仓库提供冻结 F0/D 配方，夹具的 Stage A/B 与 F0 证书以合成字节独立计算。checkpoint 路径放置用于 SHA 比较的合成字节。库内提供可重复生成器，轻量包只收报告和运行证据。

### EXECUTION_PATH_COVERAGE

| 路径 / 用例 | 实际覆盖 | 未验证或远端补证 |
|---|---|---|
| `read_history_archive` / `test_read_reviewed_history_schema` | 真 tar/CSV/TSV 解析、四指标、精确 CE/目标、人口及 Best/E30，合成合同 PASS。 | 用真实 Stage B 和独立认证的 Dev 人口复核。 |
| `setup → select_baseline → certify_d0` / `test_certify_reuse_and_actual_setup_with_cpu_boundaries` | 实读合成工件。调用冻结 `read_reference / checked_approval / checked_completion / checked_epochs / numerical_audit`，覆盖 SHA、E21、完整轨迹和 E30。实际绘图探测子进程通过。 | 夹具替换投影装载、目标带生成、CUDA runtime/配对和设备来源检查。Ubuntu 需重跑真实外层。 |
| 允许的旧资产缺失 / `test_shared_off_identity_exact_and_ordered` 与 setup 用例 | 同一 setup 删除合成 Stage B 时选择一个 `PIL-CTRL`。真实 OFF 比对接受精确记录，拒绝微小 CE 差异和错轮次。 | 如远端实际 D0 工件缺失，执行经授权的同配方 OFF，须过 210 步原门禁。 |
| `test_archive_contract_rejects_corruption` | 拒绝缺/重复字段、重复成员、缺/重复 epoch、错日志顺序和人口、非数值/非有限值、E30 错 ID/重复/错序/坏 logits、概率或指标。 | 真实 Stage B 与 F0 文件交叉认证。 |
| `test_invalid_history_never_selects_off`、`test_unexpected_history_error_is_propagated`、`test_missing_stage_a_must_stop_instead_of_selecting_off` | 坏 SHA、错 HEAD、Best、配置、checkpoint、logits、split、HOLD 与参数量被拒。注入原 AttributeError、字段 KeyError、权限、未预期路径缺失和 tar 错原样传播。 | 失败时按实际 traceback 和资产状态回审。 |
| CLI 和来源 | 实际模块入口、配置、`check_frozen_source` 和隔离 AST/inspect，通过本地任务检查。 | Stage A 完整实际依赖、Linux 数学库映射、CUDA/驱动/V100 与真实 C07-NULL 改变量。 |
| 训练、续跑和发布 / 原 PIL、D 定向回归 | 合成 OFF 原入口单步一致，C07 合成断点恢复与连续运行一致，封存发布中断恢复后 SHA 一致。 | 正式 `run_batch`、POSIX 锁、`read_completed_arm` 的 210 步重用及实际批次恢复待 Ubuntu。 |

`ASSET_UNAVAILABLE` 指已登记且确实不存在的可选 D0 工件。`IDENTITY_MISMATCH` 含 SHA、配方、HOLD、人口、E21/E30 等偏离。`UNSUPPORTED_SCHEMA` 含归档成员、表头、行宽、编码和字段类型不匹配。`ENGINEERING_ERROR` 含未知程序、依赖和权限失败。后三类保留异常并阻止预检通过。

**尚未验证。** 真实 D0/F0 工件、实际投影和 Stage A/B、V100 环境、真实预检、正式训练、远端批次续跑和正式分析。手动同步及复跑见 [RUNBOOK.md](RUNBOOK.md)。`PARTIAL` 或失败先回审。Test HOLD，真实 PKL 读取/写入为 0。

## 初次实现验收记录

以下保留 `b6a6938` 的首份测试报告。历史路径的新覆盖与最新结果以上方记录为准。

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
