# PIL_SCREEN_001 本地实施报告

## 2026-11-28 · 预检兼容修复与规范首次落地

| 任务合同 | 实际范围 |
|---|---|
| 任务 / 工作区 | `PIL_SCREEN_001_PREFLIGHT_COMPATIBILITY_FIX` · 兼容修复及规范落地 · MULTIDAG。HEAD `b6a6938ae15efb4eb47f9b1c66c069020b4ab74a`，分支 `feat/multidag-cl-paper-reimplementation`，origin fetch/push 一致，guard PASS，起始暂存区为空。 |
| 授权 / 停止条件 | 修复 PIL、做只读历史认证与合成 CPU 验收、落地规范。共同 F0、Stage A、投影或源码无法认证时停止。数据为 Train/Dev-only，Test HOLD。 |
| 科研合同 | CT-MultiDAG F0、seed100、30 epoch/210 updates、原配方、配对带和 Best selector。C01-CURRENT、C01-PREV、C07-NULL、C07-TRUE 四臂。确有 D0 工件缺失时最多一次共享 `PIL-CTRL`，需通过原 E21 和完整轨迹精确门禁。 |
| 验收 | 真实 CLI/配置、`setup`、`certify_d0`、Stage B 读取、共享 OFF、原定向测试、隔离候选树。远端另有预检回执。 |

### 根因与修复

Ubuntu 的 `setup → certify_d0 → read_history_archive` 首次访问 `d.CLASSIFICATION` 时触发 `AttributeError`。旧 D workflow 从 D runtime 显式导入部分符号。四处误用现统一直接绑定 `scripts/runtime/multidag_cl_fusion/original_dev_alignment.py:43` 定义的 `CLASSIFICATION`。D workflow 仍先于分析包导入，冻结 D 源码、兼容策略与环境初始化保留。

旧 `setup` 广泛捕获 I/O、类型和字段错误，可能将读错、身份错报为 OFF。`select_baseline` 现在仅在底层抛出 `FileNotFoundError`、路径属于指定 D0 复用工件并确认不存在时，返回 `ONE_SHARED_OFF_REQUIRED`。范围是封存 D0 文件、D0 Best checkpoint、预算锁和 Stage B 包。共同 Stage A 回执、F0 参考缺失阻断。权限、坏 SHA、HOLD、格式问题、未预期缺失和程序异常按原异常传播。

归档按 SHA 验证后，只读指定 CSV 成员，检查成员唯一性、表头、行宽、CE 的 30×2 网格和完整有序 TSV。保留分类指标、Dev hard CE 和 Fit 更新目标的未舍入精确比较。拒绝非有限消费字段、错误 E30 人口/ID/顺序、非有限 logits、越界概率及与 logits 不一致的预测。D0 Best 继续核验 Stage A 封存身份、原 E21 logits/概率/预测和逐轮记录，并核对 `dev` split、零评估更新与原参数量。共享 OFF 先核验 30 个有序 epoch 和 Best E21，再逐轮比对优化目标、梯度范数、人口、更新数和 Dev 指标。

### INTERFACE_CONTRACT

调用位置以本次 `run_pil_screen.py` 为准。D 指 `scripts/workflows/benchmarks/run_original_dev_alignment.py`，D runtime 指 `scripts/runtime/multidag_cl_fusion/original_dev_alignment.py`，Stage B 指 `scripts/workflows/benchmarks/original_dev_stage_b.py`。

| PIL 调用位置 | 真实定义、签名及默认值 | 返回结构和分支合同 |
|---|---|---|
| 39, 64, 162, 185, 263, 296 `d.ROOT / PROTOCOL / GROUP / REFERENCE_FILES` | D runtime 38–44，由 D 显式导入 | 仓库根 `Path`、`ORIGINAL_CT_MULTIDAG_DEV`、`original_dev_alignment`、四个 F0 证据 SHA。配置、输出、身份与回执。 |
| 26, 100–148, 232 `CLASSIFICATION` | D runtime 43，PIL 直接导入 | `accuracy, weighted_f1, macro_f1, uar` tuple，比例单位，历史和 OFF 共用。 |
| 74, 82, 85 `d.provenance / check_environment` | D 65 `provenance(device="cuda")`，Stage B 59 `check_environment(reviewed, actual, *, diagnostics=None)` | 返回 files_sha256、git_head、environment、process_diagnostics。源和环境字段逐项比较，GPU 路径待远端。 |
| 243, 269, 270 `d.check_frozen_source / git_identity / check_runtime` | D 45 `check_frozen_source()`，TA formal 112 `git_identity()`，D 88 `check_runtime(runtime, reference)` | 只读 Git 冻结差异与 HEAD。校验初始化、curriculum SHA、配对人口及 7 steps/epoch。 |
| 251, 255, 291 `d.read_reference / load_inputs / read_config` | D runtime 95 `read_reference(run, frozen, raw)`、118 `load_inputs(directory, frozen, historical, evidence)`、75 `read_config(frozen)` | 分别返回 F0 manifest、Best、TSV 行，或 Train/Dev datasets、冻结有序 tuples、projection、population summary，或冻结 D recipe。 |
| 167, 168, 206, 209 `d.checked_approval / arm_run / RUN_EVIDENCE` | D 293 `checked_approval(paths, approved_sha, config)`、52 `arm_run(paths, arm)`、38 `RUN_EVIDENCE` | 认证 Stage A、六个封存工件和预算锁，返回 receipt 与 lock。`arm_run` 返回 D0 run 路径。 |
| 171, 173–175, 248 `d.epoch_rows / checked_completion / checked_epochs / validate_hold` | D runtime 90 `epoch_rows(run)`，TAV confirm 103 `checked_completion(run_dir, manifest)`、90 `checked_epochs(run_dir, best_epoch)`，投影工具 57 `validate_hold(record, *, legacy=False)` | TSV 字符串字典、正式完成状态与最佳轮。Stage A 使用严格 HOLD 字段，历史 Best 沿用 legacy。 |
| 191 `d.numerical_audit` | D 157 `numerical_audit(run, historical_epochs, historical_evidence, record)` | 检查 30 轮完整字段、693/980、E21、人口、logits、概率和预测精确身份。返回结构化比对结果。 |
| 323, 437, 457, 459 `d.MANIFEST_NAME / copy_artifacts / ship` | `export_fusion_topology_bundle.py` 清单名，D 219 `copy_artifacts(run, destination)`、207 `ship(directory, archive, status)` | 复制封存轻量工件，复核 SHA，生成验收清单和归档 SHA，封存恢复实测。 |

| 其余 PIL 调用分支 | 定义及实际合同 |
|---|---|
| workflow 配方、路径与身份 | `ta_v0.LABEL_VOCAB` 六类固定顺序，`pairwise_preflight_inputs` 的投影文件名，`protocol.read_frozen(root)` 返回 frozen/raw，`read_json(path)` 返回字典。TA formal 的 `json_identity`、`sha256_file`、`write_json_atomic` 保持哈希/状态身份。`resolve_output_paths` 生成日期、组、批次和 run 路径。 |
| workflow 源与锁 | Stage B 的 `canonical(value)` 做类型保真的 JSON，`certify_sources(root, approved_sha, old, new)` 校验 D 来源策略，`batch_lock(path, *, readonly=False)` 使用 POSIX 锁。 |
| workflow 历史与指标 | `validate_predictions(predictions, expected)` 核对有序 ID、对话、gold 标签，`classification_report(labels, predictions, *, loss)` 返回 metrics、per_class、confusion_matrix。 |
| workflow setup、训练与发布 | PIL runtime 的 `build_target_tape(train, expected)`、`validate_target_tape(saved, expected, expected_sha)`、`PILRuntime(...)`、`device_parity(original, candidates)`、`train_arm(runtime, run, *, epochs=ENDPOINT, resume=False)`、`read_completed_arm(runtime, run)`。PIL analysis 的 `export_analysis(directory, results, expected, dev, state)` 与 `plot_tables(directory)`。消费目标带、配对身份、臂结果、指标与封存库存。 |
| PIL runtime 继承、采样及 IO | `FusionAuditRuntime` / `AuditRuntime` 的 `forward_batch / evaluate / loader / resume / validate_checkpoint / checkpoint_payload` 和 `identity / model / pairing / initialization / selector`。`paired_model(raw, *, device, reference=None)` 返回原类及初始化，`training_rng_identity / shared_state / state_digest` 保持状态身份。`FitView(train)` 用原评估器只读 Train，`collate / identity_collate / MODEL_INPUTS` 保留数据和采样顺序。`load_checkpoint / save_checkpoint_atomic / cpu_snapshot` 维持原存储及恢复合同。`STEPS / ENDPOINT / HOLD / SEED` 来自冻结模块。 |
| C01 模型与独立分析 | `FusionTAV / FusionTAVOutput`、原模态顺序及 causal adjacency builder；继承调用 `paper_encoder / fusion / graph_core` 返回 logits、1500D 表示、邻接和 mask。独立分析使用 NumPy、CSV 和 Matplotlib，读取端点报告、30 轮轨迹、有序人口与 PIL 状态。 |

AST 和实际导入枚举 29 个 `d.*` 使用点和 102 个直接导入绑定。调用行、可检查的真实定义与签名及 116 个本地 collector 文件哈希列在轻量包的 `SOURCE_AND_INTERFACE_MANIFEST.json`。运行期继承、惰性绘图和平台分支单列于 [路径覆盖](TEST_REPORT.md)。

### 源与资产身份

两份输入文档已完整读取。规范 v1.0 原样保存到 `docs/workflows/MERC_CODEX_WORK_STANDARD_V1_0_20261128.md`，SHA256 `08ff80af6d0bca981eef04acefdf3f441e39ca13364e53a3c93fe143f614f432`。`AGENTS.md` 的 Workstream 部分仅追加一条入口，各工作区身份沿用原 guard。任务书 SHA 为 `6d18e969ccb851e026759a2c3545ed16ad479ae8aa98794ea2e347b38aa05043`。

复核了原 PIL 实施、测试及运行说明、D 实施和重试报告、冻结配方及本地研究 07/08 镜像。镜像 SHA 与既有清单一致。此前报告记载已读取的 FINAL D、初次 PIL 实施任务书及独立《实验回传规范》当前原文件未在授权检查位置找到，本轮用现行任务书、已存报告和上述镜像核对适用合同。

| 证据 | SHA256 和核验范围 |
|---|---|
| D 兼容策略原文件 | `68668add5ff0cc6bd15bbbdf56e7b42b25f8c1956d19fbf43842c4a4c1bbec11`，本地字节核验。 |
| Stage A tar.gz | `daba97097739247cd47cfa92bac4bf51d3838189626d724a0461e832a6284366`，任务书记录的 GPT 核验。 |
| Stage B tar.gz | `6b36f5ba0dd521355a48d3894770d31a76aa0d34dbd17ff7d0f1d0f9c20ba016`，任务书记录的 GPT 核验。 |
| Stage A receipt | `0b0938bddb9cc99a7075896af0fa90b87ea068cfbbe5c3336d48c51d726235e6`，任务书记录 Stage B 内回执身份。正式读取独立回执。 |
| 修复 workflow / 新测试 | `11b8d20405e6cd564f40c6801fb553bf7e5492d0dc4cd621f5c5c63e3d1f5e4a` / `4dec0f8c0a4c547d7338c00722201988bec629d3690f4e6c4d0ed13a734de43c`。 |
| F0 / D0 | F0 runtime 的 `REFERENCE_FILES`；历史 D0 HEAD `25271aea2d9c13aed1fccac4849aa86c3b68822c`，D1 HEAD `3f0436612e06816bb07402afb250c5f50a825721`。运行目录日志、权重和真实 manifest 待 Ubuntu 认证。 |

八个交付路径见 [RUNBOOK.md](RUNBOOK.md)。起始无关变更见 [UNRELATED_DIRTY_FILES.txt](UNRELATED_DIRTY_FILES.txt)。正式训练 0，Test 访问 0，真实 PKL 加载/写入 0。本次未暂存、提交、推送或部署。

`INTERFACE_CONTRACT=PASS` · `LOCAL_IMPLEMENTATION_PASS` · `REAL_HISTORY_ASSET=REMOTE_REQUIRED` · `SYNC=PENDING` · `REMOTE_PREFLIGHT=PENDING` · `TRAINING=NOT_STARTED` · `TEST=HOLD`。

## 初次实现记录

以下保留 `b6a6938` 初次实施报告。故障和最新状态以上方修复记录为准。

`WORKSTREAM = MULTIDAG` · 2026-11-28 · 本地实施完成，远端预检待手动同步后执行。

Windows guard 通过，分支为 `feat/multidag-cl-paper-reimplementation`，本地起始 HEAD 为 `3f0436612e06816bb07402afb250c5f50a825721`。远端配方指向唯一授权 Ubuntu 工作区。当前窗口有既有未提交文件，实施使用独立新文件。未暂存、提交、推送或启动正式训练。

## 合同核对与实现

| 项 | 工程结果 |
|---|---|
| 历史 F0 | 原正式入口构造 `FusionTAV`，Direct TAV 拼接、Track 四步读出和 1500 维 `h_ta`。原分类头线性层索引为 0、2、5，dropout 为 0.4。本轮共享 OFF 和 C07 使用原类。历史 D0 E21 参考为 693/980、ACC 70.714286%、W-F1 69.962663%。 |
| C01 | 用原严格过去邻接和显式有效掩码取每行最近同人前驱。两个臂分别路由当前和前驱编码 A/V，使用同一个双端可见门，首层加无偏 `U[300,200]`，从零权重开始。padding 和无前驱位置的支路输入精确为零。两臂共 1,698,306 参数，部署状态区分 CURRENT/PREV。 |
| C07 | 辅助训练从原 `h_ta` 取两端 1500 维绝对差。独立零初始化头与 F0 参数放在原 AdamW 同组内；使用 BCE sum 除以全批有效目标数，乘 0.1 加到 hard CE。一个 backward 后全局 clip 5。训练 1,639,807 参数，部署原类 strict load 1,638,306 参数。 |
| Train tape | 前驱存在的目标来自冻结 Train 内同会话同说话人。按当前类别与稳定 utterance ID 顺序，Python 私有 `Random(10007)` 单次置乱。保存真实值、置乱值、人口 SHA、映射 SHA、每类正负数、改变率。逐批验证 ID、位置、标签、speaker 与前驱。Dev 仅在计算指标和离线分组时使用标签。 |
| 初始化与预算 | 复用 `paired_model` 的 F0 seed100、CPU/CUDA RNG、CL_OFF curriculum 与 DataLoader，运行时核对完整 30 轮每批顺序及各轮开始结束 RNG。C01 在隔离 CPU RNG 下构造，随后装载共享初始张量。额外参数从 zeros 构造。每臂保持 30 轮、210 次 AdamW 更新和原 Best 选择规则。 |
| 输出及恢复 | 原 Best selector 为 Dev W-F1 降序、Dev hard CE 升序、epoch 升序。轮末 Fit 来自固定权重全量 Train 推理，实际更新 CE 和 C07 辅助 BCE 另存。最后完整 epoch 原子 checkpoint 保存 optimizer、RNG、Best、步记录及辅助头。结构化训练 checkpoint 和部署 state_dict 留在远端。 |

为了保留按目录校验的 D0/F0 源码范围，C01 类和 PIL 运行时分别放在 `models/experimental/` 与独立 `scripts/runtime/` 文件中。该位置选择保留先前冻结入口的源码门禁。模型因果性的验收边界为已供应的 T/A/V 特征和前缀，上游特征提取因果性仍为 UNKNOWN。

## 共享对照与验收边界

远端入口先复核原 F0 附件和 D Stage A，核对固定投影、现行依赖闭包与原 D 兼容策略、环境、V100 上的合成配对、初始化、采样和 RNG 合同。D0 重用需要通过 Stage A 封存文件清单、当前最佳权重 SHA、完整 E21 数值和逐轮轨迹，轮末 Fit/Dev 曲线及 E30 预测取自已审阅 Stage B 归档。D0 证据失败但共有协议身份仍成立时，调用相同外壳补一次共享 OFF。OFF 必须与原 E21 logits、概率、标签、ID 和全轮训练/验证记录满足 D Stage A 的精确数值标准。配方无法认证或 OFF 出现数值偏离时停止。

原始任务书已完整读取。研究包的四份正文使用本地镜像，均与既有 SHA 清单一致。读取了 2026-11-27 FINAL D 关闭项目源。独立研究 ZIP 和独立《实验回传规范.md》未定位，回传字段依本轮任务书、已核验的 08 和 FINAL D 归档实现。本地未取得真实历史 D0 文件或认证 GPU 环境，当前的 D0 身份为待远端验证。

分析对同一 Best 使用整体 ACC/W-F1 及正确数，同时导出 Macro-F1、UAR、NLL、Brier、ECE10、E30、逐话语、六类和混淆、R/D/net、margin 及机制诊断。固定 gap 采用 1、2、3–4、5+ 个 turn，support 低于 30 标为仅描述。以任务书的 +0.20 pp REVIEW 起点标注 matched 比较中的微小差值，分类 GO 仍使用相对 F0 的 +0.50 pp 和 ACC 不降。Bootstrap 按同抽 20 个 Dev 对话，seed10002、10,000 次。预先暴露的固定 Dev 与两名演员约束统计外推。

## 实际新增文件

- `configs/multidag_cl/pil_screen/seed100.yaml`
- `models/experimental/multidag_cl_pil_screen.py`
- `scripts/runtime/multidag_cl_pil_screen.py`
- `scripts/workflows/benchmarks/run_pil_screen.py`
- `scripts/analysis/models/multidag_cl/pil_screen_analysis.py`
- `tests/models/multidag_cl/tav_full_audit_v1/test_pil_screen.py`
- `docs/research/pil_screen_001/IMPLEMENTATION_REPORT.md`
- `docs/research/pil_screen_001/TEST_REPORT.md`
- `docs/research/pil_screen_001/RUNBOOK.md`

测试见 [TEST_REPORT.md](TEST_REPORT.md)。手动同步、统一运行、恢复和 SCP 见 [RUNBOOK.md](RUNBOOK.md)。
