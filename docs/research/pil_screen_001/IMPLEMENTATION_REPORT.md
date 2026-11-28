# PIL_SCREEN_001 本地实施报告

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
