# CT-MultiDAG PRIMARY P 零训练 Preflight 实现报告

`WORKSTREAM = MULTIDAG` · 2026-11-24 · Windows 实现与合成验收

## 实现范围

- 新分析入口调用历史 F0 验证器，只加载 seed100。固定验证第 21 轮、历史 checkpoint SHA-256、源提交、完整 resolved config、TAV 特征维度、六类顺序、DirectConcat、TrackCore、`past_all_causal`、选点日志与 F0 Dev 证据。Dev 的 980 条记录按 ID、dialogue、目标和 batch tape 复现，且预测相同、logits 在既有审计容差内。身份失败返回 `REVIEW_INVALID_IDENTITY`，停止之后的分析。
- 前向沿用 `FusionTAV`、checkpoint 严格加载、canonical 数据校验和 collator。分析层以 `inference_mode()` 读取有效行的 logits 和 Train/Dev 目标。阶段结果只记录聚合诊断。正式入口要求 Ubuntu CUDA，Windows 验证使用合成数据。
- MAD 与四分位排序复用 Failure Audit 数值实现，分类指标复用项目六类报告。Train 阈值用第 `ceil(0.75·N_train)` 个 MAD。类别 ID 为 logits 降序的稳定 tie-break。Gate 严格使用 `u≥τ` 和 `0<m<1`，Oracle 只改 gate 内 `y=top2` 的 F0 错误。
- PASS 仅由 `F≥8` 和未舍入的 Oracle ΔW-F1 `≥+0.75 pp` 共同决定。既有设计草案中的 Train 支持计数在本阶段只作诊断。所有 15 个类对、30 个方向和 6 类均保留，包括零支持行。
- Q4 优先复用已验证的 Round 1 冻结成员。缺归档时，使用同一历史 F0 Dev logits、原顺序和原稳定排序重新取得 245 条。此口径以 `Q4_SOURCE` 标记。SHIFT 和 U/H 分层仅在逐样本键、身份及归档校验通过时输出；U/H 重建还须复现 Round 2 seed100 的捕获计数。缺项记录 `NOT_AVAILABLE`。
- 运行按 batch ID 隔离。PASS/STOP 都可生成固定清单的 CSV、JSON、MD 压缩包，包内字节及输出 SHA-256 可核对。INVALID 日志保留原因与 HOLD 状态，无可行性统计或成功归档。

## 正式执行的输入边界

现有共享 TAV pickle 含 Test 对话，正常 runtime 会反序列化整个容器。此入口接收与历史 F0 身份绑定的独立 Train/Dev 包，校验清单和 SHA-256 后才创建两个 dataset。正式运行需要 Ubuntu 数据持有方已有或单独提供的认证包。Windows 本轮没有验证 Ubuntu 上该包是否存在，未读取原始资产、checkpoint 或历史逐样本预测。正式 F0 身份与 PASS/STOP 由 Ubuntu 运行验证。[Runbook](PAIRWISE_PREFLIGHT_RUNBOOK.md)

## 本轮文件

| 操作 | 路径 |
| --- | --- |
| 修改 | `scripts/analysis/models/multidag_cl/failure_audit_inference.py` |
| 新增 | `scripts/analysis/models/multidag_cl/pairwise_preflight_inputs.py` |
| 新增 | `scripts/analysis/models/multidag_cl/pairwise_preflight_metrics.py` |
| 新增 | `scripts/analysis/models/multidag_cl/run_pairwise_preflight.py` |
| 新增 | `tests/models/multidag_cl/tav_full_audit_v1/test_pairwise_preflight.py` |
| 新增 | `docs/research/post_failure_method_design_001/PAIRWISE_PREFLIGHT_IMPLEMENTATION_REPORT.md` |
| 新增 | `docs/research/post_failure_method_design_001/PAIRWISE_PREFLIGHT_TEST_REPORT.md` |
| 新增 | `docs/research/post_failure_method_design_001/PAIRWISE_PREFLIGHT_RUNBOOK.md` |

`failure_audit_inference.py` 增加可选 seed 子集，原调用沿用默认确认批次的五 seed。分析模块没有复判头、训练目标或 history routing。正式训练和 Test 操作均为 0。Git 由用户手动控制。
