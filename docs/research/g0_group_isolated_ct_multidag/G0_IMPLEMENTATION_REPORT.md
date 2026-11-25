# G0 实现报告

2026-11-25 · MULTIDAG

## 入口与训练机制

`python -m scripts.models.multidag_cl.group_isolated` 提供三个子命令 [入口](../../../scripts/models/multidag_cl/group_isolated.py)。

| 命令 | 实际行为 |
| --- | --- |
| `validate` | 默认只验收 F0 recipe 与四折冻结清单。提供 `--projection-dir` 时认证真实投影并复核当前 fold 的完整数据身份，零训练 |
| `train` | 单折 Ubuntu CUDA，新模型、固定 30 轮。需要 `--execute`，延期折还需同名 `--release-fold` |
| `evaluate` | 完整训练后核验 30 个权重摘要，严格载入 epoch 权重，零更新回放 fit 与 held-out |

F0 构造继续调用 `paired_model`，完整传入归档等价的 F0 配置。构造层保留正式 schema，G0 的 fold、checkpoint 和输出策略由独立运行协议定义。复用 `AuditRuntime` 的 AdamW、masked CE、梯度裁剪、CL_OFF 七桶与确定性训练迭代器，以及原 TAV collate、evaluate 和 checkpoint 原子写入。历史 `formal.execute` 的原 Dev 980 条断言、best-Dev selector 及 `FusionAuditRuntime` 的历史配对训练 tape 只适用于正式 F0。G0 运行直接调度复用的批次更新与评价原语。

每轮检查实际训练到的完整对话及话语次序、总有效 targets、累计 optimizer steps，并保存批次顺序摘要。新 run 目录独占创建，失败标记保持独立。唯一正式端点是 `checkpoints/epoch30_model.pt`，另外 29 份权重支持训练后回放。评价目录一次完整发布。中途训练失败的 run 留作审计，G0 首轮没有 resume 路径。

## 数据合同

代码只接受既有 `tav_train_dev_projection_v1` 的目录，投影两文件哈希及行序摘要与 Ubuntu J 的清单绑定。认证加载器检查字节、split、canonical IDs、speaker、labels、特征维度和音频指纹。正式 Train／Dev 的 population tape 和有标签 tape 分别核验。正式 Dev 只作投影验收。训练入口只物化 fit 对话。复用运行时所需的内部 `dev` split tag 指代本折 held-out 或 fit 的只读评分视图。

G0 路径无原始混合资产构建入口，无历史 checkpoint 路径参数。原正式 F0 配置、源码与权重由原链路保留。本次以归档配置、身份报告与 J/K 已记录的 checkpoint hash 作为历史证据。Windows 无正式 checkpoint 张量和认证投影字节。

## 可回传产物

| 路径（run 内） | 内容 |
| --- | --- |
| `resolved_config.yaml` | G0 配置、正式 F0 recipe 和认证数据身份 |
| `manifests/run_manifest.json` | 状态、参数、初始化、Git/运行环境、步数、HOLD、每个 epoch 权重哈希 |
| `manifests/fit_curriculum_bucket_manifest.tsv`、`logs/fit_updates.csv` | Fit 七桶名册、每轮更新 loss、梯度范数、步数、批次摘要 |
| `reports/evaluation/epoch_trends.csv` | Fit 与 held-out 每轮全覆盖指标 |
| `reports/evaluation/*_epoch30_metrics.json` | 主分类与可靠性指标、混淆矩阵、SHIFT/STABLE/NO_HISTORY |
| `reports/evaluation/*_epoch30_predictions.csv` | 两侧逐样本身份、gold/pred、六列 logits、六列概率与错误标签 |
| `reports/evaluation/trajectory_predictions.csv.gz` | 1–29 轮两侧所有逐话语身份、标签、预测及六列 logits |
| `reports/evaluation/*_high_confidence_errors.csv`、`*_ece10_bins.csv`、`*_error_detection.csv`、`*_risk_coverage.csv` | 可靠性明细 |
| `reports/evaluation/evaluation_manifest.json` | 运行身份、固定端点、零更新计数、输出文件 SHA-256 |

[实验设计](G0_EXPERIMENT_DESIGN.md) · [测试](G0_TEST_REPORT.md) · [文件清单](REQUIRED_MODIFIED_FILES.txt) · [既有工作区修改](UNRELATED_DIRTY_FILES.txt)
