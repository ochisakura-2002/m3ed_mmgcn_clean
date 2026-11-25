# G0 跨 session／actor 隔离的 CT-MultiDAG 基线

2026-11-25 · MULTIDAG · Protocol B · 工程准备

G0 在既有正式 Train 上建立同架构、同训练配方、同固定终点的跨群组对照。首轮固定 seed100，留出 Ses03。另三折已冻结，等待首折验收与科研总控决定。

## 证据与冻结边界

11 月 25 日最新版项目源为 `MERC_GPT_PROJECT_SOURCE_UPDATE_20261125_POST_PH_JOINT_AUDIT_G0_REBASE.md`。回传 `post_ph_joint_audit_20261125_101136_analysis.tar.gz` 的 SHA-256 与项目源一致。J/K 的 F0 身份一致，K 清单中的四个表哈希全部匹配。[主模型身份](../post_ph_joint_audit_001/02_CT_MULTIDAG_MAIN_MODEL_IDENTITY.md) · [Local POST-PH 审计](../post_ph_joint_audit_001/00_EXECUTIVE_SUMMARY.md) · [内部开发协议](../post_ph_joint_audit_001/13_INTERNAL_DEV_PROTOCOL.md)

- P、H1 保持关闭。J 中 h0、h4、h0+h4 对 Dev top2 条件下纠错方向的线性 AUROC 为 .5541、.5159、.5415。
- K 前四种历史对照的 cosine 方向 AUROC 为 .4599–.5646。DISTANT_ONLY 的 .7024 来自 19 条方向样本。保留严格过去完整图。
- 已拟合 F0 的 Train 与正式 Dev 的准确率约为 .9157 与 .7071。两侧的置信度、top2 错误组成和类比例也有差异。正式 Dev 的 20 段 Ses04 对话与 Train 的 10 段 Ses04 对话共享 actor。
- G0 用新权重从头训练。固定 `paired_model → FusionTAV → DirectConcat → TrackCore ×4`、T1024/A1582/V342、A/V/T 直拼接、单向文本 LSTM、严格过去图、原分类器、CL_OFF 和 1,638,306 个参数。因果性限定在已供应的 utterance 特征边界。

## 冻结人群

原正式 Train 100 段、4,830 条、Ses01–Ses04 八位 actor。完整对话分配沿用已认证 Protocol B 索引，每段对话恰有一个留出折，四折保留原资产顺序及每段话语 canonical 顺序。

| 留出 session | Fit 对话 / 话语 | 留出对话 / 话语 | 留出支持 hap/sad/neu/ang/exc/fru | Fit 即兴 / 脚本 | 留出即兴 / 脚本 | 更新步/轮 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Ses01 | 72 / 3,457 | 28 / 1,373 | 143/194/384/229/143/280 | 1,992 / 1,465 | 633 / 740 | 5 |
| Ses02 | 70 / 3,474 | 30 / 1,356 | 125/197/362/137/210/325 | 1,947 / 1,527 | 678 / 678 | 5 |
| Ses03 | 68 / 3,261 | 32 / 1,569 | 171/305/320/240/151/382 | 1,747 / 1,514 | 878 / 691 | 5 |
| Ses04 | 90 / 4,298 | 10 / 532 | 5/43/101/105/116/162 | 2,189 / 2,109 | 436 / 96 | 6 |

正式 Dev 仅参与投影身份验收。Ses04 留出折仅含原正式 Train 的 10 段 Ses04 对话。Test 保持 HOLD。训练只接收 fit，第一轮将全部 fit 对话用于拟合，校准留空。显式冻结的身份包括 fit/held-out 对话 ID、actor、类支持、顺序摘要和带 gold 的话语摘要。[四折身份清单](../../../configs/multidag_cl/tav_full_audit_v1/group_isolated_protocol_b/fold_identities.json)

## 训练与评价

五个正式 F0 run 分别使用 seed 42、100、2024、2026、3407，均完成 30 轮、210 步。历史 F0 按正式 Dev W-F1 降序、loss 升序、epoch 升序选取 checkpoint，seed100 选中 epoch 21。G0 保留训练预算，固定 epoch 30，Ses03 共 150 步。训练过程不读 held-out batch，完成后才回放两侧曲线。

| 配方 | 冻结值 |
| --- | --- |
| Optimizer | AdamW，lr 0.0005，betas 0.9/0.999，eps 1e-6，weight decay 0，foreach=False |
| Batch | 每批最多 16 段对话，num_workers=0。训练沿用 CL_OFF 七桶和 seed+epoch shuffle。评价按 canonical 顺序 |
| Loss | 有效话语 masked CE，padding 忽略，class weight 空，label smoothing 0 |
| 正则和控制 | 分类器末线性层前 dropout 0.4，梯度范数裁剪 5，无 scheduler，无 early stopping |
| Epoch 和 seed | 30，seed100，整折从头训练 |

主结果为 epoch 30 的全覆盖 Accuracy、W-F1、Macro-F1、UAR、六类 precision/recall/F1/support、真值为行预测为列的混淆矩阵。两侧分别报告逐轮评价趋势和端点全部预测、logits、概率，并保留 1–29 轮逐话语 logits 和预测。

可靠性以 NLL、多类 Brier（六类平方误差求和）、十等宽置信度分箱 ECE、以 `1 − max(softmax)` 检测错误的 AUROC/平均精度、risk–coverage 和置信度 ≥.90 的错误为辅。risk–coverage 并列分数按冻结顺序展开，另报并列期望风险。SHIFT/STABLE 基于原顺序上一次同 actor gold，首次话语归入 NO_HISTORY。分层是事后统计。无独立校准集，首轮无温标拟合、阈值迁移或拒判决策。

## 推断边界

五 seed 的既有正式结果度量同一人群上对初始化的敏感性。Protocol A 或原 Dev 的整对话留出可以支持对未见对话的观察。Protocol B 的结果把未见 session 与该 session 的未见 actor 一起评价。

Ses03 比原正式训练少 1,569 条话语、32 段对话，更新步从 210 减至 150。Fit 与 held-out 的类先验、表演构成和 session 也同时变化。单折落差无法识别说话人特征的独立效应，Ses04 的 hap 仅 5 条。后续模型对照使用同一 fold、seed、预算和评价身份，已发表或旧 Dev 分数可作为历史背景。拒判曲线上的 accepted risk 和覆盖率属于辅助可靠性结果，全六类分数以全覆盖为准。
