# CT-MultiDAG Post-Fusion Fast-Kill 设计

2026-11-23 · WORKSTREAM = MULTIDAG · seed100 · Train / Dev · Test HOLD

## 证据和冻结身份

以桌面的 `fuision阶段项目源.md`、`交接文档.md` 和当前归档为准。读过本地的 `POST_FUSION_ROUTE_AUDIT_001.md`、`ORIGINAL_MULTIDAG_FUTURE_WORK_TRACE.md`、`NEXT_ROUTE_CANDIDATE_MATRIX.csv` 和 `TOP2_ROUTE_RECOMMENDATION.md`。本次任务授权进入两个独立 seed100 试验。11-19 GraphCore Freeze/Fusion Rebase、2026-09-16 项目更新、2026-08-18 baseline freeze、独立实验回传规范及历史全桶 Train/Val 回传仍为 `EVIDENCE_NOT_VERIFIED`。当前项目源和交接中的回传规则适用于本轮。

CTRL 复用最终 Fusion 确认的 F0，`85590fbf97d0c5aad0c5547142a98fc20e342dc4`。文本 T1024 经 Linear100 和 packed UniLSTM100，A1582 和 V342 分别经 Linear100 和 ReLU。按 A/V/T 拼为 300D，走 `past_all_causal` 的四层 TrackCore，加 h0 拼为 1500D，经 1500→300→300→6 分类器及 masked CE。CL_OFF。冻结推理参数 1,638,306。历史确认种子为 42、100、2024、2026、3407，本轮三臂均从 seed100 F0 同一初始参数和 RNG 开始。

保留固定 Train 100 对话／4,830 个目标与 Dev 20 对话／980 个目标。每轮 7 步，30 轮合计 210 步和 144,900 个训练目标。Batch 16，AdamW LR 0.0005，betas (0.9, 0.999)，eps 1e−6，weight decay 0，clip 5。选点按 Dev W-F1 降序、Dev loss 升序、epoch 升序。原冻结 CL_OFF 会依据整对话 DMF 建立 shuffle 的初始索引，三臂继承同一批次顺序。对 A1 的前缀声明适用于新增权重。

## B1_CLASS_ANCHOR

在同一次既有分类器 forward 中捕获第二个 ReLU 之后、dropout 之前的 300D 有效话语表示。六个训练原型用独立 CPU RNG（seed101）生成标准正态向量并做 L2 归一化。对全部六类计算 `cos(z_i,p_c)/0.1`，在 valid target 上取原型 CE 均值。总损失是原 masked CE 加 `0.1 × L_anchor`。原型加入同一 AdamW 参数组和全局 clip，增加 1,800 个训练参数，原模型持有 1,638,306 个参数，weights-only checkpoint 沿用原模型 schema。

`tau=0.1` 来自 EACL 的 IEMOCAP 设置，余弦 softmax 对应 EACL Eq. 9。其 Eq. 8 系数针对其他项，本轮 `lambda_anchor=0.1` 采用已发表的加性 CE + prototype 辅助项系数作固定量级先例。两个值均在当前 Dev 运行前固定。 [EACL Eq. 8–9 和 Table 6](https://aclanthology.org/2024.findings-naacl.282.pdf) · [Chen 等的加性原型系数，§3.1](https://paper-snitch.ing.unimore.it/paper/4957/pdf/)

B1 主门采用最佳 Dev checkpoint，相对新跑 CTRL 的 W-F1 ≥ +0.30 pp、Macro-F1 ≥ +0.20 pp、UAR ≥ 0 pp。hap/exc、neu/fru 至少一对的两方向真类归一化误率均值相对下降 ≥10%，两条误率各自最多上升 1 pp，两个类各自的 recall 最多下降 1 pp。该量化沿用路线推荐中预先提出的方向检查。未通过任一门槛即 `B_ROUTE=STOP`。ang/fru 为 B0 机制诊断项。

## A1_PREFIX_WES · PROJECT_ADAPTATION

ECAI 2025 Eq. (18) 在两类 valence 乘积为正时取 `max(cos(theta),0)`，乘积为负取 0，为零取 `1/6`。Eq. (19) 的 shift 权重为 `k × similarity + b`，IEMOCAP 固定 `k=1, b=0.4`，据 Table 6 和作者 `cl.py`。使用作者情绪轮中 hap/sad/ang/exc/fru 的角度 (3,29,11,7,21) 倍 π/20，neu 位于原点。每位 speaker 维护最近一次情绪，只对不同标签计 shift。 [论文 §3.5.1、Eq. 18–20 和 Table 6](https://journals.sagepub.com/doi/10.3233/FAIA251292) · [作者情绪轮](https://github.com/LiXinran6/LSDGNN_ICL/blob/main/similarity_matrix.py) · [作者 WES 实现](https://github.com/LiXinran6/LSDGNN_ICL/blob/main/cl.py)

本项目将累积窗口截到当前话语 `t`。`D_t = (sum(WES through t) + speakers_seen) / (t + speakers_seen)`。A0 用项目原 DMF 对完整对话等权计数，并比较各位置的等权前缀和 WES 前缀。

固定 `alpha_t(e) = 1 - 0.10 × max(0, (8-e)/7) × D_t/1.4`。1.4 是单次 WES 的理论上界。早期权重至少为 0.90，epoch 8 起精确为 1。训练损失为 `sum(alpha_i × CE_i) / n_valid`，以原 masked CE 的 valid-target 个数保留聚合尺度。每个 alpha 由本话语的前缀和 epoch 决定。每轮仍可见全量 Train。

A1 主门采用第 30 轮的 Dev 结果，相对 CTRL 的同轮 W-F1 ≥ +0.30 pp、Macro-F1 ≥ 0 pp、UAR ≥ 0 pp，同样导出最佳 checkpoint 作对照。每轮更新数、有效目标数和批次顺序必须匹配，后缀检查必须通过。协议失败时 A1 无效，性能门槛未达为 `A_ROUTE=STOP`。

YAML 只允许 CTRL、B1、A1 三种组合。通过门槛只意味着候选可进入后续确认。当前 Dev 已参与历史模型选择，seed100 只用于 Fast-Kill。
