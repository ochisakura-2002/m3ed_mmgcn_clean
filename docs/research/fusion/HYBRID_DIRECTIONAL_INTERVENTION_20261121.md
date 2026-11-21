# CT-MultiDAG Hybrid 定向边干预

2026-11-21 · WORKSTREAM = MULTIDAG · TEST = HOLD

冻结 `tav_track_hybrid_seed100` 的 best epoch 19，在正式 Dev 的 20 段对话、980 句上比较八个条件。H0 的四项汇总指标须复现该 run manifest 的正式 Dev 指标。所有差值均为 intervention 减 H0，以百分点报告。筛查结果为 F0 WF1 69.9627、Macro 68.4974、UAR 68.2616，Hybrid WF1 69.9001、Macro 68.6883、UAR 69.1201，Fusion-A WF1 69.1175，Fusion-C WF1 69.5515。筛查决定保留 Hybrid 做机制诊断，停止 A 和 C。

| 干预 | 关闭的有向边 |
| --- | --- |
| H0_FULL | 无 |
| H1_DROP_T_FROM_A | T ← A |
| H2_DROP_T_FROM_V | T ← V |
| H3_DROP_A_FROM_T | A ← T |
| H4_DROP_V_FROM_T | V ← T |
| H5_DROP_VISUAL_PAIR | T ← V、V ← T |
| H6_DROP_AUDIO_PAIR | T ← A、A ← T |
| H7_ALL_FUSION_OFF | 全部四条 |

在 `eval` 和 `inference_mode` 下，临时 hook 将指定 matcher 的门控消息置零，然后沿用原模型的接收端求和、残差缩放和下游图。所有条件保留 AVT 输入、参数和有效位置。H7 衡量 Hybrid 已训练 backbone 上移除全部跨模态消息的效应。正式 F0 是独立训练结果。

## Logger 口径和假设

边的 `message_receiver_ratio_mean` 是有效话语上 `‖sigmoid(score) · W_value(source)‖₂ / max(‖receiver‖₂, 1e-8)` 的均值。它在同一接收端的消息求和与 `0.1` 缩放前计算。`residual_receiver_ratio_median` 对有效的 A、V、T 接收块取中位数，分子是缩放后的合并残差范数。关闭边的 gate 仍由 matcher 计算，其消息和比值为零。边比值受接收块范数影响，两种统计量均需结合性能变化解读。

Epoch 19 已观察到 T ← A 约 0.608、T ← V 约 0.053、A ← T 约 0.244、V ← T 约 5.128，整体实际残差比中位数约 0.0538。报告会并列正式 epoch logger 和 H0 重算值。

- H1 若导致性能下降，支持 Audio 对 Text 的有效补充。
- H2 若汇总变化很小，支持当前 V → T 的下游贡献较弱。结合类别差值检查可能抵消的变化。
- H4 若提高 WF1，同时恢复 fru 和 sad，支持当前 T → V 条件化可能带来冗余或类别冲突。
- H5 若优于 H0，支持下一版优先验证精简视觉交互。H3 和 H6 分别给出 A ← T 与音频双向边的固定 checkpoint 效应。

结论限于该 seed、该 Hybrid checkpoint 和该 Dev。δ 的正负描述本次移除效应，单次诊断无法确定跨 seed 稳定性或重新训练后的结果。

## Ubuntu 手动运行

在 MULTIDAG 工作区执行以下命令。资产路径来自正式 `resolved_config.yaml`。`run_manifest.json` 的 `best_checkpoint_sha256` 绑定 `best_model.pt`，脚本核对该 SHA 和 epoch，并逐张量及额外状态核对 `best_model_weights_only.pt`。随后仅用 weights-only 权重进行推断。脚本核对训练时记录的 Dev 人口和批序列摘要。发布资产为共用 pickle 容器，整包按文件 SHA 校验并反序列化，适配器只读取 Dev 特征和标签，其他 split 仅保留身份元数据。

```bash
git pull --ff-only
python scripts/dev/check_workstream.py multidag
python scripts/diagnostics/models/multidag_cl/run_hybrid_directional_intervention.py --update-report
```

诊断写入 `outputs/20261121/fusion_seed100_screen_20261121/analysis/hybrid_directional_intervention/` 下的 `hybrid_directional_summary.csv`、`hybrid_directional_per_class.csv`、`hybrid_directional_manifest.json`。`--update-report` 将实测表格和解读写入本报告。默认输出目录已存在时，可用 `--output-dir` 指向新的同组 analysis 子目录。

## 实测结果

<!-- hybrid-directional-results -->

待 Ubuntu 手动运行。当前仅记录正式筛查结果、冻结规程与判读假设。
