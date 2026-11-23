# Fast-Kill 实现报告

2026-11-23 · WORKSTREAM = MULTIDAG

| 实验臂 | 训练改变 | 部署模型参数 | 额外训练参数 |
| --- | --- | ---: | ---: |
| CTRL | 原 masked CE | 1,638,306 | 0 |
| B1_CLASS_ANCHOR | masked CE + cosine anchor CE | 1,638,306 | 1,800 |
| A1_PREFIX_WES | 逐 valid target 加权 masked CE | 1,638,306 | 0 |

沿用 `FusionTAV`、`FusionAuditRuntime` 的初始化、采样和 Train/Dev pairing，以及正式 TAV 运行器的选点、日志和结构化 checkpoint。公共运行器新增默认关闭的实验 metadata 入口，训练参数列表的默认来源仍为原模型。B1 在独立的运行时持有原型，只有 Train forward 期间安装短期激活 hook。原型随结构化训练 checkpoint 存储，weights-only artifact 与模型 `state_dict` 保持原 schema。A1 权重由 A0 审核后固定的每对话前缀分数与 epoch 生成。两臂训练同时开启会被拒绝。

正式 launcher 在输出冲突检查后载入认证资产，执行零训练 A0，按 CTRL、B1、A1 顺序运行。每轮比较已观察到的 ordered dialogue / utterance IDs、batch SHA 和训练 RNG 边界。选点后严格重载最佳模型权重，导出 Dev 预测。B0 是单独的诊断入口，B1 额外导出新 checkpoint 的 Train/Dev 表征。

Analysis Bundle 提供三个 resolved YAML、run manifest、epoch history 和 Train/Dev loss、Dev Accuracy / W-F1 / Macro-F1 / UAR、逐类 precision / recall / F1、逐话语 Dev logits/probabilities、raw/normalized confusion、optimizer step 与 target 计数、checkpoint 和模型参数身份、Test HOLD。B1 附上 CE/anchor loss、原型范数和成对余弦、300D 几何 CSV/NPZ。A1 附上逐轮 alpha 分位数、逐类有效权重和真实 A0 原始前缀统计。打包前重新检查选点、预测与指标一致性及配对预算，归档逐文件校验 SHA-256。

正式资产沿用项目既有的完整 TAV pickle 和 manifest SHA 校验。Test 字段随整包反序列化，逻辑数据入口只建立 Train/Dev view。Test 标签与特征没有被 dataset 取用，Test loader、评估和选点计数都为零。Windows 本次只使用 synthetic dataset 和已审核的历史 Dev 预测。

[设计](POST_FUSION_PARALLEL_FASTKILL_001_DESIGN.md) · [测试](PARALLEL_FASTKILL_TEST_REPORT.md) · [运行手册](PARALLEL_FASTKILL_RUNBOOK.md)
