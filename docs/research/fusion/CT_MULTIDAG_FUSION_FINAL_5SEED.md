# CT-MultiDAG Fusion 三臂五随机种子最终确认

阶段 `FUSION_FINAL_PAIRED_5SEED_CONFIRMATION`。正式确认只执行 F0、H4_FULL_HYBRID 和 H2_AUDIO_TEXT。H3_ASYMMETRIC、Fusion-A、Fusion-C 及历史 Fusion 搜索的正式入口已关闭，历史配置仍可静态核验。

## Seed 与训练身份

`TAV_MULTISEED_CONFIRM_MATRIX.csv` 中 `track_tav_cloff` 对应冻结的 Track、TAV、CL_OFF、`past_all_causal` baseline。launcher 验证完整历史矩阵和源 YAML，从其中读取 `42, 100, 2024, 2026, 3407`，并记录矩阵行的 SHA-256。历史定义缺失或冲突时 `FAIL-STOP`。每个 seed 均重新执行三臂，包括 seed 100。

15 份实际配置由三个 seed100 topology YAML 解析生成，继承原架构、人口和训练协议。每个 seed 的候选复用 F0 backbone 初始化、两条共同边的 H4 初始化、RNG 协议和 Train/Dev 顺序。launcher 对比元素级最大绝对差、初始 RNG、各轮 Train RNG 边界和每个 batch 的身份。Test 保持 HOLD，数据入口只创建 Train 和 Dev dataset。

## Ubuntu 手动执行

在 MULTIDAG 工作区根目录，从已激活 CUDA 训练环境的 shell 运行。资产变量沿用 seed100 topology retrain，分别指向认证的 TAV feature pickle、model manifest CSV 和 provenance JSON。

```bash
git pull --ff-only && python scripts/dev/check_workstream.py multidag
```

同步和 guard 均成功后执行。省略 `--execute` 可查看计划，且无需提供资产路径。

```bash
RUN_DATE=$(TZ=Asia/Shanghai date +%Y%m%d)
python scripts/workflows/benchmarks/run_ct_multidag_fusion_final_5seed.py --execute \
  --asset-path "${TAV_ASSET:?}" \
  --model-manifest-path "${TAV_MODEL_MANIFEST:?}" \
  --provenance-manifest-path "${TAV_PROVENANCE_JSON:?}" \
  --batch-id "fusion_final_5seed_${RUN_DATE}" --experiment-date "$RUN_DATE"
```

首次运行前会检查批次、15 个 run 和归档路径的冲突。成功时自动写入 `tmp/CT_MultiDAG_Fusion三臂五随机种子最终确认_分析数据包_<DATE>.tar.gz`。分析文件也保存在 `outputs/<DATE>/fusion_final_paired_5seed_confirmation/analysis/batches/<batch_id>/`。`ANALYSIS_BUNDLE = PASS` 和 `PLOT_READY = YES` 表示 15 份结果已通过最佳 checkpoint、Dev 预测、配对与打包 SHA-256 校验。预测 CSV 包含六类 logits 与 probabilities，混淆矩阵同时保存计数和按真实类别归一化的数据。

## 判定与单位

各 run 的原始分数是 0–1，配对差值为候选减同 seed F0，以 pp 表示。统计对五个 seed 等权，SD 使用样本标准差。STRONG / KEEP 使用 +0.20 pp、至少四个 seed 非负且 Macro-F1 和 UAR 均值不降的联合条件。

稳定类别平衡改善定义为至少四个相同 seed 的 Macro-F1 和 UAR 同时严格上升，且两项平均差值均为正。稳定 WF1 wins 定义为至少四个 seed 严格胜过 F0。TRADE-OFF 与 STOP 可同时成立，报告会保留两项判定。两候选都满足 STOP 时冻结 Fusion 路线。归档仍保留全部候选与 seed 的数据。
