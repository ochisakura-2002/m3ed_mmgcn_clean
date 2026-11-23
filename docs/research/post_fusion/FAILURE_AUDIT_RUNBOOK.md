# POST_CORE_FAILURE_MECHANISM_AUDIT_001 runbook

本审计使用 CT-MultiDAG F0 的正式 Dev 最佳 checkpoint，固定 seeds 42、100、2024、2026、3407。Ubuntu CUDA 顺序执行五个 seed 的冻结前向。Windows 运行 synthetic/unit tests 和已归档 Dev 预测分析。Test 保持 HOLD。

## Ubuntu

在已激活项目 CUDA 环境的 MULTIDAG 工作区根目录执行。先核对当前分支、本审计提交和远程工作区状态。可通过 ToDesk 传输本次审计的 `.patch.txt`，将 `AUDIT_PATCH` 设为收到的文件路径。若远端已包含审计提交，从设置数据路径的命令继续。保留既有运行目录。

```bash
python scripts/dev/check_workstream.py multidag
git status --short
git apply --stat "$AUDIT_PATCH"
git apply --check "$AUDIT_PATCH"
git am "$AUDIT_PATCH"
```

```bash
python scripts/dev/check_workstream.py multidag
: "${TAV_DATA_ROOT:?set the canonical asymmetric_merc_v1 asset directory}"
FROZEN_F0_RUNS="${FROZEN_F0_RUNS:-outputs/20261123/fusion_final_paired_5seed_confirmation/runs}"
B0_DIR="${B0_DIR:-outputs/20261123/post_fusion_parallel_fastkill_001/analysis/batches/fastkill_seed100_20261123_b0/B0}"
RUN_DATE=$(TZ=Asia/Shanghai date +%Y%m%d)
BATCH="post_core_audit_${RUN_DATE}_$(TZ=Asia/Shanghai date +%H%M%S)"

CUDA_VISIBLE_DEVICES="${AUDIT_GPU:-0}" python -u scripts/analysis/models/multidag_cl/run_post_core_failure_audit.py \
  --frozen-runs "$FROZEN_F0_RUNS" --b0-geometry "$B0_DIR" \
  --asset-path "$TAV_DATA_ROOT/iemocap_asym_tav_t1024_v1.pkl" \
  --model-manifest-path "$TAV_DATA_ROOT/model_input_manifest_tav_t1024_v1.csv" \
  --provenance-manifest-path "$TAV_DATA_ROOT/tav_t1024_provenance_manifest_v1.json" \
  --experiment-date "$RUN_DATE" --batch-id "$BATCH"
```

已有 B0 可通过 `B0_DIR` 指向同一批五个 F0 的汇总目录。缺少 B0 时移除 `--b0-geometry "$B0_DIR"`，报告将记录表示分离证据缺失。主审计读取 B0 小型 CSV，验证 checkpoint 身份和数据身份。显存不足时用新的 batch ID 将 `--chunk-size` 从 8 降到 4 或 2。source SHA、checkpoint payload、canonical Dev 行序和原始预测复现均须通过。以下路径中的 ZIP 为回传包，stdout 提供 SHA-256。

```text
outputs/<YYYYMMDD>/post_core_failure_mechanism_audit_001/analysis/batches/<batch_id>.zip
```

报告与 CSV 位于同名目录。每个表记录 seed 和适用分组。回传包仅包含 CSV、JSON 与 Markdown。中途失败会写入 `failure.json` 并退出，重跑需使用新 batch ID。正式运行成功的四项 Test 计数均为 0。

## Windows

```powershell
python scripts/dev/check_workstream.py multidag
conda run -n m3ed_mmgcn python -m pytest -q tests/models/multidag_cl/tav_full_audit_v1/test_failure_audit.py
conda run -n m3ed_mmgcn python scripts/analysis/models/multidag_cl/run_post_core_failure_audit.py --archived-dev --frozen-runs $env:FROZEN_F0_RUNS --b0-geometry $env:B0_DIR --batch-id archived_dev_round1
```

`--archived-dev` 读取正式 run 的 resolved config、epoch 轨迹、manifest 和已导出的 Dev logits。该模式产出 SHIFT/STABLE、MAD、三种预测不确定性 proxy、五 seed 分歧以及可用的 B0 表。依赖重新前向的分支与路线 eligibility 记录为 `PENDING`，最终路线保持 `UNRESOLVED`。

## 口径

- 自然 SHIFT 由同一 dialogue 中同一 speaker 的上一次 gold class 定义。首位该 speaker 的样本单列，所有 gold 分组均在前向之外。以 Dev 原顺序按样本对齐五个 seed。CSV 中比例以 0–1 表示，`_pp` 字段以百分点表示。分类指标使用固定六类，class 行同时给出在完整分组混淆矩阵中的 precision、recall 和 F1。
- ECCS/EICS 从 Train 的其他 dialogue 抽取联合 T/A/V 特征，匹配 donor 在其原对话中是否为首位 speaker 的角色。目标 speaker 结构、位置数和当前原始 T/A/V 固定。每组 donor 对五个模型 seed 共用，重复种子为 1701、1702、1703。各目标只前向截至目标的 prefix，因果文本编码可以响应被替换的过去。每次替换及三次均值均落盘。DCLF 提供扰动诊断的来源。 [DCLF](https://aclanthology.org/2025.coling-main.272/)
- 图历史条件固定已编码的全部节点表示。对每个目标构造由目标及选定历史组成的诱导子图，因此被排除节点无法通过多跳图传播返回。TrackCore 保留内部 self 与 residual。CURRENT_ONLY 的文本表示仍包含因果 LSTM 已编码的过去。RECENT_ONLY 取最近 4 个历史位置，DISTANT_ONLY 取更早位置。历史相关性的研究动机来自对目标证据与上下文原因的区分，以及对历史相关度的过滤。 [ECERC](https://aclanthology.org/2025.acl-long.102/) [HiRoC](https://link.springer.com/article/10.1007/s44443-026-01163-x)
- Helpful/harmful 使用相应方向的正确性翻转或大于 `1e-6` 的 logit margin 变化。两个方向的正确性翻转率单列。STRONG 要求至少 4/5 seed 同时达到 helpful、harmful 各 10%，且同一组对比的 helpful-minus-harmful 差异同方向达到 5 pp。候选组对比为 SHIFT 对 STABLE，或 same-favored 对 other-favored。每个组须覆盖至少 5% Dev。same/other-favored 表示对应 sender 组的限制条件保留了更高的真类 margin。
- MAD 按任务给出的六类公式计算。归一化 entropy 与反向 top-2 margin 使用同方向的等频 quartile，精确并列时使用冻结 Dev 顺序。五 seed 预测熵以自然对数计。Route U 还要求至少 4/5 seed 的 MAD Q4 残余误率高于 Q1。残余错误指五个受限图历史条件均未恢复，且三个 ECCS 重复中至多一次恢复的原始错误。EICS 故意换成异标签历史，因此只进入 label-replication 诊断。MAD 是基于模型分布的诊断 proxy。 [UAF](https://aclanthology.org/2026.aacl-long.111/)
- 模态从生产 F0 `active_modalities` 入口置零不活跃的编码块。完整、单模态和 drop 条件均使用同一个 checkpoint。single-modality oracle 是诊断上界。若模型结构无法提供已有的七条件语义，模态分支标记 `NOT_AVAILABLE`。STRONG 使用 15% 可恢复误差门槛及至少 5 pp 的 disagreement 与 all-agree 误率差，同一个 seed 需同时满足两项，且达到 4/5。
- SHIFT STRONG 要求五 seed 平均 W-F1 和误率差各至少 5 pp，且至少 4/5 seed 两项同时同方向。观测性 SHIFT 支持证据矩阵，H eligibility 依赖 ECCS/EICS 或选择性历史门槛。Train→Dev 只比较 MAD、绝对图历史 logit margin 变化、图历史预测翻转和单模态分歧，并附同 checkpoint 的 B0 摘要。
- 多条路线同时 eligible 时，以同 seed 百分点效应做优先级比较。H 使用通过门槛的 EICS/ECCS 降幅差或双向正确性翻转率较小值中的较大者。U 使用 Q4−Q1 残余错误差。M 使用可恢复比例与 disagreement 误率差中的较小值。4/5 seeds 差至少 5 pp 时选更强路线。4/5 seeds 差低于 5 pp 时按 H > U > M。缺少一致胜出的路线时，按中位数效应取前两条，生成一个零训练补检验计划。无 eligible 路线时记录 `UNRESOLVED`。
