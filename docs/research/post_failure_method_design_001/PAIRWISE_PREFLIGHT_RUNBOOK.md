# CT-MultiDAG seed100 零训练 Preflight 手动交接

## 1. Windows 提交范围

在 MULTIDAG 工作区执行 guard。只暂存实现报告中的 8 个路径。先确认暂存区没有其它任务的改动，再由用户提交和推送。

```powershell
python scripts/dev/check_workstream.py multidag
git diff --cached --name-status
$preflightFiles = @(
  'scripts/analysis/models/multidag_cl/failure_audit_inference.py',
  'scripts/analysis/models/multidag_cl/pairwise_preflight_inputs.py',
  'scripts/analysis/models/multidag_cl/pairwise_preflight_metrics.py',
  'scripts/analysis/models/multidag_cl/run_pairwise_preflight.py',
  'tests/models/multidag_cl/tav_full_audit_v1/test_pairwise_preflight.py',
  'docs/research/post_failure_method_design_001/PAIRWISE_PREFLIGHT_IMPLEMENTATION_REPORT.md',
  'docs/research/post_failure_method_design_001/PAIRWISE_PREFLIGHT_TEST_REPORT.md',
  'docs/research/post_failure_method_design_001/PAIRWISE_PREFLIGHT_RUNBOOK.md'
)
git add -- $preflightFiles
git diff --cached --name-status
git diff --cached --check
```

暂存范围与上面的列表一致后，用户执行以下命令。

```powershell
git commit -m 'Add frozen seed100 F0 pairwise preflight'
git push origin HEAD:refs/heads/feat/multidag-cl-paper-reimplementation
```

## 2. Ubuntu 确认并快进

在正确的 MULTIDAG 工作区与项目 CUDA 环境中，查看待合入提交。工作区有冲突或无法快进时停止同步。

```bash
git fetch origin feat/multidag-cl-paper-reimplementation
git log --oneline HEAD..FETCH_HEAD
```

确认待合入提交后执行。

```bash
git merge --ff-only FETCH_HEAD
python scripts/dev/check_workstream.py multidag
```

## 3. 指定冻结基线及已验资产

`FROZEN_F0_RUNS` 指向历史 Fusion 最终五 seed 批次的 `runs` 目录。唯一正式来源是 `tav_track_fusion_final_f0_s100`，固定第 21 轮，Dev W-F1 69.9627%，693/980 正确。入口检查 resolved YAML、`run_manifest.json`、`validation_best_dev_evidence.json`、30 轮 epoch 日志、`best_model.pt` 字节和重放预测。历史 checkpoint SHA-256 为 `997e7db663e3a3c2b8cb9ea34957e8fdbda909593ae7ad8360bc14ef1e3022ac`。

`TAV_TRAIN_DEV_PACKAGE` 指向由数据持有方认证的独立 Train/Dev 包。包沿用历史 F0 的 T/A/V 浮点特征、原 split dialogue 顺序、ID、说话人及 gold。代码从认证路径读取以下三个固定文件。

| 文件 | 内容 |
| --- | --- |
| `train_dev_tav.pkl` | 可信 canonical TAV 结构，`splits` 只有 `train` 和 `dev`，`dialogues` 只含这两组。保留 `schema=iemocap_asym_tav_t1024_v1`、`dataset_version=asymmetric_merc_v1`、原 `feature_spec`、`label_vocab.itos`。每段的 T/A/V、`utterance_ids`、`speakers`、`benchmark_labels`、`benchmark_label_ids` 保留原行索引与字节语义。 |
| `model_input_manifest.csv` | 对应原 canonical manifest 的 Train/Dev 行，保持 source 行序，含 `datasets/iemocap/tav_full_audit_dataset.py` 中 `MANIFEST_FIELDS` 所列列名。Train 为 100 段、4,830 条，Dev 为 20 段、980 条。 |
| `train_dev_provenance.json` | `schema=ct_multidag_train_dev_projection_v1`、`splits=["train","dev"]`、两组 `split_counts`、历史 run manifest 的完整 `data_identity` 作为 `source_data_identity`、以实际文件名为键的两个 `sha256`、该投影视图的 `row_order_identity_sha256` 与 HOLD counters。 |

`row_order_identity_sha256` 使用 `datasets.iemocap.tav_full_audit_dataset.identity_sha` 计算，仅输入已独立的 Train/Dev CSV 行。认证清单写 `test_policy=HOLD`，并将 `test_loader_built`、`test_labels_inspected`、`test_forward_count`、`test_evaluation_count`、`test_checkpoint_selection_count` 写为整数 0。来源哈希和投影视图哈希均会留档。数据持有方应已认证两个 split 中的每个特征块来自原冻结资产，代码另行验证 A1582 指纹与完整 Dev 输出。若目前只有混合 split pickle，不能作为本入口输入。正式运行需等到认证 Train/Dev 包可用。

可选 `ROUND1_DIR` 指向已有解包的 Failure Audit Round 1 批次目录。读取其中的 `failure_audit_manifest.json`、`AMBIGUITY_AUDIT.csv` 和可用的 `HISTORY_UTILITY_AUDIT.csv`。先验 checkpoint、数据身份、文件 SHA-256、全量 Dev keys、gold、预测和值域都必须对齐。未提供或不可对齐时，相关分层记为 `NOT_AVAILABLE`，Q4 使用历史归档预测的 Failure Audit 稳定排序。

## 4. 用户运行冻结前向

```bash
RUN_DATE=$(TZ=Asia/Shanghai date +%Y%m%d)
BATCH="seed100_preflight_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
FROZEN_F0_RUNS="${FROZEN_F0_RUNS:-outputs/20261123/fusion_final_paired_5seed_confirmation/runs}"
ROUND1_ARGS=()
if [[ -n "${ROUND1_DIR:-}" ]]; then
  ROUND1_ARGS=(--round1-dir "$ROUND1_DIR")
fi

CUDA_VISIBLE_DEVICES="${PREFLIGHT_GPU:-0}" python -u \
  scripts/analysis/models/multidag_cl/run_pairwise_preflight.py \
  --frozen-runs "$FROZEN_F0_RUNS" \
  --train-dev-package "${TAV_TRAIN_DEV_PACKAGE:?set the certified Train/Dev-only package path}" \
  --batch-id "$BATCH" --experiment-date "$RUN_DATE" "${ROUND1_ARGS[@]}"
```

运行只打开 Train/Dev 两个 loader。先验证身份并复现完整 Dev，再做一次冻结 Train 前向。历史 batch size 16 固定。下一次运行须使用新的 batch ID。

## 5. 验收和轻量回传

用户从 stdout 检查 `F0_IDENTITY`、`TRAIN_GATE_SUPPORT`、`DEV_GATE_SUPPORT`、`F0_DEV_ERRORS`、`TRUE_TOP2_RATE_IN_GATE`、`RECOVERABLE_F`、`ORACLE_DELTA_WF1_PP`、`ZERO_TRAINING_PREFLIGHT` 和 `TEST_TOUCHED`。身份失败时查看 `PREFLIGHT = REVIEW_INVALID_IDENTITY` 与 `PAIRWISE_PREFLIGHT_DECISION.json` 原因。有效 PASS/STOP 均产生 `HANDOFF_READY = YES` 与归档 SHA-256。`INVALID` 返回 2。

输出位于以下目录。

```text
outputs/<YYYYMMDD>/pairwise_preflight_seed100/analysis/batches/<batch_id>/
outputs/<YYYYMMDD>/pairwise_preflight_seed100/analysis/batches/<batch_id>.zip
```

回传 `<batch_id>.zip` 和 stdout 的 `PREFLIGHT_ARCHIVE_SHA256`。ZIP 只含预定统计 CSV、报告 MD 和身份/判定 JSON。`run_manifest.json` 记录每个归档文件的 SHA-256。未定义的条件比率留空，分层不可用有显式状态行。正式数据、特征、权重及表示向量均留在原资产位置。

只有 `RECOVERABLE_F ≥ 8` 且 `ORACLE_DELTA_WF1_PP ≥ 0.75` 才是 PASS。PASS 只证明理论纠错余量。STOP 结束该判断，保留原方法定义。批准后续才可开始复判头与 Fast-Kill 任务。
