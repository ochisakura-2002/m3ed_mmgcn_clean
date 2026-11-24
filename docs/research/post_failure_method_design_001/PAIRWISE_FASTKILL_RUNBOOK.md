# CT-MultiDAG seed100 Pairwise Fast-Kill · Ubuntu Runbook

Windows 交接记录 `FORMAL_TRAIN_RUNS = 0`，`TEST_TOUCHED = NO`。下列正式运行命令由用户在 Ubuntu MULTIDAG 工作区的项目 CUDA 环境中前台执行。

## 同步和确认

在 Windows 核对当前任务列出的文件，按路径提交并推送到 MULTIDAG 分支。在 Ubuntu 原工作区查看 incoming commits。

```bash
git status --short
git fetch origin feat/multidag-cl-paper-reimplementation
git log --oneline HEAD..FETCH_HEAD
```

确认内容和运行状态后快进，并记录部署 commit。

```bash
git merge --ff-only FETCH_HEAD &&
  python scripts/dev/check_workstream.py multidag &&
  git rev-parse HEAD
```

快进和 guard 都通过后再执行以下步骤。正式部署通过 Git commit 同步。

## 绑定冻结资产并运行

| 变量 | 目录 |
| --- | --- |
| `FROZEN_F0_RUNS` | 历史 Fusion 五 seed 正式 runs 根目录，包含 `tav_track_fusion_final_f0_s100`。固定 epoch 21，693/980 正确，Dev W-F1 69.9627%。 |
| `TAV_TRAIN_DEV_PACKAGE` | 已通过 Preflight 的独立认证投影，包含 `train_dev_tav.pkl`、`model_input_manifest.csv`、`train_dev_provenance.json`。 |
| `ROUND1_DIR` | 零训练 Preflight 用过的 Round 1 已解包批次目录，直接包含 `failure_audit_manifest.json`、`AMBIGUITY_AUDIT.csv` 及 `HISTORY_UTILITY_AUDIT.csv`。 |

在同一个 shell 中给 `TAV_TRAIN_DEV_PACKAGE` 和 `ROUND1_DIR` 赋已验证的目录值。历史 runs 在标准位置时可使用下面的默认值。设置 `FASTKILL_GPU` 可指定空闲可见 GPU 的编号，默认 0。

```bash
(
  set -euo pipefail
  FROZEN_F0_RUNS="${FROZEN_F0_RUNS:-outputs/20261123/fusion_final_paired_5seed_confirmation/runs}"
  : "${TAV_TRAIN_DEV_PACKAGE:?set the certified Train/Dev package path}"
  : "${ROUND1_DIR:?set the verified Round 1 batch path}"
  test -d "$FROZEN_F0_RUNS/tav_track_fusion_final_f0_s100"
  test -f "$TAV_TRAIN_DEV_PACKAGE/train_dev_provenance.json"
  test -f "$ROUND1_DIR/failure_audit_manifest.json"

  RUN_STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
  RUN_DATE="${RUN_STAMP%%_*}"
  BATCH="seed100_fastkill_$RUN_STAMP"
  CUDA_VISIBLE_DEVICES="${FASTKILL_GPU:-0}" python -B -u \
    scripts/analysis/models/multidag_cl/run_pairwise_fastkill.py \
    --config configs/multidag_cl/tav_full_audit_v1/pairwise_fastkill_seed100.yaml \
    --frozen-runs "$FROZEN_F0_RUNS" \
    --train-dev-package "$TAV_TRAIN_DEV_PACKAGE" \
    --round1-dir "$ROUND1_DIR" \
    --experiment-date "$RUN_DATE" --batch-id "$BATCH"
)
```

入口先验证冻结 checkpoint、Train/Dev population、历史 Dev 输出和 Preflight 门控与 Oracle，随后执行一次固定的 seed100 训练。重复 batch ID 在任何 epoch 前拒绝。身份、人口、数值、因果、旁路或 HOLD 失败会写 `REVIEW_INVALID`，退出码为 2。只按错误阶段修复运行问题。修复后按原方案执行时使用新的 batch ID。

## 核对并回传

```text
outputs/<YYYYMMDD>/pairwise_fastkill_seed100/analysis/batches/<batch_id>/
outputs/<YYYYMMDD>/pairwise_fastkill_seed100/analysis/batches/<batch_id>.zip
```

有效终点检查 stdout 的 `F0_IDENTITY = PASS`、`HEAD_EPOCH = 20`、`TEST_TOUCHED = NO` 和 `HANDOFF_READY = YES`。从 `run_manifest.json` 核对 1 次 Train/Dev frozen dataset pass、1 次 PRIMARY Dev 评分、缓存身份、head SHA-256 和资源报告。完整 Dev 980 条、冻结 Q4 245 条，检查各自 R、D、NET。`test_hold_status.json` 中五项 HOLD 计数为 0。

回传 `<batch_id>.zip` 及 stdout 的 `FASTKILL_ARCHIVE_SHA256`。ZIP 内含表格、判定、报告和清单。正式 `pairwise_head_epoch20.pt` 留在同一 Ubuntu batch 目录，仅以路径和 SHA-256 写入清单。输入投影、F0 checkpoint、原特征、缓存与 h0/h4 向量留在 Ubuntu。

| 判定 | 后续 |
| --- | --- |
| STOP | 关闭 PRIMARY P。下一路线才可独立考虑 Selective History Routing。 |
| REVIEW_INVALID | 核对并修复运行身份或执行问题，保留已冻结的规程。 |
| REVIEW_SIGNAL | 仅复核已有输出。复核后仍未达 GO 则关闭五 seed 资格。 |
| GO | 单独申请五 seed 确认。 |

```text
FORMAL_TRAIN_RUNS = 0
TEST_TOUCHED = NO
```
