# CT-MultiDAG seed100 H1 Fast-Kill · Ubuntu

## 输入

在 Ubuntu MULTIDAG 工作区的项目 CUDA 环境执行下方命令。先核对待同步提交与远端工作区，再快进到包含 H1 的提交。Windows 验收仅用合成输入和既有 Dev 恢复 ZIP。

| 环境变量 | 冻结输入 |
| --- | --- |
| `FROZEN_F0_RUNS` | 包含 `tav_track_fusion_final_f0_s100` 的历史 runs 目录。F0 固定 epoch 21，Dev W-F1 69.9627%。 |
| `TAV_TRAIN_DEV_PACKAGE` | 认证的独立 Train/Dev 投影目录，带 `train_dev_provenance.json`、`train_dev_tav.pkl` 和 `model_input_manifest.csv`。 |
| `HISTORY_TOP25_RECOVERY_DIR` | 正式 `history_top25_seed100_20261124_212530` 解包目录。入口逐文件与 ZIP 比对。 |
| `HISTORY_TOP25_RECOVERY_ARCHIVE` | 与解包目录相邻的同名 `.zip`，可显式指定副本。SHA-256 为 `519740deb419683d978c1eadca5879855d8818e9fb3c4bdc34e628e91a229d3e`。 |

## 唯一正式运行

```bash
(
  set -euo pipefail
  python scripts/dev/check_workstream.py multidag
  git rev-parse HEAD
  : "${TAV_TRAIN_DEV_PACKAGE:?set the certified Train/Dev projection directory}"
  FROZEN_F0_RUNS="${FROZEN_F0_RUNS:-outputs/20261123/fusion_final_paired_5seed_confirmation/runs}"
  HISTORY_TOP25_RECOVERY_DIR="${HISTORY_TOP25_RECOVERY_DIR:-outputs/20261124/history_top25_provenance_recovery/analysis/batches/history_top25_seed100_20261124_212530}"
  HISTORY_TOP25_RECOVERY_ARCHIVE="${HISTORY_TOP25_RECOVERY_ARCHIVE:-${HISTORY_TOP25_RECOVERY_DIR}.zip}"
  printf '%s  %s\n' '519740deb419683d978c1eadca5879855d8818e9fb3c4bdc34e628e91a229d3e' "$HISTORY_TOP25_RECOVERY_ARCHIVE" | sha256sum --check
  RUN_STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
  CUDA_VISIBLE_DEVICES="${H1_GPU:-0}" python -B -u \
    scripts/analysis/models/multidag_cl/run_history_fastkill.py \
    --config configs/multidag_cl/tav_full_audit_v1/history_fastkill_seed100.yaml \
    --frozen-runs "$FROZEN_F0_RUNS" --train-dev-package "$TAV_TRAIN_DEV_PACKAGE" \
    --recovery-dir "$HISTORY_TOP25_RECOVERY_DIR" --recovery-archive "$HISTORY_TOP25_RECOVERY_ARCHIVE" \
    --experiment-date "${RUN_STAMP%%_*}" --batch-id "history_fastkill_seed100_$RUN_STAMP"
)
```

入口在训练前验证恢复包、认证投影、F0 checkpoint、严格 causal probe 和同批次旁路/零初始化 exact identity。确认 stdout 先出现 `F0_IDENTITY = PASS` 与两个零 lambda。训练继承 F0 batch size 16，seed100，普通 masked CE，固定 Adam 及 20 个 epoch。Dev 在训练后只做一次 PRIMARY 前向与评分。重复 batch ID 在任何 epoch 前退出。身份或运行门禁失败返回 `REVIEW_INVALID` 和退出码 2，查同批次 `run_manifest.json` 的 `failed_stage` 和 `reason`。

## 回传

```text
outputs/<YYYYMMDD>/history_fastkill_seed100/analysis/batches/<batch_id>/
outputs/<YYYYMMDD>/history_fastkill_seed100/analysis/batches/<batch_id>.zip
```

回传 ZIP 和 stdout 的 `HISTORY_FASTKILL_ARCHIVE_SHA256`。核对 `primary_epoch=20`、`primary_dev_evaluation_count=1`、`dev_used_for_model_selection=false`、`H_SUBGROUP_PROVENANCE=PASS` 和五项 Test HOLD 计数。ZIP 只含本轮 CSV、报告和清单。两个参数的 `history_attenuation_epoch20.pt` 留在 Ubuntu。

| 判定 | 含义 |
| --- | --- |
| `STOP` | 关闭当前 H1 路线和五 seed 资格。 |
| `REVIEW_SIGNAL` | 仅复核本轮结果。复核后未达到 GO 时关闭五 seed 资格。 |
| `GO` | 允许后续单独安排五 seed 确认。 |
| `REVIEW_INVALID` | 按阶段核查身份、人口、因果、数值或执行证据。 |
