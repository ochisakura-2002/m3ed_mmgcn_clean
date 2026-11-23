# Fast-Kill Ubuntu 用户手动执行

从已激活项目 CUDA 环境的 MULTIDAG 仓库根目录执行。先核对分支、remote、工作区状态和远端更新范围。若远端包含已审核的 Fast-Kill 提交且本地可快进，在该分支同步后执行 guard。

```bash
BRANCH=feat/multidag-cl-paper-reimplementation
git branch --show-current
git remote -v
git status --short
git fetch origin "$BRANCH"
git log --oneline HEAD..FETCH_HEAD
# 审完变更范围并确认是预期提交后执行以下两行。
git merge --ff-only FETCH_HEAD
python scripts/dev/check_workstream.py multidag
```

```bash
: "${TAV_DATA_ROOT:?set the canonical IEMOCAP asymmetric_merc_v1 asset directory}"
TAV_ASSET="$TAV_DATA_ROOT/iemocap_asym_tav_t1024_v1.pkl"
TAV_MODEL_MANIFEST="$TAV_DATA_ROOT/model_input_manifest_tav_t1024_v1.csv"
TAV_PROVENANCE_JSON="$TAV_DATA_ROOT/tav_t1024_provenance_manifest_v1.json"
FROZEN_F0_RUNS="${FROZEN_F0_RUNS:-outputs/20261123/fusion_final_paired_5seed_confirmation/runs}"
RUN_DATE=$(TZ=Asia/Shanghai date +%Y%m%d)
BATCH="fastkill_seed100_${RUN_DATE}"
ENTRY=scripts/workflows/benchmarks/run_ct_multidag_post_fusion_fastkill.py

python "$ENTRY" --config-only --experiment-date "$RUN_DATE" --batch-id "$BATCH"
```

```bash
python "$ENTRY" --a0 --experiment-date "$RUN_DATE" --batch-id "${BATCH}_a0" \
  --model-manifest-path "$TAV_MODEL_MANIFEST" --provenance-manifest-path "$TAV_PROVENANCE_JSON"
```

```bash
CUDA_VISIBLE_DEVICES="${FASTKILL_GPU:-0}" python "$ENTRY" --b0 --diagnostic-device cuda \
  --experiment-date "$RUN_DATE" --batch-id "${BATCH}_b0" --frozen-runs "$FROZEN_F0_RUNS" \
  --asset-path "$TAV_ASSET" --model-manifest-path "$TAV_MODEL_MANIFEST" \
  --provenance-manifest-path "$TAV_PROVENANCE_JSON"
```

确认 A0 通过。单卡执行时先完成 B0。两张独占 GPU 时，先在另一张卡启动 B0，再提交正式 launcher。

```bash
CUDA_VISIBLE_DEVICES="${FASTKILL_GPU:-0}" python "$ENTRY" --execute \
  --experiment-date "$RUN_DATE" --batch-id "$BATCH" \
  --asset-path "$TAV_ASSET" --model-manifest-path "$TAV_MODEL_MANIFEST" \
  --provenance-manifest-path "$TAV_PROVENANCE_JSON"
```

一张 V100 下依次执行。正式 launcher 在同一 CUDA 进程中按 CTRL→B1→A1 使用一张 GPU。B0 可在另一张独占 GPU 并行执行。若 A0 未通过后缀审计，停止正式 launcher 并查看 manifest 错误；B0 输出无须作为正式训练参数。已存在的 run ID 或批次不会被覆盖，本轮入口按整批启动，仅支持新 run。

训练输出位于 `outputs/<YYYYMMDD>/post_fusion_parallel_fastkill_001/runs/<arm>_seed100/`。Analysis Bundle 位于同组的 `analysis/batches/<batch_id>/`，归档在 `tmp/post_fusion_parallel_fastkill_001_<YYYYMMDD>_<batch_id>.tar.gz`。回传归档和独立 B0 的 CSV/NPZ 及 manifest。检查 batch 输出的 `ANALYSIS_BUNDLE = PASS`、`PLOT_READY = YES` 和 archive SHA。 `route_decisions.json` 给出两个决策，`paired_dev_deltas.csv` 含最佳 Dev 选点和最后一轮的差值。
