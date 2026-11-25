# G0 Ubuntu 首折运行手册

此手册用于科研审核后的手动执行。首轮是 seed100 留出 Ses03。

## 按审核范围同步

Windows 在 MULTIDAG 工作区运行 guard。按 [REQUIRED_MODIFIED_FILES](REQUIRED_MODIFIED_FILES.txt) 逐路径暂存，核对 `git diff --cached --name-status` 与 `git diff --cached --check`。人工审核后，由用户按现行工作流提交并推送。在 Ubuntu 的 MULTIDAG 工作区查看待合入提交，确认版本后快进。

```bash
git fetch origin feat/multidag-cl-paper-reimplementation
git log --oneline HEAD..FETCH_HEAD
# 确认待合入提交后执行
git merge --ff-only FETCH_HEAD
git status --short
git rev-parse HEAD
python scripts/dev/check_workstream.py multidag
```

## 校验认证投影

在项目环境中，将 `TAV_TRAIN_DEV_PACKAGE` 指向 Ubuntu 已有的三文件认证投影目录 `tav_train_dev_projection_v1`。应包含 `train_dev_tav.pkl`、`model_input_manifest.csv`、`train_dev_provenance.json`。G0 读这三个固定名字，用 J 记录的身份校验输入。输入缺失或哈希不一致时停止本次 run。

```bash
G0_CONFIG=configs/multidag_cl/tav_full_audit_v1/group_isolated_protocol_b/ses03_seed100.yaml
RUN_DATE=$(TZ=Asia/Shanghai date +%Y%m%d)
PROJECTION_DIR="${TAV_TRAIN_DEV_PACKAGE:?set the existing certified Train/Dev projection directory}"

python -B -m scripts.models.multidag_cl.group_isolated validate \
  --config "$G0_CONFIG" --projection-dir "$PROJECTION_DIR"
```

确认输出 `GROUP_ISOLATED_PROTOCOL=PASS`、`projection=VERIFIED`、两类泄漏为 0。Fit 68 段 / 3,261 条，held-out 32 段 / 1,569 条。数据目录保持只读。

## 训练后评价

以下两个命令在同一 shell 使用相同配置、日期、认证投影和可见 GPU。本轮由用户手动启动。

```bash
export CUDA_VISIBLE_DEVICES="${G0_GPU:-0}"
python -B -u -m scripts.models.multidag_cl.group_isolated train \
  --config "$G0_CONFIG" --projection-dir "$PROJECTION_DIR" --experiment-date "$RUN_DATE" --execute

# run manifest 必须是 completed，last_completed_epoch=30，optimizer_steps=150
python -B -u -m scripts.models.multidag_cl.group_isolated evaluate \
  --config "$G0_CONFIG" --projection-dir "$PROJECTION_DIR" --experiment-date "$RUN_DATE"
```

实验根目录为 `outputs/$RUN_DATE/g0_group_isolated_ct_multidag`，run 为 `runs/ct_multidag_g0_ses03_s100`。训练产生约 190 MiB 权重供本机回放。相同 run 目录和已发布的评价目录均拒绝覆盖。若训练中断，保留失败记录，由总控决定独立重试。若回放中断且评价目录未发布，使用同一已完成 run 重做回放。

## 核查与回传

检查 manifest 端点 SHA-256 和 `checkpoints/epoch30_model.pt` 一致，评价更新计数及 HOLD 计数为零。检查趋势共 60 行数据，两份 epoch 30 预测为 3,261 和 1,569 行，轨迹预测为 140,070 行，六类混淆矩阵合计等于各自样本数。

回传 `resolved_config.yaml`、run manifest、fit 七桶名册、训练日志和 `reports/evaluation/`。保留 gzip 的轨迹预测及所有端点 logits。大型权重与原特征留原地。其余三折已经具备同一入口，释放权由首折后的科研审核决定。

```bash
RUN_ROOT="outputs/$RUN_DATE/g0_group_isolated_ct_multidag/runs/ct_multidag_g0_ses03_s100"
TAG=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
mkdir -p tmp
tar -czf "tmp/g0_ses03_seed100_${TAG}_analysis.tar.gz" -C "$RUN_ROOT" \
  resolved_config.yaml manifests/run_manifest.json manifests/fit_curriculum_bucket_manifest.tsv \
  logs/fit_updates.csv reports/evaluation
```
