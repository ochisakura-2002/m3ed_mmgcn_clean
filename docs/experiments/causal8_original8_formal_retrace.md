# Causal-8 + Original-MERC-8 正式重跑说明

## 目的与边界

`scripts/experiments/run_causal8_original8_formal.py` 是一个固定版本顺序的
16-run 正式批处理器。它不改变模型数学实现、数据划分、训练超参数或正式
YAML，也不删除或迁移旧 `outputs`。Causal 配置需要接收批次级 device、输出根
和冻结日期，因此 launcher 只在本批次 manifest 内生成运行时 YAML 副本；源
配置保持只读。

## 16 组构成与顺序

索引 1--8 是 Causal Session-Holdout：

1. MMGCN / Validation Ses01
2. MMGCN / Validation Ses02
3. MMGCN / Validation Ses03
4. MMGCN / Validation Ses04
5. MultiDAG+CL / Validation Ses01
6. MultiDAG+CL / Validation Ses02
7. MultiDAG+CL / Validation Ses03
8. MultiDAG+CL / Validation Ses04

这些任务由 `scripts/run_experiment_pipeline.py` 执行。

索引 9--16 是 Original MERC screening：

9. MMGCN Legacy
10. MultiDAG+CL Legacy
11. DialogueGCN Legacy
12. GS-MCC Legacy
13. MMGCN Clean
14. MultiDAG+CL Clean
15. DialogueGCN Clean
16. GS-MCC Clean

这些任务由 `scripts/baselines/train_original_merc_baseline.py` 执行，并显式接收
`--device`、`--output-root` 和 `--experiment-date`。

## 启动前门禁

launcher 在第一个训练进程启动前检查：

- 16 个源配置及两个入口脚本都存在；
- Causal pipeline 引用的 train YAML 存在且特征声明一致；
- 请求的设备是 CUDA，且 PyTorch 能检测到 CUDA；
- Legacy 和 Clean PKL 均存在，实际 SHA256 与固定期望值一致；
- `outputs/<date>` 可写；
- Git commit 和工作区状态可记录；
- 当前输出根下没有另一个该类型批次持有活动锁（即使日期参数不同也不能并发）。

公共门禁失败不会启动任何训练。单个训练进程失败则记录状态；默认继续下一组。

## 日期锁定

`experiment_date` 在 launcher 启动时只解析一次。随后：

- 主进程设置 `MERC_EXPERIMENT_DATE`；
- 每个子进程继承相同环境变量；
- 每条命令显式携带同一 `--experiment-date`；
- Causal 运行时配置也写入同一日期。

因此即使批次跨过午夜，16 个 run 仍全部位于同一个
`outputs/<experiment_date>/` 下。

## 远程 V100 启动

正式训练前应先在本地完成 commit/push，并在远程切换到对应 commit。示例：

```bash
cd /path/to/m3ed_mmgcn_clean
git rev-parse HEAD
DATE=$(date +%Y%m%d)
mkdir -p "outputs/${DATE}/logs"
nohup env PYTHONUNBUFFERED=1 python -u \
  scripts/experiments/run_causal8_original8_formal.py \
  --experiment-date "${DATE}" \
  --device cuda \
  --output-root outputs \
  --start-index 1 \
  --end-index 16 \
  --continue-on-error \
  --poll-seconds 5 \
  > "outputs/${DATE}/logs/causal8_original8_formal.nohup.log" 2>&1 &
echo $!
```

launcher 不依赖 tmux。查找批次主日志：

```bash
find "outputs/${DATE}/logs" -path \
  '*/causal8_original8_formal_*/launcher.log' -print
```

## 每 epoch 输出与 heartbeat

子进程 stdout 单独写入：

```text
outputs/<date>/logs/<batch_id>/<index>_<label>.log
```

launcher 不依赖 stdout 判断训练进度。它用启动前快照和配置签名识别当前新
`run_dir`，轮询其中的 `logs/epoch_metrics.csv`。每个新 CSV 行只输出一次：

```text
[EPOCH][03/16][causal_mmgcn_ses03] epoch=7 train_loss=... val_loss=...
```

监控器读取实际 CSV 表头，只显示当前存在的 epoch、train、validation 和学习率
字段；某个预期字段缺失不会中止监控。

每 60 秒还会输出一条 `[HEARTBEAT]`，包含索引、标签、耗时、进程 PID/存活
状态、GPU 利用率/显存、已发现 epoch 行数和当前 run 目录。

## 断点补跑

若第 6 组失败，而 1--5 已完成，可新建一个补跑批次：

```bash
python -u scripts/experiments/run_causal8_original8_formal.py \
  --experiment-date 20260722 \
  --device cuda \
  --output-root outputs \
  --start-index 6 \
  --end-index 16 \
  --continue-on-error
```

只补第 6 组时把 `--end-index` 也设为 6。使用
`--no-continue-on-error` 可在首个单组失败后停止，未执行项保持
`NOT_STARTED`。

## 批次目录与状态文件

批次 ID 形如：

```text
causal8_original8_formal_<YYYYMMDD_HHMMSS>
```

目录为：

```text
outputs/<date>/manifests/<batch_id>/
outputs/<date>/logs/<batch_id>/
outputs/<date>/reports/<batch_id>/
```

manifest 目录包含：

- `planned_runs.tsv`：固定 16 项计划、源配置、实际启动配置和命令；
- `run_manifest.tsv`：只包含本批次实际识别到的新 run；
- `run_status.tsv`：16 项状态与 artifact 检查；
- `run_dirs.txt`：本批次新 run 的明确 allowlist；
- `launcher_metadata.json`：日期、参数、主机、Python、Git 与特征信息；
- `git_state.txt`：commit 和启动时工作区状态；
- `feature_sha256.txt`：两个 PKL 的期望与实际 SHA；
- `resolved_configs/`：Causal 任务的本批次运行时配置副本；
- `markers/`：每个已启动任务的时间标记。

报告目录包含 `final_summary.json`。单组状态取值为 `PASS`、
`FAILED_PROCESS`、`FAILED_ARTIFACTS`、`FAILED_NUMERIC`、`INTERRUPTED` 或
`NOT_STARTED`。

## 后续分析与审计包

后续分析、绘图和审计必须以该批次的 `run_manifest.tsv` 或 `run_dirs.txt` 为
allowlist，不得重新递归整个 `outputs` 后按名称猜测。尤其不得把旧 run、smoke
或其他 pipeline 历史结果补入本批次。

可导出小型调度审计包：

```bash
DATE=20260722
BATCH=causal8_original8_formal_20260722_120000
tar -czf "${BATCH}_audit.tar.gz" \
  "outputs/${DATE}/manifests/${BATCH}" \
  "outputs/${DATE}/logs/${BATCH}" \
  "outputs/${DATE}/reports/${BATCH}"
```

checkpoint 和完整 run 目录不应放入 Git；如需服务器侧归档，严格按
`run_dirs.txt` 中的路径另行打包。

## 为什么不能删除旧 outputs 后再验证

旧输出包含历史 invalid run、有效替代关系、协议迁移记录和复现实验依据。
删除它们会破坏审计链，也无法证明新 launcher 正确隔离了旧 run。正确验证方式
是保留旧目录，通过启动前快照、配置签名和批次 manifest 证明本批次只记录新
run；隔离能力必须在旧结果仍存在时成立。
