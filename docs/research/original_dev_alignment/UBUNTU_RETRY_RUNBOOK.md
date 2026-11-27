# Ubuntu / V100 重试手册

同步文件清单里的已审核代码后，在 MULTIDAG 的 Ubuntu 工作区根目录进入原 Stage A Conda 环境。保留 Stage A 使用的 CUDA 可见设备、CUBLAS 与 CPU 线程环境。以下变量沿用实际历史路径。

```bash
set -euo pipefail
python scripts/dev/check_workstream.py multidag
git rev-parse HEAD
printf '%s  %s\n' \
  68668add5ff0cc6bd15bbbdf56e7b42b25f8c1956d19fbf43842c4a4c1bbec11 \
  scripts/workflows/benchmarks/original_dev_source_compatibility.json | sha256sum -c -
: "${F0_REFERENCE_RUN:?set the archived F0 seed100 run path}"
: "${TAV_TRAIN_DEV_PACKAGE:?set the certified Train/Dev projection directory}"

module=scripts.workflows.benchmarks.run_original_dev_alignment
batch=d_alignment_seed100_132745
root=outputs/20261127/original_dev_alignment
approved=0b0938bddb9cc99a7075896af0fa90b87ea068cfbbe5c3336d48c51d726235e6
common=(
  --reference-run "$F0_REFERENCE_RUN"
  --projection-dir "$TAV_TRAIN_DEV_PACKAGE"
  --experiment-date 20261127
  --batch-id "$batch"
  --approved-stage-a-sha256 "$approved"
)
if [[ -n "${D_ALIGN_CJK_FONT:-}" ]]; then
  common+=(--figure-font "$D_ALIGN_CJK_FONT")
fi
evidence="$(mktemp -d "${TMPDIR:-/tmp}/original_dev_retry.XXXXXX")"
printf 'Logs: %s\n' "$evidence"
```

## 命令一 · 无训练预检

```bash
python -B -u -m "$module" stage-b "${common[@]}" --preflight-only 2>&1 | tee "$evidence/preflight.log"
```

预检成功会打印 `STAGE_B_PREFLIGHT=PASS`。预检核验当前 Ubuntu 环境、原回执与预算锁、投影、D0 权重及 Best 复现、D1 配对、新源码认证、历史失败和绘图依赖。原 Stage A 和当前 Stage B 产物以只读方式检查。

## 命令二 · 正式联合执行

```bash
python -B -u -m "$module" stage-b "${common[@]}" 2>&1 | tee "$evidence/stage_b.log"
```

该命令在锁内重新执行预检，成功后才归档已确认的旧失败并进入 D1。完成后自动运行 CE/LS 回放、D2/D3/D4、统计、作图和打包。旧空分析目录、旧失败回执及后续失败的分析产物归档在 `$root/manifests/batches/$batch/stage_b_attempts/`。此前已经移到 `tmp/d_alignment_seed100_132745_failed_stage_b_a1j6fixi` 的尝试保留原位置。

```bash
report="$root/analysis/batches/$batch"
package="$root/analysis/batches/${batch}_stage_b.tar.gz"
(cd "$report" && sha256sum -c SHA256SUMS.txt)
sha256sum "$package"
tar -tzf "$package"
```

### 从完整 D1 恢复

```bash
python -B -u -m "$module" stage-b "${common[@]}" --resume-analysis 2>&1 | tee "$evidence/resume.log"
```

该入口强制要求可认证的完整 D1。若作图或归档失败，复验 D1 后从回放开始。若分析目录或包已经完成封存，校验其封存哈希后继续发布。检测到未知路径、训练锁、缺失权重或不完整训练时，会停止并保留证据。报告的路径可用 `stat`、`find -maxdepth 2` 和 `sha256sum` 只读定位，在处理之前确认旧启动进程已退出。

### Ubuntu MKL 定位与旧 D0 的复用边界

若环境仍失败，在同一 shell 开两个独立 Python 进程做对照。

```bash
python -B -m "$module" environment-probe > "$evidence/training_imports.json"
python -B -m "$module" environment-probe --probe-plot-first > "$evidence/legacy_plot_imports.json"
diff -u "$evidence/training_imports.json" "$evidence/legacy_plot_imports.json" || [[ $? -eq 1 ]]
```

对照两条 `torch_build`、`math_libraries`、导入及线程环境，并以完整预检在首次 D0 初始化后采集的值为正式判定。历史 Stage A 回执记录了 MKL 报告，历史进程映射需另找原环境证据。若正常启动仍出现真实差异，继续冻结 D1，先查明历史环境的可恢复性。

只有确认旧 D0 无法复用后，才考虑在新批次重新配对。按以下命令生成候选 Stage A，保留原批次。

```bash
: "${HISTORICAL_G0_CE_ROOT:?set the same historical G0 CE source used for the 90-update lock}"
fresh_date="$(TZ=Asia/Shanghai date +%Y%m%d)"
fresh_batch="d_alignment_seed100_repair_$(TZ=Asia/Shanghai date +%H%M%S)"
fresh_common=(
  --reference-run "$F0_REFERENCE_RUN"
  --projection-dir "$TAV_TRAIN_DEV_PACKAGE"
  --experiment-date "$fresh_date"
  --batch-id "$fresh_batch"
)
python -B -u -m "$module" stage-a "${fresh_common[@]}" \
  --historical-ce-root "$HISTORICAL_G0_CE_ROOT"
```

以新批次回执重新审核完整历史数值身份和 90 步预算。审核通过后，用 `fresh_common` 和新审批 SHA 运行 Stage B 两条命令。若新 Stage A 因数值变化失败，配对实验的可比性需要单独评审。
