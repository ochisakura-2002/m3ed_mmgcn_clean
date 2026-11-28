# PIL_SCREEN_001 运行手册

本批次在 `MULTIDAG` 工作流内执行，使用 Ubuntu V100 16GB 的原 `m3ed_mmgcn` Conda 环境。Windows 交付负责代码、测试及同步说明。正式实验的起点是下述远端预检。

## Windows 手动提交

在 `E:\MERC\m3ed_mmgcn_clean` 的 PowerShell 执行。先检查暂存区为空，再逐项暂存。下面的 push 明确指向本实验分支。

```powershell
python scripts/dev/check_workstream.py multidag
if ($LASTEXITCODE -ne 0) { throw 'MULTIDAG workstream check failed' }
git status --short --branch
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'Review existing staged changes before creating the PIL commit' }
$pilFiles = @(
  'configs/multidag_cl/pil_screen/seed100.yaml'
  'models/experimental/multidag_cl_pil_screen.py'
  'scripts/runtime/multidag_cl_pil_screen.py'
  'scripts/workflows/benchmarks/run_pil_screen.py'
  'scripts/analysis/models/multidag_cl/pil_screen_analysis.py'
  'tests/models/multidag_cl/tav_full_audit_v1/test_pil_screen.py'
  'docs/research/pil_screen_001/IMPLEMENTATION_REPORT.md'
  'docs/research/pil_screen_001/TEST_REPORT.md'
  'docs/research/pil_screen_001/RUNBOOK.md'
)
git add -- $pilFiles
if ($LASTEXITCODE -ne 0) { throw 'PIL staging failed' }
git diff --cached --name-status
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'PIL staged diff check failed' }
git commit -m "Implement PIL_SCREEN_001 matched C01 and C07 screening"
if ($LASTEXITCODE -ne 0) { throw 'PIL commit failed' }
git rev-parse HEAD
git push origin HEAD:refs/heads/feat/multidag-cl-paper-reimplementation
```

保存 `git rev-parse HEAD` 的完整 SHA，供下节同步核验。

## Ubuntu 同步和预检

通过已有授权 SSH 主机登录。输入上节生成的完整提交 SHA 后，同步到当前分支。同步前查看 `git status --short --branch`，保留既有文件和运行资产。

```bash
set -euo pipefail
cd /home/zhiyuan/research_lh/m3ed_mmgcn_clean
conda activate m3ed_mmgcn
python scripts/dev/check_workstream.py multidag
git status --short --branch
read -r -p 'PIL commit SHA: ' PIL_COMMIT
[[ "$PIL_COMMIT" =~ ^[0-9a-f]{40}$ ]]
git fetch origin refs/heads/feat/multidag-cl-paper-reimplementation
git merge-base --is-ancestor "$PIL_COMMIT" FETCH_HEAD
git merge --ff-only "$PIL_COMMIT"
test "$(git rev-parse HEAD)" = "$PIL_COMMIT"
python scripts/dev/check_workstream.py multidag

PIL_DATE="$(TZ=Asia/Shanghai date +%Y%m%d)"
PIL_BATCH="pil_screen_001_seed100_$(TZ=Asia/Shanghai date +%H%M%S)"
PIL_PROJECTION=/home/zhiyuan/research_lh/data/IEMOCAP/asymmetric_merc_v1/tav_train_dev_projection_v1
PIL_MODULE=scripts.workflows.benchmarks.run_pil_screen
PIL_ARGS=(--projection-dir "$PIL_PROJECTION" --experiment-date "$PIL_DATE" --batch-id "$PIL_BATCH")
PIL_GROUP="outputs/$PIL_DATE/pil_screen_001"
PIL_LOG="$PIL_GROUP/logs/launcher/$PIL_BATCH/execution.log"
PIL_STATE="$PIL_GROUP/manifests/batches/$PIL_BATCH/state.json"
PIL_REPORT="$PIL_GROUP/analysis/batches/$PIL_BATCH"
PIL_PACKAGE="$PIL_GROUP/analysis/batches/MERC_PIL_SCREEN_001_${PIL_BATCH}_analysis.tar.gz"
printf 'PIL_DATE=%s PIL_BATCH=%s\n' "$PIL_DATE" "$PIL_BATCH"

python -B -u -m "$PIL_MODULE" preflight "${PIL_ARGS[@]}" \
  2>&1 | tee "${TMPDIR:-/tmp}/${PIL_BATCH}_preflight.log"
```

GPU 可见设备、CUBLAS 工作区配置、MKL/线程环境及其实际报告须与历史 Stage A 一致。预检核对已审阅 Stage A 的 SHA、D 源码策略、实际导入闭包、投影数据 SHA、初始化、完整原 30 轮采样带、V100 上的同形状合成前向以及已有绘图栈。若历史 D0 可复用，预检报告 `CERTIFIED_REUSE`。若只有 D0 工件认证无法完成，报告 `ONE_SHARED_OFF_REQUIRED` 和具体原因。历史 F0 原始日志或 E21 证据、Stage A 环境身份、数据或源码如无法通过，保留退出信息回审。

`PIL_PREFLIGHT=PARTIAL` 表示固定 C07-NULL 置乱没有改变任何 Train 目标。记录将 C07 两臂判为 INVALID，C01 可继续。其他预检错误会以非零状态退出。

## 一次运行全部批准轨迹和分析

在同一个 shell 中沿用预检变量。以下单条命令自动认证 D0、必要时先执行一次 OFF、按预登记顺序运行候选、分析和封包。

```bash
python -B -u -m "$PIL_MODULE" run "${PIL_ARGS[@]}"
```

在另一个 SSH 会话或现有终端复用打印的日期、批次 ID，检查日志和状态。每轮日志打印 Dev W-F1，正式计算每臂固定 30 轮。查看中间记录用于运行排障。

```bash
tail -F "$PIL_LOG"
python -m json.tool "$PIL_STATE"
```

如果需要把启动终端断开，可在首次正式启动时使用 `nohup`，并在新会话通过上述状态文件监控。以下是同一启动命令的后台形式，两种形式选其一。

```bash
nohup python -B -u -m "$PIL_MODULE" run "${PIL_ARGS[@]}" \
  >"${TMPDIR:-/tmp}/${PIL_BATCH}_launch.log" 2>&1 </dev/null &
printf 'PID=%s\n' "$!"
```

## 同批次恢复

恢复前确认早先进程已退出，并保持相同工作树、源码、环境、日期、batch ID 和投影。已写入 `result.json` 的臂只验收已有结果。

```bash
# Recover an interrupted fit from its original last_model.pt.
python -B -u -m "$PIL_MODULE" resume-training "${PIL_ARGS[@]}"

# Recover export or publication after all approved arms have outcomes.
python -B -u -m "$PIL_MODULE" resume-analysis "${PIL_ARGS[@]}"
```

训练恢复以最近一个原子提交的完整 epoch 为边界，恢复主模型、C07 头、AdamW、Best、CPU/CUDA RNG 与配对带。首轮提交之前的失败、共享 CTRL 数值偏离、配对身份或其它科学门禁失败应以原批次回执和日志回审。完成或标记 `invalid` 的批次保留其证据。

## 验收和回传

批次完成时，程序打印 `ANALYSIS_EXPORT=COMPLETE; ANALYSIS_BUNDLE=PASS; PLOT_READY=YES`，并报告压缩包绝对路径、大小和 SHA256。

```bash
python -m json.tool "$PIL_STATE"
cat "$PIL_REPORT/tables/verdicts.csv"
(cd "$PIL_REPORT" && sha256sum -c SHA256SUMS.txt)
stat -c '%s %n' "$PIL_PACKAGE"
sha256sum "$PIL_PACKAGE"
tar -tzf "$PIL_PACKAGE"
```

从 Windows PowerShell 使用已配置的 Ubuntu SSH 主机名。下面通过交互输入拿到本次实际路径，下载后对照 Ubuntu 输出的 SHA256。

```powershell
$pilSshHost = Read-Host 'Existing Ubuntu SSH host or alias'
$pilRemotePackage = Read-Host 'Absolute archive path printed by the launcher'
scp "${pilSshHost}:${pilRemotePackage}" .
if ($LASTEXITCODE -ne 0) { throw 'PIL bundle transfer failed' }
$pilDownloaded = Split-Path -Leaf $pilRemotePackage
Get-FileHash -Algorithm SHA256 -LiteralPath $pilDownloaded
tar -tzf $pilDownloaded
```

模型、优化器和部署权重留在各正式 run 的 `checkpoints/`。回传包保存权重路径与 SHA、Best/E30、源证书、运行身份、表和图。可从解压目录重绘 `python pil_screen_analysis.py .`。独立复核以未舍入指标、同 ID 的 Best 对比及 `tables/verdicts.csv` 为依据。复核后可使用包内 `PROJECT_SOURCE_UPDATE.md` 更新项目源。
