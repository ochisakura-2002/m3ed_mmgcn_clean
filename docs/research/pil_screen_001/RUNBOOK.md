# PIL_SCREEN_001 运行手册

2026-11-28 · MULTIDAG · 当前状态 *REMOTE_PREFLIGHT_PENDING*。用户手动提交并同步，随后在唯一授权 Ubuntu 工作区运行真实预检。保留远端原 D、F0、PIL 批次及日志。

## 1. Windows 手动提交和推送

在 MULTIDAG 工作区 PowerShell 执行。暂存区已有条目时先由用户核查。以下清单包含本次兼容修复和规范落地的八个文件。

```powershell
conda run -n m3ed_mmgcn python -B scripts/dev/check_workstream.py multidag
if ($LASTEXITCODE -ne 0) { throw 'MULTIDAG guard failed' }
git --no-pager status --short --branch
if ($LASTEXITCODE -ne 0) { throw 'Git status failed' }
git --no-pager diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'Review the existing index before staging PIL changes' }
$pilFiles = @(
  'AGENTS.md'
  'docs/workflows/MERC_CODEX_WORK_STANDARD_V1_0_20261128.md'
  'docs/research/pil_screen_001/IMPLEMENTATION_REPORT.md'
  'docs/research/pil_screen_001/TEST_REPORT.md'
  'docs/research/pil_screen_001/RUNBOOK.md'
  'docs/research/pil_screen_001/UNRELATED_DIRTY_FILES.txt'
  'scripts/workflows/benchmarks/run_pil_screen.py'
  'tests/models/multidag_cl/tav_full_audit_v1/test_pil_history_contract.py'
)
git --no-pager add -- $pilFiles
if ($LASTEXITCODE -ne 0) { throw 'Path-specific staging failed' }
$pilStaged = git --no-pager diff --cached --name-only
if ($LASTEXITCODE -ne 0) { throw 'Cannot read staged paths' }
if (Compare-Object ($pilFiles | Sort-Object) ($pilStaged | Sort-Object)) { throw 'Unexpected staged paths' }
git --no-pager diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Staged diff check failed' }
git --no-pager diff --cached --stat
if ($LASTEXITCODE -ne 0) { throw 'Cannot review staged diff' }
```

核对暂存内容后执行，保留打印的完整 SHA。

```powershell
git --no-pager commit -m "Fix PIL D0 preflight compatibility and adopt MERC work standard"
if ($LASTEXITCODE -ne 0) { throw 'PIL commit failed' }
$pilCommit = git rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $pilCommit -notmatch '^[0-9a-f]{40}$') { throw 'Cannot determine approved SHA' }
$pilCommit
git --no-pager push origin HEAD:refs/heads/feat/multidag-cl-paper-reimplementation
if ($LASTEXITCODE -ne 0) { throw 'PIL push failed' }
```

## 2. Ubuntu 快进同步

确认本工作区当前无使用现行源码的研究进程。在新的交互式 Bash 中执行，沿用默认 shell 选项。核对实际待合入提交和文件后确认快进。

```bash
pil_sync() {
  cd /home/zhiyuan/research_lh/m3ed_mmgcn_clean || return 1
  conda activate m3ed_mmgcn || return 1
  python -B scripts/dev/check_workstream.py multidag || return 1
  git --no-pager status --short --branch || return 1
  local changes fetched answer
  changes="$(git --no-pager status --porcelain=v1 --untracked-files=normal)" || return 1
  [[ -z "$changes" ]] || { printf 'Review Ubuntu worktree changes before sync.\n' >&2; return 1; }
  read -r -p 'Approved Windows commit (40 hex): ' PIL_COMMIT || return 1
  [[ "$PIL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || return 1
  git --no-pager fetch origin refs/heads/feat/multidag-cl-paper-reimplementation || return 1
  fetched="$(git rev-parse --verify FETCH_HEAD)" || return 1
  git merge-base --is-ancestor "$PIL_COMMIT" "$fetched" || return 1
  git merge-base --is-ancestor HEAD "$PIL_COMMIT" || return 1
  git --no-pager log --oneline "HEAD..$PIL_COMMIT" || return 1
  git --no-pager diff --name-status HEAD "$PIL_COMMIT" || return 1
  read -r -p 'Verified commits and paths? Type yes to fast-forward: ' answer || return 1
  [[ "$answer" == yes ]] || return 1
  git --no-pager merge --ff-only "$PIL_COMMIT" || return 1
  [[ "$(git rev-parse --verify HEAD)" == "$PIL_COMMIT" ]] || return 1
  python -B scripts/dev/check_workstream.py multidag || return 1
  test -f scripts/workflows/benchmarks/run_pil_screen.py &&
    test -f tests/models/multidag_cl/tav_full_audit_v1/test_pil_history_contract.py &&
    test -f configs/multidag_cl/pil_screen/seed100.yaml &&
    test -f docs/workflows/MERC_CODEX_WORK_STANDARD_V1_0_20261128.md
}
if ! pil_sync; then printf 'PIL sync stopped. Review the last error.\n' >&2; fi
```

## 3. 重新预检并保存退出码

使用原 D Stage A 回执认证的 GPU 可见性、CUBLAS、TF32、MKL 和线程环境。修复后预检创建独立尝试，记录被替代的失败 batch ID 或已知日志路径。`UNKNOWN` 表示原日志暂未定位。只有旧批次从未创建 run、状态、分析或 launcher 资产，且能单独保全旧日志并核对同一预检身份时，才可审查是否复用原 batch ID。

```bash
pil_preflight() {
  [[ -n "${PIL_COMMIT:-}" && "$(git rev-parse --verify HEAD)" == "$PIL_COMMIT" ]] || return 1
  python -B scripts/dev/check_workstream.py multidag || return 1
  local clock previous
  local -a pipe_status
  clock="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)" || return 1
  PIL_DATE="${clock%_*}"
  PIL_BATCH="pil_screen_001_seed100_${clock#*_}"
  PIL_PROJECTION=/home/zhiyuan/research_lh/data/IEMOCAP/asymmetric_merc_v1/tav_train_dev_projection_v1
  PIL_PREFLIGHT_DIR="tmp/pil_screen_001_preflight/${PIL_DATE}_$PIL_BATCH"
  read -r -p 'Supersedes (failed batch ID or log path): ' previous || return 1
  previous="${previous:-UNKNOWN}"
  mkdir -p tmp/pil_screen_001_preflight || return 1
  mkdir "$PIL_PREFLIGHT_DIR" || return 1
  printf '%s\n' "$PIL_COMMIT" "$PIL_DATE" "$PIL_BATCH" "$PIL_PROJECTION" "$previous" \
    >"$PIL_PREFLIGHT_DIR/descriptor.txt" || return 1
  python -B -c 'import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__, torch.version.cuda)' \
    >"$PIL_PREFLIGHT_DIR/environment.txt" || return 1
  printf 'PIL_DATE=%s PIL_BATCH=%s PREFLIGHT_DIR=%s\n' "$PIL_DATE" "$PIL_BATCH" "$PIL_PREFLIGHT_DIR"
  python -B -u -m scripts.workflows.benchmarks.run_pil_screen preflight \
    --projection-dir "$PIL_PROJECTION" --experiment-date "$PIL_DATE" --batch-id "$PIL_BATCH" \
    2>&1 | tee "$PIL_PREFLIGHT_DIR/preflight.log"
  pipe_status=("${PIPESTATUS[@]}")
  printf 'PREFLIGHT_EXIT_CODE=%s TEE_EXIT_CODE=%s\n' "${pipe_status[0]}" "${pipe_status[1]}" \
    | tee "$PIL_PREFLIGHT_DIR/exit_codes.txt" || return 1
  (( pipe_status[0] == 0 && pipe_status[1] == 0 )) || return 1
  grep -qx 'PIL_PREFLIGHT=PASS' "$PIL_PREFLIGHT_DIR/preflight.log"
}
if ! pil_preflight; then printf 'PIL preflight stopped. Review the receipt and exit codes.\n' >&2; fi
```

验收须同时确认两个退出码为 0、`PIL_PREFLIGHT=PASS`、回执 `blocked_arms` 为空，且基线为 `CERTIFIED_REUSE` 或确有 `ASSET_UNAVAILABLE` 的 `ONE_SHARED_OFF_REQUIRED`。回执含实际 HEAD、依赖 SHA、配置、投影 SHA、数据人口、采样带和环境。预检执行真实授权 Train/Dev 投影校验与合成 CUDA 前向，正式 optimizer 更新为零。`PARTIAL`、身份不符、归档格式异常与其他失败，保存描述符、日志和退出码后回审。

## 4. 重连、一次启动和恢复

从第 3 节输出目录取得原 `descriptor.txt`。以下函数读取原参数。恢复原环境后核对同批次进程、状态和 GPU 占用。

```bash
pil_load() {
  PIL_STATE=
  cd /home/zhiyuan/research_lh/m3ed_mmgcn_clean || return 1
  conda activate m3ed_mmgcn || return 1
  python -B scripts/dev/check_workstream.py multidag || return 1
  local descriptor
  local -a fields
  read -r -p 'Original descriptor.txt path: ' descriptor || return 1
  mapfile -t fields <"$descriptor" || return 1
  [[ "${#fields[@]}" == 5 ]] || return 1
  PIL_COMMIT="${fields[0]}" PIL_DATE="${fields[1]}" PIL_BATCH="${fields[2]}" PIL_PROJECTION="${fields[3]}"
  [[ "$PIL_COMMIT" =~ ^[0-9a-f]{40}$ && "$PIL_DATE" =~ ^[0-9]{8}$ &&
     "$PIL_BATCH" =~ ^pil_screen_001_seed100_[0-9]{6}$ ]] || return 1
  [[ "$(git rev-parse --verify HEAD)" == "$PIL_COMMIT" ]] || return 1
  PIL_PREFLIGHT_DIR="$(dirname -- "$descriptor")" || return 1
  grep -qx 'PREFLIGHT_EXIT_CODE=0 TEE_EXIT_CODE=0' "$PIL_PREFLIGHT_DIR/exit_codes.txt" || return 1
  grep -qx 'PIL_PREFLIGHT=PASS' "$PIL_PREFLIGHT_DIR/preflight.log" || return 1
  PIL_MODULE=scripts.workflows.benchmarks.run_pil_screen
  PIL_ARGS=(--projection-dir "$PIL_PROJECTION" --experiment-date "$PIL_DATE" --batch-id "$PIL_BATCH")
  PIL_GROUP="outputs/$PIL_DATE/pil_screen_001"
  PIL_LOG_ROOT="$PIL_GROUP/logs/launcher/$PIL_BATCH"
  PIL_MANIFEST_ROOT="$PIL_GROUP/manifests/batches/$PIL_BATCH"
  PIL_LOG="$PIL_LOG_ROOT/execution.log"
  PIL_STATE="$PIL_MANIFEST_ROOT/state.json"
  PIL_REPORT="$PIL_GROUP/analysis/batches/$PIL_BATCH"
  PIL_PACKAGE="$PIL_GROUP/analysis/batches/MERC_PIL_SCREEN_001_${PIL_BATCH}_analysis.tar.gz"
}

pil_idle() {
  [[ -n "${PIL_STATE:-}" && "$(git rev-parse --verify HEAD)" == "$PIL_COMMIT" ]] || return 1
  local processes
  processes="$(ps -eo pid,args)" || return 1
  if printf '%s\n' "$processes" | grep -F -- "$PIL_MODULE" | grep -F -- "$PIL_BATCH"; then
    printf 'Review the matching PIL process before starting or resuming.\n' >&2
    return 1
  fi
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
}
if ! pil_load; then printf 'PIL descriptor or preflight receipt rejected.\n' >&2; fi
```

用户审核完整预检和 GPU 查询结果后执行一次启动。`run` 会检查同批次目录碰撞并使用批次锁。已有 `launch.log` 时检查原进程和日志。

```bash
pil_start() {
  [[ -n "${PIL_STATE:-}" && "$(git rev-parse --verify HEAD)" == "$PIL_COMMIT" ]] || return 1
  pil_idle || return 1
  local path answer
  for path in "$PIL_MANIFEST_ROOT" "$PIL_LOG_ROOT" "$PIL_REPORT" "$PIL_PACKAGE"; do
    [[ ! -e "$path" && ! -L "$path" ]] || return 1
  done
  read -r -p 'Approved preflight and GPU available? Type yes to launch: ' answer || return 1
  [[ "$answer" == yes ]] || return 1
  (
    set -o noclobber
    exec >"$PIL_PREFLIGHT_DIR/launch.log" 2>&1 || exit 1
    exec nohup python -B -u -m "$PIL_MODULE" run "${PIL_ARGS[@]}" </dev/null
  ) &
  printf 'Launch PID=%s\n' "$!"
}
if ! pil_start; then printf 'PIL launch stopped. Review preflight, process and batch state.\n' >&2; fi
```

用 `tail -F "$PIL_LOG"` 和 `python -m json.tool "$PIL_STATE"` 监控。恢复须保持同一 HEAD、运行环境、原批次和投影。已有完整轮次的中断训练可续跑。分析阶段和封存发布阶段使用独立恢复入口。`invalid`、`complete`、源码身份改变或首轮提交前失败均按原回执处理。

对训练中断执行以下命令。

```bash
pil_idle && test -f "$PIL_STATE" &&
  python -B -u -m "$PIL_MODULE" resume-training "${PIL_ARGS[@]}"
```

对分析或发布中断执行以下命令。

```bash
pil_idle && test -f "$PIL_STATE" &&
  python -B -u -m "$PIL_MODULE" resume-analysis "${PIL_ARGS[@]}"
```

## 5. 验收和回传

看到 `ANALYSIS_EXPORT=COMPLETE; ANALYSIS_BUNDLE=PASS; PLOT_READY=YES` 后复核状态、清单和归档。

```bash
python -m json.tool "$PIL_STATE" &&
  cat "$PIL_REPORT/tables/verdicts.csv" &&
  (cd "$PIL_REPORT" && sha256sum -c SHA256SUMS.txt) &&
  stat -c '%s %n' "$PIL_PACKAGE" &&
  sha256sum "$PIL_PACKAGE" &&
  tar -tzf "$PIL_PACKAGE"
```

在 Windows 通过用户现有的 SSH 主机或别名回传。下载后对照 Ubuntu 输出的 SHA256。

```powershell
$pilSshHost = Read-Host 'Existing Ubuntu SSH host or alias'
$pilRemotePackage = Read-Host 'Absolute archive path printed by the launcher'
$pilDownloaded = Split-Path -Leaf $pilRemotePackage
if (Test-Path -LiteralPath $pilDownloaded) { throw 'Local destination already exists' }
scp "${pilSshHost}:${pilRemotePackage}" .
if ($LASTEXITCODE -ne 0) { throw 'PIL bundle transfer failed' }
Get-FileHash -Algorithm SHA256 -LiteralPath $pilDownloaded
tar -tzf $pilDownloaded
if ($LASTEXITCODE -ne 0) { throw 'PIL bundle read failed' }
```

归档保存 Best/E30、源证书、运行身份、表和图、远端权重路径及 SHA。解压后可用 `python pil_screen_analysis.py .` 重绘。正式裁决采用未舍入指标和相同 ID 的 Best 对比。
