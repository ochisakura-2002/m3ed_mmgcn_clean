# CT-MultiDAG 认证 Train/Dev 投影手动交接

## 1. Windows 暂存和提交

在 MULTIDAG 工作区执行 guard，只暂存以下 5 个文件。确认暂存区没有其他任务的改动。

```powershell
python scripts/dev/check_workstream.py multidag
git diff --cached --name-status
$projectionFiles = @(
  'scripts/data/build/multidag_cl/build_train_dev_projection.py',
  'tests/data/build/multidag_cl/test_train_dev_projection.py',
  'docs/research/post_failure_method_design_001/TRAIN_DEV_PROJECTION_IMPLEMENTATION_REPORT.md',
  'docs/research/post_failure_method_design_001/TRAIN_DEV_PROJECTION_TEST_REPORT.md',
  'docs/research/post_failure_method_design_001/TRAIN_DEV_PROJECTION_RUNBOOK.md'
)
git add -- $projectionFiles
git diff --cached --name-status
git diff --cached --check
```

暂存范围核对后，由用户提交并推送。

```powershell
git commit -m 'Build certified CT-MultiDAG Train/Dev projection'
git push origin HEAD:refs/heads/feat/multidag-cl-paper-reimplementation
```

## 2. Ubuntu 快进同步

从 MULTIDAG 工作区查看待合入提交。

```bash
git fetch origin feat/multidag-cl-paper-reimplementation
git log --oneline HEAD..FETCH_HEAD
```

确认范围后快进，并在已激活的项目环境执行 guard。

```bash
git merge --ff-only FETCH_HEAD
python scripts/dev/check_workstream.py multidag
```

## 3. 校验并构建正式投影

将 `IEMOCAP_ASYM_ROOT` 指向服务器现有的 `asymmetric_merc_v1` 根目录。以下命令使用 `tav` 下三个 canonical 文件，默认输出到同级新目录 `tav_train_dev_projection_v1`。使用可信的原始 canonical pickle。输出目录已经存在时会拒绝。以下步骤应在同一个 shell 执行。

```bash
TAV_SOURCE_ROOT="${IEMOCAP_ASYM_ROOT:?set the asymmetric_merc_v1 root}/tav"
TAV_PROJECTION_OUTPUT="${TAV_PROJECTION_OUTPUT:-$IEMOCAP_ASYM_ROOT/tav_train_dev_projection_v1}"
CANONICAL_ASSET="$TAV_SOURCE_ROOT/iemocap_asym_tav_t1024_v1.pkl"
CANONICAL_MANIFEST="$TAV_SOURCE_ROOT/model_input_manifest_tav_t1024_v1.csv"
CANONICAL_PROVENANCE="$TAV_SOURCE_ROOT/tav_t1024_provenance_manifest_v1.json"

SOURCE_BEFORE=$(sha256sum "$CANONICAL_ASSET" "$CANONICAL_MANIFEST" "$CANONICAL_PROVENANCE") || exit 1
printf '%s\n' "$SOURCE_BEFORE"

python -B scripts/data/build/multidag_cl/build_train_dev_projection.py \
  --asset-path "$CANONICAL_ASSET" \
  --model-manifest-path "$CANONICAL_MANIFEST" \
  --provenance-manifest-path "$CANONICAL_PROVENANCE" \
  --output-dir "$TAV_PROJECTION_OUTPUT" || exit 1

SOURCE_AFTER=$(sha256sum "$CANONICAL_ASSET" "$CANONICAL_MANIFEST" "$CANONICAL_PROVENANCE") || exit 1
if [[ "$SOURCE_BEFORE" != "$SOURCE_AFTER" ]]; then
  printf '%s\n' 'BUILD_STATUS = INVALID · canonical source SHA-256 changed' >&2
  exit 1
fi
```

正式成功终端显示 `BUILD_STATUS = PASS`、`TRAIN = 100 dialogues / 4830 utterances`、`DEV = 20 dialogues / 980 utterances`、`SOURCE_IMMUTABILITY = PASS`、`PAIRWISE_PREFLIGHT_VALIDATOR_COMPATIBLE = YES` 与 `TEST_TOUCHED = NO`。输出只含 `train_dev_tav.pkl`、`model_input_manifest.csv` 与 `train_dev_provenance.json`。

## 4. 绑定历史 F0 并单独验证

先读取正式 F0 第 21 轮的历史 run manifest 校验证书，使用的仍是 `load_projected_train_dev`。该步骤只校验工件和数据身份，无模型 forward。

```bash
FROZEN_F0_RUNS="${FROZEN_F0_RUNS:-outputs/20261123/fusion_final_paired_5seed_confirmation/runs}"
python -B - "$TAV_PROJECTION_OUTPUT" "$FROZEN_F0_RUNS" <<'PY' || exit 1
from pathlib import Path
import sys
from types import SimpleNamespace

from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import (
    F0_IDENTITY, load_projected_train_dev, read_json, validate_hold,
)
from scripts.runtime.multidag_cl_fusion.spec import make_final_config

projection, frozen_runs = map(Path, sys.argv[1:])
run_id = make_final_config("F0", F0_IDENTITY["seed"])["experiment_id"]
manifest = read_json(frozen_runs / run_id / "manifests" / "run_manifest.json")
validate_hold(manifest, legacy=True)
expected = {"run_id": run_id, "seed": F0_IDENTITY["seed"], "best_epoch": F0_IDENTITY["epoch"],
            "git_head": F0_IDENTITY["git_head"], "best_checkpoint_sha256": F0_IDENTITY["checkpoint_sha256"]}
if any(manifest.get(key) != value for key, value in expected.items()):
    raise SystemExit("historical seed100 F0 run manifest does not match")
load_projected_train_dev(projection, SimpleNamespace(manifest=manifest))
print("PAIRWISE_PREFLIGHT_VALIDATOR_COMPATIBLE = YES")
PY
export TAV_TRAIN_DEV_PACKAGE="$TAV_PROJECTION_OUTPUT"
```

认证成功的三文件为后续固定输入，保留原文件和哈希。

## 5. 用户另行启动已实现的 Preflight

Builder 在发布和验收后结束。步骤 4 通过后，由用户手动运行既有 Preflight。

```bash
RUN_DATE=$(TZ=Asia/Shanghai date +%Y%m%d)
BATCH="seed100_preflight_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="${PREFLIGHT_GPU:-0}" python -u \
  scripts/analysis/models/multidag_cl/run_pairwise_preflight.py \
  --frozen-runs "$FROZEN_F0_RUNS" \
  --train-dev-package "${TAV_TRAIN_DEV_PACKAGE:?}" \
  --batch-id "$BATCH" --experiment-date "$RUN_DATE"
```

可选 Round 1 trace 和回传要求沿用 [Preflight Runbook](PAIRWISE_PREFLIGHT_RUNBOOK.md)。
