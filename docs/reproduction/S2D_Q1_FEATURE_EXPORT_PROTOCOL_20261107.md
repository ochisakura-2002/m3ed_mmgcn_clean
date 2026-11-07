# S2D Q1 Train/Dev Feature Export Protocol — 2026-11-07

WORKSTREAM = S2D

当前交付：**Q1-2B exporter code ready；仅 synthetic / contract validation**。
本轮未读取真实 IEMOCAP，未导出真实 Train/Dev，未训练，未访问 Test 或远程。
Q1-2A 的 Ubuntu 数值验收已由用户提供，详见
[接口决定](S2D_FEATURE_BANK_INTERFACE_DECISION_20261106.md)。
它不替代本轮新增 single-pass bundle 的 Ubuntu 验收。

## 1. Scientific purpose

一次完整 Uniform16 forward 同时获得冻结的 Q1 表示，避免分别调用独立静态接口和
contextual 接口的重复计算。Q1 **D0 + B 为主，A0/C0 为诊断**；本任务不测量
表示可分性、不进行 feature ranking 或 Dev 比较。Q3 **D0 是独立候选表示**；
B 已依赖整个 clip，不能用于 pre-selection efficiency claims。

独立性仅针对固定 crop 的静态 encoder 子图。既有 bbox tracking/future-bbox
fallback 的上下文依赖保留；single-pass 导出不使整条像素流程变成 causal。

## 2. Inputs

入口：[export_features.py](../../scripts/models/s2d/iemocap/export_features.py)。
读取显式 source run 的 `resolved_config.yaml`、`run_metadata.json`、
`logs/label_vocab.json`、`logs/epochs.jsonl` 和指定纯权重文件。
沿用 `load_config` / 共享 vocab，核对 run ID、label mapping、seed、manifest/data/cache
路径和 Dev-only selection provenance；epoch 日志须已达到原停止边界。
只读取日志已经记录的 selected epoch/score，不重新取 argmax、不重新选择 epoch。

真实运行复用原 `S2DManifest`、`IEMOCAPS2DDataset`、`Uniform16Sampler`、
`CachedBBoxFaceProcessor`、`S2DTransform`。dataset 已返回 `frame_indices`、
`label_index`、`utterance_id`、`split`、`v_valid`、`face_usable`；直接使用这些字段。
没有重写采样、face detection、crop、transform 或 label logic，没有修改训练 dataset。
decoder 根据原采样器生成 indices；canonical cached crop 核对 indices/原图尺寸。
本 exporter 要求现有 face cache，缺项或 decode/cache 不一致直接中止，不在线检测。

resolved config 中的原输入路径直接沿用，无自动数据发现、重定位或 cache 生成。
相对 run/checkpoint/output 路径以项目根为基准；manifest 相对 data_root，cache 相对项目根。

## 3. Frozen run/checkpoint

计划 source run（本轮未打开）：

```text
outputs/s2d/iemocap/v0/20261105/s2d_iemocap_v0_seed100_es30
checkpoints/best_model_weights_only.pt
SHA256 = e54ccb228002578fa2c9a0faa8bdacbd61774e8952fcf06a306afa7b8518639b
用户提供的 selected epoch = 7
用户提供的 selected Dev Weighted-F1 = 0.34355913724532466
```

默认 checkpoint **仅**为 `<run-dir>/checkpoints/best_model_weights_only.pt`，
默认 expected SHA256 为上述值；支持显式 `--checkpoint` / `--expected-checkpoint-sha256`，
任何 hash/load/strict mismatch 都中止，不查找 sibling。
复用 `scripts/models/s2d/iemocap/checkpoints.py::reload_model_weights`：
`torch.load(..., map_location="cpu", weights_only=True)` + `load_state_dict(..., strict=True)`。
请求中提到的 `models/s2d/iemocap/checkpoints.py` 在当前 tree 不存在。

模型以原架构参数 `build_s2d` 构建一次；构建参数去掉两套 initialization checkpoint
路径，因为纯 wrapper state_dict 已包含 core 和 landmark 的全部权重/buffers。
不打开预训练权重、不触发 legacy pickle，不建立第二个 core 或复制模型/权重。
随后严格加载所选文件，迁到原配置设备并显式 `eval()`；原 source run、checkpoint、
resolved config、metadata 均只读。

## 4. Feature contracts A0/C0/D0/B

| 表示 | shard 字段 | 单条 shape | 用途 |
| --- | --- | --- | --- |
| A0 | `a0_rgb_mean` | `[16,768]` | frozen RGB diagnostic |
| C0 | `c0_mcp_static_mean` | `[16,768]` | MCP0 static diagnostic |
| D0 | `d0_static_residual_mean` | `[16,768]` | primary independent static representation |
| B | `s2d_contextual_slots` | `[16,768]` | full contextual S2D control |
| h_v | `h_v` | `[768]` | 原 wrapper pooled feature |
| logits | `logits_v` | `[6]` | 原 wrapper logits |

冻结定义：

```text
A0_tokens = 原 patch_embed 输出                   [B*16,196,768]
C0_map = 原 MCP0 返回值 0                        [B*16,768,14,14]
C0_tokens = C0_map.flatten(2).transpose(1,2)      [B*16,196,768]
D0_tokens = A0_tokens + C0_tokens
D0_mean = mean_spatial(D0_tokens)                [B,16,768]
```

D0 在约简前相加；不把 C0 改回主 bank。`prob_v` 非 bank 必需字段，首版不保存。
本版特征统一保存 **float32**，labels/frame_indices 为 int64，不自动使用 FP16。

## 5. Single-pass design

[features.py](../../models/s2d/iemocap/features.py) 新增：

```python
model.eval()
bundle = extract_uniform16_feature_bundle(model, video)  # video [B,3,16,224,224]
```

API 接收调用者的同一个 canonical `S2DIEMOCAP` 实例，要求所有子模块已 eval，
内部使用 `torch.inference_mode()`。不改变 training flags / requires_grad。
临时注册 RGB patch forward-hook、MCP0 forward-hook 和 head forward-pre-hook，
调用原 `model(video)` **一次**；保存边界值后计算三个空间均值。
`h_v` / `logits_v` 直接使用该次 wrapper 返回的 tensor；没有重写 S2D forward。
所有 hook 在 `finally` 删除，保留已有 caller hooks，异常也清理。提取期间同一实例串行使用。
既有 arbitrary-N 静态 API 和 contextual API 保留；完整模型的 T=16 限制保持不变。

## 6. Split policy

`ALLOWED_SPLITS = {"train", "dev"}`。CLI 必须显式选择 `--split train|dev`，
或者 `--splits train,dev`；无默认的 all-splits，保留请求顺序。
拒绝空值、重复 split、未知值和任何 Test 请求；`--split test`、
`--splits train,dev,test` 均在配置/权重/数据/输出 IO 前失败。
内部 `export_run`、dataset preparation、writer engine 和单 split writer 也有门禁。

loader 固定 `shuffle=False, drop_last=False`，使用原 batch_size/num_workers，
seed=100 的独立 generator。记录顺序保留 manifest 顺序，不按字母重新排序。
writer 将 batch ID/label/split 与 dataset records 顺序核对，拒绝乱序或混入其他 split。

## 7. Test leakage guard

所有 dataset/loader/coverage 集合严格来自用户请求的 train/dev。
writer 拒绝额外 mapping keys；不调用会枚举所有 split 的 `manifest.counts()`，
不使用 training/finalization helper，不加载 Test metrics/predictions，不构建 Test loader。
不生成 `test/`、`test.pt`、Test index 或 Test coverage。

原 manifest/cache JSON 是全量 envelope；复用 canonical parser 会读取原文件中的元数据，
但 exporter 只按请求 split 查询/构建 dataset 与统计 coverage。这不等于读取 Test 视频
或导出 Test 特征。测试对 `for_split` 加请求白名单，未请求 split 一旦被查询立即失败；
另验证 dev-only 运行只产出 dev index/coverage，不隐式展开 Train 或 Test。

## 8. Output schema

未来新输出使用显式、尚不存在的目录，例如
`outputs/<YYYYMMDD>/s2d_q1_feature_bank/analysis/<bank_id>/`；不写入历史 run。

```text
feature_bank/
  provenance.json
  coverage.json
  complete.json             # 所有请求 split 成功后最后写入
  train/                    # 仅请求 train 时创建
    part-00000.pt
    part-00001.pt
    index.json
  dev/                      # 仅请求 dev 时创建
    part-00000.pt
    index.json
```

`schema_version = s2d_uniform16_feature_bank_v1`，默认 `shard_size=256`。
每个 `.pt` 为可 `weights_only=True` 读取的 dict：

```text
schema_version: str
split: train | dev
utterance_ids: list[str]            # N 条，顺序固定
labels: LongTensor[N]
frame_indices: LongTensor[N,16]
a0_rgb_mean: FloatTensor[N,16,768]
c0_mcp_static_mean: FloatTensor[N,16,768]
d0_static_residual_mean: FloatTensor[N,16,768]
s2d_contextual_slots: FloatTensor[N,16,768]
h_v: FloatTensor[N,768]
logits_v: FloatTensor[N,6]
```

尾 shard 可不足 256；空 split 保存 row_count=0 的 index/coverage，不伪造空 shard。
仅缓冲一个 shard 加当前 inference batch 的特征；逐条 clone 避免切片持有更早 batch
的底层 storage。允许 O(N) 的轻量 index metadata，不把全 Train 特征积累在 RAM。

`index.json` 包含 schema_version、split、row_count、shards（filename/row_count）和
rows。每个 row：`global_row`（split 内 0-based）、`utterance_id`、`shard`、
`row_in_shard`（0-based），可按 ID 与 TA predictions join。

## 9. Coverage accounting

`coverage.json = {schema_version, splits: {<requested split>: ...}}`。各 split 包含：

| 字段 | 依据 |
| --- | --- |
| `manifest_total` | 原 manifest 该 split 的全部记录 |
| `v_available` | 该 split 的原 V availability |
| `face_cache_usable` | V available 且 cache entry.face_usable=True |
| `exported_count` | 实际成功写出的 row 数，必须等于 face_cache_usable |
| `excluded_count` | manifest_total - exported_count |
| `exclusions.v_unavailable` | 原 availability=False 的 ID |
| `exclusions.cached_face_unusable` | 原 cache face_usable=False 的 ID |
| `face_unusable_reasons` | 每个不可用 entry 的原始 reasons 去重值，无推测 |
| `unknown_exclusion_count` | 当前两类精确分区覆盖全部 exclusion，故为 0 |
| `failure_policy` | 缺 cache、decode/crop/cache mismatch、提取失败直接 abort |

不能把 decode 失败静默归入 face failure；本版对此不产生成功 coverage 或 complete marker。
当前 cache 对“无检测”有明确 reason，故不编造更细的原因；未来不可区分的类别应标为
UNKNOWN/unavailable，不能猜测。manifest N 与 bank N 的差异来自缺 V 和 cache 不可用过滤，
这些行保留在原 manifest 和本 coverage 审计中。

## 10. Provenance

`provenance.json` 记录 schema_version、created_by、当前 git_head、source_run_id/path、
source_git_head（来源 metadata 无值则 null）、selected_epoch、selected_dev_weighted_f1、
checkpoint filename/path/SHA256/expected hash/size/strict/weights_only、resolved config identity、
label vocab、Uniform16/16、六个 feature fields 及 shapes、float32、feature contract version
`s2d_q1_a0_c0_d0_b_v1`、shard_size、请求 splits、shuffle=False、Q1/Q3 表示角色。
manifest/cache 的实际 path/SHA256/size、data_root、columns 与 cache preprocessing_version
来自当前 API/文件；不发明未提供的资产版本。
`completion_marker="complete.json"` 指向整体完成证据；无此文件的目录不视为完整 bank。

## 11. Atomicity / overwrite policy

根目录以 `mkdir(exist_ok=False)` 独占创建，**包括已有空目录也拒绝**；并发或中断后
重跑须指定新的 output dir，不实现 resume/overwrite。
每个 shard 和 JSON 都写到目标同目录的唯一 `.tmp`，flush、fsync、close，再 `os.replace`。
writer 先拒绝已存在的目标文件；失败时清理本次临时文件，既有成功 shard 保留用于审计。
所有 split index/coverage 成功后最后原子写 `complete.json`。
单文件发布是原子的；整个目录不是事务，不将缺少 complete marker 的部分结果当作成功。

## 12. Validation

目标测试：

```text
tests/models/s2d/test_s2d_features.py
tests/models/s2d/test_s2d_feature_export.py
```

真实 canonical fixture 创建完整随机 12 层/768 维 core + MobileFaceNet，用 synthetic 输入：
对比 bundle A0/C0/D0 与原 static API、bundle B 与原 contextual API，逐元素核对 raw
boundaries，检查 mean(B)≈h_v；通过 wrapper hook 证明每次 bundle 仅一次 forward，
h_v/logits_v 与同一次原 wrapper 返回值为同一 tensor。重复输出、参数/buffers digest、
requires_grad、training flags、模块身份和 state_dict schema 均验收不变。
数值容差 `rtol=1e-5, atol=1e-6`，不承诺跨设备 bitwise 相等。

写出测试可用明确的 fake dataset/bundle 测试 shards/index/顺序/atomicity/schema；
它们不替代 canonical 数值验收。另覆盖错误 split（包括内部集合）、缓存缺项、
现有目录拒绝、失败重跑拒绝、纯 tensor checkpoint 真实 IO/strict reload、禁止 fallback、
不训练、原 run 只读及 synthetic 端到端 CLI 流程。

2026-11-07 Windows，现有 `m3ed_mmgcn` / `torch 2.6.0+cpu`：
**76 passed, 11 skipped in 2.95s**。11 个真实 canonical case 因缺 `timm/einops` 明确 skip；
其中包含原 Q1-2A 六项和新增 bundle 五项，**Q1-2B canonical numerical acceptance 待 Ubuntu**。
未安装依赖。Windows pytest 默认 mode=0700 临时目录在受限环境不可访问；写出测试沿用
现有 S2D IO 测试的 workspace UUID fixture，仅清理自己创建的 synthetic 子目录。
Conda stdout 使用 UTF-8 / no-capture-output，避免系统 GBK 包装输出错误。

本轮四个新增/修改 Python 文件的 `py_compile` **PASS**；真实 CLI `--help` **PASS**，
未打开配置/数据/权重或创建输出。tracked `git diff --check` 与三个新文件的独立
`git diff --no-index --check` 均无空白错误。原有 early-stopping 测试修改保持原样，
SHA256 仍为 `e26e9ea234f504605e33f0e82e779da64ac35c8caf98b807e4aa6cd1e30a3f7d`。

本地命令（PowerShell，环境变量仅对该进程有效）：

```powershell
$env:PYTHONNOUSERSITE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
conda run --no-capture-output -n m3ed_mmgcn python -B -m pytest tests/models/s2d/test_s2d_features.py tests/models/s2d/test_s2d_feature_export.py -q -p no:cacheprovider -rs
conda run --no-capture-output -n m3ed_mmgcn python -m py_compile models/s2d/iemocap/features.py scripts/models/s2d/iemocap/export_features.py tests/models/s2d/test_s2d_features.py tests/models/s2d/test_s2d_feature_export.py
git diff --check
git status --short
```

下一次 Ubuntu validation（本轮不连接、不执行远程/Git 写操作）：

```bash
python scripts/dev/check_workstream.py s2d
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -B -m pytest \
  tests/models/s2d/test_s2d_features.py \
  tests/models/s2d/test_s2d_feature_export.py \
  -q -p no:cacheprovider -rs
```

在现有 `s2d_iemocap` 环境要求 **0 failed、0 skipped**，默认 CPU、随机权重、synthetic only。
不得以 fake model 代替 bundle 的 canonical alignment，也不需要真实 checkpoint 或视频。

## 13. Expected real-run command（未来独立授权后）

以下只是未来命令模板，**本轮未执行**。先完成上述 Ubuntu synthetic 验收，再按独立
真实导出任务授权运行。source run 中已 resolved 的路径必须仍指向原资产。
输出日期及 bank ID 在执行时显式指定，重复目录会失败：

```bash
S2D_RUN_DIR="outputs/s2d/iemocap/v0/20261105/s2d_iemocap_v0_seed100_es30"
python -B scripts/models/s2d/iemocap/export_features.py \
  --run-dir "$S2D_RUN_DIR" \
  --checkpoint "$S2D_RUN_DIR/checkpoints/best_model_weights_only.pt" \
  --expected-checkpoint-sha256 e54ccb228002578fa2c9a0faa8bdacbd61774e8952fcf06a306afa7b8518639b \
  --splits train,dev \
  --shard-size 256 \
  --output-dir outputs/20261107/s2d_q1_feature_bank/analysis/seed100_epoch7_uniform16_v1
```

## 14. What is explicitly NOT done yet

本轮真实 IEMOCAP/manifest/cache/checkpoint 读取、真实 Train/Dev feature export、
Test split 访问、GPU、训练、远程操作、git add/commit/push 均为 **0**。
没有 probe/Linear/TCN/GRU、feature ranking、Dev comparison、Audio guidance、
dense N32 real sampling 或新 face cache。未修改模型数学、原 forward、checkpoint schema、
训练超参数、early stopping、label mapping 或 formal dataset 行为；不新增训练模块，
因而不增加 YAML 开关。Q1-3 的 probe 与 Q3 的真实选择效率实验继续独立推进。
