# S2D-IEMOCAP V0 接入说明

本地交付目标为 `CODE READY FOR REMOTE SMOKE`。这里只新增独立 V 分支，
未修改已有 MultiDAG/TA/MMGCN 的模型、训练逻辑、标签或 split。
远程已完成下述真实数据 face preprocessing 单 utterance pilot；本轮本地仅做
synthetic/mock 检查，没有训练、安装依赖、下载数据/权重或生成真实 cache。
真实 S2D 权重加载、GPU forward/backward 与显存 smoke 仍待远程完成。

## Upstream

- URL：<https://github.com/MSA-LMC/S2D>
- Commit：`e1b5e5dfb57ab15f70103c933b346bfa62943159`
- License：Apache-2.0，原文保留于 `third_party/S2D/LICENSE`。
- `third_party/S2D/` 保存 33 个原样源码/参考文件，约 347 KB；不含数据标注、
  媒体、权重和嵌套 `.git`。来源及逐文件 SHA256 见 `UPSTREAM.json`。
  `.gitattributes` 保持 Windows/Linux 字节一致。
- 官方源码只读；本项目适配代码均位于其外。不运行上游训练入口，避免其
  `datasets`/`utils` 与项目同名包冲突，也避免自动使用固定权重路径。
- 官方 README 指向的预训练文件：
  [pretrain-AffectNet-7.pth](https://huggingface.co/cyinen/S2D/blob/main/pretrain-AffectNet-7.pth)，
  349 MB；发布页 SHA256 为
  `1c0924519dc7be0e6e4c0187a61664ac282d3f8ddb17483df65f6b0f02a71c48`。
  官方 shell 示例使用 `Pretrain-AffectNet-7.pth`，大小写与发布文件不同，
  Linux 上应以实际文件名配置。此次未下载该权重。
- landmark 文件预期为 `mobilefacenet_model_best.pth.tar`，官方 Git tree 自带；
  原始 clone 含这一约 12 MB 文件，但正式源码快照明确排除它。
  后续在远程外部模型目录保存，通过配置路径加载。

## Label space

目标严格为 `benchmark_label`，K=6，索引固定为：

| index | benchmark_label | tracked loader name |
| --- | --- | --- |
| 0 | hap | Happy |
| 1 | sad | Sad |
| 2 | neu | Neutral |
| 3 | ang | Angry |
| 4 | exc | Excited |
| 5 | fru | Frustrated |

证据来自当前本地 `datasets/asymmetric_iemocap.py::LABELS` 及
`configs/_shared/data/iemocap/asymmetric_dataset_schema_v1.json::ta_asset.label_vocab`，
二者明确规定 Candidate A/C TA 的上述编码。已跟踪的
`datasets/iemocap/official_feature_dataset.py::IEMOCAPOfficialFeatureDataset.DEFAULT_LABEL_LIST`
亦使用同一顺序；测试使用显式长短名称对应验证，不按字母排序。
前两份本地文件不属于此次新增代码的运行/测试依赖。

共享配置为 `configs/_shared/data/iemocap/benchmark_label_vocab_v1.yaml`，API 为
`datasets.iemocap.label_vocab.LabelVocab`，提供 `label_to_index`、
`index_to_label`、`source` 和 `num_classes`。
`TA_LABEL_INDEX_MAPPING_UNVERIFIED=NO` 仅表示本地 Candidate A/C TA contract 已确认；
远程仍须核对实际 TA 资产。另一路 MultiDAG Candidate B 的独立 `label_vocab.pkl`
不在此对齐结论内，不得混用。`canonical_emotion` 即使存在或冲突，也不作训练 target。

## 代码入口

| 职责 | 路径 |
| --- | --- |
| 共享 vocab | `datasets/iemocap/label_vocab.py` |
| CSV adapter | `datasets/iemocap/s2d_manifest.py` |
| Uniform16 | `datasets/iemocap/s2d_sampling.py` |
| FaceProcessor | `datasets/iemocap/s2d_faces.py` |
| decoder / transform / dataset | `datasets/iemocap/s2d_dataset.py` |
| 官方核心 wrapper | `models/s2d/iemocap/model.py` |
| config reader | `models/s2d/iemocap/config.py` |
| 训练及最终导出 | `scripts/models/s2d/iemocap/train.py` |
| 远程模板 | `configs/s2d/iemocap/v0.yaml` |
| 远程最小依赖清单 | `models/s2d/iemocap/requirements-remote.txt` |
| 合成测试 | `tests/models/s2d/test_s2d_v0.py` |

## Manifest contract

CSV 必须具有 `utterance_id,video_path,benchmark_label,split,v_available`。
`label_index` 从共享 vocab 派生；CSV 若提供它，必须逐行一致。
通过 `dataset.columns` 显式映射实际列名；正式 YAML 已设置
`video_path: video_relative_path`，这是远程已验证的视频字段。

`utterance_id` 全局唯一；split 只能使用已有 `train/dev/test`，不产生新划分。
`v_available` 接受 `0/1/false/true`，为 1 时 video path 不得为空。
原始完整 universe 保存在 `S2DManifest.records/by_id`。训练 V dataset 过滤不可用 V
及 cache 中 `face_usable=False` 的 utterance；后者不改变 manifest 的 `v_available`。
远程 run 的 `metrics/coverage.json` 保存全量/可用/不可用计数、缺 V 的 ID、
各 split 的 face_usable 数量、face_unusable ID、有效检测帧数和 fallback 帧数。
已知 10 条无 V 样本保留在全量语义中；这里不把 10 硬编码成过滤数量。

`data_root` 由配置提供，相对路径以项目根目录为基准；`manifest_path` 相对
`data_root` 解析。CSV 中相对视频路径由 `data_root + relative_path` 解析，
也支持当前 OS 的绝对路径；拒绝在 Linux 静默重解释 Windows 绝对路径。
解析器不读取视频、不检查本地真实数据是否存在。

## 已有远程真实数据证据（用户提供）

- Manifest 验证：7433 rows，train/dev/test = 4830/980/1623。
- V available = **7423**，V unavailable = **10**；保留完整 manifest。
- 视频是 speaker-specific utterance-level panel MP4，可能含背景人脸。
- 样本 `Ses02F_script01_3_M001`：480×360，约 29.97 fps；real video decode **PASS**。
- Uniform16 **PASS**：`[0,3,6,9,12,16,19,22,25,28,32,35,38,41,44,48]`。
- `facenet-pytorch==2.5.3` CPU MTCNN single-frame **PASS**；
  CPU MTCNN 16/16 sampled frames 检测和目标选择 pilot **PASS**。
- GPU MTCNN 曾发生 **segmentation fault**，因此 V0 detector 固定 **CPU**；
  **S2D 模型仍使用 GPU**（`system.device: cuda`）。

以上 face detection 证据只覆盖一个真实 utterance，**不代表整个 IEMOCAP 的
face detection 已验证**。本轮正式代码仍需在同一样本上重复远程 smoke。

## Uniform16、FaceProcessor 与 transform

Uniform16 的索引为 `i * (N-1) // 15, i=0..15`，覆盖 utterance clip 两端，
不依赖模型或 label；短视频重复索引，N<=0 报错。默认远程 decoder 使用
decord 只取采样帧，RGB 输出，不随机换取其他 utterance。

在现有 `FaceProcessor` 上扩展 `MTCNNFaceProcessor`，正式流程固定为：

```text
raw speaker-panel video -> Uniform16 -> CPU MTCNN -> target tracking
  -> square face crop (margin=1.40) -> RGB 224x224 -> normalize -> S2D (GPU)
```

MTCNN 使用 `device=cpu, keep_all=True, min_face_size=20`，调用
`detect(..., landmarks=False)`，只保留有限、有效 bbox 且 probability >= 0.90
的检测。bbox 先裁到原图边界。首次有效采样帧选择最大面积脸；后续先保留
area >= 0.35 × 上一有效 selected area 的候选，再选中心最近者；若没有候选
满足面积条件，在所有有效候选中选中心最近者并记录 `area_filter_relaxed`。
跟踪状态只属于当前 utterance，不跨样本保存。

crop 以 selected bbox 中心和 `ceil(max(width,height) × 1.40)` 生成正方形，
边界处平移入图；边长最多为原图短边，不 pad 黑边。最后双线性 resize 到 224×224。
V0 **不做 5-point/其他 landmark alignment**，不拼接 68-landmark 坐标，
不加 AU/graph，不改 S2D 内部 landmark-aware MCP/TMA。
禁止将 full-frame resize 作为正式输入或失败 fallback。

每帧 `FaceResult` 保留 `face, face_valid, detection_valid, fallback_used, reason,
selected_bbox, detection_probability, crop_bbox`。其中 selected bbox 是 margin 前
原图 xyxy，crop bbox 是边界处理后的整数正方形；`face_valid` 表示有可用 crop，
不等于该帧检测成功。

- 某帧检测失败：`detection_valid=False`，使用上一有效 bbox 在**当前原始帧**上
  crop，`fallback_used=True`；probability 为 null，不伪造成功置信度。
- 首次成功之前的失败帧：整段检测完成后使用本段首个有效 bbox 回填，reason 为
  `no_valid_detection_reuse_first_future_bbox`，仍明确标记 fallback。
- 整段 16 帧无有效检测：`face_usable=False`，所有 face/bbox/probability 为 null，
  detection/fallback flags 均为 false；保留审计条目，不产生黑帧或 full-frame。
- 其他失败或 cache 不一致：携带 utterance ID 报错，不静默换样本。

原 `FaceProcessor.__call__` 与 `mock_full_frame` synthetic 路径保留；mock 需要显式
test-only 授权，不能用于真实训练/评估。facenet-pytorch 只在显式远程 MTCNN
预处理第一次检测时 lazy import；缺失时明确提示远程需要 `facenet-pytorch==2.5.3`。
配置检查、cached bbox 和非 MTCNN 测试不需要安装它。

## Face bbox cache contract 与运行模式

第一版只建立合同、读取/显式写入 API 与 Dataset 接口；本地不生成真实 cache。
`FaceBBoxCache` 的 JSON envelope 为：

```json
{
  "preprocessing_version": "s2d_iemocap_v0_uniform16_mtcnn_cpu_v1",
  "entries": {"<utterance_id>": {"utterance_id": "<utterance_id>", "...": "see fields below"}}
}
```

每个 `FaceCacheEntry` 包含：

| 字段 | 合同 |
| --- | --- |
| `utterance_id` | 与 entries 的 key 一致 |
| `frame_indices` | 16 个 Uniform16 原帧索引，允许短视频重复索引 |
| `frame_sizes` | 16 个原图 `[width,height]`，读取时与 decode 核对 |
| `selected_bboxes` | 16 个 margin 前原图 xyxy；fallback 保存复用 bbox，整段失败为 null |
| `detection_probabilities` | 16 个当前帧检测概率；检测失败为 null |
| `detection_valid` / `fallback_used` | 各 16 个 boolean |
| `reasons` | 各帧选择/失败/fallback 原因，不能为空 |
| `valid_detection_count` | 等于 detection_valid 的 true 数量 |
| `face_usable` | 等于 valid_detection_count > 0 |
| `preprocessing_version` | 必须与 envelope 和当前代码版本一致 |

合同校验拒绝缺项、非 16 帧、重复 key、key/ID 不符、非法 bbox/概率、flags/count
矛盾、旧版本和错误 fallback。版本冻结 Uniform16/CPU/min20/prob0.90/margin1.40/
no-alignment/no-full-frame；以后改变这些语义需要新版本，不能复用旧 cache。
cache 应绑定本次 manifest/原始视频资产，资产有变化时重新预处理；索引/尺寸检查
不等于视频内容哈希校验。

训练模板使用嵌套 `face.backend: cached_bbox`、`face.cache_path` 和
`face.require_cache_for_training: true`；cache 相对路径以项目根为基准。
训练入口要求所有 V-available utterance 都有条目（包括 face_usable=False 的记录），
直接用 cached bbox crop；**不允许每个 epoch/batch 在线检测，也不在 cache 缺失时
自动调用 MTCNN**。即使 backend 配为 mtcnn_cpu，只要提供 cache，训练仍走 cache。
Dataset 输出各帧审计 tensor；缺概率/bbox 的 tensor sentinel 为 -1，JSON 合同仍用 null。

只有 `build_face_processor(face_config, remote_preprocessing=True)` 且
backend=mtcnn_cpu 才启用 detector；Dataset 的同名显式模式逐条返回 cache record，
包含完全检测失败的 utterance，不能接入训练 DataLoader。远程可按下述 API 顺序
生成 cache（先单 utterance smoke，再扩展全量；这里的全量代码仅作交接示例）：

```python
from datasets.iemocap.s2d_faces import FaceBBoxCache, FaceCacheEntry, build_face_processor
from datasets.iemocap.s2d_dataset import IEMOCAPS2DDataset
from datasets.iemocap.s2d_manifest import S2DManifest
from models.s2d.iemocap.config import load_config, resolve_path

config, vocab = load_config("configs/s2d/iemocap/v0.yaml")  # 远程先填 data/cache 路径
root = resolve_path(config["dataset"]["data_root"])
manifest = S2DManifest(resolve_path(config["dataset"]["manifest_path"], root), root,
                       vocab, config["dataset"]["columns"])
face_config = dict(config["face"], backend="mtcnn_cpu")
processor = build_face_processor(face_config, remote_preprocessing=True)
entries = {}
for split in ("train", "dev", "test"):
    dataset = IEMOCAPS2DDataset(manifest, split, processor, remote_preprocessing=True)
    for i in range(len(dataset)):  # CPU，顺序预处理一次；不启动训练
        record = FaceCacheEntry.from_dict(dataset[i])
        entries[record.utterance_id] = record
FaceBBoxCache(entries).save(resolve_path(config["face"]["cache_path"]))
```

`save` 需要父目录已存在，且拒绝覆盖现有 cache；cache 是外部数据资产，不纳入 Git。
正式训练优先使用这份预计算 bbox cache，像素 crop/resize 每次读取，MTCNN 不再运行。

官方依据：`datasets/kinetics.py::VideoClsDatasetFrame`、
`datasets/volume_transforms.py::ClipToTensor`、`run_class_finetuning.py`、
`scripts/ft_dfew.sh` 以及 `model_finetuning.py::VisionTransformer.forward`。

- 模型输入 `[B,3,16,224,224]`，RGB uint8 /255 后使用
  mean `(0.485,0.456,0.406)`、std `(0.229,0.224,0.225)`。
- 官方可选训练增强包括 RandAugment（默认策略 `rand-m7-n4-mstd0.5-inc1`，
  bicubic）、空间随机 crop/flip 和 random erasing；frame dataset 使用
  `args.augment` 控制前两者。验证为双线性 resize/center crop，测试支持多段多 crop。
- S2D-V0 对各 split 统一使用单个 Uniform16 clip 和确定性 face resize/normalize，
  明确关闭上述随机增强及多 clip/crop；这是 IEMOCAP V0 适配，不宣称复现原 FER 规程。

## Wrapper 与 checkpoint

`build_s2d(vocab, model_config)` 只在调用时导入官方模型文件，
以 `pretrained=False` 构建官方 `s2d_base_patch16_224`，再从显式路径加载。
它保留官方 MCP/TMA、feature-map prompting 和 forward。
输入已经归一化的 clip 同时进入 RGB 核心与 frozen MobileFaceNet。
landmark 按官方 `Landmark.forward` 将帧双线性缩至 112×112（align_corners=True），
输出 `[B*16,128,14,14]` feature map，而非坐标 concat。

输出 `{h_v: [B,768], logits_v: [B,6], prob_v: [B,6]}`。
`h_v` 为官方 head 返回的 temporal pooled、分类前特征；训练时遵循官方 head dropout，
eval/export 时 dropout 关闭。utterance ID 和 `v_valid` 由数据层保留。

冻结规则沿用官方参数名筛选：ViT patch/embed/attention/MLP 等主体冻结；
`prompt`（含 MCP/TMA）、`Adapter`、temporal embedding、`ln_post` 和 classifier 可训练。
landmark 参数和 BatchNorm running statistics 始终冻结。SDL 固定关闭。
`model.enabled=false` 拒绝启动此独立 V 训练入口；既有 TA 入口及其数学行为不变。

加载支持 tensor state_dict 或 `model_state_dict/model/state_dict/module` 容器，
清除 `module.` 前缀；S2D 初始化可去掉 landmark keys 并将 `norm` 映射至 `ln_post`。
strict=True 要求所有 key/shape 匹配；默认 S2D 初始化为 non-strict 并显式丢弃
FER `head.fc_cls.{weight,bias}`，用 K 类新 head 替换。non-strict 仍拒绝未显式处理的
shape mismatch；缺失/额外/丢弃 keys 完整返回并记录。冻结的 ViT 参数缺失时构建失败，
避免随机冻结主体。landmark 默认 strict=True。最终项目 best checkpoint strict=True 回载。
正式入口要求两种 checkpoint 均配置；纯代码构建 API 不因没有权重而禁止使用。

## 训练与输出

模板可配置 data_root、manifest_path、两种 checkpoint、num_frames、input_size、
batch_size、num_workers、learning_rate、epochs、seed 和 `system.output_dir`。
V0 对 16、224、seed=100、CrossEntropy、SDL=OFF 强制校验。
`system.output_dir` 为 output base，公共 resolver 创建
`<output_dir>/<YYYYMMDD>/s2d_iemocap_v0/runs/<run_id>/`，不会直接写旧结构。
日期和 group 支持既有 CLI override；fixed run ID 碰撞由公共工具在 epoch 前拒绝。

仅以 dev Weighted-F1 选 best，分数相等保留最早 epoch；epoch loop 不读取 Test 视频。
训练结束 strict 回载 best 后进行一次最终 Test，指标为 Accuracy、Weighted-F1、
Macro-F1、UAR、每类 Precision/Recall/F1 和固定 6 类 confusion matrix，
复用 `utils.metrics`。指标总体是 V-available 且 face_usable 子集，须结合 coverage 报告解释。
保存 resolved config、项目/上游 commit、seed、label vocab、加载报告、epoch 日志，
最终 `test_final.json`、按 utterance ID 的 logits/prob CSV；
`export.representations=true` 时额外保存 h_v/logits_v/prob_v tensor 文件。

## 本地检查与远程交接

本地检查不要求数据、视频、checkpoint 或完整视觉依赖：

```powershell
conda run -n m3ed_mmgcn python -B scripts/models/s2d/iemocap/train.py --config configs/s2d/iemocap/v0.yaml --check-config
conda run -n m3ed_mmgcn python -B -m pytest tests/models/s2d -q -p no:cacheprovider
```

测试 fixture 仅 4 行，所有 ID/字段均标明 `TEST_SYNTHETIC`；正式 YAML 不引用它。
测试覆盖 import、vocab、Uniform16（N=5/16/31/100，另含 N=1/非法输入）、
parser、split/缺 V/路径/冲突标签、mock face/transform、wrapper contract double、
checkpoint 模式、指标/导出和源码来源一致性。
wrapper double 仅用于检查接口和冻结规则，**不是官方 S2D forward 的测试结果**。
新增 face mock 测试覆盖最大面积/连续性/概率过滤、逐帧失败/首段失败/整段失败、
utterance 状态重置、square margin/边界/224 输出、lazy import、CPU 构造参数、
cache roundtrip/损坏拒绝/decoder 一致性、训练绕过 detector、coverage 过滤和模式门禁。
本轮未安装任何本地依赖；远程 requirements 固定 `facenet-pytorch==2.5.3`，
保持 torch==1.13.1 / torchvision==0.14.1 参考路线不变。
`REMOTE_GPU_SMOKE_REQUIRED=YES`。

2026-11-02 本地检查结果：定向 pytest **25 passed**；新增适配/测试与上游源码
合计 **37 个 Python 文件 py_compile 通过**；`--check-config` 通过且未打开数据/权重；
`git diff --check` 通过。没有运行全量历史测试、实际官方 S2D forward 或任何训练。

本轮 2026-11-03 的本地 face pipeline 检查结果见下方最终验收记录。

远程下一步（在独立 S2D 工作区执行，不干预现有历史长训练）：

1. 同步经用户处理的代码版本，记录 commit；在远程独立环境核对上述依赖版本。
2. 填 data_root/manifest_path/cache_path，运行 `--check-config`；确认 7433/7423/10。
3. 先用本次正式 MTCNNFaceProcessor 在 `Ses02F_script01_3_M001` 重做 Uniform16
   单 utterance smoke，核对 bbox、crop、probability、flags 和 224×224 输出。
4. pilot 通过后，对 7423 个 V-available utterance 做一次 CPU 预处理，保存完整 cache，
   汇总 face_usable/全失败/有效帧/fallback 覆盖率并抽查背景脸跟踪；不要预先声称全量 PASS。
5. 切回 cached_bbox，验证不触发 MTCNN；填官方两种权重并核对 load keys，运行
   S2D GPU single-batch forward/backward 和显存 pilot；通过后才启动 V0 seed100 正式训练。

所有真实数据、权重和动态产物均不纳入 Git。此任务没有 commit/push 或远程操作。

### 2026-11-03 本地 face pipeline 验收

- `python -B -m pytest tests/models/s2d -q -p no:cacheprovider`：**65 passed**，
  包括原有 25 个 V0 synthetic 测试与新增 40 个 face mock 测试。
- `py_compile`：本轮 6 个新增/修改 Python 文件及未改动 Uniform16 共 **7 个文件 PASS**。
- `--check-config`：**PASS**，`CONFIG_VALID_ONLY`，未打开数据/权重；
  正式训练仍需远程填写数据、cache 及两种 checkpoint 路径。
- `git diff --check`：**PASS**。
- `LOCAL_S2D_FACE_PIPELINE_READY_FOR_REMOTE_SMOKE=YES`。

这些本地结果不代表真实 S2D GPU smoke 或全量 IEMOCAP face detection 已完成；
未改 MultiDAG、TA backbone、MMGCN、S2D 核心 MCP/TMA、既有权重或数据。
