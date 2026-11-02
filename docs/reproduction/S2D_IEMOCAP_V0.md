# S2D-IEMOCAP V0 接入说明

本地交付目标为 `CODE READY FOR REMOTE SMOKE`。这里只新增独立 V 分支，
未修改已有 MultiDAG/TA/MMGCN 的模型、训练逻辑、标签或 split。
没有执行真实视频 smoke、正式训练、权重加载或 GPU 显存测试。

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
可在 `dataset.columns` 显式映射实际列名，例如 `video_path: video_clip_path`。
真实远程 manifest 列名尚未读取，不能把这个逻辑 contract 当作真实 header 已验证。

`utterance_id` 全局唯一；split 只能使用已有 `train/dev/test`，不产生新划分。
`v_available` 接受 `0/1/false/true`，为 1 时 video path 不得为空。
原始完整 universe 保存在 `S2DManifest.records/by_id`，V dataset 仅过滤不可用 V，
并在远程 run 的 `metrics/coverage.json` 保存全量/可用/不可用计数及缺 V 的 ID。
已知 10 条无 V 样本保留在全量语义中；这里不把 10 硬编码成过滤数量。

`data_root` 由配置提供，相对路径以项目根目录为基准；`manifest_path` 相对
`data_root` 解析。CSV 中相对视频路径由 `data_root + relative_path` 解析，
也支持当前 OS 的绝对路径；拒绝在 Linux 静默重解释 Windows 绝对路径。
解析器不读取视频、不检查本地真实数据是否存在。

## Uniform16、FaceProcessor 与 transform

Uniform16 的索引为 `i * (N-1) // 15, i=0..15`，覆盖 utterance clip 两端，
不依赖模型或 label；短视频重复索引，N<=0 报错。默认远程 decoder 使用
decord 只取采样帧，RGB 输出，不缓存、不随机换取其他 utterance。

`FaceProcessor` 输入 PIL RGB frame；backend 返回
`FaceResult(face=<RGB crop/aligned image or None>, face_valid=<bool>, reason=<str>)`。
成功后双线性 resize 到 224×224；失败须提供 reason，发出 warning，并由 dataset
携带 utterance ID 抛错，不生成补帧或永久 face cache。
真实 backend 通过 `face.backend: module:factory` 和 `face.kwargs` 接入。
这里仅提供 `mock_full_frame` 测试后端，它不进行人脸检测，正式训练/评估禁止使用。
detector 的选择、对齐实现、speaker 身份及真实视频验证均留在远程阶段。

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
复用 `utils.metrics`。指标总体是 V-available 子集，须结合 coverage 报告解释。
保存 resolved config、项目/上游 commit、seed、label vocab、加载报告、epoch 日志，
最终 `test_final.json`、按 utterance ID 的 logits/prob CSV；
`export.representations=true` 时额外保存 h_v/logits_v/prob_v tensor 文件。

## 本地检查与远程交接

本地检查不要求数据、视频、checkpoint 或完整视觉依赖：

```powershell
conda run -n m3ed_mmgcn python -B scripts/models/s2d/iemocap/train.py --config configs/s2d/iemocap/v0.yaml --check-config
conda run -n m3ed_mmgcn python -B -m pytest tests/models/s2d/test_s2d_v0.py -q -p no:cacheprovider
```

测试 fixture 仅 4 行，所有 ID/字段均标明 `TEST_SYNTHETIC`；正式 YAML 不引用它。
测试覆盖 import、vocab、Uniform16（N=5/16/31/100，另含 N=1/非法输入）、
parser、split/缺 V/路径/冲突标签、mock face/transform、wrapper contract double、
checkpoint 模式、指标/导出和源码来源一致性。
wrapper double 仅用于检查接口和冻结规则，**不是官方 S2D forward 的测试结果**。
本地 Python 3.9 + CPU torch 已有；缺少 timm/einops/decord，此次未安装依赖。
`REMOTE_GPU_SMOKE_REQUIRED=YES`。

2026-11-02 本地检查结果：定向 pytest **25 passed**；新增适配/测试与上游源码
合计 **37 个 Python 文件 py_compile 通过**；`--check-config` 通过且未打开数据/权重；
`git diff --check` 通过。没有运行全量历史测试、实际官方 S2D forward 或任何训练。

审核通过后再由用户决定提交/推送。远程交接依次完成：独立环境与依赖，
真实 manifest header/TA vocab/path 确认，detector/crop/alignment 验证，
single-sample 视频 smoke，官方权重加载与 keys 核对，single-batch forward/backward、
GPU memory pilot；这些通过后才执行 V0 seed100 正式训练。
所有真实数据、权重和动态产物均不纳入 Git。此任务没有 commit/push 或远程操作。
