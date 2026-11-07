# S2D Q1 Representation Probe Protocol — 2026-11-07

WORKSTREAM = S2D

**THIS DOCUMENT DOES NOT CLAIM REAL PROBE RESULTS**

本任务交付独立 frozen-feature probe、诊断表格和组会归档接口。Windows 仅执行
synthetic tests；不打开真实 Ubuntu bank、不训练真实 probe、不访问 Test、不改动
S2D、不重导 bank，不涉及 Audio guidance、TA/V fusion、Q2 或 Q3。

本地现有 `m3ed_mmgcn` 环境验收：S2D 定向回归 **338 passed, 11 skipped**。
其中新增 probe 专项 **51 项全部通过**；现有 11 个 canonical-core case 因缺少
`timm/einops` 明确跳过，没有安装依赖或以 mock 冒充完整模型数值验收。
真实 CLI `--help` 通过。两份进入任务前已有的测试修改按 SHA256 核对保持原样。

入口：`scripts/models/s2d/iemocap/run_probes.py`。模型为独立的
`models/s2d/iemocap/probes.py`；reader/report 分别为同目录脚本
`probe_bank.py` / `probe_reports.py`。runner 不导入原 S2D model、训练器、exporter
或 finalizer，不需要 timm/einops、视频、cache 或原 checkpoint。

## 科学问题与解释边界

Q1 是“冻结视觉表示是否含有轻量分类器能够读取的情绪信息”，不以寻找最高分
classifier 为目标。D0 和 B 为主比较，A0/C0 为诊断。

| 表示 | Bank 字段 / shape | 回答的问题 |
| --- | --- | --- |
| A0 | `a0_rgb_mean [N,16,768]` | frozen RGB patch 底座是否已有可读信息 |
| C0 | `c0_mcp_static_mean [N,16,768]` | landmark-aware MCP0 静态支路的诊断信息 |
| D0 | `d0_static_residual_mean [N,16,768]` | RGB residual + MCP0 static correction 的独立表示是否可用 |
| B | `s2d_contextual_slots [N,16,768]` | 完整 contextual S2D 是否保有可读信息 |

`mean_t(B) ≈ h_v`。reader 用 `rtol=1e-5, atol=1e-6` 验证此既有表示合同；
这与下述严格 baseline score 门禁是两种不同的 tolerance。
Linear-B 与原 `fc_cls(h_v)` 的比较用于诊断 classifier/training utilization gap。

A0/C0 来自不同 residual branch，比较名称不写成简单“增益”。D0-A0 是 MCP
静态修正的净作用；B-D0 包含完整 contextualization 的多个组件（TMA、ViT、
temporal embedding 等），不能单独归因为某个时序算子。

所有 TCN 卷积均为 causal left padding，且只在 D0/B 上运行。它仍是
**offline diagnostic temporal probe**：最终对全 utterance 池化；B 上游已含
non-causal TMA，后接 causal TCN 不能消除未来信息。D0 独立性只针对固定 crop
的 encoder 子图，原 bbox tracking/future-bbox fallback 和 Uniform16 的整段输入
仍有上下文依赖。本协议不声称机器人 causal deployment。

Scenario A（所有表示低）、B（D0/B probe 高于原 head）、C（B 高于 D0）、
D（TCN-D0 高于 Linear-D0）分别为未来考察 domain mismatch、utilization、
contextualization、sequence information 的证据方向。代码只生成表格，不规定
“低”“明显”或“显著”的阈值，不自动写科学结论，不把 seed 波动当统计显著性。

## Frozen bank 与 baseline reproduction gate

用户已验收的 Ubuntu bank（本轮未打开）：

```text
outputs/20261107/s2d_q1_feature_bank/analysis/seed100_epoch7_uniform16_v1
Train: manifest 4830 / exported 4818 / excluded 12 / 19 shards
Dev:   manifest  980 / exported  978 / excluded  2 /  4 shards
unknown_exclusion_count = 0
```

reader 按 exporter v1 合同检查 complete/provenance/coverage、固定 label mapping
`hap,sad,neu,ang,exc,fru`、Uniform16、float32、int64 labels/frame indices、shapes、
finite values、全局唯一 ID、index 顺序、精确 coverage/exclusion accounting。
只读取显式 bank 内 `train/index.json`、`dev/index.json` 列出的 canonical shard；
不递归发现输入，不跟随 source_run_path/checkpoint 等 provenance 中的路径。
拒绝 traversal/非 canonical shard 名、跨目录或 symlink shard。

任何拟合（包括 Most Frequent）前，从 Dev `logits_v.argmax` 和 labels 复算共享
`utils.metrics` 的 Accuracy / Weighted-F1 / Macro-F1 / UAR，并输出每类
Precision/Recall/F1/support 和固定六类 confusion matrix。指标语义与原训练
`classification_report` 一致，`zero_division=0`，所有六类参与 macro/UAR。

固定 source evidence：**selected epoch=7，Dev Weighted-F1=0.34355913724532466**。
允许 `abs(observed-expected) <= 1e-10`；CLI/YAML 不支持覆盖 score、放宽 tolerance
或跳过门禁。provenance 也须声明同一 epoch/score。

若数值不一致，写 reference metrics/predictions/confusion 和
`baseline_reproduction_gate.json`（expected/observed/difference/tolerance/
`PROBE_TRAINING_ALLOWED=NO`），立即异常退出；不产生 dummy、probe 或 complete。
metadata/schema 失败可更早退出。不得通过放宽 tolerance 或改 constant 解决真实差异，
应先核查 exported Dev 子集、排除样本、labels、logits 与历史指标的计算路径。
本任务只证明门禁代码可工作，不宣称真实 bank 已通过该门禁。

## 固定 Linear / TCN / optimization protocol

正式配置：`configs/s2d/iemocap/q1_probe.yaml`。

| 项目 | 固定规则 |
| --- | --- |
| Linear | A0/C0/D0/B 各 `mean_t -> Linear(768,6)`，有 bias，无 hidden layer/dropout |
| TCN | 只 D0/B；1×1 projection 768→128；两 residual blocks，各一个 kernel=3 conv；dilation 1/2；GELU；dropout=.1；mean_t；Linear(128,6) |
| 优化 | unweighted CrossEntropy；AdamW；lr=.001；weight decay=.01；无 scheduler |
| 预算 | max epochs=100；batch=128；patience=10；CPU threads=1；loader workers=0 |
| 选模 | 每 epoch 全量 Train eval 与 Dev eval；Dev Weighted-F1 严格提高才更新；相同保留更早 epoch；无 improvement 连续 10 轮停止 |
| Seeds | 100/101/102/103/104，每种 trainable probe 五次，共 30 runs |
| Dummy | Most Frequent，只拟合 Train label counts；平票取最低 class index；非随机，无 seed 搜索 |

所有 feature/type 共用同一份 optimization protocol，无 per-feature override、参数网格或
seed 子集 CLI。可审查地修改全局 YAML 会形成另一个显式 protocol，必须保留解析后的
config，不能将差异归因于 representation。`enabled=false` 或 `--preflight-only`
只执行 reference/dummy 和其汇总，`complete.mode=preflight_only`、trainable_runs=0；
不会修改 S2D baseline 或给它注入模块。默认 enabled=true 运行全部 30 probes。

每 seed 重置 Python/NumPy/Torch/CUDA RNG；shuffle 使用独立 seeded generator；
强制 deterministic algorithms，CUDA cuBLAS workspace `:4096:8`，禁用 cuDNN benchmark。
不支持的 deterministic kernel 直接失败，不自动降级；记录设备和软件版本。
保证同环境可重复，不承诺跨 CPU/GPU/软件版本 bitwise 相同。

## Normalization / frozen contract

默认 `train_channel_standardize`：每个 A0/C0/D0/B 分别在 **Train 全部 utterance 和
16 slots** 上拟合每通道 population mean/std（ddof=0），std 下限 `1e-6`。
同一 feature 的该固定 affine transform 同时用于 Linear 和 TCN；它与 mean pooling
交换，因此 Linear 与 TCN 的比较不会混入不同归一化拟合规则。Dev 不拟合统计量。
`normalization: none` 是显式全局关闭路径，四种表示同样处理。

四份 JSON 保存 mean/std、fit_split、axes、count、epsilon、ddof 与 Train index identity；
每 seed config 引用相应 normalization 文件及其 SHA256。bank tensors 均为 detached
float32，梯度只进入新 probe 参数。逐 shard 验证 bank，一次仅物化一个表示/模型类型，
不把四种完整序列同时常驻内存。每次重读验证 SHA256；complete 前重查全部输入 identity。
SHA256 用于输入固定性追踪，不替代外部 provenance 可信性验收。

## Split discipline

Train 仅用于拟合；Dev 用于 early stopping、checkpoint selection、诊断比较。
**Test forbidden**。CLI 不提供任何 split 选项，未知参数（包括 `--split test`）在
bank IO 前失败。complete/provenance/coverage 必须恰好为 Train/Dev，出现 Test
split 声明直接拒绝。index/shard 的 split 也严格校验；无 all-splits fallback、
历史 predictions 查找或 finalizer 调用。额外未被 metadata 声明的文件不枚举、不打开。

## Output archive / prediction / summary schema

显式新目录：`outputs/<YYYYMMDD>/s2d_q1_probe/analysis/<probe_run_id>/`。

```text
00_provenance/
  provenance.json                 # code HEAD / runtime / mode / device / seeds
  feature_bank_reference.json     # original provenance/coverage + all input identities
  protocol.json                   # resolved protocol, architecture, selection boundaries
  normalization/{a0,c0,d0,b}.json
01_reference/
  original_s2d_dev_metrics.json
  original_s2d_dev_predictions.csv
  original_s2d_per_class_metrics.csv
  original_s2d_confusion_matrix.csv
  baseline_reproduction_gate.json
02_dummy/{config.json,metrics.json,predictions_dev.csv,per_class_metrics.csv,confusion_matrix.csv}
03_linear/{a0,c0,d0,b}/seed{100,101,102,103,104}/
04_tcn/{d0,b}/seed{100,101,102,103,104}/
05_summary/
  metrics_long.csv
  metrics_summary.csv
  pairwise_deltas.csv
  per_class_metrics.csv
  confusion_matrices/<model_seed>.csv
  training_curves/metrics_by_epoch.csv
complete.json
```

每个 trainable seed 保存 `config.json`、`metrics_best.json`、`metrics_history.csv`、
`predictions_dev.csv`、`per_class_metrics.csv`、`confusion_matrix.csv`、`best_model.pt`。
checkpoint 为纯 probe tensor state_dict；normalization 保存在上面单独的 JSON。
训练中最优 state 保留在内存，结束后 strict reload、重算最优 Dev score 再一次性写出。
history 保留 optimization loss、完整 Train eval loss/四指标、Dev loss/四指标与 epoch。

Prediction schema：`utterance_id,true_label,pred_label,prob_0,...,prob_5`，保留 bank 的
Dev ID 顺序，可后续按 ID join；概率为 float32 softmax，dummy 为 one-hot。
per-class 行为 `label_index,label,precision,recall,f1,support`。confusion CSV 为 6 行，
`true_label,true_label_name,pred_0,...,pred_5`；行=真值、列=预测，保留零支持类别。

`metrics_long.csv`：`model,feature,probe_type,seed,best_epoch,acc,weighted_f1,macro_f1,uar`。
reference/dummy 的 seed 为空，不当成五次重复；每个 trainable model 五行，全部共 32 行。
`metrics_summary.csv`：`model,feature,probe_type,n_seeds` 及每项指标的 `_mean,_std`。
五 seeds 使用 sample std（ddof=1）；reference/dummy 只有一次，std=0，仅表示固定值。
主比较为 **Dev Weighted-F1 mean ± std**，并列报告 Macro-F1、UAR、Accuracy。

`pairwise_deltas.csv` 自动产生 C0-A0、D0-A0、B-D0、LinearB-OriginalS2D、
TCND0-LinearD0、TCNB-LinearB 的四项指标。每种比较/metric 保存 per_seed delta
与 summary delta_mean/delta_std/n_pairs；训练模型按同一 seed 配对后求差，再聚合。
OriginalS2D 是固定参照，每 seed 减同一值。preflight-only 的 delta 表只有 header。

## Atomicity 与组会归档计划

根目录 `mkdir(exist_ok=False)` 独占预留；已有空目录、中断目录和并发占用均拒绝。
不提供 overwrite/resume。单文件先写同目录临时文件、flush/fsync/close，随后通过
`os.link` 原子、不覆盖地发布，finally 清理临时文件。文件系统不支持 hard link 时失败，
不回退到非原子写入。部分成功文件保留用于审计；缺少 complete 的目录不得当成功。

全部 seed、summary 和输入最终 SHA 检查成功后，最后写 `complete.json`。其中包含
mode、trainable_runs、Test=false 和此前全部产物的 SHA256/size inventory。该 marker
是最后发布的产物，不是整个目录的事务或断电恢复承诺。

后续组会可直接消费 summary、per-class、固定 label confusion matrices 与长格式训练
曲线，生成 baseline vs probe、A0/C0/D0/B、mean±std、per-class F1、static vs contextual、
linear vs temporal 图。本任务不生成最终组会图，也不只向 terminal 打印指标。

## Local validation 与下一次 Ubuntu validation

专项测试仅构建标明 `TEST_SYNTHETIC` 的小 bank。pytest 内将 expected score 显式替换成
合成 baseline 的 1.0；生产常量单独测试，CLI/YAML 无任何门禁 bypass。
全矩阵 synthetic smoke 使用同样六种真实 probe 架构和五 seeds，只缩短为 2 epochs。

```powershell
$env:PYTHONNOUSERSITE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
conda run --no-capture-output -n m3ed_mmgcn python -B -m pytest tests/models/s2d/test_s2d_probes.py -q -p no:cacheprovider
```

未来 Ubuntu 先执行工作区 guard 和相同 synthetic 测试（期望 0 failed / 0 skipped）。
代码交接由用户另行安排；本轮不连接服务器或执行 Git 写操作。
以下是真实 bank 验证/运行模板，**本轮未执行**；实际执行需单独确认远程任务范围。
两次调用必须使用不同、尚不存在的 output directory：

```bash
python scripts/dev/check_workstream.py s2d
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -B -m pytest tests/models/s2d/test_s2d_probes.py -q -p no:cacheprovider

python -B scripts/models/s2d/iemocap/run_probes.py \
  --feature-bank outputs/20261107/s2d_q1_feature_bank/analysis/seed100_epoch7_uniform16_v1 \
  --config configs/s2d/iemocap/q1_probe.yaml --device cpu --preflight-only \
  --output-dir outputs/20261107/s2d_q1_probe/analysis/epoch7_preflight_v1

python -B scripts/models/s2d/iemocap/run_probes.py \
  --feature-bank outputs/20261107/s2d_q1_feature_bank/analysis/seed100_epoch7_uniform16_v1 \
  --config configs/s2d/iemocap/q1_probe.yaml --device cuda \
  --output-dir outputs/20261107/s2d_q1_probe/analysis/epoch7_q1_five_seeds_v1
```

上述日期与 run ID 为示例，未来按实际启动日期显式指定。真实 baseline 若不一致，
停止真实训练并检查保留的差异证据；不能因为 bank integrity PASS 就跳过该数值门禁。
