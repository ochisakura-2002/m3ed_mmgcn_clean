# Frozen TA/V cache、Concat-Linear 与 VRG-v0

本实现只组合冻结表示，不导入或训练 TA、MultiDAG、S2D。当前交付验证仅使用
synthetic fixtures；真实 IEMOCAP Train/Dev 尚未运行，Test 不进入本流程。
VRG 研究的问题是视觉信息在 TA 出错时能否帮助纠错、在 TA 正确时能否减少干扰。
gate 的 Q01/Q10 均值差只作为机制事实，不据其正负判定方法有效。

## 输入与缓存

TA 固定声明为 TA-v0 seed100、best epoch18。输入两个显式 NPZ，分别为 Train
与 Dev，包含 `utterance_id[N]`、`dialogue_id[N]`、`label[N]`、
`h_ta[N,1500]`、`logits_ta[N,6]`、`prob_ta[N,6]`。ID 使用字符串数组，
不允许 pickle/object 数组。可选 `split` 元数据必须与调用方声明相符；
没有 split 字段时记录 caller-declared，不能从 NPZ 本身验证来源 checkpoint 或真实 split。

V 使用现有 `s2d_uniform16_feature_bank_v1`：显式读取 `provenance.json`、
`coverage.json`、`complete.json` 与 `train/index.json`、`dev/index.json`，
只打开 index 声明的 `part-00000.pt` 等 direct-child shards，使用
`torch.load(weights_only=True, map_location='cpu')`。校验 epoch7、Uniform16、
float32、六类词表、feature contract 和完成标记。两个 split 的 index 和 coverage
先全部校验，再读取 shard；不 glob、不回退到其他目录。

V representation 的唯一身份是 **S2D-B mean contextual slots**：
`h_v = s2d_contextual_slots.mean(dim=1)`，输入 `[N,16,768]`，输出 `[N,768]`。
stored `h_v`、A0/C0/D0、original `logits_v` 都不参与融合或 quadrant prediction。

按 `utterance_id` exact join，结果按 ID 排序。重复 ID、label mismatch、V-only ID、
跨 Train/Dev 重叠、unknown V exclusion 均失败。TA-only IDs 必须与 coverage 的
`v_unavailable`、`cached_face_unusable` exclusions 完全相同。
正式计数固定为 Train `4830/4818/12`、Dev `980/978/2`（TA/common/TA-only）；
`--synthetic` 才允许小型计数，cache 标记不能混入正式 runner。

新目录内容：

| 文件 | 合同 |
| --- | --- |
| `train_common.npz` | `utterance_id`, `dialogue_id`, `label`, `h_ta[N,1500]`, `h_v[N,768]`, `ta_logits[N,6]`, `ta_prob[N,6]`, scalar `split=train` |
| `dev_common.npz` | 同上，`split=dev`；提供显式 Linear-B 文件后还包含 `v_probe_pred[N]`, `v_probe_prob[N,6]` |
| `normalization.npz` | float64 `mu_ta[1500]`, `sigma_ta[1500]`, `mu_v[768]`, `sigma_v[768]` |
| `cache_manifest.json` | schema、complete 状态、source declarations、split/join/exclusion 审计、normalization 合同、输入与输出 SHA256；最后写入 |

normalization **只拟合 Train common** 的 raw features：逐维 population mean/std，
`ddof=0`，分母 `max(std,1e-6)`。先对 B 做 mean pooling，再计算其逐维统计量，
不同于视觉 probe 的 utterance+slot 归一化。Dev 只应用同一组统计量；标准化结果
转为 float32。runner 重新从 Train common 计算统计量并核对，拒绝使用 Dev 或
Train+Dev 统计量的 cache。

## B1 / B2 与配置开关

`configs/asymmetric_merc/frozen_tav/seed100.yaml` 的 `vrg.enabled: false` 直接创建
Concat-Linear，与 B1 的 state schema、初始化和行为完全相同；设为 true 创建 B2。
CLI `--model concat_linear|vrg` 覆盖同一开关。无 TA/S2D checkpoint schema 变化。
新线性层使用 PyTorch 默认初始化，受统一 seed 控制。

| 模型 | 结构 | 可训练参数 |
| --- | --- | ---: |
| B1 | concat(z_TA,z_V) → Linear(2268,6)，无 hidden MLP | 13,614 |
| B2 | ReLU(1500→300), ReLU(768→300)；q=[t,v,abs(t-v),t*v]；r=sigmoid(Linear(64,1)(ReLU(Linear(1200,64)(q))))；Linear(300,6)(t+r*v) | 759,735 |

B2 gate 是 `[B,1]` scalar，t/v 为 `[B,300]`，logits 为 `[B,6]`。
数学上 sigmoid 在 (0,1)；极端输入下 float32 可能舍入到0或1。
诊断保留原值并记录 `finite_precision_endpoint_count`，不 clip、不改变模型公式。
无 vector gate、attention、MoE 或新 loss，也不增加 missing-V mask architecture。

## 冻结训练协议

两模型完全相同：seed100，CrossEntropy，AdamW，lr=1e-3，weight_decay=1e-2，
batch_size=128，max_epochs=100，patience=10，Train shuffle=true，Dev=false。
按 Dev Weighted-F1 降序、Dev loss 升序、epoch 升序选 checkpoint；精确浮点比较，
无额外 tie tolerance。仅 loss 改善且 Weighted-F1 相同时也视为改进并重置 patience。
无 scheduler、class weights、label smoothing、sweep 或自动多 seed launcher。
runner 支持单次 `--seed`，首轮正式运行限制为100；其他 seed 只允许 synthetic。
`--smoke-epochs 1..3` 仅用于明确标记的 synthetic cache。

best checkpoint 保存后，重新以 strict=True 加载并按 Dev 顺序导出：
`best_model.pt`、`epoch_metrics.csv`、`best_dev_predictions.csv`、
`best_dev_metrics.json`（Accuracy/W-F1/Macro-F1/UAR/loss/六类 PRF/support/best epoch）、
`ta_common_dev_metrics.json`、`resolved_config.yaml`、`run_manifest.json`。
manifest 记录 Git HEAD/status、运行版本、参数量、输入/code/artifact 哈希及隔离字段。
输出目录必须全新，已有空目录也拒绝，不实现 resume/overwrite。

## Quadrant 与 gate 分析

`--v-quadrant-predictions` 显式声明 **Q2-1 frozen Linear-B seed100 Dev predictions**，
复用现有 Q2-2 reader，接受 CSV/TSV：`utterance_id`、`true_label`、
`pred_label` 或 `predicted_label`、`prob_0..prob_5`，可带 Dev split 列。
拒绝重复 ID、非 Dev split、label mismatch、非概率或 argmax mismatch；
预测 IDs 必须恰好等于 Dev common IDs。CLI 是 source identity 的声明，文件内容本身
不能证明 probe 的训练 seed；哈希及这一限制都会保留。

cache builder 可以不提供此文件，此时不伪造 `v_probe_*`。B1 可继续运行，B2 必须在
cache 内已有 prediction 或 runner 显式提供该文件后才能训练。
已有 cached prediction 时 runner 只允许同哈希的外部文件。

- Q11：TA correct / V correct。
- Q10：TA correct / V wrong。
- Q01：TA wrong / V correct。
- Q00：TA wrong / V wrong。

VRG best checkpoint 输出逐 ID `dev_gates.tsv`：
`utterance_id, dialogue_id, true_label, ta_pred, v_pred, vrg_pred, gate, quadrant`。
`gate_mechanism.json/.md` 给出 all、四象限、六个 true class 的 count/mean/std，
以及 `delta_rescue_harm=mean(Q01)-mean(Q10)`；std 使用 ddof=0，空组为 null。
报告无 gate 正负 PASS 判据。

B0 仍是已冻结 alpha=0.88 的 late fusion，参考 Dev common n=978、W-F1=0.570588、
Acc=0.592025、Macro-F1=0.506756、UAR=0.524482。本实现不搜索 alpha、不训练 B0，
也不把这些参考值填入 B1/B2 结果。V-invalid → TA-only fallback 的 full-split
evaluation 留待最终架构冻结后；本轮只运行 common V-valid。

## 后续 Ubuntu 调用模板（本任务未执行真实数据）

从仓库根目录传入真实资产相对路径或 shell 变量，不创建机器路径默认值。
`$CACHE_DIR` 应为 `outputs/<YYYYMMDD>/asymmetric_merc_frozen_tav/analysis/batches/<batch_id>/frozen_tav_cache`；
`$B1_RUN`、`$B2_RUN` 为同日期/组下 `runs/<new_run_id>`。日期由调用方在批次开始时确定。

```bash
python scripts/models/asymmetric_merc/build_frozen_tav_cache.py \
  --ta-train-npz "$TA_TRAIN_NPZ" --ta-dev-npz "$TA_DEV_NPZ" \
  --s2d-bank-dir "$S2D_BANK_DIR" --v-quadrant-predictions "$LINEAR_B_SEED100_DEV" \
  --output-dir "$CACHE_DIR"

python scripts/models/asymmetric_merc/train_frozen_tav.py \
  --cache-dir "$CACHE_DIR" --output-dir "$B1_RUN" --model concat_linear --seed 100

python scripts/models/asymmetric_merc/train_frozen_tav.py \
  --cache-dir "$CACHE_DIR" --output-dir "$B2_RUN" --model vrg --seed 100
```

## Test fail-closed 与本地验证

没有 Test loader、Test CLI 输入或 Test evaluation 分支。`--split test` 在 argparse
即失败；显式 Test-named input path（含 resolved symlink target）在文件 IO 前拒绝；
TA split/Test fields、bank metadata/index/shard split、越界 shard filename 和 cache split
都受校验。裸文件的实际 split 若被移除或恶意改名无法仅凭内容证明，故明确保留声明边界。

synthetic tests 用 monkeypatch 令任何提前发生的 `np.load` / `torch.load` 直接失败，
证明 forbidden path/split 在读取资产前被拦截、两个 bank index 在任何 shard IO 前完成校验。
还覆盖独立乱序 join、duplicate/label mismatch/unknown exclusions、Train-only stats、
constant dimensions、B1/B2 shape/backward、参数量、四象限及 gate statistics、
best checkpoint 重载、单次 CLI smoke、overwrite/hash/mode 隔离与 canonical outputs。

```powershell
conda run --no-capture-output -n m3ed_mmgcn python -m pytest tests/models/asymmetric_merc -q -p no:cacheprovider
```

测试只生成 synthetic 临时资产到既有忽略目录 `tmp/assistant_work/frozen_tav_tests/`。

## 本次交付验证（2026-11-12）

- WORKSTREAM=ASYMMETRIC_MERC；guard PASS。
- 起始 branch：`integration/asymmetric-merc-v0`。
- 起始 HEAD：`41ddbd0ba4ee2c5034bc3b41fc3b65e580b037a5`；起始工作区干净。
- 上述 pytest 命令：**98 passed**（新增34项 synthetic 检查，既有 Q2-2 回归64项）。
- B1/B2 synthetic train → best checkpoint → strict reload → Dev predictions/gates 已通过。
- early-stopping fixture：epoch2 的 loss tie-break 改进重置 patience，随后精确平局，epoch12 停止。
- `--split test` 在 argparse 以 exit2 拒绝；Test path 与错误 bank split 在 shard IO 前拒绝。
- 新增 Python 文件 `py_compile` 通过；tracked `git diff --check` 与10个新文件的 no-index whitespace 检查通过。
- 仅新增本任务10个文件，冻结 Q2-1/Q2-2 数值逻辑与 MultiDAG/S2D 源码未改。
- 未运行真实 IEMOCAP、未访问 Test、未 commit/push。
