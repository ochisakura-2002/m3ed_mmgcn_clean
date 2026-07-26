# DialogueGCN 与 GS-MCC full 修复实验准备

阶段：2.1；实验组：`dialoguegcn_gsmcc_full_repair`

状态：只完成静态诊断、公共诊断能力、`min_epochs` 和候选矩阵准备；未训练、未提交、未 push。

## 1. 证据边界

结论按以下等级记录：

- `CONFIRMED`：由代码、版本化配置或保留的正式分析表直接证明。
- `HIGH_CONFIDENCE`：多项现有证据一致支持，但仍需要诊断 run 验证因果解释。
- `HYPOTHESIS`：本轮候选实验要证伪的解释。
- `NOT_EVALUATED`：现有本地文件不能证明；数值写 `UNKNOWN`，不补猜。

正式训练对应 commit 为
`d8a7701b62339a6908a39de5d800cb816174ec90`。保留的分析入口为：

```text
outputs/20260726/formal_long32_primary_seed42/analysis/long32_primary_audit/
  attempts/attempt_20260726_234056_065469/
```

分析 manifest 指向的原始输入包
`_local_analysis_inputs/merc_long32_20260726/merc_long32_codex_plot_package_20260726`
当前不在本机。因此：

- 保留分析表中出现的旧 resolved-config 字段可以审计。
- 未进入派生表的旧 resolved-config 字段为 `UNKNOWN`。
- 不使用当前 YAML 猜测旧 run。
- 本机没有 Clean RoBERTa feature PKL；资产级统计为 `NOT_EVALUATED`。

任务开始时 tracked diff 为空，存在一个预先已有且与阶段交接直接相关的未跟踪文件
`docs/PROJECT_HANDOFF.md`。该文件未被本任务修改。

## 2. 正式运行与异常事实

| Run | Best / stop epoch | Best val WF1 | Test WF1 | 类别覆盖与输出 |
|---|---:|---:|---:|---|
| DialogueGCN Ses01 | 25 / 35 | 0.4315 | 0.4548 | test 6 类，dominant 0.3358，entropy 1.3386 |
| DialogueGCN Ses02 | 25 / 35 | 0.4697 | 0.4138 | test 6 类，dominant 0.2785，entropy 1.4377 |
| DialogueGCN Ses03 | 24 / 34 | 0.4435 | 0.4605 | test 5 类，dominant 0.3081，entropy 1.3998 |
| DialogueGCN Ses04 | 1 / 11 | 0.0497 | 0.0905 | test 1 类，dominant 1.0000，entropy 1.7903 |
| GS-MCC Ses01 | 4 / 24 | 0.1754 | 0.1662 | test 5 类，dominant 0.4301，entropy 1.7841 |
| GS-MCC Ses02 | 6 / 26 | 0.1656 | 0.2057 | test 4 类，dominant 0.4288，entropy 1.7785 |
| GS-MCC Ses03 | 175 / 195 | 0.4209 | 0.4781 | test 5 类，dominant 0.3734，entropy 1.2921 |
| GS-MCC Ses04 | 205 / 225 | 0.4994 | 0.4707 | test 6 类，dominant 0.3913，entropy 1.2330 |

`CONFIRMED`：

- DialogueGCN Ses04 是 near-uniform single-class collapse，不是高置信度 Neutral
  偏置。其 test mean top1-top2 margin 约为 0.00292，entropy 接近
  `ln(6)=1.79176`。
- DialogueGCN Ses03 在 epoch 11 才首次达到 val WF1 0.2，在 epoch 20
  才达到 0.4；Ses04 恰在 epoch 11 停止。
- GS-MCC Ses03/Ses04 首次达到 val WF1 0.2 分别在 epoch 47/37，首次达到
  0.3 分别在 epoch 62/82。
- GS-MCC Ses01/Ses02 的 total loss 从 2.0704/2.0746 降至
  1.7923/1.7795；classification contribution 从 1.8201/1.8217 降至
  1.7192/1.7109；由代码公式和两者差值推得的 weighted contrastive
  contribution 从 0.2503/0.2529 降至 0.0731/0.0686。

`HIGH_CONFIDENCE`：

- DialogueGCN Ses04 存在 early-stopping 截断延迟学习转折的风险。
- GS-MCC full 存在低学习率、短 patience 与 Session-sensitive 慢学习的交互风险。

`HYPOTHESIS`：

- 延迟停止可使 DialogueGCN Ses04 脱离近常量输出。
- GS-MCC Ses01/Ses02 可通过延迟停止或单一 LR 候选进入有效学习区。

以上都不是“修复成功”结论。

## 3. 数据与 split 静态审计

两个模型使用同一 Clean feature、同一 `session_holdout` 数据入口，因此下面的 split
统计同时适用于 DialogueGCN full 和 GS-MCC full。Val/Test 数量来自正式预测真值；
Train 数量由 Ses01--Ses04 四个完整 Validation Session 的真值分布做确定性补集得到。

单元格格式为 `数量 (比例)`。

| Val Session | Split | Utterances | Happy | Sad | Neutral | Angry | Excited | Frustrated |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ses01 | Train | 4437 | 361 (8.14%) | 645 (14.54%) | 940 (21.19%) | 704 (15.87%) | 599 (13.50%) | 1188 (26.77%) |
| Ses01 | Val | 1373 | 143 (10.42%) | 194 (14.13%) | 384 (27.97%) | 229 (16.68%) | 143 (10.42%) | 280 (20.39%) |
| Ses01 | Test Ses05 | 1623 | 144 (8.87%) | 245 (15.10%) | 384 (23.66%) | 170 (10.47%) | 299 (18.42%) | 381 (23.48%) |
| Ses02 | Train | 4454 | 379 (8.51%) | 642 (14.41%) | 962 (21.60%) | 796 (17.87%) | 532 (11.94%) | 1143 (25.66%) |
| Ses02 | Val | 1356 | 125 (9.22%) | 197 (14.53%) | 362 (26.70%) | 137 (10.10%) | 210 (15.49%) | 325 (23.97%) |
| Ses02 | Test Ses05 | 1623 | 144 (8.87%) | 245 (15.10%) | 384 (23.66%) | 170 (10.47%) | 299 (18.42%) | 381 (23.48%) |
| Ses03 | Train | 4241 | 333 (7.85%) | 534 (12.59%) | 1004 (23.67%) | 693 (16.34%) | 591 (13.94%) | 1086 (25.61%) |
| Ses03 | Val | 1569 | 171 (10.90%) | 305 (19.44%) | 320 (20.40%) | 240 (15.30%) | 151 (9.62%) | 382 (24.35%) |
| Ses03 | Test Ses05 | 1623 | 144 (8.87%) | 245 (15.10%) | 384 (23.66%) | 170 (10.47%) | 299 (18.42%) | 381 (23.48%) |
| Ses04 | Train | 4298 | 439 (10.21%) | 696 (16.19%) | 1066 (24.80%) | 606 (14.10%) | 504 (11.73%) | 987 (22.96%) |
| Ses04 | Val | 1512 | 65 (4.30%) | 143 (9.46%) | 258 (17.06%) | 327 (21.63%) | 238 (15.74%) | 481 (31.81%) |
| Ses04 | Test Ses05 | 1623 | 144 (8.87%) | 245 (15.10%) | 384 (23.66%) | 170 (10.47%) | 299 (18.42%) | 381 (23.48%) |

每个 split 的 dialogue 数、最短/最长/平均对话长度：

```text
Dialogue count = UNKNOWN
Minimum dialogue length = UNKNOWN
Maximum dialogue length = UNKNOWN
Mean dialogue length = UNKNOWN
Status = NOT_EVALUATED
Reason = historical split manifests/raw input bundle and local feature PKL are absent
```

Ses04 Validation 的类别分布较偏，尤其 Happy 为 4.30%、Frustrated 为
31.81%，但六类都有真值。现有证据不能把这种分布直接解释为 100% Neutral
预测的根因；没有确认数据损坏。

## 4. 标签系统

数据集没有额外 remap：PKL `videoLabels` 的整数直接成为 loss target；logit、
loss、metric 和 class weight 都按同一 index。

| Label ID | Label name | Dataset mapping | Model output | Loss target | Metric index | DialogueGCN class weight | GS-MCC class weight |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | Happy | `videoLabels=0` | 0 | 0 | 0 | 11.5278 | N/A |
| 1 | Sad | `videoLabels=1` | 1 | 1 | 1 | 6.9249 | N/A |
| 2 | Neutral | `videoLabels=2` | 2 | 2 | 2 | 4.3882 | N/A |
| 3 | Angry | `videoLabels=3` | 3 | 3 | 3 | 6.2272 | N/A |
| 4 | Excited | `videoLabels=4` | 4 | 4 | 4 | 7.8302 | N/A |
| 5 | Frustrated | `videoLabels=5` | 5 | 5 | 5 | 3.9578 | N/A |

DialogueGCN 使用内置
`(1/0.086747, 1/0.144406, 1/0.227883, 1/0.160585, 1/0.127711, 1/0.252668)`；
GS-MCC 没有配置 class weight。未发现 label/index/class-weight 顺序错位。

## 5. 特征审计

| 项目 | Text | Audio | Visual | 状态 |
|---|---:|---:|---:|---|
| 配置维度 | 768 | 1582 | 342 | CONFIRMED |
| dialogue shape | `[L,768]` | `[L,1582]` | `[L,342]` | CONFIRMED code contract |
| batch shape | `[B,T,768]` | `[B,T,1582]` | `[B,T,342]` | CONFIRMED code contract |
| finite | UNKNOWN | UNKNOWN | UNKNOWN | NOT_EVALUATED |
| 全零比例 | UNKNOWN | UNKNOWN | UNKNOWN | NOT_EVALUATED |
| mean/std/min/max | UNKNOWN | UNKNOWN | UNKNOWN | NOT_EVALUATED |
| 每 Session 方差 | UNKNOWN | UNKNOWN | UNKNOWN | NOT_EVALUATED |
| 当前资产样本数匹配 | UNKNOWN | UNKNOWN | UNKNOWN | NOT_EVALUATED |

Feature registry：

```text
registry key = clean_roberta_v1
feature_set_name = iemocap_clean_roberta_base_utterance_mean_v1
SHA256 = c604c557bc3fbb129ca02b2acd57166b669a89ef76ff0cea1e14f9cb206324bf
historical 32-run same-hash audit = CONFIRMED
current local asset = ABSENT; remote asset required
```

loader 在读取资产时逐 dialogue 检查 text/audio/visual/speaker/utterance/sentence
长度与 label 数量一致，训练入口在模型构建前核对 SHA256。历史运行未触发长度或
SHA 错误，但本轮没有资产 bytes，不能把代码门禁写成当前资产实测统计。

## 6. 图、mask 与信息边界

### DialogueGCN

- 有效节点：每 dialogue 为 `L`。
- 结构边：对每个 target 连接 `[target-10, target+10]` 范围内的有效 source，
  包含 self；实际数量为各 target 邻域长度之和。
- relation：`target speaker × source speaker × temporal direction`，2 speakers
  时共 8 类 relation。
- padding：attention mask 控制 graph、loss 和 logits；mask sum 必须等于
  `lengths`。
- speaker：dataset `M -> 0`、`F -> 1`。
- 实际 full-context 边界：不是只到 `±10`。双向 LSTM 已读取整个 dialogue，
  nodal attention 也允许每个有效 utterance 聚合整个 dialogue。

### GS-MCC Project Variant

- 有效节点：每 dialogue 为 `3L`，顺序为 audio、visual、text。
- same-modality 结构边：每个 modality 在对称 `±10` window 内连接并含 self。
- cross-modal 结构边：同一 utterance 的三模态形成有向 fully-connected block，
  每 utterance 有 6 个 cross-modal slots。
- 有效加权度必须 finite 且大于 0；否则 runtime 直接失败。
- padding node 被隔离，`node_mask` 控制 graph output。
- speaker：dataset `M -> 0`、`F -> 1`，模型 embedding 前再 clamp 到合法范围。
- 实际 full-context 边界：双向 GRU 已读取整个 dialogue；Fourier operator 又沿
  全部 graph-node axis 做 FFT，因此不是严格局部信息边界。

历史文件没有保留每 run 的对话长度、实际边张量或 speaker/mask dump，因此实际
有效节点数、有效边数、空图数和异常边数为 `UNKNOWN/NOT_EVALUATED`。代码和历史
无 runtime failure 只支持“未发现 graph/mask 问题”，不等于资产级排除。

## 7. Optimizer 与 scheduler

模型、optimizer builder、两份正式 base config 在任务开始前相对正式 commit 的
相关内容一致；本机按正式配置实例化得到：

| Model | Trainable tensors / elements | Optimizer tensors / elements | Missing | Duplicate refs | Groups |
|---|---:|---:|---:|---:|---|
| DialogueGCN | 26 / 1,298,306 | 26 / 1,298,306 | 0 | 0 | Adam, lr 3e-4, wd 0 |
| GS-MCC | 109 / 768,706 | 109 / 768,706 | 0 | 0 | AdamW, lr 1e-5, wd 1e-4 |

`CONFIRMED`：两者都只有一个 parameter group，未发现 trainable 参数漏入 optimizer
或重复加入。

两者正式 scheduler 都是 `none`，因此：

- 初始 LR 在全程保持配置值。
- scheduler 不会在有效学习前进一步降 LR。
- 公共 runtime 对其他 scheduler 的更新点是 validation 后、epoch 日志前；
  与本轮两个正式 full 配置无关。

## 8. Loss 组合

DialogueGCN：

```text
classification_loss = weighted masked cross entropy
contrastive_loss = N/A
auxiliary_loss = N/A
total_loss = classification_loss
```

GS-MCC Project Variant：

```text
classification_loss = unweighted masked cross entropy
raw_contrastive_loss = cross_frequency_contrastive_loss(...)
contrastive contribution = 1.0 * raw_contrastive_loss
auxiliary_loss beyond contrastive = N/A
total_loss = classification_loss + contrastive contribution
```

不存在的 loss 在新诊断 CSV 中写空值，不写 0。现有曲线没有显示 contrastive
contribution 压倒 classification loss；未确认 loss scaling bug，但仍需新诊断
逐 epoch 验证。

## 9. Early stopping 与 checkpoint

正式旧语义：

```text
metric = val_weighted_f1
mode = max
patience = DialogueGCN 10 / GS-MCC 20
min_delta = 0.0 (strict score > best)
min_epochs support = NO
best update = improvement 时立即更新
counter update = improvement 清零，否则 +1
stop = patience > 0 and counter >= patience
best checkpoint = improvement 时保存
test selection leakage = false
```

新增语义：

```text
early_stopping_min_epochs default = 0
epoch < min_epochs:
  best metric 仍可更新
  best checkpoint 仍可保存
  patience counter 仍按原规则更新
  不允许因 patience 停止
epoch >= min_epochs:
  按 patience 与 counter 决定是否停止
```

未配置 `early_stopping_min_epochs` 时，停止判断与旧逻辑相同。没有删除 early
stopping、没有改变 selection metric，Test 不参与任何判断。

## 10. 公共训练诊断

公共模块为 `utils/training_diagnostics.py`，两个模型通过同一
`scripts/workflows/paper_aligned/train.py` 接入，不复制 trainer。诊断默认关闭；
关闭时不创建 accumulator、不复制参数、不改变 forward/loss/optimizer/checkpoint
路径，也不额外消耗随机数。

开启后写入 run 自身 canonical 目录：

```text
<run_root>/logs/training_diagnostics.csv
<run_root>/logs/training_diagnostics_setup.json
```

CSV 字段：

```text
epoch, train_loss, val_loss, train_weighted_f1, val_weighted_f1,
learning_rate, classification_loss, contrastive_loss, auxiliary_loss,
gradient_norm, parameter_update_norm, nonzero_gradient_parameter_count,
trainable_parameter_count, logit_mean, logit_std, logit_min, logit_max,
prediction_entropy, predicted_class_count, dominant_class_ratio,
per_class_recall, effective_batch_count, early_stopping_counter, best_epoch,
best_metric, min_epochs, patience, stopped_by_early_stopping
```

统计定义：

- logit、entropy、类别覆盖和 per-class recall 来自当轮 Validation。
- gradient norm 是实际 optimizer 使用的 clipped gradient global norm 的 batch
  平均。
- nonzero gradient parameter count 是至少一个 batch 中非零 gradient element
  的最大数量。
- parameter update norm 直接比较一次 optimizer step 前后的 trainable 参数，
  不是用 gradient norm 代替；前 15 epoch 每轮采样，之后默认每 5 epoch 采样。
- 不保存 logits、gradients 或 parameter tensor dump。

## 11. 12-run 受控矩阵

| Run | Session | Candidate | 只改变 |
|---|---|---|---|
| `dialoguegcn_ses04_original_diagnostic` | Ses04 | original | diagnostics |
| `dialoguegcn_ses04_delayed_early_stop` | Ses04 | delayed stop | min_epochs 30, patience 20 |
| `dialoguegcn_ses03_original_diagnostic` | Ses03 | original control | diagnostics |
| `dialoguegcn_ses03_delayed_early_stop` | Ses03 | delayed control | min_epochs 30, patience 20 |
| `gsmcc_ses01_original_diagnostic` | Ses01 | original | diagnostics |
| `gsmcc_ses01_delayed_early_stop` | Ses01 | delayed stop | min_epochs 90, patience 40 |
| `gsmcc_ses01_lr_candidate` | Ses01 | LR | LR 1e-5 -> 3e-5 |
| `gsmcc_ses02_original_diagnostic` | Ses02 | original | diagnostics |
| `gsmcc_ses02_delayed_early_stop` | Ses02 | delayed stop | min_epochs 90, patience 40 |
| `gsmcc_ses02_lr_candidate` | Ses02 | LR | LR 1e-5 -> 3e-5 |
| `gsmcc_ses03_best_candidate_control` | Ses03 | delayed control | min_epochs 90, patience 40 |
| `gsmcc_ses04_best_candidate_control` | Ses04 | delayed control | min_epochs 90, patience 40 |

DialogueGCN 不改变 LR、batch size、hidden、modality、dropout、图、层数或 loss。
GS-MCC delayed candidate 保持原 LR；LR candidate 保持原 min_epochs=0 与
patience=20。Ses03/Ses04 采用 delayed candidate，是因为现有正式结果已证明
`1e-5` 在训练存活足够久时能够学习；提高 LR 的证据相对间接。

GS-MCC `3e-5` 不是因为任务文本出现该数值而采用。选择依据是：正式 `1e-5`
控制组跨过 WF1 0.2 需要 37/47 epochs；三倍量级是把观测转折窗口移入原 patience
范围的最小单变量诊断之一，同时仍显著低于其他 full-context baseline 的 LR。
该近似不假设学习速度与 LR 严格线性。

版本化矩阵：

```text
configs/benchmarks/repairs/dialoguegcn_gsmcc_full/repair_matrix.yaml
configs/benchmarks/repairs/dialoguegcn_gsmcc_full/repair_matrix.csv
```

## 12. Check / prepare 与输出

只读 check：

```powershell
conda run -n m3ed_mmgcn python scripts\workflows\benchmarks\prepare_full_repair.py `
  configs\benchmarks\repairs\dialoguegcn_gsmcc_full\repair_matrix.yaml `
  check --experiment-date <YYYYMMDD> --batch-id <unique_batch_id>
```

远程准备（只生成 resolved configs、matrix snapshot、CSV、commands 和 provenance，
不执行 commands）：

```bash
python scripts/workflows/benchmarks/prepare_full_repair.py \
  configs/benchmarks/repairs/dialoguegcn_gsmcc_full/repair_matrix.yaml \
  prepare --experiment-date <YYYYMMDD> --batch-id <unique_batch_id>
```

输出：

```text
outputs/<YYYYMMDD>/dialoguegcn_gsmcc_full_repair/
  runs/<run_id>/
  logs/launcher/<batch_id>/
  manifests/batches/<batch_id>/
    resolved_configs/
    commands.txt
    matrix.yaml
    matrix.csv
    git_commit.txt
    preparation_metadata.json
  review/batches/<batch_id>/
  reports/batches/<batch_id>/
  analysis/batches/<batch_id>/
```

不得写入 `outputs/long_training/` 或 `outputs/launcher_logs/`。fixed run ID 已存在时，
训练入口会在 epoch 前失败，避免覆盖。

## 13. 成功与失败判据

本阶段不判定修复成功。远程诊断后，候选只使用 Validation 和训练诊断做选择。

DialogueGCN 候选支持继续评估的必要条件：

- `predicted_class_count > 1`，dominant ratio 明显低于 1。
- logit std 脱离近零，entropy 不再长期贴近 `ln(6)`。
- Validation WF1 脱离单类基线并进入可学习区。
- gradient/update norm 非零且 finite。
- Ses03 control 的 best Validation WF1 相对原 control 不出现明显退化。

GS-MCC 候选支持继续评估的必要条件：

- Ses01/Ses02 类别覆盖、logit std、entropy 和 Validation WF1 相对 original
  diagnostic 改善。
- gradient norm 非零且稳定，直接测得的 parameter update norm 非零。
- classification 与 contrastive contribution finite，量级无失控。
- Ses03/Ses04 control 不明显退化。
- Test 仍未参与候选选择。

任一候选失败条件：

- NaN、Inf、gradient explosion、checkpoint 数值门禁失败。
- 有 gradient 但 parameter update 长期为 0，或 trainable 参数未进入 optimizer。
- 单类/近均匀状态持续到停止，Validation 不进入有效学习区。
- 正常 Session control 明显退化。
- 发生 test-selection leakage、输出碰撞、正式配置漂移或旧结果覆盖。

候选冻结后才允许对其 validation-selected `best_model.pt` 做 Ses05 Test 报告；Test
只用于最终报告，不用于选 delayed/LR 候选。

## 14. 远程执行与结果收集顺序

1. 本地完成测试、人工 diff 审查、用户确认后再 commit/push。
2. 远程确认 branch、commit、clean worktree 和 feature SHA。
3. 运行 `check`，要求 12/12 runtime validation、0 collision、0 leakage。
4. 运行 `prepare`，人工检查 manifest、resolved-config diff 和 `commands.txt`。
5. 先运行四个 original diagnostic，确认诊断字段与历史异常可复现。
6. 再运行 DialogueGCN 两个 delayed 候选。
7. 再运行 GS-MCC Ses01/Ses02 delayed 与 LR 候选。
8. 最后运行 GS-MCC Ses03/Ses04 controls。
9. 收集每 run 的 resolved config、setup JSON、diagnostic CSV、epoch metrics、
   best/last checkpoint metadata、validation/test evaluation 和 numeric status。
10. 基于 Validation 预注册判据冻结结论；保留所有旧正式结果，不覆盖、不改名。

结果收集必须明确区分 `CONFIRMED`、`HIGH_CONFIDENCE`、`HYPOTHESIS` 和
`NOT_EVALUATED`，不得把未运行候选写成修复成功。

## 15. 正式配置与模型边界

- 正式 long32 matrix 未修改。
- DialogueGCN 与 GS-MCC 正式 base configs 未修改。
- 旧 outputs 未修改。
- DialogueGCN/GS-MCC model 文件未修改，模型数学、forward contract 和
  state-dict schema 未修改。
- MMGCN、MultiDAG-CL、causal 证据缺口和论文级官方复现不在本任务范围。
