# CT-MultiDAG Pairwise Fast-Kill 实现报告

WORKSTREAM = MULTIDAG · 2026-11-24 · seed100

| Windows 交接项 | 状态 |
| --- | --- |
| PAIRWISE_HEAD_IMPLEMENTED | YES |
| PAIRWISE_HEAD_PARAMETERS | 9856 |
| SEED100_FASTKILL_IMPLEMENTED | YES |
| F0_CORE_MODIFIED | NO |
| FORMAL_TRAIN_RUNS | 0 |
| TEST_TOUCHED | NO |

## 冻结推理与训练

外置 head 只使用冻结 F0 的 `h0/h4`。两个 300 维状态分别做无仿射 LayerNorm，拼接后经过 Xavier uniform 初始化的 600→16 投影。15 行无序类对向量以零初始化，对调候选时修正反号。有限精度的 tanh 饱和截到最靠近 ±1 的区间内浮点值。写回锚定原第二名，仅改前二，遇到零修正、零间隔或写回浮点平局时逐字节透传。关闭适配器时返回原 logits。正式 YAML 只提供启用开关，已有 F0 入口直接使用原模型。

入口复用严格的历史 F0 校验。CTRL 固定为 seed100 第 21 轮，归档 Dev 为 693/980 正确、W-F1 69.9627%。历史 checkpoint SHA-256 为 `997e7db663e3a3c2b8cb9ea34957e8fdbda909593ae7ad8360bc14ef1e3022ac`。认证 Train/Dev 投影、历史数据身份、Dev batch tape、原 logits 和预测逐项对齐。F0 冻结、eval、禁用梯度，每个 split 走一遍历史 batch size 16 的前向。CPU 缓存有效行状态、logits、标签及身份；Train 优化批次和 epoch 20 推理在所选设备执行。

门沿用 Preflight 的 FP64 稳定排序和 MAD。阈值重新从同一冻结 Train logits 按原顺序统计量计算，随后固定。进入优化前复核冻结的 539/159 Train/Dev 门控条数、287 个 Dev 错误、门内真类前二 140 条、52 个理论可恢复错误和 +5.548124 pp 的 W-F1 oracle 增益。Train 只优化门内且真类位于前二的行，统计 rank≥3 样本。Adam 使用 lr 0.003、betas (0.9, 0.999)、weight decay 0、batch 128、seed100 固定 shuffle。唯一损失是 `mean(softplus(-t*(m+δ)))`。固定训练 20 轮，只有 epoch 20 进行 PRIMARY Dev 评分。

## 有效性与产物

正式入口对加载的 F0 执行五次独立合成前向，复核截断前缀、扰动未来、对话换序与额外 padding。采用已门控行的 logits 作为额外合成检查输入，确保确实执行 head。该探针与完整 Dev 单遍缓存分开。运行比较浮点结果使用 `rtol=1e-5, atol=1e-6`，候选 ID、gate、写回标志和预测采用精确比较。

冻住的 Round 1 SHIFT 和 MAD Q4 trace 必须认证。U/H 键、归档预测或等预算名单对齐失败时，U/H 表标记 `NOT_AVAILABLE`。输入 HOLD 失败会在拒绝证据中记录来源，当前执行的五项 Test 操作计数均为 0。冻结 Q4 是归档的 245 条 Dev 子集，门控仍使用 Train 阈值。

判定按 REVIEW_INVALID、STOP、GO、REVIEW_SIGNAL 执行。主指标输出百分数及百分点差，类对文件输出全部 30 个有向计数、真类分母、误率和成对净纠错。校准副指标、SHIFT/U/H 分层和 2,000 次对话配对 bootstrap CI 用于解释。实际恢复率使用全 Dev 的 R / 52。理论机会分布沿用 Preflight，ang/fru 26、neu/fru 11、neu/exc 6、hap/exc 4、hap/sad 3、sad/fru 2。

运行清单绑定 checkpoint SHA-256、投影证书及文件 SHA-256、Train τ、类序和缓存形状、dtype、人口标识。训练时间指 head 优化阶段，推理开销指缓存张量进入 label-free gate 到最终 logits 的一次计时，包含 gate 和设备同步。峰值显存采用 PyTorch 的 allocated bytes。正式目录保留独立的 epoch 20 head，回传 ZIP 只含列举的 CSV、JSON 和 Markdown。

```text
FORMAL_TRAIN_RUNS = 0
TEST_TOUCHED = NO
```

[Windows 验收](PAIRWISE_FASTKILL_TEST_REPORT.md) · [Ubuntu Runbook](PAIRWISE_FASTKILL_RUNBOOK.md)
