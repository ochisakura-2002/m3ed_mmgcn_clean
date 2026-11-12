# Q2-2A frozen Dev weighted log-probability fusion

本 runtime 属于 `ASYMMETRIC_MERC`，只读取显式指定的七份冻结 Dev prediction 文件。
它重算 Q2-1 complementarity、选择一次 alpha、固定 alpha 评估五个 Linear-B seed，
并输出审计、表格与报告。它不加载模型、checkpoint、特征、训练或评估 dataloader。
本次没有修改 TA/V 模型数学结构、forward、训练或 state-dict schema。
这是离线分析工具，不新增模型模块，因此没有模型 YAML 开关；alpha 的 1/0 端点分别等价于 TA/V prediction。

## 输入与执行

在仓库根目录执行。相对输入路径与 `--output-dir` 都相对于调用者的当前目录，
没有自动发现 prediction、checkpoint 或其他 split 的逻辑。

每个文件必须含 `utterance_id`、整数 `true_label`、`prob_0` 到 `prob_5`，
以及 `predicted_label` 或 `pred_label`。若两个预测字段都存在，必须一致。
CSV/TSV 根据首行分隔符识别，接受 UTF-8 BOM。
标签顺序固定为 `0 hap, 1 sad, 2 neu, 3 ang, 4 exc, 5 fru`，
与 `configs/_shared/data/iemocap/benchmark_label_vocab_v1.yaml` 一致。
旧 S2D exporter 的 `label_index` / `prob_hap` 等 schema 不在此输入合同中；
需要提供本任务指定的冻结数字标签 prediction，禁止猜测或静默重映射标签。

下列路径是待替换的相对路径示例，不表示本地已有真实 prediction：

```powershell
python -B scripts/analysis/models/asymmetric_merc/run_q2_2_weighted_fusion.py `
  --ta-predictions frozen_dev/ta_v0_seed100.tsv `
  --linear-b-seed100 frozen_dev/linear_b_seed100.csv `
  --linear-b-seed101 frozen_dev/linear_b_seed101.csv `
  --linear-b-seed102 frozen_dev/linear_b_seed102.csv `
  --linear-b-seed103 frozen_dev/linear_b_seed103.csv `
  --linear-b-seed104 frozen_dev/linear_b_seed104.csv `
  --original-s2d frozen_dev/original_s2d_epoch7.csv `
  --output-dir outputs/<YYYYMMDD>/q2_2_weighted_fusion/analysis/batches/<batch_id>
```

将 `<YYYYMMDD>` 替换为启动日，将 `<batch_id>` 替换为本次唯一标识。
`outputs/` 内的路径用公共 output resolver 校验日期、group 与 batch 层级。
输出目录必须尚不存在；重跑必须使用新目录，禁止合并或覆盖旧结果。
可显式指定其他 output base；本地所有 synthetic 产物仅写入 `tmp/assistant_work/`。
不需要 GPU，不需要重新训练或访问任何真实 Test 资产。

## 固定计算协议

严格校验非空且唯一的 ID；数字标签必须在 0..5；概率必须 finite、非负，
行和满足 `abs(sum - 1) <= 1e-5`（`rtol=0`）。不重归一化输入。
`argmax(probabilities)` 必须与存储预测一致，argmax 平局按固定类别顺序取首类。
若有 `split` 列，只接受 `dev`、`val`、`validation`；没有该列时记录
`caller_declared_dev`，不能仅靠 CSV 内容验证其 split 或模型/seed/epoch 身份。

TA/V 连接仅使用 `utterance_id` 的 exact inner join，按 ID 排序以保证输入乱序不改变结果。
输出 TA/V/common 行数、TA-only/V-only ID 列表、重复 ID、标签不一致 ID。
重复 ID 或 common label mismatch 导致失败，不进入对应的分析和 fusion。

四象限、rescue、disagreement、oracle 和六类细分均由原始 prediction 重算。
四项指标使用 `utils.metrics`，固定全部六类；precision/recall/F1 的 zero-division 取 0。
没有 TA error 时，rescue rate 为 JSON null / CSV 空值，不捏造为 0。
每个视觉 seed 的差值使用该 seed common subset 上的 TA 指标作为基准。

```text
score_k = alpha * log(max(P_TA(k), 1e-12))
        + (1-alpha) * log(max(P_V(k), 1e-12))
prediction = argmax_k(score_k)
```

clipping count 是严格小于 `1e-12` 的概率元素个数；manifest 记录整份输入计数，
join audit 记录实际参与每个 TA/V pair 的计数。未增加 calibration 或可学习参数。
输出 `score_0..score_5` 是未归一化 log score，不声称为 probability。

alpha grid 固定 `0.00..1.00`、步长 `0.01`、共 101 点。
只在 Linear-B seed100 common Dev 上最大化 Weighted-F1；
与全局最大值差不超过 `1e-12` 的候选中取最大 alpha。
随后固定 alpha 评估 seed100--104，不对其他 seed 寻优。
Original S2D 只输出 complementarity 和 class-wise rescue，不参与选择或 fusion。

## 回归与失败处理

用户提供的参考值单独存放在
`configs/asymmetric_merc/analysis/q2_2_dev_regression.yaml`，不是算法参数或结果替代品。
正式模式重算后检查 primary join、四象限、class-wise rescue、TA full/common 指标、
alpha、fusion 指标和差值；近似指标绝对容差为 `5e-6`，整数计数要求完全一致。
alpha grid 的离散间隔保证该容差只接受 `0.88` 这个已有参考 grid 点。
primary 不符时保存已算出的 alpha grid、selection、complementarity 和 provenance，
返回退出码 1，停止后续 robustness/fusion，不修改协议。
五个固定 alpha 结果还检查正向 Weighted-F1 delta 的 seed 数；不符同样返回失败，保留观察值。

输入错误也返回退出码 1，并保留 `run_status.json`、能取得的 input/join audit 和失败报告。
已有输出目录或无效 output layout 在分配前拒绝，不修改该目录。
只有成功完成所有输入与回归检查才返回退出码 0。
`--synthetic` 仅用于小型人工 fixtures，明确标记 `SYNTHETIC ONLY`，
不应用真实 Dev regression，不输出科研正向信号结论。
正式远程 Dev 运行不要传入 `--synthetic`。

## 产物与 provenance

```text
<output_dir>/
  provenance/
    input_manifest.json           # 七份路径、SHA256、输入审计、Git、版本、源码 hash、时间
    protocol.json                 # 固定公式、标签、grid、选择规则、隔离计数
    exact_join_audit.json          # 每个 Linear-B / Original S2D 与 TA 的连接审计及 clipping
    regression_audit.json          # expected/actual/pass，或 synthetic 不适用
    run_status.json                # PASS/FAIL、异常、实际结果摘要
  complementarity/
    primary_seed100_summary.csv
    primary_seed100_classwise.csv
    linear_b_seed_robustness.csv
    original_s2d_reference.csv
    original_s2d_reference_classwise.csv
  alpha_selection/
    alpha_grid_seed100.csv
    alpha_selection.json
  fusion/
    seed100_predictions.csv        # ID/true/TA/V/fusion prediction、输入概率与六类 scores
    seed100_metrics.json           # common TA/V/fusion/差值，另记 full TA
    seed100_per_class.csv          # 全部六类，含 hap=0 F1 情况
    seed100_confusion_matrix.csv   # source=ta/v/fusion，各六行，行=true、列=pred
  robustness/
    fixed_alpha_seed100_104.csv    # 每个 seed 四项指标、匹配 TA/V 基准与 delta
    fixed_alpha_per_class.csv     # 五个 seed × 六类
  reports/
    Q2_2_WEIGHTED_FUSION_REPORT.md
```

`test_access=false`、`test_loader_built=0`、`test_evaluation_count=0` 明确记录于 provenance。
CLI 没有 Test、feature、checkpoint、training 输入；运行时不 import 模型和 dataset 栈。
SHA256 使用 `utils.run_metadata.compute_file_sha256`，读取前后检查输入没有变化。
Git HEAD、完整 dirty/untracked 状态、UTC timestamp、Python/NumPy/sklearn/PyYAML 版本、
runtime/shared helper source hash 和 regression YAML hash 共同保留复现依据。

报告只依据实际计算值给出 FACT。正式运行通过且 primary delta 为正、五个 V seed
方向均为正时，才标记 `Q2_2_DEV_SIGNAL = POSITIVE`。
报告明确限制：alpha 在 Dev 选择，当前仍是 development evidence；只改变 V seed，TA 固定。
本地 synthetic 成功不能证明真实 Dev 回归通过。

## 本地验证

先将测试进程的系统临时路径限定在任务允许的目录中：

```powershell
$env:TEMP = Join-Path (Get-Location) 'tmp/assistant_work/system_temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
python -B -m pytest -q -p no:cacheprovider tests/models/asymmetric_merc/test_q2_2_weighted_fusion.py --basetemp=tmp/assistant_work/pytest_q2_2
```

新增测试使用 `tmp/assistant_work/q2_2_tests/<unique_id>/` 普通目录，
避免 managed Windows ACL 下 pytest `mode=0700` 临时目录不可访问的问题。
覆盖 exact join、输入乱序、各种非法输入、四象限/类别 rescue、log formula、零概率 clipping、
两个端点、确定性/全局 tie-break、仅一次 selection、五 seed 固定 alpha、
schema/provenance、正式模式 mismatch stop、collision 保护和 CLI subprocess/import isolation。
已有共享 helper 与 S2D metric/export 测试也需定向回归。

## 本次工程验收记录

2026-11-11：`WORKSTREAM_CHECK=PASS`，初始工作区干净，分支为
`integration/asymmetric-merc-v0`。未执行 add、commit、push、pull、stash 或分支切换。
未读取真实 Dev prediction，未访问真实 Test，未训练任何模型。

实际阅读的既有源码（部分文件仅阅读相关函数/片段）：

- `utils/metrics.py`：复用四项指标与固定标签混淆矩阵。
- `utils/run_metadata.py`：复用流式 SHA256。
- `utils/evaluation.py`：核对已有 prediction schema；其字段与本任务合同不同，未直接使用。
- `utils/output_paths.py`：复用 canonical output layout resolver。
- `scripts/analysis/common/build_analysis_tables.py`：检查已有 loader/report 风格；容错 loader 不适用于严格输入合同。
- `scripts/models/s2d/iemocap/train.py`：核对已有 metric/export；未 import 到新 runtime。
- `tests/test_run_metadata.py`、`tests/test_output_paths.py`：共享工具回归与库存门禁。
- `tests/models/s2d/test_s2d_v0.py`：现有 metrics/export 回归及普通 workspace temp fixture 方式。
- `tests/test_legacy_wrapper_retirement.py`：核对 canonical-only 门禁约束。
- 另读取 `configs/_shared/data/iemocap/benchmark_label_vocab_v1.yaml` 与项目要求的工作流、实验及输出协议文档。

文件变更清单：

| 类型 | 文件 | 作用 |
| --- | --- | --- |
| 新增 | `scripts/analysis/models/asymmetric_merc/__init__.py` | 分析 namespace |
| 新增 | `scripts/analysis/models/asymmetric_merc/weighted_fusion.py` | 输入审计、exact join、互补性、指标、alpha selection、固定 alpha 评估 |
| 新增 | `scripts/analysis/models/asymmetric_merc/run_q2_2_weighted_fusion.py` | 七文件 CLI、provenance、回归门禁、输出编排与退出码 |
| 新增 | `scripts/analysis/models/asymmetric_merc/fusion_report.py` | CSV/JSON 序列化与有证据边界的报告 |
| 新增 | `configs/asymmetric_merc/analysis/q2_2_dev_regression.yaml` | 用户冻结参考值，仅作比较 |
| 新增 | `tests/models/asymmetric_merc/test_q2_2_weighted_fusion.py` | 37 项 synthetic 检查 |
| 新增 | 本文档 | 输入/协议/CLI/产物/验收交接 |
| 修改 | `tests/test_output_paths.py` | 库存断言更新为 228 YAML / 158 output owners；保留逐配置布局验证 |

库存旧断言为 210/142，而 `git ls-files` 对应的任务前库存已经是 227/158；
本次只增加一份非 output-owning 的 regression YAML，因此最终为 228/158。
其他既有生产源码、配置、模型与数据资产均未修改。

验证环境为已有 `m3ed_mmgcn`：Python、NumPy 2.0.1、sklearn 1.6.1、pytest 8.4.2、torch 2.6.0+cpu。
未安装依赖。默认 Python 缺少 torch，且既有 pytest temp fixture 在 managed Windows ACL 下不可访问；
本地仅在 `tmp/assistant_work/run_q2_2_related_checks.py` 中提供普通目录 `tmp_path` fixture，
不改变被测源码与断言。该 local-only harness 调用以下相同测试集合：

```powershell
conda run -n m3ed_mmgcn python -B tmp/assistant_work/run_q2_2_related_checks.py
```

运行结果：**68 passed**（新增 37，既有 31）。
普通环境可直接运行同一集合：

```powershell
conda run -n m3ed_mmgcn python -B -m pytest -q -p no:cacheprovider `
  tests/models/asymmetric_merc/test_q2_2_weighted_fusion.py `
  tests/test_run_metadata.py tests/test_output_paths.py `
  tests/models/s2d/test_s2d_v0.py::test_metrics_and_identity_export `
  --basetemp=tmp/assistant_work/pytest_q2_2_related
```

独立 CLI smoke：

```powershell
conda run -n m3ed_mmgcn python -B tmp/assistant_work/run_q2_2_synthetic_smoke.py
```

该 local-only harness 使用已纳入源码的 synthetic fixtures，启动真正的 CLI subprocess。
结果 `Q2_2_RUNTIME=PASS`；TA 9 行、V 8 行、common 8 行，四象限 `2/1/3/2`，
rescue `0.6`、oracle accuracy `0.75`、alpha `0.49`，五个 seed 共用一次选出的 alpha。
这里的 alpha 仅属于人工 fixtures，不是对真实 Dev `0.88` 的复现。
五份新 Python 文件 AST parse 通过，Git whitespace 检查通过。

本次独立 smoke 报告位于：
`tmp/assistant_work/q2_2a_smoke/b508a28b75e947d392b5964f3888c80b/analysis/reports/Q2_2_WEIGHTED_FUSION_REPORT.md`。
真实运行报告始终位于 `<output_dir>/reports/Q2_2_WEIGHTED_FUSION_REPORT.md`。

`Q2_2A_REPRODUCIBLE_RUNTIME_READY = YES`。
`READY_FOR_REMOTE_REAL_DEV_RUN = YES`（工程准备就绪，人工审核后进行独立远程任务）。
真实 Dev 回归仍待七份冻结文件验证；本次没有据此声明真实 Dev 正向信号或模型效果结论。
