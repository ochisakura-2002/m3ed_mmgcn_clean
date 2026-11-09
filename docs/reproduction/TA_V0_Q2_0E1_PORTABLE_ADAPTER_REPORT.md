# Q2-0E.1 Portable TA Adapter Compatibility Patch

日期：2026-11-09。WORKSTREAM = MULTIDAG。

| 字段 | 结果 |
| --- | --- |
| STATUS | PASS；portable adapter compatibility patch 完成。 |
| WORKSTREAM_CHECK | PASS；MULTIDAG guard 在指定主工作区运行，分支 feat/multidag-cl-paper-reimplementation，origin fetch/push URL 均通过检查。 |
| FILES_CHANGED | 5 个 tracked 文件修改，新增 portable 测试与本报告；完整 scoped 清单见下文。 |
| PORTABLE_RUNTIME_CONTRACT | TAV0Dataset(asset, model_rows, provenance_rows, *, split, expected_rows=7433)；from_files(asset_path, model_manifest_path, provenance_manifest_path, *, split, expected_rows=7433)。正式默认固定 7433 行；小型 synthetic fixture 显式传入其行数。 |
| MODEL_MANIFEST_ROLE | model_input_manifest_v1.csv 是 current identity、split、target、TA row mapping 的 authority。 |
| PROVENANCE_MANIFEST_ROLE | iemocap_asym_manifest_v1.csv 是 canonical-order authority 与 historical provenance cross-check。两份 manifest 按 utterance_id exact join，行数、唯一 ID 与 ID set 必须一致。 |
| CANONICAL_ORDER_RULE | 仅使用 provenance.canonical_dialogue_order；每个 dialogue 内唯一、严格正整数，允许非连续；不使用 ta_row_index、dialogue_order、global_order 或 CSV 物理顺序替代。 |
| TA_ROW_MAPPING_RULE | canonical order → utterance_id → 经两份 manifest 及 asset 核对的 ta_row_index → 原始 T/A feature row。原数组、PKL、CSV 均不改写，不生成转换后的永久资产。 |
| TEST_SPLIT_ISOLATION | split 仅允许 train/dev；Test 请求在 dataset factory、manifest 访问和文件读取前失败。未构建或运行 Test loader。 |
| BACKWARD_COMPATIBILITY | 旧 synthetic bridge/global_order 契约仅由显式 LegacySyntheticTAV0Dataset 与 legacy_synthetic_package 保留；原 74 项断言行为保留。正式 portable API 不自动检测、修复或回退到旧 schema，不允许 provenance superset。 |
| TESTS_RUN | 修改前原 74 项；修改后原 74 + 新增 51 项；Train 与 Dev export 两个 synthetic CPU CLI；5 个变更 Python 文件 AST；模型、exporter、无关 dirty 文件 SHA256；scoped Git diff 检查。 |
| TEST_RESULTS | 修改前 74 passed in 3.49s；修改后 125 passed in 3.82s。两个 CLI PASS。AST 5/5 PASS。scoped git diff --check PASS。 |
| MODEL_FILES_CHANGED | NO；models/multidag_cl/ta_v0/*.py 与历史 paper_reimplementation/*.py 的文件哈希保持不变。 |
| MODEL_MATH_CHANGED | NO；graph、attention、GRU、h_ta、classifier、1500-d representation、label vocab 与 state_dict schema 均未修改。Dev exporter 文件哈希与语义契约保持不变。 |
| REAL_DATA_USED | NO；未读取或复制 Ubuntu portable package。 |
| GPU_USED | NO；现有 m3ed_mmgcn 环境，CPU synthetic 检查。未安装依赖。 |
| TEST_DATA_USED | NO；未使用真实 Test 数据。Synthetic fixture 中仅校验 Test identity/split 元数据，不获取 Test 特征或构建 Test loader。 |
| KNOWN_LIMITATIONS | 本轮仅证明 synthetic portable schema 兼容；真实 Train/Dev loader 和 single-batch forward 尚未在 Ubuntu 执行。CLI 仍保持 synthetic_only、数据路径 null，不开放真实训练或持久 checkpoint 流程。7433 行默认检查覆盖完整 package 的身份元数据，因此 Train/Dev 初始化也会校验 Test 身份/split，但不会取得 Test dataset item 或验证/使用其 feature arrays。 |
| SCOPED_GIT_STATUS | 5 个 M，2 个 ??；未 git add、commit、push、pull、stash 或修改分支/remote；原有 unrelated dirty/untracked 状态与可读文件哈希保留。 |
| NEXT_SAFE_STEP | 精确 Git 提交并完成独立代码交付后，在 Ubuntu 核对身份与状态，再 pull --ff-only → real Train/Dev loader smoke → single-batch forward。该序列属于后续任务，正式训练仍未批准。 |

**PORTABLE_ADAPTER_READY_FOR_REMOTE_SMOKE = YES**

该 verdict 仅表示 adapter 已具备进行下一轮有界远程 smoke 的条件，不表示真实资产加载、正式训练或 Test evaluation 已完成或获批。

## Scoped files

修改：

- datasets/iemocap/ta_v0_dataset.py
- scripts/models/multidag_cl/ta_v0/train.py
- configs/multidag_cl/ta_v0/iemocap/causal_context/asymmetric_merc_v1/baseline.yaml
- tests/data/test_ta_v0_dataset.py
- tests/models/multidag_cl/ta_v0/test_contracts.py

新增：

- tests/models/multidag_cl/ta_v0/test_portable_adapter.py
- docs/reproduction/TA_V0_Q2_0E1_PORTABLE_ADAPTER_REPORT.md

train.py 的修改限于数据路径字段校验、portable synthetic fixture 和显式 legacy fixture helper；runner、模型调用、loss、checkpoint selection 与 CLI 授权范围未修改。baseline.yaml 只重命名两个 null 数据路径字段。

## Portable validation

两份 manifest 对每个 utterance 核对 dialogue_id、split、speaker、benchmark_label、benchmark_label_id、ta_dialogue_id、ta_row_index；任何冲突均 ValueError，不修复。Asset 的 ID set、dialogue key、实际 row index、speaker、label ID/name 与 split membership 也必须一致。Train/Dev 的 T/A shape、float32 与 finite 检查保留。

新增 51 项测试覆盖要求 A–L：exact join、manifest 独立打乱、不连续 canonical order、原始 TA 行映射、stored A order 与 canonical order 不同、151 行 dialogue split bridge 缺省或存在但无须读取、ID set 冲突、双方各字段冲突、双份 manifest 一致但与 asset 冲突、重复 canonical order、非法 order、无 global_order、7433 行默认检查、小型 fixture 显式计数与 Test 读取隔离。

完整 7433 行 synthetic case 仅扩展身份元数据并构建 Train/Dev dataset，不加载 Test 特征；它不模拟真实 split 数量，也不替代已在 Ubuntu 完成的真实资产审计。

## Verification commands

使用现有 m3ed_mmgcn 环境；Python bytecode 和 pytest cache 关闭，测试/CLI 产物仅位于本轮新建临时目录。Windows sandbox 下 pytest 临时目录 ACL 检查失败后，以获准的沙箱外执行完成基线及最终验证；该环境问题未改动源码或依赖。

```text
python -B -m pytest -q tests/data/test_ta_v0_dataset.py tests/models/multidag_cl/ta_v0 tests/paper_reimplementation/multidag_cl/test_encoders.py tests/paper_reimplementation/multidag_cl/test_graph.py tests/paper_reimplementation/multidag_cl/test_attention_and_gru.py tests/paper_reimplementation/multidag_cl/test_model.py -p no:cacheprovider --basetemp <fresh-temporary-directory>
125 passed in 3.82s

python -B scripts/models/multidag_cl/ta_v0/train.py --synthetic-dry-run
status=PASS; device=cpu; optimizer_steps=0; training_started=false
test_evaluation_count=0; h_ta_shape=[2,3,1500]

python -B scripts/models/multidag_cl/ta_v0/export_dev.py --synthetic-dry-run --output-dir <fresh-temporary-export-directory>
rows=4; unique_utterance_ids=4; duplicate_ids=0; padding_rows=0
dimension=1500; representation_name=ta_v0_contextual_preclassifier
prediction_representation_id_set_identical=true
```

历史 encoder、graph、attention/GRU、model 共 31 项回归继续通过；相同权重下 TA text/audio encoder 与历史实现逐元素等价的检查继续通过。原有 causal/prefix/future-perturbation、Dev exporter ID/label/logit/probability/h_ta checks 全部通过。

## Scoped git diff --stat

```text
 .../asymmetric_merc_v1/baseline.yaml             |   4 +-
 datasets/iemocap/ta_v0_dataset.py                | 144 +++++++++++++++------
 scripts/models/multidag_cl/ta_v0/train.py        |  33 +++--
 tests/data/test_ta_v0_dataset.py                 |   4 +-
 tests/models/multidag_cl/ta_v0/test_contracts.py |   3 +-
 5 files changed, 135 insertions(+), 53 deletions(-)
```

普通 diff --stat 不包含两个未暂存新文件；本轮未用 git add -N 改动 index。工作区中已有的 .gitignore 修改与历史文档删除不属于本 patch。

## Scoped git status

```text
 M configs/multidag_cl/ta_v0/iemocap/causal_context/asymmetric_merc_v1/baseline.yaml
 M datasets/iemocap/ta_v0_dataset.py
 M scripts/models/multidag_cl/ta_v0/train.py
 M tests/data/test_ta_v0_dataset.py
 M tests/models/multidag_cl/ta_v0/test_contracts.py
?? tests/models/multidag_cl/ta_v0/test_portable_adapter.py
?? docs/reproduction/TA_V0_Q2_0E1_PORTABLE_ADAPTER_REPORT.md
```
