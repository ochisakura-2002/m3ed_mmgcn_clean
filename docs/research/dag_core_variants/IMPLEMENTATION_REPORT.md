# GraphCore Variants 实现与验证报告

WORKSTREAM = MULTIDAG。日期：2026-11-18。
工作区 guard 在开始和结束时均通过；branch 为
`feat/multidag-cl-paper-reimplementation`，HEAD 为
`daf0957a757044a4c00cb0827ce847add7caf1d1`。未切分支、stash、暂存、commit、push、pull。

## 交付结果

三个独立 GraphCore 已实现。编码器、融合、h0=300、1500D representation、
六分类器均由原 TAV0 初始化并复用；formal config 的模型、数据、训练、
checkpoint 条件与 T1024 baseline 等价。没有组合模型、Dune 训练模型或新依赖。
实际 IEMOCAP Train/Dev/Test 资产、正式 checkpoint 和远程工作区均未读取。
本轮只执行合成数据测试，没有正式训练或效果评估。

| 模型 | 核心 | 参数总量 | 相对 T1024 baseline |
| --- | --- | ---: | ---: |
| 冻结 T1024 baseline | 原 MultiDAG | 6,001,110 | 0 |
| MultiDAG-Track | 两条持续传播的 speaker-origin track | 1,664,306 | -4,336,804 |
| MultiDAG-Coop | 四策略概率的 sender/receiver 软控制 | 6,005,926 | +4,816 |
| MultiDAG-DCN | 每个已观察前缀上的精确 PDCN causal shifts | 1,304,306 | -4,696,804 |

参数差异全部在 GraphCore 内。参数量不相等，不能把后续结果解释为等参数量比较。
当前没有证据说明三个改动能提升 Dev 指标。

## 机制与架构文档

1. [PAPER_TO_MULTIDAG_MAPPING.md](PAPER_TO_MULTIDAG_MAPPING.md)：四篇源文件哈希、
   公式定位、SOURCE_MECHANISM 与 OUR_ADAPTATION 的区别、四阶段 adapter 和风险。
2. [GRAPHCORE_ARCHITECTURE_DECISION.md](GRAPHCORE_ARCHITECTURE_DECISION.md)：
   冻结边界、统一接口、复用策略、checkpoint identity 和输出架构。
3. [DUNE_COMPATIBILITY_ANALYSIS.md](DUNE_COMPATIBILITY_ANALYSIS.md)：
   六项兼容性问题、定理假设、诊断前置要求；本轮没有 Dune 实现。

这些文档先于模型源码写入。论文正文与方法公式已读取，必要附录已检查，
四篇方法页使用已安装的 PDF runtime 渲染并目视核对；没有仅凭摘要实现。

## 源码清单

公共路径 `models/multidag_cl/graph_core_variants/`：

- `common.py`：GraphCore 四阶段 contract、mask/关系/strict DAG validation。
- `config.py`：独立 identity、冻结 T1024 配置及少量 core 参数。
- `model.py`：原 TAV0 的继承适配，复用原编码器、融合、分类器；OFF 直接调用原 forward。
- `track.py`：MultiDAG-Track。
- `coop.py`：MultiDAG-Coop。
- `dcn.py`：MultiDAG-DCN，实际机制明确为 prefix PDCN。
- `__init__.py`：公共导出。

Track 的轨道不是情感类别，不迁移原论文的 label/pseudo-label prototype 流程。
Coop 是确定性 soft adaptation，不是离散 CO-GNN 的精确期望，也不迁移其随机
表达能力结论。DCN 采用共享参数 PDCN，保留 `W D_k W^-1`，不是邻接矩阵幂
网络；每个 receiver 只汇总已观察前缀中的 branch。原 DAG 的节点与边集保持不变，
传播算子使用的 closure/self terms 不写回 adjacency。

## 三份独立 formal config

- `configs/multidag_cl/track_t1024_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev.yaml`
- `configs/multidag_cl/coop_t1024_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev.yaml`
- `configs/multidag_cl/dcn_t1024_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev.yaml`

与父配置相比，仅新增 identity/core 配置及独立 output group；测试直接比较
model/dataset/runtime/training/checkpoint 全部字段。正式配置保留 seed100、
30 epochs、batch16、AdamW lr5e-4、eps1e-6、weight_decay0、clip5、
无 class weights/label smoothing/scheduler/early stopping/curriculum。

`graph_core.enabled: false` 可通过 synthetic CLI 验证原 baseline 等价路径；
formal variant 禁止 OFF，避免以 variant 名义记录 baseline 结果。

## Runtime 与训练入口

- `scripts/models/multidag_cl/graph_core_variants/train.py`：共享 CLI，config 选择一个模型。
- `scripts/runtime/multidag_cl_graph_core_variants/training.py`：仅扩展 checkpoint identity，
  继承原 TrainDevRuntime 的 loader、optimizer、loss、梯度裁剪、selector、保存和 resume。
- `scripts/runtime/multidag_cl_graph_core_variants/formal.py`：冻结配置验证、原 Train/Dev
  数据适配器、原 fit/Dev export 的调用、独立 manifest、canonical 输出和 synthetic fixture。
- `scripts/runtime/multidag_cl_graph_core_variants/__init__.py`：package。

三个 standalone CLI 的 synthetic dry run 均已实际执行并 PASS：
`h_ta_shape=[2,5,1500]`，`optimizer_steps=0`，Test 两个计数均为 0。

示例（只使用合成数据，无输出目录写入）：

```powershell
python scripts/models/multidag_cl/graph_core_variants/train.py --config configs/multidag_cl/track_t1024_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev.yaml --synthetic-dry-run
```

未来 formal 调用需要显式 `--formal-train-dev` 及三个数据路径；该模式本轮仅在
注入合成数据的测试中验证，没有打开真实资产。新输出固定为
`outputs/<DATE>/multidag_<variant>_t1024_formal/runs/<run_id>`，CLI 输出根覆盖仍保留
date/group/runs 层级。run_id 含模型、seed100、带时区时间戳、config/data hash。
目录碰撞在 loader/optimizer step 前失败，原 fit 的原子 mkdir 再次保护竞态。

resolved config 与 run manifest 均记录 GraphCore 默认值。Manifest 包含全部
请求字段，包括 parameter_count、git identity、data_identity、training_config、
parent_baseline、source_paper，以及 `graph_topology_changed=false`、
`future_edges_added=false`、`test_loader_built=0`、`test_evaluation_count=0`。
结构化 checkpoint 使用新 schema，并在复制权重前拒绝跨 variant/父 baseline identity。

## 测试清单与实际结果

- `tests/models/multidag_cl/graph_core_variants/conftest.py`：CPU 单线程 fixture。
- `test_cores.py`：四阶段形状、变长、padding、speaker relabel、非法边/关系、
  确定性、forward/backward/finite gradients、future-feature Jacobian、
  u4/u5 扰动不影响 u1/u2/u3、prefix extension 与未来结构扰动；
  Track 持续消息公式、Coop singleton attenuation、PDCN 每个前缀的独立矩阵公式对照。
- `test_model.py`：OFF 的 state_dict/representation/logits/dropout 逐位等价，
  非图组件初始化与 h0 完全相同、端到端 causal、边界验证、checkpoint identity。
- `test_runtime.py`：完整合成 fit、Dev export、manifest、checkpoint/reload/resume、
  正式配置突变拒绝、No-Test factory guard、YAML OFF、CLI、默认值落盘与输出碰撞。

实际执行：

```text
python -X utf8 -m pytest -q tests/models/multidag_cl/graph_core_variants tests/models/multidag_cl/ta_v0 -p no:cacheprovider --basetemp tmp/gc_20261118_b -x
378 passed in 24.44s
```

收集数核对：新 variant suite **173** 项；原 ta_v0 suite **205** 项；合计 **378**。
旧测试没有修改。三个 core 的 future perturbation、prefix equivalence 与 future
Jacobian guard 全部通过。新 Python 文件 py_compile 全部通过。

首轮沙箱运行因 pytest 临时目录 ACL 导致 setup/cleanup Access Denied，无法完成；
以正常文件权限重跑同一测试集后得到上述结果。没有通过放宽断言或修改旧测试消除问题。
未执行整个仓库 pytest；本次门禁覆盖新 suite 与用户要求的完整 ta_v0 回归。

## 保护边界与剩余限制

以下原路径的 `git diff --name-only` 为空：

```text
models/multidag_cl/ta_v0/
scripts/models/multidag_cl/ta_v0/
scripts/runtime/multidag_cl_ta_v0/
configs/multidag_cl/ta_v0/
tests/models/multidag_cl/ta_v0/
```

整个 tracked models/scripts/configs/tests 的 diff 也为空。本任务仅新增文件。
开始时的 `.gitignore` 修改、134 份文档删除和其他 untracked 文件全部保留。
入口规范中已删除的文档只读查看 HEAD，没有恢复。

Track v0 明确限制最多两个 speaker group；PDCN 的 branch sum 可能随长度放大，
且计算/存储高于普通四层 pointwise MLP。当前只证明工程和 causal contract，
没有证明优化稳定性或 Dev 改善。Dune 是否值得后续研究，需要 Train/Dev 上独立诊断。
后续组合模型和超参数 sweep 均不属于本轮交付。

## 最终状态

```text
MULTIDAG_TRACK_IMPLEMENTATION = PASS
MULTIDAG_COOP_IMPLEMENTATION = PASS
MULTIDAG_DCN_IMPLEMENTATION = PASS
DUNE_ANALYSIS = COMPLETE
BASELINE_SOURCE_MODIFIED = NO
BASELINE_TESTS = PASS (205)
CAUSAL_TEST_TRACK = PASS
CAUSAL_TEST_COOP = PASS
CAUSAL_TEST_DCN = PASS
TEST_ACCESSED = NO
READY_FOR_GPT_REVIEW = YES
```

以下 Git 输出及本任务 exact-file list 由交付时的实际命令附录补全。

## Git 实际输出与精确新增清单

`git diff --check`：exit 0，无 whitespace error；另对全部 22 个新增文件执行 `git diff --no-index --check -- /dev/null <file>`，通过。

所有本任务文件仍为 untracked，故普通 `git diff --stat` 不统计新增文件；下列 stat 是任务开始前已有的 tracked 改动，不能归因于本任务。

### git diff --stat

```text
 .gitignore                                         |   28 +
 docs/20260526_mmgcn_baseline_progress.md           |   88 -
 docs/PROJECT_HANDOFF.md                            |  617 --
 docs/active_modalities_design.md                   |  679 --
 docs/baselines/baseline_rescreen_report.md         |  203 -
 docs/baselines/causal_baseline_readiness_report.md |  128 -
 .../causal_benchmark_review_tool_notes.md          |   75 -
 docs/baselines/causal_dialoguegcn_port_notes.md    |  115 -
 docs/baselines/causal_gsmcc_port_notes.md          |  132 -
 docs/baselines/causal_model_contract.md            |   95 -
 docs/baselines/causal_repairs_changelog.md         |   26 -
 docs/baselines/configs_cleanup_audit.md            |  459 --
 docs/baselines/current_merc_model_inventory.md     |   58 -
 .../baselines/dialoguegcn_official_source_audit.md |   78 -
 docs/baselines/gsmcc_official_source_audit.md      |   95 -
 docs/baselines/iemocap_split_protocol_audit.md     |  133 -
 docs/baselines/loss_stability_training_audit.md    |   97 -
 docs/baselines/mm_dfn_smoke_review.md              |  343 -
 docs/baselines/mmgcn_causal_audit.md               |  116 -
 docs/baselines/multidag_causal_audit.md            |   88 -
 .../multidag_cl_baseline_stabilization_notes.md    |  169 -
 docs/baselines/multidag_cl_encoder_audit_notes.md  |  230 -
 .../multidag_cl_encoder_diff_audit_notes.md        |  145 -
 docs/baselines/multidag_cl_formal_run_plan.md      |  329 -
 ...ultidag_cl_formal_yaml_standardization_notes.md |  367 -
 .../multidag_cl_iemocap_experiment_matrix_notes.md |  230 -
 docs/baselines/multidag_cl_loss_stability_notes.md |  120 -
 ...dag_cl_multirun_stabilization_analysis_notes.md |  193 -
 .../multidag_cl_observability_and_figures_notes.md |  310 -
 .../baselines/multidag_cl_paper_artifacts_notes.md |  151 -
 .../multidag_cl_pipeline_integration_notes.md      |  209 -
 docs/baselines/multidag_cl_port_notes.md           |  152 -
 docs/baselines/multidag_cl_real_batch_notes.md     |  100 -
 .../multidag_cl_real_data_path_protocol.md         |  190 -
 docs/baselines/multidag_cl_smoke_review.md         |  412 -
 docs/baselines/multidag_cl_smoke_training_notes.md |   86 -
 docs/baselines/multidag_cl_yaml_parameter_guide.md |  194 -
 .../new_causal_baselines_remote_protocol.md        |  136 -
 .../baselines/new_causal_baselines_task1_report.md |  117 -
 .../baselines/new_causal_baselines_task2_report.md |   95 -
 docs/baselines/original_repro/README.md            |   84 -
 .../dialoguegcn_paper_code_mapping.md              |   22 -
 .../gsmcc_numerical_stability_report.md            |   99 -
 .../original_repro/gsmcc_paper_code_mapping.md     |   20 -
 .../original_repro/mmgcn_paper_code_mapping.md     |   15 -
 .../multidag_cl_paper_code_mapping.md              |   15 -
 .../original_merc_reproduction_handoff.md          |   89 -
 docs/baselines/original_repro/paper_targets.csv    |    4 -
 .../original_repro/reported_results_reference.csv  |    5 -
 docs/baselines/original_repro/source_audit.md      |   62 -
 docs/baselines/sdt_isolated_smoke_review.md        |  327 -
 docs/baselines/sdt_port_notes.md                   |  122 -
 docs/codex_workflow.md                             |  170 -
 docs/data/iemocap_clean_roberta_v1.md              |   80 -
 docs/experiment_protocol.md                        |  249 -
 .../DIALOGUEGCN_GSMCC_FULL_REPAIR_PREPARATION.md   |  434 --
 docs/experiments/FORMAL_FULL16_REPAIRED.md         |  104 -
 docs/experiments/LONG_TRAINING_COMMANDS.sh         |   31 -
 docs/experiments/LONG_TRAINING_MATRIX.csv          |   33 -
 .../experiments/LONG_TRAINING_RELEASE_CANDIDATE.md |  158 -
 docs/experiments/OUTPUT_DIRECTORY_LAYOUT.md        |  115 -
 .../causal8_original8_formal_retrace.md            |  191 -
 docs/experiments/date_organized_outputs_handoff.md |  229 -
 docs/experiments/original_merc_handoff.md          |   74 -
 docs/git_workflow.md                               |  293 -
 docs/module_implementation_spec.md                 |  223 -
 docs/project_map.md                                |  244 -
 docs/refactors/CONFIG_BATCH1_AFTER_SNAPSHOT.json   | 1721 -----
 docs/refactors/CONFIG_BATCH1_BEFORE_SNAPSHOT.json  | 1721 -----
 docs/refactors/CONFIG_BATCH1_MOVES.csv             |   17 -
 docs/refactors/CONFIG_BATCH1_REFACTOR_REPORT.md    |  286 -
 docs/refactors/CONFIG_BATCH1_REFERENCE_AUDIT.csv   |   50 -
 docs/refactors/CONFIG_BATCH1_SEMANTIC_DIFF.csv     |   17 -
 docs/refactors/CONFIG_BATCH2_AFTER_SNAPSHOT.json   | 2750 -------
 docs/refactors/CONFIG_BATCH2_BEFORE_SNAPSHOT.json  | 2750 -------
 docs/refactors/CONFIG_BATCH2_MOVES.csv             |   18 -
 docs/refactors/CONFIG_BATCH2_REFACTOR_REPORT.md    |   68 -
 docs/refactors/CONFIG_BATCH2_REFERENCE_AUDIT.csv   |  111 -
 docs/refactors/CONFIG_BATCH2_SEMANTIC_DIFF.csv     |   18 -
 docs/refactors/CONFIG_BATCH3_AFTER_SNAPSHOT.json   | 4243 -----------
 docs/refactors/CONFIG_BATCH3_BEFORE_SNAPSHOT.json  | 4243 -----------
 docs/refactors/CONFIG_BATCH3_MOVES.csv             |   18 -
 docs/refactors/CONFIG_BATCH3_REFACTOR_REPORT.md    |  248 -
 docs/refactors/CONFIG_BATCH3_REFERENCE_AUDIT.csv   |   69 -
 docs/refactors/CONFIG_BATCH3_SEMANTIC_DIFF.csv     |   18 -
 docs/refactors/CONFIG_BATCH4_AFTER_SNAPSHOT.json   | 2908 -------
 docs/refactors/CONFIG_BATCH4_BEFORE_SNAPSHOT.json  | 2908 -------
 docs/refactors/CONFIG_BATCH4_MOVES.csv             |   11 -
 docs/refactors/CONFIG_BATCH4_REFACTOR_REPORT.md    |  258 -
 docs/refactors/CONFIG_BATCH4_REFERENCE_AUDIT.csv   |   31 -
 docs/refactors/CONFIG_BATCH4_SEMANTIC_DIFF.csv     |   11 -
 docs/refactors/CONFIG_BATCH5_AFTER_SNAPSHOT.json   | 3507 ---------
 docs/refactors/CONFIG_BATCH5_BEFORE_SNAPSHOT.json  | 3507 ---------
 docs/refactors/CONFIG_BATCH5_MOVES.csv             |   14 -
 docs/refactors/CONFIG_BATCH5_REFACTOR_REPORT.md    |  273 -
 docs/refactors/CONFIG_BATCH5_REFERENCE_AUDIT.csv   |   47 -
 docs/refactors/CONFIG_BATCH5_SEMANTIC_DIFF.csv     |   14 -
 docs/refactors/CONFIG_BATCH6_AFTER_SNAPSHOT.json   | 7955 --------------------
 docs/refactors/CONFIG_BATCH6_BEFORE_SNAPSHOT.json  | 7955 --------------------
 .../CONFIG_BATCH6_MODEL_MEMBERSHIP_AUDIT.csv       |    3 -
 docs/refactors/CONFIG_BATCH6_MOVES.csv             |   38 -
 docs/refactors/CONFIG_BATCH6_REFACTOR_REPORT.md    |   41 -
 docs/refactors/CONFIG_BATCH6_REFERENCE_AUDIT.csv   |  169 -
 docs/refactors/CONFIG_BATCH6_SEMANTIC_DIFF.csv     |   38 -
 docs/refactors/CONFIG_BATCH7_AFTER_SNAPSHOT.json   | 6379 ----------------
 docs/refactors/CONFIG_BATCH7_BEFORE_SNAPSHOT.json  | 6379 ----------------
 docs/refactors/CONFIG_BATCH7_COLLISION_REVIEW.csv  |    3 -
 docs/refactors/CONFIG_BATCH7_MOVES.csv             |   74 -
 docs/refactors/CONFIG_BATCH7_REFACTOR_REPORT.md    |   84 -
 docs/refactors/CONFIG_BATCH7_REFERENCE_AUDIT.csv   |  306 -
 docs/refactors/CONFIG_BATCH7_SEMANTIC_DIFF.csv     |   74 -
 docs/refactors/CONFIG_CLASSIFICATION_PHASE4A.csv   |  184 -
 .../CONFIG_DUPLICATES_AND_CONFLICTS_PHASE4A.csv    |  717 --
 docs/refactors/CONFIG_ENTRYPOINT_AUDIT_PHASE4A.csv |  184 -
 docs/refactors/CONFIG_LAYOUT_PHASE4A_REPORT.md     |  355 -
 .../refactors/CONFIG_MIGRATION_BATCHES_PHASE4A.csv |    8 -
 docs/refactors/CONFIG_MIGRATION_PLAN_PHASE4A.csv   |  184 -
 docs/refactors/CONFIG_REFERENCE_GRAPH_PHASE4A.csv  |  768 --
 .../LEGACY_ANALYSIS_IMPORTS_AFTER_PHASE3B1.csv     |   48 -
 .../LEGACY_ANALYSIS_IMPORTS_AFTER_PHASE3B1.md      |   59 -
 .../LEGACY_MODEL_IMPORTS_AFTER_MODEL_REFACTOR.csv  |   54 -
 .../LEGACY_MODEL_IMPORTS_AFTER_MODEL_REFACTOR.md   |   73 -
 .../LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3A.csv  |   47 -
 .../LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3A.md   |   69 -
 .../LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3B2.csv |   11 -
 .../LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3B2.md  |   80 -
 docs/refactors/LEGACY_WRAPPER_RETIREMENT.csv       |   33 -
 .../LEGACY_WRAPPER_RETIREMENT_INVENTORY.csv        |   99 -
 docs/refactors/LEGACY_WRAPPER_RETIREMENT_REPORT.md |  168 -
 docs/refactors/MODEL_LAYOUT_REFACTOR_REPORT.md     |  213 -
 .../SCRIPT_ANALYSIS_LAYOUT_REFACTOR_REPORT.md      |  242 -
 .../SCRIPT_EXECUTION_LAYOUT_REFACTOR_REPORT.md     |  267 -
 .../SCRIPT_SUPPORT_CLASSIFICATION_PHASE3B2.csv     |   33 -
 .../SCRIPT_SUPPORT_LAYOUT_REFACTOR_REPORT.md       |  360 -
 docs/smoke_test_protocol.md                        |  163 -
 135 files changed, 28 insertions(+), 76824 deletions(-)
```

### git status --short

```text
 M .gitignore
 D docs/20260526_mmgcn_baseline_progress.md
 D docs/PROJECT_HANDOFF.md
 D docs/active_modalities_design.md
 D docs/baselines/baseline_rescreen_report.md
 D docs/baselines/causal_baseline_readiness_report.md
 D docs/baselines/causal_benchmark_review_tool_notes.md
 D docs/baselines/causal_dialoguegcn_port_notes.md
 D docs/baselines/causal_gsmcc_port_notes.md
 D docs/baselines/causal_model_contract.md
 D docs/baselines/causal_repairs_changelog.md
 D docs/baselines/configs_cleanup_audit.md
 D docs/baselines/current_merc_model_inventory.md
 D docs/baselines/dialoguegcn_official_source_audit.md
 D docs/baselines/gsmcc_official_source_audit.md
 D docs/baselines/iemocap_split_protocol_audit.md
 D docs/baselines/loss_stability_training_audit.md
 D docs/baselines/mm_dfn_smoke_review.md
 D docs/baselines/mmgcn_causal_audit.md
 D docs/baselines/multidag_causal_audit.md
 D docs/baselines/multidag_cl_baseline_stabilization_notes.md
 D docs/baselines/multidag_cl_encoder_audit_notes.md
 D docs/baselines/multidag_cl_encoder_diff_audit_notes.md
 D docs/baselines/multidag_cl_formal_run_plan.md
 D docs/baselines/multidag_cl_formal_yaml_standardization_notes.md
 D docs/baselines/multidag_cl_iemocap_experiment_matrix_notes.md
 D docs/baselines/multidag_cl_loss_stability_notes.md
 D docs/baselines/multidag_cl_multirun_stabilization_analysis_notes.md
 D docs/baselines/multidag_cl_observability_and_figures_notes.md
 D docs/baselines/multidag_cl_paper_artifacts_notes.md
 D docs/baselines/multidag_cl_pipeline_integration_notes.md
 D docs/baselines/multidag_cl_port_notes.md
 D docs/baselines/multidag_cl_real_batch_notes.md
 D docs/baselines/multidag_cl_real_data_path_protocol.md
 D docs/baselines/multidag_cl_smoke_review.md
 D docs/baselines/multidag_cl_smoke_training_notes.md
 D docs/baselines/multidag_cl_yaml_parameter_guide.md
 D docs/baselines/new_causal_baselines_remote_protocol.md
 D docs/baselines/new_causal_baselines_task1_report.md
 D docs/baselines/new_causal_baselines_task2_report.md
 D docs/baselines/original_repro/README.md
 D docs/baselines/original_repro/dialoguegcn_paper_code_mapping.md
 D docs/baselines/original_repro/gsmcc_numerical_stability_report.md
 D docs/baselines/original_repro/gsmcc_paper_code_mapping.md
 D docs/baselines/original_repro/mmgcn_paper_code_mapping.md
 D docs/baselines/original_repro/multidag_cl_paper_code_mapping.md
 D docs/baselines/original_repro/original_merc_reproduction_handoff.md
 D docs/baselines/original_repro/paper_targets.csv
 D docs/baselines/original_repro/reported_results_reference.csv
 D docs/baselines/original_repro/source_audit.md
 D docs/baselines/sdt_isolated_smoke_review.md
 D docs/baselines/sdt_port_notes.md
 D docs/codex_workflow.md
 D docs/data/iemocap_clean_roberta_v1.md
 D docs/experiment_protocol.md
 D docs/experiments/DIALOGUEGCN_GSMCC_FULL_REPAIR_PREPARATION.md
 D docs/experiments/FORMAL_FULL16_REPAIRED.md
 D docs/experiments/LONG_TRAINING_COMMANDS.sh
 D docs/experiments/LONG_TRAINING_MATRIX.csv
 D docs/experiments/LONG_TRAINING_RELEASE_CANDIDATE.md
 D docs/experiments/OUTPUT_DIRECTORY_LAYOUT.md
 D docs/experiments/causal8_original8_formal_retrace.md
 D docs/experiments/date_organized_outputs_handoff.md
 D docs/experiments/original_merc_handoff.md
 D docs/git_workflow.md
 D docs/module_implementation_spec.md
 D docs/project_map.md
 D docs/refactors/CONFIG_BATCH1_AFTER_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH1_BEFORE_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH1_MOVES.csv
 D docs/refactors/CONFIG_BATCH1_REFACTOR_REPORT.md
 D docs/refactors/CONFIG_BATCH1_REFERENCE_AUDIT.csv
 D docs/refactors/CONFIG_BATCH1_SEMANTIC_DIFF.csv
 D docs/refactors/CONFIG_BATCH2_AFTER_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH2_BEFORE_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH2_MOVES.csv
 D docs/refactors/CONFIG_BATCH2_REFACTOR_REPORT.md
 D docs/refactors/CONFIG_BATCH2_REFERENCE_AUDIT.csv
 D docs/refactors/CONFIG_BATCH2_SEMANTIC_DIFF.csv
 D docs/refactors/CONFIG_BATCH3_AFTER_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH3_BEFORE_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH3_MOVES.csv
 D docs/refactors/CONFIG_BATCH3_REFACTOR_REPORT.md
 D docs/refactors/CONFIG_BATCH3_REFERENCE_AUDIT.csv
 D docs/refactors/CONFIG_BATCH3_SEMANTIC_DIFF.csv
 D docs/refactors/CONFIG_BATCH4_AFTER_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH4_BEFORE_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH4_MOVES.csv
 D docs/refactors/CONFIG_BATCH4_REFACTOR_REPORT.md
 D docs/refactors/CONFIG_BATCH4_REFERENCE_AUDIT.csv
 D docs/refactors/CONFIG_BATCH4_SEMANTIC_DIFF.csv
 D docs/refactors/CONFIG_BATCH5_AFTER_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH5_BEFORE_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH5_MOVES.csv
 D docs/refactors/CONFIG_BATCH5_REFACTOR_REPORT.md
 D docs/refactors/CONFIG_BATCH5_REFERENCE_AUDIT.csv
 D docs/refactors/CONFIG_BATCH5_SEMANTIC_DIFF.csv
 D docs/refactors/CONFIG_BATCH6_AFTER_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH6_BEFORE_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH6_MODEL_MEMBERSHIP_AUDIT.csv
 D docs/refactors/CONFIG_BATCH6_MOVES.csv
 D docs/refactors/CONFIG_BATCH6_REFACTOR_REPORT.md
 D docs/refactors/CONFIG_BATCH6_REFERENCE_AUDIT.csv
 D docs/refactors/CONFIG_BATCH6_SEMANTIC_DIFF.csv
 D docs/refactors/CONFIG_BATCH7_AFTER_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH7_BEFORE_SNAPSHOT.json
 D docs/refactors/CONFIG_BATCH7_COLLISION_REVIEW.csv
 D docs/refactors/CONFIG_BATCH7_MOVES.csv
 D docs/refactors/CONFIG_BATCH7_REFACTOR_REPORT.md
 D docs/refactors/CONFIG_BATCH7_REFERENCE_AUDIT.csv
 D docs/refactors/CONFIG_BATCH7_SEMANTIC_DIFF.csv
 D docs/refactors/CONFIG_CLASSIFICATION_PHASE4A.csv
 D docs/refactors/CONFIG_DUPLICATES_AND_CONFLICTS_PHASE4A.csv
 D docs/refactors/CONFIG_ENTRYPOINT_AUDIT_PHASE4A.csv
 D docs/refactors/CONFIG_LAYOUT_PHASE4A_REPORT.md
 D docs/refactors/CONFIG_MIGRATION_BATCHES_PHASE4A.csv
 D docs/refactors/CONFIG_MIGRATION_PLAN_PHASE4A.csv
 D docs/refactors/CONFIG_REFERENCE_GRAPH_PHASE4A.csv
 D docs/refactors/LEGACY_ANALYSIS_IMPORTS_AFTER_PHASE3B1.csv
 D docs/refactors/LEGACY_ANALYSIS_IMPORTS_AFTER_PHASE3B1.md
 D docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_MODEL_REFACTOR.csv
 D docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_MODEL_REFACTOR.md
 D docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3A.csv
 D docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3A.md
 D docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3B2.csv
 D docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3B2.md
 D docs/refactors/LEGACY_WRAPPER_RETIREMENT.csv
 D docs/refactors/LEGACY_WRAPPER_RETIREMENT_INVENTORY.csv
 D docs/refactors/LEGACY_WRAPPER_RETIREMENT_REPORT.md
 D docs/refactors/MODEL_LAYOUT_REFACTOR_REPORT.md
 D docs/refactors/SCRIPT_ANALYSIS_LAYOUT_REFACTOR_REPORT.md
 D docs/refactors/SCRIPT_EXECUTION_LAYOUT_REFACTOR_REPORT.md
 D docs/refactors/SCRIPT_SUPPORT_CLASSIFICATION_PHASE3B2.csv
 D docs/refactors/SCRIPT_SUPPORT_LAYOUT_REFACTOR_REPORT.md
 D docs/smoke_test_protocol.md
?? configs/multidag_cl/coop_t1024_v0/
?? configs/multidag_cl/dcn_t1024_v0/
?? configs/multidag_cl/track_t1024_v0/
?? datasets/asymmetric_iemocap.py
?? docs/IEMOCAP_ASYMMETRIC_MERC_PROJECT_SOURCE_20261021_v1.2.md
?? docs/MERC_PATHS_PROJECT_SOURCE_20261022.md
?? docs/reproduction/MULTIDAG_CL_FEATURE_REPRODUCTION_STAGE_C1.md
?? docs/research/
?? models/multidag_cl/graph_core_variants/
?? scripts/analysis/models/multidag_cl/plot_overfitting_diagnostics.py
?? scripts/models/multidag_cl/graph_core_variants/
?? scripts/runtime/multidag_cl_graph_core_variants/
?? tests/analysis/
?? tests/data/build/multidag_cl/test_official_feature_pipeline.py
?? tests/models/multidag_cl/graph_core_variants/
?? tests/test_iemocap_canonical_manifest.py
```

### 本任务 changed-file exact list（22 个新增文件）

```text
configs/multidag_cl/coop_t1024_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev.yaml
configs/multidag_cl/dcn_t1024_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev.yaml
configs/multidag_cl/track_t1024_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev.yaml
docs/research/dag_core_variants/DUNE_COMPATIBILITY_ANALYSIS.md
docs/research/dag_core_variants/GRAPHCORE_ARCHITECTURE_DECISION.md
docs/research/dag_core_variants/IMPLEMENTATION_REPORT.md
docs/research/dag_core_variants/PAPER_TO_MULTIDAG_MAPPING.md
models/multidag_cl/graph_core_variants/__init__.py
models/multidag_cl/graph_core_variants/common.py
models/multidag_cl/graph_core_variants/config.py
models/multidag_cl/graph_core_variants/coop.py
models/multidag_cl/graph_core_variants/dcn.py
models/multidag_cl/graph_core_variants/model.py
models/multidag_cl/graph_core_variants/track.py
scripts/models/multidag_cl/graph_core_variants/train.py
scripts/runtime/multidag_cl_graph_core_variants/__init__.py
scripts/runtime/multidag_cl_graph_core_variants/formal.py
scripts/runtime/multidag_cl_graph_core_variants/training.py
tests/models/multidag_cl/graph_core_variants/conftest.py
tests/models/multidag_cl/graph_core_variants/test_cores.py
tests/models/multidag_cl/graph_core_variants/test_model.py
tests/models/multidag_cl/graph_core_variants/test_runtime.py
```
