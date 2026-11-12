# CHG_V0_IMPLEMENTATION

日期：2026-11-12。WORKSTREAM = MULTIDAG。未 commit、未 push，等待审核。

## 身份与范围

- branch：`feat/multidag-cl-paper-reimplementation`
- HEAD：`18cc2d3391041fead5a4cf7e255a743468f2828b`
- origin fetch/push：`https://github.com/ochisakura-2002/m3ed_mmgcn_clean.git`
- `python scripts/dev/check_workstream.py multidag`：PASS。
- 开始时已有 `.gitignore` 修改、134 个 tracked 文档删除及其他 untracked 文件。全部保留；本任务不恢复文档、不切分支、不同步远程。缺失的入口规范通过 `git show HEAD:<path>` 只读查阅。
- 未读取真实 IEMOCAP 资产或真实 checkpoint，未运行正式训练，未构造 Test loader、评估或导出 Test。测试只用现有 synthetic fixtures；负向测试验证 Test 请求被拒绝。

## 实现定位

1. `paper_reimplementation/attention.py` 的 `RelationAwareAttention.forward` 使用 concat-linear logits + masked softmax 产生原 semantic attention，随后执行原 `W_same/W_different` 与 message 聚合。CHG 插入两者之间。
2. `paper_reimplementation/graph.py` 的 `CausalPredecessorBuilder.build` 向后扫描 speaker，生成 receiver/source adjacency。未修改。
3. `paper_reimplementation/dag_layer.py` 的 `MultiDAGLayer.forward` 顺序构造 completed states，只传递 `[0,i)` prefix 与既有 predecessor row；self/future adjacency 在 attention 前报错。未修改。
4. `datasets/iemocap/ta_v0_dataset.py` 按 provenance canonical dialogue order 排序，collate 保持顺序；speaker ID 为 int64 `[B,L]`。未修改。CHG 只读已有 same/different relation。
5. 训练入口仍为 `scripts/models/multidag_cl/ta_v0/train.py`；模型五个输入与 `TAOutput(h_ta, logits_ta, prob_ta, mask)` 不变；h_ta 仍为 1500 维。

CHG 给历史语义相关性增加 speaker/turn-distance reliability 调制。只 gather 原 predecessor mask 为真的边；target 为 completed prefix 的宽度 i，source 为列位置 j，距离严格为 i-j，不使用时间戳、存储行号或 ID 后缀。

## 文件清单及原因

| 文件 | 原因 |
| --- | --- |
| `models/multidag_cl/paper_reimplementation/attention.py` | 原 softmax 与 value 聚合间增加默认 None 的可选重加权调用。 |
| `models/multidag_cl/ta_v0/causal_history_gate.py` | 新增六桶函数、2 类关系 embedding、gate MLP、归一化、显式诊断 scope。 |
| `models/multidag_cl/ta_v0/config.py` | 默认关闭开关与 8/8/64 维度参数；OFF 保留原 checkpoint config/lineage。 |
| `models/multidag_cl/ta_v0/model.py` | 每层独立挂载 CHG，原模块初始化完毕后创建新参数，保留已有参数的 seed 初始化值。 |
| `scripts/runtime/multidag_cl_ta_v0/training.py` | 接受冻结 A0 或 CHG-v0 A1；聚合 Dev 诊断并保留旧 checkpoint identity。 |
| `scripts/runtime/multidag_cl_ta_v0/formal.py` | 严格校验 A1 最小差异，隔离输出名称，每个 Dev epoch 写一条诊断。 |
| `scripts/models/multidag_cl/ta_v0/export_dev.py` | 重载 best checkpoint 后，在同一次 Dev 导出中写 chg_gate_diagnostics.json。 |
| `scripts/models/multidag_cl/ta_v0/smoke_chg.py` | CPU synthetic forward/backward/诊断，无 optimizer、真实数据或文件写入。 |
| `configs/multidag_cl/ta_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev_chg_v0.yaml` | 冻结 formal YAML 的最小差异副本。 |
| `tests/models/multidag_cl/ta_v0/test_chg.py` | 七个定向 CHG 测试，包含有界 synthetic best-checkpoint/Dev-export 集成检查。 |
| `tests/models/multidag_cl/ta_v0/test_formal_runtime.py` | checkpoint config 对照冻结 YAML，确认新增 dataclass 字段不改变旧身份。 |
| `docs/reproduction/CHG_V0_IMPLEMENTATION.md` | 交付与审计记录。 |

## 核心 forward

~~~python
# 每条边：[E,300] + [E,300] + [E,8] + [E,8]
features = torch.cat((query, history,
                      self.speaker_embedding(same_speaker.long()),
                      self.distance_embedding(distance_bucket(distances))), dim=-1)
gates = self.mlp(features).squeeze(-1)

# reweight：MultiDAG 只传入 completed [0,i) prefix。
batch, source = predecessor_mask.nonzero(as_tuple=True)
if source.numel() == 0:
    return semantic_weights
target = history_prefix.shape[1]
gates = self(query[batch], history_prefix[batch, source],
             same_mask[batch, source], target - source)
gated = torch.zeros_like(semantic_weights)
gated[batch, source] = semantic_weights[batch, source] * gates
weights = gated / (gated.sum(dim=-1, keepdim=True) + 1e-8)
~~~

这是 gate forward 与 reweight 两部分的节选。随后仍走原 relation-value message 聚合及 dual-GRU update。无 predecessor 时零 message。未建立第二套 DAG 或完整 gate 矩阵。

距离桶：1→0、2→1、3→2、4–5→3、6–10→4、>10→5；非正距离被独立函数拒绝。

## 参数量及兼容性

- 每层：speaker embedding 2×8=16；distance embedding 6×8=48；MLP 616×64+64+64×1+1=39,553。
- 每层总计 **39,617**；四层独立新增 **158,468**。
- A0 总参数 **5,975,510**；A1 总参数 **6,133,978**。
- 使用 PyTorch Embedding/Linear 默认初始化，受现有 seed 控制；不跨层共享。
- OFF 不创建 gate 参数；原 state_dict keys、extra_state 和 runtime checkpoint identity 保留。
- ON 使用 ta_v0_chg_v0 身份，A0/A1 checkpoint 不可误载。

## 配置 diff

~~~diff
   classifier_dropout: 0.4
+  use_causal_history_gate: true
+  chg_speaker_embedding_dim: 8
+  chg_distance_embedding_dim: 8
+  chg_hidden_dim: 64
...
-  experiment_group: ta_v0_formal
+  experiment_group: ta_v0_chg_v0_formal
~~~

其余 YAML 相同。split、seed、batch、optimizer、LR、weight decay、epochs、dropout、DAG 层数、hidden size、类别规则、selection metric、early stopping 均不变。原 formal YAML 未写入，SHA-256 仍为 `c18a3e68580bd3171533b44b27deeda7c3b61a0ab72a203afbb106a4c75ab7eb`。

TAV0Config 默认 use_causal_history_gate=False；旧 YAML 无需添加字段，显式 OFF 也走原路径。formal validator 仅接受冻结维度 8/8/64 的 CHG-v0。

## Gate diagnostics

- 仅在显式 collect_gate_diagnostics(model) scope 内且 model 为 eval 时累计 detached scalar moments；Train 不收集、不逐 batch 写文件。
- Dev evaluation 返回全 Dev 聚合；每 epoch 在 `logs/chg_gate_diagnostics.jsonl` 写一行。
- best checkpoint 重载后，从同一次 Dev inference 在 run 根目录输出 `chg_gate_diagnostics.json`，附 checkpoint/config 身份与 split=dev。
- all、same_speaker、different_speaker、六个 distance bucket 均含 edge_count、mean、population std。
- 计数单位为各 DAG layer 的边出现次数；all 合并四层。空组 count=0、mean/std=null。只保存 9×3 个 moments，不保存逐边 tensor。

## 验证命令与结果

在 MultiDAG worktree 中：

~~~powershell
$testTemp = Join-Path $env:TEMP ('merc_chg_verified_' + [guid]::NewGuid().ToString('N'))
conda run --no-capture-output -n m3ed_mmgcn python -B -m pytest -q tests/models/multidag_cl/ta_v0 tests/data/test_ta_v0_dataset.py tests/paper_reimplementation/multidag_cl/test_encoders.py tests/paper_reimplementation/multidag_cl/test_graph.py tests/paper_reimplementation/multidag_cl/test_attention_and_gru.py tests/paper_reimplementation/multidag_cl/test_model.py -p no:cacheprovider --basetemp $testTemp
~~~

**179 passed in 17.79s**。覆盖 bucket 边界、gate finite/range/shape、原 semantic attention 乘积与归一化、原 value/message 公式、诊断分组矩、既有 past edges、future perturbation/prefix equivalence、非法 future adjacency 在 gate 前拒绝、OFF 身份/契约、无跨层共享、配置差异、梯度、synthetic checkpoint→best reload→Dev diagnostics。

初轮有一处新断言错误信息匹配拼写（frozen/freezes），已修正。沙箱内 pytest 临时目录曾触发 Windows WinError 5，正常临时目录权限下重跑后全部通过。未安装依赖。

~~~powershell
conda run --no-capture-output -n m3ed_mmgcn python -B scripts/models/multidag_cl/ta_v0/smoke_chg.py
~~~

**PASS**：construct/forward/backward、所有 CHG 参数 tensor 非零有限梯度、OFF forward/backward、Dev diagnostics 全通过。
loss=1.8003689050674438；h_ta_shape=[2,3,1500]；added_parameters=158468。
独立 smoke optimizer_steps=0、training_started=false、test_evaluation_count=0。集成回归中的 optimizer/checkpoint 仅使用小 synthetic fixtures。

Dev smoke 的 12 个 layer-edge occurrences：all mean=0.5074022809664408、std=0.008411062012786429；same=4、different=8；bucket0=8、bucket1=4。其余桶由直接 attention 测试覆盖。

额外验证：

- 修改前保存 seed100 synthetic 原 state/batch/输出/梯度；修改后严格加载旧 state_dict，四个输出与全部参数梯度均 torch.equal：PASS。
- 冻结 formal YAML 哈希不变：PASS。
- 修改 Python 文件 py_compile.compile(..., cfile=<task temporary file>, doraise=True)：PASS；编译产物仅写系统临时目录。
- tracked git diff --check、新增文件 whitespace 检查：PASS。
- 开始时 dirty/untracked status entries 全部保留；本任务只写清单文件。

## 明确回答

| 问题 | 回答 |
| --- | --- |
| TA-v0 默认行为保持？ | 是；OFF 旧 checkpoint、输出及梯度逐元素一致，默认 YAML 未改。 |
| future edges 对 CHG 完全不可访问？ | 是；只得到完成的 [0,i) 历史 prefix，再按既有 mask gather；非法 self/future adjacency 在 gate 前拒绝。 |
| speaker relation 只用 same/different？ | 是；embedding 大小固定为 2。 |
| distance 严格来自 dialogue 内顺序？ | 是；i-j，基于现有 canonical-order batch 的 prefix 列位置。 |
| 访问过 Test？ | 未访问真实 Test；未构造 Test loader、评估或导出 Test。 |


## Git diff --stat 与 git status --short

任务范围 tracked diff（新增文件仍 untracked，未暂存）：

~~~text
 .../paper_reimplementation/attention.py            |  6 +++++
 models/multidag_cl/ta_v0/config.py                 | 24 +++++++++++++++++-
 models/multidag_cl/ta_v0/model.py                  | 14 +++++++++--
 scripts/models/multidag_cl/ta_v0/export_dev.py     |  8 +++++-
 scripts/runtime/multidag_cl_ta_v0/formal.py        | 29 ++++++++++++++++------
 scripts/runtime/multidag_cl_ta_v0/training.py      | 14 +++++++----
 .../multidag_cl/ta_v0/test_formal_runtime.py       |  4 +--
 7 files changed, 81 insertions(+), 18 deletions(-)
~~~

另有 5 个新文件，均见上方文件清单。完整命令输出包含任务开始前的 134 个文档删除和 .gitignore 修改；这些不属于 CHG。

<details>
<summary>完整 git diff --stat</summary>

~~~text
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
 .../paper_reimplementation/attention.py            |    6 +
 models/multidag_cl/ta_v0/config.py                 |   24 +-
 models/multidag_cl/ta_v0/model.py                  |   14 +-
 scripts/models/multidag_cl/ta_v0/export_dev.py     |    8 +-
 scripts/runtime/multidag_cl_ta_v0/formal.py        |   29 +-
 scripts/runtime/multidag_cl_ta_v0/training.py      |   14 +-
 .../multidag_cl/ta_v0/test_formal_runtime.py       |    4 +-
 142 files changed, 109 insertions(+), 76842 deletions(-)
~~~

</details>

<details>
<summary>完整 git status --short</summary>

~~~text
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
 M models/multidag_cl/paper_reimplementation/attention.py
 M models/multidag_cl/ta_v0/config.py
 M models/multidag_cl/ta_v0/model.py
 M scripts/models/multidag_cl/ta_v0/export_dev.py
 M scripts/runtime/multidag_cl_ta_v0/formal.py
 M scripts/runtime/multidag_cl_ta_v0/training.py
 M tests/models/multidag_cl/ta_v0/test_formal_runtime.py
?? configs/multidag_cl/ta_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev_chg_v0.yaml
?? datasets/asymmetric_iemocap.py
?? docs/IEMOCAP_ASYMMETRIC_MERC_PROJECT_SOURCE_20261021_v1.2.md
?? docs/MERC_PATHS_PROJECT_SOURCE_20261022.md
?? docs/reproduction/CHG_V0_IMPLEMENTATION.md
?? docs/reproduction/MULTIDAG_CL_FEATURE_REPRODUCTION_STAGE_C1.md
?? models/multidag_cl/ta_v0/causal_history_gate.py
?? scripts/analysis/models/multidag_cl/plot_overfitting_diagnostics.py
?? scripts/models/multidag_cl/ta_v0/smoke_chg.py
?? tests/analysis/
?? tests/data/build/multidag_cl/test_official_feature_pipeline.py
?? tests/models/multidag_cl/ta_v0/test_chg.py
?? tests/test_iemocap_canonical_manifest.py
~~~

</details>
