# Q2-0D TA-v0 最小实现报告

日期：2026-11-09。WORKSTREAM = MULTIDAG。

| 字段 | 结果 |
| --- | --- |
| STATUS | COMPLETE：最小源码、CPU synthetic 合同验证完成；不表示正式真实数据训练已经就绪。 |
| WORKSTREAM_CHECK | PASS；分支 feat/multidag-cl-paper-reimplementation，指定 MULTIDAG 主工作区，origin fetch/push 均通过 guard。 |
| CHANGED_FILES | .gitignore 仅追加一行精确例外：!/tests/data/test_ta_v0_dataset.py；原有字节与历史修改全部保留。 |
| NEW_FILES | 11 个实现/配置/测试文件，以及本报告；清单见下文。 |
| MODEL_ARCHITECTURE | T768→Linear(768,100)→packed 单层 UniLSTM(100)；A1582→Linear(1582,100)→ReLU；TA projection→4 层原 MultiDAG→classifier。模型 API 无 visual 参数、字段、encoder、零掩码槽位。 |
| TA_FUSION_PROJECTION | ta_fusion_projection = Linear(200,300)，输入 concat(T100,A100)；无新增 attention/gate/router/非线性块。 |
| H_TA_CONTRACT | concat(h0,h1,h2,h3,h4)，形状 [B,Lmax,1500]；hook 验证 classifier 接收到同一个 h_ta tensor。classifier 沿用 1500→300→300→6、ReLU、dropout 0.4。 |
| DATASET_CONTRACT | 独立 TAV0Dataset，面向 asymmetric_merc_v1 / iemocap_asym_ta_v1；显式 TA PKL、bridge CSV、canonical CSV 三个路径；不导入既有 untracked datasets/asymmetric_iemocap.py。 |
| CANONICAL_ORDER_IMPLEMENTATION | 权威 canonical global_order→utterance_id→经核对的 ta_row_index→原 T/A 行；bridge canonical_dialogue_order 必须一致。canonical 允许包含六类 TA 之外的权威 utterance。原数组与 PKL 均不改写。 |
| LABEL_CONTRACT | 固定 0 hap、1 sad、2 neu、3 ang、4 exc、5 fru；itos、可选 stoi、asset/bridge label ID 与名称逐行一致；异序 source mapping 硬失败，未实现自动 remap。 |
| TRAIN_DEV_ONLY_RUNTIME | DevelopmentRunner 仅接受 train/dev，Test 在 factory/文件读取前拒绝；curriculum OFF、seed100、Dev Weighted-F1 选择、相同分数保留较早候选、Test evaluation count=0。CLI 本阶段强制 synthetic CPU dry-run。 |
| DEV_EXPORT_CONTRACT | CSV：utterance_id,true_label,predicted_label,logit_0…5,prob_0…5；按 ID 稳定排序，排除 padding，重复 ID/缺失 ID/split 错误先失败再写出；禁止覆盖已有导出目录。真实 Dev 默认要求 expected_rows=980 且提供权威 expected_ids。 |
| H_TA_EXPORT_CONTRACT | NPZ：utterance_ids,h_ta,labels,split；无需 pickle 即可读取。JSON metadata 包含 representation_name=ta_v0_contextual_preclassifier、dimension=1500、label_vocab、checkpoint_identity、config_identity。CSV/NPZ 共用同一记录集合和排列。 |
| TESTS_RUN | 新增 43 项 synthetic tests；历史 encoder、graph、attention/GRU、model 31 项回归；10 个新增 Python 文件 compile/AST 检查；train/export 两个 CLI dry-run。 |
| TEST_RESULTS | 74 passed；语法检查 10/10；CLI 均 PASS。Synthetic Dev 导出 4 行、4 个唯一 ID、h_ta=[4,1500]、重复=0、padding=0。概率由 logits.softmax 生成且各行和约为 1，输出有限。 |
| OLD_PATH_REGRESSION | 历史模型/runtime 文件修改为 0；31 项历史组件测试通过；另新增相同权重下 TA text/audio 与历史 UniLSTM encoder 的逐元素精确等价测试。旧模型行为与 state_dict schema 不变，新 TA-v0 使用独立 checkpoint identity。 |
| REAL_DATA_USED | NO。 |
| GPU_USED | NO；现有 torch 2.6.0+cpu，所有新增执行限定 CPU。 |
| TEST_DATA_USED | NO；未读取真实数据，未构建 Test loader，未运行 Test evaluation。Synthetic fixture 中仅保留 Test split 的身份元数据用于边界校验，不送入模型。 |
| TRAINING_STARTED | NO；synthetic loss.backward 合同检查=YES，optimizer steps=0，无真实训练、无 checkpoint 写出。 |
| GIT_OPERATION_PERFORMED | 仅 status/diff/rev-parse/branch/remote/check-ignore/ls-files/show 等读取；无 stage/commit/push/reset/clean/stash/restore/pull，未变更分支或 remote。 |
| KNOWN_LIMITATIONS | 本阶段不含真实 fit/optimizer/持久 checkpoint 流程，也不含真实 checkpoint export CLI；配置中真实路径保持 null 并拒绝真实运行。仅验证 canonical utterance-order 因果性，依赖输入特征为 utterance-local；未重新认证上游 extractor、不声称 strict wall-clock streaming，也未解决 overlap 部署语义。真实 Dev 980 行尚未实测。 |
| BLOCKERS_BEFORE_REMOTE_SMOKE | 需另行审查并交付本轮文件；确认 portable package、bridge、独立权威 canonical metadata 的远程显式路径与版本/哈希；新任务中开放有界的真实 Train/Dev smoke 接口并继续隔离 Test。正式训练与持久 checkpoint runtime 仍需后续实现。 |
| NEXT_SAFE_STEP | 先代码审查本轮 TA-only 改动和报告；再以独立任务做只读真实 Train/Dev package preflight、有限 CPU forward smoke。真实训练、GPU 与 Test evaluation 不属于本轮授权。 |

因果性复用 CausalPredecessorBuilder、SpeakerRelationBuilder、RelationAwareAttention、DualGRUNodeUpdate、MultiDAGLayer；边只能从过去指向当前，禁止 future/self edge。未引入新 graph、attention 或 CL7，未修改 S2D 或融合模型。

## 新文件清单

- models/multidag_cl/ta_v0/config.py
- models/multidag_cl/ta_v0/encoders.py
- models/multidag_cl/ta_v0/model.py
- models/multidag_cl/ta_v0/__init__.py
- datasets/iemocap/ta_v0_dataset.py
- scripts/models/multidag_cl/ta_v0/train.py
- scripts/models/multidag_cl/ta_v0/export_dev.py
- configs/multidag_cl/ta_v0/iemocap/causal_context/asymmetric_merc_v1/baseline.yaml
- tests/data/test_ta_v0_dataset.py
- tests/models/multidag_cl/ta_v0/test_contracts.py
- tests/models/multidag_cl/ta_v0/test_export_dev.py
- docs/reproduction/TA_V0_Q2_0D_IMPLEMENTATION_REPORT.md

## 验证入口

使用现有 m3ed_mmgcn 环境；测试关闭 cacheprovider，临时文件在独立系统临时目录，禁用 Python bytecode，CPU 单线程。没有安装依赖，没有在 data/outputs/checkpoints/logs 下写入资产。

```text
python -B -m pytest -q tests/data/test_ta_v0_dataset.py tests/models/multidag_cl/ta_v0 tests/paper_reimplementation/multidag_cl/test_encoders.py tests/paper_reimplementation/multidag_cl/test_graph.py tests/paper_reimplementation/multidag_cl/test_attention_and_gru.py tests/paper_reimplementation/multidag_cl/test_model.py -p no:cacheprovider --basetemp <fresh-system-temp-dir>
74 passed in 1.66s

python -B scripts/models/multidag_cl/ta_v0/train.py --synthetic-dry-run
status=PASS, optimizer_steps=0, test_evaluation_count=0

python -B scripts/models/multidag_cl/ta_v0/export_dev.py --synthetic-dry-run --output-dir <fresh-system-temp-export-dir>
rows=4, unique_utterance_ids=4, duplicate_ids=0, padding_rows=0
h_ta=[4,1500]
```

## Git 审核说明

本轮新增文件均未暂存，因此普通 git diff --stat 不包含它们。下方 tracked diff 中的大量文档删除和 .gitignore 的原有 27 行增加在任务开始前已存在；本轮只在 .gitignore 额外追加 1 行。未恢复用户删除的文档。原有 143 条 dirty/untracked 状态条目均保留。

Git 报告可能出现旧 .pytest_cache / tests/_tmp_long_training 目录的历史权限警告；本轮测试使用独立系统临时目录并全部通过。

## git diff --stat

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

Exit code: 0

## git status --short

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
?? configs/multidag_cl/ta_v0/
?? datasets/asymmetric_iemocap.py
?? datasets/iemocap/ta_v0_dataset.py
?? docs/IEMOCAP_ASYMMETRIC_MERC_PROJECT_SOURCE_20261021_v1.2.md
?? docs/MERC_PATHS_PROJECT_SOURCE_20261022.md
?? docs/reproduction/MULTIDAG_CL_FEATURE_REPRODUCTION_STAGE_C1.md
?? docs/reproduction/TA_V0_Q2_0D_IMPLEMENTATION_REPORT.md
?? models/multidag_cl/ta_v0/
?? scripts/analysis/models/multidag_cl/plot_overfitting_diagnostics.py
?? scripts/models/multidag_cl/ta_v0/
?? tests/analysis/
?? tests/data/build/multidag_cl/test_official_feature_pipeline.py
?? tests/data/test_ta_v0_dataset.py
?? tests/models/multidag_cl/
?? tests/test_iemocap_canonical_manifest.py
```

Exit code: 0

## git diff --check

```text
(no output)
```

Exit code: 0
