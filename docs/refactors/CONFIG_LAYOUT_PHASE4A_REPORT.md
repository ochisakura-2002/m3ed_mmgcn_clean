# Config Layout Refactor Phase 4A Report

## 1. PRECHECK

- Repo root：`E:/MERC/m3ed_mmgcn_clean`
- Branch：`refactor/model-first-layout`
- HEAD：`4f97adf513882d90457c17e49c622ad0682413f2`
- Upstream：`origin/refactor/model-first-layout`，ahead/behind=`0/0`
- 最近四个 layout commit 已包含 model、execution scripts、analysis scripts 与 support scripts 重构。
- 执行前 tracked working tree 与 staged 区干净。
- 唯一 untracked 文件为两份受保护组会文件；未读取为生产依据、未修改、未 staged。

## 2. CONTEXT_READ

按任务指定顺序完整读取了 `AGENTS.md`、`docs/PROJECT_CONTEXT.md`、
`docs/project_map.md`、四份已完成 layout report、
`LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3B2.md`、
`docs/experiment_protocol.md`、`docs/codex_workflow.md` 与
`docs/git_workflow.md`。没有创建第二套持久项目上下文。

## 3. BASELINE_GATES

- 执行前完整 pytest：`348 passed, 3 skipped, 30 warnings`。
- `scripts/dev/validate_config_tree.py`：PASS。
- `git diff --check`：PASS。
- 受限权限下首次 pytest collection 只因既有 repository `tmp/` 与系统 pytest
  临时目录不可读而失败；正常本地权限下原命令通过，未删除或修改这些目录。

## 4. TRACKED YAML INVENTORY

使用 `git ls-files configs` 发现 183 个 `.yaml` / `.yml` 文件。每个文件均由
YAML parser 实际加载并读取内容；解析失败数为 0。

当前目录分布：

| Current area | YAML count |
|---|---:|
| `configs/pipeline/` | 63 |
| `configs/baselines/` | 62 |
| `configs/experiments/` | 18 |
| `configs/smoke/` | 16 |
| `configs/analysis/` | 11 |
| `configs/modality_ablation/` | 8 |
| `configs/` root | 4 |
| `configs/data/` | 1 |

## 5. CLASSIFICATION

职责分布：

| Scope | Count |
|---|---:|
| model | 106 |
| pipeline | 63 |
| analysis | 11 |
| benchmark | 2 |
| shared_data | 1 |

模型分布：

| Model | Count |
|---|---:|
| multidag_cl | 94 |
| mmgcn | 42 |
| dialoguegcn | 20 |
| gsmcc | 20 |
| simple_mlp | 1 |
| not applicable / cross-model | 6 |

Implementation 分布：

| Implementation | Count |
|---|---:|
| unified | 140 |
| project_variant | 20 |
| paper_aligned | 18 |
| not applicable / mixed | 5 |

Dataset 分布为 IEMOCAP 165、M3ED 11、synthetic 7。Context 分布为
`causal_context` 136、`full_context` 40、not applicable / mixed 7。
Feature 分布为 `legacy_mmgcn_features` 123、`clean_roberta_features` 33、
`m3ed_features` 11、`synthetic` 7、not applicable / mixed 9。

Purpose 分布：

| Purpose | Count |
|---|---:|
| formal | 81 |
| stability_ablation | 30 |
| modality_ablation | 22 |
| smoke | 16 |
| analysis | 12 |
| context_ablation | 10 |
| missing_modality | 4 |
| screening | 4 |
| development | 3 |
| shared_data | 1 |

`purpose=smoke` 仍表示研究用途分类，因此 purpose 分布没有改变。Correction 重新审计
了 path 或文件名含 `smoke` 但 `is_smoke=NO` 的 9 个 YAML；依据 epoch/batch
预算、run/pipeline identity、consumer 与输出用途，9 个均改为 `is_smoke=YES`。
`is_smoke=YES` 现为 25；其中 formal-purpose 8 个、modality-ablation-purpose
1 个同时保留各自研究 purpose。

机器可读的逐 YAML 结果见
`docs/refactors/CONFIG_CLASSIFICATION_PHASE4A.csv`。除任务要求字段外，CSV 还记录
top-level keys、原始 model key / variant、causal flag、split、validation/test
session、seed、parent/base config、pipeline stages、checkpoint、runtime keys、
modality 与 provenance evidence。

## 6. PROVENANCE BOUNDARIES

- `is_official=YES`：0。
- `author_official` implementation：0。
- 18 个 `paper_aligned` 配置全部明确为 non-official。
- 20 个 GS-MCC 配置全部为 `project_variant`，其中 causal 与 full-context
  都没有伪装为作者官方实现。
- `official_prefix` 只描述 validation split；`MMGCN_official` feature 路径只描述
  legacy author-style feature asset，均不证明 model implementation provenance。

## 7. PIPELINE / BENCHMARK DISTRIBUTION

- Pipeline configs：63。
- Benchmark-related configs：43。
- Smoke-purpose configs：16；执行预算/验证用途 `is_smoke=YES`：25。
- Analysis configs：12。
- Ablation configs：66，其中 missing-modality 为 4。
- Formal-marker configs：105。

分类按 YAML 内容、真实 YAML references、registry 与 consumer 完成；没有只凭旧路径
命名判断 lineage。

## 8. REFERENCE GRAPH

扫描范围严格为 Git-tracked `configs/`、`scripts/`、`tests/`、`docs/`、
`AGENTS.md` 与 `README.md`；没有扫描 `outputs/`、`data/`、`third_party/` 或
repository `tmp/`。

- Reference edges：767。
- YAML-source edges：150。
- Script-source edges：28。
- Test-source edges：47。
- Doc-source edges：542。
- Dynamic pattern edges：41。
- Historical-doc edges：458。
- Move 时需要更新的 active edges：302。

动态路径记录 pattern / prefix 与生成逻辑，没有伪造具体文件。历史 audit、notes、
report、review 与 handoff 被标为 historical doc，默认不随 move 改写。两条已失效
的 live example 分别位于 `docs/active_modalities_design.md` 和
`scripts/workflows/run_pipeline.py`；另一个不存在路径是测试中刻意构造的历史
output provenance fixture，不是 live dependency。

完整图见 `docs/refactors/CONFIG_REFERENCE_GRAPH_PHASE4A.csv`。

## 9. ENTRYPOINT AUDIT

逐配置审计结果见 `docs/refactors/CONFIG_ENTRYPOINT_AUDIT_PHASE4A.csv`：

- 183 个 YAML 均有一行。
- 181 个可执行 consumer 指向存在的 canonical entrypoint。
- 180 个 consumer 的 `--config` CLI 与配置用途匹配。
- Original-MERC pipeline manifest 使用 canonical `--manifest`，匹配。
- 2 个 declarative config 无单一 executable consumer，标为 not applicable。
- Missing canonical entrypoint：0。
- 旧 wrapper 不判失败；本计划直接记录 canonical consumer，因此无需在 YAML
  内改写不存在的 entrypoint 字段。
- Registry status：direct/pipeline supported 139、paper-aligned supported 18、
  GS-MCC project-variant supported 20、not applicable 6。

## 10. DUPLICATES

- Byte-exact duplicate groups：0。
- 忽略 comments 与 key order 后的 semantic duplicate groups：0。
- 以最多 3 个 flattened key 差异为阈值记录的 near-duplicate pairs：716。

这些近重复覆盖 session、feature、training budget、context window、modality、
stability parameter 与 output/run identity 变体。Phase 4A 不删除或合并任何
YAML。逐 pair 差异见
`docs/refactors/CONFIG_DUPLICATES_AND_CONFLICTS_PHASE4A.csv`。

## 11. SEMANTIC CONFLICTS AND TARGET COLLISIONS

Correction 后剩余 1 个 candidate collision group，共 2 行 manual review。

前两组原记录是 candidate-path 生成错误，不是需要合并的语义冲突：

1. MultiDAG train 的 context 与 modality 配置分别映射到
   `context_w5_tav.yaml` 和 `modality_w5_tav.yaml`。
2. 对应 pipeline 分别映射到 context 与 modality benchmark tree，并继续引用
   各自 train config。

这 4 行现为 `collision_status=CLEAR`、`manual_review=NO`、`confidence=high`；
formal ablation 风险仍为 `high`。`NEAR-0057` 与 `NEAR-0466` 仍保留为 near
duplicate 证据，但不再标为 candidate collision。

剩余 collision 是两份 missing-modality pipeline。active docs 明确区分 original
formal 与 stable-candidate source run，但 YAML 仅有非机器可读注释和空
`skip_train_use_run_id`，没有可供迁移器验证的 source-run 字段。两行继续使用
`collision_status=MANUAL_REVIEW`、`manual_review=YES`、`risk_level=high`、
`confidence=medium`。必须在 Config Batch 7 前决定合并为一份 canonical config，
或由后续独立任务补足机器可读 source-run 语义；Phase 4A correction 不修改 YAML。

## 12. MIGRATION PLAN

`docs/refactors/CONFIG_MIGRATION_PLAN_PHASE4A.csv` 对 183 个 tracked YAML
各提供且只提供一行 old-path mapping。候选路径：

- 使用 `/` 和 lowercase snake-style。
- 位于 `configs/` 下。
- 不含本地/远程绝对路径。
- model config 使用
  `configs/<model>/<implementation>/<dataset>/<context_mode>/<feature_set>/<purpose>.yaml`。
- cross-model config 使用批准的 `configs/benchmarks/` tree。
- shared data 使用 `configs/_shared/data/`。
- 不创建 `author_official`，不使用含混的 `original`、`official_repro`、
  `new`、`latest`、`final`、`fixed` 或 `best` path component。
- 不规划旧 YAML wrapper，也不规划新旧双份真相。

除 1 个明确 manual-review collision group 外，candidate path 均唯一。

## 13. MIGRATION BATCHES

| Batch | YAML | Active code refs | Test refs | Doc refs | YAML refs | Risk |
|---|---:|---:|---:|---:|---:|---|
| 1 | 16 | 2 | 5 | 30 | 6 | high |
| 2 | 17 | 12 | 5 | 35 | 34 | medium |
| 3 | 17 | 1 | 4 | 16 | 27 | medium |
| 4 | 10 | 1 | 4 | 1 | 12 | medium |
| 5 | 13 | 1 | 9 | 2 | 14 | high |
| 6 | 37 | 1 | 1 | 109 | 27 | high |
| 7 | 73 | 4 | 18 | 285 | 30 | high |

Batch 1 因包含 synthetic GS-MCC project-variant fixture，汇总风险为 high；这不改变
“先迁移 shared data、SimpleMLP、synthetic 与经确认低风险 smoke”的依赖顺序，
但要求保留 provenance gate。每批的具体 gate 与 rollback boundary 见
`docs/refactors/CONFIG_MIGRATION_BATCHES_PHASE4A.csv`。

9 个新确认的 smoke 均为 medium/high risk real-feature、model-coupled、
ablation-coupled 或 pipeline config，因此保持原 Batch 2--7 归属。各 batch YAML
数量与 reference 统计不变，Batch 1 仍为 16。剩余 collision 属于 Batch 7，
不阻塞 Batch 1，但在 Batch 7 前仍是强制停止点。

## 14. BATCH GATES

所有 batch 均必须：

1. 只执行一个 batch。
2. 在 move 前让 strict audit 对更新后的 mapping/reference 状态通过。
3. 同步 active code、tests、active docs 与 YAML references。
4. 运行 targeted consumer/registry tests、完整 pytest、config validation 与
   `git diff --check`。
5. 确认没有旧/新 YAML 双份真相、没有 YAML wrapper、没有 outputs snapshot move。
6. 任一 collision、provenance ambiguity、missing reference 或 gate failure 立即停止。

## 15. READ-ONLY AUDIT TOOL

新增 `scripts/dev/audit_config_migration_plan.py`。它只读 Git tracked YAML、CSV
与 filesystem metadata，不移动或修改配置。Strict mode 检查：

- tracked coverage 与 non-tracked leakage；
- old/candidate uniqueness；
- required fields、approved enums、relative `/` paths 与 target root；
- forbidden ambiguous naming；
- GS-MCC / paper-aligned / author-official provenance；
- collision manual-review marking；
- batch/risk/confidence validity；
- protected roots 与 historical outputs exclusion；
- classification / mapping 一致性和 old-path disk existence。

Targeted synthetic tests 覆盖合法计划、missing tracked YAML、duplicate candidate、
GS-MCC author-official、paper-aligned official、absolute path、forbidden naming、
未完整标记的 collision，以及两行均完整标记
`collision_status=MANUAL_REVIEW` / `manual_review=YES` 时 Phase 4A plan audit
允许通过：correction targeted tests 为 `9 passed`。该 PASS 只表示显式停止点标记
完整，不表示 collision 已解决。

Phase 4A 最终完整 pytest：`356 passed, 3 skipped, 30 warnings`。最终 config
validation、strict migration-plan audit 与 `git diff --check` 均通过。

Phase 4A correction 完整 pytest：`357 passed, 3 skipped, 30 warnings`；config
validation 通过；strict migration-plan audit 输出
`CONFIG_MIGRATION_PLAN_AUDIT=PASS tracked_yaml=183 candidate_collisions=1`；
`git diff --check` 通过。受限沙箱内首次 targeted pytest 只因既有系统 pytest
临时目录权限失败；正常本地权限下同一命令 `9 passed`，不是测试断言失败。

## 16. HISTORICAL OUTPUTS AND PROTECTED PATHS

- `outputs/` 中冻结 `experiment_config.yaml` 不扫描、不移动、不修改。
- `data/`、`third_party/`、repository `tmp/`、checkpoint、logs 与 outputs 未修改。
- 两份 untracked 组会文件未进入 tracked inventory/reference scan，未修改、未 staged。
- 本阶段没有 formal 或 smoke training。

## 17. PERSISTENT CONTEXT

增量更新 `AGENTS.md`、`docs/PROJECT_CONTEXT.md` 与 `docs/project_map.md`：

- `CONFIG_LAYOUT_AUDIT=COMPLETED`
- `CONFIG_LAYOUT_REFACTOR=NOT_STARTED`
- `CONFIG_BATCH_1..7=NOT_STARTED`
- Phase 4A correction：9 个 smoke flag 已修正；剩余 2 行 manual review、
  1 个 candidate collision group
- 下一阶段仅为 Config Batch 1
- provenance、单 batch、no YAML wrapper / no dual truth 与组会文件保护规则

没有把 migration plan 写成已经执行。

## 18. NEXT PHASE

下一阶段为 Config Batch 1，不是 Phase 4B 全量迁移。范围为 `_shared/data`、
SimpleMLP、synthetic 与审核后低风险 smoke；一次只迁移该 batch。Batch 1 完整
门禁通过前，不得开始 Batch 2，也不得接入作者官方 MultiDAG+CL 或 GS-MCC。
剩余 missing-modality collision 属于 Batch 7，不属于 Batch 1；它必须在 Batch 7
前处理，不能在实际 batch 执行中被审计 PASS 静默跳过。

## 19. ROLLBACK AND STOP CONDITIONS

Phase 4A 没有 YAML move，可通过仅撤回本报告列出的 audit CSV、只读工具、测试与
持久上下文增量来回滚。不得使用 `git clean` 或 `git reset --hard`，不得触碰两份
组会 untracked 文件。

后续任一 batch 遇到 dirty tracked tree、未知 untracked、YAML parse failure、
大量职责不确定、未消解 collision、需改模型/生产数学逻辑、需扫描 protected
paths、需正式训练或网络访问时立即停止。

## 20. STATUS

- `CONFIG_LAYOUT_PHASE4A_STATUS=PASS`
- `CONFIG_PHASE4A_CORRECTION_STATUS=PASS`
- `TRACKED_YAML_COUNT=183`
- `CLASSIFIED_YAML_COUNT=183`
- `MANUAL_REVIEW_COUNT=2`
- `CANDIDATE_COLLISION_COUNT=1`
- `SMOKE_MISMATCH_FILES_REVIEWED=9`
- `SMOKE_FLAGS_CHANGED=9`
- `SMOKE_FLAGGED_CONFIG_COUNT=25`
- `BATCH1_YAML_COUNT=16`
- `EXACT_DUPLICATE_GROUPS=0`
- `SEMANTIC_DUPLICATE_GROUPS=0`
- `REFERENCE_EDGE_COUNT=767`
- `MIGRATION_BATCH_COUNT=7`
- `HIGH_RISK_CONFIG_COUNT=28`
- `CONFIG_REFERENCE_GRAPH_CHANGED=NO`
- `CONFIG_ENTRYPOINT_AUDIT_CHANGED=NO`
- `YAML_FILES_MOVED=0`
- `YAML_FILES_MODIFIED=0`
- `MODEL_CODE_CHANGED=NO`
- `SCRIPT_PRODUCTION_CODE_CHANGED=NO`
- `FORMAL_TRAINING_STARTED=0`
- `GROUP_MEETING_FILES_CHANGED=NO`
- `GROUP_MEETING_FILES_STAGED=NO`
- `COMMIT_CREATED=NO`
- `PUSH_PERFORMED=NO`
- `READY_TO_COMMIT_PHASE4A=YES`
- `READY_FOR_CONFIG_BATCH1=YES`
