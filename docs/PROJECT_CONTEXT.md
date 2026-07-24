# MERC 持久项目上下文

本文档是跨 Codex 对话的长期项目记忆。新任务仍先读取 `AGENTS.md`，再读取本文档，并按任务类型补读 `docs/project_map.md` 与相关规范。

## 项目目标与当前阶段

当前研究任务是多模态对话情感识别（MERC），长期目标与机器人实时交互中的 causal MERC 有关。项目当前不是围绕单一 `M3ED + MMGCN` 组合开发，而是在多模型实验平台上建立作者官方复现、统一规程实现、causalization 和最终 baseline 决策的可审计链路。

当前已经完成三类实验基础：

1. 统一 Original / full-context 轨道。
2. 统一 causal 轨道。
3. Legacy 与 Clean 特征轨道。

近期 baseline 决策和正式实验主要围绕 IEMOCAP；M3ED 仍是长期数据资产和机器人代理场景的扩展方向。MMGCN Legacy 复现较稳定，是经典锚点，但尚未确定为最终论文 baseline。

`MODEL_LAYOUT_REFACTOR=COMPLETED`

`SCRIPT_EXECUTION_LAYOUT_REFACTOR=COMPLETED`

`SCRIPT_ANALYSIS_LAYOUT_REFACTOR=COMPLETED`

`SCRIPT_SUPPORT_LAYOUT_REFACTOR=COMPLETED`

`CONFIG_LAYOUT_AUDIT=COMPLETED`

`CONFIG_PHASE4A_CORRECTION_STATUS=PASS`

`CONFIG_LAYOUT_REFACTOR=IN_PROGRESS`

`CONFIG_BATCH_1=COMPLETED`

`CONFIG_BATCH_2=COMPLETED`

`CONFIG_BATCH_3=COMPLETED`

`CONFIG_BATCH_4=COMPLETED`

`CONFIG_BATCH_5=COMPLETED`

`CONFIG_BATCH_6=COMPLETED`

`CONFIG_BATCH_7=NOT_STARTED`

`OFFICIAL_MULTIDAG_REPRODUCTION=NOT_STARTED`

`OFFICIAL_GSMCC_REPRODUCTION=NOT_STARTED`

`FINAL_BASELINE_SELECTED=NO`

模型实现目录的模型优先重构、Phase 3A 生产执行脚本、Phase 3B1 analysis layout 与 Phase 3B2 support layout 均已完成。模型训练/评估入口、跨模型 runtime、workflow、结果分析、数据构建/准备/检查、数据/模型/实验诊断、维护与开发门禁均已有 canonical script tree；旧执行、分析和 support 路径保留 compatibility wrapper。Config Phase 4A audit 与 correction 已完成，183 个 tracked YAML 均已分类并进入精确迁移计划。Config Batch 1--6 已完成；Batch 6 将 2 个 cross-model benchmark 与 35 个 model-scoped ablation 迁移到各自 canonical 路径。37 个 YAML 均 byte-identical 且 semantic-identical，更新 59 个 active references（其中 27 个 Batch 7 YAML references），保留 109 个冻结历史引用，active old references 与未批准语义变化均为 0；Batch 1--6 严格回归均通过。当前剩余 2 行 manual review 和 1 个只属于 Batch 7 的 candidate collision group；下一阶段只执行 Config Batch 7。配置迁移与门禁完成前不部署作者官方 MultiDAG+CL 或 GS-MCC。

## 模型路线与命名边界

- `unified`：项目统一输入、训练和评价接口下的实现。
- `paper_aligned`：项目内依据论文结构实现，不代表作者官方源码。
- `project_variant`：借用论文思想但与作者官方实现存在明确差异。
- `experimental`：尚未进入正式 baseline 池的实验模型。
- 当前 GS-MCC 两套实现的规范显示名均为 `GS-MCC Project Variant`。
- MultiDAG+CL 中的 CL 表示 Curriculum Learning。
- 上下文模式使用 `full_context` 与 `causal_context`；causal 信息边界不等于 `author_official`。
- 作者官方 MultiDAG+CL 与 GS-MCC 尚未接入，不得创建空的 `author_official` 目录暗示已有实现。

## 当前模型与 baseline 状态

当前实验平台包括：

1. MMGCN。
2. MultiDAG-inspired + Curriculum Learning。
3. DialogueGCN。
4. GS-MCC Project Variant。
5. Causal-MMGCN。
6. Causal-MultiDAG-inspired。

当前项目 MultiDAG 是统一或 `paper_aligned` 实现，不是作者官方完整复现。GS-MCC 的 causal 与 full-context 两套实现都是 `project_variant`，也不是作者官方完整复现。MMGCN Legacy 只作为复现较稳定的经典锚点；最终 baseline 必须等官方复现、统一协议表现、causalization 降幅、复现可信度和可修改性完成量化审计后决定。

## Canonical model paths

- 公共 causal 图：`models/common/causal_graph/`
- paper-aligned 公共工具：`models/common/paper_aligned.py`
- causal registry：`models/registry/causal.py`
- paper-aligned registry：`models/registry/paper_aligned.py`
- MMGCN unified：`models/mmgcn/unified/`
- MMGCN paper-aligned：`models/mmgcn/paper_aligned/`
- MultiDAG+CL unified：`models/multidag_cl/unified/`
- MultiDAG+CL paper-aligned：`models/multidag_cl/paper_aligned/`
- DialogueGCN unified：`models/dialoguegcn/unified/`
- DialogueGCN paper-aligned：`models/dialoguegcn/paper_aligned/`
- GS-MCC causal project variant：`models/gsmcc/project_variant/causal/`
- GS-MCC full-context project variant：`models/gsmcc/project_variant/full_context/`
- SimpleMLP：`models/simple_mlp/model.py`
- SDT experimental：`models/experimental/sdt/`

MMGCN unified 的实现文件继续命名为 `mm_gcn.py`，本阶段未改为 `model.py`，以降低整模型 pickle/module-path 兼容风险。

## Config Batch 1 canonical paths

Config Batch 1 已迁移以下 16 个 YAML；旧路径无 wrapper 且已经失效，完整 old/new 映射与 SHA 见 `docs/refactors/CONFIG_BATCH1_MOVES.csv`。

- `configs/_shared/data/iemocap/feature_sets.yaml`
- `configs/benchmarks/causal_unified/analysis/four_model_audit_synthetic.yaml`
- `configs/dialoguegcn/paper_aligned/iemocap/full_context/{clean_roberta_features,legacy_mmgcn_features}/smoke.yaml`
- `configs/dialoguegcn/unified/iemocap/causal_context/legacy_mmgcn_features/smoke_real_2epoch.yaml`
- `configs/dialoguegcn/unified/synthetic/causal_context/synthetic/{audit_fixture,smoke_end_to_end}.yaml`
- `configs/gsmcc/project_variant/synthetic/causal_context/synthetic/{audit_fixture,smoke_end_to_end}.yaml`
- `configs/mmgcn/paper_aligned/iemocap/full_context/{clean_roberta_features,legacy_mmgcn_features}/smoke.yaml`
- `configs/mmgcn/unified/synthetic/full_context/synthetic/smoke.yaml`
- `configs/multidag_cl/paper_aligned/iemocap/full_context/{clean_roberta_features,legacy_mmgcn_features}/smoke.yaml`
- `configs/multidag_cl/unified/synthetic/causal_context/synthetic/smoke.yaml`
- `configs/simple_mlp/unified/m3ed/full_context/m3ed_features/development.yaml`

## Config Batch 2 canonical paths

Config Batch 2 已迁移 17 个 MMGCN YAML；完整 old/new 映射及 SHA 见 `docs/refactors/CONFIG_BATCH2_MOVES.csv`。

- `configs/mmgcn/unified/iemocap/causal_context/legacy_mmgcn_features/{val_official_prefix,val_ses01,val_ses02,val_ses03,val_ses04}.yaml`
- `configs/mmgcn/unified/iemocap/causal_context/clean_roberta_features/{smoke_real_2epoch,val_ses01,val_ses02,val_ses03,val_ses04}.yaml`
- `configs/mmgcn/unified/iemocap/full_context/legacy_mmgcn_features/val_official_prefix.yaml`
- `configs/mmgcn/unified/m3ed/{causal_context,full_context}/m3ed_features/skeleton.yaml`
- `configs/mmgcn/paper_aligned/iemocap/full_context/legacy_mmgcn_features/{screening,fivefold_base}.yaml`
- `configs/mmgcn/paper_aligned/iemocap/full_context/clean_roberta_features/{mmgcn_clean,fivefold_base}.yaml`

Batch 2 before snapshot 由 Git HEAD old paths 重建；4 个 clean-RoBERTa session YAML 只更新 `source_config`，其余 13 个 YAML 在 Git canonical text 表示下 byte-identical 且 semantic-identical。`paper_aligned` 仍不等于 `author_official`。

## Config Batch 3 canonical paths

Config Batch 3 已迁移 17 个 MultiDAG-CL YAML；完整 old/new 映射及 SHA 见 `docs/refactors/CONFIG_BATCH3_MOVES.csv`。

- `configs/multidag_cl/unified/iemocap/causal_context/legacy_mmgcn_features/{val_official_prefix,val_ses01,val_ses02,val_ses03,val_ses04,context_past_all_causal_tav_smoke,context_w5_tav_quick,context_w5_tav_smoke}.yaml`
- `configs/multidag_cl/unified/iemocap/causal_context/clean_roberta_features/{smoke_real_2epoch,val_ses01,val_ses02,val_ses03,val_ses04}.yaml`
- `configs/multidag_cl/paper_aligned/iemocap/full_context/legacy_mmgcn_features/{screening,fivefold_base}.yaml`
- `configs/multidag_cl/paper_aligned/iemocap/full_context/clean_roberta_features/{multidag_cl_clean,fivefold_base}.yaml`

Batch 3 的 13 个 YAML 内容不变，4 个 clean-RoBERTa session YAML 仅更新 `source_config`。引用审计记录 52 个 active references 已更新、16 个历史引用已标记保留、active old references 为 0；语义审计未发现未批准变化。严格 Batch 1/2/3 回归、config validator、相关测试与全量 pytest 均通过。`paper_aligned` 仍不等于 `author_official`，作者官方 MultiDAG 复现尚未开始。

## Config Batch 4 canonical paths

Config Batch 4 已迁移 10 个 DialogueGCN YAML；完整 old/new 映射及 SHA 见 `docs/refactors/CONFIG_BATCH4_MOVES.csv`。

- `configs/dialoguegcn/unified/iemocap/causal_context/legacy_mmgcn_features/{val_official_prefix,val_ses01,val_ses02,val_ses03,val_ses04}.yaml`
- `configs/dialoguegcn/unified/iemocap/causal_context/clean_roberta_features/smoke_real_2epoch.yaml`
- `configs/dialoguegcn/paper_aligned/iemocap/full_context/legacy_mmgcn_features/{screening,fivefold_base}.yaml`
- `configs/dialoguegcn/paper_aligned/iemocap/full_context/clean_roberta_features/{dialoguegcn_clean,fivefold_base}.yaml`

Batch 4 的 10 个 YAML 内容不变，均为 byte-identical 且 semantic-identical。引用审计记录 30 个 active references 已更新，其中 5 个为 pipeline references；历史旧引用和 active old references 均为 0。严格 Batch 1/2/3/4 回归、config validator、相关测试、无训练 dry-run 与完整 pytest 均通过。`paper_aligned` 仍不等于 `author_official`，本批次不是作者官方 DialogueGCN 复现。

## Config Batch 5 canonical paths

Config Batch 5 已迁移 13 个 GS-MCC `project_variant` YAML；完整 old/new 映射及 SHA 见 `docs/refactors/CONFIG_BATCH5_MOVES.csv`。

- `configs/gsmcc/project_variant/iemocap/causal_context/legacy_mmgcn_features/{val_official_prefix,val_ses01,val_ses02,val_ses03,val_ses04,smoke_real_2epoch}.yaml`
- `configs/gsmcc/project_variant/iemocap/causal_context/clean_roberta_features/smoke_real_2epoch.yaml`
- `configs/gsmcc/project_variant/iemocap/full_context/legacy_mmgcn_features/{screening,fivefold_base,smoke}.yaml`
- `configs/gsmcc/project_variant/iemocap/full_context/clean_roberta_features/{gsmcc_clean,fivefold_base,smoke}.yaml`

Batch 5 的 13 个 YAML 内容不变，均为 byte-identical 且 semantic-identical。引用审计记录 46 个 active references 已更新，其中 5 个为 pipeline references；历史旧引用和 active old references 均为 0。causal 配置继续解析为 `causal_gsmcc_inspired`，full-context 配置继续解析为 `project_paper_oriented_gsmcc`；二者都是项目 `project_variant`，不是 `author_official`。作者官方 GS-MCC 复现尚未开始。

## Config Batch 6 canonical paths

Config Batch 6 已按冻结的 Phase 4A plan 迁移 37 个 YAML；完整 old/new 映射及 SHA 见 `docs/refactors/CONFIG_BATCH6_MOVES.csv`。

- 2 个 cross-model benchmark 位于 `configs/benchmarks/causal_unified/iemocap_clean_roberta_eight_run.yaml` 与 `configs/benchmarks/original_merc/pipeline_manifest.yaml`。
- 35 个 model-scoped ablation 位于 MMGCN 或 MultiDAG-CL 的 canonical tree，`benchmark_family=NOT_APPLICABLE`；`NOT_APPLICABLE` 不计为 unknown。
- 唯一 smoke 配置同时保持 `is_smoke=YES` 与 `layout_role=model_scoped_ablation`；smoke 是正交属性。

37 个 YAML 均 byte-identical 且 semantic-identical，模型成员、模型顺序、provenance、ablation variable 与 controlled variables 均无变化。引用审计记录 59 个 active references 已更新，其中 27 个为 Batch 7 YAML 的内部路径；109 个冻结历史引用继续保留，active old references 为 0。Phase 4A plan 与 classification 未修改。作者官方 MultiDAG+CL 和 GS-MCC 复现仍未开始。

## Compatibility paths

以下旧路径至少保留一个迁移周期，只能作为薄 re-export wrapper：

- `models/baselines/causal_graph_common/`
- `models/baselines/causal_baseline_registry.py`
- `models/baselines/{mmgcn,multidag_cl,dialoguegcn,gsmcc}/`
- `models/baselines/original_repro/` 及其模型子包、`common.py`、`registry.py`
- `models/baselines/simple_mlp.py`
- `models/baselines/sdt/`

canonical execution scripts 已直接使用模型 canonical path，不再依赖 `models.baselines...`。既有模型兼容测试与部分 Phase 3B deferred scripts 暂时保留旧 model wrapper import；新代码必须直接使用 canonical path。

## Model registries

- 新 causal baseline 构造与名称规范化：`models.registry.causal`
- paper-aligned / project-variant MERC 模型构造与 provenance：`models.registry.paper_aligned`
- 旧 registry 模块只重导出同一函数和对象，不维护第二套 registry。

## 数据、特征与正式实验

- 近期正式实验与 baseline 决策主要使用 IEMOCAP。
- M3ED 是长期数据资产，并服务于后续机器人代理与 causal MERC 场景扩展。
- 特征比较同时保留 Legacy 与 Clean 轨道，不得跨轨道混写结果。
- 统一 batch 形状与模型插入约束见 `docs/module_implementation_spec.md`。
- Original MERC 三轨协议、缺失模态与从头训练消融边界见 `docs/experiment_protocol.md`。
- validation 选择 best checkpoint，test 不参与 checkpoint、epoch、超参数或模型选择。
- 正式结果至少报告 Weighted-F1、Macro-F1、UAR、Accuracy。

当前正式实验批次为 `causal8_original8_formal_20260722_012904`：包含 8 个 causal run 和 8 个 Original run，共 16 个正式实验；全部 `PASS`，数值状态均为 `FINITE`。该批次证明统一运行链路已完成并通过数值门禁，不等于作者官方复现完成，也不等于最终 baseline 已选定。

## Canonical runtime entries 与输出规则

Phase 3A canonical execution entries 为：

- MMGCN 训练：`scripts/models/mmgcn/unified/train.py`
- MultiDAG+CL 训练/评估/smoke：`scripts/models/multidag_cl/unified/{train,evaluate,smoke}.py`
- SimpleMLP 训练：`scripts/models/simple_mlp/train.py`
- 统一 checkpoint 评估：`scripts/evaluation/unified_checkpoint.py`
- causal 与 paper-aligned runtime：`scripts/runtime/{causal_graph,paper_aligned}.py`
- 通用 pipeline：`scripts/workflows/run_pipeline.py`
- causal workflow：`scripts/workflows/causal_graph/{train,evaluate}.py`
- paper-aligned workflow：`scripts/workflows/paper_aligned/`
- 正式 16-run benchmark：`scripts/workflows/benchmarks/run_causal8_original8.py`
- 模态缺失/消融 workflow：`scripts/workflows/ablations/`

Phase 3B1 canonical analysis paths 为：

- 通用结果表格、绘图与论文产物：`scripts/analysis/common/`
- causal 模型与 benchmark 分析：`scripts/analysis/causal/`
- Original MERC / paper-aligned 分析：`scripts/analysis/paper_aligned/`
- 模型专属分析：`scripts/analysis/models/<model>/`

Phase 3B2 canonical support paths 为：

- 数据/特征构建：`scripts/data/build/`
- 训练前数据准备：`scripts/data/prepare/`
- 只读数据检查：`scripts/data/inspect/`
- 数据诊断：`scripts/diagnostics/data/`
- 模型诊断：`scripts/diagnostics/models/`
- 实验与 run 诊断：`scripts/diagnostics/experiments/`
- 仓库与汇总维护：`scripts/maintenance/`
- 开发验证门禁：`scripts/dev/`

迁移前的 tracked debug、diagnose、features、prepare、inspect、deferred
`scripts/analyze/`、`scripts/dev/diagnose_gsmcc_numerics.py` 与
`scripts/maintenance/check_env.py` 路径继续作为一跳 compatibility wrapper；
新代码不得 import 这些旧 support 路径。`scripts/maintenance/` 中的 evaluation
与 experiment summary 工具以及 `scripts/dev/validate_config_tree.py` 保持原位。

迁移前的 tracked `scripts/analyze/` 分析入口继续作为 module-alias compatibility wrapper；`scripts/analysis/analyze_original_merc_results.py` 与 `scripts/analyze/export_original_merc_reproduction_report.py` 也直接委派最终 canonical Original-MERC module。新代码不得 import 这些旧 analysis wrapper。

迁移前路径继续作为 compatibility wrapper，可保持既有 CLI 与尚未迁移的 YAML 可运行；新代码不得 import 旧 script wrapper。

正式实验输出按启动日期写入 `outputs/<YYYYMMDD>/`；本地 smoke 使用 `tmp/`。显式 checkpoint/run/output 路径优先，旧输出只读兼容，不移动、不改名。不要提交数据、feature pkl、checkpoint、cache 或大体积输出。

## 已完成工作

- 从源分支 `feat/date-organized-outputs` 的 `e09e8864e3cb4fa364c185f6ca1eb342f971283b` 创建工作分支 `refactor/model-first-layout`。
- 通过 14 组 `git mv` 将 32 个已跟踪模型文件迁移到 canonical 路径。
- 在旧路径创建 32 个薄 wrapper，并将 `models/baselines/original_repro/__init__.py` 转为兼容入口。
- canonical 实现的内部 import 已切换到 `models.common`、`models.registry` 和各模型 canonical package。
- 新增 `tests/test_model_layout_compatibility.py`，验证 10 组核心新旧符号身份及 MMGCN strict state-dict 加载。
- 模型数学、forward 契约、默认超参数和 state-dict schema 未改变。
- Phase 3A 使用 19 个 `git mv` 建立 canonical execution tree，并在全部旧路径创建 19 个透明 module-alias wrapper。
- canonical scripts 的模型 import 已迁移到 `models.registry`、`models.common` 与模型 canonical package；canonical `models.baselines` import 为 0。
- pipeline、missing-modality workflow 与正式 16-run launcher 的内部 subprocess target 已指向 canonical scripts；`configs/` 内容与 runtime/checkpoint key 未改。
- 新增 `tests/test_script_layout_compatibility.py` 与 Phase 3A legacy import CSV/Markdown 审计。
- Phase 3B1 使用 13 个 `git mv` 将真实分析实现迁入 `scripts/analysis/`，并在 13 个旧 tracked 路径建立薄 module-alias wrapper；另将 2 个既有分析别名直接指向最终 canonical module。
- canonical analysis 的旧 model/script/analysis wrapper import 为 0；5 个 feature/data 或 diagnose 文件保留原位，等待 Phase 3B2。
- 新增 `tests/test_analysis_layout_compatibility.py`，覆盖新旧 import identity、CLI help、repo-root 解析和 synthetic missing-modality 输出等价。
- Phase 3B2 对 32 个 support 候选完成源码职责分类，以 29 个 `git mv` 建立 `scripts/data/`、`scripts/diagnostics/` 与规范化的 `scripts/dev/`，并在全部旧 tracked 路径保留 29 个一跳 wrapper；2 个 maintenance summary 工具和 `validate_config_tree.py` 保持原位。
- canonical support 与全部 tracked 生产 scripts 的旧 model import 均为 0；仅 `tests/test_model_layout_compatibility.py` 保留 10 条专用兼容 import，manual-review 文件为 0。
- 新增 `tests/test_support_script_layout_compatibility.py`，覆盖 29 组 import/委派、13 组 CLI help、repo-root、synthetic feature build、诊断 PASS/FAIL、只读 inspect 与危险工具不执行。

## 最近门禁结果

执行日期：2026-07-23，本地 Windows，conda 环境 `m3ed_mmgcn`。

- 迁移前完整 pytest：`176 passed, 3 skipped`。
- 迁移前 config validation：通过。
- 新旧 import 与 state-dict 专项：`11 passed`。
- 模型相关 synthetic / numerical / checkpoint 测试：`93 passed, 3 skipped`。
- 迁移后完整 pytest：`187 passed, 3 skipped`。
- 迁移后 config validation：通过。
- canonical `models/` 内 `models.baselines` 静态引用：0。
- 最终工作区与 staged `git diff --check`：通过。
- Script Phase 3A 执行前完整 pytest：`187 passed, 3 skipped`。
- Script layout compatibility：`52 passed`。
- Script runtime/checkpoint/formal-launcher/output-path 专项：`93 passed, 3 skipped`。
- Script Phase 3A 完整 pytest：`239 passed, 3 skipped`。
- canonical execution legacy model imports：0；compatibility wrapper legacy model imports：0。
- deferred scripts：10 个文件、12 条 legacy model import；tests：13 个文件、27 条 legacy model import。
- Phase 3A 基于分支 `refactor/model-first-layout`、HEAD `1988757fffccfd50532ed4a20d15fedb017eb639`；本阶段未 commit、未 push。
- Script Phase 3B1 执行前完整 pytest：`239 passed, 3 skipped`；config validation 与 diff check 通过。
- Analysis layout compatibility：`31 passed`；既有 analysis consumer 专项：`41 passed`。
- Script Phase 3B1 执行后完整 pytest：`270 passed, 3 skipped`；config validation 与 diff check 通过。
- canonical analysis legacy imports：0；compatibility wrappers：15；deferred analysis files：5；tests 中 legacy analysis import：5 个文件、6 条边；旧 CLI path 文本引用：139。
- Phase 3B1 基于分支 `refactor/model-first-layout`、HEAD `46af77a64b1a7347c46904a9710692a31576bf2f`；本阶段未 commit、未 push。
- Script Phase 3B2 执行前完整 pytest：`270 passed, 3 skipped`；config validation 与 diff check 通过。
- Support layout compatibility：`78 passed`；受 canonical model import 更新影响的既有 tests：`103 passed, 3 skipped`。
- Script Phase 3B2 执行后完整 pytest：`348 passed, 3 skipped`；config validation 与 diff check 通过。
- canonical support legacy model imports：0；tracked production script legacy model imports：0；剩余 10 条 test import 全部属于专用 model-layout compatibility test。
- Phase 3B2 基于分支 `refactor/model-first-layout`、HEAD `63dfe55fca2e3883af7736aeb081423b147b3d26`；本阶段未 commit、未 push。
- Config Phase 4A 基于分支 `refactor/model-first-layout`、HEAD `4f97adf513882d90457c17e49c622ad0682413f2`；执行前完整 pytest 为 `348 passed, 3 skipped`，config validation 与 diff check 通过。
- Phase 4A 盘点 183 个 tracked YAML，解析失败 0；分类 183，引用边 767，exact duplicate groups 0，semantic duplicate groups 0，近重复对 716。
- Phase 4A correction 将两组错误 candidate-path collision 拆成独立 context/modality path；当前迁移计划包含 7 个 batch、1 个 candidate collision group、2 行 manual review、28 个 high-risk config。
- Phase 4A correction 审计了 9 个 smoke-name mismatch，全部按 YAML 预算、identity、consumer 与用途改为 `is_smoke=YES`；`purpose` 与 Batch 1--7 数量不变，Batch 1 仍为 16。
- Phase 4A 审计工具新增完整 manual-review collision PASS 测试；plan audit PASS 只表示停止点标记完整，不表示 collision 已解决。
- Phase 4A 最终完整 pytest：`356 passed, 3 skipped`；最终 config validation、strict migration-plan audit 与 diff check 通过。
- Phase 4A correction targeted audit tests：`9 passed`；完整 pytest：`357 passed, 3 skipped`；config validation、strict audit 与 `git diff --check` 通过；strict audit 报告 `tracked_yaml=183 candidate_collisions=1`。
- Config Batch 1 基于分支 `refactor/model-first-layout`、HEAD `870ac7dd5d58b6e0a1f4012cf10335db836b2f1c`；迁移前完整 pytest 为 `357 passed, 3 skipped`，迁移后为 `369 passed, 3 skipped`。
- Batch 1 使用 16 个 `git mv`；旧路径剩余 0、新路径存在 16、tracked YAML 总数 183、YAML 内容变化 0、未批准语义变化 0。
- Batch 1 reference audit 记录 28 条已更新活动引用、20 条保留历史引用，以及 1 条仍只指向未迁移 GS-MCC smoke 的获准活动目录模式；活动代码、测试和 YAML 中 Batch 1 旧文件路径剩余 0。
- Batch 1 targeted consumer/audit tests：`70 passed`；新增 Batch audit synthetic tests：`10 passed`；Phase 4A plan audit tests：`10 passed`。
- Batch 1 config validation、Phase 4A strict plan audit、Batch 1 strict audit、Original-MERC smoke command-generation dry plan、6 个 CLI help 与 `git diff --check` 均通过。
- Phase 4A plan audit 已保持向后兼容并支持迁移后的 candidate path 单一真相；Batch strict audit 发现并修正了 `utils/iemocap_features.py` 的旧 feature-registry 活动路径漏项。
- Config Batch 2 基于分支 `refactor/model-first-layout`、Git HEAD `3e9cafbbae409b08c4f1607dae93072648a80990` 完成 post-move audit recovery；没有回滚或重新执行迁移。
- Batch 2 使用既有 17 个 `git mv`：old paths 剩余 0、candidate paths 存在 17、tracked YAML 总数 183；13 个 `unified`、4 个 `paper_aligned`。
- Git HEAD before snapshot 对比当前 candidate YAML 得到 13 个内容不变、4 个仅 `source_config` 更新、未批准语义变化 0；`source_config` target 必须是同批次精确 mapping 且保持 provenance identity。
- Batch 2 reference audit 记录 90 条已更新活动引用与 20 条保留历史引用；active old references 为 0。两个 YAML-to-YAML 历史引用由 Phase 4A reference graph 冻结，因 Batch 1/Batch 7 YAML 不属于本批次编辑范围而保留。
- Batch 2 audit tests：`26 passed`；Phase 4A plan audit tests：`10 passed`；MMGCN/config/pipeline/registry/checkpoint 相关回归：`266 passed, 3 skipped`。
- Batch 2 config validation、Phase 4A strict plan audit、Batch 1 strict regression、Batch 2 strict audit、无训练 dry-run 与 `git diff --check` 均通过；完整 pytest：`385 passed, 3 skipped`。
- Config Batch 3 基于分支 `refactor/model-first-layout`、Git HEAD `753eae191558c4ccad5c69e24a8b960c20691784` 完成；17 个 MultiDAG-CL YAML 已迁移，52 个 active references 已更新，完整 pytest：`399 passed, 3 skipped`。
- Config Batch 4 基于分支 `refactor/model-first-layout`、Git HEAD `0e1a62a84fb464632990552d8cb2cfd32ea5b055` 完成；10 个 DialogueGCN YAML 使用 `git mv` 迁移，old paths 剩余 0、new paths 存在 10、tracked YAML 总数 183。
- Batch 4 的 6 个 `unified` 与 4 个 `paper_aligned` YAML 均 byte-identical 且 semantic-identical；YAML 内容变化 0、未批准语义变化 0。
- Batch 4 reference audit 记录 30 个已更新 active references，其中 5 个 pipeline references；历史旧引用与 active old references 均为 0。
- Batch 4 audit/plan/benchmark focused tests：`64 passed`；DialogueGCN/config/pipeline/registry/checkpoint 相关回归：`181 passed, 3 skipped`；完整 tracked pytest（显式排除受保护的 untracked 组会测试）：`403 passed, 3 skipped`。
- Batch 4 config validation、Phase 4A strict plan audit、Batch 1/2/3/4 strict audit、10-config 无训练 dry-run、4 个 CLI help 与 `git diff --check` 均通过。
- Config Batch 5 基于分支 `refactor/model-first-layout`、Git HEAD `2403a801b48307f214f2fdd6768d206e40c453e9` 完成；13 个 GS-MCC `project_variant` YAML 使用 `git mv` 迁移，old paths 剩余 0、new paths 存在 13、tracked YAML 总数 183。
- Batch 5 的 7 个 `causal_context` 与 6 个 `full_context` YAML 均 byte-identical 且 semantic-identical；YAML 内容变化 0、未批准语义变化 0。
- Batch 5 reference audit 记录 46 个已更新 active references，其中 5 个 pipeline references；历史旧引用与 active old references 均为 0。
- Batch 5 相关定向回归：`219 passed, 3 skipped`；13-config 无训练 dry-run 解析配置、registry、entrypoint、output/checkpoint 与 pipeline/manifest 引用均通过。
- Batch 5 完整 tracked pytest（显式排除受保护的 untracked 组会测试）：`420 passed, 3 skipped, 14 warnings`；config validator、Phase 4A strict plan audit、Batch 1/2/3/4/5 strict audit、CLI help 与 `git diff --check` 均通过。
- Config Batch 6 基于分支 `refactor/model-first-layout`、Git HEAD `4700a8b495682c9df90c7319847f9331ddfcbceb` 完成；37 个 YAML 使用 `git mv` 迁移，old paths 剩余 0、new paths 存在 37、tracked YAML 总数 183。
- Batch 6 的 2 个 cross-model benchmark 与 35 个 model-scoped ablation 均 byte-identical 且 semantic-identical；YAML 内容变化、未批准语义变化、模型成员/顺序/provenance 变化、ablation/controlled-variable 变化均为 0。
- Batch 6 reference audit 记录 59 个已更新 active references，其中 27 个为未移动的 Batch 7 YAML 内部路径；109 个冻结历史引用保留，active old references 为 0。
- Batch 6 audit tests：`79 passed`；Phase 4A plan audit tests：`10 passed`；完整 tracked pytest（显式排除受保护的 untracked 组会测试）：`430 passed, 3 skipped, 14 warnings`。
- Batch 6 config validation、Phase 4A strict plan audit、Batch 1/2/3/4/5/6 strict audit、37-config 无训练 dry-run 与 `git diff --check` 均通过；正式训练启动次数为 0。

完整 pytest 在受限沙箱内会因既有 `tmp/pytest_*` 与系统 pytest 临时目录权限而无法收集；在具有正常本地文件权限的同一环境中通过。这不是模型断言失败，且本任务未删除或修改这些目录。

## 已知问题与保护边界

- Phase 3A production execution、Phase 3B1 analysis 与 Phase 3B2 support scripts 已迁移；全部 scripts layout refactor 已完成。
- 旧 script wrapper 与旧 model wrapper 至少保留一个迁移周期；删除前必须完成 configs 迁移和独立删除门禁。
- `prepare_m3ed_metadata.py`、真实 batch/model 诊断与 summary rebuild 工具具有数据或资产写入风险；Phase 3B2 只做 import、CLI help、synthetic 或静态安全检查，没有执行这些真实动作。
- `scripts/analyze/export_group_meeting_baseline_report.py` 和 `tests/analyze/test_group_meeting_baseline_report.py` 是本地 untracked 组会文件，必须保持 not staged，不修改、不上传。
- 不扫描或修改 `outputs/`、`data/`、`third_party/`、`tmp/`；不启动本地正式训练。
- Phase 4A 本身没有移动、重命名或修改 YAML；随后完成的 Config Batch 1--6 严格按迁移计划执行。历史 outputs 内冻结的配置快照不在迁移范围。
- 当前只剩 1 个 candidate collision group，涉及两份 missing-modality pipeline。active docs 区分 original formal 与 stable-candidate source run，但 YAML 没有机器可读 source-run 字段；必须在执行 Config Batch 7 前决定合并为一份 canonical 配置，或由后续独立任务补足机器可读语义。该 collision 不属于 Batch 1--6。
- 配置 provenance 不得按旧路径或文件名猜测；`paper_aligned` 不是 `author_official`，全部 GS-MCC 配置均为 `project_variant`。

已知论文叙事风险：

1. 不得将 `paper_aligned` 写成 `author_official`，也不得把当前 MultiDAG 或 GS-MCC 项目实现写成作者官方完整复现。
2. 配置中的 `original_repro_*` 等 key 是运行协议兼容标识，不足以证明实现 provenance。
3. 不得因 MMGCN Legacy 复现稳定就提前声明其为最终论文 baseline。
4. full-context 与 causal 比较必须控制除未来信息相关内部结构以外的变量；test 结果不得用于模型选择。
5. 当前统一规程结果、未来作者官方结果及其 causalized 结果必须分轨报告，不得混写成同一复现结论。

## 下一阶段工程顺序

模型目录重构与 Script Phase 3A、3B1、3B2 已完成，Config Phase 4A audit/correction 与 Config Batch 1--6 也已完成。下一阶段只执行 `CONFIG_BATCH_7`。每次只执行一个 batch；执行 Batch 7 前必须先解决剩余 collision。完整 config migration 与门禁完成后，才部署作者官方 MultiDAG+CL 和 GS-MCC。

Phase 3B2 结果见 `docs/refactors/SCRIPT_SUPPORT_LAYOUT_REFACTOR_REPORT.md`、`docs/refactors/SCRIPT_SUPPORT_CLASSIFICATION_PHASE3B2.csv` 与 `docs/refactors/LEGACY_MODEL_IMPORTS_AFTER_SCRIPT_PHASE3B2.*`。后续 config 重构不得删除 Phase 3A/3B1/3B2 wrapper、改变 runtime/checkpoint schema，或把 `paper_aligned` 误写为 `author_official`。

Config Phase 4A 的唯一迁移依据为：

- `docs/refactors/CONFIG_CLASSIFICATION_PHASE4A.csv`
- `docs/refactors/CONFIG_REFERENCE_GRAPH_PHASE4A.csv`
- `docs/refactors/CONFIG_ENTRYPOINT_AUDIT_PHASE4A.csv`
- `docs/refactors/CONFIG_DUPLICATES_AND_CONFLICTS_PHASE4A.csv`
- `docs/refactors/CONFIG_MIGRATION_PLAN_PHASE4A.csv`
- `docs/refactors/CONFIG_MIGRATION_BATCHES_PHASE4A.csv`
- `docs/refactors/CONFIG_LAYOUT_PHASE4A_REPORT.md`

Config Batch 1 的执行与审计依据为：

- `docs/refactors/CONFIG_BATCH1_MOVES.csv`
- `docs/refactors/CONFIG_BATCH1_BEFORE_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH1_AFTER_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH1_SEMANTIC_DIFF.csv`
- `docs/refactors/CONFIG_BATCH1_REFERENCE_AUDIT.csv`
- `docs/refactors/CONFIG_BATCH1_REFACTOR_REPORT.md`
- `scripts/dev/audit_config_batch_migration.py`

Config Batch 2 的执行与审计依据为：

- `docs/refactors/CONFIG_BATCH2_MOVES.csv`
- `docs/refactors/CONFIG_BATCH2_BEFORE_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH2_AFTER_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH2_SEMANTIC_DIFF.csv`
- `docs/refactors/CONFIG_BATCH2_REFERENCE_AUDIT.csv`
- `docs/refactors/CONFIG_BATCH2_REFACTOR_REPORT.md`

Config Batch 3 的执行与审计依据为：

- `docs/refactors/CONFIG_BATCH3_MOVES.csv`
- `docs/refactors/CONFIG_BATCH3_BEFORE_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH3_AFTER_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH3_SEMANTIC_DIFF.csv`
- `docs/refactors/CONFIG_BATCH3_REFERENCE_AUDIT.csv`
- `docs/refactors/CONFIG_BATCH3_REFACTOR_REPORT.md`
- `scripts/dev/audit_config_batch_migration.py`

Config Batch 4 的执行与审计依据为：

- `docs/refactors/CONFIG_BATCH4_MOVES.csv`
- `docs/refactors/CONFIG_BATCH4_BEFORE_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH4_AFTER_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH4_SEMANTIC_DIFF.csv`
- `docs/refactors/CONFIG_BATCH4_REFERENCE_AUDIT.csv`
- `docs/refactors/CONFIG_BATCH4_REFACTOR_REPORT.md`
- `scripts/dev/audit_config_batch_migration.py`

Config Batch 5 的执行与审计依据为：

- `docs/refactors/CONFIG_BATCH5_MOVES.csv`
- `docs/refactors/CONFIG_BATCH5_BEFORE_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH5_AFTER_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH5_SEMANTIC_DIFF.csv`
- `docs/refactors/CONFIG_BATCH5_REFERENCE_AUDIT.csv`
- `docs/refactors/CONFIG_BATCH5_REFACTOR_REPORT.md`
- `scripts/dev/audit_config_batch_migration.py`

Config Batch 6 的执行与审计依据为：

- `docs/refactors/CONFIG_BATCH6_MOVES.csv`
- `docs/refactors/CONFIG_BATCH6_BEFORE_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH6_AFTER_SNAPSHOT.json`
- `docs/refactors/CONFIG_BATCH6_SEMANTIC_DIFF.csv`
- `docs/refactors/CONFIG_BATCH6_REFERENCE_AUDIT.csv`
- `docs/refactors/CONFIG_BATCH6_MODEL_MEMBERSHIP_AUDIT.csv`
- `docs/refactors/CONFIG_BATCH6_REFACTOR_REPORT.md`
- `scripts/dev/audit_config_batch_migration.py`

## 下一阶段研究路线

1. 使用作者官方源码、官方特征和作者规程复现 MultiDAG+CL。
2. 使用作者官方源码、官方特征和作者规程复现 GS-MCC。
3. 在官方 full-context 实现基础上，仅 causal 化未来信息相关的内部结构，保留其他变量不变。
4. 全量审计作者官方 full-context、作者官方 causalized、当前统一 full-context、当前统一 causal 四条轨道。
5. 根据 causal 化下降、统一协议表现、复现可信度和可修改性决定最终 baseline。
6. 再决定论文主线是通用 MERC 还是 causal MERC，并进入模块实验。
