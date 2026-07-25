# Codex 项目上下文入口

本仓库是一个多模型多模态对话情感识别（MERC）实验平台，长期目标与机器人实时交互中的 causal MERC 有关。当前阶段正在建立作者官方复现、统一实验规程、causalization 和最终 baseline 决策；`MMGCN` 是经典锚点，但尚未确定为最终论文 baseline。后续 Codex 对话进入本仓库时，先按本文档建立上下文，不要默认全量扫描仓库。

## 任务开始规则

1. 任何任务开始前，先读取 `docs/project_map.md`。
2. 任务涉及 Git、本地/远程同步、实验执行时，同时读取 `docs/codex_workflow.md` 和 `docs/git_workflow.md`。
3. 任务涉及实验设计、缺失模态、模态消融、结果分析时，同时读取 `docs/experiment_protocol.md`。
4. 任务涉及模型结构或 `MMGCN` 模块修改时，同时读取 `docs/module_implementation_spec.md`。
5. 不要全量扫描仓库，除非 `docs/project_map.md` 不足以定位任务文件。
6. 修改前列出计划查看和修改的文件。
7. 修改后显示 `git diff --stat`。
8. 除非用户明确要求，不要执行 `git commit`。

## 路径与数据规则

1. 不要硬编码本地或远程绝对路径。
2. 配置、脚本和文档中优先使用相对路径。
3. 不要随意修改 `third_party/`、checkpoints、原始数据、大体积 `outputs`、cache 文件。
4. 不要修改 `data/`、`checkpoint/`、`checkpoints/`、`logs/`、`outputs/`，除非用户明确要求并说明用途。
5. 本地 Windows 主要用于代码编辑、轻量 smoke test、CSV 分析、图表生成；不要在本地跑正式训练。
6. 正式训练、多 seed 实验和 checkpoint 评估主要在远程 V100 服务器上执行。

## 修改原则

1. 修改必须小而可审查。
2. 每次优先只解决一个问题。
3. 新模块必须提供 YAML 开关，便于消融。
4. 新模块必须保留 baseline-equivalent 路径，也就是关闭模块后行为应尽量等价于原 baseline。
5. 优先添加 smoke test 或最小可运行检查，再讨论模型效果优化。
6. 不要直接安装依赖；如需依赖，建议加入 `requirements.txt` 或 `environment.yml`，但当前 checkout 未找到这两个文件。

## 当前模型目录与阶段

1. 模型实现已迁移到模型优先的 canonical 路径：`models/<model>/<lineage>/`；公共代码位于 `models/common/`，registry 位于 `models/registry/`。
2. 上一轮重构的 `models/baselines/` 兼容 tree 已退休；旧模型 import 路径不再受支持，只能使用 canonical model path。
3. `paper_aligned` 表示项目内按论文结构实现，不等于 `author_official`；当前 MultiDAG+CL 项目实现不是作者官方完整复现，GS-MCC 两套实现均为 `project_variant`。
4. `MMGCN` 是当前经典复现锚点，不得表述为已经选定的最终 baseline；SDT 位于 `models/experimental/sdt/`，仍不进入正式 baseline 排名。
5. 目录重构不得改变模型数学行为、forward 契约或 `state_dict` schema。
6. 模型目录、scripts layout、config layout 与 legacy wrapper 退休均已完成；`CANONICAL_ONLY_LAYOUT=COMPLETED`、`CONFIG_LAYOUT_AUDIT=COMPLETED`、`CONFIG_LAYOUT_REFACTOR=COMPLETED`。Config Batch 1--7 全部完成，183 个 tracked YAML 已进入 canonical tree，active old references、manual review、candidate collision 与未批准语义变化均为 0。
7. 作者官方 MultiDAG+CL 与 GS-MCC 复现仍未开始；后续部署必须作为独立任务执行，不得把现有 `paper_aligned` 或 `project_variant` 误写为 `author_official`。
8. 不要为定位模型代码扫描 `outputs/`、`data/`、`third_party/` 或 `tmp/`。

## 当前脚本目录与阶段

1. 生产执行脚本的 canonical 路径位于 `scripts/models/`、`scripts/runtime/`、`scripts/evaluation/` 和 `scripts/workflows/`。
2. 迁移前的训练、评估、pipeline 与实验 launcher wrapper 已退休；旧脚本执行命令不再受支持。
3. Config Batch 1--7 已迁移；生产配置、脚本、测试与活动文档必须使用 canonical config path。
4. Phase 3B1 analysis layout 已完成；canonical 分析实现位于 `scripts/analysis/{common,causal,paper_aligned,models}/`。
5. 被迁移的 tracked `scripts/analyze/` 分析 wrapper 已退休；分析必须使用 `scripts/analysis/` 下的 canonical 入口。
6. Phase 3B2 support layout 已完成；canonical 支持脚本位于 `scripts/data/{build,prepare,inspect}/`、`scripts/diagnostics/{data,models,experiments}/`、`scripts/maintenance/` 与 `scripts/dev/`。
7. `data/build` 生成数据或特征资产，`data/prepare` 做训练前数据准备，`data/inspect` 只读查看数据；`diagnostics` 判定数据、模型或实验异常；`maintenance` 可更新仓库/汇总资产，`dev` 只放静态验证与开发门禁。
8. 迁移前的 debug、diagnose、features、prepare、inspect 及 deferred analyze/support wrapper 已退休；支持脚本必须使用 canonical 路径。
9. 所有 tracked 生产 scripts 与 tests 已切换到 canonical model/script import；专用兼容性测试已由 canonical-only retirement gate 取代。
10. `CONFIG_LAYOUT_REFACTOR=COMPLETED`；`CONFIG_BATCH_1=COMPLETED`、`CONFIG_BATCH_2=COMPLETED`、`CONFIG_BATCH_3=COMPLETED`、`CONFIG_BATCH_4=COMPLETED`、`CONFIG_BATCH_5=COMPLETED`、`CONFIG_BATCH_6=COMPLETED`、`CONFIG_BATCH_7=COMPLETED`。
11. `scripts/analyze/export_group_meeting_baseline_report.py` 与 `tests/analyze/test_group_meeting_baseline_report.py` 继续作为本地 untracked 组会文件保护，不读取其内容作为生产依据、不修改、不 staged。
12. 配置 provenance 必须依据 YAML 内容、registry 与真实 consumer，不得根据旧文件名中的 `original`、`official` 等字样猜测。
13. YAML 迁移不保留双份真相，也不创建旧 YAML wrapper；发生候选路径碰撞时必须先人工消解。
14. Phase 4A 配置迁移已经结束；不得在没有新计划与独立审计的情况下继续移动 canonical YAML。
15. Batch 7 的 missing-modality collision 已按普通 context 与 stable context 的不同 `source_config` 语义拆分为两个唯一 canonical path，两份配置均保留。

## 当前 wrapper 退休状态

1. `LEGACY_MODEL_WRAPPERS=RETIRED`
2. `LEGACY_SCRIPT_WRAPPERS=RETIRED`
3. `CANONICAL_ONLY_LAYOUT=COMPLETED`
4. `FORMAL_LONG_TRAINING_STARTED=NO`
5. `LONG32_TRACK_SPLIT_REPAIR=COMPLETED`
6. 2026-07-25 远程批次 `outputs/launcher_logs/formal_long32_20260725_213652` 在首个 run `ltp_mmgcn_full_context_val_ses01_s42` 的 runtime config validation 阶段失败，未进入 epoch；根因是 full-context base 的 `clean_roberta_fivefold_fair_comparison` track 被矩阵仅覆盖 split 为 `session_holdout`，形成非法组合。
7. 32-run 长训练现统一使用 `protocol_version=long_training_session_holdout_v1` 与 `clean_roberta_session_holdout_fair_comparison -> session_holdout`，固定 Ses05 Test、轮换 Ses01--Ses04 Validation；4 个模型、16 个 full-context、16 个 causal-context 与 16 个 pair key 不变，test 不参与选择。
8. 旧模型 import 路径和旧脚本执行命令均不再受支持；后续代码、配置、测试、文档与命令必须使用 canonical 路径。
9. Wrapper 退休没有修改模型数学、forward、loss、optimizer、训练逻辑、数据划分、checkpoint/state-dict schema 或配置语义；tracked YAML 数量仍为 183。
10. Long32 修复回归在临时目录生成 32 个 resolved config 并按各自实际 entrypoint 完成 32/32 config-only runtime validation；定向回归 `15 passed`，完整 pytest `321 passed, 3 skipped`，正式训练启动次数仍为 0。

## 当前配置目录与阶段

1. Config Batch 1 已完成：16 个 canonical 路径及 old/new 映射以 `docs/refactors/CONFIG_BATCH1_MOVES.csv` 为准，覆盖 `_shared/data`、SimpleMLP、synthetic、Original-MERC smoke 与低风险 smoke。
2. Batch 1 旧 YAML 路径已失效；新代码、测试和活动文档只能引用 canonical 路径。
3. 不存在旧 YAML wrapper、redirect YAML、symlink 或新旧双份配置真相。
4. Config Batch 2 已完成：17 个 MMGCN YAML 均在 `configs/mmgcn/{unified,paper_aligned}/` 下；13 个内容不变，4 个仅将 `source_config` 更新到同批次精确 canonical target。`paper_aligned` 不等于 `author_official`。
5. Batch 2 old/new 映射、Git HEAD before snapshot、当前 after snapshot、语义差异、引用审计和门禁结果以 `docs/refactors/CONFIG_BATCH2_*` 为准；active old references 为 0。
6. Config Batch 3 已完成：17 个 MultiDAG-CL YAML 均位于 `configs/multidag_cl/{unified,paper_aligned}/` 下；13 个内容不变，4 个仅更新同批次精确 target 的 `source_config`。共更新 52 个 active references（其中 pipeline references 12 个），保留并标记 16 个历史引用，active old references 与未批准语义变化均为 0。
7. Batch 3 old/new 映射、迁移前/后快照、语义差异、引用审计与门禁结果以 `docs/refactors/CONFIG_BATCH3_*` 为准；`paper_aligned` 仍不等于 `author_official`，作者官方 MultiDAG 复现尚未开始。
8. Config Batch 4 已完成：10 个 DialogueGCN YAML 均位于 `configs/dialoguegcn/{unified,paper_aligned}/` 下；6 个 `unified`、4 个 `paper_aligned`，10 个 YAML 均 byte-identical 且 semantic-identical。共更新 30 个 active references（其中 pipeline references 5 个），历史旧引用为 0，active old references 与未批准语义变化均为 0。
9. Batch 4 old/new 映射、迁移前/后快照、语义差异、引用审计与门禁结果以 `docs/refactors/CONFIG_BATCH4_*` 为准；`paper_aligned` 仍不等于 `author_official`，本批次不是作者官方 DialogueGCN 复现。
10. Config Batch 5 已完成：13 个 GS-MCC YAML 均位于 `configs/gsmcc/project_variant/iemocap/` 下；7 个为 `causal_context`，6 个为 `full_context`，13 个 YAML 均 byte-identical 且 semantic-identical。共更新 46 个 active references（其中 pipeline references 5 个），历史旧引用为 0，active old references 与未批准语义变化均为 0。
11. Batch 5 old/new 映射、迁移前/后快照、语义差异、引用审计与门禁结果以 `docs/refactors/CONFIG_BATCH5_*` 为准；当前 GS-MCC 仍是 `project_variant`，作者官方 GS-MCC 复现尚未开始。
12. Config Batch 6 已完成：37 个 YAML 已按冻结计划移动；2 个 cross-model benchmark 分别位于 `configs/benchmarks/{causal_unified,original_merc}/`，35 个 model-scoped ablation 仍位于 MMGCN 或 MultiDAG-CL 的 canonical tree。smoke 是正交属性，不改变 `model_scoped_ablation` 角色。
13. Batch 6 的 37 个 YAML 均 byte-identical 且 semantic-identical；共更新 59 个 active references，其中 27 个为尚未移动的 Batch 7 YAML 内部路径；保留并审计 109 个冻结历史引用，active old references、模型成员/顺序/provenance 变化、ablation/controlled-variable 变化与未批准语义变化均为 0。
14. Batch 6 old/new 映射、快照、语义差异、引用审计、跨模型成员审计与门禁结果以 `docs/refactors/CONFIG_BATCH6_*` 为准；Phase 4A plan 与 classification 未修改，作者官方 MultiDAG+CL 和 GS-MCC 复现仍未开始。
15. Config Batch 7 已完成：73 个 YAML 使用 `git mv` 迁入 `configs/benchmarks/{causal_unified,original_merc,ablations}/` 下的 canonical analysis、pipeline 与 manifest tree；旧路径剩余 0，新路径存在 73，tracked YAML 总数保持 183。
16. Batch 7 的 context/stable-context collision 已拆为两个唯一 missing-modality canonical path；60 个 YAML 内容不变，13 个仅更新必要的仓库内部配置路径，未批准语义变化为 0。共审计 101 个更新引用和 204 个冻结历史引用，active old references 为 0。
17. Batch 7 old/new 映射、快照、语义差异、引用审计、碰撞审查与门禁结果以 `docs/refactors/CONFIG_BATCH7_*` 为准；Phase 4A strict plan audit 与 Batch 1--7 strict migration audit 均通过。
18. 两份本地 untracked 组会文件继续保持不读取为生产依据、不修改、不移动、不 staged。
