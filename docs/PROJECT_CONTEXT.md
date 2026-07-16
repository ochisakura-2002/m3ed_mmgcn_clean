# MERC 项目持久上下文

- **Document role：** Codex 的唯一详细项目入口与架构摘要
- **Last updated date：** 2026-07-16
- **Prepared on branch：** `audit/repository-cleanup-20260716`
- **Code snapshot commit：** `a16134e64747163ee8874531e66e5b4bd89cdedf`
- **Target working branch：** `feat/date-organized-outputs`
- **Source of truth status：** 辅助索引；源码、正式配置和运行产物是最终事实来源

本文用于按主题定位文件；Code snapshot commit 只标识所依据的代码快照，不包含当前未提交文档。
若本文与实现冲突，只针对冲突文件核验，并在重大修复后更新本文。

## 1. 项目目标

项目研究多模态对话情感识别（Multimodal Emotion Recognition in Conversation，MERC）。
输入以对话中的文本、音频、视觉特征为主，输出 utterance 级情感类别。

### 当前论文主线

- 当前小论文主线是复现高性能的原始、非因果 MERC baseline。
- 当前候选为 MMGCN、MultiDAG+CL、DialogueGCN 和项目 GS-MCC 变体。
- 先在受控协议下获得稳定、可比较的 baseline，再增加创新模块。
- 正式筛选以 Clean 特征 Validation 为主要模型选择证据。
- 当前活动阶段：`Original MERC formal screening is currently the active stage.`

### 长期研究路线

- causal MERC 是保留路线，不因当前论文重心变化而删除。
- 长期目标是服务 Human-Robot 在线情感识别、实时推理和因果对照研究。
- 稳定 baseline 之后再研究缺失模态、可靠性建模和在线上下文约束。

### 暂不推进的方向

- 暂不把 causal MERC 作为当前小论文的主要 baseline 排名线。
- 暂不在正式 baseline 未稳定前叠加多个创新模块。
- `SimpleMLP` 是工程基准，`SDT` 是隔离候选，不计入本文的四模型论文筛选。

本文详细记录 8 个 MERC 变体：4 个 Original/noncausal 候选和 4 个 causal 对照。

## 2. 当前模型路线

### 2.1 Original/noncausal MERC 共用入口

- 注册表：`models/baselines/original_repro/registry.py`
- 配置根：`configs/smoke/original_repro/`、`configs/experiments/original_merc/`
- 训练入口：`scripts/baselines/train_original_merc_baseline.py`
- 评估入口：`scripts/baselines/evaluate_original_merc_checkpoint.py`
- Pipeline：`scripts/baselines/run_original_merc_pipeline.py`
- 分析入口：`scripts/analyze/analyze_original_merc_results.py`
- 论文映射：`docs/baselines/original_repro/*_paper_code_mapping.md`

#### MMGCN

- Canonical key：`original_repro_mmgcn`
- 实现目录：`models/baselines/original_repro/mmgcn/`
- 配置目录：`configs/experiments/original_merc/{screening,clean_screening,legacy_fold_bases,clean_fold_bases}/`
- 训练入口：Original MERC 共用训练入口
- 评估入口：Original MERC 共用 checkpoint 评估入口
- 分析入口：Original MERC 共用分析入口
- 复现忠实度：`official_code_adapted`
- 主要机制：三模态 utterance 节点、official-like 多模态邻接、GCNII 和残差传播
- 当前用途：高性能原始 baseline 候选和后续创新模块宿主
- 已知限制：论文与官方代码的 speaker 注入及图层深度存在差异；真实 IEMOCAP 性能尚待正式证据

#### MultiDAG+CL

- Canonical key：`original_repro_multidag_cl`
- 实现目录：`models/baselines/original_repro/multidag_cl/`
- 配置目录：与 MMGCN 相同的 Original MERC 四阶段目录中的 `multidag_cl_*.yaml`
- 训练入口：Original MERC 共用训练入口
- 评估入口：Original MERC 共用 checkpoint 评估入口
- 分析入口：Original MERC 共用分析入口
- 复现忠实度：`official_code_adapted_with_protocol_repairs`
- 主要机制：有向无环对话图、同/异 speaker 关系、节点与上下文 GRU、baby-step curriculum
- 名称说明：`CL` 明确指 Curriculum Learning，不是 Contrastive Learning
- 当前用途：结构不同于 MMGCN 的强图模型候选
- 已知限制：项目移除了逐 epoch 使用 Test 选模型的旧行为；真实 IEMOCAP 复现结果仍待确认

#### DialogueGCN

- Canonical key：`original_repro_dialoguegcn`
- 实现目录：`models/baselines/original_repro/dialoguegcn/`
- 配置目录：Original MERC 四阶段目录中的 `dialoguegcn_*.yaml`
- 训练入口：Original MERC 共用训练入口
- 评估入口：Original MERC 共用 checkpoint 评估入口
- 分析入口：Original MERC 共用分析入口
- 复现忠实度：`paper_equation_aligned_official_code_adapted`
- 主要机制：双向序列上下文、speaker/direction 关系图、边注意力、两层关系图卷积和 nodal attention
- 当前用途：经典关系图 baseline 候选
- 已知限制：不是旧 PyG 栈的数值同一实现；完整模型不能描述为 official-code-identical

#### project_paper_oriented_gsmcc

- Canonical key：`project_paper_oriented_gsmcc`
- 实现目录：`models/baselines/original_repro/gsmcc/`
- 配置目录：Original MERC 四阶段目录中的 `gsmcc_*.yaml`
- 训练入口：Original MERC 共用训练入口
- 评估入口：Original MERC 共用 checkpoint 评估入口
- 分析入口：Original MERC 共用分析入口
- 复现忠实度：`PROJECT_VARIANT_NOT_PAPER_REPRODUCTION`
- 主要机制：项目解释的滑窗多模态图、简化 Fourier 图算子、低/高频分支和可选对比损失
- 当前用途：工程比较候选，不进入论文复现成功排名或 paper-gap 计算
- 已知限制：它不是忠实 GS-MCC 复现；报告必须保留完整项目名称与忠实度状态

### 2.2 两条 Original 实验轨道的解释

- legacy paper-adjacent 结果只用于复现诊断，不等同严格官方结果。
- Clean 结果用于四模型的公平 baseline 筛选。
- legacy 与 Clean 轨道必须分别汇总、分别排名，不能混合排序。
- Clean 结果不直接用于计算 paper gap。

### 2.3 Causal MERC

Causal 路线仍然保留，但当前不作为小论文的主要 baseline。
未来用途是在线机器人、实时前缀推理、无未来信息对照和 causal 消融。

- Causal MMGCN：`models/baselines/mmgcn/`；配置在 `configs/baselines/mmgcn/iemocap/causal_benchmark/`
- Causal MultiDAG：`models/baselines/multidag_cl/`；配置在 `configs/baselines/multidag_cl/iemocap/causal_benchmark/`
- Causal GS-MCC-inspired：`models/baselines/gsmcc/`；注册于 `models/baselines/causal_baseline_registry.py`
- Causal DialogueGCN：`models/baselines/dialoguegcn/`；注册于同一 causal registry
- 共用图工具：`models/baselines/causal_graph_common/`
- 新 causal 训练：`scripts/baselines/train_new_causal_graph_baseline.py`
- 新 causal 评估：`scripts/baselines/evaluate_new_causal_graph_checkpoint.py`
- 四模型审计：`scripts/analyze/run_four_model_causal_audit.py`
- 关键测试：`tests/test_*_causal_invariance.py`、`tests/test_four_model_causal_audit.py`
- 协议文档：`docs/baselines/causal_model_contract.md` 及各模型 causal audit

Causal 代码、配置、审计、测试和已有结果不得作为“当前不主推”而删除。

## 3. 数据与特征

### 3.1 Legacy 特征

- Feature set：`legacy_mmgcn_textcnn`
- 路径：`third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl`
- SHA256：`ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3`
- 文本维度：100
- 状态：`legacy_compatibility_only`
- 用途：paper-adjacent 复现诊断

### 3.2 Clean RoBERTa v1

- Feature set：`iemocap_clean_roberta_base_utterance_mean_v1`
- 路径：`data/processed/iemocap/IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.pkl`
- SHA256：`c604c557bc3fbb129ca02b2acd57166b669a89ef76ff0cea1e14f9cb206324bf`
- 文本维度：768
- 音频维度：1582
- 视觉维度：342
- 规模：151 个 dialogues，7433 个 utterances
- 空句：0
- 截断句：0
- 上下文边界：utterance-only，不拼接历史或未来
- Pooling：排除 padding 与 special tokens 后进行 mean pooling
- 标签边界：标签未参与特征构建
- 状态：`frozen_for_main_experiments`，immutable v1

注册与哈希来源为 `configs/data/iemocap_feature_sets.yaml`。
Clean 契约来源为 `docs/data/iemocap_clean_roberta_v1.md`。
构建入口为 `scripts/features/build_iemocap_clean_text_features.py`。

两份 PKL 均不由 Git 跟踪，因此不同 checkout 中可能不存在。
任何正式训练开始前，都必须检查固定路径、文件存在性和 SHA256。
不得因为本地缺失而生成占位文件、改变路径或放松哈希校验。
Legacy 与 Clean 是不同证据轨道，不得混合排名。

## 4. 数据划分与实验协议

Canonical manifest：`configs/experiments/original_merc/pipeline_manifest.yaml`。

### 4.1 legacy_official_split_safe_selection

- Outer Test：原始 `testVid`，对应 Ses05 Test。
- Inner Validation：仅从 `trainVid` 按 dialogue 分层抽取。
- 当前 screening 外层 Test Session：`Ses05`。
- 可比性：`paper_adjacent_not_exact`。
- 用途：复现诊断，不作为 Clean 公平排名的替代。

### 4.2 legacy_fivefold_fair_comparison

- Outer Test：Ses01 至 Ses05 逐 Session 留一，共五折。
- Inner Validation：从其余 Session 的 dialogue 中分层抽取。
- 特征：legacy MMGCN TextCNN 特征。
- 用途：在 legacy 特征内进行五折公平比较。

### 4.3 clean_roberta_fivefold_fair_comparison

- Outer Test：Ses01 至 Ses05 逐 Session 留一，共五折。
- Inner Validation：从其余 Session 的 dialogue 中分层抽取。
- 特征：冻结的 Clean RoBERTa v1。
- 用途：当前主线的公平 baseline 选择与最终证据。

### 4.4 选择边界

- Validation 的 `val_weighted_f1` 选择 best checkpoint。
- Test 不参与 checkpoint selection、early stopping 或 Top-2 选择。
- Test 只用于 best checkpoint 的最终报告。
- Top-2 主要依据 Clean single-fold screening Validation。
- 训练稳定性、复现可信度和创新插入点是后续排序依据。
- legacy Validation 只提供 paper-adjacent 诊断证据。

## 5. 正式实验阶段

当前第一阶段是 8 个正式 screening：

```text
4 个模型 × legacy/clean
```

随后按以下顺序推进：

```text
Clean Validation 选择 Top-2
→ Top-2 五折
→ 最终 baseline 多随机种子
→ 加创新模块
→ 消融
→ 对比实验
```

- Smoke 只验证最小 train → checkpoint → evaluate 闭环，不是正式结果。
- Screening 是低成本候选筛选，不是最终论文结果。
- 五折与多随机种子结果才用于稳定结论。
- 不把正在运行的 PID、临时时间或单机状态写成长期项目规则。
- Top-2 状态由 manifest 的 `top2_policy.status` 管理，目前仍为 pending。

## 6. Canonical 目录结构

- `models/`：项目模型实现与注册表
- `datasets/`：数据集、split、collate 与特征读取
- `utils/`：配置、指标、日志、metadata 和输出路径工具
- `scripts/baselines/`：正式 baseline 训练、评估与 pipeline
- `scripts/analyze/`：核心分析、审计、汇总和论文产物实现
- `scripts/analysis/`：历史兼容 wrapper，不是第二套核心实现
- `configs/smoke/`：最小闭环配置
- `configs/experiments/`：正式实验 manifest 与配置
- `configs/baselines/`：causal 和模型专项配置
- `docs/baselines/`：模型协议、审计与论文映射
- `docs/experiments/`：正式实验与输出结构交接
- `tests/`：单元、协议、因果不变性和 smoke 回归
- `outputs/YYYYMMDD/`：按启动日期冻结的动态运行产物

默认不要扫描 `outputs/`、`third_party/`、`data/processed/`、`tmp/source_audit/` 或 `.git/`。

## 7. 输出目录规则

新动态产物由 `utils/output_paths.py` 统一解析：

```text
outputs/YYYYMMDD/runs/
outputs/YYYYMMDD/logs/
outputs/YYYYMMDD/analysis/
outputs/YYYYMMDD/manifests/
outputs/YYYYMMDD/audits/
outputs/YYYYMMDD/review/
```

同一日期根还可包含 `reports/` 与 `smoke/`。
`outputs/environment/`、`outputs/reference/`、`outputs/cache/` 是全局静态目录。

- 日期格式严格为合法 `YYYYMMDD`。
- 解析优先级：CLI、`output.experiment_date`、`MERC_EXPERIMENT_DATE`、本机日期。
- Pipeline 启动时冻结一次日期，跨午夜不切换。
- Resume 使用 checkpoint 所属原 run 目录和原实验日期。
- 显式 run、checkpoint、runs-root 或 output-dir 路径优先。
- 自动发现先查日期结构，再兼容旧 `outputs/<category>` 结构。
- 历史结果不迁移、不改名、不覆盖。

## 8. Canonical 入口

本节固定 8 个主要入口；使用前仍应查看 `--help` 和相关配置。

### 8.1 Original MERC 训练

- Path：`scripts/baselines/train_original_merc_baseline.py`
- Purpose：训练四个 Original 候选并保存 Validation 选择的 checkpoint。
- Typical command：`python scripts/baselines/train_original_merc_baseline.py --config configs/experiments/original_merc/clean_screening/mmgcn_clean.yaml`

### 8.2 Original MERC checkpoint 评估

- Path：`scripts/baselines/evaluate_original_merc_checkpoint.py`
- Purpose：显式加载 checkpoint，在 val 或 test split 上评估。
- Typical command：`python scripts/baselines/evaluate_original_merc_checkpoint.py --checkpoint <last_or_best_model.pt> --split test`

### 8.3 Original MERC Pipeline

- Path：`scripts/baselines/run_original_merc_pipeline.py`
- Purpose：按 versioned manifest 规划或执行 smoke、screening、fold 和 Top-2。
- Typical command：`python scripts/baselines/run_original_merc_pipeline.py --stage clean_screening --execute`

### 8.4 Original MERC 统一分析

- Path：`scripts/analyze/analyze_original_merc_results.py`
- Purpose：分轨汇总结果、生成诊断和待审查的 Top-2 selection。
- Typical command：`python scripts/analyze/analyze_original_merc_results.py`

### 8.5 配置树校验

- Path：`scripts/dev/validate_config_tree.py`
- Purpose：检查配置 schema、canonical 路径、协议与受保护字段。
- Typical command：`python scripts/dev/validate_config_tree.py`

### 8.6 IEMOCAP feature audit

- Path：`scripts/analyze/audit_iemocap_feature_pkl.py`
- Purpose：对照 legacy PKL 审计 Clean candidate 的字段、维度、哈希和文本替换边界。
- Typical command：`python scripts/analyze/audit_iemocap_feature_pkl.py --legacy-pkl <legacy.pkl> --candidate-pkl <clean.pkl> --expected-legacy-sha256 <sha256> --expected-text-dim 768 --output-dir <audit_dir> --strict`

### 8.7 四模型 causal audit

- Path：`scripts/analyze/run_four_model_causal_audit.py`
- Purpose：验证四个 causal 变体的前缀不变性和统一合同。
- Typical command：`python scripts/analyze/run_four_model_causal_audit.py --config configs/analysis/four_model_causal_audit.yaml --strict`

### 8.8 Paper artifact export

- Path：`scripts/analyze/export_paper_artifacts.py`
- Purpose：从现有 run 导出单次实验论文表格和图，不改变训练或选择协议。
- Typical command：`python scripts/analyze/export_paper_artifacts.py --run-dir outputs/<YYYYMMDD>/runs/<run_id> --split test --also-val`

## 9. 环境与位置

- Local repository：`E:\MERC\m3ed_mmgcn_clean`
- Remote repository：`/home/zhiyuan/research/m3ed_mmgcn_clean`
- Conda environment：`m3ed_mmgcn`
- 当前主工作分支：`feat/date-organized-outputs`
- 本文更新所在审计分支：`audit/repository-cleanup-20260716`

分支会变化，任何任务开始前仍需运行：

```text
git status -sb
git branch --show-current
```

本地 Windows 用于编辑、轻量测试、CSV 分析和图表生成。
正式训练、多 seed 与 checkpoint 评估主要在远程 V100 环境执行。

## 10. Git 与远程工作流

1. 在本地创建或切换明确的任务分支。
2. 修改前检查并保留无关工作区改动。
3. 在本地运行当前门禁并人工审查 diff。
4. 只有用户决定后才 commit 和 push。
5. 远程无训练占用工作区时再 pull 对应提交。
6. 远程正式实验在 tmux 中运行并保存 resolved config 与 metadata。
7. 训练期间不修改、pull、切分支或清理远程工作区。
8. 结果通过自动整理的 review bundle 返回本地。
9. 本地分析 bundle，确认协议和完整性后再形成结论。

详细规则见 `docs/codex_workflow.md` 和 `docs/git_workflow.md`；本任务规则不授权自动 commit、push 或触碰正在训练的远程目录。

## 11. 测试和质量门禁

当前标准门禁：

```text
python -m pytest -q
python scripts/dev/validate_config_tree.py
git diff --check
```

本地典型执行方式是在 Conda 环境 `m3ed_mmgcn` 中运行。
模型改动应额外运行对应 smoke、checkpoint reload 和机制测试。
协议改动应额外运行 split、Test isolation 与分析选择测试。

Current documentation review：122 passed

Historical snapshot：cleanup-audit-only branch，137 passed

这些数字只是历史证据，不是当前 PASS。
每次交付必须运行适合当前 checkout 的测试并报告真实结果。

## 12. 复现忠实度与命名规则

- 只有论文、官方代码、数学实现和协议均完成核对，才可称 faithful reproduction。
- `official_code_adapted` 仍需同时说明适配点和真实性能证据状态。
- Paper-oriented variant 必须在名称和报告中显式标注。
- 不允许把 `project_paper_oriented_gsmcc` 写成官方 GS-MCC 复现。
- `gsmcc`、`gs-mcc` 只是 CLI 兼容别名，不改变 canonical key。
- legacy paper-adjacent 不等于严格官方结果。
- legacy 轨道可诊断 paper gap，但必须标注协议并非 exact。
- Clean track 用于公平筛选，不直接计算 paper gap。
- 报告不得把 synthetic、smoke 或单折 screening 写成最终复现结论。
- 忠实度来源：`models/baselines/original_repro/registry.py` 与论文映射文档。

## 13. 保护规则

下列内容未经明确批准不得修改、替换、移动或删除：

- legacy IEMOCAP PKL 及其固定 SHA256
- Clean RoBERTa v1 PKL 及其固定 SHA256
- feature registry 和 immutable v1 契约
- IEMOCAP 数据 split 与 Session holdout
- 正式 smoke、screening、fold 和 pipeline 配置
- Validation 选 checkpoint、Test 只做最终报告的协议
- causal 路线的代码、配置、审计、测试和历史结果
- Original MERC 四模型实现、canonical 名称和忠实度声明
- best checkpoint 与 last checkpoint
- 正式 metrics、predictions 和 confusion matrix
- run metadata、resolved config 与实验 manifest
- 论文—代码映射文档和复现审计
- 旧 outputs 读取与 resume 兼容逻辑
- 远程正在运行的代码和工作区

大文件缺失时不得创建伪造占位内容。

## 14. 已知问题与待决策

- `project_paper_oriented_gsmcc` 是项目变体，不是忠实 GS-MCC 复现。
- 四模型 Clean screening 尚未形成最终 Top-2 决策。
- Top-2 五折和最终多随机种子实验尚未完成。
- 最终创新模块尚未选择，应等待稳定 baseline 证据。
- `tmp/source_audit/` 当前存在；后续是否归档需单独人工决策，不能自动删除。
- `scripts/analysis/` 仍含兼容 wrapper；是否移除需先核对外部调用方。
- 旧输出布局仍有读取价值，当前没有迁移或删除历史结果的计划。

## 15. 重要历史决策

| Date | Decision | Reason | Impact |
| --- | --- | --- | --- |
| 2026-07-13 | 保留并建立四模型 causal 路线 | 支持无未来信息审计与长期在线推理 | causal 代码、配置、审计和测试成为受保护资产 |
| 2026-07-14 | 冻结 Clean RoBERTa v1 | 避免 legacy 文本特征训练边界不明 | 建立新 PKL、固定 SHA 与严格 feature audit |
| 2026-07-14 | Legacy 与 Clean 分轨 | 两种文本特征的证据边界不同 | 分别汇总、分别排名，禁止混合比较 |
| 2026-07-16 | 小论文转向 Original MERC baseline | 先获得高性能、可解释且稳定的比较基础 | 四个 Original 候选成为当前正式 screening 主线 |
| 2026-07-16 | Test 不参与模型选择 | 防止 checkpoint 与候选选择泄漏 | Validation 选 best；Test 只做最终报告 |
| 2026-07-16 | GS-MCC 改名为项目变体 | 论文、官方代码与项目机制无法支持忠实声明 | canonical key 固定为 `project_paper_oriented_gsmcc` |
| 2026-07-16 | 输出按 pipeline 启动日期组织 | 隔离实验、支持并发并保持可追溯性 | 动态产物写入 `outputs/YYYYMMDD/<category>` |
| 2026-07-16 | Clean Validation 用于 Top-2 | 公平且不使用 Test 的选择证据 | legacy 仅作诊断，Top-2 状态保持 pending 至结果齐全 |

## 16. Codex 任务启动规则

后续 Codex 开始任务时：

1. 阅读根目录 `AGENTS.md`。
2. 阅读本文与任务相关的章节，不默认通读所有历史文档。
3. 运行 `git status -sb` 和 `git branch --show-current`。
4. 列出计划打开、修改和测试的文件。
5. 只打开任务涉及的源码、配置、文档和测试。
6. 优先使用本文列出的 canonical 文件路径。
7. 使用窄范围 `rg`，不进行无目的全仓库扫描。
8. 默认不扫描 `outputs/`、`third_party/`、`data/processed/`、`tmp/source_audit/`、`.git/`。
9. 不读取大 PKL、checkpoint 或整个结果树来回答一般架构问题。
10. 发现上下文可能过期时，只核验冲突涉及的具体实现和正式配置。
11. 若源码与本文冲突，在结果中说明证据，不静默选择一方。
12. 任务完成后运行范围相称的测试、配置校验和 diff 检查。
13. 显示 `git diff --stat`，并列出新增、修改和删除文件。
14. 只有第 1 章所列主线或维护规则中的重大变化才更新本文。
15. 普通 bug fix、格式化和小测试不扩写本文。
16. 不创建其他并行的项目总结文档。
