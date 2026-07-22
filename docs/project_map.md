# 项目地图

本文档用于让后续 Codex 对话快速定位项目结构，避免每次重新全量扫描仓库。

跨 Codex 对话的当前阶段、canonical 路径和最近门禁结果以 `docs/PROJECT_CONTEXT.md` 为准。

## 项目目标

本项目是一个多模型多模态对话情感识别（MERC）实验平台，长期目标与机器人实时交互中的 causal MERC 有关。当前重点是建立作者官方复现、统一 Original / full-context 与 causal 规程、Legacy / Clean 特征比较和最终 baseline 决策的可审计闭环。

## 数据集与 baseline

近期正式实验与 baseline 决策主要围绕 `IEMOCAP`；当前正式批次 `causal8_original8_formal_20260722_012904` 包含 8 个 causal run 和 8 个 Original run，共 16 个实验，全部 PASS 且数值状态均为 FINITE。

`M3ED` 仍是长期数据资产，用于后续扩展到机器人代理和 causal MERC 场景。

当前模型池包括 MMGCN、MultiDAG-inspired + Curriculum Learning、DialogueGCN、GS-MCC Project Variant、Causal-MMGCN 和 Causal-MultiDAG-inspired；`SimpleMLP` 用作 sanity baseline。MMGCN Legacy 是复现较稳定的经典锚点，但最终论文 baseline 尚未选定。

## 本地与远程分工

本地 Windows 主要负责：

1. 代码编辑。
2. 轻量 smoke test。
3. 配置解析检查。
4. CSV 分析。
5. 图表生成。
6. 小型 dataloader 或 forward 检查。

远程 V100 服务器主要负责：

1. 正式训练。
2. 多 seed 实验。
3. 大规模 checkpoint 评估。
4. 需要真实 M3ED/IEMOCAP 特征文件的实验。

不要假设本地绝对路径和远程绝对路径一致。配置和文档中应优先使用相对路径。

## 当前顶层目录

当前 checkout 中找到的顶层目录：

1. `configs/`：训练配置、pipeline 配置、分析配置、模态消融配置、smoke 配置。
2. `datasets/`：M3ED、IEMOCAP、通用 feature dataset、collate、smoke dataset。
3. `docs/`：项目记录、设计说明和 Codex 上下文文档。
4. `losses/`：当前主要是占位和说明。
5. `models/`：按模型与实现 lineage 组织的多模型实现、公共组件和 registry。
6. `scripts/`：训练、评估、调试、诊断、分析、pipeline、实验辅助脚本。
7. `tmp/`：本地临时输出，已被 `.gitignore` 忽略。
8. `trainers/`：当前主要是占位和说明。
9. `utils/`：路径、seed、metrics 等共享工具。

当前 checkout 未找到：

1. `data/`
2. `outputs/`
3. `checkpoint/`
4. `checkpoints/`
5. `logs/`
6. `third_party/`
7. `requirements.txt`
8. `environment.yml`

如果未来任务依赖这些路径，先确认它们是否在当前机器或远程机器上存在，不要编造路径。

## 关键文件职责

### 训练与评估

1. `scripts/models/mmgcn/unified/train.py`：canonical `MMGCN` 主训练入口，支持 `M3ED`、`IEMOCAP` 和本地 `MMGCN_SMOKE`。
2. `scripts/evaluation/unified_checkpoint.py`：canonical 统一 checkpoint 评估入口。
3. `scripts/models/multidag_cl/unified/`：MultiDAG+CL canonical 训练、评估与 smoke 入口。
4. `scripts/models/simple_mlp/train.py`：`SimpleMLP` canonical 训练入口。
5. `scripts/runtime/`：causal graph 与 paper-aligned 跨模型 runtime。
6. `scripts/workflows/`：通用、causal、paper-aligned、formal benchmark 和 modality workflow。
7. `scripts/train_mmgcn.py`、`scripts/evaluate_checkpoint.py`、`scripts/run_experiment_pipeline.py` 等迁移前路径仅为 compatibility wrapper。

### 配置

1. `configs/train_mmgcn_m3ed.yaml`：默认 `M3ED + MMGCN` 配置。
2. `configs/train_mmgcn_m3ed_causal.yaml`：causal context 版本配置。
3. `configs/train_mmgcn_iemocap_official.yaml`：IEMOCAP official feature 配置。
4. `configs/train_simple_mlp_m3ed.yaml`：`SimpleMLP` baseline 配置。
5. `configs/pipeline/`：pipeline 配置。
6. `configs/modality_ablation/`：模态消融配置。
7. `configs/smoke/train_mmgcn_smoke.yaml`：本地 fake dialogue smoke 配置。

### 数据读取

1. `datasets/m3ed/metadata_dataset.py`：读取 `m3ed_metadata.csv`，按 dialogue 聚合 utterance。
2. `datasets/common/dialogue_feature_dataset.py`：读取 DialogueRNN/MMGCN 风格 pkl 特征。
3. `datasets/m3ed/feature_dataset.py`：对齐 M3ED metadata 与 feature pkl。
4. `datasets/m3ed/torch_dataset.py`：把 M3ED sample 包装成 PyTorch dataset item。
5. `datasets/collators/m3ed_collate.py`：把变长 dialogue padding 成 `[B, T, D]` batch。
6. `datasets/iemocap/official_feature_dataset.py`：读取 IEMOCAP official MMGCN-style feature pkl。
7. `datasets/smoke/mmgcn_smoke_dataset.py`：本地 smoke fake dialogue dataset，不读取真实数据。

### 模型

1. `models/mmgcn/unified/`：统一训练/评估接口下的 `MMGCN`，主体仍为 `mm_gcn.py`，dense graph 位于 `dense_graph.py`。
2. `models/mmgcn/paper_aligned/`：项目内按论文结构实现的 full-context MMGCN，不是作者官方源码目录。
3. `models/multidag_cl/{unified,paper_aligned}/`：MultiDAG+CL；其中 CL 表示 Curriculum Learning。
4. `models/dialoguegcn/{unified,paper_aligned}/`：DialogueGCN 的统一接口与论文对齐实现。
5. `models/gsmcc/project_variant/{causal,full_context}/`：两套 GS-MCC Project Variant，均不是 author-official 实现。
6. `models/common/` 与 `models/registry/`：共享图/论文对齐工具和 canonical registry。
7. `models/simple_mlp/model.py`：简单三模态拼接 MLP sanity baseline。
8. `models/experimental/sdt/`：实验 SDT，不进入正式 baseline 排名。
9. `models/baselines/`：旧 import 的临时兼容 wrapper；新代码不得依赖该目录。

`unified` 表示项目统一训练/评价接口下的实现；`paper_aligned` 表示项目内按论文结构实现，不等于 `author_official`；`project_variant` 表示与作者官方实现存在明确差异。当前 MultiDAG 项目实现不是作者官方完整复现，GS-MCC 两套实现均为 `project_variant`。

### 工具与分析

1. `utils/seed.py`：随机种子设置。
2. `utils/metrics.py`：分类指标计算。
3. `utils/io.py`：路径、YAML、run 目录工具。
4. `scripts/analyze/`：训练曲线、最终指标、多 run 汇总、缺失模态图表。
5. `scripts/debug/`：轻量调试脚本。
6. `scripts/inspect/`：数据和官方代码结构检查脚本。
7. `scripts/maintenance/check_env.py`：环境检查脚本。

## 当前重构阶段

模型目录重构与 Script Phase 3A execution layout 已完成；旧入口保留 compatibility wrapper。Script Phase 3B 的 analysis/debug/data/maintenance scripts 与 `configs/` 尚未迁移。下一阶段先完成 Phase 3B，再重构 `configs/`；全部目录重构和门禁完成前，不部署作者官方 MultiDAG+CL 或 GS-MCC。

## 入口脚本速查

canonical 训练：

```powershell
conda run -n m3ed_mmgcn python scripts\models\mmgcn\unified\train.py --config configs\train_mmgcn_m3ed.yaml
```

canonical 评估：

```powershell
conda run -n m3ed_mmgcn python scripts\evaluation\unified_checkpoint.py --checkpoint <run_dir>\checkpoints\best_model.pt --split test
```

canonical pipeline：

```powershell
conda run -n m3ed_mmgcn python scripts\workflows\run_pipeline.py --config configs\pipeline\mmgcn_pipeline_m3ed.yaml
```

本地 smoke test：

```powershell
conda run -n m3ed_mmgcn python scripts\models\mmgcn\unified\train.py --config configs\smoke\train_mmgcn_smoke.yaml
```

```powershell
conda run -n m3ed_mmgcn python scripts\evaluation\unified_checkpoint.py --checkpoint tmp\smoke_outputs\runs\<run_id>\checkpoints\best_model.pt --split test
```

迁移前的旧命令仍通过 compatibility wrapper 工作；新自动化与新文档应使用 canonical 路径。

## 典型 run 输出结构

训练脚本通常生成：

```text
<output_dir>/
  latest_run.txt
  runs/
    <timestamp>_<experiment_name>/
      checkpoints/
        best_model.pt
        last_model.pt
      logs/
        experiment_config.yaml
        epoch_metrics.csv
        val_predictions_best.csv
        confusion_matrix_best.csv
        per_class_recall_best.csv
        evaluations/
      figures/
```

正式实验通常使用 `outputs/`。本地 smoke test 使用 `tmp/smoke_outputs/`，避免污染正式输出。

## 不要随意修改的路径

1. `data/`
2. `checkpoint/`
3. `checkpoints/`
4. `logs/`
5. `outputs/`
6. `tmp/`
7. `third_party/`
8. `*.pkl`
9. `*.pt`
10. `*.pth`
11. `*.ckpt`
12. `__pycache__/`
13. `.venv/`
14. `venv/`

这些路径和文件大多是数据、checkpoint、缓存、本地环境或实验产物。除非任务明确要求，否则不要修改、移动或提交。
