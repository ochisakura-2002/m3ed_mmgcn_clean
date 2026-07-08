# 项目地图

本文档用于让后续 Codex 对话快速定位项目结构，避免每次重新全量扫描仓库。

## 项目目标

本项目目标是构建和复现一个多模态对话情感识别系统。当前重点不是随意添加创新模块，而是围绕 `M3ED + MMGCN` 建立稳定、可复现、可分析的 baseline 和实验闭环。

## 数据集与 baseline

当前主数据集是 `M3ED`。

辅助复现数据集是 `IEMOCAP official MMGCN-style features`，用于对齐和检查官方 `MMGCN` 风格特征读取与训练流程。

当前主 baseline 是 `MMGCN`。项目中也保留了 `SimpleMLP`，主要用于 sanity baseline 和性能对照。

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
5. `models/`：`SimpleMLP` 和 `MMGCN` 模型实现。
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

1. `scripts/train_mmgcn.py`：当前 `MMGCN` 主训练入口，支持 `M3ED`、`IEMOCAP`，当前本地 smoke 版本还支持 `MMGCN_SMOKE`。
2. `scripts/evaluate_checkpoint.py`：checkpoint 评估入口，读取 checkpoint 内保存的 config，重建模型和 dataloader。
3. `scripts/train_simple_mlp.py`：`SimpleMLP` baseline 训练入口。
4. `scripts/train.py`：早期最小入口，只做配置读取和 run 目录创建，不是当前正式 `MMGCN` 训练入口。
5. `scripts/run_experiment_pipeline.py`：实验 pipeline 入口，串联训练、评估和分析。

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

1. `models/baselines/mmgcn/mm_gcn.py`：当前 `MMGCN` 模型主体。
2. `models/baselines/mmgcn/dense_graph.py`：构建 dense multimodal adjacency。
3. `models/baselines/simple_mlp.py`：简单三模态拼接 MLP baseline。

### 工具与分析

1. `utils/seed.py`：随机种子设置。
2. `utils/metrics.py`：分类指标计算。
3. `utils/io.py`：路径、YAML、run 目录工具。
4. `scripts/analyze/`：训练曲线、最终指标、多 run 汇总、缺失模态图表。
5. `scripts/debug/`：轻量调试脚本。
6. `scripts/inspect/`：数据和官方代码结构检查脚本。
7. `scripts/maintenance/check_env.py`：环境检查脚本。

## 入口脚本速查

训练：

```powershell
conda run -n m3ed_mmgcn python scripts\train_mmgcn.py --config configs\train_mmgcn_m3ed.yaml
```

评估：

```powershell
conda run -n m3ed_mmgcn python scripts\evaluate_checkpoint.py --checkpoint <run_dir>\checkpoints\best_model.pt --split test
```

pipeline：

```powershell
conda run -n m3ed_mmgcn python scripts\run_experiment_pipeline.py --config configs\pipeline\mmgcn_pipeline_m3ed.yaml
```

本地 smoke test：

```powershell
conda run -n m3ed_mmgcn python scripts\train_mmgcn.py --config configs\smoke\train_mmgcn_smoke.yaml
```

```powershell
conda run -n m3ed_mmgcn python scripts\evaluate_checkpoint.py --checkpoint tmp\smoke_outputs\runs\<run_id>\checkpoints\best_model.pt --split test
```

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
