# 按实验启动日期组织 outputs：交接说明

## 交付状态

当前分支为 `feat/date-organized-outputs`。本次只调整输出路径、结果发现、
resume 路径、相关配置、文档和测试；未修改模型结构、causal 语义、loss、
数据划分、特征协议或训练超参数，也未执行 Git commit。

## 新目录结构

所有新动态结果默认写入：

```text
outputs/<YYYYMMDD>/
├── runs/<run_id>/
├── logs/<pipeline_or_run_name>/
├── analysis/<analysis_name>/
├── reports/<report_name>/
├── manifests/<pipeline_name>/
├── smoke/<smoke_name>/
├── audits/<audit_name>/
└── review/<review_name>/
```

以下目录仍是全局静态目录，不按日期分类：

```text
outputs/environment/
outputs/reference/
outputs/cache/
```

Smoke 配置可以继续把逻辑根设为 `tmp/smoke_outputs`；它们在该根目录下
仍采用 `<YYYYMMDD>/<category>` 结构，不污染正式 `outputs`。

## 日期解析与冻结

`experiment_date` 严格采用合法的 8 位本地日历日期 `YYYYMMDD`。解析优先级为：

1. CLI `--experiment-date`
2. 配置 `output.experiment_date`
3. 环境变量 `MERC_EXPERIMENT_DATE`
4. 启动机器的本地日期

非法日期会在创建 run 和模型初始化之前失败。Pipeline 启动时只解析一次，
保存 `experiment_date` 与 `day_output_root`，并通过显式参数及环境变量传给
所有子任务；跨午夜不会切换日期。

## 中央路径模块

新增 `utils/output_paths.py`，集中提供：

- 日期验证与解析：`validate_experiment_date`、`resolve_experiment_date`
- 日期根和分类目录：`resolve_day_output_root`、`resolve_output_category`
- 原子唯一目录：`create_unique_run_dir`、`create_unique_category_dir`
- 新旧发现：`discover_run_directories`、`discover_analysis_directories`
- 路径查找与日期推断：`find_run_directory`、`find_analysis_artifact`、
  `infer_experiment_date_from_run`
- resolved metadata：`resolved_output_metadata`

Run ID 采用 `<name>_<date>_<HHMMSS>_<6-hex>`。目录通过原子
`mkdir(exist_ok=False)` 分配，因此同秒、同名和并发启动不会互相覆盖。
`latest_run.txt` 只写到当前 run/pipeline 自己的 manifest 目录，不再使用
跨 pipeline 的共享全局指针。

## 已迁移入口

训练和 pipeline：

- `scripts/train.py`
- `scripts/train_mmgcn.py`
- `scripts/train_simple_mlp.py`
- `scripts/run_experiment_pipeline.py`
- `scripts/baselines/train_multidag_cl.py`
- `scripts/baselines/train_multidag_cl_smoke.py`
- `scripts/baselines/train_new_causal_graph_baseline.py`
- `scripts/baselines/train_original_merc_baseline.py`
- `scripts/baselines/run_original_merc_pipeline.py`
- `scripts/baselines/new_causal_graph_runtime.py`
- `scripts/baselines/original_merc_runtime.py`

评估、分析、审计、汇总和报告：

- checkpoint evaluation 继续以显式 checkpoint 为准，并写回 checkpoint 所属 run
- analysis table、单/多 run 曲线、最终分析、loss stability、Original MERC 汇总
- four-model causal audit、causal benchmark review、IEMOCAP split audit
- paper table/report export、evaluation/experiment summary rebuild
- `scripts/inspect/extract_mmgcn_core_blocks.py`

相关 baseline、smoke、analysis 和 Original MERC YAML 已改为逻辑
`output.root` 加可空的 `output.experiment_date`；正式配置没有硬编码某一天。

## Metadata、resume 与显式路径

Resolved config 和 run metadata 记录：

```text
experiment_date
output_root
day_output_root
run_dir
log_dir
analysis_dir
manifest_dir
```

Original MERC 的 `--resume <last_model.pt>` 会沿用 checkpoint 所属旧 run
目录以及该 run 的原实验日期，追加已有 history，不因当天日期变化创建新 run。
显式 `--run-dir`、checkpoint、`--runs-root` 和 `--output-dir` 始终优先，
不会被自动日期推断改写。

## 新旧目录兼容

自动发现顺序为：

1. 合法日期目录下的新结构，日期目录必须同时通过 `^\d{8}$` 和日历校验；
2. 旧 `runs` / `logs` / `analysis` 结构；
3. 规范化路径去重。

`environment`、`reference`、`cache` 以及非法日期名不会被当成日期目录。
Original MERC top-two selection 默认从新日期 analysis 开始发现，之后兼容旧
analysis；如 manifest 显式指定 selection 文件，则显式路径优先。

## 修改文件概览

本任务直接修改或新增的文件分组如下：

- 中央工具：`utils/output_paths.py`、`utils/io.py`、`utils/run_metadata.py`
- 训练/pipeline：上述“已迁移入口”中的训练和 pipeline 文件
- 分析/维护：`scripts/analyze/` 下相关入口、
  `scripts/maintenance/{collect_evaluation_summary,rebuild_experiment_summary}.py`、
  `scripts/inspect/extract_mmgcn_core_blocks.py`
- 配置：`configs/baselines/{dialoguegcn,gsmcc,multidag_cl}/**/*.yaml` 中的
  输出段、`configs/analysis/**/*.yaml`、`configs/smoke/*.yaml`、
  `configs/experiments/original_merc/**/*.yaml`
- 测试：`tests/test_output_paths.py`、`tests/test_run_metadata.py`，并复用
  `tests/original_repro/` 与 causal checkpoint/smoke 回归测试
- 文档：`docs/experiment_protocol.md`、两个 Original MERC handoff 和本文

## 验证结果

在 Conda 环境 `m3ed_mmgcn` 中完成：

```text
python -m pytest -q
122 passed, 14 warnings

python scripts/dev/validate_config_tree.py
Config validation passed.

git diff --check
passed（仅有 Git 的 LF/CRLF 转换提示）
```

另有日期输出、Original MERC runtime checkpoint、causal smoke/checkpoint 的
定向回归：`23 passed`。14 条 warning 均来自当前环境的 matplotlib/
pyparsing deprecation，不是本次输出路径逻辑。

## 旧路径搜索审计

对源码、配置和测试搜索 `outputs/runs`、`outputs/logs`、`outputs/analysis`
的结果为 0；因此没有仍默认写入旧动态目录的代码入口。

仓库文档中仍保留旧路径的位置分为两类：

- 兼容性说明：`docs/experiment_protocol.md`、
  `docs/experiments/original_merc_handoff.md`、
  `docs/baselines/original_repro/original_merc_reproduction_handoff.md`。这些位置
  用于说明如何读取/显式分析旧结果。
- 历史实验记录：`causal_benchmark_review_tool_notes.md`、
  `iemocap_split_protocol_audit.md`、`multidag_cl_formal_run_plan.md`、
  `multidag_cl_iemocap_experiment_matrix_notes.md`、
  `multidag_cl_loss_stability_notes.md`、
  `multidag_cl_multirun_stabilization_analysis_notes.md`、
  `multidag_cl_observability_and_figures_notes.md`、
  `multidag_cl_paper_artifacts_notes.md`、
  `multidag_cl_pipeline_integration_notes.md`、
  `multidag_cl_yaml_parameter_guide.md`。这些是既有结果/命令的历史快照，
  不是当前写入入口，因此未批量改写其历史语境。

## 未解决事项

- 本地未运行正式训练、多 seed 或 V100 实验；这符合仓库执行边界。
- 调试早期曾生成四个合成测试 run 及同名 manifest：
  `outputs/20260716/{runs,manifests}/original_dialoguegcn_synthetic_test_*`。
  清理需要删除 `outputs` 内容，且任务明确禁止擅自改动历史输出，因此它们
  被原样保留；修复后重跑测试未再产生新正式输出。
- 历史说明文档仍含旧路径，原因见上一节；不存在未迁移的动态写入口。

## 远程运行示例

显式冻结日期：

```bash
conda activate m3ed_mmgcn
python scripts/baselines/run_original_merc_pipeline.py \
  --stage clean_screening \
  --execute \
  --experiment-date 20260716
```

通过环境变量冻结日期：

```bash
export MERC_EXPERIMENT_DATE=20260716
python scripts/run_experiment_pipeline.py \
  --config configs/pipeline/mmgcn_pipeline.yaml
```

Resume 旧 run：

```bash
python scripts/baselines/train_original_merc_baseline.py \
  --config configs/experiments/original_merc/clean_screening/mmgcn_clean.yaml \
  --resume outputs/20260716/runs/<run_id>/checkpoints/last_model.pt
```

检查某一天并分析旧结构：

```bash
find outputs/20260716/runs -maxdepth 1 -type d
find outputs/20260716/logs -type f
find outputs/20260716/analysis -maxdepth 2 -type f
python scripts/analyze/analyze_original_merc_results.py \
  --runs-root outputs/runs \
  --output-dir <explicit-legacy-analysis-dir>
```

建议提交信息（未执行）：`feat: organize experiment outputs by date`。
