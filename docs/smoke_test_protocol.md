# Smoke Test 协议

本文档记录本地 smoke test 的范围和规则。

## 本地 smoke test 的目的

本地 smoke test 只用于确认代码链路没有明显断裂。它不是正式实验，也不用于报告模型效果。

允许用于：

1. import 检查。
2. 配置解析。
3. 语法检查。
4. CSV 分析。
5. 图表生成。
6. 小型 dataloader 检查。
7. fake dialogue forward/backward 检查。
8. train -> checkpoint -> evaluate 最小闭环检查。

不允许用于：

1. 完整 M3ED 训练。
2. 多 seed 长训练。
3. 大规模 checkpoint 批量评估。
4. 依赖真实大体积 feature pkl 的完整实验。
5. 基于 smoke 指标下结论。

## 本地不要跑完整训练

本地 Windows 只做轻量检查。正式训练、多 seed 实验、完整 checkpoint 评估应在远程 V100 服务器上执行。

如果某个本地命令预计会长时间占用 GPU、读取大体积数据或写入正式 `outputs/`，应停止并改为远程执行。

## 当前可推断的最近 smoke test 命令

根据当前上下文和本地 `tmp/smoke_outputs/latest_run.txt`，最近一次 smoke test 使用了 fake dialogue 配置：

```powershell
conda run -n m3ed_mmgcn python scripts\train_mmgcn.py --config configs\mmgcn\unified\synthetic\full_context\synthetic\smoke.yaml
```

生成的 checkpoint 随 run id 变化。最近一次已知 run id 是：

```text
20260626_150801_mmgcn_smoke_fake_dialogues
```

对应评估命令是：

```powershell
conda run -n m3ed_mmgcn python scripts\evaluate_checkpoint.py --checkpoint tmp\smoke_outputs\runs\20260626_150801_mmgcn_smoke_fake_dialogues\checkpoints\best_model.pt --split test
```

最近一次语法检查命令是：

```powershell
conda run -n m3ed_mmgcn python -m py_compile datasets\smoke\mmgcn_smoke_dataset.py scripts\train_mmgcn.py scripts\evaluate_checkpoint.py
```

如果未来 `tmp/` 被清理，最近 smoke test 的精确 run id 可能不可从当前文件可靠恢复。此时不要编造，重新运行 smoke train 后读取 `tmp/smoke_outputs/latest_run.txt`。

## 通用 smoke test checklist

最小检查顺序：

1. 查看工作区：

```powershell
git status --short
```

2. 查看变更规模：

```powershell
git diff --stat
```

3. 语法检查：

```powershell
conda run -n m3ed_mmgcn python -m py_compile <changed_python_files>
```

4. fake dialogue smoke train：

```powershell
conda run -n m3ed_mmgcn python scripts\train_mmgcn.py --config configs\mmgcn\unified\synthetic\full_context\synthetic\smoke.yaml
```

5. 读取最新 run：

```powershell
Get-Content tmp\smoke_outputs\latest_run.txt
```

6. 用最新 `best_model.pt` 评估：

```powershell
conda run -n m3ed_mmgcn python scripts\evaluate_checkpoint.py --checkpoint tmp\smoke_outputs\runs\<run_id>\checkpoints\best_model.pt --split test
```

7. 检查输出是否存在：

```powershell
Get-ChildItem tmp\smoke_outputs\runs\<run_id>\logs
```

8. 最后查看：

```powershell
git diff --stat
```

## Smoke 输出规则

本地 smoke 输出应写入：

```text
tmp/smoke_outputs/
```

不要写入正式：

```text
outputs/
```

`tmp/` 已被 `.gitignore` 忽略，不应提交。

## Smoke 配置规则

本地 smoke 配置使用：

```text
configs/mmgcn/unified/synthetic/full_context/synthetic/smoke.yaml
```

它应满足：

1. 不读取 `data/`。
2. 使用小型 fake dialogue。
3. `max_epochs` 保持很小。
4. `device` 默认使用 `cpu`。
5. `output_dir` 指向 `tmp/smoke_outputs`。
6. 不改变正式 `M3ED` 配置。

## 结果解释规则

Smoke test 只回答：

1. import 是否正常。
2. 配置是否能解析。
3. dataloader 是否能产出统一 batch。
4. 模型 forward/loss/backward 是否能执行。
5. checkpoint 是否能保存。
6. 评估脚本是否能读回 checkpoint。

Smoke test 不回答：

1. 模型效果是否好。
2. 模块是否优于 baseline。
3. 多 seed 是否稳定。
4. M3ED 正式结果是否可复现。
