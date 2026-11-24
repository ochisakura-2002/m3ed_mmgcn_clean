# H1 Fast-Kill 依赖闭包

2026-11-24 · `WORKSTREAM = MULTIDAG` · 基底 `6e26c99d35f5ab59998231515406f79820b7356a`

使用临时 bare Git 的独立索引，从基底树叠加 [实现报告所列 H1 文件](HISTORY_FASTKILL_IMPLEMENTATION_REPORT.md#required_modified_files)。通过 `git write-tree` 和 `git archive <candidate-tree>` 解包到独立目录。本地分支 HEAD、主索引和原工作树的无关修改保留。

从 `run_history_fastkill.py` 递归解析项目内绝对、相对和函数级导入，纳入包初始化文件。运行闭包有 99 个 Python 文件，10 个验收模块及 pytest `conftest.py` 增加 27 个测试侧文件。审计集合没有动态 import 调用，且与无关 dirty 路径的交集为 0。入口的 YAML 和底层 TAV 源 YAML 都从归档中解析。真实恢复 ZIP 是显式传入的外部输入。

在 Python `-I -B` 进程中检查了 128 个已加载的项目文件，来源均为候选归档。候选内运行 `--help` 和 202 项定向测试成功。Windows 沙箱对 pytest `0700` 临时目录拒绝访问，归档外的验证启动器仅在本次临时目录内让这些目录继承父目录 ACL。项目与测试文件保持候选归档版本。

还将同一合成输入分别交给原 HEAD 归档和候选归档的 `FusionTAV`。两边 `h0…h4`、logits、mask、邻接拼接字节的 SHA-256 均为 `a646c46955d05742b1244fadfcebcd474fe395da1cec616f514915c0ce0841e0`，F0 `state_dict` 键、张量形状与 extra state 的摘要均为 `19e2aa059b5dc47fff7056b2c6065022022fef19f5558c53650d3c905ef32fe2`。

```text
CLEAN_HEAD_CLI = PASS
CLEAN_HEAD_TESTS = 202 passed
DEPENDENCY_CLOSURE = PASS
FORMAL_TRAIN_RUNS = 0
TEST_TOUCHED = NO
```
