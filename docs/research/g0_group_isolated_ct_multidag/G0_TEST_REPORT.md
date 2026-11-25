# G0 测试报告

2026-11-25 · MULTIDAG

## 验证覆盖

- 从正式五 seed 归档交叉验证 resolved config、运行清单、打包 SHA 及 30 轮 / 210 步日志。
- 从已认证的 Protocol B CSV 重算四折，复核原 Train 对话顺序、100 段 / 4,830 条、角色、类别支持、群组与对话隔离。
- 核对 J/K archive SHA、两份 manifest 的 F0 checkpoint 身份、J 的投影哈希与 Train/Dev tape，以及 K 的 4 个输出哈希。
- 验证 CPU 合成投影的 canonical 顺序、原特征和值、投影哈希、Dev 标签与身份保护、无效输入拒绝在反序列化前。
- CPU 合成数据经过完整 30 轮 F0 更新和离线回放，覆盖固定端点、零评价更新、所有逐样本预测和 logits、可靠性边界、重复 run 防护。
- clean candidate 中验证全部 CLI 入口、四折配置和定向回归。详见 [依赖闭包](G0_DEPENDENCY_CLOSURE_REPORT.md)。

隔离验证使用 Python 3.9、PyTorch 2.6.0 CPU、scikit-learn 1.6.1 和 pytest 8.4.2。覆盖 G0、POST-PH Protocol B、TAV 模型与运行时、Fusion 及正式五 seed 的定向用例。Windows sandbox 下测试进程的临时目录继承工作区 ACL。该适配仅存在于本地验收进程。

```text
WORKSTREAM_CHECK = PASS
CT_MULTIDAG_MAIN_ARCHITECTURE_CHANGED = NO
ORIGINAL_F0_CHECKPOINT_CHANGED = NO
GROUP_ISOLATED_PROTOCOL = PASS
SPEAKER_LEAKAGE = 0
DIALOGUE_LEAKAGE = 0
TEST_TOUCHED = NO
FORMAL_TRAIN_RUNS = 0
CLEAN_HEAD_CLI = PASS
CLEAN_HEAD_TESTS = 113 passed
DEPENDENCY_CLOSURE = PASS
READY_FOR_GPT_REVIEW = YES
```

## 真实运行边界

协议状态来自真实归档的冻结索引、SHA 及 CPU 合成数据闭环。认证投影字节留在 Ubuntu，实际加载验收由 runbook 的 `validate --projection-dir` 完成。此处无任何真实 fold 分数。真实 Test 没有打开。正式 F0 checkpoint 没有读写，本地可核验的是历史身份记录。
