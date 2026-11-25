# G0 依赖闭包

2026-11-25 · MULTIDAG

## 候选树

基线 HEAD 为 `521886c1bb8048f1475d865bb4352285a0937463`。候选归档规则为 Git HEAD 的跟踪文件加 [精确新增白名单](REQUIRED_MODIFIED_FILES.txt)。[既有 dirty 清单](UNRELATED_DIRTY_FILES.txt) 排除于覆盖层，已跟踪的既有脏文件在归档中使用 HEAD 字节。

| 调用面 | 来自 HEAD 的主要依赖 |
| --- | --- |
| 模型工厂 | `scripts/runtime/multidag_cl_fusion/{spec,runtime}.py`、`models/multidag_cl/fusion/`、`models/multidag_cl/graph_core_variants/track.py`、formal frontend YAML |
| 投影与模型输入 | `pairwise_preflight_inputs.load_projected_train_dev`、`datasets/iemocap/tav_full_audit_dataset.py` |
| Protocol B 元数据 | `scripts/analysis/models/multidag_cl/post_ph/internal_protocol.py` |
| 更新、分桶、保存 | `AuditRuntime`、`multidag_cl_ta_v0.training`、`multidag_cl_paper_reimplementation/{curriculum,checkpoint}.py` |
| 指标、身份、目录 | POST-PH `metrics.py`、Round 1 `failure_audit_metrics.py`、`multidag_cl_ta_v0.formal`、`utils/output_paths.py` |

真实数据为 Ubuntu 已有认证投影，G0 的独立身份锚来自 J/K 回传并已置于四折清单。旧正式 F0 权重、诊断临时产物和 Windows 非跟踪文件均不参与运行依赖。

## 隔离验证

从 `git archive` 核对 885 个 HEAD 文件，逐字节叠加 17 个新增文件。候选 tar 与展开目录的逐成员 SHA-256 一致。94 个既有 dirty entries 未用于组装覆盖层。候选目录无 Git 元数据，Python 清除 `PYTHONPATH` 并关闭用户 site。直接执行主 CLI 的 help、三个子命令 help、四个 metadata-only validate 及六个 pytest 模块，校验实际导入的仓库模块均来自展开目录。

```text
ARCHIVE_MEMBERSHIP = PASS
CLEAN_HEAD_CLI = PASS
CLEAN_HEAD_TESTS = 113 passed
DEPENDENCY_CLOSURE = PASS
```

候选 tar 整包 SHA-256 和原始命令输出保存在归档旁的 `G0_CLEAN_CANDIDATE_VALIDATION.txt`。真实 GPU 运行依赖符合项目环境约定的 PyTorch、NumPy、PyYAML、scikit-learn 及其依赖。验证未改变包环境。Ubuntu 投影字节核验仍是 CUDA 开训前的独立步骤。
