# Q2-0F TA-v0 Formal Train/Dev Runtime

日期：2026-11-10。WORKSTREAM = MULTIDAG。

**TA_V0_FORMAL_RUNTIME_READY_FOR_REMOTE_SMOKE = YES**

该结论表示代码和 CPU synthetic 合同验证已具备远程有界 smoke 条件。真实资产训练、CUDA 执行、远程部署与正式 seed100 training 本轮均未启动。

| 字段 | 结果 |
| --- | --- |
| STATUS | PASS；独立 TA-v0 Train/Dev runtime、冻结正式配置、持久 checkpoint、Dev 导出与报告已实现并验证。 |
| WORKSTREAM_CHECK | PASS；指定 MULTIDAG 工作区，分支 feat/multidag-cl-paper-reimplementation，HEAD 5c4077fa159aef3e4adf119080a19ac8438c6bfa；origin fetch/push 均通过 guard。 |
| FILES_CHANGED | 2 个 tracked 修改，6 个新文件；详见文件清单。既有 baseline.yaml 和 dataset adapter 未修改。 |
| FORMAL_CLI | 必须显式 --formal-train-dev 及 --config、--asset-path、--model-manifest-path、--provenance-manifest-path、--output-dir；保留 --synthetic-dry-run。提供 --device、显式 --resume 和固定的一步 --smoke-one-step。 |
| TRAINING_PROTOCOL | seed100；30 epochs；16 dialogues/batch；AdamW(lr=0.0005, betas=[0.9,0.999], eps=1e-6, weight_decay=0)；clip norm=5；无 scheduler/class weight/label smoothing/curriculum/early stopping。Train shuffle=True，Dev=False；每个 epoch 使用独立的 seed100+epoch-1 generator。 |
| DATA_CONTRACT | 三个显式 portable 输入；universe=7433；仅构建 Train=4830/100 与 Dev=980/20。模型 manifest 为身份/split/target/row authority，provenance 为 canonical-order authority；输入只读。 |
| LOSS_CONTRACT | 仅 attention_mask=True 的有效目标参与普通 CE；检查非空有效目标、标签范围、有限 loss、所有训练参数的有限梯度；裁剪后才执行 optimizer step。epoch loss 按有效 utterance 数加权汇总。 |
| METRICS_CONTRACT | Train loss；Dev loss/accuracy/weighted_f1/macro_f1/uar；UAR 是固定六类 macro recall，空支持类别计零；六类 precision/recall/F1/support 与 6×6 confusion matrix。 |
| CHECKPOINT_CONTRACT | best_model.pt、last_model.pt、best_model_weights_only.pt；复用 ValidationCheckpointSelector 和 atomic checkpoint IO，排序只用 W-F1 降序、Dev loss 升序、epoch 升序。 |
| RESUME_CONTRACT | 仅显式路径；恢复前验证 schema/model config/label vocab/seed/完整训练协议/optimizer 实际 param-group config/data hashes 与身份；不一致硬失败。仅完整 epoch 边界可恢复，输出新 run，恢复 optimizer/global step/selector/Python/NumPy/Torch RNG，并保留历史 best。无自动 checkpoint 搜索。 |
| RUN_MANIFEST_CONTRACT | resolved_config.yaml 与 manifests/run_manifest.json；包含 run/git/device/architecture/参数量、三个输入路径与 SHA256、schema/version/counts/vocab/order policy、完整协议、选择规则、resume 来源、逐步更新的 optimizer 计数与状态；Test 两项计数恒为 0。 |
| EPOCH_LOG_CONTRACT | logs/epoch_metrics.tsv；规定字段全部具备，另有累计 optimizer_steps、train_valid_targets、dev_valid_targets；gradient_norm 表示本 epoch 最大裁剪前 norm。 |
| DEV_PREDICTION_EXPORT | 从 best checkpoint 严格 reload 后重新 Dev inference；predictions/validation_best_predictions.tsv 使用规定的 15 列、六类固定顺序。默认检查 980 行、exact unique ID set、无 padding/重复、有限输出、概率非负且和约为 1。 |
| H_TA_EXPORT | 同次 best-Dev inference 写 representations/validation_best_h_ta.npz 与 validation_best_h_ta.metadata.json；h_ta=[980,1500]，含 IDs/labels/split；metadata 含 representation 名称、维数、vocab、checkpoint SHA256、resolved config SHA256、split；两份导出的 ID set 与排列完全相同。 |
| TEST_ISOLATION | loader(test)、evaluate(split=test)、evaluate_test、test_loader、test_predictions、export_dev(split=test) 在 factory/批次迭代/特征取得前抛错；不存在训练结束后的 Test 分支，Test reports/predictions 为 0。包级 Test 身份元数据仅用于现有 adapter 的一致性校验。 |
| NEW_TESTS | 新增 47 项，覆盖 A–O、非有限梯度拒绝、协议拒绝、真实 CLI 的 synthetic 注入接线、完整 7433 行 synthetic 身份校验、split count 变更拒绝及默认 980 行导出。 |
| TEST_RESULTS | 修改前 125 passed in 3.54s；最终 172 passed in 13.53s；保留的 train/export synthetic CLI 均 PASS；6 个变更/新增 Python 文件 AST、scoped diff whitespace 检查通过。 |
| MODEL_FILES_CHANGED | NO；ta_v0 与 paper_reimplementation 共 12 个 Python 文件 SHA256 与任务开始时一致。 |
| MODEL_MATH_CHANGED | NO；encoders、graph、attention、GRU、fusion projection、h_ta/classifier/vocab 与模型 state_dict schema 均未修改。 |
| REAL_DATA_USED | NO；所有本地数据由测试构造，没有读取或复制真实 IEMOCAP package。 |
| GPU_USED | NO；现有 m3ed_mmgcn 环境 torch=2.6.0+cpu；未安装依赖。 |
| FORMAL_TRAINING_STARTED | NO；正式 CLI 接线测试只注入 synthetic dataset，未进行真实训练。 |
| OPTIMIZER_STEPS_REAL_DATA | 0；optimizer step 仅发生在 CPU synthetic tests。原 --synthetic-dry-run 仍为 0 optimizer steps。 |
| GIT_OPERATION_PERFORMED | 仅本地只读 rev-parse/branch/remote/status/show/diff 等；无 add/stage/commit/push/pull/stash/restore/reset/clean，无分支或 remote 修改，无远程操作。 |
| KNOWN_LIMITATIONS | 真实 portable package 的训练、真实 980 行 best-Dev 导出、CUDA 数值与确定性、30 epoch 资源占用及性能须在后续远程任务验证。本轮 980 行导出使用完整规模 synthetic Dev，不是实测真实表现。只声称 canonical utterance-order causal；上游特征因果性、overlap 与 wall-clock streaming 不在本轮认证范围。 |
| SCOPED_GIT_STATUS | 2 个 M，6 个新文件；Git 将 runtime 的 3 个新文件折叠为一个 ?? 目录。无关 dirty/untracked 状态及 .gitignore 哈希与任务开始时一致。 |
| NEXT_SAFE_STEP | 先审查本 patch；在获得精确 Git 交付授权的后续任务中提交并同步，再 Ubuntu guard/status → pull --ff-only → 显式 one-step Train/Dev smoke → checkpoint/reload smoke → 独立正式 seed100 training。当前回合不执行这些 Git/远程/真实训练动作。 |

## 文件清单

修改：

- scripts/models/multidag_cl/ta_v0/train.py
- scripts/models/multidag_cl/ta_v0/export_dev.py

新增：

- scripts/runtime/multidag_cl_ta_v0/__init__.py
- scripts/runtime/multidag_cl_ta_v0/training.py
- scripts/runtime/multidag_cl_ta_v0/formal.py
- configs/multidag_cl/ta_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev.yaml
- tests/models/multidag_cl/ta_v0/test_formal_runtime.py
- docs/reproduction/TA_V0_Q2_0F_FORMAL_RUNTIME_REPORT.md

实现只复用旧 runtime 的 validation selector、checkpoint IO 和公共 classification metrics；没有复制或调用整个历史 trainer，也没有引入其 formal-run Test 规则。baseline.yaml 仍为原 synthetic 配置，新正式 YAML 的模型/数据/训练协议逐字段冻结。

## 正式 CLI（仅供后续远程任务执行）

以下参数路径均由调用者显式提供。output-dir 是 canonical experiment group root；runtime 自动在 runs 下分配包含 ta_v0、seed100、带时区 timestamp、短数据 identity 的 run_id。最终 run 目录使用 exist_ok=False 原子声明，碰撞在 epoch 前失败；已有 run 不覆盖、不删除。

```bash
python -B scripts/models/multidag_cl/ta_v0/train.py \
  --formal-train-dev \
  --config configs/multidag_cl/ta_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev.yaml \
  --asset-path <package_root>/ta/iemocap_asym_ta_v1.pkl \
  --model-manifest-path <package_root>/metadata/model_input_manifest_v1.csv \
  --provenance-manifest-path <package_root>/metadata/iemocap_asym_manifest_v1.csv \
  --output-dir outputs/<YYYYMMDD>/ta_v0_formal \
  --device cuda
```

有界远程 smoke 在同一命令末尾增加 --smoke-one-step：仅一个 Train batch 的 optimizer step，随后完整 Dev evaluation、checkpoint 写入、best reload 与 Dev 导出。manifest.status=smoke_completed，training_finished=false，checkpoint.epoch_complete=false；该 partial-epoch checkpoint 只能供 inference/reload 检查，不能作为正式 resume 起点。

正常 resume 显式增加 --resume <checkpoint>，且不带 --smoke-one-step。只接受可信的本项目结构化 checkpoint；last 内嵌一份独立历史 best payload，避免恢复到 last 时把 last 权重误标为 best。数据路径可搬迁，数据 SHA256/版本/数量/order policy 必须完全一致。resume 不回写原 run；新 run 的 epoch log 记录后续 epoch，manifest 同时记录累计 step 与本次 step。

best_model_weights_only.pt 仅保存原模型 state_dict，包括模型本来就有的 _extra_state 身份字段；没有外加 optimizer 或 wrapper。strict reload 的 synthetic logits 与 h_ta 已逐元素零容差验证。完整 checkpoint 保留 optimizer 与 RNG，因此大于 weights-only 文件。

## 本地验证

测试关闭 bytecode 与 pytest cache，使用新建系统临时目录，CUDA_VISIBLE_DEVICES 为空。首次 sandbox pytest 因既有 Windows 临时目录 ACL 失败；在自动批准的相同环境执行后通过，未更改源码或依赖来规避该问题。

```text
python -B -m pytest -q tests/data/test_ta_v0_dataset.py tests/models/multidag_cl/ta_v0 tests/paper_reimplementation/multidag_cl/test_encoders.py tests/paper_reimplementation/multidag_cl/test_graph.py tests/paper_reimplementation/multidag_cl/test_attention_and_gru.py tests/paper_reimplementation/multidag_cl/test_model.py -p no:cacheprovider --basetemp <fresh-system-temp-dir>
172 passed in 13.53s

python -B scripts/models/multidag_cl/ta_v0/train.py --synthetic-dry-run
status=PASS, device=cpu, optimizer_steps=0, training_started=false, test_evaluation_count=0

python -B scripts/models/multidag_cl/ta_v0/export_dev.py --synthetic-dry-run --output-dir <fresh-system-temp-export-dir>
rows=4, unique_utterance_ids=4, duplicate_ids=0, padding_rows=0, dimension=1500

python -B scripts/models/multidag_cl/ta_v0/train.py --help
PASS；显式模式与各路径/设备/resume/smoke 参数可见。
```

完整规模 synthetic package 使用 4830/100 Train、980/20 Dev、1623/31 Test 身份元数据；Test 的 feature 字段是不可用于训练的占位值，Train/Dev 初始化仍通过，证明没有验证或取得 Test feature arrays。默认 980 行 exporter 测试使用真实冻结 TAV0.forward、CPU、无 optimizer step，并检查 [980,1500]、finite、prob sum 与 exact ID ordering。

## git diff --stat 与 scoped status

```text
 scripts/models/multidag_cl/ta_v0/export_dev.py | 31 ++++++++++++++++-----
 scripts/models/multidag_cl/ta_v0/train.py      | 37 +++++++++++++++++++++-----
 2 files changed, 55 insertions(+), 13 deletions(-)
```

该 scoped diff 不含未跟踪新文件；没有用 git add -N 或其他方式修改 index。完整工作区 diff 中的历史文档删除与 .gitignore 既有修改均为任务前状态，没有恢复或混入本轮修改。

```text
 M scripts/models/multidag_cl/ta_v0/export_dev.py
 M scripts/models/multidag_cl/ta_v0/train.py
?? configs/multidag_cl/ta_v0/iemocap/causal_context/asymmetric_merc_v1/formal_train_dev.yaml
?? scripts/runtime/multidag_cl_ta_v0/
?? tests/models/multidag_cl/ta_v0/test_formal_runtime.py
?? docs/reproduction/TA_V0_Q2_0F_FORMAL_RUNTIME_REPORT.md
```

**TA_V0_FORMAL_RUNTIME_READY_FOR_REMOTE_SMOKE = YES**
