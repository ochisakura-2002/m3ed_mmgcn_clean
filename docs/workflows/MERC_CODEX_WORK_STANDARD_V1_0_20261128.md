# MERC / CT-MultiDAG Codex 长期工作规范 v1.0

日期：2026-11-28｜WORKSTREAM=MULTIDAG｜用途：长期工程工作规范、项目源补充、Codex 任务验收依据。

状态：用户已同意推进工程治理。本文件在用户加入项目源并交给 Codex 后作为工作规范使用；并不表示仓库已经修改或门禁已实现。具体当前修复见配套 `CODEX_PIL_PREFLIGHT_FIX_AND_STANDARD_ADOPTION_20261128.md`。不替代科研任务授权，不自动批准训练、重构、Test、依赖安装或 Git 写入。未来修订需版本化（v1.1/v2.0），记录差异，不静默覆盖。

## 0. 核心原则

- 科研目的优先：真实分类性能、可比较性、因果推理、轻量部署及可复现研究。不要为了工程形式追求复杂度。
- 一次任务应具备完整闭环：研究合同→源码接口核查→实现→本地验收→用户同步→真实远端预检→授权实验→分析归档→GPT科研裁决。
- 代码写完、语法通过、pytest 通过、正式就绪、训练结束、实验完成、科研方法成功是不同状态；禁止互相替代。
- 先核查后调用。接口真实存在、实际数据合同匹配、执行分支得到验证才是依据。不得凭文件名、旧记忆、模型生成的猜测判断。
- 历史冻结不等于永远禁止研究相同领域；已关闭的是具体方法。新候选允许改变任何经批准的内部计算，但原基线及证据不可被悄悄改变。
- 先低成本消除明确工程风险，再用有价值的实际实验判断科研假设。不得为通过门禁放宽科研标准，不得为完成报告伪造数据、PASS、命令或路径。

## 1. 权威来源与任务边界

### 1.1 来源分层

用户最新明确授权、经批准的本轮科研合同和数据/安全边界优先；本规范负责执行方式，不自行变更科研合同。以下文件共同构成项目源（实际存在位置先核查）：

1. 本规范及配套当前任务；`CODEX_MERC_PIL_C01_C07_SCREEN_IMPLEMENTATION_20261128.md`（当前已批准首筛，覆盖旧文件中尚未授权的状态）；`MERC_PIL_RESEARCH_002_REVIEW.zip` 内 07/08 详细设计及实验计划。
2. `MERC_GPT_PROJECT_SOURCE_UPDATE_20261127_FINAL_D_STAGE_CLOSE.md`：D/G关闭、F0对照、历史身份及证据。
3. `MERC_GPT_PROJECT_SOURCE_UPDATE_20261125_POST_G4_LEARNING_FREEZE.md`、`MERC_GPT_PROJECT_SOURCE_UPDATE_20261125_POST_PH_JOINT_AUDIT_G0_REBASE.md`：G、联合诊断、数据与工程治理。
4. `fuision阶段项目源.md`、`MERC_GPT_PROJECT_SOURCE_UPDATE_20261119_GRAPHCORE_FREEZE_FUSION_REBASE.md`、`MERC_GPT_PROJECT_SOURCE_UPDATE_20261113_POST_GATE_REBASE.md`、`MULTIDAG_CL_BASELINE_FREEZE_20260818.md`：历史架构、路径、冻结结果。
5. `实验回传规范.md` v1.0：继续有效，本规范补充工程门禁，不减免其图表、原始数据、身份、HOLD及归档要求。
6. 本轮 `IMPLEMENTATION_REPORT.md`、`TEST_REPORT.md`、`RUNBOOK.md`、实际 terminal 日志与 Git 记录：区分报告声明与独立运行证据。

不同阶段的源文件只覆盖过期状态，不覆盖原始观测。冲突时列出文件、字段、实际证据、影响与拟处理方式；关键冲突停止受影响工作，不自行选择更宽松版本。每次任务只读相关来源，维护简洁来源索引，避免重复扫描整个仓库。用 FACT / INTERPRETATION / HYPOTHESIS / PLAN / AUTHORIZATION 标记事实、解释、假设、计划、授权。

### 1.2 工程工作区与权限

| 项目 | 固定身份 |
|---|---|
| Windows 工作区 | `E:\MERC\m3ed_mmgcn_clean` |
| Ubuntu 唯一授权工作区 | `/home/zhiyuan/research_lh/m3ed_mmgcn_clean` |
| 分支 | `feat/multidag-cl-paper-reimplementation` |
| GitHub | `https://github.com/ochisakura-2002/m3ed_mmgcn_clean.git` |
| Workstream | `MULTIDAG` |
| Conda | `m3ed_mmgcn` |
| Windows 已知环境 | Python 3.9.25 / Torch 2.6.0+cpu（以实时核验为准） |
| Ubuntu 已知环境 | V100 16GB / Python 3.9.25 / Torch 2.6.0+cu118（历史环境，以预检核验为准） |

开始前核查 `git rev-parse --show-toplevel`、`git branch --show-current`、`git remote -v`、`git status --short --branch`、`git diff --cached --name-status`、`python scripts/dev/check_workstream.py multidag`，记录 HEAD、解释器与本轮依赖环境。Git 输出优先 `git --no-pager`。不跨工作流或在授权目录外扫描、写入、删除。已批准的外部数据路径和历史资产仅限目的明确的只读访问。用户执行同步及正式训练；Codex 默认 Windows 实现与 CPU 验收，不擅自远程运行 GPU。若用户另行明确授权 Remote-SSH 操作，仍遵守唯一授权工作区和任务范围。

禁止 `git add .`、自动 commit/push/reset/clean/stash/switch、盲目 pull/push、覆盖旧 run、删除已有文件、全局改环境、擅自安装依赖。保留 unrelated dirty/staged/untracked；如果共享文件存在无关改动，停止该文件修改并给出隔离办法。只读检查不应污染旧资产；新临时产物在仓库已批准 tmp/artifact 目录隔离，按任务 ID 命名，不自行清理旧目录。历史源码的哈希策略、源码闭包及环境门禁不得为迁移方便而改写。

## 2. 科研计算合同

现阶段基线为 CT-MultiDAG F0：T1024→Linear100→单向 LSTM100，A1582/V342→Linear100+ReLU，A/V/T DirectConcat 300D，strict-past all TrackCore 四步，独立读出 h0..h4 1500D，MLP1500→300→300→6，dropout .4，CL_OFF，1,638,306参数。四层读出不反馈到 Track 传播递推；正式模型身份以 `paired_model`→`FusionTAV` 及 resolved config 为准，不凭 registry 名或摘要猜测。

- 主协议 ORIGINAL_CT_MULTIDAG_DEV；原 F0 CE Best；类别 hap,sad,neu,ang,exc,fru。原 Train 4830/100，Dev 980/20，Test 1623/31 HOLD。原 Train/Dev 话语及对话无交集，但存在 Ses04 两演员交集；G 的 session-isolated OOF 仅用于相应协议，不跨口径算涨幅。
- Train/Dev-only 投影 `/home/zhiyuan/research_lh/data/IEMOCAP/asymmetric_merc_v1/tav_train_dev_projection_v1`，`train_dev_tav.pkl` SHA `a9f23d0fb2b59abf156222a93e29896a203cf4948908ec3bbbb99cf37744c466`，`model_input_manifest.csv` SHA `9a474e94539154ba4ed72a17b6815c07b9e7b5c12605d71a5e63900744366c43`，另有 `train_dev_provenance.json`，需按原 validator 核验身份、顺序、字段、shape、dtype、split。禁止加载混合 Test pkl 或读取 Test 实际样本；禁止构造真实 Test loader、评价或选点。保留原有全部 HOLD 计数为0；不因新测试或导出而访问 Test。历史 Test 曾被访问，不宣称未见独立测试。
- 因果边界限于 supplied features 到 logits；上游抽取来源和实时可见性未认证为 UNKNOWN。必须验证 padding、同/跨说话人关系、对话隔离、未来扰动、前缀追加/截断、缺失语义。不能把 causal inference 与训练时的额外可见信息混淆。
- 正式实验开始前锁定数据、臂、seed、budget、初始化、采样带、超参、损失、选择规则、评价和停止条件。不得临时改 LR、epoch、λ、阈值、数据人口、预算等救火。当前 PKL 冻结，提取/新数据/新依赖另行审批。
- 当前 PIL 首筛 seed100，30 epoch/210 updates，每批16对话，AdamW lr5e-4、betas(.9,.999)、eps1e-6、wd0、foreach=False、clip5，hard CE，原 CL_OFF 私有随机顺序，无 scheduler；Best 为最大 Dev W-F1→最小硬 CE→最早 epoch。共享初始化/RNG/tape 以及新参数的 RNG 隔离是配对条件。F0 D0 E21 693/980、ACC70.714286%、W-F1 69.962663%。新候选从共同随机初始化训练，不从已训练 Best 微调，除非另批明确设计。
- C01 CURRENT/PREV 与 C07 NULL/TRUE 四臂批准；必要时仅一条共享 OFF，最大1050 updates；其余候选、组合、多seed不在本批授权。C01/07数学及门限见正式任务书，不由本规范重定义。
- 方法发现以信息处理、信息融合、信息学习为开放框架，沿任务需求→真实证据→一般计算问题→成熟技术→适配及可证伪设计。不要因用户举例而锁死领域。关闭 G/D 标准学习扩展和既有负结果，不自动重启，不把一般机制推测当根因。

## 3. 每项任务启动合同（先于实现）

在工作区已有任务管理位置建立/更新简洁任务记录，路径核查后决定，不强制新增重复框架。至少包含：

```yaml
TASK_ID: unique
TASK_TYPE: research | implementation | compatibility_fix | diagnostic | analysis | governance
AUTHORIZED_ACTIONS: []
FORBIDDEN_ACTIONS: []
SOURCE_OF_TRUTH: []
BASELINE_IDENTITY: actual_or_pending
REQUIRED_MODIFIED_FILES: []
UNRELATED_DIRTY_FILES: []
EXPECTED_INTERFACES: []
EXPECTED_EXECUTION_PATHS: []
DATA_VISIBILITY: train_dev_only_or_more_restrictive
TEST: HOLD
ACCEPTANCE: []
STOP_RULE: []
```

记录当前 stage，必要时复用已有任务书而非重抄。科研计划未批准时只做研究设计；兼容性修复不得改科学变量；工程治理不得隐式扩展实验授权。实现前逐项核实引用的真实路径和入口。发现既有实现与任务书不符，不能自行改变合同，提出最小修订并回审。

## 4. 门禁 A：接口真实性与依赖闭包

### 必须执行

- 为新增/修改代码跨模块引用建立 `INTERFACE_CONTRACT` 清单：调用文件/行号、真实定义文件及符号、导入路径、函数签名及关键默认参数、返回结构、关键字段/类型/单位、调用分支、测试位置。对 `import X as d` 等 alias 的每个实际属性调用核实目标模块是否真的导出；不要将 Workflow 当作 Runtime 的全量代理。
- 从真实模块直接导入稳定定义或明确适配层。不得复制未知常量、以默认值吞掉字段缺失、宽泛捕获 AttributeError/KeyError 伪装成历史资产缺失。确有可选字段时按版本合同处理并记录。
- 使用 AST 或等价现有工具枚举相关模块的属性/直接导入、核验真实属性及可检查签名；动态访问、继承、延迟导入、平台分支等列出不能静态确定的项目，必须用实际调用补测。检查脚本本身不应启动训练或读取真实数据；导入若有副作用先隔离/审查。
- `py_compile`、实际模块导入、真实 CLI `--help`、配置解析、定向静态检查。已有 Pyright/Pylint/Ruff/coverage 若可用可复用；不得擅自安装。Ruff 不能保证发现所有模块属性错误，AST/静态检查也不能代替集成验证。明确工具局限。
- 记录实际源码 SHA、依赖闭包、模块导入顺序及同环境身份。使用候选提交树/隔离 git archive 或等价显式候选快照确认仅所交付文件即可运行（未提交文件需明确装入候选快照），不能依赖本地 unrelated dirty 或缓存。不得绕过历史兼容策略。

门禁通过条件：必要符号真实存在、实际合同匹配、无未解释关键依赖、候选交付树自足；否则 `INTERFACE_CONTRACT=BLOCKED`。简单存在性测试只是第一层。

## 5. 门禁 B：历史资产合同与集成 Fixture

将历史资产接口作为稳定的数据合同，避免每个新实验重新猜测上次 CSV/TSV/JSON 结构。

- 优先采用经 SHA 认证的历史轻量归档作真实只读集成测试，例如 D Stage A/B。无需训练或复制权重/PKL。只读合法 Train/Dev 表；不把真实逐话语预测、特征、权重或大归档默认加入 Git。
- 如果本地缺资产，使用与真实表结构、字段名称、列类型、键、命名、排序、人口和完整性规则一致的小型 synthetic fixture。保留测试证书与真实历史证书隔离，按 fixture 自身计算 SHA；不可假造历史 SHA。字段/结构来源在 `SCHEMA.md` 或既有合同文档中记录。正式认证常量不改，测试可注入独立的预期身份，不能 mock 核心解析和校验。
- 真实归档保留外部本地只读、manifest 与哈希引用；可提交小型合成夹具、schema、测试代码。根据项目现有目录规划，例如 `tests/fixtures/` 下版本化合同（示例，不代表已存在）。不要因制作夹具覆盖原始资产。
- 至少覆盖合法完整输入、坏 SHA、缺/重复/错字段、错误人口/ID/顺序、非数值/非有限值、错误 Best/E30、缺 epoch/重复 epoch、非法 split/HOLD、历史资产确实缺失时的授权 fallback、配方身份错误时的拒绝。精确数值合同用未舍入值，不能从展示表格倒算。
- 对本轮实际读取函数使用真实调用。测试可替换外层磁盘路径、不可用 GPU 边界或 fixture 证书输入，但须明确哪些真实动作未覆盖；不得把整个核心认证函数 mock 成 PASS。
- 归档读取失败必须分类：ENGINEERING_ERROR（代码缺陷）、ASSET_UNAVAILABLE（确实缺失）、IDENTITY_MISMATCH（身份/科学门禁不符）、UNSUPPORTED_SCHEMA（无法兼容）；仅在任务明确允许且共同配方已认证的资产缺失场景执行共享 CTRL fallback。未知异常不能自动视为可补跑基线。

## 6. 门禁 C：本地功能与关键路径集成验收

每个任务按实际入口生成 `EXECUTION_PATH_COVERAGE` 表：路径/分支、真实函数、用例、输入来源、预期、实测、未覆盖原因、远端补证动作。需要证明关键路径确实走到，不能只给总测试数。按本轮适用范围覆盖：

| 路径 | 关键验收 |
|---|---|
| CLI及配置 | plan、help、参数缺失/非法、配置身份与输出目录 |
| 历史认证 | 真实读取与比较；reuse、允许的 fallback、拒绝路径 |
| 数据与初始化 | 授权 split、manifest、mask、参数数目、共享权重及 RNG/tape |
| 模型 | 数学/维度、OFF等价、因果、缺失、padding、会话隔离、梯度/有限值 |
| 训练与评价 | 合成最小训练走同 runtime、实际 loss 分母、clip、optimizer、selector、预算 |
| 续训 | 完整 epoch 原子 checkpoint、RNG/optimizer/辅助状态、与连续运行配对 |
| 分析与发布 | 指标从预测复算、CSV回读、bootstrap、图片、归档清单/哈希、失败后恢复 |
| 故障注入 | 错身份、缺资产、重复启动、进程中断、无效候选及部分完成状态 |

模型级单测、接口级单测、集成测试各有用途。必须针对每个真实报错补 regression，优先证明修复前失败、修复后成功；若无法重放旧代码，说明原因并构造同调用路径测试。覆盖率工具可用时附关键模块行/分支覆盖，不能用全仓百分比掩盖关键缺口。未执行的路径标记 NOT_RUN 或 REMOTE_REQUIRED，禁止标 PASS。记录 pytest 命令、通过/失败/跳过/警告及原因；固定源码后复跑，测试期间不改代码。

CPU 合成 smoke 不是正式实验；正式预算不可用 smoke 数据、步数或配置代替。Windows/V100 同环境比较与跨平台数值等价要分开，不宣称 CPU/CUDA 逐位相等。环境/测试临时目录 ACL 等问题优先复用仓库批准目录，不擅自改系统权限。

## 7. 门禁 D：用户同步与 Ubuntu 真实预检

本地通过只允许 `LOCAL_IMPLEMENTATION_PASS` / `REMOTE_PREFLIGHT_PENDING`，不是 READY_TO_TRAIN。提交前给出 REQUIRED_MODIFIED_FILES、UNRELATED_DIRTY_FILES、实际 diff/候选快照和测试报告。用户检查暂存区为空，逐个真实路径 `git add -- ...`，核对 staged 清单及 `git diff --cached --check`，用户手动 commit/push 到指定分支。若非空或含未批准改动，停止。记录完整 SHA。

Ubuntu 使用 `git fetch origin refs/heads/feat/multidag-cl-paper-reimplementation`，核查待合入提交/文件，`git merge --ff-only <用户批准SHA>`，并检查实际 HEAD=批准 SHA、workstream 和文件存在。每条关键命令显式检查返回码，失败不继续；不以旧缓存 FETCH_HEAD 误合入，不强制覆盖。所有正式指令由 Codex 根据实现的 CLI 生成，不能虚构脚本。

远端先真实 preflight：

- 校验当前 worktree/HEAD、源码兼容策略、配置、Python/Torch/CUDA/driver/GPU及原配方环境；CUDA_VISIBLE_DEVICES、CUBLAS_WORKSPACE_CONFIG、线程/MKL等按原认证合同，不随意调整。
- 认证 Train/Dev projection、历史基线、必要权重、epoch/预测/manifest及原容差，验证原数据人口、采样带和初始化，测试同 dtype 前缀因果、配对及导出依赖。
- 明确预检是否触及真实数据/执行合成 GPU 计算，禁止 optimizer update 或未经授权的真实推理；如原任务明确授权少量预检推理，严格记录权限和范围。
- 记录预检回执：task/date/batch/HEAD、配置和源码 SHA、投影身份、历史证据 SHA、环境、分支结果、退出码、日志路径。只凭文件存在、HEAD、摘要分数或退出0均不足以训练。
- 预检成功后才允许用户启动原批次授权实验。PARTIAL 必须明确哪些臂可运行、哪些 INVALID；当前按 GPT 已指示遇 PARTIAL 先回审。预检失败保存现场，不自动训练，不吞异常，不自作主张调科学配方。

## 8. 门禁 E：持久化执行、状态与恢复

目标是减少交互命令和 Shell 故障，不要求现在重构全仓。

- 交互式 bash 不开启长期 `set -euo pipefail`；独立脚本可以开启并用明确 trap/退出码保存失败证据。用 `git --no-pager` 避免误入 less。管道 `tee` 后立即捕获 `${PIPESTATUS[0]}`；日志失败也需报告，不能只读 tee 的退出0。
- 每轮保存实际 experiment_date、batch_id、commit SHA、projection、CLI、config SHA、日志/状态/归档路径及预检回执。建议使用已有 state/manifest；不足时新增轻量持久化运行描述符，在授权仓库内写入，避免依赖临时 shell 变量。重连只读取描述符或明确输入原 batch ID，不自动生成新批次。
- 启动前检查相同批次进程、已有状态、GPU资源及锁；不要根据 PID 单独判断身份，结合命令行/批次/状态；互斥锁及原子发布须测试。一次只启动批准的完整执行入口，nohup 或已批准已有机制均可。不自动重复训练，不默认杀其他进程，不清理历史锁/目录。
- 恢复区分 resume-training（最近完整 epoch、模型/优化器/RNG/tape/Best 等）与 resume-analysis（不重复训练）。完成/INVALID 的臂保持证据；只有兼容的原批次状态可恢复。若源码版本变更，须核查恢复合同；不能悄悄用新 HEAD 接续旧身份。未开始训练的预检失败可经明确验证复用批次，否则建新批次并记录 supersedes 关系，保留原失败日志。
- 未来可以开发一个薄 wrapper 统一 `preflight/run/status/resume/collect`，内部仍调用真实历史 CLI，不重复实现科研计算。先设计并测试，再逐步引入；本规范中的命令名是目标能力，不是已经存在的程序。

## 9. 门禁 F：归档与科研裁决

沿用《实验回传规范》v1.0：`TRAINING_COMPLETE` 不等于 `EXPERIMENT=COMPLETE`。每个正式实验必须导出完整 plot-ready 原始结构化数据，不依赖日后重新训练。所有数据字段、计算单位、身份、必要对照以实际任务合同为准。

- manifest、resolved config、Git/源码 SHA、运行环境、数据/人口、seed/budget/optimizer steps、RNG/tape、Test HOLD 全部计数、Best/E30、checkpoint 路径及 SHA、日志/状态/失败原因。
- 每轮 Train(Fit)/Dev 的 ACC/W-F1/Macro/UAR/hard CE，实际 update objective 单列；固定权重 Fit 与训练步内 loss 不混。无独立训练的后处理方法不虚构曲线。
- 同一 Best 逐话语 ID/dialogue/speaker/position/true/pred/logits/prob、原始/归一化混淆矩阵、逐类别 P/R/F1/support、配对错误迁移及必要分组、机制统计；gold 分组只用于分析，不输入模型。必要校准指标、bootstrap、运行时间、参数/GPU开销等按科研任务；区别同协议与跨协议，未运行标 NA+原因。
- CSV/JSON/必要 NPZ、绘图脚本、标准独立 ACC/W-F1/hard Loss/训练目标曲线 PNG+SVG、图表索引及中文说明。输出图可重绘，原始表优先于图片。归档前复读表、检查人口和预测一致性、重算主指标、核对文件及 SHA，验证压缩包可解压。大权重/PKL/原始音视频/cache 留在远端，不纳入轻量回传包。
- 成功条件：`ANALYSIS_EXPORT=COMPLETE; ANALYSIS_BUNDLE=PASS; PLOT_READY=YES`；缺必要项则 `ANALYSIS_ARCHIVE_PENDING`，不能声称实验完成。对于失败/INVALID/零正式训练，按真实状态提供诊断包而非虚构预测或曲线。科研 GO/REVIEW/STOP 与工程 PASS/FAIL 分离。候选的首筛结果不是最终创新成立，不自动扩展训练。
- 用户回传轻量包后，GPT 独立审核并给出科研裁决、汇报 MD、项目源增量更新；Codex 可提供草案，不自行宣布正式冻结或修改 ChatGPT 项目设置。

## 10. 标准交付与状态报告

每次实施任务输出中文报告（沿用已有 docs/research/<task> 布局，先核实），至少：

1. `IMPLEMENTATION_REPORT.md`：任务授权、计算合同、接口清单、实际变更、历史保护、风险与待验证。
2. `TEST_REPORT.md`：实际测试命令、结果、关键路径覆盖表、真实/合成/Mock 边界、回归、未验证、远端要求。
3. `RUNBOOK.md`：真实入口、精确 Git 文件清单、用户手动同步、预检、成功条件、一次启动、监控、恢复、SCP、SHA；尽可能少手动变量。
4. 可合并附录/机器可读 `INTERFACE_CONTRACT`、`EXECUTION_PATH_COVERAGE`、`SOURCE_AND_ASSET_MANIFEST`，清单不必为形式新增大量文档。
5. 正式实验后的 Plot-Ready Bundle、真实路径/大小/SHA 和结果摘要，按第9节。

统一状态字段建议（不是已存在代码 API）：

```yaml
TASK_ID: ...
SOURCE_IDENTITY: PASS | BLOCKED
INTERFACE_CONTRACT: PASS | BLOCKED | NOT_RUN
LOCAL_UNIT_TEST: PASS | FAIL | NOT_RUN
LOCAL_INTEGRATION_TEST: PASS | PARTIAL | FAIL | NOT_RUN
REAL_HISTORY_ASSET: PASS | REMOTE_REQUIRED | BLOCKED
SYNC: PENDING | PASS | FAIL
REMOTE_PREFLIGHT: PENDING | PASS | PARTIAL | FAIL
TRAINING: NOT_STARTED | RUNNING | COMPLETE | INVALID | INTERRUPTED
ANALYSIS_EXPORT: PENDING | COMPLETE | INCOMPLETE
ANALYSIS_BUNDLE: PENDING | PASS | INCOMPLETE
PLOT_READY: YES | NO
SCIENTIFIC_VERDICT: PENDING | GO | REVIEW | STOP | INVALID
TEST: HOLD
BLOCKERS: []
UNVERIFIED_PATHS: []
NEXT_AUTHORIZED_ACTION: ...
```

状态只根据本阶段真实证据填写；不存在的字段不可假装已接入。每次发现阻断明确给出最小修复方案，保护已完成轨迹，不通过大范围重构拖延科研。任务报告需能让用户直接按顺序操作。

## 11. 工程治理路线（分期而非立即全量实施）

**A：当前 PIL 修复（立即）**。修正错误接口、完整审计 D 调用、补真实历史读取回归、引入本轮接口与路径覆盖清单，真实远端预检后才训练。只改 PIL 范围及对应测试/报告，先保存失败记录。

**B：PIL 结束后**。审计现有 `AGENTS.md`、工作流文档、测试及公共接口，提出最小落地：短版强制规则+链接本规范，通用接口检查与真实 schema fixture，复用已有基础设施；禁止重复建立整套框架。用户审核实际落地路径后实施（不自动开启）。

**C：下一批较大实验前**。评估持久化实验入口、统一身份/指标/归档契约与薄适配层。新实验逐步使用明确版本的公共接口，历史 Workflow 保持可复现。先在新实验做隔离对照，验证旧 OFF/历史认证不变，再推进。全仓迁移、重写 frozen runtime、依赖安装、自动 CI/远程操作均另行审批。

该分期既减少可提前发现的错误，也承认 GPU 与真实资产门禁无法全部在本地模拟。工程治理的验收不是从此零报错，而是接口错误能提前暴露、失败有清晰身份和可恢复记录、不污染科学结论、用户不必反复手动排障。

## 12. 当前事件（索引，随任务推进更新而不重写历史）

截至 2026-11-28：PIL 初始实现 b6a6938 已由用户 fast-forward 同步；Windows 报告111 passed/专项18项，真实历史认证未完整覆盖。远端 preflight 在 `read_history_archive` 中访问不存在的 `d.CLASSIFICATION` 后 AttributeError，退出1；四处同类引用已定位，真实定义 `scripts/runtime/multidag_cl_fusion/original_dev_alignment.py:43`。代码/数据/环境认证尚未全部完成，正式训练尚未开始。按配套修复任务执行；此状态不因规范写成而变为 PASS。
