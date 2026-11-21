# CT-MultiDAG Fusion 正式接入与 Ubuntu 交接

2026-11-21 · WORKSTREAM = MULTIDAG · seed 100 Train/Dev 筛选

## 文件与运行入口

| 路径 | 职责 |
| --- | --- |
| `models/multidag_cl/fusion/{__init__,base,matching,experts,registry,model}.py` | 统一 Fusion 接口、三种候选、注册表与四臂共用的 TAV 接入层 |
| `configs/multidag_cl/fusion_candidates/seed100_screen/tav_track_{f0,hybrid,fusion_a,fusion_c}_seed100.yaml` | 四个正式 YAML，沿用 TAV Full Audit 的配置结构和冻结 Track 对照 |
| `scripts/runtime/multidag_cl_fusion/{__init__,spec,runtime}.py` | 严格校验筛选规程、配对初始化与实际数据顺序、聚合 Dev 读数 |
| `scripts/models/multidag_cl/tav_full_audit_v1/train.py` | 现有训练入口，接受四个正式 YAML |
| `scripts/runtime/multidag_cl_tav_full_audit_v1/{formal,runtime,confirm}.py` | 复用训练、选择和重载，按 Fusion 配置接入诊断；冻结审计的模型依赖范围排除独立候选包 |
| `scripts/workflows/benchmarks/run_ct_multidag_fusion_seed100.py` | 四臂预检、顺序执行、失败记录与 Dev 汇总 |
| `tests/models/multidag_cl/tav_full_audit_v1/test_fusion.py` | 正式模型与四臂工作流的合成回归 |

正式代码在 MultiDAG 模型命名空间内。配置复用已有的 `fusion_candidates/` 层级，`seed100_screen/` 是唯一的本轮正式筛选目录。工作区已有的设计研究、`models/experimental/ct_multidag_fusion.py`、`prototype_smoke.yaml` 和原型 smoke 入口保留为本地研究证据。部署从正式 YAML 和正式 Fusion package 构造模型。

## 架构与参数

共用编码器将 T1024、A1582、V342 映射为三个 100D 块。Fusion 统一接收 `[B,L,100]` 三路编码、attention mask 和可选的 `bool[B,L,3]` A/V/T availability，输出 `h0=[A′,V′,T′]`。四个 Track 读出各 300D，拼接 h0 后传入 1500D 分类器。图构建、speaker 关系、编码器、TrackCore、分类器、掩码交叉熵和 CL 开关复用现有组件。

| Variant | `fusion.type` | 结构 | Fusion 参数 | 实测总参数 | h0 |
| --- | --- | --- | ---: | ---: | ---: |
| F0 | `direct_concat` | 原 A/V/T 拼接 | 0 | 1,638,306 | 300 |
| Hybrid | `text_centered_low_rank_matching` | 四条有向边，100→16→100 查询 | 52,800 | 1,691,106 | 300 |
| Fusion-A | `text_centered_full_rank_matching` | 相同四条边，100→100 查询 | 80,000 | 1,718,306 | 300 |
| Fusion-C | `utterance_modality_expert_router` | 300→64→3 联合路由，三套 100→256→100 专家 | 174,127 | 1,812,433 | 300 |

Hybrid 和 Fusion-A 共享 T←A、T←V、A←T、V←T，逐话语用归一化查询与键计算点积和 sigmoid，乘以源模态的线性 value。Fusion-C 在可用专家上做 softmax，以 `n_active × p` 乘专家增量。三候选均在 YAML 固定 `residual_scale: 0.1`。padding 和缺失块精确置零。正式筛选 checkpoint 使用独立身份并绑定完整 YAML，包含 Fusion 设置。F0 实测参数和三候选增量均与原型一致。

## 配对、规程与输出

- 使用同一认证 IEMOCAP T1024/A1582/V342 包。Train 为 100 段对话、4,830 句，Dev 为 20 段对话、980 句。固定 30 epoch、每批 16 段对话、`past_all_causal`、CL OFF、交叉熵、AdamW（lr 5e−4，betas 0.9/0.999，eps 1e−6，weight decay 0）、梯度裁剪 5。scheduler 和 early stopping 均关闭。选取 Dev Weighted-F1 最高、Dev loss 最低、epoch 最早的 checkpoint。
- 先构造 F0，再复制其共享张量快照到 Hybrid、A、C。Fusion 独立种子依次为 101、102、103，构造后恢复 F0 的训练 RNG。manifest 记录共享张量 SHA-256 与 `shared_parameter_max_abs_diff`，后者必须为 0。
- 复用现有分批和独立的 epoch shuffle generator。记录 Train/Dev 人口、每个 epoch 的 Train 对话顺序及 batch 内容与序列 SHA-256、固定 Dev batch 摘要。候选的 batch 在交给优化器或 Dev forward 前与 F0 核对，结束时再次对比完整摘要。
- 每臂的 `logs/epoch_metrics.tsv` 记录 epoch、Train/Dev loss、Dev Accuracy、Weighted-F1、MacroF1 和 UAR。三候选按 epoch 写 Dev 聚合 JSON。匹配臂记录每条边的门值分位及消息/接收范数。Fusion-C 记录 A/V/T 路由分布、自然对数熵、`p<1e-3` 的比例和应用后专家残差范数。最佳 Dev checkpoint 重载后输出六类路由均值。活动比使用实际加到接收块上的残差，分母下限 `1e-8`。中位数低于 `1e-3` 时发出 `FUSION_ACTIVITY_WARNING = NEAR_IDENTITY` 并继续。磁盘只保留聚合机制数据。
- 默认输出层级为 `outputs/<YYYYMMDD>/fusion_seed100_screen_20261121/`，四臂分别进入 `runs/<run_id>/`。批次 manifest 为 `manifests/batches/<batch_id>/fusion_seed100_run_manifest.json`。成功后 `reports/batches/<batch_id>/` 包含 `fusion_seed100_summary.csv`、`fusion_seed100_per_class.csv`、`fusion_seed100_mechanism_stats.csv`。汇总的分数是 0–1，`delta_vs_F0` 是 Dev Weighted-F1 差值。launcher 按 F0、Hybrid、Fusion-A、Fusion-C 顺序执行，首个失败写入 `FAIL` 后退出。已有 run 路径会在 epoch 前被拒绝。
- Test 保持 HOLD。沿用现有认证包读取规程。单个 canonical pickle 含全包，校验完整包哈希和 split 身份，运行时仅创建 Train/Dev dataset。Test 目标逻辑上保持不透明，Test loader、评估、标签检查和 checkpoint 选择计数均为 0。

## 本地验收与 Git

Windows 本地验证使用 CPU 合成数据与 YAML 静态解析。覆盖七种固定 mask 下 F0 逐元素等价、张量维度与有限性、padding、逐话语缺失与空集、因果前缀与未来扰动、反传、实测参数、严格配置解析、四臂实际双 batch 配对、Dev checkpoint 重载、三份汇总与路径冲突停机。正式 Windows Train/Dev 运行和 Test 访问均为 0。

正式 Fusion 定向测试通过 22 项。TAV Full Audit、GraphCore、TA-v0、数据合同与因果回归合计通过 511 项，含上述 22 项。四个 YAML 逐一通过现有训练入口的 `--check-config`，master 通过 `--config-only`。额外的旧 paper-reimplementation 套件有 227 项通过和 1 项 fixture 失败，其已提交的模拟 `label_vocab` 缺少旧验证器要求的 `stoi`。

提交名为 `feat: add CT-MultiDAG fusion candidate screening`，目标为 `origin/feat/multidag-cl-paper-reimplementation`。以下命令定位包含本报告的提交。

```bash
git log -1 --format=%H -- docs/research/fusion/CT_MULTIDAG_FUSION_FORMAL_INTEGRATION_20261121.md
```

## Ubuntu V100 手动交接

先确认远程工作区当前空闲。若有本地修改或分支不符，停止同步并报告。

```bash
cd /home/zhiyuan/research_lh/m3ed_mmgcn_clean
git status --short
git branch --show-current
```

确认状态为空且分支为 `feat/multidag-cl-paper-reimplementation` 后执行。

```bash
git pull --ff-only
conda activate m3ed_mmgcn
nvidia-smi
python scripts/dev/check_workstream.py multidag
```

沿用 TAV Full Audit 的三条认证资产路径。变量 `TAV_ASSET`、`TAV_MODEL_MANIFEST` 和 `TAV_PROVENANCE_JSON` 分别指向 `iemocap_asym_tav_t1024_v1.pkl`、`model_input_manifest_tav_t1024_v1.csv` 和 `tav_t1024_provenance_manifest_v1.json`。配置预检用同一 launcher 的 `--config-only --experiment-date 20261121`。正式启动使用一条命令。

```bash
python scripts/workflows/benchmarks/run_ct_multidag_fusion_seed100.py --execute \
  --asset-path "${TAV_ASSET:?}" --model-manifest-path "${TAV_MODEL_MANIFEST:?}" \
  --provenance-manifest-path "${TAV_PROVENANCE_JSON:?}" \
  --batch-id fusion_seed100_screen_20261121 --experiment-date 20261121
```
