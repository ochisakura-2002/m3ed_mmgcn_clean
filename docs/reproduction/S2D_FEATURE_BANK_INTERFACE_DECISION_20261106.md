# S2D Feature Bank Interface Decision — 2026-11-06

WORKSTREAM = S2D

本轮 Q1-2A 仅建立只读表示接口及 synthetic 验收，不生成真实 feature bank。
本决定按本轮用户冻结要求，将此前
[表示审计](S2D_VISUAL_REPRESENTATION_INTERFACE_AUDIT_20261106.md)
中的“C0 为主 bank”建议更新为 **D0 为主静态表示、C0 为诊断**。
审计中的源码事实继续适用；本文件不回写或覆盖历史审计。

## 表示合同

| 表示 | 输出字段 | 输出 shape | 用途与边界 |
| --- | --- | --- | --- |
| A0 | `a0_rgb_mean` | `[B,N,768]` | frozen RGB control；原 RGB patch projection 的空间均值，无 landmark/TMA |
| C0 | `c0_mcp_static_mean` | `[B,N,768]` | adapted landmark-aware diagnostic；MCP0 静态支路空间均值，无 TMA 贡献 |
| D0 | `d0_static_residual_mean` | `[B,N,768]` | primary independent static feature；**S2D MCP0 static residual feature** |
| B | `s2d_contextual_slots` | `[B,16,768]` | full contextual S2D control；完整 MCP/TMA、ViT、temporal embedding 后的 slot |

上表的 B 表示名与 shape 中 batch 维 B 依上下文区分。静态接口支持任意正整数 N；
完整 S2D 仍仅接收 16 帧，N=32 的静态检查不代表完整 S2D 支持 T32。

定义严格为：

```text
A0_tokens = core.patch_embed(RGB_frames)              # [BN,196,768]
C0_map    = MCP0 的返回值 0                           # [BN,768,14,14]
C0_tokens = C0_map.flatten(2).transpose(1,2)           # [BN,196,768]
D0_tokens = A0_tokens + C0_tokens
D0_mean   = D0_tokens.mean(dim=1).reshape(B,N,768)
```

D0 在空间约简前相加，保留 A0 的 pretrained RGB 底座和 C0 的 landmark-aware、
IEMOCAP 可适配静态修正。D0 没有 TMA、12 层 contextual ViT 或 temporal embedding，
不得称“完整 S2D feature”。A0 的参数由 wrapper 冻结；“pretrained / adapted”描述正式
权重的来源或适配资格，本轮随机权重测试不证明 epoch7 的具体参数更新或表示质量。

C0 最后的 1×1 卷积是 8→768；空间均值向量的有效变化至多位于 8 维子空间，
故只作诊断，不作为默认主 bank。D0 的 RGB 残差保留了不经此瓶颈的路径，
但这不预先证明其情绪可分性或 probe 效果。

## API 与共享模块

实现位于 [features.py](../../models/s2d/iemocap/features.py)，调用者传入已经正确加载
权重的**同一个** canonical `S2DIEMOCAP` 实例，以及固定 crop 后已经 ImageNet
normalize 的浮点 RGB tensor。接口不打开文件、不加载权重、不修改模型的注册结构。

```python
import torch
from models.s2d.iemocap.features import extract_static_features, extract_contextual_slots

# model: 已有、按原合同 strict reload 的 canonical S2DIEMOCAP 实例。
model.eval()
with torch.inference_mode():
    static = extract_static_features(model, independent_crops)  # [B,3,N,224,224]
    contextual = extract_contextual_slots(model, clip16)        # [B,3,16,224,224]
    d0 = static["d0_static_residual_mean"]
    slots = contextual["s2d_contextual_slots"]
```

两个函数内部也使用 `torch.inference_mode()`；要求所有已有子模块已处于 eval，
若处于 train 或混合模式则报错，不自动修改 training / requires_grad。
结果是用于只读提取的 inference tensor；后续 probe 数据构建属于独立任务。

静态路径复用 `core.patch_embed`、原 `FrozenLandmark` / MobileFaceNet、
`core.patch_embed_prompt`、`core.prompt_norms[0]`，以及 MCP0 的原有叶模块。
上游 `Prompt_block.forward` 没有 static-only 方法，且先调用 TMA，因此接口使用
PyTorch 自带 FX 跟踪**原始方法**、仅保留返回值 0 的依赖节点。
空间/通道形状固定为 V0 的 1536×14×14，帧数 BN 不固定。

FX 将所有子模块作为叶节点：跟踪时不运行 TMA，生成后也没有 TMA 节点。
保留的 `conv0_0 / conv0_1 / fovea / conv1x1` 必须恰好符合白名单，否则失败。
graph view 引用原叶模块及其参数；不复制 S2D、不初始化子模型、不复制权重，
不手写第二套 MCP 数学、不替换原方法，也不向原模型注册额外模块。
每次静态调用构建短小 graph view；未引入跨调用权重副本或全局模型缓存。

B 通过原 `core.head` 的临时 forward-pre-hook 读取 `[B,768,16,1,1]`，
转换为 `[B,16,768]`。实际仍执行未改动的 wrapper/core forward；该位置已完成
`ln_post` 与 CLS selection，尚未 temporal mean 或 head dropout。
hook 在 `finally` 中移除，保留调用者已有 hook；提取期间同一实例应串行使用。
eval 下验收关系为 `slots.mean(dim=1) ≈ model(clip16)["h_v"]`。

## 科学用途与数据纪律

- Q1：**D0 + B 为主**，A0/C0 为诊断。D0 是独立静态表示，B 是完整上下文表示，
  二者的 probe 回答不同问题。
- Q3：**D0 为候选 feature bank**。B 的每个 slot 已读取整段 16 帧上下文，
  不用于“先选帧后编码”的效率论证。
- 静态独立性只针对固定、相同输入 crop 的 encoder 子图；已有 face tracking / 首段
  future-bbox fallback 的上下文依赖不因本接口消失，不能据此宣称整条像素流程因果。
- 离线编码全部候选后再选择，仅能评价信息选择；真实节省视觉计算需要部署时先选
  timestamp、再只编码所选帧，本轮不实现该流程。

| 数据 split | 后续允许用途 |
| --- | --- |
| Train | probe fitting |
| Dev | feature / probe / hyperparameter selection |
| Test | only after Q1 design frozen；只在 Q1 设计冻结后最终评估 |

当前已有 S2D-V0 Test 结果不得用于后续 feature、probe、超参数或模型选择。
本轮不读取真实 IEMOCAP 视频，不导出 Train/Dev/Test 特征，不生成新 face cache，
不做 N32 dense sampling、Audio guidance、probe training、TCN 或 GRU。

## 验收与本地限制

专项测试：[test_s2d_features.py](../../tests/models/s2d/test_s2d_features.py)。

真实-core fixture 通过 `build_s2d(..., {})` 创建原始 12 层 / 768 维 core 和原始
MobileFaceNet，随机初始化，无 checkpoint 路径，无下载，不缩小 depth、维度或 TMA。
它先用正常完整 forward 注册 A0、C0、MCP0 返回值和 post-CLS 的 `fc_norm` hook，
再单独调用静态接口，对比 raw token/map 和约简输出。覆盖内容为：

| 验收项 | 断言 | 本地状态 |
| --- | --- | --- |
| A0/C0/D0 shapes 与 canonical alignment | 原始 hook 边界逐元素接近；D0 严格按 tokens 相加再平均 | **BLOCKED / SKIPPED** |
| frame independence | 只改 frame 7，其余帧的三种静态表示不变；RGB 变化非空 | **BLOCKED / SKIPPED** |
| N32 与 batch 顺序 | `[1,32,768]`，与 `[2,16,768]` 分组对应；另核对 N1 | **BLOCKED / SKIPPED** |
| B shape / B mean == h_v | 与完整 forward 的 post-CLS 边界一致；时间均值等于 eval h_v | **BLOCKED / SKIPPED** |
| eval determinism | A0/C0/D0/B 重复调用接近，无 autograd 图 | **BLOCKED / SKIPPED** |
| 只读性 | 参数及 buffers 的 SHA256、requires_grad、training、模块身份、state_dict schema、原 forward 均不变 | **BLOCKED / SKIPPED** |

所有数值断言使用 `rtol=1e-5, atol=1e-6`，不是跨设备 bitwise 一致保证。
静态提取测试期间对所有 TMA、ViT block、完整 MCP0 和 head 注册禁止调用的 hook，
确保独立路径不执行它们；同时核对 num_frames、所有 TMA 的帧数及 temporal embedding
shape 均保持 16 帧合同。B 不要求逐帧独立。

2026-11-06 本地 `m3ed_mmgcn` 为 PyTorch `2.6.0+cpu`，缺少 `timm` 和 `einops`。
没有安装依赖。新增专项实际为 **23 passed, 6 skipped**：通过的是输入/eval 门禁、
FX graph 引用与排除 temporal 节点、B hook 位置/顺序的 contract 检查和异常清理。
这些小型 harness **不是 canonical core 数值等价性证据**；六个真实-core 验收全部
明确 skip，不能报告成 PASS。远程 `s2d_iemocap` 环境尚未在本轮执行。

两个新增 Python 文件的 `py_compile` **PASS**；`git diff --check` 及三个新增文件的
单独 `git diff --no-index --check` **PASS**。既有 early-stopping 测试与表示审计的
SHA256 均与任务开始时一致；本轮没有 git add、commit、push 或远程操作。

本地命令：

```powershell
conda run -n m3ed_mmgcn python -B -m pytest tests/models/s2d/test_s2d_features.py -q -p no:cacheprovider -rs
conda run -n m3ed_mmgcn python -m py_compile models/s2d/iemocap/features.py tests/models/s2d/test_s2d_features.py
git diff --check
git status --short
```

下一步在既有 Ubuntu `s2d_iemocap` 环境、按正常代码交接流程后执行（本轮不连接、
不同步远程，不停止其他实验，不执行 git add/commit/push）：

```bash
python scripts/dev/check_workstream.py s2d
python -B -m pytest tests/models/s2d/test_s2d_features.py -q -p no:cacheprovider -rs
```

默认 CPU；如由执行者明确选择现有 GPU，可在 pytest 命令前设置
`S2D_FEATURE_TEST_DEVICE=cuda`。两种模式均仅使用 synthetic 输入与随机测试权重，
不得在缺依赖时将核心对齐测试改成 mock。必须取得六项真实-core 通过结果后，
才把接口数值验收记为完成。真实 bank 及 Train/Dev probe 另行立项，Test 继续隔离。

## 保留边界

仅新增只读 Python API、专项测试和本决策文档；没有新增训练模块或修改训练配置，
因此没有新增 YAML 开关，也没有改变 baseline 执行路径。
原模型 forward 数学、checkpoint schema、Uniform16、face/cache、label mapping、
early stopping、training config 与 formal outputs 均保持原样。
进入任务前已有的 early-stopping 测试修改和表示审计原样保留。
