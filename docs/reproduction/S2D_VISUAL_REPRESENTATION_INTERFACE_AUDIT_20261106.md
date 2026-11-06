# S2D Visual Representation Interface Audit — 2026-11-06

WORKSTREAM = S2D

## 证据与范围

- **SOURCE-CONFIRMED**：本地工作区 guard 为 `WORKSTREAM_CHECK=PASS`，分支为 `feat/s2d-iemocap-v0`，审计 HEAD 为 `5b3b5b244821bf7db9a9bdb77d21d6f192475327`。
- **SOURCE-CONFIRMED**：本任务仅新增本报告。进入任务时已有 `tests/models/s2d/test_s2d_early_stopping.py` 修改，原样保留；未执行 Git 写操作。
- **UNKNOWN**：epoch7、Dev W-F1 `0.34355913724532466` 及正式 Test 已完成、W-F1 `0.36953449166596747` 是用户最新提供的运行背景，本轮未独立核验。既有接入说明仍保留“Test 未执行”的历史状态，以本次用户说明为当前背景，不据此重跑 Test。
- **SOURCE-CONFIRMED**：未访问真实视频、manifest/cache 资产或 checkpoint；未下载、安装依赖、训练、执行真实 Test、使用 GPU 或连接远程。小型检查仅使用内存合成张量，详见末尾。

以下标签适用于每条结论或整行表格：`SOURCE-CONFIRMED` 为当前源码直接规定或可由其运算推导的接口事实；`INFERENCE` 为设计判断、成本计算或数学推论；`UNKNOWN` 为本轮无资产/实验依据的结论。源码规定的可训练性不等于已核验 checkpoint 数值更新。

### 源码索引

| 简称 | 当前实际消费的文件与主要位置 |
| --- | --- |
| W | [wrapper](../../models/s2d/iemocap/model.py)：113–137 FrozenLandmark；140–174 wrapper/冻结；177–217 build/load |
| U | [S2D core](../../third_party/S2D/model_finetuning.py)：25–39 token/map 转换；42–131 Fovea/Adapter/TMAdapter；133–195 I3DHead；197–230 Prompt_block；233–271 landmark 投影；281–422 初始化；470–535 forward |
| L | [MobileFaceNet](../../third_party/S2D/landmark/mobilefacenet.py)：104–159，尤其 149–159 返回的 `feat` |
| Y | [正式 YAML](../../configs/s2d/iemocap/v0.yaml)：19–45 face/model；46–66 train/export |
| C | [配置门禁](../../models/s2d/iemocap/config.py)：44–61 |
| D | [dataset/transform](../../datasets/iemocap/s2d_dataset.py)：18–44 transform；47–58 decode；99–134 dataset |
| S | [Uniform16](../../datasets/iemocap/s2d_sampling.py)：7–20 |
| F | [face/cache](../../datasets/iemocap/s2d_faces.py)：75–120 cache 校验；240–250 cached crop；301–334 tracking/fallback |
| R | [训练入口](../../scripts/models/s2d/iemocap/train.py)：106–135 mode/loss/backward；265–283 loader/model/optimizer |

**SOURCE-CONFIRMED**：W:187–196 实际加载的是 U 与 L，没有调用上游训练入口或上游其他 landmark wrapper。U:10–16 依赖外部 `timm` / `einops`；[远程依赖清单](../../models/s2d/iemocap/requirements-remote.txt) 固定 `timm==0.6.13`、`einops==0.6.1`。本地现有 `m3ed_mmgcn` 有 torch，但没有这两个包，故未执行完整 core，也未独立审阅实际远程安装的 timm 内部实现。表中边界 shape 来自本仓库构造参数及数据流，不冒充完整模型实测。

**SOURCE-CONFIRMED**：U、L 的本地字节 SHA256 均与 `third_party/S2D/UPSTREAM.json` 对应条目一致；登记上游 commit 为 `e1b5e5dfb57ab15f70103c933b346bfa62943159`。

## 1. Executive conclusion

1. **SOURCE-CONFIRMED**：当前执行图没有“先跑完深层独立空间 ViT，再进入单个 TMA”的阶段。首个 `prompt_blocks[0]` 在 CLS/位置编码及第一个 ViT block 之前就调用 TMA，后续 11 个 deep prompt 继续调用 TMA（U:475–511）。
2. **SOURCE-CONFIRMED**：Candidate A 存在，但严格位于第一次 TMA 之前的是 RGB patch tokens 或其 LayerNorm 结果，均为 `[B*16,196,768]`，不是完成 12 层 ViT 的静态语义表示。深层、未经过任何 TMA 的现成 frame embedding：**NOT AVAILABLE**。
3. **SOURCE-CONFIRMED**：Candidate B 存在：`VisionTransformer.forward` 中 `x = self.forward_head(x)` 之后、U:532 重排之前的 `[B*16,768]`，等价于 `[B,16,768]`。它已经融合 landmark、MCP、所有 TMA 与 ViT blocks，尚未时间平均。
4. **SOURCE-CONFIRMED**：另一个真实候选 C0 是**首个 MCP 的静态空间支路输出**：`Prompt_block.forward` 返回值 0，即 U:229 的 `x0`；调用处为 U:479 的 `x_feat`，随后 U:480–481 的 tokens / `a`。它虽在程序顺序上晚于 TMA 调用，计算依赖却不包含 TMA 输出。只有第 0 个 prompt 的该支路仍是逐帧独立的。
5. **INFERENCE**：`RECOMMENDED_FEATURE_BANK_INTERFACE = Candidate C0`，定义 `z_t = spatial_mean(MCP0_static_map_t)`，D=768。它最贴合“独立 timestamp 候选、N>16、Audio 决定取哪些帧”的接口要求；必须明确命名为 **S2D 首层 MCP 静态分支特征**，不能称完整 S2D 深层表示。
6. **INFERENCE**：严格无 TMA 的 Q1 主 probe 使用 C0，另保留 B 作为**完整 S2D 上下文表示 probe**。两者回答不同问题；不能用 C0 的负结果否定完整视觉编码器，也不能用 B 的结果宣称独立逐帧编码有效。这里共同使用 C0 是因静态 probe 与采样都要求隔离跨帧混合，并非强行统一所有研究表示。
7. **SOURCE-CONFIRMED**：当前模型直接输入 T=32：**NO**。Route A 不可直接使用；Route B 可规划独立静态提取器；Route C 可保留 slot-to-timestamp 映射，但不是唯一、上下文无关的 timestamp bank。

## 2. Current forward graph

**SOURCE-CONFIRMED**：下图按 W:162–174 和 U:470–535；`BT` 总是按 `b*T+t` 排列，B 表示 clip/utterance 数量。

```text
video [B,3,16,224,224]（已 RGB/ImageNet normalize）
  ├─ FrozenLandmark:
  │    flatten BT -> bilinear 112 -> MobileFaceNet.conv_4 feat
  │    landmark_features [BT,128,14,14]
  │    -> patch_embed_prompt 1x1 -> a [BT,196,768]
  └─ RGB flatten BT -> patch_embed -> x [BT,196,768]              [A0]
        |
        + norm0(x), norm0(a) -> concat maps [BT,1536,14,14]
        |    ├─ RGB half -> TMA0 -> x1 [BT,196,768]
        |    └─ RGB/landmark halves -> 1x1 / Fovea / add / 1x1
        |                         -> static map [BT,768,14,14]    [C0]
        |                         -> tokens; a = static tokens
        + x = original RGB x + x1 + static tokens
        -> prepend CLS + spatial pos -> [BT,197,768]
        -> reshape [B*197,16,768] + temporal_embedding
        -> restore [BT,197,768] -> norm_pre (Identity)
        -> ViT block 0
        -> i=1..11:
             deep MCP/TMA_i on current patch maps + recurrent a
             patch residual update; retain current CLS
             -> ViT block i
        -> ln_post -> [BT,197,768]
        -> forward_head: take CLS, fc_norm (Identity)
        -> [BT,768] = [B,16,768]                                 [B]
        -> rearrange [B,768,16] -> unsqueeze [B,768,16,1,1]
        -> I3DHead.mean(time) -> [B,768,1,1,1]
        -> dropout(p=.5; eval 时恒等) -> flatten -> h_v [B,768]
        -> fc_cls -> logits_v [B,6] -> wrapper softmax -> prob_v [B,6]
```

**SOURCE-CONFIRMED**：默认 `prompt_type='deep'`、`global_pool='token'`、`pre_norm=False`，W 的 builder 没覆盖它们（U:281–312，W:191–195）。没有使用 `global_pool='avg'` 分支。U:532 的重排仍保留全部 16 个时间位置；真正消去时间轴的是 U:186。

## 3. Tensor shape table

**SOURCE-CONFIRMED**：本表每行均由 W/U/L 的调用和构造参数推导；没有宣称真实 checkpoint 的全图实测。`N=196=14*14`，`Ncls=197`，`D=768`，`rD=192`，MCP hidden=8。

| 模块 / tensor | 输入 → 输出 | 时间 / 空间轴 | 跨帧混合 | 依据 |
| --- | --- | --- | --- | --- |
| wrapper 输入 | `[B,3,16,224,224]` → 同 shape | T=轴2，H/W=轴3/4 | 无模型混合 | W:162–169 |
| FrozenLandmark flatten/resize | video → `[BT,3,112,112]` | T 折入轴0；112²空间 | 无；eval BN 固定 | W:127–137 |
| MobileFaceNet 前段 | 112² → 56²/64ch → 28²/64ch → 14²/128ch | 每个 BT row 独立 | 无 | L:109–114,140–152 |
| landmark `feat` | `[BT,128,14,14]` | BT 与14² | 无 | L:152,159；W:167–169 |
| RGB `patch_embed` | `[BT,3,224,224]` → `[BT,196,768]` | BT；token轴1 | 无 | U:353–359,471,573–574 |
| `patch_embed_prompt(a)` | `[BT,128,14,14]` → 1x1 projection → `[BT,196,768]` | BT；196个14²位置 | 无 | U:257–270,364–365,472 |
| 首次 MCP 输入 | 两个 normalized token map → `[BT,1536,14,14]` | BT；14² | 无 | U:476–478 |
| 首次 TMA 输入 `x2` | RGB half `[BT,768,14,14]` → `[BT,196,768]` | BT；196 token | 此边界尚无 | U:220–223 |
| TMA temporal reshape | `[BT,196,768]` → `[B*196,16,768]` | 时间轴1；空间折入轴0 | reshape本身不混合 | U:124–126 |
| `T_Adapter` | 768→192→attention over16→768 | `[B*196,16,D]` | 是，同空间位置的16帧 | U:70–103,110–111,127 |
| TMA restore / `S_Adapter` | restore `[BT,196,768]` → LN → 768→192→768 | BT；196 token | 继承已混合结果；S为逐token MLP | U:128–130；attention=False |
| MCP static支路（首层） | 两半分别768→8；RGB Fovea+landmark；8→768 | `[BT,768,14,14]` | 无 | U:225–230 |
| 首次残差合并 | 3个 `[BT,196,768]` 相加 | BT；196 patch | 是，来自 `x1` | U:479–482 |
| CLS / spatial pos | `[BT,196,768]` → `[BT,197,768]` | 同上+CLS | 继承patch混合；CLS初始独立 | U:453–468,486 |
| temporal embedding | `[BT,197,768]` ↔ `[B*197,16,768]` | token折入轴0；时间轴1 | 加位置向量本身不聚合别帧 | U:488–493 |
| 12个 ViT blocks + deep prompts | `[BT,197,768]` → 同shape | ViT token轴1；T仍在BT | 继承/反复TMA；ViT调用按BT输入 | U:394–409,494–511 |
| `ln_post` | `[BT,197,768]` → 同shape | LayerNorm最后768维 | 无新增时间聚合 | U:410–411,514 |
| `forward_head` | CLS `[BT,768]` → 同shape | BT；已去空间token轴 | 继承完整上下文 | U:518–523 |
| I3DHead 输入 | `[BT,768]` → `[B,768,16,1,1]` | 时间轴2；空间1² | 重排无新增混合 | U:531–534 |
| temporal mean | `[B,768,16,1,1]` → `[B,768,1,1,1]` | T被约简 | 是，16位置平均 | U:185–186 |
| dropout/flatten | pooled → `[B,768]` | 已无T | 训练dropout；eval恒等 | U:188–195 |
| classifier/softmax | `[B,768]` → `[B,6]` → `[B,6]` | 无T | clip级 | U:193；W:174 |

**SOURCE-CONFIRMED**：MobileFaceNet 继续计算后段及 `[BT,136]` 输出，但 W:136 用 `_` 丢弃，消费的是 `conv_4` 后的 `feat`，不是136维坐标。Prompt_PatchEmbed 的 `patch_size=16` 元数据不意味着对14² landmark再做16×16卷积；实际 `proj` 是 kernel1/stride1（U:257–270）。

## 4. TMA position

**SOURCE-CONFIRMED**：`Prompt_block.TMA` 是 `TMAdapter`（U:211），每次先取输入map的 RGB/current-main-stream 半边，再 flatten 为 `x2` 后调用（U:220–223）。第一次尚未显式融合 landmark 的 main stream 进入 TMA；后续 main stream 已包含先前融合结果。

```text
[BT,N,D]
  -> [(B*N),T,D]                 # 同一空间patch沿时间串成一条序列
  -> T_Adapter:
       Linear(768,192), GELU, Attention(192, heads=8), Linear(192,768)
  -> [BT,N,D]
  -> LayerNorm(768)
  -> S_Adapter:
       Linear(768,192), GELU, Identity(attention=False), Linear(192,768)
  -> [BT,N,D]
```

**SOURCE-CONFIRMED**：两个 Adapter 均 `skip_connect=False`，TMA 不是在内部返回 `原始x+更新`；主残差相加发生于 U:482、509–510。TMA不删帧、不缩短时间、不输出单个clip向量。其调用没有传入 causal mask；不是本仓库实现的 causal temporal adapter。

**INFERENCE**：按所调用的 temporal attention 接口，B 中时间位置 t 是“以该 slot 为锚点、受整段上下文影响”的特征，不等于独立帧特征。直接筛选这些位置无法撤销其他帧对它的影响；用于实时推理不能据 slot 索引声称无未来信息。

**SOURCE-CONFIRMED**：TMA后、时间pooling前确实存在 temporal feature sequence：最直接抓取 U:531 的 `x`，reshape `[B,16,768]`；或对 `core.head` 注册 forward-pre-hook，拿 `[B,768,16,1,1]` 转置为 `[B,16,768]`。前者已完成 spatial CLS selection，后者是相同数值的重排。

## 5. h_v generation

**SOURCE-CONFIRMED**：h_v 的完整前史是 12 次 MCP/TMA、temporal embedding、12个 ViT block、`ln_post` LayerNorm、CLS selection、Identity `fc_norm`、时间平均、head dropout、flatten（U:375–416,470–535；W:170–174）。

**SOURCE-CONFIRMED**：默认 `drop_rate=attn_drop_rate=drop_path_rate=0`，builder没有覆盖；唯一显式非零dropout为head的0.5。没有额外 h_v adapter、L2 normalization 或 softmax；softmax只作用于logits。`ln_post` 在CLS选择前，LayerNorm对每个token的通道执行；不对整个时间序列求均值/方差。

**SOURCE-CONFIRMED**：设 B候选为 `q[b,t,:]`，则在eval下：

```text
h_v[b,:] = mean_t q[b,t,:]
logits_v = head.fc_cls(h_v)
prob_v = softmax(logits_v, dim=-1)
```

**SOURCE-CONFIRMED**：训练时 h_v 是上述平均**经过dropout后**的返回值（U:188–195），并不是dropout前特征。eval下dropout恒等；这一点已用实际 I3DHead 类的合成CPU张量核验。

**INFERENCE**：h_v只有一个clip向量，16个位置已平均，不能恢复逐timestamp identity。对B做时间均值后再做线性probe，在相同eval输入下就是对h_v做线性probe，不是新增的一种独立表示。

## 6. Candidate feature interfaces

### Candidate A：严格 chronological pre-TMA

- **SOURCE-CONFIRMED**：A0 = `VisionTransformer.forward_features` 的 `x = self.patch_embed(x)` 后（U:471），`[BT,196,768]`；没有CLS、空间/时间位置编码、ViT blocks或landmark融合。参数 `patch_embed.*` 冻结。
- **SOURCE-CONFIRMED**：A1 = `Prompt_block.forward` 的首个 `x2`，U:222、调用 `self.TMA` 之前，`[BT,196,768]`。它是首层 `prompt_norms[0](A0)`；LayerNorm参数可训练，因此不能把 A1 整体一概称完全未适配，但仍不含MCP landmark融合或深层ViT。
- **INFERENCE**：将A0/A1按196个空间位置平均可定义 `z_t [768]`，或保存全部patch。平均是未来提取器新增的确定性空间约简，本轮未新增实现。没有合法的pre-TMA CLS可取。
- **SOURCE-CONFIRMED**：深层“完整空间ViT输出、且从未经过TMA”的Candidate A：**NOT AVAILABLE**。第1个及之后的TMA输入已有早前TMA上下文。

### Candidate B：完整 S2D，post-TMA / pre-temporal-pooling

- **SOURCE-CONFIRMED**：U:531 的局部 `x`，class `VisionTransformer`、def `forward`，`[BT,768]`，等价 `[B,16,768]`。选CLS、完成`ln_post`，不经过head dropout，含landmark/MCP/TMA/空间ViT。
- **INFERENCE**：最小取法是`core.head` pre-hook；不应拿`core.head`的返回值，因为后者已池化。该位置适合描述完整S2D的contextual visual representation。

### Candidate C0：首个 MCP 的静态空间支路（推荐的独立bank）

- **SOURCE-CONFIRMED**：class `Prompt_block`、def `forward`，U:225–229：

```text
rgb  = norm0(patch_embed(frame))                 # [768,14,14]
land = norm0(patch_embed_prompt(landmark(frame))) # [768,14,14]
u    = Fovea(conv0_0(rgb)) + conv0_1(land)        # [8,14,14]
M0   = conv1x1(u)                               # [768,14,14]
```

- **SOURCE-CONFIRMED**：真实边界为 `core.prompt_blocks[0].conv1x1` 输出；U:479接收为`x_feat`，U:480变为 `[BT,196,768]`，U:481赋值给`a`。`x1`/TMA输出只在U:482另行相加，不是M0的输入。
- **INFERENCE**：推荐 `z_t = M0.mean((-2,-1))`，每个timestamp对应768维。完整forward中可hook得到；独立帧提取器未来可只执行上述已有静态依赖，不必执行TMA。该提取器尚未存在，不能声称当前wrapper已支持逐帧或任意N输入。
- **SOURCE-CONFIRMED**：仅第0个prompt满足该独立性。i≥1的同名static支路使用已混合的`x`，也会递归使用已有上下文的`a`；不能拿最后一层`conv1x1`冒充独立帧encoder（U:498–510）。
- **INFERENCE**：C0是prompt更新支路，**不是**`RGB residual + MCP static + TMA`的完整主干状态。原始RGB residual没有包含在C0输出里；称其“融合RGB与landmark的S2D首层静态prompt特征”才准确。
- **INFERENCE**：C0有严重瓶颈：8→768的最后1x1卷积为仿射映射，故空间均值满足 `mean(M0)=W*mean(u)+bias`；768维向量的变化落在至多8维子空间。不能把D=768解释为768个独立语义自由度，也不能预先保证它优于A或B。可在未来额外保存8维`mean(u)`做等价性/信息量诊断，本轮不改变推荐输出维度。

### 比较

**SOURCE-CONFIRMED**：shape、上下文依赖、参与路径见上述各candidate；以下适用性/效果判断均标注为推断或未知。

| 属性 | A0 / A1 | B | C0 |
| --- | --- | --- | --- |
| 时间定位 | SOURCE-CONFIRMED：独立frame slot | SOURCE-CONFIRMED：保留slot，值依赖16-frame context | SOURCE-CONFIRMED：独立frame slot，前提固定crop |
| 情绪信息 | UNKNOWN：未做probe；A0仅patch projection | UNKNOWN：有情绪监督计算路径，未测可分性 | UNKNOWN：适配空间prompt可能有信号，但瓶颈仅8 |
| landmark | SOURCE-CONFIRMED：无 | SOURCE-CONFIRMED：有 | SOURCE-CONFIRMED：首层已融合 |
| IEMOCAP适配资格 | SOURCE-CONFIRMED：A0冻结；A1的norm可训练 | SOURCE-CONFIRMED：多处adapter/norm/embedding可训练 | SOURCE-CONFIRMED：投影/norm/MCP可训练，backbone冻结 |
| 上下文 | SOURCE-CONFIRMED：无encoder跨帧 | SOURCE-CONFIRMED：反复TMA | SOURCE-CONFIRMED：首层支路无encoder跨帧 |
| A-guided selection | INFERENCE：可，但表示很浅 | INFERENCE：不宜做先全帧编码再选帧的独立bank | INFERENCE：三者中最符合独立、已适配、含landmark条件 |
| TCN/GRU | INFERENCE：可接空间约简后的序列 | INFERENCE：可，但叠加已有时序处理/未来信息 | INFERENCE：可用于独立静态序列，保留实际时间间隔 |
| D / token shape | SOURCE-CONFIRMED：196×768；空间均值后768 | SOURCE-CONFIRMED：每slot768 | SOURCE-CONFIRMED：768×14×14；空间均值后768 |
| 提取难度 | INFERENCE：低；须独立预处理入口 | INFERENCE：低，hook即可，但必须16帧 | INFERENCE：hook低；跳过TMA的独立提取器需小型新工程 |

**INFERENCE**：仅tensor payload的存储估计，不含ID/时间/bbox/容器开销，不等于实际文件大小：

| 每个utterance | FP32，空间约简后768维 | FP16，空间约简后768维 | FP32，全196×768 tokens/maps |
| --- | --- | --- | --- |
| 8 candidates | 24 KiB | 12 KiB | 4.59375 MiB |
| 16 candidates | 48 KiB | 24 KiB | 9.1875 MiB |
| 32 candidates | 96 KiB | 48 KiB | 18.375 MiB |

**INFERENCE**：总量公式 `U*N*768*bytes_per_scalar`。B的32条目必须来自多个16-frame输入，不能据成本表推断支持一次T32；FP16保存会量化，首轮核对建议保留FP32。

## 7. T=16 constraint audit

| 位置 | 类别 | 当前约束与依据 | 标签 |
| --- | --- | --- | --- |
| YAML `model.num_frames:16` | 显式配置/初始化参数 | Y:33；builder传给core | SOURCE-CONFIRMED |
| `config.load_config` | 硬编码门禁 | C:58–59，必须(16,224) | SOURCE-CONFIRMED |
| `build_s2d` / wrapper初始化 | 硬编码门禁 | W:144–145,182–183 | SOURCE-CONFIRMED |
| wrapper forward | 硬编码输入shape | W:163–164，必须`[B,3,16,224,224]` | SOURCE-CONFIRMED |
| upstream forward | runtime T，与初始化值比较 | U:527–529，`assert T == self.num_frames` | SOURCE-CONFIRMED |
| temporal embedding | 初始化参数决定权重shape | U:375–376，`[1,num_frames,768]`，当前16；forward直接相加，无插值 | SOURCE-CONFIRMED |
| temporal embedding reshape | 初始化参数驱动 | U:489用`self.num_frames`；n来自runtime token数 | SOURCE-CONFIRMED |
| TMA reshape | 初始化参数驱动 | U:109,126；每个TMA持有自己的`num_frames`，非每次从video推导 | SOURCE-CONFIRMED |
| MCP | 空间支路无16常量；TMA接收初始化T | U:198–223,385–387；首层静态卷积/Fovea不按时间分组 | SOURCE-CONFIRMED |
| FrozenLandmark | runtime-derived T | W:133–136，b*t；L无16限制 | SOURCE-CONFIRMED |
| wrapper landmark校验 | 初始化T，间接16 | W:167–169；expected首维=B*self.num_frames | SOURCE-CONFIRMED |
| head前恢复时间轴 | runtime-derived B/T | U:532，来自forward输入；已受assert限制 | SOURCE-CONFIRMED |
| I3DHead temporal mean | runtime-derived，不限16 | U:186沿轴2平均；U:184注释中的16不是执行约束 | SOURCE-CONFIRMED |
| sampler / transform / dataset | 硬编码16 | S:9–20；D:28–35,103–104 | SOURCE-CONFIRMED |
| face bbox cache | 硬编码16合同 | F:92–93；indices/sizes必须与cache一致(F:241–242) | SOURCE-CONFIRMED |

**SOURCE-CONFIRMED**：`T32_DIRECT_INPUT = NO`，`T8_DIRECT_INPUT = NO`。已实际执行两个forward的T32前置门禁，均拒绝；没有触发模型计算或真实数据读取。

**INFERENCE**：上游构造器形式上可初始化不同`num_frames`，不等于当前frozen V0可直接改长。需要同时更改wrapper/config/data/cache合同、各TMA初始化值及temporal embedding权重shape；仅改`core.num_frames`不能同步各TMA。插值/重建embedding或重新分组时间均超出“冻结模型语义不变”，本任务不提出直接这样部署。

## 8. Dense candidate bank feasibility

### Route A：一次32帧 → 当前S2D → 32个z_t

**SOURCE-CONFIRMED**：**不可直接执行**；wrapper最先拒绝，upstream也拒绝T与初始化T不一致。现有API只返回pool后的h_v；即便重建32帧模型，也需另行暴露序列。

**INFERENCE**：不推荐作为当前epoch7冻结语义下的bank路线。不能把`B=2,T=16`的两个clip当成`B=1,T=32`：前者是两个独立上下文，后者是一个32帧上下文，含义不同。

### Route B：独立帧/小batch → frozen spatial/static子图 → z_t

**SOURCE-CONFIRMED**：A0、A1及C0的源码依赖可以逐帧计算；C0依赖RGB patch embed、MobileFaceNet、landmark projection、norm0、MCP0静态支路，不需要temporal embedding、ViT blocks或TMA输出。

**INFERENCE**：**可作为新提取接口实现，当前尚无现成API**；优先C0，A0作frozen控制。每个真实timestamp固定crop后可以分batch得到N=32或更多条目，无需把TMA的16改成32。不能调用当前完整`S2DIEMOCAP.forward`喂单帧；须从同一模型实例复用既有模块执行静态子图，不复制一份模型或伪造完整S2D结果。

**SOURCE-CONFIRMED**：A0参数在IEMOCAP训练中被冻结；C0涉及的prompt投影/norm/卷积允许训练且连接分类loss。**UNKNOWN**：epoch7哪些可训练权重实际改变、改变多少，以及C0包含多少情绪信息，本轮不加载checkpoint，不能量化或声称已证实。

**SOURCE-CONFIRMED**：当前缓存只有Uniform16，dataset/transform/cache均拒绝非16；已有cache不能直接覆盖另外16个新timestamp。**INFERENCE**：N32需要独立、版本化的candidate及face/bbox资产合同；保持旧Uniform16资产只读。对于旧16帧，相同frame与相同crop的C0输出应与完整forward的C0 hook一致，这应成为未来实现的核心验收。

### Route C：多个16-frame window → B序列 → 拼接bank

**SOURCE-CONFIRMED**：每个window可输出16个B-slot；保留输入frame_indices便能映射到原帧，而不是模型自动生成timestamp。**INFERENCE**：技术上能得到>16条目，但条目语义应是`(utterance,frame_index,window_id,slot,context_indices)`，不能只以timestamp为key。

**INFERENCE**：同一帧因不同window、temporal embedding位置、其他15帧与前处理tracking不同，会有不同表示；存在重复context、window边界/位置效应，邻接拼接也不等于一条统一32-frame forward。不能用平均重复条目掩盖这些差异而不记录聚合规则。

**INFERENCE**：不推荐作Q3主bank；可作“固定window上下文表示”独立对照。其完整16帧双向上下文及现有全utterance Uniform16输入不满足机器人逐帧即时推理。为获得低延迟改成滑窗是新的输入/部署协议，不能宣称保持现有推理语义。

### Uniform16 / Audio-guided16 / Audio-guided8 的科学边界

- **INFERENCE**：从同一个C0独立N候选bank选择16/16/8条，再使用新probe/TCN/GRU比较，是可清楚解释的特征采样实验；需要记录实际时间间隔、可用性/重复帧mask，不能把等slot距离当等物理时间。
- **INFERENCE**：从B的16个已上下文混合位置中选8个，所选特征已经用过另外8帧；不能解释为“只看8帧”或Audio省掉视觉计算。选完raw frames再跑S2D16是另一种实验；S2D8当前不支持，用重复帧凑16也不是同一个8-frame模型。
- **INFERENCE**：离线预计算全部候选后选择只能评价信息选择/下游预测；不能单凭bank实验宣称运行时视觉编码开销下降。实际节省计算需部署时先选timestamp，再只计算被选静态子图。

## 9. Landmark / MCP interaction

**SOURCE-CONFIRMED**：输入视频已按ImageNet标准化，landmark直接对这一输入双线性resize到112（W:135）；没有在W中另行反归一化或拼接坐标。MobileFaceNet的`feat`经可训练`patch_embed_prompt` 128→768，再与RGB分别用同一个`prompt_norms[0]`，形成MCP的两半（U:472,476–478）。

**SOURCE-CONFIRMED**：首个TMA只取MCP输入的前半RGB；MCP静态支路为 `Fovea(conv_RGB) + conv_landmark`，再投影到768。第一次主干注入点是 U:482 的 `x + x1 + x_feat`，之后每层deep prompt递归更新`a`。这条实际执行路径不是“预先得到landmark坐标后concat到classifier”。

| 特征 | landmark-aware MCP是否已进入 | 标签 |
| --- | --- | --- |
| A0 raw patch tokens | 否，纯RGB patch projection | SOURCE-CONFIRMED |
| A1 第一次TMA输入 | 否，只做RGB侧norm0 | SOURCE-CONFIRMED |
| C0 首层MCP static输出 | 是，已融合首层landmark prompt；没有TMA输出 | SOURCE-CONFIRMED |
| B / h_v | 是，经过完整递归MCP/TMA | SOURCE-CONFIRMED |

**INFERENCE**：推荐对外名称分别为 `S2D frozen RGB patch feature`、`S2D MCP0 static branch feature`、`S2D contextual frame-slot feature`、`S2D clip representation`，避免把来源相同误写为表示语义相同。

## 10. Frozen vs trainable representation audit

**SOURCE-CONFIRMED**：W:150–153以名称包含关系确定`requires_grad`，不是按模块类型推断：

```text
temporal_embedding, temporal_attn, cls_token_t, ln_post,
cls_head, Adapter, head, prompt
```

| 实际参数组 | IEMOCAP训练资格 | 影响哪些候选 | 标签 |
| --- | --- | --- | --- |
| `core.patch_embed.*` | frozen | A0固定、A1/C0/B的RGB底座 | SOURCE-CONFIRMED |
| `core.pos_embed`, `core.cls_token` | frozen（`cls_token`不匹配`cls_token_t`） | B路径 | SOURCE-CONFIRMED |
| `core.blocks.*` attention/MLP/norm | frozen；名称不匹配白名单 | B；输入仍可随prompt改变 | SOURCE-CONFIRMED |
| `core.patch_embed_prompt.*` | trainable（包含prompt） | C0/B | SOURCE-CONFIRMED |
| `core.prompt_norms.*` | trainable | A1/C0/B | SOURCE-CONFIRMED |
| `core.prompt_blocks.*` conv/Fovea/TMA/norm | trainable | C0静态部分及B完整路径 | SOURCE-CONFIRMED |
| `core.temporal_embedding` | trainable | B | SOURCE-CONFIRMED |
| `core.ln_post.*` | trainable | B | SOURCE-CONFIRMED |
| `core.head.fc_cls.*` | trainable | logits；其权重不参与C0/B/h_v前向值 | SOURCE-CONFIRMED |
| landmark全部参数/BN统计 | 参数frozen、强制eval | C0/B的landmark底座 | SOURCE-CONFIRMED |

**SOURCE-CONFIRMED**：R:281–283仅将`requires_grad=True`参数交给AdamW；R:121–127由logits计算CrossEntropy再backward/step。C0注入主干，存在到loss的可微路径；frozen ViT blocks仍可传播对输入的梯度，并非对整个core套no_grad。landmark则明确no_grad。

**SOURCE-CONFIRMED**：正式构建会加载外部预训练、丢弃原FER classifier，且冻结ViT参数若缺失会失败（W:198–215）。因此按该入口成功训练时，A0是固定预训练patch特征，不能声称epoch7学到了新的RGB patch projection。

**UNKNOWN**：未核对真实run的代码commit、resolved config、load report或epoch7参数差分。`trained/adapted`在本报告表示源码允许并提供监督路径，不代表本轮已经验证每个参数非零梯度或数值更新。C0不能标为“完全frozen pretrained”，也不能标为“已实证全部更新”。

## 11. Determinism audit

- **SOURCE-CONFIRMED**：`model.eval()`会把core中Dropout关闭；FrozenLandmark的`train`覆盖始终调用`super().train(False)`，BN running statistics也固定（W:124–137,157–160）。A/C0/B边界都在head dropout之前；h_v在eval下也不受它影响。
- **SOURCE-CONFIRMED**：U默认pos/block/attention/drop-path的rate为0，TMA内部attention dropout和projection dropout默认0（U:74–82）。forward没有显式随机采样代码。初始化随机并不等于加载固定checkpoint后每次forward随机。
- **SOURCE-CONFIRMED**：D:18–44当前各split用同一确定性resize/normalize；S:15–20固定Uniform16；cached_bbox走既有bbox crop，不调用在线MTCNN（D:76–85，F:240–250）。上游可选RandAugment/flip/crop/erasing没有进入当前wrapper/dataset调用链。
- **INFERENCE**：固定checkpoint、pixel crop、dtype、设备/库版本与eval下，候选没有有意保留的随机增强/dropout；未来bank提取应同时用`eval()`与`inference_mode()`。仅no_grad不会关闭Dropout；仅eval也不关闭autograd。
- **UNKNOWN**：完整core的重复运行、跨batch数值误差、GPU kernel级确定性、远程timm真实版本及跨设备bitwise一致性未验证。本轮I3DHead合成CPU重复结果相等不能替代完整S2D确定性测试；eval不是跨硬件bitwise保证。
- **SOURCE-CONFIRMED**：face tracking/fallback可依赖其他采样帧，包括首段失败时回填未来第一个有效bbox（F:320–330）。**INFERENCE**：这是确定性的上下文依赖，仍影响因果性与候选集间crop一致性；encoder的C0逐帧独立不等于当前整条pixel pipeline可在线因果执行。

## 12. Recommendation for Q1

**INFERENCE**：`RECOMMENDED_Q1_FEATURE = Candidate C0`，空间均值后的 `[B,T,768]`，作为**严格静态、IEMOCAP可适配、landmark-aware的表示probe主接口**。与只保留frozen patch projection的A0相比，它有实际任务监督路径；与B相比，它不提前做temporal adaptation，适合把线性/TCN probe的问题限定在输入静态特征上。

**INFERENCE**：Q1必须另保留B（`[B,16,768]`）作为**完整S2D上下文表示对照**。线性probe可对B取时间均值（与eval h_v相同）；TCN/GRU接B时必须披露前面已做TMA，不能解释为下游时序模块独自产生上下文收益。A0可作为“固定RGB patch底座”控制，帮助识别C0窄瓶颈的影响。

**INFERENCE**：两类科学结论分开写：C0 probe回答“首层静态MCP支路是否携带可读情绪信息”；B probe回答“已适配的完整S2D上下文编码是否携带可读情绪信息”。没有现成的“完成深层ViT、又隔离TMA”的接口，故**不能仅靠C0的负结果宣称整个visual encoder没有情绪信息**。若研究问题专指完整S2D，报告主结果必须包含B，不能只报C0。

**UNKNOWN**：哪一个候选实际更可分、TCN是否优于线性或GRU均未验证；本审计只选择语义可解释的接口，不承诺probe效果。

## 13. Recommendation for A-guided V

**INFERENCE**：`RECOMMENDED_A_GUIDED_FEATURE = Candidate C0`；采用Route B，N个独立timestamp输出768维向量，Audio仅选择timestamp/候选索引。首轮应把原始frame index、真实时间、crop/bbox版本、有效/重复/fallback信息与z一起存储；当前dataset只有frame_indices，未提供显式PTS秒值（D:121–134），不能无依据假设恒定fps。

**INFERENCE**：C0对固定crop的特征不依赖候选集的其他帧，能比较Uniform16 / Audio-guided16 / Audio-guided8并支持未来按需计算。B继续保留作16-frame完整模型对照，不作为可任意重选的独立bank。

**INFERENCE**：机器人增量推理方面，C0的encoder子图可逐帧计算，不需等待16帧；但现有utterance级face tracking及未来bbox回填不因选C0自动变因果。在线crop状态/失败策略需独立设计与版本化，且C0不是完整S2D性能的等价替代。

## 14. Minimal next engineering step

**INFERENCE**：下一独立任务只实现一个**只读、无训练、无Test的feature extractor接口及synthetic验收**，本轮不实现：

1. 从同一canonical S2D模型实例安全strict加载已冻结权重，显式eval/inference；不改模型forward或checkpoint schema。
2. 暴露两种有清楚名称的输出：`mcp0_static_mean [B,T,768]`（独立静态子图）与 `s2d_contextual_slots [B,16,768]`（完整core的head-pre-hook）。A0作为可选诊断控制。
3. 用16帧synthetic输入核对独立C0与完整forward的MCP0 hook一致；改变其他帧时目标C0不变；改变frame batch分组时保持对应；核对B均值等于eval h_v。不能用模型double冒充这项真实core对齐。
4. 用synthetic 32张独立face验证C0输出32条；这不应触碰或改变任何TMA的16帧限制。增加重复提取与原始sample索引映射检查。
5. 先冻结candidate/bbox/timestamp资产合同；旧Uniform16 cache只读保留，新dense版本独立命名。新真实资产提取及Train/Dev probe另行授权推进，Test继续不参与接口/超参选择。

**INFERENCE**：先证明C0与原图同边界数值等价，再评价信息量；不要在这个接口任务中禁用全部TMA跑12层ViT、插值temporal embedding、改T或补帧伪造静态深层模型，这些都会引入新的模型/输入语义。

## 15. Risks / unknowns

| 项目 | 结论与限制 | 标签 |
| --- | --- | --- |
| C0语义深度 | 仅首层prompt、8维瓶颈；不是完整深层S2D | SOURCE-CONFIRMED / INFERENCE（秩推导） |
| C0效果 | 不知道其情绪信息是否足够；不能先验保证最佳准确率 | UNKNOWN |
| checkpoint实际适配 | 没有epoch7对预训练参数diff，不能逐参数确认更新 | UNKNOWN |
| B独立性 | 保留slot，但依赖整段16-frame context；不是单帧观察 | SOURCE-CONFIRMED / INFERENCE（解释） |
| 任意N | 静态子图可规划；当前wrapper/dataset/cache没有任意N接口 | SOURCE-CONFIRMED / INFERENCE（方案） |
| 序列物理时间 | frame_indices不自动等于秒；短视频重复帧，A-guided可能不等间隔 | SOURCE-CONFIRMED / INFERENCE（使用要求） |
| dense crop一致性 | 现有Uniform16 tracking不能自动为新timestamp生成同语义crop | SOURCE-CONFIRMED / INFERENCE（推广限制） |
| 因果部署 | TMA无本地causal mask，bbox首段fallback可看未来 | SOURCE-CONFIRMED |
| 真实库/完整执行 | 本地缺timm/einops，未跑完整S2D或实际远程环境 | UNKNOWN（远程状态） / SOURCE-CONFIRMED（本地范围） |
| 数据/标签/效果 | 本轮不核验视频质量、资产覆盖、真实指标及probe结果 | UNKNOWN |

## 本轮检查记录

**SOURCE-CONFIRMED**：执行的只读检查为workstream guard、Git根/分支/remote/status/HEAD/diff检查、限定路径源码读取、U/L来源SHA256校验、AST提取冻结白名单并检查代表参数名。未递归扫描全仓或数据/输出目录。

**SOURCE-CONFIRMED**：在现有 `m3ed_mmgcn` 环境以 `python -B -c` 执行内存脚本，torch CPU线程数1；没有落盘测试脚本、tensor或Python cache。具体synthetic tracing：

| 检查 | 实际执行内容 | 结果 |
| --- | --- | --- |
| TMA布局等价重排 | meta tensor `[32,196,768]`（B=2,T=16）→`[392,16,768]`→恢复；另用CPU整数sentinel逐b/n/t验证索引 | PASS |
| 实际I3DHead | 从U的AST提取原类，随机初始化classifier，eval输入CPU `[2,768,16,1,1]`；验证h=`mean_t`、logits shape、两次输出相等 | PASS |
| 实际FrozenLandmark | L的随机初始化MobileFaceNet+W的原FrozenLandmark类；输入synthetic zeros `[1,3,16,224,224]`；得到`[16,128,14,14]`，所有参数frozen、所有子模块eval | PASS |
| wrapper T32门禁 | AST提取原`S2DIEMOCAP.forward`，仅提供必要self占位，meta `[1,3,32,224,224]`在首个guard抛ValueError | PASS |
| upstream T32门禁 | AST提取原`VisionTransformer.forward`，self.num_frames=16，meta T32在assert抛AssertionError | PASS |

**SOURCE-CONFIRMED**：重排检查用torch reshape/permute复现源码einops表达式的布局语义，没有执行实际TMAdapter attention；实际head/landmark检查不使用预训练权重，不能代表模型精度或全图验证。完整S2D forward、MCP0独立提取对齐、真实数据读取、backward、训练、真实Test、GPU、远程操作均为0。

**SOURCE-CONFIRMED**：最终验收执行 `git status --short`、`git diff --check`、`git diff --stat`；另对未跟踪的新报告单独检查空白错误，并核对既有测试文件SHA256仍为 `e26e9ea234f504605e33f0e82e779da64ac35c8caf98b807e4aa6cd1e30a3f7d`。新报告未暂存，普通tracked diff/stat不包含它。
