# Active MERC Workstreams

三个长期 workstream 共享 origin：

`https://github.com/ochisakura-2002/m3ed_mmgcn_clean.git`

以下 Windows 路径为本轮建立并验收的工作区；Ubuntu 路径为用户指定的远程部署目标，本轮不连接或修改服务器。
后续任务以实际 `git worktree list` 与 guard 输出核实状态，不把本登记当作远程已部署的证据。

| Workstream | Branch | Local / Windows path | Remote / Ubuntu path | 主要职责 |
| --- | --- | --- | --- | --- |
| MULTIDAG | `feat/multidag-cl-paper-reimplementation` | `E:\MERC\m3ed_mmgcn_clean` | `/home/zhiyuan/research_lh/m3ed_mmgcn_clean` | reusable TA component、TA fusion、causal MultiDAG history |
| S2D | `feat/s2d-iemocap-v0` | `E:\MERC\worktrees\s2d-iemocap-v0` | `/home/zhiyuan/research_lh/worktrees/s2d-iemocap-v0` | reusable V component、S2D Visual Expert、`h_v` |
| ASYMMETRIC_MERC | `integration/asymmetric-merc-v0` | `E:\MERC\worktrees\asymmetric-merc-v0` | `/home/zhiyuan/research_lh/worktrees/asymmetric-merc-v0` | 组件组合、TA/V late fusion、confidence、router |

## 共享外部资产

| 资产 | Ubuntu path | 使用规则 |
| --- | --- | --- |
| IEMOCAP asymmetric MERC v1 | `/home/zhiyuan/research_lh/data/IEMOCAP/asymmetric_merc_v1` | 共享外部数据，不纳入 Git，不复制到 worktree |
| S2D checkpoints | `/home/zhiyuan/research_lh/models/S2D/` | 共享外部权重，不纳入 Git，不复制到 worktree |

资产路径由用户提供，本轮不验证数据内容、不生成副本、不加载 checkpoint。
工作区身份使用明确的部署路径；模型与数据配置仍遵守既有路径配置规则。

## 任务交接

新任务先声明 `WORKSTREAM = MULTIDAG`、`WORKSTREAM = S2D` 或 `WORKSTREAM = ASYMMETRIC_MERC`，再在当前工作区运行对应 guard：

```text
python scripts/dev/check_workstream.py multidag
python scripts/dev/check_workstream.py s2d
python scripts/dev/check_workstream.py asymmetric_merc
```

Windows 两个新分支与 worktree 从同一个 governance commit 起步；原 MULTIDAG 工作区的历史 dirty/untracked 项保留，不进入该提交，也不复制到新 worktree。
本轮完成后仍需用户显式 push 三条分支，并在独立任务中部署 Ubuntu worktree。

详细 namespace、Git 检查与同步边界见 [Git Workstream Policy](../workflows/GIT_WORKSTREAM_POLICY.md)；V0--V3 组合边界见 [Asymmetric MERC Architecture](../workflows/ASYMMETRIC_MERC_ARCHITECTURE.md)。
