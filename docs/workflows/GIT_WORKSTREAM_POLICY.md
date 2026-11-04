# MERC 长期 Workstream 规则

三个 workstream 使用同一个 Git 仓库，通过 `git worktree` 隔离工作目录和分支。
本规则只建立基础设施与职责边界，不修改模型数学行为，也不启动训练。

所有工作区的 origin（fetch 和 push）均为：

`https://github.com/ochisakura-2002/m3ed_mmgcn_clean.git`

## 分支与工作区身份

| Workstream | Branch | Windows | Ubuntu |
| --- | --- | --- | --- |
| MULTIDAG | `feat/multidag-cl-paper-reimplementation` | `E:\MERC\m3ed_mmgcn_clean` | `/home/zhiyuan/research_lh/m3ed_mmgcn_clean` |
| S2D | `feat/s2d-iemocap-v0` | `E:\MERC\worktrees\s2d-iemocap-v0` | `/home/zhiyuan/research_lh/worktrees/s2d-iemocap-v0` |
| ASYMMETRIC_MERC | `integration/asymmetric-merc-v0` | `E:\MERC\worktrees\asymmetric-merc-v0` | `/home/zhiyuan/research_lh/worktrees/asymmetric-merc-v0` |

这里的绝对路径是用户指定的工作区身份白名单；数据配置、模型代码和运行输出继续使用项目既有的相对路径规则。
Ubuntu 路径是待部署目标，Windows 建立 worktree 不代表远程已经部署。

## 主要 namespace 与职责

| Workstream | 主要 namespace | 职责 |
| --- | --- | --- |
| MULTIDAG | `models/multidag_cl/`、`configs/multidag_cl/`、`scripts/models/multidag_cl/` | 维护可复用 TA 组件及 causal MultiDAG history |
| S2D | `models/s2d/`、`configs/s2d/`、`scripts/models/s2d/`、`datasets/iemocap/s2d_*.py`、`tests/models/s2d/` | 维护可复用视觉组件、S2D Visual Expert 与 `h_v` |
| ASYMMETRIC_MERC | `models/asymmetric_merc/`、`configs/asymmetric_merc/`、`scripts/models/asymmetric_merc/`、`tests/models/asymmetric_merc/` | 维护组合、late fusion、confidence 和 router |

namespace 表示职责归属，不保证列出的后续模块均已实现。共享基础设施改动必须明确影响范围，并通过可审查提交同步。
不得在错误的 worktree 内修改其他 workstream 的源代码。

## 每次任务与修改前检查

任何 Codex 新任务首先声明下列之一：

```text
WORKSTREAM = MULTIDAG
WORKSTREAM = S2D
WORKSTREAM = ASYMMETRIC_MERC
```

随后在对应工作区运行匹配的一条命令：

```text
python scripts/dev/check_workstream.py multidag
python scripts/dev/check_workstream.py s2d
python scripts/dev/check_workstream.py asymmetric_merc
```

每次修改前还必须检查：

```text
git rev-parse --show-toplevel
git branch --show-current
git remote -v
git status --short
```

guard 校验当前工作目录所属仓库的 origin fetch/push URL、分支、精确根路径、worktree 登记与共享 Git 元数据目录，并确认运行的是当前 worktree 内的 guard。
Windows 路径比较忽略大小写并统一分隔符，Ubuntu 路径区分大小写。
成功输出 `WORKSTREAM`、`PATH`、`BRANCH`、`REMOTE` 与 `WORKSTREAM_CHECK=PASS`；失败输出 `WORKSTREAM_CHECK=FAIL` 并以非零状态退出。
已有 dirty/untracked 项不会使身份检查失败；必须单独检查 status 并保护这些文件。

## Git 操作边界

- **NO git add .**：只按文件路径暂存本任务改动，提交前检查 `git diff --cached --name-status` 和 `git diff --cached --check`。
- **NO blind git push**：核对工作区、分支、remote、提交范围与授权后，使用显式分支或 refspec。
- **NO blind git pull**：先检查身份、状态与远端差异，再决定同步方式；不得替仍在运行的实验盲目拉取代码。
- **NO wrong-worktree source edits**：在负责该组件的工作区修改，并通过 reviewed commit 的 cherry-pick 或 clean milestone merge 交给 integration。
- guard 只读，不自动切 branch、不修改 remote、不 stash，也不执行网络同步。
- 不覆盖、删除或 stash 历史 dirty/untracked 文件；不得把它们混入治理提交。
- 使用 `git worktree` 建立工作区，不复制目录；目标目录已存在时先检查并报告，不删除或覆盖。
- 新分支从共同 governance commit 建立；主工作区始终保留在 MULTIDAG 分支。已有分支不得强制重置，应先核对其历史和工作区归属。

本次 Windows 基础设施任务只创建一个独立 governance commit 与两个本地 worktree，不执行 push、不操作 GitHub UI、不部署远程工作区、不运行训练。

组件组合规则见 [Asymmetric MERC Architecture](ASYMMETRIC_MERC_ARCHITECTURE.md)；工作区与外部资产登记见 [Active Workstreams](../handoffs/ACTIVE_WORKSTREAMS.md)。
