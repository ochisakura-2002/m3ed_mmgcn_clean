# Git 工作流

本文档记录本地 Windows、GitHub、远程 V100 之间的 Git 使用规则。

## 任务开始前检查

每个任务开始前先执行：

```powershell
git status --short
```

```powershell
git diff --stat
```

如果需要查看具体修改，再执行：

```powershell
git diff -- <path>
```

不要在不了解当前工作区修改来源的情况下覆盖文件。

## feature branch 工作流

建议每个较大任务使用单独分支：

```powershell
git switch -c feature/<short-name>
```

如果分支已存在：

```powershell
git switch feature/<short-name>
```

分支命名建议：

1. `feature/mmgcn-smoke`
2. `feature/modality-ablation`
3. `fix/evaluate-checkpoint`
4. `docs/codex-context`

## 什么时候 pull

在以下情况 pull：

1. 开始正式任务前。
2. 本地准备基于远程最新代码继续开发。
3. 远程 V100 准备运行刚从 GitHub 同步的代码。

推荐：

```powershell
git pull --ff-only
```

如果当前有未提交修改，先不要 pull，先处理或 stash。

## 什么时候 commit

在以下情况 commit：

1. 一个小问题已经完整解决。
2. smoke test 或必要检查已通过。
3. `git diff --stat` 可审查。
4. 没有误包含数据、checkpoint、cache 或大体积输出。

提交前检查：

```powershell
git status --short
```

```powershell
git diff --stat
```

除非用户明确要求，Codex 不要自行 commit。

## 什么时候 push

在以下情况 push：

1. 本地代码需要去远程 V100 训练。
2. 需要让 GitHub 保存当前可复现版本。
3. 需要在另一台机器继续开发或评估。

推荐：

```powershell
git push -u origin <branch>
```

已经设置 upstream 后：

```powershell
git push
```

## 远程 V100 如何 pull 本地已提交修改

本地完成 commit 和 push 后，远程 V100 执行：

```bash
git fetch origin
```

```bash
git switch <branch>
```

```bash
git pull --ff-only
```

然后在远程确认：

```bash
git status --short
```

正式训练前记录 commit：

```bash
git rev-parse HEAD
```

## 有未提交修改时如何 stash

保存当前修改：

```powershell
git stash push -m "work in progress before pull"
```

包含未跟踪文件：

```powershell
git stash push -u -m "work in progress before pull"
```

查看 stash：

```powershell
git stash list
```

恢复最近 stash：

```powershell
git stash pop
```

恢复但保留 stash：

```powershell
git stash apply
```

## 如何回退未提交修改

回退某个已跟踪文件的未提交修改：

```powershell
git restore <path>
```

回退 staged 状态：

```powershell
git restore --staged <path>
```

删除未跟踪文件前必须先确认列表：

```powershell
git clean -nd
```

真正删除未跟踪文件是破坏性操作，必须确认后再执行：

```powershell
git clean -fd
```

不要用 `git reset --hard` 作为常规回退手段。

## 如何回退已经 commit 但未 push 的修改

保留文件修改，撤销最近一次 commit：

```powershell
git reset --soft HEAD~1
```

撤销最近一次 commit，并把改动留在工作区：

```powershell
git reset --mixed HEAD~1
```

这些命令会改写本地历史。执行前先确认没有其他人基于该 commit 工作。

## 如何安全回退已经 push 的 commit

已经 push 的 commit 优先使用 revert：

```powershell
git revert <commit>
```

然后：

```powershell
git push
```

不要对已经 push 且他人可能使用的分支随意 force push。

## 如何创建备份 tag

在重要实验或大改前创建 tag：

```powershell
git tag backup/<name>
```

推送 tag：

```powershell
git push origin backup/<name>
```

查看 tag：

```powershell
git tag
```

## 如何创建备份 branch

创建当前状态的备份分支：

```powershell
git switch -c backup/<name>
```

推送备份分支：

```powershell
git push -u origin backup/<name>
```

回到原分支：

```powershell
git switch <branch>
```

## 哪些实验产物可以提交

可以提交小型、可审查的实验产物：

1. YAML 配置。
2. 小型 CSV 摘要。
3. Markdown 实验记录。
4. 小型图表，如果确实用于报告且文件不大。
5. smoke test 配置。
6. smoke test 代码。

提交前确认没有包含个人绝对路径、服务器私有路径或大体积数据。

## 哪些实验产物绝对不要提交

不要提交：

1. 真实数据。
2. feature pkl。
3. checkpoint。
4. tensor dump。
5. 大体积 `outputs/`。
6. `tmp/`。
7. `logs/`。
8. `wandb/`。
9. `mlruns/`。
10. Python cache。
11. 本地虚拟环境。
12. 个人 IDE 配置。

当前 `.gitignore` 已覆盖这些主要类别。若未来新增大体积目录，先讨论命名和忽略规则，不要把个人路径写进 `.gitignore`。
