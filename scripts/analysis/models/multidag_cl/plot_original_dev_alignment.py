"""Render the original Dev charts from the bundled CSV directory."""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


COLORS = ("#24548F", "#6184B5", "#06776C", "#51A296", "#A94E16", "#76499A", "#435D6E", "#AC5572")
CLASSES = ("hap", "sad", "neu", "ang", "exc", "fru")
GROUPS = ("NO_HISTORY", "SHIFT", "STABLE")
METRIC_NAMES = {
    "accuracy": ("准确率", "Accuracy"), "weighted_f1": ("加权 F1", "Weighted F1"),
    "macro_f1": ("宏平均 F1", "Macro F1"), "uar": ("UAR", "UAR"),
    "nll": ("硬标签 NLL", "Hard-label NLL"), "brier": ("Brier 分数", "Brier score"),
    "ece10": ("ECE10", "ECE10"),
}


def select_font(path):
    if path is not None:
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path).get_name()
    names = {entry.name for entry in font_manager.fontManager.ttflist}
    return next((name for name in ("Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
                                   "Microsoft YaHei", "DengXian", "SimHei") if name in names), None)


def render(tables, destination, *, font=None):
    family = select_font(font)
    mpl.rcParams.update({"font.family": [family or "DejaVu Sans"], "font.size": 10,
                         "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "axes.unicode_minus": False})

    def label(chinese, english):
        return chinese if family else english

    def load(name):
        return pd.read_csv(tables / f"{name}.csv")

    def axes(rows, cols, size):
        return plt.subplots(rows, cols, figsize=size, layout="constrained", squeeze=False)

    destination.mkdir(parents=True, exist_ok=False)
    index = []

    def save(fig, stem, title, *inputs):
        fig.suptitle(f"{title}\n{label('IEMOCAP 原 Dev · seed100', 'IEMOCAP original Dev · seed100')}",
                     fontsize=15, fontweight="bold")
        for ext in ("png", "svg"):
            fig.savefig(destination / f"{stem}.{ext}", dpi=220, facecolor="white")
        plt.close(fig)
        index.append((stem, title, inputs))

    def setup(ax, title, xlabel=None, ylabel=None):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.22)

    scores = load("overall_metrics").query("status == 'SCORED'")
    arms = list(scores.arm)
    palette = dict(zip(arms, COLORS))
    from_epoch = load("trajectory")
    epoch = label("轮次", "Epoch")
    fit_roles = {"Fit": label("原 Train", "Train"), "Dev": label("原 Dev", "Dev")}

    for stem, title, panels in (
        ("learning", label("训练与验证学习曲线", "Train and Dev learning curves"), (
            ("weighted_f1", label("加权 F1", "Weighted F1"), 100),
            ("hard_label_nll", label("硬标签 NLL", "Hard-label NLL"), 1),
            ("ece10", "ECE10", 1), ("brier", label("Brier 分数", "Brier score"), 1))),
        ("confidence_dynamics", label("置信度随轮次变化", "Confidence across epochs"), (
            ("mean_confidence_correct", label("正确样本平均置信度", "Mean confidence on correct predictions"), 1),
            ("mean_confidence_wrong", label("错误样本平均置信度", "Mean confidence on errors"), 1),
            ("high_confidence_coverage", label("置信度 ≥ 0.90 的比例", "Fraction with confidence ≥ 0.90"), 1),
            ("high_confidence_error_share", label("高置信错误占总体比例", "High-confidence errors / all examples"), 1))),
    ):
        fig, grid = axes(2, 2, (12, 7))
        for ax, (metric, title_ax, scale) in zip(grid.flat, panels):
            for (arm, role), frame in from_epoch.groupby(["arm", "population"], sort=False):
                ax.plot(frame.epoch, scale * frame[metric], color=palette[f"{arm} Best"],
                        ls="-" if role == "Dev" else "--", label=f"{arm} · {fit_roles[role]}")
            setup(ax, title_ax, epoch, "%" if scale == 100 else None)
        grid[0, 0].legend(ncol=2)
        save(fig, stem, title, "trajectory.csv")

    fig, grid = axes(1, 2, (12, 4))
    for ax, (metric, title) in zip(grid.flat, (
        ("train_update_loss", label("训练目标损失", "Training objective loss")),
        ("max_preclip_gradient_norm", label("每轮最大裁剪前梯度范数", "Epoch maximum pre-clip gradient norm")),
    )):
        for arm, frame in from_epoch.query("population == 'Fit'").groupby("arm", sort=False):
            ax.plot(frame.epoch, frame[metric], color=palette[f"{arm} Best"], label=arm)
        setup(ax, title, epoch)
        ax.legend()
    save(fig, "training_updates", label("训练损失与梯度", "Training loss and gradients"), "trajectory.csv")

    fig, grid = axes(1, 2, (12, 4))
    gaps = load("generalization_gap")
    for ax, (metric, title, scale) in zip(grid.flat, (
        ("fit_minus_dev_weighted_f1", label("加权 F1 · Train − Dev", "Weighted F1 · Train − Dev"), 100),
        ("dev_minus_fit_hard_nll", label("硬标签 NLL · Dev − Train", "Hard-label NLL · Dev − Train"), 1),
    )):
        for arm, frame in gaps.groupby("arm", sort=False):
            ax.plot(frame.epoch, frame[metric] * scale, color=palette[f"{arm} Best"], label=arm)
        ax.axhline(0, color="grey", lw=1)
        setup(ax, title, epoch, "pp" if scale == 100 else None)
        ax.legend()
    save(fig, "generalization_gap", label("训练与验证泛化差距", "Train–Dev generalization gap"),
         "generalization_gap.csv")

    fig, grid = axes(2, 2, (12, 8))
    cal, risk = load("reliability"), load("risk_coverage")
    roc_path = tables / "roc_pr.csv"
    roc = load("roc_pr") if roc_path.exists() else pd.DataFrame(columns=["arm", "curve"])
    for arm in arms:
        bins = cal[(cal.arm == arm) & (cal.support > 0)]
        grid[0, 0].plot(bins.mean_probability, bins.observed_rate, marker=".", color=palette[arm], label=arm)
        frame = risk[risk.arm == arm]
        grid[0, 1].plot(frame.coverage, frame.tie_expected_risk, color=palette[arm])
        for ax, kind, x, y in ((grid[1, 0], "ROC", "fpr", "tpr"), (grid[1, 1], "PR", "recall", "precision")):
            frame = roc[(roc.arm == arm) & (roc.curve == kind)]
            if not frame.empty:
                ax.plot(frame[x], frame[y], color=palette[arm])
    grid[0, 0].plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    for ax, (title, x, y) in zip(grid.flat, (
        (label("十分箱可靠性", "10-bin reliability"), label("平均置信度", "Mean confidence"), label("正确率", "Accuracy")),
        (label("风险–覆盖率", "Risk–coverage"), label("覆盖率", "Coverage"), label("接纳样本错误率", "Accepted error rate")),
        (label("错误检测 ROC", "Error-detection ROC"), "FPR", "TPR"),
        (label("错误检测 PR", "Error-detection PR"), label("召回率", "Recall"), label("精确率", "Precision")),
    )):
        setup(ax, title, x, y)
    grid[0, 0].legend(ncol=2)
    save(fig, "reliability_risk_roc_pr", label("置信度、选择性预测与错误检测", "Calibration, selection and error detection"),
         "reliability.csv", "risk_coverage.csv", *(("roc_pr.csv",) if roc_path.exists() else ()))

    fig, grid = axes(1, 3, (15, 4.5))
    per_class = load("per_class")
    for ax, metric, title in zip(grid.flat, ("precision", "recall", "f1"),
                                 (label("精确率", "Precision"), label("召回率", "Recall"), "F1")):
        frame = per_class.pivot(index="label", columns="arm", values=metric).loc[list(CLASSES), arms]
        for arm in arms:
            ax.plot(CLASSES, frame[arm], marker="o", color=palette[arm], label=arm)
        setup(ax, title, label("真值类别", "Gold class"))
    grid[0, 0].legend(ncol=2)
    save(fig, "per_class", label("逐类别指标", "Classwise metrics"), "per_class.csv")

    fig, grid = axes(2, 4, (13, 7))
    confusion = load("confusion")
    for ax, arm in zip(grid.flat, arms):
        frame = confusion[confusion.arm == arm].pivot(index="gold", columns="prediction", values="true_class_fraction")
        image = ax.imshow(frame.loc[list(CLASSES), list(CLASSES)], vmin=0, vmax=1, cmap="Blues")
        ax.set(xticks=range(6), xticklabels=CLASSES, yticks=range(6), yticklabels=CLASSES, title=arm)
    for ax in list(grid.flat)[len(arms):]:
        ax.set_visible(False)
    fig.colorbar(image, ax=list(grid.flat), shrink=.8, label=label("真类内比例", "Fraction within gold class"))
    fig.supxlabel(label("预测类别", "Predicted class"))
    fig.supylabel(label("真值类别", "Gold class"))
    save(fig, "confusion", label("混淆矩阵", "Confusion matrices"), "confusion.csv")

    fig, grid = axes(1, 2, (12, 4))
    shift = load("shift")
    for ax, metric in zip(grid.flat, ("weighted_f1", "accuracy")):
        frame = shift.pivot(index="group", columns="arm", values=metric).reindex(GROUPS)
        for arm in arms:
            ax.plot(GROUPS, frame[arm] * 100, color=palette[arm], marker="o", label=arm)
        setup(ax, label(*METRIC_NAMES[metric]), label("同说话人情绪变化", "Same-speaker emotion change"), "%")
    grid[0, 0].legend(ncol=2)
    save(fig, "emotion_shift", label("情绪转折分析", "Emotion-transition analysis"), "shift.csv")

    fig, grid = axes(2, 2, (15, 8.5))
    transitions = load("r_d_net")
    for ax, group in zip(grid.flat, ("ALL", *GROUPS)):
        frame = transitions[transitions.group == group]
        y = np.arange(len(frame))
        ax.barh(y, frame.r, color=COLORS[2], label=label("纠正 R", "Recovered R"))
        ax.barh(y, -frame.d, color=COLORS[5], label=label("损伤 −D", "Damaged −D"))
        ax.scatter(frame.net, y, color="#172837", s=22, label="NET = R − D")
        ax.set(yticks=y, yticklabels=frame.contrast)
        ax.invert_yaxis()
        setup(ax, group, label("话语数", "Utterances"))
    grid[0, 0].legend(ncol=3, fontsize=8)
    save(fig, "r_d_net", label("配对正误迁移", "Paired prediction transitions"), "r_d_net.csv")

    ci = load("paired_bootstrap")
    interaction = "交互项"

    def forest(ax, frame, labels):
        y = np.arange(len(frame))
        ax.hlines(y, frame.lower_95, frame.upper_95, color=COLORS[0])
        ax.scatter(frame.delta, y, color=COLORS[0], zorder=3)
        ax.axvline(0, color="grey", lw=1, ls="--")
        ax.set(yticks=y, yticklabels=labels)
        ax.invert_yaxis()

    fig, grid = axes(1, 2, (14, 5.5))
    for ax, metric in zip(grid.flat, ("weighted_f1", "accuracy")):
        frame = ci[(ci.metric == metric) & (ci.contrast != interaction)]
        forest(ax, frame, frame.contrast)
        setup(ax, label(*METRIC_NAMES[metric]), label("配对差与 95% 区间（pp）", "Paired delta and 95% interval (pp)"))
    save(fig, "paired_forest", label("整对话配对增益", "Whole-dialogue paired gains"), "paired_bootstrap.csv")

    fig, grid = axes(1, 2, (12, 4.5))
    factorial = load("factorial")
    for objective, frame in factorial.groupby("objective", sort=False):
        grid[0, 0].plot(frame.checkpoint, 100 * frame.weighted_f1, color=palette[f"{objective} Best"],
                        marker="o", label=objective)
    setup(grid[0, 0], label("固定端点加权 F1", "Weighted F1 at fixed endpoints"), ylabel="%")
    grid[0, 0].legend()
    frame = ci[(ci.contrast == interaction) & (ci.units == "pp")]
    forest(grid[0, 1], frame, [label(*METRIC_NAMES[key]) for key in frame.metric])
    setup(grid[0, 1], label("差中之差与 95% 区间", "Difference-in-differences and 95% interval"), "pp")
    save(fig, "factorial", label("D4 二因素交互", "D4 factorial interaction"), "factorial.csv", "paired_bootstrap.csv")

    budget_path = tables / "budget_selection.csv"
    if budget_path.exists():
        budget = load("budget_selection")
        fig, grid = axes(1, 1, (8, 4.6))
        ax = grid[0, 0]
        selected = budget[budget.session == "MEAN"].sort_values(["hard_nll", "updates"]).iloc[0]
        for session, frame in budget.groupby("session", sort=False):
            ax.plot(frame.updates, frame.hard_nll, marker="o", lw=2.8 if session == "MEAN" else 0.9,
                    label=label("四个 Session 等权", "Equal Session mean") if session == "MEAN" else session)
        ax.axvline(selected.updates, color="grey", ls="--", label=f"U* = {int(selected.updates)}")
        setup(ax, label("原 Train 内的四条 CE 验证轨迹", "Four CE validation trajectories within Train"),
              label("优化器更新步", "Optimizer update"), label("硬标签 NLL", "Hard-label NLL"))
        ax.legend(ncol=2)
        save(fig, "budget", label("D3 训练侧预算选择", "D3 Train-side budget selection"), "budget_selection.csv")

    fig, grid = axes(2, 4, (14, 7))
    for ax, metric in zip(grid.flat, METRIC_NAMES):
        ax.barh(arms, scores[metric], color=[palette[a] for a in arms])
        ax.invert_yaxis()
        setup(ax, label(*METRIC_NAMES[metric]))
    grid[1, 3].set_visible(False)
    save(fig, "overview", label("固定 Dev 端点综合表现", "Fixed Dev endpoint results"), "overall_metrics.csv")
    lines = [f"# {label('原 Dev 图目录', 'Original Dev figure index')}", "",
             label("统计范围为同一 Dev 内的 20 段对话、两个演员与一个 Session。",
                   "Scope is the same fixed Dev set of 20 dialogues, two actors and one Session."), ""]
    if not budget_path.exists():
        lines.extend([label("D3 缺少已认证的历史轨迹，未生成预算选择图。详见预算锁。",
                            "D3 has no certified historical trajectories. See the budget lock."), ""])
    lines += [f"- {title} [PNG]({stem}.png) · [SVG]({stem}.svg) · " +
              " · ".join(f"[{name}](../tables/{name})" for name in inputs) for stem, title, inputs in index]
    (destination / "FIGURE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"font": family or "DejaVu Sans", "language": "zh" if family else "en", "figures": len(index),
            "matplotlib": mpl.__version__, "pandas": pd.__version__, "numpy": np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    print(render(args.tables, args.destination, font=args.font))


if __name__ == "__main__":
    main()
