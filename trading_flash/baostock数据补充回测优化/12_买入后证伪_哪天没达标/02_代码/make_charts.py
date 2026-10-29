#!/usr/bin/env python3
"""生成买入后证伪研究的图表。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "03_结果"
FIG = BASE / "04_图表"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Heiti SC", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d1d5db",
    "axes.labelcolor": "#1f2937",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "text.color": "#1f2937",
    "axes.grid": True,
    "grid.color": "#e5e7eb",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
INK = "#1f2937"
RED = "#b91c1c"
ORANGE = "#d97706"
GREEN = "#15803d"
BLUE = "#1d4ed8"
GRAY = "#6b7280"


def pct(x, _=None):
    return f"{x:.0f}%"


def load():
    return {
        "sample": pd.read_csv(RES / "基准_样本.csv"),
        "path": pd.read_csv(RES / "基准_逐日.csv"),
        "threshold": pd.read_csv(RES / "收益阈值.csv"),
        "feature": pd.read_csv(RES / "结构特征.csv"),
        "score": pd.read_csv(RES / "预警评分.csv"),
        "composite": pd.read_csv(RES / "组合规则.csv"),
        "early": pd.read_csv(RES / "早退反事实.csv"),
        "auc": pd.read_csv(RES / "预测力_AUC.csv"),
        "cand_path": pd.read_csv(RES / "候选数量_逐日路径.csv"),
        "cand_warn": pd.read_csv(RES / "候选数量_预警.csv"),
        "state_base": pd.read_csv(RES / "路径三态_基准.csv"),
        "state_path": pd.read_csv(RES / "路径三态_逐日.csv"),
        "state_warn": pd.read_csv(RES / "路径三态_转弱.csv"),
        "state_thr": pd.read_csv(RES / "路径三态_阈值.csv"),
        "state_market": pd.read_csv(RES / "路径三态_市场环境.csv"),
    }


def chart_path(t):
    p = t["path"]
    x = p[(p["样本"] == "全部信号前二") & (p["期间"] == "全期")].sort_values("天数")
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(x["天数"], x["全样本路径均值"] * 100, color=BLUE, lw=2.2, marker="o", ms=5,
            label="全样本平均（含已止盈）")
    ax.plot(x["天数"], x["持仓路径均值"] * 100, color=RED, lw=2.2, marker="o", ms=5,
            label="仍在持仓平均")
    ax.plot(x["天数"], x["持仓路径中位"] * 100, color=RED, lw=1.3, ls="--", alpha=.7,
            label="仍在持仓中位")
    ax.axhline(0, color=INK, lw=.8, alpha=.45)
    ax.set_xlabel("买入后交易日（0=买入日收盘）")
    ax.set_ylabel("相对买入价（%）")
    ax.set_title("买入后的平均路径：全样本仍在涨，但“还没止盈的持仓”走平", loc="left",
                 fontsize=14, fontweight="bold", pad=14)
    ax2 = ax.twinx()
    ax2.plot(x["天数"], x["持仓高于买入价概率"] * 100, color=GRAY, lw=1.6, ls=":",
             marker="s", ms=4, label="持仓中高于买入价概率")
    ax2.set_ylabel("高于买入价概率（%）")
    ax2.set_ylim(0, 100)
    ax2.grid(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False, fontsize=9)
    ax.text(x["天数"].max(), x["全样本路径均值"].iloc[-1] * 100 + .35,
            f"+{x['全样本路径均值'].iloc[-1]*100:.1f}%", color=BLUE, ha="right", fontsize=9)
    ax.text(x["天数"].max(), x["持仓路径均值"].iloc[-1] * 100 - .65,
            f"{x['持仓路径均值'].iloc[-1]*100:.1f}%", color=RED, ha="right", fontsize=9)
    ax.set_xticks(x["天数"])
    ax.set_ylim(-3, 6)
    fig.tight_layout()
    fig.savefig(FIG / "01_基准路径.png", dpi=200)
    plt.close(fig)


def heat_values(t, period="全期"):
    days = [1, 2, 3, 5, 10]
    rules = [
        ("收益<-2%", "threshold", -0.02),
        ("收益<0", "threshold", 0.0),
        ("未达+3%", "threshold", 0.03),
        ("未创买入后新高", "feature", "未创买入后新高"),
        ("收盘跌破买入日最低", "feature", "收盘跌破买入日最低"),
        ("收盘低于5日线", "feature", "收盘低于5日线"),
        ("预警分≥4", "score", None),
        ("三项复合", "composite", "跌破买入价+未创新高+破5日线"),
    ]
    vals = np.full((len(rules), len(days)), np.nan)
    ns = np.zeros_like(vals)
    for j, d in enumerate(days):
        for i, (label, kind, key) in enumerate(rules):
            if kind == "threshold":
                r = t["threshold"]
                r = r[(r["样本"] == "全部信号前二") & (r["期间"] == period) &
                      (r["天数"] == d) & np.isclose(r["阈值"], key)]
            elif kind == "feature":
                r = t["feature"]
                r = r[(r["样本"] == "全部信号前二") & (r["期间"] == period) &
                      (r["天数"] == d) & (r["规则"] == key)]
            elif kind == "score":
                r = t["score"]
                r = r[(r["样本"] == "全部信号前二") & (r["期间"] == period) &
                      (r["天数"] == d) & (r["预警分"].isin([4, 5]))]
                if len(r):
                    vals[i, j] = (r["N"] * r["最终亏损率"]).sum() / r["N"].sum()
                    ns[i, j] = r["N"].sum()
                continue
            else:
                r = t["composite"]
                r = r[(r["样本"] == "全部信号前二") & (r["期间"] == period) &
                      (r["天数"] == d) & (r["规则"] == key)]
            if len(r):
                vals[i, j] = r["最终亏损率"].iloc[0]
                ns[i, j] = r["N"].iloc[0]
    return [x[0] for x in rules], days, vals, ns


def chart_heatmap(t):
    cmap = LinearSegmentedColormap.from_list("warn", ["#f8fafc", "#fde68a", "#f97316", "#b91c1c"])
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), sharey=True)
    for ax, period, title in zip(axes, ["2005-2016", "2017-2026"],
                                 ["2005-2016（旧市场）", "2017-2026（现行系统期）"]):
        rules, days, vals, ns = heat_values(t, period)
        im = ax.imshow(vals * 100, cmap=cmap, vmin=15, vmax=90, aspect="auto")
        ax.set_xticks(range(len(days)), [f"D+{d}" for d in days])
        ax.set_yticks(range(len(rules)), rules)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.grid(False)
        for i in range(len(rules)):
            for j in range(len(days)):
                if np.isfinite(vals[i, j]):
                    color = "white" if vals[i, j] > .62 else INK
                    ax.text(j, i, f"{vals[i,j]*100:.0f}%\n{int(ns[i,j])}", ha="center", va="center",
                            fontsize=8.2, color=color)
        ax.set_xlabel("检查点（收盘）")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, pad=.02)
    cbar.set_label("最终亏损率（%）")
    fig.suptitle("哪一天、哪条规则之后，最终亏损率超过 60%？", x=.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(.02, .925, "数字=最终亏损率；下方小字=样本数（全部信号前二，风险集=尚未止盈）",
             fontsize=9, color=GRAY)
    fig.savefig(FIG / "02_条件概率热力图.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def chart_threshold(t):
    th = t["threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, col, title in [(axes[0], "最终亏损率", "最终亏损率"),
                           (axes[1], "深度亏损率", "最终亏损≤-10%的概率")]:
        for d, color, marker in [(3, ORANGE, "o"), (5, RED, "s")]:
            x = th[(th["样本"] == "全部信号前二") & (th["期间"] == "全期") & (th["天数"] == d)].sort_values("阈值")
            ax.plot(x["阈值"] * 100, x[col] * 100, color=color, marker=marker, lw=2,
                    label=f"D+{d}")
        ax.axvline(-2, color=BLUE, lw=1.2, ls="--")
        ax.axvline(3, color=GREEN, lw=1.2, ls=":")
        ax.text(-2, ax.get_ylim()[1] * .97, " 已跌2%", color=BLUE, va="top", fontsize=9)
        ax.text(3, ax.get_ylim()[1] * .97, " 未达+3%", color=GREEN, va="top", fontsize=9)
        ax.set_xlabel("D 日收盘收益低于该阈值（%）")
        ax.set_ylabel(title + "（%）")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.legend(frameon=False)
    fig.suptitle("“没达到+3%”只是弱警告；“已经跌2%”才是强警告", x=.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(.02, .925, "全部信号前二，2005-2026；风险集=尚未止盈。横轴越往右=预期越宽松，越往左=实际越弱。",
             fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0, 1, .91])
    fig.savefig(FIG / "03_阈值曲线.png", dpi=200)
    plt.close(fig)


def chart_score(t):
    s = t["score"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    width = .36
    for k, (d, color) in enumerate([(3, ORANGE), (5, RED)]):
        x = s[(s["样本"] == "全部信号前二") & (s["期间"] == "全期") & (s["天数"] == d)].sort_values("预警分")
        xs = np.arange(len(x)) + (k - .5) * width
        bars = ax.bar(xs, x["最终亏损率"] * 100, width, color=color, alpha=.88, label=f"D+{d}")
        for xi, yi, ni in zip(xs, x["最终亏损率"] * 100, x["N"]):
            ax.text(xi, yi + 1.2, f"{yi:.0f}%", ha="center", va="bottom", fontsize=9, color=color)
            ax.text(xi, 1.2, f"n={ni}", ha="center", va="bottom", fontsize=7.0, color="#374151",
                    rotation=90)
    ax.set_xticks(range(6), [f"{i} 个" for i in range(6)])
    ax.set_xlabel("预警分（0-5）：收盘低于买入价、未创新高、跌破买入日低点、低于5日线、连续两日低点下移")
    ax.set_ylabel("最终亏损率（%）")
    ax.set_ylim(0, 85)
    ax.set_title("预警分越高，最终亏损率越高；D+3 已经能分层", loc="left",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left")
    ax.text(.99, .98, "全部信号前二，2005-2026", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=GRAY)
    fig.tight_layout()
    fig.savefig(FIG / "04_预警分.png", dpi=200)
    plt.close(fig)


def chart_post_warning(t):
    df = pd.read_parquet(RES / "买入后逐日路径.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    for ax, d0 in zip(axes, [3, 5]):
        cond = df[df.d == d0].set_index(["signal_date", "code"])["ret"].lt(-0.02)
        df["_grp"] = df.set_index(["signal_date", "code"]).index.map(cond)
        for g, color, label in [(True, RED, f"D+{d0} 收盘<-2%"), (False, BLUE, "未触发")]:
            x = df[(df._grp == g) & (df.d <= 14)]
            m = x.groupby("d")["ret"].mean() * 100
            ax.plot(m.index, m.values, color=color, lw=2.2, marker="o", ms=3.5, label=label)
            ax.annotate(f"{m.iloc[-1]:+.1f}%", (m.index[-1], m.iloc[-1]),
                        textcoords="offset points", xytext=(4, -10 if g else 8),
                        fontsize=9, color=color, fontweight="bold")
        ax.axvline(d0, color=GRAY, lw=1, ls="--", alpha=.7)
        ax.axhline(0, color=INK, lw=.8, alpha=.4)
        ax.set_xlim(0, 14)
        ax.set_ylim(-10, 12)
        ax.set_xlabel("买入后交易日（0=买入日收盘）")
        ax.set_title(f"D+{d0} 触发 vs 未触发", loc="left", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, loc="lower left")
    axes[0].set_ylabel("平均收益（%，相对买入价）")
    fig.suptitle("触发后不是一路下跌：会反弹，但弹不回买入价", x=.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(.02, .925, "全部信号前二，2005-2026；假设一直持有到第 14 天，含后来止盈的路径，不是实际止盈后收益。",
             fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0, 1, .91])
    fig.savefig(FIG / "08_触发后走势.png", dpi=200)
    plt.close(fig)


def chart_early(t):
    e = t["early"]
    rules = ["收益<-2%", "未达+3%", "未创买入后新高", "预警分>=4", "跌破买入价+未创新高+破5日线"]
    x = e[(e["样本"] == "全部信号前二") & (e["期间"] == "全期") & (e["天数"] == 5)]
    x = x[x["规则"].isin(rules)].set_index("规则").loc[rules].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    xx = np.arange(len(x))
    w = .38
    axes[0].bar(xx - w/2, x["持有均值"] * 100, w, color=RED, alpha=.85, label="持有到最终")
    axes[0].bar(xx + w/2, x["早退均值"] * 100, w, color=GRAY, alpha=.85, label="D+6 开盘早退")
    for i, (a, b) in enumerate(zip(x["持有均值"], x["早退均值"])):
        axes[0].text(i + w/2, b * 100 - 1.2, f"{b*100:.1f}%", ha="center", va="top", fontsize=8,
                     color=GRAY)
        axes[0].text(i - w/2, a * 100 + .6, f"{a*100:.1f}%", ha="center", va="bottom", fontsize=8,
                     color=RED)
    axes[0].axhline(0, color=INK, lw=.8)
    axes[0].set_xticks(xx, [r.replace("跌破买入价+未创新高+破5日线", "三项复合") for r in x["规则"]])
    plt.setp(axes[0].get_xticklabels(), rotation=18, ha="right")
    axes[0].set_ylabel("平均收益（%）")
    axes[0].set_title("早退的平均收益更低", loc="left", fontsize=12, fontweight="bold")
    axes[0].legend(frameon=False)
    axes[1].bar(xx - w/2, x["持有<=-15%"] * 100, w, color=RED, alpha=.85, label="持有到最终")
    axes[1].bar(xx + w/2, x["早退<=-15%"] * 100, w, color=GRAY, alpha=.85, label="D+6 开盘早退")
    for i, (a, b) in enumerate(zip(x["持有<=-15%"], x["早退<=-15%"])):
        axes[1].text(i - w/2, a * 100 + .5, f"{a*100:.1f}%", ha="center", va="bottom", fontsize=8,
                     color=RED)
        axes[1].text(i + w/2, b * 100 + .5, f"{b*100:.1f}%", ha="center", va="bottom", fontsize=8,
                     color=GRAY)
    axes[1].set_xticks(xx, [r.replace("跌破买入价+未创新高+破5日线", "三项复合") for r in x["规则"]])
    plt.setp(axes[1].get_xticklabels(), rotation=18, ha="right")
    axes[1].set_ylabel("最终≤-15%的概率（%）")
    axes[1].set_title("早退能压深尾，但不是免费午餐", loc="left", fontsize=12, fontweight="bold")
    axes[1].legend(frameon=False)
    fig.suptitle("警告能预测“最终会亏”，但不能让“现在卖”更赚钱", x=.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(.02, .925, "全部信号前二，2005-2026，D+5 触发、D+6 开盘卖；毛收益，未扣早退费用。",
             fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0, 1, .91])
    fig.savefig(FIG / "05_早退对比.png", dpi=200)
    plt.close(fig)



def chart_candidates(t):
    order = ["1-2", "3-9", "10-19", "20-49", "50-99", "100+"]
    colors = ["#9ca3af", "#6b7280", "#1d4ed8", "#15803d", "#d97706", "#b91c1c"]
    p = t["cand_path"]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    ax = axes[0]
    for b, c in zip(order, colors):
        x = p[p["候选级别"] == b].sort_values("天数")
        ax.plot(x["天数"], x["路径均值"] * 100, color=c, lw=2.0, marker="o", ms=3,
                label=f"候选 {b}")
        offs = {"1-2": (4, 8), "3-9": (4, -13), "10-19": (4, 4),
                "20-49": (4, 7), "50-99": (4, 4), "100+": (4, 4)}
        ax.annotate(f"{x['路径均值'].iloc[-1]*100:+.1f}%", (x["天数"].iloc[-1], x["路径均值"].iloc[-1] * 100),
                    textcoords="offset points", xytext=offs.get(b, (4, 4)), fontsize=8, color=c)
    ax.axhline(0, color=INK, lw=.8, alpha=.4)
    ax.set_xlim(0, 14)
    ax.set_xlabel("买入后交易日（0=买入日收盘）")
    ax.set_ylabel("平均收益（%，相对买入价）")
    ax.set_title("基准走势：候选越多不一定越好", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")

    ax = axes[1]
    w = t["cand_warn"]
    w = w[(w["期间"] == "全期") & (w["天数"] == 5) & (w["规则"] == "收益<-2%")].set_index("候选级别")
    bs = order[::-1]
    y = np.arange(len(bs))
    for i, b in enumerate(bs):
        r = w.loc[b]
        ax.plot([r["同桶基准亏损率"] * 100, r["最终亏损率"] * 100], [i, i], color="#d1d5db", lw=3, zorder=1)
        ax.scatter(r["同桶基准亏损率"] * 100, i, s=60, color=GRAY, zorder=2, label="同桶基准" if i == 0 else None)
        ax.scatter(r["最终亏损率"] * 100, i, s=70, color=RED, zorder=3, label="D+5 收盘<-2%" if i == 0 else None)
        ax.text(r["最终亏损率"] * 100 + 1.2, i, f"{r['最终亏损率']*100:.0f}%", va="center", fontsize=8.5, color=RED)
        ax.text(r["同桶基准亏损率"] * 100 - 1.2, i, f"{r['同桶基准亏损率']*100:.0f}%", va="center", ha="right",
                fontsize=8.5, color=GRAY)
        ax.text(86.5, i, f"n={int(r['笔数'])}｜+{r['差值pp']:.0f}pp", va="center", fontsize=7.6, color=INK)
    ax.set_yticks(y, [f"{b}" for b in bs])
    ax.set_xlim(20, 100)
    ax.set_xlabel("最终亏损率（%）")
    ax.set_ylabel("信号日候选数")
    ax.set_title("D+5 触发后：1-9 更可靠，20-49/100+ 反弹更强", loc="left", fontsize=12, fontweight="bold")

    fig.suptitle("信号日候选数量不同，买入后的走势和预警效果不一样", x=.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(.02, .925, "全部信号前二，2005-2026；右图 D+5 收盘<-2%，灰=同桶基准，红=触发；同桶基准=该桶当天尚未止盈的最终亏损率。",
             fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0, 1, .91])
    fig.savefig(FIG / "09_候选数量分层.png", dpi=200)
    plt.close(fig)

def chart_path_states(t):
    p = t["state_path"]
    b = t["state_base"]
    colors = {"上涨": GREEN, "中震荡": ORANGE, "下跌": RED}
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    offs = {"上涨": (4, 4), "中震荡": (4, -12), "下跌": (4, 4)}
    for st, color in colors.items():
        x = p[p["状态"] == st].sort_values("天数")
        ax.plot(x["天数"], x["路径均值"] * 100, color=color, lw=2.4, marker="o", ms=3.4,
                label=f"D+3 {st}")
        ax.annotate(f"{x['路径均值'].iloc[-1]*100:+.1f}%",
                    (x["天数"].iloc[-1], x["路径均值"].iloc[-1] * 100),
                    textcoords="offset points", xytext=offs.get(st, (4, 4)),
                    fontsize=9, color=color, fontweight="bold")
    ax.axvline(3, color=GRAY, lw=1, ls="--", alpha=.8)
    ax.text(3.15, 13.6, "D+3 分态", color=GRAY, fontsize=9)
    ax.axhline(0, color=INK, lw=.8, alpha=.4)
    ax.set_xlim(0, 14)
    ax.set_ylim(-9, 15)
    ax.set_xlabel("买入后交易日（0=买入日收盘）")
    ax.set_ylabel("平均收益（%，相对买入价）")
    ax.set_title("三态之后：上涨组继续走高，下跌组回不到买入价", loc="left",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    ax = axes[1]
    bb = b[(b["样本"] == "全部信号前二") & (b["期间"] == "全期")].set_index("状态").loc[["上涨", "中震荡", "下跌"]]
    xs = np.arange(3)
    w = .36
    ax.bar(xs - w / 2, bb["最终亏损率"] * 100, w,
           color=[colors[s] for s in bb.index], alpha=.9, label="最终亏损率")
    ax.bar(xs + w / 2, bb["深度亏损率"] * 100, w, color="#9ca3af", alpha=.9,
           label="最终≤-10%")
    for i, (s, r) in enumerate(bb.iterrows()):
        ax.text(i - w / 2, r["最终亏损率"] * 100 + 1.2, f"{r['最终亏损率']*100:.1f}%",
                ha="center", va="bottom", fontsize=9, color=colors[s], fontweight="bold")
        ax.text(i + w / 2, r["深度亏损率"] * 100 + 1.2, f"{r['深度亏损率']*100:.1f}%",
                ha="center", va="bottom", fontsize=8.5, color=GRAY)
    ax.set_xticks(xs, [f"{s}\nN={int(r['笔数'])}\n均值{r['均值']*100:+.1f}%"
                       for s, r in bb.iterrows()])
    ax.set_ylim(0, 75)
    ax.set_ylabel("概率（%）")
    ax.set_title("同一批交易，三态把结果拉开 45.8pp", loc="left",
                 fontsize=12, fontweight="bold")

    fig.suptitle("D+3 收盘：上涨 / 中震荡 / 下跌，之后怎么走", x=.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(.02, .925,
             "全部信号前二，2005-2026；分态是事后分组，D+3 之前的线只用于看三类交易如何分化，不能用于提前识别。",
             fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0, 1, .91])
    fig.savefig(FIG / "10_路径三态.png", dpi=200)
    plt.close(fig)


def chart_state_warning(t):
    w = t["state_warn"]
    w = w[(w["样本"] == "全部信号前二") & (w["期间"] == "全期")]
    picks = {
        "上涨": ["D+5 回吐到成本下", "D+5 距 D+0–5 最高回撤≥10%",
                 "D+5 比 D+3 收盘回吐≥5pp", "D+5 未创新高（相对 D+3）"],
        "中震荡": ["D+5 < -2%", "D+5 < -5%", "D+5 预警分≥4", "D+5 三项复合"],
        "下跌": ["D+5 预警分≥4", "D+5 三项复合", "D+5 比 D+3 再跌≥2pp", "D+5 < -2%"],
    }
    colors = {"上涨": GREEN, "中震荡": ORANGE, "下跌": RED}
    rows = []
    for st in ["上涨", "中震荡", "下跌"]:
        for rule in picks[st]:
            r = w[(w["状态"] == st) & (w["规则"] == rule)]
            if len(r):
                rows.append(r.iloc[0])

    n = len(rows)
    fig, ax = plt.subplots(figsize=(11.8, 7.4))
    ys = np.arange(n)[::-1]
    first = True
    group_bounds = []
    last_state = None
    for y, r in zip(ys, rows):
        st = r["状态"]
        if last_state is not None and st != last_state:
            group_bounds.append(y + .5)
        last_state = st
        base = r["同态基准"] * 100
        trig = r["最终亏损率"] * 100
        ax.plot([base, trig], [y, y], color="#d1d5db", lw=3.2, zorder=1)
        ax.scatter(base, y, s=58, color=GRAY, zorder=2,
                   label="同态基准" if first else None)
        ax.scatter(trig, y, s=72, color=colors[st], zorder=3,
                   label="触发" if first else None)
        first = False
        ax.text(base - 1.0, y, f"{base:.0f}%", va="center", ha="right",
                fontsize=8.2, color=GRAY)
        ax.text(trig + 1.0, y, f"{trig:.0f}%", va="center", ha="left",
                fontsize=8.6, color=colors[st], fontweight="bold")
        ax.text(86, y, f"n={int(r['触发笔数'])}  {r['差值pp']:+.0f}pp", va="center",
                ha="left", fontsize=7.6, color=INK)
    for y in group_bounds:
        ax.axhline(y, color="#e5e7eb", lw=1, ls=":")
    # 状态分组标签放在左侧空白
    start = 0
    for st in ["上涨", "中震荡", "下跌"]:
        cnt = sum(1 for r in rows if r["状态"] == st)
        if cnt:
            yc = np.mean(ys[start:start + cnt])
            ax.text(8.5, yc, st, color=colors[st], fontsize=11, fontweight="bold",
                    va="center", ha="left")
        start += cnt
    ax.set_yticks(ys, [r["规则"] for r in rows])
    ax.set_xlim(8, 98)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("最终亏损率（%）")
    ax.set_title("同一批交易，三态内部的“转弱”不是同一条规则", loc="left",
                 fontsize=13, fontweight="bold", pad=12)
    fig.text(.02, .925,
             "全部信号前二，2005-2026；灰=同态基准，红/橙/绿=触发；n=触发笔数，右侧=相对同态基准的差值。",
             fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0, 1, .91])
    fig.savefig(FIG / "11_三态转弱.png", dpi=200)
    plt.close(fig)


def chart_market_regime(t):
    m = t["state_market"]
    colors = {"上涨": GREEN, "中震荡": ORANGE, "下跌": RED}
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), sharey=True)
    for ax, period, title in [(axes[0], "全期", "2005-2026"),
                              (axes[1], "2017-2026", "2017-2026")]:
        mm = m[m["期间"] == period]
        x = mm[mm["状态"] != "全部"]
        regimes = ["上涨市", "中震荡市", "下跌市"]
        width = .25
        for k, st in enumerate(["上涨", "中震荡", "下跌"]):
            vals, xs = [], []
            for j, rg in enumerate(regimes):
                r = x[(x["市场环境"] == rg) & (x["状态"] == st)]
                vals.append(r["最终亏损率"].iloc[0] * 100 if len(r) else np.nan)
                xs.append(j + (k - 1) * width)
            ax.bar(xs, vals, width, color=colors[st], alpha=.9, label=st)
            for xi, vi in zip(xs, vals):
                if np.isfinite(vi):
                    ax.text(xi, vi + 1.0, f"{vi:.0f}%", ha="center", va="bottom",
                            fontsize=8.0, color=colors[st])
        allv = [mm[(mm["市场环境"] == rg) & (mm["状态"] == "全部")]["最终亏损率"].iloc[0] * 100
                for rg in regimes]
        ax.scatter([j + .42 for j in range(3)], allv, marker="D", s=26, color=INK, zorder=4,
                   label="该环境全部")
        ax.set_xticks(range(3), regimes)
        ax.set_ylim(0, 82)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    axes[0].set_ylabel("最终亏损率（%）")
    fig.suptitle("市场环境分层：三态效应在每个环境里都同向，但市场环境本身不稳",
                 x=.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(.02, .925,
             "全部信号前二；D+3 三态为个股路径，市场环境=主板等权指数相对 250 日线、120 日收益与 250 日线斜率。",
             fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0, 1, .91])
    fig.savefig(FIG / "12_市场环境稳健.png", dpi=200)
    plt.close(fig)


def chart_stability(t):
    c = t["composite"]
    rules = ["收益<-2%", "未达+3%", "未创买入后新高", "预警分>=4", "跌破买入价+未创新高+破5日线"]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for d, color, marker in [(3, ORANGE, "o"), (5, RED, "s")]:
        a = c[(c["样本"] == "全部信号前二") & (c["期间"] == "2005-2016") & (c["天数"] == d)]
        b = c[(c["样本"] == "全部信号前二") & (c["期间"] == "2017-2026") & (c["天数"] == d)]
        m = a.merge(b, on="规则", suffixes=("_05", "_17"))
        m = m[m["规则"].isin(rules)]
        ax.scatter(m["最终亏损率_05"] * 100, m["最终亏损率_17"] * 100, s=70, color=color,
                   marker=marker, alpha=.9, label=f"D+{d}", edgecolor="white", linewidth=.8)
        if d == 5:
            offsets = {
                "收益<-2%": (7, -12),
                "未达+3%": (-6, 10),
                "未创买入后新高": (8, 5),
                "预警分>=4": (8, 5),
                "跌破买入价+未创新高+破5日线": (8, 5),
            }
            for _, r in m.iterrows():
                short = r["规则"].replace("跌破买入价+未创新高+破5日线", "三项复合")
                ax.annotate(short, (r["最终亏损率_05"] * 100, r["最终亏损率_17"] * 100),
                            textcoords="offset points", xytext=offsets.get(r["规则"], (6, 4)),
                            fontsize=7.8, color=INK)
    ax.plot([30, 80], [30, 80], color=GRAY, lw=1, ls="--", alpha=.7)
    ax.set_xlabel("2005-2016 最终亏损率（%）")
    ax.set_ylabel("2017-2026 最终亏损率（%）")
    ax.set_xlim(30, 80)
    ax.set_ylim(30, 80)
    ax.set_title("两个样本期，预警规则落在 45 度线附近", loc="left",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "06_分期稳定性.png", dpi=200)
    plt.close(fig)


def chart_auc(t):
    a = t["auc"]
    x = a[(a["样本"] == "全部信号前二") & (a["期间"] == "全期")].sort_values("天数")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(x["天数"], x["AUC_收益"], color=RED, marker="o", lw=2.2, label="D 日收益")
    ax.plot(x["天数"], x["AUC_预警分"], color=ORANGE, marker="s", lw=2.2, label="预警分")
    ax.plot(x["天数"], x["AUC_复合"], color=GRAY, marker="^", lw=1.6, ls="--", label="三项复合")
    ax.axhline(.5, color=INK, lw=.8, ls=":", alpha=.6)
    ax.text(x["天数"].min(), .505, "随机=0.5", fontsize=8, color=GRAY)
    ax.set_ylim(.5, .9)
    ax.set_xlabel("检查点（买入后交易日）")
    ax.set_ylabel("AUC（预测最终亏损）")
    ax.set_title("D 日收益本身就是最强单变量；预警分没有额外优势", loc="left",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "07_AUC.png", dpi=200)
    plt.close(fig)


def main():
    t = load()
    chart_path(t)
    chart_heatmap(t)
    chart_threshold(t)
    chart_score(t)
    chart_early(t)
    chart_post_warning(t)
    chart_candidates(t)
    chart_path_states(t)
    chart_state_warning(t)
    chart_market_regime(t)
    chart_stability(t)
    chart_auc(t)
    print("图表已生成：", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
