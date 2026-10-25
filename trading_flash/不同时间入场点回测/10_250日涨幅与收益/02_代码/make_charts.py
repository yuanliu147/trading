# -*- coding: utf-8 -*-
"""250 日涨幅专题图表。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
RES = BASE / "10_250日涨幅与收益" / "03_结果"
OUT = BASE / "10_250日涨幅与收益" / "04_图表"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RED = "#b54430"
RED2 = "#d46f54"
GREEN = "#2f7665"
GOLD = "#a87832"
INK = "#1c2521"
MUTED = "#72786f"
GRID = "#e7ddce"
BG = "#fffdf8"

bin_df = pd.read_csv(RES / "六档成绩_前二.csv")
q_df = pd.read_csv(RES / "五分位_前二.csv")
yr_df = pd.read_csv(RES / "年度相关性_前二.csv")
vol_df = pd.read_csv(RES / "波动五分位_前二.csv")
tr = pd.read_csv(RES / "逐笔_前二_带250日涨幅.csv")


def style(ax):
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#cfc3b1")
    ax.tick_params(colors=INK, labelsize=9)


def save(fig, name):
    fig.patch.set_facecolor(BG)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", name)


# 1. 六档：平均收益 + 止盈率
fig, ax = plt.subplots(figsize=(9.4, 5.3))
x = np.arange(len(bin_df))
bars = ax.bar(x, bin_df["单笔均值"] * 100, color=RED, width=0.58, label="单笔均值")
ax.set_xticks(x)
ax.set_xticklabels(bin_df["涨幅档"], fontsize=10)
ax.set_ylabel("单笔收益（%）", color=INK)
ax.set_ylim(0, 8.5)
for i, (v, n) in enumerate(zip(bin_df["单笔均值"] * 100, bin_df["笔数"])):
    ax.text(i, v + 0.12, f"{v:+.1f}%", ha="center", va="bottom", fontsize=9, color=RED)
    ax.text(i, -0.62, f"n={n}", ha="center", va="top", fontsize=8.5, color=MUTED)
ax2 = ax.twinx()
ax2.plot(x, bin_df["止盈率"] * 100, color=GOLD, marker="o", linewidth=2, label="止盈率")
ax2.set_ylim(0, 75)
ax2.set_ylabel("止盈率（%）", color=GOLD)
ax2.tick_params(axis="y", colors=GOLD, labelsize=9)
ax2.spines["top"].set_visible(False)
style(ax)
ax.set_title("250 日涨幅越高，止盈率明显上升；平均收益并不单调", fontsize=14, color=INK, pad=14)
lines = [bars, ax2.lines[0]]
ax.legend(lines, ["单笔均值", "止盈率"], loc="upper left", frameon=False, fontsize=9)
save(fig, "01_六档_收益与止盈率.png")

# 2. 五分位：均值/中位 + 胜率/止盈率
fig, ax = plt.subplots(figsize=(9.4, 5.3))
x = np.arange(len(q_df))
w = 0.34
ax.bar(x - w / 2, q_df["单笔均值"] * 100, w, color=RED, label="均值")
ax.bar(x + w / 2, q_df["单笔中位"] * 100, w, color="#e8b49f", label="中位")
ax.set_xticks(x)
ax.set_xticklabels(q_df["五分位"], fontsize=10)
ax.set_ylabel("单笔收益（%）", color=INK)
ax.set_ylim(0, 12)
for i, (m, med) in enumerate(zip(q_df["单笔均值"] * 100, q_df["单笔中位"] * 100)):
    ax.text(i - w / 2, m + 0.15, f"{m:+.1f}", ha="center", fontsize=8.5, color=RED)
    ax.text(i + w / 2, med + 0.15, f"{med:+.1f}", ha="center", fontsize=8.5, color="#a85a3f")
    ax.text(i, -0.9, f"n={q_df['笔数'].iloc[i]}", ha="center", va="top", fontsize=8.5, color=MUTED)
ax2 = ax.twinx()
ax2.plot(x, q_df["止盈率"] * 100, color=GOLD, marker="o", linewidth=2, label="止盈率")
ax2.plot(x, q_df["胜率"] * 100, color=GREEN, marker="s", linewidth=1.6, label="胜率")
ax2.set_ylim(0, 80)
ax2.set_ylabel("概率（%）", color=MUTED)
ax2.tick_params(axis="y", colors=MUTED, labelsize=9)
ax2.spines["top"].set_visible(False)
style(ax)
ax.set_title("五分位：Q1 最弱，Q2–Q5 的均值差距不大；止盈率随涨幅上升", fontsize=14, color=INK, pad=14)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9, ncol=2)
save(fig, "02_五分位_收益与概率.png")

# 3. 散点 + 分档中位
fig, ax = plt.subplots(figsize=(9.4, 5.6))
x = tr["250日涨幅"] * 100
y = tr["单笔收益"] * 100
mask = x <= 300
ax.scatter(x[mask], y[mask], s=18, alpha=0.45, color=RED2, edgecolors="none", label="单笔交易")
ax.axhline(0, color="#9b8d78", linewidth=0.8)
ax.axvline(tr["250日涨幅"].median() * 100, color="#9b8d78", linewidth=0.8, linestyle="--")
ax.set_xlim(0, 300)
ax.set_ylim(-40, 25)
ax.set_xlabel("信号日前 250 个交易日涨幅（%）", color=INK)
ax.set_ylabel("单笔收益（%）", color=INK)
# bin medians
for _, row in bin_df.iterrows():
    mid = {"0–10%": 5, "10–20%": 15, "20–50%": 35, "50–100%": 75, "100–200%": 150, ">200%": 250}[row["涨幅档"]]
    ax.scatter(mid, row["单笔中位"] * 100, s=46, color=GOLD, zorder=5, edgecolors=BG, linewidths=0.8)
    ax.text(mid, row["单笔中位"] * 100 + 1.3, f"{row['单笔中位']*100:+.1f}", ha="center", fontsize=8, color=GOLD)
ax.text(296, -36, f"{int((x>300).sum())} 笔 >300% 未显示", ha="right", fontsize=8, color=MUTED)
style(ax)
ax.set_title("散点没有明显直线关系；金点是各档中位收益", fontsize=14, color=INK, pad=14)
ax.legend(loc="upper left", frameon=False, fontsize=9)
save(fig, "03_散点_250日涨幅_收益.png")

# 4. 年度相关性
fig, ax = plt.subplots(figsize=(9.4, 4.7))
x = np.arange(len(yr_df))
colors = [RED if v >= 0 else GREEN for v in yr_df["Pearson相关"]]
ax.bar(x, yr_df["Pearson相关"], color=colors, width=0.6)
ax.axhline(0, color="#9b8d78", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{int(y)}\n(n={n})" for y, n in zip(yr_df["年份"], yr_df["笔数"])], fontsize=8.5)
ax.set_ylim(-0.9, 0.95)
ax.set_ylabel("Pearson 相关", color=INK)
for i, (v, n) in enumerate(zip(yr_df["Pearson相关"], yr_df["笔数"])):
    ax.text(i, v + (0.04 if v >= 0 else -0.08), f"{v:+.2f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8, color=RED if v >= 0 else GREEN)
style(ax)
ax.set_title("年度相关性正负翻转：没有稳定的“涨得越多越赚”", fontsize=14, color=INK, pad=14)
save(fig, "04_年度相关性.png")

# 5. 波动五分位
fig, ax = plt.subplots(figsize=(9.4, 5.2))
x = np.arange(len(vol_df))
ax.bar(x, vol_df["单笔均值"] * 100, color=RED, width=0.58, label="单笔均值")
ax.set_xticks(x)
ax.set_xticklabels([f"Q{i}\nvol {v:.2f} · n={n}" for i, v, n in zip(range(1, 6), vol_df["平均个股波动"], vol_df["笔数"])], fontsize=8.5)
ax.set_ylabel("单笔收益（%）", color=INK)
ax.set_ylim(0, 10)
for i, (v, n) in enumerate(zip(vol_df["单笔均值"] * 100, vol_df["笔数"])):
    ax.text(i, v + 0.15, f"{v:+.1f}%", ha="center", fontsize=9, color=RED)

ax2 = ax.twinx()
ax2.plot(x, vol_df["止盈率"] * 100, color=GOLD, marker="o", linewidth=2, label="止盈率")
ax2.set_ylim(0, 55)
ax2.set_ylabel("止盈率（%）", color=GOLD)
ax2.tick_params(axis="y", colors=GOLD, labelsize=9)
ax2.spines["top"].set_visible(False)
style(ax)
ax.set_title("个股波动率比 250 日涨幅更有区分度：Q1 到 Q5 均值 +0.9% → +8.3%", fontsize=13, color=INK, pad=14)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9)
save(fig, "05_波动五分位.png")

# 6. 左尾
fig, ax = plt.subplots(figsize=(9.4, 5.0))
x = np.arange(len(bin_df))
ax.plot(x, bin_df["p10"] * 100, color=GREEN, marker="v", linewidth=1.8, label="p10（左尾）")
ax.plot(x, bin_df["单笔中位"] * 100, color=INK, marker="o", linewidth=1.8, label="中位")
ax.plot(x, bin_df["p90"] * 100, color=RED, marker="^", linewidth=1.8, label="p90（右尾）")
ax.axhline(0, color="#9b8d78", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(bin_df["涨幅档"], fontsize=9)
ax.set_ylim(-25, 25)
ax.set_ylabel("单笔收益（%）", color=INK)
style(ax)
ax.set_title("涨幅越高：右尾更容易到 +20%，左尾也更深", fontsize=14, color=INK, pad=14)
ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=3)
save(fig, "06_左尾与右尾.png")

print("charts done")
