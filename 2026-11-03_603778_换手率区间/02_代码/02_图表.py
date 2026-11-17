# -*- coding: utf-8 -*-
"""生成 603778 换手率研究图表。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.patches import Rectangle

ROOT = "/Users/admin/WorkSpace/trading/2026-11-03_603778_换手率区间"
FIG = f"{ROOT}/04_图表"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "PingFang SC",
    "font.size": 11,
    "axes.unicode_minus": False,
    "figure.facecolor": "#fbf8f2",
    "axes.facecolor": "#fbf8f2",
    "savefig.facecolor": "#fbf8f2",
    "axes.edgecolor": "#cfc5b4",
    "axes.labelcolor": "#4d473d",
    "xtick.color": "#6f6759",
    "ytick.color": "#6f6759",
    "text.color": "#26221c",
    "axes.grid": True,
    "grid.color": "#e5ddcf",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.75,
})

PAPER = "#fbf8f2"
INK = "#26221c"
MUTED = "#6f6759"
RED = "#b23a2b"
GREEN = "#2e6b55"
GOLD = "#9a6a18"
BLUE = "#3d5b72"
LINE = "#cfc5b4"

BINS = [0, 0.5, 1, 2, 3.5, 5, 8, 12, 20, 1e9]
LABELS = ["<0.5", "0.5-1", "1-2", "2-3.5", "3.5-5", "5-8", "8-12", "12-20", ">20"]
BAND_COLORS = ["#e8e1d2", "#d9c9a8", "#c9b077", "#c69a4e", "#b9772f",
               "#b25a35", "#a9442e", "#93302a", "#6f1f1d"]

hist = pd.read_pickle(f"{ROOT}/01_数据/603778_hist.pkl")
full = pd.read_pickle(f"{ROOT}/01_数据/603778_full.pkl")


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)


def phase_strip(ax, y0=0.965, alpha=0.08):
    spans = [
        ("2015-12-31", "2016-12-31", GOLD, "2016\n新股/高送转"),
        ("2016-12-31", "2018-01-01", "#8c6d4a", "2017\n除权下跌"),
        ("2018-01-01", "2021-01-01", BLUE, "2018–20\n地量熊市"),
        ("2021-01-01", "2023-01-01", "#7a6d55", "2021–22\n题材波动"),
        ("2023-01-01", "2025-07-01", "#6b7a6f", "2023–24\n转型HJT"),
        ("2025-07-01", "2026-04-14", RED, "2025–26\n主升"),
        ("2026-04-14", "2026-11-03", GREEN, "崩塌"),
    ]
    for a, b, color, label in spans:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        ax.axvspan(a, b, color=color, alpha=alpha, lw=0, zorder=0)
        mid = a + (b - a) / 2
        ax.text(mid, y0, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.6, color=MUTED, linespacing=1.15)


# ---------------------------------------------------------------- 图1 全景
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9.0), sharex=True,
                               gridspec_kw={"height_ratios": [3.0, 1.45]})
fig.subplots_adjust(left=0.075, right=0.975, top=0.855, bottom=0.075, hspace=0.06)
phase_strip(ax1, y0=0.985)
ax1.plot(hist.date, hist.close_h, color=INK, lw=1.35, zorder=3)
ax1.set_yscale("log")
ax1.set_ylabel("后复权收盘价（对数）", fontsize=10)
fig.text(0.075, 0.965, "603778 国晟科技（原乾景园林）· 价格与换手率全景（2016–2026）",
         fontsize=17, fontweight="bold", color=INK)
fig.text(0.075, 0.927, "后复权价已还原 2016、2017 两次“10 转 15”及后续送转；换手率为官方口径（成交量/流通股本）",
         fontsize=9.4, color=MUTED)
style_ax(ax1)
ax1.set_yticks([20, 30, 50, 70, 100, 150, 200])
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax1.yaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=2))
ax1.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))

notes = [
    ("2016-01-15", "2016-01 新股开板\n单日换手 66.5%", "2016-04-10", 112, RED),
    ("2017-06-14", "2017-06 除权后放量下跌\n换手 45.1%", "2017-09-20", 118, RED),
    ("2021-03-31", "2021-03 冲高派发\n30.1%", "2021-08-01", 62, RED),
    ("2025-10-17", "2025-10 主升起点\n31.8%", "2025-01-10", 44, GOLD),
    ("2026-04-13", "2026-04 历史最高\n换手 19.6%", "2026-06-10", 165, RED),
    ("2026-11-02", "2026-11-02\n6.92 元 / 6.4%", "2026-12-15", 56, GREEN),
]
for d, txt, tx, ty, color in notes:
    dt, tdt = pd.Timestamp(d), pd.Timestamp(tx)
    row = full[full.date == dt]
    if row.empty:
        continue
    ax1.annotate(txt, xy=(dt, row.close_h.iloc[0]), xytext=(tdt, ty),
                 fontsize=8.4, color=color, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.7))

# 换手率柱
colors = [BAND_COLORS[LABELS.index(str(b))] for b in hist.band]
ax2.bar(hist.date, hist.turn, width=1.4, color=colors, linewidth=0, alpha=0.95)
ax2.set_ylabel("换手率（%）", fontsize=10)
ax2.set_ylim(0, 70)
ax2.axhline(hist.turn.median(), color=BLUE, lw=0.9, ls="--", alpha=0.8)
ax2.text(pd.Timestamp("2016-02-01"), hist.turn.median() + 1.8,
         f"历史中位 {hist.turn.median():.1f}%", fontsize=8, color=BLUE)
ax2.axhline(12, color=RED, lw=0.8, ls=":", alpha=0.65)
ax2.text(pd.Timestamp("2016-02-01"), 13, "12%", fontsize=7.5, color=RED)
style_ax(ax2)
ax2.xaxis.set_major_locator(mdates.YearLocator(1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.set_xlim(pd.Timestamp("2015-12-01"), pd.Timestamp("2027-03-01"))

handles = [Rectangle((0, 0), 1, 1, color=c) for c in BAND_COLORS]
fig.legend(handles, [f"{l}%" for l in LABELS], ncol=9, loc="upper left",
           frameon=False, fontsize=8.0, handlelength=0.9, columnspacing=1.0,
           bbox_to_anchor=(0.075, 0.912))
fig.savefig(f"{FIG}/01_价格与换手全景.png", dpi=210, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 图2 分布
fig, ax = plt.subplots(figsize=(11.2, 5.6))
x = hist.turn.clip(lower=0.02)
bins = np.logspace(np.log10(0.02), np.log10(70), 64)
cnt, edges = np.histogram(x, bins=bins)
centers = np.sqrt(edges[:-1] * edges[1:])
bar_colors = []
for c in centers:
    idx = np.searchsorted(BINS, c, side="right") - 1
    bar_colors.append(BAND_COLORS[min(max(idx, 0), len(BAND_COLORS) - 1)])
ax.bar(centers, cnt, width=np.diff(edges) * 0.9, color=bar_colors, edgecolor=PAPER, linewidth=0.3)
ax.set_xscale("log")
ax.set_xlim(0.02, 70)
ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set_xlabel("日换手率（%，对数刻度）", fontsize=10)
ax.set_ylabel("交易日数", fontsize=10)
ax.set_title("换手率分布：一半交易日在 2.5% 以下，但 2025–26 把右尾拉得很长",
             loc="left", fontsize=15.5, fontweight="bold", pad=14)
for q, lab in [(50, "P50"), (75, "P75"), (90, "P90"), (99, "P99")]:
    v = hist.turn.quantile(q / 100)
    ax.axvline(v, color=INK, lw=0.8, ls="--", alpha=0.55)
    ax.text(v, ax.get_ylim()[1] * 0.94, f"{lab}\n{v:.1f}%", ha="center", va="top",
            fontsize=8, color=INK, bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5))
cur = full.turn.iloc[-1]
ax.axvline(cur, color=GREEN, lw=1.6)
ax.text(cur * 1.10, ax.get_ylim()[1] * 0.72, f"当前 {cur:.1f}%\n2026-11-02", ha="left",
        fontsize=9, color=GREEN, fontweight="bold")
ax.text(0.015, 0.95, f"全样本 {len(hist)} 个交易日 · 中位 {hist.turn.median():.1f}% · 均值 {hist.turn.mean():.1f}%",
        transform=ax.transAxes, fontsize=9, color=MUTED)
style_ax(ax)
fig.savefig(f"{FIG}/02_换手率分布.png", dpi=210, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 图3 区间结果
g = hist.groupby("band", observed=True)
stats = pd.DataFrame({
    "n": g.size(),
    "f20": g.f20.median(),
    "win": g.f20.apply(lambda s: (s > 0).mean() * 100),
    "mae": g.mae20.median(),
}).reindex(LABELS)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.5), gridspec_kw={"width_ratios": [1.55, 1]})
y = np.arange(len(LABELS))[::-1]
colors = [RED if v >= 0 else GREEN for v in stats.f20]
ax1.barh(y, stats.f20, color=colors, alpha=0.9, height=0.62)
for yy, v, n in zip(y, stats.f20, stats.n):
    ax1.text(v + (0.15 if v >= 0 else -0.15), yy, f"{v:+.1f}%  (n={n})",
             va="center", ha="left" if v >= 0 else "right", fontsize=8.8,
             color=INK)
ax1.axvline(0, color=INK, lw=0.8)
ax1.set_yticks(y)
ax1.set_yticklabels([f"{l}%" for l in LABELS], fontsize=10)
ax1.set_xlabel("T+1 开盘买入，持有 20 个交易日的收益中位数", fontsize=9.5)
ax1.set_xlim(-5.6, 4.6)
ax1.set_title("按换手率区间：后续 20 日收益中位数", loc="left",
              fontsize=13.5, fontweight="bold", pad=10)
style_ax(ax1)

colors2 = [RED if v >= 50 else GREEN for v in stats.win]
ax2.barh(y, stats.win, color=colors2, alpha=0.9, height=0.62)
for yy, v in zip(y, stats.win):
    ax2.text(v + 1.2, yy, f"{v:.0f}%", va="center", fontsize=8.8, color=INK)
ax2.axvline(50, color=INK, lw=0.8, ls="--", alpha=0.6)
ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.set_xlim(0, 72)
ax2.set_xlabel("T+20 上涨概率（胜率）", fontsize=9.5)
ax2.set_title("胜率", loc="left", fontsize=13.5, fontweight="bold", pad=10)
style_ax(ax2)
ax2.text(0.02, -0.19, "注：全样本 2016–2026，含 2016 新股与 2025–26 泡沫两段极端行情；样本高度重叠，不等于独立交易。",
         transform=ax2.transAxes, fontsize=7.8, color=MUTED)
fig.savefig(f"{FIG}/03_区间后续收益.png", dpi=210, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 图4 2025-26 放大
z = full[full.date >= "2025-07-01"].copy()
z["band"] = pd.cut(z.turn, BINS, labels=LABELS, right=False)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.6, 7.6), sharex=True,
                               gridspec_kw={"height_ratios": [2.7, 1.25], "hspace": 0.06})
ax1.axvspan(pd.Timestamp("2025-07-01"), pd.Timestamp("2026-04-14"), color=RED, alpha=0.05, lw=0)
ax1.axvspan(pd.Timestamp("2026-04-14"), pd.Timestamp("2026-11-05"), color=GREEN, alpha=0.055, lw=0)
ax1.plot(z.date, z.close, color=INK, lw=1.8)
ax1.set_ylabel("不复权收盘价（元）", fontsize=10)
ax1.set_ylim(2.8, 39.5)
ax1.set_title("2025-07 至 2026-11：同一只股票，换手率的含义完全相反",
              loc="left", fontsize=16.5, fontweight="bold", pad=14)
ax1.text(0, 1.012, "主升段：高换手 = 接力；崩塌段：高换手 = 出货/反弹失败。当前 6.4% 落在崩塌段的中等换手区。",
         transform=ax1.transAxes, fontsize=9.4, color=MUTED)
for d, txt, tx, ty, color in [
    ("2025-10-17", "10/17 换手 31.8%\n主升起点", "2025-08-15", 12.0, GOLD),
    ("2026-04-13", "04/13 33.10 顶\n换手 19.6%", "2026-01-20", 36.8, RED),
    ("2026-04-16", "04/16 换手 27.3%\n跌停后放量", "2026-05-20", 30.5, RED),
    ("2026-11-02", "11/02 6.92\n换手 6.4%", "2026-10-02", 4.1, GREEN),
]:
    dt = pd.Timestamp(d)
    row = z[z.date == dt]
    if row.empty:
        continue
    ax1.annotate(txt, xy=(dt, row.close.iloc[0]), xytext=(pd.Timestamp(tx), ty),
                 fontsize=8.4, color=color, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.7))
ax1.axhline(z[z.date == "2026-10-23"].low.iloc[0], color=GREEN, lw=0.8, ls=":", alpha=0.8)
ax1.text(pd.Timestamp("2025-10-25"), 6.25, "10/23 低点 6.72；11/02 盘中 6.48 后收回", fontsize=8, color=GREEN, va="top")
style_ax(ax1)

zc = [BAND_COLORS[LABELS.index(str(b))] for b in z.band]
ax2.bar(z.date, z.turn, width=1.1, color=zc, linewidth=0)
ax2.set_ylabel("换手率（%）", fontsize=10)
ax2.set_ylim(0, 38)
ax2.axvline(pd.Timestamp("2026-04-14"), color=INK, lw=0.8, ls="--", alpha=0.55)
ax2.text(pd.Timestamp("2026-04-14"), 35, " 04/14 开始崩塌", fontsize=8.2, color=INK, va="top")
style_ax(ax2)
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
ax2.set_xlim(pd.Timestamp("2025-06-20"), pd.Timestamp("2026-12-20"))
fig.savefig(f"{FIG}/04_2025_2026放大.png", dpi=210, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 图5 制度翻转
up = hist[(hist.date >= "2025-07-01") & (hist.date <= "2026-04-13")]
down = hist[hist.date >= "2026-04-14"]
gu, gd = up.groupby("band", observed=True), down.groupby("band", observed=True)
su = pd.DataFrame({"f20": gu.f20.median(), "win": gu.f20.apply(lambda s: (s > 0).mean() * 100), "n": gu.size()}).reindex(LABELS)
sd = pd.DataFrame({"f20": gd.f20.median(), "win": gd.f20.apply(lambda s: (s > 0).mean() * 100), "n": gd.size()}).reindex(LABELS)
su.loc[su.n < 5, "f20"] = np.nan
sd.loc[sd.n < 5, "f20"] = np.nan
x = np.arange(len(LABELS))
w = 0.38
fig, ax = plt.subplots(figsize=(12.6, 5.4))
ax.bar(x - w / 2, su.f20, w, color=RED, alpha=0.85, label="主升段 2025-07～2026-04-13")
ax.bar(x + w / 2, sd.f20, w, color=GREEN, alpha=0.85, label="崩塌段 2026-04-14～10-29")
for xi, v, n in zip(x - w / 2, su.f20, su.n):
    if pd.notna(v):
        ax.text(xi, v + (1.2 if v >= 0 else -1.2), f"{v:+.0f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.6, color=RED)
for xi, v, n in zip(x + w / 2, sd.f20, sd.n):
    if pd.notna(v):
        ax.text(xi, v + (1.2 if v >= 0 else -1.2), f"{v:+.0f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.6, color=GREEN)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}%" for l in LABELS])
ax.set_xlabel("换手率区间", fontsize=10)
ax.set_ylabel("T+20 收益中位数（%）", fontsize=10)
ax.set_title("同一换手率，在主升段与崩塌段的后续 20 日收益完全相反",
             loc="left", fontsize=15.5, fontweight="bold", pad=12)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_ylim(-30, 50)
ax.text(0.01, -0.155, "注：主升段/崩塌段各自样本 <5 天的区间不画；两段样本高度重叠，仅作状态对比。",
        transform=ax.transAxes, fontsize=7.8, color=MUTED)
style_ax(ax)
fig.savefig(f"{FIG}/05_制度翻转.png", dpi=210, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 图6 换手 vs 位置
fig, ax = plt.subplots(figsize=(11.4, 6.0))
sc = ax.scatter(hist.turn.clip(lower=0.02), hist.pos250, c=hist.f20.clip(-40, 60),
                cmap="RdYlGn", s=7, alpha=0.55, linewidths=0)
ax.set_xscale("log")
ax.set_xlim(0.02, 70)
ax.set_ylim(0, 102)
ax.set_xlabel("日换手率（%，对数刻度）", fontsize=10)
ax.set_ylabel("收盘价在 250 日区间的位置（%）", fontsize=10)
ax.set_title("换手率越高，越常出现在 250 日高位；低位高换手多数是“失败反弹”",
             loc="left", fontsize=14.5, fontweight="bold", pad=12)
cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.03)
cb.set_label("T+20 收益（%，截断 ±40/60）", fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)
# 当前点
cur_row = full.iloc[-1]
ax.scatter([cur_row.turn], [cur_row.pos250], s=140, facecolors="none",
           edgecolors=INK, linewidths=1.6, zorder=5)
ax.annotate(f"当前 11/02\n换手 {cur_row.turn:.1f}% / 位置 {cur_row.pos250:.0f}%",
            xy=(cur_row.turn, cur_row.pos250), xytext=(20, 25), textcoords="offset points",
            fontsize=8.6, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
style_ax(ax)
fig.savefig(f"{FIG}/06_换手与位置.png", dpi=210, bbox_inches="tight")
plt.close(fig)

print("图表已生成:", sorted(os.listdir(FIG)))
