#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成报告用图（中文标签）。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160

# 图1 两类特征点 vs 普通日
s = pd.read_csv(os.path.join(BASE, "特征点_汇总.csv"))
g = s[s.group == "全部"].copy()
labels = {"TB": "趋势成立点", "PB": "调整结束点", "UT": "普通上升日"}
periods = ["2016-2021", "2022-2026"]
setups = ["TB", "PB", "UT"]
x = np.arange(len(setups)); w = 0.35
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for i, per in enumerate(periods):
    vals = [g[(g.setup == st) & (g.period == per)]["20日超额"].values[0] * 100 for st in setups]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=per)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylim(-0.80, 0.30)
ax.set_xticks(x); ax.set_xticklabels([labels[st] for st in setups])
ax.set_ylabel("20日相对中证500超额（%）")
ax.set_title("两类特征点没有跑赢普通上升日")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "图1_两类特征点与普通日.png"), bbox_inches="tight")
plt.close(fig)

# 图2 股性矩阵：形态减普通日
d = pd.read_csv(os.path.join(BASE, "特征点_事件明细.csv"), parse_dates=["date"])
d = d[d.type.notna()]
piv = d.pivot_table(index=["period", "type"], columns="setup", values="ex20", aggfunc="mean")
piv["趋势成立点减普通日"] = (piv["TB"] - piv["UT"]) * 100
piv["调整结束点减普通日"] = (piv["PB"] - piv["UT"]) * 100
mat = piv[["趋势成立点减普通日", "调整结束点减普通日"]].reset_index()
mat["标签"] = mat["period"] + " " + mat["type"]
mat = mat.set_index("标签")[["趋势成立点减普通日", "调整结束点减普通日"]].astype(float)
fig, ax = plt.subplots(figsize=(7.0, 4.6))
im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, fontsize=10)
ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=9)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(v) > 1.2 else "black")
ax.set_title("股性 × 形态：相对普通上升日的20日超额增量（百分点）")
fig.colorbar(im, ax=ax, shrink=0.8, label="百分点")
fig.tight_layout()
fig.savefig(os.path.join(BASE, "图2_股性矩阵.png"), bbox_inches="tight")
plt.close(fig)

# 图3 443笔收益分布
t = pd.read_csv(os.path.join(ROOT, "2026-10-18_成交量融合_2010-2025", "08_深度复盘_2016_2026", "逐笔_2016-2026_低波活跃_结合.csv"))
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ret = t.ret.clip(-0.25, 0.45) * 100
ax.hist(ret, bins=40, color="#6C8EBF", edgecolor="white")
ax.axvline(t.ret.median() * 100, color="#C00000", lw=1.2, linestyle="--", label=f"中位 {t.ret.median()*100:.2f}%")
ax.axvline(t.ret.mean() * 100, color="#2E75B6", lw=1.2, linestyle="-", label=f"均值 {t.ret.mean()*100:.2f}%")
top = t.nlargest(5, "ret")
ax.scatter(top.ret.clip(-0.25, 0.45) * 100, [5] * len(top), color="#C00000", zorder=5, label="前5大盈利（截断显示）")
ax.annotate("前5笔盈利合计 +87.6 万\n超过全部净利润", xy=(44, 8), xytext=(24, 32),
            arrowprops=dict(arrowstyle="->", color="#C00000", lw=1), color="#C00000", fontsize=9)
ax.set_xlabel("单笔净收益（%，极端值截断到 -25%～+45%）")
ax.set_ylabel("交易笔数")
ax.set_title(f"443笔系统回测交易：中位为负，右尾极厚（胜率 {t.ret.gt(0).mean()*100:.1f}%）")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "图3_443笔收益分布.png"), bbox_inches="tight")
plt.close(fig)

# 图4 仓位与回撤
pos = pd.DataFrame([
    ["0.50%", 1.177, -14.9], ["0.75%", 1.266, -21.5], ["1.00%", 1.355, -27.7],
    ["1.50%", 1.588, -37.1], ["固定20%", 1.632, -38.9],
], columns=["risk", "final", "dd"])
fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
bars = ax1.bar(pos.risk, pos.final, color="#9DC3E6", label="期末权益（初始1.0）")
for b, v in zip(bars, pos.final):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax1.set_ylabel("期末权益（倍）")
ax1.set_ylim(1.0, 1.75)
ax2 = ax1.twinx()
ax2.plot(pos.risk, pos.dd, color="#C00000", marker="o", label="最大回撤（%）")
for i, (x_, v) in enumerate(zip(pos.risk, pos.dd)):
    ax2.annotate(f"{v:.1f}%", (i, v), textcoords="offset points", xytext=(0, 11),
                 ha="center", color="#C00000", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
ax2.set_ylabel("最大回撤（%）")
ax2.set_ylim(-45, 0)
ax1.set_title("单笔风险越低，收益略降、回撤显著下降")
ax1.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
fig.text(0.5, -0.06, "注：本图为按卖出顺序的简化复利；实际 443 组合（4 仓、逐日权益）为 +48.75%、最大回撤 -43.6%。",
         ha="center", fontsize=8, color="#555555")
fig.tight_layout()
fig.savefig(os.path.join(BASE, "图4_仓位与回撤.png"), bbox_inches="tight")
plt.close(fig)
print("charts saved")
