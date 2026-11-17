#!/usr/bin/env python3
"""68 零轴回踩图表（图38）"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度/03_结果"
FIG = "/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度/04_图表"
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#eeeeee"
plt.rcParams["grid.linewidth"] = 0.8

C_NEG = "#c0392b"; C_MACD = "#d95f02"; C_KDJ = "#1f77b4"; C_GRAY = "#7f7f7f"


def main():
    d = pd.read_csv(os.path.join(RES, "64_零轴回踩_日线.csv"))
    w = pd.read_csv(os.path.join(RES, "66_零轴回踩_周线.csv"))
    b = pd.read_csv(os.path.join(RES, "65_零轴回踩_波段.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ax = axes[0]
    labs = ["破0翻红", "≤20%", "≤30%", "≤50%"]
    dnames = ["回踩破0再翻红", "红柱回踩≤20%峰值", "红柱回踩≤30%峰值", "红柱回踩≤50%峰值"]
    wnames = ["周线回踩破0再翻红", "周线红柱回踩≤20%", "周线红柱回踩≤30%", "周线红柱回踩≤50%"]
    dd = [d[(d.形态 == x) & (d.周期 == 20)].iloc[0] for x in dnames]
    ww = [w[(w.形态 == x) & (w.周期 == 20)].iloc[0] for x in wnames]
    x = np.arange(len(labs))
    ax.plot(x, [v["mean"] * 100 for v in dd], marker="o", lw=2, color=C_NEG, label="日线 20 日超额")
    for xi, v, t in zip(x, [v["mean"] * 100 for v in dd], [v["t"] for v in dd]):
        ax.annotate(f"{v:+.2f}%\nt={t:.1f}", (xi, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.2, color=C_NEG)
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylim(-1.0, -0.35)
    ax.set_ylabel("20 日超额 %"); ax.set_xlabel("回踩深度（占 20 日峰值比例，越小越深）")
    wtxt = "周线 20 日：" + "、".join(f"{l} {v['mean']*100:+.2f}%" for l, v in zip(labs, ww))
    ax.text(0.30, 0.97, wtxt + "\n（t 均 < -3）", transform=ax.transAxes, fontsize=7.8,
            color=C_GRAY, va="top")
    ax.set_title("日线（红线）与周线都负：回踩后再放大不是买点", fontsize=10.5, loc="left")

    ax = axes[1]
    names_b = ["回踩破0再翻红", "红柱回踩≤10%峰值", "红柱回踩≤20%峰值", "红柱回踩≤30%峰值", "红柱回踩≤50%峰值"]
    bb = [b[(b.形态 == x) & (b.分组 == "全部")].iloc[0] for x in names_b]
    x2 = np.arange(len(names_b))
    ax.bar(x2 - 0.18, [v["形态中位剩余"] * 100 for v in bb], 0.34, color=C_NEG, label="有形态：剩余空间")
    ax.bar(x2 + 0.18, [v["基准中位剩余"] * 100 for v in bb], 0.34, color="#b0b0b0", label="无形态：剩余空间")
    for xi, v in zip(x2, bb):
        ax.annotate(f"{v['形态中位剩余']*100:.1f}%\n{v['形态中位天数']:.0f}天", (xi - 0.18, v["形态中位剩余"] * 100),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.6, color=C_NEG)
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_xticks(x2); ax.set_xticklabels(["破0翻红", "≤10%", "≤20%", "≤30%", "≤50%"])
    ax.set_ylabel("到波段高点的中位剩余空间 %"); ax.set_xlabel("回踩深度（比例越小越深）")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("在已确认上行波段里：形态日离高点更近、剩余更少", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图38_零轴回踩.png"))
    plt.close(fig)
    print("saved")


if __name__ == "__main__":
    main()
