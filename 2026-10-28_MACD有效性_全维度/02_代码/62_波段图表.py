#!/usr/bin/env python3
"""62 波段判断图表（图35-图37）"""
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

C_POS = "#2e7d32"; C_NEG = "#c0392b"; C_MACD = "#d95f02"; C_KDJ = "#1f77b4"
C_GRAY = "#7f7f7f"; C_PRICE = "#555555"


def fig35():
    ov = pd.read_csv(os.path.join(RES, "59_波段_拐点概览.csv"))
    tr = pd.read_csv(os.path.join(RES, "60_波段_捕获.csv"))
    tr = tr[tr.方案 == "价格确认"].sort_values("阈值")
    low = ov[ov.拐点类型 == "低点"].sort_values("th")
    high = ov[ov.拐点类型 == "高点"].sort_values("th")
    th = low["th"].to_numpy() * 100
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axes[0]
    ax.plot(th, low["确认时已走"] * 100, marker="o", lw=2, color=C_KDJ, label="低点确认：上行已走")
    ax.plot(th, high["确认时已走"] * 100, marker="s", lw=2, color=C_MACD, label="高点确认：下行已走")
    for x, a, b in zip(th, low["确认时已走"] * 100, high["确认时已走"] * 100):
        ax.annotate(f"{a:.0f}%", (x, a), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color=C_KDJ)
        ax.annotate(f"{b:.0f}%", (x, b), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8.5, color=C_MACD)
    ax.set_xticks(th); ax.set_xticklabels([f"{x:.0f}%" for x in th])
    ax.set_ylim(40, 75); ax.set_xlabel("ZigZag 确认阈值")
    ax.set_ylabel("确认时已走完的波段 %")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.set_title("价格确认本身已吃掉约一半波段", fontsize=10.5, loc="left")

    ax = axes[1]
    xx = np.arange(len(th))
    bars = ax.bar(xx, tr["中位收益"] * 100, width=0.55,
                  color=[C_NEG if v < 0 else C_POS for v in tr["中位收益"]])
    for b, v in zip(bars, tr["中位收益"] * 100):
        ax.annotate(f"{v:+.1f}%", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, -14 if v < 0 else 6), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_xticks(xx); ax.set_xticklabels([f"{x:.0f}%" for x in th])
    ax.set_xlim(-0.6, len(th) - 0.4)
    ax.set_xlabel("ZigZag 确认阈值"); ax.set_ylabel("每笔中位净收益 %")
    ax2 = ax.twinx()
    ax2.plot(xx, tr["胜率"] * 100, marker="o", color=C_MACD, lw=1.8)
    ax2.set_ylabel("胜率 %", color=C_MACD); ax2.tick_params(axis="y", colors=C_MACD)
    ax2.set_ylim(30, 50)
    ax.set_title("5–15% 阈值下，中位交易全部亏损", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图35_波段确认滞后.png"))
    plt.close(fig)


def fig36():
    p = pd.read_csv(os.path.join(RES, "60_波段_组合.csv"))
    # 只保留有明确标签的方案
    keep = ["全A等权", "价格确认_5%", "MACD确认多_5%", "KDJ确认多_5%", "双确认多_5%",
            "价格确认_10%", "MACD确认多_10%", "KDJ确认多_10%", "双确认多_10%"]
    p = p[p.方案.isin(keep)].copy()
    order = {k: i for i, k in enumerate(keep)}
    p["o"] = p.方案.map(order); p = p.sort_values("o", ascending=False)
    colors = {"全A等权": C_GRAY}
    for k in keep:
        if k.startswith("价格"): colors[k] = C_PRICE
        elif k.startswith("MACD"): colors[k] = C_MACD
        elif k.startswith("KDJ"): colors[k] = C_KDJ
        elif k.startswith("双"): colors[k] = C_NEG
        else: colors[k] = C_GRAY
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.barh(p.方案, p.年化 * 100, color=[colors[x] for x in p.方案], height=0.62)
    for b, v, s in zip(bars, p.年化 * 100, p.夏普):
        ax.annotate(f"{v:.1f}%  Sharpe {s:.2f}", (max(v, 0), b.get_y() + b.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5)
    ax.axvline(9.84, color=C_GRAY, ls="--", lw=0.9)
    ax.text(9.84, len(p) - 0.3, " 全A等权 9.84%", color=C_GRAY, fontsize=8.5, va="bottom")
    ax.set_xlabel("年化收益 %（2010-2026，T+1 开盘，往返 0.15%）")
    ax.set_xlim(0, 30)
    ax.set_title("加 MACD/KDJ 过滤后，年化与夏普都下降；5% 纯价格的优势只在 2018 年后", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图36_波段策略.png"))
    plt.close(fig)


def fig37():
    fl = pd.read_csv(os.path.join(RES, "59_波段_翻转.csv"))
    fl = fl[fl.阈值 == 0.10]
    rows = [
        ("低点：MACD多", "低点", "MACD多头"),
        ("低点：KDJ多", "低点", "KDJ多头"),
        ("高点：MACD空", "高点", "MACD空头"),
        ("高点：KDJ空", "高点", "KDJ空头"),
    ]
    lab, at, cf = [], [], []
    for name, typ, st in rows:
        r = fl[(fl.拐点类型 == typ) & (fl.状态 == st)].iloc[0]
        lab.append(name); at.append(r["拐点日已满足"] * 100); cf.append(r["确认时已满足"] * 100)
    x = np.arange(len(lab)); w = 0.35
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    b1 = ax.bar(x - w / 2, at, w, color="#b0b0b0", label="拐点当天")
    b2 = ax.bar(x + w / 2, cf, w, color=[C_KDJ, C_KDJ, C_MACD, C_MACD], label="确认日（已反向 10%）")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_ylim(0, 108); ax.set_ylabel("状态满足占比 %")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("低点当天 MACD 多 11%、KDJ 多 5%；反弹 10% 后变为 69%/92%——指标是反弹的结果", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图37_拐点与确认日状态.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig35(); fig36(); fig37()
    print("saved fig35-37")
