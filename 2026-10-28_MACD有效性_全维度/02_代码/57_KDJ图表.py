#!/usr/bin/env python3
"""57 MACD×KDJ 图表（图26-图34）"""
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

C_POS = "#2e7d32"
C_NEG = "#c0392b"
C_MACD = "#d95f02"
C_KDJ = "#1f77b4"
C_GRAY = "#7f7f7f"


def pct(x):
    return x * 100


def fig26():
    f = pd.read_csv(os.path.join(RES, "KDJ_状态_四态.csv"))
    lab = ["MACD多+KDJ多", "MACD多+KDJ空", "MACD空+KDJ多", "MACD空+KDJ空"]
    order = [("KDJ多", "MACD多"), ("KDJ空", "MACD多"), ("KDJ多", "MACD空"), ("KDJ空", "MACD空")]
    vals = [f[(f.KDJ == a) & (f.MACD == b)]["日均超额"].iloc[0] * 100 for a, b in order]
    ts = [f[(f.KDJ == a) & (f.MACD == b)]["t值"].iloc[0] for a, b in order]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = [C_NEG, "#e08a7a", "#8fb3d9", C_NEG]
    bars = ax.bar(lab, vals, color=colors, width=0.62)
    for b, v, t in zip(bars, vals, ts):
        ax.annotate(f"{v:+.2f}%\nt={t:+.1f}", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, -24 if v < 0 else 8), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.axhline(0, color="#555", lw=0.9)
    ax.axhline(-0.15, color=C_GRAY, lw=0.8, ls="--")
    ax.text(1.8, -0.13, "成本 -0.15%", color=C_GRAY, fontsize=8, ha="center")
    ax.set_ylabel("未来 20 日超额 %")
    ax.set_ylim(min(vals) * 1.35, 0.15)
    ax.set_title("四态全部为负：MACD多+KDJ多最差（-0.65%，t=-8.6），KDJ空反而少亏", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图26_KDJ四态.png"))
    plt.close(fig)


def fig27():
    d = pd.read_csv(os.path.join(RES, "KDJ_状态_条件.csv"))
    q = d[d.条件族 == "K五分位"].copy()
    q["q"] = q.条件.str[-1].astype(int)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for macd, lab, color in [("多头日均", "MACD 多", C_MACD), ("空头日均", "MACD 空", C_KDJ)]:
        y = q[macd].to_numpy() * 100
        ax.plot(q.q, y, marker="o", lw=2, color=color, label=lab)
        offs = {1: (0, -16), 2: (0, 10), 3: (0, 10), 4: (0, -16), 5: (0, 10)} if macd == "多头日均" else {1: (0, 10), 2: (0, 10), 3: (0, 10), 4: (0, 10), 5: (0, -16)}
        for x, v in zip(q.q, y):
            dx, dy = offs[int(x)]
            ax.annotate(f"{v:+.2f}%", (x, v), xytext=(dx, dy),
                        textcoords="offset points", ha="center", fontsize=8.5, color=color)
    ax.axhline(0, color="#555", lw=0.9)
    ax.axhline(-0.15, color=C_GRAY, lw=0.8, ls="--")
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["Q1\n最低", "Q2", "Q3\n中间", "Q4", "Q5\n最高"])
    ax.set_xlabel("KDJ K 值的横截面五分位")
    ax.set_ylabel("未来 20 日超额 %")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("KDJ 水平是 U 形：Q1、Q5 差，Q3/Q4 较好；但 Q3 也只有 +0.14%（MACD多）", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图27_KDJ水平U形.png"))
    plt.close(fig)


def fig28():
    d = pd.read_csv(os.path.join(RES, "KDJ_事件_MACD金叉_q.csv"))
    keep = ["全部MACD金叉", "KQ1(最低)", "KQ2", "KQ3", "KQ4", "KQ5(最高)",
            "KDJ多(K>D)", "K>70", "同日KDJ金叉", "KDJ顶背离", "KDJ底背离"]
    d = d[d.过滤.isin(keep)].copy()
    d["order"] = d.过滤.apply(lambda x: keep.index(x) if x in keep else 99)
    d = d.sort_values("order")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    y = np.arange(len(d))
    vals = d.净超额.to_numpy() * 100
    colors = [C_NEG if v < 0 else C_POS for v in vals]
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v, t, n in zip(y, vals, d.t值, d.信号数):
        ax.annotate(f"{v:+.2f}%  t={t:+.1f}", (v, yi), xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(d.过滤)
    ax.invert_yaxis()
    ax.axvline(0, color="#555", lw=0.9)
    ax.set_xlabel("未来 20 日净超额 %（已扣 0.15% 往返成本）")
    ax.set_xlim(min(vals) * 1.35, 0.25)
    ax.set_title("MACD 金叉的 KDJ 过滤：没有一种能转正；同日双金叉、KQ5、K>70 最差", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图28_事件过滤_MACD金叉.png"))
    plt.close(fig)


def fig29():
    d = pd.read_csv(os.path.join(RES, "KDJ_事件_KDJ金叉_q.csv"))
    keep = ["全部KDJ金叉", "MACD多", "MACD空", "前5日内MACD金叉", "前5日内无MACD金叉",
            "KQ5(最高)", "KDJ顶背离", "KDJ底背离"]
    d = d[d.过滤.isin(keep)].copy()
    d["order"] = d.过滤.apply(lambda x: keep.index(x) if x in keep else 99)
    d = d.sort_values("order")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = np.arange(len(d))
    vals = d.净超额.to_numpy() * 100
    colors = [C_NEG if v < 0 else C_POS for v in vals]
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v, t in zip(y, vals, d.t值):
        ax.annotate(f"{v:+.2f}%  t={t:+.1f}", (v, yi), xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(d.过滤)
    ax.invert_yaxis()
    ax.axvline(0, color="#555", lw=0.9)
    ax.set_xlabel("未来 20 日净超额 %（已扣 0.15% 往返成本）")
    ax.set_xlim(min(vals) * 1.35, 0.2)
    ax.set_title("KDJ 金叉：在 MACD 多头里、在 MACD 金叉后 5 日内更差；确认越晚越差", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图29_事件过滤_KDJ金叉.png"))
    plt.close(fig)


def fig30():
    d = pd.read_csv(os.path.join(RES, "KDJ_参数稳健.csv"))
    d = d[d.网格 == "KDJ"].copy()
    metrics = [("MACD金叉_KQ5-KQ2_4", "MACD金叉：KQ5 - KQ2-4"),
               ("KDJ金叉_MACD多-空", "KDJ金叉：MACD多 - MACD空")]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ns = [5, 9, 14, 19, 26]
    smooth = ["M33", "M53", "M35"]
    for ax, (metric, title) in zip(axes, metrics):
        x = d[d.指标 == metric].copy()
        x["N"] = x.参数.str.extract(r"N(\d+)").astype(int)
        x["S"] = x.参数.str.extract(r"(M\d+)")
        mat = np.full((len(ns), len(smooth)), np.nan)
        for i, n in enumerate(ns):
            for j, s in enumerate(smooth):
                sub = x[(x.N == n) & (x.S == s)]
                if len(sub):
                    mat[i, j] = sub.差值.iloc[0] * 100
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-0.8, vmax=0.2)
        ax.set_xticks(range(len(smooth)))
        ax.set_xticklabels(["3,3", "5,3", "3,5"])
        ax.set_yticks(range(len(ns)))
        ax.set_yticklabels([f"N={n}" for n in ns])
        ax.set_xlabel("平滑参数 M1,M2")
        ax.set_title(title, fontsize=10.5)
        for i in range(len(ns)):
            for j in range(len(smooth)):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
        ax.grid(False)
    fig.suptitle("参数稳健：KQ5 惩罚与“KDJ金叉在MACD多头更差”在 15 组 KDJ 参数下全部为负", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "图30_参数稳健.png"))
    plt.close(fig)


def fig31():
    d = pd.read_csv(os.path.join(RES, "KDJ_事件_逐年差异.csv"))
    labs = ["同日双金叉 vs 非", "KQ5 vs KQ2-4", "KQ1 vs KQ2-4", "KDJ多 vs 空", "K>70 vs <="]
    years = list(range(2010, 2027))
    mat = np.full((len(labs), len(years)), np.nan)
    for i, lab in enumerate(labs):
        sub = d[d.对比 == lab]
        for _, r in sub.iterrows():
            if int(r.年份) in years:
                mat[i, years.index(int(r.年份))] = r.差值 * 100
    fig, ax = plt.subplots(figsize=(12.2, 3.8))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-1.2, vmax=1.2)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, fontsize=8)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs)
    ax.grid(False)
    for i in range(len(labs)):
        for j in range(len(years)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.5)
    ax.set_title("逐年差异（pp）：确认型过滤多数年份为负；同日双金叉 16/17 年为负", fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图31_逐年.png"))
    plt.close(fig)


def fig32():
    s = pd.read_csv(os.path.join(RES, "KDJ_K20解剖.csv"))
    y = pd.read_csv(os.path.join(RES, "KDJ_K20逐年.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax = axes[0]
    labels = s.口径.tolist()
    means = s.均值.to_numpy() * 100
    meds = s.中位.to_numpy() * 100
    x = np.arange(len(labels))
    ax.bar(x - 0.18, means, width=0.34, color="#e08a7a", label="均值")
    ax.bar(x + 0.18, meds, width=0.34, color="#7f9bb5", label="中位")
    for xi, v in zip(x - 0.18, means):
        ax.annotate(f"{v:+.2f}", (xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7.5)
    for xi, v in zip(x + 0.18, meds):
        ax.annotate(f"{v:+.2f}", (xi, v), xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=7.5)
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("同一格，三种口径：中位始终为负；≥50 只/日才转正", fontsize=10.5)
    ax = axes[1]
    colors = [C_POS if v > 0 else C_NEG for v in y.日均超额]
    ax.bar(y.年份.astype(str), y.日均超额 * 100, color=colors, width=0.68)
    for xi, v in zip(range(len(y)), y.日均超额 * 100):
        ax.annotate(f"{v:+.1f}", (xi, v), xytext=(0, 4 if v >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=6.5)
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_xticks(range(len(y)))
    ax.set_xticklabels(y.年份.astype(str), rotation=45, fontsize=7.5)
    ax.set_ylabel("日度均值 %")
    ax.set_title("逐年：2014-2016、2020、2022 为负；2021/2026 为正", fontsize=10.5)
    fig.suptitle("MACD多 + KDJ K<20：均值 +0.58%，中位 -0.99%，正均值由少数暴跌反弹日拉高", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "图32_K20解剖.png"))
    plt.close(fig)


def fig33():
    d = pd.read_csv(os.path.join(RES, "KDJ_事件_期限.csv"))
    keep = ["MACD金叉全部", "MACD金叉+KQ2-4", "MACD金叉+KQ5", "MACD金叉+同日KDJ金叉", "MACD金叉+K>70"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    width = 0.15
    for k, lab in enumerate(keep):
        sub = d[d.类别 == lab].set_index("周期").reindex([5, 20, 60])
        vals = sub.净超额.to_numpy() * 100
        x = np.arange(3) + (k - 2) * width
        ax.bar(x, vals, width=width, label=lab.replace("MACD金叉", "金叉"))
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:+.1f}", (xi, v), xytext=(0, -10), textcoords="offset points",
                        ha="center", fontsize=6.5)
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["5 日", "20 日", "60 日"])
    ax.set_ylabel("净超额 %")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("期限：确认惩罚不是一日噪声，20-60 日仍为负；KQ5/双金叉 20-60 日更差", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图33_期限.png"))
    plt.close(fig)


def fig34():
    d = pd.read_csv(os.path.join(RES, "KDJ_周线极值_参数.csv"))
    x = d[(d.周期 == 20) & (d.过滤.str.contains("周K<20增量"))].copy()
    x["N"] = x.KDJ参数.astype(int)
    x = x.sort_values("N")
    y = pd.read_csv(os.path.join(RES, "KDJ_周线极值_逐年.csv"))
    y = y[y.过滤 == "周K<20"].sort_values("年份")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax = axes[0]
    ax.bar(x.N.astype(str), x["mean"] * 100, color=[C_POS if v > 0 else C_NEG for v in x["mean"]])
    for xi, v, t in zip(range(len(x)), x["mean"] * 100, x.t):
        ax.annotate(f"{v:+.2f}%\nt={t:+.1f}", (xi, v), xytext=(0, 4 if v >= 0 else -20),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_xlabel("周线 KDJ 的 N 参数（M=3,3）")
    ax.set_ylabel("增量 %（对比周K>=20）")
    ax.set_title("周K<20 + 日线MACD金叉：5 组参数全为正，t 1.2-2.1", fontsize=10)
    ax = axes[1]
    colors = [C_POS if v > 0 else C_NEG for v in y.日均超额]
    ax.bar(y.年份.astype(str), y.日均超额 * 100, color=colors, width=0.68)
    for xi, v in zip(range(len(y)), y.日均超额 * 100):
        ax.annotate(f"{v:+.1f}", (xi, v), xytext=(0, 4 if v >= 0 else -11),
                    textcoords="offset points", ha="center", fontsize=6.5)
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_xticks(range(len(y)))
    ax.set_xticklabels(y.年份.astype(str), rotation=45, fontsize=7.5)
    ax.set_ylabel("20 日超额 %")
    ax.set_title("逐年：2013、2020 为负；其余多数为正", fontsize=10)
    fig.suptitle("周线 KDJ 超卖 + 日线 MACD 金叉：方向为正但不稳，不能单独当买入策略", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "图34_周线极值.png"))
    plt.close(fig)


def main():
    fig26(); fig27(); fig28(); fig29(); fig30(); fig31(); fig32(); fig33(); fig34()
    print("done")


if __name__ == "__main__":
    main()
