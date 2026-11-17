#!/usr/bin/env python3
"""43 个股适配性图表"""
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

C_MACD = "#d95f02"
C_BENCH = "#1f77b4"
C_BH = "#7f7f7f"
C_NEG = "#c0392b"
C_POS = "#2e7d32"


def fig13():
    ev = pd.read_csv(os.path.join(RES, "个股_事件表现.csv"))
    ev = ev[ev["口径"] == "全期"]
    tm = pd.read_csv(os.path.join(RES, "个股_择时表现_分期.csv"))
    tm = tm[tm["期间"] == "全期"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2))
    x = ev["event_ex_style"].dropna() * 100
    axes[0].hist(x.clip(-15, 15), bins=70, color=C_MACD, alpha=0.75, edgecolor="white", linewidth=0.3)
    axes[0].axvline(x.median(), color="#222", ls="--", lw=1.3)
    axes[0].axvline(0, color="#888", lw=0.8)
    axes[0].set_title("逐股金叉 20 日风格中性超额（全期）", fontsize=11, loc="left")
    axes[0].set_xlabel("% / 20 日")
    axes[0].set_ylabel("股票数")
    axes[0].text(0.02, 0.95, f"中位 {x.median():.2f}%｜为正 {(x>0).mean():.1%}｜n={len(x)}",
                 transform=axes[0].transAxes, va="top", fontsize=9.5)

    y = tm["净超同暴露"].dropna() * 100
    axes[1].hist(y.clip(-40, 40), bins=70, color=C_MACD, alpha=0.75, edgecolor="white", linewidth=0.3)
    axes[1].axvline(y.median(), color="#222", ls="--", lw=1.3)
    axes[1].axvline(0, color="#888", lw=0.8)
    axes[1].set_title("逐股 DIF>DEA 择时净超同暴露（全期）", fontsize=11, loc="left")
    axes[1].set_xlabel("% / 年")
    axes[1].set_ylabel("股票数")
    axes[1].text(0.02, 0.95, f"中位 {y.median():.2f}%｜为正 {(y>0).mean():.1%}｜n={len(y)}",
                 transform=axes[1].transAxes, va="top", fontsize=9.5)
    fig.suptitle("个股层不是“少数很好”，而是“大多数为负”", fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "图13_个股分布.png"), bbox_inches="tight")
    plt.close(fig)


def fig14():
    est = pd.read_csv(os.path.join(RES, "个股_事件表现.csv"))
    pe = est.pivot(index="code", columns="口径", values="event_ex_style").dropna()
    tm = pd.read_csv(os.path.join(RES, "个股_择时表现_分期.csv"))
    a = tm[tm["期间"] == "前期"].set_index("code")
    b = tm[tm["期间"] == "后期"].set_index("code")
    j = a.join(b, lsuffix="_前", rsuffix="_后").dropna(subset=["毛超同暴露_前", "毛超同暴露_后"])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    x, y = pe["前期"] * 100, pe["后期"] * 100
    axes[0].scatter(x.clip(-12, 12), y.clip(-12, 12), s=7, alpha=0.25, color=C_MACD, edgecolors="none")
    axes[0].plot([-12, 12], [-12, 12], color="#888", lw=0.8, ls="--")
    axes[0].axhline(0, color="#aaa", lw=0.6); axes[0].axvline(0, color="#aaa", lw=0.6)
    axes[0].set_xlim(-12, 12); axes[0].set_ylim(-12, 12)
    axes[0].set_xlabel("前期逐股超额 % / 20 日"); axes[0].set_ylabel("后期逐股超额 % / 20 日")
    axes[0].set_title(f"事件：前期→后期 r={x.corr(y):.3f}，ρ={pe['前期'].corr(pe['后期'], method='spearman'):.3f}",
                      fontsize=10.5, loc="left")

    x, y = j["毛超同暴露_前"] * 100, j["毛超同暴露_后"] * 100
    axes[1].scatter(x.clip(-40, 40), y.clip(-40, 40), s=7, alpha=0.25, color=C_MACD, edgecolors="none")
    axes[1].plot([-40, 40], [-40, 40], color="#888", lw=0.8, ls="--")
    axes[1].axhline(0, color="#aaa", lw=0.6); axes[1].axvline(0, color="#aaa", lw=0.6)
    axes[1].set_xlim(-40, 40); axes[1].set_ylim(-40, 40)
    axes[1].set_xlabel("前期逐股毛超同暴露 % / 年"); axes[1].set_ylabel("后期逐股毛超同暴露 % / 年")
    axes[1].set_title(f"择时：前期→后期 r={x.corr(y):.3f}，ρ={j['毛超同暴露_前'].corr(j['毛超同暴露_后'], method='spearman'):.3f}",
                      fontsize=10.5, loc="left")
    fig.suptitle("过去好 ≠ 未来好：逐股前期与后期几乎不相关", fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "图14_持续性.png"), bbox_inches="tight")
    plt.close(fig)


def fig15():
    pers = pd.read_csv(os.path.join(RES, "最终_个股适配_持续性.csv"))
    stab = {}
    for _, r in pers.iterrows():
        if str(r["对象"]).startswith("特征稳定性/"):
            stab[str(r["对象"]).split("/", 1)[1]] = r["值"]
    tsp = pd.read_csv(os.path.join(RES, "最终_个股适配_择时价差.csv"))
    esp = pd.read_csv(os.path.join(RES, "最终_个股适配_事件价差.csv"))
    ex = pd.read_csv(os.path.join(RES, "最终_个股适配_事前价差.csv"))
    labels = ["VR20", "ER60", "AC1_60", "换手20", "规模_log", "波动20", "特质波动250", "MA60择时250", "动量12_1", "60日涨幅", "距250日高", "偏度250"]
    st = pd.read_csv(os.path.join(RES, "个股_特征.csv"))
    from scipy.stats import spearmanr
    for f in labels:
        if f in stab:
            continue
        x, y = st[f + "_前期"], st[f + "_后期"]
        ok = np.isfinite(x) & np.isfinite(y) & (st["有效天数_前期"] >= 500) & (st["有效天数_后期"] >= 500)
        if ok.sum() > 100:
            stab[f] = spearmanr(x[ok], y[ok]).correlation
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for ax, src, title, ylab in [
        (axes[0], tsp[tsp["期间"] == "后期"], "择时：同期 Q5-Q1（后期）", "Q5-Q1 毛超同暴露 %/年"),
        (axes[1], ex[ex["对象"] == "择时"], "择时：事前 Q5-Q1（前期特征→后期）", "Q5-Q1 毛超 %/年"),
        (axes[2], esp[esp["期间"] == "后期"], "事件：Q5-Q1（后期）", "Q5-Q1 超额 %/20日"),
    ]:
        for f in labels:
            if f not in stab:
                continue
            row = src[src["特征"] == f]
            if row.empty:
                continue
            if "Q5-Q1毛超" in row:
                y = row["Q5-Q1毛超"].iloc[0] * 100
            elif "Q5-Q1" in row:
                y = row["Q5-Q1"].iloc[0] * 100
            else:
                y = row["Q5Q1"].iloc[0] * 100
            ax.scatter(stab[f], y, s=45, color=C_MACD if y > 0 else C_BENCH, zorder=3)
            ax.annotate(f, (stab[f], y), xytext=(4, 3), textcoords="offset points", fontsize=7.5)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xlim(-0.02, 1.05)
        ax.set_xlabel("前期→后期特征稳定性（Spearman）")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5, loc="left")
    fig.suptitle("“同期很强”的特征（VR20/ER60）几乎不持久；稳定的特征（换手/规模/波动）只给出“少亏”",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "图15_稳定性与效果.png"), bbox_inches="tight")
    plt.close(fig)


def fig16():
    tq = pd.read_csv(os.path.join(RES, "最终_个股适配_择时分位.csv"))
    tq = tq[tq["特征"] == "VR20"]
    st = pd.read_csv(os.path.join(RES, "个股_特征.csv"))
    tm = pd.read_csv(os.path.join(RES, "个股_择时表现_分期.csv"))
    d = st.merge(tm[tm["期间"] == "后期"], on="code")
    d = d[np.isfinite(d["VR20_前期"])]
    r = d["VR20_前期"].rank(method="first", pct=True)
    d["q"] = np.minimum((r * 5).astype(int), 4)
    ex = d.groupby("q")["毛超同暴露"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2))
    for lab, color, marker in [("前期", C_BENCH, "o"), ("后期", C_MACD, "s")]:
        s = tq[tq["期间"] == lab].sort_values("分位")
        axes[0].plot(s["分位"], s["毛超同暴露"] * 100, marker=marker, color=color, lw=1.8, label=lab)
    axes[0].axhline(0, color="#888", lw=0.8)
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].set_xlabel("VR20 五分位（同期）")
    axes[0].set_ylabel("毛超同暴露 %/年")
    axes[0].set_title("同期：VR20 越高，择时越强（前期/后期都单调）", fontsize=10.5, loc="left")
    axes[0].legend(frameon=False)

    axes[1].bar(ex.index + 1, ex.values * 100, color=[C_BENCH if v < 0 else C_MACD for v in ex.values], alpha=0.8)
    axes[1].axhline(0, color="#888", lw=0.8)
    axes[1].set_xticks([1, 2, 3, 4, 5])
    axes[1].set_xlabel("前期 VR20 五分位（事前选股）")
    axes[1].set_ylabel("后期毛超同暴露 %/年")
    axes[1].set_title("事前：Q5-Q1 反而 -1.5%/年（Q5 绝对 -2.5%）", fontsize=10.5, loc="left")
    fig.suptitle("VR20 是“事后趋势性”，不是“事前股票属性”", fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "图16_VR20分位.png"), bbox_inches="tight")
    plt.close(fig)


def fig17():
    ph = pd.read_csv(os.path.join(RES, "最终_个股适配_配对相位.csv"))
    order = ["全A", "全A金叉", "低60全A", "低60金叉", "低60全A+金叉", "低波全A", "低波金叉", "低波全A+金叉"]
    ph = ph.set_index("对象").reindex(order)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    y = np.arange(len(order))
    vals = ph["相位均值"] * 100
    colors = [C_POS if v > 0 else C_NEG for v in vals]
    ax.barh(y, vals, color=colors, alpha=0.78)
    ax.errorbar(vals, y, xerr=[vals - ph["相位最小"] * 100, ph["相位最大"] * 100 - vals],
                fmt="none", ecolor="#444", capsize=3, lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.axvline(0, color="#333", lw=0.9)
    for i, v in enumerate(vals):
        ax.text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlabel("年化 %（20 相位平均，误差棒为最小/最大）")
    ax.set_title("金叉是反转/低波子集的拖累：低 60 日全 A 11.0% → 加金叉 4.5%", fontsize=11.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图17_配对对照.png"), bbox_inches="tight")
    plt.close(fig)


def fig18():
    d = pd.read_csv(os.path.join(RES, "最终_个股适配_组合_可成交.csv"))
    d = d[(d["分组"] == "全市场") & (d["策略"] == "DIF>DEA")]
    labs = ["2010-2026", "2010-2017", "2018-2026"]
    d = d.set_index("区间").reindex(labs)
    x = np.arange(len(labs)); w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    axes[0].bar(x - w, d["年化净"] * 100, w, label="MACD 择时", color=C_MACD)
    axes[0].bar(x, d["年化同暴露"] * 100, w, label="同暴露常数", color=C_BENCH)
    axes[0].bar(x + w, d["年化持有"] * 100, w, label="买入持有", color=C_BH)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labs)
    axes[0].set_ylabel("年化 %")
    axes[0].set_title("收益：择时低于买入持有，和同暴露接近", fontsize=10.5, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].bar(x - w / 2, d["夏普净"], w, label="MACD 择时", color=C_MACD)
    axes[1].bar(x + w / 2, d["夏普同暴露"], w, label="同暴露", color=C_BENCH)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labs)
    axes[1].set_ylabel("夏普")
    axes[1].set_title("风险调整后：全期略差、前期略好、后期明显差", fontsize=10.5, loc="left")
    axes[1].legend(frameon=False, fontsize=8.5)
    for i, (a, b) in enumerate(zip(d["年化净"], d["年化同暴露"])):
        axes[0].text(i, max(a, b) * 100 + 0.6, f"α {(a-b)*100:+.1f}%", ha="center", fontsize=8.5)
    fig.suptitle("全市场 MACD 择时：可成交口径下，对同暴露的 alpha 只有 +0.1%/年", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "图18_组合对比.png"), bbox_inches="tight")
    plt.close(fig)


def fig19():
    d = pd.read_csv(os.path.join(RES, "最终_个股适配_科创板.csv"))
    star = d[d["板块"] == "科创板"]
    labs = ["2019-2021", "2022-2026", "2019-2026"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    x = np.arange(3); w = 0.36
    for j, strat in enumerate(["DIF>DEA", "价格>MA60"]):
        s = star[star["策略"] == strat].set_index("区间").reindex(labs)
        axes[0].bar(x + (j - 0.5) * w, s["净超同暴露"] * 100, w, label=strat,
                    color=C_MACD if j == 0 else C_BENCH)
    axes[0].axhline(0, color="#888", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labs)
    axes[0].set_ylabel("净超同暴露 %/年")
    axes[0].set_title("科创板：2022-2026 +8.0%，但 2019-2021 -6.7%", fontsize=10.5, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5)

    yd = pd.read_csv(os.path.join(RES, "最终_个股适配_科创板逐年.csv"))
    yd = yd[yd["年份"] >= 2020]
    axes[1].bar(yd["年份"].astype(int).astype(str), yd["净超同暴露"] * 100,
                color=[C_POS if v > 0 else C_NEG for v in yd["净超同暴露"]], alpha=0.8)
    axes[1].axhline(0, color="#888", lw=0.8)
    axes[1].set_ylabel("净超同暴露 %/年")
    axes[1].set_title("逐年：2020/2021/2023 为负，不是每年有效", fontsize=10.5, loc="left")
    fig.suptitle("唯一正口袋：科创板 2022-2026 的择时，是 regime 不是稳定股票属性",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "图19_科创板.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig13(); fig14(); fig15(); fig16(); fig17(); fig18(); fig19()
    print("figures done")
