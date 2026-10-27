#!/usr/bin/env python3
"""生成报告图表（PNG，150dpi，中文字体）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "03_结果"
OUT = ROOT / "04_图表"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linestyle"] = "--"

C_A = "#8a8f98"
C_B = "#c0392b"
C_C = "#1f6fb2"
C_IDX = "#d39b1f"


def read_eq(folder: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(R / folder / f"{label}_每日权益.csv", parse_dates=["date"])
    df["nav"] = df.equity / df.equity.iloc[0]
    df["dd"] = df.nav / df.nav.cummax() - 1
    return df


def equity_chart(folder: str, c_folder: str, tag: str, title: str, log: bool):
    a = read_eq(folder, "A_仅当前上市")
    b = read_eq(folder, "B_含退市股")
    c = read_eq(c_folder, "C_排除ST")
    idx = pd.read_parquet(ROOT / "01_数据/market_daily.parquet")
    idx["date"] = pd.to_datetime(idx.date)
    idx = idx.set_index("date").sort_index()
    idx = (1 + idx.market_return.fillna(0)).cumprod()
    idx = idx.loc[a.date.iloc[0]:a.date.iloc[-1]]
    idx = idx / idx.iloc[0]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(idx.index, idx.values, color=C_IDX, lw=1.6, ls="--", label="主板等权指数")
    ax.plot(a.date, a.nav, color=C_A, lw=1.8, label="A 仅当前上市")
    ax.plot(b.date, b.nav, color=C_B, lw=1.8, label="B 含退市股")
    ax.plot(c.date, c.nav, color=C_C, lw=2.0, label="C 含退市+排风险警示")
    if log:
        ax.set_yscale("log")
        ax.set_yticks([0.5, 1, 2, 5, 10, 20, 40, 80])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}x"))
    ax.set_title(title, fontsize=13, loc="left", pad=10)
    ax.set_ylabel("净值（起点=1）")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}.png", bbox_inches="tight")
    plt.close(fig)


def dd_chart(folder: str, c_folder: str, tag: str, title: str):
    a = read_eq(folder, "A_仅当前上市")
    b = read_eq(folder, "B_含退市股")
    c = read_eq(c_folder, "C_排除ST")
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    for df, color, label in [(a, C_A, "A 仅当前上市"), (b, C_B, "B 含退市股"), (c, C_C, "C 含退市+排风险警示")]:
        ax.fill_between(df.date, df.dd * 100, 0, color=color, alpha=0.10)
        ax.plot(df.date, df.dd * 100, color=color, lw=1.6, label=label)
    ax.set_title(title, fontsize=13, loc="left", pad=10)
    ax.set_ylabel("回撤（%）")
    ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}.png", bbox_inches="tight")
    plt.close(fig)


def annual_chart(folder: str, c_folder: str, tag: str, title: str):
    def read_annual(label, fld=None):
        p = R / (fld or folder) / f"{label}_年度.csv"
        s = pd.read_csv(p, index_col=0).iloc[:, 0] * 100
        s.index = s.index.astype(str)
        return s
    a = read_annual("A_仅当前上市")
    b = read_annual("B_含退市股")
    c = read_annual("C_排除ST", c_folder)
    years = sorted(set(a.index) | set(b.index) | set(c.index))
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    w = 0.27
    for i, (s, color, label) in enumerate([(a, C_A, "A 仅上市"), (b, C_B, "B 含退市"), (c, C_C, "C 含退市+排风险警示")]):
        vals = [s.get(y, np.nan) for y in years]
        ax.bar(x + (i - 1) * w, vals, width=w, color=color, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=0, fontsize=8)
    ax.set_ylabel("年度收益（%）")
    ax.set_title(title, fontsize=13, loc="left", pad=10)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}.png", bbox_inches="tight")
    plt.close(fig)


def chart_002684():
    bars = pd.read_parquet(ROOT / "01_数据/bars.parquet",
                           columns=["code", "date", "close", "preclose", "trade_status", "volume"])
    bars["date"] = pd.to_datetime(bars.date)
    d = bars[(bars.code == "sz.002684") & (bars.date.between("2022-03-01", "2022-07-15"))].copy()
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    traded = d[(d.trade_status == 1) & (d.volume > 0)]
    pre = traded[traded.date < "2022-05-05"]
    post = traded[traded.date >= "2022-06-06"]
    ax.plot(pre.date, pre.close, color=C_B, lw=1.8, marker="o", ms=2.5, label="实际收盘价")
    ax.plot(post.date, post.close, color=C_B, lw=1.8, marker="o", ms=2.5)
    ax.axvspan(pd.Timestamp("2022-05-05"), pd.Timestamp("2022-06-02"), color="gray", alpha=0.15, label="停牌")
    ax.axvline(pd.Timestamp("2022-04-21"), color="#2c7a4b", ls=":", lw=1.2)
    ax.axvline(pd.Timestamp("2022-04-26"), color="#2c7a4b", ls="--", lw=1.2)
    ax.axvline(pd.Timestamp("2022-06-06"), color=C_B, ls="--", lw=1.2)
    ax.annotate("信号 4/21\n收盘 2.75", xy=(pd.Timestamp("2022-04-21"), 2.75), xytext=(-10, 18),
                textcoords="offset points", fontsize=8.5, color="#2c7a4b")
    ax.annotate("买入 4/26 开盘 2.69", xy=(pd.Timestamp("2022-04-26"), 2.69), xytext=(8, 30),
                textcoords="offset points", fontsize=8.5, color="#2c7a4b",
                arrowprops=dict(arrowstyle="-", color="#2c7a4b", lw=0.8))
    ax.annotate("退市整理首日 6/6\n开盘 0.46（-80%）", xy=(pd.Timestamp("2022-06-06"), 0.46),
                xytext=(12, 45), textcoords="offset points", fontsize=8.5, color=C_B,
                arrowprops=dict(arrowstyle="-", color=C_B, lw=0.8))
    ax.annotate("卖出 6/6 开盘 0.46\n-82.9%", xy=(pd.Timestamp("2022-06-06"), 0.46),
                xytext=(-96, 34), textcoords="offset points", fontsize=9, color=C_B, weight="bold",
                arrowprops=dict(arrowstyle="-", color=C_B, lw=0.8))
    ax.set_title("002684 *ST猛狮：停牌后直接进入退市整理，15日到期也卖不出", fontsize=13, loc="left", pad=10)
    ax.set_ylabel("实际价（元）")
    ax.set_xlim(pd.Timestamp("2022-03-01"), pd.Timestamp("2022-07-15"))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(OUT / "07_002684退市过程.png", bbox_inches="tight")
    plt.close(fig)


def chart_risk():
    df = pd.read_csv(R / "风险叠加汇总.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, period in zip(axes, ["2017-2026", "2005-2026"]):
        d = df[df["期间"] == period]
        for _, r in d.iterrows():
            color = C_C if r["标签"] == "C_基础_无止损" else (C_B if "止损" in r["标签"] else C_A)
            ax.scatter(-r["max_dd"] * 100, r["cagr"] * 100, s=46, color=color, zorder=3)
            ax.annotate(r["标签"].replace("C_", ""), (-r["max_dd"] * 100, r["cagr"] * 100),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_title(period, fontsize=11, loc="left")
        ax.set_xlabel("最大回撤（%，绝对值）")
    axes[0].set_ylabel("年化收益（%）")
    fig.suptitle("止损与仓位：收益-回撤前沿", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "08_止损与仓位前沿.png", bbox_inches="tight")
    plt.close(fig)


def chart_event():
    a = pd.read_csv(R / "事件研究_2005_2016.csv")
    b = pd.read_csv(R / "事件研究_2017_2026.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for ax, dim, title in [(axes[0], "20日换手率", "20日平均换手率"), (axes[1], "市场20日", "市场20日涨跌幅")]:
        xa = a[a["维度"] == dim].set_index("分组")
        xb = b[b["维度"] == dim].set_index("分组")
        order = [g for g in (["<1%", "1-2%", "2-4%", "4-7%", "7-10%", ">10%"]
                             if dim == "20日换手率" else ["-7~-5%", "-10~-7%", "-15~-10%", "<-15%"])
                 if g in xa.index or g in xb.index]
        x = np.arange(len(order))
        ax.bar(x - 0.2, [xa["均值"].get(g, np.nan) * 100 for g in order], width=0.38,
               color=C_A, label="2005-2016")
        ax.bar(x + 0.2, [xb["均值"].get(g, np.nan) * 100 for g in order], width=0.38,
               color=C_C, label="2017-2026")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=0, fontsize=8)
        ax.set_title(title + "：信号后15日平均收益", fontsize=11, loc="left")
        ax.set_ylabel("平均收益（%）")
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "09_事件研究_换手与市场深度.png", bbox_inches="tight")
    plt.close(fig)


def chart_stress():
    # 2008 年 C 与指数
    c = read_eq("风险叠加/2005-2026", "C_基础_无止损")
    c = c[c.date.dt.year == 2008]
    idx = pd.read_parquet(ROOT / "01_数据/market_daily.parquet")
    idx["date"] = pd.to_datetime(idx.date)
    idx = idx.set_index("date").sort_index()
    idx = (1 + idx.market_return.fillna(0)).cumprod()
    idx = idx.loc[c.date.iloc[0]:c.date.iloc[-1]]
    idx = idx / idx.iloc[0]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(c.date, c.nav, color=C_C, lw=1.9, label="C 策略（含退市+排风险警示）")
    ax.plot(idx.index, idx.values, color=C_IDX, lw=1.5, ls="--", label="主板等权指数")
    ax.set_title("2008 年：策略 -52.7%，主板等权指数 -58.9%", fontsize=13, loc="left", pad=10)
    ax.set_ylabel("年内净值（起点=1）")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "10_2008压力年.png", bbox_inches="tight")
    plt.close(fig)


def main():
    equity_chart("2017_2026", "指标组合_2017_2026", "01_权益对比_2017_2026", "2017-2026 净值：退市股与风险警示过滤的影响（对数轴）", True)
    dd_chart("2017_2026", "指标组合_2017_2026", "02_回撤对比_2017_2026", "2017-2026 回撤：含退市股后最大回撤从 -15.8% 扩大到 -54.3%")
    annual_chart("2017_2026", "指标组合_2017_2026", "03_年度收益_2017_2026", "2017-2026 年度收益")
    equity_chart("2005_2026", "指标优化/2005-2026", "04_权益对比_2005_2026", "2005-2026 净值：全周期年化 13%-18%（对数轴）", True)
    dd_chart("2005_2026", "指标优化/2005-2026", "05_回撤对比_2005_2026", "2005-2026 回撤：2008 年最大回撤 -72% 到 -75%")
    annual_chart("2005_2026", "指标优化/2005-2026", "06_年度收益_2005_2026", "2005-2026 年度收益")
    chart_002684()
    chart_risk()
    chart_event()
    chart_stress()
    print("charts done", list(OUT.glob("*.png")))


if __name__ == "__main__":
    main()
