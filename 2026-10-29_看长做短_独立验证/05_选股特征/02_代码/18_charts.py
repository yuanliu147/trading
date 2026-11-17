#!/usr/bin/env python3
"""Charts for the selection-layer report."""
from __future__ import annotations

import os
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from importlib import import_module

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/05_选股特征"
OUT = f"{BASE}/03_结果"
FIG = f"{BASE}/04_图表"
WORK = f"{BASE}/../02_代码/work.duckdb"
u = import_module("12_univariate")

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    ic = pd.read_csv(f"{OUT}/特征_IC.csv")

    # 1) IC bars: strongest negative + positive
    neg = ic.sort_values("ic_dev").head(18)
    pos = ic.sort_values("ic_dev", ascending=False).head(8)
    d = pd.concat([neg, pos]).drop_duplicates("feature").sort_values("ic_dev")
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(d))
    ax.barh(y, d["ic_dev"], color=np.where(d["ic_dev"] < 0, "#3b6ea5", "#c96a3a"), alpha=0.85, label="开发期 IC")
    ax.scatter(d["ic_oos"], y, color="#111", s=18, zorder=3, label="验证期 IC")
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"], fontsize=8)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("日频横截面 Spearman IC")
    ax.set_title("单特征 IC：开发期（条）与验证期（点）", fontsize=12)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    plt.savefig(f"{FIG}/01_特征IC.png")
    plt.close()

    # 2) score quintiles OOS
    q = pd.read_csv(f"{OUT}/评分_分位.csv")
    q = q[(q.period == "oos_2018_2026") & (q.group.isin(["Q1", "Q2", "Q3", "Q4", "Q5"]))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, title in [(axes[0], "mean_ex20", "相对等权全A"), (axes[1], "mean_ex_ind20", "相对行业等权")]:
        pivot = q.pivot(index="group", columns="score", values=col).loc[["Q1", "Q2", "Q3", "Q4", "Q5"]]
        x = np.arange(5); w = 0.26
        for k, s in enumerate(pivot.columns):
            ax.bar(x + (k - 1) * w, pivot[s] * 100, w, label=s)
        ax.set_xticks(x); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(f"验证期 20 日超额：{title}", fontsize=11)
        ax.set_ylabel("%")
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8, frameon=False)
    plt.savefig(f"{FIG}/02_评分分位_OOS.png")
    plt.close()

    # 3) score deciles OOS
    raw = pd.read_parquet(f"{BASE}/02_代码/信号_带评分.parquet")
    raw["date"] = pd.to_datetime(raw["date"]); raw["next_date"] = pd.to_datetime(raw["next_date"])
    ev = u.dedup_20(raw)
    dev = ev[ev.date.dt.year <= 2017]; oos = ev[ev.date.dt.year >= 2018].copy()
    qs = dev["score_A"].quantile([i / 10 for i in range(1, 10)]).values
    oos["dec"] = pd.cut(oos["score_A"], [-np.inf] + list(qs) + [np.inf], labels=[f"D{i}" for i in range(1, 11)])
    dec = oos.groupby("dec", observed=True).agg(ex20=("ex20", "mean"), r20=("r20", "mean"),
                                                n=("ex20", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(dec["dec"].astype(str), dec["ex20"] * 100, marker="o", label="20日超额（vs 等权全A）", color="#3b6ea5")
    ax.plot(dec["dec"].astype(str), dec["r20"] * 100, marker="s", label="20日绝对收益", color="#c96a3a")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("验证期评分十分位：最好的不是最高分（D9 之后回落）", fontsize=11)
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, frameon=False)
    for i, row in dec.iterrows():
        ax.text(i, row["ex20"] * 100 + 0.08, f"{row['n']}", ha="center", fontsize=6, color="#666")
    plt.savefig(f"{FIG}/03_评分十分位_OOS.png")
    plt.close()

    # 4) OOS equity curves
    con = duckdb.connect(WORK, read_only=True)
    ew = con.execute("SELECT CAST(date AS DATE) AS date, nav FROM ew_nav ORDER BY 1").fetchdf()
    con.close()
    ew["date"] = pd.to_datetime(ew["date"]); ew = ew.set_index("date")["nav"]
    ew = ew.loc["2018-01-02":]; ew = ew / ew.iloc[0]

    tags = {
        "全信号_fixed20_10": "V2全信号 + 固定20日",
        "A前30_fixed20_10": "A前30% + 固定20日",
        "A中段70_90_fixed20_10": "A中段70-90% + 固定20日",
        "A中段70_90_闸门_fixed20_10": "A中段70-90% + 闸门 + 固定20日",
    }
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(ew.index, ew.values, label="等权全A", color="#111", lw=2)
    colors = ["#9a9a9a", "#3b6ea5", "#c96a3a", "#2e8b57"]
    for (tag, label), color in zip(tags.items(), colors):
        eq = pd.read_csv(f"{OUT}/组合_选股_{tag}_权益.csv", parse_dates=["date"]).set_index("date")["equity"]
        eq = eq.loc["2018-01-02":]; eq = eq / eq.iloc[0]
        ax.plot(eq.index, eq.values, label=label, color=color, lw=1.4)
    ax.set_title("验证期（2018-2026）净值：选股改善自身，但仍跑不过等权", fontsize=11)
    ax.set_ylabel("净值（2018-01-02=1）")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, frameon=False)
    plt.savefig(f"{FIG}/04_组合权益_OOS.png")
    plt.close()

    # 5) yearly bars
    yr = pd.read_csv(f"{OUT}/组合_选股_年度.csv")
    years = [str(y) for y in range(2018, 2027)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ew_y = ew.groupby(ew.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    ax.bar(np.arange(len(years)) - 0.3, [ew_y.get(int(y), np.nan) * 100 for y in years], 0.2, label="等权全A", color="#111")
    for i, (tag, label, color) in enumerate([
        ("全信号_fixed20_10", "V2全信号+固定20", "#9a9a9a"),
        ("A中段70_90_闸门_fixed20_10", "A中段+闸门+固定20", "#2e8b57"),
    ]):
        row = yr[yr.tag == tag]
        vals = [row[y].iloc[0] * 100 if len(row) and y in row else np.nan for y in years]
        ax.bar(np.arange(len(years)) + (i - 0.1) * 0.2, vals, 0.2, label=label, color=color)
    ax.set_xticks(np.arange(len(years))); ax.set_xticklabels(years)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("逐年收益：2025 是主要拖累，2026 仍未修复", fontsize=11)
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, frameon=False)
    plt.savefig(f"{FIG}/05_组合年度.png")
    plt.close()

    # 6) liquidity
    liq = pd.read_csv(f"{OUT}/特征_流动性分层.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(liq)); w = 0.26
    for k, col, label in [(-1, "ex20_all", "全部V2"), (0, "ex20_top30", "A前30%"), (1, "ex20_band", "A中段70-90%")]:
        ax.bar(x + k * w, liq[col] * 100, w, label=label)
    ax.set_xticks(x); ax.set_xticklabels(liq["liq"])
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("验证期按20日成交额分层：低流动性本身更好，选股在各层都加分", fontsize=11)
    ax.set_ylabel("20日超额 %")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, frameon=False)
    plt.savefig(f"{FIG}/06_流动性分层.png")
    plt.close()

    # 7) family median IC
    fam = pd.read_csv(f"{OUT}/特征_族汇总.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(fam)); w = 0.36
    ax.bar(x - w / 2, fam["med_ic_dev"], w, label="开发期", color="#3b6ea5")
    ax.bar(x + w / 2, fam["med_ic_oos"], w, label="验证期", color="#c96a3a")
    ax.set_xticks(x); ax.set_xticklabels(fam["family"], fontsize=9)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("特征族中位 IC：波动/流动性最强，趋势/量价最弱", fontsize=11)
    ax.set_ylabel("中位 IC")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, frameon=False)
    plt.savefig(f"{FIG}/07_特征族.png")
    plt.close()

    print("charts saved", FIG)


if __name__ == "__main__":
    main()
