#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取严格重跑结果，画净值与年度收益对比图。"""
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "02_结果"
FIG = OUT / "图表"
FIG.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("/Users/admin/WorkSpace/trading/2026-10-18_成交量融合_2010-2025/01_数据")
START = pd.Timestamp("2018-07-01")
END = pd.Timestamp("2026-10-13")

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

names = {
    "原20池_b3_同日高触发": "原20池 · b3（同日高触发，原口径）",
    "原20池_b3_前日峰值修复": "原20池 · b3（前日峰值修复）",
    "时点固定20_b3_前日峰值": "固定时点20 · b3（修复）",
    "前20成交额_b3_前日峰值": "时点前20成交额 · b3（修复）",
    "时点500_b3_前日峰值": "时点500全池 · b3（修复）",
}
colors = {
    "原20池_b3_同日高触发": "#9e9e9e",
    "原20池_b3_前日峰值修复": "#d32f2f",
    "时点固定20_b3_前日峰值": "#2e7d32",
    "前20成交额_b3_前日峰值": "#f57c00",
    "时点500_b3_前日峰值": "#1976d2",
}

fig, ax = plt.subplots(figsize=(11, 6.2))
for key, label in names.items():
    f = OUT / f"权益_{key}.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f, parse_dates=["日期"]).sort_values("日期")
    s = df.set_index("日期")["权益"] / df["权益"].iloc[0]
    ax.plot(s.index, s.values, label=label, color=colors[key], linewidth=1.6)

# 中证500
with open(DATA_DIR / "daily_cache_2010_2025_hfq.pkl", "rb") as fh:
    a = pickle.load(fh)
with open(DATA_DIR / "daily_cache_2016_2026.pkl", "rb") as fh:
    b = pickle.load(fh)
z = pd.concat([
    a["zz500"][pd.to_datetime(a["zz500"]["date"]) <= pd.Timestamp("2025-12-31")],
    b["zz500"][pd.to_datetime(b["zz500"]["date"]) > pd.Timestamp("2025-12-31")],
])
z["date"] = pd.to_datetime(z["date"])
z = z[(z["date"] >= START) & (z["date"] <= END)].sort_values("date")
s = z.set_index("date")["close"] / z["close"].iloc[0]
ax.plot(s.index, s.values, label="中证500", color="#212121", linewidth=1.4, linestyle="--")

ax.set_yscale("log")
ax.set_title("MACD 全金叉：原20池、固定时点20、时点前20、时点500（净值，对数轴）")
ax.set_ylabel("净值（2018-07-02 = 1）")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "净值对比.png")
plt.close(fig)

# 年度收益：只画完整年份 2019-2025
y = pd.read_csv(OUT / "年度_严格重跑.csv")
y = y[y["年份"].between(2019, 2025)]
pivot = y.pivot(index="年份", columns="策略", values="策略收益")
bench = y.groupby("年份")["中证500"].first()
cols = [c for c in names if c in pivot.columns]
if cols:
    x = np.arange(len(pivot.index))
    width = 0.8 / (len(cols) + 1)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(x - 0.4 + width / 2, bench.reindex(pivot.index).values, width,
           label="中证500", color="#212121", alpha=0.75)
    for i, c in enumerate(cols):
        ax.bar(x - 0.4 + width * (i + 1.5), pivot[c].values, width,
               label=names[c], color=colors[c])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(int))
    ax.set_ylabel("年度收益（%）")
    ax.set_title("年度收益对比（2019–2025 完整年份）")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "年度收益对比.png")
    plt.close(fig)

print("图表已保存：", FIG)
