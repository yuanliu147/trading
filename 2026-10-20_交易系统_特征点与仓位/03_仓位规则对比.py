#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 443 笔真实交易比较：固定20%仓位 vs 按止损距离的风险预算仓位。"""
import os, pickle, math, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
CACHE = os.path.join(ROOT, "2026-10-18_成交量融合_2010-2025", "01_数据", "daily_cache_2016_2026.pkl")
TRADES = os.path.join(ROOT, "2026-10-18_成交量融合_2010-2025", "08_深度复盘_2016_2026", "逐笔_2016-2026_低波活跃_结合.csv")
with open(CACHE, "rb") as f:
    D = pickle.load(f)
stocks = D["stocks"]
t = pd.read_csv(TRADES, parse_dates=["signal_date", "buy", "sell"]).sort_values(["sell", "signal_date"]).reset_index(drop=True)

# 还原每笔的初始止损：max(信号日 low10, 买价×0.92)
rows = []
for _, r in t.iterrows():
    g = stocks.get(r.code)
    if g is None:
        continue
    g = g.sort_values("date").reset_index(drop=True)
    idx = g.index[g.date == pd.Timestamp(r.signal_date)]
    if len(idx) == 0:
        continue
    i = idx[0]
    low10 = g.low.iloc[max(0, i - 9):i + 1].min()
    stop = max(low10, r.buy_px * 0.92)
    if stop >= r.buy_px:
        stop = r.buy_px * 0.92
    rows.append({"idx": r.name, "stop_pct": (r.buy_px - stop) / r.buy_px,
                 "low10_pct": (r.buy_px - low10) / r.buy_px})
x = pd.DataFrame(rows).set_index("idx")
t = t.join(x)
print(t[["stop_pct", "low10_pct"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(4).to_string())

def curve(df, weights):
    eq = [1.0]
    for w, r in zip(weights, df.ret):
        eq.append(eq[-1] * (1 + w * r))
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1).min()
    return eq[-1], dd, np.mean(weights)

def heat(df, weights):
    """按开仓区间计算最大同时风险预算（权重×止损距离）。"""
    events = []
    for (_, r), w in zip(df.iterrows(), weights):
        events.append((r.buy, w * r.stop_pct))
        events.append((r.sell, -w * r.stop_pct))
    events.sort()
    cur = 0; mx = 0
    for _, v in events:
        cur += v; mx = max(mx, cur)
    return mx

print("\n--- 固定20% ---")
w = np.full(len(t), 0.20)
eq, dd, avg = curve(t, w)
print(f"终值 {eq:.4f}, 最大回撤 {dd:.4f}, 平均仓位 {avg:.1%}, 平均单笔风险 {np.mean(w*t.stop_pct):.2%}, 最大同时风险 {heat(t,w):.2%}")

print("\n--- 风险预算（单笔风险 r，单票上限20%）---")
for risk in (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03):
    w = np.minimum(0.20, risk / t.stop_pct.values)
    eq, dd, avg = curve(t, w)
    print(f"r={risk:.2%}: 终值 {eq:.4f}, 回撤 {dd:.4f}, 平均仓位 {avg:.1%}, 平均单笔风险 {np.mean(w*t.stop_pct):.2%}, 最大同时风险 {heat(t,w):.2%}, 风险调整(终值/|DD|) {eq/abs(dd):.2f}")

print("\n--- 风险预算 + 平台宽/高波半仓 ---")
# 与既有研究一致：平台宽>0.18 且 20日波动>0.50 时半仓。这里用 contract 与 s_vol 近似。
# contract 在逐笔表里是平台宽度代理；s_vol 是波动分。若无列则跳过。
if "contract" in t and "s_vol" in t:
    half = (t.contract > 0.18) & (t.s_vol > 0.50)
    for risk in (0.0075, 0.01, 0.015):
        w = np.minimum(0.20, risk / t.stop_pct.values)
        w = np.where(half, w * 0.5, w)
        eq, dd, avg = curve(t, w)
        print(f"r={risk:.2%}+半仓: 终值 {eq:.4f}, 回撤 {dd:.4f}, 平均仓位 {avg:.1%}, 平均单笔风险 {np.mean(w*t.stop_pct):.2%}, 最大同时风险 {heat(t,w):.2%}")
    print("半仓触发", half.sum(), "笔")

print("\n--- 固定20% + 平台宽/高波半仓 ---")
if "contract" in t and "s_vol" in t:
    half = (t.contract > 0.18) & (t.s_vol > 0.50)
    w = np.where(half, 0.10, 0.20)
    eq, dd, avg = curve(t, w)
    print(f"终值 {eq:.4f}, 回撤 {dd:.4f}, 平均仓位 {avg:.1%}, 平均单笔风险 {np.mean(w*t.stop_pct):.2%}, 最大同时风险 {heat(t,w):.2%}")

t.to_csv(os.path.join(BASE, "交易_止损距离.csv"), index=False, encoding="utf-8-sig")
print("done")
