#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""443 笔真实系统交易：连亏暂停、仓位数学、止损后反弹、卖错处理。"""
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
t["combo"] = t[["m1", "m2", "m3"]].astype(int).astype(str).agg("".join, axis=1)
t["loss"] = (t.ret <= 0).astype(int)

print("trades", len(t), "win", (t.ret > 0).mean().round(4), "mean", t.ret.mean().round(4))
print("\n方法组合")
print(t.groupby("combo").ret.agg(["count", "mean", "median", lambda x: (x > 0).mean()]).round(4).to_string())

# ---------- 连亏后的下一笔 ----------
t["prev1_loss"] = t.loss.shift(1)
t["prev2_loss"] = (t.loss.shift(1) + t.loss.shift(2)).eq(2)
t["prev3_loss"] = (t.loss.shift(1) + t.loss.shift(2) + t.loss.shift(3)).eq(3)
for col in ("prev1_loss", "prev2_loss", "prev3_loss"):
    g = t.loc[t[col].fillna(False).astype(bool)]
    print(f"\n{col}: n={len(g)}, next win={(g.ret>0).mean():.3f}, mean={g.ret.mean():.4f}, median={g.ret.median():.4f}")

# 按年+市场状态看连亏后的下一笔
t["year"] = t.signal_date.dt.year
rows = []
for y, g in t.groupby("year"):
    a = g.loc[g.prev2_loss.fillna(False).astype(bool)]
    if len(a):
        rows.append({"year": y, "n_after2loss": len(a), "win": (a.ret > 0).mean(), "mean": a.ret.mean(),
                     "all_win": (g.ret > 0).mean(), "all_mean": g.ret.mean()})
print("\n分年连亏两次后")
print(pd.DataFrame(rows).round(4).to_string(index=False))

# ---------- 固定20%仓位的简单权益与暂停规则 ----------
def equity_curve(df, weight=0.20):
    eq = [1.0]
    for r in df.ret:
        eq.append(eq[-1] * (1 + weight * r))
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    return eq[-1], dd.min()

def pause_filter(df, n_losses=2, pause_trades=2):
    """连亏 n_losses 后，跳过接下来 pause_trades 笔（按卖出顺序）。"""
    keep = []
    streak = 0; skip = 0
    for i, r in df.iterrows():
        if skip > 0:
            skip -= 1
            continue
        keep.append(i)
        if r.ret <= 0:
            streak += 1
            if streak >= n_losses:
                skip = pause_trades; streak = 0
        else:
            streak = 0
    return df.loc[keep]

base_eq, base_dd = equity_curve(t)
print(f"\n基准（每笔20%）：终值 {base_eq:.4f}, 最大回撤 {base_dd:.4f}")
for nl in (1, 2, 3):
    for pt in (1, 2, 3):
        f = pause_filter(t, nl, pt)
        eq, dd = equity_curve(f)
        print(f"连亏{nl}次后跳过{pt}笔：交易{len(f)}, 终值{eq:.4f}, 回撤{dd:.4f}, 胜率{(f.ret>0).mean():.3f}")

# ---------- 仓位数学 ----------
print("\n仓位数学：单笔风险=3%时，不同止损距离对应的仓位")
for stop in (0.03, 0.05, 0.08, 0.10, 0.15):
    print(f"止损{stop:.0%}: 仓位 {0.03/stop:.1%}, 若滑点/跳空到1.5倍止损，账户亏 {0.03*1.5:.1%}")
print("\n单笔风险=1%时")
for stop in (0.03, 0.05, 0.08, 0.10, 0.15):
    print(f"止损{stop:.0%}: 仓位 {0.01/stop:.1%}")

# ---------- 止损后反弹 ----------
print("\n止损后反弹分析")
stop_t = t[t.reason == "止损"].copy()
rows = []
for _, r in stop_t.iterrows():
    g = stocks.get(r.code)
    if g is None:
        continue
    g = g.sort_values("date").reset_index(drop=True)
    d = pd.Timestamp(r.sell)
    idx = g.index[g.date == d]
    if len(idx) == 0:
        continue
    i = idx[0]
    row = {"code": r.code, "signal_date": r.signal_date, "sell": d, "ret": r.ret, "sell_px": r.sell_px}
    for h in (10, 20, 40):
        if i + h < len(g):
            row[f"max{h}"] = g.high.iloc[i:i + h + 1].max() / r.sell_px - 1
            row[f"close{h}"] = g.close.iloc[i + h] / r.sell_px - 1
            row[f"above_buy{h}"] = g.close.iloc[i + h] >= r.buy_px
    rows.append(row)
reb = pd.DataFrame(rows)
print("止损笔数", len(reb))
for h in (10, 20, 40):
    if f"max{h}" in reb:
        print(f"{h}日内最高反弹: 均值{reb[f'max{h}'].mean():.2%}, 中位{reb[f'max{h}'].median():.2%}, >5%占比{(reb[f'max{h}']>0.05).mean():.1%}, 回到买价上方占比{reb[f'above_buy{h}'].mean():.1%}")
print(reb[[c for c in reb.columns if c.startswith(('max','close','above'))]].describe().round(4).to_string())
reb.to_csv(os.path.join(BASE, "止损后反弹.csv"), index=False, encoding="utf-8-sig")

# ---------- 卖错：止损后若继续持有 20 日 ----------
if "close20" in reb:
    reb["hold20_ret"] = reb.close20
    print("\n止损后继续持有20日（相对卖价）均值", reb.hold20_ret.mean().round(4),
          "中位", reb.hold20_ret.median().round(4),
          "为正比例", (reb.hold20_ret > 0).mean().round(4))
    print("原止损平均", reb.ret.mean().round(4))

# ---------- 首周跟随与最终收益（复核） ----------
rows = []
for _, r in t.iterrows():
    g = stocks.get(r.code)
    if g is None:
        continue
    g = g.sort_values("date").reset_index(drop=True)
    idx = g.index[g.date == pd.Timestamp(r.buy)]
    if len(idx) == 0:
        continue
    i = idx[0]
    if i + 5 < len(g):
        rows.append({"first5": g.close.iloc[i + 5] / r.buy_px - 1, "ret": r.ret, "year": r.signal_date.year})
f5 = pd.DataFrame(rows)
print("\n首周跟随分组")
f5["grp"] = pd.cut(f5.first5, [-1, -0.05, -0.02, 0, 0.02, 0.05, 1], labels=["<-5%", "-5~-2%", "-2~0", "0~2%", "2~5%", ">5%"])
print(f5.groupby("grp", observed=True).agg(n=("ret", "size"), win=("ret", lambda x: (x > 0).mean()), mean=("ret", "mean"), median=("ret", "median")).round(4).to_string())
print("corr", f5[["first5", "ret"]].corr().iloc[0, 1].round(4))
f5.to_csv(os.path.join(BASE, "首周跟随.csv"), index=False, encoding="utf-8-sig")
print("done")
