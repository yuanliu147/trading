#!/usr/bin/env python3
"""49b KDJ 条件快速扫描：在 MACD 多/空内部，哪些 KDJ 状态改变 5/20/60 日结果。

仅用于设计正式条件表；输出 03_结果/探索_KDJ条件扫描.csv。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
import kdj_core as K
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
MIN_CELL = 3000


def daily_sel(sel, ex, min_n=50):
    n = sel.sum(axis=1)
    s = np.nansum(np.where(sel, ex, 0.0), axis=1)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), n


def stat(sel, ex, dates):
    n = int(sel.sum())
    dm, _ = daily_sel(sel, ex)
    ok = np.isfinite(dm)
    if n < MIN_CELL or ok.sum() < 50:
        return dict(n=n, days=int(ok.sum()), mean=np.nan, t=np.nan,
                    early=np.nan, late=np.nan)
    early = ok & (dates <= pd.Timestamp("2017-12-31"))
    late = ok & (dates >= pd.Timestamp("2018-01-01"))
    return dict(n=n, days=int(ok.sum()), mean=float(np.nanmean(dm)),
                t=float(nw_t(dm, 20)),
                early=float(np.nanmean(dm[early])) if early.any() else np.nan,
                late=float(np.nanmean(dm[late])) if late.any() else np.nan)


def main():
    t0 = time.time()
    P = M.load_panel()
    dates = P["dates"]
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    kd = K.kdj_features(P)
    print("kdj", time.time() - t0)
    fwd, ok, mkt, loo = M.forward_returns(P, (5, 20, 60))
    exs = {h: (fwd[h] - loo[h]).astype(np.float32) for h in (5, 20, 60)}
    masks = {h: base & ok[h] & np.isfinite(exs[h]) for h in (5, 20, 60)}

    k, d, j = kd["k"], kd["d"], kd["j"]
    gap = kd["gap"]
    age = kd["age"]
    c = P["close_ff"]

    def recent(ev, n):
        cs = np.cumsum(ev, axis=0)
        return (cs - M.shift(cs, n, fill=0)) > 0

    conds = []
    def add(fam, lab, m):
        conds.append((fam, lab, m))

    add("KDJ方向", "K>D(多头)", kd["states"]["K>D"])
    add("KDJ方向", "K<D(空头)", ~kd["states"]["K>D"])
    add("KDJ水平", "K>80(超买)", k > 80)
    add("KDJ水平", "K<20(超卖)", k < 20)
    add("KDJ水平", "K>70(高位)", k > 70)
    add("KDJ水平", "K<30(低位)", k < 30)
    add("KDJ水平", "K>50", k > 50)
    add("KDJ水平", "K<=50", k <= 50)
    add("KDJ水平", "D>80", d > 80)
    add("KDJ水平", "D<20", d < 20)
    add("J极值", "J>100", j > 100)
    add("J极值", "J<0", j < 0)
    add("J极值", "J>120", j > 120)
    add("J极值", "J<-20", j < -20)
    for nm, x in [("K水平", k), ("D水平", d), ("J水平", j), ("K-D差", gap),
                  ("20日交叉次数", kd["cross20"])]:
        q = F.row_quantile(x, base, 5)
        for i in range(5):
            add(nm, f"Q{i+1}", q == i)
    ag = np.abs(age)
    for lo, hi in [(0, 5), (6, 15), (16, 40), (41, 120), (121, 9999)]:
        add("交叉年龄", f"{lo}-{hi}日", (ag >= lo) & (ag <= hi))
    for n in (5, 10, 20):
        add("近期金叉", f"{n}日内金叉", recent(kd["gold"], n))
        add("近期死叉", f"{n}日内死叉", recent(kd["death"], n))
    # 背离
    high20, low20 = M.rolling_max(c, 20), M.rolling_min(c, 20)
    kmax, kmin = M.rolling_max(k, 20), M.rolling_min(k, 20)
    top_div = (c >= high20 - 1e-6) & (k < kmax - 1e-9)
    bot_div = (c <= low20 + 1e-6) & (k > kmin + 1e-9)
    add("KDJ背离", "顶背离", top_div)
    add("KDJ背离", "底背离", bot_div)
    add("KDJ背离", "无背离", ~top_div & ~bot_div)
    # 周/月 KDJ
    wkd = K.period_kdj_state(P, "W")
    mkd = K.period_kdj_state(P, "M")
    add("周线KDJ", "周线多头", wkd["state"])
    add("周线KDJ", "周线空头", ~wkd["state"] & wkd["ok"])
    add("月线KDJ", "月线多头", mkd["state"])
    add("月线KDJ", "月线空头", ~mkd["state"] & mkd["ok"])
    print("conds", len(conds), time.time() - t0)

    rows = []
    for h in (5, 20, 60):
        ex = exs[h]
        mask = masks[h]
        mb = mask & macd["states"]["DIF>DEA"]
        ms = mask & ~macd["states"]["DIF>DEA"]
        for fam, lab, cm in conds:
            cm = cm & mask
            for side, sel in [("MACD多", mb), ("MACD空", ms)]:
                s = sel & cm
                if s.sum() < MIN_CELL:
                    continue
                r = stat(s, ex, dates)
                rows.append(dict(周期=h, 条件族=fam, 条件=lab, 方向=side, **r))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "探索_KDJ条件扫描.csv"), index=False, encoding="utf-8-sig")
    for h in (5, 20, 60):
        for side in ("MACD多", "MACD空"):
            x = out[(out.周期 == h) & (out.方向 == side)].sort_values("mean")
            print(f"\n===== {h}日 {side} 最差5 =====")
            print(x.head(5)[["条件族", "条件", "n", "mean", "t", "early", "late"]].to_string(index=False))
            print(f"===== {h}日 {side} 最好5 =====")
            print(x.tail(5)[["条件族", "条件", "n", "mean", "t", "early", "late"]].to_string(index=False))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
