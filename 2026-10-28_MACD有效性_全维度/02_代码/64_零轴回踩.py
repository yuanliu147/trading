#!/usr/bin/env python3
"""64 MACD 柱体回踩 0 轴后再放大：有没有增量？

口径
----
- 日线 MACD(12,26,9)，柱体按收盘价归一（hist/close*100）。
- 回踩：近 5 日柱体低点 <= 峰值20 的 ratio；分“红柱回踩不破 0”与“回踩破 0 再翻红”。
- 放大：当日柱体 > 前一日，且 >0。
- 信号 T 日收盘，T+1 开盘；5/20/60 日超额 = 个股收益 - 同日全市场等权（leave-one-out）。
- 条件增量：MACD 多头内部的“有形态 - 无形态”。
"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
import stock_core as S
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
RATIOS = (0.0, 0.1, 0.2, 0.3, 0.5)


def daily_sel(sel, ex, min_n=20):
    n = sel.sum(axis=1)
    s = np.nansum(np.where(sel, ex, 0.0), axis=1)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), n


def stat(sel, ex, dates, min_n=20):
    dm, n = daily_sel(sel, ex, min_n)
    ok = np.isfinite(dm)
    if ok.sum() < 20:
        return dict(n=int(sel.sum()), days=int(ok.sum()), mean=np.nan, t=np.nan,
                    early=np.nan, late=np.nan, pos_years=np.nan, years=np.nan)
    y = pd.Series(dm[ok], index=dates[ok]).groupby(dates[ok].year).mean()
    return dict(n=int(sel.sum()), days=int(ok.sum()),
                mean=float(np.nanmean(dm[ok])), t=float(nw_t(dm[ok], 20)),
                early=float(np.nanmean(dm[ok & (dates <= pd.Timestamp("2017-12-31"))])),
                late=float(np.nanmean(dm[ok & (dates >= pd.Timestamp("2018-01-01"))])),
                pos_years=int((y > 0).sum()), years=int(y.notna().sum()))


def inc(sel_a, sel_b, ex, dates, min_n=20):
    """a-b 的日度差序列。"""
    a, _ = daily_sel(sel_a, ex, min_n)
    b, _ = daily_sel(sel_b, ex, min_n)
    d = a - b
    ok = np.isfinite(d)
    if ok.sum() < 20:
        return np.nan, np.nan
    return float(np.nanmean(d[ok])), float(nw_t(d[ok], 20))


def main():
    t0 = time.time()
    P = S.load_base(); base = P["universe_inv"]; dates = P["dates"]
    mf = F.build_macd_features(P)
    dif, dea, hist = mf["dif"], mf["dea"], mf["hist"]
    c = P["close_ff"]
    h = (hist / c * 100).astype(np.float32)
    hp = M.shift(h, 1)
    peak20 = M.rolling_max(h, 20, minp=10)
    trough5 = M.rolling_min(h, 5, minp=3)
    red = h > 0
    expand = red & (h > hp)
    bull = (dif > dea) & P["valid"] & (P["bars"] >= 120)
    mask = base & P["valid"] & (P["bars"] >= 120)
    fwd, okf, mkt, loo = M.forward_returns(P, (5, 20, 60))
    print("loaded", P["shape"], "%.1fs" % (time.time() - t0))

    rows = []
    for ratio in RATIOS:
        if ratio == 0.0:
            pull = (trough5 <= 0)  # 破 0 再翻红
            name = "回踩破0再翻红"
        else:
            pull = (trough5 > 0) & (trough5 <= ratio * peak20)
            name = f"红柱回踩≤{ratio:.0%}峰值"
        pat = mask & pull & expand
        for hz in (5, 20, 60):
            ex = (fwd[hz] - loo[hz]).astype(np.float32)
            m = mask & okf[hz] & np.isfinite(ex)
            r = stat(pat & m, ex, dates)
            # MACD 多头内部增量
            inside = bull & m
            inc_m, inc_t = inc(pat & inside, inside & ~pat, ex, dates)
            r.update(形态=name, 周期=hz, MACD多内增量=inc_m, 增量t=inc_t)
            rows.append(r)
        print(name, "n", int(pat.sum()), "%.1fs" % (time.time() - t0))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "64_零轴回踩_日线.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
