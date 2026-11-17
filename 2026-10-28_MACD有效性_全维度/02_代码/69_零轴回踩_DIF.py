#!/usr/bin/env python3
"""69 另一种“回踩0轴”：DIF/DEA 回到 0 轴附近后再放大。"""
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
QS = (0.2, 0.5, 1.0, 2.0)


def daily_sel(sel, ex, min_n=5):
    n = sel.sum(axis=1)
    s = np.nansum(np.where(sel, ex, 0.0), axis=1)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan)


def stat(sel, ex, dates, min_n=5):
    dm = daily_sel(sel, ex, min_n); ok = np.isfinite(dm)
    if ok.sum() < 20:
        return dict(n=int(sel.sum()), days=int(ok.sum()), mean=np.nan, t=np.nan,
                    early=np.nan, late=np.nan, pos_years=np.nan, years=np.nan)
    y = pd.Series(dm[ok], index=dates[ok]).groupby(dates[ok].year).mean()
    return dict(n=int(sel.sum()), days=int(ok.sum()),
                mean=float(np.nanmean(dm[ok])), t=float(nw_t(dm[ok], 20)),
                early=float(np.nanmean(dm[ok & (dates <= pd.Timestamp("2017-12-31"))])),
                late=float(np.nanmean(dm[ok & (dates >= pd.Timestamp("2018-01-01"))])),
                pos_years=int((y > 0).sum()), years=int(y.notna().sum()))


def main():
    t0 = time.time()
    P = S.load_base(); base = P["universe_inv"]; dates = P["dates"]
    mf = F.build_macd_features(P)
    dif, dea, hist = mf["dif"], mf["dea"], mf["hist"]
    c = P["close_ff"]
    dp = (dif / c * 100).astype(np.float32)
    ep = (dea / c * 100).astype(np.float32)
    h = (hist / c * 100).astype(np.float32)
    hp = M.shift(h, 1)
    expand = (h > hp) & (h > 0)
    bull = dif > dea
    prevb = M.shift(dif, 1) > M.shift(dea, 1)
    mask = base & P["valid"] & (P["bars"] >= 120)
    fwd, okf, mkt, loo = M.forward_returns(P, (5, 20, 60))
    rows = []
    for q in QS:
        near = (np.abs(dp) <= q) & (np.abs(ep) <= q)
        pat = mask & near & bull & expand
        cross = mask & near & bull & ~prevb
        for lab, s in [("零轴附近+柱体放大", pat), ("零轴附近金叉", cross)]:
            for hz in (5, 20, 60):
                ex = (fwd[hz] - loo[hz]).astype(np.float32)
                m = mask & okf[hz] & np.isfinite(ex)
                r = stat(s & m, ex, dates)
                r.update(形态=lab, 阈值=q, 周期=hz)
                rows.append(r)
        print(q, "pat", int(pat.sum()), "cross", int(cross.sum()), "%.1fs" % (time.time() - t0))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "69_零轴回踩_DIF.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
