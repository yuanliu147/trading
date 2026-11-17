#!/usr/bin/env python3
"""66 周线 MACD 柱体回踩 0 轴后再放大：未来 20/60/120 日超额。"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import kdj_core as K
import stock_core as S
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
RATIOS = (0.0, 0.2, 0.3, 0.5)


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
    per = K.period_ohlc(P, "W")
    wc = M._fill_lead(per["close"])
    dif = M.ema_filled(wc, 12) - M.ema_filled(wc, 26)
    dea = M.ema_filled(dif, 9)
    h = (2.0 * (dif - dea)) / wc * 100.0
    hp = M.shift(h, 1)
    peak20 = M.rolling_max(h, 20, minp=10)
    trough5 = M.rolling_min(h, 5, minp=3)
    red = h > 0; expand = red & (h > hp)
    wvalid = np.isfinite(per["close"])
    wbars = np.cumsum(wvalid, axis=0)
    wmask = wvalid & (wbars >= 26)
    fwd, okf, mkt, loo = M.forward_returns(P, (20, 60, 120))
    rows = []
    for ratio in RATIOS:
        if ratio == 0.0:
            pull = trough5 <= 0; name = "周线回踩破0再翻红"
        else:
            pull = (trough5 > 0) & (trough5 <= ratio * peak20)
            name = f"周线红柱回踩≤{ratio:.0%}"
        wpat = wmask & pull & expand
        # 映射到每周最后一个交易日（该周收盘确认）
        sig = np.zeros(P["shape"], dtype=bool)
        for pp in range(per["nper"] - 1):
            sig[per["end"][pp], :] = wpat[pp, :]
        sig &= base & P["valid"]
        for hz in (20, 60, 120):
            ex = (fwd[hz] - loo[hz]).astype(np.float32)
            m = base & okf[hz] & np.isfinite(ex)
            r = stat(sig & m, ex, dates)
            r.update(形态=name, 周期=hz)
            rows.append(r)
        print(name, "n", int(sig.sum()), "%.1fs" % (time.time() - t0))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "66_零轴回踩_周线.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
