#!/usr/bin/env python3
"""26 周期特征：周线/月线的连续 MACD 特征是否也只是反转

复用 11 的周期合成口径，对周线/月线计算：
- 柱体水平、柱体变化、柱体面积、DIF 水平、DIF-DEA 距离
- 对应持有期的横截面 IC、Q5-Q1
- 按长期趋势（年线上/下/过渡，取周期末的日线状态）分桶

输出：03_结果/周期_特征.csv、周期_特征结构.csv、周期_状态.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")


def build_period(P, freq):
    dates = P["dates"]
    pid = dates.to_period("W-FRI") if freq == "W" else dates.to_period("M")
    uniq, inv = np.unique(pid, return_inverse=True)
    nper = len(uniq)
    start, end = np.full(nper, -1), np.full(nper, -1)
    for p in range(nper):
        idx = np.where(inv == p)[0]
        start[p], end[p] = idx[0], idx[-1]
    close = P["close"][end, :].copy()
    close[P["trade"][end, :] != 1] = np.nan
    open_ = P["open"][start, :].copy()
    open_[P["trade"][start, :] != 1] = np.nan
    return dict(freq=freq, pid=uniq, inv=inv, start=start, end=end,
                close=close, open=open_, nper=nper)


def sig(per, fast=12, slow=26, sg=9, min_bars=60):
    c = M._fill_lead(per["close"])
    dif = M.ema_filled(c, fast) - M.ema_filled(c, slow)
    dea = M.ema_filled(dif, sg)
    hist = 2 * (dif - dea)
    valid = np.isfinite(per["close"])
    bars = np.cumsum(valid, axis=0)
    return dif, dea, hist, valid & (bars >= min_bars)


def prets(P, per, horizons):
    nper, nc = per["close"].shape
    out, ok, mkt, loo = {}, {}, {}, {}
    for H in horizons:
        f = np.full((nper, nc), np.nan, dtype=np.float32)
        m = np.zeros((nper, nc), bool)
        for p in range(nper - H):
            e, x = per["start"][p + 1], per["end"][p + H]
            entry_ok = (P["trade"][e, :] == 1) & \
                       ~((P["raw_high"][e, :] == P["raw_low"][e, :]) &
                         (P["raw_open"][e, :] > P["preclose"][e, :]))
            exit_ok = P["trade"][x, :] == 1
            r = P["close"][x, :] / P["open"][e, :] - 1
            v = entry_ok & exit_ok & np.isfinite(r)
            f[p, v] = r[v]
            m[p] = v
        cnt = np.sum(m & np.isfinite(f), axis=1).astype(float)
        s = np.nansum(np.where(m, f, np.nan), axis=1, dtype=float)
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        lo = np.where((cnt > 1)[:, None],
                      (s[:, None] - np.where(m, f, 0)) / np.maximum(cnt[:, None] - 1, 1),
                      np.nan)
        out[H], ok[H], mkt[H], loo[H] = f, m, mean, lo
    return out, ok, mkt, loo


def ic_table(x, ex, mask, dates, nq=5, min_n=100):
    ics, spreads, q5, ns, ds = [], [], [], [], []
    for p in range(len(dates)):
        idx = np.where(mask[p] & np.isfinite(ex[p]) & np.isfinite(x[p]))[0]
        if len(idx) < min_n:
            continue
        xr, yr = rankdata(x[p, idx]), rankdata(ex[p, idx])
        q = np.minimum(((xr - 1) * nq / len(idx)).astype(np.int16), nq - 1)
        cnt = np.bincount(q, minlength=nq)
        if (cnt == 0).any():
            continue
        qm = np.bincount(q, weights=ex[p, idx], minlength=nq) / cnt
        ics.append(np.corrcoef(xr, yr)[0, 1])
        spreads.append(qm[-1] - qm[0])
        q5.append(qm[-1])
        ns.append(len(idx))
        ds.append(dates[p])
    return np.asarray(ics), np.asarray(spreads), np.asarray(q5), np.asarray(ns), pd.Index(ds)


def summ(ics, spreads, q5, ns, ds, lag):
    if len(ics) < 10:
        return None
    s, sp = pd.Series(ics, index=ds), pd.Series(spreads, index=ds)
    early = ds < pd.Timestamp("2018-01-01")
    return dict(期数=len(ics), 股票期=int(ns.sum()), 平均IC=ics.mean(),
                IC_t=nw_t(s, lag=lag), IC大于0占比=(ics > 0).mean(),
                多空5_1=spreads.mean(), 多空t=nw_t(sp, lag=lag), Q5均值=q5.mean(),
                IC_2010_2017=s[early].mean(), IC_2018_2026=s[~early].mean(),
                多空_2010_2017=sp[early].mean(), 多空_2018_2026=sp[~early].mean())


def main():
    P = M.load_panel()
    st = F.build_structure(P, M.market_features(P))
    rows, srows, state_rows = [], [], []
    for freq, label, min_bars, horizons, primary, area_w, lag in [
        ("W", "周线", 60, [4, 12], 4, 4, 12),
        ("M", "月线", 36, [3, 12], 3, 3, 6),
    ]:
        per = build_period(P, freq)
        dif, dea, hist, valid = sig(per, min_bars=min_bars)
        c = M._fill_lead(per["close"])
        hist_pct = hist / c * 100
        dif_pct = dif / c * 100
        gap = (dif - dea) / c * 100
        feats = {
            "柱体水平": hist_pct,
            "柱体变化": hist_pct - M.shift(hist_pct, 1),
            f"{area_w}期柱体均值": M.rolling_mean(hist_pct, area_w),
            "DIF水平": dif_pct,
            "DIF-DEA距离": gap,
        }
        states = {
            "红柱放大": (hist > 0) & (hist > M.shift(hist, 1)),
            "DIF>DEA": dif > dea,
        }
        rets, ok, mkt, loo = prets(P, per, horizons)
        for H in horizons:
            ex = rets[H] - loo[H]
            mask = valid & ok[H] & np.isfinite(ex)
            for name, x in feats.items():
                ics, spreads, q5, ns, ds = ic_table(x, ex, mask, P["dates"][per["end"]])
                r = summ(ics, spreads, q5, ns, ds, lag)
                if r:
                    r.update(周期=label, 持有期=H, 特征=name)
                    rows.append(r)
            # 结构：长期趋势
            trend = st["trend"][per["end"], :]
            for k, lab in enumerate(["上行", "下行", "过渡"]):
                mk = mask & (trend == k)
                for name, x in feats.items():
                    ics, spreads, q5, ns, ds = ic_table(x, ex, mk, P["dates"][per["end"]])
                    r = summ(ics, spreads, q5, ns, ds, lag)
                    if r:
                        r.update(周期=label, 持有期=H, 特征=name, 趋势=lab)
                        srows.append(r)
            # 状态
            for name, state in states.items():
                for k, lab in enumerate(["上行", "下行", "过渡"]):
                    mk = mask & (trend == k)
                    # 控制水平三分位后的状态差
                    lvl = hist_pct if name == "红柱放大" else dif_pct
                    raw, adj = [], []
                    for p in range(per["nper"]):
                        idx = np.where(mk[p] & np.isfinite(ex[p]) & np.isfinite(lvl[p]))[0]
                        if len(idx) < 100:
                            continue
                        s = state[p, idx]
                        if s.sum() < 15 or (~s).sum() < 15:
                            continue
                        y = ex[p, idx]
                        raw.append(y[s].mean() - y[~s].mean())
                        lq = np.minimum(((rankdata(lvl[p, idx]) - 1) * 3 / len(idx)).astype(np.int16), 2)
                        d, w = [], []
                        for a in range(3):
                            cell = lq == a
                            c1, c0 = int((cell & s).sum()), int((cell & ~s).sum())
                            if c1 >= 8 and c0 >= 8:
                                d.append(y[cell & s].mean() - y[cell & ~s].mean()); w.append(c1 + c0)
                        if d:
                            adj.append(np.average(d, weights=w))
                    if len(raw) < 10:
                        continue
                    rr = pd.Series(raw); aa = pd.Series(adj)
                    state_rows.append(dict(周期=label, 持有期=H, 特征=name, 趋势=lab,
                                      期数=len(rr), 原始差=rr.mean(), 原始t=nw_t(rr, lag=lag),
                                      控制后差=aa.mean() if len(aa) else np.nan,
                                      控制t=nw_t(aa, lag=lag) if len(aa) else np.nan))
        print("完成", label)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "周期_特征.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(srows).to_csv(os.path.join(RES, "周期_特征结构.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(state_rows).to_csv(os.path.join(RES, "周期_状态.csv"), index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
