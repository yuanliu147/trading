#!/usr/bin/env python3
"""11 周期研究：周线 / 月线 MACD 金叉

- 周线：按 W-FRI 合成，周五收盘确认，下周一开盘买入；持有 4/8/12/26 周。
- 月线：按自然月合成，月末收盘确认，下月首个交易日开盘买入；持有 3/6/12 月。
- 主指标：对应持有期的横截面分位；辅以超额、聚类 t、日度加权、分段。
- 参数网格：413 组，主持有期（周 4 周 / 月 3 月）。
- 输出：03_结果/周期_*.csv
"""
import os
import sys
import gc
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")

FASTS = [5, 6, 8, 10, 12, 15, 20, 25]
SLOWS = [15, 20, 26, 30, 35, 40, 50, 60]
SIGS = [3, 5, 9, 12, 15, 20]


def build_period(P, freq):
    dates = P["dates"]
    pid = dates.to_period("W-FRI") if freq == "W" else dates.to_period("M")
    uniq, inv = np.unique(pid, return_inverse=True)
    nper = len(uniq)
    start = np.full(nper, -1)
    end = np.full(nper, -1)
    for p in range(nper):
        idx = np.where(inv == p)[0]
        start[p], end[p] = idx[0], idx[-1]
    close = P["close"][end, :].copy()
    close[P["trade"][end, :] != 1] = np.nan
    open_ = P["open"][start, :].copy()
    open_[P["trade"][start, :] != 1] = np.nan
    return dict(freq=freq, pid=uniq, inv=inv, start=start, end=end,
                close=close, open=open_, nper=nper)


def period_signals(P, per, fast=12, slow=26, sig=9, min_bars=60):
    c = per["close"]
    c_ff = M._fill_lead(c)
    dif = M.ema_filled(c_ff, fast) - M.ema_filled(c_ff, slow)
    dea = M.ema_filled(dif, sig)
    valid = ~np.isnan(c)
    bars = np.cumsum(valid, axis=0)
    prev_d, prev_e = M.shift(dif, 1), M.shift(dea, 1)
    base = valid & (bars >= min_bars) & np.isfinite(dif) & np.isfinite(prev_d)
    gold = base & (dif > dea) & (prev_d <= prev_e)
    death = base & (dif < dea) & (prev_d >= prev_e)
    return dif, dea, 2 * (dif - dea), gold, death, bars


def period_returns(P, per, horizons):
    """信号在第 p 期末；第 p+1 期首开盘买入；第 p+H 期末收盘卖出。"""
    nper, nc = per["close"].shape
    n = P["shape"][0]
    out, ok, pct, mkt, loo = {}, {}, {}, {}, {}
    for H in horizons:
        f = np.full((nper, nc), np.nan, dtype=np.float32)
        m = np.zeros((nper, nc), bool)
        for p in range(nper - H):
            e = per["start"][p + 1]
            x = per["end"][p + H]
            entry_ok = (P["trade"][e, :] == 1) & \
                       ~((P["raw_high"][e, :] == P["raw_low"][e, :]) &
                         (P["raw_open"][e, :] > P["preclose"][e, :]))
            exit_ok = P["trade"][x, :] == 1
            r = P["close"][x, :] / P["open"][e, :] - 1
            v = entry_ok & exit_ok & np.isfinite(r)
            f[p, v] = r[v]
            m[p] = v
        # market mean / leave-one-out
        cnt = np.sum(m & np.isfinite(f), axis=1).astype(float)
        s = np.nansum(np.where(m, f, np.nan), axis=1, dtype=float)
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        lo = np.where((cnt > 1)[:, None],
                      (s[:, None] - np.where(m, f, 0)) / np.maximum(cnt[:, None] - 1, 1),
                      np.nan)
        # cross-sectional percentile
        pc = np.full((nper, nc), np.nan, dtype=np.float32)
        for p in range(nper):
            idx = np.where(m[p] & np.isfinite(f[p]))[0]
            if len(idx) < 50:
                continue
            x = f[p, idx]
            order = np.argsort(np.argsort(x))
            pc[p, idx] = (order + 0.5) / len(idx)
        out[H], ok[H], mkt[H], loo[H], pct[H] = f, m, mean, lo, pc
    return out, ok, mkt, loo, pct


def cluster_t(e, d):
    x = pd.DataFrame({"e": e, "d": d}).dropna()
    if len(x) < 50:
        return np.nan
    s = x.groupby("d").e.sum().to_numpy()
    return x.e.mean() / (np.sqrt(np.sum(s ** 2)) / len(x))


def event_table(P, per, dif, dea, hist, gold, death, rets, ok, mkt, loo, pct,
                feats, mf, wf, H, label):
    ii, jj = np.where(gold)
    # signal date = period end
    sd = P["dates"][per["end"][ii]]
    keep = sd >= pd.Timestamp("2010-01-01")
    ii, jj = ii[keep], jj[keep]
    sd = sd[keep]
    e = pd.DataFrame({
        "date": sd, "code": P["codes"][jj],
        "close": P["close"][per["end"][ii], jj],
        "dif": dif[ii, jj], "dea": dea[ii, jj], "hist": hist[ii, jj],
        "dif_pct": dif[ii, jj] / P["close"][per["end"][ii], jj] * 100,
        "ret20": feats["ret20"][per["end"][ii], jj],
        "ret60": feats["ret60"][per["end"][ii], jj],
        "std20": feats["std20"][per["end"][ii], jj],
        "dist_high": feats["dist_high"][per["end"][ii], jj],
        "above_ma20": P["close"][per["end"][ii], jj] > feats["ma20"][per["end"][ii], jj],
        "above_ma60": P["close"][per["end"][ii], jj] > feats["ma60"][per["end"][ii], jj],
        "above_ma200": P["close"][per["end"][ii], jj] > feats["ma200"][per["end"][ii], jj],
        "ma200_slope20": feats["ma200"][per["end"][ii], jj] / M.shift(feats["ma200"], 20)[per["end"][ii], jj] - 1,
        "amount20": feats["amount20"][per["end"][ii], jj],
        "turn20": feats["turn20"][per["end"][ii], jj],
        "vol_ratio": P["amount"][per["end"][ii], jj] / M.shift(feats["amount20"], 1)[per["end"][ii], jj],
        "is_st": P["is_st"][per["end"][ii], jj],
        "w_above": wf["w_above"][per["end"][ii], jj],
        "w_gold4": wf["w_gold4"][per["end"][ii], jj],
        "dif_slope3": (dif[ii, jj] - M.shift(dif, 3)[ii, jj]) / P["close"][per["end"][ii], jj] * 100,
    })
    e["zero_state"] = np.select(
        [(e.dif > 0) & (e.dea > 0), (e.dif > 0) & (e.dea <= 0),
         (e.dif <= 0) & (e.dea > 0), (e.dif <= 0) & (e.dea <= 0)],
        ["双上", "DIF上DEA下", "DIF下DEA上", "双下"], default="未知")
    r = rets[H][ii, jj]
    m = loo[H][ii, jj]
    v = ok[H][ii, jj] & np.isfinite(r) & np.isfinite(m)
    e[f"fwd{H}"] = np.where(v, r, np.nan)
    e[f"ex{H}"] = np.where(v, r - m, np.nan)
    e[f"ok{H}"] = v
    e[f"pct{H}"] = pct[H][ii, jj]
    for col in ["breadth20", "breadth200", "disp20", "CSI300_ret20", "CSI300_vol20",
                "CSI300_dd250", "ew_ret20", "ew_vol20", "ew_dd250"]:
        e[col] = mf[col].reindex(e.date).values
    e["index_above200"] = (mf["CSI300"].reindex(e.date).values >
                           mf["CSI300_ma200"].reindex(e.date).values)
    e["year"] = e.date.dt.year
    e["board"] = np.select(
        [e.code.str.startswith(("300", "301", "302")), e.code.str.startswith(("688", "689"))],
        ["创业板", "科创板"], default="主板")
    e["周期"] = label
    return e


def summarize(e, H, label):
    x = e.dropna(subset=[f"pct{H}", f"ex{H}"])
    dm = x.groupby("date")[f"ex{H}"].mean()
    rows = [dict(周期=label, 持有期=H, 信号数=len(x), 信号日数=x.date.nunique(),
                 平均分位=x[f"pct{H}"].mean(), 分位大于50=(x[f"pct{H}"] > 0.5).mean(),
                 平均超额=x[f"ex{H}"].mean(), 中位超额=x[f"ex{H}"].median(),
                 日度平均超额=dm.mean(), 日度t=nw_t(dm),
                 聚类t=cluster_t(x[f"ex{H}"], x.date),
                 分位_2010_2017=x[x.date < "2018-01-01"][f"pct{H}"].mean(),
                 分位_2018_2026=x[x.date >= "2018-01-01"][f"pct{H}"].mean(),
                 超额_2010_2017=x[x.date < "2018-01-01"][f"ex{H}"].mean(),
                 超额_2018_2026=x[x.date >= "2018-01-01"][f"ex{H}"].mean())]
    return pd.DataFrame(rows)


def main():
    P = M.load_panel()
    feats = M.stock_features(P)
    mf = M.market_features(P)
    wf = M.weekly_state(P)
    print("panel", P["shape"])

    all_sum, all_events, all_grid = [], [], []
    for freq, label, min_bars, horizons, primary in [
        ("W", "周线", 60, [4, 8, 12, 26], 4),
        ("M", "月线", 36, [3, 6, 12], 3),
    ]:
        per = build_period(P, freq)
        print(freq, "periods", per["nper"], "start", per["pid"][0], "end", per["pid"][-1])
        dif, dea, hist, gold, death, bars = period_signals(P, per, 12, 26, 9, min_bars)
        print("signals", int(gold.sum()))
        rets, ok, mkt, loo, pct = period_returns(P, per, horizons)
        for H in horizons:
            ev = event_table(P, per, dif, dea, hist, gold, death, rets, ok, mkt, loo, pct,
                             feats, mf, wf, H, label)
            all_sum.append(summarize(ev, H, label))
            if H == primary:
                all_events.append(ev)

        # 参数网格：主持有期
        rows = []
        H = primary
        for f in FASTS:
            for s in SLOWS:
                if f >= s:
                    continue
                for g in SIGS:
                    d2, e2, h2, g2, x2, b2 = period_signals(P, per, f, s, g, min_bars)
                    ii, jj = np.where(g2)
                    sd = P["dates"][per["end"][ii]]
                    keep = sd >= pd.Timestamp("2010-01-01")
                    ii, jj = ii[keep], jj[keep]
                    r = rets[H][ii, jj]
                    m = loo[H][ii, jj]
                    v = ok[H][ii, jj] & np.isfinite(r) & np.isfinite(m)
                    pp = pct[H][ii, jj][v]
                    ex = (r - m)[v]
                    dd = sd[keep][v]
                    if len(ex) < 100:
                        continue
                    rows.append(dict(周期=label, fast=f, slow=s, signal=g,
                                     n=len(ex), pct=float(np.nanmean(pp)),
                                     share50=float(np.nanmean(pp > 0.5)),
                                     ex=float(np.nanmean(ex)),
                                     med=float(np.nanmedian(ex)),
                                     t=cluster_t(ex, dd),
                                     ex_early=float(np.nanmean(ex[dd < pd.Timestamp("2018-01-01")])),
                                     ex_late=float(np.nanmean(ex[dd >= pd.Timestamp("2018-01-01")]))))
        grid = pd.DataFrame(rows)
        all_grid.append(grid)
        print(label, "grid", grid.shape, "pct range", grid.pct.min(), grid.pct.max())
        del dif, dea, hist, gold, death, rets, ok, mkt, loo, pct
        gc.collect()

    pd.concat(all_sum, ignore_index=True).to_csv(
        os.path.join(RES, "周期_汇总.csv"), index=False, encoding="utf-8-sig")
    pd.concat(all_events, ignore_index=True).to_parquet(
        os.path.join(RES, "周期_事件.parquet"), index=False)
    pd.concat(all_grid, ignore_index=True).to_csv(
        os.path.join(RES, "周期_参数网格.csv"), index=False, encoding="utf-8-sig")
    print(pd.concat(all_sum, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
