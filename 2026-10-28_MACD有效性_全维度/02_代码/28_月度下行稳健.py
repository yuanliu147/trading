#!/usr/bin/env python3
"""28 月度下行 pocket 稳健性：是不是 ST/小票/低价/低流动性/熊市反弹造成的

27 发现月线 MACD 在长期下行趋势里 IC 为正（3 个月 +7.7%，12 个月 +11.8%）。
这里逐层剔除：
- ST、上市不足 250 日、低价（<2 元）
- 成交额低半区
- 市场状态（沪深300 年线上/下）
并对柱体水平做连续 Fama-MacBeth（控制 1/3/12 月动量、波动、换手、成交额），
对 DIF>DEA 做二元 FM。

输出：03_结果/月度下行_稳健.csv、月度下行_稳健FM.csv、月度下行_稳健分年.csv
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


def build_period(P):
    pid = P["dates"].to_period("M")
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
    return dict(pid=uniq, inv=inv, start=start, end=end, close=close, open=open_, nper=nper)


def sig(per, fast=12, slow=26, sg=9, min_bars=36):
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


def ic_series(x, ex, mask):
    out = []
    for p in range(x.shape[0]):
        idx = np.where(mask[p] & np.isfinite(ex[p]) & np.isfinite(x[p]))[0]
        if len(idx) < 100:
            continue
        out.append((p, np.corrcoef(rankdata(x[p, idx]), rankdata(ex[p, idx]))[0, 1]))
    return pd.DataFrame(out, columns=["p", "ic"])


def qtab(x, fwd, ex, mask, nq=5):
    rows = []
    for p in range(x.shape[0]):
        idx = np.where(mask[p] & np.isfinite(fwd[p]) & np.isfinite(x[p]))[0]
        if len(idx) < 100:
            continue
        q = np.minimum(((rankdata(x[p, idx]) - 1) * nq / len(idx)).astype(np.int16), nq - 1)
        cnt = np.bincount(q, minlength=nq)
        if (cnt == 0).any():
            continue
        qa = np.bincount(q, weights=fwd[p, idx], minlength=nq) / cnt
        qe = np.bincount(q, weights=ex[p, idx], minlength=nq) / cnt
        rows.append((p, qa[0], qa[-1], qe[0], qe[-1], fwd[p, idx].mean()))
    return pd.DataFrame(rows, columns=["p", "Q1绝对", "Q5绝对", "Q1超额", "Q5超额", "样本绝对"])


def r01(x):
    return (rankdata(x) - 0.5) / len(x)


def fm_cont(level, controls, ex, mask, dates):
    rows = []
    for p in range(ex.shape[0]):
        m = mask[p] & np.isfinite(ex[p]) & np.isfinite(level[p])
        for c in controls:
            m = m & np.isfinite(c[p])
        idx = np.where(m)[0]
        if len(idx) < 200:
            continue
        X = np.column_stack([np.ones(len(idx)), r01(level[p, idx]),
                             *[r01(c[p, idx]) for c in controls]])
        y = ex[p, idx].astype(float)
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append((dates[p], beta[1]))
    d = pd.DataFrame(rows, columns=["date", "coef"]).set_index("date")
    early = d.index < pd.Timestamp("2018-01-01")
    return dict(项="level_cont", 系数=d.coef.mean(), t=nw_t(d.coef, lag=12),
                系数_2010_2017=d.coef[early].mean(), 系数_2018_2026=d.coef[~early].mean(),
                正系数年份=int((d.coef.groupby(d.index.year).mean() > 0).sum()),
                年份数=int(d.coef.groupby(d.index.year).mean().notna().sum())), len(d)


def fm_state(state, level, controls, ex, mask, dates):
    rows = []
    for p in range(ex.shape[0]):
        m = mask[p] & np.isfinite(ex[p]) & np.isfinite(level[p])
        for c in controls:
            m = m & np.isfinite(c[p])
        idx = np.where(m)[0]
        if len(idx) < 200:
            continue
        s = state[p, idx].astype(float)
        if s.sum() < 30 or (1 - s).sum() < 30:
            continue
        lv = r01(level[p, idx]) - 0.5
        X = np.column_stack([np.ones(len(idx)), lv, *[r01(c[p, idx]) for c in controls],
                             s, s * lv])
        y = ex[p, idx].astype(float)
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append((dates[p], beta[1], beta[-2], beta[-1]))
    d = pd.DataFrame(rows, columns=["date", "level", "state", "state_x_level"]).set_index("date")
    out = []
    for c in ["level", "state", "state_x_level"]:
        early = d.index < pd.Timestamp("2018-01-01")
        out.append(dict(项=c, 系数=d[c].mean(), t=nw_t(d[c], lag=12),
                        系数_2010_2017=d[c][early].mean(), 系数_2018_2026=d[c][~early].mean(),
                        正系数年份=int((d[c].groupby(d.index.year).mean() > 0).sum()),
                        年份数=int(d[c].groupby(d.index.year).mean().notna().sum())))
    return out, len(d)


def main():
    P = M.load_panel()
    mf = M.market_features(P)
    st = F.build_structure(P, mf)
    per = build_period(P)
    dif, dea, hist, valid = sig(per)
    c = M._fill_lead(per["close"])
    hist_pct = hist / c * 100
    dif_pct = dif / c * 100
    rets, ok, mkt, loo = prets(P, per, [3, 12])
    trend = st["trend"][per["end"], :]
    down = trend == 1

    is_st = P["is_st"][per["end"], :] == 1
    bars = P["bars"][per["end"], :]
    price = P["raw_close"][per["end"], :]
    amt = M.rolling_mean(P["amount"], 20)[per["end"], :]

    close_s = pd.DataFrame(per["close"])
    m1 = (close_s / close_s.shift(1) - 1).to_numpy(np.float32)
    m3 = (close_s / close_s.shift(3) - 1).to_numpy(np.float32)
    m12 = (close_s / close_s.shift(12) - 1).to_numpy(np.float32)
    vol = st["std20"][per["end"], :]
    turn = st["turn20"][per["end"], :]
    logamt = np.log1p(np.maximum(amt, 0)).astype(np.float32)
    controls = [m1, m3, m12, vol, turn, logamt]

    rows, fm_rows, year_rows = [], [], []
    for H in [3, 12]:
        ex = rets[H] - loo[H]
        fwd = rets[H]
        base = valid & ok[H] & np.isfinite(ex)
        liq_base = base & ~is_st & (bars >= 250) & (price > 2)
        # 月度成交额分位（在非 ST/高价样本内）
        liq = np.full(amt.shape, np.nan, dtype=np.float32)
        for p in range(per["nper"]):
            idx = np.where(liq_base[p] & np.isfinite(amt[p]))[0]
            if len(idx) >= 100:
                liq[p, idx] = (rankdata(amt[p, idx]) - 0.5) / len(idx)
        samples = {
            "下行_全": down,
            "下行_非ST": down & ~is_st,
            "下行_非ST_高价": down & ~is_st & (bars >= 250) & (price > 2),
            "下行_非ST_高价_高流动性": down & ~is_st & (bars >= 250) & (price > 2) & (liq >= 0.5),
            "下行_非ST_高价_低流动性": down & ~is_st & (bars >= 250) & (price > 2) & (liq < 0.5),
            "上行_非ST_高价_高流动性": (trend == 0) & ~is_st & (bars >= 250) & (price > 2) & (liq >= 0.5),
        }
        for name, sm in samples.items():
            m = base & sm
            if m.sum() < 5000:
                continue
            s = ic_series(hist_pct, ex, m)
            ss = pd.Series(s.ic.values, index=P["dates"][per["end"]][s.p.values])
            early = ss.index < pd.Timestamp("2018-01-01")
            rows.append(dict(周期=H, 样本=name, 月数=len(ss), 平均IC=ss.mean(),
                             IC_t12=nw_t(ss, lag=12), IC_2010_2017=ss[early].mean(),
                             IC_2018_2026=ss[~early].mean(), IC大于0占比=(ss > 0).mean()))
            qt = qtab(hist_pct, fwd, ex, m)
            rows[-1].update(Q1绝对=qt.Q1绝对.mean(), Q5绝对=qt.Q5绝对.mean(),
                            Q1超额=qt.Q1超额.mean(), Q5超额=qt.Q5超额.mean(),
                            样本绝对=qt.样本绝对.mean())
            if name == "下行_非ST_高价_高流动性":
                for yr, v in ss.groupby(ss.index.year).mean().items():
                    year_rows.append(dict(周期=H, 年份=int(yr), IC=v))
        # 市场状态拆分（只在高流动性非 ST 下行样本）
        hi = base & samples["下行_非ST_高价_高流动性"]
        s_mkt = (mf["CSI300"] > mf["CSI300_ma200"]).reindex(P["dates"][per["end"]]).fillna(False).to_numpy()
        v_mkt = mf["CSI300_vol20"].reindex(P["dates"][per["end"]]).to_numpy()
        v_cut = np.nanmedian(v_mkt)
        splits = {
            "下行_高流动_市场年线上": s_mkt,
            "下行_高流动_市场年线下": ~s_mkt,
            "下行_高流动_市场高波": v_mkt > v_cut,
            "下行_高流动_市场低波": v_mkt <= v_cut,
        }
        for name, sp in splits.items():
            m = hi & sp[:, None]
            if m.sum() < 3000:
                continue
            s = ic_series(hist_pct, ex, m)
            ss = pd.Series(s.ic.values, index=P["dates"][per["end"]][s.p.values])
            if len(ss) < 20:
                continue
            rows.append(dict(周期=H, 样本=name, 月数=len(ss), 平均IC=ss.mean(),
                             IC_t12=nw_t(ss, lag=12),
                             IC_2010_2017=np.nan, IC_2018_2026=np.nan,
                             IC大于0占比=(ss > 0).mean()))
        # FM：只在高流动性非 ST 下行样本
        m = base & samples["下行_非ST_高价_高流动性"]
        r = fm_cont(hist_pct, controls, ex, m, P["dates"][per["end"]])
        if r:
            r[0].update(周期=H, 检验="柱体水平连续")
            fm_rows.append(r[0])
        for sname, state in [("DIF>DEA", dif > dea)]:
            res = fm_state(state, dif_pct, controls, ex, m, P["dates"][per["end"]])
            for rr in res[0]:
                rr.update(周期=H, 检验=sname)
                fm_rows.append(rr)
        print("完成", H)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "月度下行_稳健.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(fm_rows).to_csv(os.path.join(RES, "月度下行_稳健FM.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(year_rows).to_csv(os.path.join(RES, "月度下行_稳健分年.csv"), index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(fm_rows).to_string(index=False))


if __name__ == "__main__":
    main()
