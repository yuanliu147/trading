#!/usr/bin/env python3
"""27 月度下行结构：月线 MACD 在长期下跌趋势里是否真的有效

26 发现月线结构出现强符号翻转：长期趋势下行时，柱体水平 IC 为正
（3 个月 +7.7%，12 个月 +11.8%），上行时为负。这里专门复核这个 pocket：
- 用 lag 3/6/12/24 的 Newey-West t，处理重叠收益
- 分 2010-2017 / 2018-2026、逐年
- 分市场状态、成交额、波动
- 控制月度动量/波动/换手/成交额后的 Fama-MacBeth
- Q1/Q5 的绝对收益与超额收益（判断 long-only 是否可用）

输出：03_结果/月度下行_IC.csv、月度下行_分年.csv、月度下行_FM.csv、月度下行_分位.csv
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
        out.append((p, np.corrcoef(rankdata(x[p, idx]), rankdata(ex[p, idx]))[0, 1], len(idx)))
    return pd.DataFrame(out, columns=["p", "ic", "n"])


def q_table(x, fwd, ex, mask, nq=5):
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
        rows.append((p, *qa, *qe, fwd[p, idx].mean(), idx.size))
    cols = (["p"] + [f"Q{i+1}_abs" for i in range(nq)] + [f"Q{i+1}_ex" for i in range(nq)]
            + ["市场绝对", "n"])
    return pd.DataFrame(rows, columns=cols)


def r01(x):
    return (rankdata(x) - 0.5) / len(x)


def fm_monthly(state, level, controls, ex, mask, dates, min_n=200):
    rows = []
    for p in range(ex.shape[0]):
        m = mask[p] & np.isfinite(ex[p]) & np.isfinite(level[p])
        for c in controls:
            m = m & np.isfinite(c[p])
        idx = np.where(m)[0]
        if len(idx) < min_n:
            continue
        s = state[p, idx].astype(float)
        if s.sum() < 30 or (1 - s).sum() < 30:
            continue
        lv = r01(level[p, idx]) - 0.5
        cs = [r01(c[p, idx]) for c in controls]
        y = ex[p, idx].astype(float)
        X = np.column_stack([np.ones(len(idx)), lv, *cs, s, s * lv])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append((dates[p], *beta))
    cols = ["date", "const", "level", *[f"c{i}" for i in range(len(controls))], "state", "state_x_level"]
    d = pd.DataFrame(rows, columns=cols).set_index("date")
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
    gap = (dif - dea) / c * 100
    hist_chg = hist_pct - M.shift(hist_pct, 1)
    rets, ok, mkt, loo = prets(P, per, [3, 12])
    trend = st["trend"][per["end"], :]
    down = trend == 1
    up = trend == 0

    # 月度动量控制
    close_s = pd.DataFrame(per["close"])
    m1 = (close_s / close_s.shift(1) - 1).to_numpy(np.float32)
    m3 = (close_s / close_s.shift(3) - 1).to_numpy(np.float32)
    m12 = (close_s / close_s.shift(12) - 1).to_numpy(np.float32)
    vol = st["std20"][per["end"], :]
    turn = st["turn20"][per["end"], :]
    amt = np.log1p(np.maximum(M.rolling_mean(P["amount"], 20)[per["end"], :], 0)).astype(np.float32)
    controls = [m1, m3, m12, vol, turn, amt]

    feats = {"柱体水平": hist_pct, "柱体变化": hist_chg, "DIF水平": dif_pct, "DIF-DEA距离": gap}
    ic_rows, year_rows, fm_rows, q_rows, strict_rows = [], [], [], [], []
    dates = P["dates"][per["end"]]
    for H in [3, 12]:
        ex = rets[H] - loo[H]
        fwd = rets[H]
        mask0 = valid & ok[H] & np.isfinite(ex)
        for bucket, bmask in [("下行", down), ("上行", up), ("全样本", np.ones_like(down, bool))]:
            m = mask0 & bmask
            for name, x in feats.items():
                s = ic_series(x, ex, m)
                if len(s) < 20:
                    continue
                ss = pd.Series(s.ic.values, index=dates[s.p.values])
                early = ss.index < pd.Timestamp("2018-01-01")
                ic_rows.append(dict(周期=H, 结构=bucket, 特征=name, 月数=len(ss),
                                    平均IC=ss.mean(), IC_t3=nw_t(ss, lag=3), IC_t6=nw_t(ss, lag=6),
                                    IC_t12=nw_t(ss, lag=12), IC_t24=nw_t(ss, lag=24),
                                    IC大于0占比=(ss > 0).mean(),
                                    IC_2010_2017=ss[early].mean(), IC_2018_2026=ss[~early].mean()))
                for yr, v in ss.groupby(ss.index.year).mean().items():
                    year_rows.append(dict(周期=H, 结构=bucket, 特征=name, 年份=int(yr), IC=v))
            # 状态：DIF>DEA / 红柱放大
            for sname, state in [("DIF>DEA", dif > dea), ("红柱放大", (hist > 0) & (hist > M.shift(hist, 1)))]:
                raw, adj = [], []
                for p in range(per["nper"]):
                    idx = np.where(m[p] & np.isfinite(ex[p]))[0]
                    if len(idx) < 100:
                        continue
                    sv = state[p, idx]
                    if sv.sum() < 20 or (~sv).sum() < 20:
                        continue
                    y = ex[p, idx]
                    raw.append(y[sv].mean() - y[~sv].mean())
                    lv = hist_pct[p, idx] if sname == "红柱放大" else dif_pct[p, idx]
                    lq = np.minimum(((rankdata(lv) - 1) * 3 / len(idx)).astype(np.int16), 2)
                    d, w = [], []
                    for a in range(3):
                        cell = lq == a
                        c1, c0 = int((cell & sv).sum()), int((cell & ~sv).sum())
                        if c1 >= 8 and c0 >= 8:
                            d.append(y[cell & sv].mean() - y[cell & ~sv].mean()); w.append(c1 + c0)
                    if d:
                        adj.append(np.average(d, weights=w))
                if len(raw) < 20:
                    continue
                r = pd.Series(raw); a = pd.Series(adj)
                fm_rows.append(dict(周期=H, 结构=bucket, 状态=sname, 月数=len(r),
                                    原始差=r.mean(), 原始t=nw_t(r, lag=12),
                                    控制水平后差=a.mean(), 控制水平t=nw_t(a, lag=12)))
                if bucket == "下行":
                    lvl = hist_pct if sname == "红柱放大" else dif_pct
                    for r in fm_monthly(state, lvl, controls, ex, m, dates)[0]:
                        r.update(周期=H, 状态=sname)
                        strict_rows.append(r)
            # Q1/Q5 绝对/超额
            for name in ["柱体水平", "DIF水平"]:
                qt = q_table(feats[name], fwd, ex, m)
                if qt.empty:
                    continue
                q_rows.append(dict(周期=H, 结构=bucket, 特征=name, 月数=len(qt),
                                   Q1绝对=qt.Q1_abs.mean(), Q5绝对=qt.Q5_abs.mean(),
                                   Q1超额=qt.Q1_ex.mean(), Q5超额=qt.Q5_ex.mean(),
                                   多空绝对=qt.Q5_abs.mean() - qt.Q1_abs.mean(),
                                   多空超额=qt.Q5_ex.mean() - qt.Q1_ex.mean(),
                                   市场绝对=qt.市场绝对.mean()))
        print("完成", H)
    pd.DataFrame(ic_rows).to_csv(os.path.join(RES, "月度下行_IC.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(year_rows).to_csv(os.path.join(RES, "月度下行_分年.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(fm_rows).to_csv(os.path.join(RES, "月度下行_状态.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(q_rows).to_csv(os.path.join(RES, "月度下行_分位.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(strict_rows).to_csv(os.path.join(RES, "月度下行_FM.csv"), index=False, encoding="utf-8-sig")
    print(pd.DataFrame(ic_rows).to_string(index=False))
    print(pd.DataFrame(q_rows).to_string(index=False))
    print(pd.DataFrame(fm_rows).to_string(index=False))
    print(pd.DataFrame(strict_rows).to_string(index=False))


if __name__ == "__main__":
    main()
