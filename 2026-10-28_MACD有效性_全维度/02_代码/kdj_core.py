#!/usr/bin/env python3
"""KDJ 公共引擎（与 macd_core 同一套面板/口径）

定义
----
RSV_t = (C_t - LLV(L,N)) / (HHV(H,N) - LLV(L,N)) * 100
K_t   = (M1-1)/M1 * K_{t-1} + 1/M1 * RSV_t
D_t   = (M2-1)/M2 * D_{t-1} + 1/M2 * K_t
J_t   = 3*K_t - 2*D_t

- 标准参数 N=9, M1=3, M2=3；初始 K=D=50；区间为 0 时 RSV=50。
- 价格用后复权 high/low/close；停牌日不产生信号。
- 周/月线 KDJ 用对应周期的最高/最低/收盘，已收盘周期滞后一期映射到日线。
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

import macd_core as M


def _filled_hlc(P):
    c = P["close_ff"]
    h = M._fill_lead(P["high"])
    l = M._fill_lead(P["low"])
    return h, l, c


def kdj(P, n=9, m1=3, m2=3, high=None, low=None, close=None):
    """按列计算 KDJ。返回 K/D/J/RSV，形状与面板一致。"""
    if high is None or low is None or close is None:
        h, l, c = _filled_hlc(P)
    else:
        h, l, c = high, low, close
    hh = pd.DataFrame(h).rolling(n, min_periods=n).max().to_numpy(dtype=np.float32)
    ll = pd.DataFrame(l).rolling(n, min_periods=n).min().to_numpy(dtype=np.float32)
    rng = hh - ll
    rsv = np.full(c.shape, np.nan, dtype=np.float32)
    np.divide((c - ll) * 100.0, rng, out=rsv, where=rng > 1e-6)
    rsv[rng <= 1e-6] = 50.0
    # 数据不足 N 日：用最早有效值回填，保证 lfilter 无 NaN；之后由 base_mask 截断
    rsv = M._fill_lead(rsv)
    a1 = (m1 - 1.0) / m1
    a2 = (m2 - 1.0) / m2
    k0 = np.full((1, c.shape[1]), 50.0, dtype=np.float32)
    k, _ = lfilter([1 - a1], [1, -a1], rsv, axis=0, zi=a1 * k0)
    d0 = np.full((1, c.shape[1]), 50.0, dtype=np.float32)
    d, _ = lfilter([1 - a2], [1, -a2], k, axis=0, zi=a2 * d0)
    j = 3.0 * k - 2.0 * d
    return (k.astype(np.float32), d.astype(np.float32), j.astype(np.float32),
            rsv.astype(np.float32))


def kdj_crosses(k, d, P, min_bars=120):
    prev_k, prev_d = M.shift(k, 1), M.shift(d, 1)
    base = P["valid"] & (P["bars"] >= min_bars) & np.isfinite(k) & np.isfinite(prev_k)
    gold = base & (k > d) & (prev_k <= prev_d)
    death = base & (k < d) & (prev_k >= prev_d)
    return gold, death


def cross_age(k, d, P):
    """带符号的交叉年龄：正=金叉后第几天，负=死叉后第几天。"""
    gold, death = kdj_crosses(k, d, P)
    n = k.shape[0]
    tgrid = np.arange(n, dtype=np.int32)[:, None] * np.ones((1, k.shape[1]), dtype=np.int32)
    last = np.maximum.accumulate(np.where(gold | death, tgrid, -1), axis=0)
    age = (tgrid - last).astype(np.float32)
    age[last < 0] = np.nan
    age = np.minimum(age, 250)
    return (age * np.where(k > d, 1, -1)).astype(np.float32)


def kdj_features(P, n=9, m1=3, m2=3):
    k, d, j, rsv = kdj(P, n, m1, m2)
    gold, death = kdj_crosses(k, d, P)
    age = cross_age(k, d, P)
    gap = (k - d).astype(np.float32)
    gold_cs = np.cumsum(gold, axis=0)
    death_cs = np.cumsum(death, axis=0)
    cross20 = (gold_cs - M.shift(gold_cs, 20, fill=0)) + (death_cs - M.shift(death_cs, 20, fill=0))
    return dict(
        k=k, d=d, j=j, rsv=rsv, gap=gap, age=age, gold=gold, death=death,
        cross20=cross20.astype(np.float32),
        feats={
            "K水平": k, "D水平": d, "J水平": j,
            "K-D差": gap, "K3日斜率": k - M.shift(k, 3), "K5日斜率": k - M.shift(k, 5),
            "J3日变化": j - M.shift(j, 3),
            "KDJ交叉后天数": age, "20日KDJ交叉次数": cross20,
            "RSV": rsv,
        },
        states={
            "K>D": k > d,
            "K>50": k > 50,
            "D>50": d > 50,
            "J>100": j > 100,
            "J<0": j < 0,
            "超买K>80": k > 80,
            "超卖K<20": k < 20,
            "高位K>70": k > 70,
            "低位K<30": k < 30,
        },
    )


def period_ohlc(P, freq):
    """周/月线 OHLC；close 用周期最后一个交易日，high/low 用周期内极值。"""
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
    open_ = P["open"][start, :].copy()
    close[P["trade"][end, :] != 1] = np.nan
    open_[P["trade"][start, :] != 1] = np.nan
    vh = np.where(P["valid"], P["high"], np.nan)
    vl = np.where(P["valid"], P["low"], np.nan)
    high = np.full((nper, P["shape"][1]), np.nan, dtype=np.float32)
    low = np.full_like(high, np.nan)
    for p in range(nper):
        a, b = start[p], end[p] + 1
        high[p] = np.nanmax(vh[a:b], axis=0)
        low[p] = np.nanmin(vl[a:b], axis=0)
    return dict(freq=freq, pid=uniq, inv=inv, start=start, end=end,
                open=open_, high=high, low=low, close=close, nper=nper)


def period_kdj_state(P, freq, n=9, m1=3, m2=3, min_bars=26):
    """已收盘周/月线 KDJ 状态，滞后一期映射到日线（严格无未来）。"""
    per = period_ohlc(P, freq)
    h = M._fill_lead(per["high"])
    l = M._fill_lead(per["low"])
    c = M._fill_lead(per["close"])
    k, d, j, _ = kdj(P, n, m1, m2, high=h, low=l, close=c)
    valid = ~np.isnan(per["close"])
    bars = np.cumsum(valid, axis=0)
    gold = valid & (bars >= min_bars) & (k > d) & (M.shift(k, 1) <= M.shift(d, 1))
    death = valid & (bars >= min_bars) & (k < d) & (M.shift(k, 1) >= M.shift(d, 1))
    # shift 1: 只用上一期已收盘状态
    k1, d1, j1 = M.shift(k, 1), M.shift(d, 1), M.shift(j, 1)
    g1, x1 = M.shift(gold, 1), M.shift(death, 1)
    idx = per["inv"]
    out = dict(
        k=k1[idx], d=d1[idx], j=j1[idx],
        state=(k1 > d1)[idx] & np.isfinite(k1[idx]),
        gold=g1[idx].astype(bool), death=x1[idx].astype(bool),
        ok=np.isfinite(k1[idx]),
    )
    # 过去 4 周/3 月内是否有金叉
    recent = np.zeros_like(gold, dtype=np.float32)
    look = 4 if freq == "W" else 3
    for q in range(look):
        recent += M.shift(gold.astype(np.float32), q + 1)
    out["gold_recent"] = recent[idx] > 0
    return out
