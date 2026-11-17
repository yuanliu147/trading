#!/usr/bin/env python3
"""MACD 连续特征 + 趋势/结构状态

与 02-15 的事件研究不同，这里把 MACD 当作每日可观测的连续状态：
柱体水平/变化/加速度/面积、DIF/DEA 水平与斜率、交叉年龄、红柱占比。
结构维度：长期趋势、中期排列、250 日位置、波动、短期高低点结构、换手、市场状态。
所有特征只使用 T 日及以前的数据。
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import macd_core as M


def build_macd_features(P):
    c = P["close_ff"]
    dif, dea, hist = M.macd(P)
    dif_pct = (dif / c * 100).astype(np.float32)
    dea_pct = (dea / c * 100).astype(np.float32)
    hist_pct = (hist / c * 100).astype(np.float32)
    gap_pct = ((dif - dea) / c * 100).astype(np.float32)

    h1 = M.shift(hist_pct, 1)
    h2 = M.shift(hist_pct, 2)
    hist_slope3 = (hist_pct - M.shift(hist_pct, 3)).astype(np.float32)
    hist_accel = (hist_pct - 2 * h1 + h2).astype(np.float32)

    hist_area20 = M.rolling_mean(hist_pct, 20)
    red_share20 = M.rolling_mean((hist_pct > 0).astype(np.float32), 20)

    dif_slope5 = (dif_pct - M.shift(dif_pct, 5)).astype(np.float32)
    dea_slope5 = (dea_pct - M.shift(dea_pct, 5)).astype(np.float32)

    mu = M.rolling_mean(dif_pct, 60)
    sd = M.rolling_std(dif_pct, 60)
    dif_z60 = np.full(dif_pct.shape, np.nan, dtype=np.float32)
    np.divide(dif_pct - mu, sd, out=dif_z60, where=sd > 1e-6)
    mu = M.rolling_mean(hist_pct, 60)
    sd = M.rolling_std(hist_pct, 60)
    hist_z60 = np.full(hist_pct.shape, np.nan, dtype=np.float32)
    np.divide(hist_pct - mu, sd, out=hist_z60, where=sd > 1e-6)

    gold, death = M.crosses(dif, dea, P)
    n = c.shape[0]
    tgrid = np.arange(n, dtype=np.int32)[:, None] * np.ones((1, c.shape[1]), dtype=np.int32)
    last_cross = np.maximum.accumulate(np.where(gold | death, tgrid, -1), axis=0)
    age = (tgrid - last_cross).astype(np.float32)
    age[last_cross < 0] = np.nan
    age = np.minimum(age, 250)
    cross_age = (age * np.where(dif > dea, 1, -1)).astype(np.float32)

    feats = {
        "柱体水平": hist_pct,
        "柱体3日变化": hist_slope3,
        "柱体加速度": hist_accel,
        "20日柱体均值": hist_area20,
        "柱体60日z": hist_z60,
        "DIF水平": dif_pct,
        "DIF-DEA距离": gap_pct,
        "DIF5日斜率": dif_slope5,
        "DEA5日斜率": dea_slope5,
        "DIF60日z": dif_z60,
        "交叉后天数": cross_age,
        "20日红柱占比": red_share20,
    }
    states = {
        "DIF>DEA": dif > dea,
        "DIF>0": dif > 0,
        "柱体>0": hist > 0,
        "柱体上行": hist > M.shift(hist, 1),
        "红柱放大": (hist > 0) & (hist > M.shift(hist, 1)),
        "绿柱缩短": (hist < 0) & (hist > M.shift(hist, 1)),
        "柱体上穿0": (hist > 0) & (M.shift(hist, 1) <= 0),
        "DIF上行": dif > M.shift(dif, 1),
    }
    # 柱体四态：红升/红降/绿升/绿降
    up = hist > M.shift(hist, 1)
    red = hist > 0
    hist_state = np.full(hist.shape, 4, dtype=np.int8)
    hist_state[red & up] = 0
    hist_state[red & ~up] = 1
    hist_state[~red & up] = 2
    hist_state[~red & ~up] = 3
    return dict(dif=dif, dea=dea, hist=hist, feats=feats, states=states,
                hist_state=hist_state)


def build_structure(P, mf):
    c = P["close_ff"]
    ma20 = M.rolling_mean(c, 20)
    ma60 = M.rolling_mean(c, 60)
    ma120 = M.rolling_mean(c, 120)
    ma200 = M.rolling_mean(c, 200)
    slope200 = (ma200 / M.shift(ma200, 20) - 1).astype(np.float32)

    ret1 = np.full_like(c, np.nan)
    ret1[1:] = c[1:] / c[:-1] - 1
    std20 = M.rolling_std(ret1, 20)
    turn20 = M.rolling_mean(P["turn"], 20)

    ret20 = np.full_like(c, np.nan)
    ret60 = np.full_like(c, np.nan)
    ret20[20:] = c[20:] / c[:-20] - 1
    ret60[60:] = c[60:] / c[:-60] - 1

    high20 = M.rolling_max(c, 20)
    low20 = M.rolling_min(c, 20)
    prev_high20 = M.shift(high20, 20)
    prev_low20 = M.shift(low20, 20)
    dist_high = (c / M.rolling_max(c, 250) - 1).astype(np.float32)
    dist_low = (c / M.rolling_min(c, 250) - 1).astype(np.float32)

    shape = c.shape
    trend = np.full(shape, -1, dtype=np.int8)
    up = (c > ma200) & (slope200 > 0)
    dn = (c < ma200) & (slope200 < 0)
    trend[up] = 0
    trend[dn] = 1
    trend[~up & ~dn] = 2
    trend[~np.isfinite(ma200)] = -1

    align = np.full(shape, -1, dtype=np.int8)
    bull = (c > ma20) & (ma20 > ma60) & (ma60 > ma120)
    bear = (c < ma20) & (ma20 < ma60) & (ma60 < ma120)
    align[bull] = 0
    align[bear] = 1
    align[~bull & ~bear] = 2
    align[~np.isfinite(ma120)] = -1

    pos = np.full(shape, -1, dtype=np.int8)
    pos[dist_high > -0.10] = 0
    pos[(dist_high <= -0.10) & (dist_high >= -0.30)] = 1
    pos[dist_high < -0.30] = 2
    pos[~np.isfinite(dist_high)] = -1

    struct20 = np.full(shape, -1, dtype=np.int8)
    hh = (high20 > prev_high20) & (low20 > prev_low20)
    ll = (high20 < prev_high20) & (low20 < prev_low20)
    struct20[hh] = 0
    struct20[ll] = 1
    struct20[~hh & ~ll] = 2
    struct20[~np.isfinite(prev_high20)] = -1

    market = market_regime(P, mf)
    market = np.repeat(market[:, None], c.shape[1], axis=1)

    return dict(ma20=ma20, ma60=ma60, ma120=ma120, ma200=ma200,
                slope200=slope200, std20=std20, turn20=turn20,
                ret20=ret20, ret60=ret60, dist_high=dist_high, dist_low=dist_low,
                trend=trend, align=align, pos=pos, struct20=struct20, market=market,
                dist_ma20=(c / ma20 - 1).astype(np.float32),
                dist_ma200=(c / ma200 - 1).astype(np.float32))


def row_quantile(x, base, nq):
    out = np.full(x.shape, -1, dtype=np.int8)
    for t in range(x.shape[0]):
        idx = np.where(base[t] & np.isfinite(x[t]))[0]
        if len(idx) < 20 * nq:
            continue
        r = rankdata(x[t, idx], method="average")
        q = np.minimum(((r - 1) * nq / len(idx)).astype(np.int16), nq - 1).astype(np.int8)
        out[t, idx] = q
    return out


def row_tercile(x, base):
    return row_quantile(x, base, 3)


def market_regime(P, mf):
    s = mf["CSI300"]
    ma = mf["CSI300_ma200"]
    slope = ma / ma.shift(20) - 1
    r = np.full(len(s), -1, dtype=np.int8)
    up = (s > ma) & (slope > 0)
    dn = (s < ma) & (slope < 0)
    r[up.to_numpy()] = 0
    r[dn.to_numpy()] = 1
    r[(~up & ~dn).to_numpy()] = 2
    r[~np.isfinite(s.to_numpy()) | ~np.isfinite(ma.to_numpy())] = -1
    return r


def base_mask(P):
    return P["valid"] & (P["bars"] >= 120) & ~P["is_st"].astype(bool)
