#!/usr/bin/env python3
"""MACD 有效性研究：公共引擎

口径
----
- 价格：baostock 后复权（hfq）用于 MACD/收益；不复权用于停牌、涨跌停、ST 判定。
- 信号：T 日收盘确认，T+1 开盘买入（无法买入的信号剔除）。
- 前瞻收益：fwd_N = hfq_close[T+N] / hfq_open[T+1] - 1；N=5/10/20/60。
- 超额：减去同一信号日全市场等权同口径收益（leave-one-out）。
- 市场状态：沪深300 + 全市场等权指数 + 宽度。
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = "/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度"
RES = os.path.join(ROOT, "03_结果")

HORIZONS = (5, 10, 20, 60)


def load_panel():
    df = pd.read_parquet(os.path.join(RES, "面板_日线.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "thscode"]).reset_index(drop=True)
    dates = np.sort(df["date"].unique())
    codes = np.sort(df["thscode"].unique())
    di = pd.Categorical(df["date"], categories=dates).codes
    ci = pd.Categorical(df["thscode"], categories=codes).codes
    shape = (len(dates), len(codes))

    def mat(col, dtype=np.float32):
        a = np.full(shape, np.nan, dtype=dtype)
        a[di, ci] = df[col].values
        return a

    P = dict(
        dates=pd.DatetimeIndex(dates), codes=codes, shape=shape,
        close=mat("hfq_close"), open=mat("hfq_open"),
        high=mat("hfq_high"), low=mat("hfq_low"),
        raw_open=mat("open"), raw_high=mat("high"), raw_low=mat("low"),
        raw_close=mat("close"), preclose=mat("preclose"),
        volume=mat("volume"), amount=mat("amount"), turn=mat("turn"),
        pct=mat("pct_chg"), trade=mat("trade_status"), is_st=mat("is_st"),
    )
    # 有效收盘：有数据且当日交易（停牌日不算信号）
    P["valid"] = (~np.isnan(P["close"])) & (P["trade"] == 1)
    P["bars"] = np.cumsum(P["valid"], axis=0)
    P["close_ff"] = _fill_lead(P["close"])
    # 前收/前日状态
    return P


def _fill_lead(x):
    """沿时间轴前向填充；开头的 NaN 用第一个有效值回填（输出前会 mask）。"""
    df = pd.DataFrame(x)
    return df.ffill().bfill().to_numpy(dtype=np.float32)


def ema(x, span):
    """按列（股票）计算 EMA，首值播种为第一个有效值。x 允许 NaN。"""
    return ema_filled(_fill_lead(x), span)


def ema_filled(xf, span):
    """x 已前向填充且无 NaN；按列 EMA，y0=x0。"""
    a = 2.0 / (span + 1.0)
    zi = ((1 - a) * xf[0]).reshape(1, -1)
    y, _ = lfilter([a], [1, -(1 - a)], xf, axis=0, zi=zi)
    return y.astype(np.float32)


def macd(P, fast=12, slow=26, sig=9):
    c = P.get("close_ff")
    if c is None:
        c = _fill_lead(P["close"])
    dif = ema_filled(c, fast) - ema_filled(c, slow)
    dea = ema_filled(dif, sig)
    hist = 2.0 * (dif - dea)
    return dif, dea, hist


def shift(x, k, fill=np.nan):
    out = np.full_like(x, fill)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def crosses(dif, dea, P, min_bars=120):
    prev_d = shift(dif, 1)
    prev_e = shift(dea, 1)
    base = P["valid"] & (P["bars"] >= min_bars) & ~np.isnan(dif) & ~np.isnan(prev_d)
    gold = base & (dif > dea) & (prev_d <= prev_e)
    death = base & (dif < dea) & (prev_d >= prev_e)
    return gold, death


def forward_returns(P, horizons=HORIZONS):
    """fwd[h] = close[T+h]/open[T+1]-1，附带可成交掩码。"""
    c, o = P["close"], P["open"]
    n = c.shape[0]
    out, ok = {}, {}
    for h in horizons:
        f = np.full(c.shape, np.nan, dtype=np.float32)
        if n > h + 1:
            f[: n - h] = c[h:] / o[1 : n - h + 1] - 1.0
        # 入场日可交易、非一字涨停；出场日可交易
        entry_ok = np.zeros(c.shape, dtype=bool)
        if n > h + 1:
            entry_ok[: n - h] = (P["trade"][1 : n - h + 1] == 1) & \
                                ~((P["raw_high"][1 : n - h + 1] == P["raw_low"][1 : n - h + 1]) &
                                  (P["raw_open"][1 : n - h + 1] > P["preclose"][1 : n - h + 1]))
        exit_ok = np.zeros(c.shape, dtype=bool)
        if n > h:
            exit_ok[: n - h] = P["trade"][h:] == 1
        m = entry_ok & exit_ok & np.isfinite(f)
        out[h] = f
        ok[h] = m
    # 同一信号日的等权市场收益（leave-one-out）
    mkt, mkt_loo = {}, {}
    for h in horizons:
        f = out[h]
        cnt = np.sum(ok[h] & np.isfinite(f), axis=1).astype(np.float64)
        s = np.nansum(np.where(ok[h], f, np.nan), axis=1, dtype=np.float64)
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        mkt[h] = mean.astype(np.float32)
        loo = np.where((cnt > 1)[:, None],
                       (s[:, None] - np.where(ok[h], f, 0)) / np.maximum(cnt[:, None] - 1, 1), np.nan)
        mkt_loo[h] = loo.astype(np.float32)
    return out, ok, mkt, mkt_loo


def rolling_mean(x, w, minp=None):
    if minp is None:
        minp = max(2, w // 2)
    return pd.DataFrame(x).rolling(w, min_periods=minp).mean().to_numpy(dtype=np.float32)


def rolling_max(x, w, minp=None):
    if minp is None:
        minp = max(2, w // 2)
    return pd.DataFrame(x).rolling(w, min_periods=minp).max().to_numpy(dtype=np.float32)


def rolling_min(x, w, minp=None):
    if minp is None:
        minp = max(2, w // 2)
    return pd.DataFrame(x).rolling(w, min_periods=minp).min().to_numpy(dtype=np.float32)


def rolling_std(x, w, minp=None):
    if minp is None:
        minp = max(2, w // 2)
    return pd.DataFrame(x).rolling(w, min_periods=minp).std().to_numpy(dtype=np.float32)


def market_features(P):
    c = P["close"]
    ret1 = np.full_like(c, np.nan)
    ret1[1:] = c[1:] / c[:-1] - 1
    # 全市场等权日收益（仅交易股）
    ew = np.nanmean(np.where(P["valid"], ret1, np.nan), axis=1)
    ew_idx = np.cumprod(1 + np.nan_to_num(ew, nan=0.0))
    ma20, ma200 = rolling_mean(c, 20), rolling_mean(c, 200)
    breadth20 = np.nanmean(np.where(P["valid"] & np.isfinite(ma20), c > ma20, np.nan), axis=1)
    breadth200 = np.nanmean(np.where(P["valid"] & np.isfinite(ma200), c > ma200, np.nan), axis=1)
    # 横截面离散度：20 日收益的标准差
    ret20 = np.full_like(c, np.nan)
    ret20[20:] = c[20:] / c[:-20] - 1
    disp20 = np.nanstd(np.where(P["valid"], ret20, np.nan), axis=1)
    df = pd.DataFrame({"date": P["dates"], "ew": ew, "ew_idx": ew_idx,
                       "breadth20": breadth20, "breadth200": breadth200,
                       "disp20": disp20})
    # 加入指数
    idx = pd.read_parquet(os.path.join(RES, "指数_日线.parquet"))
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.pivot_table(index="date", columns="idx_name", values="close").sort_index()
    for k in ["CSI300", "CSI500", "CYB"]:
        if k in idx.columns:
            s = idx[k].reindex(P["dates"]).ffill()
            df[k] = s.values
            df[k + "_ma200"] = s.rolling(200, min_periods=100).mean().values
            df[k + "_ma60"] = s.rolling(60, min_periods=30).mean().values
            df[k + "_ret20"] = (s / s.shift(20) - 1).values
            df[k + "_vol20"] = s.pct_change().rolling(20, min_periods=10).std().values * np.sqrt(252)
            df[k + "_dd250"] = (s / s.rolling(250, min_periods=60).max() - 1).values
    df["ew_ma200"] = pd.Series(df.ew_idx).rolling(200, min_periods=100).mean().values
    df["ew_ma60"] = pd.Series(df.ew_idx).rolling(60, min_periods=30).mean().values
    df["ew_ret20"] = pd.Series(df.ew_idx).pct_change(20).values
    df["ew_vol20"] = pd.Series(df.ew_idx).pct_change().rolling(20, min_periods=10).std().values * np.sqrt(252)
    df["ew_dd250"] = (df.ew_idx / pd.Series(df.ew_idx).rolling(250, min_periods=60).max() - 1).values
    return df.set_index("date")


def weekly_state(P, fast=12, slow=26, sig=9):
    """上周已收盘周线的 MACD 状态；映射到下一周（严格无未来）。"""
    dates = P["dates"]
    # 每周最后一个交易日（按 W-FRI）
    wk = dates.to_period("W-FRI")
    wk_codes, wk_idx = np.unique(wk, return_inverse=True)
    nw, nc = len(wk_codes), len(P["codes"])
    # 每个股票每周最后有效收盘
    wclose = np.full((nw, nc), np.nan, dtype=np.float32)
    for i in range(len(dates)):
        row = np.where(P["valid"][i], P["close"][i], np.nan)
        wclose[wk_idx[i]] = np.where(np.isfinite(row), row, wclose[wk_idx[i]])
    dif = ema(wclose, fast) - ema(wclose, slow)
    dea = ema(dif, sig)
    gold = (dif > dea) & (shift(dif, 1) <= shift(dea, 1))
    # 只使用已完成的周：对每日，取上一周（shift 1）
    w_dif = shift(dif, 1)[wk_idx]
    w_dea = shift(dea, 1)[wk_idx]
    w_gold = shift(gold, 1)[wk_idx]
    w_above = (w_dif > w_dea) & np.isfinite(w_dif)
    # 过去 4 周内周线金叉
    gold4 = np.zeros_like(gold, dtype=np.float32)
    for k in range(4):
        gold4 += shift(gold.astype(np.float32), k + 1)
    w_gold4 = gold4[wk_idx]
    return dict(w_dif=w_dif, w_dea=w_dea, w_above=w_above, w_gold=w_gold, w_gold4=w_gold4)


def stock_features(P):
    c = P["close"]
    ma20, ma60, ma200 = rolling_mean(c, 20), rolling_mean(c, 60), rolling_mean(c, 200)
    ret1 = np.full_like(c, np.nan)
    ret1[1:] = c[1:] / c[:-1] - 1
    std20 = rolling_std(ret1, 20)
    ret20, ret60 = np.full_like(c, np.nan), np.full_like(c, np.nan)
    ret20[20:] = c[20:] / c[:-20] - 1
    ret60[60:] = c[60:] / c[:-60] - 1
    high250 = rolling_max(c, 250)
    dist_high = c / high250 - 1
    amount20 = rolling_mean(P["amount"], 20)
    turn20 = rolling_mean(P["turn"], 20)
    return dict(ma20=ma20, ma60=ma60, ma200=ma200, std20=std20, ret20=ret20,
                ret60=ret60, dist_high=dist_high, amount20=amount20, turn20=turn20)


def event_frame(P, gold, feats, mf, wf, fwd, ok, mkt_loo, dif, dea, hist, death, path=None, name="gold"):
    """把信号矩阵展开为逐笔事件表。"""
    ii, jj = np.where(gold)
    # 只要 2010 起
    keep = P["dates"][ii] >= pd.Timestamp("2010-01-01")
    ii, jj = ii[keep], jj[keep]
    e = pd.DataFrame({
        "date": P["dates"][ii], "code": P["codes"][jj],
        "close": P["close"][ii, jj],
        "dif": dif[ii, jj], "dea": dea[ii, jj], "hist": hist[ii, jj],
        "dif_pct": dif[ii, jj] / P["close"][ii, jj] * 100,
        "dea_pct": dea[ii, jj] / P["close"][ii, jj] * 100,
        "ret20": feats["ret20"][ii, jj],
        "ret60": feats["ret60"][ii, jj],
        "std20": feats["std20"][ii, jj],
        "dist_high": feats["dist_high"][ii, jj],
        "above_ma20": P["close"][ii, jj] > feats["ma20"][ii, jj],
        "above_ma60": P["close"][ii, jj] > feats["ma60"][ii, jj],
        "above_ma200": P["close"][ii, jj] > feats["ma200"][ii, jj],
        "ma200_slope20": feats["ma200"][ii, jj] / shift(feats["ma200"], 20)[ii, jj] - 1,
        "amount20": feats["amount20"][ii, jj],
        "turn20": feats["turn20"][ii, jj],
        "vol_ratio": P["amount"][ii, jj] / shift(feats["amount20"], 1)[ii, jj],
        "is_st": P["is_st"][ii, jj],
        "bars": P["bars"][ii, jj],
        "w_above": wf["w_above"][ii, jj],
        "w_gold4": wf["w_gold4"][ii, jj],
        "dif_slope3": (dif[ii, jj] - shift(dif, 3)[ii, jj]) / P["close"][ii, jj] * 100,
    })
    # 零轴状态
    e["zero_state"] = np.select(
        [(e.dif > 0) & (e.dea > 0), (e.dif > 0) & (e.dea <= 0),
         (e.dif <= 0) & (e.dea > 0), (e.dif <= 0) & (e.dea <= 0)],
        ["双上", "DIF上DEA下", "DIF下DEA上", "双下"], default="未知")
    # 信号密度与上次死叉
    gold_cs = np.cumsum(gold, axis=0)
    cnt60 = gold_cs - shift(gold_cs, 60, fill=0)
    e["cross60"] = cnt60[ii, jj].astype(float) - 1  # 不含当日
    tgrid = np.arange(P["shape"][0])[:, None] * np.ones((1, P["shape"][1]))
    last_death = np.maximum.accumulate(np.where(death, tgrid, -1), axis=0)
    e["green_len"] = ii - last_death[ii, jj]
    e["last_death"] = last_death[ii, jj]
    # 收益
    for h in HORIZONS:
        r = fwd[h][ii, jj]
        m = mkt_loo[h][ii, jj]
        valid = ok[h][ii, jj] & np.isfinite(r) & np.isfinite(m)
        e[f"fwd{h}"] = np.where(valid, r, np.nan)
        e[f"ex{h}"] = np.where(valid, r - m, np.nan)
        e[f"ok{h}"] = valid
    # 市场状态
    for col in ["breadth20", "breadth200", "disp20", "CSI300_ret20", "CSI300_vol20",
                "CSI300_dd250", "CSI300_ma200", "ew_ret20", "ew_vol20", "ew_dd250", "ew_ma200"]:
        e[col] = mf[col].reindex(e.date).values
    e["index_above200"] = (mf["CSI300"].reindex(e.date).values >
                           mf["CSI300_ma200"].reindex(e.date).values)
    e["ew_above200"] = (mf["ew_idx"].reindex(e.date).values >
                        mf["ew_ma200"].reindex(e.date).values)
    e["year"] = e.date.dt.year
    e["board"] = np.select(
        [e.code.str.startswith(("300", "301", "302")), e.code.str.startswith(("688", "689"))],
        ["创业板", "科创板"], default="主板")
    if path:
        for k, v in path.items():
            e[k] = v[ii, jj]
    e["name"] = name
    return e

