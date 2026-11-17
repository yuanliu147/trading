#!/usr/bin/env python3
"""个股适配性研究：公共引擎

与全市场事件研究的区别：
- 把“这只股票上的 MACD”当成独立对象，逐股计算事件超额和状态择时。
- 事件超额提供市场/行业/风格三种中性口径；风格格 = 规模五分位 × 波动五分位。
- 状态择时用 open→open 收益，信号 T 日收盘、T+1 开盘执行，避免隔夜前视。
- 成本 0.075%/边（往返 0.15%），按仓位变化计。
"""
import os
import warnings

import numpy as np
import pandas as pd

import macd_core as M
from stats_utils import nw_t

ROOT = "/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度"
RES = os.path.join(ROOT, "03_结果")
IND_CSV = "/Users/admin/WorkSpace/trading/2026-10-18_成交量融合_2010-2025/01_数据/申万行业分类_时点.csv"
COST_SIDE = 0.00075

EARLY = (pd.Timestamp("2010-01-01"), pd.Timestamp("2017-12-31"))
LATE = (pd.Timestamp("2018-01-01"), pd.Timestamp("2026-10-31"))


def load_base():
    P = M.load_panel()
    P["open_ff"] = M._fill_lead(P["open"])
    n = P["shape"][0]
    r = np.full(P["shape"], np.nan, dtype=np.float32)
    r[:-1] = P["open_ff"][1:] / P["open_ff"][:-1] - 1.0
    P["r_oo"] = r
    r1 = np.full(P["shape"], np.nan, dtype=np.float32)
    r1[1:] = P["close_ff"][1:] / P["close_ff"][:-1] - 1.0
    P["r_cc"] = r1
    P["ew"] = np.nanmean(np.where(P["valid"], r1, np.nan), axis=1).astype(np.float32)
    P["amount20"] = M.rolling_mean(P["amount"], 20, minp=10)
    P["universe"] = (P["valid"] & (P["bars"] >= 250) & ~P["is_st"].astype(bool) &
                     np.isfinite(P["raw_close"]) & (P["raw_close"] >= 2))
    P["universe_inv"] = P["universe"] & (P["amount20"] >= 5e6)
    return P


def build_states(P):
    dif, dea, hist = M.macd(P)
    c = P["close_ff"]
    ma20 = M.rolling_mean(c, 20)
    ma60 = M.rolling_mean(c, 60)
    ma200 = M.rolling_mean(c, 200)
    ret60 = (c / M.shift(c, 60) - 1).astype(np.float32)
    st = {
        "DIF>DEA": dif > dea,
        "柱体>0": hist > 0,
        "价格>MA20": c > ma20,
        "价格>MA60": c > ma60,
        "价格>MA200": c > ma200,
        "60日动量为正": ret60 > 0,
    }
    return st, dict(dif=dif, dea=dea, hist=hist, ma20=ma20, ma60=ma60,
                    ma200=ma200, ret60=ret60)


def position(P, state):
    """T 日收盘状态 -> T+1 开盘持仓；停牌日沿用上一仓位。"""
    pos = M.shift(state.astype(np.float32), 1)
    pos = np.where(P["valid"], pos, np.nan)
    pos = pd.DataFrame(pos).ffill().fillna(0.0).to_numpy(dtype=np.float32)
    return pos


def hac_t(y, x, lag=5):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 60 or np.std(x) < 1e-10:
        return np.nan, np.nan
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        A = Xe[l:].T @ Xe[:-l]
        S += w * (A + A.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(max(V[1, 1], 0.0))
    return float(beta[1]), float(beta[1] / se) if se > 0 else np.nan


def _ann(x, n_days):
    if n_days < 250:
        return np.nan
    path = np.cumprod(1.0 + x)
    if len(path) == 0 or path[-1] <= 0:
        return -1.0
    return float(path[-1] ** (252.0 / n_days) - 1.0)


def _mdd(x):
    path = np.concatenate([[1.0], np.cumprod(1.0 + x)])
    peak = np.maximum.accumulate(path)
    return float(np.min(path / peak - 1.0))


def _sharpe(x):
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(252)) if sd > 0 else np.nan


def timing_stats(P, pos, mkt):
    """逐股状态择时统计。返回 dict of arrays。"""
    n, nc = pos.shape
    keys = ["天数", "在场比例", "在场均值", "离场均值", "择时差", "择时t",
            "年化净", "年化持有", "年化同暴露", "净超同暴露", "净超持有",
            "年化毛", "成本拖累",
            "夏普净", "夏普持有", "夏普同暴露", "回撤净", "回撤持有",
            "回撤同暴露", "回撤改善", "回撤超同暴露",
            "年换手次数", "beta", "上涨日在场", "上涨日离场",
            "下跌日在场", "下跌日离场", "在场胜率", "离场胜率", "波动净"]
    out = {k: np.full(nc, np.nan, dtype=np.float64) for k in keys}
    for j in range(nc):
        idx = np.where(P["valid"][:, j])[0]
        if len(idx) < 300:
            continue
        a, b = int(idx[0]), int(idx[-1])
        r = P["r_oo"][a:b, j].astype(np.float64)
        p = pos[a:b, j].astype(np.float64)
        m = mkt[a:b].astype(np.float64)
        ok = np.isfinite(r) & np.isfinite(p)
        if ok.sum() < 250:
            continue
        r, p, m = r[ok], p[ok], m[ok]
        nd = len(r)
        d = np.abs(np.diff(p, prepend=0.0))
        cost = COST_SIDE * d
        net = p * r - cost
        expo = p.mean()
        bench = expo * r
        ann_net, ann_bh, ann_bm = _ann(net, nd), _ann(r, nd), _ann(bench, nd)
        inm = r[p > 0.5]
        outm = r[p < 0.5]
        beta, t = hac_t(r, p)
        up = m > 0
        dn = ~up
        out["天数"][j] = nd
        out["在场比例"][j] = expo
        out["在场均值"][j] = inm.mean() if len(inm) else np.nan
        out["离场均值"][j] = outm.mean() if len(outm) else np.nan
        out["择时差"][j] = (inm.mean() - outm.mean()) if len(inm) and len(outm) else np.nan
        out["择时t"][j] = t
        out["年化净"][j] = ann_net
        out["年化持有"][j] = ann_bh
        out["年化同暴露"][j] = ann_bm
        out["净超同暴露"][j] = ann_net - ann_bm
        out["净超持有"][j] = ann_net - ann_bh
        ann_gross = _ann(p * r, nd)
        out["年化毛"][j] = ann_gross
        out["成本拖累"][j] = ann_gross - ann_net
        out["夏普净"][j] = _sharpe(net)
        out["夏普持有"][j] = _sharpe(r)
        out["夏普同暴露"][j] = _sharpe(bench)
        out["回撤净"][j] = _mdd(net)
        out["回撤持有"][j] = _mdd(r)
        out["回撤同暴露"][j] = _mdd(bench)
        out["回撤改善"][j] = _mdd(net) - _mdd(r)
        out["回撤超同暴露"][j] = _mdd(net) - _mdd(bench)
        out["年换手次数"][j] = d.sum() / (nd / 252.0)
        vm = np.var(m)
        out["beta"][j] = np.cov(r, m)[0, 1] / vm if vm > 0 else np.nan
        out["上涨日在场"][j] = r[up & (p > 0.5)].mean() if (up & (p > 0.5)).any() else np.nan
        out["上涨日离场"][j] = r[up & (p < 0.5)].mean() if (up & (p < 0.5)).any() else np.nan
        out["下跌日在场"][j] = r[dn & (p > 0.5)].mean() if (dn & (p > 0.5)).any() else np.nan
        out["下跌日离场"][j] = r[dn & (p < 0.5)].mean() if (dn & (p < 0.5)).any() else np.nan
        out["在场胜率"][j] = (inm > 0).mean() if len(inm) else np.nan
        out["离场胜率"][j] = (outm > 0).mean() if len(outm) else np.nan
        out["波动净"][j] = np.std(net, ddof=1) * np.sqrt(252)
    return out


def industry_matrix(P):
    ind = pd.read_csv(IND_CSV)
    def _fmt(x):
        a, b = str(x).split(".")
        return b.zfill(6) + "." + a.upper()
    ind["plain"] = ind["code"].map(_fmt)
    ind["start_date"] = pd.to_datetime(ind["start_date"])
    names = sorted(ind["industry"].dropna().unique().tolist())
    name2id = {x: i for i, x in enumerate(names)}
    ind["iid"] = ind["industry"].map(name2id)
    mat = np.full(P["shape"], -1, dtype=np.int16)
    c2j = {c: j for j, c in enumerate(P["codes"])}
    d0 = P["dates"][0] - pd.Timedelta(days=1)
    d1 = P["dates"][-1] + pd.Timedelta(days=1)
    for code, g in ind.groupby("plain"):
        j = c2j.get(code)
        if j is None:
            continue
        g = g.sort_values("start_date")
        starts = g["start_date"].to_numpy()
        ids = g["iid"].to_numpy()
        for k in range(len(g)):
            s = max(pd.Timestamp(starts[k]), d0)
            e = min(pd.Timestamp(starts[k + 1]) if k + 1 < len(g) else d1, d1)
            lo = np.searchsorted(P["dates"].values, np.datetime64(s))
            hi = np.searchsorted(P["dates"].values, np.datetime64(e))
            if hi > lo:
                mat[lo:hi, j] = ids[k]
    return mat, names


def group_sums(fwd, ok, group, ngroup):
    n = fwd.shape[0]
    sums = np.zeros((n, ngroup), dtype=np.float64)
    cnts = np.zeros((n, ngroup), dtype=np.float64)
    for t in range(n):
        g = group[t]
        m = ok[t] & np.isfinite(fwd[t]) & (g >= 0)
        if m.any():
            sums[t] = np.bincount(g[m], weights=fwd[t][m].astype(np.float64), minlength=ngroup)
            cnts[t] = np.bincount(g[m], minlength=ngroup)
    return sums, cnts


def loo_excess(sums, cnts, group, fwd, ii, jj):
    g = group[ii, jj].astype(np.int64)
    f = fwd[ii, jj].astype(np.float64)
    s = sums[ii, g]
    c = cnts[ii, g]
    out = f - (s - f) / np.maximum(c - 1, 1)
    out[(g < 0) | (c <= 1) | ~np.isfinite(f)] = np.nan
    return out


def roll(x, w, how, minp=None):
    if minp is None:
        minp = max(2, w // 2)
    df = pd.DataFrame(x)
    r = df.rolling(w, min_periods=minp)
    if how == "mean":
        y = r.mean()
    elif how == "std":
        y = r.std()
    elif how == "var":
        y = r.var()
    elif how == "median":
        y = r.median()
    elif how == "max":
        y = r.max()
    elif how == "min":
        y = r.min()
    elif how == "skew":
        y = r.skew()
    elif how == "sum":
        y = r.sum()
    else:
        raise ValueError(how)
    return y.to_numpy(dtype=np.float32)


def period_masks(P):
    d = P["dates"]
    return {
        "全期": (d >= pd.Timestamp("2010-01-01")),
        "前期": (d >= EARLY[0]) & (d <= EARLY[1]),
        "后期": (d >= LATE[0]) & (d <= LATE[1]),
    }


def stock_event_table(ev, P, ex20_style, ex20_ind, ex20_own):
    """事件层增强表。"""
    ii = np.searchsorted(P["dates"].values, ev["date"].values.astype("datetime64[ns]"))
    c2j = {c: j for j, c in enumerate(P["codes"])}
    jj = ev["code"].map(c2j).to_numpy()
    ok = np.isfinite(ii) & np.isfinite(jj.astype(float)) & ev["ok20"].to_numpy(bool)
    ii, jj = ii[ok].astype(int), jj[ok].astype(int)
    e = ev.loc[ok].copy()
    e["ex20_style"] = ex20_style
    e["ex20_ind"] = ex20_ind
    e["ex20_own"] = ex20_own
    e["ii"] = ii
    e["jj"] = jj
    return e


def per_stock_event(e, label):
    rows = []
    for code, s in e.groupby("code", sort=False):
        x = s["ex20_style"].dropna()
        if len(x) < 3:
            continue
        rows.append(dict(
            code=code, 口径=label, 信号数=len(x),
            event_ex_mkt=s["ex20"].mean(), event_ex_style=x.mean(),
            event_ex_ind=s["ex20_ind"].mean(), event_ex_own=s["ex20_own"].mean(),
            event_median_style=x.median(), event_hit=(x > 0).mean(),
            event_std=x.std(), event_t=nw_t(x, lag=5),
            event_fwd20=s["fwd20"].mean(), event_mfe20=s["mfe20"].mean(),
            event_mae20=s["mae20"].mean(),
            event_first=s["date"].min(), event_last=s["date"].max(),
        ))
    return pd.DataFrame(rows)
