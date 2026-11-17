#!/usr/bin/env python3
"""统计工具：按信号日聚类的 t 值、分组表、BH-FDR。"""
import numpy as np
import pandas as pd


def nw_t(x, lag=5):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 10:
        return np.nan
    mu = x.mean()
    e = x - mu
    g0 = np.dot(e, e) / n
    s = g0
    for l in range(1, min(lag, n - 1) + 1):
        g = np.dot(e[l:], e[:-l]) / n
        s += 2 * (1 - l / (lag + 1)) * g
    se = np.sqrt(max(s, 0) / n)
    return mu / se if se > 0 else np.nan


def daily_mean(e, col, date_col="date"):
    d = e[[date_col, col]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    return d.groupby(date_col)[col].mean()


def stat_block(e, col, date_col="date"):
    """一个信号子集的统计。t 值按信号日聚类（日度均值序列 Newey-West）。"""
    d = e[[date_col, col]].dropna()
    n = len(d)
    if n == 0:
        return dict(n=0, n_days=0, mean_ex=np.nan, median_ex=np.nan,
                    hit=np.nan, t_nw=np.nan, mean_fwd=np.nan,
                    years_pos=0, years_n=0)
    dm = d.groupby(date_col)[col].mean()
    fwd_col = col.replace("ex", "fwd")
    return dict(
        n=n, n_days=len(dm),
        mean_ex=d[col].mean(), mean_daily=dm.mean(), median_ex=d[col].median(),
        hit=(d[col] > 0).mean(), hit_daily=(dm > 0).mean(), t_nw=nw_t(dm),
        mean_fwd=e[fwd_col].dropna().mean() if fwd_col in e else np.nan,
        years_pos=int((e.groupby(e[date_col].dt.year)[col].mean() > 0).sum()),
        years_n=int(e.groupby(e[date_col].dt.year)[col].mean().notna().sum()),
    )


def bucket_table(e, by, horizons=(5, 10, 20, 60), min_n=100, date_col="date"):
    """按 by 分组、每个 horizon 输出统计。by 可以是列名或列名列表。"""
    rows = []
    keys = [by] if isinstance(by, str) else list(by)
    for h in horizons:
        col = f"ex{h}"
        if col not in e:
            continue
        sub = e.dropna(subset=[col]).copy()
        groups = []
        if not keys:
            groups = [("全样本", sub)]
        else:
            for k, g in sub.groupby(keys, dropna=False):
                if not isinstance(k, tuple):
                    k = (k,)
                groups.append((" / ".join(str(x) for x in k), g))
        for label, g in groups:
            if len(g) < min_n:
                continue
            st = stat_block(g, col, date_col)
            st.update(维度=" / ".join(keys) if keys else "全样本", 分组=label, 周期=h)
            rows.append(st)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # BH-FDR：同一维度×周期内
    out["q_bh"] = np.nan
    for (dim, h), idx in out.groupby(["维度", "周期"]).groups.items():
        p = out.loc[idx, "t_nw"].apply(lambda t: 2 * (1 - _norm_cdf(abs(t))) if np.isfinite(t) else np.nan)
        out.loc[idx, "q_bh"] = _bh(p.values)
    return out


def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def _bh(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty(m)
    qv[order] = np.minimum(ranked, 1.0)
    q[ok] = qv
    return q
