#!/usr/bin/env python3
"""07 多重检验：72 组参数的最大统计量 bootstrap（White Reality Check 近似）

对每组参数构造"日度横截面均值超额"序列；按 20 日块重采样日期，
对每组中心化后计算 t*，取 72 组最大值，得到零分布；
观测最大值与之比较得到 family-wise p。
"""
from __future__ import annotations

import itertools
import os

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(HERE, "02_代码", "work.duckdb")
OUT = os.path.join(HERE, "02_代码")


def nw_t(x, lag=5):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    mu = x.mean(); e = x - mu
    s = e @ e / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return mu / np.sqrt(s / n) if s > 0 else np.nan


def main() -> None:
    con = duckdb.connect(DB)
    con.execute("PRAGMA threads=8")
    grid = list(itertools.product([30, 60, 90], [0.5, 0.7, 1.0], [0.8, 1.0], [1.5, 2.0], [0.0, 0.85]))
    series = {}
    for N, q, v, m, p in grid:
        df = con.execute(f"""
            SELECT date, avg(ex60) AS ex
            FROM ev
            WHERE box_prev >= {N} AND comp_prev <= {q} AND vshr_prev <= {v}
              AND close > hh20_prev AND vr >= {m} AND pos250 >= {p}
              AND date >= DATE '2016-01-01'
            GROUP BY 1 ORDER BY 1
        """).df()
        series[(N, q, v, m, p)] = df.set_index("date")["ex"]

    mat = pd.DataFrame(series)
    dates = mat.index.to_numpy()
    obs_t = {k: nw_t(v.dropna().to_numpy()) for k, v in mat.items()}
    obs_max = np.nanmax(list(obs_t.values()))
    print("obs max t", round(obs_max, 3), "at", max(obs_t, key=lambda k: obs_t[k] if not np.isnan(obs_t[k]) else -9))

    # 中心化：每组减去自身均值
    centered = mat - mat.mean()
    rng = np.random.default_rng(7)
    block = 20
    nb = int(np.ceil(len(dates) / block))
    max_ts = []
    for b in range(500):
        starts = rng.integers(0, len(dates), nb)
        idx = np.concatenate([np.arange(s, min(s + block, len(dates))) for s in starts])
        sub = centered.iloc[idx]
        ts = []
        for k in sub.columns:
            x = sub[k].dropna().to_numpy()
            if len(x) < 30:
                ts.append(np.nan); continue
            se = x.std(ddof=1) / np.sqrt(len(x))
            ts.append(x.mean() / se if se > 0 else np.nan)
        max_ts.append(np.nanmax(ts))
    max_ts = np.array(max_ts)
    pval = float((max_ts >= obs_max).mean())
    print("bootstrap max t quantiles", np.round(np.quantile(max_ts, [0.5, 0.9, 0.95, 0.99]), 3))
    print("p", pval)

    res = pd.DataFrame([
        {"项目": "观测最大t（72组，样本外）", "值": round(obs_max, 3)},
        {"项目": "bootstrap 95%分位（零分布最大t）", "值": round(float(np.quantile(max_ts, 0.95)), 3)},
        {"项目": "family-wise p", "值": round(pval, 4)},
        {"项目": "组合数", "值": len(grid)},
        {"项目": "重采样次数", "值": 500},
    ])
    res.to_csv(os.path.join(OUT, "h3_bootstrap.csv"), index=False)
    print(res.to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
