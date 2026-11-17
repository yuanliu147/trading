#!/usr/bin/env python3
"""17 结构条件：MACD 在哪种趋势/结构下有效

两层检验：
A. 连续特征的结构内排序能力：在每个结构桶内，计算 MACD 特征对 20 日超额收益的
   日度 Spearman IC、Q5-Q1 多空。回答“这种结构下，MACD 能不能选股”。
B. 二元状态的增量：在“长期趋势 × 中期排列”九个格子里，比较 DIF>DEA、柱体>0、
   柱体上行等状态的收益差；并用 20/60 日涨幅分位做分层残差，回答“MACD 是否
   只是过去涨幅的复写”。

输出：03_结果/特征_结构.csv、特征_结构_基准.csv、特征_状态.csv
"""
import os
import sys
import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
from stats_utils import nw_t, _bh

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")

DIMS = {
    "长期趋势": ("trend", ["上行", "下行", "过渡"]),
    "中期排列": ("align", ["多头排列", "空头排列", "混合"]),
    "250日位置": ("pos", ["高位", "中位", "深位"]),
    "短期结构": ("struct20", ["高高低低", "低高低低", "收敛/其他"]),
    "波动": ("volq", ["低波", "中波", "高波"]),
    "换手": ("turnq", ["低换手", "中换手", "高换手"]),
    "市场状态": ("market", ["指数上行", "指数下行", "过渡"]),
}

KEY_FEATS = ["柱体水平", "柱体3日变化", "20日柱体均值", "DIF水平",
             "DIF-DEA距离", "DIF5日斜率", "交叉后天数", "20日红柱占比",
             "基准_20日涨幅", "基准_距MA20"]


def daily_ic(x, ex, mask, dates, nq=5, min_n=100):
    ics, spreads, q5s, ns, ds = [], [], [], [], []
    for t in range(len(dates)):
        m = mask[t] & np.isfinite(x[t]) & np.isfinite(ex[t])
        idx = np.where(m)[0]
        if len(idx) < min_n:
            continue
        xv, yv = x[t, idx], ex[t, idx]
        xr, yr = rankdata(xv), rankdata(yv)
        q = np.minimum(((xr - 1) * nq / len(idx)).astype(np.int16), nq - 1)
        cnt = np.bincount(q, minlength=nq)
        if (cnt == 0).any():
            continue
        ics.append(np.corrcoef(xr, yr)[0, 1])
        qm = np.bincount(q, weights=yv, minlength=nq) / cnt
        spreads.append(qm[-1] - qm[0])
        q5s.append(qm[-1])
        ns.append(len(idx))
        ds.append(dates[t])
    return (np.asarray(ics), np.asarray(spreads), np.asarray(q5s),
            np.asarray(ns), pd.DatetimeIndex(ds))


def summ(ics, spreads, q5s, ns, ds, h=20):
    if len(ics) < 20:
        return None
    s_ic = pd.Series(ics, index=ds)
    s_sp = pd.Series(spreads, index=ds)
    early = ds < pd.Timestamp("2018-01-01")
    return dict(
        交易日数=len(ics), 股票日=int(ns.sum()),
        平均IC=ics.mean(), IC_t=nw_t(s_ic, lag=max(20, h)),
        IC大于0占比=(ics > 0).mean(),
        多空5_1=spreads.mean(), 多空t=nw_t(s_sp, lag=max(20, h)),
        Q5均值=q5s.mean(),
        IC_2010_2017=s_ic[early].mean(), IC_2018_2026=s_ic[~early].mean(),
        多空_2010_2017=s_sp[early].mean(), 多空_2018_2026=s_sp[~early].mean(),
        多空大于0年份=int((s_sp.groupby(s_sp.index.year).mean() > 0).sum()),
        年份数=int(s_sp.groupby(s_sp.index.year).mean().notna().sum()),
    )


def residualize(ex, mask, cell, q20, q60, ncell, nq=5):
    """按 日期 × 结构格 × 20日涨幅分位 × 60日涨幅分位 去均值。"""
    n = ex.shape[0]
    key = (np.arange(n, dtype=np.int64)[:, None] * ncell * nq * nq
           + np.maximum(cell, 0) * nq * nq
           + np.maximum(q20, 0) * nq + np.maximum(q60, 0))
    valid = mask & np.isfinite(ex) & (cell >= 0) & (q20 >= 0) & (q60 >= 0)
    k = key[valid]
    y = ex[valid].astype(np.float64)
    size = n * ncell * nq * nq
    s = np.bincount(k, weights=y, minlength=size)
    c = np.bincount(k, minlength=size)
    mean = np.divide(s, c, out=np.zeros_like(s), where=c > 0)
    out = np.full(ex.shape, np.nan, dtype=np.float32)
    out[valid] = (y - mean[k]).astype(np.float32)
    return out


def state_table(states, ex, resid, mask, cell, dates, ncell=9, min_n=80):
    rows = []
    names = list(states)
    for c in range(ncell):
        rec = {name: [] for name in names}
        for t in range(len(dates)):
            m = mask[t] & (cell[t] == c) & np.isfinite(ex[t])
            idx = np.where(m)[0]
            if len(idx) < min_n:
                continue
            y = ex[t, idx]
            ra = resid[t, idx]
            for name in names:
                s1 = states[name][t, idx]
                n1, n0 = int(s1.sum()), int((~s1).sum())
                if n1 < 20 or n0 < 20:
                    continue
                rec[name].append((dates[t], y[s1].mean() - y[~s1].mean(),
                                  ra[s1].mean() - ra[~s1].mean(),
                                  y.mean(), y[s1].mean(), y[~s1].mean(),
                                  n1 / (n1 + n0)))
        for name in names:
            x = rec[name]
            if len(x) < 20:
                continue
            d = pd.DataFrame(x, columns=["date", "raw", "adj", "base", "m1", "m0", "n1share"])
            early = d.date < pd.Timestamp("2018-01-01")
            rows.append(dict(
                状态=name, 格子=c, 交易日数=len(d),
                原始差=d.raw.mean(), 原始t=nw_t(d.raw, lag=20),
                控制涨幅后差=d.adj.mean(), 控制t=nw_t(d.adj, lag=20),
                格子基准=d.base.mean(), 状态1均值=d.m1.mean(), 状态0均值=d.m0.mean(),
                状态1占比=d.n1share.mean(),
                原始差_2010_2017=d.raw[early].mean(), 原始差_2018_2026=d.raw[~early].mean(),
                控制差_2010_2017=d.adj[early].mean(), 控制差_2018_2026=d.adj[~early].mean(),
                差大于0年份=int((d.groupby(d.date.dt.year).raw.mean() > 0).sum()),
                年份数=int(d.groupby(d.date.dt.year).raw.mean().notna().sum()),
            ))
    return rows


def main():
    P = M.load_panel()
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    feats = dict(macd["feats"])
    feats["基准_20日涨幅"] = st["ret20"]
    feats["基准_距MA20"] = st["dist_ma20"]

    volq = F.row_tercile(st["std20"], base)
    turnq = F.row_tercile(st["turn20"], base)
    st["volq"] = volq
    st["turnq"] = turnq

    fwd, ok, mkt, loo = M.forward_returns(P, (20,))
    ex = fwd[20] - loo[20]
    mask = base & ok[20] & np.isfinite(ex)

    rows, base_rows = [], []
    for dim, (key, labels) in DIMS.items():
        code = st[key]
        for k, lab in enumerate(labels):
            m = mask & (code == k)
            cnt = int(m.sum())
            base_rows.append(dict(维度=dim, 分组=lab, 股票日=cnt,
                                  平均超额=ex[m].mean() if cnt else np.nan))
            for fname in KEY_FEATS:
                ics, spreads, q5s, ns, ds = daily_ic(feats[fname], ex, m, P["dates"])
                r = summ(ics, spreads, q5s, ns, ds)
                if r is None:
                    continue
                r.update(维度=dim, 分组=lab, 特征=fname, 周期=20)
                rows.append(r)
        print("完成", dim)
    out = pd.DataFrame(rows)
    # BH-FDR：同一维度×特征内
    out["q_bh"] = np.nan
    for (dim, f), idx in out.groupby(["维度", "特征"]).groups.items():
        p = out.loc[idx, "IC_t"].apply(
            lambda t: 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) if np.isfinite(t) else np.nan)
        out.loc[idx, "q_bh"] = _bh(p.values)
    out.to_csv(os.path.join(RES, "特征_结构.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(base_rows).to_csv(os.path.join(RES, "特征_结构_基准.csv"), index=False, encoding="utf-8-sig")

    # 结构矩阵：长期趋势 × 中期排列
    cell = (st["trend"].astype(np.int16) * 3 + st["align"].astype(np.int16)).astype(np.int8)
    cell[(st["trend"] < 0) | (st["align"] < 0)] = -1
    q20 = F.row_quantile(st["ret20"], mask, 5)
    q60 = F.row_quantile(st["ret60"], mask, 5)
    resid = residualize(ex, mask, cell, q20, q60, 9)
    states = dict(macd["states"])
    st_rows = state_table(states, ex, resid, mask, cell, P["dates"])
    sdf = pd.DataFrame(st_rows)
    sdf["格子名称"] = sdf["格子"].map(lambda c: f"{DIMS['长期趋势'][1][c//3]}/{DIMS['中期排列'][1][c%3]}")
    sdf.to_csv(os.path.join(RES, "特征_状态.csv"), index=False, encoding="utf-8-sig")

    print("\n=== 结构内 IC（节选，20日）===")
    show = out[out.特征.isin(["柱体水平", "DIF水平", "柱体3日变化", "20日柱体均值"])]
    print(show.pivot_table(index=["维度", "分组"], columns="特征", values="平均IC").mul(100).round(2).to_string())
    print("\n=== 状态差（控制涨幅后，节选）===")
    print(sdf[["状态", "格子名称", "原始差", "控制涨幅后差", "控制t"]].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
