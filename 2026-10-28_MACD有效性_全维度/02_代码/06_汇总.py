#!/usr/bin/env python3
"""06 汇总：把事件、参数、指数、变体、离场整理成最终中文表"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
from stats_utils import stat_block

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")


def cluster_t(e, dates):
    d = pd.DataFrame({"e": e, "d": dates}).dropna()
    if len(d) < 50:
        return np.nan
    g = d.groupby("d").e.sum().to_numpy()
    se = np.sqrt(np.sum(g ** 2)) / len(d)
    return d.e.mean() / se if se > 0 else np.nan


def overall(e):
    rows = []
    for h in M.HORIZONS:
        c = f"ex{h}"
        x = e[c].dropna()
        dm = e.dropna(subset=[c]).groupby("date")[c].mean()
        rows.append(dict(
            周期=h, 信号数=len(x), 信号日数=len(dm),
            平均超额=x.mean(), 中位超额=x.median(),
            胜率_超额大于0=(x > 0).mean(),
            日度平均超额=dm.mean(),
            日度t=dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 5 else np.nan,
            聚类t=cluster_t(e[c], e["date"]),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "最终_总体.csv"), index=False, encoding="utf-8-sig")
    return out


def yearly(e):
    rows = []
    for y, g in e.groupby("year"):
        x = g["ex20"].dropna()
        if len(x) < 100:
            continue
        rows.append(dict(年份=y, 信号数=len(x), 平均分位=g.pct20.mean(),
                         胜率_分位大于50=(g.pct20 > 0.5).mean(),
                         平均超额=x.mean(), 中位超额=x.median(),
                         日度平均超额=x.groupby(g.loc[x.index, "date"]).mean().mean()))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "最终_年度.csv"), index=False, encoding="utf-8-sig")
    return out


def conditions(e):
    dims = ["信号密度", "市场趋势", "宽度20", "指数回撤", "指数波动", "个股趋势",
            "距250日高", "波动分组", "成交额分组", "股价年龄", "60日金叉次数",
            "绿柱长度", "量比", "DIF位置", "周线状态", "zero_state", "board", "is_st"]
    rows = []
    for dim in dims:
        for key, g in e.dropna(subset=["pct20", "ex20"]).groupby(dim, observed=True):
            if len(g) < 200:
                continue
            x = g.ex20
            dm = g.groupby("date").ex20.mean()
            early = g[g.date < "2018-01-01"]
            late = g[g.date >= "2018-01-01"]
            rows.append(dict(
                维度=dim, 分组=str(key), 信号数=len(g), 信号日数=g.date.nunique(),
                平均分位20=g.pct20.mean(), 分位大于50=(g.pct20 > 0.5).mean(),
                平均超额20=x.mean(), 中位超额20=x.median(),
                日度平均超额20=dm.mean(),
                日度t=dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 5 else np.nan,
                聚类t=cluster_t(x, g.date),
                分位_2010_2017=early.pct20.mean(), 分位_2018_2026=late.pct20.mean(),
                分位差_后期减前期=late.pct20.mean() - early.pct20.mean(),
            ))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "最终_条件.csv"), index=False, encoding="utf-8-sig")
    return out


def pct_add(e, P, fwd, ok):
    """给事件表补充 pct20（若不存在）。"""
    if "pct20" in e:
        return e
    f = fwd[20]; v = ok[20] & np.isfinite(f)
    pct = np.full(P["shape"], np.nan, dtype=np.float32)
    for t in range(P["shape"][0]):
        idx = np.where(v[t])[0]
        if len(idx) < 50:
            continue
        x = f[t, idx]
        order = np.argsort(np.argsort(x))
        pct[t, idx] = (order + 0.5) / len(idx)
    ii, jj = np.where(M.crosses(*M.macd(P)[:2], P)[0])
    keep = P["dates"][ii] >= pd.Timestamp("2010-01-01")
    e["pct20"] = pct[ii[keep], jj[keep]]
    return e


def grid_summary():
    g_all = pd.read_csv(os.path.join(RES, "参数网格.csv"))
    n_all = len(g_all)
    g = g_all[np.isfinite(g_all.t20)].copy()
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(g.t20)))
    order = np.argsort(p)
    m = len(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m); q[order] = np.minimum(ranked, 1)
    g["q_bh"] = q
    g.to_csv(os.path.join(RES, "参数网格_含q.csv"), index=False, encoding="utf-8-sig")
    dflt = float(g[(g.fast == 12) & (g.slow == 26) & (g.signal == 9)].pct20.iloc[0])
    summary = pd.DataFrame([
        ("参数组合数", f"{n_all}（可评估 {len(g)}；signal=1 退化，无信号）"),
        ("平均分位20最小值", f"{g.pct20.min()*100:.2f}%"),
        ("平均分位20中位数", f"{g.pct20.median()*100:.2f}%"),
        ("平均分位20最大值", f"{g.pct20.max()*100:.2f}%"),
        ("分位>50.5%的组合数", f"{int((g.pct20 > 0.505).sum())}"),
        ("中位超额>0的组合数", f"{int((g.median20 > 0).sum())}"),
        ("胜率>50%的组合数", f"{int((g.hit20 > 0.5).sum())}"),
        ("最小FDR q值", f"{g.q_bh.min():.3f}"),
        ("q<0.10的组合数", f"{int((g.q_bh < 0.1).sum())}"),
        ("12/26/9 平均分位", f"{dflt*100:.2f}%（最差 {(g.pct20 < dflt).mean()*100:.2f}%）"),
    ], columns=["项目", "数值"])
    summary.to_csv(os.path.join(RES, "最终_参数摘要.csv"), index=False, encoding="utf-8-sig")
    return g, summary


def exit_stats(P):
    dif, dea, hist = M.macd(P)
    gold, death = M.crosses(dif, dea, P)
    fwd, ok, mkt, loo = M.forward_returns(P)
    n, d = P["shape"]
    nd = np.full((n, d), n, dtype=np.int32)
    ii, jj = np.where(death); nd[ii, jj] = ii
    nd = np.minimum.accumulate(nd[::-1], axis=0)[::-1]
    ii, jj = np.where(gold)
    keep = (P["dates"][ii] >= pd.Timestamp("2010-01-01")) & (ii < n - 2)
    ii, jj = ii[keep], jj[keep]
    ndx = nd[ii + 1, jj]
    has = ndx < n - 1
    exit_idx = np.where(has, ndx + 1, n - 1)
    exit_px = np.where(has, P["open"][exit_idx, jj], P["close"][n - 1, jj])
    ret = exit_px / P["open"][ii + 1, jj] - 1
    hold = exit_idx - (ii + 1)
    rows = [dict(离场="死叉离场", 交易数=len(ret), 平均收益=np.nanmean(ret),
                 中位收益=np.nanmedian(ret), 胜率=np.nanmean(ret > 0),
                 平均持有天数=np.nanmean(hold),
                 死叉覆盖率=has.mean())]
    for h in (5, 10, 20, 60):
        r = fwd[h][ii, jj]
        rows.append(dict(离场=f"固定{h}日", 交易数=int(np.isfinite(r).sum()),
                         平均收益=np.nanmean(r), 中位收益=np.nanmedian(r),
                         胜率=np.nanmean(r > 0), 平均持有天数=h, 死叉覆盖率=np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "最终_离场.csv"), index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    P = M.load_panel()
    e = pd.read_parquet(os.path.join(RES, "事件_金叉_12269.parquet"))
    e["date"] = pd.to_datetime(e["date"])
    if "pct20" not in e:
        fwd, ok, _, _ = M.forward_returns(P)
        e = pct_add(e, P, fwd, ok)
    print(overall(e).to_string(index=False))
    print(yearly(e).to_string(index=False))
    c = conditions(e)
    print(c.sort_values("平均分位20", ascending=False).head(15).to_string(index=False))
    grid_summary()
    print(exit_stats(P).to_string(index=False))
