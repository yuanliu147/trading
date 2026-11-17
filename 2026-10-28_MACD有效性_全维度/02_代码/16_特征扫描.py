#!/usr/bin/env python3
"""16 全样本特征扫描：MACD 不是只有金叉

对每个连续 MACD 特征，在 2010-2026 全 A（剔除 ST、上市不足 120 日）上计算：
- 日度 Spearman IC（特征 vs 20 日超额收益）
- 五分组 Q5-Q1 多空、Q5 相对市场超额
- 2010-2017 / 2018-2026 分段、逐年 IC
- 加入 20/60 日涨幅、距 MA20/MA200 作为“简单价格基准”，判断 MACD 是否只是价格动量的复写

输出：03_结果/特征_全样本.csv、特征_年度IC.csv、特征_分位均值.csv
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
HORIZONS = (5, 20, 60)


def daily_stats(x, ex, mask, dates, nq=5, min_n=200):
    ics, spreads, tops, qms, ns, ds = [], [], [], [], [], []
    for t in range(len(dates)):
        m = mask[t] & np.isfinite(x[t]) & np.isfinite(ex[t])
        idx = np.where(m)[0]
        if len(idx) < min_n:
            continue
        xv = x[t, idx]
        yv = ex[t, idx]
        xr = rankdata(xv)
        yr = rankdata(yv)
        q = np.minimum(((xr - 1) * nq / len(idx)).astype(np.int16), nq - 1)
        cnt = np.bincount(q, minlength=nq)
        if (cnt == 0).any():
            continue
        ic = np.corrcoef(xr, yr)[0, 1]
        qm = np.bincount(q, weights=yv, minlength=nq) / cnt
        ics.append(ic)
        spreads.append(qm[-1] - qm[0])
        tops.append(qm[-1] - yv.mean())
        qms.append(qm)
        ns.append(len(idx))
        ds.append(dates[t])
    return (np.asarray(ics), np.asarray(spreads), np.asarray(tops),
            np.asarray(qms), np.asarray(ns), pd.DatetimeIndex(ds))


def summarize(name, h, ics, spreads, tops, qms, ns, ds):
    if len(ics) < 20:
        return None
    s_ic = pd.Series(ics, index=ds)
    s_sp = pd.Series(spreads, index=ds)
    s_top = pd.Series(tops, index=ds)
    early = ds < pd.Timestamp("2018-01-01")
    qm = qms.mean(axis=0)
    mono = np.corrcoef(np.arange(len(qm)), qm)[0, 1]
    row = dict(
        特征=name, 周期=h, 股票日=int(ns.sum()), 交易日数=len(ics),
        平均IC=ics.mean(), IC_t=nw_t(s_ic, lag=max(20, h)), IC大于0占比=(ics > 0).mean(),
        IC_IR=ics.mean() / ics.std() if ics.std() > 0 else np.nan,
        多空5_1=spreads.mean(), 多空t=nw_t(s_sp, lag=max(20, h)),
        多头超额=tops.mean(), 多头t=nw_t(s_top, lag=max(20, h)),
        Q1=qm[0], Q2=qm[1], Q3=qm[2], Q4=qm[3], Q5=qm[4], 单调性=mono,
        IC_2010_2017=s_ic[early].mean(), IC_2018_2026=s_ic[~early].mean(),
        多空_2010_2017=s_sp[early].mean(), 多空_2018_2026=s_sp[~early].mean(),
        多空大于0年份=int((s_sp.groupby(s_sp.index.year).mean() > 0).sum()),
        年份数=int(s_sp.groupby(s_sp.index.year).mean().notna().sum()),
    )
    return row, s_ic


def main():
    P = M.load_panel()
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)

    feats = dict(macd["feats"])
    feats["基准_20日涨幅"] = st["ret20"]
    feats["基准_60日涨幅"] = st["ret60"]
    feats["基准_距MA20"] = st["dist_ma20"]
    feats["基准_距MA200"] = st["dist_ma200"]
    feats["基准_MA20-MA60"] = (st["ma20"] / st["ma60"] - 1).astype(np.float32)

    fwd, ok, mkt, loo = M.forward_returns(P, HORIZONS)
    rows, yearly = [], []
    for h in HORIZONS:
        ex = fwd[h] - loo[h]
        mask = base & ok[h] & np.isfinite(ex)
        print(f"\n=== {h} 日 ===")
        for name, x in feats.items():
            ics, spreads, tops, qms, ns, ds = daily_stats(x, ex, mask, P["dates"])
            out = summarize(name, h, ics, spreads, tops, qms, ns, ds)
            if out is None:
                continue
            row, s_ic = out
            rows.append(row)
            for yr, v in s_ic.groupby(s_ic.index.year).mean().items():
                yearly.append(dict(特征=name, 周期=h, 年份=int(yr), IC=v))
            print(f"{name:14s} IC={row['平均IC']*100:+.3f}% t={row['IC_t']:+.1f} "
                  f"多空={row['多空5_1']*100:+.3f}% t={row['多空t']:+.1f} "
                  f"Q5={row['Q5']*100:+.3f}%")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "特征_全样本.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(os.path.join(RES, "特征_年度IC.csv"), index=False, encoding="utf-8-sig")
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
