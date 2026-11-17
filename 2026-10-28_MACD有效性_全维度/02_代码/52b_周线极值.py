#!/usr/bin/env python3
"""52b 周线 KDJ 极值 × 日线 MACD 金叉：逐年、5/20/60、周线 KDJ 参数稳健。"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
import kdj_core as K
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")


def dm_series(sel, ex, min_n=30):
    n = sel.sum(axis=1)
    s = np.nansum(np.where(sel, ex, 0.0), axis=1)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), n


def stats(sel, ex, dates):
    n = int(sel.sum())
    dm, _ = dm_series(sel, ex)
    ok = np.isfinite(dm)
    if ok.sum() < 20:
        return dict(n=n, days=int(ok.sum()), mean=np.nan, t=np.nan,
                    early=np.nan, late=np.nan)
    early = ok & (dates <= pd.Timestamp("2017-12-31"))
    late = ok & (dates >= pd.Timestamp("2018-01-01"))
    return dict(n=n, days=int(ok.sum()), mean=float(np.nanmean(dm)),
                t=float(nw_t(dm, 20)),
                early=float(np.nanmean(dm[early])) if early.any() else np.nan,
                late=float(np.nanmean(dm[late])) if late.any() else np.nan)


def main():
    t0 = time.time()
    P = M.load_panel()
    dates = P["dates"]
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    base = F.base_mask(P)
    gold, _ = M.crosses(macd["dif"], macd["dea"], P)
    fwd, ok, mkt, loo = M.forward_returns(P, (5, 20, 60))
    exs = {h: (fwd[h] - loo[h]).astype(np.float32) for h in (5, 20, 60)}
    masks = {h: base & ok[h] & np.isfinite(exs[h]) for h in (5, 20, 60)}
    print("base done", time.time() - t0)

    # 周线 KDJ 参数稳健 + 逐年
    rows = []
    yr_rows = []
    for n in (5, 9, 14, 19, 26):
        wk = K.period_kdj_state(P, "W", n=n, m1=3, m2=3)
        print("W KDJ", n, time.time() - t0)
        for h in (5, 20, 60):
            g = gold & masks[h]
            for lab, cond in [("周K<20", (wk["k"] < 20) & wk["ok"]),
                              ("周K<30", (wk["k"] < 30) & wk["ok"]),
                              ("周K>80", (wk["k"] > 80) & wk["ok"]),
                              ("周K>70", (wk["k"] > 70) & wk["ok"]),
                              ("周KDJ多", wk["state"].astype(bool)),
                              ("周KDJ空", (~wk["state"].astype(bool)) & wk["ok"])]:
                s = g & cond
                r = stats(s, exs[h], dates)
                rows.append(dict(KDJ参数=n, 周期=h, 过滤=lab, **r))
            # 增量
            for lab, cond in [("周K<20", (wk["k"] < 20) & wk["ok"]),
                              ("周K>80", (wk["k"] > 80) & wk["ok"]),
                              ("周KDJ多", wk["state"].astype(bool))]:
                a = g & cond
                b = g & wk["ok"] & ~cond
                da, _ = dm_series(a, exs[h])
                db, _ = dm_series(b, exs[h])
                common = np.isfinite(da) & np.isfinite(db)
                if common.sum() < 20:
                    continue
                d = da - db
                rows.append(dict(KDJ参数=n, 周期=h, 过滤=lab + "增量", n=int(a.sum()),
                                 days=int(common.sum()), mean=float(np.nanmean(d[common])),
                                 t=float(nw_t(d[common], 20)),
                                 early=float(np.nanmean(d[common & (dates <= pd.Timestamp("2017-12-31"))])) if (common & (dates <= pd.Timestamp("2017-12-31"))).any() else np.nan,
                                 late=float(np.nanmean(d[common & (dates >= pd.Timestamp("2018-01-01"))])) if (common & (dates >= pd.Timestamp("2018-01-01"))).any() else np.nan))
        if n == 9:
            g = gold & masks[20]
            for lab, cond in [("周K<20", (wk["k"] < 20) & wk["ok"]),
                              ("周K>80", (wk["k"] > 80) & wk["ok"]),
                              ("周KDJ多", wk["state"].astype(bool)),
                              ("周KDJ空", (~wk["state"].astype(bool)) & wk["ok"])]:
                sub = g & cond
                dm, _ = dm_series(sub, exs[20])
                for y in range(2010, 2027):
                    sel = np.isfinite(dm) & (dates.year == y)
                    if sel.sum() < 20:
                        continue
                    yr_rows.append(dict(过滤=lab, 年份=y, 股票日=int((sub & (dates.year == y)[:, None]).sum()),
                                        交易日=int(sel.sum()), 日均超额=float(np.nanmean(dm[sel])),
                                        t=float(nw_t(dm[sel], 5))))
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(RES, "KDJ_周线极值_参数.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(yr_rows).to_csv(os.path.join(RES, "KDJ_周线极值_逐年.csv"), index=False, encoding="utf-8-sig")
    print("\n周线KDJ(9) 20日")
    print(g[(g.KDJ参数 == 9) & (g.周期 == 20)].to_string(index=False))
    print("\n周K<20 增量 参数稳定性")
    x = g[g.过滤 == "周K<20增量"]
    print(x.pivot_table(index="周期", columns="KDJ参数", values="mean").to_string())
    print(x.pivot_table(index="周期", columns="KDJ参数", values="t").to_string())
    print("\n逐年 周K<20")
    print(pd.DataFrame(yr_rows)[pd.DataFrame(yr_rows).过滤 == "周K<20"].to_string(index=False))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
