#!/usr/bin/env python3
"""51 参数稳健性：KDJ(5/9/14/19/26 × 平滑) 与 MACD(fast/slow/sig) 网格。

核心检验：确认型用法（KDJ多、同日双金叉、KDJ金叉在MACD金叉后）是否在参数变化下仍为负；
极端过滤（K>70、KQ1、KQ5）是否稳定。
"""
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
MIN_EV = 30


def dm_series(sel, ex, min_n=MIN_EV):
    n = sel.sum(axis=1)
    s = np.nansum(np.where(sel, ex, 0.0), axis=1)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), n


def ev(sel, ex, dates):
    dm, n = dm_series(sel, ex)
    ok = np.isfinite(dm)
    if ok.sum() < 20:
        return dict(A=np.nan, t=np.nan, days=int(ok.sum()), n=int(sel.sum()), early=np.nan, late=np.nan)
    return dict(A=float(np.nanmean(dm)), t=float(nw_t(dm, 20)), days=int(ok.sum()),
                n=int(sel.sum()),
                early=float(np.nanmean(dm[ok & (dates <= pd.Timestamp("2017-12-31"))])) if (ok & (dates <= pd.Timestamp("2017-12-31"))).any() else np.nan,
                late=float(np.nanmean(dm[ok & (dates >= pd.Timestamp("2018-01-01"))])) if (ok & (dates >= pd.Timestamp("2018-01-01"))).any() else np.nan)


def diffstat(a, b, ex):
    da, _ = dm_series(a, ex)
    db, _ = dm_series(b, ex)
    common = np.isfinite(da) & np.isfinite(db)
    if common.sum() < 20:
        return dict(A=np.nan, B=np.nan, 差值=np.nan, t=np.nan, 共同日=int(common.sum()))
    d = da - db
    return dict(A=float(np.nanmean(da[common])), B=float(np.nanmean(db[common])),
                差值=float(np.nanmean(d[common])), t=float(nw_t(d[common], 20)),
                共同日=int(common.sum()))


def recent(evmask, n):
    cs = np.cumsum(evmask, axis=0)
    return (cs - M.shift(cs, n, fill=0)) > 0


def metrics(macd_bull, gold, kdj_gold, k, d, qk, ex, base):
    res = {}
    # 事件过滤：MACD 金叉
    g = gold & base & np.isfinite(ex)
    res["MACD金叉_KDJ多-空"] = diffstat(g & (k > d), g & ~(k > d), ex)
    res["MACD金叉_同日KDJ金叉-非"] = diffstat(g & kdj_gold, g & ~kdj_gold, ex)
    res["MACD金叉_K>70-K<=70"] = diffstat(g & (k > 70), g & ~(k > 70), ex)
    res["MACD金叉_KQ1-KQ2_4"] = diffstat(g & (qk == 0), g & np.isin(qk, [1, 2, 3]), ex)
    res["MACD金叉_KQ5-KQ2_4"] = diffstat(g & (qk == 4), g & np.isin(qk, [1, 2, 3]), ex)
    # KDJ 金叉
    kg = kdj_gold & base & np.isfinite(ex)
    res["KDJ金叉_MACD多-空"] = diffstat(kg & macd_bull, kg & ~macd_bull, ex)
    res["KDJ金叉_前5日MACD金叉-无"] = diffstat(kg & recent(gold, 5), kg & ~recent(gold, 5), ex)
    # 状态交互 DiD
    mb = macd_bull & base & np.isfinite(ex)
    ms = ~macd_bull & base & np.isfinite(ex)
    for lab, cond in [("K>D", k > d), ("K>70", k > 70), ("KQ3", qk == 2), ("KQ1", qk == 0), ("KQ5", qk == 4)]:
        a, b = mb & cond, mb & ~cond
        c, dd = ms & cond, ms & ~cond
        sa, _ = dm_series(a, ex); sb, _ = dm_series(b, ex)
        sc, _ = dm_series(c, ex); sd, _ = dm_series(dd, ex)
        common = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(sc) & np.isfinite(sd)
        if common.sum() < 20:
            continue
        did = (sa - sb) - (sc - sd)
        res[f"状态DiD_{lab}"] = dict(A=float(np.nanmean((sa - sb)[common])),
                                    B=float(np.nanmean((sc - sd)[common])),
                                    差值=float(np.nanmean(did[common])),
                                    t=float(nw_t(did[common], 20)),
                                    共同日=int(common.sum()))
    return res


def main():
    t0 = time.time()
    P = M.load_panel()
    dates = P["dates"]
    mf = M.market_features(P)
    base = F.base_mask(P)
    st = F.build_structure(P, mf)
    fwd, ok, mkt, loo = M.forward_returns(P, (20,))
    ex = (fwd[20] - loo[20]).astype(np.float32)
    mask = base & ok[20] & np.isfinite(ex)

    rows = []
    # ---------- KDJ 参数 ----------
    print("KDJ grid")
    for n in (5, 9, 14, 19, 26):
        for m1, m2 in [(3, 3), (5, 3), (3, 5)]:
            kd = K.kdj_features(P, n, m1, m2)
            k, d = kd["k"], kd["d"]
            qk = F.row_quantile(k, base, 5)
            macd = F.build_macd_features(P)
            gold, _ = M.crosses(macd["dif"], macd["dea"], P)
            m = metrics(macd["states"]["DIF>DEA"], gold, kd["gold"], k, d, qk, ex, mask)
            for name, r in m.items():
                rows.append(dict(网格="KDJ", 参数=f"N{n}_M{m1}{m2}", 指标=name, **r))
            print("KDJ", n, m1, m2, "done", round(time.time() - t0, 1))
    # ---------- MACD 参数 ----------
    print("MACD grid")
    kd = K.kdj_features(P)
    k, d = kd["k"], kd["d"]
    qk = F.row_quantile(k, base, 5)
    for fast, slow, sig in [(8, 17, 9), (12, 26, 9), (16, 32, 9), (20, 40, 9), (6, 13, 5)]:
        dif, dea, hist = M.macd(P, fast, slow, sig)
        gold, death = M.crosses(dif, dea, P)
        mb = dif > dea
        m = metrics(mb, gold, kd["gold"], k, d, qk, ex, mask)
        for name, r in m.items():
            rows.append(dict(网格="MACD", 参数=f"{fast}_{slow}_{sig}", 指标=name, **r))
        print("MACD", fast, slow, sig, "done", round(time.time() - t0, 1))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "KDJ_参数稳健.csv"), index=False, encoding="utf-8-sig")
    print("saved", out.shape)
    # 汇总：每个指标在参数网格中的符号稳定性
    print("\n参数稳健汇总")
    for (grid, ind), g in out.groupby(["网格", "指标"], sort=False):
        x = g.dropna(subset=["差值"])
        if len(x) == 0:
            continue
        print(f"{grid:4s} {ind:26s} n={len(x):2d} 负={int((x.差值<0).sum()):2d} 正={int((x.差值>0).sum()):2d} "
              f"均值={x.差值.mean()*100:+.3f}% 范围=[{x.差值.min()*100:+.2f}%, {x.差值.max()*100:+.2f}%] "
              f"t<-2={int((x.t<-2).sum())} t>2={int((x.t>2).sum())}")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
