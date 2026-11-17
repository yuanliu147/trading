#!/usr/bin/env python3
"""53 事件控制：MA60/先期收益/波动双重排序、BH-FDR、5/20/60、逐年。"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
import kdj_core as K
from stats_utils import nw_t, _bh

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
COST = 0.0015


def dm(sub, col="ex20"):
    return sub.groupby("date")[col].mean()


def contrast(a, b, col="ex20"):
    da, db = dm(a, col), dm(b, col)
    common = da.index.intersection(db.index)
    if len(common) < 30:
        return dict(A=np.nan, B=np.nan, 差值=np.nan, t=np.nan, 共同日=len(common))
    x = (da.reindex(common) - db.reindex(common)).dropna()
    return dict(A=float(da.reindex(common).mean()), B=float(db.reindex(common).mean()),
                差值=float(x.mean()), t=float(nw_t(x, 20)), 共同日=len(x))


def horizon_stat(sub, h):
    col = f"ex{h}"
    x = sub.dropna(subset=[col])
    if len(x) < 100:
        return dict(信号数=len(x), 毛超额=np.nan, t=np.nan, 净超额=np.nan)
    d = x.groupby("date")[col].mean()
    return dict(信号数=len(x), 毛超额=float(d.mean()), t=float(nw_t(d, 20)),
                净超额=float(d.mean() - COST))


def main():
    t0 = time.time()
    P = M.load_panel()
    dates = P["dates"]
    base = F.base_mask(P)
    st = F.build_structure(P, M.market_features(P))
    kd = K.kdj_features(P)
    k = kd["k"]
    qk = F.row_quantile(k, base, 5)
    qr = F.row_quantile(st["ret20"], base, 5)
    qv = F.row_quantile(st["std20"], base, 5)
    print("panel done", time.time() - t0)

    # ---------- MACD 金叉事件明细 ----------
    e = pd.read_parquet(os.path.join(RES, "事件_金叉_12269.parquet"))
    e = e[np.isfinite(e.ex20)].copy()
    di = pd.Index(dates).get_indexer(e.date)
    ci = pd.Index(P["codes"]).get_indexer(e.code)
    good = (di >= 0) & (ci >= 0)
    e = e[good].copy(); di, ci = di[good], ci[good]
    e["Kq"] = qk[di, ci]
    e["qr"] = qr[di, ci]
    e["qv"] = qv[di, ci]
    e["K>D"] = kd["states"]["K>D"][di, ci]
    e["K>70"] = (k > 70)[di, ci]
    e["K<20"] = (k < 20)[di, ci]
    e["kdj_gold_today"] = kd["gold"][di, ci]
    e["K"] = k[di, ci]
    e["above60"] = e.above_ma60.astype(bool)
    print("e done", len(e))

    # 双重排序
    rows = []
    for by, byname in [("above60", "MA60上"), ("qr", "先期收益分位"), ("qv", "波动分位")]:
        vals = [True, False] if by == "above60" else range(5)
        for v in vals:
            sub = e[e[by] == v]
            if len(sub) < 5000:
                continue
            a = sub[sub.Kq == 4]
            b = sub[sub.Kq.isin([1, 2, 3])]
            r = contrast(a, b)
            rows.append(dict(分层=by, 层=v, 对比="KQ5 vs KQ2-4", nA=len(a), nB=len(b), **r))
            a1 = sub[sub.Kq == 0]
            r1 = contrast(a1, b)
            rows.append(dict(分层=by, 层=v, 对比="KQ1 vs KQ2-4", nA=len(a1), nB=len(b), **r1))
            am = sub[sub["K>D"].astype(bool)]
            bm = sub[~sub["K>D"].astype(bool)]
            r2 = contrast(am, bm)
            rows.append(dict(分层=by, 层=v, 对比="KDJ多 vs 空", nA=len(am), nB=len(bm), **r2))
    ds = pd.DataFrame(rows)
    ds.to_csv(os.path.join(RES, "KDJ_事件_双重排序.csv"), index=False, encoding="utf-8-sig")
    print("\n双重排序")
    print(ds.to_string(index=False))

    # 逐年差异
    rows = []
    for lab, a, b in [
        ("KQ5 vs KQ2-4", e[e.Kq == 4], e[e.Kq.isin([1, 2, 3])]),
        ("KQ1 vs KQ2-4", e[e.Kq == 0], e[e.Kq.isin([1, 2, 3])]),
        ("KDJ多 vs 空", e[e["K>D"].astype(bool)], e[~e["K>D"].astype(bool)]),
        ("同日双金叉 vs 非", e[e.kdj_gold_today.astype(bool)], e[~e.kdj_gold_today.astype(bool)]),
        ("K>70 vs <=", e[e["K>70"].astype(bool)], e[~e["K>70"].astype(bool)]),
    ]:
        da, db = dm(a), dm(b)
        common = da.index.intersection(db.index)
        x = (da.reindex(common) - db.reindex(common)).dropna()
        for y, val in x.groupby(x.index.year).mean().items():
            rows.append(dict(对比=lab, 年份=int(y), 差值=float(val),
                             t=float(nw_t(x[x.index.year == y], 5)),
                             共同日=int((x.index.year == y).sum())))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "KDJ_事件_逐年差异.csv"), index=False, encoding="utf-8-sig")
    print("\n逐年差异")
    print(pd.DataFrame(rows).groupby("对比").agg(年数=("年份", "count"), 正=("差值", lambda s: int((s > 0).sum())),
                                                 均值=("差值", "mean"), 最小=("差值", "min"), 最大=("差值", "max")).to_string())

    # 5/20/60
    rows = []
    for lab, sub in [
        ("MACD金叉全部", e),
        ("MACD金叉+KQ5", e[e.Kq == 4]),
        ("MACD金叉+KQ2-4", e[e.Kq.isin([1, 2, 3])]),
        ("MACD金叉+KDJ多", e[e["K>D"].astype(bool)]),
        ("MACD金叉+KDJ空", e[~e["K>D"].astype(bool)]),
        ("MACD金叉+同日KDJ金叉", e[e.kdj_gold_today.astype(bool)]),
        ("MACD金叉+K>70", e[e["K>70"].astype(bool)]),
    ]:
        for h in (5, 20, 60):
            rows.append(dict(类别=lab, 周期=h, **horizon_stat(sub, h)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "KDJ_事件_期限.csv"), index=False, encoding="utf-8-sig")
    print("\n期限")
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------- BH-FDR ----------
    print("\nBH-FDR")
    for fn, lab in [("KDJ_事件_MACD金叉.csv", "MACD金叉过滤"),
                    ("KDJ_事件_KDJ金叉.csv", "KDJ金叉过滤")]:
        d = pd.read_csv(os.path.join(RES, fn))
        p = 2 * norm.sf(np.abs(d["t值"].to_numpy(float)))
        d["q值"] = _bh(p)
        d.to_csv(os.path.join(RES, fn.replace(".csv", "_q.csv")), index=False, encoding="utf-8-sig")
        print(lab)
        print(d[["过滤", "信号数", "毛超额", "t值", "q值", "前期", "后期"]].sort_values("q值").to_string(index=False))

    # ---------- KDJ 金叉事件的 MA60 控制 ----------
    ek = pd.read_parquet(os.path.join(RES, "KDJ_事件_KDJ金叉.parquet"))
    rows = []
    for v in [True, False]:
        sub = ek[ek.above60.astype(bool) == v]
        a = sub[sub.macd_bull.astype(bool)]
        b = sub[~sub.macd_bull.astype(bool)]
        rows.append(dict(分层=f"价格>MA60={v}", 对比="KDJ金叉 MACD多 vs 空", nA=len(a), nB=len(b), **contrast(a, b)))
        a2 = sub[sub.mg5.astype(bool)]
        b2 = sub[~sub.mg5.astype(bool)]
        rows.append(dict(分层=f"价格>MA60={v}", 对比="KDJ金叉 前5日MACD金叉 vs 无", nA=len(a2), nB=len(b2), **contrast(a2, b2)))
        a3 = sub[sub.Kq == 4]
        b3 = sub[sub.Kq.isin([1, 2, 3])]
        rows.append(dict(分层=f"价格>MA60={v}", 对比="KDJ金叉 KQ5 vs Q2-4", nA=len(a3), nB=len(b3), **contrast(a3, b3)))
    ck = pd.DataFrame(rows)
    ck.to_csv(os.path.join(RES, "KDJ_事件_KDJ金叉_MA60控制.csv"), index=False, encoding="utf-8-sig")
    print("\nKDJ金叉 MA60 控制")
    print(ck.to_string(index=False))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
