#!/usr/bin/env python3
"""38 聚类修正：去极值后按前期特征分股票类型，检验 MACD 适配是否随类型变化"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import stock_core as S
from stats_utils import nw_t

RES = S.RES
FEATS = ["规模_log", "换手20", "波动20", "beta250", "ER60", "VR20", "AC1_60",
         "MA60择时250", "动量12_1", "偏度250", "年龄_天"]


def main():
    st = pd.read_csv(os.path.join(RES, "个股_特征.csv"))
    tm = pd.read_csv(os.path.join(RES, "个股_择时表现_分期.csv"))
    ev = pd.read_parquet(os.path.join(RES, "个股_事件_增强.parquet"))
    ev["date"] = pd.to_datetime(ev["date"])

    cols = [f + "_前期" for f in FEATS]
    d = st[(st["有效天数_前期"] >= 500)].copy()
    X = d[cols].to_numpy(dtype=float)
    ok = np.all(np.isfinite(X), axis=1)
    d = d[ok].copy()
    X = d[cols].to_numpy(dtype=float)
    # 1%/99% winsorize
    lo, hi = np.nanpercentile(X, [1, 99], axis=0)
    X = np.clip(X, lo, hi)
    med = np.median(X, axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    Z = (X - med) / sd
    km = KMeans(n_clusters=5, n_init=50, random_state=42).fit(Z)
    d["类型"] = km.labels_
    # 类型名：按规模/换手/波动/趋势排序
    prof = pd.DataFrame(km.cluster_centers_, columns=FEATS)
    prof["股票数"] = [int((d["类型"] == k).sum()) for k in range(5)]
    prof.to_csv(os.path.join(RES, "适配_类型画像_v2.csv"), encoding="utf-8-sig")
    d[["code", "类型"]].to_csv(os.path.join(RES, "适配_类型归属_v2.csv"), index=False, encoding="utf-8-sig")

    # 事件
    e = ev.merge(d[["code", "类型"]], on="code", how="inner")
    rows = []
    for lab, m in [("前期", e["date"] <= pd.Timestamp("2017-12-31")),
                   ("后期", e["date"] >= pd.Timestamp("2018-01-01"))]:
        sub = e[m & e["ex20_style"].notna()]
        dm = sub.groupby(["date", "类型"])["ex20_style"].mean()
        for k, s in sub.groupby("类型"):
            dd = dm.xs(k, level=1) if k in dm.index.get_level_values(1) else pd.Series(dtype=float)
            rows.append(dict(期间=lab, 类型=int(k), 事件数=len(s), 均值=s["ex20_style"].mean(),
                             中位=s["ex20_style"].median(), 日t=nw_t(dd)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "适配_类型事件_v2.csv"), index=False, encoding="utf-8-sig")

    # 择时
    dd = d.merge(tm, on="code", how="inner")
    rows = []
    for lab in ["前期", "后期"]:
        sub = dd[dd["期间"] == lab]
        for k, s in sub.groupby("类型"):
            rows.append(dict(期间=lab, 类型=int(k), 股票数=len(s),
                             净超同暴露=s["净超同暴露"].mean(),
                             毛超同暴露=s["毛超同暴露"].mean(),
                             夏普差=s["夏普差"].mean(),
                             回撤超同暴露=s["回撤超同暴露"].mean()))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "适配_类型择时_v2.csv"), index=False, encoding="utf-8-sig")

    print(prof.round(2).to_string())
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
