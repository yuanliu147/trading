#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 01 的事件结果，补股性标签并汇总。"""
import os, pickle, math, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "..", "2026-10-18_成交量融合_2010-2025", "01_数据", "daily_cache_2016_2026.pkl")
END = pd.Timestamp("2026-09-30")

with open(CACHE, "rb") as f:
    D = pickle.load(f)
stocks = D["stocks"]
res = pd.read_pickle(os.path.join(BASE, "特征点_事件结果_未分类.pkl"))
res = res.drop(columns=[c for c in ("type", "s_vol", "s_heat") if c in res.columns])
print("events:", res.shape, flush=True)


def board_limit(code, date):
    if code[3:6] in ("300", "301", "688") and date >= pd.Timestamp("2020-08-24"):
        return 0.195
    return 0.095


print("computing stock type ...", flush=True)
feat_rows = []
for j, (code, g) in enumerate(stocks.items()):
    if len(g) < 320:
        continue
    g = g.sort_values("date").reset_index(drop=True)
    m = (g["date"] >= pd.Timestamp("2015-06-01")) & (g["date"] <= END)
    if not m.any():
        continue
    c, h, l, o, a = (g[x] for x in ("close", "high", "low", "open", "amount"))
    ret = c.pct_change()
    vol120 = (ret.rolling(120).std() * math.sqrt(252)).loc[m]
    range250 = (h.rolling(250).max() / l.rolling(250).min() - 1).loc[m]
    act = (a.rolling(20).mean() / a.rolling(120).mean() - 1).loc[m]
    lim_all = np.array([board_limit(code, d) for d in g["date"]])
    limfrac = pd.Series((ret.abs().values >= lim_all * 0.95).astype(float), index=g.index).rolling(250).mean().loc[m]
    gapfrac = ((o / c.shift(1) - 1).abs() > 0.02).rolling(250).mean().loc[m]
    amt60 = a.rolling(60).mean().loc[m]
    feat_rows.append(pd.DataFrame({
        "date": g.loc[m, "date"].values, "code": code,
        "vol120": vol120.values, "range250": range250.values, "act": act.values,
        "limfrac": limfrac.values, "gapfrac": gapfrac.values, "amount60": amt60.values,
    }))
    if (j + 1) % 300 == 0:
        print("  ", j + 1, flush=True)
feat = pd.concat(feat_rows, ignore_index=True)
feat = feat[(feat.amount60 >= 2e7) & feat.vol120.notna() & feat.range250.notna() & feat.act.notna()]
feat = feat[feat.date.isin(set(res.date.unique()))].copy()
print("feature rows on event dates:", len(feat), flush=True)
for col in ("vol120", "range250", "act", "limfrac", "gapfrac"):
    feat["r_" + col] = feat.groupby("date")[col].rank(pct=True)
feat["s_vol"] = feat[["r_vol120", "r_range250"]].mean(axis=1)
feat["s_heat"] = feat[["r_act", "r_limfrac", "r_gapfrac"]].mean(axis=1)
feat["type"] = np.select(
    [(feat.s_vol >= .5) & (feat.s_heat >= .5), (feat.s_vol >= .5) & (feat.s_heat < .5),
     (feat.s_vol < .5) & (feat.s_heat >= .5)],
    ["高波活跃型", "高波低量型", "低波活跃型"], default="低波低量型")
res = res.merge(feat[["date", "code", "s_vol", "s_heat", "type"]], on=["date", "code"], how="left")
print("type missing:", res.type.isna().sum(), flush=True)

res["period"] = np.where(res.date <= pd.Timestamp("2021-12-31"), "2016-2021", "2022-2026")
res["quiet"] = ((res.vr <= 0.8) & (res.platform <= 0.18) & (res.ret20 <= 0.20))
res["mkt"] = np.select([res.mkt_ma20 & res.mkt_ma60, res.mkt_ma20 & ~res.mkt_ma60],
                       ["MA20上/MA60上", "MA20上/MA60下"], default="MA20下")
res.to_csv(os.path.join(BASE, "特征点_事件明细.csv"), index=False, encoding="utf-8-sig")
print("saved detail", flush=True)


def agg(df):
    return pd.Series({
        "n": len(df),
        "胜率20": (df.ret20 > 0).mean(),
        "20日均值": df.ret20.mean(),
        "20日中位": df.ret20.median(),
        "20日超额": df.ex20.mean(),
        "20日超额中位": df.ex20.median(),
        "MAE20": df.mae20.mean(),
        "MFE20": df.mfe20.mean(),
        "止损率20": df.stop_hit20.mean(),
        "trail均值": df.ret_trail.mean(),
        "trail中位": df.ret_trail.median(),
        "trail胜率": (df.ret_trail > 0).mean(),
        "平均R": df.R_trail.mean(),
        "平均持有": df.held.mean(),
    })


rows = []
for (setup, period), g in res.groupby(["setup", "period"]):
    rows.append({"setup": setup, "period": period, "group": "全部", **agg(g)})
for (setup, period, typ), g in res.groupby(["setup", "period", "type"]):
    rows.append({"setup": setup, "period": period, "group": typ, **agg(g)})
for (setup, period, q), g in res.groupby(["setup", "period", "quiet"]):
    rows.append({"setup": setup, "period": period, "group": "安静过滤" if q else "未过滤", **agg(g)})
for (setup, period, m), g in res.groupby(["setup", "period", "mkt"]):
    rows.append({"setup": setup, "period": period, "group": m, **agg(g)})
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(BASE, "特征点_汇总.csv"), index=False, encoding="utf-8-sig")

yearly = []
for (setup, y), g in res.groupby(["setup", res.date.dt.year]):
    yearly.append({"setup": setup, "year": y, "n": len(g), "win20": (g.ret20 > 0).mean(),
                   "ret20": g.ret20.mean(), "ex20": g.ex20.mean(), "mae20": g.mae20.mean(),
                   "stop_rate": g.stop_hit20.mean()})
pd.DataFrame(yearly).to_csv(os.path.join(BASE, "特征点_年度.csv"), index=False, encoding="utf-8-sig")

frows = []
for (setup, period, q), g in res.groupby(["setup", "period", "quiet"]):
    frows.append({"setup": setup, "period": period, "quiet": bool(q), "n": len(g),
                  "win20": (g.ret20 > 0).mean(), "ret20": g.ret20.mean(), "ex20": g.ex20.mean(),
                  "mae20": g.mae20.mean(), "stop_rate": g.stop_hit20.mean(),
                  "trail": g.ret_trail.mean(), "avgR": g.R_trail.mean()})
pd.DataFrame(frows).to_csv(os.path.join(BASE, "特征点_过滤对比.csv"), index=False, encoding="utf-8-sig")

erows = []
for (setup, period), g in res.groupby(["setup", "period"]):
    row = {"setup": setup, "period": period, "n": len(g)}
    for col, name in (("ret_trail", "止损+破MA20"), ("ret_tp2R", "固定2R"),
                      ("ret_tp3R", "固定3R"), ("ret_third", "1/3在2R+余仓trail"),
                      ("ret_half", "1/2在2R+余仓trail"), ("ret_be", "1R后保本+trail")):
        row[name + "_均值"] = g[col].mean()
        row[name + "_胜率"] = (g[col] > 0).mean()
        row[name + "_中位"] = g[col].median()
    row["trail平均R"] = g.R_trail.mean()
    erows.append(row)
pd.DataFrame(erows).to_csv(os.path.join(BASE, "出场方案对比.csv"), index=False, encoding="utf-8-sig")
print("done", flush=True)
