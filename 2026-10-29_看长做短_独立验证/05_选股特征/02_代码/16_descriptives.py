#!/usr/bin/env python3
"""Descriptive cuts for the selection report: industry, board, market state, liquidity, year."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from importlib import import_module

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/05_选股特征"
OUT = f"{BASE}/03_结果"
u = import_module("12_univariate")


def main() -> None:
    raw = pd.read_parquet(f"{BASE}/02_代码/信号_带评分.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    raw["next_date"] = pd.to_datetime(raw["next_date"])
    ev = u.dedup_20(raw)
    dev = ev[ev["date"].dt.year <= 2017]
    cut = dev["score_A"].quantile([0.5, 0.7, 0.9]).values
    ev["top30"] = ev["score_A"] >= cut[1]
    ev["band"] = (ev["score_A"] >= cut[1]) & (ev["score_A"] < cut[2])
    ev["year"] = ev["date"].dt.year

    oos = ev[ev["year"] >= 2018].copy()
    # industry
    ind = oos.groupby("industry", dropna=False).agg(
        n=("ex20", "size"), ex20_all=("ex20", "mean"), ex20_median=("ex20", "median"),
        win_all=("ex20", lambda s: (s > 0).mean()), score=("score_A", "mean"),
        top30_n=("top30", "sum"),
    ).reset_index()
    ind["ex20_top30"] = oos[oos["top30"]].groupby("industry", dropna=False)["ex20"].mean().reindex(ind["industry"]).values
    ind["ex20_top30"] = ind["ex20_top30"].fillna(np.nan)
    ind = ind[ind["n"] >= 100].sort_values("ex20_all", ascending=False)
    ind.to_csv(f"{OUT}/特征_行业.csv", index=False, encoding="utf-8-sig")

    # board
    rows = []
    for name, col in [("主板", "board_main"), ("创业板", "board_gem"), ("科创板", "board_star")]:
        g = oos[oos[col] == 1]
        if len(g):
            rows.append({"board": name, "n": len(g), "ex20_all": g["ex20"].mean(),
                         "ex20_top30": g[g["top30"]]["ex20"].mean(), "ex20_band": g[g["band"]]["ex20"].mean(),
                         "win": (g["ex20"] > 0).mean(), "score": g["score_A"].mean()})
    pd.DataFrame(rows).to_csv(f"{OUT}/特征_板块.csv", index=False, encoding="utf-8-sig")

    # market state
    rows = []
    for gate, name in [(1, "闸门开"), (0, "闸门关")]:
        g = oos[oos["mkt_gate"] == gate]
        if len(g):
            rows.append({"state": name, "n": len(g), "ex20_all": g["ex20"].mean(),
                         "ex20_top30": g[g["top30"]]["ex20"].mean(), "ex20_band": g[g["band"]]["ex20"].mean(),
                         "win_all": (g["ex20"] > 0).mean(), "win_top30": (g[g["top30"]]["ex20"] > 0).mean()})
    pd.DataFrame(rows).to_csv(f"{OUT}/特征_市场状态.csv", index=False, encoding="utf-8-sig")

    # liquidity
    bins = [0, 5e7, 1e8, 3e8, 1e12]
    labels = ["20-50M", "50-100M", "100-300M", ">300M"]
    oos["liq"] = pd.cut(oos["amt20"], bins=bins, labels=labels)
    rows = []
    for name, g in oos.groupby("liq", observed=True):
        rows.append({"liq": name, "n": len(g), "ex20_all": g["ex20"].mean(),
                     "ex20_top30": g[g["top30"]]["ex20"].mean(), "ex20_band": g[g["band"]]["ex20"].mean(),
                     "win_all": (g["ex20"] > 0).mean(), "score": g["score_A"].mean()})
    pd.DataFrame(rows).to_csv(f"{OUT}/特征_流动性分层.csv", index=False, encoding="utf-8-sig")

    # year
    rows = []
    for y, g in ev.groupby("year"):
        rows.append({"year": int(y), "n_all": len(g), "ex20_all": g["ex20"].mean(),
                     "ex20_top30": g[g["top30"]]["ex20"].mean(),
                     "ex20_band": g[g["band"]]["ex20"].mean(),
                     "win_all": (g["ex20"] > 0).mean(), "win_top30": (g[g["top30"]]["ex20"] > 0).mean()})
    pd.DataFrame(rows).to_csv(f"{OUT}/特征_分年.csv", index=False, encoding="utf-8-sig")

    # family summary from IC table
    ic = pd.read_csv(f"{OUT}/特征_IC.csv")
    fam = ic.groupby("family").agg(
        n=("feature", "size"), pass_fdr=("pass_fdr", "sum"),
        med_ic_dev=("ic_dev", "median"), med_ic_oos=("ic_oos", "median"),
        best_feature=("ic_dev", lambda s: ic.loc[s.abs().idxmax(), "feature"]),
        best_ic=("ic_dev", lambda s: s.loc[s.abs().idxmax()]),
    ).reset_index()
    fam.to_csv(f"{OUT}/特征_族汇总.csv", index=False, encoding="utf-8-sig")

    print("industry OOS top/bottom")
    print(ind[["industry", "n", "ex20_all", "ex20_top30", "score"]].head(8).to_string(index=False))
    print(ind[["industry", "n", "ex20_all", "ex20_top30", "score"]].tail(8).to_string(index=False))
    print("\nboard\n", pd.DataFrame(rows).to_string(index=False) if False else "")
    print(pd.read_csv(f"{OUT}/特征_板块.csv").to_string(index=False))
    print("\nmarket\n", pd.read_csv(f"{OUT}/特征_市场状态.csv").to_string(index=False))
    print("\nliq\n", pd.read_csv(f"{OUT}/特征_流动性分层.csv").to_string(index=False))
    print("\nyear\n", pd.read_csv(f"{OUT}/特征_分年.csv").to_string(index=False))
    print("\nfam\n", fam.to_string(index=False))


if __name__ == "__main__":
    main()
