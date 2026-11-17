#!/usr/bin/env python3
"""Latest signal watchlist with score A and the 70-90 band flag.

Frozen valuation ends 2026-10-26, so for the 2026-10-27 signals the PB ratio is
backfilled from 10-26 and score A is recomputed with the same dev-period
winsorisation/standardisation. This is a display snapshot, not a live signal.
"""
from __future__ import annotations

import math
from importlib import import_module

import duckdb
import pandas as pd

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证"
SEL = f"{BASE}/05_选股特征"
OUT = f"{SEL}/03_结果"
u = import_module("12_univariate")


def main() -> None:
    raw = pd.read_parquet(f"{SEL}/02_代码/信号_带评分.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    raw["next_date"] = pd.to_datetime(raw["next_date"])
    latest = raw["date"].max()
    g = raw[raw["date"] == latest].copy()
    g["score_raw"] = g["score_A"]

    con = duckdb.connect(f"{BASE}/02_代码/work.duckdb", read_only=True)
    names = con.execute("SELECT code, code_name FROM stocks").fetchdf()
    con.close()
    g = g.merge(names, on="code", how="left")

    # frozen metadata ends 2026-10-26; backfill board/mcap/valuation for display
    def board_of(code: str) -> str:
        if code.startswith("sh.68"):
            return "科创板"
        if code.startswith("sz.30"):
            return "创业板"
        return "主板"
    g["board"] = g["code"].map(board_of)
    hist = raw.sort_values(["code", "date"])
    for col in ["log_mcap", "pb_med_ratio250"]:
        last = hist.dropna(subset=[col]).groupby("code")[col].last()
        g[col] = g["code"].map(last).fillna(g[col])

    # recompute score A with the same dev-period stats and the backfilled PB
    dev = u.dedup_20(raw[raw["date"].dt.year <= 2017])
    feats = ["log_amt20", "vol60", "pullback_depth", "pb_med_ratio250", "stock_ind_ex60"]
    zs = []
    for f in feats:
        d = dev[f].replace([float("inf"), float("-inf")], pd.NA).dropna()
        lo, hi = d.quantile([0.01, 0.99]).values
        mu, sd = d.clip(lo, hi).mean(), d.clip(lo, hi).std()
        zs.append(-(g[f].clip(lo, hi) - mu) / sd)
    g["score_A"] = pd.DataFrame(zs).T.mean(axis=1)

    cuts = pd.read_csv(f"{OUT}/评分_切点.csv").set_index("score").loc["A"]
    q50, q70, q90 = float(cuts["q50"]), float(cuts["q70"]), float(cuts["q90"])
    g["score_pct"] = pd.cut(g["score_A"], [-1e9, q50, q70, q90, 1e9],
                            labels=["后50%", "50-70%", "70-90%", "前10%"])
    g["in_band"] = (g["score_A"] >= q70) & (g["score_A"] < q90)
    g["note"] = g["in_band"].map({True: "70-90%中段候选", False: "不在中段"})
    g.loc[g["mkt_gate"] == 0, "note"] += "；市场闸门关闭"
    g = g.sort_values("score_A", ascending=False)

    cols = ["date", "next_date", "code", "code_name", "industry", "board", "score_A", "score_raw",
            "score_pct", "in_band", "amt20", "log_mcap", "vol60", "pullback_depth",
            "pb_med_ratio250", "stock_ind_ex60", "mkt_gate", "hs300_gate", "note"]
    out = g[cols].copy()
    out["amt20_yi"] = out["amt20"] / 1e8
    out["mcap_yi"] = (out["log_mcap"].apply(lambda x: math.exp(x) if pd.notna(x) else None)) / 1e8
    out = out.drop(columns=["amt20", "log_mcap"])
    out.to_csv(f"{OUT}/当前观察名单.csv", index=False, encoding="utf-8-sig")
    print("latest", latest, "n", len(out), "gate", int(g["mkt_gate"].iloc[0]))
    print(out.to_string(index=False))
    print("\ncut", q50, q70, q90)


if __name__ == "__main__":
    main()
