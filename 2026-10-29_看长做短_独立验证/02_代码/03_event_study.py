#!/usr/bin/env python3
"""Signal-level event study for the 看长做短 mechanical proxies.

Reads candidates.parquet (created from raw baostock bars), applies the
pre-registered 20-trading-day de-duplication, and reports forward returns
and excess vs an equal-weight all-A benchmark.
"""
from __future__ import annotations

import math
import os
import numpy as np
import pandas as pd
import duckdb

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/02_代码"
OUT = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/03_结果"
COST = 0.0020  # round trip


def dedup_20(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    keep = np.zeros(len(df), dtype=bool)
    last = {}
    for i, code, ci in zip(df.index.values, df["code"].values, df["cal_idx"].values):
        if code not in last or ci - last[code] >= 20:
            keep[i] = True
            last[code] = ci
    return df.loc[keep].copy()


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for h in (5, 10, 20, 40, 60):
        df[f"r{h}"] = df[f"c{h}"] / df["next_open"] - 1.0
        df[f"r{h}_net"] = df[f"r{h}"] - COST
    df["mfe"] = df["mfe20"] / df["next_open"] - 1.0
    df["mae"] = df["mae20"] / df["next_open"] - 1.0
    return df


def attach_bench(df: pd.DataFrame, nav: pd.DataFrame, prefix: str) -> pd.DataFrame:
    max_idx = int(nav["cal_idx"].max())
    arr = np.full(max_idx + 2, np.nan)
    arr[nav["cal_idx"].values.astype(int)] = nav["nav"].values
    df = df.copy()
    ci = df["cal_idx"].values.astype(int)
    for h in (5, 10, 20, 40, 60):
        entry = np.where(ci + 1 <= max_idx, arr[np.minimum(ci + 1, max_idx)], np.nan)
        exit_ = np.where(ci + h <= max_idx, arr[np.minimum(ci + h, max_idx)], np.nan)
        df[f"{prefix}{h}"] = exit_ / entry - 1.0
        df[f"ex{h}"] = df[f"r{h}"] - df[f"{prefix}{h}"]
    return df


def period_of(d: pd.Timestamp) -> str:
    if d.year <= 2017:
        return "dev_2012_2017"
    return "oos_2018_2026"


def summarize(df: pd.DataFrame, h: int, label: str) -> list[dict]:
    rows = []
    for pname, g in [("full", df), ("dev_2012_2017", df[df["period"] == "dev_2012_2017"]),
                     ("oos_2018_2026", df[df["period"] == "oos_2018_2026"])]:
        if len(g) == 0:
            continue
        r = g[f"r{h}"]
        ex = g[f"ex{h}"]
        daily = g.groupby("date")[f"ex{h}"].mean()
        t_ex = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))) if len(daily) > 2 else np.nan
        daily_r = g.groupby("date")[f"r{h}"].mean()
        t_r = daily_r.mean() / (daily_r.std(ddof=1) / math.sqrt(len(daily_r))) if len(daily_r) > 2 else np.nan
        rows.append({
            "variant": label, "h": h, "period": pname,
            "n": len(g), "n_days": g["date"].nunique(),
            "mean": r.mean(), "median": r.median(),
            "mean_net": g[f"r{h}_net"].mean(), "median_net": g[f"r{h}_net"].median(),
            "win": (r > 0).mean(), "win_net": (g[f"r{h}_net"] > 0).mean(),
            "mean_ex": ex.mean(), "median_ex": ex.median(),
            "win_ex": (ex > 0).mean(), "t_ex_cluster": t_ex, "t_r_cluster": t_r,
            "p10": r.quantile(0.10), "p90": r.quantile(0.90),
            "mfe": g["mfe"].mean(), "mae": g["mae"].mean(),
        })
    return rows


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(f"{BASE}/candidates.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].map(period_of)

    # execution filters
    df["limit_up_open"] = df["next_raw_open"] >= df["next_raw_preclose"] * (1 + df["limit_pct"] / 100.0) - 0.001
    df["gap"] = df["next_open"] / df["hfq_close"] - 1.0
    df["tradable"] = (df["next_status"] == 1) & (~df["limit_up_open"]) & (df["gap"] <= 0.05)

    # variants
    flags = {
        "V0_daily": df["base_ok"] & df["trend"] & df["pullback"] & df["reclaim"],
        "V1_weekly": df["base_ok"] & df["trend"] & df["pullback"] & df["reclaim"] & df["wok"],
        "V2_monthly": df["base_ok"] & df["trend"] & df["pullback"] & df["reclaim"] & df["wok"] & df["mok"],
        "V7_no_liquidity": df["trend"] & df["pullback"] & df["reclaim"] & df["wok"] & df["mok"],
    }

    con = duckdb.connect(f"{BASE}/work.duckdb", read_only=True)
    ew = con.execute("select date, nav, cal_idx from ew_nav order by date").fetchdf()
    con.close()

    all_stats = []
    all_yearly = []
    all_events = []
    for label, flag in flags.items():
        g = df.loc[flag].copy()
        g = g[g["tradable"]].copy()
        g = add_returns(g)
        g = attach_bench(g, ew, "ew")
        g = dedup_20(g)
        g["variant"] = label
        all_events.append(g[[
            "variant", "code", "date", "period", "next_date", "hfq_close", "next_open",
            "r5", "r10", "r20", "r40", "r60", "ex5", "ex10", "ex20", "ex40", "ex60",
            "mfe", "mae", "gap", "amt20", "raw_close", "is_st",
        ]])
        for h in (5, 10, 20, 40, 60):
            all_stats.extend(summarize(g, h, label))
        # yearly
        for y, gy in g.groupby(g["date"].dt.year):
            all_yearly.append({
                "variant": label, "year": int(y), "n": len(gy),
                "r20_mean": gy["r20"].mean(), "r20_median": gy["r20"].median(),
                "ex20_mean": gy["ex20"].mean(), "ex20_median": gy["ex20"].median(),
                "win20": (gy["r20"] > 0).mean(), "win20_ex": (gy["ex20"] > 0).mean(),
            })
        print(f"{label}: raw={int(flag.sum())}, after_tradable_dedup={len(g)}")

    stats = pd.DataFrame(all_stats)
    yearly = pd.DataFrame(all_yearly)
    events = pd.concat(all_events, ignore_index=True)
    stats.to_csv(f"{OUT}/信号层_汇总.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(f"{OUT}/信号层_分年.csv", index=False, encoding="utf-8-sig")
    events.to_csv(f"{OUT}/信号层_逐笔.csv", index=False, encoding="utf-8-sig")

    print("\n=== 20-day full ===")
    print(stats[(stats["h"] == 20) & (stats["period"] == "full")].to_string(index=False))
    print("\n=== 20-day dev/oos ===")
    print(stats[(stats["h"] == 20) & (stats["period"] != "full")].to_string(index=False))


if __name__ == "__main__":
    main()
