#!/usr/bin/env python3
"""Event study for the MACD variants V3 and V6."""
from __future__ import annotations

import math
import pandas as pd
import duckdb
from importlib import import_module

es = import_module("03_event_study")
BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/02_代码"
OUT = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/03_结果"


def main() -> None:
    df = pd.read_parquet(f"{BASE}/candidates_macd.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].map(es.period_of)
    df["limit_up_open"] = df["next_raw_open"] >= df["next_raw_preclose"] * (1 + df["limit_pct"] / 100.0) - 0.001
    df["gap"] = df["next_open"] / df["hfq_close"] - 1.0
    df["tradable"] = (df["next_status"] == 1) & (~df["limit_up_open"]) & (df["gap"] <= 0.05)

    flags = {
        "V1_weekly": df["base_ok"] & df["trend"] & df["pullback"] & df["reclaim"] & df["wok"],
        "V2_monthly_ma": df["base_ok"] & df["trend"] & df["pullback"] & df["reclaim"] & df["wok"] & df["mok"],
        "V3_daily_macd": df["base_ok"] & df["trend"] & df["pullback"] & df["reclaim"] & df["wok"] & df["mok"] & df["macd_daily_ok"],
        "V6_monthly_macd": df["base_ok"] & df["trend"] & df["pullback"] & df["reclaim"] & df["wok"] & df["mok_macd"],
    }
    con = duckdb.connect(f"{BASE}/work.duckdb", read_only=True)
    ew = con.execute("select date, nav, cal_idx from ew_nav order by date").fetchdf()
    con.close()

    stats, yearly, events = [], [], []
    for label, flag in flags.items():
        g = df.loc[flag].copy()
        g = g[g["tradable"]].copy()
        g = es.add_returns(g)
        g = es.attach_bench(g, ew, "ew")
        g = es.dedup_20(g)
        g["variant"] = label
        events.append(g[["variant", "code", "date", "period", "next_date", "hfq_close", "next_open",
                         "r5", "r10", "r20", "r40", "r60", "ex5", "ex10", "ex20", "ex40", "ex60",
                         "mfe", "mae", "gap", "amt20", "raw_close", "is_st"]])
        for h in (5, 10, 20, 40, 60):
            stats.extend(es.summarize(g, h, label))
        for y, gy in g.groupby(g["date"].dt.year):
            yearly.append({"variant": label, "year": int(y), "n": len(gy),
                           "r20_mean": gy["r20"].mean(), "r20_median": gy["r20"].median(),
                           "ex20_mean": gy["ex20"].mean(), "ex20_median": gy["ex20"].median(),
                           "win20": (gy["r20"] > 0).mean(), "win20_ex": (gy["ex20"] > 0).mean()})
        print(f"{label}: raw={int(flag.sum())}, dedup={len(g)}")

    pd.DataFrame(stats).to_csv(f"{OUT}/信号层_MACD变体_汇总.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(f"{OUT}/信号层_MACD变体_分年.csv", index=False, encoding="utf-8-sig")
    pd.concat(events, ignore_index=True).to_csv(f"{OUT}/信号层_MACD变体_逐笔.csv", index=False, encoding="utf-8-sig")
    s = pd.DataFrame(stats)
    print(s[(s["h"] == 20)][["variant", "period", "n", "mean", "median", "mean_ex", "median_ex", "win_ex", "t_ex_cluster"]].to_string(index=False))


if __name__ == "__main__":
    main()
