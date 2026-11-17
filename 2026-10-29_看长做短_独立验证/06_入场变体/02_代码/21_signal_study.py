#!/usr/bin/env python3
"""Signal-level study for the pre-registered entry variants.

Every signal set uses the same universe, execution filter and 20-day per-stock
de-duplication. Targets: raw return, excess vs equal-weight all-A, excess vs
point-in-time industry equal-weight.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/06_入场变体"
SRC = f"{BASE}/02_代码/entries.parquet"
OUT = f"{BASE}/03_结果"
PERIODS = ["full", "dev_2012_2017", "oos_2018_2026"]


def dedup_20(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    keep = np.zeros(len(df), dtype=bool)
    last: dict[str, int] = {}
    for i, code, ci in zip(df.index.values, df["code"].values, df["cal_idx"].values):
        if code not in last or ci - last[code] >= 20:
            keep[i] = True
            last[code] = ci
    return df.loc[keep].copy()


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


def summarize(g: pd.DataFrame, entry: str, h: int) -> list[dict]:
    rows = []
    for pname in PERIODS:
        s = g if pname == "full" else g[g["period"] == pname]
        if len(s) == 0:
            continue
        r, ex, exi = s[f"r{h}"], s[f"ex{h}"], s[f"ex_ind{h}"]
        daily = s.groupby("date")[f"ex{h}"].mean()
        daily_r = s.groupby("date")[f"r{h}"].mean()
        rows.append({
            "entry": entry, "h": h, "period": pname,
            "n": len(s), "n_days": s["date"].nunique(), "n_codes": s["code"].nunique(),
            "mean": r.mean(), "median": r.median(), "win": (r > 0).mean(),
            "mean_ex": ex.mean(), "median_ex": ex.median(), "win_ex": (ex > 0).mean(),
            "t_ex_cluster": tstat(daily), "t_r_cluster": tstat(daily_r),
            "mean_ex_ind": exi.mean(), "median_ex_ind": exi.median(), "win_ex_ind": (exi > 0).mean(),
            "p10": r.quantile(0.10), "p90": r.quantile(0.90),
            "mfe": s["mfe"].mean(), "mae": s["mae"].mean(),
        })
    return rows


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"])
    df["next_date"] = pd.to_datetime(df["next_date"])
    base = df["base_ok"] & df["trend"] & df["tradable"]
    topdown = df["wok"] & df["mok"]

    sets = {
        "E0_回踩MA20": base & topdown & df["pullback"] & df["reclaim"],
        "E1_20日新高": base & topdown & df["e1"],
        "E1V_放量突破": base & topdown & df["e1v"],
        "E2_浅回踩MA10": base & topdown & df["e2"],
        "E3_平台突破": base & topdown & df["e3"],
        "E1d_仅日线突破": base & df["e1"],
        "E2d_仅日线浅回踩": base & df["e2"],
    }

    all_stats, all_year, all_ctx = [], [], []
    deduped = {}
    for name, flag in sets.items():
        g = df.loc[flag].copy()
        g = dedup_20(g)
        deduped[name] = g
        for h in (5, 10, 20):
            all_stats.extend(summarize(g, name, h))
        for y, gy in g.groupby("year"):
            all_year.append({
                "entry": name, "year": int(y), "n": len(gy),
                "mean_r20": gy["r20"].mean(), "mean_ex20": gy["ex20"].mean(),
                "median_ex20": gy["ex20"].median(), "win20": (gy["r20"] > 0).mean(),
                "mean_ex5": gy["ex5"].mean(), "mean_ex10": gy["ex10"].mean(),
            })
        # context splits on the 20-day horizon
        g["board"] = np.select(
            [g["code"].str.startswith("sh.68"), g["code"].str.startswith("sz.30")],
            ["科创板", "创业板"], default="主板")
        g["liq"] = pd.cut(g["amt20"], [0, 5e7, 1e8, 3e8, np.inf],
                          labels=["20-50M", "50-100M", "100-300M", ">300M"])
        for dim, col in [("gate", "mkt_gate"), ("board", "board"), ("liq", "liq")]:
            for k, gg in g.groupby(col, observed=True, dropna=False):
                daily = gg.groupby("date")["ex20"].mean()
                all_ctx.append({
                    "entry": name, "dim": dim, "bucket": str(k), "n": len(gg),
                    "mean_r20": gg["r20"].mean(), "mean_ex20": gg["ex20"].mean(),
                    "mean_ex_ind20": gg["ex_ind20"].mean(),
                    "t_ex_cluster": tstat(daily),
                })
        print(f"{name}: raw={int(flag.sum())}, dedup={len(g)}, "
              f"dev={int((g.period=='dev_2012_2017').sum())}, oos={int((g.period=='oos_2018_2026').sum())}",
              flush=True)

    stats = pd.DataFrame(all_stats)
    stats.to_csv(f"{OUT}/入场_信号层.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_year).to_csv(f"{OUT}/入场_信号_分年.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_ctx).to_csv(f"{OUT}/入场_信号_分层.csv", index=False, encoding="utf-8-sig")

    cols = ["entry", "period", "n", "mean", "median", "mean_ex", "median_ex", "win_ex",
            "t_ex_cluster", "mean_ex_ind", "t_r_cluster"]
    for h in (5, 10, 20):
        print(f"\n=== h={h} ===")
        print(stats[stats["h"] == h][cols].to_string(index=False))

    print("\n=== 20日 OOS 相对行业 ===")
    print(stats[(stats["h"] == 20) & (stats["period"] == "oos_2018_2026")][
        ["entry", "n", "mean_ex_ind", "median_ex_ind", "win_ex_ind"]].to_string(index=False))


if __name__ == "__main__":
    main()
