#!/usr/bin/env python3
"""Signal-day profile of the four main entries (verification period)."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/06_入场变体"
SEL = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/05_选股特征/02_代码/selection.duckdb"
OUT = f"{BASE}/03_结果/买点信号画像.csv"


def dedup_20(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    keep = np.zeros(len(df), dtype=bool)
    last: dict[str, int] = {}
    for i, code, ci in zip(df.index.values, df["code"].values, df["cal_idx"].values):
        if code not in last or ci - last[code] >= 20:
            keep[i] = True
            last[code] = ci
    return df.loc[keep].copy()


def main() -> None:
    df = pd.read_parquet(f"{BASE}/02_代码/entries.parquet")
    df["date"] = pd.to_datetime(df["date"])
    base = df["base_ok"] & df["trend"] & df["tradable"] & df["wok"] & df["mok"]
    sets = {
        "E0": base & df["pullback"] & df["reclaim"],
        "E1": base & df["e1"],
        "E2": base & df["e2"],
        "E3": base & df["e3"],
    }
    rows = []
    for k, f in sets.items():
        g = dedup_20(df.loc[f])
        g = g[g["period"] == "oos_2018_2026"][["code", "date"]].copy()
        g["set"] = k
        rows.append(g)
    s = pd.concat(rows)

    con = duckdb.connect(SEL, read_only=True)
    con.register("sig", s)
    out = con.execute("""
        SELECT s.set, count(*) AS n,
               median(d.vol60) AS vol60, median(d.beta60) AS beta60,
               median(d.log_mcap) AS log_mcap, median(d.amt20) AS amt20,
               median(d.prior_runup60) AS runup60, median(d.box_range60) AS box60,
               median(d.dist_ma20) AS dist_ma20, median(d.pb_med_ratio250) AS pb,
               median(d.ma_alignment) AS ma_align, median(d.age_years) AS age
        FROM sig s JOIN d ON s.code = d.code AND s.date = d.date
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    con.close()

    out["mcap_yi"] = np.exp(out["log_mcap"]) / 1e8
    out["amt_yi"] = out["amt20"] / 1e8
    out = out[["set", "n", "vol60", "beta60", "mcap_yi", "amt_yi", "runup60", "box60",
               "dist_ma20", "pb", "ma_align", "age"]]
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
