#!/usr/bin/env python3
"""Split portfolio results into dev / OOS / full, with benchmarks and yearly returns."""
from __future__ import annotations

import glob
import math
import os
import re
import duckdb
import numpy as np
import pandas as pd

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证"
SEL = f"{BASE}/05_选股特征"
OUT = f"{SEL}/03_结果"
WORK = f"{BASE}/02_代码/work.duckdb"


def metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    years = len(nav) / 252.0
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    mdd = (nav / nav.cummax() - 1).min()
    sharpe = ret.mean() / ret.std(ddof=1) * math.sqrt(252) if ret.std(ddof=1) > 0 else np.nan
    return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe,
            "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
            "final": nav.iloc[-1], "days": len(nav)}


def main() -> None:
    con = duckdb.connect(WORK, read_only=True)
    ew = con.execute("SELECT CAST(date AS DATE) AS date, nav FROM ew_nav ORDER BY 1").fetchdf()
    idx = con.execute("""SELECT code, CAST(date AS DATE) AS date, close FROM indices ORDER BY 1,2""").fetchdf()
    con.close()
    ew["date"] = pd.to_datetime(ew["date"])
    ew = ew.set_index("date")["nav"]
    idx["date"] = pd.to_datetime(idx["date"])
    idxp = idx.pivot(index="date", columns="code", values="close")

    periods = {
        "dev_2012_2017": (pd.Timestamp("2012-01-03"), pd.Timestamp("2017-12-29")),
        "oos_2018_2026": (pd.Timestamp("2018-01-02"), pd.Timestamp("2026-10-28")),
        "full": (pd.Timestamp("2012-01-03"), pd.Timestamp("2026-10-28")),
    }
    bench = {}
    for p, (a, b) in periods.items():
        bench[("EW", p)] = metrics(ew.loc[a:b] / ew.loc[a:b].iloc[0])
        for c in idxp.columns:
            s = idxp[c].loc[a:b].ffill()
            bench[(c, p)] = metrics(s / s.iloc[0])

    rows = []
    yearly = []
    for f in sorted(glob.glob(f"{OUT}/组合_选股_*_权益.csv")):
        tag = os.path.basename(f).replace("组合_选股_", "").replace("_权益.csv", "")
        if tag.startswith("汇总"):
            continue
        eq = pd.read_csv(f, parse_dates=["date"]).set_index("date")["equity"]
        trf = f.replace("_权益.csv", "_逐笔.csv")
        tr = pd.read_csv(trf, parse_dates=["entry_date", "exit_date"]) if os.path.exists(trf) else pd.DataFrame()
        for p, (a, b) in periods.items():
            s = eq.loc[a:b]
            if len(s) < 20:
                continue
            m = metrics(s / s.iloc[0])
            sub = tr[(tr["exit_date"] >= a) & (tr["exit_date"] <= b)] if len(tr) else tr
            if len(sub):
                pf = sub.loc[sub.ret > 0, "ret"].sum() / abs(sub.loc[sub.ret < 0, "ret"].sum()) if (sub.ret < 0).any() else np.nan
                ts = {"trades": len(sub), "win_rate": (sub.ret > 0).mean(), "avg_ret": sub.ret.mean(),
                      "median_ret": sub.ret.median(), "avg_hold": sub.hold_days.mean(), "profit_factor": pf}
            else:
                ts = {"trades": 0}
            rows.append({"tag": tag, "period": p, **m, **ts,
                         "EW_CAGR": bench[("EW", p)]["CAGR"], "EW_MDD": bench[("EW", p)]["MDD"],
                         "CSI500_CAGR": bench[("sh.000905", p)]["CAGR"],
                         "CSI1000_CAGR": bench[("sh.000852", p)]["CAGR"],
                         "excess_EW": m["CAGR"] - bench[("EW", p)]["CAGR"]})
        yr = eq.groupby(eq.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
        yearly.append({"tag": tag, **{str(k): v for k, v in yr.items()}})

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/组合_选股_分期.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(f"{OUT}/组合_选股_年度.csv", index=False, encoding="utf-8-sig")

    print("=== OOS ===")
    q = df[df.period == "oos_2018_2026"].sort_values("CAGR", ascending=False)
    print(q[["tag", "CAGR", "MDD", "Sharpe", "trades", "win_rate", "avg_ret", "excess_EW"]].to_string(index=False))
    print("\n=== DEV ===")
    q = df[df.period == "dev_2012_2017"].sort_values("CAGR", ascending=False)
    print(q[["tag", "CAGR", "MDD", "Sharpe", "trades", "win_rate", "avg_ret", "excess_EW"]].to_string(index=False))
    print("\n=== benchmarks OOS ===")
    print({k: v for k, v in bench.items() if k[1] == "oos_2018_2026"})


if __name__ == "__main__":
    main()
