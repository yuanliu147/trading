#!/usr/bin/env python3
"""04 形态事件 × 前向收益（H3 主检验）

事件（全部为 t 日收盘可得，横盘/压缩/缩量取 t-1，突破与放量取 t）：
  1 横盘游程 ≥N（箱宽20 ≤15% 的连续天数）
  2 压缩  ATR20/ATR250 ≤ q
  3 缩量  V20/V250 ≤ v
  4 突破  收盘 > 前20日最高价
  5 放量  当日量 / V20 ≥ m
  6 位置  收盘 / 250日最高收盘 ≥ p（p=0 表示不限）

入场 T+1 开盘，T+1 停牌或一字涨停剔除；超额=个股 − 同日等权基准。
"""
from __future__ import annotations

import itertools
import os

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(HERE, "02_代码", "work.duckdb")
OUT = os.path.join(HERE, "02_代码")

IS = ("2006-01-01", "2015-12-31")
OOS = ("2016-01-01", "2026-10-29")


def nw_t(x: np.ndarray, lag: int = 5) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    mu = x.mean()
    e = x - mu
    s = e @ e / n
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        s += 2 * w * (e[l:] @ e[:-l]) / n
    if s <= 0:
        return np.nan
    return mu / np.sqrt(s / n)


def prep(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE OR REPLACE TABLE pre AS
    SELECT code, date, close, hh20_prev, vwap, amt20, box_run, compress, vshr, pos250, vr,
           trade_status, is_st, bar_no,
           lag(box_run, 1) OVER w AS box_prev,
           lag(compress, 1) OVER w AS comp_prev,
           lag(vshr, 1) OVER w AS vshr_prev
    FROM feat_box
    WINDOW w AS (PARTITION BY code ORDER BY date)
    """)


def events(con, N, q, v, m, p) -> pd.DataFrame:
    return con.execute(f"""
        SELECT p.code, p.date, f.ret20, f.ret60, f.ret120
        FROM pre p
        JOIN fwd_ret f USING (code, date)
        JOIN bench b USING (date)
        WHERE p.box_prev >= {N}
          AND p.comp_prev <= {q}
          AND p.vshr_prev <= {v}
          AND p.close > p.hh20_prev
          AND p.vr >= {m}
          AND p.pos250 >= {p}
          AND p.trade_status = 1 AND p.is_st = 0 AND p.bar_no >= 120
          AND p.vwap >= 3 AND p.amt20 >= 3e7
          AND f.entry_ok AND f.ret60 IS NOT NULL
    """).df()


def summarize(df: pd.DataFrame, label: dict) -> dict:
    out = dict(label)
    out["n"] = len(df)
    out["days"] = df["date"].nunique()
    if len(df) == 0:
        return out
    d = df
    daily = d.groupby("date")["ex60"].mean()
    out["ex20"] = d["ex20"].mean()
    out["ex60"] = d["ex60"].mean()
    out["ex120"] = d["ex120"].mean()
    out["med_ex60"] = d["ex60"].median()
    out["win60"] = (d["ex60"] > 0).mean()
    out["abs60"] = d["ret60"].mean()
    out["nw_t60"] = nw_t(daily.to_numpy())
    out["ex60_all"] = d["ex60_all"].mean()
    out["nw_t60_all"] = nw_t(d.groupby("date")["ex60_all"].mean().to_numpy())
    return out


def main() -> None:
    con = duckdb.connect(DB)
    con.execute("PRAGMA threads=8")
    prep(con)
    con.execute("""
        CREATE OR REPLACE VIEW ev AS
        SELECT p.code, p.date, p.close, p.hh20_prev, p.vr, p.pos250,
               p.box_prev, p.comp_prev, p.vshr_prev,
               f.ret20, f.ret60, f.ret120,
               f.ret20 - bf.b20 AS ex20, f.ret60 - bf.b60 AS ex60, f.ret120 - bf.b120 AS ex120,
               f.ret20 - b.b20 AS ex20_all, f.ret60 - b.b60 AS ex60_all, f.ret120 - b.b120 AS ex120_all
        FROM pre p
        JOIN fwd_ret f USING (code, date)
        JOIN bench b USING (date)
        JOIN bench_f bf USING (date)
        WHERE p.trade_status = 1 AND p.is_st = 0 AND p.bar_no >= 120
          AND p.vwap >= 3 AND p.amt20 >= 3e7
          AND f.entry_ok AND f.ret60 IS NOT NULL
    """)

    rows = []
    grid = list(itertools.product([30, 60, 90], [0.5, 0.7, 1.0], [0.8, 1.0], [1.5, 2.0], [0.0, 0.85]))
    for N, q, v, m, p in grid:
        base = f"""
            SELECT * FROM ev
            WHERE box_prev >= {N} AND comp_prev <= {q} AND vshr_prev <= {v}
              AND close > hh20_prev AND vr >= {m} AND pos250 >= {p}
        """
        df = con.execute(base).df()
        for period, (d0, d1) in [("全样本", ("2000-01-01", "2026-10-29")),
                                 ("样本内06-15", IS), ("样本外16-26", OOS)]:
            sub = df[(df["date"] >= d0) & (df["date"] <= d1)]
            rows.append(summarize(sub, {
                "N": N, "q": q, "v": v, "m": m, "p": p, "period": period}))
        print(N, q, v, m, p, len(df), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "h3_grid.csv"), index=False)
    print(res[res.period == "全样本"].sort_values("nw_t60", ascending=False).head(15).to_string())
    print(res[res.period == "样本外16-26"].sort_values("nw_t60", ascending=False).head(15).to_string())
    con.close()


if __name__ == "__main__":
    main()
