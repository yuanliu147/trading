#!/usr/bin/env python3
"""Build the pre-registered entry variants (E0/E1/E1V/E2/E3) on the full trend universe.

The V2 `candidates` table only contains pullback+reclaim days, so it cannot be
used to test alternative entries. This script rebuilds the universe from raw
daily bars: every day on which the daily trend filter holds, with the monthly
and weekly filters attached point-in-time via ASOF joins. The reconstructed
monthly/weekly flags are audited against the V2 candidate table.
"""
from __future__ import annotations

import os
import time

import duckdb
import numpy as np
import pandas as pd

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证"
WORK = f"{BASE}/02_代码/work.duckdb"
SEL = f"{BASE}/05_选股特征/02_代码/selection.duckdb"
IND = "/Users/admin/WorkSpace/trading/2026-10-18_成交量融合_2010-2025/01_数据/申万行业分类_时点.csv"
HERE = f"{BASE}/06_入场变体"
DB = f"{HERE}/02_代码/entries.duckdb"
OUT = f"{HERE}/02_代码/entries.parquet"
AUDIT = f"{HERE}/03_结果/入场_构建审计.csv"


def q(con, sql, label=""):
    t0 = time.time()
    con.execute(sql)
    print(f"[{time.time()-t0:7.1f}s] {label}", flush=True)


def main() -> None:
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    os.makedirs(os.path.dirname(AUDIT), exist_ok=True)
    if os.path.exists(DB):
        os.remove(DB)
    con = duckdb.connect(DB)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='16GB'")
    con.execute(f"PRAGMA temp_directory='{HERE}/02_代码/tmp'")
    con.execute(f"ATTACH '{WORK}' AS w (READ_ONLY)")
    con.execute(f"ATTACH '{SEL}' AS s (READ_ONLY)")

    # ---------- rolling features (preceding window) + forward closes ----------
    q(con, """
        CREATE TABLE px AS
        SELECT * FROM (
            SELECT code, date, hfq_open, hfq_high, hfq_low, hfq_close,
                   raw_close, raw_preclose, volume, amount, turn,
                   trade_status, is_st, age, amt20,
                   ma5, ma10, ma20, ma60, ma60_20ago,
                   next_open, next_raw_open, next_raw_preclose, next_status, next_date,
                   max(hfq_high) OVER w20p AS hh20_prev,
                   max(hfq_high) OVER w40p AS hh40_prev,
                   min(hfq_low)  OVER w40p AS ll40_prev,
                   avg(volume)   OVER w20p AS vma20_prev,
                   min(hfq_low)  OVER w3   AS low3,
                   max(hfq_high) OVER w20c AS hh20,
                   lead(hfq_close, 5)  OVER wf AS c5,
                   lead(hfq_close, 10) OVER wf AS c10,
                   lead(hfq_close, 20) OVER wf AS c20,
                   max(hfq_high) OVER w20f AS fwd_max_high,
                   min(hfq_low)  OVER w20f AS fwd_min_low
            FROM w.daily_feat
            WHERE date >= DATE '2011-01-01'
            WINDOW
              w20p AS (PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
              w40p AS (PARTITION BY code ORDER BY date ROWS BETWEEN 40 PRECEDING AND 1 PRECEDING),
              w20c AS (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
              w3   AS (PARTITION BY code ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
              w20f AS (PARTITION BY code ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING),
              wf   AS (PARTITION BY code ORDER BY date)
        ) t
        WHERE date >= DATE '2012-01-04'
    """, "px")
    print("px", con.execute("select count(*), count(c20) from px").fetchone(), flush=True)

    # ---------- point-in-time monthly / weekly (strictly completed) ----------
    q(con, """
        CREATE TABLE u AS
        SELECT p.*,
               wk.close AS w_close, wk.ma20w, wk.ma60w, wk.ma60w_4ago,
               mo.close AS m_close, mo.ma12, mo.ma12_prev,
               (wk.close > wk.ma20w AND wk.ma20w > wk.ma60w AND wk.ma60w > wk.ma60w_4ago) AS wok,
               (mo.close > mo.ma12 AND mo.ma12 > mo.ma12_prev) AS mok,
               CASE WHEN p.is_st = 1 THEN 5
                    WHEN p.code LIKE 'sh.68%' THEN 20
                    WHEN p.code LIKE 'sz.30%' AND p.date >= DATE '2020-08-24' THEN 20
                    ELSE 10 END AS limit_pct
        FROM px p
        ASOF LEFT JOIN w.weekly wk ON p.code = wk.code AND p.date > CAST(wk.week_end AS DATE)
        ASOF LEFT JOIN w.monthly mo ON p.code = mo.code AND p.date > CAST(mo.month_end AS DATE)
    """, "u+weekly+monthly")
    con.execute("DROP TABLE px")

    # audit the reconstructed filters against the V2 candidate table
    q(con, """
        CREATE TABLE audit AS
        SELECT count(*) AS n,
               sum(CASE WHEN u.wok = c.wok THEN 0 ELSE 1 END) AS wok_mismatch,
               sum(CASE WHEN u.mok = c.mok THEN 0 ELSE 1 END) AS mok_mismatch,
               sum(CASE WHEN c.wok IS NULL THEN 1 ELSE 0 END) AS wok_null,
               sum(CASE WHEN c.mok IS NULL THEN 1 ELSE 0 END) AS mok_null
        FROM w.candidates c
        JOIN u ON c.code = u.code AND c.date = u.date
    """, "audit")
    aud = con.execute("SELECT * FROM audit").fetchdf()
    print(aud.to_string(index=False), flush=True)
    aud.to_csv(AUDIT, index=False, encoding="utf-8-sig")

    # ---------- trend universe + entries ----------
    q(con, """
        CREATE TABLE entries AS
        SELECT *,
               (hfq_close > ma20 AND ma20 > ma60 AND ma60 > ma60_20ago) AS trend,
               (is_st = 0 AND age >= 250 AND amt20 >= 20000000 AND raw_close >= 2
                AND trade_status = 1
                AND raw_close < raw_preclose * (1 + limit_pct / 100.0) - 0.001) AS base_ok,
               ((hh20 - hfq_low) / hh20 >= 0.03 AND hfq_low <= ma20 * 1.01) AS pullback,
               (hfq_close > ma5 AND hfq_close > hfq_open) AS reclaim,
               (hfq_close > hh20_prev AND hfq_close > hfq_open) AS e1,
               (hfq_close > hh20_prev AND hfq_close > hfq_open
                AND volume >= 1.2 * vma20_prev) AS e1v,
               (low3 <= ma10 * 1.01 AND hfq_close > ma5 AND hfq_close > hfq_open) AS e2,
               (ll40_prev > 0 AND (hh40_prev - ll40_prev) / ll40_prev <= 0.20
                AND hfq_close > hh40_prev AND hfq_close > hfq_open) AS e3
        FROM u
        WHERE hfq_close > ma20 AND ma20 > ma60 AND ma60 > ma60_20ago
    """, "entries")
    con.execute("DROP TABLE u")
    print("entries", con.execute("select count(*), count(distinct code) from entries").fetchone(), flush=True)

    # ---------- industry and market context ----------
    q(con, f"""
        CREATE TABLE ind_map AS
        SELECT code, CAST(start_date AS DATE) AS start_date, industry
        FROM read_csv_auto('{IND}', header=true)
        WHERE industry IS NOT NULL AND industry <> ''
    """, "ind_map")
    q(con, "CREATE TABLE ind_daily AS SELECT * FROM s.ind_daily", "ind_daily")
    q(con, """
        CREATE TABLE ind_fwd AS
        SELECT date, industry,
               CASE WHEN count(ind_ret) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING)=5
                    THEN exp(sum(ln(1+ind_ret)) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING))-1 END AS ind_fwd5,
               CASE WHEN count(ind_ret) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 10 FOLLOWING)=10
                    THEN exp(sum(ln(1+ind_ret)) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 10 FOLLOWING))-1 END AS ind_fwd10,
               CASE WHEN count(ind_ret) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING)=20
                    THEN exp(sum(ln(1+ind_ret)) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING))-1 END AS ind_fwd20
        FROM ind_daily
    """, "ind_fwd")
    q(con, "CREATE TABLE mkt AS SELECT CAST(date AS DATE) AS date, mkt_gate, ew_ma200 FROM s.market", "mkt")

    q(con, """
        CREATE TABLE ent2 AS
        WITH x AS (
            SELECT e.*, m.industry
            FROM entries e
            ASOF LEFT JOIN ind_map m ON e.code = m.code AND e.date >= m.start_date
        )
        SELECT x.*, i.ind_fwd5, i.ind_fwd10, i.ind_fwd20, k.mkt_gate, k.ew_ma200
        FROM x
        LEFT JOIN ind_fwd i ON x.date = i.date AND x.industry = i.industry
        LEFT JOIN mkt k ON x.date = k.date
    """, "ent2")
    con.execute("DROP TABLE entries")

    df = con.execute("SELECT * FROM ent2 WHERE base_ok").fetchdf()
    print("loaded base_ok", df.shape, flush=True)
    con.close()

    # ---------- execution ----------
    df["limit_up_open"] = df["next_raw_open"] >= df["next_raw_preclose"] * (1 + df["limit_pct"] / 100.0) - 0.001
    df["gap"] = df["next_open"] / df["hfq_close"] - 1.0
    df["tradable"] = (df["next_status"] == 1) & (~df["limit_up_open"]) & (df["gap"] <= 0.05)

    for h in (5, 10, 20):
        df[f"r{h}"] = df[f"c{h}"] / df["next_open"] - 1.0
    df["mfe"] = df["fwd_max_high"] / df["next_open"] - 1.0
    df["mae"] = df["fwd_min_low"] / df["next_open"] - 1.0

    ew = duckdb.connect(WORK, read_only=True).execute(
        "SELECT date, nav, cal_idx FROM ew_nav ORDER BY date").fetchdf()
    max_idx = int(ew["cal_idx"].max())
    arr = np.full(max_idx + 2, np.nan)
    arr[ew["cal_idx"].values.astype(int)] = ew["nav"].values

    # cal_idx for entries
    cal = duckdb.connect(WORK, read_only=True).execute(
        "SELECT CAST(date AS DATE) AS date, cal_idx FROM calendar_idx").fetchdf()
    cal["date"] = pd.to_datetime(cal["date"])
    df = df.merge(cal, on="date", how="left")
    ci = df["cal_idx"].values.astype(int)
    for h in (5, 10, 20):
        entry = np.where(ci + 1 <= max_idx, arr[np.minimum(ci + 1, max_idx)], np.nan)
        exit_ = np.where(ci + h <= max_idx, arr[np.minimum(ci + h, max_idx)], np.nan)
        df[f"ew_fwd{h}"] = exit_ / entry - 1.0
        df[f"ex{h}"] = df[f"r{h}"] - df[f"ew_fwd{h}"]
        df[f"ex_ind{h}"] = df[f"r{h}"] - df[f"ind_fwd{h}"]

    df["period"] = np.where(df["date"].dt.year <= 2017, "dev_2012_2017", "oos_2018_2026")
    df["year"] = df["date"].dt.year
    df["e1d"] = df["e1"]
    df["e2d"] = df["e2"]

    cols = ["code", "date", "year", "period", "cal_idx", "hfq_open", "hfq_close", "hfq_low",
            "ma5", "ma10", "ma20", "ma60", "hh20", "hh20_prev", "hh40_prev", "ll40_prev",
            "volume", "vma20_prev", "low3",
            "next_open", "next_raw_open", "next_raw_preclose", "next_status", "next_date",
            "limit_pct", "c5", "c10", "c20",
            "r5", "r10", "r20", "ew_fwd5", "ew_fwd10", "ew_fwd20",
            "ex5", "ex10", "ex20", "ind_fwd5", "ind_fwd10", "ind_fwd20",
            "ex_ind5", "ex_ind10", "ex_ind20", "mfe", "mae",
            "amt20", "raw_close", "is_st", "trade_status", "age",
            "trend", "wok", "mok", "base_ok", "pullback", "reclaim",
            "e1", "e1v", "e2", "e3", "e1d", "e2d", "tradable", "gap",
            "industry", "mkt_gate", "ew_ma200"]
    df[cols].to_parquet(OUT, index=False)
    print("saved", OUT, df.shape, flush=True)

    for k in ["pullback", "e1", "e1v", "e2", "e3"]:
        t = df[(df["wok"]) & (df["mok"]) & (df[k]) & (df["tradable"])]
        if k == "pullback":
            t = t[t["reclaim"]]
        print(k, len(t), t["date"].dt.year.min(), t["date"].dt.year.max(), flush=True)


if __name__ == "__main__":
    main()
