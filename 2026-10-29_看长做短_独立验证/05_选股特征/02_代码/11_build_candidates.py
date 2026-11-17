#!/usr/bin/env python3
"""Join V2 signals to daily features, PIT industry and market context.

Outputs cand_feat.parquet with one row per candidate signal (pre-dedup),
including forward returns and industry/market benchmarks.
"""
from __future__ import annotations

import os
import time
import duckdb
import numpy as np
import pandas as pd

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证"
WORK = f"{BASE}/02_代码/work.duckdb"
FROZEN = "/Users/admin/WorkSpace/trading/trading_趋势结构/01_数据/frozen/frozen.duckdb"
IND = "/Users/admin/WorkSpace/trading/2026-10-18_成交量融合_2010-2025/01_数据/申万行业分类_时点.csv"
SEL = f"{BASE}/05_选股特征/02_代码/selection.duckdb"
OUT = f"{BASE}/05_选股特征/02_代码/cand_feat.parquet"

FEATURES = [
    "ret5", "ret10", "ret20", "ret60", "ret120", "ret250",
    "ma20_slope20", "ma60_slope20", "ma120_slope20",
    "vol20", "vol60", "vol250", "downvol20", "atr20n", "atr60n",
    "beta60", "beta250", "r2mkt60", "resid_vol60", "skew60", "kurt60", "maxret20", "minret20",
    "trend_r2_60", "trend_r2_120", "trend_slope_60", "trend_slope_120",
    "autocorr60", "vr5", "vr20",
    "obv_slope20", "vol_ratio5_20", "vol_ratio20_60",
    "updown_vol20", "updown_vol60", "pullback_vol_ratio",
    "pullback_depth", "pullback_depth_atr", "retrace_ratio",
    "hh20_ratio", "prior_runup20", "prior_runup60",
    "days_since_hh20", "days_since_hh60", "days_since_hh250",
    "dist_hh60", "dd250", "new_high_60", "new_high_250", "box_range60",
    "close_loc", "body", "upper_shadow", "lower_shadow",
    "signal_day_ret", "signal_day_turn_ratio", "signal_day_gap", "vol_price_corr20", "vwap20", "vwap20_dev",
    "price_efficiency60", "ma_alignment", "ma20_ma60_spread", "ma60_ma120_spread",
    "dist_ma20", "dist_ma60", "dist_ma120", "dist_ma250",
    "up_ratio20", "up_ratio60", "st_days250",
    "limit_up20", "limit_up60", "limit_down20", "limit_down60", "gap_abs20",
    "log_mcap", "log_amt20", "log_amt60", "amt_trend", "amihud20",
    "turn20", "turn60", "turn_pct250", "amt_pct250", "vol_pct250",
    "pe_neg", "log_pe", "log_pb", "pe_med_ratio250", "pb_med_ratio250",
    "age_years", "board",
    "vol_of_vol60", "max_dd_250", "ulcer250", "ma20_touch_count10",
    "board_main", "board_gem", "board_star",
    "mkt_ret20", "mkt_ret60", "mkt_ret120", "mkt_vol20", "mkt_dd250", "mkt_gate",
]

# features derived in SQL that are not in d yet
EXTRA = {
    "resid_vol60": "vol60 * sqrt(greatest(0, 1 - r2mkt60))",
}


def q(con, sql, label=""):
    t0 = time.time()
    con.execute(sql)
    print(f"[{time.time()-t0:7.1f}s] {label or sql[:80]}", flush=True)


def main() -> None:
    con = duckdb.connect(SEL)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='12GB'")
    con.execute(f"PRAGMA temp_directory='{BASE}/05_选股特征/02_代码/tmp'")
    con.execute(f"ATTACH '{WORK}' AS w (READ_ONLY)")
    con.execute(f"ATTACH '{FROZEN}' AS f (READ_ONLY)")

    # add the one derived feature
    cols = [r[0] for r in con.execute("describe d").fetchall()]
    if "resid_vol60" not in cols:
        q(con, "ALTER TABLE d ADD COLUMN resid_vol60 DOUBLE", "add resid_vol60")
        q(con, "UPDATE d SET resid_vol60 = vol60 * sqrt(greatest(0, 1 - r2mkt60))", "fill resid_vol60")

    for t in ["ind_map","ind_px","ind_daily","ind_feat","ind_ctx","hs300","cand_base"]:
        con.execute(f"DROP TABLE IF EXISTS {t}")

    # ---------- industry point-in-time ----------
    q(con, f"""
        CREATE TABLE ind_map AS
        SELECT code, CAST(start_date AS DATE) AS start_date, industry, sw_index
        FROM read_csv_auto('{IND}', header=true)
        WHERE industry IS NOT NULL AND industry <> ''
        ORDER BY code, start_date
    """, "ind_map")
    print("ind_map", con.execute("select count(*), count(distinct code), count(distinct industry) from ind_map").fetchone(), flush=True)

    q(con, """
        CREATE TABLE ind_px AS
        SELECT p.code, p.date, p.ret, p.c, p.ma20, p.is_st, p.status, m.industry
        FROM (SELECT code, date, ret, c, ma20, is_st, status FROM d WHERE date >= DATE '2012-01-01') p
        ASOF LEFT JOIN ind_map m
          ON p.code = m.code AND p.date >= m.start_date
    """, "ind_px")
    print("ind_px", con.execute("select count(*), count(industry) from ind_px").fetchone(), flush=True)

    q(con, """
        CREATE TABLE ind_daily AS
        SELECT date, industry,
               avg(ret) AS ind_ret,
               count(*) AS n,
               avg(CASE WHEN ma20 IS NULL THEN NULL WHEN c > ma20 THEN 1.0 ELSE 0.0 END) AS breadth20
        FROM ind_px
        WHERE industry IS NOT NULL AND ret IS NOT NULL AND is_st = 0 AND status = 1
        GROUP BY 1, 2
    """, "ind_daily")
    print("ind_daily", con.execute("select count(*), count(distinct industry), min(date), max(date) from ind_daily").fetchone(), flush=True)

    q(con, """
        CREATE TABLE ind_feat AS
        SELECT date, industry, ind_ret, n, breadth20,
               exp(sum(ln(1 + ind_ret)) OVER w20) - 1 AS ind_ret20,
               exp(sum(ln(1 + ind_ret)) OVER w60) - 1 AS ind_ret60,
               exp(sum(ln(1 + ind_ret)) OVER w120) - 1 AS ind_ret120,
               stddev_samp(ind_ret) OVER w20 * sqrt(252) AS ind_vol20,
               CASE WHEN count(ind_ret) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) = 20
                    THEN exp(sum(ln(1 + ind_ret)) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING)) - 1 END AS ind_fwd20,
               CASE WHEN count(ind_ret) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 40 FOLLOWING) = 40
                    THEN exp(sum(ln(1 + ind_ret)) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 40 FOLLOWING)) - 1 END AS ind_fwd40,
               CASE WHEN count(ind_ret) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 60 FOLLOWING) = 60
                    THEN exp(sum(ln(1 + ind_ret)) OVER (PARTITION BY industry ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 60 FOLLOWING)) - 1 END AS ind_fwd60
        FROM ind_daily
        WINDOW
          w20 AS (PARTITION BY industry ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
          w60 AS (PARTITION BY industry ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),
          w120 AS (PARTITION BY industry ORDER BY date ROWS BETWEEN 119 PRECEDING AND CURRENT ROW)
    """, "ind_feat")
    print("ind_feat", con.execute("select count(*), count(ind_fwd20), count(ind_fwd60) from ind_feat").fetchone(), flush=True)

    # industry momentum rank and excess vs market
    q(con, """
        CREATE TABLE ind_ctx AS
        SELECT i.*,
               percent_rank() OVER (PARTITION BY date ORDER BY ind_ret20) AS ind_mom_rank,
               ind_ret20 - m.mkt_ret20 AS ind_ex_ew20
        FROM ind_feat i
        LEFT JOIN market m USING (date)
    """, "ind_ctx")

    # hs300 gate
    q(con, """
        CREATE TABLE hs300 AS
        SELECT CAST(date AS DATE) AS date, close,
               avg(close) OVER (ORDER BY CAST(date AS DATE) ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ma200,
               CASE WHEN close > avg(close) OVER (ORDER BY CAST(date AS DATE) ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) THEN 1 ELSE 0 END AS hs300_gate
        FROM w.indices
        WHERE code = 'sh.000300'
    """, "hs300")

    # ---------- candidate feature table ----------
    fcols = ", ".join([f"d.{c}" for c in FEATURES])
    q(con, f"""
        CREATE TABLE cand_base AS
        WITH cm AS (
            SELECT c.*, m.industry
            FROM w.candidates c
            ASOF LEFT JOIN ind_map m
              ON c.code = m.code AND c.date >= m.start_date
        )
        SELECT cm.*,
               {fcols},
               i.ind_ret20, i.ind_ret60, i.ind_ret120, i.ind_vol20,
               i.breadth20 AS ind_breadth20, i.ind_mom_rank, i.ind_ex_ew20,
               i.ind_fwd20, i.ind_fwd40, i.ind_fwd60,
               b.pct_above_ma20 AS market_breadth20,
               b.pct_above_ma60 AS market_breadth60,
               h.hs300_gate
        FROM cm
        JOIN d ON cm.code = d.code AND cm.date = d.date
        LEFT JOIN ind_ctx i ON cm.date = i.date AND cm.industry = i.industry
        LEFT JOIN f.market_daily b ON cm.date = b.date
        LEFT JOIN hs300 h ON cm.date = h.date
    """, "cand_base")
    print("cand_base", con.execute("select count(*), count(industry), count(ind_fwd20), count(hs300_gate) from cand_base").fetchone(), flush=True)

    df = con.execute("select * from cand_base").fetchdf()
    print("loaded", df.shape, flush=True)

    # forward equal-weight benchmark: nav[t+20]/nav[t+1]-1, using calendar index
    ew = con.execute("select date, nav, cal_idx from w.ew_nav order by date").fetchdf()
    con.close()
    max_idx = int(ew["cal_idx"].max())
    arr = np.full(max_idx + 2, np.nan)
    arr[ew["cal_idx"].values.astype(int)] = ew["nav"].values
    ci = df["cal_idx"].values.astype(int)
    for h in (20, 40, 60):
        entry = np.where(ci + 1 <= max_idx, arr[np.minimum(ci + 1, max_idx)], np.nan)
        exit_ = np.where(ci + h <= max_idx, arr[np.minimum(ci + h, max_idx)], np.nan)
        df[f"ew_fwd{h}"] = exit_ / entry - 1.0

    df["r20"] = df["c20"] / df["next_open"] - 1.0
    df["r40"] = df["c40"] / df["next_open"] - 1.0
    df["r60"] = df["c60"] / df["next_open"] - 1.0
    df["mfe"] = df["mfe20"] / df["next_open"] - 1.0
    df["mae"] = df["mae20"] / df["next_open"] - 1.0
    for h in (20, 40, 60):
        df[f"ex{h}"] = df[f"r{h}"] - df[f"ew_fwd{h}"]
    df["ex_ind20"] = df["r20"] - df["ind_fwd20"]
    df["ex_ind40"] = df["r40"] - df["ind_fwd40"]
    df["ex_ind60"] = df["r60"] - df["ind_fwd60"]
    df["ex_beta20"] = df["r20"] - df["beta60"] * df["ew_fwd20"]
    df["stock_ind_ex20"] = df["ret20"] - df["ind_ret20"]
    df["stock_ind_ex60"] = df["ret60"] - df["ind_ret60"]
    df["period"] = np.where(df["date"].dt.year <= 2017, "dev_2012_2017", "oos_2018_2026")

    # audit missingness
    print("\nmissing top:")
    print(df[FEATURES + ["industry", "ind_fwd20", "market_breadth20"]].isna().mean().sort_values(ascending=False).head(25).to_string(), flush=True)
    print("\ntargets:", df[["r20", "ex20", "ex_ind20", "ex_beta20"]].describe().to_string(), flush=True)

    df.to_parquet(OUT, index=False)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
