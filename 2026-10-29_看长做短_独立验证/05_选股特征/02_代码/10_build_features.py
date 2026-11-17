#!/usr/bin/env python3
"""Build signal-day feature panel for the 看长做短 selection layer.

Data sources (raw / point-in-time only):
  - baostock SQLite bars already materialised in work.duckdb (daily)
  - frozen.duckdb: float mktcap, board, PE/PB, limit flags, market breadth
  - 申万一级行业时点分类 CSV
No existing backtest result is read.
"""
from __future__ import annotations

import os
import time
import duckdb

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证"
WORK = f"{BASE}/02_代码/work.duckdb"
FROZEN = "/Users/admin/WorkSpace/trading/trading_趋势结构/01_数据/frozen/frozen.duckdb"
IND = "/Users/admin/WorkSpace/trading/2026-10-18_成交量融合_2010-2025/01_数据/申万行业分类_时点.csv"
OUTDB = f"{BASE}/05_选股特征/02_代码/selection.duckdb"


def q(con, sql, label=""):
    t0 = time.time()
    con.execute(sql)
    print(f"[{time.time()-t0:7.1f}s] {label or sql[:80]}", flush=True)


def main() -> None:
    if os.path.exists(OUTDB):
        os.remove(OUTDB)
    con = duckdb.connect(OUTDB)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='12GB'")
    con.execute(f"PRAGMA temp_directory='{BASE}/05_选股特征/02_代码/tmp'")
    con.execute(f"ATTACH '{WORK}' AS w (READ_ONLY)")
    con.execute(f"ATTACH '{FROZEN}' AS f (READ_ONLY)")

    # ---------- point-in-time metadata ----------
    q(con, """
        CREATE TABLE funi AS
        SELECT lower(split_part(u.symbol, '.', 2)) || '.' || split_part(u.symbol, '.', 1) AS code,
               u.date,
               u.float_mktcap,
               u.board,
               u.td_since_list,
               u.is_limit_up_close,
               u.is_limit_down_close,
               u.is_limit_up_open,
               u.is_limit_down_open,
               u.is_st AS f_is_st,
               d.pe_ttm,
               d.pb_mrq
        FROM f.universe u
        LEFT JOIN f.daily d
          ON u.symbol = d.symbol AND u.date = d.date
        WHERE u.date >= DATE '2010-01-01'
    """, "funi")
    print(con.execute("select count(*), count(float_mktcap), count(pe_ttm), count(pb_mrq) from funi").fetchone(), flush=True)

    q(con, """
        CREATE TABLE ew AS
        SELECT date, nav,
               nav / lag(nav) OVER (ORDER BY date) - 1 AS ew_ret,
               row_number() OVER (ORDER BY date) AS ew_i
        FROM w.ew_nav
    """, "ew")
    q(con, """
        CREATE TABLE market AS
        SELECT e.date, e.nav, e.ew_ret,
               avg(e.nav) OVER (ORDER BY e.date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ew_ma200,
               e.nav / lag(e.nav, 5) OVER (ORDER BY e.date) - 1 AS mkt_ret5,
               e.nav / lag(e.nav, 20) OVER (ORDER BY e.date) - 1 AS mkt_ret20,
               e.nav / lag(e.nav, 60) OVER (ORDER BY e.date) - 1 AS mkt_ret60,
               e.nav / lag(e.nav, 120) OVER (ORDER BY e.date) - 1 AS mkt_ret120,
               stddev_samp(e.ew_ret) OVER (ORDER BY e.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) * sqrt(252) AS mkt_vol20,
               e.nav / max(e.nav) OVER (ORDER BY e.date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) - 1 AS mkt_dd250,
               max(e.nav) OVER (ORDER BY e.date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) AS ew_hh250,
               CASE WHEN e.nav > avg(e.nav) OVER (ORDER BY e.date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) THEN 1 ELSE 0 END AS mkt_gate
        FROM ew e
    """, "market")

    # ---------- daily price panel ----------
    q(con, """
        CREATE TABLE px AS
        SELECT d.code, d.date,
               d.hfq_open AS o, d.hfq_high AS h, d.hfq_low AS l, d.hfq_close AS c,
               d.raw_open AS ro, d.raw_high AS rh, d.raw_low AS rl,
               d.raw_close AS rc, d.raw_preclose AS rpc,
               d.volume AS v, d.amount AS amt, d.turn AS turn,
               d.trade_status AS status, d.is_st AS is_st, d.pct_chg AS pct,
               f.float_mktcap AS mcap, f.board AS board,
               f.td_since_list AS td_since_list,
               f.is_limit_up_close AS lu, f.is_limit_down_close AS ld,
               f.is_limit_up_open AS luo, f.is_limit_down_open AS ldo,
               f.pe_ttm AS pe, f.pb_mrq AS pb,
               d.hfq_close / lag(d.hfq_close) OVER (PARTITION BY d.code ORDER BY d.date) - 1 AS ret,
               row_number() OVER (PARTITION BY d.code ORDER BY d.date) AS t
        FROM w.daily d
        LEFT JOIN funi f ON d.code = f.code AND d.date = f.date
        WHERE d.date >= DATE '2010-01-01'
        ORDER BY d.code, d.date
    """, "px")
    print(con.execute("select count(*), count(ret), count(mcap), count(pe), min(date), max(date) from px").fetchone(), flush=True)

    # # ---------- stage B: raw rolling building blocks ----------
    q(con, """
        CREATE TABLE b AS
        SELECT p.*,
               avg(c) OVER w5  AS ma5,
               avg(c) OVER w10 AS ma10,
               avg(c) OVER w20 AS ma20,
               avg(c) OVER w60 AS ma60,
               avg(c) OVER w120 AS ma120,
               avg(c) OVER w250 AS ma250,
               max(h) OVER w20 AS hh20,
               max(h) OVER w60 AS hh60,
               max(h) OVER w250 AS hh250,
               min(l) OVER w60 AS ll60,
               min(l) OVER w250 AS ll250,
               avg(amt) OVER w20 AS amt20,
               avg(amt) OVER w60 AS amt60,
               avg(amt) OVER w120 AS amt120,
               avg(v) OVER w5 AS vma5,
               avg(v) OVER w20 AS vma20,
               avg(v) OVER w60 AS vma60,
               avg(turn) OVER w20 AS turn20,
               avg(turn) OVER w60 AS turn60,
               avg(h - l) OVER w20 AS atr20,
               avg(h - l) OVER w60 AS atr60,
               avg(abs(ret)) OVER w20 AS absret20,
               sum(abs(ret)) OVER w60 AS absret60sum,
               sum(CASE WHEN ret > 0 THEN 1 ELSE 0 END) OVER w20 AS updays20,
               sum(CASE WHEN ret > 0 THEN 1 ELSE 0 END) OVER w60 AS updays60,
               sum(CASE WHEN lu THEN 1 ELSE 0 END) OVER w20 AS lu20,
               sum(CASE WHEN lu THEN 1 ELSE 0 END) OVER w60 AS lu60,
               sum(CASE WHEN ld THEN 1 ELSE 0 END) OVER w20 AS ld20,
               sum(CASE WHEN ld THEN 1 ELSE 0 END) OVER w60 AS ld60,
               sum(CASE WHEN is_st = 1 THEN 1 ELSE 0 END) OVER w250 AS st250,
               sum(CASE WHEN ret > 0 THEN v ELSE 0 END) OVER w20 AS upv20,
               sum(CASE WHEN ret < 0 THEN v ELSE 0 END) OVER w20 AS dnv20,
               sum(CASE WHEN ret > 0 THEN v ELSE 0 END) OVER w60 AS upv60,
               sum(CASE WHEN ret < 0 THEN v ELSE 0 END) OVER w60 AS dnv60,
               avg(abs(ro / rpc - 1)) OVER w20 AS gap20,
               stddev_samp(ret) OVER w20 AS sd20,
               stddev_samp(ret) OVER w60 AS sd60,
               stddev_samp(ret) OVER w250 AS sd250,
               stddev_samp(CASE WHEN ret < 0 THEN ret END) OVER w20 AS dsd20,
               skewness(ret) OVER w60 AS skew60,
               kurtosis(ret) OVER w60 AS kurt60,
               max(ret) OVER w20 AS maxret20,
               min(ret) OVER w20 AS minret20,
               avg(turn) OVER w20 / nullif(avg(turn) OVER w250, 0) AS turn_rel250,
               avg(amt) OVER w20 / nullif(avg(amt) OVER w250, 0) AS amt_rel250,
               avg(v) OVER w20 / nullif(avg(v) OVER w250, 0) AS vol_rel250,
               median(pe) OVER w250 AS pe_med250,
               median(pb) OVER w250 AS pb_med250,
               median(turn) OVER w250 AS turn_med250,
               median(amt) OVER w250 AS amt_med250,
               median(v) OVER w250 AS v_med250,
               lag(c, 5) OVER w0 AS c5ago,
               lag(c, 10) OVER w0 AS c10ago,
               lag(c, 20) OVER w0 AS c20ago,
               lag(c, 60) OVER w0 AS c60ago,
               lag(c, 120) OVER w0 AS c120ago,
               lag(c, 250) OVER w0 AS c250ago,
               avg(v) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 3 PRECEDING) AS v_prior20,
               avg(v) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS v_last3,
               sum(amt) OVER w20 AS amt_sum20,
               sum(v) OVER w20 AS vol_sum20
        FROM px p
        WINDOW
          w0 AS (PARTITION BY code ORDER BY date),
          w5 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
          w10 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
          w20 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
          w60 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),
          w120 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 119 PRECEDING AND CURRENT ROW),
          w250 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW)
    """, "stage b")
    print(con.execute("select count(*), count(ma250), count(pe_med250) from b").fetchone(), flush=True)

    # ---------- stage C0: lag-based returns and point-in-time joins ----------
    q(con, """
        CREATE TABLE c0 AS
        SELECT b.*,
               m.ew_ret AS mkt_ret, m.mkt_ret5, m.mkt_ret20, m.mkt_ret60, m.mkt_ret120,
               m.ew_ma200, m.mkt_vol20, m.mkt_dd250, m.mkt_gate,
               c / c5ago - 1 AS ret5,
               c / c10ago - 1 AS ret10,
               c / c20ago - 1 AS ret20,
               c / c60ago - 1 AS ret60,
               c / c120ago - 1 AS ret120,
               c / c250ago - 1 AS ret250,
               ma20 / lag(ma20, 20) OVER w0 - 1 AS ma20_slope20,
               ma60 / lag(ma60, 20) OVER w0 - 1 AS ma60_slope20,
               ma120 / lag(ma120, 20) OVER w0 - 1 AS ma120_slope20,
               ln(nullif(v, 0)) - ln(nullif(lag(v) OVER w0, 0)) AS vchg,
               lag(ret) OVER w0 AS ret_lag1,
               sum(sign(ret) * v) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS obv,
               ln(c) AS lnc
        FROM b
        LEFT JOIN market m USING (date)
        WINDOW w0 AS (PARTITION BY code ORDER BY date)
    """, "stage c0")
    print(con.execute("select count(*), count(ret250), count(mkt_ret), count(obv) from c0").fetchone(), flush=True)

    # ---------- stage C: rolling features on c0 columns ----------
    q(con, """
        CREATE TABLE c AS
        SELECT c0.*,
               sd20 * sqrt(252) AS vol20,
               sd60 * sqrt(252) AS vol60,
               sd250 * sqrt(252) AS vol250,
               dsd20 * sqrt(252) AS downvol20,
               atr20 / nullif(c, 0) AS atr20n,
               atr60 / nullif(c, 0) AS atr60n,
               regr_slope(ret, mkt_ret) OVER w60 AS beta60,
               regr_slope(ret, mkt_ret) OVER w250 AS beta250,
               regr_r2(ret, mkt_ret) OVER w60 AS r2mkt60,
               regr_r2(lnc, t) OVER w60 AS trend_r2_60,
               regr_r2(lnc, t) OVER w120 AS trend_r2_120,
               regr_slope(lnc, t) OVER w60 * 252 AS trend_slope_60,
               regr_slope(lnc, t) OVER w120 * 252 AS trend_slope_120,
               corr(ret, ret_lag1) OVER w60 AS autocorr60,
               var_samp(ret5) OVER w60 / nullif(5 * var_samp(ret) OVER w60, 0) AS vr5,
               var_samp(ret20) OVER w120 / nullif(20 * var_samp(ret) OVER w120, 0) AS vr20,
               regr_slope(obv, t) OVER w20 / nullif(vma20, 0) AS obv_slope20,
               vma5 / nullif(vma20, 0) AS vol_ratio5_20,
               vma20 / nullif(vma60, 0) AS vol_ratio20_60,
               upv20 / nullif(dnv20, 0) AS updown_vol20,
               upv60 / nullif(dnv60, 0) AS updown_vol60,
               v_last3 / nullif(v_prior20, 0) AS pullback_vol_ratio,
               (hh20 - l) / nullif(hh20, 0) AS pullback_depth,
               (hh20 - l) / nullif(atr20, 0) AS pullback_depth_atr,
               ((hh20 - l) / nullif(hh20, 0)) / nullif(hh20 / c20ago - 1, 0) AS retrace_ratio,
               hh20 / c - 1 AS hh20_ratio,
               hh20 / c20ago - 1 AS prior_runup20,
               hh20 / c60ago - 1 AS prior_runup60,
               t - arg_max(t, h) OVER w20 AS days_since_hh20,
               t - arg_max(t, h) OVER w60 AS days_since_hh60,
               t - arg_max(t, h) OVER w250 AS days_since_hh250,
               c / nullif(hh60, 0) - 1 AS dist_hh60,
               c / nullif(hh250, 0) - 1 AS dd250,
               CASE WHEN c >= hh60 THEN 1 ELSE 0 END AS new_high_60,
               CASE WHEN c >= hh250 THEN 1 ELSE 0 END AS new_high_250,
               (hh60 - ll60) / nullif(c, 0) AS box_range60,
               (c - l) / nullif(h - l, 0) AS close_loc,
               (c - o) / nullif(o, 0) AS body,
               (h - greatest(o, c)) / nullif(c, 0) AS upper_shadow,
               (least(o, c) - l) / nullif(c, 0) AS lower_shadow,
               pct / 100.0 AS signal_day_ret,
               turn / nullif(turn20, 0) AS signal_day_turn_ratio,
               ro / nullif(rpc, 0) - 1 AS signal_day_gap,
               corr(ret, vchg) OVER w20 AS vol_price_corr20,
               amt_sum20 / nullif(vol_sum20, 0) AS vwap20,
               c / nullif(amt_sum20 / nullif(vol_sum20, 0), 0) - 1 AS vwap20_dev,
               abs(ret60) / nullif(absret60sum, 0) AS price_efficiency60,
               (CASE WHEN ma5 > ma10 THEN 1 ELSE 0 END)
             + (CASE WHEN ma10 > ma20 THEN 1 ELSE 0 END)
             + (CASE WHEN ma20 > ma60 THEN 1 ELSE 0 END)
             + (CASE WHEN ma60 > ma120 THEN 1 ELSE 0 END) AS ma_alignment,
               ma20 / nullif(ma60, 0) - 1 AS ma20_ma60_spread,
               ma60 / nullif(ma120, 0) - 1 AS ma60_ma120_spread,
               c / nullif(ma20, 0) - 1 AS dist_ma20,
               c / nullif(ma60, 0) - 1 AS dist_ma60,
               c / nullif(ma120, 0) - 1 AS dist_ma120,
               c / nullif(ma250, 0) - 1 AS dist_ma250,
               updays20 / 20.0 AS up_ratio20,
               updays60 / 60.0 AS up_ratio60,
               st250 AS st_days250,
               lu20 AS limit_up20, lu60 AS limit_up60,
               ld20 AS limit_down20, ld60 AS limit_down60,
               gap20 AS gap_abs20,
               ln(nullif(mcap, 0)) AS log_mcap,
               ln(nullif(amt20, 0)) AS log_amt20,
               ln(nullif(amt60, 0)) AS log_amt60,
               amt20 / nullif(amt120, 0) - 1 AS amt_trend,
               absret20 / nullif(amt20, 0) * 1e8 AS amihud20,
               turn20 / nullif(turn_med250, 0) - 1 AS turn_pct250,
               amt20 / nullif(amt_med250, 0) - 1 AS amt_pct250,
               vma20 / nullif(v_med250, 0) - 1 AS vol_pct250,
               CASE WHEN pe <= 0 THEN 1 ELSE 0 END AS pe_neg,
               ln(1 + greatest(pe, 0)) AS log_pe,
               ln(1 + greatest(pb, 0)) AS log_pb,
               pe / nullif(pe_med250, 0) - 1 AS pe_med_ratio250,
               pb / nullif(pb_med250, 0) - 1 AS pb_med_ratio250,
               td_since_list / 250.0 AS age_years,
               board
        FROM c0
        WINDOW
          w20 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
          w60 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),
          w120 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 119 PRECEDING AND CURRENT ROW),
          w250 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW)
    """, "stage c")
    print(con.execute("select count(*), count(vol20), count(trend_r2_120), count(beta60) from c").fetchone(), flush=True)

    # ---------- stage D: features that need a window over a window ----------
    q(con, """
        CREATE TABLE d AS
        SELECT c.*,
               stddev_samp(vol20) OVER w60 / nullif(avg(vol20) OVER w60, 0) AS vol_of_vol60,
               min(dd250) OVER w250 AS max_dd_250,
               sqrt(avg(dd250 * dd250) OVER w250) AS ulcer250,
               sum(CASE WHEN l <= ma20 * 1.01 THEN 1 ELSE 0 END) OVER w10 AS ma20_touch_count10,
               CASE WHEN board = '主板' THEN 1 ELSE 0 END AS board_main,
               CASE WHEN board = '创业板' THEN 1 ELSE 0 END AS board_gem,
               CASE WHEN board = '科创板' THEN 1 ELSE 0 END AS board_star
        FROM c
        WINDOW
          w10 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
          w60 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),
          w250 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW)
    """, "stage d")
    print(con.execute("select count(*), count(vol_of_vol60), count(max_dd_250) from d").fetchone(), flush=True)


if __name__ == "__main__":
    main()
