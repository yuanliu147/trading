#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建冻结数据层（沪深A股日线 + 指数 + 日历 + 公司行动 + 市场宽度）。

数据源（只读，不同步、不迁移、不修改）：
  1) Baostock SQLite : /Users/admin/WorkSpace/baostock_data/data/baostock.db
       stock_kline_d (adjustflag=3 不复权 / =1 后复权), stock_basic, index_kline_d, trade_dates
  2) 同花顺 DuckDB   : /Users/admin/Library/Application Support/hithink-finance/data/market.duckdb
       raw_adjustment_events (公司行动), raw_kline_daily (北交所 + 交叉校验)

冻结日 2026-10-26（源库含 10-27/10-28 数据，一律截断）。

复权口径（v2，2026-10-28 修正；v1 的 hfq=raw*factor 已废弃）
-------------------------------------------------------------
factor_t = Π_{s<=t}(1 + pct_chg_s/100)，首条=1，是**总收益指数**（分红再投资）。
A 空间（后复权）价格必须把"价格"与"累计收益"分开：
    A_t          = factor_t / close_t          # 逐日调整乘子，只用 t 日及以前数据
    close_first  = 该证券在冻结窗口内首条有效收盘
    hfq_px_t     = raw_px_t * A_t * close_first
                 = raw_px_t * factor_t / close_t * close_first
于是 hfq_close_t/hfq_close_{t-1} - 1 = factor_t/factor_{t-1} - 1 = pct_chg_t/100，
且 hfq_open_t/hfq_close_{t-1} - 1 = open_t/preclose_t - 1（除权日也成立）。
v1 的 hfq = raw * factor 把价格收益重复计入一次，日收益≈(1+r)²-1，已废止。

前复权（仅展示/横向比较，锚定冻结日最后一条）：
    qfq_px_t = raw_px_t * factor_t / close_t * close_last / factor_last
末日 qfq_close = 原始收盘；日收益同样等于官方 pct_chg；价格水平随数据更新而变，
不得用于计算或跨日比较的持久化结果。

Baostock 后复权收盘另存 bs_hfq_close 仅作交叉校验。

用法：
  python3 02_代码/构建冻结数据.py            # 增量（已存在则复用中间件）
  python3 02_代码/构建冻结数据.py --force    # 全量重建
  python3 02_代码/构建冻结数据.py --steps daily,market
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import date, datetime
from pathlib import Path

import duckdb

FREEZE = "2026-10-26"
START = "2004-01-01"
INDEX_START = "2006-01-01"
DATA_VERSION = "v2-adjspace-2026-10-28"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "01_数据"
FROZEN = DATA_DIR / "frozen"
TMP = DATA_DIR / "_tmp"
AUDIT = DATA_DIR / "_audit"

BS_DB = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"
TH_DB = "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"

KEEP_BOARDS = ("主板", "创业板", "科创板")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(TMP / "build.duckdb"))
    con.execute("SET memory_limit='8GB'")
    con.execute(f"SET temp_directory='{TMP / 'spill'}'")
    con.execute("SET threads=6")
    con.execute("LOAD sqlite")
    con.execute(f"ATTACH IF NOT EXISTS '{BS_DB}' AS bs (TYPE SQLITE, READ_ONLY)")
    return con


def attach_th(con):
    con.execute("LOAD sqlite")
    con.execute(f"ATTACH IF NOT EXISTS '{TH_DB}' AS th (TYPE DUCKDB, READ_ONLY)")
    return con


def rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def copy_parquet(con, sql: str, out: Path) -> None:
    """分区 Parquet：先写 out__new，再原子换名，缩短下游读到半成品的窗口。"""
    stage = out.with_name(out.name + "__new")
    rm(stage)
    stage.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{stage}' (FORMAT PARQUET, PARTITION_BY (year), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)")
    old = out.with_name(out.name + "__old")
    rm(old)
    if out.exists():
        out.rename(old)
    stage.rename(out)
    rm(old)


def copy_file(con, sql: str, out: Path) -> None:
    """单文件 Parquet：写临时文件后替换。"""
    tmp = out.with_name(out.stem + ".tmp.parquet")
    rm(tmp)
    con.execute(f"COPY ({sql}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    tmp.replace(out)

# --------------------------------------------------------------------------- #
# 1. 证券主表
# --------------------------------------------------------------------------- #
def step_symbols(con) -> None:
    log("1/9 证券主表")
    con.execute(f"""
    CREATE OR REPLACE TABLE sym AS
    WITH b AS (
      SELECT code AS bs_code, code_name, CAST(ipo_date AS DATE) AS ipo_date,
             CAST(NULLIF(out_date,'') AS DATE) AS delist_date, type, status
      FROM bs.stock_basic WHERE type = 1
    ), k AS (
      SELECT code, MIN(CAST(date AS DATE)) AS first_bar, MAX(CAST(date AS DATE)) AS last_bar,
             COUNT(*) AS n_bars
      FROM bs.stock_kline_d WHERE adjustflag = 3 AND date BETWEEN '{START}' AND '{FREEZE}'
      GROUP BY 1
    )
    SELECT substr(b.bs_code,4) || '.' || upper(substr(b.bs_code,1,2)) AS symbol,
           b.bs_code, b.code_name AS name, upper(substr(b.bs_code,1,2)) AS exchange,
           CASE
             WHEN b.bs_code LIKE 'sh.60%' OR b.bs_code LIKE 'sz.00%' THEN '主板'
             WHEN b.bs_code LIKE 'sz.30%' THEN '创业板'
             WHEN b.bs_code LIKE 'sh.68%' THEN '科创板'
           END AS board,
           b.ipo_date, b.delist_date,
           CASE WHEN b.status = 0 THEN 1 ELSE 0 END AS status_delisted,
           k.first_bar, k.last_bar, coalesce(k.n_bars, 0) AS n_bars
    FROM b LEFT JOIN k ON k.code = b.bs_code
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE symbols_out AS
    SELECT * FROM sym
    WHERE board IN {KEEP_BOARDS} AND n_bars > 0
      AND (delist_date IS NULL OR delist_date >= DATE '2004-01-01')
    """)
    copy_file(con, "SELECT * FROM symbols_out", DATA_DIR / "frozen" / "symbols.parquet")
    n = con.execute("SELECT count(*) FROM symbols_out").fetchone()[0]
    nb = con.execute("SELECT board, count(*) FROM symbols_out GROUP BY 1 ORDER BY 1").fetchall()
    log(f"    symbols={n} {nb}")


# --------------------------------------------------------------------------- #
# 2/3. 原始与后复权中间件
# --------------------------------------------------------------------------- #
def step_raw(con, force: bool) -> None:
    out = TMP / "raw"
    if (out / "_ok").exists() and not force:
        log("2/9 原始日线中间件已存在，跳过")
        return
    log("2/9 抽取不复权日线（约 1850 万行）")
    t = time.time()
    copy_parquet(con, f"""
      SELECT substr(code,4) || '.' || upper(substr(code,1,2)) AS symbol,
             CAST(date AS DATE) AS date,
             open, high, low, close, preclose, volume, amount,
             turn, trade_status, is_st, pct_chg, pe_ttm, pb_mrq,
             year(CAST(date AS DATE)) AS year
      FROM bs.stock_kline_d
      WHERE adjustflag = 3 AND date BETWEEN '{START}' AND '{FREEZE}'
    """, out)
    (out / "_ok").write_text("ok")
    log(f"    完成 {time.time()-t:.0f}s")


def step_hfq(con, force: bool) -> None:
    out = TMP / "bs_hfq"
    if (out / "_ok").exists() and not force:
        log("3/9 Baostock 后复权中间件已存在，跳过")
        return
    log("3/9 抽取 Baostock 后复权收盘（仅交叉校验用）")
    t = time.time()
    copy_parquet(con, f"""
      SELECT substr(code,4) || '.' || upper(substr(code,1,2)) AS symbol,
             CAST(date AS DATE) AS date, close AS bs_hfq_close,
             year(CAST(date AS DATE)) AS year
      FROM bs.stock_kline_d
      WHERE adjustflag = 1 AND date BETWEEN '{START}' AND '{FREEZE}'
    """, out)
    (out / "_ok").write_text("ok")
    log(f"    完成 {time.time()-t:.0f}s")


# --------------------------------------------------------------------------- #
# 4. 日线主表
# --------------------------------------------------------------------------- #
def step_daily(con) -> None:
    log("4/9 构建日线主表（自建复权因子、涨跌停、流通股、上市天数）")
    t = time.time()
    con.execute(f"""
    CREATE OR REPLACE TABLE cal AS
    SELECT CAST(calendar_date AS DATE) AS d,
           row_number() OVER (ORDER BY calendar_date) - 1 AS ti
    FROM bs.trade_dates WHERE is_trading_day = 1 AND calendar_date <= '{FREEZE}'
    """)
    # 逐证券收益 -> 累计因子
    con.execute(f"""
    CREATE OR REPLACE TABLE d1 AS
    WITH r AS (
      SELECT r.*,
             CASE WHEN row_number() OVER (PARTITION BY symbol ORDER BY date) = 1 THEN 0.0
                  ELSE coalesce(pct_chg / 100.0,
                                CASE WHEN preclose > 0 THEN close / preclose - 1 END, 0.0) END AS ret
      FROM read_parquet('{TMP}/raw/*/*.parquet') r
    )
    SELECT r.* EXCLUDE (year), exp(sum(ln(greatest(1 + ret, 1e-6)))
                     OVER (PARTITION BY symbol ORDER BY date
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) AS factor
    FROM r
    """)
    con.execute("""
    CREATE OR REPLACE TABLE d1b AS
    SELECT symbol, arg_min(close, date) FILTER (WHERE close > 0) AS close_first,
           arg_max(close, date) FILTER (WHERE close > 0) AS close_last,
           arg_max(factor, date) AS factor_last
    FROM d1 GROUP BY 1
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE d2 AS
    SELECT d1.*, s.name, s.board, s.ipo_date, s.delist_date,
           b.bs_hfq_close,
           -- A 空间（后复权）：raw * factor / close_t * close_first，不做 4 位舍入
           open  * factor / nullif(close, 0) * a.close_first AS hfq_open,
           high  * factor / nullif(close, 0) * a.close_first AS hfq_high,
           low   * factor / nullif(close, 0) * a.close_first AS hfq_low,
           close * factor / nullif(close, 0) * a.close_first AS hfq_close,
           (volume IS NULL OR volume = 0) AS is_suspended,
           ti - ipo_ti + 1 AS td_since_list
    FROM d1
    LEFT JOIN read_parquet('{TMP}/bs_hfq/*/*.parquet') b USING (symbol, date)
    LEFT JOIN read_parquet('{DATA_DIR}/frozen/symbols.parquet') s USING (symbol)
    LEFT JOIN d1b a USING (symbol)
    LEFT JOIN (
      SELECT s.symbol, min(c.ti) AS ipo_ti
      FROM (SELECT symbol, ipo_date FROM read_parquet('{DATA_DIR}/frozen/symbols.parquet')) s
      JOIN cal c ON c.d >= s.ipo_date
      GROUP BY 1
    ) i USING (symbol)
    LEFT JOIN cal ON cal.d = d1.date
    """)
    # 涨跌停规则
    con.execute(f"""
    CREATE OR REPLACE TABLE d3 AS
    SELECT d2.*,
      CASE
        WHEN board = '科创板' THEN CASE WHEN td_since_list <= 5 THEN NULL ELSE 0.20 END
        WHEN board = '创业板' THEN
          CASE WHEN date >= DATE '2020-08-24' AND td_since_list <= 5 THEN NULL
               WHEN date >= DATE '2020-08-24' THEN 0.20
               WHEN td_since_list = 1 AND ipo_date >= DATE '2014-01-01' THEN 0.44
               WHEN td_since_list = 1 THEN NULL
               WHEN is_st = 1 THEN 0.05 ELSE 0.10 END
        WHEN board = '主板' THEN
          CASE WHEN ipo_date >= DATE '2023-04-10' AND td_since_list <= 5 THEN NULL
               WHEN td_since_list = 1 AND ipo_date >= DATE '2014-01-01' THEN 0.44
               WHEN td_since_list = 1 THEN NULL
               WHEN is_st = 1 AND date >= DATE '2026-07-06' THEN 0.10
               WHEN is_st = 1 THEN 0.05 ELSE 0.10 END
        ELSE NULL END AS limit_pct,
      CASE
        WHEN board = '主板' AND td_since_list = 1 AND ipo_date >= DATE '2014-01-01'
             AND ipo_date < DATE '2023-04-10' THEN 0.36
        WHEN board = '创业板' AND td_since_list = 1 AND ipo_date >= DATE '2014-01-01' THEN 0.36
        ELSE NULL END AS limit_dn_pct
    FROM d2
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE d4 AS
    SELECT d3.* EXCLUDE (limit_dn_pct),
      CASE WHEN limit_pct IS NULL OR preclose IS NULL OR preclose <= 0 THEN NULL
           ELSE floor(preclose * (1 + limit_pct) * 100 + 0.5) / 100 END AS limit_up_px,
      CASE WHEN limit_pct IS NULL OR preclose IS NULL OR preclose <= 0 THEN NULL
           ELSE floor(preclose * (1 - coalesce(limit_dn_pct, limit_pct)) * 100 + 0.5) / 100 END AS limit_down_px
    FROM d3
    """)
    con.execute("""
    CREATE OR REPLACE TABLE d5 AS
    SELECT d4.*,
      CASE WHEN limit_up_px IS NULL THEN NULL
           ELSE close >= limit_up_px - 0.001 END AS is_limit_up_close,
      CASE WHEN limit_down_px IS NULL THEN NULL
           ELSE close <= limit_down_px + 0.001 END AS is_limit_down_close,
      CASE WHEN limit_up_px IS NULL THEN NULL
           ELSE open >= limit_up_px - 0.001 END AS is_limit_up_open,
      CASE WHEN limit_down_px IS NULL THEN NULL
           ELSE open <= limit_down_px + 0.001 END AS is_limit_down_open,
      CASE WHEN limit_up_px IS NULL THEN NULL
           ELSE high > limit_up_px + 0.011 END AS px_breaks_limit_up,
      CASE WHEN limit_down_px IS NULL THEN NULL
           ELSE low < limit_down_px - 0.011 END AS px_breaks_limit_down
    FROM d4
    """)
    # 流通股（换手率反推，按 证券-年 取中位数）
    con.execute("""
    CREATE OR REPLACE TABLE fs AS
    SELECT symbol, year(date) AS y, median(volume / (turn / 100.0)) AS float_shares
    FROM d4 WHERE turn > 0 AND volume > 0 GROUP BY 1, 2
    """)
    con.execute("""
    CREATE OR REPLACE TABLE fs_sym AS
    SELECT symbol, median(volume / (turn / 100.0)) AS float_shares_sym
    FROM d4 WHERE turn > 0 AND volume > 0 GROUP BY 1
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE daily AS
    SELECT d.symbol, d.date, d.name, d.board, d.ipo_date, d.delist_date, d.td_since_list,
           d.open, d.high, d.low, d.close, d.preclose,
           d.volume, d.amount, d.turn,
           d.trade_status, d.is_st, d.is_suspended, d.pct_chg,
           d.pe_ttm, d.pb_mrq,
           d.factor, d.hfq_open, d.hfq_high, d.hfq_low, d.hfq_close,
           d.bs_hfq_close,
           round(d.close / d.preclose - 1, 8) AS close_ret_check,
           d.limit_pct, d.limit_up_px, d.limit_down_px,
           d.is_limit_up_close, d.is_limit_down_close,
           d.is_limit_up_open, d.is_limit_down_open,
           d.px_breaks_limit_up, d.px_breaks_limit_down,
           coalesce(f.float_shares, f2.float_shares_sym) AS float_shares,
           round(d.close * coalesce(f.float_shares, f2.float_shares_sym), 0) AS float_mktcap
    FROM d5 d
    LEFT JOIN fs f ON f.symbol = d.symbol AND f.y = year(d.date)
    LEFT JOIN fs_sym f2 ON f2.symbol = d.symbol
    """)
    # 构建期硬断言：hfq_close 日收益必须等于官方 pct_chg（双计会立刻失败）
    chk = con.execute("""
    WITH r AS (
      SELECT symbol, date, pct_chg, hfq_close,
             lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) prev
      FROM daily
    )
    SELECT count(*) FILTER (WHERE prev > 0 AND abs(hfq_close/prev - 1 - pct_chg/100) > 5e-5) n_bad,
           max(abs(hfq_close/prev - 1 - pct_chg/100)) FILTER (WHERE prev > 0) max_err,
           count(*) FILTER (WHERE prev > 0) n
    FROM r
    """).fetchone()
    log(f"    复权不变量：相邻日 {chk[2]:,} 对，>0.5bp 的行={chk[0]}，最大误差={chk[1]:.2e}")
    if chk[0] or (chk[1] or 0) > 1e-9:
        raise RuntimeError(f"hfq_close 日收益与 pct_chg 不一致：{chk}")

    rm(FROZEN / "daily__new")
    FROZEN.mkdir(parents=True, exist_ok=True)
    copy_parquet(con, "SELECT *, year(date) AS year FROM daily", FROZEN / "daily")
    n = con.execute("SELECT count(*) FROM daily").fetchone()[0]
    log(f"    daily rows={n:,}  {time.time()-t:.0f}s")


# --------------------------------------------------------------------------- #
# 5. 指数
# --------------------------------------------------------------------------- #
def step_index(con) -> None:
    log("5/9 指数日线")
    con.execute(f"""
    CREATE OR REPLACE TABLE idx AS
    SELECT substr(code,4) || '.' || upper(substr(code,1,2)) AS symbol,
           CAST(date AS DATE) AS date, open, high, low, close, preclose,
           volume, amount, pct_chg
    FROM bs.index_kline_d
    WHERE date BETWEEN '{INDEX_START}' AND '{FREEZE}'
    """)
    rm(FROZEN / "index_daily__new")
    copy_parquet(con, "SELECT *, year(date) AS year FROM idx", FROZEN / "index_daily")
    rows = con.execute("SELECT symbol, count(*), min(date), max(date) FROM idx GROUP BY 1 ORDER BY 1").fetchall()
    for r in rows:
        log(f"    {r}")


# --------------------------------------------------------------------------- #
# 6. 交易日历
# --------------------------------------------------------------------------- #
def step_calendar(con) -> None:
    log("6/9 交易日历")
    con.execute(f"""
    CREATE OR REPLACE TABLE calendar AS
    WITH c AS (
      SELECT CAST(calendar_date AS DATE) AS date, is_trading_day
      FROM bs.trade_dates WHERE calendar_date <= '{FREEZE}'
    )
    SELECT date, is_trading_day,
           sum(is_trading_day) OVER (ORDER BY date) - is_trading_day AS td_before,
           CASE WHEN is_trading_day = 1 THEN
             sum(is_trading_day) OVER (ORDER BY date) - 1 END AS td_index
    FROM c
    """)
    copy_file(con, "SELECT * FROM calendar", FROZEN / "calendar.parquet")
    n = con.execute("SELECT count(*), sum(is_trading_day), min(date), max(date) FROM calendar").fetchone()
    log(f"    calendar={n}")


# --------------------------------------------------------------------------- #
# 7. 公司行动
# --------------------------------------------------------------------------- #
def step_corp(con) -> None:
    log("7/9 公司行动（同花顺事件 + 因子跳变推导）")
    attach_th(con)
    con.execute(f"""
    CREATE OR REPLACE TABLE ca AS
    SELECT thscode AS symbol, CAST(ex_date AS DATE) AS ex_date,
           dividend_per_share, per_share_bonus, rights_ratio, rights_price,
           datediff('day', CAST(ex_date AS DATE), DATE '{FREEZE}') AS days_to_freeze,
           'hithink_event' AS source
    FROM th.raw_adjustment_events
    WHERE ex_date <= DATE '{FREEZE}'
      AND (dividend_per_share > 0 OR per_share_bonus > 0 OR rights_ratio > 0)
    """)
    copy_file(con, "SELECT * FROM ca", FROZEN / "corp_actions.parquet")
    log(f"    corp_actions={con.execute('SELECT count(*) FROM ca').fetchone()[0]:,}")

    # 除权除息日：preclose 与前一日收盘不一致（正常日两者严格相等）
    con.execute(f"""
    CREATE OR REPLACE TABLE ca_imp AS
    WITH j AS (
      SELECT symbol, date, close, preclose, factor,
             lag(close) OVER (PARTITION BY symbol ORDER BY date) AS c_prev,
             lag(date)  OVER (PARTITION BY symbol ORDER BY date) AS d_prev
      FROM read_parquet('{FROZEN}/daily/*/*.parquet')
    )
    SELECT symbol, date AS ex_date, d_prev, c_prev AS prev_close, preclose,
           round(preclose / c_prev, 8) AS adj_ratio,
           round(preclose / c_prev - 1, 8) AS adj_change,
           EXISTS (SELECT 1 FROM ca WHERE ca.symbol = j.symbol AND ca.ex_date = j.date) AS has_th_event
    FROM j
    WHERE c_prev IS NOT NULL AND c_prev > 0 AND abs(preclose / c_prev - 1) > 1e-6
    """)
    copy_file(con, "SELECT * FROM ca_imp", FROZEN / "corp_actions_implied.parquet")
    log(f"    corp_actions_implied={con.execute('SELECT count(*) FROM ca_imp').fetchone()[0]:,}")


# --------------------------------------------------------------------------- #
# 8. 北交所（同花顺，次级口径）
# --------------------------------------------------------------------------- #
def step_bj(con) -> None:
    log("8/9 北交所日线（同花顺，次级口径）")
    con.execute(f"""
    CREATE OR REPLACE TABLE bj AS
    WITH r AS (
      SELECT thscode AS symbol, CAST(date AS DATE) AS date, open, high, low, close, volume, amount
      FROM th.raw_kline_daily
      WHERE thscode LIKE '%.BJ' AND date BETWEEN DATE '2016-10-13' AND DATE '{FREEZE}'
    ), h AS (
      SELECT thscode AS symbol, CAST(date AS DATE) AS date, close AS hfq_close
      FROM th.v_daily_hfq
      WHERE thscode LIKE '%.BJ' AND date BETWEEN DATE '2016-10-13' AND DATE '{FREEZE}'
    )
    SELECT r.*, h.hfq_close, round(h.hfq_close / nullif(r.close,0), 6) AS factor
    FROM r LEFT JOIN h USING (symbol, date)
    """)
    rm(FROZEN / "bj_daily__new")
    copy_parquet(con, "SELECT *, year(date) AS year FROM bj", FROZEN / "bj_daily")
    log(f"    bj_daily={con.execute('SELECT count(*), count(DISTINCT symbol) FROM bj').fetchone()}")


# --------------------------------------------------------------------------- #
# 9. 市场宽度
# --------------------------------------------------------------------------- #
def step_market(con) -> None:
    log("9/9 市场宽度与基准")
    t = time.time()
    con.execute(f"""
    CREATE OR REPLACE TABLE md AS
    WITH d AS (
      SELECT symbol, date, close, hfq_close, factor, pct_chg, amount, is_suspended,
             is_limit_up_close, is_limit_down_close, float_mktcap,
             avg(hfq_close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
             avg(hfq_close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS ma60,
             count(*)     OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS n20,
             count(*)     OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS n60
      FROM read_parquet('{FROZEN}/daily/*/*.parquet')
    )
    SELECT date,
      count(*) AS n_listed,
      count(*) FILTER (WHERE NOT is_suspended) AS n_traded,
      count(*) FILTER (WHERE is_suspended) AS n_suspended,
      count(*) FILTER (WHERE NOT is_suspended AND pct_chg > 0.005) AS n_up,
      count(*) FILTER (WHERE NOT is_suspended AND pct_chg < -0.005) AS n_down,
      count(*) FILTER (WHERE NOT is_suspended AND abs(pct_chg) <= 0.005) AS n_flat,
      count(*) FILTER (WHERE is_limit_up_close) AS n_limit_up,
      count(*) FILTER (WHERE is_limit_down_close) AS n_limit_down,
      round(median(pct_chg) FILTER (WHERE NOT is_suspended), 4) AS med_ret,
      round(avg(pct_chg) FILTER (WHERE NOT is_suspended), 4) AS ew_ret,
      round(sum(amount) FILTER (WHERE NOT is_suspended) / 1e8, 2) AS amount_yi,
      round(avg(CASE WHEN NOT is_suspended AND n20 = 20 AND hfq_close > ma20 THEN 1.0
                     WHEN NOT is_suspended AND n20 = 20 THEN 0.0 END) * 100, 2) AS pct_above_ma20,
      round(avg(CASE WHEN NOT is_suspended AND n60 = 60 AND hfq_close > ma60 THEN 1.0
                     WHEN NOT is_suspended AND n60 = 60 THEN 0.0 END) * 100, 2) AS pct_above_ma60,
      round(sum(float_mktcap) / 1e12, 4) AS float_mktcap_wanyi
    FROM d GROUP BY date ORDER BY date
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE ix AS
    SELECT date, close AS hs300_close, round(pct_chg, 4) AS hs300_ret
    FROM idx WHERE symbol = '000300.SH'
    """)
    con.execute("""
    CREATE OR REPLACE TABLE market_daily AS
    SELECT m.*, i.hs300_close, i.hs300_ret
    FROM md m LEFT JOIN ix i USING (date) ORDER BY date
    """)
    copy_file(con, "SELECT * FROM market_daily", FROZEN / "market_daily.parquet")
    log(f"    market_daily={con.execute('SELECT count(*) FROM market_daily').fetchone()[0]:,} {time.time()-t:.0f}s")


# --------------------------------------------------------------------------- #
# 10. 冻结库 + 元数据
# --------------------------------------------------------------------------- #
def step_db(con) -> None:
    log("10/10 生成 frozen.duckdb 与 meta.json")
    db = FROZEN / "frozen.duckdb"
    tmp_db = FROZEN / "frozen.duckdb.tmp"
    rm(tmp_db)
    f = duckdb.connect(str(tmp_db))
    f.execute("SET memory_limit='4GB'")
    f.execute(f"""
      CREATE VIEW daily AS SELECT * FROM read_parquet('{FROZEN}/daily/*/*.parquet', hive_partitioning=true)
    """)
    f.execute(f"""
      CREATE VIEW index_daily AS SELECT * FROM read_parquet('{FROZEN}/index_daily/*/*.parquet', hive_partitioning=true)
    """)
    f.execute(f"CREATE VIEW calendar AS SELECT * FROM read_parquet('{FROZEN}/calendar.parquet')")
    f.execute(f"CREATE VIEW symbols AS SELECT * FROM read_parquet('{FROZEN}/symbols.parquet')")
    f.execute(f"CREATE VIEW corp_actions AS SELECT * FROM read_parquet('{FROZEN}/corp_actions.parquet')")
    f.execute(f"CREATE VIEW corp_actions_implied AS SELECT * FROM read_parquet('{FROZEN}/corp_actions_implied.parquet')")
    f.execute(f"CREATE VIEW market_daily AS SELECT * FROM read_parquet('{FROZEN}/market_daily.parquet')")
    f.execute(f"""
      CREATE VIEW bj_daily AS SELECT * FROM read_parquet('{FROZEN}/bj_daily/*/*.parquet', hive_partitioning=true)
    """)
    f.execute("""
      CREATE TABLE symbol_factor AS
      SELECT symbol, max_by(factor, date) AS factor_last, max_by(date, date) AS last_date,
             arg_min(close, date) FILTER (WHERE close > 0) AS close_first,
             arg_max(close, date) FILTER (WHERE close > 0) AS close_last,
             count(*) AS n_bars
      FROM daily GROUP BY 1
    """)
    f.execute("""
      CREATE VIEW daily_qfq AS
      SELECT d.*,
             d.open  * d.factor / nullif(d.close, 0) * f.close_last / f.factor_last AS qfq_open,
             d.high  * d.factor / nullif(d.close, 0) * f.close_last / f.factor_last AS qfq_high,
             d.low   * d.factor / nullif(d.close, 0) * f.close_last / f.factor_last AS qfq_low,
             d.close * d.factor / nullif(d.close, 0) * f.close_last / f.factor_last AS qfq_close
      FROM daily d JOIN symbol_factor f USING (symbol)
    """)
    f.execute("""
      CREATE VIEW universe AS
      SELECT symbol, date, name, board, close, float_mktcap, amount, is_st, is_suspended,
             limit_up_px, limit_down_px, is_limit_up_close, is_limit_down_close,
             is_limit_up_open, is_limit_down_open, td_since_list
      FROM daily
    """)
    f.close()
    tmp_db.replace(db)

    meta = build_meta(con)
    meta_tmp = DATA_DIR / "meta.json.tmp"
    meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    meta_tmp.replace(DATA_DIR / "meta.json")
    hist = AUDIT / "build_history.csv"
    new = not hist.exists()
    files = meta["files"]
    agg = hashlib.sha256()
    for f in sorted(files, key=lambda x: x["path"]):
        agg.update(f"{f['path']}:{f['rows']}:{f['sha256']}\n".encode())
    with open(hist, "a", encoding="utf-8") as fh:
        if new:
            fh.write("built_at,data_version,daily_rows,symbols,content_digest,files_agg_sha256,freeze\n")
        fh.write(f"{meta['built_at']},{meta['data_version']},{meta['tables']['daily']['rows']},"
                 f"{meta['tables']['daily']['symbols']},{meta['content_digest']['daily_sha256']},"
                 f"{agg.hexdigest()},{FREEZE}\n")
    log(f"    frozen.duckdb 完成，meta.json 写入 {len(json.dumps(meta))} bytes")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def content_digest(con) -> dict:
    """与 Parquet 编码/行序无关的内容指纹：整数化后求和，跨重建可复现。"""
    r = con.execute("""
    SELECT count(*),
           sum(cast(round(open * 100) AS hugeint)) FILTER (WHERE open IS NOT NULL),
           sum(cast(round(high * 100) AS hugeint)) FILTER (WHERE high IS NOT NULL),
           sum(cast(round(low * 100) AS hugeint)) FILTER (WHERE low IS NOT NULL),
           sum(cast(round(close * 100) AS hugeint)) FILTER (WHERE close IS NOT NULL),
           sum(cast(round(hfq_close * 10000) AS hugeint)) FILTER (WHERE hfq_close IS NOT NULL),
           sum(cast(round(factor * 1000000) AS hugeint)) FILTER (WHERE factor IS NOT NULL),
           sum(cast(round(pct_chg * 10000) AS hugeint)) FILTER (WHERE pct_chg IS NOT NULL)
    FROM daily""").fetchone()
    comps = [int(x) if x is not None else None for x in r]
    payload = "|".join("" if x is None else str(x) for x in comps)
    return {"daily_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "components": comps,
            "note": "count,sum(open*100),sum(high*100),sum(low*100),sum(close*100),sum(hfq_close*1e4),sum(factor*1e6),sum(pct_chg*1e4) as integers"}


def build_meta(con) -> dict:
    def one(sql):
        return con.execute(sql).fetchone()

    files = []
    for p in sorted(FROZEN.rglob("*.parquet")):
        n = con.execute(f"SELECT count(*) FROM read_parquet('{p}')").fetchone()[0]
        files.append({"path": str(p.relative_to(DATA_DIR)), "rows": n,
                      "bytes": p.stat().st_size, "sha256": sha256(p)})
    daily_files = [f for f in files if f["path"].startswith("frozen/daily/")]
    meta = {
        "data_version": DATA_VERSION,
        "freeze_date": FREEZE,
        "start_date": START,
        "index_start": INDEX_START,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "adjust_space": {
            "factor": "factor_t = prod_{s<=t}(1+pct_chg_s/100), first bar=1; total-return index",
            "A_t": "factor_t / close_t * close_first",
            "hfq": "raw_px_t * A_t; hfq_close return == pct_chg; first bar hfq_close == raw close",
            "qfq": "raw_px_t * factor_t/close_t * close_last/factor_last; last bar qfq_close == raw close; display only",
            "deprecated_v1": "hfq = raw_px * factor (double-counts price return; daily return ~ (1+r)^2-1)",
        },
        "start_date": START,
        "index_start": INDEX_START,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "baostock_sqlite": {"path": BS_DB, "bytes": Path(BS_DB).stat().st_size,
                                "mtime": datetime.fromtimestamp(Path(BS_DB).stat().st_mtime).isoformat(timespec="seconds")},
            "hithink_duckdb": {"path": TH_DB, "bytes": Path(TH_DB).stat().st_size,
                               "mtime": datetime.fromtimestamp(Path(TH_DB).stat().st_mtime).isoformat(timespec="seconds")},
        },
        "tables": {
            "daily": {"rows": one("SELECT count(*) FROM daily")[0],
                      "symbols": one("SELECT count(DISTINCT symbol) FROM daily")[0],
                      "date_min": str(one("SELECT min(date) FROM daily")[0]),
                      "date_max": str(one("SELECT max(date) FROM daily")[0]),
                      "partitions": len(daily_files),
                      "bytes": sum(f["bytes"] for f in daily_files)},
            "index_daily": {"rows": one("SELECT count(*) FROM idx")[0],
                            "symbols": one("SELECT count(DISTINCT symbol) FROM idx")[0],
                            "date_min": str(one("SELECT min(date) FROM idx")[0]),
                            "date_max": str(one("SELECT max(date) FROM idx")[0])},
            "calendar": {"rows": one("SELECT count(*) FROM calendar")[0],
                         "trading_days": one("SELECT sum(is_trading_day) FROM calendar")[0]},
            "symbols": {"rows": one("SELECT count(*) FROM symbols_out")[0]},
            "corp_actions": {"rows": one("SELECT count(*) FROM ca")[0]},
            "corp_actions_implied": {"rows": one("SELECT count(*) FROM ca_imp")[0]},
            "market_daily": {"rows": one("SELECT count(*) FROM market_daily")[0]},
            "bj_daily": {"rows": one("SELECT count(*) FROM bj")[0],
                         "symbols": one("SELECT count(DISTINCT symbol) FROM bj")[0]},
        },
        "by_board": {r[0]: r[1] for r in con.execute(
            "SELECT board, count(DISTINCT symbol) FROM daily GROUP BY 1").fetchall()},
        "content_digest": content_digest(con),
        "files": files,
    }
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--steps", default="all")
    args = ap.parse_args()
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FROZEN.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.force:
        for d in ("raw", "bs_hfq"):
            rm(TMP / d)
        rm(TMP / "build.duckdb")
    con = connect()

    run = lambda name: "all" in steps or name in steps
    if run("symbols"):
        step_symbols(con)
    step_raw(con, args.force)
    step_hfq(con, args.force)
    if run("daily") or steps == "all":
        step_daily(con)
    if run("index") or steps == "all":
        step_index(con)
    if run("calendar") or steps == "all":
        step_calendar(con)
    if run("corp") or steps == "all":
        step_corp(con)
    if run("bj") or steps == "all":
        step_bj(con)
    if run("market") or steps == "all":
        step_market(con)
    if run("meta") or steps == "all":
        step_db(con)
    con.close()
    rm(TMP / "build.duckdb")
    log(f"全部完成，用时 {time.time()-t0:.0f}s（已清理 _tmp/build.duckdb）")


if __name__ == "__main__":
    main()
