#!/usr/bin/env python3
"""Build an independent A-share daily panel from baostock SQLite into DuckDB.

No project backtest results are read. Only raw bars, stock master, calendar,
and index bars are used.
"""
from __future__ import annotations

import os
import time
import duckdb

BS = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"
OUT = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/02_代码/work.duckdb"


def main() -> None:
    if os.path.exists(OUT):
        os.remove(OUT)
    con = duckdb.connect(OUT)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='12GB'")
    con.execute("LOAD sqlite")

    t0 = time.time()
    con.execute(f"""
        CREATE TABLE stocks AS
        SELECT code, code_name, ipo_date, out_date, type, status
        FROM sqlite_scan('{BS}', 'stock_basic')
        WHERE type = 1
          AND (code LIKE 'sh.60%' OR code LIKE 'sh.68%'
               OR code LIKE 'sz.00%' OR code LIKE 'sz.30%')
    """)
    print("stocks", con.execute("select count(*) from stocks").fetchone()[0], round(time.time() - t0, 1))

    t0 = time.time()
    con.execute(f"""
        CREATE TABLE bars AS
        SELECT code, date, adjustflag, open, high, low, close, preclose,
               volume, amount, turn, trade_status, is_st, pct_chg
        FROM sqlite_scan('{BS}', 'stock_kline_d')
        WHERE adjustflag IN (1, 3)
          AND date >= '2009-01-01' AND date <= '2026-10-28'
          AND (code LIKE 'sh.60%' OR code LIKE 'sh.68%'
               OR code LIKE 'sz.00%' OR code LIKE 'sz.30%')
    """)
    print("bars", con.execute("select count(*) from bars").fetchone()[0], round(time.time() - t0, 1))

    con.execute(f"""
        CREATE TABLE calendar AS
        SELECT calendar_date AS date
        FROM sqlite_scan('{BS}', 'trade_dates')
        WHERE is_trading_day = 1 AND calendar_date BETWEEN '2009-01-01' AND '2026-10-28'
        ORDER BY 1
    """)

    con.execute(f"""
        CREATE TABLE indices AS
        SELECT code, date, open, high, low, close, preclose, volume, amount, pct_chg
        FROM sqlite_scan('{BS}', 'index_kline_d')
        WHERE code IN ('sh.000300', 'sh.000905', 'sh.000852')
          AND date BETWEEN '2009-01-01' AND '2026-10-28'
    """)

    # Audit
    con.execute("""
        CREATE TABLE audit_counts AS
        SELECT 'stocks' AS item, count(*)::DOUBLE AS n FROM stocks
        UNION ALL SELECT 'delisted', count(*)::DOUBLE FROM stocks WHERE out_date IS NOT NULL AND out_date <> ''
        UNION ALL SELECT 'bars_hfq', count(*)::DOUBLE FROM bars WHERE adjustflag = 1
        UNION ALL SELECT 'bars_raw', count(*)::DOUBLE FROM bars WHERE adjustflag = 3
        UNION ALL SELECT 'bars_codes', count(DISTINCT code)::DOUBLE FROM bars
        UNION ALL SELECT 'calendar_days', count(*)::DOUBLE FROM calendar
        UNION ALL SELECT 'index_rows', count(*)::DOUBLE FROM indices
    """)
    con.execute("""
        CREATE TABLE audit_dupes AS
        SELECT code, date, adjustflag, count(*) AS n
        FROM bars GROUP BY 1, 2, 3 HAVING count(*) > 1
    """)
    con.execute("""
        CREATE TABLE audit_st_by_year AS
        SELECT substr(date, 1, 4) AS year,
               count(*) FILTER (WHERE is_st = 1) AS st_rows,
               count(DISTINCT code) FILTER (WHERE is_st = 1) AS st_codes,
               count(*) AS rows
        FROM bars WHERE adjustflag = 3 GROUP BY 1 ORDER BY 1
    """)
    con.execute("""
        CREATE TABLE audit_coverage AS
        SELECT code, min(date) AS first_date, max(date) AS last_date, count(*) AS n
        FROM bars WHERE adjustflag = 3 GROUP BY 1
    """)
    print("audit done", round(time.time() - t0, 1))
    print(con.execute("select * from audit_counts").fetchall())
    print("dupes", con.execute("select count(*) from audit_dupes").fetchone()[0])
    con.close()


if __name__ == "__main__":
    main()
