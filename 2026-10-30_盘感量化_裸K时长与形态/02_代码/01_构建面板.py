#!/usr/bin/env python3
"""01 构建全历史面板（baostock 只读）→ work.duckdb

只做取数与质量审计，不做任何形态计算。
- 价格：adjustflag=1（后复权），用于形态与收益，除权不产生假跳空
- 量额：volume(股) / amount(元)，另用 amount/volume 还原真实成交均价（元），
  用于"低价股"过滤，避免再拉一份不复权数据
- 时点标记：is_st、trade_status(1=正常交易) 均来自当日记录
- 股票池：沪深主板/创业板/科创板（sh.60/sh.68/sz.00/sz.30），不含北交所与B股
"""
from __future__ import annotations

import os
import sqlite3
import time

import duckdb
import pandas as pd

BS = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "02_代码", "work.duckdb")
START, END = "1999-01-01", "2026-10-30"


def main() -> None:
    if os.path.exists(OUT):
        os.remove(OUT)
    con = duckdb.connect(OUT)
    con.execute("PRAGMA threads=8")

    # ---- 股票主表 ----
    src = sqlite3.connect(f"file:{BS}?mode=ro", uri=True)
    stocks = pd.read_sql_query(
        "SELECT code, code_name, ipo_date, out_date, type, status FROM stock_basic WHERE type=1",
        src,
    )
    stocks = stocks[stocks["code"].str.match(r"^(sh\.60|sh\.68|sz\.00|sz\.30)")]
    con.register("stocks_df", stocks)
    con.execute("CREATE TABLE stocks AS SELECT * FROM stocks_df")
    con.unregister("stocks_df")

    # ---- 日历 ----
    cal = pd.read_sql_query(
        "SELECT calendar_date AS date FROM trade_dates WHERE is_trading_day=1 "
        f"AND calendar_date BETWEEN '{START}' AND '{END}' ORDER BY 1",
        src,
    )
    con.register("cal_df", cal)
    con.execute("CREATE TABLE calendar AS SELECT date FROM cal_df")
    con.unregister("cal_df")

    # ---- 指数 ----
    idx = pd.read_sql_query(
        "SELECT code, date, open, high, low, close, preclose, volume, amount, pct_chg "
        "FROM index_kline_d WHERE code IN ('sh.000001','sh.000300','sh.000905','sh.000852') "
        f"AND date BETWEEN '{START}' AND '{END}'",
        src,
    )
    con.register("idx_df", idx)
    con.execute("CREATE TABLE indices AS SELECT * FROM idx_df")
    con.unregister("idx_df")

    # ---- 日线：按年走索引，避免整表扫描 ----
    con.execute("""
        CREATE TABLE bars(
            code VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            preclose DOUBLE, volume DOUBLE, amount DOUBLE, turn DOUBLE,
            trade_status INTEGER, is_st INTEGER, pct_chg DOUBLE
        )
    """)
    t0 = time.time()
    for y in range(1999, 2027):
        d0, d1 = f"{y}-01-01", f"{y + 1}-01-01"
        df = pd.read_sql_query(
            "SELECT code, date, open, high, low, close, preclose, volume, amount, turn, "
            "trade_status, is_st, pct_chg FROM stock_kline_d "
            "INDEXED BY idx_skd_flag_date "
            "WHERE adjustflag=1 AND date>=? AND date<? "
            "AND (code LIKE 'sh.60%' OR code LIKE 'sh.68%' OR code LIKE 'sz.00%' OR code LIKE 'sz.30%')",
            src,
            params=(d0, d1),
        )
        if len(df):
            con.register("chunk", df)
            con.execute("INSERT INTO bars SELECT * FROM chunk")
            con.unregister("chunk")
        print(y, len(df), f"{time.time() - t0:.0f}s", flush=True)
    src.close()

    # ---- 质量审计 ----
    con.execute("""
        CREATE TABLE audit AS
        SELECT 'bars' item, count(*)::DOUBLE n FROM bars
        UNION ALL SELECT 'codes', count(DISTINCT code)::DOUBLE FROM bars
        UNION ALL SELECT 'delisted_codes', count(DISTINCT code)::DOUBLE
            FROM stocks WHERE out_date IS NOT NULL AND out_date<>''
        UNION ALL SELECT 'dup_pk', (SELECT count(*)::DOUBLE FROM
            (SELECT code,date,count(*) c FROM bars GROUP BY 1,2 HAVING c>1))
        UNION ALL SELECT 'suspended_rows', count(*)::DOUBLE FROM bars WHERE trade_status=0
        UNION ALL SELECT 'st_rows', count(*)::DOUBLE FROM bars WHERE is_st=1
        UNION ALL SELECT 'zero_volume', count(*)::DOUBLE FROM bars WHERE volume IS NULL OR volume<=0
        UNION ALL SELECT 'null_open', count(*)::DOUBLE FROM bars WHERE open IS NULL OR low IS NULL
        UNION ALL SELECT 'dates', count(*)::DOUBLE FROM calendar
    """)
    con.execute("""
        CREATE TABLE audit_year AS
        SELECT substr(date,1,4) AS yr, count(*) n, count(DISTINCT code) codes,
               count(*) FILTER (WHERE is_st=1) st_rows,
               count(*) FILTER (WHERE trade_status=0) susp_rows,
               avg(amount/volume) avg_price
        FROM bars GROUP BY 1 ORDER BY 1
    """)
    # 后复权序列理论不应出现 >2 倍或 <0.3 倍的单日跳变（新股/长期停牌复牌除外，单独列出）
    con.execute("""
        CREATE TABLE audit_jumps AS
        SELECT code, date, close/preclose-1 AS ret
        FROM bars WHERE preclose>0 AND trade_status=1
          AND (close/preclose>2.0 OR close/preclose<0.3)
    """)
    print(con.execute("SELECT * FROM audit").fetchall())
    print(con.execute("SELECT count(*) FROM audit_jumps").fetchone())
    print(con.execute("SELECT * FROM audit_year").df().to_string())
    con.close()


if __name__ == "__main__":
    main()
