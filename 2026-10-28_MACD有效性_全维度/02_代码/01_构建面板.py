#!/usr/bin/env python3
"""01 构建 MACD 有效性研究面板

数据源与口径
------------
- 主数据：本地 baostock SQLite（只读）。不复权 adjustflag=3；后复权 adjustflag=1。
  后复权价用于 MACD 与收益（连续、无未来分红改写）；不复权价用于涨跌停/停牌/ST 判定。
- 补全/校验：本地 HiThink DuckDB（只读）。2026-10-28 抽样 11,789 行逐位比对，
  open/high/low/close 最大差 0.0，volume/amount 相对差 <1e-7，确认两库同源。
  DuckDB 独有的是北交所（920xxx.BJ），baostock 无；本研究限定沪深 A 股（不含北交所），
  以保持 2010-2026 全历史一致。baostock 独有的是 337 只已退市股，正是生存偏差修正来源。
- 股票池：stock_basic.type=1（5,564 只，其中 341 只已退市）。
- 区间：2008-01-01 至 2026-10-28；信号评估从 2010-01-01 起（留足 250 日预热）。

输出
----
03_结果/面板_日线.parquet        全 A 日线（raw + hfq）
03_结果/指数_日线.parquet        CSI300/500/上证/创业板指
03_结果/00_数据审计.csv          行数/代码数/重叠校验/缺失
"""
import os
import duckdb
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "03_结果")
os.makedirs(OUT, exist_ok=True)

BS_DB = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"
HX_DB = os.path.expanduser("~/Library/Application Support/hithink-finance/data/market.duckdb")

con = duckdb.connect()
con.execute("INSTALL sqlite; LOAD sqlite;")
con.execute(f"ATTACH '{BS_DB}' AS bs (TYPE sqlite, READ_ONLY)")
con.execute(f"ATTACH '{HX_DB}' AS hx (READ_ONLY)")

# ---------- 1. 股票池 ----------
con.execute("""
CREATE OR REPLACE TABLE universe AS
SELECT code AS bs_code,
       substr(code,4) || '.' || upper(substr(code,1,2)) AS thscode,
       code_name, ipo_date, out_date, status
FROM bs.stock_basic WHERE type = 1
""")
print("universe:", con.execute("SELECT count(*), sum(status=0) FROM universe").fetchone())

# ---------- 2. 日线面板 ----------
con.execute("""
CREATE OR REPLACE TABLE panel AS
WITH raw AS (
  SELECT k.code, k.date, k.open, k.high, k.low, k.close, k.preclose,
         k.volume, k.amount, k.turn, k.trade_status, k.pct_chg, k.is_st
  FROM bs.stock_kline_d k
  JOIN universe u ON u.bs_code = k.code
  WHERE k.adjustflag = 3 AND k.date >= '2008-01-01'
),
hfq AS (
  SELECT k.code, k.date, k.open AS hfq_open, k.high AS hfq_high,
         k.low AS hfq_low, k.close AS hfq_close
  FROM bs.stock_kline_d k
  JOIN universe u ON u.bs_code = k.code
  WHERE k.adjustflag = 1 AND k.date >= '2008-01-01'
)
SELECT u.thscode, u.bs_code, u.code_name, u.ipo_date, u.out_date, u.status,
       CAST(r.date AS DATE) AS date,
       r.open, r.high, r.low, r.close, r.preclose,
       r.volume, r.amount, r.turn, r.trade_status, r.pct_chg, r.is_st,
       h.hfq_open, h.hfq_high, h.hfq_low, h.hfq_close
FROM raw r
JOIN hfq h ON h.code = r.code AND h.date = r.date
JOIN universe u ON u.bs_code = r.code
""")
print("panel rows:", con.execute("SELECT count(*) FROM panel").fetchone())
print("panel codes/dates:", con.execute("SELECT count(DISTINCT thscode), min(date), max(date) FROM panel").fetchone())

# 重复键检查
dup = con.execute("""
SELECT count(*) FROM (SELECT thscode, date, count(*) c FROM panel GROUP BY 1,2 HAVING c>1)
""").fetchone()[0]
print("duplicate keys:", dup)

# ---------- 3. 指数 ----------
con.execute("""
CREATE OR REPLACE TABLE idx AS
SELECT code AS bs_code,
       CASE code
         WHEN 'sh.000300' THEN 'CSI300'
         WHEN 'sh.000905' THEN 'CSI500'
         WHEN 'sh.000001' THEN 'SHCOMP'
         WHEN 'sz.399006' THEN 'CYB'
         WHEN 'sh.000852' THEN 'CSI1000'
       END AS idx_name,
       CAST(date AS DATE) AS date, open, high, low, close, preclose, volume, amount, pct_chg
FROM bs.index_kline_d
WHERE code IN ('sh.000300','sh.000905','sh.000001','sz.399006','sh.000852')
""")

# ---------- 4. 与 DuckDB 的全量重叠校验 ----------
# 按 (thscode, date) 连接两边不复权 OHLC，统计不一致行数
overlap = con.execute("""
WITH d AS (
  SELECT thscode, date, open, high, low, close
  FROM hx.v_daily
  WHERE date >= '2016-10-13'
)
SELECT count(*) AS n,
       sum(CASE WHEN abs(p.open-d.open)>0.011 OR abs(p.close-d.close)>0.011
                 OR abs(p.high-d.high)>0.011 OR abs(p.low-d.low)>0.011 THEN 1 ELSE 0 END) AS n_bad,
       max(abs(p.open-d.open)) AS max_open_diff,
       max(abs(p.close-d.close)) AS max_close_diff
FROM panel p JOIN d ON p.thscode=d.thscode AND p.date=d.date
""").fetchone()
print("duckdb overlap:", overlap)

# 覆盖率：baostock 有而 duck 无（应主要是 2016 前/退市），duck 有而 baostock 无（北交所）
cov = con.execute("""
SELECT
 (SELECT count(DISTINCT thscode) FROM panel) AS bs_codes,
 (SELECT count(DISTINCT thscode) FROM hx.v_daily WHERE date>='2016-10-13') AS hx_codes,
 (SELECT count(*) FROM (SELECT DISTINCT thscode FROM hx.v_daily WHERE date>='2016-10-13'
    EXCEPT SELECT DISTINCT thscode FROM panel)) AS only_hx,
 (SELECT count(*) FROM (SELECT DISTINCT thscode FROM panel
    EXCEPT SELECT DISTINCT thscode FROM hx.v_daily WHERE date>='2016-10-13')) AS only_bs
""").fetchone()
print("coverage:", cov)

# ---------- 5. 导出 ----------
con.execute(f"COPY panel TO '{OUT}/面板_日线.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)")
con.execute(f"COPY idx TO '{OUT}/指数_日线.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)")

audit = pd.DataFrame([
    dict(项目="股票池代码数", 数值=con.execute("SELECT count(*) FROM universe").fetchone()[0]),
    dict(项目="退市股代码数", 数值=con.execute("SELECT count(*) FROM universe WHERE status=0").fetchone()[0]),
    dict(项目="面板行数", 数值=con.execute("SELECT count(*) FROM panel").fetchone()[0]),
    dict(项目="面板代码数", 数值=con.execute("SELECT count(DISTINCT thscode) FROM panel").fetchone()[0]),
    dict(项目="起始日", 数值=str(con.execute("SELECT min(date) FROM panel").fetchone()[0])),
    dict(项目="结束日", 数值=str(con.execute("SELECT max(date) FROM panel").fetchone()[0])),
    dict(项目="重复键行数", 数值=dup),
    dict(项目="与DuckDB重叠行数", 数值=overlap[0]),
    dict(项目="OHLC不一致行数(>0.011)", 数值=overlap[1]),
    dict(项目="最大开盘差", 数值=overlap[2]),
    dict(项目="最大收盘差", 数值=overlap[3]),
    dict(项目="仅DuckDB有代码数(北交所等)", 数值=cov[2]),
    dict(项目="仅baostock有代码数(退市)", 数值=cov[3]),
])
audit.to_csv(os.path.join(OUT, "00_数据审计.csv"), index=False, encoding="utf-8-sig")
print(audit.to_string(index=False))
print("done")
