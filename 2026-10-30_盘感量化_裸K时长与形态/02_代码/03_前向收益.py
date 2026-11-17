#!/usr/bin/env python3
"""03 前向收益与同日等权基准

- 入场：T+1 开盘；T+1 停牌或一字涨停（open=high=low 且涨幅≥9.5%）视为不可成交
- 出场：T+20/60/120 收盘
- 基准：同一交易日、同一可成交口径下全样本等权前向收益
"""
from __future__ import annotations

import os

import duckdb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(HERE, "02_代码", "work.duckdb")


def main() -> None:
    con = duckdb.connect(DB)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='10GB'")

    con.execute("""
    CREATE OR REPLACE TABLE fwd AS
    SELECT code, date, close, open,
           lead(open, 1)  OVER w AS nopen,
           lead(high, 1)  OVER w AS nh,
           lead(low, 1)   OVER w AS nl,
           lead(trade_status, 1) OVER w AS nts,
           lead(pct_chg, 1) OVER w AS npct,
           lead(close, 20)  OVER w AS c20,
           lead(close, 60)  OVER w AS c60,
           lead(close, 120) OVER w AS c120
    FROM bars
    WINDOW w AS (PARTITION BY code ORDER BY date)
    """)

    con.execute("""
    CREATE OR REPLACE TABLE fwd_ret AS
    SELECT f.*,
           CASE WHEN nopen > 0 THEN c20 / nopen - 1 END AS ret20,
           CASE WHEN nopen > 0 THEN c60 / nopen - 1 END AS ret60,
           CASE WHEN nopen > 0 THEN c120 / nopen - 1 END AS ret120,
           (nts = 1 AND nopen IS NOT NULL
            AND NOT (nh = nl AND nl = nopen AND npct >= 9.5)) AS entry_ok
    FROM fwd f
    """)

    con.execute("""
    CREATE OR REPLACE TABLE bench AS
    SELECT date,
           avg(ret20) AS b20, avg(ret60) AS b60, avg(ret120) AS b120,
           count(ret20) AS n20, count(ret60) AS n60, count(ret120) AS n120
    FROM fwd_ret f
    JOIN feat e USING (code, date)
    WHERE entry_ok AND e.trade_status = 1 AND e.is_st = 0 AND e.bar_no >= 120
    GROUP BY 1 ORDER BY 1
    """)

    con.execute("""
    CREATE OR REPLACE TABLE bench_f AS
    SELECT f.date,
           avg(ret20) AS b20, avg(ret60) AS b60, avg(ret120) AS b120,
           count(ret20) AS n20, count(ret60) AS n60, count(ret120) AS n120
    FROM fwd_ret f
    JOIN feat e USING (code, date)
    WHERE entry_ok AND e.trade_status = 1 AND e.is_st = 0 AND e.bar_no >= 120
      AND e.vwap >= 3 AND e.amt20 >= 3e7
    GROUP BY 1 ORDER BY 1
    """)
    print(con.execute("SELECT * FROM bench_f ORDER BY date DESC LIMIT 3").df().to_string())
    print(con.execute("""
        SELECT strftime(date,'%Y') y, avg(b20) a20, avg(b60) a60, avg(b120) a120, avg(n60) n
        FROM bench GROUP BY 1 ORDER BY 1
    """).df().round(4).to_string())
    con.close()


if __name__ == "__main__":
    main()
