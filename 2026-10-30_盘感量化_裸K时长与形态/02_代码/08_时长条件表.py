#!/usr/bin/env python3
"""08 时长条件表：横盘游程 / 距一年高点天数 → 未来 60 日表现

描述性，不做信号判定。回答截图第①步的两个直觉：
  - 横盘越久，后面涨得越好？
  - 下跌（离一年高点）越久，反弹越大？
"""
from __future__ import annotations

import os

import duckdb
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(HERE, "02_代码", "work.duckdb")
OUT = os.path.join(HERE, "02_代码")


def main() -> None:
    con = duckdb.connect(DB)
    con.execute("PRAGMA threads=8")
    con.execute("""
    CREATE OR REPLACE TABLE dz AS
    SELECT code, date, bar_no, close, box_run, pos250,
           bar_no - arg_max(bar_no, close) OVER (
               PARTITION BY code ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
           ) AS dz250,
           max(high) OVER (PARTITION BY code ORDER BY date
                           ROWS BETWEEN 1 FOLLOWING AND 60 FOLLOWING) AS hh60
    FROM feat_box
    """)
    con.execute("""
    CREATE OR REPLACE TABLE cond AS
    SELECT d.code, d.date, d.box_run, d.dz250, d.close,
           d.hh60 / d.close - 1 AS 未来60日最大涨幅,
           f.ret60 - b.b60 AS ex60
    FROM dz d
    JOIN fwd_ret f USING (code, date)
    JOIN bench_f b USING (date)
    JOIN feat e USING (code, date)
    WHERE e.trade_status = 1 AND e.is_st = 0 AND e.bar_no >= 120
      AND e.vwap >= 3 AND e.amt20 >= 3e7
      AND f.entry_ok AND f.ret60 IS NOT NULL AND d.hh60 IS NOT NULL
    """)

    box = con.execute("""
    SELECT CASE WHEN box_run < 10 THEN '0-9'
                WHEN box_run < 20 THEN '10-19'
                WHEN box_run < 40 THEN '20-39'
                WHEN box_run < 80 THEN '40-79'
                ELSE '80+' END AS 横盘天数,
           count(*) AS 样本,
           avg(ex60) AS 平均超额60,
           median(ex60) AS 中位超额60,
           avg("未来60日最大涨幅") AS 平均未来60日最大涨幅,
           median("未来60日最大涨幅") AS 中位未来60日最大涨幅
    FROM cond GROUP BY 1 ORDER BY 1
    """).df()
    box.round(4).to_csv(os.path.join(OUT, "h5_box_condition.csv"), index=False)
    print(box.round(3).to_string(index=False))

    dz = con.execute("""
    SELECT CASE WHEN dz250 < 20 THEN '0-19'
                WHEN dz250 < 60 THEN '20-59'
                WHEN dz250 < 120 THEN '60-119'
                WHEN dz250 < 250 THEN '120-249'
                ELSE '250+' END AS 距一年高点天数,
           count(*) AS 样本,
           avg(ex60) AS 平均超额60,
           median(ex60) AS 中位超额60,
           avg("未来60日最大涨幅") AS 平均未来60日最大涨幅,
           median("未来60日最大涨幅") AS 中位未来60日最大涨幅
    FROM cond GROUP BY 1 ORDER BY 1
    """).df()
    dz.round(4).to_csv(os.path.join(OUT, "h5_dz_condition.csv"), index=False)
    print(dz.round(3).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
