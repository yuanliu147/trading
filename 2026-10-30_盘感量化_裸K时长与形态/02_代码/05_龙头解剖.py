#!/usr/bin/env python3
"""05 龙头解剖（H2，描述性）

每个自然年取后复权涨幅前 20（当年交易≥120日），找其当年内幅度最大的主升浪
（Zigzag θ=30% 的上涨腿），在起点前一日测量：
  前一段下跌腿时长、横盘游程、压缩、缩量、距一年高点、20日振幅
并计算这些值在"同日全样本"中的百分位（50 = 中位）。
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

    # 年度收益
    con.execute("""
    CREATE OR REPLACE TABLE yr AS
    WITH a AS (
        SELECT code, strftime(date,'%Y') AS y,
               arg_min(close, date) AS c0, arg_max(close, date) AS c1,
               count(*) AS nd
        FROM bars GROUP BY 1,2
    )
    SELECT code, y, c1/c0 - 1 AS ret, nd,
           row_number() OVER (PARTITION BY y ORDER BY c1/c0 DESC) AS rk
    FROM a WHERE nd >= 120
    """)

    # 同日横截面百分位
    con.execute("""
    CREATE OR REPLACE TABLE pct AS
    SELECT code, date,
           percent_rank() OVER (PARTITION BY date ORDER BY compress) AS p_comp,
           percent_rank() OVER (PARTITION BY date ORDER BY vshr)   AS p_vshr,
           percent_rank() OVER (PARTITION BY date ORDER BY pos250) AS p_pos,
           percent_rank() OVER (PARTITION BY date ORDER BY bw20)   AS p_bw
    FROM feat_box
    WHERE trade_status = 1 AND is_st = 0 AND bar_no >= 120
      AND compress IS NOT NULL AND vshr IS NOT NULL AND pos250 IS NOT NULL
    """)

    # 龙头 → 当年最大主升浪起点
    con.execute("""
    CREATE OR REPLACE TABLE leaders AS
    WITH top AS (
        SELECT y, code, ret, rk FROM yr WHERE rk <= 20 AND y BETWEEN '2000' AND '2026'
    ), leg AS (
        SELECT l.*, strftime(l.end_date,'%Y') AS y,
               row_number() OVER (PARTITION BY l.code, strftime(l.end_date,'%Y')
                                  ORDER BY l.amplitude DESC) AS leg_rk
        FROM legs l WHERE l.theta = 0.30 AND l.direction = 1
    )
    SELECT t.y, t.code, t.ret AS 年度涨幅, t.rk AS 年度排名,
           l.start_date AS 启动日, l.end_date AS 结束日,
           l.days AS 主升浪天数, l.amplitude AS 主升浪幅度
    FROM top t
    JOIN leg l ON l.code = t.code AND l.y = t.y AND l.leg_rk = 1
    """)

    # 起点前一日特征
    con.execute("""
    CREATE OR REPLACE TABLE leader_feat AS
    SELECT ld.*,
           p.box_prev AS 前横盘游程, p.comp_prev AS 压缩, p.vshr_prev AS 缩量,
           p.pos250 AS 位置, fb.bw20 AS 箱宽,
           pc.p_comp AS 压缩分位, pc.p_vshr AS 缩量分位, pc.p_pos AS 位置分位,
           pc.p_bw AS 箱宽分位,
           d.days AS 前下跌腿天数, d.amplitude AS 前下跌腿幅度
    FROM leaders ld
    LEFT JOIN pre p ON p.code = ld.code AND p.date = ld.启动日
    LEFT JOIN feat_box fb ON fb.code = ld.code AND fb.date = ld.启动日
    LEFT JOIN pct pc ON pc.code = ld.code AND pc.date = ld.启动日
    LEFT JOIN legs d ON d.theta = 0.30 AND d.code = ld.code
                    AND d.direction = -1 AND d.end_date = ld.启动日
    """)

    summ = con.execute("""
    SELECT count(*) AS 样本,
           median("前下跌腿天数") AS 前下跌腿天数_中位,
           median("前横盘游程") AS 前横盘游程_中位,
           median("压缩") AS 压缩_中位,
           median("缩量") AS 缩量_中位,
           median("位置") AS 位置_中位,
           median("主升浪天数") AS 主升浪天数_中位,
           median("主升浪幅度") AS 主升浪幅度_中位
    FROM leader_feat
    """).df()
    summ.round(3).to_csv(os.path.join(OUT, "h2_leader_summary.csv"), index=False)
    print(summ.round(3).to_string(index=False))

    pcts = con.execute("""
    SELECT '压缩' AS 指标, median(压缩分位) AS 分位中位, quantile_cont(压缩分位,0.25) AS 下四分位,
           quantile_cont(压缩分位,0.75) AS 上四分位 FROM leader_feat
    UNION ALL SELECT '缩量', median(缩量分位), quantile_cont(缩量分位,0.25), quantile_cont(缩量分位,0.75) FROM leader_feat
    UNION ALL SELECT '位置', median(位置分位), quantile_cont(位置分位,0.25), quantile_cont(位置分位,0.75) FROM leader_feat
    UNION ALL SELECT '箱宽', median(箱宽分位), quantile_cont(箱宽分位,0.25), quantile_cont(箱宽分位,0.75) FROM leader_feat
    """).df()
    pcts.round(3).to_csv(os.path.join(OUT, "h2_leader_pct.csv"), index=False)
    print(pcts.round(3).to_string(index=False))

    by_year = con.execute("""
    SELECT y AS 年份, count(*) AS 龙头数, median("前横盘游程") AS 前横盘中位,
           median("压缩") AS 压缩中位, median("缩量") AS 缩量中位, median("位置") AS 位置中位
    FROM leader_feat GROUP BY 1 ORDER BY 1
    """).df()
    by_year.round(3).to_csv(os.path.join(OUT, "h2_leader_by_year.csv"), index=False)
    print(by_year.round(2).to_string(index=False))

    # ---- 主样本：全部 ≥50%、≤250 日的主升浪 ----
    con.execute("""
    CREATE OR REPLACE TABLE rallies AS
    SELECT l.code, l.start_date, l.end_date, l.days, l.amplitude,
           p.box_prev AS 前横盘游程, p.comp_prev AS 压缩, p.vshr_prev AS 缩量,
           p.pos250 AS 位置, fb.bw20 AS 箱宽,
           pc.p_comp AS 压缩分位, pc.p_vshr AS 缩量分位, pc.p_pos AS 位置分位,
           pc.p_bw AS 箱宽分位,
           d.days AS 前下跌腿天数, d.amplitude AS 前下跌腿幅度
    FROM legs l
    LEFT JOIN pre p ON p.code = l.code AND p.date = l.start_date
    LEFT JOIN feat_box fb ON fb.code = l.code AND fb.date = l.start_date
    LEFT JOIN pct pc ON pc.code = l.code AND pc.date = l.start_date
    LEFT JOIN legs d ON d.theta = 0.30 AND d.code = l.code
                    AND d.direction = -1 AND d.end_date = l.start_date
    WHERE l.theta = 0.30 AND l.direction = 1 AND l.complete = false OR
          (l.theta = 0.30 AND l.direction = 1 AND l.complete AND l.amplitude >= 0.5)
    """)
    con.execute("""DELETE FROM rallies WHERE amplitude < 0.5 OR days > 250
                    OR start_date < DATE '2000-01-01'""")
    rsum = con.execute("""
    SELECT count(*) AS 样本, median(days) AS 主升浪天数_中位, median(amplitude) AS 幅度_中位,
           median(前下跌腿天数) AS 前下跌腿天数_中位, median(前下跌腿幅度) AS 前下跌腿幅度_中位,
           median(压缩) AS 压缩_中位, median(缩量) AS 缩量_中位, median(位置) AS 位置_中位,
           median(前横盘游程) AS 前横盘游程_中位
    FROM rallies
    """).df()
    rsum.round(3).to_csv(os.path.join(OUT, "h2_rally_summary.csv"), index=False)
    print(rsum.round(3).to_string(index=False))
    rpct = con.execute("""
    SELECT '压缩' AS 指标, median(压缩分位) AS 分位中位, quantile_cont(压缩分位,0.25) AS 下四分位,
           quantile_cont(压缩分位,0.75) AS 上四分位 FROM rallies
    UNION ALL SELECT '缩量', median(缩量分位), quantile_cont(缩量分位,0.25), quantile_cont(缩量分位,0.75) FROM rallies
    UNION ALL SELECT '位置', median(位置分位), quantile_cont(位置分位,0.25), quantile_cont(位置分位,0.75) FROM rallies
    UNION ALL SELECT '箱宽', median(箱宽分位), quantile_cont(箱宽分位,0.25), quantile_cont(箱宽分位,0.75) FROM rallies
    """).df()
    rpct.round(3).to_csv(os.path.join(OUT, "h2_rally_pct.csv"), index=False)
    print(rpct.round(3).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
