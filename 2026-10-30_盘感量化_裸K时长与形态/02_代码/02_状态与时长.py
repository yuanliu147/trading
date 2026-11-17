#!/usr/bin/env python3
"""02 裸K特征 + 涨跌腿(Zigzag) + 箱体游程（H1）

修订版：弃用 20 日三状态机（游程中位数 2 天，见预注册修订记录），改为
  A. Zigzag 涨跌腿：θ=20%，敏感性 10/15/25/30%
  B. 箱体：箱宽20=(20日高-20日低)/收盘 ≤15%，连续在箱体内天数=横盘游程

输出：work.duckdb 的 feat / legs / box_runs，以及 h1_*.csv
"""
from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(HERE, "02_代码", "work.duckdb")
THETAS = [0.10, 0.15, 0.20, 0.25, 0.30]
BOX_MAIN = 0.15


def build_feat(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE OR REPLACE TABLE feat AS
    WITH f AS (
        SELECT code, date, open, high, low, close, preclose, volume, amount, turn,
               trade_status, is_st,
               CASE WHEN volume > 0 THEN amount / volume END AS vwap,
               row_number() OVER (PARTITION BY code ORDER BY date) AS bar_no,
               lag(close) OVER (PARTITION BY code ORDER BY date) AS pc
        FROM bars
    ), g AS (
        SELECT *,
            CASE WHEN pc > 0 THEN ln(close / pc) END AS lr,
            ln(close / lag(close, 20) OVER (PARTITION BY code ORDER BY date)) AS r20,
            avg(volume)  OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS v20,
            avg(volume)  OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) AS v250,
            avg(amount)  OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS amt20,
            max(high)    OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND 1 PRECEDING) AS hh20_prev,
            max(close)   OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) AS hc250,
            max(high)    OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS hh20,
            min(low)     OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ll20
        FROM f
    ), h AS (
        SELECT *,
            stddev_samp(lr) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) AS sd250,
            count(lr)       OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) AS n250,
            sum(abs(lr))    OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS absum20,
            greatest(high - low, abs(high - preclose), abs(low - preclose)) AS tr_
        FROM g
    )
    SELECT code, date, open, high, low, close, preclose, volume, amount, turn,
           trade_status, is_st, vwap, bar_no,
           r20, absum20,
           CASE WHEN absum20 > 0 THEN abs(r20) / absum20 END AS er20,
           sd250, n250,
           CASE WHEN sd250 > 0 AND n250 >= 120 THEN r20 / (sd250 * sqrt(20)) END AS z20,
           avg(tr_) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) / close AS atr20,
           avg(tr_) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) / close AS atr250,
           v20, v250, amt20, hh20_prev, hc250, hh20, ll20,
           (hh20 - ll20) / NULLIF(close, 0) AS bw20,
           CASE WHEN v250 > 0 THEN v20 / v250 END AS vshr,
           CASE WHEN hc250 > 0 THEN close / hc250 END AS pos250,
           CASE WHEN atr250 > 0 THEN atr20 / atr250 END AS compress,
           CASE WHEN v20 > 0 THEN volume / v20 END AS vr,
           CASE WHEN close > 0 THEN hc250 / close - 1 END AS drawdown250,
           ((hh20 - ll20) / NULLIF(close, 0)) <= 0.15 AS box_main
    FROM h
    """)

    # 箱体游程（gaps & islands，箱体条件含当日）
    con.execute("""
    CREATE OR REPLACE TABLE feat_box AS
    WITH b AS (
        SELECT *, coalesce(box_main, false) AS bx FROM feat
    ), s AS (
        SELECT *, CASE WHEN bx IS DISTINCT FROM lag(bx) OVER (PARTITION BY code ORDER BY date)
                       THEN 1 ELSE 0 END AS brk
        FROM b
    ), g AS (
        SELECT *, sum(brk) OVER (PARTITION BY code ORDER BY date
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rid
        FROM s
    )
    SELECT *, CASE WHEN bx THEN count(*) OVER (PARTITION BY code, rid) ELSE 0 END AS box_run
    FROM g
    """)
    con.execute("""
    CREATE OR REPLACE TABLE box_runs AS
    SELECT code, rid, min(date) AS start_date, max(date) AS end_date, count(*) AS len
    FROM feat_box WHERE bx GROUP BY 1, 2
    """)


def zigzag(close: np.ndarray, theta: float):
    """返回 [(i_start, i_end, direction, amplitude)]；direction=1 上涨腿，-1 下跌腿。"""
    legs = []
    n = len(close)
    if n < 30:
        return legs
    pivot = 0
    direction = 0
    ext_i = 0
    ext = close[0]
    for i in range(1, n):
        p = close[i]
        if direction >= 0:
            if p > ext:
                ext, ext_i = p, i
            elif p <= ext * (1 - theta):
                if direction == 1:
                    legs.append((pivot, ext_i, 1, ext / close[pivot] - 1, True))
                direction = -1
                pivot, ext, ext_i = ext_i, p, i
                continue
        if direction <= 0:
            if p < ext:
                ext, ext_i = p, i
            elif p >= ext * (1 + theta):
                if direction == -1:
                    legs.append((pivot, ext_i, -1, close[pivot] / ext - 1, True))
                direction = 1
                pivot, ext, ext_i = ext_i, p, i
    # 收尾：未完成腿也要记录，否则单边上涨（无 30% 回撤）的龙头一条腿都没有
    if direction == 1 and ext_i > pivot:
        legs.append((pivot, ext_i, 1, ext / close[pivot] - 1, False))
    elif direction == -1 and ext_i > pivot:
        legs.append((pivot, ext_i, -1, close[pivot] / ext - 1, False))
    elif direction == 0:
        hi, lo = int(np.argmax(close)), int(np.argmin(close))
        if hi > lo and close[hi] / close[lo] - 1 >= theta:
            legs.append((lo, hi, 1, close[hi] / close[lo] - 1, False))
        elif lo > hi and 1 - close[lo] / close[hi] >= theta:
            legs.append((hi, lo, -1, 1 - close[lo] / close[hi], False))
    return legs


def build_legs(con: duckdb.DuckDBPyConnection) -> None:
    codes = [r[0] for r in con.execute("SELECT DISTINCT code FROM bars ORDER BY 1").fetchall()]
    con.execute("""
        CREATE OR REPLACE TABLE legs(
            theta DOUBLE, code VARCHAR, i_start BIGINT, i_end BIGINT,
            start_date DATE, end_date DATE, direction INTEGER, days INTEGER, amplitude DOUBLE, complete BOOLEAN
        )
    """)
    for theta in THETAS:
        rows = []
        for code in codes:
            df = con.execute(
                "SELECT date, close FROM bars WHERE code=? ORDER BY date", [code]
            ).df()
            if len(df) < 30:
                continue
            close = df["close"].to_numpy(dtype=float)
            dates = df["date"].to_numpy()
            for a, b, d, amp, done in zigzag(close, theta):
                rows.append((theta, code, a, b, dates[a], dates[b], d, b - a, amp, done))
        if rows:
            df = pd.DataFrame(rows, columns=["theta","code","i_start","i_end",
                                             "start_date","end_date","direction","days","amplitude","complete"])
            cols = ["theta","code","i_start","i_end","start_date","end_date","direction","days","amplitude","complete"]
            df = df[cols]
            con.register("leg_chunk", df)
            con.execute("INSERT INTO legs SELECT * FROM leg_chunk")
            con.unregister("leg_chunk")
        print("theta", theta, "legs", len(rows), flush=True)


def h1_stats(con: duckdb.DuckDBPyConnection) -> None:
    # 主 θ=20%
    legs = con.execute("""
        SELECT CASE direction WHEN 1 THEN '上涨腿' ELSE '下跌腿' END AS 方向,
               count(*) AS 段数, avg(days) AS 均值天数, median(days) AS 中位天数,
               quantile_cont(days,0.25) AS 四分位25, quantile_cont(days,0.75) AS 四分位75,
               quantile_cont(days,0.90) AS 分位90,
               avg(amplitude) AS 均幅, median(amplitude) AS 中位幅
        FROM legs WHERE theta=0.20 AND complete AND start_date >= DATE '2000-01-01'
        GROUP BY 1 ORDER BY 1
    """).df()
    legs.round(3).to_csv(os.path.join(HERE, "02_代码", "h1_legs.csv"), index=False)
    print(legs.to_string(index=False))

    by_dec = con.execute("""
        SELECT strftime(start_date,'%Y') AS 起始年,
               CASE direction WHEN 1 THEN '上涨腿' ELSE '下跌腿' END AS 方向,
               count(*) AS 段数, median(days) AS 中位天数, median(amplitude) AS 中位幅
        FROM legs WHERE theta=0.20 AND complete AND start_date >= DATE '2000-01-01'
        GROUP BY 1,2 ORDER BY 1,2
    """).df()
    by_dec.round(3).to_csv(os.path.join(HERE, "02_代码", "h1_legs_by_year.csv"), index=False)

    box = con.execute("""
        SELECT count(*) AS 段数, avg(len) AS 均值天数, median(len) AS 中位天数,
               quantile_cont(len,0.25) AS 四分位25, quantile_cont(len,0.75) AS 四分位75,
               quantile_cont(len,0.90) AS 分位90, max(len) AS 最长,
               sum(len) AS 总天数
        FROM box_runs
        WHERE len >= 5 AND start_date >= DATE '2000-01-01'
    """).df()
    box.round(2).to_csv(os.path.join(HERE, "02_代码", "h1_box.csv"), index=False)
    print(box.to_string(index=False))

    # θ 敏感性
    sens = con.execute("""
        SELECT theta AS θ, direction AS 方向, count(*) AS 段数,
               median(days) AS 中位天数, median(amplitude) AS 中位幅,
               quantile_cont(days,0.9) AS 分位90天数
        FROM legs WHERE complete AND start_date >= DATE '2000-01-01'
        GROUP BY 1,2 ORDER BY 1,2
    """).df()
    sens.round(3).to_csv(os.path.join(HERE, "02_代码", "h1_sensitivity.csv"), index=False)
    print(sens.to_string(index=False))


def main() -> None:
    con = duckdb.connect(DB)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='10GB'")
    print("build feat ...", flush=True)
    build_feat(con)
    print("build legs ...", flush=True)
    build_legs(con)
    print("h1 ...", flush=True)
    h1_stats(con)
    con.close()


if __name__ == "__main__":
    main()
