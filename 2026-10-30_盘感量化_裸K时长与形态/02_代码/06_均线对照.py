#!/usr/bin/env python3
"""06 去均线对照（H4）

1) 在形态事件内部，按"均线多头（收盘>MA20>MA60，t-1）"分组比较 60 日超额；
   差值用按 20 日块 bootstrap 的置信区间。
2) 均线派基线：仅 MA20 上穿 MA60 + 放量（无任何形态条件），同样本口径。
3) 全样本 MA 多头状态的前向超额（背景值）。
"""
from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(HERE, "02_代码", "work.duckdb")
OUT = os.path.join(HERE, "02_代码")


def stats_by_date(d: pd.DataFrame, label: str) -> dict:
    if len(d) == 0:
        return {"样本": label, "n": 0}
    daily = d.groupby("date")["ex60"].mean()
    return {
        "样本": label, "n": len(d), "天数": d["date"].nunique(),
        "超额20": d["ex20"].mean(), "超额60": d["ex60"].mean(), "超额120": d["ex120"].mean(),
        "中位超额60": d["ex60"].median(), "胜率60": (d["ex60"] > 0).mean(),
        "绝对60": d["ret60"].mean(),
        "日聚类t": nw_t(daily.to_numpy()),
    }


def nw_t(x, lag=5):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    mu = x.mean(); e = x - mu
    s = e @ e / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return mu / np.sqrt(s / n) if s > 0 else np.nan


def block_boot_diff(a: pd.DataFrame, b: pd.DataFrame, n_boot=500, block=20) -> tuple[float, float, float]:
    """按日期块重采样，返回 (差值, 2.5%, 97.5%)。"""
    da = a.groupby("date")["ex60"].mean()
    db = b.groupby("date")["ex60"].mean()
    dates = np.array(sorted(set(da.index) | set(db.index)))
    obs = da.mean() - db.mean()
    rng = np.random.default_rng(42)
    boots = []
    nb = int(np.ceil(len(dates) / block))
    for _ in range(n_boot):
        starts = rng.integers(0, len(dates), nb)
        idx = np.concatenate([np.arange(s, min(s + block, len(dates))) for s in starts])
        dd = dates[idx]
        va = da.reindex(dd).dropna()
        vb = db.reindex(dd).dropna()
        if len(va) > 10 and len(vb) > 10:
            boots.append(va.mean() - vb.mean())
    boots = np.array(boots)
    return obs, np.quantile(boots, 0.025), np.quantile(boots, 0.975)


def main() -> None:
    con = duckdb.connect(DB)
    con.execute("PRAGMA threads=8")
    con.execute("""
    CREATE OR REPLACE TABLE ma0 AS
    SELECT code, date, close,
           avg(close) OVER w20 AS ma20,
           avg(close) OVER w60 AS ma60
    FROM bars
    WINDOW w20 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
           w60 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
    """)
    con.execute("""
    CREATE OR REPLACE TABLE ma AS
    SELECT code, date, close, ma20, ma60,
           lag(ma20) OVER (PARTITION BY code ORDER BY date) AS pma20,
           lag(ma60) OVER (PARTITION BY code ORDER BY date) AS pma60
    FROM ma0
    """)
    con.execute("""
    CREATE OR REPLACE TABLE ma2 AS
    SELECT code, date,
           (close > ma20 AND ma20 > ma60) AS bull,
           (ma20 > ma60 AND pma20 <= pma60) AS ma_cross
    FROM ma WHERE ma20 IS NOT NULL AND ma60 IS NOT NULL
    """)

    base = """
        SELECT p.code, p.date, p.close, p.hh20_prev, p.vr, p.pos250, p.box_prev, p.comp_prev, p.vshr_prev,
               f.ret20, f.ret60, f.ret120,
               f.ret20 - bf.b20 AS ex20, f.ret60 - bf.b60 AS ex60, f.ret120 - bf.b120 AS ex120,
               m.bull, m.ma_cross
        FROM pre p
        JOIN fwd_ret f USING (code, date)
        JOIN bench_f bf USING (date)
        JOIN ma2 m USING (code, date)
        WHERE p.trade_status = 1 AND p.is_st = 0 AND p.bar_no >= 120
          AND p.vwap >= 3 AND p.amt20 >= 3e7
          AND f.entry_ok AND f.ret60 IS NOT NULL
    """
    con.execute(f"CREATE OR REPLACE TEMP VIEW allx AS {base}")
    # t-1 的均线状态
    con.execute("""
    CREATE OR REPLACE TEMP TABLE allx2 AS
    SELECT *, lag(bull) OVER (PARTITION BY code ORDER BY date) AS bull_prev,
           lag(ma_cross) OVER (PARTITION BY code ORDER BY date) AS cross_prev
    FROM allx
    """)

    rows = []
    for N, q, v, m, p in [(60, 0.7, 0.8, 1.5, 0.85), (30, 0.7, 0.8, 1.5, 0.85)]:
        ev = con.execute(f"""
            SELECT * FROM allx2
            WHERE box_prev >= {N} AND comp_prev <= {q} AND vshr_prev <= {v}
              AND close > hh20_prev AND vr >= {m} AND pos250 >= {p}
        """).df()
        a = ev[ev["bull_prev"] == True]
        b = ev[ev["bull_prev"] == False]
        rows.append(stats_by_date(a, f"形态 N={N} + 均线多头"))
        rows.append(stats_by_date(b, f"形态 N={N} + 非多头"))
        diff, lo, hi = block_boot_diff(a, b)
        rows.append({"样本": f"形态 N={N}：多头-非多头 差值", "n": len(ev),
                     "超额60": diff, "差值下限": lo, "差值上限": hi})
        print(N, "diff", round(diff, 4), "CI", round(lo, 4), round(hi, 4))

    # 均线派基线：当日金叉 + 放量
    cross = con.execute("""
        SELECT * FROM allx2 WHERE cross_prev = true AND vr >= 1.5
    """).df()
    rows.append(stats_by_date(cross, "均线金叉+放量（无形态）"))
    # 金叉 + 放量 + 横盘≥30（均线+形态）
    cross_box = con.execute("""
        SELECT * FROM allx2 WHERE cross_prev = true AND vr >= 1.5 AND box_prev >= 30
    """).df()
    rows.append(stats_by_date(cross_box, "均线金叉+放量+横盘≥30"))
    # 背景：全样本均线多头
    bull_all = con.execute("SELECT * FROM allx2 WHERE bull_prev = true").df()
    rows.append(stats_by_date(bull_all, "全样本均线多头（背景）"))
    non = con.execute("SELECT * FROM allx2 WHERE bull_prev = false").df()
    rows.append(stats_by_date(non, "全样本非多头（背景）"))

    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(OUT, "h4_ma.csv"), index=False)
    print(res.round(4).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
