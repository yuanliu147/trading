#!/usr/bin/env python3
"""00 双源交叉核验：baostock 后复权 vs HiThink 后复权

抽样 30 只 × 2024-01-01～2026-10-29 的对数收益，比较最大绝对偏差。
"""
from __future__ import annotations

import os

import duckdb
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(HERE, "02_代码", "work.duckdb")
HX = os.path.expanduser("~/Library/Application Support/hithink-finance/data/market.duckdb")


def main() -> None:
    con = duckdb.connect(DB)
    con.execute(f"ATTACH '{HX}' AS hx (READ_ONLY)")
    codes = [r[0] for r in con.execute("""
        SELECT code FROM bars WHERE date >= DATE '2024-01-01'
        GROUP BY 1 HAVING count(*) > 400
        ORDER BY hash(code) LIMIT 30
    """).fetchall()]
    rows = []
    for c in codes:
        hx_code = c.split(".")[1] + "." + c.split(".")[0].upper()
        df = con.execute("""
            SELECT b.date, b.close AS bs, h.close AS hx
            FROM bars b JOIN hx.v_daily_hfq h
              ON h.thscode = ? AND h.date = b.date
            WHERE b.code = ? AND b.date >= DATE '2024-01-01'
            ORDER BY b.date
        """, [hx_code, c]).df()
        if len(df) < 100:
            continue
        rb = np.diff(np.log(df["bs"].to_numpy()))
        rh = np.diff(np.log(df["hx"].to_numpy()))
        diff = np.abs(rb - rh)
        rows.append((c, len(df), float(np.nanmax(diff)), float(np.nanmedian(diff))))
    out = sorted(rows, key=lambda x: -x[2])
    print("code, n, max_abs_diff, median_abs_diff")
    for r in out[:10]:
        print(r)
    print("checked", len(rows), "max", max(r[2] for r in out) if out else None)
    con.close()


if __name__ == "__main__":
    main()
