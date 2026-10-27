#!/usr/bin/env python3
"""补充审计证据：退市覆盖、002684 过程、旧官方逐笔对照、数据源清单。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "05_证据"
R = ROOT / "03_结果"
BARS = ROOT / "01_数据" / "bars.parquet"
PANEL = ROOT / "01_数据" / "panel.parquet"
OLD_TRADES = Path("/Users/admin/WorkSpace/trading/trading_flash/2026-10-24_主板恐慌反转_年化30/03_结果/实际价_修正版_逐笔.csv")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    con = duckdb.connect()
    st = pd.read_csv(E / "主板股票清单.csv")
    bars = con.execute(
        f"""SELECT code, min(date) AS first_date, max(date) AS last_date, count(*) AS n
            FROM read_parquet('{BARS}') GROUP BY code"""
    ).fetchdf()
    d = st.merge(bars, on="code", how="left")
    d["has_data"] = d.first_date.notna().astype(int)
    d["gap_days"] = (pd.to_datetime(d.out_date, errors="coerce") - pd.to_datetime(d.last_date, errors="coerce")).dt.days
    d.to_csv(E / "退市股覆盖.csv", index=False, encoding="utf-8-sig")
    print("退市覆盖", len(d), "有数据", int(d.has_data.sum()))

    x = con.execute(
        f"""SELECT date, open, high, low, close, preclose, volume, amount, turn,
                   trade_status, pct_chg, is_st, hfq_open, hfq_high, hfq_low, hfq_close
            FROM read_parquet('{BARS}')
            WHERE code='sz.002684' AND date BETWEEN '2022-04-15' AND '2022-06-15'
            ORDER BY date"""
    ).fetchdf()
    x.to_csv(E / "002684_退市过程.csv", index=False, encoding="utf-8-sig")
    print("002684 rows", len(x))

    old = pd.read_csv(OLD_TRADES)
    old["code"] = old.thscode.str[7:].str.lower() + "." + old.thscode.str[:6]
    old["entry_date"] = pd.to_datetime(old.entry_date)
    new = pd.read_csv(R / "2017_2026" / "A_仅当前上市_逐笔.csv", parse_dates=["entry_date"])
    m = old.merge(new, on=["entry_date", "code"], suffixes=("_旧", "_新"), how="outer", indicator=True)
    m["对照"] = m._merge.map({"both": "同日同码", "left_only": "仅旧官方", "right_only": "仅本文"})
    m["收益差"] = m["ret_新"] - m["ret_旧"]
    cols = ["对照", "code", "entry_date", "exit_date_旧", "exit_date_新", "ret_旧", "ret_新",
            "收益差", "exit_reason_旧", "exit_reason_新"]
    m[[c for c in cols if c in m.columns]].sort_values(["对照", "entry_date", "code"]).to_csv(
        E / "旧官方逐笔对照.csv", index=False, encoding="utf-8-sig")
    print("旧官方对照", m["对照"].value_counts().to_dict())

    manifest = {}
    for name, path in [("bars", BARS), ("panel", PANEL),
                       ("baostock_db", Path("/Users/admin/WorkSpace/baostock_data/data/baostock.db")),
                       ("hithink_db", Path("/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"))]:
        manifest[name] = dict(path=str(path), bytes=path.stat().st_size)
    manifest["panel_rows"] = int(con.execute(f"SELECT count(*) FROM read_parquet('{PANEL}')").fetchone()[0])
    manifest["bars_rows"] = int(con.execute(f"SELECT count(*) FROM read_parquet('{BARS}')").fetchone()[0])
    manifest["panel_symbols"] = int(con.execute(f"SELECT count(DISTINCT code) FROM read_parquet('{PANEL}')").fetchone()[0])
    manifest["panel_dates"] = int(con.execute(f"SELECT count(DISTINCT date) FROM read_parquet('{PANEL}')").fetchone()[0])
    (E / "数据源清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
