from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
INITIAL = ROOT.parent / "初始研究"
DUCKDB_PANEL = INITIAL / "data" / "snapshot" / "mainboard_daily.parquet"
CANDIDATES = ROOT / "results" / "entry_candidates.csv"
BAOSTOCK_DB = Path("/Users/admin/WorkSpace/baostock_data/data/baostock.db")
OUTPUT = ROOT / "results" / "baostock_optimization"
CACHE = OUTPUT / "cache"
MAINBOARD_PARQUET = CACHE / "mainboard_adjust_1_3.parquet"
SYMBOLS_PARQUET = CACHE / "symbols.parquet"
START = "2016-10-13"
END = "2026-10-23"
PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
KLINE_COLUMNS = [
    "date", "adjustflag", "open", "high", "low", "close", "preclose",
    "volume", "amount", "turn", "trade_status", "pct_chg", "pe_ttm",
    "pb_mrq", "ps_ttm", "pcf_ncf_ttm", "is_st",
]
SCHEMA = pa.schema([
    ("thscode", pa.string()),
    ("date", pa.date32()),
    ("adjustflag", pa.int8()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("preclose", pa.float64()),
    ("volume", pa.float64()),
    ("amount", pa.float64()),
    ("turn", pa.float64()),
    ("trade_status", pa.int8()),
    ("pct_chg", pa.float64()),
    ("pe_ttm", pa.float64()),
    ("pb_mrq", pa.float64()),
    ("ps_ttm", pa.float64()),
    ("pcf_ncf_ttm", pa.float64()),
    ("is_st", pa.int8()),
])


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def literal(path: Path) -> str:
    return str(path).replace("'", "''")


def baostock_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{BAOSTOCK_DB}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def load_symbols(connection: sqlite3.Connection) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in PREFIXES)
    rows = connection.execute(
        f"""SELECT code, code_name, ipo_date, out_date, status
            FROM stock_basic
            WHERE type=1 AND substr(code, 4, 3) IN ({placeholders})
            ORDER BY code""",
        PREFIXES,
    ).fetchall()
    frame = pd.DataFrame(rows, columns=["code", "code_name", "ipo_date", "out_date", "status"])
    frame["thscode"] = frame["code"].str[3:] + "." + frame["code"].str[:2].str.upper()
    frame["listing_status"] = frame["status"].map({0: "delisted", 1: "listed"})
    return frame[["thscode", "code", "code_name", "ipo_date", "out_date", "status", "listing_status"]]


def export_mainboard() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    connection = baostock_connection()
    symbols = load_symbols(connection)
    temporary = MAINBOARD_PARQUET.with_suffix(".tmp")
    if temporary.exists():
        temporary.unlink()
    writer = pq.ParquetWriter(temporary, SCHEMA, compression="zstd")
    rows_written = 0
    started = time.monotonic()
    sql = f"""SELECT {','.join(KLINE_COLUMNS)}
              FROM stock_kline_d
              WHERE code=? AND date BETWEEN ? AND ? AND adjustflag IN (1, 3)
              ORDER BY date, adjustflag"""
    try:
        for offset in range(0, len(symbols), 50):
            batch = symbols.iloc[offset:offset + 50]
            frames = []
            for row in batch.itertuples(index=False):
                rows = connection.execute(sql, (row.code, START, END)).fetchall()
                if not rows:
                    continue
                frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
                frame.insert(0, "thscode", row.thscode)
                frames.append(frame)
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                combined["date"] = pd.to_datetime(combined["date"]).dt.date
                table = pa.Table.from_pandas(combined, schema=SCHEMA, preserve_index=False)
                writer.write_table(table)
                rows_written += table.num_rows
            if offset % 250 == 0 or offset + 50 >= len(symbols):
                print(json.dumps({
                    "stage": "export", "symbols": min(offset + 50, len(symbols)),
                    "total_symbols": len(symbols), "rows": rows_written,
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }), flush=True)
    finally:
        writer.close()
        connection.close()
    os.replace(temporary, MAINBOARD_PARQUET)
    symbols.to_parquet(SYMBOLS_PARQUET, index=False)
    manifest = {
        "source": str(BAOSTOCK_DB),
        "source_bytes": BAOSTOCK_DB.stat().st_size,
        "source_mtime_ns": BAOSTOCK_DB.stat().st_mtime_ns,
        "start": START,
        "end": END,
        "adjustflags": {"1": "backward_adjusted", "3": "unadjusted"},
        "symbols": len(symbols),
        "listed_symbols": int((symbols["status"] == 1).sum()),
        "delisted_symbols": int((symbols["status"] == 0).sum()),
        "rows": rows_written,
        "elapsed_seconds": time.monotonic() - started,
        "parquet": str(MAINBOARD_PARQUET),
        "parquet_bytes": MAINBOARD_PARQUET.stat().st_size,
    }
    write_json(CACHE / "manifest.json", manifest)
    return manifest


def audit_data() -> dict:
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit TO '6GB'")
    con.execute(f"CREATE VIEW bs_all AS SELECT * FROM read_parquet('{literal(MAINBOARD_PARQUET)}')")
    con.execute(f"CREATE VIEW bs_symbols AS SELECT * FROM read_parquet('{literal(SYMBOLS_PARQUET)}')")
    con.execute(f"CREATE VIEW duck_daily AS SELECT * FROM read_parquet('{literal(DUCKDB_PANEL)}')")
    con.execute("CREATE VIEW bs_raw AS SELECT * FROM bs_all WHERE adjustflag=3")
    con.execute("CREATE VIEW bs_hfq AS SELECT * FROM bs_all WHERE adjustflag=1")
    price = con.execute("""
        WITH b AS (
            SELECT r.* FROM bs_raw r
            JOIN (SELECT DISTINCT thscode FROM duck_daily) d USING (thscode)
        ), joined AS (
            SELECT b.thscode AS b_thscode, d.thscode AS d_thscode,
                   b.open AS b_open, b.high AS b_high, b.low AS b_low, b.close AS b_close,
                   b.volume AS b_volume, b.amount AS b_amount,
                   d.open AS d_open, d.high AS d_high, d.low AS d_low, d.close AS d_close,
                   d.volume AS d_volume, d.amount AS d_amount
            FROM b FULL OUTER JOIN duck_daily d ON b.thscode=d.thscode AND b.date=d.date
        )
        SELECT count(*) AS union_rows,
               count(b_thscode) AS baostock_rows,
               count(d_thscode) AS duckdb_rows,
               count(*) FILTER (WHERE b_thscode IS NOT NULL AND d_thscode IS NOT NULL) AS matched_rows,
               count(*) FILTER (WHERE b_thscode IS NOT NULL AND d_thscode IS NULL) AS baostock_only_rows,
               count(*) FILTER (WHERE b_thscode IS NULL AND d_thscode IS NOT NULL) AS duckdb_only_rows,
               count(*) FILTER (WHERE b_thscode IS NOT NULL AND d_thscode IS NOT NULL AND (
                   abs(b_open-d_open)>0.0001 OR abs(b_high-d_high)>0.0001 OR
                   abs(b_low-d_low)>0.0001 OR abs(b_close-d_close)>0.0001)) AS ohlc_mismatch_rows,
               max(abs(b_close-d_close)) AS max_close_abs_diff,
               count(*) FILTER (WHERE b_thscode IS NOT NULL AND d_thscode IS NOT NULL AND abs(b_volume-d_volume)>0.5) AS volume_mismatch_rows,
               count(*) FILTER (WHERE b_thscode IS NOT NULL AND d_thscode IS NOT NULL AND abs(b_amount-d_amount)>0.01) AS amount_mismatch_rows,
               max(abs(b_amount-d_amount)) AS max_amount_abs_diff
        FROM joined
    """).fetchdf().iloc[0].to_dict()
    hfq = con.execute("""
        SELECT count(*) AS matched_rows,
               median(abs(b.close/nullif(d.hfq_close,0)-1)) AS median_close_relative_diff,
               quantile_cont(abs(b.close/nullif(d.hfq_close,0)-1), 0.99) AS p99_close_relative_diff,
               max(abs(b.close/nullif(d.hfq_close,0)-1)) AS max_close_relative_diff,
               count(*) FILTER (WHERE abs(b.close/nullif(d.hfq_close,0)-1)>0.00001) AS close_relative_diff_gt_1e5
        FROM bs_hfq b JOIN duck_daily d ON b.thscode=d.thscode AND b.date=d.date
    """).fetchdf().iloc[0].to_dict()
    coverage = con.execute("""
        SELECT s.listing_status, year(r.date) AS year, count(*) AS rows,
               count(DISTINCT r.thscode) AS symbols,
               count(*) FILTER (WHERE r.trade_status=0) AS suspended_rows,
               count(*) FILTER (WHERE r.is_st=1) AS st_rows,
               count(DISTINCT r.thscode) FILTER (WHERE r.is_st=1) AS st_symbols,
               count(*) FILTER (WHERE r.trade_status=1 AND r.turn IS NULL) AS normal_rows_missing_turn
        FROM bs_raw r JOIN bs_symbols s USING (thscode)
        GROUP BY s.listing_status, year(r.date)
        ORDER BY s.listing_status, year(r.date)
    """).fetchdf()
    delisted = con.execute("""
        SELECT s.thscode, s.code_name, s.ipo_date, s.out_date,
               min(r.date) AS first_bar, max(r.date) AS last_bar, count(r.date) AS bars,
               count(*) FILTER (WHERE r.is_st=1) AS st_days
        FROM bs_symbols s LEFT JOIN bs_raw r USING (thscode)
        WHERE s.status=0
        GROUP BY s.thscode, s.code_name, s.ipo_date, s.out_date
        ORDER BY s.out_date, s.thscode
    """).fetchdf()
    con.register("candidate_frame", pd.read_csv(CANDIDATES))
    candidate_coverage = con.execute("""
        SELECT count(*) AS candidates,
               count(r.date) AS matched,
               count(*) FILTER (WHERE r.is_st=1) AS st_candidates,
               count(DISTINCT c.thscode) FILTER (WHERE r.is_st=1) AS st_symbols,
               count(*) FILTER (WHERE r.turn IS NULL) AS missing_turn,
               min(r.turn) AS min_turn,
               median(r.turn) AS median_turn,
               max(r.turn) AS max_turn
        FROM candidate_frame c
        LEFT JOIN bs_raw r ON c.thscode=r.thscode AND CAST(c.signal_date AS DATE)=r.date
    """).fetchdf().iloc[0].to_dict()
    st_candidates = con.execute("""
        SELECT c.signal_date, c.thscode, c.rank, c.score, s.code_name, s.listing_status,
               r.turn, r.close, r.volume, r.amount, r.is_st
        FROM candidate_frame c
        JOIN bs_raw r ON c.thscode=r.thscode AND CAST(c.signal_date AS DATE)=r.date
        JOIN bs_symbols s ON s.thscode=c.thscode
        WHERE r.is_st=1
        ORDER BY c.signal_date, c.rank
    """).fetchdf()
    mismatch_examples = con.execute("""
        WITH b AS (
            SELECT r.* FROM bs_raw r
            JOIN (SELECT DISTINCT thscode FROM duck_daily) d USING (thscode)
        )
        SELECT b.thscode, b.date, b.open AS baostock_open, d.open AS duckdb_open,
               b.close AS baostock_close, d.close AS duckdb_close,
               b.volume AS baostock_volume, d.volume AS duckdb_volume,
               b.amount AS baostock_amount, d.amount AS duckdb_amount
        FROM b JOIN duck_daily d ON b.thscode=d.thscode AND b.date=d.date
        WHERE abs(b.open-d.open)>0.0001 OR abs(b.high-d.high)>0.0001 OR
              abs(b.low-d.low)>0.0001 OR abs(b.close-d.close)>0.0001 OR
              abs(b.volume-d.volume)>0.5 OR abs(b.amount-d.amount)>0.01
        ORDER BY b.date, b.thscode LIMIT 100
    """).fetchdf()
    con.close()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(OUTPUT / "coverage_by_status_year.csv", index=False)
    delisted.to_csv(OUTPUT / "delisted_coverage.csv", index=False)
    st_candidates.to_csv(OUTPUT / "st_candidates.csv", index=False)
    mismatch_examples.to_csv(OUTPUT / "price_mismatch_examples.csv", index=False)
    summary = {
        "window": {"start": START, "end": END},
        "unadjusted_price_comparison": price,
        "backward_adjusted_close_comparison": hfq,
        "candidate_baostock_coverage": candidate_coverage,
        "delisted_mainboard_symbols": len(delisted),
        "delisted_with_bars": int(delisted["bars"].fillna(0).gt(0).sum()),
        "delisted_with_bars_in_window": int(delisted["bars"].fillna(0).gt(0).sum()),
        "notes": [
            "Baostock suspended sessions may be absent from the DuckDB trading-only panel.",
            "This audit does not establish independent out-of-sample evidence.",
        ],
    }
    write_json(OUTPUT / "data_audit.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "audit", "all"])
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.command in ("export", "all"):
        if MAINBOARD_PARQUET.exists() and not args.rebuild:
            raise FileExistsError(f"Use --rebuild to replace {MAINBOARD_PARQUET}")
        print(json.dumps(export_mainboard(), ensure_ascii=False, default=str), flush=True)
    if args.command in ("audit", "all"):
        if not MAINBOARD_PARQUET.exists() or not SYMBOLS_PARQUET.exists():
            raise FileNotFoundError("Run export first")
        print(json.dumps(audit_data(), ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
