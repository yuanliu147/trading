#!/usr/bin/env python3
"""从 baostock SQLite 抽取主板股票日线（后复权+不复权），构建研究面板。

输出：
  01_数据/bars.parquet     全部原始日线（含停牌日），用于成交/涨跌停/退市处理
  01_数据/panel.parquet    仅在正常交易日计算的特征面板（信号用）
  05_证据/build_panel_audit.json

口径：
  - 主板前缀 600/601/603/605/000/001/002/003，type=1 股票
  - 含退市股（stock_basic.out_date 非空）
  - 日期 2004-01-01 起，保证 2005-2006 年信号也有 250 个交易日回看
  - 特征只在 trade_status=1 且 volume>0 的行上按“该股实际交易日”滚动，
    与同花顺 core.parquet 的口径一致（停牌日不占窗口）
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DB = Path("/Users/admin/WorkSpace/baostock_data/data/baostock.db")
DATA = ROOT / "01_数据"
EVID = ROOT / "05_证据"
START = "2004-01-01"
PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
KEEP = [
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "turn", "trade_status", "pct_chg", "is_st", "pe_ttm", "pb_mrq",
]

SCHEMA = pa.schema([
    ("code", pa.string()), ("date", pa.string()),
    ("open", pa.float64()), ("high", pa.float64()), ("low", pa.float64()),
    ("close", pa.float64()), ("preclose", pa.float64()),
    ("volume", pa.float64()), ("amount", pa.float64()), ("turn", pa.float64()),
    ("trade_status", pa.int64()), ("pct_chg", pa.float64()),
    ("is_st", pa.int64()), ("pe_ttm", pa.float64()), ("pb_mrq", pa.float64()),
    ("hfq_open", pa.float64()), ("hfq_high", pa.float64()),
    ("hfq_low", pa.float64()), ("hfq_close", pa.float64()),
])


def stock_list(con: sqlite3.Connection) -> pd.DataFrame:
    q = """
    SELECT code, code_name, ipo_date, out_date
    FROM stock_basic
    WHERE type=1
    """
    df = pd.read_sql_query(q, con)
    df["code_name"] = df["code_name"].fillna("")
    df["out_date"] = df["out_date"].fillna("")
    df = df[df.code.str[3:6].isin(PREFIXES)].copy()
    df["is_delisted"] = (df.out_date != "").astype(int)
    return df.sort_values("code").reset_index(drop=True)


def fetch_one(con: sqlite3.Connection, code: str) -> pd.DataFrame | None:
    q = f"""
    SELECT date, adjustflag, {', '.join(KEEP)}
    FROM stock_kline_d
    WHERE code=? AND adjustflag IN (1,3) AND date>=?
    ORDER BY date, adjustflag
    """
    df = pd.read_sql_query(q, con, params=(code, START))
    if df.empty:
        return None
    wide = None
    for flag, prefix in [(3, ""), (1, "hfq_")]:
        part = df[df.adjustflag == flag].drop(columns="adjustflag")
        if part.empty:
            continue
        part = part.rename(columns={c: prefix + c for c in KEEP})
        part = part.set_index("date")
        wide = part if wide is None else wide.join(part, how="outer")
    if wide is None:
        return None
    return wide.reset_index()


def extract(con: sqlite3.Connection, stocks: pd.DataFrame, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    writer = pq.ParquetWriter(out, SCHEMA, compression="zstd")
    t0 = time.time()
    n_rows = 0
    n_codes = 0
    missing = []
    for i, row in enumerate(stocks.itertuples(index=False), 1):
        df = fetch_one(con, row.code)
        if df is None:
            missing.append(row.code)
            continue
        df.insert(0, "code", row.code)
        # 补齐 schema 缺列
        for name in SCHEMA.names:
            if name not in df:
                df[name] = np.nan
        df = df[SCHEMA.names]
        for name, typ in zip(SCHEMA.names, SCHEMA.types):
            if pa.types.is_integer(typ):
                df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0).astype("int64")
            elif pa.types.is_floating(typ):
                df[name] = pd.to_numeric(df[name], errors="coerce").astype("float64")
            else:
                df[name] = df[name].astype(str)
        writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
        n_rows += len(df)
        n_codes += 1
        if i % 250 == 0:
            print(f"  {i}/{len(stocks)} codes, {n_rows:,} rows, {time.time()-t0:.1f}s", flush=True)
    writer.close()
    return dict(codes=n_codes, rows=n_rows, missing=missing, seconds=round(time.time() - t0, 1))



def repair_hfq(bars: Path, evid: Path) -> dict:
    """修复 baostock 后复权因因子重定基造成的“仅因子跳变”。

    判定：前收盘未变（|preclose/prev_close-1|<0.2%）但复权因子跳变 >2%。
    这类跳变不是分红送转，而是数据源重定基。用 prev_close/preclose
    作为应有跳变，把累计乘数补回，使当日后复权收益等于实际前收盘收益。
    """
    con = duckdb.connect()
    con.execute("SET threads TO 8; SET memory_limit='12GB';")
    b = str(bars).replace("'", "''")
    sql = f"""
    WITH x AS (
        SELECT *, hfq_close / nullif(close, 0) AS factor,
            lag(close) OVER w AS prev_close,
            lag(hfq_close / nullif(close, 0)) OVER w AS prev_factor
        FROM read_parquet('{b}')
        WINDOW w AS (PARTITION BY code ORDER BY date)
    ), y AS (
        SELECT *,
            CASE WHEN prev_factor > 0 AND preclose > 0 AND prev_close > 0
                      AND abs(preclose / prev_close - 1) < 0.002
                      AND abs(factor / prev_factor - 1) > 0.02
                 THEN (prev_close / preclose) / (factor / prev_factor)
                 ELSE 1.0 END AS mult
        FROM x
    ), z AS (
        SELECT *, product(mult) OVER (PARTITION BY code ORDER BY date) AS cum
        FROM y
    )
    SELECT code, date, open, high, low, close, preclose, volume, amount, turn,
           trade_status, pct_chg, is_st, pe_ttm, pb_mrq,
           open * factor * cum AS hfq_open,
           high * factor * cum AS hfq_high,
           low * factor * cum AS hfq_low,
           close * factor * cum AS hfq_close,
           mult
    FROM z
    """
    tmp = bars.parent / "bars_repaired.parquet"
    con.execute(f"COPY ({sql}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)")
    events = con.execute(
        f"""SELECT code, date, close, preclose, prev_close, factor, prev_factor,
                   factor/prev_factor-1 AS observed_jump, prev_close/preclose-1 AS expected_jump,
                   (prev_close/preclose)/(factor/prev_factor) AS mult
            FROM (
                SELECT *, hfq_close / nullif(close,0) AS factor,
                    lag(close) OVER w AS prev_close,
                    lag(hfq_close / nullif(close,0)) OVER w AS prev_factor
                FROM read_parquet('{b}')
                WINDOW w AS (PARTITION BY code ORDER BY date)
            ) WHERE abs(preclose/nullif(prev_close,0)-1)<0.002
              AND abs(factor/nullif(prev_factor,0)-1)>0.02
            ORDER BY date, code"""
    ).fetchdf()
    events.to_csv(evid / "复权修复事件.csv", index=False, encoding="utf-8-sig")
    con.close()
    tmp.replace(bars)
    return dict(repaired=int(len(events)), codes=int(events.code.nunique()) if len(events) else 0)


def build_features(bars: Path, panel: Path, audit_path: Path) -> dict:
    con = duckdb.connect()
    con.execute("SET threads TO 8; SET memory_limit='12GB';")
    b = str(bars).replace("'", "''")
    p = str(panel).replace("'", "''")
    # 只在正常交易日计算特征；停牌日保留在 bars 中供成交判断
    sql = f"""
    WITH base AS (
        SELECT *, row_number() OVER (PARTITION BY code ORDER BY date) AS listing_session
        FROM read_parquet('{b}')
        WHERE trade_status=1 AND volume>0
    ), base2 AS (
        SELECT *,
            hfq_close / nullif(lag(hfq_close, 1) OVER w, 0) - 1 AS ret1,
            hfq_close / nullif(lag(hfq_close, 5) OVER w, 0) - 1 AS return_5d,
            hfq_close / nullif(lag(hfq_close, 10) OVER w, 0) - 1 AS return_10d,
            hfq_close / nullif(lag(hfq_close, 20) OVER w, 0) - 1 AS return_20d,
            hfq_close / nullif(lag(hfq_close, 60) OVER w, 0) - 1 AS return_60d,
            hfq_close / nullif(lag(hfq_close, 120) OVER w, 0) - 1 AS return_120d,
            hfq_close / nullif(lag(hfq_close, 250) OVER w, 0) - 1 AS return_250d
        FROM base
        WINDOW w AS (PARTITION BY code ORDER BY date)
    ), calc AS (
        SELECT *,
            hfq_close / nullif(max(hfq_high) OVER w250, 0) - 1 AS drawdown250,
            avg(amount) OVER w20 AS amount20,
            avg(turn) OVER w20 AS turn20,
            avg(turn) OVER w5 AS turn5,
            (close - low) / nullif(high - low, 0) AS close_location,
            hfq_close / nullif(avg(hfq_close) OVER w20, 0) - 1 AS close_ma20,
            hfq_close / nullif(avg(hfq_close) OVER w60, 0) - 1 AS close_ma60,
            hfq_close / nullif(avg(hfq_close) OVER w250, 0) - 1 AS close_ma250,
            turn / nullif(avg(turn) OVER w20, 0) AS turn_ratio1_20,
            amount / nullif(avg(amount) OVER w20, 0) AS amount_ratio1_20,
            stddev_samp(ret1) OVER w20 * sqrt(252) AS volatility20,
            max(hfq_high) OVER w20 AS high20,
            min(hfq_low) OVER w20 AS low20
        FROM base2
        WINDOW
            w5 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
            w20 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
            w60 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),
            w250 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW)
    )
    SELECT * FROM calc
    """
    con.execute(f"COPY ({sql}) TO '{p}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)")
    stats = con.execute(
        f"""SELECT count(*) AS rows_n, count(DISTINCT code) AS codes, min(date) AS first_date, max(date) AS last_date
            FROM read_parquet('{p}')"""
    ).fetchdf().iloc[0].to_dict()
    con.close()
    return {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in stats.items()}


def add_rsi(panel: Path, evid: Path) -> dict:
    """Wilder RSI14，与同花顺 features.py 的 ewm(alpha=1/14, adjust=False) 一致。"""
    cols = ["code", "date", "hfq_close"]
    df = pq.read_table(panel, columns=cols).to_pandas()
    df = df.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    delta = df.groupby("code", sort=False)["hfq_close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(df["code"], sort=False).transform(
        lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean())
    avg_loss = loss.groupby(df["code"], sort=False).transform(
        lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean())
    rsi = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    out = df[["code", "date"]].copy()
    out["rsi14"] = rsi.astype("float64")
    rsi_path = panel.parent / "rsi14.parquet"
    out.to_parquet(rsi_path, index=False)
    con = duckdb.connect()
    con.execute("SET threads TO 8; SET memory_limit='12GB';")
    p = str(panel).replace("'", "''")
    r = str(rsi_path).replace("'", "''")
    tmp = panel.parent / "panel_rsi.parquet"
    con.execute(
        f"""COPY (SELECT p.*, r.rsi14 FROM read_parquet('{p}') p
                LEFT JOIN read_parquet('{r}') r USING(code,date))
            TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"""
    )
    tmp.replace(panel)
    rsi_path.unlink()
    check = con.execute(
        f"SELECT count(*) AS rows_n, count(*) FILTER (rsi14 IS NULL) AS nulls FROM read_parquet('{p}')"
    ).fetchdf().iloc[0].to_dict()
    con.close()
    return {k: int(v) for k, v in check.items()}


def add_market(panel: Path) -> dict:
    """市场等权指数：全主板（含退市）与仅当前上市两套，20 日收益作为市场闸门。"""
    con = duckdb.connect()
    con.execute("SET threads TO 8; SET memory_limit='12GB';")
    p = str(panel).replace("'", "''")
    # 当前上市标记：从 stock_basic 读入 pandas 后注册
    stocks = pd.read_sql_query(
        "SELECT code, out_date FROM stock_basic WHERE type=1",
        sqlite3.connect(f"file:{DB}?mode=ro", uri=True),
    )
    stocks["listed"] = stocks.out_date.isna() | (stocks.out_date == "")
    con.register("listed_tbl", stocks[stocks.listed][["code"]])
    sql = f"""
    WITH rets AS (
        SELECT date, code, ret1 AS r,
               CASE WHEN code IN (SELECT code FROM listed_tbl) THEN ret1 END AS r_listed
        FROM read_parquet('{p}')
        WHERE ret1 IS NOT NULL AND isfinite(ret1)
    ), m AS (
        SELECT date,
               avg(r) AS market_return,
               avg(r_listed) AS market_return_listed,
               count(*) AS n_all,
               count(r_listed) AS n_listed
        FROM rets GROUP BY date
    ), idx AS (
        SELECT *,
               exp(sum(ln(1+coalesce(market_return,0))) OVER (ORDER BY date)) AS index_all,
               exp(sum(ln(1+coalesce(market_return_listed,0))) OVER (ORDER BY date)) AS index_listed
        FROM m
    )
    SELECT date, market_return, market_return_listed,
           index_all / nullif(lag(index_all,20) OVER (ORDER BY date),0) - 1 AS market_return20_all,
           index_listed / nullif(lag(index_listed,20) OVER (ORDER BY date),0) - 1 AS market_return20_listed,
           n_all, n_listed
    FROM idx ORDER BY date
    """
    out = panel.parent / "market_daily.parquet"
    con.execute(f"COPY ({sql}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    # 合并回面板
    tmp = panel.parent / "panel_tmp.parquet"
    con.execute(
        f"""COPY (SELECT p.*, m.market_return, m.market_return_listed,
                        m.market_return20_all, m.market_return20_listed, m.n_all, m.n_listed
                 FROM read_parquet('{p}') p LEFT JOIN read_parquet('{out}') m USING(date))
            TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"""
    )
    tmp.replace(panel)
    check = con.execute(
        f"""SELECT count(*) AS rows_n, count(*) FILTER (market_return20_all IS NULL) AS null_all,
                   count(*) FILTER (market_return20_listed IS NULL) null_listed
            FROM read_parquet('{p}')"""
    ).fetchdf().iloc[0].to_dict()
    con.close()
    return {k: int(v) for k, v in check.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="仅抽取前 N 只股票，用于冒烟测试")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    EVID.mkdir(parents=True, exist_ok=True)
    bars = DATA / "bars.parquet"
    panel = DATA / "panel.parquet"
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    stocks = stock_list(con)
    if args.limit:
        stocks = stocks.head(args.limit).copy()
    print(f"mainboard stocks: {len(stocks)} (delisted {int(stocks.is_delisted.sum())})")
    stocks.to_csv(EVID / "主板股票清单.csv", index=False, encoding="utf-8-sig")
    ex = {"skipped": True} if args.skip_extract else extract(con, stocks, bars)
    con.close()
    print("extract:", ex)
    rp = repair_hfq(bars, EVID)
    print("repair:", rp)
    st = build_features(bars, panel, EVID)
    print("features:", st)
    rs = add_rsi(panel, EVID)
    print("rsi:", rs)
    mk = add_market(panel)
    print("market:", mk)
    audit = dict(stocks=len(stocks), delisted=int(stocks.is_delisted.sum()),
                 extract=ex, repair=rp, panel=st, rsi=rs, market=mk,
                 start=START, prefixes=list(PREFIXES))
    (EVID / "build_panel_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n")
    print("done")


if __name__ == "__main__":
    main()
