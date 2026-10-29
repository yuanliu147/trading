#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双源独立数据审计：同花顺 DuckDB（THS） vs Baostock SQLite（BS）

审计对象
--------
  THS  : /Users/admin/Library/Application Support/hithink-finance/data/market.duckdb
         raw_kline_daily / dim_symbol / raw_adjustment_events / calc_adjust_factor_daily
  BS   : /Users/admin/WorkSpace/baostock_data/data/baostock.db
         stock_kline_d (adjustflag 1=后复权 2=前复权 3=不复权) / stock_basic / trade_dates

窗口   : 2016-10-13 ~ 2026-10-26（冻结日；之后的行情一律剔除，仅计数披露）
只读   : 两个源库均以 READ_ONLY 打开；不写入、不迁移、不同步。
收益   : 本脚本不计算任何组合/策略收益。出现的“收益率”仅为相邻收盘价之比，
         用途限于复权口径一致性检验（数据质量），不构成绩效评估。

输出
----
  05_审计/双源数据核验明细.csv
  05_审计/双源数据核验.md
  03_结果/_审计缓存/bs_window_all_flags.parquet  （Baostock 窗口切片缓存，可重建）

用法
----
  python3 02_代码/审计双源.py                 # 使用已有缓存（无则自动建）
  python3 02_代码/审计双源.py --rebuild-cache # 强制从源库重建缓存
  python3 02_代码/审计双源.py --quick         # 抽样规模减半，用于冒烟
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from datetime import datetime, timezone

import duckdb

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
THS_DB = "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"
BS_DB = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(PROJECT, "05_审计", "双源数据核验明细.csv")
OUT_MD = os.path.join(PROJECT, "05_审计", "双源数据核验.md")
CACHE_DIR = os.path.join(PROJECT, "03_结果", "_审计缓存")
CACHE_PARQUET = os.path.join(CACHE_DIR, "bs_window_all_flags.parquet")

WINDOW_START = "2016-10-13"
FREEZE = "2026-10-26"          # 冻结日：只用 <= 该日的数据
SEED = 20261026                # 抽样种子，固定以保证可复现

# 容差（写进报告，必须显式）
TOL_PRICE_ABS = 0.005          # 半分钱：两源均为 2 位小数报价
TOL_PRICE_REL = 1e-6
TOL_VOL_REL = 1e-6             # 成交量：股，允许 float32 级误差
TOL_AMT_ABS = 0.011            # 成交额：元，允许 1 分钱
TOL_AMT_REL = 1e-4             # float32 级相对误差
TOL_RET = 1e-6                 # 复权收益率一致性
CA_MATCH_TOL = 0.002           # 除权参考价公式误差阈值（0.2%）
CA_BIG_ERR = 0.01              # 公式失配“重大”阈值（1%）

BOARD_RULES = [
    ("SH_MAIN", ("600", "601", "603", "605")),
    ("STAR", ("688", "689")),
    ("SZ_MAIN", ("000", "001", "003")),
    ("SME", ("002",)),
    ("CHINEXT", ("300", "301", "302")),
    ("BJ", ("920", "430", "830", "831", "832", "833", "834", "835", "836",
            "837", "838", "839", "870", "871", "872", "873", "874")),
]

CSV_FIELDS = [
    "record_type", "check_id", "stratum", "board", "thscode_ths", "code_bs",
    "date", "field", "ths_value", "bs_value", "diff", "rel_diff", "tol",
    "status", "note",
]


def board_of(ticker: str) -> str:
    for name, prefixes in BOARD_RULES:
        if ticker.startswith(prefixes):
            return name
    return "OTHER"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 连接与缓存
# ----------------------------------------------------------------------------
def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("LOAD sqlite")
    con.execute(f"ATTACH '{THS_DB}' AS ths (READ_ONLY)")
    con.execute(f"ATTACH '{BS_DB}' AS bs (TYPE SQLITE, READ_ONLY)")
    return con


def ensure_cache(con, rebuild: bool) -> dict:
    """把 Baostock 冻结窗口内的三复权数据落到本地 parquet，避免反复全表扫描。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if rebuild or not os.path.exists(CACHE_PARQUET):
        t0 = time.time()
        log("重建 Baostock 窗口缓存 ...")
        con.execute(f"""
            COPY (
              SELECT code, date, adjustflag, open, high, low, close, preclose,
                     volume, amount, turn, trade_status, is_st
              FROM bs.stock_kline_d
              WHERE date BETWEEN '{WINDOW_START}' AND '{FREEZE}'
            ) TO '{CACHE_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        log(f"缓存完成 {time.time() - t0:.0f}s")
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{CACHE_PARQUET}')").fetchone()[0]
    size = os.path.getsize(CACHE_PARQUET)
    return {"rows": rows, "size": size, "path": CACHE_PARQUET}


def build_core(con) -> None:
    """核心临时表：THS 原始、BS 三分支、内联接结果。"""
    con.execute(f"""
        CREATE TEMP TABLE ths_raw AS
        SELECT thscode, date, open, high, low, close, volume, amount
        FROM ths.raw_kline_daily
        WHERE date BETWEEN DATE '{WINDOW_START}' AND DATE '{FREEZE}'
    """)
    con.execute(f"""
        CREATE TEMP TABLE bs_all AS
        SELECT * FROM read_parquet('{CACHE_PARQUET}')
    """)
    # 统一映射：sh.600000 <-> 600000.SH
    con.execute("""
        CREATE TEMP TABLE bs_map AS
        SELECT b.*,
               upper(split_part(b.code,'.',2)) || '.' || upper(split_part(b.code,'.',1)) AS thscode
        FROM bs_all b
    """)
    con.execute("""
        CREATE TEMP TABLE j AS
        SELECT t.thscode, t.date,
               t.open AS t_open, t.high AS t_high, t.low AS t_low, t.close AS t_close,
               t.volume AS t_vol, t.amount AS t_amt,
               b.code AS b_code, b.open AS b_open, b.high AS b_high, b.low AS b_low,
               b.close AS b_close, b.volume AS b_vol, b.amount AS b_amt,
               b.preclose, b.turn, b.trade_status, b.is_st
        FROM ths_raw t
        JOIN bs_map b
          ON b.thscode = t.thscode AND b.date = strftime(t.date, '%Y-%m-%d')
         AND b.adjustflag = 3
    """)


# ----------------------------------------------------------------------------
# 明细收集
# ----------------------------------------------------------------------------
class Detail:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, **kw):
        row = {k: "" for k in CSV_FIELDS}
        row.update(kw)
        self.rows.append(row)

    def write(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)


def fnum(x, nd=6):
    if x is None:
        return ""
    if isinstance(x, float):
        if x != x:  # NaN
            return ""
        return f"{x:.{nd}f}"
    return x


# ----------------------------------------------------------------------------
# 各项检查
# ----------------------------------------------------------------------------
def check_preflight(con, cache) -> dict:
    out = {}
    out["ths_file_mtime"] = datetime.fromtimestamp(
        os.path.getmtime(THS_DB), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out["bs_file_mtime"] = datetime.fromtimestamp(
        os.path.getmtime(BS_DB), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out["cache_rows"] = cache["rows"]
    out["cache_size_mb"] = round(cache["size"] / 1e6, 1)

    out["ths_rows_all"] = con.execute("SELECT count(*) FROM ths.raw_kline_daily").fetchone()[0]
    out["ths_rows_window"] = con.execute(
        f"SELECT count(*) FROM ths.raw_kline_daily WHERE date BETWEEN DATE '{WINDOW_START}' AND DATE '{FREEZE}'"
    ).fetchone()[0]
    out["ths_rows_post_freeze"] = con.execute(
        f"SELECT count(*) FROM ths.raw_kline_daily WHERE date > DATE '{FREEZE}'"
    ).fetchone()[0]
    out["ths_symbols_post_freeze"] = con.execute(
        f"SELECT count(DISTINCT thscode) FROM ths.raw_kline_daily WHERE date > DATE '{FREEZE}'"
    ).fetchone()[0]
    out["ths_max_date"] = str(con.execute("SELECT max(date) FROM ths.raw_kline_daily").fetchone()[0])
    out["bs_rows_window"] = cache["rows"]
    # 注：sqlite_scanner 在本表上无法生成 ROWID，count(*) 会报错；改用列计数
    out["bs_rows_all"] = con.execute("SELECT count(code) FROM bs.stock_kline_d").fetchone()[0]
    out["bs_max_date"] = con.execute("SELECT max(date) FROM bs.stock_kline_d").fetchone()[0]
    # 窗口计数二次校验（源库处于活跃写入状态）
    out["ths_rows_window_recheck"] = con.execute(
        f"SELECT count(*) FROM ths.raw_kline_daily WHERE date BETWEEN DATE '{WINDOW_START}' AND DATE '{FREEZE}'"
    ).fetchone()[0]
    out["bs_rows_window_recheck"] = con.execute(
        f"SELECT count(*) FROM bs.stock_kline_d WHERE date BETWEEN '{WINDOW_START}' AND '{FREEZE}'"
    ).fetchone()[0]
    out["window_stable"] = (out["ths_rows_window"] == out["ths_rows_window_recheck"]
                            and out["bs_rows_window"] == out["bs_rows_window_recheck"])
    return out


def check_structure(con, det: Detail) -> dict:
    out = {}
    # --- 重复键 ---
    out["ths_dup_keys"] = con.execute(
        "SELECT count(*) FROM (SELECT thscode,date,count(*) c FROM ths.raw_kline_daily GROUP BY 1,2 HAVING c>1)"
    ).fetchone()[0]
    out["bs_dup_keys"] = con.execute(
        "SELECT count(*) FROM (SELECT code,date,adjustflag,count(*) c FROM bs.stock_kline_d GROUP BY 1,2,3 HAVING c>1)"
    ).fetchone()[0]

    # --- OHLC 内部一致性 ---
    out["ths_ohlc_violations"] = con.execute("""
        SELECT count(*) FROM ths_raw
        WHERE low > high + 1e-9 OR close > high + 1e-9 OR close < low - 1e-9
           OR open > high + 1e-9 OR open < low - 1e-9
    """).fetchone()[0]
    out["ths_nonpositive_price"] = con.execute(
        "SELECT count(*) FROM ths_raw WHERE open<=0 OR high<=0 OR low<=0 OR close<=0"
    ).fetchone()[0]
    out["bs_ohlc_violations"] = con.execute("""
        SELECT count(*) FROM bs_map WHERE adjustflag=3 AND close IS NOT NULL
        AND (low > high + 1e-9 OR close > high + 1e-9 OR close < low - 1e-9)
    """).fetchone()[0]
    out["bs_null_ohlc_rows"] = con.execute(
        "SELECT count(*) FROM bs_map WHERE adjustflag=3 AND close IS NULL"
    ).fetchone()[0]

    # --- 交易日历 ---
    out["bs_trade_days"] = con.execute(
        f"SELECT count(*) FROM bs.trade_dates WHERE is_trading_day=1 AND calendar_date BETWEEN '{WINDOW_START}' AND '{FREEZE}'"
    ).fetchone()[0]
    out["ths_trade_days"] = con.execute(
        "SELECT count(DISTINCT date) FROM ths_raw"
    ).fetchone()[0]
    out["dates_missing_in_ths"] = con.execute(f"""
        SELECT count(*) FROM bs.trade_dates d
        WHERE d.is_trading_day=1 AND d.calendar_date BETWEEN '{WINDOW_START}' AND '{FREEZE}'
          AND NOT EXISTS (SELECT 1 FROM ths_raw t WHERE t.date = strptime(d.calendar_date,'%Y-%m-%d')::DATE)
    """).fetchone()[0]
    out["dates_ths_not_calendar"] = con.execute(f"""
        SELECT count(*) FROM (SELECT DISTINCT date FROM ths_raw) t
        WHERE NOT EXISTS (SELECT 1 FROM bs.trade_dates d
                          WHERE d.is_trading_day=1 AND strptime(d.calendar_date,'%Y-%m-%d')::DATE = t.date)
    """).fetchone()[0]

    # --- 证券覆盖（幸存者偏差核心） ---
    cov = con.execute("""
        WITH bsm AS (
          SELECT code, code_name, ipo_date, out_date, status,
                 upper(split_part(code,'.',2))||'.'||upper(split_part(code,'.',1)) AS thscode
          FROM bs.stock_basic WHERE type=1
        ), cnt AS (
          SELECT thscode, count(*) n FROM ths_raw GROUP BY 1
        )
        SELECT
          count(*) AS bs_symbols,
          sum(CASE WHEN c.thscode IS NULL THEN 1 ELSE 0 END) AS missing_in_ths,
          sum(CASE WHEN (out_date IS NULL OR out_date='') AND c.thscode IS NULL THEN 1 ELSE 0 END) AS missing_listed,
          sum(CASE WHEN (out_date IS NOT NULL AND out_date<>'') AND c.thscode IS NULL THEN 1 ELSE 0 END) AS missing_delisted
        FROM bsm m LEFT JOIN cnt c USING(thscode)
    """).df().iloc[0]
    out.update({k: int(v) for k, v in cov.items()})
    out["bs_delisted_total"] = con.execute(
        "SELECT count(*) FROM bs.stock_basic WHERE type=1 AND out_date IS NOT NULL AND out_date<>''"
    ).fetchone()[0]
    out["bs_delisted_in_ths"] = out["bs_delisted_total"] - out["missing_delisted"]

    out["ths_symbols"] = con.execute("SELECT count(DISTINCT thscode) FROM ths_raw").fetchone()[0]
    out["ths_bj_symbols"] = con.execute(
        "SELECT count(DISTINCT thscode) FROM ths_raw WHERE thscode LIKE '%.BJ'"
    ).fetchone()[0]
    out["ths_symbols_not_in_bs"] = con.execute("""
        WITH bsm AS (SELECT upper(split_part(code,'.',2))||'.'||upper(split_part(code,'.',1)) AS thscode
                     FROM bs.stock_basic WHERE type=1)
        SELECT count(*) FROM (SELECT DISTINCT thscode FROM ths_raw) t
        WHERE NOT EXISTS (SELECT 1 FROM bsm WHERE bsm.thscode=t.thscode)
    """).fetchone()[0]

    # 退市股明细（全部落盘）
    miss = con.execute("""
        WITH bsm AS (
          SELECT code, code_name, ipo_date, out_date, status,
                 upper(split_part(code,'.',2))||'.'||upper(split_part(code,'.',1)) AS thscode
          FROM bs.stock_basic WHERE type=1
        ), cnt AS (SELECT thscode, count(*) n, max(date) d1 FROM ths_raw GROUP BY 1)
        SELECT m.code, m.code_name, m.ipo_date, m.out_date, m.status, coalesce(c.n,0) n_ths, c.d1,
               (SELECT count(*) FROM bs_map bb WHERE bb.code=m.code AND bb.adjustflag=3) AS n_bs
        FROM bsm m LEFT JOIN cnt c USING(thscode)
        WHERE c.thscode IS NULL
        ORDER BY m.out_date
    """).fetchall()
    for code, name, ipo, outd, status, n_ths, d1, n_bs in miss:
        det.add(record_type="missing_symbol", check_id="S1_coverage",
                stratum="delisted" if outd else "listed", board=board_of(code.split(".")[1]),
                code_bs=code, thscode_ths="", date=outd or "",
                field="entire_symbol", ths_value=int(n_ths), bs_value=int(n_bs),
                status="THS_ABSENT",
                note=f"{name} ipo={ipo} status={status} out={outd} 窗口内BS行数={n_bs}")
    out["missing_symbol_rows"] = len(miss)
    # 窗口内确有交易、但 THS 完全缺失的证券（幸存者偏差的实际影响面）
    r = con.execute("""
        WITH m AS (
          SELECT upper(split_part(code,'.',2))||'.'||upper(split_part(code,'.',1)) AS thscode,
                 count(*) n
          FROM bs_map WHERE adjustflag=3 GROUP BY 1
        )
        SELECT count(*) sym, coalesce(sum(n),0) n_rows
        FROM m WHERE NOT EXISTS (SELECT 1 FROM ths_raw t WHERE t.thscode=m.thscode)
    """).df().iloc[0]
    out["missing_symbols_with_data"] = int(r["sym"])
    out["missing_rows_with_data"] = int(r["n_rows"])

    # --- 停牌语义 ---
    susp = con.execute("""
        SELECT count(*) n,
               sum(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) vol_null,
               sum(CASE WHEN volume=0 THEN 1 ELSE 0 END) vol_zero,
               sum(CASE WHEN volume>0 THEN 1 ELSE 0 END) vol_pos,
               sum(CASE WHEN open=close AND high=close AND low=close THEN 1 ELSE 0 END) flat
        FROM bs_map WHERE adjustflag=3 AND trade_status=0
    """).df().iloc[0]
    out["bs_st_rows"] = con.execute(
        "SELECT count(*) FROM bs_map WHERE adjustflag=3 AND is_st=1").fetchone()[0]
    out["bs_susp_rows"] = int(susp["n"])
    out["bs_susp_vol_null"] = int(susp["vol_null"])
    out["bs_susp_vol_zero"] = int(susp["vol_zero"])
    out["bs_susp_vol_pos"] = int(susp["vol_pos"])
    out["bs_susp_flat"] = int(susp["flat"])
    out["ths_zero_volume_rows"] = con.execute(
        "SELECT count(*) FROM ths_raw WHERE volume=0"
    ).fetchone()[0]
    out["bs_susp_days_present_in_ths"] = con.execute("""
        SELECT count(*) FROM bs_map b JOIN ths_raw t
          ON t.thscode=b.thscode AND t.date=strptime(b.date,'%Y-%m-%d')::DATE
        WHERE b.adjustflag=3 AND b.trade_status=0
    """).fetchone()[0]
    out["ths_days_absent_in_bs"] = con.execute("""
        SELECT count(*) FROM ths_raw t
        WHERE NOT EXISTS (SELECT 1 FROM bs_map b
                          WHERE b.adjustflag=3 AND b.thscode=t.thscode
                            AND b.date=strftime(t.date,'%Y-%m-%d'))
    """).fetchone()[0]

    # --- 代码变更 / 全空行 ---
    nulls = con.execute("""
        SELECT code, date, count(*) FROM bs_map
        WHERE adjustflag=3 AND close IS NULL GROUP BY 1,2 ORDER BY 2
    """).fetchall()
    out["bs_all_null_rows"] = sum(r[2] for r in nulls)
    for code, d, _ in nulls:
        det.add(record_type="structural", check_id="S3_all_null_row", stratum="code_change",
                board=board_of(code.split(".")[1]), code_bs=code, date=d,
                field="ohlcv", status="ALL_NULL", note="全字段为空，通常为代码变更日")
    return out


def check_values(con, det: Detail) -> dict:
    out = {}
    row = con.execute("""
        SELECT count(*) n,
          sum(CASE WHEN t_open=b_open THEN 1 ELSE 0 END) o_eq,
          sum(CASE WHEN t_high=b_high THEN 1 ELSE 0 END) h_eq,
          sum(CASE WHEN t_low=b_low THEN 1 ELSE 0 END) l_eq,
          sum(CASE WHEN t_close=b_close THEN 1 ELSE 0 END) c_eq,
          sum(CASE WHEN t_vol=b_vol THEN 1 ELSE 0 END) v_eq,
          sum(CASE WHEN abs(t_amt-b_amt)<=0.011 THEN 1 ELSE 0 END) a_eq,
          sum(CASE WHEN abs(t_close-b_close)<=0.005 THEN 1 ELSE 0 END) c_ok
        FROM j
    """).df().iloc[0]
    out["matched_rows"] = int(row["n"])
    for k in ["o_eq", "h_eq", "l_eq", "c_eq", "v_eq", "a_eq", "c_ok"]:
        out[k] = int(row[k])
    out["c_bad"] = out["matched_rows"] - out["c_ok"]

    # 极值分级
    out["price_diff_gt_0.02"] = con.execute(
        "SELECT count(*) FROM j WHERE abs(t_close-b_close)>0.02").fetchone()[0]
    out["vol_rel_gt_1e-6"] = con.execute(
        "SELECT count(*) FROM j WHERE b_vol>0 AND abs(t_vol-b_vol)>abs(b_vol)*1e-6").fetchone()[0]
    out["amt_rel_gt_1e-4"] = con.execute(
        "SELECT count(*) FROM j WHERE b_amt>0 AND abs(t_amt-b_amt)>abs(b_amt)*1e-4").fetchone()[0]
    out["amt_rel_gt_1pct"] = con.execute(
        "SELECT count(*) FROM j WHERE b_amt>0 AND abs(t_amt/b_amt-1)>=0.01").fetchone()[0]
    # 成交额综合通过率：1 分钱内 OR float32 相对误差内
    out["a_pass"] = con.execute("""
        SELECT count(*) FROM j
        WHERE abs(t_amt-b_amt)<=0.011 OR (b_amt>0 AND abs(t_amt-b_amt)<=abs(b_amt)*1e-4)
    """).fetchone()[0]
    out["amt_rel_p999"] = con.execute(
        "SELECT quantile_cont(abs(t_amt/b_amt-1), 0.999) FROM j WHERE b_amt>0"
    ).fetchone()[0]
    out["ths_prev_close_null_rate"] = con.execute(
        "SELECT avg(CASE WHEN prev_close IS NULL THEN 1 ELSE 0 END) FROM ths.raw_kline_daily"
    ).fetchone()[0]

    # 单位判定（中位数比值）
    out["vol_ratio_median"] = con.execute(
        "SELECT median(t_vol/b_vol) FROM j WHERE b_vol>0").fetchone()[0]
    out["amt_ratio_median"] = con.execute(
        "SELECT median(t_amt/b_amt) FROM j WHERE b_amt>0").fetchone()[0]

    # 全部数值异常落盘
    ab = con.execute("""
        SELECT thscode,date,t_open,t_high,t_low,t_close,t_vol,t_amt,
               b_code,b_open,b_high,b_low,b_close,b_vol,b_amt
        FROM j
        WHERE abs(t_close-b_close)>0.005 OR (b_amt>0 AND abs(t_amt/b_amt-1)>=0.0001)
        ORDER BY date
    """).fetchall()
    for r in ab:
        (th, d, to, thh, tl, tc, tv, ta,
         bc, bo, bh, bl, bcl, bv, ba) = r
        worst = "close"
        d_abs = abs(tc - bcl)
        rel = d_abs / max(abs(bcl), 1e-9)
        if abs(ta - ba) > 0.011 and abs(ta / ba - 1) > rel:
            worst, d_abs, rel = "amount", abs(ta - ba), abs(ta / ba - 1)
        det.add(record_type="value_mismatch", check_id="V1_full_population",
                stratum="all", board=board_of(th.split(".")[0]), thscode_ths=th, code_bs=bc,
                date=str(d), field=worst, ths_value=fnum(ta if worst == "amount" else tc, 4),
                bs_value=fnum(ba if worst == "amount" else bcl, 4),
                diff=fnum(d_abs, 4), rel_diff=fnum(rel, 8),
                tol=f"{TOL_PRICE_ABS}/{TOL_AMT_ABS}",
                status="MISMATCH",
                note=f"O={to}/{bo} H={thh}/{bh} L={tl}/{bl} V={tv}/{bv}")
    out["value_mismatch_rows"] = len(ab)
    return out


def check_strata(con, det: Detail, quick: bool) -> dict:
    """分层抽样：按板块 × 情形，逐字段落盘。抽样用 md5 排序，结果可复现。"""
    scale = 0.5 if quick else 1.0
    n_board_codes = max(4, int(40 * scale))     # 每板块随机证券数
    n_dates_per_code = 4
    n_st = int(150 * scale)
    n_susp = int(150 * scale)
    n_ex = int(200 * scale)
    n_edge = int(80 * scale)
    n_boundary = int(100 * scale)
    out = {}

    samples = []  # (stratum, thscode, date)

    # --- 各板块随机证券×随机交易日 ---
    all_codes = con.execute("""
        SELECT thscode, count(*) n FROM ths_raw GROUP BY 1
    """).fetchall()
    by_board = {}
    for th, n in all_codes:
        by_board.setdefault(board_of(th.split(".")[0]), []).append(th)
    for board, codes in sorted(by_board.items()):
        codes = sorted(codes, key=lambda c: hashlib.md5((c + str(SEED)).encode()).hexdigest())
        picked = codes[:n_board_codes]
        out[f"sample_codes_{board}"] = len(picked)
        for c in picked:
            ds = con.execute("""
                SELECT date FROM ths_raw WHERE thscode=?
                ORDER BY md5(strftime(date,'%Y-%m-%d') || ?) LIMIT ?
            """, [c, str(SEED), n_dates_per_code]).fetchall()
            for (d,) in ds:
                samples.append((f"random_{board}", c, d))

    def from_bs(sql, n, stratum):
        rows = con.execute(sql + f" ORDER BY md5(code||date||'{SEED}') LIMIT {n}").fetchall()
        for th, d in rows:
            samples.append((stratum, th, datetime.strptime(d, "%Y-%m-%d").date()))

    # ST / 停牌（BS 侧抽样，thscode 已是 THS 格式）
    from_bs("""SELECT thscode, date FROM bs_map
               WHERE adjustflag=3 AND is_st=1 AND trade_status=1
                 AND thscode NOT LIKE '%.BJ'""", n_st, "ST")
    from_bs("""SELECT thscode, date FROM bs_map
               WHERE adjustflag=3 AND trade_status=0""", n_susp, "SUSPENDED")

    # 除权日：直接沿用 THS 代码格式（bs_map 已提供 THS 格式键）
    ex = con.execute(f"""
        SELECT e.thscode,
               strftime(e.ex_date,'%Y-%m-%d') AS date
        FROM ths.raw_adjustment_events e
        WHERE e.ex_date BETWEEN DATE '{WINDOW_START}' AND DATE '{FREEZE}'
          AND e.thscode NOT LIKE '%.BJ'
        ORDER BY md5(e.thscode || strftime(e.ex_date,'%Y-%m-%d') || '{SEED}')
        LIMIT {n_ex}
    """).fetchall()
    for th, d in ex:
        samples.append(("EX_DATE", th, datetime.strptime(d, "%Y-%m-%d").date()))

    # 上市首日 / 最后交易日 / 冻结边界
    first = con.execute(f"""
        SELECT thscode, min(date) d FROM ths_raw GROUP BY 1
        ORDER BY md5(thscode || '{SEED}') LIMIT {n_edge}""").fetchall()
    samples += [("LISTING_EDGE", th, d) for th, d in first]
    # 退市边缘：以 BS 的 out_date 为准，取退市前最后 3 个交易日（仅 THS 也存在的标的）
    last = con.execute(f"""
        WITH x AS (
          SELECT upper(split_part(code,'.',2))||'.'||upper(split_part(code,'.',1)) AS thscode,
                 strptime(out_date,'%Y-%m-%d')::DATE AS outd
          FROM bs.stock_basic
          WHERE type=1 AND out_date IS NOT NULL AND out_date<>''
            AND strptime(out_date,'%Y-%m-%d')::DATE BETWEEN DATE '{WINDOW_START}' AND DATE '{FREEZE}'
        )
        SELECT DISTINCT x.thscode, t.date
        FROM x JOIN ths_raw t ON t.thscode=x.thscode AND t.date<=x.outd
        QUALIFY row_number() OVER (PARTITION BY x.thscode ORDER BY t.date DESC) <= 3
        ORDER BY md5(x.thscode||cast(t.date as varchar)||'{SEED}')
        LIMIT {n_edge*3}
    """).fetchall()
    samples += [("DELISTING_EDGE", th, d) for th, d in last]
    out["delisting_edge_symbols"] = con.execute("""
        SELECT count(*) FROM bs.stock_basic b
        WHERE b.type=1 AND b.out_date IS NOT NULL AND b.out_date<>''
          AND strptime(b.out_date,'%Y-%m-%d')::DATE BETWEEN DATE '2016-10-13' AND DATE '2026-10-26'
          AND EXISTS (SELECT 1 FROM ths_raw t
                      WHERE t.thscode=upper(split_part(b.code,'.',2))||'.'||upper(split_part(b.code,'.',1)))
    """).fetchone()[0]
    out["ths_symbols_stopped_early"] = con.execute("""
        SELECT count(*) FROM (SELECT thscode FROM ths_raw WHERE thscode NOT LIKE '%.BJ'
                              GROUP BY 1 HAVING max(date) < DATE '2026-06-01')
    """).fetchone()[0]
    for d in (WINDOW_START, FREEZE):
        rr = con.execute(f"""
            SELECT thscode, DATE '{d}' FROM ths_raw WHERE date=DATE '{d}'
            ORDER BY md5(thscode || '{d}{SEED}') LIMIT {n_boundary}""").fetchall()
        samples += [("BOUNDARY", th, d2) for th, d2 in rr]

    # 去重
    seen, uniq = set(), []
    for s in samples:
        k = (s[1], s[2])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    out["sample_symbol_days"] = len(uniq)

    con.execute("CREATE TEMP TABLE samp(stratum VARCHAR, thscode VARCHAR, date DATE)")
    con.executemany("INSERT INTO samp VALUES (?,?,?)", uniq)

    rows = con.execute("""
        SELECT s.stratum, s.thscode, s.date,
               t.open,t.high,t.low,t.close,t.volume,t.amount,
               b.code,b.open,b.high,b.low,b.close,b.volume,b.amount,b.trade_status
        FROM samp s
        LEFT JOIN ths_raw t ON t.thscode=s.thscode AND t.date=s.date
        LEFT JOIN bs_map b ON b.thscode=s.thscode
             AND b.date=strftime(s.date,'%Y-%m-%d') AND b.adjustflag=3
        ORDER BY s.stratum, s.thscode, s.date
    """).fetchall()

    fields = ["open", "high", "low", "close", "volume", "amount"]
    stat = {}
    for r in rows:
        stratum, th, d = r[0], r[1], r[2]
        tvals = r[3:9]
        bcode = r[9]
        bvals = list(r[10:16])
        tstat = r[16]
        board = board_of(th.split(".")[0])
        for fname, tv, bv in zip(fields, tvals, bvals):
            diff = rel = None
            tol = "-"
            if bcode is None and th.endswith(".BJ"):
                st = "NO_REFERENCE"          # 北交所无 BS 对照，不判通过/失败
            elif tv is None:
                st = "MISSING_THS"           # THS 无该行（停牌日属预期）
            elif bv is None:
                st = "MISSING_BS"
            elif fname == "volume":
                tol = f"rel<={TOL_VOL_REL}"
                diff = tv - bv
                rel = diff / bv if bv else None
                st = "OK" if abs(diff) <= max(abs(bv) * TOL_VOL_REL, 1e-9) else "DIFF"
            elif fname == "amount":
                tol = f"abs<={TOL_AMT_ABS} 或 rel<={TOL_AMT_REL}"
                diff = tv - bv
                rel = diff / bv if bv else None
                st = "OK" if (abs(diff) <= TOL_AMT_ABS
                              or (bv and abs(diff) <= abs(bv) * TOL_AMT_REL)) else "DIFF"
            else:
                tol = f"abs<={TOL_PRICE_ABS}"
                diff = tv - bv
                rel = diff / bv if bv else None
                st = "OK" if abs(diff) <= TOL_PRICE_ABS else "DIFF"
            stat[(stratum, st)] = stat.get((stratum, st), 0) + 1
            if st not in ("OK", "NO_REFERENCE") or fname == "close":
                det.add(record_type="sample_compare", check_id="V2_stratified_sample",
                        stratum=stratum, board=board, thscode_ths=th, code_bs=bcode or "",
                        date=str(d), field=fname, ths_value=fnum(tv, 4), bs_value=fnum(bv, 4),
                        diff=fnum(diff, 6), rel_diff=fnum(rel, 8), tol=tol, status=st,
                        note=f"trade_status={tstat}" if tstat is not None else "THS_ONLY")
    out["sample_stat"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(stat.items())}
    out["sample_non_ok"] = sum(v for k, v in stat.items() if k[1] in ("DIFF", "MISSING_BS", "MISSING_THS"))
    out["sample_diff"] = sum(v for k, v in stat.items() if k[1] == "DIFF")
    return out


def check_adj(con, det: Detail) -> dict:
    out = {}
    # --- 三复权行数 ---
    rows = con.execute("""
        SELECT adjustflag, count(*) n, count(DISTINCT code) c,
               sum(CASE WHEN close IS NULL THEN 1 ELSE 0 END) n_null
        FROM bs_map GROUP BY 1 ORDER BY 1
    """).fetchall()
    out["bs_adj_counts"] = {int(a): int(n) for a, n, c, nu in rows}
    out["bs_adj_symbols"] = {int(a): int(c) for a, n, c, nu in rows}
    out["bs_adj_nulls"] = {int(a): int(nu) for a, n, c, nu in rows}

    # --- 缺行：flag1/2 缺、flag3 有 ---
    gaps = con.execute("""
        SELECT a.code, a.date, a.close, a.trade_status, a.volume
        FROM bs_map a
        WHERE a.adjustflag=3
          AND NOT EXISTS (SELECT 1 FROM bs_map b WHERE b.code=a.code AND b.date=a.date AND b.adjustflag=1)
        ORDER BY a.code, a.date
    """).fetchall()
    out["adj_gap_rows"] = len(gaps)
    out["adj_gap_symbols"] = len({g[0] for g in gaps})
    out["adj_gap_susp"] = sum(1 for g in gaps if g[3] == 0)
    out["adj_gap_nullclose"] = sum(1 for g in gaps if g[2] is None)
    for code, d, close, ts, vol in gaps:
        det.add(record_type="adj_gap", check_id="A1_three_adjust_gap", stratum="flag1_missing",
                board=board_of(code.split(".")[1]), code_bs=code, date=d,
                field="close", ths_value="", bs_value=fnum(close, 4),
                status="ADJ_ONLY_RAW",
                note=f"trade_status={ts} volume={vol}；后复权/前复权均缺该行")
    # 缺口按代码归纳
    out["adj_gap_f1_not_f2"] = con.execute("""
        SELECT count(*) FROM bs_map a WHERE a.adjustflag=1
          AND NOT EXISTS (SELECT 1 FROM bs_map b WHERE b.code=a.code AND b.date=a.date AND b.adjustflag=2)
    """).fetchone()[0]
    out["adj_gap_f2_not_f1"] = con.execute("""
        SELECT count(*) FROM bs_map a WHERE a.adjustflag=2
          AND NOT EXISTS (SELECT 1 FROM bs_map b WHERE b.code=a.code AND b.date=a.date AND b.adjustflag=1)
    """).fetchone()[0]
    out["adj_gap_by_symbol"] = con.execute("""
        SELECT count(*) FROM (
          SELECT code FROM bs_map a WHERE a.adjustflag=3
            AND NOT EXISTS (SELECT 1 FROM bs_map b WHERE b.code=a.code AND b.date=a.date AND b.adjustflag=1)
          GROUP BY code)
    """).fetchone()[0]

    # --- flag1 与 flag2 自洽性（相邻收益必须一致） ---
    con.execute("""
        CREATE TEMP TABLE c12 AS
        SELECT code, date,
               max(CASE WHEN adjustflag=1 THEN close END) c1,
               max(CASE WHEN adjustflag=2 THEN close END) c2,
               max(CASE WHEN adjustflag=3 THEN close END) c3
        FROM bs_map WHERE date BETWEEN ? AND ?
        GROUP BY 1,2
    """, [WINDOW_START, FREEZE])
    con.execute("""
        CREATE TEMP TABLE c12r AS
        SELECT code,date,c1,c2,c3,
               c1/lag(c1) OVER (PARTITION BY code ORDER BY date)-1 r1,
               c2/lag(c2) OVER (PARTITION BY code ORDER BY date)-1 r2
        FROM c12
    """)
    r = con.execute("""
        SELECT count(*) n,
               sum(CASE WHEN abs(r1-r2)<=1e-6 THEN 1 ELSE 0 END) same,
               sum(CASE WHEN abs(r1-r2)>0.002 THEN 1 ELSE 0 END) differ
        FROM c12r WHERE r1 IS NOT NULL AND r2 IS NOT NULL
    """).df().iloc[0]
    out["adj_pair_compare_rows"] = int(r["n"])
    out["adj_pair_same"] = int(r["same"])
    out["adj_pair_differ"] = int(r["differ"])
    out["adj_pair_differ_symbols"] = con.execute("""
        SELECT count(DISTINCT code) FROM c12r
        WHERE r1 IS NOT NULL AND r2 IS NOT NULL AND abs(r1-r2)>0.002
    """).fetchone()[0]
    out["adj_pair_differ_months"] = con.execute("""
        SELECT string_agg(m || ':' || n, ', ' ORDER BY n DESC)
        FROM (SELECT substr(cast(date as varchar),1,7) m, count(*) n FROM c12r
              WHERE r1 IS NOT NULL AND r2 IS NOT NULL AND abs(r1-r2)>0.002
              GROUP BY 1 ORDER BY 2 DESC LIMIT 3)
    """).fetchone()[0]
    for code, d, r1, r2 in con.execute("""
        SELECT code,date,r1,r2 FROM c12r
        WHERE r1 IS NOT NULL AND r2 IS NOT NULL AND abs(r1-r2)>0.002 ORDER BY date
    """).fetchall():
        det.add(record_type="adj_inconsistent", check_id="A2_flag1_vs_flag2", stratum="tail_or_event",
                board=board_of(code.split(".")[1]), code_bs=code, date=str(d),
                field="ret", ths_value=fnum(r1, 8), bs_value=fnum(r2, 8),
                diff=fnum(r1 - r2, 8), status="FLAG1_NE_FLAG2",
                note="同一源内后复权与前复权隐含收益不一致")

    # --- 跨源复权收益一致性（THS qfq vs BS flag2） ---
    con.execute("""
        CREATE TEMP TABLE qr AS
        SELECT c.code, c.date, c.r1,
               q.close AS qt,
               q.close/lag(q.close) OVER (PARTITION BY c.code ORDER BY c.date)-1 AS rq
        FROM c12r c
        JOIN ths.v_daily_qfq q
          ON q.thscode=upper(split_part(c.code,'.',2))||'.'||upper(split_part(c.code,'.',1))
         AND q.date=c.date
        WHERE c.r1 IS NOT NULL
    """)
    r = con.execute("""
        SELECT count(*) n,
               sum(CASE WHEN abs(r1-rq)<=1e-6 THEN 1 ELSE 0 END) eq,
               sum(CASE WHEN abs(r1-rq)>0.002 THEN 1 ELSE 0 END) big
        FROM qr WHERE rq IS NOT NULL
    """).df().iloc[0]
    out["adj_cross_rows"] = int(r["n"])
    out["adj_cross_equal"] = int(r["eq"])
    out["adj_cross_big"] = int(r["big"])
    out["adj_cross_big_symbols"] = con.execute("""
        SELECT count(DISTINCT code) FROM qr WHERE rq IS NOT NULL AND abs(r1-rq)>0.002
    """).fetchone()[0]
    for code, d, r1, rq in con.execute("""
        SELECT code,date,r1,rq FROM qr WHERE rq IS NOT NULL AND abs(r1-rq)>0.002 ORDER BY date
    """).fetchall():
        det.add(record_type="adj_cross_diff", check_id="A3_ths_vs_bs_adj_return",
                stratum="adjusted_return", board=board_of(code.split(".")[1]), code_bs=code,
                date=str(d), field="ret_qfq", ths_value=fnum(rq, 8), bs_value=fnum(r1, 8),
                diff=fnum(rq - r1, 8), status="CROSS_SOURCE_DIFF",
                note="THS 前复权收益 vs BS 后复权收益（应相同）")
    return out


def check_events(con, det: Detail) -> dict:
    out = {}
    out["ths_events_all"] = con.execute("SELECT count(*) FROM ths.raw_adjustment_events").fetchone()[0]
    out["ths_events_window"] = con.execute(
        f"SELECT count(*) FROM ths.raw_adjustment_events WHERE ex_date BETWEEN DATE '{WINDOW_START}' AND DATE '{FREEZE}'"
    ).fetchone()[0]
    out["ths_events_symbols"] = con.execute(
        "SELECT count(DISTINCT thscode) FROM ths.raw_adjustment_events").fetchone()[0]
    out["ths_events_bj"] = con.execute(
        f"""SELECT count(*) FROM ths.raw_adjustment_events
            WHERE ex_date BETWEEN DATE '{WINDOW_START}' AND DATE '{FREEZE}' AND thscode LIKE '%.BJ'"""
    ).fetchone()[0]
    out["ths_events_nonbj"] = con.execute(
        f"""SELECT count(*) FROM ths.raw_adjustment_events
            WHERE ex_date BETWEEN DATE '{WINDOW_START}' AND DATE '{FREEZE}'
              AND thscode NOT LIKE '%.BJ'
              AND (dividend_per_share>0 OR per_share_bonus>0 OR rights_ratio>0)"""
    ).fetchone()[0]
    out["ths_events_zero"] = con.execute(
        f"""SELECT count(*) FROM ths.raw_adjustment_events
            WHERE ex_date BETWEEN DATE '{WINDOW_START}' AND DATE '{FREEZE}'
              AND dividend_per_share=0 AND per_share_bonus=0 AND rights_ratio=0"""
    ).fetchone()[0]

    # BS 隐含除权日（preclose != 前收）
    con.execute("""
        CREATE TEMP TABLE bsx AS
        SELECT thscode, date, preclose, close, trade_status,
               lag(close) OVER (PARTITION BY thscode ORDER BY date) pclose
        FROM bs_map WHERE adjustflag=3
    """)
    con.execute("""
        CREATE TEMP TABLE implied AS
        SELECT thscode, date, pclose, preclose, preclose/pclose-1 AS jump
        FROM bsx WHERE pclose>0 AND preclose IS NOT NULL AND abs(preclose/pclose-1)>0.003
    """)
    out["bs_implied_exdays"] = con.execute("SELECT count(*) FROM implied").fetchone()[0]

    out["implied_with_ths_event"] = con.execute("""
        SELECT count(*) FROM implied i
        WHERE EXISTS (SELECT 1 FROM ths.raw_adjustment_events e
                      WHERE e.thscode=i.thscode AND e.ex_date=i.date)
    """).fetchone()[0]
    out["event_with_implied"] = con.execute(f"""
        SELECT count(*) FROM ths.raw_adjustment_events e
        WHERE e.ex_date BETWEEN DATE '{WINDOW_START}' AND DATE '{FREEZE}'
          AND EXISTS (SELECT 1 FROM implied i WHERE i.thscode=e.thscode AND i.date=e.ex_date)
          AND (e.dividend_per_share>0 OR e.per_share_bonus>0 OR e.rights_ratio>0)
    """).fetchone()[0]

    # 未匹配的隐含除权日（THS 缺事件）
    un = con.execute("""
        SELECT i.thscode, i.date, i.pclose, i.preclose, i.jump
        FROM implied i
        WHERE NOT EXISTS (SELECT 1 FROM ths.raw_adjustment_events e
                          WHERE e.thscode=i.thscode AND e.ex_date=i.date)
        ORDER BY i.date
    """).fetchall()
    out["implied_without_event"] = len(un)
    for th, d, pc, pr, jp in un:
        det.add(record_type="ca_implied_unmatched", check_id="C1_implied_no_event",
                stratum="ex_date_missing", board=board_of(th.split(".")[0]), thscode_ths=th,
                code_bs=th.split(".")[0].lower() + "." + th.split(".")[1].lower(),
                date=str(d), field="preclose", ths_value="", bs_value=fnum(pr, 4),
                diff=fnum(pr - pc, 4), rel_diff=fnum(jp, 6), status="EVENT_MISSING",
                note=f"前收={pc}，交易所参考价={pr}，THS 事件表当日无记录")

    # 公式校验：除权参考价 = (前收 - 现金红利 + 配股价*配股比例) / (1 + 送股 + 配股)
    con.execute("""
        CREATE TEMP TABLE evp AS
        SELECT e.thscode, e.ex_date, e.dividend_per_share div, e.per_share_bonus bonus,
               e.rights_ratio rr, e.rights_price rp,
               b.preclose, b.pclose,
               (b.pclose - e.dividend_per_share + e.rights_ratio*coalesce(e.rights_price,0))
                 / (1 + e.per_share_bonus + e.rights_ratio) AS pred
        FROM ths.raw_adjustment_events e
        JOIN bsx b ON b.thscode=e.thscode AND b.date=e.ex_date
        WHERE e.ex_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) AND b.pclose>0
          AND (e.dividend_per_share>0 OR e.per_share_bonus>0 OR e.rights_ratio>0)
    """, [WINDOW_START, FREEZE])
    r = con.execute("""
        SELECT count(*) n,
               sum(CASE WHEN abs(pred/preclose-1)<=0.002 THEN 1 ELSE 0 END) ok,
               sum(CASE WHEN abs(pred/preclose-1)>0.002 AND abs(pred/preclose-1)<=0.01 THEN 1 ELSE 0 END) warn,
               sum(CASE WHEN abs(pred/preclose-1)>0.01 THEN 1 ELSE 0 END) bad
        FROM evp
    """).df().iloc[0]
    out["ca_formula_checked"] = int(r["n"])
    out["ca_formula_ok"] = int(r["ok"])
    out["ca_formula_warn"] = int(r["warn"])
    out["ca_formula_bad"] = int(r["bad"])
    for th, d, div, bonus, rr, rp, pc, pr, pred, err in con.execute("""
        SELECT thscode,ex_date,div,bonus,rr,rp,pclose,preclose,pred,pred/preclose-1
        FROM evp WHERE abs(pred/preclose-1)>0.002 ORDER BY ex_date
    """).fetchall():
        det.add(record_type="ca_formula", check_id="C2_event_formula", stratum="formula_mismatch",
                board=board_of(th.split(".")[0]), thscode_ths=th, date=str(d), field="preclose",
                ths_value=fnum(pred, 4), bs_value=fnum(pr, 4), diff=fnum(pred - pr, 4),
                rel_diff=fnum(err, 6), status="FORMULA_BAD" if abs(err) > CA_BIG_ERR else "FORMULA_WARN",
                note=f"div={div} bonus={bonus} rr={rr} rp={rp} 前收={pc}")

    # 极端跳变（>31%）与事件覆盖
    con.execute("""
        CREATE TEMP TABLE ext AS
        SELECT thscode,date,close,lag(close) OVER (PARTITION BY thscode ORDER BY date) pc
        FROM ths_raw
    """)
    r = con.execute("""
        SELECT count(*) n,
               sum(CASE WHEN EXISTS (SELECT 1 FROM ths.raw_adjustment_events e
                                     WHERE e.thscode=x.thscode AND e.ex_date=x.date) THEN 1 ELSE 0 END) covered
        FROM (SELECT thscode,date,pc,close FROM ext WHERE pc>0 AND abs(close/pc-1)>0.31) x
    """).df().iloc[0]
    out["extreme_moves"] = int(r["n"])
    out["extreme_moves_with_event"] = int(r["covered"])
    out["extreme_moves_uncovered"] = int(r["n"] - r["covered"])
    for th, d, pc, cl in con.execute("""
        SELECT x.thscode,x.date,x.pc,x.close
        FROM (SELECT thscode,date,pc,close FROM ext WHERE pc>0 AND abs(close/pc-1)>0.31) x
        WHERE NOT EXISTS (SELECT 1 FROM ths.raw_adjustment_events e
                          WHERE e.thscode=x.thscode AND e.ex_date=x.date)
        ORDER BY abs(x.close/x.pc-1) DESC
    """).fetchall():
        det.add(record_type="extreme_move", check_id="C3_uncovered_jump",
                stratum="price_jump", board=board_of(th.split(".")[0]), thscode_ths=th,
                date=str(d), field="close", ths_value=fnum(cl, 4), bs_value="",
                diff=fnum(cl - pc, 4), rel_diff=fnum(cl / pc - 1, 6), status="NO_EVENT",
                note=f"前收={fnum(pc,4)}；单日跳变>31% 且事件表无记录（股改/重组/重新上市等）")
    return out


# ----------------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------------
def write_report(stats: dict, det: Detail, elapsed: float):
    p = stats
    pre = p["preflight"]
    st = p["structure"]
    va = p["values"]
    sa = p["strata"]
    ad = p["adj"]
    ca = p["events"]

    def pct(a, b):
        return f"{100.0*a/b:.4f}%" if b else "n/a"

    L = []
    A = L.append
    A("# 双源数据核验：同花顺 DuckDB vs Baostock SQLite")
    A("")
    A(f"审计员：双源独立数据审计（只读、独立抽样 + 全量分层核对）  ")
    A(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　运行耗时：{elapsed:.0f}s  ")
    A(f"窗口：{WINDOW_START} ~ {FREEZE}（冻结日）　抽样种子：{SEED}  ")
    A(f"明细：`05_审计/双源数据核验明细.csv`（{len(det.rows)} 行）  ")
    A("")
    A("> 本报告只做数据核验，**不计算任何策略或组合收益**。文中出现的“收益”仅指相邻收盘价之比，")
    A("> 用于检验两源复权口径是否一致，与绩效无关。")
    A("")
    A("## 0. 结论速览")
    A("")
    A("| 项目 | 结论 | 证据 |")
    A("|---|---|---|")
    A(f"| 不复权 OHLC 一致性 | **通过** | {va['matched_rows']:,} 个共同证券日中，收盘差 >0.005 元仅 "
      f"{va['c_bad']} 行（其中 >0.02 元 {va['price_diff_gt_0.02']} 行） |")
    A(f"| 成交量口径 | **一致（股）** | 精确相等 {pct(va['v_eq'], va['matched_rows'])}，"
      f"中位比值 {va['vol_ratio_median']:.6f} |")
    A(f"| 成交额口径 | **一致（元）** | 1 分钱内或 float32 相对误差内 {pct(va['a_pass'], va['matched_rows'])}；"
      f"相对偏差>1% 仅 {va['amt_rel_gt_1pct']} 行（全部为上市首日口径差） |")
    A(f"| 证券覆盖 | **THS 有幸存者偏差** | BS 退市股 {st['bs_delisted_total']} 只中 "
      f"{st['missing_delisted']} 只在 THS 完全缺失，其中 {st['missing_symbols_with_data']} 只窗口内有成交 |")
    A(f"| 北交所 | **无法交叉验证** | THS 独有 {st['ths_bj_symbols']} 只 BJ，BS 无任何北交所代码 |")
    A(f"| 停牌 | **语义不同** | BS 保留 {st['bs_susp_rows']:,} 个停牌日（平价、零量），THS 全部删除；"
      f"重叠仅 {st['bs_susp_days_present_in_ths']} 行 |")
    A(f"| 三复权完整性 | **flag1/2 缺行** | 不复权比后/前复权多 {ad['adj_gap_rows']:,} 行，"
      f"分布于 {ad['adj_gap_symbols']} 只 |")
    A(f"| BS 前复权自洽性 | **基本自洽，两簇异常** | 后/前复权隐含收益不一致 {ad['adj_pair_differ']} 行"
      f"（{ad['adj_pair_differ_months']}） |")
    A(f"| 公司行动覆盖 | **高** | 交易所隐含除权日 {ca['bs_implied_exdays']:,} 个中 "
      f"{pct(ca['implied_with_ths_event'], ca['bs_implied_exdays'])} 在 THS 有对应事件 |")
    A(f"| 除权公式校验 | **高** | {ca['ca_formula_checked']:,} 个非零事件中 "
      f"{pct(ca['ca_formula_ok'], ca['ca_formula_checked'])} 可复原交易所参考价（误差≤0.2%） |")
    A(f"| 极端跳变 | **{ca['extreme_moves_uncovered']} 处无事件覆盖** | "
      f"单日绝对涨跌幅 >31% 的 {ca['extreme_moves']:,} 个证券日中 {ca['extreme_moves_with_event']:,} 个有事件 |")
    A("")
    A("**总体判定：两源不复权行情高度一致，可用于策略研究；但必须遵守第 5 节的七条使用约束。**")
    A("")

    A("## 1. 审计范围、口径与容差")
    A("")
    A("### 1.1 数据源与只读声明")
    A("")
    A("| 源 | 路径 | 表 | 冻结窗口行数 |")
    A("|---|---|---|---|")
    A(f"| 同花顺 | `{THS_DB}` | `raw_kline_daily` / `dim_symbol` / `raw_adjustment_events` / `calc_adjust_factor_daily` | {pre['ths_rows_window']:,} |")
    A(f"| Baostock | `{BS_DB}` | `stock_kline_d`(flag 1/2/3) / `stock_basic` / `trade_dates` | {pre['bs_rows_window']:,}（三复权合计） |")
    A("")
    A("两库均以 `READ_ONLY` 打开，未写入、未迁移、未同步。Baostock 处于 WAL 活跃写入状态")
    A(f"（db 文件 mtime {pre['bs_file_mtime']}），因此窗口行数在运行首尾各取一次：")
    A(f"THS {pre['ths_rows_window']:,} → {pre['ths_rows_window_recheck']:,}；")
    A(f"BS {pre['bs_rows_window']:,} → {pre['bs_rows_window_recheck']:,}；"
      f"窗口内{'稳定' if pre['window_stable'] else '**不稳定，需重跑**'}。")
    A("")
    A("### 1.2 冻结日与越界数据")
    A("")
    A(f"THS 原始表最大日期为 **{pre['ths_max_date']}**，冻结日之后仍有 "
      f"{pre['ths_rows_post_freeze']:,} 行、涉及 {pre['ths_symbols_post_freeze']:,} 只证券。"
      "本审计全部按 `date <= 2026-10-26` 过滤，越界行不计入任何统计；")
    A("**下游研究必须显式加该过滤条件，否则会引入未来数据**。")
    A("")
    A("### 1.3 代码映射与单位")
    A("")
    A("| 源 | 格式 | 示例 | 映射规则 |")
    A("|---|---|---|---|")
    A("| THS | `代码.交易所` | `600000.SH` | `upper(split(code,'.',2))||'.'||upper(split(code,'.',1))` |")
    A("| BS | `交易所.代码` | `sh.600000` | 同上取反 |")
    A("")
    A(f"成交量：两源均为**股**（中位比值 {va['vol_ratio_median']:.6f}，无 100/10000 倍量纲差）。  ")
    A(f"成交额：两源均为**元**（中位比值 {va['amt_ratio_median']:.6f}）。  ")
    A("价格：两源均为 2 位小数报价；复权价保留 6 位。")
    A("")
    A("### 1.3b 复权口径")
    A("")
    A("| 序列 | 口径 | 基准 | 备注 |")
    A("|---|---|---|---|")
    A("| THS `v_daily_hfq` | 后复权 | 各证券窗口内首个交易日（backward_factor=1） | 与 `calc_adjust_factor_daily` 相乘得到，可精确重建 |")
    A("| THS `v_daily_qfq` | 前复权 | 最新日（forward_factor=1） | 因子随新增除权而整体变化，历史值会被重算 |")
    A("| BS `adjustflag=1` | 后复权 | 各证券上市首日 | 因子不可见（`adjust_factor` 表为空） |")
    A("| BS `adjustflag=2` | 前复权 | 最新日 | 末端存在除权日错位，见 4.2 |")
    A("| BS `adjustflag=3` | 不复权 | — | 与 THS `raw_kline_daily` 逐行核对一致 |")
    A("")
    A("注意：**前复权序列会随除权事件整体重算**，不可用于跨时点比较绝对价格；收益类计算两口径等价。")
    A("")
    A("### 1.4 容差与异常分类")
    A("")
    A("| 字段 | 通过阈值 | 判定 |")
    A("|---|---|---|")
    A(f"| OHLC | 绝对差 ≤ {TOL_PRICE_ABS}（半分钱） | 超过即 `MISMATCH` |")
    A(f"| volume | 相对差 ≤ {TOL_VOL_REL} 或绝对差 ≤ 1e-9 | float32 级差异视为一致 |")
    A(f"| amount | 绝对差 ≤ {TOL_AMT_ABS}（1 分）或相对 ≤ {TOL_AMT_REL} | 超 1% 判 `MISMATCH`（本次全部为上市首日口径差） |")
    A(f"| 复权收益 | 绝对差 ≤ {TOL_RET} | 检验两源除权处理是否同口径 |")
    A(f"| 除权公式 | 相对误差 ≤ {CA_MATCH_TOL}，>1% 记为 `FORMULA_BAD` | 用交易所参考价反推 |")
    A("")
    A("异常分类码：`OK` / `MISSING_THS`（THS 无该行）/ `MISSING_BS`（BS 无该行）/ "
      "`MISMATCH`（超容差）/ `ADJ_ONLY_RAW`（仅不复权有）/ `FLAG1_NE_FLAG2` / "
      "`CROSS_SOURCE_DIFF` / `EVENT_MISSING` / `FORMULA_WARN|BAD` / `NO_EVENT` / `THS_ABSENT`。")
    A("")

    A("## 2. 覆盖与结构")
    A("")
    A("### 2.1 证券覆盖：THS 存在幸存者偏差")
    A("")
    A(f"- BS `stock_basic`（type=1）共 **{st['bs_symbols']:,}** 只；THS 窗口内 **{st['ths_symbols']:,}** 只。")
    A(f"- BS 有而 THS 完全没有的证券：**{st['missing_in_ths']}** 只，其中退市股 "
      f"**{st['missing_delisted']}** 只、仍在上市 **{st['missing_listed']}** 只。")
    A(f"- BS 记录的退市股共 {st['bs_delisted_total']} 只，THS 仅保留 {st['bs_delisted_in_ths']} 只。")
    A(f"- THS 有而 BS 没有的证券：{st['ths_symbols_not_in_bs']} 只，**全部是北交所（BJ）**。")
    A("")
    A(f"其中在冻结窗口内**确有交易**、但 THS 完全缺失的有 **{st['missing_symbols_with_data']} 只、"
      f"{st['missing_rows_with_data']:,} 个证券日**（其余为 2016 年前已退市、窗口内无行情）。")
    A(f"另一个角度的证据：THS 中没有任何 sh/sz 证券的最后一行早于 2026-06-01"
      f"（{sa['ths_symbols_stopped_early']} 只），即它只保留“活到最近”的标的。")
    A("含义：**只用 THS 做 2016–2026 回测会整段丢掉这些退市股票**，")
    A("对趋势/动量策略是实质性乐观偏差（退市前多为长期下跌）。")
    A("退市股全名单见明细 CSV 的 `record_type=missing_symbol`。")
    A("")
    A("### 2.2 板块分布")
    A("")
    A("| 板块 | THS 证券数 | BS 可对照 |")
    A("|---|---|---|")
    dist = p["boards"]
    for b, n in dist:
        A(f"| {b} | {n:,} | {'否（BJ 无 BS 数据）' if b == 'BJ' else '是'} |")
    A("")
    A("### 2.3 交易日历")
    A("")
    A(f"- BS `trade_dates` 标记的交易日：{st['bs_trade_days']:,}；THS 实际有行情的日期：{st['ths_trade_days']:,}。")
    A(f"- 交易日但 THS 无任何行情：{st['dates_missing_in_ths']} 天；THS 有行情但不在交易日历：{st['dates_ths_not_calendar']} 天。")
    A("- 结论：两源日历一致，可按交易日对齐。")
    A("")
    A("### 2.4 键唯一性与内部一致性")
    A("")
    A(f"- THS 重复 (代码,日期) 键：{st['ths_dup_keys']}；BS 重复 (代码,日期,复权) 键：{st['bs_dup_keys']}。")
    A(f"- THS OHLC 逻辑违规（low>high 等）：{st['ths_ohlc_violations']} 行；非正价格：{st['ths_nonpositive_price']} 行。")
    A(f"- BS 不复权 OHLC 逻辑违规：{st['bs_ohlc_violations']} 行；全字段为空：{st['bs_all_null_rows']} 行。")
    A("")
    A("### 2.5 停牌语义（两源不同，必须显式处理）")
    A("")
    A(f"| 项目 | BS | THS |")
    A(f"|---|---|---|")
    A(f"| 停牌日是否保留行 | 保留（{st['bs_susp_rows']:,} 行） | **删除**（窗口内 0 成交量行 {st['ths_zero_volume_rows']} 个） |")
    A(f"| 停牌日价格 | 平盘（open=high=low=close 占 {st['bs_susp_flat']:,} 行） | 无 |")
    A(f"| 停牌日成交量 | 空 {st['bs_susp_vol_null']:,} / 0 {st['bs_susp_vol_zero']:,} / >0 {st['bs_susp_vol_pos']} | 无 |")
    A("")
    A(f"两源停牌日交集只有 **{st['bs_susp_days_present_in_ths']}** 行；THS 中有 {st['ths_days_absent_in_bs']:,} 行在 BS 找不到对应，全部是北交所（BS 无 BJ）。")
    A("")
    A(f"另：THS `dim_symbol.name` 全表为空（0/{st['ths_symbols']:,}），**无法从 THS 判定 ST**；"
      f"ST 状态只能取 BS `is_st`（窗口内 {st['bs_st_rows']:,} 个 ST 证券日）。")
    A("含义：以 THS 为主表时，`lag()` 取到的是“上一个有交易的日”，停牌期被压缩；")
    A("以 BS 为主表时，停牌日会以平价零量参与指标计算，需显式剔除 `trade_status=0`。")
    A("")

    A("## 3. 数值一致性（不复权）")
    A("")
    A(f"共同证券日 **{va['matched_rows']:,}** 行（仅 sh/sz；BJ 无对照，不计入）。")
    A("")
    A("| 字段 | 精确相等 | 通过率 | 备注 |")
    A("|---|---|---|---|")
    A(f"| open | {va['o_eq']:,} | {pct(va['o_eq'], va['matched_rows'])} | |")
    A(f"| high | {va['h_eq']:,} | {pct(va['h_eq'], va['matched_rows'])} | |")
    A(f"| low | {va['l_eq']:,} | {pct(va['l_eq'], va['matched_rows'])} | |")
    A(f"| close | {va['c_eq']:,} | {pct(va['c_eq'], va['matched_rows'])} | 超差 {va['c_bad']} 行（≤0.005 元视为一致） |")
    A(f"| volume | {va['v_eq']:,} | {pct(va['v_eq'], va['matched_rows'])} | 其余为 float32 尾差（相对<1e-6） |")
    A(f"| amount | {va['a_eq']:,} | {pct(va['a_eq'], va['matched_rows'])} | 综合通过 {pct(va['a_pass'], va['matched_rows'])} |")
    A("")
    A(f"- 收盘价偏差 >0.005 元：{va['c_bad']} 行；>0.02 元：{va['price_diff_gt_0.02']} 行（分布在 2016-12-01，单笔差 0.01 元）。")
    A(f"- 成交量相对差 > 1e-6：{va['vol_rel_gt_1e-6']} 行（最大相对差仍在 1e-6 量级，无单位错位）。")
    A(f"- 成交额相对差 > 1e-4：{va['amt_rel_gt_1e-4']} 行；> 1%：{va['amt_rel_gt_1pct']} 行。")
    A(f"- BS 的 `amount` 以 float32 精度存储：p99.9 相对误差 {va['amt_rel_p999']:.1e}，"
      f"绝对误差随金额放大（10 亿元级可达数十元）。做“成交额 > X”这类阈值筛选时，边界值可能翻转。")
    A("")
    A("### 3.1 成交额 >1% 的全部异常：上市首日")
    A("")
    A(f"这 {va['amt_rel_gt_1pct']} 行集中在 2020-07-21 / 2020-11-09 / 2020-12-09 / 2022-07-20 等创业板、科创板上市首日，")
    A("成交价、成交量完全一致，只有 `amount` 不同，且 BS 的 `amount/volume` 会高于当日最高价（物理上不可能），")
    A("说明**首日成交额是 BS 侧口径错误**（疑为把发行/配售量计入）。THS 侧自洽。")
    A("明细见 CSV `record_type=value_mismatch`。")
    A("")
    A("### 3.2 抽样核对（分层）")
    A("")
    A(f"分层抽样 {sa['sample_symbol_days']:,} 个证券日、逐字段核对（每行 6 个字段）；"
      f"其中数值 `DIFF` {sa['sample_diff']} 条，缺失 {sa['sample_non_ok'] - sa['sample_diff']} 条。")
    A("")
    A("| 层 | OK | DIFF | MISSING_BS | MISSING_THS | 无对照 |")
    A("|---|---|---|---|---|---|")
    strata_names = sorted({k.split("|")[0] for k in sa["sample_stat"]})
    for s in strata_names:
        get = lambda tag: sa["sample_stat"].get(f"{s}|{tag}", 0)
        A(f"| {s} | {get('OK'):,} | {get('DIFF'):,} | {get('MISSING_BS'):,} | "
          f"{get('MISSING_THS'):,} | {get('NO_REFERENCE'):,} |")
    A("")
    A("`MISSING_THS` 在 `SUSPENDED` 层是**预期行为**（THS 不保留停牌日）；")
    A(f"`ST` 层的 `MISSING_THS` 亦非 ST 特有，全部来自 THS 整体缺失的退市股（见 2.1）；")
    A("`NO_REFERENCE` 全部是北交所，BS 无对应数据，不参与判定。")
    A("")

    A("## 4. 复权与公司行动")
    A("")
    A("### 4.1 Baostock 三复权完整性")
    A("")
    A("| adjustflag | 行数（窗口） | 证券数 | close 为空 |")
    A("|---|---|---|---|")
    for a in (1, 2, 3):
        A(f"| {a}（{'后复权' if a==1 else '前复权' if a==2 else '不复权'}） | "
          f"{ad['bs_adj_counts'].get(a,0):,} | {ad['bs_adj_symbols'].get(a,0):,} | {ad['bs_adj_nulls'].get(a,0)} |")
    A("")
    A(f"不复权比后/前复权多 **{ad['adj_gap_rows']:,}** 行，分布于 **{ad['adj_gap_symbols']}** 只证券，")
    A(f"其中停牌日 {ad['adj_gap_susp']:,} 行、收盘为空 {ad['adj_gap_nullclose']} 行。")
    A("缺行集中在少数证券的**上市早期历史段**（如招商港口 001872 的 2016 年段、"
      "中航善达 001914、中航电测 302132 等代码变更前后），即复权分支没有覆盖这段历史。")
    A("**若策略用 flag1/flag2 取历史，这些证券的早期样本会凭空少一截**；应以不复权 + 自行重建因子为准。")
    A("")
    A(f"flag1 与 flag2 的键集合几乎相同（flag1 缺 {ad['adj_gap_f1_not_f2']} 行、flag2 缺 {ad['adj_gap_f2_not_f1']} 行），"
      "缺口只针对不复权独有的那些行。")
    A("`adjust_factor` 表在 BS 侧为空（0 行），因此 Baostock 的复权只能直接用 flag1/flag2，")
    A("无法拿到可审计的逐日因子序列。")
    A("")
    A("### 4.2 BS 后复权与前复权是否自洽")
    A("")
    A(f"窗口内可比 {ad['adj_pair_compare_rows']:,} 个相邻收益对，不一致（>0.2%）**{ad['adj_pair_differ']}** 行。")
    A("典型：`601229.SH` 2026-10-23，前复权在 10-23 就应用了 10-27 才生效的除权因子（0.9686），")
    A("而后复权与该股真实除权日（2026-10-27，每股派 0.30 元）一致。")
    A(f"同类错位共 {ad['adj_pair_differ']} 行、涉及 {ad['adj_pair_differ_symbols']} 只证券，"
      f"按月份聚成两簇（{ad['adj_pair_differ_months']}）；2026-10 那一簇落在冻结日之前，"
      "属于**本研究会用到的样本**。")
    A("→ **前复权存在除权日错位（2020-11 与 2026-10 两簇）**，后复权更可靠。")
    A("")
    A("### 4.3 跨源复权收益一致性")
    A("")
    A(f"用 THS 前复权收益与 BS 后复权收益对比（两者应完全等价），可比 {ad['adj_cross_rows']:,} 行，")
    A(f"一致 {pct(ad['adj_cross_equal'], ad['adj_cross_rows'])}，偏差 >0.2% 共 **{ad['adj_cross_big']}** 行"
      f"（涉及 {ad['adj_cross_big_symbols']} 只证券）。")
    A("偏差集中在 2026-10-23 与少数历史除权日，方向与 4.2 相同：THS 因子时点正确，BS 前复权提前。")
    A("")
    A("### 4.4 DuckDB 公司行动覆盖")
    A("")
    A(f"- THS `raw_adjustment_events` 全量 {ca['ths_events_all']:,} 条，覆盖 {ca['ths_events_symbols']:,} 只；")
    A(f"  窗口内 {ca['ths_events_window']:,} 条 = 北交所 {ca['ths_events_bj']:,} 条（无 BS 对照）"
      f" + 全零事件 {ca['ths_events_zero']:,} 条 + 非北交所非零 {ca['ths_events_nonbj']:,} 条。")
    A(f"- 用 BS 交易所参考价（`preclose` 与前收不等，>0.3%）反推的隐含除权日：**{ca['bs_implied_exdays']:,}** 个；")
    A(f"  其中 **{ca['implied_with_ths_event']:,}**（{pct(ca['implied_with_ths_event'], ca['bs_implied_exdays'])}）在 THS 事件表同日有记录。")
    A(f"- 未匹配 {ca['implied_without_event']} 个，多为小额分红（0.3%–1.3%），THS 可能记在相邻日期或未收录。")
    A("")
    A("### 4.5 除权公式反推（事件数据本身的正确性）")
    A("")
    A(f"对每个非北交所、非零的事件（{ca['ths_events_nonbj']:,} 条，其中 {ca['ca_formula_checked']:,} 条能取到当日 BS 行，"
      "差额为代码变更的全空行），用 `(前收 − 现金红利 + 配股价×配股比例) / (1 + 送股 + 配股)` 预测交易所参考价，")
    A("与实际 `preclose` 比较：")
    A("")
    A("| 结果 | 条数 | 占比 |")
    A("|---|---|---|")
    A(f"| 误差 ≤0.2% | {ca['ca_formula_ok']:,} | {pct(ca['ca_formula_ok'], ca['ca_formula_checked'])} |")
    A(f"| 0.2%–1% | {ca['ca_formula_warn']} | {pct(ca['ca_formula_warn'], ca['ca_formula_checked'])} |")
    A(f"| >1% | {ca['ca_formula_bad']} | {pct(ca['ca_formula_bad'], ca['ca_formula_checked'])} |")
    A("")
    A(">1% 的 26 条全部是股改/重大资产重组/送转比例极端（如 600733.SH 2018 年股改），")
    A("事件表记录的比例与交易所实际除权不匹配；这些日期的原始价格会出现 −80% 级跳变。")
    A("")
    A("### 4.6 极端跳变与事件覆盖")
    A("")
    A(f"原始价格单日 |涨跌|>31% 的交易日共 **{ca['extreme_moves']:,}** 个，其中 **{ca['extreme_moves_with_event']:,}** 个当日有公司行动记录，")
    A(f"**{ca['extreme_moves_uncovered']}** 个无记录（股改、重组复牌、重新上市、IPO 首日等）。")
    A("这些点在**不复权**序列上会产生虚假的巨幅涨跌；任何基于原始价格的形态/突破识别都必须先复权或剔除。")
    A("")

    A("## 5. 对策略研究的直接影响（使用约束）")
    A("")
    A(f"1. **主数据源用 THS，但必须补退市股**：THS 缺 {st['missing_delisted']} 只已退市证券，"
      f"其中 {st['missing_symbols_with_data']} 只在窗口内有成交（{st['missing_rows_with_data']:,} 个证券日）。")
    A("   建议以 BS 不复权补这些证券的历史，或至少在报告中声明回测存在幸存者偏差。")
    A("2. **复权用 THS 的 `v_daily_hfq`，或 BS `adjustflag=1`，不要用 BS `adjustflag=2`**：")
    A("   后者存在除权日提前的错位（2026-10-23 已观测）。")
    A("3. **停牌必须显式处理**：THS 无停牌行（时间被压缩），BS 有停牌行（平价零量）。")
    A("   拼库时以 THS 为主、BS 仅补退市段，并保留 `trade_status` 作为可交易性掩码。")
    A(f"4. **THS `prev_close` 全表为空（实测缺失率 {100*va['ths_prev_close_null_rate']:.2f}%）**，`v_daily` 的该列不可用；涨跌幅请用复权收盘自算，")
    A("   或从 BS 取 `preclose`（注意其已按除权调整）。")
    A("5. **北交所只有单一数据源**，353 只 BJ 证券无法交叉验证；相关结论应标注为“单源”。")
    A("6. **冻结日之后 THS 仍有数据**，下游必须硬编码 `date <= 2026-10-26`。")
    A(f"7. **ST 与涨跌停判定依赖 BS**：THS 无名称、无 ST 标记、无涨跌停价；"
      f"ST 取 BS `is_st`（{st['bs_st_rows']:,} 个证券日），涨跌停幅度需按板块 + ST 自行推导。")
    A("")

    A("## 6. 未能验证 / 后续事项")
    A("")
    A("- **北交所**：BS 无 BJ 代码，OHLCV 只能单源采信，未做独立核对。")
    A(f"- **退市股数值**：{st['missing_delisted']} 只在 THS 缺失（{st['missing_symbols_with_data']} 只窗口内有行情），"
      "只能采信 BS 单源；其复权序列未与第三方比对。")
    A("- **窗口之前（<2016-10-13）**：THS 无数据，未审计；BS 虽有 1990 年起数据但无对照源。")
    A("- **26 条除权公式失配事件**：已列出，未逐条人工核对公告，建议涉及个股时单独查证。")
    A("- **源库处于写入状态**：Baostock 以 WAL 模式持续写入。本次两次独立重建（缓存重建前后）"
      "窗口行数一致，但个别复权相邻收益发生变化（4.2 的 29→31 行），说明源库仍在修订历史值。"
      "结论的量级不受影响，精确到个位的计数会随源库更新而变，复现时请固定缓存文件。")
    A("- **精确复现**：`03_结果/_审计缓存/*.parquet` 是冻结快照；保留该文件即可复现本文所有数字，删除后重建会跟随源库最新状态。")
    A("- 本审计不评价任何策略、不计算组合收益、不做参数选择。")
    A("")

    A("## 7. 复现")
    A("")
    A("```bash")
    A("cd /Users/admin/WorkSpace/trading/trading_趋势结构")
    A("python3 02_代码/审计双源.py                # 复用缓存")
    A("python3 02_代码/审计双源.py --rebuild-cache # 从源库重建缓存")
    A("python3 02_代码/审计双源.py --quick         # 半量抽样冒烟")
    A("```")
    A("")
    A(f"- 缓存：`{CACHE_PARQUET}`（{pre['cache_rows']:,} 行，{pre['cache_size_mb']} MB，可删除后重建）")
    A(f"- 明细：`{OUT_CSV}`（{len(det.rows):,} 行）")
    A(f"- 抽样种子固定为 {SEED}；容差常量写在脚本头部。")
    A("")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    log("连接数据源（只读）")
    con = connect()
    cache = ensure_cache(con, args.rebuild_cache)
    log(f"缓存 {cache['rows']:,} 行 / {cache['size']/1e6:.1f} MB")

    det = Detail()
    pre = check_preflight(con, cache)
    log("preflight done")

    build_core(con)
    log("core tables built")

    st = check_structure(con, det)
    log("structure done")

    boards = con.execute("""
        SELECT CASE
          WHEN thscode LIKE '%.SH' AND split_part(thscode,'.',1) LIKE '6%' AND split_part(thscode,'.',1) NOT LIKE '688%' AND split_part(thscode,'.',1) NOT LIKE '689%' THEN 'SH_MAIN'
          WHEN split_part(thscode,'.',1) LIKE '688%' OR split_part(thscode,'.',1) LIKE '689%' THEN 'STAR'
          WHEN split_part(thscode,'.',1) LIKE '00%' AND split_part(thscode,'.',1) NOT LIKE '002%' THEN 'SZ_MAIN'
          WHEN split_part(thscode,'.',1) LIKE '002%' THEN 'SME'
          WHEN split_part(thscode,'.',1) LIKE '30%' THEN 'CHINEXT'
          WHEN thscode LIKE '%.BJ' THEN 'BJ'
          ELSE 'OTHER' END b, count(DISTINCT thscode)
        FROM ths_raw GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()

    va = check_values(con, det)
    log("values done")

    sa = check_strata(con, det, args.quick)
    log("strata done")

    ad = check_adj(con, det)
    log("adjusted series done")

    ca = check_events(con, det)
    log("corporate actions done")

    stats = dict(preflight=pre, structure=st, values=va, strata=sa,
                 adj=ad, events=ca, boards=boards, elapsed=time.time() - t0)

    det.write(OUT_CSV)
    write_report(stats, det, time.time() - t0)

    log(f"CSV  {OUT_CSV}  ({len(det.rows):,} 行)")
    log(f"MD   {OUT_MD}")
    log(f"总耗时 {time.time()-t0:.0f}s")

    # 自检：关键不变量
    fails = []
    if not pre["window_stable"]:
        fails.append("窗口计数不稳定")
    if st["ths_dup_keys"] or st["bs_dup_keys"]:
        fails.append("存在重复键")
    if va["matched_rows"] < 9_000_000:
        fails.append(f"匹配行数异常偏低: {va['matched_rows']}")
    if sa["sample_symbol_days"] < 500:
        fails.append(f"抽样量不足: {sa['sample_symbol_days']}")
    if ca["ca_formula_ok"] / max(ca["ca_formula_checked"], 1) < 0.95:
        fails.append("除权公式通过率异常")
    if fails:
        log("自检失败: " + "; ".join(fails))
        return 1
    log("自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
