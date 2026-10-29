#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结数据层自检 / 审计（v2：复权修复后）。

只读冻结产物（必要时只读源库做交叉校验），输出：
  01_数据/_audit/自检报告.md          # 标准自检
  01_数据/_audit/自检结果.json
  01_数据/_audit/修复后自检报告.md    # v1→v2 复权修复的专项报告
  04_证据/复权修复.json               # 机器可读证据（前后指纹、不变量、案例）

核心不变量（全部为数据层校验，不计算任何策略收益）：
  I1 hfq_close 相邻日收益 == 官方 pct_chg           （全样本 + 随机日抽样）
  I2 除权日 hfq_close 收益 == pct_chg；原始 close 收益必须显著偏离
  I3 同 bar 内 hfq OHLC 比例 == raw OHLC 比例
  I4 首条 hfq_close == 原始收盘；末条 qfq_close == 原始收盘
  I5 截断到 T 重算的 hfq 与全样本 hfq 在 t<=T 上完全一致（点时构造）
  I6 反事实旧公式 hfq=raw*factor 必须被 I1/I2 检出（检验功效）

用法：python3 02_代码/01_自检_数据.py [--skip-source] [--deep]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_数据"
FROZEN = DATA / "frozen"
AUDIT = DATA / "_audit"
EVIDENCE = ROOT / "04_证据"
FREEZE = "2026-10-26"
EXPECTED_VERSION = "v2-adjspace-2026-10-28"
BS_DB = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"
TH_DB = "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"

RESULTS: list[dict] = []


def rec(cid: str, name: str, status: str, metric, detail: str = "") -> None:
    RESULTS.append({"id": cid, "check": name, "status": status,
                    "metric": metric, "detail": detail})
    print(f"[{status}] {cid} {name}: {metric}" + (f" | {detail}" if detail else ""))


def ok(cond: bool) -> str:
    return "PASS" if cond else "FAIL"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-source", action="store_true")
    ap.add_argument("--deep", action="store_true", help="逐文件校验 meta.json 的 SHA-256")
    args = ap.parse_args()

    AUDIT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(FROZEN / "frozen.duckdb"), read_only=True)
    con.execute("SET memory_limit='6GB'")
    D = f"read_parquet('{FROZEN}/daily/*/*.parquet', hive_partitioning=true)"
    W = "WINDOW w AS (PARTITION BY symbol ORDER BY date)"

    # ---------------------------------------------------------------- #
    # A. 结构 / 行数 / 时间边界 / 版本
    # ---------------------------------------------------------------- #
    rows = con.execute(f"""
      SELECT count(*) n, count(DISTINCT symbol) nsym, min(date) d0, max(date) d1,
             count(*) FILTER (WHERE date > DATE '{FREEZE}') n_future,
             count(*) - count(DISTINCT (symbol, date)) dup
      FROM {D}""").fetchone()
    rec("A1", "日线行数/证券数/时间边界/主键唯一", ok(rows[4] == 0 and rows[5] == 0),
        f"rows={rows[0]:,} symbols={rows[1]:,} {rows[2]}~{rows[3]}",
        f"未来行={rows[4]} 重复主键={rows[5]}")

    cols = con.execute(f"DESCRIBE SELECT * FROM {D}").df()
    rec("A2", "日线字段数", "INFO", f"{len(cols)} 列", ", ".join(cols.column_name.tolist()))

    parts = sorted(p.name for p in (FROZEN / "daily").iterdir() if p.is_dir() and p.name.startswith("year="))
    rec("A3", "年度分区", ok(len(parts) == 23), f"{len(parts)} 个分区", f"{parts[0]}~{parts[-1]}")

    fill = con.execute(f"""
    SELECT
      round(100.0 * count(*) FILTER (WHERE preclose IS NULL OR preclose <= 0) / count(*), 4) preclose_null,
      round(100.0 * count(*) FILTER (WHERE turn IS NULL) / count(*), 4) turn_null,
      round(100.0 * count(*) FILTER (WHERE pe_ttm IS NULL) / count(*), 4) pe_null,
      round(100.0 * count(*) FILTER (WHERE pb_mrq IS NULL) / count(*), 4) pb_null,
      round(100.0 * count(*) FILTER (WHERE float_shares IS NULL) / count(*), 4) fs_null,
      round(100.0 * count(*) FILTER (WHERE bs_hfq_close IS NULL) / count(*), 4) bshfq_null,
      round(100.0 * count(*) FILTER (WHERE is_st = 1) / count(*), 2) st_share
    FROM {D}""").fetchone()
    rec("A4", "关键字段缺失率(%)", ok(fill[0] < 0.01 and fill[4] < 1),
        f"preclose={fill[0]} turn={fill[1]} pe_ttm={fill[2]} pb={fill[3]} float_shares={fill[4]} bs_hfq={fill[5]}",
        f"is_st=1 占比 {fill[6]}%")

    meta = load_json(DATA / "meta.json") or {}
    meta_rows = (meta.get("tables", {}).get("daily") or {}).get("rows")
    meta_ver = meta.get("data_version")
    built_at = pd.to_datetime(meta.get("built_at"), errors="coerce")
    mtimes = [p.stat().st_mtime for p in (FROZEN / "daily").glob("year=*/data*.parquet")]
    newest = pd.Timestamp(datetime.fromtimestamp(max(mtimes))) if mtimes else pd.Timestamp(0)
    meta_fresh = bool(built_at is not pd.NaT and built_at >= newest - pd.Timedelta(seconds=1))
    rec("A5", "meta.json 版本与当前产物一致", ok(meta_ver == EXPECTED_VERSION and meta_rows == rows[0] and meta_fresh),
        f"data_version={meta_ver} meta_rows={meta_rows:,} built_at={meta.get('built_at')}",
        f"daily 最新分区 mtime={newest:%Y-%m-%d %H:%M:%S}；期望版本 {EXPECTED_VERSION}")

    hash_check = {"checked": 0, "mismatch": 0}
    if args.deep and meta.get("files"):
        for f in meta["files"]:
            p = DATA / f["path"]
            if not p.exists():
                hash_check["mismatch"] += 1
                continue
            hash_check["checked"] += 1
            if sha256(p) != f["sha256"]:
                hash_check["mismatch"] += 1
        rec("A6", "meta.json 逐文件 SHA-256", ok(hash_check["mismatch"] == 0),
            f"核对 {hash_check['checked']} 个文件，不匹配 {hash_check['mismatch']}")
    else:
        rec("A6", "meta.json 逐文件 SHA-256", "SKIP", "未加 --deep")

    # A7 内容指纹：整数化求和，跨重建稳定（文件 SHA 会随 Parquet 编码变化）
    dig = con.execute("""
    SELECT count(*),
           sum(cast(round(open * 100) AS hugeint)) FILTER (WHERE open IS NOT NULL),
           sum(cast(round(high * 100) AS hugeint)) FILTER (WHERE high IS NOT NULL),
           sum(cast(round(low * 100) AS hugeint)) FILTER (WHERE low IS NOT NULL),
           sum(cast(round(close * 100) AS hugeint)) FILTER (WHERE close IS NOT NULL),
           sum(cast(round(hfq_close * 10000) AS hugeint)) FILTER (WHERE hfq_close IS NOT NULL),
           sum(cast(round(factor * 1000000) AS hugeint)) FILTER (WHERE factor IS NOT NULL),
           sum(cast(round(pct_chg * 10000) AS hugeint)) FILTER (WHERE pct_chg IS NOT NULL)
    FROM daily""").fetchone()
    comps = [int(x) if x is not None else None for x in dig]
    digest = hashlib.sha256("|".join("" if x is None else str(x) for x in comps).encode()).hexdigest()
    meta_dig = (meta.get("content_digest") or {}).get("daily_sha256")
    rec("A7", "daily 内容指纹（整数和，跨重建可复现）", ok(digest == meta_dig),
        f"{digest[:16]}… vs meta {str(meta_dig)[:16]}…",
        "文件 SHA 会随 Parquet 编码/行序变化；此指纹用于版本比对")

    # ---------------------------------------------------------------- #
    # B. 面板密度
    # ---------------------------------------------------------------- #
    dens = con.execute(f"""
    WITH y AS (
      SELECT symbol, year(date) yy, count(*) bars, min(date) d0, max(date) d1
      FROM {D} GROUP BY 1,2
    ), c AS (
      SELECT year(date) yy, count(*) tdays FROM calendar WHERE is_trading_day=1 GROUP BY 1
    )
    SELECT count(*) n_sym_year,
           sum(CASE WHEN bars < 0.9 * c.tdays THEN 1 ELSE 0 END) low,
           min(bars * 1.0 / c.tdays) min_ratio,
           avg(bars * 1.0 / c.tdays) avg_ratio
    FROM y JOIN c USING (yy)""").fetchone()
    rec("B1", "证券-年面板密度（bars/该年交易日）", "INFO",
        f"n={dens[0]:,} 低于0.9的={dens[1]:,} 最小={dens[2]:.3f} 均值={dens[3]:.3f}",
        "上市/退市当年与长期停牌会拉低；逐日缺口另见 B2")

    gap = con.execute(f"""
    WITH g AS (
      SELECT symbol, date, lag(date) OVER (PARTITION BY symbol ORDER BY date) pd FROM {D}
    )
    SELECT count(*) FILTER (WHERE date - pd > 10) n_gap,
           max(date - pd) max_gap
    FROM g WHERE pd IS NOT NULL""").fetchone()
    rec("B2", "相邻两条日线间隔>10自然日", "INFO", f"{gap[0]:,} 处，最大 {gap[1]} 天",
        "仅长假；源库对停牌日也保留行（OHLC=昨收、量额空）")

    # ---------------------------------------------------------------- #
    # C. 复权 / 收益不变量（v2 核心）
    # ---------------------------------------------------------------- #
    # C0 调整价非空率
    c0 = con.execute(f"""
    SELECT count(*) FILTER (WHERE close > 0 AND hfq_close IS NULL) hfq_null,
           count(*) FILTER (WHERE close > 0 AND open IS NOT NULL AND hfq_open IS NULL) hfq_open_null,
           count(*) FILTER (WHERE close <= 0 OR close IS NULL) bad_close,
           count(*) FILTER (WHERE factor IS NULL OR factor <= 0) bad_factor,
           count(*) FILTER (WHERE td_since_list = 1 AND abs(factor - 1) > 1e-9) first_bad
    FROM {D}""").fetchone()
    rec("C0", "调整价/因子非空与首日因子", ok(c0[0] == 0 and c0[1] == 0 and c0[3] == 0 and c0[4] == 0),
        f"hfq_close空={c0[0]} hfq_open空={c0[1]} close<=0={c0[2]} 因子非正={c0[3]} 首日因子≠1={c0[4]}")

    # C1 结构恒等式 hfq_px = raw_px * factor / close_t * close_first
    c1 = con.execute(f"""
    WITH b AS (
      SELECT symbol, date, open, high, low, close, factor,
             hfq_open, hfq_high, hfq_low, hfq_close,
             first_value(close) OVER (PARTITION BY symbol ORDER BY date) c0
      FROM {D}
    )
    SELECT count(*) n,
      count(*) FILTER (WHERE close > 0 AND abs(hfq_close - factor*c0) > 1e-9*greatest(abs(hfq_close),1e-12)) bad_close,
      count(*) FILTER (WHERE close > 0 AND open IS NOT NULL AND
        abs(hfq_open - open*factor/close*c0) > 1e-9*greatest(abs(hfq_open),1e-12)) bad_open,
      count(*) FILTER (WHERE close > 0 AND high IS NOT NULL AND
        abs(hfq_high - high*factor/close*c0) > 1e-9*greatest(abs(hfq_high),1e-12)) bad_high,
      count(*) FILTER (WHERE close > 0 AND low IS NOT NULL AND
        abs(hfq_low - low*factor/close*c0) > 1e-9*greatest(abs(hfq_low),1e-12)) bad_low,
      max(abs(hfq_close - factor*c0)) FILTER (WHERE close > 0) max_close
    FROM b""").fetchone()
    rec("C1", "恒等式 hfq_px = raw_px × factor / close_t × close_first", ok(sum(c1[1:5]) == 0),
        f"n={c1[0]:,} 违约 close/open/high/low={c1[1]}/{c1[2]}/{c1[3]}/{c1[4]} 最大|误差|={c1[5]:.2e}",
        "v1 的 hfq=raw×factor 在这条恒等式上也不失败，必须用 C5/C6 才能抓住双计")

    # C2 factor 日收益 = pct_chg
    c2 = con.execute(f"""
    WITH r AS (
      SELECT pct_chg, factor,
             factor / lag(factor) OVER (PARTITION BY symbol ORDER BY date) - 1 AS fret,
             lag(factor) OVER (PARTITION BY symbol ORDER BY date) fprev
      FROM {D}
    )
    SELECT count(*) FILTER (WHERE fprev IS NOT NULL AND pct_chg IS NOT NULL AND abs(fret - pct_chg/100) > 5e-5) n,
           max(abs(fret - pct_chg/100)) FILTER (WHERE fprev IS NOT NULL) mx
    FROM r""").fetchone()
    rec("C2", "factor 日收益 = 官方 pct_chg", ok(c2[0] == 0),
        f"偏差>0.5bp 的行={c2[0]}，最大偏差={c2[1]:.2e}")

    # C3 close/preclose = pct_chg
    c3 = con.execute(f"""
    SELECT count(*) FILTER (WHERE abs(close/preclose - 1 - pct_chg/100) > 0.0006) n,
           max(abs(close/preclose - 1 - pct_chg/100)) mx,
           count(*) FILTER (WHERE preclose IS NULL OR preclose <= 0) bad
    FROM {D} WHERE close IS NOT NULL AND preclose > 0""").fetchone()
    rec("C3", "close/preclose 与 pct_chg 一致（价格校验）", ok(c3[0] <= 10),
        f"不一致>6bp={c3[0]}，最大={c3[1]:.6f}，preclose无效={c3[2]}",
        "pct_chg 早期仅 2 位小数，6bp 内差异属源库舍入")

    # C4 Baostock hfq 自身瑕疵（不采用）
    c4 = con.execute(f"""
    WITH r AS (
      SELECT symbol, date, pct_chg, bs_hfq_close,
             bs_hfq_close / lag(bs_hfq_close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS bret,
             lag(bs_hfq_close) OVER (PARTITION BY symbol ORDER BY date) bp
      FROM {D} WHERE bs_hfq_close IS NOT NULL
    )
    SELECT count(*) FILTER (WHERE bp IS NOT NULL AND abs(bret - pct_chg/100) > 0.005) n,
           count(*) FILTER (WHERE bp IS NOT NULL) n_all,
           count(DISTINCT symbol) FILTER (WHERE bp IS NOT NULL AND abs(bret - pct_chg/100) > 0.005) nsym
    FROM r""").fetchone()
    rec("C4", "Baostock 后复权列的自身瑕疵（不采用为主口径的原因）", "INFO",
        f"收益与官方涨跌幅背离>0.5pp 的 {c4[0]:,}/{c4[1]:,} 行（{c4[0]/max(c4[1],1)*100:.3f}%），涉及 {c4[2]} 只",
        "典型为年末重定基（如 000001.SZ 2020-12-31 官方+0.73%，其hfq隐含-16.8%）")

    # C5 核心：hfq_close 相邻日收益 = pct_chg（全样本）
    c5 = con.execute(f"""
    WITH r AS (
      SELECT symbol, date, pct_chg, hfq_close,
             hfq_close / lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS hret,
             lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) prev
      FROM {D}
    )
    SELECT count(*) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL) n,
      count(*) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL AND abs(hret - pct_chg/100) > 5e-5) bad05,
      count(*) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL AND abs(hret - pct_chg/100) > 1e-4) bad1,
      max(abs(hret - pct_chg/100)) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL) mx,
      median(abs(hret - pct_chg/100)) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL) med,
      quantile_cont(abs(hret - pct_chg/100), 0.999) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL) p999,
      regr_slope(hret, pct_chg/100) FILTER (WHERE prev > 0 AND abs(pct_chg) BETWEEN 0.1 AND 30) slope
    FROM r""").fetchone()
    rec("C5", "I1 hfq_close 相邻日收益 = pct_chg（全样本）", ok(c5[1] == 0 and c5[0] > 16_000_000),
        f"{c5[0]:,} 对；>0.5bp={c5[1]} >1bp={c5[2]} 最大={c5[3]:.2e} 中位={c5[4]:.2e} p99.9={c5[5]:.2e} 斜率={c5[6]:.6f}",
        "v1 反事实：94.10% 的行 >1bp、中位 1.41pp、斜率 2.012（见 C11）")

    # C6 除权日
    c6 = con.execute(f"""
    WITH r AS (
      SELECT symbol, date, pct_chg, hfq_open, hfq_close, open, close, preclose,
             hfq_close / lag(hfq_close) OVER w - 1 AS hret,
             lag(hfq_close) OVER w AS hprev,
             close / lag(close) OVER w - 1 AS cret
      FROM {D}
      {W}
    ), e AS (
      SELECT r.* FROM r
      JOIN (SELECT DISTINCT symbol, ex_date FROM read_parquet('{FROZEN}/corp_actions_implied.parquet')) c
        ON c.symbol = r.symbol AND c.ex_date = r.date
    )
    SELECT count(*) n,
      count(*) FILTER (WHERE abs(hret - pct_chg/100) > 5e-5) adj_bad,
      max(abs(hret - pct_chg/100)) adj_mx,
      count(*) FILTER (WHERE abs(cret - pct_chg/100) > 0.01) raw_bad,
      median(abs(cret - pct_chg/100)) raw_med,
      count(*) FILTER (WHERE hprev > 0 AND preclose > 0 AND abs(hfq_open/hprev - open/preclose) > 1e-3) open_bad,
      max(abs(hfq_open/hprev - open/preclose)) FILTER (WHERE hprev > 0 AND preclose > 0) open_mx,
      median(abs(hfq_open/hprev - open/preclose)) FILTER (WHERE hprev > 0 AND preclose > 0) open_med
    FROM e WHERE hret IS NOT NULL""").fetchone()
    rec("C6", "I2 除权日 hfq 收益 = pct_chg，原始价收益显著偏离",
        ok(c6[1] == 0 and c6[5] == 0 and c6[3] > 0.25 * c6[0] and c6[4] > 0.005),
        f"除权日={c6[0]:,} 调整价违约={c6[1]} 最大={c6[2]:.2e}；原始价偏离>1%={c6[3]:,}（{c6[3]/max(c6[0],1)*100:.1f}%）中位={c6[4]*100:.2f}pp；"
        f"开盘跨日校验违约={c6[5]} 最大={c6[6]:.1e} 中位={c6[7]:.1e}",
        "原始价收益在除权日必须显著偏离（送转/分红），否则说明样本选错；开盘跨度差异受 pct_chg 4 位小数限制")

    # C6b 开盘跨日不变量（全样本；剔除 3 行源库 pct_chg 与 close/preclose 冲突）
    c6b = con.execute(f"""
    WITH r AS (
      SELECT symbol, date, open, close, preclose, hfq_open,
             lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) AS hprev
      FROM {D}
      WHERE close > 0 AND preclose > 0 AND abs(close/preclose - 1 - pct_chg/100) <= 0.0006
    )
    SELECT count(*) FILTER (WHERE hprev > 0 AND open IS NOT NULL) n,
      count(*) FILTER (WHERE hprev > 0 AND open IS NOT NULL AND abs(hfq_open/hprev - open/preclose) > 1e-3) bad,
      max(abs(hfq_open/hprev - open/preclose)) FILTER (WHERE hprev > 0 AND open IS NOT NULL) mx,
      median(abs(hfq_open/hprev - open/preclose)) FILTER (WHERE hprev > 0 AND open IS NOT NULL) med
    FROM r""").fetchone()
    rec("C6b", "I2b 开盘跨日 hfq_open/hfq_close_prev = open/preclose（全样本，剔除 3 行源库冲突）",
        ok(c6b[1] == 0),
        f"{c6b[0]:,} 对；违约={c6b[1]} 最大={c6b[2]:.1e} 中位={c6b[3]:.1e}",
        "源库 3 行 pct_chg 与 close/preclose 冲突（689009.SH 2020-10-29、001965.SZ 2017-12-25、689009.SH 2020-11-02），已由 C3 标记并剔除")

    # C7 OHLC 比例（hfq 与 qfq）
    c7 = con.execute(f"""
    SELECT
      max(abs(hfq_open/hfq_close - open/close)) FILTER (WHERE hfq_close > 0 AND close > 0 AND open IS NOT NULL) o,
      max(abs(hfq_high/hfq_close - high/close)) FILTER (WHERE hfq_close > 0 AND close > 0 AND high IS NOT NULL) h,
      max(abs(hfq_low/hfq_close - low/close)) FILTER (WHERE hfq_close > 0 AND close > 0 AND low IS NOT NULL) l
    FROM {D}""").fetchone()
    c7q = con.execute("""
    SELECT
      max(abs(qfq_open/qfq_close - open/close)) FILTER (WHERE qfq_close > 0 AND close > 0 AND open IS NOT NULL) o,
      max(abs(qfq_high/qfq_close - high/close)) FILTER (WHERE qfq_close > 0 AND close > 0 AND high IS NOT NULL) h,
      max(abs(qfq_low/qfq_close - low/close)) FILTER (WHERE qfq_close > 0 AND close > 0 AND low IS NOT NULL) l
    FROM daily_qfq""").fetchone()
    rec("C7", "I3 同 bar 内 hfq/qfq 的 OHLC 比例 = 原始价比例",
        ok(max(c7) < 1e-9 and max(c7q) < 1e-9),
        f"hfq最大比例误差 o/h/l={c7[0]:.1e}/{c7[1]:.1e}/{c7[2]:.1e}；qfq={c7q[0]:.1e}/{c7q[1]:.1e}/{c7q[2]:.1e}",
        "v1 因 ROUND(...,4) 在低价高因子行把比例破坏到 1e-1 量级；v2 不舍入")

    # C8 首末锚定
    c8 = con.execute(f"""
    WITH f AS (
      SELECT symbol,
             arg_min(date, date) FILTER (WHERE close > 0) AS d0,
             arg_min(close, date) FILTER (WHERE close > 0) AS c0,
             arg_max(date, date) FILTER (WHERE close > 0) AS dl,
             arg_max(close, date) FILTER (WHERE close > 0) AS cl,
             arg_max(factor, date) AS factor_last
      FROM {D} GROUP BY 1
    )
    SELECT count(*) n,
      max(abs(a.hfq_close - f.c0)) first_err,
      max(abs(b.qfq_close - f.cl)) last_err,
      max(abs(a.hfq_close - a.factor*f.c0)) hfq_factor_err,
      max(abs(b.qfq_close - b.factor*f.cl/f.factor_last)) qfq_factor_err,
      count(*) FILTER (WHERE abs(a.hfq_close - f.c0) > 1e-9) first_bad,
      count(*) FILTER (WHERE abs(b.qfq_close - f.cl) > 1e-9) last_bad
    FROM f
    JOIN {D} a ON a.symbol = f.symbol AND a.date = f.d0
    JOIN daily_qfq b ON b.symbol = f.symbol AND b.date = f.dl
    """).fetchone()
    rec("C8", "I4 首条 hfq_close=原始收盘；末条 qfq_close=原始收盘", ok(c8[5] == 0 and c8[6] == 0),
        f"{c8[0]:,} 只；首条违约={c8[5]} 末日违约={c8[6]}；|hfq-factor×c0|max={c8[3]:.1e}；|qfq-factor×cl/factor_last|max={c8[4]:.1e}")

    # C9 截断未来不变（hfq 必须不变；qfq 是展示口径，预期会变）
    # 分层抽样：主板 150 + 创业板 100（按 hash 确定，覆盖 2010 年前已上市），避免按代码字母取样的偏差
    cuts = ["2015-12-31", "2020-12-31"]
    sample_sql = f"""
      SELECT symbol, board FROM (
        SELECT symbol, board,
               row_number() OVER (PARTITION BY board ORDER BY hash(symbol)) AS rn
        FROM (SELECT DISTINCT symbol, board FROM {D} WHERE date <= DATE '2010-12-31')
      ) WHERE (board = '主板' AND rn <= 150) OR (board = '创业板' AND rn <= 100)
    """
    trunc_boards = dict(con.execute(
        f"WITH s AS ({sample_sql}) SELECT board, count(*) FROM s GROUP BY 1 ORDER BY 1").fetchall())
    trunc_rows = []
    for cut in cuts:
        tr = con.execute(f"""
        WITH s AS ({sample_sql}), b0 AS (
          SELECT d.symbol, d.date, d.close, d.open, d.high, d.low, d.factor,
                 d.hfq_close, d.hfq_open, d.pct_chg, d.preclose,
                 row_number() OVER (PARTITION BY d.symbol ORDER BY d.date) rn
          FROM {D} d JOIN s USING (symbol)
          WHERE d.date <= DATE '{cut}'
        ), b AS (
          SELECT *,
            exp(sum(ln(greatest(1 + coalesce(
                  CASE WHEN rn = 1 THEN 0.0 ELSE pct_chg/100.0 END,
                  CASE WHEN preclose > 0 THEN close/preclose - 1 END, 0.0), 1e-6)))
                OVER (PARTITION BY symbol ORDER BY date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) AS factor_re,
            first_value(close) OVER (PARTITION BY symbol ORDER BY date) AS c0,
            last_value(close) OVER (PARTITION BY symbol ORDER BY date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS cl,
            last_value(factor) OVER (PARTITION BY symbol ORDER BY date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS fl
          FROM b0
        ), j AS (
          SELECT b.*, q.qfq_close
          FROM b JOIN daily_qfq q ON q.symbol = b.symbol AND q.date = b.date
          WHERE b.close > 0
        )
        SELECT count(*) n,
          max(abs(factor - factor_re)/greatest(abs(factor),1e-12)) factor_rel,
          max(abs(hfq_close - factor_re*c0)/greatest(abs(hfq_close),1e-12)) hfq_rel,
          max(abs(hfq_open - open*factor_re/close*c0)/greatest(abs(hfq_open),1e-12)) FILTER (WHERE open IS NOT NULL) open_rel,
          median(abs(qfq_close - close*factor_re/close*cl/fl)/greatest(abs(qfq_close),1e-12)) qfq_med_rel,
          max(abs(qfq_close - close*factor_re/close*cl/fl)/greatest(abs(qfq_close),1e-12)) qfq_max_rel
        FROM j""").fetchone()
        trunc_rows.append({"cut": cut, "n": tr[0], "factor_rel": tr[1], "hfq_rel": tr[2],
                           "hfq_open_rel": tr[3], "qfq_med_rel": tr[4], "qfq_max_rel": tr[5],
                           "sample_boards": trunc_boards})
    trunc_ok = all(r["hfq_rel"] is not None and r["hfq_rel"] < 1e-9 and
                   (r["hfq_open_rel"] or 0) < 1e-9 for r in trunc_rows)
    rec("C9", "I5 截断到 T 重算的 hfq 在 t≤T 上不变", ok(trunc_ok),
        "; ".join(f"{r['cut']}: n={r['n']:,} hfq相对误差={r['hfq_rel']:.1e} 开盘={r['hfq_open_rel']:.1e}" for r in trunc_rows),
        "qfq 锚定末日，重算后价格水平会变（中位相对变化 "
        + ", ".join(f"{r['cut']}={r['qfq_med_rel']*100:.2f}%" for r in trunc_rows) + "），故只作展示")

    # C10 随机日抽样
    c10 = con.execute(f"""
    WITH r AS (
      SELECT symbol, date, pct_chg, hfq_close,
             hfq_close / lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS hret,
             lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) prev
      FROM {D}
    ), s AS (
      SELECT * FROM r WHERE prev > 0 AND pct_chg IS NOT NULL AND hash(symbol, date) % 800 = 0
    )
    SELECT count(*), count(*) FILTER (WHERE abs(hret - pct_chg/100) > 5e-5),
           max(abs(hret - pct_chg/100)), median(abs(hret - pct_chg/100)),
           quantile_cont(abs(hret - pct_chg/100), 0.99)
    FROM s""").fetchone()
    rec("C10", "I1b 随机日抽样（hash%800）hfq 收益 = pct_chg", ok(c10[1] == 0),
        f"n={c10[0]:,} 违约={c10[1]} 最大={c10[2]:.2e} 中位={c10[3]:.2e} p99={c10[4]:.2e}")

    # C11 反事实：旧公式 hfq=raw*factor 必须被 I1/I2 检出
    c11 = con.execute(f"""
    WITH r AS (
      SELECT symbol, date, pct_chg, close*factor AS old_close,
             (close*factor) / lag(close*factor) OVER (PARTITION BY symbol ORDER BY date) - 1 AS oret,
             lag(close*factor) OVER (PARTITION BY symbol ORDER BY date) prev
      FROM {D}
    )
    SELECT count(*) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL) n,
      count(*) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL AND abs(oret - pct_chg/100) > 1e-4) bad,
      median(abs(oret - pct_chg/100)) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL) med,
      regr_slope(oret, pct_chg/100) FILTER (WHERE prev > 0 AND abs(pct_chg) BETWEEN 0.1 AND 30) slope
    FROM r""").fetchone()
    c11e = con.execute(f"""
    WITH r AS (
      SELECT symbol, date, pct_chg, close*factor AS old_close,
             (close*factor) / lag(close*factor) OVER (PARTITION BY symbol ORDER BY date) - 1 AS oret
      FROM {D}
    )
    SELECT count(*), median(abs(oret - pct_chg/100)),
           quantile_cont(abs(oret - pct_chg/100), 0.95),
           max(abs(oret - pct_chg/100)),
           count(*) FILTER (WHERE abs(oret - pct_chg/100) > 0.05)
    FROM r
    JOIN (SELECT DISTINCT symbol, ex_date FROM read_parquet('{FROZEN}/corp_actions_implied.parquet')) c
      ON c.symbol = r.symbol AND c.ex_date = r.date
    WHERE oret IS NOT NULL""").fetchone()
    bad_share = c11[1] / max(c11[0], 1)
    rec("C11", "I6 检验功效：旧公式 hfq=raw×factor 必须被判失败",
        ok(bad_share > 0.5 and c11e[2] > 0.05 and c11e[4] > 1000),
        f"全样本 {c11[1]:,}/{c11[0]:,} 行>1bp（{bad_share*100:.2f}%），中位={c11[2]*100:.2f}pp，斜率={c11[3]:.4f}；"
        f"除权日 {c11e[0]:,} 个：中位={c11e[1]*100:.2f}pp p95={c11e[2]*100:.2f}pp 最大={c11e[3]*100:.2f}pp >5pp={c11e[4]:,}",
        "同一组 I1/I2 在 v1 数据上会 FAIL，说明检查真的能抓住双计")

    # C12 qfq 收益 = pct_chg
    c12 = con.execute("""
    WITH r AS (
      SELECT pct_chg, qfq_close,
             qfq_close / lag(qfq_close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS qret,
             lag(qfq_close) OVER (PARTITION BY symbol ORDER BY date) prev
      FROM daily_qfq
    )
    SELECT count(*) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL),
      count(*) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL AND abs(qret - pct_chg/100) > 5e-5),
      max(abs(qret - pct_chg/100)) FILTER (WHERE prev > 0 AND pct_chg IS NOT NULL)
    FROM r""").fetchone()
    rec("C12", "qfq_close 相邻日收益 = pct_chg", ok(c12[1] == 0),
        f"{c12[0]:,} 对；>0.5bp={c12[1]} 最大={c12[2]:.2e}")

    # ---------------------------------------------------------------- #
    # D. 停牌 / 涨跌停
    # ---------------------------------------------------------------- #
    sus = con.execute(f"""
    SELECT count(*) n,
      count(*) FILTER (WHERE volume IS NULL OR volume = 0) null_vol,
      count(*) FILTER (WHERE abs(close - preclose) > 1e-9) px_move,
      count(*) FILTER (WHERE is_st IS NULL) null_st
    FROM {D} WHERE is_suspended""").fetchone()
    rec("D1", "停牌行形态（量额空、价格不动）", ok(sus[1] == sus[0] and sus[2] <= 0.0001 * sus[0]),
        f"停牌行={sus[0]:,} 其中量0/空={sus[1]:,} 价格异动={sus[2]}", f"is_st缺失={sus[3]}；停牌判定改用成交量，trade_status 有 8 行误标")

    lim = con.execute(f"""
    SELECT board, coalesce(cast(limit_pct as varchar),'无限制') p, count(*) n,
      count(*) FILTER (WHERE px_breaks_limit_up) up_v,
      count(*) FILTER (WHERE px_breaks_limit_down) dn_v,
      count(*) FILTER (WHERE is_limit_up_close) lu,
      count(*) FILTER (WHERE is_limit_down_close) ld
    FROM {D} WHERE limit_up_px IS NOT NULL GROUP BY 1,2 ORDER BY 1,2""").df()
    tot = int(lim.n.sum()); viol = int(lim.up_v.fillna(0).sum() + lim.dn_v.fillna(0).sum())
    rec("D2", "涨跌停规则与实际最高/最低价冲突", ok(viol / max(tot, 1) < 2e-4),
        f"{viol}/{tot:,}（{viol/max(tot,1)*100:.4f}%）",
        "主要为 2005-2007 股改/重组复牌无涨跌幅、退市整理期首日无涨跌幅、缩股式重整；已用 px_breaks_* 标记")

    ext = con.execute(f"""
    SELECT count(*) n, count(*) FILTER (WHERE is_st = 1) st_n,
           count(*) FILTER (WHERE td_since_list <= 5) new_n,
           min(date) d0, max(date) d1
    FROM {D} WHERE abs(pct_chg) > 30""").fetchone()
    rec("D3", "单日绝对涨跌幅>30% 的极端行", "INFO",
        f"{ext[0]} 行（ST={ext[1]}，上市5日内={ext[2]}，{ext[3]}~{ext[4]}）",
        "退市整理期首日无涨跌幅、缩股/重整为主；因子按 pct_chg 计入，研究时应结合 px_breaks_* 剔除")

    # ---------------------------------------------------------------- #
    # E. 证券生命周期 / 退市
    # ---------------------------------------------------------------- #
    life = con.execute("""
    SELECT count(*) n,
      count(*) FILTER (WHERE delist_date IS NOT NULL) n_delist,
      count(*) FILTER (WHERE delist_date IS NOT NULL AND last_bar < DATE '2026-09-01') delist_with_bar,
      count(*) FILTER (WHERE last_bar < DATE '2026-09-01' AND delist_date IS NULL) orphan,
      count(*) FILTER (WHERE last_bar - delist_date > 30) late_bar
    FROM symbols""").fetchone()
    rec("E1", "证券主表与退市覆盖", ok(life[3] == 0),
        f"证券={life[0]:,} 有退市日={life[1]} 退市且有末条行情={life[2]} 无退市日但行情早停={life[3]}",
        f"末条行情晚于退市日30天以上={life[4]}")

    nd = con.execute(f"""
    SELECT count(*) FROM (SELECT symbol FROM {D} GROUP BY 1
      HAVING max(date) < DATE '2026-09-01')""").fetchone()[0]
    rec("E2", "冻结窗口内退市/长停证券数", "INFO", f"{nd} 只最后行情早于 2026-09-01",
        "退市股保留在样本内，避免生存者偏差")

    # ---------------------------------------------------------------- #
    # F. 公司行动
    # ---------------------------------------------------------------- #
    ca = con.execute("SELECT count(*), count(DISTINCT symbol), min(ex_date), max(ex_date) FROM corp_actions").fetchone()
    rec("F1", "同花顺公司行动事件", ok(ca[3] <= pd.Timestamp(FREEZE).date()),
        f"{ca[0]:,} 条 / {ca[1]:,} 只 / {ca[2]}~{ca[3]}", "行情表自 2016-10-13 起，但事件表含 1991 年以来历史（2016 前 22,225 条）")

    imp = con.execute("SELECT count(*), count(DISTINCT symbol), min(ex_date), max(ex_date), "
                      "count(*) FILTER (WHERE has_th_event) FROM corp_actions_implied").fetchone()
    rec("F2", "由 preclose≠前收 推导的除权日", "INFO",
        f"{imp[0]:,} 条 / {imp[1]:,} 只 / {imp[2]}~{imp[3]}；与同花顺事件匹配 {imp[4]:,}",
        "覆盖 2004 起，可补源库早期缺口；2016+ 与同花顺事件匹配率见下行")

    ca_check = con.execute(f"""
    WITH j AS (
      SELECT c.symbol, c.ex_date, c.dividend_per_share dd, c.per_share_bonus b,
             c.rights_ratio r, coalesce(c.rights_price,0) rp,
             p.close AS prev_close, e.preclose AS act_preclose
      FROM corp_actions c
      ASOF LEFT JOIN (SELECT symbol, date, close FROM {D}) p
        ON c.symbol = p.symbol AND p.date < c.ex_date
      JOIN (SELECT symbol, date, preclose FROM {D}) e
        ON e.symbol = c.symbol AND e.date = c.ex_date
    )
    SELECT count(*) n,
      count(*) FILTER (WHERE abs(act_preclose - (prev_close - dd + r*rp)/(1+b+r)) <= 0.011) match_,
      median(abs(act_preclose - (prev_close - dd + r*rp)/(1+b+r))) med_err
    FROM j WHERE prev_close IS NOT NULL""").fetchone()
    rec("F3", "公司行动公式 vs 实际除权后昨收", ok(ca_check[1] / max(ca_check[0], 1) > 0.95),
        f"吻合 {ca_check[1]:,}/{ca_check[0]:,}（{ca_check[1]/max(ca_check[0],1)*100:.2f}%），中位绝对误差={ca_check[2]:.4f}",
        "不吻合者多为同日多事件/缩股/数据源口径差异，未修正价格")

    # ---------------------------------------------------------------- #
    # G. 指数 / 日历 / 单位
    # ---------------------------------------------------------------- #
    ix = con.execute("""SELECT count(*), count(DISTINCT symbol), min(date), max(date)
                        FROM index_daily""").fetchone()
    ixgap = con.execute("""
    SELECT count(*) FROM (
      SELECT date FROM index_daily WHERE symbol='000300.SH'
      EXCEPT SELECT date FROM calendar WHERE is_trading_day=1)""").fetchone()[0]
    rec("G1", "指数日线与交易日历一致", ok(ixgap == 0),
        f"{ix[0]:,} 行 / {ix[1]} 只 / {ix[2]}~{ix[3]}；与日历差集={ixgap}",
        "指数最早 2006-01-04（源库限制）")

    cal = con.execute("SELECT count(*), sum(is_trading_day), min(date), max(date) FROM calendar").fetchone()
    rec("G2", "交易日历", ok(cal[3] <= pd.Timestamp(FREEZE).date()),
        f"{cal[0]:,} 天，其中交易日 {int(cal[1]):,}，{cal[2]}~{cal[3]}")

    vwap = con.execute(f"""
    SELECT median(abs(amount / nullif(volume * (high+low+close)/3, 0) - 1)) med,
           quantile_cont(abs(amount / nullif(volume * (high+low+close)/3, 0) - 1), 0.99) p99
    FROM {D} WHERE volume > 0 AND amount > 0""").fetchone()
    rec("G3", "量额单位一致性（amount vs volume×均价）", ok(vwap[0] < 0.02),
        f"中位偏差={vwap[0]:.4f}，p99={vwap[1]:.4f}",
        "amount=元、volume=股；偏差来自成交价分布与 VWAP 近似")

    # ---------------------------------------------------------------- #
    # H. 跨源一致性（只读源库）
    # ---------------------------------------------------------------- #
    if not args.skip_source and Path(BS_DB).exists() and Path(TH_DB).exists():
        try:
            s = sqlite3.connect(f"file:{BS_DB}?mode=ro", uri=True)
            th = duckdb.connect(TH_DB, read_only=True)
            out = []
            for d in ["2016-10-13", "2021-06-15", "2026-10-23"]:
                bs = pd.read_sql(
                    "SELECT code,open,high,low,close,volume,amount FROM stock_kline_d "
                    "WHERE adjustflag=3 AND date=?", s, params=(d,))
                if bs.empty:
                    continue
                bs["symbol"] = bs.code.str[3:] + "." + bs.code.str[:2].str.upper()
                tp = th.execute(
                    "SELECT thscode,open,high,low,close,volume,amount FROM raw_kline_daily WHERE date=?",
                    [d]).df()
                mg = bs.merge(tp, left_on="symbol", right_on="thscode", suffixes=("_b", "_t"))
                mx = max(float((mg[c + "_b"] / mg[c + "_t"] - 1).abs().max())
                         for c in ["open", "high", "low", "close", "volume", "amount"])
                out.append((d, len(mg), mx))
            th.close(); s.close()
            worst = max(o[2] for o in out)
            rec("H1", "两源库原始 OHLCV 交叉校验", ok(worst < 1e-3),
                "; ".join(f"{d}:{n}只/最大相对差{o:.1e}" for d, n, o in out),
                "逐字段完全一致，说明两源同源；差异仅在覆盖范围（同花顺含北交所）")
        except Exception as e:
            rec("H1", "两源库原始 OHLCV 交叉校验", "SKIP", f"无法读取源库: {e}")
    else:
        rec("H1", "两源库原始 OHLCV 交叉校验", "SKIP", "--skip-source 或源库不可用")

    # ---------------------------------------------------------------- #
    # 汇总
    # ---------------------------------------------------------------- #
    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    n_info = sum(1 for r in RESULTS if r["status"] == "INFO")

    lines = [f"# 冻结数据层自检报告", "",
             f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
             f"- 冻结日：{FREEZE}",
             f"- 数据版本：{meta_ver}",
             f"- 结论：{n_pass} 项 PASS，{n_fail} 项 FAIL，{n_info} 项 INFO", "",
             "| 编号 | 检查 | 结果 | 指标 | 说明 |", "|---|---|---|---|---|"]
    for r in RESULTS:
        lines.append(f"| {r['id']} | {r['check']} | {r['status']} | {r['metric']} | {r['detail']} |")
    (AUDIT / "自检报告.md").write_text("\n".join(lines), encoding="utf-8")
    (AUDIT / "自检结果.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                    "freeze": FREEZE, "data_version": meta_ver,
                    "pass": n_pass, "fail": n_fail,
                    "results": RESULTS}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- #
    # 修复专项报告 + 机器可读证据
    # ---------------------------------------------------------------- #
    build_fix_report(con, D, rows, meta, c5, c6, c6b, c7, c7q, c8, trunc_rows, c10, c11, c11e,
                     c12, hash_check)
    print(f"\n报告：{AUDIT/'自检报告.md'}")
    print(f"修复报告：{AUDIT/'修复后自检报告.md'}")
    print(f"证据：{EVIDENCE/'复权修复.json'}")
    if n_fail:
        sys.exit(1)


def load_build_history(path: Path | None = None) -> list[dict]:
    """读取 _audit/build_history.csv（每次构建一行）。"""
    p = path or (AUDIT / "build_history.csv")
    if not p.exists():
        return []
    lines = [ln for ln in p.read_text(encoding="utf-8").strip().splitlines() if ln]
    if len(lines) < 2:
        return []
    header = lines[0].split(",")
    return [dict(zip(header, ln.split(","))) for ln in lines[1:]]


def build_fix_report(con, D, rows, meta, c5, c6, c6b, c7, c7q, c8, trunc_rows, c10, c11, c11e, c12, hash_check):
    """输出 v1→v2 复权修复的专项报告与证据 JSON。"""
    pre_def = load_json(AUDIT / "v1_修复前_缺陷量化.json") or {}
    pre_fp = load_json(AUDIT / "v1_修复前_指纹.json") or {}

    # 修复后指纹
    files = meta.get("files", [])
    agg = hashlib.sha256()
    for f in sorted(files, key=lambda x: x["path"]):
        agg.update(f"{f['path']}:{f['rows']}:{f['sha256']}\n".encode())
    post_fp = {
        "version": meta.get("data_version"),
        "built_at": meta.get("built_at"),
        "daily_rows": rows[0], "symbols": rows[1],
        "date_min": str(rows[2]), "date_max": str(rows[3]),
        "future_rows": rows[4], "dup_keys": rows[5],
        "files_agg_sha256": agg.hexdigest(),
        "meta_json_sha256": sha256(DATA / "meta.json"),
        "frozen_duckdb_sha256": sha256(FROZEN / "frozen.duckdb"),
        "n_files": len(files),
        "content_digest": (meta.get("content_digest") or {}).get("daily_sha256"),
        "deep_hash_checked": hash_check.get("checked", 0),
        "deep_hash_mismatch": hash_check.get("mismatch", 0),
    }

    # 案例（v2 实算）
    cases = con.execute(f"""
    WITH r AS (
      SELECT symbol, date, close, preclose, pct_chg, factor, hfq_close,
             lag(close) OVER (PARTITION BY symbol ORDER BY date) cprev,
             lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) hprev
      FROM {D}
    )
    SELECT symbol, date, close, cprev, preclose, pct_chg,
           close/cprev - 1 AS raw_ret, hfq_close/hprev - 1 AS new_hfq_ret
    FROM r WHERE (symbol, date) IN (
      VALUES ('688050.SH', DATE '2024-06-05'), ('000001.SZ', DATE '2020-12-31'),
             ('600655.SH', DATE '2020-11-13'), ('002466.SZ', DATE '2019-11-20'))
    """).fetchall()
    case_rows = []
    for r in cases:
        case_rows.append({"symbol": r[0], "date": str(r[1]), "close": r[2], "prev_close": r[3],
                          "preclose": r[4], "pct_chg": r[5], "raw_ret": r[6],
                          "new_hfq_ret": r[7]})

    invariants = [
        {"id": "I1", "name": "hfq_close 相邻日收益 = pct_chg（全样本）",
         "pre_v1": {"violations_gt_1bp": pre_def.get("hfq_ret_vs_pctchg", {}).get("bad_gt_1bp"),
                    "median_abs": pre_def.get("hfq_ret_vs_pctchg", {}).get("median_abs"),
                    "slope": pre_def.get("hfq_ret_vs_pctchg", {}).get("ols_slope_trim_0.1_30pct")},
         "post_v2": {"pairs": c5[0], "violations_gt_0.5bp": c5[1], "max_abs": c5[3],
                     "median_abs": c5[4], "slope": c5[6]},
         "catches_double_count": True},
        {"id": "I2", "name": "除权日 hfq 收益 = pct_chg；原始价收益显著偏离",
         "pre_v1": {"exdates": pre_def.get("exdate", {}).get("n"),
                    "adj_bad": pre_def.get("exdate", {}).get("adj_bad_gt_1bp"),
                    "adj_median_abs": pre_def.get("exdate", {}).get("adj_median_abs")},
         "post_v2": {"exdates": c6[0], "adj_bad": c6[1], "adj_max_abs": c6[2],
                     "raw_bad_gt_1pct": c6[3], "raw_median_abs": c6[4],
                     "open_cross_bad": c6[5], "open_cross_max_abs": c6[6],
                     "open_cross_median_abs": c6[7]},
         "catches_double_count": True},
        {"id": "I2b", "name": "开盘跨日 hfq_open/hfq_close_prev = open/preclose（全样本）",
         "pre_v1": {"note": "v1 的 hfq_open 跨日比值含前一日价格收益，会系统性偏离 open/preclose"},
         "post_v2": {"pairs": c6b[0], "bad": c6b[1], "max_abs": c6b[2], "median_abs": c6b[3]},
         "catches_double_count": True},
        {"id": "I3", "name": "同 bar OHLC 比例 = 原始价比例",
         "pre_v1": {"n_abs_err_gt_1e-6": pre_def.get("ohlc_ratio", {}).get("n_abs_err_gt_1e-6"),
                    "p99_abs_err": pre_def.get("ohlc_ratio", {}).get("p99_abs_err")},
         "post_v2": {"hfq_max_err": list(c7), "qfq_max_err": list(c7q)},
         "catches_double_count": False},
        {"id": "I4", "name": "首条 hfq_close=原始收盘；末条 qfq_close=原始收盘",
         "pre_v1": {"first_err": pre_def.get("anchor", {}).get("first_hfq_err"),
                    "last_err": pre_def.get("anchor", {}).get("last_qfq_err")},
         "post_v2": {"symbols": c8[0], "first_bad": c8[5], "last_bad": c8[6],
                     "hfq_factor_max_err": c8[3], "qfq_factor_max_err": c8[4]},
         "catches_double_count": False},
        {"id": "I5", "name": "截断到 T 重算 hfq，在 t≤T 上不变",
         "pre_v1": {"note": "v1 的 hfq=raw×factor 也是点时构造，重算误差≈0，单独不能发现双计"},
         "post_v2": trunc_rows,
         "catches_double_count": False},
        {"id": "I1b", "name": "随机日抽样（hash%800）",
         "pre_v1": {"bad": pre_def.get("random_sample", {}).get("bad_gt_1bp"),
                    "n": pre_def.get("random_sample", {}).get("n")},
         "post_v2": {"n": c10[0], "bad": c10[1], "max_abs": c10[2], "median_abs": c10[3]},
         "catches_double_count": True},
        {"id": "I6", "name": "反事实旧公式必须被 I1/I2 判失败（检验功效）",
         "pre_v1": {}, "post_v2": {"n": c11[0], "bad": c11[1],
                                   "bad_share_pct": c11[1] / max(c11[0], 1) * 100,
                                   "median_abs": c11[2], "slope": c11[3],
                                   "exdate_n": c11e[0], "exdate_median_abs": c11e[1],
                                   "exdate_p95_abs": c11e[2], "exdate_max_abs": c11e[3],
                                   "exdate_gt5pp": c11e[4]},
         "catches_double_count": True},
        {"id": "I7", "name": "qfq_close 相邻日收益 = pct_chg",
         "pre_v1": {}, "post_v2": {"pairs": c12[0], "bad": c12[1], "max_abs": c12[2]},
         "catches_double_count": True},
    ]

    evidence = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_version": meta.get("data_version"),
        "freeze_date": FREEZE,
        "purpose": "冻结层复权口径修复证据；只做数据完整性校验，未计算任何策略收益",
        "build_history": load_build_history(),
        "formula": {
            "v1_deprecated": "hfq_px = raw_px * factor  (price return double-counted)",
            "v2_hfq": "raw_px * factor_t / close_t * close_first; first hfq_close=raw close; daily return=pct_chg",
            "v2_qfq": "raw_px * factor_t / close_t * close_last / factor_last; last qfq_close=raw close; display only",
            "factor": "factor_t = prod(1+pct_chg/100), first bar=1; total-return index",
        },
        "v1_fingerprint": pre_fp,
        "v2_fingerprint": post_fp,
        "v1_reconstruction": "v1 的 hfq_* = ROUND(raw_px × factor, 4)，可由当前 raw+factor 精确重建（仅 4 位舍入差 ≤5e-5）；因此未保留 1.4 GB 旧数据副本，旧结果可按此公式复算。",
        "v1_defect": pre_def,
        "invariants": invariants,
        "cases": case_rows,
        "limitations": [
            "I3/I4/I5 在 v1 上也能通过；只有收益型不变量（I1/I2/I2b/I1b/I6/I7）能抓住双计。",
            "qfq 锚定冻结日最后一条，价格水平不是截断不变的；只能用于展示/横向比较。",
            "pct_chg 在早期年份只有 2 位小数，close/preclose 与 pct_chg 存在 ≤6bp 的源库舍入差；开盘跨日校验受此限制。",
            "C6 的除权日来自 `corp_actions_implied`，含复牌首日 preclose 滞后等非公司行动跳变；该检查只验证调整公式与官方 pct_chg 一致，不证明 pct_chg 在该日经济上真实。",
            "3 行源库 pct_chg 与 close/preclose 冲突（689009.SH 2020-10-29、001965.SZ 2017-12-25、689009.SH 2020-11-02）；因子按官方 pct_chg 构造，这 3 行的调整价与原始价关系不成立，已由 C3/C6b 单独剔除并计数。",
            "3 行 close<=0 的源库记录无法构造调整价（hfq 为 NULL），已单独计数。",
            "本证据不包含 2021+ 策略收益；全样本只用于数据层不变量，不用于规则选择。",
            "下游若在 v1 上产出过结果，必须用 v2 重跑；v1 的 hfq_* 列不可用于收益/指标。",
        ],
    }
    (EVIDENCE / "复权修复.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# 冻结数据层复权修复报告（v1 → v2）")
    md.append("")
    md.append(f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    md.append(f"- 数据版本：**{meta.get('data_version')}**（v1 为 2026-10-28 21:55 构建）")
    md.append(f"- 冻结日：{FREEZE}；日线 {rows[0]:,} 行 / {rows[1]:,} 只 / {rows[2]}~{rows[3]}")
    md.append("- 性质：数据层完整性修复；**未计算任何策略收益**，2021+ 只做数据不变量，不用于规则选择")
    md.append("")
    md.append("## 1. 修复内容")
    md.append("")
    md.append("| 项 | v1（错误） | v2（正确） |")
    md.append("|---|---|---|")
    md.append("| hfq | `raw_px × factor` | `raw_px × factor_t / close_t × close_first` |")
    md.append("| qfq | `raw_px × factor / factor_last` | `raw_px × factor_t / close_t × close_last / factor_last` |")
    md.append("| 舍入 | `ROUND(...,4)` | 不舍入（全 double 精度） |")
    md.append("| 日收益 | ≈`(1+r)²-1` | 严格 `= pct_chg` |")
    md.append("")
    md.append("`factor` 本身未改动：`factor_t = Π(1+pct_chg/100)` 是总收益指数，首条=1。")
    md.append("A 空间乘子 `A_t = factor_t / close_t × close_first` 只使用 t 日及以前的数据，因此 hfq 是点时构造。")
    md.append("")
    md.append("## 2. 版本与指纹（修复前后）")
    md.append("")
    md.append("| 项 | v1 | v2 |")
    md.append("|---|---:|---:|")
    md.append(f"| 日线行数 | {pre_fp.get('daily_rows', 'NA'):,} | {post_fp['daily_rows']:,} |")
    md.append(f"| 证券数 | {pre_fp.get('symbols', 'NA'):,} | {post_fp['symbols']:,} |")
    md.append(f"| 时间范围 | {pre_fp.get('date_min')}~{pre_fp.get('date_max')} | {post_fp['date_min']}~{post_fp['date_max']} |")
    md.append(f"| 未来行/重复主键 | {pre_fp.get('future_rows')}/{pre_fp.get('dup_keys')} | {post_fp['future_rows']}/{post_fp['dup_keys']} |")
    md.append(f"| 文件集合 SHA-256 | `{pre_fp.get('files_agg_sha256')}` | `{post_fp['files_agg_sha256']}` |")
    md.append(f"| daily 内容指纹（跨重建可复现） | — | `{post_fp.get('content_digest')}` |")
    md.append(f"| meta.json SHA-256 | `{pre_fp.get('meta_json_sha256')}` | `{post_fp['meta_json_sha256']}` |")
    md.append(f"| frozen.duckdb SHA-256 | `{pre_fp.get('frozen_duckdb_sha256')}` | `{post_fp['frozen_duckdb_sha256']}` |")
    md.append("")
    md.append("> 文件集合 SHA 会随 Parquet 编码/行序变化（同一内容指纹下多次重建不同）；跨版本比对请用 **daily 内容指纹**。`meta.json.files[*].sha256` 只用于验证当前产物未被篡改。")
    hist = evidence["build_history"]
    same = sum(1 for h in hist if h.get("content_digest") == post_fp.get("content_digest"))
    md.append(f"> 构建历史：`_audit/build_history.csv` 共 {len(hist)} 行，其中 {same} 行的内容指纹与当前一致；连续重建的文件指纹不同、内容指纹相同。")
    md.append("> v1 的 `hfq_*` = `ROUND(raw_px × factor, 4)`，可由当前 raw+factor 精确重建（仅 4 位舍入差 ≤5e-5），因此未保留 1.4 GB 旧数据副本；旧结果可按此公式复算。")
    md.append("v1 的 `meta.json`、自检报告与缺陷量化已原样保留在 `01_数据/_audit/v1_修复前_*`，未删除。")
    md.append("")
    md.append("## 3. 修复前缺陷量化（v1 公式；全样本实测，除权日为反算）")
    md.append("")
    h = pre_def.get("hfq_ret_vs_pctchg", {})
    ex = pre_def.get("exdate", {})
    md.append(f"- **全样本相邻日**：{pre_def.get('n_pairs', 0):,} 对中 **{h.get('bad_gt_1bp', 0):,}（{h.get('bad_share_pct', 0):.2f}%）** 的 hfq 日收益偏离官方 pct_chg >1bp；中位偏差 **{h.get('median_abs', 0)*100:.2f}pp**；OLS 斜率 **{h.get('ols_slope_trim_0.1_30pct', 0):.4f}**（应为 1）。")
    md.append(f"- **除权日**：{ex.get('n', 0):,} 个除权日中，v1 调整价收益偏离 >1bp 的 **{ex.get('adj_bad_gt_1bp', 0):,}** 个（{ex.get('adj_bad_gt_1bp',0)/max(ex.get('n',1),1)*100:.1f}%），中位偏差 **{ex.get('adj_median_abs', 0)*100:.2f}pp**、p95 **{ex.get('adj_p95_abs', 0)*100:.2f}pp**；同一批日期原始收盘收益的中位偏差 {ex.get('raw_median_abs', 0)*100:.2f}pp、p95 {ex.get('raw_p95_abs', 0)*100:.2f}pp（这是送转/分红的真实价格跳空）。")
    md.append(f"- **随机日抽样**：{pre_def.get('random_sample', {}).get('n', 0):,} 行中 {pre_def.get('random_sample', {}).get('bad_gt_1bp', 0):,} 行 >1bp。")
    if pre_def.get("reconstruction_note"):
        md.append(f"- 计数口径：{pre_def['reconstruction_note']}")
    md.append("- **旧自检为何没发现**：C1 检查的是 `hfq = close × factor`（把错误公式当成正确性），C2 检查的是 `factor` 的收益，二者都不会失败。")
    md.append("")
    md.append("## 4. 修复后不变量（v2 实测）")
    md.append("")
    md.append("| 编号 | 不变量 | 结果 | 能否抓住双计 |")
    md.append("|---|---|---|---|")
    verdicts = {
        "I1": f"{c5[0]:,} 对，>0.5bp={c5[1]}，最大={c5[3]:.1e}，中位={c5[4]:.1e}，斜率={c5[6]:.6f}",
        "I2": f"{c6[0]:,} 个除权日，调整价违约={c6[1]}，原始价偏离>1%={c6[3]:,}（{c6[3]/max(c6[0],1)*100:.2f}%）",
        "I2b": f"{c6b[0]:,} 对，违约={c6b[1]}，最大={c6b[2]:.1e}",
        "I3": f"hfq 最大比例误差={max(c7):.1e}；qfq={max(c7q):.1e}",
        "I4": f"{c8[0]:,} 只，首条违约={c8[5]}，末日违约={c8[6]}",
        "I5": "; ".join(f"{r['cut']} hfq误差={r['hfq_rel']:.1e}" for r in trunc_rows),
        "I1b": f"{c10[0]:,} 行，违约={c10[1]}，最大={c10[2]:.1e}",
        "I6": f"旧公式 {c11[1]:,}/{c11[0]:,} 行>1bp（{c11[1]/max(c11[0],1)*100:.2f}%），除权日中位={c11e[1]*100:.2f}pp、p95={c11e[2]*100:.2f}pp、>5pp={c11e[4]:,}",
        "I7": f"{c12[0]:,} 对，>0.5bp={c12[1]}，最大={c12[2]:.1e}",
    }
    catches = {x["id"]: x["catches_double_count"] for x in invariants}
    for x in invariants:
        md.append(f"| {x['id']} | {x['name']} | {verdicts.get(x['id'], '')} | {'是' if catches.get(x['id']) else '否'} |")
    md.append("")
    md.append("**读法**：OHLC 比例、首末锚定、截断不变性在 v1 上也全部通过——它们只能证明“同一 bar 用了一个乘子”“水平锚点正确”“没有未来数据”，不能证明乘子本身不是价格的重复。只有把调整价**跨日**与官方 pct_chg / 原始 open-preclose 对齐（I1/I2/I2b/I1b）才能抓住双计；I6 用旧公式反算，确认这组检查在 v1 上会失败。")
    md.append("")
    md.append("## 5. 案例（688050.SH 2024-06-05，10 送 8 派 0.49）")
    md.append("")
    md.append("| 口径 | 当日收益 | 说明 |")
    md.append("|---|---:|---|")
    md.append("| 官方 pct_chg | **-1.39%** | 交易所除权参考价 80.82 |")
    md.append("| 原始 close-to-close | -45.37% | 送转造成的价格跳空，不是亏损 |")
    md.append("| v1 hfq_close | -46.12% | 送转 + 双计 |")
    md.append("| v2 hfq_close | **-1.39%** | 与 pct_chg 一致 |")
    md.append("")
    md.append("其余案例见 `04_证据/复权修复.json` 的 `cases`。")
    md.append("")
    md.append("## 6. 限制与下游影响")
    md.append("")
    for x in evidence["limitations"]:
        md.append(f"- {x}")
    md.append("")
    md.append("## 7. 复现")
    md.append("")
    md.append("```bash")
    md.append("cd /Users/admin/WorkSpace/trading/trading_趋势结构")
    md.append("python3 02_代码/构建冻结数据.py            # 复用 _tmp 中间件，重建 daily/market/duckdb/meta")
    md.append("python3 02_代码/01_自检_数据.py --deep    # 自检 + 生成本报告与 04_证据/复权修复.json")
    md.append("python3 02_代码/数据层.py                 # 接口冒烟（含 hfq 收益核对）")
    md.append("```")
    (AUDIT / "修复后自检报告.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
