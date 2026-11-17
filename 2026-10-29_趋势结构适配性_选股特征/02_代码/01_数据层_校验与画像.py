#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据层：校验冻结面板 + 研究期画像
项目：2026-10-29_趋势结构适配性_选股特征
预注册：00_研究预注册.md §4 §5

硬约束（预注册 §5.4）：任何取数 date <= '2020-12-31'
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import duckdb

# ---------------------------------------------------------------- 路径与常量
母项目 = Path("/Users/admin/WorkSpace/trading/trading_趋势结构")
冻结目录 = 母项目 / "01_数据" / "frozen"
BAOSTOCK_DB = Path("/Users/admin/WorkSpace/baostock_data/data/baostock.db")
本项目 = Path("/Users/admin/WorkSpace/trading/2026-10-29_趋势结构适配性_选股特征")
数据目录 = 本项目 / "01_数据"
证据目录 = 本项目 / "04_证据"

研究期起 = "2010-01-01"
研究期止 = "2020-12-31"
封存起点 = "2021-01-01"

# 预注册 §4.1 记录的冻结面板摘要
预期摘要 = "28e75745dd81ddb133cf299c5b34c46b6c1556a2cee01a7cd9bb563903233148"
预期行数 = 16_698_536
预期证券数 = 5_549
# meta.json content_digest.components:
# count, sum(open*100), sum(high*100), sum(low*100), sum(close*100),
# sum(hfq_close*1e4), sum(factor*1e6), sum(pct_chg*1e4)  —— 全部取整
预期分量 = [
    16698536, 31133474954, 31862408699, 30491995138, 31171060649,
    7932397305708, 45724328148332, 13506427811,
]

DAILY = f"read_parquet('{冻结目录}/daily/*/*.parquet')"


def 连接() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    return con


# ---------------------------------------------------------------- 1. 摘要校验
def 校验冻结摘要(con) -> dict:
    """复算 meta.json 的 content_digest.components，逐项比对。"""
    sql = f"""
    SELECT
      count(*)                                  AS c_count,
      CAST(sum(open  * 100)  AS HUGEINT)        AS c_open,
      CAST(sum(high  * 100)  AS HUGEINT)        AS c_high,
      CAST(sum(low   * 100)  AS HUGEINT)        AS c_low,
      CAST(sum(close * 100)  AS HUGEINT)        AS c_close,
      CAST(sum(hfq_close * 1e4) AS HUGEINT)     AS c_hfq,
      CAST(sum(factor * 1e6) AS HUGEINT)        AS c_factor,
      CAST(sum(pct_chg * 1e4) AS HUGEINT)       AS c_pct,
      count(DISTINCT symbol)                    AS n_sym,
      min(date) AS d_min, max(date) AS d_max
    FROM {DAILY}
    """
    r = con.execute(sql).fetchone()
    实测分量 = [int(x) for x in r[:8]]
    逐项 = []
    for i, (期望, 实测) in enumerate(zip(预期分量, 实测分量)):
        逐项.append({
            "分量序号": i, "期望": 期望, "实测": 实测,
            "一致": 期望 == 实测, "差值": 实测 - 期望,
        })
    结果 = {
        "摘要字符串_预注册记录": 预期摘要,
        "分量逐项": 逐项,
        "全部分量一致": all(x["一致"] for x in 逐项),
        "行数": {"期望": 预期行数, "实测": int(r[0]), "一致": int(r[0]) == 预期行数},
        "证券数": {"期望": 预期证券数, "实测": int(r[8]), "一致": int(r[8]) == 预期证券数},
        "日期范围": {"min": str(r[9]), "max": str(r[10])},
    }
    结果["通过"] = (结果["全部分量一致"] and 结果["行数"]["一致"]
                 and 结果["证券数"]["一致"])
    return 结果


# ---------------------------------------------------------------- 2. 研究期画像
def 研究期画像(con) -> dict:
    """研究期 2010-2020 的覆盖、缺失率、关键字段分布。决定劈半键是否可用。"""
    基础 = con.execute(f"""
        SELECT count(*) n, count(DISTINCT symbol) n_sym,
               count(DISTINCT date) n_day, min(date) d0, max(date) d1
        FROM {DAILY}
        WHERE date >= DATE '{研究期起}' AND date <= DATE '{研究期止}'
    """).fetchone()

    # 缺失率（研究期内）
    缺失 = con.execute(f"""
        SELECT
          count(*) AS n,
          sum(CASE WHEN float_mktcap IS NULL THEN 1 ELSE 0 END) AS miss_mktcap,
          sum(CASE WHEN turn         IS NULL THEN 1 ELSE 0 END) AS miss_turn,
          sum(CASE WHEN pe_ttm       IS NULL THEN 1 ELSE 0 END) AS miss_pe,
          sum(CASE WHEN pb_mrq       IS NULL THEN 1 ELSE 0 END) AS miss_pb,
          sum(CASE WHEN amount       IS NULL OR amount = 0 THEN 1 ELSE 0 END) AS miss_amt,
          sum(CASE WHEN hfq_close    IS NULL THEN 1 ELSE 0 END) AS miss_hfq,
          sum(CASE WHEN is_st        IS NULL THEN 1 ELSE 0 END) AS miss_st,
          sum(CASE WHEN is_suspended THEN 1 ELSE 0 END) AS n_susp,
          sum(CASE WHEN trade_status = 0 THEN 1 ELSE 0 END) AS n_ts0
        FROM {DAILY}
        WHERE date >= DATE '{研究期起}' AND date <= DATE '{研究期止}'
    """).fetchone()
    n = 缺失[0]
    缺失率 = {
        "float_mktcap": 缺失[1] / n, "turn": 缺失[2] / n,
        "pe_ttm": 缺失[3] / n, "pb_mrq": 缺失[4] / n,
        "amount_零或空": 缺失[5] / n, "hfq_close": 缺失[6] / n,
        "is_st": 缺失[7] / n,
        "停牌占比": 缺失[8] / n, "trade_status0占比": 缺失[9] / n,
    }

    # 分板块证券数（研究期内出现过的）
    板块 = con.execute(f"""
        SELECT board, count(DISTINCT symbol) n
        FROM {DAILY}
        WHERE date >= DATE '{研究期起}' AND date <= DATE '{研究期止}'
        GROUP BY board ORDER BY n DESC
    """).fetchall()

    # 分年证券数与行数
    分年 = con.execute(f"""
        SELECT year, count(*) n, count(DISTINCT symbol) n_sym
        FROM {DAILY}
        WHERE date >= DATE '{研究期起}' AND date <= DATE '{研究期止}'
        GROUP BY year ORDER BY year
    """).fetchall()

    # 首次出现日（用于分层劈半：取研究期内首个交易日的属性）
    首现 = con.execute(f"""
        WITH f AS (
          SELECT symbol, min(date) AS first_d
          FROM {DAILY}
          WHERE date >= DATE '{研究期起}' AND date <= DATE '{研究期止}'
          GROUP BY symbol
        )
        SELECT count(*) n_sym,
               sum(CASE WHEN d.float_mktcap IS NULL THEN 1 ELSE 0 END) AS miss_mc_at_first,
               sum(CASE WHEN d.board IS NULL THEN 1 ELSE 0 END) AS miss_board_at_first
        FROM f JOIN {DAILY} d ON d.symbol = f.symbol AND d.date = f.first_d
    """).fetchone()

    # 市场宽度表覆盖
    mkt = con.execute(f"""
        SELECT count(*) n, min(date) d0, max(date) d1,
               sum(CASE WHEN ew_ret IS NULL THEN 1 ELSE 0 END) miss_ew,
               sum(CASE WHEN hs300_ret IS NULL THEN 1 ELSE 0 END) miss_300
        FROM read_parquet('{冻结目录}/market_daily.parquet')
        WHERE date >= DATE '{研究期起}' AND date <= DATE '{研究期止}'
    """).fetchone()

    return {
        "研究期": {"起": 研究期起, "止": 研究期止},
        "基础": {"行数": int(基础[0]), "证券数": int(基础[1]),
                 "交易日数": int(基础[2]), "首日": str(基础[3]), "末日": str(基础[4])},
        "缺失率": 缺失率,
        "分板块证券数": {b: int(c) for b, c in 板块},
        "分年": [{"年": int(y), "行数": int(nn), "证券数": int(ns)} for y, nn, ns in 分年],
        "首现日属性可用性": {
            "证券数": int(首现[0]),
            "首现日流通市值缺失": int(首现[1]),
            "首现日板块缺失": int(首现[2]),
        },
        "市场宽度表": {"行数": int(mkt[0]), "首日": str(mkt[1]), "末日": str(mkt[2]),
                      "ew_ret缺失": int(mkt[3]), "hs300_ret缺失": int(mkt[4])},
    }


# ---------------------------------------------------------------- 3. baostock 补充字段画像
def baostock画像() -> dict:
    """母项目未用的 ps_ttm / pcf_ncf_ttm / adjust_factor，先看可用性。只读。"""
    con = sqlite3.connect(f"file:{BAOSTOCK_DB}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=1")
    cur = con.cursor()

    # 同步状态快照（预注册 §4.1 要求）
    cur.execute("SELECT count(*), min(last_date), max(last_date) FROM sync_state")
    同步 = cur.fetchone()

    # 估值字段覆盖（研究期、不复权）
    cur.execute(f"""
        SELECT count(*) n,
               sum(CASE WHEN ps_ttm IS NULL THEN 1 ELSE 0 END),
               sum(CASE WHEN pcf_ncf_ttm IS NULL THEN 1 ELSE 0 END),
               sum(CASE WHEN pe_ttm IS NULL THEN 1 ELSE 0 END),
               sum(CASE WHEN pb_mrq IS NULL THEN 1 ELSE 0 END),
               sum(CASE WHEN ps_ttm <= 0 THEN 1 ELSE 0 END),
               sum(CASE WHEN pcf_ncf_ttm <= 0 THEN 1 ELSE 0 END)
        FROM stock_kline_d
        WHERE adjustflag = 3 AND date >= '{研究期起}' AND date <= '{研究期止}'
    """)
    v = cur.fetchone()
    n = v[0] or 1

    # 分红表覆盖
    cur.execute(f"""
        SELECT count(*), count(DISTINCT code), min(divid_operate_date), max(divid_operate_date)
        FROM adjust_factor
        WHERE divid_operate_date >= '{研究期起}' AND divid_operate_date <= '{研究期止}'
    """)
    d = cur.fetchone()

    # 分红因子分布（隐含收益率上界 = adjust_factor - 1）
    cur.execute(f"""
        SELECT
          sum(CASE WHEN adjust_factor IS NULL THEN 1 ELSE 0 END),
          min(adjust_factor), max(adjust_factor),
          sum(CASE WHEN adjust_factor > 1.5 THEN 1 ELSE 0 END)
        FROM adjust_factor
        WHERE divid_operate_date >= '{研究期起}' AND divid_operate_date <= '{研究期止}'
    """)
    af = cur.fetchone()
    con.close()

    return {
        "只读模式": "file:...?mode=ro + PRAGMA query_only=1",
        "同步状态": {"任务数": 同步[0], "last_date最小": 同步[1], "last_date最大": 同步[2]},
        "估值字段_研究期_不复权": {
            "行数": int(v[0]),
            "ps_ttm缺失率": v[1] / n, "pcf_ncf_ttm缺失率": v[2] / n,
            "pe_ttm缺失率": v[3] / n, "pb_mrq缺失率": v[4] / n,
            "ps_ttm非正占比": v[5] / n, "pcf_ncf_ttm非正占比": v[6] / n,
        },
        "分红表_研究期": {
            "除权除息事件数": int(d[0]), "涉及证券数": int(d[1]),
            "首日": d[2], "末日": d[3],
            "adjust_factor缺失": int(af[0]),
            "adjust_factor最小": af[1], "最大": af[2],
            "大于1.5的笔数_疑送转": int(af[3]),
        },
        "口径声明": "adjust_factor-1 含送转，是隐含收益率上界，不等于股息率（预注册 §4.2）",
    }


# ---------------------------------------------------------------- main
def main():
    数据目录.mkdir(parents=True, exist_ok=True)
    证据目录.mkdir(parents=True, exist_ok=True)
    con = 连接()

    print("=" * 70)
    print("[1/3] 校验母项目冻结面板摘要（预注册 §4.1）")
    摘要 = 校验冻结摘要(con)
    print(f"  全部分量一致: {摘要['全部分量一致']}  行数一致: {摘要['行数']['一致']}  "
          f"证券数一致: {摘要['证券数']['一致']}")
    print(f"  日期范围: {摘要['日期范围']['min']} .. {摘要['日期范围']['max']}")
    if not 摘要["通过"]:
        for x in 摘要["分量逐项"]:
            if not x["一致"]:
                print(f"  !! 分量{x['分量序号']} 期望{x['期望']} 实测{x['实测']} 差{x['差值']}")

    print("\n[2/3] 研究期 2010-2020 画像")
    画像 = 研究期画像(con)
    b = 画像["基础"]
    print(f"  行数 {b['行数']:,}  证券 {b['证券数']:,}  交易日 {b['交易日数']:,}  "
          f"{b['首日']}..{b['末日']}")
    print("  缺失率:")
    for k, v in 画像["缺失率"].items():
        print(f"    {k:22s} {v:7.4%}")
    print(f"  分板块: {画像['分板块证券数']}")
    fo = 画像["首现日属性可用性"]
    print(f"  首现日: 证券{fo['证券数']}  市值缺失{fo['首现日流通市值缺失']}  "
          f"板块缺失{fo['首现日板块缺失']}")
    print(f"  市场宽度表: {画像['市场宽度表']}")

    print("\n[3/3] baostock 补充字段画像（母项目未用）")
    bs = baostock画像()
    print(f"  同步状态: {bs['同步状态']}")
    for k, v in bs["估值字段_研究期_不复权"].items():
        print(f"    {k:22s} {v if isinstance(v, int) else f'{v:7.4%}'}")
    print(f"  分红表: 事件{bs['分红表_研究期']['除权除息事件数']:,} "
          f"证券{bs['分红表_研究期']['涉及证券数']:,} "
          f"疑送转{bs['分红表_研究期']['大于1.5的笔数_疑送转']:,}")

    out = {
        "生成时间": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "脚本": "02_代码/01_数据层_校验与画像.py",
        "冻结面板摘要校验": 摘要,
        "研究期画像": 画像,
        "baostock补充字段画像": bs,
    }
    p = 证据目录 / "数据层_校验与画像.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n证据已写入: {p}")
    con.close()


if __name__ == "__main__":
    main()
