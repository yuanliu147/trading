#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
研究面板 + 横截面劈半
项目：2026-10-29_趋势结构适配性_选股特征
预注册：§4（口径）§5.2（分层确定性劈半）§5.4（封存）§5.5（预热）

产出：
  01_数据/面板_2008预热_2020.parquet   —— 含预热期，仅用于特征移动窗
  01_数据/横截面劈半.parquet            —— symbol -> 半区 A/B + 分层键
  04_证据/面板与劈半.json               —— 可审计证据（含两半区可比性检验）
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import duckdb

母项目 = Path("/Users/admin/WorkSpace/trading/trading_趋势结构")
冻结目录 = 母项目 / "01_数据" / "frozen"
本项目 = Path("/Users/admin/WorkSpace/trading/2026-10-29_趋势结构适配性_选股特征")
数据目录 = 本项目 / "01_数据"
证据目录 = 本项目 / "04_证据"

预热起 = "2008-07-01"     # §5.5 仅预热
研究期起 = "2010-01-01"
研究期止 = "2020-12-31"   # §5.4 封存线

DAILY = f"read_parquet('{冻结目录}/daily/*/*.parquet')"

# 特征计算与信号判定实际需要的列（不取全部 39 列，控制内存）
取列 = [
    "symbol", "date", "board", "td_since_list",
    "open", "high", "low", "close", "preclose",
    "volume", "amount", "turn", "trade_status", "is_st", "is_suspended",
    "pct_chg", "pe_ttm", "pb_mrq",
    "hfq_open", "hfq_high", "hfq_low", "hfq_close",
    "limit_pct", "is_limit_up_close", "is_limit_up_open", "is_limit_down_open",
    "float_mktcap",
]


def 连接():
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    return con


# ---------------------------------------------------------------- 1. 面板
def 建面板(con) -> dict:
    """含预热期的面板。硬断言不越封存线。"""
    cols = ", ".join(取列)
    out = 数据目录 / "面板_2008预热_2020.parquet"
    con.execute(f"""
        COPY (
          SELECT {cols}
          FROM {DAILY}
          WHERE date >= DATE '{预热起}' AND date <= DATE '{研究期止}'
          ORDER BY symbol, date
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    chk = con.execute(f"""
        SELECT count(*) n, count(DISTINCT symbol) ns, min(date) d0, max(date) d1,
               sum(CASE WHEN date > DATE '{研究期止}' THEN 1 ELSE 0 END) AS 越线行数,
               sum(CASE WHEN date >= DATE '{研究期起}' THEN 1 ELSE 0 END) AS 研究期行数,
               sum(CASE WHEN date <  DATE '{研究期起}' THEN 1 ELSE 0 END) AS 预热期行数
        FROM read_parquet('{out}')
    """).fetchone()

    # §5.4 硬断言
    assert int(chk[4]) == 0, f"封存线被越过：{chk[4]} 行 date > {研究期止}"
    assert str(chk[3]) <= 研究期止, f"末日越线：{chk[3]}"

    return {
        "文件": str(out), "字节": out.stat().st_size,
        "行数": int(chk[0]), "证券数": int(chk[1]),
        "首日": str(chk[2]), "末日": str(chk[3]),
        "越封存线行数": int(chk[4]),
        "研究期行数": int(chk[5]), "预热期行数": int(chk[6]),
        "列": 取列,
        "断言": "date <= 2020-12-31 全部通过",
    }


# ---------------------------------------------------------------- 2. 分层劈半
def 建劈半(con) -> dict:
    """§5.2：按 (board, 研究期首现日流通市值五分位) 分层，层内按 sha256(symbol) 升序交替分配。"""
    面板 = 数据目录 / "面板_2008预热_2020.parquet"

    # 研究期内首现日的 board 与 float_mktcap（PIT）
    首现 = con.execute(f"""
        WITH f AS (
          SELECT symbol, min(date) AS first_d
          FROM read_parquet('{面板}')
          WHERE date >= DATE '{研究期起}'
          GROUP BY symbol
        )
        SELECT f.symbol, f.first_d, d.board, d.float_mktcap
        FROM f JOIN read_parquet('{面板}') d
          ON d.symbol = f.symbol AND d.date = f.first_d
        ORDER BY f.symbol
    """).fetchall()

    # 首现日流通市值五分位（全样本横截面，PIT：用各自首现日的值做跨股票排序）
    import numpy as np
    syms = [r[0] for r in 首现]
    first_d = [r[1] for r in 首现]
    boards = [r[2] for r in 首现]
    mc = np.array([np.nan if r[3] is None else float(r[3]) for r in 首现])

    有效 = ~np.isnan(mc)
    q5 = np.full(len(mc), -1, dtype=int)          # -1 = 市值缺失组
    if 有效.sum() > 0:
        切点 = np.nanquantile(mc[有效], [0.2, 0.4, 0.6, 0.8])
        q5[有效] = np.searchsorted(切点, mc[有效], side="right")

    # 层内按 sha256 升序交替
    层 = {}
    for i, s in enumerate(syms):
        键 = (boards[i], int(q5[i]))
        层.setdefault(键, []).append((hashlib.sha256(s.encode()).hexdigest(), s, i))

    分配 = {}
    层统计 = []
    for 键, 成员 in sorted(层.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        成员.sort()                                # 按 sha256 十六进制升序
        for 位序, (h, s, i) in enumerate(成员):
            分配[s] = "A" if 位序 % 2 == 0 else "B"
        层统计.append({
            "板块": 键[0], "市值五分位": 键[1], "证券数": len(成员),
            "A": sum(1 for _, s, _ in 成员 if 分配[s] == "A"),
            "B": sum(1 for _, s, _ in 成员 if 分配[s] == "B"),
        })

    # 落盘
    con.execute("DROP TABLE IF EXISTS 劈半")
    con.execute("""CREATE TABLE 劈半(
        symbol VARCHAR, 半区 VARCHAR, 板块 VARCHAR,
        市值五分位 INTEGER, 首现日 DATE, 首现日流通市值 DOUBLE, sha256前16 VARCHAR)""")
    行 = [(syms[i], 分配[syms[i]], boards[i], int(q5[i]), first_d[i],
           None if np.isnan(mc[i]) else float(mc[i]),
           hashlib.sha256(syms[i].encode()).hexdigest()[:16])
          for i in range(len(syms))]
    con.executemany("INSERT INTO 劈半 VALUES (?,?,?,?,?,?,?)", 行)
    out = 数据目录 / "横截面劈半.parquet"
    con.execute(f"COPY 劈半 TO '{out}' (FORMAT PARQUET)")

    return {
        "文件": str(out),
        "证券总数": len(syms),
        "A半区": sum(1 for v in 分配.values() if v == "A"),
        "B半区": sum(1 for v in 分配.values() if v == "B"),
        "层数": len(层统计),
        "分层明细": 层统计,
        "方法": "层内按 sha256(symbol) 十六进制升序，偶数位→A，奇数位→B；无随机种子，完全确定性",
        "市值缺失组": int((q5 == -1).sum()),
    }


# ---------------------------------------------------------------- 3. 两半区可比性
def 可比性(con) -> dict:
    """§5.2 要求披露：两半区在属性分布上是否可比，以及日期层相关性（折扣依据）。"""
    面板 = 数据目录 / "面板_2008预热_2020.parquet"
    劈半 = 数据目录 / "横截面劈半.parquet"

    分布 = con.execute(f"""
        WITH p AS (
          SELECT d.*, s.半区
          FROM read_parquet('{面板}') d
          JOIN read_parquet('{劈半}') s USING (symbol)
          WHERE d.date >= DATE '{研究期起}'
        )
        SELECT 半区,
          count(DISTINCT symbol)                     AS 证券数,
          count(*)                                   AS 行数,
          median(float_mktcap)                       AS 流通市值中位,
          median(amount)                             AS 成交额中位,
          median(turn)                               AS 换手率中位,
          avg(CASE WHEN is_st=1 THEN 1.0 ELSE 0.0 END)      AS ST占比,
          avg(CASE WHEN is_suspended THEN 1.0 ELSE 0.0 END) AS 停牌占比,
          stddev_samp(pct_chg)                       AS 日收益标准差,
          avg(pct_chg)                               AS 日收益均值
        FROM p GROUP BY 半区 ORDER BY 半区
    """).fetchall()
    列名 = ["半区", "证券数", "行数", "流通市值中位", "成交额中位", "换手率中位",
            "ST占比", "停牌占比", "日收益标准差", "日收益均值"]
    分布字典 = [dict(zip(列名, [r[0]] + [float(x) if x is not None else None for x in r[1:]]))
                for r in 分布]

    # 两半区等权日收益序列的相关性（日期层不独立的量化折扣依据）
    相关 = con.execute(f"""
        WITH p AS (
          SELECT d.date, d.pct_chg, s.半区
          FROM read_parquet('{面板}') d
          JOIN read_parquet('{劈半}') s USING (symbol)
          WHERE d.date >= DATE '{研究期起}' AND NOT d.is_suspended
        ), g AS (
          SELECT date, 半区, avg(pct_chg) AS r FROM p GROUP BY date, 半区
        ), w AS (
          SELECT date,
                 max(CASE WHEN 半区='A' THEN r END) AS ra,
                 max(CASE WHEN 半区='B' THEN r END) AS rb
          FROM g GROUP BY date
        )
        SELECT count(*), corr(ra, rb) FROM w WHERE ra IS NOT NULL AND rb IS NOT NULL
    """).fetchone()

    return {
        "分布对比": 分布字典,
        "两半区等权日收益相关": {"交易日数": int(相关[0]), "相关系数": float(相关[1])},
        "独立性折扣声明": (
            "两半区共享同一日历与同一批市场级冲击，等权日收益相关系数见上。"
            "该相关性意味着 B 半区不是真正独立样本，全部统计量必须按信号日聚类，"
            "且 B 半区确认的证据强度须按此相关性折扣（预注册 §5.2）。"
        ),
    }


def main():
    数据目录.mkdir(parents=True, exist_ok=True)
    证据目录.mkdir(parents=True, exist_ok=True)
    con = 连接()

    print("[1/3] 建研究面板（含 §5.5 预热期）")
    面板信息 = 建面板(con)
    print(f"  行数 {面板信息['行数']:,}  证券 {面板信息['证券数']:,}  "
          f"{面板信息['首日']}..{面板信息['末日']}")
    print(f"  研究期 {面板信息['研究期行数']:,} 行 / 预热期 {面板信息['预热期行数']:,} 行  "
          f"越封存线 {面板信息['越封存线行数']} 行")
    print(f"  文件 {面板信息['字节']/1e6:.1f} MB")

    print("\n[2/3] 分层确定性劈半（§5.2）")
    劈半信息 = 建劈半(con)
    print(f"  证券 {劈半信息['证券总数']}  A半区 {劈半信息['A半区']}  "
          f"B半区 {劈半信息['B半区']}  层数 {劈半信息['层数']}")
    print(f"  市值缺失组 {劈半信息['市值缺失组']}")

    print("\n[3/3] 两半区可比性与独立性折扣")
    比较 = 可比性(con)
    for d in 比较["分布对比"]:
        print(f"  半区{d['半区']}: 证券{d['证券数']:>5}  行{d['行数']:>9,}  "
              f"市值中位{d['流通市值中位']/1e8:>8.2f}亿  成交额中位{d['成交额中位']/1e4:>8.1f}万  "
              f"换手中位{d['换手率中位']:>5.2f}%  ST{d['ST占比']:.3%}  "
              f"日波动{d['日收益标准差']:.3f}%  日均{d['日收益均值']:+.4f}%")
    print(f"  两半区等权日收益相关: {比较['两半区等权日收益相关']['相关系数']:.4f} "
          f"（{比较['两半区等权日收益相关']['交易日数']} 个交易日）")

    out = {
        "生成时间": dt.datetime.now().isoformat(timespec="seconds"),
        "脚本": "02_代码/02_面板与劈半.py",
        "预注册依据": ["§4 口径", "§5.2 分层劈半", "§5.4 封存", "§5.5 预热"],
        "面板": 面板信息, "劈半": 劈半信息, "可比性": 比较,
    }
    p = 证据目录 / "面板与劈半.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n证据已写入: {p}")
    con.close()


if __name__ == "__main__":
    main()
