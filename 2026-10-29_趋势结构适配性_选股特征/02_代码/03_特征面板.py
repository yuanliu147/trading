#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事前特征面板（全部 PIT，只用 <= 当日收盘可得的数据；入场时再 shift 到 T-1）
项目：2026-10-29_趋势结构适配性_选股特征
预注册：§7 特征清单、§7.1 估值与分红硬规则

阶段：
  S1 个股路径特征（A/B/C/F 族）   -> 01_数据/特征_S1个股.parquet
  S2 baostock 独有估值字段补充     -> 01_数据/特征_S2估值原始.parquet
  S3 横截面分位 + 分红特征         -> 01_数据/特征_完整.parquet

A 族（趋势可交易性）是母项目从未测过的维度，为本研究核心。
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

母项目 = Path("/Users/admin/WorkSpace/trading/trading_趋势结构")
冻结目录 = 母项目 / "01_数据" / "frozen"
BAOSTOCK_DB = Path("/Users/admin/WorkSpace/baostock_data/data/baostock.db")
本项目 = Path("/Users/admin/WorkSpace/trading/2026-10-29_趋势结构适配性_选股特征")
数据目录 = 本项目 / "01_数据"
证据目录 = 本项目 / "04_证据"

研究期起 = "2010-01-01"
研究期止 = "2020-12-31"
面板文件 = 数据目录 / "面板_2008预热_2020.parquet"

S1文件 = 数据目录 / "特征_S1个股.parquet"
S2文件 = 数据目录 / "特征_S2估值原始.parquet"
完整文件 = 数据目录 / "特征_完整.parquet"


# ================================================================ 工具
def _滚动R2(logp: pd.Series, n: int) -> np.ndarray:
    """log 价对时间线性回归的 R²（滚动 n）。用解析式避免逐窗 polyfit。"""
    t = pd.Series(np.arange(len(logp), dtype=float), index=logp.index)
    st = t.rolling(n).sum()
    sy = logp.rolling(n).sum()
    stt = (t * t).rolling(n).sum()
    syy = (logp * logp).rolling(n).sum()
    sty = (t * logp).rolling(n).sum()
    cov = sty - st * sy / n
    vart = stt - st * st / n
    vary = syy - sy * sy / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = (cov * cov) / (vart * vary)
    return r2.to_numpy()


def _因果枢轴数(high: np.ndarray, low: np.ndarray, k: int, 窗: int) -> np.ndarray:
    """因果分形枢轴计数（O(n)）。

    i 为枢轴需 high[i]==max(high[i-k..i+k]) 或 low[i]==min(low[i-k..i+k])；
    该枢轴在 i+k 日才能被确认（此前需要未来 k 根 K 线，用了就是前视）。
    故在 t 日可用的枢轴，其位置必须满足 i <= t-k；再叠加 trailing 窗 约束 i > t-窗。
    两个约束合成为位置上的固定宽度窗 (t-窗, t-k]，可用前缀和 O(1) 取。
    """
    n = len(high)
    if n == 0:
        return np.zeros(0, dtype=float)
    sh = pd.Series(high)
    sl = pd.Series(low)
    w = 2 * k + 1
    峰 = sh.rolling(w, center=True).max().to_numpy()
    谷 = sl.rolling(w, center=True).min().to_numpy()
    是枢轴 = ((high == 峰) | (low == 谷))
    是枢轴 = np.nan_to_num(是枢轴.astype(float), nan=0.0)
    # 前 k 与后 k 根无法形成完整中心窗，不算枢轴（rolling 已给 NaN，这里显式置 0）
    是枢轴[:k] = 0.0
    是枢轴[max(0, n - k):] = 0.0

    cs = np.concatenate([[0.0], np.cumsum(是枢轴)])      # cs[j] = sum(是枢轴[0:j])
    t = np.arange(n)
    hi = t - k                                           # 可用枢轴位置上界（含）
    lo = np.maximum(0, t - 窗 + 1)                       # 下界（含）
    有效 = hi >= lo
    计数 = np.zeros(n, dtype=float)
    计数[有效] = cs[hi[有效] + 1] - cs[lo[有效]]
    return 计数



def 个股特征(g: pd.DataFrame) -> pd.DataFrame:
    """单只股票的全部路径特征。g 已按 date 升序、已剔除停牌日。"""
    out = pd.DataFrame({"symbol": g["symbol"].to_numpy(), "date": g["date"].to_numpy()})
    c = g["hfq_close"].astype(float)
    logc = np.log(c.where(c > 0))
    r1 = logc.diff()

    # ---------------- A 族：趋势可交易性（母项目从未测过）
    absd = r1.abs()
    for n in (60, 120):
        净位移 = (logc - logc.shift(n)).abs()
        总路程 = absd.rolling(n).sum()
        out[f"ER_{n}"] = (净位移 / 总路程).to_numpy()          # Kaufman 效率比
    var1 = r1.rolling(120).var()
    for k in (5, 20):
        rk = logc - logc.shift(k)
        vark = rk.rolling(120).var()
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"VR_{k}"] = (vark / (k * var1)).to_numpy()     # 方差比
    for n in (60, 120):
        out[f"AC1_{n}"] = r1.rolling(n).corr(r1.shift(1)).to_numpy()
        out[f"R2_{n}"] = _滚动R2(logc, n)
    枢轴数 = _因果枢轴数(g["hfq_high"].to_numpy(float), g["hfq_low"].to_numpy(float), 5, 250)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["SEGLEN"] = np.where(枢轴数 > 0, 250.0 / 枢轴数, np.nan)

    # ---------------- B 族：流动性与容量
    amt = g["amount"].astype(float)
    out["ADV20"] = amt.rolling(20).mean().to_numpy()
    out["ADV60"] = amt.rolling(60).mean().to_numpy()
    out["turn20"] = g["turn"].astype(float).rolling(20).mean().to_numpy()
    out["float_mktcap"] = g["float_mktcap"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        illiq = (r1.abs() / amt.where(amt > 0)) * 1e8
    out["AMIHUD"] = illiq.rolling(20).mean().to_numpy()

    # ---------------- C 族：波动与形态
    out["SD20"] = r1.rolling(20).std().to_numpy()
    out["SD60"] = r1.rolling(60).std().to_numpy()
    h, l, pc = (g["hfq_high"].astype(float), g["hfq_low"].astype(float),
                g["hfq_close"].astype(float).shift(1))
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()   # Wilder
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ATRP"] = (atr14 / c).to_numpy()
        out["RANGE20"] = ((g["high"].astype(float) - g["low"].astype(float))
                          / g["preclose"].astype(float).replace(0, np.nan)
                          ).rolling(20).mean().to_numpy()
        跳空 = ((g["open"].astype(float) / g["preclose"].astype(float).replace(0, np.nan)) - 1).abs()
    out["GAPFREQ"] = (跳空 > 0.02).rolling(250).mean().to_numpy()
    涨停 = g["is_limit_up_close"].fillna(False).astype(float)
    out["LUFREQ"] = 涨停.rolling(250).mean().to_numpy()
    一字 = ((g["high"].astype(float) == g["low"].astype(float))
            & g["is_limit_up_close"].fillna(False)).astype(float)
    out["YZFREQ"] = 一字.rolling(250).mean().to_numpy()

    # ---------------- F 族：制度与属性
    out["BOARD"] = g["board"].to_numpy()
    out["ST_PIT"] = g["is_st"].to_numpy()
    out["AGE"] = g["td_since_list"].to_numpy()
    out["pe_ttm"] = g["pe_ttm"].to_numpy(float)
    out["pb_mrq"] = g["pb_mrq"].to_numpy(float)
    out["close_raw"] = g["close"].to_numpy(float)      # R-D1 股息率用实际价格
    return out


# ================================================================ S1
def 阶段1():
    if S1文件.exists():
        print(f"  [跳过] {S1文件.name} 已存在")
        return
    con = duckdb.connect(); con.execute("PRAGMA threads=4")
    符号 = [r[0] for r in con.execute(
        f"SELECT DISTINCT symbol FROM read_parquet('{面板文件}') ORDER BY symbol").fetchall()]
    print(f"  证券数 {len(符号)}")

    块大小 = 400
    片段 = []
    停牌统计 = []
    for b in range(0, len(符号), 块大小):
        批 = 符号[b:b + 块大小]
        占位 = ",".join(f"'{s}'" for s in 批)
        df = con.execute(f"""
            SELECT symbol, date, board, td_since_list, open, high, low, close, preclose,
                   amount, turn, is_st, is_suspended, hfq_high, hfq_low, hfq_close,
                   pe_ttm, pb_mrq, float_mktcap, is_limit_up_close
            FROM read_parquet('{面板文件}')
            WHERE symbol IN ({占位})
            ORDER BY symbol, date
        """).fetch_df()

        # 停牌频率：须在含停牌的完整序列上算
        df["_susp"] = df["is_suspended"].fillna(False).astype(float)
        df["SUSPFREQ"] = (df.groupby("symbol", sort=False)["_susp"]
                            .transform(lambda s: s.rolling(250).mean()))
        susp映射 = df.set_index(["symbol", "date"])["SUSPFREQ"]

        交易 = df[~df["is_suspended"].fillna(False)].copy()
        停牌统计.append({"块": b // 块大小, "全部行": len(df), "交易行": len(交易)})

        res = [个股特征(g) for _, g in 交易.groupby("symbol", sort=False)]
        if res:
            r = pd.concat(res, ignore_index=True)
            r["SUSPFREQ"] = susp映射.reindex(
                pd.MultiIndex.from_arrays([r["symbol"], r["date"]])).to_numpy()
            片段.append(r)
        print(f"    块 {b//块大小 + 1}/{(len(符号)-1)//块大小 + 1}  "
              f"证券{len(批)}  交易行{len(交易):,}", flush=True)

    全 = pd.concat(片段, ignore_index=True)
    全.to_parquet(S1文件, compression="zstd", index=False)
    print(f"  S1 完成: {len(全):,} 行 -> {S1文件.name} ({S1文件.stat().st_size/1e6:.1f} MB)")
    con.close()


# ================================================================ S2
def 阶段2():
    """baostock 独有的 ps_ttm / pcf_ncf_ttm（母项目冻结面板没有这两列）。只读。"""
    if S2文件.exists():
        print(f"  [跳过] {S2文件.name} 已存在")
        return
    con = duckdb.connect()
    映射 = con.execute(
        f"SELECT symbol, bs_code FROM read_parquet('{冻结目录}/symbols.parquet')").fetch_df()
    con.close()
    bs2sym = dict(zip(映射["bs_code"], 映射["symbol"]))

    sq = sqlite3.connect(f"file:{BAOSTOCK_DB}?mode=ro", uri=True)
    sq.execute("PRAGMA query_only=1")
    df = pd.read_sql_query(f"""
        SELECT code, date, ps_ttm, pcf_ncf_ttm
        FROM stock_kline_d
        WHERE adjustflag = 3 AND date >= '{研究期起}' AND date <= '{研究期止}'
    """, sq)
    sq.close()
    df["symbol"] = df["code"].map(bs2sym)
    未映射 = int(df["symbol"].isna().sum())
    df = df.dropna(subset=["symbol"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["symbol", "date", "ps_ttm", "pcf_ncf_ttm"]]
    df.to_parquet(S2文件, compression="zstd", index=False)
    print(f"  S2 完成: {len(df):,} 行  未映射行 {未映射:,} -> {S2文件.name}")


# ================================================================ S3
def 阶段3() -> dict:
    """横截面分位（§7.1 R-V1/R-V2/R-V4）+ 分红特征（R-D1/R-D2/R-D3）。"""
    con = duckdb.connect(); con.execute("PRAGMA threads=4")

    # 分红：母项目同花顺事件表（§14.1 C1）
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW 分红 AS
        SELECT symbol, ex_date, dividend_per_share, per_share_bonus, rights_ratio
        FROM read_parquet('{冻结目录}/corp_actions.parquet')
        WHERE ex_date <= DATE '{研究期止}'
    """)

    估值分位 = """
    CASE WHEN {c} IS NULL THEN -2            -- 缺失组
         WHEN {c} <= 0     THEN -1            -- 负值/非正组（R-V2：禁止当极值排序）
         ELSE ntile(5) OVER (PARTITION BY date, ({c} > 0) ORDER BY {c}) - 1
    END"""

    sql = f"""
    WITH s1 AS (
      SELECT * FROM read_parquet('{S1文件}')
      WHERE date >= DATE '{研究期起}'
    ), j AS (
      SELECT s1.*, s2.ps_ttm, s2.pcf_ncf_ttm
      FROM s1 LEFT JOIN read_parquet('{S2文件}') s2 USING (symbol, date)
    ), div36 AS (
      SELECT j.symbol, j.date,
             count(d.ex_date) AS DIVCNT36,
             count(DISTINCT CASE WHEN d.dividend_per_share > 0
                                 THEN date_part('year', d.ex_date) END) AS DIVYEARS,
             sum(CASE WHEN d.dividend_per_share > 0 THEN d.dividend_per_share ELSE 0 END)
               AS 近36月每股股利,
             count(CASE WHEN d.per_share_bonus > 0 OR d.rights_ratio > 0 THEN 1 END)
               AS 送转配股次数36
      FROM j LEFT JOIN 分红 d
        ON d.symbol = j.symbol
       AND d.ex_date <  j.date                       -- PIT：严格早于当日
       AND d.ex_date >= j.date - INTERVAL 36 MONTH
      GROUP BY j.symbol, j.date
    )
    SELECT j.*,
           {估值分位.format(c='j.pe_ttm')}       AS PE_q,
           {估值分位.format(c='j.pb_mrq')}       AS PB_q,
           {估值分位.format(c='j.ps_ttm')}       AS PS_q,
           {估值分位.format(c='j.pcf_ncf_ttm')}  AS PCF_q,
           CASE WHEN j.pcf_ncf_ttm IS NULL THEN NULL
                WHEN j.pcf_ncf_ttm <= 0 THEN 1 ELSE 0 END AS PCF_负现金流,
           COALESCE(d.DIVCNT36, 0)               AS DIVCNT36,
           COALESCE(d.DIVYEARS, 0)               AS DIVYEARS,
           CASE WHEN j.close_raw > 0
                THEN COALESCE(d.近36月每股股利, 0) / 3.0 / j.close_raw
                END                              AS 股息率_年化近似,
           COALESCE(d.送转配股次数36, 0)          AS 送转配股次数36
    FROM j LEFT JOIN div36 d USING (symbol, date)
    """
    con.execute(f"COPY ({sql}) TO '{完整文件}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    # 画像
    基础 = con.execute(f"""
        SELECT count(*), count(DISTINCT symbol), min(date), max(date)
        FROM read_parquet('{完整文件}')""").fetchone()

    特征列 = ["ER_60", "ER_120", "VR_5", "VR_20", "AC1_60", "AC1_120", "R2_60", "R2_120",
              "SEGLEN", "ADV20", "ADV60", "turn20", "float_mktcap", "AMIHUD",
              "SD20", "SD60", "ATRP", "RANGE20", "GAPFREQ", "LUFREQ", "YZFREQ",
              "AGE", "SUSPFREQ", "股息率_年化近似"]
    缺失 = con.execute(f"""
        SELECT {', '.join(f"avg(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0.0 END)" for c in 特征列)}
        FROM read_parquet('{完整文件}')""").fetchone()

    分位画像 = {}
    for c in ["PE_q", "PB_q", "PS_q", "PCF_q"]:
        rows = con.execute(f"""
            SELECT {c} AS q, count(*) n FROM read_parquet('{完整文件}')
            GROUP BY {c} ORDER BY {c}""").fetchall()
        分位画像[c] = {("缺失组" if q == -2 else "非正组" if q == -1 else f"Q{q+1}"): int(n)
                      for q, n in rows}

    A族分布 = con.execute(f"""
        SELECT
          median(ER_60), median(ER_120), median(VR_5), median(VR_20),
          median(AC1_60), median(AC1_120), median(R2_60), median(SEGLEN)
        FROM read_parquet('{完整文件}')""").fetchone()

    分红覆盖 = con.execute(f"""
        SELECT count(DISTINCT symbol) FROM 分红 WHERE dividend_per_share > 0
    """).fetchone()[0]
    面板证券 = con.execute(f"""
        SELECT count(DISTINCT symbol) FROM read_parquet('{完整文件}')""").fetchone()[0]

    画像 = {
        "行数": int(基础[0]), "证券数": int(基础[1]),
        "首日": str(基础[2]), "末日": str(基础[3]),
        "特征缺失率": {c: float(v) for c, v in zip(特征列, 缺失)},
        "估值分位分组计数": 分位画像,
        "A族中位数": dict(zip(
            ["ER_60", "ER_120", "VR_5", "VR_20", "AC1_60", "AC1_120", "R2_60", "SEGLEN"],
            [float(x) if x is not None else None for x in A族分布])),
        "分红覆盖": {"有现金分红记录的证券数": int(分红覆盖), "面板证券数": int(面板证券),
                    "覆盖率": 分红覆盖 / 面板证券},
    }
    con.close()
    return 画像


def main():
    证据目录.mkdir(parents=True, exist_ok=True)
    print("[S1] 个股路径特征（A/B/C/F 族）")
    阶段1()
    print("\n[S2] baostock 独有估值字段")
    阶段2()
    print("\n[S3] 横截面分位 + 分红特征")
    画像 = 阶段3()

    print(f"  完整特征面板: {画像['行数']:,} 行  {画像['证券数']} 只  "
          f"{画像['首日']}..{画像['末日']}")
    print("  A 族中位数（母项目从未测过的维度）:")
    for k, v in 画像["A族中位数"].items():
        print(f"    {k:10s} {v:.4f}" if v is not None else f"    {k:10s} None")
    print("  估值分位分组:")
    for c, d in 画像["估值分位分组计数"].items():
        print(f"    {c}: {d}")
    print(f"  分红覆盖率: {画像['分红覆盖']['覆盖率']:.2%} "
          f"({画像['分红覆盖']['有现金分红记录的证券数']}/{画像['分红覆盖']['面板证券数']})")
    高缺失 = {k: v for k, v in 画像["特征缺失率"].items() if v > 0.15}
    if 高缺失:
        print("  缺失率 >15% 的特征:")
        for k, v in sorted(高缺失.items(), key=lambda x: -x[1]):
            print(f"    {k:18s} {v:.2%}")

    p = 证据目录 / "特征面板画像.json"
    p.write_text(json.dumps({
        "生成时间": dt.datetime.now().isoformat(timespec="seconds"),
        "脚本": "02_代码/03_特征面板.py",
        "预注册依据": ["§7 特征清单", "§7.1 估值与分红硬规则", "§14.1 C1/C4"],
        "画像": 画像,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n证据已写入: {p}")


if __name__ == "__main__":
    main()
