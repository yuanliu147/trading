#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
条件适配性检验（本研究核心）
项目：2026-10-29_趋势结构适配性_选股特征
预注册：§8.2 分层、§8.3 配对替身、§8.4 统计、§9 门槛、§10 多重性、§12 失败解剖

核心逻辑：
  门槛1 事件层为正   —— 子集内超额的日期聚类 95% 下界 > 0
  门槛2 信号有增量   —— 配对替身差值 Δ 的 95% 下界 > 0   ← 最关键，区分 H-A 与 H-C
  门槛3 跨时代同号   —— E1-E4 四段 Δ 符号一致
  门槛4 横截面留出   —— B 半区同号且下界 > 0
  门槛5 多重性       —— BH-FDR q<=0.10

产出：03_结果/条件分层_全表.csv、03_结果/条件分层_汇总.json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

本项目 = Path("/Users/admin/WorkSpace/trading/2026-10-29_趋势结构适配性_选股特征")
数据目录 = 本项目 / "01_数据"
结果目录 = 本项目 / "03_结果"
证据目录 = 本项目 / "04_证据"

# 基准口径（§6.1 并列报告，不择优）：EW = 等权市场，300 = 沪深300
基准 = sys.argv[1] if len(sys.argv) > 1 else "EW"
assert 基准 in ("EW", "300"), "基准只能是 EW 或 300"
超额列 = f"超额{基准}_h20"
后缀 = "" if 基准 == "EW" else "_沪深300"

特征面板 = 数据目录 / "特征_完整.parquet"
事件面板 = 数据目录 / "事件_前向收益.parquet"
精简事件 = 数据目录 / f"事件_精简_h20_{基准}.parquet"
劈半面板 = 数据目录 / "横截面劈半.parquet"

主持有期 = 20
最小事件 = 200          # §8.2
最小信号日 = 100        # §8.2
BOOT次数 = 1000
BOOT种子 = 20261029

时代 = {"E1": ("2010-01-01", "2012-12-31"), "E2": ("2013-01-01", "2015-12-31"),
        "E3": ("2016-01-01", "2017-12-31"), "E4": ("2018-01-01", "2020-12-31")}

# §7 特征清单。连续特征走按日横截面五分位；其余为已分组/类别
连续特征 = [
    # A 族 趋势可交易性（母项目从未测过）
    "ER_60", "ER_120", "VR_5", "VR_20", "AC1_60", "AC1_120", "R2_60", "R2_120", "SEGLEN",
    # B 族 流动性与容量
    "ADV20", "ADV60", "turn20", "float_mktcap", "AMIHUD",
    # C 族 波动与形态
    "SD20", "SD60", "ATRP", "RANGE20", "GAPFREQ", "LUFREQ", "YZFREQ",
    # E 族 分红（§7.1 R-D1）
    "股息率_年化近似",
    # F 族
    "AGE", "SUSPFREQ",
]
已分组特征 = ["PE_q", "PB_q", "PS_q", "PCF_q"]       # D 族，§7.1 R-V1 已分组
类别特征 = ["BOARD", "ST_PIT", "PCF_负现金流", "DIVCNT36", "DIVYEARS"]

族映射 = {}
for f in ["ER_60", "ER_120", "VR_5", "VR_20", "AC1_60", "AC1_120", "R2_60", "R2_120", "SEGLEN"]:
    族映射[f] = "A_趋势可交易性"
for f in ["ADV20", "ADV60", "turn20", "float_mktcap", "AMIHUD"]:
    族映射[f] = "B_流动性与容量"
for f in ["SD20", "SD60", "ATRP", "RANGE20", "GAPFREQ", "LUFREQ", "YZFREQ"]:
    族映射[f] = "C_波动与形态"
for f in ["PE_q", "PB_q", "PS_q", "PCF_q", "PCF_负现金流"]:
    族映射[f] = "D_估值"
for f in ["股息率_年化近似", "DIVCNT36", "DIVYEARS"]:
    族映射[f] = "E_分红"
for f in ["BOARD", "ST_PIT", "AGE", "SUSPFREQ"]:
    族映射[f] = "F_制度与属性"


def 连接():
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='6GB'")
    return con


def 建分位视图(con, 特征: str, 类型: str):
    """按日横截面分位（§8.2），并给出 下一交易日 用于把特征对齐到 T-1。"""
    if 类型 == "连续":
        表达 = f"CASE WHEN {特征} IS NULL THEN NULL ELSE ntile(5) OVER (PARTITION BY date ORDER BY {特征}) END"
    elif 类型 == "已分组":
        表达 = f"CASE WHEN {特征} IS NULL THEN NULL ELSE {特征} + 1 END"   # -2/-1 -> -1/0，正组 1..5
    else:
        表达 = f"CAST({特征} AS VARCHAR)"
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW qv AS
        SELECT symbol, date,
               lead(date) OVER (PARTITION BY symbol ORDER BY date) AS 下一交易日,
               {表达} AS 分位
        FROM read_parquet('{特征面板}')
    """)


def 建精简事件(con):
    """把 2GB 事件表预先做成"真实信号 + 已配对替身超额 + 半区"的精简表。
    33 个特征各扫一遍 2GB 原表太慢，故只做一次配对与裁剪。"""
    if 精简事件.exists():
        print(f"  [复用] {精简事件.name}")
    else:
        con.execute(f"""
            COPY (
              WITH ev AS (
                SELECT symbol, 信号代码, 信号日, {超额列} AS 超额
                FROM read_parquet('{事件面板}')
                WHERE 信号代码 NOT LIKE 'P0!_%' ESCAPE '!'
                  AND NOT 作废 AND {超额列} IS NOT NULL
              ), pl AS (
                SELECT symbol, replace(信号代码, 'P0_', '') AS 信号代码,
                       配对键, {超额列} AS 超额替身
                FROM read_parquet('{事件面板}')
                WHERE 信号代码 LIKE 'P0!_%' ESCAPE '!'
                  AND NOT 作废 AND {超额列} IS NOT NULL
              )
              SELECT ev.symbol, ev.信号代码, ev.信号日,
                     ev.超额, pl.超额替身, s.半区
              FROM ev
              LEFT JOIN pl ON pl.symbol = ev.symbol
                          AND pl.信号代码 = ev.信号代码
                          AND pl.配对键 = ev.信号日
              JOIN read_parquet('{劈半面板}') s ON s.symbol = ev.symbol
            ) TO '{精简事件}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        print(f"  已建 {精简事件.name} ({精简事件.stat().st_size/1e6:.1f} MB)")
    n, np_ = con.execute(f"""
        SELECT count(*), count(超额替身) FROM read_parquet('{精简事件}')""").fetchone()
    print(f"  真实事件 {n:,}  其中已配对替身 {np_:,} ({np_/n:.2%})")
    return {"真实事件数": int(n), "已配对数": int(np_), "配对率": np_ / n}


def 取日级(con) -> pd.DataFrame:
    """按 (信号代码, 分位, 半区, 信号日) 返回日级聚合量，供 Python 做聚类 SE 与 bootstrap。"""
    return con.execute(f"""
        SELECT e.信号代码, q.分位 AS 分位, e.半区, e.信号日,
               count(*)                                   AS n,
               sum(e.超额)                                AS s_x,
               count(e.超额替身)                          AS n_p,
               sum(CASE WHEN e.超额替身 IS NOT NULL
                        THEN e.超额 - e.超额替身 END)        AS s_d
        FROM read_parquet('{精简事件}') e
        JOIN qv q ON q.symbol = e.symbol AND q.下一交易日 = e.信号日
        WHERE q.分位 IS NOT NULL
        GROUP BY e.信号代码, q.分位, e.半区, e.信号日
    """).fetch_df()


def 聚类统计(日级: pd.DataFrame, 列和: str, 列n: str) -> dict:
    """日期聚类均值与 CR0 标准误 + 按月分块 bootstrap 95% 区间（§8.4）。"""
    n = 日级[列n].sum()
    if n < 1:
        return {"n": 0, "n_days": 0, "均值": None, "聚类SE": None,
                "t": None, "下界95": None, "上界95": None}
    m = 日级[列和].sum() / n
    # CR0：sum_d (sum_i (x_i - m))^2 / n^2
    日偏差 = 日级[列和] - m * 日级[列n]
    se = np.sqrt((日偏差 ** 2).sum()) / n
    # 按月分块 bootstrap
    月 = pd.to_datetime(日级["信号日"]).dt.to_period("M").astype(str).to_numpy()
    月组 = pd.DataFrame({"月": 月, "s": 日级[列和].to_numpy(), "n": 日级[列n].to_numpy()}) \
        .groupby("月", sort=True).sum()
    rng = np.random.default_rng(BOOT种子)
    S, N = 月组["s"].to_numpy(), 月组["n"].to_numpy()
    k = len(S)
    if k >= 3:
        idx = rng.integers(0, k, size=(BOOT次数, k))
        bs = S[idx].sum(axis=1) / np.maximum(N[idx].sum(axis=1), 1e-9)
        lo, hi = np.percentile(bs, [2.5, 97.5])
    else:
        lo = hi = None
    return {"n": int(n), "n_days": int((日级[列n] > 0).sum()),
            "均值": float(m), "聚类SE": float(se),
            "t": float(m / se) if se > 0 else None,
            "下界95": None if lo is None else float(lo),
            "上界95": None if hi is None else float(hi)}


def 分析一个特征(con, 特征: str, 类型: str) -> list[dict]:
    建分位视图(con, 特征, 类型)
    日级 = 取日级(con)
    if len(日级) == 0:
        return []
    日级["信号日"] = pd.to_datetime(日级["信号日"])
    行 = []
    for (代码, 分位), g_all in 日级.groupby(["信号代码", "分位"], sort=True):
        g = g_all.groupby("信号日", as_index=False)[["n", "s_x", "n_p", "s_d"]].sum()
        超额 = 聚类统计(g, "s_x", "n")
        增量 = 聚类统计(g, "s_d", "n_p")
        if 超额["n"] < 最小事件 or 超额["n_days"] < 最小信号日:
            判定 = "F7_样本不足"
        else:
            判定 = None

        rec = {"特征": 特征, "特征族": 族映射.get(特征, "?"), "信号代码": 代码,
               "分位": str(分位), "样本判定": 判定,
               "事件数": 超额["n"], "信号日数": 超额["n_days"],
               "超额均值": 超额["均值"], "超额聚类SE": 超额["聚类SE"],
               "超额t": 超额["t"], "超额下界95": 超额["下界95"], "超额上界95": 超额["上界95"],
               "配对数": 增量["n"], "增量Δ均值": 增量["均值"],
               "增量Δ聚类SE": 增量["聚类SE"], "增量Δt": 增量["t"],
               "增量Δ下界95": 增量["下界95"], "增量Δ上界95": 增量["上界95"]}

        # 分时代（§5.3）与分半区（§5.2）
        for 名, (d0, d1) in 时代.items():
            sub = g_all[(g_all["信号日"] >= d0) & (g_all["信号日"] <= d1)]
            if len(sub):
                s2 = sub.groupby("信号日", as_index=False)[["n", "s_x", "n_p", "s_d"]].sum()
                st = 聚类统计(s2, "s_d", "n_p")
                rec[f"Δ均值_{名}"] = st["均值"]
                rec[f"Δ事件_{名}"] = st["n"]
            else:
                rec[f"Δ均值_{名}"] = None
                rec[f"Δ事件_{名}"] = 0
        for 半 in ("A", "B"):
            sub = g_all[g_all["半区"] == 半]
            if len(sub):
                s2 = sub.groupby("信号日", as_index=False)[["n", "s_x", "n_p", "s_d"]].sum()
                st = 聚类统计(s2, "s_d", "n_p")
                rec[f"Δ均值_半区{半}"] = st["均值"]
                rec[f"Δ下界95_半区{半}"] = st["下界95"]
                rec[f"Δ事件_半区{半}"] = st["n"]
            else:
                rec[f"Δ均值_半区{半}"] = None
                rec[f"Δ下界95_半区{半}"] = None
                rec[f"Δ事件_半区{半}"] = 0
        行.append(rec)
    return 行


def BH_FDR(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg。返回是否通过。"""
    n = len(p)
    order = np.argsort(p)
    通过 = np.zeros(n, dtype=bool)
    阈 = q * (np.arange(1, n + 1)) / n
    排序p = p[order]
    命中 = np.where(排序p <= 阈)[0]
    if len(命中):
        k = 命中.max()
        通过[order[:k + 1]] = True
    return 通过


def main():
    结果目录.mkdir(parents=True, exist_ok=True)
    con = 连接()
    print("[0] 预建精简配对事件表")
    配对信息 = 建精简事件(con)
    全行: list[dict] = []
    任务 = ([(f, "连续") for f in 连续特征]
            + [(f, "已分组") for f in 已分组特征]
            + [(f, "类别") for f in 类别特征])
    print(f"特征数 {len(任务)}（连续 {len(连续特征)} / 已分组 {len(已分组特征)} / "
          f"类别 {len(类别特征)}）")
    for i, (f, t) in enumerate(任务, 1):
        行 = 分析一个特征(con, f, t)
        全行 += 行
        print(f"  [{i}/{len(任务)}] {f:<18} {t:<5} 产出格 {len(行)}", flush=True)

    df = pd.DataFrame(全行)

    # §10 多重性：首要检验族 = 特征 × 信号配置 的"单调条件效应"，此处以
    # 每格的增量Δ 做 FDR（只对样本充足格），并如实报告检验总数
    有效 = df["样本判定"].isna() & df["增量Δt"].notna()
    from scipy import stats as sps
    df["增量Δ_p双侧"] = np.nan
    df.loc[有效, "增量Δ_p双侧"] = 2 * (1 - sps.norm.cdf(
        np.abs(df.loc[有效, "增量Δt"].to_numpy(float))))
    df["FDR通过_q010"] = False
    if 有效.sum() > 0:
        df.loc[有效, "FDR通过_q010"] = BH_FDR(
            df.loc[有效, "增量Δ_p双侧"].to_numpy(float), 0.10)

    # §9 七门槛（本脚本覆盖 1-5；6 容量、7 平台在后续脚本）
    df["门槛1_事件层为正"] = df["超额下界95"] > 0
    df["门槛2_增量为正"] = df["增量Δ下界95"] > 0
    era列 = [f"Δ均值_{k}" for k in 时代]
    def 同号(r):
        v = [r[c] for c in era列 if pd.notna(r[c])]
        return len(v) == 4 and (all(x > 0 for x in v) or all(x < 0 for x in v))
    df["门槛3_跨时代同号"] = df.apply(同号, axis=1)
    df["门槛3_四段皆正"] = df.apply(
        lambda r: all(pd.notna(r[c]) and r[c] > 0 for c in era列), axis=1)
    df["门槛4_B半区确认"] = (df["Δ下界95_半区B"] > 0)
    df["门槛5_FDR"] = df["FDR通过_q010"]
    df["前5门槛全过"] = (df["样本判定"].isna() & df["门槛1_事件层为正"]
                        & df["门槛2_增量为正"] & df["门槛3_四段皆正"]
                        & df["门槛4_B半区确认"] & df["门槛5_FDR"])

    # §12 失败解剖
    def 解剖(r):
        if pd.notna(r["样本判定"]):
            return r["样本判定"]
        if r["前5门槛全过"]:
            return "通过前5门槛"
        if pd.notna(r["超额均值"]) and pd.notna(r["增量Δ均值"]) \
                and r["超额均值"] > 0 and r["增量Δ均值"] <= 0:
            return "F4_替身可解释"
        if pd.notna(r["增量Δ均值"]) and r["增量Δ均值"] <= 0:
            return "F1_毛边际为零或负"
        if not r["门槛3_跨时代同号"]:
            return "F5_符号不稳"
        if not r["门槛4_B半区确认"]:
            return "F5_符号不稳_留出未确认"
        if not r["门槛5_FDR"]:
            return "F9_多重性淘汰"
        return "F1_毛边际为零或负"
    df["失败解剖"] = df.apply(解剖, axis=1)

    p_csv = 结果目录 / f"条件分层_全表{后缀}.csv"
    df.to_csv(p_csv, index=False, encoding="utf-8-sig")

    汇总 = {
        "生成时间": dt.datetime.now().isoformat(timespec="seconds"),
        "脚本": "02_代码/05_条件适配性检验.py",
        "预注册依据": ["§8.2", "§8.3", "§8.4", "§9", "§10", "§12"],
        "主口径": {"持有期": 主持有期, "滑点": "5bp单边",
                  "基准": "等权市场(EW)" if 基准 == "EW" else "沪深300"},
        "配对情况": 配对信息,
        "格总数": int(len(df)),
        "样本充足格数": int(有效.sum()),
        "多重性": {
            "实际检验数": int(有效.sum()),
            "预注册首要族预算": 124,
            "超出倍数": round(float(有效.sum()) / 124, 2),
            "说明": "预注册按 31特征×4信号族=124 估算；实际用 11 个信号配置而非 4 个族代表，"
                    "故检验数放大。如实报告超出倍数（母项目 18.33× 未披露是反面教材）",
        },
        "门槛通过计数": {
            "门槛1_事件层为正": int(df["门槛1_事件层为正"].sum()),
            "门槛2_增量为正": int(df["门槛2_增量为正"].sum()),
            "门槛3_四段皆正": int(df["门槛3_四段皆正"].sum()),
            "门槛4_B半区确认": int(df["门槛4_B半区确认"].sum()),
            "门槛5_FDR": int(df["门槛5_FDR"].sum()),
            "前5门槛全过": int(df["前5门槛全过"].sum()),
        },
        "失败解剖分布": {k: int(v) for k, v in
                        df["失败解剖"].value_counts().items()},
        "文件": str(p_csv),
    }
    p_json = 结果目录 / f"条件分层_汇总{后缀}.json"
    p_json.write_text(json.dumps(汇总, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n格总数 {len(df):,}  样本充足 {有效.sum():,}")
    print("门槛通过计数:")
    for k, v in 汇总["门槛通过计数"].items():
        print(f"  {k:<20} {v}")
    print("失败解剖分布:")
    for k, v in sorted(汇总["失败解剖分布"].items(), key=lambda x: -x[1]):
        print(f"  {k:<24} {v}")
    print(f"\n全表: {p_csv}\n汇总: {p_json}")
    con.close()


if __name__ == "__main__":
    main()
