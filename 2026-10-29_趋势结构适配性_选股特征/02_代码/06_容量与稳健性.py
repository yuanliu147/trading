#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
容量 / 成本 / 参数平台 / 去尾 / 排除清单
项目：2026-10-29_趋势结构适配性_选股特征
预注册：§9 门槛6（容量）门槛7（参数平台）、§8.4（去尾）、§11.1（排除清单优于买入清单）

检验对象：
  候选族  = R1_NEWLOW_N55 × 高波动分位（SD20_Q4 / SD60_Q5 / ATRP_Q5 / RANGE20_Q4 / RANGE20_Q5）
            —— 等权口径下界跨 0、沪深300 口径通过 5 门槛的唯一发现
  对照族  = S1_BREAK_N55 / S2_VCB_N20_P30（趋势类，用于对比）

产出：03_结果/容量与稳健性_*.csv、04_证据/容量与稳健性.json
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

本项目 = Path("/Users/admin/WorkSpace/trading/2026-10-29_趋势结构适配性_选股特征")
数据目录 = 本项目 / "01_数据"
结果目录 = 本项目 / "03_结果"
证据目录 = 本项目 / "04_证据"

特征面板 = 数据目录 / "特征_完整.parquet"
事件面板 = 数据目录 / "事件_前向收益.parquet"
劈半面板 = 数据目录 / "横截面劈半.parquet"

往返固定摩擦 = 0.025 * 2 / 100 + 0.1 / 100 + 0.002 * 2 / 100
持有期 = [5, 10, 20, 40, 60]
候选定义 = [("SD20", 4), ("SD60", 5), ("ATRP", 5), ("RANGE20", 4), ("RANGE20", 5)]
候选信号 = "R1_NEWLOW_N55"
对照信号 = ["S1_BREAK_N55", "S2_VCB_N20_P30"]
BOOT = 1000
种子 = 20261029


def 连接():
    con = duckdb.connect(); con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='6GB'")
    return con


def 建候选事件(con):
    """候选 + 对照事件，带 ADV20/流通市值/波动/实际价格/半区，供容量与画像用。"""
    信号列表 = ",".join(f"'{s}'" for s in [候选信号] + 对照信号)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW q AS
        SELECT symbol, date,
               lead(date) OVER (PARTITION BY symbol ORDER BY date) AS 下一交易日,
               ntile(5) OVER (PARTITION BY date ORDER BY SD20)    AS SD20_q,
               ntile(5) OVER (PARTITION BY date ORDER BY SD60)    AS SD60_q,
               ntile(5) OVER (PARTITION BY date ORDER BY ATRP)    AS ATRP_q,
               ntile(5) OVER (PARTITION BY date ORDER BY RANGE20) AS RANGE20_q,
               ntile(5) OVER (PARTITION BY date ORDER BY ADV20)   AS ADV20_q,
               ADV20, float_mktcap, SD20, SD60, ATRP, RANGE20,
               close_raw, ST_PIT, BOARD, AGE, turn20
        FROM read_parquet('{特征面板}')
    """)
    cols = ", ".join(
        f"q.{f}_q = {qq} AS 在_{f}_Q{qq}" for f, qq in dict.fromkeys(候选定义))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ev AS
        SELECT e.symbol, e.信号代码, e.信号日,
               {', '.join(f'e.超额EW_h{h} AS 超额EW_h{h}' for h in 持有期)},
               {', '.join(f'e.毛收益_h{h} AS 毛收益_h{h}' for h in 持有期)},
               {', '.join(f'e.基准EW_h{h} AS 基准EW_h{h}' for h in 持有期)},
               q.ADV20, q.float_mktcap, q.SD20, q.SD60, q.ATRP, q.RANGE20,
               q.ADV20_q, q.close_raw, q.ST_PIT, q.BOARD, q.AGE, q.turn20,
               {cols}, s.半区
        FROM read_parquet('{事件面板}') e
        JOIN q ON q.symbol = e.symbol AND q.下一交易日 = e.信号日
        JOIN read_parquet('{劈半面板}') s ON s.symbol = e.symbol
        WHERE e.信号代码 IN ({信号列表}) AND NOT e.作废
    """)
    n = con.execute("SELECT count(*) FROM ev").fetchone()[0]
    print(f"  候选+对照事件 {n:,}")
    return int(n)


def 聚类均值(df: pd.DataFrame, 列: str) -> dict:
    d = df.dropna(subset=[列])
    if len(d) < 50:
        return {"n": len(d), "均值": None, "下界95": None, "上界95": None}
    日 = d.groupby("信号日")[列].agg(["sum", "count"])
    n = 日["count"].sum(); m = 日["sum"].sum() / n
    月 = pd.to_datetime(日.index).to_period("M").astype(str)
    mg = pd.DataFrame({"月": 月, "s": 日["sum"].to_numpy(),
                       "c": 日["count"].to_numpy()}).groupby("月").sum()
    rng = np.random.default_rng(种子)
    S, N = mg["s"].to_numpy(), mg["c"].to_numpy()
    idx = rng.integers(0, len(S), size=(BOOT, len(S)))
    bs = S[idx].sum(1) / np.maximum(N[idx].sum(1), 1e-9)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": int(n), "均值": float(m), "下界95": float(lo), "上界95": float(hi)}


def main():
    结果目录.mkdir(parents=True, exist_ok=True)
    con = 连接()
    print("[1/5] 建候选事件")
    建候选事件(con)

    子集表达 = {f"{f}_Q{q}": f"在_{f}_Q{q}" for f, q in dict.fromkeys(候选定义)}

    # ---------- 2. 参数平台：持有期 × 滑点（门槛7）
    print("\n[2/5] 参数平台：持有期 × 滑点")
    平台 = []
    for 名, 列 in 子集表达.items():
        d = con.execute(f"""SELECT * FROM ev
            WHERE 信号代码='{候选信号}' AND {列}""").fetch_df()
        for h in 持有期:
            for s, tag in [(0.0, "0bp"), (0.0005, "5bp"), (0.0010, "10bp"), (0.0020, "20bp")]:
                d["_x"] = (d[f"毛收益_h{h}"] - (往返固定摩擦 + 2 * s)) - d[f"基准EW_h{h}"]
                r = 聚类均值(d, "_x")
                平台.append({"子集": 名, "信号": 候选信号, "持有期": h, "滑点": tag,
                            **{f"超额{k}": v for k, v in r.items()}})
    for 信号 in 对照信号:
        d = con.execute(f"SELECT * FROM ev WHERE 信号代码='{信号}'").fetch_df()
        for h in 持有期:
            d["_x"] = (d[f"毛收益_h{h}"] - (往返固定摩擦 + 2 * 0.0005)) - d[f"基准EW_h{h}"]
            r = 聚类均值(d, "_x")
            平台.append({"子集": "全样本", "信号": 信号, "持有期": h, "滑点": "5bp",
                        **{f"超额{k}": v for k, v in r.items()}})
    平台df = pd.DataFrame(平台)
    平台df.to_csv(结果目录 / "容量与稳健性_参数平台.csv", index=False, encoding="utf-8-sig")
    piv = 平台df[(平台df["信号"] == 候选信号) & (平台df["滑点"] == "5bp")].pivot(
        index="子集", columns="持有期", values="超额均值")
    print("  候选族 滑点5bp 的持有期平台（等权超额）:")
    print(piv.to_string(float_format=lambda x: f"{x:+.3%}"))

    # ---------- 3. 去尾稳健性
    print("\n[3/5] 去尾稳健性（剔除超额最大前1% / 前5%）")
    去尾 = []
    for 名, 列 in 子集表达.items():
        d = con.execute(f"""SELECT 信号日, 超额EW_h20 FROM ev
            WHERE 信号代码='{候选信号}' AND {列} AND 超额EW_h20 IS NOT NULL""").fetch_df()
        base = 聚类均值(d, "超额EW_h20")
        行 = {"子集": 名, "原始均值": base["均值"], "原始下界95": base["下界95"], "n": base["n"]}
        for p in (0.01, 0.05):
            阈 = d["超额EW_h20"].quantile(1 - p)
            r = 聚类均值(d[d["超额EW_h20"] < 阈], "超额EW_h20")
            行[f"去前{int(p*100)}%均值"] = r["均值"]
            行[f"去前{int(p*100)}%下界95"] = r["下界95"]
        去尾.append(行)
    去尾df = pd.DataFrame(去尾)
    去尾df.to_csv(结果目录 / "容量与稳健性_去尾.csv", index=False, encoding="utf-8-sig")
    print(去尾df.to_string(index=False, float_format=lambda x: f"{x:+.3%}"))

    # ---------- 4. 容量（门槛6）
    print("\n[4/5] 容量：资金 × 仓位数 × ADV 参与率")
    容量 = []
    for 名, 列 in 子集表达.items():
        d = con.execute(f"""SELECT 信号日, ADV20, 超额EW_h20, close_raw, SD20 FROM ev
            WHERE 信号代码='{候选信号}' AND {列} AND ADV20 IS NOT NULL""").fetch_df()
        每日事件 = d.groupby("信号日").size()
        for 资金 in (1e7, 5e7, 1e8):
            for 仓位数 in (10, 20):
                单笔 = 资金 / 仓位数
                for 参与率 in (0.01, 0.05, 0.10):
                    可 = d["ADV20"] * 参与率 >= 单笔
                    r = 聚类均值(d[可], "超额EW_h20") if 可.sum() > 50 else \
                        {"n": int(可.sum()), "均值": None, "下界95": None, "上界95": None}
                    # 平方根冲击成本（k 披露为 0.5，单边）
                    参与 = np.minimum(单笔 / d.loc[可, "ADV20"], 1.0) if 可.sum() else pd.Series(dtype=float)
                    冲击 = (0.5 * d.loc[可, "SD20"].abs() * np.sqrt(参与)).mean() \
                        if 可.sum() and "SD20" in d.columns else np.nan
                    容量.append({
                        "子集": 名, "资金": 资金, "仓位数": 仓位数, "单笔金额": 单笔,
                        "ADV参与率上限": 参与率,
                        "可部署事件数": int(可.sum()), "总事件数": int(len(d)),
                        "可部署率": float(可.mean()),
                        "日均可部署事件": float(可.sum() / max(len(每日事件), 1)),
                        "可部署子集超额均值": r["均值"], "可部署子集下界95": r["下界95"]})
    容量df = pd.DataFrame(容量)
    容量df.to_csv(结果目录 / "容量与稳健性_容量.csv", index=False, encoding="utf-8-sig")
    sel = 容量df[(容量df["仓位数"] == 10) & (容量df["ADV参与率上限"] == 0.05)]
    print("  仓位10 / ADV参与率5% 下的可部署率:")
    print(sel.pivot(index="子集", columns="资金", values="可部署率").to_string(
        float_format=lambda x: f"{x:.1%}"))

    # ---------- 5. 子集画像 + 排除清单（§11.1）
    print("\n[5/5] 子集画像与排除清单")
    画像 = con.execute(f"""
        SELECT '候选_高波动创新低' AS 组, count(*) n,
               median(ADV20)/1e4 AS 成交额中位_万, median(float_mktcap)/1e8 AS 流通市值中位_亿,
               median(close_raw) AS 价格中位, median(SD20) AS 日波动中位,
               avg(CASE WHEN ST_PIT=1 THEN 1.0 ELSE 0 END) AS ST占比,
               median(AGE) AS 上市天数中位, median(turn20) AS 换手率中位,
               avg(CASE WHEN ADV20_q=1 THEN 1.0 ELSE 0 END) AS 落在最低流动性分位占比
        FROM ev WHERE 信号代码='{候选信号}' AND (在_RANGE20_Q5 OR 在_ATRP_Q5)
        UNION ALL
        SELECT '对照_趋势突破全样本', count(*),
               median(ADV20)/1e4, median(float_mktcap)/1e8, median(close_raw), median(SD20),
               avg(CASE WHEN ST_PIT=1 THEN 1.0 ELSE 0 END), median(AGE), median(turn20),
               avg(CASE WHEN ADV20_q=1 THEN 1.0 ELSE 0 END)
        FROM ev WHERE 信号代码='S1_BREAK_N55'
    """).fetch_df()
    画像.to_csv(结果目录 / "容量与稳健性_子集画像.csv", index=False, encoding="utf-8-sig")
    print(画像.to_string(index=False))

    # 排除清单：趋势突破里最差的流动性分位，剔除后全样本是否改善
    排除 = []
    for 信号 in 对照信号:
        d = con.execute(f"""SELECT 信号日, 超额EW_h20, ADV20_q FROM ev
            WHERE 信号代码='{信号}' AND 超额EW_h20 IS NOT NULL""").fetch_df()
        全 = 聚类均值(d, "超额EW_h20")
        for 剔 in ([5], [4, 5], [3, 4, 5]):
            r = 聚类均值(d[~d["ADV20_q"].isin(剔)], "超额EW_h20")
            排除.append({"信号": 信号, "剔除ADV分位": str(剔),
                        "全样本均值": 全["均值"], "剔除后均值": r["均值"],
                        "改善": (r["均值"] - 全["均值"]) if r["均值"] is not None else None,
                        "剔除后下界95": r["下界95"], "剔除后n": r["n"]})
    排除df = pd.DataFrame(排除)
    排除df.to_csv(结果目录 / "容量与稳健性_排除清单.csv", index=False, encoding="utf-8-sig")
    print("\n  排除清单（剔除高成交额分位后，趋势突破是否改善）:")
    print(排除df.to_string(index=False, float_format=lambda x: f"{x:+.4%}"))

    (证据目录 / "容量与稳健性.json").write_text(json.dumps({
        "生成时间": dt.datetime.now().isoformat(timespec="seconds"),
        "脚本": "02_代码/06_容量与稳健性.py",
        "预注册依据": ["§9 门槛6", "§9 门槛7", "§8.4 去尾", "§11.1 排除清单"],
        "冲击成本模型": "平方根律 impact = k·σ·sqrt(单笔/ADV)，k=0.5 单边（披露值，非拟合）",
        "参数平台": 平台df.to_dict("records"),
        "去尾": 去尾df.to_dict("records"),
        "容量": 容量df.to_dict("records"),
        "子集画像": 画像.to_dict("records"),
        "排除清单": 排除df.to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n已写入 03_结果/容量与稳健性_*.csv 与 04_证据/容量与稳健性.json")
    con.close()


if __name__ == "__main__":
    main()
