#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号层 + 前向收益
项目：2026-10-29_趋势结构适配性_选股特征
预注册：§6 结果变量、§8.1 入场信号、§8.3 配对替身、§11.2 反向信号、§4.4/4.5 成本与可成交

不复用母项目代码（已披露缺陷：创业板涨跌幅全期 20%、out_date 前视、VCB 成本口径）。
本脚本重写最小可复现实现。

设计要点：
  - 逐块落盘（避免把数百万行累积在内存里，首版因此被 OOM）
  - **配对键**：替身事件携带其配对真实信号事件的信号日，使 §8.3 的严格配对可做

产出：01_数据/事件_前向收益.parquet
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

母项目 = Path("/Users/admin/WorkSpace/trading/trading_趋势结构")
冻结目录 = 母项目 / "01_数据" / "frozen"
本项目 = Path("/Users/admin/WorkSpace/trading/2026-10-29_趋势结构适配性_选股特征")
数据目录 = 本项目 / "01_数据"
证据目录 = 本项目 / "04_证据"
块目录 = 数据目录 / "_事件块"

研究期起 = "2010-01-01"
研究期止 = "2020-12-31"
面板文件 = 数据目录 / "面板_2008预热_2020.parquet"
事件文件 = 数据目录 / "事件_前向收益.parquet"

持有期 = [5, 10, 20, 40, 60]
最大顺延 = 2
往返固定摩擦 = 0.025 * 2 / 100 + 0.1 / 100 + 0.002 * 2 / 100   # §4.4 = 0.154%
随机种子 = 20261029

信号配置 = [
    ("S1_BREAK_N20",   "S1", {"N": 20}),
    ("S1_BREAK_N55",   "S1", {"N": 55}),
    ("S1_BREAK_N120",  "S1", {"N": 120}),
    ("S2_VCB_N20_P20", "S2", {"N": 20, "p": 0.2}),
    ("S2_VCB_N20_P30", "S2", {"N": 20, "p": 0.3}),
    ("S2_VCB_N55_P20", "S2", {"N": 55, "p": 0.2}),
    ("S2_VCB_N55_P30", "S2", {"N": 55, "p": 0.3}),
    ("S3_PULL_K05",    "S3", {"k": 0.5}),
    ("S3_PULL_K10",    "S3", {"k": 1.0}),
    ("S4_BRKPULL_W5",  "S4", {"N": 55, "W": 5}),
    ("S4_BRKPULL_W10", "S4", {"N": 55, "W": 10}),
]
反向配置 = [
    ("R1_NEWLOW_N20",  "R1", {"N": 20}),
    ("R1_NEWLOW_N55",  "R1", {"N": 55}),
    ("R1_NEWLOW_N120", "R1", {"N": 120}),
    ("R3_RALLY_K10",   "R3", {"k": 1.0}),
]

列名 = (["symbol", "信号代码", "信号日", "配对键", "作废", "顺延天数", "买入日"]
        + [f"毛收益_h{h}" for h in 持有期]
        + [f"基准EW_h{h}" for h in 持有期]
        + [f"基准300_h{h}" for h in 持有期])


def 生成信号(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """只用 <= 当日收盘可得数据。预热期不出事件（§5.5）。"""
    c = g["hfq_close"].astype(float)
    hi = g["hfq_high"].astype(float)
    lo = g["hfq_low"].astype(float)
    sig: dict[str, np.ndarray] = {}

    前高 = {N: c.shift(1).rolling(N).max() for N in (20, 55, 120)}
    前低 = {N: c.shift(1).rolling(N).min() for N in (20, 55, 120)}
    ma20 = c.rolling(20).mean()
    ma120 = c.rolling(120).mean()
    ma20上行 = ma20 > ma20.shift(5)
    ma20下行 = ma20 < ma20.shift(5)

    pc = c.shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    atrp = atr14 / c
    atrp_pct = atrp.rolling(120).rank(pct=True)
    偏离 = (c / ma20 - 1).abs()

    for 代码, 族, P in 信号配置:
        if 族 == "S1":
            sig[代码] = (c > 前高[P["N"]])
        elif 族 == "S2":
            sig[代码] = (atrp_pct.shift(1) <= P["p"]) & (c > 前高[P["N"]])
        elif 族 == "S3":
            贴近 = 偏离 <= P["k"] * atrp
            sig[代码] = (c > ma120) & ma20上行 & 贴近.shift(1) & (c > c.shift(1))
        elif 族 == "S4":
            N, W = P["N"], P["W"]
            突破 = (c > 前高[N])
            突破价 = c.where(突破).ffill()
            近期突破 = 突破.rolling(W + 1).max().shift(1).fillna(0).astype(bool)
            回踩过 = (lo <= 突破价 * 1.02).rolling(W).max().shift(1).fillna(0).astype(bool)
            sig[代码] = 近期突破 & 回踩过 & (c > c.shift(1)) & (c > ma20)

    for 代码, 族, P in 反向配置:
        if 族 == "R1":
            sig[代码] = (c < 前低[P["N"]])
        elif 族 == "R3":
            贴近 = 偏离 <= P["k"] * atrp
            sig[代码] = (c < ma120) & ma20下行 & 贴近.shift(1) & (c < c.shift(1))

    研究期内 = np.array([d >= dt.date(2010, 1, 1) for d in g["date"]])
    for k in list(sig):
        v = sig[k].fillna(False).to_numpy(bool)
        sig[k] = v & 研究期内
    return sig


def 前向收益(g, 位置, 配对键, 代码, mkt累, 日期索引, 出行: list):
    """§4.5 顺延 + §6.1 前向收益。买 T+b 开盘，卖 T+b+h 收盘。结果 append 到 出行。"""
    n = len(g)
    sym = g["symbol"].iat[0]
    日期 = g["date"].to_numpy()
    hfq_open = g["hfq_open"].to_numpy(float)
    hfq_close = g["hfq_close"].to_numpy(float)
    不可买 = (g["is_limit_up_open"].fillna(False).to_numpy(bool)
              | g["is_suspended"].fillna(False).to_numpy(bool)
              | (g["trade_status"].fillna(1).to_numpy(float) == 0))
    nan = float("nan")

    for idx, t in enumerate(位置):
        pk = 配对键[idx]
        b = -1
        for off in range(1, 最大顺延 + 2):
            j = t + off
            if j >= n:
                break
            if not 不可买[j]:
                b = j
                break
        if b < 0 or not np.isfinite(hfq_open[b]) or hfq_open[b] <= 0:
            出行.append((sym, 代码, 日期[t], pk, True, -1, None)
                        + (nan,) * (3 * len(持有期)))
            continue
        买价 = hfq_open[b]
        bi = 日期索引.get(日期[b])
        毛, bew, b300 = [], [], []
        for h in 持有期:
            s = b + h
            if s >= n or bi is None or bi == 0:
                毛.append(nan); bew.append(nan); b300.append(nan); continue
            毛.append(hfq_close[s] / 买价 - 1.0)
            si = 日期索引.get(日期[s])
            if si is None:
                bew.append(nan); b300.append(nan)
            else:
                bew.append(mkt累["EW"][si] / mkt累["EW"][bi - 1] - 1.0)
                b300.append(mkt累["300"][si] / mkt累["300"][bi - 1] - 1.0)
        出行.append((sym, 代码, 日期[t], pk, False, b - t - 1, 日期[b])
                    + tuple(毛) + tuple(bew) + tuple(b300))


def main():
    证据目录.mkdir(parents=True, exist_ok=True)
    if 块目录.exists():
        shutil.rmtree(块目录)
    块目录.mkdir(parents=True)

    rng = np.random.default_rng(随机种子)
    con = duckdb.connect(); con.execute("PRAGMA threads=4")

    mkt = con.execute(f"""
        SELECT date, ew_ret, hs300_ret
        FROM read_parquet('{冻结目录}/market_daily.parquet')
        WHERE date >= DATE '2008-07-01' AND date <= DATE '{研究期止}'
        ORDER BY date""").fetch_df()
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.date
    日期索引 = {d: i for i, d in enumerate(mkt["date"])}

    # —— 单位断言：冻结面板的 ew_ret / hs300_ret / pct_chg 均为【百分数】口径
    #    （实测 2010-2020 ew_ret min -9.741 / max 9.742 / sd 1.863；个股 pct_chg -5.1466
    #     对应自算小数 -0.051466）。首版未除 100，导致累乘爆炸、超额出现天文数字。
    for 名, 列 in (("ew_ret", "ew_ret"), ("hs300_ret", "hs300_ret")):
        峰 = float(np.nanmax(np.abs(mkt[列].to_numpy(float))))
        assert 峰 > 1.5, (f"{名} 最大绝对值 {峰:.4f} < 1.5，疑为小数口径；"
                          f"若源数据改为小数，必须同步去掉 /100")
        assert 峰 < 50.0, f"{名} 最大绝对值 {峰:.4f} 异常"
    mkt累 = {"EW": np.cumprod(1.0 + mkt["ew_ret"].fillna(0).to_numpy(float) / 100.0),
             "300": np.cumprod(1.0 + mkt["hs300_ret"].fillna(0).to_numpy(float) / 100.0)}
    # 累乘量级自检：11 年等权市场累计净值应落在合理区间
    assert 0.2 < mkt累["EW"][-1] / mkt累["EW"][0] < 100, \
        f"EW 累计净值 {mkt累['EW'][-1]/mkt累['EW'][0]:.3f} 异常"
    print(f"  基准单位自检通过：EW 累计 {mkt累['EW'][-1]/mkt累['EW'][0]:.3f}x，"
          f"HS300 累计 {mkt累['300'][-1]/mkt累['300'][0]:.3f}x（2008-07 起）")

    符号 = [r[0] for r in con.execute(
        f"SELECT DISTINCT symbol FROM read_parquet('{面板文件}') ORDER BY symbol").fetchall()]
    print(f"证券数 {len(符号)}")

    块大小 = 400
    统计: dict[str, int] = {}
    块数 = 0
    for b0 in range(0, len(符号), 块大小):
        批 = 符号[b0:b0 + 块大小]
        占位 = ",".join(f"'{s}'" for s in 批)
        df = con.execute(f"""
            SELECT symbol, date, hfq_open, hfq_high, hfq_low, hfq_close,
                   is_limit_up_open, is_suspended, trade_status
            FROM read_parquet('{面板文件}')
            WHERE symbol IN ({占位}) ORDER BY symbol, date""").fetch_df()
        df["date"] = pd.to_datetime(df["date"]).dt.date

        出行: list = []
        for _, g in df.groupby("symbol", sort=False):
            g = g.reset_index(drop=True)
            交易掩码 = ~(g["is_suspended"].fillna(False).to_numpy(bool))
            if 交易掩码.sum() < 300:
                continue
            sig = 生成信号(g)
            日期arr = g["date"].to_numpy()
            月键 = np.array([d.year * 100 + d.month for d in g["date"]], dtype=np.int32)

            for 代码, 布尔 in sig.items():
                位置 = np.where(布尔 & 交易掩码)[0]
                if len(位置) == 0:
                    continue
                统计[代码] = 统计.get(代码, 0) + len(位置)
                前向收益(g, 位置, 日期arr[位置], 代码, mkt累, 日期索引, 出行)

                # §8.3 配对替身：同股票、同自然月、随机非信号交易日；携带配对键
                候选 = np.where(交易掩码 & ~布尔)[0]
                if len(候选) == 0:
                    continue
                候选月 = 月键[候选]
                月到候选 = {int(m): 候选[候选月 == m] for m in np.unique(候选月)}
                替身位置, 替身配对键 = [], []
                for t in 位置:
                    同月 = 月到候选.get(int(月键[t]))
                    if 同月 is None or len(同月) == 0:
                        continue
                    替身位置.append(int(rng.choice(同月)))
                    替身配对键.append(日期arr[t])          # 配对真实信号日
                if 替身位置:
                    前向收益(g, np.array(替身位置), 替身配对键, f"P0_{代码}",
                            mkt累, 日期索引, 出行)

        if 出行:
            pd.DataFrame(出行, columns=列名).to_parquet(
                块目录 / f"块_{块数:03d}.parquet", compression="zstd", index=False)
            块数 += 1
        print(f"  块 {b0//块大小 + 1}/{(len(符号)-1)//块大小 + 1}  本块事件 {len(出行):,}",
              flush=True)
        del 出行, df

    # 合并 + 净收益/超额（在 duckdb 里算，避免 pandas 内存峰值）
    滑点档 = [(0.0, "0bp"), (0.0005, "5bp"), (0.0010, "10bp"), (0.0020, "20bp")]
    净列 = []
    for h in 持有期:
        for s, tag in 滑点档:
            净列.append(f"毛收益_h{h} - {往返固定摩擦 + 2*s} AS 净收益_h{h}_{tag}")
    超额列 = []
    for h in 持有期:
        超额列.append(f"(毛收益_h{h} - {往返固定摩擦 + 2*0.0005}) - 基准EW_h{h} "
                      f"AS 超额EW_h{h}")
        超额列.append(f"(毛收益_h{h} - {往返固定摩擦 + 2*0.0005}) - 基准300_h{h} "
                      f"AS 超额300_h{h}")
    con.execute(f"""
        COPY (SELECT *, {', '.join(净列)}, {', '.join(超额列)}
              FROM read_parquet('{块目录}/块_*.parquet'))
        TO '{事件文件}' (FORMAT PARQUET, COMPRESSION ZSTD)""")

    汇总行 = con.execute(f"""
        SELECT 信号代码, count(*) n, avg(CASE WHEN 作废 THEN 1.0 ELSE 0.0 END) 作废率,
               count(DISTINCT 信号日) n_days,
               avg(超额EW_h20) 均值超额EW_h20
        FROM read_parquet('{事件文件}') GROUP BY 信号代码 ORDER BY 信号代码""").fetchall()

    汇总 = {
        "生成时间": dt.datetime.now().isoformat(timespec="seconds"),
        "脚本": "02_代码/04_信号层与前向收益.py",
        "预注册依据": ["§6", "§8.1", "§8.3", "§11.2", "§4.4", "§4.5", "§5.5"],
        "随机种子": 随机种子, "往返固定摩擦": 往返固定摩擦,
        "信号事件数_按代码_含作废前": {k: int(v) for k, v in sorted(统计.items())},
        "按代码汇总": [
            {"信号代码": r[0], "事件行数": int(r[1]), "作废率": float(r[2]),
             "不同信号日数": int(r[3]),
             "均值超额EW_h20": None if r[4] is None else float(r[4])}
            for r in 汇总行],
        "文件": str(事件文件), "字节": 事件文件.stat().st_size,
    }
    p = 证据目录 / "信号层汇总.json"
    p.write_text(json.dumps(汇总, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n事件文件 {事件文件.stat().st_size/1e6:.1f} MB")
    print(f"{'信号代码':<18}{'事件行数':>10}{'信号日数':>9}{'作废率':>9}{'超额EW_h20':>12}")
    for r in 汇总["按代码汇总"]:
        v = r["均值超额EW_h20"]
        print(f"{r['信号代码']:<18}{r['事件行数']:>10,}{r['不同信号日数']:>9,}"
              f"{r['作废率']:>8.2%}{(f'{v:+.4%}' if v is not None else 'None'):>12}")
    print(f"\n证据已写入: {p}")
    con.close()


if __name__ == "__main__":
    main()
