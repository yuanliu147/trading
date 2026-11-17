#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F07 独立抽查：主板冻结面板复算 20 日方差比 VR_20 中位数
原研究：2026-10-29_趋势结构适配性_选股特征（全A 2010-2020，含创业板/科创板/退市）
原定义（02_代码/03_特征面板.py 个股特征）：
    r1 = diff(log(后复权收盘)); var1 = r1.rolling(120).var()  # pandas 默认 ddof=1
    r20 = logc - logc.shift(20); var20 = r20.rolling(120).var()
    VR_20 = var20 / (20 * var1)
    剔除停牌日后按个股时间序滚动；全市场所有(股票,日)观测取中位数。
原口径值：0.7918736164601688（特征_完整.parquet 直接核得，n=6,910,916，2010-01-04..2020-12-31）

本复算差异（主板含义检验，非精确复刻）：
1) 宇宙改为冻结主板面板 main_panel.parquet（3,425 只含退市 231 只，2010-01-04..2026-10-29）；
2) 后复权用 signal_close；交易行 = data_valid AND trade_status=1 AND signal_close>0；
3) 主板面板自 2010 年起，无 2008-2009 预热，2010 年上半年滚动窗不足自然为 NaN
   （另报 2011 起中位数作预热敏感性）；
4) 延伸计算 2021-2026 分年（原研究封存段，此处仅描述 VR，不算任何策略收益）。
只读源数据；输出仅写本目录。
"""
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
PANEL = "/Users/admin/WorkSpace/trading/trading_finally/data/main_panel.parquet"

con = duckdb.connect()
con.execute("PRAGMA threads=4")
df = con.execute(f"""
    SELECT code, date, signal_close
    FROM read_parquet('{PANEL}')
    WHERE data_valid AND trade_status = 1 AND signal_close > 0
    ORDER BY code, date
""").df()
con.close()
print(f"交易行 {len(df):,}，股票 {df['code'].nunique():,}")

parts = []
for code, g in df.groupby("code", sort=False):
    logc = np.log(g["signal_close"].astype(float))
    r1 = logc.diff()
    var1 = r1.rolling(120).var()
    r20 = logc - logc.shift(20)
    var20 = r20.rolling(120).var()
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = var20 / (20 * var1)
    parts.append(pd.DataFrame({"date": g["date"].to_numpy(), "VR_20": vr.to_numpy()}))
full = pd.concat(parts, ignore_index=True)
full["date"] = pd.to_datetime(full["date"])
full = full[np.isfinite(full["VR_20"])]
full["year"] = full["date"].dt.year

rows = []
def add(name, sub):
    v = sub["VR_20"]
    rows.append({"窗口": name, "n": len(v), "VR20中位数": float(v.median()),
                 "VR20均值": float(v.mean()),
                 "VR20_P25": float(v.quantile(.25)), "VR20_P75": float(v.quantile(.75)),
                 "占比VR<1": float((v < 1).mean())})

add("主板 2010-2020（对标原研究期）", full[full["date"] <= "2020-12-31"])
add("主板 2011-2020（去预热敏感性）", full[(full["date"] >= "2011-01-01") & (full["date"] <= "2020-12-31")])
add("主板 2021-2026（原封存段，仅描述）", full[full["date"] >= "2021-01-01"])
add("主板 2010-2026 全窗", full)
summary = pd.DataFrame(rows)
summary.to_csv(HERE / "VR20_主板_汇总.csv", index=False)

byyear = full.groupby("year")["VR_20"].agg(n="count", median="median", mean="mean")
byyear.to_csv(HERE / "VR20_主板_分年.csv")

print(summary.to_string(index=False))
print(byyear.to_string())
print("原研究全A口径值（交付物核验）：0.7918736164601688")