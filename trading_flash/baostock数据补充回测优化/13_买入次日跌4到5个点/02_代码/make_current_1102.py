#!/usr/bin/env python3
"""2026-11-02 更新：603778/605376 路径、D+3 回收分组、第三次同现分组。"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "03_结果"
PATH = ROOT / "12_买入后证伪_哪天没达标" / "03_结果" / "买入后逐日路径.parquet"
CAL = ROOT / "01_数据" / "market_daily.parquet"

def half_up(x: float, digits: int = 2) -> float:
    q = Decimal(1).scaleb(-digits)
    return float(Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def pct(x: float) -> float:
    return half_up(float(x) * 100, 2)



# baostock 2026-11-02 收盘快照
CUR = pd.DataFrame([
    ["2026-10-28", "603778", 7.18, 7.30, 7.04, 7.27, 2.54, 28_694_200, 0.73, 7.226, 7.401, 7.777, 34.02, 0.885],
    ["2026-10-29", "603778", 7.21, 7.32, 7.07, 7.08, -2.61, 26_741_300, 0.75, 7.212, 7.355, 7.701, 31.75, 0.040],
    ["2026-10-30", "603778", 6.97, 7.22, 6.76, 6.77, -4.38, 37_924_709, 1.09, 7.100, 7.266, 7.622, 28.43, 0.022],
    ["2026-11-02", "603778", 6.89, 6.96, 6.48, 6.92, 2.22, 41_114_023, 1.17, 7.026, 7.205, 7.563, 32.13, 0.917],
    ["2026-10-28", "605376", 77.80, 80.30, 76.82, 79.02, 3.39, 8_472_634, 0.85, 81.074, 85.378, 99.564, 20.67, 0.632],
    ["2026-10-29", "605376", 79.00, 79.52, 73.89, 74.06, -6.28, 7_171_892, 0.74, 78.928, 83.604, 96.309, 18.55, 0.030],
    ["2026-10-30", "605376", 71.07, 75.85, 69.76, 70.82, -4.37, 8_401_230, 0.89, 76.086, 81.741, 93.200, 17.31, 0.174],
    ["2026-11-02", "605376", 72.02, 73.35, 69.18, 71.67, 1.20, 10_758_403, 1.16, 74.400, 80.109, 90.785, 18.85, 0.597],
], columns=["日期", "代码", "开盘", "最高", "最低", "收盘", "涨跌幅(%)", "成交量", "量比20日",
            "MA5", "MA10", "MA20", "RSI14", "收盘位置"])
ENTRY = {"603778": 7.18, "605376": 77.80}
CUR["D+n"] = CUR.groupby("代码").cumcount().map(lambda x: f"D+{x}")
CUR["相对买入价(%)"] = CUR.apply(lambda r: half_up((r["收盘"] / ENTRY[r["代码"]] - 1) * 100, 2), axis=1)
CUR["最低相对买入价(%)"] = CUR.apply(lambda r: half_up((r["最低"] / ENTRY[r["代码"]] - 1) * 100, 2), axis=1)


def load() -> pd.DataFrame:
    df = pd.read_parquet(PATH)
    d0 = df[df.d == 0].set_index(["signal_date", "code"])[["final_ret_sim", "rank", "drawdown250_signal"]].add_prefix("d0_")
    d1 = df[df.d == 1].set_index(["signal_date", "code"])[["ret", "ret_prev"]].add_prefix("d1_")
    d2 = df[df.d == 2].set_index(["signal_date", "code"])[["ret", "ret_prev"]].add_prefix("d2_")
    d3 = df[df.d == 3].set_index(["signal_date", "code"])[["ret", "ret_prev"]].add_prefix("d3_")
    return d0.join([d1, d2, d3]).reset_index()


def bucket(t: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    s = t[mask]
    r = s["d0_final_ret_sim"].dropna()
    return {
        "口径": label,
        "笔数": len(s),
        "最终样本": len(r),
        "均值(%)": pct(r.mean()) if len(r) else None,
        "中位数(%)": pct(r.median()) if len(r) else None,
        "亏损率(%)": pct((r < 0).mean()) if len(r) else None,
        "止盈率(%)": pct((r >= 0.1999).mean()) if len(r) else None,
        "亏损15(%)": pct((r <= -0.15).mean()) if len(r) else None,
    }


def third_occ(t: pd.DataFrame) -> pd.DataFrame:
    s = t[["signal_date", "code", "d0_rank", "d0_final_ret_sim"]].drop_duplicates(["signal_date", "code"]).sort_values(["signal_date", "d0_rank"])
    first = s[s["d0_rank"] == 1].set_index("signal_date")
    second = s[s["d0_rank"] == 2].set_index("signal_date")
    days = pd.DataFrame({"c1": first["code"], "c2": second["code"]}).dropna().reset_index()
    days["pair"] = days.apply(lambda r: (r.c1, r.c2), axis=1)
    fr = s.set_index(["signal_date", "code"])["d0_final_ret_sim"]
    days["r1"] = [fr.get((d, c), np.nan) for d, c in zip(days.signal_date, days.c1)]
    days["r2"] = [fr.get((d, c), np.nan) for d, c in zip(days.signal_date, days.c2)]
    days["pair_ret"] = days[["r1", "r2"]].mean(axis=1)
    cal = pd.DatetimeIndex(sorted(pd.read_parquet(CAL, columns=["date"]).date.unique()))
    ci = {d: i for i, d in enumerate(cal)}
    rows = []
    for k, g in days.groupby("pair"):
        if len(g) < 3:
            continue
        g = g.sort_values("signal_date").reset_index(drop=True)
        for i in range(1, min(3, len(g))):
            if i == 2:
                rows.append({
                    "pair": k,
                    "date": g.loc[i, "signal_date"],
                    "gap": ci[g.loc[i, "signal_date"]] - ci[g.loc[i - 1, "signal_date"]],
                    "pair_ret": g.loc[i, "pair_ret"],
                    "year": g.loc[i, "signal_date"].year,
                })
    return pd.DataFrame(rows)


def main() -> None:
    CUR.to_csv(OUT / "当前案例_603778_2026-11-02_路径.csv", index=False, encoding="utf-8-sig")
    t = load()
    base = t.d2_ret <= -0.05
    rows = [
        bucket(t, base, "D+2 累计≤-5%（10/30 状态）"),
        bucket(t, base & (t.d3_ret > -0.05), "D+2≤-5 且 D+3 累计>-5（今天收回）"),
        bucket(t, base & (t.d3_ret_prev > 0), "D+2≤-5 且 D+3 当日>0"),
        bucket(t, base & (t.d3_ret_prev > 0.02), "D+2≤-5 且 D+3 当日>+2%"),
        bucket(t, base & (t.d3_ret > -0.05) & (t.d3_ret_prev > 0.02) & (t.d0_rank == 1), "当前形态+榜一（D2≤-5、D3>-5、D3当日>+2%）"),
        bucket(t, (t.d3_ret <= -0.02) & (t.d3_ret_prev > 0.02), "D+3 累计≤-2% 且 D+3 当日>+2%"),
    ]
    pd.DataFrame(rows).to_csv(OUT / "当前案例_2026-11-02_对比组.csv", index=False, encoding="utf-8-sig")

    po = third_occ(t)
    third = [
        bucket_third(po, po, "第三次同现·全部"),
        bucket_third(po, po[po.gap <= 3], "第三次同现·间隔≤3日"),
        bucket_third(po, po[po.gap == 3], "第三次同现·间隔=3日（当前）"),
        bucket_third(po, po[po.year >= 2017], "第三次同现·2017年后"),
        bucket_third(po, po[po.year == 2026], "第三次同现·2026年"),
    ]
    pd.DataFrame(third).to_csv(OUT / "当前案例_2026-11-02_第三次同现.csv", index=False, encoding="utf-8-sig")
    print(CUR.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(third).to_string(index=False))


def bucket_third(po: pd.DataFrame, sub: pd.DataFrame, label: str) -> dict:
    x = sub["pair_ret"].dropna()
    return {
        "口径": label,
        "笔数": len(sub),
        "均值(%)": pct(x.mean()) if len(x) else None,
        "中位数(%)": pct(x.median()) if len(x) else None,
        "亏损率(%)": pct((x < 0).mean()) if len(x) else None,
        "最差(%)": pct(x.min()) if len(x) else None,
        "最好(%)": pct(x.max()) if len(x) else None,
    }


if __name__ == "__main__":
    main()
