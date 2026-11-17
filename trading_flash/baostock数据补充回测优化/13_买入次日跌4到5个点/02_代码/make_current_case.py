#!/usr/bin/env python3
"""2026-10-30 当前案例：603778 买入后 D+2 累计 -5.7%。

数据：
  - 603778 10/28-10/30 OHLC/量：baostock 2026-10-30 收盘后快照。
  - 历史路径：12_买入后证伪_哪天没达标/03_结果/买入后逐日路径.parquet。
  - 历史同形态分组直接由路径数据重算，避免手抄。
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "03_结果"
PATH = ROOT / "12_买入后证伪_哪天没达标" / "03_结果" / "买入后逐日路径.parquet"

ENTRY = 7.18
CURRENT = pd.DataFrame([
    # 日期, 开盘, 最高, 最低, 收盘, 涨跌幅%, 成交量, 量比20日, MA5, MA10, MA20, RSI14, 收盘位置
    ["2026-10-28", 7.18, 7.30, 7.04, 7.27, 2.54, 28_694_200, 0.73, 7.226, 7.401, 7.777, 34.02, 0.885],
    ["2026-10-29", 7.21, 7.32, 7.07, 7.08, -2.61, 26_741_300, 0.75, 7.212, 7.355, 7.701, 31.75, 0.040],
    ["2026-10-30", 6.97, 7.22, 6.76, 6.77, -4.38, 37_924_709, 1.09, 7.100, 7.266, 7.622, 28.43, 0.022],
], columns=["日期", "开盘", "最高", "最低", "收盘", "涨跌幅(%)", "成交量", "量比20日",
            "MA5", "MA10", "MA20", "RSI14", "收盘位置"])
CURRENT["相对买入价(%)"] = (CURRENT["收盘"] / ENTRY - 1) * 100
CURRENT["最低相对买入价(%)"] = (CURRENT["最低"] / ENTRY - 1) * 100
CURRENT["D+n"] = ["D+0", "D+1", "D+2"]


def half_up(x: float, digits: int = 2) -> float:
    q = Decimal(1).scaleb(-digits)
    return float(Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def pct(x: float) -> float:
    return half_up(float(x) * 100, 2)


def load_trades() -> pd.DataFrame:
    df = pd.read_parquet(PATH)
    d0 = df[df.d == 0].set_index(["signal_date", "code"])[
        ["date", "entry_open", "final_ret_sim", "rank", "is_actual", "logged_ret", "drawdown250_signal"]
    ].add_prefix("d0_")
    d1 = df[df.d == 1].set_index(["signal_date", "code"])[["date", "ret", "ret_prev"]].add_prefix("d1_")
    d2 = df[df.d == 2].set_index(["signal_date", "code"])[["date", "ret", "ret_prev"]].add_prefix("d2_")
    d3 = df[df.d == 3].set_index(["signal_date", "code"])[["ret", "ret_prev"]].add_prefix("d3_")
    t = d0.join([d1, d2, d3])
    return t.reset_index()


def bucket(t: pd.DataFrame, mask: pd.Series, label: str, note: str) -> dict:
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
        "备注": note,
    }


def main() -> None:
    cur = CURRENT.copy()
    cur.to_csv(OUT / "当前案例_603778_路径.csv", index=False, encoding="utf-8-sig")

    t = load_trades()
    pattern = (t.d2_ret <= -0.05) & (t.d1_ret_prev > -0.03) & (t.d2_ret_prev <= -0.04)
    rows = [
        bucket(t, (t.d1_ret_prev >= -0.05) & (t.d1_ret_prev < -0.04),
               "D+1 当日跌4-5%（上次回答）", "当天跌幅口径；不是当前状态"),
        bucket(t, (t.d2_ret_prev >= -0.05) & (t.d2_ret_prev < -0.04),
               "D+2 当日跌4-5%", "只看第二天当天跌幅"),
        bucket(t, (t.d2_ret_prev >= -0.05) & (t.d2_ret_prev < -0.04) & (t.d2_ret > -0.05),
               "D+2 当日跌4-5% 且 累计 > -5%", "仍在 -5% 线以上（好的一半）"),
        bucket(t, (t.d2_ret_prev >= -0.05) & (t.d2_ret_prev < -0.04) & (t.d2_ret <= -0.05),
               "D+2 当日跌4-5% 且 累计 ≤ -5%", "当前 -5.71% 在坏的一半"),
        bucket(t, (t.d2_ret >= -0.065) & (t.d2_ret < -0.05),
               "D+2 累计 -6.5%~-5%", "当前 -5.71% 所在区间"),
        bucket(t, t.d2_ret <= -0.05,
               "D+2 累计 ≤ -5%", "更宽的累计亏损组"),
        bucket(t, pattern,
               "D+2累计≤-5% 且 D1>-3% 且 D2≤-4%", "当前形态：第一天小跌、第二天大跌"),
        bucket(t, pattern & (t.d0_rank == 1),
               "同上 + 榜一", "当前为榜一"),
        bucket(t, pattern & (t.d0_rank == 1) & (t.d0_drawdown250_signal <= -0.65),
               "同上 + 榜一 + 信号回撤≤-65%", "最窄组，样本仅 10 笔"),
    ]
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "当前案例_对比组.csv", index=False, encoding="utf-8-sig")

    # 603778 自己的同形态案例
    df = pd.read_parquet(PATH)
    d0 = df[df.d == 0].set_index(["signal_date", "code"])[["entry_close", "entry_open"]]
    d0["d0_ret"] = d0["entry_close"] / d0["entry_open"] - 1
    own = t[(t.code == "sh.603778") & t.signal_date.isin(pd.to_datetime([
        "2026-07-08", "2026-07-14", "2026-07-23", "2026-10-15"]))]
    own_rows = []
    for _, r in own.iterrows():
        d0_ret = d0.loc[(r.signal_date, r.code), "d0_ret"]
        signal = r.signal_date.strftime("%Y-%m-%d")
        own_rows.append({
            "信号日": signal,
            "买入日": r.d0_date.strftime("%Y-%m-%d"),
            "后复权买入价": round(float(r.d0_entry_open), 2),
            "D+0收盘相对买入价(%)": pct(d0_ret),
            "D+1(%)": pct(r.d1_ret_prev),
            "D+2(%)": pct(r.d2_ret_prev),
            "D+2累计(%)": pct(r.d2_ret),
            "D+3累计(%)": pct(r.d3_ret),
            "最终收益(%)": pct(r.d0_final_ret_sim) if pd.notna(r.d0_final_ret_sim) else None,
            "截至10月30日收益(%)": pct(6.77 / 7.50 - 1) if signal == "2026-10-15" else None,
            "状态": "已完成" if pd.notna(r.d0_final_ret_sim) else "未完成",
            "相似度": "相似：首日小涨、D2大跌" if signal in ("2026-07-23", "2026-10-15") else "不相似：首日大幅低开",
        })
    own_df = pd.DataFrame(own_rows)
    own_df.to_csv(OUT / "当前案例_603778_同形态.csv", index=False, encoding="utf-8-sig")

    # 当前案例摘要
    print("当前 603778：")
    print(cur.to_string(index=False))
    print("\n历史分组：")
    print(comp.to_string(index=False))
    print("\n603778 同形态：")
    print(own_df.to_string(index=False))


if __name__ == "__main__":
    main()
