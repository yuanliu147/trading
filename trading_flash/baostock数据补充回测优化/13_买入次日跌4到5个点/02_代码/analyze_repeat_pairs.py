#!/usr/bin/env python3
"""历史“前二组合反复出现”与“个股反复出信号”统计。

样本：12_买入后证伪_哪天没达标/03_结果/买入后逐日路径.parquet 的 1,256 笔前二。
口径：
  - 每个信号日取 rank1/rank2 两只；组合按“同两只、顺序无关”与“同顺序”两种。
  - 重复出现：同一组合在 >=2 个信号日出现；本次收益=两只最终收益的等权平均。
  - 间隔=两个信号日之间的交易日数（用 market_daily 日历）。
  - 个股重复：同一代码在 <=3 个交易日内再次进前二，比较后一次 vs 前一次。
"""
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


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(PATH)
    s = df[df.d == 0][["signal_date", "code", "rank", "final_ret_sim", "is_actual"]]
    s = s.drop_duplicates(["signal_date", "code"]).sort_values(["signal_date", "rank"])
    first = s[s["rank"] == 1].set_index("signal_date")
    second = s[s["rank"] == 2].set_index("signal_date")
    days = pd.DataFrame({"c1": first["code"], "c2": second["code"]}).dropna().reset_index()
    days["pair"] = days.apply(lambda r: tuple(sorted([r.c1, r.c2])), axis=1)
    days["pair_ordered"] = days.apply(lambda r: (r.c1, r.c2), axis=1)
    fr = s.set_index(["signal_date", "code"])["final_ret_sim"]
    days["r1"] = [fr.get((d, c), np.nan) for d, c in zip(days.signal_date, days.c1)]
    days["r2"] = [fr.get((d, c), np.nan) for d, c in zip(days.signal_date, days.c2)]
    days["pair_ret"] = days[["r1", "r2"]].mean(axis=1)
    return s, days


def repeat_table(days: pd.DataFrame, key: str) -> pd.DataFrame:
    cal = pd.DatetimeIndex(sorted(pd.read_parquet(CAL, columns=["date"]).date.unique()))
    idx = {d: i for i, d in enumerate(cal)}
    rows = []
    for k, g in days.groupby(key):
        if len(g) < 2:
            continue
        g = g.sort_values("signal_date").reset_index(drop=True)
        for i in range(1, len(g)):
            rows.append({
                "组合": " / ".join(k) if isinstance(k, tuple) else k,
                "重复次数": len(g),
                "首次出现": g.loc[0, "signal_date"].strftime("%Y-%m-%d"),
                "本次出现": g.loc[i, "signal_date"].strftime("%Y-%m-%d"),
                "间隔交易日": idx[g.loc[i, "signal_date"]] - idx[g.loc[i - 1, "signal_date"]],
                "本次组合平均收益(%)": pct(g.loc[i, "pair_ret"]) if pd.notna(g.loc[i, "pair_ret"]) else None,
                "本次榜一收益(%)": pct(g.loc[i, "r1"]) if pd.notna(g.loc[i, "r1"]) else None,
                "本次榜二收益(%)": pct(g.loc[i, "r2"]) if pd.notna(g.loc[i, "r2"]) else None,
                "首次组合收益(%)": pct(g.loc[0, "pair_ret"]) if pd.notna(g.loc[0, "pair_ret"]) else None,
                "年份": int(g.loc[i, "signal_date"].year),
            })
    return pd.DataFrame(rows)


def agg(df: pd.DataFrame, label: str, sub: pd.DataFrame) -> dict:
    x = sub["本次组合平均收益(%)"].dropna()
    return {
        "口径": label,
        "笔数": len(sub),
        "均值(%)": half_up(x.mean(), 2) if len(x) else None,
        "中位数(%)": half_up(x.median(), 2) if len(x) else None,
        "亏损率(%)": pct((x < 0).mean()) if len(x) else None,
        "正收益占比(%)": pct((x > 0).mean()) if len(x) else None,
        "最差(%)": half_up(x.min(), 2) if len(x) else None,
        "最好(%)": half_up(x.max(), 2) if len(x) else None,
    }


def main() -> None:
    s, days = load()
    ordered = repeat_table(days, "pair_ordered")
    unordered = repeat_table(days, "pair")
    ordered.to_csv(OUT / "历史重复前二组合.csv", index=False, encoding="utf-8-sig")

    summary = [
        agg(ordered, "同顺序组合·全部重复", ordered),
        agg(ordered, "同顺序组合·间隔≤3日", ordered[ordered["间隔交易日"] <= 3]),
        agg(ordered, "同顺序组合·间隔=3日", ordered[ordered["间隔交易日"] == 3]),
        agg(ordered, "同顺序组合·间隔≤5日", ordered[ordered["间隔交易日"] <= 5]),
        agg(ordered, "同顺序组合·2017年后·间隔≤3日", ordered[(ordered["年份"] >= 2017) & (ordered["间隔交易日"] <= 3)]),
        agg(ordered, "同顺序组合·2026年·间隔≤3日", ordered[(ordered["年份"] == 2026) & (ordered["间隔交易日"] <= 3)]),
        agg(unordered, "同两只不同顺序·间隔≤3日", unordered[unordered["间隔交易日"] <= 3]),
    ]
    pd.DataFrame(summary).to_csv(OUT / "历史重复前二组合_汇总.csv", index=False, encoding="utf-8-sig")

    # 单只个股：<=3 个交易日再次进前二
    cal = pd.DatetimeIndex(sorted(pd.read_parquet(CAL, columns=["date"]).date.unique()))
    idx = {d: i for i, d in enumerate(cal)}
    s2 = s.sort_values(["code", "signal_date"]).copy()
    s2["prev_date"] = s2.groupby("code")["signal_date"].shift(1)
    s2["gap"] = [idx[d] - idx[p] if pd.notna(p) else np.nan for d, p in zip(s2.signal_date, s2.prev_date)]
    s2["prev_ret"] = s2.groupby("code")["final_ret_sim"].shift(1)
    rep = s2[s2["gap"] <= 3].copy()
    rep["后一次更好"] = rep["final_ret_sim"] > rep["prev_ret"]
    rep_rows = pd.DataFrame({
        "代码": rep["code"],
        "信号日": rep["signal_date"].dt.strftime("%Y-%m-%d"),
        "间隔交易日": rep["gap"].astype(int),
        "前一次收益(%)": rep["prev_ret"].map(lambda x: pct(x) if pd.notna(x) else None),
        "后一次收益(%)": rep["final_ret_sim"].map(lambda x: pct(x) if pd.notna(x) else None),
        "后一次更好": np.where(rep["后一次更好"], "是", "否"),
    })
    rep_rows.to_csv(OUT / "历史重复个股_短窗.csv", index=False, encoding="utf-8-sig")

    single = pd.DataFrame([
        {"口径": "个股重复·≤3日·后一次", "笔数": len(rep),
         "均值(%)": pct(rep["final_ret_sim"].mean()), "中位数(%)": pct(rep["final_ret_sim"].median()),
         "亏损率(%)": pct((rep["final_ret_sim"] < 0).mean()),
         "后一次更好占比(%)": pct(rep["后一次更好"].mean())},
        {"口径": "全部前二·基准", "笔数": int(s["final_ret_sim"].notna().sum()),
         "均值(%)": pct(s["final_ret_sim"].mean()), "中位数(%)": pct(s["final_ret_sim"].median()),
         "亏损率(%)": pct((s["final_ret_sim"] < 0).mean()),
         "后一次更好占比(%)": None},
    ])
    single.to_csv(OUT / "历史重复个股_汇总.csv", index=False, encoding="utf-8-sig")

    # 603778 信号序列
    seq = s[s.code == "sh.603778"].copy()
    seq["信号日"] = seq.signal_date.dt.strftime("%Y-%m-%d")
    seq["最终收益(%)"] = seq["final_ret_sim"].map(lambda x: pct(x) if pd.notna(x) else None)
    seq[["信号日", "rank", "最终收益(%)", "is_actual"]].rename(
        columns={"rank": "排名", "is_actual": "实际成交"}).to_csv(
        OUT / "603778_信号序列.csv", index=False, encoding="utf-8-sig")

    print("前二组合重复：")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\n个股重复：")
    print(single.to_string(index=False))
    print("\n最频繁组合：")
    cnt = days.groupby("pair_ordered").size().sort_values(ascending=False).head(10)
    print(cnt.to_string())
    print("\n603778 信号序列：")
    print(seq[["信号日", "rank", "最终收益(%)", "is_actual"]].to_string(index=False))


if __name__ == "__main__":
    main()
