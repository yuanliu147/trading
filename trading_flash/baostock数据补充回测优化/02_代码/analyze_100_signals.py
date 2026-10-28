#!/usr/bin/env python3
"""信号日候选数 >=100 时：榜一/榜二信号本身的收益，以及组合实际有没有买到。

两层口径：
1) 信号层：假设不看仓位，T+3 开盘各买一份，+20% 止盈 / 15 个交易日到期；
2) 组合层：真实 2×50% 前二组合，看当天仓位是否已满、实际买到了谁。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lib import Config, build_signals, load_panel, load_stocks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08_百信号日"
OUT.mkdir(parents=True, exist_ok=True)

PERIODS = [
    ("2017-2026", "2017-10-19", "2026-10-23"),
    ("2005-2016", "2005-01-01", "2016-12-31"),
    ("2005-2026", "2005-01-01", "2026-10-23"),
]


def sim_trade(g: pd.DataFrame, j: int) -> dict | None:
    """从信号日索引 j 起，T+3 开盘买入，+20% 止盈 / 15 个交易日到期（含买入日）。"""
    if j + 3 + 14 >= len(g):
        return None
    entry = g.iloc[j + 3].hfq_open
    if not np.isfinite(entry) or entry <= 0:
        return None
    target = entry * 1.2
    exit_hfq = None
    reason = None
    hold = None
    for k in range(1, 15):  # 买入后第 2..15 个交易日
        r = g.iloc[j + 3 + k]
        if np.isfinite(r.hfq_open) and r.hfq_open >= target:
            exit_hfq, reason, hold = r.hfq_open, "take_profit_open", k + 1
            break
        if np.isfinite(r.hfq_high) and r.hfq_high >= target:
            exit_hfq, reason, hold = target, "take_profit_intraday", k + 1
            break
    if exit_hfq is None:
        exit_hfq, reason, hold = g.iloc[j + 3 + 14].hfq_close, "max_holding", 15
    return dict(entry_date=g.iloc[j + 3].date, ret=exit_hfq / entry - 1,
                reason=reason, hold=hold)


def main():
    panel = load_panel()
    stocks = load_stocks()
    bars = pd.read_parquet(
        ROOT / "01_数据/bars.parquet",
        columns=["code", "date", "hfq_open", "hfq_high", "hfq_close"],
    )
    bars["date"] = pd.to_datetime(bars.date)
    by_code = {c: g.sort_values("date").reset_index(drop=True) for c, g in bars.groupby("code")}

    def loss_stats(s):
        s = pd.Series(s).dropna()
        loss = s[s < 0]
        return dict(笔数=len(s), 亏损笔数=len(loss),
                    亏损率=len(loss) / len(s) if len(s) else np.nan,
                    亏损均值=loss.mean() if len(loss) else np.nan,
                    亏损中位=loss.median() if len(loss) else np.nan,
                    最差=s.min() if len(s) else np.nan,
                    亏损合计=loss.sum() if len(loss) else np.nan,
                    跌10=int((s <= -0.10).sum()), 跌20=int((s <= -0.20).sum()))

    details = []
    summary = []
    loss_rows = []
    for period, start, end in PERIODS:
        cfg = Config(start=start, end=end, universe="all", gate="all",
                     extra_filter="is_st == 0", max_rank=2, label="C_top2")
        sig = build_signals(panel, stocks, cfg)
        cnt = sig.groupby("date").size()
        high = sorted(cnt[cnt >= 100].index)
        tr = pd.read_csv(
            ROOT / "06_只买第一名/结果" / period /
            "C_含退市_排除风险警示_买前二_逐笔.csv",
            parse_dates=["signal_date", "entry_date", "exit_date"],
        )
        tr = tr.merge(sig[["date", "code", "rank"]],
                      left_on=["signal_date", "code"],
                      right_on=["date", "code"], how="left")
        eq = pd.read_csv(
            ROOT / "06_只买第一名/结果" / period /
            "C_含退市_排除风险警示_买前二_每日权益.csv",
            parse_dates=["date"],
        ).set_index("date")

        for d in high:
            top = sig[sig.date == d].sort_values("rank").head(2)
            c1 = top.iloc[0].code if len(top) else ""
            c2 = top.iloc[1].code if len(top) > 1 else ""
            s1 = s2 = None
            g1 = by_code.get(c1)
            if g1 is not None:
                pos = g1.index[g1.date == d]
                if len(pos):
                    s1 = sim_trade(g1, int(pos[0]))
            g2 = by_code.get(c2)
            if g2 is not None:
                pos = g2.index[g2.date == d]
                if len(pos):
                    s2 = sim_trade(g2, int(pos[0]))
            actual = tr[tr.signal_date == d]
            a1 = actual[actual["rank"] == 1]
            a2 = actual[actual["rank"] == 2]
            held = tr[(tr.entry_date <= d) & (tr.exit_date >= d)].code.tolist()
            npos = int(eq.loc[d, "npos"]) if d in eq.index else -1
            if len(a1) and len(a2):
                status = "两笔都买"
            elif len(a1):
                status = "只买榜一"
            elif len(a2):
                status = "只买榜二"
            else:
                status = "没买"
            details.append(dict(
                期间=period, 信号日=d, 信号数=int(cnt.loc[d]), 当日持仓数=npos,
                当日持有=",".join(held),
                榜一=c1, 榜一毛收益=s1["ret"] if s1 else np.nan,
                榜二=c2, 榜二毛收益=s2["ret"] if s2 else np.nan,
                实际买入状态=status,
                实际榜一收益=a1.ret.iloc[0] if len(a1) else np.nan,
                实际榜二收益=a2.ret.iloc[0] if len(a2) else np.nan,
                实际榜一退出=a1.exit_reason.iloc[0] if len(a1) else "",
                实际榜二退出=a2.exit_reason.iloc[0] if len(a2) else "",
            ))

        h = pd.DataFrame([x for x in details if x["期间"] == period])
        both = h.dropna(subset=["榜一毛收益", "榜二毛收益"])
        summary.append(dict(
            期间=period, 信号日数=int(len(cnt)), 百信号日=len(h),
            信号层可观察=len(both),
            信号层双盈=int(((both["榜一毛收益"] > 0) & (both["榜二毛收益"] > 0)).sum()),
            仅榜一盈=int(((both["榜一毛收益"] > 0) & (both["榜二毛收益"] <= 0)).sum()),
            仅榜二盈=int(((both["榜一毛收益"] <= 0) & (both["榜二毛收益"] > 0)).sum()),
            双亏=int(((both["榜一毛收益"] <= 0) & (both["榜二毛收益"] <= 0)).sum()),
            榜一均值=both["榜一毛收益"].mean(), 榜二均值=both["榜二毛收益"].mean(),
            实际有买入=int((h["实际买入状态"] != "没买").sum()),
            实际两笔都买=int((h["实际买入状态"] == "两笔都买").sum()),
            实际两笔都买且双盈=int(((h["实际买入状态"] == "两笔都买") &
                                (h["实际榜一收益"] > 0) & (h["实际榜二收益"] > 0)).sum()),
            两仓已满=int((h["当日持仓数"] == 2).sum()),
            空仓=int((h["当日持仓数"] == 0).sum()),
        ))
        for rk, col in [(1, "榜一毛收益"), (2, "榜二毛收益")]:
            loss_rows.append(dict(口径="100+信号日", 期间=period, 排名=rk,
                                  **loss_stats(both[col])))
        for rk in [1, 2]:
            loss_rows.append(dict(口径="全样本前二", 期间=period, 排名=rk,
                                  **loss_stats(tr[tr["rank"] == rk].ret)))

    df = pd.DataFrame(details)
    df.to_csv(OUT / "百信号日_明细.csv", index=False, encoding="utf-8-sig")
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "百信号日_汇总.csv", index=False, encoding="utf-8-sig")
    loss = pd.DataFrame(loss_rows)
    loss.to_csv(OUT / "百信号日_亏损对比.csv", index=False, encoding="utf-8-sig")
    print(sm.to_string(index=False))
    print("\n亏损对比：")
    print(loss.to_string(index=False))
    print("\n2005-2026 明细：")
    print(df[df["期间"] == "2005-2026"][[
        "信号日", "信号数", "当日持仓数", "实际买入状态",
        "榜一", "榜一毛收益", "榜二", "榜二毛收益",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
