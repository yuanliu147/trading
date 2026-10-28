#!/usr/bin/env python3
"""100+ 信号日，只买榜一+榜二，各 1 万元，不限总仓位。

两版：
- 不重复：同一只股票在持仓期内再次出信号，不再加 1w（与现有策略一致）；
- 每天加：每个 100+ 信号日都按榜一/榜二各加 1w，允许同一只股票多笔。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lib import (Config, _down_limit, _up_limit, build_signals,
                          load_panel, load_stocks)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "09_百信号日_榜一二_1w"
OUT.mkdir(parents=True, exist_ok=True)

PERIODS = [
    ("2005-2026", "2005-01-01", "2026-10-23"),
    ("2005-2016", "2005-01-01", "2016-12-31"),
    ("2017-2026", "2017-10-19", "2026-10-23"),
]
BUDGET = 10_000.0
SLIP = 0.001
COMM = 0.00025
MIN_COMM = 5.0
TRANSFER = 0.00002
STAMP_BEFORE = ("2023-08-28", 0.001)
STAMP_AFTER = 0.0005


def buy_fees(gross: float) -> float:
    return max(MIN_COMM, gross * COMM) + gross * TRANSFER


def sell_fees(gross: float, date: pd.Timestamp) -> float:
    stamp = STAMP_BEFORE[1] if date < pd.Timestamp(STAMP_BEFORE[0]) else STAMP_AFTER
    return max(MIN_COMM, gross * COMM) + gross * TRANSFER + gross * stamp


def simulate(g: pd.DataFrame, j: int) -> dict | None:
    """信号日索引 j；T+3 开盘买 1w，+20% 止盈 / 15 个交易日到期。"""
    if j + 3 >= len(g):
        return dict(status="pending_entry", reason="entry_beyond_data")
    e = g.iloc[j + 3]
    if not np.isfinite(e.open) or e.open <= 0 or not np.isfinite(e.volume) or e.volume <= 0:
        return dict(status="reject", reason="no_bar")
    lim = _up_limit(e.preclose, int(e.is_st))
    if np.isfinite(lim) and e.open >= lim - 1e-9:
        return dict(status="reject", reason="limit_up")
    entry_px = float(e.open) * (1 + SLIP)
    shares = int(np.floor(BUDGET / (entry_px * 100) + 1e-9)) * 100
    while shares > 0:
        gross = shares * entry_px
        if gross + buy_fees(gross) <= BUDGET + 1e-9:
            break
        shares -= 100
    if shares <= 0:
        return dict(status="reject", reason="budget_lt_1lot")
    gross_buy = shares * entry_px
    fees_buy = buy_fees(gross_buy)
    invested = gross_buy + fees_buy
    if j + 3 + 14 >= len(g):
        return dict(status="pending_entry", reason="insufficient_future")
    entry_hfq = float(e.hfq_open)
    target = entry_hfq * 1.2
    exit_i = None
    exit_raw = None
    exit_hfq = None
    reason = None
    pending = False
    for k in range(1, 15):
        r = g.iloc[j + 3 + k]
        if not np.isfinite(r.volume) or r.volume <= 0:
            continue
        if np.isfinite(r.hfq_open) and r.hfq_open >= target:
            exit_i = j + 3 + k
            exit_raw = float(r.open)
            exit_hfq = float(r.hfq_open)
            reason = "take_profit_open"
            break
        if np.isfinite(r.hfq_high) and r.hfq_high >= target:
            exit_i = j + 3 + k
            exit_raw = float(e.open) * (target / entry_hfq)
            exit_hfq = target
            reason = "take_profit_intraday"
            break
    if exit_i is None:
        end_i = j + 3 + 14
        r = g.iloc[end_i]
        if (np.isfinite(r.close) and np.isfinite(r.volume) and r.volume > 0
                and not (np.isfinite(_down_limit(r.preclose, int(r.is_st)))
                         and abs(float(r.close) - _down_limit(r.preclose, int(r.is_st))) <= 0.011)):
            exit_i = end_i
            exit_raw = float(r.close)
            exit_hfq = float(r.hfq_close)
            reason = "max_holding"
        else:
            k = end_i + 1
            while k < len(g):
                r = g.iloc[k]
                if np.isfinite(r.open) and np.isfinite(r.volume) and r.volume > 0:
                    lim = _down_limit(r.preclose, int(r.is_st))
                    if not (np.isfinite(lim) and abs(float(r.open) - lim) <= 0.011):
                        exit_i = k
                        exit_raw = float(r.open)
                        exit_hfq = float(r.hfq_open)
                        reason = "max_holding"
                        pending = True
                        break
                k += 1
            if exit_i is None:
                exit_i = len(g) - 1
                exit_raw = float(g.iloc[-1].close)
                exit_hfq = float(g.iloc[-1].hfq_close)
                reason = "delist_last_close"
                pending = True
    exit_date = g.iloc[exit_i].date
    gross_sell = shares * exit_raw
    fees_sell = sell_fees(gross_sell, exit_date)
    proceeds = gross_sell - fees_sell
    pnl = proceeds - invested
    return dict(
        status="filled", reason=reason, pending=pending,
        entry_date=g.iloc[j + 3].date, exit_date=exit_date,
        entry_px=float(e.open), entry_hfq=entry_hfq, exit_px=float(exit_raw),
        shares=shares, invested=invested, proceeds=proceeds, pnl=pnl,
        ret=proceeds / invested - 1, holding_sessions=exit_i - (j + 3) + 1,
    )


def main():
    panel = load_panel()
    stocks = load_stocks()
    names = stocks.set_index("code")["code_name"].to_dict()
    bars = pd.read_parquet(
        ROOT / "01_数据/bars.parquet",
        columns=["code", "date", "open", "high", "low", "close", "preclose",
                 "volume", "is_st", "hfq_open", "hfq_high", "hfq_close"],
    )
    bars["date"] = pd.to_datetime(bars.date)
    by_code = {c: g.sort_values("date").reset_index(drop=True) for c, g in bars.groupby("code")}

    all_trades = []
    all_candidates = []
    summaries = []
    for period, start, end in PERIODS:
        cfg = Config(start=start, end=end, universe="all", gate="all",
                     extra_filter="is_st == 0", max_rank=2, label="C_top2")
        sig = build_signals(panel, stocks, cfg)
        cnt = sig.groupby("date").size()
        high = sorted(cnt[cnt >= 100].index)
        candidates = []
        for T in high:
            top = sig[sig.date == T].sort_values("rank").head(2)
            for r in top.itertuples():
                g = by_code.get(r.code)
                if g is None:
                    continue
                pos = g.index[g.date == T]
                if len(pos) == 0:
                    continue
                sim = simulate(g, int(pos[0]))
                candidates.append(dict(
                    期间=period, 信号日=T, 信号数=int(cnt.loc[T]), 排名=int(r.rank),
                    代码=r.code, 名称=names.get(r.code, ""), **sim,
                ))
        cdf = pd.DataFrame(candidates)

        # 不重复：按 entry_date 顺序，持仓期内同一只股票不再加
        filled = cdf[cdf.status == "filled"].copy().sort_values(["entry_date", "信号日", "排名"])
        held_until: dict[str, pd.Timestamp] = {}
        keep = []
        for r in filled.itertuples():
            if r.代码 in held_until and held_until[r.代码] >= r.entry_date:
                keep.append(False)
            else:
                keep.append(True)
                held_until[r.代码] = r.exit_date
        filled["重复持仓"] = ~pd.Series(keep, index=filled.index)
        cdf = cdf.merge(filled[["信号日", "代码", "重复持仓"]],
                        on=["信号日", "代码"], how="left")
        cdf["重复持仓"] = cdf["重复持仓"].fillna(False)
        all_candidates.append(cdf)

        for version, mask in [("不重复", ~cdf["重复持仓"]), ("每天加", pd.Series(True, index=cdf.index))]:
            d = cdf[mask & (cdf.status == "filled")].copy()
            if d.empty:
                continue
            # 峰值并发与资金占用
            events = []
            for r in d.itertuples():
                events.append((r.entry_date, 1, r.invested))
                events.append((r.exit_date, -1, -r.invested))
            events.sort(key=lambda x: (x[0], x[1]))  # 同日先减后加
            cur_n = cur_cap = peak_n = peak_cap = 0
            for _, delta_n, delta_cap in events:
                cur_n += delta_n
                cur_cap += delta_cap
                peak_n = max(peak_n, cur_n)
                peak_cap = max(peak_cap, cur_cap)
            summaries.append(dict(
                期间=period, 版本=version, 笔数=len(d),
                盈利笔数=int((d.pnl > 0).sum()), 胜率=(d.pnl > 0).mean(),
                总盈亏=d.pnl.sum(), 平均盈亏=d.pnl.mean(), 中位盈亏=d.pnl.median(),
                最好=d.pnl.max(), 最差=d.pnl.min(),
                总投入=d.invested.sum(), 投入回报率=d.pnl.sum() / d.invested.sum(),
                峰值并发=peak_n, 峰值资金=peak_cap,
            ))
            d["版本"] = version
            all_trades.append(d)

    trades = pd.concat(all_trades, ignore_index=True)
    trades = trades.sort_values(["期间", "版本", "entry_date", "信号日", "排名"])
    trades.to_csv(OUT / "榜一二_1w_逐笔.csv", index=False, encoding="utf-8-sig")

    cand = pd.concat(all_candidates, ignore_index=True)
    cand.to_csv(OUT / "榜一二_1w_候选.csv", index=False, encoding="utf-8-sig")
    skipped = cand[(cand["重复持仓"]) & (cand["status"] == "filled")]
    skipped.to_csv(OUT / "榜一二_1w_跳过重复.csv", index=False, encoding="utf-8-sig")
    sm = pd.DataFrame(summaries)
    sm.to_csv(OUT / "榜一二_1w_汇总.csv", index=False, encoding="utf-8-sig")

    # 年度盈亏（不重复版）
    t = trades[trades.版本 == "不重复"].copy()
    t["年份"] = pd.to_datetime(t.exit_date).dt.year
    annual = t.groupby(["期间", "年份"]).agg(笔数=("pnl", "size"), 盈亏=("pnl", "sum"),
                                            胜率=("pnl", lambda x: (x > 0).mean())).reset_index()
    annual.to_csv(OUT / "榜一二_1w_年度.csv", index=False, encoding="utf-8-sig")

    # 每日盯市：未平仓按 hfq 收盘估值；平仓日换成已实现盈亏
    dates_all = pd.DatetimeIndex(sorted(bars.date.unique()))
    mtm = pd.DataFrame(index=dates_all)
    for version in ["不重复", "每天加"]:
        d = trades[(trades["期间"] == "2005-2026") & (trades["版本"] == version)].copy()
        if d.empty:
            continue
        first, last = d["entry_date"].min(), d["exit_date"].max()
        idx = dates_all[(dates_all >= first) & (dates_all <= last)]
        level = pd.Series(0.0, index=idx)
        for r in d.itertuples():
            g = by_code[r.代码]
            hold = g[(g.date >= r.entry_date) & (g.date <= r.exit_date)]
            if hold.empty:
                continue
            s = pd.Series(np.nan, index=idx)
            gross_buy = r.shares * r.entry_px
            last_val = 0.0
            for row in hold.itertuples():
                if np.isfinite(row.hfq_close) and r.entry_hfq > 0:
                    last_val = gross_buy * row.hfq_close / r.entry_hfq - gross_buy
                    s.loc[row.date] = last_val
            s = s.ffill()
            s.loc[idx < r.entry_date] = 0.0
            s.loc[idx >= r.exit_date] = r.pnl
            s = s.fillna(0.0)
            level = level + s
        mtm[version] = level

    mtm = mtm.ffill().fillna(0.0)
    mtm.index.name = "date"
    mtm.to_csv(OUT / "榜一二_1w_每日盯市.csv", encoding="utf-8-sig")
    for version in mtm.columns:
        lv = mtm[version]
        dd = (lv - lv.cummax()).min()
        print(f"盯市 {version}: 期末 {lv.iloc[-1]:.2f} 元, 最大回撤 {dd:.2f} 元")

    print(sm.to_string(index=False))
    print("\n逐笔（不重复，2005-2026）：")
    print(trades[(trades.期间 == "2005-2026") & (trades.版本 == "不重复")][[
        "信号日", "信号数", "排名", "代码", "名称", "entry_date", "exit_date",
        "invested", "pnl", "ret", "reason", "重复持仓",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
