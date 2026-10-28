#!/usr/bin/env python3
"""100+ 信号日、榜一+榜二各 1w：止盈规则与 ST 过滤的敏感性。

退出模式：
- tp20：+20% 盘中止盈 / 15 个交易日到期（基准）
- hold15：不止盈，15 个交易日到期
- tp20 / tp25 / tp30：固定 20% / 25% / 30% 止盈；未触发则 15 日到期
- trail5 / trail8 / trail10：盈利曾 >=20% 后，从最高点回撤 5% / 8% / 10% 动态止盈；未触发则 15 日到期
过滤：PIT，只排除信号日 is_st=1。按当前名称排除后来变 ST 的股票是未来函数，不采用。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lib import (Config, _down_limit, _up_limit, build_signals,
                          load_panel, load_stocks)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "10_百信号日_止盈变体"
OUT.mkdir(parents=True, exist_ok=True)

BUDGET, SLIP, COMM, MIN_COMM, TRANSFER = 10_000.0, 0.001, 0.00025, 5.0, 0.00002
STAMP_CUT, STAMP_BEFORE, STAMP_AFTER = pd.Timestamp("2023-08-28"), 0.001, 0.0005


def buy_fees(gross):
    return max(MIN_COMM, gross * COMM) + gross * TRANSFER


def sell_fees(gross, date):
    stamp = STAMP_BEFORE if date < STAMP_CUT else STAMP_AFTER
    return max(MIN_COMM, gross * COMM) + gross * TRANSFER + gross * stamp


def time_exit(g, entry_i, shares, invested, mode_reason="max_holding"):
    """第 15 个交易日收盘卖；跌停/停牌则顺延到下一个可卖开盘。"""
    end_i = entry_i + 14
    if end_i < len(g):
        r = g.iloc[end_i]
        if np.isfinite(r.close) and np.isfinite(r.volume) and r.volume > 0:
            lim = _down_limit(r.preclose, int(r.is_st))
            if not (np.isfinite(lim) and abs(float(r.close) - lim) <= 0.011):
                gross = shares * float(r.close)
                proceeds = gross - sell_fees(gross, r.date)
                return dict(exit_date=r.date, exit_px=float(r.close),
                            pnl=proceeds - invested, ret=proceeds / invested - 1,
                            reason=mode_reason, holding_sessions=end_i - entry_i + 1)
    k = end_i + 1
    while k < len(g):
        r = g.iloc[k]
        if np.isfinite(r.open) and np.isfinite(r.volume) and r.volume > 0:
            lim = _down_limit(r.preclose, int(r.is_st))
            if not (np.isfinite(lim) and abs(float(r.open) - lim) <= 0.011):
                gross = shares * float(r.open)
                proceeds = gross - sell_fees(gross, r.date)
                return dict(exit_date=r.date, exit_px=float(r.open),
                            pnl=proceeds - invested, ret=proceeds / invested - 1,
                            reason=mode_reason, holding_sessions=k - entry_i + 1)
        k += 1
    k = len(g) - 1
    gross = shares * float(g.iloc[k].close)
    proceeds = gross - sell_fees(gross, g.iloc[k].date)
    return dict(exit_date=g.iloc[k].date, exit_px=float(g.iloc[k].close),
                pnl=proceeds - invested, ret=proceeds / invested - 1,
                reason="delist_last_close", holding_sessions=k - entry_i + 1)


def simulate(g, j, mode):
    if j + 3 >= len(g) or j + 3 + 14 >= len(g):
        return dict(status="pending_entry", reason="insufficient_future")
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
    invested = gross_buy + buy_fees(gross_buy)
    entry_hfq = float(e.hfq_open)
    entry_date = e.date
    target = entry_hfq * 1.2  # trailing 的激活线；固定止盈会覆盖

    if mode in ("tp20", "tp25", "tp30"):
        tp = {"tp20": 0.20, "tp25": 0.25, "tp30": 0.30}[mode]
        target = entry_hfq * (1 + tp)
        for k in range(1, 15):
            r = g.iloc[j + 3 + k]
            if not np.isfinite(r.volume) or r.volume <= 0:
                continue
            if np.isfinite(r.hfq_open) and r.hfq_open >= target:
                raw = float(r.open)
                hfq = float(r.hfq_open)
                reason = "take_profit_open"
                break
            if np.isfinite(r.hfq_high) and r.hfq_high >= target:
                raw = float(e.open) * (target / entry_hfq)
                hfq = target
                reason = "take_profit_intraday"
                break
        else:
            return dict(status="filled", entry_date=entry_date, entry_px=float(e.open),
                        entry_hfq=entry_hfq, shares=shares, invested=invested,
                        **time_exit(g, j + 3, shares, invested))
        return dict(status="filled", entry_date=entry_date, entry_px=float(e.open),
                    entry_hfq=entry_hfq, shares=shares, invested=invested,
                    exit_date=r.date, exit_px=raw, pnl=shares * raw - sell_fees(shares * raw, r.date) - invested,
                    ret=(shares * raw - sell_fees(shares * raw, r.date)) / invested - 1,
                    reason=reason, holding_sessions=k + 1)

    if mode == "hold15":
        return dict(status="filled", entry_date=entry_date, entry_px=float(e.open),
                    entry_hfq=entry_hfq, shares=shares, invested=invested,
                    **time_exit(g, j + 3, shares, invested))

    if mode in ("trail5", "trail8", "trail10"):
        trail = {"trail5": 0.05, "trail8": 0.08, "trail10": 0.10}[mode]
        active = False
        max_high = entry_hfq
        stop = None
        active_i = None
        for k in range(0, 15):
            idx = j + 3 + k
            r = g.iloc[idx]
            if not np.isfinite(r.volume) or r.volume <= 0:
                continue
            # 先用昨日收盘后的 trailing stop 判断今天；再用今天的高点更新明天用的 stop
            if active and k > active_i:
                if np.isfinite(r.open) and r.open <= stop:
                    lim = _down_limit(r.preclose, int(r.is_st))
                    if np.isfinite(lim) and abs(float(r.open) - lim) <= 0.011:
                        continue
                    raw = float(r.open)
                elif np.isfinite(r.low) and r.low <= stop:
                    raw = float(e.open) * (stop / entry_hfq)
                else:
                    raw = None
                if raw is not None:
                    gross = shares * raw
                    proceeds = gross - sell_fees(gross, r.date)
                    return dict(status="filled", entry_date=entry_date, entry_px=float(e.open),
                                entry_hfq=entry_hfq, shares=shares, invested=invested,
                                exit_date=r.date, exit_px=raw, pnl=proceeds - invested,
                                ret=proceeds / invested - 1, reason=mode,
                                holding_sessions=k + 1)
            if np.isfinite(r.hfq_high):
                max_high = max(max_high, float(r.hfq_high))
            if not active and max_high >= target:
                active = True
                stop = max_high * (1 - trail)
                active_i = k
            elif active:
                stop = max(stop, max_high * (1 - trail))
        return dict(status="filled", entry_date=entry_date, entry_px=float(e.open),
                    entry_hfq=entry_hfq, shares=shares, invested=invested,
                    **time_exit(g, j + 3, shares, invested))

    raise ValueError(mode)


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

    cfg = Config(start="2005-01-01", end="2026-10-23", universe="all", gate="all",
                 extra_filter="is_st == 0", max_rank=2, label="C_top2")
    sig = build_signals(panel, stocks, cfg)
    cnt = sig.groupby("date").size()
    high = sorted(cnt[cnt >= 100].index)

    rows, trades_all = [], []
    for filt in ["PIT"]:
        for mode in ["tp20", "tp25", "tp30", "hold15", "trail5", "trail8", "trail10"]:
            cand = []
            for T in high:
                top = sig[sig.date == T].sort_values("rank").head(2)
                for r in top.itertuples():
                    g = by_code.get(r.code)
                    if g is None:
                        continue
                    pos = g.index[g.date == T]
                    if len(pos) == 0:
                        continue
                    sim = simulate(g, int(pos[0]), mode)
                    cand.append(dict(信号日=T, 信号数=int(cnt.loc[T]), 排名=int(r.rank),
                                     代码=r.code, 名称=names.get(r.code, ""), **sim))
            cdf = pd.DataFrame(cand)
            filled = cdf[cdf.status == "filled"].sort_values(["entry_date", "信号日", "排名"]).copy()
            held = {}
            keep = []
            for r in filled.itertuples():
                if r.代码 in held and held[r.代码] >= r.entry_date:
                    keep.append(False)
                else:
                    keep.append(True)
                    held[r.代码] = r.exit_date
            filled["重复"] = ~pd.Series(keep, index=filled.index)
            d = filled[~filled["重复"]].copy()
            if d.empty:
                continue
            events = []
            for r in d.itertuples():
                events.append((r.entry_date, 1, r.invested))
                events.append((r.exit_date, -1, -r.invested))
            events.sort(key=lambda x: (x[0], x[1]))
            cur_n = cur_cap = peak_n = peak_cap = 0
            for _, dn, dc in events:
                cur_n += dn
                cur_cap += dc
                peak_n = max(peak_n, cur_n)
                peak_cap = max(peak_cap, cur_cap)
            rows.append(dict(过滤=filt, 退出=mode, 笔数=len(d),
                             胜率=(d.pnl > 0).mean(), 总盈亏=d.pnl.sum(),
                             平均盈亏=d.pnl.mean(), 中位盈亏=d.pnl.median(),
                             最差=d.pnl.min(), 最好=d.pnl.max(),
                             峰值并发=peak_n, 峰值资金=peak_cap,
                             投入回报率=d.pnl.sum() / d.invested.sum()))
            d["过滤"] = filt
            d["退出"] = mode
            trades_all.append(d)

    sm = pd.DataFrame(rows)
    sm.to_csv(OUT / "止盈变体_汇总.csv", index=False, encoding="utf-8-sig")
    tr = pd.concat(trades_all, ignore_index=True)
    tr.to_csv(OUT / "止盈变体_逐笔.csv", index=False, encoding="utf-8-sig")
    print(sm.to_string(index=False))


if __name__ == "__main__":
    main()
