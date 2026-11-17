#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正版组合方案对比（使用已生成的交易明细，不重新跑信号）。

输出：修正版_组合方案对比.csv
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import ladder_doublek_corrected as c  # noqa: E402
import ladder_doublek_backtest as base  # noqa: E402

TRADES = HERE


def load(name, **kw):
    p = os.path.join(TRADES, name)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, parse_dates=["entry_date", "exit_date"], **kw)


def main():
    A = load("修正版_交易明细_A_ALL_2016_2026.csv")
    B = load("修正版_交易明细_B_ALL_2016_2026.csv")
    Ac = load("修正版_交易明细_A_close_ALL_2016_2026.csv")
    if A is None or B is None:
        raise SystemExit("缺少 A/B 交易明细")
    AB = pd.concat([A, B], ignore_index=True)
    ABc = pd.concat([Ac, B], ignore_index=True) if Ac is not None else None

    symbols = sorted(set(AB["symbol"]) | (set(Ac["symbol"]) if Ac is not None else set()))
    print(f"载入 {len(symbols)} 只标的的收盘价…")
    df = base.load_stocks(symbols)
    cm = c.build_close_map(df)

    variants = [
        ("A+B 放量排序 6仓", AB, "vol", 6, 0.02),
        ("A+B 代码排序 6仓", AB, "symbol", 6, 0.02),
        ("A+B 放量排序 3仓", AB, "vol", 3, 0.02),
        ("A+B 放量排序 2仓", AB, "vol", 2, 0.02),
        ("A+B 放量排序 2仓 风险1%", AB, "vol", 2, 0.01),
        ("A 放量排序 6仓", A, "vol", 6, 0.02),
        ("A 放量排序 3仓", A, "vol", 3, 0.02),
        ("B 放量排序 6仓", B, "vol", 6, 0.02),
        ("B 放量排序 3仓", B, "vol", 3, 0.02),
        ("B 放量排序 2仓", B, "vol", 2, 0.02),
    ]
    if ABc is not None:
        variants += [
            ("A_close+B 放量排序 6仓", ABc, "vol", 6, 0.02),
            ("A_close+B 放量排序 2仓", ABc, "vol", 2, 0.02),
        ]

    rows = []
    for name, T, rank, mp, risk in variants:
        eq, taken, skipped, _ = c.simulate(T, cm, rank_by=rank, max_pos=mp, risk=risk)
        s = c.eq_stats(eq)
        rows.append({
            "方案": name,
            "成交笔数": taken,
            "放弃笔数": skipped,
            "期末净值": round(s["final"], 0),
            "总收益": s["total"],
            "年化": s["cagr"],
            "最大回撤": s["mdd"],
        })
        print(f"{name}: 成交{taken} 放弃{skipped} | 期末 {s['final']:.0f} | "
              f"总 {s['total']:+.1%} | 年化 {s['cagr']:+.1%} | MDD {s['mdd']:.1%}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TRADES, "修正版_组合方案对比.csv"), index=False)
    print(f"\n已写 {os.path.join(TRADES, '修正版_组合方案对比.csv')}")


if __name__ == "__main__":
    main()
