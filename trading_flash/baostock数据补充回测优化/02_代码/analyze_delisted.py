#!/usr/bin/env python3
"""退市股影响：直接交易、Top2 重叠、002684 个案。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lib import Config, build_signals, load_panel, load_stocks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "03_结果"
EVID = ROOT / "05_证据"


def top2(sig: pd.DataFrame) -> pd.DataFrame:
    x = sig[sig["rank"] <= 2][["date", "code", "rank", "drawdown250", "is_st"]].copy()
    x["date"] = pd.to_datetime(x.date)
    return x


def main():
    panel = load_panel()
    stocks = load_stocks()
    periods = [("2017-2026", "2017-10-19", "2026-10-23"),
               ("2005-2026", "2005-01-01", "2026-10-23")]
    rows = []
    sets = {}
    for period, start, end in periods:
        for label, universe, gate, flt in [
            ("A_仅上市_无ST过滤", "listed", "listed", ""),
            ("B_含退市_无ST过滤", "all", "all", ""),
            ("C_含退市_排除ST", "all", "all", "is_st == 0"),
        ]:
            cfg = Config(start=start, end=end, universe=universe, gate=gate,
                         extra_filter=flt, label=label)
            sig = build_signals(panel, stocks, cfg)
            sets[(period, label)] = sig
            t = top2(sig)
            t.to_csv(OUT / f"Top2_{period}_{label}.csv", index=False, encoding="utf-8-sig")
    for period, _, _ in periods:
        a = sets[(period, "A_仅上市_无ST过滤")]
        b = sets[(period, "B_含退市_无ST过滤")]
        c = sets[(period, "C_含退市_排除ST")]
        for left, right, lname, rname in [(a, b, "A", "B"), (a, c, "A", "C"), (b, c, "B", "C")]:
            lt = set(zip(left.loc[left["rank"] <= 2, "date"], left.loc[left["rank"] <= 2, "code"]))
            rt = set(zip(right.loc[right["rank"] <= 2, "date"], right.loc[right["rank"] <= 2, "code"]))
            rows.append(dict(期间=period, 对比=f"{lname} vs {rname}", 左Top2=len(lt),
                             右Top2=len(rt), 交集=len(lt & rt),
                             Jaccard=len(lt & rt) / len(lt | rt) if lt | rt else np.nan,
                             相同日=len({d for d, _ in lt} & {d for d, _ in rt}),
                             信号日=left.date.nunique()))
    pd.DataFrame(rows).to_csv(OUT / "Top2重叠.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))

    # 退市股交易明细
    for period, label in [("2017-2026", "B_含退市_无ST过滤"), ("2017-2026", "C_含退市_排除ST"),
                          ("2005-2026", "B_含退市_无ST过滤"), ("2005-2026", "C_含退市_排除ST")]:
        if label == "B_含退市_无ST过滤":
            f = OUT / ("2017_2026" if period == "2017-2026" else "2005_2026") / "B_含退市股_逐笔.csv"
        elif period == "2017-2026":
            f = OUT / "指标组合_2017_2026" / "C_排除ST_逐笔.csv"
        else:
            f = OUT / "指标优化" / period / "C_排除ST_逐笔.csv"
        tr = pd.read_csv(f)
        d = tr[tr.delisted == 1].copy()
        d["pnl"] = d.proceeds - d.invested
        d.to_csv(OUT / f"退市股交易_{period}_{label}.csv", index=False, encoding="utf-8-sig")
        print(period, label, "delisted trades", len(d), "avg", round(d.ret.mean(), 4) if len(d) else None,
              "pnl", round(d.pnl.sum(), 0) if len(d) else None)

    # 002684 过程
    con_panel = panel[panel.code == "sz.002684"][
        ["date", "close", "drawdown250", "return_250d", "rsi14", "close_location",
         "amount20", "is_st", "market_return20_all"]]
    con_panel.to_csv(EVID / "002684_信号日特征.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
