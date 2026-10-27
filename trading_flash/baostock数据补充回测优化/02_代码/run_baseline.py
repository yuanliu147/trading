#!/usr/bin/env python3
"""基准对比：同花顺原口径（仅当前上市） vs baostock 含退市股。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from strategy_lib import (BARS, Config, PANEL, load_matrices, load_panel, load_stocks,
                          annual, build_signals, metrics, run, save_result)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "03_结果"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2017-10-19")
    parser.add_argument("--end", default="2026-10-23")
    parser.add_argument("--tag", default="2017_2026")
    parser.add_argument("--skip-matrices", action="store_true")
    args = parser.parse_args()

    panel = load_panel()
    stocks = load_stocks()
    M = load_matrices()
    rows = []
    sets = {}
    for label, universe, gate in [
        ("A_仅当前上市", "listed", "listed"),
        ("B_含退市股", "all", "all"),
    ]:
        cfg = Config(start=args.start, end=args.end, universe=universe, gate=gate, label=label)
        sig = build_signals(panel, stocks, cfg)
        res = run(sig, M, stocks, cfg)
        m = save_result(res, cfg, OUT / args.tag, label)
        rows.append(dict(标签=label, **m, 信号日=sig.date.nunique(), 候选行=len(sig),
                         信号股票=sig.code.nunique()))
        sets[label] = sig
        print(label, json.dumps({k: (round(v, 4) if isinstance(v, float) else str(v))
                                 for k, v in m.items()}, ensure_ascii=False))
    # 信号重叠
    a = sets["A_仅当前上市"]; b = sets["B_含退市股"]
    a2 = set(zip(a.loc[a["rank"] <= 2, "date"], a.loc[a["rank"] <= 2, "code"]))
    b2 = set(zip(b.loc[b["rank"] <= 2, "date"], b.loc[b["rank"] <= 2, "code"]))
    overlap = len(a2 & b2)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / f"基准对比_{args.tag}.csv", index=False, encoding="utf-8-sig")
    info = dict(period=[args.start, args.end], top2_a=len(a2), top2_b=len(b2),
                top2_overlap=overlap, top2_jaccard=overlap / len(a2 | b2) if a2 | b2 else None,
                signal_days_a=int(a.date.nunique()), signal_days_b=int(b.date.nunique()))
    (OUT / f"基准重叠_{args.tag}.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
