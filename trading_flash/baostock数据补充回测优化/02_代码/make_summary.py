#!/usr/bin/env python3
"""汇总核心结论，生成报告用表。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "03_结果"


def load_metric(path: Path) -> dict:
    return json.loads(path.read_text())


def main():
    rows = []
    specs = [
        ("2017-2026", "A_仅当前上市", R / "2017_2026/A_仅当前上市_指标.json"),
        ("2017-2026", "B_含退市_无ST过滤", R / "2017_2026/B_含退市股_指标.json"),
        ("2017-2026", "C_含退市_排除ST", R / "指标组合_2017_2026/C_排除ST_指标.json"),
        ("2005-2026", "A_仅当前上市", R / "2005_2026/A_仅当前上市_指标.json"),
        ("2005-2026", "B_含退市_无ST过滤", R / "2005_2026/B_含退市股_指标.json"),
        ("2005-2026", "C_含退市_排除ST", R / "指标优化/2005-2026/C_排除ST_指标.json"),
        ("2005-2016", "C_含退市_排除ST", R / "风险叠加/2005-2016/C_基础_无止损_指标.json"),
    ]
    for period, label, path in specs:
        m = load_metric(path)
        rows.append(dict(期间=period, 口径=label, 年化=m["cagr"], 最大回撤=m["max_dd"],
                         卡玛=m["calmar"], 夏普=m["sharpe"], 交易数=m["n_trades"],
                         胜率=m["win"], 单笔均值=m["avg"], 单笔中位数=m["median"],
                         持仓占比=m["exposure"], 期末倍数=m["final"],
                         期末金额=m["final_yuan"], 起始=str(m["first_date"]),
                         结束=str(m["last_date"]), 年数=m["years"]))
    df = pd.DataFrame(rows)
    df.to_csv(R / "核心结论.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    # markdown 表
    def pct(x):
        return f"{x*100:.1f}%"

    lines = ["| 期间 | 口径 | 年化 | 最大回撤 | 卡玛 | 交易数 | 胜率 | 期末倍数 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in df.itertuples():
        lines.append(f"| {r.期间} | {r.口径} | {pct(r.年化)} | {pct(r.最大回撤)} | "
                     f"{r.卡玛:.2f} | {r.交易数} | {pct(r.胜率)} | {r.期末倍数:.2f}x |")
    (R / "核心结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
