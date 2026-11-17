#!/usr/bin/env python3
"""603778 的完整信号序列：08 全信号池（到 10/23）+ 当前 10/26、10/28 信号。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "03_结果"
POOL = ROOT.parent / "不同时间入场点回测" / "08_重复信号与买入时点" / "03_结果" / "全部信号池_原始.csv"
CURRENT = ROOT.parent / "2026-10-24_主板恐慌反转_年化30" / "03_结果"


def main() -> None:
    pool = pd.read_csv(POOL)
    sub = pool[pool["code"] == "603778.SH"].copy()
    sub["信号日"] = pd.to_datetime(sub["signal_date"]).dt.strftime("%Y-%m-%d")
    sub["收益(%)"] = (sub["ret"] * 100).round(2)
    sub["是否前二"] = sub["top2"].map({True: "是", False: "否"})
    sub["状态"] = sub["status"].map({"ok": "已完成", "持有未满": "截至10/23未完成", "数据结束": "数据结束"}).fillna(sub["status"])
    rows = sub[["信号日", "rk", "是否前二", "收益(%)", "状态"]].rename(columns={"rk": "排名"})
    rows["数据来源"] = "08全信号池（到10/23）"
    rows["备注"] = ""

    # 10/15 当前进展
    mask = rows["信号日"] == "2026-10-15"
    rows.loc[mask, "备注"] = "11/2 相对 7.50 买入价 -7.73%"

    # 10/23 当前进展
    mask = rows["信号日"] == "2026-10-23"
    rows.loc[mask, "备注"] = "10/28 开盘 7.18 买入，11/2 收 6.92（-3.62%）"

    # 当前 10/26、10/28 信号
    for d in ["2026-10-26", "2026-10-28", "2026-11-02"]:
        p = CURRENT / f"当前信号_{d}.csv"
        if p.exists():
            cur = pd.read_csv(p)
            hit = cur[cur["thscode"].astype(str).str.contains("603778")]
            if len(hit):
                rank = int(hit["rank"].iloc[0])
                rows.loc[len(rows)] = {
                    "信号日": d, "排名": rank, "是否前二": "是" if rank <= 2 else "否",
                    "收益(%)": None, "状态": "当前信号", "数据来源": f"当前信号_{d}.csv",
                    "备注": {"2026-10-26": "10/29 买入，未满", "2026-10-28": "11/2 买入，未满",
                             "2026-11-02": "11/5 买入，未满"}[d],
                }
        elif d == "2026-11-02":
            rows.loc[len(rows)] = {
                "信号日": d, "排名": 1, "是否前二": "是", "收益(%)": None,
                "状态": "当前信号", "数据来源": "用户确认（11/2 前二）", "备注": "11/5 买入，未满",
            }

    rows = rows.sort_values("信号日").reset_index(drop=True)
    rows.to_csv(OUT / "603778_信号序列.csv", index=False, encoding="utf-8-sig")
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
