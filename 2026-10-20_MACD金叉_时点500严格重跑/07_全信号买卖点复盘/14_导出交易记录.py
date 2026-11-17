#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 1 槽 / 100 槽的关键回测交易日志导出为中文表头，并生成索引。

只做展示层转换：不改动原始交易日志，所有数值仍来自回测引擎。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "04_单槽精选"

KEYS = [
    ("交易_2010起_raw_w1.0.csv", "1槽100%_原始排序_2010起_明细.csv",
     "1 槽 100%：F2 + b3(10/8) + 硬止损8%，原始 hist 排序"),
    ("交易_2010起_raw_w0.3.csv", "1槽30%_原始排序_2010起_明细.csv",
     "1 槽 30%：F2 + b3(10/8) + 硬止损8%，原始 hist 排序"),
    ("交易_2010起_norm_w1.0.csv", "1槽100%_价格归一化_2010起_明细.csv",
     "1 槽 100%：F2 + b3(10/8) + 硬止损8%，hist/close 排序"),
    ("交易_2010起_norm_w0.3.csv", "1槽30%_价格归一化_2010起_明细.csv",
     "1 槽 30%：F2 + b3(10/8) + 硬止损8%，hist/close 排序"),
    ("交易_100槽_归一化.csv", "100槽_价格归一化_2010起_明细.csv",
     "100 槽（上限）：F1 + b3(10/8) + 指数MA20，hist/close 排序"),
]

REASON = {"death_cross": "死叉", "b3_trailing": "b3追踪",
          "hard_stop": "硬止损", "time_stop": "时间止损",
          "full": "全平", "退市/长期停牌": "退市/长期停牌"}

COLMAP = {
    "entry_id": "编号", "code": "代码", "name": "名称",
    "signal_date": "信号日", "entry_date": "买入日", "entry_price": "买入价",
    "exit_date": "卖出日", "exit_price": "卖出价", "exit_price_date": "计价日",
    "shares": "股数", "total_invested": "投入资金", "net_proceeds": "净回款",
    "total_pnl": "净盈亏", "total_ret": "净收益率", "hold_days": "持有交易日",
    "reason": "离场原因", "peak_profit_pct": "峰值浮盈",
}


def main() -> None:
    sig = pd.read_csv(HERE / "01_数据" / "信号特征_时点500_内部.csv",
                      parse_dates=["signal_date"], usecols=["code", "signal_date", "name"])
    names = sig.drop_duplicates(["code", "signal_date"])
    index_rows = []
    for src, dst, desc in KEYS:
        p = OUT / src
        d = pd.read_csv(p, parse_dates=["signal_date", "entry_date", "exit_date", "exit_price_date"])
        d = d.merge(names, on=["code", "signal_date"], how="left")
        d["name"] = d["name"].fillna("—")
        d["reason"] = d["reason"].map(REASON).fillna(d["reason"])
        d["净收益率%"] = d["total_ret"] * 100
        d["峰值浮盈%"] = d["peak_profit_pct"] * 100
        d = d.rename(columns=COLMAP)
        cols = ["编号", "代码", "名称", "信号日", "买入日", "买入价", "卖出日", "卖出价",
                "计价日", "股数", "投入资金", "净回款", "净盈亏", "净收益率", "净收益率%",
                "持有交易日", "离场原因", "峰值浮盈", "峰值浮盈%"]
        d = d[cols].sort_values(["买入日", "编号"]).reset_index(drop=True)
        d.to_csv(OUT / dst, index=False, encoding="utf-8-sig")
        wins = d[d["净收益率"] > 0]
        by_pnl = d.sort_values("净盈亏", ascending=False)
        total_pnl = float(by_pnl["净盈亏"].sum())
        top5 = float(by_pnl["净盈亏"].head(5).sum())
        index_rows.append({
            "文件": dst,
            "策略": desc,
            "区间": f"{d['买入日'].min().date()} ~ {d['卖出日'].max().date()}",
            "交易数": len(d),
            "胜率%": round(len(wins) / len(d) * 100, 2),
            "平均每笔%": round(d["净收益率"].mean() * 100, 3),
            "累计净盈亏": round(d["净盈亏"].sum(), 2),
            "最大单笔盈利": round(d["净盈亏"].max(), 2),
            "最大单笔亏损": round(d["净盈亏"].min(), 2),
            "平均持有交易日": round(d["持有交易日"].mean(), 1),
            "前5笔占比%": round(top5 / total_pnl * 100, 2) if total_pnl > 0 else None,
            "剔除前5笔盈亏": round(total_pnl - top5, 2),
            "备注": "已平仓口径；不含期末未平仓",
        })
        print(f"{dst}: {len(d):,} 笔，胜率 {index_rows[-1]['胜率%']}%，"
              f"平均 {index_rows[-1]['平均每笔%']}%")
    idx = pd.DataFrame(index_rows)
    idx.to_csv(OUT / "交易记录_索引.csv", index=False, encoding="utf-8-sig")
    # Markdown 索引
    lines = ["# 交易记录索引", "",
             "均为逐笔已平仓记录；金额为人民币元，收益率为小数（另附百分数列）。",
             "", "| 文件 | 策略 | 区间 | 交易数 | 胜率 | 平均每笔 | 累计净盈亏 | 最大单笔盈利 | 最大单笔亏损 | 平均持有 | 前5笔占比 | 剔除前5笔盈亏 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in index_rows:
        lines.append(f"| [{r['文件']}]({r['文件']}) | {r['策略']} | {r['区间']} | {r['交易数']:,} | "
                     f"{r['胜率%']}% | {r['平均每笔%']}% | {r['累计净盈亏']:,.0f} | "
                     f"{r['最大单笔盈利']:,.0f} | {r['最大单笔亏损']:,.0f} | {r['平均持有交易日']} | "
                     f"{'' if r['前5笔占比%'] is None else str(r['前5笔占比%']) + '%'} | {r['剔除前5笔盈亏']:,.0f} |")
    (OUT / "交易记录_索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n保存 {OUT/'交易记录_索引.csv'} / 交易记录_索引.md")


if __name__ == "__main__":
    main()
