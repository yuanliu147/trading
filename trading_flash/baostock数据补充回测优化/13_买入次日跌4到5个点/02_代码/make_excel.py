#!/usr/bin/env python3
"""把结果 CSV 汇总为中文 Excel 数据表。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "03_结果"
XLSX = BASE / "买入次日跌4到5个点_数据表.xlsx"

HEADER_FILL = PatternFill("solid", fgColor="1F3B57")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=10)
TITLE_FONT = Font(bold=True, size=14, color="1F3B57")
THIN = Side(style="thin", color="D9DEE7")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, encoding="utf-8-sig")


def style_sheet(ws, title: str, df: pd.DataFrame, start_row: int = 1) -> None:
    ws.cell(row=start_row, column=1, value=title).font = TITLE_FONT
    header_row = start_row + 2
    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(row=header_row, column=j, value=col)
        c.fill = HEADER_FILL
        c.font = HEADER_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER
    for i, row in enumerate(df.itertuples(index=False), start=header_row + 1):
        for j, v in enumerate(row, start=1):
            c = ws.cell(row=i, column=j, value=(None if pd.isna(v) else v))
            c.border = BORDER
            c.alignment = Alignment(horizontal="center" if j > 1 else "left", vertical="center")
    for j, col in enumerate(df.columns, start=1):
        width = max(10, min(32, max(len(str(col)) * 2, *(len(str(x)) for x in df[col].head(40))) + 2))
        ws.column_dimensions[get_column_letter(j)].width = width
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
    ws.sheet_view.showGridLines = False


def main() -> None:
    summary = read("汇总_口径对照.csv")
    rank = read("分榜_主口径.csv")
    period = read("分阶段_主口径.csv")
    grid = read("相邻区间.csv")
    dist = read("收益分布_主口径.csv")
    path = read("后续路径_主口径.csv")
    mkt = read("大盘分状态.csv")
    detail = read("事件交易明细_主口径.csv")

    s = summary.set_index("口径")
    ev = s.loc["个股次日收盘跌4-5%（主口径）"]
    base = s.loc["基准（全部信号前二）"]

    concl = pd.DataFrame([
        ["研究问题", "买入后 D+1 收盘跌 4-5 个点，历史发生过多少次？最终赚还是亏？"],
        ["样本", "全部信号前二 1,256 笔（含退市、信号日排除风险警示），2005-02 至 2026-10"],
        ["事件定义", "D+1 收盘 / D+0 收盘 - 1 ∈ [-5%, -4%)"],
        ["事件笔数", int(ev["笔数"])],
        ["独立事件日", int(ev["独立日期数"])],
        ["最终样本", int(ev["最终样本"])],
        ["最终均值", f"{ev['均值(%)']:.2f}%"],
        ["最终中位数", f"{ev['中位数(%)']:.2f}%"],
        ["最终亏损率", f"{ev['亏损率(%)']:.2f}%"],
        ["基准均值 / 中位 / 亏损率", f"{base['均值(%)']:.2f}% / {base['中位数(%)']:.2f}% / {base['亏损率(%)']:.2f}%"],
        ["+20% 止盈率", f"{ev['止盈率(%)']:.2f}%"],
        ["非止盈笔数 / 均值", f"{int(ev['非止盈笔数'])} 笔 / {ev['非止盈均值(%)']:.2f}%"],
        ["最终 ≤ -15%", f"{ev['亏损15(%)']:.2f}%"],
        ["均值 95% CI", f"{ev['均值95CI下限(%)']:.2f}% ~ {ev['均值95CI上限(%)']:.2f}%"],
        ["一句话", "不是死亡判决，但中位收益从 +4.9% 打到 +0.6%，亏损率从 35.0% 升到 47.1%；均值靠 8 笔 +20% 止盈撑着。"],
        ["口径提醒", "收益为后复权、未扣成本；事件是买入后的条件，不是入场信号；样本小，不能做点预测。"],
    ], columns=["项目", "内容"])

    with pd.ExcelWriter(XLSX, engine="openpyxl") as writer:
        concl.to_excel(writer, sheet_name="结论", index=False)
        summary.to_excel(writer, sheet_name="汇总对比", index=False)
        detail.to_excel(writer, sheet_name="事件明细", index=False)
        rank.to_excel(writer, sheet_name="分榜", index=False)
        period.to_excel(writer, sheet_name="分阶段", index=False)
        grid.to_excel(writer, sheet_name="相邻区间", index=False)
        dist.to_excel(writer, sheet_name="收益分布", index=False)
        path.to_excel(writer, sheet_name="后续路径", index=False)
        mkt.to_excel(writer, sheet_name="大盘分状态", index=False)

        wb = writer.book
        style_sheet(wb["结论"], "买入次日跌 4-5 个点：结论", concl)
        style_sheet(wb["汇总对比"], "口径对照：事件 vs 基准", summary)
        style_sheet(wb["事件明细"], "35 笔事件明细", detail)
        style_sheet(wb["分榜"], "榜一 vs 榜二", rank)
        style_sheet(wb["分阶段"], "2005-2016 vs 2017-2026", period)
        style_sheet(wb["相邻区间"], "D+1 涨跌幅相邻区间", grid)
        style_sheet(wb["收益分布"], "最终收益分布", dist)
        style_sheet(wb["后续路径"], "事件后路径", path)
        style_sheet(wb["大盘分状态"], "大盘 D+1 跌 4-5% 分状态", mkt)

    print("written", XLSX)


if __name__ == "__main__":
    main()
