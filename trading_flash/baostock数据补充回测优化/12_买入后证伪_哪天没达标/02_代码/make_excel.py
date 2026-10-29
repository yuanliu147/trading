#!/usr/bin/env python3
"""生成中文 Excel 数据表。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "03_结果"
XLSX = BASE / "买入后证伪_数据表.xlsx"

SHEETS = [
    ("01_基准样本", "基准_样本.csv", {"胜率", "最终亏损率", "均值", "中位", "止盈率", "15日亏损率",
                                 "平均最大浮亏", "浮亏<=-10%"}),
    ("02_基准逐日", "基准_逐日.csv", {"全样本路径均值", "全样本高于买入价概率", "全样本最终止盈占比",
                                 "持仓路径均值", "持仓路径中位", "持仓高于买入价概率", "持仓最终亏损率",
                                 "剩余收益均值", "之后5日再跌5%", "次日下跌率", "3日下跌率"}),
    ("03_收益阈值", "收益阈值.csv", {"阈值", "占比", "最终亏损率", "未触发亏损率", "深度亏损率",
                                "之后5日再跌5%", "次日下跌率", "剩余收益均值", "之后止盈率", "早退相对持有",
                                "亏损率_CI下", "亏损率_CI上"}),
    ("04_结构特征", "结构特征.csv", {"占比", "最终亏损率", "未触发亏损率", "深度亏损率",
                                "之后5日再跌5%", "次日下跌率", "剩余收益均值", "之后止盈率", "早退相对持有",
                                "亏损率_CI下", "亏损率_CI上"}),
    ("05_预警评分", "预警评分.csv", {"占比", "最终亏损率", "未触发亏损率", "深度亏损率",
                                "之后5日再跌5%", "次日下跌率", "剩余收益均值", "之后止盈率", "早退相对持有",
                                "亏损率_CI下", "亏损率_CI上"}),
    ("06_组合规则", "组合规则.csv", {"占比", "最终亏损率", "未触发亏损率", "深度亏损率",
                                "之后5日再跌5%", "次日下跌率", "剩余收益均值", "之后止盈率", "早退相对持有",
                                "亏损率_CI下", "亏损率_CI上"}),
    ("07_早退反事实", "早退反事实.csv", {"持有均值", "早退均值", "持有<=-10%", "早退<=-10%",
                                  "持有<=-15%", "早退<=-15%", "持有最差", "早退最差"}),
    ("08_预测力", "预测力_AUC.csv", {"基准亏损率"}),
    ("09_逐笔摘要", "逐笔摘要.csv", set()),
    ("10_候选数量基准", "候选数量_基准.csv", {"胜率", "最终亏损率", "均值", "中位", "止盈率", "平均最大浮亏"}),
    ("11_候选数量预警", "候选数量_预警.csv", {"触发率", "最终亏损率", "同桶基准亏损率", "深度亏损率",
                                          "之后止盈率", "早退相对持有", "95%置信下", "95%置信上"}),
    ("12_路径三态基准", "路径三态_基准.csv", {"占比", "最终亏损率", "深度亏损率", "均值", "中位",
                                          "止盈率", "平均最大浮亏", "D+3均值", "D+5均值",
                                          "95%置信下", "95%置信上"}),
    ("13_路径三态逐日", "路径三态_逐日.csv", {"路径均值", "路径中位", "高于买入价概率", "高于D+3收盘概率"}),
    ("14_路径三态转弱", "路径三态_转弱.csv", {"触发率", "最终亏损率", "同态基准", "深度亏损率",
                                          "之后止盈率", "剩余收益均值", "早退相对持有",
                                          "持有均值", "早退均值", "持有<=-15%", "早退<=-15%",
                                          "95%置信下", "95%置信上"}),
    ("15_路径阈值", "路径三态_阈值.csv", {"上涨阈值", "下跌阈值", "最终亏损率", "深度亏损率", "均值", "止盈率"}),
    ("16_市场环境", "路径三态_市场环境.csv", {"最终亏损率", "均值"}),
]

HEADER_FILL = PatternFill("solid", fgColor="1F2937")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=10)
THIN = Side(style="thin", color="D1D5DB")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
ALT_FILL = PatternFill("solid", fgColor="F9FAFB")
RED_FILL = PatternFill("solid", fgColor="FEF2F2")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / name, encoding="utf-8-sig")


def summary_table() -> pd.DataFrame:
    s = read("基准_样本.csv")
    c = read("组合规则.csv")
    e = read("早退反事实.csv")

    rows = []
    for _, r in s.iterrows():
        rows.append({
            "类别": "基准", "样本": r["样本"], "期间": r["期间"], "检查点": "—", "规则": "全部",
            "笔数": r["笔数"], "最终亏损率": r["最终亏损率"], "最终≤-10%": np.nan,
            "次日下跌率": np.nan, "剩余收益均值": r["均值"], "之后止盈率": np.nan,
            "持有均值": np.nan, "早退均值": np.nan,
            "早退-持有": np.nan, "95%CI下": np.nan, "95%CI上": np.nan,
            "说明": "平均最大浮亏 %.1f%%；最大浮亏≤-10%% 占 %.1f%%；全期止盈率 %.1f%%" % (r["平均最大浮亏"] * 100, r["浮亏<=-10%"] * 100, r["止盈率"] * 100),
        })
    rules = ["收益<-2%", "未达+3%", "未创买入后新高", "预警分>=4", "跌破买入价+未创新高+破5日线"]
    names = {"跌破买入价+未创新高+破5日线": "三项复合"}
    for _, r in c[(c["样本"] == "全部信号前二") & (c["期间"] == "全期") & (c["规则"].isin(rules))].iterrows():
        rows.append({
            "类别": "预警规则", "样本": r["样本"], "期间": r["期间"], "检查点": f"D+{int(r['天数'])}",
            "规则": names.get(r["规则"], r["规则"]), "笔数": r["N"], "最终亏损率": r["最终亏损率"],
            "最终≤-10%": r["深度亏损率"], "次日下跌率": r["次日下跌率"],
            "剩余收益均值": r["剩余收益均值"], "之后止盈率": r["之后止盈率"],
            "持有均值": np.nan, "早退均值": np.nan,
            "早退-持有": r["早退相对持有"], "95%CI下": r["亏损率_CI下"], "95%CI上": r["亏损率_CI上"],
            "说明": "未触发组亏损率 %.1f%%" % (r["未触发亏损率"] * 100),
        })
    for _, r in c[(c["样本"] == "全部信号前二") & (c["期间"] == "2017-2026") & (c["天数"] == 5) & (c["规则"].isin(rules))].iterrows():
        rows.append({
            "类别": "2017-2026复核", "样本": r["样本"], "期间": r["期间"], "检查点": "D+5",
            "规则": names.get(r["规则"], r["规则"]), "笔数": r["N"], "最终亏损率": r["最终亏损率"],
            "最终≤-10%": r["深度亏损率"], "次日下跌率": r["次日下跌率"],
            "剩余收益均值": r["剩余收益均值"], "之后止盈率": r["之后止盈率"],
            "持有均值": np.nan, "早退均值": np.nan,
            "早退-持有": r["早退相对持有"], "95%CI下": r["亏损率_CI下"], "95%CI上": r["亏损率_CI上"],
            "说明": "未触发组亏损率 %.1f%%" % (r["未触发亏损率"] * 100),
        })
    for _, r in e[(e["样本"] == "全部信号前二") & (e["期间"] == "全期") & (e["天数"] == 5) & (e["规则"].isin(rules))].iterrows():
        rows.append({
            "类别": "D+5早退反事实", "样本": r["样本"], "期间": r["期间"], "检查点": "D+5",
            "规则": names.get(r["规则"], r["规则"]), "笔数": r["笔数"], "最终亏损率": np.nan,
            "最终≤-10%": r["持有<=-10%"], "次日下跌率": np.nan, "剩余收益均值": np.nan,
            "之后止盈率": np.nan, "持有均值": r["持有均值"], "早退均值": r["早退均值"],
            "早退-持有": r["早退-持有pp"] / 100, "95%CI下": np.nan, "95%CI上": np.nan,
            "说明": "持有≤-10%% %.1f%%，早退≤-10%% %.1f%%；持有≤-15%% %.1f%%，早退≤-15%% %.1f%%" % (
                r["持有<=-10%"] * 100, r["早退<=-10%"] * 100, r["持有<=-15%"] * 100, r["早退<=-15%"] * 100),
        })
    sb = read("路径三态_基准.csv")
    sw = read("路径三态_转弱.csv")
    for st, label in [("上涨", "D+3 上涨（≥+3%）"), ("中震荡", "D+3 中震荡（-2%~+3%）"),
                      ("下跌", "D+3 下跌（≤-2%）")]:
        r = sb[(sb["样本"] == "全部信号前二") & (sb["期间"] == "全期") & (sb["状态"] == st)].iloc[0]
        rows.append({
            "类别": "D+3三态基准", "样本": r["样本"], "期间": r["期间"], "检查点": "D+3",
            "规则": label, "笔数": r["笔数"], "最终亏损率": r["最终亏损率"],
            "最终≤-10%": r["深度亏损率"], "次日下跌率": np.nan, "剩余收益均值": r["均值"],
            "之后止盈率": r["止盈率"], "持有均值": r["均值"], "早退均值": np.nan,
            "早退-持有": np.nan, "95%CI下": r["95%置信下"], "95%CI上": r["95%置信上"],
            "说明": "中位 %.1f%%；平均最大浮亏 %.1f%%；D+3均值 %.1f%%" % (
                r["中位"] * 100, r["平均最大浮亏"] * 100, r["D+3均值"] * 100),
        })
    key_rules = [("上涨", "D+5 回吐到成本下"), ("上涨", "D+5 距 D+0–5 最高回撤≥10%"),
                 ("中震荡", "D+5 < -2%"), ("中震荡", "D+5 < -5%"),
                 ("下跌", "D+5 预警分≥4"), ("下跌", "D+5 三项复合")]
    for st, rule in key_rules:
        r = sw[(sw["样本"] == "全部信号前二") & (sw["期间"] == "全期") &
               (sw["状态"] == st) & (sw["规则"] == rule)].iloc[0]
        rows.append({
            "类别": "三态转弱", "样本": r["样本"], "期间": r["期间"], "检查点": "D+5",
            "规则": f"{st}｜{rule}", "笔数": r["触发笔数"], "最终亏损率": r["最终亏损率"],
            "最终≤-10%": r["深度亏损率"], "次日下跌率": np.nan,
            "剩余收益均值": r["剩余收益均值"], "之后止盈率": r["之后止盈率"],
            "持有均值": r["持有均值"], "早退均值": r["早退均值"],
            "早退-持有": r["早退相对持有"], "95%CI下": r["95%置信下"], "95%CI上": r["95%置信上"],
            "说明": "同态基准 %.1f%%；触发率 %.1f%%；持有≤-15%% %.1f%%，早退≤-15%% %.1f%%" % (
                r["同态基准"] * 100, r["触发率"] * 100, r["持有<=-15%"] * 100, r["早退<=-15%"] * 100),
        })
    return pd.DataFrame(rows)


def calibration_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"项目": "数据", "说明": "baostock 主板日线（后复权收益+不复权成交/涨跌停），含退市股；2004-2026"},
        {"项目": "样本", "说明": "C 池：信号日排除风险警示（is_st=0），按 250 日回撤最深取前二；全部信号前二 1256 笔，实际成交 219 笔"},
        {"项目": "信号", "说明": "250 日收益>0，自 250 日高点回撤<-45%，RSI14<35，收盘位置>0.45，20 日均额≥1000 万，上市≥250 日，主板等权 20 日<-5%"},
        {"项目": "买入", "说明": "信号日 T 收盘后，T+3 开盘买入；一字涨停/停牌不买；路径统计假设可买"},
        {"项目": "持有", "说明": "15 个交易日；+20% 盘中止盈从买入次日（D+1）起算；跌停/停牌顺延（实际成交按逐笔记录）"},
        {"项目": "第 d 天", "说明": "d=0 为买入日收盘；d=1 为买入后第 1 个交易日；所有特征只用 d 日收盘及之前数据"},
        {"项目": "最终亏损", "说明": "按实际规则（+20% 止盈或 15 日到期）的最终收益<0；深度亏损=最终≤-10%"},
        {"项目": "早退", "说明": "d 日收盘触发，d+1 开盘卖出；不建模跌停卖不出；实际成交口径扣 0.2% 卖出费用"},
        {"项目": "预警分", "说明": "0-5 分：收盘低于买入价、未创买入后新高、收盘跌破买入日最低、收盘低于5日线、连续两日低点下移"},
        {"项目": "三项复合", "说明": "收盘低于买入价 + 未创买入后新高 + 收盘低于5日线"},
        {"项目": "聚类", "说明": "同一代码信号日间隔≤3 个交易日合并为一个信号段；置信区间按信号段聚类稳健标准误"},
        {"项目": "价格", "说明": "路径/信号用后复权；实际成交用不复权价定股数、后复权算收益；除权造成的假破位已避免"},
        {"项目": "D+3三态", "说明": "D+3 收盘且尚未止盈：上涨≥+3%；中震荡 -2%~+3%；下跌≤-2%。D+3 前已止盈 84 笔不进入三态；分态是事后分组，不是入场信号"},
        {"项目": "三态转弱", "说明": "D+5 收盘可知；上涨组看回吐到成本下/距 D+0–5 最高回撤≥10%；中震荡组看 D+5<-2% 或 <-5%；下跌组看 D+5 预警分≥4。早退按 D+6 开盘"},
        {"项目": "市场环境", "说明": "主板等权指数相对250日线>+2%、120日收益>+3%、250日线20日斜率>0=上涨市；三项反向=下跌市；其余=中震荡市。信号日收盘可知"},
        {"项目": "峰值", "说明": "D+3 之后最高价/最高收盘出现的交易日；15 日窗口截断右尾，只作描述，不是规则"},
        {"项目": "限制", "说明": "信号级样本高度重叠，不是 1256 个独立机会；实际成交样本小；历史不代表未来，不构成投资建议"},
    ])


def write_sheet(writer, name: str, df: pd.DataFrame, pct_cols: set[str]):
    df.to_excel(writer, sheet_name=name, index=False)
    ws = writer.sheets[name]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    header_map = {
        "N": "笔数", "差值pp": "差值(百分点)", "早退-持有pp": "早退-持有(百分点)",
        "亏损率_CI下": "95%置信下", "亏损率_CI上": "95%置信上",
        "95%CI下": "95%置信下", "95%CI上": "95%置信上", "亏损率_se": "标准误",
        "AUC_收益": "收益AUC", "AUC_预警分": "预警分AUC", "AUC_复合": "复合AUC",
        "聚类数": "信号段数",
    }
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
        if cell.value in header_map:
            cell.value = header_map[cell.value]
    for j, col in enumerate(df.columns, 1):
        letter = get_column_letter(j)
        width = max(9, min(30, max(len(str(col)) * 1.7, df[col].astype(str).str.len().quantile(.9) * 1.1)))
        ws.column_dimensions[letter].width = width
        fmt = None
        if col in pct_cols:
            fmt = "0.0%"
        elif name == "09_逐笔摘要" and (col in {"模拟最终收益", "实际净收益", "信号回撤"} or (col.startswith("第") and "收益" in col)):
            fmt = "0.0%"
        elif name == "09_逐笔摘要" and "预警分" in col:
            fmt = "0"
        elif col in {"差值pp", "早退-持有pp"}:
            fmt = '0.0"pp"'
        elif col.startswith("AUC"):
            fmt = "0.000"
        elif col in {"笔数", "触发笔数", "N", "信号日", "信号日数", "交易数", "信号段", "信号段数",
                     "可评估笔数", "全样本笔数", "持仓笔数", "聚类数", "预警分", "天数", "rank",
                     "n_days", "峰值日_收盘中位"}:
            fmt = "0"
        if fmt:
            for row in range(2, ws.max_row + 1):
                ws.cell(row=row, column=j).number_format = fmt
    for row in range(2, ws.max_row + 1):
        if row % 2 == 0:
            for j in range(1, len(df.columns) + 1):
                ws.cell(row=row, column=j).fill = ALT_FILL
        for j in range(1, len(df.columns) + 1):
            ws.cell(row=row, column=j).border = BORDER
            ws.cell(row=row, column=j).alignment = Alignment(vertical="top", wrap_text=False)


def main():
    summary = summary_table()
    summary.to_csv(RES / "结论摘要.csv", index=False, encoding="utf-8-sig")
    calib = calibration_table()
    calib.to_csv(RES / "口径说明.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(XLSX, engine="openpyxl") as writer:
        calib.to_excel(writer, sheet_name="00_口径", index=False)
        ws = writer.sheets["00_口径"]
        ws.column_dimensions["A"].width = 14
        ws.column_dimensions["B"].width = 110
        for cell in ws[1]:
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center")
        for row in range(2, ws.max_row + 1):
            ws.row_dimensions[row].height = 34
            ws.cell(row=row, column=2).alignment = Alignment(wrap_text=True, vertical="top")
            ws.cell(row=row, column=1).alignment = Alignment(vertical="top")
            ws.cell(row=row, column=1).font = Font(bold=True)
        for name, csv, pct in SHEETS:
            df = read(csv)
            id_cols = [c for c in ["样本", "期间", "状态", "市场环境", "敏感性", "天数",
                                   "候选级别", "阈值", "上涨阈值", "下跌阈值", "规则", "预警分"]
                       if c in df.columns]
            df = df[id_cols + [c for c in df.columns if c not in id_cols]]
            write_sheet(writer, name, df, pct)
        # 结论摘要
        write_sheet(writer, "17_结论摘要", summary, {"最终亏损率", "最终≤-10%", "次日下跌率",
                                                "剩余收益均值", "之后止盈率", "持有均值", "早退均值", "早退-持有"})
        # 摘要表加粗关键行
        ws = writer.sheets["17_结论摘要"]
        for row in range(2, ws.max_row + 1):
            rule = ws.cell(row=row, column=5).value
            day = ws.cell(row=row, column=4).value
            if rule in {"收益<-2%", "三项复合"} and day in {"D+3", "D+5"}:
                for j in range(1, ws.max_column + 1):
                    ws.cell(row=row, column=j).fill = RED_FILL
    print("Excel 已生成：", XLSX, "大小 %.1f KB" % (XLSX.stat().st_size / 1024))
    print("结论摘要行数：", len(summary))


if __name__ == "__main__":
    main()
