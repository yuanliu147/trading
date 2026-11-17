#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成中文表头的研究数据表。"""
import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
OUT = os.path.join(BASE, "交易系统_数据表.xlsx")
writer = pd.ExcelWriter(OUT, engine="openpyxl")
TRADES = os.path.join(ROOT, "2026-10-18_成交量融合_2010-2025", "08_深度复盘_2016_2026", "逐笔_2016-2026_低波活跃_结合.csv")

t = pd.read_csv(TRADES, parse_dates=["signal_date", "buy", "sell"])
t["combo"] = t[["m1", "m2", "m3"]].astype(int).astype(str).agg("".join, axis=1)


def pct(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") * 100
            df.rename(columns={c: c + "%"}, inplace=True)
    return df

# 01 秘诀检验
claims = pd.DataFrame([
    ["只做趋势成立点、调整结束点", "443笔中含突破的378笔贡献全部利润；不含突破的65笔合计-14.1万；宽口径两类点相对普通上升日无稳定超额", "方向对，条件不够", "突破是必要，不是充分；回踩只做低波类型"],
    ["止损近、风险可控、成功率高、空间大", "443笔胜率34.3%，中位-3.28%；事件研究胜率39%-47%；止损率46%-65%", "成功率高不成立", "低胜率+右尾；止损近与空间大是交换"],
    ["其他时候不动", "约40笔/年；大量时间空仓；普通上升日也有右尾", "成立", "空仓是等待，不是择时能力"],
    ["止损进场前设好，到了就走", "77笔止损平均-10.13%，最差-19.58%；计划8%，实际因跳空更大", "原则对，执行要算跳空", "按1.5倍止损做最坏打算"],
    ["分批止盈，不指望卖最高点", "前5笔占净利润179.7%；固定2R/3R无稳定优势；破MA20的321笔贡献+153.7万", "不猜顶对，过早分批错", "最多1/3在+2R；剩余跟MA20"],
    ["单笔亏损不超过3%，仓位由止损空间决定", "8%止损对应37.5%仓位，4笔150%；20%上限下实际风险1.6%；443笔固定20%风险1.53%，回撤-38.9%", "3%是上限不是目标", "建议0.5%-1%；组合热预算≤3%"],
    ["连亏两次暂停，复盘反思", "连亏两次后下一笔胜率26.5% vs 34.3%，但2017/2019/2021/2025反而更好", "风险纪律成立，统计优势不成立", "冷却3-5日+半仓；不要固定跳过几笔"],
    ["不同股票具体分析", "高波活跃趋势两段最差；高波低量回踩显著为负；低波类型略好但不稳定", "对，但不是四套魔法", "股性决定允不允许做、做多大、止损多宽"],
], columns=["秘诀原话", "数据检验", "判决", "对系统的含义"])
claims.to_excel(writer, sheet_name="01_秘诀检验", index=False)

# 02 443笔概览
top5 = t.pnl.nlargest(5).sum()
ov = pd.DataFrame([
    ["交易笔数", len(t), "笔"],
    ["累计收益", 48.75, "%"],
    ["年化收益", 3.75, "%"],
    ["最大回撤", -43.60, "%"],
    ["胜率", round((t.ret > 0).mean() * 100, 2), "%"],
    ["平均单笔", round(t.ret.mean() * 100, 2), "%"],
    ["中位单笔", round(t.ret.median() * 100, 2), "%"],
    ["平均盈利", round(t.loc[t.ret > 0, "ret"].mean() * 100, 2), "%"],
    ["平均亏损", round(t.loc[t.ret < 0, "ret"].mean() * 100, 2), "%"],
    ["盈亏比", round(t.loc[t.ret > 0, "ret"].sum() / abs(t.loc[t.ret < 0, "ret"].sum()), 2), "倍"],
    ["前5笔盈利合计", round(top5, 0), "元"],
    ["前5笔占净利润", round(top5 / t.pnl.sum() * 100, 1), "%"],
    ["其余438笔合计", round(t.pnl.sum() - top5, 0), "元"],
    ["止损笔数", int((t.reason == "止损").sum()), "笔"],
    ["止损平均收益", round(t.loc[t.reason == "止损", "ret"].mean() * 100, 2), "%"],
], columns=["指标", "数值", "单位"])
ov.to_excel(writer, sheet_name="02_443笔概览", index=False)

# 03 方法组合
def stat(g):
    return pd.Series({
        "样本数": len(g), "胜率": (g.ret > 0).mean(), "平均收益": g.ret.mean(),
        "中位收益": g.ret.median(),
        "盈亏比": g.loc[g.ret > 0, "ret"].sum() / abs(g.loc[g.ret < 0, "ret"].sum()),
        "总盈亏": g.pnl.sum(),
    })
combo = t.groupby("combo").apply(stat).reset_index()
combo["方法组合"] = combo.combo.map({"011": "魏氏+突破", "111": "金叉+魏氏+突破", "101": "金叉+突破", "110": "金叉+魏氏（无突破）"})
combo = combo[["方法组合", "样本数", "胜率", "平均收益", "中位收益", "盈亏比", "总盈亏"]]
combo = pct(combo, ["胜率", "平均收益", "中位收益"])
combo.to_excel(writer, sheet_name="03_方法组合", index=False)

# 04 特征点汇总
s = pd.read_csv(os.path.join(BASE, "特征点_汇总.csv"))
s = s[s.group.isin(["全部", "低波低量型", "低波活跃型", "高波低量型", "高波活跃型", "安静过滤", "未过滤", "MA20上/MA60上", "MA20上/MA60下", "MA20下"])].copy()
s = s.rename(columns={
    "setup": "形态", "period": "时期", "group": "分组", "n": "样本数",
    "胜率20": "20日胜率", "20日均值": "20日均收益", "20日中位": "20日中位收益",
    "20日超额": "20日超额", "20日超额中位": "20日超额中位",
    "MAE20": "20日平均最大浮亏", "MFE20": "20日平均最大浮盈",
    "止损率20": "20日止损率", "trail均值": "趋势离场均值", "trail中位": "趋势离场中位",
    "trail胜率": "趋势离场胜率", "平均R": "平均风险倍数", "平均持有": "平均持有天数",
})
s["形态"] = s["形态"].map({"TB": "趋势成立点", "PB": "调整结束点", "UT": "普通上升日"})
s = pct(s, ["20日胜率", "20日均收益", "20日中位收益", "20日超额", "20日超额中位",
             "20日平均最大浮亏", "20日平均最大浮盈", "20日止损率",
             "趋势离场均值", "趋势离场中位", "趋势离场胜率"])
s.to_excel(writer, sheet_name="04_特征点汇总", index=False)

# 05 股性矩阵
p = pd.read_csv(os.path.join(BASE, "特征点_事件明细.csv"), parse_dates=["date"])
p = p[p.type.notna()]
mat = p.pivot_table(index=["period", "type"], columns="setup", values=["ex20", "ret20"], aggfunc="mean").reset_index()
mat.columns = ["时期", "股性", "调整结束点_20日超额", "趋势成立点_20日超额", "普通日_20日超额",
               "调整结束点_20日均收益", "趋势成立点_20日均收益", "普通日_20日均收益"]
mat["趋势成立点减普通日"] = mat["趋势成立点_20日超额"] - mat["普通日_20日超额"]
mat["调整结束点减普通日"] = mat["调整结束点_20日超额"] - mat["普通日_20日超额"]
mat = mat[["时期", "股性", "趋势成立点_20日超额", "调整结束点_20日超额", "普通日_20日超额", "趋势成立点减普通日", "调整结束点减普通日"]]
mat = pct(mat, ["趋势成立点_20日超额", "调整结束点_20日超额", "普通日_20日超额", "趋势成立点减普通日", "调整结束点减普通日"])
mat.to_excel(writer, sheet_name="05_股性矩阵", index=False)

# 06 出场对比
e = pd.read_csv(os.path.join(BASE, "出场方案对比.csv"))
e["形态"] = e.setup.map({"TB": "趋势成立点", "PB": "调整结束点", "UT": "普通上升日"})
e = e.rename(columns={"setup": "形态代码", "period": "时期", "n": "样本数"})
e = e[["形态", "时期", "样本数", "止损+破MA20_均值", "固定2R_均值", "固定3R_均值",
       "1/3在2R+余仓trail_均值", "1/2在2R+余仓trail_均值", "1R后保本+trail_均值", "trail平均R"]]
e.columns = ["形态", "时期", "样本数", "止损+破20日均线", "固定2倍风险", "固定3倍风险", "三分之一在2倍风险+余仓", "二分之一在2倍风险+余仓", "1倍风险后保本+余仓", "平均风险倍数"]
e = pct(e, ["止损+破20日均线", "固定2倍风险", "固定3倍风险", "三分之一在2倍风险+余仓", "二分之一在2倍风险+余仓", "1倍风险后保本+余仓"])
e.to_excel(writer, sheet_name="06_出场对比", index=False)

# 07 仓位对照
pos = pd.DataFrame([
    ["固定20%", 1.632, -0.389, 0.200, 0.0153, 0.0640, 4.19],
    ["0.50%", 1.177, -0.149, 0.067, 0.0050, 0.0200, 7.91],
    ["0.75%", 1.266, -0.215, 0.100, 0.0075, 0.0300, 5.89],
    ["1.00%", 1.355, -0.277, 0.132, 0.0100, 0.0400, 4.89],
    ["1.50%", 1.588, -0.371, 0.190, 0.0145, 0.0600, 4.28],
    ["2.00%", 1.632, -0.389, 0.200, 0.0153, 0.0640, 4.19],
    ["3.00%", 1.632, -0.389, 0.200, 0.0153, 0.0640, 4.19],
], columns=["单笔风险", "期末权益", "最大回撤", "平均仓位", "平均单笔风险", "最大同时风险", "终值除回撤"])
pos = pct(pos, ["最大回撤", "平均仓位", "平均单笔风险", "最大同时风险"])
pos.to_excel(writer, sheet_name="07_仓位对照", index=False)

# 08 止损后反弹
reb = pd.read_csv(os.path.join(BASE, "止损后反弹.csv"))
reb_sum = pd.DataFrame([
    ["止损笔数", len(reb)],
    ["10日内最高反弹均值", reb.max10.mean()],
    ["10日内最高反弹中位", reb.max10.median()],
    ["20日内最高反弹均值", reb.max20.mean()],
    ["20日内最高反弹中位", reb.max20.median()],
    ["20日反弹超过5%占比", (reb.max20 > 0.05).mean()],
    ["20日收盘回到买价上方占比", reb.above_buy20.mean()],
    ["止损后继续持有20日均值", reb.close20.mean()],
    ["止损后继续持有20日中位", reb.close20.median()],
    ["止损后继续持有20日为正占比", (reb.close20 > 0).mean()],
], columns=["指标", "数值"])
reb_sum.to_excel(writer, sheet_name="08_止损后反弹", index=False)

# 09 复盘模板
cols = ["交易编号", "信号日", "买入日", "卖出日", "股票代码", "股性", "市场状态", "形态",
        "入场价", "止损价", "止损距离", "仓位比例", "计划风险", "首周收益",
        "最大浮盈", "最大浮亏", "离场原因", "实际收益", "实际风险倍数",
        "是否违反规则", "违反条款", "复盘结论"]
pd.DataFrame(columns=cols).to_excel(writer, sheet_name="09_复盘模板", index=False)

writer.close()

# 格式化
wb = load_workbook(OUT)
head_fill = PatternFill("solid", fgColor="1F4E78")
head_font = Font(color="FFFFFF", bold=True)
for ws in wb.worksheets:
    ws.freeze_panes = "A2"
    for c in ws[1]:
        c.fill = head_fill; c.font = head_font; c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for col in range(1, ws.max_column + 1):
        mx = 0
        for row in range(1, min(ws.max_row, 200) + 1):
            v = ws.cell(row=row, column=col).value
            if v is not None:
                mx = max(mx, len(str(v)))
        ws.column_dimensions[get_column_letter(col)].width = min(max(10, mx * 1.2), 42)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
wb.save(OUT)
print("saved", OUT)
