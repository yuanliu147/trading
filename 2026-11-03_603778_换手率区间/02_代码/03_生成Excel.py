# -*- coding: utf-8 -*-
"""把分析结果汇总为中文 Excel。"""
import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

ROOT = "/Users/admin/WorkSpace/trading/2026-11-03_603778_换手率区间"
RES = f"{ROOT}/03_结果"
OUT = f"{ROOT}/603778_换手率区间.xlsx"

hist = pd.read_pickle(f"{ROOT}/01_数据/603778_hist.pkl")
full = pd.read_pickle(f"{ROOT}/01_数据/603778_full.pkl")

band = pd.read_csv(f"{RES}/换手区间统计.csv")
yearly = pd.read_csv(f"{RES}/年度换手.csv")
phase = pd.read_csv(f"{RES}/阶段换手.csv")
bp = pd.read_csv(f"{RES}/区间与阶段.csv")
events = pd.read_csv(f"{RES}/极端换手事件.csv")

# 当前同类样本：分全样本 / 剔除2026 / 2018-2024
conds = {
    "全部交易日": pd.Series(True, index=hist.index),
    "深度回撤>60% + 换手<5%": (hist.dd250 < -60) & (hist.turn < 5),
    "深度回撤>60% + 换手5-8%": (hist.dd250 < -60) & (hist.turn >= 5) & (hist.turn < 8),
    "250日低位<30 + 换手3.5-5%": (hist.pos250 < 30) & (hist.turn >= 3.5) & (hist.turn < 5),
    "250日低位<30 + 换手5-8%": (hist.pos250 < 30) & (hist.turn >= 5) & (hist.turn < 8),
    "换手5-8%（全部）": (hist.turn >= 5) & (hist.turn < 8),
    "换手3.5-5%（全部）": (hist.turn >= 3.5) & (hist.turn < 5),
}
rows = []
for name, m in conds.items():
    for sample, sub in [("全样本", hist[m]),
                        ("剔除2026", hist[m & (hist.year < 2026)]),
                        ("2018-2024", hist[m & (hist.year >= 2018) & (hist.year <= 2024)])]:
        rows.append({
            "条件": name, "样本": sample, "天数": len(sub),
            "T+5中位(%)": round(sub.f5.median(), 1) if len(sub) else np.nan,
            "T+20中位(%)": round(sub.f20.median(), 1) if len(sub) else np.nan,
            "T+20胜率(%)": round((sub.f20 > 0).mean() * 100) if len(sub) else np.nan,
            "T+60中位(%)": round(sub.f60.median(), 1) if len(sub) else np.nan,
            "T+60胜率(%)": round((sub.f60 > 0).mean() * 100) if len(sub) else np.nan,
            "主要年份": "、".join(str(y) for y in sub.year.value_counts().head(3).index) if len(sub) else "",
        })
analogs = pd.DataFrame(rows)

# 区间 × 趋势
trend_rows = []
for trend, sub in [("收盘>均线60", hist[hist.close_h > hist.ma60]),
                   ("收盘<均线60", hist[hist.close_h <= hist.ma60])]:
    for b, s in sub.groupby("band", observed=True):
        trend_rows.append({"趋势": trend, "换手区间(%)": str(b), "天数": len(s),
                           "T+20中位(%)": round(s.f20.median(), 1),
                           "T+20胜率(%)": round((s.f20 > 0).mean() * 100),
                           "T+60中位(%)": round(s.f60.median(), 1)})
trend_tab = pd.DataFrame(trend_rows)

# 当前快照
last = full.iloc[-1]
factor = last.close_h / last.close
snap = pd.DataFrame([
    ["最新交易日", "2026-11-02"],
    ["收盘价（元）", f"{last.close:.2f}"],
    ["当日换手率", f"{last.turn:.1f}%"],
    ["成交额", f"{last.amount/1e8:.2f} 亿元"],
    ["流通市值", f"{last.close*6.428562:.1f} 亿元（流通股本 6.4286 亿股）"],
    ["全历史换手分位", "73.5%（历史中位 2.5%）"],
    ["2026 年内换手分位", "18.9%（本轮泡沫中的缩量）"],
    ["近 250 日换手分位", "15.6%"],
    ["250 日位置", f"{last.pos250:.0f}%（区间 4.11–34.27 元）"],
    ["距 2026-04-13 高点", f"{(last.close/33.10-1)*100:.1f}%"],
    ["均线20（不复权）", f"{last.ma20/factor:.2f} 元，收盘低 {abs(last.close_h/last.ma20-1)*100:.1f}%"],
    ["均线60（不复权）", f"{last.ma60/factor:.2f} 元，收盘低 {abs(last.close_h/last.ma60-1)*100:.1f}%"],
    ["估值", "PB 20.0 / PE(TTM) -8.3（2026-10-29）"],
    ["近期低点", "10/23 6.72；11/02 盘中 6.48 后收回"],
], columns=["项目", "数值"])

concl = pd.DataFrame([
    ["1", "换手率不是单边指标", "同样 20%+ 的换手，2016/2017 多为派发，2025-26 主升段却是接力；阶段和价格位置比绝对数值更重要。"],
    ["2", "全历史最差区间", "2-8% 是这只股票的“慢跌区”：T+20 中位 -2.4%~-3.7%，胜率 34%-42%；5-8% 在 2018-24 各阶段均为负，T+60 中位 -12%~-17%。"],
    ["3", "唯一正期望区", "<1% 地量：T+20 中位 +1.6%~+2.9%，胜率 54%-60%；主要出现在 2018-2020 与 2021-22 的长期底部，但地量可以持续数月甚至两年。"],
    ["4", "高位高换手 ≠ 见顶", "250 日位置>70% 且换手≥12%：全历史 T+20 +6.3%/52%，但 2025 年前为 -16.4%/24%，2025 主升段为 +36%/73%；低位高换手才是失败反弹（位置<30%：-5.5%/24%）。"],
    ["5", "当前 6.4% 的定位", "全历史 73.5 分位，但 2026 年 18.9 分位：是“泡沫期缩量”，不是“历史地量”。历史上 5-8% 带（剔除 2026）T+20 -2.4%、T+60 -12.5%、60 日胜率 29%。"],
    ["6", "当前不是估值底", "PB 20.0，远高于 2018-10 的 2.15、2024-02 的 2.00、2025-10 的 3.70；股价 6.92 仍比 2025-10 低点 4.11 高 68%。"],
    ["7", "需要等的信号", "要么换手降到 1-2% 并横盘数周（历史地量底），要么放量站上均线20（约 7.56 元）；当前 6.4% 处于两者之间，属于弱反弹/下跌中继区。"],
], columns=["序号", "结论", "说明"])

sheets = {
    "结论": concl,
    "当前快照": snap,
    "换手区间统计": band,
    "当前同类样本": analogs,
    "区间与趋势": trend_tab,
    "阶段换手": phase,
    "年度换手": yearly,
    "区间与阶段": bp,
    "极端换手事件": events,
}

with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
    # 说明页
    readme = pd.DataFrame([
        ["研究对象", "603778 国晟科技（原乾景园林），上交所主板"],
        ["数据源", "baostock 本地 SQLite（2015-12-31 至 2026-10-29）"],
        ["最新补充", "同花顺本地 DuckDB（2026-10-30、2026-11-02）；换手率按流通股本 6.428562 亿股估算"],
        ["换手率口径", "成交量 / 流通股本；baostock 官方 turn 字段；停牌日剔除"],
        ["收益口径", "后复权价（adjustflag=1），T+1 开盘买入，持有 N 个交易日到收盘；未扣交易成本"],
        ["区间划分", "<0.5%、0.5-1%、1-2%、2-3.5%、3.5-5%、5-8%、8-12%、12-20%、>20%"],
        ["重要提醒", "① 单只股票，样本高度重叠，n 天 ≠ n 次独立交易；② 2016 新股、2025-26 泡沫是极端阶段，会显著拉偏全样本；③ 历史统计不是收益承诺。"],
    ], columns=["项目", "说明"])
    readme.to_excel(writer, sheet_name="说明", index=False, startrow=2)
    ws = writer.sheets["说明"]
    ws["A1"] = "说明"
    ws["A1"].font = Font(name="PingFang SC", size=15, bold=True, color="8C2F24")
    ws["A2"] = "603778 换手率区间研究 · 数据与口径"
    ws["A2"].font = Font(name="PingFang SC", size=9, color="756D60")
    for name, df in sheets.items():
        df.to_excel(writer, sheet_name=name, index=False, startrow=2)
        ws = writer.sheets[name]
        ws["A1"] = name
        ws["A1"].font = Font(name="PingFang SC", size=15, bold=True, color="8C2F24")
        ws["A2"] = "603778 换手率区间研究 · 数据截至 2026-11-02"
        ws["A2"].font = Font(name="PingFang SC", size=9, color="756D60")

    # 日度明细放最后，避免大表拖慢其他页
    cols = ["date", "close", "turn", "pct_chg", "amount", "ret20", "ma20", "ma60",
            "pos250", "dd250", "f5", "f20", "f60", "mfe20", "mae20", "band", "phase"]
    daily = full[cols].copy()
    daily.columns = ["日期", "收盘", "换手率%", "当日涨跌%", "成交额", "前20日涨跌%",
                     "均线20", "均线60", "250日位置%", "距250日高%", "T+5%", "T+20%",
                     "T+60%", "20日最大浮盈%", "20日最大浮亏%", "换手区间", "阶段"]
    daily["日期"] = pd.to_datetime(daily["日期"]).dt.date
    daily.to_excel(writer, sheet_name="日度明细", index=False, startrow=2)
    ws = writer.sheets["日度明细"]
    ws["A1"] = "日度明细"
    ws["A1"].font = Font(name="PingFang SC", size=15, bold=True, color="8C2F24")
    ws["A2"] = "按日期升序；T+N 为 T+1 开盘买入后的后复权收益；空白表示未来数据不足"
    ws["A2"].font = Font(name="PingFang SC", size=9, color="756D60")

wb = load_workbook(OUT)
HEADER_FILL = PatternFill("solid", fgColor="8C2F24")
HEADER_FONT = Font(name="PingFang SC", size=10, bold=True, color="FFFFFF")
BODY_FONT = Font(name="PingFang SC", size=10)
THIN = Side(style="thin", color="DED5C6")
BORDER = Border(bottom=THIN)

for ws in wb.worksheets:
    # 表头行是第 3 行（startrow=2）
    hdr = 3
    for cell in ws[hdr]:
        if cell.value is not None:
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=hdr + 1, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            cell.font = BODY_FONT
            cell.border = BORDER
            cell.alignment = Alignment(vertical="center", wrap_text=False)
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        maxlen = 0
        for cell in col:
            v = cell.value
            if v is None:
                continue
            s = str(v)
            # 中文按 2 个字符宽估算
            width = sum(2 if ord(ch) > 127 else 1 for ch in s)
            maxlen = max(maxlen, width)
        ws.column_dimensions[letter].width = min(max(10, maxlen * 0.95 + 2), 46)
    ws.freeze_panes = f"A{hdr + 1}"
    if ws.max_row > hdr:
        ws.auto_filter.ref = f"A{hdr}:{get_column_letter(ws.max_column)}{ws.max_row}"
    ws.row_dimensions[hdr].height = 34
    ws.sheet_view.showGridLines = False

# 收益列色阶
for name in ["换手区间统计", "当前同类样本", "区间与趋势", "阶段换手"]:
    ws = wb[name]
    headers = [c.value for c in ws[3]]
    for j, h in enumerate(headers, start=1):
        if h and ("T+20中位" in str(h) or "T+60中位" in str(h) or "T+20均值" in str(h)):
            col = get_column_letter(j)
            ws.conditional_formatting.add(
                f"{col}4:{col}{ws.max_row}",
                ColorScaleRule(start_type="num", start_value=-25, start_color="2E6B55",
                               mid_type="num", mid_value=0, mid_color="FFFDF8",
                               end_type="num", end_value=25, end_color="B23A2B"))
    for j, h in enumerate(headers, start=1):
        if h and "T+20胜率" in str(h):
            col = get_column_letter(j)
            ws.conditional_formatting.add(
                f"{col}4:{col}{ws.max_row}",
                ColorScaleRule(start_type="num", start_value=20, start_color="2E6B55",
                               mid_type="num", mid_value=50, mid_color="FFFDF8",
                               end_type="num", end_value=80, end_color="B23A2B"))

# 结论 / 说明页格式
for name, height in [("结论", 48), ("说明", 34)]:
    ws = wb[name]
    for r in range(4, ws.max_row + 1):
        ws.row_dimensions[r].height = height
        for c in range(1, ws.max_column + 1):
            ws.cell(row=r, column=c).alignment = Alignment(
                horizontal="center" if c == 1 else "left",
                vertical="top", wrap_text=True)

wb.save(OUT)
print("Excel 已生成:", OUT, os.path.getsize(OUT) / 1024, "KB")
