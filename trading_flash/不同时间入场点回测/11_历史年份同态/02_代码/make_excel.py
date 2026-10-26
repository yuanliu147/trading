# -*- coding: utf-8 -*-
"""历史年份同态：中文 Excel 数据表。"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
RES = OUT / '03_结果'
XLSX = OUT / '历史年份同态.xlsx'

NAVY = '1B2A3A'
LIGHT = 'EDF1F4'
RED = 'F6DDD8'
TEAL = 'D9EAF0'
GREY = '8A969F'
WHITE = 'FFFFFF'

thin = Side(style='thin', color='D5DCE1')
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
HEADER_FILL = PatternFill('solid', fgColor=NAVY)
HEADER_FONT = Font(name='PingFang SC', color=WHITE, bold=True, size=10)
BODY_FONT = Font(name='PingFang SC', color='1B2A3A', size=10)
TITLE_FONT = Font(name='PingFang SC', color=NAVY, bold=True, size=16)
SUB_FONT = Font(name='PingFang SC', color=GREY, size=9)


def style_sheet(ws, df, pct_cols=(), width_map=None, row_height=20, highlight_years=()):
    ws.append(list(df.columns))
    for c in range(1, len(df.columns) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = BORDER
    ws.row_dimensions[1].height = 32
    for _, r in df.iterrows():
        ws.append(list(r.values))
        ridx = ws.max_row
        is_hl = ('年份' in df.columns and r.get('年份') in highlight_years)
        for c, col in enumerate(df.columns, start=1):
            cell = ws.cell(row=ridx, column=c)
            cell.font = BODY_FONT
            cell.border = BORDER
            cell.alignment = Alignment(horizontal='center' if c > 1 else 'left',
                                       vertical='center', wrap_text=False)
            if col in pct_cols:
                cell.number_format = '0.0%'
            elif col in ('年份', '单笔数', '起点数', '天数', '核心8_top100天数'):
                cell.number_format = '0'
            elif isinstance(cell.value, float):
                cell.number_format = '0.000'
        if is_hl:
            fill = PatternFill('solid', fgColor=RED if r.get('年份') == 2018 else TEAL)
            for c in range(1, len(df.columns) + 1):
                ws.cell(row=ridx, column=c).fill = fill
        ws.row_dimensions[ridx].height = row_height
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions
    for c in range(1, len(df.columns) + 1):
        letter = get_column_letter(c)
        width = (width_map or {}).get(df.columns[c - 1], 13)
        ws.column_dimensions[letter].width = width
    # 表头行高、自动换行


wb = Workbook()

# ------------------------------------------------------------------ 说明
ws = wb.active
ws.title = '说明'
ws.sheet_view.showGridLines = False
ws['A1'] = '历史年份同态：现在像哪一年？'
ws['A1'].font = TITLE_FONT
ws['A2'] = '数据截至 2026-10-23 收盘 · 沪深主板等权 · 策略样本 2017-11 至 2026-10'
ws['A2'].font = SUB_FONT

rows = [
    ('', ''),
    ('结论先行', ''),
    ('1', '按全状态 13 特征，2018 最像（年度距离 0.299）；按核心 8 特征，2024 最像（0.158）。两者只差 0.014（核心）/0.023（全状态），属于“双胞胎”，不是唯一答案。'),
    ('2', '第三名（2019/2022）距离约 0.40~0.44，是前两名的 2.4~2.8 倍；再往后没有参考价值。'),
    ('3', '两个近邻之后 120 天，市场一个 -19.5%（2018-05-30）、一个 +29.7%（2024-06-14）。所以“像哪一年”不能回答方向。'),
    ('4', '但对策略，两个同态区间内下一可执行起点的 60/120 日都是正的：2018-06-14 +9.8%/+16.0%，2024-06-14 +2.5%/+11.4%；120 日最大回撤都约 -12.5%。'),
    ('5', '2018/2024 本身也是策略丰年：+47.4% / +93.0%。策略赚的是波动中的 15 日反弹，不是指数年度方向。'),
    ('', ''),
    ('当前状态', '2026-10-23：D 熊市下跌；250 日涨幅 -5.2%，250 日回撤 -17.5%，20 日波动率 0.198，宽度20 0.190，情绪 -0.64。'),
    ('更像哪种跌法', '跌停占比 0.25%，远低于 2018-05-30 的 3.4%；宽度/波动/情绪更接近 2024-06。当前是“慢阴跌”，不是 2018 式急崩。'),
    ('', ''),
    ('口径', '“当前”= 本地市场日频最后一根（2026-10-23）；相似度 = 特征 1%/99% 缩尾、中位数/IQR 标准化后的欧氏距离；年度取该年最近 10 个交易日均值。全状态的跌停占比先做 log1p 压缩，避免 2018-05-30 的 3.4% 单日主导距离。'),
    ('口径', '市场前瞻 = 沪深主板等权指数未来 20/60/120 个交易日收益；策略起点 = 2 仓 50%、T+3 开盘、15 日、+20% 止盈的账户路径，来自 07 的 251 个信号日起点。'),
    ('限制', '市场日频从 2016-10-13 开始，2011/2015 不在样本内；相似度是描述性的，不是交易信号；同态日高度重叠，不能当作独立样本；本地数据无法识别政策/流动性等外生变量。'),
]
for i, (k, v) in enumerate(rows, start=4):
    ws.cell(row=i, column=1, value=k).font = Font(name='PingFang SC', bold=True, color=NAVY, size=10)
    ws.cell(row=i, column=2, value=v).font = BODY_FONT
    ws.cell(row=i, column=2).alignment = Alignment(wrap_text=True, vertical='top')
    ws.row_dimensions[i].height = 38 if v else 8
ws.column_dimensions['A'].width = 16
ws.column_dimensions['B'].width = 118
ws.freeze_panes = 'A4'

# ------------------------------------------------------------------ 当前状态
df = pd.read_csv(RES / '当前状态.csv')
df = df.rename(columns={'历史百分位': '历史百分位', '信号日百分位': '信号日百分位',
                        '历史中位': '历史中位', '历史P10': '历史P10', '历史P90': '历史P90'})
ws = wb.create_sheet('当前状态')
style_sheet(ws, df, pct_cols=('历史百分位', '信号日百分位', '历史中位', '历史P10', '历史P90'),
            width_map={'指标': 18, '当前值': 13, '单位': 9, '说明': 42})
# 当前值/历史分布按单位格式化（比/分不能当百分比）
for r in range(2, ws.max_row + 1):
    unit = ws.cell(row=r, column=3).value
    for c in (2, 4, 5, 6, 7, 8):
        cell = ws.cell(row=r, column=c)
        if cell.value is None or cell.value == '':
            continue
        if unit in ('%', '比例'):
            cell.number_format = '0.00%'
        else:
            cell.number_format = '0.000'

# ------------------------------------------------------------------ 年度相似度
df = pd.read_csv(RES / '年度相似度.csv')
df = df.rename(columns={
    '全状态13_距离': '全状态13距离', '全状态13_排名': '全状态13排名',
    '核心8_距离': '核心8距离', '核心8_排名': '核心8排名',
    '状态5_距离': '状态5距离', '状态5_排名': '状态5排名',
    '路径6_距离': '路径6距离', '路径6_排名': '路径6排名',
    '综合平均排名': '综合平均排名', '核心8最近日期': '核心8最近日期',
    '核心8最近日未来60日': '核心8最近日未来60日', '核心8最近日未来120日': '核心8最近日未来120日',
    '核心8_top100天数': '核心8前100天数', '策略年度收益': '策略年度收益',
    '市场年度收益': '市场年度收益', '单笔数': '当年单笔数',
    '单笔均值': '当年单笔均值', '胜率': '当年胜率',
})
ws = wb.create_sheet('年度相似度')
style_sheet(ws, df, pct_cols=('核心8最近日未来60日', '核心8最近日未来120日', '策略年度收益',
                              '市场年度收益', '当年单笔均值', '当年胜率'),
            width_map={'年份': 8, '核心8最近日期': 14, '综合平均排名': 11, '当年单笔数': 10},
            highlight_years=(2018, 2024))

# ------------------------------------------------------------------ 同态事件
df = pd.read_csv(RES / '同态事件_总表.csv')
df = df.rename(columns={
    '最小距离': '最小距离', '主同态': '主同态', '阶段': '阶段',
    '市场20日': '市场20日', '市场60日': '市场60日', '市场120日': '市场120日',
    '市场60日最大回撤': '市场60日最大回撤', '市场60日最大反弹': '市场60日最大反弹',
    '后续最低点日期': '后续最低点日期', '后续最低点回撤': '后续最低点回撤', '后续最低点交易日': '后续最低点交易日',
    '宽度回50%交易日': '宽度回50%交易日', '收复年线交易日': '收复年线交易日',
    '下一可执行起点': '策略下一开始点', '策略下一开始点60日': '策略下一开始点60日',
    '策略下一开始点120日': '策略下一开始点120日', '起点数': '区间起点数',
    '策略区间起点60日中位': '区间起点60日中位', '策略区间起点120日中位': '区间起点120日中位',
    '策略区间起点120日回撤中位': '区间起点120日回撤中位',
})
ws = wb.create_sheet('同态事件')
style_sheet(ws, df, pct_cols=('市场20日', '市场60日', '市场120日', '市场60日最大回撤',
                              '市场60日最大反弹', '后续最低点回撤', '策略下一开始点60日', '策略下一开始点120日',
                              '区间起点60日中位', '区间起点120日中位', '区间起点120日回撤中位'),
            width_map={'年份': 8, '代表日': 12, '主同态': 8, '起点': 12, '终点': 12,
                       '阶段': 22, '策略下一开始点': 13},
            highlight_years=(2018, 2024))

# ------------------------------------------------------------------ 策略起点
df = pd.read_csv(RES / '同态期_逐起点.csv')
df = df.rename(columns={'代表日': '代表日', '最小距离': '最小距离', '主同态': '主同态',
                        'signal_date': '信号日', 'ret_20': '20日', 'ret_60': '60日',
                        'ret_120': '120日', 'dd_60': '60日最大回撤', 'dd_120': '120日最大回撤'})
ws = wb.create_sheet('策略起点')
style_sheet(ws, df, pct_cols=('20日', '60日', '120日', '60日最大回撤', '120日最大回撤'),
            width_map={'代表日': 12, '信号日': 12, '最小距离': 11},
            highlight_years=())

# ------------------------------------------------------------------ 规则状态
df = pd.read_csv(RES / '规则状态_事件.csv')
df = df.rename(columns={'起点': '起点', '终点': '终点', '天数': '天数',
                        '250日涨幅': '250日涨幅', '250日回撤': '250日回撤',
                        '20日波动率': '20日波动率', '宽度20': '宽度20',
                        '未来20日': '市场20日', '未来60日': '市场60日', '未来120日': '市场120日',
                        '60日内最大回撤': '市场60日最大回撤', '宽度回50%天数': '宽度回50%交易日',
                        '收复年线天数': '收复年线交易日', '下一可执行起点': '策略下一开始点',
                        '策略起点60日': '策略60日', '策略起点120日': '策略120日'})
ws = wb.create_sheet('规则状态')
style_sheet(ws, df, pct_cols=('250日涨幅', '250日回撤', '20日波动率', '宽度20',
                              '市场20日', '市场60日', '市场120日', '市场60日最大回撤',
                              '策略60日', '策略120日'),
            width_map={'起点': 12, '终点': 12, '策略下一开始点': 13})

# ------------------------------------------------------------------ 距离预测力
df = pd.read_csv(RES / '距离预测力.csv')
df = df.rename(columns={'口径': '口径', 'Pearson': 'Pearson', 'Spearman': 'Spearman',
                        '最近10%均值': '最近10%均值', '最近10%中位': '最近10%中位',
                        '最近10%胜率': '最近10%胜率', '其余均值': '其余均值',
                        '其余中位': '其余中位', '其余胜率': '其余胜率',
                        '最近10%样本': '最近10%样本'})
ws = wb.create_sheet('距离预测力')
style_sheet(ws, df, pct_cols=('最近10%均值', '最近10%中位', '最近10%胜率',
                              '其余均值', '其余中位', '其余胜率'),
            width_map={'口径': 12, '最近10%样本': 12})

# ------------------------------------------------------------------ 最近邻
df = pd.read_csv(RES / '最近邻_核心8.csv').head(20)
df = df.rename(columns={
    'date': '日期', 'd_全状态13': '全状态13距离', 'd_核心8': '核心8距离',
    'd_状态5': '状态5距离', 'd_路径6': '路径6距离',
    'equal_weight_index': '等权指数', 'ret250': '250日涨幅', 'dd250': '250日回撤',
    'ret120': '120日涨幅', 'ret60': '60日涨幅', 'ret20': '20日涨幅',
    'market_volatility20': '20日波动率', 'breadth20': '宽度20', 'breadth60': '宽度60',
    'sentiment_score': '情绪分', 'limit_down10_share': '跌停占比',
    'amount_ratio20_60': '20/60成交额比', 'f20': '未来20日', 'f60': '未来60日',
    'f120': '未来120日', 'min60': '60日最大回撤', 'max60': '60日最大反弹',
    'd_breadth50': '宽度回50%交易日', 'd_ma250': '收复年线交易日', '阶段': '阶段',
})
ws = wb.create_sheet('最近邻')
style_sheet(ws, df, pct_cols=('250日涨幅', '250日回撤', '120日涨幅', '60日涨幅', '20日涨幅',
                              '20日波动率', '宽度20', '宽度60', '跌停占比',
                              '未来20日', '未来60日', '未来120日', '60日最大回撤',
                              '60日最大反弹'),
            width_map={'日期': 12, '阶段': 22})

wb.save(XLSX)
print('saved', XLSX)
