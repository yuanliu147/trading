# -*- coding: utf-8 -*-
"""603778 K 线与特征：中文 Excel。"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
RES = OUT / '03_结果'
XLSX = OUT / '603778_K线与特征.xlsx'

NAVY = '1B2A3A'
WHITE = 'FFFFFF'
LIGHT = 'EDF1F4'
RED = 'F6DDD8'
TEAL = 'D9EAF0'
GREY = '8A969F'

thin = Side(style='thin', color='D5DCE1')
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
HEADER_FILL = PatternFill('solid', fgColor=NAVY)
HEADER_FONT = Font(name='PingFang SC', color=WHITE, bold=True, size=10)
BODY_FONT = Font(name='PingFang SC', color=NAVY, size=10)
TITLE_FONT = Font(name='PingFang SC', color=NAVY, bold=True, size=16)
SUB_FONT = Font(name='PingFang SC', color=GREY, size=9)


def style(ws, df, pct_cols=(), width_map=None, highlight_dates=(), date_col=None):
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
        hl = date_col and r.get(date_col) in highlight_dates
        for c, col in enumerate(df.columns, start=1):
            cell = ws.cell(row=ridx, column=c)
            cell.font = BODY_FONT
            cell.border = BORDER
            cell.alignment = Alignment(horizontal='center' if c > 1 else 'left', vertical='center')
            if col in pct_cols:
                cell.number_format = '0.0%'
            elif isinstance(cell.value, float):
                cell.number_format = '0.000'
        if hl:
            fill = PatternFill('solid', fgColor=RED if '10-23' in str(r.get(date_col)) else TEAL)
            for c in range(1, len(df.columns) + 1):
                ws.cell(row=ridx, column=c).fill = fill
        ws.row_dimensions[ridx].height = 20
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions
    for c, col in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(c)].width = (width_map or {}).get(col, 13)


wb = Workbook()
ws = wb.active
ws.title = '说明'
ws.sheet_view.showGridLines = False
ws['A1'] = '603778.SH：K 线与特征'
ws['A1'].font = TITLE_FONT
ws['A2'] = '2026-10-23 收盘；10/26 仅收盘/部分特征；后复权折算当前价等价'
ws['A2'].font = SUB_FONT
rows = [
    ('', ''),
    ('结论 1', '14 倍泡沫（2.49→34.27）后 -78.6%；250 日涨幅仍 +110.6%，过滤器已旧。'),
    ('结论 2', '10/23 是 BOR + 高收盘 0.924；深回撤+BOR+高收盘 17 笔全为正，均值 +14.5%、中位 +19.6%、最小 +1.4%。'),
    ('结论 3', '10/26 不是 BOR，收盘位置 0.4545，弱跟随；深回撤+低收盘 73 笔 +8.8%/74%，弱于 10/23。'),
    ('结论 4', '结构仍空：破 7 月低点 7.56，低于所有均线；7/21 以来 -14.5% vs 市场 +3.8%（相对 -17.6pp）。'),
    ('结论 5', '15 日战术 6.5/10，结构性底部 3/10；按 2 仓规则做，不因榜一加仓；跌破 6.72 则 BOR 失败。'),
    ('', ''),
    ('口径', 'BOR=今日低点破昨日低点、收盘破昨日收盘、收盘高于开盘；收益 T+3 开盘、+20% 止盈、15 日、0.175%/边。'),
    ('折算', '后复权价 × 0.3970 = 当前价等价；历史价格不是当时成交价。'),
    ('限制', '10/26 OHLC 未入库；BOR 17 笔样本内、事后定义；当前精确组合含 r250>100 历史 0 笔；不是投资建议。'),
]
for i, (k, v) in enumerate(rows, start=4):
    ws.cell(row=i, column=1, value=k).font = Font(name='PingFang SC', bold=True, color=NAVY, size=10)
    ws.cell(row=i, column=2, value=v).font = BODY_FONT
    ws.cell(row=i, column=2).alignment = Alignment(wrap_text=True, vertical='top')
    ws.row_dimensions[i].height = 42 if v else 8
ws.column_dimensions['A'].width = 14
ws.column_dimensions['B'].width = 110
ws.freeze_panes = 'A4'

S = pd.read_csv(RES / '信号日特征.csv')
S['信号日'] = pd.to_datetime(S['信号日']).dt.strftime('%m-%d')
S = S[['信号日', '排名', '案例', '15日收益', '250日回撤', '250日涨幅', 'RSI14', '收盘位置',
       'MACD柱', 'MACD柱变化', '量比1/20', '量比5/20', '距MA5', '距MA10', '距MA20',
       '距MA60', '距MA250', '下影', 'BOR', '市场20日', '市场宽度20']]
S = S.rename(columns={'距MA5': '距5日均线', '距MA10': '距10日均线', '距MA20': '距20日均线',
                      '距MA60': '距60日均线', '距MA250': '距250日均线', 'BOR': '外包反转'})
ws = wb.create_sheet('信号日特征')
style(ws, S, pct_cols=('15日收益', '250日回撤', '250日涨幅', '收盘位置', 'MACD柱', 'MACD柱变化',
                       '量比1/20', '量比5/20', '距5日均线', '距10日均线', '距20日均线', '距60日均线',
                       '距250日均线', '下影', '市场20日', '市场宽度20'),
      width_map={'信号日': 9, '排名': 6, '案例': 12}, highlight_dates=('10-23',), date_col='信号日')

C = pd.read_csv(RES / '条件统计.csv')
C = C.rename(columns={'条件': '条件', '样本': '样本', '均值': '均值', '中位': '中位', '胜率': '胜率',
                      '最差': '最差', '最好': '最好', 'p10': 'p10'})
ws = wb.create_sheet('条件统计')
style(ws, C, pct_cols=('均值', '中位', '胜率', '最差', '最好', 'p10'),
      width_map={'条件': 34, '样本': 8}, highlight_dates=('深回撤 + BOR + 高收盘',), date_col='条件')

K = pd.read_csv(RES / '关键价位.csv')
K = K.rename(columns={'类型': '类型', '日期': '日期', '价格': '价格', '距现价': '距现价'})
K['类型'] = K['类型'].replace({'MA5': '5日均线', 'MA10': '10日均线', 'MA20': '20日均线', 'MA60': '60日均线', 'MA250': '250日均线'})
ws = wb.create_sheet('关键价位')
style(ws, K, pct_cols=('距现价',), width_map={'类型': 24, '日期': 12, '价格': 10})

CUR = pd.read_csv(RES / '当前快照.csv')
CUR = CUR.rename(columns={'日期': '日期', '排名': '排名', '实际收盘': '实际收盘', '250日回撤': '250日回撤',
                          '250日涨幅': '250日涨幅', 'RSI14': 'RSI14', '收盘位置': '收盘位置',
                          'MACD柱': 'MACD柱', 'MACD柱变化': 'MACD柱变化', '量比1/20': '量比1/20',
                          '量比5/20': '量比5/20', '距MA5': '距5日均线', '距MA10': '距10日均线', '距MA20': '距20日均线',
                          '距MA60': '距60日均线', '距MA250': '距250日均线', '20日波动率': '20日波动率',
                          'ATR20': 'ATR20', '振幅': '振幅', '下影': '下影', '上影': '上影', '实体': '实体',
                          '市场20日': '市场20日', '市场宽度20': '市场宽度20'})
ws = wb.create_sheet('当前快照')
style(ws, CUR, pct_cols=('250日回撤', '250日涨幅', '收盘位置', 'MACD柱', 'MACD柱变化', '量比1/20',
                         '量比5/20', '距5日均线', '距10日均线', '距20日均线', '距60日均线', '距250日均线',
                         '20日波动率', 'ATR20', '振幅', '下影', '上影', '实体', '市场20日', '市场宽度20'),
      width_map={'日期': 12}, highlight_dates=('2026-10-23',), date_col='日期')

R = pd.read_csv(RES / '相对强弱.csv', parse_dates=['date'])
R = R.rename(columns={'date': '日期', 'stock': '603778', 'market': '沪深主板等权'})
R['日期'] = R['日期'].dt.strftime('%Y-%m-%d')
ws = wb.create_sheet('相对强弱')
style(ws, R, width_map={'日期': 12, '603778': 10, '沪深主板等权': 14})

wb.save(XLSX)
print('saved', XLSX)
