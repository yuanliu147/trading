# -*- coding: utf-8 -*-
"""信号日回撤深度与收益：中文 Excel。"""
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
RES = OUT / '03_结果'

BUCKET = pd.read_csv(RES / '深度分档.csv')
DS = pd.read_csv(RES / '深度分组对比.csv')
PAIR_SUM = pd.read_csv(RES / '同日排名对比.csv')
PAIR_YEAR = pd.read_csv(RES / '同日排名_年度.csv')
PAIR_DETAIL = pd.read_csv(RES / '同日排名配对.csv', parse_dates=['信号日'])
YEAR = pd.read_csv(RES / '年度深度对比.csv')
PHASE = pd.read_csv(RES / '阶段深度对比.csv')
REG = pd.read_csv(RES / '深度回归.csv')
M26 = pd.read_csv(RES / '2026深度分档.csv')
MECH = pd.read_csv(RES / '深度机制.csv')
CUR = pd.read_csv(RES / '当前信号_深度对照.csv', parse_dates=['信号日'])
REP = pd.read_csv(RES / '重复信号深度_汇总.csv')
REPC = pd.read_csv(RES / '重复信号深度_相关性.csv')
REP_DETAIL = pd.read_csv(RES / '重复信号深度_配对.csv', parse_dates=['首笔信号日', '第二笔信号日'])
EXT = pd.read_csv(RES / '深度极值_70以上.csv', parse_dates=['信号日'])
CASE = pd.read_csv(RES / '603778_深度与收益.csv', parse_dates=['信号日', '入场日', '退出日'])
TRADE = pd.read_csv(RES / '前二逐笔_深度.csv', parse_dates=['信号日'])
DAY = pd.read_csv(RES / '信号日深度与组合收益.csv', parse_dates=['信号日'])
DAYC = pd.read_csv(RES / '日级相关性.csv')
CROWD = pd.read_csv(RES / '信号拥挤度分组.csv')

REP_DETAIL['第二笔更深'] = REP_DETAIL['第二笔更深'].map({True: '是', False: '否'})

SUMMARY = pd.DataFrame([
    ('问题', '信号日当天，这些股票较 250 日最高价分别跌了多少？跌幅对后续盈利有没有影响？'),
    ('口径', '后复权收盘 / 250 日最高后复权价 - 1；按信号日收盘计算，不是今天。收益为 T+3 开盘买、+20% 止盈、15 日到期、0.175%/边。'),
    ('样本', '前二 450 笔完整（246 个完整信号日）；全部信号 4019 条（用于对照）。只有前二是策略实际买入候选。'),
    ('总答案', '有影响，而且方向稳定：跌得越深，后续 15 日反弹的平均收益、胜率、止盈率都更高。'),
    ('前二深vs浅', '≤-65% 共 109 笔：平均 +9.21%、胜率 77.1%、止盈率 43.1%；>-65% 共 341 笔：+3.99%、61.9%、23.5%；差 +5.21pp（秩和 p<0.001）。'),
    ('同日验证', '同一天只比较排名1和排名2：排名1平均 +6.29%，排名2 +4.25%，差 +2.04pp；配对 t 检验 p=0.026；排名1更好占 52.0% 的天数。'),
    ('连续斜率', '前二：回撤每深 10pp，单笔平均多 +3.95pp（t=5.04）；控制市场回撤后仍 +3.20pp（t=3.97）。全部信号：+2.79pp → +1.40pp。'),
    ('年度稳定', '在同时有深组和浅组的 8 个年份里，深组全部胜出：2017/2018/2020/2021/2022/2024/2025/2026。2019、2023 没有 ≤-65% 的前二样本。'),
    ('2026拆分', '6 月深组 +0.29% vs 浅组 -9.30%（深组只是打平，不是赚钱）；7 月深组 +12.15% vs 浅组 +6.11%。深度能缓冲，但不能替代市场转暖。'),
    ('重复信号', '同一股票后面一次信号如果更深，并不会更好：第二笔更深组，第二笔-第一笔 -0.28pp；第二笔更浅组 -0.09pp。深度是“同一天选哪只”的排序，不是“等它再跌深一点”的择时。'),
    ('当前信号', '603778 -78.61%、605376 -71.30%、002969 -70.68%、605117 -45.32%（后复权，信号日口径）。前三只在历史最好档；605117 最浅，且实际价 -61.07% 是 10 转 4 除权放大，不能直接比。'),
    ('当前动作', '603778 已持有（10/15 信号、10/20 入场），不重复买；新标的首选 605376，其次 002969。深度支持“可以买”，但当前市场 D 阶段、20 日波动 0.198、广度 0.19，不支持重仓追。'),
    ('最大局限', '前二本身就是按跌幅最深选出来的，深组不是随机实验；全部信号高度重叠；2026 深组收益主要集中在 7 月。结论是描述性统计，不是新的实盘规则。'),
], columns=['项目', '结论'])

PARAMS = pd.DataFrame([
    ('信号条件', '250 日回撤<-45%；250 日涨幅>0；RSI14<35；收盘位置>0.45；市场 20 日<-5%'),
    ('排序', '按个股 250 日回撤最深取前二'),
    ('仓位', '2 仓，每仓 50% 当前权益；无信号空仓'),
    ('入场', '信号后第 3 个交易日开盘（N=3）'),
    ('退出', '+20% 盘中止盈；15 个交易日收盘到期；无普通止损'),
    ('成本', '0.175%/边'),
    ('深度口径', '后复权收盘 / 过去 250 日最高后复权价 - 1；在信号日收盘计算'),
    ('收益口径', '逐只独立：T+3 开盘买、+20% 止盈、15 日到期；不模拟 2 仓占位'),
    ('复权口径', '深度和收益都用后复权；实际价只用于核对。送转会让实际价回撤失真，例如 605117 的 10 转 4'),
    ('数据截止', '2026-10-23'),
], columns=['参数', '设定'])

NOTES = pd.DataFrame([
    ('数据源', '同花顺本地 DuckDB 派生 research_panel.parquet；沪深主板等权 market_daily.parquet。'),
    ('前二样本', '459 行原始（251 个信号日），450 行完整（246 个完整信号日）；9 行未完整：5 行持有未满、3 行数据结束、1 行一字涨停无法买入。'),
    ('全部信号', '4195 条原始，4019 条完整，246 个信号日；用于对照，不用于实盘候选。'),
    ('为什么不是 251', '收益完整要求退出日已在数据内；2026-10-14 之后的部分信号尚未满 15 日，未纳入统计。'),
    ('深度分组', '深组=回撤≤-65%；浅组=回撤>-65%。另给出 -60% 分界做稳健性对照。'),
    ('重叠警告', '全部信号的 4019 条不是 4019 个独立机会：同一股票会连续多日触发，收益窗口高度重叠。前二 450 笔也有跨日重叠。'),
    ('2026警告', '2026 年深组的前二收益主要集中在 7 月（24 笔 +12.15%）；6 月深组 9 笔仅 +0.29%。不要把 7 月的反弹外推到所有深跌。'),
    ('实际价', '605117 后复权回撤 -45.32%，实际价回撤 -61.07%：10 转 4 除权导致，不是真实亏损幅度；信号只看后复权。'),
    ('幸存者偏差', '股票池为当前上市主板，未含退市股；熊市年份可能高估。'),
    ('历史不代表未来', '先纸面跟踪 50 笔；任何新过滤都要独立样本外验证。'),
], columns=['项目', '说明'])

path = OUT / '信号日回撤深度与收益.xlsx'
with pd.ExcelWriter(path, engine='openpyxl') as writer:
    SUMMARY.to_excel(writer, sheet_name='结论摘要', index=False)
    PARAMS.to_excel(writer, sheet_name='口径与参数', index=False)
    BUCKET.to_excel(writer, sheet_name='深度分档', index=False)
    DS.to_excel(writer, sheet_name='深浅对比', index=False)
    PAIR_SUM.to_excel(writer, sheet_name='同日排名汇总', index=False)
    PAIR_YEAR.to_excel(writer, sheet_name='同日排名年度', index=False)
    PAIR_DETAIL.to_excel(writer, sheet_name='同日排名明细', index=False)
    YEAR.to_excel(writer, sheet_name='年度对比', index=False)
    PHASE.to_excel(writer, sheet_name='阶段对比', index=False)
    REG.to_excel(writer, sheet_name='深度回归', index=False)
    M26.to_excel(writer, sheet_name='2026月度', index=False)
    MECH.to_excel(writer, sheet_name='深度机制', index=False)
    CUR.to_excel(writer, sheet_name='当前信号', index=False)
    REP.to_excel(writer, sheet_name='重复信号汇总', index=False)
    REP_DETAIL.to_excel(writer, sheet_name='重复信号配对', index=False)
    REPC.to_excel(writer, sheet_name='重复信号相关', index=False)
    EXT.to_excel(writer, sheet_name='极值明细', index=False)
    CASE.to_excel(writer, sheet_name='603778案例', index=False)
    TRADE.to_excel(writer, sheet_name='前二逐笔', index=False)
    DAY.to_excel(writer, sheet_name='日级组合收益', index=False)
    DAYC.to_excel(writer, sheet_name='日级相关', index=False)
    CROWD.to_excel(writer, sheet_name='信号拥挤度', index=False)
    NOTES.to_excel(writer, sheet_name='数据说明', index=False)

    wb = writer.book
    head_fill = PatternFill('solid', fgColor='1F4E79')
    head_font = Font(color='FFFFFF', bold=True, size=10)
    thin = Side(style='thin', color='D9D9D9')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    date_headers = ('信号日', '入场日', '退出日', '首笔信号日', '第二笔信号日')
    count_headers = ('笔数', '深组笔数', '浅组笔数', '剔除2026笔数', '剔除2026深组笔数',
                     '剔除2026浅组笔数', '可配对数', '对数', '样本数', '信号日数',
                     '排名', '年份', '段内位置', '历史同档笔数', '2026同档笔数', '前二只数', '当日信号数')
    price_headers = ('信号日收盘_后复权', '250日最高_后复权')
    corr_headers = ('皮尔逊相关', '斯皮尔曼相关', '数值')
    t_headers = ('t统计量', '市场t统计量')
    p_sci_headers = ('秩和检验p',)
    p_headers = ('配对t检验p', '皮尔逊p', '斯皮尔曼p', '显著性p')

    def fmt_for(header):
        if header in date_headers:
            return 'yyyy-mm-dd'
        if header in p_sci_headers:
            return '0.00E+00'
        if header in p_headers:
            return '0.0000'
        if header in count_headers:
            return '0'
        if header in ('平均持有交易日', '当日信号数中位'):
            return '0.0'
        if header == '持有交易日':
            return '0'
        if header in price_headers:
            return '0.00'
        if header == '相对强弱14':
            return '0.0'
        if header == '收盘位置':
            return '0.000'
        if header in corr_headers:
            return '0.000'
        if header in t_headers:
            return '0.00'
        if '百分点' in header:
            return '0.00'
        if any(t in header for t in ('收益', '胜率', '止盈率', '回撤', '占比',
                                     '标准差', '最差', '最好', '中位', '平均')):
            return '0.00"%"'
        return None

    for ws in wb.worksheets:
        ws.freeze_panes = 'A2'
        for cell in ws[1]:
            cell.fill = head_fill
            cell.font = head_font
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = border
        for col in range(1, ws.max_column + 1):
            letter = get_column_letter(col)
            header = str(ws.cell(1, col).value)
            width = max(10, min(44, max(len(str(ws.cell(r, col).value or ''))
                                   for r in range(1, min(ws.max_row, 300) + 1)) + 2))
            ws.column_dimensions[letter].width = width
            fmt = fmt_for(header)
            if fmt:
                for r in range(2, ws.max_row + 1):
                    ws.cell(r, col).number_format = fmt
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(vertical='center')

    for name in ('结论摘要', '口径与参数', '数据说明'):
        ws = wb[name]
        ws.column_dimensions['A'].width = 18
        ws.column_dimensions['B'].width = 110
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.alignment = Alignment(vertical='top', wrap_text=True)
        ws.freeze_panes = 'A2'

    for name in ('同日排名明细', '重复信号配对', '前二逐笔', '极值明细', '日级组合收益'):
        ws = wb[name]
        ws.freeze_panes = 'B2'
        ws.auto_filter.ref = ws.dimensions

    # 结论摘要隔行浅底，便于读
    ws = wb['结论摘要']
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = 44
    ws = wb['口径与参数']
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = 30
    ws = wb['数据说明']
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = 40
    ws = wb['当前信号']
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = 32
    for r in range(2, ws.max_row + 1):
        if r % 2 == 0:
            for c in range(1, 3):
                ws.cell(r, c).fill = PatternFill('solid', fgColor='F2F6FA')

print('written', path)
