# -*- coding: utf-8 -*-
"""把结果表汇总为一个中文 Excel（表名/表头全中文）。"""
import pandas as pd, numpy as np, os
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

BASE = '/Users/admin/WorkSpace/trading/2026-10-22_当前阶段_情绪同态_50%策略'
R = os.path.join(BASE, '03_结果')
OUT = os.path.join(BASE, '当前阶段_情绪同态_50%策略_数据表.xlsx')

def read(name, **kw):
    p = os.path.join(R, name)
    return pd.read_csv(p, encoding='utf-8-sig', **kw) if os.path.exists(p) else None

sheets = {}
sheets['说明'] = pd.DataFrame({
    '项目': ['研究日期', '数据区间', '行情口径', '市场指数', '当前阶段', '策略', '重要提醒', '复现'],
    '内容': ['2026-10-22 收盘', '2016-10-13 至 2026-10-22（个股面板到 2026-10-13，市场/行业扩展到 10-22）',
             '全A后复权日线；T+1 开盘；单边 0.15%，最低 5 元',
             '全A等权指数：个股日收益截断 ±20% 后横截面均值，累乘',
             '高位调整后的冰点：宽度20 11.8%，情绪分 3% 分位，指数低于 250 日线，距高点 -20%',
             '70/15 趋势系统，2 仓、单票 50%、20 日持有、-20% 灾难止损；空仓等待宽度>50%且行业闸门打开',
             '历史阶段收益不等于未来；2018/2023/2024 底部/2026 调整为负或接近零；前五笔占利润 70.7%，去掉前五笔净盈亏 6.29 万（总 21.50 万）',
             '见《03_数据与复现说明.md》；每日先跑 10 再跑 05']})

for src, dst in [('当前状态_2026-10-22.csv', '当前状态'),
                 ('阶段表.csv', '阶段表'),
                 ('当前同态_事件表.csv', '同态事件'),
                 ('同态事件_明细.csv', '同态事件明细'),
                 ('当前同态_最近邻_含长期.csv', '最近邻'),
                 ('同态横截面_组合.csv', '同态组合'),
                 ('同态横截面_行业.csv', '同态行业'),
                 ('同态横截面_特征分箱.csv', '同态特征分箱'),
                 ('阶段触发时间.csv', '阶段触发时间'),
                 ('核心2仓50_阶段.csv', '策略阶段'),
                 ('核心2仓50_年度.csv', '策略年度'),
                 ('核心2仓50_分段.csv', '策略分段'),
                 ('核心2仓50_逐笔.csv', '策略逐笔'),
                 ('7015_仓位版本.csv', '仓位版本'),
                 ('当前行业_2026-10-22.csv', '当前行业'),
                 ('当前超跌观察池_2026-10-22.csv', '超跌观察池'),
                 ('今日闸门_2026-10-22.csv', '今日闸门'),
                 ('主要指数_2026-10-22.csv', '主要指数'),
                 ('闸门后前瞻.csv', '闸门后前瞻'),
                 ('同态条件_前瞻.csv', '同态条件前瞻')]:
    d = read(src)
    if d is not None:
        sheets[dst] = d


# 全中文表头：按完整列名替换，避免 f20/f60 等拉丁残留
colmap = {
    'date': '日期', 'thscode': '代码', 'market_index': '等权指数', 'idx_pct': '指数分位',
    'ret5': '5日收益', 'ret20': '20日收益', 'ret60': '60日收益', 'ret120': '120日收益',
    'ret250': '250日收益', 'dd250': '250日回撤', 'breadth20': '宽度20', 'breadth60': '宽度60',
    'breadth120': '宽度120', 'limitup': '涨停占比', 'limitdown': '跌停占比', 'newhigh': '新高占比',
    'amt_ratio': '成交额热度', 'dispersion': '离散度', 'vol20': '20日波动', 'sent': '情绪分',
    'regime': '日度状态', 'f5': '5日', 'f10': '10日', 'f20': '20日', 'f60': '60日', 'f120': '120日',
    'min20': '20日路径最低', 'min60': '60日路径最低', 'min120': '120日路径最低',
    'max20': '20日路径最高', 'max60': '60日路径最高', 'max120': '120日路径最高',
    'f5均值': '5日均值', 'f10均值': '10日均值', 'f20均值': '20日均值', 'f60均值': '60日均值',
    'f120均值': '120日均值', 'f20中位': '20日中位', 'f60中位': '60日中位',
    'f20胜率': '20日胜率', 'f60胜率': '60日胜率', 'f120胜率': '120日胜率',
    'min20均值': '20日路径最低均值', 'min60均值': '60日路径最低均值', 'min120均值': '120日路径最低均值',
    'max20均值': '20日路径最高均值', 'max60均值': '60日路径最高均值', 'max120均值': '120日路径最高均值',
    'min60最差': '60日路径最低最差', 'max60最好': '60日路径最高最好',
    'P25': '25分位', 'P75': '75分位', 'P90': '90分位', '超30': '涨超30%', '超50': '涨超50%',
    'ATR%': '波幅%', 'R20': '20日收益', 'DD60': '60日回撤', '破MA250': '破250日线',
    '20日均额万': '20日均额(万)', '高水位': '高水位',
    'med_ret1': '中位日收益', 'mkt_ret': '市场日收益', 'ma20': '20日线', 'ma60': '60日线',
    'ma120': '120日线', 'ma250': '250日线', 'slope60': '60日线斜率', 'ret500': '500日收益'
}
for k in list(sheets):
    sheets[k] = sheets[k].rename(columns={c: colmap.get(c, c) for c in sheets[k].columns})

# 中文表头/值
if '策略逐笔' in sheets:
    t = sheets['策略逐笔']
    ren = {'code': '代码', 'entry_date': '买入日', 'exit_date': '卖出日', 'entry': '买入价',
           'exit': '卖出价', 'ret': '单笔收益', 'reason': '卖出原因', 'hold': '持有天数',
           'shares': '股数', 'pnl': '净盈亏', 'weight': '目标权重'}
    t = t.rename(columns={k: v for k, v in ren.items() if k in t.columns})
    sheets['策略逐笔'] = t

if '同态特征分箱' in sheets:
    t = sheets['同态特征分箱']
    mp = {'市值': '市值', '收盘': '收盘价', 'ATR': '波幅', 'RSI': '相对强弱', '波动20': '20日波动',
          'R5': '5日收益', 'R20': '20日收益', 'R60': '60日收益', 'R120': '120日收益',
          '距MA20': '距20日线', '距MA60': '距60日线', '回撤60': '60日回撤', '回撤250': '250日回撤',
          '涨停20': '20日涨停数', '成交额': '20日均额', '行业宽度': '行业宽度', '行业20日': '行业20日收益'}
    t['特征'] = t['特征'].map(mp).fillna(t['特征'])
    sheets['同态特征分箱'] = t

with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    for name, d in sheets.items():
        d.to_excel(w, sheet_name=name[:31], index=False)
        ws = w.sheets[name[:31]]
        ws.freeze_panes = 'A2'
        header_fill = PatternFill('solid', fgColor='D9E2F3')
        thin = Side(style='thin', color='BFBFBF')
        for c in range(1, ws.max_column + 1):
            cell = ws.cell(row=1, column=c)
            cell.font = Font(bold=True, size=10)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = Border(bottom=thin)
        ws.row_dimensions[1].height = 30
        for c in range(1, ws.max_column + 1):
            vals = [ws.cell(row=r, column=c).value for r in range(2, min(ws.max_row, 200) + 1)]
            ln = max([len(str(v)) for v in vals if v is not None] + [len(str(ws.cell(row=1, column=c).value))])
            ws.column_dimensions[get_column_letter(c)].width = min(42, max(9, ln * 1.15 + 2))
        for r in range(2, ws.max_row + 1):
            for c in range(1, ws.max_column + 1):
                v = ws.cell(row=r, column=c).value
                if isinstance(v, float):
                    ws.cell(row=r, column=c).number_format = '0.00%' if abs(v) < 3 and ('率' in str(ws.cell(row=1, column=c).value) or '收益' in str(ws.cell(row=1, column=c).value) or '回撤' in str(ws.cell(row=1, column=c).value)) else '0.0000'
                ws.cell(row=r, column=c).alignment = Alignment(vertical='center')
print('saved', OUT, 'sheets', len(sheets))
