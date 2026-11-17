# -*- coding: utf-8 -*-
"""路径分析的图与 Excel。读取 01_路径分析.py 生成的 CSV。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

DIR = '/Users/admin/WorkSpace/trading/2026-10-22_持仓天数_涨跌路径'
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

PATH = pd.read_csv(os.path.join(DIR, '路径统计.csv'), encoding='utf-8-sig')
GPATH = pd.read_csv(os.path.join(DIR, '赢家输家_路径.csv'), encoding='utf-8-sig')
EXIT = pd.read_csv(os.path.join(DIR, '固定天数退出.csv'), encoding='utf-8-sig')
SUMMARY = pd.read_csv(os.path.join(DIR, '逐笔路径摘要.csv'), encoding='utf-8-sig', low_memory=False)

C = {'主系统150': '#c0392b', '全信号': '#2e86c1', '70/15子池': '#27ae60'}
LS = {'主系统150': '-', '全信号': '-', '70/15子池': '--'}

# ---------------- 图1：2x2 ----------------
fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))

# (a) 上涨概率
ax = axes[0, 0]
for name in ['全信号', '70/15子池', '主系统150']:
    z = PATH[(PATH['样本'] == name) & (PATH['持有天数'] <= 20)]
    ax.plot(z['持有天数'], z['上涨概率'], color=C[name], ls=LS[name], lw=2, label=name)
ax.axhline(.5, color='#888', ls=':', lw=1)
ax.axvline(20, color='#888', ls=':', lw=1)
ax.set_title('① 持有到第 d 天仍上涨的概率（收盘）', fontsize=12)
ax.set_xlabel('持有天数 d')
ax.set_ylabel('上涨概率')
ax.set_xlim(1, 20); ax.set_ylim(.40, .65)
ax.set_xticks([1, 5, 10, 15, 20])
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.grid(alpha=.25)
ax.legend(loc='lower right', fontsize=9)

# (b) 条件涨幅/跌幅
ax = axes[0, 1]
for name in ['全信号', '主系统150']:
    z = PATH[(PATH['样本'] == name) & (PATH['持有天数'] <= 20)]
    ax.plot(z['持有天数'], z['上涨平均涨幅'], color=C[name], ls='-', lw=2,
            label=f'{name}·上涨时')
    ax.plot(z['持有天数'], z['下跌平均跌幅'], color=C[name], ls='--', lw=2,
            label=f'{name}·下跌时')
ax.axhline(0, color='#888', lw=.8)
ax.axvline(20, color='#888', ls=':', lw=1)
ax.set_title('② 上涨时平均涨多少 / 下跌时平均跌多少', fontsize=12)
ax.set_xlabel('持有天数 d')
ax.set_xlim(1, 20)
ax.set_xticks([1, 5, 10, 15, 20])
ax.yaxis.set_major_locator(MultipleLocator(.02))
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.grid(alpha=.25)
ax.legend(loc='lower left', fontsize=9)

# (c) 赢家/输家路径
ax = axes[1, 0]
for name, lw, alpha in [('全信号', 2.2, 1.0), ('主系统150', 1.4, .75)]:
    for grp, color, ls in [('最终上涨', '#27ae60', '-'), ('最终下跌', '#c0392b', '-')]:
        z = GPATH[(GPATH['样本'] == name) & (GPATH['组别'] == grp)]
        ax.plot(z['持有天数'], z['平均收益'], color=color, ls=ls, lw=lw, alpha=alpha,
                label=f'{name}·{grp}' if name == '全信号' else None)
ax.axhline(0, color='#888', lw=.8)
ax.axvline(20, color='#888', ls=':', lw=1)
ax.set_title('③ 最终赢家 vs 最终输家：平均路径（全信号实线，主系统细线）', fontsize=12)
ax.set_xlabel('持有天数 d')
ax.set_ylabel('平均收益')
ax.set_xlim(1, 20)
ax.set_xticks([1, 5, 10, 15, 20])
ax.yaxis.set_major_locator(MultipleLocator(.02))
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.grid(alpha=.25)
ax.legend(loc='center left', fontsize=9)

# (d) 赢家最高点 / 输家最低点出现在第几天
ax = axes[1, 1]
z = SUMMARY[SUMMARY['样本'] == '全信号']
w = z[z['第20天收益'] > 0]
l = z[z['第20天收益'] < 0]
pk = w['收盘最大浮盈日'].value_counts(normalize=True).reindex(range(1, 21), fill_value=0)
tr = l['收盘最大浮亏日'].value_counts(normalize=True).reindex(range(1, 21), fill_value=0)
x = np.arange(1, 21)
ax.bar(x - .2, pk.values, width=.4, color='#27ae60', label='最终赢家·最高点')
ax.bar(x + .2, tr.values, width=.4, color='#c0392b', label='最终输家·最低点')
ax.set_title('④ 赢家的最高点 vs 输家的最低点：出现在第几天', fontsize=12)
ax.set_xlabel('持有天数 d')
ax.set_ylabel('占本组比例')
ax.set_xticks([1, 3, 5, 7, 10, 12, 15, 18, 20])
ax.yaxis.set_major_locator(MultipleLocator(.02))
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.grid(alpha=.25, axis='y')
ax.legend(fontsize=9)

fig.suptitle('买入后 1–20 个交易日：涨跌概率、条件幅度与路径', fontsize=15, y=.995)
fig.tight_layout(rect=[0, 0, 1, .97])
fig.savefig(os.path.join(DIR, '持仓天数_涨跌路径.png'), dpi=160)
plt.close(fig)

# ---------------- 图2：固定持有天数 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))

ax = axes[0]
for name in ['全信号', '70/15子池', '主系统150']:
    z = EXIT[(EXIT['样本'] == name) & (EXIT['持有天数'] <= 60)]
    ax.plot(z['持有天数'], z['净平均'], color=C[name], lw=2, label=name)
ax.axhline(0, color='#888', lw=.8)
ax.axvline(20, color='#888', ls=':', lw=1)
ax.set_title('固定持有 N 天卖出：扣费后平均收益', fontsize=12)
ax.set_xlabel('持有天数 N')
ax.set_ylabel('平均净收益/笔')
ax.set_xlim(1, 60)
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.grid(alpha=.25)
ax.legend(fontsize=9)

ax = axes[1]
for name in ['全信号', '70/15子池', '主系统150']:
    z = EXIT[(EXIT['样本'] == name) & (EXIT['持有天数'] <= 60)]
    ax.plot(z['持有天数'], z['净胜率'], color=C[name], lw=2, label=name)
ax.axhline(.5, color='#888', ls=':', lw=1)
ax.axvline(20, color='#888', ls=':', lw=1)
ax.set_title('固定持有 N 天卖出：扣费后胜率', fontsize=12)
ax.set_xlabel('持有天数 N')
ax.set_ylabel('净胜率')
ax.set_xlim(1, 60)
ax.set_ylim(.35, .62)
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.grid(alpha=.25)
ax.legend(fontsize=9)

fig.suptitle('固定持有 N 天卖出：20 日附近胜率最高，继续拉长主要增加右尾', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, .94])
fig.savefig(os.path.join(DIR, '固定天数退出_对比.png'), dpi=160)
plt.close(fig)

# ---------------- Excel ----------------
NOTES = pd.DataFrame([
    ['样本', '主系统150：2026-10-21 目录的 150 笔实际交易；全信号：40,279 笔每笔1万；70/15子池：6,827 笔'],
    ['买入价', '信号次日开盘（t+1 open）。主系统150与全信号的买入价与面板开盘价最大差 < 0.000004'],
    ['持有第1天', '买入当日收盘；持有第20天=买入日+19个交易日收盘，与原系统“持有20日”一致'],
    ['收益口径', '路径表为毛收益（未扣费）；固定天数退出表同时给毛/净；净口径=单边0.15%、最低5元'],
    ['停牌', '停牌日收盘用前收盘填充；盘中最高/最低按窗口内可得数据累计'],
    ['对账', '全信号第20天净收益与原逐笔平均差 0.018个百分点；81笔原持有>20天（一字跌停顺延），其中72笔价格差>1个百分点，已保留但均值影响可忽略'],
    ['重叠', '全信号/70-15的逐笔高度重叠，概率按笔加权，不是独立样本；不能把40,279当40,279个独立机会'],
    ['止损', '主系统150有4笔-20%灾难止损；路径表是“不止损、持有到第20天”的反事实，净均值3.40% vs 实际3.30%'],
    ['结论边界', '描述性统计，不是新规则的回测；不构成投资建议'],
], columns=['项目', '说明'])

SHEETS = [
    ('说明', NOTES),
    ('路径概率', PATH),
    ('条件分布', pd.read_csv(os.path.join(DIR, '条件分布.csv'), encoding='utf-8-sig')),
    ('条件阈值概率', pd.read_csv(os.path.join(DIR, '条件阈值概率.csv'), encoding='utf-8-sig')),
    ('赢家输家', pd.read_csv(os.path.join(DIR, '赢家输家.csv'), encoding='utf-8-sig')),
    ('赢家输家路径', GPATH),
    ('极值触达', pd.read_csv(os.path.join(DIR, '极值触达.csv'), encoding='utf-8-sig')),
    ('固定天数退出', EXIT),
    ('续持条件', pd.read_csv(os.path.join(DIR, '续持条件.csv'), encoding='utf-8-sig')),
    ('分年关键天数', pd.read_csv(os.path.join(DIR, '分年_关键天数.csv'), encoding='utf-8-sig')),
]
out = os.path.join(DIR, '持仓天数_涨跌路径.xlsx')
with pd.ExcelWriter(out, engine='openpyxl') as writer:
    for sheet, df in SHEETS:
        df.to_excel(writer, sheet_name=sheet, index=False)

wb = load_workbook(out)
thin = Side(style='thin', color='DDDDDD')
border = Border(bottom=thin)
head_fill = PatternFill('solid', fgColor='1F4E79')
head_font = Font(color='FFFFFF', bold=True, size=10)
pct_cols = ('概率', '率', '收益', '涨幅', '跌幅', '浮盈', '浮亏', '回吐', '反弹', '保留',
            '增量', '分位', '平均', '中位', '占比', 'P10', 'P25', 'P75', 'P90', '差异')
day_cols = ('持有天数', '收盘最大浮盈日', '收盘最大浮亏日', '峰值日中位', '谷值日中位',
            '原记录持有天数', '20日有效收盘数', '笔数', '样本数', '年份')
for ws in wb.worksheets:
    ws.freeze_panes = 'A2'
    for cell in ws[1]:
        cell.fill = head_fill
        cell.font = head_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.row_dimensions[1].height = 30
    headers = [c.value for c in ws[1]]
    for j, h in enumerate(headers, start=1):
        width = max(9, min(26, len(str(h)) * 2 + 2))
        if h in ('说明',):
            width = 100
        ws.column_dimensions[get_column_letter(j)].width = width
        for i in range(2, ws.max_row + 1):
            cell = ws.cell(row=i, column=j)
            cell.border = border
            if h in ('样本', '组别', '代码', '买入日', '原记录退出原因', '项目', '说明', '年份'):
                cell.alignment = Alignment(horizontal='left', vertical='center')
                continue
            if h in day_cols:
                cell.number_format = '0'
            elif h and '盈亏比' in h:
                cell.number_format = '0.00'
            elif h and ('总盈亏' in h or ('净盈亏' in h and '率' not in h)):
                cell.number_format = '#,##0'
            elif h and ('买入价' in h or '价格' in h):
                cell.number_format = '0.000'
            elif h and any(k in h for k in pct_cols):
                cell.number_format = '0.0%'
            else:
                cell.number_format = '0.000'
    ws.auto_filter.ref = ws.dimensions
    if ws.title == '说明':
        ws.column_dimensions['A'].width = 14
        ws.column_dimensions['B'].width = 110
        for i in range(2, ws.max_row + 1):
            ws.row_dimensions[i].height = 44
            ws.cell(row=i, column=2).alignment = Alignment(wrap_text=True, vertical='top')
wb.save(out)

print('saved', out)
print('figs saved')
