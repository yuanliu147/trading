# -*- coding: utf-8 -*-
"""603778 K 线与特征：图表。"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
RES = OUT / '03_结果'
FIG = OUT / '04_图表'
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 180
plt.rcParams['savefig.bbox'] = 'tight'

RED = '#b03a2e'
GREEN = '#2e7d5b'
TEAL = '#1f6f8b'
NAVY = '#1b2a3a'
GREY = '#8a969f'
GOLD = '#b8892f'
UP = '#c0392b'
DOWN = '#2e8b57'

PANEL = '/Users/admin/WorkSpace/trading/trading_flash/2026-10-24_主板恐慌反转_年化30/01_数据/research_panel.parquet'
cols = ['thscode', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount',
        'hfq_open', 'hfq_high', 'hfq_low', 'hfq_close',
        'rsi14', 'macd_dif_pct', 'macd_dea_pct', 'macd_hist_pct', 'macd_hist_delta',
        'close_location', 'drawdown250', 'return_250d', 'close_ma5', 'close_ma10',
        'close_ma20', 'close_ma60', 'close_ma250', 'amount_ratio1_20']
raw = pd.read_parquet(PANEL, columns=cols)
raw['date'] = pd.to_datetime(raw['date'])
g = raw[raw['thscode'] == '603778.SH'].sort_values('date').reset_index(drop=True).copy()
cur_act = float(g.loc[g['date'] == '2026-10-23', 'close'].iloc[0])
cur_hfq = float(g.loc[g['date'] == '2026-10-23', 'hfq_close'].iloc[0])
fac = cur_act / cur_hfq
for c in ['hfq_open', 'hfq_high', 'hfq_low', 'hfq_close']:
    g[c.replace('hfq_', 'px_')] = g[c] * fac
g['ma5'] = g['px_close'].rolling(5).mean()
g['ma10'] = g['px_close'].rolling(10).mean()
g['ma20'] = g['px_close'].rolling(20).mean()
g['ma60'] = g['px_close'].rolling(60).mean()
g['ma250'] = g['px_close'].rolling(250).mean()
g['vol20'] = g['volume'].rolling(20).mean()

# 10/26 partial
SIG26 = '/Users/admin/WorkSpace/trading/trading_flash/2026-10-24_主板恐慌反转_年化30/03_结果/当前信号_2026-10-26.csv'
r26 = pd.read_csv(SIG26)
r26 = r26[r26['thscode'] == '603778.SH'].iloc[0]


def candles(ax, data, width=1.0, up=UP, down=DOWN):
    for i, r in data.iterrows():
        color = up if r['px_close'] >= r['px_open'] else down
        ax.vlines(r['date'], r['px_low'], r['px_high'], color=color, linewidth=.8, zorder=3)
        body_low = min(r['px_open'], r['px_close'])
        body_h = max(abs(r['px_close'] - r['px_open']), 0.002)
        ax.add_patch(Rectangle((mdates.date2num(r['date']) - width / 2, body_low), width, body_h,
                               facecolor=color, edgecolor=color, linewidth=.3, zorder=4))


# ------------------------------------------------------------------ 1 两年K线
d1 = g[(g['date'] >= '2025-04-01') & (g['date'] <= '2026-10-23')].copy()
fig, (ax, axv) = plt.subplots(2, 1, figsize=(11.0, 6.2), sharex=True,
                              gridspec_kw={'height_ratios': [3.4, 1]})
candles(ax, d1)
for c, color, lw, label in [('ma20', TEAL, 1.2, 'MA20'), ('ma60', GOLD, 1.2, 'MA60'), ('ma250', NAVY, 1.4, 'MA250')]:
    ax.plot(d1['date'], d1[c], color=color, lw=lw, label=label)
ax.set_ylabel('当前价等价（元）', fontsize=9)
ax.grid(alpha=.14)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=3)
# annotations
peak = d1.loc[d1['px_high'].idxmax()]
jul = d1[(d1['date'] >= '2026-07-01') & (d1['date'] <= '2026-07-31')]
jul_low = jul.loc[jul['px_low'].idxmin()]
octl = d1[(d1['date'] >= '2026-10-01')]
oct_low = octl.loc[octl['px_low'].idxmin()]
ann = [
    (peak['date'], peak['px_high'], f"14倍高点\n{peak['px_high']:.2f}", (20, 18), RED),
    (jul_low['date'], jul_low['px_low'], f"7月低点 {jul_low['px_low']:.2f}", (-50, -28), GREEN),
    (oct_low['date'], oct_low['px_low'], f"10月低点 {oct_low['px_low']:.2f}", (-40, -38), RED),
    (pd.Timestamp('2026-10-23'), 7.33, '10/23 收盘 7.33', (-45, 30), NAVY),
]
for x, y, text, offset, color in ann:
    ax.annotate(text, (x, y), xytext=offset, textcoords='offset points', fontsize=8,
                color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=color, lw=.7, alpha=.8))
# fib
for level, label, color in [(9.29, '78.6% 9.29', GREY), (6.11, '88.6% 6.11', RED)]:
    ax.axhline(level, color=color, lw=.8, ls=(0, (4, 3)), alpha=.7)
    ax.text(d1['date'].iloc[-1], level, f' {label}', va='center', fontsize=7.5, color=color)
ax.set_title('603778：14 倍泡沫后的 -79%，现在在 88.6% 回撤附近', loc='left', fontsize=13, color=NAVY)

colors = [UP if r['px_close'] >= r['px_open'] else DOWN for _, r in d1.iterrows()]
axv.bar(d1['date'], d1['volume'] / 1e6, color=colors, width=1.0, alpha=.85)
axv.plot(d1['date'], d1['vol20'] / 1e6, color=NAVY, lw=1.0, label='20日均量')
axv.set_ylabel('成交量（百万股）', fontsize=8)
axv.grid(alpha=.14)
axv.legend(loc='upper left', fontsize=7.5, frameon=False)
for s in ['top', 'right']:
    axv.spines[s].set_visible(False)
axv.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axv.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(axv.get_xticklabels(), fontsize=8)
fig.text(.01, -.02, '后复权价格按 2026-10-23 实际价/后复权价折算成当前价等价；历史价格不是当时成交价。红涨绿跌。',
         fontsize=7.4, color=GREY)
plt.tight_layout()
plt.savefig(FIG / '01_两年K线.png', facecolor='white')
plt.close()

# ------------------------------------------------------------------ 2 近月K线
d2 = g[(g['date'] >= '2026-08-15') & (g['date'] <= '2026-10-23')].copy()
fig = plt.figure(figsize=(11.0, 7.0))
gs = fig.add_gridspec(4, 1, height_ratios=[3.0, 1.0, 1.1, 1.1], hspace=.08)
ax = fig.add_subplot(gs[0])
axv = fig.add_subplot(gs[1], sharex=ax)
axr = fig.add_subplot(gs[2], sharex=ax)
axm = fig.add_subplot(gs[3], sharex=ax)
candles(ax, d2, width=.9)
for c, color, lw, label in [('ma5', GREY, .9, 'MA5'), ('ma10', TEAL, 1.0, 'MA10'), ('ma20', GOLD, 1.1, 'MA20')]:
    ax.plot(d2['date'], d2[c], color=color, lw=lw, label=label)
ax.axhline(7.56, color=GREEN, lw=.8, ls=(0, (4, 3)), alpha=.7)
ax.text(d2['date'].iloc[0], 7.58, '7月低点 7.56', fontsize=7.5, color=GREEN, va='bottom')
ax.axhline(6.72, color=RED, lw=.8, ls=(0, (4, 3)), alpha=.7)
ax.text(d2['date'].iloc[0], 6.74, '10/23低点 6.72', fontsize=7.5, color=RED, va='bottom')
ax.axhline(8.80, color=NAVY, lw=.8, ls=(0, (2, 3)), alpha=.6)
ax.text(d2['date'].iloc[0], 8.83, '+20%目标 8.80', fontsize=7.5, color=NAVY, va='bottom')
ax.set_ylabel('当前价等价', fontsize=8)
ax.grid(alpha=.14)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(loc='upper right', fontsize=7.5, frameon=False, ncol=3)
# annotations
for d, y, text, offset, color in [
    ('2026-10-23', 7.33, '10/23 BOR\n低点破前低，收盘破前收\nCL 0.92', (-100, 72), GREEN),
]:
    row = d2[d2['date'] == pd.Timestamp(d)].iloc[0]
    ax.annotate(text, (row['date'], y), xytext=offset, textcoords='offset points',
                fontsize=7.8, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=.78),
                arrowprops=dict(arrowstyle='-', color=color, lw=.7, alpha=.85))
# 10/26 dot
ax.scatter([pd.Timestamp('2026-10-26')], [7.29], color=GOLD, s=45, zorder=8, edgecolor='white', linewidth=.8)
ax.annotate('10/26 收 7.29（CL 0.45，仅收盘）', (pd.Timestamp('2026-10-26'), 7.29),
            xytext=(-8, -30), textcoords='offset points', ha='right', fontsize=7.5, color=GOLD,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=.82),
            arrowprops=dict(arrowstyle='-', color=GOLD, lw=.7))
ax.set_title('近月：10/23 是 BOR 反转，10/26 是弱跟随', loc='left', fontsize=12.5, color=NAVY)
# volume
colors = [UP if r['px_close'] >= r['px_open'] else DOWN for _, r in d2.iterrows()]
axv.bar(d2['date'], d2['volume'] / 1e6, color=colors, width=.9, alpha=.85)
axv.plot(d2['date'], d2['vol20'] / 1e6, color=NAVY, lw=1.0)
axv.set_ylabel('量（百万）', fontsize=7.5)
axv.grid(alpha=.14)
for s in ['top', 'right']:
    axv.spines[s].set_visible(False)
# RSI
axr.plot(d2['date'], d2['rsi14'], color=NAVY, lw=1.2)
axr.axhline(30, color=RED, lw=.7, ls='--', alpha=.7)
axr.axhline(50, color=GREY, lw=.7, ls=':', alpha=.7)
axr.set_ylabel('RSI14', fontsize=7.5)
axr.set_ylim(15, 60)
axr.grid(alpha=.14)
for s in ['top', 'right']:
    axr.spines[s].set_visible(False)
# MACD hist
colors = [UP if x >= 0 else DOWN for x in d2['macd_hist_pct']]
axm.bar(d2['date'], d2['macd_hist_pct'] * 100, color=colors, width=.9, alpha=.85)
axm.axhline(0, color=NAVY, lw=.7)
axm.set_ylabel('MACD柱（pp）', fontsize=7.5)
axm.grid(alpha=.14)
for s in ['top', 'right']:
    axm.spines[s].set_visible(False)
axm.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
axm.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.setp(axm.get_xticklabels(), fontsize=8)
fig.text(.01, -.01, '后复权；10/26 仅有收盘与部分特征，OHLC 未入库。红涨绿跌；BOR=低点破前低、收盘高于前收、收盘高于开盘。',
         fontsize=7.3, color=GREY)
plt.savefig(FIG / '02_近月K线.png', facecolor='white')
plt.close()

# ------------------------------------------------------------------ 3 信号日散点
S = pd.read_csv(RES / '信号日特征.csv', parse_dates=['信号日'])
S['x'] = S['250日回撤'] * 100
S['y'] = S['收盘位置']
fig, ax = plt.subplots(figsize=(9.8, 4.6))
ax.axvspan(-90, -65, color='#f5f0e3', alpha=.55, zorder=0)
ax.axhspan(0.8, 1.05, color='#e8f1f3', alpha=.55, zorder=0)
for _, r in S.iterrows():
    if pd.isna(r['15日收益']):
        color = GOLD
    elif r['15日收益'] > 0:
        color = TEAL
    else:
        color = RED
    marker = '*' if r['信号日'] == pd.Timestamp('2026-10-23') else ('o' if r.get('BOR', False) else 's')
    ax.scatter(r['x'], r['y'], s=120 if marker == '*' else 55, color=color, marker=marker,
               edgecolor='white' if marker != '*' else NAVY, linewidth=.8, zorder=5)
    label = r['信号日'].strftime('%m-%d') + ('（第2）' if r['排名'] == 2 else '')
    if r['信号日'] == pd.Timestamp('2026-10-23'):
        label += '\n当前'
    ax.annotate(label, (r['x'], r['y']), xytext=(6, 7 if r['信号日'] != pd.Timestamp('2026-07-16') else -14),
                textcoords='offset points', fontsize=7.5, color=NAVY)
# 10/26 partial
ax.scatter([-78.73], [0.4545], marker='D', color=GOLD, s=70, edgecolor=NAVY, linewidth=.8, zorder=6)
ax.annotate('10-26 当前（CL 0.45）', (-78.73, 0.4545), xytext=(12, -10), textcoords='offset points',
            ha='left', fontsize=7.2, color=GOLD)
ax.axvline(-65, color=GREY, lw=.8, ls='--')
ax.axhline(0.8, color=GREY, lw=.8, ls='--')
ax.set_xlim(-82, -58)
ax.set_ylim(0.35, 1.04)
ax.set_xlabel('信号日 250 日回撤（%）', fontsize=9)
ax.set_ylabel('收盘位置（0=最低，1=最高）', fontsize=9)
ax.grid(alpha=.14)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
# legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=RED, markersize=7, label='亏损'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAL, markersize=7, label='盈利'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=GOLD, markersize=7, label='未完成/当前'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=GREY, markeredgecolor=NAVY, markersize=7, label='BOR（圆）'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=GREY, markersize=7, label='非BOR（方）'),
]
ax.legend(handles=handles, loc='lower right', fontsize=7.5, frameon=False, ncol=2)
ax.set_title('信号日特征：10/23 在“深回撤 + 高收盘”格子，10/26 掉到低收盘', loc='left',
             fontsize=12.5, color=NAVY)
fig.text(.01, -.03, '颜色=15日结果；形状=BOR/非BOR；浅色区域=深回撤（≤-65%）与高收盘（>0.8）。样本仅 603778 的 9 个信号日。',
         fontsize=7.4, color=GREY)
plt.tight_layout()
plt.savefig(FIG / '03_信号日散点.png', facecolor='white')
plt.close()

# ------------------------------------------------------------------ 4 条件收益
C = pd.read_csv(RES / '条件统计.csv')
C = C[~C['条件'].str.contains('当前近似组合')].copy()
order = ['全部信号', '深回撤 ≤-65%', '深回撤 + 收盘位置≤0.8', '深回撤 + 收盘位置>0.8',
         '深回撤 + BOR', '深回撤 + BOR + 高收盘',
         '深回撤 + 高收盘 + MACD≤0', '深回撤 + 高收盘 + 量比<1', '深回撤 + 高收盘 + RSI≥30']
C = C.set_index('条件').loc[order].reset_index()
C = C.sort_values('均值')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.0), sharey=True,
                               gridspec_kw={'width_ratios': [1.35, 1]})
y = np.arange(len(C))
colors = [RED if 'BOR + 高收盘' in x else (GOLD if ('高收盘' in x or 'BOR' in x) else GREY) for x in C['条件']]
ax1.barh(y, C['均值'] * 100, color=colors, height=.6)
for i, (_, r) in enumerate(C.iterrows()):
    ax1.text(r['均值'] * 100 + .25, i, f"{r['均值']*100:+.1f}%  (n={int(r['样本'])})",
             va='center', fontsize=8, color=NAVY)
ax1.set_yticks(y)
ax1.set_yticklabels(C['条件'], fontsize=8.5)
ax1.axvline(0, color=NAVY, lw=.8)
ax1.set_xlabel('未来 15 日平均收益', fontsize=9)
ax1.xaxis.set_major_formatter(PercentFormatter())
ax1.grid(axis='x', alpha=.16)
for s in ['top', 'right']:
    ax1.spines[s].set_visible(False)
ax2.barh(y, C['胜率'] * 100, color=colors, height=.6)
for i, (_, r) in enumerate(C.iterrows()):
    ax2.text(r['胜率'] * 100 + .6, i, f"{r['胜率']*100:.0f}%", va='center', fontsize=8, color=NAVY)
ax2.set_xlim(0, 105)
ax2.set_xlabel('胜率', fontsize=9)
ax2.xaxis.set_major_formatter(PercentFormatter())
ax2.grid(axis='x', alpha=.16)
for s in ['top', 'right']:
    ax2.spines[s].set_visible(False)
ax1.set_title('BOR + 高收盘是最强的一档；但只有 17 笔，不能当概率', loc='left', fontsize=12, color=NAVY)
fig.text(.01, -.035, '450 条完整信号；分组重叠、样本非独立；这是描述性统计，不是新规则。金条=高收盘（>0.8）组合。',
         fontsize=7.4, color=GREY)
plt.tight_layout()
plt.savefig(FIG / '04_条件收益.png', facecolor='white')
plt.close()

# ------------------------------------------------------------------ 5 相对强弱
mkt = pd.read_parquet('/Users/admin/WorkSpace/trading/trading_hithink/初始研究/data/work/market_daily.parquet',
                      columns=['date', 'equal_weight_index'])
mkt['date'] = pd.to_datetime(mkt['date'])
rel = pd.DataFrame({'stock': g.set_index('date')['hfq_close'],
                    'market': mkt.set_index('date')['equal_weight_index']}).loc['2026-07-21':'2026-10-23'].dropna()
rel = rel / rel.iloc[0] * 100
fig, ax = plt.subplots(figsize=(9.8, 4.0))
ax.plot(rel.index, rel['stock'], color=RED, lw=1.8, label='603778')
ax.plot(rel.index, rel['market'], color=NAVY, lw=1.5, label='沪深主板等权')
ax.axhline(100, color=GREY, lw=.7, alpha=.6)
ax.annotate(f"603778 {rel['stock'].iloc[-1]-100:+.1f}%", (rel.index[-1], rel['stock'].iloc[-1]),
            xytext=(-10, -14), textcoords='offset points', ha='right', fontsize=8.5, color=RED)
ax.annotate(f"市场 {rel['market'].iloc[-1]-100:+.1f}%", (rel.index[-1], rel['market'].iloc[-1]),
            xytext=(-10, 8), textcoords='offset points', ha='right', fontsize=8.5, color=NAVY)
ax.set_ylabel('2026-07-21 = 100', fontsize=9)
ax.grid(alpha=.16)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(loc='upper right', fontsize=8, frameon=False)
ax.set_title('相对强弱：7 月低点以来，市场 +3.8%，603778 -14.5%', loc='left', fontsize=12.5, color=NAVY)
fig.text(.01, -.035, '后复权；市场为沪深主板等权。自 2026-04-09 高点以来，603778 -77.7%，市场 -11.2%。',
         fontsize=7.4, color=GREY)
plt.tight_layout()
plt.savefig(FIG / '05_相对强弱.png', facecolor='white')
plt.close()

print('charts done')
