# -*- coding: utf-8 -*-
"""历史年份同态：图表。"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
TEAL = '#1f6f8b'
NAVY = '#1b2a3a'
GREY = '#8a969f'
LIGHT = '#d9dee2'
ORANGE = '#d68910'

Y = pd.read_csv(RES / '年度相似度.csv')
S = pd.read_csv(RES / '同态期_策略起点.csv')
D = pd.read_csv(RES / '同态期_逐起点.csv')
M = pd.read_csv(RES / '最近邻_核心8.csv', parse_dates=['date'])
H = pd.read_csv(RES / '全部历史日_距离.csv', parse_dates=['date'])

# ------------------------------------------------------------------ 1 年度相似度
top = Y.head(6).copy()
top['year'] = top['年份'].astype(int).astype(str)

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
for ax, col, title in [
    (axes[0], '核心8_距离', '核心 8 特征'),
    (axes[1], '全状态13_距离', '全状态 13 特征'),
]:
    t = top.sort_values(col, ascending=False)
    colors = [RED if y == '2018' else TEAL if y == '2024' else LIGHT for y in t['year']]
    ax.barh(t['year'], t[col], color=colors, height=.62)
    for i, (_, r) in enumerate(t.iterrows()):
        ax.text(r[col] + .008, i, f"{r[col]:.3f}", va='center', fontsize=8.5, color=NAVY)
        if r['year'] in ('2018', '2024'):
            ax.text(r[col] * .82, i, '第一' if i == len(t) - 1 else '第二', va='center',
                    ha='right', fontsize=7.2, color='white', fontweight='bold')
    ax.set_xlim(0, max(t[col].max() * 1.22, .5))
    ax.set_title(title, loc='left', fontsize=10.5, color=NAVY, pad=8)
    ax.grid(axis='x', alpha=.18)
    ax.set_xlabel('距离（越小越像）', fontsize=8.5)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
axes[0].set_ylabel('')
fig.suptitle('2018 与 2024 稳居前二；换一套特征，第一第二名互换', x=.055, ha='left',
             fontsize=13, color=NAVY, y=1.02)
fig.text(.055, -.04, '距离 = 特征稳健标准化后的欧氏距离；年度取该年最近 10 个交易日均值。核心 8 口径下，第三名距离约为前两名的 2.5 倍；全状态口径差距较小。',
         fontsize=7.6, color=GREY)
plt.tight_layout()
plt.savefig(FIG / '01_年度相似度.png', facecolor='white')
plt.close()

# ------------------------------------------------------------------ 2 未来 120 日路径
# 重建全量日线路径
MP = HERE.parents[3].parent / 'trading_hithink' / '初始研究' / 'data' / 'work' / 'market_daily.parquet'
if not MP.exists():
    MP = HERE.parents[3].parent / 'trading_hithink' / 'data' / 'work' / 'market_daily.parquet'
mkt = pd.read_parquet(MP, columns=['date', 'equal_weight_index'])
mkt['date'] = pd.to_datetime(mkt['date'])
mkt = mkt.sort_values('date').reset_index(drop=True)
mkt = mkt[mkt['date'] >= '2018-01-01'].reset_index(drop=True)

cases = [
    ('2018-05-30', '2018-05-30 → -19.5%', RED, 2.0, '-'),
    ('2024-06-14', '2024-06-14 → +29.7%', TEAL, 2.0, '-'),
    ('2022-04-22', '2022-04-22 → +10.1%', GREY, 1.0, '--'),
    ('2021-01-29', '2021-01-29 → +13.8%', '#b9c2c9', 1.0, ':'),
]
fig, ax = plt.subplots(figsize=(10.2, 4.4))
for day, label, color, lw, ls in cases:
    pos = int(mkt.index[mkt['date'] == pd.Timestamp(day)][0])
    seg = mkt.iloc[pos:pos + 121]
    y = seg['equal_weight_index'] / seg['equal_weight_index'].iloc[0] * 100
    ax.plot(range(len(y)), y, color=color, lw=lw, ls=ls, label=label)
    ax.text(len(y) - 1, y.iloc[-1], f"{y.iloc[-1] - 100:+.1f}%", color=color,
            fontsize=9, fontweight='bold', va='center', ha='left')
ax.axhline(100, color=NAVY, lw=.7, alpha=.5)
ax.axvline(60, color=GREY, lw=.7, ls=(0, (2, 3)), alpha=.7)
ax.text(61, 103.5, '60 日', fontsize=7.5, color=GREY)
ax.scatter([0], [100], color=NAVY, zorder=5, s=22)
ax.set_xlim(0, 133)
ax.set_ylim(72, 137)
ax.set_ylabel('等权指数（代表日 = 100）', fontsize=9)
ax.set_xlabel('同态日后的交易日', fontsize=9)
ax.grid(alpha=.16)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
ax.set_title('相同的价量状态，120 天后：-19.5% 与 +29.7%', loc='left', fontsize=13, color=NAVY)
fig.text(.01, -.03, '沪深主板等权指数；代表日收盘 = 100。2022/2021 为较远的次级同态，仅作参照。',
         fontsize=7.6, color=GREY)
plt.tight_layout()
plt.savefig(FIG / '02_同态路径.png', facecolor='white')
plt.close()

# ------------------------------------------------------------------ 3 距离 vs 未来60日
df = H[['date', 'd_核心8', 'f60']].dropna().copy()
fig, ax = plt.subplots(figsize=(10.2, 4.3))
ax.scatter(df['d_核心8'], df['f60'] * 100, s=8, color=GREY, alpha=.20, linewidth=0)
hi = df[df['d_核心8'] <= .25]
ax.scatter(hi['d_核心8'], hi['f60'] * 100, s=26, color=ORANGE, alpha=.85, linewidth=0,
           label='距离 ≤ 0.25 的最近邻')
for d, name, color in [('2018-05-30', '2018-05-30（-14.8%）', RED), ('2024-06-14', '2024-06-14（-8.1%）', TEAL)]:
    r = df[df['date'] == pd.Timestamp(d)]
    if len(r):
        ax.scatter(r['d_核心8'], r['f60'] * 100, s=55, color=color, zorder=5,
                   edgecolor='white', linewidth=.8, label=name)
ax.axhline(0, color=NAVY, lw=.8, alpha=.55)
coef = np.polyfit(df['d_核心8'], df['f60'], 1)
xx = np.linspace(df['d_核心8'].min(), df['d_核心8'].max(), 100)
ax.plot(xx, np.polyval(coef, xx) * 100, color=NAVY, lw=1.2, ls='--', alpha=.8)
r = df['d_核心8'].corr(df['f60'])
ax.text(.98, .06, f'Pearson r = {r:+.2f}（弱）', transform=ax.transAxes, ha='right',
        fontsize=8.5, color=NAVY)
ax.set_xlabel('与当前状态的距离（越小越像）', fontsize=9)
ax.set_ylabel('未来 60 日等权市场收益', fontsize=9)
ax.yaxis.set_major_formatter(PercentFormatter())
ax.grid(alpha=.16)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(loc='upper right', fontsize=8, frameon=False)
ax.set_title('相似度不是方向信号：最近的两段，60 日都是负的', loc='left', fontsize=13, color=NAVY)
fig.text(.01, -.035, '2018-05 至 2025-12 的 1,865 个交易日；点高度重叠，相关不代表因果。',
         fontsize=7.6, color=GREY)
plt.tight_layout()
plt.savefig(FIG / '03_距离与未来60日.png', facecolor='white')
plt.close()

# ------------------------------------------------------------------ 4 市场 vs 策略
main = S[S['主同态'] == '是'].set_index('代表日')


def market_val(day, col):
    r = M[M['date'] == pd.Timestamp(day)]
    return float(r[col].iloc[0]) if len(r) else np.nan


labels = ['2018\n市场', '2018\n策略', '2024-06\n市场', '2024-06\n策略']
v60 = [market_val('2018-05-30', 'f60'), main.loc['2018-05-30', '下一可执行起点60日'],
       market_val('2024-06-14', 'f60'), main.loc['2024-06-14', '下一可执行起点60日']]
v120 = [market_val('2018-05-30', 'f120'), main.loc['2018-05-30', '下一可执行起点120日'],
        market_val('2024-06-14', 'f120'), main.loc['2024-06-14', '下一可执行起点120日']]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={'width_ratios': [1.15, 1]})
ax = axes[0]
x = np.arange(len(labels))
w = .36
b1 = ax.bar(x - w / 2, np.array(v60) * 100, w, color=NAVY, label='60 日')
b2 = ax.bar(x + w / 2, np.array(v120) * 100, w, color='#7fb3d5', label='120 日')
for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + (1.2 if h >= 0 else -2.2),
                f"{h:+.1f}%", ha='center', va='bottom' if h >= 0 else 'top',
                fontsize=8, color=NAVY)
ax.axhline(0, color=NAVY, lw=.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('收益', fontsize=9)
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylim(-32, 38)
ax.grid(axis='y', alpha=.16)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
ax.set_title('市场分叉，策略没有分叉', loc='left', fontsize=11.5, color=NAVY)

ax = axes[1]
for i, (day, color, name) in enumerate([('2018-05-30', RED, '2018'), ('2024-06-14', TEAL, '2024-06')]):
    g = D[D['代表日'] == day]['ret_60'].dropna() * 100
    jitter = np.linspace(-.12, .12, len(g)) if len(g) > 1 else np.zeros(len(g))
    ax.scatter(np.full(len(g), i) + jitter, g, s=30, color=color, alpha=.8, linewidth=0,
               zorder=4)
    ax.hlines(g.median(), i - .22, i + .22, color=color, lw=2.2, zorder=5)
    ax.text(i + .26, g.median(), f'中位 {g.median():+.1f}%', va='center', fontsize=8,
            color=color, fontweight='bold')
ax.axhline(0, color=NAVY, lw=.8, alpha=.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(['2018 同态区间', '2024-06 同态区间'], fontsize=8.5)
ax.set_ylabel('策略起点后 60 日收益', fontsize=9)
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_xlim(-.5, 1.8)
ax.grid(axis='y', alpha=.16)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set_title('同一个同态区间，起点差几天也差很多', loc='left', fontsize=11.5, color=NAVY)
fig.text(.01, -.035, '左：市场为等权指数；策略为下一可执行信号日起点、2 仓 50%、15 日、+20% 止盈。右：区间内每个信号日起点，一点一个起点。',
         fontsize=7.5, color=GREY)
plt.tight_layout()
plt.savefig(FIG / '04_市场与策略.png', facecolor='white')
plt.close()

print('charts done')
