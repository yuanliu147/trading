# -*- coding: utf-8 -*-
"""市场阶段适配：图表。"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
plt.rcParams['figure.dpi'] = 150

A = pd.read_csv(RES / '年度对比.csv')
P = pd.read_csv(RES / '阶段统计.csv')
V = pd.read_csv(RES / '波动分组.csv')
M = pd.read_csv(RES / '市场阶段日度.csv', parse_dates=['date'])
F = pd.read_csv(RES / '起点固定持有期.csv', parse_dates=['signal_date'])
F['y'] = F['signal_date'].dt.year
C = pd.read_csv(RES / '2026起点.csv', parse_dates=['signal_date'])
D = pd.read_csv(RES / '全部信号日逐只_带阶段.csv', parse_dates=['signal_date'])

BLUE, RED, ORANGE, GREY = '#1f4e79', '#c0392b', '#e67e22', '#7f8c8d'

# 1 年度策略 vs 市场
fig, ax = plt.subplots(figsize=(10.5, 4.8))
x = np.arange(len(A))
w = 0.38
ax.bar(x - w / 2, A['策略收益'] * 100, w, color=BLUE, label='策略（连续账户）')
ax.bar(x + w / 2, A['市场等权'] * 100, w, color='#bdc3c7', label='沪深主板等权市场')
ax.axhline(0, color='k', lw=.8)
ax.set_xticks(x)
ax.set_xticklabels([f'{int(y)}*' if y in (2017, 2026) else str(int(y)) for y in A['年份']])
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylabel('年度收益')
ax.set_title('策略不是“牛市策略”：2018、2022 熊市反而大幅跑赢，2021 牛市几乎没赚')
ax.grid(axis='y', alpha=.2)
ax.legend(loc='upper left', fontsize=9)
for i, r in A.iterrows():
    if r['年份'] in (2018, 2021, 2022, 2026):
        y = r['策略收益'] * 100
        ax.annotate(f"{y:+.1f}%", (i - w / 2, y), xytext=(0, 4 if y >= 0 else -12),
                    textcoords='offset points', ha='center', fontsize=8,
                    color=BLUE, fontweight='bold')
ax.text(.99, .02, '2017 从 11-23 起；2026 截至 10-23', transform=ax.transAxes,
        ha='right', fontsize=8, color=GREY)
plt.tight_layout()
plt.savefig(FIG / '01_年度策略与市场.png')
plt.close()

# 2 环境平均单笔（左：事前；右：事后市场）
left_labels = ['A 牛市上升段急跌', 'B 高位转弱/跌破年线', 'C 熊市反弹',
               'D 熊市下跌', '高波动', '低波动',
               '市场回撤<-20%', '市场回撤>-10%']
left_keys = ['A 牛市上升段急跌', 'B 高位转弱/跌破年线', 'C 熊市反弹',
             'D 熊市下跌', '高波动', '低波动',
             '市场250日回撤<-20%', '市场250日回撤>-10%']
L = P.set_index('分类').loc[left_keys]
R = P.set_index('分类').loc[['持有期市场上涨', '持有期市场下跌']]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                         gridspec_kw={'width_ratios': [3.2, 1.2]})
for ax, dat, labels, title in [
        (axes[0], L, left_labels, '事前可观察的环境'),
        (axes[1], R, ['持有期上涨', '持有期下跌'], '事后：持有期市场涨/跌')]:
    y = np.arange(len(dat))
    colors = [BLUE if v >= 0 else RED for v in dat['平均']]
    ax.barh(y, dat['平均'] * 100, color=colors, alpha=.88)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='k', lw=.8)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_title(title, fontsize=10)
    ax.grid(axis='x', alpha=.2)
    for i, (_, r) in enumerate(dat.iterrows()):
        if r['平均'] >= 0:
            tx, ha = r['平均'] * 100 + 0.5, 'left'
        else:
            tx, ha = 0.5, 'left'
        ax.text(tx, i,
                f"{r['平均']*100:+.1f}%  n={int(r['笔数'])}  胜{r['胜率']*100:.0f}%",
                va='center', ha=ha, fontsize=7.5)
axes[0].invert_yaxis(); axes[1].invert_yaxis()
axes[0].set_xlim(-5, 13); axes[1].set_xlim(-6, 15)
fig.suptitle('单笔平均收益：高波动/深回撤更好；真正决定盈亏的是持有期市场有没有反弹', fontsize=12)
plt.tight_layout()
plt.savefig(FIG / '02_阶段平均单笔.png')
plt.close()

# 3 波动 × 250日涨幅
mat = V.pivot(index='波动', columns='市场250日涨幅', values='平均')
cnt = V.pivot(index='波动', columns='市场250日涨幅', values='笔数')
win = V.pivot(index='波动', columns='市场250日涨幅', values='胜率')
mat = mat.loc[['低波动', '高波动'], ['<0', '0~10%', '>10%']]
cnt = cnt.loc[mat.index, mat.columns]
win = win.loc[mat.index, mat.columns]

fig, ax = plt.subplots(figsize=(7.6, 4.6))
im = ax.imshow(mat.values * 100, cmap='RdYlGn', vmin=-8, vmax=12, aspect='auto')
ax.set_xticks(range(mat.shape[1]))
ax.set_xticklabels(mat.columns)
ax.set_yticks(range(mat.shape[0]))
ax.set_yticklabels(mat.index)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j] * 100
        ax.text(j, i, f"{v:+.1f}%\n n={int(cnt.values[i,j])}\n胜率{win.values[i,j]*100:.0f}%",
                ha='center', va='center', fontsize=9,
                color='white' if abs(v) > 7 else 'black')
ax.set_xlabel('信号日市场 250 日涨幅')
ax.set_ylabel('信号日市场 20 日波动率')
ax.set_title('环境矩阵：最差的是“低波动 + 250日涨幅>10%”格，但主要是 2026 年')
fig.colorbar(im, ax=ax, label='单笔平均收益（%）', shrink=.85)
fig.text(.12, .02, '注：该格 2026 年 46 笔均值 -6.1%，剔除 2026 后 +4.5%。'
                   '不能把这一格单独当成禁用规则。', fontsize=8, color=GREY)
plt.tight_layout(rect=[0, .05, 1, 1])
plt.savefig(FIG / '03_环境矩阵.png')
plt.close()

# 4 2026 指数与信号
m26 = M[M['date'] >= '2026-01-01'].copy().reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(m26['date'], m26['equal_weight_index'], color=BLUE, lw=1.5, label='沪深主板等权指数')
ax.plot(m26['date'], m26['ma250'], color=ORANGE, lw=1.4, ls='--', label='250日均线')
sig = m26[m26['是否信号日'] == 1]
ax.scatter(sig['date'], sig['equal_weight_index'], marker='v', s=24,
           color=RED, alpha=.75, zorder=4, label='信号日')
for a, b, txt, col in [
        ('2026-03-01', '2026-04-30', '3-4月\n第一波', '#f1c40f'),
        ('2026-06-01', '2026-06-30', '6月\n跌破年线', '#e74c3c'),
        ('2026-07-01', '2026-07-31', '7月\n见底', '#27ae60'),
        ('2026-10-01', '2026-10-23', '10月\n再回落', '#8e44ad')]:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), alpha=.07, color=col)
    ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2,
            m26['equal_weight_index'].max() * 1.015, txt, ha='center',
            fontsize=8, color=col)
peak = m26.loc[m26['equal_weight_index'].idxmax()]
low = m26.loc[m26['equal_weight_index'].idxmin()]
ax.annotate(f"2/27 峰值 {peak['equal_weight_index']:.3f}",
            (peak['date'], peak['equal_weight_index']), xytext=(-90, -18),
            textcoords='offset points', fontsize=8, color=GREY)
ax.annotate(f"7/24 低点 {low['equal_weight_index']:.3f}",
            (low['date'], low['equal_weight_index']), xytext=(8, -22),
            textcoords='offset points', fontsize=8, color=GREY)
ax.set_title('2026：2月见顶 → 6月跌破年线 → 7月见底 → 10月再回落')
ax.set_ylabel('等权指数（2026-01-05 = 1）')
ax.set_ylim(m26['equal_weight_index'].min() * .97, m26['equal_weight_index'].max() * 1.06)
ax.grid(alpha=.2)
ax.legend(loc='lower left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.tight_layout()
plt.savefig(FIG / '04_2026指数与信号.png')
plt.close()

# 5 2026 起点 60 日 vs 历史
hist = F.loc[(F['y'] < 2026), 'ret_60'].dropna() * 100
cur = F.loc[F['y'] == 2026, 'ret_60'].dropna() * 100
fig, ax = plt.subplots(figsize=(8.6, 4.8))
bp = ax.boxplot([hist, cur], vert=True, widths=.45, patch_artist=True,
                showfliers=False, medianprops=dict(color='black', lw=1.5))
for patch, col in zip(bp['boxes'], ['#bdc3c7', '#f5b041']):
    patch.set_facecolor(col)
    patch.set_alpha(.85)
rng = np.random.default_rng(7)
ax.scatter(np.ones(len(hist)) + rng.uniform(-.12, .12, len(hist)), hist,
           s=9, color=GREY, alpha=.45, zorder=3)
ax.scatter(np.ones(len(cur)) * 2 + rng.uniform(-.12, .12, len(cur)), cur,
           s=20, color=RED, alpha=.75, zorder=4)
ax.axhline(0, color='k', lw=.8)
ax.set_xticks([1, 2])
ax.set_xticklabels([f'历史 2017-2025（n={len(hist)}）', f'2026（n={len(cur)}）'])
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylabel('起点后 60 个交易日收益')
ax.set_title('2026 起点 60 日：中位 +5.3%，历史 +10.9%；亏损占比 40% vs 15.8%')
ax.set_ylim(-33, 85)
ax.grid(axis='y', alpha=.2)
plt.tight_layout()
plt.savefig(FIG / '05_2026起点60日.png')
plt.close()

# 6 单笔收益 vs 持有期市场
zz = D[D['complete'] == True].dropna(subset=['ret', '持有期市场']).copy()
zz['year'] = zz['signal_date'].dt.year
fig, ax = plt.subplots(figsize=(9, 5.2))
sc = ax.scatter(zz['持有期市场'] * 100, zz['ret'] * 100, c=zz['year'],
                cmap='viridis', s=16, alpha=.65, zorder=3)
x = np.linspace(zz['持有期市场'].min(), zz['持有期市场'].max(), 50)
b, a = np.polyfit(zz['持有期市场'], zz['ret'], 1)
ax.plot(x * 100, (a + b * x) * 100, color=RED, lw=1.8, label=f'回归线 β={b:.2f}')
ax.axhline(0, color='k', lw=.8); ax.axvline(0, color='k', lw=.8)
ax.axhline(20, color=GREY, lw=.8, ls=':')
ax.text(-19.5, 21.2, '止盈上限 +20%', ha='left', fontsize=8, color=GREY)
ax.xaxis.set_major_formatter(PercentFormatter())
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_xlabel('持有期（入场→退出）沪深主板等权市场收益')
ax.set_ylabel('单笔收益')
corr = np.corrcoef(zz['持有期市场'], zz['ret'])[0, 1]
ax.set_title(f'单笔收益 vs 持有期市场：β={b:.2f}，相关系数={corr:.2f}，R²={corr**2:.2f}')
ax.grid(alpha=.2)
ax.legend(loc='upper left', fontsize=9)
fig.colorbar(sc, ax=ax, label='信号年份', shrink=.85)
plt.tight_layout()
plt.savefig(FIG / '06_单笔收益与市场.png')
plt.close()

# 7 2026 起点总收益
C = C.sort_values('signal_date').reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11, 4.8))
colors = [BLUE if v >= 0 else RED for v in C['total_ret']]
ax.bar(C['signal_date'], C['total_ret'] * 100, width=1.6, color=colors, alpha=.85)
ax.axhline(0, color='k', lw=.8)
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylabel('2026 起点至 10-23 总收益')
ax.set_title('2026 年 49 个起点：34 正、13 负、2 未入场；最差 6/4 -28.1%，最好 7/9 +48.9%')
ax.set_ylim(-33, 56)
ax.grid(axis='y', alpha=.2)
worst = C.loc[C['total_ret'].idxmin()]
best = C.loc[C['total_ret'].idxmax()]
for r, col in [(worst, RED), (best, BLUE)]:
    ax.annotate(f"{r['signal_date'].date()}\n{r['total_ret']*100:+.1f}%",
                (r['signal_date'], r['total_ret'] * 100), xytext=(9, 0),
                textcoords='offset points', ha='left', va='center', fontsize=8, color=col)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.tight_layout()
plt.savefig(FIG / '07_2026起点收益.png')
plt.close()

print('charts done')
