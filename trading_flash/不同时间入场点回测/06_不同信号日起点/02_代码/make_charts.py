# -*- coding: utf-8 -*-
"""不同信号日起点：图表。"""
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
plt.rcParams['figure.dpi'] = 150

C = pd.read_csv(RES / '起点汇总.csv', parse_dates=['signal_date'])
P = pd.read_csv(RES / '相邻信号日.csv', parse_dates=['date_a', 'date_b'])
B = pd.read_csv(RES / '逐日首笔.csv', parse_dates=['signal_date'])
C['year'] = C.signal_date.dt.year

# 1 起点年化分布
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
for ax, col, label in [(axes[0], 'ret_3年', '满3年（146个起点）'),
                       (axes[1], 'ret_5年', '满5年（101个起点）')]:
    s = C.loc[C[col].notna(), 'cagr'].dropna() * 100
    ax.hist(s, bins=12, color='#1f4e79', alpha=.82, edgecolor='white')
    ax.axvline(s.median(), color='#e67e22', lw=2,
               label=f'中位 {s.median():.1f}%')
    ax.axvline(40.0, color='#c0392b', ls='--', lw=1.5,
               label='母报告 40.0%')
    ax.set_title(label)
    ax.set_xlabel('全期年化（%）')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
axes[0].set_ylabel('起点数')
fig.suptitle('不同信号日起点：长期年化多在 31%–43%，3年样本最高 51.3%', fontsize=12)
plt.tight_layout()
plt.savefig(FIG / '01_起点年化_分布.png')
plt.close()

# 2 首年收益 vs 起点日期
s = C[C['ret_1年'].notna()].sort_values('signal_date')
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(s.signal_date, s['ret_1年'] * 100, '-', color='#bdc3c7', lw=.8, zorder=1)
sc = ax.scatter(s.signal_date, s['ret_1年'] * 100, c=s.year, cmap='viridis',
                s=28, zorder=3)
ax.axhline(0, color='k', lw=.8)
ax.axhline(s['ret_1年'].median() * 100, color='#e67e22', ls='--', lw=1.2,
           label=f"中位 {s['ret_1年'].median()*100:.1f}%")
lo = s.loc[s['ret_1年'].idxmin()]
hi = s.loc[s['ret_1年'].idxmax()]
ax.annotate(f"{lo.signal_date.date()}\n{lo['ret_1年']*100:.1f}%",
            (lo.signal_date, lo['ret_1年'] * 100), xytext=(15, 18),
            textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle='->', lw=.7))
ax.annotate(f"{hi.signal_date.date()}\n{hi['ret_1年']*100:.1f}%",
            (hi.signal_date, hi['ret_1年'] * 100), xytext=(-60, -28),
            textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle='->', lw=.7))
ax.set_ylim(-5, 150)
ax.set_ylabel('起点后 252 个交易日收益（%）')
ax.set_title('同一个策略、不同的第一个信号日：首年收益 1.3%–138.6%')
ax.grid(alpha=.2)
ax.legend(loc='upper left', fontsize=8)
plt.colorbar(sc, ax=ax, label='起点年份', shrink=.8)
plt.tight_layout()
plt.savefig(FIG / '02_首年收益_按起点.png')
plt.close()

# 3 五年期末倍数 vs 起点日期
s5 = C[C['ret_5年'].notna()].sort_values('signal_date')
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(s5.signal_date, 1 + s5['ret_5年'], 'o-', color='#1f4e79',
        lw=1.2, ms=4)
ax.axhline((1 + s5['ret_5年']).median(), color='#e67e22', ls='--', lw=1.2,
           label=f"中位 {(1+s5['ret_5年']).median():.2f} 倍")
ax.axhline(1, color='k', lw=.8)
ax.set_ylabel('起点后5年期末倍数')
ax.set_title('起点不同，5年终值 3.28–6.68 倍（不是只有“40%”一条线）')
ax.grid(alpha=.2)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / '03_五年期末倍数_按起点.png')
plt.close()

# 4 相邻信号日 A vs B：首年收益
X = P[(P.gap_td == 1) & P.both_ge2 & P.y1_a.notna() & P.y1_b.notna()].copy()
fig, ax = plt.subplots(figsize=(7.2, 6.6))
for flag, lab, col, mk in [(False, '前二不同', '#e67e22', 'o'),
                           (True, '前二完全相同', '#1f4e79', 's')]:
    q = X[X.same_top == flag]
    ax.scatter(q.y1_a * 100, q.y1_b * 100, s=30, color=col, marker=mk,
               alpha=.75, label=f'{lab}（{len(q)}对）', edgecolor='white', lw=.4)
lim = max(X.y1_a.max(), X.y1_b.max()) * 100 + 8
ax.plot([0, lim], [0, lim], color='k', ls='--', lw=1, label='45°线')
for a, b, txt in [('2019-08-06', '2019-08-07', 'A 62% / B 33%'),
                  ('2024-06-11', '2024-06-12', 'A 49% / B 76%')]:
    r = X[(X.date_a == a) & (X.date_b == b)]
    if len(r):
        r = r.iloc[0]
        ax.annotate(txt, (r.y1_a * 100, r.y1_b * 100), xytext=(8, -16),
                    textcoords='offset points', fontsize=8,
                    arrowprops=dict(arrowstyle='->', lw=.7))
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('交易员 A：m 日信号，首年收益（%）')
ax.set_ylabel('交易员 B：m+1 日信号，首年收益（%）')
ax.set_title('只差一个交易日，首年收益可差 20–30 个百分点')
ax.grid(alpha=.2); ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig(FIG / '04_相邻信号日_首年对比.png')
plt.close()

# 5 首笔批次收益分布
s = B[B.batch_complete].batch_ret.dropna() * 100
fig, ax = plt.subplots(figsize=(9, 4.8))
n, bins, patches = ax.hist(s, bins=24, edgecolor='white')
for p, left in zip(patches, bins[:-1]):
    p.set_facecolor('#c0392b' if left < 0 else '#2e7d32')
    p.set_alpha(.8)
ax.axvline(s.median(), color='#1f4e79', lw=2, label=f'中位 {s.median():.1f}%')
ax.axvline(s.mean(), color='#e67e22', lw=1.5, ls='--', label=f'均值 {s.mean():.1f}%')
ax.axvline(0, color='k', lw=.8)
ax.set_xlabel('首笔批次收益（%）；两只各 50%，仅一只则半仓')
ax.set_ylabel('信号日数')
ax.set_title(f'每个信号日的第一笔：中位 {s.median():.1f}%，{(s<0).mean()*100:.0f}% 为负')
ax.grid(alpha=.2, axis='y'); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / '05_首笔批次收益_分布.png')
plt.close()

# 6 信号日聚集
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(C.signal_date, C.n_signals, width=12, color='#1f4e79', alpha=.8)
ax.set_yscale('log')
ax.set_ylabel('当日满足条件的股票数（对数）')
ax.set_title('信号日高度聚集：连续两天都有信号时，第二天的前二往往是另一批')
for d, txt in [('2018-02-06', '2018-02'), ('2020-02-04', '2020-02'),
               ('2022-04-26', '2022-04'), ('2024-02-05', '2024-02'),
               ('2026-10-23', '2026-10')]:
    d = pd.Timestamp(d)
    q = C.iloc[(C.signal_date - d).abs().argsort()[:1]]
    if len(q):
        ax.annotate(txt, (q.signal_date.iloc[0], q.n_signals.iloc[0]),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=8)
ax.grid(alpha=.2, axis='y')
plt.tight_layout()
plt.savefig(FIG / '06_信号日聚集.png')
plt.close()

# 7 前二重合 vs 首年差
X['abs_y1_diff'] = X.y1_diff.abs() * 100
fig, ax = plt.subplots(figsize=(7.2, 4.8))
groups = [X.loc[X.overlap == k, 'y1_diff'] * 100 for k in [0, 1, 2]]
labels = [f'重合{k}只\n(n={len(g)})' for k, g in zip([0, 1, 2], groups)]
bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=.55,
                showfliers=False)
for patch, col in zip(bp['boxes'], ['#e67e22', '#f1c40f', '#1f4e79']):
    patch.set_facecolor(col); patch.set_alpha(.7)
rng = np.random.default_rng(7)
for k, g in zip([0, 1, 2], groups):
    ax.scatter(np.full(len(g), k + 1) + rng.uniform(-.12, .12, len(g)),
               g, s=14, color='#2c3e50', alpha=.55, zorder=3)
ax.axhline(0, color='k', lw=.8)
ax.set_ylabel('B 首年收益 − A 首年收益（百分点）')
ax.set_title('前二重合越少，A/B 首年差异越大')
ax.grid(alpha=.2, axis='y')
plt.tight_layout()
plt.savefig(FIG / '07_前二重合_首年差.png')
plt.close()

print('charts done', len(list(FIG.glob('*.png'))))
