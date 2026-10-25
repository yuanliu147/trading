# -*- coding: utf-8 -*-
"""信号日回撤深度与收益：图表。"""
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

BLUE, RED, ORANGE, GREY = '#1f4e79', '#c0392b', '#e67e22', '#7f8c8d'
DEEP = ['≤-75%', '-75~-70%', '-70~-65%']

B = pd.read_csv(RES / '前二深度分档.csv')
Y = pd.read_csv(RES / '年度深度对比.csv')
P = pd.read_csv(RES / '同日排名配对.csv', parse_dates=['信号日'])
TR = pd.read_csv(RES / '前二逐笔_深度.csv', parse_dates=['信号日'])
M = pd.read_csv(RES / '2026深度分档.csv')

# 1 深度分档：收益 + 胜率
cats = B['深度档'].tolist()
x = np.arange(len(cats))
colors = [RED if c in DEEP else '#8fa9bf' for c in cats]
fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True,
                         gridspec_kw={'height_ratios': [3, 2], 'hspace': .12})
ax = axes[0]
ax.bar(x, B['平均收益'], .62, color=colors)
for i, (_, r) in enumerate(B.iterrows()):
    ax.text(i, r['平均收益'] + .45, f"{r['平均收益']:+.1f}%\nn={int(r['笔数'])}",
            ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, 25)
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylabel('单笔平均收益')
ax.set_title('信号日跌得越深，后续 15 日反弹越强（前二 450 笔）', pad=10)
ax.axvline(2.5, color=GREY, ls='--', lw=1)
ax.text(1, 23, '深组 ≤-65%', ha='center', color=RED, fontsize=9, fontweight='bold')
ax.text(4.5, 23, '浅组 >-65%', ha='center', color=BLUE, fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=.2)

ax = axes[1]
ax.bar(x, B['胜率'], .62, color=colors)
for i, (_, r) in enumerate(B.iterrows()):
    ax.text(i, r['胜率'] + 1.5, f"{r['胜率']:.0f}%", ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylabel('胜率')
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9)
ax.axvline(2.5, color=GREY, ls='--', lw=1)
ax.grid(axis='y', alpha=.2)
plt.savefig(FIG / '01_深度分档_收益与胜率.png', bbox_inches='tight'); plt.close()

# 2 年度：深 vs 浅
Z = Y[Y['样本'] == '前二'].copy()
years = sorted(Z['年份'].tolist())
x = np.arange(len(years))
w = .36
deep = Z.set_index('年份').reindex(years)
fig, ax = plt.subplots(figsize=(10.2, 4.9))
ax.bar(x - w / 2, deep['深组平均'].fillna(0), w, color=RED, label='深组 ≤-65%')
ax.bar(x + w / 2, deep['浅组平均'].fillna(0), w, color='#8fa9bf', label='浅组 >-65%')
for i, y in enumerate(years):
    r = deep.loc[y]
    d = r['深组平均']
    s = r['浅组平均']
    if pd.isna(d):
        ax.text(i - w / 2, .2, '无深组', ha='center', va='bottom', fontsize=7, color=GREY)
    else:
        ax.text(i - w / 2, max(d, 0) + .4, f"{d:+.1f}", ha='center', va='bottom', fontsize=7.5, color=RED)
    ax.text(i + w / 2, max(s, 0) + .4, f"{s:+.1f}", ha='center', va='bottom', fontsize=7.5, color=BLUE)
    if pd.notna(r['平均差(百分点)']):
        ax.text(i, max(d, s) + 2.6, f"{r['平均差(百分点)']:+.1f}pp", ha='center', va='bottom',
                fontsize=7.5, color='k')
ax.axhline(0, color='k', lw=.8)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylabel('单笔平均收益')
ax.set_ylim(-6, 24)
ax.set_title('前二：深组在 8 个可比年份全部胜出；2026 年 +8.9% vs -3.4%')
ax.legend(fontsize=8.5, loc='upper left')
ax.grid(axis='y', alpha=.2)
plt.tight_layout(); plt.savefig(FIG / '02_年度深度对比.png'); plt.close()

# 3 同日排名1 vs 排名2
fig, ax = plt.subplots(figsize=(6.6, 5.6))
col = np.where(P['年份'] == 2026, RED, BLUE)
ax.scatter(P['排名1收益'], P['排名2收益'], c=col, s=24, alpha=.75, zorder=3)
lim = [-35, 33]
ax.plot(lim, lim, color=GREY, ls='--', lw=1, label='45°线（两只相同）')
ax.axhline(0, color='k', lw=.8); ax.axvline(0, color='k', lw=.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.xaxis.set_major_formatter(PercentFormatter()); ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_xlabel('排名1（跌更深）收益')
ax.set_ylabel('排名2收益')
ax.set_title('同一天：排名1 +6.3% vs 排名2 +4.3%，差 +2.0pp', pad=10)
ax.text(.03, .97,
        f"n=204 个可配对信号日\n排名1更好 {52.0:.0f}% 的天数\n配对 t 检验 p=0.026\n红点=2026",
        transform=ax.transAxes, va='top', fontsize=8.5,
        bbox=dict(boxstyle='round', fc='white', ec='#cccccc', alpha=.9))
ax.scatter([P['排名1收益'].mean()], [P['排名2收益'].mean()], marker='*', s=180,
           color=ORANGE, edgecolor='white', zorder=4, label='均值')
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=.2)
plt.tight_layout(); plt.savefig(FIG / '03_同日排名对比.png'); plt.close()

# 4 深度 vs 收益散点 + 分档均值
fig, ax = plt.subplots(figsize=(9.2, 5.2))
col = np.where(TR['年份'] == 2026, RED, BLUE)
ax.scatter(TR['回撤250'], TR['收益'], c=col, s=18, alpha=.45, zorder=2)
order = ['≤-75%', '-75~-70%', '-70~-65%', '-65~-60%', '-60~-55%', '-55~-50%', '-50~-45%']
g = TR.groupby('深度档', observed=True).agg(x=('回撤250', 'mean'), y=('收益', 'mean'),
                                            n=('收益', 'size')).reindex(order)
ax.plot(g['x'], g['y'], color=ORANGE, lw=1.8, marker='o', ms=6, zorder=4, label='分档均值')
for _, r in g.iterrows():
    ax.annotate(f"n={int(r['n'])}", (r['x'], r['y']), textcoords='offset points',
                xytext=(0, 9), ha='center', fontsize=7, color=ORANGE,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=.85))
ax.axvline(-65, color=GREY, ls='--', lw=1)
ax.axhline(0, color='k', lw=.8)
ax.set_xlim(-86, -43); ax.set_ylim(-36, 34)
ax.xaxis.set_major_formatter(PercentFormatter()); ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_xlabel('信号日较 250 日最高价回撤（后复权）')
ax.set_ylabel('单笔收益')
ax.set_title('回撤每深 10pp，单笔平均多 3.9pp（年度固定效应，t=5.0）', pad=10)
ax.text(.02, .03, '蓝点=非2026  红点=2026  橙线=分档均值', transform=ax.transAxes,
        fontsize=8, color=GREY)
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=.2)
plt.tight_layout(); plt.savefig(FIG / '04_深度与收益.png'); plt.close()

# 5 2026 六月 vs 七月
Z = M[M['月份'].isin(['2026-06', '2026-07'])].copy()
fig, ax = plt.subplots(figsize=(7.2, 4.6))
x = np.arange(len(Z)); w = .34
ax.bar(x - w / 2, Z['深组平均'], w, color=RED, label='深组 ≤-65%')
ax.bar(x + w / 2, Z['浅组平均'], w, color='#8fa9bf', label='浅组 >-65%')
for i, (_, r) in enumerate(Z.iterrows()):
    ax.text(i - w / 2, r['深组平均'] + .35, f"{r['深组平均']:+.1f}%\nn={int(r['深组笔数'])}",
            ha='center', va='bottom', fontsize=8)
    ax.text(i + w / 2, r['浅组平均'] - .35, f"{r['浅组平均']:+.1f}%\nn={int(r['浅组笔数'])}",
            ha='center', va='top', fontsize=8)
ax.axhline(0, color='k', lw=.8)
ax.set_xticks(x)
ax.set_xticklabels(['2026 年 6 月', '2026 年 7 月'])
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylabel('单笔平均收益')
ax.set_ylim(-13, 16)
ax.set_title('2026 六月深组也只打平，七月才反弹：深度不能替代市场转暖')
ax.legend(fontsize=8.5, loc='upper left')
ax.grid(axis='y', alpha=.2)
plt.tight_layout(); plt.savefig(FIG / '05_2026六月七月.png'); plt.close()

print('charts done')
