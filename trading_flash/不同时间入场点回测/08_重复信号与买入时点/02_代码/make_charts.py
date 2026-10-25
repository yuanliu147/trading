# -*- coding: utf-8 -*-
"""重复信号与买入时点：图表。"""
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

BLUE, RED, ORANGE, GREY = '#1f4e79', '#c0392b', '#e67e22', '#7f8c8d'

P = pd.read_csv(RES / '位置统计.csv')
P['位置'] = P['位置'].astype(str)
PAIR = pd.read_csv(RES / '首笔vs第二笔_明细.csv', parse_dates=['first_date', 'second_date'])
F = pd.read_csv(RES / '特征分组.csv')
D = pd.read_csv(RES / '入场延迟.csv')
PF = pd.read_csv(RES / '组合入场延迟对照.csv')
C = pd.read_csv(RES / '603778案例.csv', parse_dates=['信号日'])
CUR = pd.read_csv(RES / '当前信号_250日回撤.csv', parse_dates=['信号日', '最高日_hfq', '最高日_实际'])

# 1 位置
fig, ax = plt.subplots(figsize=(8.6, 4.6))
cats = ['1', '2', '3', '4+']
x = np.arange(len(cats)); w = .36
for i, (sample, col) in enumerate([('前二', BLUE), ('全部信号', '#95a5a6')]):
    z = P[P['样本'] == sample].set_index('位置').reindex(cats)
    ax.bar(x + (i - .5) * w, z['平均'] * 100, w, color=col, label=sample)
    for j, (_, r) in enumerate(z.iterrows()):
        if pd.notna(r['平均']):
            ax.text(j + (i - .5) * w, r['平均'] * 100 + .15,
                    f"{r['平均']*100:+.1f}%\nn={int(r['笔数'])}",
                    ha='center', va='bottom', fontsize=7.5, color=col)
ax.axhline(0, color='k', lw=.8)
ax.set_xticks(x); ax.set_xticklabels([f'第{c}次信号' if c != '4+' else '第4次及以后' for c in cats])
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylabel('单笔平均收益')
ax.set_title('第一次信号通常不差；等到第 4 次及以后，收益明显下降')
ax.set_ylim(0, 8.4)
ax.grid(axis='y', alpha=.2); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(FIG / '01_重复位置_收益.png'); plt.close()

# 2 首笔 vs 第二笔
fig, ax = plt.subplots(figsize=(6.6, 5.4))
col = np.where(PAIR['y'] == 2026, RED, BLUE)
ax.scatter(PAIR['first'] * 100, PAIR['second'] * 100, c=col, s=26, alpha=.75, zorder=3)
lim = [-35, 28]
ax.plot(lim, lim, color=GREY, ls='--', lw=1, label='45°线（两笔相同）')
ax.axhline(0, color='k', lw=.8); ax.axvline(0, color='k', lw=.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.xaxis.set_major_formatter(PercentFormatter()); ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_xlabel('第一次信号买（T+3）收益'); ax.set_ylabel('第二次信号买（T+3）收益')
ax.set_title(f"等第二信号：均值差 -0.19pp，第二次更好仅 37%（n={len(PAIR)}）")
ax.grid(alpha=.2)
ax.text(.03, .97, f"首笔均值 {PAIR['first'].mean()*100:+.1f}%\n第二笔均值 {PAIR['second'].mean()*100:+.1f}%\n相关 {PAIR[['first','second']].corr().iloc[0,1]:.2f}\n红点=2026",
        transform=ax.transAxes, va='top', fontsize=8.5,
        bbox=dict(boxstyle='round', fc='white', ec='#cccccc', alpha=.9))
ax.scatter([PAIR['first'].mean()*100], [PAIR['second'].mean()*100], marker='*', s=180,
           color=ORANGE, edgecolor='white', zorder=4, label='均值')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig(FIG / '02_首笔vs第二笔.png'); plt.close()

# 3 特征
LEFT = [
    ('前一日上涨', '买入日前一天涨跌', '前一日上涨'),
    ('前一日下跌', '买入日前一天涨跌', '前一日下跌'),
    ('量比 Q1（低）', '买入日前一天量比(20日)', None),
    ('量比 Q5（高）', '买入日前一天量比(20日)', None),
    ('MACD柱为正', '买入日前一天MACD柱', 'MACD柱为正'),
    ('MACD柱为负', '买入日前一天MACD柱', 'MACD柱为负'),
    ('前一日金叉', '买入日前一天MACD金叉', '前一日金叉'),
    ('前一日非金叉', '买入日前一天MACD金叉', '前一日非金叉'),
    ('站上MA20', '前一日收盘 vs MA20', '站上MA20'),
    ('MA20下方', '前一日收盘 vs MA20', 'MA20下方'),
]
RIGHT = [
    ('同向：个股涨+市场涨', '个股涨跌 × 市场涨跌', '个股涨+市场涨'),
    ('同向：个股跌+市场跌', '个股涨跌 × 市场涨跌', '个股跌+市场跌'),
    ('背离：个股涨+市场跌', '个股涨跌 × 市场涨跌', '个股涨+市场跌'),
    ('背离：个股跌+市场涨', '个股涨跌 × 市场涨跌', '个股跌+市场涨'),
]
FT = F[F['样本'] == '前二']


def pick(feature, group):
    z = FT[(FT['特征'] == feature) & (FT['分组'] == group)]
    return z.iloc[0] if len(z) else None


def draw(ax, items, title):
    vals, labs, ns = [], [], []
    for lab, feat, grp in items:
        if grp is None:
            z = FT[FT['特征'] == feat]
            z = z[z['分组'].str.startswith('Q1')] if lab.startswith('Q1') else z[z['分组'].str.startswith('Q5')]
        else:
            z = FT[(FT['特征'] == feat) & (FT['分组'] == grp)]
        if len(z) == 0:
            continue
        r = z.iloc[0]
        vals.append(r['平均'] * 100); labs.append(lab); ns.append(int(r['笔数']))
    colors = [BLUE if v >= 0 else RED for v in vals]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, alpha=.85)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis(); ax.axvline(0, color='k', lw=.8)
    ax.xaxis.set_major_formatter(PercentFormatter())
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(v + (.15 if v >= 0 else -.15), i, f"{v:+.1f}%  n={n}",
                va='center', ha='left' if v >= 0 else 'right', fontsize=7)
    ax.set_title(title, fontsize=10); ax.grid(axis='x', alpha=.2)
    ax.set_xlim(-2, 9)


fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={'width_ratios': [2.2, 1.2]})
draw(axes[0], LEFT, '前一日单变量')
draw(axes[1], RIGHT, '个股 × 市场（前一日）')
fig.suptitle('前一日 K 线/量能/MACD：没有稳定确认信号；“个股与市场同向”才是较稳的改善', fontsize=12)
plt.tight_layout(); plt.savefig(FIG / '03_前一日特征.png'); plt.close()

# 4 入场延迟
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for sample, col, marker in [('前二全部', BLUE, 'o'), ('前二首笔', '#95a5a6', 's'), ('前二第二笔', ORANGE, '^')]:
    z = D[D['样本'] == sample].sort_values('N')
    axes[0].plot(z['N'], z['平均'] * 100, marker=marker, color=col, label=sample)
axes[0].axhline(0, color='k', lw=.8)
axes[0].set_xticks([1, 2, 3, 4, 5]); axes[0].set_xlabel('信号后第 N 个交易日开盘')
axes[0].yaxis.set_major_formatter(PercentFormatter()); axes[0].set_ylabel('单笔平均收益')
axes[0].set_title('单笔均值：N=1..5 差异很小', fontsize=10)
axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)
axes[1].bar(PF['N'], PF['组合年化'] * 100, color=[GREY, GREY, BLUE, GREY, GREY])
for _, r in PF.iterrows():
    axes[1].text(r['N'], r['组合年化'] * 100 + .8, f"{r['组合年化']*100:.1f}%\nDD {r['最大回撤']*100:.1f}%",
                 ha='center', fontsize=7.5)
axes[1].set_xticks([1, 2, 3, 4, 5]); axes[1].set_xlabel('信号后第 N 个交易日开盘')
axes[1].yaxis.set_major_formatter(PercentFormatter()); axes[1].set_ylabel('组合年化（母策略）')
axes[1].set_title('组合年化：N=3 最高 40.0%，回撤也较小', fontsize=10)
axes[1].grid(axis='y', alpha=.2); axes[1].set_ylim(0, 48)
fig.suptitle('T+3 是实测最优：单笔均值再等一天略高，但组合年化和回撤在 N=3 最好', fontsize=12)
plt.tight_layout(); plt.savefig(FIG / '04_入场延迟.png'); plt.close()

# 5 603778
C = C.sort_values('信号日').reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
cols = [BLUE if (r['是否前二'] and pd.notna(r['收益'])) else (ORANGE if pd.notna(r['收益']) else GREY)
        for _, r in C.iterrows()]
vals = [r['收益'] * 100 if pd.notna(r['收益']) else 0 for _, r in C.iterrows()]
xs = np.arange(len(C))
ax.bar(xs, vals, width=.62, color=cols, alpha=.85)
ax.axhline(0, color='k', lw=.8)
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylabel('T+3 买入、15 日持有收益')
ax.set_title('603778：2026 年 10 次信号；6-7 月初两次失败，7 月中下旬连续止盈，10 月又回到 -78% 回撤')
for i, r in C.iterrows():
    if pd.isna(r['收益']):
        ax.text(i, 1, '待入场', rotation=90, ha='center', va='bottom', fontsize=7, color=GREY)
    else:
        ax.text(i, r['收益'] * 100 + (1 if r['收益'] >= 0 else -1),
                f"{r['收益']*100:+.0f}%", ha='center', va='bottom' if r['收益'] >= 0 else 'top',
                fontsize=7.5)
ax.set_xticks(xs)
ax.set_xticklabels([d.strftime('%m-%d') for d in C['信号日']], rotation=45, ha='right', fontsize=8)
ax.grid(axis='y', alpha=.2)
plt.tight_layout(); plt.savefig(FIG / '05_603778案例.png'); plt.close()

# 6 当前回撤
CUR = CUR.sort_values(['信号日', '排名']).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.2, 4.6))
labels = [f"{r['代码']}\n{r['信号日'].strftime('%m-%d')} 第{int(r['排名'])}" for _, r in CUR.iterrows()]
vals = CUR['回撤_hfq'] * 100
cols = [BLUE if r['排名'] <= 2 else '#95a5a6' for _, r in CUR.iterrows()]
y = np.arange(len(CUR))
ax.barh(y, vals, color=cols, alpha=.88)
ax.axvline(-45, color=RED, ls='--', lw=1.2, label='信号阈值 -45%')
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(PercentFormatter())
for i, r in CUR.iterrows():
    ax.text(r['回撤_hfq'] * 100 - .5, i, f"{r['回撤_hfq']*100:.2f}%", va='center', ha='right', fontsize=8)
ax.set_xlim(-90, 0)
ax.set_xlabel('信号日收盘较 250 日最高价（后复权）')
ax.set_title('当前候选的 250 日回撤：603778 -78.61%，605376 -71.30%，002969 -70.68%')
ax.legend(fontsize=8, loc='lower left')
ax.grid(axis='x', alpha=.2)
plt.tight_layout(); plt.savefig(FIG / '06_当前信号回撤.png'); plt.close()

print('charts done')
