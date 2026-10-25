# -*- coding: utf-8 -*-
"""生成不同入场时点回测图表。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUT = '04_图表'
S = pd.read_csv('03_结果/组合汇总_全部时点.csv')
A = pd.read_csv('03_结果/年度收益_全部时点.csv')
G = pd.read_csv('03_结果/信号级汇总_全部信号.csv')
O = pd.read_csv('03_结果/信号集合变化_按档位.csv')
E = pd.read_csv('03_结果/权益曲线_全部时点.csv', parse_dates=['date']).set_index('date')
T = pd.read_csv('03_结果/逐笔_组合_全部时点.csv', parse_dates=['信号日', '入场日'])


def pick(prefix, nmin=1, nmax=15):
    x = S[S.case_id.str.match(f'^{prefix}\\d+$')].copy()
    x['n'] = x.case_id.str.extract(r'(\d+)$').astype(int)
    return x[(x.n >= nmin) & (x.n <= nmax)].sort_values('n')


Oo = pick('O')
Cc = pick('C')
Ll = pick('L', 1, 10)
Hh = pick('H', 1, 10)

# 1 年化 vs 入场档位
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Oo.n, Oo.cagr * 100, 'o-', color='#1f4e79', lw=2.2, label='开盘买入')
ax.plot(Cc.n, Cc.cagr * 100, 's-', color='#e67e22', lw=1.8, label='收盘买入')
ax.plot(Ll.n, Ll.cagr * 100, '^--', color='#2e7d32', lw=1.3, alpha=.8, label='当日最低价（乐观边界）')
ax.plot(Hh.n, Hh.cagr * 100, 'v--', color='#c0392b', lw=1.3, alpha=.8, label='当日最高价（悲观边界）')
base = S[S.case_id == 'O03'].iloc[0]
ax.scatter([3], [base.cagr * 100], s=90, color='#f39c12', zorder=5, edgecolor='k')
ax.annotate(f'基准 第3日开盘\n{base.cagr*100:.1f}%', (3, base.cagr * 100),
            xytext=(3.4, 45), fontsize=9, arrowprops=dict(arrowstyle='->', lw=.8))
ax.axhline(30, color='#f39c12', ls=':', lw=1, label='30% 参考线')
ax.axhline(0, color='k', lw=.8)
ax.set_xlabel('信号后第 N 个交易日入场')
ax.set_ylabel('全期年化（%）')
ax.set_title('入场越晚不一定越好：年化在“第3个交易日开盘”见顶')
ax.set_xticks(range(1, 16)); ax.grid(alpha=.25); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/01_年化_按入场档位.png'); plt.close()

# 2 年度收益热力图
rows = ['O01', 'O02', 'O03', 'O04', 'O05', 'O06', 'O07', 'O08', 'O09', 'O10',
        'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10']
hm = A[A.case_id.isin(rows)].set_index('case_id').loc[rows]
years = [str(y) for y in range(2017, 2027)]
vals = hm[years].values * 100
fig, ax = plt.subplots(figsize=(11, 7.5))
im = ax.imshow(vals, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=100)
ax.set_xticks(range(len(years)))
ax.set_xticklabels(['2017\n(10-19起)'] + years[1:-1] + ['2026\n(至10-23)'], fontsize=8)
ax.set_yticks(range(len(rows)))
labels = [f'{c} 开盘·第{int(c[1:])}日' if c.startswith('O') else f'{c} 收盘·第{int(c[1:])}日' for c in rows]
ax.set_yticklabels(labels, fontsize=8)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        if np.isfinite(v):
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7,
                    color='white' if (v > 65 or v < -15) else 'black')
ax.set_title('年度收益（%）：同一信号，不同入场日，年度路径差别很大')
plt.colorbar(im, ax=ax, label='年度收益（%）', shrink=.8)
plt.tight_layout(); plt.savefig(f'{OUT}/02_年度收益_热力图.png'); plt.close()

# 3 净值与回撤
sel = ['O01', 'O02', 'O03', 'O04', 'O05', 'O10']
colors = ['#95a5a6', '#e67e22', '#1f4e79', '#27ae60', '#8e44ad', '#c0392b']
fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
for c, col in zip(sel, colors):
    ax[0].plot(E.index, E[c], color=col, lw=1.6 if c == 'O03' else 1.1,
               label=f'{c}（第{int(c[1:])}日开盘）')
ax[0].set_yscale('log'); ax[0].set_ylabel('净值（对数，起点=1）')
ax[0].set_title('不同入场档位的净值路径（10 万本金，后复权、2 仓、+20%止盈、15日）')
ax[0].grid(alpha=.25); ax[0].legend(fontsize=8, ncol=3)
dd = E['O03'] / E['O03'].cummax() - 1
ax[1].fill_between(dd.index, dd.values, 0, color='#c0392b', alpha=.55)
ax[1].set_ylabel('基准回撤'); ax[1].yaxis.set_major_formatter(PercentFormatter(1)); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig(f'{OUT}/03_净值_主要档位.png'); plt.close()

# 4 信号级收益
Go = G[G.case_id.str.match(r'^O\d+$')].copy(); Go['n'] = Go.case_id.str.extract(r'(\d+)$').astype(int)
Gc = G[G.case_id.str.match(r'^C\d+$')].copy(); Gc['n'] = Gc.case_id.str.extract(r'(\d+)$').astype(int)
Go = Go[Go.n <= 10].sort_values('n'); Gc = Gc[Gc.n <= 10].sort_values('n')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Go.n, Go['全部_平均'] * 100, 'o-', color='#1f4e79', label='全部信号·开盘买')
ax.plot(Go.n, Go['前二_平均'] * 100, 's-', color='#2e7d32', label='每日前二·开盘买')
ax.plot(Go.n, Go['基准99_平均'] * 100, '^-', color='#f39c12', lw=2, label='基准99笔同信号·开盘买')
ax.plot(Gc.n, Gc['全部_平均'] * 100, 'o--', color='#e67e22', alpha=.8, label='全部信号·收盘买')
ax.set_xlabel('信号后第 N 个交易日入场')
ax.set_ylabel('平均单笔收益（%）')
ax.set_title('信号级：平均单笔收益对 1–5 日并不敏感，基准99笔在第3日最高')
ax.set_xticks(range(1, 11)); ax.grid(alpha=.25); ax.legend(fontsize=8)
ax.axvline(3, color='#999', ls=':', lw=1)
plt.tight_layout(); plt.savefig(f'{OUT}/04_信号级收益_按档位.png'); plt.close()

# 5 信号集合重叠
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                              gridspec_kw={'height_ratios': [1.2, 1]})
ax.bar(O['第N个交易日'], O['平均Jaccard'] * 100, color='#1f4e79', alpha=.78)
for _, r in O.iterrows():
    if r['第N个交易日'] in (1, 3, 5, 10, 15):
        ax.text(r['第N个交易日'], r['平均Jaccard'] * 100 + .3,
                f"{r['平均Jaccard']*100:.1f}%", ha='center', fontsize=8)
ax.set_ylabel('平均 Jaccard（%）')
ax.set_title('在入场日重新扫描，信号点已经大不相同')
ax.set_ylim(0, O['平均Jaccard'].max() * 100 * 1.18)
ax.grid(axis='y', alpha=.25)
ax2.plot(O['第N个交易日'], O['前二存活率'] * 100, 'o-', color='#c0392b', label='前二存活率')
ax2.plot(O['第N个交易日'], O['前二完全重合比例'] * 100, 's--', color='#f39c12', label='前二完全重合')
for _, r in O.iterrows():
    if r['第N个交易日'] in (1, 3, 5, 10, 15):
        ax2.text(r['第N个交易日'], r['前二存活率'] * 100 + .8,
                 f"{r['前二存活率']*100:.1f}%", ha='center', fontsize=8, color='#c0392b')
ax2.set_xlabel('信号后第 N 个交易日')
ax2.set_ylabel('前二存活/重合（%）')
ax2.set_xticks(range(1, 16)); ax2.grid(alpha=.25); ax2.legend(fontsize=8, loc='upper right')
ax2.set_ylim(0, O['前二存活率'].max() * 100 * 1.15)
plt.tight_layout(); plt.savefig(f'{OUT}/05_信号集合重叠.png'); plt.close()

# 6 回撤
fig, ax = plt.subplots(figsize=(10, 4.8))
w = .38
ax.bar(Oo.n - w / 2, Oo.dd * 100, w, color='#1f4e79', label='开盘买入')
ax.bar(Cc.n + w / 2, Cc.dd * 100, w, color='#e67e22', label='收盘买入')
ax.axhline(-20, color='#c0392b', ls=':', lw=1, label='-20% 参考线')
ax.scatter([3 - w / 2], [base.dd * 100], color='#f39c12', s=70, zorder=5, edgecolor='k')
ax.set_xlabel('信号后第 N 个交易日入场')
ax.set_ylabel('最大回撤（%）')
ax.set_title('越晚入场，回撤通常越大；基准 -16.0% 是较小的一端')
ax.set_xticks(range(1, 16)); ax.grid(axis='y', alpha=.25); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/06_回撤_按档位.png'); plt.close()

# 7 第3日四种价格
sel = ['L03', 'O03', 'C03', 'H03']
names = ['最低价\n（乐观）', '开盘价\n（基准）', '收盘价', '最高价\n（悲观）']
vals = [S[S.case_id == c].iloc[0].cagr * 100 for c in sel]
finals = [S[S.case_id == c].iloc[0].final for c in sel]
fig, ax = plt.subplots(figsize=(8, 4.6))
bars = ax.bar(names, vals, color=['#2e7d32', '#1f4e79', '#e67e22', '#c0392b'], alpha=.85)
for b, v, f in zip(bars, vals, finals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f'{v:.1f}%\n期末 {f:.1f}x',
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel('全期年化（%）')
ax.set_title('同一天（第3个交易日）不同成交价：40.0% 只是开盘价的结果')
ax.set_ylim(0, max(vals) * 1.25); ax.grid(axis='y', alpha=.25)
plt.tight_layout(); plt.savefig(f'{OUT}/07_第3日四种价格.png'); plt.close()

print('charts done')
