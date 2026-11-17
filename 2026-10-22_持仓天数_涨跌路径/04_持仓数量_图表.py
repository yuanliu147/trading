# -*- coding: utf-8 -*-
"""持仓数量与收益的图：全信号 vs 70/15，1/2/3/5/10仓。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator

DIR = '/Users/admin/WorkSpace/trading/2026-10-22_持仓天数_涨跌路径'
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

M = pd.read_csv(os.path.join(DIR, '持仓数量_全信号_组合.csv'), encoding='utf-8-sig')
S = pd.read_csv(os.path.join(DIR, '持仓数量_组合.csv'), encoding='utf-8-sig')

# 等权序列（2仓取等权，不含45%基准）
M = M[M['方案'] != '全信号_2仓45%基准'].copy()
S = S[S['方案'] != '2仓45%基准'].copy()
order = ['1', '2', '3', '5', '10']
M['x'] = M['仓位数'].astype(str)
S['x'] = S['仓位数'].astype(str)
M = M.set_index('x').loc[order].reset_index()
S = S.set_index('x').loc[order].reset_index()
xs = np.arange(len(order))

C1, C2 = '#2e86c1', '#27ae60'
fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))

ax = axes[0, 0]
ax.plot(xs, M['年化'], 'o-', color=C1, lw=2.2, label='全信号（主系统）')
ax.plot(xs, S['年化'], 's--', color=C2, lw=2.2, label='70/15 子池')
ax.axhline(.216, color=C1, ls=':', lw=1.2, alpha=.7)
ax.axhline(.303, color=C2, ls=':', lw=1.2, alpha=.7)
ax.annotate('2仓45%基准 21.6%', xy=(1, .216), xytext=(1.1, .235), color=C1, fontsize=9)
ax.annotate('2仓45%基准 30.3%', xy=(1, .303), xytext=(1.1, .315), color=C2, fontsize=9)
ax.set_title('① 年化收益 vs 同时持有上限', fontsize=12)
ax.set_ylabel('年化')
ax.set_xticks(xs); ax.set_xticklabels(order)
ax.set_xlabel('最多同时持有几只')
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.grid(alpha=.25); ax.legend(fontsize=9)

ax = axes[0, 1]
ax.plot(xs, M['最大回撤'], 'o-', color=C1, lw=2.2, label='全信号')
ax.plot(xs, S['最大回撤'], 's--', color=C2, lw=2.2, label='70/15 子池')
ax.set_title('② 最大回撤 vs 同时持有上限', fontsize=12)
ax.set_ylabel('最大回撤')
ax.set_xticks(xs); ax.set_xticklabels(order)
ax.set_xlabel('最多同时持有几只')
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.grid(alpha=.25); ax.legend(fontsize=9)

ax = axes[1, 0]
ax.plot(xs, M['单笔净均值'], 'o-', color=C1, lw=2.2, label='全信号')
ax.plot(xs, S['单笔净均值'], 's--', color=C2, lw=2.2, label='70/15 子池')
ax.set_title('③ 单笔扣费后平均收益 vs 同时持有上限', fontsize=12)
ax.set_ylabel('单笔净均值')
ax.set_xticks(xs); ax.set_xticklabels(order)
ax.set_xlabel('最多同时持有几只')
ax.set_yticks([.01, .02, .03, .04, .05])
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.grid(alpha=.25); ax.legend(fontsize=9)

ax = axes[1, 1]
ax.plot(xs, np.minimum(M['前五笔净盈亏占比'], 1.0), 'o-', color=C1, lw=2.2, label='全信号')
ax.plot(xs, S['前五笔净盈亏占比'], 's--', color=C2, lw=2.2, label='70/15 子池')
ax.annotate('全信号1仓 125%\n(截断显示)', xy=(0, 1.0), xytext=(0.25, .90),
            color=C1, fontsize=9, arrowprops=dict(arrowstyle='->', color=C1, lw=1))
ax.set_title('④ 前五笔净盈亏 / 总净盈亏（集中度）', fontsize=12)
ax.set_ylabel('前五笔净盈亏占比')
ax.set_xticks(xs); ax.set_xticklabels(order)
ax.set_xlabel('最多同时持有几只')
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.grid(alpha=.25); ax.legend(fontsize=9)

fig.suptitle('收益与持仓数量：不是越多越好，也不是越少越好', fontsize=15, y=.995)
fig.tight_layout(rect=[0, 0, 1, .97])
fig.savefig(os.path.join(DIR, '持仓数量_收益关系.png'), dpi=160)
plt.close(fig)
print('saved 持仓数量_收益关系.png')
