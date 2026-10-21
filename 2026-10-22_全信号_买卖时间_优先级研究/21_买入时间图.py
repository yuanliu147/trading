# -*- coding: utf-8 -*-
"""买入月份与回测起点敏感性图。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = '/Users/admin/WorkSpace/trading/2026-10-22_全信号_买卖时间_优先级研究'
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False
M = pd.read_csv(os.path.join(DIR, '7015_买入月份.csv'), encoding='utf-8-sig')
S = pd.read_csv(os.path.join(DIR, '7015_起点敏感性.csv'), encoding='utf-8-sig', parse_dates=['起点'])
S = S[S['起点类型'] == '2018年内月份'].copy()

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
fig.subplots_adjust(top=0.82, bottom=0.2, left=0.08, right=0.90, wspace=0.36)

ax = axes[0]
x = np.arange(1, 13)
ax.bar(x, M['全期_平均'], color=['#54A24B' if v > 0 else '#E45756' for v in M['全期_平均']], alpha=.85, label='全期')
ax.plot(x, M['训练2018-2021_平均'], 'o-', color='#4C78A8', lw=1.4, ms=4, label='训练')
ax.plot(x, M['检验2022-2026_平均'], 's--', color='#F58518', lw=1.4, ms=4, label='检验')
ax.axhline(0, color='black', lw=.6, alpha=.4)
ax.set_xticks(x); ax.set_xticklabels([f'{i}月' for i in x])
ax.set_ylabel('平均每笔净盈亏（元）')
ax.set_title('买入月份：全样本差距很大，训练/检验经常反向', loc='left', fontsize=12, weight='bold')
ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper right')
ax.grid(axis='y', alpha=.22)

ax = axes[1]
ax.bar(S['起点'].dt.month, S['年化'] * 100, color='#4C78A8', alpha=.85, label='年化')
ax.set_ylim(0, 40)
ax.set_xticks(range(1, 13)); ax.set_xticklabels([f'{i}月' for i in range(1, 13)])
ax.set_ylabel('年化（%）')
ax2 = ax.twinx()
ax2.plot(S['起点'].dt.month, S['回撤'] * 100, 'o-', color='#E45756', lw=1.4, ms=4, label='回撤')
ax2.set_ylim(-30, 0); ax2.set_yticks([-30, -20, -10, 0]); ax2.set_yticklabels(['-30%', '-20%', '-10%', '0']); ax2.set_ylabel('最大回撤（%）', labelpad=10)
ax.set_title('2018年不同月份开始：结果几乎一样', loc='left', fontsize=12, weight='bold')
lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9, loc='lower right')
for i, v in enumerate(S['年化'] * 100):
    ax.text(i + 1, v + .8, f'{v:.0f}', ha='center', fontsize=8)

fig.suptitle('70/15 子池：买入月份 vs 回测起点，哪个影响大', fontsize=15, y=0.94)
fig.text(0.5, 0.035,
         '左：6,827笔全信号，每笔1万，按买入日所在月份分组；右：同一70/15两仓系统，2018年不同月份开始，终点2026-10-13。'
         '月份差异是市场状态/年份构成，不是稳定日历效应。',
         ha='center', fontsize=8, color='#555555')
plt.savefig(os.path.join(DIR, '7015_买入时间.png'), dpi=200)
plt.close()
print('ok')
