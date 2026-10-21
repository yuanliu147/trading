# -*- coding: utf-8 -*-
"""全信号1万元逐笔的盈亏分布与年度平均。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = '/Users/admin/WorkSpace/trading/2026-10-22_全信号_买卖时间_优先级研究'
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

T = pd.read_csv(os.path.join(DIR, '全信号_一万_逐笔.csv'), encoding='utf-8-sig')
Y = pd.read_csv(os.path.join(DIR, '全信号_一万_年度.csv'), encoding='utf-8-sig')

x = T['止损后净盈亏'].dropna().values
lo, hi = np.percentile(x, [1, 99])
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
fig.subplots_adjust(top=0.84, bottom=0.19, left=0.08, right=0.98, wspace=0.24)

ax = axes[0]
x_plot = x[(x >= lo) & (x <= hi)]
ax.hist(x_plot, bins=80, color='#4C78A8', alpha=.85)
ax.axvline(np.median(x), color='#F58518', lw=1.6, label=f'中位 {np.median(x):,.0f} 元')
ax.axvline(x.mean(), color='#E45756', lw=1.6, label=f'平均 {x.mean():,.0f} 元')
ax.set_xlabel('单笔净盈亏（元，1万元名义本金）')
ax.set_ylabel('笔数')
ax.set_title('4万笔信号：多数小赚小亏，少数大右尾', loc='left', fontsize=12, weight='bold')
ax.legend(frameon=False, fontsize=9)
ax.grid(axis='y', alpha=.22)
ax.text(0.02, 0.95, f"胜率 {(x>0).mean():.1%}　正收益 {int((x>0).sum()):,} / {len(x):,}\n"
                    f"前1%交易贡献 {np.sort(x)[-max(1,int(len(x)*.01)):].sum()/x.sum():.1%} 总利润\n显示1%–99%，剔除{len(x)-len(x_plot)}笔极端值",
        transform=ax.transAxes, va='top', fontsize=9, color='#333333')

ax = axes[1]
colors = ['#E45756' if v < 0 else '#54A24B' for v in Y['止损后平均']]
ax.bar(Y['年份'].astype(str), Y['止损后平均'], color=colors)
ax.axhline(0, color='black', lw=.6, alpha=.4)
ax.set_ylabel('平均每笔净盈亏（元）')
ax.set_title('全信号平均：2022、2023 是负的', loc='left', fontsize=12, weight='bold')
ax.grid(axis='y', alpha=.22)
ax.set_ylim(-650, 450)
for i, v in enumerate(Y['止损后平均']):
    ax.text(i, v + (-18 if v < 0 else 15), f'{v:,.0f}', ha='center', fontsize=8.5,
            va='top' if v < 0 else 'bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=1))

fig.suptitle('全部信号、每笔1万元、无仓位上限：盈利到底长什么样', fontsize=15, y=0.94)
fig.text(0.5, 0.035,
         '口径：40,279个可执行信号段，每段首日次日开盘买、20个交易日收盘卖；-20%止损；单边0.15%、最低5元；'
         '每笔1万元名义本金、股数可为小数、不限制同时持仓。不是2万账户的实际收益。',
         ha='center', fontsize=8, color='#555555')
plt.savefig(os.path.join(DIR, '全信号_一万_分布.png'), dpi=200)
plt.close()
print('ok')
