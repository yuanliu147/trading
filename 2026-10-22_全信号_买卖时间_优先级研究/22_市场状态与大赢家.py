# -*- coding: utf-8 -*-
"""70/15子池：市场状态定义、Top交易特征、右尾贡献。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = '/Users/admin/WorkSpace/trading/2026-10-22_全信号_买卖时间_优先级研究'
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

T = pd.read_csv(os.path.join(DIR, '全信号_一万_逐笔.csv'), encoding='utf-8-sig', parse_dates=['首个信号日', '买入日'])
X = T[(T['成交额'] <= 70e6) & (T['股价'] <= 15)].copy()
cols = ['代码', '首个信号日', '买入日', '止损后净盈亏', '止损后收益率', '市场宽度20', '市场宽度60',
        '市场20日收益', '市场波动', '行业宽度', '行业20日收益', '趋势年龄', '真实波幅', '股价', '成交额', '信号日跳空']
X.nlargest(30, '止损后净盈亏')[cols].to_csv(os.path.join(DIR, '市场状态_大赢家.csv'), index=False,
                                            encoding='utf-8-sig', float_format='%.6f', date_format='%Y-%m-%d')

X['组'] = pd.qcut(X['止损后净盈亏'], 5, labels=['最差20%', '较差', '中间', '较好', '最好20%'])
Q = X.groupby('组', observed=True).agg(
    笔数=('代码', 'size'), 平均净盈亏=('止损后净盈亏', 'mean'),
    市场宽度20=('市场宽度20', 'mean'), 市场宽度60=('市场宽度60', 'mean'),
    市场20日收益=('市场20日收益', 'mean'), 市场波动=('市场波动', 'mean'),
    行业宽度=('行业宽度', 'mean'), 行业20日收益=('行业20日收益', 'mean')).reset_index()
Q.to_csv(os.path.join(DIR, '市场状态_分组.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')

thr = X['止损后净盈亏'].quantile(.9)
X['大赢家组'] = np.where(X['止损后净盈亏'] >= thr, '前10%', '后90%')
G = X.groupby('大赢家组').agg(笔数=('代码', 'size'), 平均净盈亏=('止损后净盈亏', 'mean'),
                             市场宽度20=('市场宽度20', 'mean'), 市场宽度60=('市场宽度60', 'mean'),
                             市场20日收益=('市场20日收益', 'mean'), 市场波动=('市场波动', 'mean'),
                             行业宽度=('行业宽度', 'mean'), 行业20日收益=('行业20日收益', 'mean')).reset_index()
G.to_csv(os.path.join(DIR, '市场状态_前10对比.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print(Q.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print(G.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

# 图：右尾贡献 + 市场宽度分布
x = X['止损后净盈亏'].sort_values(ascending=False).values
cum = np.cumsum(x) / x.sum()
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
fig.subplots_adjust(top=0.82, bottom=0.2, left=0.08, right=0.98, wspace=0.28)

ax = axes[0]
ax.plot(np.arange(1, len(x) + 1) / len(x) * 100, cum * 100, color='#4C78A8', lw=1.8)
for p in [1, 5, 10]:
    n = max(1, int(len(x) * p / 100))
    ax.axvline(p, color='#999999', ls='--', lw=.8)
    ax.scatter([p], [cum[n - 1] * 100], color='#E45756', zorder=5, s=20)
    ax.text(p + .4, cum[n - 1] * 100 - 4, f'{p}%: {cum[n-1]*100:.1f}%', fontsize=8.5, color='#E45756')
ax.set_xlabel('按单笔净盈亏从高到低排序的交易占比（%）')
ax.set_ylabel('累计贡献总利润（%）')
ax.set_title('少数大赢家：前10%交易贡献96.5%利润', loc='left', fontsize=12, weight='bold')
ax.axhline(100, color='#777777', ls=':', lw=.8)
ax.set_xlim(0, 100); ax.set_ylim(0, 145)
ax.grid(alpha=.22)
ax.text(.03, .08, '把全部6,827笔按盈亏排序；\n前10%赚约219.8万，后90%合计只赚约7.9万。\n前20%贡献134.5%，因为后80%净亏。',
        transform=ax.transAxes, fontsize=9, color='#333333',
        bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=2))

ax = axes[1]
order = ['最差20%', '较差', '中间', '较好', '最好20%']
data = [X[X['组'] == k]['市场宽度20'].dropna().values for k in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
for patch in bp['boxes']:
    patch.set_facecolor('#9ECAE1'); patch.set_alpha(.85)
for med in bp['medians']:
    med.set_color('#E45756'); med.set_linewidth(1.5)
ax.set_ylabel('买入日市场宽度20（收盘>MA20占比）')
ax.set_title('大赢家所在的市场状态：略强，但重叠很大', loc='left', fontsize=12, weight='bold')
ax.grid(axis='y', alpha=.22)
ax.text(.02, .95, '最好20%组平均宽度0.794，最差20%组0.680；\n不是“高宽度就一定赢”，2020年1月高宽度也连续大亏。',
        transform=ax.transAxes, va='top', fontsize=9, color='#333333')

fig.suptitle('70/15子池：怎么判断“市场状态”和“少数大赢家”', fontsize=15, y=0.94)
fig.text(0.5, 0.035,
         '市场宽度20 = 全A收盘价高于MA20的股票占比，信号日收盘可见；大赢家组为单笔净盈亏前10%。'
         '这是事后分组，不是买入前的选股规则。',
         ha='center', fontsize=8, color='#555555')
plt.savefig(os.path.join(DIR, '市场状态_大赢家.png'), dpi=200)
plt.close()
print('ok')
