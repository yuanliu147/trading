# -*- coding: utf-8 -*-
"""图9：原25.8%与回撤优化候选的权益/回撤对比。"""
import os, pandas as pd, numpy as np, pickle
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False
BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'; R = os.path.join(BASE, '03_结果')
files = {
 '旧25.8%（-36.4%）': ('回撤优化_权益_旧25.8%.csv', '#E45756', 1.4),
 '回撤优先 20.2%（-25.6%）': ('回撤优化_权益_回撤优先_42.5_波动20.csv', '#4C78A8', 1.5),
 '收益优先 21.6%（-26.4%）': ('回撤优化_权益_收益优先_45_波动20.csv', '#F58518', 1.8),
 '均衡 18.8%（-26.8%）': ('回撤优化_权益_均衡_45_波动20_分批20.csv', '#54A24B', 1.4),
 '严格3%风险 8.4%（-14.9%）': ('回撤优化_权益_严格3%风险_止损20.csv', '#7B61A8', 1.4),
}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.0), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1.15]})
fig.subplots_adjust(top=0.92, bottom=0.18, left=0.08, right=0.98, hspace=0.08)
last = None
for name, (f, c, lw) in files.items():
    E = pd.read_csv(os.path.join(R, f), parse_dates=['date']).set_index('date')
    ax1.plot(E.index, E.equity / 10000, label=name, color=c, lw=lw)
    ax2.plot(E.index, E.equity / E.equity.cummax() - 1, color=c, lw=1)
    last = E
P = pickle.load(open(os.path.join(BASE, '01_数据/全A面板_2016_2026_含退市.pkl'), 'rb'))
bm = pd.Series(P['market_index'], index=P['meta']['dates']).reindex(last.index).ffill()
bm = bm / bm.iloc[0] * 2
ax1.plot(bm.index, bm.values, color='#999999', lw=1, alpha=.7, label='全部股票等权（起点2万）')
ax1.set_ylabel('账户权益（万元）')
fig.suptitle('2万元：25.8%回测的审计与回撤优化（整手、最低佣金、t+1开盘）',
             fontsize=16, y=0.975)
fig.text(0.5, 0.045, '口径：收益优先 21.6% 为 2018 年起连续资金；2021 年末清仓再起算 17.3%，2022 年新账户 18.5%',
         ha='center', fontsize=8.5, color='#555555')
ax1.legend(frameon=False, fontsize=9, loc='upper left')
ax1.grid(alpha=.22)
ax1.axhline(2, color='black', lw=.6, alpha=.3)
ax2.axhline(0, color='black', lw=.6, alpha=.3)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y:.0%}'))
ax2.set_ylabel('回撤'); ax2.grid(alpha=.22)
ax2.set_xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(BASE, '04_图表/图9_回撤优化对比.png'), dpi=200)
plt.close()
print('ok')
