# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
BASE = '/Users/admin/WorkSpace/trading/2026-10-22_当前阶段_情绪同态_50%策略'
OUT = os.path.join(BASE, '04_图表')
os.makedirs(OUT, exist_ok=True)
f = pd.read_csv(os.path.join(BASE, '01_数据/市场日频_2016_2026.csv'), parse_dates=['date']).set_index('date')
eq = pd.read_csv(os.path.join(BASE, '03_结果/核心2仓50_权益.csv'), parse_dates=['date']).set_index('date')
ev = pd.read_csv(os.path.join(BASE, '03_结果/当前同态_事件表.csv'), parse_dates=['date']).set_index('date')
phase = pd.read_csv(os.path.join(BASE, '03_结果/核心2仓50_阶段.csv'))

# 1 指数与阶段
fig, ax = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax[0].plot(f.index, f.market_index, color='#1f4e79', lw=1.2, label='全A等权指数')
ax[0].plot(f.index, f.ma60, color='#c55a11', lw=.9, label='MA60')
ax[0].plot(f.index, f.ma250, color='#7f7f7f', lw=1.0, label='MA250')
shade = [('2019-01-04', '2021-12-13', '#d9ead3'), ('2024-09-18', '2026-05-13', '#d9ead3'),
         ('2026-05-13', '2026-10-22', '#f4cccc')]
for a, b, c in shade:
    ax[0].axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=.5, lw=0)
ax[0].axvline(pd.Timestamp('2026-10-22'), color='k', ls='--', lw=1)
ax[0].annotate('2026-10-22\n宽度 11.8%', xy=(pd.Timestamp('2026-10-22'), f.market_index.iloc[-1]),
               xytext=(-110, 30), textcoords='offset points', fontsize=9, color='k',
               arrowprops=dict(arrowstyle='->', color='k', lw=.8))
ax[0].set_ylabel('指数（2016-10-13=1）')
ax[0].legend(loc='upper left', fontsize=9)
ax[0].set_title('全A等权指数：2016-2026 的阶段与当前位置', fontsize=13)
ax[0].grid(alpha=.25)
ax[1].fill_between(f.index, 0, f.breadth20, color='#2e75b6', alpha=.55, lw=0)
ax[1].axhline(.5, color='#c00000', ls='--', lw=.8)
ax[1].axhline(.2, color='#7f7f7f', ls=':', lw=.8)
ax[1].set_ylabel('宽度20')
ax[1].set_ylim(0, 1)
ax[1].grid(alpha=.25)
ax[1].xaxis.set_major_locator(mdates.YearLocator())
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(os.path.join(OUT, '01_指数与阶段.png'), dpi=160)
plt.close()

# 2 2025-2026 放大：指数/宽度/情绪
z = f.loc['2025-01-01':]
fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
ax[0].plot(z.index, z.market_index, color='#1f4e79', lw=1.4)
ax[0].plot(z.index, z.ma20, color='#c55a11', lw=.9, label='MA20')
ax[0].plot(z.index, z.ma60, color='#7f7f7f', lw=1.0, label='MA60')
ax[0].set_title('2025-2026：从主升到高位调整', fontsize=13)
ax[0].legend(fontsize=9); ax[0].grid(alpha=.25)
ax[0].axvline(pd.Timestamp('2026-05-13'), color='#c00000', ls='--', lw=.9)
ax[0].axvline(pd.Timestamp('2026-09-22'), color='#7f7f7f', ls='--', lw=.9)
ax[1].fill_between(z.index, 0, z.breadth20, color='#2e75b6', alpha=.55, lw=0)
ax[1].axhline(.5, color='#c00000', ls='--', lw=.8)
ax[1].set_ylabel('宽度20'); ax[1].set_ylim(0, 1); ax[1].grid(alpha=.25)
ax[2].plot(z.index, z.sent, color='#7030a0', lw=1.1)
ax[2].axhline(0, color='#7f7f7f', lw=.7)
ax[2].fill_between(z.index, 0, z.sent, where=z.sent < 0, color='#f4cccc', alpha=.7, lw=0)
ax[2].set_ylabel('情绪分'); ax[2].grid(alpha=.25)
ax[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
plt.tight_layout()
plt.savefig(os.path.join(OUT, '02_2025_2026放大.png'), dpi=160)
plt.close()

# 3 策略阶段收益
fig, ax = plt.subplots(figsize=(10, 5.5))
colors = ['#c00000' if v >= .5 else '#7f7f7f' for v in phase['收益']]
ax.bar(phase['阶段'], phase['收益'] * 100, color=colors)
ax.axhline(50, color='#c00000', ls='--', lw=1, label='50% 目标线')
for i, v in enumerate(phase['收益'] * 100):
    ax.text(i, v + (4 if v >= 0 else -9), f'{v:+.1f}%', ha='center', fontsize=9)
ax.set_ylabel('阶段收益（%）'); ax.set_title('2仓50% 策略：按市场阶段拆分', fontsize=13)
ax.legend(); ax.grid(axis='y', alpha=.25)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '03_策略阶段收益.png'), dpi=160)
plt.close()

# 4 同态事件：f60 vs f120
fig, ax = plt.subplots(figsize=(8.5, 6))
col = np.where(ev['idx_pct'] > .7, '#c00000', '#2e75b6')
ax.scatter(ev['f60'] * 100, ev['f120'] * 100, c=col, s=55, alpha=.8)
for d, r in ev.iterrows():
    if d.year >= 2024 or (r.f60 < -.1) or (r.f120 < -.1):
        ax.annotate(d.strftime('%y-%m'), (r.f60 * 100, r.f120 * 100), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
ax.axhline(0, color='k', lw=.6); ax.axvline(0, color='k', lw=.6)
ax.set_xlabel('未来60日收益（%）'); ax.set_ylabel('未来120日收益（%）')
ax.set_title('20个同态低点：后续60/120日（红=高水位事件）', fontsize=12)
ax.grid(alpha=.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '04_同态事件.png'), dpi=160)
plt.close()

# 5 策略权益（对数）
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.plot(eq.index, eq.equity, color='#1f4e79', lw=1.3)
ax.set_yscale('log')
ax.set_title('2仓50% 核心策略：2018-2026 权益（对数）', fontsize=13)
ax.set_ylabel('权益（元，对数）')
ax.grid(alpha=.25)
for a, b, c in [('2020-01-01', '2021-12-31', '#d9ead3'), ('2022-01-01', '2024-01-31', '#fce4d6'),
                ('2024-09-18', '2026-05-13', '#d9ead3'), ('2026-05-13', '2026-10-13', '#f4cccc')]:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=.4, lw=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '05_策略权益.png'), dpi=160)
plt.close()

print('done', os.listdir(OUT))
