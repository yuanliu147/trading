# -*- coding: utf-8 -*-
"""生成报告图表。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

BASE = '/Users/admin/WorkSpace/trading/2026-10-22_龙虎榜涨停_行业影响与策略'
OUT = os.path.join(BASE, '03_结果')
FIG = os.path.join(BASE, '04_图表')
os.makedirs(FIG, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
C1, C2, C3, C4 = '#2c6e8f', '#d9822b', '#4a7c59', '#a33b3b'

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)

# 1. 行业事件 vs 非事件
d = pd.read_csv(os.path.join(OUT, '行业预测_事件与非事件.csv'))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, ev, title in [(axes[0], '涨停', '涨停事件：同行业非事件股'), (axes[1], '龙虎榜', '龙虎榜事件：同行业非事件股')]:
    x = d[(d.事件 == ev) & (d.口径 == '剔除两者')].sort_values('h')
    ax.bar(x.h.astype(str), x['差'] * 100, color=C1, alpha=0.85)
    for i, (_, r) in enumerate(x.iterrows()):
        ax.text(i, r['差'] * 100 - 0.03, f"t={r.t:.1f}", ha='center', va='top', fontsize=8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title(title)
    ax.set_xlabel('未来交易日')
    ax.set_ylabel('事件组 - 非事件组（%）')
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
save(fig, '图1_行业外溢_事件与非事件.png')

# 2. 关键特征 IC
ic = pd.read_csv(os.path.join(OUT, '行业预测_RankIC.csv'))
feats = ['lhb_org_ratio', 'lhb_net_ratio', 'lhb_share', 'lu_share', 'lu2_share', 'lu_open_share']
labels = ['机构净买/成交额', '龙虎榜净买/成交额', '龙虎榜上榜占比', '涨停占比', '连板占比', '炸板占比']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for ax, h in [(axes[0], 3), (axes[1], 5)]:
    x = ic[(ic.期 == '全样本') & (ic.目标 == f'行业剔除两者_h{h}') & (ic.特征.isin(feats))]
    x = x.set_index('特征').reindex(feats)
    colors = [C3 if v >= 0 else C4 for v in x.IC]
    ax.barh(labels, x.IC * 100, color=colors, alpha=0.85)
    for i, (v, t) in enumerate(zip(x.IC, x.t)):
        ax.text(v * 100 + (0.05 if v >= 0 else -0.05), i, f"t={t:.1f}", va='center',
                ha='left' if v >= 0 else 'right', fontsize=8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_title(f'行业剔除事件股后，未来 {h} 日 Rank IC')
    ax.set_xlabel('IC（%）')
save(fig, '图2_行业IC_关键特征.png')

# 3. 个股事件收益
lhb = pd.read_csv(os.path.join(OUT, '事件研究_龙虎榜_个股.csv'))
lu = pd.read_csv(os.path.join(OUT, '事件研究_涨停_个股.csv'))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
nb = ['净卖最多', '净卖', '中性', '净买', '净买最多']
for i, h in enumerate([1, 3, 5]):
    x = lhb[(lhb.分组 == 'net_bucket') & (lhb.h == h)].set_index('组值').reindex(nb)
    axes[0].plot(nb, x['mean'] * 100, marker='o', label=f'{h}日', color=[C1, C2, C3][i])
axes[0].axhline(0, color='black', lw=0.8)
axes[0].set_title('龙虎榜净买分组：个股 T+1 开盘买入')
axes[0].set_ylabel('平均收益（%）')
axes[0].legend()
cb = ['首板', '2连板', '3连板', '4板+']
for i, h in enumerate([1, 3, 5]):
    x = lu[(lu.分组 == 'consec_bucket') & (lu.h == h)].set_index('组值').reindex(cb)
    axes[1].plot(cb, x['mean'] * 100, marker='o', label=f'{h}日', color=[C1, C2, C3][i])
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_title('涨停连板高度：个股 T+1 开盘买入')
axes[1].set_ylabel('平均收益（%）')
axes[1].legend()
save(fig, '图3_个股事件收益.png')

# 4. 前视审计：理想 vs 实盘
fig, ax = plt.subplots(figsize=(11, 4.8))
for lab, f, col in [('理想版（前视：剔除T+1收盘跌停）', '前视审计_理想权益.csv', C1),
                    ('实盘版（跌停顺延T+2开盘）', '前视审计_实盘权益.csv', C4)]:
    x = pd.read_csv(os.path.join(OUT, f), index_col=0, parse_dates=True).iloc[:, 0]
    ax.plot(x.index, (1 + x).cumprod(), label=lab, lw=1.5, color=col)
m = pd.read_parquet(os.path.join(BASE, '01_数据/市场日频.parquet'))
m['date'] = pd.to_datetime(m.date)
m['nav'] = (1 + m.mkt_ret1_eq.fillna(0)).cumprod()
ax.plot(m.date, m.nav, label='全A等权', color='black', lw=1.1, ls='--')
ax.set_yscale('log')
ax.set_title('前视审计：同一个信号，只差“T+1 收盘跌停是否剔除”')
ax.set_ylabel('净值（对数轴）'); ax.legend(); ax.grid(alpha=0.25)
save(fig, '图4_前视审计_理想vs实盘.png')

# 5. 分年与回撤
x = pd.read_csv(os.path.join(OUT, '最终策略_权益_均衡_N10_45bp.csv'), index_col=0, parse_dates=True).iloc[:, 0]
nav = (1 + x).cumprod(); dd = nav / nav.cummax() - 1
yr = (1 + x).groupby(x.index.year).prod() - 1
fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=False, gridspec_kw={'height_ratios': [2, 1]})
axes[0].bar(yr.index.astype(str), yr.values * 100, color=[C3 if v >= 0 else C4 for v in yr.values], alpha=0.85)
axes[0].axhline(0, color='black', lw=0.8)
axes[0].set_title('实盘建模版 N=10：年度收益（%，45bp；T+1收盘跌停顺延T+2开盘）')
axes[0].yaxis.set_major_formatter(PercentFormatter(decimals=0))
axes[1].fill_between(dd.index, dd.values * 100, 0, color=C4, alpha=0.45)
axes[1].set_title('回撤（%）')
axes[1].yaxis.set_major_formatter(PercentFormatter(decimals=0))
save(fig, '图5_分年收益与回撤.png')

# 6. 成本敏感
c = pd.read_csv(os.path.join(OUT, '最终策略_成本敏感.csv'))
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(c['单边成本'] * 100, c['年化'] * 100, marker='o', color=C1, label='年化')
ax.plot(c['单边成本'] * 100, c['最大回撤'] * 100, marker='s', color=C4, label='最大回撤')
ax.axvline(0.45, color='gray', ls='--', lw=0.8)
ax.set_title('实盘建模版：成本敏感（单边成本，卖出另加印花税）')
ax.set_xlabel('单边成本（%）')
ax.set_ylabel('%')
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.legend()
ax.grid(alpha=0.25)
save(fig, '图6_成本敏感.png')

# 7. 参数邻域热力图（N × 低开区间，按低开最深排序）
p = pd.read_csv(os.path.join(OUT, '最终策略_参数邻域.csv'))
p = p[p['排序'] == 'gap'].copy()
p['区间'] = p.apply(lambda r: f"{abs(r['低开上限'])*100:.1f}–{abs(r['低开下限'])*100:.1f}%", axis=1)
piv = p.pivot_table(index='N', columns='区间', values='年化') * 100
fig, ax = plt.subplots(figsize=(8.4, 4.2))
im = ax.imshow(piv.values, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=9)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f'{piv.values[i,j]:.0f}%', ha='center', va='center', fontsize=9)
ax.set_xlabel('低开区间（开盘/T收盘-1）'); ax.set_ylabel('最多持仓数 N')
ax.set_title('实盘建模版：参数邻域年化（45bp，按低开最深排序）')
fig.colorbar(im, ax=ax, label='年化 %')
save(fig, '图7_参数邻域.png')

# 8. 前视审计：分年收益
d1 = pd.read_csv(os.path.join(OUT, '前视审计_理想权益.csv'), index_col=0, parse_dates=True).iloc[:, 0]
d2 = pd.read_csv(os.path.join(OUT, '前视审计_实盘权益.csv'), index_col=0, parse_dates=True).iloc[:, 0]
y1 = (1 + d1).groupby(d1.index.year).prod() - 1
y2 = (1 + d2).groupby(d2.index.year).prod() - 1
years = sorted(set(y1.index) | set(y2.index))
fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(len(years)); w = 0.38
ax.bar(x - w/2, [y1.get(y, 0)*100 for y in years], width=w, label='理想版（前视）', color=C1, alpha=0.85)
ax.bar(x + w/2, [y2.get(y, 0)*100 for y in years], width=w, label='实盘版', color=C4, alpha=0.85)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(years, rotation=45)
ax.set_title('前视审计：分年收益（%，45bp）')
ax.set_ylabel('%'); ax.yaxis.set_major_formatter(PercentFormatter(decimals=0)); ax.legend()
ax.grid(axis='y', alpha=0.25)
save(fig, '图8_前视审计_分年收益.png')

print('done')
