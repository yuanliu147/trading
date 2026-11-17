# -*- coding: utf-8 -*-
"""前视审计：同样信号，只差“T+1 收盘跌停是否在入场时剔除”。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

BASE = '/Users/admin/WorkSpace/trading/2026-10-22_龙虎榜涨停_行业影响与策略'
DATA = os.path.join(BASE, '01_数据'); OUT = os.path.join(BASE, '03_结果'); FIG = os.path.join(BASE, '04_图表')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

M = pd.read_parquet(os.path.join(DATA, '市场日频.parquet')); M['date'] = pd.to_datetime(M.date)
MD = pd.DatetimeIndex(M.date); BENCH = M.set_index('date')['mkt_ret1_eq'].fillna(0)

S = pd.read_parquet(os.path.join(DATA, '个股日频.parquet'),
                    columns=['date','thscode','st','age','liq','next_open','next_close','next_low',
                             'next_ld_seal','next2_open','next2_close','next_limit_rate','next_unbuyable',
                             'next_open_ret','next_date'])
S['date'] = pd.to_datetime(S.date)
nd = pd.Series(MD[1:].values, index=MD[:-1].values)
S['next_trade_date'] = S.date.map(nd); S['gap_ok'] = S.next_date.eq(S.next_trade_date)
S['ld_open'] = S.next_open_ret <= -(S.next_limit_rate - 0.002)
S['stamp'] = np.where(S.date >= '2023-08-28', 0.0005, 0.001)
S = S[(S.st==0)&(S.age>=60)&(S.liq==1)&S.next_open.notna()&S.next_close.notna()&
      (S.next_unbuyable==0)&(~S.ld_open)&S.gap_ok&(S.next_open_ret<=0.07)].copy()

L = pd.read_parquet(os.path.join(DATA, '龙虎榜事件_含前瞻.parquet')); L['date'] = pd.to_datetime(L.date)
L = L.drop(columns=[c for c in ['next_open','next_close','next_low','next_ld_seal','next2_open','next2_close',
                                'next_limit_rate','next_unbuyable','next_open_ret','next_date','gap_ok','ld_open'] if c in L.columns])
L = L.merge(S, on=['date','thscode'], how='inner')
L = L[(L.next_open.notna())&(L.next_close.notna())&(L.next_unbuyable==0)&(~L.ld_open)&L.gap_ok&
      (L.next_open_ret<=0.07)]
L = L[~L.reason_type.isin(['跌幅','连续跌幅','退市'])]
x = L[(L.next_open_ret>=-0.05)&(L.next_open_ret<=-0.005)].sort_values(['date','next_open_ret'])
x['rk'] = x.groupby('date').cumcount(); x = x[x.rk<10].copy()


def run(ideal):
    y = x.copy()
    if ideal:
        y = y[y.next_ld_seal.fillna(0)==0].copy()
    entry = y.next_open.values; exitp = y.next_close.values.copy()
    stuck = (y.next_ld_seal.fillna(0).values==1) | (y.next_close.values<=0)
    if stuck.any():
        sub = y.loc[stuck]
        alt = sub.next2_open.values
        alt = np.where(np.isfinite(alt), alt, sub.next2_close.values)
        exitp[stuck] = alt
    r = exitp*(1-0.0045-y.stamp.values)/(entry*1.0045)-1
    y['ret'] = r; y['wr'] = r/10
    d = y.groupby('next_date').wr.sum(); d.index = pd.to_datetime(d.index)
    d = d.reindex(MD).fillna(0)
    nav = (1+d).cumprod(); n = len(d)
    m = dict(年化=nav.iloc[-1]**(252/n)-1, 最大回撤=(nav/nav.cummax()-1).min(),
             Sharpe=d.mean()/(d.std()+1e-9)*np.sqrt(252), 交易数=len(y),
             平均单笔=r.mean(), 单笔中位=np.median(r), 单笔胜率=(r>0).mean(),
             跌停卖不出=(stuck.sum() if not ideal else 0),
             跌停卖不出占比=(stuck.mean() if not ideal else 0),
             跌停单平均=r[stuck].mean() if not ideal and stuck.any() else np.nan)
    return d, y, m

d1, y1, m1 = run(True)
d2, y2, m2 = run(False)
A = pd.DataFrame([dict(版本='理想版（剔除T+1收盘跌停，前视）', **m1),
                  dict(版本='实盘版（跌停顺延T+2开盘）', **m2)])
A.to_csv(os.path.join(OUT, '前视审计_理想vs实盘.csv'), index=False, encoding='utf-8-sig')
d1.to_frame('ret').to_csv(os.path.join(OUT, '前视审计_理想权益.csv'), encoding='utf-8-sig')
d2.to_frame('ret').to_csv(os.path.join(OUT, '前视审计_实盘权益.csv'), encoding='utf-8-sig')
print(A.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(d1.index, (1+d1).cumprod(), label='理想版（前视：剔除T+1收盘跌停）', color='#2c6e8f', lw=1.5)
ax.plot(d2.index, (1+d2).cumprod(), label='实盘版（跌停顺延T+2开盘）', color='#a33b3b', lw=1.5)
b = BENCH.reindex(MD).fillna(0); ax.plot(b.index, (1+b).cumprod(), label='全A等权', color='black', ls='--', lw=1.1)
ax.set_yscale('log'); ax.set_title('前视审计：同一个信号，只差“T+1 收盘跌停是否剔除”')
ax.set_ylabel('净值（对数轴）'); ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, '图4_前视审计_理想vs实盘.png'), dpi=170, bbox_inches='tight')
plt.close(fig)
print('saved chart')
