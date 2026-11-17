# -*- coding: utf-8 -*-
"""核心版本 2仓50%：保存权益/逐笔，训练/检验、年度、阶段与集中度审计。"""
import os, pandas as pd, numpy as np
from importlib.machinery import SourceFileLoader

BASE = '/Users/admin/WorkSpace/trading'
DIR = os.path.join(BASE, '2026-10-22_全信号_买卖时间_优先级研究')
OUT = os.path.join(BASE, '2026-10-22_当前阶段_情绪同态_50%策略/03_结果')
mod = SourceFileLoader('eng2', os.path.join(DIR, '108_稳健引擎.py')).load_module()

class E2(mod.Engine):
    def __init__(self):
        super().__init__()
        self.base_nogate = self.base_nogate & (self.AMT <= 70e6) & (self.C <= 15)

e = E2()
eq, t, met = e.run('2018-01-01', '2026-10-13', slots=2, sizing='fixed', maxpos=.50,
                   heat_cap=.10, gate_thr=.50, vol_target=.20, rank='amt', hold=20,
                   cost_rate=.0015)
eq.to_csv(os.path.join(OUT, '核心2仓50_权益.csv'), encoding='utf-8-sig', float_format='%.4f')
t.to_csv(os.path.join(OUT, '核心2仓50_逐笔.csv'), index=False, encoding='utf-8-sig',
         float_format='%.6f', date_format='%Y-%m-%d')

s = eq['equity']
rows = [dict(口径='全期2018-2026', 年化=met['cagr'], 最大回撤=met['maxdd'], 交易=len(t),
             胜率=(t.ret > 0).mean(), 盈亏比=t[t.ret > 0].pnl.sum() / max(1e-9, -t[t.ret < 0].pnl.sum()),
             前五笔占比=t.pnl.nlargest(5).sum() / t.pnl.sum() if t.pnl.sum() > 0 else np.nan,
             期末权益=s.iloc[-1])]
for a, b, lab in [('2018-01-01', '2021-12-31', '训练2018-2021'), ('2022-01-01', '2026-10-13', '检验2022-2026')]:
    ee = E2(); eqq, tt, mm = ee.run(a, b, slots=2, sizing='fixed', maxpos=.50, heat_cap=.10,
                                    gate_thr=.50, vol_target=.20, rank='amt', hold=20, cost_rate=.0015)
    rows.append(dict(口径=lab, 年化=mm['cagr'], 最大回撤=mm['maxdd'], 交易=len(tt),
                     胜率=(tt.ret > 0).mean(), 盈亏比=tt[tt.ret > 0].pnl.sum() / max(1e-9, -tt[tt.ret < 0].pnl.sum()),
                     前五笔占比=tt.pnl.nlargest(5).sum() / tt.pnl.sum() if tt.pnl.sum() > 0 else np.nan,
                     期末权益=eqq['equity'].iloc[-1]))
A = pd.DataFrame(rows)

# 年度
yrows = []
for y in range(2018, 2027):
    x = s[s.index.year == y]
    if len(x) == 0: continue
    prev = s[s.index < x.index[0]]
    base = prev.iloc[-1] if len(prev) else x.iloc[0]
    yrows.append(dict(年份=y, 收益=x.iloc[-1] / base - 1, 最大回撤=(x / x.cummax() - 1).min(),
                      交易=len(t[t.entry_date.dt.year == y]) if 'entry_date' in t else np.nan))
Y = pd.DataFrame(yrows)

# 阶段
phases = [('2018熊', '2018-01-24', '2019-01-04'), ('2019-21牛', '2019-01-04', '2021-12-13'),
          ('2022-24熊', '2021-12-13', '2024-02-05'), ('2024底部', '2024-02-05', '2024-09-18'),
          ('2024-26牛', '2024-09-18', '2026-05-13'), ('2026调整', '2026-05-13', '2026-10-13')]
prows = []
for name, a, b in phases:
    x = s.loc[a:b]
    prows.append(dict(阶段=name, 起点=a, 终点=b, 收益=x.iloc[-1] / x.iloc[0] - 1,
                      最大回撤=(x / x.cummax() - 1).min(), 交易日=len(x)))
P = pd.DataFrame(prows)

A.to_csv(os.path.join(OUT, '核心2仓50_分段.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
Y.to_csv(os.path.join(OUT, '核心2仓50_年度.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
P.to_csv(os.path.join(OUT, '核心2仓50_阶段.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
print(A.round(4).to_string(index=False))
print('\n', Y.round(4).to_string(index=False))
print('\n', P.round(4).to_string(index=False))
print('\n平均仓位 %.4f 平均总仓位 %.4f 平均持仓数 %.2f' % (
    t.weight.mean() if 'weight' in t else np.nan, eq.exposure.mean(), eq.npos.mean()))
print('前五笔:\n', t.nlargest(5, 'pnl')[['code', 'entry_date', 'exit_date', 'ret', 'pnl', 'reason']].to_string(index=False))
