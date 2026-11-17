# -*- coding: utf-8 -*-
"""持仓数量与组合收益：全信号（不加70/15过滤），1/2/3/5/10仓等权模拟。

这是主系统150笔的同一套引擎与参数；2仓/45%基准应复现 21.6%/-26.4%/150笔。
"""
import os
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

BASE = '/Users/admin/WorkSpace/trading'
SRC = os.path.join(BASE, '2026-10-22_全信号_买卖时间_优先级研究')
DIR = os.path.join(BASE, '2026-10-22_持仓天数_涨跌路径')
mod = SourceFileLoader('eng_full2', os.path.join(SRC, '108_稳健引擎.py')).load_module()

CASES = [
    ('全信号_1仓', 1, 0.95),
    ('全信号_2仓等权', 2, 0.475),
    ('全信号_3仓', 3, 0.95 / 3),
    ('全信号_5仓', 5, 0.95 / 5),
    ('全信号_10仓', 10, 0.95 / 10),
    ('全信号_2仓45%基准', 2, 0.45),
]

rows = []
year_rows = []
for label, slots, maxpos in CASES:
    e = mod.Engine()
    eq, t, met = e.run('2018-01-01', '2026-10-13', slots=slots, sizing='fixed',
                       maxpos=maxpos, heat_cap=.10, gate_thr=.50, vol_target=.20,
                       rank='amt', hold=20, cost_rate=.0015)
    t = t.copy()
    t['net_ret'] = t['pnl'] / (t['entry'] * t['shares'])
    t['net_win'] = t['net_ret'] > 0
    rows.append(dict(方案=label, 仓位数=slots, 单票上限=maxpos,
                     年化=met['cagr'], 最大回撤=met['maxdd'], 最大回撤_最低价=met['maxdd_low'],
                     交易笔数=met['trades'], 胜率=met['win'], 盈亏比=met['pf'],
                     平均持有=met['avg_hold'], 平均持仓数=met['avg_pos'],
                     平均单票权重=met['avg_weight'], 平均仓位=met['avg_exposure'],
                     单笔净均值=t['net_ret'].mean(), 单笔净中位=t['net_ret'].median(),
                     单笔净胜率=t['net_win'].mean(),
                     单笔净盈亏均值=t['pnl'].mean(),
                     前五笔占比=met['top5_share'], 最差单笔贡献=met['worst_trade'],
                     期末权益=met['final']))
    t.to_csv(os.path.join(DIR, f'持仓数量_逐笔_{label}.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f', date_format='%Y-%m-%d')
    s = eq['equity']
    for y in range(2018, 2027):
        d = s[s.index.year == y]
        if len(d) == 0:
            continue
        prev = s[s.index < d.index[0]]
        base = prev.iloc[-1] if len(prev) else d.iloc[0]
        year_rows.append(dict(方案=label, 年份=y, 年度收益=d.iloc[-1] / base - 1,
                              年内最大回撤=(d / d.cummax() - 1).min()))
    print(label, 'cagr %.3f dd %.3f trades %d win %.3f avgpos %.2f exp %.2f' % (
        met['cagr'], met['maxdd'], met['trades'], met['win'], met['avg_pos'], met['avg_exposure']))

R = pd.DataFrame(rows)
R.to_csv(os.path.join(DIR, '持仓数量_全信号_组合.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
Y = pd.DataFrame(year_rows)
Y.to_csv(os.path.join(DIR, '持仓数量_全信号_年度.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')

print('\n=== 全信号组合 ===')
print(R[['方案', '仓位数', '单票上限', '年化', '最大回撤', '交易笔数', '胜率', '盈亏比',
         '平均持仓数', '平均仓位', '单笔净均值', '单笔净胜率', '前五笔占比']]
      .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n=== 全信号年度 ===')
print(Y.pivot(index='年份', columns='方案', values='年度收益')
      .to_string(float_format=lambda v: f'{v:.3f}'))
