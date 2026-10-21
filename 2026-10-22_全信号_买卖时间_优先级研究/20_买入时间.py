# -*- coding: utf-8 -*-
"""70/15子池：买入月份/季度与回测起点敏感性。"""
import os
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
DIR = '/Users/admin/WorkSpace/trading/2026-10-22_全信号_买卖时间_优先级研究'
T = pd.read_csv(os.path.join(DIR, '全信号_一万_逐笔.csv'), encoding='utf-8-sig', parse_dates=['买入日', '首个信号日'])
X = T[(T['成交额'] <= 70e6) & (T['股价'] <= 15)].copy()
X['期'] = np.where(X['年份'] <= 2021, '训练2018-2021', '检验2022-2026')
X['月'] = X['买入日'].dt.month
X['季'] = X['买入日'].dt.quarter

def agg(z):
    x = z['止损后净盈亏'].dropna()
    return dict(笔数=len(x), 平均=x.mean(), 中位=x.median(), 总盈亏=x.sum(),
                胜率=(x > 0).mean(), 盈亏比=x[x > 0].sum() / max(1e-9, -x[x < 0].sum()))

rows = []
for by, name in [('月', '月份'), ('季', '季度')]:
    for k, z in X.groupby(by):
        r = agg(z); r[name] = k; rows.append(r)
M = pd.DataFrame(rows)
# 训练/检验月份
rows = []
for m in range(1, 13):
    r = {'月份': m}
    for per in ['训练2018-2021', '检验2022-2026', '全期']:
        z = X[X['月'] == m] if per == '全期' else X[(X['月'] == m) & (X['期'] == per)]
        a = agg(z); r[f'{per}_笔数'] = a['笔数']; r[f'{per}_平均'] = a['平均']; r[f'{per}_胜率'] = a['胜率']
    rows.append(r)
MM = pd.DataFrame(rows)
MM.to_csv(os.path.join(DIR, '7015_买入月份.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
M.to_csv(os.path.join(DIR, '7015_买入季度.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print(MM.to_string(index=False, float_format=lambda v: f'{v:.1f}'))

mod = SourceFileLoader('eng', os.path.join(DIR, '108_稳健引擎.py')).load_module()
class E2(mod.Engine):
    def __init__(self):
        super().__init__(); self.base_nogate = self.base_nogate & (self.AMT <= 70e6) & (self.C <= 15)

rows = []
for m in range(1, 13):
    s = f'2018-{m:02d}-01'; e = E2()
    eq, t, met = e.run(s, '2026-10-13', slots=2, sizing='fixed', maxpos=.45, heat_cap=.10,
                       gate_thr=.50, vol_target=.20, rank='amt', hold=20, cost_rate=.0015)
    rows.append(dict(起点类型='2018年内月份', 起点=s, 年化=met['cagr'], 回撤=met['maxdd'],
                     交易=len(t), 胜率=(t['ret'] > 0).mean(), 期末权益=eq['equity'].iloc[-1],
                     首笔= t['entry_date'].min() if len(t) else pd.NaT))
for y in range(2018, 2026):
    s = f'{y}-01-01'; e = E2()
    eq, t, met = e.run(s, '2026-10-13', slots=2, sizing='fixed', maxpos=.45, heat_cap=.10,
                       gate_thr=.50, vol_target=.20, rank='amt', hold=20, cost_rate=.0015)
    rows.append(dict(起点类型='年初', 起点=s, 年化=met['cagr'], 回撤=met['maxdd'],
                     交易=len(t), 胜率=(t['ret'] > 0).mean(), 期末权益=eq['equity'].iloc[-1],
                     首笔= t['entry_date'].min() if len(t) else pd.NaT))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(DIR, '7015_起点敏感性.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print('\nstart sensitivity')
print(S.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
