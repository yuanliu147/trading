# -*- coding: utf-8 -*-
"""信号优化候选：把全信号分析里看起来有效的过滤，放回真实的2仓系统测试。"""
import os
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
DIR = '/Users/admin/WorkSpace/trading/2026-10-22_全信号_买卖时间_优先级研究'
mod = SourceFileLoader('eng', os.path.join(DIR, '108_稳健引擎.py')).load_module()
base_engine = mod.Engine()
base_sig = base_engine.base_nogate & (base_engine.MB[:, None] > 0.5)
first_mask = np.zeros_like(base_sig, dtype=bool)
for j in range(base_sig.shape[1]):
    col = base_sig[:, j]
    idx = np.where(col)[0]
    if len(idx) == 0:
        continue
    t = idx[0]
    while t < len(col) and col[t]:
        first_mask[t, j] = True
        t += 1
TURNZ = base_engine.P['turn_z250'].values

class E2(mod.Engine):
    def __init__(self, kind, cap=None, first=False, turn=None, age=None):
        super().__init__()
        b = self.base_nogate
        if kind == 'amt':
            b = b & (self.AMT <= cap)
        elif kind == 'price':
            b = b & (self.C <= cap)
        elif kind == 'both':
            b = b & (self.AMT <= cap[0]) & (self.C <= cap[1])
        elif kind == 'turn':
            b = b & (TURNZ <= (turn if turn is not None else cap))
        elif kind == 'age':
            b = b & (self.AGE <= (age if age is not None else cap))
        elif kind == 'first':
            b = first_mask
        self.base_nogate = b

variants = [
    ('基准', 'base', None, 0.50),
    ('成交额≤4000万', 'amt', 40e6, 0.50),
    ('成交额≤7000万', 'amt', 70e6, 0.50),
    ('股价≤20元', 'price', 20, 0.50),
    ('成交额≤7000万+股价≤15元', 'both', (70e6, 15), 0.50),
    ('市场宽度≥75%', 'base', None, 0.75),
    ('只做每股首段信号', 'first', None, 0.50),
    ('换手异常值≤1.63', 'turn', 1.632, 0.50),
    ('趋势年龄≤21日', 'age', 21, 0.50),
]
rows = []
for name, kind, cap, gate in variants:
    e = E2(kind, cap)
    out = {}
    for start, end, lab in [('2018-01-01', '2026-10-13', '全期'),
                            ('2018-01-01', '2021-12-31', '训练'),
                            ('2022-01-01', '2026-10-13', '检验'),
                            ('2023-01-01', '2026-10-13', '2023起')]:
        eq, t, m = e.run(start, end, slots=2, sizing='fixed', maxpos=.45, heat_cap=.10,
                         gate_thr=gate, vol_target=.20, rank='amt', hold=20, cost_rate=.0015)
        out[lab] = (m['cagr'], m['maxdd'], len(t), (t['ret'] > 0).mean())
        if lab == '全期':
            s = eq['equity']; p = t['pnl']
            top5 = p.nlargest(5).sum()
            out['top5占比'] = top5 / p.sum() if p.sum() > 0 else np.nan
            if out['top5占比'] is not np.nan and out['top5占比'] > 1: out['top5占比'] = np.nan
            out['去前五净盈亏'] = p.sum() - top5
            for y in [2018, 2025]:
                d = s[s.index.year == y]; prev = s[s.index < d.index[0]]
                base = prev.iloc[-1] if len(prev) else d.iloc[0]
                out[f'{y}收益'] = d.iloc[-1] / base - 1
    rows.append(dict(
        方案=name,
        全期年化=out['全期'][0], 全期回撤=out['全期'][1], 全期交易=out['全期'][2], 全期胜率=out['全期'][3],
        训练年化=out['训练'][0], 训练回撤=out['训练'][1],
        检验年化=out['检验'][0], 检验回撤=out['检验'][1],
        近三年半年化=out['2023起'][0], 近三年半回撤=out['2023起'][1],
        前五笔占比=out['top5占比'], 去掉前五净盈亏=out['去前五净盈亏'],
        一八年收益=out['2018收益'], 二五年收益=out['2025收益'],
    ))
    print(name, out)

D = pd.DataFrame(rows)
D.to_csv(os.path.join(DIR, '信号优化_候选.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print(D.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
