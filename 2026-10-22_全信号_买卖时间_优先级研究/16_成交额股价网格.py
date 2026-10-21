# -*- coding: utf-8 -*-
"""成交额×股价 条件网格：看 7000万/15元 是平台还是尖峰。"""
import os
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
DIR = '/Users/admin/WorkSpace/trading/2026-10-22_全信号_买卖时间_优先级研究'
mod = SourceFileLoader('eng', os.path.join(DIR, '108_稳健引擎.py')).load_module()

class E2(mod.Engine):
    def __init__(self, amt, price):
        super().__init__()
        self.base_nogate = self.base_nogate & (self.AMT <= amt) & (self.C <= price)

rows = []
for amt in [40e6, 50e6, 60e6, 70e6, 80e6, 100e6]:
    for price in [10, 12, 15, 18, 20]:
        e = E2(amt, price)
        out = {}
        for start, end, lab in [('2018-01-01', '2026-10-13', '全期'),
                                ('2018-01-01', '2021-12-31', '训练'),
                                ('2022-01-01', '2026-10-13', '检验'),
                                ('2023-01-01', '2026-10-13', '2023起')]:
            eq, t, m = e.run(start, end, slots=2, sizing='fixed', maxpos=.45, heat_cap=.10,
                             gate_thr=.50, vol_target=.20, rank='amt', hold=20, cost_rate=.0015)
            out[lab] = (m['cagr'], m['maxdd'], len(t), (t['ret'] > 0).mean())
        rows.append(dict(成交额上限万=amt / 1e4, 股价上限=price,
                         全期年化=out['全期'][0], 全期回撤=out['全期'][1], 全期交易=out['全期'][2],
                         全期胜率=out['全期'][3], 训练年化=out['训练'][0], 训练回撤=out['训练'][1],
                         检验年化=out['检验'][0], 检验回撤=out['检验'][1],
                         近三年半年化=out['2023起'][0]))
        print(amt / 1e4, price, out['全期'][0], out['全期'][1], out['训练'][0], out['检验'][0])

D = pd.DataFrame(rows)
D.to_csv(os.path.join(DIR, '成交额股价网格.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print('\n按训练年化排序')
print(D.sort_values('训练年化', ascending=False).head(10).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n按检验年化排序')
print(D.sort_values('检验年化', ascending=False).head(10).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n70/15')
print(D[(D['成交额上限万'] == 7000) & (D['股价上限'] == 15)].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
