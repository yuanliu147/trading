# -*- coding: utf-8 -*-
"""7000万/15元组合的脆弱性：成本、去掉大赢家、分段。"""
import os
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
DIR = '/Users/admin/WorkSpace/trading/2026-10-22_全信号_买卖时间_优先级研究'
mod = SourceFileLoader('eng', os.path.join(DIR, '108_稳健引擎.py')).load_module()

class E2(mod.Engine):
    def __init__(self, amt=None, price=None, exclude=(), cost_rate=.0015):
        super().__init__()
        self.cost_rate = cost_rate
        b = self.base_nogate
        if amt: b = b & (self.AMT <= amt)
        if price: b = b & (self.C <= price)
        for code in exclude:
            if code in self.code_idx:
                b[:, self.code_idx[code]] = False
        self.base_nogate = b

def run(e, start, end, cost=.0015):
    eq, t, m = e.run(start, end, slots=2, sizing='fixed', maxpos=.45, heat_cap=.10,
                     gate_thr=.50, vol_target=.20, rank='amt', hold=20, cost_rate=cost)
    s = eq['equity']
    return m, t, s

def year_ret(s, y):
    d = s[s.index.year == y]
    prev = s[s.index < d.index[0]]
    base = prev.iloc[-1] if len(prev) else d.iloc[0]
    return d.iloc[-1] / base - 1 if len(d) else np.nan

rows = []
for label, amt, price in [('基准', None, None), ('70/15', 70e6, 15)]:
    e = E2(amt, price)
    m, t, s = run(e, '2018-01-01', '2026-10-13')
    p = t['pnl']; top5 = p.nlargest(5).sum()
    row = dict(方案=label, 成本='0.15%', 全期年化=m['cagr'], 全期回撤=m['maxdd'], 交易=len(t),
               胜率=(t['ret'] > 0).mean(), 总盈亏=p.sum(),
               前五笔占比=top5 / p.sum() if p.sum() > 0 else np.nan,
               去前五盈亏=p.sum() - top5,
               二零一八=year_ret(s, 2018), 二零二零=year_ret(s, 2020),
               二零二二=year_ret(s, 2022), 二零二五=year_ret(s, 2025))
    rows.append(row)
    # 分段
    for start, end, lab in [('2018-01-01', '2024-12-31', '2018-2024'),
                            ('2022-01-01', '2024-12-31', '2022-2024'),
                            ('2025-01-01', '2026-10-13', '2025-2026')]:
        m2, t2, s2 = run(e, start, end)
        rows.append(dict(方案=label, 成本=lab, 全期年化=m2['cagr'], 全期回撤=m2['maxdd'], 交易=len(t2),
                         胜率=(t2['ret'] > 0).mean() if len(t2) else np.nan))
    # 成本
    for cost in [0.0035, 0.005]:
        m2, t2, s2 = run(e, '2018-01-01', '2026-10-13', cost)
        rows.append(dict(方案=label, 成本=f'{cost*100:.2f}%', 全期年化=m2['cagr'], 全期回撤=m2['maxdd'],
                         交易=len(t2), 胜率=(t2['ret'] > 0).mean()))
    # 去掉大赢家
    for ex, exlab in [(['688098.SH'], '去掉688098'), (['688098.SH', '603557.SH', '002694.SZ', '600768.SH', '300018.SZ'], '去掉前五')]:
        e2 = E2(amt, price, ex)
        m2, t2, s2 = run(e2, '2018-01-01', '2026-10-13')
        rows.append(dict(方案=label, 成本=exlab, 全期年化=m2['cagr'], 全期回撤=m2['maxdd'],
                         交易=len(t2), 胜率=(t2['ret'] > 0).mean() if len(t2) else np.nan,
                         总盈亏=t2['pnl'].sum()))

D = pd.DataFrame(rows)
D.to_csv(os.path.join(DIR, '组合脆弱性.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print(D.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
