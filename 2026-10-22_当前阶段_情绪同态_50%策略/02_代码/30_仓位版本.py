# -*- coding: utf-8 -*-
"""70/15 趋势系统的仓位/仓数版本对比，以及按市场阶段拆分的收益。

基础引擎来自 2026-10-22_全信号_买卖时间_优先级研究/108_稳健引擎.py，
E2 = 原上涨趋势信号 + 20日均额<=7000万 + 后复权价<=15。
"""
import os, sys, pandas as pd, numpy as np
from importlib.machinery import SourceFileLoader

BASE = '/Users/admin/WorkSpace/trading'
DIR = os.path.join(BASE, '2026-10-22_全信号_买卖时间_优先级研究')
OUT = os.path.join(BASE, '2026-10-22_当前阶段_情绪同态_50%策略/03_结果')
mod = SourceFileLoader('eng', os.path.join(DIR, '108_稳健引擎.py')).load_module()

class E2(mod.Engine):
    def __init__(self):
        super().__init__()
        self.base_nogate = self.base_nogate & (self.AMT <= 70e6) & (self.C <= 15)

variants = [
    ('2仓45%', 2, .45),
    ('2仓50%', 2, .50),
    ('3仓33%', 3, .33),
    ('3仓40%', 3, .40),
    ('4仓25%', 4, .25),
]
rows = []
eqs = {}
for name, slots, maxpos in variants:
    e = E2()
    eq, t, met = e.run('2018-01-01', '2026-10-13', slots=slots, sizing='fixed', maxpos=maxpos,
                       heat_cap=.10, gate_thr=.50, vol_target=.20, rank='amt', hold=20,
                       cost_rate=.0015)
    eqs[name] = eq['equity']
    rows.append(dict(版本=name, 年化=met['cagr'], 最大回撤=met['maxdd'], 交易=len(t),
                     胜率=(t['ret'] > 0).mean(), 盈亏比=t[t.ret > 0].pnl.sum() / max(1e-9, -t[t.ret < 0].pnl.sum()),
                     平均持仓=t.get('weight', pd.Series([np.nan])).mean() if 'weight' in t else np.nan,
                     前五笔占比=t.pnl.nlargest(5).sum() / t.pnl.sum() if t.pnl.sum() > 0 else np.nan))
    t.to_csv(os.path.join(OUT, f'7015_{name}_逐笔.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, '7015_仓位版本.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
print(S.round(4).to_string(index=False))

phases = [
    ('2018熊', '2018-01-24', '2019-01-04'),
    ('2019-21牛', '2019-01-04', '2021-12-13'),
    ('2022-24熊', '2021-12-13', '2024-02-05'),
    ('2024底部', '2024-02-05', '2024-09-18'),
    ('2024-26牛', '2024-09-18', '2026-05-13'),
    ('2026调整', '2026-05-13', '2026-10-13'),
]
prows = []
E = pd.read_csv(os.path.join(OUT, '同态事件_明细.csv'))  # noqa
for name, eq in eqs.items():
    for pname, a, b in phases:
        x = eq.loc[a:b]
        if len(x) < 2: continue
        prows.append(dict(版本=name, 阶段=pname, 收益=x.iloc[-1] / x.iloc[0] - 1,
                          回撤=(x / x.cummax() - 1).min(), 天数=len(x)))
P = pd.DataFrame(prows)
P.to_csv(os.path.join(OUT, '7015_仓位版本_阶段收益.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
print('\n', P.pivot(index='阶段', columns='版本', values='收益').round(4).to_string())
print('\n', P.pivot(index='阶段', columns='版本', values='回撤').round(4).to_string())
