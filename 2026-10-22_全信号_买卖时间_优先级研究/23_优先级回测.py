# -*- coding: utf-8 -*-
"""70/15子池：同一天多信号时的优先级比较（2仓、45%、20日、-20%止损）。

预注册假设：不改仓位/持有期/止损，只改候选排序。
基准：成交额升序（现行）。
候选：市值升序/降序、行业宽度高/中、行业20日高、趋势年龄小、距60日高点近、
      低波、回调、低价。
审计：全期/训练/检验/2023起、成本、年度、起点、随机排序零分布。
"""
import os
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
DIR = '/Users/admin/WorkSpace/trading/2026-10-22_全信号_买卖时间_优先级研究'
mod = SourceFileLoader('eng', os.path.join(DIR, '108_稳健引擎.py')).load_module()

class E2(mod.Engine):
    def __init__(self):
        super().__init__()
        self.base_nogate = self.base_nogate & (self.AMT <= 70e6) & (self.C <= 15)

ranks = ['amt', 'mcap', 'mcap_desc', 'sbr', 'sbr_mid', 'sr20', 'age', 'dd60', 'lowvol', 'pullback', 'price']
rows = []
for rank in ranks:
    e = E2(); out = {}
    for start, end, lab in [('2018-01-01', '2026-10-13', '全期'),
                            ('2018-01-01', '2021-12-31', '训练'),
                            ('2022-01-01', '2026-10-13', '检验'),
                            ('2023-01-01', '2026-10-13', '2023起')]:
        eq, t, met = e.run(start, end, slots=2, sizing='fixed', maxpos=.45, heat_cap=.10,
                           gate_thr=.50, vol_target=.20, rank=rank, hold=20, cost_rate=.0015)
        out[lab] = (met['cagr'], met['maxdd'], len(t), (t['ret'] > 0).mean(), t['pnl'].sum())
        if lab == '全期':
            p = t['pnl']; top5 = p.nlargest(5).sum()
            out['top5'] = top5 / p.sum() if p.sum() > 0 else np.nan
            out['ex_top5'] = p.sum() - top5
    rows.append(dict(排序=rank, 全期年化=out['全期'][0], 全期回撤=out['全期'][1],
                     全期交易=out['全期'][2], 全期胜率=out['全期'][3], 训练年化=out['训练'][0],
                     训练回撤=out['训练'][1], 检验年化=out['检验'][0], 检验回撤=out['检验'][1],
                     近三年半年化=out['2023起'][0], 前五笔占比=out['top5'], 去前五净盈亏=out['ex_top5']))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(DIR, '优先级_回测.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print(D.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

# 成本 + 年度 + 起点 + 交易记录
year_rows = []
start_rows = []
for rank in ['amt', 'mcap']:
    for cost in [0.0015, 0.0035]:
        e = E2(); eq, t, met = e.run('2018-01-01', '2026-10-13', slots=2, sizing='fixed',
                                     maxpos=.45, heat_cap=.10, gate_thr=.50, vol_target=.20,
                                     rank=rank, hold=20, cost_rate=cost)
        s = eq['equity']; p = t['pnl']; top5 = p.nlargest(5).sum()
        year_rows.append(dict(排序=rank, 成本=cost, 全期年化=met['cagr'], 全期回撤=met['maxdd'],
                              交易=len(t), 胜率=(t['ret'] > 0).mean(),
                              前五笔占比=top5 / p.sum() if p.sum() > 0 else np.nan))
        if cost == 0.0015:
            t.to_csv(os.path.join(DIR, f'7015_{rank}_逐笔.csv'), index=False, encoding='utf-8-sig',
                     float_format='%.6f', date_format='%Y-%m-%d')
            for y in range(2018, 2027):
                d = s[s.index.year == y]; prev = s[s.index < d.index[0]]
                base = prev.iloc[-1] if len(prev) else d.iloc[0]
                year_rows.append(dict(排序=rank, 成本=f'年度{y}', 全期年化=d.iloc[-1] / base - 1,
                                      全期回撤=np.nan, 交易=np.nan, 胜率=np.nan, 前五笔占比=np.nan))
    for m in range(1, 13):
        e = E2(); eq, t, met = e.run(f'2018-{m:02d}-01', '2026-10-13', slots=2, sizing='fixed',
                                     maxpos=.45, heat_cap=.10, gate_thr=.50, vol_target=.20,
                                     rank=rank, hold=20, cost_rate=.0015)
        start_rows.append(dict(排序=rank, 起点=f'2018-{m:02d}', 年化=met['cagr'],
                               回撤=met['maxdd'], 交易=len(t), 胜率=(t['ret'] > 0).mean()))
pd.DataFrame(year_rows).to_csv(os.path.join(DIR, '优先级_年度成本.csv'), index=False,
                               encoding='utf-8-sig', float_format='%.6f')
pd.DataFrame(start_rows).to_csv(os.path.join(DIR, '优先级_起点.csv'), index=False,
                                encoding='utf-8-sig', float_format='%.6f')
print('\ncost/yearly')
print(pd.DataFrame(year_rows).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\nstart range')
print(pd.DataFrame(start_rows).groupby('排序')['年化'].agg(['min', 'median', 'max']).to_string(float_format=lambda v: f'{v:.3f}'))

# 随机排序零分布 + 市值降序负对照
rnd = []
for seed in range(30):
    e = E2(); eq, t, met = e.run('2018-01-01', '2026-10-13', slots=2, sizing='fixed',
                                 maxpos=.45, heat_cap=.10, gate_thr=.50, vol_target=.20,
                                 rank='random', hold=20, cost_rate=.0015, rank_seed=seed)
    rnd.append(dict(seed=seed, cagr=met['cagr'], dd=met['maxdd'], n=len(t)))
R = pd.DataFrame(rnd)
R.to_csv(os.path.join(DIR, '优先级_随机零分布.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print('\nrandom null')
print(R['cagr'].describe().to_string(float_format=lambda v: f'{v:.4f}'))
print('mcap 32.2 pct vs random', (R['cagr'] < 0.322).mean())
