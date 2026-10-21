# -*- coding: utf-8 -*-
"""成交量/换手率对全信号20日收益的关系（训练/检验）。"""
import os
import pickle
import numpy as np
import pandas as pd

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
DIR = '/Users/admin/WorkSpace/trading/2026-10-22_全信号_买卖时间_优先级研究'
P = pickle.load(open(os.path.join(BASE, '01_数据/全A面板_2016_2026_含退市.pkl'), 'rb'))
dates = pd.DatetimeIndex(P['meta']['dates']); codes = np.array(P['meta']['codes'])
code_idx = {c: i for i, c in enumerate(codes)}
FEATS = {
    '成交额': P['amt20'].values,
    '成交额比': P['amt_ratio20'].values,
    '换手率': P['turn'].values,
    '换手异常值': P['turn_z250'].values,
    '量比5_20': P['vr5_20'].values,
    '量价相关20': P['vp_corr20'].values,
    '成交额趋势': P['amt_trend'].values,
    '成交量历史分位': P['vol_hist_z'].values,
    '20日波动': P['vol20'].values,
}
T = pd.read_csv(os.path.join(DIR, '全信号_一万_逐笔.csv'), encoding='utf-8-sig', parse_dates=['首个信号日'])
T['ts'] = dates.searchsorted(T['首个信号日'])
T['j'] = T['代码'].map(code_idx)
T = T[T['j'].notna()].copy(); T['j'] = T['j'].astype(int)
for name, arr in FEATS.items():
    T[name] = arr[T['ts'].values, T['j'].values]
T['期'] = np.where(T['年份'] <= 2021, '训练2018-2021', '检验2022-2026')
T['收益'] = T['止损后收益率']

rows = []
for f in FEATS:
    x = T[[f, '期', '收益']].replace([np.inf, -np.inf], np.nan).dropna()
    tr = x[x['期'] == '训练2018-2021'][f]
    if len(tr) < 500 or tr.nunique() < 5: continue
    bins = pd.qcut(tr, 5, duplicates='drop', retbins=True)[1]
    for b in range(len(bins) - 1):
        lo, hi = bins[b], bins[b + 1]
        mask = (x[f] >= lo) & ((x[f] <= hi) if b == len(bins) - 2 else (x[f] < hi))
        for per in ['训练2018-2021', '检验2022-2026']:
            z = x[mask & (x['期'] == per)]
            if len(z):
                rows.append(dict(特征=f, 分箱=f'{lo:.4g}~{hi:.4g}', 期=per, 笔数=len(z),
                                 平均收益率=z['收益'].mean(), 胜率=(z['收益'] > 0).mean()))
B = pd.DataFrame(rows); B.to_csv(os.path.join(DIR, '量能换手_分箱.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')

corr = []
for f in FEATS:
    x = T[[f, '期', '收益']].replace([np.inf, -np.inf], np.nan).dropna()
    a = x[x['期'] == '训练2018-2021']; b = x[x['期'] == '检验2022-2026']
    corr.append(dict(特征=f, 训练相关=a[f].corr(a['收益'], method='spearman'),
                     检验相关=b[f].corr(b['收益'], method='spearman'),
                     训练笔数=len(a), 检验笔数=len(b)))
C = pd.DataFrame(corr).sort_values('训练相关')
C.to_csv(os.path.join(DIR, '量能换手_相关.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
print(C.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

base_tr = T[T['期'] == '训练2018-2021']['收益'].mean(); base_te = T[T['期'] == '检验2022-2026']['收益'].mean()
print('\nbase', base_tr, base_te)
for f in FEATS:
    d = B[B['特征'] == f]
    print('\n==', f)
    print(d.pivot_table(index='分箱', columns='期', values=['笔数', '平均收益率', '胜率']).to_string(float_format=lambda x: f'{x:.3f}'))
