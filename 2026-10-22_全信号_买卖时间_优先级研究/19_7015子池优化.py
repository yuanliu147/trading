# -*- coding: utf-8 -*-
"""7000万+15元 子池：单笔1万、不限仓位下的再优化分析。"""
import os
import numpy as np
import pandas as pd

DIR = '/Users/admin/WorkSpace/trading/2026-10-22_全信号_买卖时间_优先级研究'
T = pd.read_csv(os.path.join(DIR, '全信号_一万_逐笔.csv'), encoding='utf-8-sig')
X = T[(T['成交额'] <= 70e6) & (T['股价'] <= 15)].copy()
X['期'] = np.where(X['年份'] <= 2021, '训练2018-2021', '检验2022-2026')

def summ(z):
    x = z['止损后净盈亏'].dropna()
    if len(x) == 0:
        return dict(笔数=0, 总净盈亏=np.nan, 平均每笔=np.nan, 中位每笔=np.nan, 胜率=np.nan, 盈亏比=np.nan, 前百分之一占比=np.nan)
    return dict(笔数=len(x), 总净盈亏=x.sum(), 平均每笔=x.mean(), 中位每笔=x.median(),
                胜率=(x > 0).mean(), 盈亏比=x[x > 0].sum() / max(1e-9, -x[x < 0].sum()),
                前百分之一占比=x.nlargest(max(1, int(len(x) * .01))).sum() / x.sum() if x.sum() > 0 else np.nan)

filters = {
    '基准70/15': pd.Series(True, index=X.index),
    '+市场宽度20≥0.60': X['市场宽度20'] >= 0.60,
    '+市场宽度20≥0.6766': X['市场宽度20'] >= 0.6766,
    '+市场宽度20≥0.8021': X['市场宽度20'] >= 0.8021,
    '+行业宽度≥0.9451': X['行业宽度'] >= 0.9451,
    '+市场20日≥0.1374': X['市场20日收益'] >= 0.1374,
    '+行业20日≥0.1432': X['行业20日收益'] >= 0.1432,
    '+市场宽度60≥0.9389': X['市场宽度60'] >= 0.9389,
    '+市场波动≥0.2036': X['市场波动'] >= 0.2036,
    '+成交额≤6000万+股价≤5': (X['成交额'] <= 60e6) & (X['股价'] <= 5),
}
rows = []
for name, m in filters.items():
    for per in ['训练2018-2021', '检验2022-2026', '全期']:
        z = X[m] if per == '全期' else X[m & (X['期'] == per)]
        r = summ(z); r['方案'] = name; r['期'] = per
        rows.append(r)
S = pd.DataFrame(rows)[['方案', '期', '笔数', '总净盈亏', '平均每笔', '中位每笔', '胜率', '盈亏比', '前百分之一占比']]
S.to_csv(os.path.join(DIR, '7015_子池汇总.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print(S[S['期'] != '全期'].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

features = ['趋势年龄', '真实波幅', '成交额', '距60日高点', '距20日均线', '距60日均线',
            '涨幅20', '涨幅60', '波动20', '换手异常值', '成交额比', '信号日跳空',
            '上影线', '收盘位置', '行业宽度', '行业20日收益', '个股行业相对强度', '行业相对强度',
            '自身基准', '自身胜率', '市值', '股价', '市场宽度20', '市场宽度60',
            '市场20日收益', '市场波动', '市场60日斜率']
corr = []
for f in features:
    for per in ['训练2018-2021', '检验2022-2026']:
        z = X[X['期'] == per]
        corr.append(dict(特征=f, 期=per, 相关=z[f].corr(z['止损后收益率'], method='spearman')))
C = pd.DataFrame(corr).pivot(index='特征', columns='期', values='相关').reset_index()
C['方向一致'] = np.sign(C['训练2018-2021']) == np.sign(C['检验2022-2026'])
C.to_csv(os.path.join(DIR, '7015_特征相关.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
print('\ncorr')
print(C.sort_values('方向一致', ascending=False).to_string(index=False, float_format=lambda v: f'{v:.3f}'))

# 5箱训练/检验
bins = []
for f in features:
    tr = X[X['期'] == '训练2018-2021'][f].dropna()
    if len(tr) < 300 or tr.nunique() < 5:
        continue
    edges = pd.qcut(tr, 5, duplicates='drop', retbins=True)[1]
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        mask = (X[f] >= lo) & ((X[f] <= hi) if b == len(edges) - 2 else (X[f] < hi))
        for per in ['训练2018-2021', '检验2022-2026']:
            z = X[mask & (X['期'] == per)]['止损后净盈亏']
            if len(z):
                bins.append(dict(特征=f, 分箱=f'{lo:.4g}~{hi:.4g}', 期=per, 笔数=len(z),
                                 平均净盈亏=z.mean(), 胜率=(z > 0).mean()))
B = pd.DataFrame(bins)
B.to_csv(os.path.join(DIR, '7015_特征分箱.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')

# 阈值前沿
frontier = []
for f, thrs in [('市场宽度20', [.50, .55, .60, .65, .6766, .70, .75, .80, .85]),
                ('行业宽度', [.50, .60, .70, .80, .85, .90, .9451]),
                ('市场20日收益', [0, .03, .05, .08, .10, .12, .1374]),
                ('行业20日收益', [0, .05, .08, .10, .12, .1432]),
                ('市场宽度60', [.30, .50, .60, .70, .80, .90, .9389])]:
    for th in thrs:
        for per in ['训练2018-2021', '检验2022-2026']:
            z = X[(X['期'] == per) & (X[f] >= th)]['止损后净盈亏']
            frontier.append(dict(特征=f, 阈值=th, 期=per, 笔数=len(z), 平均净盈亏=z.mean(),
                                 总净盈亏=z.sum(), 胜率=(z > 0).mean()))
F = pd.DataFrame(frontier)
F.to_csv(os.path.join(DIR, '7015_阈值前沿.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print('\nfrontier saved')
