# -*- coding: utf-8 -*-
"""用补全后的换手率缓存重建全覆盖市值矩阵（与原面板同公式：MCAP = 成交额/(换手率/100)）。

- 输入：面板（对齐 dates×codes）+ turnover_akshare/*.pkl（turn 单位 %）
- 对账：原面板覆盖的约 1,280 只上，新矩阵与原 mcap 应完全一致
- 输出：市值矩阵_全覆盖.pkl（DataFrame，float32，与面板对齐）
"""
import os, pickle, glob
import numpy as np
import pandas as pd

BASE = '/Users/admin/WorkSpace/trading'
PANEL = os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl')
TURN_DIR = os.path.join(BASE, '2026-10-18_成交量融合_2010-2025/01_数据/turnover_akshare')
OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化/市值矩阵_全覆盖.pkl'

P = pickle.load(open(PANEL, 'rb'))
codes = P['meta']['codes']
dates = pd.DatetimeIndex(P['meta']['dates'])
A = P['amount'].values.astype('float64')   # 面板成交额
old = P['mcap'].values.astype('float64')   # 原面板市值（约1,280只覆盖）

def to_file(c):
    a, b = c.split('.')
    return b.lower() + '.' + a

TURN = np.full((len(dates), len(codes)), np.nan, dtype='float64')
have = 0
for j, c in enumerate(codes):
    fp = os.path.join(TURN_DIR, to_file(c) + '.pkl')
    if not os.path.exists(fp):
        continue
    x = pickle.load(open(fp, 'rb'))
    s = pd.Series(x['turn'].values, index=pd.DatetimeIndex(x['date']))
    s = s.reindex(dates)
    TURN[:, j] = s.values
    have += 1
print(f'换手率覆盖 {have}/{len(codes)} 只')

with np.errstate(all='ignore'):
    MCAP = A / (TURN / 100.0)
MCAP[~np.isfinite(MCAP)] = np.nan

# 对账：原面板覆盖部分
mask_old = np.isfinite(old)
both = mask_old & np.isfinite(MCAP)
print(f'原覆盖格子 {mask_old.sum():,}，新矩阵覆盖其中 {both.sum():,}')
if both.sum():
    rel = np.abs(MCAP[both] / old[both] - 1)
    print(f'对账相对误差：中位 {np.median(rel):.2e}，p99 {np.quantile(rel, 0.99):.2e}，'
          f'>1% 的比例 {(rel > 0.01).mean():.4%}')

cov = np.isfinite(MCAP)
print(f'新矩阵覆盖率：股票 {(cov.any(axis=0)).mean():.1%}，格子 {cov.mean():.1%}')
M = pd.DataFrame(MCAP.astype('float32'), index=dates, columns=codes)
pickle.dump(M, open(OUT, 'wb'), protocol=4)
print('saved', OUT, os.path.getsize(OUT) / 1e6, 'MB')

# 缺失名单（多为退市股）
missing = [c for j, c in enumerate(codes) if not cov[:, j].any()]
print(f'完全无市值数据 {len(missing)} 只:', missing[:10])
