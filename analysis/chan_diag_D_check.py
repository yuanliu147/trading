#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# chan_diag_D_check.py — 复核 D 组（中枢震荡）：
#   1) 池化(stock-day)口径 vs 月度等权口径的差异来源
#   2) 5/10/20 日三个视野的方向是否一致
#   3) 保存月度扩散序列供报告引用
import duckdb
import numpy as np
import pandas as pd

DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'
OUT = '/Users/admin/WorkSpace/trading/analysis/'


def nw_t(x, lags=None):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    v = (e * e).mean()
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        v += 2 * w * (e[l:] * e[:-l]).mean()
    return x.mean() / np.sqrt(v / n) if v > 0 else np.nan


con = duckdb.connect(DB, read_only=True)
df = con.execute("""
    SELECT v.thscode, v.date, v.high, v.low, v.close
    FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
    WHERE s.exchange <> 'BJ'
    ORDER BY v.thscode, v.date
""").df()
con.close()
df['thscode'] = df['thscode'].astype('category')
for c in ['high', 'low', 'close']:
    df[c] = df[c].astype('float32')
g = df.groupby('thscode', observed=True)
df['hh40'] = g['high'].transform(lambda s: s.rolling(40, min_periods=40).max().shift(1))
df['ll40'] = g['low'].transform(lambda s: s.rolling(40, min_periods=40).min().shift(1))
df['age'] = g.cumcount()
for k in (1, 6, 11, 21):
    df[f'c{k}'] = g['close'].transform(lambda s, k=k: s.shift(-k))
df['fwd5'] = (df['c6'] / df['c1'] - 1)
df['fwd10'] = (df['c11'] / df['c1'] - 1)
df['fwd20'] = (df['c21'] / df['c1'] - 1)

univ = df[df['age'] >= 250]
D = univ[(univ['hh40'].notna()) & ((univ['hh40'] / univ['ll40'] - 1) <= 0.25)].copy()
D['pos'] = (D['close'] - D['ll40']) / (D['hh40'] - D['ll40'])
D = D[D['pos'].notna() & D['fwd5'].notna()]
D['bucket'] = pd.cut(D['pos'], [0, 0.2, 0.4, 0.6, 0.8, 1.00001],
                     labels=['0-20', '20-40', '40-60', '60-80', '80-100'])
D['era'] = pd.cut(pd.to_datetime(D['date']),
                  [pd.Timestamp('2016-10-01'), pd.Timestamp('2019-12-31'),
                   pd.Timestamp('2022-12-31'), pd.Timestamp('2030-01-01')],
                  labels=['E1', 'E2', 'E3'])
D['ym'] = pd.to_datetime(D['date']).dt.to_period('M')

print("== 分时代 × 桶（stock-day 加权, fwd5/10/20）==")
r = D.groupby(['era', 'bucket'], observed=True).agg(
    n=('fwd5', 'size'), fwd5=('fwd5', 'mean'), fwd10=('fwd10', 'mean'),
    fwd20=('fwd20', 'mean')).round(4)
print(r.to_string())

print("\n== 池化(stock-day)口径：下沿-上沿 全样本 ==")
for col in ['fwd5', 'fwd10', 'fwd20']:
    dn = D[D['bucket'] == '0-20'][col].mean()
    up = D[D['bucket'] == '80-100'][col].mean()
    print(f"{col}: 下沿={dn:+.4%} 上沿={up:+.4%} 差(下-上)={dn - up:+.4%}")

print("\n== 月度口径对比（等权月均 vs 计数加权 vs 池化）==")
cnt = D.groupby(['ym', 'bucket'], observed=True).size().unstack().dropna()
tot = cnt.sum(axis=1)
for col in ['fwd5', 'fwd10', 'fwd20']:
    mmx = D.groupby(['ym', 'bucket'], observed=True)[col].mean().unstack().dropna()
    d = mmx['0-20'] - mmx['80-100']
    eq, wt = d.mean(), (d * tot).sum() / tot.sum()
    corr = np.corrcoef(tot.values, d.values)[0, 1]
    print(f"{col}: 等权月均={eq:+.4%} (NW-t={nw_t(d.values):+.2f}) | 计数加权月均={wt:+.4%} | "
          f"corr(月样本量,扩散)={corr:+.2f}")
    d.to_csv(OUT + f'chan_diag_D_monthly_{col}.csv')

print("\n== 月度扩散 |d| 最大10个月（含样本量）==")
mmx = D.groupby(['ym', 'bucket'], observed=True)['fwd5'].mean().unstack().dropna()
d = mmx['0-20'] - mmx['80-100']
tmp = pd.DataFrame({'d_fwd5': d, 'n_total': tot})
tmp = tmp.reindex(d.abs().sort_values(ascending=False).index)
print(tmp.head(10).round(4).to_string())

print("\n== 崩盘反弹月抽查（2024-01/02）==")
for ym in ['2024-01', '2024-02']:
    g = D[D['ym'] == str(ym)]
    if len(g):
        dn = g[g['bucket'] == '0-20']
        up = g[g['bucket'] == '80-100']
        print(f"{ym}: 下沿 n={len(dn)} fwd5={dn['fwd5'].mean():+.4%} fwd20={dn['fwd20'].mean():+.4%} | "
              f"上沿 n={len(up)} fwd5={up['fwd5'].mean():+.4%} fwd20={up['fwd20'].mean():+.4%}")
