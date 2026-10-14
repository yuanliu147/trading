#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# chan_diag_A_check.py — 复核 A 组（防狼术）：
#   均值收益之外，检查 0轴状态对尾部风险（左尾/深亏概率）的区分度
#   若"双下0轴"未来均值更高但左尾更肥，则防狼术在风险维度仍有依据
import duckdb
import numpy as np
import pandas as pd

DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'

con = duckdb.connect(DB, read_only=True)
df = con.execute("""
    SELECT v.thscode, v.date, v.close
    FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
    WHERE s.exchange <> 'BJ'
    ORDER BY v.thscode, v.date
""").df()
con.close()
df['thscode'] = df['thscode'].astype('category')
df['close'] = df['close'].astype('float32')
g = df.groupby('thscode', observed=True)
df['ema12'] = g['close'].transform(lambda s: s.ewm(span=12, adjust=False).mean())
df['ema26'] = g['close'].transform(lambda s: s.ewm(span=26, adjust=False).mean())
df['dif'] = df['ema12'] - df['ema26']
df.drop(columns=['ema12', 'ema26'], inplace=True)
g = df.groupby('thscode', observed=True)
df['dea'] = g['dif'].transform(lambda s: s.ewm(span=9, adjust=False).mean())
df['age'] = g.cumcount()
for k in (1, 21):
    df[f'c{k}'] = g['close'].transform(lambda s, k=k: s.shift(-k))
df['fwd20'] = (df['c21'] / df['c1'] - 1)

univ = df[df['age'] >= 250]
dn = np.sort(df['date'].unique())
sample = set(dn[::5])
A = univ[np.isin(univ['date'].values, list(sample)) & univ['dif'].notna()].copy()
A['state'] = np.select(
    [(A['dif'] > 0) & (A['dea'] > 0), (A['dif'] < 0) & (A['dea'] < 0)],
    ['双上0轴', '双下0轴'], default='跨0轴')
A['era'] = pd.cut(pd.to_datetime(A['date']),
                  [pd.Timestamp('2016-10-01'), pd.Timestamp('2019-12-31'),
                   pd.Timestamp('2022-12-31'), pd.Timestamp('2030-01-01')],
                  labels=['E1', 'E2', 'E3'])
A = A[A['fwd20'].notna()]

print("== 各状态 fwd20 分布特征（尾部风险视角）==")
r = A.groupby(['era', 'state'], observed=True)['fwd20'].agg(
    n='size', mean='mean', median='median',
    std='std',
    p_lev_neg=lambda x: (x <= 0).mean(),
    p_dn15=lambda x: (x <= -0.15).mean(),
    p_dn25=lambda x: (x <= -0.25).mean(),
    p_up15=lambda x: (x >= 0.15).mean(),
    q05=lambda x: x.quantile(0.05),
    q95=lambda x: x.quantile(0.95)).round(4)
print(r.to_string())

print("\n== 全样本（不分时代）==")
r2 = A.groupby('state', observed=True)['fwd20'].agg(
    n='size', mean='mean', median='median',
    p_dn15=lambda x: (x <= -0.15).mean(), p_dn25=lambda x: (x <= -0.25).mean(),
    p_up15=lambda x: (x >= 0.15).mean()).round(4)
print(r2.to_string())

print("\n== 读数说明 ==")
print("若 双下0轴 的 p_dn15/p_dn25 显著高于 双上0轴，则防狼术在'避开深亏'维度仍有统计学依据；")
print("若同时其 mean/median 也更高，则该规则是'用收益换安心'，与当下市场特性错配。")
