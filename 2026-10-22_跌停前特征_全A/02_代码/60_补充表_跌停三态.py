# -*- coding: utf-8 -*-
"""
补充表：
  1) 昨日跌停三态（一字/非一字封死/收盘打开）-> 次日风险；
  2) 跌停日三态（收盘跌停一字/非一字封死/盘中打开）-> 次日与未来5日风险。
"""
import os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
ROOT = os.path.join(BASE, '2026-10-22_跌停前特征_全A')
OUT = os.path.join(ROOT, '03_结果')
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.1f}s] {m}', flush=True)

cols = ['date', 'code_id', 'board_id', 'ld1', 'ld5', 'ldt1', 'ld_yesterday',
        'one_word_ld_yesterday', 'opened_ld_yesterday', 'market_breadth20',
        'mkt_below_ma60', 'mkt_ld_share', 'st_proxy', 'approx', 'is_new']
df = pd.read_parquet(os.path.join(ROOT, '01_数据', '分析面板.parquet'), columns=cols)
df['date'] = pd.to_datetime(df['date'])
main = (df['st_proxy'].fillna(0) == 0) & (df['approx'] == 0) & (df['is_new'] == 0)
df = df.loc[main].copy(); df['year'] = df['date'].dt.year
base = df['ld1'].mean()
log(f'主样本 {len(df):,}，基准 {base:.4%}')

def cluster_ci(mask, B=1000):
    sub = df.loc[mask, ['date', 'ld1']]
    g = sub.groupby('date').agg(n=('ld1', 'size'), k=('ld1', 'sum'))
    n = g['n'].values.astype(float); k = g['k'].values.astype(float)
    rng = np.random.default_rng(20261022); nd = len(n); r = np.empty(B)
    for i in range(B):
        j = rng.integers(0, nd, nd); r[i] = k[j].sum() / max(n[j].sum(), 1)
    return np.percentile(r, 2.5), np.percentile(r, 97.5)

# ---------------- 1. 昨日三态 ----------------
log('昨日跌停三态')
states = {
    '一字跌停': (df['ld_yesterday'] >= 1) & (df['one_word_ld_yesterday'] >= 1),
    '非一字封死': (df['ld_yesterday'] >= 1) & (df['one_word_ld_yesterday'] == 0),
    '收盘打开': (df['ld_yesterday'] == 0) & (df['opened_ld_yesterday'] >= 1),
    '昨日无跌停': (df['ld_yesterday'] == 0) & (df['opened_ld_yesterday'] == 0),
}
rows = []
for name, m in states.items():
    m = m.fillna(False)
    n = int(m.sum()); k = int(df.loc[m, 'ld1'].sum())
    lo, hi = cluster_ci(m)
    tr = m & (df['date'] < '2022-01-01'); te = m & (df['date'] >= '2022-01-01')
    rows.append({'昨日状态': name, '样本数': n, '跌停数': k, 'P(次日跌停)': k / n,
                 'lift': (k / n) / base, 'cluster下限': lo, 'cluster上限': hi,
                 'P(5日内跌停)': df.loc[m, 'ld5'].mean(), 'P(盘中触及)': df.loc[m, 'ldt1'].mean(),
                 '训练期概率': df.loc[tr, 'ld1'].mean(), '检验期概率': df.loc[te, 'ld1'].mean()})
t3 = pd.DataFrame(rows)
t3.to_csv(os.path.join(OUT, '昨日跌停三态_次日风险.csv'), index=False, encoding='utf-8-sig')
print(t3.to_string(index=False, float_format=lambda z: f'{z:.4g}'))

# ---------------- 2. 跌停日三态（含打开） ----------------
log('跌停日三态（含打开）')
m = pd.read_parquet(os.path.join(ROOT, '01_数据', '跌停事件表_主.parquet'),
                    columns=['thscode', 'date', 'board', 'ld_ratio', 'ldt_ratio', 'high', 'low', 'close'])
s = pd.read_parquet(os.path.join(ROOT, '01_数据', '跌停事件表_补充.parquet'),
                    columns=['thscode', 'date', 'board', 'ld_ratio', 'ldt_ratio', 'high', 'low', 'close'])
e = pd.concat([m, s], ignore_index=True); e['date'] = pd.to_datetime(e['date'])
e = e[(e['board'] != '北交所') & ((e['ld_ratio'] > 0) | (e['ldt_ratio'] > 0))].copy()
e['类别'] = np.select(
    [(e['ld_ratio'] > 0) & (np.abs(e['high'] - e['low']) < 1e-9),
     (e['ld_ratio'] > 0) & (np.abs(e['high'] - e['low']) >= 1e-9),
     (e['ld_ratio'].fillna(0) == 0) & (e['ldt_ratio'] > 0)],
    ['一字跌停', '非一字封死', '盘中打开'], default='其他')
cm = pd.read_csv(os.path.join(ROOT, '01_数据', '代码表.csv')).set_index('code')['code_id'].to_dict()
e['code_id'] = e['thscode'].map(cm); e = e[e['code_id'].notna()].copy(); e['code_id'] = e['code_id'].astype(int)
# 只保留主样本的非ST事件：用扁平表 T-1 的 st_proxy / is_new
info = df[['date', 'code_id', 'st_proxy']]
e = e.merge(info, on=['date', 'code_id'], how='inner', suffixes=('', '_y'))
e = e[(e['st_proxy'].fillna(0) == 0)]
# 下一日标签
dates = np.array(sorted(df['date'].unique())); dpos = {d: i for i, d in enumerate(dates)}
e['di'] = e['date'].map(dpos); e = e[e['di'].notna()].copy(); e['di'] = e['di'].astype(int)
pos = e['di'].values + 1; ok = (pos >= 0) & (pos < len(dates))
sub = e.loc[ok, ['code_id', '类别', 'date']].copy(); sub['target'] = dates[pos[ok]]
look = df.set_index(['date', 'code_id'])[['ld1', 'ldt1', 'ld5']]
idx = pd.MultiIndex.from_frame(sub[['target', 'code_id']])
nxt = look.reindex(idx).reset_index(drop=True)
sub = sub.reset_index(drop=True)
sub['次日跌停'] = nxt['ld1'].values; sub['次日触及'] = nxt['ldt1'].values; sub['未来5日跌停'] = nxt['ld5'].values
rows = []
for name in ['一字跌停', '非一字封死', '盘中打开']:
    x = sub[sub['类别'] == name]
    rows.append({'跌停日类型': name, '事件数': len(x), '次日跌停率': x['次日跌停'].mean(),
                 '次日触及率': x['次日触及'].mean(), '未来5日跌停率': x['未来5日跌停'].mean()})
x = sub
rows.append({'跌停日类型': '全部', '事件数': len(x), '次日跌停率': x['次日跌停'].mean(),
             '次日触及率': x['次日触及'].mean(), '未来5日跌停率': x['未来5日跌停'].mean()})
post3 = pd.DataFrame(rows)
post3.to_csv(os.path.join(OUT, '跌停日三态_次日风险.csv'), index=False, encoding='utf-8-sig')
print(post3.to_string(index=False, float_format=lambda z: f'{z:.4g}'))
log('完成')
