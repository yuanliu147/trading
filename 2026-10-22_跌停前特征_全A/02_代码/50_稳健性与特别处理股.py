# -*- coding: utf-8 -*-
"""
稳健性：分年、分板块、危机年/平静年、退市近似样本、5% ST 跌停。
"""
import os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
ROOT = os.path.join(BASE, '2026-10-22_跌停前特征_全A')
OUT = os.path.join(ROOT, '03_结果')
t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:7.1f}s] {msg}', flush=True)

cols = ['date', 'code_id', 'board_id', 'approx', 'ld1', 'ldt1', 'ld5', 'st_proxy', 'is_new',
        'ret20', 'vol20', 'dist_ma20', 'dist_ma60', 'amt_ratio20', 'lu_yesterday',
        'ld_yesterday', 'consec_lu', 'ld_count20', 'market_breadth20', 'mkt_below_ma60',
        'mkt_ld_share']
log('读取扁平表')
df = pd.read_parquet(os.path.join(ROOT, '01_数据', '分析面板.parquet'), columns=cols)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
main = (df['st_proxy'].fillna(0) == 0) & (df['approx'] == 0) & (df['is_new'] == 0)
base_all = df.loc[main, 'ld1'].mean()
log(f'主样本 {main.sum():,}，基准 {base_all:.4%}')

conditions = {
    '昨收跌停': df['ld_yesterday'] >= 1,
    '昨收涨停': df['lu_yesterday'] >= 1,
    '≥2连板': df['consec_lu'] >= 2,
    '20日涨≥20%': df['ret20'] >= 0.20,
    '距20日线≥10%': df['dist_ma20'] >= 0.10,
    '20日波动≥50%': df['vol20'] >= 0.50,
    '20日内曾跌停': df['ld_count20'] >= 1,
    '市场跌停占比1~2%': (df['mkt_ld_share'] >= 0.01) & (df['mkt_ld_share'] < 0.02),
    '市场宽度20~30%': (df['market_breadth20'] >= 0.20) & (df['market_breadth20'] < 0.30),
    '高位加速+弱市场': (df['ret20'] >= 0.20) & (df['dist_ma60'] >= -0.03) &
                      (df['market_breadth20'] < 0.40) & (df['mkt_below_ma60'] >= 0.5),
}

# ---------------- 1. 分年 ----------------
log('分年稳健性')
rows = []
for name, mask in conditions.items():
    mask = mask.fillna(False)
    for year, g in df.loc[main].groupby('year'):
        m = mask.reindex(g.index).fillna(False)
        n = int(m.sum()); k = int(g.loc[m, 'ld1'].sum())
        if n < 100:
            continue
        base_y = g['ld1'].mean()
        rows.append({'条件': name, '年份': int(year), '样本数': n, '跌停数': k,
                     '概率': k / n, '当年基准': base_y, 'lift': (k / n) / base_y})
by_year = pd.DataFrame(rows)
by_year.to_csv(os.path.join(OUT, '稳健性_关键条件分年.csv'), index=False, encoding='utf-8-sig')

# ---------------- 2. 分板块 ----------------
log('分板块稳健性')
df['板块期间'] = np.select(
    [df['board_id'] == 1, (df['board_id'] == 2) & (df['date'] < '2020-08-24'),
     (df['board_id'] == 2) & (df['date'] >= '2020-08-24'), df['board_id'] == 3],
    ['主板', '创业板-10%时代', '创业板-20%时代', '科创板'], default='其他')
rows = []
for seg, g in df.loc[main].groupby('板块期间'):
    base_seg = g['ld1'].mean()
    rows.append({'板块期间': seg, '条件': '基准', '样本数': len(g), '跌停数': int(g['ld1'].sum()),
                 '概率': base_seg, '板块基准': base_seg, 'lift': 1.0})
    for name, mask in conditions.items():
        m = mask.reindex(g.index).fillna(False)
        n = int(m.sum()); k = int(g.loc[m, 'ld1'].sum())
        if n < 100:
            continue
        rows.append({'板块期间': seg, '条件': name, '样本数': n, '跌停数': k,
                     '概率': k / n, '板块基准': base_seg, 'lift': (k / n) / base_seg})
by_board = pd.DataFrame(rows)
by_board.to_csv(os.path.join(OUT, '稳健性_分板块.csv'), index=False, encoding='utf-8-sig')

# ---------------- 3. 危机年 vs 平静年 ----------------
log('危机年/平静年')
df['年景'] = np.select(
    [df['year'].isin([2018, 2024]), df['year'].isin([2019, 2021, 2023, 2025])],
    ['危机年（2018/2024）', '平静年（2019/2021/2023/2025）'], default='其他年份')
rows = []
for seg, g in df.loc[main].groupby('年景'):
    base_seg = g['ld1'].mean()
    rows.append({'年景': seg, '条件': '基准', '样本数': len(g), '跌停数': int(g['ld1'].sum()),
                 '概率': base_seg, '段内基准': base_seg, 'lift': 1.0})
    for name, mask in conditions.items():
        m = mask.reindex(g.index).fillna(False)
        n = int(m.sum()); k = int(g.loc[m, 'ld1'].sum())
        if n < 100:
            continue
        rows.append({'年景': seg, '条件': name, '样本数': n, '跌停数': k,
                     '概率': k / n, '段内基准': base_seg, 'lift': (k / n) / base_seg})
regime = pd.DataFrame(rows)
regime.to_csv(os.path.join(OUT, '稳健性_危机年平静年.csv'), index=False, encoding='utf-8-sig')

# ---------------- 4. 退市近似样本 ----------------
log('退市近似样本')
approx = df[(df['approx'] == 1) & (df['is_new'] == 0) & (df['board_id'] != 4)].copy()
rows = []
if len(approx):
    base_a = approx['ld1'].mean()
    rows.append({'条件': '基准', '样本数': len(approx), '跌停数': int(approx['ld1'].sum()),
                 '概率': base_a, 'lift': 1.0})
    for name, mask in conditions.items():
        m = mask.reindex(approx.index).fillna(False)
        n = int(m.sum()); k = int(approx.loc[m, 'ld1'].sum())
        if n < 50:
            continue
        rows.append({'条件': name, '样本数': n, '跌停数': k, '概率': k / n,
                     'lift': (k / n) / base_a})
pd.DataFrame(rows).to_csv(os.path.join(OUT, '稳健性_退市近似样本.csv'), index=False, encoding='utf-8-sig')

# ---------------- 5. 分年 × 板块事件数 ----------------
ev = df.loc[main].groupby(['year', '板块期间']).agg(样本数=('ld1', 'size'), 跌停数=('ld1', 'sum')).reset_index()
ev['跌停率'] = ev['跌停数'] / ev['样本数']
ev.to_csv(os.path.join(OUT, '分年分板块基准率.csv'), index=False, encoding='utf-8-sig')

# ---------------- 6. 5% ST 跌停 ----------------
log('5% ST 跌停')
# 事件表 5% 标志
m = pd.read_parquet(os.path.join(ROOT, '01_数据', '跌停事件表_主.parquet'),
                    columns=['thscode', 'date', 'board', 'ld_seal5', 'lu_seal5'])
s = pd.read_parquet(os.path.join(ROOT, '01_数据', '跌停事件表_补充.parquet'),
                    columns=['thscode', 'date', 'board', 'ld_seal5', 'lu_seal5'])
ee = pd.concat([m, s], ignore_index=True)
ee = ee[(ee['board'] == '主板') & ((ee['ld_seal5']) | (ee['lu_seal5']))].copy()
ee['date'] = pd.to_datetime(ee['date'])
cm = pd.read_csv(os.path.join(ROOT, '01_数据', '代码表.csv')).set_index('code')['code_id'].to_dict()
ee['code_id'] = ee['thscode'].map(cm)
ee = ee[ee['code_id'].notna()].copy(); ee['code_id'] = ee['code_id'].astype(int)
dates_list = np.array(sorted(df['date'].unique()))
date_pos = {d: i for i, d in enumerate(dates_list)}
di = pd.Series(ee['date']).map(date_pos).values
ok = pd.notna(di)
D, C = len(dates_list), int(df['code_id'].max()) + 1
ld5_mat = np.zeros((D, C), dtype='int8'); lu5_mat = np.zeros((D, C), dtype='int8')
ld5_mat[di[ok].astype(int), ee['code_id'].values[ok]] = ee['ld_seal5'].values[ok].astype('int8')
lu5_mat[di[ok].astype(int), ee['code_id'].values[ok]] = ee['lu_seal5'].values[ok].astype('int8')
# 特征
ld5_y = np.zeros_like(ld5_mat); ld5_y[1:] = ld5_mat[:-1]
lu5_y = np.zeros_like(lu5_mat); lu5_y[1:] = lu5_mat[:-1]
ld5_c20 = pd.DataFrame(ld5_mat).rolling(20, min_periods=1).sum().values
# ST 样本：主板、st_proxy=1、非近似、非次新
st = df[(df['board_id'] == 1) & (df['st_proxy'].fillna(0) == 1) &
        (df['approx'] == 0) & (df['is_new'] == 0)].copy()
st['di'] = st['date'].map(date_pos)
st = st[st['di'].notna()].copy(); st['di'] = st['di'].astype(int)
st['ld5_label'] = ld5_mat[st['di'].values, st['code_id'].values]
st['ld5_y'] = ld5_y[st['di'].values, st['code_id'].values]
st['lu5_y'] = lu5_y[st['di'].values, st['code_id'].values]
st['ld5_c20'] = ld5_c20[st['di'].values, st['code_id'].values]
base_st = st['ld5_label'].mean()
log(f'ST/5%样本 {len(st):,}，5%跌停 {int(st.ld5_label.sum()):,}，基准 {base_st:.4%}')
st_conds = {
    '昨日5%跌停': st['ld5_y'] >= 1,
    '昨日5%涨停': st['lu5_y'] >= 1,
    '20日内5%跌停≥1': st['ld5_c20'] >= 1,
    '20日涨≥20%': st['ret20'] >= 0.20,
    '20日波动≥50%': st['vol20'] >= 0.50,
    '距20日线≥10%': st['dist_ma20'] >= 0.10,
    '市场跌停占比1~2%': (st['mkt_ld_share'] >= 0.01) & (st['mkt_ld_share'] < 0.02),
    '弱市场': (st['market_breadth20'] < 0.40) & (st['mkt_below_ma60'] >= 0.5),
}
rows = [{'条件': '基准', '样本数': len(st), '5%跌停数': int(st.ld5_label.sum()),
         '概率': base_st, 'lift': 1.0}]
for name, mask in st_conds.items():
    mask = mask.fillna(False)
    n = int(mask.sum()); k = int(st.loc[mask, 'ld5_label'].sum())
    if n == 0:
        continue
    rows.append({'条件': name, '样本数': n, '5%跌停数': k, '概率': k / n, 'lift': (k / n) / base_st})
pd.DataFrame(rows).to_csv(os.path.join(OUT, '特别处理股5%跌停概率.csv'), index=False, encoding='utf-8-sig')
# 5%事件分年
ev5 = ee[ee['ld_seal5']].copy()
ev5['year'] = ev5['date'].dt.year
ev5.groupby('year').agg(事件数=('thscode', 'size'), 股票数=('thscode', 'nunique')).reset_index().to_csv(
    os.path.join(OUT, '特别处理股5%跌停分年.csv'), index=False, encoding='utf-8-sig')
ev5[['date', 'thscode', 'code_id', 'ld_seal5', 'lu_seal5']].to_csv(
    os.path.join(OUT, '特别处理股5%跌停事件清单.csv'), index=False, encoding='utf-8-sig')

log('完成')
