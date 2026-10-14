#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# chan_diag_macro.py — 市场结构变迁年度统计（缠论生态假设的宏观背景）
# 产出：每年 波动/离散度/涨跌停占比/MA150上方占比/成交额中位/等权年收益与回撤/平均成对相关/沪深300年收益
# 附加：沪深300为远端 CLI 拉取（market history），其余由本地 DuckDB 全A前复权面板计算
import json

import duckdb
import numpy as np
import pandas as pd

DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'
OUT = '/Users/admin/WorkSpace/trading/analysis/'
HS300_JSON = OUT + 'chan_diag_hs300_raw.json'

con = duckdb.connect(DB, read_only=True)
df = con.execute("""
    SELECT v.thscode, v.date, v.close, v.amount
    FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
    WHERE s.exchange <> 'BJ'
    ORDER BY v.thscode, v.date
""").df()
con.close()
df['thscode'] = df['thscode'].astype('category')
df['close'] = df['close'].astype('float32')
g = df.groupby('thscode', observed=True)
df['ret1'] = g['close'].transform(lambda s: s.pct_change())
df['ma150'] = g['close'].transform(lambda s: s.rolling(150, min_periods=150).mean())
df['above150'] = (df['close'] > df['ma150'])

# 涨跌停近似（板块幅度规则；ST 5% 幅度与新股无涨跌幅期未建模，为近似口径）
cats = df['thscode'].cat.categories
pref = np.array([str(c).split('.')[0][:3] for c in cats])
is_cyb = np.isin(pref, ['300', '301'])
is_star = np.isin(pref, ['688', '689'])
codes = df['thscode'].cat.codes.values
dates = df['date'].values
switch = np.datetime64('2020-08-24')
lim = np.where(is_cyb[codes], np.where(dates >= switch, 0.20, 0.10),
               np.where(is_star[codes], 0.20, 0.10)).astype('float32')
df['lu'] = (df['ret1'] >= lim - 0.002)
df['ld'] = (df['ret1'] <= -(lim - 0.002))

print("[macro] 日度聚合 ...", flush=True)
day = df.groupby('date').agg(
    n=('ret1', 'size'),
    ew_ret=('ret1', 'mean'),
    med_abs=('ret1', lambda x: x.abs().median()),
    xs_std=('ret1', 'std'),
    lu_r=('lu', 'mean'),
    ld_r=('ld', 'mean'),
    ab150=('above150', 'mean'),
    med_amt=('amount', 'median'))
day['year'] = day.index.year

rows = []
for y, gd in day.groupby('year'):
    er = gd['ew_ret'].astype(float)
    cum = (1 + er).cumprod()
    dd = float((cum / cum.cummax() - 1).min())
    rows.append(dict(year=int(y), days=len(gd), n_avg=int(gd['n'].mean()),
                     ew_ann=float(cum.iloc[-1] - 1),
                     ew_vol=float(er.std() * np.sqrt(252)), ew_maxdd=dd,
                     med_abs=float(gd['med_abs'].mean()), xs_std=float(gd['xs_std'].mean()),
                     lu_r=float(gd['lu_r'].mean()), ld_r=float(gd['ld_r'].mean()),
                     ab150=float(gd['ab150'].mean()),
                     med_amt_yi=float(gd['med_amt'].mean() / 1e8)))
Y = pd.DataFrame(rows).set_index('year')

# 平均成对相关（每年抽 <=300 只全年度覆盖的股票）
print("[macro] 年度相关性 ...", flush=True)
cov = df.groupby('thscode', observed=True)['date'].agg(['min', 'max'])
sub = df[['thscode', 'date', 'ret1']].copy()
sub['year'] = sub['date'].dt.year
corr_rows = {}
for y, gy in sub.groupby('year'):
    fdate, ldate = gy['date'].min(), gy['date'].max()
    elig = cov.index[(cov['min'] <= fdate) & (cov['max'] >= ldate)]
    if len(elig) < 50:
        continue
    rng = np.random.default_rng(2026 + int(y))
    pick = elig if len(elig) <= 300 else rng.choice(np.array(elig), 300, replace=False)
    gy2 = gy[gy['thscode'].isin(set(pick))].copy()
    gy2['thscode'] = gy2['thscode'].astype(str)
    piv = gy2.pivot_table(index='date', columns='thscode', values='ret1', observed=False).dropna(axis=1)
    cm = np.corrcoef(piv.values.T)
    n = cm.shape[0]
    if n > 2:
        mask = ~np.eye(n, dtype=bool)
        corr_rows[y] = float(cm[mask].mean())
Y['pair_corr'] = pd.Series(corr_rows)

# 沪深300（远端 CLI 拉取，date_ms 为北京时间 0 点的 UTC 表示 → +8 换算回北京时间日期）
hs = json.load(open(HS300_JSON))
h = pd.DataFrame(hs['data']['item'])
h['date'] = (pd.to_datetime(h['date_ms'], unit='ms', utc=True)
             .dt.tz_convert('Asia/Shanghai').dt.normalize().dt.tz_localize(None))
h = h.sort_values('date').reset_index(drop=True)
last_by_year = h.groupby(h['date'].dt.year)['close_price'].last()
first = h['close_price'].iloc[0]
ann = {}
prev = None
for yv, lc in last_by_year.items():
    ann[int(yv)] = float(lc / first - 1) if prev is None else float(lc / prev - 1)
    prev = lc
Y['hs300_ann'] = pd.Series(ann)
Y['ew_minus_300'] = Y['ew_ann'] - Y['hs300_ann']

Y = Y.round(4)
pd.set_option('display.width', 200)
print("\n===== 年度市场结构表（2016 与 2026 为部分年度）=====")
print(Y.to_string())
Y.to_csv(OUT + 'chan_diag_macro_yearly.csv')
print(f"\n[macro] 输出: chan_diag_macro_yearly.csv | 等权指数覆盖 "
      f"{day.index.min().date()} ~ {day.index.max().date()}", flush=True)
