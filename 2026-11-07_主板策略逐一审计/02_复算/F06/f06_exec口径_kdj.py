# -*- coding: utf-8 -*-
"""
F06 补充：按旧研究执行口径复核 KDJ 结论（主板）。
fwd_exec = signal_close[t+20] / signal_open[t+1] - 1；
剔除 T+1 停牌与一字涨停（实际价 high==low 且 open>prev_close）；
基准 = 同日主板非ST、warm120、可执行股票等权（leave-one-out）。
用于核对“MACD多+KDJ多 -0.65%”与“同日双金叉 -1.25%、0/17年正”。
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter
import os

BASE = '/Users/admin/WorkSpace/trading'
OUT = os.path.join(BASE, '2026-11-07_主板策略逐一审计/02_复算/F06')
PANEL = os.path.join(BASE, 'trading_finally/data/main_panel.parquet')

cols = ['date', 'code', 'signal_close', 'signal_open', 'signal_high', 'signal_low',
        'open', 'high', 'low', 'prev_close', 'is_st', 'trade_status', 'data_valid']
df = pd.read_parquet(PANEL, columns=cols).sort_values(['code', 'date']).reset_index(drop=True)

def ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False, ignore_na=True).mean().to_numpy()

def kdj(h, l, c, n=9):
    hs = pd.Series(h).rolling(n, min_periods=n).max().to_numpy()
    ls = pd.Series(l).rolling(n, min_periods=n).min().to_numpy()
    rng = hs - ls
    rsv = np.where(rng > 0, (c - ls) / np.where(rng > 0, rng, 1) * 100, 50.0)
    rsv = np.where(np.isnan(rng), 50.0, rsv)
    k = lfilter([1/3], [1, -2/3], rsv, zi=[50*2/3])[0]
    d = lfilter([1/3], [1, -2/3], k, zi=[50*2/3])[0]
    return k, d

parts = []
for code, g in df.groupby('code', sort=False):
    c = g.signal_close.to_numpy(dtype='float64')
    n = len(c)
    dif = ema(c, 12) - ema(c, 26)
    dea = ema(dif, 9)
    k, d = kdj(g.signal_high.to_numpy(), g.signal_low.to_numpy(), c)
    fwd = np.full(n, np.nan)
    if n > 20:
        fwd[:n-20] = c[20:] / g.signal_open.to_numpy()[1:n-19] - 1
    gc = np.zeros(n, bool); gc[1:] = (dif[1:] > dea[1:]) & (dif[:-1] <= dea[:-1])
    kgc = np.zeros(n, bool); kgc[1:] = (k[1:] > d[1:]) & (k[:-1] <= d[:-1])
    # T+1 可执行：交易且非一字涨停
    ts = g.trade_status.to_numpy(); hi = g.high.to_numpy(); lo = g.low.to_numpy()
    op = g.open.to_numpy(); pc = g.prev_close.to_numpy()
    tradable = (ts == 1) & ~((hi == lo) & (op > pc))
    exec_ok = np.zeros(n, bool); exec_ok[:n-1] = tradable[1:]
    parts.append(pd.DataFrame({
        'fwd_exec': fwd.astype('float32'), 'exec_ok': exec_ok,
        'macd_up': dif > dea, 'kdj_up': k > d, 'gc': gc, 'kgc': kgc,
        'bars': np.arange(n, dtype='int32'),
    }))
ind = pd.concat(parts, ignore_index=True)
df = pd.concat([df[['date', 'code', 'is_st', 'trade_status', 'data_valid']], ind], axis=1)
del parts, ind

dv = df.data_valid.to_numpy(); code_arr = df.code.to_numpy()
ok_end = np.zeros(len(df), bool); ok_end[:len(df)-20] = dv[20:]
same = np.zeros(len(df), bool); same[:len(df)-20] = code_arr[20:] == code_arr[:-20]
df['univ'] = (df.data_valid & (df.is_st == 0) & df.exec_ok & ok_end & same
              & np.isfinite(df.fwd_exec) & (df.bars >= 120) & (df.trade_status == 1)
              & (df.date >= '2010-01-01'))
u = df.loc[df.univ, ['date', 'fwd_exec']].copy()
gsum = u.groupby('date').fwd_exec.transform('sum')
gcnt = u.groupby('date').fwd_exec.transform('count')
u['ex'] = u.fwd_exec - (gsum - u.fwd_exec) / (gcnt - 1)
df['ex'] = np.nan
df.loc[u.index, 'ex'] = u.ex
st = df.loc[df.univ].copy()
st['year'] = st.date.str[:4]

rows = []
def add(label, e):
    daily = e.groupby('date').ex.mean()
    yearly = daily.groupby(daily.index.str[:4]).mean()
    rows.append({'类别': label, '样本数': len(e), '交易日': e.date.nunique(),
                 '日均毛超额': daily.mean(), '日均净超额': daily.mean() - 0.0015,
                 '信号加权毛超额': e.ex.mean(), '信号加权净超额': e.ex.mean() - 0.0015,
                 '中位超额': e.ex.median(),
                 '日度t_未NW_净': (daily.mean()-0.0015)/(daily.std()/np.sqrt(len(daily))),
                 '年数': len(yearly), '正年数_净': int((yearly > 0).sum())})

add('全部MACD金叉', st[st.gc])
add('同日双金叉', st[st.gc & st.kgc])
add('非同日KDJ金叉的MACD金叉', st[st.gc & ~st.kgc])
for mu, ml in [(True, 'MACD多'), (False, 'MACD空')]:
    for ku, kl in [(True, 'KDJ多'), (False, 'KDJ空')]:
        add(ml + '+' + kl, st[(st.macd_up == mu) & (st.kdj_up == ku)])
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'f06_exec口径_金叉与四态.csv'), index=False)
print(res.to_string())
print('DONE')