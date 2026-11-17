# -*- coding: utf-8 -*-
"""
F06 独立复算：主板 MACD(12,26,9) 金叉事件 + 零轴分层 + KDJ(9,3,3) 四态/同日双金叉
只读 main_panel.parquet；输出仅写本目录。
口径（按审计任务定义，与旧研究的差异在报告中注明）：
- 价格：signal_close（后复权）；金叉=前一日 DIF<=DEA 且当日 DIF>DEA
- 事件：date>=2010-01-01，data_valid=True，is_st=0；fwd20=signal_close[t+20]/signal_close[t]-1（行移位，须有后续数据且当日有效）
- 横截面分位/超额基准：同日主板全部 data_valid、非ST、fwd20 可得股票（等权，超额 leave-one-out）
- 另报 warm120 口径（>=120 根K线且 trade_status=1，更接近旧研究）
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter
import os

BASE = '/Users/admin/WorkSpace/trading'
OUT = os.path.join(BASE, '2026-11-07_主板策略逐一审计/02_复算/F06')
PANEL = os.path.join(BASE, 'trading_finally/data/main_panel.parquet')

cols = ['date', 'code', 'signal_close', 'signal_high', 'signal_low',
        'is_st', 'trade_status', 'data_valid']
df = pd.read_parquet(PANEL, columns=cols)
df = df.sort_values(['code', 'date']).reset_index(drop=True)
print('rows', len(df), 'codes', df.code.nunique())

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
    h = g.signal_high.to_numpy(dtype='float64')
    l = g.signal_low.to_numpy(dtype='float64')
    dif = ema(c, 12) - ema(c, 26)
    dea = ema(dif, 9)
    k, d = kdj(h, l, c)
    n = len(c)
    fwd = np.full(n, np.nan)
    if n > 20:
        fwd[:n-20] = c[20:] / c[:n-20] - 1
    gc = np.zeros(n, bool); gc[1:] = (dif[1:] > dea[1:]) & (dif[:-1] <= dea[:-1])
    kgc = np.zeros(n, bool); kgc[1:] = (k[1:] > d[1:]) & (k[:-1] <= d[:-1])
    parts.append(pd.DataFrame({
        'fwd20': fwd.astype('float32'),
        'dif': dif.astype('float32'), 'dea': dea.astype('float32'),
        'gc': gc, 'macd_up': dif > dea,
        'k': k.astype('float32'), 'd': d.astype('float32'),
        'kdj_up': k > d, 'kgc': kgc,
        'bars': np.arange(n, dtype='int32'),
    }))
ind = pd.concat(parts, ignore_index=True)
df = pd.concat([df[['date', 'code', 'is_st', 'trade_status', 'data_valid']], ind], axis=1)
del parts, ind

# 后续第20行须 data_valid（避免以无效行为终点）
dv = df.data_valid.to_numpy()
ok_end = np.zeros(len(df), bool); ok_end[:len(df)-20] = dv[20:]
# 终点还须是同一只股票（行移位跨股票边界检查）
code_arr = df.code.to_numpy()
same = np.zeros(len(df), bool); same[:len(df)-20] = code_arr[20:] == code_arr[:-20]
df['fwd_ok'] = ok_end & same & np.isfinite(df.fwd20)

df['univ'] = df.data_valid & (df.is_st == 0) & df.fwd_ok & (df.date >= '2010-01-01')
u = df.loc[df.univ, ['date', 'fwd20']].copy()
gsum = u.groupby('date').fwd20.transform('sum')
gcnt = u.groupby('date').fwd20.transform('count')
u['ex20'] = u.fwd20 - (gsum - u.fwd20) / (gcnt - 1)
u['pct20'] = (u.groupby('date').fwd20.rank(method='average') - 0.5) / gcnt
df['ex20'] = np.nan; df['pct20'] = np.nan
df.loc[u.index, 'ex20'] = u.ex20
df.loc[u.index, 'pct20'] = u.pct20

ev = df.loc[df.univ & df.gc].copy()
ev['zero'] = np.where((ev.dif > 0) & (ev.dea > 0), '零轴双上',
              np.where((ev.dif < 0) & (ev.dea < 0), '零轴双下', '混合(DIF上DEA下)'))
ev['warm120'] = (ev.bars >= 120) & (ev.trade_status == 1)
ev['year'] = ev.date.str[:4]

def summarize(e, label):
    daily = e.groupby('date').ex20.mean()
    t_daily = daily.mean() / (daily.std() / np.sqrt(len(daily)))
    return {
        '口径': label, '事件数': len(e),
        '胜率_fwd20>0': (e.fwd20 > 0).mean(),
        '均值_fwd20': e.fwd20.mean(), '中位_fwd20': e.fwd20.median(),
        '平均分位20': e.pct20.mean(), '分位>50%比例': (e.pct20 > 0.5).mean(),
        '平均超额20': e.ex20.mean(), '中位超额20': e.ex20.median(),
        '日均超额20': daily.mean(), '日度t_未NW': t_daily,
        '信号日数': e.date.nunique(),
    }

rows = [summarize(ev, '主板全事件(任务口径)'),
        summarize(ev[ev.warm120], '主板warm120+可交易(近似旧口径)')]
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'f06_个股_汇总.csv'), index=False)

zrows = []
for label, e0 in [('任务口径', ev), ('warm120', ev[ev.warm120])]:
    for z, e in e0.groupby('zero'):
        r = summarize(e, label); r['零轴状态'] = z; zrows.append(r)
pd.DataFrame(zrows).to_csv(os.path.join(OUT, 'f06_个股_零轴.csv'), index=False)

yrows = []
for y, e in ev.groupby('year'):
    r = summarize(e, '任务口径'); r['年份'] = y; yrows.append(r)
pd.DataFrame(yrows).to_csv(os.path.join(OUT, 'f06_个股_年度.csv'), index=False)

ev[['date', 'code', 'fwd20', 'ex20', 'pct20', 'dif', 'dea', 'zero',
    'kdj_up', 'kgc', 'k', 'warm120']].to_parquet(os.path.join(OUT, '事件_主板金叉.parquet'), index=False)

# ---- KDJ 四态（股票日状态层，warm120 口径，非ST 主板） ----
st = df.loc[df.univ & (df.bars >= 120) & (df.trade_status == 1)].copy()
st['state'] = np.where(st.macd_up, 'MACD多', 'MACD空') + '+' + np.where(st.kdj_up, 'KDJ多', 'KDJ空')
st['year'] = st.date.str[:4]
srows = []
for s, e in st.groupby('state'):
    daily = e.groupby('date').ex20.mean()
    yearly = e.groupby('year').ex20.mean()
    srows.append({'状态': s, '股票日': len(e), '交易日': e.date.nunique(),
                  '日均超额20': daily.mean(), 't_未NW': daily.mean()/(daily.std()/np.sqrt(len(daily))),
                  '中位超额20': e.ex20.median(),
                  '年数': len(yearly), '正年数': int((yearly > 0).sum())})
pd.DataFrame(srows).to_csv(os.path.join(OUT, 'f06_KDJ_四态.csv'), index=False)

# 同日双金叉（事件层）
dbl = ev[ev.warm120 & ev.kgc]
nondbl = ev[ev.warm120 & ~ev.kgc]
drows = []
for label, e in [('同日双金叉', dbl), ('非同日KDJ金叉的MACD金叉', nondbl)]:
    daily_g = e.groupby('date').ex20.mean()
    yearly = (daily_g - 0.0015).groupby(e.groupby('date').ex20.mean().index.str[:4]).mean() if len(e) else pd.Series(dtype=float)
    dnet = daily_g - 0.0015
    drows.append({'类别': label, '信号数': len(e),
                  '毛超额_信号加权': e.ex20.mean(), '净超额_信号加权': e.ex20.mean() - 0.0015,
                  '日均毛超额': daily_g.mean(), '日均净超额': dnet.mean(),
                  '日度t_未NW': dnet.mean()/(dnet.std()/np.sqrt(len(dnet))),
                  '年数': len(yearly), '正年数_净': int((yearly > 0).sum())})
pd.DataFrame(drows).to_csv(os.path.join(OUT, 'f06_KDJ_双金叉.csv'), index=False)

print(pd.DataFrame(rows).to_string())
print(pd.DataFrame(zrows).to_string())
print(pd.DataFrame(srows).to_string())
print(pd.DataFrame(drows).to_string())
print('DONE')