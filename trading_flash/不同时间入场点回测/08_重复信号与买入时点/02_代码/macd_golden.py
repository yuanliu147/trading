# -*- coding: utf-8 -*-
"""信号日 MACD 金叉/金叉状态 的后续上涨概率。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
ROOT = HERE.parents[2]
MOTHER = ROOT.parent / '2026-10-24_主板恐慌反转_年化30'
RES = OUT / '03_结果'
RES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / '02_代码'))
import timing_lib as tl  # noqa: E402

PANEL = MOTHER / '01_数据' / 'research_panel.parquet'
NTH, HOLD, TP, COST = 3, 15, 0.20, 0.00175

D = tl.load(str(PANEL))
tl.add_signal(D)
b = D['base']
dates = pd.DatetimeIndex(D['dates'])
si = {x: i for i, x in enumerate(D['syms'])}

cols = ['thscode', 'date', 'macd_dif_pct', 'macd_dea_pct', 'macd_hist_pct',
        'macd_golden_cross', 'macd_dead_cross', 'hfq_close']
df = pd.read_parquet(PANEL, columns=cols)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['thscode', 'date']).reset_index(drop=True)
for n in [5, 15]:
    df[f'fwd{n}'] = df.groupby('thscode')['hfq_close'].shift(-n) / df['hfq_close'] - 1
df['bull'] = (df['macd_dif_pct'] > df['macd_dea_pct']).astype(int)
df['gc'] = df['macd_golden_cross'].fillna(0).astype(int)
df['gc_cum'] = df.groupby('thscode')['gc'].cumsum()
df['days_since_gc'] = df.groupby(['thscode', 'gc_cum']).cumcount()
df.loc[df['gc_cum'] == 0, 'days_since_gc'] = np.nan
df['recent5'] = df['days_since_gc'] <= 5

ev = b[b['ev_custom']].copy()
ev['di'] = ev['date'].map({d: i for i, d in enumerate(dates)})
ev['rk'] = ev.groupby('date')['drawdown250'].rank(ascending=True, method='first')
ev = ev.merge(df, on=['thscode', 'date'], how='left', suffixes=('', '_m'))

# 前二完整
det = pd.read_csv(ROOT / '06_不同信号日起点' / '03_结果' / '逐日首笔_逐只.csv',
                  parse_dates=['signal_date', 'entry_date', 'exit_date'])
det = det[det['complete']].copy()
T = det.merge(df, left_on=['code', 'signal_date'], right_on=['thscode', 'date'],
              how='left', suffixes=('', '_m'))
T['y'] = T['signal_date'].dt.year

# 全部信号完整
A = ev.copy()
rets = []
for r in A.itertuples():
    st = tl.standalone_trade(D, si[r.thscode], int(r.di), NTH, 'open', HOLD, TP, COST)
    ok = (st.get('status') == 'ok' and st.get('exit_reason') in ('target', 'time')
          and int(st.get('entry_i', 10**9)) + HOLD - 1 <= len(dates) - 1)
    rets.append(st.get('ret', np.nan) if ok else np.nan)
A['ret'] = rets
A = A.dropna(subset=['ret']).copy()
A['y'] = A['date'].dt.year


def summarize(z, mask, label, sample):
    g = z[mask].dropna(subset=['ret'])
    if not len(g):
        return None
    ex = g[g.y < 2026]
    y26 = g[g.y == 2026]
    return dict(
        样本=sample, 口径=label, 笔数=len(g),
        策略平均=g.ret.mean(), 策略胜率=(g.ret > 0).mean(), 策略中位=g.ret.median(),
        五日均值=g.fwd5.mean(), 五日上涨率=(g.fwd5 > 0).mean(),
        十五日均值=g.fwd15.mean(), 十五日上涨率=(g.fwd15 > 0).mean(),
        剔除2026胜率=(ex.ret > 0).mean() if len(ex) else np.nan,
        y2026胜率=(y26.ret > 0).mean() if len(y26) else np.nan,
    )


rows = []
for sample, z in [('前二', T), ('全部信号', A)]:
    rows.append(summarize(z, z.gc == 1, '金叉日', sample))
    rows.append(summarize(z, z.gc == 0, '非金叉日', sample))
    rows.append(summarize(z, z.bull == 1, '金叉状态(DIF>DEA)', sample))
    rows.append(summarize(z, z.bull == 0, '非金叉状态', sample))
    rows.append(summarize(z, z.recent5 == True, '近5日内金叉', sample))
    rows.append(summarize(z, z.recent5 == False, '非近5日金叉', sample))
    rows.append(summarize(z, pd.Series(True, index=z.index), '全部', sample))
out = pd.DataFrame([r for r in rows if r is not None])
out.to_csv(RES / '信号日金叉.csv', index=False, encoding='utf-8-sig')
print(out.round(4).to_string(index=False))

cur = df[df.date.isin([pd.Timestamp('2026-10-22'), pd.Timestamp('2026-10-23')])]
cur = cur[cur.thscode.isin(['603778.SH', '605376.SH', '002969.SZ', '603256.SH',
                            '002796.SZ', '605117.SH'])]
print('\n当前 MACD')
print(cur[['thscode', 'date', 'macd_dif_pct', 'macd_dea_pct', 'macd_hist_pct',
           'macd_golden_cross', 'bull', 'days_since_gc']].round(5).to_string(index=False))
