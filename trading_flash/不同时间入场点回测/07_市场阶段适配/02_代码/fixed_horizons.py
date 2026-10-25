# -*- coding: utf-8 -*-
"""
每个信号日起点，固定持有 20/60/120/250 个交易日后的账户收益。

用途：2026 年开始的 cohort 还没跑满 1 年，不能拿“期末总收益/年化”和
2017-2025 的 5 年结果比。这里统一截断到相同交易日数，做同口径比较。

口径与 06_不同信号日起点 完全一致：
- 信号日收盘确认，T+3 开盘买；
- 2 仓、每仓 50% 当前权益；
- +20% 盘中止盈，15 个交易日到期收盘卖；
- 成本 0.175%/边；一字涨停/停牌不买、不递补。
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / '02_代码'))
import timing_lib as tl  # noqa: E402

PANEL = ROOT / '01_数据' / 'research_panel.parquet'
RES = OUT / '03_结果'
RES.mkdir(parents=True, exist_ok=True)

NTH, HOLD, SLOTS, TP, COST = 3, 15, 2, 0.20, 0.00175
HORIZONS = (20, 60, 120, 250)
END_ALL = pd.Timestamp('2026-10-23')

D = tl.load(str(PANEL))
tl.add_signal(D)
dates = pd.DatetimeIndex(D['dates'])
base = D['base']
signal_days = pd.DatetimeIndex(sorted(base.loc[base['ev_custom'], 'date'].unique()))

rows = []
for k, d in enumerate(signal_days):
    i0 = int(np.searchsorted(dates, np.datetime64(d)))
    i_end = min(i0 + max(HORIZONS), len(dates) - 1)
    eq, tr, _ = tl.simulate_timing(
        D, start=pd.Timestamp(d), end=pd.Timestamp(dates[i_end]),
        nth=NTH, fill='open', hold=HOLD, slots=SLOTS, tp=TP, cost=COST)
    rec = dict(signal_date=pd.Timestamp(d))
    if eq is None or eq.empty:
        rows.append(rec)
        continue
    e = eq.set_index('date')['equity']
    npos = eq.set_index('date')['npos']
    rec['first_entry'] = tr['entry_date'].min() if (tr is not None and len(tr)) else pd.NaT
    for h in HORIZONS:
        j = i0 + h
        if j > len(dates) - 1:
            continue
        dt = pd.Timestamp(dates[j])
        if dt not in e.index:
            continue
        sub = e.loc[:dt]
        rec[f'ret_{h}'] = float(e.loc[dt] - 1.0)
        rec[f'dd_{h}'] = float((sub / sub.cummax() - 1.0).min())
        rec[f'n_{h}'] = int(npos.loc[dt])
        rec[f'trades_{h}'] = int(len(tr[tr['exit_date'] <= dt])) if (tr is not None and len(tr)) else 0
    rows.append(rec)

out = pd.DataFrame(rows)
out.to_csv(RES / '起点固定持有期.csv', index=False, encoding='utf-8-sig')
print('written', len(out))
print(out.tail(10).to_string(index=False))
