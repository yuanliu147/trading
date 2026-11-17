# -*- coding: utf-8 -*-
"""市值缺失审计：复现原 rank='mcap' 的问题与修正后的数字。

运行：
    cd "2026-10-24_结构交易_主板40系统/06_代码"
    python3 audit.py
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, '05_证据')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
import mb_lib as L

FULL = L.FULL
BASE = dict(st_mode='x', age_min=10, age_max=30, vol20_min=0.20, dd60_min=-0.15,
            hold=20, partial_trigger=0.25, partial_frac=0.5, profit_lock=True,
            lock_trigger=0.10, lock_gain=0.0, trail_trigger=0.25, trail_pct=0.10,
            rank='mcap', slots=2, maxpos=0.45, vol_target=0.20, heat_cap=0.10,
            cost_rate=0.0025)


def run(rank='mcap', require_mcap=False, hold=20, maxpos=0.45, vol=0.20):
    cfg = dict(BASE); cfg.update(rank=rank, hold=hold, maxpos=maxpos, vol_target=vol,
                                 heat_cap=0.10 if vol is not None else 1.0)
    e = L.fresh_engine()
    base = L.build_base(e, cfg)
    if require_mcap:
        base = base & np.isfinite(e.MCAP)
    e.base_nogate = base
    E, T, m = e.run(FULL[0], FULL[1], slots=2, sizing='fixed', maxpos=maxpos,
                    heat_cap=cfg['heat_cap'], gate_thr=0.50, gate_mode='mb', rank=rank,
                    hold=hold, cost_rate=0.0025, vol_target=vol, stop_pct=0.20,
                    skip_entry_lu=True, partial_trigger=0.25, partial_frac=0.5,
                    profit_lock=True, lock_trigger=0.10, lock_gain=0.0,
                    trail_trigger=0.25, trail_pct=0.10, ret_daily=True)
    T = L.add_net(T)
    return e, E, T, m


def miss_stats(e, T):
    if len(T) == 0:
        return dict(n=0, miss_n=0, miss_share=np.nan, miss_pnl=np.nan,
                    total_pnl=np.nan, miss_pnl_share=np.nan)
    miss = []
    for _, x in T.iterrows():
        i = e.dates.searchsorted(x.entry_date)
        j = e.code_idx[x.code]
        miss.append(not np.isfinite(e.MCAP[i, j]))
    miss = np.array(miss)
    pnl = T.pnl.values
    return dict(n=len(T), miss_n=int(miss.sum()), miss_share=float(miss.mean()),
                miss_pnl=float(pnl[miss].sum()), total_pnl=float(pnl.sum()),
                miss_pnl_share=float(pnl[miss].sum() / pnl.sum()) if abs(pnl.sum()) > 1e-9 else np.nan)


if __name__ == '__main__':
    # 原始面板缺失率
    import pickle
    P = pickle.load(open('/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl', 'rb'))
    m = P['mcap'].values
    print('面板 mcap 缺失率:', round(float(np.mean(~np.isfinite(m))), 4))

    rows = []
    for label, rank, req, hold, mp, vol in [
        ('原mcap_不要求市值_核心', 'mcap', False, 20, 0.45, 0.20),
        ('原mcap_要求市值_核心', 'mcap', True, 20, 0.45, 0.20),
        ('原mcap_不要求市值_进攻', 'mcap', False, 10, 0.50, None),
        ('原mcap_要求市值_进攻', 'mcap', True, 10, 0.50, None),
        ('修正amt_核心', 'amt', False, 20, 0.45, 0.20),
        ('修正amt_进攻', 'amt', False, 10, 0.50, None),
    ]:
        e, E, T, mm = run(rank, req, hold, mp, vol)
        s = miss_stats(e, T)
        s.update(版本=label, 年化=mm['cagr'], 最大回撤=mm['maxdd'], 交易=mm['trades'])
        rows.append(s)
        print(label, 'cagr', round(mm['cagr'], 4), 'dd', round(mm['maxdd'], 4),
              'n', mm['trades'], 'miss', s['miss_n'], 'miss_share', round(s['miss_share'], 3))
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(OUT, '市值缺失审计.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f')
    print('\n', A.round(4).to_string(index=False))
