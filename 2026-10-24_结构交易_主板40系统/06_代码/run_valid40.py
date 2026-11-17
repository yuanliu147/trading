# -*- coding: utf-8 -*-
"""生成 40% 版：成交额升序、10 日、2×50%、1.5 倍融资、单边 0.15% 成本。

运行：
    cd "2026-10-24_结构交易_主板40系统/06_代码"
    python3 run_valid40.py
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
            hold=10, partial_trigger=0.25, partial_frac=0.5, profit_lock=True,
            lock_trigger=0.10, lock_gain=0.0, trail_trigger=0.25, trail_pct=0.10,
            rank='amt', slots=2, maxpos=0.50, vol_target=None, heat_cap=1.0,
            cost_rate=0.0015)


def run(st='x', cost=0.0015):
    cfg = dict(BASE); cfg.update(st_mode=st, cost_rate=cost)
    e = L.fresh_engine(); e.base_nogate = L.build_base(e, cfg)
    E, T, m = e.run(FULL[0], FULL[1], slots=2, sizing='fixed', maxpos=0.50,
                    heat_cap=1.0, gate_thr=0.50, gate_mode='mb', rank='amt', hold=10,
                    cost_rate=cost, vol_target=None, stop_pct=0.20, skip_entry_lu=True,
                    partial_trigger=0.25, partial_frac=0.5, profit_lock=True,
                    lock_trigger=0.10, lock_gain=0.0, trail_trigger=0.25, trail_pct=0.10,
                    ret_daily=True)
    T = L.add_net(T)
    return E, T, m


def leverage(E, Lv=1.5, rate=0.065, start=20000.0):
    r0 = E.equity.pct_change().fillna(0).values
    n = len(r0)
    dates = E.index
    eq = start; debt = 0.0; assets = start
    rows = []
    for i in range(n):
        if i > 0:
            assets *= (1 + r0[i]); debt *= (1 + rate / 252); eq = assets - debt
        if i == 0 or dates[i].quarter != dates[i - 1].quarter:
            debt = (Lv - 1) * eq; assets = Lv * eq
        rows.append(dict(date=dates[i], equity=eq, debt=debt, assets=assets,
                         collateral=(eq / debt if debt > 1e-9 else np.nan)))
    D = pd.DataFrame(rows).set_index('date')
    D['ret'] = D.equity.pct_change().fillna(0)
    return D


def yearly(E, start=20000.0):
    rows = []; prev = start
    for y in range(2018, 2027):
        x = E.loc[str(y)]
        if len(x) == 0:
            continue
        rows.append(dict(年份=y, 收益=x.equity.iloc[-1] / prev - 1, 年末权益=x.equity.iloc[-1]))
        prev = x.equity.iloc[-1]
    return pd.DataFrame(rows)


def metrics(D, start=20000.0):
    cagr = (D.equity.iloc[-1] / start) ** (252 / len(D)) - 1
    dd = (D.equity / D.equity.cummax() - 1).min()
    return dict(cagr=cagr, maxdd=dd, min_collateral=D.collateral.min(), final=D.equity.iloc[-1])


if __name__ == '__main__':
    E, T, m = run('x', 0.0015)
    D = leverage(E, 1.5, 0.065)
    E.index.name = 'date'
    T.to_csv(os.path.join(OUT, '进攻40_AMT_逐笔.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f', date_format='%Y-%m-%d')
    D.to_csv(os.path.join(OUT, '进攻40_1.5倍_权益.csv'), encoding='utf-8-sig',
             float_format='%.4f', date_format='%Y-%m-%d')
    yearly(D).to_csv(os.path.join(OUT, '进攻40_1.5倍_年度.csv'), index=False,
                     encoding='utf-8-sig', float_format='%.6f')
    mm = metrics(D)
    print('40%版（1.5倍，0.15%成本）:', {k: round(v, 4) for k, v in mm.items()})
    print('无杠杆:', {k: round(v, 4) for k, v in m.items() if isinstance(v, (int, float))})

    rows = []
    for Lv in [1.2, 1.4, 1.5, 1.6, 1.7]:
        d = leverage(E, Lv, 0.065)
        x = metrics(d)
        rows.append(dict(杠杆=Lv, 年化=x['cagr'], 最大回撤=x['maxdd'],
                         最低抵押率=x['min_collateral'], 期末权益=x['final']))
    lev = pd.DataFrame(rows)
    lev.to_csv(os.path.join(OUT, '融资杠杆表_0.15成本.csv'), index=False,
               encoding='utf-8-sig', float_format='%.6f')
    print('\n杠杆:\n', lev.round(4).to_string(index=False))

    cost_rows = []
    for cr in [0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040]:
        Ex, Tx, mx = run('x', cr)
        dx = leverage(Ex, 1.5, 0.065)
        x = metrics(dx)
        cost_rows.append(dict(单边成本=cr, 无杠杆年化=mx['cagr'], 无杠杆回撤=mx['maxdd'],
                              杠杆年化=x['cagr'], 杠杆回撤=x['maxdd'],
                              最低抵押率=x['min_collateral']))
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(os.path.join(OUT, '成本敏感性_AMT.csv'), index=False,
                encoding='utf-8-sig', float_format='%.6f')
    print('\n成本:\n', cost.round(4).to_string(index=False))

    st_rows = []
    for st, label in [('x', '基准STX'), ('one', '1次5%触板'), ('two', '2次5%触板'), ('y', '更严STY')]:
        Es, Ts, ms = run(st, 0.0015)
        ds = leverage(Es, 1.5, 0.065)
        x = metrics(ds)
        st_rows.append(dict(ST口径=label, 无杠杆年化=ms['cagr'], 无杠杆回撤=ms['maxdd'],
                            杠杆年化=x['cagr'], 杠杆回撤=x['maxdd'], 交易=ms['trades']))
    st = pd.DataFrame(st_rows)
    st.to_csv(os.path.join(OUT, 'ST口径_AMT.csv'), index=False, encoding='utf-8-sig',
              float_format='%.6f')
    print('\nST:\n', st.round(4).to_string(index=False))
