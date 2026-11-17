# -*- coding: utf-8 -*-
"""用“不复权实际价”重做 2–15 元价格过滤，比较与后复权价过滤的差异。

只改价格过滤，不动信号、排序、仓位、成本：
- 原来：收盘（后复权）在 2–15 元
- 现在：收盘（不复权实际价）在 2–15 元
其余条件完全相同。用于回答“能不能用除权价”。

运行：
    cd "2026-10-24_结构交易_主板40系统/06_代码"
    python3 run_actual_price.py
"""
import os
import sys
import pickle
import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, '05_证据')
DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'
PANEL = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl'
sys.path.insert(0, HERE)
import mb_lib as L


def load_raw_close(dates, codes):
    con = duckdb.connect(DB, read_only=True)
    q = """
    SELECT thscode, date, close FROM v_daily
    WHERE date BETWEEN DATE '2016-10-13' AND DATE '2026-10-13'
      AND regexp_matches(thscode, '^(600|601|603|605|000|001|002|003)')
    """
    df = con.execute(q).fetchdf()
    con.close()
    df['date'] = pd.to_datetime(df['date'])
    raw = df.pivot(index='date', columns='thscode', values='close')
    raw = raw.reindex(index=pd.DatetimeIndex(dates), columns=list(codes))
    return raw.values.astype(float)


def build_base_raw(e, cfg, C_RAW):
    """mb_lib.build_base 的副本，唯一改动：价格过滤用不复权实际价。"""
    g = cfg.get
    dates = e.dates
    date_ok = (dates >= pd.Timestamp(g('date_start', '2017-10-13')))[:, None]
    mask = L.main_mask(e.codes)
    base = (e.U & date_ok & mask
            & (e.AMT >= g('amt_min', 3e7)) & (e.AMT <= g('amt_max', 70e6))
            & (C_RAW >= g('price_min', 2.0)) & (C_RAW <= g('price_max', 15.0))
            & (e.AGE >= g('age_min', 5)) & (e.AGE <= g('age_max', 30))
            & (e.ATR >= g('atr_min', 0.0)) & (e.ATR <= g('atr_max', 0.044))
            & (e.LU <= g('lu_max', 0))
            & (e.SBR >= g('sbr_min', 0.60)) & (e.SR20 > g('sr20_min', 0.05))
            & (e.DD60 >= g('dd60_min', -0.08)))
    if g('vol20_min') is not None:
        base &= e.VOL20 >= g('vol20_min')
    if g('vol20_max') is not None:
        base &= e.VOL20 <= g('vol20_max')
    st_mode = g('st_mode', 'none')
    if st_mode == 'one':
        base &= e.C5 < 1
    elif st_mode == 'two':
        base &= e.C5 < 2
    elif st_mode == 'x':
        base &= ~e.STX
    elif st_mode == 'y':
        base &= ~e.STY
    elif st_mode not in (None, 'none'):
        raise ValueError(st_mode)
    return base


CORE = dict(st_mode='x', age_min=10, age_max=30, vol20_min=0.20, dd60_min=-0.15,
            hold=20, partial_trigger=0.25, partial_frac=0.5, profit_lock=True,
            lock_trigger=0.10, lock_gain=0.0, trail_trigger=0.25, trail_pct=0.10,
            rank='amt', slots=2, maxpos=0.45, vol_target=0.20, heat_cap=0.10,
            cost_rate=0.0025)
ATTACK = dict(CORE, hold=10, maxpos=0.50, vol_target=None, heat_cap=1.0)
V40 = dict(ATTACK, cost_rate=0.0015)


def run(e, cfg, C_RAW):
    e.base_nogate = build_base_raw(e, cfg, C_RAW)
    e.MB = e._orig['MB'].copy()
    E, T, m = e.run(L.FULL[0], L.FULL[1], slots=cfg['slots'], sizing='fixed',
                    maxpos=cfg['maxpos'], heat_cap=cfg['heat_cap'], gate_thr=0.50,
                    gate_mode='mb', rank=cfg['rank'], hold=cfg['hold'],
                    cost_rate=cfg['cost_rate'], vol_target=cfg['vol_target'],
                    stop_pct=0.20, skip_entry_lu=True,
                    partial_trigger=cfg['partial_trigger'], partial_frac=cfg['partial_frac'],
                    profit_lock=cfg['profit_lock'], lock_trigger=cfg['lock_trigger'],
                    lock_gain=cfg['lock_gain'], trail_trigger=cfg['trail_trigger'],
                    trail_pct=cfg['trail_pct'], ret_daily=True)
    return E, L.add_net(T), m


def leverage(E, Lv=1.5, rate=0.065, start=20000.0):
    """杠杆叠加。

    标准维持担保比例 = 总资产 / 负债（不是净资产 / 负债）。
    `collateral` 存标准口径；`collateral_net` 存旧口径（净资产/负债），仅供对照。

    重要假设：本函数把“无杠杆权益收益”直接乘到总资产上，等价于
    “单票上限 = 总资产的 50%，随杠杆同比例放大”。
    1.5 倍时单票约 75% 净资产，1.9 倍时约 95% 净资产。
    若坚持“单票 50% 净资产”，借来的钱只会变成闲置现金，
    扣除 6.5% 融资利息后，1.9 倍只有约 20%，不如不加杠杆。
    """
    r0 = E.equity.pct_change().fillna(0).values
    dates = E.index
    eq = start
    debt = 0.0
    assets = start
    rows = []
    for i in range(len(r0)):
        if i > 0:
            assets *= (1 + r0[i])
            debt *= (1 + rate / 252)
            eq = assets - debt
        if i == 0 or dates[i].quarter != dates[i - 1].quarter:
            debt = (Lv - 1) * eq
            assets = Lv * eq
        rows.append(dict(date=dates[i], equity=eq, debt=debt, assets=assets,
                         collateral=(assets / debt if debt > 1e-9 else np.nan),
                         collateral_net=(eq / debt if debt > 1e-9 else np.nan)))
    D = pd.DataFrame(rows).set_index('date')
    D['ret'] = D.equity.pct_change().fillna(0)
    return D


def yret(E, start=20000.0):
    prev = start
    out = {}
    for y in range(2018, 2027):
        x = E.loc[str(y)]
        if len(x):
            out[y] = x.equity.iloc[-1] / prev - 1
            prev = x.equity.iloc[-1]
    return out


def main():
    P = pickle.load(open(PANEL, 'rb'))
    dates = pd.DatetimeIndex(P['meta']['dates'])
    codes = P['meta']['codes']
    C_RAW = load_raw_close(dates, codes)
    print('actual close matrix', C_RAW.shape, 'finite', np.isfinite(C_RAW).mean())

    rows = []
    for label, cfg, lev in [('核心版', CORE, 1.0), ('进攻版', ATTACK, 1.0), ('四成版', V40, 1.5)]:
        e = L.fresh_engine()
        E, T, m = run(e, cfg, C_RAW)
        if lev > 1:
            D = leverage(E, lev, 0.065)
            cagr = (D.equity.iloc[-1] / 20000) ** (252 / len(D)) - 1
            dd = (D.equity / D.equity.cummax() - 1).min()
            low = D.collateral.min()
            yy = yret(D)
        else:
            cagr, dd, low = m['cagr'], m['maxdd'], np.nan
            yy = yret(E)
        row = dict(版本=label, 价格口径='不复权实际价', 年化=cagr, 最大回撤=dd,
                   交易=m['trades'], 胜率=m['win'], 盈亏比=m['pf'],
                   期末权益=(D.equity.iloc[-1] if lev > 1 else E.equity.iloc[-1]),
                   最低标准担保=low)
        for y in [2018, 2020, 2023, 2026]:
            row[f'{y}年'] = yy.get(y, np.nan)
        rows.append(row)
        suffix = {'核心版': '核心_实际价', '进攻版': '进攻_实际价', '四成版': '四成_实际价'}[label]
        E.index.name = 'date'
        E.to_csv(os.path.join(OUT, suffix + '_权益.csv'), encoding='utf-8-sig',
                 float_format='%.4f', date_format='%Y-%m-%d')
        T.to_csv(os.path.join(OUT, suffix + '_逐笔.csv'), index=False, encoding='utf-8-sig',
                 float_format='%.6f', date_format='%Y-%m-%d')
        if lev > 1:
            D.to_csv(os.path.join(OUT, suffix + '_1.5倍_权益.csv'), encoding='utf-8-sig',
                     float_format='%.4f', date_format='%Y-%m-%d')

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, '实际价过滤_对照.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f')
    print(R.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
