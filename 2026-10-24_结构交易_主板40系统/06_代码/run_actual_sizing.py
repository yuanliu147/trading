# -*- coding: utf-8 -*-
"""完整“实际价成交”版：实际价选股 + 实际价买整手 + 实际金额结算。

信号/止损仍用后复权（总收益），避免除权假跳空。
运行：
    cd "2026-10-24_结构交易_主板40系统/06_代码"
    python3 run_actual_sizing.py
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
from engine_actual import ActualEngine
from run_actual_price import build_base_raw, CORE, ATTACK, V40, leverage, yret


def load_raw(dates, codes):
    con = duckdb.connect(DB, read_only=True)
    q = """
    SELECT thscode, date, open, close FROM v_daily
    WHERE date BETWEEN DATE '2016-10-13' AND DATE '2026-10-13'
      AND regexp_matches(thscode, '^(600|601|603|605|000|001|002|003)')
    """
    df = con.execute(q).fetchdf()
    con.close()
    df['date'] = pd.to_datetime(df['date'])
    out = {}
    for col in ['open', 'close']:
        x = df.pivot(index='date', columns='thscode', values=col)
        out[col] = x.reindex(index=pd.DatetimeIndex(dates), columns=list(codes)).values.astype(float)
    return out['open'], out['close']


def fresh_actual():
    e = ActualEngine()
    for name, key in [('RSI', 'rsi14'), ('RET20', 'ret20'), ('VOL20', 'vol20'),
                      ('dist_ma20', 'dist_ma20'), ('dist_ma60', 'dist_ma60'),
                      ('TURN', 'turn'), ('VR', 'vr5_20')]:
        if not hasattr(e, name):
            setattr(e, name, np.asarray(e.P[key]))
    C = e.C.astype(float)
    H = e.H.astype(float)
    Lo = e.L.astype(float)
    r = np.full_like(C, np.nan)
    r[1:] = C[1:] / C[:-1] - 1
    lu5 = (r > 0.045) & (r < 0.055) & ((C == H) | (H == Lo))
    ld5 = (r < -0.045) & (r > -0.055) & ((C == Lo) | (H == Lo))
    c5 = (pd.DataFrame(lu5).rolling(60, min_periods=1).sum().values
          + pd.DataFrame(ld5).rolling(60, min_periods=1).sum().values)
    maxabs60 = pd.DataFrame(np.abs(r)).rolling(60, min_periods=20).max().values
    e.C5 = c5
    e.MAXABS60 = maxabs60
    e.STX = (c5 >= 2) | ((c5 >= 1) & (maxabs60 < 0.060))
    e.STY = e.STX | ((maxabs60 < 0.055) & (C < 8))
    e._orig = {'base': e.base_nogate.copy(), 'MB': e.MB.copy()}
    return e


def run(e, cfg, C_RAW, O_RAW):
    e.set_raw_prices(O_RAW, O_RAW, O_RAW, C_RAW)  # 只用 RO；其余占位
    e.base_nogate = build_base_raw(e, cfg, C_RAW)
    e.MB = e._orig['MB'].copy()
    return e.run(L.FULL[0], L.FULL[1], slots=cfg['slots'], sizing='fixed',
                 maxpos=cfg['maxpos'], heat_cap=cfg['heat_cap'], gate_thr=0.50,
                 gate_mode='mb', rank=cfg['rank'], hold=cfg['hold'],
                 cost_rate=cfg['cost_rate'], vol_target=cfg['vol_target'],
                 stop_pct=0.20, skip_entry_lu=True,
                 partial_trigger=cfg['partial_trigger'], partial_frac=cfg['partial_frac'],
                 profit_lock=cfg['profit_lock'], lock_trigger=cfg['lock_trigger'],
                 lock_gain=cfg['lock_gain'], trail_trigger=cfg['trail_trigger'],
                 trail_pct=cfg['trail_pct'], hold_jitter=cfg.get('hold_jitter', 0),
                 hold_seed=cfg.get('hold_seed', 1), ret_daily=True)


def main():
    P = pickle.load(open(PANEL, 'rb'))
    dates = pd.DatetimeIndex(P['meta']['dates'])
    codes = P['meta']['codes']
    O_RAW, C_RAW = load_raw(dates, codes)
    print('raw open/close', O_RAW.shape, np.isfinite(O_RAW).mean(), np.isfinite(C_RAW).mean())

    rows = []
    for label, cfg, lev in [('核心版', CORE, 1.0), ('进攻版', ATTACK, 1.0), ('四成版', V40, 1.5)]:
        e = fresh_actual()
        E, T, m = run(e, cfg, C_RAW, O_RAW)
        T = L.add_net(T)
        T['net_ret'] = T.pnl / (T.shares * T.entry_actual)
        if lev > 1:
            D = leverage(E, lev, 0.065)
            cagr = (D.equity.iloc[-1] / 20000) ** (252 / len(D)) - 1
            dd = (D.equity / D.equity.cummax() - 1).min()
            low = D.collateral.min()
            final = D.equity.iloc[-1]
            yy = yret(D)
        else:
            cagr, dd, low, final = m['cagr'], m['maxdd'], np.nan, E.equity.iloc[-1]
            yy = yret(E)
        row = dict(版本=label, 口径='实际价选股+实际价整手', 年化=cagr, 最大回撤=dd,
                   交易=m['trades'], 胜率=m['win'], 盈亏比=m['pf'], 期末权益=final,
                   最低标准担保=low)
        for y in [2018, 2020, 2023, 2026]:
            row[f'{y}年'] = yy.get(y, np.nan)
        rows.append(row)
        suffix = {'核心版': '核心_实际价成交', '进攻版': '进攻_实际价成交', '四成版': '四成_实际价成交'}[label]
        E.index.name = 'date'
        E.to_csv(os.path.join(OUT, suffix + '_权益.csv'), encoding='utf-8-sig',
                 float_format='%.4f', date_format='%Y-%m-%d')
        T.to_csv(os.path.join(OUT, suffix + '_逐笔.csv'), index=False, encoding='utf-8-sig',
                 float_format='%.6f', date_format='%Y-%m-%d')
        if lev > 1:
            D.to_csv(os.path.join(OUT, suffix + '_1.5倍_权益.csv'), encoding='utf-8-sig',
                     float_format='%.4f', date_format='%Y-%m-%d')
        print(label, 'done', round(cagr, 4), round(dd, 4), m['trades'])

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, '实际价成交_对照.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f')
    print(R.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
