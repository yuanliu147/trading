# -*- coding: utf-8 -*-
"""2016-10-13→2026-10-23 窗口的进攻版回测公共库。

与 2026-10-24_结构交易_主板40系统/06_代码/mb_lib.py 的差异（且仅有的差异）：
  1) FULL/TRAIN/TEST 窗口参数化（默认 2017-03-01→2026-10-23；2017-03 前为指标预热期）；
  2) 年度收益从 2017 年起算；
  3) 支持把全 A 宽度闸门替换为纯主板宽度（对照用）。
其余执行语义与 mb_lib.run_one 完全一致；一致性由
04_回测_进攻版_2016_2026.py 的"同窗口复现段"保障（须逐位复现 27.685%/-24.594%/222）。
"""
import os
import sys
import numpy as np
import pandas as pd

BASE = '/Users/admin/WorkSpace/trading'
ENGINE_DIR = os.path.join(BASE, '2026-10-24_结构交易_主板40系统/06_代码')
sys.path.insert(0, ENGINE_DIR)
from mb_engine import Engine  # noqa: E402  执行语义不变的引擎

PANEL_V2 = os.path.join(BASE, '2026-10-24_进攻版_2016_2026_回测与优化',
                        '01_数据/全A面板_20161013_20261023.pkl')

# 信号起点：面板首日 2016-10-13 + 80 个交易日（MA60 与 MA60.shift(20) 全部有效）
# 再留 3 周缓冲，确保趋势年龄/行业宽度/涨停计数等 20 日窗指标稳定。
SIG_START = '2017-03-01'
FULL = (SIG_START, '2026-10-23')
TRAIN = ('2017-03-01', '2021-12-31')
TEST = ('2022-01-01', '2026-10-23')


def board_of(c):
    c = str(c)
    if c.startswith(('600', '601', '603', '605')):
        return '沪主板'
    if c.startswith(('000', '001', '002', '003')):
        return '深主板'
    if c.startswith(('300', '301')):
        return '创业板'
    if c.startswith(('688', '689')):
        return '科创板'
    if c.startswith(('8', '4', '920')):
        return '北交所'
    return '其他'


def main_mask(codes):
    return np.array([board_of(c) in ('沪主板', '深主板') for c in codes], dtype=bool)[None, :]


def fresh_engine(panel_path=PANEL_V2, gate='alla'):
    e = Engine(panel_path=panel_path)
    for name, key in [('RSI', 'rsi14'), ('RET20', 'ret20'), ('VOL20', 'vol20'),
                      ('dist_ma20', 'dist_ma20'), ('dist_ma60', 'dist_ma60'),
                      ('TURN', 'turn'), ('VR', 'vr5_20')]:
        if not hasattr(e, name):
            setattr(e, name, np.asarray(e.P[key]))
    C = e.C.astype(float); H = e.H.astype(float); Lo = e.L.astype(float)
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
    # 纯主板宽度（对照闸门）：主板股票中收盘>MA20 的占比
    if gate == 'main':
        mb_codes = main_mask(e.codes)[0]
        above = (e.C > e.MA20) & np.isfinite(e.C)
        sub = above[:, mb_codes]
        with np.errstate(all='ignore'):
            mb_main = np.where(sub.sum(axis=1) > 0,
                               sub.mean(axis=1), np.nan)
        e.MB = mb_main.astype('float32')
    e._orig = {'base': e.base_nogate.copy(), 'MB': e.MB.copy()}
    return e


def build_base(e, cfg):
    g = cfg.get
    dates = e.dates
    date_ok = (dates >= pd.Timestamp(g('date_start', SIG_START)))[:, None]
    mask = main_mask(e.codes)
    base = (e.U & date_ok & mask
            & (e.AMT >= g('amt_min', 3e7)) & (e.AMT <= g('amt_max', 70e6))
            & (e.C >= g('price_min', 2.0)) & (e.C <= g('price_max', 15.0))
            & (e.AGE >= g('age_min', 5)) & (e.AGE <= g('age_max', 30))
            & (e.ATR >= g('atr_min', 0.0)) & (e.ATR <= g('atr_max', 0.044))
            & (e.LU <= g('lu_max', 0))
            & (e.SBR >= g('sbr_min', 0.60)) & (e.SR20 > g('sr20_min', 0.05))
            & (e.DD60 >= g('dd60_min', -0.08)))
    if g('rsi_min') is not None:
        base &= e.RSI >= g('rsi_min')
    if g('rsi_max') is not None:
        base &= e.RSI <= g('rsi_max')
    if g('ret20_min') is not None:
        base &= e.RET20 >= g('ret20_min')
    if g('ret20_max') is not None:
        base &= e.RET20 <= g('ret20_max')
    if g('mcap_min') is not None:
        base &= e.MCAP >= g('mcap_min')
    if g('mcap_max') is not None:
        base &= e.MCAP <= g('mcap_max')
    if g('vol20_min') is not None:
        base &= e.VOL20 >= g('vol20_min')
    if g('vol20_max') is not None:
        base &= e.VOL20 <= g('vol20_max')
    if g('dist20_min') is not None:
        base &= e.dist_ma20 >= g('dist20_min')
    if g('dist20_max') is not None:
        base &= e.dist_ma20 <= g('dist20_max')
    if g('dist60_min') is not None:
        base &= e.dist_ma60 >= g('dist60_min')
    if g('dist60_max') is not None:
        base &= e.dist_ma60 <= g('dist60_max')
    if g('turn_min') is not None:
        base &= e.TURN >= g('turn_min')
    if g('turn_max') is not None:
        base &= e.TURN <= g('turn_max')
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


def annual_returns(E, years=range(2017, 2027)):
    out = {}
    prev = E.equity.iloc[0]
    for y in years:
        x = E.equity.loc[f'{y}-01-01':f'{y}-12-31']
        if len(x) == 0:
            continue
        out[f'{y}年'] = x.iloc[-1] / prev - 1
        prev = x.iloc[-1]
    return out


def sub_metrics(E, T, start, end):
    x = E.loc[start:end]
    t = T[(T.entry_date >= pd.Timestamp(start)) & (T.entry_date <= pd.Timestamp(end))]
    if len(x) < 2:
        return dict(cagr=np.nan, maxdd=np.nan, maxdd_low=np.nan, trades=len(t),
                    win=np.nan, pf=np.nan)
    cagr = (x.equity.iloc[-1] / x.equity.iloc[0]) ** (252 / len(x)) - 1
    dd = (x.equity / x.equity.cummax() - 1).min()
    ddl = (x.equity_low / x.equity.cummax() - 1).min()
    win = (t.net_ret > 0).mean() if len(t) else np.nan
    pf = t.pnl[t.pnl > 0].sum() / max(1e-9, -t.pnl[t.pnl < 0].sum()) if len(t) else np.nan
    return dict(cagr=cagr, maxdd=dd, maxdd_low=ddl, trades=len(t), win=win, pf=pf)


def add_net(T):
    T = T.copy()
    if len(T) == 0:
        T = pd.DataFrame(columns=['code', 'entry_date', 'exit_date', 'entry', 'exit', 'ret',
                                  'reason', 'hold', 'shares', 'pnl', 'weight'])
        T['entry_date'] = pd.to_datetime(T['entry_date'])
        T['exit_date'] = pd.to_datetime(T['exit_date'])
        T['net_ret'] = np.nan
        T['year'] = np.nan
        T['board'] = np.nan
        return T
    T['net_ret'] = T.pnl / (T.shares * T.entry)
    T['year'] = T.entry_date.dt.year
    T['board'] = T.code.map(board_of)
    return T


def run_one(e, cfg, window=None, run_kw=None, return_daily=False):
    window = window or FULL
    base = build_base(e, cfg)
    e.base_nogate = base
    e.MB = e._orig['MB'].copy()
    kw = dict(slots=2, sizing='fixed', maxpos=.45, heat_cap=.10, gate_thr=.50,
              gate_mode='mb', rank='amt', hold=20, cost_rate=.0015, vol_target=.20,
              stop_pct=.20, skip_entry_lu=True)
    if run_kw:
        kw.update(run_kw)
    valid = set(kw) | {'risk', 'min_cost', 'rank_seed', 'start_cash', 'risk_basis',
                       'vol_floor', 'size_breadth', 'dd_half', 'dd_stop', 'dd_factor',
                       'profit_lock', 'lock_trigger', 'lock_gain', 'trail_trigger',
                       'trail_pct', 'partial_trigger', 'partial_frac', 'exit_ma20',
                       'exit_min_hold', 'sector_cap', 'force_close', 'loss_pause_n',
                       'loss_pause_days', 'dd_peak_window', 'eq_vol_target',
                       'eq_vol_floor', 'regime_exit', 'hold_jitter', 'hold_seed',
                       'extend_if_gain', 'extend_days', 'extend_max'}
    for k in list(valid):
        if k in cfg:
            kw[k] = cfg[k]
    if kw.get('gate_thr') is None:
        kw['gate_thr'] = -1
    E, T, m = e.run(window[0], window[1], ret_daily=return_daily, **kw)
    T = add_net(T)
    row = {'cfg': cfg.get('name', '')}
    for label, (a, b) in [('全期', window), ('训练', TRAIN), ('检验', TEST)]:
        if pd.Timestamp(a) >= pd.Timestamp(window[0]) and pd.Timestamp(b) <= pd.Timestamp(window[1]):
            sm = sub_metrics(E, T, a, b)
            for k, v in sm.items():
                row[f'{label}_{k}'] = v
    row.update(annual_returns(E))
    row['全期_最大单笔亏损'] = T.net_ret.min() if len(T) else np.nan
    row['全期_前5笔占比'] = (T.nlargest(5, 'pnl').pnl.sum() / T.pnl.sum()
                          if len(T) and abs(T.pnl.sum()) > 1e-9 else np.nan)
    row['全期_平均仓位'] = E.exposure.mean()
    row['全期_期末权益'] = E.equity.iloc[-1]
    yrs = [y for y in range(2017, 2027) if row.get(f'{y}年') is not None]
    row['负年份数'] = sum(1 for y in yrs if row[f'{y}年'] < 0)
    row['最差年'] = min([row[f'{y}年'] for y in yrs], default=np.nan)
    return E, T, row


ATTACK = dict(
    name='进攻版', st_mode='x', age_min=10, age_max=30, vol20_min=0.20, dd60_min=-0.15,
    hold=10, partial_trigger=0.25, partial_frac=0.5, profit_lock=True,
    lock_trigger=0.10, lock_gain=0.0, trail_trigger=0.25, trail_pct=0.10,
    rank='amt', slots=2, maxpos=0.50, vol_target=None, heat_cap=1.0,
    cost_rate=0.0025,
)
CORE = dict(
    ATTACK, name='核心版', hold=20, maxpos=0.45, vol_target=0.20, heat_cap=0.10,
)
