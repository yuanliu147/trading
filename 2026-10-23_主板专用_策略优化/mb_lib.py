# -*- coding: utf-8 -*-
"""主板专用研究的公共回测工具。

只做三件事：
1) 用面板原始字段重建主板股票池，所有过滤参数显式传入；
2) 调用原 108 稳健引擎，保证成交/费用/止损口径与既有研究一致；
3) 统一输出全期、训练期、检验期、年度、尾部指标，便于横向比较。

注意：这里是研究代码，不是实盘下单程序。
"""
import os
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

BASE = '/Users/admin/WorkSpace/trading'
ENG = os.path.join(BASE, '2026-10-22_全信号_买卖时间_优先级研究', '108_稳健引擎.py')
eng = SourceFileLoader('mb_eng', ENG).load_module()

TRAIN = ('2018-01-01', '2021-12-31')
TEST = ('2022-01-01', '2026-10-13')
FULL = ('2018-01-01', '2026-10-13')


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


def fresh_engine(panel_path=None):
    e = eng.Engine(panel_path=panel_path)
    # 2026-10-24 修正：面板自带 mcap 只覆盖约 1,280 只（换手率缓存），
    # 补抓后的全覆盖矩阵存在 sidecar；存在则替换，并对账覆盖率。
    side = os.path.join(BASE, '2026-10-23_主板专用_策略优化', '市值矩阵_全覆盖.pkl')
    if os.path.exists(side):
        M = pd.read_pickle(side)
        assert list(M.index) == list(e.dates) and list(M.columns) == list(e.codes)
        e.MCAP = M.values.astype('float64')
    for name, key in [('RSI', 'rsi14'), ('RET20', 'ret20'), ('VOL20', 'vol20'),
                      ('dist_ma20', 'dist_ma20'), ('dist_ma60', 'dist_ma60'),
                      ('TURN', 'turn'), ('VR', 'vr5_20')]:
        if not hasattr(e, name):
            setattr(e, name, np.asarray(e.P[key]))
    # 历史没有 ST 名称表，用“5% 涨跌停”行为做点时代理：
    # 主板正常股 10% 限制；ST 为 5%。过去 60 日出现 >=2 次 5% 触板，
    # 或出现 1 次且 60 日最大绝对日收益 <6%，视为 ST 样态。
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
    e._orig = {
        'base': e.base_nogate.copy(),
        'MB': e.MB.copy(),
    }
    return e


def build_base(e, cfg):
    """按 cfg 重建信号矩阵。cfg 中未出现的条件使用原 70/15 默认值。"""
    g = cfg.get
    dates = e.dates
    date_ok = (dates >= pd.Timestamp(g('date_start', '2017-10-13')))[:, None]
    mask = main_mask(e.codes)
    base = (e.U & date_ok & mask
            & (e.AMT >= g('amt_min', 3e7)) & (e.AMT <= g('amt_max', 70e6))
            & (e.C >= g('price_min', 2.0)) & (e.C <= g('price_max', 15.0))
            & (e.AGE >= g('age_min', 5)) & (e.AGE <= g('age_max', 30))
            & (e.ATR >= g('atr_min', 0.0)) & (e.ATR <= g('atr_max', 0.044))
            & (e.LU <= g('lu_max', 0))
            & (e.SBR >= g('sbr_min', 0.60)) & (e.SR20 > g('sr20_min', 0.05))
            & (e.DD60 >= g('dd60_min', -0.08)))
    # 以下特征含缺失值：只有显式给出阈值时才过滤，避免默认把缺失样本全部删掉。
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


def build_oversold(e, cfg):
    """极端超跌卫星：宽度<阈值、RSI低、ATR可控、成交额够。"""
    g = cfg.get
    thr = g('ov_mb', 0.10)
    return ((e.MB[:, None] < thr) & (e.RSI < g('ov_rsi', 30))
            & (e.ATR <= g('ov_atr', 0.06)) & (e.AMT >= g('ov_amt', 1e8))
            & (e.C >= g('ov_price_min', 2.0)) & main_mask(e.codes)
            & (e.dates >= pd.Timestamp(g('date_start', '2017-10-13')))[:, None])


def annual_returns(E, years=range(2018, 2027)):
    out = {}
    prev = E.equity.iloc[0]
    for y in years:
        x = E.equity.loc[str(y)]
        if len(x) == 0:
            out[f'{y}年'] = np.nan
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


def run_one(e, cfg, run_kw=None, oversold=False, return_daily=False):
    """cfg 同时包含股票池过滤和 run 参数；返回 E,T,metrics,cfg。"""
    base = build_base(e, cfg)
    e.base_nogate = base
    e.MB = e._orig['MB'].copy()
    if oversold:
        ov = build_oversold(e, cfg)
        e.base_nogate = base | ov
        e.MB = np.ones_like(e.MB)
    kw = dict(slots=2, sizing='fixed', maxpos=.45, heat_cap=.10, gate_thr=.50,
              gate_mode='mb', rank='amt', hold=20, cost_rate=.0015, vol_target=.20,
              stop_pct=.20, skip_entry_lu=True)
    if run_kw:
        kw.update(run_kw)
    # cfg 里所有引擎 run 参数都透传；None 也保留（例如关闭波动目标/回撤节流）。
    valid = set(kw) | {'risk', 'min_cost', 'rank_seed', 'start_cash', 'risk_basis',
                       'vol_floor', 'size_breadth', 'dd_half', 'dd_stop', 'dd_factor',
                       'profit_lock', 'lock_trigger', 'lock_gain', 'trail_trigger',
                       'trail_pct', 'partial_trigger', 'partial_frac', 'exit_ma20',
                       'exit_min_hold', 'sector_cap', 'force_close', 'loss_pause_n',
                       'loss_pause_days', 'dd_peak_window', 'eq_vol_target',
                       'eq_vol_floor', 'regime_exit'}
    for k in list(valid):
        if k in cfg:
            kw[k] = cfg[k]
    if kw.get('gate_thr') is None:
        kw['gate_thr'] = -1
    E, T, m = e.run(FULL[0], FULL[1], ret_daily=return_daily, **kw)
    T = add_net(T)
    row = {'cfg': cfg.get('name', '')}
    for label, (a, b) in [('全期', FULL), ('训练', TRAIN), ('检验', TEST)]:
        sm = sub_metrics(E, T, a, b)
        for k, v in sm.items():
            row[f'{label}_{k}'] = v
    row.update(annual_returns(E))
    row['全期_最大单笔亏损'] = T.net_ret.min() if len(T) else np.nan
    row['全期_前5笔占比'] = (T.nlargest(5, 'pnl').pnl.sum() / T.pnl.sum()
                          if len(T) and abs(T.pnl.sum()) > 1e-9 else np.nan)
    row['全期_平均仓位'] = E.exposure.mean()
    row['全期_期末权益'] = E.equity.iloc[-1]
    row['负年份数'] = sum(1 for y in range(2018, 2027) if row.get(f'{y}年', 0) < 0)
    row['最差年'] = min(row.get(f'{y}年', 0) for y in range(2018, 2027))
    return E, T, row


def bootstrap_dd(E, n, sims=2000, block=10, seed=20261023):
    r = E.ret.dropna().values
    if len(r) < block * 2:
        return np.array([np.nan])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(sims):
        x = []
        while len(x) < n:
            i = rng.integers(0, len(r) - block)
            x.extend(r[i:i + block])
        eq = np.cumprod(1 + np.array(x[:n]))
        out.append((eq / np.maximum.accumulate(eq) - 1).min())
    return np.array(out)


def summarize(row):
    cols = ['cfg', '全期_cagr', '全期_maxdd', '全期_trades', '全期_win', '全期_pf',
            '训练_cagr', '训练_maxdd', '训练_trades', '检验_cagr', '检验_maxdd',
            '检验_trades', '全期_最大单笔亏损', '全期_前5笔占比', '负年份数', '最差年']
    return pd.DataFrame([row])[cols].round(4)
