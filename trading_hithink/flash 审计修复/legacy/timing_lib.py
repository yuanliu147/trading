# -*- coding: utf-8 -*-
"""
不同入场时点回测：复用“主板恐慌洗盘反转”信号，只改变入场时点。

核心口径与母策略一致：
- 信号日 T 收盘确认五条件；
- 2 仓、每仓 50% 当前权益、无信号空仓；
- +20% 止盈（T+1 起，盘中触价），持有 15 个交易日收盘卖出；
- 成本 0.175%/边；一字涨停/停牌不买；无普通止损。

本文件在 panic_lib.py 的基础上扩展：
- nth：信号后第 N 个交易日入场（N=1 次日；N=3 为母策略基准）；
- fill：open/close/high/low，分别表示开盘价、收盘价、当日最高价、当日最低价；
- 同日收盘（nth=0, fill=close）作为前视参照单独标注；
- 返回逐笔、权益、跳过统计，便于审计。
"""
import numpy as np
import pandas as pd

PANEL = '01_数据/research_panel.parquet'

COLS = ['thscode', 'date', 'open', 'close', 'high', 'low',
        'hfq_open', 'hfq_close', 'hfq_high', 'hfq_low',
        'next_actual_open', 'next_up_limit_10', 'next_down_limit_10',
        'ret1', 'ret_o5', 'ret_o10', 'ret_o20', 'mfe10', 'mae10',
        'amount20', 'amount_ratio1_20', 'amount_ratio5_20',
        'return_5d', 'return_10d', 'return_20d', 'return_60d',
        'return_120d', 'return_250d',
        'atr20_pct', 'volatility20', 'drawdown60', 'drawdown250',
        'price_position60', 'close_location', 'rsi14', 'kdj_j',
        'up_days20', 'near_limit_up_days20', 'limit_up_10',
        'broken_limit_up_10', 'up_limit_05',
        'bullish_engulfing', 'hammer_shape', 'range20_pct',
        'week_return12', 'week_drawdown12', 'month_close_ma6',
        'market_return', 'market_return5', 'market_return20',
        'breadth20', 'breadth60', 'sentiment_score',
        'limit_up10_share', 'advance_share']


def load(panel=PANEL):
    """加载研究面板，构造回测矩阵。"""
    df = pd.read_parquet(panel, columns=COLS)
    df['date'] = pd.to_datetime(df.date)
    df = df.sort_values(['date', 'thscode']).reset_index(drop=True)

    df['liquid'] = df.amount20 >= 10_000_000
    df['st_proxy'] = df.up_limit_05 == 1

    dates = np.sort(df.date.unique())
    syms = np.sort(df.thscode.unique())
    di = {d: i for i, d in enumerate(dates)}
    si = {s: i for i, s in enumerate(syms)}
    df['d'] = df.date.map(di)
    df['s'] = df.thscode.map(si)

    shape = (len(dates), len(syms))
    O = np.full(shape, np.nan, dtype=np.float32)   # 后复权开盘
    C = np.full(shape, np.nan, dtype=np.float32)   # 后复权收盘
    H = np.full(shape, np.nan, dtype=np.float32)   # 后复权最高
    L = np.full(shape, np.nan, dtype=np.float32)   # 后复权最低
    AO = np.full(shape, np.nan, dtype=np.float32)  # 实际开盘
    AC = np.full(shape, np.nan, dtype=np.float32)  # 实际收盘
    AH = np.full(shape, np.nan, dtype=np.float32)  # 实际最高
    AL = np.full(shape, np.nan, dtype=np.float32)  # 实际最低

    O[df.d.values, df.s.values] = df.hfq_open.values
    C[df.d.values, df.s.values] = df.hfq_close.values
    H[df.d.values, df.s.values] = df.hfq_high.values
    L[df.d.values, df.s.values] = df.hfq_low.values
    AO[df.d.values, df.s.values] = df.open.values
    AC[df.d.values, df.s.values] = df.close.values
    AH[df.d.values, df.s.values] = df.high.values
    AL[df.d.values, df.s.values] = df.low.values

    base = df[df.liquid & ~df.st_proxy].copy()
    return dict(df=df, dates=dates, syms=syms, O=O, C=C, H=H, L=L,
                AO=AO, AC=AC, AH=AH, AL=AL, base=base)


def add_signal(D, name='custom',
               dd=-0.45, rsi=35, cl=0.45, mkt=-0.05, r250=0.0):
    """母策略五条件，写入 base['ev_<name>']。"""
    b = D['base']
    m = ((b.drawdown250 < dd) & (b.return_250d > r250) &
         (b.rsi14 < rsi) & (b.close_location > cl) &
         (b.market_return20 < mkt))
    b['ev_' + name] = m.values
    return m


def _px(D, fill):
    return {'open': D['O'], 'close': D['C'],
            'high': D['H'], 'low': D['L']}[fill]


def _entry_px(D, i, s, fill):
    """入场价与可买性检查；返回 (价格, 跳过原因)。"""
    O, C, AO, AC, AH, AL = D['O'], D['C'], D['AO'], D['AC'], D['AH'], D['AL']
    # 停牌/无行情：开盘模式看开盘，收盘模式看收盘
    if fill in ('open', 'high', 'low'):
        if np.isnan(O[i, s]) or np.isnan(C[i, s]):
            return np.nan, '停牌/无行情'
    else:
        if np.isnan(C[i, s]):
            return np.nan, '停牌/无行情'
    px = _px(D, fill)[i, s]
    if np.isnan(px) or px <= 0:
        return np.nan, '无有效价格'
    if i > 0:
        pc = AC[i - 1, s]
        if not np.isnan(pc) and pc > 0:
            lim = pc * 1.098
            if fill in ('open', 'close'):
                ap = AO[i, s] if fill == 'open' else AC[i, s]
                if not np.isnan(ap) and ap >= lim:
                    return np.nan, '一字涨停/涨停价'
            else:
                # 最高/最低价模式：只排除“一字板”（全天最低都在涨停）
                al = AL[i, s]
                if not np.isnan(al) and al >= lim:
                    return np.nan, '一字涨停/涨停价'
    return float(px), ''


def simulate_timing(D, ev='custom', nth=3, fill='open', hold=15, slots=2,
                    score='drawdown250', tp=0.20, cost=0.00175,
                    start='2017-01-01', end='2026-10-23',
                    max_signals_per_day=None):
    """
    nth: 信号后第 N 个交易日。nth=0 仅用于 fill='close'（同日收盘，前视参照）。
    fill: open/close/high/low（后复权价）。
    返回 eq, tr, skips。
    """
    if nth < 0:
        raise ValueError('nth 必须 >= 0')
    if nth == 0 and fill != 'close':
        raise ValueError('nth=0 只能配合 fill=close（同日收盘）')

    base, dates = D['base'], D['dates']
    n_days, n_syms = len(dates), len(D['syms'])
    sig = base[base['ev_' + ev] &
               (base.date >= pd.Timestamp(start)) &
               (base.date <= pd.Timestamp(end))].copy()
    if sig.empty:
        return None, None, {}

    ascending = score in ('rsi14', 'kdj_j', 'amount20', 'return_250d',
                          'drawdown250', 'ret1', 'close_location')
    sig['rk'] = sig.groupby('date')[score].rank(ascending=ascending, method='first')
    if max_signals_per_day:
        sig = sig[sig.rk <= max_signals_per_day]
    sig = sig.sort_values(['date', 'rk'])

    sigday = {}
    for d, g in sig.groupby('d'):
        sigday[int(d)] = [(int(r.s), float(r.rk)) for r in g.head(slots * 5).itertuples()]

    i0 = int(np.searchsorted(dates, np.datetime64(start)))
    i1 = int(np.searchsorted(dates, np.datetime64(end), side='right')) - 1

    mtm, pos, trades = [], [], []
    cash, prev_equity = 1.0, 1.0
    skips = {'一字涨停/涨停价': 0, '停牌/无行情': 0, '无有效价格': 0, '资金不足': 0}
    entry_at_close = (fill == 'close')

    def do_entries(i):
        nonlocal cash
        j = i - nth
        if j < i0 or j not in sigday:
            return
        held = {p['s'] for p in pos}
        cand = [x for x in sigday[j] if x[0] not in held]
        # 与母引擎一致：先按名次截取，再检查涨停/停牌；被跳过不递补
        take = cand[:max(0, slots - len(pos))]
        if not take:
            return
        alloc_each = prev_equity / slots
        for s, rk in take:
            px, why = _entry_px(D, i, s, fill)
            if np.isnan(px):
                skips[why] = skips.get(why, 0) + 1
                continue
            alloc = min(alloc_each, cash)
            if alloc <= 1e-4:
                skips['资金不足'] += 1
                continue
            inv = alloc * (1 - cost)
            pos.append(dict(
                s=s, thscode=D['syms'][s],
                signal_date=dates[j], entry_date=dates[i], entry_i=i,
                entry_px=px, last_px=px, value=inv,
                days=1 if entry_at_close else 0,
                rk=rk, alloc=alloc, high=px,
                equity_at_entry=prev_equity,
                alloc_frac=(alloc / prev_equity if prev_equity > 0 else np.nan),
                fill=fill, nth=nth
            ))
            cash -= alloc

    for i in range(i0, i1 + 1):
        if not entry_at_close:
            do_entries(i)

        exit_now = []
        for p in pos:
            s = p['s']
            hi, lo, op, px = D['H'][i, s], D['L'][i, s], D['O'][i, s], D['C'][i, s]
            if not np.isnan(px) and p['last_px'] > 0:
                p['value'] *= px / p['last_px']
                p['last_px'] = px
                p['high'] = max(p['high'], px)
            p['days'] += 1

            reason = None
            fill_px = None
            if (tp is not None and not np.isnan(hi) and i > p['entry_i']):
                trig = p['entry_px'] * (1 + tp)
                if hi >= trig:
                    reason = 'target'
                    fill_px = max(op, trig) if not np.isnan(op) else trig
            if reason is None and p['days'] >= hold:
                reason = 'time'
                fill_px = px
            if reason:
                if fill_px is None or np.isnan(fill_px):
                    fill_px = p['last_px']
                p['exit_fill'] = fill_px
                p['exit_reason'] = reason
                exit_now.append(p)

        keep = []
        for p in pos:
            if p in exit_now:
                fill_px = p['exit_fill']
                if np.isnan(fill_px) or fill_px <= 0:
                    fill_px = p['last_px']
                realized = p['value'] * (fill_px / p['last_px'])
                cash += realized * (1 - cost)
                p['ret'] = fill_px / p['entry_px'] * (1 - cost) * (1 - cost) - 1
                p['exit_date'] = dates[i]
                p['last_px'] = fill_px
                trades.append(p)
            else:
                keep.append(p)
        pos = keep

        if entry_at_close:
            do_entries(i)

        eq = cash + sum(p['value'] for p in pos)
        mtm.append((dates[i], eq, cash, len(pos)))
        prev_equity = eq

    for p in pos:
        p['ret'] = p['last_px'] / p['entry_px'] * (1 - cost) * (1 - cost) - 1
        p['exit_date'] = dates[i1]
        p['exit_reason'] = 'eod'
        trades.append(p)

    eqdf = pd.DataFrame(mtm, columns=['date', 'equity', 'cash', 'npos'])
    tr = pd.DataFrame(trades)
    if not tr.empty:
        attrs = base[['date', 'thscode', 'drawdown250', 'rsi14',
                      'close_location', 'return_250d', 'amount20',
                      'market_return20']].copy()
        attrs = attrs.rename(columns={'date': 'signal_date'})
        tr = tr.merge(attrs, on=['signal_date', 'thscode'], how='left')
        tr['exit_date'] = pd.to_datetime(tr['exit_date'])
        tr['signal_date'] = pd.to_datetime(tr['signal_date'])
        tr['entry_date'] = pd.to_datetime(tr['entry_date'])
    return eqdf, tr, skips


def metrics(eq, tr):
    if eq is None or len(eq) < 10 or tr is None or len(tr) == 0:
        return {}
    e = eq.set_index('date').equity
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = e.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = (e / e.cummax() - 1).min()
    r = e.pct_change().dropna()
    pos_sum = tr.loc[tr.ret > 0, 'ret'].sum()
    neg_sum = tr.loc[tr.ret < 0, 'ret'].sum()
    pf = pos_sum / abs(neg_sum) if neg_sum < 0 else np.inf
    return dict(
        cagr=float(cagr), dd=float(dd),
        calmar=float(cagr / abs(dd)) if dd < 0 else np.nan,
        sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
        n=int(len(tr)), win=float((tr.ret > 0).mean()),
        avg=float(tr.ret.mean()), median=float(tr.ret.median()),
        best=float(tr.ret.max()), worst=float(tr.ret.min()),
        pf=float(pf),
        expo=float((eq.npos > 0).mean()),
        final=float(e.iloc[-1]),
        total_ret=float(e.iloc[-1] - 1),
        profit_100k=float((e.iloc[-1] - 1) * 100000),
        avg_win=float(tr.loc[tr.ret > 0, 'ret'].mean()) if (tr.ret > 0).any() else np.nan,
        avg_loss=float(tr.loc[tr.ret < 0, 'ret'].mean()) if (tr.ret < 0).any() else np.nan,
        n_target=int((tr.exit_reason == 'target').sum()),
        n_time=int((tr.exit_reason == 'time').sum()),
        n_eod=int((tr.exit_reason == 'eod').sum()),
    )


def annual(eq):
    e = eq.set_index('date').equity
    ye = e.resample('YE').last()
    a = ye.pct_change()
    a.iloc[0] = ye.iloc[0] - 1.0
    return a


def dd_spells(eq):
    e = eq.set_index('date').equity
    dds = e / e.cummax() - 1
    out, inset, start, low, lowd = [], False, None, 0.0, None
    for d, v in dds.items():
        if v < 0 and not inset:
            inset, start, low, lowd = True, d, v, d
        elif v < 0 and inset:
            if v < low:
                low, lowd = v, d
        elif v >= 0 and inset:
            out.append(dict(start=start, trough=lowd, recovery=d,
                            depth=low, days=(d - start).days))
            inset = False
    if inset:
        out.append(dict(start=start, trough=lowd, recovery=None,
                        depth=low, days=(e.index[-1] - start).days))
    return pd.DataFrame(out)


def standalone_trade(D, s, j, nth, fill='open', hold=15, tp=0.20,
                     cost=0.00175):
    """
    单信号独立收益（无仓位/名额约束），用于剥离“2 仓占位”的影响。
    j 为信号日索引，s 为股票索引。
    """
    n_days = len(D['dates'])
    i = j + nth
    if i >= n_days:
        return dict(status='数据结束')
    px, why = _entry_px(D, i, s, fill)
    if np.isnan(px):
        return dict(status=why)
    entry_i = i
    entry_px = px
    last_px = px
    exit_px, exit_i, reason = None, None, None
    end_i = min(entry_i + hold - 1, n_days - 1)
    for k in range(entry_i + 1, end_i + 1):
        hi, op = D['H'][k, s], D['O'][k, s]
        if tp is not None and not np.isnan(hi):
            trig = entry_px * (1 + tp)
            if hi >= trig:
                exit_px = max(op, trig) if not np.isnan(op) else trig
                exit_i, reason = k, 'target'
                break
        ck = D['C'][k, s]
        if not np.isnan(ck):
            last_px = ck
    if reason is None:
        ck = D['C'][end_i, s]
        exit_px = ck if not np.isnan(ck) else last_px
        exit_i, reason = end_i, 'time'
        if np.isnan(ck):
            reason = 'time_suspended'
    ret = exit_px / entry_px * (1 - cost) * (1 - cost) - 1
    sig_close = D['C'][j, s]
    pre_ret = entry_px / sig_close - 1 if (not np.isnan(sig_close) and sig_close > 0) else np.nan
    return dict(status='ok', entry_i=entry_i, exit_i=exit_i,
                entry_date=D['dates'][entry_i], exit_date=D['dates'][exit_i],
                entry_px=entry_px, exit_px=exit_px, ret=ret,
                pre_ret=pre_ret, exit_reason=reason)
