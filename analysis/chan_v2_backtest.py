#!/usr/bin/env python3
"""真缠论版回测(v2): 用确定性缠论引擎(chan_engine)的买卖点替换上篇MA5/MA10代理规则
对比对象: chan_system_backtest.py(v1 代理版)

与 v1 保持一致: 总开关(上证MACD 0轴)、仓位(1/6首笔,≤2只,现金≥1/3)、T+1、费用、跌停顺延、尾盘=收盘
不同点:
- 买入: 引擎三买/二买信号(默认), --buy1 加一买; 默认不要求年线入场券(买点本身含位置约束), --ticket 可叠加
- 止损: 二买=一买低点, 三买=中枢ZG, 一买=背驰段低点; 收盘破止损→次日清仓
- 止盈/离场: 一卖(顶背驰)→次日减半; 三卖(反抽不回中枢)→次日清仓; 收盘破MA10→次日清仓(机械底线)
用法: python3 chan_v2_backtest.py [--start 2018-01-01] [--end 2026-10-09] [--equity 10000] [--buy1] [--ticket] [--no-fee]
"""
import argparse, os, time
import numpy as np
import pandas as pd
from chan_engine import analyze
from chan_system_backtest import (universe, load_all, load_index, load_st_names, report)

HERE = os.path.dirname(os.path.abspath(__file__))
COMMISSION, MIN_COMM, STAMP, SLIP = 0.00025, 5.0, 0.0005, 0.0005
MAX_POS, FIRST_FRAC, CASH_FLOOR = 2, 1/6, 1/3


def run(start, end, equity0, use_buy1=False, use_ticket=False):
    t0 = time.time()
    st_set = load_st_names()
    syms = [s for s in universe() if s not in st_set and not s.endswith('.BJ')]
    print(f"宇宙: {len(syms)} 只", flush=True)
    df = load_all(syms)
    print(f"行情: {len(df)} 行, {time.time()-t0:.0f}s", flush=True)

    # 入场券需要的列(可选)
    sig_rows, bars_map = [], {}
    for i, (code, g) in enumerate(df.groupby('thscode')):
        r = analyze(code, g)
        c = r['close'].to_numpy()
        r['ma250'] = pd.Series(c).rolling(250).mean().to_numpy()
        r['ret250'] = c / pd.Series(c).shift(250).to_numpy() - 1
        bars_map[code] = r.set_index('date')
        buy_mask = r['sig2'] | r['sig3'] | (r['sig1'] if use_buy1 else False)
        s = r[buy_mask]
        for _, row in s.iterrows():
            if row['sig3']:
                typ, stop = '三买', row['stop3']
            elif row['sig2']:
                typ, stop = '二买', row['stop2']
            else:
                typ, stop = '一买', row['stop1']
            sig_rows.append((row['date'], code, typ, stop, row['vol_x']))
        if (i + 1) % 500 == 0:
            print(f"  引擎 {i+1}/{len(syms)}, {time.time()-t0:.0f}s", flush=True)
    sigs = pd.DataFrame(sig_rows, columns=['date', 'code', 'type', 'stop', 'vol_x'])
    sigs['date'] = pd.to_datetime(sigs['date'])
    sig_by_day = {d: g.sort_values('vol_x', ascending=False).to_dict('records')
                  for d, g in sigs.groupby('date')}
    print(f"信号: {len(sigs)} 条 {sigs.type.value_counts().to_dict()}, {time.time()-t0:.0f}s", flush=True)

    idx = load_index()
    cal = [d for d in idx.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    cash, equity = equity0, equity0
    pos, pending, trades, curve = {}, [], [], []

    def bar_of(code, d):
        b = bars_map.get(code)
        return b.loc[d] if b is not None and d in b.index else None

    for d in cal:
        # 1. 执行昨日挂卖
        still = []
        for od in pending:
            r = bar_of(od['code'], d)
            if r is None or r['limit_dn']:  # 无数据(停牌)或跌停 → 顺延
                still.append(od)
                continue
            p = pos.get(od['code'])
            if p is None:
                continue
            px = r['close'] * (1 - SLIP)
            sh = min(od['shares'], p['shares'])
            amt = sh * px
            fee = max(amt * COMMISSION, MIN_COMM) + amt * (STAMP + SLIP)
            cash += amt - fee
            p['shares'] -= sh
            trades.append(dict(code=od['code'], buy_date=p['buy_date'], sell_date=d, sell_px=px,
                               shares=sh, reason=od['reason'], type=p['type'],
                               pnl=(px - p['cost']) * sh - fee, ret=px / p['cost'] - 1))
            if p['shares'] <= 0:
                del pos[od['code']]
        pending = still

        for code, p in pos.items():
            r = bar_of(code, d)
            if r is not None:
                p['last'] = r['close']
        equity = cash + sum(p['shares'] * p['last'] for p in pos.values())
        curve.append((d, equity))

        # 2. 卖出触发(次日执行)
        pend_codes = {od['code'] for od in pending}
        for code, p in list(pos.items()):
            if p['buy_date'] >= d or code in pend_codes:
                continue
            r = bar_of(code, d)
            if r is None:
                continue
            reason, kind = None, 'all'
            if r['close'] < p['stop']:
                reason = '理由失效'
            elif r['close'] < r['ma10']:
                reason = '破10日线'
            elif r['sig3sell']:
                reason = '三卖'
            elif r['sig1sell'] and not p['halved']:
                reason, kind = '一卖减半', 'half'
            if reason:
                sh = p['shares'] if kind == 'all' else int(p['shares'] // 200) * 100
                if sh >= 100:
                    pending.append(dict(code=code, shares=sh, reason=reason))
                    pend_codes.add(code)
                    if kind == 'half':
                        p['halved'] = True

        # 3. 买入(总开关开)
        if bool(idx.loc[d, 'gate_on']):
            for od in sig_by_day.get(d, []):
                if len(pos) >= MAX_POS:
                    break
                code = od['code']
                if code in pos:
                    continue
                r = bar_of(code, d)
                if r is None or r['limit_up'] or r['limit_dn'] or r['chg'] > 0.05 or r['chg'] < -0.05:
                    continue
                if use_ticket and not (r['close'] > r['ma250'] and r['ret250'] < 1.0):
                    continue
                px = r['close'] * (1 + SLIP)
                sh = int(equity * FIRST_FRAC / px // 100) * 100
                amt = sh * px
                fee = max(amt * COMMISSION, MIN_COMM) + amt * SLIP
                if sh < 100 or cash - amt - fee < equity * CASH_FLOOR:
                    continue
                cash -= amt + fee
                pos[code] = dict(shares=sh, cost=px, stop=od['stop'], type=od['type'],
                                 halved=False, buy_date=d, last=r['close'])
                trades.append(dict(code=code, buy_date=d, sell_date=None, sell_px=None, shares=sh,
                                   reason=f"买({od['type']})", type=od['type'], pnl=None, ret=None))
        equity = cash + sum(p['shares'] * p['last'] for p in pos.values())
        curve[-1] = (d, equity)

    last_d = cal[-1]
    for code, p in list(pos.items()):
        px = p['last'] * (1 - SLIP)
        amt = p['shares'] * px
        fee = max(amt * COMMISSION, MIN_COMM) + amt * (STAMP + SLIP)
        cash += amt - fee
        trades.append(dict(code=code, buy_date=p['buy_date'], sell_date=last_d, sell_px=px,
                           shares=p['shares'], reason='期末', type=p['type'],
                           pnl=(px - p['cost']) * p['shares'] - fee, ret=px / p['cost'] - 1))
    curve[-1] = (last_d, cash)
    eq = pd.DataFrame(curve, columns=['date', 'equity']).drop_duplicates('date', keep='last')
    return eq, pd.DataFrame(trades)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2018-01-01')
    ap.add_argument('--end', default='2026-10-09')
    ap.add_argument('--equity', type=float, default=10000)
    ap.add_argument('--buy1', action='store_true', help='一买也开仓')
    ap.add_argument('--ticket', action='store_true', help='叠加年线入场券')
    ap.add_argument('--no-fee', action='store_true')
    a = ap.parse_args()
    if a.no_fee:
        COMMISSION = MIN_COMM = STAMP = SLIP = 0.0
    eq, tr = run(a.start, a.end, a.equity, a.buy1, a.ticket)
    tag = 'v2' if not a.no_fee else 'v2_nofee'
    eq, closed = report(eq, tr, a.equity, a.start, a.end)
    if len(closed):
        print(f"\n按买点类型:\n{closed.groupby('type')['pnl'].agg(['count','sum','mean']).to_string()}")
        print(f"按类型胜率:\n{closed.groupby('type')['ret'].apply(lambda s:(s>0).mean()).to_string()}")
    eq.to_csv(os.path.join(HERE, f'chan_{tag}_eq_daily.csv'))
    tr.to_csv(os.path.join(HERE, f'chan_{tag}_trades.csv'), index=False)
    print(f"\n已保存 chan_{tag}_eq_daily.csv / chan_{tag}_trades.csv")
