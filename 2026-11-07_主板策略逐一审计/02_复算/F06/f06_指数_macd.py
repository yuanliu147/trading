# -*- coding: utf-8 -*-
"""
F06 指数月线/日线 MACD(12,26,9) 择时复算：中证500(000905.SH) 与 沪深300(000300.SH)。
- 信号：DIF>DEA 持有、否则空仓；EMA 播种首收盘，warmup=35 根K线（slow+signal）后才允许持仓。
- 执行：收盘确认、次日开盘切换，单边 0.05%；日频收盘计价净值（切换日拆隔夜/日内两段，无前视）。
- 月线：自然月最后交易日收盘计算 MACD，下一月首日开盘切换，整月持有。
- 窗口：2006-01-04→2026-10-29 全窗；2016-01-01 起子窗（仓位延续，净值重置）。
- 年化：几何，按日历天数 (end/start)^(365.25/days)-1；MDD 按日频净值。
"""
import numpy as np
import pandas as pd
import os

BASE = '/Users/admin/WorkSpace/trading'
OUT = os.path.join(BASE, '2026-11-07_主板策略逐一审计/02_复算/F06')
PANEL = os.path.join(BASE, 'trading_finally/data/index_panel.parquet')
idx = pd.read_parquet(PANEL)

def ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()

def metrics(dates, ret, start=None):
    m = np.ones(len(dates), bool) if start is None else (dates >= start).to_numpy()
    d = dates.to_numpy()[m]; r = ret[m]
    eq = np.cumprod(1 + r)
    yrs = (pd.Timestamp(d[-1]) - pd.Timestamp(d[0])).days / 365.25
    ann = eq[-1] ** (1 / yrs) - 1
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return ann, mdd

rows = []
for code, name in [('000905.SH', '中证500'), ('000300.SH', '沪深300')]:
    g = idx[idx.code == code].sort_values('date').reset_index(drop=True)
    dates = g.date
    c = g.close.to_numpy(); o = g.open.to_numpy()
    n = len(c)
    overnight = np.zeros(n); intraday = np.zeros(n)
    overnight[1:] = o[1:] / c[:-1] - 1
    intraday = c / o - 1
    bh = np.zeros(n); bh[1:] = c[1:] / c[:-1] - 1

    dif = ema(c, 12) - ema(c, 26); dea = ema(dif, 9)
    s_daily = (dif > dea) & (np.arange(n) >= 35)

    month = dates.str[:7]
    mend = ~month.eq(month.shift(-1))  # 当月最后交易日
    mc = c[mend.to_numpy()]
    mdif = ema(mc, 12) - ema(mc, 26); mdea = ema(mdif, 9)
    month_rules = []
    for warm in [0, 35]:
        msig = (mdif > mdea) & (np.arange(len(mc)) >= warm)
        sig_map = dict(zip(month[mend], msig))
        prev_month_sig = pd.Series(month.unique()).sort_values().map(sig_map).shift(1)
        pm = dict(zip(pd.Series(month.unique()).sort_values(), prev_month_sig))
        s_month = month.map(pm).fillna(False).to_numpy(dtype=bool)
        month_rules.append((f'月线DIF>DEA(warm{warm})', s_month))

    for label, sig in [('日线DIF>DEA', s_daily)] + month_rules:
        # ip[t]: 日 t 日内段仓位；op[t]: 隔夜段仓位 = 前一交易日日内仓位
        # 日线：收盘 t-1 信号 → 开盘 t 切换，ip[t]=sig[t-1]
        # 月线：s_month[t] 已是“上一月信号”，即本月应持仓位，ip=s_month（月首日开盘切换）
        if label.startswith('日线'):
            ip = np.zeros(n); ip[1:] = sig[:-1]
        else:
            ip = sig.astype(float)
        op = np.zeros(n); op[1:] = ip[:-1]
        turn = np.abs(ip - op)
        ret = op * overnight + ip * intraday - 0.0005 * turn
        for win, start in [('2006-2026', None), ('2016-2026', '2016-01-01')]:
            a, m = metrics(dates, ret, start)
            ab, mb = metrics(dates, bh, start)
            rows.append({'指数': name, '规则': label, '区间': win,
                         '年化': a, 'MDD': m, '持有年化': ab, '持有MDD': mb,
                         '平均仓位': ip[(dates >= start).to_numpy() if start else slice(None)].mean(),
                         '切换次数': int(turn[(dates >= start).to_numpy() if start else slice(None)].sum())})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'f06_指数_择时.csv'), index=False)
print(res.to_string())
print('DONE')