# -*- coding: utf-8 -*-
"""
实际价、100 股整手、含费用回放（修正版）。

修正旧版 actual_engine.py 的两处问题：
1. 送转调整的因子基准不一致（旧版入场用 hfq_close/实际开盘，除权日用 hfq_close/实际收盘，
   002365 被调成 1.5476 倍，正确是 10 转 5 的 1.5 倍）；
2. 盈亏成本用“调整后股数 × 原始买入价”，把送转股又按原价买了一遍；
   正确成本是原始买入现金（含买入费用）。

输出：
- 03_结果/实际价_修正版_逐笔.csv
- 03_结果/交易记录_中文_修正版.csv
- 03_结果/实际价_修正版_每日权益.csv
- 03_结果/实际价_修正版_指标.json
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '02_代码')
from panic_lib import load  # noqa: E402

OUT = '03_结果'
CAP = 100000.0

D = load()
df = D['df']
dates = D['dates']
syms = D['syms']
di = {pd.Timestamp(d): i for i, d in enumerate(dates)}
si = {s: i for i, s in enumerate(syms)}

# 实际价与后复权收盘矩阵
AO = np.full((len(dates), len(syms)), np.nan)
AC = np.full_like(AO, np.nan)
AH = np.full_like(AO, np.nan)
AL = np.full_like(AO, np.nan)
HC = np.full_like(AO, np.nan)
for c, arr in [('open', AO), ('close', AC), ('high', AH), ('low', AL),
               ('hfq_close', HC)]:
    arr[df.d.values, df.s.values] = df[c].values

trh = pd.read_csv(f'{OUT}/最终_逐笔.csv',
                  parse_dates=['entry_date', 'exit_date', 'signal_date'])
trh = trh.sort_values(['entry_date', 'thscode']).reset_index(drop=True)
entries = {}
for _, t in trh.iterrows():
    entries.setdefault(pd.Timestamp(t.entry_date), []).append(t)
exits = {}
for _, t in trh.iterrows():
    exits.setdefault(pd.Timestamp(t.exit_date), []).append(t)

start, end = min(entries), max(exits)
cash = CAP
last = {}
pos = []
trades = []
eqrows = []

# 已知送转：002365.SZ 2018-05-15 十转五
SPLITS = {('002365.SZ', pd.Timestamp('2018-05-15')): 1.5}


def sell_cost(sell):
    return max(5.0, sell * 0.00025) + sell * 0.0005 + sell * 0.001


for d in dates:
    d = pd.Timestamp(d)
    if d < start or d > end:
        continue
    i = di[d]

    # 开盘入场
    for t in entries.get(d, []):
        s = si[t.thscode]
        px = AO[i, s]
        if np.isnan(px) or px <= 0:
            continue
        equity = cash + sum(p['shares'] * last.get(p['s'], p['entry_px'])
                            for p in pos)
        alloc = min(equity / 2, cash * 0.999)
        shares = int(alloc / px / 100) * 100
        if shares <= 0:
            continue
        buy = shares * px
        cb = max(5.0, buy * 0.00025)
        slip = buy * 0.001
        cost_basis = buy + cb + slip
        cash -= cost_basis
        pos.append(dict(
            s=s, thscode=t.thscode, shares=shares,
            entry_date=d, entry_px=px, cost_basis=cost_basis,
            signal_date=t.signal_date, factor=HC[i, s] / AC[i, s],
            split_factor=1.0, dividend=0.0, entry_fees=cb + slip,
        ))
        last[s] = px

    # 送转/分红调整，再按收盘盯市
    for p in pos:
        s = p['s']
        prev_close = AC[i - 1, s] if i > 0 else np.nan
        px = AC[i, s]
        if not np.isnan(px):
            last[s] = px
        hc = HC[i, s]
        if np.isnan(px) or px <= 0 or np.isnan(hc):
            continue
        f = hc / px
        ratio = f / p['factor'] if p['factor'] > 0 else 1.0
        if ratio <= 1.02:
            continue
        split = SPLITS.get((p['thscode'], d), 1.0)
        div = 0.0
        if not np.isnan(prev_close) and prev_close > 0:
            residual = ratio / split
            if residual > 1.0:
                # hfq 因子里除送转外的部分按现金分红处理
                div = prev_close * (1.0 - 1.0 / residual)
        if div > 0:
            cash += p['shares'] * div
            p['dividend'] += p['shares'] * div
        if split != 1.0:
            p['shares'] *= split
            p['split_factor'] *= split
        p['factor'] = f
        last[s] = px

    # 收盘/止盈退出
    for t in exits.get(d, []):
        p = next((x for x in pos if x['thscode'] == t.thscode
                  and x['entry_date'] == pd.Timestamp(t.entry_date)), None)
        if p is None:
            continue
        s = p['s']
        ac, ah, aop = AC[i, s], AH[i, s], AO[i, s]
        if t.exit_reason == 'target' and not np.isnan(ah):
            target = p['entry_px'] * 1.20
            fill = max(aop, target) if ah >= target and not np.isnan(aop) else ac
        else:
            fill = ac
        if np.isnan(fill) or fill <= 0:
            fill = last.get(s, p['entry_px'])
        sell = p['shares'] * fill
        sc = sell_cost(sell)
        cash += sell - sc
        pnl = sell + p['dividend'] - p['cost_basis'] - sc
        ret = pnl / p['cost_basis']
        trades.append(dict(
            thscode=p['thscode'], signal_date=p['signal_date'],
            entry_date=p['entry_date'], exit_date=d,
            entry_px=p['entry_px'], exit_px=fill, shares=p['shares'],
            pnl=pnl, ret=ret, exit_reason=t.exit_reason,
            hfq_ret=t.ret, cost_basis=p['cost_basis'],
            split_factor=p['split_factor'], dividend=p['dividend'],
        ))
        pos.remove(p)
        last[s] = fill

    eq = cash + sum(p['shares'] * last.get(p['s'], p['entry_px']) for p in pos)
    eqrows.append(dict(date=d, equity=eq, cash=cash, npos=len(pos)))

eqa = pd.DataFrame(eqrows)
tra = pd.DataFrame(trades)
eqa.to_csv(f'{OUT}/实际价_修正版_每日权益.csv', index=False)
tra.to_csv(f'{OUT}/实际价_修正版_逐笔.csv', index=False)

e = eqa.set_index('date').equity
yrs = (e.index[-1] - e.index[0]).days / 365.25
cagr = (e.iloc[-1] / CAP) ** (1 / yrs) - 1
dd = float((e / e.cummax() - 1).min())
win = float((tra.ret > 0).mean())
avg = float(tra.ret.mean())
metrics = dict(
    start=str(e.index[0].date()), end=str(e.index[-1].date()),
    final=float(e.iloc[-1]), cagr=float(cagr), dd=dd,
    n=int(len(tra)), win=win, avg=avg,
    sum_pnl=float(tra.pnl.sum()),
    exit_reasons=tra.exit_reason.value_counts().to_dict(),
    avg_hfq_ret=float(tra.hfq_ret.mean()),
    mean_abs_diff=float((tra.ret - tra.hfq_ret).abs().mean()),
)
json.dump(metrics, open(f'{OUT}/实际价_修正版_指标.json', 'w'),
          ensure_ascii=False, indent=2)

# 中文逐笔
cn = tra.copy()
cn['退出原因'] = cn.exit_reason.map(
    {'time': '15日到期', 'target': '+20%止盈', 'eod': '期末未平'})
cn = cn.rename(columns={
    'thscode': '代码', 'signal_date': '信号日', 'entry_date': '买入日',
    'exit_date': '卖出/期末日', 'entry_px': '买入价', 'exit_px': '卖出价',
    'shares': '股数', 'pnl': '净盈亏(元)', 'ret': '净收益率',
    'hfq_ret': '复权收益率', 'cost_basis': '买入成本(元)',
    'split_factor': '送转倍数', 'dividend': '现金分红(元)',
})
cn = cn[['代码', '信号日', '买入日', '卖出/期末日', '买入价', '卖出价',
         '股数', '净盈亏(元)', '净收益率', '退出原因', '复权收益率',
         '买入成本(元)', '送转倍数', '现金分红(元)']]
cn.to_csv(f'{OUT}/交易记录_中文_修正版.csv', index=False, encoding='utf-8-sig')

print(json.dumps(metrics, ensure_ascii=False, indent=2))
print('\n002365')
print(cn[cn['代码'] == '002365.SZ'].to_string(index=False))
print('\n002607')
print(cn[cn['代码'] == '002607.SZ'].to_string(index=False))
print('\nold 002365')
old = pd.read_csv(f'{OUT}/旧版_不可靠/实际价_逐笔_旧版.csv')
print(old[old.thscode == '002365.SZ'].to_string(index=False))
