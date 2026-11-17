# -*- coding: utf-8 -*-
"""对2万激进档25.8%做真实性诊断：集中度、路径运气、可成交性、自举、去最优年份。"""
import importlib.util, os, numpy as np, pandas as pd

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = importlib.util.spec_from_file_location('m102', os.path.join(BASE, '02_代码/102_两万组合.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
P = m.P
dates = pd.DatetimeIndex(P['meta']['dates']); codes = P['meta']['codes']
O = P['open'].values; H = P['high'].values; L = P['low'].values; C = P['close'].values
LU = P['limit_up'].values; LD = P['limit_down'].values
OWU = P['one_word_up'].values; OWD = P['one_word_down'].values
MCAP = P['mcap'].values; AMT = P['amt20'].values

KW = dict(slots=2, sizing='fixed', risk=0, maxpos=0.5, heat_cap=0.10,
          gate_thr=0.50, rank='amt', hold=20, start_cash=20000, cost_rate=0.0015)
E, T, met = m.run('2018-01-01', '2026-10-13', **KW)
T = T.copy()
T['entry_i'] = [dates.get_loc(x) for x in T.entry_date]
T['exit_i'] = [dates.get_loc(x) for x in T.exit_date]
T['acct_contrib'] = T.weight * T.ret

print('===== 基本结果 =====')
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()})

# 回撤明细
s = E.equity; peak = s.cummax(); dd = s / peak - 1
tr = dd.idxmin(); pk = s.loc[:tr].idxmax()
rec = s.loc[tr:][s.loc[tr:] >= s.loc[pk]]
print('\n===== 回撤 =====')
print('close peak', pk.date(), 'trough', tr.date(), 'depth', round(dd.min(), 4),
      'recover', rec.index[0].date() if len(rec) else None)
s2 = E.equity_low; dd2 = s2 / peak - 1; tr2 = dd2.idxmin()
print('low   trough', tr2.date(), 'depth', round(dd2.min(), 4))
print('underwater days >5%:', int((dd < -0.05).sum()), '>10%:', int((dd < -0.10).sum()))
print('longest underwater (days):', int(dd.lt(0).groupby((~dd.lt(0)).cumsum()).sum().max()))

# 年度/月度
yr = E.equity.resample('YE').last().pct_change()
yr.iloc[0] = E.equity.loc[E.index.year == E.index.year[0]].iloc[-1] / E.equity.iloc[0] - 1
print('\n===== 年度 =====')
print((yr * 100).round(1).to_string())
mret = E.equity.resample('ME').last().pct_change()
print('\n最差5个月：')
print((mret.sort_values().head(5) * 100).round(1).to_string())
print('最好5个月：')
print((mret.sort_values().tail(5) * 100).round(1).to_string())

# 集中度
print('\n===== 交易集中度 =====')
print('trades', len(T), 'positive', int((T.pnl > 0).sum()), 'negative', int((T.pnl <= 0).sum()))
top = T.sort_values('acct_contrib', ascending=False)
print('top10 account contributions:')
print(top[['code', 'entry_date', 'exit_date', 'ret', 'weight', 'acct_contrib', 'reason']].head(10).round(4).to_string(index=False))
print('top5 sum contrib %.3f, top10 %.3f, total logret %.3f' % (
    top.acct_contrib.head(5).sum(), top.acct_contrib.head(10).sum(), np.log(E.equity.iloc[-1] / 20000)))
# 去掉贡献最大的N笔后的近似总收益（按账户收益线性扣减）
for n in [1, 3, 5, 10]:
    print(f'ex-top{n} approx total: {np.exp(np.log(E.equity.iloc[-1]/20000) - top.acct_contrib.head(n).sum()) - 1:.3f}')
print('\n按年利润：')
T['year'] = pd.DatetimeIndex(T.exit_date).year
print(T.groupby('year').agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), win=('ret', lambda x: (x > 0).mean()),
                            avg=('ret', 'mean')).round(3).to_string())
print('\n按退出原因：')
print(T.groupby('reason').agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), win=('ret', lambda x: (x > 0).mean()),
                              avg=('ret', 'mean')).round(3).to_string())

# 可成交性审计
print('\n===== 可成交性 =====')
T['entry_lu'] = [LU[i, codes.index(c)] if c in codes else False for i, c in zip(T.entry_i, T.code)]
T['entry_owu'] = [OWU[i, codes.index(c)] if c in codes else False for i, c in zip(T.entry_i, T.code)]
T['entry_ld'] = [LD[i, codes.index(c)] if c in codes else False for i, c in zip(T.entry_i, T.code)]
T['exit_ld'] = [LD[i, codes.index(c)] if c in codes else False for i, c in zip(T.exit_i, T.code)]
print('entry limit_up:', int(T.entry_lu.sum()), 'one_word_up:', int(T.entry_owu.sum()),
      'exit limit_down:', int(T.exit_ld.sum()))
for flag in ['entry_lu', 'entry_owu']:
    x = T[T[flag]]
    if len(x): print(flag, 'n', len(x), 'avg ret', round(x.ret.mean(), 4), 'win', round((x.ret > 0).mean(), 3),
                     'pnl', round(x.pnl.sum(), 0))
# 开盘价相对前收的跳空
T['gap'] = [O[i, codes.index(c)] / C[i - 1, codes.index(c)] - 1 for i, c in zip(T.entry_i, T.code)]
print('entry gap: mean %.4f median %.4f p90 %.4f p99 %.4f max %.4f' % (
    T.gap.mean(), T.gap.median(), T.gap.quantile(.9), T.gap.quantile(.99), T.gap.max()))

# 市值、价格、成交额
idx = [codes.index(c) for c in T.code]
T['mcap'] = MCAP[[i for i in T.entry_i], idx]
T['amt20'] = AMT[[i for i in T.entry_i], idx]
T['price'] = [O[i, j] for i, j in zip(T.entry_i, idx)]
print('\n持仓特征(中位数): price %.2f mcap %.2f亿 amt20 %.2f亿' % (
    T.price.median(), T.mcap.median() / 1e8, T.amt20.median() / 1e8))
print('price>100占比 %.3f, mcap<20亿占比 %.3f' % ((T.price > 100).mean(), (T.mcap < 2e9).mean()))

# 平稳块自举：日收益
r = E.ret.values[1:]
rng = np.random.default_rng(42)
Lb = 20; n = len(r); nboot = 2000
cagrs = np.empty(nboot); dds = np.empty(nboot)
for b in range(nboot):
    idxs = []
    while len(idxs) < n:
        st = rng.integers(0, n)
        idxs.extend(((st + np.arange(Lb)) % n).tolist())
    x = r[np.array(idxs[:n])]
    eq = np.cumprod(1 + x)
    cagrs[b] = eq[-1] ** (252 / n) - 1
    dds[b] = (eq / np.maximum.accumulate(eq) - 1).min()
print('\n===== 日收益平稳块自举(%d) =====' % nboot)
print('CAGR p5/p25/p50/p75/p95:', np.round(np.percentile(cagrs, [5, 25, 50, 75, 95]) * 100, 1))
print('MaxDD p5/p25/p50/p75/p95:', np.round(np.percentile(dds, [5, 25, 50, 75, 95]) * 100, 1))

# 去最优年份
print('\n===== 去掉最好年份 =====')
for y in yr.sort_values().tail(3).index.year:
    eq2 = E.equity[E.index.year != y]
    # 近似：把该年收益按1处理（不严谨的粗算），仅看其余年份复利
    other = yr[yr.index.year != y]
    print('去掉%d后其余年份复合: %.3f, 简单复合CAGR %.3f' % (
        y, (1 + other).prod() - 1, (1 + other).prod() ** (252 / (len(other) * 252)) - 1))

# 逐笔收益分布与盈亏比
print('\n===== 逐笔 =====')
print(T.ret.describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
print('avg win %.3f, avg loss %.3f, payoff %.2f' % (
    T.ret[T.ret > 0].mean(), T.ret[T.ret <= 0].mean(), T.ret[T.ret > 0].mean() / -T.ret[T.ret <= 0].mean()))

# 最大连续亏损
w = (T.ret > 0).values
mx = cur = 0
for x in w:
    cur = 0 if x else cur + 1
    mx = max(mx, cur)
print('max consecutive losses:', mx)

# 等权基准
mkt = pd.Series(P['market_index'], index=dates).loc[E.index]
print('\n市场等权指数同期: %.3f, CAGR %.3f, MaxDD %.3f' % (
    mkt.iloc[-1] / mkt.iloc[0] - 1, (mkt.iloc[-1] / mkt.iloc[0]) ** (252 / len(mkt)) - 1,
    (mkt / mkt.cummax() - 1).min()))

out = dict(total=met['total'], cagr=met['cagr'], maxdd=met['maxdd'], maxdd_low=met['maxdd_low'],
           trades=len(T), top5_contrib=top.acct_contrib.head(5).sum(),
           max_consec_loss=mx, entry_owu=int(T.entry_owu.sum()), exit_ld=int(T.exit_ld.sum()))
pd.Series(out).to_csv(os.path.join(BASE, '03_结果/二十五点八_诊断摘要.csv'), encoding='utf-8-sig')
T.to_csv(os.path.join(BASE, '03_结果/二十五点八_逐笔诊断.csv'), index=False, encoding='utf-8-sig')
