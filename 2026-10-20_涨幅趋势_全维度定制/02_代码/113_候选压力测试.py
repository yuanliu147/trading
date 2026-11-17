# -*- coding: utf-8 -*-
"""最终候选的严谨性压力测试：成本、参数邻域、自举、去前五笔、DSR、分年/滚动、成交特征。"""
import os, numpy as np, pandas as pd
from scipy.stats import norm
from importlib.util import spec_from_file_location, module_from_spec

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = spec_from_file_location('eng', os.path.join(BASE, '02_代码/108_稳健引擎.py'))
mod = module_from_spec(spec); spec.loader.exec_module(mod)
E = mod.Engine()
dates = E.dates; codes = E.codes

B = dict(slots=2, sizing='fixed', risk=0, maxpos=0.5, heat_cap=0.10, gate_thr=0.50,
         rank='amt', hold=20, start_cash=20000, cost_rate=0.0015)
F = {
 '原25.8%': {},
 '收益优先_40_波动20': dict(maxpos=.40, vol_target=.20),
 '均衡_40_波动20_分批20': dict(maxpos=.40, vol_target=.20, partial_trigger=.20),
 '高收益_40_波动20_暂停': dict(maxpos=.40, vol_target=.20, loss_pause_n=2, loss_pause_days=5),
 '严格3%风险_止损20': dict(sizing='risk', risk=.03, risk_basis='stop', stop_pct=.20, heat_cap=.20),
}

def runf(name, start='2018-01-01', end='2026-10-13', **over):
    a = B.copy(); a.update(F[name]); a.update(over)
    return E.run(start, end, **a)

print('===== 候选基准 =====')
store = {}
for name in F:
    e, t, m = runf(name)
    store[name] = (e, t, m)
    print(f'{name:28s} CAGR {m["cagr"]*100:6.1f}%  DD {m["maxdd"]*100:6.1f}%  低点 {m["maxdd_low"]*100:6.1f}%  '
          f'Sharpe {m["sharpe"]:.2f}  交易 {m["trades"]:3d}  胜率 {m["win"]*100:4.1f}%  PF {m["pf"]:.2f}  '
          f'最差 {m["worst_trade"]*100:5.1f}%  前5 {m["top5_share"]*100:5.0f}%  暴露 {m["avg_exposure"]*100:4.0f}%')

# 年度、回撤日期
print('\n===== 分年收益 =====')
for name in F:
    e = store[name][0]
    y = e.equity.resample('YE').last().pct_change()
    y.iloc[0] = e.equity.loc[e.index.year == e.index.year[0]].iloc[-1] / 20000 - 1
    print(f'{name:28s}', ' '.join(f'{i.year}:{v*100:5.1f}%' for i, v in y.items()))

print('\n===== 回撤日期 =====')
for name in F:
    e = store[name][0]; s = e.equity; dd = s / s.cummax() - 1
    tr = dd.idxmin(); pk = s.loc[:tr].idxmax(); rec = s.loc[tr:][s.loc[tr:] >= s.loc[pk]]
    print(f'{name:28s} peak {pk.date()} trough {tr.date()} {dd.min()*100:6.1f}% '
          f'recover {rec.index[0].date() if len(rec) else "未恢复"} '
          f'underwater>10% {int((dd < -.10).sum())}天')

# 成本
print('\n===== 成本敏感性 =====')
for name in F:
    for c in [0.0015, 0.0025, 0.0035, 0.0050]:
        e, t, m = runf(name, cost_rate=c)
        print(f'{name:28s} 单边{c*100:.2f}%  CAGR {m["cagr"]*100:6.1f}%  DD {m["maxdd"]*100:6.1f}%')

# 参数邻域
print('\n===== 参数邻域 =====')
print('单票上限（波动20 / 波动20+分批20）')
for cap in [0.35, 0.375, 0.40, 0.425, 0.45]:
    a = runf('收益优先_40_波动20', maxpos=cap)[2]
    b = runf('均衡_40_波动20_分批20', maxpos=cap)[2]
    print(f'  cap {cap*100:4.1f}%  纯波动 {a["cagr"]*100:5.1f}%/{a["maxdd"]*100:6.1f}%   '
          f'加分批 {b["cagr"]*100:5.1f}%/{b["maxdd"]*100:6.1f}%')
print('市场波动目标（cap40）')
for vt in [0.15, 0.175, 0.20, 0.225, 0.25]:
    a = runf('收益优先_40_波动20', vol_target=vt)[2]
    print(f'  target {vt*100:4.1f}%  {a["cagr"]*100:5.1f}%/{a["maxdd"]*100:6.1f}%')
print('分批触发（cap40/波动20）')
for pt in [0.15, 0.20, 0.25]:
    a = runf('均衡_40_波动20_分批20', partial_trigger=pt)[2]
    print(f'  +{pt*100:4.0f}%  {a["cagr"]*100:5.1f}%/{a["maxdd"]*100:6.1f}%  前5 {a["top5_share"]*100:3.0f}%')
print('宽度门槛（cap40/波动20）')
for g in [0.50, 0.55, 0.60]:
    a = runf('收益优先_40_波动20', gate_thr=g)[2]
    print(f'  {g*100:3.0f}%  {a["cagr"]*100:5.1f}%/{a["maxdd"]*100:6.1f}%')
print('持有期（cap40/波动20）')
for h in [15, 20, 25]:
    a = runf('收益优先_40_波动20', hold=h)[2]
    print(f'  {h}日  {a["cagr"]*100:5.1f}%/{a["maxdd"]*100:6.1f}%')

# 自举
print('\n===== 平稳块自举(2000, block=20) =====')
rng = np.random.default_rng(7)
for name in ['原25.8%', '收益优先_40_波动20', '均衡_40_波动20_分批20', '高收益_40_波动20_暂停']:
    r = store[name][0].ret.values[1:]; n = len(r); bs = 20; nb = 2000
    cg = np.empty(nb); dd = np.empty(nb)
    for b in range(nb):
        ix = []
        while len(ix) < n:
            st = rng.integers(0, n); ix.extend(((st + np.arange(bs)) % n).tolist())
        x = r[np.array(ix[:n])]; eq = np.cumprod(1 + x)
        cg[b] = eq[-1] ** (252 / n) - 1; dd[b] = (eq / np.maximum.accumulate(eq) - 1).min()
    print(f'{name:28s} CAGR p5/50/95 {np.percentile(cg,[5,50,95])*100}  DD p5/50/95 {np.percentile(dd,[5,50,95])*100}')

# 去前五笔
print('\n===== 去掉贡献最大的5笔 =====')
for name in ['原25.8%', '收益优先_40_波动20', '均衡_40_波动20_分批20', '高收益_40_波动20_暂停']:
    e, t, m = store[name]
    g = t.groupby(['code', 'entry_date']).apply(lambda x: (x.weight * x.ret).sum(), include_groups=False)
    top = g.nlargest(5)
    skip = {(c, pd.Timestamp(d)) for c, d in top.index}
    e2, t2, m2 = runf(name, skip_entries=skip)
    print(f'{name:28s} 原 {m["cagr"]*100:5.1f}%/{m["maxdd"]*100:6.1f}%  '
          f'去5笔 {m2["cagr"]*100:5.1f}%/{m2["maxdd"]*100:6.1f}%  前5贡献 {m["top5_share"]*100:3.0f}%')

# DSR: 以44组方差为参考，尝试次数119
print('\n===== Deflated Sharpe（尝试次数119） =====')
var_sr = 0.000884
N = 119; G = 0.5772156649015329
sr0 = np.sqrt(var_sr) * ((1 - G) * norm.ppf(1 - 1 / N) + G * norm.ppf(1 - 1 / (N * np.e)))
for name in ['原25.8%', '收益优先_40_波动20', '均衡_40_波动20_分批20', '高收益_40_波动20_暂停']:
    r = store[name][0].ret.values[1:]; sr = r.mean() / r.std(); T = len(r)
    sk = pd.Series(r).skew(); ku = pd.Series(r).kurtosis() + 3
    dsr = norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    print(f'{name:28s} 年化SR {sr*np.sqrt(252):.2f}  DSR {dsr:.3f}  (SR0年化 {sr0*np.sqrt(252):.2f})')

# 成交特征
print('\n===== 成交与持仓特征（去重入场） =====')
for name in ['收益优先_40_波动20', '均衡_40_波动20_分批20']:
    e, t, m = store[name]
    t = t.drop_duplicates(['code', 'entry_date'])
    idx = [E.code_idx[c] for c in t.code]; ii = [dates.get_loc(d) for d in t.entry_date]
    gap = [E.O[i, j] / E.C[i - 1, j] - 1 for i, j in zip(ii, idx)]
    price = [E.O[i, j] for i, j in zip(ii, idx)]
    mc = [E.MCAP[i, j] for i, j in zip(ii, idx)]
    amt = [E.AMT[i, j] for i, j in zip(ii, idx)]
    print(f'{name:28s} 入场{len(t)}笔 涨停开盘{int(sum(E.LU[i,j] for i,j in zip(ii,idx)))} '
          f'一字跌停开盘{int(sum(E.OWU[i,j] for i,j in zip(ii,idx)))} '
          f'跳空中位{np.median(gap)*100:.2f}% 最大{max(gap)*100:.2f}%')
    print(f'  价格中位 {np.median(price):.1f}  市值中位 {np.median(mc)/1e8:.0f}亿  成交额中位 {np.median(amt)/1e8:.2f}亿  '
          f'价格>80占比 {np.mean(np.array(price)>80)*100:.0f}%')

# 滚动三年
print('\n===== 滚动三年（非重叠） =====')
for name in ['收益优先_40_波动20', '均衡_40_波动20_分批20']:
    e = store[name][0]; s = e.equity
    print(name)
    for y0 in [2018, 2021, 2024]:
        x = s[(s.index.year >= y0) & (s.index.year <= y0 + 2)]
        if len(x) < 100: continue
        cg = (x.iloc[-1] / x.iloc[0]) ** (252 / len(x)) - 1
        dd = (x / x.cummax() - 1).min()
        print(f'  {y0}-{y0+2}: CAGR {cg*100:5.1f}%  DD {dd*100:6.1f}%')

# 基准
mkt = pd.Series(E.MKT, index=dates).loc[store['原25.8%'][0].index]
print('\n等权全A：总 %.1f%%  CAGR %.1f%%  DD %.1f%%' % (
    (mkt.iloc[-1] / mkt.iloc[0] - 1) * 100,
    ((mkt.iloc[-1] / mkt.iloc[0]) ** (252 / len(mkt)) - 1) * 100,
    ((mkt / mkt.cummax() - 1).min()) * 100))
