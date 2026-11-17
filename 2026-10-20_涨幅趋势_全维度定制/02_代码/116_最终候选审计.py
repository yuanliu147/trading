# -*- coding: utf-8 -*-
"""最终候选 W1/W2/W4 的完整审计与图表数据导出。"""
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
 '旧25.8%': {},
 '回撤优先_42.5_波动20': dict(maxpos=.425, vol_target=.20),
 '收益优先_45_波动20': dict(maxpos=.45, vol_target=.20),
 '均衡_45_波动20_分批20': dict(maxpos=.45, vol_target=.20, partial_trigger=.20),
 '高收益_45_波动20_暂停': dict(maxpos=.45, vol_target=.20, loss_pause_n=2, loss_pause_days=5),
 '严格3%风险_止损20': dict(sizing='risk', risk=.03, risk_basis='stop', stop_pct=.20, heat_cap=.20),
}
def runf(name, start='2018-01-01', end='2026-10-13', **over):
    a = B.copy(); a.update(F[name]); a.update(over)
    return E.run(start, end, **a)
store = {}
print('===== 最终候选 =====')
for name in F:
    e, t, m = runf(name); store[name] = (e, t, m)
    print(f'{name:26s} CAGR {m["cagr"]*100:6.2f}%  DD {m["maxdd"]*100:6.2f}%  低点 {m["maxdd_low"]*100:6.2f}%  '
          f'Sharpe {m["sharpe"]:.2f}  Calmar {m["calmar"]:.2f}  交易 {m["trades"]:3d}  胜率 {m["win"]*100:4.1f}%  '
          f'PF {m["pf"]:.2f}  最差 {m["worst_trade"]*100:5.1f}%  前5 {m["top5_share"]*100:3.0f}%  暴露 {m["avg_exposure"]*100:3.0f}%  期末 {m["final"]:.0f}')
    e.to_csv(os.path.join(BASE, f'03_结果/回撤优化_权益_{name}.csv'), encoding='utf-8-sig')
    t.to_csv(os.path.join(BASE, f'03_结果/回撤优化_逐笔_{name}.csv'), index=False, encoding='utf-8-sig')

print('\n===== 训练/检验 =====')
for name in F:
    for start, end, per in [('2018-01-01','2021-12-31','训练'),('2022-01-01','2026-10-13','检验')]:
        e,t,m = runf(name,start,end)
        print(f'{name:26s} {per} CAGR {m["cagr"]*100:6.2f}%  DD {m["maxdd"]*100:6.2f}%  '
              f'胜率 {m["win"]*100:4.1f}%  PF {m["pf"]:.2f}  交易 {m["trades"]:3d}')

print('\n===== 年度 =====')
for name in F:
    e=store[name][0]; y=e.equity.resample('YE').last().pct_change()
    y.iloc[0]=e.equity.loc[e.index.year==e.index.year[0]].iloc[-1]/20000-1
    print(f'{name:26s}', ' '.join(f'{i.year}:{v*100:5.1f}%' for i,v in y.items()))

print('\n===== 回撤日期 =====')
for name in F:
    e=store[name][0]; s=e.equity; dd=s/s.cummax()-1; tr=dd.idxmin(); pk=s.loc[:tr].idxmax()
    rec=s.loc[tr:][s.loc[tr:]>=s.loc[pk]]
    print(f'{name:26s} peak {pk.date()} trough {tr.date()} {dd.min()*100:6.1f}% '
          f'recover {rec.index[0].date() if len(rec) else "未恢复"}  水下>10% {int((dd<-.10).sum())}天')

print('\n===== 成本0.35% =====')
for name in F:
    e,t,m=runf(name,cost_rate=.0035)
    print(f'{name:26s} CAGR {m["cagr"]*100:6.2f}%  DD {m["maxdd"]*100:6.2f}%')

print('\n===== 平稳块自举(2000) =====')
rng=np.random.default_rng(11)
for name in F:
    r=store[name][0].ret.values[1:]; n=len(r); nb=2000; cg=np.empty(nb); dd=np.empty(nb); bl=20
    for b in range(nb):
        ix=[]
        while len(ix)<n:
            st=rng.integers(0,n); ix.extend(((st+np.arange(bl))%n).tolist())
        x=r[np.array(ix[:n])]; q=np.cumprod(1+x)
        cg[b]=q[-1]**(252/n)-1; dd[b]=(q/np.maximum.accumulate(q)-1).min()
    print(f'{name:26s} CAGR p5/50/95 {np.round(np.percentile(cg,[5,50,95])*100,1)}  '
          f'DD p5/50/95 {np.round(np.percentile(dd,[5,50,95])*100,1)}')

print('\n===== 去前五笔 =====')
for name in F:
    e,t,m=store[name]
    g=t.groupby(['code','entry_date']).apply(lambda x:(x.weight*x.ret).sum(),include_groups=False)
    top=g.nlargest(5); skip={(c,pd.Timestamp(d)) for c,d in top.index}
    e2,t2,m2=runf(name,skip_entries=skip)
    print(f'{name:26s} 原 {m["cagr"]*100:5.1f}%/{m["maxdd"]*100:6.1f}%  去5笔 {m2["cagr"]*100:5.1f}%/{m2["maxdd"]*100:6.1f}%')

print('\n===== DSR(N=119) =====')
var_sr=0.000884; N=119; G=.5772156649015329
sr0=np.sqrt(var_sr)*((1-G)*norm.ppf(1-1/N)+G*norm.ppf(1-1/(N*np.e)))
for name in F:
    r=store[name][0].ret.values[1:]; sr=r.mean()/r.std(); T=len(r)
    sk=pd.Series(r).skew(); ku=pd.Series(r).kurtosis()+3
    dsr=norm.cdf((sr-sr0)*np.sqrt(T-1)/np.sqrt(1-sk*sr+(ku-1)/4*sr**2))
    print(f'{name:26s} 年化SR {sr*np.sqrt(252):.2f}  DSR {dsr:.3f}')

print('\n===== 三年块 =====')
for name in F:
    s=store[name][0].equity; print(name)
    for y0 in [2018,2021,2024]:
        x=s[(s.index.year>=y0)&(s.index.year<=y0+2)]
        if len(x)<100: continue
        cg=(x.iloc[-1]/x.iloc[0])**(252/len(x))-1; dd=(x/x.cummax()-1).min()
        print(f'  {y0}-{y0+2}: {cg*100:5.1f}% / {dd*100:6.1f}%')

print('\n===== 成交特征 =====')
for name in ['收益优先_45_波动20','均衡_45_波动20_分批20']:
    t=store[name][1].drop_duplicates(['code','entry_date'])
    idx=[E.code_idx[c] for c in t.code]; ii=[dates.get_loc(d) for d in t.entry_date]
    gap=[E.O[i,j]/E.C[i-1,j]-1 for i,j in zip(ii,idx)]
    price=[E.O[i,j] for i,j in zip(ii,idx)]
    print(f'{name:26s} 入场{len(t)}笔  跳空中位 {np.median(gap)*100:.2f}% 最大 {max(gap)*100:.2f}%  '
          f'价格中位 {np.median(price):.1f}  价格>90占比 {np.mean(np.array(price)>90)*100:.0f}%')

# 基准
mkt=pd.Series(E.MKT,index=dates).loc[store['旧25.8%'][0].index]
print('\n等权全A：总 %.1f%%  CAGR %.1f%%  DD %.1f%%' % (
    (mkt.iloc[-1]/mkt.iloc[0]-1)*100, ((mkt.iloc[-1]/mkt.iloc[0])**(252/len(mkt))-1)*100,
    ((mkt/mkt.cummax()-1).min())*100))
