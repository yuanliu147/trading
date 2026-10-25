import pandas as pd, numpy as np, sys
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics, annual
D=load(); base=D['base']
m0=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05)
base['ev_custom']=m0.values
CFG=dict(hold=15,slots=2,delay=2,take_profit=0.20,score='drawdown250',dyn=False)
PER={'train':('2017-01-01','2021-12-31'),'val':('2022-01-01','2023-12-31'),'test':('2024-01-01','2026-10-23'),'full':('2017-01-01','2026-10-23')}
def run(**over):
    kw={**CFG,**over}; out={}
    for p,(s,e) in PER.items():
        eq,tr=simulate2(D,'custom',start=s,end=e,**kw); out[p]=(eq,tr,metrics(eq,tr))
    return out
# 1 delay / tp / hold neighborhood
print('=== delay/tp/hold neighborhood (2 slots) ===')
for tp in [0.15,0.20,0.25]:
  for delay in [0,1,2,3,4]:
    o=run(take_profit=tp,delay=delay)
    print(f'tp{tp} d{delay}: train {o["train"][2]["cagr"]:.3f} val {o["val"][2]["cagr"]:.3f} test {o["test"][2]["cagr"]:.3f} full {o["full"][2]["cagr"]:.3f} dd {o["full"][2]["dd"]:.3f} n {o["full"][2]["n"]}')
# 2 cost/delay
print('\n=== cost ===')
for c in [0.0015,0.0025,0.0035,0.005]:
    o=run(cost=c); print(c,{p:round(o[p][2]['cagr'],3) for p in PER},'dd',round(o['full'][2]['dd'],3))
# 3 random selection controls
print('\n=== random selection 30 seeds ===')
cag=[]; dds=[]; ns=[]
rng=np.random.default_rng(42)
for seed in range(30):
    base['randscore']=rng.random(len(base))
    o=run(score='randscore')
    cag.append(o['full'][2]['cagr']); dds.append(o['full'][2]['dd']); ns.append(o['full'][2]['n'])
print('random cagr mean/p10/p50/p90:',np.mean(cag),np.percentile(cag,10),np.median(cag),np.percentile(cag,90))
print('random dd median/worst:',np.median(dds),min(dds),'n median',np.median(ns))
# 4 remove top N winners
o=run(); eq,tr,mm=o['full']
tr2=tr.sort_values('ret',ascending=False)
print('\n=== remove top winners ===')
for n in [1,3,5,10]:
    drop=tr2.head(n)[['entry_date','thscode']]
    mask=m0.copy()
    for _,r in drop.iterrows():
        mask &= ~((base.date==r.entry_date)&(base.thscode==r.thscode))
    base['ev_custom']=mask.values
    oo=run(); base['ev_custom']=m0.values
    print(f'drop{n}: cagr {oo["full"][2]["cagr"]:.3f} dd {oo["full"][2]["dd"]:.3f} n {oo["full"][2]["n"]}')
base['ev_custom']=m0.values
# 5 event threshold perturbation
print('\n=== event thresholds (tp20 d2 slots2) ===')
for dd in [-0.35,-0.40,-0.45,-0.50,-0.55]:
  for rsi in [30,35,40]:
    m=(base.drawdown250<dd)&(base.return_250d>0)&(base.rsi14<rsi)&(base.close_location>0.45)&(base.market_return20<-0.05)
    base['ev_custom']=m.values; o=run(); base['ev_custom']=m0.values
    print(f'dd{dd} rsi{rsi}: train {o["train"][2]["cagr"]:.3f} val {o["val"][2]["cagr"]:.3f} test {o["test"][2]["cagr"]:.3f} full {o["full"][2]["cagr"]:.3f} dd {o["full"][2]["dd"]:.3f} n {o["full"][2]["n"]}')
# 6 detail best
o=run(); eq,tr,mm=o['full']; e=eq.set_index('date').equity
print('\n=== best detail ===',{k:round(v,4) for k,v in mm.items()})
print('annual:',annual(eq).round(3).to_dict())
print('exit reasons:',tr.exit_reason.value_counts().to_dict())
print('avg hold days',tr.days.mean(),'avg ret',tr.ret.mean(),'median',tr.ret.median())
print('\ntop 12 trades:')
print(tr.sort_values('ret',ascending=False)[['thscode','entry_date','exit_date','ret','days','exit_reason','rk']].head(12).to_string(index=False))
print('\nworst 8:')
print(tr.sort_values('ret')[['thscode','entry_date','exit_date','ret','days','exit_reason','rk']].head(8).to_string(index=False))
peak=e.cummax(); ddser=e/peak-1; trough=ddser.idxmin(); peakdate=e.loc[:trough].idxmax()
print('\nmax dd',ddser.min(),'peak',peakdate,'trough',trough)
print('2024 trades:')
t24=tr[pd.to_datetime(tr.entry_date).dt.year==2024]
print(t24[['thscode','entry_date','exit_date','ret','days','exit_reason']].to_string(index=False))
tr.to_csv('03_结果/candidate_trades.csv',index=False)
eq.to_csv('03_结果/candidate_equity.csv',index=False)
