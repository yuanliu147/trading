import pandas as pd, numpy as np, sys
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics, annual
D=load(); base=D['base']
m=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05)
base['ev_custom']=m.values
PER={'train':('2017-01-01','2021-12-31'),'val':('2022-01-01','2023-12-31'),'test':('2024-01-01','2026-10-23'),'full':('2017-01-01','2026-10-23')}
CFG=dict(hold=15,slots=2,delay=2,take_profit=0.20,score='drawdown250',dyn=False)
def run(**kw):
    out={}
    for p,(s,e) in PER.items():
        eq,tr=simulate2(D,'custom',start=s,end=e,**{**CFG,**kw}); out[p]=metrics(eq,tr)
    return out
print('=== candidate period ===')
o=run(); print({p:{k:round(v,3) for k,v in o[p].items()} for p in o})
print('\n=== cliffs (full) ===')
for name,vals,key in [('delay',[0,1,2,3,4],'delay'),('tp',[0.15,0.2,0.25],'take_profit'),('rsi',[30,35,40],None),('dd',[-0.35,-0.4,-0.45,-0.5,-0.55],None)]:
    for v in vals:
        if key: r=run(**{key:v})['full']
        elif name=='rsi':
            mm=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<v)&(base.close_location>0.45)&(base.market_return20<-0.05); base['ev_custom']=mm.values; r=run()['full']; base['ev_custom']=m.values
        else:
            mm=(base.drawdown250<v)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05); base['ev_custom']=mm.values; r=run()['full']; base['ev_custom']=m.values
        print(f'{name}{v}: cagr {r.get("cagr",0):.3f} dd {r.get("dd",0):.3f} n {r.get("n",0)}')
print('\n=== remove top winners ===')
eq,tr=simulate2(D,'custom',start='2017-01-01',end='2026-10-23',**CFG); tr2=tr.sort_values('ret',ascending=False)
for n in [1,3,5,10]:
    mask=m.copy()
    for _,r in tr2.head(n).iterrows(): mask &= ~((base.date==r.signal_date)&(base.thscode==r.thscode))
    base['ev_custom']=mask.values; rr=run()['full']; base['ev_custom']=m.values
    print(f'drop{n}: cagr {rr["cagr"]:.3f} dd {rr["dd"]:.3f} n {rr["n"]}')
print('\n=== random 20 seeds ===')
rng=np.random.default_rng(7); cs=[]; ds=[]
for seed in range(20):
    base['rand']=rng.random(len(base)); rr=run(score='rand')['full']; cs.append(rr['cagr']); ds.append(rr['dd'])
print('mean',np.mean(cs),'median',np.median(cs),'p10',np.percentile(cs,10),'p90',np.percentile(cs,90),'dd med',np.median(ds))
