import pandas as pd, numpy as np, sys
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics
D=load(); base=D['base']
r=pd.read_csv('03_结果/walkforward_grid_train.csv')
rows=[]
for _,g in r.iterrows():
    m=(base.drawdown250<g.dd)&(base.return_250d>0)&(base.rsi14<g.rsi)&(base.close_location>0.45)&(base.market_return20<-0.05)
    base['ev_custom']=m.values
    out={}
    for p,(s,e) in {'val':('2022-01-01','2023-12-31'),'test':('2024-01-01','2026-10-23'),'full':('2017-01-01','2026-10-23')}.items():
        eq,tr=simulate2(D,'custom',start=s,end=e,hold=int(g.hold),slots=2,delay=int(g.delay),take_profit=float(g.tp),score='drawdown250',dyn=False)
        out[p]=metrics(eq,tr)
    row={k:g[k] for k in ['dd','rsi','hold','delay','tp']}
    for k in ['cagr','dd','n','win','expo']: row[f'train_{k}']=g[f'train_{k}']
    for p in out:
        for k,v in out[p].items(): row[f'{p}_{k}']=v
    rows.append(row)
res=pd.DataFrame(rows)
res.to_csv('03_结果/grid_all_periods.csv',index=False)
res['min3']=res[['train_cagr','val_cagr','test_cagr']].min(axis=1)
res['mean3']=res[['train_cagr','val_cagr','test_cagr']].mean(axis=1)
print('combos',len(res))
print('pass all >=30 & dd>=-0.25:',((res.min3>=0.30)&(res.full_dd>=-0.25)).sum())
print('pass all >=28 & dd>=-0.25:',((res.min3>=0.28)&(res.full_dd>=-0.25)).sum())
print('pass all >=25 & dd>=-0.25:',((res.min3>=0.25)&(res.full_dd>=-0.25)).sum())
print('\n=== best min3 ===')
print(res.sort_values('min3',ascending=False).head(20)[['dd','rsi','hold','delay','tp','train_cagr','val_cagr','test_cagr','full_cagr','full_dd','min3','full_n']].to_string(index=False))
print('\n=== by delay: max/median min3 ===')
print(res.groupby('delay').min3.agg(['max','median','count']).to_string())
print('\n=== by tp: max/median min3 ===')
print(res.groupby('tp').min3.agg(['max','median','count']).to_string())
print('\n=== by hold: max/median min3 ===')
print(res.groupby('hold').min3.agg(['max','median','count']).to_string())
print('\n=== by dd: max/median min3 ===')
print(res.groupby('dd').min3.agg(['max','median','count']).to_string())
print('\n=== d2 tp20 hold15 across dd/rsi ===')
print(res[(res.delay==2)&(res.tp==0.2)&(res.hold==15)][['dd','rsi','train_cagr','val_cagr','test_cagr','full_dd','min3']].to_string(index=False))
