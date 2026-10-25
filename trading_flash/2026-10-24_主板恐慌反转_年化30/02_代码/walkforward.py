import pandas as pd, numpy as np, sys, itertools
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics
D=load(); base=D['base']
grid=[]
for dd in [-0.35,-0.40,-0.45,-0.50]:
  for rsi in [30,35,40]:
    for hold in [10,12,15,18]:
      for delay in [0,1,2,3]:
        for tp in [0.15,0.20,0.25]:
          grid.append(dict(dd=dd,rsi=rsi,hold=hold,delay=delay,tp=tp))
print('combos',len(grid))
def runone(g,start,end):
    m=(base.drawdown250<g['dd'])&(base.return_250d>0)&(base.rsi14<g['rsi'])&(base.close_location>0.45)&(base.market_return20<-0.05)
    base['ev_custom']=m.values
    eq,tr=simulate2(D,'custom',start=start,end=end,hold=int(g['hold']),slots=2,delay=int(g['delay']),take_profit=float(g['tp']),score='drawdown250',dyn=False)
    return metrics(eq,tr)
rows=[]
for g in grid:
    tr=runone(g,'2017-01-01','2021-12-31')
    rows.append({**g,**{f'train_{k}':v for k,v in tr.items()}})
r=pd.DataFrame(rows)
r.to_csv('03_结果/walkforward_grid_train.csv',index=False)
# select on train: require n>=15, dd>=-0.28; rank by CAGR then Calmar
cand=r[(r.train_n>=15)&(r.train_dd>=-0.28)].sort_values(['train_cagr','train_calmar'],ascending=False)
print('\n=== top train-selected ===')
print(cand.head(10)[['dd','rsi','hold','delay','tp','train_cagr','train_dd','train_n','train_win']].to_string(index=False))
best=cand.iloc[0].to_dict()
val=runone(best,'2022-01-01','2023-12-31')
print('\ntrain-selected applied to val:',{k:round(v,3) if isinstance(v,(int,float,np.floating)) else v for k,v in val.items()})
# select on train+val
rows2=[]
for _,g in r.iterrows():
    gg={k:g[k] for k in ['dd','rsi','hold','delay','tp']}
    ev=runone(gg,'2017-01-01','2023-12-31')
    rows2.append({**gg,**{f'tv_{k}':v for k,v in ev.items()}})
r2=pd.DataFrame(rows2)
r2.to_csv('03_结果/walkforward_grid_trainval.csv',index=False)
cand2=r2[(r2.tv_n>=25)&(r2.tv_dd>=-0.30)].sort_values(['tv_cagr','tv_calmar'],ascending=False)
print('\n=== top train+val-selected ===')
print(cand2.head(10)[['dd','rsi','hold','delay','tp','tv_cagr','tv_dd','tv_n','tv_win']].to_string(index=False))
best2=cand2.iloc[0].to_dict()
test=runone(best2,'2024-01-01','2026-10-23')
print('\ntrain+val-selected applied to test:',{k:round(v,3) if isinstance(v,(int,float,np.floating)) else v for k,v in test.items()})
print('selected params:',{k:best2[k] for k in ['dd','rsi','hold','delay','tp']})
