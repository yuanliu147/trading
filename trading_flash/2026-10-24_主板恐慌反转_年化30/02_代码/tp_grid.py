import pandas as pd, numpy as np, sys, itertools
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics, annual
D=load(); base=D['base']
PERIODS={'train':('2017-01-01','2021-12-31'),'val':('2022-01-01','2023-12-31'),'test':('2024-01-01','2026-10-23'),'full':('2017-01-01','2026-10-23')}
m0=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05)
base['ev_custom']=m0.values
rows=[]
for tp in [0.10,0.15,0.20,0.25,0.30,None]:
  for hold in [10,12,15,18,20]:
    for slots in [1,2]:
      for delay in [0,1,2]:
        for stop in [None,0.10,0.12]:
          out={}
          for pname,(s,e) in PERIODS.items():
            eq,tr=simulate2(D,'custom',hold=hold,slots=slots,score='drawdown250',dyn=False,delay=delay,stop_loss=stop,take_profit=tp,start=s,end=e)
            out[pname]=metrics(eq,tr)
          if not out['full']: continue
          row=dict(tp=tp,hold=hold,slots=slots,delay=delay,stop=stop)
          for p in PERIODS:
            for k,v in out[p].items(): row[f'{p}_{k}']=v
          rows.append(row)
r=pd.DataFrame(rows)
r['min_cagr']=r[['train_cagr','val_cagr','test_cagr']].min(axis=1)
r.to_csv('03_结果/tp_grid.csv',index=False)
cols=['tp','hold','slots','delay','stop','min_cagr','train_cagr','val_cagr','test_cagr','full_cagr','full_dd','full_n','full_win','full_expo','test_dd','test_n']
print(r.sort_values('min_cagr',ascending=False)[cols].head(30).to_string(index=False))
