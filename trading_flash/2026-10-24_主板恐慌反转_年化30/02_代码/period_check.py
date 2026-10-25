import pandas as pd, numpy as np, sys
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics, annual
D=load()
m=(D['base'].drawdown250<-0.45)&(D['base'].return_250d>0)&(D['base'].rsi14<35)&(D['base'].close_location>0.45)&(D['base'].market_return20<-0.05)
D['base']['ev_custom']=m.values
for pname,(s,e) in {'train':('2017-01-01','2021-12-31'),'val':('2022-01-01','2023-12-31'),'test':('2024-01-01','2026-10-23'),'full':('2017-01-01','2026-10-23')}.items():
    eq,tr=simulate2(D,'custom',hold=15,slots=1,score='drawdown250',dyn=False,start=s,end=e)
    mm=metrics(eq,tr)
    print(pname,{k:(round(v,4) if isinstance(v,(int,float,np.floating)) else v) for k,v in mm.items()})
    print('  annual',annual(eq).round(4).to_dict())
