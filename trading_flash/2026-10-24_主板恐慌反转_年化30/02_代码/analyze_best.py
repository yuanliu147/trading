import pandas as pd, numpy as np, sys
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics, annual
D=load(); base=D['base']
m0=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05)
base['ev_custom']=m0.values
configs={
 'A_2slot_tp15_d2':dict(hold=15,slots=2,delay=2,take_profit=0.15),
 'B_1slot_tp20_d1':dict(hold=15,slots=1,delay=1,take_profit=0.20),
 'C_2slot_tp20_d2':dict(hold=15,slots=2,delay=2,take_profit=0.20),
 'D_2slot_tp25_d2':dict(hold=15,slots=2,delay=2,take_profit=0.25),
 'E_2slot_tp15_d0':dict(hold=15,slots=2,delay=0,take_profit=0.15),
}
for name,kw in configs.items():
    print('\n===',name,kw)
    for pname,(s,e) in {'train':('2017-01-01','2021-12-31'),'val':('2022-01-01','2023-12-31'),'test':('2024-01-01','2026-10-23'),'full':('2017-01-01','2026-10-23')}.items():
        eq,tr=simulate2(D,'custom',score='drawdown250',dyn=False,start=s,end=e,**kw)
        mm=metrics(eq,tr)
        print(f"  {pname}: cagr={mm.get('cagr',np.nan):.3f} dd={mm.get('dd',np.nan):.3f} n={mm.get('n',0)} win={mm.get('win',np.nan):.3f} expo={mm.get('expo',np.nan):.3f}")
    eq,tr=simulate2(D,'custom',score='drawdown250',dyn=False,start='2017-01-01',end='2026-10-23',**kw)
    tr['year']=pd.to_datetime(tr.entry_date).dt.year
    pos=tr[tr.ret>0].ret.sort_values(ascending=False)
    print('  conc top1/5/10:',round(pos.head(1).sum()/pos.sum(),3),round(pos.head(5).sum()/pos.sum(),3),round(pos.head(10).sum()/pos.sum(),3))
    print('  yearly:'); print(annual(eq).round(3).to_string())
    print('  trades by year:',tr.groupby('year').size().to_dict())
