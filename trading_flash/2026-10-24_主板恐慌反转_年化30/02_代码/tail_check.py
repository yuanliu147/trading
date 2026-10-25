import pandas as pd, numpy as np, sys
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics
D=load(); base=D['base']
m=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05)
base['ev_custom']=m.values
print('=== catastrophic stop (tp20, d2, 2 slots) ===')
for stop in [None,0.08,0.10,0.12,0.15,0.18,0.20]:
    out={}
    for p,(s,e) in {'train':('2017-01-01','2021-12-31'),'val':('2022-01-01','2023-12-31'),'test':('2024-01-01','2026-10-23'),'full':('2017-01-01','2026-10-23')}.items():
        eq,tr=simulate2(D,'custom',start=s,end=e,hold=15,slots=2,delay=2,take_profit=0.20,stop_loss=stop,score='drawdown250',dyn=False)
        out[p]=metrics(eq,tr)
    print(f'stop{stop}: train {out["train"].get("cagr",0):.3f} val {out["val"].get("cagr",0):.3f} test {out["test"].get("cagr",0):.3f} full {out["full"].get("cagr",0):.3f} dd {out["full"].get("dd",0):.3f} n {out["full"].get("n",0)}')
# exit-day limit-down check on actual trades
tr=pd.read_csv('03_结果/实际价_逐笔.csv',parse_dates=['entry_date','exit_date'])
df=D['df']
flag=[]
for _,t in tr.iterrows():
    row=df[(df.thscode==t.thscode)&(df.date==t.exit_date)]
    if row.empty: continue
    prev=df[(df.thscode==t.thscode)&(df.date<t.exit_date)].sort_values('date').tail(1)
    if prev.empty: continue
    chg=row.close.iloc[0]/prev.close.iloc[0]-1
    if chg<-0.095: flag.append((t.thscode,t.exit_date,chg,t.exit_reason))
print('\nexit-day <= -9.5%:',len(flag),flag[:10])
# next-day return after exit for those
