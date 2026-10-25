import pandas as pd, numpy as np, sys, json
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics, annual
D=load(); base=D['base']
m=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05)
base['ev_custom']=m.values
CFG=dict(hold=15,slots=2,delay=2,take_profit=0.20,score='drawdown250',dyn=False,cost=0.00175)
eq,tr=simulate2(D,'custom',start='2017-01-01',end='2026-10-23',**CFG)
eq.to_csv('03_结果/最终_每日权益.csv',index=False)
tr.to_csv('03_结果/最终_逐笔.csv',index=False)
mm=metrics(eq,tr); print('metrics',{k:round(v,4) for k,v in mm.items()})
print('annual'); print(annual(eq).round(4).to_string())
# lot-aware audit at 100k, 2 slots 50%
cap=100000.0
tr2=tr.copy(); rows=[]
for _,t in tr2.iterrows():
    # actual entry open
    d0=pd.Timestamp(t.signal_date); d1=pd.Timestamp(t.entry_date); d2=pd.Timestamp(t.exit_date)
    row=D['df'][(D['df'].thscode==t.thscode)&(D['df'].date==d1)]
    if row.empty: continue
    aopen=float(row.next_actual_open.iloc[0]) if not np.isnan(row.next_actual_open.iloc[0]) else np.nan
    # actual prices on exit date
    rex=D['df'][(D['df'].thscode==t.thscode)&(D['df'].date==d2)]
    if rex.empty or np.isnan(aopen): continue
    ac=float(rex.close.iloc[0]); ah=float(rex.high.iloc[0]); al=float(rex.low.iloc[0])
    shares=int((cap/2)/aopen/100)*100
    if shares<=0: continue
    if t.exit_reason=='target':
        target=aopen*1.20
        fill=max(float(rex.open.iloc[0]),target) if ah>=target else ac
    else:
        fill=ac
    buy=aopen*shares; sell=fill*shares
    comm_b=max(5,buy*0.00025); comm_s=max(5,sell*0.00025)+sell*0.0005
    slip=(buy+sell)*0.001
    pnl=sell-buy-comm_b-comm_s-slip
    rows.append(dict(thscode=t.thscode,signal_date=d0,entry_date=d1,exit_date=d2,entry_actual=aopen,exit_actual=fill,
                     shares=shares,buy=buy,sell=sell,costs=comm_b+comm_s+slip,pnl=pnl,ret=pnl/(buy+comm_b+slip),
                     hfq_ret=t.ret,exit_reason=t.exit_reason))
la=pd.DataFrame(rows)
la.to_csv('03_结果/最终_逐笔_实际价整手.csv',index=False)
print('\nlot-aware trades',len(la),'sum pnl',la.pnl.sum(),'avg ret',la.ret.mean(),'win',(la.ret>0).mean())
print('hfq avg ret',tr.ret.mean(),'hfq final',eq.equity.iloc[-1])
print('lot-aware return on 100k',la.pnl.sum()/100000)
# yearly lot-aware pnl (not compounded, uses fixed 50k per trade)
la['year']=pd.to_datetime(la.entry_date).dt.year
print(la.groupby('year').agg(n=('pnl','size'),pnl=('pnl','sum'),ret=('ret','mean'),win=('ret',lambda x:(x>0).mean())).to_string())
print('\nexit reason',tr.exit_reason.value_counts().to_dict())
