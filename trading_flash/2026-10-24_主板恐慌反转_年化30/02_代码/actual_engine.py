# DEPRECATED 2026-10-25: 本脚本有送转/成本基准 bug，不要重跑覆盖旧结果。
# 修正版见 actual_engine_fixed.py；旧版 42.6%/236.2万不可用。
import pandas as pd, numpy as np, sys
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics, annual
D=load(); base=D['base']; df=D['df']
m=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05)
base['ev_custom']=m.values
CFG=dict(hold=15,slots=2,delay=2,take_profit=0.20,score='drawdown250',dyn=False,cost=0.00175)
eqh,trh=simulate2(D,'custom',start='2017-01-01',end='2026-10-23',**CFG)
# build actual matrices
dates=D['dates']; syms=D['syms']; di={d:i for i,d in enumerate(dates)}; si={s:i for i,s in enumerate(syms)}
AO=np.full((len(dates),len(syms)),np.nan); AH=np.full_like(AO,np.nan); AL=np.full_like(AO,np.nan); AC=np.full_like(AO,np.nan); HC=np.full_like(AO,np.nan)
for c,arr in [('open',AO),('high',AH),('low',AL),('close',AC),('hfq_close',HC)]:
    arr[df.d.values,df.s.values]=df[c].values
# replay trade decisions with actual prices, 100-share lots, costs
CAP=100000.0; cash=CAP; last={}; pos=[]; trades=[]; eqrows=[]
trh=trh.sort_values(['entry_date','thscode']).reset_index(drop=True)
entries={}
for _,t in trh.iterrows():
    entries.setdefault(pd.Timestamp(t.entry_date),[]).append(t)
exits={}
for _,t in trh.iterrows():
    exits.setdefault(pd.Timestamp(t.exit_date),[]).append(t)
# process day by day from first entry to last exit
start=min(entries); end=max(exits)
for d in dates:
    d=pd.Timestamp(d)
    if d<start or d>end: continue
    i=di[d]
    # entries at open
    for t in entries.get(d,[]):
        s=si[t.thscode]; px=AO[i,s]
        if np.isnan(px) or px<=0: continue
        equity=cash+sum(p['shares']*last.get(p['s'],p['entry_px']) for p in pos)
        alloc=min(equity/2,cash*0.999)
        shares=int(alloc/px/100)*100
        if shares<=0: continue
        buy=shares*px; cb=max(5,buy*0.00025); slip=buy*0.001
        cash-=buy+cb+slip
        pos.append(dict(s=s,thscode=t.thscode,shares=shares,entry_date=d,entry_px=px,entry_cost=cb+slip,signal_date=t.signal_date,factor=HC[i,s]/px if px>0 else 1.0))
        last[s]=px
    # mark to market; adjust shares for bonus/split (hfq/actual factor jump >5%)
    for p in pos:
        px=AC[i,p['s']]
        if not np.isnan(px): last[p['s']]=px
        hc=HC[i,p['s']]
        if not np.isnan(px) and px>0 and not np.isnan(hc):
            f=hc/px
            if p['factor']>0 and f/p['factor']>1.05:
                p['shares']=p['shares']*f/p['factor']; p['factor']=f
    # exits at target/time (use actual high for target)
    for t in exits.get(d,[]):
        p=next((x for x in pos if x['thscode']==t.thscode and x['entry_date']==pd.Timestamp(t.entry_date)),None)
        if p is None: continue
        s=p['s']; ah=AH[i,s]; al=AL[i,s]; aop=AO[i,s]; ac=AC[i,s]
        if t.exit_reason=='target' and not np.isnan(ah):
            target=p['entry_px']*1.20
            if ah>=target:
                fill=max(aop,target) if not np.isnan(aop) else target
            else:
                fill=ac
        else:
            fill=ac
        if np.isnan(fill): fill=last[s]
        sell=p['shares']*fill; cs=max(5,sell*0.00025)+sell*0.0005; slip=sell*0.001
        cash+=sell-cs-slip
        pnl=sell-p['shares']*p['entry_px']-p['entry_cost']-cs-slip
        trades.append(dict(thscode=p['thscode'],signal_date=p['signal_date'],entry_date=p['entry_date'],exit_date=d,
                           entry_px=p['entry_px'],exit_px=fill,shares=p['shares'],pnl=pnl,
                           ret=pnl/(p['shares']*p['entry_px']+p['entry_cost']),exit_reason=t.exit_reason,
                           hfq_ret=t.ret))
        pos.remove(p)
    eq=cash+sum(p['shares']*last.get(p['s'],p['entry_px']) for p in pos)
    eqrows.append(dict(date=d,equity=eq,cash=cash,npos=len(pos)))
eqa=pd.DataFrame(eqrows); tra=pd.DataFrame(trades)
eqa.to_csv('03_结果/实际价_每日权益.csv',index=False); tra.to_csv('03_结果/实际价_逐笔.csv',index=False)
def mets(e,tr):
    e=e.set_index('date').equity; yrs=(e.index[-1]-e.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1; dd=(e/e.cummax()-1).min(); r=e.pct_change().dropna()
    return dict(cagr=cagr,dd=dd,calmar=cagr/abs(dd),sharpe=r.mean()/r.std()*np.sqrt(252),n=len(tr),win=(tr.ret>0).mean(),avg=tr.ret.mean(),final=e.iloc[-1],expo=(eqa.npos>0).mean())
print('ACTUAL metrics',{k:round(v,4) for k,v in mets(eqa,tra).items()})
print('annual'); print(annual(eqa).round(4).to_string())
print('trades',len(tra),'sum pnl',round(tra.pnl.sum(),2))
print('exit reason',tra.exit_reason.value_counts().to_dict())
print('avg ret',tra.ret.mean(),'win',(tra.ret>0).mean())
print('top5 share',tra[tra.ret>0].ret.nlargest(5).sum()/tra[tra.ret>0].ret.sum())
print(tra.sort_values('ret',ascending=False)[['thscode','entry_date','exit_date','entry_px','exit_px','shares','ret','exit_reason']].head(8).to_string(index=False))
