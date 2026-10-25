import pandas as pd, numpy as np, os, json, itertools
pd.set_option('display.width',240); pd.set_option('display.max_columns',80)
f='01_数据/research_panel.parquet'
cols=['thscode','date','open','close','hfq_open','hfq_close','next_actual_open','next_up_limit_10','next_down_limit_10',
 'ret1','ret_o5','ret_o10','ret_o20','mfe10','mae10','amount20','amount_ratio1_20','amount_ratio5_20',
 'return_5d','return_10d','return_20d','return_60d','return_120d','return_250d',
 'atr20_pct','volatility20','drawdown60','drawdown250','price_position60','close_location','rsi14','kdj_j',
 'up_days20','near_limit_up_days20','limit_up_10','broken_limit_up_10','up_limit_05',
 'bullish_engulfing','hammer_shape','range20_pct','week_return12','week_drawdown12','month_close_ma6',
 'market_return','market_return5','market_return20','breadth20','breadth60','sentiment_score','limit_up10_share','advance_share']
df=pd.read_parquet(f,columns=cols)
df['date']=pd.to_datetime(df.date)
df=df.sort_values(['date','thscode']).reset_index(drop=True)
df['tradable']=df.next_actual_open.notna() & (df.next_actual_open<df.next_up_limit_10*0.998) & (df.next_actual_open>df.next_down_limit_10*1.002)
df['liquid']=df.amount20>=10_000_000
df['st_proxy']=df.up_limit_05==1
# daily matrices
dates=np.sort(df.date.unique()); di={d:i for i,d in enumerate(dates)}
syms=np.sort(df.thscode.unique()); si={s:i for i,s in enumerate(syms)}
# map each row
df['d']=df.date.map(di); df['s']=df.thscode.map(si)
O=np.full((len(dates),len(syms)),np.nan,dtype=np.float32)
C=np.full_like(O,np.nan); A=np.full_like(O,np.nan); OC=np.full_like(O,np.nan)
O[df.d.values,df.s.values]=df.hfq_open.values
C[df.d.values,df.s.values]=df.hfq_close.values
A[df.d.values,df.s.values]=df.next_actual_open.values
OC[df.d.values,df.s.values]=df.close_location.values
# note: C is hfq close, actual close not needed
base=df[df.tradable & df.liquid & ~df.st_proxy].copy()
print('base rows',len(base),'dates',len(dates))
# event definitions
E={}
E['panic_capit']=(base.drawdown250<-0.60)&(base.rsi14<25)&(base.close_location>0.30)&(base.market_return20<-0.08)
E['panic_vol']=(base.drawdown250<-0.50)&(base.rsi14<30)&(base.close_location>0.40)&(base.market_return20<-0.05)&(base.amount_ratio1_20>=1.3)
E['panic_base']=(base.drawdown250<-0.50)&(base.rsi14<30)&(base.close_location>0.40)&(base.market_return20<-0.05)
E['panic_engulf']=(base.drawdown250<-0.50)&(base.rsi14<30)&(base.close_location>0.40)&(base.market_return20<-0.05)&(base.bullish_engulfing==1)
E['panic_union']=((base.drawdown250<-0.60)&(base.rsi14<25)&(base.close_location>0.30)&(base.market_return20<-0.08))|((base.drawdown250<-0.50)&(base.rsi14<30)&(base.close_location>0.40)&(base.market_return20<-0.05)&(base.amount_ratio1_20>=1.3))
E['panic_strong']=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05)
E['three_down']=(base.up_days20<=3)&(base.return_5d<-0.12)&(base.close_location>0.60)&(base.market_return20<-0.03)
for name,mask in E.items():
    base['ev_'+name]=mask.values
    print(name, mask.sum())

def simulate(ev, hold=10, slots=2, score='rsi14', cost=0.0015, topk=True, start='2017-01-01', end='2026-10-23'):
    sig=base[base['ev_'+ev] & (base.date>=start)&(base.date<=end)].copy()
    if sig.empty: return None,None
    asc = score in ('rsi14','kdj_j','amount20','return_250d','drawdown250','ret1')
    sig['rk']=sig.groupby('date')[score].rank(ascending=asc,method='first')
    if topk: sig=sig[sig.rk<=slots*3]
    sig=sig.sort_values(['date','rk'])
    sigday={}
    for d,g in sig.groupby('d'):
        sigday[int(d)]=[(int(r.s),float(r.rk),float(r.next_actual_open)) for r in g.head(slots*3).itertuples()]
    mtm=[]; pos=[]; trades=[]
    cash=1.0; prev_equity=1.0
    for i,d in enumerate(dates):
        # entries at open from previous close signals
        if i>0 and (i-1) in sigday:
            held={p['s'] for p in pos}
            for s,rk,act in sigday[i-1]:
                if len(pos)>=slots: break
                if s in held: continue
                op=O[i,s]
                if np.isnan(op): continue
                alloc=prev_equity/slots
                alloc=min(alloc,cash)
                if alloc<=1e-4: continue
                inv=alloc*(1-cost)
                pos.append(dict(s=s,entry_date=dates[i],entry_px=op,last_px=op,value=inv,days=0,thscode=syms[s],rk=rk))
                cash-=alloc
        # mark to market at close
        for p in pos:
            px=C[i,p['s']]
            if not np.isnan(px) and p['last_px']>0:
                p['value']*=px/p['last_px']; p['last_px']=px
            p['days']+=1
        # exits at close
        keep=[]
        for p in pos:
            if p['days']>=hold:
                cash+=p['value']*(1-cost)
                p['ret']=p['value']*(1-cost)/(p['entry_px']*0+1)  # placeholder
                p['exit_date']=dates[i]
                # actual return on invested capital: value at exit net sell / (alloc*(1-cost))
                # recover alloc from entry: value path already includes buy cost; approximate via entry record
                p['ret']=p['value']*(1-cost)/ (p['value']/ (p['last_px']/p['entry_px'])) -1 if p['last_px']>0 else np.nan
                # simpler: net return from entry price
                p['ret']=p['last_px']/p['entry_px']*(1-cost)*(1-cost)-1
                trades.append(p)
            else: keep.append(p)
        pos=keep
        eq=cash+sum(p['value'] for p in pos)
        mtm.append((dates[i],eq,cash,len(pos)))
        prev_equity=eq
    if pos:
        for p in pos:
            p['ret']=p['last_px']/p['entry_px']*(1-cost)*(1-cost)-1
            p['exit_date']=dates[-1]; trades.append(p)
    eqdf=pd.DataFrame(mtm,columns=['date','equity','cash','npos'])
    tr=pd.DataFrame(trades)
    return eqdf,tr

def metrics(eq,tr):
    if eq is None or len(eq)<10: return {}
    e=eq.set_index('date').equity
    yrs=(e.index[-1]-e.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1
    dd=(e/e.cummax()-1).min()
    r=e.pct_change().dropna()
    sharpe=r.mean()/r.std()*np.sqrt(252) if r.std()>0 else np.nan
    return dict(cagr=cagr,dd=dd,sharpe=sharpe,calmar=cagr/abs(dd) if dd<0 else np.nan,
                n=len(tr),win=(tr.ret>0).mean() if len(tr) else np.nan,avg=tr.ret.mean() if len(tr) else np.nan,
                expo=(eq.npos>0).mean(),final=e.iloc[-1])
if __name__=='__main__':
    rows=[]
    for ev in E:
        for hold in [5,10,15,20]:
            for slots in [2,3]:
                for score in ['rsi14','drawdown250','amount20','return_250d','close_location','amount_ratio1_20']:
                    eq,tr=simulate(ev,hold,slots,score)
                    m=metrics(eq,tr)
                    if m: rows.append(dict(ev=ev,hold=hold,slots=slots,score=score,**m))
    r=pd.DataFrame(rows).sort_values('cagr',ascending=False)
    r.to_csv('results/development/panic_sim_grid.csv',index=False)
    print(r.head(35).to_string(index=False))
