import pandas as pd, numpy as np, itertools, json
exec(open('sim_panic.py').read().split("if __name__=='__main__':")[0])

def simulate2(ev, hold=10, slots=2, score='drawdown250', cost=0.0015, dyn=True, start='2017-01-01', end='2026-10-23'):
    sig=base[base['ev_'+ev] & (base.date>=start)&(base.date<=end)].copy()
    if sig.empty: return None,None
    asc = score in ('rsi14','kdj_j','amount20','return_250d','drawdown250','ret1')
    sig['rk']=sig.groupby('date')[score].rank(ascending=asc,method='first')
    sig=sig[sig.rk<=slots*3].sort_values(['date','rk'])
    sigday={int(d):[(int(r.s),float(r.rk)) for r in g.head(slots*3).itertuples()] for d,g in sig.groupby('d')}
    mtm=[]; pos=[]; trades=[]; cash=1.0; prev_equity=1.0
    for i,d in enumerate(dates):
        if i>0 and (i-1) in sigday:
            held={p['s'] for p in pos}
            cand=[x for x in sigday[i-1] if x[0] not in held and not np.isnan(O[i,x[0]])]
            room=slots-len(pos)
            take=cand[:room]
            if take:
                n_after=len(pos)+len(take)
                alloc_each=prev_equity/(n_after if dyn else slots)
                for s,rk in take:
                    op=O[i,s]; alloc=min(alloc_each,cash)
                    if alloc<=1e-4: break
                    inv=alloc*(1-cost)
                    pos.append(dict(s=s,entry_date=dates[i],entry_px=op,last_px=op,value=inv,days=0,thscode=syms[s],rk=rk,alloc=alloc))
                    cash-=alloc
        for p in pos:
            px=C[i,p['s']]
            if not np.isnan(px) and p['last_px']>0:
                p['value']*=px/p['last_px']; p['last_px']=px
            p['days']+=1
        keep=[]
        for p in pos:
            if p['days']>=hold:
                cash+=p['value']*(1-cost)
                p['ret']=p['last_px']/p['entry_px']*(1-cost)*(1-cost)-1
                p['exit_date']=dates[i]; trades.append(p)
            else: keep.append(p)
        pos=keep
        eq=cash+sum(p['value'] for p in pos)
        mtm.append((dates[i],eq,cash,len(pos))); prev_equity=eq
    for p in pos:
        p['ret']=p['last_px']/p['entry_px']*(1-cost)*(1-cost)-1; p['exit_date']=dates[-1]; trades.append(p)
    eqdf=pd.DataFrame(mtm,columns=['date','equity','cash','npos']); tr=pd.DataFrame(trades)
    return eqdf,tr

def detail(eq,tr,label=''):
    e=eq.set_index('date').equity; yrs=(e.index[-1]-e.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1; dd=(e/e.cummax()-1).min()
    annual=e.resample('YE').last().pct_change(); annual.iloc[0]=e.resample('YE').last().iloc[0]-1
    monthly=e.resample('ME').last().pct_change(); monthly.iloc[0]=np.nan
    out=dict(label=label,cagr=cagr,dd=dd,calmar=cagr/abs(dd),n=len(tr),win=(tr.ret>0).mean(),avg=tr.ret.mean(),
             expo=(eq.npos>0).mean(),final=e.iloc[-1],sharpe=e.pct_change().mean()/e.pct_change().std()*np.sqrt(252),
             worst_month=monthly.min(),best_year=annual.max(),worst_year=annual.min())
    return out,annual,monthly

if __name__=='__main__':
    rows=[]; details={}
    for ev in ['panic_capit','panic_vol','panic_base','panic_engulf','panic_union','panic_strong','three_down']:
        for hold in [5,8,10,12,15,20]:
            for slots in [1,2,3,4]:
                for score in ['drawdown250','rsi14','amount20','return_250d','close_location','amount_ratio1_20']:
                    for dyn in [True,False]:
                        eq,tr=simulate2(ev,hold,slots,score,dyn=dyn)
                        if eq is None: continue
                        m,annual,monthly=detail(eq,tr,f'{ev}_h{hold}_s{slots}_{score}_dyn{int(dyn)}')
                        rows.append(m)
                        details[m['label']]=(annual,monthly)
    r=pd.DataFrame(rows).sort_values('cagr',ascending=False)
    r.to_csv('results/development/panic_sim2_grid.csv',index=False)
    print(r.head(40).to_string(index=False))
    # annual for top 5
    for lab in r.head(5).label:
        a,mo=details[lab]
        print('\n==',lab); print(a.round(4).to_string())
