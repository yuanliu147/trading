import pandas as pd, numpy as np
PANEL='01_数据/research_panel.parquet'
COLS=['thscode','date','open','close','high','low','hfq_open','hfq_close','hfq_high','hfq_low','next_actual_open','next_up_limit_10','next_down_limit_10',
 'ret1','ret_o5','ret_o10','ret_o20','mfe10','mae10','amount20','amount_ratio1_20','amount_ratio5_20',
 'return_5d','return_10d','return_20d','return_60d','return_120d','return_250d',
 'atr20_pct','volatility20','drawdown60','drawdown250','price_position60','close_location','rsi14','kdj_j',
 'up_days20','near_limit_up_days20','limit_up_10','broken_limit_up_10','up_limit_05',
 'bullish_engulfing','hammer_shape','range20_pct','week_return12','week_drawdown12','month_close_ma6',
 'market_return','market_return5','market_return20','breadth20','breadth60','sentiment_score','limit_up10_share','advance_share']

def load():
    df=pd.read_parquet(PANEL,columns=COLS)
    df['date']=pd.to_datetime(df.date)
    df=df.sort_values(['date','thscode']).reset_index(drop=True)
    df['tradable']=df.next_actual_open.notna() & (df.next_actual_open<df.next_up_limit_10*0.998) & (df.next_actual_open>df.next_down_limit_10*1.002)
    df['liquid']=df.amount20>=10_000_000
    df['st_proxy']=df.up_limit_05==1
    dates=np.sort(df.date.unique()); syms=np.sort(df.thscode.unique())
    di={d:i for i,d in enumerate(dates)}; si={s:i for i,s in enumerate(syms)}
    df['d']=df.date.map(di); df['s']=df.thscode.map(si)
    O=np.full((len(dates),len(syms)),np.nan,dtype=np.float32)
    C=np.full_like(O,np.nan); A=np.full_like(O,np.nan); H=np.full_like(O,np.nan); L=np.full_like(O,np.nan); AC=np.full_like(O,np.nan); AO=np.full_like(O,np.nan)
    O[df.d.values,df.s.values]=df.hfq_open.values
    C[df.d.values,df.s.values]=df.hfq_close.values
    H[df.d.values,df.s.values]=df.hfq_high.values
    L[df.d.values,df.s.values]=df.hfq_low.values
    AC[df.d.values,df.s.values]=df.close.values
    AO[df.d.values,df.s.values]=df.open.values
    A[df.d.values,df.s.values]=df.next_actual_open.values
    base=df[df.liquid & ~df.st_proxy].copy()
    E={}
    E['panic_capit']=(base.drawdown250<-0.60)&(base.rsi14<25)&(base.close_location>0.30)&(base.market_return20<-0.08)
    E['panic_vol']=(base.drawdown250<-0.50)&(base.rsi14<30)&(base.close_location>0.40)&(base.market_return20<-0.05)&(base.amount_ratio1_20>=1.3)
    E['panic_base']=(base.drawdown250<-0.50)&(base.rsi14<30)&(base.close_location>0.40)&(base.market_return20<-0.05)
    E['panic_engulf']=(base.drawdown250<-0.50)&(base.rsi14<30)&(base.close_location>0.40)&(base.market_return20<-0.05)&(base.bullish_engulfing==1)
    E['panic_union']=E['panic_capit']|E['panic_vol']
    E['panic_strong']=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05)
    E['three_down']=(base.up_days20<=3)&(base.return_5d<-0.12)&(base.close_location>0.60)&(base.market_return20<-0.03)
    for k,v in E.items(): base['ev_'+k]=v.values
    # forward confirmation features (known before delayed entry)
    base=base.sort_values(['thscode','date'])
    base['close_t1']=base.groupby('thscode')['hfq_close'].shift(-1)
    base['close_t2']=base.groupby('thscode')['hfq_close'].shift(-2)
    base['close_t3']=base.groupby('thscode')['hfq_close'].shift(-3)
    mkt=base.groupby('date').agg(mret=('market_return','first'),b20=('breadth20','first')).sort_index()
    mkt['mkt_fwd2']=(1+mkt.mret.shift(-1))*(1+mkt.mret.shift(-2))-1
    mkt['b20_t2']=mkt.b20.shift(-2)
    base=base.merge(mkt[['mkt_fwd2','b20_t2']],left_on='date',right_index=True,how='left')
    base['conf_close2']=base.close_t2>base.hfq_close
    base['conf_breadth2']=base.b20_t2>base.breadth20
    base['conf_mkt2']=base.mkt_fwd2>-0.01
    return dict(df=df,dates=dates,syms=syms,O=O,C=C,A=A,H=H,L=L,AC=AC,AO=AO,base=base,E=E)

def simulate2(D, ev, hold=15, slots=1, score='drawdown250', cost=0.0015, dyn=False,
              delay=0, stop_loss=None, take_profit=None, stop_at='close', start='2017-01-01', end='2026-10-23', max_signals_per_day=None):
    base=D["base"]; dates=D["dates"]; syms=D["syms"]; O=D["O"]; C=D["C"]; A=D["A"]; H=D["H"]; L=D["L"]; AC=D["AC"]; AO=D["AO"]
    sig=base[base['ev_'+ev] & (base.date>=start)&(base.date<=end)].copy()
    if sig.empty: return None,None
    asc = score in ('rsi14','kdj_j','amount20','return_250d','drawdown250','ret1')
    sig['rk']=sig.groupby('date')[score].rank(ascending=asc,method='first')
    if max_signals_per_day: sig=sig[sig.rk<=max_signals_per_day]
    sig=sig.sort_values(['date','rk'])
    sigday={int(d):[(int(r.s),float(r.rk)) for r in g.head(slots*5).itertuples()] for d,g in sig.groupby('d')}
    mtm=[]; pos=[]; trades=[]; cash=1.0; prev_equity=1.0
    i0=int(np.searchsorted(dates,np.datetime64(start)))
    i1=int(np.searchsorted(dates,np.datetime64(end),side='right'))-1
    for i in range(i0,i1+1):
        d=dates[i]
        # entries at open, from signals delay days earlier
        j=i-1-delay
        if j>=i0 and j in sigday:
            held={p['s'] for p in pos}
            cand=[x for x in sigday[j] if x[0] not in held and not np.isnan(O[i,x[0]])]
            take=cand[:max(0,slots-len(pos))]
            if take:
                n_after=len(pos)+len(take)
                alloc_each=prev_equity/(n_after if dyn else slots)
                for s,rk in take:
                    op=O[i,s]; alloc=min(alloc_each,cash)
                    if alloc<=1e-4: break
                    if i>0 and not np.isnan(AC[i-1,s]) and not np.isnan(AO[i,s]):
                        if AO[i,s] >= AC[i-1,s]*1.098: continue
                    inv=alloc*(1-cost)
                    pos.append(dict(s=s,entry_date=dates[i],signal_date=dates[j],entry_i=i,entry_px=op,last_px=op,value=inv,days=0,thscode=syms[s],rk=rk,alloc=alloc,high=op))
                    cash-=alloc
        # mark to market; exits with optional intraday triggers
        exit_now=[]
        for p in pos:
            hi=H[i,p['s']]; lo=L[i,p['s']]; op=O[i,p['s']]
            px=C[i,p['s']]
            if not np.isnan(px) and p['last_px']>0:
                p['value']*=px/p['last_px']; p['last_px']=px; p['high']=max(p['high'],px)
            p['days']+=1
            reason=None; fill=None
            if stop_loss is not None and not np.isnan(lo) and i>p['entry_i']:
                trig=p['entry_px']*(1-stop_loss)
                if lo<=trig:
                    reason='stop'; fill=min(op,trig) if not np.isnan(op) else trig
            if reason is None and take_profit is not None and not np.isnan(hi) and i>p['entry_i']:
                trig=p['entry_px']*(1+take_profit)
                if hi>=trig:
                    reason='target'; fill=max(op,trig) if not np.isnan(op) else trig
            if reason is None and p['days']>=hold:
                reason='time'; fill=px
            if reason:
                p['exit_fill']=fill; p['exit_reason']=reason; exit_now.append(p)
        keep=[]
        for p in pos:
            if p in exit_now:
                fill=p['exit_fill']
                if np.isnan(fill): fill=p['last_px']
                # realized gross return from entry to fill, then costs
                realized=p['value']*(fill/p['last_px'])
                cash+=realized*(1-cost)
                p['ret']=fill/p['entry_px']*(1-cost)*(1-cost)-1
                p['exit_date']=dates[i]; p['last_px']=fill
                trades.append(p)
            else: keep.append(p)
        pos=keep
        eq=cash+sum(p['value'] for p in pos)
        mtm.append((dates[i],eq,cash,len(pos))); prev_equity=eq
    for p in pos:
        p['ret']=p['last_px']/p['entry_px']*(1-cost)*(1-cost)-1; p['exit_date']=dates[-1]; p['exit_reason']='eod'; trades.append(p)
    eqdf=pd.DataFrame(mtm,columns=['date','equity','cash','npos']); tr=pd.DataFrame(trades)
    return eqdf,tr

def metrics(eq,tr):
    if eq is None or len(eq)<10 or tr is None or len(tr)==0: return {}
    e=eq.set_index('date').equity; yrs=(e.index[-1]-e.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1; dd=(e/e.cummax()-1).min(); r=e.pct_change().dropna()
    return dict(cagr=cagr,dd=dd,calmar=cagr/abs(dd) if dd<0 else np.nan,sharpe=r.mean()/r.std()*np.sqrt(252) if r.std()>0 else np.nan,
                n=len(tr),win=(tr.ret>0).mean(),avg=tr.ret.mean(),expo=(eq.npos>0).mean(),final=e.iloc[-1])

def annual(eq):
    e=eq.set_index('date').equity
    a=e.resample('YE').last().pct_change(); a.iloc[0]=e.resample('YE').last().iloc[0]-1
    return a
