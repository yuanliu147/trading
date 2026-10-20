# -*- coding: utf-8 -*-
"""候选上涨趋势系统的组合回测。4仓、风险预算、次日开盘、20日固定持有。"""
import pickle, numpy as np, pandas as pd, os, time, itertools
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; nD=len(dates)
O=P['open'].values; H=P['high'].values; L=P['low'].values; C=P['close'].values
ATR=P['atr_pct'].values; AMT=P['amt20'].values; F20=P['fwd20'].values
U=P['up_struct'].values; AGE=P['trend_age'].values; DD60=P['dd60'].values
LU=P['lu_count20'].values; SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values; DMA20=P['dist_ma20'].values
MKT=P['regime_id'].values.reshape(-1,1); MB=P['market_breadth20'].values.reshape(-1,1)
NG=P['next_gap'].values; NLU=P['next_limit_up'].values
OWU=P['one_word_up'].values; OWD=P['one_word_down'].values
MKT_IDX=P['market_index'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)

def run(start,end,gate='regime',rank='amt',slots=4,risk=0.0075,maxpos=0.20,stop_mode='atr2',hold=20,gap_filter=True,cost=0.0015,gate_thr=0.5,verbose=False,
        sbr_thr=0.60,sr_thr=0.05,age_max=30,atr_q=0.40,dd_thr=-0.08,lu_filter=True,amt_floor=3e7,dd_pause=None,dd_resume=0.03,first_week_stop=None):
    valid=(U & (AMT>=amt_floor) & (C>=2) & (AGE>=5) & date_ok)
    tr=valid & (period[:,None]==0)
    atr_thr=0.044  # 固定阈值，避免用训练期分位造成早期样本的前视
    sig=(valid & (AGE<=age_max) & (ATR<=atr_thr) & (SBR>=sbr_thr) & (SR20>sr_thr) & (DD60>dd_thr))
    if lu_filter: sig=sig & (LU==0)
    if gate=='regime': sig=sig & (MKT>=4)
    elif gate=='breadth': sig=sig & (MB>gate_thr)
    elif gate=='none': pass
    si=dates.searchsorted(pd.Timestamp(start)); ei=dates.searchsorted(pd.Timestamp(end),side='right')-1
    cash=1.0; positions=[]; equity_curve=[]; trades=[]; holdings_rows=[]; peak=1.0; allow_entries=True
    for t in range(si,ei+1):
        # 1) 开盘/盘中止损
        for pos in list(positions):
            j=pos['j']
            if t<pos['entry_i']: continue
            if first_week_stop is not None and t==pos['entry_i']+4 and np.isfinite(C[t,j]) and C[t,j]/pos['entry']-1<=first_week_stop and not OWD[t,j]:
                fill=C[t,j]; cash+=pos['shares']*fill*(1-cost)
                trades.append(dict(code=codes[j],entry_date=dates[pos['entry_i']],exit_date=dates[t],
                    entry=pos['entry'],exit=fill,ret=fill/pos['entry']-1,reason='首周止损',hold=t-pos['entry_i']+1,
                    shares=pos['shares'],pnl=pos['shares']*(fill-pos['entry'])-pos['shares']*(pos['entry']+fill)*cost,
                    weight=pos['weight'],entry_eq=pos['entry_eq'],entry_amt=pos['entry_amt']))
                positions.remove(pos); continue
            if np.isfinite(L[t,j]) and np.isfinite(pos['stop']) and L[t,j]<=pos['stop']:
                if OWD[t,j]: continue
                fill=min(O[t,j],pos['stop']) if np.isfinite(O[t,j]) else pos['stop']
                cash += pos['shares']*fill*(1-cost)
                trades.append(dict(code=codes[j],entry_date=dates[pos['entry_i']],exit_date=dates[t],
                    entry=pos['entry'],exit=fill,ret=fill/pos['entry']-1,reason='止损',hold=t-pos['entry_i']+1,
                    shares=pos['shares'],pnl=pos['shares']*(fill-pos['entry'])-pos['shares']*(pos['entry']+fill)*cost,
                    weight=pos['weight'],entry_eq=pos['entry_eq'],entry_amt=pos['entry_amt']))
                positions.remove(pos)
        # 2) 到期收盘卖出（放在开盘买入之后，保守）
        # 3) 开盘买入
        if t>si and allow_entries:
            cand=np.where(sig[t-1])[0]
            cand=[j for j in cand if not any(p['j']==j for p in positions)]
            if cand:
                if rank=='amt': key=AMT[t-1,cand]
                elif rank=='sbr': key=-SBR[t-1,cand]
                elif rank=='score':
                    key=-(SBR[t-1,cand]+SR20[t-1,cand]-DMA20[t-1,cand]-0.3*(AMT[t-1,cand]/1e8))
                else: key=np.zeros(len(cand))
                order=np.argsort(key)
                for k in order:
                    if len(positions)>=slots: break
                    j=cand[k]
                    if not np.isfinite(O[t,j]) or O[t,j]<=0: continue
                    if gap_filter and np.isfinite(NG[t-1,j]) and NG[t-1,j]>0.05: continue
                    entry=O[t,j]; atr=ATR[t-1,j]
                    if not np.isfinite(atr): continue
                    if stop_mode=='atr2': sd=float(np.clip(2*atr,0.04,0.12))
                    elif stop_mode=='atr15': sd=float(np.clip(1.5*atr,0.03,0.10))
                    elif stop_mode=='fixed8': sd=0.08
                    elif stop_mode=='fixed12': sd=0.12
                    elif stop_mode=='fixed15': sd=0.15
                    elif stop_mode=='cat15': sd=0.15
                    elif stop_mode=='cat20': sd=0.20
                    elif stop_mode=='cat25': sd=0.25
                    elif stop_mode=='none': sd=0.99
                    else: sd=float(np.clip(2*atr,0.04,0.12))
                    sd_size=float(np.clip(2*atr,0.04,0.12)) if stop_mode in ('none','cat15','cat20','cat25') else sd
                    stop=entry*(1-sd) if stop_mode!='none' else -np.inf
                    risk_ps=entry*sd_size
                    eq=cash+sum(p['shares']*C[t-1,p['j']] if np.isfinite(C[t-1,p['j']]) else p['shares']*p['last'] for p in positions)
                    risk_amt=eq*risk
                    heat=sum(p['shares']*p['risk_ps'] for p in positions)/max(eq,1e-9)
                    if heat+risk>0.0301: continue
                    sh=min(risk_amt/(entry*sd_size), eq*maxpos/entry, cash/(entry*(1+cost)))
                    if sh*entry<eq*0.01: continue
                    cost_amt=sh*entry*cost
                    cash-=sh*entry+cost_amt
                    positions.append(dict(j=j,entry_i=t,entry=entry,shares=sh,stop=stop,risk_ps=risk_ps,last=entry,exit_i=t+hold-1,
                        weight=sh*entry/eq,entry_eq=eq,entry_amt=AMT[t-1,j]))
        # 4) 到期收盘卖出
        for pos in list(positions):
            j=pos['j']
            if t>=pos['exit_i']:
                if OWD[t,j]: continue
                if not np.isfinite(C[t,j]):
                    # 停牌：不按最后价假装卖出；有未来数据就顺延，数据终止才按50%折价结算
                    future=np.where(np.isfinite(C[t:,j]))[0]
                    if len(future)>0: continue
                    fill=pos['last']*0.5; reason='数据终止'
                else:
                    fill=C[t,j]; reason='到期'
                cash += pos['shares']*fill*(1-cost)
                trades.append(dict(code=codes[j],entry_date=dates[pos['entry_i']],exit_date=dates[t],
                    entry=pos['entry'],exit=fill,ret=fill/pos['entry']-1,reason=reason,hold=t-pos['entry_i']+1,
                    shares=pos['shares'],pnl=pos['shares']*(fill-pos['entry'])-pos['shares']*(pos['entry']+fill)*cost,
                    weight=pos['weight'],entry_eq=pos['entry_eq'],entry_amt=pos['entry_amt']))
                positions.remove(pos)
        # 5) 估值
        val=0; val_low=0
        for pos in positions:
            j=pos['j']; px=C[t,j] if np.isfinite(C[t,j]) else pos['last']
            lpx=L[t,j] if np.isfinite(L[t,j]) else px
            pos['last']=px; val+=pos['shares']*px; val_low+=pos['shares']*lpx
            holdings_rows.append(dict(date=dates[t],code=codes[j],shares=pos['shares'],price=px,
                value=pos['shares']*px,weight=pos['shares']*px/(cash+val+1e-9),entry=pos['entry'],
                ret=px/pos['entry']-1,hold=t-pos['entry_i']+1))
        eq_now=cash+val; eq_low=cash+val_low; peak=max(peak,eq_now)
        if dd_pause is not None:
            if (eq_now/peak-1)<=-dd_pause: allow_entries=False
            elif (eq_now/peak-1)>=-dd_resume: allow_entries=True
        equity_curve.append((dates[t],eq_now,cash,len(positions),eq_low))
    E=pd.DataFrame(equity_curve,columns=['date','equity','cash','npos','equity_low']).set_index('date')
    E['ret']=E.equity.pct_change().fillna(0)
    T=pd.DataFrame(trades)
    # 指标
    days=len(E); total=E.equity.iloc[-1]/E.equity.iloc[0]-1
    cagr=(1+total)**(252/max(days,1))-1
    dd=(E.equity/E.equity.cummax()-1).min()
    dd_low=(E.equity_low/E.equity.cummax()-1).min()
    sharpe=E.ret.mean()/ (E.ret.std()+1e-9)*np.sqrt(252)
    if len(T):
        win=(T.ret>0).mean(); pf=T.pnl[T.pnl>0].sum()/max(1e-9,-T.pnl[T.pnl<0].sum())
        avg_hold=T.hold.mean(); expo=E.npos.mean()/slots
    else: win=pf=avg_hold=expo=np.nan
    bench=MKT_IDX[si:ei+1]; btotal=bench[-1]/bench[0]-1; bdd=(bench/np.maximum.accumulate(bench)-1).min()
    bret=pd.Series(bench).pct_change().fillna(0).values
    bsharpe=bret.mean()/(bret.std()+1e-9)*np.sqrt(252)
    return E,T,pd.DataFrame(holdings_rows),dict(start=str(dates[si].date()),end=str(dates[ei].date()),days=days,total=total,cagr=cagr,maxdd=dd,sharpe=sharpe,
        trades=len(T),win=win,pf=pf,avg_hold=avg_hold,exposure=expo,bench_total=btotal,bench_cagr=(1+btotal)**(252/days)-1,
        bench_dd=bdd,bench_sharpe=bsharpe,final=E.equity.iloc[-1],maxdd_low=dd_low)

if __name__=='__main__':
    scen=[]
    base=dict(gate='breadth',rank='amt',slots=6,risk=0.005,maxpos=0.20,hold=20)
    scen.append(dict(name='主方案_cat20',stop_mode='cat20',**base))
    for st in ['none','fixed8','fixed12','fixed15']:
        scen.append(dict(name=f'止损_{st}',stop_mode=st,**base))
    scen.append(dict(name='四仓_cat20',stop_mode='cat20',slots=4,risk=0.0075,maxpos=0.20,hold=20,gate='breadth',rank='amt'))
    for c in [0.0025,0.004]:
        scen.append(dict(name=f'成本{int(c*10000)}',stop_mode='cat20',cost=c,**base))
    for af in [5e7,8e7,1e8]:
        scen.append(dict(name=f'流动性{int(af/1e7)}千万',stop_mode='cat20',amt_floor=af,**base))
    rows=[]
    for sc in scen:
        name=sc.pop('name')
        for start,end,per in [('2018-01-01','2021-12-31','训练'),('2022-01-01','2026-10-13','检验'),('2018-01-01','2026-10-13','全期')]:
            E,T,H,m=run(start,end,**sc)
            m.update(dict(场景=name,期=per))
            rows.append(m)
            if name=='主方案_cat20' and per=='检验':
                E.to_csv(os.path.join(BASE,'03_结果/权益_含退市_检验.csv'),encoding='utf-8-sig')
                T.to_csv(os.path.join(BASE,'03_结果/逐笔_含退市_检验.csv'),index=False,encoding='utf-8-sig')
                H.to_csv(os.path.join(BASE,'03_结果/每日持仓_含退市_检验.csv'),index=False,encoding='utf-8-sig')
            if name=='主方案_cat20' and per=='训练':
                E.to_csv(os.path.join(BASE,'03_结果/权益_含退市_训练.csv'),encoding='utf-8-sig')
                T.to_csv(os.path.join(BASE,'03_结果/逐笔_含退市_训练.csv'),index=False,encoding='utf-8-sig')
                H.to_csv(os.path.join(BASE,'03_结果/每日持仓_含退市_训练.csv'),index=False,encoding='utf-8-sig')
        print(name,'done',flush=True)
    out=pd.DataFrame(rows)
    cols=['场景','期','days','total','cagr','maxdd','maxdd_low','sharpe','trades','win','pf','avg_hold','exposure','bench_total','bench_dd','bench_sharpe']
    out=out[cols]
    out.to_csv(os.path.join(BASE,'03_结果/组合回测_含退市.csv'),index=False,encoding='utf-8-sig')
    pd.set_option('display.width',300); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',100)
    print(out.to_string(index=False))
