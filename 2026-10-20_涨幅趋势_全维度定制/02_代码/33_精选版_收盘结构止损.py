# -*- coding: utf-8 -*-
"""精选版：只做结构止损5%-7%的信号，收盘跌破结构低点次日开盘卖。"""
import pickle,numpy as np,pandas as pd,os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; nD=len(dates)
O=P['open'].values; H=P['high'].values; L=P['low'].values; C=P['close'].values
ATR=P['atr_pct'].values; F20=P['fwd20'].values; AMT=P['amt20'].values
U=P['up_struct'].values; AGE=P['trend_age'].values; DD60=P['dd60'].values
LU=P['lu_count20'].values; SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values
MB=P['market_breadth20'].values; NG=P['next_gap'].values; NLU=P['next_limit_up'].values
OWD=P['one_word_down'].values; OWU=P['one_word_up'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]; yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U&(AMT>=3e7)&(C>=2)&(AGE>=5)&date_ok)
tr=valid&(period[:,None]==0); atr_thr=0.044
base=valid&(AGE<=30)&(ATR<=atr_thr)&(LU==0)&(SBR>=0.6)&(SR20>0.05)&(DD60>-0.08)&(MB[:,None]>0.5)
L5=pd.DataFrame(L,index=dates,columns=codes).rolling(5,min_periods=3).min().values

def run(start,end,sd_min=0.05,sd_max=0.07):
    si=dates.searchsorted(pd.Timestamp(start)); ei=dates.searchsorted(pd.Timestamp(end),side='right')-1
    cash=1.0; pos=[]; eqc=[]; trades=[]
    for t in range(si,ei+1):
        # 收盘确认止损：前一日收盘跌破，今日开盘卖
        for p in list(pos):
            j=p['j']
            if p.get('pending') and t>p['entry_i']:
                if OWD[t,j]: continue
                fill=O[t,j] if np.isfinite(O[t,j]) else p['last']
                cash+=p['sh']*fill*(1-0.0015)
                trades.append(dict(code=codes[j],entry_date=dates[p['entry_i']],exit_date=dates[t],entry=p['entry'],exit=fill,
                    ret=fill/p['entry']-1,reason='收盘结构止损',hold=t-p['entry_i']+1,shares=p['sh'],
                    pnl=p['sh']*(fill-p['entry'])-p['sh']*(p['entry']+fill)*0.0015,weight=p['w']))
                pos.remove(p); continue
        # 收盘后标记
        for p in pos:
            j=p['j']
            if t>p['entry_i'] and np.isfinite(C[t,j]) and C[t,j]<p['stop']:
                p['pending']=True
        # 20日到期（在买入前后均可）
        for p in list(pos):
            j=p['j']
            if t>=p['exit_i']:
                if OWD[t,j]: continue
                if not np.isfinite(C[t,j]):
                    future=np.where(np.isfinite(C[t:,j]))[0]
                    if len(future)>0: continue
                    fill=p['last']*0.5; reason='数据终止'
                else:
                    fill=C[t,j]; reason='到期'
                cash+=p['sh']*fill*(1-0.0015)
                trades.append(dict(code=codes[j],entry_date=dates[p['entry_i']],exit_date=dates[t],entry=p['entry'],exit=fill,
                    ret=fill/p['entry']-1,reason=reason,hold=t-p['entry_i']+1,shares=p['sh'],
                    pnl=p['sh']*(fill-p['entry'])-p['sh']*(p['entry']+fill)*0.0015,weight=p['w']))
                pos.remove(p)
        # 买入
        if t>si:
            cand=np.where(base[t-1])[0]; cand=[j for j in cand if not any(p['j']==j for p in pos)]
            if cand:
                # 先按结构止损距离过滤，再按成交额排序
                ok=[]
                for j in cand:
                    if not np.isfinite(O[t,j]) or O[t,j]<=0: continue
                    if np.isfinite(NG[t-1,j]) and NG[t-1,j]>0.05: continue
                    sl=L5[t-1,j]
                    if not np.isfinite(sl): continue
                    sd=1-(sl*0.995)/O[t,j]
                    if sd_min<=sd<=sd_max: ok.append((j,sd,sl))
                ok=sorted(ok,key=lambda x:AMT[t-1,x[0]])
                for j,sd,sl in ok:
                    if len(pos)>=6: break
                    entry=O[t,j]
                    eq=cash+sum(p['sh']*(C[t-1,p['j']] if np.isfinite(C[t-1,p['j']]) else p['last']) for p in pos)
                    heat=sum(p['sh']*p['risk'] for p in pos)/max(eq,1e-9)
                    if heat+0.005>0.0301: continue
                    sh=min(eq*0.005/(entry*sd), eq*0.20/entry, cash/(entry*1.0015))
                    if sh*entry<eq*0.01: continue
                    cash-=sh*entry*(1+0.0015)
                    pos.append(dict(j=j,entry_i=t,entry=entry,sh=sh,stop=sl*0.995,risk=0.005,last=entry,exit_i=t+19,w=sh*entry/eq,pending=False))
        val=0; val_low=0
        for p in pos:
            j=p['j']; px=C[t,j] if np.isfinite(C[t,j]) else p['last']; lpx=L[t,j] if np.isfinite(L[t,j]) else px
            p['last']=px; val+=p['sh']*px; val_low+=p['sh']*lpx
        eqc.append((dates[t],cash+val,cash,len(pos),cash+val_low))
    E=pd.DataFrame(eqc,columns=['date','equity','cash','npos','equity_low']).set_index('date'); E['ret']=E.equity.pct_change().fillna(0)
    T=pd.DataFrame(trades)
    total=E.equity.iloc[-1]/E.equity.iloc[0]-1; cagr=(1+total)**(252/len(E))-1; dd=(E.equity/E.equity.cummax()-1).min(); ddl=(E.equity_low/E.equity.cummax()-1).min()
    win=(T.ret>0).mean() if len(T) else np.nan; pf=T.pnl[T.pnl>0].sum()/max(1e-9,-T.pnl[T.pnl<0].sum()) if len(T) else np.nan
    return E,T,dict(start=str(dates[si].date()),end=str(dates[ei].date()),total=total,cagr=cagr,maxdd=dd,sharpe=E.ret.mean()/(E.ret.std()+1e-9)*np.sqrt(252),
        trades=len(T),win=win,pf=pf,avg_hold=T.hold.mean() if len(T) else np.nan,exposure=E.npos.mean()/6,maxdd_low=ddl)
rows=[]
for start,end,per in [('2018-01-01','2021-12-31','训练'),('2022-01-01','2026-10-13','检验'),('2018-01-01','2026-10-13','全期')]:
    E,T,m=run(start,end); m['期']=per; rows.append(m)
    if per=='检验': E.to_csv(os.path.join(BASE,'03_结果/精选版_权益_检验.csv'),encoding='utf-8-sig'); T.to_csv(os.path.join(BASE,'03_结果/精选版_逐笔_检验.csv'),index=False,encoding='utf-8-sig')
    if per=='训练': E.to_csv(os.path.join(BASE,'03_结果/精选版_权益_训练.csv'),encoding='utf-8-sig'); T.to_csv(os.path.join(BASE,'03_结果/精选版_逐笔_训练.csv'),index=False,encoding='utf-8-sig')
out=pd.DataFrame(rows)[['期','start','end','total','cagr','maxdd','sharpe','trades','win','pf','avg_hold','exposure']]
out.to_csv(os.path.join(BASE,'03_结果/精选版_组合.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',250); print(out.round(4).to_string(index=False))
