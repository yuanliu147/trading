# -*- coding: utf-8 -*-
"""2万元、1-2只持仓、整手交易、最低佣金5元的严谨组合回测。"""
import pickle,numpy as np,pandas as pd,os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; nD=len(dates)
O=P['open'].values; H=P['high'].values; L=P['low'].values; C=P['close'].values
ATR=P['atr_pct'].values; AMT=P['amt20'].values
U=P['up_struct'].values; AGE=P['trend_age'].values; DD60=P['dd60'].values
LU=P['lu_count20'].values; SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values
MB=P['market_breadth20'].values; NG=P['next_gap'].values; OWD=P['one_word_down'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]; yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U&(AMT>=3e7)&(C>=2)&(AGE>=5)&date_ok)
base_nogate=valid&(AGE<=30)&(ATR<=0.044)&(LU==0)&(SBR>=0.6)&(SR20>0.05)&(DD60>-0.08)

def run(start,end,slots=2,sizing='risk',risk=0.02,maxpos=0.5,heat_cap=0.05,hold=20,rank='amt',cost_rate=0.0015,min_cost=5.0,start_cash=20000.0,gate_thr=0.5,risk_basis='atr',stop_pct=0.20):
    si=dates.searchsorted(pd.Timestamp(start)); ei=dates.searchsorted(pd.Timestamp(end),side='right')-1
    sig=base_nogate&(MB[:,None]>gate_thr)
    cash=start_cash; pos=[]; eqc=[]; trades=[]
    for t in range(si,ei+1):
        # 止损（灾难-20%）
        for p in list(pos):
            j=p['j']
            if t<p['entry_i']: continue
            if np.isfinite(L[t,j]) and L[t,j]<=p['stop']:
                if OWD[t,j]: continue
                fill=min(O[t,j],p['stop']) if np.isfinite(O[t,j]) else p['stop']
                gross=p['sh']*fill; fee=max(min_cost,gross*cost_rate); cash+=gross-fee
                trades.append(dict(code=codes[j],entry_date=dates[p['entry_i']],exit_date=dates[t],entry=p['entry'],exit=fill,
                    ret=fill/p['entry']-1,reason='止损',hold=t-p['entry_i']+1,shares=p['sh'],pnl=(fill-p['entry'])*p['sh']-fee-p['entry_fee'],weight=p['w']))
                pos.remove(p)
        # 到期
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
                gross=p['sh']*fill; fee=max(min_cost,gross*cost_rate); cash+=gross-fee
                trades.append(dict(code=codes[j],entry_date=dates[p['entry_i']],exit_date=dates[t],entry=p['entry'],exit=fill,
                    ret=fill/p['entry']-1,reason=reason,hold=t-p['entry_i']+1,shares=p['sh'],pnl=(fill-p['entry'])*p['sh']-fee-p['entry_fee'],weight=p['w']))
                pos.remove(p)
        # 买入
        if t>si:
            cand=np.where(sig[t-1])[0]
            cand=[j for j in cand if not any(p['j']==j for p in pos)]
            if cand:
                if rank=='amt': cand=sorted(cand,key=lambda j:AMT[t-1,j])
                elif rank=='sbr': cand=sorted(cand,key=lambda j:-SBR[t-1,j])
                for j in cand:
                    if len(pos)>=slots: break
                    if not np.isfinite(O[t,j]) or O[t,j]<=0: continue
                    if np.isfinite(NG[t-1,j]) and NG[t-1,j]>0.05: continue
                    entry=O[t,j]
                    eq=cash+sum(p['sh']*(C[t-1,p['j']] if np.isfinite(C[t-1,p['j']]) else p['last']) for p in pos)
                    if entry*100>eq*maxpos or entry*100>cash: continue
                    sd=stop_pct if risk_basis=='stop' else float(np.clip(2*ATR[t-1,j],0.04,0.12))
                    if sizing=='risk':
                        if risk_basis=='stop':
                            target=int((eq*risk/(entry*stop_pct))//100)*100
                        else:
                            target=int((eq*risk/(entry*sd))//100)*100
                        # 单票上限：风险预算不能突破 maxpos
                        target=min(target,int((eq*maxpos/entry)//100)*100)
                    else:
                        target=int((eq*maxpos/entry)//100)*100
                    if target<100:
                        if risk_basis=='stop':
                            # 严格风险口径：一手就超过风险预算的股票直接跳过
                            target=0
                        else:
                            target=100 if entry*100<=eq*maxpos and entry*100<=cash else 0
                    if target<=0: continue
                    # 热预算
                    heat=sum(p['sh']*p['entry']*p['sd'] for p in pos)/max(eq,1e-9)
                    sh=target
                    while sh>=100 and heat+sh*entry*sd/max(eq,1e-9)>heat_cap:
                        sh-=100
                    if sh<100: continue
                    gross=sh*entry; fee=max(min_cost,gross*cost_rate)
                    if gross+fee>cash: 
                        sh=int((cash/(entry*(1+cost_rate)))//100)*100
                        if sh<100: continue
                        gross=sh*entry; fee=max(min_cost,gross*cost_rate)
                        if gross+fee>cash: continue
                    cash-=gross+fee
                    stop=entry*(1-stop_pct)
                    pos.append(dict(j=j,entry_i=t,entry=entry,sh=sh,sd=sd,stop=stop,last=entry,exit_i=t+hold-1,w=gross/eq,entry_fee=fee))
        val=0; val_low=0
        for p in pos:
            j=p['j']; px=C[t,j] if np.isfinite(C[t,j]) else p['last']; lpx=L[t,j] if np.isfinite(L[t,j]) else px
            p['last']=px; val+=p['sh']*px; val_low+=p['sh']*lpx
        eqc.append((dates[t],cash+val,cash,len(pos),cash+val_low))
    E=pd.DataFrame(eqc,columns=['date','equity','cash','npos','equity_low']).set_index('date'); E['ret']=E.equity.pct_change().fillna(0)
    T=pd.DataFrame(trades)
    total=E.equity.iloc[-1]/start_cash-1; cagr=(1+total)**(252/len(E))-1; dd=(E.equity/E.equity.cummax()-1).min(); ddl=(E.equity_low/E.equity.cummax()-1).min()
    win=(T.ret>0).mean() if len(T) else np.nan; pf=T.pnl[T.pnl>0].sum()/max(1e-9,-T.pnl[T.pnl<0].sum()) if len(T) else np.nan
    return E,T,dict(total=total,cagr=cagr,maxdd=dd,maxdd_low=ddl,sharpe=E.ret.mean()/(E.ret.std()+1e-9)*np.sqrt(252),
        trades=len(T),win=win,pf=pf,avg_hold=T.hold.mean() if len(T) else np.nan,avg_pos=E.npos.mean(),final=E.equity.iloc[-1],
        avg_weight=T.weight.mean() if len(T) else np.nan)

if __name__=='__main__':
    scen=[]
    for gate in [0.50,0.55,0.60]:
        for rank in ['amt','sbr']:
            for risk in [0.02,0.03,0.04]:
                for heat in [0.05,0.08]:
                    scen.append(dict(name=f'2只_风险{risk}_热{heat}_宽度{gate}_{rank}',slots=2,sizing='risk',risk=risk,maxpos=0.5,heat_cap=heat,gate_thr=gate,rank=rank))
            scen.append(dict(name=f'2只_固定50_宽度{gate}_{rank}',slots=2,sizing='fixed',risk=0,maxpos=0.5,heat_cap=0.10,gate_thr=gate,rank=rank))
    for hold in [15,25]:
        scen.append(dict(name=f'2只_风险3_热8_宽度55_持有{hold}',slots=2,sizing='risk',risk=0.03,maxpos=0.5,heat_cap=0.08,gate_thr=0.55,rank='amt',hold=hold))
    rows=[]
    for sc in scen:
        name=sc.pop('name')
        for start,end,per in [('2018-01-01','2021-12-31','训练'),('2022-01-01','2026-10-13','检验'),('2018-01-01','2026-10-13','全期')]:
            E,T,m=run(start,end,**sc); m.update({'场景':name,'期':per}); rows.append(m)
    out=pd.DataFrame(rows)
    out.to_csv(os.path.join(BASE,'03_结果/两万组合_扫描.csv'),index=False,encoding='utf-8-sig')
    pd.set_option('display.width',300); pd.set_option('display.max_rows',200)
    pv=out.pivot_table(index='场景',columns='期',values=['total','cagr','maxdd','win','pf','avg_pos'])
    print(pv.round(4).to_string())
