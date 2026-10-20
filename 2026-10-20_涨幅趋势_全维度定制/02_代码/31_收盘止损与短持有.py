# -*- coding: utf-8 -*-
"""收盘确认的结构止损：避免盘中假突破；比较10/15/20日持有。"""
import pickle,numpy as np,pandas as pd,os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; nD=len(dates)
O=P['open'].values; H=P['high'].values; L=P['low'].values; C=P['close'].values
ATR=P['atr_pct'].values; F20=P['fwd20'].values; AMT=P['amt20'].values
U=P['up_struct'].values; AGE=P['trend_age'].values; DD60=P['dd60'].values
LU=P['lu_count20'].values; SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values
MB=P['market_breadth20'].values; NG=P['next_gap'].values; NLU=P['next_limit_up'].values
OWD=P['one_word_down'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U&np.isfinite(F20)&(AMT>=3e7)&(C>=2)&(AGE>=5)&date_ok)
tr=valid&(period[:,None]==0); atr_thr=np.nanquantile(ATR[tr],0.40)
sig=valid&(AGE<=30)&(ATR<=atr_thr)&(LU==0)&(SBR>=0.6)&(SR20>0.05)&(DD60>-0.08)&(MB[:,None]>0.5)&~NLU&(NG<0.05)
L5=pd.DataFrame(L,index=dates,columns=codes).rolling(5,min_periods=3).min().values
ii,jj=np.where(sig); rows=[]
for i,j in zip(ii,jj):
    e=i+1
    if e>=nD: continue
    entry=O[e,j]
    if not np.isfinite(entry) or entry<=0: continue
    sl=L5[i,j]
    if not np.isfinite(sl): continue
    stop=sl*0.995; sd=1-stop/entry
    if not np.isfinite(sd) or sd<=0 or sd>0.12: continue
    rec={'period':period[i],'sd':sd}
    for h in [10,15,20]:
        end=min(e+h-1,nD-1); hit=None
        for t in range(e+1,end+1):
            if np.isfinite(C[t,j]) and C[t,j]<stop: hit=t; break
        if hit is not None and not OWD[min(hit+1,nD-1),j]:
            px=O[min(hit+1,nD-1),j] if np.isfinite(O[min(hit+1,nD-1),j]) else C[hit,j]
            rec[f'r{h}_closestop']=px/entry-1
        else:
            rec[f'r{h}_closestop']=C[end,j]/entry-1 if np.isfinite(C[end,j]) else np.nan
        rec[f'r{h}_fixed']=C[end,j]/entry-1 if np.isfinite(C[end,j]) else np.nan
    rows.append(rec)
R=pd.DataFrame(rows); R['期']=R.period.map({0:'训练',1:'检验'})
R['距离组']=pd.cut(R.sd,[0,0.03,0.05,0.07,0.10,0.12],labels=['0-3%','3-5%','5-7%','7-10%','10-12%'])
rows=[]
for (per,g) in R.groupby('期'):
    for lab,x in g.groupby('距离组',observed=True):
        if len(x)<100: continue
        for h in [10,15,20]:
            for mode in ['fixed','closestop']:
                y=x[f'r{h}_{mode}'].dropna()-0.003
                rows.append(dict(期=per,距离组=lab,持有=h,方式=mode,样本=len(y),均值=y.mean(),中位=np.median(y),胜率=(y>0).mean(),盈亏比=y[y>0].sum()/max(1e-9,-y[y<0].sum())))
out=pd.DataFrame(rows); out.to_csv(os.path.join(BASE,'03_结果/收盘止损_短持有.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',260); pd.set_option('display.max_rows',200)
print(out.round(4).to_string(index=False))
