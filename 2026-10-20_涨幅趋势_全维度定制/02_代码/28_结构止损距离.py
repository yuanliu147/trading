# -*- coding: utf-8 -*-
"""当前信号按“结构低点距离”分组：距离近的样本是否胜率更高、止损后期望如何。"""
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
# 结构低点：过去5日最低价
LOW5=pd.DataFrame(L,index=dates,columns=codes).rolling(5,min_periods=3).min().values
ii,jj=np.where(sig)
rows=[]
for i,j in zip(ii,jj):
    e=i+1
    if e>=nD: continue
    entry=O[e,j]
    if not np.isfinite(entry) or entry<=0: continue
    sl=LOW5[i,j]
    if not np.isfinite(sl): continue
    stop=sl*0.995; sd=1-stop/entry
    if not np.isfinite(sd) or sd<=0 or sd>0.20: continue
    end=min(e+19,nD-1)
    r20=C[end,j]/entry-1 if np.isfinite(C[end,j]) else np.nan
    # 结构止损
    hit=None
    for t in range(e,end+1):
        if np.isfinite(L[t,j]) and L[t,j]<=stop:
            hit=t; break
    if hit is not None:
        if OWD[hit,j]: r_stop=np.nan
        else: r_stop=(min(O[hit,j],stop) if np.isfinite(O[hit,j]) else stop)/entry-1
    else: r_stop=r20
    rows.append((dates[i],j,period[i],sd,r20,r_stop,np.nanmin(L[e:end+1,j]/entry-1),np.nanmax(H[e:end+1,j]/entry-1)))
R=pd.DataFrame(rows,columns=['date','j','period','sd','r20','r_stop','mae','mfe'])
R['期']=R.period.map({0:'训练 2017-2021',1:'检验 2022-2026'})
R['止损距离组']=pd.cut(R.sd,[0,0.03,0.05,0.07,0.10,0.20],labels=['0-3%','3-5%','5-7%','7-10%','10-20%'])
agg=R.groupby(['期','止损距离组'],observed=True).agg(样本=('r20','size'),平均止损距离=('sd','mean'),
    固定20日均值=('r20',lambda s:np.nanmean(s)-0.003),固定20日胜率=('r20',lambda s:np.nanmean(s>0)),
    结构止损均值=('r_stop',lambda s:np.nanmean(s)-0.003),结构止损胜率=('r_stop',lambda s:np.nanmean(s>0)),
    平均最大不利=('mae','mean'),平均最大有利=('mfe','mean'))
agg.to_csv(os.path.join(BASE,'03_结果/结构止损距离分组.csv'),encoding='utf-8-sig')
R.to_csv(os.path.join(BASE,'03_结果/结构止损距离_明细.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',250)
print(agg.round(4).to_string())
print('\n止损距离分位',R.groupby('期').sd.describe(percentiles=[.1,.25,.5,.75,.9]).round(3).to_string())
