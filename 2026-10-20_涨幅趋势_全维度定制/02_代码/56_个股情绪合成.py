# -*- coding: utf-8 -*-
"""把涨停、跳空、量比、上影线、收盘位置合成为“情绪热度”，检验高低组。"""
import pickle,numpy as np,pandas as pd,os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; C=P['close'].values; F20=P['fwd20'].values; AMT=P['amt20'].values
U=P['up_struct'].values; AGE=P['trend_age'].values; ATR=P['atr_pct'].values
LU=P['lu_count20'].values; GAPC=P['gap_count20'].values; VR=P['vr5_20'].values
AR=P['amt_ratio20'].values; UP=P['upper'].values; CP=P['close_pos'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]; yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U&np.isfinite(F20)&(AMT>=3e7)&(C>=2)&(AGE>=5)&date_ok)
# 每日在趋势日内做百分位，合成情绪热度
ranks=[]
for t in range(len(dates)):
    m=valid[t]
    if m.sum()<50: ranks.append(np.full(C.shape[1],np.nan)); continue
    r=np.zeros(C.shape[1]); cnt=np.zeros(C.shape[1])
    for arr in [LU, GAPC, VR, AR, UP, CP]:
        x=arr[t]; idx=np.where(m&np.isfinite(x))[0]
        if len(idx)==0: continue
        s=pd.Series(x[idx]).rank(pct=True).values
        r[idx]+=s; cnt[idx]+=1
    rr=np.full(C.shape[1],np.nan); ok=cnt>0; rr[ok]=r[ok]/cnt[ok]
    ranks.append(rr)
SENT=np.vstack(ranks)
rows=[]
for pi,per in [(0,'训练 2017-2021'),(1,'检验 2022-2026')]:
    m=valid&(period[:,None]==pi)&np.isfinite(SENT)
    x=SENT[m]; f=F20[m]; e=f-np.nanmean(f)
    q=pd.qcut(x,5,labels=False,duplicates='drop')
    for g in range(5):
        sel=q==g
        rows.append({'期':per,'情绪组':g+1,'样本':int(sel.sum()),'情绪均值':x[sel].mean(),'20日均值':f[sel].mean(),
            '20日胜率':(f[sel]>0).mean(),'超额均值':e[sel].mean(),'平均最大不利':P['fwd_min20_entry'].values[m][sel].mean(),
            '平均最大有利':P['fwd_max20_entry'].values[m][sel].mean()})
out=pd.DataFrame(rows); out.to_csv(os.path.join(BASE,'03_结果/个股情绪合成.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',250); print(out.round(4).to_string(index=False))
