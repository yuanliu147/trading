# -*- coding: utf-8 -*-
"""最终信号相对全部趋势日的随机置换检验。"""
import pickle,numpy as np,pandas as pd,os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; C=P['close'].values; F20=P['fwd20'].values; AMT=P['amt20'].values
U=P['up_struct'].values; AGE=P['trend_age'].values; ATR=P['atr_pct'].values; DD60=P['dd60'].values
LU=P['lu_count20'].values; SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values
MB=P['market_breadth20'].values; NG=P['next_gap'].values; NLU=P['next_limit_up'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]; yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U&np.isfinite(F20)&(AMT>=3e7)&(C>=2)&(AGE>=5)&date_ok)
tr=valid&(period[:,None]==0); atr_thr=np.nanquantile(ATR[tr],0.40)
sig=valid&(AGE<=30)&(ATR<=atr_thr)&(LU==0)&(SBR>=0.6)&(SR20>0.05)&(DD60>-0.08)&(MB[:,None]>0.5)&~NLU&(NG<0.05)
rng=np.random.default_rng(20261020); rows=[]
for pi,per in [(0,'训练 2017-2021'),(1,'检验 2022-2026')]:
    pool=np.where((valid&(period[:,None]==pi)).ravel())[0]
    sm=np.where((sig&(period[:,None]==pi)).ravel())[0]
    y=F20.ravel()[pool]-0.003
    actual=F20.ravel()[sm].mean()-0.003
    n=len(sm); rand=np.array([rng.choice(y,n,replace=False).mean() for _ in range(500)])
    p=(rand>=actual).mean()
    rows.append(dict(期=per,趋势日样本=len(pool),信号数=n,实际净收益=actual,随机均值=rand.mean(),随机95分位=np.quantile(rand,.95),置换p值=p))
out=pd.DataFrame(rows); out.to_csv(os.path.join(BASE,'03_结果/置换检验.csv'),index=False,encoding='utf-8-sig')
print(out.round(4).to_string(index=False))
