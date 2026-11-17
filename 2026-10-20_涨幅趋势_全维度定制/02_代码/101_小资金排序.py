# -*- coding: utf-8 -*-
"""严格口径下，同日排名前1/2/4的信号表现。"""
import pickle,numpy as np,pandas as pd,os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; C=P['close'].values; F20=P['fwd20'].values
AMT=P['amt20'].values; ATR=P['atr_pct'].values; AGE=P['trend_age'].values; DD60=P['dd60'].values
LU=P['lu_count20'].values; SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values
MB=P['market_breadth20'].values; U=P['up_struct'].values; DMA20=P['dist_ma20'].values
SSZ=P['stock_sector_z'].values; GAPC=P['gap_count20'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]; yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U&(AMT>=3e7)&(C>=2)&(AGE>=5)&date_ok)
base=valid&(AGE<=30)&(ATR<=0.044)&(LU==0)&(SBR>=0.6)&(SR20>0.05)&(DD60>-0.08)&(MB[:,None]>0.5)
ii,jj=np.where(base)
D=pd.DataFrame({'date':dates[ii],'code':[codes[j] for j in jj],'period':period[ii],'r':F20[ii,jj]-0.003,
 'amt':AMT[ii,jj],'atr':ATR[ii,jj],'sbr':SBR[ii,jj],'sr':SR20[ii,jj],'dma':DMA20[ii,jj],'age':AGE[ii,jj],
 'ssz':SSZ[ii,jj],'gapc':GAPC[ii,jj],'price':C[ii,jj]})
# 排名：1=最好
D['r_amt']=D.groupby(['period','date']).amt.rank(method='first')
D['r_atr']=D.groupby(['period','date']).atr.rank(method='first')
D['r_sbr']=D.groupby(['period','date']).sbr.rank(ascending=False,method='first')
D['r_sr']=D.groupby(['period','date']).sr.rank(ascending=False,method='first')
D['r_score']=D.groupby(['period','date']).apply(lambda g: (g.sbr+g.sr-g.dma-g.age*0.01-g.gapc*0.02).rank(ascending=False,method='first')).reset_index(level=[0,1],drop=True)
rows=[]
for col in ['r_amt','r_atr','r_sbr','r_sr','r_score']:
    for n in [1,2,4]:
        for pi,per in [(0,'训练'),(1,'检验')]:
            x=D[(D.period==pi)&(D[col]<=n)].r
            if len(x)==0: continue
            rows.append(dict(排序=col,前N=n,期=per,样本=len(x),均值=x.mean(),中位=np.median(x),胜率=(x>0).mean(),盈亏比=x[x>0].sum()/max(1e-9,-x[x<0].sum())))
out=pd.DataFrame(rows); out.to_csv(os.path.join(BASE,'03_结果/小资金排序.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',240)
print(out.round(4).to_string(index=False))
print('每日候选数',D.groupby(['period','date']).size().groupby('period').mean())
