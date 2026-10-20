# -*- coding: utf-8 -*-
"""同一天多个信号时，比较不同排序取前4/前8的效果（事件级，20日持有）。"""
import pickle, numpy as np, pandas as pd, os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; C=P['close'].values; O=P['open'].values
F20=P['fwd20'].values; U=P['up_struct'].values; AGE=P['trend_age'].values
ATR=P['atr_pct'].values; AMT=P['amt20'].values; DD60=P['dd60'].values; DMA20=P['dist_ma20'].values
LU=P['lu_count20'].values; SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values
SSZ=P['stock_sector_z'].values; MKT=P['regime_id'].values.reshape(-1,1)
MB=P['market_breadth20'].values.reshape(-1,1); NG=P['next_gap'].values; NLU=P['next_limit_up'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U & np.isfinite(F20) & (AMT>=3e7) & (C>=2) & (AGE>=5) & date_ok)
tr=valid & (period[:,None]==0); atr_thr=np.nanquantile(ATR[tr],0.40)
sig=valid & (AGE<=30) & (ATR<=atr_thr) & (MKT>=4) & (LU==0) & (SBR>=0.6) & (SR20>0.05) & (DD60>-0.08)
sig=sig & ~NLU & (NG<0.05)
ii,jj=np.where(sig)
d=pd.DataFrame({'signal_date':dates[ii],'code':[codes[j] for j in jj],'period':period[ii],
 'r20':F20[ii,jj]-0.003,'atr':ATR[ii,jj],'amt':AMT[ii,jj],'age':AGE[ii,jj],'dd60':DD60[ii,jj],
 'dma20':DMA20[ii,jj],'sbr':SBR[ii,jj],'sr20':SR20[ii,jj],'ssz':SSZ[ii,jj],'mb':MB[ii,0]})
# 每日排名（1=最好）
d['r_atr']=d.groupby(['period','signal_date']).atr.rank(method='first')
d['r_amt']=d.groupby(['period','signal_date']).amt.rank(method='first')
d['r_age']=d.groupby(['period','signal_date']).age.rank(method='first')
d['r_dd']=d.groupby(['period','signal_date']).dd60.rank(ascending=False,method='first')
d['r_dma']=d.groupby(['period','signal_date']).dma20.rank(method='first')
d['r_sbr']=d.groupby(['period','signal_date']).sbr.rank(ascending=False,method='first')
d['r_sr']=d.groupby(['period','signal_date']).sr20.rank(ascending=False,method='first')
d['r_ssz']=d.groupby(['period','signal_date']).ssz.rank(method='first')
# 综合分：低波、板块强、距MA20近、年龄小、市场宽度
z=lambda x:(x-x.mean())/(x.std()+1e-9)
d['score']=(-z(d.atr) + z(d.sbr) + z(d.sr20) - z(d.dma20) - 0.5*z(d.age) + 0.5*z(d.mb))
d['r_score']=d.groupby(['period','signal_date']).score.rank(ascending=False,method='first')
d['r_score2']=d.groupby(['period','signal_date']).apply(lambda g: (g.score - 0.3*z(g.amt)).rank(ascending=False,method='first')).reset_index(level=[0,1],drop=True)
rows=[]
for rank_col in ['r_atr','r_amt','r_age','r_dd','r_dma','r_sbr','r_sr','r_ssz','r_score','r_score2']:
    for topn in [4,8,20]:
        for per,pername in [(0,'训练 2017-2021'),(1,'检验 2022-2026')]:
            x=d[(d.period==per)&(d[rank_col]<=topn)].r20
            if len(x)==0: continue
            rows.append(dict(排序=rank_col,前N=topn,期=pername,样本=len(x),均值=x.mean(),中位=np.median(x),
                胜率=(x>0).mean(),盈亏比=x[x>0].sum()/max(1e-9,-x[x<0].sum())))
out=pd.DataFrame(rows)
out.to_csv(os.path.join(BASE,'03_结果/信号排序.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',250); pd.set_option('display.max_rows',100)
print(out[out.期.str.startswith('训练')].pivot_table(index=['排序','前N'],columns='期',values='均值').to_string())
print('\n--- 检验 ---')
print(out[out.期.str.startswith('检验')].pivot_table(index=['排序','前N'],columns='期',values='均值').to_string())
print('\n样本/日',d.groupby(['period','signal_date']).size().groupby('period').mean())
