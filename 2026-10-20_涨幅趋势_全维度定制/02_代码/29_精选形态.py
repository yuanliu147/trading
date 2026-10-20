# -*- coding: utf-8 -*-
"""寻找“涨幅可观、胜率较高、结构止损近”的窄形态。"""
import pickle,numpy as np,pandas as pd,os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; nD=len(dates)
O=P['open'].values; H=P['high'].values; L=P['low'].values; C=P['close'].values
ATR=P['atr_pct'].values; F20=P['fwd20'].values; AMT=P['amt20'].values
U=P['up_struct'].values; AGE=P['trend_age'].values; DD60=P['dd60'].values
LU=P['lu_count20'].values; SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values
MB=P['market_breadth20'].values; NG=P['next_gap'].values; NLU=P['next_limit_up'].values
OWD=P['one_word_down'].values; DMA20=P['dist_ma20'].values; VR=P['vr5_20'].values
MA5=P['ma5'].values; MA10=P['ma10'].values; MA20=P['ma20'].values; CLOSE_POS=P['close_pos'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U&np.isfinite(F20)&(AMT>=3e7)&(C>=2)&(AGE>=5)&date_ok)
tr=valid&(period[:,None]==0); atr_thr=np.nanquantile(ATR[tr],0.40)
# 10日平台宽度
H10=pd.DataFrame(H,index=dates,columns=codes).rolling(10,min_periods=6).max().values
L10=pd.DataFrame(L,index=dates,columns=codes).rolling(10,min_periods=6).min().values
W10=H10/L10-1
L5=pd.DataFrame(L,index=dates,columns=codes).rolling(5,min_periods=3).min().values
base_gate=(MB[:,None]>0.5)&(SBR>=0.6)&(SR20>0.05)&(LU==0)&~NLU&(NG<0.05)
pat={
 '平台突破收紧6%': valid&base_gate&(AGE>=10)&(AGE<=60)&(ATR<=atr_thr)&(W10<=0.06)&(C>=H10*0.99),
 '平台突破收紧8%': valid&base_gate&(AGE>=10)&(AGE<=60)&(ATR<=atr_thr)&(W10<=0.08)&(C>=H10*0.98),
 '首次回踩20日线': valid&base_gate&(AGE>=10)&(AGE<=40)&(ATR<=atr_thr)&(DMA20>=-0.02)&(DMA20<=0.02)&(DD60<=-0.03)&(DD60>=-0.10)&(VR<=1.0)&(C>MA5),
 '高收紧加板块': valid&base_gate&(AGE>=10)&(AGE<=60)&(ATR<=atr_thr)&(DMA20<0.03)&(DMA20>-0.03)&(DD60>-0.05),
 '平台突破无板块': valid&(AGE>=10)&(AGE<=60)&(ATR<=atr_thr)&(W10<=0.08)&(C>=H10*0.98)&(LU==0)&(MB[:,None]>0.5),
}
rows=[]
for name,m in pat.items():
    ii,jj=np.where(m)
    recs=[]
    for i,j in zip(ii,jj):
        e=i+1
        if e>=nD: continue
        entry=O[e,j]
        if not np.isfinite(entry) or entry<=0: continue
        sl=L5[i,j]
        if not np.isfinite(sl): continue
        stop=sl*0.995; sd=1-stop/entry
        if not np.isfinite(sd) or sd<=0 or sd>0.12: continue
        end=min(e+19,nD-1); r20=C[end,j]/entry-1 if np.isfinite(C[end,j]) else np.nan
        hit=None
        for t in range(e,end+1):
            if np.isfinite(L[t,j]) and L[t,j]<=stop: hit=t; break
        if hit is not None:
            rstop=np.nan if OWD[hit,j] else (min(O[hit,j],stop) if np.isfinite(O[hit,j]) else stop)/entry-1
        else: rstop=r20
        recs.append((period[i],sd,r20,rstop,np.nanmin(L[e:end+1,j]/entry-1),np.nanmax(H[e:end+1,j]/entry-1)))
    if not recs: continue
    D=pd.DataFrame(recs,columns=['period','sd','r20','rstop','mae','mfe'])
    for pi,per in [(0,'训练'),(1,'检验')]:
        x=D[D.period==pi]
        if len(x)<50: continue
        rows.append(dict(形态=name,期=per,样本=len(x),止损距离=round(x.sd.median(),4),
            固定20日净=x.r20.mean()-0.003,固定胜率=(x.r20>0.003).mean(),
            止损后净=np.nanmean(x.rstop)-0.003,止损后胜率=np.nanmean(x.rstop>0.003),
            MAE=x.mae.mean(),MFE=x.mfe.mean()))
out=pd.DataFrame(rows); out.to_csv(os.path.join(BASE,'03_结果/精选形态.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',250)
print(out.round(4).to_string(index=False))
