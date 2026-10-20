# -*- coding: utf-8 -*-
"""候选信号的持有期/止损/止盈网格（事件级）。"""
import pickle, numpy as np, pandas as pd, os, time
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; nD=len(dates)
O=P['open'].values; H=P['high'].values; L=P['low'].values; C=P['close'].values
ATR=P['atr_pct'].values; F20=P['fwd20'].values
U=P['up_struct'].values; AGE=P['trend_age'].values; AMT=P['amt20'].values
DD60=P['dd60'].values; LU=P['lu_count20'].values
SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values
MKT=P['regime_id'].values.reshape(-1,1); NG=P['next_gap'].values; NLU=P['next_limit_up'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U & np.isfinite(F20) & (AMT>=3e7) & (C>=2) & (AGE>=5) & date_ok)
tr=valid & (period[:,None]==0); atr_thr=np.nanquantile(ATR[tr],0.40)
sig=valid & (AGE<=30) & (ATR<=atr_thr) & (MKT>=4) & (LU==0) & (SBR>=0.6) & (SR20>0.05) & (DD60>-0.08)
sig=sig & ~NLU & (NG<0.05)
ii,jj=np.where(sig); print('信号',len(ii))
rows=[]
for i,j in zip(ii,jj):
    e=i+1
    if e>=nD: continue
    entry=O[e,j]
    if not np.isfinite(entry) or entry<=0: continue
    atr=ATR[i,j]
    if not np.isfinite(atr): continue
    end=min(i+40,nD-1)
    rec={'signal_date':dates[i],'code':codes[j],'period':period[i],'entry':entry,'atr':atr,'amt':AMT[i,j]}
    # 固定持有
    for h in [5,10,15,20,25,30,40]:
        ex=min(e+h-1,nD-1)
        rec[f'r{h}']=C[ex,j]/entry-1 if np.isfinite(C[ex,j]) else np.nan
    # 止损 + 20日
    for sd_name,sd in [('st15',np.clip(1.5*atr,0.03,0.08)),('st20',np.clip(2*atr,0.04,0.12)),('st8',0.08)]:
        stop=entry*(1-sd); ex=None
        for t in range(e,min(e+20,nD)):
            if np.isfinite(L[t,j]) and L[t,j]<=stop:
                px=min(O[t,j],stop) if np.isfinite(O[t,j]) else stop
                ex=px/entry-1; break
        if ex is None:
            ex=C[min(e+19,nD-1),j]/entry-1 if np.isfinite(C[min(e+19,nD-1),j]) else np.nan
        rec[sd_name+'_h20']=ex
    # 2R止盈一半 + 20日
    stop=entry*(1-np.clip(2*atr,0.04,0.12)); R=np.clip(2*atr,0.04,0.12)
    hit_idx=None
    for t in range(e,min(e+20,nD)):
        if np.isfinite(H[t,j]) and H[t,j]>=entry*(1+2*R): hit_idx=t; break
    stop_idx=None
    for t in range(e,min(e+20,nD)):
        if np.isfinite(L[t,j]) and L[t,j]<=stop: stop_idx=t; break
    if hit_idx is not None and (stop_idx is None or hit_idx<=stop_idx):
        ex20=C[min(e+19,nD-1),j]/entry-1 if np.isfinite(C[min(e+19,nD-1),j]) else np.nan
        rec['tp2r_h20']=0.5*(2*R)+0.5*ex20
        rec['hit2r']=1
    elif stop_idx is not None:
        rec['tp2r_h20']=min(O[stop_idx,j],stop)/entry-1 if np.isfinite(O[stop_idx,j]) else -R
        rec['hit2r']=0
    else:
        ex20=C[min(e+19,nD-1),j]/entry-1 if np.isfinite(C[min(e+19,nD-1),j]) else np.nan
        rec['tp2r_h20']=ex20; rec['hit2r']=0
    rec['mae40']=np.nanmin(L[e:end+1,j]/entry-1); rec['mfe40']=np.nanmax(H[e:end+1,j]/entry-1)
    rows.append(rec)
R=pd.DataFrame(rows)
R.to_csv(os.path.join(BASE,'03_结果/持有期网格_事件级.csv'),index=False,encoding='utf-8-sig')
R['period_name']=R.period.map({0:'训练 2017-2021',1:'检验 2022-2026'})
cols=[c for c in R.columns if c.startswith('r') or c.startswith('st') or c.startswith('tp')]
out=[]
for per,g in R.groupby('period_name'):
    for c in cols:
        x=g[c].dropna()-0.003
        if len(x)==0: continue
        out.append(dict(期=per,规则=c,样本=len(x),均值=x.mean(),中位=np.median(x),胜率=(x>0).mean(),
            盈亏比=x[x>0].sum()/max(1e-9,-x[x<0].sum())))
out=pd.DataFrame(out)
out.to_csv(os.path.join(BASE,'03_结果/持有期网格_汇总.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',250)
print(out.to_string(index=False))
print('命中2R比例',R.groupby('period_name').hit2r.mean())
print('平均MAE/MFE',R.groupby('period_name')[['mae40','mfe40']].mean())
