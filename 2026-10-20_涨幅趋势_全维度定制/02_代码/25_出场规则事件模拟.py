# -*- coding: utf-8 -*-
"""在最终候选信号上比较出场规则（事件级，非组合）。"""
import pickle, numpy as np, pandas as pd, os, time
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; nD=len(dates)
O=P['open'].values; H=P['high'].values; L=P['low'].values; C=P['close'].values
MA20=P['ma20'].values; MA10=P['ma10'].values; ATR=P['atr_pct'].values
F20=P['fwd20'].values; EXC=F20-np.nanmean(F20,axis=1)[:,None]
U=P['up_struct'].values; AGE=P['trend_age'].values; AMT=P['amt20'].values
DD60=P['dd60'].values; LU=P['lu_count20'].values
SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values
MKT=P['regime_id'].values.reshape(-1,1); NG=P['next_gap'].values
NLU=P['next_limit_up'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U & np.isfinite(F20) & (AMT>=3e7) & (C>=2) & (AGE>=5) & date_ok)
tr=valid & (period[:,None]==0)
atr_thr=np.nanquantile(ATR[tr],0.40)
sig=valid & (AGE<=30) & (ATR<=atr_thr) & (MKT>=4) & (LU==0) & (SBR>=0.6) & (SR20>0.05) & (DD60>-0.08)
sig=sig & ~NLU & (NG<0.05)
print('信号数',sig.sum())
ii,jj=np.where(sig)
print(len(ii))
res=[]
t0=time.time()
for k,(i,j) in enumerate(zip(ii,jj)):
    e=i+1
    if e>=nD: continue
    entry=O[e,j]
    if not np.isfinite(entry) or entry<=0: continue
    atr=ATR[i,j]
    if not np.isfinite(atr): continue
    sd=float(np.clip(2*atr,0.04,0.12)); stop=entry*(1-sd); R=sd
    end=min(i+40,nD-1)
    path_low=L[e:end+1,j]; path_high=H[e:end+1,j]; path_close=C[e:end+1,j]
    path_ma20=MA20[e:end+1,j]; path_ma10=MA10[e:end+1,j]
    validpath=np.isfinite(path_close)
    if validpath.sum()<1: continue
    mae=float(np.nanmin(path_low/entry-1)); mfe=float(np.nanmax(path_high/entry-1))
    # 固定20
    idx20=min(e+20-1,nD-1); r20=C[idx20,j]/entry-1 if np.isfinite(C[idx20,j]) else np.nan
    r40=C[end,j]/entry-1 if np.isfinite(C[end,j]) else np.nan
    # 止损
    stop_idx=None
    for t in range(e,end+1):
        lt=L[t,j]
        if np.isfinite(lt) and lt<=stop:
            stop_idx=t; break
    if stop_idx is not None:
        px=min(O[stop_idx,j],stop) if np.isfinite(O[stop_idx,j]) else stop
        r_stop=px/entry-1
        hold_stop=stop_idx-i
    else:
        r_stop=r40; hold_stop=end-i
    # 止损+MA20收盘跌破（收盘确认，次日开盘走，保守一点用次日开盘）
    if stop_idx is not None:
        r_st=px/entry-1; hold_st=stop_idx-i
    else:
        st=None
        for t in range(e+2,end+1):
            if np.isfinite(path_close[t-e]) and np.isfinite(MA20[t,j]) and C[t,j]<MA20[t,j]:
                st=t; break
        if st is not None:
            # 次日开盘
            ex=min(st+1,end)
            px2=O[ex,j] if np.isfinite(O[ex,j]) else C[st,j]
            r_st=px2/entry-1; hold_st=ex-i
        else:
            r_st=r40; hold_st=end-i
    # 止损+MA20+2R分批1/3
    hit2=False
    if stop_idx is not None:
        r_part=r_st; hold_p=hold_st; hit2=False
    else:
        hit2=None
        for t in range(e,end+1):
            if np.isfinite(H[t,j]) and H[t,j]>=entry*(1+2*R):
                hit2=t; break
        if hit2 is not None and (st is None or hit2<=st):
            px_part=entry*(1+2*R)
            if st is None:
                r_rem=r40
            else:
                ex=min(st+1,end); r_rem=O[ex,j]/entry-1 if np.isfinite(O[ex,j]) else C[st,j]/entry-1
            r_part=(1/3)*(2*R)+(2/3)*r_rem; hold_p=(st-i if st is not None else end-i); hit2=True
        else:
            r_part=r_st; hold_p=hold_st; hit2=False
    res.append((dates[i],codes[j],period[i],r20,r40,r_stop,r_st,r_part,mae,mfe,hold_st,int(hit2)))
    if (k+1)%50000==0: print(k+1,time.time()-t0,flush=True)
R=pd.DataFrame(res,columns=['signal_date','code','period','r20','r40','r_stop40','r_stoptrail20','r_partial','mae','mfe','hold','hit2R'])
R['period']=R.period.map({0:'训练 2017-2021',1:'检验 2022-2026'})
R.to_csv(os.path.join(BASE,'03_结果/出场规则_事件级.csv'),index=False,encoding='utf-8-sig')
rows=[]
for per,g in R.groupby('period'):
    for col in ['r20','r40','r_stop40','r_stoptrail20','r_partial']:
        x=g[col].dropna()-0.003
        rows.append(dict(期=per,规则=col,样本=len(x),均值=x.mean(),中位=np.median(x),胜率=(x>0).mean(),
            盈亏比=x[x>0].sum()/max(1e-9,-x[x<0].sum()),平均持有=g['hold'].mean(),
            平均MAE=g.mae.mean(),平均MFE=g.mfe.mean()))
out=pd.DataFrame(rows)
out.to_csv(os.path.join(BASE,'03_结果/出场规则_汇总.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',250)
print(out.to_string(index=False))
print('耗时',time.time()-t0)
