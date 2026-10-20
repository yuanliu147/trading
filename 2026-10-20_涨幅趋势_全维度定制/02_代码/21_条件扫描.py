# -*- coding: utf-8 -*-
"""上涨趋势日内的条件扫描：每个特征分5组，训练期/检验期分别统计20日超额。"""
import pickle, numpy as np, pandas as pd, os, time
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
OUT=os.path.join(BASE,'03_结果'); os.makedirs(OUT,exist_ok=True)
t0=time.time()
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026.pkl'),'rb'))
dates=P['meta']['dates']; C=P['close']
U=P['up_struct'].values
F20=P['fwd20'].values
# 基准：当日全A等权平均20日收益（不是中位数）
bench=pd.Series(np.nanmean(F20,axis=1),index=dates)
bench_arr=bench.reindex(dates).values[:,None]
EXC=F20-bench_arr
MIN=P['fwd_min20_entry'].values; MAX=P['fwd_max20_entry'].values
AGE=P['trend_age'].values; AMT=P['amt20'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
valid=(U & np.isfinite(F20) & np.isfinite(EXC) & (AMT>=3e7) & (C.values>=2) & (AGE>=5) & date_ok)
print('趋势日样本',valid.sum(), '日期',dates[valid.any(axis=1)][0], dates[valid.any(axis=1)][-1])
yr=np.array([d.year for d in dates])
period=np.where(yr<=2021,0,1)
rows=[]
features=['trend_age','ep_gain','dist_ma20','dd60','er20','ma20_slope','ma60_slope','up_ratio20','nh20_count',
 'vol20','atr_pct','amt_trend','vr5_20','rsi14','macd_hist','consec_up','ret20_hist_z','dist_high250','dist_high500',
 'own_base20','own_win20','vol_hist_z','sector_breadth20','sector_breadth60','sector_ret20','sector_ret60','sector_rs',
 'stock_sector_z','sw_above60','sw_ret20','sw_slope60','market_breadth20','market_sent','market_dispersion',
 'market_limitup','market_amt_ratio','market_newhigh','market_ret20','amt_ratio20','gap','upper','close_pos',
 'lu_count20','vp_corr20','gap_count20','turn_z250','mcap','amt20']
for f in features:
    x=P[f].values
    if x.ndim==1: x=np.broadcast_to(x[:,None],F20.shape)
    for pi,per in [(0,'训练 2017-2021'),(1,'检验 2022-2026')]:
        m=valid & (period[:,None]==pi) & np.isfinite(x)
        if m.sum()<5000: continue
        xv=x[m]
        try:
            q=pd.qcut(xv,5,labels=False,duplicates='drop')
        except Exception:
            continue
        ex=EXC[m]; f20=F20[m]; mn=MIN[m]; mx=MAX[m]
        if pd.isna(q).all(): continue
        for g in range(int(np.nanmax(q))+1):
            sel=(q==g)
            if sel.sum()<200: continue
            e=ex[sel]
            rows.append(dict(特征=f,期=per,组=g+1,样本=len(e),下限=float(np.nanquantile(xv,[0,.2,.4,.6,.8,1][g])),
                上限=float(np.nanquantile(xv,[0,.2,.4,.6,.8,1][g+1])),
                超额均值=e.mean(),超额中位=np.median(e),超额胜率=(e>0).mean(),
                绝对收益均值=f20[sel].mean(),绝对胜率=(f20[sel]>0).mean(),
                平均最大不利=mn[sel].mean(),平均最大有利=mx[sel].mean(),
                盈亏比=(f20[sel][f20[sel]>0].sum()/max(1e-9,-f20[sel][f20[sel]<0].sum()))))
out=pd.DataFrame(rows)
out.to_csv(os.path.join(OUT,'上涨趋势_条件扫描.csv'),index=False,encoding='utf-8-sig')
# 汇总：每特征两期最高组-最低组
piv=out.pivot_table(index=['特征','期'],columns='组',values='超额均值')
spread=[]
for f in out.特征.unique():
    rec={'特征':f}
    for per in ['训练 2017-2021','检验 2022-2026']:
        try:
            s=piv.loc[(f,per)]
            rec[per+'_低']=s.iloc[0]; rec[per+'_高']=s.iloc[-1]; rec[per+'_差']=s.iloc[-1]-s.iloc[0]
            rec[per+'_方向一致']=np.sign(rec[per+'_差'])==np.sign(rec['训练 2017-2021_差']) if '训练 2017-2021_差' in rec else np.nan
        except Exception: pass
    spread.append(rec)
sp=pd.DataFrame(spread)
sp['稳定差']=sp[['训练 2017-2021_差','检验 2022-2026_差']].abs().min(axis=1)*np.sign(sp['训练 2017-2021_差'])
sp=sp.sort_values('稳定差',key=lambda s:s.abs(),ascending=False)
sp.to_csv(os.path.join(OUT,'上涨趋势_条件扫描_汇总.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
print(sp.head(25).to_string(index=False))
print('耗时',time.time()-t0)
