# -*- coding: utf-8 -*-
"""五只代表性股票的历史走势与系统信号画像。"""
import pickle,numpy as np,pandas as pd,os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; ci={c:i for i,c in enumerate(codes)}
C=P['close'].values; O=P['open'].values; H=P['high'].values; L=P['low'].values
ATR=P['atr_pct'].values; AGE=P['trend_age'].values; DD60=P['dd60'].values; LU=P['lu_count20'].values
SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values; MB=P['market_breadth20'].values
DMA20=P['dist_ma20'].values; F20=P['fwd20'].values; R20=P['ret20'].values; AMT=P['amt20'].values
U=P['up_struct'].values; NG=P['next_gap'].values; NLU=P['next_limit_up'].values
IND=P['industry_id'].values; industries=P['meta']['industries']
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U&(AMT>=3e7)&(C>=2)&(AGE>=5)&date_ok)
tr=valid&(period[:,None]==0); atr_thr=0.044
sig=valid&(AGE<=30)&(ATR<=atr_thr)&(LU==0)&(SBR>=0.6)&(SR20>0.05)&(DD60>-0.08)&(MB[:,None]>0.5)
stock_list=[('600519.SH','贵州茅台','消费/低波慢牛'),('300750.SZ','宁德时代','成长/高波'),('300308.SZ','中际旭创','通信/AI动量'),
            ('688256.SH','寒武纪','半导体/极端波动'),('600036.SH','招商银行','银行/低波价值')]
rows=[]; detail=[]
for code,name,style in stock_list:
    j=ci.get(code)
    if j is None: print('missing',code); continue
    c=pd.Series(C[:,j],index=dates).dropna()
    if len(c)==0: continue
    r20=pd.Series(R20[:,j],index=dates)
    # 最大的20日涨幅，间隔至少20日
    tops=r20.dropna().sort_values(ascending=False)
    chosen=[]
    for dt,v in tops.items():
        if all(abs((dt-x).days)>=25 for x in chosen): chosen.append(dt)
        if len(chosen)>=5: break
    ep=sig[:,j]
    idx=np.where(ep)[0]
    # 趋势episode
    u=U[:,j].astype(int); dd=np.diff(np.r_[0,u,0]); starts=np.where(dd==1)[0]; ends=np.where(dd==-1)[0]-1
    eps=[(dates[s],dates[e],e-s+1,C[e,j]/C[s,j]-1) for s,e in zip(starts,ends) if e-s+1>=5 and np.isfinite(C[e,j]) and np.isfinite(C[s,j])]
    rows.append(dict(代码=code,名称=name,风格=style,首日=str(c.index[0].date()),末日=str(c.index[-1].date()),
        累计收益=c.iloc[-1]/c.iloc[0]-1,最大回撤=(c/c.cummax()-1).min(),趋势段数=len(eps),
        中位持续=np.median([x[2] for x in eps]) if eps else np.nan,中位涨幅=np.median([x[3] for x in eps]) if eps else np.nan,
        最大单段=max([x[3] for x in eps]) if eps else np.nan,系统信号日=int(ep.sum()),
        信号后20日均值=np.nanmean(F20[ep,j]) if ep.sum() else np.nan,信号后20日胜率=np.nanmean(F20[ep,j]>0) if ep.sum() else np.nan))
    for dt in chosen:
        i=np.where(dates==dt)[0][0]
        if i+1>=len(dates): continue
        detail.append({'代码':code,'名称':name,'起点':str(dt.date()),'20日涨幅':R20[i,j],'趋势年龄':AGE[i,j],'波幅':ATR[i,j],
            '行业':industries[IND[i,j]-1] if IND[i,j]>0 else '未知','行业宽度':SBR[i,j],'行业20日涨幅':SR20[i,j],
            '市场宽度':MB[i],'距20日均线':DMA20[i,j],'20日涨停次数':LU[i,j],'是否系统信号':bool(sig[i,j])})
S=pd.DataFrame(rows); D=pd.DataFrame(detail)
S.to_csv(os.path.join(BASE,'03_结果/个股案例_概览.csv'),index=False,encoding='utf-8-sig')
D.to_csv(os.path.join(BASE,'03_结果/个股案例_大涨幅起点.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',260); pd.set_option('display.max_columns',30)
print(S.round(4).to_string(index=False))
print('\n大涨幅起点'); print(D.round(4).to_string(index=False))
