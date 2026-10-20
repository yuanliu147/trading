# -*- coding: utf-8 -*-
import os,pickle,numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif']=['PingFang SC','Arial Unicode MS','Heiti SC']; plt.rcParams['axes.unicode_minus']=False
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; ci={c:i for i,c in enumerate(codes)}
C=P['close'].values; ATR=P['atr_pct'].values; AGE=P['trend_age'].values; DD60=P['dd60'].values
LU=P['lu_count20'].values; SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values
MB=P['market_breadth20'].values; F20=P['fwd20'].values; AMT=P['amt20'].values
U=P['up_struct'].values; NG=P['next_gap'].values; NLU=P['next_limit_up'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U&np.isfinite(F20)&(AMT>=3e7)&(C>=2)&(AGE>=5)&date_ok)
tr=valid&(period[:,None]==0); atr_thr=np.nanquantile(ATR[tr],0.40)
sig=valid&(AGE<=30)&(ATR<=atr_thr)&(LU==0)&(SBR>=0.6)&(SR20>0.05)&(DD60>-0.08)&(MB[:,None]>0.5)&~NLU&(NG<0.05)
stock_list=[('600519.SH','贵州茅台'),('300750.SZ','宁德时代'),('300308.SZ','中际旭创'),('688256.SH','寒武纪'),('600036.SH','招商银行')]
fig,axes=plt.subplots(3,2,figsize=(14,11)); axes=axes.ravel()
for ax,(code,name) in zip(axes,stock_list):
    j=ci[code]; c=pd.Series(C[:,j],index=dates).dropna(); c=c[c.index>=pd.Timestamp('2018-01-01')]
    norm=c/c.iloc[0]
    ax.plot(norm.index,norm.values,color='#4C78A8',lw=1.1)
    s=sig[:,j]&(dates>=c.index[0])
    ax.scatter(dates[s],norm.reindex(dates[s]).values,s=9,color='#E45756',alpha=.55,label='系统信号')
    r20=pd.Series(P['ret20'].values[:,j],index=dates).reindex(c.index)
    tops=r20.dropna().sort_values(ascending=False); chosen=[]
    for dt,v in tops.items():
        if all(abs((dt-x).days)>=25 for x in chosen): chosen.append(dt)
        if len(chosen)>=3: break
    for dt in chosen:
        if dt in norm.index: ax.scatter(dt,norm.loc[dt],marker='*',s=150,color='#F2A900',edgecolor='black',lw=.5,zorder=5,label='最大20日涨幅起点' if dt==chosen[0] else None)
    ax.set_yscale('log'); ax.set_title(f'{name}（{code}）')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y:.1f}'))
    ax.grid(alpha=.2); ax.legend(frameon=False,fontsize=8,loc='upper left')
axes[-1].axis('off')
fig.suptitle('五只代表性股票：系统信号与最大 20 日涨幅起点',fontsize=15)
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(BASE,'04_图表/图7_五只股票案例.png'),dpi=200); plt.close()
print('ok')
