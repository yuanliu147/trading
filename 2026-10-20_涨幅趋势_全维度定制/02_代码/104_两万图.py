# -*- coding: utf-8 -*-
import os,pandas as pd,numpy as np,pickle
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif']=['PingFang SC','Arial Unicode MS','Heiti SC']; plt.rcParams['axes.unicode_minus']=False
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'; OUT=os.path.join(BASE,'03_结果')
files={'稳健':'两万_权益_稳健.csv','均衡':'两万_权益_均衡.csv','激进':'两万_权益_激进.csv'}
colors={'稳健':'#4C78A8','均衡':'#F58518','激进':'#E45756'}
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(13,8.5),sharex=True,gridspec_kw={'height_ratios':[3,1]})
for name,f in files.items():
    E=pd.read_csv(os.path.join(OUT,f),parse_dates=['date']).set_index('date')
    ax1.plot(E.index,E.equity/10000,label=f'{name}（期末{E.equity.iloc[-1]/10000:.1f}万）',color=colors[name],lw=1.5)
    dd=E.equity/E.equity.cummax()-1
    ax2.plot(E.index,dd,color=colors[name],lw=1,label=name)
# benchmark
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
bm=pd.Series(P['market_index'],index=P['meta']['dates']).reindex(E.index).ffill(); bm=bm/bm.iloc[0]*2
ax1.plot(bm.index,bm.values,color='gray',lw=1,alpha=.6,label='全部股票等权（起点2万）')
ax1.set_ylabel('账户权益（万元）'); ax1.set_title('2万元账户：1–2只持仓三档方案（整手、最低佣金、严谨回撤）')
ax1.legend(frameon=False,fontsize=9); ax1.grid(alpha=.2)
ax2.fill_between([], [], color='none')
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y:.0%}'))
ax2.set_ylabel('回撤'); ax2.legend(frameon=False,ncol=3,fontsize=8); ax2.grid(alpha=.2)
plt.tight_layout(); plt.savefig(os.path.join(BASE,'04_图表/图8_两万账户三档.png'),dpi=200); plt.close()
print('ok')
