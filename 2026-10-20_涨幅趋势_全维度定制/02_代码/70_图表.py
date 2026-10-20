# -*- coding: utf-8 -*-
import os, pickle, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.rcParams['font.sans-serif']=['PingFang SC','Arial Unicode MS','Heiti SC']
plt.rcParams['axes.unicode_minus']=False
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
OUT=os.path.join(BASE,'03_结果'); PIC=os.path.join(BASE,'04_图表'); os.makedirs(PIC,exist_ok=True)

# 图1 全部股票20日收益分布 + 年度
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; R20=P['ret20'].values
xall=R20[(dates>=pd.Timestamp('2017-10-13'))[:,None] & np.isfinite(R20)]
x=xall[(xall>-0.6)&(xall<0.8)]
xall64=xall.astype('float64')
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
ax[0].hist(x,bins=180,color='#4C78A8',alpha=.85)
ax[0].axvline(np.mean(xall),color='#E45756',lw=2,label=f'均值 {np.mean(xall64):.2%}')
ax[0].axvline(np.median(xall),color='#F58518',lw=2,label=f'中位 {np.median(xall64):.2%}')
ax[0].set_title('全部股票 20 日收益分布（2017-10 至 2026-10）')
ax[0].set_xlabel('20 日收益'); ax[0].set_ylabel('股票-日数')
ax[0].xaxis.set_major_formatter(PercentFormatter(1.0)); ax[0].legend(frameon=False)
ax[0].text(0.02,0.95,f'胜率 {np.mean(xall>0):.1%}\n>+10% {np.mean(xall>0.1):.1%}\n<-10% {np.mean(xall<-0.1):.1%}',
    transform=ax[0].transAxes,va='top',fontsize=10,bbox=dict(fc='white',ec='none',alpha=.8))
ann=pd.read_csv(os.path.join(OUT,'A股年度分布.csv'))
ann=ann[ann['年份']>=2018]
ax[1].bar(ann['年份'].astype(str),ann['等权年度收益'],color=['#54A24B' if v>=0 else '#E45756' for v in ann['等权年度收益']])
ax[1].axhline(0,color='black',lw=.8)
ax[1].set_title('全部股票等权年度收益')
ax[1].yaxis.set_major_formatter(PercentFormatter(1.0))
for i,v in enumerate(ann['等权年度收益']): ax[1].text(i,v+(0.01 if v>=0 else -0.03),f'{v:.1%}',ha='center',fontsize=9)
ax[1].set_ylim(min(ann['等权年度收益'])-.08,max(ann['等权年度收益'])+.08)
plt.tight_layout(); plt.savefig(os.path.join(PIC,'图1_全部股票收益分布.png'),dpi=200); plt.close()

# 图2 趋势年龄
AT=pd.read_csv(os.path.join(OUT,'上涨趋势_年龄阶段.csv'))
fig,ax1=plt.subplots(figsize=(11,5.2))
ax1.bar(AT['年龄'],AT['20日均值'],color='#4C78A8',alpha=.85)
ax1.axhline(0,color='black',lw=.8); ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
ax1.set_ylabel('未来 20 日平均收益'); ax1.set_xlabel('趋势年龄（交易日）')
ax1.set_ylim(-0.025,0.013)
ax2=ax1.twinx(); ax2.plot(AT['年龄'],AT['20日胜率'],color='#E45756',marker='o',lw=2)
ax2.yaxis.set_major_formatter(PercentFormatter(1.0)); ax2.set_ylabel('20 日胜率')
ax1.set_title('上涨趋势越年轻，未来 20 日收益越高')
for i,v in enumerate(AT['20日均值']): ax1.text(i,v+(0.0005 if v>=0 else -0.0008),f'{v:.2%}',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
plt.tight_layout(); plt.subplots_adjust(bottom=0.18); plt.savefig(os.path.join(PIC,'图2_趋势年龄与未来收益.png'),dpi=200); plt.close()

# 图3 条件特征高低组差
SC=pd.read_csv(os.path.join(OUT,'上涨趋势_条件扫描.csv'))
piv=SC.pivot_table(index=['特征','期'],columns='组',values='超额均值')
name_map={'atr_pct':'平均真实波幅','vol20':'20日波动率','amt20':'20日成交额','trend_age':'趋势年龄',
 'dist_ma20':'距20日均线','lu_count20':'20日涨停次数','gap_count20':'20日跳空次数','sector_breadth60':'行业宽度60',
 'sw_ret20':'行业指数20日涨幅','market_ret20':'市场20日涨幅','market_sent':'市场情绪','turn_z250':'换手率历史标准分',
 'ret20_hist_z':'20日涨幅自身历史标准分','own_base20':'自身历史趋势基准','ma60_slope':'60日均线斜率',
 'sector_rs':'行业相对强弱','stock_sector_z':'行业内标准分','dist_high250':'距250日高点','vr5_20':'5/20日量比'}
feats=['atr_pct','vol20','amt20','trend_age','dist_ma20','lu_count20','gap_count20','sector_breadth60','sw_ret20','market_ret20','market_sent','turn_z250']
rows=[]
for f in feats:
    try:
        tr=piv.loc[(f,'训练 2017-2021')]; te=piv.loc[(f,'检验 2022-2026')]
        rows.append(dict(特征=name_map.get(f,f),训练=tr.iloc[-1]-tr.iloc[0],检验=te.iloc[-1]-te.iloc[0]))
    except Exception: pass
D=pd.DataFrame(rows).sort_values('训练')
fig,ax=plt.subplots(figsize=(10.5,6.2))
y=np.arange(len(D)); h=.36
ax.barh(y-h/2,D['训练'],height=h,color='#4C78A8',label='训练 2017-2021')
ax.barh(y+h/2,D['检验'],height=h,color='#F58518',label='检验 2022-2026')
ax.set_yticks(y); ax.set_yticklabels(D['特征']); ax.axvline(0,color='black',lw=.8)
ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.set_xlabel('最高组减最低组的 20 日超额收益差')
ax.set_title('哪些上涨趋势特征有稳定的组间差异')
ax.legend(frameon=False); plt.tight_layout(); plt.savefig(os.path.join(PIC,'图3_特征组间差.png'),dpi=200); plt.close()

# 图4 系统权益与回撤
Tr=pd.read_csv(os.path.join(OUT,'权益_含退市_训练.csv'),parse_dates=['date']).set_index('date')
Te=pd.read_csv(os.path.join(OUT,'权益_含退市_检验.csv'),parse_dates=['date']).set_index('date')
_scale=Tr.equity.iloc[-1]; Te['equity']=Te['equity']*_scale; Te['cash']=Te['cash']*_scale; Te['equity_low']=Te['equity_low']*_scale
E=pd.concat([Tr,Te]).set_index('date') if False else pd.concat([Tr,Te])
E=E[~E.index.duplicated()]
idx=pd.Series(P['market_index'].values,index=dates).reindex(E.index).ffill()
bench=idx/idx.iloc[0]
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[3,1]})
ax1.plot(E.index,E.equity,label='定制系统',color='#E45756',lw=1.8)
ax1.plot(E.index,bench,label='全部股票等权',color='#4C78A8',lw=1.2,alpha=.8)
ax1.set_yscale('log'); ax1.set_ylabel('净值（对数轴）'); ax1.legend(frameon=False)
ax1.axvline(pd.Timestamp('2022-01-01'),color='gray',ls='--',lw=1)
ax1.text(pd.Timestamp('2022-01-10'),ax1.get_ylim()[1]*.9,'样本外起点',fontsize=9,color='gray')
ax1.set_title('定制系统：全期最大回撤 -13.9%（收盘）/ -14.2%（按最低价）')
dd=(E.equity/E.equity.cummax()-1)
dd_low=(E.equity_low/E.equity.cummax()-1)
ddb=(bench/bench.cummax()-1)
ax2.fill_between(E.index,dd,0,color='#E45756',alpha=.40,label='系统回撤（收盘）')
ax2.plot(E.index,dd_low,color='#B22222',lw=1,ls='--',label='系统回撤（按最低价）')
ax2.plot(E.index,ddb,color='#4C78A8',lw=1,alpha=.8,label='等权回撤')
ax2.yaxis.set_major_formatter(PercentFormatter(1.0)); ax2.legend(frameon=False,ncol=3,fontsize=8); ax2.set_ylabel('回撤')
plt.tight_layout(); plt.savefig(os.path.join(PIC,'图4_系统权益与回撤.png'),dpi=200); plt.close()

# 图5 年度收益对比
def yr_ret(series):
    s=series.copy(); out={}
    for y,g in s.groupby(s.index.year):
        base=s[s.index<g.index[0]]
        start=base.iloc[-1] if len(base) else g.iloc[0]
        out[y]=g.iloc[-1]/start-1
    return pd.Series(out)
sy=yr_ret(E.equity); by=yr_ret(bench)
Y=pd.DataFrame({'系统':sy,'等权':by}).dropna()
fig,ax=plt.subplots(figsize=(11,5.2))
x=np.arange(len(Y)); w=.38
ax.bar(x-w/2,Y['系统'],w,label='定制系统',color='#E45756')
ax.bar(x+w/2,Y['等权'],w,label='全部股票等权',color='#4C78A8',alpha=.8)
ax.set_xticks(x); ax.set_xticklabels(Y.index.astype(str)); ax.axhline(0,color='black',lw=.8)
ax.yaxis.set_major_formatter(PercentFormatter(1.0)); ax.set_title('年度收益：系统每年为正（除 2018）')
ax.legend(frameon=False); plt.tight_layout(); plt.savefig(os.path.join(PIC,'图5_年度收益.png'),dpi=200); plt.close()

# 图6 事件类型
EV=pd.read_csv(os.path.join(OUT,'上涨趋势_事件类型.csv'))
sel=['全部趋势日','早期趋势(5-30日)','突破20日新高','放量突破(量比>=1.3)','回踩MA20(缩量收阳)',
     '高收紧(距MA20<3%,回撤<5%)','低波趋势','低波+板块宽度>=60%','早期+低波+市场>=4']
D=EV[EV.事件.isin(sel)].pivot_table(index='事件',columns='期',values='净收益_扣成本').reindex(sel)
fig,ax=plt.subplots(figsize=(11,6))
y=np.arange(len(D)); h=.36
ax.barh(y-h/2,D['训练 2017-2021'],h,label='训练 2017-2021',color='#4C78A8')
ax.barh(y+h/2,D['检验 2022-2026'],h,label='检验 2022-2026',color='#F58518')
ax.set_yticks(y); ax.set_yticklabels(D.index); ax.axvline(0,color='black',lw=.8)
ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.set_xlabel('20 日净收益（扣 0.3% 成本）')
ax.set_title('不同上涨趋势形态的 20 日净收益')
ax.legend(frameon=False); plt.tight_layout(); plt.savefig(os.path.join(PIC,'图6_事件类型.png'),dpi=200); plt.close()
print('图表完成')
