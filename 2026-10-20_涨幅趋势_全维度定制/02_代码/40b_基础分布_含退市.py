# -*- coding: utf-8 -*-
"""A股涨跌幅基础分布、上涨趋势episode、行业/行情分组。"""
import pickle, numpy as np, pandas as pd, os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
OUT=os.path.join(BASE,'03_结果'); os.makedirs(OUT,exist_ok=True)
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; industries=P['meta']['industries']
R={h:P[f'ret{h}'].values for h in [1,5,20,60,120,250]}
C=P['close'].values; U=P['up_struct'].values; AGE=P['trend_age'].values
ATR=P['atr_pct'].values; AMT=P['amt20'].values; ATRR=P['amt_ratio20'].values
IND=P['industry_id'].values; R1=P['ret1'].values
YR=np.array([d.year for d in dates]); date_ok=dates>=pd.Timestamp('2017-10-13')

# ---------- 1. 全A横截面日度 ----------
rows=[]
for t in range(len(dates)):
    if not date_ok[t]: continue
    rec={'date':dates[t]}
    for h in [1,5,20,60]:
        x=R[h][t]; x=x[np.isfinite(x)]
        if len(x)==0: continue
        rec[f'n{h}']=len(x); rec[f'mean{h}']=x.mean(); rec[f'med{h}']=np.median(x)
        rec[f'win{h}']=(x>0).mean(); rec[f'p10_{h}']=np.quantile(x,.1); rec[f'p90_{h}']=np.quantile(x,.9)
        if h==1:
            up=x[x>0]; dn=x[x<0]
            rec['mean_up']=up.mean() if len(up) else np.nan; rec['mean_dn']=dn.mean() if len(dn) else np.nan
            rec['updn_ratio']=abs(up.mean()/dn.mean()) if len(dn) and dn.mean()!=0 else np.nan
            rec['limitup']=np.mean(x>=0.095); rec['limitdown']=np.mean(x<=-0.095)
    rows.append(rec)
D=pd.DataFrame(rows)
D.to_csv(os.path.join(OUT,'A股日度横截面.csv'),index=False,encoding='utf-8-sig')

# 年度表
D['year']=D.date.dt.year
ann=[]
for y,g in D.groupby('year'):
    rec={'年份':y,'交易日':len(g)}
    for h in [1,5,20,60]:
        rec[f'{h}日均值']=g[f'mean{h}'].mean(); rec[f'{h}日中位']=g[f'med{h}'].mean(); rec[f'{h}日胜率']=g[f'win{h}'].mean()
    rec['平均涨幅']=g.mean_up.mean(); rec['平均跌幅']=g.mean_dn.mean(); rec['涨跌比']=g.updn_ratio.mean()
    rec['涨停占比']=g.limitup.mean(); rec['跌停占比']=g.limitdown.mean()
    # 当年个股收益中位数（用年末最后交易日 ret250 或 ret1 累计）
    t_end=np.where((dates.year==y))[0][-1]
    x=C[t_end]/C[np.where(dates.year==y)[0][0]]-1
    x=x[np.isfinite(x)]
    rec['等权年度收益']=np.nanmean(x); rec['个股年度中位']=np.nanmedian(x); rec['正收益占比']=(x>0).mean()
    ann.append(rec)
ANN=pd.DataFrame(ann); ANN.to_csv(os.path.join(OUT,'A股年度分布.csv'),index=False,encoding='utf-8-sig')

# 全样本分布
dist=[]
for h in [1,5,20,60]:
    x=R[h][date_ok[:,None] & np.isfinite(R[h])]
    dist.append(dict(期限=f'{h}日',样本=len(x),均值=x.mean(),中位=np.median(x),标准差=x.std(),
        偏度=pd.Series(x).skew(),p1=np.quantile(x,.01),p5=np.quantile(x,.05),p10=np.quantile(x,.1),
        p25=np.quantile(x,.25),p75=np.quantile(x,.75),p90=np.quantile(x,.9),p95=np.quantile(x,.95),p99=np.quantile(x,.99),
        胜率=(x>0).mean(),大涨10=(x>0.10).mean(),大涨20=(x>0.20).mean(),大跌10=(x<-0.10).mean(),大跌20=(x<-0.20).mean()))
DIST=pd.DataFrame(dist); DIST.to_csv(os.path.join(OUT,'A股收益分布_全样本.csv'),index=False,encoding='utf-8-sig')

# 成交额分位（规模/流动性代理）
amt_q=np.full_like(R[20],np.nan)
valid_d=date_ok[:,None] & np.isfinite(R[20]) & np.isfinite(AMT)
# 每日横截面分位
for t in range(len(dates)):
    if not date_ok[t]: continue
    x=AMT[t]; m=np.isfinite(x)&np.isfinite(R[20][t])
    if m.sum()<50: continue
    q=pd.qcut(x[m],5,labels=False,duplicates='drop')
    amt_q[t,m]=q
qrows=[]
for q in range(5):
    m=(amt_q==q)&np.isfinite(R[20])
    for y in [None,2022]:
        pass
    x=R[20][m]
    qrows.append({'成交额组':q+1,'样本':len(x),'20日均值':x.mean(),'20日中位':np.median(x),'20日胜率':(x>0).mean(),'20日p10':np.quantile(x,.1),'20日p90':np.quantile(x,.9)})
QT=pd.DataFrame(qrows); QT.to_csv(os.path.join(OUT,'A股_成交额分组.csv'),index=False,encoding='utf-8-sig')

# 行业分组
indrows=[]
for k,ind in enumerate(industries,start=1):
    m=(IND==k)&np.isfinite(R[20])&date_ok[:,None]
    x=R[20][m]
    if len(x)<1000: continue
    indrows.append({'行业':ind,'样本':len(x),'20日均值':x.mean(),'20日中位':np.median(x),'20日胜率':(x>0).mean(),'20日p10':np.quantile(x,.1),'20日p90':np.quantile(x,.9)})
IND_T=pd.DataFrame(indrows).sort_values('20日中位',ascending=False); IND_T.to_csv(os.path.join(OUT,'A股_行业分组.csv'),index=False,encoding='utf-8-sig')

# ---------- 2. 上涨趋势episode ----------
ep_rows=[]
for j in range(len(codes)):
    u=U[:,j].astype(int)
    if u.sum()==0: continue
    # 连续段
    d=np.diff(np.r_[0,u,0]); starts=np.where(d==1)[0]; ends=np.where(d==-1)[0]-1
    for s,e in zip(starts,ends):
        if not date_ok[s]: continue
        dur=e-s+1
        if dur<5: continue
        seg=C[s:e+1,j]; seg=seg[np.isfinite(seg)]
        if len(seg)<5: continue
        gain=seg[-1]/seg[0]-1
        dd=np.min(seg/np.maximum.accumulate(seg)-1)
        atr=np.nanmean(ATR[s:e+1,j]); vr=np.nanmean(ATRR[s:e+1,j])
        ep_rows.append(dict(code=codes[j],start=dates[s],end=dates[e],dur=dur,gain=gain,maxdd=dd,
            atr=atr,amt_ratio=vr,industry=industries[IND[s,j]-1] if IND[s,j]>0 else '未知',start_year=dates[s].year))
EP=pd.DataFrame(ep_rows); EP.to_csv(os.path.join(OUT,'上涨趋势_episode明细.csv'),index=False,encoding='utf-8-sig')
if len(EP):
    summ=EP.groupby('start_year').agg(数量=('code','size'),中位持续=('dur','median'),平均持续=('dur','mean'),
        中位涨幅=('gain','median'),平均涨幅=('gain','mean'),涨幅胜率=('gain',lambda s:(s>0).mean()),
        中位回撤=('maxdd','median'),中位波动=('atr','median'),中位量比=('amt_ratio','median'))
    summ.to_csv(os.path.join(OUT,'上涨趋势_episode年度.csv'),encoding='utf-8-sig')
    print('episodes',len(EP)); print(summ.round(4).to_string())
    print('duration quantiles',EP.dur.quantile([.1,.25,.5,.75,.9,.95,.99]).round(1).to_dict())
    print('gain quantiles',EP.gain.quantile([.1,.25,.5,.75,.9,.95,.99]).round(3).to_dict())

# 存活曲线：episode 年龄 -> 继续概率
maxage=250
surv=[]
for a in range(1,maxage+1):
    at_risk=0; cont=0
    for j in range(len(codes)):
        pass
    # 用 AGE 矩阵近似：某日 age>=a 的 episode 数 / 起始数
    # 直接由 EP 计算
    at_risk=(EP.dur>=a).sum(); cont=(EP.dur>a).sum()
    if at_risk>0: surv.append(dict(年龄=a,仍在趋势=at_risk,下一日仍在=cont,继续概率=cont/at_risk))
SURV=pd.DataFrame(surv); SURV.to_csv(os.path.join(OUT,'上涨趋势_存活曲线.csv'),index=False,encoding='utf-8-sig')

# 年龄阶段的前瞻20日
agebins=[5,10,20,30,60,90,120,180,250,10000]
labels=['5-10','11-20','21-30','31-60','61-90','91-120','121-180','181-250','>250']
m=U&np.isfinite(P['fwd20'].values)&date_ok[:,None]&(AMT>=3e7)
arows=[]
for lo,hi,lab in zip(agebins[:-1],agebins[1:],labels):
    mm=m&(AGE>=lo)&(AGE<hi)
    if mm.sum()<100: continue
    x=P['fwd20'].values[mm]; e=P['excess20'].values[mm]
    arows.append({'年龄':lab,'样本':int(mm.sum()),'20日均值':x.mean(),'20日中位':np.median(x),'20日胜率':(x>0).mean(),
        '超额均值':e.mean(),'平均最大不利':P['fwd_min20_entry'].values[mm].mean(),'平均最大有利':P['fwd_max20_entry'].values[mm].mean()})
AT=pd.DataFrame(arows); AT.to_csv(os.path.join(OUT,'上涨趋势_年龄阶段.csv'),index=False,encoding='utf-8-sig')
print('\n年龄阶段'); print(AT.round(4).to_string(index=False))
print('\n年度'); print(ANN.round(4).to_string(index=False))
print('\n分布'); print(DIST.round(4).to_string(index=False))
