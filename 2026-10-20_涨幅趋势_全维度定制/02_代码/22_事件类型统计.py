# -*- coding: utf-8 -*-
"""上涨趋势内部的形态事件：突破、回踩、早期、晚期，及组合条件。"""
import pickle, numpy as np, pandas as pd, os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026.pkl'),'rb'))
dates=P['meta']['dates']; C=P['close'].values; F20=P['fwd20'].values
bench=np.nanmean(F20,axis=1)[:,None]; EXC=F20-bench
U=P['up_struct'].values; AGE=P['trend_age'].values; V=P['vr5_20'].values
V20=P['vol20'].values; ATR=P['atr_pct'].values; AMT=P['amt20'].values
DMA20=P['dist_ma20'].values; DD60=P['dd60'].values; ER=P['er20'].values
NH20=P['newhigh20'].values; NH60=P['newhigh60'].values; NH250=P['newhigh250'].values
R1=P['ret1'].values; MA5=P['ma5'].values; MA10=P['ma10'].values; MA20=P['ma20'].values
VR=P['vr5_20'].values; AR=P['amt_ratio20'].values
SBR=P['sector_breadth20'].values; SBR60=P['sector_breadth60'].values; SR20=P['sector_ret20'].values
SW60=P['sw_above60'].values; SWR=P['sw_ret20'].values; SSZ=P['stock_sector_z'].values
MKT=P['regime_id'].values.reshape(-1,1); MB=P['market_breadth20'].values.reshape(-1,1)
MS=P['market_sent'].values.reshape(-1,1); MR20=P['market_ret20'].values.reshape(-1,1)
LU=P['lu_count20'].values; GAPC=P['gap_count20'].values
TURNZ=P['turn_z250'].values
MIN=P['fwd_min20_entry'].values; MAX=P['fwd_max20_entry'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
valid=(U & np.isfinite(F20) & (AMT>=3e7) & (C>=2) & (AGE>=5) & date_ok)
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
lowvol = ATR <= np.nanquantile(ATR[valid],0.4)
lowturn = np.isfinite(TURNZ) & (TURNZ<=0.0)
events={
 '全部趋势日': valid,
 '早期趋势(5-30日)': valid & (AGE<=30),
 '中期趋势(31-90日)': valid & (AGE>30) & (AGE<=90),
 '晚期趋势(>120日)': valid & (AGE>120),
 '突破20日新高': valid & NH20,
 '放量突破(量比>=1.3)': valid & NH20 & (VR>=1.3),
 '缩量突破(量比<=0.9)': valid & NH20 & (VR<=0.9),
 '首次突破60日高(年龄<=30)': valid & NH60 & (AGE<=30),
 '回踩MA20(缩量收阳)': valid & (DMA20>=-0.03) & (DMA20<=0.03) & (DD60<=-0.03) & (DD60>=-0.12) & (VR<=0.9) & (R1>0) & (C>MA5),
 '回踩MA10(缩量收阳)': valid & (C>=MA10*0.985) & (C<=MA10*1.03) & (DD60<=-0.03) & (VR<=0.9) & (R1>0),
 '高收紧(距MA20<3%,回撤<5%)': valid & (DMA20<0.03) & (DMA20>-0.03) & (DD60>-0.05) & (ER>0.3),
 '低波趋势': valid & lowvol,
 '高波趋势': valid & (ATR>=np.nanquantile(ATR[valid],0.6)),
 '无近期涨停': valid & (LU==0),
 '有近期涨停': valid & (LU>0),
 '无跳空异动': valid & (GAPC==0),
 '板块宽度>=60%': valid & (SBR>=0.6),
 '板块指数在MA60上': valid & (SW60>0.5),
 '板块20日动量>5%': valid & (SR20>0.05),
 '市场上行/强上行': valid & (MKT>=4),
 '市场强上行': valid & (MKT==5),
 '市场情绪<0': valid & (MS<0),
 '市场情绪>1': valid & (MS>1),
 '低波+市场上行': valid & lowvol & (MKT>=4),
 '低波+板块宽度>=60%': valid & lowvol & (SBR>=0.6) & (SW60>0.5),
 '低波+板块动量>5%+市场>=4': valid & lowvol & (SR20>0.05) & (MKT>=4),
 '无涨停+低波+市场>=4': valid & lowvol & (LU==0) & (MKT>=4),
 '早期+低波+市场>=4': valid & (AGE<=30) & lowvol & (MKT>=4),
 '回踩+低波+板块宽度>=60%': valid & (DMA20>=-0.03) & (DMA20<=0.03) & (DD60<=-0.03) & (DD60>=-0.12) & (VR<=0.9) & (R1>0) & (C>MA5) & lowvol & (SBR>=0.6),
 '突破+低波+市场>=4': valid & NH20 & lowvol & (MKT>=4),
}
rows=[]
for name,m in events.items():
    for pi,per in [(0,'训练 2017-2021'),(1,'检验 2022-2026')]:
        mm=m & (period[:,None]==pi)
        n=mm.sum()
        if n<100: continue
        e=EXC[mm]; f=F20[mm]; mn=MIN[mm]; mx=MAX[mm]
        rows.append(dict(事件=name,期=per,样本=int(n),超额均值=e.mean(),超额中位=np.median(e),超额胜率=(e>0).mean(),
            绝对均值=f.mean(),绝对中位=np.median(f),绝对胜率=(f>0).mean(),
            平均最大不利=mn.mean(),平均最大有利=mx.mean(),盈亏比=(f[f>0].sum()/max(1e-9,-f[f<0].sum())),
            净收益_扣成本=f.mean()-0.003))
out=pd.DataFrame(rows)
out.to_csv(os.path.join(BASE,'03_结果/上涨趋势_事件类型.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',300); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',100)
print(out[out.期.str.startswith('训练')][['事件','样本','超额均值','超额胜率','绝对均值','绝对胜率','平均最大不利','平均最大有利','盈亏比','净收益_扣成本']].to_string(index=False))
print('\n--- 检验 ---')
print(out[out.期.str.startswith('检验')][['事件','样本','超额均值','超额胜率','绝对均值','绝对胜率','平均最大不利','平均最大有利','盈亏比','净收益_扣成本']].to_string(index=False))
