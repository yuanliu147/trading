# -*- coding: utf-8 -*-
"""在训练期上做小规模规则组合，检验期只读一次；输出基线与逐项加过滤器。"""
import pickle, numpy as np, pandas as pd, os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026.pkl'),'rb'))
dates=P['meta']['dates']; C=P['close'].values; F20=P['fwd20'].values
bench=np.nanmean(F20,axis=1)[:,None]; EXC=F20-bench
U=P['up_struct'].values; AGE=P['trend_age'].values; ATR=P['atr_pct'].values
AMT=P['amt20'].values; DMA20=P['dist_ma20'].values; DD60=P['dd60'].values
LU=P['lu_count20'].values; GAPC=P['gap_count20'].values; VR=P['vr5_20'].values
SBR=P['sector_breadth20'].values; SBR60=P['sector_breadth60'].values; SR20=P['sector_ret20'].values
SW60=P['sw_above60'].values; SSZ=P['stock_sector_z'].values
MKT=P['regime_id'].values.reshape(-1,1); MB=P['market_breadth20'].values.reshape(-1,1)
MR20=P['market_ret20'].values.reshape(-1,1); MS=P['market_sent'].values.reshape(-1,1)
MIN=P['fwd_min20_entry'].values; MAX=P['fwd_max20_entry'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U & np.isfinite(F20) & (AMT>=3e7) & (C>=2) & (AGE>=5) & date_ok)
# 训练期阈值（只用训练期）
tr=valid & (period[:,None]==0)
atr_thr=np.nanquantile(ATR[tr],0.40); vol_thr=np.nanquantile(P['vol20'].values[tr],0.40)
print('训练期阈值 ATR40=',atr_thr,'VOL40=',vol_thr)
base=valid & (AGE<=30) & (ATR<=atr_thr) & (MKT>=4)
filters={
 '基线': np.ones_like(base),
 '+无近期涨停': LU==0,
 '+无跳空异动': GAPC==0,
 '+距MA20<3%': DMA20<0.03,
 '+回撤<8%': DD60>-0.08,
 '+板块宽度>=50%': SBR>=0.5,
 '+板块宽度>=60%': SBR>=0.6,
 '+板块指数在MA60上': SW60>0.5,
 '+板块20日动量>0': SR20>0,
 '+板块20日动量>5%': SR20>0.05,
 '+行业内相对强度0-1': (SSZ>=0)&(SSZ<=1),
 '+量比<1.3': VR<1.3,
 '+市场宽度>50%': MB[:,0]>0.5,
 '+市场20日动量>2%': MR20[:,0]>0.02,
 '+市场情绪<1': MS[:,0]<1,
 '+成交额<2亿': AMT<2e8,
 '+成交额1-8亿': (AMT>=5e7)&(AMT<8e8),
}
rows=[]
for name,f in filters.items():
    if f.ndim==1: f=np.broadcast_to(f[:,None],base.shape)
    m=base & f
    for pi,per in [(0,'训练 2017-2021'),(1,'检验 2022-2026')]:
        mm=m & (period[:,None]==pi)
        n=mm.sum()
        if n<5000: continue
        e=EXC[mm]; x=F20[mm]; mn=MIN[mm]; mx=MAX[mm]
        rows.append(dict(规则=name,期=per,样本=int(n),超额均值=e.mean(),超额胜率=(e>0).mean(),
            绝对均值=x.mean(),绝对胜率=(x>0).mean(),平均最大不利=mn.mean(),平均最大有利=mx.mean(),
            盈亏比=x[x>0].sum()/max(1e-9,-x[x<0].sum()),净收益=x.mean()-0.003))
out=pd.DataFrame(rows)
out.to_csv(os.path.join(BASE,'03_结果/上涨趋势_规则组合网格.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',300); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',100)
print(out[out.期.str.startswith('训练')].to_string(index=False))
print('\n--- 检验 ---')
print(out[out.期.str.startswith('检验')].to_string(index=False))
