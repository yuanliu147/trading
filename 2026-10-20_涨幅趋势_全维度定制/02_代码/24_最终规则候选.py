# -*- coding: utf-8 -*-
"""最终候选规则在训练/检验两期的20日事件统计。"""
import pickle, numpy as np, pandas as pd, os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026.pkl'),'rb'))
dates=P['meta']['dates']; C=P['close'].values; F20=P['fwd20'].values
bench=np.nanmean(F20,axis=1)[:,None]; EXC=F20-bench
U=P['up_struct'].values; AGE=P['trend_age'].values; ATR=P['atr_pct'].values
AMT=P['amt20'].values; DMA20=P['dist_ma20'].values; DD60=P['dd60'].values
LU=P['lu_count20'].values; GAPC=P['gap_count20'].values
SBR=P['sector_breadth20'].values; SR20=P['sector_ret20'].values; SW60=P['sw_above60'].values
MKT=P['regime_id'].values.reshape(-1,1); NG=P['next_gap'].values
MIN=P['fwd_min20_entry'].values; MAX=P['fwd_max20_entry'].values
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=(U & np.isfinite(F20) & (AMT>=3e7) & (C>=2) & (AGE>=5) & date_ok)
tr=valid & (period[:,None]==0)
atr_thr=np.nanquantile(ATR[tr],0.40)
base=valid & (AGE<=30) & (ATR<=atr_thr) & (MKT>=4)
cands={
 '基线': base,
 '基+无涨停': base & (LU==0),
 '基+无涨停+板块宽度60': base & (LU==0) & (SBR>=0.6),
 '基+无涨停+板块动量5': base & (LU==0) & (SR20>0.05),
 '基+无涨停+宽度60+动量5': base & (LU==0) & (SBR>=0.6) & (SR20>0.05),
 '基+无涨停+宽度60+回撤8': base & (LU==0) & (SBR>=0.6) & (DD60>-0.08),
 '基+无涨停+宽度60+动量5+回撤8': base & (LU==0) & (SBR>=0.6) & (SR20>0.05) & (DD60>-0.08),
 '基+无涨停+宽度60+动量5+回撤8+额<2亿': base & (LU==0) & (SBR>=0.6) & (SR20>0.05) & (DD60>-0.08) & (AMT<2e8),
 '基+无涨停+宽度60+板块MA60上': base & (LU==0) & (SBR>=0.6) & (SW60>0.5),
 '基+无涨停+宽度60+距MA20<5%': base & (LU==0) & (SBR>=0.6) & (DMA20<0.05),
 '基+无涨停+宽度60+次日高开<5%': base & (LU==0) & (SBR>=0.6) & (NG<0.05),
 '基+无涨停+宽度60+ATR<4%': base & (LU==0) & (SBR>=0.6) & (ATR<=0.04),
}
rows=[]
for name,m in cands.items():
    for pi,per in [(0,'训练 2017-2021'),(1,'检验 2022-2026')]:
        mm=m & (period[:,None]==pi)
        n=mm.sum()
        if n<3000: continue
        e=EXC[mm]; x=F20[mm]; mn=MIN[mm]; mx=MAX[mm]
        rows.append(dict(规则=name,期=per,样本=int(n),超额均值=e.mean(),超额胜率=(e>0).mean(),
            绝对均值=x.mean(),绝对胜率=(x>0).mean(),平均最大不利=mn.mean(),平均最大有利=mx.mean(),
            盈亏比=x[x>0].sum()/max(1e-9,-x[x<0].sum()),净收益=x.mean()-0.003,
            日均信号=round(n/len(np.unique(dates[(dates>=pd.Timestamp('2017-10-13' if pi==0 else '2022-01-01')) & (dates<=pd.Timestamp('2021-12-31' if pi==0 else '2026-10-13'))])),1)))
out=pd.DataFrame(rows)
out.to_csv(os.path.join(BASE,'03_结果/上涨趋势_最终规则候选.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',300); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',100)
print(out[out.期.str.startswith('训练')].to_string(index=False))
print('\n--- 检验 ---')
print(out[out.期.str.startswith('检验')].to_string(index=False))
print('\nATR阈值',atr_thr)
