# -*- coding: utf-8 -*-
import os,pandas as pd,numpy as np
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
OUT=os.path.join(BASE,'03_结果')
Tr=pd.read_csv(os.path.join(OUT,'权益_含退市_训练.csv'),parse_dates=['date']).set_index('date')
Te=pd.read_csv(os.path.join(OUT,'权益_含退市_检验.csv'),parse_dates=['date']).set_index('date')
_scale=Tr.equity.iloc[-1]; Te['equity']=Te['equity']*_scale; Te['equity_low']=Te['equity_low']*_scale; Te['cash']=Te['cash']*_scale
E=pd.concat([Tr,Te]); E=E[~E.index.duplicated()]
E=E[~E.index.duplicated()]
rows=[]
for col,label,basecol in [('equity','收盘权益','equity'),('equity_low','按当日最低价估值（自身峰值）','equity')]:
    s=E[col]; peak=s.cummax(); dd=s/peak-1
    tr=dd.idxmin(); depth=dd.min(); pk=s.loc[:tr].idxmax()
    after=s.loc[tr:]; rec=after[after>=s.loc[pk]]
    recd=rec.index[0] if len(rec) else None
    rows.append(dict(口径=label,峰值日=str(pk.date()),谷底日=str(tr.date()),最大回撤=depth,
        恢复日=str(recd.date()) if recd is not None else '未恢复',下跌交易日=int(E.index.get_loc(tr)-E.index.get_loc(pk)),
        恢复交易日=int(E.index.get_loc(recd)-E.index.get_loc(tr)) if recd is not None else np.nan))
# 补充：低估值相对收盘峰值
s=E.equity_low; peak=E.equity.cummax(); dd=s/peak-1; tr=dd.idxmin(); rows.append(dict(口径='最低价估值相对收盘峰值',峰值日=str(E.equity.loc[:tr].idxmax().date()),谷底日=str(tr.date()),
    最大回撤=dd.min(),恢复日='',下跌交易日=np.nan,恢复交易日=np.nan))
D=pd.DataFrame(rows); D.to_csv(os.path.join(OUT,'回撤审计.csv'),index=False,encoding='utf-8-sig')
print(D.round(4).to_string(index=False))
# 月度最差
for f,lab in [('权益_含退市_训练.csv','训练'),('权益_含退市_检验.csv','检验')]:
    X=pd.read_csv(os.path.join(OUT,f),parse_dates=['date']).set_index('date')
    m=X.equity.resample('ME').last().pct_change().dropna()
    print(lab,'最差月',round(m.min(),4),str(m.idxmin().date()),'最好月',round(m.max(),4),str(m.idxmax().date()))
