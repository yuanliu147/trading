# -*- coding: utf-8 -*-
"""当前时段的基础统计：宽度分桶前瞻、当前状态同态样本、近信号分宽度表现。
数据：全A面板（2016-10-13~2026-10-13）。
"""
import pickle, numpy as np, pandas as pd
P=pickle.load(open('/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl','rb'))
dates=pd.DatetimeIndex(P['meta']['dates']); g=lambda k:P[k].values
m=pd.DataFrame({'mkt':P['market_index'],'mb':P['market_breadth20'],'mb60':P['market_breadth60'],
                'ma20':P['market_ma20'],'ma60':P['market_ma60'],'ma120':P['market_ma120'],
                'slope':P['market_slope60']},index=dates)
for h in [5,10,20,60]: m[f'f{h}']=m.mkt.shift(-h)/m.mkt-1
m['fwdmin20']=m.mkt.iloc[::-1].rolling(20,min_periods=1).min().iloc[::-1]/m.mkt-1
m['r20']=m.mkt/m.mkt.shift(20)-1
m['mb_bucket']=pd.cut(m.mb,[0,.1,.2,.3,.5,.7,1.01],labels=['<10%','10-20%','20-30%','30-50%','50-70%','>70%'])
B=m.groupby('mb_bucket',observed=True).agg(天数=('f20','size'),未来5日=('f5','mean'),未来10日=('f10','mean'),
    未来20日=('f20','mean'),未来60日=('f60','mean'),未来20日胜率=('f20',lambda s:(s>0).mean()),
    未来60日胜率=('f60',lambda s:(s>0).mean()),未来20日路径最低均值=('fwdmin20','mean'),未来20日路径最低最差=('fwdmin20','min'))
B.to_csv('07_宽度分桶_前瞻.csv',encoding='utf-8-sig',float_format='%.4f')
# 当前同态：宽度<20% + 低于三条均线 + 斜率<0
A=m[(m.mb<.2)&(m.mkt<m.ma20)&(m.mkt<m.ma60)&(m.mkt<m.ma120)&(m.slope<0)]
A[['mb','r20','f5','f10','f20','f60','fwdmin20']].to_csv('07_当前同态_样本.csv',encoding='utf-8-sig',float_format='%.4f')
# 近信号（个股条件，不含行业/市场）分宽度
U=g('up_struct').astype(bool); AGE=g('trend_age'); ATR=g('atr_pct'); LU=g('lu_count20'); DD=g('dd60')
AMT=g('amt20'); C=g('close'); SBR=g('sector_breadth20'); SR=g('sector_ret20'); MB=g('market_breadth20')
F20=g('fwd20'); FMIN=g('fwd_min20_entry'); NG=g('next_gap'); NLU=g('next_limit_up').astype(bool)
valid=np.isfinite(F20)&(C>=2)&(AMT>=3e7)&~(np.isfinite(NG)&(NG>0.05))&~NLU
valid[:np.searchsorted(dates,pd.Timestamp('2017-10-13'))]=False
near=valid&U&(AGE>=5)&(AGE<=30)&(ATR<=.044)&(LU==0)&(DD>-.08)
full=near&(AMT<=70e6)&(C<=15)&(SBR>=.6)&(SR>.05)
rows=[]
for lab,sel in [('宽度<10%',MB<.10),('宽度10-20%',(MB>=.10)&(MB<.20)),('宽度20-50%',(MB>=.20)&(MB<.50)),('宽度>50%',MB>=.50)]:
    for nm,mask in [('近信号(个股条件)',near),('70/15完整',full)]:
        x=F20[mask&sel[:,None]]
        if len(x)==0: continue
        rows.append(dict(宽度=lab,口径=nm,样本数=len(x),均值=x.mean(),中位=np.median(x),胜率=(x>0).mean(),
                         盈亏比=x[x>0].sum()/max(1e-9,-x[x<0].sum()),P10=np.quantile(x,.1),
                         路径最低均值=FMIN[mask&sel[:,None]].mean(),路径最低最差=FMIN[mask&sel[:,None]].min()))
pd.DataFrame(rows).to_csv('07_近信号_分宽度.csv',index=False,encoding='utf-8-sig',float_format='%.4f')
print(B.round(4).to_string())
print('\n同态样本 n=',len(A),' 未来20日均值',A.f20.mean(),'胜率',(A.f20>0).mean(),'路径最低均值',A.fwdmin20.mean())
print(pd.DataFrame(rows).round(4).to_string(index=False))
