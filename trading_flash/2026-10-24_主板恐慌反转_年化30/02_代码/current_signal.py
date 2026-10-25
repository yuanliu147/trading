import pandas as pd, numpy as np, sys
sys.path.insert(0,'02_代码')
from panic_lib import load
D=load(); df=D['df']; base=D['base']
last=df.date.max(); print('last',last)
# market state
row=df[df.date==last].iloc[0]
mkt={k:float(row[k]) for k in ['market_return20','breadth20','breadth60','sentiment_score','limit_up10_share']}
print('market',mkt)
# event components on last date
d=base[base.date==last].copy()
d['cond_dd']=d.drawdown250<-0.45; d['cond_r250']=d.return_250d>0; d['cond_rsi']=d.rsi14<35; d['cond_cl']=d.close_location>0.45
d['cond_all']=d.cond_dd&d.cond_r250&d.cond_rsi&d.cond_cl
print('counts',d[['cond_dd','cond_r250','cond_rsi','cond_cl','cond_all']].sum().to_dict())
cols=['thscode','close','amount20','return_250d','drawdown250','rsi14','close_location','atr20_pct','market_return20','limit_up_10','up_limit_05']
show=d[d.cond_all].sort_values('drawdown250')[cols]
print('\nall stock signals:'); print(show.to_string(index=False))
# rank top2 by drawdown
print('\ntop2 for entry after 2 sessions:'); print(show.head(2).to_string(index=False))
show.head(10).to_csv('03_结果/当前信号_2026-10-23.csv',index=False)
# near signals (market gate open, 3 of 4)
d['ncond']=d[['cond_dd','cond_r250','cond_rsi','cond_cl']].sum(axis=1)
near=d[d.ncond>=3].sort_values(['ncond','drawdown250'],ascending=[False,True])[cols+['ncond']]
near.head(15).to_csv('03_结果/近信号_2026-10-23.csv',index=False)
print('\nnear count',len(near)); print(near.head(10).to_string(index=False))
