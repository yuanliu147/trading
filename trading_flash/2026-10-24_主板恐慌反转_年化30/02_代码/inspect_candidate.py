import pandas as pd, numpy as np, sys
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics, annual
D=load()
eq,tr=simulate2(D,'panic_strong',hold=15,slots=1,score='drawdown250',dyn=False)
e=eq.set_index('date').equity
print('final %.3f cagr %.4f dd %.4f'%(e.iloc[-1],e.iloc[-1]**(365.25/(e.index[-1]-e.index[0]).days)-1,(e/e.cummax()-1).min()))
print('trades',len(tr),'win',round((tr.ret>0).mean(),3),'avg',round(tr.ret.mean(),4),'median',round(tr.ret.median(),4))
cols=['thscode','entry_date','exit_date','entry_px','last_px','ret','days','rk','alloc','exit_reason']
print('\nTop 15:'); print(tr.sort_values('ret',ascending=False)[cols].head(15).to_string(index=False))
print('\nBottom 10:'); print(tr.sort_values('ret')[cols].head(10).to_string(index=False))
pos=tr[tr.ret>0].ret.sort_values(ascending=False)
print('\nprofit concentration top1/3/5/10:',round(pos.head(1).sum()/pos.sum(),3),round(pos.head(3).sum()/pos.sum(),3),round(pos.head(5).sum()/pos.sum(),3),round(pos.head(10).sum()/pos.sum(),3))
tr['year']=pd.to_datetime(tr.entry_date).dt.year
print('\nby year:'); print(tr.groupby('year').agg(n=('ret','size'),mean=('ret','mean'),win=('ret',lambda x:(x>0).mean()),total=('ret','sum')).to_string())
print('\nannual equity:'); print(annual(eq).round(4).to_string())
