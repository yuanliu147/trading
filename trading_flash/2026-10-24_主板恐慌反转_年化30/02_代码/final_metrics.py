import pandas as pd, numpy as np, json
eq=pd.read_csv('03_结果/实际价_每日权益.csv',parse_dates=['date'])
tr=pd.read_csv('03_结果/实际价_逐笔.csv',parse_dates=['signal_date','entry_date','exit_date'])
CAP=100000.0; e=eq.set_index('date').equity
yrs=(e.index[-1]-e.index[0]).days/365.25
cagr=(e.iloc[-1]/CAP)**(1/yrs)-1; dd=(e/e.cummax()-1).min()
r=e.pct_change().dropna(); sharpe=r.mean()/r.std()*np.sqrt(252)
# annual
ye=e.resample('YE').last(); ann=ye.pct_change(); ann.iloc[0]=ye.iloc[0]/CAP-1
# monthly
me=e.resample('ME').last(); mon=me.pct_change(); mon.iloc[0]=me.iloc[0]/CAP-1
# drawdown series
dds=e/e.cummax()-1
# spells
sp=[]; inset=False
for d,v in dds.items():
    if v<0 and not inset: start=d; inset=True; low=v; lowd=d
    elif v<0 and inset:
        if v<low: low=v; lowd=d
    elif v>=0 and inset:
        sp.append(dict(start=start,trough=lowd,recovery=d,depth=low,days=(d-start).days)); inset=False
if inset: sp.append(dict(start=start,trough=lowd,recovery=None,depth=low,days=(e.index[-1]-start).days))
# exposure & yearly
eq['year']=eq.date.dt.year; eq['month']=eq.date.dt.to_period('M')
expo=(eq.npos>0).mean(); yearly_trades=tr.assign(year=tr.entry_date.dt.year).groupby('year').agg(n=('ret','size'),win=('ret',lambda x:(x>0).mean()),avg=('ret','mean'),pnl=('pnl','sum'))
out=dict(capital=CAP,start=str(e.index[0].date()),end=str(e.index[-1].date()),final_equity=float(e.iloc[-1]),cagr=float(cagr),max_drawdown=float(dd),calmar=float(cagr/abs(dd)),sharpe=float(sharpe),trades=int(len(tr)),win_rate=float((tr.ret>0).mean()),avg_trade=float(tr.ret.mean()),median_trade=float(tr.ret.median()),exposure=float(expo),profit_factor=float(tr[tr.pnl>0].pnl.sum()/abs(tr[tr.pnl<0].pnl.sum())),top1_share=float(tr[tr.ret>0].ret.nlargest(1).sum()/tr[tr.ret>0].ret.sum()),top5_share=float(tr[tr.ret>0].ret.nlargest(5).sum()/tr[tr.ret>0].ret.sum()),top10_share=float(tr[tr.ret>0].ret.nlargest(10).sum()/tr[tr.ret>0].ret.sum()),avg_hold=float(tr.entry_date.sub(tr.signal_date).dt.days.mean()+tr.exit_date.sub(tr.entry_date).dt.days.mean()),exit_reasons=tr.exit_reason.value_counts().to_dict(),max_dd_start=str(sp[0]['start'].date()) if sp else None,max_dd_trough=str(min(sp,key=lambda x:x['depth'])['trough'].date()) if sp else None)
json.dump(out,open('03_结果/最终_指标.json','w'),ensure_ascii=False,indent=1)
ann.to_csv('03_结果/最终_年度.csv',header=['return']); mon.to_csv('03_结果/最终_月度.csv',header=['return']); dds.to_csv('03_结果/最终_回撤.csv',header=['drawdown']); pd.DataFrame(sp).to_csv('03_结果/最终_回撤区间.csv',index=False)
yearly_trades.to_csv('03_结果/最终_年度交易.csv')
print(json.dumps(out,ensure_ascii=False,indent=1))
print('\nannual'); print(ann.round(4).to_string())
print('\nmonthly worst'); print(mon.sort_values().head(8).round(4).to_string())
print('\ndd spells'); print(pd.DataFrame(sp).to_string(index=False))
