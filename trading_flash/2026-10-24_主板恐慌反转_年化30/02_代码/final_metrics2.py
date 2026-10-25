# NOTE 2026-10-25: 实际价部分已被 actual_engine_fixed.py 修正；
# 旧版实际价 42.6%/236.2万不可用，见 03_结果/实际价_修正说明.md。
import pandas as pd, numpy as np, json
def calc(path,label,cap):
    eq=pd.read_csv(path,parse_dates=['date']); e=eq.set_index('date').equity
    yrs=(e.index[-1]-e.index[0]).days/365.25
    cagr=(e.iloc[-1]/cap)**(1/yrs)-1; dd=(e/e.cummax()-1).min(); r=e.pct_change().dropna()
    ye=e.resample('YE').last(); ann=ye.pct_change(); ann.iloc[0]=ye.iloc[0]/cap-1
    me=e.resample('ME').last(); mon=me.pct_change(); mon.iloc[0]=me.iloc[0]/cap-1
    dds=e/e.cummax()-1
    sp=[]; inset=False
    for d,v in dds.items():
        if v<0 and not inset: start=d; inset=True; low=v; lowd=d
        elif v<0 and inset:
            if v<low: low=v; lowd=d
        elif v>=0 and inset: sp.append(dict(start=start,trough=lowd,recovery=d,depth=low,days=(d-start).days)); inset=False
    if inset: sp.append(dict(start=start,trough=lowd,recovery=None,depth=low,days=(e.index[-1]-start).days))
    return dict(label=label,start=str(e.index[0].date()),end=str(e.index[-1].date()),final=float(e.iloc[-1]),cagr=float(cagr),dd=float(dd),calmar=float(cagr/abs(dd)),sharpe=float(r.mean()/r.std()*np.sqrt(252))),ann,mon,pd.DataFrame(sp)
h,ha,hm,hsp=calc('03_结果/最终_每日权益.csv','hfq',1.0)
a,aa,am,asp=calc('03_结果/实际价_每日权益.csv','actual',100000.0)
tr=pd.read_csv('03_结果/最终_逐笔.csv'); tra=pd.read_csv('03_结果/实际价_逐笔.csv')
M=dict(hfq=h,actual=a,hfq_trades=int(len(tr)),actual_trades=int(len(tra)),hfq_win=float((tr.ret>0).mean()),actual_win=float((tra.ret>0).mean()),hfq_avg=float(tr.ret.mean()),actual_avg=float(tra.ret.mean()),hfq_reasons=tr.exit_reason.value_counts().to_dict(),actual_reasons=tra.exit_reason.value_counts().to_dict(),hfq_top5=float(tr[tr.ret>0].ret.nlargest(5).sum()/tr[tr.ret>0].ret.sum()),actual_top5=float(tra[tra.ret>0].ret.nlargest(5).sum()/tra[tra.ret>0].ret.sum()))
json.dump(M,open('03_结果/最终_指标.json','w'),ensure_ascii=False,indent=1)
def month_labels(idx):
    out=[]
    for d in idx:
        if d.year==2017 and d.month==10: out.append('2017-10（10-19起）')
        elif d.year==2026 and d.month==10: out.append('2026-10（截至10-23）')
        else: out.append(d.strftime('%Y-%m'))
    return out
hm_disp=hm.copy(); hm_disp.index=month_labels(hm.index)
am_disp=am.copy(); am_disp.index=month_labels(am.index)
ha.to_csv('03_结果/最终_年度.csv',header=['return']); hm_disp.to_csv('03_结果/最终_月度.csv',header=['return']); hsp.to_csv('03_结果/最终_回撤区间.csv',index=False)
def year_labels(idx):
    out=[]
    for d in idx:
        if d.year==2017: out.append('2017（10-19起）')
        elif d.year==2026: out.append('2026（截至10-23）')
        else: out.append(str(d.year))
    return out
ha_disp=ha.copy(); ha_disp.index=year_labels(ha.index); ha_disp.to_csv('03_结果/最终_年度_展示.csv',header=['return'])
aa.to_csv('03_结果/实际_年度.csv',header=['return']); am_disp.to_csv('03_结果/实际_月度.csv',header=['return'])
print(json.dumps(M,ensure_ascii=False,indent=1)); print('\nhfq annual'); print(ha.round(4).to_string()); print('\nactual annual'); print(aa.round(4).to_string()); print('\nhfq dd spells'); print(hsp.sort_values('depth').head(6).to_string(index=False)); print('\nhfq worst months'); print(hm.sort_values().head(6).round(4).to_string())
