# -*- coding: utf-8 -*-
"""2万元账户：三档仓位方案，整手、最低佣金、严谨回撤。"""
import importlib.util,os,pandas as pd,numpy as np
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec=importlib.util.spec_from_file_location('m102',os.path.join(BASE,'02_代码/102_两万组合.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
profiles={
 '稳健':dict(slots=2,sizing='risk',risk=0.02,maxpos=0.5,heat_cap=0.05,gate_thr=0.55,rank='amt',hold=20,start_cash=20000),
 '均衡':dict(slots=2,sizing='risk',risk=0.03,maxpos=0.5,heat_cap=0.08,gate_thr=0.55,rank='amt',hold=20,start_cash=20000),
 '激进':dict(slots=2,sizing='fixed',risk=0,maxpos=0.5,heat_cap=0.10,gate_thr=0.50,rank='amt',hold=20,start_cash=20000),
}
rows=[]; eqs={}
for name,kw in profiles.items():
    for start,end,per in [('2018-01-01','2021-12-31','训练'),('2022-01-01','2026-10-13','检验'),('2018-01-01','2026-10-13','全期')]:
        E,T,met=m.run(start,end,**kw); met.update({'方案':name,'期':per}); rows.append(met)
        if per=='全期':
            eqs[name]=E.copy()
            E.to_csv(os.path.join(BASE,f'03_结果/两万_权益_{name}.csv'),encoding='utf-8-sig')
            T.to_csv(os.path.join(BASE,f'03_结果/两万_逐笔_{name}.csv'),index=False,encoding='utf-8-sig')
out=pd.DataFrame(rows); out.to_csv(os.path.join(BASE,'03_结果/两万_组合对比.csv'),index=False,encoding='utf-8-sig')
print(out[['方案','期','total','cagr','maxdd','maxdd_low','sharpe','trades','win','pf','avg_pos','final']].round(4).to_string(index=False))
# 回撤日期
for name,E in eqs.items():
    s=E.equity; peak=s.cummax(); dd=s/peak-1; tr=dd.idxmin(); pk=s.loc[:tr].idxmax(); rec=s.loc[tr:][s.loc[tr:]>=s.loc[pk]]
    print(name,'peak',pk.date(),'trough',tr.date(),'depth',round(dd.min(),4),'recover',rec.index[0].date() if len(rec) else None)
# 年度
for name,E in eqs.items():
    y=E.equity.resample('YE').last().pct_change(); y.iloc[0]=E.equity.loc[E.index.year==E.index.year[0]].iloc[-1]/E.equity.iloc[0]-1
    print(name,'年度',(y*100).round(1).to_dict())
