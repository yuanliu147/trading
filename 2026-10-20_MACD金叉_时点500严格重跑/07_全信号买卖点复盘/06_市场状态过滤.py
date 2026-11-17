#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 100 槽分散组合上，测试中证 500 均线状态过滤。"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent; DATA=HERE/'01_数据'; OUT=HERE/'03_真实回测'
spec=importlib.util.spec_from_file_location('bt10', HERE/'04_真实回测.py')
bt=importlib.util.module_from_spec(spec); sys.modules[spec.name]=bt; spec.loader.exec_module(bt)

def main():
    data=bt.eng.load_pit_data()
    bt.eng.START, bt.eng.END = bt.FULL_START, bt.FULL_END
    bench=bt.eng.load_benchmark(); dates=[d for d in pd.to_datetime(bench['date']) if bt.FULL_START<=d<=bt.FULL_END]
    uni=bt.eng.Universe(bt.eng.load_snapshots())
    sig=pd.read_csv(DATA/'信号特征_时点500_内部.csv', parse_dates=['signal_date'])
    s=sig[sig.in_universe==1].copy()
    F1=(s.ret20<=.15)&(s.dist_ma20<=.08)&(s.rsi14<=70)&(s.n_golden_60>=3)&(s.s_vol<=.72)&(s.amt5_60<=.26)&(s.type!='高波活跃型')
    F2=(s.ret20<=.15)&(s.dist_ma20<=.08)&(s.rsi14<=70)&(s.n_golden_60>=3)&(s.s_vol<=.72)&(s.amt5_60<=.26)&(s.dist_ma60>=-.08)
    masks={
      'F1_b3_指数MA20上': (F1&(s.idx_above_ma20==1),'b3_prior_peak'),
      'F1_b3_指数MA60上': (F1&(s.idx_above_ma60==1),'b3_prior_peak'),
      'F2_死叉_指数MA20上': (F2&(s.idx_above_ma20==1),'death_cross'),
      'F2_死叉_指数MA60上': (F2&(s.idx_above_ma60==1),'death_cross'),
    }
    rows=[]
    for name,(m,strat) in masks.items():
        allowed=set(zip(s.loc[m,'code'],s.loc[m,'signal_date']))
        print(f'\n{name}: {len(allowed):,} signals',flush=True)
        r=bt.run_one(name,strat,allowed,bench,dates,data,uni,100,.01)
        for label,start in [('全期',bt.FULL_START),('2018起',bt.SUB_START)]:
            pm=bt.period_metrics(r['equity_curve'],start,bt.FULL_END)
            tr=r['trades'].copy(); tr['entry_date']=pd.to_datetime(tr['entry_date']); tr=tr[tr.entry_date>=start]
            ts=bt.trade_stats(tr)
            rows.append({'配置':name,'区间':label,**{k:round(v,3) if isinstance(v,(int,float)) else v for k,v in {**pm,**ts}.items()}})
        eq=r['equity_curve'].copy(); eq.columns=['日期','权益','持仓数','持仓市值','仓位比例']
        eq.to_csv(OUT/f'权益_{name}.csv',index=False,encoding='utf-8-sig')
    df=pd.DataFrame(rows); df.to_csv(OUT/'组合汇总_市场过滤.csv',index=False,encoding='utf-8-sig')
    cols=['配置','区间','年化','最大回撤','Calmar','Sharpe','交易','胜率','平均每笔','PF','平均仓位']
    print(df[cols].round(2).to_string(index=False))

if __name__=='__main__': main()
