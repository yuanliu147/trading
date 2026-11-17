#!/usr/bin/env python3
"""10 交互格：8 维交叉中的窄口袋（探索性，非预注册）"""
import os, sys
import numpy as np
import pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
ROOT=os.path.dirname(HERE); RES=os.path.join(ROOT,"03_结果")

def cluster_t(x):
    d=pd.DataFrame({"e":x.ex20,"d":x.date}).dropna()
    s=d.groupby("d").e.sum().to_numpy()
    return d.e.mean()/(np.sqrt(np.sum(s**2))/len(d))

def main():
    e=pd.read_parquet(os.path.join(RES,"事件_金叉_12269.parquet"))
    e=e.dropna(subset=["pct20","ex20"])
    e["零轴"]=np.where(e.zero_state=="双下","双下",np.where(e.zero_state=="双上","双上","混合"))
    e["周线"]=np.where(e.周线状态=="周线空头","空头",np.where(e.周线状态=="周线多头","多头","近金叉"))
    e["密度"]=np.where(e.信号密度==">100",">100",np.where(e.信号密度.isin(["51-100","21-50"]),"21-100","<=20"))
    cells={
     "A 弱市·强股·中波·低额·双下·空头·洗盘": dict(市场趋势="下跌(年线下)",个股趋势="年线上+年线升",波动分组="中波",成交额分组="低额",零轴="双下",周线="空头",密度=">100",is_st=0.0),
     "B 强市·年线降·低波·低额·双上·多头·洗盘": dict(市场趋势="强趋势(年线上+20日涨)",个股趋势="年线上+年线降",波动分组="低波",成交额分组="低额",零轴="双上",周线="多头",密度=">100",is_st=0.0),
     "C 弱市·年线下·低波·低额·双上·近金叉·洗盘": dict(市场趋势="下跌(年线下)",个股趋势="年线下",波动分组="低波",成交额分组="低额",零轴="双上",周线="近金叉",密度=">100",is_st=0.0),
     "D 强市·年线升·低波·低额·双下·空头·洗盘": dict(市场趋势="强趋势(年线上+20日涨)",个股趋势="年线上+年线升",波动分组="低波",成交额分组="低额",零轴="双下",周线="空头",密度=">100",is_st=0.0),
    }
    rows=[]
    for name,cond in cells.items():
        x=e.copy()
        for k,v in cond.items(): x=x[x[k]==v]
        dm=x.groupby("date").ex20.mean()
        rows.append(dict(格子=name, 信号数=len(x), 信号日数=x.date.nunique(),
            每年信号=len(x)/((x.date.max()-x.date.min()).days/365),
            平均分位20=x.pct20.mean(), 平均超额20=x.ex20.mean(), 中位超额20=x.ex20.median(),
            日度平均超额20=dm.mean(), 日度t=dm.mean()/(dm.std()/np.sqrt(len(dm))),
            聚类t=cluster_t(x), 分位_2010_2017=x[x.date<"2018-01-01"].pct20.mean(),
            分位_2018_2026=x[x.date>="2018-01-01"].pct20.mean(),
            超额_2010_2017=x[x.date<"2018-01-01"].ex20.mean(),
            超额_2018_2026=x[x.date>="2018-01-01"].ex20.mean(),
            超额_2023_2026=x[x.date>="2023-01-01"].ex20.mean(),
            MFE20=x.mfe20.mean(), MAE20=x.mae20.mean(),
            日均成交额万元=x.amount20.median()/1e4,
            前5笔占盈亏=x.ex20.nlargest(5).sum()/x.ex20.sum() if x.ex20.sum()>0 else np.nan))
    out=pd.DataFrame(rows)
    out.to_csv(os.path.join(RES,"最终_交互格.csv"),index=False,encoding="utf-8-sig")
    print(out.to_string(index=False))

if __name__=="__main__":
    main()
