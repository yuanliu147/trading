#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股性/个股固定效应：交易收益有多少来自“哪只股票”，多少来自“哪次入场”。"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE=Path(__file__).resolve().parent; DATA=HERE/'01_数据'

def icc_oneway(df, group):
    g=df.groupby(group)['净收益率']
    n=g.size(); mean=g.mean(); var=g.var(ddof=1)
    N=n.sum(); k=len(n); grand=df['净收益率'].mean()
    msb=(n*(mean-grand)**2).sum()/(k-1)
    msw=((n-1)*var.fillna(0)).sum()/(N-k)
    n0=(N-(n**2).sum()/N)/(k-1)
    vb=max((msb-msw)/n0,0)
    return vb/(vb+msw) if (vb+msw)>0 else 0, N, k

def perm_icc(df, group, nperm=200, seed=42):
    rng=np.random.default_rng(seed)
    obs,_,_=icc_oneway(df,group)
    vals=[]
    y=df['净收益率'].to_numpy()
    for _ in range(nperm):
        x=df.copy(); x['净收益率']=rng.permutation(y)
        vals.append(icc_oneway(x,group)[0])
    return obs, float(np.mean(np.array(vals)>=obs))

def main():
    tr=pd.read_csv(DATA/'全信号交易_分规则.csv',encoding='utf-8-sig',parse_dates=['信号日'])
    sig=pd.read_csv(DATA/'信号特征_时点500_内部.csv',encoding='utf-8-sig',parse_dates=['signal_date'])
    d=tr[tr.规则=='死叉'].merge(sig,left_on=['代码','信号日'],right_on=['code','signal_date'],how='left')
    d['净收益率']=d['净收益率'].astype(float)
    rows=[]
    for col,label in [('代码','个股'),('type','股性类型'),('subtype','股性子类')]:
        x=d.dropna(subset=[col,'净收益率']).copy()
        obs,p=perm_icc(x,col,nperm=100)
        rows.append({'分组':label,'组数':x[col].nunique(),'样本':len(x),'ICC':obs,'置换p':p})
        print(label,'ICC',round(obs,4),'p',p)
    pd.DataFrame(rows).to_csv(DATA/'股性固定效应_ICC.csv',index=False,encoding='utf-8-sig')

    # 按股票看：交易数>=10的股票，前后半段平均收益的稳定性
    d=d.sort_values('信号日')
    half=d.groupby('代码',group_keys=False).apply(lambda x: pd.Series({
        'n':len(x),'前半':x.iloc[:len(x)//2]['净收益率'].mean(),
        '后半':x.iloc[len(x)//2:]['净收益率'].mean(),
        '全':x['净收益率'].mean()}), include_groups=False)
    h=half[half.n>=10].dropna()
    rho,p=stats.spearmanr(h['前半'],h['后半'])
    print('前后半相关 股票数',len(h),'Spearman',round(rho,3),'p',p)
    h[['n','前半','后半','全']].sort_values('全').to_csv(DATA/'个股前后半收益.csv',encoding='utf-8-sig')
    # 分类型看
    for t,x in d.groupby('type'):
        if len(x)>100:
            print(t,'n',len(x),'每笔均值',round(x['净收益率'].mean()*100,2),'中位',round(x['净收益率'].median()*100,2))

if __name__=='__main__': main()
