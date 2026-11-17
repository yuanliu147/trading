#!/usr/bin/env python3
"""07 最终表：指数择时、变体组合、收益分布"""
import os, sys
import numpy as np
import pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import macd_core as M
ROOT=os.path.dirname(HERE); RES=os.path.join(ROOT,"03_结果"); FEE=0.0005

def perf(r,p,lag,fee=FEE):
    p=pd.Series(p).fillna(0).clip(0,1); turn=p.diff().abs().fillna(p.abs())
    rr=p.shift(lag).fillna(0)*r-turn*fee; eq=(1+rr).cumprod()
    return dict(年化=(eq.iloc[-1]**(252/len(rr))-1), 最大回撤=(eq/eq.cummax()-1).min(),
                夏普=rr.mean()/rr.std()*np.sqrt(252), 仓位=p.mean(), 换手=int((turn>0).sum()))

def sig(c,kind="cross",f=12,s=26,g=9):
    dif=c.ewm(span=f,adjust=False).mean()-c.ewm(span=s,adjust=False).mean(); dea=dif.ewm(span=g,adjust=False).mean()
    return {"cross":(dif>dea),"zero":(dif>0),"both":((dif>dea)&(dif>0))}[kind].astype(float)

def index_table():
    df=pd.read_parquet(os.path.join(RES,"指数_日线.parquet")); df["date"]=pd.to_datetime(df["date"])
    idx={n:g.sort_values("date").set_index("date") for n,g in df.groupby("idx_name")}
    rows=[]
    for name in ["CSI300","CSI500","CSI1000","CYB","SHCOMP"]:
        d=idx[name]; r=(d.open.shift(-1)/d.open-1).fillna(0)
        for lab,sel in [("全样本",d.index>=d.index.min()),("2016-2026",d.index.year>=2016)]:
            rr=r[sel]
            rows.append(dict(指数=name,规则="买入持有",区间=lab,**perf(rr,pd.Series(1.0,index=rr.index),0)))
            for kind,kl in [("cross","DIF>DEA"),("zero","DIF>0"),("both","DIF>DEA且DIF>0")]:
                p=sig(d.close,kind)[sel]
                rows.append(dict(指数=name,规则=kl,区间=lab,**perf(rr,p,1)))
    out=pd.DataFrame(rows); out.to_csv(os.path.join(RES,"最终_指数择时.csv"),index=False,encoding="utf-8-sig")
    return out

def ew_table():
    P=M.load_panel(); n,d=P["shape"]
    ret1=np.full((n,d),np.nan,np.float32); ret1[1:]=P["close"][1:]/P["close"][:-1]-1
    ew=np.nanmean(np.where(P["valid"],ret1,np.nan),axis=1); ew=pd.Series(ew,index=P["dates"]).fillna(0); ewi=(1+ew).cumprod()
    rows=[]
    for lab,yr in [("2010-2026",(2010,2026)),("2010-2017",(2010,2017)),("2018-2026",(2018,2026))]:
        sel=(ewi.index.year>=yr[0])&(ewi.index.year<=yr[1])
        r=ew[sel]
        rows.append(dict(指数="全A等权",规则="买入持有",区间=lab,**perf(r,pd.Series(1.0,index=r.index),0)))
        for kind,kl in [("cross","DIF>DEA"),("zero","DIF>0"),("both","DIF>DEA且DIF>0")]:
            p=sig(ewi,kind)[sel]
            rows.append(dict(指数="全A等权",规则=kl,区间=lab,**perf(r,p,2)))
    out=pd.DataFrame(rows); out.to_csv(os.path.join(RES,"最终_指数择时_等权.csv"),index=False,encoding="utf-8-sig")
    return out

def variant_portfolio():
    P=M.load_panel(); dif,dea,hist=M.macd(P); gold,death=M.crosses(dif,dea,P)
    wf=M.weekly_state(P); n,d=P["shape"]
    ret1=np.full((n,d),np.nan,np.float32); ret1[1:]=P["close"][1:]/P["close"][:-1]-1
    mret=np.nanmean(np.where(P["valid"],ret1,np.nan),axis=1)
    h1,h2=M.shift(hist,1),M.shift(hist,2)
    green=P["valid"]&(P["bars"]>=120)&(hist<0)&(hist>h1)&(h1<=h2)
    sigs={"金叉":gold,"绿柱缩短":green,"金叉+零轴下":gold&(dif<0),"金叉+周线空头":gold&(~wf["w_above"]),"金叉+ST":gold&(P["is_st"]==1)}
    rows=[]
    for name,sig in sigs.items():
        cs=np.cumsum(sig.astype(np.int32),axis=0)
        cs1=np.vstack([np.zeros((1,d),np.int32),cs[:-1]])
        csH=np.vstack([np.zeros((21,d),np.int32),cs[:-21]])
        active=(cs1-csH)>0; active&=P["valid"]; nact=np.sum(active,axis=1)
        port=np.nansum(np.where(active,ret1,np.nan),axis=1)/np.maximum(nact,1); ex=port-mret
        ent=np.sum(sig,axis=1).astype(float); cost=0.0015*ent/np.maximum(nact,1); exn=ex-cost
        v=np.isfinite(ex)&(nact>0); vn=np.isfinite(exn)&(nact>0)
        rows.append(dict(变体=name,日均信号=ent.mean(),日均持仓=nact[v].mean(),
                         毛年化=(1+np.nanmean(ex[v]))**250-1,毛t=np.nanmean(ex[v])/(np.nanstd(ex[v])/np.sqrt(v.sum())),
                         净年化=(1+np.nanmean(exn[vn]))**250-1,净t=np.nanmean(exn[vn])/(np.nanstd(exn[vn])/np.sqrt(vn.sum()))))
    out=pd.DataFrame(rows); out.to_csv(os.path.join(RES,"最终_变体组合.csv"),index=False,encoding="utf-8-sig")
    return out

def distribution():
    e=pd.read_parquet(os.path.join(RES,"事件_金叉_12269.parquet"))
    x=e.ex20.dropna(); lo,hi=x.quantile([.01,.99])
    rows=[dict(指标="20日超额",样本=len(x),均值=x.mean(),缩尾均值=x.clip(lo,hi).mean(),
               中位数=x.median(),P10=x.quantile(.1),P25=x.quantile(.25),P75=x.quantile(.75),P90=x.quantile(.9),
               偏度=x.skew(),峰度=x.kurt(),大于0=(x>0).mean(),大于10=(x>.1).mean(),小于负10=(x<-.1).mean())]
    for c in ["mfe20","mae20"]:
        y=e[c].dropna(); rows.append(dict(指标=c,样本=len(y),均值=y.mean(),缩尾均值=y.mean(),中位数=y.median(),
            P10=y.quantile(.1),P25=y.quantile(.25),P75=y.quantile(.75),P90=y.quantile(.9),偏度=y.skew(),峰度=y.kurt(),
            大于0=(y>0).mean(),大于10=np.nan,小于负10=np.nan))
    out=pd.DataFrame(rows); out.to_csv(os.path.join(RES,"最终_分布.csv"),index=False,encoding="utf-8-sig")
    return out

if __name__=="__main__":
    print(index_table().to_string(index=False))
    print("\nEW"); print(ew_table().to_string(index=False))
    print("\nVAR"); print(variant_portfolio().to_string(index=False))
    print("\nDIST"); print(distribution().to_string(index=False))
