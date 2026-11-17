#!/usr/bin/env python3
"""12 多周期：日/周/月状态与金叉的组合"""
import os, sys, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import macd_core as M
import importlib; P11=importlib.import_module('11_周期研究')
ROOT=os.path.dirname(HERE); RES=os.path.join(ROOT,"03_结果")

def cluster_t(e,d):
    x=pd.DataFrame({"e":e,"d":d}).dropna()
    if len(x)<50: return np.nan
    s=x.groupby("d").e.sum().to_numpy(); return x.e.mean()/(np.sqrt(np.sum(s**2))/len(x))

def stats(e,col="pct20",ex="ex20"):
    x=e.dropna(subset=[col,ex]); dm=x.groupby("date")[ex].mean()
    return dict(n=len(x),pct=x[col].mean(),share50=(x[col]>0.5).mean(),ex=x[ex].mean(),
                med=x[ex].median(),day=dm.mean(),t_day=dm.mean()/(dm.std()/np.sqrt(len(dm))) if len(dm)>5 else np.nan,
                t_cl=cluster_t(x[ex],x.date),
                early=x[x.date<"2018-01-01"][col].mean(),late=x[x.date>="2018-01-01"][col].mean())

def main():
    P=M.load_panel(); dates=P["dates"]; codes=P["codes"]
    dpos={pd.Timestamp(d):i for i,d in enumerate(dates)}; cpos={c:i for i,c in enumerate(codes)}
    # monthly state
    perM=P11.build_period(P,"M")
    difM,deaM,_,_,_,_=P11.period_signals(P,perM,12,26,9,36)
    stateM=(difM>deaM)
    mend=dates[perM["end"]]
    # previous completed month for each daily date
    posM=np.searchsorted(mend.values,dates.values,side="left")-1
    m_on_daily=np.full(P["shape"],np.nan,np.float32)
    for t in range(len(dates)):
        p=posM[t]
        if p>=0: m_on_daily[t,:]=stateM[p,:]
    # weekly state on daily already in wf; monthly state on week end
    wf=M.weekly_state(P)
    # ---------- daily subsets ----------
    e=pd.read_parquet(os.path.join(RES,"事件_金叉_12269.parquet"))
    e["di"]=e.date.map(dpos); e["ci"]=e.code.map(cpos)
    e=e.dropna(subset=["di","ci"]); e["di"]=e.di.astype(int); e["ci"]=e.ci.astype(int)
    e["m_above"]=m_on_daily[e.di.values,e.ci.values]
    e["w_above2"]=wf["w_above"][e.di.values,e.ci.values]
    rows=[]
    subsets={
        "日线金叉(基准)": np.ones(len(e),bool),
        "+周线多头": e.w_above2,
        "+月线多头": e.m_above==1,
        "+周线多头+月线多头": (e.w_above2)&(e.m_above==1),
        "+周线空头+月线空头": (~e.w_above2)&(e.m_above==0),
        "+周线多头+月线空头": (e.w_above2)&(e.m_above==0),
        "+周线空头+月线多头": (~e.w_above2)&(e.m_above==1),
    }
    for name,m in subsets.items():
        x=e[m]
        st=stats(x); st.update(组合=name,信号数=len(x)); rows.append(st)
    d=pd.DataFrame(rows); d.to_csv(os.path.join(RES,"多周期_股票_日线.csv"),index=False,encoding="utf-8-sig")
    print("\n=== 日线金叉 + 周/月状态 ===")
    print(d[["组合","信号数","pct","share50","ex","med","t_cl","early","late"]].to_string(index=False))

    # ---------- weekly subsets ----------
    perW=P11.build_period(P,"W"); difW,deaW,_,goldW,_,_=P11.period_signals(P,perW,12,26,9,60)
    retsW,okW,mktW,looW,pctW=P11.period_returns(P,perW,[4,12])
    # monthly state at week end
    wend=dates[perW["end"]]
    pM=np.searchsorted(mend.values,wend.values,side="left")-1
    m_on_week=np.full((perW["nper"],len(codes)),np.nan,np.float32)
    for p in range(perW["nper"]):
        if pM[p]>=0: m_on_week[p,:]=stateM[pM[p],:]
    rows=[]
    for H in [4,12]:
        ii,jj=np.where(goldW); sd=dates[perW["end"][ii]]; keep=sd>=pd.Timestamp("2010-01-01")
        ii,jj=ii[keep],jj[keep]; sd=sd[keep]
        r=retsW[H][ii,jj]; m=looW[H][ii,jj]; v=okW[H][ii,jj]&np.isfinite(r)&np.isfinite(m)
        base=pd.DataFrame({"date":sd,"pct":pctW[H][ii,jj],"ex":r-m,"m":m_on_week[ii,jj],
                           "valid":v})
        for name,mask in [("周线金叉(基准)",base.valid),
                          ("+月线多头",base.valid&(base.m==1)),
                          ("+月线空头",base.valid&(base.m==0))]:
            x=base[mask]
            st=stats(x,col="pct",ex="ex"); st.update(组合=name,持有周=H,信号数=len(x)); rows.append(st)
    w=pd.DataFrame(rows); w.to_csv(os.path.join(RES,"多周期_股票_周线.csv"),index=False,encoding="utf-8-sig")
    print("\n=== 周线金叉 + 月线状态 ===")
    print(w[["组合","持有周","信号数","pct","share50","ex","med","t_cl","early","late"]].to_string(index=False))

    # ---------- monthly subsets ----------
    difM2,deaM2,_,goldM,_,_=P11.period_signals(P,perM,12,26,9,36)
    retsM,okM,mktM,looM,pctM=P11.period_returns(P,perM,[3,12])
    rows=[]
    for H in [3,12]:
        ii,jj=np.where(goldM); sd=dates[perM["end"][ii]]; keep=sd>=pd.Timestamp("2010-01-01")
        ii,jj=ii[keep],jj[keep]; sd=sd[keep]
        r=retsM[H][ii,jj]; m=looM[H][ii,jj]; v=okM[H][ii,jj]&np.isfinite(r)&np.isfinite(m)
        # weekly state at month end (previous completed week)
        wstate=wf["w_above"][perM["end"][ii],jj]
        base=pd.DataFrame({"date":sd,"pct":pctM[H][ii,jj],"ex":r-m,"w":wstate,"valid":v})
        for name,mask in [("月线金叉(基准)",base.valid),
                          ("+周线多头",base.valid&(base.w==1)),
                          ("+周线空头",base.valid&(base.w==0))]:
            x=base[mask]
            st=stats(x,col="pct",ex="ex"); st.update(组合=name,持有月=H,信号数=len(x)); rows.append(st)
    mo=pd.DataFrame(rows); mo.to_csv(os.path.join(RES,"多周期_股票_月线.csv"),index=False,encoding="utf-8-sig")
    print("\n=== 月线金叉 + 周线状态 ===")
    print(mo[["组合","持有月","信号数","pct","share50","ex","med","t_cl","early","late"]].to_string(index=False))

if __name__=="__main__": main()
