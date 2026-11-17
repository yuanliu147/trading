#!/usr/bin/env python3
"""13 周期指数表：日/周/月 MACD 择时，BH lag0、MACD lag1，统一口径"""
import os, numpy as np, pandas as pd
ROOT="/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度"; RES=os.path.join(ROOT,"03_结果")
FEE=.0005
df=pd.read_parquet(os.path.join(RES,"指数_日线.parquet")); df["date"]=pd.to_datetime(df["date"])
idx={n:g.sort_values("date").set_index("date") for n,g in df.groupby("idx_name")}
def bars(d,freq):
    if freq=="D": return d[["open","close"]]
    pid=d.index.to_period("W-FRI") if freq=="W" else d.index.to_period("M")
    g=d.groupby(pid); return pd.DataFrame({"open":g.open.first(),"close":g.close.last()})
def perf(r,p,ppy,fee=FEE,lag=1):
    p=pd.Series(p).fillna(0).clip(0,1); turn=p.diff().abs().fillna(p.abs()); rr=p.shift(lag).fillna(0)*r-turn*fee
    eq=(1+rr).cumprod(); cagr=eq.iloc[-1]**(ppy/len(rr))-1; mdd=(eq/eq.cummax()-1).min()
    sh=rr.mean()/rr.std()*np.sqrt(ppy) if rr.std()>0 else np.nan
    return cagr,mdd,sh,p.mean(),int((turn>0).sum())
def sig(c,kind):
    dif=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); dea=dif.ewm(span=9,adjust=False).mean()
    return {"cross":(dif>dea),"zero":(dif>0),"both":((dif>dea)&(dif>0))}[kind].astype(float)
rows=[]
for name in ["CSI300","CSI500","CSI1000"]:
    d=idx[name]
    for freq,lab,ppy in [("D","日线",252),("W","周线",52),("M","月线",12)]:
        b=bars(d,freq); r=(b.open.shift(-1)/b.open-1).fillna(0)
        for kind,kl in [("cross","DIF>DEA"),("zero","DIF>0"),("both","DIF>DEA且DIF>0")]:
            p=sig(b.close,kind)
            for seg in ["2006-2026","2016-2026"]:
                sel=(b.index.year>=2016) if seg=="2016-2026" else (b.index>=b.index.min())
                a=perf(r[sel],p[sel],ppy,lag=1); bh=perf(r[sel],pd.Series(1.0,index=r[sel].index),ppy,lag=0)
                rows.append(dict(指数=name,周期=lab,规则=kl,区间=seg,年化=a[0],回撤=a[1],夏普=a[2],仓位=a[3],换手=a[4]))
                if kind=="cross":
                    rows.append(dict(指数=name,周期=lab,规则="持有",区间=seg,年化=bh[0],回撤=bh[1],夏普=bh[2],仓位=bh[3],换手=bh[4]))
out=pd.DataFrame(rows).drop_duplicates()
out.to_csv(os.path.join(RES,"最终_周期指数择时.csv"),index=False,encoding="utf-8-sig")
print(out[out.规则.isin(["DIF>DEA","持有"])].to_string(index=False))
