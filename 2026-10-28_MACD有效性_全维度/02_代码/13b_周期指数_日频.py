#!/usr/bin/env python3
"""13b 周期指数：日频路径重算（先 shift 后截取；月/周信号映射到每日仓位）"""
import os, numpy as np, pandas as pd
ROOT="/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度"; RES=os.path.join(ROOT,"03_结果")
FEE=.0005
df=pd.read_parquet(os.path.join(RES,"指数_日线.parquet")); df["date"]=pd.to_datetime(df["date"])
idx={n:g.sort_values("date").set_index("date") for n,g in df.groupby("idx_name")}
def period_bars(d,freq):
    pid=d.index.to_period("W-FRI") if freq=="W" else d.index.to_period("M")
    g=d.groupby(pid); return pd.DataFrame({"open":g.open.first(),"close":g.close.last()}),pid
def sig(c,kind):
    dif=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); dea=dif.ewm(span=9,adjust=False).mean()
    return {"cross":(dif>dea),"zero":(dif>0),"both":((dif>dea)&(dif>0))}[kind].astype(float)
def daily_pos(d,p,freq):
    if freq=="D": return p.shift(1).fillna(0)
    pid=d.index.to_period("W-FRI") if freq=="W" else d.index.to_period("M")
    return pd.Series(pid.map(p.shift(1)),index=d.index).fillna(0)
def perf(r,p,fee=FEE):
    p=pd.Series(p).fillna(0).clip(0,1); turn=p.diff().abs().fillna(p.abs()); rr=p*r-turn*fee
    eq=(1+rr).cumprod(); return dict(cagr=eq.iloc[-1]**(252/len(rr))-1,mdd=(eq/eq.cummax()-1).min(),
        sharpe=rr.mean()/rr.std()*np.sqrt(252),expo=p.mean(),switches=int((turn>0).sum()))
rows=[]
for name in ["CSI300","CSI500","CSI1000"]:
    d=idx[name]; r=(d.open.shift(-1)/d.open-1).fillna(0)
    for freq,lab in [("D","日线"),("W","周线"),("M","月线")]:
        if freq=="D": b=None
        else: b,pid=period_bars(d,freq)
        c=d.close if freq=="D" else b.close
        for kind,kl in [("cross","DIF>DEA"),("zero","DIF>0"),("both","DIF>DEA且DIF>0")]:
            p=sig(c,kind); pos=daily_pos(d,p,freq)
            for seg in ["2006-2026","2016-2026"]:
                sel=(d.index.year>=2016) if seg=="2016-2026" else (d.index>=d.index.min())
                a=perf(r[sel],pos[sel]); bh=perf(r[sel],pd.Series(1.0,index=d.index[sel]))
                rows.append(dict(指数=name,周期=lab,规则=kl,区间=seg,**a))
                if kind=="cross": rows.append(dict(指数=name,周期=lab,规则="持有",区间=seg,**bh))
out=pd.DataFrame(rows).drop_duplicates()
out=out.rename(columns={"cagr":"年化","mdd":"回撤","sharpe":"夏普","expo":"仓位","switches":"换手"})
out.to_csv(os.path.join(RES,"最终_周期指数择时.csv"),index=False,encoding="utf-8-sig")
print(out[out.规则.isin(["DIF>DEA","持有"])].to_string(index=False))
