#!/usr/bin/env python3
"""15 周期图表：股票分位、指数 2016+、月线择时"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT="/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度"; RES=os.path.join(ROOT,"03_结果"); FIG=os.path.join(ROOT,"04_图表")
plt.rcParams["font.sans-serif"]=["PingFang SC","Heiti SC","Arial Unicode MS"]; plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["figure.dpi"]=150; plt.rcParams["savefig.dpi"]=150; plt.rcParams["axes.grid"]=True; plt.rcParams["grid.color"]="#eeeeee"
C_POS="#1f8a70"; C_NEG="#c0392b"; C_BLUE="#2c6fbb"; C_GRAY="#95a5a6"; C_ORANGE="#e67e22"

s=pd.read_csv(os.path.join(RES,"最终_周期股票.csv"))
labels=[]; vals=[]
for _,r in s.iterrows():
    lab=f"{r.周期}{r.持有期}"; labels.append(lab); vals.append(r.平均分位)
fig,axes=plt.subplots(1,3,figsize=(15.5,4.6))
ax=axes[0]; colors=[C_POS if v>=0.5 else C_NEG for v in vals]
ax.bar(labels,vals,color=colors,alpha=.85)
ax.axhline(.5,color="#333",lw=1)
for i,v in enumerate(vals): ax.text(i,v+0.0008,f"{v*100:.2f}",ha="center",fontsize=7.5)
ax.set_ylim(0.487,0.512); ax.set_yticks([0.49,0.50,0.51]); ax.set_yticklabels(["49%","50%","51%"]); ax.set_ylabel("平均分位")
ax.set_title("个股：日/周/月都不超过 50.7%",fontsize=10.5,loc="left"); ax.tick_params(axis="x",labelrotation=45,labelsize=8)

idx=pd.read_csv(os.path.join(RES,"最终_周期指数择时.csv"))
idx=idx[(idx.区间=="2016-2026")&(idx.规则.isin(["DIF>DEA","持有"]))]
x=np.arange(3); w=.2
for j,(per,col) in enumerate([("日线",C_BLUE),("周线",C_ORANGE),("月线",C_POS),("持有",C_GRAY)]):
    vals2=[]
    for name in ["CSI300","CSI500","CSI1000"]:
        if per=="持有":
            q=idx[(idx.指数==name)&(idx.周期=="日线")&(idx.规则=="持有")]
        else:
            q=idx[(idx.指数==name)&(idx.周期==per)&(idx.规则=="DIF>DEA")]
        vals2.append(q.年化.iloc[0] if len(q) else np.nan)
    axes[1].bar(x+(j-1.5)*w,vals2,width=w,color=col,label=per,alpha=.9)
axes[1].axhline(0,color="#333",lw=1); axes[1].set_xticks(x); axes[1].set_xticklabels(["沪深300","中证500","中证1000"])
axes[1].yaxis.set_major_formatter(PercentFormatter(1.0,decimals=0)); axes[1].set_ylabel("2016-2026 年化")
axes[1].set_title("指数：月线在沪深300/500最好，周线最差",fontsize=10.5,loc="left"); axes[1].legend(frameon=False,fontsize=8,ncol=4)
for j,(per,col) in enumerate([("日线",C_BLUE),("周线",C_ORANGE),("月线",C_POS),("持有",C_GRAY)]):
    vals2=[]
    for name in ["CSI300","CSI500","CSI1000"]:
        if per=="持有":
            q=idx[(idx.指数==name)&(idx.周期=="日线")&(idx.规则=="持有")]
        else:
            q=idx[(idx.指数==name)&(idx.周期==per)&(idx.规则=="DIF>DEA")]
        vals2.append(q.回撤.iloc[0] if len(q) else np.nan)
    axes[2].bar(x+(j-1.5)*w,vals2,width=w,color=col,label=per,alpha=.9)
axes[2].set_xticks(x); axes[2].set_xticklabels(["沪深300","中证500","中证1000"])
axes[2].yaxis.set_major_formatter(PercentFormatter(1.0,decimals=0)); axes[2].set_ylabel("2016-2026 最大回撤")
axes[2].set_title("月线回撤 -20% ~ -32%，仍显著低于持有",fontsize=10.5,loc="left")
fig.suptitle("周期对比：个股无解；指数月线在沪深300/500 有效，周线最差",fontsize=12,x=0.01,ha="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"图7_周期对比.png"),bbox_inches="tight"); plt.close(fig)

# monthly equity curves
df=pd.read_parquet(os.path.join(RES,"指数_日线.parquet")); df["date"]=pd.to_datetime(df["date"])
idxd={n:g.sort_values("date").set_index("date") for n,g in df.groupby("idx_name")}
fig,axes=plt.subplots(3,2,figsize=(12.5,9.2),sharex="col")
for row,name in enumerate(["CSI300","CSI500","CSI1000"]):
    d=idxd[name]; pid=d.index.to_period("M"); g=d.groupby(pid)
    b=pd.DataFrame({"open":g.open.first(),"close":g.close.last()})
    c=b.close; dif=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); dea=dif.ewm(span=9,adjust=False).mean(); p=(dif>dea).astype(float)
    r=(d.open.shift(-1)/d.open-1).fillna(0)
    pos=pd.Series(pid.map(p.shift(1)),index=d.index).fillna(0)
    turn=pos.diff().abs().fillna(pos.abs()); rr=pos*r-turn*.0005
    eqm=(1+rr).cumprod(); eqb=(1+r).cumprod()
    ax=axes[row,0]; ax.plot(eqb.index,eqb,color=C_GRAY,lw=1.3,label="买入持有"); ax.plot(eqm.index,eqm,color=C_BLUE,lw=1.6,label="月线 MACD")
    ax.set_yscale("log"); ax.set_ylabel("净值（对数）"); ax.set_title(f"{name}：月线 MACD vs 持有",fontsize=10.5,loc="left"); ax.legend(frameon=False,fontsize=8)
    cm=(eqm.iloc[-1]**(252/len(eqm))-1)*100; cb=(eqb.iloc[-1]**(252/len(eqb))-1)*100
    ax.text(0.02,.96,f"MACD 年化 {cm:.1f}% / 回撤 {(eqm/eqm.cummax()-1).min()*100:.0f}%",transform=ax.transAxes,fontsize=8.3,color=C_BLUE,va="top")
    ax.text(0.02,.88,f"持有 年化 {cb:.1f}% / 回撤 {(eqb/eqb.cummax()-1).min()*100:.0f}%",transform=ax.transAxes,fontsize=8.3,color=C_GRAY,va="top")
    ax2=axes[row,1]; ax2.fill_between(eqm.index,(eqm/eqm.cummax()-1)*100,0,color=C_BLUE,alpha=.35); ax2.plot(eqb.index,(eqb/eqb.cummax()-1)*100,color=C_GRAY,lw=1)
    ax2.set_ylabel("回撤（%）"); ax2.set_title("回撤：月线（蓝） vs 持有（灰）",fontsize=10.5,loc="left")
fig.suptitle("月线 MACD（日频路径）：换手 4–16 次，主要价值是躲开 2018/2022 的深回撤",fontsize=12,x=0.01,ha="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"图8_月线择时.png"),bbox_inches="tight"); plt.close(fig)
print("done")
