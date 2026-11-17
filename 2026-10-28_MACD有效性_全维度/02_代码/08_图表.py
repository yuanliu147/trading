#!/usr/bin/env python3
"""08 图表：条件分位、参数热图、指数择时、年度、分布、参数分布"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import macd_core as M
ROOT=os.path.dirname(HERE); RES=os.path.join(ROOT,"03_结果"); FIG=os.path.join(ROOT,"04_图表")
os.makedirs(FIG,exist_ok=True)
plt.rcParams["font.sans-serif"]=["PingFang SC","Heiti SC","Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["figure.dpi"]=150
plt.rcParams["savefig.dpi"]=150
plt.rcParams["axes.edgecolor"]="#cccccc"
plt.rcParams["axes.grid"]=True
plt.rcParams["grid.color"]="#eeeeee"
plt.rcParams["grid.linewidth"]=0.8
C_POS="#1f8a70"; C_NEG="#c0392b"; C_NEU="#7f8c8d"; C_BLUE="#2c6fbb"; C_ORANGE="#e67e22"

def fig_conditions():
    c=pd.read_csv(os.path.join(RES,"最终_条件.csv"))
    # 选关键条件，避免重复维度
    wanted=[
        ("is_st","1.0","ST股"),("成交额分组","低额","成交额低"),("波动分组","低波","低波动"),
        ("量比","<0.7","量比<0.7"),("个股趋势","年线下","年线下方"),
        ("周线状态","周线空头","周线空头"),("zero_state","双下","零轴双下"),
        ("信号密度",">100","信号密度>100"),("DIF位置","<0","DIF<0"),
        ("距250日高","<-30%","距高<-30%"),("绿柱长度","21-60","绿柱21-60"),
        ("board","主板","主板"),("市场趋势","强趋势(年线上+20日涨)","强趋势"),
        ("个股趋势","年线上+年线升","年线上+年线升"),("zero_state","双上","零轴双上"),
        ("周线状态","周线多头","周线多头"),("量比",">2","量比>2"),("成交额分组","高额","成交额高"),
        ("波动分组","高波","高波动"),("股价年龄","<1年","上市<1年"),
    ]
    rows=[]
    for dim,grp,label in wanted:
        x=c[(c.维度==dim)&(c.分组==grp)]
        if len(x): rows.append(dict(label=label,pct=x.平均分位20.iloc[0],n=x.信号数.iloc[0],
                                    ex=x.平均超额20.iloc[0],t=x.聚类t.iloc[0],late=x.分位_2018_2026.iloc[0]))
    d=pd.DataFrame(rows).sort_values("pct")
    fig,ax=plt.subplots(figsize=(9.2,7.4))
    colors=[C_POS if v>0.5 else C_NEG for v in d.pct]
    bars=ax.barh(d.label,d.pct-0.5,left=0.5,color=colors,alpha=0.85,height=0.68)
    ax.axvline(0.5,color="#333",lw=1.2)
    for i,(v,n) in enumerate(zip(d.pct,d.n)):
        ax.text(0.549,i,f"{v*100:.2f}%  n={n/1000:.0f}k",va="center",
                ha="right",fontsize=8.5,color="#222")
    ax.set_xlim(0.455,0.552)
    ax.axvline(0.535,color="#dddddd",lw=0.8)
    ax.set_xlabel("20 日收益的当日横截面平均分位（50%=与市场同步）")
    ax.set_title("MACD 金叉：哪些条件下跑赢/跑输等权市场\n2010-2026，T+1 开盘买入，横截面分位对右偏更稳健",fontsize=12,loc="left")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0,decimals=0))
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"图1_条件分位.png"),bbox_inches="tight"); plt.close(fig)

def fig_heatmap():
    g=pd.read_csv(os.path.join(RES,"参数网格_含q.csv"))
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.6))
    for ax,sig in zip(axes,[9,20]):
        sub=g[g.signal==sig].pivot_table(index="fast",columns="slow",values="pct20")
        im=ax.imshow((sub.values-0.5)*100,cmap="RdYlGn",vmin=-0.5,vmax=0.5,aspect="auto")
        ax.set_xticks(range(len(sub.columns))); ax.set_xticklabels(sub.columns)
        ax.set_yticks(range(len(sub.index))); ax.set_yticklabels(sub.index)
        ax.set_xlabel("slow"); ax.set_ylabel("fast")
        ax.set_title(f"signal={sig}：平均分位20（相对50%的百分点）",fontsize=10.5)
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                v=sub.values[i,j]
                if np.isfinite(v):
                    ax.text(j,i,f"{v*100:.2f}",ha="center",va="center",fontsize=7.2,
                            color="#222" if abs(v-0.5)<0.002 else "white")
        ax.grid(False)
    fig.suptitle("413 组参数：最好的组合也只到 50.25%，没有任何一组把典型交易推过 50.5%",fontsize=12,x=0.01,ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"图2_参数热图.png"),bbox_inches="tight"); plt.close(fig)

def fig_param_dist():
    g=pd.read_csv(os.path.join(RES,"参数网格_含q.csv"))
    fig,ax=plt.subplots(figsize=(8.6,4.2))
    ax.hist(g.pct20*100,bins=30,color=C_BLUE,alpha=0.8,edgecolor="white")
    ax.axvline(50,color="#333",lw=1.2,ls="--",label="50%（无 edge）")
    d=g[(g.fast==12)&(g.slow==26)&(g.signal==9)].pct20.iloc[0]*100
    ax.axvline(d,color=C_NEG,lw=2,label=f"12/26/9：{d:.2f}%")
    ax.axvline(g.pct20.max()*100,color=C_POS,lw=2,ls=":",label=f"最优：{g.pct20.max()*100:.2f}%")
    ax.set_xlabel("平均分位20（%）"); ax.set_ylabel("参数组合数")
    ax.set_title("参数不是解药：413 组全部落在 49.8%–50.25%",fontsize=12,loc="left")
    ax.legend(frameon=False,fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"图3_参数分布.png"),bbox_inches="tight"); plt.close(fig)

def fig_index():
    df=pd.read_parquet(os.path.join(RES,"指数_日线.parquet")); df["date"]=pd.to_datetime(df["date"])
    idx={n:g.sort_values("date").set_index("date") for n,g in df.groupby("idx_name")}
    fig,axes=plt.subplots(3,2,figsize=(12.5,9.2),sharex="col")
    for row,name in enumerate(["CSI300","CSI500","CSI1000"]):
        d=idx[name]; r=(d.open.shift(-1)/d.open-1).fillna(0)
        dif=d.close.ewm(span=12,adjust=False).mean()-d.close.ewm(span=26,adjust=False).mean()
        dea=dif.ewm(span=9,adjust=False).mean(); p=(dif>dea).astype(float)
        turn=p.diff().abs().fillna(p.abs()); rr=p.shift(1).fillna(0)*r-turn*0.0005
        eqm=(1+rr).cumprod(); eqb=(1+r).cumprod()
        ax=axes[row,0]
        ax.plot(eqb.index,eqb,color=C_NEU,lw=1.3,label="买入持有")
        ax.plot(eqm.index,eqm,color=C_BLUE,lw=1.6,label="MACD 金叉")
        ax.set_yscale("log"); ax.set_ylabel("净值（对数）")
        ax.set_title(f"{name}：MACD 择时 vs 买入持有",fontsize=10.5,loc="left")
        ax.legend(frameon=False,fontsize=8)
        cm=(eqm.iloc[-1]**(252/len(eqm))-1)*100; cb=(eqb.iloc[-1]**(252/len(eqb))-1)*100
        ax.text(0.02,0.96,f"MACD 年化 {cm:.1f}% / 回撤 {(eqm/eqm.cummax()-1).min()*100:.0f}%",
                transform=ax.transAxes,fontsize=8.5,va="top",color=C_BLUE)
        ax.text(0.02,0.88,f"持有 年化 {cb:.1f}% / 回撤 {(eqb/eqb.cummax()-1).min()*100:.0f}%",
                transform=ax.transAxes,fontsize=8.5,va="top",color=C_NEU)
        ax2=axes[row,1]
        ax2.fill_between(eqm.index,(eqm/eqm.cummax()-1)*100,0,color=C_BLUE,alpha=0.35)
        ax2.plot(eqb.index,(eqb/eqb.cummax()-1)*100,color=C_NEU,lw=1.0,alpha=0.8)
        ax2.set_ylabel("回撤（%）")
        ax2.set_title("回撤：MACD（蓝） vs 买入持有（灰）",fontsize=10.5,loc="left")
    fig.suptitle("指数层面：MACD 把回撤减半；收益增益集中在部分指数/时期（T+1 开盘执行，单边 0.05%）",fontsize=11.5,x=0.01,ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"图4_指数择时.png"),bbox_inches="tight"); plt.close(fig)

def fig_yearly():
    y=pd.read_csv(os.path.join(RES,"最终_年度.csv"))
    fig,ax=plt.subplots(figsize=(9.4,4.0))
    colors=[C_POS if v>=0.5 else C_NEG for v in y.平均分位]
    ax.bar(y.年份,y.平均分位*100,color=colors,alpha=0.85)
    ax.axhline(50,color="#333",lw=1.2)
    for i,v in enumerate(y.平均分位*100):
        ax.text(y.年份.iloc[i],v+(0.06 if v>=50 else -0.12),f"{v:.1f}",ha="center",fontsize=7.5)
    ax.set_ylim(48.5,51.2); ax.set_ylabel("平均分位20（%）")
    ax.set_title("逐年平均分位：没有一年显著高于 50%，2023 年后反而偏弱",fontsize=12,loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"图5_年度分位.png"),bbox_inches="tight"); plt.close(fig)

def fig_dist():
    e=pd.read_parquet(os.path.join(RES,"事件_金叉_12269.parquet"))
    x=e.ex20.dropna()*100
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.2))
    ax=axes[0]
    ax.hist(x.clip(-40,40),bins=80,color=C_BLUE,alpha=0.8)
    ax.axvline(0,color="#333",lw=1); ax.axvline(x.mean(),color=C_ORANGE,lw=1.5,label=f"均值 {x.mean():.2f}%")
    ax.axvline(x.median(),color=C_NEG,lw=1.5,ls="--",label=f"中位 {x.median():.2f}%")
    ax.set_xlabel("20 日超额收益（%，截断到±40）"); ax.set_ylabel("信号数")
    ax.set_title("超额分布：均值被右尾拉起，典型交易在中位以下",fontsize=10.5,loc="left")
    ax.legend(frameon=False,fontsize=8)
    ax=axes[1]
    q=pd.qcut(e.ex20,10,labels=False,duplicates="drop")
    g=e.assign(q=q).groupby("q").agg(ex=("ex20","mean"),mfe=("mfe20","mean"),mae=("mae20","mean"))
    x=np.arange(len(g))
    ax.bar(x-0.2,g.mfe*100,width=0.4,color=C_POS,alpha=0.85,label="MFE20 均值")
    ax.bar(x+0.2,-g.mae*100,width=0.4,color=C_NEG,alpha=0.85,label="|MAE20| 均值")
    ax.set_xticks(x); ax.set_xticklabels([f"D{i+1}" for i in x],fontsize=7)
    ax.set_xlabel("按 20 日超额收益从低到高分成十组")
    ax.set_ylabel("%")
    ax.set_title("即使最差组也有约 7% 的 MFE；赚钱靠兑现，不靠方向",fontsize=10.5,loc="left")
    ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"图6_收益分布.png"),bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    fig_conditions(); fig_heatmap(); fig_param_dist(); fig_index(); fig_yearly(); fig_dist()
    print("charts done", os.listdir(FIG))
