# -*- coding: utf-8 -*-
"""对 70/15 基准、+极端超跌增强、增强+回撤节流做严谨回撤审计。
输出：策略对比、年度、压力窗口、滚动收益、block bootstrap 回撤分布、图。
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.rcParams['font.sans-serif']=['PingFang SC','Heiti SC','Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False

D='.'
def load(name):
    e=pd.read_csv(f'08_{name}_权益.csv',parse_dates=['date']).set_index('date').sort_index()
    return e
liq=load('liq1e8'); trend=load('trend')
# 回撤节流版：用引擎重跑太重，这里按日收益近似“回撤减半”不可靠；改为已跑出的结果参数表引用
# 为严谨，回撤节流版单独保存在 08_liq1e8_dd 中（若无则跳过）
import os
liqdd=load('liq1e8_dd') if os.path.exists('08_liq1e8_dd_权益.csv') else None
series={'70/15基准':trend,'70/15+超跌':liq}
if liqdd is not None: series['增强+回撤节流']=liqdd

def metrics(e,name):
    s=e.equity; r=s.pct_change().fillna(0)
    dd=s/s.cummax()-1
    # drawdown duration
    underwater=dd<0
    # longest underwater streak
    longest=0; cur=0
    for u in underwater:
        cur=cur+1 if u else 0; longest=max(longest,cur)
    rec=None
    trough=dd.idxmin(); peak=s[:trough].idxmax()
    after=s[trough:][s[trough:].cummax()>s[peak]]
    rec=after.index[0] if len(after) else None
    ulcer=np.sqrt((dd**2).mean())
    downside=r[r<0].std()*np.sqrt(252)
    ann=(1+r).prod()**(252/len(r))-1
    return dict(策略=name,年化=ann,年化波动=r.std()*np.sqrt(252),夏普=r.mean()/(r.std()+1e-9)*np.sqrt(252),
                索提诺=r.mean()*252/max(downside,1e-9),最大回撤=dd.min(),最大回撤_盘中=(e.equity_low/e.equity.cummax()-1).min(),
                卡玛=ann/abs(dd.min()),Ulcer=ulcer,最长水下天数=longest,
                回撤峰=peak.date(),回撤谷=trough.date(),修复日=rec.date() if rec is not None else None,
                最差20日=r.rolling(20).sum().min(),最差月=(1+r).resample('ME').prod().sub(1).min(),
                平均仓位=e.exposure.mean(),期末权益=s.iloc[-1])
M=pd.DataFrame([metrics(e,n) for n,e in series.items()])
M.to_csv('10_策略对比.csv',index=False,encoding='utf-8-sig',float_format='%.4f')

# yearly
Y=[]
for n,e in series.items():
    s=e.equity
    for y in range(2018,2027):
        d=s[s.index.year==y]
        if not len(d): continue
        prev=s[s.index<d.index[0]]; base=prev.iloc[-1] if len(prev) else d.iloc[0]
        Y.append(dict(策略=n,年份=y,收益=d.iloc[-1]/base-1,年内回撤=(d/d.cummax()-1).min(),
                      年内波动=d.pct_change().std()*np.sqrt(252)))
Y=pd.DataFrame(Y); Y.to_csv('10_年度表现.csv',index=False,encoding='utf-8-sig',float_format='%.4f')

# stress windows
windows={
 '2018全年':('2018-01-01','2018-12-31'),
 '2018Q4急跌':('2018-10-01','2018-12-31'),
 '2020疫情':('2020-01-20','2020-03-23'),
 '2021Q4-2022Q1':('2021-12-01','2022-03-31'),
 '2022Q3':('2022-08-01','2022-10-31'),
 '2024年初':('2024-01-01','2024-02-29'),
 '2024下半年':('2024-06-01','2024-09-30'),
 '2025Q4':('2025-08-01','2025-12-31'),
 '2026Q1':('2026-03-01','2026-04-30'),
 '当前回撤':('2026-09-22','2026-10-13'),
}
rows=[]
for n,e in series.items():
    s=e.equity
    for w,(a,b) in windows.items():
        d=s[(s.index>=a)&(s.index<=b)]
        if len(d)<2: continue
        prev=s[s.index<a]; base=prev.iloc[-1] if len(prev) else d.iloc[0]
        rows.append(dict(策略=n,窗口=w,区间收益=d.iloc[-1]/base-1,区间最大回撤=(d/d.cummax()-1).min()))
S=pd.DataFrame(rows); S.to_csv('10_压力测试.csv',index=False,encoding='utf-8-sig',float_format='%.4f')

# rolling 252d
R=[]
for n,e in series.items():
    s=e.equity; r=s.pct_change().fillna(0)
    roll=(1+r).rolling(252).apply(np.prod,raw=True)-1
    for v in roll.dropna().values:
        R.append(dict(策略=n,滚动252日收益=v))
R=pd.DataFrame(R); R.to_csv('10_滚动收益.csv',index=False,encoding='utf-8-sig',float_format='%.4f')

# block bootstrap max drawdown
def block_bootstrap(r,N,block=10,npaths=3000,seed=42):
    rng=np.random.default_rng(seed); r=np.asarray(r)
    out=[]
    nb=int(np.ceil(N/block))
    for _ in range(npaths):
        starts=rng.integers(0,len(r)-block,nb)
        x=np.concatenate([r[s:s+block] for s in starts])[:N]
        eq=np.cumprod(1+x); dd=(eq/np.maximum.accumulate(eq)-1).min()
        out.append(dd)
    return np.array(out)
B=[]
for n,e in series.items():
    r=e.equity.pct_change().fillna(0).values
    for N,label in [(252,'1年'),(756,'3年')]:
        x=block_bootstrap(r,N)
        B.append(dict(策略=n,期限=label,路径数=len(x),中位回撤=np.median(x),P95回撤=np.quantile(x,.05),
                      P99回撤=np.quantile(x,.01),跌破20概率=(x<=-.20).mean(),跌破30概率=(x<=-.30).mean(),
                     最差回撤=x.min()))
B=pd.DataFrame(B); B.to_csv('10_自举回撤.csv',index=False,encoding='utf-8-sig',float_format='%.4f')

# trade stats
T=[]
file_map={'70/15基准':'08_trend_逐笔.csv','70/15+超跌':'08_liq1e8_逐笔.csv','增强+回撤节流':'08_liq1e8_dd_逐笔.csv'}
for n,e in series.items():
    t=pd.read_csv(file_map[n])
    t['net_ret']=t['pnl']/(t['entry']*t['shares'])
    contrib=t.pnl
    T.append(dict(策略=n,笔数=len(t),单笔净均值=t.net_ret.mean(),单笔净中位=t.net_ret.median(),
                  净胜率=(t.net_ret>0).mean(),盈亏比=t.pnl[t.pnl>0].sum()/max(1e-9,-t.pnl[t.pnl<0].sum()),
                  前五笔净盈亏占比=t.pnl.nlargest(5).sum()/max(1e-9,t.pnl.sum()),
                  最差单笔=t.net_ret.min(),最好单笔=t.net_ret.max(),平均持有=t.hold.mean()))
T=pd.DataFrame(T); T.to_csv('10_逐笔统计.csv',index=False,encoding='utf-8-sig',float_format='%.4f')

print('=== 策略对比 ==='); print(M.round(4).to_string(index=False))
print('\n=== 年度 ==='); print(Y.pivot(index='年份',columns='策略',values='收益').round(4).to_string())
print('\n=== 压力 ==='); print(S.pivot(index='窗口',columns='策略',values='区间收益').round(4).to_string())
print('\n=== 自举回撤 ==='); print(B.round(4).to_string(index=False))
print('\n=== 逐笔 ==='); print(T.round(4).to_string(index=False))

# charts
fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
for n,e in series.items():
    axes[0].plot(e.index,e.equity/20000,label=n,lw=1.5)
axes[0].set_yscale('log'); axes[0].set_ylabel('权益（倍，对数）'); axes[0].legend(); axes[0].grid(alpha=.25)
axes[0].set_title('权益曲线与回撤（2018-01 至 2026-10-13，2万起点）')
for n,e in series.items():
    dd=e.equity/e.equity.cummax()-1
    axes[1].plot(dd.index,dd,label=n,lw=1.0,alpha=.8)
axes[1].yaxis.set_major_formatter(PercentFormatter(1.0)); axes[1].set_ylabel('回撤'); axes[1].legend(); axes[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig('10_权益与回撤.png',dpi=160); plt.close()

fig,ax=plt.subplots(figsize=(12,5))
piv=S.pivot(index='窗口',columns='策略',values='区间收益').reindex(list(windows))
piv.plot(kind='bar',ax=ax,width=.8)
ax.yaxis.set_major_formatter(PercentFormatter(1.0)); ax.set_ylabel('区间收益'); ax.grid(axis='y',alpha=.25)
ax.set_title('压力窗口收益'); plt.xticks(rotation=30,ha='right'); plt.tight_layout(); plt.savefig('10_压力测试.png',dpi=160); plt.close()

fig,ax=plt.subplots(figsize=(10,5))
for n,g in R.groupby('策略'):
    ax.hist(g['滚动252日收益'],bins=50,alpha=.5,label=n)
ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.set_title('滚动252日收益分布'); ax.legend(); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig('10_滚动收益分布.png',dpi=160); plt.close()
