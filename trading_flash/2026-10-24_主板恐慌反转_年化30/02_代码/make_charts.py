import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang SC','Heiti SC','SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=150
eq=pd.read_csv('03_结果/实际价_每日权益.csv',parse_dates=['date']).set_index('date')
tr=pd.read_csv('03_结果/实际价_逐笔.csv',parse_dates=['entry_date','exit_date'])
ann=pd.read_csv('03_结果/最终_年度.csv',index_col=0,parse_dates=True).iloc[:,0]
dd=eq.equity/eq.equity.cummax()-1
# 1 equity + dd
fig,ax=plt.subplots(2,1,figsize=(11,7),gridspec_kw={'height_ratios':[3,1]},sharex=True)
ax[0].plot(eq.index,eq.equity/100000,color='#1f4e79',lw=1.6,label='策略净值（实际价+整手+费用）')
ax[0].set_yscale('log'); ax[0].set_ylabel('净值（对数）'); ax[0].grid(alpha=.25)
ax[0].legend(loc='upper left'); ax[0].set_title('主板恐慌洗盘反转：净值与回撤（2017-11-23 至 2026-10-23，10 万本金）')
ax[1].fill_between(dd.index,dd.values,0,color='#c0392b',alpha=.55)
ax[1].set_ylabel('回撤'); ax[1].yaxis.set_major_formatter(PercentFormatter(1)); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig('04_图表/01_净值与回撤.png'); plt.close()
# 2 annual
fig,ax=plt.subplots(figsize=(10,4.5))
vals=ann.values*100; cols=['#c0392b' if v<0 else '#2e7d32' for v in vals]
ax.bar([str(i.year) for i in ann.index],vals,color=cols)
for i,v in enumerate(vals): ax.text(i,v+(1 if v>=0 else -2),f'{v:.1f}%',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
ax.axhline(30,color='#f39c12',ls='--',lw=1,label='30% 目标线'); ax.legend()
ax.set_ylabel('年度收益'); ax.set_title('年度收益：10 个年度全部为正'); ax.grid(axis='y',alpha=.25)
plt.tight_layout(); plt.savefig('04_图表/02_年度收益.png'); plt.close()
# 3 trade dist
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ax[0].hist(tr.ret*100,bins=24,color='#1f4e79',alpha=.8)
ax[0].axvline(0,color='k',lw=.8); ax[0].set_xlabel('单笔净收益（%）'); ax[0].set_ylabel('笔数'); ax[0].set_title('97 笔单笔收益分布')
ax[0].grid(alpha=.25)
reason=tr.exit_reason.map({'time':'15日到期','target':'+20%止盈','eod':'期末'}).value_counts()
ax[1].bar(reason.index,reason.values,color=['#2e7d32','#1f4e79','#7f8c8d'])
for i,v in enumerate(reason.values): ax[1].text(i,v+.5,str(v),ha='center')
ax[1].set_title('退出原因'); ax[1].set_ylabel('笔数'); ax[1].grid(axis='y',alpha=.25)
plt.tight_layout(); plt.savefig('04_图表/03_逐笔收益与退出.png'); plt.close()
# 4 heatmap delay x tp of min3
g=pd.read_csv('03_结果/grid_all_periods.csv'); g['min3']=g[['train_cagr','val_cagr','test_cagr']].min(axis=1)
piv=g.pivot_table(index='delay',columns='tp',values='min3',aggfunc='max')
fig,ax=plt.subplots(figsize=(7.5,4.2))
im=ax.imshow(piv.values*100,cmap='YlGnBu',aspect='auto')
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f'{c:.0%}' for c in piv.columns])
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f'延迟{int(i)}日' for i in piv.index])
for i in range(piv.shape[0]):
  for j in range(piv.shape[1]):
    v=piv.values[i,j]*100
    ax.text(j,i,f'{v:.1f}%',ha='center',va='center',fontsize=9,color='white' if v>30 else 'black')
ax.set_xlabel('止盈阈值（收盘/盘中触价）'); ax.set_ylabel('信号后延迟入场')
ax.set_title('参数敏感性：三段最低年化（2 仓、15 日、RSI<35、DD<-45%）')
plt.colorbar(im,ax=ax,label='三段最低年化（%）'); plt.tight_layout()
plt.savefig('04_图表/04_参数热力图.png'); plt.close()
# 5 walk-forward
fig,ax=plt.subplots(figsize=(8,4.2))
labels=['2017–2021\n训练','2022–2023\n验证','2024–2026\n检验']
cand=[33.9,40.0,51.8]; wf=[None,26.0,48.5]
x=np.arange(3); ax.bar(x-.18,cand,.36,label='候选参数（全期选参）',color='#1f4e79')
ax.bar(x+.18,[0 if v is None else v for v in wf],.36,label='走步选参（仅用过去）',color='#e67e22')
for i,v in enumerate(cand): ax.text(i-.18,v+.8,f'{v:.1f}%',ha='center')
for i,v in enumerate(wf):
  if v is not None: ax.text(i+.18,v+.8,f'{v:.1f}%',ha='center')
ax.axhline(30,color='#f39c12',ls='--',lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('年化收益（%）'); ax.set_title('候选参数 vs 诚实走步：30%+ 依赖选参'); ax.legend(); ax.grid(axis='y',alpha=.25)
plt.tight_layout(); plt.savefig('04_图表/05_走步检验.png'); plt.close()
# 6 event family
ev=pd.read_csv('03_结果/event_scan.csv')
ev10=ev[ev.h==10].set_index('event')
names={'rev_panic':'恐慌超跌反转','rev_deep':'深度超跌','pull_ma20_tight':'均线回踩(紧)','macd_lowcross':'MACD零下金叉','macd_highcross':'MACD零上金叉','brk60_vol':'60日突破放量','lu_first':'首板','lu_multi':'连板','lu_broken':'炸板','gap_up':'高开高走','trend_quiet':'安静趋势','engulf_up':'看涨吞没'}
sel=[k for k in names if k in ev10.index]
vals=[ev10.loc[k,'test_ex']*100 for k in sel]
order=np.argsort(vals)
fig,ax=plt.subplots(figsize=(9,5))
ax.barh([names[sel[i]] for i in order],[vals[i] for i in order],color=['#2e7d32' if vals[i]>0 else '#c0392b' for i in order])
ax.axvline(0,color='k',lw=.8); ax.set_xlabel('2024–2026 十日超额收益（%）'); ax.set_title('多角度事件扫描：次日开盘买入、扣成本前的十日超额')
ax.grid(axis='x',alpha=.25)
plt.tight_layout(); plt.savefig('04_图表/06_事件族对比.png'); plt.close()
print('charts done')
