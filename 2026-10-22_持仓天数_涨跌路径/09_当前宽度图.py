# -*- coding: utf-8 -*-
"""当前宽度与阈值图：2025-10 至 2026-10-22 的宽度20、等权指数、MA20/60。"""
import duckdb, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.rcParams['font.sans-serif']=['PingFang SC','Heiti SC','Arial Unicode MS']; plt.rcParams['axes.unicode_minus']=False
con=duckdb.connect('/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb',read_only=True)
d=con.execute("""select thscode,date,close from v_daily_hfq
 where date>='2025-01-01' and date<='2026-10-22' order by thscode,date""").df()
d['date']=pd.to_datetime(d['date'])
C=d.pivot(index='date',columns='thscode',values='close').sort_index()
R=C.pct_change(fill_method=None)
m=pd.DataFrame({'ret':R.mean(axis=1).clip(-.2,.2)}).fillna(0)
m['mkt']=(1+m.ret).cumprod(); m['ma20']=m.mkt.rolling(20,min_periods=20).mean(); m['ma60']=m.mkt.rolling(60,min_periods=60).mean()
m['breadth20']=(C>C.rolling(20,min_periods=20).mean()).where(C.notna()).mean(axis=1)
m=m[m.index>='2025-10-01']
fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,gridspec_kw={'height_ratios':[1,1.2]})
axes[0].plot(m.index,m.breadth20*100,color='#1f4e78',lw=1.4,label='宽度20')
axes[0].axhline(50,color='#c00000',ls='--',lw=1,label='趋势闸门 50%')
axes[0].axhline(10,color='#2e7d32',ls='--',lw=1,label='超跌闸门 10%')
last=m.iloc[-1]
axes[0].scatter([last.name],[last.breadth20*100],color='#c00000',zorder=5)
axes[0].annotate(f'{last.breadth20:.1%}',(last.name,last.breadth20*100),xytext=(-10,12),textcoords='offset points',ha='right',color='#c00000',fontweight='bold')
axes[0].set_ylabel('宽度（%）'); axes[0].set_ylim(0,100); axes[0].legend(loc='upper left'); axes[0].grid(alpha=.25)
axes[0].set_title('当前宽度与三档阈值（2025-10 至 2026-10-22）')
axes[1].plot(m.index,m.mkt,color='#333333',lw=1.3,label='等权全A')
axes[1].plot(m.index,m.ma20,color='#1f77b4',lw=1,label='MA20')
axes[1].plot(m.index,m.ma60,color='#ff7f0e',lw=1,label='MA60')
axes[1].set_ylabel('等权净值'); axes[1].legend(loc='upper left'); axes[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig('09_当前宽度与阈值.png',dpi=160); plt.close()
print('saved 09_当前宽度与阈值.png; last breadth',round(last.breadth20,4))
