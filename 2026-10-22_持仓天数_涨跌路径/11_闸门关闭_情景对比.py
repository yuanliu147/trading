# -*- coding: utf-8 -*-
"""汇总“市场闸门关闭 / 当前同态”情景的收益与回撤。读取 11_*_权益.csv 与 11_*_逐笔.csv。"""
import pandas as pd, numpy as np
cases=[
 ('70/15 基准（闸门开）','11_base_proper','2018-2026，宽度>50%才开新仓'),
 ('70/15 无闸门','11_nogate_proper','不看宽度，始终按70/15个股+行业信号交易'),
 ('70/15 仅闸门关','11_offgate70','只在宽度≤50%时开新仓'),
 ('当前同态 70/15 完整','11_currentlike70','宽度10-20%+低于三线+斜率<0；仅2笔'),
 ('当前同态 近信号 2仓45%','11_near_current_2','当前同态；只买个股近信号，不看行业'),
 ('当前同态 近信号 5仓19%','11_near_current_5','当前同态；5仓分散'),
 ('极端<10% 超跌 5仓19%','11_ov_extreme_5','宽度<10%+低于三线+斜率<0'),
]
rows=[]
for name,stem,note in cases:
    e=pd.read_csv(f'{stem}_权益.csv',parse_dates=['date']).set_index('date').sort_index()
    s=e.equity; r=s.pct_change().fillna(0); dd=(s/s.cummax()-1).min()
    t=pd.read_csv(f'{stem}_逐笔.csv')
    if 'net_ret' not in t.columns:
        t['net_ret']=t.pnl/(t.entry*t.shares)
    rows.append(dict(情景=name,年化=(1+r).prod()**(252/len(r))-1,最大回撤=dd,
                     交易数=len(t),胜率=(t.net_ret>0).mean() if len(t) else np.nan,
                     盈亏比=t.pnl[t.pnl>0].sum()/max(1e-9,-t.pnl[t.pnl<0].sum()) if len(t) else np.nan,
                     平均仓位=e.exposure.mean(),期末权益=s.iloc[-1],备注=note))
R=pd.DataFrame(rows)
R.to_csv('11_闸门关闭_情景对比.csv',index=False,encoding='utf-8-sig',float_format='%.4f')
# 指数前瞻
P=pd.read_pickle('/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl')
dates=pd.DatetimeIndex(P['meta']['dates']); m=pd.Series(P['market_index'],index=dates)
mb=pd.Series(P['market_breadth20'],index=dates); ma20=pd.Series(P['market_ma20'],index=dates)
ma60=pd.Series(P['market_ma60'],index=dates); ma120=pd.Series(P['market_ma120'],index=dates)
slope=pd.Series(P['market_slope60'],index=dates)
f20=m.shift(-20)/m-1; fmin=m.iloc[::-1].rolling(20,min_periods=1).min().iloc[::-1]/m-1
cur=(mb>=.10)&(mb<.20)&(m<ma20)&(m<ma60)&(m<ma120)&(slope<0)
ext=(mb<.10)&(m<ma20)&(m<ma60)&(m<ma120)&(slope<0)
idx=[]
for name,sel in [('当前同态指数20日',cur),('极端<10%指数20日',ext)]:
    x=f20[sel].dropna()
    idx.append(dict(情景=name,样本天数=len(x),未来20日均值=x.mean(),胜率=(x>0).mean(),
                    最差路径=fmin[sel].min(),路径最低均值=fmin[sel].mean(),备注='等权全A，按日重叠'))
I=pd.DataFrame(idx); I.to_csv('11_当前同态_指数前瞻.csv',index=False,encoding='utf-8-sig',float_format='%.4f')
# 闸门日收益分解
e=pd.read_csv('08_trend_权益.csv',parse_dates=['date']).set_index('date').sort_index()
e['r']=e.equity.pct_change().fillna(0); e['mb']=mb.reindex(e.index); e['gate_on']=e.mb>.5
e['logr']=np.log1p(e.r)
G=e.groupby('gate_on').agg(天数=('r','size'),日均收益=('r','mean'),累计收益=('r',lambda s:(1+s).prod()-1),
                           对数贡献=('logr','sum'),平均持仓=('npos','mean'),空仓占比=('npos',lambda s:(s==0).mean())).reset_index()
G['闸门']=np.where(G.gate_on,'开（宽度>50%）','关（宽度≤50%）')
G=G[['闸门','天数','日均收益','累计收益','对数贡献','平均持仓','空仓占比']]
G.to_csv('11_闸门开关_日收益分解.csv',index=False,encoding='utf-8-sig',float_format='%.4f')
print(R.round(4).to_string(index=False)); print(); print(I.round(4).to_string(index=False)); print(); print(G.round(4).to_string(index=False))
