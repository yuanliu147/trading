# -*- coding: utf-8 -*-
"""2026-10-22 收盘：市场宽度、情绪与“极端超跌”观察名单。
策略只在宽度<10%时启动；本脚本只做观察，不产生买入指令。
价格：v_daily_hfq 为后复权（回测口径）；v_daily_qfq 为实际价（下单参考）。
"""
import duckdb, pandas as pd, numpy as np
DB='/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'

def board_of(c):
    c=str(c)
    if c.startswith(('600','601','603','605')): return '沪主板'
    if c.startswith(('000','001','002','003')): return '深主板'
    if c.startswith(('300','301')): return '创业板'
    if c.startswith(('688','689')): return '科创板'
    if c.startswith(('8','4','920')): return '北交所'
    return '其他'
con=duckdb.connect(DB,read_only=True)
d=con.execute("""select thscode,date,high,low,close,amount from v_daily_hfq
 where date>='2026-03-01' and date<='2026-10-22' order by thscode,date""").df()
d['date']=pd.to_datetime(d['date'])
C=d.pivot(index='date',columns='thscode',values='close').sort_index()
H=d.pivot(index='date',columns='thscode',values='high').sort_index()
L=d.pivot(index='date',columns='thscode',values='low').sort_index()
A=d.pivot(index='date',columns='thscode',values='amount').sort_index()
R=C.pct_change(fill_method=None)
gain=R.clip(lower=0).rolling(14,min_periods=10).mean(); loss=(-R.clip(upper=0)).rolling(14,min_periods=10).mean()
RSI=100-100/(1+gain/(loss+1e-9))
TR=pd.DataFrame(np.maximum.reduce([(H-L).values,np.abs(H-C.shift(1)).values,np.abs(L-C.shift(1)).values]),
                index=C.index,columns=C.columns)
ATR=TR.rolling(14,min_periods=10).mean()/C
AMT20=A.rolling(20,min_periods=15).mean(); R20=C/C.shift(20)-1
MA20=C.rolling(20,min_periods=20).mean(); MA60=C.rolling(60,min_periods=60).mean(); MA120=C.rolling(120,min_periods=80).mean()
last=pd.DataFrame({'后复权收盘':C.iloc[-1],'相对强弱指标14':RSI.iloc[-1],'真实波幅占收盘':ATR.iloc[-1],
                   '20日均成交额_万':AMT20.iloc[-1]/1e4,'20日收益':R20.iloc[-1],
                   '20日均线':MA20.iloc[-1],'60日均线':MA60.iloc[-1],'120日均线':MA120.iloc[-1]})
last['距20日均线']=last['后复权收盘']/last['20日均线']-1
q=con.execute("select thscode,close as 实际收盘 from v_daily_qfq where date='2026-10-22'").df().set_index('thscode')['实际收盘']
last['实际收盘']=last.index.map(q)
names=pd.read_csv('股票名称_2026-10-22.csv',dtype=str)[['thscode','name']].set_index('thscode')['name']
last['名称']=last.index.map(names); last['是否特别处理']=last['名称'].fillna('').str.contains('ST')
last['板块']=last.index.map(board_of)
last['主板可买']=last['板块'].isin(['沪主板','深主板'])&~last['是否特别处理']
last=last[['名称','板块','主板可买','实际收盘','后复权收盘','相对强弱指标14','真实波幅占收盘','20日均成交额_万','20日收益','距20日均线','是否特别处理']]
last.index.name='代码'
f=last[(last['相对强弱指标14']<30)&(last['真实波幅占收盘']<=.06)&(last['20日均成交额_万']>=10000)&(last['后复权收盘']>=2)&~last['是否特别处理']].sort_values('20日均成交额_万')
f.to_csv('09_今日超跌观察_2026-10-22.csv',encoding='utf-8-sig',float_format='%.4f')
f[(f['相对强弱指标14']<25)&(f['真实波幅占收盘']<=.045)].sort_values('相对强弱指标14').to_csv('09_今日超跌重点_2026-10-22.csv',encoding='utf-8-sig',float_format='%.4f')
fm=f[f['主板可买']]
fm.to_csv('09_今日超跌观察_主板_2026-10-22.csv',encoding='utf-8-sig',float_format='%.4f')
fm[(fm['相对强弱指标14']<25)&(fm['真实波幅占收盘']<=.045)].sort_values('相对强弱指标14').to_csv('09_今日超跌重点_主板_2026-10-22.csv',encoding='utf-8-sig',float_format='%.4f')

# 市场状态与情绪
mkt=pd.DataFrame({'ret1':R.mean(axis=1).clip(-.2,.2)}).fillna(0)
mkt['mkt']=(1+mkt.ret1).cumprod()
mkt['ma20']=mkt.mkt.rolling(20,min_periods=20).mean(); mkt['ma60']=mkt.mkt.rolling(60,min_periods=60).mean(); mkt['ma120']=mkt.mkt.rolling(120,min_periods=80).mean()
mkt['breadth20']=(C>MA20).where(C.notna()).mean(axis=1); mkt['breadth60']=(C>MA60).where(C.notna()).mean(axis=1); mkt['breadth120']=(C>MA120).where(C.notna()).mean(axis=1)
mkt['r20']=mkt.mkt/mkt.mkt.shift(20)-1; mkt['vol20']=np.log(mkt.mkt).diff().rolling(20,min_periods=15).std()*np.sqrt(252)
lr=R.iloc[-1].dropna()
def lim(code):
    if code.startswith(('300','301','688')): return .196
    if code.startswith(('8','4','920')): return .296
    return .098
lu=sum(1 for c,v in lr.items() if v>=lim(c)-.005); ld=sum(1 for c,v in lr.items() if v<=-lim(c)+.005)
da=A.sum(axis=1)
state=pd.DataFrame([{
    '日期':C.index[-1].strftime('%Y-%m-%d'),'股票数':len(lr),'中位涨跌':lr.median(),'平均涨跌':lr.mean(),
    '下跌占比':(lr<0).mean(),'跌超5%占比':(lr<-.05).mean(),'涨停数':lu,'跌停数':ld,
    '成交额_亿':da.iloc[-1]/1e8,'20日均额_亿':da.iloc[-21:-1].mean()/1e8,'成交额比':da.iloc[-1]/da.iloc[-21:-1].mean(),
    '市场宽度20':mkt.breadth20.iloc[-1],'市场宽度60':mkt.breadth60.iloc[-1],'市场宽度120':mkt.breadth120.iloc[-1],
    '20日收益':mkt.r20.iloc[-1],'20日波动':mkt.vol20.iloc[-1],'等权净值':mkt.mkt.iloc[-1],
    '20日均线':mkt.ma20.iloc[-1],'60日均线':mkt.ma60.iloc[-1],'120日均线':mkt.ma120.iloc[-1],
}])
state.to_csv('09_今日市场状态_2026-10-22.csv',encoding='utf-8-sig',float_format='%.4f',index=False)
print(state.T.to_string(header=False))
print('超跌观察',len(f),'重点',len(f[(f['相对强弱指标14']<25)&(f['真实波幅占收盘']<=.045)]))
print('主板超跌观察',len(fm),'主板重点',len(fm[(fm['相对强弱指标14']<25)&(fm['真实波幅占收盘']<=.045)]))
print('策略排序前5（全板块）'); print(f.head(5).round(3).to_string())
print('策略排序前5（主板）'); print(fm.head(5).round(3).to_string())
