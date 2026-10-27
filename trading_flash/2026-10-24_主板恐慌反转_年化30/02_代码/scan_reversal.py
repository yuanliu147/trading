import pandas as pd, numpy as np
pd.set_option('display.width',240); pd.set_option('display.max_columns',80)
f='01_数据/research_panel.parquet'
cols=['thscode','date','close','amount','amount20','amount5','next_actual_open','next_up_limit_10','next_down_limit_10',
 'ret1','ret_o3','ret_o5','ret_o10','ret_o20','mfe5','mae5','mfe10','mae10','mfe20','mae20',
 'return_5d','return_10d','return_20d','return_60d','return_120d','return_250d',
 'close_ma5','close_ma20','close_ma60','close_ma120','close_ma250','ma20_slope5','ma60_slope10','ma20_ma60',
 'range20_pct','atr20_pct','volatility20','volatility60','volatility_contraction',
 'drawdown60','drawdown250','price_position60','distance_low20','distance_low60',
 'breakout20','breakout60','breakout120','breakout250','amount_ratio1_20','amount_ratio5_20','amount_ratio20_60',
 'up_days20','near_limit_up_days20','limit_up_10','broken_limit_up_10','up_limit_05',
 'candle_body','upper_shadow','lower_shadow','range_pct','close_location','gap_open','hammer_shape','bullish_engulfing','inside_bar','narrow_range7',
 'macd_dif_pct','macd_dea_pct','macd_hist_pct','macd_hist_delta','macd_golden_cross','macd_dead_cross','rsi14','stoch9','kdj_j',
 'week_close_ma4','week_return4','week_return12','week_drawdown12','month_close_ma6','month_return3','month_return12',
 'market_return','market_return5','market_return20','market_return60','breadth20','breadth60','breadth120',
 'market_close_ma20','market_ma20_slope5','market_volatility20','advance_share','limit_up10_share','limit_down10_share','sentiment_score','sentiment_change5','new_high250_share']
df=pd.read_parquet(f,columns=cols)
for c in df.columns:
    if df[c].dtype=='float64': df[c]=df[c].astype('float32')
df['date']=pd.to_datetime(df['date'])
df['tradable']=df['next_actual_open'].notna() & (df['next_actual_open']<df['next_up_limit_10']*0.998) & (df['next_actual_open']>df['next_down_limit_10']*1.002)
df['liquid']=df['amount20']>=10_000_000
df['st_proxy']=(df['up_limit_05']==1)
for h in [3,5,10,20]:
    m=df.groupby('date')[f'ret_o{h}'].transform('mean')
    df[f'ex{h}']=df[f'ret_o{h}']-m
df['period']=np.select([df.date<='2021-12-31',df.date<='2023-12-31'],['train','val'],'test')
df['year']=df.date.dt.year
base=df[df.tradable & df.liquid & ~df.st_proxy].copy()

C={}
C['rev_panic']=(base.drawdown250<-0.5)&(base.rsi14<30)&(base.close_location>0.4)&(base.market_return20<-0.05)
C['rev_panic_engulf']=C['rev_panic']&(base.bullish_engulfing==1)
C['rev_panic_vol']=C['rev_panic']&(base.amount_ratio1_20>=1.3)
C['rev_panic_small']=C['rev_panic']&(base.amount20<50e6)
C['rev_panic_liquid']=C['rev_panic']&(base.amount20.between(50e6,300e6))
C['rev_panic_strong']=C['rev_panic']&(base.return_250d>0)
C['rev_panic_weak']=C['rev_panic']&(base.return_250d<0)
C['rev_panic_above60']=C['rev_panic']&(base.close>base.close_ma60)
C['rev_panic_below60']=C['rev_panic']&(base.close<base.close_ma60)
C['shakeout_uptrend']=(base.close>base.close_ma250)&(base.ma60_slope10>0)&(base.return_10d<-0.12)&(base.rsi14<40)&(base.close_location>0.5)&(base.market_return20<-0.05)
C['shakeout_ma60']=(base.close>base.close_ma250)&(base.ma60_slope10>0)&(base.close<base.close_ma60*1.02)&(base.close>base.close_ma120)&(base.return_10d<-0.08)&(base.close_location>0.5)
C['deep_panic_capit']=(base.drawdown250<-0.6)&(base.rsi14<25)&(base.close_location>0.3)&(base.market_return20<-0.08)
C['limit_down_rev']=(base.ret1<-0.095)&(base.close_location>0.2)
C['limit_down_next']=(base.ret1<-0.095)&(base.close_location>0.5)
C['three_down']=(base.up_days20<=3)&(base.return_5d<-0.12)&(base.close_location>0.6)&(base.market_return20<-0.03)
C['down7_rev']=(base.return_10d<-0.15)&(base.ret1>0)&(base.close_location>0.6)
C['capit_breadth']=(base.breadth20<0.15)&(base.market_return20<-0.06)&(base.close_location>0.5)&(base.rsi14<35)
C['rsi_extreme']=(base.rsi14<20)&(base.close_location>0.5)
C['bb_squeeze_panic']=(base.drawdown250<-0.4)&(base.volatility_contraction==1)&(base.close_location>0.5)&(base.market_return20<-0.05)

def stats(mask,h):
    s=base.loc[mask]
    return dict(n=len(s), r=s[f'ret_o{h}'].mean(), ex=s[f'ex{h}'].mean(), win=(s[f'ret_o{h}']>0).mean(),
                med=s[f'ret_o{h}'].median(), mae=s[f'mae{h}'].mean(), mfe=s[f'mfe{h}'].mean(),
                t=s[s.period=='train'][f'ex{h}'].mean(), v=s[s.period=='val'][f'ex{h}'].mean(), te=s[s.period=='test'][f'ex{h}'].mean())
rows=[]
for name,mask in C.items():
    for h in [5,10,20]:
        st=stats(mask,h); rows.append(dict(event=name,h=h,**st))
r=pd.DataFrame(rows)
r.to_csv('03_结果/reversal_scan.csv',index=False)
print(r[r.h==10].sort_values('ex',ascending=False).to_string(index=False))
print('\n=== yearly rev_panic 10d ===')
s=base[C['rev_panic']]
print(s.groupby('year').agg(n=('ret_o10','size'),mean=('ret_o10','mean'),ex=('ex10','mean'),win=('ret_o10',lambda x:(x>0).mean()),mae=('mae10','mean')).to_string())
print('\n=== rev_panic per period counts by year ===')
print(s.groupby(['period','year']).size().to_string())
