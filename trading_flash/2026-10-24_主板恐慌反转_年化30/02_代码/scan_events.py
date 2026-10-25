import pandas as pd, numpy as np, duckdb, json, os
pd.set_option('display.width',240); pd.set_option('display.max_columns',60)
f='01_数据/research_panel.parquet'
cols=['thscode','date','close','amount','amount20','hfq_close','next_actual_open','next_up_limit_10','next_down_limit_10',
 'ret1','ret_o3','ret_o5','ret_o10','ret_o20','ret_o30','mfe5','mae5','mfe10','mae10','mfe20','mae20',
 'return_5d','return_10d','return_20d','return_60d','return_120d','return_250d',
 'close_ma5','close_ma20','close_ma60','close_ma120','close_ma250','ma20_slope5','ma60_slope10','ma20_ma60',
 'amplitude' if False else 'range20_pct','atr20_pct','volatility20','volatility60','volatility_contraction',
 'drawdown60','drawdown250','price_position60','distance_low20','distance_low60','distance_swing_high',
 'breakout20','breakout60','breakout120','breakout250','amount_ratio1_20','amount_ratio5_20','amount_ratio20_60',
 'up_days20','near_limit_up_days20','limit_up_10','broken_limit_up_10','up_limit_05',
 'candle_body','upper_shadow','lower_shadow','range_pct','close_location','gap_open','hammer_shape','bullish_engulfing','inside_bar','narrow_range7',
 'macd_dif_pct','macd_dea_pct','macd_hist_pct','macd_hist_delta','macd_golden_cross','macd_dead_cross','rsi14','stoch9','kdj_j',
 'week_close_ma4','week_return4','week_return12','week_drawdown12','month_close_ma6','month_return3','month_return12',
 'market_return','market_return5','market_return20','market_return60','breadth20','breadth60','breadth120',
 'market_close_ma20','market_ma20_slope5','market_volatility20','advance_share','limit_up10_share','sentiment_score','sentiment_change5','new_high250_share']
df=pd.read_parquet(f,columns=cols)
for c in df.columns:
    if df[c].dtype=='float64': df[c]=df[c].astype('float32')
df['date']=pd.to_datetime(df['date'])
# mainboard & basic feasibility
df['tradable']=df['next_actual_open'].notna() & (df['next_actual_open']<df['next_up_limit_10']*0.998) & (df['next_actual_open']>df['next_down_limit_10']*1.002)
df['liquid']=df['amount20']>=10_000_000
df['st_proxy']=(df['up_limit_05']==1)
# market benchmark per date & horizon
for h in [3,5,10,20,30]:
    m=df.groupby('date')[f'ret_o{h}'].transform('mean')
    df[f'ex{h}']=df[f'ret_o{h}']-m
# period
df['period']=np.select([df.date<='2021-12-31',df.date<='2023-12-31'],['train','val'],'test')
base=df[df.tradable & df.liquid & ~df.st_proxy].copy()
print('rows',len(df),'base',len(base))
print(base.groupby('period').size())

C={}
C['lu_first']=(base.limit_up_10==1)&(base.near_limit_up_days20==0)
C['lu_first_vol']=C['lu_first']&(base.amount_ratio1_20>=1.5)
C['lu_first_hi']=C['lu_first']&(base.breakout250==1)
C['lu_first_lowamt']=C['lu_first']&(base.amount20<50e6)
C['lu_multi']=(base.limit_up_10==1)&(base.near_limit_up_days20>=1)
C['lu_broken']=(base.broken_limit_up_10==1)
C['lu_recent_pull']=(base.near_limit_up_days20>=1)&(base.ret1<0)&(base.close_ma5>-0.01)&(base.amount_ratio1_20<1)
C['lu_recent_pull20']=(base.near_limit_up_days20>=1)&(base.return_5d<0)&(base.close_ma20>0)&(base.amount_ratio5_20<1)
C['brk20_vol']=(base.breakout20==1)&(base.amount_ratio1_20>=1.5)&(base.close_ma60>0)&(base.ma20_ma60>0)
C['brk60_vol']=(base.breakout60==1)&(base.amount_ratio1_20>=1.5)&(base.close_ma60>0)&(base.ma20_ma60>0)
C['brk120_vol']=(base.breakout120==1)&(base.amount_ratio1_20>=1.5)&(base.close_ma60>0)&(base.ma20_ma60>0)
C['brk250_vol']=(base.breakout250==1)&(base.amount_ratio1_20>=1.5)&(base.close_ma60>0)&(base.ma20_ma60>0)
C['brk250_tight']=(base.breakout250==1)&(base.range20_pct<0.10)&(base.amount_ratio1_20>=1.5)
C['brk60_lowvol']=(base.breakout60==1)&(base.atr20_pct<0.04)&(base.amount_ratio1_20>=1.3)&(base.close_ma60>0)
C['pull_ma20']=(base.close_ma60>0)&(base.ma20_ma60>0)&(base.ma20_slope5>0)&(base.close_ma20>0)&(base.return_5d<0)&(base.amount_ratio5_20<1.2)
C['pull_ma20_tight']=C['pull_ma20']&(base.atr20_pct<0.04)&(base.range20_pct<0.12)
C['rev_deep']=(base.drawdown250<-0.5)&(base.rsi14<30)&(base.close_location>0.4)
C['rev_panic']=C['rev_deep']&(base.market_return20<-0.05)
C['macd_lowcross']=(base.macd_golden_cross==1)&(base.macd_dif_pct<0)&(base.close_ma60>0)
C['macd_highcross']=(base.macd_golden_cross==1)&(base.macd_dif_pct>0)&(base.close_ma60>0)&(base.ma60_slope10>0)
C['macd_cross_ma20']=(base.macd_golden_cross==1)&(base.close_ma20>0)&(base.close_ma60>0)&(base.ma20_ma60>0)
C['macd_wk']=(base.macd_golden_cross==1)&(base.week_close_ma4>0)&(base.week_return4>0)&(base.close_ma60>0)
C['mtf_align']=(base.month_close_ma6>0)&(base.week_close_ma4>0)&(base.week_return12>0)&(base.close_ma20>0)&(base.macd_golden_cross==1)
C['mtf_pull']=(base.month_close_ma6>0)&(base.week_close_ma4>0)&(base.week_return12>0)&(base.close_ma20>0)&(base.return_5d<0)&(base.amount_ratio5_20<1.2)
C['hammer_sup']=(base.hammer_shape==1)&(base.distance_low60<0.05)&(base.close_ma60>0)
C['engulf_up']=(base.bullish_engulfing==1)&(base.close_ma60>0)&(base.ma20_ma60>0)
C['coil']=(base.inside_bar==1)&(base.narrow_range7==1)&(base.close_ma60>0)&(base.ma20_ma60>0)
C['narrow_brk']=(base.narrow_range7==1)&(base.breakout20==1)&(base.amount_ratio1_20>=1.5)
C['gap_up']=(base.gap_open>0.03)&(base.close_location>0.6)&(base.amount_ratio1_20>=1.5)&(base.close_ma60>0)
C['pos_high']=(base.price_position60>0.9)&(base.volatility_contraction==1)&(base.close_ma60>0)
C['wk_ret12']=(base.week_return12>0.1)&(base.week_drawdown12>-0.05)&(base.close_ma20>0)
C['mkt_hot_lu']=C['lu_first']&(base.breadth20>0.5)
C['mkt_cold_lu']=C['lu_first']&(base.breadth20<0.4)
C['lu_first_ma60']=C['lu_first']&(base.close_ma60>0)&(base.ma20_ma60>0)
C['lu_first_vol_ma60']=C['lu_first_vol']&(base.close_ma60>0)&(base.ma20_ma60>0)
C['trend_quiet']=(base.close_ma20>0)&(base.close_ma60>0)&(base.ma20_ma60>0)&(base.ma60_slope10>0)&(base.atr20_pct<0.04)
C['trend_mom']=(base.close_ma20>0)&(base.close_ma60>0)&(base.ma20_ma60>0)&(base.return_60d>0.1)&(base.return_20d>0)
def stat(mask,h=10):
    s=base.loc[mask]
    if len(s)==0: return None
    out={}
    for per in ['train','val','test']:
        x=s[s.period==per]
        out[per+'_n']=len(x)
        out[per+'_r']=x[f'ret_o{h}'].mean() if len(x) else np.nan
        out[per+'_ex']=x[f'ex{h}'].mean() if len(x) else np.nan
        out[per+'_win']=(x[f'ret_o{h}']>0).mean() if len(x) else np.nan
    out['all_n']=len(s); out['all_r']=s[f'ret_o{h}'].mean(); out['all_ex']=s[f'ex{h}'].mean()
    out['all_win']=(s[f'ret_o{h}']>0).mean(); out['med_ex']=s[f'ex{h}'].median()
    return out
rows=[]
for name,mask in C.items():
    for h in [5,10,20]:
        st=stat(mask,h)
        if st: rows.append(dict(event=name,h=h,**st))
res=pd.DataFrame(rows)
res=res.sort_values(['h','all_ex'],ascending=[True,False])
res.to_csv('results/development/event_scan.csv',index=False)
print(res[res.h==10][['event','train_n','train_r','train_ex','val_n','val_r','val_ex','test_n','test_r','test_ex','all_n','all_r','all_ex','all_win']].to_string(index=False))
