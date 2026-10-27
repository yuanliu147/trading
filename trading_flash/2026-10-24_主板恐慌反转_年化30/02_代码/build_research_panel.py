import duckdb, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
con=duckdb.connect()
con.execute("PRAGMA threads=8; PRAGMA memory_limit='8GB';")
src=str(ROOT / '01_数据' / 'features.parquet')
out=str(ROOT / '01_数据' / 'research_panel.parquet')
if os.path.exists(out): os.remove(out)
cols = """
thscode,date,open,high,low,close,volume,amount,listing_session,
hfq_open,hfq_high,hfq_low,hfq_close,next_hfq_open,next_actual_open,next_up_limit_10,next_down_limit_10,
ret1,return_5d,return_10d,return_20d,return_60d,return_120d,return_250d,
close_ma5,close_ma10,close_ma20,close_ma50,close_ma60,close_ma120,close_ma250,
ma20_slope5,ma60_slope10,ma20_ma60,amount5,amount20,amount60,volume20,
atr20_pct,volatility20,volatility60,volatility_contraction,range20_pct,
drawdown60,drawdown250,price_position60,distance_low20,distance_low60,distance_swing_high,distance_swing_low,
breakout20,breakout60,breakout120,breakout250,amount_ratio1_20,amount_ratio5_20,amount_ratio20_60,
up_days20,near_limit_up_days20,limit_up_10,broken_limit_up_10,up_limit_05,limit_up_05,
candle_body,upper_shadow,lower_shadow,range_pct,close_location,gap_open,hammer_shape,bullish_engulfing,inside_bar,narrow_range7,
macd_dif_pct,macd_dea_pct,macd_hist_pct,macd_hist_delta,macd_golden_cross,macd_dead_cross,rsi14,stoch9,kdj_j,
week_close_ma4,week_return4,week_return12,week_drawdown12,month_close_ma6,month_return3,month_return12,
market_return,market_return5,market_return20,market_return60,breadth20,breadth60,breadth120,
market_close_ma20,market_ma20_slope5,market_volatility20,advance_share,limit_up10_share,limit_down10_share,
sentiment_score,sentiment_change5,new_high250_share
"""
q=f"""
with base as (
 select {cols} from read_parquet('{src}')
 where substr(thscode,1,3) in ('600','601','603','605','000','001','002','003')
   and listing_session>=250
),
fwd as (
 select *,
   lead(hfq_close,3) over w as close_3,
   lead(hfq_close,5) over w as close_5,
   lead(hfq_close,10) over w as close_10,
   lead(hfq_close,20) over w as close_20,
   lead(hfq_close,30) over w as close_30,
   max(hfq_high) over (partition by thscode order by date rows between 1 following and 5 following) as max_high_5,
   min(hfq_low)  over (partition by thscode order by date rows between 1 following and 5 following) as min_low_5,
   max(hfq_high) over (partition by thscode order by date rows between 1 following and 10 following) as max_high_10,
   min(hfq_low)  over (partition by thscode order by date rows between 1 following and 10 following) as min_low_10,
   max(hfq_high) over (partition by thscode order by date rows between 1 following and 20 following) as max_high_20,
   min(hfq_low)  over (partition by thscode order by date rows between 1 following and 20 following) as min_low_20
 from base window w as (partition by thscode order by date)
)
select *,
 round(close_3/next_hfq_open-1,5) ret_o3,
 round(close_5/next_hfq_open-1,5) ret_o5,
 round(close_10/next_hfq_open-1,5) ret_o10,
 round(close_20/next_hfq_open-1,5) ret_o20,
 round(close_30/next_hfq_open-1,5) ret_o30,
 round(max_high_5/next_hfq_open-1,5) mfe5,
 round(min_low_5/next_hfq_open-1,5) mae5,
 round(max_high_10/next_hfq_open-1,5) mfe10,
 round(min_low_10/next_hfq_open-1,5) mae10,
 round(max_high_20/next_hfq_open-1,5) mfe20,
 round(min_low_20/next_hfq_open-1,5) mae20
from fwd
"""
t=time.time()
con.execute(f"COPY ({q}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
print('written',out,os.path.getsize(out)/1e9,'GB',time.time()-t,'s')
print(con.execute(f"select count(*) from read_parquet('{out}')").fetchone())
