# -*- coding: utf-8 -*-
"""
构建 2016-10-13～2026-10-13 全 A 跌停事件表。

核心口径：
  跌停价 = round(调整前收 × (1-限制比例), 2)
  调整前收 = 当日未复权收盘 × 前一日后复权收盘 ÷ 当日后复权收盘
  收盘跌停 = 收盘 = 最低 = 跌停价
  盘中触及 = 最低 = 跌停价
板块比例：主板 10%，创业板 2020-08-24 前 10%/后 20%，科创板 20%，北交所 30%。
ST 的 5% 单独标记，不混入主口径。

输出：
  01_数据/跌停事件表_主.parquet     DuckDB 全量（5,574 只）
  01_数据/跌停事件表_补充.parquet   退市缓存 102 只，近似口径，approx=1
"""
import os, time, pickle, warnings
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
OUT = os.path.join(BASE, '2026-10-22_跌停前特征_全A', '01_数据')
DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:7.1f}s] {msg}', flush=True)

con = duckdb.connect(DB, read_only=True)

# ---------------- 1. DuckDB 主体 ----------------
log('构建 DuckDB 事件表')
sql = r"""
create or replace temp table ev as
with x as (
  select d.thscode, d.date, d.open, d.high, d.low, d.close, d.volume, d.amount,
         h.close as hfq,
         lag(h.close) over w as hfq_prev,
         lag(d.close) over w as raw_prev,
         lag(d.volume) over w as vol_prev,
         row_number() over w as rn
  from v_daily d
  join v_daily_hfq h on d.thscode = h.thscode and d.date = h.date
  where d.date >= date '2016-10-13' and d.date <= date '2026-10-13'
  window w as (partition by d.thscode order by d.date)
), y as (
  select *,
         close * hfq_prev / nullif(hfq, 0) as adj_prev,
         case
           when thscode like '300%' or thscode like '301%' then '创业板'
           when thscode like '688%' then '科创板'
           when thscode like '8%' or thscode like '4%' or thscode like '920%' then '北交所'
           else '主板'
         end as board
  from x
  where hfq_prev is not null and hfq > 0 and raw_prev > 0 and close > 0
), z as (
  select *,
    round(adj_prev * 0.95, 2) as lim_dn5,
    round(adj_prev * 0.90, 2) as lim_dn10,
    round(adj_prev * 0.80, 2) as lim_dn20,
    round(adj_prev * 0.70, 2) as lim_dn30,
    round(adj_prev * 1.05, 2) as lim_up5,
    round(adj_prev * 1.10, 2) as lim_up10,
    round(adj_prev * 1.20, 2) as lim_up20,
    round(adj_prev * 1.30, 2) as lim_up30,
    close / adj_prev - 1 as ret_raw
  from y
)
select
  thscode, date, board, open, high, low, close, volume, amount,
  adj_prev, ret_raw, lim_dn5, lim_dn10, lim_dn20, lim_dn30,
  lim_up5, lim_up10, lim_up20, lim_up30,
  (abs(close - lim_dn5)  < 0.005 and abs(low - close) < 0.001) as ld_seal5,
  (abs(close - lim_dn10) < 0.005 and abs(low - close) < 0.001) as ld_seal10,
  (abs(close - lim_dn20) < 0.005 and abs(low - close) < 0.001) as ld_seal20,
  (abs(close - lim_dn30) < 0.005 and abs(low - close) < 0.001) as ld_seal30,
  (abs(low - lim_dn5)  < 0.005) as ld_touch5,
  (abs(low - lim_dn10) < 0.005) as ld_touch10,
  (abs(low - lim_dn20) < 0.005) as ld_touch20,
  (abs(low - lim_dn30) < 0.005) as ld_touch30,
  (abs(close - lim_up5)  < 0.005 and abs(high - close) < 0.001) as lu_seal5,
  (abs(close - lim_up10) < 0.005 and abs(high - close) < 0.001) as lu_seal10,
  (abs(close - lim_up20) < 0.005 and abs(high - close) < 0.001) as lu_seal20,
  (abs(close - lim_up30) < 0.005 and abs(high - close) < 0.001) as lu_seal30,
  (abs(high - lim_up5)  < 0.005) as lu_touch5,
  (abs(high - lim_up10) < 0.005) as lu_touch10,
  (abs(high - lim_up20) < 0.005) as lu_touch20,
  (abs(high - lim_up30) < 0.005) as lu_touch30,
  case
    when board = '主板' and ld_seal10 then 10
    when board = '创业板' and date < date '2020-08-24' and ld_seal10 then 10
    when board = '创业板' and date >= date '2020-08-24' and ld_seal20 then 20
    when board = '科创板' and ld_seal20 then 20
    when board = '北交所' and ld_seal30 then 30
  end as ld_ratio,
  case
    when board = '主板' and ld_touch10 then 10
    when board = '创业板' and date < date '2020-08-24' and ld_touch10 then 10
    when board = '创业板' and date >= date '2020-08-24' and ld_touch20 then 20
    when board = '科创板' and ld_touch20 then 20
    when board = '北交所' and ld_touch30 then 30
  end as ldt_ratio,
  case
    when board = '主板' and lu_seal10 then 10
    when board = '创业板' and date < date '2020-08-24' and lu_seal10 then 10
    when board = '创业板' and date >= date '2020-08-24' and lu_seal20 then 20
    when board = '科创板' and lu_seal20 then 20
    when board = '北交所' and lu_seal30 then 30
  end as lu_ratio,
  case
    when board = '主板' and lu_touch10 then 10
    when board = '创业板' and date < date '2020-08-24' and lu_touch10 then 10
    when board = '创业板' and date >= date '2020-08-24' and lu_touch20 then 20
    when board = '科创板' and lu_touch20 then 20
    when board = '北交所' and lu_touch30 then 30
  end as lut_ratio
from z
"""
con.execute(sql)
log('事件表建立完成，导出 parquet')
out_main = os.path.join(OUT, '跌停事件表_主.parquet')
con.execute(f"copy (select *, 0::tinyint as approx from ev) to '{out_main}' (format parquet, compression zstd)")
n = con.execute("select count(*) from ev").fetchone()[0]
log(f'主表 {n:,} 行 -> {out_main} ({os.path.getsize(out_main)/1e6:.0f} MB)')
# 统计
stat = con.execute("""
select board, count(*) n,
 sum(case when ld_ratio is not null then 1 else 0 end) ld_seal,
 sum(case when ldt_ratio is not null then 1 else 0 end) ld_touch,
 sum(case when lu_ratio is not null then 1 else 0 end) lu_seal,
 sum(case when lut_ratio is not null then 1 else 0 end) lu_touch
from ev group by 1 order by 1
""").df()
print(stat.to_string(index=False))
con.close()

# ---------------- 2. 退市缓存补充 ----------------
log('构建退市缓存补充事件表')
cache = pickle.load(open(os.path.join(BASE, '2026-10-18_成交量融合_2010-2025/01_数据/daily_cache_2010_2025_hfq.pkl'), 'rb'))['stocks']
main_codes = set(pd.read_parquet(out_main, columns=['thscode']).thscode.unique())
def cv(k):
    a, b = k.split('.')
    return b + '.' + a.upper()
rows = []
for k, x in cache.items():
    code = cv(k)
    if code in main_codes:
        continue
    x = x.copy()
    x['date'] = pd.to_datetime(x['date'])
    x = x[(x.date >= '2016-10-13') & (x.date <= '2026-10-13')]
    if len(x) == 0:
        continue
    x = x.sort_values('date').reset_index(drop=True)
    if 'pctChg' not in x.columns:
        continue
    r = x['pctChg'].astype(float) / 100.0
    prev = x['close'] / (1 + r)
    low_ret = x['low'] / prev - 1
    high_ret = x['high'] / prev - 1
    board = ('创业板' if code.startswith(('300', '301')) else
             '科创板' if code.startswith('688') else
             '北交所' if code.startswith(('8', '4', '920')) else '主板')
    ratio = np.where((board == '创业板') & (x.date >= '2020-08-24'), 0.20,
             np.where(board == '创业板', 0.10,
             np.where(board == '科创板', 0.20,
             np.where(board == '北交所', 0.30, 0.10))))
    ld_seal = (np.abs(r + ratio) <= 0.004) & (np.abs(x.low - x.close) < 0.006)
    ld_touch = np.abs(low_ret + ratio) <= 0.004
    lu_seal = (np.abs(r - ratio) <= 0.004) & (np.abs(x.high - x.close) < 0.006)
    lu_touch = np.abs(high_ret - ratio) <= 0.004
    out = pd.DataFrame({
        'thscode': code, 'date': x.date, 'board': board,
        'open': x.open, 'high': x.high, 'low': x.low, 'close': x.close,
        'volume': x.volume, 'amount': x.amount,
        'adj_prev': prev, 'ret_raw': r,
        'lim_dn5': np.nan, 'lim_dn10': np.nan, 'lim_dn20': np.nan, 'lim_dn30': np.nan,
        'lim_up5': np.nan, 'lim_up10': np.nan, 'lim_up20': np.nan, 'lim_up30': np.nan,
        'ld_seal5': False, 'ld_seal10': ld_seal & (ratio == 0.10),
        'ld_seal20': ld_seal & (ratio == 0.20), 'ld_seal30': ld_seal & (ratio == 0.30),
        'ld_touch5': False, 'ld_touch10': ld_touch & (ratio == 0.10),
        'ld_touch20': ld_touch & (ratio == 0.20), 'ld_touch30': ld_touch & (ratio == 0.30),
        'lu_seal5': False, 'lu_seal10': lu_seal & (ratio == 0.10),
        'lu_seal20': lu_seal & (ratio == 0.20), 'lu_seal30': lu_seal & (ratio == 0.30),
        'lu_touch5': False, 'lu_touch10': lu_touch & (ratio == 0.10),
        'lu_touch20': lu_touch & (ratio == 0.20), 'lu_touch30': lu_touch & (ratio == 0.30),
        'ld_ratio': np.where(ld_seal, (ratio * 100).astype(int), np.nan),
        'ldt_ratio': np.where(ld_touch, (ratio * 100).astype(int), np.nan),
        'lu_ratio': np.where(lu_seal, (ratio * 100).astype(int), np.nan),
        'lut_ratio': np.where(lu_touch, (ratio * 100).astype(int), np.nan),
    })
    rows.append(out)
if rows:
    sup = pd.concat(rows, ignore_index=True)
    sup['approx'] = 1
    out_sup = os.path.join(OUT, '跌停事件表_补充.parquet')
    sup.to_parquet(out_sup, index=False)
    log(f'补充表 {len(sup):,} 行, {sup.thscode.nunique()} 只 -> {out_sup}')
    print(sup.groupby('board').agg(n=('thscode','size'), ld=('ld_ratio','count'), lu=('lu_ratio','count')).to_string())
else:
    log('无补充股票')
log('完成')
