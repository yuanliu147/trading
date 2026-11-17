"""只读2010-2020层级研究数据，2008-2009仅预热。"""
from pathlib import Path
import hashlib
import json
import duckdb
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OLD = PROJECT / '01_数据/面板_2008预热_2020.parquet'
SOURCE = Path('/Users/admin/WorkSpace/trading/2026-10-18_成交量融合_2010-2025/01_数据')
FROZEN = Path('/Users/admin/WorkSpace/trading/trading_趋势结构/01_数据/frozen')
START, END = '2008-07-01', '2020-12-31'

def main():
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    # 只摄入2020前的分类记录；不回填尚未存在的行业。
    hist = pd.read_csv(SOURCE / '申万行业分类_时点.csv', parse_dates=['start_date'])
    hist = hist[hist.start_date.le(END)].copy()
    hist['symbol'] = hist.code.str[3:] + '.' + hist.code.str[:2].str.upper()
    hist = hist.rename(columns={'start_date': '有效起日', 'industry': '行业', 'sw_index': '行业码'})
    hist = hist[['symbol', '有效起日', '行业', '行业码']].drop_duplicates(['symbol', '有效起日'], keep='last')
    hist.to_parquet(ROOT / '数据/行业有效记录.parquet', index=False)
    con.register('history', hist)
    con.execute(f"""CREATE TABLE raw AS
      SELECT *, lag(date) OVER (PARTITION BY symbol ORDER BY date) AS 前日,
      lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) AS 前收
      FROM read_parquet('{OLD}') WHERE date BETWEEN DATE '{START}' AND DATE '{END}'""")
    con.execute(f"""CREATE TABLE calendar AS
      SELECT date,lag(date) OVER (ORDER BY date) AS 市场前日
      FROM read_parquet('{FROZEN}/index_daily/year=*/*.parquet')
      WHERE symbol='000300.SH' AND date BETWEEN DATE '{START}' AND DATE '{END}'""")
    con.execute("""CREATE TABLE mapped AS
      SELECT r.*,
      CASE WHEN r.前日 IS NOT NULL AND h.有效起日<=r.前日 THEN h.行业 END AS 行业,
      CASE WHEN r.前日 IS NOT NULL AND h.有效起日<=r.前日 THEN h.行业码 END AS 行业码,
      CASE WHEN r.前日 IS NOT NULL AND h.有效起日<=r.前日 THEN h.有效起日 END AS 有效起日,
      hfq_close/前收-1 AS 日收益,
      hfq_open/前收-1 AS 隔夜收益,
      hfq_close/hfq_open-1 AS 日内收益
      FROM raw r ASOF LEFT JOIN history h ON r.symbol=h.symbol AND r.前日>=h.有效起日""")
    # COUNT与所有收益均值必须使用同一有效成分集；缺股票市场日不视作单日收益。
    con.execute("""CREATE TABLE valid AS
      SELECT m.* FROM mapped m JOIN calendar c USING(date)
      WHERE m.前日=c.市场前日 AND td_since_list>=60
       AND 前收>0 AND hfq_open>0 AND hfq_close>0
       AND isfinite(前收) AND isfinite(hfq_open) AND isfinite(hfq_close)
       AND isfinite(日收益) AND isfinite(隔夜收益) AND isfinite(日内收益)""")
    con.execute("""CREATE TABLE sectors AS
      SELECT date, 行业码, 行业, count(*) AS 成分数,
       avg(日收益) AS 日收益, avg(隔夜收益) AS 隔夜收益,
       1.0 AS 有效价格率
      FROM valid WHERE 行业 IS NOT NULL
      GROUP BY date, 行业码, 行业""")
    # 残差用同日同行业留一基准，停牌仍保留零收益。
    con.execute(f"""COPY (
      SELECT m.symbol,m.date,m.board,m.td_since_list,m.hfq_open,m.hfq_close,
      m.open,m.high,m.low,m.close,m.preclose,m.amount,m.trade_status,m.is_st,
      m.is_suspended,m.is_limit_up_open,m.is_limit_down_open,m.日收益,
      m.隔夜收益,m.日内收益,m.行业,m.行业码,m.有效起日,
      CASE WHEN v.symbol IS NOT NULL AND s.成分数>10 AND m.日收益>-1
        AND (s.日收益*s.成分数-m.日收益)/(s.成分数-1)>-1
        THEN ln(1+m.日收益)-ln(1+(s.日收益*s.成分数-m.日收益)/(s.成分数-1)) END AS 行业残差,
      CASE WHEN v.symbol IS NOT NULL AND s.成分数>10
        THEN (s.日收益*s.成分数-m.日收益)/(s.成分数-1) END AS 留一行业收益
      FROM mapped m LEFT JOIN sectors s ON m.date=s.date AND m.行业码=s.行业码
      LEFT JOIN valid v ON m.symbol=v.symbol AND m.date=v.date
    ) TO '{ROOT}/数据/个股与行业.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    con.execute(f"COPY sectors TO '{ROOT}/数据/行业等权日收益.parquet' (FORMAT PARQUET)")
    cov = con.execute("""SELECT year(date) 年,count(*) 股票日,
      sum(CASE WHEN 行业 IS NOT NULL THEN 1 ELSE 0 END) 有行业股票日,
      count(DISTINCT symbol) 股票数,count(DISTINCT 行业) 行业数
      FROM mapped WHERE date>=DATE '2010-01-01' GROUP BY 年 ORDER BY 年""").fetch_df()
    cov['归属覆盖率'] = cov['有行业股票日']/cov['股票日']
    cov.to_csv(ROOT / '结果/行业覆盖率.csv',index=False,encoding='utf-8-sig')
    # 同一涨幅区间的等权合成开收盘基准；未计日再平衡费用，明确是合成非可投资基准。
    ew = con.execute("""SELECT date,avg(日收益) r,avg(隔夜收益) g,count(*) n
      FROM valid GROUP BY date ORDER BY date""").fetch_df()
    ew['close'] = (1+ew.r).cumprod()
    ew['open'] = ew.close.shift(1).fillna(1)*(1+ew.g)
    ew['symbol'], ew['name'], ew['kind'] = 'EW_ALL','全A等权合成','全A合成'
    ew['live_start'] = pd.Timestamp('2010-01-01')
    ew[['date','symbol','name','kind','open','close','live_start']].to_parquet(ROOT/'数据/全A等权开收盘.parquet',index=False)
    idx = con.execute(f"""SELECT symbol,date,open,close
      FROM read_parquet('{FROZEN}/index_daily/year=*/*.parquet')
      WHERE date BETWEEN DATE '{START}' AND DATE '{END}'
      AND symbol IN ('000001.SH','000016.SH','000300.SH','000905.SH','000852.SH','399006.SZ')""").fetch_df()
    names = {'000001.SH':'上证综指','000016.SH':'上证50','000300.SH':'沪深300',
             '000905.SH':'中证500','000852.SH':'中证1000','399006.SZ':'创业板指'}
    idx['name'] = idx.symbol.map(names); idx['kind'] = '宽基指数'
    live = {'000852.SH':'2014-10-17', '399006.SZ':'2010-06-01'}
    idx['live_start'] = pd.to_datetime(idx.symbol.map(live).fillna('2010-01-01'))
    sw = pd.read_csv(SOURCE/'申万一级行业指数.csv',parse_dates=['date'],dtype={'sw_index':str})
    sw = sw[sw.date.between(START,END)].copy()
    # 2021新增行业不把历史回算当实盘可知。
    sw = sw[sw.sw_index.astype(int)<801950].copy()
    sw['symbol'] = 'SW'+sw.sw_index; sw['name'] = sw.industry; sw['kind'] = '行业价格指数'
    sw['live_start'] = pd.Timestamp('2015-01-01')
    assets = pd.concat([idx,sw[['symbol','date','name','kind','open','close','live_start']],
      ew[['symbol','date','name','kind','open','close','live_start']]],ignore_index=True)
    assert assets[['symbol','date']].duplicated().sum()==0
    assert assets.date.max()<=pd.Timestamp(END)
    assets = assets.sort_values(['symbol','date']).reset_index(drop=True)
    assets.to_parquet(ROOT/'数据/资产开收盘.parquet',index=False,compression='zstd')
    assets.groupby(['symbol','name','kind'],as_index=False).agg(
      数据首日=('date','min'),数据末日=('date','max'),行数=('date','size'),主评价首日=('live_start','max')
    ).to_csv(ROOT/'结果/资产覆盖.csv',index=False,encoding='utf-8-sig')
    sectors = con.execute('SELECT * FROM sectors ORDER BY 行业码,date').fetch_df()
    parts=[]
    for code,g in sectors.groupby('行业码'):
        g=g.sort_values('date').copy()
        g['close']=(1+g['日收益']).cumprod()
        g['open']=g.close.shift(1).fillna(1)*(1+g['隔夜收益'])
        g['symbol']='EW_SW'+str(code);g['name']=g['行业'];g['kind']='行业等权合成'
        g['live_start']=pd.Timestamp('2015-01-01')
        parts.append(g)
    synth=pd.concat(parts,ignore_index=True)
    synth.to_parquet(ROOT/'数据/行业合成开收盘.parquet',index=False,compression='zstd')
    count=con.execute('SELECT count(*),count(DISTINCT symbol),min(date),max(date) FROM mapped').fetchone()
    invalid=con.execute('SELECT count(*) FROM mapped WHERE 有效起日>前日 OR (前日 IS NULL AND 有效起日 IS NOT NULL)').fetchone()[0]
    audit={'日期上限':END,'个股行数':count[0],'个股证券数':count[1],'首日':str(count[2]),'末日':str(count[3]),
      '行业映射使用未来记录行数':invalid,'行业史读取':len(hist),'资产数':assets.symbol.nunique(),
      '官方行业数':sw.symbol.nunique(),'宽基数':idx.symbol.nunique(),'归属覆盖':cov.to_dict('records'),
      '限制':['行业历史表已将旧前缀映射到新版名称，不能保证2014前完整；主窗2015后',
      '退市证券可能不在申万回溯表中，缺失不回填','官方指数多为价格指数，未含股息、ETF追踪和费率',
      '全A与行业等权是日再平衡合成基准，未计再平衡费，不作为可投资收益',
      '不将宽基发布之前回算路径作为主绩效'],
      '源文件哈希':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'申万行业分类_时点.csv',SOURCE/'申万一级行业指数.csv',ROOT/'证据/研究设计冻结.json']}}
    assert invalid==0
    (ROOT/'证据/数据核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str))
    print('个股',count,'资产',assets.symbol.nunique(),'未来归属',invalid,flush=True)
    print(cov.to_string(index=False))

if __name__=='__main__':
    main()
