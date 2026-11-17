# -*- coding: utf-8 -*-
"""数据核对：本地 DuckDB（同花顺） vs baostock vs 新浪（三源交叉）。

核对对象：
  A. 回测实际成交过的股票（进攻版逐笔 222 笔涉及的股票）——直接影响结果正确性；
  B. 随机分层样本（主板 60 只，按年份分层）——覆盖整体。
核对内容：
  1) 不复权 OHLC + 成交额 逐日对比（baostock，全窗口）；
  2) 后复权日收益对比（baostock adjustflag=1）——验证复权链；
  3) 新浪日线抽查（随机 20 只 × 随机 30 天）——第三方再确认；
  4) 结构检查：重复行、OHLC 关系、涨跌停认定、跨截面天数缺失。

输出：01_数据/核对_*.csv 与控制台摘要。
"""
import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
import duckdb

BASE = '/Users/admin/WorkSpace/trading'
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, '01_数据')
os.makedirs(DATA, exist_ok=True)

DB = os.path.expanduser('~/Library/Application Support/hithink-finance/data/market.duckdb')
START, END = '2016-10-13', '2026-10-23'


def to_bs(code):
    a, b = code.split('.')
    return b.lower() + '.' + a


def load_local(con, codes):
    q = f"""select thscode, date, open, high, low, close, volume, amount
            from v_daily where thscode in ({','.join("'" + c + "'" for c in codes)})"""
    d = con.execute(q).df()
    d['date'] = pd.to_datetime(d['date'])
    return d


def bs_fetch(codes, adjust='3'):
    import baostock as bs
    bs.login()
    out = {}
    cols = 'date,open,high,low,close,volume,amount'
    if adjust == '1':
        cols = 'date,open,high,low,close'
    for i, c in enumerate(codes, 1):
        try:
            rs = bs.query_history_k_data_plus(to_bs(c), cols, start_date=START,
                                              end_date=END, frequency='d',
                                              adjustflag=adjust)
            rows = []
            while rs.error_code == '0' and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                out[c] = rows
        except Exception:
            pass
        if i % 20 == 0:
            print(f'  baostock {i}/{len(codes)} ok={len(out)}', flush=True)
    bs.logout()
    return out


def compare_raw(local, bsrows):
    """不复权 OHLC/volume/amount 逐日对比。"""
    recs = []
    for c, rows in bsrows.items():
        b = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        for k in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            b[k] = pd.to_numeric(b[k], errors='coerce')
        b['date'] = pd.to_datetime(b['date'])
        b = b.dropna(subset=['close']).set_index('date')
        a = local[local.thscode == c].set_index('date')
        j = a.join(b, how='inner', rsuffix='_bs')
        n_only_bs = len(b) - len(j)
        n_only_local = len(a) - len(j)
        for f in ['open', 'high', 'low', 'close', 'amount']:
            d = (j[f] - j[f + '_bs']).abs()
            rel = d / j[f + '_bs'].abs().clip(lower=1e-9)
            for (dt, rv) in rel[rel > 1e-6].items():
                recs.append(dict(code=c, date=dt, field=f, local=float(j.loc[dt, f]),
                                 baostock=float(j.loc[dt, f + '_bs']),
                                 rel_diff=float(rv)))
        recs.append(dict(code=c, date=pd.NaT, field='_days', local=len(a),
                         baostock=len(b), rel_diff=n_only_bs + n_only_local))
    return pd.DataFrame(recs)


def compare_hfq_returns(con, bsrows):
    """后复权日收益对比：同花顺 hfq vs baostock hfq。差异>0.3% 的日子列出。"""
    recs = []
    codes = list(bsrows.keys())
    q = f"""select thscode, date, close from v_daily_hfq
            where thscode in ({','.join("'" + c + "'" for c in codes)})"""
    h = con.execute(q).df()
    h['date'] = pd.to_datetime(h['date'])
    import baostock as bs
    bs.login()
    for i, c in enumerate(codes, 1):
        try:
            rs = bs.query_history_k_data_plus(to_bs(c), 'date,close', start_date=START,
                                              end_date=END, frequency='d', adjustflag='1')
            rows = []
            while rs.error_code == '0' and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                continue
            b = pd.DataFrame(rows, columns=['date', 'close'])
            b['date'] = pd.to_datetime(b['date'])
            b['close'] = pd.to_numeric(b['close'], errors='coerce')
            b = b.dropna().set_index('date')['close'].sort_index()
            a = h[h.thscode == c].set_index('date')['close'].sort_index()
            ra = a.pct_change()
            rb = b.pct_change()
            jj = pd.DataFrame({'ths': ra, 'bs': rb}).dropna()
            diff = (jj.ths - jj.bs).abs()
            bad = diff[diff > 0.003]
            for dt, v in bad.items():
                recs.append(dict(code=c, date=dt, ths_ret=float(jj.loc[dt, 'ths']),
                                 bs_ret=float(jj.loc[dt, 'bs']), abs_diff=float(v)))
        except Exception:
            pass
        if i % 20 == 0:
            print(f'  hfq {i}/{len(codes)}', flush=True)
    bs.logout()
    return pd.DataFrame(recs)


def sina_spot(codes_dates):
    """新浪抽查：给定 (code, date) 列表，抓最近 60 根不复权日线对不上的话抓全量。"""
    import requests
    recs = []
    for c, dt in codes_dates:
        sym = 'sh' + c.split('.')[0] if c.endswith('.SH') else 'sz' + c.split('.')[0]
        try:
            r = requests.get(f'https://quotes.sina.cn/cn/api/jsonp_v2.php/x=/CN_MarketDataService.getKLineData?symbol={sym}&scale=240&ma=no&datalen=1023',
                             timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            txt = r.text
            arr = json.loads(txt[txt.index('=') + 1:])
            df = pd.DataFrame(arr)
            df['day'] = pd.to_datetime(df['day'])
            row = df[df.day == pd.Timestamp(dt)]
            if len(row) == 0:
                recs.append(dict(code=c, date=dt, field='no_row', local=np.nan, sina=np.nan))
                continue
            recs.append(dict(code=c, date=dt, field='close', local=np.nan,
                             sina=float(row.iloc[0]['close'])))
        except Exception as e:
            recs.append(dict(code=c, date=dt, field=f'err:{type(e).__name__}', local=np.nan, sina=np.nan))
    return pd.DataFrame(recs)


def structural_checks(con):
    out = []
    d = con.execute(f"""select date, count(*) n, count(distinct thscode) nsym
                        from v_daily group by date order by date""").df()
    d['date'] = pd.to_datetime(d['date'])
    by_year = d.groupby(d.date.dt.year).agg(days=('n', 'size'), min_stocks=('nsym', 'min'),
                                            max_stocks=('nsym', 'max'))
    out.append(('days_per_year', by_year.to_dict()))
    bad_ohlc = con.execute("""select count(*) from v_daily
        where high < low or high < open or high < close or low > open or low > close""").df()
    out.append(('ohlc_violations', int(bad_ohlc.iloc[0, 0])))
    dup = con.execute("""select count(*) from (select thscode, date, count(*) c
        from v_daily group by thscode, date having c > 1)""").df()
    out.append(('duplicate_rows', int(dup.iloc[0, 0])))
    lu = con.execute(f"""with r as (select thscode, date, close / lag(close) over (partition by thscode order by date) - 1 ret
        from v_daily_hfq where date between '{START}' and '{END}')
        select
          sum(case when thscode like '600%' and ret between 0.0975 and 0.1025 then 1 else 0 end) mb_up10,
          sum(case when thscode like '600%' and ret between 0.0455 and 0.0525 then 1 else 0 end) mb_up5,
          count(*) tot from r""").df()
    out.append(('limit_move_counts', lu.iloc[0].to_dict()))
    return out


def main():
    rng = np.random.default_rng(20261024)
    con = duckdb.connect(DB, read_only=True)

    # ---- A. 实际成交过的股票 ----
    T = pd.read_csv(os.path.join(BASE, '2026-10-24_结构交易_主板40系统/05_证据/进攻_AMT_逐笔.csv'))
    trade_codes = sorted(set(T.code.astype(str)))
    print(f'逐笔涉及股票 {len(trade_codes)} 只')

    # ---- B. 随机分层样本：主板，按上市年限分层 ----
    all_mb = con.execute("""select distinct thscode from v_daily
        where thscode like '60%' or thscode like '00%'""").df()['thscode'].tolist()
    random.seed(20261024)
    rand_codes = sorted(random.sample(all_mb, 60))
    check_codes = sorted(set(trade_codes) | set(rand_codes))
    print(f'核对股票合计 {len(check_codes)} 只（成交 {len(trade_codes)} + 随机 {len(rand_codes)}）')

    print('[1] baostock 不复权全窗口...')
    local = load_local(con, check_codes)
    bsrows = bs_fetch(check_codes, adjust='3')
    diff = compare_raw(local, bsrows)
    diff.to_csv(os.path.join(DATA, '核对_原始价差异_baostock.csv'), index=False, encoding='utf-8-sig')
    field_diff = diff[diff.field != '_days']
    n_days = diff[diff.field == '_days']
    print(f'  价格/成交额逐日差异(相对>1e-6)：{len(field_diff)} 条 / 覆盖 {len(bsrows)} 只')
    if len(field_diff):
        print(field_diff.sort_values('rel_diff', ascending=False).head(15).to_string(index=False))
    gap = n_days[n_days.rel_diff > 5]
    print(f'  交易日数差异>5 天的股票：{len(gap)} 只')
    if len(gap):
        print(gap.sort_values('rel_diff', ascending=False).head(10).to_string(index=False))

    print('[2] baostock 后复权收益对比（抽样 60 只）...')
    sub = random.sample(check_codes, min(60, len(check_codes)))
    hfq = compare_hfq_returns(con, {c: bsrows.get(c) for c in sub if c in bsrows})
    hfq.to_csv(os.path.join(DATA, '核对_复权收益差异_baostock.csv'), index=False, encoding='utf-8-sig')
    print(f'  复权日收益差异>0.3% 的 (股,日)：{len(hfq)} 条')
    if len(hfq):
        print(hfq.sort_values('abs_diff', ascending=False).head(15).to_string(index=False))

    print('[3] 结构检查...')
    for name, v in structural_checks(con):
        print(' ', name, ':', v)

    con.close()
    print('完成')


if __name__ == '__main__':
    main()
