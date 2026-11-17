# -*- coding: utf-8 -*-
"""收尾核对：
  1) 完备性：对两轮核对过的全部股票，验证 本地交易日 ⊇ baostock 有效交易日(成交量>0)；
  2) 新浪第三方抽查（修复 dtype）。
输出：01_数据/核对_交易日完备性.csv、核对_新浪抽查.csv
"""
import os
import time
import json
import numpy as np
import pandas as pd
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, '01_数据')
DB = os.path.expanduser('~/Library/Application Support/hithink-finance/data/market.duckdb')
START, END = '2016-10-13', '2026-10-23'


def to_bs(c):
    a, b = c.split('.')
    return b.lower() + '.' + a


def main():
    d1 = pd.read_csv(os.path.join(DATA, '核对_原始价差异_baostock.csv'))
    d2 = pd.read_csv(os.path.join(DATA, '核对_原始价差异_baostock_补拉.csv'))
    codes = sorted(set(d1[d1.field == '_days'].code.astype(str))
                   | set(d2[d2.field == '_days'].code.astype(str)))
    print(f'完备性检查股票数: {len(codes)}')

    con = duckdb.connect(DB, read_only=True)
    import baostock as bs
    bs.login()
    rows = []
    for i, c in enumerate(codes, 1):
        a = con.execute(f"select date from v_daily where thscode='{c}'").df()
        loc = set(pd.to_datetime(a.date))
        try:
            rs = bs.query_history_k_data_plus(to_bs(c), 'date,volume', start_date=START,
                                              end_date=END, frequency='d', adjustflag='3')
            rr = []
            while rs.error_code == '0' and rs.next():
                rr.append(rs.get_row_data())
            b = pd.DataFrame(rr, columns=['date', 'volume'])
            b['volume'] = pd.to_numeric(b.volume.replace('', 'nan'), errors='coerce')
            b['date'] = pd.to_datetime(b.date)
            eff = set(b[(b.volume > 0) & b.volume.notna()].date)
            missing = sorted(eff - loc)
            rows.append(dict(code=c, 本地天数=len(loc), bs有效天数=len(eff),
                             本地缺失有效日=len(missing),
                             缺失样例=';'.join(str(x.date()) for x in missing[:3])))
        except Exception as e:
            rows.append(dict(code=c, 本地天数=len(loc), bs有效天数=np.nan,
                             本地缺失有效日=np.nan, 缺失样例=f'err:{type(e).__name__}'))
        time.sleep(0.9)
        if i % 20 == 0:
            print(f'  {i}/{len(codes)}', flush=True)
    bs.logout()
    con.close()
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(DATA, '核对_交易日完备性.csv'), index=False, encoding='utf-8-sig')
    bad = M[(M.本地缺失有效日.fillna(0) > 0) | M.本地缺失有效日.isna()]
    print('本地缺失 baostock 有效交易日的股票数:', len(bad))
    if len(bad):
        print(bad.head(10).to_string(index=False))

    # ---- 新浪抽查（修复 dtype）----
    T = pd.read_csv('/Users/admin/WorkSpace/trading/2026-10-24_结构交易_主板40系统/05_证据/进攻_AMT_逐笔.csv')
    trade_codes = sorted(set(T.code.astype(str)))
    import requests
    rng = np.random.default_rng(7)
    sina_codes = [c for c in trade_codes if c.endswith('.SZ')]
    picks = list(rng.choice(len(sina_codes), size=12, replace=False))
    out = []
    for k in picks:
        c = sina_codes[k]
        sym = 'sz' + c.split('.')[0]
        try:
            r = requests.get(f'https://quotes.sina.cn/cn/api/jsonp_v2.php/x=/CN_MarketDataService.getKLineData?symbol={sym}&scale=240&ma=no&datalen=1023',
                             timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            txt = r.text
            arr = json.loads(txt[txt.index('=') + 1:])
            df = pd.DataFrame(arr)
            df['day'] = pd.to_datetime(df['day'])
            row = df.iloc[int(rng.integers(0, len(df)))]
            con2 = duckdb.connect(DB, read_only=True)
            loc = con2.execute(f"""select close from v_daily
                where thscode='{c}' and date='{str(row.day.date())}'""").df()
            con2.close()
            lc = float(loc.close.iloc[0]) if len(loc) else np.nan
            sc = float(row.close)
            out.append(dict(code=c, date=str(row.day.date()), sina_close=sc, local_close=lc,
                            diff=(lc - sc) if np.isfinite(lc) else np.nan))
        except Exception as e:
            out.append(dict(code=c, date='err', sina_close=np.nan, local_close=np.nan,
                            diff=np.nan))
        time.sleep(0.6)
    s = pd.DataFrame(out)
    s['rel'] = s['diff'].abs() / s.sina_close
    s.to_csv(os.path.join(DATA, '核对_新浪抽查.csv'), index=False, encoding='utf-8-sig')
    print('新浪抽查(不复权 close):')
    print(s.to_string(index=False))
    print('超1e-4相对差异数:', int((s.rel > 1e-4).sum()))


if __name__ == '__main__':
    main()
