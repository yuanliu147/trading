# -*- coding: utf-8 -*-
"""拉取 baostock 后复权日线前缀段（2015-01-01→2016-10-12），用于拼接真·2016 全年面板。

- 股票清单 = 面板 v2 全部代码（5,681 只，含退市缓存股；baostock 缺的股票该段留空）；
- adjustflag=1（后复权），字段 date/open/high/low/close/volume/amount；
- 断点续传：每 100 只落盘一次 01_数据/baostock前缀_partial.pkl；
- 节流：每股间隔 0.55s（baostock 限流保守值），失败重试 2 次。
"""
import os
import time
import pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, '01_数据')
PANEL = os.path.join(DATA, '全A面板_20161013_20261023.pkl')
PARTIAL = os.path.join(DATA, 'baostock前缀_partial.pkl')
OUT = os.path.join(DATA, 'baostock前缀_2015_2016.pkl')

START, END = '2015-01-01', '2016-10-13'


def to_bs(c):
    a, b = c.split('.')
    return b.lower() + '.' + a


def main():
    P = pickle.load(open(PANEL, 'rb'))
    codes = [str(c) for c in P['meta']['codes']]
    print(f'目标 {len(codes)} 只，窗口 {START}→{END}', flush=True)

    got = {}
    if os.path.exists(PARTIAL):
        got = pickle.load(open(PARTIAL, 'rb'))
        print(f'续传：已有 {len(got)} 只', flush=True)

    import baostock as bs
    bs.login()
    t0 = time.time()
    fails = []
    for i, c in enumerate(codes, 1):
        if c in got:
            continue
        rows = None
        for attempt in range(3):
            try:
                rs = bs.query_history_k_data_plus(
                    to_bs(c), 'date,open,high,low,close,volume,amount',
                    start_date=START, end_date=END, frequency='d', adjustflag='1')
                rr = []
                while rs.error_code == '0' and rs.next():
                    rr.append(rs.get_row_data())
                if rr:
                    df = pd.DataFrame(rr, columns=['date', 'open', 'high', 'low', 'close',
                                                    'volume', 'amount'])
                    for k in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                        df[k] = pd.to_numeric(df[k].replace('', 'nan'), errors='coerce')
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.dropna(subset=['close'])
                    if len(df):
                        rows = df
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if rows is not None:
            got[c] = rows
        else:
            fails.append(c)
        time.sleep(0.55)
        if i % 100 == 0:
            pickle.dump(got, open(PARTIAL, 'wb'), protocol=4)
            el = time.time() - t0
            done = sum(1 for x in codes[:i] if x in got)
            print(f'{i}/{len(codes)} 有效{len(got)} 失败{len(fails)} 用时{el/60:.1f}分', flush=True)
    bs.logout()
    pickle.dump(got, open(OUT, 'wb'), protocol=4)
    print(f'完成：有效 {len(got)} 只，失败 {len(fails)} 只', flush=True)
    if fails:
        pickle.dump(fails, open(os.path.join(DATA, 'baostock前缀_失败清单.pkl'), 'wb'))


if __name__ == '__main__':
    main()
