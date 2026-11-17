# -*- coding: utf-8 -*-
"""补拉首轮被 baostock 限流挡掉的成交股（加 1.2s 间隔），并用新浪做第三方抽查。

输出：01_数据/核对_原始价差异_baostock_补拉.csv、核对_新浪抽查.csv
"""
import os
import time
import json
import numpy as np
import pandas as pd
import duckdb

BASE = '/Users/admin/WorkSpace/trading'
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, '01_数据')
DB = os.path.expanduser('~/Library/Application Support/hithink-finance/data/market.duckdb')
START, END = '2016-10-13', '2026-10-23'


def to_bs(c):
    a, b = c.split('.')
    return b.lower() + '.' + a


def main():
    T = pd.read_csv(os.path.join(BASE, '2026-10-24_结构交易_主板40系统/05_证据/进攻_AMT_逐笔.csv'))
    trade_codes = sorted(set(T.code.astype(str)))
    done1 = set()
    f1 = os.path.join(DATA, '核对_原始价差异_baostock.csv')
    if os.path.exists(f1):
        d1 = pd.read_csv(f1)
        done1 = set(d1[d1.field == '_days'].code.astype(str))
    todo = [c for c in trade_codes if c not in done1]
    print(f'成交股 {len(trade_codes)}，首轮已核对 {len(done1 & set(trade_codes))}，补拉 {len(todo)}')

    import baostock as bs
    bs.login()
    con = duckdb.connect(DB, read_only=True)
    q = f"""select thscode, date, open, high, low, close, amount
            from v_daily where thscode in ({','.join("'" + c + "'" for c in todo)})"""
    local = con.execute(q).df()
    con.close()
    local['date'] = pd.to_datetime(local['date'])

    recs = []
    ok = 0
    for i, c in enumerate(todo, 1):
        try:
            rs = bs.query_history_k_data_plus(to_bs(c), 'date,open,high,low,close,amount',
                                              start_date=START, end_date=END,
                                              frequency='d', adjustflag='3')
            rows = []
            while rs.error_code == '0' and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                recs.append(dict(code=c, date=pd.NaT, field='_empty', local=np.nan,
                                 baostock=np.nan, rel_diff=np.nan))
                time.sleep(1.2)
                continue
            b = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'amount'])
            for k in ['open', 'high', 'low', 'close', 'amount']:
                b[k] = pd.to_numeric(b[k], errors='coerce')
            b['date'] = pd.to_datetime(b['date'])
            b = b.dropna(subset=['close'])
            a = local[local.thscode == c].set_index('date')
            j = a.join(b.set_index('date'), how='inner', rsuffix='_bs')
            for f in ['open', 'high', 'low', 'close', 'amount']:
                rel = (j[f] - j[f + '_bs']).abs() / j[f + '_bs'].abs().clip(lower=1e-9)
                for dt, rv in rel[rel > 1e-6].items():
                    recs.append(dict(code=c, date=dt, field=f, local=float(j.loc[dt, f]),
                                     baostock=float(j.loc[dt, f + '_bs']), rel_diff=float(rv)))
            recs.append(dict(code=c, date=pd.NaT, field='_days', local=len(a),
                             baostock=len(b), rel_diff=len(b) - len(j)))
            ok += 1
        except Exception as e:
            recs.append(dict(code=c, date=pd.NaT, field=f'_err:{type(e).__name__}',
                             local=np.nan, baostock=np.nan, rel_diff=np.nan))
        time.sleep(1.2)
        if i % 10 == 0:
            print(f'  {i}/{len(todo)} ok={ok}', flush=True)
    bs.logout()
    d = pd.DataFrame(recs)
    d.to_csv(os.path.join(DATA, '核对_原始价差异_baostock_补拉.csv'), index=False,
             encoding='utf-8-sig')
    fd = d[d.field.str.startswith('_') == False] if len(d) else d
    print(f'补拉完成 ok={ok}/{len(todo)}；价格/成交额差异条数: {len(fd)}')
    if len(fd):
        print(fd.sort_values("rel_diff", ascending=False).head(10).to_string(index=False))

    # 新浪第三方抽查：10 只 × 最近交易日 + 随机历史日
    import requests
    rng = np.random.default_rng(20261024)
    sina_codes = [c for c in trade_codes if c.endswith('.SZ')][:10]
    out = []
    for c in sina_codes:
        sym = 'sz' + c.split('.')[0]
        try:
            r = requests.get(f'https://quotes.sina.cn/cn/api/jsonp_v2.php/x=/CN_MarketDataService.getKLineData?symbol={sym}&scale=240&ma=no&datalen=1023',
                             timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            txt = r.text
            arr = json.loads(txt[txt.index('=') + 1:])
            df = pd.DataFrame(arr)
            df['day'] = pd.to_datetime(df['day'])
            for _ in range(3):
                row = df.iloc[rng.integers(0, len(df))]
                con2 = duckdb.connect(DB, read_only=True)
                loc = con2.execute(f"""select close, amount from v_daily
                    where thscode='{c}' and date='{str(row.day.date())}'""").df()
                con2.close()
                out.append(dict(code=c, date=str(row.day.date()),
                                sina_close=float(row.close), sina_amount=float(row.amount if 'amount' in row else np.nan),
                                local_close=float(loc.close.iloc[0]) if len(loc) else np.nan))
        except Exception as e:
            out.append(dict(code=c, date='err', sina_close=np.nan, sina_amount=np.nan,
                            local_close=type(e).__name__))
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(DATA, '核对_新浪抽查.csv'), index=False, encoding='utf-8-sig')
    s['close_diff'] = (s.local_close - s.sina_close).abs() / s.sina_close
    print('新浪抽查（不复权 close）：')
    print(s.to_string(index=False))


if __name__ == '__main__':
    main()
