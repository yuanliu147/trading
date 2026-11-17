# -*- coding: utf-8 -*-
"""补齐面板缺失股票的换手率/流通股本（akshare 新浪，与 31_构建2016_2026缓存.py 同口径）。

- 输入：/tmp/mcap_missing_codes.pkl（面板 thscode 列表，如 600004.SH）
- 输出：2026-10-18 项目 turnover_akshare/{exchange}.{code}.pkl，断点续传
- 退市股新浪大概率无数据；失败集中在第二轮用东财 stock_zh_a_hist 补试。
"""
import os, pickle, time, sys
import akshare as ak
import pandas as pd


TURN_DIR = '/Users/admin/WorkSpace/trading/2026-10-18_成交量融合_2010-2025/01_数据/turnover_akshare'
CODES = pickle.load(open('/tmp/mcap_missing_codes.pkl', 'rb'))

def to_ak(c):
    a, b = c.split('.')          # 600004.SH -> sh600004
    return b.lower() + a

def to_file(c):
    a, b = c.split('.')          # 600004.SH -> sh.600004
    return b.lower() + '.' + a

def fetch_one(c):
    fp = os.path.join(TURN_DIR, to_file(c) + '.pkl')
    if os.path.exists(fp):
        return c, 'cached'
    for attempt in (1, 2, 3):
        try:
            df = ak.stock_zh_a_daily(symbol=to_ak(c), start_date='20141201',
                                     end_date='20261016', adjust='hfq')
            if df is None or len(df) == 0:
                raise RuntimeError('empty')
            x = df[['date', 'turnover', 'outstanding_share']].copy()
            x['date'] = pd.to_datetime(x['date'])
            x['turn'] = pd.to_numeric(x['turnover'], errors='coerce') * 100.0
            x['float_share'] = pd.to_numeric(x['outstanding_share'], errors='coerce')
            x = x.drop(columns=['turnover', 'outstanding_share']).sort_values('date')
            with open(fp + '.tmp', 'wb') as f:
                pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(fp + '.tmp', fp)
            return c, 'ok'
        except Exception as e:
            if attempt == 3:
                return c, f'ERR {type(e).__name__}'
            time.sleep(0.8 * attempt)

def main():
    # 注意：akshare 内部 py_mini_racer 非线程安全，多线程会崩（exit 133），
    # 只能单进程顺序抓取；0.5-1s/只，4,400 只约 45-70 分钟，断点续传。
    todo = [c for c in CODES
            if not os.path.exists(os.path.join(TURN_DIR, to_file(c) + '.pkl'))]
    print(f'todo={len(todo)}', flush=True)
    t0 = time.time(); ok = 0; fail = []
    for i, c in enumerate(todo, 1):
        _, status = fetch_one(c)
        if status == 'ok':
            ok += 1
        elif status.startswith('ERR'):
            fail.append(c)
        if i % 50 == 0:
            eta = (time.time() - t0) / i * (len(todo) - i) / 60
            print(f'{i}/{len(todo)} ok={ok} fail={len(fail)} '
                  f'elapsed={time.time()-t0:.0f}s eta={eta:.0f}min', flush=True)
    pickle.dump(fail, open('/tmp/mcap_fetch_fail.pkl', 'wb'))
    print(f'done ok={ok} fail={len(fail)} elapsed={time.time()-t0:.0f}s', flush=True)
    print('fail sample:', fail[:20], flush=True)

if __name__ == '__main__':
    main()
