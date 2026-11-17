# -*- coding: utf-8 -*-
"""第二路补齐：baostock 换手率，覆盖新浪缺失的退市/长停股。

- 输入：/tmp/mcap_fetch_fail.pkl
- 输出：turnover_akshare/{exchange}.{code}.pkl（同格式：date/turn/float_share）
- 口径：换手率 %，与新浪一致（成交量/流通股本×100）；float_share 不提供，置 NaN
"""
import os, pickle
import baostock as bs
import pandas as pd

TURN_DIR = '/Users/admin/WorkSpace/trading/2026-10-18_成交量融合_2010-2025/01_数据/turnover_akshare'
FAIL = pickle.load(open('/tmp/mcap_fetch_fail.pkl', 'rb'))


def to_bs(c):
    a, b = c.split('.')
    return b.lower() + '.' + a


def to_file(c):
    a, b = c.split('.')
    return b.lower() + '.' + a


def main():
    bs.login()
    ok, fail = 0, []
    for i, c in enumerate(FAIL, 1):
        code = to_bs(c)
        try:
            rs = bs.query_history_k_data_plus(code, 'date,turn',
                                              start_date='2014-12-01',
                                              end_date='2026-10-16',
                                              frequency='d', adjustflag='1')
            rows = []
            while rs.error_code == '0' and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                fail.append(c)
                continue
            x = pd.DataFrame(rows, columns=['date', 'turn'])
            x['date'] = pd.to_datetime(x['date'])
            x['turn'] = pd.to_numeric(x['turn'], errors='coerce')
            x = x[x.turn.notna()].sort_values('date')
            x['float_share'] = float('nan')
            if len(x) == 0:
                fail.append(c)
                continue
            fp = os.path.join(TURN_DIR, to_file(c) + '.pkl')
            with open(fp + '.tmp', 'wb') as f:
                pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(fp + '.tmp', fp)
            ok += 1
        except Exception as e:
            fail.append(c)
        if i % 20 == 0:
            print(f'{i}/{len(FAIL)} ok={ok} fail={len(fail)}', flush=True)
    bs.logout()
    pickle.dump(fail, open('/tmp/mcap_fetch_fail2.pkl', 'wb'))
    print(f'done ok={ok} fail={len(fail)}', fail, flush=True)


if __name__ == '__main__':
    main()
