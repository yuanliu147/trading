# -*- coding: utf-8 -*-
"""整理 2016-2026 龙虎榜明细与机构买卖统计，生成逐股票-日事件表。
关键口径：
- 同一股票-日可能有多条上榜原因，金额口径不同（单日榜 vs 连续三日榜），不能直接相加。
- 主口径优先单日榜（涨幅>换手>振幅>无涨跌幅限制>跌幅），没有再取连续三日榜。
- 机构净买与主榜原因对齐；没有机构记录时 org_present=0，而不是把“无数据”当 0。
"""
import os, glob, re
import numpy as np
import pandas as pd

BASE = '/Users/admin/WorkSpace/trading/2026-10-22_龙虎榜涨停_行业影响与策略'
DATA = os.path.join(BASE, '01_数据')
OUT = os.path.join(BASE, '03_结果')
os.makedirs(OUT, exist_ok=True)


def to_thscode(x):
    s = str(x).strip().split('.')[0].zfill(6)
    if s.startswith('6'):
        return s + '.SH'
    if s.startswith(('0', '3')):
        return s + '.SZ'
    if s.startswith(('4', '8', '9')):
        return s + '.BJ'
    return s


def classify_reason(r):
    r = '' if pd.isna(r) else str(r)
    if '连续' in r and '涨幅' in r:
        return '连续涨幅'
    if '连续' in r and '跌幅' in r:
        return '连续跌幅'
    if '涨幅' in r:
        return '涨幅'
    if '跌幅' in r:
        return '跌幅'
    if '振幅' in r:
        return '振幅'
    if '换手' in r:
        return '换手'
    if '无价格涨跌幅限制' in r:
        return '无涨跌幅限制'
    if '退市' in r:
        return '退市'
    if 'ST' in r:
        return 'ST'
    return '其他'


PRIORITY = {'涨幅': 0, '换手': 1, '振幅': 2, '无涨跌幅限制': 3, '跌幅': 4,
            '连续涨幅': 5, '连续跌幅': 6, 'ST': 7, '其他': 8, '退市': 9}
WINDOW = {'涨幅': 1, '换手': 1, '振幅': 1, '无涨跌幅限制': 1, '跌幅': 1,
          '连续涨幅': 3, '连续跌幅': 3, 'ST': 3, '其他': 1, '退市': 1}
UP_TYPES = {'涨幅', '换手', '振幅', '无涨跌幅限制', '连续涨幅'}


def dedupe(df, cols_keep):
    df = df.copy()
    df['_prio'] = df['reason_type'].map(PRIORITY).fillna(9).astype(int)
    df = df.sort_values(['thscode', 'date', '_prio'])
    primary = df.drop_duplicates(['thscode', 'date'], keep='first').copy()
    agg = df.groupby(['thscode', 'date']).agg(
        n_rows=('reason_type', 'size'),
        n_reasons=('reason_type', 'nunique'),
        reason_types=('reason_type', lambda x: '|'.join(sorted(set(x)))),
        has_1d=('window', lambda x: int((x == 1).any())),
        has_3d=('window', lambda x: int((x == 3).any())),
        has_up=('reason_type', lambda x: int(any(t in UP_TYPES for t in x))),
        has_down=('reason_type', lambda x: int(any(t in {'跌幅', '连续跌幅'} for t in x))),
        max_net_buy=('net_buy', 'max'),
        min_net_buy=('net_buy', 'min'),
    ).reset_index()
    out = primary.merge(agg, on=['thscode', 'date'], how='left')
    return out


def main():
    # ---------- 明细 ----------
    files = sorted(glob.glob(os.path.join(DATA, '龙虎榜明细_*.parquet')))
    print('明细文件', len(files))
    dd = []
    for p in files:
        x = pd.read_parquet(p)
        x['源文件'] = os.path.basename(p)
        dd.append(x)
    d = pd.concat(dd, ignore_index=True)
    print('原始明细', len(d))
    d['代码'] = d['代码'].astype(str).str.zfill(6)
    d['thscode'] = d['代码'].map(to_thscode)
    d['date'] = pd.to_datetime(d['上榜日'])
    d['reason'] = d['上榜原因'].astype(str)
    d['reason_type'] = d['reason'].map(classify_reason)
    d['window'] = d['reason_type'].map(WINDOW).fillna(1).astype(int)
    d['net_buy'] = pd.to_numeric(d['龙虎榜净买额'], errors='coerce')
    d['buy'] = pd.to_numeric(d['龙虎榜买入额'], errors='coerce')
    d['sell'] = pd.to_numeric(d['龙虎榜卖出额'], errors='coerce')
    d['lhb_amount'] = pd.to_numeric(d['龙虎榜成交额'], errors='coerce')
    d['mkt_amount'] = pd.to_numeric(d['市场总成交额'], errors='coerce')
    d['net_ratio'] = pd.to_numeric(d['净买额占总成交比'], errors='coerce') / 100.0
    d['lhb_amount_ratio'] = pd.to_numeric(d['成交额占总成交比'], errors='coerce') / 100.0
    d['turnover'] = pd.to_numeric(d['换手率'], errors='coerce')
    d['float_mcap'] = pd.to_numeric(d['流通市值'], errors='coerce')
    d['chg'] = pd.to_numeric(d['涨跌幅'], errors='coerce') / 100.0
    d['series'] = d['代码'].map(lambda s: '688' if s.startswith('688') else ('300' if s.startswith(('300', '301')) else ('BSE' if s.startswith(('4', '8', '9')) else 'MAIN')))
    # 同一原因可能重复（极少），先去重
    d = d.drop_duplicates(['thscode', 'date', 'reason'])
    d = dedupe(d, None)
    print('去重后股票-日', len(d), '日期', d.date.min().date(), '~', d.date.max().date())
    print(d.reason_type.value_counts().to_string())

    # ---------- 机构 ----------
    jf = sorted(glob.glob(os.path.join(DATA, '机构买卖_*.parquet')))
    print('机构文件', len(jf))
    jj = []
    for p in jf:
        x = pd.read_parquet(p)
        x['源文件'] = os.path.basename(p)
        jj.append(x)
    j = pd.concat(jj, ignore_index=True)
    print('原始机构', len(j))
    j['代码'] = j['代码'].astype(str).str.zfill(6)
    j['thscode'] = j['代码'].map(to_thscode)
    j['date'] = pd.to_datetime(j['上榜日期'])
    j['reason'] = j['上榜原因'].astype(str)
    j['reason_type'] = j['reason'].map(classify_reason)
    j['window'] = j['reason_type'].map(WINDOW).fillna(1).astype(int)
    j['org_buy'] = pd.to_numeric(j['机构买入总额'], errors='coerce')
    j['org_sell'] = pd.to_numeric(j['机构卖出总额'], errors='coerce')
    j['org_net'] = pd.to_numeric(j['机构买入净额'], errors='coerce')
    j['org_net_ratio'] = pd.to_numeric(j['机构净买额占总成交额比'], errors='coerce') / 100.0
    j['org_buyers'] = pd.to_numeric(j['买方机构数'], errors='coerce')
    j['org_sellers'] = pd.to_numeric(j['卖方机构数'], errors='coerce')
    j['reason_prio'] = j['reason_type'].map(PRIORITY).fillna(9).astype(int)
    j = j.drop_duplicates(['thscode', 'date', 'reason'])
    j = j.sort_values(['thscode', 'date', 'reason_prio'])
    jp = j.drop_duplicates(['thscode', 'date'], keep='first').copy()
    jp['org_present'] = 1
    jp = jp[['thscode', 'date', 'org_buy', 'org_sell', 'org_net', 'org_net_ratio',
             'org_buyers', 'org_sellers', 'org_present']]
    out = d.merge(jp, on=['thscode', 'date'], how='left')
    for c in ['org_buy', 'org_sell', 'org_net', 'org_net_ratio']:
        out[c] = out[c].fillna(0.0)
    out['org_present'] = out['org_present'].fillna(0).astype(int)
    out['org_buyers'] = out['org_buyers'].fillna(0).astype(int)
    out['org_sellers'] = out['org_sellers'].fillna(0).astype(int)
    out['is_up'] = (out['chg'] > 0).astype(int)
    out['is_limit_up_lhb'] = (out['chg'] >= 0.095).astype(int)  # 仅粗标记，精确涨停后续用事件表
    out['year'] = out.date.dt.year

    cols = ['date', 'thscode', '代码', '名称', 'series', 'reason', 'reason_type', 'window',
            'net_buy', 'buy', 'sell', 'lhb_amount', 'mkt_amount', 'net_ratio', 'lhb_amount_ratio',
            'turnover', 'float_mcap', 'chg', 'is_up', 'is_limit_up_lhb',
            'org_present', 'org_buy', 'org_sell', 'org_net', 'org_net_ratio', 'org_buyers', 'org_sellers',
            'n_rows', 'n_reasons', 'reason_types', 'has_1d', 'has_3d', 'has_up', 'has_down',
            'max_net_buy', 'min_net_buy', 'year']
    out = out[cols].sort_values(['date', 'thscode']).reset_index(drop=True)
    out.to_parquet(os.path.join(DATA, '龙虎榜事件.parquet'), index=False)
    print('保存', len(out), '行 ->', os.path.join(DATA, '龙虎榜事件.parquet'))

    # 汇总
    summ = out.groupby(['year', 'reason_type']).agg(
        事件数=('thscode', 'size'),
        股票数=('thscode', 'nunique'),
        净买额亿元=('net_buy', lambda x: x.sum() / 1e8),
        机构净买亿元=('org_net', lambda x: x.sum() / 1e8),
        机构参与率=('org_present', 'mean'),
    ).reset_index()
    summ.to_csv(os.path.join(OUT, '龙虎榜_分年分类汇总.csv'), index=False, encoding='utf-8-sig')
    out.groupby('series').agg(事件数=('thscode', 'size'), 股票数=('thscode', 'nunique')).to_csv(
        os.path.join(OUT, '龙虎榜_分板块.csv'), encoding='utf-8-sig')
    print(summ.head(20).to_string())


if __name__ == '__main__':
    main()
