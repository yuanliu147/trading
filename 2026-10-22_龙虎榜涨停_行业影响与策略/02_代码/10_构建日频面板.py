# -*- coding: utf-8 -*-
"""构建个股日频与行业日频面板（2016-10-14 ~ 2026-10-13）。
输入：本地 DuckDB 后复权日线、跌停研究项目分析面板、涨停事件清单、龙虎榜事件表。
输出：01_数据/个股日频.parquet, 行业日频.parquet, 市场日频.parquet
"""
import os, time, warnings
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
ROOT = os.path.join(BASE, '2026-10-22_龙虎榜涨停_行业影响与策略')
DATA = os.path.join(ROOT, '01_数据')
DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'
PANEL = os.path.join(BASE, '2026-10-22_跌停前特征_全A/01_数据/分析面板.parquet')
LU = os.path.join(BASE, '2026-10-22_跌停前特征_全A/01_数据/涨停事件清单.csv')
CODES = os.path.join(BASE, '2026-10-22_跌停前特征_全A/01_数据/代码表.csv')
MKT = os.path.join(BASE, '2026-10-22_跌停前特征_全A/01_数据/市场日表.csv')
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

# ---------- 1. 日线 ----------
log('读取 DuckDB 后复权日线')
con = duckdb.connect(DB, read_only=True)
d = con.execute("""
    select thscode, date, open, high, low, close, volume, amount
    from v_daily_hfq
    where date >= '2016-10-14' and date <= '2026-10-13'
""").df()
con.close()
d['date'] = pd.to_datetime(d['date'])
d = d.sort_values(['thscode', 'date']).reset_index(drop=True)
log(f'日线 {len(d):,} 行 {d.thscode.nunique()} 只')

# ---------- 2. 面板字段 ----------
log('读取分析面板（时点行业/技术特征）')
pcols = ['date', 'code_id', 'industry_id', 'board_id', 'st_proxy', 'is_new', 'age',
         'ret1', 'ret5', 'ret20', 'dist_ma20', 'dist_ma60', 'vol20', 'amt_ratio20',
         'log_amt20', 'rsi14', 'up_struct', 'trend_age', 'consec_lu', 'lu_count20',
         'one_word_lu_yesterday', 'opened_lu_yesterday', 'lu_yesterday',
         'sector_ret20', 'sector_ret60', 'sector_breadth20', 'sector_breadth60',
         'sector_rs', 'sw_ret20', 'sw_above60', 'sw_slope60',
         'market_breadth20', 'market_breadth60', 'mkt_ret20', 'mkt_above_ma20', 'mkt_below_ma60']
P = pd.read_parquet(PANEL, columns=pcols)
P['date'] = pd.to_datetime(P['date'])
codes = pd.read_csv(CODES)
P = P.merge(codes[['code_id', 'code']], on='code_id', how='left')
P = P.rename(columns={'code': 'thscode'})
P = P.drop(columns=['code_id'])
log(f'面板 {len(P):,} 行，股票 {P.thscode.nunique()}')
d = d.merge(P, on=['thscode', 'date'], how='left')
log(f'合并后 {len(d):,} 行，行业缺失 {d.industry_id.isna().sum():,}')

# ---------- 3. 涨停/跌停/可交易性 ----------
log('计算涨停/跌停/次日可交易性')
d['st'] = d['st_proxy'].fillna(0).astype(int)
d['board_id'] = d['board_id'].fillna(1).astype(int)
d['industry_id'] = d['industry_id'].fillna(0).astype(int)
d['prev_close'] = d.groupby('thscode')['close'].shift(1)
d['ret1_calc'] = d['close'] / d['prev_close'] - 1
d['limit_rate'] = np.select(
    [d.board_id.eq(4), d.board_id.eq(3), d.board_id.eq(2) & (d.date >= '2020-08-24'),
     d.board_id.eq(2) & (d.date < '2020-08-24'), d.st.eq(1) & d.board_id.eq(1)],
    [0.30, 0.20, 0.20, 0.10, 0.05], default=0.10)
# 次新/无有效前收：不做涨跌停判定
valid_prev = d.prev_close.notna() & (d.prev_close > 0) & (d.age >= 1)
d.loc[~valid_prev, 'limit_rate'] = np.nan
# 涨停事件清单（精确收盘封板/一字）
lu = pd.read_csv(LU, usecols=['date', 'thscode', 'lu_ratio', 'one_word'])
lu['date'] = pd.to_datetime(lu['date'])
lu['lu_event'] = 1
lu['lu_one_event'] = lu['one_word'].astype(int)
d = d.merge(lu[['date', 'thscode', 'lu_event', 'lu_one_event']], on=['date', 'thscode'], how='left')
d['lu_event'] = d['lu_event'].fillna(0).astype(int)
d['lu_one_event'] = d['lu_one_event'].fillna(0).astype(int)
d['lu_calc'] = ((d['close'] / d['prev_close'] - 1) >= (d['limit_rate'] - 0.002)).astype(int)
d['lu_seal'] = ((d.lu_event == 1) | (d.lu_calc == 1)).astype(int)
d['lu_one'] = ((d.lu_one_event == 1) | ((d.lu_seal == 1) & (d.high == d.low) & (d.high == d.close))).astype(int)
d['lu_touch'] = (((d['high'] / d['prev_close'] - 1) >= (d['limit_rate'] - 0.002)) | (d.lu_seal == 1)).astype(int)
d['lu_open'] = ((d.lu_touch == 1) & (d.lu_seal == 0)).astype(int)
d['ld_calc'] = ((d['close'] / d['prev_close'] - 1) <= -(d['limit_rate'] - 0.002)).astype(int)
d['ld_seal'] = d['ld_calc']
d['ld_touch'] = ((d['low'] / d['prev_close'] - 1) <= -(d['limit_rate'] - 0.002)).astype(int)
d['ld_open'] = ((d.ld_touch == 1) & (d.ld_seal == 0)).astype(int)
# 次日开盘可买/可卖（用 T+1 的 open/close 回填到 T 行）
g = d.groupby('thscode', sort=False)
d['next_open'] = g['open'].shift(-1)
d['next_close'] = g['close'].shift(-1)
d['next_low'] = g['low'].shift(-1)
d['next_high'] = g['high'].shift(-1)
d['next2_open'] = g['open'].shift(-2)
d['next2_close'] = g['close'].shift(-2)
d['next2_date'] = g['date'].shift(-2)
d['next2_ld_seal'] = g['ld_seal'].shift(-2)
d['next_date'] = g['date'].shift(-1)
d['next_prev_close'] = g['prev_close'].shift(-1)
d['next_limit_rate'] = g['limit_rate'].shift(-1)
d['next_open_ret'] = d['next_open'] / d['close'] - 1
d['next_unbuyable'] = ((d.next_open_ret >= d.next_limit_rate - 0.002) | d.next_open.isna()).astype(int)
d['next_high_open'] = (d.next_open_ret > 0.07).astype(int)
d['next_ld_seal'] = g['ld_seal'].shift(-1)
# 个股前瞻收益（T 收盘 -> T+h 收盘；T+1 开盘 -> T+h 收盘）
for h in [1, 2, 3, 5, 10, 20]:
    d[f'fwd{h}_close'] = g['close'].shift(-h) / d['close'] - 1
    d[f'fwd{h}_open'] = g['close'].shift(-h) / d['next_open'] - 1

# 有效样本
d['tradable'] = ((d.age >= 60) & (d.st == 0) & (d.industry_id > 0) & (d.close > 0) &
                 (d.volume > 0) & (d.amount > 0)).astype(int)
d['amt20'] = g['amount'].transform(lambda x: x.rolling(20, min_periods=15).mean())
d['ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=20).mean())
d['ma60'] = g['close'].transform(lambda x: x.rolling(60, min_periods=40).mean())
d['liq'] = (d.amt20 >= 3e7).astype(int)
# 连板（用事件清单/收盘涨停重算，避免面板 consec_lu 的口径差异）
d['lu_seal_prev'] = g['lu_seal'].shift(1).fillna(0).astype(int)
d['consec_lu_calc'] = d.groupby(['thscode', (~d.lu_seal.astype(bool)).cumsum()])['lu_seal'].cumsum()

keep = ['date', 'thscode', 'industry_id', 'board_id', 'st', 'is_new', 'age',
        'open', 'high', 'low', 'close', 'volume', 'amount', 'prev_close', 'ret1_calc',
        'limit_rate', 'lu_seal', 'lu_one', 'lu_touch', 'lu_open', 'ld_seal', 'ld_touch', 'ld_open',
        'next_open', 'next_close', 'next_low', 'next_high', 'next2_open', 'next2_close', 'next2_date', 'next2_ld_seal', 'next_date', 'next_open_ret', 'next_limit_rate', 'next_unbuyable', 'next_high_open', 'next_ld_seal',
        'tradable', 'liq', 'amt20', 'ma20', 'ma60', 'ret5', 'ret20', 'dist_ma20', 'dist_ma60',
        'vol20', 'amt_ratio20', 'log_amt20', 'rsi14', 'up_struct', 'trend_age',
        'consec_lu_calc', 'lu_count20', 'sector_ret20', 'sector_ret60', 'sector_breadth20',
        'sector_breadth60', 'sector_rs', 'sw_ret20', 'sw_above60', 'sw_slope60',
        'market_breadth20', 'market_breadth60', 'mkt_ret20', 'mkt_above_ma20', 'mkt_below_ma60']
for h in [1, 2, 3, 5, 10, 20]:
    keep += [f'fwd{h}_close', f'fwd{h}_open']
S = d[keep].copy()
for c in ['open', 'high', 'low', 'close', 'prev_close', 'next_open', 'next_close', 'amount', 'amt20', 'ma20', 'ma60']:
    S[c] = S[c].astype('float32')
S.to_parquet(os.path.join(DATA, '个股日频.parquet'), index=False)
log(f'保存个股日频 {len(S):,} 行')

# ---------- 4. 行业日频 ----------
log('构建行业日频')
V = S[(S.tradable == 1) & S.ret1_calc.notna()].copy()
# 龙虎榜事件标记
L = pd.read_parquet(os.path.join(DATA, '龙虎榜事件.parquet'))
L['date'] = pd.to_datetime(L['date'])
L = L.merge(S[['date', 'thscode', 'industry_id']], on=['date', 'thscode'], how='left')
L['industry_id'] = L.industry_id.fillna(0).astype(int)
L['is_up_event'] = ((L.has_up == 1) & (L.chg > 0)).astype(int)
L['is_down_event'] = ((L.has_down == 1) & (L.chg < 0)).astype(int)
L['net_buy'] = pd.to_numeric(L.net_buy, errors='coerce').fillna(0)
L['org_net'] = pd.to_numeric(L.org_net, errors='coerce').fillna(0)
L['lhb_amt'] = pd.to_numeric(L.lhb_amount, errors='coerce').fillna(0)
L['up_1d'] = ((L.is_up_event == 1) & (L.window == 1)).astype(int)
L['up_3d'] = ((L.is_up_event == 1) & (L.window == 3)).astype(int)
L['down_1d'] = ((L.is_down_event == 1) & (L.window == 1)).astype(int)
L['org_pos'] = ((L.org_present == 1) & (L.org_net > 0)).astype(int)
L['org_neg'] = ((L.org_present == 1) & (L.org_net < 0)).astype(int)
# 事件标记回填到个股日频，供“剔除事件股”的行业收益使用
Lflag = L.groupby(['date', 'thscode']).agg(lhb_up=('up_1d', 'max'), lhb_down=('down_1d', 'max')).reset_index()
V = V.merge(Lflag, on=['date', 'thscode'], how='left')
V['lhb_up'] = V.lhb_up.fillna(0).astype(int)
V['lhb_down'] = V.lhb_down.fillna(0).astype(int)
V['ret1_ex_lhb'] = V.ret1_calc * (1 - V.lhb_up)
V['ret1_ex_lu'] = V.ret1_calc * (1 - V.lu_seal)
V['ret1_ex_both'] = V.ret1_calc * (1 - ((V.lhb_up == 1) | (V.lu_seal == 1)).astype(int))
V['one_lhb'] = 1 - V.lhb_up
V['one_lu'] = 1 - V.lu_seal
V['one_both'] = 1 - ((V.lhb_up == 1) | (V.lu_seal == 1)).astype(int)

def agg_lhb(x):
    return pd.Series({
        'lhb_up_count': x.up_1d.sum(),
        'lhb_up3_count': x.up_3d.sum(),
        'lhb_down_count': x.down_1d.sum(),
        'lhb_net': x.loc[x.up_1d == 1, 'net_buy'].sum(),
        'lhb_org_net': x.loc[x.up_1d == 1, 'org_net'].sum(),
        'lhb_pos_count': ((x.up_1d == 1) & (x.net_buy > 0)).sum(),
        'lhb_org_pos_count': ((x.up_1d == 1) & (x.org_pos == 1)).sum(),
        'lhb_org_neg_count': ((x.up_1d == 1) & (x.org_neg == 1)).sum(),
        'lhb_amt_sum': x.loc[x.up_1d == 1, 'lhb_amt'].sum(),
    })
LH = L.groupby(['date', 'industry_id']).apply(agg_lhb, include_groups=False).reset_index()
for c in ['lhb_up_count', 'lhb_up3_count', 'lhb_down_count', 'lhb_pos_count',
          'lhb_org_pos_count', 'lhb_org_neg_count']:
    LH[c] = LH[c].astype(int)

A = V.groupby(['date', 'industry_id']).agg(
    n=('ret1_calc', 'size'),
    sum_ret=('ret1_calc', 'sum'),
    sum_ret_ex_lhb=('ret1_ex_lhb', 'sum'),
    sum_ret_ex_lu=('ret1_ex_lu', 'sum'),
    sum_ret_ex_both=('ret1_ex_both', 'sum'),
    n_ex_lhb=('one_lhb', 'sum'),
    n_ex_lu=('one_lu', 'sum'),
    n_ex_both=('one_both', 'sum'),
    ind_amt=('amount', 'sum'),
    lu_count=('lu_seal', 'sum'),
    lu_one_count=('lu_one', 'sum'),
    lu_open_count=('lu_open', 'sum'),
    lu_touch_count=('lu_touch', 'sum'),
    lu2_count=('consec_lu_calc', lambda x: int((x >= 2).sum())),
    max_consec=('consec_lu_calc', 'max'),
    ld_count=('ld_seal', 'sum'),
    ld_touch_count=('ld_touch', 'sum'),
    above_ma20=('close', lambda x: 0),  # 占位，下面重算
    above_ma60=('close', lambda x: 0),
    ind_ret5_eq=('ret5', 'mean'),
    ind_ret20_eq=('ret20', 'mean'),
    ind_vol20=('vol20', 'mean'),
).reset_index()
# 均线宽度单独算（lambda 里拿不到 ma20，改用向量化）
MA = V.assign(a20=(V.close > V.ma20).astype(int), a60=(V.close > V.ma60).astype(int))
MA = MA.groupby(['date', 'industry_id']).agg(above_ma20=('a20', 'sum'), above_ma60=('a60', 'sum')).reset_index()
A = A.drop(columns=['above_ma20', 'above_ma60']).merge(MA, on=['date', 'industry_id'], how='left')
A['ind_ret1'] = A.sum_ret / A.n
A['ind_ret1_ex_lhb'] = np.where(A.n_ex_lhb > 0, A.sum_ret_ex_lhb / A.n_ex_lhb, np.nan)
A['ind_ret1_ex_lu'] = np.where(A.n_ex_lu > 0, A.sum_ret_ex_lu / A.n_ex_lu, np.nan)
A['ind_ret1_ex_both'] = np.where(A.n_ex_both > 0, A.sum_ret_ex_both / A.n_ex_both, np.nan)
A = A.merge(LH, on=['date', 'industry_id'], how='left')
num_cols = [c for c in A.columns if c not in ['date', 'industry_id']]
A[num_cols] = A[num_cols].fillna(0)
A['lu_share'] = A.lu_count / A.n
A['lu_one_share'] = A.lu_one_count / A.n
A['lu_open_share'] = A.lu_open_count / A.n
A['lu2_share'] = A.lu2_count / A.n
A['ld_share'] = A.ld_count / A.n
A['lhb_share'] = A.lhb_up_count / A.n
A['lhb_net_ratio'] = A.lhb_net / A.ind_amt.replace(0, np.nan)
A['lhb_org_ratio'] = A.lhb_org_net / A.ind_amt.replace(0, np.nan)
A['lhb_pos_share'] = A.lhb_pos_count / A.lhb_up_count.replace(0, np.nan)
A['lhb_org_pos_share'] = A.lhb_org_pos_count / A.lhb_up_count.replace(0, np.nan)
A['lhb_amt_share'] = A.lhb_amt_sum / A.ind_amt.replace(0, np.nan)
A['breadth20'] = A.above_ma20 / A.n
A['breadth60'] = A.above_ma60 / A.n
# LHB 与涨停重叠：另算
LUO = L[L.is_up_event == 1].merge(V[['date', 'thscode', 'lu_seal']], on=['date', 'thscode'], how='left')
LUO = LUO.groupby(['date', 'industry_id']).agg(lhb_lu_count=('lu_seal', 'sum')).reset_index()
A = A.merge(LUO, on=['date', 'industry_id'], how='left')
A['lhb_lu_count'] = A.lhb_lu_count.fillna(0)
A['lhb_lu_share'] = A.lhb_lu_count / A.lhb_up_count.replace(0, np.nan)
# 行业收益指数与前瞻收益（含剔除事件股口径）
A = A.sort_values(['industry_id', 'date']).reset_index(drop=True)
for tag, col in [('', 'ind_ret1'), ('_ex_lhb', 'ind_ret1_ex_lhb'), ('_ex_lu', 'ind_ret1_ex_lu'), ('_ex_both', 'ind_ret1_ex_both')]:
    nav = f'ind_nav{tag}'
    A[nav] = A.groupby('industry_id')[col].transform(lambda x: (1 + x.fillna(0)).cumprod())
    for h in [1, 2, 3, 5, 10, 20]:
        A[f'fwd{h}{tag}'] = A.groupby('industry_id')[nav].shift(-h) / A[nav] - 1
A.to_parquet(os.path.join(DATA, '行业日频.parquet'), index=False)
log(f'保存行业日频 {len(A):,} 行')

# ---------- 5. 市场日频 ----------
M = V.groupby('date').apply(lambda x: pd.Series({
    'mkt_ret1_eq': x.ret1_calc.mean(),
    'mkt_n': len(x),
    'mkt_lu_count': x.lu_seal.sum(),
    'mkt_lu_share': x.lu_seal.mean(),
    'mkt_lu_one_share': x.lu_one.mean(),
    'mkt_lu_open_share': x.lu_open.mean(),
    'mkt_ld_share': x.ld_seal.mean(),
    'mkt_breadth20': (x.close > x.ma20).mean(),
    'mkt_breadth60': (x.close > x.ma60).mean(),
    'mkt_amt': x.amount.sum(),
}), include_groups=False).reset_index()
M['mkt_nav'] = (1 + M.mkt_ret1_eq.fillna(0)).cumprod()
M = M.sort_values('date').reset_index(drop=True)
mkt0 = pd.read_csv(MKT)
mkt0['date'] = pd.to_datetime(mkt0['date'])
M = M.merge(mkt0[['date', 'regime_id', 'market_sent', 'market_amt_ratio']], on='date', how='left')
M.to_parquet(os.path.join(DATA, '市场日频.parquet'), index=False)
log(f'保存市场日频 {len(M):,} 行')

# 摘要
log('数据摘要')
print(S[['lu_seal', 'lu_one', 'lu_open', 'ld_seal']].sum().to_string())
print(A[['lhb_up_count', 'lu_count', 'lu_share', 'lhb_net_ratio']].describe().to_string())
print(M.tail(3).to_string())


if __name__ == '__main__':
    pass
