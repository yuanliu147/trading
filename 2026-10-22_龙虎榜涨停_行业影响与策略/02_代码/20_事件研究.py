# -*- coding: utf-8 -*-
"""龙虎榜/涨停事件研究：
1) 个股事件后的个股收益（T+1 开盘 -> T+h 收盘）
2) 对同行业非事件股的外溢（行业等权、剔除当日事件股，减全 A 等权）
分组：净买强度、机构净买、涨幅/涨停、上榜原因、首次/重复、板块、市场状态、行业热度。
"""
import os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading/2026-10-22_龙虎榜涨停_行业影响与策略'
DATA = os.path.join(BASE, '01_数据')
OUT = os.path.join(BASE, '03_结果')
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

HS = [1, 2, 3, 5, 10, 20]
END = pd.Timestamp('2026-09-01')

# ---------- 载入 ----------
log('载入个股/行业/市场/事件')
S = pd.read_parquet(os.path.join(DATA, '个股日频.parquet'),
                    columns=['date', 'thscode', 'industry_id', 'board_id', 'st', 'age',
                             'close', 'amount', 'ret1_calc', 'lu_seal', 'lu_one', 'lu_open',
                             'lu_touch', 'consec_lu_calc', 'tradable', 'liq', 'amt20',
                             'ret5', 'ret20', 'dist_ma20', 'sector_ret20', 'sector_breadth20',
                             'mkt_ret20', 'market_breadth20', 'next_unbuyable', 'next_high_open',
                             'next_open_ret', 'next_date', 'next_limit_rate'] +
                            [f'fwd{h}_{w}' for h in HS for w in ['close', 'open']])
A = pd.read_parquet(os.path.join(DATA, '行业日频.parquet'))
M = pd.read_parquet(os.path.join(DATA, '市场日频.parquet'))
L = pd.read_parquet(os.path.join(DATA, '龙虎榜事件.parquet'))
L['date'] = pd.to_datetime(L['date'])
L = L[L.date <= END].copy()
# 市场前瞻
M = M.sort_values('date').reset_index(drop=True)
M['mkt_nav'] = (1 + M.mkt_ret1_eq.fillna(0)).cumprod()
for h in HS:
    M[f'mkt_fwd{h}'] = M.mkt_nav.shift(-h) / M.mkt_nav - 1
Mcols = ['date', 'regime_id', 'mkt_breadth20'] + [f'mkt_fwd{h}' for h in HS]
M2 = M[Mcols].copy()
cal = pd.DataFrame({'date': M.date.values[:-1], 'next_trade_date': M.date.values[1:]})
cal['date'] = pd.to_datetime(cal.date); cal['next_trade_date'] = pd.to_datetime(cal.next_trade_date)

# ---------- 龙虎榜事件 ----------
log('整理龙虎榜事件分组')
L['is_up'] = ((L.has_up == 1) & (L.chg > 0)).astype(int)
L = L[L.is_up == 1].copy()
# 只保留单日榜主口径，连续三日榜量级不同，主表先排除
L1 = L[L.window == 1].copy()
L1 = L1.merge(S[['date', 'thscode', 'industry_id']], on=['date', 'thscode'], how='left')
L1['industry_id'] = L1.industry_id.fillna(0).astype(int)
# 重复上榜：同股票过去 30 天是否有龙虎榜上涨事件
L1 = L1.sort_values(['thscode', 'date']).reset_index(drop=True)
L1['prev_date'] = L1.groupby('thscode')['date'].shift(1)
L1['days_since_prev'] = (L1.date - L1.prev_date).dt.days
L1['repeat30'] = (L1.days_since_prev <= 30).fillna(False).astype(int)
# 事件当天是否涨停 / 是否一字
Sflag = S[['date', 'thscode', 'lu_seal', 'lu_one', 'lu_open', 'consec_lu_calc',
           'tradable', 'liq', 'amt20', 'st', 'age', 'next_unbuyable', 'next_high_open', 'next_open_ret', 'next_date', 'next_limit_rate'] +
          [f'fwd{h}_open' for h in HS]].copy()
L1 = L1.merge(Sflag, on=['date', 'thscode'], how='left')
L1['lu_seal'] = L1.lu_seal.fillna(0).astype(int)
L1['lu_one'] = L1.lu_one.fillna(0).astype(int)
L1['chg_bucket'] = pd.cut(L1.chg, [-1, 0, 0.05, 0.095, 0.20, 1.0],
                          labels=['下跌', '0-5%', '5-9.5%', '涨停', '>10%'])
L1['net_bucket'] = pd.qcut(L1.net_ratio, 5, labels=['净卖最多', '净卖', '中性', '净买', '净买最多'], duplicates='drop')
L1['org_bucket'] = np.select(
    [L1.org_present.eq(0), L1.org_net.gt(0), L1.org_net.lt(0)],
    ['无机构席位', '机构净买', '机构净卖'], default='其他')
L1['reason_type'] = L1.reason_type.replace({'连续涨幅': '连续涨幅', '涨幅': '涨幅'})
# 行业热度/动量
Afeat = A[['date', 'industry_id', 'lhb_share', 'lhb_net_ratio', 'lhb_org_ratio', 'lhb_up_count',
           'lu_share', 'lu_one_share', 'lu_open_share', 'lu2_share', 'lhb_lu_share',
           'ind_ret1', 'ind_ret20_eq', 'breadth20', 'n'] +
          [f'fwd{h}{tag}' for h in HS for tag in ['', '_ex_lhb', '_ex_lu', '_ex_both']]]
L1 = L1.merge(Afeat, on=['date', 'industry_id'], how='left')
L1 = L1.merge(M2, on='date', how='left')
L1 = L1.merge(cal, on='date', how='left')
L1['gap_ok'] = (L1.next_date == L1.next_trade_date).astype(int)
L1['next_limit_down_open'] = (L1.next_open_ret <= -(L1.next_limit_rate - 0.002)).astype(int)
# 个股前瞻 = T+1 开盘 -> T+h 收盘；T+1 一字/开盘涨停视为买不到
for h in HS:
    L1[f'stock_fwd{h}'] = np.where((L1.next_unbuyable == 1) | (L1.gap_ok == 0) | (L1.next_limit_down_open == 1), np.nan, L1[f'fwd{h}_open'])
    L1[f'ind_fwd{h}'] = L1[f'fwd{h}_ex_lhb']
    L1[f'ind_abn{h}'] = L1[f'ind_fwd{h}'] - L1[f'mkt_fwd{h}']
L1['sector_mom_bucket'] = pd.qcut(L1.ind_ret20_eq, 5, labels=['行业最弱', '弱', '中', '强', '行业最强'], duplicates='drop')
L1['mkt_regime'] = L1.regime_id.map({1: '熊市', 2: '震荡', 3: '弱震荡', 4: '牛市', 5: '强牛'}).fillna('未知')
L1.to_parquet(os.path.join(DATA, '龙虎榜事件_含前瞻.parquet'), index=False)
log(f'龙虎榜上涨单日事件 {len(L1):,} 个')

# ---------- 涨停事件 ----------
log('整理涨停事件分组')
U = S[(S.lu_seal == 1) & (S.date <= END) & (S.st == 0)].copy()
U = U.merge(M2, on='date', how='left')
U = U.merge(cal, on='date', how='left')
U['gap_ok'] = (U.next_date == U.next_trade_date).astype(int)
U['next_limit_down_open'] = (U.next_open_ret <= -(U.next_limit_rate - 0.002)).astype(int)
U = U.merge(Afeat, on=['date', 'industry_id'], how='left')
for h in HS:
    U[f'stock_fwd{h}'] = np.where((U.next_unbuyable == 1) | (U.gap_ok == 0) | (U.next_limit_down_open == 1), np.nan, U[f'fwd{h}_open'])
    U[f'ind_fwd{h}'] = U[f'fwd{h}_ex_lu']
    U[f'ind_abn{h}'] = U[f'ind_fwd{h}'] - U[f'mkt_fwd{h}']
U['consec_bucket'] = pd.cut(U.consec_lu_calc, [0, 1, 2, 3, 99], labels=['首板', '2连板', '3连板', '4板+'])
U['lu_share_bucket'] = pd.qcut(U.lu_share, 5, labels=['行业涨停最少', '少', '中', '多', '行业涨停最多'], duplicates='drop')
U['sector_mom_bucket'] = pd.qcut(U.ind_ret20_eq, 5, labels=['行业最弱', '弱', '中', '强', '行业最强'], duplicates='drop')
U['mkt_regime'] = U.regime_id.map({1: '熊市', 2: '震荡', 3: '弱震荡', 4: '牛市', 5: '强牛'}).fillna('未知')
U.to_parquet(os.path.join(DATA, '涨停事件_含前瞻.parquet'), index=False)
log(f'涨停事件 {len(U):,} 个')

# ---------- 统计函数 ----------
def cluster_stats(df, col):
    x = df[['date', col]].dropna()
    if len(x) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, win=np.nan, t=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    daily = x.groupby('date')[col].mean()
    m = daily.mean()
    t = m / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else np.nan
    rng = np.random.default_rng(20261022)
    bs = np.array([daily.sample(len(daily), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(300)])
    return dict(n=len(x), mean=x[col].mean(), median=x[col].median(), win=(x[col] > 0).mean(),
                t=t, ci_lo=np.quantile(bs, 0.025), ci_hi=np.quantile(bs, 0.975))


def summarize(df, group_cols, value_prefix, horizons=HS, min_n=30):
    rows = []
    # 全样本
    for h in horizons:
        col = f'{value_prefix}{h}'
        st = cluster_stats(df, col)
        st.update({'分组': '全部', '组值': '全部', 'h': h})
        rows.append(st)
    for g in group_cols:
        if g not in df.columns:
            continue
        for val, sub in df.groupby(g, observed=True):
            if len(sub) < min_n:
                continue
            for h in horizons:
                col = f'{value_prefix}{h}'
                if col not in sub:
                    continue
                st = cluster_stats(sub, col)
                st.update({'分组': g, '组值': str(val), 'h': h})
                rows.append(st)
    out = pd.DataFrame(rows)
    cols = ['分组', '组值', 'h', 'n', 'mean', 'median', 'win', 't', 'ci_lo', 'ci_hi']
    return out[cols]


log('计算龙虎榜个股收益')
r1 = summarize(L1, ['net_bucket', 'org_bucket', 'chg_bucket', 'reason_type', 'repeat30', 'board_id',
                    'mkt_regime', 'sector_mom_bucket'], 'stock_fwd', min_n=50)
r1['口径'] = '龙虎榜_个股'
log('计算龙虎榜行业外溢')
r2 = summarize(L1, ['net_bucket', 'org_bucket', 'chg_bucket', 'reason_type', 'repeat30', 'board_id',
                    'mkt_regime', 'sector_mom_bucket'], 'ind_abn', min_n=50)
r2['口径'] = '龙虎榜_行业外溢'
log('计算涨停个股收益')
r3 = summarize(U, ['consec_bucket', 'lu_one', 'lu_share_bucket', 'mkt_regime', 'sector_mom_bucket', 'board_id'],
               'stock_fwd', min_n=50)
r3['口径'] = '涨停_个股'
log('计算涨停行业外溢')
r4 = summarize(U, ['consec_bucket', 'lu_one', 'lu_share_bucket', 'mkt_regime', 'sector_mom_bucket', 'board_id'],
               'ind_abn', min_n=50)
r4['口径'] = '涨停_行业外溢'

out = pd.concat([r1, r2, r3, r4], ignore_index=True)
out.to_csv(os.path.join(OUT, '事件研究_汇总.csv'), index=False, encoding='utf-8-sig')
for nm, x in [('龙虎榜_个股', r1), ('龙虎榜_行业外溢', r2), ('涨停_个股', r3), ('涨停_行业外溢', r4)]:
    x.to_csv(os.path.join(OUT, f'事件研究_{nm}.csv'), index=False, encoding='utf-8-sig')

# 可买率/分布
buy = L1.groupby('chg_bucket', observed=True).agg(
    事件数=('thscode', 'size'),
    次日买不到=('next_unbuyable', 'mean'),
    次日高开7=('next_high_open', 'mean')).reset_index()
buy.to_csv(os.path.join(OUT, '龙虎榜_可买率.csv'), index=False, encoding='utf-8-sig')
print(out[out.h.isin([1, 3, 5])].sort_values(['口径', '分组', 'h']).to_string(index=False)[:12000])
log('完成')
