# -*- coding: utf-8 -*-
"""龙虎榜次日反转策略组合回测（T 日收盘信号，T+1 开盘买，T+1 收盘卖）。
- 主策略：龙虎榜净卖 + 次日低开，日内反转。
- 对照：净卖不筛低开、净买、机构净买、行业冷/热过滤。
- 最多 N 只，每只 1/N 权益，未用仓位现金；计双边成本。
"""
import os, time, itertools, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading/2026-10-22_龙虎榜涨停_行业影响与策略'
DATA = os.path.join(BASE, '01_数据')
OUT = os.path.join(BASE, '03_结果')
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

L = pd.read_parquet(os.path.join(DATA, '龙虎榜事件_含前瞻.parquet'))
S = pd.read_parquet(os.path.join(DATA, '个股日频.parquet'),
                    columns=['date', 'thscode', 'next_open', 'next_close', 'next_low', 'next_high',
                             'next_ld_seal', 'next_limit_rate', 'next_unbuyable', 'next_open_ret',
                             'next_date', 'amt20', 'industry_id'])
S['date'] = pd.to_datetime(S.date)
L['date'] = pd.to_datetime(L.date)
L = L.drop(columns=[c for c in ['industry_id'] if c in L.columns])
L = L.merge(S, on=['date', 'thscode'], how='left', suffixes=('', '_s'))
M = pd.read_parquet(os.path.join(DATA, '市场日频.parquet'))
M['date'] = pd.to_datetime(M.date)
M = M.sort_values('date').reset_index(drop=True)
M['bench_nav'] = (1 + M.mkt_ret1_eq.fillna(0)).cumprod()
# 行业热度（用于过滤）
A = pd.read_parquet(os.path.join(DATA, '行业日频.parquet'), columns=['date', 'industry_id', 'lu_share', 'lhb_share', 'lhb_net_ratio'])
A['date'] = pd.to_datetime(A.date)
L = L.merge(A, on=['date', 'industry_id'], how='left', suffixes=('', '_ind'))

# 基础可交易过滤
base = ((L.st == 0) & (L.age >= 60) & (L.liq == 1) &
        (L.next_open.notna()) & (L.next_close.notna()) &
        (L.next_unbuyable == 0) & (L.gap_ok == 1) &
        (L.next_limit_down_open == 0) & (L.next_ld_seal == 0) &
        (L.next_open_ret <= 0.07))
L = L[base].copy()
L['period'] = np.where(L.date.dt.year <= 2021, '设计', np.where(L.date.dt.year <= 2023, '验证', 'OOS'))
L['entry'] = L.next_open.astype(float)
L['exit_close'] = L.next_close.astype(float)
L['low_t1'] = L.next_low.astype(float)
L['high_t1'] = L.next_high.astype(float)
L['stamp'] = np.where(L.date >= '2023-08-28', 0.0005, 0.001)

# 信号
SIGS = {}
SIGS['净卖全部'] = (L.net_ratio <= -0.015)
SIGS['净卖+低开0~5%'] = (L.net_ratio <= -0.015) & (L.next_open_ret <= -0.005) & (L.next_open_ret >= -0.05)
SIGS['净卖+低开1~5%'] = (L.net_ratio <= -0.015) & (L.next_open_ret <= -0.01) & (L.next_open_ret >= -0.05)
SIGS['净卖+低开0~3%'] = (L.net_ratio <= -0.015) & (L.next_open_ret <= -0.005) & (L.next_open_ret >= -0.03)
SIGS['净卖+低开3~5%'] = (L.net_ratio <= -0.015) & (L.next_open_ret <= -0.03) & (L.next_open_ret >= -0.05)
SIGS['净卖+涨停'] = (L.net_ratio <= -0.015) & (L.chg >= 0.095)
SIGS['净卖+行业冷'] = (L.net_ratio <= -0.015) & (L.lu_share <= 0.005)
SIGS['净卖+行业热'] = (L.net_ratio <= -0.015) & (L.lu_share >= 0.02)
SIGS['净买+低开0~5%'] = (L.net_ratio >= 0.05) & (L.next_open_ret <= -0.005) & (L.next_open_ret >= -0.05)
SIGS['机构净买+低开0~5%'] = (L.org_net > 0) & (L.next_open_ret <= -0.005) & (L.next_open_ret >= -0.05)


def run(name, cond, N=5, cost=0.003, stop=None, tp=None, rank='net'):
    x = L[cond].copy()
    if len(x) == 0:
        return None, None
    x = x.sort_values(['date', 'net_ratio'], ascending=[True, True])
    # 每日选最多 N 只
    x['rk'] = x.groupby('date').cumcount()
    x = x[x.rk < N].copy()
    # 单票收益（含成本；止损/止盈用 T+1 日内高低点）
    entry = x.entry.values
    exitp = x.exit_close.values.copy()
    if stop is not None:
        hit = x.low_t1.values <= entry * (1 - stop)
        exitp[hit] = entry[hit] * (1 - stop)
    if tp is not None:
        hit_tp = (x.high_t1.values >= entry * (1 + tp)) & ~(x.low_t1.values <= entry * (1 - stop if stop else -1))
        exitp[hit_tp] = entry[hit_tp] * (1 + tp)
    buy_cost = cost
    sell_cost = x.stamp.values + cost
    x['ret'] = exitp * (1 - sell_cost) / (entry * (1 + buy_cost)) - 1
    # 每日组合：每只 1/N，未用现金
    x['w'] = 1.0 / N
    daily = x.groupby('next_date').apply(lambda z: pd.Series({
        'ret': (z.w * z.ret).sum(),
        'n': len(z),
        'gross': (z.w * (z.exit_close / z.entry - 1)).sum(),
    }), include_groups=False).reset_index().rename(columns={'next_date': 'date'})
    daily['date'] = pd.to_datetime(daily.date)
    daily = daily.set_index('date').sort_index()
    # 对齐市场交易日，缺失=0
    daily = daily.reindex(pd.DatetimeIndex(M.date)).fillna({'ret': 0, 'n': 0, 'gross': 0})
    daily['nav'] = (1 + daily.ret).cumprod()
    return daily, x


def metrics(daily, name):
    r = daily.ret
    nav = daily.nav
    n = len(r)
    cagr = nav.iloc[-1] ** (252 / max(n, 1)) - 1
    dd = (nav / nav.cummax() - 1).min()
    sh = r.mean() / (r.std() + 1e-9) * np.sqrt(252)
    win = (r > 0).mean()
    return dict(场景=name, 天数=n, 年化=cagr, 最大回撤=dd, Sharpe=sh, 日胜率=win,
                日均收益=r.mean(), 日均持仓=daily.n.mean(), 总交易=int(daily.n.sum()))


rows = []
for nm, cond in SIGS.items():
    for N in [3, 5, 10]:
        for cost in [0.002, 0.003, 0.004]:
            daily, trades = run(nm, cond, N=N, cost=cost)
            if daily is None:
                continue
            m = metrics(daily, f'{nm}|N{N}|成本{cost*1000:.0f}bp')
            rows.append(m)
out = pd.DataFrame(rows).sort_values('年化', ascending=False)
out.to_csv(os.path.join(OUT, '次日反转_参数网格.csv'), index=False, encoding='utf-8-sig')
print(out.head(30).to_string(index=False))

# 主策略分年/分期
main_name = '净卖+低开0~5%'
main_cond = SIGS[main_name]
for cost in [0.002, 0.003, 0.004, 0.005]:
    daily, trades = run(main_name, main_cond, N=5, cost=cost)
    if daily is None:
        continue
    daily['year'] = daily.index.year
    yr = daily.groupby('year').ret.apply(lambda s: (1 + s).prod() - 1)
    per = daily.groupby(np.where(daily.index.year <= 2021, '设计', np.where(daily.index.year <= 2023, '验证', 'OOS'))).ret.apply(
        lambda s: (1 + s).prod() ** (252 / len(s)) - 1)
    print(f'\n== {main_name} 成本{cost*1000:.0f}bp ==')
    print(yr.round(3).to_string())
    print(per.round(3).to_string())
    if cost == 0.003:
        daily.to_csv(os.path.join(OUT, '次日反转_主策略_权益.csv'), encoding='utf-8-sig')
        trades.to_csv(os.path.join(OUT, '次日反转_主策略_逐笔.csv'), index=False, encoding='utf-8-sig')
        print(trades.groupby('period').agg(n=('ret', 'size'), mean=('ret', 'mean'), win=('ret', lambda x: (x > 0).mean())).round(4).to_string())
log('完成')
