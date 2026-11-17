# -*- coding: utf-8 -*-
"""最终策略：龙虎榜上涨事件 + 次日低开的日内反转。
主方案：全部龙虎榜上涨事件（不要求净卖），T+1 低开 0.5%–5%，按低开幅度最深排序，
最多 N=10 只，每只 1/N；T+1 开盘买、收盘卖；45bp 单边成本。
对照：龙虎榜净卖子集、全 A 低开。
"""
import os, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading/2026-10-22_龙虎榜涨停_行业影响与策略'
DATA = os.path.join(BASE, '01_数据')
OUT = os.path.join(BASE, '03_结果')
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

M = pd.read_parquet(os.path.join(DATA, '市场日频.parquet'))
M['date'] = pd.to_datetime(M.date)
M = M.sort_values('date').reset_index(drop=True)
MD = pd.DatetimeIndex(M.date)
M['bench_nav'] = (1 + M.mkt_ret1_eq.fillna(0)).cumprod()
BENCH = M.set_index('date')['mkt_ret1_eq'].fillna(0)

S = pd.read_parquet(os.path.join(DATA, '个股日频.parquet'),
                    columns=['date', 'thscode', 'industry_id', 'st', 'age', 'liq', 'amt20',
                             'next_open', 'next_close', 'next_low', 'next_high', 'next_ld_seal',
                             'next2_open', 'next2_close', 'next2_ld_seal',
                             'next_limit_rate', 'next_unbuyable', 'next_open_ret', 'next_date'])
S['date'] = pd.to_datetime(S.date)
nd = pd.Series(MD[1:].values, index=MD[:-1].values)
S['next_trade_date'] = S.date.map(nd)
S['gap_ok'] = S.next_date.eq(S.next_trade_date)
S['ld_open'] = S.next_open_ret <= -(S.next_limit_rate - 0.002)
S['stamp'] = np.where(S.date >= '2023-08-28', 0.0005, 0.001)
BASE_OK = ((S.st == 0) & (S.age >= 60) & (S.liq == 1) & S.next_open.notna() & S.next_close.notna() &
           (S.next_unbuyable == 0) & (~S.ld_open) & S.gap_ok &
           (S.next_open_ret <= 0.07))
S = S[BASE_OK].copy()

L = pd.read_parquet(os.path.join(DATA, '龙虎榜事件_含前瞻.parquet'))
L['date'] = pd.to_datetime(L.date)
L = L.drop(columns=[c for c in ['industry_id', 'next_open_ret', 'next_unbuyable', 'next_high_open',
                                'next_date', 'next_limit_rate', 'next_ld_seal', 'gap_ok',
                                'next_limit_down_open', 'next_trade_date'] if c in L.columns])
L = L.merge(S[['date', 'thscode', 'next_open', 'next_close', 'next_low', 'next_high', 'next_ld_seal',
               'next2_open', 'next2_close', 'next2_ld_seal',
               'next_limit_rate', 'next_unbuyable', 'next_open_ret', 'next_date', 'next_trade_date',
               'gap_ok', 'ld_open', 'stamp', 'industry_id']], on=['date', 'thscode'], how='inner')
L = L[(L.next_open.notna()) & (L.next_close.notna()) & (L.next_unbuyable == 0) & (~L.ld_open) &
      L.gap_ok & (L.next_open_ret <= 0.07)].copy()
L = L[~L.reason_type.isin(['跌幅', '连续跌幅', '退市'])].copy()
L['period'] = np.where(L.date.dt.year <= 2021, '设计', np.where(L.date.dt.year <= 2023, '验证', 'OOS'))
log(f'可交易龙虎榜上涨事件 {len(L):,}')


def run_lhb(N=10, cost=0.0045, gap_lo=-0.05, gap_hi=-0.005, use_net=False, net_thr=-0.015,
            stop=None, rank='gap'):
    x = L[(L.next_open_ret >= gap_lo) & (L.next_open_ret <= gap_hi)].copy()
    if use_net:
        x = x[x.net_ratio <= net_thr].copy()
    if rank == 'gap':
        x = x.sort_values(['date', 'next_open_ret'], ascending=[True, True])
    else:
        x = x.sort_values(['date', 'net_ratio'], ascending=[True, True])
    x['rk'] = x.groupby('date').cumcount()
    x = x[x.rk < N].copy()
    entry = x.next_open.values; exitp = x.next_close.values.copy()
    # T+1 收盘跌停/停牌卖不出：顺延到 T+2 开盘；这是实盘约束，不是入场过滤
    stuck = (x.next_ld_seal.fillna(0).values == 1) | (x.next_close.values <= 0)
    if stuck.any():
        sub = x.loc[stuck]
        alt = sub.next2_open.values
        alt = np.where(np.isfinite(alt), alt, sub.next2_close.values)
        exitp[stuck] = alt
    if stop is not None:
        hit = x.next_low.values <= entry * (1 - stop)
        exitp[hit] = entry[hit] * (1 - stop)
    x['ret'] = exitp * (1 - cost - x.stamp.values) / (entry * (1 + cost)) - 1
    x['w'] = 1.0 / N
    x['wr'] = x.w * x.ret
    d = x.groupby('next_date')['wr'].sum()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index().reindex(MD).fillna(0)
    return d, x


def run_allgap(N=10, cost=0.0040, gap_lo=-0.05, gap_hi=-0.005):
    x = S[(S.next_open_ret >= gap_lo) & (S.next_open_ret <= gap_hi)].copy()
    x = x.sort_values(['date', 'next_open_ret'], ascending=[True, True])
    x['rk'] = x.groupby('date').cumcount()
    x = x[x.rk < N].copy()
    exitp = x.next_close.values.copy()
    stuck = (x.next_ld_seal.fillna(0).values == 1) | (x.next_close.values <= 0)
    if stuck.any():
        sub = x.loc[stuck]
        alt = sub.next2_open.values
        alt = np.where(np.isfinite(alt), alt, sub.next2_close.values)
        exitp[stuck] = alt
    x['ret'] = exitp * (1 - cost - x.stamp.values) / (x.next_open.values * (1 + cost)) - 1
    x['w'] = 1.0 / N
    x['wr'] = x.w * x.ret
    d = x.groupby('next_date')['wr'].sum()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index().reindex(MD).fillna(0)
    return d, x


def stats(d, name, trades, N):
    nav = (1 + d).cumprod()
    n = len(d)
    cagr = nav.iloc[-1] ** (252 / n) - 1
    dd = (nav / nav.cummax() - 1).min()
    sh = d.mean() / (d.std() + 1e-9) * np.sqrt(252)
    sortino = d.mean() / (d[d < 0].std() + 1e-9) * np.sqrt(252)
    bench = BENCH.reindex(d.index).fillna(0)
    bnav = (1 + bench).cumprod()
    bcagr = bnav.iloc[-1] ** (252 / n) - 1
    bdd = (bnav / bnav.cummax() - 1).min()
    monthly = (1 + d).groupby(d.index.to_period('M')).prod() - 1
    rng = np.random.default_rng(20261022)
    bs = []
    for _ in range(1000):
        idx = rng.integers(0, len(monthly), len(monthly))
        r = monthly.iloc[idx]
        bs.append(((1 + r).prod() ** (12 / len(r)) - 1, r.mean() / (r.std() + 1e-9) * np.sqrt(12)))
    bs = np.array(bs)
    Ntr = 80
    sr = sh
    gamma = 0.5772156649
    srs = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    sr0 = srs.std(ddof=1) * ((1 - gamma) * norm.ppf(1 - 1 / Ntr) + gamma * norm.ppf(1 - 1 / (Ntr * np.e)))
    sk = skew(d.values); ku = kurtosis(d.values, fisher=False)
    dsr = norm.cdf((sr - sr0) * np.sqrt(n - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    return dict(策略=name, 天数=n, 年化=cagr, 基准年化=bcagr, 超额年化=cagr - bcagr,
                最大回撤=dd, 基准最大回撤=bdd, Sharpe=sh, Sortino=sortino, 日胜率=(d > 0).mean(),
                日均收益=d.mean(), 日均持仓=trades.groupby('next_date').size().reindex(d.index).fillna(0).mean(),
                交易数=len(trades), 平均单笔=trades.ret.mean(), 单笔胜率=(trades.ret > 0).mean(),
                单笔中位=trades.ret.median(),
                年化波动=d.std() * np.sqrt(252), 月度CAGR_CI低=np.quantile(bs[:, 0], 0.025),
                月度CAGR_CI高=np.quantile(bs[:, 0], 0.975), 月度Sharpe_CI低=np.quantile(bs[:, 1], 0.025),
                月度Sharpe_CI高=np.quantile(bs[:, 1], 0.975), DSR=dsr, 试验数=Ntr)


rows = []
for name, N, cost, fn, kw in [
    ('稳健_N15_45bp', 15, 0.0045, 'lhb', dict()),
    ('均衡_N10_45bp', 10, 0.0045, 'lhb', dict()),
    ('进攻_N5_45bp', 5, 0.0045, 'lhb', dict()),
    ('净卖对照_N5_45bp', 5, 0.0045, 'lhb', dict(use_net=True)),
    ('全A低开对照_N10_40bp', 10, 0.0040, 'all', dict()),
]:
    if fn == 'lhb':
        d, tr = run_lhb(N=N, cost=cost, **kw)
    else:
        d, tr = run_allgap(N=N, cost=cost)
    m = stats(d, name, tr, N)
    rows.append(m)
    d.to_frame('ret').to_csv(os.path.join(OUT, f'最终策略_权益_{name}.csv'), encoding='utf-8-sig')
    tr.to_csv(os.path.join(OUT, f'最终策略_逐笔_{name}.csv'), index=False, encoding='utf-8-sig')
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)

summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, '最终策略_指标.csv'), index=False, encoding='utf-8-sig')

# 主策略分年/分期
d, tr = run_lhb(N=10, cost=0.0045)
yr = (1 + d).groupby(d.index.year).prod() - 1
per_label = np.where(d.index.year <= 2021, '设计', np.where(d.index.year <= 2023, '验证', 'OOS'))
per = (1 + d).groupby(per_label).prod() ** (252 / pd.Series(per_label, index=d.index).groupby(per_label).size()) - 1
yr.to_csv(os.path.join(OUT, '最终策略_分年.csv'), encoding='utf-8-sig')
per.to_csv(os.path.join(OUT, '最终策略_分期.csv'), encoding='utf-8-sig')
print('分年\n', yr.round(3).to_string())
print('分期\n', per.round(3).to_string())

# 成本敏感（主策略 N=10）
rows = []
for c in [0.002, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006]:
    dd, tt = run_lhb(N=10, cost=c)
    nav = (1 + dd).cumprod(); cagr = nav.iloc[-1] ** (252 / len(dd)) - 1
    rows.append(dict(单边成本=c, 年化=cagr, 最大回撤=(nav / nav.cummax() - 1).min(),
                     Sharpe=dd.mean() / (dd.std() + 1e-9) * np.sqrt(252)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, '最终策略_成本敏感.csv'), index=False, encoding='utf-8-sig')
print('成本敏感\n', pd.DataFrame(rows).round(3).to_string(index=False))

# 参数邻域（N × 低开区间；不设净卖阈值）
rows = []
for N in [5, 10, 15, 20]:
    for gap in [(-0.05, -0.005), (-0.05, -0.01), (-0.03, -0.005), (-0.03, -0.01), (-0.05, -0.03)]:
        for rank in ['gap', 'net']:
            dd, tt = run_lhb(N=N, cost=0.0045, gap_lo=gap[0], gap_hi=gap[1], rank=rank)
            nav = (1 + dd).cumprod()
            rows.append(dict(N=N, 低开下限=gap[0], 低开上限=gap[1], 排序=rank,
                             年化=nav.iloc[-1] ** (252 / len(dd)) - 1,
                             最大回撤=(nav / nav.cummax() - 1).min(),
                             Sharpe=dd.mean() / (dd.std() + 1e-9) * np.sqrt(252),
                             交易数=len(tt)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, '最终策略_参数邻域.csv'), index=False, encoding='utf-8-sig')
print('参数邻域正比例', (pd.DataFrame(rows).年化 > 0).mean())

mon = (1 + d).groupby(d.index.to_period('M')).prod() - 1
mon.to_csv(os.path.join(OUT, '最终策略_月度收益.csv'), encoding='utf-8-sig')

# 用实际参数网格的 Sharpe 离散度重算 DSR（避免用拍脑袋的 srs）
grid = pd.read_csv(os.path.join(OUT, '最终策略_参数邻域.csv'))
Ntr = len(grid)
gamma = 0.5772156649
sr0 = grid.Sharpe.std(ddof=1) * ((1 - gamma) * norm.ppf(1 - 1 / Ntr) + gamma * norm.ppf(1 - 1 / (Ntr * np.e)))
print(f'试验数={Ntr}, 试验SR均值={grid.Sharpe.mean():.2f}, 标准差={grid.Sharpe.std():.2f}, SR0={sr0:.2f}')
for i, row in summ.iterrows():
    eq = pd.read_csv(os.path.join(OUT, f"最终策略_权益_{row.策略}.csv"), index_col=0, parse_dates=True).iloc[:, 0]
    sr = eq.mean() / eq.std() * np.sqrt(252)
    sk = skew(eq.values); ku = kurtosis(eq.values, fisher=False)
    summ.loc[i, 'DSR'] = norm.cdf((sr - sr0) * np.sqrt(len(eq) - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    summ.loc[i, '试验数'] = Ntr
summ.to_csv(os.path.join(OUT, '最终策略_指标.csv'), index=False, encoding='utf-8-sig')
print(summ[['策略', '年化', '最大回撤', 'Sharpe', 'DSR', '试验数']].round(3).to_string(index=False))
log('完成')
