# -*- coding: utf-8 -*-
"""把全A等权行情/宽度/情绪从面板末日(2026-10-13)扩展到最新收盘(2026-10-22)。

口径与 2026-10-20 面板一致：
- 个股日收益 = 后复权收盘 / 前一日后复权收盘 - 1，横截面均值，单日截断 [-20%, +20%]
- 等权指数 = 累乘
- 宽度20/60 = 收盘>MA20 / 收盘>MA60 的股票占比
- 涨停/跌停：主板 9.8%、创业/科创 19.6%、北交 29.6%（近似）
- 成交额热度 = 全A成交额 / 20日均额
- 情绪分 = 0.35*z(涨停占比)+0.30*z(成交额热度)+0.20*z(250日新高占比)+0.15*z(宽度20)，250日滚动z
- 离散度 = 个股20日收益横截面标准差
只用当日及以前数据。
"""
import duckdb, pandas as pd, numpy as np, pickle, os

BASE = '/Users/admin/WorkSpace/trading'
OUT = os.path.join(BASE, '2026-10-22_当前阶段_情绪同态_50%策略/01_数据')
PANEL = os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl')
DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'

P = pickle.load(open(PANEL, 'rb'))
dates0 = pd.DatetimeIndex(P['meta']['dates'])
mkt0 = pd.Series(P['market_index'], index=dates0)
last0 = dates0[-1]  # 2026-10-13

con = duckdb.connect(DB, read_only=True)
# 取面板末日前 320 个交易日到最新，计算 MA 与收益
start = (last0 - pd.Timedelta(days=520)).strftime('%Y-%m-%d')
df = con.execute(f"""
    select thscode, date, open, high, low, close, volume, amount
    from v_daily_hfq where date >= '{start}'
    order by thscode, date
""").df()
con.close()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['thscode', 'date'])
g = df.groupby('thscode', sort=False)
df['ret1'] = g['close'].pct_change()
df['ma20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=15).mean())
df['ma60'] = g['close'].transform(lambda s: s.rolling(60, min_periods=40).mean())
df['ma120'] = g['close'].transform(lambda s: s.rolling(120, min_periods=80).mean())
df['high250'] = g['close'].transform(lambda s: s.rolling(250, min_periods=120).max())
df['ret20'] = g['close'].transform(lambda s: s / s.shift(20) - 1)

def board_limit(code):
    if code.endswith('.BJ'):
        return 0.296
    if code.startswith(('300', '301', '688', '689')):
        return 0.196
    return 0.098

lim = df['thscode'].map(board_limit).values
df['limit_up'] = (df['ret1'] >= lim - 0.002).astype(float)
df['limit_down'] = (df['ret1'] <= -lim + 0.002).astype(float)
df['newhigh250'] = (df['close'] >= df['high250'] * 0.999).astype(float)

# 逐日横截面
agg = df.groupby('date').agg(
    n=('close', 'size'),
    mkt_ret=('ret1', lambda s: s.clip(-0.2, 0.2).mean()),
    breadth20=('close', lambda s: np.nan),  # placeholder
)
rows = []
for dt, x in df.groupby('date', sort=True):
    ok = x['close'].notna()
    if not ok.any():
        continue
    br20 = (x.loc[ok, 'close'] > x.loc[ok, 'ma20']).mean() if x.loc[ok, 'ma20'].notna().any() else np.nan
    br60 = (x.loc[ok, 'close'] > x.loc[ok, 'ma60']).mean() if x.loc[ok, 'ma60'].notna().any() else np.nan
    br120 = (x.loc[ok, 'close'] > x.loc[ok, 'ma120']).mean() if x.loc[ok, 'ma120'].notna().any() else np.nan
    rows.append(dict(
        date=dt, n=int(ok.sum()),
        mkt_ret=x.loc[ok, 'ret1'].clip(-0.2, 0.2).mean(),
        breadth20=br20, breadth60=br60, breadth120=br120,
        limitup=x.loc[ok, 'limit_up'].mean(), limitdown=x.loc[ok, 'limit_down'].mean(),
        newhigh=x.loc[ok, 'newhigh250'].mean(),
        amt=float(x.loc[ok, 'amount'].sum()),
        dispersion=x.loc[ok, 'ret20'].std(),
        med_ret1=x.loc[ok, 'ret1'].median(),
    ))
M = pd.DataFrame(rows).set_index('date').sort_index()
M['amt_ratio'] = M['amt'] / M['amt'].rolling(20, min_periods=15).mean()

# 只保留面板最后日之后，用面板指数衔接；同时重算 10-13 之前的市场特征做校验
# 扩展指数：从面板最后日的指数值起，乘以后续日子的 mkt_ret
ext = M.loc[M.index > last0].copy()
ext['market_index'] = float(mkt0.iloc[-1]) * (1 + ext['mkt_ret']).cumprod()
# 拼成完整序列：面板历史 + 扩展（索引取并集，面板交易日不一定都在 520 日窗口内）
idx = dates0.union(M.index)
full = pd.DataFrame(index=idx)
full['market_index'] = np.nan
full.loc[dates0, 'market_index'] = mkt0.values
full.loc[ext.index, 'market_index'] = ext['market_index'].values
# 10-13 当天用面板值；扩展日从 10-14 开始
full['market_index'] = full['market_index'].ffill()
for c in ['breadth20', 'breadth60', 'breadth120', 'limitup', 'limitdown', 'newhigh', 'amt_ratio', 'dispersion', 'med_ret1', 'mkt_ret']:
    full[c] = M[c]
# 面板历史的市场特征优先（口径一致），扩展日再用重算
for c, pc in [('breadth20', 'market_breadth20'), ('breadth60', 'market_breadth60'),
              ('limitup', 'market_limitup'), ('limitdown', 'market_limitdown'),
              ('newhigh', 'market_newhigh'), ('amt_ratio', 'market_amt_ratio'),
              ('dispersion', 'market_dispersion'), ('mkt_ret', None)]:
    if pc and pc in P:
        full.loc[dates0, c] = P[pc]
    elif c == 'mkt_ret':
        full.loc[dates0, 'mkt_ret'] = full.loc[dates0, 'market_index'].pct_change()

full['ma20'] = full['market_index'].rolling(20, min_periods=15).mean()
full['ma60'] = full['market_index'].rolling(60, min_periods=40).mean()
full['ma120'] = full['market_index'].rolling(120, min_periods=80).mean()
full['ma250'] = full['market_index'].rolling(250, min_periods=150).mean()
full['slope60'] = full['ma60'] / full['ma60'].shift(20) - 1
for h in [5, 20, 60, 120, 250]:
    full[f'ret{h}'] = full['market_index'] / full['market_index'].shift(h) - 1
full['dd250'] = full['market_index'] / full['market_index'].rolling(250, min_periods=120).max() - 1
full['vol20'] = full['mkt_ret'].rolling(20, min_periods=15).std() * np.sqrt(252)

def zs(s):
    return (s - s.rolling(250, min_periods=60).mean()) / (s.rolling(250, min_periods=60).std() + 1e-9)
full['sent'] = (0.35 * zs(full['limitup']).fillna(0) + 0.30 * zs(full['amt_ratio']).fillna(0) +
                0.20 * zs(full['newhigh']).fillna(0) + 0.15 * zs(full['breadth20']).fillna(0))

# 因果日度状态（与面板同一套规则）
reg = pd.Series('震荡', index=full.index)
reg[(full.market_index > full.ma60) & (full.slope60 > 0) & (full.breadth20 > 0.50)] = '上行'
reg[(full.market_index < full.ma60) & (full.slope60 < 0) & (full.breadth20 < 0.50)] = '下行'
reg[(full.market_index > full.ma60) & (full.market_index > full.ma120) & (full.slope60 > 0.005) & (full.breadth20 > 0.60)] = '强上行'
reg[(full.market_index < full.ma60) & (full.market_index < full.ma120) & (full.slope60 < -0.005) & (full.breadth20 < 0.40)] = '强下行'
full['regime'] = reg

full.to_csv(os.path.join(OUT, '市场日频_2016_2026.csv'), encoding='utf-8-sig', float_format='%.6f')

# 当前快照 + 历史百分位
last = full.iloc[-1]
snap = []
for c in ['market_index', 'ret5', 'ret20', 'ret60', 'ret120', 'ret250', 'dd250', 'breadth20', 'breadth60',
          'breadth120', 'limitup', 'limitdown', 'newhigh', 'amt_ratio', 'dispersion', 'vol20', 'sent']:
    v = last[c]
    pct = (full[c].dropna() <= v).mean()
    snap.append(dict(指标=c, 当前值=v, 历史百分位=pct,
                     历史中位=full[c].median(), 历史P10=full[c].quantile(.1), 历史P90=full[c].quantile(.9)))
pd.DataFrame(snap).to_csv(os.path.join(OUT, '当前状态_2026-10-22.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')

print(full.tail(8)[['market_index', 'mkt_ret', 'breadth20', 'breadth60', 'limitup', 'limitdown', 'amt_ratio', 'dispersion', 'sent', 'regime']].round(4).to_string())
print('\n当前快照:')
print(pd.DataFrame(snap).round(4).to_string(index=False))
