# -*- coding: utf-8 -*-
"""2026-10-22 收盘的行业宽度/动量与超跌观察池（不是买入信号）。"""
import duckdb, pandas as pd, numpy as np, pickle, os

BASE = '/Users/admin/WorkSpace/trading'
OUT = os.path.join(BASE, '2026-10-22_当前阶段_情绪同态_50%策略/03_结果')
P = pickle.load(open(os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl'), 'rb'))
ind_names = P['meta']['industries']
codes = P['meta']['codes']
ind_last = P['industry_id'].iloc[-1].values
code2ind = {c: (ind_names[int(ind_last[j]) - 1] if np.isfinite(ind_last[j]) and ind_last[j] > 0 else '未知')
            for j, c in enumerate(codes)}

con = duckdb.connect('/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb', read_only=True)
df = con.execute("""
    select thscode, date, open, high, low, close, amount
    from v_daily_hfq where date >= '2026-05-01' order by thscode, date
""").df()
con.close()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['thscode', 'date'])
g = df.groupby('thscode', sort=False)
df['ma20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=15).mean())
df['ma60'] = g['close'].transform(lambda s: s.rolling(60, min_periods=40).mean())
df['r20'] = g['close'].transform(lambda s: s / s.shift(20) - 1)
df['r5'] = g['close'].transform(lambda s: s / s.shift(5) - 1)
df['amt20'] = g['amount'].transform(lambda s: s.rolling(20, min_periods=15).mean())
df['high60'] = g['high'].transform(lambda s: s.rolling(60, min_periods=40).max())
df['dd60'] = df['close'] / df['high60'] - 1
# ATR14
tr = pd.concat([df['high'] - df['low'], (df['high'] - g['close'].shift()).abs(), (df['low'] - g['close'].shift()).abs()], axis=1).max(axis=1)
df['atr'] = tr.groupby(df['thscode']).transform(lambda s: s.rolling(14, min_periods=10).mean()) / df['close']
last = df[df.date == df.date.max()].copy()
last['行业'] = last.thscode.map(code2ind)

# 行业宽度/动量
rows = []
for name, x in last.groupby('行业'):
    if name == '未知' or len(x) < 3: continue
    rows.append(dict(行业=name, 股票数=len(x), 宽度20=(x.close > x.ma20).mean(),
                     宽度60=(x.close > x.ma60).mean(), 行业20日=x.r20.median(),
                     行业5日=x.r5.median(), 行业成交额亿=x.amount.sum() / 1e8))
I = pd.DataFrame(rows).sort_values('行业20日', ascending=False)
I.to_csv(os.path.join(OUT, '当前行业_2026-10-22.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')

# 市场宽度复算
mb20 = (last.close > last.ma20).mean(); mb60 = (last.close > last.ma60).mean()
print('市场宽度20 %.4f 宽度60 %.4f 股票数 %d' % (mb20, mb60, len(last)))
print('\n行业前8:')
print(I.head(8).round(4).to_string(index=False))
print('\n行业后8:')
print(I.tail(8).round(4).to_string(index=False))

# 超跌观察池：R20最低10%，ATR高，价格低，流动性
base = last.dropna(subset=['r20', 'atr', 'amt20', 'close']).copy()
base = base[(base.amt20 >= 3e7) & (base.close >= 2)]
q20 = base.r20.quantile(.10); qatr = base.atr.quantile(.60)
pool = base[(base.r20 <= q20) & (base.atr >= qatr)].copy()
pool = pool.sort_values('r20')
pool['后复权收盘'] = pool.close
pool['20日均额万'] = pool.amt20 / 1e4
pool['ATR%'] = pool.atr
pool['R20'] = pool.r20
pool['DD60'] = pool.dd60
pool[['thscode', '行业', '后复权收盘', '20日均额万', 'ATR%', 'R20', 'DD60']].to_csv(
    os.path.join(OUT, '当前超跌观察池_2026-10-22.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
print('\n超跌观察池 n=%d, R20<=%.3f, ATR>=%.3f' % (len(pool), q20, qatr))
print(pool[['thscode', '行业', 'close', 'r20', 'atr', 'amt20']].head(20).round(4).to_string(index=False))
