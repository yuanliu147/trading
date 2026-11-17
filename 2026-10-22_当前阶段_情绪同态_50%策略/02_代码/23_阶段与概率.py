# -*- coding: utf-8 -*-
"""阶段表、当前同态的条件概率、最近邻（含长期趋势特征）。"""
import pandas as pd, numpy as np, os

BASE = '/Users/admin/WorkSpace/trading/2026-10-22_当前阶段_情绪同态_50%策略'
OUT = os.path.join(BASE, '03_结果')
f_all = pd.read_csv(os.path.join(BASE, '01_数据/市场日频_2016_2026.csv'), parse_dates=['date']).set_index('date').sort_index()
f_all = f_all.loc['2016-10-13':]
f_all['idx_pct'] = f_all.market_index.rank(pct=True)
f_all['ret250'] = f_all.market_index / f_all.market_index.shift(250) - 1
f_all['ret500'] = f_all.market_index / f_all.market_index.shift(500) - 1
f_all['dd250'] = f_all.market_index / f_all.market_index.rolling(250, min_periods=120).max() - 1
for h in [20, 60, 120]:
    f_all[f'f{h}'] = f_all.market_index.shift(-h) / f_all.market_index - 1
    f_all[f'min{h}'] = f_all.market_index.iloc[::-1].rolling(h, min_periods=1).min().iloc[::-1] / f_all.market_index - 1

# 同态/最近邻从 2017-01 起（250 日特征预热），阶段表仍用 2016-10 起
f = f_all.loc['2017-01-01':].copy()
f['dd250'] = f.market_index / f.market_index.rolling(250, min_periods=120).max() - 1

# 阶段表（用等权指数）
phases = [('2016-10~2018-01 慢牛', '2016-10-13', '2018-01-24'),
          ('2018-01~2019-01 熊', '2018-01-24', '2019-01-04'),
          ('2019-01~2021-12 结构牛', '2019-01-04', '2021-12-13'),
          ('2021-12~2024-02 熊', '2021-12-13', '2024-02-05'),
          ('2024-02~2024-09 底部', '2024-02-05', '2024-09-18'),
          ('2024-09~2026-05 主升', '2024-09-18', '2026-05-13'),
          ('2026-05~至今 调整', '2026-05-13', '2026-10-22')]
rows = []
for name, a, b in phases:
    x = f_all.loc[a:b]
    rows.append(dict(阶段=name, 起点=a, 终点=b, 交易日=len(x),
                     指数收益=x.market_index.iloc[-1] / x.market_index.iloc[0] - 1,
                     最大回撤=(x.market_index / x.market_index.cummax() - 1).min(),
                     平均宽度20=x.breadth20.mean(), 最低宽度20=x.breadth20.min(),
                     平均情绪=x.sent.mean(), 涨停占比均值=x.limitup.mean(), 跌停占比均值=x.limitdown.mean()))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(OUT, '阶段表.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
print('=== 阶段表 ===')
print(P.round(4).to_string(index=False))

# 当前同态事件（条件 A 的首日，20日去重）
mask = (f.dd250 < -.15) & (f.breadth20 < .20)
dates = f.index[mask.fillna(False)]
eps = []
for d in dates:
    if not eps or (f.index.get_loc(d) - f.index.get_loc(eps[-1][-1])) > 20:
        eps.append([d])
    else:
        eps[-1].append(d)
starts = [e[0] for e in eps]
E = f.loc[starts].copy()
E['高水位'] = E.idx_pct > .70
E['破MA250'] = E.market_index < E.ma250
E['一年为负'] = E.ret250 < 0
E.to_csv(os.path.join(OUT, '当前同态_事件表.csv'), encoding='utf-8-sig', float_format='%.4f')
print('\n=== 当前同态事件（条件A首日） ===')
print(E[['market_index', 'idx_pct', 'ret250', 'dd250', 'breadth20', 'sent', 'f20', 'f60', 'f120', '高水位', '破MA250']].round(4).to_string())

print('\n=== 子样本 ===')
for name, g in [('全部', E), ('高水位(>70%)', E[E.高水位]), ('非高水位', E[~E.高水位]),
                ('破MA250', E[E.破MA250]), ('未破MA250', E[~E.破MA250]),
                ('高水位且破MA250', E[E.高水位 & E.破MA250])]:
    print(f'{name:14s} n={len(g):2d} f20 {g.f20.mean():+.3f} f60 {g.f60.mean():+.3f} f120 {g.f120.mean():+.3f} '
          f'win60 {(g.f60 > 0).mean():.2f} win120 {(g.f120 > 0).mean():.2f}')

# 最近邻（含长期趋势）
feat = ['ret5', 'ret20', 'ret60', 'ret120', 'ret250', 'dd250', 'breadth20', 'breadth60',
        'limitup', 'limitdown', 'newhigh', 'amt_ratio', 'dispersion', 'vol20', 'sent', 'idx_pct']
X = f[feat].copy()
Z = (X - X.mean()) / X.std()
cur = Z.iloc[-1]
dist = np.sqrt(((Z - cur) ** 2).sum(axis=1))
nn = pd.DataFrame({'距离': dist, '日期': f.index}).sort_values('距离')
# 去掉最近 60 个交易日（与当前同一段）和重复相邻日期
nn = nn[nn.日期 <= f.index[-1] - pd.Timedelta(days=90)]
picked = []
for _, r in nn.iterrows():
    if all(abs((r.日期 - p).days) > 20 for p in picked):
        picked.append(r.日期)
    if len(picked) >= 15: break
N = f.loc[picked].copy()
N['距离'] = [dist.loc[d] for d in picked]
N = N.sort_values('距离')
N[['距离', 'market_index', 'idx_pct', 'ret250', 'dd250', 'breadth20', 'sent', 'f20', 'f60', 'f120']].to_csv(
    os.path.join(OUT, '当前同态_最近邻_含长期.csv'), encoding='utf-8-sig', float_format='%.4f')
print('\n=== 最近邻（含长期趋势，去同段） ===')
print(N[['距离', 'market_index', 'idx_pct', 'ret250', 'dd250', 'breadth20', 'sent', 'f20', 'f60', 'f120']].round(4).to_string())
