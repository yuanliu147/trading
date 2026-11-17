# -*- coding: utf-8 -*-
"""从当前同态低点，到“市场宽度>50”“行业闸门打开”的时间分布。
行业闸门：任一申万一级行业 宽度>=60% 且 行业20日收益>5%。
"""
import pickle, pandas as pd, numpy as np, os

BASE = '/Users/admin/WorkSpace/trading'
OUT = os.path.join(BASE, '2026-10-22_当前阶段_情绪同态_50%策略/03_结果')
P = pickle.load(open(os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl'), 'rb'))
dates = pd.DatetimeIndex(P['meta']['dates'])
V = {k: P[k].values for k in P if k != 'meta'}
mb = pd.Series(V['market_breadth20'], index=dates)
sbr = V['sector_breadth20']; sr = V['sector_ret20']; ind = V['industry_id']
# 每个行业每日中位宽度/收益
gate = []
for i in range(len(dates)):
    ok = np.isfinite(ind[i]) & (ind[i] > 0) & np.isfinite(sbr[i]) & np.isfinite(sr[i])
    if ok.sum() < 100:
        gate.append(np.nan); continue
    df = pd.DataFrame({'ind': ind[i][ok], 'b': sbr[i][ok], 'r': sr[i][ok]})
    gg = df.groupby('ind').median()
    gate.append(float(((gg.b >= .60) & (gg.r > .05)).any()))
gate = pd.Series(gate, index=dates)

# 当前同态事件起点（与 23 相同）
f = pd.read_csv(os.path.join(BASE, '2026-10-22_当前阶段_情绪同态_50%策略/01_数据/市场日频_2016_2026.csv'),
                parse_dates=['date']).set_index('date')
dd = f.dd250.reindex(dates).ffill(); b = f.breadth20.reindex(dates).ffill()
mask = (dd < -.15) & (b < .20)
dts = dates[mask.fillna(False)]
eps = []
for d in dts:
    if not eps or (dates.get_loc(d) - dates.get_loc(eps[-1][-1])) > 20:
        eps.append([d])
    else:
        eps[-1].append(d)
starts = [e[0] for e in eps]

rows = []
for s in starts:
    after = dates[dates >= s]
    b50 = after[mb.loc[after] > .50]
    g1 = after[(mb.loc[after] > .50) & (gate.loc[after] == 1)]
    b50d = b50[0] if len(b50) else pd.NaT
    gd = g1[0] if len(g1) else pd.NaT
    rows.append(dict(起点=s, 宽度回50日=b50d, 天数_宽度=(dates.get_loc(b50d) - dates.get_loc(s)) if not pd.isna(b50d) else np.nan,
                     行业闸门日=gd, 天数_闸门=(dates.get_loc(gd) - dates.get_loc(s)) if not pd.isna(gd) else np.nan,
                     起点宽度=b.loc[s], 起点回撤=dd.loc[s]))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, '阶段触发时间.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
print(T.to_string(index=False))
print('\n宽度回50: 中位 %.0f 日, P25 %.0f, P75 %.0f, 最大 %.0f, 未回 %d' % (
    T.天数_宽度.median(), T.天数_宽度.quantile(.25), T.天数_宽度.quantile(.75), T.天数_宽度.max(), T.天数_宽度.isna().sum()))
print('行业闸门: 中位 %.0f 日, P25 %.0f, P75 %.0f, 最大 %.0f, 未开 %d' % (
    T.天数_闸门.median(), T.天数_闸门.quantile(.25), T.天数_闸门.quantile(.75), T.天数_闸门.max(), T.天数_闸门.isna().sum()))
