# -*- coding: utf-8 -*-
"""低宽度后的“右侧确认”触发：宽度回到25%/30% + 指数在MA5上，买超跌高波小票的前瞻收益。

这是信号层检验，不是组合回测；收益已用 next_gap 调整为“次日开盘买”，
并剔除次日涨停/一字/停牌。用于判断反弹模块是否有边际。
"""
import pickle, pandas as pd, numpy as np, os

BASE = '/Users/admin/WorkSpace/trading'
OUT = os.path.join(BASE, '2026-10-22_当前阶段_情绪同态_50%策略/03_结果')
P = pickle.load(open(os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl'), 'rb'))
dates = pd.DatetimeIndex(P['meta']['dates'])
V = {k: P[k].values for k in P if k != 'meta'}
C = V['close']; O = V['open']; NG = V['next_gap']; NLU = V['next_limit_up'].astype(bool)
f = pd.read_csv(os.path.join(BASE, '2026-10-22_当前阶段_情绪同态_50%策略/01_数据/市场日频_2016_2026.csv'),
                parse_dates=['date']).set_index('date')
f['ma5'] = f.market_index.rolling(5).mean()

# 对齐面板日期
mb = pd.Series(V['market_breadth20'], index=dates)
idx = f['market_index'].reindex(dates).ffill()
dd = f['dd250'].reindex(dates).ffill()
ma5 = f['ma5'].reindex(dates).ffill()
watch = (dd < -.15) & (mb < .20)
watch10 = watch.rolling(10, min_periods=1).max().astype(bool)

triggers = []
for ver, thr in [('25%', .25), ('30%', .30), ('35%', .35)]:
    trg = watch10 & (mb > thr) & (mb.shift(1) <= thr) & (idx > ma5)
    triggers.append((ver, trg))
    print(ver, trg.sum(), list(dates[trg.fillna(False)])[:20])

def fwd_adj(key, h):
    r = V[key]
    return (1 + r) / (1 + NG) - 1

for ver, trg in triggers:
    rows = []
    for i in np.where(trg.fillna(False).values)[0]:
        # 当天满足的股票：超跌+高波+流动性
        r20 = V['ret20'][i]; atr = V['atr_pct'][i]; amt = V['amt20'][i]; c = C[i]
        base = np.isfinite(r20) & np.isfinite(atr) & np.isfinite(amt) & (amt >= 3e7) & (c >= 2)
        base &= np.isfinite(NG[i]) & ~NLU[i] & (NG[i] < .05)
        q20 = np.nanquantile(r20[base], .2) if base.sum() > 50 else -9
        qatr = np.nanquantile(atr[base], .6) if base.sum() > 50 else 9
        mask = base & (r20 <= q20) & (atr >= qatr)
        for j in np.where(mask)[0]:
            rows.append(dict(触发=ver, 日期=dates[i], 代码=P['meta']['codes'][j],
                             f20=fwd_adj('fwd20', 20)[i, j] if np.isfinite(V['fwd20'][i, j]) else np.nan,
                             f60=fwd_adj('fwd60', 60)[i, j] if np.isfinite(V['fwd60'][i, j]) else np.nan,
                             R20=r20[j], ATR=atr[j], 市值=V['mcap'][i, j]))
    D = pd.DataFrame(rows)
    if len(D) == 0: continue
    D['触发'] = ver
    D.to_csv(os.path.join(OUT, f'反弹触发_{ver}_个股.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
    print(f'\n=== 触发 {ver} n={len(D)} 日期数={D.日期.nunique()} ===')
    print(D.groupby('日期').agg(n=('代码', 'size'), f20=('f20', 'mean'), f60=('f60', 'mean'),
                                 win20=('f20', lambda s: (s > 0).mean()), win60=('f60', lambda s: (s > 0).mean())).round(3).to_string())
    print('合计 f20 %.3f f60 %.3f win20 %.3f win60 %.3f P(f60>30) %.3f P(f60>50) %.3f' % (
        D.f20.mean(), D.f60.mean(), (D.f20 > 0).mean(), (D.f60 > 0).mean(), (D.f60 > .3).mean(), (D.f60 > .5).mean()))
