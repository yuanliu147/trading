# -*- coding: utf-8 -*-
"""同态低点/确认日之后的个股涨幅结构：市值、价格、波动、前期涨跌、行业、情绪。

数据：全A面板（2016-10-13~2026-10-13），fwd20/fwd60/fwd120 是收盘到收盘，
仅用于特征研究；可执行性由后续事件回测检验。
"""
import pickle, pandas as pd, numpy as np, os, json

BASE = '/Users/admin/WorkSpace/trading'
OUT = os.path.join(BASE, '2026-10-22_当前阶段_情绪同态_50%策略/03_结果')
P = pickle.load(open(os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl'), 'rb'))
dates = pd.DatetimeIndex(P['meta']['dates'])
V = {k: P[k].values for k in P if k != 'meta'}
# 面板没有 fwd120，用收盘价补算
C = V['close'].astype(float)
fwd120 = np.full_like(C, np.nan)
fwd120[:-120] = C[120:] / C[:-120] - 1
V['fwd120'] = fwd120
ind_names = P['meta']['industries']

# 读取事件明细中的低点与确认日
E = pd.read_csv(os.path.join(OUT, '同态事件_明细.csv'), parse_dates=['日期'])
low_dates = sorted(set(E.loc[E.入场 == '区间最低', '日期'].dropna()))
rec_dates = sorted(set(E.loc[E.入场 == '宽度回到25%', '日期'].dropna()))

def cross(dt, entry_type):
    if pd.Timestamp(dt) not in dates:
        return None
    i = dates.get_loc(pd.Timestamp(dt))
    rows = []
    codes = P['meta']['codes']
    ind = V['industry_id'][i]
    for j, c in enumerate(codes):
        f60 = V['fwd60'][i, j]; f20 = V['fwd20'][i, j]; f120 = V['fwd120'][i, j]
        if not np.isfinite(f60):
            continue
        rows.append(dict(日期=pd.Timestamp(dt), 类型=entry_type, 代码=c,
                         行业=ind_names[int(ind[j]) - 1] if np.isfinite(ind[j]) and ind[j] > 0 else '未知',
                         f20=f20, f60=f60, f120=f120,
                         市值=V['mcap'][i, j], 收盘=V['close'][i, j], ATR=V['atr_pct'][i, j],
                         RSI=V['rsi14'][i, j], 波动20=V['vol20'][i, j],
                         R5=V['ret5'][i, j], R20=V['ret20'][i, j], R60=V['ret60'][i, j], R120=V['ret120'][i, j],
                         距MA20=V['dist_ma20'][i, j], 距MA60=V['dist_ma60'][i, j],
                         回撤60=V['dd60'][i, j], 回撤250=V['dd250'][i, j],
                         涨停20=V['lu_count20'][i, j], 成交额=V['amt20'][i, j],
                         行业宽度=V['sector_breadth20'][i, j], 行业20日=V['sector_ret20'][i, j],
                         市场宽度=V['market_breadth20'][i]))
    return pd.DataFrame(rows)

frames = []
for d in low_dates:
    x = cross(d, '低点')
    if x is not None: frames.append(x)
for d in rec_dates:
    x = cross(d, '宽度回到25%')
    if x is not None: frames.append(x)
D = pd.concat(frames, ignore_index=True)
D.to_parquet(os.path.join(OUT, '同态横截面_个股.parquet'), index=False)

# 分位分组统计
def qtab(df, feat, q=5, target='f60'):
    x = df.dropna(subset=[feat, target]).copy()
    if len(x) < 100: return pd.DataFrame()
    try:
        x['组'] = pd.qcut(x[feat], q, labels=False, duplicates='drop') + 1
    except Exception:
        return pd.DataFrame()
    t = x.groupby('组').agg(样本=(target, 'size'), 均值=(target, 'mean'), 中位=(target, 'median'),
                            胜率=(target, lambda s: (s > 0).mean()),
                            P25=(target, lambda s: s.quantile(.25)), P75=(target, lambda s: s.quantile(.75)),
                            P90=(target, lambda s: s.quantile(.9)),
                            超30=(target, lambda s: (s > .30).mean()), 超50=(target, lambda s: (s > .50).mean())).reset_index()
    t.insert(0, '特征', feat); t.insert(1, '分组', q)
    return t

feats = ['市值', '收盘', 'ATR', 'RSI', '波动20', 'R5', 'R20', 'R60', 'R120', '距MA20', '距MA60',
         '回撤60', '回撤250', '涨停20', '成交额', '行业宽度', '行业20日']
out = []
for typ, df in D.groupby('类型'):
    for feat in feats:
        t = qtab(df, feat)
        if len(t): t.insert(0, '类型', typ); out.append(t)
T = pd.concat(out, ignore_index=True)
T.to_csv(os.path.join(OUT, '同态横截面_特征分箱.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')

# 行业统计（低点后60日）
ind = (D[D.类型 == '低点'].groupby('行业')
       .agg(样本=('f60', 'size'), 均值=('f60', 'mean'), 中位=('f60', 'median'),
            胜率=('f60', lambda s: (s > 0).mean()), 超30=('f60', lambda s: (s > .30).mean()),
            超50=('f60', lambda s: (s > .50).mean()))
       .sort_values('均值', ascending=False).reset_index())
ind.to_csv(os.path.join(OUT, '同态横截面_行业.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')

# 分类：低估值/高波动/低价/超跌 等组合
def bucket_summary(df, name, mask):
    x = df[mask]
    if len(x) < 30: return None
    return dict(组合=name, 样本=len(x), f20均值=x.f20.mean(), f60均值=x.f60.mean(), f120均值=x.f120.mean(),
                f60中位=x.f60.median(), f60胜率=(x.f60 > 0).mean(),
                超30=(x.f60 > .30).mean(), 超50=(x.f60 > .50).mean(), 最差=x.f60.min(), 最好=x.f60.max())

low = D[D.类型 == '低点'].copy()
med = low[['市值', '收盘', 'ATR', 'R20', 'R60', '涨停20', '成交额']].median()
combos = []
for name, m in [
    ('全部', pd.Series(True, index=low.index)),
    ('超跌(R20<-20%)', low.R20 < -.20),
    ('深回撤(DD250<-35%)', low.回撤250 < -.35),
    ('低价(<=8)', low.收盘 <= 8),
    ('小市值(<=中位)', low.市值 <= med['市值']),
    ('高波(ATR>=5%)', low.ATR >= .05),
    ('无涨停20', low.涨停20 == 0),
    ('近期有涨停', low.涨停20 >= 1),
    ('超跌+小市值', (low.R20 < -.20) & (low.市值 <= med['市值'])),
    ('超跌+高波', (low.R20 < -.20) & (low.ATR >= .05)),
    ('超跌+低价', (low.R20 < -.20) & (low.收盘 <= 8)),
    ('强于市场(R20>0)', low.R20 > 0),
    ('行业强(宽度>=60%)', low.行业宽度 >= .60),
    ('行业弱(宽度<20%)', low.行业宽度 < .20),
]:
    s = bucket_summary(low, name, m)
    if s: combos.append(s)
C = pd.DataFrame(combos)
C.to_csv(os.path.join(OUT, '同态横截面_组合.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')

print('事件低点:', len(low_dates), '确认日:', len(rec_dates), '个股样本:', len(D))
print('\n=== 组合 ===')
print(C.round(4).to_string(index=False))
print('\n=== 行业前15 ===')
print(ind.head(15).round(4).to_string(index=False))
print('\n=== 关键分箱: 低点 R20 / 市值 / ATR / 涨停20 ===')
for feat in ['R20', '市值', 'ATR', '涨停20', '行业宽度']:
    print('\n', feat)
    print(T[(T.类型 == '低点') & (T.特征 == feat)].round(4).to_string(index=False))
