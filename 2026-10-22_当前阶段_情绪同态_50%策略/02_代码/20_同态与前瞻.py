# -*- coding: utf-8 -*-
"""历史同态阶段与前瞻收益。

输出：
- 03_结果/同态条件_前瞻.csv：不同条件、不同入场时点的未来收益
- 03_结果/同态事件_明细.csv：每次同态事件的日期、后续路径
- 03_结果/当前同态_最近邻.csv：按标准化特征距离找最近的历史日期
"""
import pandas as pd, numpy as np, os

BASE = '/Users/admin/WorkSpace/trading/2026-10-22_当前阶段_情绪同态_50%策略'
IN = os.path.join(BASE, '01_数据')
OUT = os.path.join(BASE, '03_结果')
f = pd.read_csv(os.path.join(IN, '市场日频_2016_2026.csv'), parse_dates=['date']).set_index('date').sort_index()
f = f[~f.index.duplicated()]
f = f.loc['2017-01-01':]

# 未来路径
for h in [1, 3, 5, 10, 20, 60, 120]:
    f[f'f{h}'] = f['market_index'].shift(-h) / f['market_index'] - 1
for h in [20, 60, 120]:
    f[f'min{h}'] = f['market_index'].iloc[::-1].rolling(h, min_periods=1).min().iloc[::-1] / f['market_index'] - 1
    f[f'max{h}'] = f['market_index'].iloc[::-1].rolling(h, min_periods=1).max().iloc[::-1] / f['market_index'] - 1

# 特征
f['idx_pct'] = f['market_index'].rank(pct=True)
f['ret60'] = f['market_index'] / f['market_index'].shift(60) - 1
f['ret120'] = f['market_index'] / f['market_index'].shift(120) - 1
f['dd250'] = f['market_index'] / f['market_index'].rolling(250, min_periods=120).max() - 1
f['breadth_chg5'] = f['breadth20'] - f['breadth20'].shift(5)

conds = {
    'A_回撤15_宽度20': (f.dd250 < -0.15) & (f.breadth20 < 0.20),
    'B_回撤15_宽度15_情绪负': (f.dd250 < -0.15) & (f.breadth20 < 0.15) & (f.sent < 0),
    'C_宽度15_20日跌5': (f.breadth20 < 0.15) & (f.ret20 < -0.05),
    'D_破60和120_宽度20_回撤15': (f.market_index < f.ma60) & (f.market_index < f.ma120) & (f.breadth20 < 0.20) & (f.dd250 < -0.15),
    'E_高位_回撤15_宽度20': (f.idx_pct > 0.70) & (f.dd250 < -0.15) & (f.breadth20 < 0.20),
    'F_见顶后_宽度20_60日跌': (f.dd250 < -0.10) & (f.breadth20 < 0.20) & (f.ret60 < -0.05),
}

def episodes(mask, gap=20):
    dates = f.index[mask.fillna(False)]
    out = []
    for d in dates:
        if not out or (f.index.get_loc(d) - f.index.get_loc(out[-1][-1])) > gap:
            out.append([d])
        else:
            out[-1].append(d)
    return out

rows = []
details = []
for name, mask in conds.items():
    eps = episodes(mask)
    for k, ep in enumerate(eps):
        start = ep[0]
        # 区间最低（按指数）
        seg = f.loc[ep[0]:ep[-1]]
        trough = seg['market_index'].idxmin()
        # 确认日：事件开始后首次宽度>25%
        after = f.loc[start:]
        rec = after[after.breadth20 > 0.25]
        rec_d = rec.index[0] if len(rec) else pd.NaT
        # 事件结束日：宽度>25%后的第5个交易日（仅分析用）
        for entry_name, entry in [('首日', start), ('区间最低', trough), ('宽度回到25%', rec_d)]:
            if pd.isna(entry):
                continue
            r = f.loc[entry]
            rows.append(dict(条件=name, 事件=k+1, 入场=entry_name, 日期=entry,
                             指数=r.market_index, 宽度20=r.breadth20, 回撤250=r.dd250,
                             f5=r.f5, f10=r.f10, f20=r.f20, f60=r.f60, f120=r.f120,
                             min20=r.min20, max20=r.max20, min60=r.min60, max60=r.max60,
                             min120=r.min120, max120=r.max120))
        details.append(dict(条件=name, 事件=k+1, 开始=start, 结束=ep[-1], 天数=len(ep),
                            区间最低日=trough, 最低指数=f.loc[trough, 'market_index'],
                            首日指数=f.loc[start, 'market_index'],
                            首日宽度=f.loc[start, 'breadth20'], 最低宽度=seg.breadth20.min(),
                            确认日=rec_d))

D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, '同态事件_明细.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')

# 汇总
summ = []
for (c, e), g in D.groupby(['条件', '入场']):
    summ.append(dict(条件=c, 入场=e, 样本=len(g),
                     f5均值=g.f5.mean(), f20均值=g.f20.mean(), f60均值=g.f60.mean(), f120均值=g.f120.mean(),
                     f20中位=g.f20.median(), f60中位=g.f60.median(),
                     f20胜率=(g.f20 > 0).mean(), f60胜率=(g.f60 > 0).mean(), f120胜率=(g.f120 > 0).mean(),
                     min20均值=g.min20.mean(), min60均值=g.min60.mean(), min120均值=g.min120.mean(),
                     max20均值=g.max20.mean(), max60均值=g.max60.mean(), max120均值=g.max120.mean(),
                     max60最好=g.max60.max(), min60最差=g.min60.min()))
S = pd.DataFrame(summ)
S.to_csv(os.path.join(OUT, '同态条件_前瞻.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')

# 最近邻：标准化特征距离
feat = ['ret5', 'ret20', 'ret60', 'ret120', 'dd250', 'breadth20', 'breadth60', 'limitup', 'limitdown',
        'newhigh', 'amt_ratio', 'dispersion', 'vol20', 'sent']
X = f[feat].copy()
mu, sd = X.mean(), X.std()
Z = (X - mu) / sd
cur = Z.iloc[-1]
dist = np.sqrt(((Z - cur) ** 2).sum(axis=1))
nn = pd.DataFrame({'距离': dist, '日期': f.index}).sort_values('距离').head(40)
nn['指数'] = f.loc[nn['日期'], 'market_index'].values
nn['宽度20'] = f.loc[nn['日期'], 'breadth20'].values
nn['回撤250'] = f.loc[nn['日期'], 'dd250'].values
nn['未来20日'] = f.loc[nn['日期'], 'f20'].values
nn['未来60日'] = f.loc[nn['日期'], 'f60'].values
nn['未来120日'] = f.loc[nn['日期'], 'f120'].values
nn.to_csv(os.path.join(OUT, '当前同态_最近邻.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')

print('=== 条件汇总 ===')
print(S.round(4).to_string(index=False))
print('\n=== 同态事件 ===')
print(pd.DataFrame(details).round(4).to_string(index=False))
print('\n=== 最近邻 ===')
print(nn.head(25).round(4).to_string(index=False))
