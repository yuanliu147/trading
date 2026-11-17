# -*- coding: utf-8 -*-
"""行业面板预测力检验：
- 事件 vs 非事件（行业外溢，剔除事件股）
- 每日横截面 Rank IC
- 每日横截面回归（控制行业动量/波动/宽度/规模；含行业固定效应）
- 每日 Top3 vs Bottom3 行业组合（研究用，A 股不做空）
分设计期/验证期/样本外。
"""
import os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading/2026-10-22_龙虎榜涨停_行业影响与策略'
DATA = os.path.join(BASE, '01_数据')
OUT = os.path.join(BASE, '03_结果')
HS = [1, 2, 3, 5, 10, 20]
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

A = pd.read_parquet(os.path.join(DATA, '行业日频.parquet'))
M = pd.read_parquet(os.path.join(DATA, '市场日频.parquet'))
M['date'] = pd.to_datetime(M.date)
M = M.sort_values('date').reset_index(drop=True)
M['mkt_nav'] = (1 + M.mkt_ret1_eq.fillna(0)).cumprod()
for h in HS:
    M[f'mkt_fwd{h}'] = M.mkt_nav.shift(-h) / M.mkt_nav - 1
A['date'] = pd.to_datetime(A.date)
A = A.merge(M[['date'] + [f'mkt_fwd{h}' for h in HS]], on='date', how='left')
A['period'] = np.where(A.date.dt.year <= 2021, '设计', np.where(A.date.dt.year <= 2023, '验证', 'OOS'))
A['log_n'] = np.log(A.n.clip(lower=1))

FEATS = ['lhb_share', 'lhb_net_ratio', 'lhb_org_ratio', 'lhb_pos_share', 'lhb_org_pos_share',
         'lhb_up_count', 'lhb_lu_share', 'lu_share', 'lu_one_share', 'lu_open_share',
         'lu2_share', 'max_consec']
HS = [1, 3, 5, 10]
# 组合分：行业内标准化后相加
for c in FEATS:
    A[c] = pd.to_numeric(A[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
A['z_lhb'] = A.groupby('date')['lhb_share'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
A['z_lhborg'] = A.groupby('date')['lhb_org_ratio'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
A['z_lu'] = A.groupby('date')['lu_share'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
A['z_lu2'] = A.groupby('date')['lu2_share'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
A['z_open'] = A.groupby('date')['lu_open_share'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
A['combo_lhb'] = A[['z_lhb', 'z_lhborg']].sum(axis=1)
A['combo_lu'] = A[['z_lu', 'z_lu2']].sum(axis=1) - A['z_open']
A['combo_all'] = A[['combo_lhb', 'combo_lu']].sum(axis=1)
FEATS += ['combo_lhb', 'combo_lu', 'combo_all']
TARGETS = {f'fwd{h}': f'行业全部收益_h{h}' for h in HS}
for h in HS:
    TARGETS[f'fwd{h}_ex_lhb'] = f'行业剔除LHB_h{h}'
    TARGETS[f'fwd{h}_ex_lu'] = f'行业剔除涨停_h{h}'
    TARGETS[f'fwd{h}_ex_both'] = f'行业剔除两者_h{h}'
CONTROLS = ['ind_ret20_eq', 'ind_vol20', 'breadth20', 'log_n', 'ind_ret5_eq']


def winsor(s, q=0.01):
    s = pd.to_numeric(s, errors='coerce')
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


def daily_ic(df, feat, target):
    d = df[['date', feat, target]].dropna()
    if len(d) < 200:
        return np.nan, np.nan, 0, np.nan
    g = d.groupby('date')
    cnt = g[feat].transform('size')
    nu = g[feat].transform('nunique')
    d = d[(cnt >= 15) & (nu >= 5)]
    if len(d) < 200:
        return np.nan, np.nan, 0, np.nan
    g = d.groupby('date')
    rf = g[feat].rank()
    ry = g[target].rank()
    rf = rf - rf.groupby(d.date).transform('mean')
    ry = ry - ry.groupby(d.date).transform('mean')
    num = (rf * ry).groupby(d.date).sum()
    den = np.sqrt((rf ** 2).groupby(d.date).sum() * (ry ** 2).groupby(d.date).sum())
    ic = (num / den).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ic) < 10:
        return np.nan, np.nan, len(ic), np.nan
    t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
    return ic.mean(), t, len(ic), (ic > 0).mean()


def daily_reg(df, feat, target, controls=CONTROLS):
    """每日横截面回归；特征/控制变量 rank 标准化；行业固定效应用行业内去均值。"""
    betas = []
    for d, sub in df.groupby('date'):
        cols = [feat] + controls
        if len(sub) < 20:
            continue
        sub = sub.copy()
        y = winsor(sub[target], 0.01)
        m = sub[cols].notna().all(axis=1) & y.notna()
        if m.sum() < 20:
            continue
        sub = sub[m].copy(); y = y[m]
        # 行业内去均值（行业固定效应）
        for c in cols:
            sub[c] = sub[c] - sub.groupby('industry_id')[c].transform('mean')
        y = y - y.groupby(sub.industry_id).transform('mean')
        X = sub[cols].rank(pct=True)
        X = (X - 0.5) * 2
        XX = np.column_stack([np.ones(len(X)), X.values])
        try:
            b = np.linalg.lstsq(XX, y.values, rcond=None)[0]
            betas.append((d, b[1]))
        except Exception:
            continue
    if not betas:
        return np.nan, np.nan, 0
    s = pd.Series(dict(betas))
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan
    return s.mean(), t, len(s)


def top_bottom(df, feat, target, k=3):
    d = df[['date', feat, target]].dropna()
    if len(d) < 200:
        return {}
    cnt = d.groupby('date')[feat].transform('size')
    nu = d.groupby('date')[feat].transform('nunique')
    d = d[(cnt >= 15) & (nu >= 5)].copy()
    if len(d) < 200:
        return {}
    r = d.groupby('date')[feat].rank(pct=True)
    d['r'] = r
    top = d[d.r > 1 - k / d.groupby('date')['r'].transform('size')].groupby('date')[target].mean()
    bot = d[d.r <= k / d.groupby('date')['r'].transform('size')].groupby('date')[target].mean()
    s = pd.DataFrame({'top': top, 'bot': bot}).dropna()
    s['ls'] = s.top - s.bot
    out = {}
    for c in ['top', 'bot', 'ls']:
        t = s[c].mean() / (s[c].std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan
        out[c] = (s[c].mean(), t, len(s))
    return out


log('事件 vs 非事件')
rows = []
for ev, evname in [('lu_share', '涨停'), ('lhb_up_count', '龙虎榜')]:
    for h in HS:
        for tag, ycol in [('全部', f'fwd{h}'), ('剔除LHB', f'fwd{h}_ex_lhb'), ('剔除涨停', f'fwd{h}_ex_lu'), ('剔除两者', f'fwd{h}_ex_both')]:
            y = A[ycol] - A[f'mkt_fwd{h}']
            e = A[ev] > 0
            a, b = y[e], y[~e]
            diff = a.mean() - b.mean()
            # 日期聚类 t：先按日求事件/非事件均值差
            daily = (y[e].groupby(A.loc[e, 'date']).mean() - y[~e].groupby(A.loc[~e, 'date']).mean()).dropna()
            t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else np.nan
            rows.append(dict(事件=evname, 口径=tag, h=h, 事件数=int(e.sum()), 非事件数=int((~e).sum()),
                             事件均值=a.mean(), 非事件均值=b.mean(), 差=diff, t=t))
pd.DataFrame(rows).to_csv(os.path.join(OUT, '行业预测_事件与非事件.csv'), index=False, encoding='utf-8-sig')

log('Rank IC')
rows = []
for feat in FEATS:
    for ycol, yname in TARGETS.items():
        h = int(ycol.split('_')[0].replace('fwd', ''))
        ic, t, nd, pos = daily_ic(A, feat, ycol)
        rows.append(dict(特征=feat, 目标=yname, h=h, IC=ic, t=t, 天数=nd, 正IC占比=pos, 期='全样本'))
        for p in ['设计', '验证', 'OOS']:
            sub = A[A.period == p]
            ic, t, nd, pos = daily_ic(sub, feat, ycol)
            rows.append(dict(特征=feat, 目标=yname, h=h, IC=ic, t=t, 天数=nd, 正IC占比=pos, 期=p))
IC = pd.DataFrame(rows)
IC.to_csv(os.path.join(OUT, '行业预测_RankIC.csv'), index=False, encoding='utf-8-sig')

log('横截面回归')
rows = []
REG_FEATS = ['lhb_share', 'lhb_net_ratio', 'lhb_org_ratio', 'lu_share', 'lu_one_share', 'combo_all']
REG_TARGETS = ['fwd1_ex_lu', 'fwd3_ex_lu', 'fwd5_ex_lu', 'fwd1_ex_lhb', 'fwd3_ex_lhb']
for feat in REG_FEATS:
    for ycol in REG_TARGETS:
        yname = TARGETS[ycol]
        h = int(ycol.split('_')[0].replace('fwd', ''))
        b, t, nd = daily_reg(A, feat, ycol)
        rows.append(dict(特征=feat, 目标=yname, h=h, beta=b, t=t, 天数=nd, 期='全样本'))
        for p in ['设计', '验证', 'OOS']:
            b, t, nd = daily_reg(A[A.period == p], feat, ycol)
            rows.append(dict(特征=feat, 目标=yname, h=h, beta=b, t=t, 天数=nd, 期=p))
pd.DataFrame(rows).to_csv(os.path.join(OUT, '行业预测_横截面回归.csv'), index=False, encoding='utf-8-sig')

log('Top3 vs Bottom3')
rows = []
for feat in FEATS:
    for ycol, yname in TARGETS.items():
        h = int(ycol.split('_')[0].replace('fwd', ''))
        for p, sub in [('全样本', A)] + [(x, A[A.period == x]) for x in ['设计', '验证', 'OOS']]:
            o = top_bottom(sub, feat, ycol)
            if not o:
                continue
            rows.append(dict(特征=feat, 目标=yname, h=h, 期=p,
                             多头=o['top'][0], 多头t=o['top'][1], 空头=o['bot'][0], 多空=o['ls'][0], 多空t=o['ls'][1], 天数=o['top'][2]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, '行业预测_TopBottom.csv'), index=False, encoding='utf-8-sig')

# 关键摘要
log('关键摘要')
ic = pd.read_csv(os.path.join(OUT, '行业预测_RankIC.csv'))
key = ic[(ic.期 == '全样本') & (ic.目标.isin(['行业剔除LHB_h3', '行业剔除涨停_h3', '行业全部收益_h3', '行业剔除LHB_h5', '行业剔除涨停_h5']))]
print(key.sort_values(['目标', 'IC'], ascending=[True, False]).head(30).to_string(index=False))
print('完成')
