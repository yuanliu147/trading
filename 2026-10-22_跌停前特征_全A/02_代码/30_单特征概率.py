# -*- coding: utf-8 -*-
"""
单特征条件概率、事件侧构成、分段稳定性、二维交叉表。

主样本：
  非北交所、上市>60个交易日、非ST代理、精确跌停口径、剔除102只退市近似股。
  训练期 2016-10-14～2021-12-31；检验期 2022-01-01～2026-10-13。
"""
import os, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
ROOT = os.path.join(BASE, '2026-10-22_跌停前特征_全A')
OUT = os.path.join(ROOT, '03_结果')
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:7.1f}s] {msg}', flush=True)

# ---------------- 1. 读取 ----------------
cols = ['date', 'code_id', 'board_id', 'industry_id', 'approx', 'ld1', 'ldt1', 'ld5', 'one_word1',
        'st_proxy', 'is_new', 'age',
        'ret1', 'ret3', 'ret5', 'ret10', 'ret20', 'ret60',
        'dist_ma5', 'dist_ma10', 'dist_ma20', 'dist_ma60', 'dist_ma120',
        'ma20_slope', 'ma60_slope', 'up_struct', 'trend_age',
        'dist_high20', 'dist_high60', 'dist_high250', 'dd20', 'dd60', 'dd250',
        'newhigh20', 'newhigh60', 'newhigh250', 'nh20_count', 'rsi14', 'macd_hist',
        'vol20', 'vol60', 'atr_pct', 'vol_hist_z', 'er20',
        'max_drop20', 'drop5_count20', 'drop5_count60',
        'ld_count20', 'ld_count60', 'ld_touch_count20', 'ld_touch_count60', 'ld_open_count20',
        'lu_count5', 'lu_count20', 'lu_open_count20', 'lu_touch_count20',
        'ld_yesterday', 'lu_yesterday', 'one_word_ld_yesterday', 'opened_ld_yesterday',
        'one_word_lu_yesterday', 'opened_lu_yesterday',
        'consec_ld', 'consec_lu', 'days_since_ld', 'days_since_lu',
        'amt_ratio20', 'vr5_20', 'amt_trend', 'log_amt20', 'log_price',
        'vp_corr20', 'gap_count20', 'down_vol_ratio20',
        'close_pos', 'upper', 'lower', 'body', 'gap', 'beta60',
        'market_breadth20', 'market_breadth60', 'market_ret20', 'market_ret60',
        'market_slope60', 'market_sent', 'market_amt_ratio',
        'mkt_ret1', 'mkt_ret5', 'mkt_ret20', 'mkt_ld_share', 'mkt_ld_touch_share',
        'mkt_lu_share', 'mkt_ld_rank250', 'mkt_above_ma20', 'mkt_below_ma60',
        'sector_breadth20', 'sector_breadth60', 'sector_ret20', 'sector_ret60',
        'sector_rs', 'sw_ret20', 'sw_ret60', 'sw_above60', 'sw_slope60',
        'sector_ld_share', 'sector_ld_touch_share', 'sector_lu_share',
        'sector_ret1', 'sector_ret5']
log('读取分析面板')
df = pd.read_parquet(os.path.join(ROOT, '01_数据', '分析面板.parquet'), columns=cols)
log(f'读取 {len(df):,} 行')

main = (df['st_proxy'].fillna(0) == 0) & (df['approx'] == 0) & (df['is_new'] == 0)
df = df.loc[main].copy()
N = len(df)
K = int(df['ld1'].sum())
base = K / N
train_mask = (df['date'] < '2022-01-01').values
N_tr, K_tr = int(train_mask.sum()), int(df.loc[train_mask, 'ld1'].sum())
N_te, K_te = int((~train_mask).sum()), int(df.loc[~train_mask, 'ld1'].sum())
base_tr, base_te = K_tr / N_tr, K_te / N_te
log(f'主样本 {N:,}，跌停 {K:,}，基准 {base:.4%}；训练 {base_tr:.4%}；检验 {base_te:.4%}')
df['period'] = np.where(train_mask, 'train', 'test')

# ---------------- 2. 分箱定义 ----------------
INF = np.inf
cut_spec = {
    'ret1': ([-1, -0.095, -0.05, -0.02, 0, 0.02, 0.05, 0.095, 3],
             ['≤-9.5%', '-9.5~-5%', '-5~-2%', '-2~0%', '0~2%', '2~5%', '5~9.5%', '≥9.5%']),
    'ret5': ([-1, -0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 20],
             ['≤-20%', '-20~-10%', '-10~-5%', '-5~0%', '0~5%', '5~10%', '10~20%', '≥20%']),
    'ret20': ([-1, -0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30, 0.50, 20],
              ['≤-30%', '-30~-20%', '-20~-10%', '-10~0%', '0~10%', '10~20%', '20~30%', '30~50%', '≥50%']),
    'ret60': ([-1, -0.40, -0.20, 0, 0.20, 0.50, 1.0, 20],
              ['≤-40%', '-40~-20%', '-20~0%', '0~20%', '20~50%', '50~100%', '≥100%']),
    'dist_ma20': ([-1, -0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 3],
                  ['≤-20%', '-20~-10%', '-10~-5%', '-5~0%', '0~5%', '5~10%', '10~20%', '≥20%']),
    'dist_ma60': ([-1, -0.30, -0.15, -0.05, 0, 0.10, 0.20, 5],
                  ['≤-30%', '-30~-15%', '-15~-5%', '-5~0%', '0~10%', '10~20%', '≥20%']),
    'dist_high60': ([-1, -0.50, -0.30, -0.15, -0.05, 0, 0.5],
                    ['≤-50%', '-50~-30%', '-30~-15%', '-15~-5%', '-5~0%', '≥0%']),
    'dist_high250': ([-1, -0.60, -0.40, -0.20, -0.10, 0, 0.5],
                     ['≤-60%', '-60~-40%', '-40~-20%', '-20~-10%', '-10~0%', '≥0%']),
    'vol20': ([0, 0.20, 0.30, 0.40, 0.50, 0.70, 5],
              ['≤20%', '20~30%', '30~40%', '40~50%', '50~70%', '≥70%']),
    'atr_pct': ([0, 0.02, 0.03, 0.04, 0.06, 1],
                ['≤2%', '2~3%', '3~4%', '4~6%', '≥6%']),
    'rsi14': ([0, 20, 30, 40, 50, 60, 70, 80, 100],
              ['≤20', '20~30', '30~40', '40~50', '50~60', '60~70', '70~80', '≥80']),
    'amt_ratio20': ([0, 0.5, 0.8, 1.0, 1.5, 2.5, 5, 30],
                    ['≤0.5', '0.5~0.8', '0.8~1.0', '1.0~1.5', '1.5~2.5', '2.5~5', '≥5']),
    'vr5_20': ([0, 0.5, 0.8, 1.0, 1.5, 2.5, 10],
               ['≤0.5', '0.5~0.8', '0.8~1.0', '1.0~1.5', '1.5~2.5', '≥2.5']),
    'close_pos': ([-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01],
                  ['0~10%', '10~30%', '30~50%', '50~70%', '70~90%', '90~100%']),
    'upper': ([-1, 0, 0.01, 0.02, 0.04, 10],
              ['≤0', '0~1%', '1~2%', '2~4%', '≥4%']),
    'gap': ([-1, -0.03, -0.01, 0, 0.01, 0.03, 1],
            ['≤-3%', '-3~-1%', '-1~0%', '0~1%', '1~3%', '≥3%']),
    'days_since_ld': ([0, 1, 2, 6, 21, 61, 2000],
                      ['昨日', '1日前', '2~5日前', '6~20日前', '21~60日前', '>60日/从未']),
    'days_since_lu': ([0, 1, 2, 6, 21, 61, 2000],
                      ['昨日', '1日前', '2~5日前', '6~20日前', '21~60日前', '>60日/从未']),
    'consec_lu': ([0, 1, 2, 3, 100],
                  ['0', '1', '2', '≥3']),
    'ld_count20': ([0, 1, 2, 3, 100],
                   ['0', '1', '2', '≥3']),
    'lu_count20': ([0, 1, 2, 3, 6, 100],
                   ['0', '1', '2', '3~5', '≥6']),
    'beta60': ([-5, 0.5, 0.8, 1.0, 1.2, 1.5, 6],
               ['≤0.5', '0.5~0.8', '0.8~1.0', '1.0~1.2', '1.2~1.5', '≥1.5']),
    'log_amt20': ([-10, 16, 17, 18, 19, 20, 30],
                  ['<16', '16~17', '17~18', '18~19', '19~20', '≥20']),
    'age': ([0, 250, 500, 1000, 2000],
            ['60~250日', '250~500日', '500~1000日', '≥1000日']),
    'market_breadth20': ([-0.01, 0.20, 0.30, 0.40, 0.50, 0.60, 1.01],
                         ['≤20%', '20~30%', '30~40%', '40~50%', '50~60%', '≥60%']),
    'mkt_ld_share': ([-0.001, 0.0001, 0.002, 0.005, 0.01, 0.02, 0.04, 1],
                     ['0', '0~0.2%', '0.2~0.5%', '0.5~1%', '1~2%', '2~4%', '≥4%']),
    'mkt_ld_rank250': ([-0.01, 0.50, 0.80, 0.90, 0.95, 0.99, 1.01],
                       ['≤50%', '50~80%', '80~90%', '90~95%', '95~99%', '≥99%']),
    'sector_breadth20': ([-0.01, 0.20, 0.30, 0.40, 0.50, 0.60, 1.01],
                         ['≤20%', '20~30%', '30~40%', '40~50%', '50~60%', '≥60%']),
    'sector_ld_share': ([-0.001, 0.0001, 0.005, 0.01, 0.02, 0.05, 1],
                        ['0', '0~0.5%', '0.5~1%', '1~2%', '2~5%', '≥5%']),
    'sector_ret20': ([-1, -0.15, -0.08, -0.03, 0.03, 0.08, 0.15, 5],
                     ['≤-15%', '-15~-8%', '-8~-3%', '-3~3%', '3~8%', '8~15%', '≥15%']),
    'sector_ret5': ([-1, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 5],
                    ['≤-10%', '-10~-5%', '-5~-2%', '-2~2%', '2~5%', '5~10%', '≥10%']),
    'mkt_ret1': ([-1, -0.03, -0.01, 0, 0.01, 0.03, 1],
                 ['≤-3%', '-3~-1%', '-1~0%', '0~1%', '1~3%', '≥3%']),
    'market_sent': ([-10, -1, 0, 1, 2, 10],
                    ['≤-1', '-1~0', '0~1', '1~2', '≥2']),
    'sector_rs': ([-1, -0.10, -0.03, 0.03, 0.10, 1],
                  ['≤-10%', '-10~-3%', '-3~3%', '3~10%', '≥10%']),
}
bool_cols = ['ld_yesterday', 'lu_yesterday', 'one_word_ld_yesterday', 'opened_ld_yesterday',
             'one_word_lu_yesterday', 'opened_lu_yesterday',
             'newhigh20', 'newhigh60', 'newhigh250', 'up_struct',
             'mkt_above_ma20', 'mkt_below_ma60', 'sw_above60']
cat_cols = {'board_id': {1: '主板', 2: '创业板', 3: '科创板'},
            'regime_id': {1: '强下行', 2: '下行', 3: '震荡', 4: '上行', 5: '强上行'}}

# ---------------- 3. 工具 ----------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, mid - half), min(1, mid + half))

def bh(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pp = p[ok]
    order = np.argsort(pp)
    ranked = pp[order]
    m = len(pp)
    qq = ranked * m / (np.arange(m) + 1)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    qq = np.clip(qq, 0, 1)
    out = np.empty(m)
    out[order] = qq
    q[ok] = out
    return q

def two_prop_p(k1, n1, k0, n0):
    if n1 == 0 or n0 == 0:
        return np.nan
    p1, p0 = k1 / n1, k0 / n0
    p = (k1 + k0) / (n1 + n0)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    if se == 0:
        return np.nan
    z = (p1 - p0) / se
    return 2 * stats.norm.sf(abs(z))

def stability(tr_lift, te_lift):
    if not np.isfinite(tr_lift) or not np.isfinite(te_lift):
        return '样本不足'
    if (tr_lift - 1) * (te_lift - 1) <= 0:
        return '方向不一致'
    if tr_lift >= 1.2 and te_lift >= 1.2:
        return '稳定高'
    if tr_lift <= 0.8 and te_lift <= 0.8:
        return '稳定低'
    return '弱'

# ---------------- 4. 单特征 ----------------
log('单特征条件概率')
rows = []
pvals_bin = []
for f, (bins, labels) in cut_spec.items():
    if f not in df.columns:
        continue
    s = df[f]
    b = pd.cut(s, bins=bins, labels=labels, right=False, include_lowest=True)
    tmp = pd.DataFrame({'bin': b, 'ld1': df['ld1'], 'ldt1': df['ldt1'],
                        'ld5': df['ld5'], 'period': df['period'], 'date': df['date']})
    g = tmp.groupby('bin', observed=False).agg(
        n=('ld1', 'size'), ld=('ld1', 'sum'), ldt=('ldt1', 'sum'), ld5=('ld5', 'mean'))
    gt = tmp.groupby(['bin', 'period'], observed=False).agg(n=('ld1', 'size'), ld=('ld1', 'sum')).reset_index()
    gt['rate'] = gt['ld'] / gt['n']
    tr = gt[gt.period == 'train'].set_index('bin')['rate']
    te = gt[gt.period == 'test'].set_index('bin')['rate']
    g['train_rate'] = np.array(g.index.map(tr), dtype=float)
    g['test_rate'] = np.array(g.index.map(te), dtype=float)
    g['rate'] = g['ld'] / g['n']
    g['lift'] = g['rate'] / base
    g['train_lift'] = g['train_rate'] / base_tr
    g['test_lift'] = g['test_rate'] / base_te
    g['ld5_rate'] = g['ld5']
    g['ldt_rate'] = g['ldt'] / g['n']
    g['event_share'] = g['ld'] / K
    g['nonevent_share'] = (g['n'] - g['ld']) / (N - K)
    g['event_diff_pp'] = (g['event_share'] - g['nonevent_share']) * 100
    ci = np.array([wilson(k, n) for k, n in zip(g['ld'], g['n'])])
    g['ci_low'], g['ci_high'] = ci[:, 0], ci[:, 1]
    # bin vs rest
    for bin_name, r in g.iterrows():
        p = two_prop_p(r['ld'], r['n'], K - r['ld'], N - r['n'])
        pvals_bin.append(p)
        rows.append({
            '特征': f, '分组': bin_name, '样本数': int(r['n']), '跌停数': int(r['ld']),
            'P(次日跌停)': r['rate'], '基准率': base, 'lift': r['lift'],
            '95%下限': r['ci_low'], '95%上限': r['ci_high'],
            'P(5日内跌停)': r['ld5_rate'], 'P(盘中触及)': r['ldt_rate'],
            '事件侧占比': r['event_share'], '同日非事件占比': r['nonevent_share'],
            '事件侧差(pp)': r['event_diff_pp'],
            '训练期概率': r['train_rate'], '检验期概率': r['test_rate'],
            '训练期lift': r['train_lift'], '检验期lift': r['test_lift'],
            '稳定性': stability(r['train_lift'], r['test_lift']),
            'p值': p,
        })
    log(f'  {f} 完成')

univ = pd.DataFrame(rows)
univ['q值'] = bh(univ['p值'].values)
univ['样本充足'] = univ['样本数'] >= 500
univ.to_csv(os.path.join(OUT, '单特征概率表.csv'), index=False, encoding='utf-8-sig')

# 布尔
rows = []
pvals_bin = []
for f in bool_cols:
    if f not in df.columns:
        continue
    s = df[f].fillna(0).astype(int).map({0: '否', 1: '是'})
    tmp = pd.DataFrame({'bin': s, 'ld1': df['ld1'], 'ldt1': df['ldt1'],
                        'ld5': df['ld5'], 'period': df['period'], 'date': df['date']})
    g = tmp.groupby('bin', observed=False).agg(
        n=('ld1', 'size'), ld=('ld1', 'sum'), ldt=('ldt1', 'sum'), ld5=('ld5', 'mean'))
    gt = tmp.groupby(['bin', 'period'], observed=False).agg(n=('ld1', 'size'), ld=('ld1', 'sum')).reset_index()
    gt['rate'] = gt['ld'] / gt['n']
    tr = gt[gt.period == 'train'].set_index('bin')['rate']
    te = gt[gt.period == 'test'].set_index('bin')['rate']
    g['train_rate'] = np.array(g.index.map(tr), dtype=float)
    g['test_rate'] = np.array(g.index.map(te), dtype=float)
    g['rate'] = g['ld'] / g['n']
    g['lift'] = g['rate'] / base
    g['train_lift'] = g['train_rate'] / base_tr
    g['test_lift'] = g['test_rate'] / base_te
    g['ld5_rate'] = g['ld5']
    g['ldt_rate'] = g['ldt'] / g['n']
    g['event_share'] = g['ld'] / K
    g['nonevent_share'] = (g['n'] - g['ld']) / (N - K)
    g['event_diff_pp'] = (g['event_share'] - g['nonevent_share']) * 100
    ci = np.array([wilson(k, n) for k, n in zip(g['ld'], g['n'])])
    g['ci_low'], g['ci_high'] = ci[:, 0], ci[:, 1]
    for bin_name, r in g.iterrows():
        p = two_prop_p(r['ld'], r['n'], K - r['ld'], N - r['n'])
        pvals_bin.append(p)
        rows.append({
            '特征': f, '分组': bin_name, '样本数': int(r['n']), '跌停数': int(r['ld']),
            'P(次日跌停)': r['rate'], '基准率': base, 'lift': r['lift'],
            '95%下限': r['ci_low'], '95%上限': r['ci_high'],
            'P(5日内跌停)': r['ld5_rate'], 'P(盘中触及)': r['ldt_rate'],
            '事件侧占比': r['event_share'], '同日非事件占比': r['nonevent_share'],
            '事件侧差(pp)': r['event_diff_pp'],
            '训练期概率': r['train_rate'], '检验期概率': r['test_rate'],
            '训练期lift': r['train_lift'], '检验期lift': r['test_lift'],
            '稳定性': stability(r['train_lift'], r['test_lift']),
            'p值': p,
        })
univ_b = pd.DataFrame(rows)
univ_b['q值'] = bh(univ_b['p值'].values)
univ_b['样本充足'] = univ_b['样本数'] >= 500
univ_b.to_csv(os.path.join(OUT, '单特征概率表_布尔.csv'), index=False, encoding='utf-8-sig')
log('单特征表完成')

# 分类
rows = []
for f, mp in cat_cols.items():
    if f not in df.columns:
        continue
    s = df[f].map(mp)
    tmp = pd.DataFrame({'bin': s, 'ld1': df['ld1'], 'ldt1': df['ldt1'],
                        'ld5': df['ld5'], 'period': df['period']})
    g = tmp.groupby('bin', observed=False).agg(
        n=('ld1', 'size'), ld=('ld1', 'sum'), ldt=('ldt1', 'sum'), ld5=('ld5', 'mean'))
    g['rate'] = g['ld'] / g['n']
    g['lift'] = g['rate'] / base
    ci = np.array([wilson(k, n) for k, n in zip(g['ld'], g['n'])])
    g['ci_low'], g['ci_high'] = ci[:, 0], ci[:, 1]
    for bin_name, r in g.iterrows():
        rows.append({'特征': f, '分组': bin_name, '样本数': int(r['n']), '跌停数': int(r['ld']),
                     'P(次日跌停)': r['rate'], 'lift': r['lift'],
                     '95%下限': r['ci_low'], '95%上限': r['ci_high'],
                     'P(5日内跌停)': r['ld5'], 'P(盘中触及)': r['ldt'] / r['n']})
pd.DataFrame(rows).to_csv(os.path.join(OUT, '单特征概率表_板块行情.csv'), index=False, encoding='utf-8-sig')

# ---------------- 5. 特征级汇总 ----------------
log('特征级汇总与事件侧对比')
summ = []
for f, g in univ.groupby('特征', sort=False):
    gg = g[g['样本充足']]
    if len(gg) == 0:
        continue
    hi = gg.loc[gg['lift'].idxmax()]
    lo = gg.loc[gg['lift'].idxmin()]
    summ.append({
        '特征': f, '最强组': hi['分组'], '最强组概率': hi['P(次日跌停)'], '最强组lift': hi['lift'],
        '最强组事件侧占比': hi['事件侧占比'], '最强组稳定性': hi['稳定性'],
        '最弱组': lo['分组'], '最弱组概率': lo['P(次日跌停)'], '最弱组lift': lo['lift'],
        '最弱组稳定性': lo['稳定性'],
        '组间倍数': hi['lift'] / lo['lift'] if lo['lift'] > 0 else np.nan,
        '事件侧最大差(pp)': gg['事件侧差(pp)'].abs().max(),
        '最小q值': gg['q值'].min(),
        '稳定高组数': int((gg['稳定性'] == '稳定高').sum()),
        '稳定低组数': int((gg['稳定性'] == '稳定低').sum()),
    })
summ = pd.DataFrame(summ).sort_values('最强组lift', ascending=False)
summ.to_csv(os.path.join(OUT, '单特征汇总.csv'), index=False, encoding='utf-8-sig')

# 事件侧连续特征对比 + 单变量 AUC
cont_feats = list(cut_spec.keys())
ev = df['ld1'].values == 1
rng = np.random.default_rng(20261022)
neg_idx = np.where(~ev)[0]
neg_sample = rng.choice(neg_idx, size=min(1_000_000, len(neg_idx)), replace=False)
sample_idx = np.concatenate([np.where(ev)[0], neg_sample])
sample_idx.sort()
ds = df.iloc[sample_idx]
ev_s = ds['ld1'].values == 1
comp = []
for f in cont_feats:
    if f not in ds.columns:
        continue
    x = ds[f].values.astype(float)
    ok = np.isfinite(x)
    if ok.sum() < 1000 or ev_s[ok].sum() < 50:
        continue
    x, y = x[ok], ev_s[ok]
    med_e, med_n = np.median(x[y]), np.median(x[~y])
    mean_e, mean_n = np.mean(x[y]), np.mean(x[~y])
    sd_e, sd_n = np.std(x[y]), np.std(x[~y])
    smd = (mean_e - mean_n) / np.sqrt((sd_e ** 2 + sd_n ** 2) / 2 + 1e-12)
    ks = stats.ks_2samp(x[y], x[~y]).statistic
    try:
        auc = roc_auc_score(y, x)
    except Exception:
        auc = np.nan
    comp.append({'特征': f, '事件中位数': med_e, '非事件中位数': med_n,
                 '事件均值': mean_e, '非事件均值': mean_n,
                 '标准化差(SMD)': smd, 'KS统计量': ks, '单变量AUC': auc,
                 '方向': '越高越危险' if auc >= 0.5 else '越低越危险'})
comp = pd.DataFrame(comp).sort_values('单变量AUC', ascending=False)
comp.to_csv(os.path.join(OUT, '事件侧特征对比.csv'), index=False, encoding='utf-8-sig')

# ---------------- 6. 二维交叉表 ----------------
log('二维交叉表')
def cross2(f1, b1, l1, f2, b2, l2, name, minn=500):
    x1 = pd.cut(df[f1], bins=b1, labels=l1, right=False, include_lowest=True)
    x2 = pd.cut(df[f2], bins=b2, labels=l2, right=False, include_lowest=True)
    tmp = pd.DataFrame({'x1': x1, 'x2': x2, 'ld': df['ld1']})
    g = tmp.groupby(['x1', 'x2'], observed=False).agg(n=('ld', 'size'), ld=('ld', 'sum')).reset_index()
    g['rate'] = g['ld'] / g['n']
    g = g[g['n'] >= minn]
    g.to_csv(os.path.join(OUT, name), index=False, encoding='utf-8-sig')
    return g

cross2('ret20', cut_spec['ret20'][0], cut_spec['ret20'][1],
       'dist_ma60', cut_spec['dist_ma60'][0], cut_spec['dist_ma60'][1],
       '二维_20日收益与距60日线.csv')
cross2('market_breadth20', cut_spec['market_breadth20'][0], cut_spec['market_breadth20'][1],
       'ret20', cut_spec['ret20'][0], cut_spec['ret20'][1],
       '二维_市场宽度与个股20日收益.csv')
cross2('mkt_ld_share', cut_spec['mkt_ld_share'][0], cut_spec['mkt_ld_share'][1],
       'vol20', cut_spec['vol20'][0], cut_spec['vol20'][1],
       '二维_市场跌停占比与个股波动.csv')
cross2('board_id', [0.5, 1.5, 2.5, 3.5, 4.5], ['主板', '创业板', '科创板', '北交所'],
       'ret20', cut_spec['ret20'][0], cut_spec['ret20'][1],
       '二维_板块与20日收益.csv')

# ---------------- 7. 分年、分行业 ----------------
log('分年/分行业')
by_year = df.groupby(df['date'].dt.year).agg(
    样本数=('ld1', 'size'), 跌停数=('ld1', 'sum'),
    盘中触及数=('ldt1', 'sum'), 平均市场宽度=('market_breadth20', 'mean'),
    平均市场跌停占比=('mkt_ld_share', 'mean')).reset_index().rename(columns={'date': '年份'})
by_year['跌停率'] = by_year['跌停数'] / by_year['样本数']
by_year['盘中触及率'] = by_year['盘中触及数'] / by_year['样本数']
by_year.to_csv(os.path.join(OUT, '分年基准率.csv'), index=False, encoding='utf-8-sig')

ind_names = pd.read_csv(os.path.join(ROOT, '01_数据', '行业日表.csv'), usecols=['industry_id', 'industry']).drop_duplicates()
ind_names.index = ind_names['industry_id'].astype(int)
ind_names = ind_names.set_index('industry_id')['industry'].to_dict()
by_ind = df.groupby('industry_id').agg(样本数=('ld1', 'size'), 跌停数=('ld1', 'sum')).reset_index()
by_ind['行业'] = by_ind['industry_id'].map(ind_names)
by_ind['跌停率'] = by_ind['跌停数'] / by_ind['样本数']
by_ind['lift'] = by_ind['跌停率'] / base
by_ind = by_ind.sort_values('跌停率', ascending=False)
by_ind[['industry_id', '行业', '样本数', '跌停数', '跌停率', 'lift']].to_csv(
    os.path.join(OUT, '分行业基准率.csv'), index=False, encoding='utf-8-sig')

# ---------------- 8. 关键条件 cluster bootstrap ----------------
log('关键条件按日期 cluster bootstrap')
def cluster_ci(mask, name, B=1000):
    sub = df.loc[mask, ['date', 'ld1']]
    g = sub.groupby('date').agg(n=('ld1', 'size'), k=('ld1', 'sum'))
    n = g['n'].values.astype(float); k = g['k'].values.astype(float)
    rng = np.random.default_rng(20261022)
    rates = np.empty(B)
    nd = len(n)
    for b in range(B):
        idx = rng.integers(0, nd, nd)
        rates[b] = k[idx].sum() / max(n[idx].sum(), 1)
    return {'条件': name, '样本数': int(mask.sum()), '跌停数': int(sub['ld1'].sum()),
            '概率': sub['ld1'].mean(), 'cluster下限': np.percentile(rates, 2.5),
            'cluster上限': np.percentile(rates, 97.5), 'lift': sub['ld1'].mean() / base}
keys = [
    ('昨收跌停', df['ld_yesterday'] == 1),
    ('昨收涨停', df['lu_yesterday'] == 1),
    ('20日涨>20%', df['ret20'] > 0.20),
    ('20日涨>20%且距60日线>-3%', (df['ret20'] > 0.20) & (df['dist_ma60'] > -0.03)),
    ('距20日线>10%', df['dist_ma20'] > 0.10),
    ('20日波动>50%', df['vol20'] > 0.50),
    ('20日曾跌停', df['ld_count20'] >= 1),
    ('市场宽度<30%', df['market_breadth20'] < 0.30),
    ('市场跌停占比1~2%', (df['mkt_ld_share'] >= 0.01) & (df['mkt_ld_share'] < 0.02)),
    ('市场跌停占比>2%', df['mkt_ld_share'] >= 0.02),
    ('行业跌停占比>2%', df['sector_ld_share'] >= 0.02),
]
ci_rows = [cluster_ci(m, n) for n, m in keys]
pd.DataFrame(ci_rows).to_csv(os.path.join(OUT, '关键条件_按日聚类区间.csv'), index=False, encoding='utf-8-sig')

log('完成')
