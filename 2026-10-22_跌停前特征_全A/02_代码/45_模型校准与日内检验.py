# -*- coding: utf-8 -*-
"""
模型校准与“日内”检验：
  训练 2016-10-14～2020-12-31；校准/验证 2021；检验 2022-01-01～2026-10-13。
  1) 梯度提升排序 + 等渗校准；
  2) 简单清单（6条规则）对照；
  3) 整体 AUC vs 日内 AUC（剔除市场择时）；
  4) top1% 的构成。
"""
import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
ROOT = os.path.join(BASE, '2026-10-22_跌停前特征_全A')
OUT = os.path.join(ROOT, '03_结果')
t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:7.1f}s] {msg}', flush=True)

cols = ['date', 'code_id', 'board_id', 'industry_id', 'approx', 'ld1', 'st_proxy', 'is_new',
        'ret1', 'ret5', 'ret20', 'ret60', 'dist_ma20', 'dist_ma60', 'dist_high60', 'dist_high250',
        'vol20', 'vol60', 'atr_pct', 'rsi14', 'macd_hist', 'er20',
        'amt_ratio20', 'vr5_20', 'log_amt20', 'vp_corr20', 'gap_count20',
        'close_pos', 'upper', 'lower', 'body', 'gap',
        'ld_count20', 'ld_touch_count20', 'lu_count20', 'lu_open_count20',
        'ld_yesterday', 'lu_yesterday', 'opened_lu_yesterday', 'one_word_lu_yesterday',
        'consec_ld', 'consec_lu', 'days_since_ld', 'days_since_lu',
        'beta60', 'market_breadth20', 'market_breadth60', 'market_ret20', 'market_ret60',
        'market_slope60', 'market_sent', 'market_amt_ratio',
        'mkt_ret1', 'mkt_ret5', 'mkt_ld_share', 'mkt_ld_rank250',
        'mkt_above_ma20', 'mkt_below_ma60',
        'sector_breadth20', 'sector_breadth60', 'sector_ret20', 'sector_ret60',
        'sector_rs', 'sector_ld_share', 'sector_ret1', 'sector_ret5']
log('读取数据')
df = pd.read_parquet(os.path.join(ROOT, '01_数据', '分析面板.parquet'), columns=cols)
main = (df['st_proxy'].fillna(0) == 0) & (df['approx'] == 0) & (df['is_new'] == 0)
df = df.loc[main].reset_index(drop=True)
df['date'] = pd.to_datetime(df['date'])
log(f'主样本 {len(df):,}，正例 {int(df.ld1.sum()):,}')

market_feats = ['market_breadth20', 'market_breadth60', 'market_ret20', 'market_ret60',
                'market_slope60', 'market_sent', 'market_amt_ratio',
                'mkt_ret1', 'mkt_ret5', 'mkt_ld_share', 'mkt_ld_rank250',
                'mkt_above_ma20', 'mkt_below_ma60']
stock_feats = ['ret1', 'ret5', 'ret20', 'ret60', 'dist_ma20', 'dist_ma60', 'dist_high60',
               'dist_high250', 'vol20', 'vol60', 'atr_pct', 'rsi14', 'macd_hist', 'er20',
               'amt_ratio20', 'vr5_20', 'log_amt20', 'vp_corr20', 'gap_count20',
               'close_pos', 'upper', 'lower', 'body', 'gap',
               'ld_count20', 'ld_touch_count20', 'lu_count20', 'lu_open_count20',
               'ld_yesterday', 'lu_yesterday', 'opened_lu_yesterday', 'one_word_lu_yesterday',
               'consec_ld', 'consec_lu', 'days_since_ld', 'days_since_lu', 'beta60',
               'sector_breadth20', 'sector_breadth60', 'sector_ret20', 'sector_ret60',
               'sector_rs', 'sector_ld_share', 'sector_ret1', 'sector_ret5']
all_feats = market_feats + [c for c in stock_feats if c not in market_feats] + ['board_id', 'industry_id']
no_bi_feats = [c for c in all_feats if c not in ('board_id', 'industry_id')]

fit_mask = (df['date'] <= '2020-12-31').values
val_mask = ((df['date'] >= '2021-01-01') & (df['date'] <= '2021-12-31')).values
test_mask = (df['date'] >= '2022-01-01').values
log(f'训练 {fit_mask.sum():,}，验证 {val_mask.sum():,}，检验 {test_mask.sum():,}')

rng = np.random.default_rng(20261022)
fit_pos = np.where(fit_mask & (df.ld1.values == 1))[0]
fit_neg = np.where(fit_mask & (df.ld1.values == 0))[0]
neg_s = rng.choice(fit_neg, size=min(len(fit_neg), len(fit_pos) * 10), replace=False)
fit_idx = np.concatenate([fit_pos, neg_s])
w = np.concatenate([np.ones(len(fit_pos), dtype='float32'),
                    np.full(len(neg_s), len(fit_neg) / len(neg_s), dtype='float32')])
perm = rng.permutation(len(fit_idx))
fit_idx = fit_idx[perm]; w = w[perm]
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]
log(f'训练下采样 {len(fit_idx):,}')

def X(idx, feats):
    return df.loc[idx, feats].astype('float32').values

def metrics(y, p, name):
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    order = np.argsort(-p)
    out = {'模型': name, 'AUC': auc, 'PR-AUC': ap, 'Brier': brier,
           '基准率': y.mean(), 'Brier基准': y.mean() * (1 - y.mean())}
    for frac in [0.01, 0.05, 0.10]:
        k = int(len(y) * frac); top = order[:k]
        out[f'top{int(frac*100)}%捕获'] = y[top].sum() / y.sum()
        out[f'top{int(frac*100)}%精度'] = y[top].mean()
        out[f'top{int(frac*100)}%lift'] = y[top].mean() / y.mean()
    return out

# ---------------- 1. 主模型 ----------------
log('训练主模型（市场+个股+板块行业）')
model = HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
    l2_regularization=1.0, max_bins=128, early_stopping=False, random_state=42,
    categorical_features=[all_feats.index('board_id'), all_feats.index('industry_id')])
model.fit(X(fit_idx, all_feats), df.loc[fit_idx, 'ld1'].values, sample_weight=w)
p_val_raw = model.predict_proba(X(val_idx, all_feats))[:, 1]
p_test_raw = model.predict_proba(X(test_idx, all_feats))[:, 1]
y_val = df.loc[val_idx, 'ld1'].values
y_test = df.loc[test_idx, 'ld1'].values
iso = IsotonicRegression(out_of_bounds='clip').fit(p_val_raw, y_val)
p_test = iso.predict(p_test_raw)
log(f'原始 AUC {roc_auc_score(y_test, p_test_raw):.4f}；校准后 AUC {roc_auc_score(y_test, p_test):.4f}')

res = pd.DataFrame([
    metrics(y_test, p_test_raw, '市场+个股（原始概率）'),
    metrics(y_test, p_test, '市场+个股（等渗校准）'),
])
res.to_csv(os.path.join(OUT, '模型校准后对比.csv'), index=False, encoding='utf-8-sig')
print(res.to_string(index=False, float_format=lambda z: f'{z:.4g}'))

# 校准表
cal = pd.DataFrame({'p': p_test, 'y': y_test})
cal['十分位'] = pd.qcut(cal['p'], 10, labels=False, duplicates='drop') + 1
cal_tab = cal.groupby('十分位').agg(样本数=('y', 'size'), 预测均值=('p', 'mean'),
                                    实际跌停率=('y', 'mean')).reset_index()
cal_tab['lift'] = cal_tab['实际跌停率'] / y_test.mean()
cal_tab.to_csv(os.path.join(OUT, '模型校准表_校准后.csv'), index=False, encoding='utf-8-sig')
print(cal_tab.to_string(index=False, float_format=lambda z: f'{z:.4g}'))

# ---------------- 2. 简单清单 ----------------
log('简单清单对照')
simple = pd.DataFrame({
    'x1': (df['ld_yesterday'] >= 1).astype('float32'),
    'x2': (df['consec_lu'] >= 2).astype('float32'),
    'x3': (df['vol20'] >= 0.50).astype('float32'),
    'x4': (df['mkt_ld_share'] >= 0.005).astype('float32'),
    'x5': (df['ret20'] >= 0.20).astype('float32'),
    'x6': (df['amt_ratio20'] >= 2.0).astype('float32'),
})
# 用训练期拟合一个 6 变量逻辑回归（仅用于排序，不做手工权重）
Xtr_s = simple.loc[fit_idx].values
Xva_s = simple.loc[val_idx].values
Xte_s = simple.loc[test_idx].values
lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
lr.fit(Xtr_s, df.loc[fit_idx, 'ld1'].values, sample_weight=w)
p_simple_val = lr.predict_proba(Xva_s)[:, 1]
p_simple = lr.predict_proba(Xte_s)[:, 1]
iso_s = IsotonicRegression(out_of_bounds='clip').fit(p_simple_val, y_val)
p_simple_cal = iso_s.predict(p_simple)
res_s = pd.DataFrame([metrics(y_test, p_simple_cal, '简单清单（6条，校准）')])
res_s.to_csv(os.path.join(OUT, '模型简单清单.csv'), index=False, encoding='utf-8-sig')
print(res_s.to_string(index=False, float_format=lambda z: f'{z:.4g}'))
coef = pd.DataFrame({'变量': ['昨日跌停', '≥2连板', '20日波动≥50%', '市场跌停占比≥0.5%',
                              '20日涨幅≥20%', '量比≥2'],
                     '系数': lr.coef_[0]})
coef.to_csv(os.path.join(OUT, '模型简单清单系数.csv'), index=False, encoding='utf-8-sig')

# ---------------- 3. 日内 AUC ----------------
log('日内 AUC（剔除市场择时）')
def within_date_auc(dates, y, p, min_n=200):
    aucs, weights, top_rates, top_base = [], [], [], []
    s = pd.DataFrame({'d': dates, 'y': y, 'p': p})
    for d, g in s.groupby('d', sort=False):
        if len(g) < min_n or g.y.nunique() < 2:
            continue
        aucs.append(roc_auc_score(g.y, g.p))
        weights.append(g.y.sum())
        k = max(1, int(len(g) * 0.10))
        top = g.nlargest(k, 'p')
        top_rates.append(top.y.mean())
        top_base.append(g.y.mean())
    aucs = np.array(aucs); weights = np.array(weights)
    return (np.average(aucs, weights=weights), len(aucs),
            np.average(top_rates, weights=weights), np.average(top_base, weights=weights))
dates_te = df.loc[test_idx, 'date'].values
for name, pp in [('市场+个股（校准）', p_test), ('简单清单（校准）', p_simple_cal)]:
    wa, nd, tr, tb = within_date_auc(dates_te, y_test, pp)
    log(f'{name}: 日内AUC {wa:.4f}，有效交易日 {nd}，日内top10%跌停率 {tr:.4%}，基准 {tb:.4%}，lift {tr/tb:.2f}')
    out_name = '模型日内排序能力_市场+个股.csv' if '市场' in name else '模型日内排序能力_简单清单.csv'
    pd.DataFrame([{'模型': name, '日内AUC': wa, '有效交易日': nd,
                   '日内top10%跌停率': tr, '日内基准率': tb, '日内top10%lift': tr / tb}]).to_csv(
        os.path.join(OUT, out_name), index=False, encoding='utf-8-sig')

# ---------------- 4. top1% 构成 ----------------
log('top1% 构成')
k = int(len(p_test) * 0.01)
top_idx = np.argsort(-p_test)[:k]
top = df.loc[test_idx[top_idx], ['ld_yesterday', 'lu_yesterday', 'consec_lu', 'vol20',
                                 'ret20', 'amt_ratio20', 'mkt_ld_share', 'market_breadth20',
                                 'ld1']].copy()
comp = {
    'top1%样本数': k, 'top1%跌停数': int(top.ld1.sum()), 'top1%精度': top.ld1.mean(),
    'top1%中昨日跌停占比': (top.ld_yesterday >= 1).mean(),
    'top1%中昨日涨停占比': (top.lu_yesterday >= 1).mean(),
    'top1%中≥2连板占比': (top.consec_lu >= 2).mean(),
    'top1%中高波动占比': (top.vol20 >= 0.50).mean(),
    'top1%中20日涨>20%占比': (top.ret20 >= 0.20).mean(),
    'top1%中量比≥2占比': (top.amt_ratio20 >= 2.0).mean(),
    'top1%中市场跌停占比≥0.5%占比': (top.mkt_ld_share >= 0.005).mean(),
    'top1%中弱市场占比': (top.market_breadth20 < 0.40).mean(),
}
pd.DataFrame([comp]).to_csv(os.path.join(OUT, '模型前1%构成.csv'), index=False, encoding='utf-8-sig')
print(pd.DataFrame([comp]).T.to_string())

# 保存抽样预测
rng2 = np.random.default_rng(7)
pos = np.where(y_test == 1)[0]; neg = np.where(y_test == 0)[0]
samp = np.sort(np.concatenate([pos, rng2.choice(neg, size=min(300000, len(neg)), replace=False)]))
pd.DataFrame({'date': dates_te[samp], 'code_id': df.loc[test_idx[samp], 'code_id'].values,
              'ld1': y_test[samp], 'pred_cal': p_test[samp], 'pred_raw': p_test_raw[samp]}).to_csv(
    os.path.join(OUT, '模型检验预测抽样_校准.csv'), index=False, encoding='utf-8-sig')
log('完成')
