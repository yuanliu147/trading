# -*- coding: utf-8 -*-
"""
多变量模型：市场、个股、市场+个股；时间切分样本外。
训练 2016-10-14～2021-12-31；检验 2022-01-01～2026-10-13。
训练集负样本下采样 10:1，并用权重还原 prevalence；检验集全量。
"""
import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.inspection import permutation_importance
from scipy import sparse

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
ROOT = os.path.join(BASE, '2026-10-22_跌停前特征_全A')
OUT = os.path.join(ROOT, '03_结果')
t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:7.1f}s] {msg}', flush=True)

# ---------------- 1. 数据 ----------------
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
               'sector_rs', 'sector_ld_share', 'sector_ret1', 'sector_ret5',
               'board_id', 'industry_id']
all_feats = market_feats + [c for c in stock_feats if c not in market_feats]

train = (df['date'] <= '2021-12-31').values
test = (df['date'] >= '2022-01-01').values
log(f'训练 {train.sum():,}，检验 {test.sum():,}')

# 训练下采样
rng = np.random.default_rng(20261022)
tr_pos = np.where(train & (df['ld1'].values == 1))[0]
tr_neg = np.where(train & (df['ld1'].values == 0))[0]
neg_ratio = 10
tr_neg_s = rng.choice(tr_neg, size=min(len(tr_neg), len(tr_pos) * neg_ratio), replace=False)
tr_idx = np.concatenate([tr_pos, tr_neg_s])
w = np.concatenate([np.ones(len(tr_pos), dtype='float32'),
                    np.full(len(tr_neg_s), len(tr_neg) / len(tr_neg_s), dtype='float32')])
perm = rng.permutation(len(tr_idx))
tr_idx = tr_idx[perm]
w = w[perm]
log(f'训练下采样 {len(tr_idx):,}（正 {len(tr_pos):,}，负 {len(tr_neg_s):,}）')
te_idx = np.where(test)[0]

def get_X(idx, feats):
    return df.loc[idx, feats].astype('float32').values

def eval_pred(y, p, name):
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    order = np.argsort(-p)
    out = {'模型': name, 'AUC': auc, 'PR-AUC': ap, 'Brier': brier,
           '测试样本': len(y), '跌停数': int(y.sum()), '基准率': y.mean()}
    for frac in [0.01, 0.05, 0.10]:
        k = int(len(y) * frac)
        top = order[:k]
        cap = y[top].sum() / y.sum()
        prec = y[top].mean()
        out[f'top{int(frac*100)}%捕获'] = cap
        out[f'top{int(frac*100)}%精度'] = prec
        out[f'top{int(frac*100)}%lift'] = prec / y.mean()
    return out

# ---------------- 2. 模型 ----------------
results = []
preds = {}
for name, feats in [('仅市场', market_feats), ('仅个股', stock_feats), ('市场+个股', all_feats)]:
    log(f'训练 {name}：{len(feats)} 个特征')
    X = get_X(tr_idx, feats)
    y = df.loc[tr_idx, 'ld1'].values
    cat_idx = [i for i, c in enumerate(feats) if c in ('board_id', 'industry_id')]
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, max_bins=128, early_stopping=False, random_state=42,
        categorical_features=cat_idx if cat_idx else None)
    model.fit(X, y, sample_weight=w)
    log(f'  预测检验集')
    p = model.predict_proba(get_X(te_idx, feats))[:, 1]
    preds[name] = p
    res = eval_pred(df.loc[te_idx, 'ld1'].values, p, name)
    results.append(res)
    log(f'  {name}: AUC={res["AUC"]:.4f} AP={res["PR-AUC"]:.4f} top1%lift={res["top1%lift"]:.2f}')
    # 保存模型用于重要性（仅市场+个股）
    if name == '市场+个股':
        best_model, best_feats = model, feats

res = pd.DataFrame(results)
res.to_csv(os.path.join(OUT, '模型对比.csv'), index=False, encoding='utf-8-sig')
print(res.to_string(index=False, float_format=lambda z: f'{z:.4g}'))

# ---------------- 3. 逻辑回归（可解释对照） ----------------
log('逻辑回归对照')
feats_lr = ['ret1', 'ret5', 'ret20', 'ret60', 'dist_ma20', 'dist_ma60', 'dist_high60',
            'dist_high250', 'vol20', 'atr_pct', 'rsi14', 'amt_ratio20', 'vr5_20',
            'close_pos', 'upper', 'gap', 'ld_count20', 'lu_count20', 'consec_lu',
            'ld_yesterday', 'lu_yesterday', 'days_since_ld', 'days_since_lu', 'log_amt20',
            'beta60', 'sector_ret20', 'sector_ld_share', 'market_breadth20',
            'mkt_ld_share', 'mkt_ld_rank250', 'mkt_ret1', 'mkt_below_ma60']
Xtr = df.loc[tr_idx, feats_lr].astype('float32')
Xte = df.loc[te_idx, feats_lr].astype('float32')
med = Xtr.median()
Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
# 1/99 缩尾
lo, hi = Xtr.quantile(0.01), Xtr.quantile(0.99)
Xtr = Xtr.clip(lo, hi, axis=1); Xte = Xte.clip(lo, hi, axis=1)
sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr).astype('float32'); Xte_s = sc.transform(Xte).astype('float32')
lr = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-5, max_iter=30, tol=1e-4,
                   random_state=42, learning_rate='optimal')
lr.fit(Xtr_s, df.loc[tr_idx, 'ld1'].values, sample_weight=w)
p_lr = lr.predict_proba(Xte_s)[:, 1]
res_lr = eval_pred(df.loc[te_idx, 'ld1'].values, p_lr, '逻辑回归-市场+个股')
pd.DataFrame([res_lr]).to_csv(os.path.join(OUT, '模型对比_逻辑回归.csv'), index=False, encoding='utf-8-sig')
print(pd.DataFrame([res_lr]).to_string(index=False, float_format=lambda z: f'{z:.4g}'))

# ---------------- 4. 校准、分年、分状态 ----------------
log('校准与分年')
p = preds['市场+个股']
y = df.loc[te_idx, 'ld1'].values
cal = pd.DataFrame({'p': p, 'y': y})
cal['十分位'] = pd.qcut(cal['p'], 10, labels=False, duplicates='drop') + 1
cal_tab = cal.groupby('十分位').agg(样本数=('y', 'size'), 预测均值=('p', 'mean'),
                                    实际跌停率=('y', 'mean')).reset_index()
cal_tab['lift'] = cal_tab['实际跌停率'] / y.mean()
cal_tab.to_csv(os.path.join(OUT, '模型校准_旧模型.csv'), index=False, encoding='utf-8-sig')
# 校准斜率
eps = 1e-7
logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
cs = LogisticRegression(C=1e6, solver='lbfgs').fit(logit, y)
log(f'校准斜率 {cs.coef_[0][0]:.3f}，截距 {cs.intercept_[0]:.3f}')

# 分年 AUC
yr = df.loc[te_idx, 'date'].dt.year.values
yr_rows = []
for year in sorted(np.unique(yr)):
    m = yr == year
    if m.sum() > 1000 and len(np.unique(y[m])) == 2:
        yr_rows.append({'年份': int(year), '样本数': int(m.sum()), '跌停数': int(y[m].sum()),
                        '基准率': y[m].mean(), 'AUC': roc_auc_score(y[m], p[m]),
                        'top5%lift': (y[m][np.argsort(-p[m])[:int(m.sum() * 0.05)]].mean() / y[m].mean())})
pd.DataFrame(yr_rows).to_csv(os.path.join(OUT, '模型分年排序能力.csv'), index=False, encoding='utf-8-sig')
print(pd.DataFrame(yr_rows).to_string(index=False, float_format=lambda z: f'{z:.4g}'))

# 分市场状态
mb = df.loc[te_idx, 'market_breadth20'].values
state_rows = []
for name, m in [('宽度≤30%', mb <= 0.30), ('宽度30~50%', (mb > 0.30) & (mb <= 0.50)),
                ('宽度>50%', mb > 0.50)]:
    if m.sum() > 1000 and len(np.unique(y[m])) == 2:
        pp = p[m]; yy = y[m]
        k = int(len(yy) * 0.05)
        state_rows.append({'市场状态': name, '样本数': int(m.sum()), '跌停数': int(yy.sum()),
                           '基准率': yy.mean(), 'AUC': roc_auc_score(yy, pp),
                           'top5%精度': yy[np.argsort(-pp)[:k]].mean(),
                           'top5%lift': yy[np.argsort(-pp)[:k]].mean() / yy.mean()})
pd.DataFrame(state_rows).to_csv(os.path.join(OUT, '模型分市场状态.csv'), index=False, encoding='utf-8-sig')

# ---------------- 5. 置换重要性 ----------------
log('置换重要性（20万检验样本）')
rng2 = np.random.default_rng(7)
pos_te = np.where(y == 1)[0]
neg_te = np.where(y == 0)[0]
samp = np.concatenate([pos_te, rng2.choice(neg_te, size=min(200000, len(neg_te)), replace=False)])
samp = np.sort(samp)
Xs = get_X(te_idx[samp], best_feats)
ys = y[samp]
base_auc = roc_auc_score(ys, best_model.predict_proba(Xs)[:, 1])
imp_rows = []
for j, f in enumerate(best_feats):
    drops = []
    for rep in range(3):
        Xp = Xs.copy()
        Xp[:, j] = rng2.permutation(Xp[:, j])
        drops.append(base_auc - roc_auc_score(ys, best_model.predict_proba(Xp)[:, 1]))
    imp_rows.append({'特征': f, 'AUC下降均值': np.mean(drops), 'AUC下降标准差': np.std(drops)})
imp = pd.DataFrame(imp_rows).sort_values('AUC下降均值', ascending=False)
imp.to_csv(os.path.join(OUT, '模型置换重要性.csv'), index=False, encoding='utf-8-sig')
print(imp.head(20).to_string(index=False, float_format=lambda z: f'{z:.5f}'))

# 保存检验集预测（10%抽样，便于作图）
samp2 = np.sort(np.concatenate([pos_te, rng2.choice(neg_te, size=min(300000, len(neg_te)), replace=False)]))
pd.DataFrame({'date': df.loc[te_idx[samp2], 'date'].values,
              'code_id': df.loc[te_idx[samp2], 'code_id'].values,
              'ld1': y[samp2], 'pred': p[samp2]}).to_csv(
    os.path.join(OUT, '模型检验预测抽样_旧模型.csv'), index=False, encoding='utf-8-sig')
log('完成')
