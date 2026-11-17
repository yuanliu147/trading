# -*- coding: utf-8 -*-
"""
跌停前形态组合、事件侧原型、事件窗口路径、连续跌停风险。
"""
import os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
ROOT = os.path.join(BASE, '2026-10-22_跌停前特征_全A')
OUT = os.path.join(ROOT, '03_结果')
t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:7.1f}s] {msg}', flush=True)

# 读取扁平表
cols = ['date', 'code_id', 'board_id', 'industry_id', 'approx', 'ld1', 'ldt1', 'ld5',
        'st_proxy', 'is_new', 'age', 'ret1', 'ret5', 'ret20', 'ret60',
        'vol20', 'atr_pct', 'dist_ma20', 'dist_ma60', 'dist_high60', 'dist_high250',
        'rsi14', 'amt_ratio20', 'vr5_20', 'close_pos', 'upper', 'gap',
        'ld_count20', 'lu_count20', 'consec_ld', 'consec_lu',
        'ld_yesterday', 'lu_yesterday', 'opened_lu_yesterday', 'one_word_lu_yesterday',
        'market_breadth20', 'mkt_below_ma60', 'mkt_ld_share', 'mkt_ld_rank250',
        'sector_ld_share', 'sector_ret20', 'sector_rs']
log('读取扁平表')
df = pd.read_parquet(os.path.join(ROOT, '01_数据', '分析面板.parquet'), columns=cols)
main = (df['st_proxy'].fillna(0) == 0) & (df['approx'] == 0) & (df['is_new'] == 0)
df = df.loc[main].copy()
N = len(df); K = int(df['ld1'].sum()); base = K / N
train = (df['date'] < '2022-01-01').values
base_tr = df.loc[train, 'ld1'].mean(); base_te = df.loc[~train, 'ld1'].mean()
log(f'主样本 {N:,}，跌停 {K:,}，基准 {base:.4%}')

# ---------------- 1. 形态组合 ----------------
log('形态组合概率')
def cluster_ci(mask, B=1000):
    sub = df.loc[mask, ['date', 'ld1']]
    g = sub.groupby('date').agg(n=('ld1', 'size'), k=('ld1', 'sum'))
    n = g['n'].values.astype(float); k = g['k'].values.astype(float)
    rng = np.random.default_rng(20261022)
    nd = len(n); rates = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, nd, nd)
        rates[b] = k[idx].sum() / max(n[idx].sum(), 1)
    return np.percentile(rates, 2.5), np.percentile(rates, 97.5)

patterns = [
    ('弱市场：宽度<40%且指数<MA60', (df['market_breadth20'] < 0.40) & (df['mkt_below_ma60'] >= 0.5), '预注册'),
    ('市场恐慌：跌停占比近250日≥90分位', df['mkt_ld_rank250'] >= 0.90, '预注册'),
    ('高位加速：20日涨>20%且距60日线>-3%', (df['ret20'] > 0.20) & (df['dist_ma60'] > -0.03), '预注册'),
    ('远离均线：距20日线>10%', df['dist_ma20'] > 0.10, '预注册'),
    ('昨日涨停', df['lu_yesterday'] >= 1, '预注册'),
    ('连板：昨日≥2连板', df['consec_lu'] >= 2, '预注册'),
    ('昨日炸板：触涨停未封', df['opened_lu_yesterday'] >= 1, '预注册'),
    ('放量滞涨：量比>1.5且|涨跌|<2%且上影>2%', (df['amt_ratio20'] > 1.5) & (df['ret1'].abs() < 0.02) & (df['upper'] > 0.02), '预注册'),
    ('破位：跌破20日线且5日跌>5%', (df['dist_ma20'] < 0) & (df['ret5'] < -0.05), '预注册'),
    ('高波动：20日波动>50%', df['vol20'] >= 0.50, '预注册'),
    ('前期跌停：20日内有收盘跌停', df['ld_count20'] >= 1, '预注册'),
    ('高位加速+弱市场', (df['ret20'] > 0.20) & (df['dist_ma60'] > -0.03) & (df['market_breadth20'] < 0.40) & (df['mkt_below_ma60'] >= 0.5), '预注册'),
    ('连板+弱市场', (df['consec_lu'] >= 2) & (df['market_breadth20'] < 0.40) & (df['mkt_below_ma60'] >= 0.5), '探索'),
    ('高波动+弱市场', (df['vol20'] >= 0.50) & (df['market_breadth20'] < 0.40) & (df['mkt_below_ma60'] >= 0.5), '探索'),
    ('前期跌停+弱市场', (df['ld_count20'] >= 1) & (df['market_breadth20'] < 0.40) & (df['mkt_below_ma60'] >= 0.5), '探索'),
    ('连板+高波动', (df['consec_lu'] >= 2) & (df['vol20'] >= 0.50), '探索'),
    ('市场恐慌+高波动+前期跌停', (df['mkt_ld_rank250'] >= 0.90) & (df['vol20'] >= 0.50) & (df['ld_count20'] >= 1), '探索'),
]
prows = []
for name, mask, typ in patterns:
    mask = mask.fillna(False)
    n = int(mask.sum()); k = int((df.loc[mask, 'ld1']).sum())
    if n == 0:
        continue
    rate = k / n
    tr = mask & train
    te = mask & (~train)
    ntr, ktr = int(tr.sum()), int(df.loc[tr, 'ld1'].sum())
    nte, kte = int(te.sum()), int(df.loc[te, 'ld1'].sum())
    lo, hi = cluster_ci(mask)
    prows.append({
        '形态': name, '类型': typ, '样本数': n, '跌停数': k, 'P(次日跌停)': rate,
        'lift': rate / base, 'cluster下限': lo, 'cluster上限': hi,
        'P(5日内跌停)': df.loc[mask, 'ld5'].mean(),
        'P(盘中触及)': df.loc[mask, 'ldt1'].mean(),
        '事件覆盖率': k / K, '训练期概率': ktr / ntr if ntr else np.nan,
        '检验期概率': kte / nte if nte else np.nan,
        '训练期lift': (ktr / ntr) / base_tr if ntr else np.nan,
        '检验期lift': (kte / nte) / base_te if nte else np.nan,
    })
pat = pd.DataFrame(prows).sort_values('lift', ascending=False)
pat['判定'] = np.where((pat['训练期lift'] >= 1.3) & (pat['检验期lift'] >= 1.3), '两段稳定',
                       np.where((pat['训练期lift'] - 1) * (pat['检验期lift'] - 1) > 0, '方向一致', '方向不一致'))
pat.to_csv(os.path.join(OUT, '形态组合概率.csv'), index=False, encoding='utf-8-sig')
print(pat[['形态', '样本数', '跌停数', 'P(次日跌停)', 'lift', 'P(5日内跌停)', '事件覆盖率',
           '训练期lift', '检验期lift', '判定']].to_string(index=False, float_format=lambda z: f'{z:.4g}'))

# ---------------- 2. 事件清单与原型 ----------------
log('事件侧原型')
code_map = pd.read_csv(os.path.join(ROOT, '01_数据', '代码表.csv'))
code_map = code_map.set_index('code')['code_id'].to_dict()
ev = pd.read_csv(os.path.join(ROOT, '01_数据', '跌停事件清单.csv'), parse_dates=['date'])
df['date'] = pd.to_datetime(df['date'])
ev['code_id'] = ev['thscode'].map(code_map)
ev = ev[ev['code_id'].notna()].copy()
ev['code_id'] = ev['code_id'].astype(int)
# 合并 T 行（特征为 T-1）
feat_cols = ['date', 'code_id', 'board_id', 'industry_id', 'st_proxy', 'is_new',
             'ret1', 'ret5', 'ret20', 'ret60', 'vol20', 'atr_pct', 'dist_ma20', 'dist_ma60',
             'dist_high60', 'dist_high250', 'rsi14', 'amt_ratio20', 'vr5_20', 'close_pos',
             'upper', 'gap', 'ld_count20', 'lu_count20', 'consec_lu', 'ld_yesterday', 'lu_yesterday',
             'market_breadth20', 'mkt_below_ma60', 'mkt_ld_share', 'mkt_ld_rank250',
             'sector_ld_share', 'sector_ret20', 'sector_rs', 'ld5']
ev = ev.merge(df[feat_cols], on=['date', 'code_id'], how='inner')
ev = ev[(ev['st_proxy'].fillna(0) == 0) & (ev['is_new'] == 0)]
ev['首次跌停'] = ev['consec_ld'] <= 1
log(f'可分析跌停事件 {len(ev):,}，其中首次 {int(ev["首次跌停"].sum()):,}')

# 原型
def arch(r):
    if r['ret20'] > 0.10 and r['dist_ma60'] > 0:
        return '高位强势型'
    if r['ret20'] < -0.10 and r['dist_ma60'] < 0:
        return '下跌延续型'
    if r['ret20'] < -0.10 and r['dist_ma60'] >= 0:
        return '超跌反弹型'
    if r['ret20'] > 0.10 and r['dist_ma60'] <= 0:
        return '高位回落型'
    return '中性'
def arch_vec(d):
    r = d['ret20'].values; m = d['dist_ma60'].values
    out = np.full(len(d), '中性', dtype=object)
    out[(r > 0.10) & (m > 0)] = '高位强势型'
    out[(r < -0.10) & (m < 0)] = '下跌延续型'
    out[(r < -0.10) & (m >= 0)] = '超跌反弹型'
    out[(r > 0.10) & (m <= 0)] = '高位回落型'
    return out
ev['原型'] = arch_vec(ev)
# 事件侧原型占比 & 原型条件概率（用全样本）
df['原型'] = arch_vec(df)
arch_tab = df.groupby('原型').agg(样本数=('ld1', 'size'), 跌停数=('ld1', 'sum')).reset_index()
arch_tab['P(次日跌停)'] = arch_tab['跌停数'] / arch_tab['样本数']
arch_tab['lift'] = arch_tab['P(次日跌停)'] / base
arch_ev = ev['原型'].value_counts(normalize=True).rename('事件侧占比')
arch_tab = arch_tab.merge(arch_ev, left_on='原型', right_index=True, how='left')
arch_tab.to_csv(os.path.join(OUT, '跌停原型.csv'), index=False, encoding='utf-8-sig')
print('\n原型：')
print(arch_tab.to_string(index=False, float_format=lambda z: f'{z:.4g}'))

# 事件侧分型统计
def ev_profile(sub, name):
    out = {'分组': name, '事件数': len(sub)}
    for c in ['ret20', 'ret60', 'vol20', 'atr_pct', 'dist_ma20', 'dist_ma60', 'rsi14',
              'amt_ratio20', 'close_pos', 'upper', 'gap', 'ld_count20', 'lu_count20',
              'market_breadth20', 'mkt_ld_share']:
        out[c + '_中位数'] = sub[c].median()
    out['ret20>20%占比'] = (sub['ret20'] > 0.20).mean()
    out['ret20<-10%占比'] = (sub['ret20'] < -0.10).mean()
    out['vol20>50%占比'] = (sub['vol20'] > 0.50).mean()
    out['amt_ratio20>1.5占比'] = (sub['amt_ratio20'] > 1.5).mean()
    out['dist_ma20>10%占比'] = (sub['dist_ma20'] > 0.10).mean()
    out['dist_ma20<-10%占比'] = (sub['dist_ma20'] < -0.10).mean()
    out['昨涨停占比'] = (sub['lu_yesterday'] >= 1).mean()
    out['20日有跌停占比'] = (sub['ld_count20'] >= 1).mean()
    out['弱市场占比'] = ((sub['market_breadth20'] < 0.40) & (sub['mkt_below_ma60'] >= 0.5)).mean()
    return out
profile = pd.DataFrame([
    ev_profile(ev, '全部跌停'),
    ev_profile(ev[ev['首次跌停']], '首次跌停'),
    ev_profile(ev[~ev['首次跌停']], '连续跌停'),
    ev_profile(ev[ev['原型'] == '高位强势型'], '高位强势型'),
    ev_profile(ev[ev['原型'] == '下跌延续型'], '下跌延续型'),
])
profile.to_csv(os.path.join(OUT, '事件侧分组画像.csv'), index=False, encoding='utf-8-sig')
print('\n事件侧画像：')
print(profile[['分组', '事件数', 'ret20_中位数', 'vol20_中位数', 'ret20>20%占比', 'ret20<-10%占比',
               'vol20>50%占比', '昨涨停占比', '20日有跌停占比', '弱市场占比']].to_string(index=False, float_format=lambda z: f'{z:.4g}'))

# ---------------- 3. 事件窗口路径 ----------------
log('事件窗口路径')
dates_list = np.array(sorted(df['date'].unique()))
date_pos = {d: i for i, d in enumerate(dates_list)}
ev['di'] = ev['date'].map(date_pos)
ev = ev[ev['di'].notna()].copy()
ev['di'] = ev['di'].astype(int)
path_feats = ['ret1', 'ret5', 'ret20', 'ret60', 'vol20', 'atr_pct', 'dist_ma20',
              'dist_ma60', 'rsi14', 'amt_ratio20', 'close_pos', 'upper',
              'market_breadth20', 'mkt_ld_share']
# 以 (date, code_id) 建索引
look = df.set_index(['date', 'code_id'])[path_feats]
rows = []
for off in range(-10, 7):
    pos = ev['di'].values + off
    ok = (pos >= 0) & (pos < len(dates_list))
    sub = ev.loc[ok, ['code_id', '首次跌停', '原型']].copy()
    sub['date'] = dates_list[pos[ok]]
    idx = pd.MultiIndex.from_frame(sub[['date', 'code_id']])
    vals = look.reindex(idx).reset_index(drop=True)
    sub = sub.reset_index(drop=True)
    groups = {
        '全部': np.ones(len(sub), dtype=bool),
        '首次': sub['首次跌停'].values,
        '连续': (~sub['首次跌停']).values,
        '高位强势型': (sub['原型'] == '高位强势型').values,
        '下跌延续型': (sub['原型'] == '下跌延续型').values,
    }
    for gname, gmask in groups.items():
        for f in path_feats:
            x = vals[f].values[gmask]
            x = x[np.isfinite(x.astype(float))]
            rows.append({'offset': off, 'group': gname, 'feature': f,
                         'median': np.median(x) if len(x) else np.nan,
                         'mean': np.mean(x) if len(x) else np.nan,
                         'count': len(x)})
path = pd.DataFrame(rows)
path.to_csv(os.path.join(OUT, '事件窗口_特征路径.csv'), index=False, encoding='utf-8-sig')

# ---------------- 4. 连续跌停风险（事件后） ----------------
log('连续跌停风险')
# 事件日 E 的下一行 E+1：ld1 是 E+1 日跌停；ld5 覆盖 E+1..E+5
pos = ev['di'].values + 1
ok = (pos >= 0) & (pos < len(dates_list))
sub = ev.loc[ok, ['code_id', 'date', 'one_word', 'consec_ld', 'board', 'vol20', 'amt_ratio20',
                  'market_breadth20', 'mkt_ld_share', 'lu_count20', 'ld_count20']].copy()
sub['target_date'] = dates_list[pos[ok]]
idx = pd.MultiIndex.from_frame(sub[['target_date', 'code_id']])
look2 = df.set_index(['date', 'code_id'])[['ld1', 'ldt1', 'ld5']]
nxt = look2.reindex(idx).reset_index(drop=True)
sub = sub.reset_index(drop=True)
for c in ['ld1', 'ldt1', 'ld5']:
    sub['次日_' + c] = nxt[c].values
sub['次日跌停'] = sub['次日_ld1']
sub['次日触及'] = sub['次日_ldt1']
sub['未来5日跌停'] = sub['次日_ld5']
rows = []
for name, mask in [
    ('全部跌停', pd.Series(True, index=sub.index)),
    ('一字跌停', sub['one_word'] == True),
    ('打开跌停', sub['one_word'] == False),
    ('首次跌停', sub['consec_ld'] <= 1),
    ('连续跌停', sub['consec_ld'] >= 2),
    ('主板', sub['board'] == '主板'),
    ('创业板', sub['board'] == '创业板'),
    ('科创板', sub['board'] == '科创板'),
    ('弱市场', sub['market_breadth20'] < 0.40),
    ('市场跌停占比1~2%', (sub['mkt_ld_share'] >= 0.01) & (sub['mkt_ld_share'] < 0.02)),
    ('市场跌停占比≥2%', sub['mkt_ld_share'] >= 0.02),
    ('事件日高波动', sub['vol20'] > 0.50),
    ('事件日放量', sub['amt_ratio20'] > 1.5),
]:
    x = sub.loc[mask]
    if len(x) == 0:
        continue
    rows.append({'分组': name, '事件数': len(x), '次日跌停率': x['次日跌停'].mean(),
                 '次日触及率': x['次日触及'].mean(), '未来5日跌停率': x['未来5日跌停'].mean()})
post = pd.DataFrame(rows).sort_values('次日跌停率', ascending=False)
post.to_csv(os.path.join(OUT, '跌停后连续风险_旧口径.csv'), index=False, encoding='utf-8-sig')
print('\n连续风险：')
print(post.to_string(index=False, float_format=lambda z: f'{z:.4g}'))

log('完成')
