# -*- coding: utf-8 -*-
"""信号日回撤深度与收益：主分析。

输入（均来自本目录上游，已校验）：
  06_不同信号日起点/03_结果/逐日首笔_逐只.csv      前二逐只（459 行，450 完整）
  08_重复信号与买入时点/03_结果/全部信号重复序列_完整.csv  全部信号（4019 条完整）
  08_重复信号与买入时点/03_结果/首笔vs第二笔_明细.csv      78 对可比重复信号
  08_重复信号与买入时点/03_结果/603778案例.csv
  08_重复信号与买入时点/03_结果/当前信号_250日回撤.csv

输出：03_结果/ 下中文表 + 指标.json。
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
BASE = HERE.parents[2]
RES = OUT / '03_结果'
RES.mkdir(parents=True, exist_ok=True)

TOP2_PATH = BASE / '06_不同信号日起点/03_结果/逐日首笔_逐只.csv'
ALL_PATH = BASE / '08_重复信号与买入时点/03_结果/全部信号重复序列_完整.csv'
PAIR_PATH = BASE / '08_重复信号与买入时点/03_结果/首笔vs第二笔_明细.csv'
CASE_PATH = BASE / '08_重复信号与买入时点/03_结果/603778案例.csv'
CUR_PATH = BASE / '08_重复信号与买入时点/03_结果/当前信号_250日回撤.csv'

EDGES = [-1.01, -0.75, -0.70, -0.65, -0.60, -0.55, -0.50, -0.44]
LABELS = ['≤-75%', '-75~-70%', '-70~-65%', '-65~-60%', '-60~-55%', '-55~-50%', '-50~-45%']


def add_bucket(df):
    df = df.copy()
    df['深度档'] = pd.cut(df['dd250'], bins=EDGES, labels=LABELS, right=False)
    return df


def load_data():
    t = pd.read_csv(TOP2_PATH)
    t = t[t['complete'] == True].copy()  # noqa: E712
    t['y'] = pd.to_datetime(t['signal_date']).dt.year
    t['m'] = pd.to_datetime(t['signal_date']).dt.strftime('%Y-%m')
    t['days'] = (pd.to_datetime(t['exit_date']) - pd.to_datetime(t['entry_date'])).dt.days

    a = pd.read_csv(ALL_PATH)
    a = a[a['complete'] == True].copy()  # noqa: E712
    a['days'] = (pd.to_datetime(a['exit_date']) - pd.to_datetime(a['entry_date'])).dt.days

    meta = a.drop_duplicates('signal_date').set_index('signal_date')
    t['阶段'] = t['signal_date'].map(meta['阶段'])
    t['mkt_dd250'] = t['signal_date'].map(meta['mkt_dd250'])

    t = add_bucket(t)
    a = add_bucket(a)
    return t, a


def cn_reason(x):
    return {'target': '止盈', 'time': '到期', 'time_suspended': '停牌到期'}.get(x, x)


def metrics(g):
    ex = g[g['y'] != 2026]
    return pd.Series({
        '笔数': len(g),
        '平均收益': g['ret'].mean(),
        '中位收益': g['ret'].median(),
        '胜率': (g['ret'] > 0).mean(),
        '止盈率': (g['exit_reason'] == 'target').mean(),
        '平均持有交易日': g['days'].mean(),
        '最差': g['ret'].min(),
        '最好': g['ret'].max(),
        '标准差': g['ret'].std(),
        '剔除2026平均': ex['ret'].mean() if len(ex) else np.nan,
        '剔除2026笔数': len(ex),
    })


def pct_round(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = (df[c] * 100).round(2)
    return df


def round_out(df, int_cols=(), p_cols=(), decimals=2):
    """输出前统一小数位；计数列转整数，p 值保留更多位。"""
    df = df.copy()
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype('Int64')
    for c in df.columns:
        if c in int_cols or not pd.api.types.is_float_dtype(df[c]):
            continue
        if c in p_cols:
            # 常规 p 保留 4 位；极小 p 保留科学计数，避免被抹成 0
            df[c] = df[c].map(lambda v: round(float(v), 4) if pd.notna(v) and float(v) >= 1e-4 else v)
            continue
        df[c] = df[c].round(decimals)
    return df


def bucket_table(D, sample):
    rows = []
    for lab in LABELS:
        g = D[D['深度档'] == lab]
        if len(g) == 0:
            rows.append({'样本': sample, '深度档': lab, '笔数': 0})
            continue
        s = metrics(g).to_dict()
        s['样本'] = sample
        s['深度档'] = lab
        rows.append(s)
    df = pd.DataFrame(rows)
    df = df[['样本', '深度档', '笔数', '平均收益', '中位收益', '胜率', '止盈率',
             '平均持有交易日', '最差', '最好', '标准差', '剔除2026平均', '剔除2026笔数']]
    df = pct_round(df, ['平均收益', '中位收益', '胜率', '止盈率', '最差', '最好', '标准差', '剔除2026平均'])
    df['平均持有交易日'] = df['平均持有交易日'].round(1)
    return round_out(df, int_cols=['笔数', '剔除2026笔数'])


def deep_shallow(D, sample, split):
    deep = D[D['dd250'] <= split]
    sh = D[D['dd250'] > split]
    exd = deep[deep['y'] != 2026]
    exs = sh[sh['y'] != 2026]
    return {
        '样本': sample,
        '分界': f'≤{split*100:.0f}% / >{split*100:.0f}%',
        '深组笔数': len(deep),
        '深组平均': deep['ret'].mean() * 100,
        '深组中位': deep['ret'].median() * 100,
        '深组胜率': (deep['ret'] > 0).mean() * 100,
        '深组止盈率': (deep['exit_reason'] == 'target').mean() * 100,
        '浅组笔数': len(sh),
        '浅组平均': sh['ret'].mean() * 100,
        '浅组中位': sh['ret'].median() * 100,
        '浅组胜率': (sh['ret'] > 0).mean() * 100,
        '浅组止盈率': (sh['exit_reason'] == 'target').mean() * 100,
        '平均差(百分点)': (deep['ret'].mean() - sh['ret'].mean()) * 100,
        '秩和检验p': stats.mannwhitneyu(deep['ret'], sh['ret'], alternative='two-sided').pvalue,
        '剔除2026深组平均': exd['ret'].mean() * 100,
        '剔除2026浅组平均': exs['ret'].mean() * 100,
        '剔除2026差(百分点)': (exd['ret'].mean() - exs['ret'].mean()) * 100,
        '剔除2026深组笔数': len(exd),
        '剔除2026浅组笔数': len(exs),
    }


def ols_year_fe(D, xcols):
    D = D.dropna(subset=['ret', 'y'] + xcols).copy()
    X = np.column_stack(
        [np.ones(len(D))]
        + [D[c].astype(float).values for c in xcols]
        + [pd.get_dummies(D['y'], drop_first=True, dtype=float).values]
    )
    y = D['ret'].astype(float).values
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    dof = len(D) - X.shape[1]
    s2 = float(resid @ resid / dof)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return b, se, len(D)


def rank_pair_table(T):
    ret = T.pivot_table(index='signal_date', columns='rank', values='ret', aggfunc='first')
    dd = T.pivot_table(index='signal_date', columns='rank', values='dd250', aggfunc='first')
    ret.index = pd.to_datetime(ret.index)
    dd.index = pd.to_datetime(dd.index)
    p = pd.DataFrame({
        '排名1收益': ret[1], '排名2收益': ret[2],
        '排名1回撤': dd[1], '排名2回撤': dd[2],
    }).dropna()
    p['收益差(百分点)'] = (p['排名1收益'] - p['排名2收益']) * 100
    p['年份'] = pd.to_datetime(p.index).year
    p = p.reset_index().rename(columns={'signal_date': '信号日'})
    p['排名1回撤'] = (p['排名1回撤'] * 100).round(2)
    p['排名2回撤'] = (p['排名2回撤'] * 100).round(2)
    p['排名1收益'] = (p['排名1收益'] * 100).round(2)
    p['排名2收益'] = (p['排名2收益'] * 100).round(2)
    p['收益差(百分点)'] = p['收益差(百分点)'].round(2)
    return p, ret.dropna(), dd


def pair_summary(ret, label):
    diff = ret[1] - ret[2]
    tt = stats.ttest_rel(ret[1], ret[2])
    return {
        '样本': label,
        '可配对数': len(ret),
        '排名1平均': ret[1].mean() * 100,
        '排名2平均': ret[2].mean() * 100,
        '平均差(百分点)': diff.mean() * 100,
        '配对t检验p': tt.pvalue,
        '排名1更好占比': (diff > 0).mean() * 100,
        '排名1中位': ret[1].median() * 100,
        '排名2中位': ret[2].median() * 100,
    }


def main():
    T, A = load_data()

    # 1. 分档
    bt = pd.concat([bucket_table(T, '前二'), bucket_table(A, '全部信号')], ignore_index=True)
    bt.to_csv(RES / '深度分档.csv', index=False, encoding='utf-8-sig')
    bucket_table(T, '前二').to_csv(RES / '前二深度分档.csv', index=False, encoding='utf-8-sig')
    bucket_table(A, '全部信号').to_csv(RES / '全部信号深度分档.csv', index=False, encoding='utf-8-sig')

    # 2. 深/浅对比
    ds = pd.DataFrame([
        deep_shallow(T, '前二', -0.65),
        deep_shallow(T, '前二', -0.60),
        deep_shallow(A, '全部信号', -0.65),
        deep_shallow(A, '全部信号', -0.60),
    ])
    ds = round_out(ds, int_cols=['深组笔数', '浅组笔数', '剔除2026深组笔数', '剔除2026浅组笔数'],
                   p_cols=['秩和检验p'])
    ds.to_csv(RES / '深度分组对比.csv', index=False, encoding='utf-8-sig')

    # 3. 同日排名配对
    pair_detail, pair_ret, pair_dd = rank_pair_table(T)
    pair_detail.to_csv(RES / '同日排名配对.csv', index=False, encoding='utf-8-sig')
    pr = pd.DataFrame([
        pair_summary(pair_ret, '全部'),
        pair_summary(pair_ret[pair_ret.index.year != 2026], '剔除2026'),
    ])
    pr = round_out(pr, int_cols=['可配对数'], p_cols=['配对t检验p'])
    pr.to_csv(RES / '同日排名对比.csv', index=False, encoding='utf-8-sig')

    year_pair = []
    for y, g in pair_ret.groupby(pair_ret.index.year):
        d = g[1] - g[2]
        year_pair.append({
            '年份': y, '可配对数': len(g),
            '排名1平均': g[1].mean() * 100, '排名2平均': g[2].mean() * 100,
            '平均差(百分点)': d.mean() * 100, '排名1更好占比': (d > 0).mean() * 100,
            '配对t检验p': stats.ttest_rel(g[1], g[2]).pvalue if len(g) > 1 else np.nan,
        })
    year_pair = round_out(pd.DataFrame(year_pair), int_cols=['年份', '可配对数'], p_cols=['配对t检验p'])
    year_pair.to_csv(RES / '同日排名_年度.csv', index=False, encoding='utf-8-sig')

    # 4. 日级：当天前二平均深度 vs 当天前二平均收益
    day = T.groupby('signal_date').agg(
        前二平均回撤=('dd250', 'mean'),
        前二平均收益=('ret', 'mean'),
        前二只数=('ret', 'size'),
        当日信号数=('n_signals', 'first'),
        年份=('y', 'first'),
    ).reset_index().rename(columns={'signal_date': '信号日'})
    day['前二平均回撤'] = (day['前二平均回撤'] * 100).round(2)
    day['前二平均收益'] = (day['前二平均收益'] * 100).round(2)
    day.to_csv(RES / '信号日深度与组合收益.csv', index=False, encoding='utf-8-sig')

    day_raw = T.groupby('signal_date').agg(dd=('dd250', 'mean'), ret=('ret', 'mean'))
    day_ex = T[T['y'] != 2026].groupby('signal_date').agg(dd=('dd250', 'mean'), ret=('ret', 'mean'))
    day_rows = []
    for lab, g in [('全部', day_raw), ('剔除2026', day_ex)]:
        day_rows.append({
            '样本': lab, '信号日数': len(g),
            '皮尔逊相关': stats.pearsonr(g['dd'], g['ret']).statistic,
            '皮尔逊p': stats.pearsonr(g['dd'], g['ret']).pvalue,
            '斯皮尔曼相关': stats.spearmanr(g['dd'], g['ret']).statistic,
            '斯皮尔曼p': stats.spearmanr(g['dd'], g['ret']).pvalue,
        })
    day_rows = round_out(pd.DataFrame(day_rows), int_cols=['信号日数'],
                         p_cols=['皮尔逊p', '斯皮尔曼p'], decimals=3)
    day_rows.to_csv(RES / '日级相关性.csv', index=False, encoding='utf-8-sig')

    # 4b. 拥挤度：当天信号数三分组，深度排序是否仍有效
    crowd = T.groupby('signal_date').agg(
        回撤=('dd250', 'mean'), 收益=('ret', 'mean'), 当日信号数=('n_signals', 'first')
    ).reset_index()
    crowd['拥挤度'] = pd.qcut(crowd['当日信号数'], 3, labels=['小池子', '中池子', '大池子'])
    crowd_rows = []
    for lab, g in crowd.groupby('拥挤度', observed=True):
        r = stats.spearmanr(g['回撤'], g['收益'])
        crowd_rows.append({
            '拥挤度': lab, '信号日数': len(g),
            '当日信号数中位': g['当日信号数'].median(),
            '平均回撤': g['回撤'].mean() * 100,
            '平均收益': g['收益'].mean() * 100,
            '斯皮尔曼相关': r.statistic, '显著性p': r.pvalue,
        })
    crowd_tbl = round_out(pd.DataFrame(crowd_rows), int_cols=['信号日数'],
                          p_cols=['显著性p'], decimals=3)
    crowd_tbl.to_csv(RES / '信号拥挤度分组.csv', index=False, encoding='utf-8-sig')

    # 5. 回归：每深 10pp 的收益
    T['depth_pp'] = (-T['dd250'] - 0.45) * 100
    A['depth_pp'] = (-A['dd250'] - 0.45) * 100
    reg_rows = []
    for sample, D, xcols, model in [
        ('前二', T, ['depth_pp'], '年度固定效应'),
        ('前二', T, ['depth_pp', 'mkt_dd250'], '年度固定效应+市场回撤'),
        ('全部信号', A, ['depth_pp'], '年度固定效应'),
        ('全部信号', A, ['depth_pp', 'mkt_dd250'], '年度固定效应+市场回撤'),
    ]:
        b, se, n = ols_year_fe(D, xcols)
        row = {
            '样本': sample, '模型': model, '样本数': n,
            '每深10个百分点收益(百分点)': b[1] * 10 * 100,
            '标准误(百分点)': se[1] * 10 * 100,
            't统计量': b[1] / se[1],
        }
        if 'mkt_dd250' in xcols:
            row['市场每深10个百分点收益(百分点)'] = -b[2] * 0.1 * 100
            row['市场t统计量'] = b[2] / se[2]
        reg_rows.append(row)
    reg = round_out(pd.DataFrame(reg_rows), int_cols=['样本数'])
    reg.to_csv(RES / '深度回归.csv', index=False, encoding='utf-8-sig')

    # 6. 年度 / 阶段 深 vs 浅
    def split_table(D, group_col, sample):
        rows = []
        for key, g in D.groupby(group_col):
            deep = g[g['dd250'] <= -0.65]
            sh = g[g['dd250'] > -0.65]
            rows.append({
                '样本': sample, group_col: key,
                '深组笔数': len(deep), '深组平均': deep['ret'].mean() * 100 if len(deep) else np.nan,
                '深组胜率': (deep['ret'] > 0).mean() * 100 if len(deep) else np.nan,
                '浅组笔数': len(sh), '浅组平均': sh['ret'].mean() * 100 if len(sh) else np.nan,
                '浅组胜率': (sh['ret'] > 0).mean() * 100 if len(sh) else np.nan,
                '平均差(百分点)': (deep['ret'].mean() - sh['ret'].mean()) * 100 if len(deep) and len(sh) else np.nan,
            })
        return pd.DataFrame(rows)

    year_tbl = pd.concat([split_table(T, 'y', '前二'), split_table(A, 'y', '全部信号')],
                         ignore_index=True).rename(columns={'y': '年份'})
    year_tbl = round_out(year_tbl, int_cols=['年份', '深组笔数', '浅组笔数'])
    year_tbl.to_csv(RES / '年度深度对比.csv', index=False, encoding='utf-8-sig')
    phase_tbl = pd.concat([split_table(T, '阶段', '前二'), split_table(A, '阶段', '全部信号')],
                          ignore_index=True)
    phase_tbl = round_out(phase_tbl, int_cols=['深组笔数', '浅组笔数'])
    phase_tbl.to_csv(RES / '阶段深度对比.csv', index=False, encoding='utf-8-sig')

    # 7. 2026 月度
    rows = []
    for m, g in T[T['y'] == 2026].groupby('m'):
        deep = g[g['dd250'] <= -0.65]
        sh = g[g['dd250'] > -0.65]
        rows.append({
            '月份': m, '笔数': len(g), '平均收益': g['ret'].mean() * 100,
            '胜率': (g['ret'] > 0).mean() * 100, '平均回撤': g['dd250'].mean() * 100,
            '深组笔数': len(deep), '深组平均': deep['ret'].mean() * 100 if len(deep) else np.nan,
            '浅组笔数': len(sh), '浅组平均': sh['ret'].mean() * 100 if len(sh) else np.nan,
        })
    m26 = round_out(pd.DataFrame(rows), int_cols=['笔数', '深组笔数', '浅组笔数'])
    m26.to_csv(RES / '2026深度分档.csv', index=False, encoding='utf-8-sig')

    # 8. 机制：止盈/到期、持有天数、尾部
    T['深度组'] = np.where(T['dd250'] <= -0.65, '≤-65%', '>-65%')
    mech = T.groupby('深度组').apply(lambda g: pd.Series({
        '笔数': len(g),
        '止盈占比': (g['exit_reason'] == 'target').mean() * 100,
        '到期占比': (g['exit_reason'] == 'time').mean() * 100,
        '停牌到期占比': (g['exit_reason'] == 'time_suspended').mean() * 100,
        '平均持有交易日': g['days'].mean(),
        '平均收益': g['ret'].mean() * 100,
        '中位收益': g['ret'].median() * 100,
        '最差': g['ret'].min() * 100,
        '最好': g['ret'].max() * 100,
        '标准差': g['ret'].std() * 100,
    }), include_groups=False).reset_index()
    mech = round_out(mech, int_cols=['笔数'])
    mech.to_csv(RES / '深度机制.csv', index=False, encoding='utf-8-sig')

    # 9. 前二逐笔（含深度档），供图表与复核
    T['退出原因中文'] = T['exit_reason'].map(cn_reason)
    trade = T[['signal_date', 'rank', 'code', 'dd250', 'ret', '退出原因中文', 'days', 'y', '深度档']].copy()
    trade = trade.rename(columns={'signal_date': '信号日', 'rank': '排名', 'code': '代码',
                                  'dd250': '回撤250', 'ret': '收益', '退出原因中文': '退出原因',
                                  'days': '持有交易日', 'y': '年份'})
    trade['回撤250'] = (trade['回撤250'] * 100).round(2)
    trade['收益'] = (trade['收益'] * 100).round(2)
    trade = trade.sort_values(['信号日', '排名'])
    trade.to_csv(RES / '前二逐笔_深度.csv', index=False, encoding='utf-8-sig')

    # 10. ≤-70% 极值明细
    ext = T[T['dd250'] <= -0.70].sort_values('signal_date')[
        ['signal_date', 'code', 'dd250', 'ret', '退出原因中文', 'y']
    ].rename(columns={'signal_date': '信号日', 'code': '代码', 'dd250': '回撤250',
                      'ret': '收益', '退出原因中文': '退出原因', 'y': '年份'})
    ext['回撤250'] = (ext['回撤250'] * 100).round(2)
    ext['收益'] = (ext['收益'] * 100).round(2)
    ext.to_csv(RES / '深度极值_70以上.csv', index=False, encoding='utf-8-sig')

    # 10. 重复信号：第二笔更深是否更好
    P = pd.read_csv(PAIR_PATH)
    P['回撤变化(百分点)'] = (P['second_dd'] - P['first_dd']) * 100
    P['收益差(百分点)'] = (P['second'] - P['first']) * 100
    P['第二笔更深'] = P['回撤变化(百分点)'] < 0
    pd_detail = P.rename(columns={
        'code': '代码', 'first_date': '首笔信号日', 'second_date': '第二笔信号日',
        'first': '首笔收益', 'second': '第二笔收益', 'first_dd': '首笔回撤',
        'second_dd': '第二笔回撤', 'y': '年份',
    })
    pd_detail = pd_detail[['代码', '首笔信号日', '第二笔信号日', '首笔回撤', '第二笔回撤',
                           '回撤变化(百分点)', '首笔收益', '第二笔收益', '收益差(百分点)', '第二笔更深', '年份']]
    for c in ['首笔回撤', '第二笔回撤']:
        pd_detail[c] = (pd_detail[c] * 100).round(2)
    for c in ['首笔收益', '第二笔收益', '收益差(百分点)']:
        pd_detail[c] = pd_detail[c].round(2)
    pd_detail['回撤变化(百分点)'] = pd_detail['回撤变化(百分点)'].round(2)
    pd_detail.to_csv(RES / '重复信号深度_配对.csv', index=False, encoding='utf-8-sig')

    rep_rows = []
    for key, g in P.groupby('第二笔更深'):
        rep_rows.append({
            '分组': '第二笔更深' if key else '第二笔更浅',
            '对数': len(g),
            '第二笔-第一笔收益差(百分点)': g['收益差(百分点)'].mean(),
            '第二笔更好占比': (g['收益差(百分点)'] > 0).mean() * 100,
            '回撤变化均值(百分点)': g['回撤变化(百分点)'].mean(),
        })
    rep = pd.DataFrame(rep_rows)
    corr_all = stats.pearsonr(P['回撤变化(百分点)'], P['收益差(百分点)'])
    P_ex = P[P['y'] != 2026]
    corr_ex = stats.pearsonr(P_ex['回撤变化(百分点)'], P_ex['收益差(百分点)'])
    rep_extra = pd.DataFrame([
        {'项目': '回撤变化与收益差相关（全部78对）', '数值': corr_all.statistic, '显著性p': corr_all.pvalue,
         '说明': '正相关=第二笔更浅时反而更好'},
        {'项目': '回撤变化与收益差相关（剔除2026）', '数值': corr_ex.statistic, '显著性p': corr_ex.pvalue,
         '说明': f'n={len(P_ex)}'},
    ])
    rep = round_out(rep, int_cols=['对数'])
    rep.to_csv(RES / '重复信号深度_汇总.csv', index=False, encoding='utf-8-sig')
    rep_extra = round_out(rep_extra, p_cols=['显著性p'], decimals=3)
    rep_extra.to_csv(RES / '重复信号深度_相关性.csv', index=False, encoding='utf-8-sig')

    # 段内（全部信号）与 603778 案例
    E = A.copy()
    E['ret_dm'] = E['ret'] - E.groupby('eid')['ret'].transform('mean')
    E['dd_dm'] = E['dd250'] - E.groupby('eid')['dd250'].transform('mean')
    within = stats.pearsonr(E['dd_dm'], E['ret_dm'])
    within_ex = stats.pearsonr(E[E['y'] != 2026]['dd_dm'], E[E['y'] != 2026]['ret_dm'])

    C = pd.read_csv(CASE_PATH)
    C['信号日'] = pd.to_datetime(C['信号日'])
    Cok = C[(C['状态'] == 'ok') & (C['是否前二'] == True)]  # noqa: E712
    case_corr = stats.pearsonr(Cok['回撤250'], Cok['收益'])
    case_spear = stats.spearmanr(Cok['回撤250'], Cok['收益'])
    C2 = C.rename(columns={'是否前二': '是否前二', '段内位置': '段内位置', 'RSI': '相对强弱14'})
    C2['是否前二'] = C2['是否前二'].map({True: '是', False: '否'})
    C2['状态'] = C2['状态'].map(lambda x: '已完成' if x == 'ok' else x)
    C2['回撤250'] = (C2['回撤250'] * 100).round(2)
    C2['收益'] = C2['收益'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
    C2[['信号日', '排名', '是否前二', '段内位置', '回撤250', '相对强弱14', '收盘位置',
        '入场日', '退出日', '收益', '状态']].to_csv(
        RES / '603778_深度与收益.csv', index=False, encoding='utf-8-sig')

    # 11. 当前信号：信号日当天回撤 + 同档历史
    CUR = pd.read_csv(CUR_PATH, parse_dates=['信号日', '最高日_hfq', '最高日_实际'])
    cur = CUR.copy()
    cur['深度档'] = pd.cut(cur['回撤_hfq'], bins=EDGES, labels=LABELS, right=False)

    def bucket_stat(b):
        g = T[T['深度档'] == b]
        g26 = g[g['y'] == 2026]
        return {
            '历史同档笔数': len(g),
            '历史同档平均': g['ret'].mean() * 100 if len(g) else np.nan,
            '历史同档胜率': (g['ret'] > 0).mean() * 100 if len(g) else np.nan,
            '2026同档笔数': len(g26),
            '2026同档平均': g26['ret'].mean() * 100 if len(g26) else np.nan,
            '2026同档胜率': (g26['ret'] > 0).mean() * 100 if len(g26) else np.nan,
        }

    notes = {
        '605117.SH': '10转4，实际价回撤被除权放大；信号看后复权',
        '603778.SH': '已持有（10/15信号、10/20入场），不重复买',
        '605376.SH': '10/23第2名；已持有603778时，空位下的第2名候选',
        '002969.SZ': '10/23第3名；不在前二，规则不买',
        '603256.SH': '10/23第4名，非前二',
        '002796.SZ': '10/23第5名，非前二',
    }
    rows = []
    for _, r in cur.iterrows():
        s = bucket_stat(r['深度档'])
        rows.append({
            '信号日': r['信号日'], '排名': int(r['排名']), '代码': r['代码'],
            '信号日收盘_后复权': round(float(r['当日收盘_hfq']), 2),
            '250日最高_后复权': round(float(r['最高250_hfq']), 2),
            '信号日回撤_后复权': round(float(r['回撤_hfq']) * 100, 2),
            '深于-45(百分点)': round(float(r['深于45阈值_pp']), 2),
            '实际价回撤': round(float(r['回撤_实际']) * 100, 2),
            '深度档': str(r['深度档']),
            **{k: (round(v, 2) if isinstance(v, float) and pd.notna(v) else v) for k, v in s.items()},
            '备注': notes.get(r['代码'], ''),
        })
    cur_df = pd.DataFrame(rows).sort_values(['信号日', '排名'])
    cur_df.to_csv(RES / '当前信号_深度对照.csv', index=False, encoding='utf-8-sig')

    # 12. 指标
    def p(x):
        return round(float(x), 4)

    deep_t = T[T['dd250'] <= -0.65]
    shallow_t = T[T['dd250'] > -0.65]
    deep_a = A[A['dd250'] <= -0.65]
    shallow_a = A[A['dd250'] > -0.65]
    june = T[(T['y'] == 2026) & (T['m'] == '2026-06')]
    july = T[(T['y'] == 2026) & (T['m'] == '2026-07')]

    metrics_json = {
        '样本': {
            '前二完整': int(len(T)),
            '前二信号日': int(T['signal_date'].nunique()),
            '全部信号完整': int(len(A)),
            '全部信号信号日': int(A['signal_date'].nunique()),
        },
        '深度分档': {
            '前二最浅一档': p(shallow_t[shallow_t['深度档'] == '-50~-45%']['ret'].mean()),
            '前二≤-65笔数': int(len(deep_t)),
            '前二≤-65平均': p(deep_t['ret'].mean()),
            '前二＞-65平均': p(shallow_t['ret'].mean()),
            '前二平均差(百分点)': p((deep_t['ret'].mean() - shallow_t['ret'].mean()) * 100),
            '前二≤-65胜率': p((deep_t['ret'] > 0).mean()),
            '前二＞-65胜率': p((shallow_t['ret'] > 0).mean()),
            '前二≤-65止盈率': p((deep_t['exit_reason'] == 'target').mean()),
            '前二＞-65止盈率': p((shallow_t['exit_reason'] == 'target').mean()),
            '全部≤-65平均': p(deep_a['ret'].mean()),
            '全部＞-65平均': p(shallow_a['ret'].mean()),
            '全部平均差(百分点)': p((deep_a['ret'].mean() - shallow_a['ret'].mean()) * 100),
        },
        '同日排名': {
            '可配对数': int(len(pair_ret)),
            '排名1平均': p(pair_ret[1].mean()),
            '排名2平均': p(pair_ret[2].mean()),
            '平均差(百分点)': p((pair_ret[1].mean() - pair_ret[2].mean()) * 100),
            '配对t检验p': p(stats.ttest_rel(pair_ret[1], pair_ret[2]).pvalue),
            '排名1更好占比': p(((pair_ret[1] - pair_ret[2]) > 0).mean()),
        },
        '拥挤度': {r['拥挤度']: {'相关': p(r['斯皮尔曼相关']), 'p': p(r['显著性p'])} for r in crowd_rows},
        '回归': {r['样本'] + '_' + r['模型']: {
            '每深10个百分点收益(百分点)': p(r['每深10个百分点收益(百分点)']), 't统计量': p(r['t统计量'])
        } for r in reg_rows},
        '2026': {
            '6月深组笔数': int(len(june[june['dd250'] <= -0.65])),
            '6月深组平均': p(june[june['dd250'] <= -0.65]['ret'].mean()),
            '6月浅组笔数': int(len(june[june['dd250'] > -0.65])),
            '6月浅组平均': p(june[june['dd250'] > -0.65]['ret'].mean()),
            '7月深组笔数': int(len(july[july['dd250'] <= -0.65])),
            '7月深组平均': p(july[july['dd250'] <= -0.65]['ret'].mean()),
            '7月浅组平均': p(july[july['dd250'] > -0.65]['ret'].mean()),
        },
        '重复信号': {
            '第二笔更深对数': int(P['第二笔更深'].sum()),
            '第二笔更浅对数': int((~P['第二笔更深']).sum()),
            '深度变化与收益差相关': p(corr_all.statistic),
            '相关显著性p': p(corr_all.pvalue),
            '剔除2026相关': p(corr_ex.statistic),
            '剔除2026显著性p': p(corr_ex.pvalue),
            '段内相关': p(within.statistic),
            '段内相关p': p(within.pvalue),
            '段内相关_剔除2026': p(within_ex.statistic),
            '603778_7笔相关': p(case_corr.statistic),
            '603778_7笔相关p': p(case_corr.pvalue),
        },
        '当前': {
            '603778回撤': -78.61,
            '605376回撤': -71.30,
            '002969回撤': -70.68,
            '605117回撤': -45.32,
            '605117实际回撤': -61.07,
        },
    }
    with open(RES / '指标.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics_json, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
