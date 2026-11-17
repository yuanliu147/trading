# -*- coding: utf-8 -*-
"""主板专用：70/15 信号面的多角度诊断。

目的不是再讲一遍“全A怎么样”，而是回答：
1) 只做主板后，原 70/15 的信号在哪些特征上还有区分度？
2) 这些区分度在 2018-2021（训练）和 2022-2026（检验）是否同向？
3) 行业、市场宽度、个股路径有什么结构性差异？
4) 哪些维度值得进组合回测，哪些只是样本内噪声？

口径：
- 主板：沪 600/601/603/605；深 000/001/002/003。排除创业板、科创板、北交所。
- 信号日 t 收盘；收益 fwd20 = C(t+20)/O(t+1)-1，与原系统 20 日持有对齐。
- excess20 = fwd20 - 当日全市场 fwd20 中位数，用于剥离市场 beta。
- 原 70/15 基线：宽度>50%、上升结构、年龄 5-30、ATR<=4.4%、20日无涨停、
  行业宽度>=60%、行业20日>5%、DD60>-8%、成交额 3000万-7000万、股价<=15。
- 本脚本只用历史面板，不做交易建议。
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

BASE = '/Users/admin/WorkSpace/trading'
SRC = os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制')
PANEL = os.path.join(SRC, '01_数据/全A面板_2016_2026_含退市.pkl')
OUT = os.path.join(BASE, '2026-10-23_主板专用_策略优化')

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

TRAIN0, TRAIN1 = '2018-01-01', '2021-12-31'
TEST0, TEST1 = '2022-01-01', '2026-10-13'
START = '2017-10-13'


def board_of(c):
    c = str(c)
    if c.startswith(('600', '601', '603', '605')):
        return '沪主板'
    if c.startswith(('000', '001', '002', '003')):
        return '深主板'
    if c.startswith(('300', '301')):
        return '创业板'
    if c.startswith(('688', '689')):
        return '科创板'
    if c.startswith(('8', '4', '920')):
        return '北交所'
    return '其他'


def qbins(x, q=5):
    """按分位切组；缺失值保留为 NA，重复值用 rank 兜底。"""
    s = pd.Series(x)
    b = pd.Series(np.nan, index=s.index)
    ok = s.replace([np.inf, -np.inf], np.nan).notna()
    if ok.sum() < q * 20:
        return b
    v = s[ok]
    try:
        b.loc[ok] = pd.qcut(v, q, labels=False, duplicates='drop').values
    except ValueError:
        r = v.rank(method='first', pct=True)
        b.loc[ok] = np.minimum((r * q).astype(int).values, q - 1)
    return b


def main():
    P = pickle.load(open(PANEL, 'rb'))
    dates = pd.DatetimeIndex(P['meta']['dates'])
    codes = np.array(P['meta']['codes'])
    industries = P['meta']['industries']
    g = lambda k: np.asarray(P[k])

    C = g('close').astype(float)
    AMT = g('amt20').astype(float)
    ATR = g('atr_pct').astype(float)
    AGE = g('trend_age').astype(float)
    DD60 = g('dd60').astype(float)
    U = g('up_struct').astype(bool)
    LU = g('lu_count20').astype(float)
    SBR = g('sector_breadth20').astype(float)
    SR20 = g('sector_ret20').astype(float)
    MB = g('market_breadth20').astype(float)
    F20 = g('fwd20').astype(float)
    E20 = g('excess20').astype(float)
    FMIN = g('fwd_min20_entry').astype(float)
    FMAX = g('fwd_max20_entry').astype(float)
    RSI = g('rsi14').astype(float)
    RET20 = g('ret20').astype(float)
    DIST20 = g('dist_ma20').astype(float)
    DIST60 = g('dist_ma60').astype(float)
    MCAP = g('mcap').astype(float)
    TURN = g('turn').astype(float)
    VOL20 = g('vol20').astype(float)
    VR = g('vr5_20').astype(float)
    AMTR = g('amt_ratio20').astype(float)
    AMTT = g('amt_trend').astype(float)
    OWNB = g('own_base20').astype(float)
    OWNW = g('own_win20').astype(float)
    GAPC = g('gap_count20').astype(float)
    UPR = g('up_ratio20').astype(float)
    ER20 = g('er20').astype(float)
    CPOS = g('close_pos').astype(float)
    MACD = g('macd_hist').astype(float)
    NG = g('next_gap').astype(float)
    GAP = g('gap').astype(float)
    MKT = g('market_index').astype(float)
    MKT20 = g('market_ret20').astype(float)
    REG = g('regime_id').astype(float)
    IND = np.asarray(P['industry_id'])
    MKT_V = pd.Series(MKT).pct_change().rolling(20, min_periods=15).std().values * np.sqrt(252)

    main_mask = np.array([board_of(c) in ('沪主板', '深主板') for c in codes], dtype=bool)[None, :]
    date_ok = (dates >= pd.Timestamp(START))[:, None]
    valid = U & (AMT >= 3e7) & (C >= 2) & (AGE >= 5) & date_ok & main_mask
    base70 = (valid & (AGE <= 30) & (ATR <= 0.044) & (LU == 0) &
              (SBR >= 0.6) & (SR20 > 0.05) & (DD60 > -0.08) &
              (AMT <= 70e6) & (C <= 15) & (MB > 0.5)[:, None] & np.isfinite(F20))
    ii, jj = np.where(base70)
    print('主板 70/15 信号数', len(ii))

    F = pd.DataFrame({
        '日期': dates[ii], '代码': codes[jj], '行业': [industries[int(IND[i, j])]
                                                   if 0 <= int(IND[i, j]) < len(industries) else '未知'
                                                   for i, j in zip(ii, jj)],
        '板块': [board_of(c) for c in codes[jj]],
        'fwd20': F20[ii, jj], '超额20': E20[ii, jj],
        '最大不利': FMIN[ii, jj], '最大有利': FMAX[ii, jj],
        '成交额20': AMT[ii, jj] / 1e6, '收盘价': C[ii, jj], 'ATR': ATR[ii, jj],
        '趋势年龄': AGE[ii, jj], 'DD60': DD60[ii, jj], 'RSI14': RSI[ii, jj],
        '20日涨幅': RET20[ii, jj], '距MA20': DIST20[ii, jj], '距MA60': DIST60[ii, jj],
        '市值': MCAP[ii, jj] / 1e8, '换手率': TURN[ii, jj], '20日波动': VOL20[ii, jj],
        '量比5_20': VR[ii, jj], '成交额比20': AMTR[ii, jj], '成交额趋势': AMTT[ii, jj],
        '自身基准率': OWNB[ii, jj], '自身胜率': OWNW[ii, jj], '跳空次数20': GAPC[ii, jj],
        '上涨日占比20': UPR[ii, jj], '效率ER20': ER20[ii, jj], '收盘位置': CPOS[ii, jj],
        'MACD柱': MACD[ii, jj], '当日跳空': GAP[ii, jj], '次日跳空': NG[ii, jj],
        '行业宽度': SBR[ii, jj], '行业20日': SR20[ii, jj], '市场宽度': MB[ii],
        '市场20日': MKT20[ii], '市场波动': MKT_V[ii], '市场状态': REG[ii],
    })
    F['年份'] = F['日期'].dt.year
    F['训练期'] = np.where(F['日期'] <= pd.Timestamp(TRAIN1), '2018-2021', '2022-2026')
    F['盈利'] = F['fwd20'] > 0
    F['赢家'] = np.where(F['fwd20'] > 0, '赢家', '输家')
    F.to_csv(os.path.join(OUT, '01_信号样本.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f', date_format='%Y-%m-%d')

    feats = ['成交额20', '收盘价', 'ATR', '趋势年龄', 'DD60', 'RSI14', '20日涨幅', '距MA20',
             '距MA60', '市值', '换手率', '20日波动', '量比5_20', '成交额比20', '成交额趋势',
             '自身基准率', '自身胜率', '跳空次数20', '上涨日占比20', '效率ER20', '收盘位置',
             'MACD柱', '当日跳空', '次日跳空', '行业宽度', '行业20日', '市场宽度', '市场20日',
             '市场波动']
    feats = [f for f in feats if f in F.columns]

    # 1) 全样本五分位
    rows = []
    for f in feats:
        x = F[f].replace([np.inf, -np.inf], np.nan)
        if x.notna().sum() < 100:
            continue
        b = qbins(x, 5)
        for q in range(5):
            s = F.loc[b == q]
            if len(s) == 0:
                continue
            rows.append({'特征': f, '分位': q + 1, '样本': len(s),
                         '下界': s[f].min(), '上界': s[f].max(), '中位': s[f].median(),
                         '20日均值': s['fwd20'].mean(), '超额均值': s['超额20'].mean(),
                         '胜率': s['盈利'].mean(), 'P10': s['fwd20'].quantile(.10),
                         '最大不利均值': s['最大不利'].mean()})
    Q = pd.DataFrame(rows)
    Q.to_csv(os.path.join(OUT, '01_信号特征分位.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')

    # 2) 训练/检验 + 逐年同向性
    stab = []
    for f in feats:
        x = F[f].replace([np.inf, -np.inf], np.nan)
        if x.notna().sum() < 200:
            continue
        b = qbins(x, 5)
        d = F.assign(Q=b)
        sp = d.groupby('Q').agg(超额=('超额20', 'mean'), 样本=('超额20', 'size'))
        if len(sp) < 5:
            continue
        spread = sp.loc[4, '超额'] - sp.loc[0, '超额']
        tr = d[d['训练期'] == '2018-2021'].groupby('Q').agg(超额=('超额20', 'mean'))
        te = d[d['训练期'] == '2022-2026'].groupby('Q').agg(超额=('超额20', 'mean'))
        if len(tr) < 5 or len(te) < 5:
            continue
        trs = tr.loc[4, '超额'] - tr.loc[0, '超额']
        tes = te.loc[4, '超额'] - te.loc[0, '超额']
        yearly = d.groupby(['年份', 'Q']).agg(超额=('超额20', 'mean')).reset_index()
        yspread = []
        for y, gy in yearly.groupby('年份'):
            gy = gy.set_index('Q')
            if 4 in gy.index and 0 in gy.index:
                yspread.append(gy.loc[4, '超额'] - gy.loc[0, '超额'])
        same = np.mean([np.sign(v) == np.sign(spread) for v in yspread]) if yspread else np.nan
        stab.append({'特征': f, '全期高低差': spread, '训练高低差': trs, '检验高低差': tes,
                     '训练检验同向': np.sign(trs) == np.sign(tes),
                     '逐年同向比例': same, '有效年份': len(yspread),
                     'Q5样本': int(sp.loc[4, '样本']), 'Q1样本': int(sp.loc[0, '样本'])})
    S = pd.DataFrame(stab).sort_values(['训练检验同向', '逐年同向比例', '全期高低差'],
                                       ascending=[False, False, False])
    S.to_csv(os.path.join(OUT, '01_特征稳定性.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
    print('\n特征稳定性（训练/检验同向，按逐年同向比例排序）')
    print(S.round(4).to_string(index=False))

    # 3) 行业与板块
    ind = F.groupby('行业').agg(样本=('fwd20', 'size'), 均值=('fwd20', 'mean'),
                                超额=('超额20', 'mean'), 胜率=('盈利', 'mean'),
                                最大不利=('最大不利', 'mean')).sort_values('样本', ascending=False)
    ind.to_csv(os.path.join(OUT, '01_行业分布.csv'), encoding='utf-8-sig', float_format='%.6f')
    board = F.groupby('板块').agg(样本=('fwd20', 'size'), 均值=('fwd20', 'mean'),
                                  超额=('超额20', 'mean'), 胜率=('盈利', 'mean'),
                                  最大不利=('最大不利', 'mean'))
    board.to_csv(os.path.join(OUT, '01_板块分布.csv'), encoding='utf-8-sig', float_format='%.6f')

    # 4) 市场状态
    F['宽度桶'] = pd.cut(F['市场宽度'], [.5, .55, .6, .65, .7, .8, 1.0],
                        labels=['50-55', '55-60', '60-65', '65-70', '70-80', '80+'])
    F['市场波动桶'] = pd.cut(F['市场波动'], [0, .15, .20, .25, .30, 2],
                         labels=['<15%', '15-20%', '20-25%', '25-30%', '>30%'])
    mkt = F.groupby(['宽度桶'], observed=True).agg(样本=('fwd20', 'size'), 均值=('fwd20', 'mean'),
                                                   超额=('超额20', 'mean'), 胜率=('盈利', 'mean'),
                                                   最大不利=('最大不利', 'mean'))
    mkt.to_csv(os.path.join(OUT, '01_市场宽度分桶.csv'), encoding='utf-8-sig', float_format='%.6f')
    mktv = F.groupby(['市场波动桶'], observed=True).agg(样本=('fwd20', 'size'), 均值=('fwd20', 'mean'),
                                                        超额=('超额20', 'mean'), 胜率=('盈利', 'mean'),
                                                        最大不利=('最大不利', 'mean'))
    mktv.to_csv(os.path.join(OUT, '01_市场波动分桶.csv'), encoding='utf-8-sig', float_format='%.6f')

    # 5) 赢家/输家路径
    wl = F.groupby('赢家').agg(样本=('fwd20', 'size'), 平均收益=('fwd20', 'mean'),
                               平均最大不利=('最大不利', 'mean'), 平均最大有利=('最大有利', 'mean'),
                               曾浮亏比例=('最大不利', lambda x: (x < 0).mean()),
                               曾浮盈比例=('最大有利', lambda x: (x > 0).mean()))
    wl.to_csv(os.path.join(OUT, '01_赢家输家路径.csv'), encoding='utf-8-sig', float_format='%.6f')

    # 6) 图：最稳的 6 个特征的五分位 + 年度高低差
    top = S.head(6)['特征'].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, f in zip(axes.flat, top):
        x = F[f].replace([np.inf, -np.inf], np.nan)
        b = qbins(x, 5)
        d = F.assign(Q=b).groupby('Q').agg(超额=('超额20', 'mean'), 胜率=('盈利', 'mean'),
                                           样本=('超额20', 'size'))
        ax.bar(d.index + 1, d['超额'], color='#3b6ea5')
        ax.axhline(0, color='black', lw=.8)
        ax.set_title(f'{f}：超额20（Q1→Q5）', fontsize=10)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        for q, row in d.iterrows():
            ax.text(q + 1, row['超额'], f"n={int(row['样本'])}", ha='center',
                    va='bottom' if row['超额'] >= 0 else 'top', fontsize=6)
    fig.suptitle('主板 70/15 信号：训练/检验同向的特征（五分位超额）', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '01_主板信号诊断.png'), dpi=160)
    plt.close()

    print('\n行业（前10）')
    print(ind.head(10).round(4).to_string())
    print('\n板块')
    print(board.round(4).to_string())
    print('\n宽度')
    print(mkt.round(4).to_string())
    print('\n市场波动')
    print(mktv.round(4).to_string())
    print('\n赢家输家')
    print(wl.round(4).to_string())


if __name__ == '__main__':
    main()
