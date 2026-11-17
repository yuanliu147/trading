# -*- coding: utf-8 -*-
"""70/15 的两个负收益年，以及“弱势区 → 闸门重启”的历史收益。

输出：
  13_年度归因.csv
  13_负年份交易.csv
  13_负年份月度.csv
  13_闸门重启事件.csv
  13_闸门重启_汇总.csv
  13_年度自举.csv
  13_负年份与闸门重启.png

口径：
  - 组合：11_base_proper_权益.csv / 11_base_proper_逐笔.csv（70/15 基准，2仓45%，20日，-20%止损）。
  - 面板：2018-01 至 2026-10-13；市场宽度、等权指数、MA20/60/120 均来自全A面板。
  - 年度收益：上年末到本年末；2018 从 20,000 起始。逐笔统计按买入年份归集，跨年交易与净值会有小幅差异。
  - 弱势区：宽度20 < 20% 且等权指数同时低于 MA20/60/120，连续至少 5 个交易日。
  - 闸门重启：弱势区结束后的第一个“宽度20 > 50%”交易日。
  - 同一闸门若对应多段弱势，保留最低宽度的一段；前向收益从闸门日收盘算起。
"""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

BASE = '/Users/admin/WorkSpace/trading'
PANEL = os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl')
EQ = '11_base_proper_权益.csv'
TR = '11_base_proper_逐笔.csv'
START, END = '2018-01-01', '2026-10-13'

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    P = pickle.load(open(PANEL, 'rb'))
    dates = pd.DatetimeIndex(P['meta']['dates'])
    g = lambda k: np.asarray(P[k]).ravel()
    M = pd.DataFrame({
        'mkt': g('market_index'), 'mb': g('market_breadth20'),
        'ma20': g('market_ma20'), 'ma60': g('market_ma60'), 'ma120': g('market_ma120'),
        'slope': g('market_slope60'), 'r20': g('market_ret20'),
    }, index=dates).loc['2017-10-13':]
    M['below3'] = (M.mkt < M.ma20) & (M.mkt < M.ma60) & (M.mkt < M.ma120)
    M['weak'] = (M.mb < 0.20) & M.below3

    E = pd.read_csv(EQ, parse_dates=['date']).set_index('date')
    E = E.reindex(M.index).ffill()
    T = pd.read_csv(TR, parse_dates=['entry_date', 'exit_date'])
    T['year'] = T.entry_date.dt.year
    T['gross_pnl'] = (T.exit - T.entry) * T.shares
    T['fees'] = T.gross_pnl - T.pnl
    T['gross_ret'] = T.gross_pnl / (T.entry * T.shares)
    return P, M, E, T


def annual_table(M, E, T):
    rows = []
    for y in range(2018, 2027):
        ee = E.loc[str(y)]
        if len(ee) == 0:
            continue
        prev = E.loc[:f'{y-1}-12-31'].equity
        start_eq = ee.equity.dropna().iloc[0] if y == 2018 else prev.iloc[-1]
        mm = M.loc[str(y)]
        on = (mm.mb > 0.5).values
        cross = on & ~np.r_[False, on[:-1]]
        tt = T[T.year == y]
        rows.append({
            '年份': y,
            '策略收益': ee.equity.iloc[-1] / start_eq - 1,
            '市场收益': mm.mkt.iloc[-1] / mm.mkt.iloc[0] - 1,
            '交易数': len(tt),
            '胜率': (tt.net_ret > 0).mean() if len(tt) else np.nan,
            '净均值': tt.net_ret.mean() if len(tt) else np.nan,
            '毛均值': tt.gross_ret.mean() if len(tt) else np.nan,
            '盈亏比': tt.pnl[tt.pnl > 0].sum() / max(1e-9, -tt.pnl[tt.pnl < 0].sum()) if len(tt) else np.nan,
            '总净盈亏': tt.pnl.sum(),
            '总费用': tt.fees.sum(),
            '止损数': int((tt.reason == '止损').sum()),
            '平均仓位': ee.exposure.mean(),
            '闸门开天数': int(on.sum()),
            '闸门开启次数': int(cross.sum()),
            '最好一笔': tt.net_ret.max() if len(tt) else np.nan,
            '最差一笔': tt.net_ret.min() if len(tt) else np.nan,
            '前三大盈利合计': tt.pnl.nlargest(3).sum(),
            '前三大亏损合计': tt.pnl.nsmallest(3).sum(),
            '平均持有天数': tt.hold.mean() if len(tt) else np.nan,
        })
    return pd.DataFrame(rows)


def monthly_table(M, E, years=(2018, 2023)):
    rows = []
    for y in years:
        ee = E.loc[str(y)]
        mm = M.loc[str(y)]
        for m in range(1, 13):
            e = ee[ee.index.month == m]
            x = mm[mm.index.month == m]
            if len(e) == 0 or len(x) == 0:
                continue
            rows.append({
                '年份': y, '月份': m,
                '策略收益': (1 + e.ret).prod() - 1,
                '平均仓位': e.exposure.mean(),
                '闸门开天数': int((x.mb > 0.5).sum()),
                '市场收益': x.mkt.iloc[-1] / x.mkt.iloc[0] - 1,
            })
    return pd.DataFrame(rows)


def event_table(M, E, T):
    on = (M.mb > 0.5).values
    cross = on & ~np.r_[False, on[:-1]]
    # 每个交易日前已经连续关闭的天数
    prior_off, c = [], 0
    for v in on:
        prior_off.append(c)
        c = 0 if v else c + 1
    prior_off = np.array(prior_off)

    def fwd(arr, i, h):
        return arr[min(i + h, len(arr) - 1)] / arr[i] - 1

    def dd(arr, i, h):
        s = arr[i:min(i + h + 1, len(arr))]
        return s.min() / s[0] - 1

    def metrics(i, a=None, b=None, weak_len=None):
        tdate = M.index[i]
        rec = {
            '闸门打开': tdate, '闸门宽度': M.mb.iloc[i],
            '市场高于60日线': '是' if M.mkt.iloc[i] > M.ma60.iloc[i] else '否',
            '60日线斜率正': '是' if M.slope.iloc[i] > 0 else '否',
        }
        for h in (5, 20, 60, 120):
            rec[f'指数{h}日'] = fwd(M.mkt.values, i, h)
            rec[f'策略{h}日'] = fwd(E.equity.values, i, h)
        rec['策略60日内回撤'] = dd(E.equity.values, i, 60)
        rec['指数60日内回撤'] = dd(M.mkt.values, i, 60)
        tt = T[(T.entry_date > tdate) & (T.entry_date <= tdate + pd.Timedelta(days=90))].sort_values('entry_date')
        rec['90日交易数'] = len(tt)
        rec['首笔间隔天数'] = (tt.entry_date.min() - tdate).days if len(tt) else np.nan
        rec['首笔净收益'] = tt.net_ret.iloc[0] if len(tt) else np.nan
        rec['交易均值'] = tt.net_ret.mean() if len(tt) else np.nan
        rec['交易胜率'] = (tt.net_ret > 0).mean() if len(tt) else np.nan
        if a is not None:
            sub = M.mkt.iloc[a:i + 1]
            tr = sub.idxmin()
            rec.update({
                '弱势开始': M.index[a], '弱势结束': M.index[b], '弱势天数': weak_len,
                '最低宽度': M.mb.iloc[a:b + 1].min(),
                '等待天数': i - b,
                '低点到闸门天数': i - M.index.get_loc(tr),
                '低点到闸门收益': M.mkt.iloc[i] / sub.min() - 1,
                '弱势开始到闸门收益': M.mkt.iloc[i] / M.mkt.iloc[a] - 1,
                '早买指数60日': fwd(M.mkt.values, a, 60),
                '早买指数60日回撤': dd(M.mkt.values, a, 60),
            })
        return rec

    # 1) 弱势区（>=5日）后的第一个闸门
    w = M.weak.values
    runs, i = [], 0
    while i < len(w):
        if w[i]:
            j = i
            while j + 1 < len(w) and w[j + 1]:
                j += 1
            if j - i + 1 >= 5:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1

    candidates = []
    for a, b in runs:
        c = np.where(cross & (np.arange(len(on)) > b))[0]
        if len(c):
            candidates.append(metrics(c[0], a=a, b=b, weak_len=b - a + 1))
    W = pd.DataFrame(candidates)
    W = W.sort_values(['闸门打开', '最低宽度']).drop_duplicates('闸门打开', keep='first')
    W = W.sort_values('闸门打开').reset_index(drop=True)

    # 2) 全部闸门重启（关闭 >=20 个交易日后重新 >50%），用于稳健性
    all_rows = [metrics(int(i)) for i in np.where(cross & (prior_off >= 20))[0]]
    A = pd.DataFrame(all_rows)
    return W, A


def stats(s):
    s = pd.Series(s).dropna()
    return {
        '样本数': len(s), '均值': s.mean(), '中位数': s.median(), '胜率': (s > 0).mean(),
        '10分位': s.quantile(.1), '90分位': s.quantile(.9), '最差': s.min(), '最好': s.max(),
    }


def bootstrap_years(E, sims=3000):
    r = E.ret.loc[START:].dropna().values
    rng = np.random.default_rng(20261023)
    block, n, years = 10, 252, 9
    neg, ann = [], []
    for _ in range(sims):
        out = []
        while len(out) < years * n:
            i = rng.integers(0, len(r) - block)
            out.extend(r[i:i + block])
        ar = (1 + np.array(out[:years * n]).reshape(years, n)).prod(axis=1) - 1
        ann.append(ar)
        neg.append((ar < 0).sum())
    neg, ann = np.array(neg), np.array(ann)
    rows = [
        ('单年负收益概率', (ann < 0).mean()),
        ('单年收益5分位', np.quantile(ann, .05)),
        ('单年收益中位数', np.median(ann)),
        ('单年收益95分位', np.quantile(ann, .95)),
        ('9年负收益年数均值', neg.mean()),
        ('9年出现0个负年概率', (neg == 0).mean()),
        ('9年出现1个负年概率', (neg == 1).mean()),
        ('9年出现2个负年概率', (neg == 2).mean()),
        ('9年出现>=2个负年概率', (neg >= 2).mean()),
        ('9年出现>=3个负年概率', (neg >= 3).mean()),
    ]
    return pd.DataFrame(rows, columns=['指标', '数值'])


def make_chart(annual, monthly23, W):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    ax = axes[0, 0]
    x = np.arange(len(annual))
    colors = ['#c0392b' if v < 0 else '#2e6da4' for v in annual['策略收益']]
    ax.bar(x, annual['策略收益'], color=colors, alpha=.85, label='70/15 基准')
    ax.plot(x, annual['市场收益'], color='#e67e22', marker='o', lw=1.5, ms=4, label='等权全A')
    ax.axhline(0, color='black', lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(annual['年份'], rotation=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for i, v in enumerate(annual['策略收益']):
        if v < 0:
            ax.annotate(f'{v:.1%}', (i, v), textcoords='offset points', xytext=(0, -14),
                        ha='center', fontsize=8, color='#c0392b')
    ax.set_title('① 年度收益：2018、2023 为负')
    ax.legend(fontsize=8, loc='upper left')

    ax = axes[0, 1]
    m = monthly23
    cols = ['#c0392b' if v < 0 else '#2e6da4' for v in m['策略收益']]
    ax.bar(m['月份'], m['策略收益'], color=cols, alpha=.85)
    ax.axhline(0, color='black', lw=.8)
    ax.set_xticks(range(1, 13))
    ax.set_ylim(-0.105, 0.09)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.annotate('11月最后两笔\n亏 6,579 元（权益 -8.5%）', xy=(12, m.loc[m['月份'] == 12, '策略收益'].iloc[0]),
                xytext=(8.2, -0.082), fontsize=8, color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=.9))
    ax.set_title('② 2023：前 11 月 +3.0%，最后一个月 -6.8%')

    ax = axes[1, 0]
    labels = ['早期买入\n(弱势开始)', '等闸门\n(指数)', '等闸门\n(策略)']
    rets = [W['早买指数60日'].mean(), W['指数60日'].mean(), W['策略60日'].mean()]
    dds = [W['早买指数60日回撤'].mean(), W['指数60日内回撤'].mean(), W['策略60日内回撤'].mean()]
    xx = np.arange(3)
    ax.bar(xx - .18, rets, width=.34, color='#2e6da4', label='60日收益均值')
    ax.bar(xx + .18, dds, width=.34, color='#c0392b', label='60日内最大回撤均值')
    ax.axhline(0, color='black', lw=.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(-0.13, 0.06)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for i, v in enumerate(rets):
        ax.annotate(f'{v:.1%}', (i - .18, v), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8)
    for i, v in enumerate(dds):
        ax.annotate(f'{v:.1%}', (i + .18, v), textcoords='offset points', xytext=(0, -12), ha='center', fontsize=8)
    ax.set_title('③ 提前买 vs 等闸门：等闸门收益更高、回撤更浅')
    ax.legend(fontsize=8, loc='upper right')

    ax = axes[1, 1]
    x = np.arange(len(W))
    ax.bar(x - .2, W['策略60日'], width=.4, color='#2e6da4', label='70/15 策略')
    ax.bar(x + .2, W['指数60日'], width=.4, color='#e67e22', alpha=.8, label='等权全A')
    ax.axhline(0, color='black', lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(W['闸门打开'].dt.strftime('%Y-%m'), rotation=45, ha='right', fontsize=7.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f'④ 闸门重启后 60 日：策略 {int((W["策略60日"] > 0).sum())}/{len(W)} 为正')
    ax.legend(fontsize=8, loc='upper left')

    fig.suptitle('70/15：两个负收益年，与“弱势区 → 闸门重启”的历史样本', fontsize=15, y=.995)
    plt.tight_layout()
    plt.savefig('13_负年份与闸门重启.png', dpi=160)
    plt.close()


def main():
    P, M, E, T = load_data()
    annual = annual_table(M, E, T)
    annual.to_csv('13_年度归因.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    print('年度归因')
    print(annual.round(4).to_string(index=False))

    neg = T[T.year.isin([2018, 2023])].sort_values(['year', 'entry_date']).copy()
    neg_out = neg[['year', 'entry_date', 'exit_date', 'code', 'net_ret', 'gross_ret', 'fees', 'pnl', 'reason', 'hold']].copy()
    neg_out.columns = ['年份', '买入日', '卖出日', '代码', '净收益率', '毛收益率', '费用', '净盈亏', '退出原因', '持有天数']
    neg_out.to_csv('13_负年份交易.csv', index=False, encoding='utf-8-sig', float_format='%.4f')

    monthly = monthly_table(M, E)
    monthly.to_csv('13_负年份月度.csv', index=False, encoding='utf-8-sig', float_format='%.4f')

    W, A = event_table(M, E, T)
    W.to_csv('13_闸门重启事件.csv', index=False, encoding='utf-8-sig', float_format='%.4f', date_format='%Y-%m-%d')
    metrics = ['指数20日', '策略20日', '指数60日', '策略60日', '指数120日', '策略120日',
               '策略60日内回撤', '指数60日内回撤', '首笔净收益', '交易均值', '交易胜率',
               '等待天数', '低点到闸门收益', '弱势开始到闸门收益',
               '早买指数60日', '早买指数60日回撤']
    rows = []
    for label, D in [('全部重启(26)', A), ('弱势后重启(16)', W)]:
        for col in metrics:
            if col not in D.columns:
                continue
            st = stats(D[col])
            rows.append({'口径': label, '指标': col, **st})
    summary = pd.DataFrame(rows)
    summary.to_csv('13_闸门重启_汇总.csv', index=False, encoding='utf-8-sig', float_format='%.4f')

    boot = bootstrap_years(E)
    boot.to_csv('13_年度自举.csv', index=False, encoding='utf-8-sig', float_format='%.4f')

    make_chart(annual, monthly[monthly['年份'] == 2023], W)

    print('\n弱势后重启：', len(W), '个闸门日')
    print(W[['弱势开始', '闸门打开', '等待天数', '最低宽度', '低点到闸门收益',
             '指数60日', '策略60日', '策略120日']].round(4).to_string(index=False))
    print('\n汇总（弱后重启）')
    print(summary[summary['口径'] == '弱势后重启(16)'].round(4).to_string(index=False))
    print('\n自举')
    print(boot.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
