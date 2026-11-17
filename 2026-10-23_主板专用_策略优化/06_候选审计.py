# -*- coding: utf-8 -*-
"""主板非ST：首选候选 C1 的审计。

C1 = 主板非ST，宽度>50%，年龄9-25，ATR<=4.4%，20日无涨停，
     行业宽度>=60%，行业20日>5%，DD60>=-15%，成交额3000-7000万，股价<=15，
     2槽45%，20日波动>=20%，10日持有，涨25%卖一半，-20%止损。

输出：权益/逐笔、年度/月度/板块/退出分解、自举回撤、随机排序零分布、
成本与滑点、ST残留、剔除前5笔的粗略年化。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import mb_lib as L

OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'
C1 = dict(name='C1', st_mode='x', age_min=9, age_max=25, vol20_min=0.20,
          dd60_min=-0.15, hold=10, partial_trigger=0.25, partial_frac=0.5, rank='amt')

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def annual_pnl(T):
    rows = []
    for y, g in T.groupby('year'):
        top = g.nlargest(1, 'pnl').pnl.iloc[0] if len(g) else np.nan
        top5 = g.nlargest(5, 'pnl').pnl.sum() if len(g) else np.nan
        rows.append({'年份': int(y), '笔数': len(g), '净盈亏': g.pnl.sum(),
                     '胜率': (g.net_ret > 0).mean(), '净均值': g.net_ret.mean(),
                     '最大单笔盈利': g.pnl.max(), '最大单笔亏损': g.pnl.min(),
                     '第一笔占比': top / g.pnl.sum() if abs(g.pnl.sum()) > 1e-9 else np.nan,
                     '前5笔占比': top5 / g.pnl.sum() if abs(g.pnl.sum()) > 1e-9 else np.nan})
    return pd.DataFrame(rows)


def main():
    e = L.fresh_engine()
    E, T, r = L.run_one(e, C1)
    E.index.name = 'date'
    E.to_csv(os.path.join(OUT, '06_C1_权益.csv'), encoding='utf-8-sig',
             float_format='%.4f', date_format='%Y-%m-%d')
    T.to_csv(os.path.join(OUT, '06_C1_逐笔.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f', date_format='%Y-%m-%d')
    print('C1 汇总')
    print(L.summarize(r).to_string(index=False))

    ap = annual_pnl(T)
    ap.to_csv(os.path.join(OUT, '06_C1_年度分解.csv'), index=False, encoding='utf-8-sig',
              float_format='%.4f')
    print('\n年度分解')
    print(ap.round(4).to_string(index=False))

    # 月度（按买入月）
    T['月'] = T.entry_date.dt.to_period('M').astype(str)
    m = T.groupby('月').agg(笔数=('code', 'size'), 净盈亏=('pnl', 'sum'),
                            净均值=('net_ret', 'mean'))
    m.to_csv(os.path.join(OUT, '06_C1_月度.csv'), encoding='utf-8-sig', float_format='%.4f')
    print('\n最差10个月')
    print(m.sort_values('净盈亏').head(10).round(4).to_string())
    print('\n最好10个月')
    print(m.sort_values('净盈亏', ascending=False).head(10).round(4).to_string())

    # 板块/退出
    b = T.groupby('board').agg(笔数=('code', 'size'), 净盈亏=('pnl', 'sum'),
                               胜率=('net_ret', lambda x: (x > 0).mean()),
                               净均值=('net_ret', 'mean'))
    b.to_csv(os.path.join(OUT, '06_C1_板块.csv'), encoding='utf-8-sig', float_format='%.4f')
    print('\n板块')
    print(b.round(4).to_string())
    print('\n退出原因')
    print(T.groupby('reason').agg(笔数=('code', 'size'), 净盈亏=('pnl', 'sum'),
                                  净均值=('net_ret', 'mean')).round(4).to_string())

    # ST 残留
    rows = []
    for _, x in T.iterrows():
        j = e.code_idx[x.code]; i = e.dates.searchsorted(x.entry_date)
        rows.append(dict(code=x.code, entry=x.entry_date, c5=int(e.C5[i, j]),
                         maxabs60=float(e.MAXABS60[i, j]), price=float(e.C[i, j]),
                         amt=float(e.AMT[i, j]) / 1e6, pnl=x.pnl))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, '06_C1_ST残留.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f', date_format='%Y-%m-%d')
    print('\nST残留 c5>=1:', int((D.c5 >= 1).sum()), 'c5>=2:', int((D.c5 >= 2).sum()),
          'c5>=1盈亏:', round(D[D.c5 >= 1].pnl.sum(), 0))

    # 自举回撤
    for n, label in [(252, '1年'), (756, '3年')]:
        dd = L.bootstrap_dd(E, n, sims=3000)
        print(f'{label}自举回撤 P5/P50/P95',
              round(np.quantile(dd, .05), 4), round(np.quantile(dd, .50), 4),
              round(np.quantile(dd, .95), 4))

    # 剔除前5笔的粗略年化
    total = T.pnl.sum()
    top5 = T.nlargest(5, 'pnl').pnl.sum()
    years = len(E) / 252
    print('\n总盈亏', round(total, 0), '前5笔', round(top5, 0),
          '剔除前5笔粗略年化', round(((20000 + total - top5) / 20000) ** (1 / years) - 1, 4))

    # 成本/滑点
    cost_rows = []
    for cr in [0.0015, 0.0020, 0.0025, 0.0030, 0.0040]:
        c = dict(C1); c['name'] = f'单边{cr:.2%}'; c['cost_rate'] = cr
        _, _, rr = L.run_one(e, c)
        cost_rows.append(rr)
    C = pd.DataFrame(cost_rows)
    C.to_csv(os.path.join(OUT, '06_C1_成本.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f')
    print('\n成本敏感性')
    print(C[['cfg', '全期_cagr', '全期_maxdd', '全期_win', '全期_pf', '检验_cagr']].round(4).to_string(index=False))

    # 随机排序零分布
    rand_rows = []
    for seed in range(30):
        c = dict(C1); c['name'] = f'随机{seed}'; c['rank'] = 'random'; c['rank_seed'] = seed
        _, _, rr = L.run_one(e, c)
        rand_rows.append(rr)
    Rd = pd.DataFrame(rand_rows)
    Rd.to_csv(os.path.join(OUT, '06_C1_随机排序.csv'), index=False, encoding='utf-8-sig',
              float_format='%.4f')
    print('\n随机排序 30 次')
    print(Rd['全期_cagr'].describe(percentiles=[.05, .25, .5, .75, .95]).round(4).to_string())
    print('C1 在随机分布中的分位', round((Rd['全期_cagr'] < r['全期_cagr']).mean(), 3))

    # 图：净值/回撤 + 年度
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    eq = E.equity / E.equity.iloc[0]
    ax.plot(E.index, eq, color='#1f4e79', lw=1.4, label='C1 净值')
    ax.plot(E.index, E.equity.cummax() / E.equity.iloc[0], color='#999999', lw=.8, ls='--')
    ax.set_yscale('log')
    ax.set_title('① C1 净值（对数轴）')
    ax.legend()
    ax2 = ax.twinx()
    ax2.fill_between(E.index, (E.equity / E.equity.cummax() - 1), 0,
                     color='#c0504d', alpha=.18)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_ylim(-0.5, 0)

    ax = axes[1]
    years = ap['年份'].tolist()
    vals = ap['净盈亏'].tolist()
    colors = ['#c0504d' if v < 0 else '#4f81bd' for v in vals]
    ax.bar([str(y) for y in years], vals, color=colors)
    ax.axhline(0, color='black', lw=.8)
    ax.set_title('② 年度净盈亏（元，2万本金）')
    ax.set_ylabel('净盈亏（元）')
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v/1000:.0f}k', ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)
    fig.suptitle('C1：主板非ST，9–25日年龄，10日持有，涨25%卖一半', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '06_C1_审计.png'), dpi=160)
    plt.close()


if __name__ == '__main__':
    main()
