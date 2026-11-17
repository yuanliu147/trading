# -*- coding: utf-8 -*-
"""主板非ST：最终候选 F2 的完整审计。

F2 = 主板非ST；宽度>50%；年龄 10–30；ATR<=4.4%；20日无涨停；
     行业宽度>=60%；行业20日>5%；DD60>=-15%；成交额 3000–7000 万；股价<=15；
     20日波动>=20%；按市值升序选；2 槽 45%；20日持有；涨25%卖一半；
     盘中最高浮盈>=10% 后止损上移到成本；-20% 灾难止损。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import mb_lib as L
from mb_engine import Engine as LocalEngine

L.eng.Engine = LocalEngine
OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'
F2 = dict(name='F2', st_mode='x', age_min=10, age_max=30, vol20_min=0.20,
          dd60_min=-0.15, hold=20, partial_trigger=0.25, partial_frac=0.5,
          profit_lock=True, lock_trigger=0.10, lock_gain=0.0, trail_trigger=0.25,
          trail_pct=0.10, rank='mcap')

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def annual(T):
    rows = []
    for y, g in T.groupby('year'):
        u = g.groupby(['code', 'entry_date']).agg(pnl=('pnl', 'sum'),
                                                   shares=('shares', 'sum'),
                                                   entry=('entry', 'first')).reset_index()
        u['ret'] = u.pnl / (u.shares * u.entry)
        rows.append({'年份': int(y), '笔数': len(g), '独立仓位': len(u),
                     '净盈亏': g.pnl.sum(), '胜率': (g.net_ret > 0).mean(),
                     '仓位胜率': (u.ret > 0).mean(), '净均值': g.net_ret.mean(),
                     '最大单笔盈利': g.pnl.max(), '最大单笔亏损': g.pnl.min()})
    return pd.DataFrame(rows)


def main():
    e = L.fresh_engine()
    E, T, r = L.run_one(e, F2)
    E.index.name = 'date'
    E.to_csv(os.path.join(OUT, '13_F2_权益.csv'), encoding='utf-8-sig',
             float_format='%.4f', date_format='%Y-%m-%d')
    T.to_csv(os.path.join(OUT, '13_F2_逐笔.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f', date_format='%Y-%m-%d')
    print('F2 汇总')
    print(L.summarize(r).to_string(index=False))
    ap = annual(T)
    ap.to_csv(os.path.join(OUT, '13_F2_年度.csv'), index=False, encoding='utf-8-sig',
              float_format='%.4f')
    print('\n年度')
    print(ap.round(4).to_string(index=False))

    # 月度
    T['月'] = T.entry_date.dt.to_period('M').astype(str)
    m = T.groupby('月').agg(笔数=('code', 'size'), 净盈亏=('pnl', 'sum'),
                            净均值=('net_ret', 'mean'))
    m.to_csv(os.path.join(OUT, '13_F2_月度.csv'), encoding='utf-8-sig', float_format='%.4f')
    print('\n最差8个月')
    print(m.sort_values('净盈亏').head(8).round(4).to_string())

    # 板块/退出/ST/持仓特征
    print('\n板块')
    print(T.groupby('board').agg(笔数=('code', 'size'), 净盈亏=('pnl', 'sum'),
                                  胜率=('net_ret', lambda x: (x > 0).mean()),
                                  净均值=('net_ret', 'mean')).round(4).to_string())
    print('\n退出原因')
    print(T.groupby('reason').agg(笔数=('code', 'size'), 净盈亏=('pnl', 'sum'),
                                  净均值=('net_ret', 'mean')).round(4).to_string())
    rows = []
    for _, x in T.iterrows():
        j = e.code_idx[x.code]; i = e.dates.searchsorted(x.entry_date)
        rows.append(dict(code=x.code, entry=x.entry_date, c5=int(e.C5[i, j]),
                         maxabs60=float(e.MAXABS60[i, j]), price=float(e.C[i, j]),
                         amt=float(e.AMT[i, j]) / 1e6, mcap=float(e.MCAP[i, j]) / 1e8,
                         pnl=x.pnl))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, '13_F2_持仓特征.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f', date_format='%Y-%m-%d')
    print('\n持仓特征：成交额中位 %.1f 百万，股价中位 %.2f，市值中位 %.1f 亿，最小 %.1f 亿'
          % (D.amt.median(), D.price.median(), D.mcap.median(), D.mcap.min()))
    print('ST残留 c5>=1:', int((D.c5 >= 1).sum()), 'c5>=2:', int((D.c5 >= 2).sum()),
          'c5>=1盈亏:', round(D[D.c5 >= 1].pnl.sum(), 0))
    print('数据终止笔数:', int((T.reason == '数据终止').sum()))

    # 自举
    for n, label in [(252, '1年'), (756, '3年')]:
        dd = L.bootstrap_dd(E, n, sims=3000)
        print(f'{label}自举回撤 P5/P50/P95',
              round(np.quantile(dd, .05), 4), round(np.quantile(dd, .50), 4),
              round(np.quantile(dd, .95), 4))

    # 成本/ST
    audit = [r]
    for cr in [0.0020, 0.0030, 0.0040, 0.0050]:
        c = dict(F2); c['name'] = f'成本{cr:.2%}'; c['cost_rate'] = cr
        _, _, rr = L.run_one(e, c); audit.append(rr)
    for st in ['one', 'two', 'y']:
        c = dict(F2); c['name'] = f'ST={st}'; c['st_mode'] = st
        _, _, rr = L.run_one(e, c); audit.append(rr)
    A = pd.DataFrame(audit)
    A.to_csv(os.path.join(OUT, '13_F2_压力.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f')
    print('\n成本/ST压力')
    print(A[['cfg', '全期_cagr', '全期_maxdd', '全期_trades', '全期_win', '全期_pf',
             '检验_cagr']].round(4).to_string(index=False))

    # 去前5笔粗略
    total, top5 = T.pnl.sum(), T.nlargest(5, 'pnl').pnl.sum()
    years = len(E) / 252
    print('\n总盈亏', round(total), '前5笔', round(top5),
          '剔除前5笔粗略年化', round(((20000 + total - top5) / 20000) ** (1 / years) - 1, 4))

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    ax.plot(E.index, E.equity / E.equity.iloc[0], color='#1f4e79', lw=1.5)
    ax.set_yscale('log')
    ax.set_title('① F2 净值（对数轴）')
    ax2 = ax.twinx()
    ax2.fill_between(E.index, E.equity / E.equity.cummax() - 1, 0,
                     color='#c0504d', alpha=.18)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_ylim(-0.5, 0)
    ax = axes[1]
    vals = ap['净盈亏'].tolist()
    ax.bar(ap['年份'].astype(str), vals,
           color=['#c0504d' if v < 0 else '#4f81bd' for v in vals])
    ax.axhline(0, color='black', lw=.8)
    ax.set_title('② 年度净盈亏（元，2万本金）')
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v/1000:.0f}k', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=7)
    fig.suptitle('F2：主板非ST，20日持有，涨25%卖一半，浮盈10%保本/25%后10%移动止盈', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '13_F2_审计.png'), dpi=160)
    plt.close()


if __name__ == '__main__':
    main()
