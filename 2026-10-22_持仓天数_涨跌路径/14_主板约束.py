# -*- coding: utf-8 -*-
"""账户只能买主板：原 70/15 回测的板块贡献，以及主板口径重跑。

输出：
  14_板块贡献.csv
  14_主板_年度.csv
  14_主板_对照.csv
  14_主板_权益.csv / 14_主板_逐笔.csv
  14_主板超跌_权益.csv / 14_主板超跌_逐笔.csv
  14_主板超跌35_权益.csv / 14_主板超跌35_逐笔.csv
  14_主板约束.png

主板定义（按当前券商常用口径）：
  沪主板 600/601/603/605；深主板 000/001/002/003。
  排除创业板 300/301、科创板 688/689、北交所 8/4/920。
  002 原中小板，2021 年并入深主板，本处按主板处理。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from importlib.machinery import SourceFileLoader

BASE = '/Users/admin/WorkSpace/trading'
ENG = os.path.join(BASE, '2026-10-22_全信号_买卖时间_优先级研究', '108_稳健引擎.py')
WEAK = os.path.join(BASE, '2026-10-22_持仓天数_涨跌路径', '07_弱势引擎.py')
PANEL = os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl')
EQ0 = '11_base_proper_权益.csv'
TR0 = '11_base_proper_逐笔.csv'
START, END = '2018-01-01', '2026-10-13'

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

eng = SourceFileLoader('eng', ENG).load_module()
weak = SourceFileLoader('weak', WEAK).load_module()


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


def is_main(c):
    return board_of(c) in ('沪主板', '深主板')


def main_mask(codes):
    return np.array([is_main(c) for c in codes], dtype=bool)[None, :]


class MainBase(eng.Engine):
    def __init__(self):
        super().__init__()
        self.base_nogate = (self.base_nogate & (self.AMT <= 70e6) & (self.C <= 15)
                            & main_mask(self.codes))


class MainEnhanced(weak.Engine):
    def __init__(self):
        super().__init__()
        g = lambda k: self.P[k].values
        self.MB0 = g('market_breadth20').copy()
        self.mask = main_mask(self.codes)
        dates = self.dates
        self.valid = (dates >= pd.Timestamp('2017-10-13'))[:, None] & (self.AMT >= 3e7) & (self.C >= 2) & self.mask
        self.trend_t = (self.base_nogate & (self.AMT <= 70e6) & (self.C <= 15)
                        & (self.MB0 > 0.5)[:, None] & self.mask)
        self.ov = (self.MB0 < .10)[:, None] & (self.RSI < 30) & (self.ATR <= .06) & (self.AMT >= 1e8) & self.mask
        self.MB = np.ones_like(self.MB0)
        self.base_nogate = self.trend_t | (self.valid & self.ov)


def annual_returns(E):
    out, prev = {}, E.equity.iloc[0]
    for y in range(2018, 2027):
        x = E.equity.loc[str(y)]
        if len(x) == 0:
            out[f'{y}年'] = np.nan
            continue
        out[f'{y}年'] = x.iloc[-1] / prev - 1
        prev = x.iloc[-1]
    return out


def add_net(T):
    T = T.copy()
    T['net_ret'] = T.pnl / (T.shares * T.entry)
    T['year'] = T.entry_date.dt.year
    return T


def bootstrap_dd(E, n, sims=3000, block=10):
    r = E.ret.dropna().values
    rng = np.random.default_rng(20261023)
    ds = []
    for _ in range(sims):
        out = []
        while len(out) < n:
            i = rng.integers(0, len(r) - block)
            out.extend(r[i:i + block])
        x = np.array(out[:n])
        eq = np.cumprod(1 + x)
        ds.append((eq / np.maximum.accumulate(eq) - 1).min())
    return np.array(ds)


def save_curve(name, E, T):
    E = E.copy()
    E.index.name = 'date'
    E.to_csv(f'14_{name}_权益.csv', encoding='utf-8-sig', float_format='%.4f', date_format='%Y-%m-%d')
    T.to_csv(f'14_{name}_逐笔.csv', index=False, encoding='utf-8-sig', float_format='%.6f', date_format='%Y-%m-%d')


def main():
    # 1) 原基线逐笔的板块贡献
    T0 = pd.read_csv(TR0, parse_dates=['entry_date', 'exit_date'])
    T0['板块'] = T0.code.map(board_of)
    total_pnl = T0.pnl.sum()
    rows = []
    for b, g in T0.groupby('板块'):
        rows.append({
            '板块': b, '笔数': len(g), '笔数占比': len(g) / len(T0),
            '净盈亏': g.pnl.sum(), '净盈亏占比': g.pnl.sum() / total_pnl,
            '胜率': (g.net_ret > 0).mean(), '净均值': g.net_ret.mean(),
            '最大单笔盈利': g.pnl.max(), '最大单笔亏损': g.pnl.min(),
        })
    contrib = pd.DataFrame(rows).sort_values('净盈亏', ascending=False)
    contrib.to_csv('14_板块贡献.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    print('板块贡献')
    print(contrib.round(4).to_string(index=False))

    # 2) 主板重跑
    b = MainBase()
    E1, T1, m1 = b.run(START, END, slots=2, sizing='fixed', maxpos=.45, heat_cap=.10,
                       gate_thr=.50, gate_mode='mb', rank='amt', hold=20, cost_rate=.0015,
                       vol_target=.20, stop_pct=.20, skip_entry_lu=True)
    T1 = add_net(T1)
    save_curve('主板', E1, T1)

    e = MainEnhanced()
    E2, T2, m2 = e.run(START, END, slots=2, sizing='fixed', maxpos=.45, heat_cap=.10,
                       gate_thr=-1, rank='amt', hold=20, cost_rate=.0015,
                       vol_target=.20, stop_pct=.20, skip_entry_lu=True)
    T2 = add_net(T2)
    save_curve('主板超跌', E2, T2)

    e35 = MainEnhanced()
    E3, T3, m3 = e35.run(START, END, slots=2, sizing='fixed', maxpos=.35, heat_cap=.10,
                         gate_thr=-1, rank='amt', hold=20, cost_rate=.0015,
                         vol_target=.20, stop_pct=.20, skip_entry_lu=True)
    T3 = add_net(T3)
    save_curve('主板超跌35', E3, T3)

    # 3) 对照表 + 年度
    E0 = pd.read_csv(EQ0, parse_dates=['date']).set_index('date')
    variants = [
        ('全板块 70/15（原基准）', E0, T0, None),
        ('主板 70/15', E1, T1, m1),
        ('主板 70/15+超跌 2×45%', E2, T2, m2),
        ('主板 70/15+超跌 2×35%', E3, T3, m3),
    ]
    rows, annual_rows = [], []
    for label, E, T, m in variants:
        ar = annual_returns(E)
        if m is None:
            m = {
                'cagr': (E.equity.iloc[-1] / E.equity.iloc[0]) ** (252 / len(E)) - 1,
                'maxdd': (E.equity / E.equity.cummax() - 1).min(),
                'maxdd_low': (E.equity_low / E.equity.cummax() - 1).min(),
                'trades': len(T), 'win': (T.net_ret > 0).mean(),
                'pf': T.pnl[T.pnl > 0].sum() / max(1e-9, -T.pnl[T.pnl < 0].sum()),
                'avg_exposure': E.exposure.mean(),
            }
        dd1 = bootstrap_dd(E, 252)
        dd3 = bootstrap_dd(E, 756)
        row = {
            '口径': label, '年化': m['cagr'], '最大回撤': m['maxdd'],
            '盘中最大回撤': m['maxdd_low'], '交易数': m['trades'],
            '胜率': m['win'], '盈亏比': m['pf'], '平均仓位': m['avg_exposure'],
            '1年自举P5回撤': np.quantile(dd1, .05),
            '3年自举P5回撤': np.quantile(dd3, .05),
        }
        row.update(ar)
        rows.append(row)
        for k, v in ar.items():
            annual_rows.append({'口径': label, '年份': int(k[:4]), '收益': v})
    comp = pd.DataFrame(rows)
    comp.to_csv('14_主板_对照.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    pd.DataFrame(annual_rows).to_csv('14_主板_年度.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    print('\n主板对照')
    print(comp.round(4).to_string(index=False))

    # 4) 图：净值 + 年度
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    for label, E, _, _ in variants:
        ax.plot(E.index, E.equity / E.equity.iloc[0], label=label, lw=1.4)
    ax.set_yscale('log')
    ax.set_title('① 净值：全板块 vs 主板口径')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylabel('净值（对数轴）')

    ax = axes[1]
    x = np.arange(9)
    years = list(range(2018, 2027))
    for i, (label, _, _, _) in enumerate(variants):
        vals = [comp.loc[comp['口径'] == label, f'{y}年'].iloc[0] for y in years]
        ax.bar(x + (i - 1.5) * .2, vals, width=.2, label=label)
    ax.axhline(0, color='black', lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title('② 年度收益：主板口径的负年更多')
    ax.legend(fontsize=7, loc='upper left')

    fig.suptitle('账户只能买主板：原 30.3% 不能直接套用', fontsize=14)
    plt.tight_layout()
    plt.savefig('14_主板约束.png', dpi=160)
    plt.close()

    print('\n各口径逐年')
    print(comp[['口径'] + [f'{y}年' for y in years]].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
