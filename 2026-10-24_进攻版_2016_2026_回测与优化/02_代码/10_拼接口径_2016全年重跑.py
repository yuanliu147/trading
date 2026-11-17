# -*- coding: utf-8 -*-
"""拼接面板（v3）上的"真 2016 全年"口径重跑。

  [P0] 一致性对照：v3 与 v2 在共同窗口 2017-03→2026-10 的进攻版逐笔差异计数
       （差异应集中在 2017 年初：v3 的趋势年龄从 2015 年起算，更准确）；
  [P1] 进攻版 2016-01-04→2026-10-23（含 2016 全年）；
  [P2] V2 混合版 同窗口；
  [P3] 核心版 同窗口；
  [P4] 进攻版/V2 的 ±1 天抖动（10 种子）与 0.35% 成本。
输出：03_结果/拼接_*.csv
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bt2016 as B  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, '03_结果')
PANEL_V3 = os.path.join(ROOT, '01_数据', '全A面板_2015_2026_拼接.pkl')
W16 = ('2016-01-04', '2026-10-23')


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False, encoding='utf-8-sig',
              float_format='%.6f')


def met(E, T, start_cash=20000.0):
    total = E.equity.iloc[-1] / start_cash - 1
    cagr = (1 + total) ** (252 / len(E)) - 1
    dd = (E.equity / E.equity.cummax() - 1).min()
    out = dict(cagr=cagr, maxdd=dd, trades=len(T),
               win=(T.net_ret > 0).mean() if len(T) else np.nan,
               pf=(T.pnl[T.pnl > 0].sum() / max(1e-9, -T.pnl[T.pnl < 0].sum())) if len(T) else np.nan)
    prev = E.equity.iloc[0]
    for y in range(2016, 2027):
        x = E.equity.loc[f'{y}-01-01':f'{y}-12-31']
        if len(x) == 0:
            continue
        out[f'{y}年'] = x.iloc[-1] / prev - 1
        prev = x.iloc[-1]
    yrs = [y for y in range(2016, 2027) if f'{y}年' in out]
    out['最差年'] = min(out[f'{y}年'] for y in yrs)
    out['负年份数'] = sum(1 for y in yrs if out[f'{y}年'] < 0)
    out['前5笔占比'] = (T.nlargest(5, 'pnl').pnl.sum() / T.pnl.sum()
                      if len(T) and abs(T.pnl.sum()) > 1e-9 else np.nan)
    return out


def main():
    e3 = B.fresh_engine(panel_path=PANEL_V3)

    # P0 一致性对照
    print('[P0] v3 vs v2 共同窗口对照...', flush=True)
    cfg_cmp = dict(B.ATTACK, name='P0', date_start='2017-03-01')
    E3c, T3c, _ = B.run_one(e3, cfg_cmp, window=('2017-03-01', '2026-10-23'))
    T2c = pd.read_csv(os.path.join(OUT, 'R1_进攻版_逐笔.csv'))
    k3 = {(str(c), str(d)[:10]) for c, d in zip(T3c.code, T3c.entry_date)}
    k2 = {(str(c), str(d)[:10]) for c, d in zip(T2c.code, T2c.entry_date)}
    only3 = k3 - k2
    only2 = k2 - k3
    print(f'  v3独有 {len(only3)} 笔 / v2独有 {len(only2)} 笔')
    by_year3 = pd.Series([d[1][:4] for d in only3]).value_counts().to_dict()
    by_year2 = pd.Series([d[1][:4] for d in only2]).value_counts().to_dict()
    print('  v3独有按年:', by_year3, ' v2独有按年:', by_year2)
    save(pd.DataFrame([dict(口径='v3独有', **by_year3), dict(口径='v2独有', **by_year2)]),
         '拼接_P0_逐笔差异按年.csv')

    # P1 进攻版 2016 全年
    print('[P1] 进攻版 2016-01-04→2026-10-23 ...', flush=True)
    cfg16 = dict(B.ATTACK, name='P1', date_start='2016-01-04')
    E1, T1, r1 = B.run_one(e3, cfg16, window=W16)
    save(pd.DataFrame([met(E1, T1)]), '拼接_P1_进攻版指标.csv')
    save(T1, '拼接_P1_进攻版逐笔.csv')
    E1.index.name = 'date'
    E1.to_csv(os.path.join(OUT, '拼接_P1_进攻版权益.csv'), encoding='utf-8-sig',
              float_format='%.4f', date_format='%Y-%m-%d')

    # P2 V2 混合版
    print('[P2] V2 混合版 ...', flush=True)
    EA, TA, _ = B.run_one(e3, dict(B.ATTACK, date_start='2016-01-04', start_cash=10000.0), window=W16)
    EB, TB, _ = B.run_one(e3, dict(B.CORE, date_start='2016-01-04', start_cash=10000.0), window=W16)
    idx = EA.index.union(EB.index)
    eq = EA.equity.reindex(idx).ffill() + EB.equity.reindex(idx).ffill()
    E2 = pd.DataFrame({'equity': eq})
    T2 = pd.concat([TA, TB], ignore_index=True)
    save(pd.DataFrame([met(E2, T2)]), '拼接_P2_V2混合版指标.csv')
    save(T2, '拼接_P2_V2混合版逐笔.csv')

    # P3 核心版
    print('[P3] 核心版 ...', flush=True)
    E3, T3, r3 = B.run_one(e3, dict(B.CORE, name='P3', date_start='2016-01-04'), window=W16)
    save(pd.DataFrame([met(E3, T3)]), '拼接_P3_核心版指标.csv')

    # P4 抖动与成本
    print('[P4] 抖动/成本 ...', flush=True)
    rows = []
    for tag, mk in [('进攻版', lambda **kw: B.run_one(e3, dict(B.ATTACK, date_start='2016-01-04', **kw), window=W16)),
                    ('V2混合', None)]:
        if mk is None:
            def mk(**kw):
                a = B.run_one(e3, dict(B.ATTACK, date_start='2016-01-04', start_cash=10000.0, **kw), window=W16)[0]
                b = B.run_one(e3, dict(B.CORE, date_start='2016-01-04', start_cash=10000.0, **kw), window=W16)[0]
                idx2 = a.index.union(b.index)
                return pd.DataFrame({'equity': a.equity.reindex(idx2).ffill() + b.equity.reindex(idx2).ffill()})
        vals = []
        for seed in range(1, 11):
            Ej = mk(hold_jitter=1, hold_seed=seed)
            if isinstance(Ej, tuple):
                Ej = Ej[0]
            vals.append((Ej.equity.iloc[-1] / 20000.0) ** (252 / len(Ej)) - 1)
        Ec = mk(cost_rate=0.0035)
        if isinstance(Ec, tuple):
            Ec = Ec[0]
        cc = (Ec.equity.iloc[-1] / 20000.0) ** (252 / len(Ec)) - 1
        rows.append(dict(变体=tag, 抖动均值年化=float(np.mean(vals)), 抖动最差=float(np.min(vals)),
                         抖动最好=float(np.max(vals)), 成本035年化=cc))
    P4 = pd.DataFrame(rows)
    save(P4, '拼接_P4_抖动成本.csv')

    print('\nP1 进攻版(2016起):', {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in met(E1, T1).items()})
    print('P2 V2混合(2016起):', {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in met(E2, T2).items()})
    print('P3 核心版(2016起):', {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in met(E3, T3).items()})
    print('\nP4:\n', P4.round(4).to_string(index=False))
    print('完成')


if __name__ == '__main__':
    main()
