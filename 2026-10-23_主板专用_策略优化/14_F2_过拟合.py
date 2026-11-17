# -*- coding: utf-8 -*-
"""F2 所属候选族的走步、PBO、DSR。

候选族（192 个）：
  年龄区间 4 种 × 排序 2 种 × 分批 4 种 × 盈利保护 3 种 × 是否延长 2 种，
  其余底座固定为 F2（非ST、vol20>=20%、DD60>=-15%、20日持有、2槽45%）。
F2 本身 = 10-30 | mcap | 分批25% | 浮盈10%保本 | 不延长。
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
import mb_lib as L
from mb_engine import Engine as LocalEngine

L.eng.Engine = LocalEngine
OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'
BASE = dict(st_mode='x', vol20_min=0.20, dd60_min=-0.15, hold=20,
            partial_frac=0.5, trail_trigger=0.25, trail_pct=0.10)


def make_cases():
    cases = []
    for amin, amax in [(9, 25), (10, 25), (10, 30), (9, 30)]:
        for rank in ['amt', 'mcap']:
            for pt in [None, 0.20, 0.25, 0.30]:
                for lock in [None, 0.10, 0.15]:
                    for ext in [False, True]:
                        name = f'{amin}-{amax}|{rank}|{pt}|{lock}|{ext}'
                        cfg = dict(BASE, age_min=amin, age_max=amax, rank=rank)
                        kw = {}
                        if pt is not None:
                            cfg.update(partial_trigger=pt)
                        if lock is not None:
                            cfg.update(profit_lock=True, lock_trigger=lock, lock_gain=0.0)
                        if ext:
                            kw.update(extend_if_gain=0.03, extend_days=5, extend_max=25)
                        cases.append((name, cfg, kw))
    return cases


def perf(r):
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return np.nan, np.nan
    eq = np.cumprod(1 + r)
    return eq[-1] ** (252 / len(r)) - 1, (eq / np.maximum.accumulate(eq) - 1).min()


def sharpe(r):
    r = np.asarray(r, dtype=float)
    return r.mean() / (r.std() + 1e-12) * np.sqrt(252)


def main():
    e = L.fresh_engine()
    cases = make_cases()
    cols, rows = {}, []
    for i, (name, cfg, kw) in enumerate(cases):
        c = dict(cfg); c['name'] = name
        E, T, r = L.run_one(e, c, run_kw=kw, return_daily=True)
        cols[name] = E.ret
        r['name'] = name
        rows.append(r)
        if (i + 1) % 40 == 0:
            print('run', i + 1, '/', len(cases), flush=True)
    R = pd.DataFrame(cols).sort_index()
    R.to_csv(os.path.join(OUT, '14_F2族_日收益.csv'), encoding='utf-8-sig',
             float_format='%.6f', date_format='%Y-%m-%d')
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(OUT, '14_F2族_汇总.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f')

    # 走步
    picks, wf = [], []
    for ty in range(2021, 2027):
        tr0, tr1 = f'{ty-3}-01-01', f'{ty-1}-12-31'
        te0, te1 = f'{ty}-01-01', f'{ty}-12-31'
        scores = []
        for name in R.columns:
            cagr, dd = perf(R[name].loc[tr0:tr1].dropna())
            scores.append((name, cagr, dd, cagr / (abs(dd) + 1e-9)))
        S = pd.DataFrame(scores, columns=['name', 'cagr', 'dd', 'calmar'])
        p = S.sort_values('calmar', ascending=False).iloc[0]
        picks.append({'测试年': ty, '选中参数': p['name'], '训练年化': p['cagr'],
                      '训练回撤': p['dd'], '训练Calmar': p['calmar']})
        wf.append(R[p['name']].loc[te0:te1].dropna())
    P = pd.DataFrame(picks)
    P.to_csv(os.path.join(OUT, '14_F2族_走步选择.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f')
    W = pd.concat(wf).sort_index()
    W.to_csv(os.path.join(OUT, '14_F2族_走步日收益.csv'), encoding='utf-8-sig',
             float_format='%.6f', date_format='%Y-%m-%d')
    wc, wd = perf(W.values)
    print('\n走步 2021-2026：年化', round(wc, 4), '最大回撤', round(wd, 4))
    print(P.round(4).to_string(index=False))
    f2name = '10-30|mcap|0.25|0.1|False'
    fc, fd = perf(R[f2name].loc['2021-01-01':'2026-12-31'].dropna().values)
    print('F2 同期：年化', round(fc, 4), '最大回撤', round(fd, 4))

    # PBO
    X = R.fillna(0).values
    T, N = X.shape
    S = 16
    b = np.linspace(0, T, S + 1).astype(int)
    blocks = [X[b[i]:b[i + 1]] for i in range(S)]
    rng = np.random.default_rng(20261023)
    combos = list(itertools.combinations(range(S), S // 2))
    if len(combos) > 3000:
        idx = rng.choice(len(combos), 3000, replace=False)
        combos = [combos[i] for i in idx]
    pr = []
    for combo in combos:
        is_idx = set(combo)
        isx = np.vstack([blocks[i] for i in is_idx])
        oosx = np.vstack([blocks[i] for i in range(S) if i not in is_idx])
        iss = np.array([sharpe(isx[:, j]) for j in range(N)])
        oos = np.array([sharpe(oosx[:, j]) for j in range(N)])
        best = int(np.argmax(iss))
        pr.append((oos < oos[best]).mean())
    pr = np.array(pr)
    pd.DataFrame({'样本外排名分位': pr, '是否低于中位': pr < .5}).to_csv(
        os.path.join(OUT, '14_F2族_PBO分割.csv'), index=False, encoding='utf-8-sig',
        float_format='%.4f')
    print('\nPBO', round(float((pr < .5).mean()), 4), '（低于中位比例）')

    # DSR
    r = R[f2name].dropna().values
    sr = r.mean() / (r.std() + 1e-12) * np.sqrt(252)
    skew = pd.Series(r).skew()
    kurt = pd.Series(r).kurt() + 3
    trials = np.array([sharpe(R[c].dropna().values) for c in R.columns])
    sr0 = np.sqrt(np.var(trials, ddof=1)) * (
        (1 - 0.5772156649) * norm.ppf(1 - 1 / N)
        + 0.5772156649 * norm.ppf(1 - 1 / (N * np.e)))
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    dsr = norm.cdf((sr - sr0) / (denom + 1e-12) * np.sqrt(len(r) / 252))
    print('F2 年化夏普', round(sr, 4), '偏度', round(skew, 3), '峰度', round(kurt, 3),
          '试验数', N, 'SR0', round(sr0, 4), 'DSR', round(dsr, 4))

    # 图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    ax.plot(W.index, np.cumprod(1 + W.values), label='走步样本外', color='#1f4e79', lw=1.5)
    ax.plot(R[f2name].loc['2021-01-01':'2026-12-31'].index,
            np.cumprod(1 + R[f2name].loc['2021-01-01':'2026-12-31'].values),
            label='固定 F2', color='#c0504d', lw=1.0, alpha=.8)
    ax.set_yscale('log')
    ax.set_title('① F2 族走步 vs 固定 F2（2021–2026）')
    ax.legend()
    ax = axes[1]
    ax.hist(pr, bins=20, color='#4f81bd', edgecolor='white')
    ax.axvline(.5, color='#c0504d', ls='--', lw=1)
    ax.set_title(f'② PBO = {(pr < .5).mean():.1%}')
    ax.set_xlabel('样本内最优在样本外的排名分位')
    fig.suptitle('F2 候选族：走步与 PBO', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '14_F2族_过拟合.png'), dpi=160)
    plt.close()


if __name__ == '__main__':
    main()
