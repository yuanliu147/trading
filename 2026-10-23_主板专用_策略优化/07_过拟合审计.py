# -*- coding: utf-8 -*-
"""主板非ST：候选族的过拟合审计。

三件事：
1) 走步（walk-forward）：每年只用过去 3 年选参数，下一年用选中的参数，
   把 2021-2026 的样本外日收益拼起来；看“选参数”这件事本身能不能赚钱。
2) PBO（CSCV）：把 108 个候选的日收益切成 16 段，随机 8 段做样本内、
   另外 8 段做样本外；看样本内第一名在样本外的排名，估计过拟合概率。
3) DSR：对 C1 的夏普做多重试验缩水。

候选族来自 05：年龄区间 7 种 × 持有 3 种 × 分批 3 种 × 排序 2 种 = 126，
去掉重复后实际 126 个；每个都只改结构，不改市场/行业/流动性底座。
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
import mb_lib as L

OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'
C1 = dict(name='C1', st_mode='x', age_min=9, age_max=25, vol20_min=0.20,
          dd60_min=-0.15, hold=10, partial_trigger=0.25, partial_frac=0.5, rank='amt')


def make_cases():
    cases = []
    for amin, amax in [(8, 20), (9, 20), (10, 20), (9, 25), (10, 25), (10, 30), (8, 30)]:
        for hold in [10, 15, 20]:
            for pt in [None, 0.20, 0.25]:
                for rank in ['amt', 'mcap']:
                    name = f'{amin}-{amax}|{hold}|{pt}|{rank}'
                    cfg = dict(st_mode='x', age_min=amin, age_max=amax, vol20_min=0.20,
                               dd60_min=-0.15, hold=hold, rank=rank)
                    if pt is not None:
                        cfg.update(partial_trigger=pt, partial_frac=0.5)
                    cases.append((name, cfg))
    return cases


def perf(r):
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return np.nan, np.nan
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (252 / len(r)) - 1
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return cagr, dd


def sharpe(r):
    r = np.asarray(r, dtype=float)
    return r.mean() / (r.std() + 1e-12) * np.sqrt(252)


def main():
    e = L.fresh_engine()
    cases = make_cases()
    rets = {}
    rows = []
    for i, (name, cfg) in enumerate(cases):
        c = dict(cfg); c['name'] = name
        E, T, row = L.run_one(e, c, return_daily=True)
        rets[name] = E.ret
        row['name'] = name
        rows.append(row)
        if (i + 1) % 30 == 0:
            print('run', i + 1, '/', len(cases), flush=True)
    R = pd.DataFrame({k: v for k, v in rets.items()})
    R.index = pd.DatetimeIndex(R.index)
    R = R.sort_index()
    R.to_csv(os.path.join(OUT, '07_候选族_日收益.csv'), encoding='utf-8-sig',
             float_format='%.6f', date_format='%Y-%m-%d')
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(OUT, '07_候选族_汇总.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f')

    # 1) 走步
    wf_rows, picks = [], []
    test_years = list(range(2021, 2027))
    for ty in test_years:
        tr0, tr1 = f'{ty-3}-01-01', f'{ty-1}-12-31'
        te0, te1 = f'{ty}-01-01', f'{ty}-12-31'
        scores = []
        for name in R.columns:
            x = R[name].loc[tr0:tr1].dropna()
            cagr, dd = perf(x)
            calmar = cagr / (abs(dd) + 1e-9)
            scores.append((name, cagr, dd, calmar))
        S = pd.DataFrame(scores, columns=['name', 'cagr', 'dd', 'calmar'])
        pick = S.sort_values('calmar', ascending=False).iloc[0]
        picks.append({'测试年': ty, '选中参数': pick['name'],
                      '训练年化': pick['cagr'], '训练回撤': pick['dd'],
                      '训练Calmar': pick['calmar']})
        wf_rows.append(R[pick['name']].loc[te0:te1].dropna())
    W = pd.concat(wf_rows).sort_index()
    W.to_csv(os.path.join(OUT, '07_走步_样本外日收益.csv'), encoding='utf-8-sig',
             float_format='%.6f', date_format='%Y-%m-%d')
    p = pd.DataFrame(picks)
    p.to_csv(os.path.join(OUT, '07_走步_选择.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f')
    wc, wd = perf(W.values)
    print('\n走步 2021-2026：年化', round(wc, 4), '最大回撤', round(wd, 4))
    print(p.round(4).to_string(index=False))

    # 固定 C1 在 2021-2026 的对照
    c1_name = '9-25|10|0.25|amt'
    c1 = R[c1_name].loc['2021-01-01':'2026-12-31'].dropna()
    cc, cd = perf(c1.values)
    print('固定 C1 同期：年化', round(cc, 4), '最大回撤', round(cd, 4))

    # 2) PBO / CSCV
    X = R.dropna(how='all').fillna(0.0).values
    T, N = X.shape
    S = 16
    bounds = np.linspace(0, T, S + 1).astype(int)
    blocks = [X[bounds[i]:bounds[i + 1]] for i in range(S)]
    rng = np.random.default_rng(20261023)
    pbo_rows = []
    combos = list(itertools.combinations(range(S), S // 2))
    if len(combos) > 3000:
        idx = rng.choice(len(combos), 3000, replace=False)
        combos = [combos[i] for i in idx]
    for combo in combos:
        is_idx = set(combo)
        is_x = np.vstack([blocks[i] for i in is_idx])
        oos_x = np.vstack([blocks[i] for i in range(S) if i not in is_idx])
        is_sr = np.array([sharpe(is_x[:, j]) for j in range(N)])
        oos_sr = np.array([sharpe(oos_x[:, j]) for j in range(N)])
        best = int(np.argmax(is_sr))
        rank = (oos_sr < oos_sr[best]).mean()
        pbo_rows.append({'样本内最优': best, '样本外排名': rank,
                         '是否低于中位': rank < 0.5})
    P = pd.DataFrame(pbo_rows)
    P.to_csv(os.path.join(OUT, '07_PBO_分割.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f')
    print('\nPBO（样本内最优在样本外低于中位的概率）', round(P['是否低于中位'].mean(), 4))
    print('样本外排名均值', round(P['样本外排名'].mean(), 4))

    # 3) DSR
    r = R[c1_name].dropna().values
    sr_d = r.mean() / (r.std() + 1e-12)
    sr_a = sr_d * np.sqrt(252)
    skew = pd.Series(r).skew()
    kurt = pd.Series(r).kurt() + 3
    trial_sr = np.array([sharpe(R[c].dropna().values) for c in R.columns])
    var_sr = np.var(trial_sr, ddof=1)
    gamma = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * norm.ppf(1 - 1 / N)
                             + gamma * norm.ppf(1 - 1 / (N * np.e)))
    # 用年化口径统一
    denom = np.sqrt(1 - skew * sr_a + (kurt - 1) / 4 * sr_a ** 2)
    dsr = norm.cdf((sr_a - sr0) / (denom + 1e-12) * np.sqrt(len(r) / 252))
    print('\nC1 年化夏普', round(sr_a, 4), '偏度', round(skew, 3), '峰度', round(kurt, 3))
    print('试验数', N, 'SR0', round(sr0, 4), 'DSR', round(dsr, 4))

    # 图：走步样本外净值 + PBO分布
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    ax.plot(W.index, np.cumprod(1 + W.values), label='走步样本外', color='#1f4e79', lw=1.5)
    ax.plot(c1.index, np.cumprod(1 + c1.values), label='固定C1同期', color='#c0504d', lw=1.0, alpha=.8)
    ax.set_yscale('log')
    ax.set_title('① 走步选择 vs 固定 C1（2021–2026）')
    ax.legend()
    ax = axes[1]
    ax.hist(P['样本外排名'], bins=20, color='#4f81bd', edgecolor='white')
    ax.axvline(.5, color='#c0504d', ls='--', lw=1)
    ax.set_title(f"② PBO：低于中位 {P['是否低于中位'].mean():.1%}")
    ax.set_xlabel('样本内最优在样本外的排名分位')
    fig.suptitle('候选族的过拟合审计：走步 + PBO', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '07_过拟合审计.png'), dpi=160)
    plt.close()


if __name__ == '__main__':
    main()
