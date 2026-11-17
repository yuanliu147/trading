# -*- coding: utf-8 -*-
"""44组参数的多重检验：Deflated Sharpe、PBO(CSCV)、训练/检验排名。"""
import importlib.util, os, numpy as np, pandas as pd
from scipy.stats import norm

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = importlib.util.spec_from_file_location('m102', os.path.join(BASE, '02_代码/102_两万组合.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

scen = []
for gate in [0.50, 0.55, 0.60]:
    for rank in ['amt', 'sbr']:
        for risk in [0.02, 0.03, 0.04]:
            for heat in [0.05, 0.08]:
                scen.append(dict(name=f'风险{risk}_热{heat}_宽度{gate}_{rank}', slots=2, sizing='risk',
                                 risk=risk, maxpos=0.5, heat_cap=heat, gate_thr=gate, rank=rank))
        scen.append(dict(name=f'固定50_宽度{gate}_{rank}', slots=2, sizing='fixed', risk=0,
                         maxpos=0.5, heat_cap=0.10, gate_thr=gate, rank=rank))
for hold in [15, 25]:
    scen.append(dict(name=f'风险3_热8_宽度55_持有{hold}', slots=2, sizing='risk', risk=0.03,
                     maxpos=0.5, heat_cap=0.08, gate_thr=0.55, rank='amt', hold=hold))
print('trials', len(scen))

rets = {}; stats = []
for sc in scen:
    name = sc.pop('name')
    row = {'name': name}
    for start, end, per in [('2018-01-01', '2021-12-31', 'train'),
                            ('2022-01-01', '2026-10-13', 'test'),
                            ('2018-01-01', '2026-10-13', 'full')]:
        E, T, met = m.run(start, end, **sc)
        row[f'{per}_cagr'] = met['cagr']; row[f'{per}_dd'] = met['maxdd']; row[f'{per}_sharpe'] = met['sharpe']
        if per == 'full':
            rets[name] = E.ret.values
    stats.append(row)
S = pd.DataFrame(stats)
S['rank_train'] = S.train_cagr.rank(ascending=False).astype(int)
S['rank_test'] = S.test_cagr.rank(ascending=False).astype(int)
S['rank_full'] = S.full_cagr.rank(ascending=False).astype(int)
print('\n===== 44组参数（按全期年化排序，前12） =====')
print(S.sort_values('full_cagr', ascending=False).head(12)[
    ['name', 'train_cagr', 'train_dd', 'test_cagr', 'test_dd', 'full_cagr', 'full_dd', 'full_sharpe',
     'rank_train', 'rank_test', 'rank_full']].round(4).to_string(index=False))
print('\n25.8%%配置排名：训练 %d/44，检验 %d/44，全期 %d/44' % tuple(
    S.loc[S.name == '固定50_宽度0.5_amt', ['rank_train', 'rank_test', 'rank_full']].values[0]))
print('\n训练最优：')
print(S.sort_values('train_cagr', ascending=False).head(3)[
    ['name', 'train_cagr', 'test_cagr', 'full_cagr', 'test_dd']].round(4).to_string(index=False))
print('\n检验最优：')
print(S.sort_values('test_cagr', ascending=False).head(3)[
    ['name', 'train_cagr', 'test_cagr', 'full_cagr', 'test_dd']].round(4).to_string(index=False))

# ---- Deflated Sharpe ----
R = pd.DataFrame(rets)
sr_daily = R.mean() / R.std()
best = '固定50_宽度0.5_amt'
sr = sr_daily[best]
T = len(R)
skew = R[best].skew(); kurt = R[best].kurtosis() + 3
N = len(scen)
var_sr = sr_daily.var(ddof=1)
e = np.e
GAMMA = 0.5772156649015329
sr0 = np.sqrt(var_sr) * ((1 - GAMMA) * norm.ppf(1 - 1 / N) +
                         GAMMA * norm.ppf(1 - 1 / (N * e)))
dsr = norm.cdf((sr - sr0) * np.sqrt(T - 1) /
               np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
psr0 = norm.cdf(sr * np.sqrt(T - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
print('\n===== Deflated Sharpe =====')
print('best daily SR %.4f annual %.3f; trials %d; var(SR) %.6f; SR0(daily) %.4f annual %.3f' % (
    sr, sr * np.sqrt(252), N, var_sr, sr0, sr0 * np.sqrt(252)))
print('PSR vs 0: %.4f; DSR vs expected max: %.4f' % (psr0, dsr))
print('skew %.3f kurt %.3f T %d' % (skew, kurt, T))

# ---- PBO CSCV ----
X = R.values  # T x N
Sblocks = 8
edges = np.linspace(0, T, Sblocks + 1).astype(int)
blocks = [np.arange(edges[i], edges[i + 1]) for i in range(Sblocks)]
from itertools import combinations
combos = list(combinations(range(Sblocks), Sblocks // 2))
logits = []; oos_sr = []
for comb in combos:
    is_idx = np.concatenate([blocks[i] for i in comb])
    oos_idx = np.concatenate([blocks[i] for i in range(Sblocks) if i not in comb])
    is_sr = X[is_idx].mean(0) / X[is_idx].std(0)
    oos = X[oos_idx].mean(0) / X[oos_idx].std(0)
    j = int(np.nanargmax(is_sr))
    rank = (oos < oos[j]).sum() + 1
    w = rank / (N + 1)
    logits.append(np.log(w / (1 - w)))
    oos_sr.append(oos[j])
logits = np.array(logits)
pbo = float((logits <= 0).mean())
print('\n===== PBO (CSCV, 8块, %d组合) =====' % len(combos))
print('PBO = %.4f; OOS SR median %.4f; 选中配置OOS为负比例 %.3f' % (
    pbo, np.median(oos_sr), np.mean(np.array(oos_sr) <= 0)))

S.to_csv(os.path.join(BASE, '03_结果/四十四组_多重检验.csv'), index=False, encoding='utf-8-sig')
R.to_csv(os.path.join(BASE, '03_结果/四十四组_日收益.csv'), encoding='utf-8-sig')
pd.Series({'dsr': dsr, 'psr0': psr0, 'pbo': pbo, 'sr_daily': sr, 'sr0_daily': sr0,
           'trials': N, 'n_days': T}).to_csv(os.path.join(BASE, '03_结果/多重检验摘要.csv'),
                                             encoding='utf-8-sig')
