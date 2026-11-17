# -*- coding: utf-8 -*-
"""F05 附加：独立重写 Holm + White Reality Check（同暴露日超额，stationary bootstrap）。
算法按 stats_hist.py 的口径独立实现；用相同种子 20261018 时应能精确复现交付 p 值。只读。"""
import os

import numpy as np
import pandas as pd
from scipy import stats as st

SRC = "/Users/admin/WorkSpace/trading/八系统回测"
HIST = os.path.join(SRC, "结果_历史成分")
OUT = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = ["MACD入门", "MACD波段", "波段为王1", "波段为王2", "看盘", "短线30技", "裸K", "KDJ"]
B, AVG_BLOCK, SEED = 5000, 20, 20261018

bench = pd.read_pickle(os.path.join(SRC, "zz500_index.pkl")).set_index("date").close
excess = {}
for name in SYSTEMS:
    eq = pd.read_csv(os.path.join(HIST, f"权益_历史成分_{name}.csv"), parse_dates=["date"]).set_index("date")
    r = eq.equity.pct_change().fillna(0.0)
    b = bench.reindex(eq.index).ffill()
    rm = b.pct_change().fillna(0.0)
    exp = (eq.market / eq.equity).shift(1).fillna(0.0)
    excess[name] = r - exp * rm
E = pd.DataFrame(excess).dropna()
T, k = E.shape
obs = E.mean().values * 252

# Holm（单侧 t）
mu = E.mean().values
se = E.std(ddof=1).values / np.sqrt(T)
p_raw = 1 - st.t.cdf(mu / se, df=T - 1)
order = np.argsort(p_raw)
p_holm = np.empty(k)
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (k - rank) * p_raw[i]))
    p_holm[i] = run

# Reality Check
centered = (E - E.mean()).values
rng = np.random.default_rng(SEED)
max_boot = np.empty(B)
boot_means = np.empty((B, k))
R = E.values
for i in range(B):
    idx = np.empty(T, dtype=np.int64)
    idx[0] = rng.integers(T)
    u = rng.random(T - 1)
    jumps = rng.integers(T, size=T - 1)
    for t in range(1, T):
        idx[t] = jumps[t - 1] if u[t - 1] < 1.0 / AVG_BLOCK else (idx[t - 1] + 1) % T
    bm = centered[idx].mean(axis=0)
    boot_means[i] = bm
    max_boot[i] = bm.max() * 252
p_ind = (1 + (boot_means >= (obs / 252)).sum(axis=0)) / (B + 1)
p_rc = (1 + int((max_boot >= obs.max()).sum())) / (B + 1)

claim = pd.read_csv(os.path.join(HIST, "统计_多重检验.csv")).set_index("system")
rows = []
lines = []
for j, name in enumerate(SYSTEMS):
    c = claim.loc[name]
    row = dict(system=name, excess_ann=obs[j], d_excess=obs[j] - c.excess_ann,
               p_boot=p_ind[j], d_pboot=p_ind[j] - c.p_bootstrap,
               p_holm=p_holm[j], claim_holm=c.p_holm_wald)
    rows.append(row)
    lines.append(f"{name:8s} excess_ann={obs[j]:+.4%} (Δ={row['d_excess']:+.2e}) "
                 f"p_boot={p_ind[j]:.4f} (交付 {c.p_bootstrap:.4f}) p_holm={p_holm[j]:.3f}")
best = SYSTEMS[int(np.argmax(obs))]
lines.append(f"Reality Check: 最优 {best} excess={obs.max():+.4%}, p_rc={p_rc:.4f} (交付 {claim.loc[best,'p_reality_check_max']:.4f})")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "PD_Holm_RC_复算.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "f05_holm_rc_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))