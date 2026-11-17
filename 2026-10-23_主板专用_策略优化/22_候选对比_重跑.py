# -*- coding: utf-8 -*-
"""F1–F4 最终候选对比（2026-10-24 修复后重跑：全覆盖市值 + mcap 哨兵修复）。

配置自 12_最终候选对比.csv 的命名还原：
  全部 st_mode=x, vol20>=20%, DD60>=-15%, 20日持有, 分批25%,
  移动止盈 25%/10%, rank=mcap；差异仅在年龄区间与保本触发。
每个候选：固定持有期 + 持有期±1天抖动（10个种子）。
"""
import os
import pandas as pd
import mb_lib as L
from mb_engine import Engine as LocalEngine

L.eng.Engine = LocalEngine
OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'

BASE = dict(st_mode='x', vol20_min=0.20, dd60_min=-0.15, hold=20,
            partial_trigger=0.25, partial_frac=0.5, profit_lock=True,
            lock_gain=0.0, trail_trigger=0.25, trail_pct=0.10, rank='mcap')
CASES = [
    ('F1 10-25 lock15', dict(BASE, age_min=10, age_max=25, lock_trigger=0.15)),
    ('F2 10-30 lock10', dict(BASE, age_min=10, age_max=30, lock_trigger=0.10)),
    ('F3 10-25 lock10', dict(BASE, age_min=10, age_max=25, lock_trigger=0.10)),
    ('F4 10-30 lock15', dict(BASE, age_min=10, age_max=30, lock_trigger=0.15)),
]


def main():
    e = L.fresh_engine()
    rows = []
    for name, cfg in CASES:
        c = dict(cfg); c['name'] = name
        E, T, r = L.run_one(e, c)
        r['variant'] = name; r['mode'] = 'fixed'; rows.append(r)
        for seed in range(1, 11):
            c = dict(cfg); c['name'] = f'{name} j{seed}'
            E, T, rr = L.run_one(e, c, run_kw=dict(hold_jitter=1, hold_seed=seed))
            rr['variant'] = name; rr['mode'] = 'jitter'; rows.append(rr)
        print(name, 'fixed', round(r['全期_cagr'], 4), round(r['全期_maxdd'], 4),
              r['全期_trades'], '训练', round(r['训练_cagr'], 4),
              '检验', round(r['检验_cagr'], 4), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, '12_最终候选对比.csv'), index=False,
             encoding='utf-8-sig', float_format='%.4f')
    print('\njitter 均值:')
    j = R[R['mode'] == 'jitter'].groupby('variant').agg(
        全期_cagr=('全期_cagr', 'mean'), 全期_maxdd=('全期_maxdd', 'mean'),
        检验_cagr=('检验_cagr', 'mean'))
    print(j.round(4).to_string())


if __name__ == '__main__':
    main()
