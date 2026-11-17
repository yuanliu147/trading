# -*- coding: utf-8 -*-
"""稳健退出候选对比：R1–R7。

R1/R2：15日到期；第15天仍盈利>=3%则再给5天；分批25%；
R3/R5：20日固定；分批25%；
R4：15+5，盈利>=0 就延长；R6：15+5，盈利>=5% 延长；R7：R1 的3槽版本。
每个版本同时跑固定和持有期 ±1 天抖动（10次）。
"""
import pandas as pd
import mb_lib as L
from mb_engine import Engine as LocalEngine

L.eng.Engine = LocalEngine
BASE = dict(st_mode='x', age_min=9, age_max=25, vol20_min=0.20, dd60_min=-0.15,
            partial_trigger=0.25, partial_frac=0.5)
CASES = [
    ('R1 15+5 盈3% amt', dict(BASE, hold=15, rank='amt'),
     dict(extend_if_gain=.03, extend_days=5, extend_max=20)),
    ('R2 15+5 盈3% mcap', dict(BASE, hold=15, rank='mcap'),
     dict(extend_if_gain=.03, extend_days=5, extend_max=20)),
    ('R3 20日 mcap', dict(BASE, hold=20, rank='mcap'), {}),
    ('R4 15+5 盈0 amt', dict(BASE, hold=15, rank='amt'),
     dict(extend_if_gain=0.0, extend_days=5, extend_max=20)),
    ('R5 20日 amt', dict(BASE, hold=20, rank='amt'), {}),
    ('R6 15+5 盈5% amt', dict(BASE, hold=15, rank='amt'),
     dict(extend_if_gain=.05, extend_days=5, extend_max=20)),
    ('R7 15+5 盈3% 3槽', dict(BASE, hold=15, rank='amt', slots=3, maxpos=.33),
     dict(extend_if_gain=.03, extend_days=5, extend_max=20)),
]


def main():
    e = L.fresh_engine()
    rows = []
    for name, cfg, kw in CASES:
        c = dict(cfg); c['name'] = name
        E, T, r = L.run_one(e, c, run_kw=kw)
        r['variant'] = name; r['mode'] = 'fixed'; rows.append(r)
        for seed in range(1, 11):
            c = dict(cfg); c['name'] = name + f' j{seed}'
            k = dict(kw); k.update(hold_jitter=1, hold_seed=seed)
            E, T, rr = L.run_one(e, c, run_kw=k)
            rr['variant'] = name; rr['mode'] = 'jitter'; rows.append(rr)
    R = pd.DataFrame(rows)
    R.to_csv('10_稳健候选.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    for name, _, _ in CASES:
        x = R[R.variant == name]
        f = x[x['mode'] == 'fixed'].iloc[0]
        j = x[x['mode'] == 'jitter']
        print('\n', name)
        print('fixed', round(f['全期_cagr'], 4), round(f['全期_maxdd'], 4),
              f['全期_trades'], round(f['全期_win'], 4), round(f['检验_cagr'], 4),
              f['负年份数'], round(f['全期_前5笔占比'], 4))
        print('jitter', round(j['全期_cagr'].mean(), 4), round(j['全期_cagr'].std(), 4),
              round(j['全期_cagr'].min(), 4), round(j['全期_maxdd'].mean(), 4),
              round(j['全期_maxdd'].min(), 4), round(j['负年份数'].mean(), 2))


if __name__ == '__main__':
    main()
