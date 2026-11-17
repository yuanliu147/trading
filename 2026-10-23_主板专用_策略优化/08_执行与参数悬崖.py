# -*- coding: utf-8 -*-
"""主板非ST C1：执行细节与参数悬崖。

C1 的收益对“10日持有”很敏感；本脚本检查 8–12 日、分批 20–30%、
单票 40–50%、2/3 槽、止损 15–25%、开盘跳空过滤、行业分散。
"""
import os
import numpy as np
import pandas as pd
import mb_lib as L

OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'
C1 = dict(st_mode='x', age_min=9, age_max=25, vol20_min=0.20,
          dd60_min=-0.15, hold=10, partial_trigger=0.25, partial_frac=0.5, rank='amt')
CASES = [('C1', {})]


def add(name, **kw):
    c = dict(C1); c.update(kw); CASES.append((name, c))


for x in [8, 9, 10, 11, 12, 13]:
    add(f'持有{x}日', hold=x)
for x in [0.18, 0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30]:
    add(f'涨{x:.0%}卖一半', partial_trigger=x)
for x in [0.40, 0.425, 0.45, 0.475, 0.50]:
    add(f'单票{x:.1%}', maxpos=x)
for x in [2, 3, 4]:
    add(f'{x}槽', slots=x)
for x in [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
    add(f'止损{x:.0%}', stop_pct=x)
add('行业分散', sector_cap=True)
add('3槽+行业分散', slots=3, sector_cap=True)
add('3槽+单票33%', slots=3, maxpos=0.33)
add('3槽+单票30%', slots=3, maxpos=0.30)
add('2槽+单票40%', maxpos=0.40)
add('2槽+单票35%', maxpos=0.35)


def gap_skip(e, base, gap_max, gap_min=None):
    """信号 t-1 已知的次日跳空，t 开盘不买。"""
    skip = set()
    T = len(e.dates)
    for t in range(1, T):
        cand = np.where(base[t - 1])[0]
        if len(cand) == 0:
            continue
        ng = e.NG[t - 1, cand]
        bad = ng > gap_max
        if gap_min is not None:
            bad |= ng < gap_min
        for j in cand[bad]:
            skip.add((e.codes[j], e.dates[t]))
    return skip


def main():
    e = L.fresh_engine()
    # 预先构造 C1 的 base，用于跳空过滤
    base = L.build_base(e, C1)
    rows = []
    for name, cfg in CASES:
        c = dict(cfg); c['name'] = name
        E, T, r = L.run_one(e, c)
        rows.append(r)
    for gmax in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        c = dict(C1); c['name'] = f'跳空<={gmax:.0%}'
        skip = gap_skip(e, base, gmax)
        E, T, r = L.run_one(e, c, run_kw={'skip_entries': skip})
        rows.append(r)
    for gmin in [-0.05, -0.03, -0.01, 0.0]:
        c = dict(C1); c['name'] = f'跳空>={gmin:.0%}'
        skip = gap_skip(e, base, 0.05, gmin)
        E, T, r = L.run_one(e, c, run_kw={'skip_entries': skip})
        rows.append(r)
    R = pd.DataFrame(rows)
    front = ['cfg', '全期_cagr', '全期_maxdd', '全期_trades', '全期_win', '全期_pf',
             '训练_cagr', '训练_maxdd', '训练_trades', '检验_cagr', '检验_maxdd',
             '检验_trades', '全期_最大单笔亏损', '全期_前5笔占比', '负年份数', '最差年']
    R = R[front + [c for c in R.columns if c not in front]]
    R.to_csv(os.path.join(OUT, '08_执行与悬崖.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f')
    print(R[front].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
