# -*- coding: utf-8 -*-
"""主板非ST：年龄区间 × 持有天数 × 分批止盈 × 排序。

目的：检查“年龄>=10”是不是一个孤立的点；用年龄上/下限组成区间，
再看 10/15/20 日持有和分批止盈能否把结果拉成平台。
"""
import os
import pandas as pd
import mb_lib as L

OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'
ROWS = []
for amin, amax in [(8, 20), (9, 20), (10, 20), (9, 25), (10, 25), (10, 30), (8, 30)]:
    for hold in [10, 15, 20]:
        for pt in [None, 0.20, 0.25]:
            for rank in ['amt', 'mcap']:
                name = f'年龄{amin}-{amax} 持有{hold} ' + (f'分批{pt:.0%} ' if pt else '不分批 ') + f'排序{rank}'
                cfg = dict(st_mode='x', age_min=amin, age_max=amax, vol20_min=0.20,
                           dd60_min=-0.15, hold=hold, rank=rank)
                if pt is not None:
                    cfg.update(partial_trigger=pt, partial_frac=0.5)
                ROWS.append((name, cfg))


def main():
    e = L.fresh_engine()
    rows = []
    for name, cfg in ROWS:
        c = dict(cfg); c['name'] = name
        E, T, r = L.run_one(e, c)
        rows.append(r)
    R = pd.DataFrame(rows)
    front = ['cfg', '全期_cagr', '全期_maxdd', '全期_trades', '全期_win', '全期_pf',
             '训练_cagr', '训练_maxdd', '训练_trades', '检验_cagr', '检验_maxdd',
             '检验_trades', '全期_最大单笔亏损', '全期_前5笔占比', '负年份数', '最差年']
    R = R[front + [c for c in R.columns if c not in front]]
    R.to_csv(os.path.join(OUT, '05_年龄区间_持有_分批.csv'), index=False,
             encoding='utf-8-sig', float_format='%.4f')
    print(R[front].sort_values('全期_cagr', ascending=False).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
