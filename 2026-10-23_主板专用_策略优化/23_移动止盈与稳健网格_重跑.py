# -*- coding: utf-8 -*-
"""F2 移动止盈网格重跑（2026-10-24 修复口径）+ 用 14 族汇总重建 11 表。

17 表：trail_trigger ∈ {20%,25%,30%,40%,关闭} × trail_pct ∈ {5%,8%,10%,15%}，
其余固定为 F2（age10-30、vol20>=20%、DD60>=-15%、20日、分批25、保本10、mcap）。
11 表：14_F2族_汇总.csv 的 192 个配置，仅重命名成原表习惯。
"""
import os
import pandas as pd
import mb_lib as L
from mb_engine import Engine as LocalEngine

L.eng.Engine = LocalEngine
OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'
F2 = dict(st_mode='x', age_min=10, age_max=30, vol20_min=0.20, dd60_min=-0.15,
          hold=20, partial_trigger=0.25, partial_frac=0.5, profit_lock=True,
          lock_trigger=0.10, lock_gain=0.0, rank='mcap')

e = L.fresh_engine()
rows = []
for tt in [0.20, 0.25, 0.30, 0.40, 9.0]:
    for tp in [0.05, 0.08, 0.10, 0.15]:
        c = dict(F2, trail_trigger=tt, trail_pct=tp, name=f'回撤{tt}/{tp}')
        _, _, r = L.run_one(e, c)
        r['tt'] = tt; r['tp'] = tp
        rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, '17_F2_移动止盈.csv'), index=False,
                          encoding='utf-8-sig', float_format='%.4f')
print('17 done')

M = pd.read_csv(os.path.join(OUT, '14_F2族_汇总.csv'))
def rename(s):
    a, rest = s.split('|', 1)
    rank, pt, lock, ext = rest.split('|')
    return f'{a} {rank} pt{pt} lock{lock} ext{ext}'
M['cfg'] = M.cfg.map(rename)
M.to_csv(os.path.join(OUT, '11_持有20_稳健网格.csv'), index=False,
         encoding='utf-8-sig', float_format='%.4f')
print('11 rebuilt', len(M), 'rows')
