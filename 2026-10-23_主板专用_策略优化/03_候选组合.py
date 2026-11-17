# -*- coding: utf-8 -*-
"""主板非ST：候选组合。

单因素显示两个方向在全期/训练/检验都同向：
- 趋势年龄 >=10：过滤太年轻的趋势；
- 20日波动 >=20%：过滤过于沉闷的股票。
本脚本只做少量、可解释的组合，不做大网格；每个组合都保留训练/检验/年度/尾部。
"""
import os
import pandas as pd
import mb_lib as L

OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'
BASE = dict(st_mode='x')

CASES = [
    ('非ST基线', {}),
    ('年龄>=10', {'age_min': 10}),
    ('波动>=20%', {'vol20_min': 0.20}),
    ('年龄>=8+波动>=20%', {'age_min': 8, 'vol20_min': 0.20}),
    ('年龄>=10+波动>=18%', {'age_min': 10, 'vol20_min': 0.18}),
    ('年龄>=10+波动>=20%', {'age_min': 10, 'vol20_min': 0.20}),
    ('年龄>=10+波动>=22%', {'age_min': 10, 'vol20_min': 0.22}),
    ('年龄>=10+波动>=25%', {'age_min': 10, 'vol20_min': 0.25}),
    ('年龄>=12+波动>=20%', {'age_min': 12, 'vol20_min': 0.20}),
    ('年龄>=10+波动>=20%+行业50', {'age_min': 10, 'vol20_min': 0.20, 'sbr_min': 0.50}),
    ('年龄>=10+波动>=20%+行业55', {'age_min': 10, 'vol20_min': 0.20, 'sbr_min': 0.55}),
    ('年龄>=10+波动>=20%+DD60>=-12', {'age_min': 10, 'vol20_min': 0.20, 'dd60_min': -0.12}),
    ('年龄>=10+波动>=20%+DD60>=-15', {'age_min': 10, 'vol20_min': 0.20, 'dd60_min': -0.15}),
    ('年龄>=10+波动>=20%+持有15', {'age_min': 10, 'vol20_min': 0.20, 'hold': 15}),
    ('年龄>=10+波动>=20%+持有15+分批20', {'age_min': 10, 'vol20_min': 0.20, 'hold': 15,
                                      'partial_trigger': 0.20, 'partial_frac': 0.5}),
    ('年龄>=10+波动>=20%+分批20', {'age_min': 10, 'vol20_min': 0.20,
                                 'partial_trigger': 0.20, 'partial_frac': 0.5}),
    ('年龄>=10+波动>=20%+浮盈20保本', {'age_min': 10, 'vol20_min': 0.20, 'profit_lock': True,
                                     'lock_trigger': 0.20, 'lock_gain': 0.0}),
    ('年龄>=10+波动>=20%+1槽', {'age_min': 10, 'vol20_min': 0.20, 'slots': 1}),
    ('年龄>=10+波动>=20%+3槽', {'age_min': 10, 'vol20_min': 0.20, 'slots': 3}),
    ('年龄>=10+波动>=20%+单票35%', {'age_min': 10, 'vol20_min': 0.20, 'maxpos': 0.35}),
    ('年龄>=10+波动>=20%+排序mcap', {'age_min': 10, 'vol20_min': 0.20, 'rank': 'mcap'}),
    ('年龄>=10+波动>=20%+排序sbr', {'age_min': 10, 'vol20_min': 0.20, 'rank': 'sbr'}),
    ('年龄>=10+波动>=20%+ATR>=3%', {'age_min': 10, 'vol20_min': 0.20, 'atr_min': 0.03}),
    ('年龄>=10+波动>=20%+价格<=8', {'age_min': 10, 'vol20_min': 0.20, 'price_max': 8}),
    ('年龄>=10+波动>=20%+成交额>=5000万', {'age_min': 10, 'vol20_min': 0.20, 'amt_min': 5e7}),
    ('年龄>=10+波动>=20%+严ST', {'age_min': 10, 'vol20_min': 0.20, 'st_mode': 'one'}),
    ('年龄>=10+波动>=20%+更严ST', {'age_min': 10, 'vol20_min': 0.20, 'st_mode': 'y'}),
    ('年龄>=10+波动>=20%+松ST', {'age_min': 10, 'vol20_min': 0.20, 'st_mode': 'two'}),
]


def main():
    e = L.fresh_engine()
    rows = []
    for name, cfg in CASES:
        c = dict(BASE); c.update(cfg); c['name'] = name
        E, T, r = L.run_one(e, c)
        rows.append(r)
        print(name, round(r['全期_cagr'], 4), round(r['全期_maxdd'], 4),
              r['全期_trades'], round(r['检验_cagr'], 4), r['负年份数'], flush=True)
    R = pd.DataFrame(rows)
    front = ['cfg', '全期_cagr', '全期_maxdd', '全期_trades', '全期_win', '全期_pf',
             '训练_cagr', '训练_maxdd', '训练_trades', '检验_cagr', '检验_maxdd',
             '检验_trades', '全期_最大单笔亏损', '全期_前5笔占比', '负年份数', '最差年']
    R = R[front + [c for c in R.columns if c not in front]]
    R.to_csv(os.path.join(OUT, '03_候选组合.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f')
    print(R[front].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
