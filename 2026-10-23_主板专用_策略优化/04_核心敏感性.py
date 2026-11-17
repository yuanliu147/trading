# -*- coding: utf-8 -*-
"""主板非ST：核心候选的参数悬崖与结构敏感性。

核心来自 03：
  年龄>=10、20日波动>=20%、DD60>=-15%、非ST、宽度>50%、2槽45%、20日、-20%止损。
本脚本只在这个核心周围做单维扰动，检查“是不是只有某一个点好”。
"""
import os
import pandas as pd
import mb_lib as L

OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'
CORE = dict(st_mode='x', age_min=10, vol20_min=0.20, dd60_min=-0.15)
CASES = [('核心', {})]


def add(name, **kw):
    c = dict(CORE); c.update(kw)
    CASES.append((name, c))


for x in [8, 9, 10, 11, 12, 14]:
    add(f'年龄>={x}', age_min=x)
for x in [0.16, 0.18, 0.19, 0.20, 0.21, 0.22, 0.25]:
    add(f'波动>={x:.0%}', vol20_min=x)
for x in [-0.08, -0.10, -0.12, -0.14, -0.15, -0.18, -0.20]:
    add(f'DD60>={x:.0%}', dd60_min=x)
for x in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    add(f'宽度>{x:.0%}', gate_thr=x)
for x in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    add(f'行业宽度>={x:.0%}', sbr_min=x)
for x in [0.0, 0.05, 0.08, 0.10, 0.12, 0.15]:
    add(f'行业20日>{x:.0%}', sr20_min=x)
for x in ['amt', 'mcap', 'sbr', 'sbr_mid', 'lowvol', 'dd60', 'price', 'age']:
    add(f'排序={x}', rank=x)
for x in [10, 15, 20, 25, 30]:
    add(f'持有{x}日', hold=x)
for x in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
    add(f'止损{x:.0%}', stop_pct=x)
for x in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    add(f'单票{x:.0%}', maxpos=x)
for x in [1, 2, 3, 4]:
    add(f'{x}槽', slots=x)
for x in [None, 0.10, 0.15, 0.20, 0.25, 0.30]:
    add('不分批' if x is None else f'涨{x:.0%}卖一半', partial_trigger=x, partial_frac=0.5)
for x in [None, 0.10, 0.15, 0.20, 0.25, 0.30]:
    add('无盈利保护' if x is None else f'浮盈{x:.0%}保本', profit_lock=x is not None,
        lock_trigger=x if x is not None else 0.15, lock_gain=0.0)
for x in [20, 30, 40, 50, 60]:
    add(f'成交额>={x}00万', amt_min=x * 1e6)
for x in [8, 10, 12, 15, 20]:
    add(f'股价<={x}元', price_max=x)
for x in [0.0015, 0.0020, 0.0030, 0.0040]:
    add(f'单边成本{x:.2%}', cost_rate=x)
for x in [None, 0.06, 0.10, 0.12]:
    add('无回撤节流' if x is None else f'回撤{x:.0%}减半', dd_half=x)
for x in ['two', 'x', 'one', 'y']:
    add(f'ST口径={x}', st_mode=x)


def main():
    e = L.fresh_engine()
    rows = []
    for name, cfg in CASES:
        c = dict(cfg); c['name'] = name
        E, T, r = L.run_one(e, c)
        rows.append(r)
    R = pd.DataFrame(rows)
    front = ['cfg', '全期_cagr', '全期_maxdd', '全期_trades', '全期_win', '全期_pf',
             '训练_cagr', '训练_maxdd', '训练_trades', '检验_cagr', '检验_maxdd',
             '检验_trades', '全期_最大单笔亏损', '全期_前5笔占比', '负年份数', '最差年']
    R = R[front + [c for c in R.columns if c not in front]]
    R.to_csv(os.path.join(OUT, '04_核心敏感性.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f')
    print(R[front].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
