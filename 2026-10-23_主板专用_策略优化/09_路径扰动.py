# -*- coding: utf-8 -*-
"""持有期抖动：检验“固定 10 日/15 日/20 日”是不是单点运气。

C1 底座：主板非ST、年龄9-25、20日波动>=20%、DD60>=-15%、
2槽45%、涨25%卖一半、-20%止损。持有期在 9–11 日之间随机抖动。
"""
import pandas as pd
import mb_lib as L
from mb_engine import Engine as LocalEngine

L.eng.Engine = LocalEngine
C1 = dict(st_mode='x', age_min=9, age_max=25, vol20_min=0.20, dd60_min=-0.15,
          partial_trigger=0.25, partial_frac=0.5, rank='amt')


def main():
    e = L.fresh_engine()
    rows = []
    for hold in [10, 11, 15, 20]:
        for jit in [0, 1, 2, 3]:
            seeds = [1] if jit == 0 else list(range(1, 21))
            for seed in seeds:
                c = dict(C1); c.update(name=f'h{hold} j{jit} s{seed}', hold=hold)
                E, T, r = L.run_one(e, c, run_kw={'hold_jitter': jit, 'hold_seed': seed})
                r['hold'] = hold; r['jitter'] = jit; r['seed'] = seed
                rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv('09_路径扰动.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    print(R.groupby(['hold', 'jitter']).agg(
        n=('seed', 'size'), 年化均值=('全期_cagr', 'mean'), 年化标准差=('全期_cagr', 'std'),
        年化最差=('全期_cagr', 'min'), 年化最好=('全期_cagr', 'max'),
        回撤均值=('全期_maxdd', 'mean'), 回撤最差=('全期_maxdd', 'min'),
        检验均值=('检验_cagr', 'mean'), 负年数=('负年份数', 'mean')).round(4).to_string())


if __name__ == '__main__':
    main()
