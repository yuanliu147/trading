# -*- coding: utf-8 -*-
"""胜者深化审计：V2_blend1020 / V1_core20 / V0_attack10。

内容：
  1) 去掉各自利润前 5 笔（skip_entries 精确重跑，不是近似扣减）后的年化；
  2) 块自举 1 年 / 3 年回撤 P5/P50（与既有研究同法：block=10, sims=2000）；
  3) 最差 5 个月、平均持仓天数、平均同时持仓数、月度胜率；
  4) 年度收益对照表。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bt2016 as B  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, '03_结果')
START_CASH = 20000.0


def bootstrap_dd(rets, n, sims=2000, block=10, seed=20261024):
    r = pd.Series(rets).dropna().values
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(sims):
        x = []
        while len(x) < n:
            i = rng.integers(0, max(1, len(r) - block))
            x.extend(r[i:i + block])
        eq = np.cumprod(1 + np.array(x[:n]))
        out.append((eq / np.maximum.accumulate(eq) - 1).min())
    return np.array(out)


def rerun_drop_top(cfgA, cfgB, T, k=5, half=10000.0):
    """剔除利润前 k 笔：把对应 (股票, 信号日=入场前一日) 的信号格置 False 后精确重跑。

    说明：mb_engine 不支持 skip_entries，这里在信号矩阵层面屏蔽，效果等价于
    "当天不选这只股"，下一个候选照常顶上——正是"错过这几笔"的语义。
    """
    top = T.nlargest(k, 'pnl')
    cells = set()
    for c, d in zip(top.code, top.entry_date):
        j = ENGINE.code_idx.get(str(c))
        try:
            t_entry = ENGINE.dates.get_loc(pd.Timestamp(d))
        except KeyError:
            continue
        if j is None or t_entry is None or t_entry == 0:
            continue
        cells.add((t_entry - 1, j))  # 信号在 t-1，入场在 t
    orig_build = B.build_base

    def patched(e, cfg):
        base = orig_build(e, cfg)
        for t, j in cells:
            base[t, j] = False
        return base

    B.build_base = patched
    try:
        if cfgB is None:
            E, T2, _ = B.run_one(ENGINE, dict(cfgA))
        else:
            EA, TA, _ = B.run_one(ENGINE, dict(cfgA, start_cash=half))
            EB, TB, _ = B.run_one(ENGINE, dict(cfgB, start_cash=half))
            idx = EA.index.union(EB.index)
            eq = EA.equity.reindex(idx).ffill() + EB.equity.reindex(idx).ffill()
            E = pd.DataFrame({'equity': eq})
            E['ret'] = E.equity.pct_change().fillna(0)
            T2 = pd.concat([TA, TB], ignore_index=True)
    finally:
        B.build_base = orig_build
    key0 = {(str(c), pd.Timestamp(d)) for c, d in zip(top.code, top.entry_date)}
    n_dropped = sum(1 for c, d in zip(T2.code, T2.entry_date)
                    if (str(c), pd.Timestamp(d)) in key0)
    return E, T2, len(key0) - n_dropped


def stats(name, E, T, start_cash=START_CASH):
    total = E.equity.iloc[-1] / start_cash - 1
    cagr = (1 + total) ** (252 / len(E)) - 1
    dd = (E.equity / E.equity.cummax() - 1).min()
    m = E.equity.resample('ME').last()
    prev = E.equity.iloc[0]
    mret = m / m.shift(1, fill_value=prev) - 1
    r1 = bootstrap_dd(E.ret.values, 252)
    r3 = bootstrap_dd(E.ret.values, 756)
    return dict(变体=name, 全期年化=cagr, 全期回撤=dd, 交易=len(T),
                平均持仓天=T.hold.mean(),
                月胜率=(mret > 0).mean(),
                最差月=mret.min(), 最好月=mret.max(),
                自举1年回撤P5=np.percentile(r1, 5), 自举1年回撤中位=np.percentile(r1, 50),
                自举3年回撤P5=np.percentile(r3, 5), 自举3年回撤中位=np.percentile(r3, 50))


def main():
    global ENGINE
    ENGINE = B.fresh_engine()
    A10 = dict(B.ATTACK)
    C20 = dict(B.CORE)

    variants = {}
    # V0
    E0, T0, _ = B.run_one(ENGINE, A10)
    variants['V0_attack10'] = (E0, T0, A10, None)
    # V1
    E1, T1, _ = B.run_one(ENGINE, C20)
    variants['V1_core20'] = (E1, T1, C20, None)
    # V2
    EA, TA, _ = B.run_one(ENGINE, dict(A10, start_cash=10000.0))
    EB, TB, _ = B.run_one(ENGINE, dict(C20, start_cash=10000.0))
    idx = EA.index.union(EB.index)
    eq = EA.equity.reindex(idx).ffill() + EB.equity.reindex(idx).ffill()
    E2 = pd.DataFrame({'equity': eq})
    E2['ret'] = E2.equity.pct_change().fillna(0)
    T2 = pd.concat([TA, TB], ignore_index=True)
    variants['V2_blend1020'] = (E2, T2, A10, C20)

    rows = []
    drop_rows = []
    for name, (E, T, cfgA, cfgB) in variants.items():
        print(f'审计 {name} ...', flush=True)
        rows.append(stats(name, E, T))
        Ed, Td, ndrop = rerun_drop_top(cfgA, cfgB, T, k=5)
        cd = (Ed.equity.iloc[-1] / START_CASH) ** (252 / len(Ed)) - 1
        ddd = (Ed.equity / Ed.equity.cummax() - 1).min()
        drop_rows.append(dict(变体=name, 去前5笔年化=cd, 去前5笔回撤=ddd,
                              实际剔除笔数=ndrop, 剩余笔数=len(Td)))
    S = pd.DataFrame(rows)
    D = pd.DataFrame(drop_rows)
    S.to_csv(os.path.join(OUT, '深化_胜者审计.csv'), index=False, encoding='utf-8-sig')
    D.to_csv(os.path.join(OUT, '深化_去前5笔.csv'), index=False, encoding='utf-8-sig')

    # 年度对照
    yr = {}
    for name, (E, T, *_ ) in variants.items():
        prev = E.equity.iloc[0]
        vals = {}
        for y in range(2017, 2027):
            x = E.equity.loc[f'{y}-01-01':f'{y}-12-31']
            if len(x) == 0:
                continue
            vals[y] = x.iloc[-1] / prev - 1
            prev = x.iloc[-1]
        yr[name] = vals
    Y = pd.DataFrame(yr)
    Y.to_csv(os.path.join(OUT, '深化_年度对照.csv'), encoding='utf-8-sig')

    print(S.round(4).to_string(index=False))
    print()
    print(D.round(4).to_string(index=False))
    print()
    print(Y.round(4).to_string())
    print('完成')


if __name__ == '__main__':
    main()
