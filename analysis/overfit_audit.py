#!/usr/bin/env python3
"""过拟合审计：0.25%风险 + 下降态停B 这个"最优格子"是不是数据挖掘出来的？

A. 训练段(2016~2022)选参 → 盲测段(2023~2026-10)验证：风险档×B禁 全网格两段对照
   （此前所有回撤方案都是在全样本上调的，本节是唯一干净的择参-验证分离）
B. 参数扰动：regime MA60→40/120、放量1.5→1.2/2.0、RSI 30→25，重建信号后重测推荐配置
C. 横截面子样本：随机各取一半股票重测
"""
import numpy as np
import pandas as pd
import ladder_doublek_backtest as m
from backtest_audit import gen_signals
from drawdown_control import simulate

NSEED = 30


def summ(g, init=10000.0):
    return dict(cagr=g.cagr.median(), cagr_p5=g.cagr.quantile(.05),
                dd=g.maxdd.median(), dd_worst=g.maxdd.min(), ruin=(g.final < init).mean())


def line(name, s):
    return (f"{name}: 年化中位 {s['cagr']:+.1%} (5%分位 {s['cagr_p5']:+.1%}) | "
            f"回撤中位 {s['dd']:.1%} 最差 {s['dd_worst']:.1%} | 亏损 {s['ruin']:.0%}")


def run_mc(S, C, cal, colidx, state_arr, **kw):
    dayidx = {d: i for i, d in enumerate(cal)}
    g = pd.DataFrame([simulate(S, C, cal, colidx, dayidx, seed, state_arr=state_arr, **kw)
                      for seed in range(NSEED)])
    return g


def main():
    print("载入样本与基准信号…", flush=True)
    syms = m.liquid_symbols(main_only=True)
    df, bundles = m.load_bundles(syms)
    S, close_map, _ = gen_signals(bundles, df)
    close_df = pd.DataFrame(close_map).sort_index().ffill()
    cal = close_df.index
    C = close_df.to_numpy()
    colidx = {s: i for i, s in enumerate(close_df.columns)}
    regime3 = pd.Series(m.build_regime3(df))
    state_arr = regime3.reindex(cal).ffill().to_numpy()
    rows = []

    def record(section, name, g, init=10000.0):
        s = summ(g, init)
        print(line(f"{section} | {name}", s), flush=True)
        rows.append(dict(section=section, config=name, **s))
        pd.DataFrame(rows).to_csv("overfit_audit_results.csv", index=False)

    # ===== A：训练段选参 → 盲测段验证 =====
    split = pd.Timestamp("2022-12-31")
    for pname, lo, hi in [("A训练段16-22", None, split),
                          ("A盲测段23-26", split + pd.Timedelta(days=1), None)]:
        mask = np.ones(len(cal), bool)
        if lo is not None:
            mask &= cal >= lo
        if hi is not None:
            mask &= cal <= hi
        calp, Cp, statep = cal[mask], C[mask], state_arr[mask]
        Sp = S
        if lo is not None:
            Sp = Sp[Sp.entry_date >= lo]
        if hi is not None:
            Sp = Sp[Sp.entry_date <= hi]
        for risk in [0.0025, 0.005, 0.01, 0.02]:
            for bb in [False, True]:
                g = run_mc(Sp, Cp, calp, colidx, statep,
                           risk_pct=risk, block_b_down=bb, min_fee=5.0)
                record(pname, f"风险{risk*100}% B禁{int(bb)}", g)

    # ===== B：参数扰动（全段，推荐配置 0.25%+B禁+最低佣金5元） =====
    g = run_mc(S, C, cal, colidx, state_arr, risk_pct=0.0025, block_b_down=True, min_fee=5.0)
    record("B基准", "MA60/放量1.5/RSI30", g)
    for name, kw in [("regime MA40", dict(ma_n=40)),
                     ("regime MA120", dict(ma_n=120)),
                     ("放量1.2x", dict(vol_mult=1.2)),
                     ("放量2.0x", dict(vol_mult=2.0)),
                     ("RSI<25", dict(rsi_in=25.0))]:
        S2, _, _ = gen_signals(bundles, df, **kw)
        st2 = pd.Series(m.build_regime3(df, ma_n=kw.get("ma_n", 60))).reindex(cal).ffill().to_numpy()
        g = run_mc(S2, C, cal, colidx, st2, risk_pct=0.0025, block_b_down=True, min_fee=5.0)
        record("B扰动", name, g)

    # ===== C：横截面一半股票 =====
    rng = np.random.default_rng(7)
    usyms = np.array(sorted(S.symbol.unique()))
    perm = rng.permutation(usyms)
    for k, half in enumerate([perm[: len(perm) // 2], perm[len(perm) // 2:]]):
        S3 = S[S.symbol.isin(half)]
        g = run_mc(S3, C, cal, colidx, state_arr, risk_pct=0.0025, block_b_down=True, min_fee=5.0)
        record("C半样本", f"第{k+1}半({len(half)}只)", g)


if __name__ == "__main__":
    main()
