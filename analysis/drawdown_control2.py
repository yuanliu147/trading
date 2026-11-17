#!/usr/bin/env python3
"""回撤控制第二轮：低风险档细化 + 最低佣金现实约束 + 本金规模稳健性

第一轮发现：0.5% 单笔风险同时降低回撤、提高年化（波动拖累+持仓集中度机制）。
本轮验证：
- 0.25% / 0.75% / 12持仓，确认 0.5% 不是孤点
- 券商最低佣金 5 元/笔（小仓位在 1 万本金下的真实成本）
- 10 万本金下结论是否保持
"""
import numpy as np
import pandas as pd
import ladder_doublek_backtest as m
from backtest_audit import gen_signals
from drawdown_control import simulate


def main():
    print("载入主板样本与信号…", flush=True)
    syms = m.liquid_symbols(main_only=True)
    df, bundles = m.load_bundles(syms)
    S, close_map, _ = gen_signals(bundles, df)
    close_df = pd.DataFrame(close_map).sort_index().ffill()
    cal = close_df.index
    C = close_df.to_numpy()
    colidx = {s: i for i, s in enumerate(close_df.columns)}
    dayidx = {d: i for i, d in enumerate(cal)}
    state_arr = pd.Series(m.build_regime3(df)).reindex(cal).ffill().to_numpy()

    configs = [
        ("0.25%风险", dict(risk_pct=0.0025)),
        ("0.5%风险", dict(risk_pct=0.005)),
        ("0.75%风险", dict(risk_pct=0.0075)),
        ("0.5% + 12持仓", dict(risk_pct=0.005, max_pos=12)),
        ("0.5% + B禁下降态", dict(risk_pct=0.005, block_b_down=True)),
        ("0.5% + 最低佣金5元", dict(risk_pct=0.005, min_fee=5.0)),
        ("0.5% + B禁下降态 + 最低佣金5元", dict(risk_pct=0.005, block_b_down=True, min_fee=5.0)),
        ("1% + B禁下降态 + 最低佣金5元", dict(risk_pct=0.01, block_b_down=True, min_fee=5.0)),
        ("10万本金 0.5% + B禁下降态 + 最低佣金5元",
         dict(risk_pct=0.005, block_b_down=True, min_fee=5.0, init=100000.0)),
    ]
    NSEED = 30
    rows = []
    curves = {}
    for name, kw in configs:
        res = [simulate(S, C, cal, colidx, dayidx, seed, state_arr=state_arr,
                        store_eq=True, **kw) for seed in range(NSEED)]
        g = pd.DataFrame([{k: v for k, v in r.items() if k != "eq"} for r in res])
        init = kw.get("init", 10000.0)
        row = dict(config=name, cagr_med=g.cagr.median(), cagr_p5=g.cagr.quantile(.05),
                   dd_med=g.maxdd.median(), dd_p5=g.maxdd.quantile(.05), dd_worst=g.maxdd.min(),
                   p_dd50=(g.maxdd < -0.5).mean(), p_dd30=(g.maxdd < -0.3).mean(),
                   p_ruin=(g.final < init).mean(), uw_med=g.uw_days.median(),
                   taken_med=g.taken.median(), final_med=g.final.median())
        rows.append(row)
        print(f"{name}: 年化中位 {row['cagr_med']:+.1%} (5%分位 {row['cagr_p5']:+.1%}) | "
              f"回撤中位 {row['dd_med']:.1%} 尾部 {row['dd_p5']:.1%} 最差 {row['dd_worst']:.1%} | "
              f"P(>30%){row['p_dd30']:.0%} P(>50%){row['p_dd50']:.0%} | 亏损 {row['p_ruin']:.0%} | "
              f"成交 {row['taken_med']:.0f} | 水下 {row['uw_med']:.0f}天 | 期末中位 {row['final_med']:.0f}", flush=True)
        med = sorted(res, key=lambda r: r["maxdd"])[NSEED // 2]
        curves[name] = med["eq"]
        pd.DataFrame(rows).to_csv("drawdown_control2_results.csv", index=False)

    name = "0.5% + B禁下降态 + 最低佣金5元"
    eq = curves[name]
    y = eq.resample("YE").last()
    yr = y.pct_change(); yr.iloc[0] = y.iloc[0] / 10000 - 1
    print(f"\n{name} | 中位回撤路径按年：")
    print("  " + "  ".join(f"{d.year}:{r:+.0%}" for d, r in yr.items()))
    eq.to_csv("drawdown_control_recommended_eq.csv")
    pd.DataFrame(curves).to_csv("drawdown_control2_curves.csv")


if __name__ == "__main__":
    main()
