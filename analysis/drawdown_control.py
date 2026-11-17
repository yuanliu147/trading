#!/usr/bin/env python3
"""主板双系统组合层回撤控制方案对比（蒙特卡洛，含印花税，市值计价仓位）

信号与逐笔引擎不变，只测组合层风控叠加：
- 单笔风险 2% / 1% / 0.5%
- 总敞口上限（持仓市值/净值）
- 大盘下降态：B系统停做 / 全部停开新仓（信号生成时的大盘状态，T-1 收盘判定）
- 净值 < 自身60日均线 → 停开新仓（持仓正常离场）
- 回撤分级：回撤>15% 风险减半，>25% 停开新仓
每配置 30 次同日信号顺序重抽，报告中位数与 5% 尾部。
"""
import numpy as np
import pandas as pd
import ladder_doublek_backtest as m
from backtest_audit import gen_signals, STAMP_CUT


def simulate(S, C, cal, colidx, dayidx, seed, state_arr=None,
             risk_pct=0.02, pos_cap=0.30, max_pos=6, init=10000.0,
             max_exposure=None, eq_ma_n=None, dd_throttle=False,
             block_b_down=False, block_all_down=False, store_eq=False,
             min_fee=0.0):
    n = len(cal)
    E = S["entry"].to_numpy(); X = S["exit"].to_numpy(); ST = S["stop"].to_numpy()
    ei = S["entry_date"].map(dayidx).to_numpy()
    xi = S["exit_date"].map(dayidx).to_numpy()
    col = S["symbol"].map(colidx).to_numpy()
    is_b = (S["sys"] == "B").to_numpy()
    stamp = np.where(cal < STAMP_CUT, 0.001, 0.0005)
    rng = np.random.default_rng(seed)
    by_day = {d: rng.permutation(np.flatnonzero(ei == d)) for d in np.unique(ei)}
    cash, peak = init, init
    pos = []
    maxdd, uw_days, uw_max = 0.0, 0, 0
    taken = 0
    eqs = []
    eq_curve = {}
    for t in range(n):
        if pos:
            still = []
            for i, sh in pos:
                if xi[i] == t:
                    cash += sh * X[i] * (1 - stamp[t]) - max(0.0, min_fee - sh * X[i] * 0.0001)
                else:
                    still.append((i, sh))
            pos = still
        # 风控状态（用截至昨日的净值，不用当日）
        risk_mult, allow_new = 1.0, True
        if eqs:
            if eq_ma_n is not None and len(eqs) >= eq_ma_n and eqs[-1] < np.mean(eqs[-eq_ma_n:]):
                allow_new = False
            if dd_throttle:
                dd_now = eqs[-1] / max(max(eqs), init) - 1
                if dd_now < -0.25:
                    allow_new = False
                elif dd_now < -0.15:
                    risk_mult = 0.5
        for i in by_day.get(t, ()):
            if len(pos) >= max_pos or cash <= 0 or not allow_new:
                continue
            if state_arr is not None and t >= 1:
                st = state_arr[t - 1]
                if block_all_down and st == "down":
                    continue
                if block_b_down and is_b[i] and st == "down":
                    continue
            eq_now = cash
            t0 = max(t - 1, 0)
            expo_now = 0.0
            for j, sh in pos:
                px = C[t0, col[j]]
                v = sh * (px if np.isfinite(px) else E[j])
                eq_now += v
                expo_now += v
            risk = max(E[i] - ST[i], 1e-9)
            sh = risk_pct * risk_mult * eq_now / risk
            sh = min(sh, pos_cap * eq_now / E[i], cash / E[i])
            if max_exposure is not None:
                sh = min(sh, max(0.0, max_exposure * eq_now - expo_now) / E[i])
            if sh * E[i] < 500:
                continue
            cash -= sh * E[i] + max(0.0, min_fee - sh * E[i] * 0.0001)
            pos.append((i, sh))
            taken += 1
        eq = cash
        for j, sh in pos:
            px = C[t, col[j]]
            eq += sh * (px if np.isfinite(px) else E[j])
        eqs.append(eq)
        if store_eq:
            eq_curve[cal[t]] = eq
        if eq > peak:
            peak = eq
        maxdd = min(maxdd, eq / peak - 1)
        if eq < peak - 1e-9:
            uw_days += 1
            uw_max = max(uw_max, uw_days)
        else:
            uw_days = 0
    yrs = (cal[-1] - cal[0]).days / 365.25
    out = dict(final=eq, cagr=(eq / init) ** (1 / yrs) - 1, maxdd=maxdd,
               uw_days=uw_max, taken=taken)
    if store_eq:
        out["eq"] = pd.Series(eq_curve)
    return out


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
    regime3 = pd.Series(m.build_regime3(df))
    state_arr = regime3.reindex(cal).ffill().to_numpy()
    print(f"信号 {len(S)}，日历 {len(cal)} 天\n", flush=True)

    configs = [
        ("基准 2%风险", dict()),
        ("风险1%", dict(risk_pct=0.01)),
        ("风险0.5%", dict(risk_pct=0.005)),
        ("1% + 总敞口≤50%", dict(risk_pct=0.01, max_exposure=0.5)),
        ("1% + B不做下降态", dict(risk_pct=0.01, block_b_down=True)),
        ("1% + 下降态全停新仓", dict(risk_pct=0.01, block_all_down=True)),
        ("1% + 净值<60日线停新仓", dict(risk_pct=0.01, eq_ma_n=60)),
        ("1% + 回撤分级降风险", dict(risk_pct=0.01, dd_throttle=True)),
        ("1% + B禁下降态 + 净值60日过滤", dict(risk_pct=0.01, block_b_down=True, eq_ma_n=60)),
        ("0.5% + B禁下降态 + 净值60日过滤", dict(risk_pct=0.005, block_b_down=True, eq_ma_n=60)),
        ("2% + B禁下降态 + 净值60日过滤", dict(risk_pct=0.02, block_b_down=True, eq_ma_n=60)),
    ]
    NSEED = 30
    rows = []
    for name, kw in configs:
        res = [simulate(S, C, cal, colidx, dayidx, seed, state_arr=state_arr, **kw)
               for seed in range(NSEED)]
        g = pd.DataFrame([{k: v for k, v in r.items() if k != "eq"} for r in res])
        row = dict(config=name,
                   cagr_med=g.cagr.median(), cagr_p5=g.cagr.quantile(.05),
                   dd_med=g.maxdd.median(), dd_p5=g.maxdd.quantile(.05),
                   dd_worst=g.maxdd.min(),
                   p_dd50=(g.maxdd < -0.5).mean(), p_dd30=(g.maxdd < -0.3).mean(),
                   p_ruin=(g.final < 10000).mean(),
                   uw_med=g.uw_days.median(), taken_med=g.taken.median())
        rows.append(row)
        print(f"{name}: 年化中位 {row['cagr_med']:+.1%} (5%分位 {row['cagr_p5']:+.1%}) | "
              f"回撤中位 {row['dd_med']:.1%} 尾部5% {row['dd_p5']:.1%} 最差 {row['dd_worst']:.1%} | "
              f"P(回撤>50%) {row['p_dd50']:.0%} P(回撤>30%) {row['p_dd30']:.0%} | "
              f"亏损概率 {row['p_ruin']:.0%} | 水下中位 {row['uw_med']:.0f}天", flush=True)
        pd.DataFrame(rows).to_csv("drawdown_control_results.csv", index=False)

    # 代表配置的中位路径按年收益
    for name, kw in [configs[8], configs[9]]:
        res = [simulate(S, C, cal, colidx, dayidx, seed, state_arr=state_arr,
                        store_eq=True, **kw) for seed in range(NSEED)]
        med = sorted(res, key=lambda r: r["maxdd"])[NSEED // 2]
        eq = med["eq"]
        y = eq.resample("YE").last()
        yr = y.pct_change(); yr.iloc[0] = y.iloc[0] / 10000 - 1
        print(f"\n{name} | 中位回撤路径按年：")
        print("  " + "  ".join(f"{d.year}:{r:+.0%}" for d, r in yr.items()), flush=True)


if __name__ == "__main__":
    main()
