#!/usr/bin/env python3
"""阶梯双K主板回测严谨性审计（数据源：同花顺 DuckDB，主板 1063 只）

审计项：
1) 放量过滤的当日量未来函数：amt[t] 全日成交额在盘中突破成交时不可知 → 改昨日量复测
2) 涨停收盘日进场的成交可行性：一字/封板买不进 → 剔除后复测
3) 主板股票印花税：卖出 0.1%（2023-08-28 前）/ 0.05%（之后），原回测未计 → 逐笔精确补扣
4) 同一根K线同时触及止盈与止损的冲突单：引擎按止盈成交偏乐观 → 统计占比
5) 样本外复测：~2022 训练 / 2023~ 盲测
6) 组合模拟蒙特卡洛：同日信号顺序重抽（只成交 ~3% 信号的路径运气），
   修正原模拟两处近似（成本价算净值做仓位、同标的多笔出场价错配），
   另给出盘中最低价口径的悲观回撤上界
"""
import sys
import numpy as np
import pandas as pd
import ladder_doublek_backtest as m

STAMP_CUT = np.datetime64("2023-08-28")


def stats(T, stamp=False):
    if not len(T):
        return dict(trades=0)
    R = T["R"].to_numpy().copy()
    pnl = T["pnl"].to_numpy().copy()
    if stamp:
        rates = np.where(T["exit_date"].to_numpy() < STAMP_CUT, 0.001, 0.0005)
        cost = rates * T["exit"].to_numpy() * T["shares"].to_numpy()
        pnl = pnl - cost
        R = R - cost / T["risk_amt"].to_numpy()
    wins = int((R >= 0.1).sum())
    losses = int((pnl < 0).sum())
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    return dict(trades=len(T), wins=wins, losses=losses,
                win_rate=wins / max(1, wins + losses),
                pf=gp / gl if gl > 0 else np.inf,
                avg_R=R.mean(), sum_R=R.sum())


def fmt(name, s):
    if s.get("trades", 0) == 0:
        return f"{name}: 无交易"
    return (f"{name}: {s['trades']}笔 | 胜率 {s['win_rate']:.1%} | PF {s['pf']:.2f} | "
            f"平均R {s['avg_R']:+.3f} | 总R {s['sum_R']:+.0f}")


# ---------- 组合模拟（修正版 + 蒙特卡洛） ----------
def gen_signals(bundles, df, ma_n=60, vol_mult=1.5, rsi_in=30.0, vol_lag=0):
    regime3 = m.build_regime3(df, ma_n=ma_n)
    sigs, close_map, low_map = [], {}, {}
    for sym, dates, o, h, l, c, a, vmed in bundles:
        darr = pd.to_datetime(dates.to_numpy())
        close_map[sym] = pd.Series(c, index=darr)
        low_map[sym] = pd.Series(l, index=darr)
        up = (dates.map(regime3) == "up").fillna(False).to_numpy(dtype=bool)
        tr, _ = m.backtest_symbol(o, h, l, c, m.limit_pct(sym), amt=a, vmed=vmed,
                                  reg_ok=up, vol_mult=vol_mult, vol_lag=vol_lag,
                                  trail=True, trend_gate="none")
        if len(tr):
            tr = tr.copy(); tr["symbol"] = sym; tr["sys"] = "A"
            tr["entry_date"] = darr[tr["entry_i"].to_numpy()]
            tr["exit_date"] = darr[tr["exit_i"].to_numpy()]
            sigs.append(tr)
        tr2, _ = m.backtest_mr(o, h, l, c, m.limit_pct(sym), allow=~up, rsi_in=rsi_in)
        if len(tr2):
            tr2 = tr2.copy(); tr2["symbol"] = sym; tr2["sys"] = "B"
            tr2["entry_date"] = darr[tr2["entry_i"].to_numpy()]
            tr2["exit_date"] = darr[tr2["exit_i"].to_numpy()]
            sigs.append(tr2)
    return pd.concat(sigs, ignore_index=True), close_map, low_map


def simulate(S, C, L, cal, colidx, dayidx, seed=None, risk_pct=0.02,
             pos_cap=0.30, max_pos=6, init=10000.0):
    """S 信号表（entry_date/exit_date/entry/exit/stop/symbol/sys）。
    C/L 为日历×股票的收盘/最低价矩阵（已ffill）。seed=None 按 symbol 排序（复刻原版）。"""
    n = len(cal)
    E = S["entry"].to_numpy(); X = S["exit"].to_numpy(); ST = S["stop"].to_numpy()
    ei = S["entry_date"].map(dayidx).to_numpy()
    xi = S["exit_date"].map(dayidx).to_numpy()
    col = S["symbol"].map(colidx).to_numpy()
    stamp = np.where(cal < STAMP_CUT, 0.001, 0.0005)
    by_day = {}
    order = np.arange(len(S))
    if seed is not None:
        rng = np.random.default_rng(seed)
        for d in np.unique(ei):
            idx = order[ei == d]
            by_day[d] = rng.permutation(idx)
    else:
        srt = S.sort_values(["entry_date", "symbol"]).index.to_numpy()
        for d in np.unique(ei):
            by_day[d] = srt[ei[srt] == d]
    cash, init0 = init, init
    pos = []  # (i, shares)
    peak, maxdd, maxdd_low = init, 0.0, 0.0
    taken = skipped = 0
    for t in range(n):
        if pos:
            still = []
            for i, sh in pos:
                if xi[i] == t:
                    cash += sh * X[i] * (1 - stamp[t])
                else:
                    still.append((i, sh))
            pos = still
        for i in by_day.get(t, ()):
            if len(pos) >= max_pos or cash <= 0:
                skipped += 1; continue
            eq_now = cash
            t0 = max(t - 1, 0)  # 仓位按前一收盘净值估算，不用当日收盘
            for j, sh in pos:
                px = C[t0, col[j]]
                eq_now += sh * (px if np.isfinite(px) else E[j])
            risk = max(E[i] - ST[i], 1e-9)
            sh = risk_pct * eq_now / risk
            sh = min(sh, pos_cap * eq_now / E[i], cash / E[i])
            if sh * E[i] < 500:
                skipped += 1; continue
            cash -= sh * E[i]
            pos.append((i, sh))
            taken += 1
        eq = cash
        eq_low = cash
        for j, sh in pos:
            px = C[t, col[j]]
            eq += sh * (px if np.isfinite(px) else E[j])
            pxl = L[t, col[j]]
            eq_low += sh * (pxl if np.isfinite(pxl) else E[j])
        if eq > peak:
            peak = eq
        maxdd = min(maxdd, eq / peak - 1)
        maxdd_low = min(maxdd_low, eq_low / peak - 1)
    yrs = (cal[-1] - cal[0]).days / 365.25
    return dict(final=eq, cagr=(eq / init0) ** (1 / yrs) - 1,
                maxdd=maxdd, maxdd_low=maxdd_low, taken=taken, skipped=skipped)


def main():
    print("载入主板样本…", flush=True)
    syms = m.liquid_symbols(main_only=True)
    df, bundles = m.load_bundles(syms)
    regime = m.build_regime(df)
    print(f"样本 {len(syms)} 只，{len(df)} 行\n", flush=True)

    # ===== 1~4：变体复测 + 印花税 + 冲突单 =====
    rows = []
    configs = [("基线1:1", {}),
               ("放量1.5x", dict(vol_mult=1.5)),
               ("regime+放量+跟踪", dict(vol_mult=1.5, trail=True, use_reg=True))]
    variants = [("原版", {}), ("量改昨日", dict(vol_lag=1)),
                ("剔除涨停收盘进场", dict(no_limitup_entry=True))]
    for cname, ccfg in configs:
        for vname, vcfg in variants:
            if cname == "基线1:1" and vname == "量改昨日":
                continue
            cfg = dict(ccfg)
            use_reg = cfg.pop("use_reg", False)
            cfg.update(vcfg)
            T = m.run_pooled(bundles, regime=regime, use_reg=use_reg, **cfg)
            s0, s1 = stats(T), stats(T, stamp=True)
            conflict = int(T["conflict"].sum()) if len(T) and "conflict" in T else 0
            print(fmt(f"{cname} | {vname}", s0))
            print(fmt(f"{cname} | {vname} | +印花税", s1))
            if cname == "基线1:1" and vname == "原版":
                print(f"  同日止盈+止损冲突单 {conflict} 笔（{conflict/len(T):.2%}），"
                      f"若全按止损处理总R {s0['sum_R']-2*conflict:+.0f}")
            rows.append(dict(config=cname, variant=vname, conflict=conflict,
                             **{f"raw_{k}": v for k, v in s0.items()},
                             **{f"stamp_{k}": v for k, v in s1.items()}))
            pd.DataFrame(rows).to_csv("audit_variant_results.csv", index=False)
            print(flush=True)

    # ===== 5：样本外 =====
    print("===== 样本外（~2022 训练 / 2023~ 盲测，含印花税） =====", flush=True)
    for name, extra in [("regime+放量+跟踪 原版", {}),
                        ("regime+放量+跟踪 量改昨日", dict(vol_lag=1))]:
        T = m.run_pooled(bundles, regime=regime, use_reg=True,
                         vol_mult=1.5, trail=True, **extra)
        tr = T[T["exit_date"] <= "2022-12-31"]
        te = T[T["exit_date"] > "2022-12-31"]
        print(fmt(f"{name} | 训练段", stats(tr, stamp=True)))
        print(fmt(f"{name} | 盲测段", stats(te, stamp=True)), flush=True)

    # ===== 6：组合蒙特卡洛 =====
    print("\n===== 组合模拟（修正版：市值计价仓位 + 印花税 + 真实交易日历） =====", flush=True)
    S, close_map, low_map = gen_signals(bundles, df)
    print(f"信号 {len(S)}（A:{(S.sys=='A').sum()} B:{(S.sys=='B').sum()}）", flush=True)
    close_df = pd.DataFrame(close_map).sort_index().ffill()
    low_df = pd.DataFrame(low_map).reindex(close_df.index).ffill()
    cal = close_df.index
    C = close_df.to_numpy(); L = low_df.to_numpy()
    colidx = {s: i for i, s in enumerate(close_df.columns)}
    dayidx = {d: i for i, d in enumerate(cal)}
    base = simulate(S, C, L, cal, colidx, dayidx, seed=None)
    print(f"复刻原版顺序: 期末 {base['final']:.0f} | 年化 {base['cagr']:+.1%} | "
          f"收盘回撤 {base['maxdd']:.1%} | 盘中悲观回撤 {base['maxdd_low']:.1%} | 成交 {base['taken']}", flush=True)

    mc_rows = []
    grids = [("基准 2%风险/6持仓/30%上限", dict(), 100),
             ("风险1%", dict(risk_pct=0.01), 30),
             ("最多4持仓", dict(max_pos=4), 30),
             ("最多8持仓", dict(max_pos=8), 30),
             ("单票上限20%", dict(pos_cap=0.20), 30)]
    for gname, gkw, nseed in grids:
        for seed in range(nseed):
            r = simulate(S, C, L, cal, colidx, dayidx, seed=seed, **gkw)
            mc_rows.append(dict(grid=gname, seed=seed, **r))
        g = pd.DataFrame([r for r in mc_rows if r["grid"] == gname])
        print(f"\n{gname}（{nseed} 次重抽）:")
        for k in ["cagr", "maxdd", "maxdd_low", "final"]:
            v = g[k]
            print(f"  {k}: 中位 {v.median():+.3f} | 5%分位 {v.quantile(.05):+.3f} | "
                  f"95%分位 {v.quantile(.95):+.3f} | 最差 {v.min():+.3f}" if k != "final" else
                  f"  final: 中位 {v.median():.0f} | 5% {v.quantile(.05):.0f} | 95% {v.quantile(.95):.0f} | 最差 {v.min():.0f}")
        print(f"  回撤超50%概率: {(g.maxdd < -0.5).mean():.0%} | 亏损收场概率: {(g.final < 10000).mean():.0%}", flush=True)
        pd.DataFrame(mc_rows).to_csv("audit_mc_results.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
