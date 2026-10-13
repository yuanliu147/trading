#!/usr/bin/env python3
"""组合层资金模拟：把全市场信号流放进一个 1 万元的真实账户

规则：
- 本金 1 万；每笔风险 = 当前净值 2%；单票仓位上限 30%；同时最多 6 个持仓
- 现金不够 → 放弃该信号（信号流来自逐标的独立回测，跳过信号不改变已有持仓路径，
  属于近似，但方向保守：真实信号只会更多不会更少）
- 子系统：A=趋势突破(大盘上升态, 无均线门, 放量1.5x, 跟踪止盈)
          B=超跌反弹(非上升态, RSI<30, MA10/10日/-8% 出场)
- 价格用逐标的回测记录的进出场价（已含滑点手续费）
"""
import numpy as np
import pandas as pd
import ladder_doublek_backtest as m

INIT = 10000.0
RISK = 0.02
POS_CAP = 0.30
MAX_POS = 6

def gen_trades():
    syms = m.liquid_symbols()
    df, bundles = m.load_bundles(syms)
    regime3 = m.build_regime3(df)
    trades = []
    close_map = {}
    for sym, dates, o, h, l, c, a, vmed in bundles:
        darr = dates.to_numpy()
        close_map[sym] = pd.Series(c, index=pd.to_datetime(darr))
        up = (dates.map(regime3) == "up").fillna(False).to_numpy(dtype=bool)
        tr, _ = m.backtest_symbol(o, h, l, c, m.limit_pct(sym), amt=a, vmed=vmed,
                                  reg_ok=up, vol_mult=1.5, trail=True, trend_gate="none")
        if len(tr):
            tr = tr.copy(); tr["symbol"] = sym; tr["sys"] = "A"
            tr["entry_date"] = pd.to_datetime(darr[tr["entry_i"].to_numpy()])
            tr["exit_date"] = pd.to_datetime(darr[tr["exit_i"].to_numpy()])
            trades.append(tr)
        tr2, _ = m.backtest_mr(o, h, l, c, m.limit_pct(sym), allow=~up)
        if len(tr2):
            tr2 = tr2.copy(); tr2["symbol"] = sym; tr2["sys"] = "B"
            tr2["entry_date"] = pd.to_datetime(darr[tr2["entry_i"].to_numpy()])
            tr2["exit_date"] = pd.to_datetime(darr[tr2["exit_i"].to_numpy()])
            trades.append(tr2)
    T = pd.concat(trades, ignore_index=True)
    return T, close_map

def simulate(T, close_map):
    T = T.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    by_day_in = {d: g for d, g in T.groupby("entry_date")}
    by_day_out = {d: g for d, g in T.groupby("exit_date")}
    days = pd.bdate_range(T["entry_date"].min(), T["exit_date"].max())
    days = [d for d in days if d in set(T["entry_date"]) | set(T["exit_date"]) or True]

    cash, equity = INIT, INIT
    positions = []  # dict: symbol, shares, entry, exit_px, exit_date
    eq_curve = {}
    taken = skipped = 0
    for d in days:
        # 先处理出场（按记录的出场价成交）
        out_today = by_day_out.get(d)
        if out_today is not None:
            syms_out = set(out_today["symbol"])
            still = []
            for p in positions:
                if p["symbol"] in syms_out:
                    row = out_today[out_today["symbol"] == p["symbol"]].iloc[-1]
                    cash += p["shares"] * row["exit"]
                else:
                    still.append(p)
            positions = still
        # 再处理进场
        in_today = by_day_in.get(d)
        if in_today is not None:
            for _, r in in_today.iterrows():
                if len(positions) >= MAX_POS or cash <= 0:
                    skipped += 1; continue
                risk = max(r["entry"] - r["stop"], 1e-9)
                eq_now = cash + sum(p["shares"] * p["entry"] for p in positions)
                sh = RISK * eq_now / risk
                sh = min(sh, POS_CAP * eq_now / r["entry"], cash / r["entry"])
                if sh * r["entry"] < 500:   # 太小不做
                    skipped += 1; continue
                cash -= sh * r["entry"]
                positions.append({"symbol": r["symbol"], "shares": sh,
                                  "entry": r["entry"]})
                taken += 1
        # 收盘按市价估值
        mv = 0.0
        for p in positions:
            s = close_map[p["symbol"]]
            px = s.get(d)
            if px is None:
                px = s[s.index <= d].iloc[-1] if (s.index <= d).any() else p["entry"]
            mv += p["shares"] * px
        equity = cash + mv
        eq_curve[d] = equity
    eq = pd.Series(eq_curve)
    return eq, taken, skipped

def main():
    T, close_map = gen_trades()
    print(f"信号总数 {len(T)}（A:{(T.sys=='A').sum()} B:{(T.sys=='B').sum()}）")
    eq, taken, skipped = simulate(T, close_map)
    print(f"成交 {taken} 笔，因仓位/现金不足放弃 {skipped} 笔")
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    peak = eq.cummax()
    print(f"\n===== 1万本金组合模拟 {eq.index[0].date()} ~ {eq.index[-1].date()} =====")
    print(f"期末净值 {eq.iloc[-1]:.0f} 元 | 总收益 {eq.iloc[-1]/INIT-1:+.1%} | "
          f"年化 {(eq.iloc[-1]/INIT)**(1/yrs)-1:+.1%} | 最大回撤 {(eq/peak-1).min():.1%}")
    print("\n按年收益：")
    y = eq.resample("YE").last()
    yr = y.pct_change(); yr.iloc[0] = y.iloc[0] / INIT - 1
    for d, r in yr.items():
        print(f"  {d.year}: {r:+8.1%}   (年末 {y[d]:.0f} 元)")
    print("\n按月收益表（%）：")
    mth = eq.resample("ME").last().pct_change() * 100
    mth.iloc[0] = (y.iloc[0] / INIT - 1) * 100 if len(y) else 0
    mt = mth.to_frame("ret")
    mt["Y"] = mt.index.year; mt["M"] = mt.index.month
    piv = mt.pivot_table(index="Y", columns="M", values="ret").round(1)
    print(piv.to_string())
    eq.to_csv("portfolio_equity_1w.csv")

if __name__ == "__main__":
    main()
