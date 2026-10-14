#!/usr/bin/env python3
"""MACD 金叉买入后逐日监控：出现"走弱"特征就提前卖出，是否优于等死叉

每个持仓日收盘检查规则，触发则次日开盘卖出（一字跌停顺延）；死叉仍然照常卖出。
严格版：提前卖出后空仓，等下一次金叉。
买回版：提前卖出后，若仍在金叉区间（DIF>DEA）且红柱重新放大，次日开盘买回。
股票池、数据、费用、1 万本金全仓复利口径与 MACD金叉死叉研究.py 完全一致。
"""
import numpy as np, pandas as pd, duckdb
from MACD金叉死叉研究 import pick_leaders, prep, stamp, perf, DB, HERE, BASE, CAPITAL, COMM, WARM, SPLIT

INF = float("inf")


def arrays(df):
    A = {k: df[k].values for k in ("open", "high", "low", "close", "dif", "dea", "hist")}
    A["date"] = df.date.values
    dif, dea, hist, c = A["dif"], A["dea"], A["hist"], A["close"]
    A["golden"] = (dif > dea) & (np.roll(dif, 1) <= np.roll(dea, 1))
    A["death"] = (dif < dea) & (np.roll(dif, 1) >= np.roll(dea, 1))
    s = np.diff(dif, prepend=np.nan)
    A["slope"] = s
    h1, h2, h3 = (np.roll(hist, k) for k in (1, 2, 3))
    s1, s2, s3 = (np.roll(s, k) for k in (1, 2, 3))
    A["hs1"] = hist < h1
    A["hs2"] = A["hs1"] & (h1 < h2)
    A["hs3"] = A["hs2"] & (h2 < h3)
    A["ss1"] = s < s1
    A["ss2"] = A["ss1"] & (s1 < s2)
    A["ss3"] = A["ss2"] & (s2 < s3)
    A["dif_down"] = s < 0
    A["hist_up"] = (hist > h1) & (hist > 0)
    cs = pd.Series(c)
    A["below_ma5"] = c < cs.rolling(5).mean().values
    A["below_ma10"] = c < cs.rolling(10).mean().values
    return A


RULES = [
    ("红柱缩短1天", lambda A, t, p: A["hs1"][t]),
    ("红柱连缩2天", lambda A, t, p: A["hs2"][t]),
    ("红柱连缩3天", lambda A, t, p: A["hs3"][t]),
    ("DIF斜率放缓1天", lambda A, t, p: A["ss1"][t]),
    ("DIF斜率连缓2天", lambda A, t, p: A["ss2"][t]),
    ("DIF斜率连缓3天", lambda A, t, p: A["ss3"][t]),
    ("DIF拐头向下", lambda A, t, p: A["dif_down"][t]),
    ("DIF斜率<持仓期峰值1/2", lambda A, t, p: p["peak_s"] > 0 and A["slope"][t] < 0.5 * p["peak_s"]),
    ("红柱较峰值回落1/3", lambda A, t, p: p["peak_h"] > 0 and A["hist"][t] <= p["peak_h"] * 2 / 3),
    ("红柱较峰值回落1/2", lambda A, t, p: p["peak_h"] > 0 and A["hist"][t] <= p["peak_h"] / 2),
    ("收盘跌破MA5", lambda A, t, p: A["below_ma5"][t]),
    ("收盘跌破MA10", lambda A, t, p: A["below_ma10"][t]),
    ("收盘浮盈曾≥5%后红柱连缩2天", lambda A, t, p: p["mfe"] >= 0.05 and A["hs2"][t]),
    ("收盘浮盈曾≥5%后DIF斜率连缓2天", lambda A, t, p: p["mfe"] >= 0.05 and A["ss2"][t]),
    ("收盘浮盈曾≥10%后红柱较峰值回落1/2",
     lambda A, t, p: p["mfe"] >= 0.10 and p["peak_h"] > 0 and A["hist"][t] <= p["peak_h"] / 2),
    ("对照:持有满5天就卖", lambda A, t, p: t - p["e"] + 1 >= 5),
    ("对照:持有满10天就卖", lambda A, t, p: t - p["e"] + 1 >= 10),
]


def update(A, t, p):
    p["peak_h"] = max(p["peak_h"], A["hist"][t])
    p["peak_s"] = max(p["peak_s"], A["slope"][t])
    p["mfe"] = max(p["mfe"], A["close"][t] / p["px"] - 1)


def simulate(A, rule=None, reentry=False):
    o, h, l, c, dates = A["open"], A["high"], A["low"], A["close"], A["date"]
    n = len(o)
    cash, sh, p, pend, phase = CAPITAL, 0.0, None, None, False
    eq, held, trades = np.empty(n), np.zeros(n, bool), []
    for t in range(n):
        if pend == "sell":
            if not (h[t] == l[t] and o[t] < c[t - 1]):
                proceeds = sh * o[t]
                cash = proceeds - max(5.0, proceeds * COMM) - proceeds * stamp(pd.Timestamp(dates[t]))
                trades.append(dict(e=p["e"], d=p["d"], x=t, px=p["px"], exit_px=o[t],
                                   ret=cash / p["cash0"] - 1, why=p["why"], kind=p["kind"],
                                   date=pd.Timestamp(dates[p["e"]])))
                sh, p, pend = 0.0, None, None
        elif pend in ("buy", "rebuy"):
            kind, pend = pend, None
            if not (h[t] == l[t] and o[t] > c[t - 1]):
                p = dict(e=t, px=o[t], cash0=cash, peak_h=-INF, peak_s=-INF, mfe=-INF, kind=kind)
                sh, cash = (cash - max(5.0, cash * COMM / (1 + COMM))) / o[t], 0.0
        eq[t], held[t] = cash + sh * c[t], p is not None
        if t < WARM or t >= n - 1:
            continue
        if A["golden"][t]:
            phase = True
        if A["death"][t]:
            phase = False
        if pend is not None:
            continue
        if p is not None:
            update(A, t, p)
            if A["death"][t]:
                pend, p["why"], p["d"] = "sell", "death", t
            elif rule is not None and rule(A, t, p):
                pend, p["why"], p["d"] = "sell", "rule", t
        elif A["golden"][t]:
            pend = "buy"
        elif reentry and phase and A["hist_up"][t]:
            pend = "rebuy"
    return eq, held, trades


def stock_metrics(A, eq, held):
    e = eq[WARM:]
    k = np.searchsorted(pd.to_datetime(A["date"][WARM:]), SPLIT) - 1
    return dict(final=e[-1], mdd=(e / np.maximum.accumulate(e) - 1).min(),
                expo=held[WARM:].mean(), g1=e[k] / CAPITAL, g2=e[-1] / e[k])


def first_trigger(A, rule, tr):
    p = dict(e=tr["e"], px=tr["px"], peak_h=-INF, peak_s=-INF, mfe=-INF)
    for t in range(tr["e"], tr["d"]):
        update(A, t, p)
        if rule(A, t, p):
            return t
    return None


def main():
    pd.set_option("display.width", 250, "display.max_columns", 40)
    con = duckdb.connect(DB, read_only=True)
    lead, _ = pick_leaders(con)
    raw = con.execute("""SELECT thscode, date, open, high, low, close, amount FROM v_daily_qfq
                         WHERE thscode IN (SELECT UNNEST(?::VARCHAR[])) ORDER BY thscode, date""",
                      [lead.thscode.tolist()]).fetchdf()
    con.close()
    raw["date"] = pd.to_datetime(raw.date)
    hs = pd.read_csv(f"{BASE}/hs300_daily.csv", parse_dates=["date"])[["date", "close"]]
    hs["hs_below_ma60"] = hs.close < hs.close.rolling(60).mean()
    hs = hs[["date", "hs_below_ma60"]]
    AA = {code: arrays(prep(df.reset_index(drop=True), hs)) for code, df in raw.groupby("thscode")}

    # ---------- 基准 ----------
    base = {}
    for code, A in AA.items():
        eq, held, tr = simulate(A)
        base[code] = (stock_metrics(A, eq, held), tr)
    ref = pd.read_csv(f"{HERE}/MACD金叉死叉个股汇总.csv").set_index("thscode").final_equity
    diff = max(abs(base[c][0]["final"] - ref[c]) for c in AA)
    ntr = sum(len(base[c][1]) for c in AA)
    print(f"基准复核：已平仓 {ntr} 笔，与上次回测期末权益最大差 {diff:.6f} 元")
    assert diff < 1e-6 and ntr == 1724

    # ---------- 一、信号符合率：在基准交易上，第一次触发就卖 vs 等死叉卖 ----------
    rows = []
    for name, rule in RULES:
        recs = []
        for code, A in AA.items():
            for tr in base[code][1]:
                t = first_trigger(A, rule, tr)
                base_g = tr["exit_px"] / tr["px"] - 1
                if t is None:
                    recs.append((tr["ret"] > 0, False, 0.0, np.nan, tr["date"] < SPLIT))
                else:
                    early_g = A["open"][t + 1] / tr["px"] - 1
                    recs.append((tr["ret"] > 0, True, early_g - base_g, tr["x"] - t - 1, tr["date"] < SPLIT))
        R = pd.DataFrame(recs, columns=["win", "fired", "gain", "days_early", "h1"])
        F, W, Lz = R[R.fired], R.win, ~R.win
        rows.append({
            "规则": name, "触发率": R.fired.mean(),
            "亏损单触发率": R.fired[Lz].mean(), "盈利单触发率": R.fired[W].mean(),
            "符合率(提前卖更好)": (F.gain > 0).mean(),
            "亏损单符合率": (F.gain[~F.win] > 0).mean(), "盈利单符合率": (F.gain[F.win] > 0).mean(),
            "亏损单平均改善": F.gain[~F.win].mean(), "盈利单平均改善": F.gain[F.win].mean(),
            "每笔净改善": R.gain.mean(),
            "前段每笔净改善": R.gain[R.h1].mean(), "后段每笔净改善": R.gain[~R.h1].mean(),
            "提前天数中位": F.days_early.median(),
        })
    TR = pd.DataFrame(rows)
    TR.to_csv(f"{HERE}/MACD提前退出信号符合率.csv", index=False)
    print("\n=== 一、逐日监控信号的符合率（基准 1724 笔交易；持仓中第一次触发就次日开盘卖，对比等死叉卖）===")
    print("符合率 = 触发后提前卖出价 > 死叉卖出价 的比例；改善为价格收益差（正=提前卖更好）")
    print(TR.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---------- 二、账户级回测 ----------
    bm = pd.DataFrame({c: base[c][0] for c in AA}).T
    bt = pd.DataFrame([t for c in AA for t in base[c][1]])
    out = []

    def summarize(name, mode, M, T):
        p = perf(T.ret)
        out.append({
            "规则": name, "模式": mode, "笔数": len(T), "胜率": p["win_rate"], "盈亏比": p["payoff"],
            "PF": p["pf"], "每笔均收益": p["avg_ret"], "持仓时间占比": M.expo.mean(),
            "期末中位数": M.final.median(), "期末均值": M.final.mean(),
            "胜基准只数": int((M.final > bm.final + 1e-9).sum()),
            "前段胜基准": int((M.g1 > bm.g1 + 1e-9).sum()), "后段胜基准": int((M.g2 > bm.g2 + 1e-9).sum()),
            "前段增长中位": M.g1.median(), "后段增长中位": M.g2.median(), "MDD中位": M.mdd.median(),
        })

    summarize("基准:等死叉", "-", bm, bt)
    for name, rule in RULES:
        modes = [("严格", False)] + ([] if name.startswith("对照") else [("买回", True)])
        for mode, re in modes:
            ms, ts = {}, []
            for code, A in AA.items():
                eq, held, tr = simulate(A, rule, re)
                ms[code] = stock_metrics(A, eq, held)
                ts += tr
            summarize(name, mode, pd.DataFrame(ms).T, pd.DataFrame(ts))
    O = pd.DataFrame(out)
    O.to_csv(f"{HERE}/MACD提前退出账户对比.csv", index=False)
    print("\n=== 二、账户级回测（20 只各 1 万全仓复利；严格=提前卖后等下次金叉；买回=金叉区间内红柱再放大就买回）===")
    print(f"前段=开始~2021-12-31 权益增长倍数，后段=2022-01-01~2026-10-13；胜基准只数满分 20")
    print(O.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\n输出: MACD提前退出信号符合率.csv({len(TR)}行) / MACD提前退出账户对比.csv({len(O)}行)")


if __name__ == "__main__":
    main()
