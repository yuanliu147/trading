#!/usr/bin/env python3
"""MACD 预判金叉：DIF 还在 DEA 下方但已足够接近时提前买入，对比等收盘确认金叉再买

预判买入（金叉尚未发生，DIF<DEA）：
- 绿柱型：绿柱正在缩短，且已缩到本段绿柱最长时的 1/2、1/3、1/5 以内 -> 次日开盘买
- 金叉价型：由 EMA 递推反解"明天收盘多少钱会金叉"（金叉价 P*），
  明天所需涨幅 <= 0%/1%/2%/3% -> 次日开盘买
- 条件单型：每天收盘后挂"股价 >= P* 买入"条件单，次日盘中触及即按 max(开盘, P*) 成交
预判失败退出：买入后还没金叉、绿柱又放大（1 天 / 连续 2 天）-> 次日开盘卖
已金叉后：与原系统相同，死叉次日开盘卖。若某次金叉未被预判到，仍按原规则确认后买入。
股票池、数据、费用、1 万本金全仓复利口径与 MACD金叉死叉研究.py 一致。
"""
import numpy as np, pandas as pd, duckdb
from MACD金叉死叉研究 import pick_leaders, prep, stamp, perf, DB, HERE, BASE, CAPITAL, COMM, WARM, SPLIT
from MACD持仓退出研究 import arrays, stock_metrics

K = 2 / 13 - 2 / 27


def add_anticipation(A):
    c = pd.Series(A["close"])
    e12 = c.ewm(span=12, adjust=False).mean().values
    e26 = c.ewm(span=26, adjust=False).mean().values
    A["pstar"] = (A["dea"] - 11 / 13 * e12 + 25 / 27 * e26) / K
    A["need"] = A["pstar"] / A["close"] - 1
    hist = pd.Series(A["hist"])
    neg = hist < 0
    A["trough"] = hist.where(neg).groupby((neg != neg.shift()).cumsum()).cummin().values
    A["green_shrink"] = neg.values & (A["hist"] > np.roll(A["hist"], 1))
    return A


SIGNALS = [
    ("绿柱缩到最长时1/2内", "next", lambda A, t: A["green_shrink"][t] and A["hist"][t] >= A["trough"][t] / 2),
    ("绿柱缩到最长时1/3内", "next", lambda A, t: A["green_shrink"][t] and A["hist"][t] >= A["trough"][t] / 3),
    ("绿柱缩到最长时1/5内", "next", lambda A, t: A["green_shrink"][t] and A["hist"][t] >= A["trough"][t] / 5),
    ("明天不涨也会金叉", "next", lambda A, t: A["need"][t] <= 0),
    ("明天涨1%内就金叉", "next", lambda A, t: A["need"][t] <= 0.01),
    ("明天涨2%内就金叉", "next", lambda A, t: A["need"][t] <= 0.02),
    ("明天涨3%内就金叉", "next", lambda A, t: A["need"][t] <= 0.03),
    ("条件单:盘中触及金叉价即买", "stop", None),
]
FAILS = [("绿柱放大1天就卖", 1), ("绿柱连放大2天才卖", 2)]


def simulate(A, sig=None, mode=None, fail_n=1):
    o, h, l, c, dates = A["open"], A["high"], A["low"], A["close"], A["date"]
    dif, dea, hist = A["dif"], A["dea"], A["hist"]
    n = len(o)
    cash, sh, p, pend = CAPITAL, 0.0, None, None
    eq, held, trades = np.empty(n), np.zeros(n, bool), []
    for t in range(n):
        if pend is not None and pend[0] == "sell":
            if not (h[t] == l[t] and o[t] < c[t - 1]):
                proceeds = sh * o[t]
                cash = proceeds - max(5.0, proceeds * COMM) - proceeds * stamp(pd.Timestamp(dates[t]))
                trades.append(dict(e=p["e"], x=t, px=p["px"], ret=cash / p["cash0"] - 1, kind=p["kind"],
                                   crossed=p["d"] is not None, d=p["d"], why=p["why"],
                                   date=pd.Timestamp(dates[p["e"]])))
                sh, p, pend = 0.0, None, None
        elif pend is not None:
            (kind, price), pend = pend, None
            if not (h[t] == l[t] and o[t] > c[t - 1]):
                fill = o[t] if kind != "stop" else (max(o[t], price) if h[t] >= price else None)
                if fill is not None:
                    p = dict(e=t, px=fill, cash0=cash, kind=kind,
                             d=t - 1 if kind == "confirm" else None, why=None)
                    sh, cash = (cash - max(5.0, cash * COMM / (1 + COMM))) / fill, 0.0
        eq[t], held[t] = cash + sh * c[t], p is not None
        if t < WARM or t >= n - 1 or pend is not None:
            continue
        if p is not None:
            if p["d"] is None and dif[t] > dea[t]:
                p["d"] = t
            if p["d"] is not None:
                if A["death"][t]:
                    pend, p["why"] = ("sell", None), "death"
            elif all(hist[t - k] < hist[t - k - 1] for k in range(fail_n)):
                pend, p["why"] = ("sell", None), "fail"
        elif A["golden"][t]:
            pend = ("confirm", None)
        elif mode is not None and hist[t] < 0:
            if mode == "stop":
                pend = ("stop", A["pstar"][t])
            elif sig(A, t):
                pend = ("anti", None)
    return eq, held, trades


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
    AA = {code: add_anticipation(arrays(prep(df.reset_index(drop=True), hs)))
          for code, df in raw.groupby("thscode")}

    # 复核金叉价公式：DIF<DEA 的日子里，"次日收盘 > P*" 应与 "次日金叉" 完全一致
    tot = bad = 0
    for A in AA.values():
        m = np.where(A["hist"][WARM:-1] < 0)[0] + WARM
        tot += len(m)
        bad += int(((A["close"][m + 1] > A["pstar"][m]) != A["golden"][m + 1]).sum())
    print(f"金叉价公式复核：{tot} 个绿柱日，次日是否金叉的预测不一致 {bad} 个")
    assert bad <= tot * 1e-3

    base = {}
    for code, A in AA.items():
        eq, held, tr = simulate(A)
        base[code] = stock_metrics(A, eq, held), tr
    ref = pd.read_csv(f"{HERE}/MACD金叉死叉个股汇总.csv").set_index("thscode").final_equity
    diff = max(abs(base[c][0]["final"] - ref[c]) for c in AA)
    ntr = sum(len(base[c][1]) for c in AA)
    print(f"基准复核：已平仓 {ntr} 笔，与上次回测期末权益最大差 {diff:.6f} 元")
    assert diff < 1e-6 and ntr == 1724
    bm = pd.DataFrame({c: base[c][0] for c in AA}).T
    bt = pd.DataFrame([t for c in AA for t in base[c][1]])

    q_rows, a_rows = [], []

    def account(name, fail, M, T):
        pf = perf(T.ret)
        r = np.log(M.final / bm.final)
        sd = r.std(ddof=1)
        a_rows.append({
            "预判规则": name, "失败退出": fail, "笔数": len(T), "胜率": pf["win_rate"], "盈亏比": pf["payoff"],
            "PF": pf["pf"], "每笔均收益": pf["avg_ret"], "持仓时间占比": M.expo.mean(),
            "期末中位数": M.final.median(), "期末均值": M.final.mean(),
            "逐只期末/基准(几何均值)": np.exp(r.mean()), "t值": r.mean() / (sd / np.sqrt(len(r))) if sd > 0 else np.nan,
            "胜基准只数": int((M.final > bm.final + 1e-9).sum()),
            "前段胜基准": int((M.g1 > bm.g1 + 1e-9).sum()), "后段胜基准": int((M.g2 > bm.g2 + 1e-9).sum()),
            "前段增长中位": M.g1.median(), "后段增长中位": M.g2.median(), "MDD中位": M.mdd.median(),
        })

    account("基准:收盘确认金叉,次日开盘买", "-", bm, bt)
    for name, mode, sig in SIGNALS:
        for fname, fn in FAILS:
            ms, ts, recs = {}, [], []
            for code, A in AA.items():
                eq, held, tr = simulate(A, sig, mode, fn)
                ms[code] = stock_metrics(A, eq, held)
                ts += tr
                for r in tr:
                    if r["kind"] in ("anti", "stop"):
                        d = r["d"]
                        save = A["open"][d + 1] / r["px"] - 1 if d is not None and d + 1 < len(A["open"]) else np.nan
                        recs.append((r["crossed"], save, r["ret"], r["x"] - r["e"], r["date"] < SPLIT))
            T = pd.DataFrame(ts)
            account(name, fname, pd.DataFrame(ms).T, T)
            R = pd.DataFrame(recs, columns=["ok", "cheaper", "ret", "hold", "h1"])
            S, Fl = R[R.ok], R[~R.ok]
            q_rows.append({
                "预判规则": name, "失败退出": fname, "预判买入次数": len(R),
                "占全部交易": len(R) / len(T),
                "预判成功率": R.ok.mean(),
                "前段成功率": R.ok[R.h1].mean(), "后段成功率": R.ok[~R.h1].mean(),
                "成功时比确认后买便宜(均值)": S.cheaper.mean(), "成功时便宜(中位)": S.cheaper.median(),
                "失败单平均收益": Fl.ret.mean(), "失败单中位持有天数": Fl.hold.median(),
                "成功单胜率": (S.ret > 0).mean(), "成功单每笔均收益": S.ret.mean(),
            })
    Q, O = pd.DataFrame(q_rows), pd.DataFrame(a_rows)
    Q.to_csv(f"{HERE}/MACD预判金叉准确率.csv", index=False)
    O.to_csv(f"{HERE}/MACD预判金叉账户对比.csv", index=False)
    fmt = lambda x: f"{x:,.3f}"
    print("\n=== 一、预判质量（成功=买入后、卖出前真的出现了收盘金叉）===")
    print("便宜 = 原系统在同一次金叉的买入价（确认次日开盘）相对预判买入价高出的比例")
    print(Q.to_string(index=False, float_format=fmt))
    print("\n=== 二、账户级回测（20 只各 1 万全仓复利；胜基准只数满分 20；前段至 2021-12-31）===")
    print(O.to_string(index=False, float_format=fmt))
    print(f"\n输出: MACD预判金叉准确率.csv({len(Q)}行) / MACD预判金叉账户对比.csv({len(O)}行)")


if __name__ == "__main__":
    main()
