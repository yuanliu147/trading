#!/usr/bin/env python3
"""MACD(12,26,9) 日线金叉买 / 死叉卖：随机行业龙头回测 + 失效/盈利特征符合率

规则：T 日收盘确认金叉 -> T+1 开盘全仓买入；T 日收盘确认死叉 -> T+1 开盘全部卖出。
T+1 一字涨停买不进则放弃该信号；一字跌停卖不出顺延到下一个开盘。
每只股票独立账户，本金 1 万，全仓复利，允许碎股（前复权价无法正确模拟 100 股整手）。
费用：佣金万 2.5（最低 5 元）双边；印花税卖出千 1（2023-08-28 起万 5）。
龙头定义：随机抽取的同花顺行业内，近 250 日日均成交额最大的非 ST 股（需 >=1900 根日线）。
数据：本地 hithink-finance DuckDB 前复权日线 v_daily_qfq；大盘环境用 analysis/hs300_daily.csv。
"""
import os, json, random
import duckdb, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))    # analysis/macd
BASE = os.path.dirname(HERE)                         # analysis，共享数据所在目录
DB = os.path.expanduser("~/Library/Application Support/hithink-finance/data/market.duckdb")
SEED, N_IND, MIN_BARS = 2026, 20, 1900
CAPITAL, COMM, WARM = 10_000.0, 0.00025, 150
SPLIT = pd.Timestamp("2022-01-01")


def pick_leaders(con):
    smap = json.load(open(f"{BASE}/sector_map.json"))
    chosen = random.Random(SEED).sample(sorted(set(smap.values())), N_IND)
    codes = [c for c, ind in smap.items() if ind in chosen]
    st = con.execute("""
        WITH d AS (SELECT thscode, date, amount,
                          ROW_NUMBER() OVER (PARTITION BY thscode ORDER BY date DESC) rn
                   FROM v_daily_qfq WHERE thscode IN (SELECT UNNEST(?::VARCHAR[])))
        SELECT thscode, COUNT(*) n, AVG(CASE WHEN rn <= 250 THEN amount END) amt250, MAX(date) last_date
        FROM d GROUP BY thscode""", [codes]).fetchdf()
    st["name"] = st.thscode.map(json.load(open(f"{BASE}/name_map.json")))
    st["industry"] = st.thscode.map(smap)
    last = st["last_date"].max()
    ok = (st.n >= MIN_BARS) & (st["last_date"] == last) & ~st.name.fillna("").str.contains("ST|退")
    lead = st[ok].sort_values("amt250", ascending=False).groupby("industry").head(1)
    return lead.sort_values("industry").reset_index(drop=True), sorted(chosen)


def weekly_macd_partial(dates, close):
    """周线 MACD，当周未走完时用当日收盘作为本周收盘（盘中可见口径，无未来函数）。"""
    wk = dates.dt.to_period("W-FRI")
    wc = close.groupby(wk.values).last()
    e12, e26 = wc.ewm(span=12, adjust=False).mean(), wc.ewm(span=26, adjust=False).mean()
    dea = (e12 - e26).ewm(span=9, adjust=False).mean()
    prev = pd.DataFrame({"e12": e12, "e26": e26, "dea": dea}).shift(1).loc[wk.values].values
    E12 = 2 / 13 * close.values + 11 / 13 * prev[:, 0]
    E26 = 2 / 27 * close.values + 25 / 27 * prev[:, 1]
    difw = E12 - E26
    return difw, 0.2 * difw + 0.8 * prev[:, 2]


def prep(df, hs):
    c = df.close
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    df["dif"], df["dea"], df["hist"] = dif, dea, 2 * (dif - dea)
    df["ma60"], df["ma250"] = c.rolling(60).mean(), c.rolling(250).mean()
    df["difw"], df["deaw"] = weekly_macd_partial(df.date, c)
    df["std20"] = c.pct_change().rolling(20).std() * 100
    df["amt_ratio"] = df.amount / df.amount.shift(1).rolling(20).mean()
    df = pd.merge_asof(df, hs, on="date", direction="backward")
    return df


def entry_features(df, i):
    o, l, c = df.open.values, df.low.values, df.close.values
    dif, dea, hist = df.dif.values, df.dea.values, df["hist"].values
    s = i - 1
    while s > 0 and hist[s - 1] <= 0:
        s -= 1
    green = slice(s, i)                       # 金叉前的绿柱段
    k = s - 1
    while k > 0 and hist[k] > 0:
        k -= 1
    e2 = k
    while k > 0 and hist[k - 1] <= 0:
        k -= 1
    prev_green = slice(k, e2 + 1)             # 再前一段绿柱
    area = -hist[green].sum() / c[i] * 100
    prev_area = -hist[prev_green].sum() / c[i] * 100
    lower_low = l[green].min() < l[prev_green].min()
    sgn = np.sign(hist[i - 61:i])
    pg = e2 + 1                               # 上一次金叉日
    return dict(
        dif_pct=dif[i] / c[i] * 100, dea_pct=dea[i] / c[i] * 100,
        green_area=area, green_len=i - s, prev_green_area=prev_area,
        dif_slope3=(dif[i] - dif[i - 3]) / c[i] * 100,
        dea_falling=dea[i] < dea[i - 1],
        crosses60=int((sgn[1:] != sgn[:-1]).sum()),
        air_refuel=bool(dea[i] > 0 and dif[green].min() > 0),
        second_cross_below=bool(dif[i] < 0 and dif[pg] < 0 and dif[i] > dif[pg] and i - pg <= 40),
        bull_div=bool(lower_low and dif[green].min() > dif[prev_green].min() and dif[green].min() < 0),
        area_div=bool(lower_low and area < prev_area),
        below_ma60=c[i] < df.ma60.values[i],
        ma60_down=df.ma60.values[i] < df.ma60.values[i - 10],
        below_ma250=c[i] < df.ma250.values[i],
        weekly_bear=df.difw.values[i] < df.deaw.values[i],
        weekly_dif_neg=df.difw.values[i] < 0,
        amt_ratio=df.amt_ratio.values[i],
        std20=df.std20.values[i],
        rebound20=c[i] / l[i - 19:i + 1].min() - 1,
        ret60=c[i] / c[i - 60] - 1,
        day_chg=c[i] / c[i - 1] - 1,
        mkt_below_ma60=bool(df.hs_below_ma60.values[i]),
    )


def stamp(d):
    return 0.0005 if d >= pd.Timestamp("2023-08-28") else 0.001


def backtest(code, df):
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    dif, dea, hist = df.dif.values, df.dea.values, df["hist"].values
    dates = df.date
    golden = (dif > dea) & (np.roll(dif, 1) <= np.roll(dea, 1))
    death = (dif < dea) & (np.roll(dif, 1) >= np.roll(dea, 1))
    n = len(df)
    cash, shares, pos = CAPITAL, 0.0, None
    want_buy = want_sell = False
    eq = np.full(n, np.nan)
    trades, skipped = [], 0

    def close_trade(x, px, is_open):
        nonlocal cash, shares, pos
        proceeds = shares * px
        fee = 0.0 if is_open else max(5.0, proceeds * COMM) + proceeds * stamp(dates.iloc[x])
        cash_after = proceeds - fee
        i, e = pos["signal_idx"], pos["entry_idx"]
        d = pos.get("exit_signal_idx", n - 1)
        rec = dict(thscode=code, signal_date=dates.iloc[i], entry_date=dates.iloc[e],
                   exit_date=dates.iloc[x], entry_px=pos["px"], exit_px=px,
                   ret=cash_after / pos["cash_before"] - 1, pnl=cash_after - pos["cash_before"],
                   hold=x - e, mfe=h[e:max(x, e + 1)].max() / pos["px"] - 1,
                   mae=l[e:max(x, e + 1)].min() / pos["px"] - 1,
                   red_area=hist[i:d][hist[i:d] > 0].sum() / c[i] * 100,
                   dif_cross_zero=bool(dif[i] < 0 and dif[i:d + 1].max() > 0),
                   is_open=is_open)
        if x >= e + 3:                         # 第3个持仓日收盘时仍持有
            k = e + 2
            rec.update(early_loss3=c[k] < pos["px"],
                       early_hist_shrink=hist[k] < hist[k - 1],
                       early_dif_down=dif[k] < dif[k - 1],
                       early_break_low=l[e:k + 1].min() < l[i],
                       early_mfe3=h[e:k + 1].max() / pos["px"] - 1,
                       alt_ret=(1 + rec["ret"]) * o[e + 3] / px - 1)
        rec.update(pos["feat"])
        trades.append(rec)
        if not is_open:
            cash, shares, pos = cash_after, 0.0, None

    for t in range(n):
        if want_sell and pos is not None:
            if h[t] == l[t] and o[t] < c[t - 1]:
                pass                                   # 一字跌停卖不出，顺延
            else:
                close_trade(t, o[t], False)
                want_sell = False
        if want_buy and pos is None:
            want_buy = False
            if h[t] == l[t] and o[t] > c[t - 1]:
                skipped += 1                           # 一字涨停买不进
            else:
                fee = max(5.0, cash * COMM / (1 + COMM))
                shares = (cash - fee) / o[t]
                pos = dict(signal_idx=t - 1, entry_idx=t, px=o[t], cash_before=cash,
                           feat=entry_features(df, t - 1))
                cash = 0.0
        eq[t] = cash + shares * c[t]
        if t >= WARM and t < n - 1:
            if pos is None and golden[t]:
                want_buy = True
            elif pos is not None and death[t] and not want_sell:
                want_sell = True
                pos["exit_signal_idx"] = t
    if pos is not None:
        close_trade(n - 1, c[-1], True)
    e = pd.Series(eq[WARM:], index=dates.iloc[WARM:].values)
    yrs = (dates.iloc[-1] - dates.iloc[WARM]).days / 365.25
    final = e.iloc[-1]
    summary = dict(thscode=code, start=dates.iloc[WARM].date(), end=dates.iloc[-1].date(),
                   final_equity=final, total_ret=final / CAPITAL - 1,
                   cagr=(final / CAPITAL) ** (1 / yrs) - 1,
                   mdd=(e / e.cummax() - 1).min(),
                   bh_ret=c[-1] / o[WARM + 1] - 1,
                   bh_mdd=(pd.Series(c[WARM:]) / pd.Series(c[WARM:]).cummax() - 1).min(),
                   skipped_limit_up=skipped)
    return trades, summary


def perf(r):
    w, ls = r[r > 0], r[r <= 0]
    return dict(n=len(r), win_rate=len(w) / len(r) if len(r) else np.nan,
                avg_win=w.mean(), avg_loss=ls.mean(),
                payoff=w.mean() / abs(ls.mean()) if len(ls) else np.nan,
                pf=w.sum() / abs(ls.sum()) if len(ls) else np.nan,
                avg_ret=r.mean(), median_ret=r.median())


def feature_table(T, feats):
    win = T.ret > 0
    h1 = T.signal_date < SPLIT
    base = win.mean()
    rows = []
    for name, kind, m in feats:
        m = m.fillna(False).astype(bool)
        n1, n0 = int(m.sum()), int((~m).sum())
        if n1 == 0 or n0 == 0:
            continue
        wr1, wr0 = win[m].mean(), win[~m].mean()
        se = np.sqrt(base * (1 - base) * (1 / n1 + 1 / n0))
        p = perf(T.ret[m])
        rows.append({
            "特征": name, "类型": kind, "样本": n1, "占比": n1 / len(T),
            "亏损符合率P(f|亏)": m[~win].mean(), "盈利符合率P(f|盈)": m[win].mean(),
            "出现时胜率": wr1, "未出现胜率": wr0, "z": (wr1 - wr0) / se,
            "亏损倍数": (1 - wr1) / (1 - base),
            "出现时盈亏比": p["payoff"], "出现时PF": p["pf"], "出现时均收益": p["avg_ret"],
            "亏损额占比": T.ret[m & ~win].sum() / T.ret[~win].sum(),
            "前段胜率16-21": win[m & h1].mean(), "后段胜率22-26": win[m & ~h1].mean(),
            "前段均收益": T.ret[m & h1].mean(), "后段均收益": T.ret[m & ~h1].mean(),
        })
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    con = duckdb.connect(DB, read_only=True)
    lead, chosen = pick_leaders(con)
    print(f"随机行业(seed={SEED}): {chosen}\n")
    print(lead[["industry", "thscode", "name", "n", "amt250"]].to_string(index=False))
    lead.to_csv(f"{HERE}/MACD研究股票名单.csv", index=False)

    raw = con.execute("""SELECT thscode, date, open, high, low, close, amount FROM v_daily_qfq
                         WHERE thscode IN (SELECT UNNEST(?::VARCHAR[])) ORDER BY thscode, date""",
                      [lead.thscode.tolist()]).fetchdf()
    con.close()
    raw["date"] = pd.to_datetime(raw.date)
    assert (raw[["open", "high", "low", "close"]] > 0).all().all(), "前复权价出现非正值"
    hs = pd.read_csv(f"{BASE}/hs300_daily.csv", parse_dates=["date"])[["date", "close"]]
    hs["hs_below_ma60"] = hs.close < hs.close.rolling(60).mean()
    hs = hs[["date", "hs_below_ma60"]]

    all_trades, sums = [], []
    for code, df in raw.groupby("thscode"):
        df = prep(df.reset_index(drop=True), hs)
        tr, sm = backtest(code, df)
        all_trades += tr
        sums.append(sm)
    T = pd.DataFrame(all_trades)
    S = pd.DataFrame(sums).merge(lead[["thscode", "name", "industry"]], on="thscode")
    closed = T[~T.is_open].copy()
    for code, g in closed.groupby("thscode"):
        p = perf(g.ret)
        for k in ("n", "win_rate", "payoff", "pf", "avg_ret"):
            S.loc[S.thscode == code, k] = p[k]
    T.merge(lead[["thscode", "name", "industry"]], on="thscode") \
        .to_csv(f"{HERE}/MACD金叉死叉逐笔交易.csv", index=False)
    S.to_csv(f"{HERE}/MACD金叉死叉个股汇总.csv", index=False)

    print("\n=== 一、逐只回测（本金1万，全仓复利，T+1开盘成交，含费用）===")
    show = S[["name", "industry", "n", "win_rate", "payoff", "pf", "final_equity", "total_ret",
              "cagr", "mdd", "bh_ret", "bh_mdd"]].sort_values("total_ret", ascending=False)
    print(show.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\n跑赢买入持有: {(S.total_ret > S.bh_ret).sum()}/{len(S)}；"
          f"盈利账户: {(S.total_ret > 0).sum()}/{len(S)}；"
          f"最终权益中位数 {S.final_equity.median():,.0f}，均值 {S.final_equity.mean():,.0f}；"
          f"买入持有中位数 {(CAPITAL * (1 + S.bh_ret)).median():,.0f}")

    P = perf(closed.ret)
    print("\n=== 二、全部已平仓交易汇总 ===")
    print({k: round(v, 4) for k, v in P.items()})
    print(f"平均持仓 {closed.hold.mean():.1f} 天，中位 {closed.hold.median():.0f} 天；"
          f"赢家平均持仓 {closed[closed.ret > 0].hold.mean():.1f} 天，输家 {closed[closed.ret <= 0].hold.mean():.1f} 天")
    W = closed[closed.ret > 0]
    print(f"盈利交易：平均最大浮盈 {W.mfe.mean():.2%}，平均实现 {W.ret.mean():.2%}，"
          f"利润兑现率(实现/最大浮盈)中位数 {(W.ret / W.mfe).median():.1%}")
    top = closed.ret.sort_values(ascending=False)
    print(f"收益集中度：最好的 10% 交易贡献的收益和 = {top.head(len(top) // 10).sum():.2f}，"
          f"全部交易收益和 = {top.sum():.2f}")
    for lo, hi, lab in [(pd.Timestamp("2016-01-01"), SPLIT, "2016-2021"),
                        (SPLIT, pd.Timestamp("2030-01-01"), "2022-2026")]:
        sub = closed[(closed.signal_date >= lo) & (closed.signal_date < hi)]
        print(f"  {lab}: " + str({k: round(float(v), 4) for k, v in perf(sub.ret).items()}))

    # ---------- 特征 ----------
    h1 = closed[closed.signal_date < SPLIT]
    q = lambda col, p: h1[col].quantile(p)
    C = closed
    entry = [
        ("零轴下金叉(DIF<0且DEA<0)", (C.dif_pct < 0) & (C.dea_pct < 0)),
        ("零轴上金叉(DIF>0且DEA>0)", (C.dif_pct > 0) & (C.dea_pct > 0)),
        ("前绿柱面积小(下1/3)", C.green_area <= q("green_area", 1 / 3)),
        ("前绿柱面积大(上1/3)", C.green_area >= q("green_area", 2 / 3)),
        ("前绿柱持续<=5天", C.green_len <= 5),
        ("前绿柱持续>=15天", C.green_len >= 15),
        ("金叉角度平缓(DIF3日斜率下1/3)", C.dif_slope3 <= q("dif_slope3", 1 / 3)),
        ("金叉角度陡峭(DIF3日斜率上1/3)", C.dif_slope3 >= q("dif_slope3", 2 / 3)),
        ("金叉时DEA仍向下", C.dea_falling),
        ("近60日已交叉>=4次(反复缠绕)", C.crosses60 >= 4),
        ("空中加油(零轴上回调DIF未破零再金叉)", C.air_refuel),
        ("零轴下二次金叉(DIF低点抬高,40日内)", C.second_cross_below),
        ("DIF深处零轴下(下1/3)", C.dif_pct <= q("dif_pct", 1 / 3)),
        ("底背离(价新低+DIF低点抬高)", C.bull_div),
        ("绿柱面积背驰(价新低+面积缩小)", C.area_div),
        ("收盘在MA60下方", C.below_ma60),
        ("MA60向下(低于10日前)", C.ma60_down),
        ("收盘在年线MA250下方", C.below_ma250),
        ("周线MACD空头(DIFw<DEAw)", C.weekly_bear),
        ("周线DIF<0", C.weekly_dif_neg),
        ("放量金叉(成交额>=1.5倍20日均)", C.amt_ratio >= 1.5),
        ("缩量金叉(成交额<0.8倍20日均)", C.amt_ratio < 0.8),
        ("高波动(20日波动率上1/3)", C.std20 >= q("std20", 2 / 3)),
        ("距20日低点已反弹>10%", C.rebound20 > 0.10),
        ("前60日跌幅>15%", C.ret60 < -0.15),
        ("前60日涨幅>20%", C.ret60 > 0.20),
        ("金叉当日涨幅>=4%", C.day_chg >= 0.04),
        ("大盘沪深300在MA60下方", C.mkt_below_ma60),
    ]
    during = [
        ("持仓<=5天即死叉", C.hold <= 5),
        ("持仓>=20天", C.hold >= 20),
        ("最大浮盈<2%(几乎没涨)", C.mfe < 0.02),
        ("最大浮盈>=5%", C.mfe >= 0.05),
        ("最大浮盈>=10%", C.mfe >= 0.10),
        ("最大浮亏<=-8%", C.mae <= -0.08),
        ("持仓期红柱面积小(下1/3)", C.red_area <= q("red_area", 1 / 3)),
        ("零轴下金叉且持仓中DIF上穿零轴", C.dif_cross_zero),
        ("利润回吐(浮盈>=5%最终亏损)", (C.mfe >= 0.05) & (C.ret <= 0)),
    ]
    F = pd.concat([feature_table(C, [(n, "金叉时可见", m) for n, m in entry]),
                   feature_table(C, [(n, "持仓过程(事后)", m) for n, m in during])])
    F.to_csv(f"{HERE}/MACD基础特征符合率.csv", index=False)
    fmt = lambda x: f"{x:.3f}"
    cols = ["特征", "样本", "占比", "亏损符合率P(f|亏)", "盈利符合率P(f|盈)", "出现时胜率", "未出现胜率",
            "z", "出现时盈亏比", "出现时PF", "出现时均收益", "前段胜率16-21", "后段胜率22-26",
            "前段均收益", "后段均收益"]
    print(f"\n=== 三、金叉当日可见特征（基准胜率 {P['win_rate']:.3f}，基准PF {P['pf']:.3f}）===")
    E = F[F["类型"] == "金叉时可见"].sort_values("出现时均收益")
    print(E[cols].to_string(index=False, float_format=fmt))
    print("\n=== 四、持仓过程特征（事后描述，不可用于进场过滤）===")
    print(F[F["类型"] != "金叉时可见"][cols[:10]].to_string(index=False, float_format=fmt))

    L = C[C.ret <= 0]
    types = {
        "快速假金叉(持仓<=5天且浮盈<3%)": (L.hold <= 5) & (L.mfe < 0.03),
        "利润回吐(浮盈>=5%后死叉卖出亏损)": L.mfe >= 0.05,
        "一买就跌(浮盈<1%)": L.mfe < 0.01,
    }
    print(f"\n=== 五、亏损交易失效形态（共 {len(L)} 笔，可重叠）===")
    for k, m in types.items():
        print(f"  {k}: {m.mean():.1%} 笔，占亏损额 {L.ret[m].sum() / L.ret.sum():.1%}")

    # ---------- 样本外检验：前段选“坏特征”，后段剔除 ----------
    pool = []
    for n, m in entry:
        m = m.fillna(False).astype(bool)
        a = C[(C.signal_date < SPLIT) & m]
        if len(a) >= 60:
            pool.append((perf(a.ret)["avg_ret"], n, m))
    base1 = perf(C[C.signal_date < SPLIT].ret)["avg_ret"]
    bad = [x for x in sorted(pool, key=lambda x: x[0]) if x[0] < base1][:3]
    good = [x for x in sorted(pool, key=lambda x: -x[0]) if x[0] > base1][:3]
    h2 = C.signal_date >= SPLIT
    print(f"\n=== 六、样本外检验（2016-2021 选特征，2022-2026 检验）===")
    print("前段最差特征:", [f"{n}({r:.2%})" for r, n, _ in bad])
    print("前段最好特征:", [f"{n}({r:.2%})" for r, n, _ in good])
    excl = np.zeros(len(C), bool)
    for _, _, m in bad:
        excl |= m.values
    keep = np.zeros(len(C), bool)
    for _, _, m in good:
        keep |= m.values
    for label, mask in [("后段-原始", h2.values), ("后段-剔除任一坏特征", h2.values & ~excl),
                        ("后段-仅保留含任一好特征", h2.values & keep)]:
        sub = C[mask]
        eqs = sub.groupby("thscode").ret.apply(lambda r: CAPITAL * (1 + r).prod())
        print(f"  {label}: " + str({k: round(v, 4) for k, v in perf(sub.ret).items()}) +
              f" 每只账户期末中位数 {eqs.median():,.0f}")
    print(f"\n输出: MACD研究股票名单.csv / MACD金叉死叉逐笔交易.csv({len(T)}行) / MACD金叉死叉个股汇总.csv({len(S)}行) / "
          f"MACD基础特征符合率.csv({len(F)}行)")

    # ---------- 持仓第3天收盘可观测的早期预警 ----------
    D = C[C.alt_ret.notna()].copy()
    early = [
        ("第3天收盘仍浮亏", D.early_loss3.astype(bool)),
        ("3天内红柱已缩短(第3天<第2天)", D.early_hist_shrink.astype(bool)),
        ("3天内DIF拐头向下", D.early_dif_down.astype(bool)),
        ("3天内跌破金叉日最低价", D.early_break_low.astype(bool)),
        ("3天内最大浮盈<2%", D.early_mfe3 < 0.02),
    ]
    G = feature_table(D, [(n, "持仓第3天可见", m) for n, m in early])
    print(f"\n=== 七、持仓第3天收盘可观测特征（{len(D)} 笔仍持有到第3天收盘，"
          f"子样本胜率 {(D.ret > 0).mean():.3f}）===")
    print(G[cols].to_string(index=False, float_format=fmt))
    print("\n  规则测试：特征出现则第4天开盘离场（其余照常等死叉）")
    for n, m in early:
        for lab, per in [("全期", D.signal_date.notna()), ("16-21", D.signal_date < SPLIT),
                         ("22-26", D.signal_date >= SPLIT)]:
            r0 = D.ret[per]
            r1 = np.where(m[per], D.alt_ret[per], r0)
            p0, p1 = perf(r0), perf(pd.Series(r1))
            print(f"    {n:<22}{lab}: 均收益 {p0['avg_ret']:.4f}->{p1['avg_ret']:.4f}  "
                  f"PF {p0['pf']:.3f}->{p1['pf']:.3f}  胜率 {p0['win_rate']:.3f}->{p1['win_rate']:.3f}  "
                  f"盈亏比 {p0['payoff']:.2f}->{p1['payoff']:.2f}")
    pd.concat([F, G]).to_csv(f"{HERE}/MACD基础特征符合率.csv", index=False)


if __name__ == "__main__":
    main()
