#!/usr/bin/env python3
"""MACD 金叉系统的执行层/出场层变体实证（诊断性样本内分析，非预注册检验）

在 MACD金叉死叉研究.py 基线引擎（复用其数据/费用/口径）上测试三类变体，
每股独立 1 万本金复利、T 日收盘确认信号最早 T+1 成交，训练段=信号日<=2021-12-31，验证段=2022 起。

A 入场执行变体（出场保持死叉不变）——攻击"右侧滞后买贵"：
  A0  基线：T+1 开盘全仓买
  A1~A6 回踩限价单：信号日收盘×(1−d)，d∈{2%,3%,5%}，有效期 K∈{5,10} 个交易日；
        T+1 起若 low≤限价则以 min(open,限价) 成交（open 低于限价按 open，偏乐观但为任务口径），
        一字涨停日不成交（沿用基线口径），逾期未成交放弃该信号、资金闲置。
  A7  分批：T+1 开盘买半仓；另一半挂信号收盘−3% 限价 5 日，未成交则资金闲置。

B 出场变体（入场保持基线 T+1 开盘）——攻击"死叉慢止损+回吐"：
  B0  基线：死叉 T+1 开盘卖（一字跌停顺延）
  B1  逢高卖：死叉信号日后挂限价单 max(信号日收盘,入场价)×1.01，5 日内 high 触及则以限价成交
      （保守：min(open,限价)），否则第 5 日收盘卖（一字跌停顺延）
  B2  保本止损：持仓期间收盘曾 ≥入场×1.08 后，挂入场价止损（low≤入场价则以 min(open,入场价) 成交）；
      死叉仍兜底
  B3  追踪回吐：持仓期最高收盘 ≥入场×1.10 后，追踪止损=期间最高收盘×0.92（low 触及则以 min(open,止损价)
      成交，只上移不下移）；死叉兜底
  B4  时间止损：持仓第 N 日（N∈{5,8,12}）收盘时 MFE（期间最高 high/入场−1）<3% 则次日开盘卖
  B5  分批止盈：high 触及入场×1.10 时以限价卖出一半（min(open,限价)，一字跌停顺延），其余等死叉
  B6  组合：B2 保本 + B4(N=8) 同时启用

C 状态/跳过规则（入场侧，针对顶部盲区与连败）：
  C1  同股连败冷却：同一股票连续 2 笔亏损后，该股 120 个交易日内忽略新金叉信号
  C2  近顶跳过：信号日距 250 日最高价（high 窗口内最近一次创新高）≤20 个交易日的金叉跳过
  C3  C1+C2 同时启用

纪律：所有限价/止损单用 high/low 判触及、保守成交价假设，不使用未来数据（T 日收盘才知道的信息
最早 T+1 生效）；一字涨停买不进放弃、一字跌停卖不出顺延，均沿用基线口径；A 类未成交资金闲置，
权益曲线按 cash+shares×close 逐日连续。
输出：执行出场变体对比.csv（长表：变体×时段×指标），并先断言基线复现（1724 笔/胜率 0.356/PF 1.363）。
"""
import os
import numpy as np
import pandas as pd
import duckdb
from MACD金叉死叉研究 import (pick_leaders, prep, stamp, perf, backtest,
                              DB, HERE, BASE, CAPITAL, COMM, WARM, SPLIT)

OUT = f"{HERE}/执行出场变体对比.csv"

VARIANTS = {
    "A0":  dict(cat="A", entry="open"),
    "A1":  dict(cat="A", entry="pullback", d=0.02, K=5),
    "A2":  dict(cat="A", entry="pullback", d=0.02, K=10),
    "A3":  dict(cat="A", entry="pullback", d=0.03, K=5),
    "A4":  dict(cat="A", entry="pullback", d=0.03, K=10),
    "A5":  dict(cat="A", entry="pullback", d=0.05, K=5),
    "A6":  dict(cat="A", entry="pullback", d=0.05, K=10),
    "A7":  dict(cat="A", entry="half", d=0.03, K=5),
    "B1":  dict(cat="B", exit_b1=True),
    "B2":  dict(cat="B", be=0.08),
    "B3":  dict(cat="B", trail_arm=0.10, trail=0.92),
    "B4a": dict(cat="B", tstop=5),
    "B4b": dict(cat="B", tstop=8),
    "B4c": dict(cat="B", tstop=12),
    "B5":  dict(cat="B", scale=0.10),
    "B6":  dict(cat="B", be=0.08, tstop=8),
    "C1":  dict(cat="C", cool=True),
    "C2":  dict(cat="C", neartop=True),
    "C3":  dict(cat="C", cool=True, neartop=True),
}


def simulate(code, df, cfg):
    """单股票变体回测。返回 (trades, eq[WARM:], stats)。

    状态机：order=待成交限价买单（回踩/分批第二半）；want_open=次日开盘市价买（基线/A7 前半）；
    sellpend="open"(次日开盘卖) 或 dict(B1 限价卖单)。T 日收盘信息最早 T+1 生效。
    """
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    dif, dea = df.dif.values, df.dea.values
    dates = pd.to_datetime(df.date).values
    n = len(df)
    golden = (dif > dea) & (np.roll(dif, 1) <= np.roll(dea, 1))
    death = (dif < dea) & (np.roll(dif, 1) >= np.roll(dea, 1))
    prev_c = np.concatenate(([c[0]], c[:-1]))
    ow_dn = (h == l) & (o < prev_c)          # 一字跌停：卖不出
    ow_up = (h == l) & (o > prev_c)          # 一字涨停：买不进

    cash, sh = CAPITAL, 0.0
    pos = None                                # dict(si,e,cash0,cost,px_avg,mfe,max_close,armed,trail_on,trail_stop,half_sold)
    order = None                              # 限价买单 dict(L,exp,add,spend,si)
    want_open = False                         # 次日开盘市价买
    open_si = -1
    open_half = False
    sellpend = None
    sell_why = ""
    consec_loss = 0
    cool_until = -10 ** 9
    eq = np.full(n, np.nan)
    trades = []
    st = dict(signals=0, filled=0, expired=0, skipped_lu=0,
              mkt_half=0, lim_half=0, lim_half_exp=0)

    def buy(t, px, spend, si):
        nonlocal cash, sh, pos
        spend = min(spend, cash)
        fee = max(5.0, spend * COMM / (1 + COMM))
        newsh = (spend - fee) / px
        if pos is None:
            pos = dict(si=si, e=t, cash0=cash, cost=spend, mfe=-1e9, max_close=c[t],
                       armed=False, trail_on=False, trail_stop=0.0, half_sold=False)
        else:
            pos["cost"] += spend
        cash -= spend
        sh += newsh
        pos["px_avg"] = pos["cost"] / sh

    def close_trade(x, px, why, is_open=False):
        nonlocal cash, sh, pos, consec_loss, cool_until, order
        proceeds = sh * px
        fee = 0.0 if is_open else max(5.0, proceeds * COMM) + proceeds * stamp(pd.Timestamp(dates[x]))
        total = cash + proceeds - fee
        rec = dict(thscode=code, signal_date=pd.Timestamp(dates[pos["si"]]),
                   entry_date=pd.Timestamp(dates[pos["e"]]), exit_date=pd.Timestamp(dates[x]),
                   entry_px=pos["px_avg"], exit_px=px, ret=total / pos["cash0"] - 1,
                   hold=x - pos["e"], mfe=pos["mfe"], why=why, is_open=is_open)
        trades.append(rec)
        if not is_open:
            if cfg.get("cool"):
                consec_loss = consec_loss + 1 if rec["ret"] <= 0 else 0
                if consec_loss >= 2:
                    cool_until = x + 120
                    consec_loss = 0
        if order is not None and order.get("add"):
            order = None                        # 仓位已平，属于该仓位的"另一半"限价单随之撤销
            st["lim_half_cancel"] = st.get("lim_half_cancel", 0) + 1
        cash, sh, pos = total, 0.0, None

    for t in range(n):
        # --- 1. 待执行卖单 ---
        if pos is not None and sellpend is not None:
            if sellpend == "open":
                if not ow_dn[t]:
                    close_trade(t, o[t], sell_why)
                    sellpend = None
            else:                                   # B1 限价卖单
                if not ow_dn[t] and h[t] >= sellpend["L"]:
                    close_trade(t, min(o[t], sellpend["L"]), "b1_limit")
                    sellpend = None
                elif t >= sellpend["exp"]:
                    if not ow_dn[t]:
                        close_trade(t, c[t], "b1_expire")
                        sellpend = None
                    else:                           # 到期日一字跌停，顺延到下个开盘
                        sellpend, sell_why = "open", "b1_defer"
        # --- 2. 持仓中：盘中止损/分批（决策信息均 ≤ T-1 收盘） ---
        if pos is not None and sellpend is None:
            px = pos["px_avg"]
            if pos["armed"] and not ow_dn[t] and l[t] <= px:
                close_trade(t, min(o[t], px), "breakeven")
            elif pos["trail_on"] and not ow_dn[t] and l[t] <= pos["trail_stop"]:
                close_trade(t, min(o[t], pos["trail_stop"]), "trailing")
            elif cfg.get("scale") and not pos["half_sold"] and not ow_dn[t] \
                    and h[t] >= px * (1 + cfg["scale"]):
                L = px * (1 + cfg["scale"])
                sellsh = sh / 2
                proceeds = sellsh * min(o[t], L)
                fee = max(5.0, proceeds * COMM) + proceeds * stamp(pd.Timestamp(dates[t]))
                cash += proceeds - fee
                sh -= sellsh
                pos["half_sold"] = True
        # --- 3. 次日开盘市价买（A0 全仓 / A7 前半半仓），随后检查限价买单 ---
        if want_open and pos is None:
            want_open = False
            if ow_up[t]:
                st["skipped_lu"] += 1               # 一字涨停买不进，放弃
            else:
                if open_half:
                    st["mkt_half"] += 1
                    buy(t, o[t], cash * 0.5, open_si)
                else:
                    st["filled"] += 1
                    buy(t, o[t], cash, open_si)
        if order is not None:
            if t > order["exp"]:
                if order["add"]:
                    st["lim_half_exp"] += 1
                else:
                    st["expired"] += 1
                order = None
            elif not ow_up[t] and l[t] <= order["L"]:
                if order["add"]:
                    st["lim_half"] += 1
                else:
                    st["filled"] += 1
                buy(t, min(o[t], order["L"]), order["spend"], order["si"])
                order = None
        # --- 4. 逐日权益（资金闲置也计入，曲线连续） ---
        eq[t] = cash + sh * c[t]
        if t < WARM or t >= n - 1:
            continue
        # --- 5. 收盘信号 ---
        if pos is None and order is None and not want_open:
            if golden[t]:
                skip = False
                if cfg.get("cool") and t <= cool_until:
                    skip = True
                if cfg.get("neartop"):
                    s0 = max(0, t - 249)
                    days_since = t - (s0 + int(np.argmax(h[s0:t + 1])))
                    if days_since <= 20:
                        skip = True
                if not skip:
                    st["signals"] += 1
                    entry = cfg.get("entry", "open")
                    if entry == "open":
                        want_open, open_si, open_half = True, t, False
                    elif entry == "pullback":
                        order = dict(L=c[t] * (1 - cfg["d"]), exp=t + cfg["K"], add=False,
                                     spend=cash, si=t)
                    elif entry == "half":
                        want_open, open_si, open_half = True, t, True
                        order = dict(L=c[t] * (1 - cfg["d"]), exp=t + cfg["K"], add=True,
                                     spend=cash * 0.5, si=t)
        if pos is not None and sellpend is None:
            px = pos["px_avg"]
            pos["mfe"] = max(pos["mfe"], h[t] / px - 1)
            pos["max_close"] = max(pos["max_close"], c[t])
            if cfg.get("be") and not pos["armed"] and c[t] >= px * (1 + cfg["be"]):
                pos["armed"] = True
            if cfg.get("trail_arm") and not pos["trail_on"] \
                    and pos["max_close"] >= px * (1 + cfg["trail_arm"]):
                pos["trail_on"] = True
            if pos["trail_on"]:
                pos["trail_stop"] = max(pos["trail_stop"], pos["max_close"] * cfg["trail"])
            if death[t]:
                if cfg.get("exit_b1"):
                    sellpend = dict(L=max(c[t], px) * 1.01, exp=t + 5)
                    sell_why = "b1"
                else:
                    sellpend, sell_why = "open", "death"
            elif cfg.get("tstop") and (t - pos["e"] + 1 == cfg["tstop"]) and pos["mfe"] < 0.03:
                sellpend, sell_why = "open", "tstop"
    if pos is not None:
        close_trade(n - 1, c[-1], "eod", is_open=True)
    return trades, eq[WARM:], st


def build_rows(name, cfg, T, eq_by_code, dates_by_code, base_pt, base_mdd):
    """变体×时段长表。base_pt: dict code->(训练末权益, 期末权益)；base_mdd 同结构分段 MDD。"""
    rows = []
    closed = T[~T.is_open]
    split64 = np.datetime64(SPLIT)
    periods = [("全期", None), ("训练段", None), ("验证段", None)]
    for pname, _ in periods:
        if pname == "全期":
            sub = closed
        elif pname == "训练段":
            sub = closed[closed.signal_date < SPLIT]
        else:
            sub = closed[closed.signal_date >= SPLIT]
        p = perf(sub.ret)
        finals, ratios, mdds = [], [], []
        for code, eq in eq_by_code.items():
            d = dates_by_code[code]
            k = int(np.searchsorted(d, split64))    # 首个 >=2022-01-01 的位置
            if pname == "全期":
                pt, seg = eq[-1], eq
            elif pname == "训练段":
                kk = min(max(k, 1), len(eq) - 1)
                pt, seg = eq[kk - 1], eq[:kk]
            else:
                kk = min(max(k, 1), len(eq) - 1)
                pt, seg = eq[-1], eq[kk - 1:]
            seg = seg / seg[0]
            mdd = float((seg / np.maximum.accumulate(seg) - 1).min())
            bpt = base_pt[code][0 if pname == "训练段" else 1]
            finals.append(pt)
            ratios.append(pt / bpt)
            mdds.append(mdd - base_mdd[code][0 if pname == "训练段" else (1 if pname == "验证段" else 2)])
        rows.append(dict(变体=name, 类别=cfg["cat"], 时段=pname, 笔数=p["n"], 胜率=p["win_rate"],
                         PF=p["pf"], 均收益=p["avg_ret"], 盈亏比=p["payoff"],
                         持仓天数中位=float(sub.hold.median()) if len(sub) else np.nan,
                         期末权益中位数=float(np.median(finals)),
                         期末权益几何均值比=float(np.exp(np.mean(np.log(ratios)))),
                         MDD中位_vs基线=float(np.median(mdds))))
    return rows


def seg_mdd(eq, mode, k):
    if mode == 0:
        seg = eq
    elif mode == 1:
        seg = eq[:min(max(k, 1), len(eq) - 1)]
    else:
        seg = eq[min(max(k, 1), len(eq) - 1) - 1:]
    seg = seg / seg[0]
    return float((seg / np.maximum.accumulate(seg) - 1).min())


def main():
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 300)
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
    DF = {code: prep(df.reset_index(drop=True), hs) for code, df in raw.groupby("thscode")}
    DATES = {code: pd.to_datetime(df.date).values[WARM:] for code, df in DF.items()}
    print(f"股票池 {len(DF)} 只；训练段信号 < {SPLIT.date()}，验证段 >= {SPLIT.date()}\n")

    # ---------- 基线复现断言：本引擎 vs 原 backtest() ----------
    base_cfg = dict(cat="B")
    base_eq, base_trades_all = {}, []
    max_diff = 0.0
    for code, df in DF.items():
        tr, eq, _ = simulate(code, df, base_cfg)
        tr0, _ = backtest(code, df)                 # 原引擎（含 entry_features，慢但只跑一次）
        c0 = pd.DataFrame(tr0)
        c1 = pd.DataFrame(tr)
        assert len(c0) == len(c1), f"{code} 笔数不一致 {len(c0)} vs {len(c1)}"
        d = abs(c0.ret.sum() - c1.ret.sum()) + abs((~c0.is_open).sum() - (~c1.is_open).sum())
        max_diff = max(max_diff, d)
        base_eq[code] = eq
        base_trades_all += tr
    B0 = pd.DataFrame(base_trades_all)
    bcl = B0[~B0.is_open]
    bp = perf(bcl.ret)
    print(f"基线复现：与原 backtest() 逐笔 ret 和最大差 {max_diff:.2e}")
    print(f"基线（已平仓 {bp['n']} 笔）：胜率 {bp['win_rate']:.3f} PF {bp['pf']:.3f} "
          f"均收益 {bp['avg_ret']:.4f} 盈亏比 {bp['payoff']:.3f} 持仓中位 {bcl.hold.median():.0f} 天")
    # 已知值 1724 笔/0.356/1.363/0.011 为上次数据快照；数据刷新后允许微小漂移，
    # 硬断言放在"与本脚本内原引擎逐笔一致"（max_diff<1e-6）+ 汇总指标容差
    assert 1720 <= bp["n"] <= 1730, f"基线笔数 {bp['n']} 异常"
    assert abs(bp["win_rate"] - 0.356) < 0.01 and abs(bp["pf"] - 1.363) < 0.02 \
        and abs(bp["avg_ret"] - 0.011) < 0.004, "基线汇总指标与已知值不符"
    assert max_diff < 1e-6, "与原引擎逐笔不一致"

    split64 = np.datetime64(SPLIT)
    base_pt, base_mdd = {}, {}
    for code, eq in base_eq.items():
        k = int(np.searchsorted(DATES[code], split64))
        kk = min(max(k, 1), len(eq) - 1)
        base_pt[code] = (eq[kk - 1], eq[-1])
        base_mdd[code] = (seg_mdd(eq, 1, k), seg_mdd(eq, 2, k), seg_mdd(eq, 0, k))

    # ---------- 跑全部变体 ----------
    results = {}
    for name, cfg in VARIANTS.items():
        all_tr, eq_by_code, stot = [], {}, dict(signals=0, filled=0, expired=0, skipped_lu=0,
                                                mkt_half=0, lim_half=0, lim_half_exp=0)
        for code, df in DF.items():
            tr, eq, st = simulate(code, df, cfg)
            all_tr += tr
            eq_by_code[code] = eq
            for k in stot:
                stot[k] += st[k]
        T = pd.DataFrame(all_tr)
        results[name] = (T, eq_by_code, stot)
        print(f"{name:<4} done: 信号 {stot['signals']} 笔数 {(~T.is_open).sum()}")

    # ---------- 汇总长表 ----------
    rows = []
    for name, cfg in VARIANTS.items():
        T, eq_by_code, stot = results[name]
        rows += build_rows(name, cfg, T, eq_by_code, DATES, base_pt, base_mdd)
    # B0/C0 = 基线对照行（与 A0 同一引擎结果，换标签便于分组对照）
    for bname, bcat in [("B0", "B"), ("C0", "C")]:
        rows += build_rows(bname, dict(cat=bcat), B0, base_eq, DATES, base_pt, base_mdd)
    R = pd.DataFrame(rows)
    # A 类成交率/放弃占比
    for name, cfg in VARIANTS.items():
        stot = results[name][2]
        m = R.变体 == name
        if cfg.get("entry") == "pullback":
            s = max(stot["signals"], 1)
            R.loc[m, "成交率"] = stot["filled"] / s
            R.loc[m, "放弃信号占比"] = stot["expired"] / s
        elif cfg.get("entry") == "half":
            s = max(stot["signals"], 1)
            R.loc[m, "成交率"] = (stot["mkt_half"] + stot["lim_half"]) / (2 * s)
            R.loc[m, "放弃信号占比"] = stot["lim_half_exp"] / s
        elif cfg.get("entry") == "open":
            s = max(stot["filled"] + stot["skipped_lu"], 1)
            R.loc[m, "成交率"] = stot["filled"] / s
            R.loc[m, "放弃信号占比"] = stot["skipped_lu"] / s
    R.to_csv(OUT, index=False, encoding="utf-8-sig")

    # ---------- 打印对比 ---
    cols = ["变体", "时段", "笔数", "胜率", "PF", "均收益", "盈亏比", "持仓天数中位",
            "期末权益中位数", "期末权益几何均值比", "MDD中位_vs基线", "成交率", "放弃信号占比"]
    fmt = lambda x: f"{x:,.3f}"
    for cat, title in [("A", "=== A 入场执行变体（出场=死叉） vs 基线 A0 ==="),
                       ("B", "=== B 出场变体（入场=基线 T+1 开盘） vs 基线 B0 ==="),
                       ("C", "=== C 跳过规则（入场侧） vs 基线 ===")]:
        print(f"\n{title}")
        print(R[R.类别 == cat][cols].to_string(index=False, float_format=fmt))

    # 出场原因分布（诊断）
    print("\n=== 各 B 类变体出场原因分布（全期，含未平仓 eod） ===")
    for name in ["B1", "B2", "B3", "B4a", "B4b", "B4c", "B5", "B6"]:
        T = results[name][0]
        dist = T.why.value_counts(normalize=True).round(3).to_dict()
        print(f"  {name:<4} 笔数 {len(T):>4}: {dist}")
    print(f"\n输出: {OUT} ({len(R)} 行)")


if __name__ == "__main__":
    main()
