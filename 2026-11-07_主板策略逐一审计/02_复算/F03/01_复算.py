# -*- coding: utf-8 -*-
"""
F03 候选B族主板变体：只读复算（2026-11-07）
只读旧交付CSV与冻结面板；所有输出写入本目录。不改任何旧文件、不访问源数据库。
口径：几何年化（自然日365.25 与 252交易日两种）；MDD按逐日权益；
仅有逐笔盈亏的版本用"已实现盈亏重建权益"（exit日入账，无MTM浮动），注明口径差异。
"""
import os
import pandas as pd
import numpy as np
import duckdb

ROOT = "/Users/admin/WorkSpace/trading"
OUT = os.path.join(ROOT, "2026-11-07_主板策略逐一审计/02_复算/F03")
PANEL = os.path.join(ROOT, "trading_finally/data/main_panel.parquet")

def rd(path):
    return pd.read_csv(path, encoding="utf-8-sig")

def eq_stats(df, datecol, eqcol, lowcol=None):
    d = pd.to_datetime(df[datecol])
    e = df[eqcol].astype(float).values
    days = (d.iloc[-1] - d.iloc[0]).days
    n = len(e)
    total = e[-1] / e[0]
    cagr_cal = total ** (365.25 / days) - 1
    cagr_252 = total ** (252 / (n - 1)) - 1
    peak = np.maximum.accumulate(e)
    mdd = (e / peak - 1).min()
    out = dict(start=str(d.iloc[0].date()), end=str(d.iloc[-1].date()),
               n_days=n, eq_start=e[0], eq_end=e[-1],
               total_ret=total - 1, cagr_calendar=cagr_cal,
               cagr_252tradedays=cagr_252, mdd=mdd)
    if lowcol and lowcol in df.columns:
        el = df[lowcol].astype(float).values
        pk = np.maximum.accumulate(el)
        out["mdd_low"] = (el / pk - 1).min()
    return out

def is_main(code):
    c = str(code).split(".")[0]
    return c.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))

def board(code):
    c = str(code).split(".")[0]
    if is_main(code):
        return "主板"
    if c.startswith(("688", "689")):
        return "科创板"
    if c.startswith(("300", "301")):
        return "创业板"
    return "其他"

def trade_stats(df, codecol, pnlcol, exitcol, name):
    pnl = df[pnlcol].astype(float)
    top5 = pnl.sort_values(ascending=False).head(5)
    pos = pnl[pnl > 0].sum()
    b = df[codecol].map(board)
    nonmain = df[b != "主板"]
    row = dict(name=name, n=len(df), wins=(pnl > 0).sum(),
               win_rate=(pnl > 0).mean(), total_pnl=pnl.sum(),
               top5_pnl=top5.sum(), top5_share_net=top5.sum() / pnl.sum(),
               top5_share_gross=top5.sum() / pos if pos > 0 else np.nan,
               nonmain_n=len(nonmain),
               nonmain_pnl=nonmain[pnlcol].astype(float).sum())
    return row, df.assign(board=b, pnl_f=pnl)

def rebuild_equity(trades, dates, pnlcol="pnl", exitcol="exit_date", init=20000.0):
    """已实现盈亏重建：exit日把pnl计入权益；无逐日MTM。"""
    s = trades.groupby(pd.to_datetime(trades[exitcol]))[pnlcol].sum()
    e = init + s.reindex(dates, fill_value=0).cumsum()
    return e

# ---------- 1. 权益曲线复算 ----------
EQ = [
    ("F2", "2026-10-23_主板专用_策略优化/13_F2_权益.csv", "date", "equity", "equity_low", "14.96%/-21.14%"),
    ("40核心_实际价过滤", "2026-10-24_结构交易_主板40系统/05_证据/核心_实际价_权益.csv", "date", "equity", "equity_low", "8.87%/-21.54%"),
    ("40核心_实际价成交", "2026-10-24_结构交易_主板40系统/05_证据/核心_实际价成交_权益.csv", "date", "equity", "equity_low", "8.93%/-21.57%"),
    ("40进攻_实际价过滤", "2026-10-24_结构交易_主板40系统/05_证据/进攻_实际价_权益.csv", "date", "equity", "equity_low", "23.78%/-28.07%"),
    ("40进攻_实际价成交", "2026-10-24_结构交易_主板40系统/05_证据/进攻_实际价成交_权益.csv", "date", "equity", "equity_low", "23.96%/-29.35%"),
    ("进攻版R1_2017起", "2026-10-24_进攻版_2016_2026_回测与优化/03_结果/R1_进攻版_权益.csv", "date", "equity", "equity_low", "24.53%/-36.32%"),
    ("进攻版V2混合", "2026-10-24_进攻版_2016_2026_回测与优化/03_结果/优化_V2_blend1020_权益.csv", "date", "equity", "equity_low", "21.15%/-26.75%"),
    ("45仓位波动20", "2026-10-21_45仓位_波动20_交易系统/权益曲线.csv", "日期", "权益", "最低价权益", "21.6%/-26.4%"),
]
rows = []
for name, rel, dc, ec, lc, claim in EQ:
    df = rd(os.path.join(ROOT, rel))
    r = eq_stats(df, dc, ec, lc)
    r.update(name=name, file=rel, reported=claim)
    rows.append(r)
eq_out = pd.DataFrame(rows)[["name", "reported", "start", "end", "n_days", "eq_start", "eq_end",
                             "total_ret", "cagr_calendar", "cagr_252tradedays", "mdd", "mdd_low", "file"]]
eq_out.to_csv(os.path.join(OUT, "F03_权益复算.csv"), index=False, encoding="utf-8-sig")
print("=== 权益复算 ===")
print(eq_out.drop(columns=["file"]).to_string(index=False))

# ---------- 2. 逐笔统计 ----------
TR = [
    ("F2", "2026-10-23_主板专用_策略优化/13_F2_逐笔.csv", "code", "pnl", "exit_date"),
    ("40核心_实际价过滤", "2026-10-24_结构交易_主板40系统/05_证据/核心_实际价_逐笔.csv", "code", "pnl", "exit_date"),
    ("40核心_实际价成交", "2026-10-24_结构交易_主板40系统/05_证据/核心_实际价成交_逐笔.csv", "code", "pnl", "exit_date"),
    ("40进攻_实际价过滤", "2026-10-24_结构交易_主板40系统/05_证据/进攻_实际价_逐笔.csv", "code", "pnl", "exit_date"),
    ("40进攻_实际价成交", "2026-10-24_结构交易_主板40系统/05_证据/进攻_实际价成交_逐笔.csv", "code", "pnl", "exit_date"),
    ("进攻版R1", "2026-10-24_进攻版_2016_2026_回测与优化/03_结果/R1_进攻版_逐笔.csv", "code", "pnl", "exit_date"),
    ("进攻版V2", "2026-10-24_进攻版_2016_2026_回测与优化/03_结果/优化_V2_blend1020_逐笔.csv", "code", "pnl", "exit_date"),
    ("45仓位", "2026-10-21_45仓位_波动20_交易系统/逐笔交易记录.csv", "代码", "净盈亏", "卖出日"),
    ("7015_amt(45%)", "2026-10-22_全信号_买卖时间_优先级研究/7015_amt_逐笔.csv", "code", "pnl", "exit_date"),
    ("7015_2仓45%", "2026-10-22_当前阶段_情绪同态_50%策略/03_结果/7015_2仓45%_逐笔.csv", "code", "pnl", "exit_date"),
    ("7015_2仓50%", "2026-10-22_当前阶段_情绪同态_50%策略/03_结果/7015_2仓50%_逐笔.csv", "code", "pnl", "exit_date"),
]
trows, top5rows, nonmain_rows, reason_rows = [], [], [], []
tradeCache = {}
for name, rel, cc, pc, xc in TR:
    df = rd(os.path.join(ROOT, rel))
    r, df2 = trade_stats(df, cc, pc, xc, name)
    trows.append(r)
    tradeCache[name] = (df2, cc, pc, xc)
    t5 = df2.sort_values("pnl_f", ascending=False).head(5)
    for _, t in t5.iterrows():
        top5rows.append(dict(name=name, code=t[cc], entry=t.get("entry_date", t.get("买入日")),
                             exit=t[xc], pnl=t["pnl_f"], board=t["board"],
                             reason=t.get("reason", t.get("退出原因", ""))))
    nm = df2[df2["board"] != "主板"]
    for _, t in nm.iterrows():
        nonmain_rows.append(dict(name=name, code=t[cc], entry=t.get("entry_date", t.get("买入日")),
                                 exit=t[xc], pnl=t["pnl_f"], board=t["board"]))
    rc = "reason" if "reason" in df2.columns else ("退出原因" if "退出原因" in df2.columns else None)
    if rc:
        g = df2.groupby(rc)["pnl_f"].agg(["count", "sum"])
        for reason, gg in g.iterrows():
            reason_rows.append(dict(name=name, reason=reason, n=int(gg["count"]), pnl=gg["sum"]))

pd.DataFrame(trows).to_csv(os.path.join(OUT, "F03_逐笔统计.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(top5rows).to_csv(os.path.join(OUT, "F03_前5笔明细.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(nonmain_rows).to_csv(os.path.join(OUT, "F03_非主板交易.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(reason_rows).to_csv(os.path.join(OUT, "F03_退出原因统计.csv"), index=False, encoding="utf-8-sig")
print("\n=== 逐笔统计 ===")
print(pd.DataFrame(trows).to_string(index=False))
print("\n=== 非主板交易（全部） ===")
print(pd.DataFrame(nonmain_rows).to_string(index=False))

# F2 独立仓位
f2 = rd(os.path.join(ROOT, "2026-10-23_主板专用_策略优化/13_F2_逐笔.csv"))
print("\nF2 退出记录数 =", len(f2), " 独立仓位(code+entry) =", f2.groupby(["code", "entry_date"]).ngroups)

# 45仓位 账户贡献口径
t45 = rd(os.path.join(ROOT, "2026-10-21_45仓位_波动20_交易系统/逐笔交易记录.csv"))
c = t45["账户贡献"].astype(float)
top5c = c.sort_values(ascending=False).head(5)
print("45仓位 账户贡献: top5合计=%.4f, 净贡献合计=%.4f, top5/净=%.3f, top5/正贡献=%.3f" % (
    top5c.sum(), c.sum(), top5c.sum() / c.sum(), top5c.sum() / c[c > 0].sum()))

# ---------- 3. 70/15 与 45仓位：重建权益 + 主板-only 近似 ----------
dates45 = pd.to_datetime(rd(os.path.join(ROOT, "2026-10-21_45仓位_波动20_交易系统/权益曲线.csv"))["日期"])
reb = []
for name in ["45仓位", "7015_2仓45%", "7015_2仓50%", "7015_amt(45%)"]:
    df2, cc, pc, xc = tradeCache[name]
    for label, sub in [("全部", df2), ("仅主板(剔除法)", df2[df2["board"] == "主板"])]:
        e = rebuild_equity(sub, dates45, "pnl_f", xc)
        edf = pd.DataFrame({"date": dates45, "equity": e.values})
        r = eq_stats(edf, "date", "equity")
        r.update(name=name, scope=label, method="已实现盈亏重建,无MTM")
        # 年度已实现盈亏
        y = sub.groupby(pd.to_datetime(sub[xc]).dt.year)["pnl_f"].sum()
        r["pnl_2025"] = y.get(2025, 0.0)
        ev = e.values
        dv = dates45.values
        eq24 = ev[dv <= np.datetime64("2024-12-31")][-1]
        eq25 = ev[dv <= np.datetime64("2025-12-31")][-1]
        r["ret_2025_realized"] = eq25 / eq24 - 1
        reb.append(r)
        edf.to_csv(os.path.join(OUT, f"F03_重建权益_{name}_{label}.csv"), index=False, encoding="utf-8-sig")
reb_out = pd.DataFrame(reb)[["name", "scope", "eq_end", "total_ret", "cagr_calendar",
                             "cagr_252tradedays", "mdd", "pnl_2025", "ret_2025_realized", "method"]]
reb_out.to_csv(os.path.join(OUT, "F03_重建权益汇总.csv"), index=False, encoding="utf-8-sig")
print("\n=== 重建权益（已实现口径，无MTM，剔除法主板-only） ===")
print(reb_out.to_string(index=False))

# 688098 专项检查
rows688 = []
for name in ["7015_2仓45%", "7015_2仓50%", "7015_amt(45%)", "45仓位"]:
    df2, cc, pc, xc = tradeCache[name]
    sub = df2[df2[cc].astype(str).str.startswith("688098")]
    y2025 = df2[pd.to_datetime(df2[xc]).dt.year == 2025]["pnl_f"].sum()
    for _, t in sub.iterrows():
        rows688.append(dict(name=name, code=t[cc], entry=t.get("entry_date", t.get("买入日")),
                            exit=t[xc], pnl=t["pnl_f"], pnl_2025_all=y2025,
                            share_2025=t["pnl_f"] / y2025 if y2025 else np.nan,
                            share_total=t["pnl_f"] / df2["pnl_f"].sum()))
pd.DataFrame(rows688).to_csv(os.path.join(OUT, "F03_688098专项.csv"), index=False, encoding="utf-8-sig")
print("\n=== 688098 专项 ===")
print(pd.DataFrame(rows688).to_string(index=False))

# ---------- 4. F2 成交价 vs 冻结面板实际价 ----------
con = duckdb.connect()
pairs = f2[["code", "entry_date"]].drop_duplicates()
vals = ",".join(f"('{r.code}','{r.entry_date}')" for r in pairs.itertuples())
pan = con.sql(f"SELECT code,date,open,close,signal_open FROM '{PANEL}' WHERE (code,date) IN ({vals})").df()
m = f2.merge(pan, left_on=["code", "entry_date"], right_on=["code", "date"], how="left")
m["entry_over_actual_open"] = m["entry"] / m["open"]
m["entry_over_hfq_open"] = m["entry"] / m["signal_open"]
chk = m[["code", "entry_date", "entry", "open", "signal_open",
         "entry_over_actual_open", "entry_over_hfq_open"]]
chk.to_csv(os.path.join(OUT, "F03_F2价格口径核查.csv"), index=False, encoding="utf-8-sig")
print("\n=== F2 入场价/面板实际开盘价 比值分布 ===")
print(chk["entry_over_actual_open"].describe().to_string())
print("比值=1(±1%)的笔数:", int((abs(chk["entry_over_actual_open"] - 1) < 0.01).sum()), "/", len(chk))
print(chk.head(8).to_string(index=False))

# ---------- 5. 持有期尖峰证据（读旧敏感性CSV，不重跑） ----------
hh = []
p1 = os.path.join(ROOT, "2026-10-24_进攻版_2016_2026_回测与优化/03_结果/稳健_持有邻域.csv")
if os.path.exists(p1):
    d = rd(p1); d["source"] = "进攻版2016-2026/稳健_持有邻域.csv"; hh.append(d)
p2 = os.path.join(ROOT, "2026-10-24_结构交易_主板40系统/05_证据/持有期邻域_实际价.csv")
if os.path.exists(p2):
    d = rd(p2); d["source"] = "主板40/持有期邻域_实际价.csv"; hh.append(d)
if hh:
    hd = pd.concat(hh, ignore_index=True)
    hd.to_csv(os.path.join(OUT, "F03_持有期邻域_旧CSV汇编.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 持有期邻域（旧CSV） ===")
    print(hd.to_string(index=False))

print("\n全部输出已写入", OUT)