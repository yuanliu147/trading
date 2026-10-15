#!/usr/bin/env python3
"""MACD 金叉系统第三轮：完全分类网格 + 组合终审（诊断性样本内分析，非预注册）

Part 1（20龙头）：两个新出场/执行变体单测
  V1 结构失效退出：持仓期间收盘 < 信号日最低价 → 次日开盘退出（一字跌停顺延），死叉仍兜底。
  V2 中阴半仓：T+1 开盘买半仓；第 5 个持仓日（入场后第4个交易日，与基线 early_* 口径一致）
      收盘仍持有且收盘>入场均价 → 次日开盘补另一半；否则半仓持有到死叉。权益按实际仓位计。
  引擎复用 MACD执行出场变体研究.py 的 simulate() 口径（追加 cfg 参数实现 V1/V2）。

Part 2（20龙头）：金叉上下文完全分类网格
  维度（全部 ≤信号日 可观测、机械化）：
    dd250 三档（训练段三分位）× cross_cnt120 二档（训练段中位）× pct20 三档（训练段三分位）
    pct20 无数据信号（2017-10-19 前）单独一格“无数据” → 3×2×3+1 = 19 格
    近顶格（days_high250≤20，与 C2 一致）作为独立禁入格叠加。
  训练段逐格统计 n/胜率/PF/均收益。
  白名单规则（事先写死）：训练段 n≥20 且 均收益>训练段全体均值 且 胜率≥全体均值。
  验证段对白名单整体只验收一次（逐笔统计+每股期末比）；黑名单格（训练段均收益<0）作对照。
  输出格子稳定性（两段符号翻转格数）。多重比较诚实声明：19 格样本内诊断，逐格不可单独采信，
  只有白名单整体验收有效。

Part 3（20龙头）：组合系统 = 白名单格入场 + B3 追踪回吐 + （V1 若期末比≥0.98 则加入）。

Part 4（池50终审）：复刻 pool50_turnpoint_backtest.py 机制（H5 月度池 N=50，2017-10 起，
  T+1 收盘入场、FEE=0.15% 往返），网格三维度 + 近顶跳过应用到池内金叉：
  - 逐笔：全部金叉 vs 白名单（沿用龙头训练段切点=可移植）vs 池50自身训练段重定切点（切点过拟合对照）
  - 组合模拟 10槽×10% 事件驱动：白名单+B3 vs 全部金叉死叉出场；H5 躺平基准为文字引用。
  - 缠论状态不算（太慢且已证明不可移植）。

输出：MACD完全分类网格研究.log / 网格_逐格统计.csv（20龙头+池50 逐格）/ 网格_池50终审.csv
"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))    # analysis/macd
BASE = os.path.dirname(HERE)                         # analysis
sys.path.insert(0, HERE)
sys.path.insert(0, BASE)

from MACD金叉死叉研究 import pick_leaders, prep, perf, DB, BASE as MBASE, CAPITAL, COMM, WARM, SPLIT
import MACD执行出场变体研究 as VX                      # 复用 simulate()/seg_mdd()/stamp()
from MACD执行出场变体研究 import seg_mdd

NEARTOP = 20          # 近顶格：距250日高点 ≤20 个交易日（与 C2 一致）
P50_START, P50_FEE, P50_SLOTS, P50_W = "2017-10-01", 0.0015, 10, 0.10
BREADTH_START = pd.Timestamp("2017-10-19")


# ============ Part 1：V1 / V2 状态机（VX.simulate 口径 + cfg 扩展） ============
def simulate_x(code, df, cfg):
    """在 VX.simulate 的状态机口径上实现 V1（结构失效退出）/ V2（中阴半仓）。
    非 v1/v2 的 cfg 原样转发 VX.simulate。"""
    if not (cfg.get("v1") or cfg.get("v2")):
        return VX.simulate(code, df, cfg)
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
    pos = None
    want_open, open_si = False, -1
    sellpend, sell_why = None, ""
    eq = np.full(n, np.nan)
    trades = []
    st = dict(signals=0, filled=0, half_mkt=0, half_add=0, v1_exit=0)

    def buy(t, px, spend, si):
        nonlocal cash, sh, pos
        spend = min(spend, cash)
        fee = max(5.0, spend * COMM / (1 + COMM))
        newsh = (spend - fee) / px
        if pos is None:
            pos = dict(si=si, e=t, cash0=cash, cost=spend, mfe=-1e9, max_close=c[t],
                       sig_low=l[si], want_add=False)
        else:
            pos["cost"] += spend
        cash -= spend
        sh += newsh
        pos["px_avg"] = pos["cost"] / sh

    def close_trade(x, px, why, is_open=False):
        nonlocal cash, sh, pos
        proceeds = sh * px
        fee = 0.0 if is_open else max(5.0, proceeds * COMM) + proceeds * VX.stamp(pd.Timestamp(dates[x]))
        total = cash + proceeds - fee
        trades.append(dict(thscode=code, signal_date=pd.Timestamp(dates[pos["si"]]),
                           entry_date=pd.Timestamp(dates[pos["e"]]), exit_date=pd.Timestamp(dates[x]),
                           entry_px=pos["px_avg"], exit_px=px, ret=total / pos["cash0"] - 1,
                           hold=x - pos["e"], mfe=pos["mfe"], why=why, is_open=is_open))
        if why == "v1":
            st["v1_exit"] += 1
        cash, sh, pos = total, 0.0, None

    for t in range(n):
        # --- 0. V2 次日开盘补另一半 ---
        if pos is not None and pos.get("want_add") and sellpend is None:
            if not ow_up[t]:
                st["half_add"] += 1
                buy(t, o[t], cash, pos["si"])
                pos["want_add"] = False
        # --- 1. 次日开盘卖（含 V1 一字跌停顺延） ---
        if pos is not None and sellpend == "open":
            if not ow_dn[t]:
                close_trade(t, o[t], sell_why)
                sellpend, sell_why = None, ""
        # --- 2. 持仓中：V1 结构失效（≤T-1 收盘信息） ---
        if pos is not None and sellpend is None and cfg.get("v1") \
                and c[t - 1] < pos["sig_low"] and t - 1 > pos["e"]:
            if not ow_dn[t]:
                close_trade(t, o[t], "v1")
            else:
                sellpend, sell_why = "open", "v1_defer"
        # --- 3. 次日开盘买（V2 半仓） ---
        if want_open and pos is None:
            want_open = False
            if ow_up[t]:
                pass                                # 一字涨停买不进，放弃
            elif cfg.get("v2"):
                st["half_mkt"] += 1
                buy(t, o[t], cash * 0.5, open_si)
            else:
                st["filled"] += 1
                buy(t, o[t], cash, open_si)
        # --- 4. 权益 ---
        eq[t] = cash + sh * c[t]
        if t < WARM or t >= n - 1:
            continue
        # --- 5. 收盘信号 ---
        if pos is None and not want_open and golden[t]:
            st["signals"] += 1
            want_open, open_si = True, t
        if pos is not None and sellpend is None:
            px = pos["px_avg"]
            pos["mfe"] = max(pos["mfe"], h[t] / px - 1)
            pos["max_close"] = max(pos["max_close"], c[t])
            if cfg.get("v2") and not pos["want_add"] and (t - pos["e"]) == 4 and c[t] > px:
                pos["want_add"] = True              # 第5个持仓日收盘浮盈 → 次日开盘补仓
            if death[t]:
                sellpend, sell_why = "open", "death"
    if pos is not None:
        close_trade(n - 1, c[-1], "eod", is_open=True)
    return trades, eq[WARM:], st


# ============ Part 2 用：特征组装与分格 ============
def grid_features(code, df, sig_rows, br):
    """信号行网格特征（≤信号日，与入场特征挖掘同口径）；br=宽度面板(date,pct20)。"""
    h, l, c = df.high.values, df.low.values, df.close.values
    golden = ((df.dif > df.dea) & (df.dif.shift(1) <= df.dea.shift(1))).values
    gc_cum = np.cumsum(golden)
    dd250 = c / pd.Series(h).rolling(250).max().values - 1
    d2i = {d: i for i, d in enumerate(pd.to_datetime(df.date))}
    sub = br.set_index("date")["pct20"]
    out = {}
    for s in sig_rows:
        i = d2i[pd.Timestamp(s)]
        key = pd.Timestamp(df.date.values[i])
        w = h[max(0, i - 249):i + 1]
        b = sub.asof(key)
        out[key] = dict(dd250=dd250[i],
                        cross_cnt120=int(gc_cum[i] - (gc_cum[i - 120] if i >= 120 else 0)),
                        pct20=(np.nan if pd.isna(b) else float(b)),
                        days_high250=len(w) - 1 - int(np.argmax(w)))
    return out


def cell_key_row(dd250, xcnt, pct20, cuts):
    q_dd, q_xc, q_br = cuts
    a = "深" if dd250 <= q_dd[0] else ("中" if dd250 <= q_dd[1] else "浅")
    b = "少" if xcnt <= q_xc else "多"
    cc = "无" if pd.isna(pct20) else ("低" if pct20 <= q_br[0] else ("中" if pct20 <= q_br[1] else "高"))
    return f"{a}/{b}/{cc}"


def assign_cells(F, cuts):
    """返回 (格键Series, 近顶掩码Series)。"""
    q_dd, q_xc, q_br = cuts
    dd = pd.cut(F.dd250, [-np.inf, q_dd[0], q_dd[1], np.inf],
                labels=["深", "中", "浅"]).astype(object)
    dd[F.dd250.isna()] = "无DD"
    xc = pd.Series(np.where(F.cross_cnt120 <= q_xc, "少", "多"), index=F.index)
    br = pd.cut(F.pct20, [-np.inf, q_br[0], q_br[1], np.inf], labels=["低", "中", "高"]).astype(object)
    br[F.pct20.isna()] = "无"
    key = dd.astype(str) + "/" + xc + "/" + pd.Series(br, index=F.index).astype(str)
    return key, (F.days_high250 <= NEARTOP)


def cell_perf(T, key, neartop, label):
    """逐格 × 三段统计（T 须含 signal_date/ret）。"""
    rows = []
    segs = [("全期", T.signal_date.notna()), ("训练段", T.signal_date < SPLIT),
            ("验证段", T.signal_date >= SPLIT)]
    for cell in sorted(set(key) | {"近顶禁入"}):
        m = (neartop if cell == "近顶禁入" else (key == cell)).values
        for sname, sm in segs:
            sub = T[m & sm.values]
            if len(sub):
                p = perf(sub.ret)
                rows.append(dict(来源=label, 格=cell, 时段=sname, n=p["n"],
                                 胜率=p["win_rate"], PF=p["pf"], 均收益=p["avg_ret"]))
            else:
                rows.append(dict(来源=label, 格=cell, 时段=sname, n=0,
                                 胜率=np.nan, PF=np.nan, 均收益=np.nan))
    return rows


# ============ Part 3：白名单格入场 + B3（可叠 V1） ============
def simulate_wl(code, df, wl, cuts, br, b3=True, v1=False):
    golden = ((df.dif > df.dea) & (df.dif.shift(1) <= df.dea.shift(1))).values
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    dif, dea = df.dif.values, df.dea.values
    dates = pd.to_datetime(df.date).values
    n = len(df)
    cand = {pd.Timestamp(dates[i]) for i in np.where(golden)[0]}
    feats = grid_features(code, df, list(cand), br)
    sig_ok = {}
    for k, f in feats.items():
        cell = cell_key_row(f["dd250"], f["cross_cnt120"], f["pct20"], cuts)
        sig_ok[k] = (cell in wl) and f["days_high250"] > NEARTOP
    d2i = {d: i for i, d in enumerate(pd.to_datetime(df.date))}
    golden_f = np.zeros(n, bool)
    for k, ok in sig_ok.items():
        if ok:
            golden_f[d2i[k]] = True
    death = (dif < dea) & (np.roll(dif, 1) >= np.roll(dea, 1))
    prev_c = np.concatenate(([c[0]], c[:-1]))
    ow_dn = (h == l) & (o < prev_c)
    ow_up = (h == l) & (o > prev_c)

    cash, sh = CAPITAL, 0.0
    pos = None
    want_open, open_si = False, -1
    sellpend, sell_why = None, ""
    eq = np.full(n, np.nan)
    trades = []
    st = dict(signals=int(golden_f.sum()), filled=0, v1_exit=0)

    def buy(t, px, spend, si):
        nonlocal cash, sh, pos
        spend = min(spend, cash)
        fee = max(5.0, spend * COMM / (1 + COMM))
        newsh = (spend - fee) / px
        if pos is None:
            pos = dict(si=si, e=t, cash0=cash, cost=spend, mfe=-1e9, max_close=c[t],
                       sig_low=l[si], trail_on=False, trail_stop=0.0)
        else:
            pos["cost"] += spend
        cash -= spend
        sh += newsh
        pos["px_avg"] = pos["cost"] / sh

    def close_trade(x, px, why, is_open=False):
        nonlocal cash, sh, pos
        proceeds = sh * px
        fee = 0.0 if is_open else max(5.0, proceeds * COMM) + proceeds * VX.stamp(pd.Timestamp(dates[x]))
        total = cash + proceeds - fee
        trades.append(dict(thscode=code, signal_date=pd.Timestamp(dates[pos["si"]]),
                           entry_date=pd.Timestamp(dates[pos["e"]]), exit_date=pd.Timestamp(dates[x]),
                           entry_px=pos["px_avg"], exit_px=px, ret=total / pos["cash0"] - 1,
                           hold=x - pos["e"], mfe=pos["mfe"], why=why, is_open=is_open))
        if why == "v1":
            st["v1_exit"] += 1
        cash, sh, pos = total, 0.0, None

    for t in range(n):
        if pos is not None and sellpend == "open":
            if not ow_dn[t]:
                close_trade(t, o[t], sell_why)
                sellpend, sell_why = None, ""
        if pos is not None and sellpend is None:
            px = pos["px_avg"]
            if v1 and c[t - 1] < pos["sig_low"] and t - 1 > pos["e"]:
                if not ow_dn[t]:
                    close_trade(t, o[t], "v1")
                else:
                    sellpend, sell_why = "open", "v1_defer"
            elif b3 and pos["trail_on"] and not ow_dn[t] and l[t] <= pos["trail_stop"]:
                close_trade(t, min(o[t], pos["trail_stop"]), "trailing")
        if want_open and pos is None:
            want_open = False
            if ow_up[t]:
                pass
            else:
                st["filled"] += 1
                buy(t, o[t], cash, open_si)
        eq[t] = cash + sh * c[t]
        if t < WARM or t >= n - 1:
            continue
        if pos is None and not want_open and golden_f[t]:
            want_open, open_si = True, t
        if pos is not None and sellpend is None:
            px = pos["px_avg"]
            pos["mfe"] = max(pos["mfe"], h[t] / px - 1)
            pos["max_close"] = max(pos["max_close"], c[t])
            if b3 and not pos["trail_on"] and pos["max_close"] >= px * 1.10:
                pos["trail_on"] = True
            if pos["trail_on"]:
                pos["trail_stop"] = max(pos["trail_stop"], pos["max_close"] * 0.92)
            if death[t]:
                sellpend, sell_why = "open", "death"
    if pos is not None:
        close_trade(n - 1, c[-1], "eod", is_open=True)
    return trades, eq[WARM:], st


# ============ Part 4：池50 ============
def build_pool50(con):
    """复刻 pool50_turnpoint_backtest.py 的数据装载 + 逐股指标 + 月度池（不含缠论）。"""
    df = con.execute("""SELECT v.thscode, v.date, v.close, v.high, v.low, v.volume, v.amount
      FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
      WHERE s.exchange <> 'BJ' ORDER BY v.thscode, v.date""").df()
    cats = df.thscode.astype("category")
    code_ids = cats.cat.codes.values.astype(np.int64)
    names = cats.cat.categories
    dates = df.date.values.astype("datetime64[D]")
    close = df.close.values.astype(float)
    high = df.high.values.astype(float)
    low = df.low.values.astype(float)
    vol = df.volume.values.astype(float)
    amt = df.amount.values.astype(float)
    N = len(df)
    del df
    r20 = np.full(N, np.nan); dd = np.full(N, np.nan); vratio = np.full(N, np.nan)
    amt60 = np.full(N, np.nan); age = np.arange(N, dtype=np.int64)
    dif_a = np.full(N, np.nan); dea_a = np.full(N, np.nan); gc = np.zeros(N, bool)
    h250 = np.full(N, np.nan); xcnt = np.full(N, np.nan); dd250_all = np.full(N, np.nan)
    uniq, starts = np.unique(code_ids, return_index=True)
    bounds = np.append(starts, N)
    print("  池50 逐股指标...", flush=True)
    for k in range(len(uniq)):
        a, b = bounds[k], bounds[k + 1]
        c = pd.Series(close[a:b]); h = pd.Series(high[a:b]); vv = pd.Series(vol[a:b])
        am = pd.Series(amt[a:b]); hv = h.values
        r20[a:b] = (c / c.shift(20) - 1).values
        dd[a:b] = (c / h.rolling(20).max() - 1).values
        vratio[a:b] = (vv.rolling(5).mean() / vv.shift(1).rolling(20).mean()).values
        amt60[a:b] = am.rolling(60).mean().values
        dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        dea = dif.ewm(span=9, adjust=False).mean()
        dif_a[a:b] = dif.values; dea_a[a:b] = dea.values
        g = ((dif > dea) & (dif.shift(1) <= dea.shift(1))).values
        gc[a:b] = g
        gcc = np.cumsum(g)
        nb = b - a
        xi = np.arange(nb)
        xcnt[a:b] = np.array([gcc[i] - (gcc[i - 120] if i >= 120 else 0) for i in xi])
        dd250_all[a:b] = (c / h.rolling(250).max() - 1).values
        for i in range(250, nb):
            w = hv[i - 249:i + 1]
            h250[a + i] = len(w) - 1 - int(np.argmax(w))
    age = age - starts[np.searchsorted(starts, np.arange(N), "right") - 1] + 1
    all_days = np.unique(dates)
    t0 = np.searchsorted(all_days, np.datetime64(P50_START))
    reb = all_days[t0::20]
    pools = []
    for k, T in enumerate(reb):
        nxt = reb[k + 1] if k + 1 < len(reb) else np.datetime64("2262-01-01")
        rows = np.where(dates == T)[0]
        ok = rows[(age[rows] >= 250) & ~np.isnan(r20[rows])]
        r = r20[ok]; thr = np.nanquantile(r, 0.8)
        f1 = r >= thr
        f2 = (dd[ok] >= -0.15) & (dd[ok] <= -0.05) & (vratio[ok] >= 1.6)
        rows_p = ok[~f1 & ~f2]
        pct = pd.Series(amt60[rows_p]).rank(pct=True).fillna(0).values
        dec = np.minimum((pct * 10).astype(int), 9)
        rng = np.random.default_rng(int(pd.Timestamp(str(T)).strftime("%Y%m%d")))
        picks = []
        for dv in range(10):
            m = np.where(dec == dv)[0]; kk = min(5, len(m))
            if kk:
                picks.append(rows_p[m[rng.choice(len(m), size=kk, replace=False)]])
        sel = np.concatenate(picks)[:50] if picks else np.array([], dtype=np.int64)
        pools.append((T, nxt, {names[code_ids[p]] for p in sel}))
    print(f"  池期数 {len(pools)}", flush=True)
    return dict(names=names, code_ids=code_ids, dates=dates, close=close, high=high, low=low,
                dif=dif_a, dea=dea_a, gc=gc, h250=h250, xcnt=xcnt, dd250=dd250_all,
                uniq=uniq, bounds=bounds, all_days=all_days, reb=reb, pools=pools)


def p50_trades(P, keep_fn):
    """池50逐笔回测：池内金叉 + T+1 收盘入场（复刻原脚本）、死叉收盘出场、FEE 往返。
    keep_fn(dd250, xcnt, h250, sig_ts) -> bool。"""
    trs = []
    names, dates, close = P["names"], P["dates"], P["close"]
    dif, dea, gc = P["dif"], P["dea"], P["gc"]
    for k in range(len(P["uniq"])):
        a, b = P["bounds"][k], P["bounds"][k + 1]
        if b - a < 60:
            continue
        name = names[P["uniq"][k]]
        ki = np.searchsorted(P["reb"], dates[a:b], "right") - 1
        in_p = np.array([ki[j] >= 0 and name in P["pools"][ki[j]][2] for j in range(b - a)])
        flat = True; pend = -1; sig_row = -1; e = -1; entry_px = np.nan
        for j in range(a, b):
            r = j - a
            if flat:
                if pend == r:
                    e = r; entry_px = close[j]; flat = False; pend = -1
                    continue
                if gc[j] and j + 1 < b and in_p[r] and keep_fn(
                        P["dd250"][j], P["xcnt"][j], P["h250"][j], pd.Timestamp(str(dates[j]))):
                    pend = r + 1; sig_row = r
            else:
                if dif[j] < dea[j] and dif[j - 1] >= dea[j - 1]:
                    trs.append(dict(code=name, sig=dates[sig_row], entry=dates[a + e],
                                    exit=dates[j], ret=close[j] / entry_px - 1 - P50_FEE,
                                    hold=j - (a + e), s_row=a + sig_row, e_row=a + e, x_row=j))
                    flat = True; pend = -1
        if not flat and sig_row >= 0:
            j = b - 1
            trs.append(dict(code=name, sig=dates[sig_row], entry=dates[a + e], exit=dates[j],
                            ret=close[j] / entry_px - 1 - P50_FEE, hold=j - (a + e),
                            s_row=a + sig_row, e_row=a + e, x_row=j, is_open=True))
    return trs


def p50_stats(tr):
    if not tr:
        return dict(n=0, 胜率=np.nan, 平均=np.nan, PF=np.nan, 盈亏比=np.nan)
    r = np.array([t["ret"] for t in tr]); w = r[r > 0]; l = r[r <= 0]
    return dict(n=len(r), 胜率=(r > 0).mean(), 平均=r.mean(),
                PF=(w.sum() / abs(l.sum())) if len(w) and len(l) and l.sum() != 0 else np.nan,
                盈亏比=(w.mean() / abs(l.mean())) if len(w) and len(l) else np.nan)


def p50_portfolio(P, trs, label, b3=False):
    """10槽×10% 事件驱动组合；出场=死叉，b3=True 时叠加追踪回吐（收盘口径，T+1 收盘成交）。
    结构同原脚本 portfolio2：先出场、再按事件日入场、每槽=总资产×10%。"""
    held = {}; cash = 1.0; eq = []
    ev = sorted(trs, key=lambda t: (t["entry"], t["code"]))
    ei = 0
    name2k = {P["names"][P["uniq"][k]]: k for k in range(len(P["uniq"]))}
    dates, close, dif, dea = P["dates"], P["close"], P["dif"], P["dea"]
    bounds, all_days, reb = P["bounds"], P["all_days"], P["reb"]
    for d in all_days[all_days >= reb[0]]:
        for nm in list(held):
            pos = held[nm]; k = name2k[nm]; a, b = bounds[k], bounds[k + 1]
            jj = np.searchsorted(dates[a:b], d, "right") - 1
            if jj < 0:
                continue
            r = a + jj; px = close[r]
            hit = None
            if b3:
                if r - 1 >= pos["e_row"]:
                    pos["max_close"] = max(pos["max_close"], close[r - 1])
                if pos["max_close"] >= pos["entry_px"] * 1.10:
                    pos["trail_stop"] = max(pos["trail_stop"], pos["max_close"] * 0.92)
                if px <= pos["trail_stop"]:
                    hit = "trail"
            if not hit and r - 1 >= a and dif[r] < dea[r] and dif[r - 1] >= dea[r - 1]:
                hit = "dc"
            elif not hit and r == b - 1:
                hit = "eod"
            if hit:
                cash += pos["sh"] * px * (1 - P50_FEE / 2)
                del held[nm]
        while ei < len(ev) and ev[ei]["entry"] == d:
            t = ev[ei]; ei += 1
            if t["code"] in held or len(held) >= P50_SLOTS:
                continue
            k = name2k[t["code"]]; a, b = bounds[k], bounds[k + 1]
            e = np.searchsorted(dates[a:b], d) + a
            if e >= b:
                continue
            mk = cash
            for nm, pos in held.items():
                kk = name2k[nm]; aa, bb = bounds[kk], bounds[kk + 1]
                jj = np.searchsorted(dates[aa:bb], d, "right") - 1
                mk += pos["sh"] * (close[aa + jj] if jj >= 0 else pos["entry_px"])
            px = close[e]
            if px > 0:
                alloc = mk * P50_W
                cash -= alloc
                held[t["code"]] = dict(sh=alloc * (1 - P50_FEE / 2) / px, entry_px=px,
                                       e_row=e, max_close=px, trail_stop=0.0)
        v = cash
        for nm, pos in held.items():
            k = name2k[nm]; a, b = bounds[k], bounds[k + 1]
            jj = np.searchsorted(dates[a:b], d, "right") - 1
            v += pos["sh"] * (close[a + jj] if jj >= 0 else pos["entry_px"])
        eq.append((d, v))
    s = pd.DataFrame(eq, columns=["date", "port"])
    e = s.port.values
    yrs = (pd.Timestamp(str(s.date.iloc[-1])) - pd.Timestamp(str(s.date.iloc[0]))).days / 365.25
    mdd = (e / np.maximum.accumulate(e) - 1).min()
    ann = (e[-1] / e[0]) ** (1 / yrs) - 1
    print(f"  [组合] {label}: 期末 {e[-1]:.3f} ({e[-1]/e[0]-1:+.1%})  年化 {ann:+.2%}  "
          f"MDD {mdd:.1%}  Calmar {ann/abs(mdd):.2f}", flush=True)
    return dict(组合=label, 期末=e[-1], 年化=ann, MDD=mdd, Calmar=ann / abs(mdd))


# ============ 主流程 ============
def main():
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 300)
    t_start = time.time()
    import duckdb
    con = duckdb.connect(DB, read_only=True)
    lead, _ = pick_leaders(con)
    raw = con.execute("""SELECT thscode, date, open, high, low, close, volume, amount FROM v_daily_qfq
                         WHERE thscode IN (SELECT UNNEST(?::VARCHAR[])) ORDER BY thscode, date""",
                      [lead.thscode.tolist()]).fetchdf()
    raw["date"] = pd.to_datetime(raw.date)
    hs = pd.read_csv(f"{MBASE}/hs300_daily.csv", parse_dates=["date"])[["date", "close"]]
    hs["hs_below_ma60"] = hs.close < hs.close.rolling(60).mean()
    hs = hs[["date", "hs_below_ma60"]]
    br = pd.read_csv(f"{MBASE}/市场宽度历史面板.csv", parse_dates=["date"])[["date", "pct20"]]
    DF = {code: prep(df.reset_index(drop=True), hs) for code, df in raw.groupby("thscode")}
    DATES = {code: pd.to_datetime(df.date).values[WARM:] for code, df in DF.items()}
    print(f"股票池 {len(DF)} 只；训练段信号 < {SPLIT.date()}，验证段 >= {SPLIT.date()}\n", flush=True)

    # ---------- 基线 ----------
    base_eq, base_tr = {}, []
    for code, df in DF.items():
        tr, eq, _ = VX.simulate(code, df, dict(cat="B"))
        base_eq[code] = eq
        base_tr += tr
    B0 = pd.DataFrame(base_tr)
    bcl = B0[~B0.is_open]
    bp = perf(bcl.ret)
    print(f"[基线] 已平仓 {bp['n']} 笔 胜率 {bp['win_rate']:.4f} PF {bp['pf']:.4f} "
          f"均收益 {bp['avg_ret']:.4f}", flush=True)
    assert 1720 <= bp["n"] <= 1730 and abs(bp["pf"] - 1.366) < 0.03, "基线复现失败"
    split64 = np.datetime64(SPLIT)
    base_pt, base_mdd = {}, {}
    for code, eq in base_eq.items():
        k = int(np.searchsorted(DATES[code], split64))
        kk = min(max(k, 1), len(eq) - 1)
        base_pt[code] = (eq[kk - 1], eq[-1])
        base_mdd[code] = (seg_mdd(eq, 1, k), seg_mdd(eq, 2, k), seg_mdd(eq, 0, k))

    def eq_stats(eq_by_code, pname):
        finals, ratios, mdds = [], [], []
        for code, eq in eq_by_code.items():
            d = DATES[code]
            k = int(np.searchsorted(d, split64))
            kk = min(max(k, 1), len(eq) - 1)
            if pname == "全期":
                pt, seg, bi = eq[-1], eq, 2
            elif pname == "训练段":
                pt, seg, bi = eq[kk - 1], eq[:kk], 0
            else:
                pt, seg, bi = eq[-1], eq[kk - 1:], 1
            seg = seg / seg[0]
            finals.append(pt)
            ratios.append(pt / base_pt[code][0 if pname == "训练段" else 1])
            mdds.append(float((seg / np.maximum.accumulate(seg) - 1).min()) - base_mdd[code][bi])
        return dict(期末权益中位=float(np.median(finals)),
                    期末比=float(np.exp(np.mean(np.log(ratios)))),
                    MDD中位_vs基线=float(np.median(mdds)))

    def report(name, T, eq_by_code):
        rows = []
        closed = T[~T.is_open]
        for pname in ["全期", "训练段", "验证段"]:
            sub = closed if pname == "全期" else \
                closed[(closed.signal_date < SPLIT) if pname == "训练段" else (closed.signal_date >= SPLIT)]
            es = eq_stats(eq_by_code, pname)
            if len(sub):
                p = perf(sub.ret)
                rows.append(dict(变体=name, 时段=pname, 笔数=p["n"], 胜率=p["win_rate"], PF=p["pf"],
                                 均收益=p["avg_ret"], 盈亏比=p["payoff"],
                                 持仓中位=float(sub.hold.median()),
                                 期末权益中位数=es["期末权益中位"], 期末比=es["期末比"],
                                 MDD中位_vs基线=es["MDD中位_vs基线"]))
            else:
                rows.append(dict(变体=name, 时段=pname, 笔数=0, 胜率=np.nan, PF=np.nan,
                                 均收益=np.nan, 盈亏比=np.nan, 持仓中位=np.nan,
                                 期末权益中位数=es["期末权益中位"], 期末比=es["期末比"],
                                 MDD中位_vs基线=es["MDD中位_vs基线"]))
        return rows

    # ---------- Part 1 ----------
    print("\n=== Part 1: V1 结构失效退出 / V2 中阴半仓 ===", flush=True)
    p1_rows = []
    for name, cfg in [("V1", dict(cat="B", v1=True)), ("V2", dict(cat="B", v2=True))]:
        all_tr, eqc = [], {}
        stot = dict(signals=0, filled=0, half_mkt=0, half_add=0, v1_exit=0)
        for code, df in DF.items():
            tr, eq, st = simulate_x(code, df, cfg)
            all_tr += tr
            eqc[code] = eq
            for kk in stot:
                stot[kk] += st.get(kk, 0)
        T = pd.DataFrame(all_tr)
        p1_rows += report(name, T, eqc)
        print(f"{name}: 已平仓 {(~T.is_open).sum()}  出场分布 {T.why.value_counts(normalize=True).round(3).to_dict()}",
              flush=True)
        if name == "V2":
            print(f"  V2 补仓统计: {stot}", flush=True)
    p1_rows += report("B0基线", B0, base_eq)
    R1 = pd.DataFrame(p1_rows)
    print(R1.to_string(index=False, float_format=lambda x: f"{x:,.3f}"), flush=True)

    # ---------- Part 2 ----------
    print("\n=== Part 2: 完全分类网格（切点只用训练段） ===", flush=True)
    feat_rows = []
    for code, g in bcl.groupby("thscode"):
        f = grid_features(code, DF[code], g.signal_date.tolist(), br)
        for s, d in f.items():
            feat_rows.append(dict(thscode=code, signal_date=s, **d))
    F = pd.DataFrame(feat_rows).merge(
        bcl[["thscode", "signal_date", "ret", "hold"]], on=["thscode", "signal_date"])
    tr_m = F.signal_date < SPLIT
    cuts = (F.dd250[tr_m].quantile(1 / 3), F.dd250[tr_m].quantile(2 / 3)), \
           float(F.cross_cnt120[tr_m].median()), \
           (F.pct20[tr_m].quantile(1 / 3), F.pct20[tr_m].quantile(2 / 3))
    base_tr = perf(F.ret[tr_m])
    print(f"切点(训练段): dd250 三分位 {cuts[0][0]:.4f}/{cuts[0][1]:.4f}；"
          f"cross_cnt120 中位 {cuts[1]:.0f}；pct20 三分位 {cuts[2][0]:.4f}/{cuts[2][1]:.4f}", flush=True)
    print(f"训练段全体: n={base_tr['n']} 胜率={base_tr['win_rate']:.3f} PF={base_tr['pf']:.3f} "
          f"均收益={base_tr['avg_ret']:.4f}", flush=True)

    key, neartop = assign_cells(F, cuts)
    grid_rows = cell_perf(F, key, neartop, "20龙头")
    G = pd.DataFrame(grid_rows)
    pv = G[G.时段 == "训练段"].set_index("格")
    show_cells = G[G.时段 == "训练段"][["格", "n", "胜率", "PF", "均收益"]]
    print("\n训练段逐格统计:", flush=True)
    print(show_cells.to_string(index=False, float_format=lambda x: f"{x:,.3f}"), flush=True)

    wl, bl = [], []
    for cell in sorted(set(key)):
        r = pv.loc[cell]
        if r["n"] >= 20 and r["均收益"] > base_tr["avg_ret"] and r["胜率"] >= base_tr["win_rate"]:
            wl.append(cell)
        elif r["n"] >= 20 and r["均收益"] < 0:
            bl.append(cell)
    print(f"\n白名单格({len(wl)}): {wl}", flush=True)
    print(f"黑名单格({len(bl)}): {bl}", flush=True)
    wm = F[np.array([k in wl for k in key]) & (~neartop).values]
    bm = F[np.array([k in bl for k in key]) & (~neartop).values]
    for label, sub in [("白名单格整体", wm), ("黑名单格整体(对照)", bm)]:
        for sname, sm in [("训练段", tr_m), ("验证段", ~tr_m)]:
            p = perf(sub.ret[sm.loc[sub.index].values]) if len(sub) else dict(n=0, win_rate=np.nan, pf=np.nan, avg_ret=np.nan)
            print(f"  {label} {sname}: n={p['n']} 胜率={p['win_rate']:.3f} PF={p['pf']:.3f} "
                  f"均收益={p['avg_ret']:.4f}", flush=True)
    for sname, sm in [("训练段", tr_m), ("验证段", ~tr_m)]:
        sub = wm[sm.loc[wm.index].values]
        finals = {code: CAPITAL * (1 + g.ret).prod() for code, g in sub.groupby("thscode")}
        rr = [finals.get(c, CAPITAL) / base_pt[c][0 if sname == "训练段" else 1] for c in DF]
        print(f"  白名单每股期末比({sname}): 中位 {np.median(rr):.3f}  "
              f"几何均值比 {np.exp(np.mean(np.log(rr))):.3f}", flush=True)
    piv = G[G.来源 == "20龙头"].pivot_table(index="格", columns="时段", values="均收益")
    stab = piv[piv["训练段"] * piv["验证段"] < 0]
    print(f"\n格子稳定性: 训练/验证均收益符号翻转 {len(stab)}/{len(piv)} 格: {stab.index.tolist()}", flush=True)
    print("[多重比较声明] 19 格为样本内诊断，训练段逐格统计不可单独采信；"
          "仅白名单整体规则的验证段整体验收有效（一次）。", flush=True)

    # ---------- Part 3 ----------
    print("\n=== Part 3: 组合系统（白名单+B3，视 V1 期末比决定是否叠加） ===", flush=True)
    v1_full = R1[(R1.变体 == "V1") & (R1.时段 == "全期")].期末比.iloc[0]
    use_v1 = bool(v1_full >= 0.98)
    print(f"V1 全期期末比 {v1_full:.3f} → {'加入 V1' if use_v1 else '不加入 V1'}", flush=True)
    combos = [("白名单+死叉", dict(b3=False)), ("白名单+B3", dict(b3=True))]
    if use_v1:
        combos.append(("白名单+B3+V1", dict(b3=True, v1=True)))
    p3_rows = []
    for cname, kw in combos:
        all_tr, eqc = [], {}
        for code, df in DF.items():
            tr, eq, _ = simulate_wl(code, df, wl, cuts, br, **kw)
            all_tr += tr
            eqc[code] = eq
        p3_rows += report(cname, pd.DataFrame(all_tr), eqc)
    all_tr, eqc = [], {}
    for code, df in DF.items():
        tr, eq, _ = VX.simulate(code, df, dict(cat="C", neartop=True))
        all_tr += tr; eqc[code] = eq
    p3_rows += report("仅C2近顶跳过", pd.DataFrame(all_tr), eqc)
    p3_rows += report("B0基线", B0, base_eq)
    R3 = pd.DataFrame(p3_rows)
    print(R3.to_string(index=False, float_format=lambda x: f"{x:,.3f}"), flush=True)

    # ---------- Part 4 ----------
    print("\n=== Part 4: 池50终审（H5 月度池 N=50，T+1 收盘入场，FEE 0.15% 往返） ===", flush=True)
    P = build_pool50(con)
    con.close()
    breadth = br.set_index("date")["pct20"]

    # 池内全部信号（用于池50自身切点估计 + 逐格统计）
    sig_rows = []
    for k in range(len(P["uniq"])):
        a, b = P["bounds"][k], P["bounds"][k + 1]
        if b - a < 60:
            continue
        name = P["names"][P["uniq"][k]]
        ki = np.searchsorted(P["reb"], P["dates"][a:b], "right") - 1
        for j in range(a, b):
            if P["gc"][j] and j + 1 < b and ki[j - a] >= 0 and name in P["pools"][ki[j - a]][2]:
                sig_rows.append((P["dd250"][j], P["xcnt"][j], P["h250"][j],
                                 pd.Timestamp(str(P["dates"][j]))))
    sig_df = pd.DataFrame(sig_rows, columns=["dd250", "xcnt", "h250", "sig"])
    sig_tr = (sig_df.sig < SPLIT).values
    brv = np.array(breadth.reindex(sig_df.sig, method="ffill").values, dtype=float)
    brv[np.array(sig_df.sig < BREADTH_START)] = np.nan
    sig_df["pct20"] = brv
    cuts_p50 = (np.nanquantile(sig_df.dd250[sig_tr], 1 / 3), np.nanquantile(sig_df.dd250[sig_tr], 2 / 3)), \
               float(np.nanmedian(sig_df.xcnt[sig_tr])), \
               (np.nanquantile(brv[sig_tr], 1 / 3), np.nanquantile(brv[sig_tr], 2 / 3))
    print(f"池50自身训练段切点: dd250 {cuts_p50[0][0]:.4f}/{cuts_p50[0][1]:.4f}  "
          f"xcnt {cuts_p50[1]:.0f}  pct20 {cuts_p50[2][0]:.4f}/{cuts_p50[2][1]:.4f}", flush=True)

    def make_keep(cuts_use):
        def kf(dd250, xcnt, h250, sig_ts):
            if h250 <= NEARTOP or np.isnan(dd250):
                return False
            b = breadth.asof(sig_ts)
            return cell_key_row(dd250, xcnt, b, cuts_use) in wl
        return kf

    def kf_all(dd250, xcnt, h250, sig_ts):
        return True

    def kf_nt(dd250, xcnt, h250, sig_ts):
        return not (h250 <= NEARTOP or np.isnan(dd250))

    groups = [("全部金叉", kf_all), ("近顶跳过", kf_nt),
              ("白名单(龙头切点)", make_keep(cuts)),
              ("白名单(池50重定切点)", make_keep(cuts_p50))]
    tr_by, p4_rows = {}, []
    for gname, kf in groups:
        trs = p50_trades(P, kf)
        tr_by[gname] = trs
        for sname in ["全期", "训练段", "验证段"]:
            if sname == "训练段":
                sub = [t for t in trs if pd.Timestamp(str(t["sig"])) < SPLIT]
            elif sname == "验证段":
                sub = [t for t in trs if pd.Timestamp(str(t["sig"])) >= SPLIT]
            else:
                sub = trs
            s = p50_stats(sub)
            p4_rows.append(dict(组=gname, 时段=sname, n=s["n"], 胜率=s["胜率"], 平均=s["平均"],
                                PF=s["PF"], 盈亏比=s["盈亏比"]))
        s = p50_stats(trs)
        print(f"{gname}: n={s['n']} 胜率={s['胜率']:.1%} 平均={s['平均']:+.2%} PF={s['PF']:.3f}",
              flush=True)
    R4 = pd.DataFrame(p4_rows)

    # 池50 逐格统计（全部金叉组，两种切点口径），并入总表
    TA = pd.DataFrame([dict(signal_date=pd.Timestamp(str(t["sig"])), ret=t["ret"],
                            dd250=P["dd250"][t["s_row"]], cross_cnt120=P["xcnt"][t["s_row"]],
                            pct20=breadth.asof(pd.Timestamp(str(t["sig"]))),
                            days_high250=P["h250"][t["s_row"]]) for t in tr_by["全部金叉"]])
    for lab, cu in [("池50(龙头切点)", cuts), ("池50(池50重定切点)", cuts_p50)]:
        ka, na = assign_cells(TA, cu)
        grid_rows += cell_perf(TA, ka, na, lab)
    G = pd.DataFrame(grid_rows)
    G.to_csv(f"{HERE}/网格_逐格统计.csv", index=False, encoding="utf-8-sig")

    print("\n=== Part 4 组合模拟（10槽×10%，事件驱动） ===", flush=True)
    port_rows = [p50_portfolio(P, tr_by["白名单(龙头切点)"], "白名单+死叉出场", b3=False),
                 p50_portfolio(P, tr_by["白名单(龙头切点)"], "白名单+B3出场", b3=True),
                 p50_portfolio(P, tr_by["全部金叉"], "全部金叉(死叉出场)"),
                 p50_portfolio(P, tr_by["全部金叉"], "全部金叉+B3出场", b3=True)]
    PR = pd.DataFrame(port_rows)
    print(PR.to_string(index=False, float_format=lambda x: f"{x:,.3f}"), flush=True)
    print("\n[基准] H5 同池躺平(时间退出不择时): 年化 +6.2%(训练)/+16.3%(样本外) "
          "MDD −39.4%/−32.6%（文字引用 pool50_turnpoint_backtest.py 结论）", flush=True)

    with open(f"{HERE}/网格_池50终审.csv", "w", encoding="utf-8-sig") as fh:
        R4.to_csv(fh, index=False)
        fh.write("\n=== 组合模拟（10槽×10%, 事件驱动, T+1收盘, FEE 0.15%往返） ===\n")
        PR.to_csv(fh, index=False)

    print(f"\n输出: 网格_逐格统计.csv({len(G)}行) / 网格_池50终审.csv；"
          f"耗时 {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
