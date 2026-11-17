#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶梯双K / 双系统 —— 审计修正版回测

相对原版 (ladder_doublek_backtest.py) 的修正：
1. 信号日收盘确认，次日开盘入场；放量用信号日全日成交额（收盘已知），
   不再出现“用全日量确认、却按日内突破价成交”的时点错位。
2. 突破需收盘价也站上突破位（原版只看日内最高价）。
3. 今天开盘/收盘涨停不买；跌停不卖顺延；涨跌停幅度按板块与日期区分。
4. 佣金、卖出印花税、滑点全部计入逐笔与组合现金。
5. 组合模拟按信号质量（放量倍数）排序，并给出随机成交顺序的分布，
   不再只报一条路径。
6. 原版作为对照保留；本脚本不声称样本外，所有规则都是在 2016-2026
   全样本上选择的。

仍然无法修正的：
- 数据库只有现存股票，退市未补（幸存者偏差）；
- ST 5% 涨跌停无法识别；
- 没有 14:30 的分时成交量，只能用“收盘确认 + 次日开盘”作为保守代理；
- 2023-2026 不是预注册盲测。

用法：
  python3 ladder_doublek_corrected.py --quick          # 5 只股票冒烟测试
  python3 ladder_doublek_corrected.py                  # 全样本 A/B + 组合
  python3 ladder_doublek_corrected.py --pit            # 另跑 PIT 中证500 子样本
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_doublek_backtest as base  # noqa: E402

DB_PATH = os.environ.get(
    "HITHINK_DB",
    os.path.expanduser("~/Library/Application Support/hithink-finance/data/market.duckdb"),
)
base.DB_PATH = DB_PATH

OUT_DIR = os.path.join(HERE, "交易系统_审计修正")
os.makedirs(OUT_DIR, exist_ok=True)

COMM = 0.0001          # 万1，双边
SLIP = 0.0005          # 万5，单边
STAMP_OLD = 0.001      # 2023-08-28 前卖出印花税
STAMP_NEW = 0.0005     # 2023-08-28 起
STAMP_CUTOFF = pd.Timestamp("2023-08-28")


def stamp_rate(date) -> float:
    return STAMP_NEW if pd.Timestamp(date) >= STAMP_CUTOFF else STAMP_OLD


def limit_for(sym: str, date) -> float:
    """按板块与制度日期返回涨跌停幅度。ST 5% 无法识别，未处理。"""
    tic = sym.split(".")[0]
    d = pd.Timestamp(date)
    if sym.endswith(".BJ"):
        return 0.30 if d >= pd.Timestamp("2021-11-15") else 0.10
    if tic.startswith("688"):
        return 0.20 if d >= pd.Timestamp("2019-07-22") else 0.10
    if tic.startswith("30"):
        return 0.20 if d >= pd.Timestamp("2020-08-24") else 0.10
    return 0.10


def build_lim(sym: str, dates) -> np.ndarray:
    return np.array([limit_for(sym, d) for d in dates], dtype=float)


def backtest_symbol_corrected(
    sym, dates, o, h, l, c, amt, vmed, reg_ok=None, universe_ok=None,
    risk_pct=0.02, ma_n=24, be_tol=0.002, stop_buf=0.001,
    max_risk_dist=0.06, pause_days=0, tp_ratio=1.0, trail=True,
    vol_mult=None, trend_gate="none", brk_n=2, gap_mult=1.5,
    entry_mode="next_open", vol_scale=0.8,
    comm=COMM, slip=SLIP,
):
    """信号日收盘确认。

    entry_mode="next_open"：次日开盘入场（完全因果，保守）；
    entry_mode="close"：信号日收盘价入场，放量按 vol_scale 折减为 14:30 前的量
    （14:30 代理，仍然不是分时数据，只用于给出乐观边界）。
    """
    n = len(c)
    if n < ma_n + 10:
        return [], None
    lim = build_lim(sym, dates)
    ma = pd.Series(c).rolling(ma_n).mean().to_numpy()

    zig = []

    def zig_update(t):
        cnd = t - 1
        if cnd < 1:
            return
        pts = []
        if h[cnd] > h[cnd - 1] and h[cnd] > h[t]:
            pts.append(("H", h[cnd]))
        if l[cnd] < l[cnd - 1] and l[cnd] < l[t]:
            pts.append(("L", l[cnd]))
        for typ, price in pts:
            if zig and zig[-1][0] == typ:
                if (typ == "H" and price > zig[-1][1]) or (typ == "L" and price < zig[-1][1]):
                    zig[-1] = (typ, price, cnd)
            else:
                zig.append((typ, price, cnd))
        del zig[:-12]

    def last_two(typ):
        p = [x for x in zig if x[0] == typ]
        return p[-2:] if len(p) >= 2 else None

    def last_swing(typ):
        p = [x for x in zig if x[0] == typ]
        return p[-1] if p else None

    ladder_on, ladder_high = False, None
    box_on, box_top, box_bot = False, None, None
    pause_until, consec_loss = -1, 0
    capital, cash, shares = 1.0, 1.0, 0.0
    pos, pending, exit_pending = None, None, False
    trades = []
    equity = np.empty(n)

    def sell_proceeds(px, sh, date):
        return sh * px * (1 - comm - stamp_rate(date))

    def buy_cost(px, sh):
        return sh * px * (1 + comm)

    def register_exit(px, t, reason):
        nonlocal shares, pos, exit_pending, consec_loss, pause_until, cash
        proceeds = sell_proceeds(px, shares, dates[t])
        cash += proceeds
        cost = buy_cost(pos["entry"], shares)
        pnl = proceeds - cost
        trades.append({**pos, "exit_i": t, "exit": px, "reason": reason,
                       "pnl": pnl, "R": pnl / pos["risk_amt"],
                       "exit_date": dates[t]})
        if pnl < 0:
            consec_loss += 1
            if pause_days > 0 and consec_loss >= 2:
                pause_until = t + pause_days
                consec_loss = 0
        else:
            consec_loss = 0
        shares, pos, exit_pending = 0.0, None, False

    for t in range(n):
        # ---- 0. 执行昨日收盘产生的信号（今日开盘）----
        if pending is not None:
            if pos is None and not exit_pending and t == pending["signal_i"] + 1:
                open_up = o[t] >= c[t - 1] * (1 + lim[t] - 0.002)
                open_dn = o[t] <= c[t - 1] * (1 - lim[t] + 0.002)
                entry = o[t] * (1 + slip)
                risk = entry - pending["stop"]
                if (not open_up and not open_dn and risk > 0
                        and risk / entry <= max_risk_dist
                        and risk <= gap_mult * pending["planned_risk"]):
                    sh = min(capital * risk_pct / risk, cash / (entry * (1 + comm)))
                    if sh > 0:
                        cash -= buy_cost(entry, sh)
                        shares = sh
                        pos = {"entry_i": t, "entry": entry, "stop": pending["stop"],
                               "tp": entry + risk * tp_ratio, "shares": sh,
                               "risk_amt": sh * risk, "eq0": capital, "armed": False,
                               "signal_i": pending["signal_i"],
                               "vol_ratio": pending["vol_ratio"],
                               "risk_dist": risk / entry}
                        if c[t] < pos["stop"]:
                            exit_pending = True
            pending = None

        # ---- 1. 结构状态（只用 t-1 及之前）----
        struct_up = False
        if t > 0:
            hh, ll = last_two("H"), last_two("L")
            if hh and ll and hh[0][1] > hh[1][1] and ll[0][1] > ll[1][1]:
                ladder_on, ladder_high = True, hh[1][1]
            if ladder_on and c[t - 1] > ladder_high:
                ladder_on = False
            if hh and ll and hh[0][1] > hh[1][1] and ll[0][1] < ll[1][1]:
                if not box_on:
                    box_on, box_top, box_bot = True, hh[1][1], ll[1][1]
            if box_on and (c[t - 1] > box_top or c[t - 1] < box_bot):
                box_on = False
            struct_up = bool(hh and ll and hh[0][1] < hh[1][1] and ll[0][1] < ll[1][1])

        # ---- 2. 持仓管理（t > entry_i，T+1）----
        if exit_pending:
            down_sealed = o[t] <= c[t - 1] * (1 - lim[t] + 0.002)
            if not down_sealed:
                register_exit(o[t] * (1 - slip), t, "stop")
        elif pos and t > pos["entry_i"]:
            if trail:
                if not pos["armed"] and h[t] >= pos["tp"]:
                    pos["armed"] = True
                    pos["stop"] = max(pos["stop"], pos["entry"])
                if pos["armed"]:
                    sw = last_swing("L")
                    if sw:
                        pos["stop"] = max(pos["stop"], sw[1] * (1 - stop_buf))
                if c[t] < pos["stop"]:
                    if t == n - 1:
                        register_exit(c[t] * (1 - slip), t, "trail")
                    else:
                        exit_pending = True
            else:
                if not pos["armed"] and h[t] >= pos["tp"] * (1 - be_tol):
                    pos["stop"] = max(pos["stop"], pos["entry"])
                    pos["armed"] = True
                if h[t] >= pos["tp"]:
                    register_exit(pos["tp"] * (1 - slip), t, "tp")
                elif c[t] < pos["stop"]:
                    if t == n - 1:
                        register_exit(c[t] * (1 - slip), t, "stop")
                    else:
                        exit_pending = True

        # ---- 3. 收盘更新摆动点（确认 t-1）----
        if t >= 2:
            zig_update(t)

        # ---- 4. 收盘产生信号（次日开盘执行）----
        if pos is None and not exit_pending and pending is None and t >= max(ma_n, brk_n):
            if trend_gate == "price":
                trend_ok = c[t] > ma[t] if not np.isnan(ma[t]) else False
            elif trend_gate == "slope":
                trend_ok = t >= 6 and not np.isnan(ma[t - 6]) and ma[t] > ma[t - 6]
            elif trend_gate == "struct":
                trend_ok = struct_up
            else:
                trend_ok = True

            level = np.max(h[t - brk_n:t])
            breakout = h[t] > level and c[t] > level
            blocked = ladder_on or box_on or t <= pause_until
            # 今日涨停/开盘涨停都不追；收盘确认后次日开盘买
            if c[t] >= c[t - 1] * (1 + lim[t] - 0.002):
                blocked = True
            if o[t] >= c[t - 1] * (1 + lim[t] - 0.002):
                blocked = True
            if reg_ok is not None and not reg_ok[t]:
                blocked = True
            if universe_ok is not None and not universe_ok[t]:
                blocked = True
            vol_available = amt[t] * (vol_scale if entry_mode == "close" else 1.0)
            if vol_mult is not None and vmed is not None and not np.isnan(vmed[t]):
                if not vol_available > vol_mult * vmed[t]:
                    blocked = True

            if trend_ok and breakout and not blocked:
                sw = last_swing("L")
                base_px = sw[1] if sw else np.min(l[max(0, t - 10):t])
                stop = base_px * (1 - stop_buf)
                planned_risk = c[t] - stop
                if 0 < planned_risk / c[t] <= max_risk_dist:
                    vol_ratio = (vol_available / vmed[t]) if (vmed is not None and
                                                              not np.isnan(vmed[t]) and vmed[t] > 0) else np.nan
                    if entry_mode == "close":
                        entry = c[t] * (1 + slip)
                        risk = entry - stop
                        if 0 < risk / entry <= max_risk_dist:
                            sh = min(capital * risk_pct / risk, cash / (entry * (1 + comm)))
                            if sh > 0:
                                cash -= buy_cost(entry, sh)
                                shares = sh
                                pos = {"entry_i": t, "entry": entry, "stop": stop,
                                       "tp": entry + risk * tp_ratio, "shares": sh,
                                       "risk_amt": sh * risk, "eq0": capital, "armed": False,
                                       "signal_i": t, "vol_ratio": vol_ratio,
                                       "risk_dist": risk / entry}
                                if c[t] < pos["stop"]:
                                    exit_pending = True
                    else:
                        pending = {
                            "signal_i": t, "stop": stop,
                            "planned_risk": planned_risk,
                            "vol_ratio": vol_ratio,
                        }

        equity[t] = cash + shares * c[t]
        capital = equity[t]

    if pos:
        px = c[-1] * (1 - slip)
        register_exit(px, n - 1, "eod")
        equity[-1] = cash

    tr = pd.DataFrame(trades)
    return tr, equity


def backtest_mr_corrected(
    sym, dates, o, h, l, c, allow=None, rsi_n=14, rsi_in=30.0,
    ma_exit=10, max_hold=10, stop_pct=0.08, risk_pct=0.02,
    comm=COMM, slip=SLIP,
):
    """超跌反弹：RSI(14)<30 收盘信号，次日开盘买；
    收盘站回 MA10 / 持满10日 / 收盘跌破-8% 止损 → 次日开盘卖。"""
    n = len(c)
    if n < 40:
        return [], None
    lim = build_lim(sym, dates)
    s = pd.Series(c)
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_n, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss)).to_numpy()
    maxit = s.rolling(ma_exit).mean().to_numpy()

    capital, cash, shares = 1.0, 1.0, 0.0
    pos, exit_pending = None, False
    trades = []
    equity = np.empty(n)

    def sell_proceeds(px, sh, date):
        return sh * px * (1 - comm - stamp_rate(date))

    def buy_cost(px, sh):
        return sh * px * (1 + comm)

    def register_exit(px, t, reason):
        nonlocal shares, pos, exit_pending, cash
        proceeds = sell_proceeds(px, shares, dates[t])
        cash += proceeds
        pnl = proceeds - buy_cost(pos["entry"], shares)
        trades.append({**pos, "exit_i": t, "exit": px, "reason": reason,
                       "pnl": pnl, "R": pnl / pos["risk_amt"],
                       "exit_date": dates[t]})
        shares, pos, exit_pending = 0.0, None, False

    for t in range(n):
        if exit_pending:
            down_sealed = o[t] <= c[t - 1] * (1 - lim[t] + 0.002)
            if not down_sealed:
                register_exit(o[t] * (1 - slip), t, "mr_exit")
        elif pos and t > pos["entry_i"]:
            if c[t] < pos["stop"] or c[t] > maxit[t] or t - pos["entry_i"] >= max_hold:
                if t == n - 1:
                    register_exit(c[t] * (1 - slip), t, "mr_exit")
                else:
                    exit_pending = True

        if pos is None and not exit_pending and t >= 1 and not np.isnan(rsi[t - 1]):
            open_up = o[t] >= c[t - 1] * (1 + lim[t] - 0.002)
            open_dn = o[t] <= c[t - 1] * (1 - lim[t] + 0.002)
            ok = allow[t - 1] if allow is not None else True
            if ok and rsi[t - 1] < rsi_in and not open_up and not open_dn:
                entry = o[t] * (1 + slip)
                stop = entry * (1 - stop_pct)
                risk = entry - stop
                sh = min(capital * risk_pct / risk, cash / (entry * (1 + comm)))
                if sh > 0:
                    cash -= buy_cost(entry, sh)
                    shares = sh
                    pos = {"entry_i": t, "entry": entry, "stop": stop,
                           "tp": np.nan, "shares": sh, "risk_amt": sh * risk,
                           "eq0": capital, "armed": False,
                           "vol_ratio": 1.0, "risk_dist": stop_pct,
                           "signal_i": t - 1}
        equity[t] = cash + shares * c[t]
        capital = equity[t]

    if pos:
        register_exit(c[-1] * (1 - slip), n - 1, "eod")
        equity[-1] = cash
    tr = pd.DataFrame(trades)
    return tr, equity


def trade_stats(tr: pd.DataFrame) -> dict:
    if not len(tr):
        return dict(trades=0)
    wins = tr[tr["R"] >= 0.1]
    be = tr[(tr["R"] > -0.1) & (tr["R"] < 0.1)]
    n_loss = int((tr["pnl"] < 0).sum())
    gp = tr.loc[tr["pnl"] > 0, "pnl"].sum()
    gl = -tr.loc[tr["pnl"] < 0, "pnl"].sum()
    return dict(trades=len(tr), wins=len(wins), be=len(be), losses=n_loss,
                win_rate=len(wins) / max(1, len(wins) + n_loss),
                pf=gp / gl if gl > 0 else np.inf,
                avg_R=tr["R"].mean(), sum_R=tr["R"].sum(),
                median_R=tr["R"].median())


def fmt_stats(name, s):
    if s.get("trades", 0) == 0:
        return f"{name}: 无交易"
    return (f"{name}: {s['trades']}笔 | 胜{s['wins']} 负{s['losses']} 保本{s['be']} | "
            f"胜率 {s['win_rate']:.1%} | PF {s['pf']:.2f} | 平均R {s['avg_R']:+.3f} | "
            f"中位R {s['median_R']:+.3f} | 总R {s['sum_R']:+.0f}")


def load_pit_snapshots():
    """返回 (sorted dates, {date: set(DuckDB code)})。"""
    snaps = {}
    for fn in ("hist_snapshots_2020.pkl", "hist_snapshots.pkl"):
        p = os.path.join(HERE, "..", "八系统回测", fn)
        if not os.path.exists(p):
            continue
        d = pickle.load(open(p, "rb"))
        for dt, codes in d.items():
            conv = set()
            for x in codes:
                if x.startswith("sh."):
                    conv.add(x[3:] + ".SH")
                elif x.startswith("sz."):
                    conv.add(x[3:] + ".SZ")
                else:
                    conv.add(x)
            snaps[pd.Timestamp(dt)] = conv
    ds = sorted(snaps)
    return ds, snaps


def universe_ok_for(sym, dates, pit_dates, pit_snaps):
    if not pit_dates:
        return None
    out = np.zeros(len(dates), dtype=bool)
    for i, d in enumerate(pd.to_datetime(dates)):
        # 最近一个 <= d 的快照
        idx = np.searchsorted(pit_dates, d.to_datetime64(), side="right") - 1
        if idx >= 0:
            out[i] = sym in pit_snaps[pit_dates[idx]]
    return out


def run_corrected(syms, start=None, end=None, use_pit=False, label="A",
                  entry_mode="next_open", vol_scale=0.8):
    df = base.load_stocks(syms)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)].copy()
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)].copy()
    if not len(df):
        return pd.DataFrame(), {}
    regime3 = base.build_regime3(df)
    pit_dates, pit_snaps = load_pit_snapshots() if use_pit else ([], {})
    all_trades = []
    for sym, g in df.groupby("thscode", sort=False):
        g = g.sort_values("date")
        dates = pd.to_datetime(g["date"]).reset_index(drop=True)
        o = g["open"].to_numpy(); h = g["high"].to_numpy()
        l = g["low"].to_numpy(); c = g["close"].to_numpy()
        a = g["amount"].to_numpy()
        vmed = pd.Series(a).rolling(20).median().shift(1).to_numpy()
        reg_ok = dates.map(regime3).eq("up").to_numpy()
        uok = universe_ok_for(sym, dates, pit_dates, pit_snaps) if use_pit else None
        tr, _ = backtest_symbol_corrected(
            sym, dates, o, h, l, c, a, vmed, reg_ok=reg_ok, universe_ok=uok,
            vol_mult=1.5, trail=True, trend_gate="none", pause_days=0,
            entry_mode=entry_mode, vol_scale=vol_scale,
        )
        if len(tr):
            tr = tr.copy()
            tr["symbol"] = sym
            tr["sys"] = label
            tr["entry_date"] = pd.to_datetime(dates.to_numpy()[tr["entry_i"].to_numpy()])
            all_trades.append(tr)
    if not all_trades:
        return pd.DataFrame(), dict(symbols=df["thscode"].nunique(), bars=len(df))
    T = pd.concat(all_trades, ignore_index=True)
    return T, dict(symbols=df["thscode"].nunique(), bars=len(df))


def run_mr_corrected(syms, start=None, end=None, use_pit=False):
    df = base.load_stocks(syms)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)].copy()
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)].copy()
    if not len(df):
        return pd.DataFrame(), {}
    regime3 = base.build_regime3(df)
    pit_dates, pit_snaps = load_pit_snapshots() if use_pit else ([], {})
    all_trades = []
    for sym, g in df.groupby("thscode", sort=False):
        g = g.sort_values("date")
        dates = pd.to_datetime(g["date"]).reset_index(drop=True)
        o = g["open"].to_numpy(); h = g["high"].to_numpy()
        l = g["low"].to_numpy(); c = g["close"].to_numpy()
        allow = ~dates.map(regime3).eq("up").to_numpy()
        uok = universe_ok_for(sym, dates, pit_dates, pit_snaps) if use_pit else None
        if uok is not None:
            allow = allow & uok
        tr, _ = backtest_mr_corrected(sym, dates, o, h, l, c, allow=allow)
        if len(tr):
            tr = tr.copy()
            tr["symbol"] = sym
            tr["sys"] = "B"
            tr["entry_date"] = pd.to_datetime(dates.to_numpy()[tr["entry_i"].to_numpy()])
            all_trades.append(tr)
    if not all_trades:
        return pd.DataFrame(), dict(symbols=df["thscode"].nunique(), bars=len(df))
    T = pd.concat(all_trades, ignore_index=True)
    return T, dict(symbols=df["thscode"].nunique(), bars=len(df))


# ---------------- 组合层 ----------------
def build_close_map(df):
    m = {}
    for sym, g in df.groupby("thscode", sort=False):
        m[sym] = pd.Series(g["close"].to_numpy(),
                           index=pd.to_datetime(g["date"]).to_numpy())
    return m


def simulate(T, close_map, rank_by="vol", seed=0, init=10000.0,
             risk=0.02, pos_cap=0.30, max_pos=6, comm=COMM):
    """1万账户模拟。rank_by: symbol / vol / random。"""
    if not len(T):
        return None, 0, 0, pd.DataFrame()
    T = T.copy().reset_index(drop=True)
    T["trade_id"] = np.arange(len(T))
    rng = np.random.default_rng(seed)
    if rank_by == "vol":
        T["rank_score"] = T["vol_ratio"].fillna(1.0)
    elif rank_by == "random":
        T["rank_score"] = rng.random(len(T))
    else:
        T["rank_score"] = 0.0
    T = T.sort_values(["entry_date", "rank_score", "symbol"],
                      ascending=[True, False, True]).reset_index(drop=True)

    by_in = {d: g for d, g in T.groupby("entry_date")}
    by_out = {d: g for d, g in T.groupby("exit_date")}
    days = pd.bdate_range(T["entry_date"].min(), T["exit_date"].max())

    cash = init
    positions = []
    eq_curve = {}
    taken = skipped = 0
    log = []

    def price_on(sym, d):
        s = close_map.get(sym)
        if s is None or not len(s):
            return np.nan
        if d in s.index:
            return float(s.loc[d])
        prev = s[s.index <= d]
        return float(prev.iloc[-1]) if len(prev) else np.nan

    for d in days:
        # 卖出
        out_today = by_out.get(d)
        if out_today is not None:
            exits = {int(r.trade_id): r for _, r in out_today.iterrows()}
            still = []
            for p in positions:
                r = exits.get(p["trade_id"])
                if r is None:
                    still.append(p); continue
                cash += p["shares"] * r["exit"] * (1 - comm - stamp_rate(d))
                log.append({"trade_id": p["trade_id"], "symbol": p["symbol"],
                            "entry_date": p["entry_date"], "exit_date": d,
                            "entry": p["entry"], "exit": r["exit"],
                            "shares": p["shares"], "sys": p["sys"],
                            "pnl": p["shares"] * r["exit"] * (1 - comm - stamp_rate(d))
                                   - p["shares"] * p["entry"] * (1 + comm)})
            positions = still

        # 买入
        in_today = by_in.get(d)
        if in_today is not None:
            for _, r in in_today.iterrows():
                if len(positions) >= max_pos or cash <= 0:
                    skipped += 1; continue
                eq_now = cash
                for p in positions:
                    px = price_on(p["symbol"], d)
                    eq_now += p["shares"] * (px if np.isfinite(px) else p["entry"])
                risk_amt = max(r["entry"] - r["stop"], 1e-9)
                sh = min(risk * eq_now / risk_amt,
                         pos_cap * eq_now / r["entry"],
                         cash / (r["entry"] * (1 + comm)))
                if sh * r["entry"] < 500:
                    skipped += 1; continue
                cash -= sh * r["entry"] * (1 + comm)
                positions.append({"trade_id": int(r["trade_id"]), "symbol": r["symbol"],
                                  "shares": sh, "entry": r["entry"],
                                  "entry_date": d, "sys": r["sys"]})
                taken += 1

        mv = 0.0
        for p in positions:
            px = price_on(p["symbol"], d)
            mv += p["shares"] * (px if np.isfinite(px) else p["entry"])
        eq_curve[d] = cash + mv

    eq = pd.Series(eq_curve).sort_index()
    return eq, taken, skipped, pd.DataFrame(log)


def eq_stats(eq, init=10000.0):
    if eq is None or not len(eq):
        return {}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    peak = eq.cummax()
    mdd = (eq / peak - 1).min()
    return dict(final=float(eq.iloc[-1]), total=float(eq.iloc[-1] / init - 1),
                cagr=float((eq.iloc[-1] / init) ** (1 / yrs) - 1) if yrs > 0 else np.nan,
                mdd=float(mdd), years=float(yrs))


def yearly(eq, init=10000.0):
    y = eq.resample("YE").last()
    r = y.pct_change()
    if len(r):
        r.iloc[0] = y.iloc[0] / init - 1
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pit", action="store_true")
    ap.add_argument("--nrandom", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    syms = base.liquid_symbols()
    if args.quick:
        syms = syms[:5]
    print(f"数据: {DB_PATH}")
    print(f"样本: {len(syms)} 只（≥600根K线且日成交额中位数≥1亿）")

    if args.pit:
        pit_dates, pit_snaps = load_pit_snapshots()
        union = set().union(*pit_snaps.values()) if pit_snaps else set()
        syms = [s for s in syms if s in union]
        print(f"PIT 中证500 子样本: {len(syms)} 只，快照 {len(pit_dates)} 期 "
              f"({pit_dates[0].date()} ~ {pit_dates[-1].date()})")
        A, meta = run_corrected(syms, start="2020-01-01", end="2025-12-31",
                                use_pit=True, label="A_pit")
        B, _ = run_mr_corrected(syms, start="2020-01-01", end="2025-12-31", use_pit=True)
        tag = "PIT_2020_2025"
    else:
        A, meta = run_corrected(syms, label="A")
        B, _ = run_mr_corrected(syms)
        tag = "ALL_2016_2026"

    print(f"载入 {meta.get('bars', 0)} 行日线")
    print(fmt_stats(f"修正版 A ({tag})", trade_stats(A)))
    print(fmt_stats(f"修正版 B ({tag})", trade_stats(B)))
    AB = pd.concat([A, B], ignore_index=True) if len(A) or len(B) else pd.DataFrame()
    print(fmt_stats(f"修正版 A+B ({tag})", trade_stats(AB)))

    if len(A):
        A.to_csv(os.path.join(OUT_DIR, f"修正版_交易明细_A_{tag}.csv"), index=False)
    if len(B):
        B.to_csv(os.path.join(OUT_DIR, f"修正版_交易明细_B_{tag}.csv"), index=False)
    if len(AB):
        AB.to_csv(os.path.join(OUT_DIR, f"修正版_交易明细_A+B_{tag}.csv"), index=False)

    if args.quick:
        print("quick 模式结束")
        return

    df = base.load_stocks(syms)
    close_map = build_close_map(df)

    # 组合：A+B，原顺序 / 放量排序 / 随机顺序分布
    rows = []
    for rank_by in ("symbol", "vol"):
        eq, taken, skipped, log = simulate(AB, close_map, rank_by=rank_by)
        s = eq_stats(eq)
        rows.append(dict(policy=rank_by, seed=-1, taken=taken, skipped=skipped, **s))
        print(f"\n组合 A+B | {rank_by} | 成交{taken} 放弃{skipped} | "
              f"期末 {s['final']:.0f} | 总收益 {s['total']:+.1%} | "
              f"年化 {s['cagr']:+.1%} | 最大回撤 {s['mdd']:.1%}")
        if rank_by == "vol":
            eq.to_csv(os.path.join(OUT_DIR, f"修正版_组合净值_1万_{tag}_vol.csv"))
            log.to_csv(os.path.join(OUT_DIR, f"修正版_组合成交_1万_{tag}_vol.csv"), index=False)
            ye = yearly(eq)
            ye.to_csv(os.path.join(OUT_DIR, f"修正版_组合年度_{tag}_vol.csv"))

    rng = np.random.default_rng(20261020)
    for seed in range(args.nrandom):
        eq, taken, skipped, _ = simulate(AB, close_map, rank_by="random",
                                         seed=int(rng.integers(0, 2**31 - 1)))
        s = eq_stats(eq)
        rows.append(dict(policy="random", seed=seed, taken=taken, skipped=skipped, **s))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT_DIR, f"修正版_组合路径分布_{tag}.csv"), index=False)
    rnd = R[R["policy"] == "random"]
    if len(rnd):
        print(f"\n随机成交顺序分布（A+B，{len(rnd)} 条路径）:")
        for col, name in [("cagr", "年化"), ("mdd", "最大回撤")]:
            q = rnd[col].quantile([0.05, 0.5, 0.95])
            print(f"  {name}: 5% {q.iloc[0]:+.1%} | 中位 {q.iloc[1]:+.1%} | 95% {q.iloc[2]:+.1%}")
        q = rnd["final"].quantile([0.05, 0.5, 0.95])
        print(f"  期末: 5% {q.iloc[0]:,.0f}元 | 中位 {q.iloc[1]:,.0f}元 | 95% {q.iloc[2]:,.0f}元")

    # A-only 对照
    eqA, takenA, skippedA, _ = simulate(A, close_map, rank_by="vol")
    sA = eq_stats(eqA)
    print(f"\n组合 A-only | vol | 成交{takenA} 放弃{skippedA} | "
          f"期末 {sA['final']:.0f} | 总收益 {sA['total']:+.1%} | "
          f"年化 {sA['cagr']:+.1%} | 最大回撤 {sA['mdd']:.1%}")

    print(f"\n耗时 {time.time() - t0:.1f}s | 输出目录 {OUT_DIR}")


if __name__ == "__main__":
    main()
