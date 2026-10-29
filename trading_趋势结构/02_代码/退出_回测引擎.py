"""退出规则的配对回测引擎。

撮合与时序（关键，不能错）：
  第 t 日收盘 -> 状态机判定 -> 止损/目标/分批条件单在 t+1 起生效
  第 t+1 日开盘 -> 先处理 t 日收盘发出的"次日开盘离场"，再处理日内条件单
  买入当日（T+1）不可卖出
  仓位规模使用"前一日收盘净值"，不使用当日收盘（那是未来信息）

价格口径：
  指标、止损、目标 = hfq 空间
  撮合 = 原始价空间（涨跌停、整股、费用都在这里）
  记账 = shares × exit_raw × (f_exit / f_entry)，等价于"原始价 + 分红累计"

2021+ 由 持仓状态机.assert_in_window 拦截，引擎自身也把交易日历截到 2020-12-31。

两种运行模式共用同一个 step_position()，保证"组合级"与"单笔级"口径一致：
  run_portfolio : 10 槽资金约束，用于组合绩效
  run_trades    : 每笔独立、不占资金，用于剥离资金约束后的纯退出效应
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from 持仓状态机 import (  # noqa: E402
    Bar,
    ExitPolicy,
    Indicators,
    Position,
    assert_in_window,
    initial_stop_hfq,
    size_equal_weight,
    size_risk_budget,
    size_vol_target,
)
from 退出_数据层 import Panel  # noqa: E402

DEV_START, DEV_END = "2008-01-01", "2015-12-31"
VAL_START, VAL_END = "2016-01-01", "2020-12-31"


def snap(p: Panel, start: str, end: str) -> tuple[int, int]:
    """把自然日区间吸附到面板交易日历（start 向后、end 向前）。"""
    i0 = int(np.searchsorted(p.dates, start, side="left"))
    i1 = int(np.searchsorted(p.dates, end, side="right")) - 1
    if i0 >= p.n_days or i1 < 0 or i0 > i1:
        raise ValueError(f"区间 {start}~{end} 与面板日历无交集")
    return i0, i1


@dataclass
class BTConfig:
    arm: str = "B0"
    sizing: str = "EQ"
    risk_pct: float = 0.01
    vol_pct: float = 0.01
    slots: int = 10
    capital: float = 1_000_000.0
    risk_cap: float | None = None
    max_new_per_day: int = 2
    slippage: float = 0.0005
    commission: float = 0.00025
    min_commission: float = 5.0
    stamp: float = 0.001
    transfer: float = 0.00001
    testbed: str = "TB1"
    start: str = DEV_START
    end: str = DEV_END
    selection: str = "random"   # random(中性) | amount(流动性优先)
    seed: int = 20261026


# --------------------------------------------------------------------------
# 信号
# --------------------------------------------------------------------------


def build_signals(p: Panel, testbed: str = "TB1", cache_dir: str | None = None) -> np.ndarray:
    """t 日收盘产生的入场候选，bool [n_codes, n_days]。"""
    key = os.path.join(cache_dir, "", f"sig_{testbed}.npy") if cache_dir else None
    if key and os.path.exists(key):
        return np.load(key)

    c, amt, atr = p["c"], p["amt"], p["atr"]
    nc, nd = c.shape
    hfq_c = c * p["f"]

    ma120 = np.full((nc, nd), np.nan, dtype=np.float32)
    amt20 = np.full((nc, nd), np.nan, dtype=np.float32)
    run60max = np.full((nc, nd), np.nan, dtype=np.float32)
    for i in range(nc):
        s = pd.Series(hfq_c[i])
        ma120[i] = s.rolling(120, min_periods=120).mean().to_numpy()
        amt20[i] = pd.Series(amt[i]).rolling(20, min_periods=15).mean().to_numpy()
        run60max[i] = s.rolling(60, min_periods=60).max().to_numpy()

    eligible = (
        p.tradable
        & (~p.is_st)
        & (p.listed >= 250)
        & (c > 2.0)
        & (amt20 > 5e7)
        & np.isfinite(atr)
        & (atr / hfq_c < 0.12)
    )
    if testbed == "TB1":
        sig = eligible & (hfq_c >= run60max - 1e-9) & (hfq_c > ma120)
    elif testbed == "TB2":
        raw = np.zeros((nc, nd), dtype=bool)
        for i in range(nc):
            pos = np.where(p.tradable[i])[0]
            if len(pos) < 300:
                continue
            raw[i, pos[260::20]] = True
        sig = raw & eligible
    else:
        raise ValueError(testbed)

    if key:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(key, sig)
    return sig


# --------------------------------------------------------------------------
# 单日推进（两种模式共用）
# --------------------------------------------------------------------------


def _fees(cfg: BTConfig, amount: float, sell: bool) -> float:
    fee = max(amount * cfg.commission, cfg.min_commission) + amount * cfg.transfer
    if sell:
        fee += amount * cfg.stamp
    return fee


@dataclass
class Fill:
    frac: float
    px_raw: float
    shares: float
    gross: float
    reason: str


def step_position(p: Panel, ci: int, pos: Position, entry_f: float, di: int,
                  cfg: BTConfig, can_exit: bool, bar=None) -> tuple[list[Fill], bool]:
    """推进一个持仓一天。返回 (fills, 是否已清仓)。bar 可外部预取以复用。

    can_exit=False 表示当天是买入当日：A股 T+1，任何离场（含止损）都不能成交。
    """
    b = bar if bar is not None else p.bar(ci, di)
    if b is None or not b["tradable"]:
        pos.susp_bars += 1
        return [], False

    pos.susp_bars = 0
    pos.mark_extremes(b["hh"], b["lh"])
    if not can_exit:
        return [], False
    f = b["f"]
    locked_dn = b["h"] <= b["l"] + 1e-12 and b["l"] <= b["dn"] + 0.011
    fills: list[Fill] = []

    def sell(px: float, sh: float, frac: float, reason: str):
        sh = float(min(sh, pos.shares))
        if sh <= 1e-9:
            return
        fills.append(Fill(frac, px, sh, px * (f / entry_f) * sh, reason))
        pos.shares -= sh

    # 1) 前一日收盘发出的开盘离场
    if pos.pending_exit_frac > 0 and not locked_dn:
        frac = pos.pending_exit_frac
        pos.pending_exit_frac = 0.0
        sell(b["o"] * (1 - cfg.slippage), pos.shares * frac, frac,
             pos.pending_reason or "收盘离场")
        if pos.shares <= 1e-9:
            return fills, True

    # 2) 止损（保守：与目标同日触发时止损优先）
    if not locked_dn and np.isfinite(pos.stop):
        sr = max(pos.stop / f, b["dn"])
        px = None
        if b["o"] <= sr:
            px = b["o"] * (1 - cfg.slippage)
        elif b["l"] <= sr:
            px = sr * (1 - cfg.slippage)
        if px is not None:
            sell(px, pos.shares, 1.0, pos.stop_kind)
            return fills, pos.shares <= 1e-9

    # 3) 目标 / 分批（限价单）
    _, scale_px, target_px = pos.intraday_orders()
    for px_h, kind, frac in (
        (target_px, "固定目标止盈", 1.0),
        (scale_px, "分批止盈", pos.policy.scale_frac),
    ):
        if not np.isfinite(px_h) or pos.shares <= 1e-9:
            continue
        pr = min(px_h / f, b["up"])
        if b["o"] >= pr:
            px = b["o"] * (1 - cfg.slippage)
        elif b["h"] >= pr:
            px = pr * (1 - cfg.slippage)
        else:
            continue
        sell(px, pos.shares * frac, frac, kind)
        if kind == "固定目标止盈":
            pos.target_active = False
        else:
            pos.scale_done = True
        if pos.shares <= 1e-9:
            return fills, True

    # 4) 收盘评估（T+1 已在函数入口拦截：入场当日不会走到这里）
    ind = Indicators(
        atr20=b["atr"], ma20=b["ma20"], ma60=b["ma60"],
        ma60_slope=_slope(p["ma60"], ci, di), vol_ma20=b["vma20"],
        low20=_min_back(p["lh"], ci, max(di - 1, 0), 20), close_max3=_max_back(p["ch"], ci, di, 3),
        pivot_low=(None if not np.isfinite(b["piv"]) else b["piv"]),
    )
    bar = _bar(b)
    pos.update_state(bar, ind)
    pos.update_stop(bar, ind)
    pos.decide(bar, ind)
    return fills, False


# --------------------------------------------------------------------------
# 组合回测
# --------------------------------------------------------------------------


@dataclass
class _Open:
    pos: Position
    ci: int
    entry_f: float
    entry_raw: float
    entry_di: int
    init_shares: int
    init_cost: float
    init_risk: float
    init_atr_pct: float
    proceeds: float = 0.0
    was_partial: bool = False
    last_di: int = 0
    last_px: float = np.nan
    last_reason: str = "持仓中"


def _mark_to_market(p: Panel, ci, shares, entry_f, di, last_close, entry_raw):
    b = p.bar(ci, di)
    if b is not None:
        return b["c"] * shares * (b["f"] / entry_f)
    return last_close.get(ci, entry_raw) * shares


def run_portfolio(p: Panel, sig: np.ndarray, cfg: BTConfig, policy: ExitPolicy):
    i0, i1 = snap(p, cfg.start, cfg.end)
    assert_in_window(p.dates[i0])
    assert_in_window(p.dates[i1])
    g = p.m

    cash = cfg.capital
    opens: list[_Open] = []
    trades: list[dict] = []
    nav_hist = np.full(i1 - i0 + 1, np.nan)
    np_hist = np.zeros(i1 - i0 + 1, dtype=int)
    last_close: dict[int, float] = {}
    nav_ref = cfg.capital          # 上一交易日收盘净值，用于 sizing

    def record(o: _Open):
        trades.append(dict(
            code=p.code_str[o.ci], entry_date=p.date_str[o.entry_di],
            exit_date=p.date_str[o.last_di], entry_px=o.entry_raw, exit_px=o.last_px,
            shares=o.init_shares, cost=o.init_cost, proceeds=o.proceeds,
            init_risk=o.init_risk, ret=o.proceeds / o.init_cost - 1.0,
            R_mult=(o.proceeds - o.init_cost) / o.init_risk if o.init_risk > 0 else np.nan,
            hold_days=o.last_di - o.entry_di, reason=o.last_reason,
            state_at_exit=o.pos.state.value, mfe_R=o.pos.mfe_R, mae_R=o.pos.mae_R,
            board="创业板" if p.code_str[o.ci][3:6] in ("300", "301") else "主板",
            atr_pct=o.init_atr_pct, partial=o.was_partial,
            window=("dev" if p.dates[o.last_di] < VAL_START else "val"),
        ))

    for step, di in enumerate(range(i0, i1 + 1)):
        date = p.date_str[di]
        assert_in_window(date)

        still: list[_Open] = []
        for o in opens:
            b = p.bar(o.ci, di)
            if b is None or not b["tradable"]:
                o.pos.susp_bars += 1
                if o.pos.susp_bars >= policy.max_suspension_bars:
                    px = last_close.get(o.ci, o.entry_raw) * (1 - cfg.slippage)
                    gross = px * (o.entry_f / o.entry_f) * o.pos.shares
                    net = gross - _fees(cfg, gross, True)
                    cash += net
                    o.proceeds += net
                    o.pos.shares = 0.0
                    o.last_di, o.last_px, o.last_reason = di, px, "停牌超期"
                    record(o)
                    continue
                still.append(o)
                continue

            last_close[o.ci] = b["c"]
            fills, closed = step_position(p, o.ci, o.pos, o.entry_f, di, cfg, di > o.entry_di)
            for fl in fills:
                net = fl.gross - _fees(cfg, fl.gross, True)
                cash += net
                o.proceeds += net
                if fl.frac < 1.0:
                    o.was_partial = True
            if fills:
                o.last_di, o.last_px, o.last_reason = di, fills[-1].px_raw, fills[-1].reason
            if closed:
                record(o)
            else:
                still.append(o)
        opens = still

        # 新入场：t-1 收盘信号 -> t 日开盘；规模用 nav_ref（前收净值）
        si = di - 1
        if si >= 0 and di < i1:
            cand = np.where(sig[:, si])[0]
            if len(cand):
                if cfg.selection == "amount":
                    order = np.argsort(-np.nan_to_num(g["amt"][cand, si], nan=0.0), kind="stable")
                else:
                    rng = np.random.default_rng(cfg.seed + si)
                    order = rng.permutation(len(cand))
                n_new = 0
                held = {o.ci for o in opens}
                for ci in cand[order]:
                    if len(opens) >= cfg.slots or n_new >= cfg.max_new_per_day:
                        break
                    if ci in held:
                        continue
                    b = p.bar(ci, di)
                    if b is None or not b["tradable"]:
                        continue
                    if b["h"] <= b["l"] + 1e-12 and b["l"] >= b["up"] - 0.011:
                        continue
                    if not np.isfinite(b["atr"]) or b["atr"] <= 0:
                        continue
                    fill = b["o"] * (1 + cfg.slippage)
                    entry_hfq = fill * b["f"]
                    stop_h = initial_stop_hfq(
                        policy, entry_hfq, b["atr"],
                        None if not np.isfinite(b["piv"]) else b["piv"])
                    per_risk = (entry_hfq - stop_h) / b["f"]
                    if per_risk <= 0:
                        continue
                    max_notional = nav_ref / cfg.slots
                    if cfg.sizing == "EQ":
                        sh = size_equal_weight(nav_ref, cfg.slots, fill)
                    elif cfg.sizing == "RB":
                        sh = size_risk_budget(nav_ref, fill, stop_h / b["f"], cfg.risk_pct, max_notional)
                    elif cfg.sizing == "VOL":
                        sh = size_vol_target(nav_ref, fill, b["atr"] / b["f"], cfg.vol_pct, max_notional)
                    else:
                        raise ValueError(cfg.sizing)
                    if sh <= 0:
                        continue
                    notional = sh * fill
                    fee = _fees(cfg, notional, False)
                    if notional + fee > cash:
                        sh = int(cash / (fill * 1.002) // 100) * 100
                        if sh <= 0:
                            continue
                        notional = sh * fill
                        fee = _fees(cfg, notional, False)
                        if notional + fee > cash:
                            continue
                    if cfg.risk_cap is not None:
                        cur = 0.0
                        for o in opens:
                            bb = p.bar(o.ci, di)
                            ff = bb["f"] if bb else o.entry_f
                            px = bb["c"] if bb else last_close.get(o.ci, o.entry_raw)
                            cur += max(px - o.pos.stop / ff, 0.0) * o.pos.shares
                        if (cur + per_risk * sh) / nav_ref > cfg.risk_cap:
                            continue
                    cash -= notional + fee
                    pos = Position(
                        code=p.code_str[ci], entry_date=date, entry_px=entry_hfq,
                        shares=sh, init_stop=stop_h, R=entry_hfq - stop_h, policy=policy,
                    )
                    o = _Open(pos, ci, b["f"], fill, di, sh, notional + fee,
                              per_risk * sh, b["atr"] / b["f"] / fill)
                    o.last_di, o.last_px = di, fill
                    opens.append(o)
                    held.add(ci)
                    n_new += 1

        nav = cash + sum(
            _mark_to_market(p, o.ci, o.pos.shares, o.entry_f, di, last_close, o.entry_raw)
            for o in opens
        )
        nav_hist[step] = nav
        np_hist[step] = len(opens)
        nav_ref = nav

    # 期末强平
    for o in opens:
        px = last_close.get(o.ci, o.entry_raw) * (1 - cfg.slippage)
        bb = p.bar(o.ci, i1)
        ff = bb["f"] if bb else o.entry_f
        gross = px * (ff / o.entry_f) * o.pos.shares
        net = gross - _fees(cfg, gross, True)
        cash += net
        o.proceeds += net
        o.last_di, o.last_px, o.last_reason = i1, px, "数据末端"
        record(o)
    nav_hist[-1] = cash

    eq = pd.Series(nav_hist, index=pd.Index(p.dates[i0 : i1 + 1], name="date")).ffill()
    return eq, pd.DataFrame(trades), np_hist


# --------------------------------------------------------------------------
# 单笔级（无资金约束）
# --------------------------------------------------------------------------


def run_trades(p: Panel, sig: np.ndarray, cfg: BTConfig, policy: ExitPolicy) -> pd.DataFrame:
    i0, i1 = snap(p, cfg.start, cfg.end)
    assert_in_window(p.dates[i0])
    assert_in_window(p.dates[i1])
    s0 = max(i0 - 1, 0)
    rows: list[dict] = []
    for ci in range(p.n_codes):
        for si in np.where(sig[ci, s0 : i1 + 1])[0] + s0:
            ed = si + 1
            if ed > i1 or ed < i0 or ed >= i1:
                continue
            b = p.bar(ci, ed)
            if b is None or not b["tradable"]:
                continue
            if b["h"] <= b["l"] + 1e-12 and b["l"] >= b["up"] - 0.011:
                continue
            if not np.isfinite(b["atr"]) or b["atr"] <= 0:
                continue
            fill = b["o"] * (1 + cfg.slippage)
            entry_hfq = fill * b["f"]
            stop_h = initial_stop_hfq(
                policy, entry_hfq, b["atr"],
                None if not np.isfinite(b["piv"]) else b["piv"])
            per_risk = (entry_hfq - stop_h) / b["f"]
            if per_risk <= 0:
                continue
            pos = Position(p.code_str[ci], p.date_str[ed], entry_hfq, 1.0,
                           stop_h, entry_hfq - stop_h, policy)
            entry_f = b["f"]
            realised = 0.0
            last_raw = fill
            last_di, last_px, last_reason = ed, fill, "数据末端"
            closed = False
            for di in range(ed, i1 + 1):
                bb = p.bar(ci, di)
                if bb is None or not bb["tradable"]:
                    pos.susp_bars += 1
                    if pos.susp_bars >= policy.max_suspension_bars:
                        px = last_raw * (1 - cfg.slippage)
                        realised += px * (p["f"][ci, di] / entry_f) * pos.shares
                        last_di, last_px, last_reason, closed = di, px, "停牌超期", True
                        break
                    continue
                last_raw = bb["c"]
                fills, done = step_position(p, ci, pos, entry_f, di, cfg, di > ed)
                for fl in fills:
                    realised += fl.gross
                    last_di, last_px, last_reason = di, fl.px_raw, fl.reason
                if done:
                    closed = True
                    break
            if not closed:
                px = last_raw * (1 - cfg.slippage)
                realised += px * (p["f"][ci, i1] / entry_f) * pos.shares
                last_di, last_px, last_reason = i1, px, "数据末端"
            rows.append(dict(
                code=p.code_str[ci], entry_date=p.date_str[ed], exit_date=p.date_str[last_di],
                ret=realised / fill - 1.0,
                R_mult=(realised - fill) / per_risk,
                hold_days=last_di - ed, reason=last_reason, state_at_exit=pos.state.value,
                mfe_R=pos.mfe_R, mae_R=pos.mae_R,
                board="创业板" if p.code_str[ci][3:6] in ("300", "301") else "主板",
                atr_pct=per_risk / fill,
                window=("dev" if p.dates[ed] < VAL_START else "val"),
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------


def _bar(b):
    """状态机只吃 hfq 空间的价格；原始价只用于撮合。"""
    return Bar(date=b["date"], open=b["oh"], high=b["hh"], low=b["lh"], close=b["ch"],
               preclose=b["pch"], volume=b["vol"], amount=b["amt"],
               tradable=b["tradable"], limit_up=b["up"] * b["f"], limit_down=b["dn"] * b["f"])


def _slope(mat, ci, di, lag=5):
    a, b = mat[ci, di], mat[ci, max(di - lag, 0)]
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        return 0.0
    return a / b - 1.0


def _min_back(mat, ci, di, n):
    seg = mat[ci, max(di - n + 1, 0) : di + 1]
    seg = seg[np.isfinite(seg)]
    return float(seg.min()) if len(seg) else np.nan


def _max_back(mat, ci, di, n):
    seg = mat[ci, max(di - n + 1, 0) : di + 1]
    seg = seg[np.isfinite(seg)]
    return float(seg.max()) if len(seg) else np.nan


# --------------------------------------------------------------------------
# 绩效
# --------------------------------------------------------------------------


def perf(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 3:
        return {}
    r = eq.pct_change().dropna()
    yrs = len(eq) / 244.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[0] > 0 else np.nan
    dd = float((eq / eq.cummax() - 1).min())
    return dict(
        total_ret=float(eq.iloc[-1] / eq.iloc[0] - 1),
        cagr=float(cagr),
        mdd=dd,
        calmar=float(cagr / abs(dd)) if dd < 0 else np.nan,
        sharpe=float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan,
        vol=float(r.std() * np.sqrt(244)),
        nav_end=float(eq.iloc[-1]),
    )


def trade_stats(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {}
    r = df["R_mult"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return {}
    return dict(
        n=int(len(df)),
        win=float((df["ret"] > 0).mean()),
        avg_ret=float(df["ret"].mean()),
        med_ret=float(df["ret"].median()),
        avg_R=float(r.mean()),
        med_R=float(r.median()),
        p05_R=float(r.quantile(0.05)),
        p95_R=float(r.quantile(0.95)),
        avg_hold=float(df["hold_days"].mean()),
    )


# --------------------------------------------------------------------------
# 多臂并行：同一入场、同一 bar 序列，一次遍历产出全部规则臂
# --------------------------------------------------------------------------


def run_trades_multi(p: Panel, sig: np.ndarray, cfg: BTConfig,
                     policies: dict[str, ExitPolicy],
                     state_log_arm: str | None = None,
                     state_log: list | None = None) -> dict[str, pd.DataFrame]:
    """同一批入场事件上并行推进多条退出规则。

    关键点：入场集合与每个 bar 的取数对所有臂完全相同，因此臂间差异
    只能来自退出规则本身——这是"配对设计"的代码级保证。
    """
    i0, i1 = snap(p, cfg.start, cfg.end)
    assert_in_window(p.dates[i0])
    assert_in_window(p.dates[i1])
    s0 = max(i0 - 1, 0)
    out: dict[str, list[dict]] = {k: [] for k in policies}

    for ci in range(p.n_codes):
        sig_days = np.where(sig[ci, s0 : i1 + 1])[0] + s0
        if len(sig_days) == 0:
            continue
        for si in sig_days:
            ed = si + 1
            if ed > i1 or ed < i0 or ed >= i1:
                continue          # 末端 bar 买入无法持有，直接剔除
            b = p.bar(ci, ed)
            if b is None or not b["tradable"]:
                continue
            if b["h"] <= b["l"] + 1e-12 and b["l"] >= b["up"] - 0.011:
                continue
            if not np.isfinite(b["atr"]) or b["atr"] <= 0:
                continue
            fill = b["o"] * (1 + cfg.slippage)
            entry_hfq = fill * b["f"]
            piv = None if not np.isfinite(b["piv"]) else b["piv"]

            live = {}
            for name, pol in policies.items():
                stop_h = initial_stop_hfq(pol, entry_hfq, b["atr"], piv)
                pr = (entry_hfq - stop_h) / b["f"]
                if pr <= 0:
                    continue
                pos = Position(p.code_str[ci], p.date_str[ed], entry_hfq, 1.0,
                               stop_h, entry_hfq - stop_h, pol)
                live[name] = dict(pos=pos, per_risk=pr, realised=0.0,
                                  last_di=ed, last_px=fill, last_reason="数据末端")
            if not live:
                continue

            last_raw = fill
            entry_f = b["f"]
            for di in range(ed, i1 + 1):
                bb = p.bar(ci, di) if di > ed else b
                if bb is None or not bb["tradable"]:
                    for name in list(live):
                        st = live[name]
                        st["pos"].susp_bars += 1
                        if st["pos"].susp_bars >= policies[name].max_suspension_bars:
                            px = last_raw * (1 - cfg.slippage)
                            st["realised"] += px * (p["f"][ci, di] / entry_f) * st["pos"].shares
                            st.update(last_di=di, last_px=px, last_reason="停牌超期")
                            _emit(out[name], p, ci, ed, st, entry_f)
                            del live[name]
                    if not live:
                        break
                    continue
                last_raw = bb["c"]
                for name in list(live):
                    st = live[name]
                    fills, done = step_position(p, ci, st["pos"], entry_f, di, cfg,
                                                di > ed, bar=bb)
                    if (state_log is not None and name == state_log_arm
                            and di > ed and not done and di + 5 <= i1):
                        o5 = p["oh"][ci, di + 1]
                        c5 = p["ch"][ci, di + 5]
                        if np.isfinite(o5) and np.isfinite(c5) and o5 > 0:
                            state_log.append((st["pos"].state.value, c5 / o5 - 1.0,
                                              p.dates[ed] < VAL_START, p.board_id[ci]))
                    for fl in fills:
                        st["realised"] += fl.gross
                        st["last_di"], st["last_px"], st["last_reason"] = di, fl.px_raw, fl.reason
                    if done:
                        _emit(out[name], p, ci, ed, st, entry_f)
                        del live[name]
                if not live:
                    break

            for name, st in live.items():
                px = last_raw * (1 - cfg.slippage)
                st["realised"] += px * (p["f"][ci, i1] / entry_f) * st["pos"].shares
                st.update(last_di=i1, last_px=px, last_reason="数据末端")
                _emit(out[name], p, ci, ed, st, entry_f)

    return {k: pd.DataFrame(v) for k, v in out.items()}


def _emit(bucket: list, p: Panel, ci: int, ed: int, st: dict, entry_f: float):
    pos = st["pos"]
    fill = pos.entry_px / entry_f
    bucket.append(dict(
        code=p.code_str[ci], entry_date=p.date_str[ed],
        exit_date=p.date_str[st["last_di"]],
        entry_px=fill, exit_px=st["last_px"],
        ret=st["realised"] / fill - 1.0,
        R_mult=(st["realised"] - fill) / st["per_risk"],
        hold_days=st["last_di"] - ed,
        reason=st["last_reason"], state_at_exit=pos.state.value,
        mfe_R=pos.mfe_R, mae_R=pos.mae_R,
        board="创业板" if p.board_id[ci] else "主板",
        atr_pct=st["per_risk"] / fill,
        window=("dev" if p.dates[ed] < VAL_START else "val"),
    ))
