#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交易容量与执行压力（趋势结构项目第二阶段，独立研究员）。

用途
----
在 2008-2020 开发/验证窗口内，对两条公开 55 日突破基准与两条第一阶段候选，
重算资金规模、持仓数、最低佣金、历史税费、滑点、ADV20 参与率、延迟、涨停顺延、
跌停持久挂单与随机排序种子对净年化的影响。

边界
----
* 只读 `01_数据/frozen/frozen.duckdb`；不写源库、不改他人文件。
* 所有数据在 `2020-12-31` 硬截断；2021+ 不读取、不计算。
* 成交/整股/涨跌停/费用用原始价；持仓收益用冻结层 `factor` 的总收益口径。
* 本文件的独立模拟器只用于压力矩阵；最终账户结论另用
  `02_代码/账户撮合.py` 复核（`reconcile` 子命令）。

子命令
------
    python3 02_代码/候选_执行压力.py build
    python3 02_代码/候选_执行压力.py smoke
    python3 02_代码/候选_执行压力.py run
    python3 02_代码/候选_执行压力.py append
    python3 02_代码/候选_执行压力.py reconcile
    python3 02_代码/候选_执行压力.py report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "03_结果" / "执行压力"
CACHE = OUT / "_cache"
DB = ROOT / "01_数据" / "frozen" / "frozen.duckdb"

sys.path.insert(0, str(HERE))
import 费用与交易约束 as fee  # noqa: E402

FREEZE = date(2026, 10, 26)
DATA_START = "2006-01-01"
DATA_END = "2020-12-31"
DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")
WINDOWS = {"dev": DEV, "val": VAL}

TRADING_DAYS = 252
EPS = 1e-9

# 预注册的基线
BASE_SLIPPAGE = 10.0
BASE_CAPITAL = 1_000_000.0
BASE_SLOTS = 10
BASE_PARTICIPATION = 0.02
BASE_MIN_COMMISSION = 5.0
BASE_SEEDS = (11, 23, 37, 51, 73)
KEY_SEEDS = (11, 23, 37, 51, 73, 89, 101, 113, 127, 139,
             151, 163, 179, 191, 211, 223, 239, 251, 269, 281)


# ---------------------------------------------------------------------------
# 费用配置
# ---------------------------------------------------------------------------

def make_fee_cfg(slippage_bps: float, min_commission: float = 5.0,
                 tax_mode: str = "historical", transfer_mode: str = "module") -> fee.FeeConfig:
    stamp = fee.DEFAULT_STAMP_SCHEDULE
    transfer = fee.DEFAULT_TRANSFER_SCHEDULE
    if tax_mode == "zero":
        stamp = (fee.StampRule(date(1990, 1, 1), 0.0, "both"),)
    if transfer_mode == "par_0.1pct":
        transfer = (
            fee.TransferRule(date(1990, 1, 1), "par_sh", 0.0010, "SH"),
            fee.TransferRule(date(2015, 8, 1), "amount", 0.00002, "BOTH"),
            fee.TransferRule(date(2022, 4, 29), "amount", 0.00001, "BOTH"),
        )
    return fee.FeeConfig(
        commission_rate=0.00025,
        min_commission=float(min_commission),
        stamp_schedule=stamp,
        transfer_schedule=transfer,
        slippage_bps=float(slippage_bps),
        max_participation=1.0,
        dividend_tax_rate=0.0,
    )


# ---------------------------------------------------------------------------
# 面板
# ---------------------------------------------------------------------------

FIELDS = [
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "factor", "td_since_list", "is_st", "is_suspended",
    "limit_up_open", "limit_down_open",
]


@dataclass
class Panel:
    symbols: list[str]
    dates: np.ndarray              # datetime64[D]
    arrays: dict[str, np.ndarray]  # symbol × day
    first_day: np.ndarray
    last_day: np.ndarray
    delist_day: np.ndarray
    n_days: int
    n_sym: int
    sym_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.sym_index:
            self.sym_index = {s: i for i, s in enumerate(self.symbols)}

    def day_index(self, s: str) -> int:
        return int(np.searchsorted(self.dates, np.datetime64(s), side="left"))

    def date_at(self, i: int) -> date:
        return pd.Timestamp(self.dates[i]).date()

    def adv20(self) -> np.ndarray:
        return rolling_mean(self.arrays["amount"], 20)

    def hfq_close(self) -> np.ndarray:
        # 冻结层的 `factor` 是累计总收益指数（Π(1+pct_chg)），本身就是总收益价格序列。
        # 不能再乘原始收盘价：`close * factor` 会把价格涨跌计算两次。
        return self.arrays["factor"].astype(np.float32)


def rolling_sum_last(a: np.ndarray, w: int) -> np.ndarray:
    """按行计算“包含当日”的 w 日和；不足 w 个有效值为 NaN。"""
    fin = np.isfinite(a)
    x = np.where(fin, a, 0.0).astype(np.float64)
    cs = np.cumsum(x, axis=1)
    cnt = np.cumsum(fin, axis=1)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    out[:, w - 1:] = cs[:, w - 1:]
    out[:, w:] = cs[:, w:] - cs[:, :-w]
    cnt_out = np.zeros(a.shape, dtype=np.int32)
    cnt_out[:, w - 1:] = cnt[:, w - 1:]
    cnt_out[:, w:] = cnt[:, w:] - cnt[:, :-w]
    out[cnt_out < w] = np.nan
    return out.astype(np.float32)


def rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    s = rolling_sum_last(a, w)
    return (s / float(w)).astype(np.float32)


def rolling_max_prior(a: np.ndarray, w: int) -> np.ndarray:
    """按行计算“不含当日”的前 w 日最大值；要求 w 个值全部有效。"""
    n_sym, n_day = a.shape
    out = np.full((n_sym, n_day), np.nan, dtype=np.float32)
    if n_day <= w:
        return out
    for i in range(0, n_sym, 200):
        chunk = a[i:i + 200]
        sw = np.lib.stride_tricks.sliding_window_view(chunk, w, axis=1)
        # sw[:, j, :] = a[:, j:j+w]；前 w 日最大值对应 j = d-w
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mx = np.max(sw[:, :-1, :], axis=2)
        out[i:i + chunk.shape[0], w:] = mx[:, :n_day - w]
    return out


def build_panel(force: bool = False) -> Panel:
    CACHE.mkdir(parents=True, exist_ok=True)
    npz = CACHE / "panel_2006_2020.npz"
    meta = CACHE / "panel_meta.json"
    if npz.exists() and meta.exists() and not force:
        z = np.load(npz, allow_pickle=False)
        symbols = [str(x) for x in z["symbols"]]
        dates = z["dates"]
        arrays = {k: z[k] for k in FIELDS}
        p = Panel(symbols=symbols, dates=dates, arrays=arrays,
                  first_day=z["first_day"], last_day=z["last_day"],
                  delist_day=z["delist_day"], n_days=int(z["n_days"]), n_sym=int(z["n_sym"]))
        return p

    t0 = time.time()
    con = duckdb.connect(str(DB), read_only=True)
    q = f"""
    SELECT symbol, date, open, high, low, close, preclose, volume, amount,
           factor, td_since_list, is_st, is_suspended,
           is_limit_up_open AS limit_up_open, is_limit_down_open AS limit_down_open,
           delist_date
    FROM daily
    WHERE board = '主板' AND date BETWEEN DATE '{DATA_START}' AND DATE '{DATA_END}'
    ORDER BY symbol, date
    """
    df = con.execute(q).fetchdf()
    cal = con.execute(f"""
        SELECT date FROM calendar
        WHERE is_trading_day = 1 AND date BETWEEN DATE '{DATA_START}' AND DATE '{DATA_END}'
        ORDER BY date
    """).fetchdf()
    con.close()

    dates = pd.to_datetime(cal["date"]).to_numpy(dtype="datetime64[D]")
    day_map = {d: i for i, d in enumerate(dates)}
    symbols = sorted(df["symbol"].unique().tolist())
    sym_map = {s: i for i, s in enumerate(symbols)}
    n_sym, n_day = len(symbols), len(dates)
    print(f"  面板: {n_sym} 只 × {n_day} 日 = {n_sym*n_day:,} 格; 行 {len(df):,}; {time.time()-t0:.1f}s")

    sid = df["symbol"].map(sym_map).to_numpy(dtype=np.int32)
    did = pd.to_datetime(df["date"]).map(day_map).to_numpy(dtype=np.int32)

    arrays: dict[str, np.ndarray] = {}
    for f in FIELDS:
        arr = np.full((n_sym, n_day), np.nan, dtype=np.float32)
        if f in ("is_st", "is_suspended", "limit_up_open", "limit_down_open"):
            v = df[f].fillna(False).astype(bool).to_numpy()
            tmp = np.zeros((n_sym, n_day), dtype=bool)
            tmp[sid, did] = v
            arrays[f] = tmp
        else:
            v = pd.to_numeric(df[f], errors="coerce").astype(np.float32).to_numpy()
            arr[sid, did] = v
            arrays[f] = arr

    first_day = np.full(n_sym, n_day, dtype=np.int32)
    last_day = np.full(n_sym, -1, dtype=np.int32)
    np.minimum.at(first_day, sid, did)
    np.maximum.at(last_day, sid, did)

    delist_day = np.full(n_sym, n_day, dtype=np.int32)
    dd = df[["symbol", "delist_date"]].dropna().drop_duplicates("symbol")
    for _, r in dd.iterrows():
        s = r["symbol"]
        if s in sym_map and pd.notna(r["delist_date"]):
            d = np.datetime64(pd.Timestamp(r["delist_date"]).date(), "D")
            delist_day[sym_map[s]] = int(np.searchsorted(dates, d, side="left"))

    p = Panel(symbols=symbols, dates=dates, arrays=arrays,
              first_day=first_day, last_day=last_day, delist_day=delist_day,
              n_days=n_day, n_sym=n_sym)
    np.savez(npz, symbols=np.array(symbols, dtype="U12"), dates=dates,
             first_day=first_day, last_day=last_day, delist_day=delist_day,
             n_days=n_day, n_sym=n_sym, **arrays)
    meta.write_text(json.dumps({
        "n_sym": n_sym, "n_days": n_day, "n_rows": len(df),
        "start": DATA_START, "end": DATA_END, "built": time.strftime("%F %T"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  缓存: {npz} ({npz.stat().st_size/1e6:.0f} MB), {time.time()-t0:.1f}s")
    return p


def compute_signals(p: Panel) -> np.ndarray:
    hfq = p.hfq_close()
    hh55 = rolling_max_prior(hfq, 55)
    ma120 = rolling_mean(hfq, 120)
    adv20 = p.adv20()
    close = p.arrays["close"]
    td = p.arrays["td_since_list"]
    st = p.arrays["is_st"]
    susp = p.arrays["is_suspended"]
    ok = (
        np.isfinite(hfq) & np.isfinite(hh55) & np.isfinite(ma120) & np.isfinite(adv20)
        & (hfq > hh55) & (hfq > ma120)
        & (close >= 2.0) & (td >= 250) & (~st) & (~susp)
        & (adv20 >= 3.0e7)
    )
    lo = p.day_index(WINDOWS["dev"][0])
    hi = p.day_index(WINDOWS["val"][1])
    mask = np.zeros_like(ok, dtype=bool)
    mask[:, lo:hi + 1] = ok[:, lo:hi + 1]
    return mask


# ---------------------------------------------------------------------------
# 模拟器
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimCfg:
    candidate: str
    hold_days: int
    window: str
    capital: float = BASE_CAPITAL
    slots: int = BASE_SLOTS
    slippage_bps: float = BASE_SLIPPAGE
    seed: int = 11
    participation: float = BASE_PARTICIPATION
    min_commission: float = BASE_MIN_COMMISSION
    tax_mode: str = "historical"
    transfer_mode: str = "module"
    impact: bool = False
    delay: int = 0
    limit_up: str = "cancel"      # cancel / defer
    limit_down: str = "persist"   # persist / optimistic
    max_sell_defer: int = 10
    ordering: str = "random"      # random / liquidity
    collect_trades: bool = False
    start: str | None = None
    end: str | None = None

    @property
    def key(self) -> tuple:
        return (self.candidate, self.window, self.capital, self.slots, self.slippage_bps,
                self.seed, self.participation, self.min_commission, self.tax_mode,
                self.transfer_mode, self.impact, self.delay, self.limit_up,
                self.limit_down, self.max_sell_defer, self.ordering, self.start, self.end)


class _Pos:
    __slots__ = ("sym", "shares", "initial", "entry_day", "entry_px", "entry_close",
                 "factor_entry", "exit_due", "defer", "entry_fee", "entry_defer")

    def __init__(self, sym, shares, entry_day, entry_px, entry_close, factor_entry,
                 exit_due, entry_fee, entry_defer):
        self.sym = sym
        self.shares = int(shares)
        self.initial = int(shares)
        self.entry_day = int(entry_day)
        self.entry_px = float(entry_px)
        self.entry_close = float(entry_close)
        self.factor_entry = float(factor_entry)
        self.exit_due = int(exit_due)
        self.defer = 0
        self.entry_fee = float(entry_fee)
        self.entry_defer = int(entry_defer)


class _Ord:
    __slots__ = ("sym", "decision_day", "exec_day", "expire_day", "target", "rank", "defer", "status")

    def __init__(self, sym, decision_day, exec_day, expire_day, target, rank):
        self.sym = sym
        self.decision_day = int(decision_day)
        self.exec_day = int(exec_day)
        self.expire_day = int(expire_day)
        self.target = float(target)
        self.rank = float(rank)
        self.defer = 0
        self.status = "active"


def _slip_price(px: float, side: str, bps: float) -> float:
    return fee.apply_slippage(float(px), side, float(bps))


def simulate(p: Panel, sig: np.ndarray, cfg: SimCfg) -> dict:
    if cfg.start and cfg.end:
        start, end = cfg.start, cfg.end
    else:
        start, end = WINDOWS[cfg.window]
    d0 = p.day_index(start)
    d1 = p.day_index(end)
    if d1 <= d0:
        raise ValueError("窗口为空")
    if end > DATA_END:
        raise ValueError("窗口越过 2020-12-31")

    o, c, f = p.arrays["open"], p.arrays["close"], p.arrays["factor"]
    pre = p.arrays["preclose"]
    susp, st = p.arrays["is_suspended"], p.arrays["is_st"]
    lu, ld = p.arrays["limit_up_open"], p.arrays["limit_down_open"]
    adv = p.adv20()
    fclose = pd.DataFrame(c).ffill(axis=1).to_numpy(np.float32)
    ffactor = pd.DataFrame(f).ffill(axis=1).to_numpy(np.float32)
    fadv = pd.DataFrame(adv).ffill(axis=1).to_numpy(np.float32)
    volume = p.arrays["volume"]

    dates = p.dates
    date_objs = [pd.Timestamp(x).date() for x in dates]
    syms = p.symbols
    fee_cfg = make_fee_cfg(cfg.slippage_bps, cfg.min_commission, cfg.tax_mode, cfg.transfer_mode)
    rng = np.random.default_rng(cfg.seed)

    cash = float(cfg.capital)
    positions: dict[int, _Pos] = {}
    orders: list[_Ord] = []
    trades: list[dict] = []
    failures: list[dict] = []
    nav = np.full(d1 - d0 + 1, np.nan, dtype=np.float64)
    nav_mark = np.full_like(nav, np.nan)
    cash_hist = np.full_like(nav, np.nan)
    npos_hist = np.zeros_like(nav, dtype=np.int32)
    expo_hist = np.zeros_like(nav, dtype=np.float64)
    buy_amt = 0.0
    sell_amt = 0.0
    total_fees = 0.0
    min_comm_total = 0.0
    min_comm_count = 0
    part_entry: list[float] = []
    part_exit: list[float] = []
    buy_signals = 0
    buy_filled = 0
    sell_defer_days = 0
    forced = 0
    zero_lot = 0
    no_cash = 0
    limit_up_block = 0
    susp_block = 0
    no_slot = 0
    already = 0

    def add_failure(d, sym, side, reason, target=0.0, advd=0.0):
        if not cfg.collect_trades:
            return
        failures.append({
            "candidate": cfg.candidate, "window": cfg.window, "capital": cfg.capital,
            "slots": cfg.slots, "slippage_bps": cfg.slippage_bps, "seed": cfg.seed,
            "date": dates[d].astype(str), "code": syms[sym], "side": side,
            "reason": reason, "target_value": round(float(target), 2),
            "adv20": round(float(advd), 2) if np.isfinite(advd) else None,
        })

    for di, d in enumerate(range(d0, d1 + 1)):
        # ---------- 1) 卖：先处理到期/顺延 ----------
        for sym in list(positions.keys()):
            pos = positions.get(sym)
            if pos is None or pos.shares <= 0:
                continue
            if d < pos.exit_due or d <= pos.entry_day:
                continue  # 未到期 / T+1 硬约束
            px_open = o[sym, d]
            can = (np.isfinite(px_open) and px_open > 0 and not susp[sym, d]
                   and np.isfinite(volume[sym, d]) and volume[sym, d] > 0)
            if not can:
                pos.defer += 1
                sell_defer_days += 1
                if pos.defer >= cfg.max_sell_defer:
                    px_close = fclose[sym, d]
                    if np.isfinite(px_close) and px_close > 0:
                        px = _slip_price(px_close, "sell", cfg.slippage_bps)
                        cash, f_amt, amt = _sell_fill(
                            p, cfg, fee_cfg, pos, d, px, pos.shares, cash, trades,
                            dates, syms, "forced_no_bar", ffactor[sym, d],
                            part=np.nan, raw_px=px_close)
                        total_fees += f_amt
                        sell_amt += amt
                        positions.pop(sym, None)
                        forced += 1
                continue
            if ld[sym, d] and cfg.limit_down == "persist":
                pos.defer += 1
                sell_defer_days += 1
                if pos.defer >= cfg.max_sell_defer:
                    px_close = fclose[sym, d]
                    if np.isfinite(px_close) and px_close > 0:
                        px = _slip_price(px_close, "sell", cfg.slippage_bps)
                        cash, f_amt, amt = _sell_fill(
                            p, cfg, fee_cfg, pos, d, px, pos.shares, cash, trades,
                            dates, syms, "forced_limit_down", ffactor[sym, d],
                            part=np.nan, raw_px=px_close)
                        total_fees += f_amt
                        sell_amt += amt
                        positions.pop(sym, None)
                        forced += 1
                continue
            # 正常开盘卖出
            px = _slip_price(px_open, "sell", cfg.slippage_bps)
            a = fadv[sym, max(0, d - 1)]
            if cfg.impact and np.isfinite(a) and a > 0:
                part = min(1.0, pos.shares * px / a)
                bps = min(100.0, cfg.slippage_bps + 10000.0 * 0.02 * math.sqrt(max(0.0, part)))
                px = _slip_price(px_open, "sell", bps)
            cap_amt = a * cfg.participation
            cap_qty = int(cap_amt / px) if np.isfinite(cap_amt) and px > 0 else pos.shares
            qty = pos.shares if cap_qty >= pos.shares else max(0, (cap_qty // 100) * 100)
            if qty <= 0:
                qty = pos.shares if pos.shares < 100 else 0
            if qty <= 0:
                pos.defer += 1
                sell_defer_days += 1
                continue
            part = (qty * px / a) if np.isfinite(a) and a > 0 else np.nan
            cash, f_amt, amt = _sell_fill(
                p, cfg, fee_cfg, pos, d, px, qty, cash, trades, dates, syms,
                "open_exit", ffactor[sym, d], part=part, raw_px=px_open,
                factor_prev=ffactor[sym, max(0, d - 1)], preclose=pre[sym, d],
                exit_at="open")
            total_fees += f_amt
            sell_amt += amt
            pos.shares -= qty
            if pos.shares <= 0:
                positions.pop(sym, None)

        # ---------- 2) 买：先卖后买，现金可用 ----------
        active = [x for x in orders if x.status == "active" and x.exec_day <= d]
        # 已持仓/已过期
        keep = []
        for od in active:
            if od.sym in positions:
                od.status = "cancelled"
                already += 1
                add_failure(d, od.sym, "buy", "already_held", od.target, adv[od.sym, od.decision_day])
                continue
            if d > od.expire_day:
                od.status = "cancelled"
                if od.defer > 0:
                    limit_up_block += 1
                    add_failure(d, od.sym, "buy", "limit_up_expired", od.target, adv[od.sym, od.decision_day])
                else:
                    no_slot += 1
                    add_failure(d, od.sym, "buy", "no_slot_expired", od.target, adv[od.sym, od.decision_day])
                continue
            keep.append(od)
        active = keep
        # 排序：已顺延优先，其次 rank
        active.sort(key=lambda x: (0 if x.defer > 0 else 1, x.rank))
        slots_left = cfg.slots - len(positions)
        for od in active:
            if slots_left <= 0:
                if od.defer > 0:
                    continue
                od.status = "cancelled"
                no_slot += 1
                add_failure(d, od.sym, "buy", "no_slot", od.target, adv[od.sym, od.decision_day])
                continue
            sym = od.sym
            px_open = o[sym, d]
            can = (np.isfinite(px_open) and px_open > 0 and not susp[sym, d]
                   and np.isfinite(volume[sym, d]) and volume[sym, d] > 0)
            if not can:
                if cfg.limit_up == "defer" and d < od.expire_day:
                    od.defer += 1
                    susp_block += 1
                    continue
                od.status = "cancelled"
                susp_block += 1
                add_failure(d, sym, "buy", "suspended", od.target, adv[sym, od.decision_day])
                continue
            if lu[sym, d]:
                if cfg.limit_up == "defer" and d < od.expire_day:
                    od.defer += 1
                    limit_up_block += 1
                    continue
                od.status = "cancelled"
                limit_up_block += 1
                add_failure(d, sym, "buy", "limit_up", od.target, adv[sym, od.decision_day])
                continue
            px = _slip_price(px_open, "buy", cfg.slippage_bps)
            # 先按目标与参与率算股数
            target = od.target
            qty_t = int((target / px) // 100 * 100) if px > 0 else 0
            a = fadv[sym, od.decision_day]
            cap_qty = int((a * cfg.participation / px) // 100 * 100) if np.isfinite(a) and a > 0 and px > 0 else qty_t
            qty = min(qty_t, cap_qty)
            if cfg.impact and np.isfinite(a) and a > 0 and qty > 0:
                part = min(1.0, qty * px / a)
                bps = min(100.0, cfg.slippage_bps + 10000.0 * 0.02 * math.sqrt(max(0.0, part)))
                px = _slip_price(px_open, "buy", bps)
                qty = int((min(target, a * cfg.participation) / px) // 100 * 100)
            if qty < 100:
                od.status = "cancelled"
                zero_lot += 1
                add_failure(d, sym, "buy", "below_lot", target, a)
                continue
            # 现金约束
            while qty >= 100:
                fcost = fee.trade_fees(date_objs[d], syms[sym], "buy", px, qty, fee_cfg)
                if qty * px + fcost.total <= cash + 1e-6:
                    break
                qty -= 100
            if qty < 100:
                od.status = "cancelled"
                no_cash += 1
                add_failure(d, sym, "buy", "insufficient_cash", target, a)
                continue
            fcost = fee.trade_fees(date_objs[d], syms[sym], "buy", px, qty, fee_cfg)
            amount = qty * px
            cash -= amount + fcost.total
            total_fees += fcost.total
            if fcost.commission <= cfg.min_commission + 1e-9:
                min_comm_total += fcost.commission
                min_comm_count += 1
            part = amount / a if np.isfinite(a) and a > 0 else np.nan
            part_entry.append(part)
            pos = _Pos(sym, qty, d, px, fclose[sym, d], f[sym, d], d + cfg.hold_days, fcost.total, od.defer)
            positions[sym] = pos
            od.status = "filled"
            buy_filled += 1
            buy_amt += amount
            slots_left -= 1
        # 清理已成交/取消订单
        orders = [x for x in orders if x.status == "active"]

        # ---------- 3) 收盘：估值、生成 t+1 信号 ----------
        mv = 0.0
        for sym, pos in positions.items():
            fac = ffactor[sym, d]
            if not np.isfinite(fac) or fac <= 0:
                fac = pos.factor_entry
            mv += pos.shares * pos.entry_close * fac / pos.factor_entry
        cur_nav = cash + mv
        nav[di] = cur_nav
        nav_mark[di] = cur_nav
        cash_hist[di] = cash
        npos_hist[di] = len(positions)
        expo_hist[di] = mv / cur_nav if cur_nav > 0 else 0.0

        if d < d1:
            sig_syms = np.where(sig[:, d])[0]
            if sig_syms.size:
                buy_signals += int(sig_syms.size)
                existing = set(positions.keys())
                active_syms = {x.sym for x in orders}
                for sym in sig_syms:
                    si = int(sym)
                    if si in existing or si in active_syms:
                        continue
                    target = cur_nav / max(1, cfg.slots)
                    rank = rng.random() if cfg.ordering == "random" else -float(adv[si, d]) if np.isfinite(adv[si, d]) else 0.0
                    exec_day = d + 1 + cfg.delay
                    if exec_day > d1:
                        continue
                    expire = exec_day + (3 if cfg.limit_up == "defer" else 0)
                    orders.append(_Ord(si, d, exec_day, expire, target, rank))
                    active_syms.add(si)

    # ---------- 4) 期末强制平仓 ----------
    for sym, pos in list(positions.items()):
        px_raw = fclose[sym, d1]
        if not np.isfinite(px_raw) or px_raw <= 0:
            px_raw = pos.entry_px
        px = _slip_price(px_raw, "sell", cfg.slippage_bps)
        cash, f_amt, amt = _sell_fill(
            p, cfg, fee_cfg, pos, d1, px, pos.shares, cash, trades,
            dates, syms, "end_of_window", ffactor[sym, d1], part=np.nan, raw_px=px_raw)
        total_fees += f_amt
        sell_amt += amt
        forced += 1
        positions.pop(sym, None)

    nav[-1] = cash
    nav_arr = nav
    nav_full = np.concatenate([[float(cfg.capital)], nav_arr])
    ret = nav_full[1:] / nav_full[:-1] - 1.0
    days = (pd.Timestamp(dates[d1]) - pd.Timestamp(dates[d0])).days
    years = days / 365.25
    cagr = (nav_arr[-1] / cfg.capital) ** (1.0 / years) - 1.0 if years > 0 and nav_arr[-1] > 0 else np.nan
    mdd = float(np.max(1.0 - nav_full / np.maximum.accumulate(nav_full))) if nav_full.size else np.nan
    vol = float(np.nanstd(ret, ddof=1) * math.sqrt(TRADING_DAYS)) if ret.size > 1 else np.nan
    sharpe = float(np.nanmean(ret) / np.nanstd(ret, ddof=1) * math.sqrt(TRADING_DAYS)) if ret.size > 1 and np.nanstd(ret, ddof=1) > 0 else np.nan
    calmar = cagr / mdd if mdd and mdd > 0 else np.nan
    avg_nav = float(np.nanmean(nav_arr))
    turnover = (buy_amt + sell_amt) / avg_nav / years if avg_nav > 0 and years > 0 else np.nan
    win = float(np.mean([t["net_ret"] > 0 for t in trades])) if trades else np.nan
    holds = [t["hold_days"] for t in trades]
    avg_hold = float(np.mean(holds)) if holds else np.nan
    comm = [t["commission"] for t in trades]
    min_comm_share = (sum(c for c in comm if c <= cfg.min_commission + 1e-9) / sum(comm)) if comm and sum(comm) > 0 else np.nan
    pex = [t["part_exit"] for t in trades if t.get("part_exit") is not None]

    result = {
        "candidate": cfg.candidate, "window": cfg.window,
        "window_start": start, "window_end": end,
        "adjust": "raw_fill/factor_return", "cost_model_id": "C-STRESS-V1",
        "capital": cfg.capital, "slots": cfg.slots, "slippage_bps": cfg.slippage_bps,
        "seed": cfg.seed, "participation": cfg.participation,
        "min_commission": cfg.min_commission, "tax_mode": cfg.tax_mode,
        "transfer_mode": cfg.transfer_mode, "impact": cfg.impact, "delay": cfg.delay,
        "limit_up": cfg.limit_up, "limit_down": cfg.limit_down,
        "max_sell_defer": cfg.max_sell_defer, "ordering": cfg.ordering,
        "n_days": int(d1 - d0 + 1), "years": round(years, 4),
        "cagr": cagr, "mdd": mdd, "calmar": calmar, "vol": vol, "sharpe": sharpe,
        "nav_end": float(nav_arr[-1]), "nav_mark_end": float(nav_mark[-1]),
        "n_trades": len(trades),
        "win_rate": win, "avg_hold": avg_hold, "turnover": turnover,
        "avg_pos": float(np.mean(npos_hist)), "avg_exposure": float(np.mean(expo_hist)),
        "cash_pct": float(1.0 - np.mean(expo_hist)),
        "total_fees": total_fees, "fees_pct_capital": total_fees / cfg.capital,
        "min_comm_total": min_comm_total, "min_comm_count": min_comm_count,
        "min_comm_share": min_comm_share,
        "buy_signals": int(buy_signals), "buy_filled": int(buy_filled),
        "buy_limit_up_block": int(limit_up_block), "buy_suspended_block": int(susp_block),
        "buy_no_slot": int(no_slot), "buy_below_lot": int(zero_lot),
        "buy_no_cash": int(no_cash), "buy_already": int(already),
        "sell_defer_days": int(sell_defer_days), "forced_exits": int(forced),
        "part_entry_med": float(np.nanmedian(part_entry)) if part_entry else np.nan,
        "part_entry_p95": float(np.nanpercentile(part_entry, 95)) if part_entry else np.nan,
        "part_entry_max": float(np.nanmax(part_entry)) if part_entry else np.nan,
        "part_exit_med": float(np.nanmedian(pex)) if pex else np.nan,
        "part_exit_p95": float(np.nanpercentile(pex, 95)) if pex else np.nan,
        "avg_trade_value": buy_amt / max(1, buy_filled),
    }
    return {"metrics": result, "nav": nav_arr, "dates": dates[d0:d1 + 1],
            "trades": trades, "failures": failures}


def _sell_fill(p, cfg, fee_cfg, pos, d, px, qty, cash, trades, dates, syms,
               reason, factor_exit, part=np.nan, raw_px=None, factor_prev=None,
               preclose=None, exit_at="close"):
    """卖出一笔；px 已是含滑点的成交价。返回 (cash, fees, amount)。

    factor 是累计总收益指数，入场日收盘价 × factor 比值才是持仓总收益市值。
    开盘卖出用 factor_{d-1} 与 preclose_d 还原隔夜收益，避免把因子与原始价重复相乘。
    """
    qty = int(min(qty, pos.shares))
    if qty <= 0:
        return cash, 0.0, 0.0
    fcost = fee.trade_fees(pd.Timestamp(dates[d]).date(), syms[pos.sym], "sell", px, qty, fee_cfg)
    amount = qty * px
    fac = float(factor_exit) if np.isfinite(factor_exit) and factor_exit > 0 else pos.factor_entry
    if exit_at == "open":
        fp = float(factor_prev) if (factor_prev is not None and np.isfinite(factor_prev) and factor_prev > 0) else pos.factor_entry
        pc = float(preclose) if (preclose is not None and np.isfinite(preclose) and preclose > 0) else raw_px
        ratio = (raw_px / pc) if (raw_px is not None and pc and pc > 0) else 1.0
        ratio = min(1.5, max(0.5, ratio))
        gross_value = qty * pos.entry_close * (fp / pos.factor_entry) * ratio
    else:
        gross_value = qty * pos.entry_close * (fac / pos.factor_entry)
    slip_ratio = (px / raw_px) if (raw_px is not None and np.isfinite(raw_px) and raw_px > 0) else (1.0 - cfg.slippage_bps / 10000.0)
    gross_value *= max(0.0, slip_ratio)
    if not np.isfinite(gross_value) or gross_value <= 0:
        gross_value = amount
    proceeds = gross_value - fcost.total
    cash += proceeds
    entry_fee_alloc = pos.entry_fee * qty / max(1, pos.initial)
    pnl = proceeds - qty * pos.entry_px - entry_fee_alloc
    cost_basis = qty * pos.entry_px + entry_fee_alloc
    if np.isfinite(part):
        part_exit_append = float(part)
    else:
        part_exit_append = None
    trades.append({
        "candidate": cfg.candidate, "window": cfg.window, "capital": cfg.capital,
        "slots": cfg.slots, "slippage_bps": cfg.slippage_bps, "seed": cfg.seed,
        "code": syms[pos.sym], "entry_date": dates[pos.entry_day].astype(str),
        "exit_date": dates[d].astype(str), "entry_px": round(pos.entry_px, 4),
        "exit_px": round(px, 4), "shares": qty,
        "entry_value": round(qty * pos.entry_px, 2),
        "gross_ret": round((gross_value / qty) / pos.entry_px - 1.0, 6),
        "net_ret": round(pnl / cost_basis, 6) if cost_basis > 0 else None,
        "pnl": round(pnl, 2), "fees": round(fcost.total, 2),
        "commission": round(fcost.commission, 2),
        "stamp": round(fcost.stamp_duty, 2),
        "transfer": round(fcost.transfer_fee, 2),
        "hold_days": int(d - pos.entry_day), "exit_reason": reason,
        "entry_defer": int(pos.entry_defer), "part_entry": None,
        "part_exit": part_exit_append,
        "forced": 1 if (reason.startswith("forced") or reason == "end_of_window") else 0,
        "amount": round(amount, 2),
    })
    return cash, fcost.total, amount


# ---------------------------------------------------------------------------
# 运行矩阵
# ---------------------------------------------------------------------------

def _configs() -> list[SimCfg]:
    cfgs: list[SimCfg] = []
    seen = set()

    def add(**kw):
        c = SimCfg(**kw)
        if c.key in seen:
            return
        seen.add(c.key)
        cfgs.append(c)

    # 主网格
    for cap in (20_000, 100_000, 1_000_000, 10_000_000):
        for slots in (5, 10, 20):
            for bps in (5, 10, 20, 30):
                for seed in BASE_SEEDS:
                    add(candidate="B55H20", hold_days=20, window="dev", capital=cap,
                        slots=slots, slippage_bps=bps, seed=seed)
                    add(candidate="B55H20", hold_days=20, window="val", capital=cap,
                        slots=slots, slippage_bps=bps, seed=seed)
    for cap in (20_000, 100_000, 1_000_000, 10_000_000):
        for slots in (5, 10, 20):
            for bps in (10, 30):
                for seed in BASE_SEEDS:
                    add(candidate="B55H60", hold_days=60, window="dev", capital=cap,
                        slots=slots, slippage_bps=bps, seed=seed)
                    add(candidate="B55H60", hold_days=60, window="val", capital=cap,
                        slots=slots, slippage_bps=bps, seed=seed)

    # 成本分解（100万/10仓/10bp）
    for name, kw in {
        "min_comm_0": dict(min_commission=0.0),
        "tax_0": dict(tax_mode="zero"),
        "transfer_par_0.1pct": dict(transfer_mode="par_0.1pct"),
        "impact_on": dict(impact=True),
    }.items():
        for seed in BASE_SEEDS:
            for w in ("dev", "val"):
                add(candidate="B55H20", hold_days=20, window=w, capital=1_000_000,
                    slots=10, slippage_bps=10, seed=seed, **kw)

    # 容量网格（10仓/10bp）
    for cap in (20_000, 100_000, 1_000_000, 10_000_000):
        for part in (0.005, 0.02, 0.05):
            for seed in BASE_SEEDS:
                for w in ("dev", "val"):
                    add(candidate="B55H20", hold_days=20, window=w, capital=cap,
                        slots=10, slippage_bps=10, seed=seed, participation=part)

    # 执行规则（100万/1000万，10仓，10bp）
    for cap in (1_000_000, 10_000_000):
        for delay in (0, 1):
            for lu_mode in ("cancel", "defer"):
                for ld_mode in ("persist", "optimistic"):
                    for seed in BASE_SEEDS:
                        for w in ("dev", "val"):
                            add(candidate="B55H20", hold_days=20, window=w, capital=cap,
                                slots=10, slippage_bps=10, seed=seed, delay=delay,
                                limit_up=lu_mode, limit_down=ld_mode)

    # 卖出顺延长度（100万/10仓/10bp）
    for sd in (1, 5, 10, 20):
        for seed in BASE_SEEDS:
            for w in ("dev", "val"):
                add(candidate="B55H20", hold_days=20, window=w, capital=1_000_000,
                    slots=10, slippage_bps=10, seed=seed, max_sell_defer=sd)

    # 种子分布
    for cand, hold in (("B55H20", 20), ("B55H60", 60)):
        for cap, slots in ((20_000, 10), (1_000_000, 10), (10_000_000, 10), (10_000_000, 5)):
            for seed in KEY_SEEDS:
                for w in ("dev", "val"):
                    add(candidate=cand, hold_days=hold, window=w, capital=cap,
                        slots=slots, slippage_bps=10, seed=seed)
    return cfgs


def run_matrix(cfgs: list[SimCfg] | None = None, limit: int | None = None,
               resume: bool = True) -> pd.DataFrame:
    p = build_panel()
    sig = compute_signals(p)
    np.save(CACHE / "signals.npy", sig)
    cfgs = cfgs or _configs()
    if limit:
        cfgs = cfgs[:limit]
    rows_path = OUT / "01_压力矩阵.csv"
    done = set()
    if resume and rows_path.exists():
        old = pd.read_csv(rows_path)
        done = set(map(tuple, old[[
            "candidate", "window", "capital", "slots", "slippage_bps", "seed",
            "participation", "min_commission", "tax_mode", "transfer_mode", "impact",
            "delay", "limit_up", "limit_down", "max_sell_defer", "ordering",
        ]].astype(str).values))
        rows = old.to_dict("records")
    else:
        rows = []
    t0 = time.time()
    for i, cfg in enumerate(cfgs):
        key = tuple(str(x) for x in (
            cfg.candidate, cfg.window, cfg.capital, cfg.slots, cfg.slippage_bps, cfg.seed,
            cfg.participation, cfg.min_commission, cfg.tax_mode, cfg.transfer_mode,
            cfg.impact, cfg.delay, cfg.limit_up, cfg.limit_down, cfg.max_sell_defer, cfg.ordering,
        ))
        if key in done:
            continue
        out = simulate(p, sig, cfg)
        rows.append(out["metrics"])
        if (i + 1) % 20 == 0 or i == len(cfgs) - 1:
            print(f"  {i+1}/{len(cfgs)}  {time.time()-t0:.1f}s  "
                  f"{cfg.candidate} {cfg.window} {cfg.capital:.0f}/{cfg.slots} {cfg.slippage_bps}bp "
                  f"cagr={out['metrics']['cagr']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(rows_path, index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    df.to_csv(rows_path, index=False, encoding="utf-8-sig")
    print(f"写入 {rows_path}  {len(df)} 行")
    return df


# ---------------------------------------------------------------------------
# 第一阶段候选：固定交易集合重定价
# ---------------------------------------------------------------------------

def _canon_code(code: str) -> str:
    s = str(code)
    if s.startswith("sh."):
        return s[3:] + ".SH"
    if s.startswith("sz."):
        return s[3:] + ".SZ"
    if s.startswith("bj."):
        return s[3:] + ".BJ"
    return s


def _replay_fixed(p: Panel, trades: pd.DataFrame, capital: float, slots: int,
                  bps: float, window: str, participation: float = BASE_PARTICIPATION) -> dict:
    start, end = WINDOWS[window]
    d0, d1 = p.day_index(start), p.day_index(end)
    dates = p.dates
    date_objs = [pd.Timestamp(x).date() for x in dates]
    f = p.arrays["factor"]
    c = p.arrays["close"]
    o = p.arrays["open"]
    pre = p.arrays["preclose"]
    adv = p.adv20()
    fclose = pd.DataFrame(c).ffill(axis=1).to_numpy(np.float32)
    ffactor = pd.DataFrame(f).ffill(axis=1).to_numpy(np.float32)
    fee_cfg = make_fee_cfg(bps)

    sub = trades.copy()
    n_input_all = len(sub)
    sub["code"] = sub["code"].map(_canon_code)
    sub = sub[sub["code"].isin(p.sym_index)].copy()
    n_missing_symbol = n_input_all - len(sub)
    sub["entry_day"] = sub["entry_date"].map(lambda x: p.day_index(pd.Timestamp(x).strftime("%Y-%m-%d")))
    sub["exit_day"] = sub["exit_date"].map(lambda x: p.day_index(pd.Timestamp(x).strftime("%Y-%m-%d")))
    sub = sub[(sub["entry_day"] >= d0) & (sub["entry_day"] <= d1)].copy()
    sub = sub.sort_values(["entry_day", "code"])
    by_entry: dict[int, list[dict]] = {}
    for r in sub.to_dict("records"):
        by_entry.setdefault(int(r["entry_day"]), []).append(r)

    cash = float(capital)
    pos: dict[str, dict] = {}
    nav = np.full(d1 - d0 + 1, np.nan)
    n_fill = 0
    n_skip_slot = 0
    n_skip_lot = 0
    n_skip_dup = 0
    n_capped = 0
    fees_total = 0.0
    fill_ratios = []
    values = []
    wins = 0
    pnls = []

    for d in range(d0, d1 + 1):
        # 到期卖出
        for code in list(pos.keys()):
            q = pos[code]
            if q["exit_day"] > d:
                continue
            px = q["exit_px"] * (1.0 - bps / 10000.0)
            px = max(0.01, round(px, 2))
            qty = q["shares"]
            fcost = fee.trade_fees(date_objs[d], code, "sell", px, qty, fee_cfg)
            si = p.sym_index[code]
            fac_prev = ffactor[si, max(0, q["exit_day"] - 1)]
            if not np.isfinite(fac_prev) or fac_prev <= 0:
                fac_prev = q["factor_entry"]
            pc = pre[si, q["exit_day"]]
            ratio = (q["exit_px"] / pc) if (np.isfinite(pc) and pc > 0) else 1.0
            ratio = min(1.5, max(0.5, ratio))
            gross = qty * q["entry_close"] * (fac_prev / q["factor_entry"]) * ratio * (px / q["exit_px"])
            proceeds = gross - fcost.total
            cash += proceeds
            fees_total += fcost.total
            entry_cost = qty * q["entry_px"] + q["entry_fee"]
            pnl = proceeds - entry_cost
            pnls.append(pnl)
            wins += 1 if pnl > 0 else 0
            del pos[code]

        # 当日入场
        for r in by_entry.get(d, []):
            code = r["code"]
            if len(pos) >= slots:
                n_skip_slot += 1
                continue
            if code in pos:
                n_skip_dup += 1
                continue
            si = p.sym_index[code]
            target = (nav[d - 1 - d0] if d > d0 else capital) / max(1, slots)
            px = r["entry_px"] * (1.0 + bps / 10000.0)
            px = max(0.01, round(px, 2))
            qty_t = int((target / px) // 100 * 100) if px > 0 else 0
            a = adv[si, d]
            cap_qty = int((a * participation / px) // 100 * 100) if np.isfinite(a) and a > 0 else qty_t
            qty = min(qty_t, cap_qty)
            if qty < qty_t:
                n_capped += 1
            while qty >= 100:
                fcost = fee.trade_fees(date_objs[d], code, "buy", px, qty, fee_cfg)
                if qty * px + fcost.total <= cash + 1e-6:
                    break
                qty -= 100
            if qty < 100:
                n_skip_lot += 1
                continue
            fcost = fee.trade_fees(date_objs[d], code, "buy", px, qty, fee_cfg)
            cost = qty * px + fcost.total
            cash -= cost
            fees_total += fcost.total
            fac = ffactor[si, d]
            if not np.isfinite(fac) or fac <= 0:
                fac = 1.0
            pos[code] = {
                "shares": qty, "entry_px": px, "entry_close": fclose[si, d],
                "factor_entry": fac, "entry_fee": fcost.total,
                "exit_day": int(r["exit_day"]), "exit_px": float(r["exit_px"]),
            }
            n_fill += 1
            fill_ratios.append((qty * px) / target if target > 0 else np.nan)
            values.append(qty * px)

        # 估值
        mv = 0.0
        for code, q in pos.items():
            si = p.sym_index[code]
            fac = ffactor[si, d]
            if not np.isfinite(fac) or fac <= 0:
                fac = q["factor_entry"]
            mv += q["shares"] * q["entry_close"] * fac / q["factor_entry"]
        nav[d - d0] = cash + mv

    # 期末强平
    for code, q in list(pos.items()):
        px_raw = fclose[p.sym_index[code], d1]
        if not np.isfinite(px_raw) or px_raw <= 0:
            px_raw = q["entry_px"]
        px = max(0.01, round(px_raw * (1.0 - bps / 10000.0), 2))
        fcost = fee.trade_fees(date_objs[d1], code, "sell", px, q["shares"], fee_cfg)
        fac = ffactor[p.sym_index[code], d1]
        if not np.isfinite(fac) or fac <= 0:
            fac = q["factor_entry"]
        proceeds = q["shares"] * q["entry_close"] * (fac / q["factor_entry"]) * (px / px_raw) - fcost.total
        cash += proceeds
        fees_total += fcost.total
        entry_cost = q["shares"] * q["entry_px"] + q["entry_fee"]
        pnls.append(proceeds - entry_cost)
        wins += 1 if proceeds - entry_cost > 0 else 0
        del pos[code]
    nav[-1] = cash

    nav_full = np.concatenate([[capital], nav])
    ret = nav_full[1:] / nav_full[:-1] - 1.0
    days = (pd.Timestamp(dates[d1]) - pd.Timestamp(dates[d0])).days
    years = days / 365.25
    cagr = (cash / capital) ** (1.0 / years) - 1.0 if years > 0 and cash > 0 else np.nan
    mdd = float(np.max(1.0 - nav_full / np.maximum.accumulate(nav_full)))
    vol = float(np.nanstd(ret, ddof=1) * math.sqrt(TRADING_DAYS)) if ret.size > 1 else np.nan
    return {
        "candidate": "fixed_set", "window": window,
        "window_start": start, "window_end": end,
        "adjust": "raw_fill/factor_return", "cost_model_id": "C-STRESS-V1",
        "replay": "fixed_trade_set", "capital": capital, "slots": slots,
        "slippage_bps": bps, "participation": participation,
        "n_input_all": n_input_all, "n_missing_symbol": n_missing_symbol,
        "n_trades": n_fill, "n_exits": len(pnls), "win_rate": (wins / len(pnls)) if pnls else np.nan,
        "cagr": cagr, "mdd": mdd, "vol": vol,
        "sharpe": float(np.nanmean(ret) / np.nanstd(ret, ddof=1) * math.sqrt(TRADING_DAYS)) if ret.size > 1 and np.nanstd(ret, ddof=1) > 0 else np.nan,
        "nav_end": cash, "fees_total": fees_total, "fees_pct_capital": fees_total / capital,
        "n_skip_slot": n_skip_slot, "n_skip_lot": n_skip_lot, "n_skip_dup": n_skip_dup,
        "n_capped": n_capped, "avg_fill_ratio": float(np.nanmean(fill_ratios)) if fill_ratios else np.nan,
        "avg_value": float(np.mean(values)) if values else np.nan,
        "avg_pnl": float(np.mean(pnls)) if pnls else np.nan,
    }


def run_append():
    p = build_panel()
    src = ROOT / "03_结果" / "趋势族" / "交易明细_全部.parquet"
    df = pd.read_parquet(src)
    specs = [
        ("S4时间60", "S4_结构突破55", "纯时间60"),
        ("S1时间60", "S1_月周日共振突破", "纯时间60"),
    ]
    rows = []
    t0 = time.time()
    for cand, sname, rule in specs:
        for w in ("dev", "val"):
            sub = df[(df["signal"] == sname) & (df["rule"] == rule) & (df["period"] == w)].copy()
            print(f"  {cand} {w}: {len(sub)} 笔")
            for cap in (20_000, 100_000, 1_000_000, 10_000_000):
                for slots in (5, 10, 20):
                    for bps in (5, 10, 20, 30):
                        r = _replay_fixed(p, sub, cap, slots, bps, w)
                        r["candidate"] = cand
                        r["source_signal"] = sname
                        r["source_rule"] = rule
                        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "06_候选_交易级重算.csv", index=False, encoding="utf-8-sig")
    print(f"写入 {OUT/'06_候选_交易级重算.csv'}  {len(out)} 行, {time.time()-t0:.1f}s")
    return out


def run_trade_cost():
    """候选交易明细的“单笔成本”重算：不依赖账户路径，只看每笔在不同体量下的费用拖累。"""
    p = build_panel()
    adv = p.adv20()
    src = ROOT / "03_结果" / "趋势族" / "交易明细_全部.parquet"
    df = pd.read_parquet(src)
    specs = [("S4时间60", "S4_结构突破55", "纯时间60"),
             ("S1时间60", "S1_月周日共振突破", "纯时间60")]
    rows = []
    for cand, sname, rule in specs:
        for w in ("dev", "val"):
            sub = df[(df["signal"] == sname) & (df["rule"] == rule) & (df["period"] == w)].copy()
            n_input_all = len(sub)
            sub["code"] = sub["code"].map(_canon_code)
            sub = sub[sub["code"].isin(p.sym_index)]
            n_missing_symbol = n_input_all - len(sub)
            for cap in (20_000, 100_000, 1_000_000, 10_000_000):
                for slots in (5, 10, 20):
                    for bps in (5, 10, 20, 30):
                        fee_cfg = make_fee_cfg(bps)
                        target = cap / slots
                        nets = []
                        costs = []
                        skips = 0
                        caps = 0
                        for r in sub.itertuples(index=False):
                            si = p.sym_index[r.code]
                            d = p.day_index(pd.Timestamp(r.entry_date).strftime("%Y-%m-%d"))
                            de = p.day_index(pd.Timestamp(r.exit_date).strftime("%Y-%m-%d"))
                            if d >= p.n_days or de >= p.n_days:
                                skips += 1
                                continue
                            px_in = float(r.entry_px)
                            px_out = float(r.exit_px)
                            qty = int((target / px_in) // 100 * 100) if px_in > 0 else 0
                            a = adv[si, d]
                            capq = int((a * BASE_PARTICIPATION / px_in) // 100 * 100) if np.isfinite(a) and a > 0 else qty
                            if qty > capq:
                                qty = capq
                                caps += 1
                            if qty < 100:
                                skips += 1
                                continue
                            amount_in = qty * px_in
                            amount_out = qty * px_out
                            f_in = fee.trade_fees(pd.Timestamp(p.dates[d]).date(), r.code, "buy", px_in, qty, fee_cfg)
                            f_out = fee.trade_fees(pd.Timestamp(p.dates[de]).date(), r.code, "sell", px_out, qty, fee_cfg)
                            slip_cost = qty * (px_in + px_out) * bps / 10000.0
                            cost = (f_in.total + f_out.total + slip_cost) / amount_in
                            # 原交易的净收益（含原成本）；加回原成本得到毛收益
                            orig_shares = max(1, int(r.shares))
                            orig_cost = (orig_shares * (px_out - px_in) - float(r.pnl)) / (orig_shares * px_in)
                            gross = float(r.ret) + orig_cost
                            nets.append(gross - cost)
                            costs.append(cost)
                        rows.append({
                            "candidate": cand, "window": w, "capital": cap, "slots": slots,
                            "window_start": WINDOWS[w][0], "window_end": WINDOWS[w][1],
                            "adjust": "raw_fill/factor_return", "cost_model_id": "C-STRESS-V1",
                            "slippage_bps": bps, "participation": BASE_PARTICIPATION,
                            "n_input_all": n_input_all, "n_missing_symbol": n_missing_symbol,
                            "n_input": len(sub), "n_used": len(nets), "n_skip": skips, "n_capped": caps,
                            "gross_mean": float(np.mean([x + c for x, c in zip(nets, costs)])) if nets else np.nan,
                            "net_mean": float(np.mean(nets)) if nets else np.nan,
                            "net_median": float(np.median(nets)) if nets else np.nan,
                            "net_p10": float(np.percentile(nets, 10)) if nets else np.nan,
                            "cost_mean": float(np.mean(costs)) if costs else np.nan,
                            "cost_bps": float(np.mean(costs) * 10000) if costs else np.nan,
                            "win_rate": float(np.mean([x > 0 for x in nets])) if nets else np.nan,
                        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "06b_候选_单笔成本.csv", index=False, encoding="utf-8-sig")
    print(f"写入 {OUT/'06b_候选_单笔成本.csv'}  {len(out)} 行")
    return out


def run_key_details():
    """关键配置的逐笔、失败订单与分年净值（用于报告与审计）。"""
    p = build_panel()
    sig = compute_signals(p)
    all_trades = []
    all_fails = []
    all_nav = []
    for cand, hold in (("B55H20", 20), ("B55H60", 60)):
        for w in ("dev", "val"):
            for cap in (20_000, 1_000_000, 10_000_000):
                cfg = SimCfg(candidate=cand, hold_days=hold, window=w, capital=cap,
                             slots=10, slippage_bps=10, seed=11, collect_trades=True)
                out = simulate(p, sig, cfg)
                all_trades.extend(out["trades"])
                all_nav.append(pd.DataFrame({
                    "candidate": cand, "window": w, "capital": cap, "slots": 10,
                    "slippage_bps": 10, "seed": 11, "date": pd.to_datetime(out["dates"]),
                    "nav": out["nav"],
                }))
                f = pd.DataFrame(out["failures"])
                if not f.empty:
                    ns = f[f["reason"] == "no_slot"]
                    if len(ns) > 2000:
                        ns = ns.sample(n=2000, random_state=11)
                    keep = f[f["reason"] != "no_slot"]
                    all_fails.append(pd.concat([keep, ns], ignore_index=True))
    pd.DataFrame(all_trades).to_csv(OUT / "05_交易明细_关键配置.csv", index=False, encoding="utf-8-sig")
    if all_fails:
        pd.concat(all_fails, ignore_index=True).to_csv(OUT / "04_失败订单.csv", index=False, encoding="utf-8-sig")
    if all_nav:
        nav = pd.concat(all_nav, ignore_index=True)
        nav.to_csv(OUT / "05b_关键配置_净值.csv", index=False, encoding="utf-8-sig")
        rows = []
        for (cand, w, cap), g in nav.groupby(["candidate", "window", "capital"]):
            g = g.sort_values("date")
            g["year"] = g["date"].dt.year
            prev = cap
            for y, gy in g.groupby("year"):
                end = float(gy["nav"].iloc[-1])
                rows.append({
                    "candidate": cand, "window": w, "capital": cap, "slots": 10,
                    "slippage_bps": 10, "seed": 11, "year": int(y),
                    "return": end / prev - 1.0 if prev > 0 else np.nan,
                    "nav_end": end,
                })
                prev = end
        pd.DataFrame(rows).to_csv(OUT / "03_逐年.csv", index=False, encoding="utf-8-sig")
    print(f"写入 05_交易明细_关键配置.csv ({len(all_trades)} 行)、03_逐年.csv 与 04_失败订单.csv")


def run_reconcile():
    """用统一账户撮合引擎复核 2014 年的同一策略路径。"""
    import 账户撮合 as am

    p = build_panel()
    sig = compute_signals(p)
    year = ("2014-01-01", "2014-12-31")
    d0, d1 = p.day_index(year[0]), p.day_index(year[1])
    cal = [pd.Timestamp(x).date() for x in p.dates[d0:d1 + 1]]
    o, c, h, l = (p.arrays[k] for k in ("open", "close", "high", "low"))
    pre, vol, amt = p.arrays["preclose"], p.arrays["volume"], p.arrays["amount"]
    susp, st = p.arrays["is_suspended"], p.arrays["is_st"]
    adv = p.adv20()

    con = duckdb.connect(str(DB), read_only=True)
    ev = con.execute(f"""
      SELECT symbol, ex_date, dividend_per_share, per_share_bonus, rights_ratio, rights_price
      FROM corp_actions WHERE ex_date BETWEEN DATE '{year[0]}' AND DATE '{year[1]}'
    """).fetchdf()
    con.close()
    ev_by_day: dict[date, list] = {}
    for r in ev.itertuples(index=False):
        d = pd.Timestamp(r.ex_date).date()
        ev_by_day.setdefault(d, []).append(am.CorporateAction(
            code=r.symbol, ex_date=d,
            dividend_per_share=float(r.dividend_per_share or 0),
            per_share_bonus=float(r.per_share_bonus or 0),
            rights_ratio=float(r.rights_ratio or 0),
            rights_price=float(r.rights_price or 0),
            source="frozen.corp_actions",
        ))

    fee_cfg = make_fee_cfg(10.0, 5.0, "historical", "module")
    fee_cfg = am.fee.FeeConfig(
        commission_rate=fee_cfg.commission_rate, min_commission=fee_cfg.min_commission,
        stamp_schedule=fee_cfg.stamp_schedule, transfer_schedule=fee_cfg.transfer_schedule,
        slippage_bps=10.0, max_participation=1.0, dividend_tax_rate=0.0,
    )
    acfg = am.AccountConfig(initial_cash=1_000_000.0, fee=fee_cfg, cash_settle_days=0,
                            enforce_t1=True, allow_buy_downsize=True, normalize_buy_orders=True,
                            rights_policy="skip", dividend_tax_rate=0.0,
                            liquidate_on_finalize=True)
    acct = am.Account(acfg, calendar=cal, freeze_date=cal[-1])

    slots = 10
    hold = 20
    entry_idx: dict[str, int] = {}
    sell_submitted: set[str] = set()
    last_trade_n = 0
    for d in range(d0, d1 + 1):
        date_obj = pd.Timestamp(p.dates[d]).date()
        valid = (np.isfinite(o[:, d]) & np.isfinite(h[:, d]) & np.isfinite(l[:, d])
                 & np.isfinite(c[:, d]) & np.isfinite(vol[:, d]) & (vol[:, d] > 0)
                 & (~susp[:, d]) & (o[:, d] > 0))
        bars = {}
        for si in np.where(valid)[0]:
            sym = p.symbols[si]
            bars[sym] = am.Bar(
                code=sym, date=date_obj, open=float(o[si, d]), high=float(h[si, d]),
                low=float(l[si, d]), close=float(c[si, d]), volume=float(vol[si, d]),
                amount=float(amt[si, d]) if np.isfinite(amt[si, d]) else None,
                prev_close=float(pre[si, d]) if np.isfinite(pre[si, d]) else None,
                is_st=bool(st[si, d]), trade_status=1,
            )
        st_codes = {p.symbols[si] for si in np.where(st[:, d])[0]}
        rec = acct.on_day(date_obj, bars, ev_by_day.get(date_obj, []), st_codes)

        # 记录新买入的入场日
        for tr in acct.trades[last_trade_n:]:
            if tr.side == "buy":
                entry_idx[tr.code] = p.day_index(tr.date.strftime("%Y-%m-%d"))
        last_trade_n = len(acct.trades)

        if d >= d1:
            continue
        # 卖出：持有满 20 个交易日的，收盘提交次日开盘卖
        next_i = d + 1
        for code, pos in list(acct.positions.items()):
            if pos.shares <= 0 or code in sell_submitted:
                continue
            ei = entry_idx.get(code)
            if ei is not None and ei + hold <= next_i:
                acct.sell_open(code, pos.shares, decision_date=date_obj)
                sell_submitted.add(code)
        # 买入：按 ADV 降序，补足空槽；把次日开盘将卖出的仓位视为可用空槽
        n_pos = sum(1 for x in acct.positions.values() if x.shares > 0)
        pending_buy = sum(1 for x in acct.orders.values() if x.active and x.side == "buy")
        pending_sell = sum(1 for x in acct.orders.values() if x.active and x.side == "sell")
        slots_left = slots - n_pos + pending_sell - pending_buy
        if slots_left > 0:
            cand = [int(i) for i in np.where(sig[:, d])[0]]
            cand.sort(key=lambda i: -(adv[i, d] if np.isfinite(adv[i, d]) else 0.0))
            nav_prev = rec["nav"]
            for si in cand:
                if slots_left <= 0:
                    break
                sym = p.symbols[si]
                if sym in acct.positions or any(x.active and x.side == "buy" and x.code == sym for x in acct.orders.values()):
                    continue
                px = float(c[si, d]) * (1.0 + 10.0 / 10000.0)
                qty = int((nav_prev / slots) / px // 100 * 100)
                if qty < 100:
                    continue
                acct.buy_open(sym, qty, decision_date=date_obj)
                slots_left -= 1

    acct.finalize(liquidate=True)
    eng = pd.DataFrame(acct.daily)
    eng["date"] = pd.to_datetime(eng["date"])

    my = simulate(p, sig, SimCfg(
        candidate="B55H20", hold_days=hold, window="dev", capital=1_000_000, slots=slots,
        slippage_bps=10, seed=11, participation=1.0, limit_up="cancel", limit_down="persist",
        max_sell_defer=10, ordering="liquidity", collect_trades=True,
        start=year[0], end=year[1],
    ))
    mine = pd.DataFrame({"date": pd.to_datetime(my["dates"]), "nav_mine": my["nav"]})
    cmp = eng[["date", "nav"]].rename(columns={"nav": "nav_engine"}).merge(mine, on="date", how="outer")
    cmp["diff"] = cmp["nav_engine"] / cmp["nav_mine"] - 1.0
    cmp.to_csv(OUT / "08_统一引擎复核_日净值.csv", index=False, encoding="utf-8-sig")

    eng_sells = [t for t in acct.trades if t.side == "sell"]
    summary = pd.DataFrame([{
        "window": "2014-01-01..2014-12-31", "candidate": "B55H20", "capital": 1_000_000,
        "slots": slots, "slippage_bps": 10, "n_days": len(cmp),
        "nav_engine_end": float(eng["nav"].iloc[-1]), "nav_mine_end": float(my["nav"][-1]),
        "nav_diff_end": float(cmp["diff"].iloc[-1]) if pd.notna(cmp["diff"].iloc[-1]) else np.nan,
        "max_abs_diff": float(cmp["diff"].abs().max()),
        "mean_abs_diff": float(cmp["diff"].abs().mean()),
        "n_trades_engine_sells": len(eng_sells), "n_trades_mine": len(my["trades"]),
        "fees_engine": sum(t.fees for t in acct.trades), "fees_mine": float(my["metrics"]["total_fees"]),
        "algo_engine": "账户撮合.py", "participation_engine": 1.0, "participation_mine": 1.0,
        "note": "引擎无 ADV20 容量与 slot 排序；本对账只验证费用/整股/涨跌停/T+1 口径",
    }])
    summary.to_csv(OUT / "08_统一引擎复核.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return summary


def _md_table(df: pd.DataFrame, cols: list, floatfmt: str = "{:.3f}") -> str:
    d = df[list(cols)].copy()
    d.columns = [str(c) for c in cols]
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))
    head = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join(str(x) for x in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def _orig_account_cagr() -> pd.DataFrame:
    rows = []
    for cand, fname in (
        ("S4时间60", "净值_S4_结构突破55_纯时间60"),
        ("S1时间60", "净值_S1_月周日共振突破_纯时间60"),
    ):
        for w in ("dev", "val"):
            p = ROOT / "03_结果" / "趋势族" / f"{fname}_{w}.parquet"
            if not p.exists():
                continue
            d = pd.read_parquet(p).sort_values("date")
            eq = d["equity"].to_numpy()
            dates = pd.to_datetime(d["date"])
            yrs = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
            cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1
            mdd = float(np.max(1 - eq / np.maximum.accumulate(eq)))
            rows.append({"candidate": cand, "window": w, "orig_cagr": cagr, "orig_mdd": mdd,
                         "orig_nav_end": eq[-1], "orig_avg_pos": d["n_pos"].mean()})
    return pd.DataFrame(rows)


def make_report():
    rows_all = pd.read_csv(OUT / "01_压力矩阵.csv")
    # 网格比较只用预注册的 5 个种子，保证所有格子同种子集；20 种子的分布另表报告
    rows = rows_all[rows_all["seed"].isin(BASE_SEEDS)].copy()
    keys = ["candidate", "window", "capital", "slots", "slippage_bps", "participation",
            "min_commission", "tax_mode", "transfer_mode", "impact", "delay",
            "limit_up", "limit_down", "max_sell_defer", "ordering"]
    agg = rows.groupby(keys, dropna=False).agg(
        n_seeds=("seed", "nunique"),
        cagr_med=("cagr", "median"), cagr_min=("cagr", "min"), cagr_max=("cagr", "max"),
        mdd_med=("mdd", "median"), trades_med=("n_trades", "median"),
        fees_med=("fees_pct_capital", "median"), expo_med=("avg_exposure", "median"),
        filled_med=("buy_filled", "median"), noslot_med=("buy_no_slot", "median"),
        belowlot_med=("buy_below_lot", "median"), lu_med=("buy_limit_up_block", "median"),
        defer_med=("sell_defer_days", "median"), forced_med=("forced_exits", "median"),
        part_med=("part_entry_med", "median"),
    ).reset_index()
    agg.to_csv(OUT / "02_配置汇总.csv", index=False, encoding="utf-8-sig")

    # 基线表
    base = agg[(agg.slots == 10) & (agg.slippage_bps == 10) & (agg.participation == 0.02)
               & (agg.min_commission == 5) & (agg.tax_mode == "historical")
               & (agg.transfer_mode == "module") & (~agg.impact) & (agg.delay == 0)
               & (agg.limit_up == "cancel") & (agg.limit_down == "persist")
               & (agg.max_sell_defer == 10) & (agg.ordering == "random")]
    # 仓位表
    slots = agg[(agg.capital == 1_000_000) & (agg.slippage_bps == 10) & (agg.participation == 0.02)
                & (agg.min_commission == 5) & (agg.tax_mode == "historical")
                & (agg.transfer_mode == "module") & (~agg.impact) & (agg.delay == 0)
                & (agg.limit_up == "cancel") & (agg.limit_down == "persist")
                & (agg.max_sell_defer == 10) & (agg.ordering == "random")]
    # 滑点表
    slip = agg[(agg.capital == 1_000_000) & (agg.slots == 10) & (agg.participation == 0.02)
               & (agg.min_commission == 5) & (agg.tax_mode == "historical")
               & (agg.transfer_mode == "module") & (~agg.impact) & (agg.delay == 0)
               & (agg.limit_up == "cancel") & (agg.limit_down == "persist")
               & (agg.max_sell_defer == 10) & (agg.ordering == "random")]
    # 成本分解
    cost = agg[(agg.capital == 1_000_000) & (agg.slots == 10) & (agg.slippage_bps == 10)
               & (agg.participation == 0.02) & (agg.delay == 0)
               & (agg.limit_up == "cancel") & (agg.limit_down == "persist")
               & (agg.max_sell_defer == 10) & (agg.ordering == "random")]
    # 容量
    cap = agg[(agg.slots == 10) & (agg.slippage_bps == 10) & (agg.min_commission == 5)
              & (agg.tax_mode == "historical") & (agg.transfer_mode == "module") & (~agg.impact)
              & (agg.delay == 0) & (agg.limit_up == "cancel") & (agg.limit_down == "persist")
              & (agg.max_sell_defer == 10) & (agg.ordering == "random")]
    # 执行规则
    exe = agg[(agg.slots == 10) & (agg.slippage_bps == 10) & (agg.participation == 0.02)
              & (agg.min_commission == 5) & (agg.tax_mode == "historical")
              & (agg.transfer_mode == "module") & (~agg.impact)
              & (agg.max_sell_defer == 10) & (agg.ordering == "random")]
    # 种子分布：只用基线成本/执行配置，避免把 30bp 或成本变体混进来
    seed_base = rows_all[(rows_all.participation == 0.02) & (rows_all.min_commission == 5)
                         & (rows_all.tax_mode == "historical") & (rows_all.transfer_mode == "module")
                         & (~rows_all.impact) & (rows_all.delay == 0) & (rows_all.limit_up == "cancel")
                         & (rows_all.limit_down == "persist") & (rows_all.max_sell_defer == 10)
                         & (rows_all.ordering == "random") & (rows_all.slippage_bps == 10)]
    seed = seed_base.groupby(["candidate", "window", "capital", "slots"]).agg(
        n_seeds=("seed", "nunique"), cagr_med=("cagr", "median"),
        cagr_min=("cagr", "min"), cagr_max=("cagr", "max"),
        mdd_med=("mdd", "median")).reset_index()
    seed.to_csv(OUT / "07_种子分布.csv", index=False, encoding="utf-8-sig")

    orig = _orig_account_cagr()
    append = pd.read_csv(OUT / "06_候选_交易级重算.csv")
    app10 = append[(append.slots == 10) & (append.slippage_bps == 10)]
    tcost = pd.read_csv(OUT / "06b_候选_单笔成本.csv")
    tcost10 = tcost[(tcost.slots == 10) & (tcost.slippage_bps == 10)]
    yearly = pd.read_csv(OUT / "03_逐年.csv") if (OUT / "03_逐年.csv").exists() else pd.DataFrame()
    rec = pd.read_csv(OUT / "08_统一引擎复核.csv") if (OUT / "08_统一引擎复核.csv").exists() else pd.DataFrame()

    def fmt(df, cols, n=30):
        if df.empty:
            return "_无数据_"
        return _md_table(df.head(n), cols)

    lines = []
    lines.append("# 执行压力与交易容量：报告\n")
    lines.append("- 窗口：开发 2008-01-01..2015-12-31，验证 2016-01-01..2020-12-31；2021+ 未读取。")
    lines.append("- 数据：`01_数据/frozen/frozen.duckdb`，原始价成交 + `factor` 总收益；主板；ADV20 ≥ 3,000 万。")
    lines.append("- 基准候选：B55H20（55 日突破，固定 20 日）、B55H60（固定 60 日）。")
    lines.append("- 第一阶段候选追加：S4时间60、S1时间60（固定交易集合重定价，非原账户路径）。")
    lines.append("- 预注册：`00_预注册_执行压力.md`。\n")

    lines.append("## 1. 基线（10 仓，10bp，2% ADV20，5 元最低佣金，历史税费）\n")
    lines.append(fmt(base, ["candidate", "window", "capital", "slots", "cagr_med", "cagr_min", "cagr_max",
                            "mdd_med", "trades_med", "fees_med", "expo_med", "filled_med",
                            "noslot_med", "belowlot_med"], 24))
    lines.append("\n## 2. 仓位数（100 万，10bp）\n")
    lines.append(fmt(slots, ["candidate", "window", "slots", "cagr_med", "cagr_min", "cagr_max",
                             "mdd_med", "trades_med", "fees_med", "expo_med"], 24))
    lines.append("\n## 3. 滑点（100 万/10 仓）\n")
    lines.append(fmt(slip, ["candidate", "window", "slippage_bps", "cagr_med", "mdd_med",
                            "trades_med", "fees_med"], 24))
    lines.append("\n## 4. 成本分解（100 万/10 仓/10bp）\n")
    lines.append(fmt(cost, ["candidate", "window", "min_commission", "tax_mode", "transfer_mode",
                            "impact", "cagr_med", "mdd_med", "fees_med"], 24))
    lines.append("\n## 5. 容量（10 仓/10bp）\n")
    lines.append(fmt(cap, ["candidate", "window", "capital", "participation", "cagr_med",
                           "mdd_med", "filled_med", "noslot_med", "part_med"], 24))
    lines.append("\n## 6. 执行规则（10 仓/10bp）\n")
    lines.append(fmt(exe, ["candidate", "window", "capital", "delay", "limit_up", "limit_down",
                           "cagr_med", "trades_med", "lu_med", "defer_med", "forced_med"], 40))
    lines.append("\n## 7. 种子分布\n")
    lines.append(fmt(seed, ["candidate", "window", "capital", "slots", "n_seeds", "cagr_med",
                            "cagr_min", "cagr_max", "mdd_med"], 24))
    lines.append("\n## 8. 第一阶段候选：原账户 vs 固定交易集合重定价\n")
    lines.append(fmt(orig, ["candidate", "window", "orig_cagr", "orig_mdd", "orig_nav_end", "orig_avg_pos"], 8))
    lines.append("\n固定交易集合重定价（10 仓/10bp）：\n")
    lines.append(fmt(app10, ["candidate", "window", "capital", "slots", "cagr", "mdd", "n_trades",
                             "n_skip_slot", "n_skip_lot", "n_capped", "avg_fill_ratio"], 24))
    lines.append("\n单笔成本重算（10 仓/10bp）：\n")
    lines.append(fmt(tcost10, ["candidate", "window", "capital", "slots", "n_used", "n_skip",
                               "net_mean", "net_median", "cost_bps", "win_rate"], 24))
    if not yearly.empty:
        lines.append("\n## 9. 关键配置分年收益（10 仓/10bp/种子 11）\n")
        y = yearly.pivot_table(index=["candidate", "window", "capital"], columns="year",
                               values="return", aggfunc="first").reset_index()
        lines.append(fmt(y, list(y.columns), 24))
    if not rec.empty:
        lines.append("\n## 10. 统一引擎复核（2014）\n")
        lines.append(fmt(rec, list(rec.columns), 4))
    lines.append("\n## 11. 结论（待填）\n")
    lines.append("_由研究员在查看全部表格后手工填写，不用脚本自动生成结论。_")

    (OUT / "09_报告_执行压力.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 02_配置汇总.csv, 07_种子分布.csv, 09_报告_执行压力.md; matrix rows={len(rows)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "smoke", "run", "append", "keydetails", "reconcile", "report"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.cmd == "build":
        p = build_panel(force=args.force)
        sig = compute_signals(p)
        np.save(CACHE / "signals.npy", sig)
        print("signal count by year:")
        years = pd.to_datetime(p.dates).year
        s = pd.Series(sig.sum(axis=0), index=years).groupby(level=0).sum()
        print(s.to_string())
    elif args.cmd == "smoke":
        p = build_panel(force=args.force)
        sig = compute_signals(p)
        cfg = SimCfg(candidate="B55H20", hold_days=20, window="dev", capital=1_000_000,
                     slots=10, slippage_bps=10, seed=11, collect_trades=True)
        t0 = time.time()
        out = simulate(p, sig, cfg)
        print(json.dumps(out["metrics"], ensure_ascii=False, indent=2, default=str))
        print(f"耗时 {time.time()-t0:.1f}s, trades={len(out['trades'])}, failures={len(out['failures'])}")
    elif args.cmd == "run":
        run_matrix(limit=args.limit)
    elif args.cmd == "append":
        run_append()
        run_trade_cost()
    elif args.cmd == "keydetails":
        run_key_details()
    elif args.cmd == "reconcile":
        run_reconcile()
    elif args.cmd == "report":
        make_report()


if __name__ == "__main__":
    main()
