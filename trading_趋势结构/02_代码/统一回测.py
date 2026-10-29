#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一开发/验证回测入口（趋势结构第二阶段）。

本文件只写自己的产物：`03_结果/统一引擎/` 与 `04_证据/统一引擎自检.json`。
它不修改 `数据层.py`、`账户撮合.py`、`费用与交易约束.py`、`核心特征.py`、`统计验证.py`
以及各策略族引擎；冲突在 `03_结果/统一引擎/口径冲突与裁决.md` 中列明。

核心设计
--------
* 引擎：复用第一阶段 `账户撮合.Account` 的现金/股数/T+1/涨跌停/停牌/费用/退市逻辑；
  通过子类补三件事：
    1. 公司行动以交易所 `preclose` 为权威，按“拆分成功则真实股数、拆分失败则现金等价”
       处理，保证除权日净值连续；
    2. 卖出时按持股期限结算差别化红利税（≤30天 20%、≤365天 10%、>365天 0%）；
    3. 按父预注册 9.1 实现滑点（10bp + 0.02√参与率，上限1%）与 2% ADV20 参与率上限，
       以及复牌日滑点加倍。
* 会计：主口径 `shares`（真实股数现金账）；审计口径 `tr`（总回报等价账，股数不变、
  全部调整记现金、零税）。两者都应复现冻结层 `factor` 的总回报。
* 时序：t 收盘信号/订单 → t+1 开盘成交；买单 day 单失败不追；卖单 GTC 顺延 ≤10 日。
* 时间纪律：本文件任何收益路径都在 2008-2020；加载 2021+ 直接抛错。

用法
----
    python3 02_代码/统一回测.py --mode selfcheck
    python3 02_代码/统一回测.py --mode smoke --window all
    python3 02_代码/统一回测.py --mode smoke --window dev --sizes
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from 账户撮合 import Account, AccountConfig, Order, SymbolInfo, canonical_code  # noqa: E402
import 费用与交易约束 as fee  # noqa: E402
from 统一数据适配 import (  # noqa: E402
    FrozenPanel, EventBook, WINDOWS, DEV, VAL, FREEZE, HOLDOUT_START,
    MIN_ADV20, MIN_LIST_CAL_DAYS, MIN_LIST_TD, MIN_PRICE, DELIST_BLOCK_TD,
)
from 统计验证 import (  # noqa: E402
    BacktestResult, compute_metrics, annual_table, monthly_matrix, block_bootstrap,
)

OUT = ROOT / "03_结果" / "统一引擎"
EVID_DIR = ROOT / "04_证据"
EVID = EVID_DIR / "统一引擎自检.json"

TRADING_DAYS = 243
RUN_START = time.time()


# ---------------------------------------------------------------------------
# 成本与账户
# ---------------------------------------------------------------------------

def make_fee_config(
    model: str = "C-REAL-V1",
    slippage_bps: float = 10.0,
    transfer_par_rate: float = 0.0010,
) -> fee.FeeConfig:
    """父预注册 9.1 的成本口径。

    transfer_par_rate=0.0010 是父文件的保守上界（沪市 2015-08-01 前按面值 0.1%）；
    历史实际为 0.06%（每千股 0.6 元），用 0.0006 只做敏感性，不做主口径。
    """
    stamp = fee.DEFAULT_STAMP_SCHEDULE
    transfer = (
        fee.TransferRule(date(2005, 1, 1), "par_sh", float(transfer_par_rate), "SH"),
        fee.TransferRule(date(2015, 8, 1), "amount", 0.00002, "BOTH"),
        fee.TransferRule(date(2022, 4, 29), "amount", 0.00001, "BOTH"),
    )
    return fee.FeeConfig(
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_schedule=stamp,
        transfer_schedule=transfer,
        slippage_bps=float(slippage_bps),
        max_participation=0.02,
        dividend_tax_rate=0.0,
    )


class UnifiedAccount(Account):
    """在 `Account` 之上补公司行动、红利税、动态滑点与参与率。"""

    def __init__(self, *args, accounting: str = "shares", **kwargs):
        if accounting not in ("shares", "tr"):
            raise ValueError("accounting 必须是 shares / tr")
        super().__init__(*args, **kwargs)
        self.accounting = accounting
        self.event_ledger: list[dict] = []
        self.tax_ledger: list[dict] = []
        self.ca_rows: list[dict] = []
        self.round_trips: list[dict] = []
        self.hold_meta: dict[str, dict] = {}
        self._ca_row: np.ndarray | None = None
        self._code_pos: dict[str, int] | None = None
        self._adv: dict[str, float] = {}
        self._resume_today: set[str] = set()
        self._force_counts: dict[str, int] = {}
        self._forced_ages: list[tuple[str, int]] = []

    # -- 公司行动 ---------------------------------------------------------
    def _ca_factor_of(self, code: str) -> float:
        if self._ca_row is None or self._code_pos is None:
            return 1.0
        j = self._code_pos.get(code)
        if j is None:
            return 1.0
        return float(self._ca_row[j])

    def _apply_events(self, d, events, prev_map):
        """以 preclose 为权威应用除权；返回 {} 让限价直接用交易所 preclose。"""
        for ev in events:
            if ev.ex_date != d:
                continue
            pos = self.positions.get(ev.symbol)
            if not pos or pos.shares <= 0:
                continue
            s0 = int(pos.shares)
            prev, pre = float(ev.prev_close), float(ev.preclose)
            # 送转/配股必须改股数；"股数不变、全部记现金"在送转日之后会失效，
            # 因此 tr 只作为“shares + 零红利税”的敏感性，不改变公司行动处理。
            bonus = int(math.floor(s0 * ev.bonus_ratio + 1e-9))
            rights_sh = int(math.floor(s0 * ev.rights_ratio + 1e-9))
            rights_cost = rights_sh * float(ev.rights_price)
            new_shares = bonus + rights_sh
            # 配股款已经体现在 preclose 的下调里：credit = S0*prev - S1*pre，
            # 对纯配股为负（=配股款），不能再额外扣一次 rights_cost。
            credit = s0 * prev - (s0 + new_shares) * pre
            status = ev.status
            nav_loss = 0.0
            if rights_sh > 0 and self.cash_total + credit < -1e-6:
                # 现金不足，放弃配股：只收现金分红，不承担配股，净值按稀释真实下降
                credit = s0 * max(0.0, float(ev.dividend))
                new_shares = bonus
                status = ev.status + "+rights_skipped"
                nav_loss = s0 * (prev - pre) - credit
            residual = s0 * prev - ((s0 + new_shares) * pre + credit)
            div_gross = s0 * max(0.0, float(ev.dividend))

            self.cash_total += credit
            self.cash_available += credit
            if new_shares:
                pos.add_free(new_shares)
                pos.available += new_shares
            meta = self.hold_meta.get(ev.symbol)
            if meta is not None and div_gross:
                meta["pending_div"] += div_gross

            row = {
                "date": d.isoformat(),
                "ex_date": ev.ex_date.isoformat(),
                "code": ev.symbol,
                "accounting": self.accounting,
                "status": status,
                "shares_before": s0,
                "shares_after": s0 + new_shares,
                "prev_close": round(prev, 6),
                "preclose": round(pre, 6),
                "adj_ratio": round(float(ev.adj_ratio), 8),
                "credit": round(float(credit), 4),
                "rights_cost": round(float(rights_cost), 4),
                "dividend_gross": round(float(div_gross), 4),
                "nav_loss_vs_continuity": round(float(nav_loss), 4),
                "residual": round(float(residual), 8),
            }
            self.ca_rows.append(row)
            self.event_ledger.append({
                "date": d.isoformat(), "code": ev.symbol, "type": "corporate_action",
                "amount": float(credit), "status": status,
            })
        return {}

    # -- 撮合：动态滑点 + 参与率 ------------------------------------------
    def _fill(self, order, d, bar, base_price, execution, reason):
        old = self.config.fee
        slip = float(old.slippage_bps)
        cap = float(old.max_participation)
        adv = float(self._adv.get(order.code, 0.0))
        if adv > 0 and bar.volume > 0:
            notional = float(base_price) * max(int(order.remaining), 1)
            part = notional / adv
            slip = min(100.0, slip + 0.02 * math.sqrt(max(part, 0.0)) * 10000.0)
            cap = min(cap, 0.02 * adv / float(bar.volume))
        if order.code in self._resume_today:
            slip = min(200.0, slip * 2.0)
        self.config.fee = dataclasses.replace(
            old, slippage_bps=slip, max_participation=max(cap, 1e-9))
        pos = self.positions.get(order.code)
        s0 = int(pos.shares) if pos else 0
        cost0 = float(pos.cost_total) if pos else 0.0
        try:
            ok = super()._fill(order, d, bar, base_price, execution, reason)
        finally:
            self.config.fee = old
        if not ok:
            return False
        t = self.trades[-1]
        pos = self.positions.get(order.code)
        if order.side == "buy":
            if s0 == 0 and pos is not None and pos.shares > 0:
                f = self._ca_factor_of(order.code)
                entry_ca = float(t.price) / f if f > 0 else float(t.price)
                self.hold_meta[order.code] = {
                    "entry_date": d,
                    "entry_idx": bisect.bisect_left(self.calendar, d),
                    "entry_price": float(t.price),
                    "entry_ca": entry_ca,
                    "entry_shares": int(t.qty),
                    "cost_basis": float(t.amount) + float(t.fees),
                    "entry_fees": float(t.fees),
                    "pending_div": 0.0,
                    "highest_ca": entry_ca,
                    "stop_ca": float("nan"),
                    "stop_order": None,
                    "stop_price_raw": float("nan"),
                    "exit_order": None,
                    "sell_gross": 0.0,
                    "sell_fees": 0.0,
                    "sell_qty": 0,
                    "tax": 0.0,
                    "exit_reason": "",
                }
        else:
            meta = self.hold_meta.get(order.code)
            if meta is not None:
                meta["sell_gross"] += float(t.amount)
                meta["sell_fees"] += float(t.fees)
                meta["sell_qty"] += int(t.qty)
                meta["exit_reason"] = t.reason
            if pos is not None and pos.shares == 0 and order.code in self.hold_meta:
                tax = self._settle_tax(order.code, d)
                self._finalize_round_trip(order.code, d, tax)
        return True

    def _force_sell(self, code, d, price, reason):
        pos = self.positions.get(code)
        s0 = int(pos.shares) if pos else 0
        super()._force_sell(code, d, price, reason)
        pos = self.positions.get(code)
        if s0 > 0 and pos is not None and pos.shares == 0 and code in self.hold_meta:
            t = self.trades[-1]
            meta = self.hold_meta[code]
            meta["sell_gross"] += float(t.amount)
            meta["sell_fees"] += float(t.fees)
            meta["sell_qty"] += int(t.qty)
            meta["exit_reason"] = reason
            tax = self._settle_tax(code, d)
            self._finalize_round_trip(code, d, tax)
            self._force_counts[reason] = self._force_counts.get(reason, 0) + 1

    # -- 红利税与平仓记录 --------------------------------------------------
    def _settle_tax(self, code: str, d) -> float:
        if self.accounting == "tr":
            return 0.0
        meta = self.hold_meta.get(code)
        if not meta or meta.get("pending_div", 0.0) <= 0:
            return 0.0
        days = (d - meta["entry_date"]).days
        rate = 0.20 if days <= 30 else (0.10 if days <= 365 else 0.0)
        tax = float(meta["pending_div"]) * rate
        if tax:
            self.cash_total -= tax
            self.cash_available -= tax
            self.tax_ledger.append({
                "date": d.isoformat(), "code": code, "type": "dividend_tax",
                "amount": -tax, "hold_days": days, "rate": rate,
                "dividend_gross": float(meta["pending_div"]),
            })
        meta["tax"] = float(meta.get("tax", 0.0)) + tax
        return tax

    def _finalize_round_trip(self, code: str, d, tax: float) -> None:
        meta = self.hold_meta.pop(code)
        pnl = (meta["sell_gross"] - meta["sell_fees"] - meta["cost_basis"]
               + meta["pending_div"] - tax)
        cost = float(meta["cost_basis"])
        self.round_trips.append({
            "code": code,
            "entry_date": meta["entry_date"].isoformat(),
            "entry_price": round(float(meta["entry_price"]), 4),
            "exit_date": d.isoformat(),
            "exit_price": round(meta["sell_gross"] / meta["sell_qty"], 4) if meta["sell_qty"] else np.nan,
            "shares": int(meta["entry_shares"]),
            "sell_qty": int(meta["sell_qty"]),
            "pnl": round(float(pnl), 4),
            "ret": float(pnl / cost) if cost > 0 else np.nan,
            "fees": round(float(meta["entry_fees"] + meta["sell_fees"]), 4),
            "dividends": round(float(meta["pending_div"]), 4),
            "tax": round(float(tax), 4),
            "hold_days": (d - meta["entry_date"]).days,
            "reason": meta.get("exit_reason", ""),
            "accounting": self.accounting,
        })

    # -- 性能：每日裁掉已结束的持仓/挂单 -----------------------------------
    def on_day(self, d, bars=None, events=None, st_codes=None, is_trading_day=True):
        rec = super().on_day(d, bars=bars, events=events, st_codes=st_codes,
                             is_trading_day=is_trading_day)
        # 持仓已平的卖单必须撤掉，否则会成为永远 active 的孤儿单，
        # 既污染顺延统计，也会让后续的“同票不再买”规则误判。
        for o in self._orders:
            if o.active and o.side == "sell":
                pos = self.positions.get(o.code)
                if pos is None or pos.shares <= 0:
                    o.status = "cancelled"
                    o.reason = "position_closed"
        self._orders = [o for o in self._orders if o.active]
        for code in [c for c, p in self.positions.items() if p.shares <= 0]:
            del self.positions[code]
        return rec


# ---------------------------------------------------------------------------
# 运行配置与结果
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    variant: str
    window: str
    accounting: str = "shares"
    initial_cash: float = 1_000_000.0
    slots: int = 10
    max_new_per_day: int = 2
    max_hold: int = 20
    atr_mult: float = 3.0
    risk_per_trade: float = 0.012
    slippage_bps: float = 10.0
    transfer_par_rate: float = 0.0010
    delist_haircut: float = 0.0
    max_persist_days: int = 10
    rank: str = "mom"          # mom = 20日动量降序；adv = 仅 ADV20 降序（消融）
    label: str = ""


@dataclass
class RunResult:
    cfg: RunConfig
    equity: pd.DataFrame
    fills: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    failed: pd.DataFrame
    orders: pd.DataFrame
    ca: pd.DataFrame
    tax: pd.DataFrame
    metrics: dict
    annual: pd.DataFrame
    monthly: pd.DataFrame
    checks: dict
    benchmark: pd.Series
    meta: dict = field(default_factory=dict)


def _json_safe(x):
    if isinstance(x, (date, datetime, pd.Timestamp)):
        return str(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def load_index_series(start: str, end: str, symbol: str = "000300.SH") -> pd.Series:
    import duckdb
    con = duckdb.connect(str(ROOT / "01_数据" / "frozen" / "frozen.duckdb"), read_only=True)
    try:
        df = con.execute(
            "SELECT date, close FROM index_daily WHERE symbol=? AND date BETWEEN ? AND ? ORDER BY date",
            [symbol, start, end]).df()
    finally:
        con.close()
    s = pd.Series(df["close"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(df["date"])), name=symbol)
    return s


def drawdown(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    return nav / peak - 1.0


# ---------------------------------------------------------------------------
# 账户构造与逐日循环
# ---------------------------------------------------------------------------

def _make_account(panel: FrozenPanel, cfg: RunConfig) -> UnifiedAccount:
    info = {
        code: SymbolInfo(code=code, name="", ipo_date=panel.ipo_date[j],
                         out_date=panel.delist_date[j])
        for j, code in enumerate(panel.codes)
    }
    fee_cfg = make_fee_config(
        slippage_bps=cfg.slippage_bps, transfer_par_rate=cfg.transfer_par_rate)
    acfg = AccountConfig(
        initial_cash=float(cfg.initial_cash),
        fee=fee_cfg,
        cash_settle_days=0,              # A股卖出资金当日可用于买入
        enforce_t1=True,
        allow_buy_downsize=True,
        normalize_buy_orders=True,
        rights_policy="subscribe",
        dividend_tax_rate=0.0,           # 由 UnifiedAccount 按持股期限结算
        delisting_missing_days=10_000_000,  # 不用“连续无 bar”触发；只认 out_date
        delisting_haircut=float(cfg.delist_haircut),
        liquidate_on_finalize=False,
        finalize_haircut=0.0,
        strict=False,
    )
    return UnifiedAccount(
        acfg, panel.dates, freeze_date=panel.dates[-1],
        symbol_info=info, accounting=cfg.accounting)


def _rank(cand: np.ndarray, mom: np.ndarray, adv: np.ndarray) -> np.ndarray:
    js = np.flatnonzero(cand)
    if not len(js):
        return js
    score = np.where(np.isfinite(mom[js]), mom[js], -np.inf)
    liq = np.where(np.isfinite(adv[js]), adv[js], -np.inf)
    order = np.lexsort((js, -liq, -score))
    return js[order]


def _size_shares(cfg: RunConfig, nav: float, cash: float, price: float,
                 atr_raw: float) -> int:
    if not (np.isfinite(price) and price > 0 and nav > 0):
        return 0
    budget = nav / max(cfg.slots, 1)
    shares = int(budget / price / 100) * 100
    if cfg.variant == "ATR3" and np.isfinite(atr_raw) and atr_raw > 0:
        risk_budget = cfg.risk_per_trade * nav
        risk_shares = int(risk_budget / (cfg.atr_mult * atr_raw) / 100) * 100
        shares = min(shares, risk_shares)
    cash_shares = int(max(cash, 0.0) * 0.98 / (price * 1.01) / 100) * 100
    return int(min(shares, cash_shares))


def _cancel(acct: UnifiedAccount, oid: str | None, reason: str) -> None:
    if oid:
        acct.cancel(oid, reason)


def run_variant(cfg: RunConfig, panel: FrozenPanel, events: EventBook,
                collect_positions: bool = True) -> RunResult:
    start, end = WINDOWS[cfg.window]
    i0, i1 = panel.window_index(start, end)
    if i0 == 0:
        raise ValueError("窗口首日没有前一日，无法生成首个信号")
    acct = _make_account(panel, cfg)
    acct._code_pos = panel.code_idx

    # 日频检查
    checks = {
        "cash_resid_max": 0.0, "nav_resid_max": 0.0,
        "cash_resid_bad_days": 0, "nav_resid_bad_days": 0,
        "t1_violations": 0, "negative_share_events": 0,
        "trades_on_suspension": 0, "buy_at_limit_up_open": 0,
        "sell_at_limit_down_open": 0, "limit_examples": [],
        "persist_max_days": 0, "persist_violations": 0,
        "forced_count": 0, "delist_forced": 0,
        "max_date": "", "n_rows_after_2020": 0,
        "n_orders": 0, "n_orders_filled": 0,
    }
    pos_rows: list[dict] = []
    trade_ptr = 0
    ca_ptr = 0
    tax_ptr = 0
    prev_cash = float(acct.cash_total)
    prev_date = None
    n_bad_cash = 0
    n_bad_nav = 0

    def _bar_close(i: int, j: int, pos) -> float:
        px = float(panel.mat["close"][i, j])
        if not np.isfinite(px) or px <= 0:
            px = float(pos.last_price) if pos.last_price else 0.0
        return px

    for i in range(i0 - 1, i1 + 1):
        d = panel.dates[i]
        if d >= date(2021, 1, 1):
            raise RuntimeError("试图运行 2021+")
        held = list(acct.positions.keys())
        active = [o.code for o in acct._orders if o.active]
        ev_today = events.on(d)
        needed = set(held) | set(active) | {e.symbol for e in ev_today}
        acct._ca_row = panel.ca_factor[i]
        acct._adv = {
            c: float(panel.adv20[i - 1, panel.code_idx[c]]) if i > 0 and c in panel.code_idx else 0.0
            for c in set(active) | set(held)
        }
        acct._resume_today = {panel.codes[j] for j in np.flatnonzero(panel.resume[i])}

        rec = acct.on_day(d, bars=panel.bars_on(i, needed), events=ev_today)

        # 卖出顺延上限：第 10 个交易日收盘强制退出（在当日记录之后执行，再重记）
        forced_now = False
        for o in list(acct._orders):
            if not (o.active and o.side == "sell" and o.filled_qty == 0):
                continue
            try:
                age = i - panel.day_idx[o.execute_from]
            except KeyError:
                continue
            if age >= cfg.max_persist_days:
                pos = acct.positions.get(o.code)
                j = panel.code_idx.get(o.code)
                if pos is None or j is None:
                    continue
                px = _bar_close(i, j, pos)
                acct._force_sell(o.code, d, px, "sell_persistence_10d")
                acct._forced_ages.append((o.order_id, int(age)))
                forced_now = True
        if forced_now:
            rec = acct._mark_and_record(d)

        # 逐日现金流恒等
        new_fills = acct.trades[trade_ptr:]
        buys = sum(t.amount + t.fees for t in new_fills if t.side == "buy")
        sells = sum(t.amount - t.fees for t in new_fills if t.side == "sell")
        new_ca = acct.ca_rows[ca_ptr:]
        ca_cash = sum(x["credit"] for x in new_ca)
        new_tax = acct.tax_ledger[tax_ptr:]
        tax_cash = sum(x["amount"] for x in new_tax)
        flows = sells - buys + ca_cash + tax_cash
        delta = float(rec["cash_total"]) - prev_cash
        cash_resid = abs(delta - flows)
        checks["cash_resid_max"] = max(checks["cash_resid_max"], cash_resid)
        if cash_resid > 0.02:
            n_bad_cash += 1
        prev_cash = float(rec["cash_total"])

        # 逐日市值恒等
        mv = 0.0
        for code, pos in acct.positions.items():
            j = panel.code_idx.get(code)
            if j is None:
                continue
            mv += pos.shares * _bar_close(i, j, pos)
        nav_resid = abs(mv - float(rec["market_value"]))
        checks["nav_resid_max"] = max(checks["nav_resid_max"], nav_resid)
        if nav_resid > 0.02:
            n_bad_nav += 1

        checks["max_date"] = max(checks["max_date"], d.isoformat())
        if i >= i0 and collect_positions:
            nav = float(rec["nav"])
            for code, pos in acct.positions.items():
                j = panel.code_idx.get(code)
                if j is None:
                    continue
                px = _bar_close(i, j, pos)
                meta = acct.hold_meta.get(code, {})
                stop_raw = np.nan
                if meta and np.isfinite(meta.get("stop_ca", np.nan)):
                    f = float(panel.ca_factor[i, j])
                    stop_raw = meta["stop_ca"] * f if f > 0 else np.nan
                pos_rows.append({
                    "date": d.isoformat(), "code": code, "shares": int(pos.shares),
                    "close": round(px, 4), "market_value": round(pos.shares * px, 2),
                    "weight": round(pos.shares * px / nav, 6) if nav > 0 else np.nan,
                    "entry_date": meta.get("entry_date", "").isoformat() if meta.get("entry_date") else "",
                    "stop_price": round(stop_raw, 4) if np.isfinite(stop_raw) else np.nan,
                })

        trade_ptr = len(acct.trades)
        ca_ptr = len(acct.ca_rows)
        tax_ptr = len(acct.tax_ledger)
        prev_date = d

        if i >= i1:
            break

        # ---------------- t 收盘决策（t+1 开盘执行） ----------------
        nav = float(rec["nav"])

        if cfg.variant == "BH10":
            if i == i0 - 1:
                js = np.flatnonzero(panel.elig[i])
                for j in js[: cfg.slots]:
                    code = panel.codes[j]
                    price = float(panel.mat["close"][i, j])
                    shares = _size_shares(cfg, nav, float(acct.cash_available), price, np.nan)
                    if shares >= 100:
                        acct.submit(Order(code=code, side="buy", qty=shares,
                                          kind="open", tif="day", tag="bh10"), d)
            continue

        # 1) 退出维护
        for code, meta in list(acct.hold_meta.items()):
            pos = acct.positions.get(code)
            if not pos or pos.shares <= 0:
                continue
            j = panel.code_idx.get(code)
            if j is None:
                continue
            if cfg.variant == "FIX20":
                cur = acct.orders.get(meta.get("exit_order") or "")
                if (meta.get("exit_order") is None or not (cur and cur.active)) \
                        and i - meta["entry_idx"] >= cfg.max_hold - 1:
                    _cancel(acct, meta.get("exit_order"), "fix20_replace")
                    o = acct.submit(Order(code=code, side="sell", qty=pos.shares,
                                          kind="open", tif="gtc", tag="fix20"), d)
                    meta["exit_order"] = o.order_id
            elif cfg.variant == "ATR3":
                ca_high = float(panel.mat["ca_high"][i, j])
                atr = float(panel.atr14[i, j])
                if np.isfinite(ca_high) and np.isfinite(atr) and atr > 0:
                    meta["highest_ca"] = max(float(meta["highest_ca"]), ca_high)
                    prev_stop = meta.get("stop_ca", float("nan"))
                    stop = float(meta["highest_ca"]) - cfg.atr_mult * atr
                    if np.isfinite(prev_stop):
                        stop = max(stop, float(prev_stop))
                    meta["stop_ca"] = stop
                    f_next = float(panel.ca_factor[i, j])
                    for ev in events.on(panel.dates[i + 1]):
                        if ev.symbol == code:
                            f_next *= float(ev.adj_ratio)
                    stop_raw = stop * f_next if f_next > 0 else stop
                    close_raw = float(panel.mat["close"][i, j])
                    cur = acct.orders.get(meta.get("stop_order") or "")
                    if stop_raw >= close_raw - 1e-9:
                        # 止损已在收盘之上：次日开盘直接离场
                        _cancel(acct, meta.get("stop_order"), "stop_above_close")
                        meta["stop_order"] = None
                        if meta.get("exit_order") is None or not (acct.orders.get(meta["exit_order"]) and acct.orders[meta["exit_order"]].active):
                            o = acct.submit(Order(code=code, side="sell", qty=pos.shares,
                                                  kind="open", tif="gtc", tag="atr_break"), d)
                            meta["exit_order"] = o.order_id
                    else:
                        need = (cur is None or not cur.active
                                or abs(stop_raw - float(meta.get("stop_price_raw", np.nan))) > 0.005
                                or int(meta.get("stop_qty", -1)) != int(pos.shares))
                        if need:
                            _cancel(acct, meta.get("stop_order"), "stop_reprice")
                            price = round(max(stop_raw, 0.01), 2)
                            o = acct.submit(Order(code=code, side="sell", qty=pos.shares,
                                                  kind="stop", price=price, tif="gtc",
                                                  tag="atr_trail"), d)
                            meta["stop_order"] = o.order_id
                            meta["stop_price_raw"] = stop_raw
                            meta["stop_qty"] = int(pos.shares)

        # 2) 入场
        open_buys = {o.code for o in acct._orders if o.active and o.side == "buy"}
        n_hold = sum(1 for p in acct.positions.values() if p.shares > 0)
        free = cfg.slots - n_hold - len(open_buys)
        if free > 0:
            held = set(acct.positions)
            cand = panel.signal[i].copy()
            for code in held:
                j = panel.code_idx.get(code)
                if j is not None:
                    cand[j] = False
            for code in open_buys:
                j = panel.code_idx.get(code)
                if j is not None:
                    cand[j] = False
            if cfg.rank == "adv":
                ranked = _rank(cand, np.zeros(panel.n_codes, dtype=np.float32), panel.adv20[i])
            else:
                ranked = _rank(cand, panel.mom20[i], panel.adv20[i])
            n_new = 0
            cash_est = float(acct.cash_available)
            for j in ranked:
                if n_new >= min(free, cfg.max_new_per_day):
                    break
                code = panel.codes[j]
                # 有未完成卖单的票不再买
                if any(o.active and o.side == "sell" and o.code == code for o in acct._orders):
                    continue
                price = float(panel.mat["close"][i, j])
                shares = _size_shares(cfg, nav, cash_est, price, float(panel.atr_raw[i, j]))
                if shares < 100:
                    continue
                acct.submit(Order(code=code, side="buy", qty=shares, kind="open",
                                  tif="day", tag="donch60"), d)
                cash_est -= shares * price * 1.01
                n_new += 1

    # 收尾：取消未成交挂单，不再估值（已逐日记录）
    acct.finalize(panel.dates[i1])

    # ---------------- 组装输出 ----------------
    eq = pd.DataFrame(acct.daily).rename(columns={"cash_total": "cash"})
    eq = eq[(eq["date"] >= start) & (eq["date"] <= end)].copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date").sort_index()

    fills = pd.DataFrame([t.as_dict() for t in acct.trades])
    if len(fills):
        fills["date"] = pd.to_datetime(fills["date"])
        by_day = fills.groupby("date")
        eq["turnover"] = by_day["amount"].sum().reindex(eq.index).fillna(0.0)
        eq["cost"] = by_day["fees"].sum().reindex(eq.index).fillna(0.0)
    else:
        eq["turnover"] = 0.0
        eq["cost"] = 0.0

    trades = pd.DataFrame(acct.round_trips)
    if len(trades):
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])
        trades = trades.sort_values("exit_date").reset_index(drop=True)
    pos_df = pd.DataFrame(pos_rows)
    orders_df = pd.DataFrame([o.as_dict() for o in acct.orders.values()])
    if len(orders_df):
        orders_df["decision_date"] = pd.to_datetime(orders_df["decision_date"])
        orders_df["execute_from"] = pd.to_datetime(orders_df["execute_from"])
    failed = _failed_orders(orders_df, acct, panel)
    ca_df = pd.DataFrame(acct.ca_rows)
    tax_df = pd.DataFrame(acct.tax_ledger)

    bench = panel.benchmark_ew(i0, i1)
    bench_nav = (1.0 + bench).cumprod()
    bench_nav.index = pd.to_datetime(bench_nav.index)
    bench_nav = bench_nav.reindex(eq.index).ffill()

    res = BacktestResult(eq[["nav", "market_value", "cash", "n_positions", "turnover", "cost"]],
                         trades, meta={"initial_cash": cfg.initial_cash})
    metrics = compute_metrics(res, initial_cash=cfg.initial_cash, benchmark=bench_nav)
    metrics["benchmark_name"] = "B1_EW_MAINBOARD"
    metrics["variant"] = cfg.variant
    metrics["window"] = cfg.window
    metrics["accounting"] = cfg.accounting
    metrics["initial_cash"] = cfg.initial_cash
    try:
        boot = block_bootstrap(res.returns, n_boot=2000, method="stationary",
                               seed=20261026, dates=eq.index)
        metrics["bootstrap"] = boot
    except Exception as exc:
        metrics["bootstrap"] = {"error": str(exc)}
    ann = annual_table(res, initial_cash=cfg.initial_cash)
    mon = monthly_matrix(res, initial_cash=cfg.initial_cash)

    checks.update(_trade_level_checks(acct, panel, i0, i1, cfg))
    checks["forced_count"] = int(sum(acct._force_counts.values()))
    checks["delist_forced"] = int(acct._force_counts.get("delisting_out_date", 0))
    checks["n_orders"] = int(len(acct.orders))
    checks["n_orders_filled"] = int(sum(1 for o in acct.orders.values() if o.status == "filled"))
    checks["cash_resid_bad_days"] = int(n_bad_cash)
    checks["nav_resid_bad_days"] = int(n_bad_nav)
    checks["n_rows_after_2020"] = int((eq.index >= pd.Timestamp("2021-01-01")).sum())
    checks["max_date"] = str(eq.index.max().date()) if len(eq) else ""
    checks["gates"] = _gate_summary(checks)

    meta = {
        "variant": cfg.variant, "window": cfg.window, "accounting": cfg.accounting,
        "start": start, "end": end, "initial_cash": cfg.initial_cash,
        "slots": cfg.slots, "max_new_per_day": cfg.max_new_per_day,
        "max_hold": cfg.max_hold, "atr_mult": cfg.atr_mult,
        "slippage_bps": cfg.slippage_bps, "transfer_par_rate": cfg.transfer_par_rate,
        "delist_haircut": cfg.delist_haircut, "n_fills": int(len(fills)),
        "n_round_trips": int(len(trades)), "n_ca_events": int(len(ca_df)),
        "n_tax_events": int(len(tax_df)),
        "force_counts": dict(acct._force_counts),
        "bad_bars": panel.n_bad_bars,
    }
    return RunResult(cfg, eq, fills, trades, pos_df, failed, orders_df, ca_df, tax_df,
                     metrics, ann, mon, checks, bench_nav, meta)


def _failed_orders(orders: pd.DataFrame, acct: UnifiedAccount, panel: FrozenPanel) -> pd.DataFrame:
    """真正失败的订单：排除策略主动撤单与收盘撤单。"""
    if not len(orders):
        return orders
    ignore = {"manual_cancel", "stop_reprice", "fix20_replace", "stop_above_close",
              "finalize_cancel", "position_closed"}
    bad = orders[(orders["status"].isin(["cancelled", "rejected", "partial"]))
                 & (~orders["reason"].isin(ignore))].copy()
    return bad.reset_index(drop=True)


def _trade_level_checks(acct: UnifiedAccount, panel: FrozenPanel, i0: int, i1: int,
                        cfg: RunConfig) -> dict:
    out = {
        "t1_violations": 0, "negative_share_events": 0,
        "trades_on_suspension": 0, "buy_at_limit_up_open": 0,
        "sell_at_limit_down_open": 0, "limit_examples": [],
        "persist_max_days": 0, "persist_violations": 0,
        "ca_resid_max": 0.0, "ca_skipped_rights": 0, "ca_rows": 0,
        "n_trades": 0,
    }
    trades = acct.trades
    # 公司行动带来的送转股（trades 不含），按日期并入股数流
    ca_flow: dict[tuple, int] = {}
    for r in acct.ca_rows:
        delta = int(r.get("shares_after", 0)) - int(r.get("shares_before", 0))
        if delta:
            ca_flow[(r["date"], r["code"])] = ca_flow.get((r["date"], r["code"]), 0) + delta

    # 逐票 T+1 / 负股数
    cur: dict[str, dict] = {}
    ca_keys = sorted(ca_flow)
    ci = 0
    for t in trades:
        while ci < len(ca_keys) and ca_keys[ci][0] <= t.date.isoformat():
            dd, cc = ca_keys[ci]
            st = cur.setdefault(cc, {"shares": 0, "buy_date": None})
            st["shares"] += ca_flow[(dd, cc)]
            ci += 1
        st = cur.setdefault(t.code, {"shares": 0, "buy_date": None})
        if t.side == "buy":
            if st["shares"] == 0:
                st["buy_date"] = t.date
            st["shares"] += t.qty
        else:
            if st["buy_date"] is not None and t.date <= st["buy_date"]:
                out["t1_violations"] += 1
            st["shares"] -= t.qty
            if st["shares"] < 0:
                out["negative_share_events"] += 1
            if st["shares"] == 0:
                st["buy_date"] = None
    out["n_trades"] = len(trades)
    # 停牌/涨跌停
    for t in trades:
        i = panel.day_idx.get(t.date)
        j = panel.code_idx.get(t.code)
        if i is None or j is None:
            continue
        forced = str(t.reason).startswith(("sell_persistence", "delisting", "end_of_period"))
        if forced:
            out["forced_trades"] = out.get("forced_trades", 0) + 1
            continue
        if panel.suspended[i, j]:
            out["trades_on_suspension"] += 1
        if t.side == "buy" and panel.mat["is_limit_up_open"][i, j]:
            out["buy_at_limit_up_open"] += 1
            if len(out["limit_examples"]) < 10:
                out["limit_examples"].append({"date": t.date.isoformat(), "code": t.code,
                                              "side": t.side, "reason": "limit_up_open"})
        if t.side == "sell" and panel.mat["is_limit_down_open"][i, j]:
            out["sell_at_limit_down_open"] += 1
            if len(out["limit_examples"]) < 10:
                out["limit_examples"].append({"date": t.date.isoformat(), "code": t.code,
                                              "side": t.side, "reason": "limit_down_open"})
    # 订单顺延
    fill_day = {}
    for t in trades:
        fill_day[t.order_id] = panel.day_idx.get(t.date)
    forced_age = dict(acct._forced_ages)
    for o in acct.orders.values():
        if o.side != "sell":
            continue
        if o.status == "cancelled" and "sell_persistence" not in str(o.reason):
            continue
        exec_i = panel.day_idx.get(o.execute_from)
        if exec_i is None:
            continue
        if o.order_id in forced_age:
            age = int(forced_age[o.order_id])
        else:
            end_i = fill_day.get(o.order_id)
            if end_i is None:
                end_i = i1
            age = max(0, int(end_i) - int(exec_i))
        out["persist_max_days"] = max(out["persist_max_days"], age)
        if age > cfg.max_persist_days:
            out["persist_violations"] += 1
    # 公司行动连续性
    for r in acct.ca_rows:
        out["ca_rows"] += 1
        if "rights_skipped" in str(r.get("status", "")):
            out["ca_skipped_rights"] += 1
            continue
        out["ca_resid_max"] = max(out["ca_resid_max"], abs(float(r.get("residual", 0.0))))
    return out


def prev_date_guess(order) -> date:
    return order.execute_from or date(2000, 1, 1)


def _gate_summary(c: dict) -> dict:
    g = {
        "G-E1_cash": c["cash_resid_max"] <= 0.02 and c["cash_resid_bad_days"] == 0,
        "G-E2_nav": c["nav_resid_max"] <= 0.02 and c["nav_resid_bad_days"] == 0,
        "G-E3_t1": c["t1_violations"] == 0 and c["negative_share_events"] == 0,
        "G-E4_suspension": c["trades_on_suspension"] == 0,
        "G-E5_limit": c["buy_at_limit_up_open"] == 0 and c["sell_at_limit_down_open"] == 0,
        "G-E6_ca": c["ca_resid_max"] <= 1e-6,
        "G-E9_persist": c["persist_violations"] == 0,
        "G-E10_time": c["n_rows_after_2020"] == 0 and c["max_date"] <= "2020-12-31",
    }
    g["all_pass"] = all(g.values())
    return g


# ---------------------------------------------------------------------------
# 买入持有夹具（公司行动等价性）
# ---------------------------------------------------------------------------

def run_hold_codes(panel: FrozenPanel, events: EventBook, codes: Sequence[str],
                   accounting: str = "shares", initial_cash: float = 1_000_000.0,
                   start: str = "2008-01-02", end: str = "2020-12-31") -> dict:
    """指定代码等权买入并持有，用于公司行动/记账口径审计（不产生策略结论）。"""
    i0, i1 = panel.window_index(start, end)
    dec = i0 - 1
    cfg = RunConfig(variant="HOLD", window="dev", accounting=accounting,
                    initial_cash=initial_cash, slots=len(codes))
    acct = _make_account(panel, cfg)
    acct._code_pos = panel.code_idx
    per = initial_cash / max(len(codes), 1)
    submitted = []
    for code in codes:
        j = panel.code_idx.get(code)
        if j is None or not panel.listed[dec, j] or panel.suspended[dec, j]:
            continue
        px = float(panel.mat["close"][dec, j])
        if not np.isfinite(px) or px <= 0:
            continue
        qty = int(per / (px * 1.01) / 100) * 100
        if qty < 100:
            continue
        acct.submit(Order(code=code, side="buy", qty=qty, kind="open", tif="day",
                          tag="hold"), panel.dates[dec])
        submitted.append(code)
    for i in range(i0, i1 + 1):
        d = panel.dates[i]
        held = list(acct.positions.keys())
        active = [o.code for o in acct._orders if o.active]
        needed = set(held) | set(active) | {e.symbol for e in events.on(d)}
        acct._ca_row = panel.ca_factor[i]
        acct._adv = {c: 0.0 for c in set(active) | set(held)}
        acct._resume_today = {panel.codes[j] for j in np.flatnonzero(panel.resume[i])}
        acct.on_day(d, bars=panel.bars_on(i, needed), events=events.on(d))
    acct.finalize(panel.dates[i1])

    # 每只票：实际（持仓+已收现金调整）vs 总回报指数（把现金调整按除权日收盘再投资）
    by_code: dict[str, list[dict]] = {}
    for r in acct.ca_rows:
        by_code.setdefault(r["code"], []).append(r)
    rows = []
    entry_trades = {}
    for t in acct.trades:
        if t.side == "buy" and t.code not in entry_trades:
            entry_trades[t.code] = t
    for code, t in entry_trades.items():
        j = panel.code_idx[code]
        pos = acct.positions.get(code)
        if pos is None:
            # 退市强平：不纳入等价性（单独报告）
            rows.append({"code": code, "status": "closed_by_force"})
            continue
        meta = acct.hold_meta.get(code, {})
        qty = int(t.qty)
        open_entry = float(panel.mat["open"][i0, j])
        hfq_entry = float(panel.tr_index[i0, j])
        hfq_end = float(panel.tr_index[i1, j])
        close_end = float(panel.mat["close"][i1, j])
        close_entry = float(panel.mat["close"][i0, j])
        tr = hfq_end / hfq_entry if hfq_entry > 0 else np.nan
        # 入场在 i0 开盘，i0 收盘后的总回报才是 factor 的后续；用 close_entry 作起点
        expected_reinv = qty * close_entry * tr
        credits = by_code.get(code, [])
        credit_nominal = sum(float(x["credit"]) for x in credits)
        credit_reinv = 0.0
        missing = 0
        for x in credits:
            ei = panel.day_idx.get(date.fromisoformat(x["date"]))
            if ei is None:
                missing += 1
                continue
            h0 = float(panel.tr_index[i1, j])
            h1 = float(panel.tr_index[ei, j])
            credit_reinv += float(x["credit"]) * (h0 / h1) if h1 > 0 else float(x["credit"])
        actual_position = pos.shares * close_end
        actual_nominal = actual_position + credit_nominal
        actual_reinv = actual_position + credit_reinv
        err_reinv_bp = (actual_reinv / expected_reinv - 1.0) * 1e4 if expected_reinv else np.nan
        err_nominal_bp = (actual_nominal / expected_reinv - 1.0) * 1e4 if expected_reinv else np.nan
        # 独立拆分校验：持仓价格回报（含送转、不含分红）vs Baostock hfq 价格回报
        bs0 = float(panel.bs_hfq[i0, j])
        bs1 = float(panel.bs_hfq[i1, j])
        price_ret_acct = (pos.shares * close_end) / (qty * close_entry) if qty and close_entry > 0 else np.nan
        price_ret_bs = bs1 / bs0 if np.isfinite(bs0) and np.isfinite(bs1) and bs0 > 0 else np.nan
        err_price_bs_bp = ((price_ret_acct / price_ret_bs) - 1.0) * 1e4 if np.isfinite(price_ret_bs) else np.nan
        rows.append({
            "code": code, "status": "ok", "qty_entry": qty,
            "open_entry": open_entry, "fill_price": float(t.price),
            "qty_end": int(pos.shares), "close_end": close_end,
            "tr_factor": round(float(tr), 8),
            "expected_reinvested": round(float(expected_reinv), 2),
            "actual_position": round(float(actual_position), 2),
            "credit_nominal": round(float(credit_nominal), 2),
            "credit_reinvested": round(float(credit_reinv), 2),
            "pending_div_gross_for_tax": round(float(meta.get("pending_div", 0.0)), 2),
            "actual_nominal": round(float(actual_nominal), 2),
            "actual_reinvested": round(float(actual_reinv), 2),
            "err_reinvested_bp": round(float(err_reinv_bp), 3),
            "err_nominal_bp": round(float(err_nominal_bp), 3),
            "price_ret_account": round(float(price_ret_acct), 8) if np.isfinite(price_ret_acct) else None,
            "price_ret_bs_hfq": round(float(price_ret_bs), 8) if np.isfinite(price_ret_bs) else None,
            "err_price_bs_bp": round(float(err_price_bs_bp), 3) if np.isfinite(err_price_bs_bp) else None,
            "n_ca_events": len(credits), "missing_ca_dates": missing,
        })
    nav = float(acct.daily[-1]["nav"])
    held = set(entry_trades)
    exp = 0
    if len(events.events):
        sub = events.events[events.events["symbol"].isin(held)]
        for code, t in entry_trades.items():
            d0 = t.date.isoformat()
            exp += int(((sub["symbol"] == code)
                        & (sub["ex_date"].astype(str) > d0)
                        & (sub["ex_date"].astype(str) <= end)).sum())
    ca_resid = max([abs(float(x.get("residual", 0.0))) for x in acct.ca_rows
                    if "rights_skipped" not in str(x.get("status", ""))] or [0.0])
    return {
        "accounting": accounting, "nav": nav, "rows": rows,
        "n_submitted": len(submitted), "n_entry_fills": len(entry_trades),
        "force_counts": dict(acct._force_counts),
        "n_ca_rows": len(acct.ca_rows), "n_expected_events": int(exp),
        "ca_resid_max": float(ca_resid),
        "fills_tail": [t.as_dict() for t in acct.trades[-5:]],
    }


def audit_delist_lookahead(panel: FrozenPanel) -> dict:
    """父预注册用未来 out_date 剔除退市整理期，量化被挡信号数与其后 20 日收益。"""
    breakout = (
        panel.listed & ~panel.suspended & ~panel.mat["is_st"]
        & (panel.mat["td_since_list"] >= MIN_LIST_TD)
        & (panel.mat["close"] >= MIN_PRICE)
        & (panel.adv20 >= MIN_ADV20)
        & (panel.days_since_ipo >= MIN_LIST_CAL_DAYS)
        & ~panel.resume
        & (panel.mat["ca_close"] >= panel.rmax60 - 1e-6)
        & (panel.mat["ca_close"] > panel.ma120)
        & np.isfinite(panel.atr14)
    )
    blocked = breakout & panel.delist_window
    fwd = np.full_like(panel.tr_index, np.nan, dtype=np.float32)
    fwd[:-20] = panel.tr_index[20:] / panel.tr_index[:-20] - 1.0
    vals = fwd[blocked]
    vals = vals[np.isfinite(vals)]
    all_vals = fwd[breakout & ~panel.delist_window]
    all_vals = all_vals[np.isfinite(all_vals)]
    return {
        "n_signals_total": int(breakout.sum()),
        "n_blocked_by_delist_window": int(blocked.sum()),
        "blocked_share": round(float(blocked.sum() / max(1, breakout.sum())), 6),
        "blocked_fwd20_mean": round(float(np.mean(vals)), 6) if len(vals) else None,
        "blocked_fwd20_median": round(float(np.median(vals)), 6) if len(vals) else None,
        "allowed_fwd20_mean": round(float(np.mean(all_vals)), 6) if len(all_vals) else None,
        "note": "仅审计父预注册规则的前视偏差，不参与任何选择；20日收益为 tr_index 总回报。",
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def selfcheck(panel: FrozenPanel, events: EventBook, write: bool = True) -> dict:
    t0 = time.time()
    ev = events.events
    codes = []
    if len(ev):
        main = set(panel.codes)
        sub = ev[(ev["symbol"].isin(main)) & (ev["ex_date"].astype(str) >= "2008-01-02")]
        counts = sub.groupby("symbol").size().sort_values(ascending=False)
        for code in counts.index:
            j = panel.code_idx.get(code)
            if j is None:
                continue
            if panel.delist_date[j] is not None:
                continue
            if not (panel.listed[0, j] and panel.listed[-1, j]):
                continue
            if not (panel.listed[panel.index_of("2008-01-02"), j]
                    and ~panel.suspended[panel.index_of("2008-01-02"), j]):
                continue
            codes.append(code)
            if len(codes) >= 20:
                break

    shares = run_hold_codes(panel, events, codes, accounting="shares")
    tr = run_hold_codes(panel, events, codes, accounting="tr")
    srows = {r["code"]: r for r in shares["rows"] if r.get("status") == "ok"}
    trows = {r["code"]: r for r in tr["rows"] if r.get("status") == "ok"}
    common = sorted(set(srows) & set(trows))
    err_s = np.array([abs(srows[c]["err_reinvested_bp"]) for c in common], dtype=float)
    err_t = np.array([abs(trows[c]["err_reinvested_bp"]) for c in common], dtype=float)
    err_nom = np.array([abs(srows[c]["err_nominal_bp"]) for c in common], dtype=float)
    diff_nav = abs(shares["nav"] / tr["nav"] - 1.0) if tr["nav"] else np.nan
    missing_events = int(shares["n_expected_events"] - shares["n_ca_rows"])

    # 独立总回报核对：factor vs Baostock hfq（后者有已知年末重定基，只作披露）
    bs_err = []
    for c in common:
        j = panel.code_idx[c]
        f0, f1 = float(panel.tr_index[0, j]), float(panel.tr_index[-1, j])
        b0, b1 = float(panel.bs_hfq[0, j]), float(panel.bs_hfq[-1, j])
        if all(np.isfinite(x) and x > 0 for x in (f0, f1, b0, b1)):
            bs_err.append(((f1 / f0) / (b1 / b0) - 1.0) * 1e4)
    bs_err = np.array(bs_err, dtype=float)

    checks = {
        "event_book": events.stats,
        "panel": {
            "n_days": panel.n_days, "n_codes": panel.n_codes,
            "n_rows": panel.n_rows_raw, "n_bad_bars": panel.n_bad_bars,
            "bad_bar_examples": panel.bad_bar_examples,
            "max_date": panel.dates[-1].isoformat(),
            "min_date": panel.dates[0].isoformat(),
        },
        "equivalence": {
            "n_codes": len(common),
            "codes": common,
            "shares_median_abs_err_bp": round(float(np.median(err_s)), 4) if len(err_s) else None,
            "shares_max_abs_err_bp": round(float(np.max(err_s)), 4) if len(err_s) else None,
            "tr_median_abs_err_bp": round(float(np.median(err_t)), 4) if len(err_t) else None,
            "tr_max_abs_err_bp": round(float(np.max(err_t)), 4) if len(err_t) else None,
            "shares_median_nominal_err_bp": round(float(np.median(err_nom)), 4) if len(err_nom) else None,
            "shares_vs_tr_nav_rel_diff": round(float(diff_nav), 10),
            "shares_nav": round(float(shares["nav"]), 2),
            "tr_nav": round(float(tr["nav"]), 2),
            "shares_rows": shares["rows"],
            "force_counts_shares": shares["force_counts"],
            "force_counts_tr": tr["force_counts"],
            "n_ca_rows": shares["n_ca_rows"],
            "n_expected_events": shares["n_expected_events"],
            "missing_events": missing_events,
            "ca_resid_max": shares["ca_resid_max"],
            "bs_hfq_vs_factor_median_abs_bp": round(float(np.median(np.abs(bs_err))), 3) if len(bs_err) else None,
            "bs_hfq_vs_factor_max_abs_bp": round(float(np.max(np.abs(bs_err))), 3) if len(bs_err) else None,
            "bs_hfq_note": "Baostock hfq 有已知年末重定基；此列只披露，不作门槛。",
            "equivalence_note": (
                "现金分红账 vs 分红再投资指数。整手取整与分红到账时点会留下不可消除的残差；"
                "公司行动的硬门槛是逐事件连续性 G-E6 与事件数一致。"
            ),
        },
        "delist_lookahead": audit_delist_lookahead(panel),
        "guard_2021": _guard_2021(),
        "gates": {},
    }
    g = {
        "G-E6_ca_continuity": checks["equivalence"]["ca_resid_max"] <= 1e-6,
        "G-E6b_ca_no_missing_events": missing_events == 0,
        "G-E7_equivalence_median_50bp": checks["equivalence"]["shares_median_abs_err_bp"] is not None
        and checks["equivalence"]["shares_median_abs_err_bp"] <= 50.0,
        "G-E7_equivalence_max_300bp": checks["equivalence"]["shares_max_abs_err_bp"] is not None
        and checks["equivalence"]["shares_max_abs_err_bp"] <= 300.0,
        "G-E8_accounting_1pct": np.isfinite(diff_nav) and diff_nav <= 0.01,
        "G-E10_no_2021": panel.dates[-1] < date(2021, 1, 1),
        "adapter_bad_bars_0": panel.n_bad_bars == 0,
        "events_invalid_0": events.stats.get("n_invalid", 0) == 0,
        "G-E10b_2021_guard": checks["guard_2021"]["panel_2021_blocked"]
        and checks["guard_2021"]["event_2021_blocked"],
    }
    g["all_pass"] = all(g.values())
    checks["gates"] = g
    checks["elapsed_s"] = round(time.time() - t0, 2)

    if write:
        EVID_DIR.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        payload = _json_safe({
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "freeze": FREEZE,
                "window": [panel.start, panel.end],
                "purpose": "统一引擎自检；不含 2021+ 收益",
                "code": _code_meta(),
            },
            "checks": checks,
        })
        EVID.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        (OUT / "自检_统一引擎.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return checks


def _guard_2021() -> dict:
    """显式验证 2021+ 被适配层拒绝。"""
    out = {"panel_2021_blocked": False, "event_2021_blocked": False}
    try:
        FrozenPanel("2021-01-01", "2021-12-31")
    except Exception:
        out["panel_2021_blocked"] = True
    try:
        EventBook().load("2021-01-01", "2021-12-31")
    except Exception:
        out["event_2021_blocked"] = True
    return out


def _code_meta() -> dict:
    files = ["02_代码/统一回测.py", "02_代码/统一数据适配.py",
             "02_代码/账户撮合.py", "02_代码/费用与交易约束.py", "02_代码/统计验证.py"]
    out = {}
    for f in files:
        p = ROOT / f
        if p.exists():
            out[f] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                      "lines": sum(1 for _ in p.open("rb"))}
    return out


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def write_run(res: RunResult, hs300: pd.Series | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{res.cfg.variant}_{res.cfg.window}" + ("" if res.cfg.rank == "mom" else f"_{res.cfg.rank}") + ("" if res.cfg.accounting == "shares" else f"_{res.cfg.accounting}")
    if res.cfg.initial_cash != 1_000_000.0:
        tag += f"_{int(res.cfg.initial_cash/1000)}k"
    eq = res.equity.copy()
    eq["bench_b1"] = res.benchmark.reindex(eq.index).ffill()
    if hs300 is not None:
        eq["bench_hs300"] = hs300.reindex(eq.index).ffill()
    eq["drawdown"] = drawdown(eq["nav"])
    eq.to_csv(OUT / f"净值_{tag}.csv", encoding="utf-8-sig")
    if len(res.fills):
        res.fills.to_csv(OUT / f"成交_{tag}.csv", index=False, encoding="utf-8-sig")
    if len(res.trades):
        res.trades.to_csv(OUT / f"交易_{tag}.csv", index=False, encoding="utf-8-sig")
    if len(res.positions):
        res.positions.to_csv(OUT / f"持仓_{tag}.csv", index=False, encoding="utf-8-sig")
    if len(res.failed):
        res.failed.to_csv(OUT / f"失败订单_{tag}.csv", index=False, encoding="utf-8-sig")
    if len(res.annual):
        res.annual.to_csv(OUT / f"年度_{tag}.csv", index=False, encoding="utf-8-sig")
    if len(res.monthly):
        res.monthly.to_csv(OUT / f"月度_{tag}.csv", encoding="utf-8-sig")
    if len(res.ca):
        res.ca.to_csv(OUT / f"公司行动_{tag}.csv", index=False, encoding="utf-8-sig")
    if len(res.tax):
        res.tax.to_csv(OUT / f"红利税_{tag}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "tag": tag, "config": dataclasses.asdict(res.cfg),
        "metrics": res.metrics, "checks": res.checks, "meta": res.meta,
    }
    return _json_safe(summary)


def run_cash(window: str, panel: FrozenPanel, initial_cash: float = 1_000_000.0) -> RunResult:
    start, end = WINDOWS[window]
    i0, i1 = panel.window_index(start, end)
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in panel.dates[i0:i1 + 1]])
    eq = pd.DataFrame({
        "nav": initial_cash, "market_value": 0.0, "cash": initial_cash,
        "n_positions": 0, "turnover": 0.0, "cost": 0.0,
    }, index=dates)
    bench = panel.benchmark_ew(i0, i1)
    bench_nav = (1 + bench).cumprod()
    bench_nav.index = pd.to_datetime(bench_nav.index)
    bench_nav = bench_nav.reindex(eq.index).ffill()
    res = BacktestResult(eq, pd.DataFrame(), meta={"initial_cash": initial_cash})
    metrics = compute_metrics(res, initial_cash=initial_cash, benchmark=bench_nav)
    cfg = RunConfig(variant="CASH", window=window, initial_cash=initial_cash)
    checks = {
        "cash_resid_max": 0.0, "nav_resid_max": 0.0, "cash_resid_bad_days": 0,
        "nav_resid_bad_days": 0, "t1_violations": 0, "negative_share_events": 0,
        "trades_on_suspension": 0, "buy_at_limit_up_open": 0, "sell_at_limit_down_open": 0,
        "limit_examples": [], "persist_max_days": 0, "persist_violations": 0,
        "forced_count": 0, "delist_forced": 0, "max_date": str(eq.index.max().date()),
        "n_rows_after_2020": 0, "n_orders": 0, "n_orders_filled": 0,
        "ca_resid_max": 0.0, "ca_skipped_rights": 0, "ca_rows": 0, "n_trades": 0,
    }
    checks["gates"] = _gate_summary(checks)
    return RunResult(cfg, eq, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                     pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                     metrics, annual_table(res, initial_cash=initial_cash),
                     monthly_matrix(res, initial_cash=initial_cash),
                     checks, bench_nav, {"variant": "CASH", "window": window})


def _bench_summary(name: str, series: pd.Series, initial: float = 1.0) -> dict:
    s = series.dropna()
    if len(s) < 2:
        return {"name": name, "error": "empty"}
    nav = s / s.iloc[0] * initial
    years = max((nav.index[-1] - nav.index[0]).days, 1) / 365.25
    cagr = (nav.iloc[-1] / initial) ** (1 / years) - 1 if years > 0 else np.nan
    mdd = float((nav / nav.cummax() - 1).min())
    r = nav.pct_change().dropna()
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else np.nan
    return {
        "name": name, "start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
        "years": round(years, 4), "total_return": round(float(nav.iloc[-1] / initial - 1), 6),
        "cagr": round(float(cagr), 6), "mdd": round(mdd, 6),
        "sharpe": round(float(sharpe), 4) if np.isfinite(sharpe) else None,
    }


def write_signal_diagnostics(panel: FrozenPanel) -> dict:
    """信号前向收益诊断（仅诊断，不参与选择；2008-2020）。"""
    i0, i1 = panel.window_index("2008-01-01", "2020-12-31")
    fwd = np.full_like(panel.tr_index, np.nan, dtype=np.float32)
    fwd[:-20] = panel.tr_index[20:] / panel.tr_index[:-20] - 1.0
    sig = panel.signal[i0:i1 + 1]
    f = fwd[i0:i1 + 1]
    years = np.array([d.year for d in panel.dates[i0:i1 + 1]])
    rows = []
    for y in range(2008, 2021):
        m = years == y
        sel = sig & np.isfinite(f)
        v = f[m][sel[m]]
        u = f[m][panel.elig[i0:i1 + 1][m] & np.isfinite(f[m])]
        rows.append({
            "year": y, "n_signal": int(len(v)),
            "signal_fwd20_mean": round(float(np.mean(v)), 6) if len(v) else None,
            "signal_fwd20_median": round(float(np.median(v)), 6) if len(v) else None,
            "universe_fwd20_mean": round(float(np.mean(u)), 6) if len(u) else None,
            "excess_mean": round(float(np.mean(v) - np.mean(u)), 6) if len(v) and len(u) else None,
        })
    daily = pd.DataFrame(rows)
    daily.to_csv(OUT / "诊断_信号前向收益.csv", index=False, encoding="utf-8-sig")

    # 20 日动量十分位（仅 dev+val 的信号样本）
    mom = panel.mom20[i0:i1 + 1]
    idx = np.flatnonzero(sig & np.isfinite(f) & np.isfinite(mom))
    mi = np.unravel_index(idx, f.shape)
    mm = mom[mi]
    rr = f[mi]
    q = pd.qcut(pd.Series(mm), 10, labels=False, duplicates="drop")
    dec = (pd.DataFrame({"momentum_decile": q, "fwd20": rr})
           .groupby("momentum_decile").fwd20.agg(["size", "mean", "median"])
           .reset_index())
    dec.to_csv(OUT / "诊断_动量分位.csv", index=False, encoding="utf-8-sig")
    return {
        "by_year": rows,
        "momentum_decile": dec.to_dict(orient="records"),
        "note": "前向 20 日总回报（tr_index）；仅诊断信号/排名机制，不参与任何选择。",
    }


def _write_markdown(summary: dict, selfchecks: dict | None) -> None:
    lines = []
    lines.append("# 统一引擎 · 冒烟结果\n")
    lines.append(f"- 生成：{summary['meta']['generated_at']}　冻结日：{FREEZE}")
    lines.append(f"- 窗口：开发 2008-2015 / 验证 2016-2020；**未运行 2021+**")
    lines.append(f"- 成本：C-REAL-V1（佣金2.5bp最低5元、历史印花税、沪市2015前过户费0.1%面额、"
                 f"滑点10bp+0.02√参与率、2% ADV20）")
    lines.append(f"- 记账：shares 主口径；tr = shares + 零红利税，仅作敏感性")
    lines.append(f"- 本地试验：{summary['meta']['n_strategy_configs']} 个策略配置 × 2 窗口 = "
                 f"{summary['meta']['n_strategy_runs']} 次账户运行；基准/审计不计入 N")
    lines.append("- T-U03/T-U04 是看到 T-U01 结果后的排名键消融（post-hoc），不得当作独立证据\n")
    lines.append("## 结果\n")
    lines.append("| trial | 变体 | 窗口 | 年化 | 最大回撤 | Calmar | 成交(回合) | 胜率 | 平均持仓 | 持仓占比 | 门槛 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in summary["runs"]:
        m = r["metrics"]
        c = r["checks"]
        gate = "PASS" if c.get("gates", {}).get("all_pass") else "FAIL"
        lines.append(
            f"| {r['trial']} | {r['config']['variant']} | {r['config']['window']} | "
            f"{_pct(m.get('cagr'))} | {_pct(m.get('mdd'))} | {_num(m.get('calmar'))} | "
            f"{m.get('n_trades', 0)} | {_pct(m.get('win_rate'))} | "
            f"{_num(m.get('avg_positions'), 2)} | {_pct(m.get('exposure'))} | {gate} |")
    lines.append("\n## 基准\n")
    for b in summary["benchmarks"]:
        lines.append(f"- {b['name']}（{b.get('start')}..{b.get('end')}）：年化 {_pct(b.get('cagr'))}，"
                     f"最大回撤 {_pct(b.get('mdd'))}，夏普 {_num(b.get('sharpe'))}")
    if selfchecks:
        e = selfchecks["equivalence"]
        lines.append("\n## 引擎自检\n")
        lines.append(f"- 公司行动等价（{e['n_codes']} 只，再投资口径）：中位 |误差| "
                     f"{e['shares_median_abs_err_bp']} bp，最大 {e['shares_max_abs_err_bp']} bp")
        lines.append(f"- shares vs tr 期末净值相对差：{e['shares_vs_tr_nav_rel_diff']:.2e}")
        lines.append(f"- 前置门槛：{selfchecks['gates']}")
        lines.append(f"- 退市前视审计：{selfchecks['delist_lookahead']}")
    size_file = OUT / "规模敏感性.csv"
    if size_file.exists():
        try:
            sd = pd.read_csv(size_file)
            lines.append("\n## 规模敏感性（T-U01 dev，仅敏感性）\n")
            lines.append("| 初始资金 | 年化 | 最大回撤 | 成交 | 费用/初始 |")
            lines.append("|---:|---:|---:|---:|---:|")
            for r in sd.itertuples():
                lines.append(f"| {r.initial_cash:,.0f} | {_pct(r.cagr)} | {_pct(r.mdd)} | "
                             f"{int(r.n_trades)} | {_pct(r.cost_pct_of_initial)} |")
        except Exception:
            pass
    lines.append("\n## 关键结论\n")
    lines.append("- 四个公开基线配置在 dev 与 val **全部为负**（-18.3% 到 -47.2%），"
                 "同期 B1-EW 为 +6.9% / +1.5%。这是负结果，不是引擎失败：逐笔收益与信号"
                 "前向 20 日收益按年对齐（见 `诊断_信号前向收益.csv`）。")
    lines.append("- 主因：60 日新高突破在 A 股 2008-2015 的 20 日前向收益在 dev 的 5/8 年为负、相对宇宙超额 6/8 年为负（2008-2020 全段 8/13 年为负、超额 11/13 年为负）；"
                 "20 日动量排名进一步把组合推入反转最差分位（2008-2020 最高分位 fwd20 -3.48%，最低 +1.37%；dev 单段 -4.13%/+2.97%）。")
    lines.append("- 成本不是唯一原因：全期费用约 0.9%/年（1M 账户），但 2008 年双边 0.3% 印花税"
                 "使当年往返成本 >0.6%。20k 账户的费用占初始资金 38.5%。")
    lines.append("- 因此本阶段**不支持**任何 60% 主张；冒烟策略本身应被淘汰。")
    lines.append("\n## 缺陷与下一步\n")
    lines.append("- 见 `口径冲突与裁决.md` 第 16 节；失败订单逐条落盘。")
    lines.append("- 下一步：把策略族的 28 组候选接入统一引擎复核；本轮只跑公开基线。")
    lines.append("- 中央注册表未写入；4 个配置需由方法负责人登记。")
    (OUT / "结果摘要.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x*100:.2f}%"


def _num(x, d: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


CONFLICTS_MD = """# 第一阶段口径冲突与统一裁决

> 本文件只记录“会改变结论”的冲突；不评价谁对谁错，只写实测证据与统一引擎的裁决。
> 所有实测均可由 `02_代码/统一回测.py --mode selfcheck` 或文中 SQL 复现。

## 1. `hfq = close × factor` 既不是价格，也不是总收益指数

冻结层 `factor = Π(1+pct_chg)` **本身已经是总收益指数**（每个交易日乘 `1+r`，
除权日 `r` 用交易所参考价，送转不产生额外收益）。再乘 `close` 会**把价格收益重复计入一次**：
- 正常日 `hfq` 的日收益 ≈ `(1+r)²−1`，不是 `r`；
- 送转日 `close` 减半、`factor` 不变，`hfq` 也跟着减半，持仓的真实回报却是 0。

所以 `hfq_close/hfq_open` 既不能当价格水平，也不能当总收益。数据字典 5.1 的
“复权收益 vs 官方 pct_chg 偏差 0”只可能是在 `factor` 或 `bs_hfq_close` 上算的，
不是在 `close×factor` 上算的。实测 `000159.SZ` 2008-01-02→2020-12-31：
- `factor` 比值 0.7368（正确总收益）；
- `close×factor` 比值 0.3375×0.7368=0.2487（把价格收益算了两遍）；
- `close/ca_factor` 比值 0.7368（正确价格水平）。

裁决：统一引擎用 `ca_factor = Π(preclose/prev_close)`（只在除权日跳变），
`ca_adj = raw / ca_factor` 为连续价格水平；总收益指数直接用 `factor`（`tr_index`）。
`close×factor` 不进入任何规则。

## 2. `策略族_趋势.py` 读取 Baostock `adjustflag=1`

`策略族_趋势.py:140` 把 `adjustflag=1` 的后复权收盘写入 `bs_hfq.parquet`，指标用它。
数据字典 5.1 已明确该列有年末重定基。实测 2007-2020：
- 60 个证券日的后复权日收益与官方 `pct_chg` 偏离 >0.5pp，涉及 29 只证券，最大 227pp；
- `000001.SZ` 2020-12-31：官方 +0.73%，Baostock hfq 隐含 −16.2%。

裁决：统一引擎**不使用** `bs_hfq_close`，也不使用 Baostock 后复权列；只用冻结层
`close/preclose` 与 `corp_actions_implied`。`策略族_趋势.py` 在 dev/val 的
120 日均线在这些断点上会被污染，其历史结果需在统一引擎复核后才可引用。

## 3. 现金结算：`cash_settle_days=1` vs 父预注册 8.7

`账户撮合.AccountConfig` 默认 `cash_settle_days=1`（卖出回款次日可用）。
A 股实际是“卖出资金当日可用于买入、T+1 可取”，父预注册 8.7 的伪代码也写明
“卖出现金可用于同日开盘买”。

裁决：统一引擎用 `cash_settle_days=0`，并在 `_execute_open_orders` 中先卖后买。
影响：默认设置会把同日“卖旧买新”判成资金不足，系统性降低换手与收益。

## 4. 滑点与参与率

| 模块 | 滑点 | 参与率 |
|---|---|---|
| `账户撮合.FeeConfig` 默认 | 5bp 固定 | 10% 当日成交量 |
| `策略族_趋势` | 5bp 固定 | 无 |
| `策略族_结构` | 10bp 固定 | 无 |
| `退出_回测引擎` | 5bp 固定 | 无 |
| `统计验证.CostModel` | 10bp 固定 | 无 |
| 父预注册 9.1 | 10bp + 0.02√参与率，上限1% | 2% ADV20 |

裁决：统一引擎在 `UnifiedAccount._fill` 内按父预注册实现，订单金额/ADV20 决定冲击项，
超过 2% ADV20 缩减订单。10 万单 / 3,000 万 ADV20 的参与率 0.33%，冲击约 11.5bp，
合计约 21.5bp，是第一阶段 5bp 假设的 4 倍。**这是冒烟结果与族引擎结果不可比的主因之一。**

## 5. 费用历史

- `策略族_趋势`、`策略族_结构`、`退出_回测引擎` 都把印花税固定为卖出 0.1%，
  不含 2007-05-30..2008-04-23 的 0.3% 与 2008-04-24..2008-09-18 的双边 0.1%；
- `策略族_趋势` 的 ADV 门槛是 2,000 万，父预注册是 3,000 万；
- 过户费：`费用与交易约束` 沪市 2015 前按面额 0.06%，父预注册按 0.1% 保守上界。
  统一引擎主口径用 0.1%，`--transfer-par 0.0006` 可跑敏感性。

`2008-01-01..2008-09-18` 落在开发期，固定 0.1% 会低估卖出成本 0.2pp/次（双边期）。

## 6. 红利税：固定 20% vs 差别化

`账户撮合` 默认在除权日按 20% 扣红利税。真实规则是卖出时按持股期限：
≤1 月 20%、1 月-1 年 10%、>1 年 0%。

裁决：统一引擎在卖出时按上述三档结算，未平仓部分作为“未结算税负债”单列。
对 20 日持有策略，主口径接近 20%；对买入持有夹具，主口径接近 0%。固定 20%
会系统性低估长持有策略的收益。

## 7. `delisting_missing_days=10` 会把“长期停牌”当退市

`账户撮合._check_delistings` 对连续 10 个交易日没有可交易 bar 的持仓执行
`_force_sell`，理由 `delisting_missing_bars`。A 股 2015 停牌潮、重组停牌常超过 10 日，
这会把停牌仓位按停牌前最后价强制卖出，既不是退市，也不是真实可成交价。

裁决：统一引擎设 `delisting_missing_days=10_000_000`，停牌期间按最后收盘估值，
只认 `symbols.delist_date` 的退市。`out_date` 后按最后收盘价退出（主口径不折扣）。

## 8. 公司行动一致性门控用“开盘价”会误杀

`账户撮合._event_price_consistent` 用当日开盘价与事件公式参考价比对，容差为
涨跌停幅度+2%。复牌、股改、缩股、退市整理期首日会超出容差，事件被跳过，
净值出现假跌。

裁决：统一引擎改用交易所 `preclose` 对账：`adj_ratio = preclose/prev_close` 为权威；
事件表只用于拆分分红/送转/配股，且仅当事件公式复原参考价与 `preclose`
相对误差 ≤0.5% 时采信；拆不开时全部记现金（总回报等价）。2007-2020 实测：
26,392 个除权日中 93.57% 可拆，6.13% 走现金等价，0.30% 事件表与价格冲突，
0 个比例非法。

## 9. 宇宙用未来 `out_date` 剔除退市整理期（父预注册自身的前视）

父预注册 4.4 第 4 条要求“退市整理期（out_date 前 30 个交易日）不新开仓”。
`out_date` 在 t 日并不可知，这是**未来信息**，会系统性避开退市前的下跌。
统一引擎遵守父规则，但 `audit_delist_lookahead` 量化被挡信号数与其后 20 日收益，
并在结果中标注。若影响大，应在下一版预注册中改为“只用 t 日已知的退市风险标记”。

## 10. `统计验证.limit_ratio` 与 `费用与交易约束.ipo_limit_pct` 不一致

- `统计验证.limit_ratio`：创业板 2020-08-24 后一律 20%，主板 ST 一律 5%，
  不含 2026-07-06 主板 ST 改 10%；也不处理新股前 5 日无限制。
- `费用与交易约束.ipo_limit_pct`：含板块、ST 历史、新股窗口、2026 变更。

裁决：撮合以 `费用与交易约束` 为准；`统计验证` 的涨跌停标记只用于研究标签，
不得用于“能否成交”。本阶段窗口不触及 2026，差异只在历史 ST/创业板边界。

## 11. `策略族_结构` 与 `退出_回测引擎` 的记账

- `退出_回测引擎`：`shares × exit_raw × (f_exit/f_entry)`，等价于“原始价+分红累计”，
  但把分红按因子再投资，且不模拟真实现金/整手/最低佣金；
- `策略族_结构`：事件级、次开、T+1、整手、费用滑点，但不含历史印花税分段。

裁决：族引擎只用于**信号筛选与消融**；最终账户结论必须由统一引擎复核。
本阶段统一引擎只跑公开基线，不复核族引擎的 28 组候选（那是下一阶段的工作）。

## 13. `数据层.daily(adjust="hfq")` 的 `close` 也是 `close × factor`

`数据层.py` 的 hfq 分支把 `hfq_close AS close` 返回（`数据层.py:112-119`），
与冻结层 `hfq = raw × factor` 同源。它的 `close.pct_change()` 不是总收益：
正常日约为 `(1+r)²−1`，送转日还会随原始价减半。任何用
`dl.daily(adjust="hfq")` 算收益或指标的策略都会重复计入价格收益。

裁决：统一引擎不调用 `DataLayer.daily(adjust="hfq")`；直接读 `daily` 的
`close/open/high/low/preclose/factor` 与 `corp_actions_implied`，自建 `ca_factor`。
`DataLayer` 的 raw 模式与 `universe()` 仍可用于点时筛选。

## 14. `核心特征.py` 不选定复权口径，但指标对传入 `close` 敏感

`核心特征.Bars/feature_matrix` 不决定用 raw 还是 hfq；所有指标（MA、ATR、枢轴、
Donchian、RSI）都直接在传入的 `close/high/low` 上算。族引擎传 Baostock hfq 时，
指标被年末重定基污染；传 raw 时，除权日跳变会伪造突破/止损。两种都不对。

裁决：统一引擎在 `ca_adj = raw / ca_factor` 上算指标，在 raw 上撮合。
`核心特征.py` 的因果规则（分型确认、周期聚合、截断重算）可复用，但必须显式传入
`ca_adj` 价格的 `Bars`，不能传 `hfq_close` 或 raw。

## 15. `DataLayer.universe()` 缺 `resume` 与退市整理期

`数据层.py` 的 `universe()` 过滤了停牌、ST、价格、ADV、上市天数，但没有
`resume`（复牌首日）与 `out_date` 前 30 个交易日。直接用它会买入复牌首日与
退市整理期个股。统一引擎在适配层补了这两个掩码，并在口径冲突第 9 节披露
`out_date` 本身的前视问题。

## 16. 未解决 / 需主协调者裁决

1. 中央 `03_结果/实验注册表.csv` 的登记由方法负责人完成；4 个配置 × 2 窗口 = 8 次账户运行目前只在
   `03_结果/统一引擎/试验日志.csv` 本地登记。
2. 退市处置只实现“最后收盘价”与 `--delist-haircut 0.3` 两档；真实三板转让回收率未建模。
3. 分红税按持仓期在卖出时结算，但送转股的持期起算按原买入日近似；真实规则更细。
4. `preclose` 在极少数复牌日滞后（数据字典 4.2 第 6 条），本引擎按“不复权价真实变动”
   处理，可能把数据滞后计成真实亏损；已用 `resume` 禁止新开仓，但未修正估值。
5. 2% ADV20 的成交上限依赖当日成交量做二次约束，开盘时点无法知道全日量；
   统一引擎取 `min(2% ADV20, 2% 当日量)`，属于保守近似（不是 10% 当日量）。
"""


def write_trial_log(runs: list[dict]) -> None:
    rows = []
    for r in runs:
        m = r["metrics"]
        rows.append({
            "trial_id": r["trial"], "window": r["config"]["window"],
            "variant": r["config"]["variant"], "accounting": r["config"]["accounting"],
            "status": "done", "n_configs": r.get("n_configs", 1),
            "cagr": m.get("cagr"), "mdd": m.get("mdd"), "sharpe": m.get("sharpe"),
            "n_trades": m.get("n_trades", 0),
            "decision": r.get("decision", "inconclusive"),
            "notes": r.get("notes", ""),
        })
    pd.DataFrame(rows).to_csv(OUT / "试验日志.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _bench_frame(panel: FrozenPanel, window: str) -> pd.DataFrame:
    start, end = WINDOWS[window]
    i0, i1 = panel.window_index(start, end)
    b1 = panel.benchmark_ew(i0, i1)
    b1_nav = (1 + b1).cumprod()
    df = pd.DataFrame({"date": [pd.Timestamp(d) for d in panel.dates[i0:i1 + 1]],
                       "B1_EW": b1_nav.to_numpy()})
    hs = load_index_series(start, end, "000300.SH").reindex(pd.to_datetime(df["date"])).ffill()
    df["HS300"] = hs.to_numpy()
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="统一引擎：开发/验证冒烟与自检")
    ap.add_argument("--mode", choices=["selfcheck", "smoke", "sizes"], default="smoke")
    ap.add_argument("--window", choices=["dev", "val", "all"], default="all")
    ap.add_argument("--accounting", choices=["shares", "tr"], default="shares")
    ap.add_argument("--variant", default="", help="逗号分隔：FIX20,ATR3,BH10")
    ap.add_argument("--rank", choices=["mom", "adv", "both"], default="mom",
                    help="排名键：mom=20日动量（预注册主口径）；adv=仅ADV20（消融）；both=两套都跑")
    ap.add_argument("--sizes", action="store_true", help="额外跑 2万/10万 规模敏感性")
    ap.add_argument("--slippage-bps", type=float, default=10.0)
    ap.add_argument("--transfer-par", type=float, default=0.0010)
    ap.add_argument("--delist-haircut", type=float, default=0.0)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] 加载事件簿与面板 2007-2020 …", flush=True)
    events = EventBook().load("2007-01-01", "2020-12-31")
    panel = FrozenPanel("2007-01-01", "2020-12-31", events=events)
    print(f"  面板 {panel.n_days} 日 × {panel.n_codes} 只，{panel.n_rows_raw:,} 行；"
          f"除权 {events.stats['n_events']:,} 条", flush=True)

    if args.mode == "selfcheck":
        checks = selfcheck(panel, events, write=not args.no_write)
        print(json.dumps(_json_safe(checks), ensure_ascii=False, indent=1)[:4000])
        print(f"[{datetime.now():%H:%M:%S}] 自检完成 {time.time()-t0:.1f}s")
        return 0

    if args.mode == "sizes":
        rows = []
        for cash_amt in (20_000.0, 100_000.0, 1_000_000.0):
            cfg = RunConfig(variant="FIX20", window="dev", accounting="shares",
                            initial_cash=cash_amt, rank="mom",
                            slippage_bps=args.slippage_bps,
                            transfer_par_rate=args.transfer_par)
            print(f"[{datetime.now():%H:%M:%S}] 规模 {cash_amt:,.0f} …", flush=True)
            res = run_variant(cfg, panel, events)
            hs = load_index_series(*WINDOWS["dev"], "000300.SH")
            summary = write_run(res, hs)
            m = res.metrics
            rows.append({
                "initial_cash": cash_amt, "cagr": m.get("cagr"), "mdd": m.get("mdd"),
                "n_trades": m.get("n_trades"), "avg_trade_ret": m.get("avg_trade_ret"),
                "turnover": m.get("turnover"), "cost_paid": m.get("cost_paid"),
                "cost_pct_of_initial": (m.get("cost_paid") or 0) / cash_amt,
                "gate": res.checks["gates"]["all_pass"],
                "max_participation_note": "下单量/ADV20 越高，冲击项越大；小资金更接近 10bp 基础滑点",
            })
            print(f"    CAGR {m.get('cagr', float('nan')):.2%}  MDD {m.get('mdd', float('nan')):.2%}  "
                  f"trades {m.get('n_trades', 0)}", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "规模敏感性.csv", index=False, encoding="utf-8-sig")
        (OUT / "规模敏感性.json").write_text(
            json.dumps(_json_safe({"note": "仅敏感性，不参与选择；T-U01 dev 同规则",
                                   "rows": rows, "elapsed_s": round(time.time()-t0, 2)}),
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(df.to_string(index=False))
        return 0

    windows = ["dev", "val"] if args.window == "all" else [args.window]
    variants = [v.strip() for v in args.variant.split(",") if v.strip()] or ["FIX20", "ATR3", "BH10"]
    trial_map = {("FIX20", "mom"): "T-U01", ("ATR3", "mom"): "T-U02",
                 ("FIX20", "adv"): "T-U03", ("ATR3", "adv"): "T-U04",
                 ("BH10", "mom"): "F-U01"}

    runs: list[dict] = []
    bench_rows: list[dict] = []
    selfchecks = None
    for window in windows:
        print(f"[{datetime.now():%H:%M:%S}] === {window} ===", flush=True)
        for variant in variants:
            trade_variant = variant in ("FIX20", "ATR3")
            ranks = (["mom", "adv"] if args.rank == "both" else [args.rank]) if trade_variant else ["mom"]
            for rank in ranks:
                cfg = RunConfig(variant=variant, window=window, accounting=args.accounting,
                                slippage_bps=args.slippage_bps,
                                transfer_par_rate=args.transfer_par,
                                delist_haircut=args.delist_haircut, rank=rank)
                print(f"  运行 {variant} {window} rank={rank} ({cfg.accounting}) …", flush=True)
                res = run_variant(cfg, panel, events)
                hs = load_index_series(*WINDOWS[window], "000300.SH")
                summary = write_run(res, hs)
                summary["trial"] = trial_map.get((variant, rank), f"X-{variant}-{rank}")
                summary["n_configs"] = 1 if trade_variant else 0
                summary["decision"] = "discard" if trade_variant else "na"
                summary["post_hoc"] = (rank == "adv")
                summary["notes"] = ("冒烟公开基线，不用于 60% 结论；参数在净值前冻结"
                                    if trade_variant and rank == "mom" else
                                    ("排名键消融，post-hoc；不得当作独立证据" if trade_variant
                                     else "基准/夹具"))
                runs.append(summary)
                m = res.metrics
                print(f"    CAGR {m.get('cagr', float('nan')):.2%}  MDD {m.get('mdd', float('nan')):.2%}  "
                      f"trades {m.get('n_trades', 0)}  gate {res.checks['gates']['all_pass']}", flush=True)

        cash = run_cash(window, panel)
        cs = write_run(cash)
        cs["trial"] = "B-U03"
        cs["n_configs"] = 0
        cs["decision"] = "na"
        runs.append(cs)

        bf = _bench_frame(panel, window)
        if not args.no_write:
            bf.to_csv(OUT / f"基准_{window}.csv", index=False, encoding="utf-8-sig")
        b1 = pd.Series(bf["B1_EW"].to_numpy(), index=pd.DatetimeIndex(bf["date"]))
        hs = pd.Series(bf["HS300"].to_numpy(), index=pd.DatetimeIndex(bf["date"])).dropna()
        bench_rows.append(_bench_summary(f"B1-EW 主宇宙等权（{window}）", b1))
        bench_rows.append(_bench_summary(f"B2 沪深300 价格指数（{window}）", hs))

        if args.sizes and window == "dev":
            for cash_amt in (20_000.0, 100_000.0):
                cfg = RunConfig(variant="FIX20", window=window, accounting=args.accounting,
                                initial_cash=cash_amt, slippage_bps=args.slippage_bps,
                                transfer_par_rate=args.transfer_par)
                print(f"  规模敏感性 {cash_amt:,.0f} …", flush=True)
                res = run_variant(cfg, panel, events)
                summary = write_run(res, hs)
                summary["trial"] = "T-U01-SIZE"
                summary["n_configs"] = 0
                summary["decision"] = "na"
                summary["notes"] = "规模敏感性，不参与选择"
                runs.append(summary)

    if not args.no_write:
        selfchecks = selfcheck(panel, events, write=True)
        cov = panel.coverage_table()
        cov.to_csv(OUT / "覆盖_股票池.csv", index=False, encoding="utf-8-sig")
        ev = events.events.copy()
        if len(ev):
            ev["year"] = pd.to_datetime(ev["ex_date"]).dt.year
            tab = (ev.groupby(["year", "status"]).size().unstack(fill_value=0)
                   .reset_index())
            tab.to_csv(OUT / "公司行动_对账_分年.csv", index=False, encoding="utf-8-sig")
            bad = ev[ev["status"] != "verified_ths"].head(3000)
            bad.to_csv(OUT / "公司行动_对账_非验证样本.csv", index=False, encoding="utf-8-sig")
        write_trial_log(runs)
        diag = write_signal_diagnostics(panel)
        (OUT / "诊断_信号前向收益.json").write_text(
            json.dumps(_json_safe(diag), ensure_ascii=False, indent=1), encoding="utf-8")
        (OUT / "口径冲突与裁决.md").write_text(CONFLICTS_MD, encoding="utf-8")
        (OUT / "操作说明与复现.md").write_text(REPRO_MD, encoding="utf-8")

        summary = {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "freeze": FREEZE,
                "windows": windows,
                "cost_model": "C-REAL-V1",
                "accounting": args.accounting,
                "slippage_bps_base": args.slippage_bps,
                "transfer_par_rate": args.transfer_par,
                "n_strategy_configs": 4,
                "n_strategy_runs": sum(1 for r in runs if r["n_configs"] == 1),
                "elapsed_s": round(time.time() - t0, 2),
                "code": _code_meta(),
                "no_2021_returns": True,
            },
            "event_book": events.stats,
            "coverage": cov.to_dict(orient="records"),
            "runs": runs,
            "benchmarks": bench_rows,
            "selfcheck_gates": selfchecks["gates"] if selfchecks else None,
            "selfcheck_equivalence": (selfchecks or {}).get("equivalence"),
            "delist_lookahead": (selfchecks or {}).get("delist_lookahead"),
            "decisions": {
                "strategy_results_are_smoke_only": True,
                "no_60pct_claim": True,
                "tuning": "none; parameters frozen in 00_预注册_统一引擎冒烟.md",
                "central_registry": "未写入；由方法负责人登记",
            },
        }
        (OUT / "汇总.json").write_text(
            json.dumps(_json_safe(summary), ensure_ascii=False, indent=1), encoding="utf-8")
        _write_markdown(summary, selfchecks)
        print(f"[{datetime.now():%H:%M:%S}] 输出写入 {OUT}；总耗时 {time.time()-t0:.1f}s")
    return 0


REPRO_MD = """# 统一引擎操作与复现

## 1. 文件职责

| 文件 | 职责 |
|---|---|
| `02_代码/统一数据适配.py` | 冻结 Parquet → 交易日×代码矩阵、点时宇宙、除权对账、逐日 Bar |
| `02_代码/统一回测.py` | 统一账户（公司行动/红利税/滑点/参与率）、公开基线、逐日会计恒等、自检、输出 |
| `03_结果/统一引擎/` | 本阶段全部结果、预注册、试验日志、冲突裁决 |
| `04_证据/统一引擎自检.json` | 机器可读自检（事件簿、等价性、前视审计、门槛） |

## 2. 复现命令

```bash
cd /Users/admin/WorkSpace/trading/trading_趋势结构
python3 02_代码/统一回测.py --mode selfcheck
python3 02_代码/统一回测.py --mode smoke --window dev --rank both
python3 02_代码/统一回测.py --mode smoke --window val --rank both
python3 02_代码/统一回测.py --mode smoke --window all --rank both
# 规模敏感性（开发期，仅敏感性，不参与选择；单独写 规模敏感性.csv，不覆盖 汇总.json）
python3 02_代码/统一回测.py --mode sizes
# 过户费历史 0.06% 敏感性
python3 02_代码/统一回测.py --mode smoke --window dev --transfer-par 0.0006
# 退市压力价
python3 02_代码/统一回测.py --mode smoke --window dev --delist-haircut 0.3
```

## 3. 输出表

- `净值_*.csv`：date,nav,cash,market_value,n_positions,turnover,cost,bench_b1,bench_hs300,drawdown
- `成交_*.csv`：逐笔成交（含费用拆分、滑点后价格、order_id、原因）
- `交易_*.csv`：回合级（entry/exit、pnl、ret、fees、dividends、tax、hold_days）
- `持仓_*.csv`：逐日持仓（shares、close、mv、weight、entry_date、stop_price）
- `失败订单_*.csv`：涨停/跌停/停牌/资金不足/参与率为零等
- `年度_*.csv`、`月度_*.csv`、`公司行动_*.csv`、`红利税_*.csv`
- `汇总.json`、`试验日志.csv`、`覆盖_股票池.csv`、`结果摘要.md`

## 4. 口径要点

- 信号 t 收盘、订单 t+1 开盘；买单 day 单失败不追；卖单 GTC 顺延 ≤10 日。
- 现金账：`cash_settle_days=0`（卖出资金当日可用）。
- 公司行动：`preclose/prev_close` 为权威；事件表仅拆分，拆不开走现金等价。
- 红利税：卖出时按持股期限（20%/10%/0%）结算；未平仓部分记负债。
- 退市：只认 `delist_date`；停牌不强制卖出；退市后按最后收盘价退出。
- 滑点：10bp + 0.02√参与率，上限 1%；复牌日加倍；成交上限 2% ADV20。
- 不用 Baostock 后复权列；不用冻结层 `factor` 当价格水平。

## 5. 已知限制

见 `口径冲突与裁决.md` 第 16 节。
"""


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
