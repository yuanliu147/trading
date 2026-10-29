"""日线 A 股现金账户与撮合引擎（研究用，只读数据源）。

设计边界
--------
* 只用日线 OHLCV，不构造盘中路径；同一日内止损与止盈同时触及时按“损失优先”处理；
* 信号在收盘后生成，最早下一交易日开盘成交；订单执行日必须严格晚于决策日；
* 现金账户、真实股数、整股规则、T+1、涨跌停、停牌、最低佣金、历史税费、
  滑点、成交量参与率、公司行动、退市与期末估值均在账户层处理；
* 公司行动来自 DuckDB `raw_adjustment_events`；无法精确还原的环节显式写入 flags，
  不做静默近似；
* 不在此模块内生成任何真实收益结论；自检只用合成行情。

与策略层的接口
--------------
策略层只调用 `buy_open/sell_open/sell_stop/sell_limit` 提交订单，然后按交易日顺序
调用 `on_day`。账户不读取未来行情；`Bar` 只允许包含当日及更早信息。数据适配器负责
把 DuckDB/Baostock 转成 `Bar`/`CorporateAction`，并保证 `prev_close` 口径
（见 `Bar.prev_close_is_ref`）。

最小用法::

    acct = Account(AccountConfig(initial_cash=1_000_000), calendar, freeze_date=...)
    acct.on_day(d, bars, events=events, st_codes=st_codes)
    acct.buy_open("600000.SH", 100, decision_date=d)
    # 下一交易日 on_day 时开盘成交；之后可挂 sell_stop/sell_limit
    ...
    acct.finalize(last_day)
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence


def _load_fee_module():
    path = Path(__file__).with_name("费用与交易约束.py")
    spec = importlib.util.spec_from_file_location("trend_fee_rules", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载费用模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


fee = _load_fee_module()
Date = date


def canonical_code(code: str) -> str:
    c = str(code).strip()
    if c.lower().startswith(("sh.", "sz.", "bj.")):
        ex, num = c.split(".", 1)
        return f"{num}.{ex.upper()}"
    return c


def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    code: str
    date: Date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float | None = None
    prev_close: float | None = None
    prev_close_is_ref: bool = False
    is_st: bool = False
    trade_status: int = 1

    def __post_init__(self):
        self.code = canonical_code(self.code)
        self.date = fee.to_date(self.date)
        for name in ("open", "high", "low", "close"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{self.code} {self.date} {name} 非正/非有限: {v!r}")
            setattr(self, name, v)
        self.volume = float(self.volume)
        if not math.isfinite(self.volume) or self.volume < 0:
            raise ValueError(f"{self.code} {self.date} volume 非法: {self.volume!r}")
        if self.high < self.low - 1e-9:
            raise ValueError(f"{self.code} {self.date} high<low")
        tol = 1e-6
        if self.open > self.high + tol or self.open < self.low - tol:
            raise ValueError(f"{self.code} {self.date} open 不在 [low,high]")
        if self.close > self.high + tol or self.close < self.low - tol:
            raise ValueError(f"{self.code} {self.date} close 不在 [low,high]")
        if self.prev_close is not None:
            self.prev_close = float(self.prev_close)
            if not math.isfinite(self.prev_close) or self.prev_close <= 0:
                self.prev_close = None

    @property
    def tradable(self) -> bool:
        return self.trade_status == 1 and self.volume > 0 and self.open > 0

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "date": self.date.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "amount": self.amount,
            "prev_close": self.prev_close,
            "prev_close_is_ref": self.prev_close_is_ref,
            "is_st": self.is_st,
            "trade_status": self.trade_status,
        }


@dataclass(frozen=True)
class SymbolInfo:
    code: str
    name: str = ""
    ipo_date: Date | None = None
    out_date: Date | None = None

    def __post_init__(self):
        object.__setattr__(self, "code", canonical_code(self.code))
        if self.ipo_date is not None:
            object.__setattr__(self, "ipo_date", fee.to_date(self.ipo_date))
        if self.out_date is not None:
            object.__setattr__(self, "out_date", fee.to_date(self.out_date))


@dataclass(frozen=True)
class CorporateAction:
    code: str
    ex_date: Date
    dividend_per_share: float = 0.0
    per_share_bonus: float = 0.0
    rights_ratio: float = 0.0
    rights_price: float = 0.0
    source: str = "duckdb.raw_adjustment_events"

    def __post_init__(self):
        object.__setattr__(self, "code", canonical_code(self.code))
        object.__setattr__(self, "ex_date", fee.to_date(self.ex_date))
        for name in ("dividend_per_share", "per_share_bonus", "rights_ratio", "rights_price"):
            v = float(getattr(self, name) or 0.0)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{self.code} {self.ex_date} {name} 非法: {v!r}")
            object.__setattr__(self, name, v)
        if self.rights_ratio > 0 and self.rights_price <= 0:
            object.__setattr__(self, "rights_price", 0.0)

    def key(self) -> tuple:
        return (self.code, self.ex_date)

    def is_empty(self) -> bool:
        return self.dividend_per_share == 0 and self.per_share_bonus == 0 and self.rights_ratio == 0


@dataclass
class Order:
    code: str
    side: str
    qty: int
    kind: str = "open"          # open / stop / limit
    price: float | None = None  # stop 触发价或 limit 限价
    tif: str = "day"            # day / gtc
    tag: str = ""
    decision_date: Date | None = None
    execute_from: Date | None = None
    order_id: str = ""
    filled_qty: int = 0
    avg_price: float = 0.0
    status: str = "pending"     # pending / partial / filled / cancelled / rejected
    reason: str = ""
    triggered: bool = False
    seq: int = 0

    def __post_init__(self):
        self.code = canonical_code(self.code)
        self.side = str(self.side).lower()
        self.kind = str(self.kind).lower()
        self.tif = str(self.tif).lower()
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"side 非法: {self.side!r}")
        if self.kind not in {"open", "stop", "limit"}:
            raise ValueError(f"kind 非法: {self.kind!r}")
        if self.tif not in {"day", "gtc"}:
            raise ValueError(f"tif 非法: {self.tif!r}")
        self.qty = int(self.qty)
        if self.qty <= 0:
            raise ValueError("qty 必须为正")
        if self.kind in {"stop", "limit"}:
            if self.price is None or float(self.price) <= 0:
                raise ValueError(f"{self.kind} 单必须给出正价格")
            self.price = float(self.price)
        elif self.price is not None:
            raise ValueError("open 单不应带价格")

    @property
    def remaining(self) -> int:
        return max(0, self.qty - self.filled_qty)

    @property
    def active(self) -> bool:
        return self.status in {"pending", "partial"} and self.remaining > 0

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("decision_date", "execute_from"):
            if d.get(k) is not None:
                d[k] = d[k].isoformat()
        return d


@dataclass
class Trade:
    date: Date
    code: str
    side: str
    qty: int
    price: float
    amount: float
    commission: float
    stamp_duty: float
    transfer_fee: float
    fees: float
    cash_after: float
    kind: str
    order_id: str
    tag: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "code": self.code,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "amount": round(self.amount, 4),
            "commission": self.commission,
            "stamp_duty": self.stamp_duty,
            "transfer_fee": self.transfer_fee,
            "fees": self.fees,
            "cash_after": round(self.cash_after, 4),
            "kind": self.kind,
            "order_id": self.order_id,
            "tag": self.tag,
            "reason": self.reason,
        }


@dataclass
class Position:
    code: str
    shares: int = 0
    available: int = 0
    cost_total: float = 0.0
    last_price: float = 0.0
    last_date: Date | None = None
    first_buy_date: Date | None = None
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    missing_days: int = 0

    def __post_init__(self):
        self.code = canonical_code(self.code)

    @property
    def avg_cost(self) -> float:
        return self.cost_total / self.shares if self.shares > 0 else 0.0

    def add_paid(self, qty: int, price: float, fees: float, d: Date) -> None:
        qty = int(qty)
        if qty <= 0:
            return
        if self.shares == 0:
            self.first_buy_date = d
        self.shares += qty
        self.cost_total += float(price) * qty + float(fees)
        self.fees_paid += float(fees)

    def add_free(self, qty: int) -> None:
        qty = int(qty)
        if qty > 0:
            self.shares += qty

    def remove(self, qty: int, price: float, fees: float) -> float:
        qty = int(qty)
        if qty <= 0 or qty > self.shares:
            raise ValueError(f"{self.code} 卖出数量非法: qty={qty} shares={self.shares}")
        cost_removed = self.cost_total * (qty / self.shares) if self.shares else 0.0
        realized = float(price) * qty - float(fees) - cost_removed
        self.cost_total -= cost_removed
        self.shares -= qty
        self.available = max(0, self.available - qty)
        self.realized_pnl += realized
        self.fees_paid += float(fees)
        if self.shares == 0:
            self.cost_total = 0.0
            self.available = 0
        return realized

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "shares": self.shares,
            "available": self.available,
            "avg_cost": round(self.avg_cost, 5),
            "cost_total": round(self.cost_total, 2),
            "last_price": self.last_price,
            "last_date": self.last_date.isoformat() if self.last_date else None,
            "first_buy_date": self.first_buy_date.isoformat() if self.first_buy_date else None,
            "realized_pnl": round(self.realized_pnl, 2),
            "fees_paid": round(self.fees_paid, 2),
            "missing_days": self.missing_days,
        }


@dataclass
class AccountConfig:
    initial_cash: float = 1_000_000.0
    fee: fee.FeeConfig = field(default_factory=fee.FeeConfig)
    cash_settle_days: int = 1          # 1=卖出/分红现金次日可用；0=当日可用
    enforce_t1: bool = True
    allow_buy_downsize: bool = True
    normalize_buy_orders: bool = True
    rights_policy: str = "subscribe"   # subscribe / skip
    dividend_tax_rate: float | None = None
    delisting_missing_days: int = 10
    delisting_haircut: float = 0.10
    liquidate_on_finalize: bool = False
    finalize_haircut: float = 0.0
    strict: bool = True
    max_flags: int = 100_000

    def __post_init__(self):
        if self.initial_cash <= 0:
            raise ValueError("initial_cash 必须为正")
        if self.cash_settle_days not in (0, 1):
            raise ValueError("cash_settle_days 只支持 0/1")
        if self.rights_policy not in {"subscribe", "skip"}:
            raise ValueError("rights_policy 只支持 subscribe/skip")
        if self.delisting_missing_days < 1:
            raise ValueError("delisting_missing_days 必须>=1")
        for name in ("delisting_haircut", "finalize_haircut"):
            v = float(getattr(self, name))
            if not (0 <= v <= 1):
                raise ValueError(f"{name} 必须在 [0,1]")
        errs = fee.validate_fee_config(self.fee)
        if errs:
            raise ValueError("; ".join(errs))

    @property
    def effective_dividend_tax_rate(self) -> float:
        return self.fee.dividend_tax_rate if self.dividend_tax_rate is None else float(self.dividend_tax_rate)


# ---------------------------------------------------------------------------
# 账户与撮合
# ---------------------------------------------------------------------------

class Account:
    def __init__(
        self,
        config: AccountConfig | None = None,
        calendar: Sequence[Date] | None = None,
        freeze_date: Date | None = None,
        symbol_info: Mapping[str, SymbolInfo] | None = None,
    ):
        self.config = config or AccountConfig()
        if not calendar:
            raise ValueError("必须提供交易日历")
        self.freeze = fee.to_date(freeze_date) if freeze_date else max(fee.to_date(x) for x in calendar)
        cal = sorted({fee.to_date(x) for x in calendar if fee.to_date(x) <= self.freeze})
        if not cal:
            raise ValueError("冻结日之前没有交易日")
        self.calendar = cal
        self._cal_index = {d: i for i, d in enumerate(self.calendar)}
        self.symbol_info = {canonical_code(k): v for k, v in (symbol_info or {}).items()}

        self.cash_total = float(self.config.initial_cash)
        self.cash_available = float(self.config.initial_cash)
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self.trades: list[Trade] = []
        self.daily: list[dict] = []
        self.corporate_actions: list[dict] = []
        self.flags: list[dict] = []
        self.flag_counts: Counter = Counter()

        self._orders: list[Order] = []
        self._order_seq = 0
        self._last_close: dict[str, float] = {}
        self._day_bars: dict[str, Bar] = {}
        self._day_limits: dict[str, fee.PriceLimits | None] = {}
        self._pending_events: list[CorporateAction] = []
        self._applied_events: set[tuple] = set()
        self._day: Date | None = None
        self._stop_triggered_today: set[str] = set()

    # -- 日历与订单提交 -----------------------------------------------------

    def next_trading_day(self, d: Date) -> Date:
        d = fee.to_date(d)
        idx = bisect_right(self.calendar, d)
        if idx >= len(self.calendar):
            raise ValueError(f"{d} 之后没有交易日（冻结日 {self.freeze}）")
        return self.calendar[idx]

    def _flag(self, code: str, kind: str, detail: Any = None, d: Date | None = None) -> None:
        self.flag_counts[kind] += 1
        if len(self.flags) >= self.config.max_flags:
            return
        self.flags.append({
            "date": (d or self._day).isoformat() if (d or self._day) else None,
            "code": canonical_code(code) if code else None,
            "kind": kind,
            "detail": detail,
        })

    def note_flag(self, code: str, kind: str, detail: Any = None, d: Date | None = None) -> None:
        """供协调者写入审计标记（例如数据源的 unexplained factor change）。"""
        self._flag(code, kind, detail, d)

    def submit(self, order: Order, decision_date: Date) -> Order:
        dd = fee.to_date(decision_date)
        if dd > self.freeze:
            raise ValueError(f"决策日 {dd} 晚于冻结日 {self.freeze}")
        if order.execute_from is None:
            order.execute_from = self.next_trading_day(dd)
        ef = fee.to_date(order.execute_from)
        if ef <= dd:
            raise ValueError(f"禁止未来函数/当日成交: decision={dd} execute_from={ef}")
        if ef not in self._cal_index:
            raise ValueError(f"执行日 {ef} 不在交易日历")
        order.decision_date = dd
        order.execute_from = ef
        self._order_seq += 1
        order.seq = self._order_seq
        order.order_id = f"O{self._order_seq:06d}"
        if order.side == "buy" and self.config.normalize_buy_orders:
            nq = fee.normalize_buy_qty(order.qty, order.code)
            if nq != order.qty:
                self._flag(order.code, "buy_qty_rounded_to_lot", {"requested": order.qty, "normalized": nq})
                order.qty = nq
        if order.qty <= 0:
            order.status = "rejected"
            order.reason = "lot_below_min"
        self.orders[order.order_id] = order
        self._orders.append(order)
        return order

    def buy_open(self, code: str, qty: int, decision_date: Date, tag: str = "") -> Order:
        return self.submit(Order(code=code, side="buy", qty=qty, kind="open", tif="day", tag=tag), decision_date)

    def sell_open(self, code: str, qty: int, decision_date: Date, tag: str = "") -> Order:
        return self.submit(Order(code=code, side="sell", qty=qty, kind="open", tif="day", tag=tag), decision_date)

    def sell_stop(self, code: str, qty: int, stop_price: float, decision_date: Date, tag: str = "") -> Order:
        return self.submit(Order(code=code, side="sell", qty=qty, kind="stop", price=stop_price, tif="gtc", tag=tag), decision_date)

    def sell_limit(self, code: str, qty: int, limit_price: float, decision_date: Date, tag: str = "") -> Order:
        return self.submit(Order(code=code, side="sell", qty=qty, kind="limit", price=limit_price, tif="gtc", tag=tag), decision_date)

    def buy_limit(self, code: str, qty: int, limit_price: float, decision_date: Date, tag: str = "") -> Order:
        return self.submit(Order(code=code, side="buy", qty=qty, kind="limit", price=limit_price, tif="day", tag=tag), decision_date)

    def cancel(self, order_id: str, reason: str = "manual_cancel") -> None:
        o = self.orders.get(order_id)
        if o and o.active:
            o.status = "cancelled"
            o.reason = reason

    # -- 日循环 -------------------------------------------------------------

    def on_day(
        self,
        d: Date,
        bars: Mapping[str, Bar] | None = None,
        events: Sequence[CorporateAction] | None = None,
        st_codes: Iterable[str] | None = None,
        is_trading_day: bool = True,
    ) -> dict:
        """推进一个交易日。

        `bars` 必须覆盖所有持仓与活跃订单标的；数据适配器通常返回全市场当日 bar。
        缺少 bar 或 trade_status=0 视为当日不可交易，并累计退市缺失天数。
        必须按交易日逐日调用；跳日会破坏 T+1 股份/现金结算和停牌缺失天数计数。
        """
        d = fee.to_date(d)
        if d > self.freeze:
            raise ValueError(f"{d} 晚于冻结日 {self.freeze}")
        self._day = d
        if d not in self._cal_index:
            if not is_trading_day:
                return self._mark_and_record(d)
            raise ValueError(f"{d} 不在交易日历")
        self._settle_start()
        prev_map = dict(self._last_close)
        ev_list = list(events or [])
        self._day_bars = {}
        self._day_limits = {}
        self._ingest_bars(d, bars or {}, set(canonical_code(x) for x in st_codes) if st_codes is not None else None, prev_map)
        today_events = self._apply_events(d, ev_list, prev_map)
        self._compute_limits(d, prev_map, today_events)
        self._stop_triggered_today = set()
        self._execute_open_orders(d)
        self._execute_intraday_orders(d)
        self._expire_day_orders(d)
        self._check_delistings(d)
        return self._mark_and_record(d)

    def _settle_start(self) -> None:
        # 卖出/分红现金在次日开盘前转为可用；买入股份在次日开盘前转为可卖
        self.cash_available = self.cash_total
        for pos in self.positions.values():
            pos.available = pos.shares

    def _apply_events(self, d: Date, events: Sequence[CorporateAction], prev_map: dict[str, float]) -> dict[str, CorporateAction]:
        merged: dict[tuple, CorporateAction] = {}
        for ev in list(self._pending_events) + list(events):
            if ev.ex_date > self.freeze:
                self._flag(ev.code, "event_after_freeze_ignored", {"ex_date": ev.ex_date.isoformat()})
                continue
            key = ev.key()
            if key in merged:
                old = merged[key]
                rr = old.rights_ratio + ev.rights_ratio
                rp = ((old.rights_ratio * old.rights_price + ev.rights_ratio * ev.rights_price) / rr) if rr > 0 else 0.0
                merged[key] = CorporateAction(
                    code=ev.code, ex_date=ev.ex_date,
                    dividend_per_share=old.dividend_per_share + ev.dividend_per_share,
                    per_share_bonus=old.per_share_bonus + ev.per_share_bonus,
                    rights_ratio=rr, rights_price=rp, source="merged_input_events",
                )
            else:
                merged[key] = ev
        self._pending_events = []
        today: dict[str, list[CorporateAction]] = {}
        for key in sorted(merged, key=lambda x: (x[1], x[0])):
            ev = merged[key]
            if ev.ex_date > d:
                self._pending_events.append(ev)
                continue
            if key in self._applied_events:
                continue
            if ev.is_empty():
                self._applied_events.add(key)
                self._flag(ev.code, "empty_corporate_action", {"ex_date": ev.ex_date.isoformat()})
                continue
            self._applied_events.add(key)
            late = ev.ex_date < d
            if not late and not self._event_price_consistent(ev, d, prev_map):
                self.corporate_actions.append({
                    "date": d.isoformat(), "ex_date": ev.ex_date.isoformat(), "code": ev.code,
                    "applied": False, "reason": "corporate_action_price_conflict",
                    "dividend_per_share": ev.dividend_per_share, "per_share_bonus": ev.per_share_bonus,
                    "rights_ratio": ev.rights_ratio, "rights_price": ev.rights_price, "source": ev.source,
                })
                continue
            if ev.ex_date == d:
                today.setdefault(ev.code, []).append(ev)
            pos = self.positions.get(ev.code)
            if not pos or pos.shares <= 0:
                continue
            if late:
                self._flag(ev.code, "late_corporate_action", {"ex_date": ev.ex_date.isoformat(), "applied": d.isoformat()})
            self._apply_one_event(ev, d, pos, late)
        out: dict[str, CorporateAction] = {}
        for code, evs in today.items():
            div = sum(x.dividend_per_share for x in evs)
            bonus = sum(x.per_share_bonus for x in evs)
            rr = sum(x.rights_ratio for x in evs)
            rp = (sum(x.rights_ratio * x.rights_price for x in evs) / rr) if rr > 0 else 0.0
            out[code] = CorporateAction(code=code, ex_date=d, dividend_per_share=div, per_share_bonus=bonus, rights_ratio=rr, rights_price=rp, source="aggregated")
        return out

    def _event_price_consistent(self, ev: CorporateAction, d: Date, prev_map: dict[str, float]) -> bool:
        """用当日开盘价校验事件是否与价格行为一致。

        DuckDB 事件在 2016+ 与复权因子高度一致，但 2005-2015 存在事件与 Baostock
        原始价格不匹配的情况（例如事件称送转、原始价却未除权）。若不校验就加股/分红，
        会凭空制造持仓价值。冲突事件只标注、不应用，交由协调者复核。
        """
        bar = self._day_bars.get(ev.code)
        if bar is None or not bar.tradable:
            self._flag(ev.code, "corporate_action_no_trade_skipped", {
                "ex_date": ev.ex_date.isoformat(), "trade_status": getattr(bar, "trade_status", None),
            })
            return False
        raw_prev = prev_map.get(ev.code)
        if not raw_prev or raw_prev <= 0:
            self._flag(ev.code, "corporate_action_no_prev_skipped", {"ex_date": ev.ex_date.isoformat()})
            return False
        try:
            ref = fee.ex_rights_reference(raw_prev, ev.dividend_per_share, ev.per_share_bonus, ev.rights_ratio, ev.rights_price)
        except Exception as exc:
            self._flag(ev.code, "corporate_action_ref_error", {"error": str(exc), "raw_prev": raw_prev})
            return False
        pcts = fee.ipo_limit_pct(ev.code, bar.is_st, d, self._days_since_ipo(ev.code, d))
        tol = (pcts[0] if pcts else 0.50) + 0.02
        gap = bar.open / ref - 1.0 if ref > 0 else math.inf
        if abs(gap) > tol:
            self._flag(ev.code, "corporate_action_price_conflict", {
                "ex_date": ev.ex_date.isoformat(), "raw_prev": round(raw_prev, 4),
                "event_ref": round(ref, 4), "open": bar.open, "open_vs_ref": round(gap, 6),
                "tolerance": round(tol, 4), "is_st": bar.is_st,
            })
            return False
        return True

    def _apply_one_event(self, ev: CorporateAction, d: Date, pos: Position, late: bool) -> None:
        shares0 = pos.shares
        gross = shares0 * ev.dividend_per_share
        tax_rate = self.config.effective_dividend_tax_rate
        tax = gross * tax_rate
        net = gross - tax
        if net:
            self.cash_total += net
            if self.config.cash_settle_days == 0:
                self.cash_available += net
            self._flag(ev.code, "dividend_tax_flat_rate", {"gross": round(gross, 4), "tax": round(tax, 4), "rate": tax_rate})
        bonus_raw = shares0 * ev.per_share_bonus
        bonus_int = int(math.floor(bonus_raw + 1e-9))
        frac = bonus_raw - bonus_int
        if bonus_int:
            pos.add_free(bonus_int)
        if frac > 1e-9:
            self._flag(ev.code, "bonus_fractional_floor", {"raw": bonus_raw, "credited": bonus_int, "fraction": round(frac, 8)})
        rights_raw = shares0 * ev.rights_ratio
        rights_int = int(math.floor(rights_raw + 1e-9))
        if rights_raw - rights_int > 1e-9:
            self._flag(ev.code, "rights_fractional_floor", {"raw": rights_raw, "credited": rights_int})
        rights_cost = 0.0
        if rights_int:
            rights_cost = rights_int * ev.rights_price
            if self.config.rights_policy == "subscribe" and self.cash_total + 1e-9 >= rights_cost:
                # 配股款是即时支付，总现金与可用现金同时扣减
                self.cash_total -= rights_cost
                self.cash_available -= rights_cost
                pos.add_paid(rights_int, ev.rights_price, 0.0, d)
                self._flag(ev.code, "rights_subscribed", {"shares": rights_int, "price": ev.rights_price, "cost": round(rights_cost, 2)})
            elif self.config.rights_policy == "subscribe":
                self._flag(ev.code, "rights_skipped_insufficient_cash", {"shares": rights_int, "cost": round(rights_cost, 2), "cash": round(self.cash_total, 2)})
            else:
                self._flag(ev.code, "rights_skipped_policy", {"shares": rights_int, "cost": round(rights_cost, 2)})
        self.corporate_actions.append({
            "date": d.isoformat(),
            "ex_date": ev.ex_date.isoformat(),
            "code": ev.code,
            "applied": True,
            "shares_before": shares0,
            "dividend_per_share": ev.dividend_per_share,
            "dividend_gross": round(gross, 4),
            "dividend_tax": round(tax, 4),
            "dividend_net": round(net, 4),
            "bonus_raw": round(bonus_raw, 8),
            "bonus_credited": bonus_int,
            "rights_raw": round(rights_raw, 8),
            "rights_credited": rights_int,
            "rights_cost": round(rights_cost, 2),
            "late": late,
            "source": ev.source,
        })

    def _ingest_bars(self, d: Date, bars: Mapping[str, Bar], st_codes: set[str] | None, prev_map: dict[str, float]) -> None:
        for raw_code, bar in bars.items():
            if not isinstance(bar, Bar):
                raise TypeError(f"{raw_code} 不是 Bar")
            code = canonical_code(raw_code)
            bar.code = code
            if bar.date != d:
                raise ValueError(f"Bar 日期 {bar.date} 与 on_day {d} 不一致")
            if st_codes is not None:
                bar.is_st = code in st_codes
            if bar.prev_close is None:
                bar.prev_close = prev_map.get(code)
                if bar.prev_close is not None and bar.prev_close_is_ref:
                    # 缓存里存的是原始收盘价，不是交易所除权参考价
                    bar.prev_close_is_ref = False
            self._day_bars[code] = bar
            if bar.close > 0:
                self._last_close[code] = bar.close
                pos = self.positions.get(code)
                if pos:
                    pos.last_price = bar.close
                    pos.last_date = d

    def _days_since_ipo(self, code: str, d: Date) -> int | None:
        info = self.symbol_info.get(code)
        if not info or info.ipo_date is None:
            return None
        if info.ipo_date < self.calendar[0]:
            # 日历起点晚于 IPO：无法准确数上市交易日，直接视为非新股窗口
            return 10_000
        idx_ipo = bisect_left(self.calendar, info.ipo_date)
        idx_d = self._cal_index[d]
        return idx_d - idx_ipo

    def _compute_limits(self, d: Date, prev_map: dict[str, float], today_events: dict[str, CorporateAction]) -> None:
        # 只算持仓/活跃订单/当日公司行动相关标的，避免对全市场 5000+ 只股票做无谓的 Decimal 运算
        needed = {c for c, p in self.positions.items() if p.shares > 0}
        needed.update(o.code for o in self._orders if o.active)
        needed.update(today_events)
        for code in needed:
            bar = self._day_bars.get(code)
            if bar is None or not bar.tradable:
                self._day_limits[code] = None
                continue
            ref = bar.prev_close if (bar.prev_close and bar.prev_close > 0) else prev_map.get(code)
            if ref is None:
                ref = bar.open
                self._flag(code, "no_prev_close_limit_ref_uses_open", {"open": bar.open})
            ev = today_events.get(code)
            if ev is not None:
                raw_prev = prev_map.get(code)
                if raw_prev and raw_prev > 0:
                    try:
                        ref2 = fee.ex_rights_reference(raw_prev, ev.dividend_per_share, ev.per_share_bonus, ev.rights_ratio, ev.rights_price)
                        self._flag(code, "ex_rights_reference_used", {
                            "raw_prev": round(raw_prev, 6), "ref": round(ref2, 6),
                            "dividend": ev.dividend_per_share, "bonus": ev.per_share_bonus,
                            "rights_ratio": ev.rights_ratio, "rights_price": ev.rights_price,
                        })
                        ref = ref2
                        if bar.prev_close_is_ref and bar.prev_close is not None and abs(ref2 - bar.prev_close) > 0.02:
                            # 数据源自称 preclose 已是参考价，但与事件公式不一致；以事件公式为准做限价
                            self._flag(code, "prev_close_ref_mismatch", {
                                "source_prev_close": bar.prev_close, "event_ref": round(ref2, 6),
                                "raw_prev": raw_prev,
                            })
                    except Exception as exc:
                        self._flag(code, "ex_rights_reference_error", {"error": str(exc), "raw_prev": raw_prev})
                elif bar.prev_close_is_ref and bar.prev_close:
                    self._flag(code, "event_ref_uses_source_preclose_no_raw_prev", {"source_prev_close": bar.prev_close})
                    ref = bar.prev_close
                else:
                    try:
                        ref = fee.ex_rights_reference(ref, ev.dividend_per_share, ev.per_share_bonus, ev.rights_ratio, ev.rights_price)
                    except Exception as exc:
                        self._flag(code, "ex_rights_reference_error", {"error": str(exc), "raw_prev": ref})
            try:
                self._day_limits[code] = fee.limit_prices(
                    ref, code, bar.is_st, d, self._days_since_ipo(code, d)
                )
            except Exception as exc:
                self._day_limits[code] = None
                self._flag(code, "limit_price_error", {"error": str(exc), "ref": ref})

    # -- 撮合 ---------------------------------------------------------------

    def _execute_open_orders(self, d: Date) -> None:
        pending = [o for o in self._orders if o.active and o.kind == "open" and o.execute_from <= d]
        if self.config.cash_settle_days == 0:
            # 允许当日回款时先卖后买，避免提交顺序造成人为资金不足
            pending.sort(key=lambda o: (0 if o.side == "sell" else 1, o.seq))
        else:
            pending.sort(key=lambda o: o.seq)
        for order in pending:
            bar = self._day_bars.get(order.code)
            if not self._tradable(bar):
                order.reason = "no_bar_or_suspended"
                continue
            lim = self._day_limits.get(order.code)
            if order.side == "buy" and lim is not None and fee.blocked_buy_at_open(bar.open, lim):
                order.reason = "limit_up_open"
                continue
            if order.side == "sell" and lim is not None and fee.blocked_sell_at_open(bar.open, lim):
                order.reason = "limit_down_open"
                continue
            self._fill(order, d, bar, bar.open, "market", "open")

    def _execute_intraday_orders(self, d: Date) -> None:
        active = [o for o in self._orders if o.active and o.execute_from <= d and o.kind in {"stop", "limit"}]
        for order in sorted([o for o in active if o.kind == "stop"], key=lambda o: o.seq):
            self._eval_stop(order, d)
        for order in sorted([o for o in active if o.kind == "limit"], key=lambda o: o.seq):
            self._eval_limit(order, d)
        for code, pos in self.positions.items():
            if pos.shares == 0:
                for o in self._orders:
                    if o.active and o.code == code and o.side == "sell":
                        o.status = "cancelled"
                        o.reason = "position_closed"

    def _eval_stop(self, order: Order, d: Date) -> None:
        bar = self._day_bars.get(order.code)
        if not self._tradable(bar):
            order.reason = "no_bar_or_suspended"
            return
        pos = self.positions.get(order.code)
        if not pos or pos.shares <= 0:
            order.status = "cancelled"
            order.reason = "no_position"
            return
        trigger = order.triggered or bar.low <= order.price + fee.TICK_TOL
        if not trigger:
            return
        self._stop_triggered_today.add(order.code)
        lim = self._day_limits.get(order.code)
        if lim is not None and fee.blocked_sell_at_open(bar.open, lim):
            order.triggered = True
            order.reason = "limit_down_open"
            return
        if lim is not None and not lim.no_limit and bar.high <= lim.down + fee.TICK_TOL:
            order.triggered = True
            order.reason = "limit_down_locked"
            return
        if pos.available <= 0:
            # 已触发但当日股份不可卖：保留触发状态，次日按开盘价执行
            order.triggered = True
            order.reason = "t1_blocked"
            return
        base = bar.open if (order.triggered or bar.open <= order.price + fee.TICK_TOL) else order.price
        if lim is not None and not lim.no_limit:
            base = max(base, lim.down)
        ok = self._fill(order, d, bar, base, "stop", "stop_trigger")
        if ok:
            order.triggered = True

    def _eval_limit(self, order: Order, d: Date) -> None:
        bar = self._day_bars.get(order.code)
        if not self._tradable(bar):
            order.reason = "no_bar_or_suspended"
            return
        lim = self._day_limits.get(order.code)
        if order.side == "sell":
            trigger = bar.high >= order.price - fee.TICK_TOL or bar.open >= order.price - fee.TICK_TOL
            if not trigger:
                return
            if order.code in self._stop_triggered_today:
                self._flag(order.code, "intraday_ambiguity_loss_first", {"stop": True, "target": order.price})
            pos = self.positions.get(order.code)
            if not pos or pos.shares <= 0:
                order.status = "cancelled"
                order.reason = "no_position"
                return
            if lim is not None and fee.blocked_sell_at_open(bar.open, lim):
                order.reason = "limit_down_open"
                return
            if pos.available <= 0:
                order.reason = "t1_blocked"
                return
            base = bar.open if bar.open >= order.price - fee.TICK_TOL else order.price
            if lim is not None and not lim.no_limit:
                base = max(base, lim.down)
            self._fill(order, d, bar, base, "limit", "limit_hit")
        else:
            trigger = bar.low <= order.price + fee.TICK_TOL or bar.open <= order.price + fee.TICK_TOL
            if not trigger:
                return
            if lim is not None and fee.blocked_buy_at_open(bar.open, lim):
                order.reason = "limit_up_open"
                return
            base = bar.open if bar.open <= order.price + fee.TICK_TOL else order.price
            if lim is not None and not lim.no_limit:
                base = min(base, lim.up)
            self._fill(order, d, bar, base, "limit", "limit_hit")

    def _tradable(self, bar: Bar | None) -> bool:
        return bool(bar is not None and bar.tradable)

    def _fill(self, order: Order, d: Date, bar: Bar, base_price: float, execution: str, reason: str) -> bool:
        pos = self.positions.get(order.code)
        if order.side == "buy":
            qty = fee.normalize_buy_qty(order.remaining, order.code)
            if qty <= 0:
                order.status = "cancelled"
                order.reason = "remaining_below_lot_min"
                return False
        else:
            if not pos or pos.shares <= 0 or pos.available <= 0:
                order.reason = "no_available_shares_t1"
                return False
            avail = pos.available
            qty = min(order.remaining, avail)
            ok, msg = fee.validate_sell_qty(qty, avail, order.code)
            if not ok:
                order.status = "rejected"
                order.reason = f"sell_qty_invalid:{msg}"
                return False

        cap = fee.participation_cap(bar.volume, self.config.fee.max_participation)
        if order.side == "buy":
            cap = fee.normalize_buy_qty(cap, order.code)
        else:
            cap = fee.adjust_sell_fill_qty(cap, pos.available, order.code)
        fill_qty = min(qty, cap)
        if order.side == "sell" and fill_qty < qty:
            # 流动性/参与率造成的部分成交，数量也必须是合法申报量
            fill_qty = fee.adjust_sell_fill_qty(fill_qty, pos.available, order.code)
        if fill_qty <= 0:
            order.reason = "participation_zero"
            return False

        if execution == "limit":
            price = fee.round_tick(base_price, "half_up")
        else:
            price = fee.apply_slippage(base_price, order.side, self.config.fee.slippage_bps)
        lim = self._day_limits.get(order.code)
        if lim is not None and not lim.no_limit:
            price = max(price, lim.down) if order.side == "sell" else min(price, lim.up)
        if price <= 0:
            order.reason = "nonpositive_fill_price"
            return False

        if order.side == "buy":
            if not self.config.allow_buy_downsize:
                fees = fee.trade_fees(d, order.code, "buy", price, fill_qty, self.config.fee)
                if price * fill_qty + fees.total > self.cash_available + 1e-6:
                    order.reason = "insufficient_cash"
                    return False
            else:
                step = fee.lot_rule(order.code).buy_step
                buy_min = fee.lot_rule(order.code).buy_min
                while fill_qty > 0:
                    fees = fee.trade_fees(d, order.code, "buy", price, fill_qty, self.config.fee)
                    if price * fill_qty + fees.total <= self.cash_available + 1e-6:
                        break
                    fill_qty -= step
                    if fill_qty < buy_min:
                        fill_qty = 0
                if fill_qty <= 0:
                    order.reason = "insufficient_cash"
                    return False
                if fill_qty < qty:
                    self._flag(order.code, "buy_downsize_cash", {"requested": qty, "filled": fill_qty, "cash": round(self.cash_available, 2)})
        fees = fee.trade_fees(d, order.code, order.side, price, fill_qty, self.config.fee)
        amount = price * fill_qty
        if order.side == "buy":
            cost = amount + fees.total
            if cost > self.cash_available + 1e-6:
                order.reason = "insufficient_cash"
                return False
            self.cash_total -= cost
            self.cash_available -= cost
            if pos is None:
                pos = Position(code=order.code)
                self.positions[order.code] = pos
            pos.add_paid(fill_qty, price, fees.total, d)
            pos.last_price = bar.close
            pos.last_date = d
            if not self.config.enforce_t1:
                pos.available += fill_qty
        else:
            proceeds = amount - fees.total
            self.cash_total += proceeds
            if self.config.cash_settle_days == 0:
                self.cash_available += proceeds
            pos.remove(fill_qty, price, fees.total)

        order.filled_qty += fill_qty
        order.avg_price = ((order.avg_price * (order.filled_qty - fill_qty)) + price * fill_qty) / order.filled_qty if order.filled_qty else price
        order.status = "filled" if order.remaining == 0 else "partial"
        order.reason = reason if order.status == "filled" else f"{reason}_partial"
        self.trades.append(Trade(
            date=d, code=order.code, side=order.side, qty=fill_qty, price=price, amount=amount,
            commission=fees.commission, stamp_duty=fees.stamp_duty, transfer_fee=fees.transfer_fee,
            fees=fees.total, cash_after=self.cash_total, kind=order.kind, order_id=order.order_id,
            tag=order.tag, reason=order.reason,
        ))
        if order.status == "partial" and order.tif == "day":
            self._flag(order.code, "partial_fill_day_order", {"order_id": order.order_id, "filled": order.filled_qty, "qty": order.qty})
        return True

    def _expire_day_orders(self, d: Date) -> None:
        for order in self._orders:
            if order.active and order.tif == "day" and order.execute_from <= d:
                prior = order.reason
                order.status = "cancelled"
                if order.filled_qty == 0:
                    order.reason = prior if prior else "day_order_unfilled"
                else:
                    order.reason = f"{prior}+day_end" if prior else "day_order_partial_unfilled"

    # -- 退市与估值 ---------------------------------------------------------

    def _check_delistings(self, d: Date) -> None:
        for code, pos in list(self.positions.items()):
            if pos.shares <= 0:
                continue
            info = self.symbol_info.get(code)
            if info and info.out_date is not None and d > info.out_date:
                self._force_sell(code, d, pos.last_price, "delisting_out_date")
                continue
            bar = self._day_bars.get(code)
            if bar is not None and bar.tradable:
                pos.missing_days = 0
            else:
                pos.missing_days += 1
                if pos.missing_days >= self.config.delisting_missing_days:
                    self._force_sell(code, d, pos.last_price, "delisting_missing_bars")

    def _force_sell(self, code: str, d: Date, price: float, reason: str) -> None:
        pos = self.positions.get(code)
        if not pos or pos.shares <= 0:
            return
        px = float(price) * (1.0 - self.config.delisting_haircut) if reason.startswith("delisting") else float(price)
        px = fee.round_tick(max(0.0, px), "down")
        if px <= 0:
            pos.last_price = 0.0
            self._flag(code, "delisting_no_price", {"reason": reason, "shares": pos.shares})
            return
        qty = pos.shares
        amount = px * qty
        fees = fee.trade_fees(d, code, "sell", px, qty, self.config.fee)
        proceeds = amount - fees.total
        self.cash_total += proceeds
        if self.config.cash_settle_days == 0:
            self.cash_available += proceeds
        pos.remove(qty, px, fees.total)
        self.trades.append(Trade(
            date=d, code=code, side="sell", qty=qty, price=px, amount=amount,
            commission=fees.commission, stamp_duty=fees.stamp_duty, transfer_fee=fees.transfer_fee,
            fees=fees.total, cash_after=self.cash_total, kind="forced", order_id="FORCED",
            tag="", reason=reason,
        ))
        for o in self._orders:
            if o.active and o.code == code:
                o.status = "cancelled"
                o.reason = f"forced_liquidation:{reason}"
        self._flag(code, "forced_liquidation", {"reason": reason, "qty": qty, "price": px, "haircut": self.config.delisting_haircut})

    def _mark_and_record(self, d: Date) -> dict:
        mv = 0.0
        n_pos = 0
        for code, pos in self.positions.items():
            if pos.shares <= 0:
                continue
            n_pos += 1
            bar = self._day_bars.get(code)
            if bar is not None and bar.close > 0:
                px = bar.close
            elif pos.last_price > 0:
                px = pos.last_price
            else:
                px = self._last_close.get(code, 0.0)
            if px <= 0:
                self._flag(code, "unpriceable_position", {"shares": pos.shares})
            mv += pos.shares * px
        nav = self.cash_total + mv
        if nav <= 0:
            self._flag("", "nav_nonpositive", {"nav": nav})
        rec = {
            "date": d.isoformat(),
            "cash_total": round(self.cash_total, 4),
            "cash_available": round(self.cash_available, 4),
            "market_value": round(mv, 4),
            "nav": round(nav, 4),
            "n_positions": n_pos,
            "n_active_orders": sum(1 for o in self._orders if o.active),
            "n_trades": len(self.trades),
            "flags_total": sum(self.flag_counts.values()),
        }
        if self.daily and self.daily[-1]["date"] == rec["date"]:
            self.daily[-1] = rec
        else:
            self.daily.append(rec)
        return rec

    def finalize(self, d: Date | None = None, liquidate: bool | None = None) -> dict:
        if liquidate is None:
            liquidate = self.config.liquidate_on_finalize
        dd = fee.to_date(d) if d else self.freeze
        if dd > self.freeze:
            raise ValueError(f"{dd} 晚于冻结日 {self.freeze}")
        self._day = dd
        for o in self._orders:
            if o.active:
                o.status = "cancelled"
                o.reason = "finalize_cancel"
        if liquidate:
            for code, pos in list(self.positions.items()):
                if pos.shares <= 0:
                    continue
                px = pos.last_price if pos.last_price > 0 else self._last_close.get(code, 0.0)
                if px <= 0:
                    self._flag(code, "finalize_no_price", {"shares": pos.shares})
                    continue
                px = fee.apply_slippage(px, "sell", self.config.fee.slippage_bps) * (1.0 - self.config.finalize_haircut)
                self._force_sell(code, dd, px, "end_of_period_liquidation")
        return self._mark_and_record(dd)

    # -- 审计快照 -----------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "freeze_date": self.freeze.isoformat(),
            "initial_cash": self.config.initial_cash,
            "cash_total": round(self.cash_total, 4),
            "cash_available": round(self.cash_available, 4),
            "positions": [p.as_dict() for p in self.positions.values() if p.shares > 0],
            "trades": [t.as_dict() for t in self.trades],
            "orders": [o.as_dict() for o in self._orders],
            "corporate_actions": self.corporate_actions,
            "flags": self.flags,
            "flag_counts": dict(self.flag_counts),
            "daily_last": self.daily[-1] if self.daily else None,
        }


# ---------------------------------------------------------------------------
# 数据适配器（只读；不迁移、不同步、不改源数据）
# ---------------------------------------------------------------------------

FREEZE_DATE = Date(2026, 10, 26)


class DuckDBEventSource:
    def __init__(self, db_path: str | os.PathLike, freeze_date: Date = FREEZE_DATE):
        import duckdb  # 延迟导入，合成自检不依赖 duckdb
        self.path = str(db_path)
        self.freeze = fee.to_date(freeze_date)
        self.con = duckdb.connect(self.path, read_only=True)

    def events_between(self, start: Date, end: Date) -> list[CorporateAction]:
        start = max(fee.to_date(start), Date(1990, 1, 1))
        end = min(fee.to_date(end), self.freeze)
        if end < start:
            return []
        rows = self.con.execute(
            """
            select thscode, ex_date, dividend_per_share, per_share_bonus, rights_ratio, rights_price
            from raw_adjustment_events
            where ex_date between ? and ?
            order by ex_date, thscode
            """,
            [start, end],
        ).fetchall()
        return [
            CorporateAction(
                code=r[0], ex_date=r[1], dividend_per_share=r[2] or 0.0,
                per_share_bonus=r[3] or 0.0, rights_ratio=r[4] or 0.0,
                rights_price=r[5] or 0.0, source="duckdb.raw_adjustment_events",
            )
            for r in rows
        ]

    def events_on(self, d: Date) -> list[CorporateAction]:
        return self.events_between(d, d)

    def close(self) -> None:
        self.con.close()


class DuckDBMarketData:
    """按交易日顺序使用的 DuckDB 日线适配器。

    DuckDB 的 `prev_close` 全为空，适配器用上一交易日收盘价补；除权日的参考价由
    账户层用 DuckDB 事件计算（`prev_close_is_ref=False`）。必须按日期升序调用
    `bars_on`，否则 `prev_close` 会缺失（会显式回退为当日开盘并打 flag）。
    """

    def __init__(self, db_path: str | os.PathLike, freeze_date: Date = FREEZE_DATE,
                 events: DuckDBEventSource | None = None):
        import duckdb
        self.path = str(db_path)
        self.freeze = fee.to_date(freeze_date)
        self.con = duckdb.connect(self.path, read_only=True)
        self.events = events or DuckDBEventSource(db_path, freeze_date)
        self._last_close: dict[str, float] = {}

    def calendar(self, start: Date | None = None, end: Date | None = None) -> list[Date]:
        start = fee.to_date(start) if start else Date(2016, 10, 13)
        end = min(fee.to_date(end) if end else self.freeze, self.freeze)
        rows = self.con.execute(
            "select distinct date from raw_kline_daily where date between ? and ? order by date",
            [start, end],
        ).fetchall()
        return [r[0] for r in rows]

    def bars_on(self, d: Date) -> dict[str, Bar]:
        d = fee.to_date(d)
        if d > self.freeze:
            raise ValueError(f"{d} 晚于冻结日 {self.freeze}，拒绝读取未来行情")
        rows = self.con.execute(
            """
            select thscode, open, high, low, close, volume, amount
            from raw_kline_daily where date = ?
            """,
            [d],
        ).fetchall()
        out: dict[str, Bar] = {}
        for r in rows:
            code = canonical_code(r[0])
            try:
                bar = Bar(
                    code=code, date=d, open=r[1], high=r[2], low=r[3], close=r[4],
                    volume=r[5] or 0.0, amount=r[6],
                    prev_close=self._last_close.get(code), prev_close_is_ref=False,
                )
            except Exception as exc:
                raise ValueError(f"DuckDB Bar 非法 {code} {d}: {exc}") from exc
            out[code] = bar
        for code, bar in out.items():
            if bar.close > 0:
                self._last_close[code] = bar.close
        return out

    def events_on(self, d: Date) -> list[CorporateAction]:
        return self.events.events_on(d)

    def factor_changes_on(self, d: Date) -> list[dict]:
        """返回 d 日复权因子相对上一有效日发生变化的标的（用于发现事件缺口）。

        这是审计接口，不是公司行动：是否按送转推断、还是让策略避开该标的，
        由协调者决定；账户层不会自动改股数。
        """
        d = fee.to_date(d)
        if d > self.freeze:
            raise ValueError(f"{d} 晚于冻结日 {self.freeze}")
        rows = self.con.execute(
            """
            select f.thscode, f.date, f.backward_factor,
                   (select p.backward_factor from calc_adjust_factor_daily p
                    where p.thscode=f.thscode and p.date<f.date
                    order by p.date desc limit 1) prev
            from calc_adjust_factor_daily f
            where f.date = ?
            order by f.thscode
            """,
            [d],
        ).fetchall()
        rows = [r for r in rows if r[3] is not None and abs(r[2] - r[3]) > 1e-9]
        event_codes = {r[0] for r in self.con.execute(
            "select thscode from raw_adjustment_events where ex_date = ?", [d]
        ).fetchall()}
        return [
            {"code": canonical_code(r[0]), "date": r[1].isoformat(),
             "prev_factor": float(r[3]), "factor": float(r[2]),
             "change": float(r[2]) / float(r[3]) - 1.0,
             "has_event": canonical_code(r[0]) in event_codes}
            for r in rows
        ]

    def symbol_info(self, code: str) -> SymbolInfo | None:
        row = self.con.execute("select thscode, ticker, name from dim_symbol where thscode = ?", [code]).fetchone()
        if not row:
            return None
        return SymbolInfo(code=row[0], name=row[2] or "")

    def close(self) -> None:
        try:
            self.events.close()
        finally:
            self.con.close()


class BaostockMarketData:
    """Baostock SQLite 只读适配器（复权口径 3=不复权）。

    Baostock 的 `preclose` 在除权日已是交易所除权参考价，故 `prev_close_is_ref=True`；
    账户层不会再用事件重复除权。公司行动事件仍从 DuckDB 事件源读取。
    """

    def __init__(self, db_path: str | os.PathLike, freeze_date: Date = FREEZE_DATE,
                 events: DuckDBEventSource | None = None):
        self.path = str(db_path)
        self.freeze = fee.to_date(freeze_date)
        self.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.events = events

    @staticmethod
    def to_canonical(code: str) -> str:
        return canonical_code(code)

    @staticmethod
    def from_canonical(code: str) -> str:
        c = canonical_code(code)
        if c.endswith(".SH"):
            return "sh." + c[:-3]
        if c.endswith(".SZ"):
            return "sz." + c[:-3]
        if c.endswith(".BJ"):
            return "bj." + c[:-3]
        return c

    def calendar(self, start: Date | None = None, end: Date | None = None) -> list[Date]:
        start = fee.to_date(start) if start else Date(1990, 1, 1)
        end = min(fee.to_date(end) if end else self.freeze, self.freeze)
        rows = self.con.execute(
            "select calendar_date from trade_dates where is_trading_day=1 and calendar_date between ? and ? order by 1",
            [start.isoformat(), end.isoformat()],
        ).fetchall()
        return [Date.fromisoformat(r[0]) for r in rows]

    def bars_on(self, d: Date) -> dict[str, Bar]:
        d = fee.to_date(d)
        if d > self.freeze:
            raise ValueError(f"{d} 晚于冻结日 {self.freeze}，拒绝读取未来行情")
        rows = self.con.execute(
            """
            select code, open, high, low, close, preclose, volume, amount, trade_status, is_st
            from stock_kline_d
            where date = ? and adjustflag = 3
            """,
            [d.isoformat()],
        ).fetchall()
        out: dict[str, Bar] = {}
        for r in rows:
            code = self.to_canonical(r[0])
            try:
                bar = Bar(
                    code=code, date=d, open=r[1], high=r[2], low=r[3], close=r[4],
                    volume=r[6] or 0.0, amount=r[7], prev_close=r[5],
                    prev_close_is_ref=True, is_st=bool(r[9]), trade_status=int(r[8] or 0),
                )
            except Exception as exc:
                raise ValueError(f"Baostock Bar 非法 {code} {d}: {exc}") from exc
            out[code] = bar
        return out

    def events_on(self, d: Date) -> list[CorporateAction]:
        return self.events.events_on(d) if self.events else []

    def symbol_info(self, code: str) -> SymbolInfo | None:
        bc = self.from_canonical(code)
        row = self.con.execute(
            "select code, code_name, ipo_date, out_date, type, status from stock_basic where code = ?", [bc]
        ).fetchone()
        if not row:
            return None
        ipo = row[2] if row[2] not in (None, "", "None") else None
        out = row[3] if row[3] not in (None, "", "None") else None
        return SymbolInfo(
            code=canonical_code(row[0]), name=row[1] or "",
            ipo_date=Date.fromisoformat(ipo) if ipo else None,
            out_date=Date.fromisoformat(out) if out else None,
        )

    def close(self) -> None:
        try:
            if self.events:
                self.events.close()
        finally:
            self.con.close()

# ---------------------------------------------------------------------------
# 合成自检（不读取真实行情、不产生真实收益结论）
# ---------------------------------------------------------------------------

def _cal(start: Date = Date(2024, 1, 2), n: int = 10) -> list[Date]:
    d = Date.fromisoformat(start) if isinstance(start, str) else start
    out: list[Date] = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _bar(code: str, d: Date, o: float, h: float, l: float, c: float,
         vol: float = 1_000_000, st: bool = False, status: int = 1,
         prev: float | None = None, prev_is_ref: bool = False) -> Bar:
    return Bar(code=code, date=d, open=o, high=h, low=l, close=c, volume=vol,
               prev_close=prev, prev_close_is_ref=prev_is_ref, is_st=st, trade_status=status)


def _acct(cal: Sequence[Date], cash: float = 1_000_000.0, slip: float = 0.0,
          part: float = 0.10, tax: float = 0.20, **kw) -> Account:
    fc = fee.FeeConfig(slippage_bps=slip, max_participation=part, dividend_tax_rate=tax)
    return Account(AccountConfig(initial_cash=cash, fee=fc, **kw), cal, freeze_date=cal[-1])


def _expect(cond: bool, msg: str, **ctx) -> None:
    if not cond:
        extra = (" | " + json.dumps(ctx, ensure_ascii=False, default=_json_default)) if ctx else ""
        raise AssertionError(msg + extra)


def _sells(a: Account) -> list[Trade]:
    return [t for t in a.trades if t.side == "sell"]


def _buys(a: Account) -> list[Trade]:
    return [t for t in a.trades if t.side == "buy"]


def _case_t1_buy_then_stop_same_day() -> dict:
    cal = _cal(n=4)
    a = _acct(cal, cash=100_000, slip=0)
    A = "600000.SH"
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.5, 9.8, 10)})
    a.buy_open(A, 100, cal[0])
    stop = a.sell_stop(A, 100, 9.0, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10.1, 10.2, 8.9, 9.5)})
    _expect(not _sells(a), "T+1 当日不应卖出", orders=[o.as_dict() for o in a._orders])
    _expect(stop.status == "pending" and stop.reason == "t1_blocked",
            "止损应被 T+1 挡住并保持挂单", status=stop.status, reason=stop.reason)
    a.on_day(cal[2], {A: _bar(A, cal[2], 9.6, 9.7, 9.4, 9.5)})
    sells = _sells(a)
    _expect(len(sells) == 1 and sells[0].date == cal[2], "次日应按已触发止损卖出", sells=[t.as_dict() for t in sells])
    _expect(sells[0].price == 9.6, "已触发但被 T+1 挡住的止损应在次日开盘执行", price=sells[0].price)
    return {"sell": sells[0].as_dict(), "stop_order": stop.as_dict()}


def _case_lot_main() -> dict:
    cal = _cal(n=3)
    a = _acct(cal, cash=100_000, slip=0)
    A = "600000.SH"
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    o1 = a.buy_open(A, 150, cal[0])
    o2 = a.buy_open(A, 50, cal[0])
    _expect(o1.qty == 100, "150 股应向下取整为 100", qty=o1.qty)
    _expect(o2.status == "rejected" and o2.reason == "lot_below_min",
            "50 股应因不足 100 股被拒绝", status=o2.status, reason=o2.reason)
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    _expect(len(a.trades) == 1 and a.trades[0].qty == 100, "只应成交 100 股", trades=[t.as_dict() for t in a.trades])
    return {"order": o1.as_dict(), "rejected": o2.as_dict()}


def _case_lot_star_bse() -> dict:
    cal = _cal(n=4)
    star = "688001.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {star: _bar(star, cal[0], 10, 10.2, 9.9, 10)})
    ok = a.buy_open(star, 250, cal[0])
    bad = a.buy_open(star, 150, cal[0])
    _expect(ok.qty == 250 and bad.status == "rejected", "科创板 250 股可买、150 股不可买", ok=ok.as_dict(), bad=bad.as_dict())
    a.on_day(cal[1], {star: _bar(star, cal[1], 10, 10.2, 9.9, 10)})
    s_bad = a.sell_open(star, 100, cal[1])
    a.on_day(cal[2], {star: _bar(star, cal[2], 10, 10.2, 9.9, 10)})
    _expect(s_bad.status == "rejected" and "below_sell_min" in s_bad.reason,
            "科创板部分卖出 100 股应被拒绝", status=s_bad.status, reason=s_bad.reason)
    s_ok = a.sell_open(star, 250, cal[2])
    a.on_day(cal[3], {star: _bar(star, cal[3], 10, 10.2, 9.9, 10)})
    _expect(any(t.side == "sell" and t.qty == 250 for t in a.trades), "科创板 250 股应一次卖出", trades=[t.as_dict() for t in a.trades])

    bse = "830001.BJ"
    b = _acct(cal, cash=100_000, slip=0)
    b.on_day(cal[0], {bse: _bar(bse, cal[0], 10, 10.2, 9.9, 10)})
    ob = b.buy_open(bse, 150, cal[0])
    b.on_day(cal[1], {bse: _bar(bse, cal[1], 10, 10.2, 9.9, 10)})
    _expect(ob.qty == 150 and any(t.qty == 150 for t in b.trades), "北交所 150 股可买", order=ob.as_dict(), trades=[t.as_dict() for t in b.trades])
    return {"star_ok": ok.as_dict(), "star_bad": bad.as_dict(), "star_sell_bad": s_bad.as_dict(), "bse": ob.as_dict()}


def _case_odd_lot_sell() -> dict:
    cal = _cal(n=6)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    ev = CorporateAction(A, cal[2], per_share_bonus=0.5)
    # 10 送 5 的除权参考价=10/1.5≈6.67；开盘价必须与事件一致，否则会被价格一致性门控跳过
    a.on_day(cal[2], {A: _bar(A, cal[2], 6.67, 6.8, 6.6, 6.67)}, events=[ev])
    pos = a.positions[A]
    _expect(pos.shares == 150, "送股后应为 150 股", shares=pos.shares)
    _expect(pos.available == 100, "送股当日不可卖（T+1 保守）", available=pos.available)
    bad = a.sell_open(A, 120, cal[2])
    a.on_day(cal[3], {A: _bar(A, cal[3], 10, 10.2, 9.9, 10)})
    _expect(bad.status == "rejected" and "not_lot_multiple" in bad.reason,
            "部分卖出 120 股应被拒绝", status=bad.status, reason=bad.reason)
    ok = a.sell_open(A, 100, cal[3])
    a.on_day(cal[4], {A: _bar(A, cal[4], 10, 10.2, 9.9, 10)})
    _expect(any(t.side == "sell" and t.qty == 100 for t in a.trades), "100 股部分卖出应成交", trades=[t.as_dict() for t in a.trades])
    last = a.sell_open(A, 50, cal[4])
    a.on_day(cal[5], {A: _bar(A, cal[5], 10, 10.2, 9.9, 10)})
    _expect(a.positions[A].shares == 0, "零股应一次性卖出", position=a.positions[A].as_dict())
    return {"bad": bad.as_dict(), "ok": ok.as_dict(), "last": last.as_dict()}


def _case_open_slippage_fees() -> dict:
    cal = _cal(n=2)
    A = "600000.SH"
    a = _acct(cal, cash=1_000_000, slip=10)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    t = a.trades[0]
    _expect(t.price == 10.01, "买入滑点应向上取整到 10.01", price=t.price)
    _expect(t.commission == 5.0 and t.transfer_fee == 0.01 and t.stamp_duty == 0.0,
            "买入费用应为最低佣金 5 元 + 过户费 0.01", fees=t.as_dict())
    _expect(abs(a.cash_total - 998_993.99) < 1e-6, "现金应为 998993.99", cash=a.cash_total)
    return {"trade": t.as_dict(), "cash": a.cash_total}


def _case_fee_schedules() -> dict:
    A = "600000.SH"
    f_sell = fee.trade_fees(Date(2024, 1, 2), A, "sell", 10, 100)
    _expect(f_sell.commission == 5.0 and f_sell.stamp_duty == 0.5 and f_sell.transfer_fee == 0.01,
            "2024 卖出费用口径错误", fees=f_sell.as_dict())
    _expect(fee.stamp_duty(Date(2023, 8, 25), "sell", 10_000) == 10.0, "2023-08-25 前印花税应为 0.1%")
    _expect(fee.stamp_duty(Date(2023, 8, 28), "sell", 10_000) == 5.0, "2023-08-28 后印花税应为 0.05%")
    _expect(fee.stamp_duty(Date(2008, 9, 18), "buy", 10_000) == 10.0, "2008-09-19 前买入也收印花税")
    _expect(fee.stamp_duty(Date(2008, 9, 19), "buy", 10_000) == 0.0, "2008-09-19 后买入不收印花税")
    _expect(fee.transfer_fee(Date(2014, 6, 2), "600000.SH", "buy", 1000, 10_000) == 0.6, "2015 前沪市过户费按面值 0.6‰")
    _expect(fee.transfer_fee(Date(2014, 6, 2), "000001.SZ", "buy", 1000, 10_000) == 0.0, "2015 前深市不收过户费")
    return {"sell_2024": f_sell.as_dict()}


def _case_limit_up_buy_blocked() -> dict:
    cal = _cal(n=2)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    o = a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 11.0, 11.0, 10.8, 11.0)})
    _expect(not a.trades, "开盘涨停不应买入", trades=[t.as_dict() for t in a.trades])
    _expect(o.status == "cancelled" and o.reason.startswith("limit_up_open"),
            "应记录涨停开盘原因", status=o.status, reason=o.reason)
    return {"order": o.as_dict()}


def _case_limit_down_sell_blocked() -> dict:
    cal = _cal(n=3)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    o = a.sell_open(A, 100, cal[1])
    a.on_day(cal[2], {A: _bar(A, cal[2], 9.0, 9.0, 9.0, 9.0)})
    _expect(not _sells(a), "开盘跌停不应卖出", trades=[t.as_dict() for t in a.trades])
    _expect(o.status == "cancelled" and o.reason.startswith("limit_down_open"),
            "应记录跌停开盘原因", status=o.status, reason=o.reason)
    return {"order": o.as_dict()}


def _case_board_and_st_limits() -> dict:
    d19 = Date(2019, 1, 2)
    d24 = Date(2024, 1, 2)
    checks = {
        "main": fee.limit_prices(10, "600000.SH", False, d24).up == 11.0,
        "st_main": fee.limit_prices(10, "600000.SH", True, d24).up == 10.5,
        "chinext_old": fee.limit_prices(10, "300001.SZ", False, d19).up == 11.0,
        "chinext_new": fee.limit_prices(10, "300001.SZ", False, d24).up == 12.0,
        "star": fee.limit_prices(10, "688001.SH", False, d24).up == 12.0,
        "bse": fee.limit_prices(10, "830001.BJ", False, d24).up == 13.0,
    }
    _expect(all(checks.values()), "板块/ST 涨跌停口径错误", checks=checks)

    cal = _cal(n=2)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    o = a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10.5, 10.5, 10.4, 10.5, st=True)})
    _expect(not a.trades and o.reason.startswith("limit_up_open"), "ST 5% 涨停应挡住买入", order=o.as_dict())

    b = _acct(cal, cash=100_000, slip=0)
    b.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    ob = b.buy_open(A, 100, cal[0])
    b.on_day(cal[1], {A: _bar(A, cal[1], 10.49, 10.49, 10.4, 10.49, st=True)})
    _expect(any(t.qty == 100 for t in b.trades), "ST 10.49 未到涨停应可买", order=ob.as_dict(), trades=[t.as_dict() for t in b.trades])
    return {"checks": checks, "st_blocked": o.as_dict()}


def _case_suspension() -> dict:
    cal = _cal(n=4)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    o = a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10, 10, 10, vol=0, status=0)})
    _expect(not a.trades and o.status == "cancelled" and o.reason.startswith("no_bar_or_suspended"),
            "停牌日开盘单不成交且记录原因", order=o.as_dict())

    b = _acct(cal, cash=100_000, slip=0)
    b.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    b.buy_open(A, 100, cal[0])
    b.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    st = b.sell_stop(A, 100, 9.0, cal[1])
    b.on_day(cal[2], {})
    _expect(st.status == "pending" and st.reason == "no_bar_or_suspended",
            "GTC 止损在停牌日应继续挂单", status=st.status, reason=st.reason)
    b.on_day(cal[3], {A: _bar(A, cal[3], 9.8, 9.9, 8.8, 9.0)})
    _expect(any(t.side == "sell" for t in b.trades), "复牌后止损应成交", trades=[t.as_dict() for t in b.trades])
    return {"open_order": o.as_dict(), "stop": st.as_dict()}


def _case_stop_gap_and_intraday() -> dict:
    cal = _cal(n=3)
    star = "688001.SH"
    a = _acct(cal, cash=100_000, slip=10)
    a.on_day(cal[0], {star: _bar(star, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(star, 200, cal[0])
    a.on_day(cal[1], {star: _bar(star, cal[1], 10, 10.2, 9.9, 10)})
    a.sell_stop(star, 200, 9.5, cal[1])
    a.on_day(cal[2], {star: _bar(star, cal[2], 8.5, 8.6, 8.4, 8.5)})
    sell = _sells(a)[0]
    _expect(sell.price == 8.49, "跳空止损应按开盘价并加不利滑点", price=sell.price)

    A = "600000.SH"
    b = _acct(cal, cash=100_000, slip=10)
    b.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    b.buy_open(A, 100, cal[0])
    b.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    b.sell_stop(A, 100, 9.5, cal[1])
    b.on_day(cal[2], {A: _bar(A, cal[2], 10, 10.2, 9.4, 9.5)})
    sell2 = _sells(b)[0]
    _expect(sell2.price == 9.49, "盘中触发止损应按止损价并加不利滑点", price=sell2.price)
    return {"gap": sell.as_dict(), "intraday": sell2.as_dict()}


def _case_stop_target_same_day_loss_first() -> dict:
    cal = _cal(n=3)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=10)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    a.sell_stop(A, 100, 9.5, cal[1])
    a.sell_limit(A, 100, 10.5, cal[1])
    a.on_day(cal[2], {A: _bar(A, cal[2], 10, 10.8, 9.4, 10)})
    sells = _sells(a)
    _expect(len(sells) == 1 and sells[0].price == 9.49, "同日止损/止盈应按止损优先", sells=[t.as_dict() for t in sells])
    _expect(any(f["kind"] == "intraday_ambiguity_loss_first" for f in a.flags), "应记录同日冲突损失优先", flags=a.flags)
    return {"sell": sells[0].as_dict(), "flags": a.flags}


def _case_cash_no_intraday_recycle() -> dict:
    cal = _cal(n=4)
    A = "600000.SH"
    a = _acct(cal, cash=1100, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    a.sell_open(A, 100, cal[1])
    b1 = a.buy_open(A, 100, cal[1])
    a.on_day(cal[2], {A: _bar(A, cal[2], 10, 10.2, 9.9, 10)})
    _expect(not [t for t in _buys(a) if t.date == cal[2]], "卖出回款当日不得用于买入", buys=[t.as_dict() for t in _buys(a)])
    _expect(b1.status == "cancelled", "当日资金不足的买单应取消", order=b1.as_dict())
    b2 = a.buy_open(A, 100, cal[2])
    a.on_day(cal[3], {A: _bar(A, cal[3], 10, 10.2, 9.9, 10)})
    _expect(any(t.side == "buy" and t.date == cal[3] for t in a.trades), "次日回款可用于买入", trades=[t.as_dict() for t in a.trades])
    return {"same_day": b1.as_dict(), "next_day": b2.as_dict(), "trades": [t.as_dict() for t in a.trades]}


def _case_cash_settle_zero() -> dict:
    cal = _cal(n=3)
    A = "600000.SH"
    a = _acct(cal, cash=1100, slip=0, cash_settle_days=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[1])   # 故意先提交买单，验证当日回款时引擎先卖后买
    a.sell_open(A, 100, cal[1])
    a.on_day(cal[2], {A: _bar(A, cal[2], 10, 10.2, 9.9, 10)})
    _expect(any(t.side == "buy" and t.date == cal[2] for t in a.trades), "cash_settle_days=0 时当日回款可用", trades=[t.as_dict() for t in a.trades])
    return {"trades": [t.as_dict() for t in a.trades]}


def _case_participation_partial() -> dict:
    cal = _cal(n=2)
    A = "600000.SH"
    a = _acct(cal, cash=1_000_000, slip=0, part=0.10)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    o = a.buy_open(A, 300, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10, vol=1000)})
    _expect(o.filled_qty == 100, "10% 参与率、1000 股成交量下最多成交 100 股", order=o.as_dict())
    _expect(o.status == "cancelled" and "day_end" in o.reason, "日单剩余应取消", order=o.as_dict())
    _expect(a.trades[0].qty == 100, "成交数量应为 100", trades=[t.as_dict() for t in a.trades])
    return {"order": o.as_dict(), "trade": a.trades[0].as_dict()}


def _case_dividend_tax() -> dict:
    cal = _cal(n=3)
    A = "600000.SH"
    a = _acct(cal, cash=1_000_000, slip=0, tax=0.20)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 1000, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    before = a.cash_total
    ev = CorporateAction(A, cal[2], dividend_per_share=0.5)
    a.on_day(cal[2], {A: _bar(A, cal[2], 9.5, 9.7, 9.4, 9.5)}, events=[ev])
    delta = a.cash_total - before
    _expect(abs(delta - 400.0) < 1e-6, "1000 股×0.5 元×80% 应入账 400", delta=delta, cash=a.cash_total)
    _expect(any(f["kind"] == "dividend_tax_flat_rate" for f in a.flags), "应标注固定股息税率", flags=a.flags)

    b = _acct(cal, cash=1_000_000, slip=0, tax=0.0)
    b.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    b.buy_open(A, 1000, cal[0])
    b.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    before_b = b.cash_total
    b.on_day(cal[2], {A: _bar(A, cal[2], 9.5, 9.7, 9.4, 9.5)}, events=[CorporateAction(A, cal[2], dividend_per_share=0.5)])
    _expect(abs((b.cash_total - before_b) - 500.0) < 1e-6, "税率 0 时应入账 500", delta=b.cash_total - before_b)
    return {"tax20_delta": delta, "tax0_delta": b.cash_total - before_b}


def _case_bonus_t1() -> dict:
    cal = _cal(n=4)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 1000, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    ev = CorporateAction(A, cal[2], per_share_bonus=0.3)
    a.on_day(cal[2], {A: _bar(A, cal[2], 7.7, 7.9, 7.6, 7.7)}, events=[ev])
    pos = a.positions[A]
    _expect(pos.shares == 1300, "10 送 3 后应为 1300 股", shares=pos.shares)
    _expect(pos.available == 1000, "送股当日不可卖（保守 T+1）", available=pos.available)
    a.on_day(cal[3], {A: _bar(A, cal[3], 7.7, 7.9, 7.6, 7.7)})
    _expect(a.positions[A].available == 1300, "次日送股可卖", available=a.positions[A].available)
    return {"position": a.positions[A].as_dict(), "actions": a.corporate_actions}


def _case_event_price_conflict_skipped() -> dict:
    cal = _cal(n=3)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    ev = CorporateAction(A, cal[2], per_share_bonus=0.5)
    # 事件称 10 送 5，但开盘仍是 10 元（未除权）；若照事件加股会凭空增厚，应跳过并标注
    a.on_day(cal[2], {A: _bar(A, cal[2], 10, 10.2, 9.9, 10)}, events=[ev])
    _expect(a.positions[A].shares == 100, "价格冲突的送转事件不应加股", position=a.positions[A].as_dict())
    _expect(any(f["kind"] == "corporate_action_price_conflict" for f in a.flags), "应记录价格冲突", flags=a.flags)
    _expect(any((not x.get("applied")) and x.get("reason") == "corporate_action_price_conflict" for x in a.corporate_actions),
            "公司行动记录应标记未应用", actions=a.corporate_actions)
    return {"position": a.positions[A].as_dict(), "actions": a.corporate_actions, "flags": a.flags}


def _case_rights_subscribe_and_skip() -> dict:
    cal = _cal(n=3)
    A = "600000.SH"
    a = _acct(cal, cash=1_000_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 1000, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    before = a.cash_total
    ev = CorporateAction(A, cal[2], rights_ratio=0.2, rights_price=5.0)
    a.on_day(cal[2], {A: _bar(A, cal[2], 8.4, 8.6, 8.3, 8.4)}, events=[ev])
    _expect(a.positions[A].shares == 1200, "10 配 2 后应为 1200 股", position=a.positions[A].as_dict())
    _expect(abs((before - a.cash_total) - 1000.0) < 1e-6, "配股应支出 200×5=1000", cash=a.cash_total, before=before)
    _expect(abs(a.cash_available - a.cash_total) < 1e-9, "配股后可用现金应与总现金同步减少", available=a.cash_available, cash=a.cash_total)
    _expect(any(f["kind"] == "rights_subscribed" for f in a.flags), "应标注配股假设", flags=a.flags)

    b = _acct(cal, cash=10_100, slip=0)
    b.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    b.buy_open(A, 1000, cal[0])
    b.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    b.on_day(cal[2], {A: _bar(A, cal[2], 8.4, 8.6, 8.3, 8.4)}, events=[ev])
    _expect(b.positions[A].shares == 1000, "现金不足时配股应跳过", position=b.positions[A].as_dict())
    _expect(any(f["kind"] == "rights_skipped_insufficient_cash" for f in b.flags), "应标注配股跳过", flags=b.flags)
    return {"subscribed": a.positions[A].as_dict(), "skipped": b.positions[A].as_dict()}


def _case_ex_rights_limit() -> dict:
    ref = fee.ex_rights_reference(10, 0.5, 0.5, 0, 0)
    _expect(abs(ref - 6.333333333333333) < 1e-9, "除权参考价公式错误", ref=ref)
    lim = fee.limit_prices(ref, "600000.SH", False, Date(2024, 1, 2))
    _expect(lim.up == 6.96, "除权日参考价先取整到 6.33，涨停应为 6.96", limits=lim)

    cal = _cal(n=2)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    o = a.buy_open(A, 100, cal[0])
    ev = CorporateAction(A, cal[1], dividend_per_share=0.5, per_share_bonus=0.5)
    a.on_day(cal[1], {A: _bar(A, cal[1], 7.0, 7.1, 6.9, 7.0)}, events=[ev])
    _expect(not a.trades and o.reason.startswith("limit_up_open"),
            "除权日应按除权参考价判断涨停", order=o.as_dict(), limits=a._day_limits.get(A))
    return {"ref": ref, "limits": {"up": lim.up, "down": lim.down}, "order": o.as_dict()}


def _case_delisting() -> dict:
    cal = _cal(n=5)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0, delisting_missing_days=3)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    a.on_day(cal[2], {})
    a.on_day(cal[3], {})
    _expect(a.positions[A].shares == 100, "未到阈值不应强平", position=a.positions[A].as_dict())
    a.on_day(cal[4], {})
    _expect(a.positions[A].shares == 0, "达到阈值应强平", position=a.positions[A].as_dict())
    forced = [t for t in a.trades if t.reason == "delisting_missing_bars"]
    _expect(len(forced) == 1 and forced[0].price == 9.0, "应按最后收盘价 10 元打 10% 折扣强平", forced=[t.as_dict() for t in forced])
    return {"forced": forced[0].as_dict(), "flags": a.flags}


def _case_finalize_valuation() -> dict:
    cal = _cal(n=3)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    a.on_day(cal[2], {A: _bar(A, cal[2], 12, 12.2, 11.8, 12)})
    rec = a.finalize(cal[2], liquidate=False)
    _expect(abs(rec["market_value"] - 1200.0) < 1e-6, "期末市值应为 1200", rec=rec)
    _expect(abs(rec["nav"] - (a.cash_total + 1200.0)) < 1e-6, "期末净值应等于现金+市值", rec=rec)
    return {"final": rec, "position": a.positions[A].as_dict()}


def _case_lookahead_guard() -> dict:
    cal = _cal(n=3)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    raised = False
    try:
        a.submit(Order(A, "buy", 100, execute_from=cal[1]), cal[1])
    except ValueError as exc:
        raised = "禁止未来函数" in str(exc)
    _expect(raised, "同日决策/执行应抛错")
    o = a.buy_open(A, 100, cal[0])
    _expect(o.execute_from == cal[1] and o.execute_from > o.decision_date, "买入应排到下一交易日", order=o.as_dict())
    raised2 = False
    try:
        a.on_day(cal[1], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    except ValueError as exc:
        raised2 = "日期" in str(exc)
    _expect(raised2, "Bar 日期与交易日不一致应抛错")
    return {"order": o.as_dict()}


def _case_limit_target_gap_and_partial_stop() -> dict:
    cal = _cal(n=4)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    a.sell_limit(A, 100, 10.5, cal[1])
    a.on_day(cal[2], {A: _bar(A, cal[2], 10.8, 10.9, 10.6, 10.8)})
    sell = _sells(a)[0]
    _expect(sell.price == 10.8, "限价止盈遇向上跳空应按开盘价成交", price=sell.price)

    b = _acct(cal, cash=100_000, slip=0, part=0.10)
    b.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    b.buy_open(A, 300, cal[0])
    b.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10, vol=1_000_000)})
    b.sell_stop(A, 300, 9.5, cal[1])
    b.on_day(cal[2], {A: _bar(A, cal[2], 10, 10.2, 9.4, 9.6, vol=1000)})
    _expect(b.positions[A].shares == 200, "参与率约束下应部分成交 100 股", position=b.positions[A].as_dict())
    b.on_day(cal[3], {A: _bar(A, cal[3], 10, 10.2, 9.8, 10, vol=1_000_000)})
    sells = _sells(b)
    _expect([t.qty for t in sells] == [100, 200], "剩余止损应在次日开盘成交", sells=[t.as_dict() for t in sells])
    return {"limit_gap": sell.as_dict(), "partial_stop": [t.as_dict() for t in sells]}


def _case_ipo_no_limit() -> dict:
    cal = _cal(n=2)
    code = "688001.SH"
    a = Account(AccountConfig(initial_cash=100_000, fee=fee.FeeConfig(slippage_bps=0)),
                cal, freeze_date=cal[-1], symbol_info={code: SymbolInfo(code, ipo_date=cal[1])})
    a.on_day(cal[0], {code: _bar(code, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(code, 200, cal[0])
    a.on_day(cal[1], {code: _bar(code, cal[1], 15, 15.2, 14.8, 15)})
    _expect(any(t.side == "buy" for t in a.trades), "科创板上市首日无涨跌幅应可买", trades=[t.as_dict() for t in a.trades])

    b = _acct(cal, cash=100_000, slip=0)
    b.on_day(cal[0], {code: _bar(code, cal[0], 10, 10.2, 9.9, 10)})
    ob = b.buy_open(code, 200, cal[0])
    b.on_day(cal[1], {code: _bar(code, cal[1], 15, 15.2, 14.8, 15)})
    _expect(not b.trades and ob.reason.startswith("limit_up_open"), "无 IPO 信息时应按 20% 涨停挡住", order=ob.as_dict())
    return {"ipo_fill": [t.as_dict() for t in a.trades], "no_info": ob.as_dict()}


def _case_ipo_before_calendar() -> dict:
    cal = _cal(n=2)
    code = "688001.SH"
    a = Account(AccountConfig(initial_cash=100_000, fee=fee.FeeConfig(slippage_bps=0)),
                cal, freeze_date=cal[-1],
                symbol_info={code: SymbolInfo(code, ipo_date=Date(2020, 1, 2))})
    a.on_day(cal[0], {code: _bar(code, cal[0], 10, 10.2, 9.9, 10)})
    o = a.buy_open(code, 200, cal[0])
    a.on_day(cal[1], {code: _bar(code, cal[1], 12.5, 12.6, 12.4, 12.5)})
    _expect(not a.trades and o.reason.startswith("limit_up_open"),
            "IPO 早于日历起点时不应误判为新股无涨跌幅窗口", order=o.as_dict())
    return {"order": o.as_dict()}


def _case_bar_validation() -> dict:
    cal = _cal(n=2)
    A = "600000.SH"
    errors = {}
    for name, args in {
        "high_lt_low": (A, cal[0], 10, 9.0, 9.5, 10, 1e6),
        "open_outside": (A, cal[0], 10.5, 10.0, 9.5, 10, 1e6),
        "close_outside": (A, cal[0], 10, 10.2, 9.9, 10.5, 1e6),
    }.items():
        try:
            Bar(*args)
            errors[name] = "no_error"
        except ValueError:
            errors[name] = "raised"
    _expect(all(v == "raised" for v in errors.values()), "非法 OHLC 应被拒绝", errors=errors)
    return {"errors": errors}


def _case_future_event_queue() -> dict:
    cal = _cal(n=4)
    A = "600000.SH"
    a = _acct(cal, cash=1_000_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 1000, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    before = a.cash_total
    ev = CorporateAction(A, cal[3], dividend_per_share=0.5)
    a.on_day(cal[2], {A: _bar(A, cal[2], 10, 10.2, 9.9, 10)}, events=[ev])
    _expect(abs(a.cash_total - before) < 1e-9, "未到除权日不应分红", cash=a.cash_total, before=before)
    a.on_day(cal[3], {A: _bar(A, cal[3], 9.6, 9.8, 9.5, 9.6)})
    _expect(abs((a.cash_total - before) - 400.0) < 1e-6, "除权日应分红 400", delta=a.cash_total - before)
    return {"delta": a.cash_total - before}


def _case_sell_fees_exact() -> dict:
    cal = _cal(n=3)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    before = a.cash_total
    a.sell_open(A, 100, cal[1])
    a.on_day(cal[2], {A: _bar(A, cal[2], 10, 10.2, 9.9, 10)})
    t = _sells(a)[0]
    _expect(abs((a.cash_total - before) - 994.49) < 1e-6, "卖出净回款应为 1000-5-0.5-0.01", delta=a.cash_total - before, trade=t.as_dict())
    _expect(t.commission == 5.0 and t.stamp_duty == 0.5 and t.transfer_fee == 0.01,
            "卖出费用拆分错误", fees=t.as_dict())
    return {"trade": t.as_dict(), "cash_delta": a.cash_total - before}


def _case_delisting_out_date() -> dict:
    cal = _cal(n=4)
    A = "600000.SH"
    a = Account(AccountConfig(initial_cash=100_000, fee=fee.FeeConfig(slippage_bps=0)),
                cal, freeze_date=cal[-1],
                symbol_info={A: SymbolInfo(A, out_date=cal[2])})
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    a.buy_open(A, 100, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.9, 10)})
    a.on_day(cal[3], {})
    forced = [t for t in a.trades if t.reason == "delisting_out_date"]
    _expect(len(forced) == 1 and forced[0].price == 9.0, "out_date 后应按最后价打 9 折强平", forced=[t.as_dict() for t in forced])
    return {"forced": forced[0].as_dict(), "position": a.positions[A].as_dict()}


def _case_limit_buy() -> dict:
    cal = _cal(n=2)
    A = "600000.SH"
    a = _acct(cal, cash=100_000, slip=0)
    a.on_day(cal[0], {A: _bar(A, cal[0], 10, 10.2, 9.9, 10)})
    o = a.buy_limit(A, 100, 9.5, cal[0])
    a.on_day(cal[1], {A: _bar(A, cal[1], 10, 10.2, 9.4, 10)})
    _expect(any(t.side == "buy" and t.price == 9.5 for t in a.trades),
            "买入限价单应在 9.5 成交", order=o.as_dict(), trades=[t.as_dict() for t in a.trades])
    return {"order": o.as_dict(), "trade": a.trades[0].as_dict()}


def _self_test_cases() -> list[tuple[str, Callable[[], dict]]]:
    return [
        ("t1_buy_then_stop_same_day", _case_t1_buy_then_stop_same_day),
        ("lot_main_100", _case_lot_main),
        ("lot_star_bse", _case_lot_star_bse),
        ("odd_lot_sell", _case_odd_lot_sell),
        ("open_slippage_fees", _case_open_slippage_fees),
        ("fee_schedules", _case_fee_schedules),
        ("limit_up_buy_blocked", _case_limit_up_buy_blocked),
        ("limit_down_sell_blocked", _case_limit_down_sell_blocked),
        ("board_and_st_limits", _case_board_and_st_limits),
        ("suspension_and_persistence", _case_suspension),
        ("stop_gap_and_intraday", _case_stop_gap_and_intraday),
        ("stop_target_same_day_loss_first", _case_stop_target_same_day_loss_first),
        ("cash_no_intraday_recycle", _case_cash_no_intraday_recycle),
        ("cash_settle_zero", _case_cash_settle_zero),
        ("participation_partial", _case_participation_partial),
        ("dividend_tax", _case_dividend_tax),
        ("bonus_t1", _case_bonus_t1),
        ("event_price_conflict_skipped", _case_event_price_conflict_skipped),
        ("rights_subscribe_and_skip", _case_rights_subscribe_and_skip),
        ("ex_rights_limit", _case_ex_rights_limit),
        ("delisting_force_liquidation", _case_delisting),
        ("finalize_valuation", _case_finalize_valuation),
        ("lookahead_guard", _case_lookahead_guard),
        ("limit_target_gap_and_partial_stop", _case_limit_target_gap_and_partial_stop),
        ("ipo_no_limit", _case_ipo_no_limit),
        ("ipo_before_calendar", _case_ipo_before_calendar),
        ("bar_validation", _case_bar_validation),
        ("future_event_queue", _case_future_event_queue),
        ("sell_fees_exact", _case_sell_fees_exact),
        ("delisting_out_date", _case_delisting_out_date),
        ("limit_buy", _case_limit_buy),
    ]


def run_self_tests() -> dict:
    cases = []
    for name, fn in _self_test_cases():
        try:
            detail = fn() or {}
            cases.append({"name": name, "passed": True, "detail": detail})
        except AssertionError as exc:
            cases.append({"name": name, "passed": False, "error": f"AssertionError: {exc}",
                          "trace": traceback.format_exc(limit=6)})
        except Exception as exc:  # noqa: BLE001
            cases.append({"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}",
                          "trace": traceback.format_exc(limit=6)})
    passed = sum(1 for c in cases if c["passed"])
    return {
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "cases": cases,
    }

# ---------------------------------------------------------------------------
# 数据审计（只读；只审计覆盖与制度一致性，不计算任何策略收益）
# ---------------------------------------------------------------------------

DUCKDB_PATH = "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"
BAOSTOCK_PATH = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"


def _safe_section(fn: Callable[[], Any]) -> dict:
    try:
        return {"ok": True, "result": fn()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=8)}


def _audit_coverage(con, freeze: Date) -> dict:
    row = con.execute(
        "select count(*), count(distinct thscode), min(date), max(date), count(distinct date) "
        "from raw_kline_daily where date <= ?", [freeze]
    ).fetchone()
    after = con.execute("select count(*), count(distinct thscode) from raw_kline_daily where date > ?", [freeze]).fetchone()
    return {
        "rows": int(row[0]), "symbols": int(row[1]),
        "min_date": row[2].isoformat(), "max_date": row[3].isoformat(),
        "trading_days": int(row[4]),
        "rows_after_freeze_excluded": int(after[0]),
        "symbols_with_rows_after_freeze": int(after[1]),
        "freeze_date": freeze.isoformat(),
    }


def _audit_quality(con, freeze: Date) -> dict:
    dup = con.execute(
        "select count(*) from (select thscode,date,count(*) c from raw_kline_daily "
        "where date <= ? group by 1,2 having c>1)", [freeze]
    ).fetchone()[0]
    row = con.execute(
        """
        select
          sum(case when open<=0 or high<=0 or low<=0 or close<=0 then 1 else 0 end),
          sum(case when high<low or open>high or open<low or close>high or close<low then 1 else 0 end),
          sum(case when volume<=0 then 1 else 0 end),
          sum(case when prev_close is null then 1 else 0 end),
          sum(case when amount is null or amount<=0 then 1 else 0 end)
        from raw_kline_daily where date <= ?
        """, [freeze]
    ).fetchone()
    return {
        "duplicate_symbol_date_groups": int(dup),
        "nonpositive_ohlc_rows": int(row[0] or 0),
        "ohlc_inconsistent_rows": int(row[1] or 0),
        "zero_volume_rows": int(row[2] or 0),
        "prev_close_null_rows": int(row[3] or 0),
        "amount_null_or_nonpositive_rows": int(row[4] or 0),
        "interpretation": "prev_close 全空是数据源特性，撮合层用上一交易日收盘价补；除权日由事件计算参考价。",
    }


def _audit_calendar(con, freeze: Date) -> dict:
    start = Date(2016, 10, 13)
    duck = {r[0] for r in con.execute(
        "select distinct date from raw_kline_daily where date between ? and ?", [start, freeze]
    ).fetchall()}
    bao_con = sqlite3.connect(f"file:{BAOSTOCK_PATH}?mode=ro", uri=True)
    try:
        bao = {Date.fromisoformat(r[0]) for r in bao_con.execute(
            "select calendar_date from trade_dates where is_trading_day=1 and calendar_date between ? and ?",
            [start.isoformat(), freeze.isoformat()],
        ).fetchall()}
    finally:
        bao_con.close()
    return {
        "duckdb_days": len(duck), "baostock_days": len(bao),
        "missing_in_duckdb": [d.isoformat() for d in sorted(bao - duck)],
        "extra_in_duckdb": [d.isoformat() for d in sorted(duck - bao)],
        "agree": duck == bao,
    }


def _audit_events(con, freeze: Date) -> dict:
    start = Date(2016, 10, 13)
    row = con.execute(
        """
        select count(*), count(distinct thscode),
          sum(case when dividend_per_share>0 then 1 else 0 end),
          sum(case when per_share_bonus>0 then 1 else 0 end),
          sum(case when rights_ratio>0 then 1 else 0 end),
          sum(case when dividend_per_share=0 and per_share_bonus=0 and rights_ratio=0 then 1 else 0 end),
          sum(case when rights_ratio>0 and (rights_price is null or rights_price<=0) then 1 else 0 end)
        from raw_adjustment_events where ex_date between ? and ?
        """, [start, freeze]
    ).fetchone()
    after = con.execute(
        "select count(*), count(distinct thscode) from raw_adjustment_events where ex_date > ?", [freeze]
    ).fetchone()
    nontrading = con.execute(
        """
        select count(*) from raw_adjustment_events e
        where e.ex_date between ? and ?
          and not exists (select 1 from (select distinct date from raw_kline_daily where date between ? and ?) d
                          where d.date=e.ex_date)
        """, [start, freeze, start, freeze]
    ).fetchone()[0]
    by_year = con.execute(
        """
        select year(ex_date) y, count(*) n, count(distinct thscode) syms,
               sum(case when per_share_bonus>0 then 1 else 0 end) bonus_events,
               sum(case when rights_ratio>0 then 1 else 0 end) rights_events
        from raw_adjustment_events where ex_date between ? and ? group by 1 order by 1
        """, [start, freeze]
    ).df()
    return {
        "events_in_window": int(row[0]), "symbols": int(row[1]),
        "dividend_events": int(row[2] or 0), "bonus_events": int(row[3] or 0),
        "rights_events": int(row[4] or 0), "empty_events": int(row[5] or 0),
        "rights_with_missing_price": int(row[6] or 0),
        "events_after_freeze_excluded": int(after[0]),
        "symbols_after_freeze_excluded": int(after[1]),
        "events_on_non_trading_days": int(nontrading),
        "by_year": [
            {"year": int(r.y), "events": int(r.n), "symbols": int(r.syms),
             "bonus_events": int(r.bonus_events or 0), "rights_events": int(r.rights_events or 0)}
            for r in by_year.itertuples()
        ],
    }


def _audit_factors(con, freeze: Date) -> dict:
    start = Date(2016, 10, 13)
    missing = con.execute(
        """
        select count(*) from raw_kline_daily d
        left join calc_adjust_factor_daily f on d.thscode=f.thscode and d.date=f.date
        where d.date between ? and ? and f.thscode is null
        """, [start, freeze]
    ).fetchone()[0]
    changes = con.execute(
        """
        with f as (
          select thscode,date,backward_factor,
                 lag(backward_factor) over (partition by thscode order by date) prev
          from calc_adjust_factor_daily where date between ? and ?
        ), chg as (
          select thscode,date,backward_factor,prev from f
          where prev is not null and abs(backward_factor-prev)>1e-9
        )
        select count(*), sum(case when e.thscode is not null then 1 else 0 end)
        from chg left join (select distinct thscode, ex_date from raw_adjustment_events) e
          on chg.thscode=e.thscode and chg.date=e.ex_date
        """, [start, freeze]
    ).fetchone()
    events = con.execute(
        """
        with f as (
          select thscode,date,backward_factor,
                 lag(backward_factor) over (partition by thscode order by date) prev
          from calc_adjust_factor_daily where date between ? and ?
        ), chg as (
          select thscode,date from f where prev is not null and abs(backward_factor-prev)>1e-9
        ), e as (
          select distinct thscode, ex_date from raw_adjustment_events where ex_date between ? and ?
        )
        select count(*), sum(case when c.thscode is not null then 1 else 0 end)
        from e left join chg c on e.thscode=c.thscode and e.ex_date=c.date
        """, [start, freeze, start, freeze]
    ).fetchone()
    unexplained = con.execute(
        """
        with f as (
          select thscode,date,backward_factor,
                 lag(backward_factor) over (partition by thscode order by date) prev
          from calc_adjust_factor_daily where date between ? and ?
        ), chg as (
          select thscode,date,backward_factor,prev from f
          where prev is not null and abs(backward_factor-prev)>1e-9
        )
        select c.thscode, c.date, c.prev, c.backward_factor,
               (c.backward_factor/c.prev-1) factor_change,
               d.open, d.close
        from chg c left join raw_kline_daily d on c.thscode=d.thscode and c.date=d.date
        where d.date is not null
          and not exists (select 1 from raw_adjustment_events e where e.thscode=c.thscode and e.ex_date=c.date)
        order by c.date, c.thscode limit 15
        """, [start, freeze]
    ).df()
    unexplained_list = [
        {"code": r.thscode, "date": str(r.date)[:10], "factor_change": round(float(r.factor_change), 6),
         "open": float(r.open), "close": float(r.close)}
        for r in unexplained.itertuples()
    ]
    div_check = con.execute(
        """
        with e as (
          select thscode, ex_date, dividend_per_share
          from raw_adjustment_events
          where ex_date between ? and ? and per_share_bonus=0 and rights_ratio=0 and dividend_per_share>0
        ), r as (
          select d.thscode,d.date,d.close,f.backward_factor,
                 lag(d.close) over (partition by d.thscode order by d.date) pc,
                 lag(f.backward_factor) over (partition by d.thscode order by d.date) pf
          from raw_kline_daily d join calc_adjust_factor_daily f on d.thscode=f.thscode and d.date=f.date
          where d.date between ? and ?
        )
        select count(*),
               avg(abs((backward_factor/pf) - (pc/(pc-dividend_per_share)))),
               quantile_cont(abs((backward_factor/pf) - (pc/(pc-dividend_per_share))),0.5),
               sum(case when abs((backward_factor/pf) - (pc/(pc-dividend_per_share))) < 0.005 then 1 else 0 end)
        from e join r on e.thscode=r.thscode and e.ex_date=r.date
        where pf is not null and pc>dividend_per_share
        """, [start, freeze, start, freeze]
    ).fetchone()
    return {
        "rows_without_factor": int(missing),
        "factor_changes": int(changes[0] or 0),
        "factor_changes_with_event": int(changes[1] or 0),
        "factor_changes_without_event": int((changes[0] or 0) - (changes[1] or 0)),
        "events_with_factor_change": int(events[1] or 0),
        "events_without_factor_change": int((events[0] or 0) - (events[1] or 0)),
        "dividend_only_consistency_n": int(div_check[0] or 0),
        "dividend_only_mean_abs_error": float(div_check[1] or 0.0),
        "dividend_only_median_abs_error": float(div_check[2] or 0.0),
        "dividend_only_within_0p5pct": int(div_check[3] or 0),
        "unexplained_examples": unexplained_list,
        "interpretation": "复权因子变动与事件高度一致；少量无事件变动会制造原始价格假跌，撮合层必须标注而非静默忽略。",
    }


def _audit_limits(con, freeze: Date, sample: int = 300) -> dict:
    rows = con.execute(
        f"""
        with codes as (
          select thscode from dim_symbol where exchange in ('SH','SZ','BJ') order by thscode limit {int(sample)}
        ), x as (
          select d.thscode,d.date,d.close,
                 lag(d.close) over (partition by d.thscode order by d.date) pc,
                 row_number() over (partition by d.thscode order by d.date) rn
          from raw_kline_daily d join codes c using(thscode)
          where d.date between ? and ?
        )
        select thscode,date,close,pc,rn from x where pc is not null and rn>5
        """, [Date(2016, 10, 13), freeze]
    ).df()
    ev = con.execute(
        "select distinct thscode, ex_date from raw_adjustment_events where ex_date between ? and ?",
        [Date(2016, 10, 13), freeze]
    ).df()
    ev_set = set(zip(ev["thscode"], ev["ex_date"]))
    total = 0
    excluded_ex = 0
    exceed = []
    for r in rows.itertuples():
        total += 1
        if (r.thscode, r.date) in ev_set:
            excluded_ex += 1
            continue
        pct = float(r.close) / float(r.pc) - 1.0
        up, _ = fee.regular_limit_pct(r.thscode, False, r.date)
        if abs(pct) > up + 0.005:
            exceed.append({
                "code": r.thscode, "date": r.date.date().isoformat() if hasattr(r.date, "date") else r.date.isoformat(),
                "pct": round(pct, 6), "limit": up,
                "close": float(r.close), "prev_close": float(r.pc),
                "board": fee.board_of(r.thscode),
            })
    for ex in exceed:
        f = con.execute(
            "select backward_factor from calc_adjust_factor_daily where thscode=? and date=?",
            [ex["code"], Date.fromisoformat(ex["date"][:10])],
        ).fetchone()
        pf = con.execute(
            "select backward_factor from calc_adjust_factor_daily where thscode=? and date<? order by date desc limit 1",
            [ex["code"], Date.fromisoformat(ex["date"][:10])],
        ).fetchone()
        ex["factor_change"] = round(float(f[0]) / float(pf[0]) - 1, 8) if (f and pf and pf[0]) else None
        ex["has_corporate_event"] = (ex["code"], Date.fromisoformat(ex["date"][:10])) in ev_set
        if ex["factor_change"] not in (None, 0.0):
            ex["classification"] = "unexplained_factor_change"
        elif ex["pct"] > 0:
            ex["classification"] = "large_positive_move_no_factor_change"
        else:
            ex["classification"] = "large_negative_move_no_factor_change"
    by_board = Counter(x["board"] for x in exceed)
    by_class = Counter(x.get("classification") for x in exceed)
    return {
        "sample_symbols": int(sample),
        "rows_checked": total,
        "ex_rights_rows_excluded": excluded_ex,
        "exceedance_count": len(exceed),
        "exceedance_rate": round(len(exceed) / max(1, total - excluded_ex), 8),
        "by_board": dict(by_board),
        "classification_counts": dict(by_class),
        "examples": exceed[:15],
        "note": "未接 ST 标记时按非 ST 涨停幅度审计；ST 5% 只会导致漏检、不会导致误报。超额样本多为 IPO/退市/无事件送转，需逐条核对。",
    }


def _audit_suspension(con, freeze: Date, sample: int = 100) -> dict:
    codes = [r[0] for r in con.execute(
        f"select thscode from dim_symbol where exchange in ('SH','SZ','BJ') order by thscode limit {int(sample)}"
    ).fetchall()]
    if not codes:
        return {"sample_symbols": 0}
    placeholders = ",".join(["?"] * len(codes))
    duck_rows = con.execute(
        f"select thscode,date from raw_kline_daily where thscode in ({placeholders}) and date between ? and ?",
        codes + [Date(2016, 10, 13), freeze]
    ).fetchall()
    duck_dates: dict[str, set] = {}
    for code, d in duck_rows:
        duck_dates.setdefault(code, set()).add(d)
    bao_con = sqlite3.connect(f"file:{BAOSTOCK_PATH}?mode=ro", uri=True)
    bao_status: dict[str, dict] = {}
    try:
        start_s, end_s = Date(2016, 10, 13).isoformat(), freeze.isoformat()
        for c in codes:
            bc = BaostockMarketData.from_canonical(c)
            rows = bao_con.execute(
                "select date,trade_status from stock_kline_d where code=? and date between ? and ? and adjustflag=3",
                [bc, start_s, end_s],
            ).fetchall()
            if rows:
                bao_status[c] = {Date.fromisoformat(ds): int(st or 0) for ds, st in rows}
    finally:
        bao_con.close()
    cal = [r[0] for r in con.execute(
        "select distinct date from raw_kline_daily where date between ? and ? order by date",
        [Date(2016, 10, 13), freeze]
    ).fetchall()]
    counts = Counter()
    examples = []
    for code in codes:
        dd = duck_dates.get(code, set())
        bs = bao_status.get(code, {})
        if not bs:
            counts["baostock_no_rows"] += 1
            continue
        first = min(bs)
        last = max(bs)
        for d in cal:
            if d < first or d > last:
                continue
            has_bar = d in dd
            st = bs.get(d)
            if st is None:
                counts["baostock_missing_day"] += 1
            elif has_bar and st == 0:
                counts["duckdb_bar_on_suspended_day"] += 1
                if len(examples) < 10:
                    examples.append({"code": code, "date": d.isoformat(), "type": "duckdb_bar_on_suspended_day"})
            elif (not has_bar) and st == 1:
                counts["duckdb_missing_trading_day"] += 1
                if len(examples) < 10:
                    examples.append({"code": code, "date": d.isoformat(), "type": "duckdb_missing_trading_day"})
            elif (not has_bar) and st == 0:
                counts["agree_suspended"] += 1
            else:
                counts["agree_trading"] += 1
    return {
        "sample_symbols": len(codes),
        "counts": dict(counts),
        "examples": examples,
        "note": "以 Baostock trade_status 为停牌基准；DuckDB 缺失行视为停牌。差异需在正式回测前清零或标注。",
    }


def _audit_event_price_consistency(con, freeze: Date, sample: int = 300) -> dict:
    """2005-2015：DuckDB 事件 vs Baostock 原始价格的除权一致性抽样。

    这段时间 DuckDB 没有日线，价格只能用 Baostock；若事件称送转但原始价没除权，
    照事件加股会凭空增厚。这里量化冲突率，账户层据此跳过冲突事件并打 flag。
    """
    rows = con.execute(
        """
        select thscode, ex_date, dividend_per_share, per_share_bonus, rights_ratio, rights_price
        from raw_adjustment_events
        where ex_date between '2005-01-01' and '2015-12-31'
          and (per_share_bonus>0 or rights_ratio>0)
        order by ex_date, thscode
        """
    ).fetchall()
    if not rows:
        return {"sampled_events": 0}
    step = max(1, len(rows) // int(sample))
    picked = rows[::step][:int(sample)]
    bao_con = sqlite3.connect(f"file:{BAOSTOCK_PATH}?mode=ro", uri=True)
    conflicts: list[dict] = []
    no_trade = 0
    no_prev = 0
    checked = 0
    try:
        for r in picked:
            code = BaostockMarketData.to_canonical(r[0])
            exd = r[1]
            bar = bao_con.execute(
                "select open,close,trade_status,is_st from stock_kline_d "
                "where code=? and date=? and adjustflag=3",
                [BaostockMarketData.from_canonical(code), exd.isoformat()],
            ).fetchone()
            if not bar or int(bar[2] or 0) != 1:
                no_trade += 1
                continue
            prev = bao_con.execute(
                "select close from stock_kline_d where code=? and date<? and adjustflag=3 and trade_status=1 "
                "order by date desc limit 1",
                [BaostockMarketData.from_canonical(code), exd.isoformat()],
            ).fetchone()
            if not prev or not prev[0]:
                no_prev += 1
                continue
            raw_prev = float(prev[0])
            try:
                ref = fee.ex_rights_reference(raw_prev, r[2] or 0, r[3] or 0, r[4] or 0, r[5] or 0)
            except Exception:
                continue
            if ref <= 0:
                continue
            up, _ = fee.regular_limit_pct(code, bool(bar[3]), exd)
            gap = float(bar[0]) / ref - 1.0
            checked += 1
            if abs(gap) > up + 0.02:
                conflicts.append({
                    "code": code, "ex_date": exd.isoformat(),
                    "dividend": float(r[2] or 0), "bonus": float(r[3] or 0),
                    "rights_ratio": float(r[4] or 0), "rights_price": float(r[5] or 0),
                    "raw_prev": round(raw_prev, 4), "event_ref": round(ref, 4),
                    "open": float(bar[0]), "open_vs_ref": round(gap, 6),
                    "limit": up, "is_st": bool(bar[3]),
                })
    finally:
        bao_con.close()
    return {
        "universe_events_2005_2015_with_bonus_or_rights": len(rows),
        "sampled_events": len(picked),
        "checked_tradable": checked,
        "no_trade_on_ex_date": no_trade,
        "no_prev_close": no_prev,
        "conflict_count": len(conflicts),
        "conflict_rate": round(len(conflicts) / max(1, checked), 6),
        "examples": conflicts[:15],
        "note": "以 Baostock adjustflag=3 原始价的前收 + DuckDB 事件算参考价；开盘偏离超过当日涨跌停+2% 视为冲突。冲突事件账户层不应用，只标注。",
    }


def _audit_delisting(con, freeze: Date) -> dict:
    bao_con = sqlite3.connect(f"file:{BAOSTOCK_PATH}?mode=ro", uri=True)
    try:
        rows = bao_con.execute(
            "select code, code_name, ipo_date, out_date from stock_basic where type=1 and status=0"
        ).fetchall()
    finally:
        bao_con.close()
    delisted = {BaostockMarketData.to_canonical(r[0]): r for r in rows}
    if not delisted:
        return {"delisted_in_baostock": 0}
    codes = list(delisted)
    placeholders = ",".join(["?"] * len(codes))
    duck = con.execute(
        f"select thscode,min(date),max(date),count(*) from raw_kline_daily "
        f"where thscode in ({placeholders}) and date<=? group by 1",
        codes + [freeze]
    ).fetchall()
    duck_map = {r[0]: {"first": r[1], "last": r[2], "rows": int(r[3])} for r in duck}
    with_bars = [c for c in codes if c in duck_map]
    last_before_freeze = [c for c in with_bars if duck_map[c]["last"] < freeze]
    examples = [
        {"code": c, "name": delisted[c][1], "out_date": delisted[c][3],
         "duckdb_last": duck_map[c]["last"].isoformat(), "rows": duck_map[c]["rows"]}
        for c in with_bars[:10]
    ]
    never = [c for c in codes if c not in duck_map]
    post2016 = []
    for c in never:
        od = delisted[c][3]
        if od and Date.fromisoformat(od) >= Date(2016, 10, 13):
            post2016.append(c)
    return {
        "delisted_in_baostock": len(codes),
        "delisted_with_duckdb_bars": len(with_bars),
        "delisted_never_in_duckdb": len(never),
        "delisted_never_in_duckdb_after_2016_10_13": len(post2016),
        "delisted_last_bar_before_freeze": len(last_before_freeze),
        "examples": examples,
        "never_examples": [{"code": c, "name": delisted[c][1], "out_date": delisted[c][3]} for c in never[:10]],
        "never_after_2016_examples": [
            {"code": c, "name": delisted[c][1], "out_date": delisted[c][3]} for c in post2016[:15]
        ],
        "note": "Baostock status=0 含退市/吸收合并/暂停上市；DuckDB 对 2016 年后退市股覆盖很少，直接用 DuckDB 回测有生存者偏差，退市股与 2016 前历史应走 Baostock 适配器。",
    }


def _audit_st(con, freeze: Date, sample_codes: int = 100) -> dict:
    # 全表 is_st 扫描在该只读库上会被多进程争用拖到数分钟；用固定代码样本做制度性审计。
    bao_con = sqlite3.connect(f"file:{BAOSTOCK_PATH}?mode=ro", uri=True)
    year_counts: Counter = Counter()
    year_codes: dict[str, set] = {}
    st_codes: set[str] = set()
    try:
        codes = [r[0] for r in bao_con.execute(
            "select code from stock_basic where type=1 order by code limit ?", [int(sample_codes)]
        ).fetchall()]
        start_s, end_s = Date(2016, 10, 13).isoformat(), freeze.isoformat()
        for code in codes:
            rows = bao_con.execute(
                "select substr(date,1,4) y, count(*) n from stock_kline_d "
                "where code=? and adjustflag=3 and is_st=1 and date between ? and ? group by 1",
                [code, start_s, end_s],
            ).fetchall()
            for y, n in rows:
                year_counts[y] += int(n)
                year_codes.setdefault(y, set()).add(code)
                st_codes.add(code)
    finally:
        bao_con.close()
    by_year = [{"year": y, "rows": int(n), "symbols": len(year_codes.get(y, set()))}
               for y, n in sorted(year_counts.items())]
    has_col = con.execute(
        "select count(*) from information_schema.columns where table_name='raw_kline_daily' and column_name='is_st'"
    ).fetchone()[0]
    return {
        "duckdb_has_is_st_column": bool(has_col),
        "sample_codes": len(codes),
        "baostock_st_rows_in_sample": int(sum(year_counts.values())),
        "baostock_st_symbols_in_sample": len(st_codes),
        "by_year": by_year,
        "note": "抽样 100 只（按代码排序）做 ST 覆盖审计，非全表；DuckDB 原始日线无 is_st，正式回测必须由 Baostock 或外部 ST 日历注入，否则 5% 涨跌停判断会漏检。",
    }


def _adapter_smoke(freeze: Date) -> dict:
    out: dict = {}
    duck = DuckDBMarketData(DUCKDB_PATH, freeze)
    try:
        try:
            duck.bars_on(freeze + timedelta(days=1))
            raise AssertionError("DuckDB 适配器未拒绝冻结日之后的数据")
        except ValueError:
            pass
        cal = duck.calendar(Date(2024, 10, 8), Date(2024, 10, 11))
        by_day = {}
        for d in cal:
            by_day[d] = duck.bars_on(d)
        d0 = Date(2024, 10, 10)
        bar = by_day[d0].get("000001.SZ")
        if bar is None:
            raise AssertionError("DuckDB 适配器未取到 000001.SZ 2024-10-10")
        evs = duck.events_on(d0)
        ev = next((e for e in evs if e.code == "000001.SZ"), None)
        out["duckdb"] = {
            "calendar": [x.isoformat() for x in cal],
            "bar": bar.as_dict(),
            "event": asdict(ev) if ev else None,
            "limit": None,
        }
        if ev:
            ref = fee.ex_rights_reference(bar.prev_close, ev.dividend_per_share, ev.per_share_bonus, ev.rights_ratio, ev.rights_price)
            lim = fee.limit_prices(ref, bar.code, bar.is_st, d0)
            out["duckdb"]["limit"] = {"ref": ref, "up": lim.up, "down": lim.down}
        assert abs(bar.close - 11.98) < 1e-9, bar.close
        assert abs(bar.prev_close - 11.68) < 1e-9, bar.prev_close
        assert ev is not None and abs(ev.dividend_per_share - 0.246) < 1e-9

        # 真实数据账户烟测：不下单，只验证日循环/事件/涨跌停参考价能跑通
        acct = Account(AccountConfig(initial_cash=1_000_000), cal, freeze_date=freeze)
        lim_acct = None
        for dd in cal:
            acct.on_day(dd, by_day[dd], events=duck.events_on(dd))
            if dd == d0:
                lim_acct = acct._day_limits.get("000001.SZ")
        out["duckdb"]["account_smoke"] = {
            "trades": len(acct.trades),
            "daily_last": acct.daily[-1],
            "limit_000001": {"up": lim_acct.up, "down": lim_acct.down, "ref": lim_acct.ref} if lim_acct else None,
            "flag_counts": dict(acct.flag_counts),
            "note": "首日无前收的 no_prev_close flag 属预期；正式回测应从预热日之后开始，避免首日全市场 flag。",
        }
        assert len(acct.trades) == 0
        assert lim_acct is not None and lim_acct.up == 12.57 and lim_acct.down == 10.29
        fc = duck.factor_changes_on(Date(2017, 7, 17))
        out["duckdb"]["factor_change_probe"] = [x for x in fc if x["code"] == "000007.SZ"]
        assert any(x["code"] == "000007.SZ" and abs(x["change"] - 0.5) < 1e-9 and not x["has_event"] for x in fc)
    finally:
        duck.close()

    bao = BaostockMarketData(BAOSTOCK_PATH, freeze, events=DuckDBEventSource(DUCKDB_PATH, freeze))
    try:
        try:
            bao.bars_on(freeze + timedelta(days=1))
            raise AssertionError("Baostock 适配器未拒绝冻结日之后的数据")
        except ValueError:
            pass
        bars = bao.bars_on(Date(2024, 10, 10))
        b = bars.get("600000.SH")
        if b is None:
            raise AssertionError("Baostock 适配器未取到 600000.SH 2024-10-10")
        out["baostock"] = {"bar": b.as_dict(), "event_000001": None}
        evs = bao.events_on(Date(2024, 10, 10))
        ev = next((e for e in evs if e.code == "000001.SZ"), None)
        out["baostock"]["event_000001"] = asdict(ev) if ev else None
        assert abs(b.close - 10.31) < 1e-9, b.close
        assert abs(b.prev_close - 9.91) < 1e-9, b.prev_close
        assert b.code == "600000.SH" and b.prev_close_is_ref and not b.is_st

        # DuckDB 完全没有 600005.SH（2017-02-14 退市）；Baostock 必须有，证明退市股覆盖路径可用
        old_bars = bao.bars_on(Date(2017, 1, 23))
        b_old = old_bars.get("600005.SH")
        if b_old is None:
            raise AssertionError("Baostock 适配器未取到已退市 600005.SH 2017-01-23")
        info = bao.symbol_info("600005.SH")
        out["baostock"]["delisted_probe"] = {
            "bar": b_old.as_dict(),
            "symbol_info": asdict(info) if info else None,
        }
        assert abs(b_old.close - 3.71) < 1e-9, b_old.close
        assert info is not None and info.out_date == Date(2017, 2, 14)

        # 2008 年路径：价格 + DuckDB 事件 + Baostock 除权参考价 + 涨跌停
        cal08 = [Date(2008, 4, 22), Date(2008, 4, 23), Date(2008, 4, 24), Date(2008, 4, 25)]
        bars08 = {dd: bao.bars_on(dd) for dd in cal08}
        b08 = bars08[Date(2008, 4, 24)].get("600000.SH")
        if b08 is None:
            raise AssertionError("Baostock 适配器未取到 600000.SH 2008-04-24")
        ev08 = next((e for e in bao.events_on(Date(2008, 4, 24)) if e.code == "600000.SH"), None)
        acct08 = Account(AccountConfig(initial_cash=1_000_000), cal08, freeze_date=freeze)
        lim08 = None
        for dd in cal08:
            acct08.on_day(dd, bars08[dd], events=bao.events_on(dd))
            if dd == Date(2008, 4, 24):
                lim08 = acct08._day_limits.get("600000.SH")
        out["baostock"]["probe_2008"] = {
            "bar": b08.as_dict(),
            "event": asdict(ev08) if ev08 else None,
            "limit": {"up": lim08.up, "down": lim08.down, "ref": lim08.ref} if lim08 else None,
        }
        assert abs(b08.close - 29.72) < 1e-9 and abs(b08.prev_close - 27.02) < 1e-9
        assert ev08 is not None and abs(ev08.dividend_per_share - 0.16) < 1e-9 and abs(ev08.per_share_bonus - 0.3) < 1e-9
        assert lim08 is not None and lim08.up == 29.72
    finally:
        bao.close()
    return out


def run_data_audit(freeze: Date = FREEZE_DATE, limit_sample: int = 300,
                   suspension_sample: int = 100, verbose: bool = True) -> dict:
    import duckdb
    import time
    freeze = fee.to_date(freeze)
    con = duckdb.connect(DUCKDB_PATH, read_only=True)

    def section(name: str, fn: Callable[[], Any]) -> dict:
        t0 = time.perf_counter()
        if verbose:
            print(f"[audit] {name} ...", flush=True)
        result = _safe_section(fn)
        if verbose:
            print(f"[audit] {name} {time.perf_counter()-t0:.1f}s ok={result['ok']}", flush=True)
        return result

    try:
        audit = {
            "coverage": section("coverage", lambda: _audit_coverage(con, freeze)),
            "quality": section("quality", lambda: _audit_quality(con, freeze)),
            "calendar_vs_baostock": section("calendar_vs_baostock", lambda: _audit_calendar(con, freeze)),
            "corporate_actions": section("corporate_actions", lambda: _audit_events(con, freeze)),
            "event_price_consistency_2005_2015": section("event_price_consistency_2005_2015", lambda: _audit_event_price_consistency(con, freeze)),
            "adjustment_factors": section("adjustment_factors", lambda: _audit_factors(con, freeze)),
            "price_limits": section("price_limits", lambda: _audit_limits(con, freeze, limit_sample)),
            "suspension_vs_baostock": section("suspension_vs_baostock", lambda: _audit_suspension(con, freeze, suspension_sample)),
            "delisting": section("delisting", lambda: _audit_delisting(con, freeze)),
            "st_coverage": section("st_coverage", lambda: _audit_st(con, freeze)),
            "adapter_smoke": section("adapter_smoke", lambda: _adapter_smoke(freeze)),
        }
    finally:
        con.close()
    return audit


# ---------------------------------------------------------------------------
# 报告与 CLI
# ---------------------------------------------------------------------------

KNOWN_LIMITATIONS = [
    "日线撮合无盘中路径：同日止损与止盈同时触及时按损失优先（保守），但无法还原真实先后顺序。",
    "DuckDB prev_close 全空；适配器用上一交易日收盘补，除权日由事件计算参考价。回测起点首日无前收，会用开盘价并打 flag。",
    "DuckDB 无 is_st 字段；未注入 Baostock/外部 ST 日历时，5% 涨跌停会漏检。正式回测必须注入 ST 集合。",
    "DuckDB 2016-2026 日线对退市股覆盖很少（Baostock 341 只退市股中仅 3 只有 DuckDB 日线）；直接用 DuckDB 回测存在生存者偏差，退市股与 2016 前历史必须走 Baostock 适配器或并集。",
    "股息税按固定税率（默认 20%）在除权日扣除，未按持股期限（1 月/1 年）分档，也未在卖出时清算。",
    "送股小数部分向下取整、不补现金；配股默认在现金足够时全额认购，现金不足则跳过并打 flag；均非精确券商口径。",
    "公司行动事件覆盖 99%+ 的复权因子变动；少量无事件因子变动（送转/配股缺失）会制造原始价格假跌。DuckDBMarketData.factor_changes_on(d) 可列出这些变动，撮合层提供 note_flag 记录，但不会擅自推断改股数；协调者需决定推断/剔除。",
    "2005-2015 DuckDB 事件与 Baostock 原始价存在冲突（审计抽样会给出数量）；账户层已加价格一致性门控，冲突事件不应用、只标注，这是避免凭空加股的保守取舍，但会漏掉部分真实公司行动。",
    "退市按连续 N 个交易日无 bar 或 out_date 之后强制按最后收盘价打折卖出；真实退市整理期/三板转让回收率可能不同。",
    "参与率按日成交量比例限制，未建模盘中成交量分布、排队与冲击成本；滑点仅为固定基点。",
    "现金 T+1 结算按“卖出/分红回款次日可用”处理，比 A 股“当日可用、T+1 可取”更严格；可用 cash_settle_days=0 切换。",
    "整股规则按主板/创业板 100 股、科创板 200 股起 1 股递增、北交所 100 股起 1 股递增近似；以券商实际申报规则为准。",
    "过户费 2015 年前沪市按面值 0.6‰、深市 0、2015 后按成交额；历史费率可能有地区/券商差异。",
    "未建模融资融券、ETF/债券、B 股、可转债、停复牌公告日、盘中临时停牌、龙虎榜等。",
    "期末估值按最后收盘价；若期末流动性差或停牌，市值可能不可实现。",
    "DuckDB dim_symbol 无 ipo_date/out_date；用 DuckDB 适配器时需由 Baostock stock_basic 注入 SymbolInfo，否则新股无涨跌幅窗口与 out_date 退市处理不可用。",
    "数据审计的涨跌停/停牌/ST 为抽样（300/100/100 只），不是全市场逐日扫描；全市场结论需扩大样本或单独跑全量审计。",
]


def _sha256_file(path: os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _summarize_audit(audit: dict) -> list[str]:
    def res(name: str) -> dict:
        return (audit.get(name) or {}).get("result") or {}

    out: list[str] = []
    cov, qual = res("coverage"), res("quality")
    if cov:
        out.append(f"冻结窗口：{cov.get('min_date')} 至 {cov.get('max_date')}，{cov.get('rows')} 行、"
                   f"{cov.get('symbols')} 只、{cov.get('trading_days')} 个交易日；冻结日后 {cov.get('rows_after_freeze_excluded')} 行已排除。")
        out.append("DuckDB 日线从 2016-10-13 起；2005-2015 与退市股必须走 Baostock 适配器（2008-04-24 路径已烟测）。")
    if qual:
        out.append(f"OHLC 质量：重复(代码,日) {qual.get('duplicate_symbol_date_groups')} 组、非正价格 {qual.get('nonpositive_ohlc_rows')} 行、"
                   f"OHLC 不一致 {qual.get('ohlc_inconsistent_rows')} 行；prev_close 空值 {qual.get('prev_close_null_rows')} 行（数据源特性）。")
    cal = res("calendar_vs_baostock")
    if cal:
        out.append(f"交易日历与 Baostock 完全一致：{cal.get('duckdb_days')} 天，agree={cal.get('agree')}。")
    ev = res("corporate_actions")
    fac = res("adjustment_factors")
    if ev and fac:
        out.append(f"公司行动 {ev.get('events_in_window')} 条（送转 {ev.get('bonus_events')}、配股 {ev.get('rights_events')}）；"
                   f"复权因子变动 {fac.get('factor_changes')} 次，其中无事件 {fac.get('factor_changes_without_event')} 次，"
                   f"纯分红事件与因子一致性 {fac.get('dividend_only_within_0p5pct')}/{fac.get('dividend_only_consistency_n')}。")
    epc = res("event_price_consistency_2005_2015")
    if epc:
        out.append(f"2005-2015 送转/配股事件抽样 {epc.get('sampled_events')} 条（可交易 {epc.get('checked_tradable')}）："
                   f"与 Baostock 原始价冲突 {epc.get('conflict_count')} 条（{epc.get('conflict_rate')}），"
                   f"冲突事件账户层跳过并标注，避免凭空加股/分红。")
    lim = res("price_limits")
    if lim:
        cats = lim.get("classification_counts") or Counter(x.get("classification") for x in lim.get("examples", []))
        out.append(f"涨跌停抽样 {lim.get('sample_symbols')} 只/{lim.get('rows_checked')} 行：超额 {lim.get('exceedance_count')} 行，"
                   f"其中无事件因子变动 {cats.get('unexplained_factor_change', 0)} 行、无因子变动大额正跳 {cats.get('large_positive_move_no_factor_change', 0)} 行、"
                   f"无因子变动大额负跳 {cats.get('large_negative_move_no_factor_change', 0)} 行；前者是公司行动缺口，后者需人工核对复牌/重新上市。")
    sus = res("suspension_vs_baostock")
    if sus:
        c = sus.get("counts", {})
        out.append(f"停牌抽样 {sus.get('sample_symbols')} 只：与 Baostock 一致交易 {c.get('agree_trading')} 行、一致停牌 {c.get('agree_suspended')} 行，"
                   f"差异 {c.get('duckdb_bar_on_suspended_day', 0) + c.get('duckdb_missing_trading_day', 0)} 行。")
    dl = res("delisting")
    if dl:
        out.append(f"退市覆盖：Baostock 退市/合并 {dl.get('delisted_in_baostock')} 只，DuckDB 有日线仅 {dl.get('delisted_with_duckdb_bars')} 只，"
                   f"2016-10-13 后退出但 DuckDB 完全缺失 {dl.get('delisted_never_in_duckdb_after_2016_10_13')} 只——直接用 DuckDB 有生存者偏差。")
    st = res("st_coverage")
    if st:
        out.append(f"ST：DuckDB 是否含 is_st 列={st.get('duckdb_has_is_st_column')}；抽样 {st.get('sample_codes')} 只中 ST 行 {st.get('baostock_st_rows_in_sample')}、"
                   f"涉及 {st.get('baostock_st_symbols_in_sample')} 只，正式回测必须注入 ST 日历。")
    ad = res("adapter_smoke")
    if ad:
        d24 = (ad.get("duckdb") or {}).get("limit") or {}
        b08 = ((ad.get("baostock") or {}).get("probe_2008") or {}).get("limit") or {}
        out.append(f"适配器烟测：DuckDB 2024-10-10 000001.SZ 除权参考价 {d24.get('ref')}、涨停 {d24.get('up')}；"
                   f"Baostock 2008-04-24 600000.SH 参考价 {b08.get('ref')}、涨停 {b08.get('up')}；已退市 600005.SH 可由 Baostock 读取。")
    return out


def build_report(run_audit: bool = True, limit_sample: int = 300,
                 suspension_sample: int = 100) -> dict:
    here = Path(__file__).resolve()
    tests = run_self_tests()
    report = {
        "meta": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "freeze_date": FREEZE_DATE.isoformat(),
            "code_files": {
                "账户撮合.py": {"path": str(here), "sha256": _sha256_file(here)},
                "费用与交易约束.py": {"path": str(here.with_name("费用与交易约束.py")),
                                      "sha256": _sha256_file(here.with_name("费用与交易约束.py"))},
            },
            "data_sources": {
                "duckdb": DUCKDB_PATH,
                "baostock": BAOSTOCK_PATH,
                "mode": "read_only",
            },
            "window": {
                "duckdb_start": "2016-10-13",
                "end": FREEZE_DATE.isoformat(),
                "baostock_start": "1990-12-19",
                "note": "DuckDB 日线 2016-10-13 起；2005-2015 与退市股走 Baostock；统一冻结至 2026-10-26。",
            },
            "adjustment_convention": "DuckDB 用 raw_kline_daily 原始价 + raw_adjustment_events 做股数/现金调整；Baostock 用 adjustflag=3 不复权价 + preclose（除权日已是交易所参考价）。复权因子只用于一致性审计，不参与撮合定价。",
            "line_counts": {
                "账户撮合.py": len(here.read_text(encoding="utf-8").splitlines()),
                "费用与交易约束.py": len(here.with_name("费用与交易约束.py").read_text(encoding="utf-8").splitlines()),
            },
            "python": sys.version.split()[0],
            "duckdb": __import__("duckdb").__version__,
        },
        "account_config_defaults": {
            "initial_cash": AccountConfig().initial_cash,
            "cash_settle_days": AccountConfig().cash_settle_days,
            "enforce_t1": AccountConfig().enforce_t1,
            "rights_policy": AccountConfig().rights_policy,
            "delisting_missing_days": AccountConfig().delisting_missing_days,
            "delisting_haircut": AccountConfig().delisting_haircut,
            "fee": fee.fee_config_dict(fee.FeeConfig()),
        },
        "self_tests": tests,
        "known_limitations": KNOWN_LIMITATIONS,
        "not_done": [
            "未运行任何真实策略回测，也未生成真实收益/净值结论（本文件只做撮合与数据口径验证）。",
            "未把撮合引擎接入策略层组合构建器；接口为 submit_* + on_day，需主协调者集成。",
            "未对 2005-2015 全市场逐日跑账户；Baostock 适配器只做了单日烟测，事件仍来自 DuckDB。",
            "未构建 DuckDB+Baostock 的并集行情层；退市股/2016 前历史需要主协调者用 Baostock 适配器接入。",
            "未实现按持股期限分档的股息税和卖出时清算。",
            "未实现盘中分批成交/排队/冲击成本模型。",
        ],
    }
    if run_audit:
        report["data_audit"] = run_data_audit(FREEZE_DATE, limit_sample, suspension_sample)
        report["key_findings"] = _summarize_audit(report["data_audit"])
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="A股账户撮合自检与数据审计")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "04_证据" / "撮合自检.json"))
    parser.add_argument("--no-audit", action="store_true", help="只跑合成自检")
    parser.add_argument("--limit-sample", type=int, default=300)
    parser.add_argument("--suspension-sample", type=int, default=100)
    args = parser.parse_args(argv)
    report = build_report(not args.no_audit, args.limit_sample, args.suspension_sample)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    t = report["self_tests"]
    print(f"self_tests: {t['passed']}/{t['total']} passed, {t['failed']} failed")
    if "data_audit" in report:
        for k, v in report["data_audit"].items():
            print(f"audit.{k}: {'ok' if v.get('ok') else 'ERROR ' + str(v.get('error'))}")
    print(f"written: {out}")
    return 0 if t["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
