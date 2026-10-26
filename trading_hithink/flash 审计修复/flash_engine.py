from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd

from flash_data import Bar, Candidate, FlashData


@dataclass(frozen=True)
class EngineConfig:
    initial_capital: float
    max_positions: int = 2
    entry_lag_sessions: int = 3
    holding_sessions: int = 15
    profit_target: float = 0.20
    take_profit_mode: str = "intraday"
    limit_rate: float = 0.10
    reject_open_limit_down: bool = False
    slippage: float = 0.001
    commission_rate: float = 0.00025
    minimum_commission: float = 5.0
    max_participation: float | None = None
    stamp_duty_sell: tuple[tuple[pd.Timestamp, float], ...] = ()
    transfer_fee_both_sides: tuple[tuple[pd.Timestamp, float], ...] = ()
    same_open_sale_proceeds_reused: bool = False
    allow_leverage: bool = False
    allow_short: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EngineConfig:
        data = dict(value)
        data["stamp_duty_sell"] = rate_schedule(data["stamp_duty_sell"])
        data["transfer_fee_both_sides"] = rate_schedule(data["transfer_fee_both_sides"])
        config = cls(**data)
        if config.initial_capital <= 0 or config.max_positions <= 0:
            raise ValueError("Capital and positions must be positive")
        if config.entry_lag_sessions < 1 or config.holding_sessions < 1:
            raise ValueError("Entry lag and holding sessions must be positive")
        if not 0 < config.limit_rate < 1 or not 0 <= config.slippage < 1:
            raise ValueError("Invalid limit rate or slippage")
        if not 0 < config.profit_target < 1:
            raise ValueError("Invalid profit target")
        if config.max_participation is not None and not 0 < config.max_participation <= 1:
            raise ValueError("Invalid participation cap")
        if config.take_profit_mode not in {"intraday", "next_open"}:
            raise ValueError("Unsupported take-profit mode")
        if config.allow_leverage or config.allow_short or config.same_open_sale_proceeds_reused:
            raise ValueError("This audit permits neither leverage, shorting, nor same-open proceeds reuse")
        return config


@dataclass
class Position:
    symbol: int
    thscode: str
    buy_date: pd.Timestamp
    signal_date: pd.Timestamp
    buy_order_id: str
    buy_shares: int
    shares: int
    available_shares: int
    bonus_shares: int
    buy_price: float
    buy_raw_open: float
    buy_gross: float
    buy_commission: float
    buy_transfer_fee: float
    buy_slippage_cost: float
    buy_total_cost: float
    entry_hfq_open: float
    last_close: float
    holding_sessions: int = 0
    dividend_receivable: float = 0.0
    pending_exit: dict[str, Any] | None = None
    exit_attempts: int = 0


@dataclass
class BacktestResult:
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    corporate_actions: pd.DataFrame
    positions_daily: pd.DataFrame
    open_positions: pd.DataFrame
    metrics: dict[str, Any]


ORDER_COLUMNS = [
    "order_id", "position_id", "date", "signal_date", "thscode", "side", "reason",
    "status", "status_reason", "attempt", "execution_timing", "shares", "raw_open",
    "reference_close", "limit_up", "limit_down", "trigger_price", "benchmark_price", "price", "gross",
    "commission", "stamp_duty", "transfer_fee", "slippage_cost", "dividend_receivable",
    "dividend_tax", "dividend_received", "cash_delta", "score", "amount20",
    "participation", "participation_cap", "slot_budget", "order_budget",
]
TRADE_COLUMNS = [
    "trade_id", "thscode", "buy_date", "sell_date", "buy_signal_date", "sell_signal_date",
    "buy_order_id", "sell_order_id", "shares", "buy_shares", "bonus_shares", "sell_shares",
    "buy_price", "sell_price", "buy_gross", "sell_gross", "buy_commission", "sell_commission",
    "sell_stamp_duty", "buy_transfer_fee", "sell_transfer_fee", "buy_slippage_cost",
    "sell_slippage_cost", "dividend_received_gross", "dividend_tax", "dividend_received",
    "buy_total_cost", "sell_net_proceeds", "net_pnl", "return", "holding_days",
    "holding_sessions", "exit_reason",
]
EVENT_COLUMNS = [
    "date", "thscode", "position_id", "event_type", "shares_before", "available_shares_before",
    "dividend_per_share", "per_share_bonus", "rights_ratio", "rights_price",
    "cash_equivalent_per_share", "dividend_added", "bonus_shares_added", "rights_action",
    "shares_after", "available_shares_after", "dividend_receivable_after", "assumption",
]
POSITION_COLUMNS = [
    "date", "thscode", "position_id", "shares", "available_shares", "buy_date",
    "holding_sessions", "last_close", "market_value", "dividend_receivable", "stale_valuation",
]
DAILY_COLUMNS = [
    "date", "cash_start", "cash", "market_value", "dividend_receivable", "equity",
    "daily_return", "drawdown", "exposure", "positions", "shares", "stale_valuation_positions",
    "buy_gross", "buy_commission", "buy_transfer_fee", "sell_gross", "sell_commission",
    "sell_stamp_duty", "sell_transfer_fee", "dividend_added", "dividend_settled",
    "dividend_tax", "dividend_cash_received", "same_open_sale_proceeds", "buy_budget_start",
    "buy_budget_end", "cash_flow_error", "equity_identity_error", "receivable_identity_error",
]


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def limit_price(reference: float, rate: float, upper: bool) -> float:
    multiplier = Decimal("1") + (decimal(rate) if upper else -decimal(rate))
    return float((decimal(reference) * multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def execution_price(raw: float, slippage: float, side: str, lower: float, upper: float) -> float:
    if side == "buy":
        slipped = decimal(raw) * (Decimal("1") + decimal(slippage))
        price = slipped.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    else:
        slipped = decimal(raw) * (Decimal("1") - decimal(slippage))
        price = slipped.quantize(Decimal("0.01"), rounding=ROUND_FLOOR)
    return float(min(max(price, decimal(lower)), decimal(upper)))


def rate_schedule(values: list[dict[str, Any]]) -> tuple[tuple[pd.Timestamp, float], ...]:
    result = []
    for item in values:
        date = pd.Timestamp(item["start"]).normalize()
        rate = float(item["rate"])
        if not finite(rate) or rate < 0:
            raise ValueError("Invalid fee schedule")
        result.append((date, rate))
    return tuple(sorted(result))


def rate_on(schedule: tuple[tuple[pd.Timestamp, float], ...], date: pd.Timestamp) -> float:
    rate = 0.0
    for start, value in schedule:
        if start > date:
            break
        rate = value
    return rate


def dividend_tax_rate(buy_date: pd.Timestamp, sell_date: pd.Timestamp) -> float:
    if sell_date <= buy_date + pd.DateOffset(months=1):
        return 0.20
    if sell_date <= buy_date + pd.DateOffset(years=1):
        return 0.10
    return 0.0


def close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tolerance * max(1.0, abs(float(left)), abs(float(right)))


def frame(rows: list[dict[str, Any]], columns: list[str], dates: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=columns)
    for name in dates:
        if name in result and not result.empty:
            result[name] = pd.to_datetime(result[name])
    return result


class Account:
    def __init__(self, data: FlashData, config: EngineConfig):
        self.data = data
        self.config = config
        self.cash = float(config.initial_capital)
        self.buy_budget = float(config.initial_capital)
        self.cash_start = float(config.initial_capital)
        self.previous_equity = float(config.initial_capital)
        self.running_peak = float(config.initial_capital)
        self.positions: dict[int, Position] = {}
        self.orders: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.daily: list[dict[str, Any]] = []
        self.position_rows: list[dict[str, Any]] = []
        self.order_sequence = 0
        self.trade_sequence = 0
        self.flows: dict[str, float] = {}
        self.bought_today: set[int] = set()
        self.sold_today: set[int] = set()

    def record_order(
        self,
        date: pd.Timestamp,
        signal_date: Any,
        thscode: str,
        side: str,
        reason: str,
        status: str,
        status_reason: str = "",
        position_id: str = "",
        attempt: int = 1,
        **values: Any,
    ) -> str:
        self.order_sequence += 1
        order_id = f"O{self.order_sequence:08d}"
        row = {column: 0.0 for column in ORDER_COLUMNS}
        row.update({
            "order_id": order_id, "position_id": position_id, "date": date,
            "signal_date": signal_date, "thscode": thscode, "side": side, "reason": reason,
            "status": status, "status_reason": status_reason, "position_id": position_id,
            "attempt": attempt, "execution_timing": "none", "shares": 0,
            "raw_open": np.nan, "reference_close": np.nan, "limit_up": np.nan,
            "limit_down": np.nan, "trigger_price": np.nan, "benchmark_price": np.nan, "price": np.nan,
            "gross": 0.0, "score": np.nan, "amount20": np.nan, "participation": np.nan,
            "slot_budget": np.nan, "order_budget": np.nan,
        })
        unknown = set(values) - set(ORDER_COLUMNS)
        if unknown:
            raise RuntimeError(f"Unknown order fields: {sorted(unknown)}")
        row.update(values)
        self.orders.append(row)
        return order_id

    def target_price(self, position: Position, bar: Bar) -> float:
        raw = position.entry_hfq_open * (1 + self.config.profit_target) / bar.backward_factor
        return float(decimal(raw).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    def apply_corporate_action(self, date_index: int, symbol: int) -> None:
        position = self.positions[symbol]
        date = self.data.dates[date_index]
        action = self.data.events.get((date_index, symbol))
        bar = self.data.bar(date_index, symbol)
        shares_before = position.shares
        available_before = position.available_shares
        dividend_added = 0.0
        bonus_added = 0
        cash_equivalent = 0.0
        event_type = ""
        assumption = ""
        rights_action = "not_applicable"
        if action is not None:
            event_type = "explicit_corporate_action"
            dividend_added = shares_before * action.dividend_per_share
            bonus_added = int((decimal(shares_before) * decimal(action.per_share_bonus)).to_integral_value(rounding=ROUND_FLOOR))
            position.shares += bonus_added
            position.bonus_shares += bonus_added
            rights_action = "not_exercised" if action.rights_ratio > 0 else "not_applicable"
            assumption = "explicit_event_records_used; rights_not_exercised"
        elif bar is not None and bar.factor_only_adjustment:
            event_type = "factor_only_cash_equivalent"
            if not finite(bar.previous_actual_close) or not finite(bar.reference_close):
                raise ValueError("Factor-only event lacks actual/reference close")
            cash_equivalent = bar.previous_actual_close - bar.reference_close
            if cash_equivalent < -1e-10:
                raise ValueError("Negative factor-only cash equivalent")
            cash_equivalent = max(0.0, cash_equivalent)
            dividend_added = shares_before * cash_equivalent
            assumption = "cash-equivalent receivable from factor-restored reference; not a reconstruction of the missing corporate action"
        if not event_type:
            return
        position.dividend_receivable += dividend_added
        self.flows["dividend_added"] += dividend_added
        if position.shares < 0 or position.available_shares < 0 or position.dividend_receivable < 0:
            raise AssertionError("Negative position state after corporate action")
        self.events.append({
            "date": date, "thscode": position.thscode, "position_id": position.buy_order_id,
            "event_type": event_type, "shares_before": shares_before,
            "available_shares_before": available_before,
            "dividend_per_share": action.dividend_per_share if action else 0.0,
            "per_share_bonus": action.per_share_bonus if action else 0.0,
            "rights_ratio": action.rights_ratio if action else 0.0,
            "rights_price": action.rights_price if action else 0.0,
            "cash_equivalent_per_share": cash_equivalent,
            "dividend_added": dividend_added, "bonus_shares_added": bonus_added,
            "rights_action": rights_action, "shares_after": position.shares,
            "available_shares_after": position.available_shares,
            "dividend_receivable_after": position.dividend_receivable,
            "assumption": assumption,
        })

    def sell(
        self,
        position: Position,
        date_index: int,
        reason: str,
        timing: str,
        benchmark_price: float,
        trigger_price: float = np.nan,
        signal_date: Any = pd.NaT,
    ) -> bool:
        date = self.data.dates[date_index]
        symbol = position.symbol
        bar = self.data.bar(date_index, symbol)
        pending = position.pending_exit
        if pending is not None:
            reason = pending["reason"]
            signal_date = pending["signal_date"]
        attempt = position.exit_attempts + 1
        if date <= position.buy_date:
            raise AssertionError("T+1 forbids selling on the buy date")
        if symbol in self.bought_today:
            raise AssertionError("Same-stock same-day buy/sell")
        common: dict[str, Any] = {
            "position_id": position.buy_order_id,
            "attempt": attempt,
            "execution_timing": timing,
            "shares": position.shares,
            "trigger_price": trigger_price,
            "benchmark_price": benchmark_price,
        }
        if bar is not None:
            common["raw_open"] = bar.open
            if finite(bar.reference_close) and bar.reference_close > 0:
                upper = limit_price(bar.reference_close, self.config.limit_rate, True)
                lower = limit_price(bar.reference_close, self.config.limit_rate, False)
                common.update(reference_close=bar.reference_close, limit_up=upper, limit_down=lower)

        def defer(detail: str) -> bool:
            position.exit_attempts += 1
            position.pending_exit = {"reason": reason, "signal_date": signal_date}
            self.record_order(
                date, signal_date, position.thscode, "sell", reason, "deferred", detail, **common
            )
            return False

        if bar is None:
            return defer("no_bar")
        if not all(finite(value) and value > 0 for value in [bar.open, bar.high, bar.low, bar.close, bar.reference_close]):
            return defer("invalid_bar")
        if not finite(bar.volume) or bar.volume <= 0:
            return defer("zero_volume")
        upper = common["limit_up"]
        lower = common["limit_down"]
        if timing == "open" and bar.open <= lower + 1e-12:
            return defer("open_limit_down")
        if timing == "intraday":
            if not finite(trigger_price) or bar.high + 1e-12 < trigger_price:
                return False
            if abs(bar.high - bar.low) < 1e-12 and bar.close <= lower + 1e-12:
                return defer("one_price_limit_down")
        if timing == "close" and bar.close <= lower + 1e-12:
            return defer("close_limit_down")
        if position.available_shares != position.shares:
            return defer("shares_not_sellable")
        if position.shares <= 0:
            raise AssertionError("A held position must have positive shares")
        if not finite(benchmark_price) or benchmark_price <= 0:
            return defer("invalid_execution_price")
        price = execution_price(benchmark_price, self.config.slippage, "sell", lower, upper)
        shares = position.shares
        gross = shares * price
        commission = max(self.config.minimum_commission, gross * self.config.commission_rate)
        stamp_duty = gross * rate_on(self.config.stamp_duty_sell, date)
        transfer_fee = gross * rate_on(self.config.transfer_fee_both_sides, date)
        dividend_gross = position.dividend_receivable
        dividend_tax = dividend_gross * dividend_tax_rate(position.buy_date, date)
        dividend_net = dividend_gross - dividend_tax
        sale_net = gross - commission - stamp_duty - transfer_fee
        cash_delta = sale_net + dividend_net
        self.cash += cash_delta
        if -1e-8 < self.cash < 0:
            self.cash = 0.0
        if self.cash < 0:
            raise AssertionError("Negative cash after sell")
        slippage_cost = (benchmark_price - price) * shares
        position.exit_attempts += 1
        order_id = self.record_order(
            date, signal_date, position.thscode, "sell", reason, "filled",
            price=price, gross=gross, commission=commission, stamp_duty=stamp_duty,
            transfer_fee=transfer_fee, slippage_cost=slippage_cost,
            dividend_receivable=dividend_gross, dividend_tax=dividend_tax,
            dividend_received=dividend_net, cash_delta=cash_delta, **common,
        )
        self.flows["sell_gross"] += gross
        self.flows["sell_commission"] += commission
        self.flows["sell_stamp_duty"] += stamp_duty
        self.flows["sell_transfer_fee"] += transfer_fee
        self.flows["dividend_settled"] += dividend_gross
        self.flows["dividend_tax"] += dividend_tax
        self.flows["dividend_cash_received"] += dividend_net
        self.trade_sequence += 1
        holding_sessions = position.holding_sessions + (1 if timing == "open" else 0)
        net_pnl = cash_delta - position.buy_total_cost
        self.trades.append({
            "trade_id": f"T{self.trade_sequence:08d}", "thscode": position.thscode,
            "buy_date": position.buy_date, "sell_date": date,
            "buy_signal_date": position.signal_date, "sell_signal_date": signal_date,
            "buy_order_id": position.buy_order_id, "sell_order_id": order_id,
            "shares": shares, "buy_shares": position.buy_shares,
            "bonus_shares": position.bonus_shares, "sell_shares": shares,
            "buy_price": position.buy_price, "sell_price": price,
            "buy_gross": position.buy_gross, "sell_gross": gross,
            "buy_commission": position.buy_commission, "sell_commission": commission,
            "sell_stamp_duty": stamp_duty, "buy_transfer_fee": position.buy_transfer_fee,
            "sell_transfer_fee": transfer_fee, "buy_slippage_cost": position.buy_slippage_cost,
            "sell_slippage_cost": slippage_cost,
            "dividend_received_gross": dividend_gross, "dividend_tax": dividend_tax,
            "dividend_received": dividend_net, "buy_total_cost": position.buy_total_cost,
            "sell_net_proceeds": sale_net, "net_pnl": net_pnl,
            "return": net_pnl / position.buy_total_cost,
            "holding_days": (date - position.buy_date).days,
            "holding_sessions": holding_sessions, "exit_reason": reason,
        })
        position.dividend_receivable = 0.0
        position.shares = 0
        position.available_shares = 0
        self.sold_today.add(symbol)
        del self.positions[symbol]
        return True

    def buy(self, candidate: Candidate, date_index: int, final_session: bool) -> bool:
        date = self.data.dates[date_index]
        symbol = candidate.symbol
        thscode = self.data.symbols[symbol]
        signal_date = self.data.dates[candidate.signal_index]
        if candidate.signal_index >= date_index:
            raise AssertionError("Buy signal must precede execution")
        slot_budget = self.previous_equity / self.config.max_positions
        order_budget = min(self.buy_budget, slot_budget)
        common: dict[str, Any] = {
            "position_id": "", "attempt": 1, "execution_timing": "open",
            "score": candidate.score, "amount20": candidate.amount20,
            "slot_budget": slot_budget, "order_budget": order_budget,
        }

        def reject(detail: str, **values: Any) -> bool:
            self.record_order(
                date, signal_date, thscode, "buy", "candidate", "rejected", detail, **common, **values
            )
            return False

        if final_session:
            return reject("end_of_test")
        if symbol in self.sold_today:
            return reject("same_day_sell")
        if symbol in self.bought_today:
            return reject("same_day_buy")
        if symbol in self.positions:
            return reject("already_held")
        if len(self.positions) >= self.config.max_positions:
            return reject("position_limit")
        bar = self.data.bar(date_index, symbol)
        if bar is None:
            return reject("no_bar")
        if not all(finite(value) and value > 0 for value in [
            bar.open, bar.high, bar.low, bar.close, bar.reference_close, bar.hfq_open
        ]):
            return reject("invalid_bar")
        if not finite(bar.volume) or bar.volume <= 0:
            return reject("zero_volume")
        upper = limit_price(bar.reference_close, self.config.limit_rate, True)
        lower = limit_price(bar.reference_close, self.config.limit_rate, False)
        common.update(raw_open=bar.open, reference_close=bar.reference_close, limit_up=upper, limit_down=lower)
        if bar.open >= upper:
            return reject("open_limit_up")
        if self.config.reject_open_limit_down and bar.open <= lower:
            return reject("open_limit_down")
        price = execution_price(bar.open, self.config.slippage, "buy", lower, upper)
        lots = int(np.floor((order_budget + 1e-9) / (price * 100)))
        participation_lots = lots
        participation_budget = np.nan
        if self.config.max_participation is not None:
            participation_budget = candidate.amount20 * self.config.max_participation
            participation_lots = int(np.floor((participation_budget + 1e-9) / (price * 100)))
            lots = min(lots, participation_lots)
        common["participation_cap"] = participation_budget
        quantity = 0
        gross = commission = transfer_fee = total_cost = 0.0
        while lots > 0:
            candidate_quantity = lots * 100
            candidate_gross = candidate_quantity * price
            candidate_commission = max(self.config.minimum_commission, candidate_gross * self.config.commission_rate)
            candidate_transfer = candidate_gross * rate_on(self.config.transfer_fee_both_sides, date)
            candidate_total = candidate_gross + candidate_commission + candidate_transfer
            if (
                candidate_total <= order_budget + 1e-9
                and candidate_total <= self.buy_budget + 1e-9
                and candidate_total <= self.cash + 1e-9
            ):
                quantity = candidate_quantity
                gross = candidate_gross
                commission = candidate_commission
                transfer_fee = candidate_transfer
                total_cost = candidate_total
                break
            lots -= 1
        if quantity == 0:
            if self.config.max_participation is not None and participation_lots == 0:
                detail = "participation_cap"
            else:
                detail = "slot_budget_including_fees" if slot_budget <= self.buy_budget else "insufficient_cash_including_fees"
            return reject(detail, price=price)
        if quantity % 100:
            raise AssertionError("Buy is not a whole lot")
        if total_cost > slot_budget + 1e-8 or total_cost > self.buy_budget + 1e-8 or total_cost > self.cash + 1e-8:
            raise AssertionError("Buy exceeds cash or slot budget")
        slippage_cost = (price - bar.open) * quantity
        self.cash -= total_cost
        self.buy_budget -= total_cost
        if -1e-8 < self.cash < 0:
            self.cash = 0.0
        if -1e-8 < self.buy_budget < 0:
            self.buy_budget = 0.0
        if self.cash < 0 or self.buy_budget < 0:
            raise AssertionError("Negative cash after buy")
        self.flows["buy_gross"] += gross
        self.flows["buy_commission"] += commission
        self.flows["buy_transfer_fee"] += transfer_fee
        participation = gross / candidate.amount20 if candidate.amount20 > 0 else np.nan
        order_id = self.record_order(
            date, signal_date, thscode, "buy", "candidate", "filled",
            shares=quantity, price=price, gross=gross, commission=commission,
            transfer_fee=transfer_fee, slippage_cost=slippage_cost,
            cash_delta=-total_cost, participation=participation, **common,
        )
        self.positions[symbol] = Position(
            symbol=symbol, thscode=thscode, buy_date=date, signal_date=signal_date,
            buy_order_id=order_id, buy_shares=quantity, shares=quantity, available_shares=0,
            bonus_shares=0, buy_price=price, buy_raw_open=bar.open, buy_gross=gross,
            buy_commission=commission, buy_transfer_fee=transfer_fee,
            buy_slippage_cost=slippage_cost, buy_total_cost=total_cost,
            entry_hfq_open=bar.hfq_open, last_close=bar.open,
        )
        self.bought_today.add(symbol)
        return True

    def run(self) -> BacktestResult:
        flow_names = [
            "buy_gross", "buy_commission", "buy_transfer_fee", "sell_gross",
            "sell_commission", "sell_stamp_duty", "sell_transfer_fee",
            "dividend_added", "dividend_settled", "dividend_tax", "dividend_cash_received",
        ]
        final_index = len(self.data.dates) - 1
        for date_index, date in enumerate(self.data.dates):
            final_session = date_index == final_index
            self.flows = {name: 0.0 for name in flow_names}
            self.cash_start = self.cash
            self.buy_budget = self.cash_start
            receivable_start = sum(position.dividend_receivable for position in self.positions.values())
            self.bought_today = set()
            self.sold_today = set()
            held_at_open = set(self.positions)
            for symbol in list(self.positions):
                position = self.positions[symbol]
                if position.buy_date >= date:
                    raise AssertionError("A new buy reached a later open incorrectly")
                position.available_shares = position.shares
            for symbol in list(self.positions):
                self.apply_corporate_action(date_index, symbol)
            for symbol in list(self.positions):
                position = self.positions[symbol]
                bar = self.data.bar(date_index, symbol)
                trigger = np.nan
                if (
                    position.pending_exit is None
                    and self.config.take_profit_mode == "intraday"
                    and date > position.buy_date
                    and bar is not None
                ):
                    trigger = self.target_price(position, bar)
                    if finite(bar.open) and bar.open >= trigger:
                        position.pending_exit = {"reason": "profit_target", "signal_date": pd.NaT}
                if position.pending_exit is not None:
                    if position.pending_exit["reason"] == "profit_target" and bar is not None:
                        trigger = self.target_price(position, bar)
                    self.sell(
                        position, date_index, position.pending_exit["reason"], "open",
                        bar.open if bar is not None else np.nan, trigger_price=trigger,
                    )
            signal_index = date_index - self.config.entry_lag_sessions
            if signal_index >= 0:
                candidates = [
                    candidate for candidate in self.data.candidates.get(signal_index, [])
                    if candidate.symbol not in held_at_open
                ]
                vacancies = max(0, self.config.max_positions - len(self.positions))
                for candidate in candidates[:vacancies]:
                    self.buy(candidate, date_index, final_session)
            for symbol in list(self.positions):
                position = self.positions[symbol]
                position.holding_sessions += 1
                bar = self.data.bar(date_index, symbol)
                stale = bar is None
                if bar is not None and finite(bar.close) and bar.close > 0:
                    position.last_close = bar.close
                if not final_session and position.pending_exit is None and date > position.buy_date and bar is not None:
                    target = self.target_price(position, bar)
                    target_touched = finite(bar.high) and bar.high + 1e-12 >= target
                    if self.config.take_profit_mode == "intraday" and target_touched:
                        self.sell(
                            position, date_index, "profit_target", "intraday", target,
                            trigger_price=target,
                        )
                    elif self.config.take_profit_mode == "next_open" and target_touched:
                        position.exit_attempts += 1
                        position.pending_exit = {"reason": "profit_target", "signal_date": date}
                        self.record_order(
                            date, date, position.thscode, "sell", "profit_target", "pending",
                            "triggered_waiting_next_open", position_id=position.buy_order_id,
                            attempt=position.exit_attempts, execution_timing="next_open",
                            shares=position.shares, raw_open=bar.open,
                            reference_close=bar.reference_close, trigger_price=target,
                        )
                    if symbol in self.positions and self.positions[symbol].pending_exit is None and position.holding_sessions >= self.config.holding_sessions:
                        self.sell(position, date_index, "max_holding", "close", bar.close)
                elif not final_session and position.pending_exit is None and position.holding_sessions >= self.config.holding_sessions:
                    self.sell(position, date_index, "max_holding", "close", np.nan)
                if symbol in self.positions:
                    self.position_rows.append({
                        "date": date, "thscode": position.thscode,
                        "position_id": position.buy_order_id, "shares": position.shares,
                        "available_shares": position.available_shares, "buy_date": position.buy_date,
                        "holding_sessions": position.holding_sessions, "last_close": position.last_close,
                        "market_value": position.shares * position.last_close,
                        "dividend_receivable": position.dividend_receivable,
                        "stale_valuation": stale,
                    })
            market_value = sum(position.shares * position.last_close for position in self.positions.values())
            receivable = sum(position.dividend_receivable for position in self.positions.values())
            equity = self.cash + market_value + receivable
            expected_cash = (
                self.cash_start - self.flows["buy_gross"] - self.flows["buy_commission"]
                - self.flows["buy_transfer_fee"] + self.flows["sell_gross"]
                - self.flows["sell_commission"] - self.flows["sell_stamp_duty"]
                - self.flows["sell_transfer_fee"] + self.flows["dividend_settled"]
                - self.flows["dividend_tax"]
            )
            expected_receivable = receivable_start + self.flows["dividend_added"] - self.flows["dividend_settled"]
            expected_budget = self.cash_start - self.flows["buy_gross"] - self.flows["buy_commission"] - self.flows["buy_transfer_fee"]
            if self.cash < -1e-8:
                raise AssertionError("Negative cash at close")
            if not close(self.cash, expected_cash):
                raise AssertionError(f"Cash conservation failed on {date}: {self.cash} != {expected_cash}")
            if not close(receivable, expected_receivable):
                raise AssertionError("Dividend receivable conservation failed")
            if not close(self.buy_budget, expected_budget):
                raise AssertionError("Same-open sale proceeds entered the buy budget")
            self.running_peak = max(self.running_peak, equity)
            open_sale_proceeds = sum(
                row["cash_delta"] for row in self.orders
                if row["date"] == date and row["side"] == "sell"
                and row["status"] == "filled" and row["execution_timing"] == "open"
            )
            stale_count = sum(
                1 for row in self.position_rows
                if row["date"] == date and row["stale_valuation"]
            )
            self.daily.append({
                "date": date, "cash_start": self.cash_start, "cash": self.cash,
                "market_value": market_value, "dividend_receivable": receivable, "equity": equity,
                "daily_return": equity / self.previous_equity - 1.0,
                "drawdown": equity / self.running_peak - 1.0,
                "exposure": market_value / equity if equity > 0 else np.nan,
                "positions": len(self.positions),
                "shares": sum(position.shares for position in self.positions.values()),
                "stale_valuation_positions": stale_count,
                **{name: self.flows[name] for name in flow_names},
                "same_open_sale_proceeds": open_sale_proceeds,
                "buy_budget_start": self.cash_start, "buy_budget_end": self.buy_budget,
                "cash_flow_error": self.cash - expected_cash,
                "equity_identity_error": equity - self.cash - market_value - receivable,
                "receivable_identity_error": receivable - expected_receivable,
            })
            self.previous_equity = equity
        daily = frame(self.daily, DAILY_COLUMNS, ["date"])
        trades = frame(self.trades, TRADE_COLUMNS, ["buy_date", "sell_date", "buy_signal_date", "sell_signal_date"])
        orders = frame(self.orders, ORDER_COLUMNS, ["date", "signal_date"])
        corporate_actions = frame(self.events, EVENT_COLUMNS, ["date"])
        positions_daily = frame(self.position_rows, POSITION_COLUMNS, ["date", "buy_date"])
        open_positions = positions_daily[positions_daily["date"] == self.data.dates[-1]].copy() if not positions_daily.empty else positions_daily.copy()
        equity = daily["equity"].to_numpy(dtype=float)
        elapsed = (daily["date"].iloc[-1] - daily["date"].iloc[0]).days / 365.25
        cagr = (equity[-1] / self.config.initial_capital) ** (1 / elapsed) - 1 if elapsed > 0 else np.nan
        peaks = np.maximum.accumulate(np.concatenate([[self.config.initial_capital], equity]))
        drawdown = np.concatenate([[self.config.initial_capital], equity]) / peaks - 1
        daily_returns = daily["daily_return"].dropna()
        wins = trades[trades["net_pnl"] > 0]["net_pnl"] if not trades.empty else pd.Series(dtype=float)
        losses = trades[trades["net_pnl"] < 0]["net_pnl"] if not trades.empty else pd.Series(dtype=float)
        filled = orders[orders["status"] == "filled"]
        filled_buys = filled[filled["side"] == "buy"]
        max_cash_error = float(daily["cash_flow_error"].abs().max())
        max_equity_error = float(daily["equity_identity_error"].abs().max())
        max_receivable_error = float(daily["receivable_identity_error"].abs().max())
        linked = (
            set(trades["buy_order_id"]).issubset(set(filled_buys["order_id"]))
            and set(trades["sell_order_id"]).issubset(set(filled[filled["side"] == "sell"]["order_id"]))
        )
        reconciled = (
            max_cash_error < 1e-7 and max_equity_error < 1e-7 and max_receivable_error < 1e-7
            and bool((daily["cash"] >= -1e-8).all()) and linked
        )
        metrics = {
            "start": str(daily["date"].iloc[0].date()), "end": str(daily["date"].iloc[-1].date()),
            "initial_capital": self.config.initial_capital, "final_equity": float(equity[-1]),
            "total_return": float(equity[-1] / self.config.initial_capital - 1),
            "cagr": float(cagr), "max_drawdown": float(drawdown.min()),
            "max_drawdown_abs": float(-drawdown.min()),
            "calmar": float(cagr / abs(drawdown.min())) if drawdown.min() < 0 else np.nan,
            "sharpe": float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else np.nan,
            "trade_count": int(len(trades)), "ending_open_positions": int(len(self.positions)),
            "win_rate": float((trades["return"] > 0).mean()) if not trades.empty else np.nan,
            "average_trade_return": float(trades["return"].mean()) if not trades.empty else np.nan,
            "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else np.nan,
            "average_holding_sessions": float(trades["holding_sessions"].mean()) if not trades.empty else np.nan,
            "time_in_market": float((daily["positions"] > 0).mean()),
            "buy_order_count": int((orders["side"] == "buy").sum()),
            "rejected_buy_count": int(((orders["side"] == "buy") & (orders["status"] == "rejected")).sum()),
            "deferred_sell_count": int(((orders["side"] == "sell") & (orders["status"] == "deferred")).sum()),
            "profit_target_exits": int((trades["exit_reason"] == "profit_target").sum()) if not trades.empty else 0,
            "time_exits": int((trades["exit_reason"] == "max_holding").sum()) if not trades.empty else 0,
            "corporate_action_count": int(len(corporate_actions)),
            "factor_only_adjustment_count": int((corporate_actions["event_type"] == "factor_only_cash_equivalent").sum()) if not corporate_actions.empty else 0,
            "dividend_added": float(corporate_actions["dividend_added"].sum()) if not corporate_actions.empty else 0.0,
            "dividend_tax": float(trades["dividend_tax"].sum()) if not trades.empty else 0.0,
            "max_buy_participation": float(filled_buys["participation"].max()) if not filled_buys.empty else np.nan,
            "max_abs_cash_flow_error": max_cash_error,
            "max_abs_equity_identity_error": max_equity_error,
            "max_abs_receivable_identity_error": max_receivable_error,
            "final_reconciliation_passed": bool(reconciled),
            "entry_lag_sessions": self.config.entry_lag_sessions,
            "holding_sessions": self.config.holding_sessions,
            "max_positions": self.config.max_positions,
            "limit_rate": self.config.limit_rate,
            "take_profit_mode": self.config.take_profit_mode,
            "evaluation_status": "historical_repair_not_independent_holdout",
        }
        if not reconciled:
            raise AssertionError("Final account reconciliation failed")
        return BacktestResult(daily, trades, orders, corporate_actions, positions_daily, open_positions, metrics)


def run_backtest(data: FlashData, config: EngineConfig) -> BacktestResult:
    return Account(data, config).run()


def annual_returns(daily: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    equity = daily.set_index("date")["equity"].sort_index()
    year_end = equity.resample("YE").last()
    returns = year_end.pct_change()
    if not returns.empty:
        returns.iloc[0] = year_end.iloc[0] / initial_capital - 1
    return pd.DataFrame({"year": returns.index.year, "return": returns.to_numpy()})


def drawdown_spells(daily: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    equity = daily.set_index("date")["equity"].sort_index()
    peaks = np.maximum.accumulate(np.concatenate([[initial_capital], equity.to_numpy()]))[1:]
    drawdown = equity.to_numpy() / peaks - 1
    rows = []
    active = None
    for date, value, current_equity in zip(equity.index, drawdown, equity):
        if value < 0 and active is None:
            active = {"start": date, "trough": date, "depth": value, "peak_equity": current_equity / (1 + value)}
        elif value < 0 and active is not None:
            if value < active["depth"]:
                active["depth"] = value
                active["trough"] = date
        elif value >= 0 and active is not None:
            active.update(recovery=date, calendar_days=(date - active["start"]).days)
            rows.append(active)
            active = None
    if active is not None:
        active.update(recovery=pd.NaT, calendar_days=(equity.index[-1] - active["start"]).days)
        rows.append(active)
    return pd.DataFrame(rows, columns=["start", "trough", "recovery", "depth", "peak_equity", "calendar_days"])
