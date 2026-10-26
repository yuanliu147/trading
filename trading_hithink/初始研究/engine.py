from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    event_ledger: pd.DataFrame
    metrics: dict[str, Any]


@dataclass
class _Position:
    thscode: str
    buy_date: pd.Timestamp
    buy_signal_date: pd.Timestamp
    buy_order_id: str
    buy_shares: int
    shares: int
    available_shares: int
    bonus_shares: int
    buy_price: float
    buy_gross: float
    buy_commission: float
    buy_transfer_fee: float
    buy_slippage_cost: float
    buy_total_cost: float
    last_close: float
    holding_sessions: int = 0
    dividend_receivable: float = 0.0
    peak_position_value: float | None = None
    pending_exit: dict[str, Any] | None = None
    exit_attempts: int = 0


_DAILY_REQUIRED_COLUMNS = [
    "date",
    "thscode",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]
_DAILY_OPTIONAL_COLUMNS = [
    "prev_close",
    "reference_close",
    "previous_actual_close",
    "factor_only_adjustment",
]
_DAILY_INPUT_COLUMNS = _DAILY_REQUIRED_COLUMNS + _DAILY_OPTIONAL_COLUMNS

_ORDER_COLUMNS = [
    "order_id",
    "date",
    "signal_date",
    "thscode",
    "side",
    "reason",
    "status",
    "status_reason",
    "attempt",
    "shares",
    "raw_open",
    "reference_close",
    "price",
    "gross",
    "commission",
    "stamp_duty",
    "transfer_fee",
    "slippage_cost",
    "dividend_receivable",
    "dividend_tax",
    "dividend_received",
    "cash_delta",
    "limit_up",
    "limit_down",
    "score",
    "amount20",
    "participation_cap",
    "slot_budget",
    "order_budget",
]

_TRADE_COLUMNS = [
    "trade_id",
    "thscode",
    "buy_date",
    "sell_date",
    "buy_signal_date",
    "sell_signal_date",
    "buy_order_id",
    "sell_order_id",
    "shares",
    "buy_shares",
    "bonus_shares",
    "sell_shares",
    "buy_price",
    "sell_price",
    "buy_gross",
    "sell_gross",
    "buy_commission",
    "sell_commission",
    "buy_stamp_duty",
    "sell_stamp_duty",
    "buy_transfer_fee",
    "sell_transfer_fee",
    "buy_slippage_cost",
    "sell_slippage_cost",
    "dividend_receivable",
    "dividend_received_gross",
    "dividend_tax",
    "dividend_received",
    "buy_total_cost",
    "sell_net_proceeds",
    "net_pnl",
    "return",
    "holding_days",
    "holding_sessions",
    "exit_reason",
]

_FACTOR_ONLY_ASSUMPTION = (
    "Cash-equivalent receivable equals held shares times "
    "previous_actual_close-reference_close; this is a scenario assumption, "
    "not a reconstruction of the actual corporate action."
)

_EVENT_COLUMNS = [
    "date",
    "ex_date",
    "thscode",
    "position_id",
    "event_type",
    "shares_before",
    "available_shares_before",
    "reference_close",
    "previous_actual_close",
    "cash_equivalent_per_share",
    "dividend_per_share",
    "per_share_bonus",
    "rights_ratio",
    "dividend_receivable_added",
    "bonus_shares_added",
    "rights_shares_added",
    "rights_cash_paid",
    "shares_after",
    "available_shares_after",
    "dividend_receivable_after",
    "rights_action",
    "factor_only_adjustment",
    "adjustment_assumption",
]

_FLOW_KEYS = [
    "buy_gross",
    "buy_commission",
    "buy_transfer_fee",
    "sell_gross",
    "sell_commission",
    "sell_stamp_duty",
    "sell_transfer_fee",
    "dividend_receivable_added",
    "dividend_receivable_settled",
    "dividend_tax",
    "dividend_cash_received",
]

_DAILY_COLUMNS = [
    "date",
    "cash_start",
    "cash",
    "market_value",
    "dividend_receivable_start",
    "dividend_receivable",
    "dividend_receivable_added",
    "dividend_receivable_settled",
    "dividend_tax",
    "dividend_cash_received",
    "buy_gross",
    "buy_commission",
    "buy_transfer_fee",
    "sell_gross",
    "sell_commission",
    "sell_stamp_duty",
    "sell_transfer_fee",
    "same_open_sale_proceeds",
    "buy_budget_start",
    "buy_budget_end",
    "equity",
    "daily_return",
    "drawdown",
    "exposure",
    "positions",
    "shares",
    "held_during_session",
    "stale_valuation_positions",
    "cash_flow_error",
    "equity_identity_error",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _is_close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tolerance * max(
        1.0, abs(float(left)), abs(float(right))
    )


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("dates must not be missing")
    if result.tzinfo is not None:
        result = result.tz_localize(None)
    return result.normalize()


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise").map(_timestamp)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _limit_price(prev_close: float, limit_rate: float, upper: bool) -> float:
    multiplier = Decimal("1") + (
        _decimal(limit_rate) if upper else -_decimal(limit_rate)
    )
    price = _decimal(prev_close) * multiplier
    return float(price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _execution_price(
    raw_open: float,
    slippage: float,
    side: str,
    limit_down: float,
    limit_up: float,
) -> float:
    if side == "buy":
        slipped = _decimal(raw_open) * (Decimal("1") + _decimal(slippage))
        price = slipped.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    else:
        slipped = _decimal(raw_open) * (Decimal("1") - _decimal(slippage))
        price = slipped.quantize(Decimal("0.01"), rounding=ROUND_FLOOR)
    price = min(max(price, _decimal(limit_down)), _decimal(limit_up))
    return float(price)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _boolean_flag(value: Any, name: str) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    elif _finite(value) and float(value) in {0.0, 1.0}:
        return bool(value)
    raise ValueError(f"{name} must contain boolean or 0/1 values")


def _execution_reference_close(bar: Any) -> float | None:
    for value in (bar.reference_close, bar.prev_close):
        if _finite(value) and float(value) > 0:
            return float(value)
    return None


def _positive_number(config: dict, key: str, default: float | None = None) -> float:
    value = config.get(key, default)
    if value is None or not _finite(value) or float(value) <= 0:
        raise ValueError(f"{key} must be a positive finite number")
    return float(value)


def _optional_fraction(config: dict, key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    if not _finite(value) or not 0 < float(value) < 1:
        raise ValueError(f"{key} must be None or a number in (0, 1)")
    return float(value)


def _nonnegative_number(value: Any, name: str) -> float:
    if not _finite(value) or float(value) < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return float(value)


def _integer_config(config: dict, key: str) -> int:
    value = config[key]
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a positive integer") from exc
    if result <= 0 or isinstance(value, bool) or float(value) != result:
        raise ValueError(f"{key} must be a positive integer")
    return result


def _parse_rate_schedule(value: Any, default: Any, name: str) -> list[tuple[pd.Timestamp, float]]:
    if value is None:
        value = default
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return [(pd.Timestamp.min, _nonnegative_number(value, name))]
    schedule: list[tuple[pd.Timestamp, float]] = []
    for item in value:
        if not isinstance(item, dict) or "start" not in item or "rate" not in item:
            raise ValueError(f"{name} schedule entries need start and rate")
        schedule.append(
            (_timestamp(item["start"]), _nonnegative_number(item["rate"], name))
        )
    schedule.sort(key=lambda item: item[0])
    return schedule


def _rate_on(schedule: list[tuple[pd.Timestamp, float]], date: pd.Timestamp) -> float:
    rate = 0.0
    for start, scheduled_rate in schedule:
        if start > date:
            break
        rate = scheduled_rate
    return rate


def _frame(rows: list[dict], columns: list[str], date_columns: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=columns)
    for column in date_columns:
        if column in result and not result.empty:
            result[column] = pd.to_datetime(result[column])
    return result


def _dividend_tax_rate(buy_date: pd.Timestamp, sell_date: pd.Timestamp) -> float:
    if sell_date <= buy_date + pd.DateOffset(months=1):
        return 0.20
    if sell_date <= buy_date + pd.DateOffset(years=1):
        return 0.10
    return 0.0


def run_backtest(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    exit_signals: set[tuple[pd.Timestamp, pd.Timestamp | str]] | None,
    config: dict,
) -> BacktestResult:
    """Run a long-only, next-session-open A-share account simulation."""

    start_date = _timestamp(config["start_date"])
    end_date = _timestamp(config["end_date"])
    if end_date < start_date:
        raise ValueError("end_date must not precede start_date")
    initial_capital = _positive_number(config, "initial_capital")
    max_positions = _integer_config(config, "max_positions")
    holding_sessions_limit = _integer_config(config, "holding_sessions")
    limit_rate = _positive_number(config, "limit_rate", 0.10)
    if limit_rate >= 1:
        raise ValueError("limit_rate must be less than 1")
    participation = _positive_number(config, "participation", 0.001)
    slot_fraction = _positive_number(config, "slot_fraction", 1.0)
    if slot_fraction > 1:
        raise ValueError("slot_fraction must be in (0, 1]")
    stop_loss = _optional_fraction(config, "stop_loss")
    profit_target = _optional_fraction(config, "profit_target")
    trailing_stop = _optional_fraction(config, "trailing_stop")
    costs = config.get("costs", {})
    slippage = _nonnegative_number(
        config.get("slippage", costs.get("slippage_each_side_baseline", 0.001)),
        "slippage",
    )
    if slippage >= 1:
        raise ValueError("slippage must be less than 1")
    commission_rate = _nonnegative_number(
        costs.get("commission_rate", 0.0003), "commission_rate"
    )
    minimum_commission = _nonnegative_number(
        costs.get("minimum_commission", 5.0), "minimum_commission"
    )
    stamp_schedule = _parse_rate_schedule(
        costs.get("stamp_duty_sell"),
        [
            {"start": "1900-01-01", "rate": 0.001},
            {"start": "2023-08-28", "rate": 0.0005},
        ],
        "stamp_duty_sell",
    )
    transfer_schedule = _parse_rate_schedule(
        costs.get("transfer_fee_both_sides"),
        [
            {"start": "1900-01-01", "rate": 0.00002},
            {"start": "2022-04-29", "rate": 0.00001},
        ],
        "transfer_fee_both_sides",
    )

    if not isinstance(daily, pd.DataFrame):
        raise TypeError("daily must be a pandas DataFrame")
    missing_daily = set(_DAILY_REQUIRED_COLUMNS) - set(daily.columns)
    if missing_daily:
        raise ValueError(f"daily is missing columns: {sorted(missing_daily)}")
    selected_daily = _DAILY_REQUIRED_COLUMNS + [
        column for column in _DAILY_OPTIONAL_COLUMNS if column in daily.columns
    ]
    daily_frame = daily.loc[:, selected_daily].copy()
    for column in _DAILY_OPTIONAL_COLUMNS:
        if column not in daily_frame:
            daily_frame[column] = False if column == "factor_only_adjustment" else np.nan
    daily_frame = daily_frame.loc[:, _DAILY_INPUT_COLUMNS]
    if daily_frame["thscode"].isna().any():
        raise ValueError("daily.thscode must not be missing")
    daily_frame["date"] = _dates(daily_frame["date"])
    daily_frame["thscode"] = daily_frame["thscode"].astype(str)
    daily_frame = daily_frame[
        daily_frame["date"].between(start_date, end_date)
    ].copy()
    numeric_daily = [
        column
        for column in _DAILY_INPUT_COLUMNS
        if column not in {"date", "thscode", "factor_only_adjustment"}
    ]
    for column in numeric_daily:
        daily_frame[column] = pd.to_numeric(daily_frame[column], errors="coerce")
    daily_frame["factor_only_adjustment"] = daily_frame[
        "factor_only_adjustment"
    ].map(lambda value: _boolean_flag(value, "daily.factor_only_adjustment"))
    if daily_frame.empty:
        raise ValueError("no daily bars inside [start_date, end_date]")
    if daily_frame.duplicated(["date", "thscode"]).any():
        raise ValueError("daily has duplicate (date, thscode) bars")
    daily_frame.sort_values(["date", "thscode"], inplace=True)
    sessions = list(pd.Index(daily_frame["date"].unique()).sort_values())
    session_set = set(sessions)
    bars = {
        (row.date, row.thscode): row
        for row in daily_frame.itertuples(index=False)
    }

    candidate_lookup: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")
    if not candidates.empty:
        required_candidates = {"date", "thscode", "score", "amount20"}
        missing_candidates = required_candidates - set(candidates.columns)
        if missing_candidates:
            raise ValueError(
                f"candidates is missing columns: {sorted(missing_candidates)}"
            )
        candidate_frame = candidates.loc[:, list(required_candidates)].copy()
        if candidate_frame["thscode"].isna().any():
            raise ValueError("candidates.thscode must not be missing")
        candidate_frame["date"] = _dates(candidate_frame["date"])
        candidate_frame["thscode"] = candidate_frame["thscode"].astype(str)
        candidate_frame = candidate_frame[
            candidate_frame["date"].isin(session_set)
        ].copy()
        if not candidate_frame.empty:
            candidate_frame["score"] = pd.to_numeric(
                candidate_frame["score"], errors="coerce"
            )
            candidate_frame["amount20"] = pd.to_numeric(
                candidate_frame["amount20"], errors="coerce"
            )
            if not candidate_frame["score"].map(_finite).all():
                raise ValueError("candidate scores must be finite")
            if not candidate_frame["amount20"].map(_finite).all() or (
                candidate_frame["amount20"] < 0
            ).any():
                raise ValueError("candidate amount20 must be finite and non-negative")
            if candidate_frame.duplicated(["date", "thscode"]).any():
                raise ValueError("candidates has duplicate (date, thscode) rows")
            for signal_date, group in candidate_frame.groupby("date", sort=True):
                records = [
                    {
                        "date": row.date,
                        "thscode": row.thscode,
                        "score": float(row.score),
                        "amount20": float(row.amount20),
                    }
                    for row in group.itertuples(index=False)
                ]
                records.sort(key=lambda row: (-row["score"], row["thscode"]))
                candidate_lookup[signal_date] = records

    event_lookup: dict[tuple[pd.Timestamp, str], dict[str, float]] = {}
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame")
    if not events.empty:
        missing_events = {"ex_date", "thscode"} - set(events.columns)
        if missing_events:
            raise ValueError(f"events is missing columns: {sorted(missing_events)}")
        event_columns = [
            "ex_date",
            "thscode",
            "dividend_per_share",
            "per_share_bonus",
            "rights_ratio",
        ]
        event_frame = events.copy()
        for column in event_columns[2:]:
            if column not in event_frame:
                event_frame[column] = 0.0
        event_frame = event_frame.loc[:, event_columns]
        if event_frame["thscode"].isna().any():
            raise ValueError("events.thscode must not be missing")
        event_frame["ex_date"] = _dates(event_frame["ex_date"])
        event_frame["thscode"] = event_frame["thscode"].astype(str)
        event_frame = event_frame[
            event_frame["ex_date"].map(lambda date: date in session_set)
        ].copy()
        if not event_frame.empty:
            for column in event_columns[2:]:
                event_frame[column] = pd.to_numeric(
                    event_frame[column], errors="coerce"
                ).fillna(0.0)
                if not event_frame[column].map(_finite).all() or (
                    event_frame[column] < 0
                ).any():
                    raise ValueError(f"events.{column} must be finite and non-negative")
            grouped_events = (
                event_frame.groupby(["ex_date", "thscode"], as_index=False)[
                    event_columns[2:]
                ]
                .sum()
                .itertuples(index=False)
            )
            for row in grouped_events:
                event_lookup[(row.ex_date, row.thscode)] = {
                    "dividend_per_share": float(row.dividend_per_share),
                    "per_share_bonus": float(row.per_share_bonus),
                    "rights_ratio": float(row.rights_ratio),
                }

    normalized_exit_signals: set[tuple[pd.Timestamp, str]] = set()
    for signal_date, thscode in exit_signals or set():
        normalized_date = _timestamp(signal_date)
        if normalized_date in session_set:
            normalized_exit_signals.add((normalized_date, str(thscode)))

    cash = initial_capital
    positions: dict[str, _Position] = {}
    order_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    order_sequence = 0
    trade_sequence = 0
    buy_budget = initial_capital

    def record_order(
        date: pd.Timestamp,
        signal_date: pd.Timestamp | pd.NaT,
        thscode: str,
        side: str,
        reason: str,
        status: str,
        status_reason: str = "",
        attempt: int = 1,
        **values: Any,
    ) -> str:
        nonlocal order_sequence
        order_sequence += 1
        order_id = f"O{order_sequence:08d}"
        row: dict[str, Any] = {column: 0.0 for column in _ORDER_COLUMNS}
        row.update(
            {
                "order_id": order_id,
                "date": date,
                "signal_date": signal_date,
                "thscode": thscode,
                "side": side,
                "reason": reason,
                "status": status,
                "status_reason": status_reason,
                "attempt": attempt,
                "shares": 0,
                "raw_open": np.nan,
                "reference_close": np.nan,
                "price": np.nan,
                "limit_up": np.nan,
                "limit_down": np.nan,
                "score": np.nan,
                "amount20": np.nan,
                "participation_cap": np.nan,
                "slot_budget": np.nan,
                "order_budget": np.nan,
            }
        )
        unknown_values = set(values) - set(_ORDER_COLUMNS)
        if unknown_values:
            raise RuntimeError(f"unknown order fields: {sorted(unknown_values)}")
        row.update(values)
        order_rows.append(row)
        return order_id

    def execute_sell(
        position: _Position,
        date: pd.Timestamp,
        reason: str,
        flows: dict[str, float],
        sold_today: set[str],
        bought_today: set[str],
    ) -> None:
        nonlocal cash, trade_sequence
        thscode = position.thscode
        pending = position.pending_exit
        _require(pending is not None, "a sell requires a pending exit")
        signal_date = pending["signal_date"]
        _require(date > position.buy_date, "T+1 forbids selling on the buy date")
        _require(
            pd.isna(signal_date) or signal_date < date,
            "a sell cannot use a same-day or future signal",
        )
        _require(thscode not in bought_today, "same stock cannot be bought and sold in one day")
        attempt = position.exit_attempts + 1
        common = {
            "shares": position.shares,
            "score": np.nan,
        }

        def defer(detail: str, **values: Any) -> None:
            position.exit_attempts += 1
            record_order(
                date=date,
                signal_date=signal_date,
                thscode=thscode,
                side="sell",
                reason=reason,
                status="deferred",
                status_reason=detail,
                attempt=attempt,
                **common,
                **values,
            )

        bar = bars.get((date, thscode))
        if bar is None:
            defer("no_bar")
            return
        if not all(_finite(value) for value in (bar.open, bar.volume)):
            defer("invalid_bar")
            return
        if float(bar.volume) <= 0:
            defer("zero_volume")
            return
        raw_open = float(bar.open)
        reference_close = _execution_reference_close(bar)
        if raw_open <= 0 or reference_close is None:
            defer("invalid_bar")
            return
        limit_up = _limit_price(reference_close, limit_rate, True)
        limit_down = _limit_price(reference_close, limit_rate, False)
        common.update(
            {
                "raw_open": raw_open,
                "reference_close": reference_close,
                "limit_up": limit_up,
                "limit_down": limit_down,
            }
        )
        if raw_open <= limit_down:
            defer("open_limit_down")
            return
        if position.available_shares != position.shares:
            defer("shares_not_sellable")
            return
        quantity = position.shares
        _require(quantity > 0, "a held position must have positive shares")
        _require(
            0 < quantity <= position.available_shares,
            "sell quantity exceeds available shares",
        )
        _require(
            quantity == position.shares == position.available_shares,
            "a position must exit in full rather than partially",
        )
        price = _execution_price(
            raw_open, slippage, "sell", limit_down, limit_up
        )
        gross = quantity * price
        commission = max(gross * commission_rate, minimum_commission)
        stamp_duty = gross * _rate_on(stamp_schedule, date)
        transfer_fee = gross * _rate_on(transfer_schedule, date)
        dividend_gross = position.dividend_receivable
        dividend_tax = dividend_gross * _dividend_tax_rate(position.buy_date, date)
        dividend_net = dividend_gross - dividend_tax
        sale_net = gross - commission - stamp_duty - transfer_fee
        cash_delta = sale_net + dividend_net
        slippage_cost = (raw_open - price) * quantity
        cash += cash_delta
        if -1e-8 < cash < 0:
            cash = 0.0
        _require(cash >= 0, "cash became negative after a sell")
        flows["sell_gross"] += gross
        flows["sell_commission"] += commission
        flows["sell_stamp_duty"] += stamp_duty
        flows["sell_transfer_fee"] += transfer_fee
        flows["dividend_receivable_settled"] += dividend_gross
        flows["dividend_tax"] += dividend_tax
        flows["dividend_cash_received"] += dividend_net
        order_id = record_order(
            date=date,
            signal_date=signal_date,
            thscode=thscode,
            side="sell",
            reason=reason,
            status="filled",
            attempt=attempt,
            price=price,
            gross=gross,
            commission=commission,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            slippage_cost=slippage_cost,
            dividend_receivable=dividend_gross,
            dividend_tax=dividend_tax,
            dividend_received=dividend_net,
            cash_delta=cash_delta,
            **common,
        )
        trade_sequence += 1
        net_pnl = cash_delta - position.buy_total_cost
        trade_rows.append(
            {
                "trade_id": f"T{trade_sequence:08d}",
                "thscode": thscode,
                "buy_date": position.buy_date,
                "sell_date": date,
                "buy_signal_date": position.buy_signal_date,
                "sell_signal_date": signal_date,
                "buy_order_id": position.buy_order_id,
                "sell_order_id": order_id,
                "shares": quantity,
                "buy_shares": position.buy_shares,
                "bonus_shares": position.bonus_shares,
                "sell_shares": quantity,
                "buy_price": position.buy_price,
                "sell_price": price,
                "buy_gross": position.buy_gross,
                "sell_gross": gross,
                "buy_commission": position.buy_commission,
                "sell_commission": commission,
                "buy_stamp_duty": 0.0,
                "sell_stamp_duty": stamp_duty,
                "buy_transfer_fee": position.buy_transfer_fee,
                "sell_transfer_fee": transfer_fee,
                "buy_slippage_cost": position.buy_slippage_cost,
                "sell_slippage_cost": slippage_cost,
                "dividend_receivable": dividend_gross,
                "dividend_received_gross": dividend_gross,
                "dividend_tax": dividend_tax,
                "dividend_received": dividend_net,
                "buy_total_cost": position.buy_total_cost,
                "sell_net_proceeds": sale_net,
                "net_pnl": net_pnl,
                "return": net_pnl / position.buy_total_cost,
                "holding_days": (date - position.buy_date).days,
                "holding_sessions": position.holding_sessions,
                "exit_reason": reason,
            }
        )
        position.dividend_receivable = 0.0
        position.shares = 0
        position.available_shares = 0
        _require(position.shares >= 0 and position.available_shares >= 0, "negative shares")
        sold_today.add(thscode)
        del positions[thscode]

    def execute_buy(
        candidate: dict[str, Any],
        date: pd.Timestamp,
        final_session: bool,
        flows: dict[str, float],
        sold_today: set[str],
        bought_today: set[str],
    ) -> None:
        nonlocal cash, buy_budget
        thscode = candidate["thscode"]
        signal_date = candidate["date"]
        score = candidate["score"]
        amount20 = candidate["amount20"]
        participation_cap = amount20 * participation
        slot_budget = previous_equity * slot_fraction
        order_budget = min(buy_budget, slot_budget)
        _require(signal_date < date, "a buy cannot use a same-day or future signal")
        common = {
            "score": score,
            "amount20": amount20,
            "participation_cap": participation_cap,
            "slot_budget": slot_budget,
            "order_budget": order_budget,
        }

        def reject(detail: str, **values: Any) -> None:
            record_order(
                date=date,
                signal_date=signal_date,
                thscode=thscode,
                side="buy",
                reason="candidate",
                status="rejected",
                status_reason=detail,
                **common,
                **values,
            )

        if final_session:
            reject("end_of_test")
            return
        if thscode in sold_today or thscode in bought_today:
            reject("same_day_sell" if thscode in sold_today else "same_day_buy")
            return
        if thscode in positions:
            reject("already_held")
            return
        if len(positions) >= max_positions:
            reject("position_limit")
            return
        bar = bars.get((date, thscode))
        if bar is None:
            reject("no_bar")
            return
        if not all(_finite(value) for value in (bar.open, bar.volume)):
            reject("invalid_bar")
            return
        if float(bar.volume) <= 0:
            reject("zero_volume")
            return
        raw_open = float(bar.open)
        reference_close = _execution_reference_close(bar)
        if raw_open <= 0 or reference_close is None:
            reject("invalid_bar")
            return
        limit_up = _limit_price(reference_close, limit_rate, True)
        limit_down = _limit_price(reference_close, limit_rate, False)
        common.update(
            {
                "raw_open": raw_open,
                "reference_close": reference_close,
                "limit_up": limit_up,
                "limit_down": limit_down,
            }
        )
        if raw_open >= limit_up:
            reject("open_limit_up")
            return
        price = _execution_price(raw_open, slippage, "buy", limit_down, limit_up)
        lot_gross = price * 100
        participation_lots = int(np.floor((participation_cap + 1e-12) / lot_gross))
        if participation_lots < 1:
            reject("participation_cap", price=price)
            return
        budget_lots = int(np.floor((order_budget + 1e-12) / lot_gross))
        lots = min(participation_lots, budget_lots)
        quantity = 0
        gross = commission = transfer_fee = total_cost = 0.0
        while lots > 0:
            candidate_quantity = lots * 100
            candidate_gross = candidate_quantity * price
            candidate_commission = max(
                candidate_gross * commission_rate, minimum_commission
            )
            candidate_transfer = candidate_gross * _rate_on(transfer_schedule, date)
            candidate_total = candidate_gross + candidate_commission + candidate_transfer
            if (
                candidate_total <= order_budget + 1e-9
                and candidate_total <= buy_budget + 1e-9
                and candidate_total <= cash + 1e-9
            ):
                quantity = candidate_quantity
                gross = candidate_gross
                commission = candidate_commission
                transfer_fee = candidate_transfer
                total_cost = candidate_total
                break
            lots -= 1
        if quantity == 0:
            detail = "slot_budget" if slot_budget < buy_budget else "insufficient_cash"
            reject(detail, price=price)
            return
        _require(quantity > 0 and quantity % 100 == 0, "buys must be whole 100-share lots")
        _require(gross <= participation_cap + 1e-8, "buy exceeds signal-date participation cap")
        _require(total_cost <= slot_budget + 1e-8, "buy exceeds its equity slot budget")
        _require(total_cost <= order_budget + 1e-8, "buy exceeds its order budget")
        _require(total_cost <= buy_budget + 1e-8, "buy reuses same-open sale proceeds")
        _require(total_cost <= cash + 1e-8, "buy exceeds cash")
        _require(thscode not in sold_today, "same stock cannot be bought and sold in one day")
        slippage_cost = (price - raw_open) * quantity
        cash -= total_cost
        buy_budget -= total_cost
        if -1e-8 < cash < 0:
            cash = 0.0
        if -1e-8 < buy_budget < 0:
            buy_budget = 0.0
        _require(cash >= 0 and buy_budget >= 0, "cash became negative after a buy")
        flows["buy_gross"] += gross
        flows["buy_commission"] += commission
        flows["buy_transfer_fee"] += transfer_fee
        order_id = record_order(
            date=date,
            signal_date=signal_date,
            thscode=thscode,
            side="buy",
            reason="candidate",
            status="filled",
            price=price,
            shares=quantity,
            gross=gross,
            commission=commission,
            stamp_duty=0.0,
            transfer_fee=transfer_fee,
            slippage_cost=slippage_cost,
            cash_delta=-total_cost,
            **common,
        )
        positions[thscode] = _Position(
            thscode=thscode,
            buy_date=date,
            buy_signal_date=signal_date,
            buy_order_id=order_id,
            buy_shares=quantity,
            shares=quantity,
            available_shares=0,
            bonus_shares=0,
            buy_price=price,
            buy_gross=gross,
            buy_commission=commission,
            buy_transfer_fee=transfer_fee,
            buy_slippage_cost=slippage_cost,
            buy_total_cost=total_cost,
            last_close=raw_open,
        )
        bought_today.add(thscode)
        _require(
            positions[thscode].shares >= 0
            and positions[thscode].available_shares >= 0,
            "negative shares after a buy",
        )

    previous_equity = initial_capital
    running_peak = initial_capital
    final_index = len(sessions) - 1
    for session_index, date in enumerate(sessions):
        final_session = session_index == final_index
        flows = {key: 0.0 for key in _FLOW_KEYS}
        cash_start = cash
        buy_budget = cash_start
        receivable_start = sum(
            position.dividend_receivable for position in positions.values()
        )
        sold_today: set[str] = set()
        bought_today: set[str] = set()
        held_at_open = bool(positions)

        for thscode in sorted(positions):
            position = positions[thscode]
            _require(position.buy_date < date, "a new buy reached a later open incorrectly")
            position.available_shares = position.shares
            _require(
                position.shares >= 0 and position.available_shares >= 0,
                "negative shares at session start",
            )

        for thscode in sorted(list(positions)):
            position = positions[thscode]
            corporate_action = event_lookup.get((date, thscode))
            if corporate_action is None:
                bar = bars.get((date, thscode))
                if bar is None or not bar.factor_only_adjustment:
                    continue
                previous_actual = bar.previous_actual_close
                reference = bar.reference_close
                if not all(
                    _finite(value) and float(value) > 0
                    for value in (previous_actual, reference)
                ):
                    raise ValueError(
                        "factor_only_adjustment requires positive finite "
                        f"previous_actual_close and reference_close for {thscode} on {date}"
                    )
                per_share_decimal = _decimal(previous_actual) - _decimal(reference)
                if per_share_decimal < 0:
                    raise ValueError(
                        "factor_only_adjustment produced negative cash equivalent "
                        f"for {thscode} on {date}: previous_actual_close="
                        f"{previous_actual}, reference_close={reference}"
                    )
                per_share_adjustment = float(per_share_decimal)
                shares_before = position.shares
                available_before = position.available_shares
                dividend_added = shares_before * per_share_adjustment
                position.dividend_receivable += dividend_added
                flows["dividend_receivable_added"] += dividend_added
                _require(
                    position.shares == shares_before
                    and position.available_shares == available_before
                    and position.dividend_receivable >= 0,
                    "factor-only adjustment must preserve shares and non-negative receivables",
                )
                event_rows.append(
                    {
                        "date": date,
                        "ex_date": date,
                        "thscode": thscode,
                        "position_id": position.buy_order_id,
                        "event_type": "factor_only_cash_equivalent",
                        "shares_before": shares_before,
                        "available_shares_before": available_before,
                        "reference_close": float(reference),
                        "previous_actual_close": float(previous_actual),
                        "cash_equivalent_per_share": per_share_adjustment,
                        "dividend_per_share": 0.0,
                        "per_share_bonus": 0.0,
                        "rights_ratio": 0.0,
                        "dividend_receivable_added": dividend_added,
                        "bonus_shares_added": 0,
                        "rights_shares_added": 0,
                        "rights_cash_paid": 0.0,
                        "shares_after": position.shares,
                        "available_shares_after": position.available_shares,
                        "dividend_receivable_after": position.dividend_receivable,
                        "rights_action": "not_applicable",
                        "factor_only_adjustment": True,
                        "adjustment_assumption": _FACTOR_ONLY_ASSUMPTION,
                    }
                )
                continue

            shares_before = position.shares
            available_before = position.available_shares
            dividend_added = (
                shares_before * corporate_action["dividend_per_share"]
            )
            bonus_added = int(
                (
                    _decimal(shares_before)
                    * _decimal(corporate_action["per_share_bonus"])
                ).to_integral_value(rounding=ROUND_FLOOR)
            )
            position.shares += bonus_added
            position.bonus_shares += bonus_added
            position.dividend_receivable += dividend_added
            flows["dividend_receivable_added"] += dividend_added
            _require(
                position.shares >= 0
                and position.available_shares >= 0
                and position.dividend_receivable >= 0,
                "negative position state after a corporate action",
            )
            _require(
                position.available_shares == available_before,
                "new ex-date shares must not become sellable on the ex-date",
            )
            event_rows.append(
                {
                    "date": date,
                    "ex_date": date,
                    "thscode": thscode,
                    "position_id": position.buy_order_id,
                    "event_type": "explicit_corporate_action",
                    "shares_before": shares_before,
                    "available_shares_before": available_before,
                    "reference_close": np.nan,
                    "previous_actual_close": np.nan,
                    "cash_equivalent_per_share": 0.0,
                    "dividend_per_share": corporate_action[
                        "dividend_per_share"
                    ],
                    "per_share_bonus": corporate_action["per_share_bonus"],
                    "rights_ratio": corporate_action["rights_ratio"],
                    "dividend_receivable_added": dividend_added,
                    "bonus_shares_added": bonus_added,
                    "rights_shares_added": 0,
                    "rights_cash_paid": 0.0,
                    "shares_after": position.shares,
                    "available_shares_after": position.available_shares,
                    "dividend_receivable_after": position.dividend_receivable,
                    "rights_action": "not_exercised",
                    "factor_only_adjustment": False,
                    "adjustment_assumption": "explicit_event_records_used",
                }
            )

        if final_session:
            for position in positions.values():
                prior_signal = (
                    position.pending_exit["signal_date"]
                    if position.pending_exit is not None
                    else pd.NaT
                )
                position.pending_exit = {
                    "reason": "end_of_test",
                    "signal_date": prior_signal,
                }

        for thscode in sorted(list(positions)):
            position = positions[thscode]
            if position.pending_exit is not None:
                execute_sell(
                    position,
                    date,
                    position.pending_exit["reason"],
                    flows,
                    sold_today,
                    bought_today,
                )

        _require(
            _is_close(buy_budget, cash_start),
            "same-open sells changed the buy-side cash budget",
        )
        if session_index > 0:
            signal_date = sessions[session_index - 1]
            for candidate in candidate_lookup.get(signal_date, []):
                execute_buy(
                    candidate,
                    date,
                    final_session,
                    flows,
                    sold_today,
                    bought_today,
                )

        stale_valuations = 0
        for thscode in sorted(positions):
            position = positions[thscode]
            _require(position.shares >= 0, "negative shares at close")
            position.holding_sessions += 1
            bar = bars.get((date, thscode))
            if bar is not None and _finite(bar.close) and float(bar.close) > 0:
                position.last_close = float(bar.close)
            else:
                stale_valuations += 1
            position_value = (
                position.shares * position.last_close
                + position.dividend_receivable
            )
            if position.peak_position_value is None:
                position.peak_position_value = position_value
            else:
                position.peak_position_value = max(
                    position.peak_position_value, position_value
                )
            if not final_session and position.pending_exit is None:
                exit_reason = None
                if (date, thscode) in normalized_exit_signals:
                    exit_reason = "exit_signal"
                elif (
                    stop_loss is not None
                    and position_value
                    <= position.buy_total_cost * (1.0 - stop_loss)
                ):
                    exit_reason = "stop_loss"
                elif (
                    profit_target is not None
                    and position_value
                    >= position.buy_total_cost * (1.0 + profit_target)
                ):
                    exit_reason = "profit_target"
                elif (
                    trailing_stop is not None
                    and position_value
                    <= position.peak_position_value * (1.0 - trailing_stop)
                ):
                    exit_reason = "trailing_stop"
                elif position.holding_sessions >= holding_sessions_limit:
                    exit_reason = "max_holding"
                if exit_reason is not None:
                    position.pending_exit = {
                        "reason": exit_reason,
                        "signal_date": date,
                    }

        market_value = sum(
            position.shares * position.last_close for position in positions.values()
        )
        dividend_receivable = sum(
            position.dividend_receivable for position in positions.values()
        )
        equity = cash + market_value + dividend_receivable
        expected_cash = (
            cash_start
            - flows["buy_gross"]
            - flows["buy_commission"]
            - flows["buy_transfer_fee"]
            + flows["sell_gross"]
            - flows["sell_commission"]
            - flows["sell_stamp_duty"]
            - flows["sell_transfer_fee"]
            + flows["dividend_receivable_settled"]
            - flows["dividend_tax"]
        )
        cash_flow_error = cash - expected_cash
        equity_identity_error = equity - cash - market_value - dividend_receivable
        expected_receivable = (
            receivable_start
            + flows["dividend_receivable_added"]
            - flows["dividend_receivable_settled"]
        )
        _require(cash >= 0, "negative cash at close")
        _require(_is_close(cash, expected_cash), "daily cash conservation failed")
        _require(
            _is_close(dividend_receivable, expected_receivable),
            "daily dividend receivable conservation failed",
        )
        _require(
            _is_close(equity, cash + market_value + dividend_receivable),
            "daily equity identity failed",
        )
        expected_budget = (
            cash_start
            - flows["buy_gross"]
            - flows["buy_commission"]
            - flows["buy_transfer_fee"]
        )
        _require(
            _is_close(buy_budget, expected_budget),
            "buy budget includes proceeds unavailable at the open",
        )
        running_peak = max(running_peak, equity)
        daily_return = equity / previous_equity - 1.0
        drawdown = equity / running_peak - 1.0 if running_peak > 0 else np.nan
        exposure = market_value / equity if equity > 0 else np.nan
        same_open_sale_proceeds = (
            flows["sell_gross"]
            - flows["sell_commission"]
            - flows["sell_stamp_duty"]
            - flows["sell_transfer_fee"]
            + flows["dividend_cash_received"]
        )
        daily_rows.append(
            {
                "date": date,
                "cash_start": cash_start,
                "cash": cash,
                "market_value": market_value,
                "dividend_receivable_start": receivable_start,
                "dividend_receivable": dividend_receivable,
                **{key: flows[key] for key in _FLOW_KEYS if key not in {"dividend_receivable_start"}},
                "same_open_sale_proceeds": same_open_sale_proceeds,
                "buy_budget_start": cash_start,
                "buy_budget_end": buy_budget,
                "equity": equity,
                "daily_return": daily_return,
                "drawdown": drawdown,
                "exposure": exposure,
                "positions": len(positions),
                "shares": sum(position.shares for position in positions.values()),
                "held_during_session": held_at_open
                or bool(bought_today)
                or bool(positions),
                "stale_valuation_positions": stale_valuations,
                "cash_flow_error": cash_flow_error,
                "equity_identity_error": equity_identity_error,
            }
        )
        previous_equity = equity

    daily_equity = _frame(daily_rows, _DAILY_COLUMNS, ["date"])
    orders = _frame(order_rows, _ORDER_COLUMNS, ["date", "signal_date"])
    trades = _frame(
        trade_rows,
        _TRADE_COLUMNS,
        ["buy_date", "sell_date", "buy_signal_date", "sell_signal_date"],
    )
    event_ledger = _frame(event_rows, _EVENT_COLUMNS, ["date", "ex_date"])

    filled_orders = [row for row in order_rows if row["status"] == "filled"]
    filled_buys = [row for row in filled_orders if row["side"] == "buy"]
    filled_sells = [row for row in filled_orders if row["side"] == "sell"]
    buy_order_ids = {row["order_id"] for row in filled_buys}
    sell_order_ids = {row["order_id"] for row in filled_sells}
    expected_buy_ids = {row["buy_order_id"] for row in trade_rows} | {
        position.buy_order_id for position in positions.values()
    }
    expected_sell_ids = {row["sell_order_id"] for row in trade_rows}
    _require(len(buy_order_ids) == len(filled_buys), "duplicate filled buy order id")
    _require(len(sell_order_ids) == len(filled_sells), "duplicate filled sell order id")
    _require(buy_order_ids == expected_buy_ids, "filled buys and positions/trades disagree")
    _require(sell_order_ids == expected_sell_ids, "filled sells and trades disagree")
    _require(
        _is_close(
            cash,
            initial_capital + sum(row["cash_delta"] for row in filled_orders),
        ),
        "final cash does not reconcile to filled orders",
    )
    final_shares = sum(position.shares for position in positions.values())
    expected_final_shares = (
        sum(int(row["shares"]) for row in filled_buys)
        + sum(int(row["bonus_shares_added"]) for row in event_rows)
        - sum(int(row["shares"]) for row in filled_sells)
    )
    _require(final_shares == expected_final_shares, "share ledger does not reconcile")
    _require(
        _is_close(
            sum(position.dividend_receivable for position in positions.values()),
            sum(row["dividend_receivable_added"] for row in event_rows)
            - sum(row["dividend_receivable"] for row in trade_rows),
        ),
        "final dividend receivable does not reconcile",
    )
    order_by_id = {row["order_id"]: row for row in order_rows}
    _require(len(order_by_id) == len(order_rows), "order ids are not unique")
    events_by_position: dict[str, dict[str, float]] = {}
    for event_row in event_rows:
        position_events = events_by_position.setdefault(
            event_row["position_id"], {"dividend": 0.0, "bonus": 0.0}
        )
        position_events["dividend"] += event_row["dividend_receivable_added"]
        position_events["bonus"] += event_row["bonus_shares_added"]
    _require(
        set(events_by_position) <= expected_buy_ids,
        "an event was applied to a position that never existed",
    )
    for trade in trade_rows:
        buy_order = order_by_id[trade["buy_order_id"]]
        sell_order = order_by_id[trade["sell_order_id"]]
        _require(trade["sell_date"] > trade["buy_date"], "trade violates T+1")
        _require(
            trade["buy_signal_date"] < trade["buy_date"],
            "filled buy has a same-day or future signal",
        )
        _require(
            pd.isna(trade["sell_signal_date"])
            or trade["sell_signal_date"] < trade["sell_date"],
            "filled sell has a same-day or future signal",
        )
        _require(
            int(buy_order["shares"]) == trade["buy_shares"]
            and int(sell_order["shares"]) == trade["sell_shares"],
            "trade and order quantities disagree",
        )
        for trade_field, order, order_field in [
            ("buy_price", buy_order, "price"),
            ("buy_gross", buy_order, "gross"),
            ("buy_commission", buy_order, "commission"),
            ("buy_transfer_fee", buy_order, "transfer_fee"),
            ("sell_price", sell_order, "price"),
            ("sell_gross", sell_order, "gross"),
            ("sell_commission", sell_order, "commission"),
            ("sell_stamp_duty", sell_order, "stamp_duty"),
            ("sell_transfer_fee", sell_order, "transfer_fee"),
            ("dividend_receivable", sell_order, "dividend_receivable"),
            ("dividend_tax", sell_order, "dividend_tax"),
            ("dividend_received", sell_order, "dividend_received"),
        ]:
            _require(
                _is_close(trade[trade_field], order[order_field]),
                f"trade field {trade_field} disagrees with its order",
            )
        position_events = events_by_position.get(
            trade["buy_order_id"], {"dividend": 0.0, "bonus": 0.0}
        )
        _require(
            _is_close(position_events["dividend"], trade["dividend_receivable"])
            and _is_close(position_events["bonus"], trade["bonus_shares"]),
            "trade and corporate-action ledger disagree",
        )
        expected_net_pnl = (
            trade["sell_gross"]
            - trade["sell_commission"]
            - trade["sell_stamp_duty"]
            - trade["sell_transfer_fee"]
            + trade["dividend_received"]
            - trade["buy_total_cost"]
        )
        _require(_is_close(trade["net_pnl"], expected_net_pnl), "trade net pnl is inconsistent")
    for thscode, position in positions.items():
        position_events = events_by_position.get(
            position.buy_order_id, {"dividend": 0.0, "bonus": 0.0}
        )
        _require(
            _is_close(position_events["dividend"], position.dividend_receivable)
            and _is_close(position_events["bonus"], position.bonus_shares),
            f"open position {thscode} and corporate-action ledger disagree",
        )
    filled_by_day_stock: dict[tuple[pd.Timestamp, str], set[str]] = {}
    for order in filled_orders:
        _require(
            pd.isna(order["signal_date"]) or order["signal_date"] < order["date"],
            "filled order uses a same-day or future signal",
        )
        sides = filled_by_day_stock.setdefault((order["date"], order["thscode"]), set())
        sides.add(order["side"])
        _require(
            sides != {"buy", "sell"},
            "same stock was bought and sold on the same day",
        )

    final_equity = float(daily_rows[-1]["equity"])
    elapsed_days = (sessions[-1] - sessions[0]).days
    if elapsed_days > 0:
        cagr = (final_equity / initial_capital) ** (365.0 / elapsed_days) - 1.0
    else:
        cagr = np.nan
    total_return = final_equity / initial_capital - 1.0
    max_drawdown = max(0.0, -min(float(row["drawdown"]) for row in daily_rows))
    calmar = cagr / max_drawdown if max_drawdown > 0 and np.isfinite(cagr) else np.nan
    returns = np.array([row["daily_return"] for row in daily_rows], dtype=float)
    if len(returns) >= 2:
        daily_volatility = float(np.std(returns, ddof=1))
    else:
        daily_volatility = np.nan
    sharpe = (
        float(np.mean(returns) / daily_volatility * np.sqrt(252.0))
        if np.isfinite(daily_volatility) and daily_volatility > 0
        else np.nan
    )
    net_pnls = np.array([row["net_pnl"] for row in trade_rows], dtype=float)
    winning_pnls = net_pnls[net_pnls > 0]
    losing_pnls = net_pnls[net_pnls < 0]
    win_rate = float(len(winning_pnls) / len(net_pnls)) if len(net_pnls) else np.nan
    if len(winning_pnls) and len(losing_pnls):
        profit_loss_ratio = float(
            np.mean(winning_pnls) / abs(np.mean(losing_pnls))
        )
    else:
        profit_loss_ratio = np.nan
    positive_profit = float(np.sum(winning_pnls))
    profit_shares = {}
    for count in (1, 5, 10):
        profit_shares[count] = (
            float(np.sum(np.sort(winning_pnls)[::-1][:count]) / positive_profit)
            if positive_profit > 0
            else np.nan
        )
    residual_positions = []
    for thscode in sorted(positions):
        position = positions[thscode]
        residual_positions.append(
            {
                "thscode": thscode,
                "buy_order_id": position.buy_order_id,
                "buy_date": position.buy_date.strftime("%Y-%m-%d"),
                "buy_shares": position.buy_shares,
                "bonus_shares": position.bonus_shares,
                "shares": position.shares,
                "available_shares": position.available_shares,
                "last_raw_close": position.last_close,
                "market_value": position.shares * position.last_close,
                "dividend_receivable": position.dividend_receivable,
                "exit_reason": (
                    position.pending_exit["reason"]
                    if position.pending_exit is not None
                    else "end_of_test"
                ),
            }
        )
    metrics: dict[str, Any] = {
        "start_session": sessions[0],
        "end_session": sessions[-1],
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "final_cash": cash,
        "final_market_value": float(daily_rows[-1]["market_value"]),
        "final_dividend_receivable": float(
            daily_rows[-1]["dividend_receivable"]
        ),
        "cagr": cagr,
        "cagr_elapsed_natural_days": elapsed_days,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "sharpe": sharpe,
        "trade_count": len(trade_rows),
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "holding_time_ratio": float(
            np.mean([row["held_during_session"] for row in daily_rows])
        ),
        "average_holding_days": (
            float(np.mean([row["holding_days"] for row in trade_rows]))
            if trade_rows
            else np.nan
        ),
        "average_holding_sessions": (
            float(np.mean([row["holding_sessions"] for row in trade_rows]))
            if trade_rows
            else np.nan
        ),
        "top_1_profit_share": profit_shares[1],
        "top_5_profit_share": profit_shares[5],
        "top_10_profit_share": profit_shares[10],
        "ending_open_positions": len(positions),
        "rejected_orders": sum(row["status"] == "rejected" for row in order_rows),
        "deferred_orders": sum(row["status"] == "deferred" for row in order_rows),
        "filled_buy_orders": len(filled_buys),
        "filled_sell_orders": len(filled_sells),
        "max_abs_cash_flow_error": float(
            np.max(np.abs(daily_equity["cash_flow_error"]))
        ),
        "max_abs_equity_identity_error": float(
            np.max(np.abs(daily_equity["equity_identity_error"]))
        ),
        "residual_positions": residual_positions,
        "slot_fraction": slot_fraction,
        "stop_loss": stop_loss,
        "profit_target": profit_target,
        "trailing_stop": trailing_stop,
        "factor_only_adjustment_count": sum(
            row["event_type"] == "factor_only_cash_equivalent"
            for row in event_rows
        ),
        "factor_only_adjustment_position_count": len(
            {
                row["position_id"]
                for row in event_rows
                if row["event_type"] == "factor_only_cash_equivalent"
            }
        ),
        "factor_only_adjustment_codes": sorted(
            {
                row["thscode"]
                for row in event_rows
                if row["event_type"] == "factor_only_cash_equivalent"
            }
        ),
        "factor_only_receivable_total": float(
            sum(
                row["dividend_receivable_added"]
                for row in event_rows
                if row["event_type"] == "factor_only_cash_equivalent"
            )
        ),
        "factor_only_adjustment_assumption": _FACTOR_ONLY_ASSUMPTION,
        "assumptions": [
            "Sessions are the union of main-board bar dates inside the configured period.",
            "Candidates are ranked only by score descending then thscode ascending and expire after one session.",
            "Execution limits prefer reference_close and fall back to prev_close; raw close is only used for valuation.",
            "Each buy's total cost is capped by previous close equity times slot_fraction and by pre-open cash.",
            "Buy cash is the cash present before same-open exits; same-open sale and dividend proceeds are excluded.",
            "At non-final closes, exit priority is explicit exit_signals, stop_loss, profit_target, trailing_stop, then max_holding; existing pending exits are never overwritten.",
            "Position value uses actual shares at the latest actual raw close plus gross receivables; buy_total_cost includes buy fees but not future sell fees or dividend tax.",
            "Trailing stop uses the highest daily-close position value since entry, including the buy close; all holding exits are next-open signals and gaps can cross their thresholds.",
            "Holding sessions include the buy session; max holding schedules an exit after that many closes.",
            "A full exit is deferred if any shares are unavailable, including ex-date bonus shares.",
            "Bonus shares become sellable on the next session; fractional bonus entitlements are floored to whole shares.",
            "Rights are never exercised: no rights shares and no rights cash payment.",
            "Dividend tax uses calendar holding boundaries of one month and one year.",
            "Missing or invalid close values use the last actual raw price and are counted as stale valuations.",
            _FACTOR_ONLY_ASSUMPTION,
            "Slippage cost is informational and already embedded in executed prices, not deducted twice.",
            "Sharpe uses zero risk-free rate and sqrt(252); CAGR uses elapsed natural days and 365 days per year.",
            "Top-N profit shares use positive net trade pnl divided by total positive net trade pnl.",
            "A single configurable limit_rate is used without claiming historical ST status is known.",
        ],
        "final_reconciliation_passed": True,
    }
    return BacktestResult(daily_equity, trades, orders, event_ledger, metrics)


def _self_check() -> None:
    costs = {
        "commission_rate": 0.0003,
        "minimum_commission": 5.0,
        "stamp_duty_sell": [
            {"start": "1900-01-01", "rate": 0.001},
            {"start": "2023-08-28", "rate": 0.0005},
        ],
        "transfer_fee_both_sides": [
            {"start": "1900-01-01", "rate": 0.00002},
            {"start": "2022-04-29", "rate": 0.00001},
        ],
        "slippage_each_side_baseline": 0.001,
    }

    def make_daily(
        dates: list[str], specifications: dict[str, list[tuple[float, float, float, float]]]
    ) -> pd.DataFrame:
        rows = []
        for thscode, values in specifications.items():
            _require(len(values) == len(dates), "self-check bar specification length mismatch")
            for date, (open_price, close, prev_close, volume) in zip(dates, values):
                rows.append(
                    {
                        "date": date,
                        "thscode": thscode,
                        "open": open_price,
                        "high": max(open_price, close),
                        "low": min(open_price, close),
                        "close": close,
                        "prev_close": prev_close,
                        "volume": volume,
                        "amount": close * volume,
                    }
                )
        return pd.DataFrame(rows)

    def make_config(dates: list[str], initial_capital: float) -> dict:
        return {
            "start_date": dates[0],
            "end_date": dates[-1],
            "initial_capital": initial_capital,
            "max_positions": 2,
            "holding_sessions": 1,
            "limit_rate": 0.10,
            "participation": 0.001,
            "costs": costs,
        }

    def no_events() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "ex_date",
                "thscode",
                "dividend_per_share",
                "per_share_bonus",
                "rights_ratio",
            ]
        )

    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    daily = make_daily(
        dates,
        {
            "000001": [
                (10, 10, 10, 1000000),
                (10, 10, 10, 1000000),
                (11, 11, 10, 1000000),
                (11, 11, 11, 1000000),
            ],
            "000002": [(10, 10, 10, 1000000)] * 4,
        },
    )
    candidates = pd.DataFrame(
        [
            {"date": dates[0], "thscode": "000001", "score": 1.0, "amount20": 10000000},
            {"date": dates[1], "thscode": "000001", "score": 2.0, "amount20": 10000000},
            {"date": dates[1], "thscode": "000002", "score": 1.0, "amount20": 10000000},
        ]
    )
    config = make_config(dates, 1100.0)
    config["holding_sessions"] = 10
    result = run_backtest(
        daily,
        no_events(),
        candidates,
        {(pd.Timestamp(dates[1]), "000001")},
        config,
    )
    _require(len(result.trades) == 1, "execution scenario should close one trade")
    trade = result.trades.iloc[0]
    _require(trade["buy_date"] == pd.Timestamp(dates[1]), "candidate was not bought next session")
    _require(trade["sell_date"] == pd.Timestamp(dates[2]), "exit signal was not sold next session")
    _require(trade["sell_date"] > trade["buy_date"], "T+1 self-check failed")
    _require(trade["buy_shares"] == 100 and trade["buy_shares"] % 100 == 0, "board-lot self-check failed")
    _require(_is_close(trade["buy_price"], 10.01), "buy slippage/tick rounding failed")
    _require(_is_close(trade["buy_commission"], 5.0), "minimum buy commission failed")
    _require(_is_close(trade["sell_commission"], 5.0), "minimum sell commission failed")
    _require(trade["exit_reason"] == "exit_signal", "explicit exit reason was lost")
    bbb_reject = result.orders[
        (result.orders["thscode"] == "000002") & (result.orders["status"] == "rejected")
    ]
    _require(len(bbb_reject) == 1, "second candidate should have one rejected order")
    _require(
        bbb_reject.iloc[0]["status_reason"] == "insufficient_cash",
        "same-open sale proceeds were reused for a buy",
    )
    aaa_reject = result.orders[
        (result.orders["date"] == pd.Timestamp(dates[2]))
        & (result.orders["thscode"] == "000001")
        & (result.orders["side"] == "buy")
    ]
    _require(
        len(aaa_reject) == 1
        and aaa_reject.iloc[0]["status_reason"] == "same_day_sell",
        "same-stock same-day buy/sell guard failed",
    )
    _require(
        _is_close(result.metrics["final_equity"], 1100.0 + trade["net_pnl"]),
        "closed-trade cash reconciliation failed",
    )

    dates = [
        "2024-02-05",
        "2024-02-06",
        "2024-02-07",
        "2024-02-08",
        "2024-02-19",
        "2024-02-20",
    ]
    daily = make_daily(
        dates,
        {
            "000001": [
                (10, 10, 10, 1000000),
                (11, 11, 10, 1000000),
                (10, 10, 11, 1000000),
                (9, 9, 10, 1000000),
                (9.5, 9.5, 9, 1000000),
                (9.5, 9.5, 9.5, 1000000),
            ]
        },
    )
    candidates = pd.DataFrame(
        [
            {"date": dates[0], "thscode": "000001", "score": 1.0, "amount20": 10000000},
            {"date": dates[1], "thscode": "000001", "score": 1.0, "amount20": 10000000},
        ]
    )
    config = make_config(dates, 5000.0)
    config["slippage"] = 0.0
    result = run_backtest(daily, no_events(), candidates, None, config)
    limit_reject = result.orders[
        (result.orders["side"] == "buy") & (result.orders["status"] == "rejected")
    ]
    _require(
        len(limit_reject) == 1
        and limit_reject.iloc[0]["date"] == pd.Timestamp(dates[1])
        and limit_reject.iloc[0]["status_reason"] == "open_limit_up",
        "opening limit-up buy was not rejected",
    )
    filled_buy = result.orders[
        (result.orders["side"] == "buy") & (result.orders["status"] == "filled")
    ].iloc[0]
    _require(
        filled_buy["signal_date"] == pd.Timestamp(dates[1])
        and filled_buy["date"] == pd.Timestamp(dates[2]),
        "a rejected candidate was chased or a new signal was not used",
    )
    deferred = result.orders[result.orders["status"] == "deferred"]
    _require(
        len(deferred) == 1
        and deferred.iloc[0]["date"] == pd.Timestamp(dates[3])
        and deferred.iloc[0]["status_reason"] == "open_limit_down",
        "opening limit-down sell was not deferred and logged",
    )
    _require(
        len(result.trades) == 1
        and result.trades.iloc[0]["sell_date"] == pd.Timestamp(dates[4])
        and result.trades.iloc[0]["exit_reason"] == "max_holding",
        "deferred sell did not execute on the next session",
    )

    dates = ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"]
    daily = make_daily(
        dates, {"000001": [(10, 10, 10, 1000000)] * len(dates)}
    )
    candidates = pd.DataFrame(
        [{"date": dates[0], "thscode": "000001", "score": 1.0, "amount20": 1100000}]
    )
    events = pd.DataFrame(
        [
            {
                "ex_date": dates[2],
                "thscode": "000001",
                "dividend_per_share": 1.0,
                "per_share_bonus": 0.2,
                "rights_ratio": 0.1,
            }
        ]
    )
    config = make_config(dates, 5000.0)
    config["slippage"] = 0.0
    result = run_backtest(daily, events, candidates, None, config)
    _require(len(result.event_ledger) == 1, "corporate action was not logged exactly once")
    event = result.event_ledger.iloc[0]
    _require(
        _is_close(event["dividend_receivable_added"], 100.0)
        and event["bonus_shares_added"] == 20
        and event["rights_shares_added"] == 0
        and _is_close(event["rights_cash_paid"], 0.0),
        "dividend/bonus/rights accounting failed",
    )
    ex_date_defer = result.orders[
        (result.orders["date"] == pd.Timestamp(dates[2]))
        & (result.orders["side"] == "sell")
    ]
    _require(
        len(ex_date_defer) == 1
        and ex_date_defer.iloc[0]["status"] == "deferred"
        and ex_date_defer.iloc[0]["status_reason"] == "shares_not_sellable",
        "ex-date new shares were treated as sellable or a partial exit occurred",
    )
    ex_date_equity = result.daily_equity[
        result.daily_equity["date"] == pd.Timestamp(dates[2])
    ].iloc[0]
    _require(
        _is_close(ex_date_equity["dividend_receivable"], 100.0)
        and _is_close(ex_date_equity["cash"], ex_date_equity["cash_start"]),
        "dividend became spendable cash before a full exit",
    )
    _require(len(result.trades) == 1, "corporate-action position did not fully exit")
    trade = result.trades.iloc[0]
    _require(
        trade["buy_shares"] == 100
        and trade["bonus_shares"] == 20
        and trade["sell_shares"] == 120,
        "odd-lot full sale or bonus share reconciliation failed",
    )
    _require(
        _is_close(trade["dividend_receivable"], 100.0)
        and _is_close(trade["dividend_tax"], 20.0)
        and _is_close(trade["dividend_received"], 80.0),
        "dividend tax or net dividend settlement failed",
    )
    _require(
        trade["buy_gross"] <= 1100000 * 0.001 + 1e-8,
        "signal-date amount20 participation cap failed",
    )

    dates = ["2024-04-01", "2024-04-02", "2024-04-03"]
    daily = make_daily(
        dates,
        {
            "000001": [
                (10, 10, 10, 1000000),
                (10, 10, 10, 1000000),
                (9, 9, 10, 1000000),
            ]
        },
    )
    candidates = pd.DataFrame(
        [{"date": dates[0], "thscode": "000001", "score": 1.0, "amount20": 1100000}]
    )
    events = pd.DataFrame(
        [
            {
                "ex_date": dates[2],
                "thscode": "000001",
                "dividend_per_share": 1.0,
                "per_share_bonus": 0.0,
                "rights_ratio": 0.0,
            }
        ]
    )
    config = make_config(dates, 5000.0)
    config["slippage"] = 0.0
    config["holding_sessions"] = 10
    result = run_backtest(daily, events, candidates, None, config)
    _require(result.trades.empty, "an unfillable end-of-test exit fabricated a trade")
    _require(result.metrics["ending_open_positions"] == 1, "residual position was not disclosed")
    _require(
        _is_close(result.metrics["final_market_value"], 900.0)
        and _is_close(result.metrics["final_dividend_receivable"], 100.0),
        "residual raw-close valuation or dividend receivable disclosure failed",
    )
    final_sell = result.orders[
        (result.orders["date"] == pd.Timestamp(dates[-1]))
        & (result.orders["side"] == "sell")
    ].iloc[0]
    _require(
        final_sell["status"] == "deferred"
        and final_sell["reason"] == "end_of_test"
        and final_sell["status_reason"] == "open_limit_down",
        "final forced exit did not preserve the actual failed order",
    )
    _require(
        _is_close(result.daily_equity.iloc[-1]["cash"], result.daily_equity.iloc[-1]["cash_start"]),
        "residual dividend or sale fabricated final cash",
    )

    dates = ["2024-05-06", "2024-05-07", "2024-05-08"]
    daily = make_daily(
        dates,
        {
            "000001": [(10, 10, 10, 1000000)] * 3,
            "000002": [(10, 10, 10, 1000000)] * 3,
        },
    )
    candidates = pd.DataFrame(
        [
            {"date": dates[0], "thscode": "000001", "score": 2.0, "amount20": 100000000},
            {"date": dates[0], "thscode": "000002", "score": 1.0, "amount20": 100000000},
        ]
    )
    config = make_config(dates, 20000.0)
    config["slippage"] = 0.0
    config["holding_sessions"] = 10
    config["slot_fraction"] = 0.5
    two_slot_result = run_backtest(daily, no_events(), candidates, None, config)
    slot_buys = two_slot_result.orders[
        (two_slot_result.orders["side"] == "buy")
        & (two_slot_result.orders["status"] == "filled")
    ]
    _require(len(slot_buys) == 2, "slot budget prevented the second position from opening")
    for order in slot_buys.itertuples(index=False):
        total_cost = -order.cash_delta
        _require(order.date == pd.Timestamp(dates[1]), "two-slot buys filled on the wrong date")
        _require(order.shares == 900, "total-cost slot budget did not include fees and board lots")
        _require(total_cost <= 10000.0 + 1e-8, "buy exceeded half of previous equity")
        _require(total_cost <= order.slot_budget + 1e-8, "buy exceeded its recorded slot budget")
        _require(total_cost <= order.order_budget + 1e-8, "buy exceeded its recorded order budget")
        _require(order.gross <= order.participation_cap + 1e-8, "slot buy exceeded participation")
    _require(
        two_slot_result.daily_equity.loc[1, "positions"] == 2
        and len(two_slot_result.trades) == 2,
        "two-position account did not establish and close both positions",
    )

    dates = ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07"]
    daily = make_daily(
        dates,
        {
            "000001": [
                (10, 10, 10, 1000000),
                (10, 10, 10, 1000000),
                (9, 9, 9, 1000000),
                (9, 9, 9, 1000000),
                (9, 9, 9, 1000000),
            ]
        },
    )
    daily["reference_close"] = [10.0, 10.0, 9.0, 9.0, 9.0]
    daily["previous_actual_close"] = [10.0, 10.0, 10.0, 9.0, 9.0]
    daily["factor_only_adjustment"] = [False, False, True, False, False]
    daily["prev_close"] = np.nan
    daily.loc[1, "prev_close"] = 9.0
    daily.loc[1, "close"] = np.nan
    daily.loc[3, "close"] = np.nan
    candidates = pd.DataFrame(
        [{"date": dates[0], "thscode": "000001", "score": 1.0, "amount20": 1100000}]
    )
    config = make_config(dates, 5000.0)
    config["slippage"] = 0.0
    config["holding_sessions"] = 2
    factor_result = run_backtest(daily, no_events(), candidates, None, config)
    factor_buy = factor_result.orders[
        (factor_result.orders["side"] == "buy")
        & (factor_result.orders["status"] == "filled")
    ].iloc[0]
    _require(
        factor_buy["date"] == pd.Timestamp(dates[1])
        and _is_close(factor_buy["reference_close"], 10.0)
        and _is_close(factor_buy["limit_up"], 11.0),
        "reference_close did not take priority over prev_close",
    )
    _require(
        factor_result.daily_equity.loc[1, "stale_valuation_positions"] == 1,
        "missing buy-day close was not handled as a stale valuation",
    )
    _require(len(factor_result.event_ledger) == 1, "factor-only adjustment was not logged")
    factor_event = factor_result.event_ledger.iloc[0]
    _require(
        factor_event["event_type"] == "factor_only_cash_equivalent"
        and _is_close(factor_event["previous_actual_close"], 10.0)
        and _is_close(factor_event["reference_close"], 9.0)
        and _is_close(factor_event["cash_equivalent_per_share"], 1.0)
        and _is_close(factor_event["dividend_receivable_added"], 100.0)
        and factor_event["bonus_shares_added"] == 0
        and factor_event["shares_after"] == factor_event["shares_before"]
        and "not a reconstruction" in factor_event["adjustment_assumption"],
        "factor-only cash-equivalent ledger is inconsistent",
    )
    factor_day = factor_result.daily_equity.iloc[2]
    _require(
        _is_close(factor_day["dividend_receivable"], 100.0)
        and _is_close(factor_day["cash"], factor_day["cash_start"])
        and _is_close(factor_day["market_value"], 900.0)
        and _is_close(factor_day["equity"], factor_result.daily_equity.iloc[1]["equity"]),
        "factor-only receivable did not offset the raw-price adjustment",
    )
    _require(len(factor_result.trades) == 1, "factor-only position did not close")
    factor_trade = factor_result.trades.iloc[0]
    _require(
        factor_trade["sell_date"] == pd.Timestamp(dates[3])
        and _is_close(factor_trade["dividend_receivable"], 100.0)
        and _is_close(factor_trade["dividend_tax"], 20.0)
        and _is_close(factor_trade["dividend_received"], 80.0),
        "missing sell-day close blocked execution or factor receivable was not taxed",
    )
    _require(
        factor_result.metrics["factor_only_adjustment_count"] == 1
        and factor_result.metrics["factor_only_adjustment_position_count"] == 1
        and factor_result.metrics["factor_only_adjustment_codes"] == ["000001"]
        and _is_close(factor_result.metrics["factor_only_receivable_total"], 100.0)
        and "not a reconstruction" in factor_result.metrics["factor_only_adjustment_assumption"],
        "factor-only metrics disclosure is incomplete",
    )
    expected_receivable = (
        factor_result.daily_equity["dividend_receivable_start"]
        + factor_result.daily_equity["dividend_receivable_added"]
        - factor_result.daily_equity["dividend_receivable_settled"]
    )
    _require(
        np.allclose(
            expected_receivable,
            factor_result.daily_equity["dividend_receivable"],
            atol=1e-7,
        ),
        "factor-only receivable conservation failed",
    )
    negative_daily = daily.copy()
    negative_daily.loc[2, "reference_close"] = 11.0
    try:
        run_backtest(negative_daily, no_events(), candidates, None, config)
    except ValueError as error:
        _require(
            "negative cash equivalent" in str(error),
            "negative factor-only difference raised an unclear error",
        )
    else:
        _require(False, "negative factor-only difference did not raise an error")

    dates = [
        "2024-07-01",
        "2024-07-02",
        "2024-07-03",
        "2024-07-04",
        "2024-07-05",
        "2024-07-08",
    ]
    daily = make_daily(
        dates,
        {
            "000001": [
                (10, 10, 10, 1000000),
                (10, 10, 10, 1000000),
                (9.5, 8, 10, 1000000),
                (7.5, 7.5, 8, 1000000),
                (7.5, 7.5, 7.5, 1000000),
                (7.5, 7.5, 7.5, 1000000),
            ]
        },
    )
    candidates = pd.DataFrame(
        [{"date": dates[0], "thscode": "000001", "score": 1.0, "amount20": 1100000}]
    )
    default_config = make_config(dates, 5000.0)
    default_config["slippage"] = 0.0
    default_config["holding_sessions"] = 3
    default_exit_result = run_backtest(daily, no_events(), candidates, None, default_config)
    none_config = default_config.copy()
    none_config.update(
        {"stop_loss": None, "profit_target": None, "trailing_stop": None}
    )
    none_exit_result = run_backtest(daily, no_events(), candidates, None, none_config)
    for frame_name in ("daily_equity", "trades", "orders", "event_ledger"):
        pd.testing.assert_frame_equal(
            getattr(default_exit_result, frame_name),
            getattr(none_exit_result, frame_name),
        )
    _require(
        default_exit_result.trades.iloc[0]["exit_reason"] == "max_holding"
        and default_exit_result.trades.iloc[0]["sell_date"] == pd.Timestamp(dates[4])
        and default_exit_result.metrics["stop_loss"] is None
        and default_exit_result.metrics["profit_target"] is None
        and default_exit_result.metrics["trailing_stop"] is None,
        "default disabled holding exits changed existing behavior",
    )
    stop_config = default_config.copy()
    stop_config["holding_sessions"] = 2
    stop_config["stop_loss"] = 0.10
    stop_result = run_backtest(daily, no_events(), candidates, None, stop_config)
    _require(len(stop_result.trades) == 1, "stop-loss scenario did not close one trade")
    stop_trade = stop_result.trades.iloc[0]
    stop_order = stop_result.orders[
        (stop_result.orders["side"] == "sell")
        & (stop_result.orders["status"] == "filled")
    ].iloc[0]
    _require(
        stop_trade["exit_reason"] == "stop_loss"
        and stop_order["signal_date"] == pd.Timestamp(dates[2])
        and stop_trade["sell_date"] == pd.Timestamp(dates[3])
        and _is_close(stop_trade["sell_price"], 7.5)
        and _is_close(stop_order["raw_open"], 7.5),
        "stop loss was not close-confirmed or did not fill at the next real open",
    )
    priority_result = run_backtest(
        daily,
        no_events(),
        candidates,
        {(pd.Timestamp(dates[2]), "000001")},
        stop_config,
    )
    _require(
        priority_result.trades.iloc[0]["exit_reason"] == "exit_signal",
        "explicit exit did not retain priority over stop loss and max holding",
    )

    dates = ["2024-07-15", "2024-07-16", "2024-07-17", "2024-07-18"]
    daily = make_daily(
        dates,
        {
            "000001": [
                (10, 10, 10, 1000000),
                (10, 9, 10, 1000000),
                (8.5, 8.5, 9, 1000000),
                (8.5, 8.5, 8.5, 1000000),
            ]
        },
    )
    candidates = pd.DataFrame(
        [{"date": dates[0], "thscode": "000001", "score": 1.0, "amount20": 1100000}]
    )
    config = make_config(dates, 5000.0)
    config["slippage"] = 0.0
    config["holding_sessions"] = 10
    config["stop_loss"] = 0.05
    first_day_stop_result = run_backtest(daily, no_events(), candidates, None, config)
    first_day_trade = first_day_stop_result.trades.iloc[0]
    first_day_order = first_day_stop_result.orders[
        (first_day_stop_result.orders["side"] == "sell")
        & (first_day_stop_result.orders["status"] == "filled")
    ].iloc[0]
    _require(
        first_day_trade["exit_reason"] == "stop_loss"
        and first_day_trade["buy_date"] == pd.Timestamp(dates[1])
        and first_day_order["signal_date"] == pd.Timestamp(dates[1])
        and first_day_trade["sell_date"] == pd.Timestamp(dates[2])
        and _is_close(first_day_trade["sell_price"], 8.5),
        "buy-day close could not signal a T+1 stop-loss exit",
    )

    dates = [
        "2024-08-01",
        "2024-08-02",
        "2024-08-05",
        "2024-08-06",
        "2024-08-07",
        "2024-08-08",
    ]
    daily = make_daily(
        dates,
        {
            "000001": [
                (10, 10, 10, 1000000),
                (10, 10, 10, 1000000),
                (10.5, 11, 10, 1000000),
                (11.1, 11.2, 11, 1000000),
                (11.8, 11.8, 11.2, 1000000),
                (11.8, 11.8, 11.8, 1000000),
            ]
        },
    )
    candidates = pd.DataFrame(
        [{"date": dates[0], "thscode": "000001", "score": 1.0, "amount20": 1100000}]
    )
    config = make_config(dates, 5000.0)
    config["slippage"] = 0.0
    config["holding_sessions"] = 10
    config["profit_target"] = 0.10
    profit_result = run_backtest(daily, no_events(), candidates, None, config)
    _require(len(profit_result.trades) == 1, "profit-target scenario did not close one trade")
    profit_trade = profit_result.trades.iloc[0]
    profit_order = profit_result.orders[
        (profit_result.orders["side"] == "sell")
        & (profit_result.orders["status"] == "filled")
    ].iloc[0]
    _require(
        profit_trade["exit_reason"] == "profit_target"
        and profit_order["signal_date"] == pd.Timestamp(dates[3])
        and profit_trade["sell_date"] == pd.Timestamp(dates[4])
        and _is_close(profit_trade["sell_price"], 11.8)
        and _is_close(profit_order["raw_open"], 11.8),
        "profit target ignored buy fees, close confirmation, or next-open execution",
    )

    dates = [
        "2024-09-02",
        "2024-09-03",
        "2024-09-04",
        "2024-09-05",
        "2024-09-06",
        "2024-09-09",
        "2024-09-10",
    ]
    daily = make_daily(
        dates,
        {
            "000001": [
                (10, 10, 10, 1000000),
                (10, 10, 10, 1000000),
                (11.5, 12, 10, 1000000),
                (11.6, 11.5, 12, 1000000),
                (11.4, 11.3, 11.5, 1000000),
                (10.8, 10.8, 11.3, 1000000),
                (10.8, 10.8, 10.8, 1000000),
            ]
        },
    )
    candidates = pd.DataFrame(
        [{"date": dates[0], "thscode": "000001", "score": 1.0, "amount20": 1100000}]
    )
    config = make_config(dates, 5000.0)
    config["slippage"] = 0.0
    config["holding_sessions"] = 10
    config["trailing_stop"] = 0.05
    trailing_result = run_backtest(daily, no_events(), candidates, None, config)
    _require(len(trailing_result.trades) == 1, "trailing-stop scenario did not close one trade")
    trailing_trade = trailing_result.trades.iloc[0]
    trailing_order = trailing_result.orders[
        (trailing_result.orders["side"] == "sell")
        & (trailing_result.orders["status"] == "filled")
    ].iloc[0]
    _require(
        trailing_trade["exit_reason"] == "trailing_stop"
        and trailing_order["signal_date"] == pd.Timestamp(dates[4])
        and trailing_trade["sell_date"] == pd.Timestamp(dates[5])
        and _is_close(trailing_trade["sell_price"], 10.8)
        and _is_close(trailing_order["raw_open"], 10.8),
        "trailing stop did not use the historical close peak and next real open",
    )

    dates = [
        "2024-10-07",
        "2024-10-08",
        "2024-10-09",
        "2024-10-10",
        "2024-10-11",
        "2024-10-14",
    ]
    daily = make_daily(
        dates,
        {
            "000001": [
                (10, 10, 10, 1000000),
                (10, 10, 10, 1000000),
                (4.5, 4.5, 4.5, 1000000),
                (4.5, 4.5, 4.5, 1000000),
                (4.5, 4.5, 4.5, 1000000),
                (4.5, 4.5, 4.5, 1000000),
            ]
        },
    )
    candidates = pd.DataFrame(
        [{"date": dates[0], "thscode": "000001", "score": 1.0, "amount20": 1100000}]
    )
    events = pd.DataFrame(
        [
            {
                "ex_date": dates[2],
                "thscode": "000001",
                "dividend_per_share": 1.0,
                "per_share_bonus": 1.0,
                "rights_ratio": 0.0,
            }
        ]
    )
    config = make_config(dates, 5000.0)
    config["slippage"] = 0.0
    config["holding_sessions"] = 3
    config["stop_loss"] = 0.05
    config["profit_target"] = 0.20
    config["trailing_stop"] = 0.05
    corporate_stop_result = run_backtest(daily, events, candidates, None, config)
    _require(
        len(corporate_stop_result.trades) == 1
        and corporate_stop_result.trades.iloc[0]["exit_reason"] == "max_holding"
        and corporate_stop_result.trades.iloc[0]["sell_date"] == pd.Timestamp(dates[4])
        and corporate_stop_result.trades.iloc[0]["sell_shares"] == 200,
        "corporate-action value created a false holding-exit signal",
    )
    _require(
        not corporate_stop_result.orders["reason"].isin(
            ["stop_loss", "profit_target", "trailing_stop"]
        ).any(),
        "corporate-action shares or receivables were omitted from position value",
    )
    _require(
        _is_close(
            corporate_stop_result.daily_equity.iloc[2]["equity"],
            corporate_stop_result.daily_equity.iloc[1]["equity"],
        ),
        "normal corporate-action position value did not remain continuous",
    )

    for result in locals().values():
        if isinstance(result, BacktestResult):
            _require((result.daily_equity["cash"] >= 0).all(), "negative cash self-check failed")
            _require(
                np.allclose(result.daily_equity["cash_flow_error"], 0.0, atol=1e-7),
                "daily cash identity self-check failed",
            )
            _require(
                np.allclose(result.daily_equity["equity_identity_error"], 0.0, atol=1e-7),
                "daily equity identity self-check failed",
            )

    print("ok - T+1, 100-share lots, minimum commission, signal-date participation")
    print("ok - opening limit-up rejection, limit-down deferral, no rejected-order chasing")
    print("ok - same-open sale proceeds isolation and same-stock same-day guard")
    print("ok - dividend receivable, bonus shares, rights refusal, dividend tax, odd-lot exit")
    print("ok - end-of-test residual valuation and final order/trade/event reconciliation")
    print("ok - two-slot total-cost budgets establish both positions")
    print("ok - reference_close priority and open execution without same-day close")
    print("ok - factor-only receivable, tax, conservation, disclosure, and negative-difference error")
    print("ok - stop-loss, profit-target, and historical-peak trailing exits use next real opens")
    print("ok - explicit-exit priority, disabled-default equivalence, and corporate-action value continuity")
    print("self-check passed: all synthetic account checks succeeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A-share daily account engine")
    parser.add_argument("--self-check", action="store_true", help="run in-memory synthetic checks")
    arguments = parser.parse_args()
    if arguments.self_check:
        _self_check()
    else:
        parser.print_help()
