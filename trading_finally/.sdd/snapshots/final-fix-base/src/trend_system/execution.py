"""A-share execution primitives using raw, unadjusted prices."""

from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from numbers import Integral

from .config import ResearchConfig


_CENT = Decimal("0.01")
_LOT_SIZE = 100


@dataclass(frozen=True)
class Fill:
    price: float
    quantity: int
    amount: float
    costs: dict[str, float]
    cash_delta: float


@dataclass(frozen=True)
class Rejection:
    reason: str
    disposition: str


def _decimal(value) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _money(value) -> Decimal:
    return _decimal(value).quantize(_CENT, rounding=ROUND_HALF_UP)


def _as_date(value) -> date_type:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date_type):
        return value
    if isinstance(value, str):
        return date_type.fromisoformat(value[:10])
    return value.date()


def price_limits(prev_close: float, is_st: bool) -> tuple[float, float]:
    close = _decimal(prev_close)
    if not close.is_finite() or close <= 0:
        raise ValueError("prev_close must be positive and finite")
    rate = Decimal("0.05") if is_st else Decimal("0.10")
    upper = _money(close * (Decimal("1") + rate))
    lower = _money(close * (Decimal("1") - rate))
    return float(upper), float(lower)


def transaction_costs(date, amount, side, multiplier=1.0) -> dict:
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    value = _decimal(amount)
    scale = _decimal(multiplier)
    if not value.is_finite() or value <= 0:
        raise ValueError("amount must be positive and finite")
    if not scale.is_finite() or scale <= 0:
        raise ValueError("multiplier must be positive and finite")

    trade_date = _as_date(date)
    commission = max(value * Decimal("0.00025"), Decimal("5"))
    stamp_duty = Decimal("0")
    if side == "sell":
        stamp_rate = (
            Decimal("0.0005")
            if trade_date >= date_type(2023, 8, 28)
            else Decimal("0.001")
        )
        stamp_duty = value * stamp_rate
    transfer_rate = (
        Decimal("0.00001")
        if trade_date >= date_type(2022, 4, 29)
        else Decimal("0.00002")
    )
    components = {
        "commission": commission,
        "stamp_duty": stamp_duty,
        "transfer_fee": value * transfer_rate,
    }
    costs = {
        name: float(_money(component * scale)) for name, component in components.items()
    }
    costs["total"] = float(sum((_decimal(cost) for cost in costs.values()), Decimal("0")))
    return costs


def _slipped_price(open_price: Decimal, side: str, config: ResearchConfig) -> Decimal:
    slippage = _decimal(config.slippage_bps) / Decimal("10000")
    direction = Decimal("1") if side == "buy" else Decimal("-1")
    return open_price * (Decimal("1") + direction * slippage)


def _final_fill_price(record, side: str, config: ResearchConfig) -> Decimal:
    slipped = _slipped_price(_decimal(record["open"]), side, config)
    high = _decimal(record.get("high", slipped))
    low = _decimal(record.get("low", slipped))
    prev_close = record.get("prev_close")
    if prev_close is None:
        upper = lower = slipped
    else:
        upper, lower = (
            _decimal(value)
            for value in price_limits(prev_close, record.get("is_st", False))
        )
    return min(max(slipped, max(low, lower)), min(high, upper))


def _rejection(reason: str, side: str) -> Rejection:
    return Rejection(reason, "keep" if side == "sell" else "cancel")


def open_fill(order, bar, config) -> Fill | Rejection:
    side = order.get("side")
    quantity = order.get("quantity")
    if side not in ("buy", "sell"):
        return Rejection("invalid_order", "cancel")
    if (
        isinstance(quantity, bool)
        or not isinstance(quantity, Integral)
        or quantity <= 0
        or (side == "buy" and quantity % _LOT_SIZE != 0)
    ):
        return _rejection("invalid_order", side)
    quantity = int(quantity)

    if not bar.get("data_valid", True):
        return _rejection("invalid_bar", side)
    if bar.get("trade_status", 1) != 1:
        return _rejection("suspended", side)
    if bar.get("execution_conflict", False):
        return _rejection("execution_conflict", side)

    try:
        open_price = _decimal(bar["open"])
        high = _decimal(bar["high"])
        low = _decimal(bar["low"])
        upper, lower = (_decimal(value) for value in price_limits(bar["prev_close"], bar["is_st"]))
        trade_date = bar["date"]
        _as_date(trade_date)
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return _rejection("invalid_bar", side)

    prices = (open_price, high, low, upper, lower)
    if any(not price.is_finite() or price <= 0 for price in prices) or high < low:
        return _rejection("invalid_bar", side)
    if not (low <= open_price <= high and lower <= open_price <= upper):
        return _rejection("price_out_of_bounds", side)
    if side == "buy" and open_price == upper:
        return _rejection("buy_limit_up", side)
    if side == "sell" and open_price == lower:
        return _rejection("sell_limit_down", side)

    fill_price = _final_fill_price(bar, side, config)
    amount = _money(fill_price * quantity)
    costs = transaction_costs(trade_date, float(amount), side)
    total_cost = _decimal(costs["total"])
    cash_delta = -(amount + total_cost) if side == "buy" else amount - total_cost
    return Fill(
        price=float(fill_price),
        quantity=quantity,
        amount=float(amount),
        costs=costs,
        cash_delta=float(_money(cash_delta)),
    )


def size_buy(order, equity, cash, invested_value, config) -> int:
    open_price = _decimal(order["open"])
    equity_value = _decimal(equity)
    cash_value = _decimal(cash)
    invested = _decimal(invested_value)
    values = (open_price, equity_value, cash_value, invested)
    if any(not value.is_finite() for value in values):
        return 0
    if open_price <= 0 or equity_value <= 0 or cash_value <= 0:
        return 0

    single_stock_cap = equity_value * _decimal(config.max_weight)
    remaining_exposure = equity_value * _decimal(config.max_exposure) - invested
    target_value = min(single_stock_cap, remaining_exposure)
    if target_value <= 0:
        return 0

    quantity = int(target_value / open_price) // _LOT_SIZE * _LOT_SIZE
    fill_price = _final_fill_price(order, "buy", config)
    while quantity > 0:
        amount = _money(fill_price * quantity)
        costs = transaction_costs(order["date"], float(amount), "buy")
        debit = amount + _decimal(costs["total"])
        if debit <= cash_value:
            return quantity
        quantity -= _LOT_SIZE
    return 0
