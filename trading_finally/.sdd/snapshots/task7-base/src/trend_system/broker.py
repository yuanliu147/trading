"""Daily account ledger with deferred exits and corporate actions."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from .execution import Fill, Rejection, open_fill, size_buy


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    orders: pd.DataFrame
    round_trips: pd.DataFrame
    daily_account: pd.DataFrame
    daily_positions: pd.DataFrame
    action_log: pd.DataFrame


@dataclass
class _Position:
    code: str
    quantity: int
    available_quantity: int
    entry_date: str
    entry_quantity: int
    entry_amount: float
    entry_costs: float
    entry_cash: float
    last_close: float
    out_date: str | None
    exit_amount: float = 0.0
    exit_costs: float = 0.0
    exit_cash: float = 0.0
    exit_quantity: int = 0


_TRADE_COLUMNS = [
    "order_id", "signal_date", "date", "code", "side", "quantity", "price",
    "amount", "commission", "stamp_duty", "transfer_fee", "costs", "cash_delta",
]
_ORDER_COLUMNS = [
    "order_id", "signal_date", "date", "code", "side", "quantity", "status", "reason",
]
_ROUND_TRIP_COLUMNS = [
    "code", "entry_date", "exit_date", "quantity", "entry_quantity", "exit_quantity",
    "entry_amount", "entry_costs", "exit_amount", "exit_costs", "total_costs", "gross_pnl",
    "net_pnl", "return_pct", "exit_reason",
]
_ACCOUNT_COLUMNS = [
    "date", "cash", "receivables", "market_value", "equity", "exposure", "drawdown",
    "stale_valuation", "position_count",
]
_POSITION_COLUMNS = [
    "date", "code", "entry_date", "quantity", "available_quantity", "close",
    "market_value", "stale_valuation", "pending_exit",
]
_ACTION_COLUMNS = ["date", "code", "action", "quantity", "amount", "detail"]


def _money(value) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _out_date(value) -> str | None:
    if value is None or pd.isna(value) or str(value) == "":
        return None
    return str(value)[:10]


def _number(value) -> float:
    return 0.0 if value is None or pd.isna(value) else float(value)


def run_backtest(panel, signals, corporate_actions, config) -> BacktestResult:
    panel = panel.copy()
    signals = signals.copy()
    corporate_actions = corporate_actions.copy()
    panel["date"] = panel["date"].astype(str)
    signals["date"] = signals["date"].astype(str)
    if not corporate_actions.empty:
        corporate_actions["date"] = corporate_actions["date"].astype(str)

    bars = {
        (row["date"], row["code"]): row
        for row in panel.to_dict("records")
    }
    signal_map = {
        (row["date"], row["code"]): row
        for row in signals.to_dict("records")
    }
    action_map = {}
    if not corporate_actions.empty:
        for row in corporate_actions.to_dict("records"):
            action_map.setdefault((row["date"], row["code"]), []).append(row)

    dates = sorted(panel["date"].unique().tolist())
    cash = _money(config.initial_cash)
    receivables = 0.0
    peak = cash
    positions = {}
    pending_sells = {}
    buy_batch = None
    next_order_id = 1
    trade_rows = []
    order_rows = []
    round_trip_rows = []
    account_rows = []
    position_rows = []
    action_rows = []

    def log_action(day, code, action, quantity=0, amount=0.0, detail=""):
        action_rows.append(
            {
                "date": day,
                "code": code,
                "action": action,
                "quantity": int(quantity),
                "amount": float(amount),
                "detail": detail,
            }
        )

    def log_order(order, day, quantity, status, reason=""):
        order_rows.append(
            {
                "order_id": order["order_id"],
                "signal_date": order["signal_date"],
                "date": day,
                "code": order["code"],
                "side": order["side"],
                "quantity": int(quantity),
                "status": status,
                "reason": reason,
            }
        )

    def log_trade(order, day, fill):
        trade_rows.append(
            {
                "order_id": order["order_id"],
                "signal_date": order["signal_date"],
                "date": day,
                "code": order["code"],
                "side": order["side"],
                "quantity": fill.quantity,
                "price": fill.price,
                "amount": fill.amount,
                "commission": fill.costs["commission"],
                "stamp_duty": fill.costs["stamp_duty"],
                "transfer_fee": fill.costs["transfer_fee"],
                "costs": fill.costs["total"],
                "cash_delta": fill.cash_delta,
            }
        )

    def close_round_trip(position, day, reason):
        net_pnl = position.exit_cash - position.entry_cash
        round_trip_rows.append(
            {
                "code": position.code,
                "entry_date": position.entry_date,
                "exit_date": day,
                "quantity": int(position.entry_quantity),
                "entry_quantity": int(position.entry_quantity),
                "exit_quantity": int(position.exit_quantity),
                "entry_amount": position.entry_amount,
                "entry_costs": position.entry_costs,
                "exit_amount": position.exit_amount,
                "exit_costs": position.exit_costs,
                "total_costs": position.entry_costs + position.exit_costs,
                "gross_pnl": position.exit_amount - position.entry_amount,
                "net_pnl": net_pnl,
                "return_pct": net_pnl / position.entry_cash if position.entry_cash else 0.0,
                "exit_reason": reason,
            }
        )

    for date_index, day in enumerate(dates):
        todays_buys = buy_batch
        buy_batch = None
        conflict_logs = set()

        for code, position in list(positions.items()):
            bar = bars.get((day, code))
            if bar is None:
                continue
            invalid = not bool(bar.get("data_valid", True))
            conflict = bool(bar.get("execution_conflict", False))
            if invalid or conflict:
                raise RuntimeError(
                    "held position execution aborted: "
                    f"date={day} code={code} data_valid={not invalid} "
                    f"execution_conflict={conflict}"
                )

        for code, position in list(positions.items()):
            bar = bars.get((day, code))
            if bar is None:
                continue
            position.out_date = _out_date(bar.get("out_date")) or position.out_date
            if bool(bar.get("adjustment_source_conflict", False)):
                log_action(
                    day,
                    code,
                    "adjustment_source_conflict",
                    detail="held row retained; independent sensitivity required",
                )
                conflict_logs.add(code)

        for code, position in list(positions.items()):
            if position.out_date is not None and day > position.out_date:
                if code in pending_sells:
                    log_order(
                        pending_sells.pop(code),
                        day,
                        position.quantity,
                        "cancelled",
                        "delisting_zero_residual",
                    )
                log_action(
                    day,
                    code,
                    "delisting_zero_residual",
                    quantity=position.quantity,
                    detail="after out_date; no reliable liquidation consideration",
                )
                position.exit_quantity += position.quantity
                close_round_trip(position, day, "delisting_zero_residual")
                del positions[code]

        for position in positions.values():
            position.available_quantity = position.quantity

        dividends = []
        for (event_day, code), events in action_map.items():
            if event_day != day or code not in positions:
                continue
            position = positions[code]
            for event in events:
                dividend = _money(position.quantity * _number(event.get("dividend_per_share")))
                if dividend > 0:
                    receivables = _money(receivables + dividend)
                    dividends.append((code, dividend))
                    log_action(
                        day,
                        code,
                        "dividend_receivable",
                        quantity=position.quantity,
                        amount=dividend,
                        detail="ex-date approximation; settles at same-day close",
                    )
                bonus = int(position.quantity * _number(event.get("per_share_bonus")))
                if bonus > 0:
                    position.quantity += bonus
                    log_action(
                        day,
                        code,
                        "bonus_shares",
                        quantity=bonus,
                        detail="new shares become sellable on the next trading day",
                    )
                if _number(event.get("rights_ratio")) > 0:
                    log_action(
                        day,
                        code,
                        "rights_not_participated",
                        detail="conservative assumption: no automatic rights payment",
                    )

        for code, order in list(pending_sells.items()):
            position = positions.get(code)
            if position is None:
                continue
            quantity = position.available_quantity
            if quantity <= 0:
                log_order(order, day, 0, "pending", "t1_not_sellable")
                continue
            bar = bars.get((day, code))
            if bar is None:
                log_order(order, day, quantity, "pending", "missing_bar")
                continue
            fill = open_fill({"side": "sell", "quantity": quantity}, bar, config)
            if isinstance(fill, Rejection):
                log_order(order, day, quantity, "pending", fill.reason)
                continue

            updated_cash = _money(cash + fill.cash_delta)
            if updated_cash < 0:
                log_order(order, day, quantity, "pending", "insufficient_cash_for_sell")
                continue
            cash = updated_cash
            position.quantity -= fill.quantity
            position.available_quantity -= fill.quantity
            position.exit_quantity += fill.quantity
            position.exit_amount += fill.amount
            position.exit_costs += fill.costs["total"]
            position.exit_cash += fill.cash_delta
            log_order(order, day, fill.quantity, "filled")
            log_trade(order, day, fill)
            if position.quantity == 0:
                close_round_trip(position, day, order["reason"])
                del positions[code]
                del pending_sells[code]

        if todays_buys is not None:
            remaining_cash = todays_buys["cash"]
            remaining_invested = todays_buys["invested"]
            remaining_slots = todays_buys["slots"]
            for order in todays_buys["orders"]:
                if remaining_slots <= 0:
                    log_order(order, day, 0, "cancelled", "no_slot")
                    continue
                code = order["code"]
                bar = bars.get((day, code))
                if code in positions:
                    log_order(order, day, 0, "cancelled", "already_held")
                    continue
                if bar is None:
                    log_order(order, day, 0, "cancelled", "missing_bar")
                    continue
                sizing_order = {
                    key: bar[key]
                    for key in ("date", "open", "high", "low", "prev_close", "is_st")
                }
                quantity = size_buy(
                    sizing_order,
                    todays_buys["equity"],
                    min(remaining_cash, cash),
                    remaining_invested,
                    config,
                )
                if quantity <= 0:
                    log_order(order, day, 0, "cancelled", "no_size")
                    continue
                fill = open_fill({"side": "buy", "quantity": quantity}, bar, config)
                if isinstance(fill, Rejection):
                    log_order(order, day, quantity, "cancelled", fill.reason)
                    continue

                cash = _money(cash + fill.cash_delta)
                remaining_cash = _money(remaining_cash + fill.cash_delta)
                remaining_invested += fill.amount
                remaining_slots -= 1
                positions[code] = _Position(
                    code=code,
                    quantity=fill.quantity,
                    available_quantity=0,
                    entry_date=day,
                    entry_quantity=fill.quantity,
                    entry_amount=fill.amount,
                    entry_costs=fill.costs["total"],
                    entry_cash=-fill.cash_delta,
                    last_close=float(bar["close"]),
                    out_date=_out_date(bar.get("out_date")),
                )
                log_order(order, day, fill.quantity, "filled")
                log_trade(order, day, fill)

        for code, amount in dividends:
            log_action(day, code, "dividend_settled", amount=amount)
        if receivables:
            cash = _money(cash + receivables)
            receivables = 0.0

        market_values = {}
        stale_codes = set()
        for code, position in positions.items():
            bar = bars.get((day, code))
            if bar is not None and int(bar.get("trade_status", 1)) == 1:
                position.last_close = float(bar["close"])
            else:
                stale_codes.add(code)
            market_values[code] = position.quantity * position.last_close
            if (
                bar is not None
                and bool(bar.get("adjustment_source_conflict", False))
                and code not in conflict_logs
            ):
                log_action(
                    day,
                    code,
                    "adjustment_source_conflict",
                    detail="held row retained; independent sensitivity required",
                )

        market_value = sum(market_values.values(), 0.0)
        equity = cash + receivables + market_value
        peak = max(peak, equity)
        exposure = market_value / equity if equity > 0 else 0.0
        drawdown = equity / peak - 1 if peak > 0 else 0.0

        next_day = dates[date_index + 1] if date_index + 1 < len(dates) else None
        new_buys = []
        if next_day is not None:
            for code, position in positions.items():
                if code in pending_sells:
                    continue
                signal = signal_map.get((day, code), {})
                is_st_exit = bool(signal.get("st_exit", False))
                if is_st_exit or bool(signal.get("exit_signal", False)):
                    order = {
                        "order_id": next_order_id,
                        "signal_date": day,
                        "execute_date": next_day,
                        "code": code,
                        "side": "sell",
                        "reason": "st_exit" if is_st_exit else "exit_signal",
                    }
                    next_order_id += 1
                    pending_sells[code] = order

            candidates = []
            for code in sorted(panel.loc[panel["date"] == day, "code"].unique().tolist()):
                if code in positions:
                    continue
                signal = signal_map.get((day, code), {})
                if not bool(signal.get("entry_signal", False)):
                    continue
                rank = signal.get("rank_return_120")
                rank_order = -float(rank) if not pd.isna(rank) else float("inf")
                candidates.append((rank_order, code))
            candidates.sort()
            empty_slots = max(config.slots - len(positions), 0)
            for _, code in candidates:
                new_buys.append(
                    {
                        "order_id": next_order_id,
                        "signal_date": day,
                        "execute_date": next_day,
                        "code": code,
                        "side": "buy",
                    }
                )
                next_order_id += 1
            buy_batch = {
                "orders": new_buys,
                "cash": cash,
                "equity": equity,
                "invested": market_value,
                "slots": empty_slots,
            }

        account_rows.append(
            {
                "date": day,
                "cash": cash,
                "receivables": receivables,
                "market_value": market_value,
                "equity": equity,
                "exposure": exposure,
                "drawdown": drawdown,
                "stale_valuation": bool(stale_codes),
                "position_count": len(positions),
            }
        )
        for code, position in positions.items():
            position_rows.append(
                {
                    "date": day,
                    "code": code,
                    "entry_date": position.entry_date,
                    "quantity": position.quantity,
                    "available_quantity": position.available_quantity,
                    "close": position.last_close,
                    "market_value": market_values[code],
                    "stale_valuation": code in stale_codes,
                    "pending_exit": code in pending_sells,
                }
            )

    return BacktestResult(
        trades=pd.DataFrame(trade_rows, columns=_TRADE_COLUMNS),
        orders=pd.DataFrame(order_rows, columns=_ORDER_COLUMNS),
        round_trips=pd.DataFrame(round_trip_rows, columns=_ROUND_TRIP_COLUMNS),
        daily_account=pd.DataFrame(account_rows, columns=_ACCOUNT_COLUMNS),
        daily_positions=pd.DataFrame(position_rows, columns=_POSITION_COLUMNS),
        action_log=pd.DataFrame(action_rows, columns=_ACTION_COLUMNS),
    )
