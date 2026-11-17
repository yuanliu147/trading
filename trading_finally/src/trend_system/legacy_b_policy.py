"""Optional account policy for recovered legacy candidate B only."""

from dataclasses import dataclass
import math

import pandas as pd

from . import execution


@dataclass
class _BPositionState:
    risk_ps: float
    entry_factor: float
    entry_index: int
    stop_anchor: float


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class LegacyBPolicy:
    """Candidate-B ordering, risk sizing, heat cap and close-confirmed exits."""

    def __init__(self, config, gap_max=0.05):
        self.candidate = next(
            candidate
            for candidate in config.candidates
            if candidate.name == "early_trend_breadth"
        )
        self.gap_max = gap_max
        self.pending: dict[str, dict] = {}
        self.last_rejection_reason = "policy_entry_rejected"

    @property
    def max_slots(self):
        return self.candidate.max_slots

    @property
    def max_new_per_day(self):
        return self.candidate.max_slots

    def order_candidates(self, candidates):
        if not candidates:
            return []
        frame = pd.DataFrame(candidates).sort_values(
            ["date", "legacy_rank_score", "code"],
            ascending=[True, False, True],
            kind="stable",
        )
        for row in frame.to_dict("records"):
            self.pending[str(row["code"])] = row
        return frame["code"].tolist()

    def size_order(self, bar, equity, cash, invested, account_config, positions=None):
        code = str(bar.get("code"))
        signal = self.pending.get(code, {})
        atr = _finite(signal.get("atr_pct"))
        open_price = _finite(bar.get("open"))
        if atr is None or open_price is None or open_price <= 0 or equity <= 0 or cash <= 0:
            return 0
        sd_size = min(max(2.0 * atr, self.candidate.risk_size_min), self.candidate.risk_size_max)
        risk_ps = open_price * sd_size
        if risk_ps <= 0:
            return 0
        heat = 0.0
        for position in (positions or {}).values():
            state = position.policy_state
            if state is not None:
                heat += position.quantity * state.risk_ps
        heat /= equity
        if heat + self.candidate.risk_per_trade > self.candidate.heat_reject_threshold:
            return 0
        risk_amount = equity * self.candidate.risk_per_trade
        target = min(
            risk_amount / risk_ps,
            equity * self.candidate.max_weight / open_price,
        )
        quantity = int(target) // 100 * 100
        fill_price = float(execution._final_fill_price(bar, "buy", account_config))
        while quantity > 0:
            amount = fill_price * quantity
            costs = execution.transaction_costs(
                str(bar.get("date")), float(amount), "buy"
            )
            if amount + float(costs["total"]) <= cash:
                break
            quantity -= 100
        if quantity <= 0 or quantity * open_price < equity * self.candidate.min_position_weight:
            return 0
        return quantity

    def entry_state(self, signal, bar, fill, date_index=None):
        factor = _finite(bar.get("adjustment_factor"))
        open_price = _finite(bar.get("open"))
        prev_close = _finite(bar.get("prev_close"))
        if factor is None or factor <= 0 or open_price is None or prev_close is None:
            self.last_rejection_reason = "policy_entry_rejected"
            return None
        if open_price / prev_close - 1 > self.gap_max:
            self.last_rejection_reason = "gap_filter"
            return None
        atr = _finite(signal.get("atr_pct"))
        if atr is None:
            self.last_rejection_reason = "policy_entry_rejected"
            return None
        sd_size = min(max(2.0 * atr, self.candidate.risk_size_min), self.candidate.risk_size_max)
        if date_index is None:
            self.last_rejection_reason = "policy_entry_rejected"
            return None
        return _BPositionState(
            risk_ps=fill.price * sd_size,
            entry_factor=factor,
            entry_index=int(date_index),
            stop_anchor=fill.price * (1.0 - self.candidate.catastrophe_stop_pct),
        )

    def exit_reason(self, state, bar, signal, date_index=None):
        if bar is None or date_index is None:
            return None
        toggle = _finite(bar.get("close"))
        factor = _finite(bar.get("adjustment_factor"))
        normal = bar.get("trade_status", 1) == 1 and toggle is not None and toggle > 0
        if not normal or factor is None or factor <= 0:
            return None
        reasons = []
        stop = state.stop_anchor * state.entry_factor / factor
        if toggle <= stop:
            reasons.append("catastrophe_stop")
        if date_index - state.entry_index + 1 >= self.candidate.max_hold:
            reasons.append("time_exit")
        st_exit = bar.get("is_st", 0) == 1
        if not reasons:
            return "st_exit" if st_exit else None
        reason = "|".join(reasons)
        return f"{reason};st_exit" if st_exit else reason
