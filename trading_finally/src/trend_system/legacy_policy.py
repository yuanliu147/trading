"""Optional account policy for frozen legacy candidate A only."""

from dataclasses import dataclass
import math

import pandas as pd

from .execution import size_buy
from .legacy_a import order_legacy_a_candidates


@dataclass(frozen=True)
class _SizingConfig:
    max_weight: float
    max_exposure: float
    slippage_bps: int


@dataclass
class _PositionState:
    stop_anchor: float
    entry_factor: float
    held: int = 0


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _less(left, right):
    left = _finite(left)
    right = _finite(right)
    return left is not None and right is not None and left < right


class LegacyPolicy:
    """Candidate-A ordering, sizing, fixed entry anchor and close exits."""

    def __init__(self, config):
        self.candidate = next(
            candidate
            for candidate in config.candidates
            if candidate.name == "three_of_two_low_vol"
        )

    @property
    def max_slots(self):
        return self.candidate.max_slots

    @property
    def max_new_per_day(self):
        return self.candidate.max_new_per_day

    def order_candidates(self, candidates):
        if not candidates:
            return []
        ordered = order_legacy_a_candidates(pd.DataFrame(candidates))
        return ordered["code"].tolist()

    def size_order(self, bar, equity, cash, invested, account_config, positions=None):
        sizing_config = _SizingConfig(
            max_weight=self.candidate.max_weight,
            max_exposure=1.0,
            slippage_bps=account_config.slippage_bps,
        )
        return size_buy(bar, equity, cash, invested, sizing_config)

    def entry_state(self, signal, bar, fill, date_index=None):
        entry_factor = _finite(bar.get("adjustment_factor"))
        stop_ref = _finite(signal.get("stop_ref"))
        if entry_factor is None or entry_factor <= 0 or stop_ref is None:
            raise ValueError("Candidate-A entry requires positive factor and finite stop_ref")
        structural_stop = stop_ref / entry_factor
        percentage_stop = fill.price * (1 - self.candidate.entry_stop_pct)
        stop_anchor = max(structural_stop, percentage_stop)
        if self.candidate.skip_if_stop_at_or_above_entry and stop_anchor >= fill.price:
            return None
        return _PositionState(stop_anchor=stop_anchor, entry_factor=entry_factor)

    def exit_reason(self, state, bar, signal, date_index=None):
        if bar is None:
            return None
        reasons = []
        has_stock_trading_bar = (
            bar.get("trade_status", 1) == 1
            and (_finite(bar.get("volume")) or 0.0) > 0
        )
        if has_stock_trading_bar:
            state.held += 1
            close = _finite(bar.get("close"))
            factor = _finite(bar.get("adjustment_factor"))
            if close is not None and factor is not None and factor > 0:
                # adjustment_factor = signal_close / raw close grows at a
                # bonus/split. A fixed actual-scale anchor therefore shrinks
                # by entry_factor / current_factor, not the inverse.
                stop = state.stop_anchor * state.entry_factor / factor
                if close <= stop:
                    reasons.append("stop_loss")
            if state.held >= self.candidate.max_hold:
                reasons.append("max_hold")
            if _less(signal.get("ma5"), signal.get("ma20")):
                reasons.append("ma5_below_ma20")
            if (_finite(signal.get("whist")) or 0.0) < 0:
                reasons.append("whist_negative")
            if _less(signal.get("signal_close"), signal.get("ma20")):
                reasons.append("close_below_ma20")
        st_exit = bar.get("is_st", 0) == 1
        if not reasons:
            return "st_exit" if st_exit else None
        reason = "|".join(reasons)
        return f"{reason};st_exit" if st_exit else reason
