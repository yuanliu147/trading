"""A股交易制度适配器：只读 交易制度参数.json，供回测引擎调用。

设计原则：
- 不修改公共撮合；本模块只回答“某板块/某日/某状态下，涨跌停是多少、费用是多少、能不能成交”。
- 所有日期使用 ISO 字符串或 datetime.date；区间为闭区间。
- 费率均为成交金额的比例（0.001 = 0.1%），不是百分数。
- 价格计算用 Decimal ROUND_HALF_UP，避免浮点把 10.03*1.1 算成 11.032999 后舍错。
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping, Optional

DEFAULT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "交易制度参数.json")
TICK = Decimal("0.01")
TOL = 0.001


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _get(bar: Any, key: str, default=None):
    if isinstance(bar, Mapping):
        return bar.get(key, default)
    return getattr(bar, key, default)


def _round_tick(value: Decimal) -> float:
    return float(value.quantize(TICK, rounding=ROUND_HALF_UP))


class RuleBook:
    def __init__(self, path: str = DEFAULT_JSON):
        with open(path, encoding="utf-8") as fh:
            self.raw = json.load(fh)
        self.boards = self.raw["board_definitions"]
        self.price_limits = self.raw["price_limits"]
        self.ipo_rules = self.raw["ipo_rules"]
        self.delisting_rules = self.raw["delisting_rules"]
        self.execution = self.raw["execution_rules"]
        self.fees = self.raw["fees"]
        self._prefix_map = []
        for board, meta in self.boards.items():
            if board == "EXCLUDED":
                continue
            for prefix in meta.get("code_prefixes", []):
                self._prefix_map.append((prefix, board))
        self._prefix_map.sort(key=lambda item: len(item[0]), reverse=True)

    # ---------- 代码与板块 ----------
    @staticmethod
    def normalize_code(code: str) -> str:
        text = str(code).strip().upper()
        if "." in text:
            head, tail = text.split(".", 1)
            if head in {"SH", "SZ", "BJ"}:
                text = tail
            else:
                text = head
        return text

    def board_of(self, code: str) -> str:
        text = self.normalize_code(code)
        for prefix, board in self._prefix_map:
            if text.startswith(prefix):
                return board
        return "UNKNOWN"

    # ---------- 区间 ----------
    @staticmethod
    def _segment(segments, day: date):
        for seg in segments:
            start = _as_date(seg["from"])
            end = _as_date(seg["to"]) if seg.get("to") else date.max
            if start <= day <= end:
                return seg
        return None

    def _price_segment(self, board: str, day: date):
        return self._segment(self.price_limits.get(board, []), day)

    # ---------- 涨跌停 ----------
    def price_limit_pct(
        self,
        code: str,
        day,
        is_st: bool = False,
        ipo_session: Optional[int] = None,
        delisting: bool = False,
        delisting_session: int = 1,
    ) -> Optional[float]:
        """返回涨跌幅比例；None 表示当日不设涨跌幅限制。

        ipo_session: 上市后第几个交易日（1-based）。None 表示非 IPO 初期。
        delisting_session: 退市整理期第几个交易日（1-based）。

        注意：2014-2023 核准制 IPO 首日返回的是 44% 涨幅帽，跌幅帽是 36%，
        不是对称比例；买卖判断应优先调用 ipo_first_day_cap() 并用 issue_price 计算。
        """
        board = self.board_of(code)
        if board == "UNKNOWN":
            return None
        day = _as_date(day)

        if delisting:
            seg = self._segment(self.delisting_rules.get(board, []), day)
            if seg is None:
                return None
            if delisting_session <= 1 and seg.get("first_day_no_limit"):
                return None
            return float(seg["daily_limit"])

        if ipo_session is not None:
            seg = self._segment(self.ipo_rules.get(board, []), day)
            if seg is not None:
                n = int(seg.get("first_n_sessions_no_limit", 0) or 0)
                if ipo_session <= n:
                    return None
                if seg.get("first_day_cap_up") is not None and ipo_session == 1:
                    # 44%/36% 不是对称比例，调用方应使用 ipo_first_day_cap()
                    return float(seg["first_day_cap_up"])

        seg = self._price_segment(board, day)
        if seg is None:
            return None
        return float(seg["st"] if is_st else seg["normal"])

    def ipo_first_day_cap(self, code: str, day, ipo_session: Optional[int] = None) -> Optional[dict]:
        """2014-2023核准制首日的44%/36%价格帽；返回相对发行价的上下限比例。

        不传 ipo_session 时，只表示“该日期所处制度段存在首日帽”，并不代表
        该日期就是首日；调用方应传 ipo_session=1 或自行确认。传了非 1 的值
        会直接返回 None。
        """
        if ipo_session is not None and ipo_session != 1:
            return None
        board = self.board_of(code)
        if board == "UNKNOWN":
            return None
        seg = self._segment(self.ipo_rules.get(board, []), _as_date(day))
        if not seg or seg.get("first_day_cap_up") is None:
            return None
        return {
            "cap_up": float(seg["first_day_cap_up"]),
            "cap_down": float(seg["first_day_cap_down"]),
            "basis": seg.get("cap_basis", "issue_price"),
        }

    def limit_prices(self, prev_close: float, pct: Optional[float]):
        """pct 为涨跌幅比例的大小（正数）；返回 (跌停价, 涨停价)。"""
        if pct is None:
            return (None, None)
        base = Decimal(str(prev_close))
        step = Decimal(str(abs(pct)))
        up = _round_tick(base * (Decimal(1) + step))
        down = _round_tick(base * (Decimal(1) - step))
        return (down, up)

    def is_limit_up(self, bar, prev_close: float, pct: Optional[float]) -> bool:
        if pct is None:
            return False
        _, up = self.limit_prices(prev_close, pct)
        if up is None:
            return False
        high = _get(bar, "high")
        close = _get(bar, "close")
        return (high is not None and high >= up - TOL) or (close is not None and close >= up - TOL)

    def is_limit_down(self, bar, prev_close: float, pct: Optional[float]) -> bool:
        if pct is None:
            return False
        down, _ = self.limit_prices(prev_close, pct)
        if down is None:
            return False
        low = _get(bar, "low")
        close = _get(bar, "close")
        return (low is not None and low <= down + TOL) or (close is not None and close <= down + TOL)

    def is_one_price_board(self, bar, prev_close: float, pct: Optional[float]) -> bool:
        if pct is None:
            return False
        down, up = self.limit_prices(prev_close, pct)
        high, low, close = _get(bar, "high"), _get(bar, "low"), _get(bar, "close")
        if high is None or low is None or close is None:
            return False
        if abs(high - low) > TOL:
            return False
        at_up = up is not None and abs(close - up) <= TOL
        at_down = down is not None and abs(close - down) <= TOL
        return bool(at_up or at_down)

    # ---------- 最小单位 ----------
    def min_buy_shares(self, code: str) -> int:
        return int(self.execution["min_buy_unit"].get(self.board_of(code), 100))

    def share_increment(self, code: str) -> int:
        return int(self.execution["share_increment"].get(self.board_of(code), 100))

    def round_lot(self, shares: float, code: str) -> int:
        """买入股数向下取整到该板块最小单位；不足最小单位返回0。"""
        unit = self.min_buy_shares(code)
        inc = self.share_increment(code)
        qty = int(shares)
        if qty < unit:
            return 0
        if inc == 1:
            return qty
        return (qty // inc) * inc

    # ---------- 费用 ----------
    def _commission_pct(self, day: date, scenario: str) -> float:
        segs = self.fees["commission"]["scenarios"][scenario]
        seg = self._segment(segs, day)
        if seg is None:
            raise ValueError(f"no commission segment for {day} scenario={scenario}")
        return float(seg["pct"])

    def commission(self, day, amount: float, side: str = "buy", scenario: str = "conservative") -> float:
        pct = self._commission_pct(_as_date(day), scenario)
        floor = float(self.fees["commission"]["min_per_order_cny"])
        return round(max(floor, float(amount) * pct), 2)

    def stamp_duty(self, day, side: str, amount: float) -> float:
        seg = self._segment(self.fees["stamp_duty"], _as_date(day))
        if seg is None:
            return 0.0
        return round(float(amount) * float(seg.get(side, 0.0)), 2)

    def transfer_fee(self, day, board: str, amount: float, shares: Optional[float] = None) -> float:
        seg = self._segment(self.fees["transfer_fee"]["schedule"], _as_date(day))
        if seg is None:
            return 0.0
        if "ALL" in seg:
            return round(float(amount) * float(seg["ALL"]["pct"]), 2)
        exchange = self.boards.get(board, {}).get("exchange", "")
        key = "SSE" if exchange == "SSE" else "SZSE" if exchange == "SZSE" else "BSE"
        item = seg.get(key)
        if item is None:
            return 0.0
        if item["basis"] == "par_value":
            if shares is None:
                raise ValueError("transfer_fee with par_value basis requires shares")
            return round(float(shares) * float(item["pct"]), 2)
        return round(float(amount) * float(item["pct"]), 2)

    def exchange_handling_fee(self, day, board: str, amount: float) -> float:
        exchange = self.boards.get(board, {}).get("exchange", "")
        schedule = self.fees["exchange_handling_fee"]["schedule_by_exchange"].get(exchange, [])
        seg = self._segment(schedule, _as_date(day))
        if seg is None:
            return 0.0
        return round(float(amount) * float(seg["pct"]), 2)

    def regulatory_fee(self, amount: float) -> float:
        return round(float(amount) * float(self.fees["regulatory_fee"]["pct"]), 2)

    def total_fees(
        self,
        day,
        board: str,
        side: str,
        amount: float,
        shares: Optional[float] = None,
        scenario: str = "conservative",
    ) -> dict:
        commission = self.commission(day, amount, side, scenario)
        stamp = self.stamp_duty(day, side, amount)
        transfer = self.transfer_fee(day, board, amount, shares)
        return {
            "commission": commission,
            "stamp_duty": stamp,
            "transfer_fee": transfer,
            "total": commission + stamp + transfer,
        }

    def dividend_tax_rate(self, day, holding_days: int) -> float:
        seg = self._segment(self.fees["dividend_tax"], _as_date(day))
        if seg is None:
            return 0.0
        if holding_days <= 30:
            return float(seg["le_1m"])
        if holding_days <= 365:
            return float(seg["m1_to_y1"])
        return float(seg["gt_1y"])

    # ---------- 成交可行性 ----------
    def can_buy(
        self,
        code: str,
        day,
        bar,
        prev_close: float,
        is_st: bool = False,
        ipo_session: Optional[int] = None,
        delisting: bool = False,
        delisting_session: int = 1,
        issue_price: Optional[float] = None,
        suspended: bool = False,
    ):
        if suspended:
            return False, "suspended"
        if delisting:
            return False, "delisting_period"
        board = self.board_of(code)
        if board == "UNKNOWN":
            return False, "unknown_board"
        open_px = _get(bar, "open")
        if open_px is None or open_px <= 0:
            return False, "no_open"
        cap = self.ipo_first_day_cap(code, day) if ipo_session == 1 else None
        if cap and issue_price is not None:
            cap_price = _round_tick(Decimal(str(issue_price)) * (Decimal(1) + Decimal(str(cap["cap_up"]))))
            if open_px >= cap_price - TOL:
                return False, "ipo_first_day_cap_up"
            return True, "ok"
        pct = self.price_limit_pct(code, day, is_st, ipo_session, delisting, delisting_session)
        if pct is None:
            return True, "ok_no_limit"
        _, up = self.limit_prices(prev_close, pct)
        if up is not None and open_px >= up - TOL:
            return False, "limit_up_open"
        if self.is_one_price_board(bar, prev_close, pct):
            return False, "one_price_limit_board"
        return True, "ok"

    def can_sell(
        self,
        code: str,
        day,
        bar,
        prev_close: float,
        is_st: bool = False,
        delisting: bool = False,
        delisting_session: int = 1,
        held_days: int = 1,
        suspended: bool = False,
        ipo_session: Optional[int] = None,
        issue_price: Optional[float] = None,
    ):
        if suspended:
            return False, "suspended"
        if held_days < 1:
            return False, "t_plus_one"
        board = self.board_of(code)
        if board == "UNKNOWN":
            return False, "unknown_board"
        if delisting and self._segment(self.delisting_rules.get(board, []), _as_date(day)) is None:
            return False, "no_delisting_period"
        open_px = _get(bar, "open")
        if open_px is None or open_px <= 0:
            return False, "no_open"
        cap = self.ipo_first_day_cap(code, day) if ipo_session == 1 else None
        if cap and issue_price is not None:
            floor = _round_tick(Decimal(str(issue_price)) * (Decimal(1) - Decimal(str(cap["cap_down"]))))
            if open_px <= floor + TOL:
                return False, "ipo_first_day_cap_down"
            return True, "ok"
        pct = self.price_limit_pct(code, day, is_st, None, delisting, delisting_session)
        if pct is None:
            return True, "ok_no_limit"
        down, _ = self.limit_prices(prev_close, pct)
        if down is not None and open_px <= down + TOL:
            return False, "limit_down_open"
        if self.is_one_price_board(bar, prev_close, pct):
            return False, "one_price_limit_board"
        return True, "ok"


if __name__ == "__main__":
    rb = RuleBook()
    for code in ("sh.600000", "sz.300750", "sh.688001", "920242.BJ", "sz.002594"):
        print(code, rb.board_of(code), rb.min_buy_shares(code), rb.price_limit_pct(code, "2026-10-26"))
