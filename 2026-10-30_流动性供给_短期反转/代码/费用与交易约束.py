"""A股交易费用、涨跌停、整股、滑点规则（冻结副本）。

来源：/Users/admin/WorkSpace/trading/trading_趋势结构/02_代码/费用与交易约束.py
源文件 SHA256：149b0f7f1d4bfdb6f77cb3cc82e14e007f3a4b7f7d96c88bd0bb5d0542c33d7f
复制日期：2026-10-30。母项目只读，不复用其运行状态；此处保留纯函数规则，
供本项目独立调用。未做任何数值/规则修改。

口径要点
--------
* 佣金双边收取，单笔有最低佣金；印花税仅卖出（2008-09-19 起单边）；
* 过户费：2015-08-01 前沪市按面值 0.6‰（每股 0.0006 元），深市不收；
  2015-08-01 起沪深按成交额 0.02‰ 双边，2022-04-29 起 0.01‰；
* 涨跌停按板块与 ST 状态、历史制度日期区分；新股无涨跌幅窗口单独处理；
* 价格最小变动单位 0.01 元；涨跌停价四舍五入到分；
* 买入数量按板块整股规则向下取整。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import math
from typing import Sequence

TICK = 0.01
TICK_TOL = 0.005
EPS = 1e-9


def to_date(value) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError(f"无法转换为日期: {value!r}")


def round_tick(price: float, mode: str = "half_up") -> float:
    """按 0.01 元取整。mode: half_up / up / down。"""
    if price is None:
        raise ValueError("price 不能为空")
    quant = Decimal("0.01")
    rounding = {"half_up": ROUND_HALF_UP, "up": ROUND_CEILING, "down": ROUND_FLOOR}[mode]
    return float(Decimal(str(float(price))).quantize(quant, rounding=rounding))


def round_money(value: float) -> float:
    return float(Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def code_digits(code: str) -> str:
    c = str(code).strip()
    if "." in c:
        a, b = c.split(".", 1)
        if a.lower() in {"sh", "sz", "bj"}:
            return b
        return a
    return c


def board_of(code: str) -> str:
    """返回 MAIN / STAR / CHINEXT / BSE / B_SHARE。"""
    c = code_digits(code)
    if c.startswith(("688", "689")):
        return "STAR"
    if c.startswith(("300", "301")):
        return "CHINEXT"
    if c.startswith(("900", "200")):
        return "B_SHARE"
    if c.startswith(("4", "8", "920")):
        return "BSE"
    return "MAIN"


def exchange_of(code: str) -> str:
    c = str(code).strip()
    if c.lower().startswith("sh.") or c.upper().endswith(".SH"):
        return "SH"
    if c.lower().startswith("sz.") or c.upper().endswith(".SZ"):
        return "SZ"
    if c.lower().startswith("bj.") or c.upper().endswith(".BJ"):
        return "BJ"
    c = code_digits(c)
    if c.startswith(("6", "9")):
        return "SH"
    if c.startswith(("0", "2", "3")):
        return "SZ"
    return "BJ"


@dataclass(frozen=True)
class StampRule:
    effective: date
    rate: float
    sides: str


@dataclass(frozen=True)
class TransferRule:
    effective: date
    mode: str
    rate: float
    scope: str


DEFAULT_STAMP_SCHEDULE: tuple[StampRule, ...] = (
    StampRule(date(2005, 1, 24), 0.0010, "both"),
    StampRule(date(2007, 5, 30), 0.0030, "both"),
    StampRule(date(2008, 4, 24), 0.0010, "both"),
    StampRule(date(2008, 9, 19), 0.0010, "sell"),
    StampRule(date(2023, 8, 28), 0.0005, "sell"),
)

DEFAULT_TRANSFER_SCHEDULE: tuple[TransferRule, ...] = (
    TransferRule(date(2005, 1, 1), "par_sh", 0.0006, "SH"),
    TransferRule(date(2015, 8, 1), "amount", 0.00002, "BOTH"),
    TransferRule(date(2022, 4, 29), "amount", 0.00001, "BOTH"),
)


@dataclass(frozen=True)
class FeeConfig:
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_schedule: tuple[StampRule, ...] = DEFAULT_STAMP_SCHEDULE
    transfer_schedule: tuple[TransferRule, ...] = DEFAULT_TRANSFER_SCHEDULE
    slippage_bps: float = 10.0
    max_participation: float = 0.10
    dividend_tax_rate: float = 0.20


def _pick_rule(schedule: Sequence, d: date):
    if not schedule:
        raise ValueError("empty schedule")
    ordered = sorted(schedule, key=lambda r: r.effective)
    chosen = ordered[0]
    for rule in ordered:
        if rule.effective <= d:
            chosen = rule
        else:
            break
    return chosen


def stamp_duty(d: date, side: str, amount: float, schedule: Sequence[StampRule] = DEFAULT_STAMP_SCHEDULE) -> float:
    if amount <= 0:
        return 0.0
    rule = _pick_rule(schedule, to_date(d))
    if rule.sides == "sell" and side != "sell":
        return 0.0
    return round_money(amount * rule.rate)


def transfer_fee(
    d: date,
    code: str,
    side: str,
    shares: int,
    amount: float,
    schedule: Sequence[TransferRule] = DEFAULT_TRANSFER_SCHEDULE,
) -> float:
    if shares <= 0 or amount <= 0:
        return 0.0
    ex = exchange_of(code)
    rule = _pick_rule(schedule, to_date(d))
    if rule.scope == "SH" and ex != "SH":
        return 0.0
    if rule.mode == "par_sh":
        return round_money(shares * rule.rate)
    return round_money(amount * rule.rate)


@dataclass(frozen=True)
class Fees:
    commission: float
    stamp_duty: float
    transfer_fee: float
    total: float

    def as_dict(self) -> dict:
        return {
            "commission": self.commission,
            "stamp_duty": self.stamp_duty,
            "transfer_fee": self.transfer_fee,
            "total": self.total,
        }


def trade_fees(
    d: date,
    code: str,
    side: str,
    price: float,
    qty: int,
    config: FeeConfig | None = None,
) -> Fees:
    if side not in {"buy", "sell"}:
        raise ValueError(f"side 必须是 buy/sell: {side!r}")
    if qty <= 0 or price <= 0:
        return Fees(0.0, 0.0, 0.0, 0.0)
    cfg = config or FeeConfig()
    amount = float(price) * int(qty)
    commission = round_money(max(amount * cfg.commission_rate, cfg.min_commission))
    stamp = stamp_duty(d, side, amount, cfg.stamp_schedule)
    transfer = transfer_fee(d, code, side, int(qty), amount, cfg.transfer_schedule)
    total = round_money(commission + stamp + transfer)
    return Fees(commission, stamp, transfer, total)


@dataclass(frozen=True)
class LotRule:
    buy_min: int
    buy_step: int
    sell_min: int
    sell_step: int


LOT_RULES = {
    "MAIN": LotRule(100, 100, 100, 100),
    "CHINEXT": LotRule(100, 100, 100, 100),
    "STAR": LotRule(200, 1, 200, 1),
    "BSE": LotRule(100, 1, 100, 1),
    "B_SHARE": LotRule(100, 100, 100, 100),
}


def lot_rule(code: str) -> LotRule:
    return LOT_RULES[board_of(code)]


def normalize_buy_qty(qty: int, code: str) -> int:
    q = int(qty)
    if q <= 0:
        return 0
    rule = lot_rule(code)
    if q < rule.buy_min:
        return 0
    if rule.buy_step <= 1:
        return q
    return (q // rule.buy_step) * rule.buy_step


def apply_slippage(price: float, side: str, bps: float) -> float:
    if price is None or price <= 0:
        raise ValueError(f"价格必须为正: {price!r}")
    if side == "buy":
        raw = float(price) * (1.0 + float(bps) / 10000.0)
        return max(TICK, round_tick(raw, "up"))
    if side == "sell":
        raw = float(price) * (1.0 - float(bps) / 10000.0)
        return max(TICK, round_tick(raw, "down"))
    raise ValueError(f"side 必须是 buy/sell: {side!r}")


def participation_cap(volume: float, rate: float) -> int:
    if volume is None or volume <= 0 or rate <= 0:
        return 0
    return int(math.floor(float(volume) * float(rate) + EPS))


def fee_config_dict(cfg: FeeConfig) -> dict:
    return {
        "commission_rate": cfg.commission_rate,
        "min_commission": cfg.min_commission,
        "slippage_bps": cfg.slippage_bps,
        "max_participation": cfg.max_participation,
        "dividend_tax_rate": cfg.dividend_tax_rate,
        "stamp_schedule": [
            {"effective": r.effective.isoformat(), "rate": r.rate, "sides": r.sides}
            for r in sorted(cfg.stamp_schedule, key=lambda x: x.effective)
        ],
        "transfer_schedule": [
            {"effective": r.effective.isoformat(), "mode": r.mode, "rate": r.rate, "scope": r.scope}
            for r in sorted(cfg.transfer_schedule, key=lambda x: x.effective)
        ],
    }
