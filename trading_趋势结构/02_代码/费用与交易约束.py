"""A股交易费用、涨跌停、整股、滑点与除权参考价规则。

本模块只做纯函数计算，不读数据库、不联网、不持有账户状态。所有规则均可配置，
默认值按“保守、可解释、可审计”选取，而不是按历史收益最优选取。

口径要点
--------
* 佣金双边收取，单笔有最低佣金；印花税仅卖出（2008-09-19 起单边，此前双边）；
* 过户费：2015-08-01 前沪市按面值 0.6‰（每股 0.0006 元），深市不收；
  2015-08-01 起沪深按成交额 0.02‰ 双边，2022-04-29 起 0.01‰；
* 涨跌停按板块与 ST 状态、历史制度日期区分；新股无涨跌幅窗口单独处理；
* 价格最小变动单位 0.01 元；涨跌停价四舍五入到分；
* 买入数量按板块整股规则向下取整；卖出允许零股一次性卖出，但部分卖出须满足板块规则；
* 滑点按基点向不利方向取整到分；限价单成交价不劣于限价。
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


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 费用规则
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StampRule:
    effective: date
    rate: float
    sides: str  # both / sell


@dataclass(frozen=True)
class TransferRule:
    effective: date
    mode: str      # par_sh / amount
    rate: float
    scope: str     # SH / BOTH


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
    slippage_bps: float = 5.0
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
        # 沪市按面值（默认 1 元/股）收取，与成交价无关
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


# ---------------------------------------------------------------------------
# 涨跌停
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PriceLimits:
    ref: float
    up: float
    down: float
    pct_up: float
    pct_down: float
    no_limit: bool = False


def regular_limit_pct(code: str, is_st: bool, d: date) -> tuple[float, float]:
    board = board_of(code)
    if board == "STAR":
        return 0.20, 0.20
    if board == "CHINEXT":
        if to_date(d) >= date(2020, 8, 24):
            return 0.20, 0.20
        return (0.05, 0.05) if is_st else (0.10, 0.10)
    if board == "BSE":
        return 0.30, 0.30
    return (0.05, 0.05) if is_st else (0.10, 0.10)


def ipo_limit_pct(code: str, is_st: bool, d: date, days_since_ipo: int | None):
    """返回 (涨幅上限, 跌幅上限)；None 表示当日无涨跌幅限制。

    days_since_ipo=0 表示上市首日。主板 2023-04-10 前首日 44%/36%，
    此后前 5 日无限制；创业板 2020-08-24 前首日 44%/36%，此后前 5 日无限制；
    科创板前 5 日、北交所首日无限制。制度细节按公开规则近似，报告中有标注。
    """
    if days_since_ipo is None or days_since_ipo < 0:
        return regular_limit_pct(code, is_st, d)
    board = board_of(code)
    dd = to_date(d)
    if board == "STAR":
        if days_since_ipo < 5:
            return None
        return 0.20, 0.20
    if board == "CHINEXT":
        if dd >= date(2020, 8, 24):
            if days_since_ipo < 5:
                return None
            return 0.20, 0.20
        return (0.44, 0.36) if days_since_ipo == 0 else ((0.05, 0.05) if is_st else (0.10, 0.10))
    if board == "BSE":
        if days_since_ipo < 1:
            return None
        return 0.30, 0.30
    # 主板
    if dd >= date(2023, 4, 10):
        if days_since_ipo < 5:
            return None
        return (0.05, 0.05) if is_st else (0.10, 0.10)
    return (0.44, 0.36) if days_since_ipo == 0 else ((0.05, 0.05) if is_st else (0.10, 0.10))


def limit_prices(
    ref: float,
    code: str,
    is_st: bool,
    d: date,
    days_since_ipo: int | None = None,
) -> PriceLimits:
    if ref is None or ref <= 0:
        raise ValueError(f"参考价必须为正: {ref!r}")
    # 交易所先公布四舍五入到 0.01 的除权参考价，再以它为基数算涨跌停
    ref = round_tick(float(ref), "half_up")
    pcts = ipo_limit_pct(code, is_st, d, days_since_ipo)
    if pcts is None:
        return PriceLimits(ref=float(ref), up=math.inf, down=0.0, pct_up=math.inf, pct_down=math.inf, no_limit=True)
    up_pct, down_pct = pcts
    up = round_tick(float(ref) * (1.0 + up_pct), "half_up")
    down = round_tick(max(0.0, float(ref) * (1.0 - down_pct)), "half_up")
    return PriceLimits(ref=float(ref), up=up, down=down, pct_up=up_pct, pct_down=down_pct, no_limit=False)


def at_limit_up(price: float, limits: PriceLimits) -> bool:
    return (not limits.no_limit) and price >= limits.up - TICK_TOL


def at_limit_down(price: float, limits: PriceLimits) -> bool:
    return (not limits.no_limit) and price <= limits.down + TICK_TOL


def blocked_buy_at_open(open_price: float, limits: PriceLimits) -> bool:
    # 一字/开盘即涨停时保守视为买不到；盘中打开涨停的机会不建模
    return at_limit_up(open_price, limits)


def blocked_sell_at_open(open_price: float, limits: PriceLimits) -> bool:
    return at_limit_down(open_price, limits)


def ex_rights_reference(
    prev_close: float,
    dividend_per_share: float = 0.0,
    per_share_bonus: float = 0.0,
    rights_ratio: float = 0.0,
    rights_price: float = 0.0,
) -> float:
    """除权除息参考价 = (前收 - 每股现金红利 + 配股价×配股比例) / (1 + 送转比例 + 配股比例)。"""
    denom = 1.0 + float(per_share_bonus) + float(rights_ratio)
    if denom <= EPS:
        raise ValueError("送转/配股比例导致分母<=0")
    return (float(prev_close) - float(dividend_per_share) + float(rights_price) * float(rights_ratio)) / denom


# ---------------------------------------------------------------------------
# 整股与申报数量
# ---------------------------------------------------------------------------

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
    """买入申报数量按板块规则向下取整；不足最小申报数量返回 0。"""
    q = int(qty)
    if q <= 0:
        return 0
    rule = lot_rule(code)
    if q < rule.buy_min:
        return 0
    if rule.buy_step <= 1:
        return q
    return (q // rule.buy_step) * rule.buy_step


def adjust_sell_fill_qty(qty: int, available: int, code: str) -> int:
    """在部分成交/流动性约束下，把卖出数量调整成合法申报量。"""
    q = min(int(qty), int(available))
    if q <= 0:
        return 0
    if q == int(available):
        return q  # 余额一次性卖出，允许零股
    rule = lot_rule(code)
    if q < rule.sell_min:
        return 0
    if rule.sell_step <= 1:
        return q
    return (q // rule.sell_step) * rule.sell_step


def validate_sell_qty(qty: int, available: int, code: str) -> tuple[bool, str]:
    q = int(qty)
    a = int(available)
    if q <= 0:
        return False, "qty<=0"
    if q > a:
        return False, "qty>available"
    if q == a:
        return True, "all_remaining"
    rule = lot_rule(code)
    if q < rule.sell_min:
        return False, "below_sell_min"
    if rule.sell_step > 1 and q % rule.sell_step != 0:
        return False, "not_lot_multiple"
    return True, "ok"


# ---------------------------------------------------------------------------
# 滑点与成交量约束
# ---------------------------------------------------------------------------

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


def validate_fee_config(cfg: FeeConfig) -> list[str]:
    errs: list[str] = []
    if not (0 <= cfg.commission_rate <= 0.01):
        errs.append("commission_rate 超出 [0,1%]")
    if cfg.min_commission < 0:
        errs.append("min_commission 不能为负")
    if not (0 <= cfg.slippage_bps <= 500):
        errs.append("slippage_bps 超出 [0,500]")
    if not (0 < cfg.max_participation <= 1):
        errs.append("max_participation 必须在 (0,1]")
    if not (0 <= cfg.dividend_tax_rate <= 1):
        errs.append("dividend_tax_rate 必须在 [0,1]")
    for r in cfg.stamp_schedule:
        if r.rate < 0 or r.sides not in {"both", "sell"}:
            errs.append(f"非法印花税规则: {r}")
    for r in cfg.transfer_schedule:
        if r.rate < 0 or r.mode not in {"par_sh", "amount"} or r.scope not in {"SH", "BOTH"}:
            errs.append(f"非法过户费规则: {r}")
    return errs
