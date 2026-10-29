"""持仓状态机与退出策略对象。

设计约束（与 03_结果/退出与仓位_实验规范.md 一致，改动需同步改规范）：

1. 因果性：所有判定只用 t 日收盘及之前的数据；订单最早在 t+1 开盘成交。
2. 状态是判定输出，不是可调参数。
3. 止损/目标/分批是 t 日收盘挂出的条件单，t+1 日起生效；同日既触止损又触目标时按止损先成交。
4. MFE（最大有利偏移）只在平仓后作为诊断指标，绝不参与离场判定。
5. 2021-01-01 之后禁止计算盈亏，由 assert_in_window 硬拦截。

本模块不依赖任何第三方库，可独立导入。引擎（退出_回测引擎.py）负责撮合、
费用、复权与组合记账；本模块只负责"该不该继续持有"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

FREEZE_END = "2026-10-26"
DEV_START = "2008-01-01"
VAL_START = "2016-01-01"
HOLDOUT_START = "2021-01-01"
WARMUP_START = "2006-01-04"


class HoldoutViolation(RuntimeError):
    """试图在留出窗口计算盈亏。"""


def assert_in_window(date: str, purpose: str = "绩效计算") -> None:
    """拦截留出窗口内的绩效计算。

    只允许 purpose='audit' 的覆盖审计穿过。任何策略收益、净值、交易盈亏
    都必须在 <=2020-12-31 上计算。
    """
    if purpose == "audit":
        return
    if date >= HOLDOUT_START:
        raise HoldoutViolation(
            f"{date} 属于留出窗口（>= {HOLDOUT_START}），禁止计算{purpose}。"
            "规则冻结后由主协调者单次运行。"
        )


def window_of(date: str) -> str:
    if date < DEV_START:
        return "warmup"
    if date < VAL_START:
        return "dev"
    if date < HOLDOUT_START:
        return "val"
    return "holdout"


# --------------------------------------------------------------------------
# 状态
# --------------------------------------------------------------------------


class PosState(str, Enum):
    ON_TRACK = "预期内推进"
    CONSOLIDATING = "盘整待确认"
    CAUTION = "谨慎"
    THESIS_INVALID = "假设失效"
    FORCED_EXIT = "强制离场"
    TREND_EXTENSION = "趋势延伸"
    CLOSED = "已离场"


EXIT_STATES = frozenset({PosState.THESIS_INVALID, PosState.FORCED_EXIT})


class ExitReason(str, Enum):
    INIT_STOP = "初始止损"
    TRAIL_STOP = "追踪止损"
    STRUCT_STOP = "结构止损"
    BREAKEVEN_STOP = "保本止损"
    STRUCT_EXIT = "结构破坏离场"
    TREND_FAIL = "趋势失效离场"
    TIME_STOP = "时间止损"
    TARGET = "固定目标止盈"
    SCALE_OUT = "分批止盈"
    HARD_MAX_HOLD = "最长持有到期"
    SUSPENSION = "停牌超期"
    DELIST = "退市强平"
    END_OF_DATA = "数据末端"
    RISK_CAP = "组合风险超限减仓"


# --------------------------------------------------------------------------
# 策略对象
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitPolicy:
    """一个规则臂的完整参数。默认值 = B0 基准臂（仅初始止损，无追踪/无目标）。"""

    name: str = "B0"
    # 初始止损口径：atr | fixed | struct
    init_stop_mode: str = "atr"
    init_atr_mult: float = 2.0
    init_fixed_pct: float = 0.07
    min_stop_atr: float = 0.5        # 止损距离下限，防止结构位过近导致仓位爆炸
    # 追踪止损（ATR 吊灯）；None 表示不启用
    trail_atr_mult: Optional[float] = None
    # 结构移动止损：最近已确认分型低点 - struct_buf*ATR
    struct_trail: bool = False
    struct_buf: float = 0.25
    # 保本推进：MFE >= 该 R 倍数时把止损抬到成本
    breakeven_at_R: Optional[float] = None
    # 时间止损
    time_stop_days: Optional[int] = None
    time_stop_min_R: float = 0.5
    # 分批止盈（条件单，日内成交）
    scale_at_R: Optional[float] = None
    scale_frac: float = 0.5
    # 固定目标（条件单，日内成交）
    target_R: Optional[float] = None
    # 结构破坏离场：收盘 < 最近 N 日最低
    struct_exit_lookback: Optional[int] = None
    # 趋势失效离场：浮亏且收盘 < MA60 且 MA60 下行
    ma60_exit: bool = False
    # 状态分层是否影响追踪宽度
    state_overlay: bool = False
    caution_trail_mult: float = 2.0
    extension_trail_mult: float = 4.5
    # 硬边界
    hard_max_hold: int = 250
    max_suspension_bars: int = 20


def initial_stop_hfq(policy: ExitPolicy, entry_hfq: float, atr: float,
                     pivot_low: Optional[float]) -> float:
    """按策略口径给出初始止损（hfq）。所有口径统一施加 min_stop_atr 距离下限。"""
    floor = entry_hfq - policy.min_stop_atr * atr
    if policy.init_stop_mode == "atr":
        stop = entry_hfq - policy.init_atr_mult * atr
    elif policy.init_stop_mode == "fixed":
        stop = entry_hfq * (1.0 - policy.init_fixed_pct)
    elif policy.init_stop_mode == "struct":
        if pivot_low is None or not (pivot_low < entry_hfq):
            stop = entry_hfq - policy.init_atr_mult * atr
        else:
            stop = pivot_low - policy.struct_buf * atr
    else:
        raise ValueError(policy.init_stop_mode)
    return min(stop, floor)


# 臂定义。B0 是唯一基准，其余每个臂只改一处。
ARMS: dict[str, ExitPolicy] = {
    # --- 基准：仅初始止损 ---
    "B0": ExitPolicy(name="B0"),
    # --- 初始止损口径对照 ---
    "S_fix": ExitPolicy(name="S_fix", init_stop_mode="fixed", init_fixed_pct=0.07),
    "S_struct": ExitPolicy(name="S_struct", init_stop_mode="struct"),
    # --- 移动止损 ---
    "T2": ExitPolicy(name="T2", trail_atr_mult=2.0),
    "T3": ExitPolicy(name="T3", trail_atr_mult=3.0),
    "T45": ExitPolicy(name="T45", trail_atr_mult=4.5),
    "T_struct": ExitPolicy(name="T_struct", struct_trail=True),
    "T6": ExitPolicy(name="T6", trail_atr_mult=6.0),
    # --- 保本 ---
    "BE": ExitPolicy(name="BE", breakeven_at_R=1.0),
    # --- 时间止损 ---
    "TS10": ExitPolicy(name="TS10", time_stop_days=10),
    "TS20": ExitPolicy(name="TS20", time_stop_days=20),
    "TS40": ExitPolicy(name="TS40", time_stop_days=40),
    # --- 分批止盈 / 固定目标 ---
    "SC": ExitPolicy(name="SC", scale_at_R=2.0, trail_atr_mult=3.0),
    "TG2": ExitPolicy(name="TG2", target_R=2.0),
    # --- 结构破坏离场 ---
    "SE20": ExitPolicy(name="SE20", struct_exit_lookback=20),
    "MX60": ExitPolicy(name="MX60", ma60_exit=True),
    # --- 状态机分层 ---
    "STK": ExitPolicy(
        name="STK", trail_atr_mult=3.0, breakeven_at_R=1.0, state_overlay=True,
        caution_trail_mult=2.0, extension_trail_mult=4.5,
    ),
    # --- 组合臂：由原则拼装，不由数据挑选 ---
    "ALL": ExitPolicy(
        name="ALL", trail_atr_mult=3.0, breakeven_at_R=1.0,
        time_stop_days=40, struct_exit_lookback=20,
    ),
}


# --------------------------------------------------------------------------
# 运行期数据结构
# --------------------------------------------------------------------------


@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    preclose: float
    volume: float
    amount: float
    tradable: bool = True          # 有 bar 且 volume>0 且 trade_status==1
    limit_up: float = float("nan")
    limit_down: float = float("nan")

    @property
    def one_price_board(self) -> bool:
        return self.high <= self.low + 1e-12

    def is_locked_up(self) -> bool:
        """一字涨停：开盘起就封死，买不进。"""
        return self.one_price_board and self.limit_up == self.limit_up and self.low >= self.limit_up - 0.011

    def is_locked_down(self) -> bool:
        """一字跌停：卖不出。"""
        return self.one_price_board and self.limit_down == self.limit_down and self.high <= self.limit_down + 0.011


@dataclass
class Indicators:
    """t 日收盘后可得的指标，全部基于 hfq 价格。"""

    atr20: float
    ma20: float
    ma60: float
    ma60_slope: float                # ma60[t] / ma60[t-5] - 1
    vol_ma20: float
    low20: float                     # 前 20 日最低（不含 t；含 t 会让结构离场永不触发）
    close_max3: float                # 最近 3 日最高收盘
    pivot_low: Optional[float] = None  # 本 bar 新确认的分型低点（j+2==t）


@dataclass
class Decision:
    state: PosState
    exit_frac: float = 0.0
    reasons: tuple[str, ...] = ()
    stop: float = float("nan")
    notes: tuple[str, ...] = ()


@dataclass
class Position:
    """一个持仓的生命周期。所有价格均为后复权(hfq)。"""

    code: str
    entry_date: str
    entry_px: float
    shares: float
    init_stop: float
    R: float
    policy: ExitPolicy
    stop: float = float("nan")
    stop_kind: str = ExitReason.INIT_STOP.value
    struct_stop: float = float("-inf")
    n_bars: int = 0
    susp_bars: int = 0
    max_high: float = float("-inf")
    max_close: float = float("-inf")
    min_low: float = float("inf")
    scale_done: bool = False
    target_active: bool = True
    state: PosState = PosState.CONSOLIDATING
    pending_exit_frac: float = 0.0
    pending_reason: Optional[str] = None
    # 事后诊断，不参与判定
    mfe_R: float = 0.0
    mae_R: float = 0.0
    realized: list = field(default_factory=list)   # [(date, frac, px, reason)]

    def __post_init__(self):
        if not self.stop == self.stop:
            self.stop = self.init_stop
        self.max_high = max(self.max_high, self.entry_px)
        self.max_close = max(self.max_close, self.entry_px)
        self.min_low = min(self.min_low, self.entry_px)

    # -- 状态判定 -----------------------------------------------------------

    def mark_extremes(self, high: float, low: float) -> None:
        """每日先记录极值，保证中途离场的交易也有正确的 MFE/MAE 诊断。"""
        self.max_high = max(self.max_high, high)
        self.min_low = min(self.min_low, low)
        self.mfe_R = (self.max_high - self.entry_px) / self.R
        self.mae_R = (self.min_low - self.entry_px) / self.R

    def update_state(self, bar: Bar, ind: Indicators) -> PosState:
        """t 日收盘判定。只用 t 及之前的数据。"""
        self.n_bars += 1
        self.max_high = max(self.max_high, bar.high)
        self.max_close = max(self.max_close, bar.close)
        self.min_low = min(self.min_low, bar.low)
        self.mfe_R = (self.max_high - self.entry_px) / self.R
        self.mae_R = (self.min_low - self.entry_px) / self.R
        if ind.pivot_low is not None and self.policy.struct_trail:
            cand = ind.pivot_low - self.policy.struct_buf * ind.atr20
            self.struct_stop = max(self.struct_stop, cand)

        r_cur = (bar.close - self.entry_px) / self.R
        state = self._classify(bar, ind, r_cur)
        self.state = state
        return state

    def _classify(self, bar: Bar, ind: Indicators, r_cur: float) -> PosState:
        p = self.policy

        # 1) 强制离场：与"买入理由"无关的硬边界
        if self.n_bars >= p.hard_max_hold:
            return PosState.FORCED_EXIT
        if self.susp_bars >= p.max_suspension_bars:
            return PosState.FORCED_EXIT
        if p.time_stop_days is not None and self.n_bars >= p.time_stop_days and r_cur < p.time_stop_min_R:
            return PosState.FORCED_EXIT

        # 2) 假设失效：结构破坏。买入理由是不再成立，而不是"亏了"。
        if p.struct_exit_lookback is not None and bar.close < ind.low20:
            return PosState.THESIS_INVALID
        if p.ma60_exit and r_cur < 0 and bar.close < ind.ma60 and ind.ma60_slope < 0:
            return PosState.THESIS_INVALID

        # 3) 谨慎：出现预警信号
        warn_ma20 = bar.close < ind.ma20
        giveback = self.mfe_R >= 1.0 and (self.mfe_R - r_cur) >= 0.5 * self.mfe_R
        day_ret = bar.close / bar.preclose - 1.0 if bar.preclose > 0 else 0.0
        atr_pct = ind.atr20 / bar.close if bar.close > 0 else 0.0
        warn_dump = day_ret < -1.5 * atr_pct and ind.vol_ma20 > 0 and bar.volume > 1.5 * ind.vol_ma20
        if warn_ma20 or giveback or warn_dump:
            return PosState.CAUTION

        # 4) 趋势延伸：仍在推进且回吐有限
        if bar.close >= ind.close_max3 - 1e-12 and r_cur > 0.5:
            return PosState.TREND_EXTENSION
        if self.mfe_R >= 2.0 and r_cur >= 0.6 * self.mfe_R:
            return PosState.TREND_EXTENSION

        # 5) 预期内推进
        if r_cur >= 0.3:
            return PosState.ON_TRACK

        # 6) 盘整待确认
        return PosState.CONSOLIDATING

    # -- 止损更新 -----------------------------------------------------------

    def update_stop(self, bar: Bar, ind: Indicators) -> None:
        """t 日收盘更新止损，t+1 起生效。只上移不下移。"""
        p = self.policy
        cands: list[tuple[float, str]] = []

        if self.policy.struct_trail and self.struct_stop > float("-inf"):
            cands.append((self.struct_stop, ExitReason.STRUCT_STOP.value))

        if p.trail_atr_mult is not None:
            k = p.trail_atr_mult
            if p.state_overlay:
                if self.state is PosState.TREND_EXTENSION:
                    k = p.extension_trail_mult
                elif self.state is PosState.CAUTION:
                    k = p.caution_trail_mult
            cands.append((self.max_high - k * ind.atr20, ExitReason.TRAIL_STOP.value))

        if p.breakeven_at_R is not None and self.mfe_R >= p.breakeven_at_R:
            cands.append((self.entry_px, ExitReason.BREAKEVEN_STOP.value))

        if not cands:
            return
        best_px, best_kind = max(cands, key=lambda x: x[0])
        if best_px > self.stop:
            self.stop = best_px
            self.stop_kind = best_kind

    # -- 收盘决策 -----------------------------------------------------------

    def decide(self, bar: Bar, ind: Indicators) -> Decision:
        """收盘评估，返回下一交易日生效的订单意图。"""
        reasons: list[str] = []
        frac = 0.0

        if self.state is PosState.FORCED_EXIT:
            if self.n_bars >= self.policy.hard_max_hold:
                reasons.append(ExitReason.HARD_MAX_HOLD.value)
            elif self.susp_bars >= self.policy.max_suspension_bars:
                reasons.append(ExitReason.SUSPENSION.value)
            else:
                reasons.append(ExitReason.TIME_STOP.value)
            frac = 1.0
        elif self.state is PosState.THESIS_INVALID:
            if (self.policy.struct_exit_lookback is not None
                    and bar.close < ind.low20):
                reasons.append(ExitReason.STRUCT_EXIT.value)
            else:
                reasons.append(ExitReason.TREND_FAIL.value)
            frac = 1.0

        r_cur = (bar.close - self.entry_px) / self.R
        p = self.policy
        # 固定目标按收盘判定 → 次日开盘执行；日内目标由引擎的条件单处理
        if p.target_R is not None and self.target_active and r_cur >= p.target_R:
            reasons.append(ExitReason.TARGET.value)
            frac = 1.0

        if frac >= 1.0:
            self.pending_exit_frac = 1.0
            self.pending_reason = reasons[0] if reasons else None
        return Decision(
            state=self.state,
            exit_frac=frac,
            reasons=tuple(reasons),
            stop=self.stop,
        )

    # -- 日内条件单 ---------------------------------------------------------

    def intraday_orders(self) -> tuple[float, float, float]:
        """返回 (stop_px, scale_px, target_px)，NaN 表示未挂。"""
        nan = float("nan")
        p = self.policy
        scale_px = nan
        if p.scale_at_R is not None and not self.scale_done:
            scale_px = self.entry_px + p.scale_at_R * self.R
        target_px = nan
        if p.target_R is not None and self.target_active:
            target_px = self.entry_px + p.target_R * self.R
        return self.stop, scale_px, target_px


# --------------------------------------------------------------------------
# 仓位口径
# --------------------------------------------------------------------------


def size_equal_weight(nav: float, slots: int, price: float, lot: int = 100) -> int:
    if price <= 0 or slots <= 0:
        return 0
    budget = nav / slots
    return int(budget // (price * lot)) * lot


def size_risk_budget(
    nav: float,
    price: float,
    stop: float,
    risk_pct: float,
    max_notional: float,
    lot: int = 100,
) -> int:
    """按"入场到初始止损的每股风险"反推股数。

    risk_pct 是账户权益的百分比（0.01 = 1%）。
    """
    per_share_risk = price - stop
    if per_share_risk <= 0 or price <= 0:
        return 0
    budget = risk_pct * nav
    shares_risk = budget / per_share_risk
    shares_cap = max_notional / price
    shares = min(shares_risk, shares_cap)
    return int(shares // lot) * lot


def size_vol_target(
    nav: float,
    price: float,
    atr20: float,
    vol_pct: float,
    max_notional: float,
    lot: int = 100,
) -> int:
    if atr20 <= 0 or price <= 0:
        return 0
    atr_frac = atr20 / price
    notional_by_vol = (vol_pct * nav) / atr_frac
    notional = min(notional_by_vol, max_notional)
    return int((notional / price) // lot) * lot


def open_risk(positions, price_of) -> float:
    """组合未平仓风险 = Σ (price - stop) × shares。"""
    total = 0.0
    for p in positions:
        px = price_of(p.code)
        if px is None or px != px:
            px = p.entry_px
        per = max(px - p.stop, 0.0)
        total += per * p.shares * (1.0 if not p.realized else 0.0)
    return total


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------


def _selfcheck() -> None:
    """最小因果性与单调性自检，无需外部数据。"""
    p = ExitPolicy(name="T", trail_atr_mult=3.0)
    pos = Position(
        code="x", entry_date="2010-01-04", entry_px=10.0, shares=1000,
        init_stop=8.0, R=2.0, policy=p,
    )
    assert pos.stop == 8.0
    ind = Indicators(atr20=0.5, ma20=10.0, ma60=9.5, ma60_slope=0.01,
                     vol_ma20=1e5, low20=9.0, close_max3=11.2)
    b = Bar("2010-01-05", 10.0, 11.35, 9.9, 11.2, 10.0, 1e5, 1e6)

    # T+1: 入场当日不计 bar，首个可卖 bar 才更新
    st = pos.update_state(b, ind)
    pos.update_stop(b, ind)
    assert st is PosState.TREND_EXTENSION, st
    # 追踪止损 = max_high(11.35) - 3*0.5 = 9.85 > 8.0
    assert abs(pos.stop - 9.85) < 1e-9, pos.stop
    assert pos.mfe_R > 0.6

    # 止损只上移：价格回落时追踪位不得下移
    b2 = Bar("2010-01-06", 11.0, 11.1, 9.2, 9.3, 11.2, 1e5, 1e6)
    pos.update_state(b2, ind)
    pos.update_stop(b2, ind)
    assert pos.stop >= 9.85 - 1e-12, pos.stop

    # 亏损时不允许把止损下移
    pos2 = Position("y", "2010-01-04", 10.0, 1000, 8.0, 2.0, p)
    bad = Indicators(0.5, 11.0, 10.8, 0.02, 1e5, 9.5, 11.0)
    pos2.update_state(Bar("2010-01-05", 10, 10.1, 9.6, 9.7, 10, 1e5, 1e6), bad)
    pos2.update_stop(Bar("2010-01-05", 10, 10.1, 9.6, 9.7, 10, 1e5, 1e6), bad)
    assert pos2.stop >= 8.0

    # 时间止损触发
    p3 = ExitPolicy(name="T3", time_stop_days=3, time_stop_min_R=0.5)
    pos3 = Position("z", "2010-01-04", 10.0, 1000, 8.0, 2.0, p3)
    flat = Indicators(0.5, 10.0, 10.0, 0.0, 1e5, 9.0, 10.0)
    for i in range(1, 4):
        pos3.update_state(Bar(f"2010-01-0{4+i}", 10, 10.05, 9.95, 10.0, 10, 1e5, 1e6), flat)
    assert pos3.state is PosState.FORCED_EXIT, pos3.state
    d = pos3.decide(Bar("2010-01-08", 10, 10.05, 9.95, 10.0, 10, 1e5, 1e6), flat)
    assert d.exit_frac == 1.0
    assert ExitReason.TIME_STOP.value in d.reasons

    # 留出窗口守卫
    try:
        assert_in_window("2021-01-04")
    except HoldoutViolation:
        pass
    else:
        raise AssertionError("留出窗口未被拦截")
    assert_in_window("2026-10-26", purpose="audit")

    # 仓位口径
    assert size_equal_weight(1_000_000, 10, 12.34) == 8100
    # 1% 风险、每股风险 2 元 → 10000/2 = 5000 股；单槽上限 100000 元未约束
    assert size_risk_budget(1_000_000, 10.0, 8.0, 0.01, 100_000) == 5000
    # 单槽上限约束：100000/10=10000 → 1000 股
    assert size_risk_budget(1_000_000, 10.0, 9.99, 0.02, 10_000) == 1000

    print("持仓状态机 selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
