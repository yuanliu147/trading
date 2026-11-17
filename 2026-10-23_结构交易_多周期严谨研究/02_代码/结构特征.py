"""单股票、收盘时点可知的结构特征；不包含交易执行或收益回测。

输入为无时区、日期归一化的 DatetimeIndex，calendar 必须是完整市场日历。
有效成交要求六个输入字段有限且为正、OHLC 合法；NaN/零量额行视为缺失，
不补造 K 线。EMA、分型、RSI 的根数只计有效 K 线；空的高周期也跳过。
自然周为周一至周日，季度/年度为自然季度/年度。calendar 尾部尚未跨过
自然周期末日的周期保守排除，即使尾日恰为周五，也不推断周末休市。

输出 date 索引覆盖 calendar 中不晚于 cutoff 的所有日期，包括上市前。
每层 d_/w_/m_/q_/y_ 有 BASE_COLUMNS；另外只有 d_ 层有 DAILY_COLUMNS，
合计 108 列。布尔初值 False，数值初值 NaN，d_age 上市前为 0。
日 OHLC 缺失不填充；已知日技术状态延续。高周期状态按 availability
向后 asof 对齐。EVENT_COLUMNS 只在原确认/发生日且当天有有效成交时为真，
停牌不发信号、不顺延补发；状态延续不等于当天可以交易。
"""

import numpy as np
import pandas as pd


INPUT_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
BASE_COLUMNS = (
    "open", "high", "low", "close", "ema6", "ema10", "up6", "up10",
    "pivot_high", "pivot_low", "previous_pivot_high", "previous_pivot_low",
    "higher_low", "up_pivot", "breakout", "pullback_trigger", "rsi14",
    "bear_divergence", "bull_low", "line_dist",
)
DAILY_COLUMNS = (
    "ret20", "ret60", "atr14", "amount20", "volume20_prior",
    "volume5_prior", "volatility20", "age",
)
EVENT_COLUMNS = ("breakout", "pullback_trigger", "bear_divergence")
BOOL_COLUMNS = ("up6", "up10", "higher_low", "up_pivot") + EVENT_COLUMNS
PERIODS = {"w": "W-SUN", "m": "M", "q": "Q-DEC", "y": "Y-DEC"}


def _check_dates(index: pd.DatetimeIndex, name: str) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} must be a DatetimeIndex")
    if index.tz is not None or index.hasnans or not index.equals(index.normalize()):
        raise ValueError(f"{name} must contain timezone-naive, normalized dates")
    if index.has_duplicates:
        raise ValueError(f"{name} must not contain duplicate dates")


def _rsi14(close: pd.Series) -> pd.Series:
    # 明确采用 Cutler/简单 rolling RSI，不是 Wilder：14 个有效 K 线价差，
    # 因而至少 15 根收盘；全平为 50，只有涨/跌时分别为 100/0。
    change = close.diff()
    gain = change.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-change.clip(upper=0)).rolling(14, min_periods=14).mean()
    total = gain + loss
    return (100 * gain / total).mask(total.eq(0), 50.0)


def _base_features(bars: pd.DataFrame) -> pd.DataFrame:
    result = bars.loc[:, ["open", "high", "low", "close"]].copy()
    close = bars["close"]
    for span in (6, 10):
        line = close.ewm(span=span, adjust=False, min_periods=span).mean()
        result[f"ema{span}"] = line
        result[f"up{span}"] = close.gt(line) & line.ge(line.shift(1))
    result["rsi14"] = _rsi14(close)

    for side in ("high", "low"):
        price = bars[side]
        candidate = price.shift(2)
        # 当前 t 只能确认 t-2；比较 t-4,t-3,t-1,t，不使用负向 shift。
        confirmed = pd.Series(True, index=bars.index, dtype=bool)
        for lag in (4, 3, 1, 0):
            neighbor = price.shift(lag)
            confirmed &= candidate.gt(neighbor) if side == "high" else candidate.lt(neighbor)
        events = candidate.loc[confirmed]
        result[f"pivot_{side}"] = events.reindex(bars.index).ffill()
        result[f"previous_pivot_{side}"] = events.shift(1).reindex(bars.index).ffill()
        if side == "high":
            peak_rsi = result["rsi14"].shift(2).loc[confirmed]
            divergence = events.gt(events.shift(1)) & peak_rsi.lt(peak_rsi.shift(1))
            result["bear_divergence"] = divergence.reindex(bars.index, fill_value=False)

    result["higher_low"] = result["pivot_low"].gt(result["previous_pivot_low"])
    result["up_pivot"] = (
        result["higher_low"]
        & result["pivot_high"].gt(result["previous_pivot_high"])
        & close.gt(result["pivot_low"])
    )
    # 两个收盘都与同一条“上一根已知高点”比较，禁止使用当天移动的新阈值。
    prior_high = result["pivot_high"].shift(1)
    result["breakout"] = close.gt(prior_high) & close.shift(1).le(prior_high)
    result["pullback_trigger"] = result["breakout"] & result["higher_low"]
    result["bull_low"] = bars["low"].where(close.gt(bars["open"])).ffill()
    result["line_dist"] = close / result["ema10"] - 1
    return result.loc[:, BASE_COLUMNS]


def _completed_bars(
    bars: pd.DataFrame, calendar: pd.DatetimeIndex, cutoff: pd.Timestamp, frequency: str
) -> pd.DataFrame:
    """仅日历确定 availability；绝不以股票数据末日推断退市/周期结束。"""
    if calendar.empty:
        return bars.iloc[:0]
    calendar_periods = calendar.to_period(frequency)
    availability = pd.Series(calendar, index=calendar_periods).groupby(level=0).max()
    completed = availability.index.end_time.normalize() <= calendar[-1]
    # 数据集开始于周期中途时，不把缺失的周期前半段当成完整年/月/周。
    # 保守按自然周期边界判断；日历从假期之后开始时，首周期也会被排除。
    complete_start = availability.index.start_time.normalize() >= calendar[0]
    availability = availability.loc[completed & complete_start & availability.le(cutoff)]
    aggregated = bars.groupby(bars.index.to_period(frequency)).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), amount=("amount", "sum"),
    )
    aggregated = aggregated.loc[aggregated.index.isin(availability.index)].copy()
    aggregated.index = pd.DatetimeIndex(availability.reindex(aggregated.index), name="date")
    return aggregated.sort_index()


def _align(
    features: pd.DataFrame, dates: pd.DatetimeIndex, traded: pd.Series
) -> pd.DataFrame:
    # asof 传播状态而不是逐列 ffill，保留每根新 bar 上可能出现的 NaN。
    if features.empty:
        aligned = pd.DataFrame(np.nan, index=dates, columns=features.columns)
    else:
        aligned = pd.merge_asof(
            pd.DataFrame(index=dates), features,
            left_index=True, right_index=True, direction="backward",
        )
    for column in BOOL_COLUMNS:
        aligned[column] = aligned[column].eq(True)
    for column in EVENT_COLUMNS:
        aligned[column] = features[column].reindex(dates).eq(True) & traded
    return aligned


def _daily_statistics(bars: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """收益/流动性窗口按市场交易日计数，缺失不填零、不缩成有效日窗口。"""
    daily = bars.reindex(dates)
    close = daily["close"]
    result = pd.DataFrame(index=dates)
    for window in (20, 60):
        # 端点收益，不要求中间每天成交；任一端点无成交则未知。
        result[f"ret{window}"] = close / close.shift(window) - 1
    # TR 用上一有效收盘，复牌跳空不能因昨天停牌而漏算。
    # ATR 为 14 个市场日 TR 的简单均值（价格单位），要求 14 个有效 TR。
    previous_close = bars["close"].shift(1).reindex(dates)
    tr = pd.concat(
        [daily["high"] - daily["low"],
         (daily["high"] - previous_close).abs(),
         (daily["low"] - previous_close).abs()], axis=1,
    ).max(axis=1)
    result["atr14"] = tr.rolling(14, min_periods=14).mean()
    result["amount20"] = daily["amount"].rolling(20, min_periods=20).mean()
    for window in (20, 5):
        result[f"volume{window}_prior"] = (
            daily["volume"].shift(1).rolling(window, min_periods=window).mean()
        )
    # 简单日收益的样本标准差，20 个收益，ddof=1，不年化。
    result["volatility20"] = close.pct_change(fill_method=None).rolling(
        20, min_periods=20
    ).std(ddof=1)
    result["age"] = close.notna().cumsum().astype("int64")
    return result.loc[:, DAILY_COLUMNS]


def build_features(
    bars: pd.DataFrame, calendar: pd.DatetimeIndex, cutoff=None
) -> pd.DataFrame:
    """生成截至 cutoff 收盘的 108 列特征，不读取 cutoff 之后的价格。

    cutoff 默认原 bars 的最后日期（非最后有效成交日）；显式 cutoff 可以晚于
    股票数据末日，此时只延续已知状态，不判定退市。空 bars 且无 cutoff 返回
    空行但保留完整 schema。日期可未排序，重复日期/日历外行情会明确报错。
    仅返回日历日期，不额外插入非交易日。函数不修改输入。
    """
    _check_dates(bars.index, "bars.index")
    _check_dates(calendar, "calendar")
    missing = set(INPUT_COLUMNS).difference(bars.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    calendar = calendar.sort_values().rename("date")
    if cutoff is None:
        if bars.empty:
            calendar = calendar[:0]
            cutoff = pd.Timestamp("1970-01-01")
        else:
            cutoff = bars.index.max()
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tz is not None or cutoff != cutoff.normalize():
        raise ValueError("cutoff must be a timezone-naive, normalized date")
    dates = calendar[calendar <= cutoff]
    # 先截断再过滤/聚合，未来价格及未来停牌状态均不参与计算。
    history = bars.loc[bars.index <= cutoff, INPUT_COLUMNS].sort_index().astype(float)
    # pandas 的 ns/us 日期数值相等仍可能不能 asof；保留日历精度统一索引。
    history.index = history.index.astype(calendar.dtype)
    if not history.index.isin(calendar).all():
        raise ValueError("bars dates must belong to calendar")
    valid = np.isfinite(history).all(axis=1) & history.gt(0).all(axis=1)
    valid &= history["high"].ge(history[["open", "close", "low"]].max(axis=1))
    valid &= history["low"].le(history[["open", "close"]].min(axis=1))
    history = history.loc[valid]
    traded = pd.Series(dates.isin(history.index), index=dates)
    daily_features = _align(_base_features(history), dates, traded)
    # 没有成交的日子 OHLC 留 NaN，不能让陈旧收盘伪装成当天行情。
    daily_features.loc[:, ["open", "high", "low", "close"]] = history.reindex(dates)[
        ["open", "high", "low", "close"]
    ]
    pieces = [daily_features.add_prefix("d_"), _daily_statistics(history, dates).add_prefix("d_")]
    for prefix, frequency in PERIODS.items():
        completed = _completed_bars(history, calendar, cutoff, frequency)
        pieces.append(_align(_base_features(completed), dates, traded).add_prefix(f"{prefix}_"))
    return pd.concat(pieces, axis=1)
