"""将事前固定结构假设转成收盘后订单；不读取任何前瞻收益字段。"""
import numpy as np
import pandas as pd


ORDER_COLUMNS = ["signal_date", "code", "stop_adj", "initial_risk_adj", "rank",
                 "episode_key", "signal_close_adj", "stop_distance", "st_status_known"]


def make_orders(code, raw, f, strategy, context=None):
    context = {} if context is None else context
    if context.get("mainboard_only", True) and not code.startswith(("000", "001", "002", "003", "600", "601", "603", "605")):
        return pd.DataFrame(columns=ORDER_COLUMNS)
    raw = raw.reindex(f.index)
    s, t, p = strategy.structure_period, strategy.trigger_period, strategy.frame_period
    base = (f.d_age >= 120) & (f.d_amount20 >= 3e7) & (raw.close >= 2)
    base &= (raw.volume > 0) & (raw.amount > 0) & (f.index >= pd.Timestamp("2018-01-01"))
    known = raw.get("isST", pd.Series(np.nan, index=f.index)).notna()
    # 真实已知ST排除；未知不冒充已知非ST，明确保留缺口供审计/压力测试。
    base &= raw.get("isST", pd.Series(np.nan, index=f.index)).ne(1)
    if strategy.trigger == "channel":
        line = f.d_high.shift(1).rolling(20, min_periods=20).max()
        trigger = (f.d_close > line) & (f.d_close.shift(1) <= line)
    else:
        trigger = f[f"{t}_pullback_trigger" if strategy.trigger == "pullback" else f"{t}_breakout"].copy()
    if strategy.structure_gate:
        trigger &= f[f"{s}_up_pivot" if strategy.line == "pivot" else f"{s}_up10"]
    if strategy.frame_gate:
        trigger &= f[f"{p}_up_pivot" if strategy.line == "pivot" else f"{p}_up6"]
    if strategy.quarter_gate:
        trigger &= f.q_up6
    if strategy.volume_gate == "moderate":
        ratio = raw.volume / f.d_volume20_prior
        trigger &= ratio.between(1, 2)
    elif strategy.volume_gate == "contraction":
        trigger &= f.d_volume5_prior < f.d_volume20_prior
    if strategy.market_gate:
        if "market_breadth60" not in context:
            raise ValueError("市场过滤需要已核验的全市场时点宽度")
        trigger &= context["market_breadth60"].reindex(f.index) > .5
    if strategy.industry_gate:
        if "industry_breadth60" not in context or "industry_ret20" not in context:
            raise ValueError("行业过滤需要时点行业映射，不能用当前行业")
        trigger &= (context["industry_breadth60"].reindex(f.index) > .5)
        trigger &= (context["industry_ret20"].reindex(f.index) > 0)
    if strategy.near_line:
        # 距离用当日价格与已完成周结构线，不能用上周收盘代替实际买点位置。
        trigger &= (f.d_close / f[f"{s}_ema10"] - 1).between(0, .10)
    if strategy.stop == "fixed8":
        stop = f.d_close * .92
    elif strategy.stop == "atr2":
        stop = f.d_close - 2 * f.d_atr14
    elif strategy.stop == "trigger":
        stop = f[f"{t}_pivot_low"] * .995
    else:
        stop = f[f"{s}_pivot_low"] * .995
    risk = f.d_close - stop
    distance = risk / f.d_close
    selected = base & trigger & stop.notna() & (stop > 0) & (risk > 0)
    selected &= distance <= strategy.max_stop_distance
    # 此键使用策略自己的结构定义；分型结构不能套用均线结构阶段。
    state = f[f"{s}_up_pivot" if strategy.line == "pivot" else f"{s}_up10"]
    begins = state & ~state.shift(1, fill_value=False)
    begin_date = pd.Series(f.index.where(begins), index=f.index).ffill()
    out = pd.DataFrame({"signal_date": f.index, "code": code, "stop_adj": stop,
                        "initial_risk_adj": risk, "rank": f.d_ret60.fillna(-1),
                        "episode_key": code + "_" + begin_date.dt.strftime("%Y%m%d").fillna("无上升段"),
                        "signal_close_adj": f.d_close, "stop_distance": distance,
                        "st_status_known": known}, index=f.index)
    return out.loc[selected, ORDER_COLUMNS].reset_index(drop=True)


def close_instruction(strategy, feature_row, position, holding_days, partial_already_requested=False):
    """单只收盘指令。日期调用者传当日截面，不能喂次日数据。"""
    current = float(position["止损后复权"])
    s, t = strategy.structure_period, strategy.trigger_period
    instruction = {"code": str(position["代码"])}
    # 固定止损/ATR小周期初始止损消融保持自身支撑规则，不偷偷松回周线。
    if strategy.stop == "trigger":
        support = feature_row.get(f"{t}_pivot_low", np.nan) * .995
    elif strategy.stop in ("fixed8", "atr2"):
        support = current
    else:
        support = feature_row.get(f"{s}_pivot_low", np.nan) * .995
    if np.isfinite(support) and support > current:
        instruction["stop_adj"] = float(support)
    full_exit = False
    if strategy.exit == "time20" and holding_days >= 20:
        full_exit = True
    elif strategy.exit == "line":
        full_exit = feature_row[f"{s}_close"] < feature_row[f"{s}_ema10"]
    elif strategy.exit == "divergence":
        full_exit = bool(feature_row[f"{t}_bear_divergence"])
    elif strategy.exit == "partial":
        overbought = feature_row[f"{t}_rsi14"] > 70 and feature_row[f"{t}_line_dist"] > .10
        if overbought and not partial_already_requested:
            instruction["next_open_sell_fraction"] = .5
        if partial_already_requested or overbought:
            # 初次超买同一收盘同时发出减仓和保护止损，均最早次日生效。
            bull = feature_row.get(f"{s}_bull_low", np.nan)
            if np.isfinite(bull) and bull > max(current, instruction.get("stop_adj", current)):
                instruction["stop_adj"] = float(bull)
    if full_exit:
        instruction["next_open_sell_fraction"] = 1.0
    return instruction if len(instruction) > 1 else None
