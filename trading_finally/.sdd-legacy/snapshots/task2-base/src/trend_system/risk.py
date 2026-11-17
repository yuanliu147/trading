"""Non-gate account risk diagnostics for the corrected frozen baseline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _number(series, statistic="max"):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(getattr(values, statistic)())


def _records(frame):
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict("records")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _empty(columns):
    return pd.DataFrame(columns=columns)


def _entry_risk(trades, signals, account):
    columns = [
        "order_id", "signal_date", "entry_date", "code", "quantity", "entry_price",
        "initial_exit_20d", "signed_risk_per_share", "initial_risk_amount",
        "account_equity_at_signal", "risk_account_share",
    ]
    if trades.empty:
        return _empty(columns)
    entries = trades.loc[trades["side"] == "buy"].copy()
    if entries.empty:
        return _empty(columns)
    levels = signals[["date", "code", "channel_low_120"]].rename(
        columns={"date": "signal_date", "channel_low_120": "initial_exit_20d"}
    )
    entries = entries.merge(
        levels, on=["signal_date", "code"], how="left", validate="many_to_one"
    )
    equity = account[["date", "equity"]].rename(
        columns={"date": "signal_date", "equity": "account_equity_at_signal"}
    )
    entries = entries.merge(
        equity, on="signal_date", how="left", validate="many_to_one"
    )
    if entries[["initial_exit_20d", "account_equity_at_signal"]].isna().any().any():
        raise ValueError("Every entry requires its signal-date 20-day exit level and account equity")
    entries["signed_risk_per_share"] = entries["price"] - entries["initial_exit_20d"]
    entries["initial_risk_amount"] = (
        entries["signed_risk_per_share"].clip(lower=0) * entries["quantity"]
    )
    entries["risk_account_share"] = (
        entries["initial_risk_amount"] / entries["account_equity_at_signal"]
    )
    result = entries.rename(columns={"date": "entry_date", "price": "entry_price"})[
        columns
    ].sort_values(["entry_date", "code"], kind="stable").reset_index(drop=True)
    return result


def _commission_diagnostics(trades, minimum_commission):
    columns = ["order_id", "date", "code", "side", "amount", "commission", "commission_to_amount", "minimum_applied"]
    if trades.empty:
        return _empty(columns), {
            "minimum_applied_trades": 0,
            "minimum_applied_share": 0.0,
            "mean_commission_to_amount": 0.0,
            "max_commission_to_amount": 0.0,
        }
    result = trades.copy()
    result["commission_to_amount"] = result["commission"] / result["amount"]
    result["minimum_applied"] = np.isclose(
        result["commission"], minimum_commission, atol=0.005
    )
    result = result[columns]
    minimum = result.loc[result["minimum_applied"]]
    return result, {
        "minimum_applied_trades": int(len(minimum)),
        "minimum_applied_share": float(len(minimum) / len(result)),
        "mean_commission_to_amount": _number(minimum["commission_to_amount"], "mean"),
        "median_commission_to_amount": _number(minimum["commission_to_amount"], "median"),
        "max_commission_to_amount": _number(minimum["commission_to_amount"]),
    }


def _holding_correlations(positions, panel):
    columns = ["code_a", "code_b", "overlapping_days", "return_correlation"]
    if positions.empty:
        return _empty(columns), {"pair_count": 0, "mean_correlation": 0.0, "max_correlation": 0.0}
    held = positions[["date", "code", "entry_date"]].drop_duplicates().merge(
        panel[["date", "code", "signal_close"]],
        on=["date", "code"],
        how="left",
        validate="one_to_one",
    )
    held = held.sort_values(["code", "entry_date", "date"], kind="stable")
    held["stock_return"] = held.groupby(["code", "entry_date"])["signal_close"].pct_change(fill_method=None)
    pivot = held.pivot(index="date", columns="code", values="stock_return")
    correlations = pivot.corr(min_periods=2)
    rows = []
    codes = list(correlations.columns)
    for index, code_a in enumerate(codes):
        for code_b in codes[index + 1:]:
            value = correlations.loc[code_a, code_b]
            overlap = int(pivot[[code_a, code_b]].dropna().shape[0])
            if pd.notna(value):
                rows.append({
                    "code_a": str(code_a),
                    "code_b": str(code_b),
                    "overlapping_days": overlap,
                    "return_correlation": float(value),
                })
    result = pd.DataFrame(rows, columns=columns)
    summary = {
        "pair_count": int(len(result)),
        "mean_correlation": _number(result.get("return_correlation", pd.Series(dtype=float)), "mean"),
        "median_correlation": _number(result.get("return_correlation", pd.Series(dtype=float)), "median"),
        "max_correlation": _number(result.get("return_correlation", pd.Series(dtype=float))),
        "method": "按(code,entry_date)隔离持仓episode，使用后复权signal_close在episode内计算日收益；按股票对重叠交易日计算Pearson相关，至少2个重叠收益",
    }
    return result, summary


def _same_day_exits(round_trips):
    columns = ["exit_date", "exit_count", "losing_count", "exit_amount", "net_pnl"]
    if round_trips.empty:
        return _empty(columns), {"max_exits": 0, "multiple_exit_days": 0, "worst_day": None, "worst_net_pnl": 0.0}
    grouped = round_trips.groupby("exit_date", sort=True)
    result = grouped.agg(
        exit_count=("code", "size"),
        losing_count=("net_pnl", lambda values: int(values.lt(0).sum())),
        exit_amount=("exit_amount", "sum"),
        net_pnl=("net_pnl", "sum"),
    ).reset_index()
    worst = result.loc[result["net_pnl"].idxmin()]
    return result, {
        "max_exits": int(result["exit_count"].max()),
        "multiple_exit_days": int(result["exit_count"].gt(1).sum()),
        "worst_day": str(worst["exit_date"]),
        "worst_net_pnl": float(worst["net_pnl"]),
    }


def _exit_execution(trades, orders, panel):
    gap_columns = ["order_id", "date", "code", "quantity", "previous_close", "execution_open", "overnight_gap_extra_loss"]
    delay_columns = ["order_id", "code", "first_limit_down_date", "filled_quantity", "limit_down_delay_days", "delay_extra_loss"]
    bars = panel[["date", "code", "open", "prev_close"]].rename(
        columns={"prev_close": "previous_close"}
    )
    sells = trades.loc[trades["side"] == "sell"].copy() if not trades.empty else trades.copy()
    if sells.empty:
        gaps = _empty(gap_columns)
    else:
        gaps = sells.merge(
            bars[["date", "code", "open", "previous_close"]],
            on=["date", "code"],
            how="left",
            validate="many_to_one",
        ).rename(columns={"open": "execution_open"})
        gaps["overnight_gap_extra_loss"] = (
            (gaps["previous_close"] - gaps["execution_open"]).clip(lower=0) * gaps["quantity"]
        )
        gaps = gaps[gap_columns]

    delay_rows = []
    if not orders.empty:
        limit_down = orders.loc[
            orders["side"].eq("sell") & orders["reason"].eq("sell_limit_down")
        ]
        for order_id, group in limit_down.groupby("order_id", sort=False):
            first = group.sort_values("date", kind="stable").iloc[0]
            fills = trades.loc[trades["order_id"].eq(order_id)] if not trades.empty else trades
            quantity = float(fills["quantity"].sum()) if not fills.empty else 0.0
            first_bar = bars.loc[
                bars["date"].eq(first["date"]) & bars["code"].eq(first["code"])
            ]
            loss = None
            if not fills.empty and not first_bar.empty:
                reference = float(first_bar.iloc[0]["open"]) * quantity
                loss = float(max(reference - fills["amount"].sum(), 0.0))
            delay_rows.append({
                "order_id": order_id,
                "code": first["code"],
                "first_limit_down_date": first["date"],
                "filled_quantity": quantity,
                "limit_down_delay_days": int(group["date"].nunique()),
                "delay_extra_loss": loss,
            })
    delays = pd.DataFrame(delay_rows, columns=delay_columns)
    resolved_losses = pd.to_numeric(delays.get("delay_extra_loss", pd.Series(dtype=float)), errors="coerce").dropna()
    summary = {
        "overnight_gap_extra_loss": _number(gaps.get("overnight_gap_extra_loss", pd.Series(dtype=float)), "sum"),
        "overnight_gap_affected_sells": int(gaps.get("overnight_gap_extra_loss", pd.Series(dtype=float)).gt(0).sum()),
        "limit_down_orders": int(len(delays)),
        "resolved_limit_down_orders": int(resolved_losses.count()),
        "unresolved_limit_down_orders": int(delays["delay_extra_loss"].isna().sum()) if not delays.empty else 0,
        "max_limit_down_delay_days": int(delays["limit_down_delay_days"].max()) if not delays.empty else 0,
        "limit_down_delay_extra_loss": float(resolved_losses.sum()),
        "definitions": "跳空损失=max(前一实际收盘-实际退出日开盘,0)×卖出数量；跌停顺延损失=max(首次跌停开盘×最终卖出数量-实际卖出成交额,0)，含其后价格变化与滑点，不作因果分解",
    }
    return gaps, delays, summary


def _stale_stress(account, positions):
    daily = account[["date", "equity"]].copy().sort_values("date", kind="stable")
    if positions.empty:
        stale = pd.DataFrame(columns=["date", "stale_market_value"])
    else:
        stale = positions.loc[positions["stale_valuation"].astype(bool)].groupby("date")["market_value"].sum().rename("stale_market_value").reset_index()
    daily = daily.merge(stale, on="date", how="left", validate="one_to_one")
    daily["stale_market_value"] = daily["stale_market_value"].fillna(0.0)
    result = {}
    for label, discount in (("10pct", 0.10), ("20pct", 0.20)):
        column = f"stressed_equity_{label}"
        daily[column] = daily["equity"] - discount * daily["stale_market_value"]
        drawdown = daily[column] / daily[column].cummax() - 1
        result[label] = {
            "discount": discount,
            "final_equity": float(daily[column].iloc[-1]),
            "max_drawdown": float(-drawdown.min()),
        }
    result["stale_position_days"] = int(daily["stale_market_value"].gt(0).sum())
    result["max_stale_market_value"] = float(daily["stale_market_value"].max())
    result["accounting_only"] = True
    return daily, result


def _markdown(risk):
    lines = [
        "# 修正后基线非门槛风险诊断",
        "",
        "仅披露账户风险，不生成订单、不改变策略、不参与机械门槛。停牌压力只对陈旧估值持仓市值作额外会计折价。",
        "",
        "## 单笔初始风险与执行额外损失",
        "",
        f"- 入场笔数：{risk['entry_risk']['entries']}；初始风险金额均值/最大值：{risk['entry_risk']['mean_initial_risk_amount']:.6f}/{risk['entry_risk']['max_initial_risk_amount']:.6f} 元。",
        f"- 单笔初始风险占信号日账户权益均值/最大值：{risk['entry_risk']['mean_risk_account_share']:.6f}/{risk['entry_risk']['max_risk_account_share']:.6f}。",
        f"- 退出隔夜跳空额外损失合计：{risk['exit_execution']['overnight_gap_extra_loss']:.6f} 元；跌停顺延额外损失合计：{risk['exit_execution']['limit_down_delay_extra_loss']:.6f} 元。",
        "",
        "## 持仓相关性、集中退出与最低佣金",
        "",
        f"- 可计算持仓收益相关股票对：{risk['holding_return_correlation']['pair_count']}；均值/最大值：{risk['holding_return_correlation']['mean_correlation']:.6f}/{risk['holding_return_correlation']['max_correlation']:.6f}。",
        f"- 单日最多退出：{risk['same_day_exits']['max_exits']} 笔；同日多笔退出天数：{risk['same_day_exits']['multiple_exit_days']}。",
        f"- 触发最低佣金成交：{risk['minimum_commission']['minimum_applied_trades']} 笔；其佣金/成交额均值/最大值：{risk['minimum_commission']['mean_commission_to_amount']:.6f}/{risk['minimum_commission']['max_commission_to_amount']:.6f}。",
        "",
        "## 停牌陈旧估值会计压力",
        "",
        "| 额外折价 | 期末权益 | 最大回撤 |",
        "| --- | ---: | ---: |",
    ]
    for label in ("10pct", "20pct"):
        scenario = risk["stale_valuation_stress"][label]
        lines.append(f"| {scenario['discount']:.0%} | {scenario['final_equity']:.6f} | {scenario['max_drawdown']:.6f} |")
    lines.extend(["", "逐笔和逐日明细见同目录CSV及risk_report.json；跳空和跌停顺延指标是描述性口径，不是独立因果归因。", ""])
    return "\n".join(lines)


def build_risk_report(panel, signals, backtest, config, output_dir):
    """Calculate and persist diagnostics without creating orders or gate inputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = panel.copy()
    signals = signals.copy()
    panel["date"] = panel["date"].astype(str)
    signals["date"] = signals["date"].astype(str)
    account = backtest.daily_account.copy()
    account["date"] = account["date"].astype(str)
    trades = backtest.trades.copy()
    orders = backtest.orders.copy()
    positions = backtest.daily_positions.copy()
    round_trips = backtest.round_trips.copy()
    for frame in (trades, orders, positions, round_trips):
        if "date" in frame:
            frame["date"] = frame["date"].astype(str)
        if "signal_date" in frame:
            frame["signal_date"] = frame["signal_date"].astype(str)

    entry_risk = _entry_risk(trades, signals, account)
    commissions, commission_summary = _commission_diagnostics(trades, config.min_commission)
    correlations, correlation_summary = _holding_correlations(positions, panel)
    same_day, same_day_summary = _same_day_exits(round_trips)
    gaps, delays, execution_summary = _exit_execution(trades, orders, panel)
    stale_daily, stale_summary = _stale_stress(account, positions)
    tables = {
        "entry_risk": entry_risk,
        "minimum_commission": commissions,
        "holding_return_correlations": correlations,
        "same_day_exits": same_day,
        "exit_overnight_gaps": gaps,
        "limit_down_delays": delays,
        "stale_valuation_stress_daily": stale_daily,
    }
    risk = {
        "diagnostic_only": True,
        "orders_generated": False,
        "included_in_mechanical_gates": False,
        "entry_risk": {
            "entries": int(len(entry_risk)),
            "nonpositive_signed_risk_entries": int(entry_risk.get("signed_risk_per_share", pd.Series(dtype=float)).le(0).sum()),
            "mean_initial_risk_amount": _number(entry_risk.get("initial_risk_amount", pd.Series(dtype=float)), "mean"),
            "max_initial_risk_amount": _number(entry_risk.get("initial_risk_amount", pd.Series(dtype=float))),
            "mean_risk_account_share": _number(entry_risk.get("risk_account_share", pd.Series(dtype=float)), "mean"),
            "max_risk_account_share": _number(entry_risk.get("risk_account_share", pd.Series(dtype=float))),
            "definition": "max(实际入场价-信号日此前20日通道退出位,0)×买入数量；分母为信号日收盘账户权益",
        },
        "holding_return_correlation": correlation_summary,
        "same_day_exits": same_day_summary,
        "exit_execution": execution_summary,
        "minimum_commission": commission_summary,
        "stale_valuation_stress": stale_summary,
        "tables": {name: _records(frame) for name, frame in tables.items()},
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "risk_report.json").write_text(
        json.dumps(risk, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "risk_report.md").write_text(_markdown(risk), encoding="utf-8")
    return risk
