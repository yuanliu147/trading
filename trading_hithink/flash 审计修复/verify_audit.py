from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flash_data import CORE, EVENTS, PANEL, sha256
from migrate_initial import verify as verify_migration
from run_audit import (
    ACCOUNT_FILES, AUDIT, EVIDENCE, LEGACY, MOTHER, RESULTS, SECOND,
    code_hashes, run_all, write_json,
)

FILES_WITH_DATES = {
    "orders": ["date", "signal_date"],
    "trades": ["buy_date", "sell_date", "buy_signal_date", "sell_signal_date"],
    "daily": ["date"],
    "corporate": ["date"],
    "positions": ["date", "buy_date"],
}
FILENAMES = {
    "orders": "orders.csv",
    "trades": "trades.csv",
    "daily": "daily_equity.csv",
    "corporate": "corporate_actions.csv",
    "positions": "positions_daily.csv",
}


def allclose(left: Any, right: Any, tolerance: float = 1e-7) -> bool:
    return bool(np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), atol=tolerance, rtol=1e-9, equal_nan=True))


def scheduled(schedule: list[dict[str, Any]], date: pd.Timestamp) -> float:
    result = 0.0
    for item in sorted(schedule, key=lambda value: value["start"]):
        if pd.Timestamp(item["start"]) <= date:
            result = float(item["rate"])
    return result


def tax_rate(buy_date: pd.Timestamp, sell_date: pd.Timestamp) -> float:
    if sell_date <= buy_date + pd.DateOffset(months=1):
        return 0.20
    if sell_date <= buy_date + pd.DateOffset(years=1):
        return 0.10
    return 0.0


def rounded_limit(reference: float, rate: float, upper: bool) -> float:
    multiplier = Decimal("1") + (Decimal(str(rate)) if upper else -Decimal(str(rate)))
    return float((Decimal(str(reference)) * multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def slipped_price(raw: float, slippage: float, side: str, lower: float, upper: float) -> float:
    if side == "buy":
        price = (Decimal(str(raw)) * (Decimal("1") + Decimal(str(slippage)))).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    else:
        price = (Decimal(str(raw)) * (Decimal("1") - Decimal(str(slippage)))).quantize(Decimal("0.01"), rounding=ROUND_FLOOR)
    price = min(max(price, Decimal(str(lower))), Decimal(str(upper)))
    return float(price)


def audit_account(account: Path) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: Any, detail: Any = "") -> None:
        try:
            passed = bool(condition)
        except Exception:
            passed = False
        checks[name] = {"passed": passed, "detail": str(detail)[:2000]}

    config = json.loads((account / "run_config.json").read_text())
    metrics = json.loads((account / "metrics.json").read_text())
    expected_manifest = json.loads((account / "manifest.json").read_text())
    for name in ACCOUNT_FILES:
        path = account / name
        check(f"manifest_{name}", path.is_file() and name in expected_manifest and sha256(path) == expected_manifest[name]["sha256"])
    if not all(item["passed"] for key, item in checks.items() if key.startswith("manifest_")):
        return {"account": account.name, "passed": False, "checks": checks}
    frames = {}
    for name, date_columns in FILES_WITH_DATES.items():
        frames[name] = pd.read_csv(account / FILENAMES[name], parse_dates=date_columns)
    orders = frames["orders"]
    trades = frames["trades"]
    daily = frames["daily"]
    corporate = frames["corporate"]
    positions = frames["positions"]
    execution = config["execution"]
    capital = float(execution["initial_capital"])
    filled = orders[orders["status"] == "filled"].copy()
    buys = filled[filled["side"] == "buy"].copy()
    sells = filled[filled["side"] == "sell"].copy()
    check("unique_order_ids", orders["order_id"].is_unique)
    check("unique_trade_ids", trades["trade_id"].is_unique)
    check("unique_daily_dates", daily["date"].is_unique and daily["date"].is_monotonic_increasing)
    check("unique_daily_positions", not positions.duplicated(["date", "thscode"]).any())
    check("nonnegative_cash", (daily["cash"] >= -1e-8).all(), daily["cash"].min())
    check("whole_lot_buys", (buys["shares"] > 0).all() and (buys["shares"] % 100 == 0).all())
    check("positive_sales", (sells["shares"] > 0).all())
    check("t_plus_one", (trades["sell_date"] > trades["buy_date"]).all())
    check("buy_signal_before_execution", (buys["signal_date"] < buys["date"]).all())
    check("sell_signal_before_execution", (sells["signal_date"].isna() | (sells["signal_date"] < sells["date"])).all())
    check("buy_fee_budget", (-buys["cash_delta"] <= buys["order_budget"] + 1e-7).all() and (-buys["cash_delta"] <= buys["slot_budget"] + 1e-7).all())
    participation_cap = execution.get("max_participation")
    check("participation_cap", participation_cap is None or (buys["gross"] <= buys["amount20"] * participation_cap + 1e-7).all())
    check("buy_cash_delta", allclose(buys["cash_delta"], -(buys["gross"] + buys["commission"] + buys["transfer_fee"])))
    check("sell_cash_delta", allclose(sells["cash_delta"], sells["gross"] - sells["commission"] - sells["stamp_duty"] - sells["transfer_fee"] + sells["dividend_received"]))
    check("sell_dividend_net", allclose(sells["dividend_received"], sells["dividend_receivable"] - sells["dividend_tax"]))
    check("gross_values", allclose(buys["gross"], buys["shares"] * buys["price"]) and allclose(sells["gross"], sells["shares"] * sells["price"]))
    expected_buy_commission = np.maximum(execution["minimum_commission"], buys["gross"] * execution["commission_rate"])
    expected_sell_commission = np.maximum(execution["minimum_commission"], sells["gross"] * execution["commission_rate"])
    check("commission_schedule", allclose(buys["commission"], expected_buy_commission) and allclose(sells["commission"], expected_sell_commission))
    expected_buy_transfer = [gross * scheduled(execution["transfer_fee_both_sides"], date) for gross, date in zip(buys["gross"], buys["date"])]
    expected_sell_transfer = [gross * scheduled(execution["transfer_fee_both_sides"], date) for gross, date in zip(sells["gross"], sells["date"])]
    expected_stamp = [gross * scheduled(execution["stamp_duty_sell"], date) for gross, date in zip(sells["gross"], sells["date"])]
    check("historical_fee_schedules", allclose(buys["transfer_fee"], expected_buy_transfer) and allclose(sells["transfer_fee"], expected_sell_transfer) and allclose(sells["stamp_duty"], expected_stamp))
    expected_buy_price = [slipped_price(raw, execution["slippage"], "buy", lower, upper) for raw, lower, upper in zip(buys["raw_open"], buys["limit_down"], buys["limit_up"])]
    expected_sell_price = [slipped_price(raw, execution["slippage"], "sell", lower, upper) for raw, lower, upper in zip(sells["benchmark_price"], sells["limit_down"], sells["limit_up"])]
    check("execution_prices", allclose(buys["price"], expected_buy_price) and allclose(sells["price"], expected_sell_price))
    check("buy_slippage", allclose(buys["slippage_cost"], (buys["price"] - buys["raw_open"]) * buys["shares"]))
    check("sell_slippage", allclose(sells["slippage_cost"], (sells["benchmark_price"] - sells["price"]) * sells["shares"]))
    check("price_tick", allclose((filled["price"] * 100).round(), filled["price"] * 100))
    check("opening_limit_up_rejection", (buys["raw_open"] < buys["limit_up"] + 1e-9).all())
    check("sell_links", set(sells["position_id"]).issubset(set(buys["order_id"])))
    check("trade_order_links", set(trades["buy_order_id"]).issubset(set(buys["order_id"])) and set(trades["sell_order_id"]).issubset(set(sells["order_id"])))
    check("trade_share_conservation", (trades["sell_shares"] == trades["buy_shares"] + trades["bonus_shares"]).all())
    expected_tax = [gross * tax_rate(buy, sell) for gross, buy, sell in zip(trades["dividend_received_gross"], trades["buy_date"], trades["sell_date"])]
    check("dividend_tax_schedule", allclose(trades["dividend_tax"], expected_tax))
    expected_pnl = trades["sell_net_proceeds"] + trades["dividend_received"] - trades["buy_total_cost"]
    check("trade_pnl", allclose(trades["net_pnl"], expected_pnl) and allclose(trades["return"], trades["net_pnl"] / trades["buy_total_cost"]))
    if not trades.empty:
        linked_buy = trades.merge(buys, left_on="buy_order_id", right_on="order_id", suffixes=("", "_buy_order"))
        linked_sell = trades.merge(sells, left_on="sell_order_id", right_on="order_id", suffixes=("", "_sell_order"))
        check("trade_buy_order_values", allclose(linked_buy["buy_total_cost"], -linked_buy["cash_delta"]) and (linked_buy["buy_date"] == linked_buy["date"]).all())
        check("trade_sell_order_values", allclose(linked_sell["sell_net_proceeds"], linked_sell["gross"] - linked_sell["commission"] - linked_sell["stamp_duty"] - linked_sell["transfer_fee"]) and (linked_sell["sell_date"] == linked_sell["date"]).all())
    if not corporate.empty:
        check("corporate_bonus_conservation", allclose(corporate["shares_after"], corporate["shares_before"] + corporate["bonus_shares_added"]))
        check("corporate_availability", allclose(corporate["available_shares_after"], corporate["available_shares_before"]))
        check("corporate_nonnegative", (corporate["dividend_added"] >= 0).all() and (corporate["bonus_shares_added"] >= 0).all())
    state: dict[str, dict[str, Any]] = {}
    position_errors = []
    expected_cash = capital
    expected_receivable = 0.0
    previous_equity = capital
    running_peak = capital
    expected_daily_return = []
    expected_drawdown = []
    corporate_groups = {date: group for date, group in corporate.groupby("date")} if not corporate.empty else {}
    filled_groups = {date: group for date, group in filled.groupby("date")} if not filled.empty else {}
    position_groups = {date: group for date, group in positions.groupby("date")} if not positions.empty else {}
    for day in daily.itertuples():
        date = day.date
        for value in state.values():
            value["available"] = value["shares"]
        day_cash_delta = 0.0
        day_buy_delta = 0.0
        day_open_sale = 0.0
        day_sell_dividend = 0.0
        for event in corporate_groups.get(date, pd.DataFrame()).itertuples():
            if event.thscode not in state or state[event.thscode]["position_id"] != event.position_id:
                position_errors.append(f"corporate_action_without_matching_position:{date}:{event.thscode}")
                continue
            state[event.thscode]["shares"] += int(event.bonus_shares_added)
            expected_receivable += float(event.dividend_added)
        for order in filled_groups.get(date, pd.DataFrame()).itertuples():
            day_cash_delta += float(order.cash_delta)
            if order.side == "buy":
                day_buy_delta += float(order.cash_delta)
                if order.thscode in state:
                    position_errors.append(f"duplicate_position:{date}:{order.thscode}")
                state[order.thscode] = {"shares": int(order.shares), "available": 0, "position_id": order.order_id}
            else:
                day_sell_dividend += float(order.dividend_receivable)
                if order.execution_timing == "open":
                    day_open_sale += float(order.cash_delta)
                if order.thscode not in state or state[order.thscode]["position_id"] != order.position_id:
                    position_errors.append(f"sell_without_matching_position:{date}:{order.thscode}")
                    continue
                state[order.thscode]["shares"] -= int(order.shares)
                state[order.thscode]["available"] -= int(order.shares)
                if state[order.thscode]["shares"] == 0:
                    del state[order.thscode]
        expected_cash += day_cash_delta
        expected_receivable -= day_sell_dividend
        actual = position_groups.get(date, pd.DataFrame(columns=positions.columns))
        if set(actual["thscode"]) != set(state):
            position_errors.append(f"position_set:{date}")
        for row in actual.itertuples():
            expected = state.get(row.thscode)
            if expected is None:
                continue
            if row.shares != expected["shares"] or row.available_shares != expected["available"] or row.position_id != expected["position_id"]:
                position_errors.append(f"position_state:{date}:{row.thscode}")
            if not np.isfinite(row.last_close) or row.last_close <= 0:
                position_errors.append(f"invalid_stale_price:{date}:{row.thscode}")
        market_value = float(actual["market_value"].sum()) if not actual.empty else 0.0
        stale_count = int(actual["stale_valuation"].sum()) if not actual.empty else 0
        expected_equity = expected_cash + market_value + expected_receivable
        if not np.isclose(day.cash, expected_cash, atol=1e-7):
            position_errors.append(f"cash:{date}")
        if not np.isclose(day.dividend_receivable, expected_receivable, atol=1e-7):
            position_errors.append(f"receivable:{date}")
        if not np.isclose(day.equity, expected_equity, atol=1e-7):
            position_errors.append(f"equity:{date}")
        if not np.isclose(day.market_value, market_value, atol=1e-7) or day.positions != len(state) or day.shares != sum(value["shares"] for value in state.values()):
            position_errors.append(f"market_value_or_count:{date}")
        if not np.isclose(day.cash_start, expected_cash - day_cash_delta, atol=1e-7):
            position_errors.append(f"cash_start:{date}")
        if not np.isclose(day.buy_budget_start, day.cash_start, atol=1e-7) or not np.isclose(day.buy_budget_end, day.cash_start + day_buy_delta, atol=1e-7):
            position_errors.append(f"buy_budget:{date}")
        if not np.isclose(day.same_open_sale_proceeds, day_open_sale, atol=1e-7):
            position_errors.append(f"same_open_proceeds:{date}")
        if day.stale_valuation_positions != stale_count:
            position_errors.append(f"stale_count:{date}")
        expected_daily_return.append(expected_equity / previous_equity - 1)
        running_peak = max(running_peak, expected_equity)
        expected_drawdown.append(expected_equity / running_peak - 1)
        previous_equity = expected_equity
    check("independent_daily_ledger", not position_errors, position_errors[:8])
    check("independent_daily_returns", allclose(daily["daily_return"], expected_daily_return))
    check("independent_drawdown", allclose(daily["drawdown"], expected_drawdown))
    expected_open = positions[positions["date"] == daily["date"].iloc[-1]]
    saved_open = pd.read_csv(account / "open_positions.csv")
    check("open_positions_not_trades", len(saved_open) == len(expected_open) and len(trades) + len(saved_open) == len(buys))
    check("open_positions_valuation", allclose(saved_open["market_value"], expected_open["market_value"]) and allclose(saved_open["dividend_receivable"], expected_open["dividend_receivable"]))
    equity = daily["equity"].to_numpy(dtype=float)
    elapsed = (daily["date"].iloc[-1] - daily["date"].iloc[0]).days / 365.25
    cagr = (equity[-1] / capital) ** (1 / elapsed) - 1
    peaks = np.maximum.accumulate(np.concatenate([[capital], equity]))
    max_drawdown = float((np.concatenate([[capital], equity]) / peaks - 1).min())
    check("metric_final_equity", np.isclose(metrics["final_equity"], equity[-1], atol=1e-7))
    check("metric_cagr", np.isclose(metrics["cagr"], cagr, atol=1e-11))
    check("metric_max_drawdown", np.isclose(metrics["max_drawdown"], max_drawdown, atol=1e-11))
    check("metric_trade_counts", metrics["trade_count"] == len(trades) and metrics["ending_open_positions"] == len(saved_open) and metrics["closed_plus_open"] == len(buys))
    year_end = daily.set_index("date")["equity"].resample("YE").last()
    annual = year_end.pct_change()
    annual.iloc[0] = year_end.iloc[0] / capital - 1
    saved_annual = pd.read_csv(account / "annual.csv")
    check("annual_returns", saved_annual["year"].tolist() == annual.index.year.tolist() and allclose(saved_annual["return"], annual.to_numpy()))
    return {
        "account": account.name,
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "cagr": float(cagr),
        "max_drawdown": max_drawdown,
        "final_equity": float(equity[-1]),
        "trade_count": int(len(trades)),
        "open_positions": int(len(saved_open)),
        "minimum_cash": float(daily["cash"].min()),
        "deferred_sells": int(((orders["side"] == "sell") & (orders["status"] == "deferred")).sum()),
        "failed_checks": [name for name, item in checks.items() if not item["passed"]],
    }


def static_verification() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    run_manifest = json.loads((EVIDENCE / "run_manifest.json").read_text())
    inputs = json.loads((EVIDENCE / "input_manifest.json").read_text())
    accounts = sorted(path for path in (RESULTS / "accounts").iterdir() if path.is_dir())
    checks["run_completed"] = run_manifest["status"] == "completed"
    checks["scenario_list"] = set(run_manifest["scenarios"]) == {path.name for path in accounts} and run_manifest["scenario_count"] == len(accounts)
    checks["strategy_frozen"] = run_manifest["strategy_sha256"] == sha256(AUDIT / "config" / "strategy.json")
    checks["code_not_changed"] = run_manifest["code_sha256"] == inputs["code_sha256"] == code_hashes()
    checks["input_not_changed"] = all(sha256(path) == inputs["input_sha256"][name] for name, path in [("panel", PANEL), ("core", CORE), ("events", EVENTS)])
    checks["no_remote_or_third_party"] = inputs["remote_data_used"] is False and inputs["third_party_data_used"] is False
    legacy_manifest = json.loads((LEGACY / "manifest.json").read_text())
    checks["legacy_preserved"] = all(sha256(LEGACY / name) == record["sha256"] for name, record in legacy_manifest.items())
    baseline = json.loads((EVIDENCE / "baseline_evidence.json").read_text())
    checks["baseline_defects_recorded"] = baseline["negative_cash_days"] == 14 and baseline["minimum_cash"] < 0 and baseline["up_limit_05_field"]["up_limit_05_type"] == "DOUBLE"
    signal = json.loads((EVIDENCE / "signal_check.json").read_text())
    checks["signals_frozen"] = signal["signal_count"] == 4195 and signal["candidate_rows"] == 1526 and signal["last_day_top2"] == ["603778.SH", "605376.SH"]
    checks["candidate_file_hash"] = sha256(RESULTS / "entry_candidates.csv") == signal["candidate_sha256"]
    entrypoints = [
        MOTHER / "02_代码" / "verify_delivery.py",
        MOTHER / "02_代码" / "verify_actual_fixed.py",
        SECOND / "02_代码" / "verify_delivery.py",
    ]
    checks["legacy_entrypoints_redirected"] = all(
        "runpy.run_path" in path.read_text() and "verify_audit.py" in path.read_text()
        for path in entrypoints
    )
    legacy_signals = pd.read_csv(MOTHER / "03_结果" / "当前信号_2026-10-23.csv")
    filtered_signals = pd.read_csv(MOTHER / "03_结果" / "当前信号_2026-10-23_exclude5pct.csv")
    checks["live_signal_modes"] = (
        len(legacy_signals) == 123 and len(filtered_signals) == 122
        and legacy_signals["thscode"].head(2).tolist() == filtered_signals["thscode"].head(2).tolist()
    )
    migration_error = ""
    try:
        verify_migration()
    except Exception as exc:
        migration_error = str(exc)
    checks["migration_integrity"] = not migration_error
    account_results = [audit_account(path) for path in accounts]
    return {
        "passed": all(checks.values()) and all(item["passed"] for item in account_results),
        "static_checks": checks,
        "migration_error": migration_error,
        "accounts": account_results,
        "evaluation_status": "historical_repair_not_independent_holdout",
        "remaining_limits": [
            "No untouched out-of-sample interval remains.",
            "Current-listed universe only; delisted stocks are not reconstructed as requested.",
            "Historical ST is unavailable; 5% and 10% are execution scenarios, not reconstructed ST labels.",
            "Daily OHLC cannot prove intraday queue fills; next-open take-profit is a separate stress scenario.",
        ],
    }


def tracked_hashes() -> dict[str, str]:
    result = {}
    for account in sorted((RESULTS / "accounts").iterdir()):
        if not account.is_dir():
            continue
        for name in ACCOUNT_FILES + ["manifest.json"]:
            path = account / name
            result[str(path.relative_to(AUDIT))] = sha256(path)
    for name in ["summary.csv", "annual_returns.csv", "drawdown_spells.csv", "original_vs_repaired_100k.csv", "entry_candidates.csv"]:
        path = RESULTS / name
        result[str(path.relative_to(AUDIT))] = sha256(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    rerun = None
    if args.rerun:
        before = tracked_hashes()
        try:
            run_all()
        except Exception as exc:
            write_json(EVIDENCE / "verification.json", {"passed": False, "stage": "rerun", "error": str(exc)})
            raise
        after = tracked_hashes()
        changed = [name for name in before if before[name] != after.get(name)]
        rerun = {"identical": before == after, "changed_files": changed}
        if before != after:
            write_json(EVIDENCE / "verification.json", {"passed": False, "stage": "determinism", "rerun": rerun})
            raise AssertionError(f"Audit rerun was not deterministic: {changed}")
    try:
        result = static_verification()
    except Exception as exc:
        write_json(EVIDENCE / "verification.json", {"passed": False, "stage": "static_verification", "error": str(exc), "rerun": rerun})
        raise
    result["rerun"] = rerun
    write_json(EVIDENCE / "verification.json", result)
    print(json.dumps({
        "passed": result["passed"],
        "static_checks": result["static_checks"],
        "accounts": [{key: value for key, value in item.items() if key != "checks"} for item in result["accounts"]],
        "rerun": rerun,
    }, ensure_ascii=False, indent=2, default=str))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
