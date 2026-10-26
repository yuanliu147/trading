from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parent
RES = OUT / "results"
ROOT = HERE.parents[2]
PANEL = ROOT.parent / "trading_flash/2026-10-24_主板恐慌反转_年化30/01_数据/research_panel.parquet"
CORE = ROOT / "初始研究/data/work/core.parquet"
EVENTS = ROOT / "初始研究/data/snapshot/adjustment_events.parquet"
DATABASE = Path("/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb")
CONFIG = json.loads((HERE.parents[1] / "config/strategy.json").read_text())
FILES = [
    "current_stock_snapshots.csv", "current_two_cohorts.csv", "current_market.csv", "current_peer_proxy.csv",
    "signal_outcomes_audited.csv", "fixed_condition_stats.csv", "strong_reversal_date_bootstrap.json",
    "fixed_feature_stats.csv", "adjacent_signal_pairs.csv", "adjacent_signal_pair_stats.csv", "market_phase_stats.csv",
    "original_17_case_reconciliation.csv", "execution_status.csv", "analysis_summary.json",
    "strong_reversal_delay_pairs.csv", "strong_reversal_delay_stats.csv", "strong_reversal_delay_summary.json",
    "current_strong_cohort_followthrough.csv", "current_followthrough_summary.json",
]
CHECKS = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition, name: str, detail="") -> None:
    CHECKS.append({"check": name, "passed": bool(condition), "detail": str(detail)})


def close(actual, expected, tolerance=1e-7) -> bool:
    return bool(np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=0, atol=tolerance, equal_nan=True))


def dataframe(path: str, dates=()) -> pd.DataFrame:
    return pd.read_csv(RES / path, parse_dates=list(dates), low_memory=False)


def json_file(name: str):
    return json.loads((RES / name).read_text())


def execution_price(raw: float, side: str, reference: float) -> float:
    slippage = Decimal("1.001") if side == "buy" else Decimal("0.999")
    rounding = ROUND_CEILING if side == "buy" else ROUND_FLOOR
    price = (Decimal(str(raw)) * slippage).quantize(Decimal("0.01"), rounding=rounding)
    upper = (Decimal(str(reference)) * Decimal("1.1")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    lower = (Decimal(str(reference)) * Decimal("0.9")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(min(max(price, lower), upper))


def verify_ledger(original: pd.DataFrame, sessions: pd.DatetimeIndex, label: str, lag: int) -> None:
    f = original[original["closed"].eq(True)].copy()
    check(f["account_reconciled"].eq(True).all(), f"{label}:every_account_reconciled", len(f))
    check(f["entry_date"].eq(f["di"].map(lambda i: sessions[int(i) + lag])).all(), f"{label}:entry_lag")
    check((f["exit_date"] > f["entry_date"]).all(), f"{label}:T_plus_1")
    check((f["buy_shares"] > 0).all() and (f["buy_shares"] % 100 == 0).all(), f"{label}:whole_lots")
    check((f["buy_total_cost"] <= 25000 + 1e-7).all(), f"{label}:slot_budget")
    check(close(f["buy_gross"], f["buy_shares"] * f["entry_price"]), f"{label}:buy_gross")
    check(close(f["buy_total_cost"], f["buy_gross"] + f["buy_commission"] + f["buy_transfer_fee"]), f"{label}:buy_cost_identity")
    check(close(f["buy_commission"], np.maximum(5, f["buy_gross"] * .00025)), f"{label}:buy_commission")
    buy_transfer = f["buy_gross"] * np.where(f["entry_date"] >= pd.Timestamp("2022-04-29"), .00001, .00002)
    sell_transfer = f["sell_gross"] * np.where(f["exit_date"] >= pd.Timestamp("2022-04-29"), .00001, .00002)
    check(close(f["buy_transfer_fee"], buy_transfer) and close(f["sell_transfer_fee"], sell_transfer), f"{label}:transfer_fee_history")
    check(close(f["sell_gross"], f["shares"] * f["exit_price"]), f"{label}:sell_gross")
    check(close(f["shares"], f["buy_shares"] + f["bonus_shares"]), f"{label}:share_conservation")
    check(close(f["sell_commission"], np.maximum(5, f["sell_gross"] * .00025)), f"{label}:sell_commission")
    stamp = f["sell_gross"] * np.where(f["exit_date"] >= pd.Timestamp("2023-08-28"), .0005, .001)
    check(close(f["sell_stamp_duty"], stamp), f"{label}:stamp_duty_history")
    tax_rate = np.where(f["exit_date"] <= f["entry_date"] + pd.DateOffset(months=1), .2, np.where(f["exit_date"] <= f["entry_date"] + pd.DateOffset(years=1), .1, 0))
    check(close(f["dividend_tax"], f["dividend_received_gross"] * tax_rate), f"{label}:dividend_tax")
    check(close(f["dividend_received"], f["dividend_received_gross"] - f["dividend_tax"]), f"{label}:dividend_net")
    sale_net = f["sell_gross"] - f["sell_commission"] - f["sell_stamp_duty"] - f["sell_transfer_fee"]
    check(close(f["sell_net_proceeds"], sale_net), f"{label}:sell_net_proceeds")
    pnl = sale_net + f["dividend_received"] - f["buy_total_cost"]
    check(close(f["net_pnl"], pnl), f"{label}:cash_conservation")
    check(close(f["net_return"], f["net_pnl"] / f["buy_total_cost"]), f"{label}:return_recalculation")
    check(close(f["account_final_equity"], 50000 + f["net_pnl"]), f"{label}:equity_recalculation")

    con = duckdb.connect()
    entries = f[["thscode", "entry_date"]].drop_duplicates()
    con.register("entries", entries)
    entry_bars = con.execute(
        f"""SELECT e.thscode,e.entry_date,p.open,p.hfq_open,c.reference_close,c.volume
            FROM entries e JOIN read_parquet('{PANEL}') p ON p.thscode=e.thscode AND p.date=e.entry_date
            JOIN read_parquet('{CORE}') c ON c.thscode=e.thscode AND c.date=e.entry_date"""
    ).fetchdf()
    exits = f[["thscode", "exit_date"]].drop_duplicates()
    con.register("exits", exits)
    exit_bars = con.execute(
        f"""SELECT e.thscode,e.exit_date,p.open,p.high,p.low,p.close,p.hfq_low,p.hfq_high,c.reference_close,c.backward_factor,c.volume
            FROM exits e JOIN read_parquet('{PANEL}') p ON p.thscode=e.thscode AND p.date=e.exit_date
            JOIN read_parquet('{CORE}') c ON c.thscode=e.thscode AND c.date=e.exit_date"""
    ).fetchdf()
    paths = f[["thscode", "entry_date", "exit_date"]].copy()
    con.register("paths", paths)
    extrema = con.execute(
        f"""SELECT k.thscode,k.entry_date,k.exit_date,min(p.hfq_low) min_low,max(p.hfq_high) max_high
            FROM paths k JOIN read_parquet('{PANEL}') p ON p.thscode=k.thscode AND p.date BETWEEN k.entry_date AND k.exit_date
            GROUP BY k.thscode,k.entry_date,k.exit_date"""
    ).fetchdf()
    con.close()
    entry_bars = entry_bars.rename(columns={"open": "entry_bar_open", "hfq_open": "entry_bar_hfq_open", "reference_close": "entry_reference", "volume": "entry_volume"})
    exit_bars = exit_bars.rename(columns={"open": "exit_open", "high": "exit_high", "low": "exit_low", "close": "exit_close", "reference_close": "exit_reference", "volume": "exit_volume"})
    f = f.merge(entry_bars, on=["thscode", "entry_date"], how="left", validate="many_to_one").merge(exit_bars, on=["thscode", "exit_date"], how="left", validate="many_to_one").merge(extrema, on=["thscode", "entry_date", "exit_date"], how="left", validate="one_to_one")
    check(f["entry_bar_open"].notna().all() and f["exit_close"].notna().all() and f["max_high"].notna().all(), f"{label}:all_execution_bars_found")
    check(close(f["raw_entry_open"], f["entry_bar_open"]) and close(f["entry_hfq_open"], f["entry_bar_hfq_open"]), f"{label}:entry_raw_and_adjusted_price")
    check((f["entry_volume"] > 0).all() and (f["exit_volume"] > 0).all(), f"{label}:positive_execution_volume")
    expected_buy = [execution_price(row.entry_bar_open, "buy", row.entry_reference) for row in f.itertuples()]
    check(close(f["entry_price"], expected_buy), f"{label}:independent_entry_execution")
    normal_time = f["exit_reason"].eq("max_holding") & f["deferred_sell_count"].eq(0)
    expected_time = [execution_price(row.exit_close, "sell", row.exit_reference) for row in f[normal_time].itertuples()]
    check(close(f.loc[normal_time, "exit_price"], expected_time), f"{label}:time_exit_execution")
    check(f.loc[normal_time, "exit_date"].eq(f.loc[normal_time, "di"].map(lambda i: sessions[int(i) + lag + 14])).all(), f"{label}:fifteen_session_exit")
    target = f["exit_reason"].eq("profit_target") & f["deferred_sell_count"].eq(0)
    expected_target = []
    target_trigger_valid = []
    for row in f[target].itertuples():
        trigger = (Decimal(str(row.entry_hfq_open)) * Decimal("1.2") / Decimal(str(row.backward_factor))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        benchmark = row.exit_open if row.exit_open >= trigger else float(trigger)
        expected_target.append(execution_price(benchmark, "sell", row.exit_reference))
        lower = (Decimal(str(row.exit_reference)) * Decimal("0.9")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        one_price_unsellable = row.exit_high == row.exit_low and row.exit_close <= lower
        target_trigger_valid.append(row.exit_high + 1e-12 >= float(trigger) and not one_price_unsellable)
    check(close(f.loc[target, "exit_price"], expected_target), f"{label}:target_execution")
    check(all(target_trigger_valid), f"{label}:target_was_reachable")
    deferred = f["deferred_sell_count"] > 0
    expected_deferred = [execution_price(row.exit_open, "sell", row.exit_reference) for row in f[deferred].itertuples()]
    check(close(f.loc[deferred, "exit_price"], expected_deferred), f"{label}:deferred_exit_execution")
    check(close(f["mfe"], f["max_high"] / f["entry_hfq_open"] - 1) and close(f["mae"], f["min_low"] / f["entry_hfq_open"] - 1), f"{label}:independent_mae_mfe")


def verify_conditions(o: pd.DataFrame) -> None:
    closed = o["closed"].eq(True) & o["net_return"].notna()
    samples = {
        "top2": o["rank"] <= 2, "top2_episode_first": (o["rank"] <= 2) & o["episode_first"].eq(True),
        "top2_2017_2023": (o["rank"] <= 2) & o["period"].eq("2017_2023"), "top2_2024_2026": (o["rank"] <= 2) & o["period"].eq("2024_2026"),
        "top10": o["rank"] <= 10, "top10_non_overlapping": (o["rank"] <= 10) & o["non_overlapping"].eq(True),
        "top10_2017_2023": (o["rank"] <= 10) & o["period"].eq("2017_2023"), "top10_2024_2026": (o["rank"] <= 10) & o["period"].eq("2024_2026"),
        "rank3_10": o["rank"].between(3, 10), "above10": o["rank"] > 10,
        "all_signals": pd.Series(True, index=o.index), "all_non_overlapping": o["non_overlapping"].eq(True),
    }
    conditions = {
        "all": pd.Series(True, index=o.index), "deep": o["deep"].eq(True), "deep_BOR_high_close": o["strong_reversal"].eq(True),
        "deep_without_complete_reversal": o["deep"].eq(True) & ~o["strong_reversal"].eq(True),
        "deep_BOR": o["deep"].eq(True) & o["BOR"].eq(True), "deep_high_close": o["deep"].eq(True) & o["high_close"].eq(True),
        "deep_BOR_not_high_close": o["deep"].eq(True) & o["BOR"].eq(True) & ~o["high_close"].eq(True),
        "deep_not_BOR_not_high_close": o["deep"].eq(True) & ~o["BOR"].eq(True) & ~o["high_close"].eq(True),
        "raw_report17": o["unadjusted_report17"].eq(True),
    }
    errors = []
    for row in dataframe("fixed_condition_stats.csv").itertuples():
        f = o[samples[row.sample] & conditions[row.condition] & closed]
        if len(f) != row.closed_n or not close([f["net_return"].mean()], [row.mean]) or not close([(f["net_return"] > 0).mean()], [row.win_rate]) or not close([f["mae"].mean()], [row.mean_mae]):
            errors.append((row.sample, row.condition, len(f), row.closed_n))
    check(not errors, "fixed_condition_stats_rebuilt", errors[:5])
    masks = {
        "BOR": o["BOR"].eq(True), "close_location_gt_0.8": o["high_close"].eq(True), "MACD_hist_gt_0": o["macd_hist_pct"] > 0,
        "MACD_hist_improving": o["macd_hist_delta"] > 0, "amount_ratio1_20_gte_1": o["amount_ratio1_20"] >= 1,
        "amount_ratio5_20_gte_1": o["amount_ratio5_20"] >= 1, "RSI_lt_30": o["rsi14"] < 30,
        "prior_250d_gain_gt_100pct": o["return_250d"] > 1, "close_ma20_gte_minus10pct": o["close_ma20"] >= -.1,
        "above_MA5": o["close_ma5"] > 0, "MA20_slope5_positive": o["ma20_slope5"] > 0,
        "month_return3_gt_minus30pct": o["month_return3"] > -.3, "sentiment_change5_positive": o["sentiment_change5"] > 0,
        "lagged_peer_return5_gt_market5": o["peer_return5"] > o["market_return5"], "lagged_peer_breadth20_gt_market": o["peer_breadth20"] > o["breadth20"],
        "T2_stock_return_positive": o["lead2_stock_return"] > 0, "T2_close_above_signal_high": o["lead2_above_signal_high"].eq(True),
        "T1_T2_low_below_signal_low": o["lead2_below_signal_low"].eq(True), "T2_above_MA5": o["lead_close_ma5"] > 0,
        "T2_MACD_hist_positive": o["lead_macd_hist_pct"] > 0, "T1_T2_market_return_positive": o["lead2_market_return"] > 0,
    }
    sources = {
        "BOR": None, "close_location_gt_0.8": "close_location", "MACD_hist_gt_0": "macd_hist_pct", "MACD_hist_improving": "macd_hist_delta",
        "amount_ratio1_20_gte_1": "amount_ratio1_20", "amount_ratio5_20_gte_1": "amount_ratio5_20", "RSI_lt_30": "rsi14",
        "prior_250d_gain_gt_100pct": "return_250d", "close_ma20_gte_minus10pct": "close_ma20", "above_MA5": "close_ma5",
        "MA20_slope5_positive": "ma20_slope5", "month_return3_gt_minus30pct": "month_return3", "sentiment_change5_positive": "sentiment_change5",
        "lagged_peer_return5_gt_market5": "peer_return5", "lagged_peer_breadth20_gt_market": "peer_breadth20",
        "T2_stock_return_positive": "lead2_stock_return", "T2_close_above_signal_high": "lead_hfq_close",
        "T1_T2_low_below_signal_low": "lead2_below_signal_low", "T2_above_MA5": "lead_close_ma5",
        "T2_MACD_hist_positive": "lead_macd_hist_pct", "T1_T2_market_return_positive": "lead2_market_return",
    }
    feature_samples = {
        "top2_deep": (o["rank"] <= 2) & o["deep"].eq(True), "top10_deep": (o["rank"] <= 10) & o["deep"].eq(True),
        "all_deep": o["deep"].eq(True), "all_deep_non_overlapping": o["deep"].eq(True) & o["non_overlapping"].eq(True),
    }
    errors = []
    for row in dataframe("fixed_feature_stats.csv").itertuples():
        sample_name, period = next(
            (name, row.sample[len(name) + 1:])
            for name in sorted(feature_samples, key=len, reverse=True)
            if row.sample.startswith(name + "_") and row.sample[len(name) + 1:] in {"all", "2017_2023", "2024_2026"}
        )
        feature, side = row.condition.rsplit("=", 1)
        base = feature_samples[sample_name].copy()
        if period != "all":
            base &= o["period"].eq(period)
        source = sources[feature]
        if source is not None:
            base &= o[source].notna()
        if feature == "T2_close_above_signal_high":
            base &= o["has_two_sessions"].eq(True)
        base &= masks[feature] if side == "true" else ~masks[feature]
        f = o[base & closed]
        if len(f) != row.closed_n or not close([f["net_return"].mean()], [row.mean]) or not close([(f["net_return"] > 0).mean()], [row.win_rate]):
            errors.append((row.sample, row.condition, len(f), row.closed_n))
    check(not errors, "fixed_feature_stats_rebuilt", errors[:5])
    strong = o[o["strong_reversal"].eq(True) & closed]
    check(len(strong) == 40 and strong["date"].nunique() == 14, "strong_sample_and_cluster_count", f"{len(strong)}/{strong.date.nunique()}")
    check((strong["date"] == pd.Timestamp("2022-04-27")).sum() == 15, "strong_cluster_concentration_recorded")
    original = o[o["unadjusted_report17"].eq(True)]
    check(len(original) == 17 and original["closed"].all() and (original["net_return"] > 0).all(), "original_17_reconciled")
    check(o["BOR"].ne(o["raw_BOR"]).sum() == 2, "raw_vs_adjusted_BOR_differences")


def verify_delay(pairs: pd.DataFrame, sessions: pd.DatetimeIndex) -> None:
    check(len(pairs) == 50 and (pairs["baseline_closed"] & pairs["delay4_closed"]).sum() == 40, "delay_pair_counts")
    mapping = {date: i for i, date in enumerate(sessions)}
    pairs["di"] = pairs["date"].map(mapping)
    filled = pairs["delay4_closed"].eq(True)
    check(pairs.loc[filled, "delay4_entry_date"].eq(pairs.loc[filled, "di"].map(lambda i: sessions[int(i) + 4])).all(), "delay4_entry_lag")
    baseline = pairs["baseline_closed"].eq(True)
    check(pairs.loc[baseline, "baseline_entry_date"].eq(pairs.loc[baseline, "di"].map(lambda i: sessions[int(i) + 3])).all(), "delay_baseline_entry_lag")
    check(pairs.loc[pairs["next_consolidation"].isna(), "date"].eq(pd.Timestamp("2026-10-23")).all(), "current_cases_not_used_in_delay_returns")
    samples = {"all": pd.Series(True, index=pairs.index), "top2": pairs["rank"] <= 2, "top10": pairs["rank"] <= 10, "non_overlapping": pairs["non_overlapping"].eq(True)}
    groups = {
        "all": pd.Series(True, index=pairs.index), "next_consolidation": pairs["next_consolidation"].eq(True),
        "not_next_consolidation": pairs["next_consolidation"].eq(False), "next_breaks_first_low": pairs["next_breaks_first_low"].eq(True),
        "next_still_original_signal": pairs["next_still_original_signal"].eq(True), "next_not_original_signal": pairs["next_still_original_signal"].eq(False),
        "2017_2023": pairs["period"].eq("2017_2023"), "2024_2026": pairs["period"].eq("2024_2026"),
    }
    errors = []
    for row in dataframe("strong_reversal_delay_stats.csv").itertuples():
        f = pairs[samples[row.sample] & groups[row.group] & pairs["baseline_closed"].eq(True) & pairs["delay4_closed"].eq(True)]
        difference = f["delay4_net_return"] - f["baseline_net_return"]
        if len(f) != row.both_closed_n or not close([f["baseline_net_return"].mean()], [row.baseline_mean]) or not close([f["delay4_net_return"].mean()], [row.delay4_mean]) or not close([difference.mean()], [row.delay4_minus_baseline_mean]) or not close([(difference > 1e-12).mean()], [row.delay4_better_share]):
            errors.append((row.sample, row.group))
    check(not errors, "delay_stats_rebuilt", errors)


def verify_current() -> None:
    snapshot = dataframe("current_stock_snapshots.csv", ["date"])
    con = duckdb.connect(str(DATABASE), read_only=True)
    raw = con.execute("SELECT thscode,date,open,high,low,close,volume,amount FROM v_daily WHERE thscode IN ('603778.SH','605376.SH','002240.SZ') AND date IN (DATE '2026-10-23',DATE '2026-10-26')").fetchdf()
    errors = []
    for row in snapshot.itertuples():
        for field in ["open", "high", "low", "close", "rsi14", "macd_hist_pct", "close_location", "amount_ratio1_20", "atr20_pct", "candle_body", "upper_shadow", "lower_shadow"]:
            if field in {"open", "high", "low", "close"}:
                expected = raw.loc[raw["thscode"].eq(row.thscode) & raw["date"].eq(row.date), field].iloc[0]
            else:
                bars = con.execute("SELECT date,open,high,low,close FROM v_daily_hfq WHERE thscode=? AND date<=? ORDER BY date", [row.thscode, row.date]).fetchdf()
                delta = bars["close"].diff()
                gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
                loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
                previous = bars["close"].shift(1)
                if field == "rsi14":
                    expected = float((100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1])
                elif field == "macd_hist_pct":
                    dif = bars["close"].ewm(span=12, adjust=False, min_periods=26).mean() - bars["close"].ewm(span=26, adjust=False, min_periods=26).mean()
                    expected = float((2 * (dif - dif.ewm(span=9, adjust=False, min_periods=9).mean()) / bars["close"]).iloc[-1])
                elif field == "close_location":
                    expected = float(((bars["close"] - bars["low"]) / (bars["high"] - bars["low"])).iloc[-1])
                elif field == "amount_ratio1_20":
                    amounts = con.execute("SELECT amount FROM v_daily WHERE thscode=? AND date<=? ORDER BY date DESC LIMIT 20", [row.thscode, row.date]).fetchdf()["amount"]
                    expected = float(amounts.iloc[0] / amounts.mean())
                else:
                    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - previous).abs(), (bars["low"] - previous).abs()], axis=1).max(axis=1)
                    if field == "atr20_pct":
                        expected = float((tr / previous).rolling(20).mean().iloc[-1])
                    elif field == "candle_body":
                        expected = float(((bars["close"] - bars["open"]) / previous).iloc[-1])
                    elif field == "upper_shadow":
                        expected = float(((bars["high"] - bars[["open", "close"]].max(axis=1)) / previous).iloc[-1])
                    else:
                        expected = float(((bars[["open", "close"]].min(axis=1) - bars["low"]) / previous).iloc[-1])
            if not close([getattr(row, field)], [expected], 1e-9):
                errors.append((row.thscode, str(row.date), field, getattr(row, field), expected))
    check(not errors, "current_indicators_independently_rebuilt", errors[:5])
    count, symbols, zero = con.execute("SELECT count(*),count(DISTINCT thscode),count(*) FILTER(WHERE volume=0 OR amount=0) FROM v_daily WHERE date=DATE '2026-10-26'").fetchone()
    con.close()
    summary = json_file("analysis_summary.json")
    check([count, symbols, zero] == [summary["raw_database_rows_on_cutoff"], summary["raw_database_symbols_on_cutoff"], summary["raw_database_zero_activity_rows_on_cutoff"]], "raw_database_coverage")
    current = snapshot[snapshot["date"].eq(pd.Timestamp("2026-10-26"))].set_index("thscode")
    check(current.loc["603778.SH", "close"] == 7.29 and current.loc["603778.SH", "low"] == 7.19 and not current.loc["603778.SH", "BOR"], "603778_26_bar")
    previous = snapshot[snapshot["date"].eq(pd.Timestamp("2026-10-23"))].set_index("thscode")
    check(all(previous.loc[code, "BOR"] and previous.loc[code, "strong_reversal"] for code in previous.index), "23_three_reversals")
    base = float(previous.loc["603778.SH", "close"] / (1 + previous.loc["603778.SH", "return_250d"]))
    check(3.47 < base < 3.50, "r250_base_correct_not_8_80", base)
    peers = dataframe("current_peer_proxy.csv", ["date"])
    check(peers["official_industry_index"].eq(False).all(), "peer_proxy_not_official_industry")
    peer26 = peers[peers["date"].eq(pd.Timestamp("2026-10-26"))].set_index("target")
    check(peer26.loc["002240.SZ", "members"] == 47 and peer26.loc["603778.SH", "cluster_id"] < 0, "peer_membership_and_exclusion")
    follow = dataframe("current_strong_cohort_followthrough.csv")
    check(len(follow) == 10 and follow["higher_low"].sum() == 10 and follow["above_first_open"].sum() == 6 and follow["consolidation"].sum() == 4, "current_breadth_followthrough")
    check(follow["future_return_known"].eq(False).all(), "no_current_forward_returns")
    indexed = follow.set_index("thscode")
    check(bool(indexed.loc["603778.SH", "consolidation"]) and bool(indexed.loc["002240.SZ", "consolidation"]) and not bool(indexed.loc["605376.SH", "consolidation"]), "three_current_followthrough_states")
    market = dataframe("current_market.csv", ["date"]).set_index("date").sort_index()
    market_con = duckdb.connect(str(DATABASE), read_only=True)
    prefixes = ",".join(f"'{prefix}'" for prefix in CONFIG["signal"]["mainboard_prefixes"])
    advance = market_con.execute(
        f"""WITH x AS (SELECT thscode,date,close,lag(close) OVER(PARTITION BY thscode ORDER BY date) previous_close
            FROM v_daily_hfq WHERE substr(thscode,1,3) IN ({prefixes}))
            SELECT date,avg((close>previous_close)::INTEGER) advance FROM x
            WHERE date IN (DATE '2026-10-23',DATE '2026-10-26') GROUP BY date ORDER BY date"""
    ).fetchdf()
    market_con.close()
    check(close(market["advance_share"].to_numpy(), advance["advance"].to_numpy(), 1e-10), "market_advance_share")


def verify_hashes() -> None:
    frozen = json.loads((HERE.parents[1] / "audit/input_manifest.json").read_text())["input_sha256"]
    for name, path in [("panel", PANEL), ("core", CORE), ("events", EVENTS)]:
        check(sha256(path) == frozen[name], f"frozen_input_{name}")
    summary = json_file("analysis_summary.json")
    check(summary["code_hash"] == sha256(OUT / "analyze_signals.py"), "analysis_code_hash")
    check(summary["config_hash"] == sha256(OUT / "analysis_config.json"), "analysis_config_hash")
    check(summary["input_hashes"]["raw_database"] == sha256(DATABASE), "raw_database_hash")
    check(summary["remote_data_used"] is False, "local_only_data")
    follow = json_file("strong_reversal_delay_summary.json")
    check(follow["source_outcomes_sha256"] == sha256(RES / "signal_outcomes_audited.csv"), "delay_input_hash")
    check(follow["code_sha256"] == sha256(OUT / "followup_delay.py") and follow["config_sha256"] == sha256(OUT / "followup_config.json"), "delay_code_config_hash")
    current = json_file("current_followthrough_summary.json")
    check(current["source_outcomes_sha256"] == sha256(RES / "signal_outcomes_audited.csv"), "current_input_hash")
    check(current["code_sha256"] == sha256(OUT / "current_followthrough.py"), "current_code_hash")
    check(current["raw_database_sha256"] == sha256(DATABASE), "current_raw_hash")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    for name in FILES:
        check((RES / name).is_file(), f"required_file:{name}")
    if not all(item["passed"] for item in CHECKS):
        raise FileNotFoundError([item for item in CHECKS if not item["passed"]])
    before = {name: sha256(RES / name) for name in FILES}
    rerun = None
    if args.rerun:
        logs = {}
        for script in ["analyze_signals.py", "followup_delay.py", "current_followthrough.py"]:
            process = subprocess.run([sys.executable, "-B", str(OUT / script)], capture_output=True, text=True, check=False)
            (RES / f"reproduction_{script}.log").write_text(process.stdout + process.stderr)
            logs[script] = process.returncode
            if process.returncode:
                raise RuntimeError(f"{script} failed; see reproduction_{script}.log")
        after = {name: sha256(RES / name) for name in FILES}
        rerun = {"executed": True, "returncodes": logs, "identical": before == after, "changed_files": [name for name in FILES if before[name] != after[name]]}
        check(rerun["identical"], "complete_reproduction_byte_identical", rerun["changed_files"])
    sessions = pd.DatetimeIndex(duckdb.connect().execute(f"SELECT DISTINCT date FROM read_parquet('{PANEL}') ORDER BY date").fetchdf()["date"])
    o = dataframe("signal_outcomes_audited.csv", ["date", "entry_date", "exit_date"])
    check(len(o) == CONFIG["data"]["expected_signal_count"] and not o.duplicated(["thscode", "date"]).any(), "signal_count_and_unique_keys")
    check(len(o[o["rank"] <= 10]) == CONFIG["data"]["expected_candidate_rows"], "top10_count")
    check(o["closed"].sum() == 4010, "closed_trade_count")
    check(len(sessions) == CONFIG["data"]["expected_sessions"], "market_calendar")
    check(o.loc[o["date"] >= pd.Timestamp("2026-10-23"), "closed"].eq(False).all(), "no_future_trade_outcomes")
    ordered = o.sort_values(["thscode", "di"])
    gaps = ordered.groupby("thscode")["di"].diff()
    check(ordered["episode_first"].eq(gaps.isna() | (gaps > 3)).all(), "episode_de_duplication")
    con = duckdb.connect()
    signal = CONFIG["signal"]
    source = con.execute(
        f"""WITH s AS (SELECT thscode,date,amount20,drawdown250,row_number() OVER(PARTITION BY date ORDER BY drawdown250,thscode) rank
            FROM read_parquet('{PANEL}') WHERE amount20>={signal['minimum_amount20']} AND drawdown250<{signal['drawdown250_lt']}
            AND return_250d>{signal['return_250d_gt']} AND rsi14<{signal['rsi14_lt']} AND close_location>{signal['close_location_gt']}
            AND market_return20<{signal['market_return20_lt']}) SELECT thscode,date,amount20,rank FROM s"""
    ).fetchdf()
    source["date"] = pd.to_datetime(source["date"])
    comparison = o[["thscode", "date", "amount20", "rank"]].merge(source, on=["thscode", "date"], how="outer", suffixes=("_out", "_source"), indicator=True)
    check(comparison["_merge"].eq("both").all() and comparison["rank_out"].eq(comparison["rank_source"]).all() and close(comparison["amount20_out"], comparison["amount20_source"], 1e-4), "signals_rebuilt_from_frozen_panel")
    con.close()
    verify_ledger(o, sessions, "primary", 3)
    verify_conditions(o)
    pairs = dataframe("strong_reversal_delay_pairs.csv", ["date", "baseline_entry_date", "baseline_exit_date", "delay4_entry_date", "delay4_exit_date", "next_date"])
    verify_delay(pairs, sessions)
    delayed = pairs[pairs["delay4_closed"].eq(True)].copy()
    delayed.columns = [column.removeprefix("delay4_") for column in delayed.columns]
    delayed["closed"] = delayed["closed"].eq(True)
    verify_ledger(delayed, sessions, "delay4", 4)
    verify_current()
    verify_hashes()
    report = {
        "passed": all(item["passed"] for item in CHECKS), "checks": len(CHECKS), "failures": [item for item in CHECKS if not item["passed"]],
        "signals": len(o), "closed_accounts_verified": int(o["closed"].sum()), "delay_accounts_verified": int(pairs["delay4_closed"].sum()),
        "historical_return_cutoff": "2026-10-23", "current_bar_cutoff": "2026-10-26", "rerun": rerun,
        "hashes": {name: sha256(RES / name) for name in FILES},
        "code_sha256": {name: sha256(OUT / name) for name in ["analyze_signals.py", "followup_delay.py", "current_followthrough.py", "verify_analysis.py"]},
        "scope": "Local DuckDB only; standalone candidates, not a new portfolio or independent out-of-sample proof.",
    }
    (RES / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"hashes", "code_sha256"}}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
