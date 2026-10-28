from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parent
RES = OUT / "results"
REPAIR = HERE.parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(REPAIR))
from flash_data import INITIAL, PANEL, sha256  # noqa: E402

CONFIG = json.loads((OUT / "analysis_config.json").read_text())
DATABASE = Path(CONFIG["data"]["database"])
TODAY = pd.Timestamp(CONFIG["data"]["cutoff"])
FIRST = pd.Timestamp(CONFIG["data"]["first_signal_date"])
SECOND = pd.Timestamp(CONFIG["data"]["second_signal_date"])
FILES = [
    "source_daily_quality.csv", "market_snapshots.csv", "current_stock_features.csv", "current_original_signals.csv",
    "cohort_23_followthrough.csv", "cohort_26_followthrough.csv", "current_peer_proxy.csv", "source_quality.json", "analysis_summary.json",
]
CHECKS = []


def check(condition: bool, name: str, detail="") -> None:
    CHECKS.append({"check": name, "passed": bool(condition), "detail": str(detail)})


def close(actual, expected, tolerance=1e-9) -> bool:
    return bool(np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=0, atol=tolerance, equal_nan=True))


def verify_target(con: duckdb.DuckDBPyConnection, features: pd.DataFrame, summary: dict) -> None:
    raw = con.execute(
        """SELECT d.date,d.open,d.high,d.low,d.close,d.volume,d.amount,
                  h.open hfq_open,h.high hfq_high,h.low hfq_low,h.close hfq_close,f.backward_factor
           FROM v_daily d JOIN v_daily_hfq h USING(thscode,date) JOIN calc_adjust_factor_daily f USING(thscode,date)
           WHERE d.thscode='603778.SH' ORDER BY d.date"""
    ).fetchdf()
    raw["date"] = pd.to_datetime(raw["date"])
    actual = raw.iloc[-1]
    row = features.set_index("thscode").loc["603778.SH"]
    check(actual["date"] == TODAY, "target_cutoff")
    check(close([actual[field] for field in ["open", "high", "low", "close", "volume", "amount"]], [7.35, 7.43, 7.07, 7.09, 20333025, 147926741.25]), "target_raw_ohlcv_exact")
    close_ = raw["hfq_close"]
    previous = close_.shift(1)
    delta = close_.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ema12 = close_.ewm(span=12, adjust=False, min_periods=26).mean()
    ema26 = close_.ewm(span=26, adjust=False, min_periods=26).mean()
    dif = ema12 - ema26
    macd = 2 * (dif - dif.ewm(span=9, adjust=False, min_periods=9).mean()) / close_
    expected = {
        "ret1": close_.iloc[-1] / close_.iloc[-2] - 1,
        "rsi14": (100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1],
        "macd_hist_pct": macd.iloc[-1],
        "macd_hist_delta": macd.iloc[-1] - macd.iloc[-2],
        "close_location": (actual["close"] - actual["low"]) / (actual["high"] - actual["low"]),
        "amount_ratio1_20": raw["amount"].iloc[-1] / raw["amount"].iloc[-20:].mean(),
        "drawdown250": close_.iloc[-1] / raw["hfq_high"].iloc[-250:].max() - 1,
        "return_250d": close_.iloc[-1] / close_.iloc[-251] - 1,
        "return_20d": close_.iloc[-1] / close_.iloc[-21] - 1,
        "upper_shadow": (raw["hfq_high"].iloc[-1] - max(raw["hfq_open"].iloc[-1], close_.iloc[-1])) / close_.iloc[-2],
    }
    for window in [5, 10, 20, 60]:
        expected[f"ma{window}_actual"] = close_.iloc[-window:].mean() / raw["backward_factor"].iloc[-1]
    expected["close_ma20"] = close_.iloc[-1] / close_.iloc[-20:].mean() - 1
    for field, value in expected.items():
        check(close(row[field], value), f"target_independent_{field}", value)
    check(row["ret1"] < 0 and row["close_location"] < .1, "target_rejected_intraday_strength")
    check(row["ma5_actual"] > row["close"] and row["ma10_actual"] > row["close"] and row["ma20_actual"] > row["close"], "target_below_short_medium_ma")
    check(row["macd_hist_pct"] < 0 and row["macd_hist_delta"] < 0, "target_macd_not_confirmed")
    check(not row["BOR"] and not row["strong_reversal"] and not row["in_original_signal_today"], "target_not_a_new_signal")
    weekly = raw[raw["date"] <= FIRST].groupby(raw["date"].dt.to_period("W-SUN")).agg(week_close=("hfq_close", "last"))
    expected_week4 = weekly["week_close"].iloc[-1] / weekly["week_close"].iloc[-5] - 1
    expected_week12 = weekly["week_close"].iloc[-1] / weekly["week_close"].iloc[-13] - 1
    check(close(row["completed_week_return4"], expected_week4) and close(row["completed_week_return12"], expected_week12), "completed_week_features")
    previous_month_close = raw.loc[raw["date"] < TODAY.replace(day=1), "hfq_close"].iloc[-1]
    check(close(row["month_to_date_return"], close_.iloc[-1] / previous_month_close - 1), "unfinished_month_return")
    check(not row["week_to_date_complete"] and not row["month_to_date_complete"], "unfinished_higher_timeframes_labelled")
    for field in ["thscode", "open", "high", "low", "close", "rsi14", "macd_hist_pct", "amount_ratio1_20"]:
        if field == "thscode":
            check(summary["target"][field] == row.name, f"summary_target_{field}")
        else:
            check(close(summary["target"][field], row[field]), f"summary_target_{field}")


def verify_cohorts(con: duckdb.DuckDBPyConnection, features: pd.DataFrame, follow: pd.DataFrame, summary: dict) -> None:
    symbols = follow["thscode"].tolist()
    con.register("cohort_symbols", pd.DataFrame({"thscode": symbols}))
    bars = con.execute(
        """WITH selected AS (
               SELECT d.thscode,d.date,d.open,d.high,d.low,d.close,d.amount,
                      h.open hfq_open,h.high hfq_high,h.low hfq_low,h.close hfq_close,
                      avg(d.amount) OVER(PARTITION BY d.thscode ORDER BY d.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) amount20
               FROM v_daily d JOIN v_daily_hfq h USING(thscode,date)
               JOIN cohort_symbols s ON d.thscode=s.thscode
           ) SELECT * FROM selected WHERE date IN (?,?,?) ORDER BY thscode,date"""
        , [FIRST, SECOND, TODAY]).fetchdf()
    bars["date"] = pd.to_datetime(bars["date"])
    check(len(follow) == 10 and follow["thscode"].nunique() == 10, "first_cohort_10_unique")
    errors = []
    for row in follow.itertuples():
        first = bars[bars["thscode"].eq(row.thscode) & bars["date"].eq(FIRST)].iloc[0]
        second = bars[bars["thscode"].eq(row.thscode) & bars["date"].eq(SECOND)].iloc[0]
        current = bars[bars["thscode"].eq(row.thscode) & bars["date"].eq(TODAY)].iloc[0]
        expected = {
            "return_since_first_close": current["hfq_close"] / first["hfq_close"] - 1,
            "higher_low_than_first": current["hfq_low"] > first["hfq_low"],
            "below_first_low": current["hfq_low"] < first["hfq_low"],
            "above_first_open": current["hfq_close"] > first["hfq_open"],
            "close_above_first_high": current["hfq_close"] > first["hfq_high"],
            "high_above_first_high": current["hfq_high"] > first["hfq_high"],
            "close_above_second_high": current["hfq_close"] > second["hfq_high"],
            "close_below_second_low": current["hfq_close"] < second["hfq_low"],
        }
        if not close(row.return_since_first_close, expected["return_since_first_close"]):
            errors.append((row.thscode, "return"))
        for field in list(expected)[1:]:
            if bool(getattr(row, field)) != expected[field]:
                errors.append((row.thscode, field))
        amount_ratio = current["amount"] / current["amount20"]
        consolidation = expected["higher_low_than_first"] and expected["above_first_open"] and abs((current["hfq_close"] - current["hfq_open"]) / current["hfq_open"]) <= .02 and amount_ratio < 1
        if bool(row.consolidation_27) != consolidation or not close(row.amount_ratio1_20, amount_ratio):
            errors.append((row.thscode, "consolidation"))
        if not close(row.close, current["close"]):
            errors.append((row.thscode, "raw_close"))
    check(not errors, "cohort_bars_returns_and_structure_rebuilt", errors)
    observed = follow[follow["bar27_observed"]]
    group = summary["first_23_strong_cohort"]
    check(group["positive_since_23_close"] == int(observed["return_since_first_close"].gt(0).sum()) == 0, "cohort_all_negative_T2")
    check(group["close_above_23_high"] == int(observed["close_above_first_high"].sum()) == 0, "cohort_no_close_above_first_high")
    check(group["intraday_below_23_low"] == int(observed["below_first_low"].sum()) == 1, "cohort_one_broken_low")
    check(group["close_above_23_open"] == int(observed["above_first_open"].sum()) == 5, "cohort_five_above_first_open")
    check(group["small_body_low_volume_consolidation_27"] == int(observed["consolidation_27"].sum()) == 1, "cohort_one_consolidation")
    check(close(group["mean_since_23_close"], observed["return_since_first_close"].mean()), "cohort_mean_rebuilt")
    check(close(group["median_since_23_close"], observed["return_since_first_close"].median()), "cohort_median_rebuilt")
    target = follow.set_index("thscode").loc["603778.SH"]
    check(not target["below_first_low"] and target["above_first_open"], "target_first_low_not_broken")
    check(target["close_below_second_low"] and not target["consolidation_27"], "target_second_consolidation_failed")
    check(not target["close_above_first_high"] and target["high_above_first_high"], "target_only_intraday_probe_above_high")
    second = pd.read_csv(RES / "cohort_26_followthrough.csv", parse_dates=["date_signal", "date_current"], low_memory=False)
    top2 = second[second["rank"] <= 2]
    check(len(top2) == 2 and close(summary["second_26_top2_T1"]["mean_since_26_close"], top2["return_since_26_close"].mean()), "second_top2_T1_mean")
    check(set(top2["thscode"]) == {"603778.SH", "002240.SZ"}, "second_top2_members")
    check(follow["completed_trade_outcome_known"].eq(False).all() and second["future_trade_outcome_known"].eq(False).all(), "no_future_trade_outcomes")
    frozen = pd.read_csv(REPAIR / "603778_信号对比/results/current_strong_cohort_followthrough.csv")
    stable = True
    for old in frozen.itertuples():
        for day in [23, 26]:
            current = bars[bars["thscode"].eq(old.thscode) & bars["date"].eq(pd.Timestamp(f"2026-10-{day}"))].iloc[0]
            for field in ["open", "high", "low", "close"]:
                stable &= close(current[field], getattr(old, f"actual_{field}_raw_{day}"))
                stable &= close(current[f"hfq_{field}"], getattr(old, f"{field}_hfq_{day}"))
    check(stable, "all_160_frozen_raw_and_hfq_ohlc_values_stable")


def verify_market(con: duckdb.DuckDBPyConnection, market: pd.DataFrame) -> None:
    prefixes = "('600','601','603','605','000','001','002','003')"
    result = con.execute(
        f"""WITH base AS (
                SELECT thscode,date,close,
                       row_number() OVER(PARTITION BY thscode ORDER BY date) listing_session,
                       close/lag(close) OVER(PARTITION BY thscode ORDER BY date)-1 ret1,
                       avg(close) OVER(PARTITION BY thscode ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) ma20
                FROM v_daily_hfq WHERE substr(thscode,1,3) IN {prefixes}
            ), daily AS (
                SELECT date,avg(ret1) market_return,avg(coalesce((ret1>0)::INTEGER,0)) advance_share,
                       avg(CASE WHEN listing_session>=20 THEN (close>ma20)::INTEGER END) breadth20
                FROM base GROUP BY date
            ), cumulative AS (
                SELECT *,sum(ln(1+coalesce(market_return,0))) OVER(ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) log_index
                FROM daily
            ), indexed AS (
                SELECT *,exp(log_index) equal_weight_index,
                       exp(log_index-lag(log_index,20) OVER(ORDER BY date))-1 market_return20
                FROM cumulative
            ) SELECT date,market_return,market_return20,breadth20,advance_share
              FROM indexed WHERE date IN (?,?,?) ORDER BY date""", [FIRST, SECOND, TODAY]
    ).fetchdf()
    result["date"] = pd.to_datetime(result["date"])
    actual = market.set_index("date")
    expected = result.set_index("date")
    for date in [FIRST, SECOND, TODAY]:
        for field in ["market_return", "market_return20", "breadth20", "advance_share"]:
            check(close(actual.loc[date, field], expected.loc[date, field]), f"independent_market_{date.date()}_{field}")
    current = actual.loc[TODAY]
    check(current["market_return20"] < -.10 and current["breadth20"] < .20 and current["advance_share"] < .15, "market_risk_off_confirmed")
    previous = pd.read_csv(REPAIR / "603778_信号对比/results/current_market.csv", parse_dates=["date"]).set_index("date")
    for date in [FIRST, SECOND]:
        check(close(actual.loc[date, ["market_return", "market_return20", "breadth20"]], previous.loc[date, ["market_return", "market_return20", "breadth20"]]), f"frozen_market_stable_{date.date()}")


def verify_original_scan(summary: dict, signals: pd.DataFrame, features: pd.DataFrame) -> None:
    path = Path(summary["original_signal_module"])
    spec = importlib.util.spec_from_file_location("verify_original_signal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wide, market = module.features(module.load(CONFIG["data"]["original_feature_start"]))
    settings = json.loads((REPAIR / "config/strategy.json").read_text())["signal"]
    eligible = wide[wide["date"].eq(TODAY) & wide["listing_session"].ge(250) & wide["amount20"].ge(settings["minimum_amount20"])].copy()
    passed = eligible["drawdown250"].lt(settings["drawdown250_lt"]) & eligible["return_250d"].gt(settings["return_250d_gt"]) & eligible["rsi14"].lt(settings["rsi14_lt"]) & eligible["close_location"].gt(settings["close_location_gt"]) & (market.loc[TODAY, "market_return20"] < settings["market_return20_lt"])
    expected = eligible[passed].sort_values(["drawdown250", "thscode"]).copy()
    expected["rank"] = np.arange(1, len(expected) + 1)
    check(signals[["rank", "thscode"]].equals(expected[["rank", "thscode"]].reset_index(drop=True)), "original_scan_members_and_rank_rebuilt")
    check(len(signals) == 1 and signals.iloc[0]["thscode"] == "603843.SH", "only_one_loose_signal")
    if len(expected):
        expected = expected.set_index("thscode")
        for row in signals.itertuples():
            for field in ["drawdown250", "return_250d", "rsi14", "close_location", "amount20"]:
                check(close(getattr(row, field), expected.loc[row.thscode, field]), f"original_signal_{row.thscode}_{field}")
    check(signals["strong_reversal"].eq(False).all() and summary["strong_reversal_count_within_original_signals"] == 0, "no_strict_reversal_today")
    candidate = signals.iloc[0]
    check(candidate["limit_up_10"] and close(candidate["ref_close"], 4.52) and close(candidate["close"], 4.97), "new_candidate_limit_up_verified")
    check(candidate["BOR"] and candidate["drawdown250"] > -.65, "new_candidate_fails_deep65_only")
    current = features.set_index("thscode")
    check(current["original_rank_today"].dropna().astype(int).to_dict() == dict(zip(signals["thscode"], signals["rank"])), "current_rank_mapping")


def verify_peers(con: duckdb.DuckDBPyConnection, peers: pd.DataFrame) -> None:
    check(peers["official_industry_index"].eq(False).all(), "peer_proxy_not_official_industry")
    target = peers.set_index("target").loc["603778.SH"]
    check(target["note"] == "no_lagged_PCA_peer_group" and pd.isna(target["peer_return1"]), "603778_no_fabricated_peer_evidence")
    membership = pd.read_parquet(INITIAL / "data/work/cluster_membership.parquet", filters=[("effective_month", "=", pd.Timestamp("2026-10-01")), ("cluster_id", "=", 2)])
    con.register("peer_members", membership[["thscode"]])
    result = con.execute(
        """WITH base AS (
               SELECT h.thscode,h.date,h.close,h.close/lag(h.close) OVER(PARTITION BY h.thscode ORDER BY h.date)-1 ret1,
                      h.close/lag(h.close,20) OVER(PARTITION BY h.thscode ORDER BY h.date)-1 ret20,
                      h.close/avg(h.close) OVER(PARTITION BY h.thscode ORDER BY h.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)-1 ma20d
               FROM v_daily_hfq h JOIN peer_members m ON h.thscode=m.thscode
           ) SELECT count(*) members,avg(ret1) peer_return1,avg(ret20) peer_return20,avg((ma20d>0)::INTEGER) peer_breadth20
             FROM base WHERE date=? AND thscode<>'002240.SZ'""", [TODAY]
    ).fetchdf().iloc[0]
    row = peers.set_index("target").loc["002240.SZ"]
    check(int(row["members"]) == int(result["members"]) == 47, "002240_peer_member_count")
    check(close(row[["peer_return1", "peer_return20", "peer_breadth20"]], result[["peer_return1", "peer_return20", "peer_breadth20"]]), "002240_peer_metrics_rebuilt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    check(sha256(DATABASE) == CONFIG["data"]["raw_database_sha256_at_discovery"], "database_hash_frozen")
    cli = json.loads((RES / "cli_validation.json").read_text())
    validation = next(item for item in cli["checks"] if "data validate" in item["command"])
    migration = next(item for item in cli["checks"] if "data migrate" in item["command"])
    check(validation["ok"] and validation["data"]["ok"] and not validation["data"]["issues"], "official_quality_envelope_passed")
    check(migration["applied"] is False and migration["versions"] == [], "no_pending_or_applied_migration")
    quality = json.loads((RES / "source_quality.json").read_text())
    summary = json.loads((RES / "analysis_summary.json").read_text())
    features = pd.read_csv(RES / "current_stock_features.csv", parse_dates=["date"], low_memory=False)
    follow = pd.read_csv(RES / "cohort_23_followthrough.csv", parse_dates=["date", "first_signal_date"], low_memory=False)
    market = pd.read_csv(RES / "market_snapshots.csv", parse_dates=["date"])
    signals = pd.read_csv(RES / "current_original_signals.csv", parse_dates=["date"])
    peers = pd.read_csv(RES / "current_peer_proxy.csv", parse_dates=["date"])
    check(summary["code_sha256"] == sha256(HERE.parent / "followup_today.py"), "analysis_code_hash")
    check(summary["config_sha256"] == sha256(OUT / "analysis_config.json"), "analysis_config_hash")
    check(quality["database_hash_unchanged_during_analysis"] and quality["remote_or_third_party_data_used"] is False, "only_frozen_local_database")
    check(quality["latest_import"]["status"] == "complete" and "20261027/" in quality["latest_import"]["source"], "latest_complete_daily_release")
    con = duckdb.connect(str(DATABASE), read_only=True)
    latest_count, latest_symbols = con.execute("SELECT count(*),count(DISTINCT thscode) FROM v_daily WHERE date=?", [TODAY]).fetchone()
    daily_quality = pd.read_csv(RES / "source_daily_quality.csv", parse_dates=["date"]).set_index("date")
    check(latest_count == latest_symbols == daily_quality.loc[TODAY, "row_count"] == 5568, "latest_row_and_symbol_counts")
    verify_target(con, features, summary)
    verify_cohorts(con, features, follow, summary)
    verify_market(con, market)
    con.close()
    verify_original_scan(summary, signals, features)
    con = duckdb.connect(str(DATABASE), read_only=True)
    verify_peers(con, peers)
    con.close()
    check(summary["execution"]["actual_entry_or_completed_return_known_at_cutoff"] is False, "no_premature_entry_or_return")
    check(signals["future_trade_outcome_known"].eq(False).all(), "new_signal_future_not_known")
    before = {name: sha256(RES / name) for name in FILES}
    if args.rerun:
        process = subprocess.run([sys.executable, "-B", str(HERE.parent / "followup_today.py")], cwd=ROOT, capture_output=True, text=True, timeout=300)
        check(process.returncode == 0, "full_rerun_exit_zero", process.stderr[-2000:])
        if process.returncode == 0:
            after = {name: sha256(RES / name) for name in FILES}
            changed = [name for name in FILES if before[name] != after[name]]
            check(not changed, "full_rerun_byte_identical", {"files": len(FILES), "changed": changed})
    check(sha256(DATABASE) == CONFIG["data"]["raw_database_sha256_at_discovery"], "database_unchanged_at_end")
    report = {
        "cutoff": CONFIG["data"]["cutoff"],
        "passed": all(item["passed"] for item in CHECKS),
        "checks_passed": sum(item["passed"] for item in CHECKS),
        "checks_total": len(CHECKS),
        "rerun_requested": args.rerun,
        "raw_database_sha256": sha256(DATABASE),
        "analysis_output_files": FILES,
        "checks": CHECKS,
    }
    (RES / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, ensure_ascii=False, indent=2))
    failed = [item for item in CHECKS if not item["passed"]]
    if failed:
        print(json.dumps(failed, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
