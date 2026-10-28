from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from baostock_optimization import END, MAINBOARD_PARQUET, OUTPUT, SYMBOLS_PARQUET, literal, write_json
from baostock_signal_research import PANEL, SIGNAL_DATASET, PERIODS, add_groups, rank_candidates, summarize


def market_and_current_signals(con: duckdb.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = con.execute("""
        WITH joined AS (
            SELECT r.thscode, r.date, h.close AS hfq_close,
                   h.close/nullif(lag(h.close) OVER (PARTITION BY r.thscode ORDER BY r.date),0)-1 AS ret1
            FROM bs r
            JOIN bs h ON r.thscode=h.thscode AND r.date=h.date AND h.adjustflag=1
            WHERE r.adjustflag=3 AND r.trade_status=1 AND r.volume>0
        )
        SELECT date, avg(ret1) AS market_return
        FROM joined GROUP BY date ORDER BY date
    """).fetchdf()
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date").reset_index(drop=True)
    market["equal_weight_index"] = (1 + market["market_return"].fillna(0)).cumprod()
    market["market_return20_all"] = market["equal_weight_index"].pct_change(20, fill_method=None)
    con.register("all_market", market[["date", "market_return20_all"]])
    current = con.execute(f"""
        WITH eligible AS (
            SELECT p.thscode, p.date, p.drawdown250, p.return_250d, p.rsi14,
                   p.close_location, p.amount20, m.market_return20_all
            FROM panel p JOIN all_market m ON p.date=m.date
            WHERE p.date BETWEEN DATE '2017-01-01' AND DATE '{END}'
              AND p.amount20>=10000000 AND p.drawdown250 < -0.45
              AND p.return_250d > 0 AND p.rsi14 < 35 AND p.close_location > 0.45
              AND m.market_return20_all < -0.05
        )
        SELECT p.*, f.turn, f.is_st, f.turn5, f.turn20,
               f.turn/nullif(f.turn20,0) AS turn1_20,
               f.turn5/nullif(f.turn20,0) AS turn5_20,
               f.entry_date, f.exit_date, f.entry_hfq_open, f.exit_hfq_close,
               f.exit_hfq_close/nullif(f.entry_hfq_open,0)-1 AS event_return
        FROM eligible p JOIN features f ON p.thscode=f.thscode AND p.date=f.date
        ORDER BY p.date, p.drawdown250, p.thscode
    """).fetchdf()
    original_market = con.execute("""
        SELECT DISTINCT date, market_return20 FROM panel ORDER BY date
    """).fetchdf()
    return market, current, original_market


def delisted_features(con: duckdb.Connection, original_market: pd.DataFrame) -> pd.DataFrame:
    frame = con.execute("""
        SELECT f.thscode, f.date, f.hfq_close, f.turn, f.is_st, f.turn5, f.turn20,
               f.turn/nullif(f.turn20,0) AS turn1_20,
               f.turn5/nullif(f.turn20,0) AS turn5_20,
               f.return_250d_bs AS return_250d, f.drawdown250_bs AS drawdown250,
               f.amount20_bs AS amount20, f.close_location_bs AS close_location,
               f.entry_date, f.exit_date, f.entry_hfq_open, f.exit_hfq_close,
               f.exit_hfq_close/nullif(f.entry_hfq_open,0)-1 AS event_return,
               s.code_name, s.out_date, s.listing_status
        FROM features f JOIN bs_symbols s ON f.thscode=s.thscode
        WHERE s.status=0 AND f.date>=DATE '2017-01-01'
        ORDER BY f.thscode, f.date
    """).fetchdf()
    frame = frame.sort_values(["thscode", "date"], kind="stable").reset_index(drop=True)
    grouped = frame.groupby("thscode", sort=False)
    delta = grouped["hfq_close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.groupby(frame["thscode"], sort=False).transform(
        lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    average_loss = loss.groupby(frame["thscode"], sort=False).transform(
        lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    frame["rsi14"] = 100 - 100 / (1 + average_gain / average_loss.replace(0, np.nan))
    original_market["date"] = pd.to_datetime(original_market["date"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.merge(original_market, on="date", how="left", validate="many_to_one")
    return frame


def select_delisted(frame: pd.DataFrame, market_column: str) -> pd.DataFrame:
    selected = frame[
        (frame["amount20"] >= 10000000)
        & (frame["drawdown250"] < -0.45)
        & (frame["return_250d"] > 0)
        & (frame["rsi14"] < 35)
        & (frame["close_location"] > 0.45)
        & (frame[market_column] < -0.05)
    ].copy()
    return selected.sort_values(["date", "drawdown250", "thscode"], kind="stable")


def combined_candidates(current: pd.DataFrame, delisted: pd.DataFrame) -> pd.DataFrame:
    current = current.copy()
    current["listing_status"] = "listed"
    current["date"] = pd.to_datetime(current["date"])
    columns = [
        "thscode", "date", "drawdown250", "return_250d", "rsi14", "close_location",
        "amount20", "turn", "is_st", "turn5", "turn20", "turn1_20", "turn5_20",
        "entry_date", "exit_date", "entry_hfq_open", "exit_hfq_close", "event_return", "listing_status",
    ]
    combined = pd.concat([current[columns], delisted[columns]], ignore_index=True)
    combined["signal_date"] = combined["date"]
    return rank_candidates(combined)


def period_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = add_groups(frame)
    return summarize(grouped, ["listing_status", "period"])


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit TO '6GB'")
    con.execute(f"CREATE VIEW bs AS SELECT * FROM read_parquet('{literal(MAINBOARD_PARQUET)}')")
    con.execute(f"CREATE VIEW bs_symbols AS SELECT * FROM read_parquet('{literal(SYMBOLS_PARQUET)}')")
    con.execute(f"CREATE VIEW panel AS SELECT * FROM read_parquet('{literal(PANEL)}')")
    con.execute("""
        CREATE VIEW features AS
        WITH joined AS (
            SELECT r.thscode, r.date, r.amount, r.turn, r.is_st, r.open, r.high, r.low, r.close,
                   h.open AS hfq_open, h.high AS hfq_high, h.close AS hfq_close,
                   h.close/nullif(lag(h.close) OVER (PARTITION BY r.thscode ORDER BY r.date),0)-1 AS ret1
            FROM bs r
            JOIN bs h ON r.thscode=h.thscode AND r.date=h.date AND h.adjustflag=1
            WHERE r.adjustflag=3 AND r.trade_status=1 AND r.volume>0
        )
        SELECT *,
               hfq_close/nullif(lag(hfq_close,250) OVER g,0)-1 AS return_250d_bs,
               hfq_close/nullif(max(hfq_high) OVER w250,0)-1 AS drawdown250_bs,
               avg(amount) OVER w20 AS amount20_bs,
               (close-low)/nullif(high-low,0) AS close_location_bs,
               avg(turn) OVER w5 AS turn5,
               avg(turn) OVER w20 AS turn20,
               lead(hfq_open,3) OVER g AS entry_hfq_open,
               lead(hfq_close,17) OVER g AS exit_hfq_close,
               lead(date,3) OVER g AS entry_date,
               lead(date,17) OVER g AS exit_date
        FROM joined
        WINDOW g AS (PARTITION BY thscode ORDER BY date),
               w5 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
               w20 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
               w250 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW)
    """)
    market, current_inclusive, original_market = market_and_current_signals(con)
    delisted_all = delisted_features(con, original_market)
    delisted_all = delisted_all.merge(market[["date", "market_return20_all"]], on="date", how="left", validate="many_to_one")
    con.close()
    current_fixed = pd.read_parquet(SIGNAL_DATASET)
    current_fixed["listing_status"] = "listed"
    delisted_fixed = select_delisted(delisted_all, "market_return20")
    delisted_inclusive = select_delisted(delisted_all, "market_return20_all")
    fixed_candidates = combined_candidates(current_fixed, delisted_fixed)
    inclusive_candidates = combined_candidates(current_inclusive, delisted_inclusive)
    market_comparison = original_market.merge(market, on="date", how="inner")
    market_comparison["gate_original"] = market_comparison["market_return20"] < -0.05
    market_comparison["gate_all"] = market_comparison["market_return20_all"] < -0.05
    summary = {
        "market_comparison": {
            "sessions": len(market_comparison),
            "return20_correlation": market_comparison[["market_return20", "market_return20_all"]].corr().iloc[0, 1],
            "median_absolute_difference": (market_comparison["market_return20"] - market_comparison["market_return20_all"]).abs().median(),
            "p99_absolute_difference": (market_comparison["market_return20"] - market_comparison["market_return20_all"]).abs().quantile(0.99),
            "gate_disagreement_sessions": int((market_comparison["gate_original"] != market_comparison["gate_all"]).sum()),
        },
        "fixed_original_market_gate": {
            "current_signals": len(current_fixed),
            "delisted_signals": len(delisted_fixed),
            "delisted_signal_symbols": delisted_fixed["thscode"].nunique(),
            "combined_candidates": len(fixed_candidates),
            "delisted_candidates": int((fixed_candidates["listing_status"] == "delisted").sum()),
            "delisted_candidate_symbols": fixed_candidates.loc[fixed_candidates["listing_status"] == "delisted", "thscode"].nunique(),
            "delisted_top2_candidates": int(((fixed_candidates["listing_status"] == "delisted") & (fixed_candidates["rank"] <= 2)).sum()),
        },
        "all_universe_market_gate": {
            "current_signals": len(current_inclusive),
            "delisted_signals": len(delisted_inclusive),
            "delisted_signal_symbols": delisted_inclusive["thscode"].nunique(),
            "combined_candidates": len(inclusive_candidates),
            "delisted_candidates": int((inclusive_candidates["listing_status"] == "delisted").sum()),
            "delisted_candidate_symbols": inclusive_candidates.loc[inclusive_candidates["listing_status"] == "delisted", "thscode"].nunique(),
            "delisted_top2_candidates": int(((inclusive_candidates["listing_status"] == "delisted") & (inclusive_candidates["rank"] <= 2)).sum()),
        },
        "limitations": [
            "Signal-level audit only; no portfolio CAGR is reported for the delisted universe.",
            "Delisted-stock corporate actions, final cash settlement, and residual value require explicit execution rules before an account replay.",
            "The all-universe market gate also reflects small vendor/code-history coverage differences.",
        ],
    }
    fixed_events = period_summary(pd.concat([
        current_fixed.assign(date=pd.to_datetime(current_fixed["date"])),
        delisted_fixed,
    ], ignore_index=True))
    inclusive_events = period_summary(inclusive_candidates)
    fixed_events.to_csv(OUTPUT / "delisted_fixed_gate_event_study.csv", index=False)
    inclusive_events.to_csv(OUTPUT / "delisted_inclusive_candidate_event_study.csv", index=False)
    delisted_fixed.to_csv(OUTPUT / "delisted_signals_fixed_gate.csv", index=False)
    delisted_inclusive.to_csv(OUTPUT / "delisted_signals_inclusive_gate.csv", index=False)
    fixed_candidates.to_csv(OUTPUT / "candidates_with_delisted_fixed_gate.csv", index=False)
    inclusive_candidates.to_csv(OUTPUT / "candidates_with_delisted_inclusive_gate.csv", index=False)
    market.to_csv(OUTPUT / "all_universe_market_daily.csv", index=False)
    write_json(OUTPUT / "delisted_signal_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
