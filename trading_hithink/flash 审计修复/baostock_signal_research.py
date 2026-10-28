from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from baostock_optimization import (
    CANDIDATES,
    END,
    MAINBOARD_PARQUET,
    OUTPUT,
    ROOT,
    literal,
    write_json,
)

PANEL = ROOT.parent.parent / "trading_flash" / "2026-10-24_主板恐慌反转_年化30" / "01_数据" / "research_panel.parquet"
SIGNAL_DATASET = OUTPUT / "cache" / "signal_dataset.parquet"
PERIODS = {
    "development_2018_2021": ("2018-01-01", "2021-12-31"),
    "validation_2022_2023": ("2022-01-01", "2023-12-31"),
    "reused_holdout_2024_2026": ("2024-01-01", "2026-10-23"),
}


def build_signal_dataset() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit TO '6GB'")
    con.execute(f"CREATE VIEW panel AS SELECT * FROM read_parquet('{literal(PANEL)}')")
    con.execute(f"CREATE VIEW bs AS SELECT * FROM read_parquet('{literal(MAINBOARD_PARQUET)}')")
    frame = con.execute(f"""
        WITH raw_trading AS (
            SELECT r.thscode, r.date, r.turn, r.is_st,
                   h.open AS hfq_open, h.close AS hfq_close
            FROM bs r
            JOIN bs h ON r.thscode=h.thscode AND r.date=h.date AND h.adjustflag=1
            WHERE r.adjustflag=3 AND r.trade_status=1 AND r.volume>0
        ), features AS (
            SELECT *,
                   avg(turn) OVER w5 AS turn5,
                   avg(turn) OVER w20 AS turn20,
                   stddev_samp(turn) OVER w20 AS turn_std20,
                   lead(hfq_open, 3) OVER g AS entry_hfq_open,
                   lead(hfq_close, 17) OVER g AS exit_hfq_close,
                   lead(date, 3) OVER g AS entry_date,
                   lead(date, 17) OVER g AS exit_date
            FROM raw_trading
            WINDOW w5 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
                   w20 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
                   g AS (PARTITION BY thscode ORDER BY date)
        ), signals AS (
            SELECT thscode, date, drawdown250, return_250d, rsi14, close_location,
                   market_return20, amount20
            FROM panel
            WHERE date BETWEEN DATE '2017-01-01' AND DATE '{END}'
              AND amount20>=10000000
              AND drawdown250 < -0.45
              AND return_250d > 0
              AND rsi14 < 35
              AND close_location > 0.45
              AND market_return20 < -0.05
        )
        SELECT p.*, f.turn, f.is_st, f.turn5, f.turn20, f.turn_std20,
               f.turn/nullif(f.turn20,0) AS turn1_20,
               f.turn5/nullif(f.turn20,0) AS turn5_20,
               f.turn_std20/nullif(f.turn20,0) AS turn_cv20,
               f.entry_date, f.exit_date, f.entry_hfq_open, f.exit_hfq_close,
               f.exit_hfq_close/nullif(f.entry_hfq_open,0)-1 AS event_return
        FROM signals p
        JOIN features f ON p.thscode=f.thscode AND p.date=f.date
        ORDER BY p.date, p.drawdown250, p.thscode
    """).fetchdf()
    con.close()
    if len(frame) != 4195:
        raise AssertionError(f"Signal coverage/count changed: {len(frame)}")
    if frame[["turn", "turn20", "is_st"]].isna().any().any():
        raise AssertionError("Missing signal-day turnover or ST fields")
    frame.to_parquet(SIGNAL_DATASET, index=False)
    return frame


def add_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["signal_date"] = pd.to_datetime(result["date"])
    result["period"] = "predevelopment_2017"
    for name, (start, end) in PERIODS.items():
        result.loc[result["signal_date"].between(start, end), "period"] = name
    result["st_group"] = np.where(result["is_st"] == 1, "ST", "non_ST")
    for column, name in [("turn1_20", "turn1_band"), ("turn5_20", "turn5_band")]:
        result[name] = pd.cut(
            result[column], [-np.inf, 0.8, 1.2, np.inf],
            labels=["low_le_0.8", "middle_0.8_1.2", "high_ge_1.2"],
        )
    result["absolute_turn_band"] = pd.cut(
        result["turn"], [-np.inf, 1, 3, 8, np.inf],
        labels=["low_le_1", "middle_1_3", "high_3_8", "very_high_ge_8"],
    )
    return result


def rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(["signal_date", "drawdown250", "thscode"], kind="stable").copy()
    ranked["rank"] = ranked.groupby("signal_date").cumcount() + 1
    return ranked[ranked["rank"] <= 10].copy()


def candidate_scenarios(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "baseline": frame,
        "no_st": frame[frame["is_st"] == 0],
        "turn5_low": frame[frame["turn5_20"] <= 0.8],
        "turn5_high": frame[frame["turn5_20"] >= 1.2],
        "turn1_low": frame[frame["turn1_20"] <= 0.8],
        "turn1_high": frame[frame["turn1_20"] >= 1.2],
    }


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(group_columns, observed=True, dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        returns = group["event_return"].dropna()
        date_means = group.dropna(subset=["event_return"]).groupby("signal_date")["event_return"].mean()
        row = dict(zip(group_columns, key))
        row.update({
            "signals": len(group),
            "signal_dates": group["signal_date"].nunique(),
            "symbols": group["thscode"].nunique(),
            "labeled_signals": len(returns),
            "mean_return": returns.mean(),
            "median_return": returns.median(),
            "win_rate": (returns > 0).mean(),
            "date_weighted_mean_return": date_means.mean(),
            "date_weighted_median_return": date_means.median(),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_rank_overlays(frame: pd.DataFrame) -> None:
    baseline = rank_candidates(add_groups(frame))
    candidate_dir = OUTPUT / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for name, column, ascending in [
        ("rank_turn_high", "turn", False),
        ("rank_turn_low", "turn", True),
        ("rank_turn5_high", "turn5_20", False),
        ("rank_turn5_low", "turn5_20", True),
    ]:
        ranked = baseline.sort_values(
            ["signal_date", column, "thscode"], ascending=[True, ascending, True], kind="stable"
        ).copy()
        ranked["rank"] = ranked.groupby("signal_date").cumcount() + 1
        ranked["score"] = -ranked["drawdown250"]
        ranked[["signal_date", "thscode", "rank", "score", "amount20"]].rename(
            columns={"signal_date": "date"}
        ).to_csv(candidate_dir / f"{name}.csv", index=False)


def diagnose(frame: pd.DataFrame) -> dict:
    build_rank_overlays(frame)
    grouped = add_groups(frame)
    study_frames = []
    for factor in ["st_group", "turn1_band", "turn5_band", "absolute_turn_band"]:
        study = summarize(grouped, ["period", factor])
        study.insert(0, "factor", factor)
        study.rename(columns={factor: "group"}, inplace=True)
        study_frames.append(study)
    event_study = pd.concat(study_frames, ignore_index=True)
    event_study.to_csv(OUTPUT / "raw_signal_event_study.csv", index=False)
    candidate_dir = OUTPUT / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_studies = []
    counts = []
    scenario_frames = {}
    for name, subset in candidate_scenarios(grouped).items():
        candidates = rank_candidates(subset)
        scenario_frames[name] = candidates
        candidates["scenario"] = name
        study = summarize(candidates, ["scenario", "period"])
        candidate_studies.append(study)
        for period, group in candidates.groupby("period"):
            counts.append({"scenario": name, "period": period, "candidates": len(group)})
        export = candidates.copy()
        export["score"] = -export["drawdown250"]
        export[["signal_date", "thscode", "rank", "score", "amount20"]].rename(
            columns={"signal_date": "date"}
        ).to_csv(candidate_dir / f"{name}.csv", index=False)
    candidate_study = pd.concat(candidate_studies, ignore_index=True)
    candidate_study.to_csv(OUTPUT / "candidate_event_study.csv", index=False)
    pd.DataFrame(counts).to_csv(OUTPUT / "candidate_counts.csv", index=False)
    baseline = scenario_frames["baseline"]
    frozen = pd.read_csv(CANDIDATES, parse_dates=["signal_date"])
    actual = baseline[["signal_date", "thscode", "rank"]].reset_index(drop=True)
    expected = frozen[["signal_date", "thscode", "rank"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    summary = {
        "raw_signals": len(grouped),
        "raw_signal_st": int((grouped["is_st"] == 1).sum()),
        "baseline_candidates": len(baseline),
        "baseline_candidate_st": int((baseline["is_st"] == 1).sum()),
        "event_label": "Baostock backward-adjusted open at signal+3 sessions to close at signal+17 sessions; no costs or early profit target",
        "selection_policy": "Paired symmetric turnover hypotheses; development and validation are diagnostics, 2024-2026 is reused rather than independent holdout",
        "scenarios": candidate_study.to_dict("records"),
    }
    write_json(OUTPUT / "diagnostic_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "diagnose", "all"])
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.command in ("build", "all"):
        if SIGNAL_DATASET.exists() and not args.rebuild:
            raise FileExistsError(f"Use --rebuild to replace {SIGNAL_DATASET}")
        frame = build_signal_dataset()
    else:
        if not SIGNAL_DATASET.exists():
            raise FileNotFoundError("Run build first")
        frame = pd.read_parquet(SIGNAL_DATASET)
    if args.command in ("diagnose", "all"):
        print(json.dumps(diagnose(frame), ensure_ascii=False, default=str), flush=True)
    else:
        print(json.dumps({"rows": len(frame), "path": str(SIGNAL_DATASET)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
