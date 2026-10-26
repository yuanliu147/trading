from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from diagnostics import connection, prepare_eligible

ROOT = Path(__file__).resolve().parent

BASE = "bullish_engulfing AND return_10d < -0.08 AND rsi14 < 40"
VARIANTS = {
    "engulfing_deep50": BASE + " AND drawdown250 < -0.50",
    "engulfing_deep50_lowamount": BASE + " AND drawdown250 < -0.50 AND amount20 < 50000000",
    "engulfing_deep50_panic": BASE + " AND drawdown250 < -0.50 AND market_return20 < -0.10",
    "engulfing_deep50_atr3": BASE + " AND drawdown250 < -0.50 AND atr20_pct >= 0.03",
    "engulfing_deep50_rsi20": BASE + " AND drawdown250 < -0.50 AND rsi14 >= 20",
    "engulfing_deep50_drop10": BASE + " AND drawdown250 < -0.50 AND return_10d < -0.10",
    "engulfing_deep50_drop15": BASE + " AND drawdown250 < -0.50 AND return_10d < -0.15",
    "engulfing_panic": BASE + " AND market_return20 < -0.10",
    "engulfing_lowamount": BASE + " AND amount20 < 50000000",
    "engulfing_deep50_lowamount_atr3": BASE + " AND drawdown250 < -0.50 AND amount20 < 50000000 AND atr20_pct >= 0.03",
}


def main() -> None:
    con = connection()
    frames = []
    try:
        prepare_eligible(con, "development")
        for identifier, condition in VARIANTS.items():
            overall = con.execute(
                f"""
                SELECT count(*) AS observations, count(DISTINCT thscode) AS symbols,
                       count(DISTINCT date) AS active_days,
                       avg(fwd5) AS mean_fwd5, avg(fwd10) AS mean_fwd10,
                       avg(fwd20) AS mean_fwd20, avg(fwd60) AS mean_fwd60,
                       median(fwd20) AS median_fwd20, avg((fwd20 > 0)::INTEGER) AS win_rate20,
                       avg(fwd20 - baseline20) AS excess20,
                       avg(fwd20 - baseline20) - 0.006 AS net60bp_excess20
                FROM eligible JOIN daily_baseline USING (date)
                WHERE fwd20 IS NOT NULL AND ({condition})
                """
            ).fetchdf()
            yearly = con.execute(
                f"""
                SELECT year(date) AS year, count(*) AS observations,
                       avg(fwd20) AS mean_fwd20, avg(fwd20 - baseline20) AS excess20
                FROM eligible JOIN daily_baseline USING (date)
                WHERE fwd20 IS NOT NULL AND ({condition})
                GROUP BY year(date) ORDER BY year
                """
            ).fetchdf()
            overall.insert(0, "id", identifier)
            overall["positive_excess_years"] = int((yearly["excess20"] > 0).sum())
            overall["worst_year_excess20"] = yearly["excess20"].min()
            overall["condition"] = condition
            frames.append(overall)
            yearly.insert(0, "id", identifier)
            yearly.to_csv(ROOT / "results" / "development" / f"round2_{identifier}_yearly.csv", index=False)
        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values("net60bp_excess20", ascending=False)
        path = ROOT / "results" / "development" / "round2_summary.csv"
        result.to_csv(path, index=False)
        print(json.dumps({"ok": True, "output": str(path), "holdout_used": False}))
    finally:
        con.close()


if __name__ == "__main__":
    main()
