from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from diagnostics import connection, prepare_eligible

ROOT = Path(__file__).resolve().parent

DIMENSIONS = {
    "market_regime": "CASE WHEN breadth20 <= 0.35 AND market_return20 < 0 THEN 'bear' WHEN breadth20 >= 0.60 AND market_close_ma20 > 0 THEN 'bull' WHEN market_volatility20 >= 0.25 THEN 'high_vol' ELSE 'mixed' END",
    "rsi14_band": "CASE WHEN rsi14 < 20 THEN '<20' WHEN rsi14 < 30 THEN '20-30' WHEN rsi14 < 40 THEN '30-40' WHEN rsi14 < 60 THEN '40-60' ELSE '>=60' END",
    "return10_band": "CASE WHEN return_10d < -0.20 THEN '<-20%' WHEN return_10d < -0.10 THEN '-20--10%' WHEN return_10d < 0 THEN '-10-0%' WHEN return_10d < 0.10 THEN '0-10%' ELSE '>=10%' END",
    "body_band": "CASE WHEN candle_body < 0.01 THEN '<1%' WHEN candle_body < 0.03 THEN '1-3%' WHEN candle_body < 0.06 THEN '3-6%' ELSE '>=6%' END",
    "gap_band": "CASE WHEN gap_open < -0.03 THEN '<-3%' WHEN gap_open < 0 THEN '-3-0%' WHEN gap_open < 0.03 THEN '0-3%' ELSE '>=3%' END",
    "amount_ratio_band": "CASE WHEN amount_ratio1_20 < 0.8 THEN '<0.8' WHEN amount_ratio1_20 < 1.5 THEN '0.8-1.5' WHEN amount_ratio1_20 < 3 THEN '1.5-3' ELSE '>=3' END",
    "price_band": "CASE WHEN close < 5 THEN '<5' WHEN close < 10 THEN '5-10' WHEN close < 20 THEN '10-20' WHEN close < 50 THEN '20-50' ELSE '>=50' END",
    "amount20_band": "CASE WHEN amount20 < 50000000 THEN '20-50m' WHEN amount20 < 100000000 THEN '50-100m' WHEN amount20 < 300000000 THEN '100-300m' ELSE '>=300m' END",
    "atr_band": "CASE WHEN atr20_pct < 0.025 THEN '<2.5%' WHEN atr20_pct < 0.04 THEN '2.5-4%' WHEN atr20_pct < 0.06 THEN '4-6%' ELSE '>=6%' END",
    "volatility_band": "CASE WHEN volatility20 < 0.20 THEN '<20%' WHEN volatility20 < 0.35 THEN '20-35%' WHEN volatility20 < 0.55 THEN '35-55%' ELSE '>=55%' END",
    "macd_state": "CASE WHEN macd_hist_pct < 0 AND macd_hist_delta > 0 THEN 'negative_rising' WHEN macd_hist_pct < 0 THEN 'negative_falling' WHEN macd_hist_delta > 0 THEN 'positive_rising' ELSE 'positive_falling' END",
    "weekly_state": "CASE WHEN week_close_ma4 > 0 AND week_return4 > 0 THEN 'trend_up' WHEN week_close_ma4 > 0 THEN 'above_ma_pullback' WHEN week_return4 > 0 THEN 'below_ma_rebound' ELSE 'down' END",
    "monthly_state": "CASE WHEN month_close_ma6 > 0 AND month_return3 > 0 THEN 'trend_up' WHEN month_close_ma6 > 0 THEN 'above_ma_pullback' WHEN month_return3 > 0 THEN 'below_ma_rebound' ELSE 'down' END",
    "breadth_band": "CASE WHEN breadth20 < 0.20 THEN '<20%' WHEN breadth20 < 0.40 THEN '20-40%' WHEN breadth20 < 0.60 THEN '40-60%' ELSE '>=60%' END",
    "market_return20_band": "CASE WHEN market_return20 < -0.10 THEN '<-10%' WHEN market_return20 < 0 THEN '-10-0%' WHEN market_return20 < 0.10 THEN '0-10%' ELSE '>=10%' END",
    "sentiment_band": "CASE WHEN sentiment_score < -1 THEN '<-1' WHEN sentiment_score < 0 THEN '-1-0' WHEN sentiment_score < 1 THEN '0-1' ELSE '>=1' END",
    "limit_emotion_band": "CASE WHEN limit_up10_share < 0.005 THEN '<0.5%' WHEN limit_up10_share < 0.015 THEN '0.5-1.5%' WHEN limit_up10_share < 0.03 THEN '1.5-3%' ELSE '>=3%' END",
    "recent_limit_band": "CASE WHEN near_limit_up_days20 = 0 THEN 'none' WHEN near_limit_up_days20 = 1 THEN 'one' ELSE 'multiple' END",
    "drawdown250_band": "CASE WHEN drawdown250 < -0.50 THEN '<-50%' WHEN drawdown250 < -0.20 THEN '-50--20%' WHEN drawdown250 < -0.05 THEN '-20--5%' ELSE '>=-5%' END",
    "cluster_return_band": "CASE WHEN peer_return20 IS NULL THEN 'no_cluster' WHEN peer_return20 < -0.05 THEN '<-5%' WHEN peer_return20 < 0.05 THEN '-5-5%' WHEN peer_return20 < 0.15 THEN '5-15%' ELSE '>=15%' END",
    "score_decile": "ntile(10) OVER (ORDER BY score)",
}


def main() -> None:
    hypotheses = json.loads((ROOT / "config" / "hypotheses.json").read_text())["hypotheses"]
    selected_ids = ["engulfing_reversal", "quiet_trend", "high_vol_momentum"]
    con = connection()
    outputs = []
    try:
        prepare_eligible(con, "development")
        for hypothesis in hypotheses:
            if hypothesis["id"] not in selected_ids:
                continue
            con.execute(
                f"""
                CREATE TEMP TABLE interaction_base AS
                SELECT eligible.*, daily_baseline.baseline20, {hypothesis['score']} AS score
                FROM eligible JOIN daily_baseline USING (date)
                WHERE fwd20 IS NOT NULL AND ({hypothesis['entry']})
                """
            )
            for dimension, expression in DIMENSIONS.items():
                if dimension == "score_decile":
                    con.execute("CREATE TEMP TABLE interaction_scored AS SELECT *, ntile(10) OVER (ORDER BY score) AS score_decile FROM interaction_base")
                    source = "interaction_scored"
                else:
                    source = "interaction_base"
                result = con.execute(
                    f"""
                    WITH grouped AS (
                        SELECT {expression} AS bucket, * FROM {source}
                    ), overall AS (
                        SELECT bucket, count(*) AS observations, count(DISTINCT thscode) AS symbols,
                               avg(fwd5) AS mean_fwd5, avg(fwd10) AS mean_fwd10,
                               avg(fwd20) AS mean_fwd20, avg(fwd60) AS mean_fwd60,
                               median(fwd20) AS median_fwd20, avg((fwd20 > 0)::INTEGER) AS win_rate20,
                               avg(fwd20 - baseline20) AS excess20,
                               avg(fwd20 - baseline20) - 0.006 AS net60bp_excess20
                        FROM grouped GROUP BY bucket
                    ), yearly AS (
                        SELECT bucket, year(date) AS year, avg(fwd20 - baseline20) AS yearly_excess20
                        FROM grouped GROUP BY bucket, year(date)
                    ), yearly_stats AS (
                        SELECT bucket, avg((yearly_excess20 > 0)::INTEGER) AS positive_year_share,
                               min(yearly_excess20) AS worst_year_excess20
                        FROM yearly GROUP BY bucket
                    )
                    SELECT * FROM overall JOIN yearly_stats USING (bucket) ORDER BY bucket
                    """
                ).fetchdf()
                result.insert(0, "dimension", dimension)
                result.insert(0, "candidate_id", hypothesis["id"])
                outputs.append(result)
            con.execute("DROP TABLE interaction_base")
            if dimension == "score_decile":
                con.execute("DROP TABLE IF EXISTS interaction_scored")
        frame = pd.concat(outputs, ignore_index=True)
        path = ROOT / "results" / "development" / "nonlinear_interactions.csv"
        frame.to_csv(path, index=False)
        print(json.dumps({"ok": True, "rows": len(frame), "output": str(path)}))
    finally:
        con.close()


if __name__ == "__main__":
    main()
