
        WITH x AS (
            SELECT thscode, date, hfq_open, hfq_high, hfq_low, hfq_close, ret1,
                   previous_hfq_open, previous_hfq_high, previous_hfq_low, previous_hfq_close,
                   lag(hfq_open, 2) OVER g AS open2,
                   lag(hfq_high, 2) OVER g AS high2,
                   lag(hfq_low, 2) OVER g AS low2,
                   lag(hfq_close, 2) OVER g AS close2,
                   lag(hfq_open, 3) OVER g AS open3,
                   lag(hfq_close, 3) OVER g AS close3,
                   lag(ret1, 1) OVER g AS previous_ret1
            FROM read_parquet('/Users/admin/WorkSpace/trading/trading_hithink/data/work/core.parquet')
            WINDOW g AS (PARTITION BY thscode ORDER BY date)
        )
        SELECT *,
            close2 < open2 AND abs(close2 - open2) / nullif(open2, 0) > 0.025
              AND abs(previous_hfq_close - previous_hfq_open) / nullif(previous_hfq_open, 0) < 0.012
              AND hfq_close > hfq_open AND (hfq_close - hfq_open) / nullif(hfq_open, 0) > 0.015
              AND hfq_close > (open2 + close2) / 2 AS morning_star,
            close2 < open2 AND previous_hfq_close < previous_hfq_open
              AND hfq_close > hfq_open AND hfq_close > previous_hfq_high AND ret1 > 0.02 AS three_bar_turnaround,
            previous_hfq_close < previous_hfq_open
              AND (previous_hfq_open - previous_hfq_close) / nullif(previous_hfq_open, 0) > 0.03
              AND hfq_open < previous_hfq_low AND hfq_close > hfq_open
              AND hfq_close > (previous_hfq_open + previous_hfq_close) / 2
              AND hfq_close < previous_hfq_open AS piercing,
            previous_hfq_close < previous_hfq_open
              AND (previous_hfq_open - previous_hfq_close) / nullif(previous_hfq_open, 0) > 0.03
              AND hfq_open > previous_hfq_close AND hfq_close < previous_hfq_open
              AND hfq_close > hfq_open
              AND abs(hfq_close - hfq_open) < 0.5 * abs(previous_hfq_close - previous_hfq_open) AS bullish_harami,
            previous_hfq_close < open2 AND hfq_close > open2
              AND hfq_close > hfq_open AND ret1 > 0.02 AS two_day_recovery,
            close2 > open2 AND previous_hfq_close > previous_hfq_open AND hfq_close > hfq_open
              AND close2 > close3 AND previous_hfq_close > close2 AND hfq_close > previous_hfq_close
              AND (close2 - open2) / nullif(open2, 0) > 0.01
              AND (previous_hfq_close - previous_hfq_open) / nullif(previous_hfq_open, 0) > 0.01
              AND (hfq_close - hfq_open) / nullif(hfq_open, 0) > 0.01
              AND (close2 - low2) / nullif(high2 - low2, 0) > 0.5
              AND (previous_hfq_close - previous_hfq_low) / nullif(previous_hfq_high - previous_hfq_low, 0) > 0.5
              AND (hfq_close - hfq_low) / nullif(hfq_high - hfq_low, 0) > 0.5 AS three_white_soldiers,
            hfq_open > previous_hfq_close * 1.01 AND hfq_close > hfq_open
              AND hfq_close > previous_hfq_high AND ret1 > 0.02 AS gap_up_confirmation,
            previous_ret1 < -0.07 AND hfq_close > hfq_open AND ret1 > 0.02
              AND (hfq_close - hfq_low) / nullif(hfq_high - hfq_low, 0) > 0.7 AS limit_washout_rebound
        FROM x
    