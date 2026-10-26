
        WITH weekly_raw AS (
            SELECT thscode, date_trunc('week', date) AS week, max(date) AS week_end,
                   arg_max(hfq_close, date) AS week_close, max(hfq_high) AS week_high,
                   min(hfq_low) AS week_low, sum(amount) AS week_amount
            FROM read_parquet('/Users/admin/WorkSpace/trading/trading_hithink/data/work/core.parquet')
            GROUP BY thscode, date_trunc('week', date)
        ), weekly AS (
            SELECT *,
                   week_close / nullif(avg(week_close) OVER (PARTITION BY thscode ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW), 0) - 1 AS week_close_ma4,
                   week_close / nullif(lag(week_close, 4) OVER (PARTITION BY thscode ORDER BY week), 0) - 1 AS week_return4,
                   week_close / nullif(lag(week_close, 12) OVER (PARTITION BY thscode ORDER BY week), 0) - 1 AS week_return12,
                   week_close / nullif(max(week_high) OVER (PARTITION BY thscode ORDER BY week ROWS BETWEEN 12 PRECEDING AND CURRENT ROW), 0) - 1 AS week_drawdown12
            FROM weekly_raw
        ), monthly_raw AS (
            SELECT thscode, date_trunc('month', date) AS month, max(date) AS month_end,
                   arg_max(hfq_close, date) AS month_close, max(hfq_high) AS month_high,
                   min(hfq_low) AS month_low, sum(amount) AS month_amount
            FROM read_parquet('/Users/admin/WorkSpace/trading/trading_hithink/data/work/core.parquet')
            GROUP BY thscode, date_trunc('month', date)
        ), monthly AS (
            SELECT *,
                   month_close / nullif(avg(month_close) OVER (PARTITION BY thscode ORDER BY month ROWS BETWEEN 5 PRECEDING AND CURRENT ROW), 0) - 1 AS month_close_ma6,
                   month_close / nullif(lag(month_close, 3) OVER (PARTITION BY thscode ORDER BY month), 0) - 1 AS month_return3,
                   month_close / nullif(lag(month_close, 12) OVER (PARTITION BY thscode ORDER BY month), 0) - 1 AS month_return12
            FROM monthly_raw
        ), high_swings AS (
            SELECT thscode, date AS high_confirmation_date, confirmed_swing_high
            FROM read_parquet('/Users/admin/WorkSpace/trading/trading_hithink/data/work/core.parquet') WHERE swing_high_confirmed
        ), low_swings AS (
            SELECT thscode, date AS low_confirmation_date, confirmed_swing_low
            FROM read_parquet('/Users/admin/WorkSpace/trading/trading_hithink/data/work/core.parquet') WHERE swing_low_confirmed
        )
        SELECT c.*, o.* EXCLUDE (thscode, date), m.* EXCLUDE (date),
               w.week_close_ma4, w.week_return4, w.week_return12, w.week_drawdown12,
               mo.month_close_ma6, mo.month_return3, mo.month_return12,
               hs.high_confirmation_date, hs.confirmed_swing_high,
               ls.low_confirmation_date, ls.confirmed_swing_low,
               c.hfq_close / nullif(hs.confirmed_swing_high, 0) - 1 AS distance_swing_high,
               c.hfq_close / nullif(ls.confirmed_swing_low, 0) - 1 AS distance_swing_low,
               date_diff('day', hs.high_confirmation_date, c.date) AS days_since_swing_high,
               date_diff('day', ls.low_confirmation_date, c.date) AS days_since_swing_low
        FROM read_parquet('/Users/admin/WorkSpace/trading/trading_hithink/data/work/core.parquet') c
        JOIN read_parquet('/Users/admin/WorkSpace/trading/trading_hithink/data/work/oscillators.parquet') o USING (thscode, date)
        JOIN read_parquet('/Users/admin/WorkSpace/trading/trading_hithink/data/work/market_daily.parquet') m USING (date)
        ASOF LEFT JOIN weekly w ON c.thscode = w.thscode AND w.week_end < date_trunc('week', c.date)
        ASOF LEFT JOIN monthly mo ON c.thscode = mo.thscode AND mo.month_end < date_trunc('month', c.date)
        ASOF LEFT JOIN high_swings hs ON c.thscode = hs.thscode AND hs.high_confirmation_date <= c.date
        ASOF LEFT JOIN low_swings ls ON c.thscode = ls.thscode AND ls.low_confirmation_date <= c.date
    