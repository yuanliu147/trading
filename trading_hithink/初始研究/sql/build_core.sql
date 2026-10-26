WITH raw AS (
    SELECT
        d.*,
        row_number() OVER (PARTITION BY d.thscode ORDER BY d.date) AS listing_session,
        lag(d.close) OVER (PARTITION BY d.thscode ORDER BY d.date) AS previous_actual_close,
        lag(d.high) OVER (PARTITION BY d.thscode ORDER BY d.date) AS previous_actual_high,
        lag(d.low) OVER (PARTITION BY d.thscode ORDER BY d.date) AS previous_actual_low,
        lag(d.hfq_close) OVER (PARTITION BY d.thscode ORDER BY d.date) AS previous_hfq_close,
        lag(d.hfq_open) OVER (PARTITION BY d.thscode ORDER BY d.date) AS previous_hfq_open,
        lag(d.hfq_high) OVER (PARTITION BY d.thscode ORDER BY d.date) AS previous_hfq_high,
        lag(d.hfq_low) OVER (PARTITION BY d.thscode ORDER BY d.date) AS previous_hfq_low,
        lag(d.backward_factor) OVER (PARTITION BY d.thscode ORDER BY d.date) AS previous_backward_factor,
        coalesce(e.dividend_per_share, 0) AS dividend_per_share,
        coalesce(e.per_share_bonus, 0) AS per_share_bonus,
        coalesce(e.rights_ratio, 0) AS rights_ratio,
        coalesce(e.rights_price, 0) AS rights_price
    FROM read_parquet('@DAILY@') d
    LEFT JOIN read_parquet('@EVENTS@') e
      ON e.thscode = d.thscode AND e.ex_date = d.date
), base AS (
    SELECT
        thscode, date, open, high, low, close, volume, amount,
        hfq_open, hfq_high, hfq_low, hfq_close, backward_factor, previous_backward_factor,
        listing_session, previous_actual_close, previous_hfq_close,
        previous_hfq_open, previous_hfq_high, previous_hfq_low,
        dividend_per_share, per_share_bonus, rights_ratio, rights_price,
        previous_backward_factor IS NOT NULL
          AND abs(backward_factor / nullif(previous_backward_factor, 0) - 1) > 0.000001
          AND dividend_per_share = 0 AND per_share_bonus = 0 AND rights_ratio = 0 AS factor_only_adjustment,
        CASE
            WHEN previous_actual_close IS NULL OR previous_backward_factor IS NULL THEN NULL
            ELSE previous_actual_close * previous_backward_factor / nullif(backward_factor, 0)
        END AS reference_close,
        hfq_close / nullif(previous_hfq_close, 0) - 1 AS ret1,
        greatest(hfq_high, previous_hfq_close) - least(hfq_low, previous_hfq_close) AS true_range_hfq,
        lag(hfq_high, 2) OVER (PARTITION BY thscode ORDER BY date) > greatest(
            lag(hfq_high, 4) OVER (PARTITION BY thscode ORDER BY date),
            lag(hfq_high, 3) OVER (PARTITION BY thscode ORDER BY date),
            lag(hfq_high, 1) OVER (PARTITION BY thscode ORDER BY date),
            hfq_high
        ) AS swing_high_confirmed,
        lag(hfq_low, 2) OVER (PARTITION BY thscode ORDER BY date) < least(
            lag(hfq_low, 4) OVER (PARTITION BY thscode ORDER BY date),
            lag(hfq_low, 3) OVER (PARTITION BY thscode ORDER BY date),
            lag(hfq_low, 1) OVER (PARTITION BY thscode ORDER BY date),
            hfq_low
        ) AS swing_low_confirmed,
        lag(hfq_high, 2) OVER (PARTITION BY thscode ORDER BY date) AS confirmed_swing_high,
        lag(hfq_low, 2) OVER (PARTITION BY thscode ORDER BY date) AS confirmed_swing_low
    FROM raw
), computed AS (
    SELECT
        *,
        close / nullif(reference_close, 0) - 1 AS raw_ret1,
        open / nullif(reference_close, 0) - 1 AS gap_open,
        (close - open) / nullif(reference_close, 0) AS candle_body,
        (high - greatest(open, close)) / nullif(reference_close, 0) AS upper_shadow,
        (least(open, close) - low) / nullif(reference_close, 0) AS lower_shadow,
        (high - low) / nullif(reference_close, 0) AS range_pct,
        (close - low) / nullif(high - low, 0) AS close_location,
        true_range_hfq / nullif(previous_hfq_close, 0) AS tr_pct,
        round(reference_close * 1.10, 2) AS up_limit_10,
        round(reference_close * 0.90, 2) AS down_limit_10,
        round(reference_close * 1.05, 2) AS up_limit_05,
        round(reference_close * 0.95, 2) AS down_limit_05,
        hfq_high < previous_hfq_high AND hfq_low > previous_hfq_low AS inside_bar,
        lower_shadow > 2 * abs(candle_body) AND upper_shadow < greatest(abs(candle_body), 0.002) AS hammer_shape,
        hfq_close > hfq_open AND previous_hfq_close < previous_hfq_open
          AND hfq_open <= previous_hfq_close AND hfq_close >= previous_hfq_open AS bullish_engulfing
    FROM base
), rolling AS (
    SELECT
        *,
        avg(hfq_close) OVER w5 AS ma5,
        avg(hfq_close) OVER w10 AS ma10,
        avg(hfq_close) OVER w20 AS ma20,
        avg(hfq_close) OVER w50 AS ma50,
        avg(hfq_close) OVER w60 AS ma60,
        avg(hfq_close) OVER w120 AS ma120,
        avg(hfq_close) OVER w250 AS ma250,
        avg(amount) OVER w5 AS amount5,
        avg(amount) OVER w20 AS amount20,
        avg(amount) OVER w60 AS amount60,
        avg(volume) OVER w20 AS volume20,
        avg(tr_pct) OVER w20 AS atr20_pct,
        avg(range_pct) OVER w20 AS range20_pct,
        stddev_samp(ret1) OVER w20 * sqrt(252) AS volatility20,
        stddev_samp(ret1) OVER w60 * sqrt(252) AS volatility60,
        stddev_samp(ret1) OVER w5 / nullif(stddev_samp(ret1) OVER w20, 0) AS volatility_contraction,
        max(hfq_high) OVER w20p AS previous_high20,
        max(hfq_high) OVER w60p AS previous_high60,
        max(hfq_high) OVER w120p AS previous_high120,
        max(hfq_high) OVER w250p AS previous_high250,
        min(hfq_low) OVER w20p AS previous_low20,
        min(hfq_low) OVER w60p AS previous_low60,
        max(hfq_high) OVER w60 AS rolling_high60,
        min(hfq_low) OVER w60 AS rolling_low60,
        max(hfq_high) OVER w250 AS rolling_high250,
        min(range_pct) OVER w7p AS previous_min_range7,
        sum(CASE WHEN ret1 > 0 THEN 1 ELSE 0 END) OVER w20 AS up_days20,
        sum(CASE WHEN raw_ret1 >= 0.095 THEN 1 ELSE 0 END) OVER w20 AS near_limit_up_days20,
        sum(CASE WHEN raw_ret1 <= -0.095 THEN 1 ELSE 0 END) OVER w20 AS near_limit_down_days20,
        max(raw_ret1) OVER w20 AS max_daily_return20,
        min(raw_ret1) OVER w20 AS min_daily_return20,
        avg(CASE WHEN ret1 > 0 THEN amount END) OVER w20 AS up_amount20,
        avg(CASE WHEN ret1 < 0 THEN amount END) OVER w20 AS down_amount20
    FROM computed
    WINDOW
        w5 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
        w10 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
        w20 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
        w50 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),
        w60 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),
        w120 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 119 PRECEDING AND CURRENT ROW),
        w250 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW),
        w20p AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
        w60p AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING),
        w120p AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 120 PRECEDING AND 1 PRECEDING),
        w250p AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 250 PRECEDING AND 1 PRECEDING),
        w7p AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)
)
SELECT
    *,
    hfq_close / nullif(previous_hfq_close, 0) - 1 AS return_1d,
    hfq_close / nullif(lag(hfq_close, 5) OVER g, 0) - 1 AS return_5d,
    hfq_close / nullif(lag(hfq_close, 10) OVER g, 0) - 1 AS return_10d,
    hfq_close / nullif(lag(hfq_close, 20) OVER g, 0) - 1 AS return_20d,
    hfq_close / nullif(lag(hfq_close, 60) OVER g, 0) - 1 AS return_60d,
    hfq_close / nullif(lag(hfq_close, 120) OVER g, 0) - 1 AS return_120d,
    hfq_close / nullif(lag(hfq_close, 250) OVER g, 0) - 1 AS return_250d,
    hfq_close / nullif(ma5, 0) - 1 AS close_ma5,
    hfq_close / nullif(ma10, 0) - 1 AS close_ma10,
    hfq_close / nullif(ma20, 0) - 1 AS close_ma20,
    hfq_close / nullif(ma50, 0) - 1 AS close_ma50,
    hfq_close / nullif(ma60, 0) - 1 AS close_ma60,
    hfq_close / nullif(ma120, 0) - 1 AS close_ma120,
    hfq_close / nullif(ma250, 0) - 1 AS close_ma250,
    ma20 / nullif(lag(ma20, 5) OVER g, 0) - 1 AS ma20_slope5,
    ma60 / nullif(lag(ma60, 10) OVER g, 0) - 1 AS ma60_slope10,
    ma20 / nullif(ma60, 0) - 1 AS ma20_ma60,
    amount / nullif(amount20, 0) AS amount_ratio1_20,
    amount5 / nullif(amount20, 0) AS amount_ratio5_20,
    amount20 / nullif(amount60, 0) AS amount_ratio20_60,
    volume / nullif(volume20, 0) AS volume_ratio1_20,
    up_amount20 / nullif(down_amount20, 0) AS amount_up_down20,
    hfq_close / nullif(previous_high20, 0) - 1 AS breakout20,
    hfq_close / nullif(previous_high60, 0) - 1 AS breakout60,
    hfq_close / nullif(previous_high120, 0) - 1 AS breakout120,
    hfq_close / nullif(previous_high250, 0) - 1 AS breakout250,
    hfq_close / nullif(previous_low20, 0) - 1 AS distance_low20,
    hfq_close / nullif(previous_low60, 0) - 1 AS distance_low60,
    hfq_close / nullif(rolling_high60, 0) - 1 AS drawdown60,
    hfq_close / nullif(rolling_high250, 0) - 1 AS drawdown250,
    (hfq_close - rolling_low60) / nullif(rolling_high60 - rolling_low60, 0) AS price_position60,
    range_pct <= previous_min_range7 AS narrow_range7,
    close >= up_limit_10 - 0.0001 AS limit_up_10,
    close <= down_limit_10 + 0.0001 AS limit_down_10,
    high >= up_limit_10 - 0.0001 AND close < up_limit_10 - 0.0001 AS broken_limit_up_10,
    close >= up_limit_05 - 0.0001 AS limit_up_05,
    close <= down_limit_05 + 0.0001 AS limit_down_05,
    high >= up_limit_05 - 0.0001 AND close < up_limit_05 - 0.0001 AS broken_limit_up_05,
    lead(hfq_open, 1) OVER g AS next_hfq_open,
    lead(hfq_open, 6) OVER g AS open_after_5_sessions,
    lead(hfq_open, 11) OVER g AS open_after_10_sessions,
    lead(hfq_open, 21) OVER g AS open_after_20_sessions,
    lead(hfq_open, 61) OVER g AS open_after_60_sessions,
    lead(open, 1) OVER g AS next_actual_open,
    lead(up_limit_10, 1) OVER g AS next_up_limit_10,
    lead(up_limit_05, 1) OVER g AS next_up_limit_05,
    lead(down_limit_10, 1) OVER g AS next_down_limit_10,
    lead(down_limit_05, 1) OVER g AS next_down_limit_05
FROM rolling
WINDOW g AS (PARTITION BY thscode ORDER BY date)
