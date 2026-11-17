SELECT
    d.thscode AS code,
    CAST(d.date AS VARCHAR) AS date,
    d.open AS hithink_open,
    h.open AS hithink_signal_open,
    d.high AS hithink_high,
    h.high AS hithink_signal_high,
    d.low AS hithink_low,
    h.low AS hithink_signal_low,
    d.close AS hithink_close,
    h.close AS hithink_signal_close,
    d.volume AS hithink_volume,
    d.amount AS hithink_amount
FROM v_daily AS d
LEFT JOIN v_daily_hfq AS h
    ON h.thscode = d.thscode AND h.date = d.date
WHERE d.date >= CAST(? AS DATE)
  AND d.date <= CAST(? AS DATE)
  AND (
      substr(d.thscode, 1, 3) IN ('600', '601', '603', '605', '000', '001', '002', '003')
  )
  AND (
      d.thscode LIKE '%.SH' OR d.thscode LIKE '%.SZ'
  )
