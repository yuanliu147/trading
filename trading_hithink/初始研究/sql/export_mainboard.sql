SELECT
    d.thscode,
    d.date,
    d.open,
    d.high,
    d.low,
    d.close,
    d.prev_close,
    d.volume,
    d.amount,
    h.open AS hfq_open,
    h.high AS hfq_high,
    h.low AS hfq_low,
    h.close AS hfq_close,
    f.forward_factor,
    f.backward_factor
FROM v_daily d
JOIN v_symbol s ON s.thscode = d.thscode
JOIN v_daily_hfq h ON h.thscode = d.thscode AND h.date = d.date
LEFT JOIN calc_adjust_factor_daily f ON f.thscode = d.thscode AND f.date = d.date
WHERE s.asset_type = 'a-share'
  AND substr(s.ticker, 1, 3) IN ('600', '601', '603', '605', '000', '001', '002', '003')
ORDER BY d.thscode, d.date
