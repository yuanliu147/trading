SELECT e.*
FROM raw_adjustment_events e
JOIN v_symbol s ON s.thscode = e.thscode
WHERE s.asset_type = 'a-share'
  AND substr(s.ticker, 1, 3) IN ('600', '601', '603', '605', '000', '001', '002', '003')
ORDER BY e.thscode, e.ex_date
