SELECT
    substr(r.code, 4, 6) || '.' || upper(substr(r.code, 1, 2)) AS code,
    r.date,
    r.open,
    r.high,
    r.low,
    r.close,
    r.preclose AS prev_close,
    h.open AS signal_open,
    h.high AS signal_high,
    h.low AS signal_low,
    h.close AS signal_close,
    r.volume,
    r.amount,
    r.trade_status,
    r.is_st,
    b.ipo_date,
    b.out_date
FROM stock_kline_d AS r
JOIN stock_basic AS b ON b.code = r.code
LEFT JOIN stock_kline_d AS h
    ON h.code = r.code AND h.date = r.date AND h.adjustflag = 1
WHERE r.adjustflag = 3
  AND r.date >= '2010-01-01'
  AND r.date <= ?
  AND b.type = 1
  AND (
      substr(b.code, 4, 3) IN ('600', '601', '603', '605', '000', '001', '002', '003')
  )
  AND (
      b.code LIKE 'sh.%' OR b.code LIKE 'sz.%'
  )
