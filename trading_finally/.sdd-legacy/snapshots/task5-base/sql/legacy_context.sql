SELECT
    substr(b.code, 4, 6) || '.' || upper(substr(b.code, 1, 2)) AS code,
    d.calendar_date AS date,
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
FROM stock_basic AS b
JOIN trade_dates AS d
    ON d.is_trading_day = 1
   AND d.calendar_date >= b.ipo_date
   AND (b.out_date IS NULL OR b.out_date = '' OR d.calendar_date < b.out_date)
LEFT JOIN stock_kline_d AS r
    ON r.code = b.code AND r.date = d.calendar_date AND r.adjustflag = 3
LEFT JOIN stock_kline_d AS h
    ON h.code = b.code AND h.date = d.calendar_date AND h.adjustflag = 1
WHERE b.type = 1
  AND (b.code LIKE 'sh.%' OR b.code LIKE 'sz.%')
  AND d.calendar_date >= ?
  AND d.calendar_date <= ?
