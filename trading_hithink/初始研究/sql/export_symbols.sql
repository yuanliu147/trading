SELECT thscode, ticker, name, exchange, asset_type, updated_at
FROM v_symbol
WHERE asset_type = 'a-share'
  AND substr(ticker, 1, 3) IN ('600', '601', '603', '605', '000', '001', '002', '003')
ORDER BY thscode
