# -*- coding: utf-8 -*-
"""F08 独立抽查①：主板跌停基准率与次日再跌停率（只读）。
数据：trading_finally/data/main_panel.parquet（冻结主板含退市面板，as-of 2026-10-29）。
口径（按审计任务指定近似）：
  非ST：close/prev_close - 1 <= -9.8%（主板10%跌停，四舍五入近似，注明）
  ST  ：close/prev_close - 1 <= -4.8%（5%跌停近似，is_st 点时标记）
  有效行：data_valid 非 False 且 close>0、prev_close>0。
窗口：A=2016-01-01..2026-10-29（任务窗口）；B=2016-10-14..2026-10-13（与原全A研究同窗）。
次日再跌停两种口径：
  cal：下一市场交易日有有效行情才入分母（停牌剔除）；
  lead：按个股下一有效行情行（含停牌后复牌首日）。
对照：原全A研究 基准率0.391%、昨日收盘跌停后再跌停12.34%。
"""
import duckdb, os, pandas as pd

PANEL = '/Users/admin/WorkSpace/trading/trading_finally/data/main_panel.parquet'
OUT = os.path.dirname(os.path.abspath(__file__))
con = duckdb.connect()

con.execute(f"""
CREATE TEMP TABLE v AS
SELECT date, code, coalesce(is_st,0) AS is_st,
       close/prev_close - 1 AS ret,
       CASE WHEN coalesce(is_st,0)=1 THEN close/prev_close - 1 <= -0.048
            ELSE close/prev_close - 1 <= -0.098 END AS ld,
       CASE WHEN coalesce(is_st,0)=1 THEN close <= round(prev_close*0.95, 2) + 1e-9
            ELSE close <= round(prev_close*0.90, 2) + 1e-9 END AS ld_exact
FROM read_parquet('{PANEL}')
WHERE data_valid IS NOT FALSE AND close > 0 AND prev_close > 0
""")
# 注意：LEAD 必须在过滤跌停之前、对全部有效行计算（单独建表，避免优化器把 WHERE 下推到窗口之下）
con.execute("""
CREATE TEMP TABLE w AS
SELECT *, lead(ld) OVER (PARTITION BY code ORDER BY date) AS lead_ld,
          lead(ld_exact) OVER (PARTITION BY code ORDER BY date) AS lead_ld_exact
FROM v
""")
con.execute("""
CREATE TEMP TABLE cal AS
SELECT date, row_number() OVER (ORDER BY date) AS rn FROM (SELECT DISTINCT date FROM v)
""")
con.execute("""
CREATE TEMP TABLE nxt AS SELECT a.date, b.date AS ndate FROM cal a JOIN cal b ON b.rn = a.rn + 1
""")
con.execute("""
CREATE TEMP TABLE ev AS
SELECT w.date, w.code, w.is_st, n.ndate,
       (v2.code IS NOT NULL) AS has_next,
       coalesce(v2.ld, FALSE) AS next_ld,
       coalesce(v2.ld_exact, FALSE) AS next_ld_exact,
       w.lead_ld, w.lead_ld_exact
FROM w JOIN nxt n ON w.date = n.date
LEFT JOIN v v2 ON v2.code = w.code AND v2.date = n.ndate
WHERE w.ld
""")
con.execute("""
CREATE TEMP TABLE ev_exact AS
SELECT w.date, w.code, w.is_st, n.ndate,
       (v2.code IS NOT NULL) AS has_next,
       coalesce(v2.ld_exact, FALSE) AS next_ld_exact,
       w.lead_ld_exact
FROM w JOIN nxt n ON w.date = n.date
LEFT JOIN v v2 ON v2.code = w.code AND v2.date = n.ndate
WHERE w.ld_exact
""")

rows = []
for wname, d0, d1 in [('A_2016至2026面板尾', '2016-01-01', '2026-10-29'),
                      ('B_原研究同窗', '2016-10-14', '2026-10-13')]:
    base = con.execute(f"""
      SELECT count(*) AS stock_days, sum(ld::INT) AS ld_cnt,
             sum(CASE WHEN ld AND is_st=1 THEN 1 ELSE 0 END) AS ld_st,
             sum(CASE WHEN ld AND is_st=0 THEN 1 ELSE 0 END) AS ld_nonst,
             sum(ld_exact::INT) AS ld_exact_cnt
      FROM v WHERE date BETWEEN '{d0}' AND '{d1}'
    """).df().iloc[0]
    cond = con.execute(f"""
      SELECT sum(has_next::INT) AS n_cal, sum(CASE WHEN has_next THEN next_ld::INT ELSE 0 END) AS n_cal_re,
             count(lead_ld) AS n_lead, sum(CASE WHEN lead_ld THEN 1 ELSE 0 END) AS n_lead_re,
             sum(CASE WHEN is_st=0 AND has_next THEN 1 ELSE 0 END) AS nonst_den,
             sum(CASE WHEN is_st=0 AND has_next THEN next_ld::INT ELSE 0 END) AS nonst_re,
             sum(CASE WHEN is_st=1 AND has_next THEN 1 ELSE 0 END) AS st_den,
             sum(CASE WHEN is_st=1 AND has_next THEN next_ld::INT ELSE 0 END) AS st_re
      FROM ev WHERE date BETWEEN '{d0}' AND '{d1}'
    """).df().iloc[0]
    nonst_base = con.execute(f"""
      SELECT sum(CASE WHEN is_st=0 THEN 1 ELSE 0 END) AS days,
             sum(CASE WHEN is_st=0 AND ld THEN 1 ELSE 0 END) AS lds,
             sum(CASE WHEN is_st=0 AND ld_exact THEN 1 ELSE 0 END) AS lds_x
      FROM v WHERE date BETWEEN '{d0}' AND '{d1}'
    """).df().iloc[0]
    exact = con.execute(f"""
      SELECT sum(has_next::INT) AS n_cal, sum(CASE WHEN has_next THEN next_ld_exact::INT ELSE 0 END) AS n_cal_re,
             count(lead_ld_exact) AS n_lead, sum(CASE WHEN lead_ld_exact THEN 1 ELSE 0 END) AS n_lead_re,
             sum(CASE WHEN is_st=0 AND has_next THEN 1 ELSE 0 END) AS nonst_den,
             sum(CASE WHEN is_st=0 AND has_next THEN next_ld_exact::INT ELSE 0 END) AS nonst_re
      FROM ev_exact t
      WHERE date BETWEEN '{d0}' AND '{d1}'
    """).df().iloc[0]
    rows.append(dict(窗口=wname, 起=d0, 止=d1,
                     股票日=int(base.stock_days), 跌停次数_近似=int(base.ld_cnt),
                     其中ST=int(base.ld_st), 其中非ST=int(base.ld_nonst),
                     基准率_近似=base.ld_cnt / base.stock_days,
                     跌停次数_精确取整=int(base.ld_exact_cnt),
                     基准率_精确取整=base.ld_exact_cnt / base.stock_days,
                     次日再跌停_日历分母=int(cond.n_cal), 次日再跌停_日历分子=int(cond.n_cal_re),
                     次日再跌停率_日历_近似=cond.n_cal_re / cond.n_cal,
                     次日再跌停_lead分母=int(cond.n_lead), 次日再跌停_lead分子=int(cond.n_lead_re),
                     次日再跌停率_lead_近似=cond.n_lead_re / cond.n_lead,
                     次日再跌停率_日历_精确=exact.n_cal_re / exact.n_cal,
                     次日再跌停率_lead_精确=exact.n_lead_re / exact.n_lead,
                     非ST股票日=int(nonst_base.days), 非ST跌停_近似=int(nonst_base.lds),
                     非ST基准率_近似=nonst_base.lds / nonst_base.days,
                     非ST基准率_精确=nonst_base.lds_x / nonst_base.days,
                     非ST次日再跌停率_近似=cond.nonst_re / cond.nonst_den,
                     非ST次日再跌停率_精确=exact.nonst_re / exact.nonst_den,
                     ST次日再跌停率_近似=cond.st_re / cond.st_den))
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT, 'F08_主板跌停基准.csv'), index=False)
print(summary.to_string(index=False))

yearly = con.execute("""
SELECT substr(date,1,4) AS y, count(*) AS stock_days, sum(ld::INT) AS ld_cnt_approx,
       sum(ld_exact::INT) AS ld_cnt_exact,
       sum(ld::INT)*1.0/count(*) AS base_rate_approx,
       sum(ld_exact::INT)*1.0/count(*) AS base_rate_exact
FROM v WHERE date >= '2016-01-01' GROUP BY 1 ORDER BY 1
""").df()
yearly.to_csv(os.path.join(OUT, 'F08_主板跌停分年.csv'), index=False)
print(yearly.to_string(index=False))