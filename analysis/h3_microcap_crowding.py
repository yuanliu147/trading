#!/usr/bin/env python3
# h3_microcap_crowding.py
# H3：微盘拥挤度裁判——拥挤度 vs 未来60日微盘超额（Spearman），及 0/1 规避规则的回撤改善。
# 口径冻结于 FINDINGS.md「H3 预注册」(2026-10-12)。训练集 <=2022-12-31。
import duckdb
import numpy as np
import pandas as pd

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'
TRAIN_END = '2022-12-31'
SEED = 20261012
N_BOOT = 2000

SQL = """
  WITH px AS (
    SELECT v.thscode, v.date,
           v.close / LAG(v.close) OVER w - 1.0 AS ret1,
           LAG(v.amount) OVER w AS amount_prev,
           v.volume / NULLIF(AVG(v.volume) OVER wv, 0) AS vr,
           COUNT(*) OVER (PARTITION BY v.thscode ORDER BY v.date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS age
    FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
    WHERE v.date <= '{end}' AND s.exchange <> 'BJ'
    WINDOW w  AS (PARTITION BY v.thscode ORDER BY v.date),
           wv AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING)),
  ranked AS (
    SELECT *, PERCENT_RANK() OVER (PARTITION BY date ORDER BY amount_prev) AS amt_pct
    FROM px WHERE age >= 250 AND amount_prev IS NOT NULL)
  SELECT date,
         AVG(ret1) AS all_ret,
         AVG(vr) AS all_vr,
         AVG(ret1) FILTER (amt_pct < 0.2) AS micro_ret,
         AVG(vr) FILTER (amt_pct < 0.2) AS micro_vr,
         COUNT(*) FILTER (amt_pct < 0.2) AS n_micro
  FROM ranked GROUP BY date ORDER BY date
""".format(end=TRAIN_END)


def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    con = duckdb.connect(DB, read_only=True)
    d = con.execute(SQL).df()
    con.close()
    d = d.dropna(subset=['micro_ret', 'all_ret']).sort_values('date').reset_index(drop=True)

    ex = d['micro_ret'] - d['all_ret']
    d['ex'] = ex
    d['ex60'] = ex.rolling(60).sum()
    sig20 = ex.rolling(20).std()
    vr = (d['micro_vr'] / d['all_vr']).rolling(5).mean()
    d['C'] = sig20 * vr
    d['fwd60'] = ex.rolling(60).sum().shift(-60)          # t+1..t+60 的未来超额
    d['micro_nav'] = (1 + d['micro_ret']).cumprod()

    obs = d.iloc[::20].dropna(subset=['C', 'fwd60']).copy()
    rho = spearman(obs['C'], obs['fwd60'])
    rng = np.random.default_rng(SEED)
    x = obs['C'].values
    y = obs['fwd60'].values
    n = len(obs)
    nb = int(np.ceil(n / 3))
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = rng.integers(0, n, nb)
        idx = np.concatenate([(np.arange(a, a + 3)) % n for a in s])[:n]
        boots[i] = spearman(x[idx], y[idx])
    p_one = float((boots >= 0).mean())
    ci = np.percentile(boots, [2.5, 97.5])

    print(f"日序列 {len(d)} 天；观测 {n} 期（窗口 {obs['date'].iloc[0]} ~ {obs['date'].iloc[-1]}）")
    print(f"Spearman ρ(C, 未来60日微盘超额) = {rho:+.3f}，bootstrap p(ρ>=0)={p_one:.3f}，"
          f"95%CI=[{ci[0]:+.3f},{ci[1]:+.3f}]")

    # 规则化择时（因果：当日收盘判定，次日生效）
    q80 = d['C'].rolling(756).quantile(0.8)
    avoid = (d['C'] > q80).astype(float)
    pos = 1 - avoid.shift(1).fillna(0)                     # 次日仓位
    sw = pos.diff().abs().fillna(0) * 0.00075              # 单边 0.075%/次
    strat = pos * d['micro_ret'] - sw
    rule_ok = q80.notna()
    nav_s = (1 + strat[rule_ok]).cumprod()
    nav_b = (1 + d.loc[rule_ok, 'micro_ret']).cumprod()
    dd_s = float((nav_s / nav_s.cummax() - 1).min())
    dd_b = float((nav_b / nav_b.cummax() - 1).min())
    impr = (abs(dd_b) - abs(dd_s)) / abs(dd_b)
    print(f"\n规则窗 {d.loc[rule_ok, 'date'].iloc[0]} ~ {d.loc[rule_ok, 'date'].iloc[-1]}，"
          f"规避天数占比 {(1 - pos[rule_ok]).mean():.1%}，切换 {int(pos[rule_ok].diff().abs().sum() / 2)} 次往返")
    print(f"微盘买入持有: maxDD={dd_b:.2%} | 拥挤择时: maxDD={dd_s:.2%} | 回撤改善 {impr:.1%}")

    obs.to_csv('/Users/xianyi/trading/analysis/h3_obs.csv', index=False)

    print("\n===== H3 判决（对照预注册判据线） =====")
    if rho <= -0.15 and p_one < 0.05 and impr >= 0.25:
        v = '成立：微盘拥挤度裁判可采用'
    elif rho >= 0 or p_one >= 0.2:
        v = '证伪'
    else:
        v = '证据不足：不采用'
    print(f"  ρ={rho:+.3f}（线 ≤-0.15）, p={p_one:.3f}（线 <0.05）, 回撤改善 {impr:.1%}（线 ≥25%）")
    print(f"  判决: {v}")


if __name__ == '__main__':
    main()
