#!/usr/bin/env python3
# h1_market_regime.py
# H1(重定位)：市况裁判（沪深300>200dma × 广度>=50%）对全宇宙等权未来20日收益的条件区分度。
# 口径冻结于 FINDINGS.md「H1 预注册（重定位）」(2026-10-12)。训练集 <=2022-12-31。
import duckdb
import json
import datetime as dt
import numpy as np
import pandas as pd

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'
TRAIN_END = dt.date(2022, 12, 31)
COST_RT = 0.0015
SEED = 20261012
N_BOOT = 2000

SQL = """
  WITH px AS (
    SELECT v.thscode, v.date, v.close,
           AVG(v.close) OVER w150 AS ma150,
           COUNT(*) OVER (PARTITION BY v.thscode ORDER BY v.date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS age,
           LEAD(v.close, 1)  OVER wo AS c_in,
           LEAD(v.close, 21) OVER wo AS c_out
    FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
    WHERE v.date <= '{end}' AND s.exchange <> 'BJ'
    WINDOW w150 AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 149 PRECEDING AND CURRENT ROW),
           wo   AS (PARTITION BY v.thscode ORDER BY v.date))
  SELECT date,
         COUNT(*) FILTER (age >= 150) AS n_b,
         COUNT(*) FILTER (age >= 150 AND close > ma150) AS n_up,
         AVG(c_out / c_in - 1.0) FILTER (age >= 250 AND c_in > 0 AND c_out > 0) AS ew_fwd
  FROM px GROUP BY date ORDER BY date
""".format(end=TRAIN_END)


def load_index():
    wu = json.load(open('/tmp/hs300_warmup.json'))['data']['item']
    fix = lambda ms: (dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc) + dt.timedelta(days=1)).date()
    wu_df = pd.DataFrame([{'date': fix(it['date_ms']), 'close': it['close_price']} for it in wu])
    loc = pd.read_csv('/Users/xianyi/trading/analysis/hs300_daily.csv', parse_dates=['date'])[['date', 'close']]
    loc['date'] = loc['date'].dt.date
    idx = (pd.concat([wu_df, loc]).drop_duplicates('date').sort_values('date').reset_index(drop=True))
    idx = idx[idx['date'] <= TRAIN_END].copy()
    idx['ma200'] = idx['close'].rolling(200).mean()
    idx['A'] = idx['close'] > idx['ma200']
    return idx[['date', 'A', 'ma200']]


def main():
    idx = load_index()
    con = duckdb.connect(DB, read_only=True)
    daily = con.execute(SQL).df()
    con.close()
    daily['date'] = pd.to_datetime(daily['date']).dt.date
    daily['breadth'] = daily['n_up'] / daily['n_b']

    df = (daily.merge(idx, on='date', how='inner')
               .dropna(subset=['ew_fwd', 'ma200'])
               .query('n_b >= 1000')
               .sort_values('date').reset_index(drop=True))
    obs = df.iloc[::5].copy()          # 周频观测（相位冻结：iloc[::5]）
    obs['net'] = obs['ew_fwd'] - COST_RT
    obs['B'] = obs['breadth'] >= 0.5
    obs['cell'] = np.select([(obs['A'] & obs['B']), (obs['A'] & ~obs['B']),
                             (~obs['A'] & obs['B']), (~obs['A'] & ~obs['B'])],
                            ['A+B+', 'A+B-', 'A-B+', 'A-B-'], default='NA')
    obs['month'] = pd.to_datetime(obs['date']).dt.to_period('M')

    print(f"窗口 {obs['date'].iloc[0]} ~ {obs['date'].iloc[-1]}，周频观测 {len(obs)} 个")
    print("\n四象限（净，未来20日全宇宙等权）：")
    for c in ['A+B+', 'A+B-', 'A-B+', 'A-B-']:
        g = obs[obs.cell == c]['net']
        if len(g):
            print(f"  {c}: n={len(g):3d}（时间占比 {len(g) / len(obs):.1%}） "
                  f"均值 {g.mean() * 100:+.2f}%/20日 中位 {g.median() * 100:+.2f}% "
                  f"年化等效 {g.mean() * 12.6 * 100:+.1f}%")

    pp = obs[obs.cell == 'A+B+']['net'].mean()
    mm = obs[obs.cell == 'A-B-']['net'].mean()
    diff = pp - mm

    # 月块 bootstrap（3个月循环块），按月聚合两单元的 sum/cnt 后向量化
    months = obs['month'].unique()
    stat = {c: obs[obs.cell == c].groupby('month')['net'].agg(['sum', 'count'])
            for c in ['A+B+', 'A-B-']}
    rng = np.random.default_rng(SEED)
    M = len(months)
    nb = int(np.ceil(M / 3))
    draws = rng.integers(0, M, (N_BOOT, nb))
    diffs = np.full(N_BOOT, np.nan)
    msum = {c: stat[c]['sum'].reindex(months, fill_value=0).values for c in stat}
    mcnt = {c: stat[c]['count'].reindex(months, fill_value=0).values for c in stat}
    for i in range(N_BOOT):
        midx = np.concatenate([(np.arange(s, s + 3)) % M for s in draws[i]])
        spp = msum['A+B+'][midx].sum()
        npp = mcnt['A+B+'][midx].sum()
        smm = msum['A-B-'][midx].sum()
        nmm = mcnt['A-B-'][midx].sum()
        if npp > 0 and nmm > 0:
            diffs[i] = spp / npp - smm / nmm
    diffs = diffs[~np.isnan(diffs)]
    t_boot = diffs.mean() / diffs.std(ddof=1)
    ci = np.percentile(diffs, [2.5, 97.5])

    print(f"\nA+B+ − A−B−: 均值差 {diff * 100:+.2f}pp/20日, t_boot={t_boot:+.2f}, "
          f"95%CI=[{ci[0] * 100:+.2f},{ci[1] * 100:+.2f}]pp, P(差<=0)={(diffs <= 0).mean():.3f}")

    print("\n年份×象限 时间占比（描述性）:")
    obs['yr'] = pd.to_datetime(obs['date']).dt.year
    tab = obs.pivot_table(index='yr', columns='cell', values='net', aggfunc='size', fill_value=0)
    print((tab.T / tab.sum(axis=1)).T.round(2).to_string())

    obs.to_csv('/Users/xianyi/trading/analysis/h1_regime_obs.csv', index=False)

    print("\n===== H1 判决（对照预注册判据线） =====")
    if diff >= 0.015 and t_boot >= 2 and pp > 0:
        v = '成立：市况裁判可采用（全绿环境多头净期望为正且显著优于全红）'
    elif diff <= 0 and t_boot <= -2:
        v = '反向证伪：全绿显著差于全红（裁判方向反了）'
    else:
        v = '证据不足：市况裁判不采用'
    print(f"  均值差 {diff * 100:+.2f}pp（线 1.5pp）, t_boot {t_boot:+.2f}（线 2）, "
          f"A+B+净均值 {pp * 100:+.2f}%（线 >0）")
    print(f"  判决: {v}")


if __name__ == '__main__':
    main()
