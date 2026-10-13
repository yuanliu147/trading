#!/usr/bin/env python3
# h4_sector_sponsorship.py
# H4：板块主线资金赞助——主线板块（相对强度前20%）内个股 vs 非主线板块个股的未来20日收益。
# 口径冻结于 FINDINGS.md「H4 预注册」(2026-10-12)。训练集 <=2022-12-31。
import duckdb
import json
import numpy as np
import pandas as pd
from h0_momentum_reversal import nw_t

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'
TRAIN_END = '2022-12-31'
COST_RT = 0.0015
SEED = 20261012
N_BOOT = 2000

SQL = """
  WITH cal AS (SELECT DISTINCT date FROM v_daily WHERE date <= '{end}'),
  ci AS (SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS rn FROM cal),
  samp AS (SELECT date FROM ci WHERE rn % 5 = 0),
  px AS (
    SELECT v.thscode, v.date,
           v.close / LAG(v.close, 20) OVER w - 1.0 AS r20,
           LEAD(v.close, 1)  OVER w AS c_in,
           LEAD(v.close, 21) OVER w AS c_out,
           COUNT(*) OVER (PARTITION BY v.thscode ORDER BY v.date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS age
    FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
    WHERE v.date <= '{end}' AND s.exchange <> 'BJ'
    WINDOW w AS (PARTITION BY v.thscode ORDER BY v.date))
  SELECT p.date, p.thscode, p.r20, p.c_out / p.c_in - 1.0 AS fwd
  FROM px p JOIN samp ON p.date = samp.date
  WHERE p.age >= 250 AND p.r20 IS NOT NULL AND p.c_in > 0 AND p.c_out > 0
""".format(end=TRAIN_END)


def boot_ci_t(x, months, block=3, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    um = np.unique(months)
    M = len(um)
    sums = np.array([x[months == m].sum() for m in um])
    cnts = np.array([(months == m).sum() for m in um])
    nb = int(np.ceil(M / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, M, nb)
        idx = np.concatenate([(np.arange(a, a + block)) % M for a in s])
        out[i] = sums[idx].sum() / cnts[idx].sum()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), \
        float(out.mean() / out.std(ddof=1)), float((out <= 0).mean())


def main():
    smap = json.load(open('/Users/xianyi/trading/analysis/sector_map.json'))
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(SQL).df()
    con.close()
    n_univ = len(df)
    df['sector'] = df['thscode'].map(smap)
    df = df.dropna(subset=['sector']).reset_index(drop=True)
    print(f"宇宙（采样期×股）{n_univ:,}，已映射 {len(df):,}（{len(df) / n_univ:.1%}）")
    df['net'] = df['fwd'] - COST_RT

    rows = []
    sec_rows = []
    for d, g in df.groupby('date'):
        sec = g.groupby('sector').agg(r20m=('r20', 'mean'), fwdm=('net', 'mean'), n=('r20', 'size'))
        sec = sec[sec['n'] >= 10]
        if len(sec) < 25:
            continue
        k = int(np.ceil(len(sec) * 0.2))
        main_secs = set(sec['r20m'].nlargest(k).index)
        gm = g['sector'].isin(main_secs)
        a, b = g.loc[gm, 'net'], g.loc[~gm, 'net']
        rows.append(dict(date=d, diff=a.mean() - b.mean(), a=a.mean(), b=b.mean(),
                         bench=g['net'].mean(), na=len(a), nb=len(b)))
        # 板块层动量（描述性）：强度五分位 top-bottom 的板块等权未来收益
        sec['rq'] = pd.qcut(sec['r20m'].rank(method='first'), 5, labels=False)
        srow = {'date': d}
        for qv in range(5):
            srow[f'q{qv}'] = sec.loc[sec['rq'] == qv, 'fwdm'].mean()
        sec_rows.append(srow)
        # 剂量反应：板块强度五分位 → 成员个股净收益
        sec_rank = sec['r20m'].rank(pct=True)
        g2 = g.copy()
        g2['spct'] = g2['sector'].map(sec_rank)
        for qv in range(5):
            lo, hi = qv / 5, (qv + 1) / 5
            sub = g2[(g2['spct'] > lo) & (g2['spct'] <= hi)]['net']
            rows[-1][f'stq{qv}'] = sub.mean() if len(sub) else np.nan

    res = pd.DataFrame(rows)
    months = pd.to_datetime(res['date']).dt.to_period('M').values
    diff = res['diff'].values
    lo, hi, tb, p0 = boot_ci_t(diff, months)
    print(f"\n窗口 {res['date'].iloc[0]} ~ {res['date'].iloc[-1]}，{len(res)} 期")
    print(f"主线组均 {int(res['na'].mean())} 只/期，非主线组均 {int(res['nb'].mean())} 只/期")
    print(f"主线组净均值 {res['a'].mean() * 100:+.2f}%/20日 | 非主线 {res['b'].mean() * 100:+.2f}% "
          f"| 全宇宙基准 {res['bench'].mean() * 100:+.2f}%")
    print(f"差值(主线-非主线): {diff.mean() * 100:+.2f}pp/20日, NW-t={nw_t(diff):+.2f}, "
          f"t_boot={tb:+.2f}, 95%CI=[{lo * 100:+.2f},{hi * 100:+.2f}]pp, P(<=0)={p0:.3f}")

    print("\n板块强度五分位 → 成员个股净收益（%/20日，剂量反应，描述性）:")
    for qv in range(5):
        print(f"  Q{qv + 1}: {res[f'stq{qv}'].mean() * 100:+.2f}%")

    sres = pd.DataFrame(sec_rows)
    ls = sres['q4'] - sres['q0']
    lo2, hi2, tb2, _ = boot_ci_t(ls.values, months[:len(sres)])
    print(f"\n板块层动量多空（描述性）: topQ−botQ 净均值 {ls.mean() * 100:+.2f}%/20日, "
          f"NW-t={nw_t(ls.values):+.2f}, t_boot={tb2:+.2f}")

    res['yr'] = pd.to_datetime(res['date']).dt.year
    print("\n分年度差值（pp/20日）:")
    for yr, g in res.groupby('yr'):
        print(f"  {yr}: {g['diff'].mean() * 100:+.2f}pp (n={len(g)})")

    res.to_csv('/Users/xianyi/trading/analysis/h4_periods.csv', index=False)

    print("\n===== H4 判决（对照预注册判据线） =====")
    if diff.mean() >= 0.01 and tb >= 2 and res['a'].mean() > res['bench'].mean():
        v = '成立：板块主线赞助可采用'
    elif diff.mean() <= 0 and tb <= -2:
        v = '反向证伪'
    else:
        v = '证据不足：板块主线条件不采用'
    print(f"  均值差 {diff.mean() * 100:+.2f}pp（线 1.0pp）, t_boot {tb:+.2f}（线 2）, "
          f"主线>基准 {res['a'].mean() > res['bench'].mean()}")
    print(f"  判决: {v}")


if __name__ == '__main__':
    main()
