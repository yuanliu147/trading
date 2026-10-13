#!/usr/bin/env python3
# h2_pullback_volume.py
# H2 检验：趋势内缩量回调 vs 放量回调（事件研究）。口径冻结于 FINDINGS.md「H2 预注册」(2026-10-12)。
# 只读 <=2022-12-31 数据（训练集）。
import duckdb
import numpy as np
import pandas as pd
from h0_momentum_reversal import nw_t, limit_pct

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'
TRAIN_END = '2022-12-31'
COST_RT = 0.0015
SEED = 20261012
N_BOOT = 2000
MIN_PER_MONTH = 5

PX = """
  WITH px AS (
    SELECT v.thscode, v.date, v.close, v.volume,
           MAX(v.high)   OVER w20  AS high20,
           AVG(v.close)  OVER w150 AS ma150,
           AVG(v.volume) OVER w5   AS vol5,
           AVG(v.volume) OVER w20v AS vol20,
           COUNT(*) OVER (PARTITION BY v.thscode ORDER BY v.date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS age,
           LEAD(v.close, 1)  OVER wo AS c_in,
           LEAD(v.close, 21) OVER wo AS c_out
    FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
    WHERE v.date <= '{end}' AND s.exchange <> 'BJ'
    WINDOW w20  AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
           w150 AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 149 PRECEDING AND CURRENT ROW),
           w5   AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
           w20v AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
           wo   AS (PARTITION BY v.thscode ORDER BY v.date)),
  ev AS (
    SELECT px.*, close / high20 - 1.0 AS dd,
           LAG(close / high20 - 1.0) OVER (PARTITION BY thscode ORDER BY date) AS dd_prev,
           LAG(ma150, 20) OVER (PARTITION BY thscode ORDER BY date) AS ma150_l20
    FROM px)
"""

_PX = PX.format(end=TRAIN_END)

SQL_EVENTS = _PX + """
  SELECT date, thscode, vol5 / vol20 AS vratio,
         c_out / c_in - 1.0 AS fwd,
         c_in / close - 1.0 AS gap1
  FROM ev
  WHERE age >= 250 AND close > ma150 AND ma150 > ma150_l20
    AND dd BETWEEN -0.15 AND -0.05 AND dd_prev > -0.05
    AND vol20 > 0 AND vol5 > 0 AND c_in > 0 AND c_out > 0
"""

SQL_BENCH = _PX + """
  SELECT AVG(c_out / c_in - 1.0) AS bench_fwd, COUNT(*) AS bench_days
  FROM ev
  WHERE age >= 250 AND close > ma150 AND ma150 > ma150_l20
    AND c_in > 0 AND c_out > 0
"""


def boot_mean_ci(x, block, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n = len(x)
    nb = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        out[i] = x[idx].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    con = duckdb.connect(DB, read_only=True)
    ev = con.execute(SQL_EVENTS).df()
    bench_fwd, bench_days = con.execute(SQL_BENCH).fetchone()
    con.close()

    n0 = len(ev)
    # 入场日涨停封死剔除（买不进）
    lim = np.array([limit_pct(t, d) for t, d in zip(ev['thscode'], ev['date'])])
    ev = ev[ev['gap1'] < lim - 0.001].copy()
    ev['net'] = ev['fwd'] - COST_RT
    ev['month'] = pd.to_datetime(ev['date']).dt.to_period('M')
    ev['grp'] = np.where(ev['vratio'] < 0.7, 'A_lowvol', 'B_highvol')

    a, b = ev[ev.grp == 'A_lowvol'], ev[ev.grp == 'B_highvol']
    bench_net = bench_fwd - COST_RT

    # 月度聚合差值序列（主统计）
    m = ev.pivot_table(index='month', columns='grp', values='net', aggfunc='mean')
    cnt = ev.pivot_table(index='month', columns='grp', values='net', aggfunc='count')
    ok = (cnt >= MIN_PER_MONTH).all(axis=1)
    m = m[ok].dropna()
    diff = (m['A_lowvol'] - m['B_highvol']).values
    ci = boot_mean_ci(diff, block=3)
    t = nw_t(diff)

    print(f"事件总数 {n0} → 剔除入场日涨停后 {len(ev)}（缩量组 {len(a)} / 放量组 {len(b)}）")
    print(f"窗口 {ev['date'].min()} ~ {ev['date'].max()}")
    print(f"趋势内基准: 均值 {bench_net * 100:+.2f}%/20日（净），{bench_days:,} 趋势日")
    print(f"缩量组A: 净均值 {a['net'].mean() * 100:+.2f}%，胜率 {(a['net'] > 0).mean():.1%}，n={len(a)}")
    print(f"放量组B: 净均值 {b['net'].mean() * 100:+.2f}%，胜率 {(b['net'] > 0).mean():.1%}，n={len(b)}")
    print(f"A相对基准超额: {(a['net'].mean() - bench_net) * 100:+.2f}pp")
    print(f"月度差值(A-B): {len(diff)} 个月，均值 {diff.mean() * 100:+.2f}pp，"
          f"NW-t={t:+.2f}，bootstrap 95%CI=[{ci[0] * 100:+.2f},{ci[1] * 100:+.2f}]pp")

    print("\n缩量比五分位剂量反应（净，%/20日，n）:")
    ev['rq'] = pd.qcut(ev['vratio'].rank(method='first'), 5, labels=False)
    for qv, g in ev.groupby('rq'):
        print(f"  Q{qv + 1} [vratio {g['vratio'].min():.2f}~{g['vratio'].max():.2f}]: "
              f"{g['net'].mean() * 100:+.2f}%  (n={len(g)}, 胜率 {(g['net'] > 0).mean():.1%})")

    print("\n分年度（净均值 %/20日）:")
    ev['yr'] = pd.to_datetime(ev['date']).dt.year
    for yr, g in ev.groupby('yr'):
        ga, gb = g[g.grp == 'A_lowvol'], g[g.grp == 'B_highvol']
        print(f"  {yr}: A {ga['net'].mean() * 100:+.2f}% (n={len(ga)}) | "
              f"B {gb['net'].mean() * 100:+.2f}% (n={len(gb)})")

    ev.to_csv('/Users/xianyi/trading/analysis/h2_events.csv', index=False)

    print("\n===== H2 判决（对照预注册判据线） =====")
    passed = diff.mean() >= 0.015 and t >= 2 and a['net'].mean() > bench_net
    reverse = diff.mean() <= 0 and t <= -2
    if passed:
        v = '成立：缩量条件可采用'
    elif reverse:
        v = '反向证伪：缩量显著更差'
    else:
        v = '证据不足：缩量条件不采用'
    print(f"  均值差 {diff.mean() * 100:+.2f}pp（线 1.5pp）, NW-t {t:+.2f}（线 2）, "
          f"A>基准 {a['net'].mean() > bench_net}")
    print(f"  判决: {v}")


if __name__ == '__main__':
    main()
