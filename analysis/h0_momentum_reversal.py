#!/usr/bin/env python3
# h0_momentum_reversal.py
# H0 判决检验：动量 vs 反转 五分位分层多空。主口径 h=20；h∈{5,10,60} 为冻结的稳健性对照。
# 口径冻结于 FINDINGS.md「H0 预注册」(2026-10-12)。只读 <=2022-12-31 数据（训练集）。
import duckdb
import numpy as np
import pandas as pd

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'
TRAIN_END = '2022-12-31'
COST_RT = 0.0015             # 双边往返成本（手续费+印花税+滑点）
HORIZONS = [20, 5, 10, 60]   # 20 = 主口径，其余为一次性稳健性对照（不择优）
MIN_NAMES = 200              # 每期最少标的数
SEED = 20261012
N_BOOT = 2000


def limit_pct(thscode, date):
    t = thscode.split('.')[0]
    if t[:3] in ('300', '301'):
        return 0.20 if str(date) >= '2020-08-24' else 0.10
    if t[:3] == '688':
        return 0.20
    return 0.10


def load(con, h):
    sql = f"""
      WITH cal AS (SELECT DISTINCT date FROM v_daily WHERE date <= '{TRAIN_END}'),
      ci AS (SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS rn FROM cal),
      rebal AS (SELECT date FROM ci WHERE rn % {h} = 0),
      px AS (
        SELECT v.thscode, v.date, v.close,
               LAG(v.close, {h}) OVER w AS c_back,
               LEAD(v.close, 1) OVER w AS c_in,
               LEAD(v.close, {h + 1}) OVER w AS c_out,
               COUNT(*) OVER (PARTITION BY v.thscode ORDER BY v.date
                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS age
        FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
        WHERE v.date <= '{TRAIN_END}' AND s.exchange <> 'BJ'
        WINDOW w AS (PARTITION BY v.thscode ORDER BY v.date))
      SELECT p.date, p.thscode,
             p.close / p.c_back - 1.0 AS mom,
             p.c_out  / p.c_in   - 1.0 AS fwd,
             p.c_in   / p.close  - 1.0 AS gap1
      FROM px p JOIN rebal r ON p.date = r.date
      WHERE p.c_back > 0 AND p.c_in > 0 AND p.c_out > 0 AND p.age >= 250
    """
    return con.execute(sql).df()


def nw_t(x, lags=None):
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    v = (e * e).mean()
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        v += 2 * w * (e[l:] * e[:-l]).mean()
    return x.mean() / np.sqrt(v / n) if v > 0 else np.nan


def metrics(r, ppy):
    r = np.asarray(r, float)
    n = len(r)
    eq = np.cumprod(1 + r)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    ann = float(eq[-1] ** (ppy / n) - 1)
    sd = r.std(ddof=1)
    return dict(n=n, ann=ann, vol=float(sd * np.sqrt(ppy)),
                sharpe=float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan,
                maxdd=dd, calmar=float(ann / abs(dd)) if dd < 0 else np.nan,
                t=float(nw_t(r)))


def boot_diff(a, b, ppy, block, n_boot=N_BOOT, seed=SEED):
    """循环块 bootstrap：每期均值差(a-b) 与 Calmar差(a-b) 的经验分布"""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    nb = int(np.ceil(n / block))
    md = np.empty(n_boot)
    cd = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        md[i] = a[idx].mean() - b[idx].mean()
        cd[i] = metrics(a[idx], ppy)['calmar'] - metrics(b[idx], ppy)['calmar']
    return md, cd


def run_horizon(df, h):
    ppy = 252 / h
    rows = []
    prev = {'mq5': set(), 'mq1': set(), 'rq1': set(), 'rq5': set()}
    qtab = []
    for d, g in df.groupby('date'):
        if len(g) < MIN_NAMES:
            continue
        g = g.copy()
        g['q'] = pd.qcut(g['mom'].rank(method='first'), 5, labels=False)
        lim = g['thscode'].map(lambda t: limit_pct(t, d))
        lu = (g['gap1'] >= (lim - 0.001)).values   # 入场日涨停封死→多头买不进
        ld = (g['gap1'] <= -(lim - 0.001)).values  # 入场日跌停封死→空头卖不出
        ew = float(g['fwd'].mean())
        qtab.append(g.groupby('q')['fwd'].mean())

        def leg(qv, excl, key):
            sub = g[(g['q'] == qv).values & ~excl]
            if len(sub) == 0:
                return np.nan, 0.0, 0
            turn = 1 - len(prev[key] & set(sub['thscode'])) / len(sub)
            prev[key] = set(sub['thscode'])
            return float(sub['fwd'].mean()), turn * COST_RT, len(sub)

        m_l, m_lc, n_m5 = leg(4, lu, 'mq5')
        m_s, m_sc, n_m1 = leg(0, ld, 'mq1')
        r_l, r_lc, n_r1 = leg(0, lu, 'rq1')
        r_s, r_sc, n_r5 = leg(4, ld, 'rq5')
        rows.append(dict(
            date=d, ew=ew, n=len(g),
            mom_ls=(m_l - m_s) - (m_lc + m_sc),
            rev_ls=(r_l - r_s) - (r_lc + r_sc),
            mom_ls_gross=m_l - m_s, rev_ls_gross=r_l - r_s,
            q5_exc=(m_l - m_lc) - ew, q1_exc=(r_l - r_lc) - ew,
            turn_cost=m_lc + m_sc + r_lc + r_sc,
            n_leg=min(n_m5, n_m1, n_r1, n_r5)))
    res = pd.DataFrame(rows).dropna(subset=['mom_ls', 'rev_ls']).reset_index(drop=True)
    mom, rev = res['mom_ls'].values, res['rev_ls'].values
    diff = mom - rev
    block = max(3, round(120 / h))
    md, cd = boot_diff(mom, rev, ppy, block)
    qt = pd.DataFrame(qtab).mean()
    summ = dict(
        h=h, ppy=ppy, periods=len(res),
        win_start=str(res['date'].iloc[0]), win_end=str(res['date'].iloc[-1]),
        names_mean=float(res['n'].mean()),
        mom=metrics(mom, ppy), rev=metrics(rev, ppy),
        mom_gross=metrics(res['mom_ls_gross'].values, ppy),
        rev_gross=metrics(res['rev_ls_gross'].values, ppy),
        q5_exc=metrics(res['q5_exc'].values, ppy),
        q1_exc=metrics(res['q1_exc'].values, ppy),
        diff_t=float(nw_t(diff)),
        diff_mean_bp=float(diff.mean() * 1e4),
        boot_mean_ci=(float(np.percentile(md, 2.5)) * 1e4, float(np.percentile(md, 97.5)) * 1e4),
        boot_calmar_ci=(float(np.percentile(cd, 2.5)), float(np.percentile(cd, 97.5))),
        boot_p_diff_le0=float((md <= 0).mean()),
        qtab=qt.round(4).to_dict(),
        turn_cost_mean_bp=float(res['turn_cost'].mean() * 1e4))
    return res, summ


def fmt_m(m):
    return (f"ann={m['ann']:+.2%} vol={m['vol']:.2%} sharpe={m['sharpe']:+.2f} "
            f"maxDD={m['maxdd']:.2%} calmar={m['calmar']:+.2f} t={m['t']:+.2f}")


def main():
    con = duckdb.connect(DB, read_only=True)
    summaries = []
    all_series = []
    for h in HORIZONS:
        df = load(con, h)
        res, s = run_horizon(df, h)
        summaries.append(s)
        res.insert(0, 'h', h)
        all_series.append(res)
    con.close()
    pd.concat(all_series).to_csv('/Users/xianyi/trading/analysis/h0_period_returns.csv', index=False)

    for s in summaries:
        print(f"\n===== h={s['h']} （{'主口径' if s['h'] == 20 else '稳健性对照'}） "
              f"窗口 {s['win_start']} ~ {s['win_end']}，{s['periods']} 期，"
              f"均{s['names_mean']:.0f}只/期，平均每期双边成本 {s['turn_cost_mean_bp']:.0f}bp =====")
        print(f"  动量LS(净):   {fmt_m(s['mom'])}")
        print(f"  反转LS(净):   {fmt_m(s['rev'])}")
        print(f"  动量LS(毛):   {fmt_m(s['mom_gross'])}")
        print(f"  反转LS(毛):   {fmt_m(s['rev_gross'])}")
        print(f"  Q5多头超额(净,vs等权): {fmt_m(s['q5_exc'])}")
        print(f"  Q1多头超额(净,vs等权): {fmt_m(s['q1_exc'])}")
        print(f"  差值(动量-反转): mean={s['diff_mean_bp']:+.0f}bp/期, NW-t={s['diff_t']:+.2f}, "
              f"bootstrap均值差95%CI=[{s['boot_mean_ci'][0]:+.0f},{s['boot_mean_ci'][1]:+.0f}]bp, "
              f"Calmar差95%CI=[{s['boot_calmar_ci'][0]:+.2f},{s['boot_calmar_ci'][1]:+.2f}], "
              f"P(diff<=0)={s['boot_p_diff_le0']:.3f}")
        print(f"  五分位毛收益(Q1→Q5): {s['qtab']}")

    s20 = [s for s in summaries if s['h'] == 20][0]
    print("\n===== H0 判决（对照预注册判据线，h=20 净口径） =====")
    dt, mc, rc = s20['diff_t'], s20['mom']['calmar'], s20['rev']['calmar']
    if dt <= -2 and (rc - mc) >= 0.5:
        verdict = '趋势框架证伪（反转显著占优）→ 放弃趋势框架，转反转/均值回复框架重设计'
    elif dt >= 2 and mc >= 0.3:
        verdict = '趋势框架过第 0 关（动量显著占优且净 Calmar≥0.3）'
    else:
        verdict = '截面层面无判别力 → 框架价值须由准入/择时层证明'
    print(f"  diff_t={dt:+.2f}, 动量净Calmar={mc:+.2f}, 反转净Calmar={rc:+.2f}")
    print(f"  判决: {verdict}")
    print("  偏差提醒: 幸存者宇宙使最差组Q1收益被高估 → 偏差方向利于反转结论；"
          "若判反转胜须打折采信，若判动量胜为保守方向。")


if __name__ == '__main__':
    main()
