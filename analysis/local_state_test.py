#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# local_state_test.py
# 技术指标的【条件表现】：市场状态(ADX 趋势/震荡) × 信号类型(趋势跟随/摆动均值回归) 的未来相对收益
#   数据：本地 DuckDB 全A前复权日线(v_daily_qfq)，剔除北交所；幸存者宇宙
#   因果：指标只用 <=t 信息；状态用 t-1 的 ADX 判定；t=信号日，t+1 收盘入场（稳健性另给 t+1 开盘入场）
#   相对收益 = 个股 h 日收益 - 同期全A等权指数 h 日收益（市场调整；另给样本内等权基准）
#   状态切分：ADX>25 / <=20 为主口径，另切 波动状态、价在MA60上下、三个分期
#   显著性：日聚类 bootstrap 与月度聚类 t 检验；对照同状态随机日基准（匹配事件数）
#   自检：因果性（截断重算）与收益对齐（逐笔重算）在每次运行时都会执行并打印结果
import duckdb
import numpy as np
import pandas as pd
from math import erfc

DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'
OUT_CSV = '/Users/admin/Documents/缠论与技术指标_Research_20261014/local_indicator_state_test.csv'
MIN_DAYS = 1000          # 交易不足 1000 天的标的剔除
N_SAMPLE = 400           # 随机抽样股票数
SEED = 42
HORIZONS = (5, 20)       # 持有交易日数（自 t+1 入场算起）
BOOT = 1000              # 日聚类 bootstrap 次数
RAND = 200               # 随机日基准重复次数
ADX_TREND, ADX_RANGE = 25, 20   # >25 TREND，<=20 RANGE，20~25 作为缓冲带丢弃


def wilder(x, n):
    """Wilder 平滑：首值取前 n 个有效值均值，其后 v=(v*(n-1)+x_i)/n"""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    buf, prev = [], np.nan
    for i, v in enumerate(x):
        if not np.isfinite(v):
            continue
        if np.isnan(prev):
            buf.append(v)
            if len(buf) == n:
                prev = float(np.mean(buf))
                out[i] = prev
        else:
            prev = (prev * (n - 1) + v) / n
            out[i] = prev
    return out


def smooth_kdj(rsv, alpha=1.0 / 3):
    """KDJ 递归平滑，K/D 初值 50（国内通行 9,3,3 口径）"""
    k = np.full(len(rsv), np.nan)
    d = np.full(len(rsv), np.nan)
    kv = dv = 50.0
    for i, r in enumerate(rsv):
        if not np.isfinite(r):
            continue
        kv = (1 - alpha) * kv + alpha * r
        dv = (1 - alpha) * dv + alpha * kv
        k[i], d[i] = kv, dv
    return k, d


def add_ind(df):
    """仅用当根及之前 K 线计算全部指标"""
    c, h, l = df['close'], df['high'], df['low']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)

    atr = wilder(tr.values, 14)
    with np.errstate(invalid='ignore', divide='ignore'):
        pdi = 100 * wilder(pdm.values, 14) / atr
        ndi = 100 * wilder(ndm.values, 14) / atr
        dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
    df['adx'] = wilder(dx, 14)
    df['atrp'] = atr / c.values

    chg = c.diff()
    with np.errstate(invalid='ignore', divide='ignore'):
        df['rsi'] = 100 - 100 / (1 + wilder(chg.clip(lower=0).values, 14)
                                 / wilder((-chg).clip(lower=0).values, 14))

    llv = l.rolling(9, min_periods=9).min()
    hhv = h.rolling(9, min_periods=9).max()
    rng = (hhv - llv).replace(0, np.nan)
    rsv = ((c - llv) / rng * 100).fillna(50.0)
    df['k'], df['d'] = smooth_kdj(rsv.values)

    df['ma20'] = c.rolling(20, min_periods=20).mean()
    df['ma60'] = c.rolling(60, min_periods=60).mean()
    df['bblow'] = df['ma20'] - 2 * c.rolling(20, min_periods=20).std(ddof=0)
    df['volmed'] = df['atrp'].rolling(250, min_periods=120).median()
    return df


def signals(df):
    """全部信号在 t 日收盘确认，只用 <=t 数据"""
    c = df['close']
    ev = {}
    ev['S1_MA20xMA60'] = (df['ma20'].shift(1) <= df['ma60'].shift(1)) & (df['ma20'] > df['ma60'])
    ev['S2_CLOSE_x_MA60'] = (c.shift(1) <= df['ma60'].shift(1)) & (c > df['ma60'])
    ev['S3_RSI14_x30'] = (df['rsi'].shift(1) < 30) & (df['rsi'] >= 30)
    ev['S4_KDJ_lowgold'] = ((df['k'].shift(1) <= df['d'].shift(1)) & (df['k'] > df['d'])
                            & (pd.concat([df['k'].shift(1), df['k']], axis=1).min(axis=1) < 20))
    ev['S5_BB_lower_reclaim'] = (c.shift(1) < df['bblow'].shift(1)) & (c >= df['bblow'])
    return pd.DataFrame({k: v.fillna(False).values for k, v in ev.items()}, index=df.index)


def fwd_at(P, h, off):
    """信号在第 r 行、入场在第 r+off 行时，自入场起的 h 日收益"""
    N = P.shape[0]
    out = np.full_like(P, np.nan)
    if N > h + off:
        out[:N - h - off] = P[h + off:] / P[off:N - h] - 1
    return out


def daily_cluster_boot(x, d, rng, B=BOOT):
    """日聚类 bootstrap：按事件日重抽样（同日事件整体进出）"""
    ud, inv = np.unique(d, return_inverse=True)
    w = np.bincount(inv, weights=x)
    n = np.bincount(inv).astype(float)
    nu = len(ud)
    boot = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, nu, nu)
        boot[b] = w[pick].sum() / n[pick].sum()
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(np.clip(p, 1.0 / B, 1.0))


def load_market(con):
    """全A(非北交所)等权指数：close-to-close 与 open/prev_close 两组日收益"""
    m = con.execute("""
        WITH r AS (
          SELECT v.thscode, v.date, v.open, v.close,
                 lag(v.close) OVER w AS pc, lag(v.date) OVER w AS pd
          FROM v_daily_qfq v JOIN dim_symbol s USING(thscode)
          WHERE s.exchange <> 'BJ'
          WINDOW w AS (PARTITION BY v.thscode ORDER BY v.date)
        )
        SELECT date,
               avg(CASE WHEN date_diff('day',pd,date) <= 7 THEN close/pc - 1 END) AS mret,
               avg(CASE WHEN date_diff('day',pd,date) <= 7 THEN open/pc - 1 END)  AS oret
        FROM r GROUP BY date ORDER BY date
    """).df()
    m['mret'] = m['mret'].fillna(0.0)
    m['midx'] = (1 + m['mret']).cumprod()
    return m


def self_test_causal(con):
    """无未来函数自检：截断样本重算，指标与信号须逐点一致"""
    codes = con.execute("""
        SELECT v.thscode FROM v_daily_qfq v JOIN dim_symbol s USING(thscode) WHERE s.exchange <> 'BJ'
        GROUP BY 1 HAVING count(*) > 1500 ORDER BY 1 LIMIT 5
    """).df()['thscode'].tolist()
    n, bad = 0, 0
    for code in codes:
        g = con.execute("""SELECT date, open, high, low, close FROM v_daily_qfq
                           WHERE thscode = ? ORDER BY date""", [code]).df()
        full = add_ind(g.copy())
        sfull = signals(full)
        for cut in (800, 1400, 2000):
            if cut >= len(g):
                continue
            part = add_ind(g.iloc[:cut].copy())
            sp = signals(part)
            for col in ('adx', 'rsi', 'k', 'd', 'atrp', 'ma20', 'ma60', 'bblow'):
                a, b = full[col].iloc[cut - 1], part[col].iloc[cut - 1]
                bad += not ((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9)
                n += 1
            for col in sfull.columns:
                bad += bool(sfull[col].iloc[cut - 1]) != bool(sp[col].iloc[cut - 1])
                n += 1
    return n, bad


def self_test_align(K, raw, rel, mk, SIG, INV, CX, POSX, h, rng):
    """收益对齐自检：矩阵化事件收益 vs 朴素逐笔重算（窗口含停牌的样本跳过）"""
    n, bad = 0, 0
    for M in SIG.values():
        ix = np.flatnonzero(M.ravel())
        for f in rng.choice(ix, size=min(30, len(ix)), replace=False) if len(ix) else []:
            r, i = divmod(int(f), K)
            j = INV[r, i]
            if j < 0 or not np.isfinite(raw[r, i]) or j + h + 1 >= len(CX[i]):
                continue
            if POSX[i][j + 1 + h] != r + 1 + h:
                continue
            man = CX[i][j + 1 + h] / CX[i][j + 1] - 1
            n += 1
            bad += abs(man - raw[r, i]) > 1e-9 or abs(man - mk[r] - rel[r, i]) > 1e-9
    return n, bad


def main():
    con = duckdb.connect(DB, read_only=True)
    mkt = load_market(con)
    cal = pd.DatetimeIndex(pd.to_datetime(mkt['date']))
    N = len(cal)
    midx = mkt['midx'].values
    oret = mkt['oret'].fillna(0.0).values

    uni = con.execute("""
        WITH d AS (SELECT v.thscode, count(*) nd FROM v_daily_qfq v JOIN dim_symbol s USING(thscode)
                   WHERE s.exchange <> 'BJ' GROUP BY 1)
        SELECT thscode FROM d WHERE nd >= ? ORDER BY thscode
    """, [MIN_DAYS]).df()['thscode'].tolist()
    np.random.seed(SEED)
    pick = sorted(np.random.choice(uni, size=min(N_SAMPLE, len(uni)), replace=False).tolist())
    print(f'全A非北交所可抽样 {len(uni)} 只，随机抽 {len(pick)} 只（seed={SEED}）；交易日 {N} 天 '
          f'{cal[0].date()} ~ {cal[-1].date()}')

    con.register('pick_df', pd.DataFrame({'thscode': pick}))
    px = con.execute("""
        SELECT v.thscode, v.date, v.open, v.high, v.low, v.close
        FROM v_daily_qfq v JOIN pick_df p USING(thscode)
        ORDER BY v.thscode, v.date
    """).df()

    K = len(pick)
    C = np.full((N, K), np.nan)
    O = np.full((N, K), np.nan)
    INV = np.full((N, K), -1, int)          # 日历行 → 个股自身行号
    CX, POSX = [], []                       # 个股收盘序列 / 其日历行号
    LAST = np.zeros(K, int)
    STATE = np.zeros((N, K), np.int8)       # 0 缓冲/未知 1 TREND 2 RANGE
    VOLST = np.zeros((N, K), np.int8)       # 0 未知 1 高波动 2 低波动
    DIRS = np.zeros((N, K), np.int8)        # 0 未知 1 收盘在 MA60 上 2 之下
    SIG, names = {}, None
    groups = {c: g.reset_index(drop=True) for c, g in px.groupby('thscode', sort=False)}
    for i, code in enumerate(pick):
        g = add_ind(groups[code])
        pos = cal.get_indexer(pd.DatetimeIndex(pd.to_datetime(g['date'])))
        C[pos, i] = g['close'].values
        O[pos, i] = g['open'].values
        INV[pos, i] = np.arange(len(g))
        CX.append(g['close'].values)
        POSX.append(pos)
        LAST[i] = pos[-1]

        adx_p = g['adx'].shift(1).values          # t-1 状态，避免同日泄漏
        atrp, vmed = g['atrp'].shift(1).values, g['volmed'].shift(1).values
        st = np.zeros(len(g), np.int8)
        st[adx_p > ADX_TREND] = 1
        st[adx_p <= ADX_RANGE] = 2
        STATE[pos, i] = st
        vs = np.zeros(len(g), np.int8)
        with np.errstate(invalid='ignore'):
            hi = atrp > vmed
            vs[hi] = 1
            vs[np.isfinite(atrp) & np.isfinite(vmed) & ~hi] = 2
        VOLST[pos, i] = vs
        c_p, ma60_p = g['close'].shift(1).values, g['ma60'].shift(1).values
        ds = np.zeros(len(g), np.int8)
        with np.errstate(invalid='ignore'):
            up = c_p > ma60_p
            ds[up] = 1
            ds[np.isfinite(c_p) & np.isfinite(ma60_p) & ~up] = 2
        DIRS[pos, i] = ds

        sg = signals(g)
        if names is None:
            names = list(sg.columns)
            SIG = {k: np.zeros((N, K), bool) for k in names}
        for k in names:
            SIG[k][pos, i] = sg[k].values

    # 前复权收盘前向填充（停牌沿用上一收盘），退市后不再外推
    Cff = np.array(pd.DataFrame(C, index=cal).ffill().values, dtype=float)
    for i in range(K):
        Cff[LAST[i] + 1:, i] = np.nan

    rng = np.random.default_rng(SEED + 1)
    drow = np.repeat(cal.values, K)
    ERAS = [('E1_2016_2019', '2016-10-13', '2019-12-31'), ('E2_2020_2022', '2020-01-01', '2022-12-31'),
            ('E3_2023_2026', '2023-01-01', '2026-10-13')]
    era_row = {name: (cal >= a) & (cal <= b) for name, a, b in ERAS}
    rows = []

    def month_stats(x, d):
        """月度聚类统计：把每月事件均值当独立观测，剔除横截面与日内相关"""
        if len(x) < 60:
            return {}
        ms = pd.Series(x).groupby(pd.DatetimeIndex(d).to_period('M')).mean()
        if len(ms) < 8 or ms.std(ddof=1) <= 0:
            return {}
        t_m = ms.mean() / (ms.std(ddof=1) / np.sqrt(len(ms)))
        return dict(n_months=len(ms), month_mean_bp=ms.mean() * 1e4, month_pos_frac=(ms > 0).mean(),
                    month_p10_bp=np.percentile(ms, 10) * 1e4, month_p90_bp=np.percentile(ms, 90) * 1e4,
                    t_month=t_m, p_month=erfc(abs(t_m) / np.sqrt(2)))

    def rec_of(grp, sig, lab, h, era_lab, ix, x_all, raw, drow, pool, rel_o):
        """一个格子(信号×状态×持有期)的事件统计"""
        n_ev = len(ix)
        d = drow[ix] if n_ev else np.array([])
        r = dict(group=grp, signal=sig, state=lab, era=era_lab, horizon=h, n_events=n_ev,
                 n_dates=len(np.unique(d)) if n_ev else 0, mean_raw_bp=np.nan, mean_rel_bp=np.nan,
                 mean_rel_bp_samp=np.nan, median_rel_bp=np.nan, hit_rate=np.nan, t_stat=np.nan,
                 p_boot=np.nan, rand_mean_bp=np.nan, rand_p5_bp=np.nan, rand_p95_bp=np.nan,
                 p_rand=np.nan, excess_vs_rand_bp=np.nan, mean_rel_bp_open=np.nan,
                 n_months=0, month_mean_bp=np.nan, month_pos_frac=np.nan, month_p10_bp=np.nan,
                 month_p90_bp=np.nan, t_month=np.nan, p_month=np.nan)
        if n_ev < 20 or pool is None:
            return r
        x, xr = x_all[ix], raw.ravel()[ix]
        sd = x.std(ddof=1)
        r.update(mean_raw_bp=xr.mean() * 1e4, mean_rel_bp=x.mean() * 1e4, median_rel_bp=np.median(x) * 1e4,
                 hit_rate=(x > 0).mean(),
                 t_stat=x.mean() / (sd / np.sqrt(n_ev)) if sd > 0 else np.nan,
                 p_boot=daily_cluster_boot(x, d, rng))
        rmean = np.array([x_all[pool[rng.integers(0, len(pool), n_ev)]].mean() for _ in range(RAND)])
        r.update(rand_mean_bp=rmean.mean() * 1e4, rand_p5_bp=np.percentile(rmean, 5) * 1e4,
                 rand_p95_bp=np.percentile(rmean, 95) * 1e4, p_rand=(rmean >= x.mean()).mean(),
                 excess_vs_rand_bp=(x.mean() - rmean.mean()) * 1e4,
                 mean_rel_bp_open=np.nanmean(rel_o.ravel()[ix]) * 1e4)
        r.update(month_stats(x, d))
        return r

    def combo(a, av, b, bv):
        return ((a == av) & (b == bv)).astype(np.int8)

    GROUPS = [('ADX', [('TREND', STATE, 1), ('RANGE', STATE, 2)]),
              ('VOL', [('HIGH_VOL', VOLST, 1), ('LOW_VOL', VOLST, 2)]),
              ('XSV', [('TREND_HI', combo(STATE, 1, VOLST, 1), 1), ('TREND_LO', combo(STATE, 1, VOLST, 2), 1),
                       ('RANGE_HI', combo(STATE, 2, VOLST, 1), 1), ('RANGE_LO', combo(STATE, 2, VOLST, 2), 1)]),
              ('XSD', [('TREND_UP', combo(STATE, 1, DIRS, 1), 1), ('TREND_DN', combo(STATE, 1, DIRS, 2), 1),
                       ('RANGE_UP', combo(STATE, 2, DIRS, 1), 1), ('RANGE_DN', combo(STATE, 2, DIRS, 2), 1)])]

    for h in HORIZONS:
        # 收盘入场：信号 row r → 入场 row r+1 → 持有至 row r+1+h
        raw = fwd_at(Cff, h, 1)
        mk = np.full(N, np.nan)
        if N > h + 1:
            mk[:N - h - 1] = (midx[h + 1:] / midx[1:N - h] - 1)
        rel = raw - mk.reshape(-1, 1)
        cnt = np.isfinite(raw).sum(1)
        cs = np.where(cnt > 0, np.nansum(raw, axis=1) / np.maximum(cnt, 1), np.nan).reshape(-1, 1)
        rel_s = raw - cs                                          # 样本内等权基准（稳健性）
        valid = np.isfinite(rel)
        x_all = rel.ravel()
        # 开盘入场（稳健性）：信号 r → 入场 open(r+1) → 持有至 close(r+1+h)
        raw_o = np.full_like(Cff, np.nan)
        if N > h + 1:
            raw_o[:N - h - 1] = Cff[h + 1:] / O[1:N - h] - 1
        mo = np.full(N, np.nan)
        if N > h + 1:
            mo[:N - h - 1] = midx[h + 1:] / (midx[:N - h - 1] * (1 + oret[1:N - h])) - 1
        rel_o = raw_o - mo.reshape(-1, 1)
        n_a, bad_a = self_test_align(K, raw, rel, mk, SIG, INV, CX, POSX, h, rng)
        print(f'  对齐自检 h={h}: 抽查 {n_a} 个事件，失配 {bad_a}')

        for grp, labs in GROUPS:
            for lab, SM, st in labs:
                base = (SM == st) & valid
                pool = np.flatnonzero(base.ravel())
                bx = x_all[pool]
                brec = dict(group=grp, signal='BASE_ALL_DAYS', state=lab, era='ALL', horizon=h,
                            n_events=len(pool), n_dates=len(np.unique(drow[pool])),
                            mean_raw_bp=np.nanmean(raw.ravel()[pool]) * 1e4, mean_rel_bp=bx.mean() * 1e4,
                            mean_rel_bp_samp=np.nan, median_rel_bp=np.median(bx) * 1e4,
                            hit_rate=(bx > 0).mean(), t_stat=np.nan, p_boot=np.nan, rand_mean_bp=np.nan,
                            rand_p5_bp=np.nan, rand_p95_bp=np.nan, p_rand=np.nan, excess_vs_rand_bp=np.nan,
                            mean_rel_bp_open=np.nan, n_months=0, month_mean_bp=np.nan, month_pos_frac=np.nan,
                            month_p10_bp=np.nan, month_p90_bp=np.nan, t_month=np.nan, p_month=np.nan)
                brec.update(month_stats(bx, drow[pool]))
                rows.append(brec)
                for k in names:
                    sel = SIG[k] & base
                    ix = np.flatnonzero(sel.ravel()) if sel.sum() >= 20 else np.array([], int)
                    rec = rec_of(grp, k, lab, h, 'ALL', ix, x_all, raw, drow, pool, rel_o)
                    if len(ix):
                        rec['mean_rel_bp_samp'] = rel_s.ravel()[ix].mean() * 1e4
                    rows.append(rec)
                    if grp == 'ADX' and len(ix):         # 分期稳健性
                        for name, _, _ in ERAS:
                            sub = np.flatnonzero((sel & era_row[name].reshape(-1, 1)).ravel())
                            rows.append(rec_of('ERA', k, lab, h, name, sub, x_all, raw, drow, pool, rel_o))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'写入 {OUT_CSV}  行数={len(out)}')

    for h in HORIZONS:
        s = out[(out.group == 'ADX') & (out.horizon == h) & (out.signal != 'BASE_ALL_DAYS')]
        print(f'\n=== h={h} 日：信号 × 状态（平均相对收益 bp）===')
        print(s.pivot_table(index='signal', columns='state', values='mean_rel_bp').round(1).to_string())
        print('--- 事件数 ---')
        print(s.pivot_table(index='signal', columns='state', values='n_events').to_string())
        print('--- 命中率 ---')
        print(s.pivot_table(index='signal', columns='state', values='hit_rate').round(3).to_string())
        print('--- p(日聚类bootstrap) ---')
        print(s.pivot_table(index='signal', columns='state', values='p_boot').round(3).to_string())
        print('--- p(月聚类，保守口径) ---')
        print(s.pivot_table(index='signal', columns='state', values='p_month').round(3).to_string())
    b = out[(out.group == 'ADX') & (out.signal == 'BASE_ALL_DAYS')]
    print('\n=== 状态基准：全部交易日的平均相对收益 bp ===')
    print(b.pivot_table(index='state', columns='horizon', values='mean_rel_bp').round(2).to_string())
    v = out[(out.group == 'VOL') & (out.signal != 'BASE_ALL_DAYS')]
    print('\n=== 波动状态切分（平均相对收益 bp）===')
    print(v.pivot_table(index=['signal', 'horizon'], columns='state', values='mean_rel_bp').round(1).to_string())


if __name__ == '__main__':
    main()
