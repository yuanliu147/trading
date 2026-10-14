#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# chan_diag_signals.py
# 缠论核心可检验主张的【诊断性】检验（探索性质，非预注册检验；结论若要用作交易规则须另行预注册）
#   A) 防狼术：MACD 黄白线（0轴）状态的未来收益区分度
#   B) 底背驰一买：60 日新低 + DIF 背离的后续表现（观测：背驰组 vs 非背驰组）
#   C) 三买：平台突破 + 回抽确认（不跌回平台）vs 直接追突破 vs 假突破回落
#   D) 中枢震荡：箱体内位置（下沿/上沿）的未来短周期收益梯度
# 数据：本地 DuckDB（hithink-finance CLI 同步的全 A 前复权日线），2016-10-11 起
# 口径：t+1 收盘入场（信号仅用 <=t 数据）；展示净口径时扣 0.15% 往返成本
import duckdb
import numpy as np
import pandas as pd

DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'
OUT = '/Users/admin/WorkSpace/trading/analysis/'
COST_RT = 0.0015

ERAS = [('2016-10-01', '2019-12-31', 'E1 2017-2019'),
        ('2020-01-01', '2022-12-31', 'E2 2020-2022'),
        ('2023-01-01', '2030-01-01', 'E3 2023-2026')]


def era_of(s):
    lab = pd.Series('E0', index=s.index, dtype=object)
    for a, b, name in ERAS:
        lab[(s >= pd.Timestamp(a)) & (s <= pd.Timestamp(b))] = name
    return lab


def nw_t(x, lags=None):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
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


def fmt(x, pct=True, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '  nan'
    return f"{x*100:+.{nd}f}%" if pct else f"{x:+.{nd}f}"


def main():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("""
        SELECT v.thscode, v.date, v.high, v.low, v.close, v.volume, v.amount
        FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
        WHERE s.exchange <> 'BJ'
        ORDER BY v.thscode, v.date
    """).df()
    con.close()
    for c in ['high', 'low', 'close', 'volume', 'amount']:
        df[c] = df[c].astype('float32')
    df['thscode'] = df['thscode'].astype('category')
    print(f"[load] rows={len(df):,} symbols={df['thscode'].nunique()} "
          f"span={df['date'].min().date()} ~ {df['date'].max().date()}", flush=True)

    g = df.groupby('thscode', observed=True)
    print("[feat] EMA/MACD ...", flush=True)
    df['ema12'] = g['close'].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    df['ema26'] = g['close'].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df['dif'] = df['ema12'] - df['ema26']
    df.drop(columns=['ema12', 'ema26'], inplace=True)
    g = df.groupby('thscode', observed=True)
    df['dea'] = g['dif'].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df['hist'] = df['dif'] - df['dea']
    print("[feat] rolling ...", flush=True)
    df['rmin60'] = g['low'].transform(lambda s: s.rolling(60, min_periods=60).min())
    df['hh30'] = g['high'].transform(lambda s: s.rolling(30, min_periods=30).max().shift(1))
    df['ll30'] = g['low'].transform(lambda s: s.rolling(30, min_periods=30).min().shift(1))
    df['hh40'] = g['high'].transform(lambda s: s.rolling(40, min_periods=40).max().shift(1))
    df['ll40'] = g['low'].transform(lambda s: s.rolling(40, min_periods=40).min().shift(1))
    df['age'] = g.cumcount()
    print("[feat] forward returns ...", flush=True)
    for k in (1, 6, 11, 21, 61):
        df[f'c{k}'] = g['close'].transform(lambda s, k=k: s.shift(-k))
    df['fwd5'] = (df['c6'] / df['c1'] - 1).astype('float32')
    df['fwd10'] = (df['c11'] / df['c1'] - 1).astype('float32')
    df['fwd20'] = (df['c21'] / df['c1'] - 1).astype('float32')
    df['fwd60'] = (df['c61'] / df['c1'] - 1).astype('float32')
    df.drop(columns=[f'c{k}' for k in (1, 6, 11, 21, 61)], inplace=True)
    df['pw30'] = (df['hh30'] / df['ll30'] - 1).astype('float32')

    univ = df[df['age'] >= 250]
    mkt = univ.groupby('date')[['fwd5', 'fwd10', 'fwd20', 'fwd60']].mean()
    print(f"[univ] age>=250 行数={len(univ):,}；全市场等权基准已算", flush=True)

    # ================= A) 防狼术：MACD 0轴状态 =================
    print("\n=========== A) 防狼术：0轴状态 × 未来收益 ===========", flush=True)
    dn = np.sort(df['date'].unique())
    sample = set(dn[::5])  # 每5个交易日取一个观测日，控制自相关
    A = univ[np.isin(univ['date'].values, list(sample)) & univ['dif'].notna()].copy()
    A['state'] = np.select(
        [(A['dif'] > 0) & (A['dea'] > 0), (A['dif'] < 0) & (A['dea'] < 0)],
        ['双上0轴', '双下0轴'], default='跨0轴')
    A['era'] = era_of(A['date'])
    tab = A.groupby(['era', 'state'], observed=True).agg(
        n=('fwd20', 'size'), fwd5=('fwd5', 'mean'), fwd10=('fwd10', 'mean'),
        fwd20=('fwd20', 'mean'), win20=('fwd20', lambda x: (x > 0).mean())).round(4)
    tab['fwd20_net'] = (tab['fwd20'] - COST_RT).round(4)
    print(tab.to_string())
    tab.to_csv(OUT + 'chan_diag_A_state.csv')
    # 月度序列：双上 - 双下（fwd20），NW-t
    A['ym'] = A['date'].dt.to_period('M')
    mm = A.groupby(['ym', 'state'], observed=True)['fwd20'].mean().unstack().dropna()
    d = mm['双上0轴'] - mm['双下0轴']
    print(f"[A] 月度(双上-双下) fwd20: mean={fmt(d.mean())} NW-t={nw_t(d.values):+.2f} "
          f"月份数={len(d)}")
    for e, gg in A.groupby('era', observed=True):
        share = gg['state'].value_counts(normalize=True).round(3).to_dict()
        print(f"[A] {e} 状态时间占比: {share}")

    # ================= B) 底背驰：60日新低 + DIF背离 =================
    print("\n=========== B) 底背驰一买：60日新低事件 ===========", flush=True)
    B = univ[univ['dif'].notna()].copy()
    B['nl'] = (B['low'] <= B['rmin60'] + 1e-9).fillna(False).astype(bool)
    gb = B.groupby('thscode', observed=True)
    B['eps_start'] = B['nl'] & (~gb['nl'].shift(1).fillna(False).astype(bool))
    B['eps_end'] = B['nl'] & (~gb['nl'].shift(-1).fillna(False).astype(bool))
    B['neg'] = np.minimum(B['hist'].fillna(0.0), 0.0)
    B['neg_cum'] = B.groupby('thscode', observed=True)['neg'].cumsum()
    reps = B[B['eps_end']][['thscode', 'date', 'age', 'low', 'close', 'dif', 'dea',
                            'neg_cum', 'fwd5', 'fwd10', 'fwd20', 'fwd60']].copy()
    print(f"[B] 60日新低段（episode）数 = {len(reps):,}", flush=True)
    gr = reps.groupby('thscode', observed=True)
    for c in ['low', 'close', 'dif', 'age', 'neg_cum']:
        reps[c + '_prev'] = gr[c].shift(1)
    reps['neg_cum_prev2'] = gr['neg_cum'].shift(2)
    pairs = reps.dropna(subset=['low_prev', 'dif_prev', 'age_prev']).copy()
    pairs = pairs[(pairs['age'] - pairs['age_prev']) <= 240]
    newlow = pairs['low'] < pairs['low_prev']
    pairs = pairs[newlow].copy()
    pairs['div'] = pairs['dif'] > pairs['dif_prev']
    pairs['div_strict'] = pairs['div'] & (pairs['dif'] < 0) & (pairs['dif_prev'] < 0)
    area_cur = pairs['neg_cum'] - pairs['neg_cum_prev']
    area_before = pairs['neg_cum_prev'] - pairs['neg_cum_prev2']
    pairs['div_area'] = pairs['div'] & (area_cur < area_before)
    pairs['era'] = era_of(pairs['date'])
    pairs['mkt_fwd20'] = pairs['date'].map(mkt['fwd20'])
    print(f"[B] 相邻新低对（间隔<=240交易日、创新低）= {len(pairs):,}，"
          f"其中DIF背离占比={pairs['div'].mean():.1%}")
    for key, label in [('div', 'DIF背离'), ('div_strict', 'DIF背离(严格:双负)'),
                       ('div_area', 'DIF背离+柱子面积收缩')]:
        print(f"\n[B] 分组: {label}")
        r = pairs.groupby(['era', key], observed=True).agg(
            n=('fwd20', 'size'), fwd5=('fwd5', 'mean'), fwd10=('fwd10', 'mean'),
            fwd20=('fwd20', 'mean'), fwd20_net=('fwd20', lambda x: x.mean() - COST_RT),
            fwd60=('fwd60', 'mean'), win20=('fwd20', lambda x: (x > 0).mean())).round(4)
        print(r.to_string())
        pairs['ym'] = pairs['date'].dt.to_period('M')
        mm = pairs.groupby(['ym', key], observed=True)['fwd20'].mean().unstack().dropna()
        if True in mm.columns and False in mm.columns:
            d = mm[True] - mm[False]
            print(f"    月度(背离-非背离) fwd20: mean={fmt(d.mean())} NW-t={nw_t(d.values):+.2f} 月数={len(d)}")
    mkt_era = mkt.groupby(era_of(pd.Series(mkt.index, index=mkt.index)))['fwd20'].mean().round(4).to_dict()
    print(f"[B] 基准(全市场等权 fwd20): 全期={fmt(mkt['fwd20'].mean())} | 分era: {mkt_era}")
    pairs.drop(columns=['low_prev', 'close_prev', 'dif_prev', 'age_prev', 'neg_cum_prev', 'neg_cum_prev2']).to_csv(
        OUT + 'chan_diag_B_events.csv', index=False)

    # ================= C) 三买：平台突破 + 回抽确认 =================
    print("\n=========== C) 三买：平台突破 + 回抽确认 ===========", flush=True)
    C0 = univ[univ['hh30'].notna() & univ['ll30'].notna()].copy()
    C0['brk'] = ((C0['close'] > C0['hh30']) & (C0['pw30'] <= 0.15)).fillna(False)
    close_all = C0['close'].values
    low_all = C0['low'].values
    hh_all = C0['hh30'].values
    date_all = C0['date'].values
    age_all = C0['age'].values
    brk_all = C0['brk'].values
    fwd_cols = {h: C0[f'fwd{h}'].values for h in (5, 10, 20, 60)}
    codes = C0['thscode'].cat.codes.values
    cats = C0['thscode'].cat.categories
    starts = np.concatenate([[0], np.flatnonzero(codes[1:] != codes[:-1]) + 1])
    ends = np.concatenate([starts[1:], [len(codes)]])
    events = []
    for s0, e0 in zip(starts, ends):
        m = e0 - s0
        if m < 300:
            continue
        brk_idx = np.flatnonzero(brk_all[s0:e0])
        kept = []
        last_age = -10 ** 9
        for i in brk_idx:
            if age_all[s0 + i] - last_age >= 30:  # 同一股票30个交易日内只留首次突破
                kept.append(i)
                last_age = age_all[s0 + i]
        for i in kept:
            H = hh_all[s0 + i]
            outcome, p = None, -1
            for j in range(i + 1, min(i + 11, m)):
                if close_all[s0 + j] < H:
                    outcome, p = 'failed', j
                    break
                if low_all[s0 + j] <= H * 1.03:
                    outcome, p = 'confirmed', j
                    break
            if outcome is None:
                outcome = 'runaway' if m - i >= 12 else 'truncated'
            rec = dict(thscode=str(cats[codes[s0 + i]]),
                       b_date=date_all[s0 + i], H=H, outcome=outcome)
            for h in (5, 10, 20, 60):
                rec[f'b_fwd{h}'] = fwd_cols[h][s0 + i]
                rec[f'p_fwd{h}'] = fwd_cols[h][s0 + p] if p >= 0 else np.nan
            rec['p_date'] = date_all[s0 + p] if p >= 0 else np.datetime64('NaT')
            events.append(rec)
    ev = pd.DataFrame(events)
    ev['b_date'] = pd.to_datetime(ev['b_date'])
    ev['p_date'] = pd.to_datetime(ev['p_date'])
    ev['era'] = era_of(ev['b_date'])
    ev['b_mkt20'] = ev['b_date'].map(mkt['fwd20'])
    ev['p_mkt20'] = ev['p_date'].map(mkt['fwd20'])
    ev.to_csv(OUT + 'chan_diag_C_events.csv', index=False)
    print(f"[C] 突破事件数={len(ev):,}，结局分布: {ev['outcome'].value_counts().to_dict()}")
    print("\n[C] 三买确认入场（p+1） vs 无回抽直接追（b+1） vs 假突破（b+1）:")
    r = ev.groupby(['era', 'outcome'], observed=True).agg(
        n=('outcome', 'size'),
        b_fwd5=('b_fwd5', 'mean'), b_fwd10=('b_fwd10', 'mean'), b_fwd20=('b_fwd20', 'mean'),
        b_fwd60=('b_fwd60', 'mean'),
        p_fwd5=('p_fwd5', 'mean'), p_fwd10=('p_fwd10', 'mean'), p_fwd20=('p_fwd20', 'mean'),
        p_fwd60=('p_fwd60', 'mean'),
        win20_p=('p_fwd20', lambda x: (x > 0).mean())).round(4)
    print(r.to_string())
    # 关键对比：确认后入场 vs 确认前追入（同组事件） 与 vs 假突破
    conf = ev[ev['outcome'] == 'confirmed']
    print(f"\n[C] 同一批确认事件: p_fwd20 - b_fwd20 = "
          f"{fmt((conf['p_fwd20'] - conf['b_fwd20']).mean())} "
          f"(正=回抽后再买更好; 负=等待有代价)")
    for e, gg in ev.groupby('era', observed=True):
        c2 = gg[gg['outcome'] == 'confirmed']
        f2 = gg[gg['outcome'] == 'failed']
        print(f"[C] {e}: confirmed n={len(c2)} p_fwd20均值={fmt(c2['p_fwd20'].mean())} "
              f"净={fmt(c2['p_fwd20'].mean()-COST_RT)} | failed n={len(f2)} "
              f"p_fwd20(跌回后)={fmt(f2['p_fwd20'].mean())} b_fwd20(追入)={fmt(f2['b_fwd20'].mean())}")

    # ================= D) 中枢震荡：箱体内位置 =================
    print("\n=========== D) 中枢震荡：箱体内位置 × 未来收益 ===========", flush=True)
    D = univ[(univ['hh40'].notna()) & ((univ['hh40'] / univ['ll40'] - 1) <= 0.25)].copy()
    D['pos'] = (D['close'] - D['ll40']) / (D['hh40'] - D['ll40'])
    D = D[D['pos'].notna() & D['fwd5'].notna()]
    D['bucket'] = pd.cut(D['pos'], [0, 0.2, 0.4, 0.6, 0.8, 1.00001],
                         labels=['0-20 下沿', '20-40', '40-60', '60-80', '80-100 上沿'])
    D['era'] = era_of(D['date'])
    r = D.groupby(['era', 'bucket'], observed=True).agg(
        n=('fwd5', 'size'), fwd5=('fwd5', 'mean'), fwd5_net=('fwd5', lambda x: x.mean() - COST_RT),
        fwd10=('fwd10', 'mean'), win5=('fwd5', lambda x: (x > 0).mean())).round(4)
    print(r.to_string())
    r.to_csv(OUT + 'chan_diag_D_buckets.csv')
    D['ym'] = D['date'].dt.to_period('M')
    mm = D.groupby(['ym', 'bucket'], observed=True)['fwd5'].mean().unstack().dropna()
    d = mm['0-20 下沿'] - mm['80-100 上沿']
    print(f"[D] 月度(下沿-上沿) fwd5: mean={fmt(d.mean())} NW-t={nw_t(d.values):+.2f} 月数={len(d)}")
    mm10 = D.groupby(['ym', 'bucket'], observed=True)['fwd10'].mean().unstack().dropna()
    d10 = mm10['0-20 下沿'] - mm10['80-100 上沿']
    print(f"[D] 月度(下沿-上沿) fwd10: mean={fmt(d10.mean())} NW-t={nw_t(d10.values):+.2f} 月数={len(d10)}")

    print("\n[benchmark] 全市场等权 fwd20 全期均值:", fmt(mkt['fwd20'].mean()),
          "| fwd5:", fmt(mkt['fwd5'].mean()))
    print("\n[done] 输出: chan_diag_A_state.csv / chan_diag_B_events.csv / "
          "chan_diag_C_events.csv / chan_diag_D_buckets.csv", flush=True)


if __name__ == '__main__':
    main()
