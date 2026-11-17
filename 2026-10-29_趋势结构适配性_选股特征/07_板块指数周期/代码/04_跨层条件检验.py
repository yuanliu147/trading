"""冻结2010—2020跨层事件研究；仅写结果/04_*，不复用旧事件。"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
from collections import Counter

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '结果'
END = pd.Timestamp('2020-12-31')
COST = 0.00254
HORIZONS = (5, 20, 60)
CYCLES = (20, 60, 120, 250)
METRICS = ('扣费收益', '相对全A', '相对行业', '相对留一行业')
PERIODS = {
    '主窗2015-2020': ('2015-01-01', '2020-12-31'),
    '2015-2017': ('2015-01-01', '2017-12-31'),
    '2018-2020': ('2018-01-01', '2020-12-31'),
    '补充2010-2012': ('2010-01-01', '2012-12-31'),
    '补充2013-2014': ('2013-01-01', '2014-12-31'),
}
STATUS = {-1: '非红', 0: '缺失', 1: '红'}
BUY = {0: '可买', 1: '信号未准入', 2: '越封存线', 3: '停牌或不可交易',
       4: '价格或涨停规则不可判', 5: '开盘涨停'}
EXIT = {0: '有效完结', 1: '未买入', 2: '越封存线未实现',
        3: '缺价或卖出规则不可判', 4: '延后20日仍不可卖未实现'}


def valid_price(x):
    return np.isfinite(x) & (x > 0)


def limits(board, dates, st, preclose):
    """按原始preclose四舍五入至分；未知板块/ST/preclose明确NA。"""
    b = np.asarray(board)
    st = np.asarray(st, dtype=float)
    p = np.full(len(b), np.nan)
    main, gem, star = b == '主板', b == '创业板', b == '科创板'
    p[main & (st == 0)] = .10
    p[main & (st == 1)] = .05
    new_gem = np.asarray(dates) >= np.datetime64('2020-08-24')
    p[gem & ~new_gem & (st == 0)] = .10
    p[gem & ~new_gem & (st == 1)] = .05
    p[gem & new_gem & np.isin(st, [0, 1])] = .20
    p[star & np.isin(st, [0, 1])] = .20
    preclose = np.asarray(preclose, dtype=float)
    p[~valid_price(preclose)] = np.nan
    upper = np.floor(preclose * (1 + p) * 100 + .5 + 1e-9) / 100
    lower = np.floor(preclose * (1 - p) * 100 + .5 + 1e-9) / 100
    return upper, lower


def resolve_exit(planned, sell):
    """sell=0可卖、1停牌/跌停、2不可判；遇不可判即停止而不臆测顺延。"""
    planned = np.asarray(planned, dtype=int)
    actual = np.full(len(planned), -1, dtype=int)
    reason = np.full(len(planned), 4, dtype=np.int8)
    active = np.ones(len(planned), dtype=bool)
    for delay in range(21):
        x = planned + delay
        out = active & (x >= len(sell))
        reason[out] = 2
        active[out] = False
        ix = np.flatnonzero(active)
        if not len(ix):
            break
        code = sell[x[ix]]
        unknown, done = ix[code == 2], ix[code == 0]
        reason[unknown] = 3
        actual[done], reason[done] = x[done], 0
        active[unknown] = False
        active[done] = False
    return actual, reason


def interval_product(g, intra, entry, exit_):
    """entry日仅日内，此后隔夜×日内；任何缺失均阻断，不填零收益。"""
    g, intra = np.asarray(g), np.asarray(intra)
    daily = (1 + g) * (1 + intra)
    good = valid_price(daily)
    logs = np.zeros(len(g))
    logs[good] = np.log(daily[good])
    acc = np.r_[0., np.cumsum(logs)]
    bad = np.r_[0, np.cumsum(~good)]
    e, x = np.asarray(entry), np.asarray(exit_)
    ans = np.full(len(e), np.nan)
    ok = (e >= 0) & (x >= e) & (x < len(g))
    k = np.flatnonzero(ok)
    ok[k] &= valid_price(1 + intra[e[k]]) & (bad[x[k] + 1] == bad[e[k] + 1])
    k = np.flatnonzero(ok)
    ans[k] = np.exp(np.log1p(intra[e[k]]) + acc[x[k] + 1] - acc[e[k] + 1]) - 1
    return ans


def same_window(open_, close_, count, entry, exit_):
    n = len(close_)
    good = valid_price(close_) & (count >= 10)
    bad = np.r_[0, np.cumsum(~good)]
    e, x = np.asarray(entry), np.asarray(exit_)
    ans = np.full(len(e), np.nan)
    k = np.flatnonzero((e >= 0) & (x >= e) & (x < n))
    ok = valid_price(open_[e[k]]) & (bad[x[k] + 1] == bad[e[k]])
    k = k[ok]
    ans[k] = close_[x[k]] / open_[e[k]] - 1
    return ans


def tercile(s, bins):
    # 同值平均秩，不按symbol打破同值；不事后挪动边界以凑满各格。
    pct = s.rank(method='average', pct=True)
    return np.minimum(np.floor(pct * bins).fillna(-1).astype(int), bins - 1) + 1


def self_test():
    d = pd.to_datetime(['2020-08-21', '2020-08-24', '2020-08-21', '2020-08-24',
                        '2020-08-24', '2020-08-24', '2020-08-24'])
    u, l = limits(['创业板', '创业板', '创业板', '创业板', '主板', '科创板', '未知'],
                  d, [0, 0, 1, 1, 1, 0, 0], np.full(7, 10.))
    np.testing.assert_allclose(l[:6], [9, 8, 9.5, 8, 9.5, 8])
    assert np.isnan(u[-1])
    a, r = resolve_exit([0, 1, 3, 4], np.array([0, 1, 1, 2]))
    assert a.tolist() == [0, -1, -1, -1] and r.tolist() == [0, 3, 3, 2]
    a, r = resolve_exit([0], np.ones(30, dtype=int))
    assert r[0] == 4
    a, r = resolve_exit([0], np.r_[np.ones(20, dtype=int), 0])
    assert a[0] == 20 and r[0] == 0
    v = interval_product(np.array([np.nan, .2, .3]), np.array([.1, .1, .1]), [0], [2])
    np.testing.assert_allclose(v, [1.1 * 1.2 * 1.1 * 1.3 * 1.1 - 1])
    assert np.isnan(interval_product(np.array([0., np.nan]), np.array([.1, .1]), [0], [1])[0])
    # 日历缺一行不能压缩为相邻收盘；past n不含信号日。
    s = pd.Series([1., 2., np.nan, 4., 5., 6.])
    assert np.isnan(s.shift(1).rolling(3, min_periods=3).max().iloc[5])
    assert s.shift(1).rolling(2, min_periods=2).max().iloc[5] == 5
    assert hashlib.sha256(b'000001.SZ').digest()[-1] & 1 in (0, 1)
    print('执行规则、首日日内、缺失不压缩、20日延后边界自检通过', flush=True)


class Writer:
    def __init__(self, name):
        self.path = OUT / name
        self.writer = None
        self.parts = []

    def add(self, df):
        if len(df):
            self.parts.append(df)

    def flush(self):
        if not self.parts:
            return
        df = pd.concat(self.parts, ignore_index=True)
        t = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, t.schema, compression='zstd')
        self.writer.write_table(t.cast(self.writer.schema))
        self.parts.clear()

    def close(self):
        self.flush()
        if self.writer:
            self.writer.close()


def load_benchmarks(con, cal, audit):
    path = ROOT / '数据/个股与行业.parquet'
    calendar = pd.DataFrame({'date': cal, '市场前日': pd.Series(cal).shift(1)})
    con.register('calendar', calendar)
    # 原表首次记录ASOF在左侧前日NULL时存在误配，主动剔除而非使用未来行业。
    sql = f"""WITH x AS (
        SELECT symbol,date,行业码,有效起日,td_since_list,hfq_open,hfq_close,
          lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) pc,
          lag(date) OVER (PARTITION BY symbol ORDER BY date) pd
        FROM read_parquet('{path}')
      ), v AS (
        SELECT x.*,hfq_open/pc-1 g,hfq_close/hfq_open-1 i
        FROM x JOIN calendar c USING(date)
        WHERE pd=市场前日 AND 有效起日<=市场前日 AND 行业码<801950
          AND td_since_list>=60 AND pc>0 AND hfq_open>0 AND hfq_close>0
          AND isfinite(pc) AND isfinite(hfq_open) AND isfinite(hfq_close)
      ) SELECT date,行业码,count(*) n,sum(g) sg,sum((1+g)*(1+i)-1) si FROM v GROUP BY date,行业码"""
    agg = con.execute(sql).fetchdf()
    sectors = pd.read_parquet(ROOT / '数据/行业合成开收盘.parquet')
    sectors['date'] = pd.to_datetime(sectors['date'])
    codes = sorted(sectors['行业码'].unique())
    names = sectors.groupby('行业码')['行业'].last().to_dict()
    bases = {}
    for code in codes:
        s = sectors[sectors['行业码'] == code].set_index('date').reindex(cal)
        a = agg[agg['行业码'] == code].set_index('date').reindex(cal)
        # 无成分时n=0、和=0仅代表空集合，后续n>=10仍强制为缺失。
        bases[code] = {k: a[k].fillna(0).to_numpy(float) for k in ('n', 'sg', 'si')}
        bases[code].update({'open': s.open.to_numpy(float), 'close': s.close.to_numpy(float),
                           'count': s['成分数'].to_numpy(float)})
    market = pd.read_parquet(ROOT / '数据/全A等权开收盘.parquet')
    market['date'] = pd.to_datetime(market.date)
    assert not market.date.duplicated().any() and market.date.max() <= END
    market = market.set_index('date').reindex(cal)
    audit['市场基准缺价交易日'] = int((~valid_price(market.open) | ~valid_price(market.close)).sum())
    # 主窗指标不跨2014旧分类预热；新增行业还必须自己的完整n日。
    states, momenta = {}, {}
    for n in CYCLES:
        arr = np.zeros((len(cal), len(codes)), dtype=np.int8)
        for j, code in enumerate(codes):
            b = bases[code]
            price = pd.Series(b['close'], index=cal).where(b['count'] >= 10)
            modern = price.where(cal >= pd.Timestamp('2014-01-01'))
            ma = price.rolling(n, min_periods=n).mean().shift(1)
            mam = modern.rolling(n, min_periods=n).mean().shift(1)
            prev = price.shift(1)
            use = pd.Series(np.where(cal >= pd.Timestamp('2015-01-01'), mam, ma), index=cal)
            good = prev.notna() & use.notna()
            arr[good.to_numpy(), j] = np.where(prev[good] > use[good], 1, -1)
        states[n] = arr
    for n in (20, 60, 120):
        m = np.full((len(cal), len(codes)), np.nan)
        for j, code in enumerate(codes):
            price = pd.Series(bases[code]['close']).where(bases[code]['count'] >= 10)
            good = price.rolling(n + 1, min_periods=n + 1).count() == n + 1
            m[good.to_numpy(), j] = np.log(price[good] / price.shift(n)[good])
        momenta[n] = m
    month_end = np.flatnonzero(pd.Series(cal).dt.to_period('M').ne(pd.Series(cal).dt.to_period('M').shift(-1)))
    industry_ranks = {}
    for n in (60, 120):
        ranks = np.zeros_like(momenta[n], dtype=np.int8)
        for d in month_end:
            ranks[d] = tercile(pd.Series(momenta[n][d]), 3).to_numpy()
        industry_ranks[n] = ranks
    return codes, names, bases, market, states, momenta, industry_ranks, month_end


def stock_groups(con):
    cols = 'symbol,date,board,td_since_list,hfq_open,hfq_close,open,high,low,close,preclose,amount,trade_status,is_st,is_suspended,is_limit_up_open,行业码,有效起日'
    q = f"SELECT {cols} FROM read_parquet('{ROOT / '数据/个股与行业.parquet'}') ORDER BY symbol,date"
    reader = con.execute(q).to_arrow_reader(batch_size=65536)
    pending = None
    for batch in reader:
        frame = batch.to_pandas()
        if pending is not None:
            frame = pd.concat([pending, frame], ignore_index=True)
        last = frame.symbol.iloc[-1]
        pending = frame[frame.symbol == last].copy()
        for symbol, group in frame[frame.symbol != last].groupby('symbol', sort=False):
            yield symbol, group
    if pending is not None:
        yield pending.symbol.iloc[0], pending


def build_events(con, cal, audit):
    codes, names, bases, market, states, momentum, ranks, monthly = load_benchmarks(con, cal, audit)
    code_index = {c: j for j, c in enumerate(codes)}
    cal_index = pd.Index(cal)
    nday = len(cal)
    after_start = cal >= pd.Timestamp('2010-01-01')
    ew = Writer('04_事件明细.parquet')
    mw = Writer('04_月末候选明细.parquet')
    counts = Counter()
    cover = []
    t0 = time.time()
    for sid, (symbol, raw) in enumerate(stock_groups(con)):
        raw['date'] = pd.to_datetime(raw.date)
        assert not raw.date.duplicated().any()
        raw_days = cal_index.get_indexer(raw.date)
        assert (raw_days >= 0).all()
        g = raw.set_index('date').reindex(cal)
        present = g.symbol.notna().to_numpy()
        prev_cal = pd.Series(cal).shift(1).to_numpy()
        pit = (g['有效起日'].to_numpy() <= prev_cal) & g['行业码'].isin(codes).to_numpy()
        bad_pit = present & g['行业码'].notna().to_numpy() & ~pit
        counts['源行业生效晚于市场前日或首日不可判'] += int(bad_pit.sum())
        industry = g['行业码'].where(pit).to_numpy(float)
        ci = np.array([code_index.get(x, -1) for x in industry], dtype=np.int16)
        o, c = g.hfq_open.to_numpy(float), g.hfq_close.to_numpy(float)
        ro, rc, hi, lo, pc = [g[k].to_numpy(float) for k in ('open', 'close', 'high', 'low', 'preclose')]
        anomalies = present & ((hi < lo) | (hi + .005 < np.maximum(ro, rc)) |
                               (lo - .005 > np.minimum(ro, rc)) |
                               ~valid_price(ro) | ~valid_price(rc) | ~valid_price(hi) | ~valid_price(lo))
        counts['源rawhigh小于low'] += int((present & (hi < lo)).sum())
        counts['源原始OHLC异常'] += int(anomalies.sum())
        counts['源preclose缺失或非正'] += int((present & ~valid_price(pc)).sum())
        up, down = limits(g.board.fillna('').to_numpy(), cal, g.is_st.to_numpy(float), pc)
        suspended = g.is_suspended.eq(True).to_numpy() | g.trade_status.eq(0).to_numpy()
        trade = present & g.trade_status.eq(1).to_numpy() & g.is_suspended.eq(False).to_numpy()
        rule = valid_price(up) & valid_price(down)
        computed_up = ro >= up - .005
        source_up = g.is_limit_up_open.eq(True).to_numpy()
        known_flag = g.is_limit_up_open.notna().to_numpy() & rule & valid_price(ro)
        counts['源开盘涨停标记与重算不一致'] += int((known_flag & (source_up != computed_up)).sum())
        counts['源涨停标记缺失'] += int((present & g.is_limit_up_open.isna().to_numpy()).sum())
        buy_day = np.full(nday, 4, dtype=np.int8)
        buy_day[trade & rule & valid_price(o) & ~anomalies] = 0
        buy_day[trade & rule & valid_price(o) & ~anomalies & (computed_up | source_up)] = 5
        buy_day[suspended] = 3
        sell = np.full(nday, 2, dtype=np.int8)
        sell[trade & rule & valid_price(c) & ~anomalies] = 0
        sell[trade & rule & valid_price(c) & ~anomalies & (rc <= down + .005)] = 1
        sell[suspended] = 1
        eligible = (g.td_since_list.ge(250).to_numpy() & g.is_st.eq(0).to_numpy() & trade &
                    valid_price(c) & ~anomalies)
        histclose = pd.Series(c).where(valid_price(c) & ~anomalies)
        high20 = histclose.shift(1).rolling(20, min_periods=20).max().to_numpy()
        high55 = histclose.shift(1).rolling(55, min_periods=55).max().to_numpy()
        low55 = histclose.shift(1).rolling(55, min_periods=55).min().to_numpy()
        signal_masks = (c > high20, c > high55, c < low55)
        half = hashlib.sha256(symbol.encode()).digest()[-1] & 1
        # 真实日历上逐日开收盘因子；仅同一日行业、已有60日上市且前日价格有效者参与。
        prev_c = np.r_[np.nan, c[:-1]]
        own_valid = (g.td_since_list.ge(60).to_numpy() & pit & valid_price(prev_c) &
                     valid_price(o) & valid_price(c))
        own_g = np.full(nday, np.nan)
        own_i = np.full(nday, np.nan)
        own_g[own_valid] = o[own_valid] / prev_c[own_valid] - 1
        own_i[own_valid] = c[own_valid] / o[own_valid] - 1
        loo_cache = {}

        def returns(signal_days, horizon):
            e = signal_days + 1
            br = np.full(len(e), 2, dtype=np.int8)
            within = e < nday
            br[within] = buy_day[e[within]]
            br[~eligible[signal_days]] = 1
            bought = br == 0
            x = np.full(len(e), -1, dtype=int)
            er = np.full(len(e), 1, dtype=np.int8)
            if bought.any():
                x[bought], er[bought] = resolve_exit(e[bought] + horizon - 1, sell)
            done = er == 0
            stock = np.full(len(e), np.nan)
            stock[done] = c[x[done]] / o[e[done]] - 1
            mr = same_window(market.open.to_numpy(), market.close.to_numpy(), np.full(nday, 10), e, x)
            sr, lr = np.full(len(e), np.nan), np.full(len(e), np.nan)
            for code in np.unique(industry[signal_days][np.isfinite(industry[signal_days])]):
                ix = np.flatnonzero(industry[signal_days] == code)
                b = bases[code]
                sr[ix] = same_window(b['open'], b['close'], b['count'], e[ix], x[ix])
                if code not in loo_cache:
                    member = own_valid & (industry == code)
                    nn = b['n'] - member
                    gg, ii = np.full(nday, np.nan), np.full(nday, np.nan)
                    ok = nn >= 10
                    gg[ok] = (b['sg'][ok] - np.where(member[ok], own_g[ok], 0)) / nn[ok]
                    own_daily = (1+own_g)*(1+own_i)-1
                    rr = (b['si'][ok] - np.where(member[ok], own_daily[ok], 0)) / nn[ok]
                    # 固定上一收盘等权，隔夜漂移后日内收益不能再次简单等权。
                    ii[ok] = (1+rr)/(1+gg[ok])-1
                    loo_cache[code] = (gg, ii)
                lr[ix] = interval_product(*loo_cache[code], e[ix], x[ix])
            frame = pd.DataFrame({'股票编号': np.full(len(e), sid, dtype=np.int32),
                                  '证券': symbol, '哈希半区': np.full(len(e), half, dtype=np.int8),
                                  '信号日序': signal_days.astype(np.int16), '信号日': cal[signal_days],
                                  '行业码': industry[signal_days], '持有日': np.full(len(e), horizon, dtype=np.int8),
                                  '准入': eligible[signal_days], '买入状态': br, '退出状态': er,
                                  '入场日序': e.astype(np.int16),
                                  '计划退出日序': (e + horizon - 1).astype(np.int16),
                                  '实际退出日序': x.astype(np.int16),
                                  '延后日数': np.where(done, x - e - horizon + 1, -1).astype(np.int8),
                                  '原始收益': stock, '扣费收益': stock - COST,
                                  '全A同窗收益': mr, '行业同窗收益': sr, '留一行业同窗收益': lr,
                                  '相对全A': stock - COST - mr, '相对行业': stock - COST - sr,
                                  '相对留一行业': stock - COST - lr})
            # 状态在T时仅使用行业T-1收盘与T-1 MA；缺行业不拿最后行业回填。
            j = ci[signal_days]
            for n in CYCLES:
                z = np.zeros(len(j), dtype=np.int8)
                z[j >= 0] = states[n][signal_days[j >= 0], j[j >= 0]]
                frame[f'大周期{n}'] = z
            return frame

        for event, mask in zip(('20日新高', '55日新高', '55日新低'), signal_masks):
            days = np.flatnonzero(mask & after_start & valid_price(c) & ~anomalies)
            for h in HORIZONS:
                f = returns(days, h)
                f['事件'] = event
                ew.add(f)
        # 月末排序先在PIT可入场研究宇宙冻结，再检验T+1真实执行。
        days = monthly[after_start[monthly]]
        stock20 = (histclose / histclose.shift(20)).apply(np.log)
        complete20 = histclose.rolling(21, min_periods=21).count().eq(21).to_numpy()
        past_amount = g.amount.shift(1).rolling(20, min_periods=20).mean().to_numpy()
        rel20 = np.full(nday, np.nan)
        ix = np.flatnonzero((ci >= 0) & complete20)
        rel20[ix] = stock20.to_numpy()[ix] - momentum[20][ix, ci[ix]]
        selected = days[eligible[days] & np.isfinite(rel20[days]) &
                        np.isfinite(past_amount[days]) & (ci[days] >= 0)]
        m = returns(selected, 20)
        m['相对行业20日动量'] = rel20[selected]
        m['过去20日成交额'] = past_amount[selected]
        for n in (60, 120):
            m[f'行业动量{n}三分位'] = ranks[n][selected, ci[selected]]
        mw.add(m)
        cover.append({'证券': symbol, '哈希半区': half, '原始行数': len(raw),
                      '上市后范围内缺失市场日': int((~present[raw_days.min():raw_days.max() + 1]).sum()),
                      '行业前日生效异常行数': int(bad_pit.sum()),
                      '2015后原始行数': int((raw.date >= '2015-01-01').sum()),
                      '2015后PIT行业覆盖行数': int((pit & present & (cal >= '2015-01-01')).sum()),
                      '月末PIT可选数': int(eligible[days].sum()), '月末完整特征候选数': len(selected)})
        if (sid + 1) % 100 == 0:
            ew.flush()
            mw.flush()
        if (sid + 1) % 500 == 0:
            print(f'已构造{sid + 1}股，耗时{time.time() - t0:.0f}秒', flush=True)
    ew.close()
    mw.close()
    pd.DataFrame(cover).to_csv(OUT / '04_个股覆盖.csv', index=False, encoding='utf-8-sig')
    audit['原始异常核验'] = dict(counts)
    audit['覆盖'] = {'股票数': len(cover), '行业数': len(codes), '交易日数': nday,
                     '行业名称': {str(k): v for k, v in names.items()},
                     '月末PIT可选数': sum(x['月末PIT可选数'] for x in cover),
                     '月末完整特征候选数': sum(x['月末完整特征候选数'] for x in cover),
                     '上市后范围内缺失股票市场日': sum(x['上市后范围内缺失市场日'] for x in cover)}
    return ew.path, mw.path


def aggregate_and_test(events, monthly, cal, bootstrap, audit):
    rows, tests = [], []
    daily_writer = Writer('04_日级聚合.parquet')
    # 完整交易日日历，无事件日保留收益和=0、事件数=0，绝非补零收益。
    def add_cell(f, info, days_in_period, contrast=False):
        n = len(f)
        if contrast:
            raise AssertionError('对比必须通过日级差值专用入口')
        raw_dates = f['信号日序'].nunique()
        br, er = f['买入状态'], f['退出状态']
        common = dict(info, 原始信号数=n, 原始信号日数=int(raw_dates),
                      PIT准入数=int(f['准入'].sum()), 可买入数=int(br.eq(0).sum()),
                      有效完结数=int(er.eq(0).sum()), 不可买数=int(br.ne(0).sum()),
                      未实现数=int(er.isin([2, 4]).sum()), 不可判退出数=int(er.eq(3).sum()),
                      延后退出数=int((f['延后日数'] > 0).sum()))
        for metric in METRICS:
            vals = f[metric]
            v = np.isfinite(vals)
            d, x = f.loc[v, '信号日序'].to_numpy(int), vals[v].to_numpy(float)
            sums = np.bincount(d, weights=x, minlength=len(cal))
            counts = np.bincount(d, minlength=len(cal))
            mean = float(x.mean()) if len(x) else np.nan
            dates = int((counts > 0).sum())
            item = dict(common, 指标=metric, 指标有效数=len(x), 指标有效信号日数=dates,
                        完结后基准缺失数=int(er.eq(0).sum()) - len(x),
                        均值=mean, 中位数=float(np.median(x)) if len(x) else np.nan,
                        胜率=float((x > 0).mean()) if len(x) else np.nan,
                        胜率分母='该指标有效完结事件，不含不可买/未实现/基准缺失',
                        样本标记='足量' if len(x) >= 200 and dates >= 100 else '统计不足：少于200事件或100信号日',
                        CI下界=np.nan, CI上界=np.nan, 块bootstrap_p=np.nan, 统一FDR_q=np.nan,
                        整体maxstat_p=np.nan, bootstrap有效次数=0)
            row_id = len(rows)
            rows.append(item)
            if len(x):
                tests.append((row_id, info['时段'], sums, counts, mean))
            daily_writer.add(pd.DataFrame({'格编号': row_id, '日期': cal[days_in_period],
                                           '收益和': sums[days_in_period], '有效事件数': counts[days_in_period]}))
            if row_id % 192 == 0:
                daily_writer.flush()

    def halves(f):
        yield '全部', f
        for h in (0, 1):
            yield f'哈希{h}（横截面诊断）', f[f['哈希半区'] == h]

    print('聚合事件全部格（不挑参数）', flush=True)
    for pname, (start, stop) in PERIODS.items():
        days = np.flatnonzero((cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(stop)))
        sample = events[events['信号日序'].isin(days)]
        for half, halfsample in halves(sample):
            for (event, h), f in halfsample.groupby(['事件', '持有日'], observed=True):
                for cycle in CYCLES:
                    for z, label in STATUS.items():
                        sub = f[f[f'大周期{cycle}'] == z]
                        info = {'研究': '跨层事件', '时段': pname, '半区': half, '事件': event,
                                '持有日': int(h), '大周期': cycle, '大周期状态': label,
                                '行业动量日': 0, '行业三分位': 0, '个股五分位': 0, '流动性': '全部'}
                        add_cell(sub, info, days)
    monthly['个股五分位'] = monthly.groupby(['信号日序', '行业码'], observed=True)['相对行业20日动量'].transform(lambda s: tercile(s, 5)).astype(np.int8)
    monthly['流动性三分位'] = monthly.groupby('信号日序')['过去20日成交额'].transform(lambda s: tercile(s, 3)).astype(np.int8)
    monthly.to_parquet(OUT / '04_双重排序明细.parquet', index=False, compression='zstd')
    print('聚合双重排序全部格', flush=True)
    for pname, (start, stop) in PERIODS.items():
        if pname.startswith('补充'):
            continue
        days = np.flatnonzero((cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(stop)))
        sample = monthly[monthly['信号日序'].isin(days)]
        for half, halfsample in halves(sample):
            for n in (60, 120):
                for itier in (1, 2, 3):
                    for qtier in (1, 2, 3, 4, 5):
                        grid = halfsample[(halfsample[f'行业动量{n}三分位'] == itier) & (halfsample['个股五分位'] == qtier)]
                        for liq in (0, 1, 2, 3):
                            sub = grid if liq == 0 else grid[grid['流动性三分位'] == liq]
                            info = {'研究': '月末双重排序', '时段': pname, '半区': half, '事件': '月末排序',
                                    '持有日': 20, '大周期': 0, '大周期状态': '不适用', '行业动量日': n,
                                    '行业三分位': itier, '个股五分位': qtier, '流动性': '全部' if liq == 0 else f'成交额{liq}档'}
                            add_cell(sub, info, days)
    # 强行业弱股与强股的显式同日差；同一信号日两腿均有有效收益才可比较。
    for pname, (start, stop) in PERIODS.items():
        if pname.startswith('补充'):
            continue
        days = np.flatnonzero((cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(stop)))
        sample = monthly[monthly['信号日序'].isin(days)]
        for half, hs in halves(sample):
            for n in (60, 120):
                for liq in (0, 1, 2, 3):
                    strong = hs[hs[f'行业动量{n}三分位'] == 3]
                    if liq:
                        strong = strong[strong['流动性三分位'] == liq]
                    weak = strong[strong['个股五分位'] == 1]
                    high = strong[strong['个股五分位'] == 5]
                    for metric in METRICS:
                        # 同日同业配对后行业等权，避免两腿行业权重不同伪装成反转。
                        keys = ['信号日序', '行业码']
                        w = weak.groupby(keys)[metric].mean()
                        h = high.groupby(keys)[metric].mean()
                        delta = (w - h).dropna().groupby(level=0).mean()
                        d = delta.index.to_numpy(int)
                        sums = np.zeros(len(cal))
                        counts = np.zeros(len(cal), dtype=int)
                        sums[d], counts[d] = delta.to_numpy(), 1
                        mean = float(delta.mean()) if len(delta) else np.nan
                        item = {'研究': '强行业弱股减强股（同日同业配对）', '时段': pname, '半区': half,
                                '事件': '配对差', '持有日': 20, '大周期': 0, '大周期状态': '不适用',
                                '行业动量日': n, '行业三分位': 3, '个股五分位': 0,
                                '流动性': '全部' if liq == 0 else f'成交额{liq}档', '指标': metric,
                                '原始信号数': len(weak) + len(high), '原始信号日数': strong['信号日序'].nunique(),
                                'PIT准入数': len(weak) + len(high), '可买入数': int(weak['买入状态'].eq(0).sum() + high['买入状态'].eq(0).sum()),
                                '有效完结数': int(weak['退出状态'].eq(0).sum() + high['退出状态'].eq(0).sum()),
                                '指标有效数': len(delta), '指标有效信号日数': len(delta), '均值': mean,
                                '中位数': float(delta.median()) if len(delta) else np.nan,
                                '胜率': float((delta > 0).mean()) if len(delta) else np.nan,
                                '胜率分母': '两腿同日同业配对有效的月末信号日',
                                '样本标记': '统计不足：月末配对信号日少于100，非事件数口径',
                                'CI下界': np.nan, 'CI上界': np.nan, '块bootstrap_p': np.nan,
                                '统一FDR_q': np.nan, '整体maxstat_p': np.nan, 'bootstrap有效次数': 0}
                        row_id = len(rows)
                        rows.append(item)
                        if len(delta):
                            tests.append((row_id, pname, sums, counts, mean))
                        daily_writer.add(pd.DataFrame({'格编号': row_id, '日期': cal[days],
                                                       '收益和': sums[days], '有效事件数': counts[days]}))
                        if row_id % 192 == 0:
                            daily_writer.flush()
    result = pd.DataFrame(rows)
    result.index.name = '格编号'
    daily_writer.close()
    rng = np.random.default_rng(20261030)
    nulls, maxima = [], np.zeros(bootstrap)
    print(f'统一检验{len(tests)}项，每项{bootstrap}次60交易日连续移动块bootstrap', flush=True)
    research_days = np.flatnonzero(cal >= pd.Timestamp('2010-01-01'))
    length, block = len(research_days), 60
    assert length >= block
    # 全检验族共用2010—2020完整日历移动块；每格窗外分子、分母均为0。
    # 因而重叠时段/参数/半区/配对差的共同相关性进入整体max-stat。
    weights = np.zeros((bootstrap, length), dtype=np.float64)
    nb = math.ceil(length / block)
    starts = rng.integers(0, length, size=(bootstrap, nb))
    rid = np.arange(bootstrap)
    for b in range(nb):
        size = min(block, length - b * block)
        for offset in range(size):
            np.add.at(weights, (rid, (starts[:, b] + offset) % length), 1)
    for pname, (start, stop) in PERIODS.items():
        group = [t for t in tests if t[1] == pname]
        if not group:
            continue
        for chunk in range(0, len(group), 192):
            g = group[chunk:chunk + 192]
            sums = np.column_stack([t[2][research_days] for t in g])
            counts = np.column_stack([t[3][research_days] for t in g])
            means = np.array([t[4] for t in g])
            den = weights @ counts
            # 直接在完整日历对收益和减去事件均值×事件数后重采样；不是正态近似。
            centered = weights @ (sums - counts * means)
            errors = np.divide(centered, den, out=np.full_like(centered, np.nan), where=den > 0)
            sigma = np.nanstd(errors, axis=0, ddof=1)
            for j, t in enumerate(g):
                row_id = t[0]
                err = errors[:, j]
                finite = np.isfinite(err)
                z = err[finite]
                if not len(z):
                    continue
                mean = t[4]
                lo, hi = np.quantile(z, [.025, .975])
                result.loc[row_id, ['CI下界', 'CI上界', '块bootstrap_p', 'bootstrap有效次数']] = [
                    mean - hi, mean - lo, (1 + (np.abs(z) >= abs(mean) - 1e-15).sum()) / (1 + len(z)), len(z)]
                scale = sigma[j]
                observed = abs(mean) / scale if scale > 0 else (np.inf if mean else 0.)
                nulls.append((row_id, observed))
                stat = np.divide(np.abs(err), scale, out=np.zeros_like(err), where=finite & (scale > 0))
                maxima = np.maximum(maxima, stat)
        print(f'{pname}完成{len(group)}项', flush=True)
    ids = result.index[result['块bootstrap_p'].notna()].to_numpy()
    p = result.loc[ids, '块bootstrap_p'].to_numpy(float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result.loc[ids[order], '统一FDR_q'] = np.minimum(q, 1)
    for row_id, stat in nulls:
        result.loc[row_id, '整体maxstat_p'] = (1 + (maxima >= stat).sum()) / (bootstrap + 1)
    audit['检验'] = {'全部计划格指标数': len(result), '有数据实际检验数': len(ids),
                    '足量检验数': int(result.loc[ids, '样本标记'].eq('足量').sum()),
                    '统一BH_FDR族': '全部事件/周期/状态/持有日/时段/哈希诊断/排序/流动性/配对差/四指标；不足也纳入',
                    'bootstrap次数': bootstrap, '随机种子': 20261030, '块长交易日': 60,
                    '完整日历': '共用2010-2020全市场日历；格的窗外及无事件日收益和与事件数均0，不是零收益事件；循环连续块可overlap，各日边际抽样概率相同',
                    '估计对象': '事件等权均值，日级收益和/事件数比率；配对差为同日同业差再行业等权',
                    'p值': '收益和逐日减事件均值×事件数后，直接连续块bootstrap双侧p，(1+极端次数)/(1+有效次数)',
                    'CI': '去均值bootstrap基本法95%CI（原均值减误差分位数）',
                    '整体maxstat': '共同日历重采样的全族标准化去均值统计量最大值；不使用正态p',
                    '最小p分辨率': 1 / (bootstrap + 1),
                    'FDR小于等于0.05数': int(result['统一FDR_q'].le(.05).sum())}
    result.reset_index().to_csv(OUT / '04_全部格检验.csv', index=False, encoding='utf-8-sig')
    result[result['研究'] == '跨层事件'].reset_index().to_csv(OUT / '04_跨层事件.csv', index=False, encoding='utf-8-sig')
    result[result['研究'] != '跨层事件'].reset_index().to_csv(OUT / '04_双重排序.csv', index=False, encoding='utf-8-sig')
    return result


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if value is pd.NA:
        return None
    return value


def finish(result, events, monthly, cal, audit):
    audit['执行覆盖'] = {}
    execution = []
    for name, data in [('原始事件', events), ('月末特征候选', monthly)]:
        for pname, (start, stop) in PERIODS.items():
            if name.startswith('月末') and pname.startswith('补充'):
                continue
            d = data[data['信号日'].between(start, stop)]
            group = d.groupby(['事件', '持有日'], observed=True) if name == '原始事件' else [(('月末候选', 20), d)]
            for (event, h), f in group:
                item = {'研究': name, '时段': pname, '事件': event, '持有日': int(h),
                        '原始信号数': len(f), 'PIT准入数': int(f['准入'].sum()),
                        '可买入数': int(f['买入状态'].eq(0).sum()), '有效完结数': int(f['退出状态'].eq(0).sum()),
                        '行业缺失信号数': int(f['行业码'].isna().sum()),
                        '留一基准缺失完结数': int((f['退出状态'].eq(0) & f['相对留一行业'].isna()).sum()),
                        '实际退出越封存线数': int((f['实际退出日序'] >= len(cal)).sum())}
                item.update({f'买入：{v}': int(f['买入状态'].eq(k).sum()) for k, v in BUY.items()})
                item.update({f'退出：{v}': int(f['退出状态'].eq(k).sum()) for k, v in EXIT.items()})
                item['延后退出数'] = int(f['延后日数'].gt(0).sum())
                assert sum(item[f'买入：{v}'] for v in BUY.values()) == len(f)
                assert sum(item[f'退出：{v}'] for v in EXIT.values()) == len(f)
                assert item['有效完结数'] <= item['可买入数'] <= item['PIT准入数'] <= len(f)
                assert item['实际退出越封存线数'] == 0
                execution.append(item)
    pd.DataFrame(execution).to_csv(OUT / '04_执行覆盖.csv', index=False, encoding='utf-8-sig')
    audit['执行覆盖'] = execution
    keys = ['研究', '半区', '事件', '持有日', '大周期', '大周期状态', '行业动量日', '行业三分位', '个股五分位', '流动性', '指标']
    stability = []
    for key, f in result.groupby(keys, dropna=False, observed=True):
        item = dict(zip(keys, key))
        item['两段'] = {}
        signs = []
        for period in ('2015-2017', '2018-2020'):
            z = f[f['时段'] == period]
            if not len(z):
                continue
            r = z.iloc[0]
            mean = r['均值']
            sign = None if not np.isfinite(mean) else int(np.sign(mean))
            signs.append(sign)
            item['两段'][period] = {'均值': mean, '符号': sign, 'CI下界': r['CI下界'],
                                     'CI上界': r['CI上界'], 'q': r['统一FDR_q'],
                                     '有效数': r['指标有效数'], '信号日数': r['指标有效信号日数'], '样本标记': r['样本标记']}
        item['两段同号'] = len(signs) == 2 and signs[0] is not None and signs[0] == signs[1]
        stability.append(item)
    audit['各格分段符号与CI'] = stability
    unavailable = result[result['指标有效数'].eq(0)]
    audit['数据不可用格'] = unavailable.reset_index()[['格编号', *keys, '时段', '原始信号数', '可买入数', '有效完结数', '指标有效数']].to_dict('records')
    audit['统计不足格数'] = int(result['样本标记'].ne('足量').sum())
    audit['重要限制'] = [
        '行业/全A及留一是不可投资、未扣日再平衡费的合成基准；个股仅扣0.254%假设往返成本，不含逐股冲击。',
        '行业映射源表首条记录有ASOF前日NULL误配；本脚本按有效起日<=市场前日主动去除，未改源数据。',
        '行业主窗指标预热从2014-01-01开始，新增行业250日MA在2015年初不足为NA；2010-2014只补充。',
        '卖出收盘触及跌停一律保守不可卖，即便盘中曾成交；不声称真实成交概率，延后最多20日。',
        '买入同时尊重重算涨停和源涨停标记，冲突保守不可买；缺原始preclose/未知板块/ST不猜。',
        '基准冻结信号所属行业；持有期间即使股票改行业也不换基准，仅在实际参与原行业当天剔除自身。',
        '留一行业当日成分资格为上市>=60日且前一市场日价格、当前开收盘有效，扣自身后至少10；首日必须有效日内因子。',
        '停牌原表保留价格及零收益；日历缺行仍NA，不把停牌删除或插值压缩时间。',
        '月底排序分位在全体PIT候选中冻结后才分哈希/流动性；流动性为T-20..T-1日平均amount横截面三分位。',
        '月末主窗最多72个信号日，两段最多36日，所有排序/配对差均低于100信号日底线，不能宣称确认。',
        '固定sha256完整整数最低bit分半只是横截面诊断，非独立时间留出；既有时段被研究过。',
        '全部有限p统一FDR；不足样本也纳入但不能作为通过，bootstrap有限次数限制p分辨率。',
        '未进行额外成本压力/20与120日块敏感性，不能宣称满足冻结设计的全部策略线索条件。',
        '大周期红只为条件分组，不保证小周期能买、能卖或有正超额；本任务为非投资事件研究。',
    ]
    audit['自检'] = {'执行分母恒等式': '通过', '实际退出日期封存约束': '通过',
                     '20日/55日过去收盘不含当日': '全局交易日shift(1)，完整滚动min_periods=n',
                     '行业前日状态': 'T-1 close与T-1 MA，未知单列',
                     '未实现收益补零': False, '旧事件表使用': False}
    (OUT / '04_审计与分段.json').write_text(json.dumps(clean_json(audit), ensure_ascii=False, indent=2, allow_nan=False))
    print('完成；结果仅写04_前缀文件', flush=True)
    print(result[(result['时段'] == '主窗2015-2020') & (result['半区'] == '全部') &
                 (result['研究'] == '跨层事件') & (result['指标'] == '相对留一行业') &
                 (result['大周期状态'] == '红')][['事件', '持有日', '大周期', '均值', 'CI下界', 'CI上界', '统一FDR_q', '样本标记']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=4999)
    parser.add_argument('--reuse', action='store_true', help='仅重算本脚本自己已生成的新明细，非旧事件表')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    frozen_path = ROOT / '证据/研究设计冻结.json'
    design = json.loads(frozen_path.read_text())
    assert design['时间']['研究止'] == '2020-12-31'
    assert OUT.is_dir() and args.bootstrap >= 999
    con = duckdb.connect()
    con.execute('SET threads=4')
    con.execute("SET memory_limit='3GB'")
    source = ROOT / '数据/个股与行业.parquet'
    cal = pd.to_datetime(con.execute(f"SELECT DISTINCT date FROM read_parquet('{source}') ORDER BY date").fetchdf().date).to_numpy()
    cal = pd.DatetimeIndex(cal)
    assert cal.max() <= END
    audit = {'研究设计版本': design['版本'], '证据等级': design['证据等级'],
             '研究封存线': str(END.date()), '成本假设往返': COST,
             '冻结设计SHA256': hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
             '脚本SHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             '源文件': {str(ROOT / '数据' / f): {'字节数': (ROOT / '数据' / f).stat().st_size,
                                                'SHA256': hashlib.sha256((ROOT / '数据' / f).read_bytes()).hexdigest()}
                         for f in ('个股与行业.parquet', '行业合成开收盘.parquet', '全A等权开收盘.parquet')},
             '状态代码': {'买入': BUY, '退出': EXIT, '大周期': STATUS},
             '日序日期对照': {str(i): str(d.date()) for i, d in enumerate(cal)}}
    if args.reuse:
        # 可重算本次新明细；审计核验保留原构建记录。
        old = json.loads((OUT / '04_审计与分段.json').read_text())
        assert old['源文件'] == audit['源文件']
        for k in ('原始异常核验', '覆盖', '市场基准缺价交易日'):
            audit[k] = old[k]
        event_path, monthly_path = OUT / '04_事件明细.parquet', OUT / '04_月末候选明细.parquet'
    else:
        event_path, monthly_path = build_events(con, cal, audit)
    con.close()
    events = pd.read_parquet(event_path)
    monthly = pd.read_parquet(monthly_path)
    for f in (events, monthly):
        assert f['信号日'].max() <= END
        assert (f.loc[f['退出状态'].eq(0), '实际退出日序'] < len(cal)).all()
        f['证券'] = f['证券'].astype('category')
    events['事件'] = events['事件'].astype('category')
    result = aggregate_and_test(events, monthly, cal, args.bootstrap, audit)
    finish(result, events, monthly, cal, audit)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
