"""冻结网格的2010—2020价格指数代理研究；缺口分段，不是时间样本外验证。"""
from pathlib import Path
import hashlib
import json
import os
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
END = pd.Timestamp('2020-12-31')
START = pd.Timestamp('2010-01-01')
NS = [10, 20, 60, 120, 250]
LOOKS = [20, 60, 120, 250]
SKIPS = [0, 20]
TOPS = [3, 5, 10]
COSTS = [0.0005, 0.001, 0.002]
MAIN_COST = 0.001
BOOT = 999
SEED = 20261030


def read_before_2021(path):
    # Parquet谓词下推：不摄入、不评价2021及以后行。
    df = pd.read_parquet(path, filters=[('date', '<=', END)])
    df['date'] = pd.to_datetime(df['date'])
    df['live_start'] = pd.to_datetime(df['live_start'])
    assert df.date.max() <= END
    assert not df[['symbol', 'date']].duplicated().any()
    return df.sort_values(['symbol', 'date'])


def segments(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def prepared(g, calendar, synthetic=False):
    meta = g.iloc[0]
    f = g.set_index('date').reindex(calendar)
    valid = f.open.gt(0) & f.close.gt(0) & np.isfinite(f.open) & np.isfinite(f.close)
    if synthetic:
        valid &= f['成分数'].ge(10) & f['有效价格率'].eq(1)
    # 缺失不前填，收盘指标在完整市场日历上计算。
    f.loc[~valid, ['open', 'close']] = np.nan
    live = max(START, pd.Timestamp(meta.live_start))
    if str(meta['kind']) == '行业价格指数' or synthetic:
        live = max(live, pd.Timestamp('2015-01-01'))
    if str(meta['symbol']) == '000852.SH':
        live = max(live, pd.Timestamp('2014-10-17'))
    if str(meta['symbol']) == '399006.SZ':
        live = max(live, pd.Timestamp('2010-06-01'))
    common = (f.close.rolling(251, min_periods=251).count().shift(1).eq(251)
              & valid & (calendar >= live))
    return f, common.to_numpy(copy=True), meta


def timing_signal(close, rule, n):
    if rule == '均线':
        value = close.rolling(n, min_periods=n).mean()
        signal = close.gt(value).astype(float).where(value.notna() & close.notna())
    else:
        value = close.shift(n)
        mature = close.rolling(n + 1, min_periods=n + 1).count().eq(n + 1)
        signal = close.gt(value).astype(float).where(mature & close.notna() & value.notna())
    return signal.shift(1).to_numpy()


def single_path(op, cl, previous_close, target, cost, daily_rebalance=False):
    """现金+持股账本；二元与分数目标均为费后开盘权重，每日按目标交易。"""
    op, cl, target = map(lambda x: np.asarray(x, float), (op, cl, target))
    overnight = op / np.r_[previous_close, cl[:-1]] - 1
    intraday = cl / op - 1
    close_weight = target * (1 + intraday) / (1 + target * intraday)
    prev = np.r_[0.0, close_weight[:-1]]
    night_factor = 1 + prev * overnight
    open_hold = prev * (1 + overnight) / night_factor
    # 买卖金额均以费前开盘equity为分母；目标金额以费后equity为分母。
    denominator = np.where(target >= open_hold, 1 + cost * target, 1 - cost * target)
    fee = cost * np.abs(target - open_hold) / denominator
    turnover = np.abs(target * (1 - fee) - open_hold)
    assert np.allclose(fee, cost * turnover, rtol=0, atol=1e-14)
    factor = night_factor * (1 - fee) * (1 + target * intraday)
    terminal = cost * close_weight[-1]
    factor[-1] *= 1 - terminal
    assert np.all(factor > 0) and np.all(np.isfinite(factor))
    entries = int(np.sum((target > 0) & (np.r_[0.0, target[:-1]] == 0)))
    return {'log': np.log(factor), 'nav': np.cumprod(factor), 'weight': target,
            'prev': prev, 'open_hold': open_hold, 'overnight': overnight, 'intraday': intraday,
            'turnover': turnover, 'terminal': terminal, 'rounds': entries,
            'events': int(np.count_nonzero(turnover > 1e-12)) + int(close_weight[-1] > 0),
            'side_volume': float(turnover.sum() + close_weight[-1]),
            'cost_factor': 1 - fee}


def month_end_flags(calendar):
    periods = calendar.to_period('M')
    return np.r_[periods[:-1] != periods[1:], True]


def rotation_scores(close, look, skip):
    full = close.rolling(look + skip + 1, min_periods=look + skip + 1).count().eq(look + skip + 1)
    return (close.shift(skip) / close.shift(look + skip) - 1).where(full)


def rotation_schedule(close, calendar, look, skip, k, direction):
    scores = rotation_scores(close, look, skip).to_numpy()
    flags = month_end_flags(calendar)
    frozen = np.full_like(scores, np.nan)
    signal_dates = np.full(len(calendar), -1, dtype=int)
    current = None
    current_date = -1
    # 在i日开盘使用至多i-1日月末收盘排名。
    for i in range(1, len(calendar)):
        if flags[i - 1]:
            finite = np.flatnonzero(np.isfinite(scores[i - 1]))
            current = None
            if len(finite) >= k:
                order = np.argsort(scores[i - 1, finite], kind='stable')
                picked = finite[order[-k:]] if direction == '买最强' else finite[order[:k]]
                current = np.zeros(close.shape[1])
                current[picked] = 1 / k
            current_date = i - 1
        if current is not None:
            frozen[i] = current
            signal_dates[i] = current_date
    return frozen, signal_dates


def portfolio_path(op, cl, previous_close, targets, rebalance, cost):
    """现金+逐股持仓；费后目标用固定点求费，非调仓日保持真实漂移且无费用。"""
    op, cl = np.asarray(op, float), np.asarray(cl, float)
    overnight = op / np.vstack([previous_close, cl[:-1]]) - 1
    intraday = cl / op - 1
    t, assets = op.shape
    previous = np.zeros(assets)
    logs, weights, turnovers, fees, old, opening = [], [], [], [], [], []
    for i in range(t):
        old.append(previous.copy())
        night_factor = 1 + previous @ overnight[i]
        open_hold = previous * (1 + overnight[i]) / night_factor
        opening.append(open_hold.copy())
        fee = 0.0
        if rebalance[i]:
            new = targets[i].copy()
            for _ in range(20):
                next_fee = cost * float(np.abs(new * (1 - fee) - open_hold).sum())
                converged = abs(next_fee - fee) <= 1e-14
                fee = next_fee
                if converged:
                    break
            assert converged, '费后权重固定点未收敛'
            traded = float(np.abs(new * (1 - fee) - open_hold).sum())
            assert abs(fee - cost * traded) <= 1e-14
        else:
            new = open_hold
            traded = 0.0
        fee_factor = 1 - fee
        day_factor = 1 + new @ intraday[i]
        logs.append(np.log(night_factor * fee_factor * day_factor))
        weights.append(new.copy())
        turnovers.append(traded)
        fees.append(fee_factor)
        previous = new * (1 + intraday[i]) / day_factor
    terminal = cost * previous.sum()
    logs[-1] += np.log1p(-terminal)
    logs = np.asarray(logs)
    assert np.isfinite(logs).all()
    nonzero = np.asarray(turnovers) > 1e-12
    return {'log': logs, 'nav': np.exp(np.cumsum(logs)), 'weight': np.asarray(weights),
            'prev': np.asarray(old), 'open_hold': np.asarray(opening),
            'overnight': overnight, 'intraday': intraday,
            'turnover': np.asarray(turnovers), 'terminal': float(terminal),
            'rounds': int(nonzero.sum()), 'events': int(nonzero.sum()) + int(previous.sum() > 0),
            'side_volume': float(np.sum(turnovers) + previous.sum()),
            'cost_factor': np.asarray(fees)}


def annual(log):
    return float(np.expm1(np.mean(log) * 252))


def mdd(log):
    wealth = np.r_[1.0, np.exp(np.cumsum(log))]
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1))


def metrics(path):
    weight = path['weight']
    exposure = weight if weight.ndim == 1 else weight.sum(axis=1)
    return {'净年化(%)': 100 * annual(path['log']), '最大回撤(%)': 100 * mdd(path['log']),
            '平均开盘仓位(%)': 100 * float(exposure.mean()), '交易轮次': path['rounds'],
            '交易事件数含清仓': path['events'], '单边换手总量': path['side_volume'],
            '末日清仓费率(%)': 100 * path['terminal'], '末日净值': float(path['nav'][-1])}


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    order = idx[np.argsort(p[idx])]
    if len(order):
        adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
        q[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return q


def moving_weights(length, block, repeats, rng):
    assert block <= length
    count = int(np.ceil(length / block))
    weights = np.zeros((repeats, length))
    for b in range(repeats):
        starts = rng.integers(0, length, count)
        draws = ((starts[:, None] + np.arange(block)) % length).ravel()[:length]
        weights[b] = np.bincount(draws, minlength=length)
    return weights


def inference(tests, calendar, rows, block, repeats=BOOT):
    # 共享真实日历上的循环移动块；空缺为mask，不把有效日压缩后再抽块。
    chosen = [t for t in tests if t['n'] >= 2 * block and (block == 60 or t['main'])]
    output = []
    if not chosen:
        return {'块长': block, '次数': repeats, '纳入maxstat假设数': 0, '不足长度或非主成本数': len(tests)}
    width = len(chosen)
    values = np.zeros((len(calendar), width))
    mask = np.zeros_like(values)
    mean = np.empty(width)
    for j, test in enumerate(chosen):
        idx = test['idx']
        mean[j] = test['delta'].mean()
        values[idx, j] = test['delta'] - mean[j]
        mask[idx, j] = 1
    weights = moving_weights(len(calendar), block, repeats, np.random.default_rng(SEED + block))
    total = weights @ values
    counts = weights @ mask
    good = counts >= np.maximum(2, np.array([t['n'] for t in chosen]) * 0.1)[None, :]
    null = np.divide(total, counts, out=np.full_like(total, np.nan), where=good)
    eligible = good.mean(axis=0) >= 0.8
    se = np.nanstd(null, axis=0, ddof=1)
    # 零差值检验合法但不进入studentized max-stat，p=q=1、区间为0。
    active = eligible & np.isfinite(se) & (se > 1e-15)
    if active.any():
        z = np.abs(null[:, active] / se[active])
        maximum = np.nanmax(z, axis=1)
    else:
        maximum = np.zeros(repeats)
    for j, test in enumerate(chosen):
        if not eligible[j]:
            continue
        dist = null[:, j]
        dist = dist[np.isfinite(dist)]
        zero = se[j] <= 1e-15
        p = 1.0 if zero and abs(mean[j]) < 1e-15 else (1 + np.count_nonzero(np.abs(dist) >= abs(mean[j]))) / (len(dist) + 1)
        maxp = (1.0 if zero and abs(mean[j]) < 1e-15 else
                (1 + np.count_nonzero(maximum >= abs(mean[j]) / max(se[j], 1e-15))) / (repeats + 1))
        low, high = (mean[j] + np.quantile(dist, [0.025, 0.975])) * 252 * 100
        row = rows[test['row']]
        prefix = test['reference'] + f'_块{block}'
        row[prefix + '_年化对数差下界(%)'] = float(low)
        row[prefix + '_年化对数差上界(%)'] = float(high)
        row[prefix + '_中心化双侧p'] = float(p)
        row[prefix + '_全族maxstat_p'] = float(maxp)
        output.append((test['row'], prefix, p))
    # 统一所有报告的可估计假设：资产、规则、窗口、成本、方向、两个基准及合成敏感性。
    q = bh_fdr([x[2] for x in output])
    for (row_index, prefix, _), value in zip(output, q):
        rows[row_index][prefix + '_同块长整族BH_FDR_q'] = float(value)
    return {'块长': block, '次数': repeats, '随机种子': SEED + block,
            '可估计假设数': len(output), '纳入maxstat假设数': int(active.sum()),
            '零差值非studentized数': int((eligible & ~active).sum()),
            '不足长度或非主成本数': len(tests) - len(chosen),
            '抽样有效不足排除数': int((~eligible).sum()),
            'maxstat范围': '本02全部数据组的时序与轮动可估计假设共享循环日历抽样；60块包含所有成本，20/120块包含所有组10bp主成本；非其它子任务整体maxstat',
            'NA规则': '有效段不足2倍块长不做该块检验；不压缩缺失日；无效抽样不足80%则NA'}


def unit_tests():
    checks = []
    def check(name, assertion):
        assert assertion, name
        checks.append({'名称': name, '通过': True})
    op = np.array([110., 105., 120., 125.])
    cl = np.array([115., 100., 130., 140.])
    cash = single_path(op, cl, 100., np.zeros(4), 0.001)
    check('全现金净值恒为1且收益费用均0', np.array_equal(cash['nav'], np.ones(4)))
    full = single_path(op, cl, 100., np.ones(4), 0.001)
    expected = cl[-1] / op[0] * (1 - 0.001) / (1 + 0.001)
    check('全持有匹配首开买入末收卖出BH', np.isclose(full['nav'][-1], expected, rtol=1e-12))
    exit_path = single_path(op, cl, 100., [1, 0, 0, 1], 0.001)
    check('隔夜归旧仓新仓不承隔夜', np.isclose(exit_path['log'][1], np.log(op[1] / cl[0] * 0.999)))
    check('首日新开仓不赚取前收至首开的收益', np.isclose(full['log'][0], np.log(cl[0] / op[0] / 1.001)))
    no_fee = single_path(op, cl, 100., np.ones(4), 0.)
    check('末次退出费计在末日', np.isclose(full['log'][-1] - no_fee['log'][-1], np.log(0.999)))
    dates = pd.bdate_range('2018-01-01', periods=700)
    series = pd.Series(100 + np.arange(700) + 15 * np.sin(np.arange(700) / 11), index=dates)
    changed = series.copy()
    changed.iloc[520:] *= 7
    for rule in ['均线', '动量']:
        for n in NS:
            check(f'未来数据不改变过去时序仓位_{rule}{n}', np.allclose(
                timing_signal(series, rule, n)[:520], timing_signal(changed, rule, n)[:520], equal_nan=True))
    close = pd.DataFrame({'A': series, 'B': 800 - np.arange(700) * 0.3}, index=dates)
    target, frozen = rotation_schedule(close, dates, 20, 0, 1, '买最强')
    altered = close.copy()
    altered.iloc[520:] *= np.array([0.01, 100])
    other, _ = rotation_schedule(altered, dates, 20, 0, 1, '买最强')
    check('未来数据不改变过去轮动目标', np.allclose(target[:520], other[:520], equal_nan=True))
    for i in np.flatnonzero(month_end_flags(dates)):
        if i + 1 < len(dates) and np.isfinite(target[i + 1]).all():
            check('月末排名只能下一交易日open执行', frozen[i + 1] == i and frozen[i] < i)
            break
    pc = np.array([100., 100.])
    po = np.array([[100., 100.], [200., 50.]])
    pp = portfolio_path(po, po.copy(), pc, np.array([[1., 0.], [0., 1.]]), [True, True], 0.)
    check('轮动隔夜归卖出前组合', np.isclose(pp['nav'][-1], 2.))
    bh = portfolio_path(op[:, None], cl[:, None], [100.], np.ones((4, 1)), [True, False, False, False], 0.001)
    check('组合全持有亦匹配BH含首尾费', np.isclose(bh['nav'][-1], expected))
    constant = single_path(op, cl, 100., np.full(4, 0.5), 0.001, daily_rebalance=True)
    check('同暴露每日重置有漂移再平衡费', constant['turnover'][1:].sum() > 0)
    broken = series.copy()
    broken.iloc[510] = np.nan
    check('缺失交易日不压缩指标预热', np.isnan(timing_signal(broken, '动量', 250)[600]))
    check('BH-FDR排序与单调', np.allclose(bh_fdr([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04]))
    rng = np.random.default_rng(42)
    random_open = np.exp(rng.normal(0, .02, 300)).cumprod() * 100
    random_close = random_open * np.exp(rng.normal(0, .02, 300))
    random_target = rng.integers(0, 2, 300).astype(float)
    vector = single_path(random_open, random_close, 100., random_target, .002)
    cash_amount, shares = 1., 0.
    ledger = []
    for i in range(300):
        if random_target[i] == 1 and shares == 0:
            shares = cash_amount / (random_open[i] * (1 + .002))
            cash_amount -= shares * random_open[i] * (1 + .002)
        elif random_target[i] == 0 and shares > 0:
            cash_amount += shares * random_open[i] * (1 - .002)
            shares = 0.
        ledger.append(cash_amount + shares * random_close[i])
    cash_amount += shares * random_close[-1] * (1 - .002)
    ledger[-1] = cash_amount
    check('300日独立现金股数账本匹配二元含首末费用', np.allclose(vector['nav'], ledger, rtol=1e-12))

    def cash_ledger(open_prices, close_prices, targets, rebalance, cost):
        # 独立金额账本：二分求可支付交易费的资金，不使用权重漂移或费用固定点公式。
        cash_amount = 1.
        shares = np.zeros(open_prices.shape[1])
        nav, volumes, opening, fees = [], [], [], []
        for prices, closes, new, trade in zip(open_prices, close_prices, targets, rebalance):
            held_amount = shares * prices
            equity = cash_amount + held_amount.sum()
            opening.append(held_amount / equity)
            traded_amount = paid = 0.
            if trade:
                low, high = 0., equity
                for _ in range(100):
                    remaining = (low + high) / 2
                    required = remaining + cost * np.abs(new * remaining - held_amount).sum()
                    if required > equity:
                        high = remaining
                    else:
                        low = remaining
                change = new * ((low + high) / 2) - held_amount
                traded_amount = np.abs(change).sum()
                paid = cost * traded_amount
                cash_amount -= change.sum() + paid
                shares += change / prices
            volumes.append(traded_amount / equity)
            fees.append(paid / equity)
            nav.append(cash_amount + shares @ closes)
            assert cash_amount >= -1e-12
        cash_amount += np.sum(shares * close_prices[-1] * (1 - cost))
        nav[-1] = cash_amount
        return np.array(nav), np.array(volumes), np.array(opening), np.array(fees)

    fractional = rng.uniform(0, 1, 300)
    for cost in COSTS:
        target_path = single_path(random_open, random_close, 100., fractional, cost)
        nav, volume, opening, fee = cash_ledger(random_open[:, None], random_close[:, None],
                                               fractional[:, None], np.ones(300, bool), cost)
        check(f'300日fractional现金账本净值与过夜换手_{cost}',
              np.allclose(target_path['nav'], nav, rtol=1e-12)
              and np.allclose(target_path['turnover'], volume, atol=1e-12)
              and np.allclose(target_path['open_hold'], opening[:, 0], atol=1e-12)
              and np.allclose(1 - target_path['cost_factor'], fee, atol=1e-14))
        daily = single_path(random_open, random_close, 100., fractional, cost, daily_rebalance=True)
        check(f'二元与fractional不分旧近似分支_{cost}', np.array_equal(target_path['nav'], daily['nav']))
        one = portfolio_path(random_open[:, None], random_close[:, None], [100.],
                             fractional[:, None], np.ones(300, bool), cost)
        check(f'单股portfolio与single现金账本等价_{cost}', np.allclose(one['nav'], nav, rtol=1e-12))

    gap_open = np.array([100., 200., 50.])
    gap = single_path(gap_open, gap_open, 100., np.full(3, .5), .002)
    nav, volume, opening, fee = cash_ledger(gap_open[:, None], gap_open[:, None],
                                           np.full((3, 1), .5), np.ones(3, bool), .002)
    check('纯隔夜漂移先卖后买费用与换手独立金额验证',
          np.allclose(gap['nav'], nav, rtol=1e-12) and np.allclose(gap['turnover'], volume, atol=1e-12)
          and np.isclose(opening[1, 0], 2 / 3) and np.isclose(opening[2, 0], 1 / 5)
          and np.all(gap['turnover'][1:] > 0) and np.all(fee[1:] > 0))
    multi_open = np.exp(rng.normal(0, .02, (300, 4))).cumprod(axis=0) * 100
    multi_close = multi_open * np.exp(rng.normal(0, .02, (300, 4)))
    multi_target = rng.dirichlet(np.ones(5), 300)[:, :4]
    rebalance = np.arange(300) % 21 == 0
    for cost in COSTS:
        portfolio = portfolio_path(multi_open, multi_close, np.full(4, 100.), multi_target, rebalance, cost)
        nav, volume, opening, fee = cash_ledger(multi_open, multi_close, multi_target, rebalance, cost)
        check(f'300日多股月中真漂移与费后目标独立金额验证_{cost}',
              np.allclose(portfolio['nav'], nav, rtol=1e-12)
              and np.allclose(portfolio['turnover'], volume, atol=1e-12)
              and np.allclose(portfolio['open_hold'], opening, atol=1e-12)
              and np.allclose(1 - portfolio['cost_factor'], fee, atol=1e-14)
              and np.all(portfolio['cost_factor'][~rebalance] == 1)
              and np.all(portfolio['turnover'][~rebalance] == 0)
              and np.array_equal(portfolio['weight'][~rebalance], portfolio['open_hold'][~rebalance]))
        allocation = np.full((300, 4), .25)
        fullhold = portfolio_path(multi_open, multi_close, np.full(4, 100.), allocation,
                                  np.arange(300) == 0, cost)
        nav, _, _, _ = cash_ledger(multi_open, multi_close, allocation, np.arange(300) == 0, cost)
        stock_nav = multi_close / multi_open[0] / (1 + cost)
        stock_nav[-1] *= 1 - cost
        check(f'多股fullhold等价逐股cashledger_{cost}',
              np.allclose(fullhold['nav'], nav, rtol=1e-12)
              and np.allclose(fullhold['nav'], stock_nav.mean(axis=1), rtol=1e-12))
    draws = moving_weights(31, 7, 100000, np.random.default_rng(SEED))
    check('循环移动块末块exact长度且非负整数计数',
          np.all(draws.sum(axis=1) == 31) and np.all(draws >= 0) and np.all(draws == np.floor(draws)))
    check('循环抽样100000次端点与所有每日边际期望均为1', np.max(np.abs(draws.mean(axis=0) - 1)) < .025)
    whole = moving_weights(31, 31, 200, np.random.default_rng(5))
    check('整日历循环块各日恰出现一次', np.array_equal(whole, np.ones_like(whole)))
    return checks


class Research:
    def __init__(self, calendar):
        self.calendar = calendar
        self.eval_calendar = calendar[calendar >= START]
        self.loc = {d: i for i, d in enumerate(self.eval_calendar)}
        self.rows, self.tests, self.curves, self.events, self.windows, self.coverage = [], [], [], [], [], []
        self.mask_rows = []
        self.segment_rows = []

    def add(self, info, dates, path, references):
        row_index = len(self.rows)
        row = dict(info)
        row.update({'试验编号': f'T{row_index + 1:05d}', '有效段首日': str(dates[0].date()),
                    '有效段末日': str(dates[-1].date()), '有效交易日数': len(dates),
                    '主成本(bp)': int(round(MAIN_COST * 10000)),
                    '成本角色': '主成本10bp' if info['单边成本(bp)'] == 10 else '5/20bp并列敏感性，不择优',
                    '证据标签': '样本内描述；指数/合成非可投资基准；不是ETF'})
        row.update(metrics(path))
        for reference, base in references.items():
            row[reference + '净年化(%)'] = 100 * annual(base['log'])
            row[reference + '最大回撤(%)'] = 100 * mdd(base['log'])
            row[reference + '单边换手总量'] = base['side_volume']
            delta = path['log'] - base['log']
            row['超' + reference + '_年化对数差(%)'] = 100 * float(delta.mean() * 252)
            row['超' + reference + '_CAGR差(百分点)'] = 100 * (annual(path['log']) - annual(base['log']))
            self.tests.append({'row': row_index, 'reference': '超' + reference,
                               'idx': np.array([self.loc[d] for d in dates]), 'delta': delta,
                               'n': len(dates), 'main': info['单边成本(bp)'] == 10})
        era_ref = '同暴露' if '同暴露' in references else '行业等权'
        era = row['超' + era_ref + '_年化对数差(%)']
        row['ERA定义'] = '策略减同暴露/轮动减同行业集合等权的日对数差年化'
        row['ERA符号'] = '正' if era > 1e-12 else ('负' if era < -1e-12 else '零')
        self.rows.append(row)
        bins = [('2010-2012', '2010-01-01', '2012-12-31'), ('2013-2015', '2013-01-01', '2015-12-31'),
                ('2016-2017', '2016-01-01', '2017-12-31'), ('2018-2020', '2018-01-01', '2020-12-31')]
        if '行业' in str(info['资产类别']):
            bins = [('2015-2017', '2015-01-01', '2017-12-31'), ('2018-2020', '2018-01-01', '2020-12-31')]
        if info['单边成本(bp)'] == 10:
            for label, start, stop in bins:
                selected = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(stop))
                if selected.any():
                    diagnostic = {'试验编号': row['试验编号'], '轨道': info['轨道'], '诊断窗口': label,
                                  '实际首日': str(dates[selected][0].date()), '实际末日': str(dates[selected][-1].date()),
                                  '有效日数': int(selected.sum()), '净年化(%)': 100 * annual(path['log'][selected]),
                                  '口径': '沿有效段既有资金曲线截取；非独立新试验，不在子窗额外重启或补价'}
                    for reference, base in references.items():
                        d = path['log'][selected] - base['log'][selected]
                        diagnostic['超' + reference + '_年化对数差(%)'] = 100 * float(d.mean() * 252)
                        diagnostic['超' + reference + '_CAGR差(百分点)'] = 100 * (annual(path['log'][selected]) - annual(base['log'][selected]))
                    self.windows.append(diagnostic)
            exposure = path['weight'] if path['weight'].ndim == 1 else path['weight'].sum(axis=1)
            old_exposure = path['prev'] if path['prev'].ndim == 1 else path['prev'].sum(axis=1)
            opening = path['open_hold'] if path['open_hold'].ndim == 1 else path['open_hold'].sum(axis=1)
            curve = pd.DataFrame({'试验编号': row['试验编号'], '轨道': info['轨道'], '数据组': info['数据组'],
                                  '资产代码': info['资产代码'], '单边成本(bp)': info['单边成本(bp)'],
                                  '规则': info['规则'], '回看': info['回看'], '跳过': info['跳过'], '持仓数': info['持仓数'],
                                  '有效段': info['有效段'], '日期': dates.strftime('%Y-%m-%d'),
                                  '策略净值': path['nav'], '策略日对数收益(%)': 100 * path['log'],
                                  '旧收盘总仓位(%)': 100 * old_exposure, '费前开盘持仓(%)': 100 * opening,
                                  '新费后开盘总仓位(%)': 100 * exposure,
                                  '开盘单边换手(%)': 100 * path['turnover'], '开盘成本乘数': path['cost_factor'],
                                  '末日强制清仓费率(%)': 100 * np.r_[np.zeros(len(dates) - 1), path['terminal']],
                                  '初始资金': 1.0, '标签': '主成本10bp；费前equity分母换手；每段资金1；合成不是ETF'})
            if path['weight'].ndim == 1:
                curve['资产隔夜收益(%)'] = 100 * path['overnight']
                curve['资产日内收益(%)'] = 100 * path['intraday']
            else:
                curve['旧组合隔夜贡献(%)'] = 100 * np.sum(path['prev'] * path['overnight'], axis=1)
                curve['新组合日内贡献(%)'] = 100 * np.sum(path['weight'] * path['intraday'], axis=1)
            for reference, base in references.items():
                curve[reference + '净值'] = base['nav']
                curve[reference + '日对数收益(%)'] = 100 * base['log']
                curve[reference + '开盘换手(%)'] = 100 * base['turnover']
            self.curves.append(curve)
        return row['试验编号']

    def record_coverage(self, meta, f, mask, group):
        periods = segments(mask)
        longest = max(periods, key=lambda p: p[1] - p[0]) if periods else None
        missing = f.index[f.open.isna() | f.close.isna()]
        self.coverage.append({'数据组': group, '资产代码': meta['symbol'], '资产名称': meta['name'],
                              '资产类别': meta['kind'], '日历首日': str(f.index[0].date()),
                              '日历末日': str(f.index[-1].date()), '缺失价格交易日数': len(missing),
                              '缺失价格交易日': ';'.join(missing.strftime('%Y-%m-%d')),
                              '有效段数': len(periods), '总有效日数': int(mask.sum()),
                              '首个有效日': str(f.index[np.flatnonzero(mask)[0]].date()) if mask.any() else '',
                              '最长段首日': str(f.index[longest[0]].date()) if longest else '',
                              '最长段末日': str(f.index[longest[1] - 1].date()) if longest else '',
                              '缺口处理': '分段终止并末收清仓；成熟条件重新满足后新段首开进场；不是可交易的缺口预知规则'})
        for i, (a, b) in enumerate(periods, 1):
            self.segment_rows.append({'数据组': group, '资产代码': meta['symbol'], '有效段': i,
                                      '首日': str(f.index[a].date()), '末日': str(f.index[b - 1].date()),
                                      '有效日数': b - a, '最长连续段': (a, b) == longest})
        for i in np.flatnonzero((f.index >= START) & ~mask):
            self.mask_rows.append({'数据组': group, '资产代码': meta['symbol'], '日期': str(f.index[i].date()),
                                   '原因': '缺价/当时未发布/未满共同最长指标预热或月末排名不可用',
                                   '收益处理': 'NA，未补零、未补价、未将日期压缩'})
        return periods, longest

    def timing(self, data, group='官方及全A主网格', matched=None):
        for symbol, g in data.groupby('symbol', sort=True):
            synthetic = group in ('行业合成敏感性', '完整行业合成')
            f, mask, meta = prepared(g, self.calendar, synthetic)
            if matched is not None:
                official_symbol = symbol.replace('EW_SW', 'SW')
                mask &= matched[official_symbol]
            periods, longest = self.record_coverage(meta, f, mask, group)
            for rule in ['均线', '动量']:
                for n in NS:
                    weights = timing_signal(f.close, rule, n)
                    assert np.isfinite(weights[mask]).all()
                    for cost in ([MAIN_COST] if group == '行业合成敏感性' else COSTS):
                        for seg, (a, b) in enumerate(periods, 1):
                            op, cl = f.open.iloc[a:b].to_numpy(), f.close.iloc[a:b].to_numpy()
                            pc = float(f.close.iloc[a - 1])
                            path = single_path(op, cl, pc, weights[a:b], cost)
                            bh = single_path(op, cl, pc, np.ones(b - a), cost)
                            exposure = np.full(b - a, weights[a:b].mean())
                            equal = single_path(op, cl, pc, exposure, cost, daily_rebalance=True)
                            info = {'轨道': '时序趋势', '数据组': group, '资产代码': symbol,
                                    '资产名称': meta['name'], '资产类别': meta['kind'], '规则': rule,
                                    '回看': n, '跳过': 0, '持仓数': 1, '单边成本(bp)': int(round(cost * 10000)),
                                    '有效段': seg, '最长连续段': (a, b) == longest,
                                    '资金曲线口径': '段内完整资金曲线；各段独立首开买入末收退出；禁止跨缺口CAGR',
                                    '交易轮次口径': '从开仓至退出的完整轮次，末收强制退出也计费'}
                            self.add(info, self.calendar[a:b], path, {'买入持有': bh, '同暴露': equal})
            print('时序', group, symbol, '有效段', len(periods), flush=True)

    def rotation(self, data, group='官方行业主网格', matched_mask=None):
        groups = []
        for symbol, g in data.groupby('symbol', sort=True):
            f, _, meta = prepared(g, self.calendar, group in ('行业合成敏感性', '完整行业合成'))
            groups.append((symbol, meta['name'], f))
        symbols = [x[0] for x in groups]
        names = [x[1] for x in groups]
        op = pd.DataFrame({s: f.open for s, _, f in groups}, index=self.calendar)
        cl = pd.DataFrame({s: f.close for s, _, f in groups}, index=self.calendar)
        # 固定28行业集合，不因缺价临时改集合。最长250+skip20需要271个完整收盘。
        mature = cl.rolling(271, min_periods=271).count().eq(271).all(axis=1).to_numpy()
        last_mature = np.zeros(len(cl), bool)
        flags = month_end_flags(self.calendar)
        state = False
        for i in range(1, len(cl)):
            if flags[i - 1]:
                state = bool(mature[i - 1])
            last_mature[i] = state
        mask = (last_mature & np.r_[False, mature[:-1]] & op.notna().all(axis=1).to_numpy()
                & cl.notna().all(axis=1).to_numpy() & (self.calendar >= pd.Timestamp('2015-01-01')))
        if matched_mask is not None:
            mask &= matched_mask
        pseudo = pd.DataFrame({'open': op.where(op.notna().all(axis=1)).mean(axis=1),
                               'close': cl.where(cl.notna().all(axis=1)).mean(axis=1)}, index=self.calendar)
        meta = {'symbol': '28行业固定集合', 'name': '28行业月末横截面轮动',
                'kind': '行业等权合成' if group in ('行业合成敏感性', '完整行业合成') else '行业价格指数'}
        periods, longest = self.record_coverage(meta, pseudo, mask, group)
        for look in LOOKS:
            for skip in SKIPS:
                for k in TOPS:
                    for direction in ['买最强', '买最弱']:
                        target, signal_dates = rotation_schedule(cl, self.calendar, look, skip, k, direction)
                        assert np.isfinite(target[mask]).all()
                        for cost in ([MAIN_COST] if group == '行业合成敏感性' else COSTS):
                            for seg, (a, b) in enumerate(periods, 1):
                                rebalance = np.r_[True, flags[a:b - 1]]
                                op_array, cl_array = op.iloc[a:b].to_numpy(), cl.iloc[a:b].to_numpy()
                                pc = cl.iloc[a - 1].to_numpy()
                                path = portfolio_path(op_array, cl_array, pc, target[a:b], rebalance, cost)
                                equal = portfolio_path(op_array, cl_array, pc, np.full((b - a, len(symbols)), 1 / len(symbols)),
                                                       np.ones(b - a, bool), cost)
                                info = {'轨道': '行业轮动', '数据组': group, '资产代码': '28行业固定集合',
                                        '资产名称': '固定当期可知28行业；旧采掘保留至2020', '资产类别': meta['kind'],
                                        '规则': direction, '回看': look, '跳过': skip, '持仓数': k,
                                        '单边成本(bp)': int(round(cost * 10000)), '有效段': seg,
                                        '最长连续段': (a, b) == longest,
                                        '资金曲线口径': '各段独立；月末排名次开换仓，月中真实漂移持有',
                                        '交易轮次口径': '发生实际开盘交易的组合换仓次数；清仓另列交易事件'}
                                trial = self.add(info, self.calendar[a:b], path, {'行业等权': equal})
                                if cost == MAIN_COST:
                                    for j in np.flatnonzero(rebalance):
                                        i = a + j
                                        for col in range(len(symbols)):
                                            self.events.append({'试验编号': trial, '数据组': group, '有效段': seg,
                                                                '排名收盘日': str(self.calendar[signal_dates[i]].date()),
                                                                '执行开盘日': str(self.calendar[i].date()), '行业代码': symbols[col],
                                                                '行业名称': names[col], '目标费后开盘权重(%)': 100 * float(target[i, col]),
                                                                '调仓前旧收盘权重(%)': 100 * float(path['prev'][j, col]),
                                                                '调仓前费前开盘权重(%)': 100 * float(path['open_hold'][j, col]),
                                                                '回看': look, '跳过': skip, '持仓数': k, '方向': direction,
                                                                '标签': '段首使用最近既有月末排名；非当月未来排名'})
        print('轮动', group, '有效段', len(periods), '有效日', int(mask.sum()), flush=True)
        return mask, symbols


def save_csv(records, path):
    df = pd.concat(records, ignore_index=True) if records and isinstance(records[0], pd.DataFrame) else pd.DataFrame(records)
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.10g')
    return len(df)


def main():
    started = time.time()
    freeze_path = ROOT / '证据/研究设计冻结.json'
    freeze = json.loads(freeze_path.read_text())
    assert freeze['待验证维度']['择时回看'] == NS
    assert freeze['待验证维度']['执行成本单边'] == COSTS
    assert freeze['待验证维度']['行业轮动回看'] == LOOKS
    assert freeze['待验证维度']['行业轮动跳过'] == SKIPS
    assert freeze['待验证维度']['行业轮动持仓数'] == TOPS
    assert freeze['时间']['研究止'] == '2020-12-31'
    source_paths = [ROOT / '数据/资产开收盘.parquet', ROOT / '数据/行业合成开收盘.parquet']
    input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    print('全量运行前确认源SHA256', json.dumps(input_hashes, ensure_ascii=False), flush=True)
    print('MAIN_COST=10bp；5/10/20bp并列报告；BOOT=999，不降低统计次数', flush=True)
    checks = unit_tests()
    print('单元验证通过', len(checks), flush=True)
    assets = read_before_2021(source_paths[0])
    synth = read_before_2021(source_paths[1])
    assert assets.symbol.nunique() == 35
    industry = assets[assets.kind.eq('行业价格指数')].copy()
    assert industry.symbol.nunique() == 28
    assert all(int(s[2:]) < 801950 for s in industry.symbol.unique())
    calendar = pd.DatetimeIndex(assets.loc[assets.symbol.eq('000300.SH'), 'date'].sort_values())
    assert assets.date.isin(calendar).all() and synth.date.isin(calendar).all()
    research = Research(calendar)
    official_masks = {s: prepared(g, calendar)[1] for s, g in industry.groupby('symbol')}
    research.timing(assets)
    rotation_mask, symbols = research.rotation(industry)
    research.timing(synth, '行业合成敏感性', official_masks)
    research.rotation(synth, '行业合成敏感性', rotation_mask)
    research.timing(synth, '完整行业合成')
    research.rotation(synth, '完整行业合成')
    reports = []
    for block in [60, 20, 120]:
        print('共享日历块bootstrap', block, '试验', len(research.rows), '假设', len(research.tests), flush=True)
        reports.append(inference(research.tests, research.eval_calendar, research.rows, block))
    # 除同块长整族q外，再把全部主检验和块长敏感性合为一个更宽的FDR族。
    all_p = [(i, key, value) for i, row in enumerate(research.rows)
             for key, value in row.items() if key.endswith('_中心化双侧p')]
    for (i, key, _), q in zip(all_p, bh_fdr([item[2] for item in all_p])):
        research.rows[i][key.replace('_中心化双侧p', '_全块长统一BH_FDR_q')] = float(q)
    for row in research.rows:
        for block in [60, 20, 120]:
            has_p = any(key.endswith(f'_块{block}_中心化双侧p') for key in row)
            row[f'块{block}检验状态'] = ('已计算' if has_p else
                ('非主成本未做敏感性' if block != 60 and row['单边成本(bp)'] != 10 else '有效段过短或有效重抽不足，NA'))
    counts = {}
    rows = research.rows
    for label, selected in [('时序趋势', [r for r in rows if r['轨道'] == '时序趋势']),
                            ('行业轮动', [r for r in rows if r['轨道'] == '行业轮动'])]:
        counts[label + '网格.csv'] = save_csv(selected, ROOT / ('结果/02_' + label + '网格.csv'))
        curves = [c for c in research.curves if c['轨道'].iloc[0] == label]
        counts[label + '主成本资金曲线.csv'] = save_csv(curves, ROOT / ('结果/02_' + label + '主成本资金曲线.csv'))
    counts['分窗稳定性'] = save_csv(research.windows, ROOT / '结果/02_分窗稳定性.csv')
    counts['轮动排名与目标权重'] = save_csv(research.events, ROOT / '结果/02_轮动排名与目标权重.csv')
    counts['有效区间与覆盖'] = save_csv(research.coverage, ROOT / '结果/02_有效区间与覆盖.csv')
    counts['有效分段'] = save_csv(research.segment_rows, ROOT / '结果/02_有效分段.csv')
    counts['日历缺口与预热mask'] = save_csv(research.mask_rows, ROOT / '结果/02_日历缺口与预热mask.csv')
    main_rows = [r for r in rows if r['单边成本(bp)'] == 10
                 and r['数据组'] in ('官方及全A主网格', '官方行业主网格') and r['最长连续段']]
    positive = 0
    discoveries = 0
    for row in main_rows:
        reference = '超同暴露' if row['轨道'] == '时序趋势' else '超行业等权'
        positive += row['ERA符号'] == '正'
        discoveries += row.get(reference + '_块60_全块长统一BH_FDR_q', 1) <= 0.05
    group_names = sorted({r['数据组'] for r in rows})
    parameter_fields = ['轨道', '资产代码', '规则', '回看', '跳过', '持仓数', '单边成本(bp)']
    actual_grids = {group: len({tuple(r[k] for k in parameter_fields) for r in rows if r['数据组'] == group})
                    for group in group_names}
    complete_rows = [r for r in rows if r['数据组'] == '完整行业合成' and r['最长连续段']]
    windows = {}
    for window in research.windows:
        windows.setdefault(window['试验编号'], []).append(window)
    key_fields = ['轨道', '资产代码', '规则', '回看', '跳过', '持仓数', '有效段']
    costs_by_key = {}
    for row in complete_rows:
        costs_by_key.setdefault(tuple(row[k] for k in key_fields), {})[row['单边成本(bp)']] = row
    complete_summary = {}
    for track in ['时序趋势', '行业轮动']:
        selected = [r for r in complete_rows if r['轨道'] == track and r['单边成本(bp)'] == 10]
        reference = '超同暴露' if track == '时序趋势' else '超行业等权'
        delta_col = reference + '_年化对数差(%)'
        gates = {'ERA正': 0, '两窗均正': 0, '再通过60块下界及统一FDR': 0,
                 '再通过邻域多数同向及三成本不翻号': 0, '再通过20/120块敏感性': 0,
                 '再通过三块maxstat': 0}
        candidates = []
        for row in selected:
            if row[delta_col] <= 0:
                continue
            gates['ERA正'] += 1
            subwindows = windows.get(row['试验编号'], [])
            if len(subwindows) != 2 or not all(w[delta_col] > 0 for w in subwindows):
                continue
            gates['两窗均正'] += 1
            def passes(block):
                prefix = reference + f'_块{block}'
                return (row.get(prefix + '_年化对数差下界(%)', -np.inf) > 0
                        and row.get(prefix + '_全块长统一BH_FDR_q', 1) <= .05)
            if not passes(60):
                continue
            gates['再通过60块下界及统一FDR'] += 1
            cost_rows = costs_by_key[tuple(row[k] for k in key_fields)]
            if set(cost_rows) != {5, 10, 20} or not all(r[delta_col] > 0 for r in cost_rows.values()):
                continue
            axes = {'回看': NS} if track == '时序趋势' else {'回看': LOOKS, '跳过': SKIPS, '持仓数': TOPS}
            neighbors = []
            for axis, grid in axes.items():
                location = grid.index(row[axis])
                for offset in [-1, 1]:
                    if 0 <= location + offset < len(grid):
                        neighbor_key = tuple(grid[location + offset] if k == axis else row[k] for k in key_fields)
                        neighbor = costs_by_key.get(neighbor_key, {}).get(10)
                        if neighbor is not None:
                            neighbors.append(neighbor[delta_col] > 0)
            if not neighbors or sum(neighbors) <= len(neighbors) / 2:
                continue
            gates['再通过邻域多数同向及三成本不翻号'] += 1
            robust_blocks = all(passes(block) for block in [20, 120])
            gates['再通过20/120块敏感性'] += int(robust_blocks)
            maxstat = robust_blocks and all(row.get(reference + f'_块{block}_全族maxstat_p', 1) <= .05
                                           for block in [60, 20, 120])
            gates['再通过三块maxstat'] += int(maxstat)
            candidates.append({'试验编号': row['试验编号'], '资产代码': row['资产代码'], '规则': row['规则'],
                               '回看': row['回看'], '跳过': row['跳过'], '持仓数': row['持仓数'],
                               'ERA(%)': row[delta_col], '三块敏感性通过': robust_blocks, '三块maxstat通过': maxstat})
        by_cost = {}
        for cost in [5, 10, 20]:
            cost_rows = [r for r in complete_rows if r['轨道'] == track and r['单边成本(bp)'] == cost]
            by_cost[str(cost) + 'bp'] = {'格数': len(cost_rows), 'ERA正格数': sum(r[delta_col] > 0 for r in cost_rows),
                                       'ERA中位数(%)': float(np.median([r[delta_col] for r in cost_rows]))}
        complete_summary[track] = {'主成本(bp)': 10, '最长段主成本格数': len(selected), '逐级门槛计数': gates,
                                  '三成本并列': by_cost, '样本内稳定线索': candidates,
                                  '说明': '稳定线索依冻结判定：正ERA、两窗均正、60块下界>0与全块长统一FDR<=0.05、邻域严格多数正及5/10/20bp不翻号；额外列20/120块与maxstat，不称ETF或样本外验证'}
    final_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    assert final_hashes == input_hashes, '运行过程中源数据被更改'
    assert all(r['有效段末日'] <= '2020-12-31' for r in rows)
    assert all(c['日期'].max() <= '2020-12-31' for c in research.curves)
    assert all(e['执行开盘日'] <= '2020-12-31' and e['排名收盘日'] <= '2020-12-31' for e in research.events)
    evidence = {
        '脚本': str(Path(__file__).resolve()), '完成': True, '日期上限': '2020-12-31',
        '禁止读取评估2021及以后': True, '研究等级': freeze['证据等级'],
        '冻结设计SHA256': hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        '脚本SHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        '输入哈希': input_hashes, '运行前后输入哈希一致': final_hashes == input_hashes,
        '主成本(bp)': 10, '并列成本(bp)': [5, 10, 20],
        '成本跨任务说明': '本02主成本10bp；主agent其它子任务为5bp，不混同、不跨任务择优挑成本',
        '百分数口径': '所有收益/仓位/日换手/费率百分数列带(%)且数值乘100；CAGR差为百分点；净值、成本乘数和累计换手总量非百分数',
        '资产数': 35, '官方行业数': 28, '固定行业代码': symbols,
        '试验数': {'官方与全A时序参数成本格': 35 * 2 * 5 * 3,
                    '官方轮动参数成本格': 4 * 2 * 3 * 2 * 3,
                    '合成行业时序敏感性参数格': 28 * 2 * 5,
                    '合成行业轮动敏感性参数格': 4 * 2 * 3 * 2,
                    '完整行业合成时序参数成本格': 28 * 2 * 5 * 3,
                    '完整行业合成轮动参数成本格': 4 * 2 * 3 * 2 * 3,
                    '按数据组实际参数成本格': actual_grids,
                    '按数据组含有效分段实际报告格': {g: sum(r['数据组'] == g for r in rows) for g in group_names},
                    '全部参数格': sum(actual_grids.values()),
                    '含有效分段实际报告格': len(rows), '两个基准分开计算总假设': len(research.tests),
                    '主成本分窗描述行数不独立检验': len(research.windows),
                    '附加块长检验计数': sum(x['可估计假设数'] for x in reports[1:]),
                    '所有块长合计可估计检验次数': sum(x['可估计假设数'] for x in reports)},
        'bootstrap': reports, '输出行数': counts, '单元验证': checks,
        '假设与口径': [
            'MAIN_COST=10bp；5/10/20bp单边比例费并列、无印花税、现金收益0；指数与合成为非可投资基准，不是ETF；不与其它子任务5bp主成本混同。',
            '官方宽基/行业为价格指数：未含股息、ETF追踪误差、基金管理费、流动性与涨跌停可成交限制。',
            '全A与行业等权合成为非可投资基准；底层个股日再平衡成本未在输入合成价格计入，本脚本只计代理层费用。',
            '时序二元与分数权重统一现金+持股：night_factor=1+wprev*g；open_hold=wprev*(1+g)/night_factor；目标w为费后开盘权重；买入f=c*abs(w-open_hold)/(1+c*w)，卖出分母1-c*w；日因子=night_factor*(1-f)*(1+w*i)，收盘权重=w*(1+i)/(1+w*i)；首日旧仓0、末收费用c*收盘权重。',
            '同暴露以策略同段平均开盘权重每日重置，旧仓为实际收盘权重，先承隔夜并漂移到开盘；单边换手=abs(w*(1-f)-open_hold)，以费前开盘equity为分母，费用乘数1-f；与二元时序使用同一账本。',
            '恒定同暴露是事后风险暴露匹配的诊断基准，不是事前可用的交易预测；对数差与CAGR差不同列。',
            '轮动在月末真实收盘排名，close.shift(skip)/close.shift(lookback+skip)-1，次交易日开盘执行；月中不偷偷日再平衡。',
            '轮动旧收盘权重承隔夜并漂移至open_hold；new为费后目标，以f=c*sum(abs(new*(1-f)-open_hold))固定点迭代最多20次且误差<=1e-14，成本乘数1-f；换手为费前equity分母交易金额；不调仓日new=open_hold且无费，月中真实漂移；末收清仓费c*收盘总权重。',
            '轮动基准为同28可用行业集合每日再平衡等权，底层漂移换手、首开与末收费用一并计入。',
            '行业固定集合不动态剔除缺价行业；旧采掘SW801020在2020前仍有数据，7个共同缺失日不回填，2021新增行业不读。',
            '时序在市场日历上要求前日251个完整收盘以满足动量250和所有MA；轮动前月末及前日共同271收盘以满足250+skip20。',
            '缺价/未发布/共同长指标未成熟均为NA；只报告连续有效段，每段独立资金1、首开和末收计费，无跨缺口净值或CAGR。',
            '缺口分段末收清仓属于行政样本截断，不是能事前预知缺价的交易策略；各段不可拼接冒称2015—2020完整实盘。',
            '最长连续段只按数据覆盖长度标记而非按收益选取；所有短段与所有参数也报告和计数，不称独立验证。',
            '净年化为有效连续段日对数收益均值乘252后指数化；CAGR采用同段252交易日年制，MDD含初始净值1与末收费用。',
            '60日循环移动块999次；每块起点在0..length-1均匀抽样，索引(start+offset)%length，末块裁切至exact length，每日含端点边际期望权重均为1；中心化日对数差双侧p、95%块区间；同块长及60/20/120全块长BH-FDR均纳入全部数据组，不为每资产单独校正。',
            'max-stat为共享全市场交易日历循环抽样的studentized双侧最大值，按各有效段空缺mask计算重抽均值；本02时序与轮动全部组统一，包括matched与完整合成，不冒称跨其它子任务全局校正。',
            '20/120块各999次，对全部数据组10bp主成本统一整族检验；块长敏感性不是独立证据，不降低统计次数；不可将全网格看完后称留出验证。',
            '没有任何20日未来标签用于训练；本任务只有历史价格规则和月末横截面排名，不含学习器。',
            '行业合成敏感性保留matched官方有效日历；完整行业合成不传matched，以自身数据独立评估2015—2020时序与轮动；两组均要求成分至少10且有效价格率1、固定28行业，不用合成回填官方缺价，均为非投资合成基准且不是ETF。'],
        '覆盖证据': research.coverage, '完整行业合成稳定性': complete_summary,
        '主成本官方最长段描述': {'格数': len(main_rows), 'ERA正格数': positive,
                                      '60块FDR小于等于0.05格数不区分正负': discoveries,
                                      '说明': '仅描述网格分布，不挑最优、不宣称盈利已验证'},
        '运行秒数': round(time.time() - started, 2)}
    path = ROOT / '证据/02_时序与行业轮动核验.json'
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('完成', json.dumps(evidence['试验数'], ensure_ascii=False), '运行秒', evidence['运行秒数'], flush=True)


if __name__ == '__main__':
    main()
