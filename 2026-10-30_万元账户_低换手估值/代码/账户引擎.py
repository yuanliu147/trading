"""冻结研究的日线近似账户；仅消费调用者传入的数组，不读取行情文件。"""
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import hashlib
import math

import numpy as np
import pandas as pd


@dataclass
class Spec:
    id: str
    band: float = 0.2
    months: int = 3
    exit_kind: str = 'none'
    window: int = 0
    multiple: float = 0.0
    slots: int = 2
    score: str = 'both'
    lag: int = 0
    group: str = 'core'


@dataclass
class Assumptions:
    fee_mult: float = 1.0
    dividend_delay: int = 30
    bonus_lock: int = 20
    bonus_tax: float = 0.2
    cash_div_tax: float = 0.2
    stale_haircut: float = 0.0


def _cents(value):
    return int((Decimal(str(value)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def _positive(value):
    return np.isfinite(value) and value > 0


def _fees(symbol, date, qty, price, sell, multiplier):
    gross = Decimal(str(price)) * qty
    mult = Decimal(str(multiplier))
    commission = _cents(max(Decimal('5'), gross * Decimal('0.00025')) * mult)
    stamp = _cents(gross * Decimal('0.001') * mult) if sell else 0
    # 早期沪市费率是冻结的偏高假设，不声称精确还原历史分段费率。
    if date < '2015-08-01':
        shanghai = symbol.lower().startswith('sh') or symbol.lower().endswith('.sh') or symbol.startswith('6')
        transfer = max(Decimal('1'), Decimal(qty) * Decimal('0.001')) if shanghai else Decimal('0')
    else:
        transfer = gross * Decimal('0.00002')
    return commission, stamp, _cents(transfer * mult)


@dataclass
class _Position:
    i: int
    round_id: int
    buy_t: int
    buy_date: str
    bought: int
    buy_gross: int
    buy_fee: int
    qty: int
    available: int
    last_close: float
    last_quote_t: int
    missing_rows: int = 0
    locks: list = field(default_factory=list)
    pending: str = ''
    bonus: int = 0
    sold: int = 0
    sell_net: int = 0
    sell_fee: int = 0
    dividend: int = 0
    bonus_tax: int = 0
    rights: float = 0.0
    stop: float = float('nan')
    peak: float = float('nan')
    end_t: int = -1
    end_date: str = ''

    @property
    def cost(self):
        return self.buy_gross + self.buy_fee

    @property
    def pnl(self):
        return self.sell_net + self.dividend - self.cost - self.bonus_tax


class Engine:
    """O、C 等 N×T 数组为 data 顶层键；日期、证券使用 meta 的固定顺序。"""

    NAV_COLUMNS = ['日期', '净值', '现金', '持仓市值', '应收红利', '应付税', '持仓数', '仓位', '最大单票权重']
    LOG_COLUMNS = ['日期', '证券', '回合ID', '事件', '原因', '股数', '价格', '金额',
                   '费用', '佣金', '印花税', '过户费', '现金', '应付税', '预算', '信号日',
                   '可用日', '连续成交价', '止损连续价', '旧股数', '公式除权参考价', '原始除权参考价']
    ROUND_COLUMNS = ['证券', '回合ID', '买入日', '卖出日', '买入股数', '送转股数', '卖出股数',
                     '买入金额', '买入费用', '买入总支出', '净卖出金额', '净红利', '送转计提税',
                     '配股放弃股数', '费用合计', '盈亏', '收益率', '持有交易日']
    HOLD_COLUMNS = ['证券', '回合ID', '买入日', '股数', '可售股数', '锁定股数', '买入总支出',
                    '累计净卖出', '净红利', '送转计提税', '持仓市值', '累计盈亏', '最后报价',
                    '最后报价日', '报价缺日数', '连续缺行数', '待卖原因', '止损连续价',
                    '不可立即变现股数', '不可立即变现市值']

    def __init__(self, data: dict):
        self.symbols = list(data['meta']['证券'])
        self.dates = list(data['meta']['日期'])
        if self.symbols != sorted(set(self.symbols)):
            raise ValueError('证券必须唯一且按代码排序')
        if self.dates != sorted(set(self.dates)) or not self.dates:
            raise ValueError('日期必须唯一且升序')
        if self.dates[0] < '2010-01-01' or self.dates[-1] > '2020-12-31':
            raise ValueError('仅允许2010至2020日期')
        if '2010-12-31' not in self.dates:
            raise ValueError('必须包含共同净值基点2010-12-31')
        self.start = self.dates.index('2010-12-31')
        self.calendar = pd.DatetimeIndex(self.dates)
        self.a = {key: data[key] for key in (
            'O', 'C', 'PC', 'LUP', 'LDP', 'H', 'L', 'V', 'AMT', 'PE', 'PB', 'PCT',
            'AC', 'AO', 'ATR20', 'ADV20', 'VAVG20', 'ST', 'SUSP', 'RESUME', 'U',
            'CONFLICT', 'PRESENT')}
        shape = (len(self.symbols), len(self.dates))
        if any(array.shape != shape for array in self.a.values()):
            raise ValueError('行情数组形状必须为证券数×日期数')
        self.index = np.asarray(data['index'])
        if self.index.shape != (len(self.dates),):
            raise ValueError('指数必须与交易日历等长')
        self.actions = {}
        symbol_index = {symbol: i for i, symbol in enumerate(self.symbols)}
        columns = ['证券', '除权日', '每股分红', '送转比例', '配股比例', '配股价']
        for symbol, date, div, bonus, rights, rights_price in data['actions'][columns].itertuples(index=False, name=None):
            if symbol in symbol_index:
                values = tuple(0.0 if pd.isna(v) else float(v) for v in (div, bonus, rights, rights_price))
                if any(v < 0 or not np.isfinite(v) for v in values):
                    raise ValueError('公司行动数值必须有限且非负')
                self.actions.setdefault(str(date)[:10], []).append((symbol_index[symbol], *values))
        self.pool_cache = {}
        self.hash_cache = {}
        self.ma_cache = {}

    def _pool(self, spec, t):
        key = (spec.score, spec.band, spec.lag, t)
        if key in self.pool_cache:
            return self.pool_cache[key]
        mask = np.array(self.a['U'][:, t], dtype=bool, copy=True)
        value_t = t - spec.lag
        if value_t < 0:
            mask[:] = False
        elif spec.lag > 0 or spec.score != 'control':
            pe, pb = self.a['PE'][:, value_t], self.a['PB'][:, value_t]
            mask &= np.isfinite(pe) & (pe > 0) & np.isfinite(pb) & (pb > 0)
        indices = np.flatnonzero(mask)
        if spec.score != 'control' and len(indices):
            ep = pd.Series(1.0 / self.a['PE'][indices, value_t].astype(float)).rank(pct=True).to_numpy()
            bp = pd.Series(1.0 / self.a['PB'][indices, value_t].astype(float)).rank(pct=True).to_numpy()
            score = (ep + bp) / 2 if spec.score == 'both' else (ep if spec.score == 'ep' else bp)
            indices = indices[np.lexsort((indices, -score))[:math.ceil(len(indices) * spec.band)]]
        pool = frozenset(int(i) for i in indices)
        self.pool_cache[key] = pool
        return pool

    def _market_open(self, spec, t):
        if spec.exit_kind != 'market':
            return True
        if spec.window not in self.ma_cache:
            # rolling 只包含当日及历史，预先计算不改变任何历史可见集。
            self.ma_cache[spec.window] = pd.Series(self.index).rolling(spec.window, min_periods=spec.window).mean().to_numpy()
        ma = self.ma_cache[spec.window][t]
        return bool(np.isfinite(ma) and np.isfinite(self.index[t]) and self.index[t] > ma)

    def _review(self, t, months):
        if t == self.start:
            return True
        day = self.calendar[t]
        return day.month % months == 0 and (t + 1 == len(self.dates) or self.calendar[t + 1].month != day.month)

    def _price(self, i, t, qty, sell, assumptions):
        opening = float(self.a['O'][i, t])
        avg = float(self.a['VAVG20'][i, t - 1])
        slip = (0.001 + 0.001 * math.sqrt(qty / avg)) * assumptions.fee_mult
        price = (Decimal(str(opening)) * (Decimal('1') + Decimal(str(-slip if sell else slip)))).quantize(
            Decimal('0.01'), rounding=ROUND_FLOOR if sell else ROUND_CEILING)
        lower, upper = self.a['LDP'][i, t], self.a['LUP'][i, t]
        if _positive(lower):
            price = max(price, Decimal(str(float(lower))))
        if _positive(upper):
            price = min(price, Decimal(str(float(upper))))
        return float(price)

    def _blocked(self, i, t, sell):
        a = self.a
        if not a['PRESENT'][i, t] or a['SUSP'][i, t]:
            return '停牌或缺行'
        if not _positive(a['O'][i, t]) or not _positive(a['V'][i, t]):
            return '无有效开盘或成交量'
        if sell and _positive(a['LDP'][i, t]) and a['O'][i, t] <= a['LDP'][i, t]:
            return '开盘跌停'
        if not sell and _positive(a['LUP'][i, t]) and a['O'][i, t] >= a['LUP'][i, t]:
            return '开盘涨停'
        if t == 0 or not _positive(a['VAVG20'][i, t - 1]):
            return '前日均量缺窗'
        return ''

    @staticmethod
    def _sell_qty(available, capacity):
        if available <= capacity:
            return available
        return (capacity // 100) * 100

    def run(self, spec: Spec, seed: int, assumptions: Assumptions | None = None, detail: bool = False) -> dict:
        assumptions = assumptions if assumptions is not None else Assumptions()
        if spec.exit_kind not in ('none', 'market', 'trail') or spec.score not in ('both', 'ep', 'bp', 'control'):
            raise ValueError('未知exit_kind或score')
        if spec.months not in (3, 6) or spec.slots < 1 or int(spec.slots) != spec.slots or spec.lag < 0 or int(spec.lag) != spec.lag:
            raise ValueError('审视为3/6个月，槽位为正整数，lag为非负整数')
        if spec.score != 'control' and not 0 < spec.band <= 1:
            raise ValueError('band须在(0,1]')
        if spec.exit_kind == 'market' and (spec.window < 1 or int(spec.window) != spec.window):
            raise ValueError('市场均线必须指定正整数满窗')
        if spec.exit_kind == 'trail' and (not np.isfinite(spec.multiple) or spec.multiple <= 0):
            raise ValueError('跟踪退出必须指定正倍数')
        if not np.isfinite(assumptions.fee_mult) or assumptions.fee_mult < 0:
            raise ValueError('fee_mult必须有限且非负')
        for delay in (assumptions.dividend_delay, assumptions.bonus_lock):
            if delay < 0 or int(delay) != delay:
                raise ValueError('到账和锁定延迟必须为非负整数')
        if any(not 0 <= x <= 1 for x in (assumptions.bonus_tax, assumptions.cash_div_tax, assumptions.stale_haircut)):
            raise ValueError('税率和折价须在[0,1]')
        if seed not in self.hash_cache:
            self.hash_cache[seed] = [hashlib.sha256(f'{seed}|{symbol}'.encode('utf-8')).digest() for symbol in self.symbols]
        hashes = self.hash_cache[seed]
        cash = 1000000
        tax_due = 0
        receivables = []  # (自然日到期日, 回合对象, 净额分)
        positions = {}
        ledger = []  # detail=False 也保留每次开仓的完整经济账本。
        orders = []
        previous_nav = cash
        nav_rows, logs = [], []
        total = dict(buy_gross=0, buy_cost=0, sell_net=0, fee=0, dividend=0, received=0,
                     bonus_tax=0, tax_paid=0, rights=0.0, buys=0, sells=0, failed=0, delayed=0,
                     reference_mismatches=0)
        checks = {key: 0 for key in (
            '现金流残差', '净值恒等式残差', '应收红利残差', '应付税残差', '股数残差', '可售锁定股数残差',
            '负现金次数', '负税负次数', '超槽次数', '重复开仓次数', '非整手买入次数',
            '买入预算违规次数', 'T+1违规次数', '提前解锁卖出次数', '成交容量违规次数',
            '当日量不足成交次数', '非审视买入次数')}

        def log(t, event, p=None, i=None, **fields):
            if detail:
                idx = p.i if p is not None else i
                logs.append({'日期': self.dates[t], '证券': self.symbols[idx] if idx is not None else '',
                             '回合ID': p.round_id if p is not None else None, '事件': event,
                             '现金': cash / 100, '应付税': tax_due / 100, **fields})

        def pay_tax(t):
            nonlocal cash, tax_due
            paid = min(cash, tax_due)
            if paid:
                cash -= paid
                tax_due -= paid
                total['tax_paid'] += paid
                log(t, '支付送转税', 金额=paid / 100)

        def value(p):
            haircut = assumptions.stale_haircut if p.missing_rows >= 20 else 0.0
            return _cents(Decimal(str(p.last_close)) * p.qty * (Decimal('1') - Decimal(str(haircut))))

        def execute_check(p, t, qty, capacity, selling):
            checks['成交容量违规次数'] += int(qty > capacity)
            checks['当日量不足成交次数'] += int(qty > self.a['V'][p.i, t])
            if selling:
                checks['T+1违规次数'] += int(t <= p.buy_t)
                checks['提前解锁卖出次数'] += int(qty > p.available)
            else:
                checks['非整手买入次数'] += int(qty % 100 != 0)
                checks['非审视买入次数'] += int(not self._review(t - 1, spec.months))

        for t in range(self.start, len(self.dates)):
            date, day = self.dates[t], self.calendar[t]
            # 1. 先到账/解锁；无持仓的已完成回合仍可收到红利。
            remaining_receivables = []
            for due, p, amount in receivables:
                if due <= day:
                    cash += amount
                    total['received'] += amount
                    log(t, '红利到账', p, 金额=amount / 100, 可用日=due.strftime('%Y-%m-%d'))
                else:
                    remaining_receivables.append((due, p, amount))
            receivables = remaining_receivables
            for p in positions.values():
                locked = []
                for release, qty in p.locks:
                    if release <= t:
                        p.available += qty
                        log(t, '股份解锁', p, 股数=qty)
                    else:
                        locked.append((release, qty))
                p.locks = locked
            pay_tax(t)

            # 2. 当日所有行动都用行动前旧总股数，含尚锁定股份。
            old_quantities = {i: p.qty for i, p in positions.items()}
            reference_events = {}
            for i, div, bonus, rights, rights_price in self.actions.get(date, ()):
                if i not in positions:
                    continue
                p, old = positions[i], old_quantities[i]
                if p.buy_t >= t:
                    continue
                net_div = _cents(Decimal(old) * Decimal(str(div)) * (Decimal('1') - Decimal(str(assumptions.cash_div_tax))))
                new_qty = int((Decimal(old) * Decimal(str(bonus))).to_integral_value(rounding=ROUND_FLOOR))
                new_tax = _cents(Decimal(new_qty) * Decimal(str(assumptions.bonus_tax)))
                abandoned = float(Decimal(old) * Decimal(str(rights)))
                if net_div:
                    p.dividend += net_div
                    total['dividend'] += net_div
                    due = day + pd.Timedelta(days=assumptions.dividend_delay)
                    log(t, '红利计提', p, 旧股数=old, 金额=net_div / 100, 可用日=due.strftime('%Y-%m-%d'))
                    if due <= day:
                        cash += net_div
                        total['received'] += net_div
                        log(t, '红利到账', p, 金额=net_div / 100, 可用日=due.strftime('%Y-%m-%d'))
                    else:
                        receivables.append((due, p, net_div))
                if new_qty:
                    p.qty += new_qty
                    p.bonus += new_qty
                    p.bonus_tax += new_tax
                    tax_due += new_tax
                    total['bonus_tax'] += new_tax
                    release = t + assumptions.bonus_lock
                    if release <= t:
                        p.available += new_qty
                    else:
                        p.locks.append((release, new_qty))
                    release_date = self.dates[release] if release < len(self.dates) else f'交易日索引{release}(窗外)'
                    log(t, '送转计提', p, 旧股数=old, 股数=new_qty, 金额=new_tax / 100, 可用日=release_date)
                if abandoned:
                    p.rights += abandoned
                    total['rights'] += abandoned
                    log(t, '放弃配股', p, 旧股数=old, 股数=abandoned, 价格=rights_price)
                parts = reference_events.setdefault(i, [0.0, 0.0, 0.0, 0.0])
                for k, v in enumerate((div, bonus, rights, rights * rights_price)):
                    parts[k] += v
            for i, (div, bonus, rights, subscription) in reference_events.items():
                p = positions[i]
                reference = (p.last_close - div + subscription) / (1 + bonus + rights)
                observed = float(self.a['PC'][i, t])
                mismatch = _positive(observed) and abs(observed - reference) > 0.011
                total['reference_mismatches'] += int(mismatch)
                log(t, '公司行动参考价核对', p, 公式除权参考价=reference, 原始除权参考价=observed,
                    原因='不一致（不删除事件）' if mismatch else '一致或原始参考价缺失')
            pay_tax(t)

            # 3. 待卖不可撤单；先卖可售股份，余股继续占槽。
            for i, p in list(positions.items()):
                if not p.pending:
                    continue
                reason = self._blocked(i, t, True)
                capacity = int(math.floor(float(self.a['VAVG20'][i, t - 1]) * 0.01)) if not reason else 0
                qty = self._sell_qty(p.available, capacity) if not reason else 0
                if not reason and qty == 0:
                    reason = '股份锁定或容量不足'
                if not reason and self.a['V'][i, t] < qty:
                    reason = '当日量不足订单'
                if not reason:
                    price = self._price(i, t, qty, True, assumptions)
                    fee_parts = _fees(self.symbols[i], date, qty, price, True, assumptions.fee_mult)
                    fee = sum(fee_parts)
                    proceeds = _cents(Decimal(str(price)) * qty) - fee
                    if price <= 0 or cash + proceeds < 0:
                        reason = '卖出净额不足支付费用'
                    else:
                        execute_check(p, t, qty, capacity, True)
                        cash += proceeds
                        p.qty -= qty
                        p.available -= qty
                        p.sold += qty
                        p.sell_net += proceeds
                        p.sell_fee += fee
                        total['sell_net'] += proceeds
                        total['fee'] += fee
                        total['sells'] += 1
                        log(t, '卖出', p, 原因=p.pending, 股数=qty, 价格=price, 金额=proceeds / 100,
                            费用=fee / 100, 佣金=fee_parts[0] / 100, 印花税=fee_parts[1] / 100, 过户费=fee_parts[2] / 100)
                        pay_tax(t)
                        if p.qty == 0:
                            p.end_t, p.end_date = t, date
                            del positions[i]
                if p.qty:
                    total['delayed'] += 1
                    log(t, '卖出延期', p, 原因=reason or '部分卖出后仍有余股', 股数=p.qty)

            # 4. 昨收固定候选一次尝试，不用今天的成交量反推缩小订单，不替补。
            for n, i in enumerate(orders):
                reason = ''
                budget = min(previous_nav // spec.slots, max(0, cash - tax_due) // (len(orders) - n))
                if len(positions) >= spec.slots:
                    reason = '待卖未清导致无空槽'
                elif i in positions:
                    reason = '已有持仓禁止第二批'
                else:
                    reason = self._blocked(i, t, False)
                if not reason and spec.exit_kind == 'trail':
                    atr = self.a['ATR20'][i, t - 1]
                    # 连续价换算只用开盘前已知的昨日连续收盘与今日除权参考价。
                    if (not np.isfinite(atr) or atr < 0 or not _positive(self.a['AC'][i, t - 1])
                            or not _positive(self.a['PC'][i, t])):
                        reason = 'ATR缺窗或连续开盘无效'
                qty, price, fee, fee_parts = 0, 0.0, 0, (0, 0, 0)
                if not reason:
                    capacity = int(math.floor(float(self.a['VAVG20'][i, t - 1]) * 0.01))
                    # 支出随手数单调；二分寻找同时包含费用的最大合法整手。
                    lo, hi = 0, capacity // 100
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        trial_qty = mid * 100
                        trial_price = self._price(i, t, trial_qty, False, assumptions)
                        trial_fee = sum(_fees(self.symbols[i], date, trial_qty, trial_price, False, assumptions.fee_mult))
                        if _cents(Decimal(str(trial_price)) * trial_qty) + trial_fee <= budget:
                            lo = mid
                        else:
                            hi = mid - 1
                    qty = lo * 100
                    if qty == 0:
                        reason = '预算或容量不足100股'
                    elif self.a['V'][i, t] < qty:
                        reason = '当日量不足订单'
                    else:
                        price = self._price(i, t, qty, False, assumptions)
                        fee_parts = _fees(self.symbols[i], date, qty, price, False, assumptions.fee_mult)
                        fee = sum(fee_parts)
                if not reason:
                    valid = np.flatnonzero(self.a['PRESENT'][i, :t] & np.isfinite(self.a['C'][i, :t]) & (self.a['C'][i, :t] > 0))
                    if not len(valid):
                        reason = '无历史有效原始收盘'
                if reason:
                    total['failed'] += 1
                    log(t, '买入失败', i=i, 原因=reason, 股数=qty, 预算=budget / 100, 信号日=self.dates[t - 1])
                    continue
                gross = _cents(Decimal(str(price)) * qty)
                checks['买入预算违规次数'] += int(gross + fee > budget or gross + fee > cash - tax_due)
                checks['重复开仓次数'] += int(i in positions)
                cash -= gross + fee
                last_quote = int(valid[-1])
                missing_rows = 0
                for k in range(t - 1, -1, -1):
                    if self.a['PRESENT'][i, k]:
                        break
                    missing_rows += 1
                p = _Position(i, len(ledger) + 1, t, date, qty, gross, fee, qty, 0,
                              float(self.a['C'][i, last_quote]), last_quote, missing_rows)
                p.locks.append((t + 1, qty))  # 普通买入股份也显式执行T+1可售。
                continuous_entry = float('nan')
                if spec.exit_kind == 'trail':
                    continuous_entry = price * float(self.a['AC'][i, t - 1]) / float(self.a['PC'][i, t])
                    p.stop = continuous_entry - spec.multiple * float(self.a['ATR20'][i, t - 1])
                execute_check(p, t, qty, capacity, False)
                positions[i] = p
                ledger.append(p)
                total['buy_gross'] += gross
                total['buy_cost'] += gross + fee
                total['fee'] += fee
                total['buys'] += 1
                log(t, '买入', p, 股数=qty, 价格=price, 金额=gross / 100, 费用=fee / 100,
                    佣金=fee_parts[0] / 100, 印花税=fee_parts[1] / 100, 过户费=fee_parts[2] / 100,
                    预算=budget / 100, 信号日=self.dates[t - 1], 连续成交价=continuous_entry, 止损连续价=p.stop)
            orders = []

            # 5. 原始最后报价MTM，缺行20日才应用情景折价；不使用未来退市信息。
            for i, p in positions.items():
                present = bool(self.a['PRESENT'][i, t])
                p.missing_rows = 0 if present else p.missing_rows + 1
                if present and _positive(self.a['C'][i, t]):
                    p.last_close, p.last_quote_t = float(self.a['C'][i, t]), t
            market_values = {i: value(p) for i, p in positions.items()}
            market_value = sum(market_values.values())
            receivable = sum(amount for _, _, amount in receivables)
            nav = cash + market_value + receivable - tax_due
            weight = max(market_values.values(), default=0) / nav if nav > 0 else float('nan')
            exposure = market_value / nav if nav > 0 else float('nan')
            nav_rows.append([date, nav / 100, cash / 100, market_value / 100, receivable / 100,
                             tax_due / 100, len(positions), exposure, weight])
            expected_cash = 1000000 + total['sell_net'] + total['received'] - total['buy_cost'] - total['tax_paid']
            expected_nav = 1000000 + total['sell_net'] + total['dividend'] - total['buy_cost'] - total['bonus_tax'] + market_value
            for key, residual in (
                ('现金流残差', cash - expected_cash), ('净值恒等式残差', nav - expected_nav),
                ('应收红利残差', receivable - (total['dividend'] - total['received'])),
                ('应付税残差', tax_due - (total['bonus_tax'] - total['tax_paid']))):
                checks[key] = max(checks[key], abs(residual) / 100)
            checks['负现金次数'] += int(cash < 0)
            checks['负税负次数'] += int(tax_due < 0)
            checks['超槽次数'] += int(len(positions) > spec.slots)
            for p in positions.values():
                checks['股数残差'] = max(checks['股数残差'], abs(p.qty - p.bought - p.bonus + p.sold))
                checks['可售锁定股数残差'] = max(checks['可售锁定股数残差'], abs(p.qty - p.available - sum(q for _, q in p.locks)))

            market_open = self._market_open(spec, t)
            review = self._review(t, spec.months)
            pool = self._pool(spec, t) if review else frozenset()
            for i, p in positions.items():
                trigger = ''
                # 缺行的ST是数据准备的防买入哨兵；只有当日有源行且标记ST才算已观察到风险警示。
                if self.a['ST'][i, t] and self.a['PRESENT'][i, t]:
                    trigger = 'ST'
                elif not market_open:
                    trigger = 'market'
                if spec.exit_kind == 'trail':
                    close, atr = float(self.a['AC'][i, t]), float(self.a['ATR20'][i, t])
                    if _positive(close):
                        p.peak = max(p.peak, close) if np.isfinite(p.peak) else close
                    if np.isfinite(p.peak) and np.isfinite(atr) and atr >= 0:
                        p.stop = max(p.stop, p.peak - spec.multiple * atr)
                    if not trigger and _positive(close) and close < p.stop:
                        trigger = 'trail'
                if not trigger and review and i not in pool:
                    trigger = 'valuation'
                if trigger and not p.pending:
                    p.pending = trigger
                    log(t, '卖出信号', p, 原因=trigger, 股数=p.qty, 止损连续价=p.stop)
                elif trigger and p.pending:
                    priority = {'ST': 0, 'market': 1, 'trail': 2, 'valuation': 3}
                    if priority[trigger] < priority[p.pending]:
                        p.pending = trigger
                        log(t, '退出原因升级', p, 原因=trigger, 股数=p.qty)
            if review and market_open:
                retained = sum(not p.pending for p in positions.values())
                candidates = [i for i in pool if i not in positions and not any(
                    self.a[flag][i, t] for flag in ('ST', 'SUSP', 'RESUME', 'CONFLICT'))]
                candidates.sort(key=lambda i: (hashes[i], self.symbols[i]))
                orders = candidates[:max(0, spec.slots - retained)]
                for i in orders:
                    log(t, '买入信号', i=i, 信号日=date)
            previous_nav = nav

        nav_frame = pd.DataFrame(nav_rows, columns=self.NAV_COLUMNS)
        completed = [p for p in ledger if p.end_t >= 0]
        round_rows = []
        for p in completed:
            checks['股数残差'] = max(checks['股数残差'], abs(p.bought + p.bonus - p.sold))
            round_rows.append([self.symbols[p.i], p.round_id, p.buy_date, p.end_date, p.bought, p.bonus, p.sold,
                               p.buy_gross / 100, p.buy_fee / 100, p.cost / 100, p.sell_net / 100,
                               p.dividend / 100, p.bonus_tax / 100, p.rights, (p.buy_fee + p.sell_fee) / 100,
                               p.pnl / 100, p.pnl / p.cost, p.end_t - p.buy_t])
        final_t = len(self.dates) - 1
        hold_rows = []
        illiquid_value = 0
        for i, p in sorted(positions.items()):
            # 期末可变现性仅用截至终日的状态/收盘价，不虚构下一开盘清仓。
            blocked = (p.buy_t == final_t or not self.a['PRESENT'][i, final_t] or self.a['SUSP'][i, final_t]
                       or not _positive(self.a['C'][i, final_t]) or not _positive(self.a['V'][i, final_t])
                       or (_positive(self.a['LDP'][i, final_t]) and self.a['C'][i, final_t] <= self.a['LDP'][i, final_t]))
            avg = float(self.a['VAVG20'][i, final_t - 1]) if final_t > 0 else float('nan')
            capacity = int(math.floor(avg * 0.01)) if _positive(avg) else 0
            liquid_qty = 0 if blocked else self._sell_qty(p.available, capacity)
            if liquid_qty > self.a['V'][i, final_t]:
                liquid_qty = 0
            illiquid_qty = p.qty - liquid_qty
            mv = value(p)
            illiquid = _cents(Decimal(mv) / 100 * illiquid_qty / p.qty)
            illiquid_value += illiquid
            hold_rows.append([self.symbols[i], p.round_id, p.buy_date, p.qty, p.available,
                              sum(q for _, q in p.locks), p.cost / 100, p.sell_net / 100,
                              p.dividend / 100, p.bonus_tax / 100, mv / 100, (p.pnl + mv) / 100,
                              p.last_close, self.dates[p.last_quote_t], final_t - p.last_quote_t,
                              p.missing_rows, p.pending, p.stop, illiquid_qty, illiquid / 100])
        # 与逐回合账本再交叉核对（包括已售完但红利未到账的回合）。
        checks['回合账本净值残差'] = abs(previous_nav - (1000000 + sum(p.pnl for p in ledger) + sum(value(p) for p in positions.values()))) / 100
        checks['回合账本费用残差'] = abs(total['fee'] - sum(p.buy_fee + p.sell_fee for p in ledger)) / 100

        def cagr(start_t, end_t):
            days = (self.calendar[end_t] - self.calendar[start_t]).days
            first = float(nav_frame.iloc[start_t - self.start]['净值'])
            last = float(nav_frame.iloc[end_t - self.start]['净值'])
            return (last / first) ** (365.25 / days) - 1 if days > 0 and first > 0 and last >= 0 else float('nan')

        phase1 = [t for t in range(self.start, len(self.dates)) if self.dates[t] <= '2015-12-31']
        boundary = phase1[-1]
        phase2_cagr = cagr(boundary, final_t) if self.dates[boundary].startswith('2015-') and self.dates[final_t] >= '2016-01-01' else float('nan')
        pnls = [p.pnl / 100 for p in completed]
        wins, losses = [x for x in pnls if x > 0], [x for x in pnls if x < 0]
        ratio = float(np.mean(wins) / -np.mean(losses)) if wins and losses else (float('inf') if wins else float('nan'))
        daily = nav_frame.iloc[1:] if len(nav_frame) > 1 else nav_frame
        summary = {
            'CAGR': cagr(self.start, final_t),
            '最大回撤': float((nav_frame['净值'] / nav_frame['净值'].cummax() - 1).min()),
            '期末净值': previous_nav / 100, '总收益': previous_nav / 1000000 - 1,
            '2011-2015_CAGR': cagr(self.start, boundary), '2016-2020_CAGR': phase2_cagr,
            '平均仓位': float(daily['仓位'].mean()), '平均现金比例': float((daily['现金'] / daily['净值']).mean()),
            '买入笔数': total['buys'], '卖出笔数': total['sells'], '完整回合数': len(completed),
            '胜率': len(wins) / len(completed) if completed else float('nan'), '盈亏比': ratio,
            '费用合计': total['fee'] / 100, '分红净额': total['dividend'] / 100,
            '送转税计提': total['bonus_tax'] / 100, '配股放弃股数': total['rights'],
            '买入金额': total['buy_gross'] / 100, '最大单票权重': float(nav_frame['最大单票权重'].max()),
            '期末持仓数': len(positions), '期末不可立即变现市值': illiquid_value / 100,
            '买入失败数': total['failed'], '卖出延期天次': total['delayed'],
            '公司行动参考价不一致天次': total['reference_mismatches'],
        }
        result = {'净值': nav_frame, '汇总': summary, '自检': checks}
        if detail:
            result.update({'日志': pd.DataFrame(logs, columns=self.LOG_COLUMNS),
                           '回合': pd.DataFrame(round_rows, columns=self.ROUND_COLUMNS),
                           '期末持仓': pd.DataFrame(hold_rows, columns=self.HOLD_COLUMNS)})
        return result
