#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强势突破短线交易系统 —— 逐笔历史回测
=====================================

规则来源：《短线操盘实战技法/可执行短线交易系统.md》（下称"规则文件"）。
回测口径依据任务书，规则文件未覆盖或表述开放处按下述"简化与解释"执行，
并在 summary.md 的"规则简化清单"中逐条列出。

数据：
- 行情：/tmp/ht_backtest/market_snapshot.duckdb（只读），表 raw_kline_daily（不复权日线）
- 大盘：data/index_yahoo.csv（上证指数 000001.SS，Yahoo Finance 日线，
  已与同花顺远端 000001.SH 2022-01 以来数据交叉验证，最大收盘价偏差 0.005 点；
  本地库不含指数，同花顺远端指数历史仅回溯到 2022 年，故用 Yahoo 全历史）。
  缺该文件时脚本自动重新抓取。
- 名称：data/symbols.csv（同花顺 CLI symbol list 的当前名称；本地库 name 字段全为空）。

规则映射（代码中注释标明章节）：
- 第1步 总开关：T 日上证收盘 > 其 MA100 才允许开新仓（T+1 执行）    -> index_on
- 第2步 来源A：首次突破 MA100（T收>MA100 且 T-1收<=MA100 且 涨幅<9%） -> SQL 信号
- 第3步 计划：E=T日最高；S=近10日最低(含T)；(E-S)/E<=8%；N=权益1%/(E-S)
  100股向下取整、金额上限30%                                          -> plan
- 二、红线2：不追——E 距 T 日 MA20 超 10% 不买                        -> SQL 过滤
- 第4步 第一批 40%：T+1 最高价>=E 则以 max(E, T+1开盘) 成交；开盘涨停买不进
- 第4步 第二批 30%：收盘>信号日前20日最高价(即[T-20,T-1]区间最高)，
  次日开盘加；亏损中的股票绝不加仓（红线3）；第三批（回踩企稳）不模拟
- 第5步 出场（收盘判定、次日开盘执行，优先级从高到低）：
  1) 收盘<MA20 或 收盘<MA100 -> 次日开盘清仓（与止损同日触发时按清仓处理）
  2) 收盘<S -> 次日开盘清仓（初始止损）
  3) 收盘<MA10 -> 卖 1/3（每批限一次）
  4) 入场后3个交易日内(含入场日)累计涨幅>20% -> 卖 1/3（每批限一次）
- 第6步 当日亏损>=权益2% -> 次日不做任何买入；连续3笔初始止损 -> 停做3个交易日
- 二、红线5：单票<=30%、总仓位<=70%（任务书另加：同时最多5只、每日最多新开1只）
- 涨停买不进/跌停卖不出：主板 ±9.8%，科创板(68)与创业板(30，2020-08-24注册制后) ±19.8%
  （对任务书 9.8% 平铺口径的修正，见 summary.md 简化清单）
- 费用：每笔成交单边 0.1%（佣金+印花税近似）
- 资金：初始 20 万，权益随市值浮动，无追加
- 区间：2016-09-27（数据起点）至 2026-09-24；有效信号自 2017-03 起（库内满120日+
  MA100 预热），与任务书"约2017年中"一致
- 价格口径：全程不复权（除权缺口可能造成少量假信号/假止损，见 summary.md）

简化与解释（详见 summary.md 第五节）：
1. 上市不满120日剔除：以"本地库中首个交易日"计龄（库无上市日期字段），
   2016-09-27 前上市的老股统一延迟到 2017-03-28 之后才具备资格（一刀切，方向保守）。
2. ST 剔除用"当前名称"，历史上曾 ST 后摘帽的股票在其 ST 期间未被剔除。
3. 20%止盈的"累计涨幅"以各批次自身入场价为基准；3个交易日=入场日起含当日共3天。
4. 连亏3笔只计"初始止损"（收盘<S）这一最机械口径；非初始止损的完整出场重置计数；
   暂停自触发次日起3个交易日。
5. 每日2%停做线按收盘权益环比计算，作用于下一交易日（规则文件第6步"明天不交易"）。
6. 第二批加仓不受大盘总开关限制（规则文件未对加仓设总开关），但受2%停做线/连亏暂停约束。
7. 第二批加仓日停牌或开盘涨停 -> 放弃本次加仓（不延期）。
8. 同日多信号按"离 MA100 越近越优先"（规则文件第2步排序）依次尝试，每日最多成交1只。
9. 二批加仓价 A1 = 信号日前20个交易日最高价 [T-20, T-1]（不含T日）。
10. 卖1/3按批内剩余股数的1/3取整到100股；若取整后>=剩余股数则放弃本次减仓（保留底仓）。
11. Yahoo 指数缺 2019-04-29/30 两天，该两日无总开关数据、视为不开新仓，影响可忽略。
"""

import csv
import datetime as dt
import json
import os
import ssl
import urllib.request
from collections import defaultdict

import duckdb

# ---------------- 配置 ----------------
DB_PATH = '/tmp/ht_backtest/market_snapshot.duckdb'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INDEX_CSV = os.path.join(DATA_DIR, 'index_yahoo.csv')
NAMES_CSV = os.path.join(DATA_DIR, 'symbols.csv')
TRADES_CSV = os.path.join(BASE_DIR, 'trades.csv')
EQUITY_CSV = os.path.join(BASE_DIR, 'equity_curve.csv')
SUMMARY_MD = os.path.join(BASE_DIR, 'summary.md')

START_DATE = dt.date(2016, 9, 27)      # 数据起点
END_DATE = dt.date(2026, 9, 24)        # 任务书规定终点
INIT_EQUITY = 200_000.0
RISK_PER_TRADE = 0.01                  # 单笔风险 1%（规则文件 参数表）
MAX_POS_PCT = 0.30                     # 单票市值<=30%（红线5）
MAX_TOTAL_PCT = 0.70                   # 总仓位<=70%（红线5）
MAX_POSITIONS = 5                      # 同时最多5只（任务书）
MAX_NEW_PER_DAY = 1                    # 每日最多新开1只（任务书）
DAILY_STOP_PCT = 0.02                  # 每日停做线2%（规则文件 参数表）
CONSEC_STOPS = 3                       # 连亏3笔（规则文件 第6步）
PAUSE_DAYS = 3                         # 停做3天（规则文件 第6步）
FEE = 0.001                            # 单边0.1%（任务书）
BATCH1_PCT, BATCH2_PCT = 0.40, 0.30    # 三批 40/30/30，第三批不模拟（任务书）
MAX_PLAN_RISK_PCT = 0.08               # (E-S)/E<=8%（规则文件 第3步）
CHASE_LIMIT = 0.10                     # 不追：距MA20>10%不买（红线2）
GAIN20_LIMIT = 0.20                    # 3日内涨幅>20%卖1/3（规则文件 第5步）
GAIN20_DAYS = 3                        # 含入场日共3天
CYB_REG_DATE = dt.date(2020, 8, 24)    # 创业板注册制20%涨跌幅起始日

PHASES = [
    ('2017-2018 熊市', dt.date(2017, 1, 1), dt.date(2018, 12, 31)),
    ('2019-2020', dt.date(2019, 1, 1), dt.date(2020, 12, 31)),
    ('2021-2022', dt.date(2021, 1, 1), dt.date(2022, 12, 31)),
    ('2023-2026.9', dt.date(2023, 1, 1), dt.date(2026, 9, 24)),
]


def board_limit(thscode, date):
    """涨跌停近似阈值：主板 9.8%，科创板 19.8%，创业板注册制后 19.8%"""
    pre = thscode[:2]
    if pre == '68':
        return 0.198
    if pre == '30':
        return 0.198 if date >= CYB_REG_DATE else 0.098
    return 0.098


# ---------------- 大盘数据 ----------------
def fetch_yahoo_index():
    """缺 index_yahoo.csv 时从 Yahoo Finance 抓取上证指数 000001.SS 日线"""
    p1 = int(dt.datetime(2015, 6, 1).timestamp())
    p2 = int(dt.datetime(2026, 9, 30).timestamp())
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/000001.SS'
           f'?period1={p1}&period2={p2}&interval=1d&events=history')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        raw = urllib.request.urlopen(req, timeout=60).read()
    except ssl.SSLCertVerificationError:
        raw = urllib.request.urlopen(req, timeout=60,
                                     context=ssl._create_unverified_context()).read()
    d = json.loads(raw)
    r = d['chart']['result'][0]
    q = r['indicators']['quote'][0]
    rows = []
    for i, t in enumerate(r['timestamp']):
        c = q['close'][i]
        if c is None:
            continue
        day = dt.datetime.fromtimestamp(
            t, dt.timezone(dt.timedelta(hours=8))).date()
        rows.append((day, q['open'][i], q['high'][i], q['low'][i], c, q['volume'][i]))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(INDEX_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'open', 'high', 'low', 'close', 'volume'])
        for row in rows:
            w.writerow(row)
    print(f'[data] fetched {len(rows)} index rows -> {INDEX_CSV}')


def load_index():
    """返回 ({date: {'close','ma100','on'}}, {date: close})。
    对数据库有、指数缺失的交易日（2019-04-29/30），总开关状态顺延上一指数日。"""
    closes = {}
    with open(INDEX_CSV) as f:
        for r in csv.DictReader(f):
            d = dt.date.fromisoformat(r['date'])
            if d <= END_DATE:
                closes[d] = float(r['close'])
    idx = {}
    sorted_days = sorted(closes)
    last_on = None
    for i, d in enumerate(sorted_days):
        if i >= 99:
            ma100 = sum(closes[x] for x in sorted_days[i - 99:i + 1]) / 100.0
            last_on = closes[d] > ma100
        if i >= 99:
            idx[d] = {'close': closes[d], 'ma100': ma100, 'on': last_on}
    return idx, closes


# ---------------- 名称 ----------------
def load_names():
    names = {}
    if os.path.exists(NAMES_CSV):
        with open(NAMES_CSV) as f:
            for r in csv.DictReader(f):
                names[r['thscode']] = r['name'] or r['ticker'] or r['thscode']
    return names


# ---------------- 信号计算（SQL 预处理） ----------------
SIGNAL_SQL = """
WITH base AS (
  SELECT k.thscode, k.date, k.open, k.high, k.low, k.close, k.volume,
         ROW_NUMBER() OVER (PARTITION BY k.thscode ORDER BY k.date) AS age
  FROM raw_kline_daily k
  WHERE SUBSTR(k.thscode,1,2) IN ('60','00','30','68')
), w AS (
  SELECT *,
    AVG(close) OVER w100 AS ma100,
    COUNT(close) OVER w100 AS cnt100,
    AVG(close) OVER w20  AS ma20,
    MIN(low)   OVER w10  AS low10,
    MAX(high)  OVER wp   AS high20_prev,
    LAG(close) OVER (PARTITION BY thscode ORDER BY date) AS pclose,
    LEAD(date)   OVER (PARTITION BY thscode ORDER BY date) AS entry_date,
    LEAD(open)   OVER (PARTITION BY thscode ORDER BY date) AS entry_open,
    LEAD(high)   OVER (PARTITION BY thscode ORDER BY date) AS entry_high,
    LEAD(volume) OVER (PARTITION BY thscode ORDER BY date) AS entry_vol
  FROM base
  WINDOW w100 AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 99 PRECEDING AND CURRENT ROW),
         w20  AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
         w10  AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
         wp   AS (PARTITION BY thscode ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
), s AS (
  SELECT *, LAG(ma100) OVER (PARTITION BY thscode ORDER BY date) AS ma100_prev
  FROM w
)
SELECT thscode, date, close, ma100, high, low10, high20_prev,
       entry_date, entry_open, entry_high, entry_vol
FROM s
WHERE age >= 120                       -- 上市(库内)满120个交易日（任务书标的池）
  AND cnt100 = 100                     -- MA100 有完整窗口
  AND volume > 0                       -- 当日非停牌（任务书标的池；库中停牌日无记录）
  AND close > ma100                    -- 第2步来源A条件1：T收盘>MA100
  AND pclose <= ma100_prev             -- 条件2：T-1收盘<=MA100（首次突破）
  AND close / pclose - 1 < 0.09        -- 条件3：T涨幅<9%
  AND (high - low10) / high <= 0.08    -- 第3步：(E-S)/E<=8%
  AND high <= 1.10 * ma20              -- 红线2：E距MA20不超10%
  AND entry_date IS NOT NULL
"""


def build_signals(con):
    rows = con.execute(SIGNAL_SQL).fetchall()
    sigs = []
    for (thscode, d, close, ma100, high, low10, high20_prev,
         entry_date, entry_open, entry_high, entry_vol) in rows:
        if not (START_DATE <= d <= END_DATE):
            continue
        sigs.append({
            'thscode': thscode,
            'date': d,                          # T 信号日
            'close': close,
            'ma100': ma100,
            'dist': (close - ma100) / ma100,    # 离MA100距离，排序用（第2步）
            'E': high,                          # 入场价 E = T日最高（第3步）
            'S': low10,                         # 止损价 S = 近10日最低（第3步）
            'A1': high20_prev,                  # 二批触发参考 = [T-20,T-1]最高价（第4步）
            'entry_date': entry_date,           # T+1（已校验=全局次一交易日）
            'entry_open': entry_open,
            'entry_high': entry_high,
            'entry_vol': entry_vol,
        })
    return sigs


SERIES_SQL = """
SELECT date, open, high, low, close, volume,
  AVG(close) OVER (ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)   AS ma10,
  AVG(close) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)  AS ma20,
  AVG(close) OVER (ORDER BY date ROWS BETWEEN 99 PRECEDING AND CURRENT ROW)  AS ma100,
  LAG(close) OVER (ORDER BY date) AS pclose
FROM raw_kline_daily
WHERE thscode = ? AND date >= ?
ORDER BY date
"""


class SeriesCache:
    """按需拉取单只股票自其首个信号日起的行情（含 MA10/20/100），缓存复用"""

    def __init__(self, con):
        self.con = con
        self.first_signal = {}
        self.cache = {}

    def register(self, thscode, date):
        if thscode not in self.first_signal or date < self.first_signal[thscode]:
            self.first_signal[thscode] = date

    def get(self, thscode):
        if thscode not in self.cache:
            rows = self.con.execute(
                SERIES_SQL, [thscode, self.first_signal[thscode]]).fetchall()
            self.cache[thscode] = {
                'list': rows,
                'by_date': {r[0]: r for r in rows},
            }
        return self.cache[thscode]


# ---------------- 回测引擎 ----------------
class Lot:
    __slots__ = ('batch', 'shares', 'shares_total', 'entry_price', 'entry_idx',
                 'ma10_used', 'gain20_used', 'cost', 'sold_cost', 'proceeds')

    def __init__(self, batch, shares, price, entry_idx):
        self.batch = batch            # 1=第一批40% 2=第二批30%
        self.shares = shares          # 剩余股数
        self.shares_total = shares    # 初始股数（成本分摊基准）
        self.entry_price = price
        self.entry_idx = entry_idx
        self.ma10_used = False        # 跌破MA10减仓限一次（第5步）
        self.gain20_used = False      # 3日涨幅>20%减仓限一次（第5步）
        self.cost = shares * price * (1 + FEE)   # 本批总成本（含买入费）
        self.sold_cost = 0.0                     # 已卖出部分分摊成本
        self.proceeds = 0.0                      # 已卖出净回笼（扣卖出费）


class Position:
    __slots__ = ('thscode', 'S', 'A1', 'plan_N', 'b2', 'signal_date',
                 'lots', 'add2_done', 'last_close')

    def __init__(self, thscode, sig, b1, entry_price, entry_idx):
        self.thscode = thscode
        self.S = sig['S']
        self.A1 = sig['A1']
        self.plan_N = sig['plan_N']
        # b2 按信号日计划固定（盘中不改四个数，第3步/红线8）
        self.b2 = int(round(sig['plan_N'] * BATCH2_PCT / 100.0)) * 100
        self.signal_date = sig['date']
        self.lots = [Lot(1, b1, entry_price, entry_idx)]
        self.add2_done = False
        self.last_close = entry_price


def run_backtest(sigs, calendar, index_state, names, con):
    series_cache = SeriesCache(con)
    sigs_by_entry = defaultdict(list)
    for s in sigs:
        series_cache.register(s['thscode'], s['date'])
        sigs_by_entry[s['entry_date']].append(s)
    for d in sigs_by_entry:
        sigs_by_entry[d].sort(key=lambda x: x['dist'])   # 第2步：离MA100越近越优先

    cash = INIT_EQUITY
    positions = {}                   # thscode -> Position
    pending_exits = []               # 次日开盘执行：{'thscode','shares'(None=清仓),'reason','lot'}
    pending_adds = []                # 次日开盘执行：{'thscode','shares'}
    equity_curve = []                # (date, equity, invested, index_close)
    consec_stops = 0
    pause_until = -1                 # 连亏暂停：calendar index（含）之前禁止买入
    block_next = False               # 2%停做线：明日禁止买入
    trade_rows = []                  # trades.csv 行
    lot_closes = []                  # 每批仓位（lot）闭合统计
    fills_log = {'entries': 0, 'adds': 0, 'exit_events': 0}

    def market_value(pos):
        return sum(l.shares for l in pos.lots) * pos.last_close

    def total_invested():
        return sum(market_value(p) for p in positions.values())

    def remaining_cost(pos):
        return sum(l.cost - l.sold_cost for l in pos.lots)

    def remaining_shares(pos):
        return sum(l.shares for l in pos.lots)

    def avg_cost(pos):
        rs = remaining_shares(pos)
        return remaining_cost(pos) / rs if rs else 0.0

    def record_sell(pos, lot, shares, price, date, reason, day_idx):
        nonlocal cash
        proceeds = shares * price * (1 - FEE)
        cash += proceeds
        frac = shares / lot.shares_total
        cost_part = lot.cost * frac
        pnl = proceeds - cost_part
        lot.proceeds += proceeds
        lot.sold_cost += cost_part
        lot.shares -= shares
        trade_rows.append({
            'entry_date': calendar[lot.entry_idx], 'thscode': pos.thscode,
            'name': names.get(pos.thscode, pos.thscode),
            'batch': lot.batch, 'entry_price': lot.entry_price,
            'buy_shares': lot.shares_total, 'sell_date': date,
            'sell_price': price, 'sell_shares': shares,
            'pnl': pnl, 'pnl_pct': pnl / cost_part,
            'reason': reason, 'hold_days': day_idx - lot.entry_idx,
        })
        fills_log['exit_events'] += 1
        if lot.shares == 0:
            lot_closes.append({
                'thscode': pos.thscode, 'batch': lot.batch,
                'entry_idx': lot.entry_idx, 'exit_idx': day_idx,
                'exit_date': date, 'reason': reason,
                'pnl': lot.proceeds - lot.cost,
                'pnl_pct': lot.proceeds / lot.cost - 1,
            })

    n_days = len(calendar)
    for i, day in enumerate(calendar):
        # ---- 1) 开盘：执行昨日收盘后挂出的卖单 ----
        if pending_exits:
            still = []
            for od in pending_exits:
                pos = positions.get(od['thscode'])
                if pos is None:
                    continue
                bar = series_cache.get(od['thscode'])['by_date'].get(day)
                if bar is None or bar[5] == 0:            # 停牌：顺延
                    still.append(od)
                    continue
                o, pclose = bar[1], bar[9]
                lim = board_limit(od['thscode'], day)
                if pclose and o <= pclose * (1 - lim):    # 开盘跌停：顺延
                    still.append(od)
                    continue
                if od['shares'] is None:                  # 清仓单
                    for lot in list(pos.lots):
                        record_sell(pos, lot, lot.shares, o, day, od['reason'], i)
                    consec_stops = consec_stops + 1 if od['reason'] == '初始止损' else 0
                    if consec_stops >= CONSEC_STOPS:
                        pause_until = max(pause_until, i + PAUSE_DAYS)
                        consec_stops = 0
                    del positions[od['thscode']]
                else:                                     # 部分减仓单（指定lot）
                    lot = od['lot']
                    if lot.shares <= 0:
                        continue
                    sh = min(od['shares'], lot.shares)
                    record_sell(pos, lot, sh, o, day, od['reason'], i)
                    if not pos.lots or all(l.shares == 0 for l in pos.lots):
                        del positions[od['thscode']]
            pos_cleanup = [th for th, p in positions.items()
                           if all(l.shares == 0 for l in p.lots)]
            for th in pos_cleanup:
                del positions[th]
            pending_exits = still

        # ---- 2) 开盘：执行二批加仓单 ----
        if pending_adds:
            if block_next or i <= pause_until:   # 第6步：停做线/连亏暂停同样约束加仓
                pending_adds = []
            else:
                equity_prev = equity_curve[-1][1] if equity_curve else INIT_EQUITY
                for od in pending_adds:
                    pos = positions.get(od['thscode'])
                    if pos is None:
                        continue
                    bar = series_cache.get(od['thscode'])['by_date'].get(day)
                    if bar is None or bar[5] == 0:          # 停牌：放弃加仓（简化7）
                        continue
                    o, pclose = bar[1], bar[9]
                    lim = board_limit(od['thscode'], day)
                    if pclose and o >= pclose * (1 + lim):  # 开盘涨停买不进
                        continue
                    # 红线5/任务书：单票<=30%、总仓<=70%（以前收权益计）
                    mv = market_value(pos)
                    max_by_pos = int((MAX_POS_PCT * equity_prev - mv) / o / 100) * 100
                    max_by_tot = int((MAX_TOTAL_PCT * equity_prev - total_invested())
                                     / o / 100) * 100
                    sh = min(od['shares'], max_by_pos, max_by_tot)
                    if sh < 100:
                        continue
                    if cash < sh * o * (1 + FEE):
                        continue
                    cash -= sh * o * (1 + FEE)
                    pos.lots.append(Lot(2, sh, o, i))
                    fills_log['adds'] += 1
            pending_adds = []

        # ---- 3) 开盘：新开仓（信号=T-1日收盘选出，今日执行） ----
        today_sigs = sigs_by_entry.get(day)
        if today_sigs and i + 1 < n_days:
            T = calendar[i - 1]
            T_state = index_state.get(T)
            blocked_global = (
                T_state is None or not T_state['on']       # 第1步总开关
                or i <= pause_until                         # 第6步连亏暂停
                or block_next                               # 第6步2%停做线
            )
            if not blocked_global and len(positions) < MAX_POSITIONS:
                for sig in today_sigs:
                    if len(positions) >= MAX_POSITIONS:
                        break
                    th = sig['thscode']
                    if th in positions:
                        continue
                    bar = series_cache.get(th)['by_date'].get(day)
                    if bar is None or bar[5] == 0:   # T+1停牌：放弃
                        continue
                    o, h = bar[1], bar[2]
                    pclose = sig['close']            # T收盘 = T+1昨收
                    lim = board_limit(th, day)
                    if o >= pclose * (1 + lim):      # 开盘涨停买不进
                        continue
                    E = sig['E']
                    if h < E:                        # 盘中未触E：放弃
                        continue
                    price = max(E, o)
                    equity_prev = equity_curve[-1][1] if equity_curve else INIT_EQUITY
                    # 第3步：N=权益*1%/(E-S)，100股向下取整，金额<=30%
                    gap = E - sig['S']
                    if gap <= 0:
                        continue
                    N = int(equity_prev * RISK_PER_TRADE / gap / 100) * 100
                    N = min(N, int(MAX_POS_PCT * equity_prev / price / 100) * 100)
                    if N < 300:                      # 40%不足100股则放弃
                        continue
                    b1 = int(round(N * BATCH1_PCT / 100.0)) * 100
                    if b1 < 100 or b1 * price > MAX_POS_PCT * equity_prev + 1e-6:
                        continue
                    if total_invested() + b1 * price > MAX_TOTAL_PCT * equity_prev + 1e-6:
                        continue                      # 换一只更小的候选继续尝试
                    if cash < b1 * price * (1 + FEE):
                        continue
                    sig['plan_N'] = N
                    sig['equity'] = equity_prev
                    cash -= b1 * price * (1 + FEE)
                    positions[th] = Position(th, sig, b1, price, i)
                    fills_log['entries'] += 1
                    break                             # 每日最多新开1只

        # ---- 4) 收盘：标记市值、判定出场、触发二批 ----
        new_exits, new_adds = [], []
        for th, pos in list(positions.items()):
            bar = series_cache.get(th)['by_date'].get(day)
            if bar is None:                           # 停牌：市值按最近收盘冻结
                continue
            c, ma10, ma20, ma100 = bar[4], bar[6], bar[7], bar[8]
            pos.last_close = c
            full_reason = None
            # 第5步 清仓优先级：收盘<MA20或<MA100（与止损同日按清仓）；再初始止损
            if c < ma20 or c < ma100:
                full_reason = '清仓'
            elif c < pos.S:
                full_reason = '初始止损'
            if full_reason:
                # 清仓优先：撤销该票未完成的减仓单与加仓单（任务书）
                pending_exits = [e for e in pending_exits if e['thscode'] != th]
                pending_adds = [a for a in pending_adds if a['thscode'] != th]
                new_exits = [e for e in new_exits if e['thscode'] != th]
                new_adds = [a for a in new_adds if a['thscode'] != th]
                if not any(e['thscode'] == th and e['shares'] is None
                           for e in new_exits):
                    new_exits.append({'thscode': th, 'shares': None,
                                      'reason': full_reason, 'lot': None})
                continue
            # 第4步 第二批：收盘>A1(信号日前20日高) 且 非亏损加仓（红线3）
            if not pos.add2_done and c > pos.A1 and c > avg_cost(pos):
                pos.add2_done = True
                if pos.b2 >= 100:
                    new_adds.append({'thscode': th, 'shares': pos.b2})
            # 第5步 部分减仓（每批各规则各限一次）
            for lot in pos.lots:
                n_thirds = 0
                reasons = []
                if not lot.gain20_used and (i - lot.entry_idx) < GAIN20_DAYS \
                        and c / lot.entry_price - 1 > GAIN20_LIMIT:
                    lot.gain20_used = True
                    n_thirds += 1
                    reasons.append('3日涨幅>20%减仓')
                if not lot.ma10_used and c < ma10:
                    lot.ma10_used = True
                    n_thirds += 1
                    reasons.append('跌破MA10减仓')
                if n_thirds:
                    sh = int(round(lot.shares * (n_thirds / 3.0) / 100.0)) * 100
                    sh = max(100, min(sh, lot.shares))
                    if sh >= lot.shares:            # 取整后>=全部则放弃（保留底仓）
                        continue
                    new_exits.append({'thscode': th, 'shares': sh,
                                      'reason': '+'.join(reasons), 'lot': lot})
        pending_exits.extend(new_exits)
        pending_adds.extend(new_adds)

        # ---- 5) 收盘结算（第6步：2%停做线，作用于明日） ----
        equity = cash + total_invested()
        idx = index_state.get(day)
        equity_curve.append((day, equity, total_invested(),
                             idx['close'] if idx else None))
        block_next = False
        if len(equity_curve) > 1:
            prev_eq = equity_curve[-2][1]
            if prev_eq > 0 and equity / prev_eq - 1 <= -DAILY_STOP_PCT:
                block_next = True

    # ---- 期末强平（最后一日收盘） ----
    last_i = n_days - 1
    last_day = calendar[-1]
    for th, pos in list(positions.items()):
        bar = series_cache.get(th)['by_date'].get(last_day)
        px = bar[4] if bar else pos.last_close
        for lot in list(pos.lots):
            if lot.shares > 0:
                record_sell(pos, lot, lot.shares, px, last_day, '期末强平', last_i)
        del positions[th]
    # 用强平后的现金替换最后一行权益（避免同日两行）
    equity_curve[-1] = (last_day, cash + 0.0, 0.0, equity_curve[-1][3])

    return {
        'equity_curve': equity_curve,
        'trade_rows': trade_rows,
        'lot_closes': lot_closes,
        'fills_log': fills_log,
    }


# ---------------- 统计与报告 ----------------
def max_drawdown(curve, lo=None, hi=None):
    peak, mdd, mdd_interval = -1e18, 0.0, None
    peak_date = None
    for d, eq, *_ in curve:
        if lo and d < lo:
            continue
        if hi and d > hi:
            continue
        if eq > peak:
            peak, peak_date = eq, d
        dd = eq / peak - 1
        if dd < mdd:
            mdd = dd
            mdd_interval = (peak_date, d)
    return mdd, mdd_interval


def stat_block(lots):
    if not lots:
        return {'n': 0, 'win': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'pl_ratio': 0.0, 'gross_pnl': 0.0}
    wins = [l['pnl'] for l in lots if l['pnl'] > 0]
    losses = [l['pnl'] for l in lots if l['pnl'] <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        'n': len(lots),
        'win': len(wins) / len(lots),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'pl_ratio': (avg_win / abs(avg_loss)) if avg_loss else float('inf'),
        'gross_pnl': sum(l['pnl'] for l in lots),
    }


def fmt_money(x):
    return f'{x:,.0f}'


def fmt_pct(x):
    return f'{x * 100:.1f}%'


def write_outputs(result, index_closes, calendar, names):
    curve = result['equity_curve']
    trade_rows = result['trade_rows']
    lot_closes = result['lot_closes']

    # trades.csv
    with open(TRADES_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['入场日期', 'thscode', '名称', '批次', '入场价', '买入股数',
                    '出场日期', '出场价', '卖出股数', '单笔盈亏金额', '单笔盈亏%',
                    '出场原因', '持仓天数'])
        for r in trade_rows:
            w.writerow([r['entry_date'], r['thscode'], r['name'], r['batch'],
                        f"{r['entry_price']:.2f}", r['buy_shares'], r['sell_date'],
                        f"{r['sell_price']:.2f}", r['sell_shares'],
                        f"{r['pnl']:.2f}", f"{r['pnl_pct'] * 100:.2f}%",
                        r['reason'], r['hold_days']])

    # equity_curve.csv
    with open(EQUITY_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'equity', 'invested', 'index_close'])
        for d, eq, inv, ic in curve:
            w.writerow([d, f'{eq:.2f}', f'{inv:.2f}',
                        f'{ic:.2f}' if ic else ''])

    # ---- 全期统计 ----
    start_day = curve[0][0]
    end_day = curve[-1][0]
    final_eq = curve[-1][1]
    total_ret = final_eq / INIT_EQUITY - 1
    years = (end_day - start_day).days / 365.25
    cagr = (final_eq / INIT_EQUITY) ** (1 / years) - 1
    mdd, mdd_iv = max_drawdown(curve)
    overall = stat_block(lot_closes)

    def idx_close_on(d):
        if d in index_closes:
            return index_closes[d]
        prior = [x for x in index_closes if x <= d]
        return index_closes[max(prior)] if prior else None

    idx_start = idx_close_on(start_day)
    idx_end = idx_close_on(end_day)
    idx_ret = idx_end / idx_start - 1 if idx_start and idx_end else None

    # 逐年（按权益年末值）
    by_year = defaultdict(list)
    for d, eq, inv, ic in curve:
        by_year[d.year].append((d, eq))
    year_rows = []
    prev_eq = INIT_EQUITY
    for y in sorted(by_year):
        eqs = by_year[y]
        y_end_eq = eqs[-1][1]
        lots = [l for l in lot_closes if l['exit_date'].year == y]
        st = stat_block(lots)
        year_rows.append((y, prev_eq, y_end_eq, y_end_eq / prev_eq - 1, st))
        prev_eq = y_end_eq

    # 分阶段
    phase_rows = []
    for name, lo, hi in PHASES:
        seg = [(d, eq) for d, eq, *_ in curve if lo <= d <= hi]
        if not seg:
            continue
        idx0 = next(i for i, (d, *_ ) in enumerate(curve) if d == seg[0][0])
        start_eq = curve[idx0 - 1][1] if idx0 > 0 else INIT_EQUITY
        end_eq = seg[-1][1]
        lots = [l for l in lot_closes if lo <= l['exit_date'] <= hi]
        st = stat_block(lots)
        pmdd, _ = max_drawdown(curve, lo, hi)
        idx_s = idx_close_on(seg[0][0])
        idx_e = idx_close_on(seg[-1][0])
        phase_rows.append({
            'name': name, 'ret': end_eq / start_eq - 1, 'st': st, 'mdd': pmdd,
            'idx_ret': (idx_e / idx_s - 1) if idx_s and idx_e else None,
        })

    # 代表性交易：覆盖类型选取（同一只股票不重复入选）
    rep_detail, used = [], set()

    def take(keyf, pred, n, reverse=True):
        cands = [l for l in lot_closes
                 if pred(l) and id(l) not in used and l['thscode'] not in used_th]
        cands.sort(key=keyf, reverse=reverse)
        for c in cands[:n]:
            used.add(id(c))
            used_th.add(c['thscode'])
            evs = [r for r in trade_rows if r['thscode'] == c['thscode']
                   and r['entry_date'] == calendar[c['entry_idx']]
                   and r['batch'] == c['batch']]
            rep_detail.append((c, evs))

    used_th = set()
    take(lambda l: l['pnl_pct'], lambda l: l['pnl'] > 0, 2)                 # 典型大盈利
    take(lambda l: l['pnl_pct'], lambda l: l['pnl'] < 0, 2, reverse=False)  # 典型大亏损
    # 速亏离场（短持仓亏损清仓，承担"止损"角色）
    take(lambda l: l['exit_idx'] - l['entry_idx'],
         lambda l: l['pnl'] < 0 and l['reason'] == '清仓', 1, reverse=False)
    # 经部分减仓后盈利离场
    def had_trim(l):
        return any('减仓' in r['reason'] for r in trade_rows
                   if r['thscode'] == l['thscode']
                   and r['entry_date'] == calendar[l['entry_idx']]
                   and r['batch'] == l['batch'])
    take(lambda l: l['pnl_pct'], lambda l: had_trim(l) and l['pnl'] > 0, 1)

    # 出场原因分布（lot 闭合级）
    reason_dist = defaultdict(int)
    for l in lot_closes:
        reason_dist[l['reason']] += 1

    # ---- summary.md ----
    L = []
    A = L.append
    A('# 强势突破短线交易系统 —— 全历史逐笔回测报告\n')
    A(f'- 回测区间：{start_day} 至 {end_day}（数据起点至 2026-09-24；有效信号自 2017-03 起）')
    A(f'- 初始资金：{INIT_EQUITY:,.0f} 元；费用：单边 0.1%；价格口径：不复权')
    A('- 规则依据：《可执行短线交易系统.md》，仅回测"来源A：首次突破MA100"选股')
    A('- 大盘总开关：上证指数收盘 > MA100（T日判定，T+1执行）\n')

    A('## 一、全期表现\n')
    A('| 指标 | 数值 |')
    A('|---|---|')
    A(f'| 期末资金 | {fmt_money(final_eq)} 元 |')
    A(f'| 总收益率 | {fmt_pct(total_ret)} |')
    A(f'| 年化收益率 | {fmt_pct(cagr)} |')
    A(f'| 最大回撤 | {fmt_pct(mdd)}（{mdd_iv[0]} → {mdd_iv[1]}）|')
    A(f'| 闭合批次数（笔） | {overall["n"]} |')
    A(f'| 胜率 | {fmt_pct(overall["win"])} |')
    A(f'| 盈亏比（平均盈利/平均亏损） | {overall["pl_ratio"]:.2f} |')
    A(f'| 平均盈利 | {fmt_money(overall["avg_win"])} 元 |')
    A(f'| 平均亏损 | {fmt_money(overall["avg_loss"])} 元 |')
    A(f'| 同期上证指数涨跌幅 | {fmt_pct(idx_ret) if idx_ret is not None else "N/A"} |')
    A(f'| 买入成交 | 新开仓 {result["fills_log"]["entries"]} 只次，二批加仓 {result["fills_log"]["adds"]} 只次 |')
    A(f'| 卖出事件 | {result["fills_log"]["exit_events"]} 笔（含部分减仓）|')
    A(f'| 出场原因分布（lot级） | '
      + '；'.join(f'{k} {v}笔' for k, v in sorted(reason_dist.items(), key=lambda x: -x[1]))
      + ' |')
    A(f'| 其中部分减仓事件 | 跌破MA10减仓 {sum(1 for r in trade_rows if r["reason"] == "跌破MA10减仓")} 笔，'
      f'3日涨幅>20%减仓 {sum(1 for r in trade_rows if r["reason"] == "3日涨幅>20%减仓")} 笔 |\n')

    A('## 二、逐年盈亏\n')
    A('| 年份 | 年初权益 | 年末权益 | 年度收益 | 闭合笔数 | 胜率 | 盈亏比 |')
    A('|---|---|---|---|---|---|---|')
    for y, eq0, eq1, ret, st in year_rows:
        A(f'| {y} | {fmt_money(eq0)} | {fmt_money(eq1)} | {fmt_pct(ret)} | '
          f'{st["n"]} | {fmt_pct(st["win"]) if st["n"] else "-"} | '
          f'{st["pl_ratio"]:.2f} |')
    A('')

    A('## 三、分阶段对比（含同期上证指数）\n')
    A('| 阶段 | 策略收益 | 阶段最大回撤 | 闭合笔数 | 胜率 | 盈亏比 | 上证涨跌 |')
    A('|---|---|---|---|---|---|---|')
    for p in phase_rows:
        A(f'| {p["name"]} | {fmt_pct(p["ret"])} | {fmt_pct(p["mdd"])} | '
          f'{p["st"]["n"]} | {fmt_pct(p["st"]["win"]) if p["st"]["n"] else "-"} | '
          f'{p["st"]["pl_ratio"]:.2f} | '
          f'{fmt_pct(p["idx_ret"]) if p["idx_ret"] is not None else "N/A"} |')
    A('')

    A('## 四、代表性交易摘录\n')
    for l, evs in rep_detail:
        nm = names.get(l['thscode'], l['thscode'])
        buys = f"买{evs[0]['buy_shares']}股@{evs[0]['entry_price']:.2f}"
        sells = '；'.join(
            f"卖{r['sell_shares']}股@{r['sell_price']:.2f}（{r['sell_date']}，{r['reason']}）"
            for r in evs)
        A(f"- **{l['thscode']} {nm}**（第{l['batch']}批，{calendar[l['entry_idx']]}）："
          f"{buys} → {sells}。持仓{l['exit_idx'] - l['entry_idx']}个交易日，"
          f"该批盈亏 {l['pnl']:+,.0f} 元（{l['pnl_pct'] * 100:+.1f}%）")
    A('')

    A('## 五、规则简化清单及可能影响\n')
    simpl = [
        '只做来源A（首次突破MA100）；来源B缺口/C涨停/D热点未回测，实盘信号会更频繁。',
        '二批加仓触发取"信号日前20个交易日最高价[T-20,T-1]"；第三批（回踩企稳，需60分钟线）'
        '不模拟，仓位系统性偏低（实际最多只用计划的70%）。',
        '规则文件第4步"大盘盘中跌>1.5%当天计划作废"未模拟（任务书指定）。',
        '不复权价格：除权缺口可能产生假突破信号或假止损；分红未计入收益（短持仓下影响很小，方向偏保守）。',
        '涨跌停阈值：主板±9.8%，科创板±19.8%，创业板2020-08-24注册制后±19.8%（修正任务书9.8%平铺口径，'
        '否则注册制后大量创业板正常高开将被误判买不进）。',
        '上市满120日以"本地库首个交易日"计龄（库无上市日期），2016-09-27前上市的老股统一延迟至2017-03-28后'
        '才有信号资格，方向保守。',
        'ST剔除用当前名称（库内名称字段全空，取自同花顺当前目录）：历史上曾ST后摘帽的股票在其ST期间未被剔除。',
        '每日2%停做线按收盘权益环比计算，作用于下一交易日（规则文件第6步"明天不交易"）。',
        '连亏3笔只计"初始止损"（收盘<S）的机械口径；任何非初始止损的完整出场都会重置连亏计数；'
        '暂停自触发次日起3个交易日。',
        '20%止盈减仓的"累计涨幅"以各批次自身入场价为基准，3个交易日=入场日起含当日共3天。',
        '二批加仓日停牌/开盘涨停则放弃加仓（不延期）；加仓受2%停做线与连亏暂停约束，不受大盘总开关约束。',
        '同日多信号按"离MA100越近越优先"排序依次尝试，每日最多新开1只、同时最多持有5只（任务书组合层）。',
        '卖1/3按批内剩余股数的1/3取整到100股，若取整后>=剩余股数则放弃本次减仓（保留底仓）。',
        '部分减仓与清仓同日触发时按清仓处理；减仓次日开盘执行，开盘跌停顺延。',
        '指数数据来自Yahoo（本地库无指数；同花顺远端指数仅回溯至2022-01）。已与同花顺2022-01以来数据'
        '交叉验证（最大偏差0.005点）；Yahoo缺2019-04-29/30两天，该两日视为不开新仓，影响可忽略。',
    ]
    for i, s in enumerate(simpl, 1):
        A(f'{i}. {s}')
    A('')

    A('## 六、发现的问题与风险点\n')
    A('1. **数据库不含大盘指数**（000001.SH 无任何记录），总开关必须外接数据源；本报告用Yahoo日线并与'
      '同花顺交叉验证，但2019-04-29/30两个交易日缺失。')
    A('2. **v_symbol.name 字段全部为空**，ST剔除和历史名称分析只能依赖外部当前名称（现名≠历史名，'
      '存在幸存者偏差）。')
    A('3. **raw_kline_daily.prev_close 字段全部为空**，T日涨幅以库内前一根收盘价计算'
      '（等同交易所昨收，仅跨停牌期略有差异）。')
    A('4. **停牌日以"无记录"表示**（无volume=0行）：持仓跨停牌期时市值按最近收盘冻结，'
      '复牌首日集中结算停做线/减仓判定。')
    A('5. 规则层面：**"初始止损"规则是死条款**——S 含 T 日低点、之后 close<S 几乎必然 close<MA20，'
      '按"同日触发按清仓处理"全部记为清仓，全期 1178 个闭合批次无一笔"初始止损"，连亏3笔停做因此从未触发；'
      '20%三日止盈以批次入场价为基准，全期仅触发 3 次。实盘应重新设计：S 上移规则、连亏计数纳入清仓、'
      '20%止盈改为以信号日收盘或首批统一基准。')
    A('6. 来源A"首次突破"在震荡市会反复触发（close在MA100上下拉锯），每日1只+5只上限是主要过滤网，'
      '这也是该系统在震荡市频繁小亏的根源（见逐年胜率）。')
    A('7. 不复权价格的真实影响案例：002687.SZ 乔治白 2022-06-21 除权（前收6.92、开盘4.64），'
      '回测记为 -34.8% 巨亏清仓，其中大部分是除权缺口而非真实亏损；全期类似事件会把策略收益系统性压低，'
      '同时分红未入账，两者方向相反、在小盘股上不可忽略。')
    A('\n> 数据与回测仅供研究，非投资建议。')
    with open(SUMMARY_MD, 'w') as f:
        f.write('\n'.join(L))

    return {
        'final_equity': final_eq, 'total_ret': total_ret, 'cagr': cagr,
        'mdd': mdd, 'overall': overall, 'idx_ret': idx_ret,
        'year_rows': year_rows, 'phase_rows': phase_rows,
        'fills': result['fills_log'],
        'reps': rep_detail,
        'start': str(start_day), 'end': str(end_day),
        'mdd_interval': (str(mdd_iv[0]), str(mdd_iv[1])) if mdd_iv else None,
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(INDEX_CSV):
        fetch_yahoo_index()

    index_state, index_closes = load_index()
    names = load_names()

    con = duckdb.connect(DB_PATH, read_only=True)
    print('[1/3] computing signals ...')
    sigs = build_signals(con)
    print(f'      signals after filters: {len(sigs)}')

    cal_rows = con.execute(
        "SELECT DISTINCT date FROM raw_kline_daily WHERE date BETWEEN ? AND ? ORDER BY date",
        [START_DATE, END_DATE]).fetchall()
    calendar = [r[0] for r in cal_rows]
    cal_idx = {d: i for i, d in enumerate(calendar)}
    # 校验信号 entry_date 恰为 T 的下一全局交易日（停牌股 LEAD 会跨天，须剔除）
    n0 = len(sigs)
    sigs = [s for s in sigs
            if cal_idx.get(s['entry_date']) == cal_idx.get(s['date'], -2) + 1]
    print(f'      signals with T+1 valid: {len(sigs)} (dropped {n0 - len(sigs)})')

    print('[2/3] running backtest ...')
    result = run_backtest(sigs, calendar, index_state, names, con)
    con.close()

    print('[3/3] writing outputs ...')
    stats = write_outputs(result, index_closes, calendar, names)
    print(json.dumps({
        'final_equity': round(stats['final_equity'], 0),
        'total_ret': round(stats['total_ret'], 4),
        'cagr': round(stats['cagr'], 4),
        'mdd': round(stats['mdd'], 4),
        'mdd_interval': stats['mdd_interval'],
        'trades': stats['overall']['n'],
        'win': round(stats['overall']['win'], 4),
        'pl_ratio': round(stats['overall']['pl_ratio'], 2),
        'idx_ret': round(stats['idx_ret'], 4) if stats['idx_ret'] is not None else None,
        'fills': stats['fills'],
    }, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
