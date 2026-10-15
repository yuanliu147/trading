#!/usr/bin/env python3
"""缠论A股实战交易系统(直观执行版) —— 全市场回测
规则来源: ~/self_dir/chzhshch-108-plus/books/缠论A股实战交易系统.md 上篇

- 总开关: 上证指数日线 MACD(12,26,9) DIF/DEA 均在 0 轴上方才允许买入
- 入场券: 收盘>年线; 年线20日走平/上拐; 上市≥130根K线; 近250日涨幅<100%; 非ST(name_map快照)
- 图形一(上升回踩≈二买): MA5>MA10; 近40日摆动高点h(3≤t-h≤25); 回调段均量<0.5×上涨段(高点前10日)均量;
  回调最低价触及MA10×1.03以内; 回调无连续2日收MA10下; 当日放量(>1.5×前5日均量)收阳收复MA5
- 图形二(突破回抽≈三买): 15日箱体(振幅≤15%); 放量(≥2×前5日均量)阳线(涨幅≥3%)突破箱顶;
  回抽2~15日, 回抽均量<0.5×突破日量, 回抽最低价不破箱顶×0.99; 当日重新放量上涨
- 禁令: 当日涨幅>5%不买; 涨停不买; 跌停不买
- 仓位: 首笔=权益1/6; 浮盈且再现图形一信号补仓1/6(限一次, 市值≤1/3); 同时≤2只; 现金≥权益1/3
- 卖出(次日尾盘成交, T+1): ①收盘跌破买入理由位(回抽低点/箱顶)→清仓; ②收盘跌破MA10→清仓;
  ③60日新高+当前红柱峰<0.5×前波红柱峰+(上影>2×实体且>1%收盘价|阴线实体≥2%)→减半(一次); MA5死叉MA10→清仓; 执行日跌停顺延
- 费用: 佣金万2.5(最低5元)双边; 印花税卖出0.05%; 滑点万5双边
- 数据: 本地DuckDB(hithink-finance)前复权日线; 上证指数 shcomposite_daily.csv
用法: python3 chan_system_backtest.py [--start 2018-01-01] [--end 2026-10-09] [--equity 10000]
"""
import argparse, json, os, sys, time
import duckdb, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'

COMMISSION, MIN_COMM, STAMP, SLIP = 0.00025, 5.0, 0.0005, 0.0005
MAX_POS, FIRST_FRAC, ADD_FRAC, CASH_FLOOR = 2, 1/6, 1/6, 1/3


# ---------- 数据 ----------
def universe(min_bars=300, min_med_amt=5e7):
    con = duckdb.connect(DB, read_only=True)
    rows = con.execute("""
        SELECT thscode FROM (
          SELECT thscode, COUNT(*) n, MEDIAN(amount) med FROM v_daily_qfq GROUP BY thscode
        ) WHERE n >= ? AND med >= ?""", [min_bars, min_med_amt]).fetchall()
    con.close()
    return [r[0] for r in rows]

def load_all(symbols):
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("""
        SELECT thscode, date, open, high, low, close, volume, amount
        FROM v_daily_qfq WHERE thscode IN (SELECT UNNEST(?::VARCHAR[]))
        ORDER BY thscode, date""", [symbols]).fetchdf()
    con.close()
    return df

def load_index():
    df = pd.read_csv(os.path.join(HERE, 'shcomposite_daily.csv'), parse_dates=['date'])
    c = df['close']
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    df['gate_on'] = (dif > 0) & (dea > 0)
    return df.set_index('date')

def load_st_names():
    d = json.load(open(os.path.join(HERE, 'name_map.json')))
    return {k for k, v in d.items() if 'ST' in str(v).upper()}


# ---------- 单票信号预计算 ----------
def limit_pct(code, date):
    pre = code.split('.')[0][:3]
    if pre in ('300', '301'):
        return 0.20 if date >= pd.Timestamp('2020-08-24') else 0.10
    if pre == '688':
        return 0.20
    if pre in ('920', '8') or code.endswith('.BJ'):
        return 0.30
    return 0.10

def prep_stock(code, g):
    """返回该票带信号/止损的日线表, 信号只用t日及之前数据"""
    g = g.sort_values('date').reset_index(drop=True)
    n = len(g)
    c, o, h, l, v = (g[x].to_numpy(float) for x in ('close', 'open', 'high', 'low', 'volume'))
    dates = g['date'].to_numpy()

    ma5 = pd.Series(c).rolling(5).mean().to_numpy()
    ma10 = pd.Series(c).rolling(10).mean().to_numpy()
    ma250 = pd.Series(c).rolling(250).mean().to_numpy()
    vol5p = pd.Series(v).shift(1).rolling(5).mean().to_numpy()
    dif = pd.Series(c).ewm(span=12, adjust=False).mean() - pd.Series(c).ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    bar = 2 * (dif - dea)
    bar = bar.to_numpy()

    bars = np.arange(n)
    ret250 = c / pd.Series(c).shift(250).to_numpy() - 1
    ma250_flat_up = ma250 >= pd.Series(ma250).shift(20).to_numpy()
    ticket = (c > ma250) & ma250_flat_up & (ret250 < 1.0) & (bars >= 130)

    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    chg = c / prev_c - 1
    lp = np.array([limit_pct(code, pd.Timestamp(d)) for d in dates])
    limit_up = chg >= lp - 0.005
    limit_dn = chg <= -lp + 0.005

    # MACD 红柱段峰值: 当前正bar段截至t的最大值, 及前一正段最大值
    red_pk_cur = np.full(n, np.nan)
    red_pk_prev = np.full(n, np.nan)
    seg_pk, prev_pk, in_seg = 0.0, np.nan, False
    for t in range(n):
        if bar[t] > 0:
            seg_pk = max(seg_pk, bar[t]) if in_seg else bar[t]
            in_seg = True
        else:
            if in_seg:
                prev_pk = seg_pk
            in_seg, seg_pk = False, 0.0
        red_pk_cur[t] = seg_pk if in_seg else np.nan
        red_pk_prev[t] = prev_pk

    # 60日收盘新高
    hi60 = pd.Series(c).rolling(60).max().to_numpy()
    new_high = c >= hi60

    # 15日箱体
    boxH = pd.Series(h).rolling(15).max().to_numpy()
    boxL = pd.Series(l).rolling(15).min().to_numpy()
    box_rng = (boxH - boxL) / boxL
    is_box = box_rng <= 0.15

    sig1 = np.zeros(n, bool); stop1 = np.full(n, np.nan)
    sig2 = np.zeros(n, bool); stop2 = np.full(n, np.nan)
    vol_x = v / vol5p  # 放量倍数(对前5日)

    # 图形二突破日标记: 昨日收盘处于箱体且今日放量阳线收破箱顶
    brk = np.zeros(n, bool); brk_top = np.full(n, np.nan)
    for t in range(15, n):
        if is_box[t-1] and not np.isnan(vol5p[t]) and c[t] > boxH[t-1] and o[t] < c[t] \
                and v[t] >= 2 * vol5p[t] and chg[t] >= 0.03:
            brk[t] = True; brk_top[t] = boxH[t-1]

    for t in range(60, n):
        if not ticket[t]:
            continue
        if np.isnan(vol5p[t]) or limit_up[t] or limit_dn[t] or chg[t] > 0.05:
            continue
        # ---- 图形一 ----
        if ma5[t] > ma10[t] and c[t] > o[t] and c[t] > ma5[t] and v[t] > 1.5 * vol5p[t] and chg[t] > 0:
            i0 = max(0, t - 40)
            hh = h[i0:t+1]
            hp = i0 + int(np.argmax(hh))  # 近40日最高价日(最早出现)
            # 取窗口内最后一次新高日
            for cand in range(t, i0 - 1, -1):
                if h[cand] == hh.max():
                    hp = cand
                    break
            if 3 <= t - hp <= 25 and hp - 10 >= 0:
                up_v = v[hp-10:hp+1].mean()
                pb_v = v[hp+1:t].mean() if t - 1 > hp else np.inf
                pb_lo = l[hp+1:t].min() if t - 1 > hp else l[hp]
                touch_ma10 = bool((l[hp+1:t] <= ma10[hp+1:t] * 1.03).any())
                below = c[hp+1:t] < ma10[hp+1:t]
                consec2 = bool((pd.Series(below).rolling(2).sum() >= 2).any()) if len(below) else False
                if pb_v < 0.5 * up_v and touch_ma10 and not consec2:
                    sig1[t] = True; stop1[t] = pb_lo
        # ---- 图形二 ----
        if c[t] > prev_c[t] and v[t] > 1.5 * vol5p[t]:
            b0 = max(15, t - 15)
            for b in range(t - 2, b0 - 1, -1):  # 最近突破日
                if brk[b]:
                    pb_lo = l[b+1:t].min()
                    pb_v = v[b+1:t].mean()
                    if pb_lo >= brk_top[b] * 0.99 and pb_v < 0.5 * v[b]:
                        sig2[t] = True; stop2[t] = brk_top[b]
                    break

    g['ma5'], g['ma10'], g['ma250'] = ma5, ma10, ma250
    g['bar'], g['red_pk_cur'], g['red_pk_prev'], g['new_high'] = bar, red_pk_cur, red_pk_prev, new_high
    g['sig1'], g['stop1'], g['sig2'], g['stop2'] = sig1, stop1, sig2, stop2
    g['vol_x'], g['chg'], g['limit_dn'] = vol_x, chg, limit_dn
    g['ticket'] = ticket
    return g


# ---------- 回测 ----------
def run(start, end, equity0):
    t0 = time.time()
    st_set = load_st_names()
    syms = [s for s in universe() if s not in st_set and not s.endswith('.BJ')]
    print(f"宇宙: {len(syms)} 只 (剔除ST/北交所)", flush=True)
    df = load_all(syms)
    print(f"行情: {len(df)} 行, 加载 {time.time()-t0:.0f}s", flush=True)

    sig_rows, bars_map = [], {}
    for i, (code, g) in enumerate(df.groupby('thscode')):
        g = prep_stock(code, g)
        bars_map[code] = g.set_index('date')
        s = g[(g['sig1'] | g['sig2']) & g['ticket']]
        for _, r in s.iterrows():
            typ = '一' if r['sig1'] else '二'
            sig_rows.append((r['date'], code, typ, r['stop1'] if r['sig1'] else r['stop2'], r['vol_x']))
        if (i + 1) % 500 == 0:
            print(f"  预计算 {i+1}/{len(syms)}, {time.time()-t0:.0f}s", flush=True)
    sigs = pd.DataFrame(sig_rows, columns=['date', 'code', 'type', 'stop', 'vol_x'])
    sigs['date'] = pd.to_datetime(sigs['date'])
    sig_by_day = {d: g.sort_values('vol_x', ascending=False).to_dict('records')
                  for d, g in sigs.groupby('date')}
    print(f"信号: {len(sigs)} 条, 预计算完成 {time.time()-t0:.0f}s", flush=True)

    idx = load_index()
    cal = [d for d in idx.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)]

    cash, equity = equity0, equity0
    pos = {}          # code -> dict(shares, cost, stop, type, added, halved, buy_date)
    pending = []      # list of dict(code, shares, reason, kind) 次日尾盘执行
    trades, curve = [], []

    def fee_buy(amt):
        return max(amt * COMMISSION, MIN_COMM) + amt * SLIP

    def fee_sell(amt):
        return max(amt * COMMISSION, MIN_COMM) + amt * (STAMP + SLIP)

    def bar_of(code, d):
        b = bars_map.get(code)
        if b is None or d not in b.index:
            return None
        return b.loc[d]

    for d in cal:
        # 1. 执行昨日挂卖(今日尾盘价成交; 跌停顺延)
        still = []
        for od in pending:
            r = bar_of(od['code'], d)
            if r is None:
                still.append(od); continue
            if r['limit_dn']:
                still.append(od); continue
            p = pos.get(od['code'])
            if p is None:
                continue
            px = r['close'] * (1 - SLIP)
            sh = min(od['shares'], p['shares'])
            amt = sh * px
            fee = max(amt * COMMISSION, MIN_COMM) + amt * (STAMP + SLIP)
            cash += amt - fee
            p['shares'] -= sh
            trades.append(dict(code=od['code'], sell_date=d, sell_px=px, shares=sh,
                               reason=od['reason'], pnl=(px - p['cost']) * sh - fee,
                               ret=px / p['cost'] - 1))
            if p['shares'] <= 0:
                del pos[od['code']]
        pending = still

        # 2. 更新市值
        for code, p in pos.items():
            r = bar_of(code, d)
            if r is not None:
                p['last'] = r['close']
        mv = sum(p['shares'] * p['last'] for p in pos.values())
        equity = cash + mv
        curve.append((d, equity))

        # 3. 持仓卖出触发(次日执行)
        pend_codes = {od['code'] for od in pending}
        for code, p in list(pos.items()):
            if p['buy_date'] >= d or code in pend_codes:  # T+1; 已有挂单不重复
                continue
            r = bar_of(code, d)
            if r is None:
                continue
            reason, kind = None, 'all'
            if r['close'] < p['stop']:
                reason = '理由失效'
            elif r['close'] < r['ma10']:
                reason = '破10日线'
            elif r['ma5'] < r['ma10']:
                b = bars_map[code]
                loc = b.index.get_loc(d)
                if loc > 0 and b['ma5'].iloc[loc-1] >= b['ma10'].iloc[loc-1]:
                    reason = '5死叉10'
            if reason is None and not p['halved'] and r['new_high'] and not np.isnan(r['red_pk_prev']) \
                    and not np.isnan(r['red_pk_cur']) and r['red_pk_cur'] < 0.5 * r['red_pk_prev']:
                body = abs(r['close'] - r['open'])
                upper = r['high'] - max(r['close'], r['open'])
                big_yin = (r['open'] - r['close']) / r['open'] >= 0.02
                long_up = upper > 2 * body and upper / r['close'] > 0.01
                if big_yin or long_up:
                    reason, kind = '顶背驰减半', 'half'
            if reason:
                sh = p['shares'] if kind == 'all' else int(p['shares'] // 200) * 100
                if sh >= 100:
                    pending.append(dict(code=code, shares=sh, reason=reason))
                    pend_codes.add(code)
                    if kind == 'half':
                        p['halved'] = True

        # 4. 买入(总开关开才允许)
        gate = bool(idx.loc[d, 'gate_on'])
        if gate:
            # 补仓: 浮盈+再现图形一+未补过+市值<权益1/3
            for code, p in list(pos.items()):
                r = bar_of(code, d)
                if r is None or p['added'] or not r['sig1'] or not r['ticket']:
                    continue
                if r['close'] > p['cost'] and p['shares'] * r['close'] < equity / 3:
                    tgt = equity * ADD_FRAC
                    px = r['close'] * (1 + SLIP)
                    sh = int(tgt / px // 100) * 100
                    amt = sh * px
                    fee = max(amt * COMMISSION, MIN_COMM) + amt * SLIP
                    if sh >= 100 and cash - amt - fee >= equity * CASH_FLOOR:
                        cash -= amt + fee
                        p['cost'] = (p['cost'] * p['shares'] + px * sh) / (p['shares'] + sh)
                        p['shares'] += sh
                        p['added'] = True
                        trades.append(dict(code=code, sell_date=None, sell_px=None, shares=sh,
                                           reason='补仓', pnl=None, ret=None))
            # 新开仓
            for od in sig_by_day.get(d, []):
                if len(pos) >= MAX_POS:
                    break
                code = od['code']
                if code in pos:
                    continue
                r = bar_of(code, d)
                if r is None:
                    continue
                px = r['close'] * (1 + SLIP)
                sh = int(equity * FIRST_FRAC / px // 100) * 100
                amt = sh * px
                fee = max(amt * COMMISSION, MIN_COMM) + amt * SLIP
                if sh < 100 or cash - amt - fee < equity * CASH_FLOOR:
                    continue
                cash -= amt + fee
                pos[code] = dict(shares=sh, cost=px, stop=od['stop'], type=od['type'],
                                 added=False, halved=False, buy_date=d, last=r['close'])
                trades.append(dict(code=code, sell_date=None, sell_px=None, shares=sh,
                                   reason=f"买(图{od['type']})", pnl=None, ret=None))
        # 刷新当日权益(买入后)
        mv = sum(p['shares'] * p['last'] for p in pos.values())
        equity = cash + mv
        curve[-1] = (d, equity)

    # 期末强平
    last_d = cal[-1]
    for code, p in list(pos.items()):
        px = p['last'] * (1 - SLIP)
        amt = p['shares'] * px
        fee = max(amt * COMMISSION, MIN_COMM) + amt * (STAMP + SLIP)
        cash += amt - fee
        trades.append(dict(code=code, sell_date=last_d, sell_px=px, shares=p['shares'],
                           reason='期末', pnl=(px - p['cost']) * p['shares'] - fee,
                           ret=px / p['cost'] - 1))
    equity = cash
    curve[-1] = (last_d, equity)

    eq = pd.DataFrame(curve, columns=['date', 'equity']).drop_duplicates('date', keep='last')
    tr = pd.DataFrame(trades)
    return eq, tr


def report(eq, tr, equity0, start, end):
    eq = eq.set_index('date')
    eq['ret'] = eq['equity'].pct_change()
    tot = eq['equity'].iloc[-1] / equity0 - 1
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (1 + tot) ** (1 / yrs) - 1
    dd = (eq['equity'] / eq['equity'].cummax() - 1).min()
    closed = tr[tr['sell_date'].notna() & tr['pnl'].notna()]
    win = (closed['pnl'] > 0).mean() if len(closed) else float('nan')
    pl = (closed.loc[closed.pnl > 0, 'pnl'].mean() / abs(closed.loc[closed.pnl <= 0, 'pnl'].mean())
          if len(closed) and (closed.pnl <= 0).any() else float('nan'))

    print(f"\n===== 总览 {start} ~ {end}  本金 {equity0:,.0f} =====")
    print(f"期末权益 {eq['equity'].iloc[-1]:,.0f} | 总收益 {tot:+.1%} | 年化 {ann:+.1%} | 最大回撤 {dd:.1%}")
    print(f"平仓交易 {len(closed)} 笔 | 胜率 {win:.1%} | 盈亏比 {pl:.2f}")
    print(f"卖出原因分布:\n{closed.groupby('reason')['pnl'].agg(['count', 'sum']).to_string()}")

    yr = eq['equity'].resample('YE').last()
    yr_ret = yr.pct_change()
    yr_ret.iloc[0] = yr.iloc[0] / equity0 - 1
    print("\n===== 年度盈亏 =====")
    for d, r in yr_ret.items():
        e = yr.loc[d]
        print(f"{d.year}: {r:+8.1%}   年末权益 {e:>10,.0f}")

    mo = eq['equity'].resample('ME').last()
    mo_ret = mo.pct_change()
    mo_ret.iloc[0] = mo.iloc[0] / equity0 - 1
    print("\n===== 月度盈亏 =====")
    for y in sorted(set(mo_ret.index.year)):
        row = []
        for m in range(1, 13):
            key = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
            if key in mo_ret.index:
                row.append(f"{m:>2}月{mo_ret.loc[key]:+7.1%}")
        if row:
            print(f"{y}: " + ' '.join(row))
    return eq, closed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2018-01-01')
    ap.add_argument('--end', default='2026-10-09')
    ap.add_argument('--equity', type=float, default=10000)
    ap.add_argument('--no-fee', action='store_true', help='零佣金/印花税/滑点')
    a = ap.parse_args()
    if a.no_fee:
        COMMISSION = MIN_COMM = STAMP = SLIP = 0.0
    eq, tr = run(a.start, a.end, a.equity)
    eq, closed = report(eq, tr, a.equity, a.start, a.end)
    eq.to_csv(os.path.join(HERE, 'chan_eq_daily.csv'))
    tr.to_csv(os.path.join(HERE, 'chan_trades.csv'), index=False)
    print(f"\n已保存 chan_eq_daily.csv / chan_trades.csv")
