#!/usr/bin/env python3
"""阶梯双K交易法则 —— A股日线改造版回测（含变体扫描）

相对书中原版（USD/JPY 1H、可做空、T+0）的 A 股化改造：
- 仅做多；日线周期（规避 T+1）；24 日均线趋势过滤
- 进场：前收 > 24 日线，日内突破前两根 K 线最高价；无下降阶梯/收敛震荡阻挡；
  涨停日放弃进场；止损距离 >6% 的信号放弃
- 止损：最近确认摆动低点（道氏低点）下方 0.1%；收盘确认跌破，次日开盘离场；
  次日跌停封板则顺延
- 止盈：默认 1:1；可扫 tp_ratio；trail 模式下 1R 后改摆动低点跟踪（让利润奔跑）
- 保本规则：最高价进入止盈位 0.2% 以内，止损上移至成本
- 连亏 2 笔暂停 5 个交易日；每笔风险 = 本金 2%，不用杠杆；费万1+滑点万5

可选过滤器（--sweep 扫描）：
- regime：大盘等权指数 > MA60 才允许开仓（大盘状态过滤）
- vol_mult：进场日成交额 > k × 20日中位数（放量确认，过滤假突破）

数据源：本地 DuckDB（hithink-finance）A股前复权日线 2016-10 ~ 2026-10；
ETF 走 fund history 远程缓存（2024-01 ~ 2025-10）。

用法：
  python3 ladder_doublek_backtest.py                 # 全市场基线
  python3 ladder_doublek_backtest.py --sweep         # 变体扫描
  python3 ladder_doublek_backtest.py --symbol 600519.SH [--tp 2 --trail --regime --vol 1.5]
  python3 ladder_doublek_backtest.py --etf
"""
import json, os, sys, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = "/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb"

# ---------- 数据加载 ----------
def liquid_symbols(min_bars=600, min_med_amt=1e8):
    import duckdb
    con = duckdb.connect(DB_PATH, read_only=True)
    rows = con.execute("""
        SELECT thscode FROM (
          SELECT thscode, COUNT(*) n, MEDIAN(amount) med
          FROM v_daily_qfq GROUP BY thscode
        ) WHERE n >= ? AND med >= ?""", [min_bars, min_med_amt]).fetchall()
    con.close()
    return [r[0] for r in rows]

def load_stocks(symbols):
    import duckdb
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT thscode, date, open, high, low, close, amount
        FROM v_daily_qfq WHERE thscode IN (SELECT UNNEST(?::VARCHAR[]))
        ORDER BY thscode, date""", [symbols]).fetchdf()
    con.close()
    return df

def load_etf(sym):
    frames = []
    for f in glob.glob(f"/tmp/etf_{sym}_*.json"):
        d = json.load(open(f))
        for it in (d.get("data") or {}).get("item") or []:
            frames.append({
                "date": pd.to_datetime(it["date_ms"] + 8 * 3600_000, unit="ms").normalize(),
                "open": it["open_price"], "high": it["high_price"],
                "low": it["low_price"], "close": it["close_price"],
                "amount": it.get("turnover", 0.0)})
    if not frames:
        return None
    return pd.DataFrame(frames).drop_duplicates("date").sort_values("date").reset_index(drop=True)

def build_regime(df, ma_n=60):
    """全样本等权大盘代理：日收益均值累计成指数，指数 > MA60 为允许多单状态。
    用日期 T 收盘计算，engine 在 T+1 使用，无未来函数。"""
    px = df.pivot_table(index="date", columns="thscode", values="close")
    rets = px.pct_change()
    mkt = (1 + rets.mean(axis=1, skipna=True).fillna(0)).cumprod()
    ok = mkt > mkt.rolling(ma_n).mean()
    return {d: bool(v) for d, v in ok.items()}

# ---------- 板块相对强度 ----------
def load_sector_map():
    """股票 -> 同花顺一级行业（881xxx.TI）映射（成分缓存自 index constituents）"""
    m = {}
    for f in glob.glob(os.path.join(HERE, "data_cache/sectors/881*.json")):
        code = os.path.basename(f).replace(".json", "")
        d = json.load(open(f))
        for it in (d.get("data") or {}).get("item") or []:
            m.setdefault(it["thscode"], code)
    return m

def load_all_closes():
    import duckdb
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("SELECT thscode, date, close FROM v_daily_qfq ORDER BY thscode, date").fetchdf()
    con.close()
    return df

def build_sector_strength(df, smap, lookback=60, top_pct=0.2):
    """行业等权指数（成分股日收益均值累计），按近 lookback 日涨幅横截面排名，
    前 top_pct 为强势行业。返回 DataFrame[bool]：index=date, columns=行业代码。
    用日期 T 收盘计算，engine 在 T+1 使用，无未来函数。
    注意：成分股为当前快照，历史成员变动未考虑。"""
    df = df[df["thscode"].isin(smap)]
    px = df.pivot_table(index="date", columns="thscode", values="close")
    rets = px.pct_change()
    groups = {}
    for col in px.columns:
        groups.setdefault(smap[col], []).append(col)
    ind_rets = pd.DataFrame({ind: rets[cols].mean(axis=1) for ind, cols in groups.items()})
    ind_idx = (1 + ind_rets.fillna(0)).cumprod()
    perf = ind_idx / ind_idx.shift(lookback) - 1
    rank = perf.rank(axis=1, pct=True)
    return rank >= (1 - top_pct)

# ---------- 涨跌停阈值 ----------
def limit_pct(thscode):
    tic = thscode.split(".")[0]
    if thscode.endswith(".BJ"):
        return 0.30
    if tic.startswith(("30", "68")):
        return 0.20
    return 0.10

# ---------- 单标的回测引擎 ----------
def backtest_symbol(o, h, l, c, lim, amt=None, vmed=None, reg_ok=None, sec_ok=None,
                    risk_pct=0.02, ma_n=24, be_tol=0.002, stop_buf=0.001,
                    max_risk_dist=0.06, pause_days=5, comm=0.0001, slip=0.0005,
                    tp_ratio=1.0, trail=False, vol_mult=None,
                    trend_gate="price", brk_n=2):
    n = len(c)
    if n < ma_n + 10:
        return [], None
    ma = pd.Series(c).rolling(ma_n).mean().to_numpy()

    zig = []  # 已确认摆动点 (typ, price, idx)，高低交替
    def zig_update(t):  # t 日收盘后确认 t-1
        cnd = t - 1
        if cnd < 1:
            return
        pts = []
        if h[cnd] > h[cnd - 1] and h[cnd] > h[t]:
            pts.append(("H", h[cnd]))
        if l[cnd] < l[cnd - 1] and l[cnd] < l[t]:
            pts.append(("L", l[cnd]))
        for typ, price in pts:
            if zig and zig[-1][0] == typ:
                if (typ == "H" and price > zig[-1][1]) or (typ == "L" and price < zig[-1][1]):
                    zig[-1] = (typ, price, cnd)
            else:
                zig.append((typ, price, cnd))
        del zig[:-12]

    def last_two(typ):
        p = [x for x in zig if x[0] == typ]
        return p[-2:] if len(p) >= 2 else None

    def last_swing(typ):
        p = [x for x in zig if x[0] == typ]
        return p[-1] if p else None

    ladder_on, ladder_high = False, None
    box_on, box_top, box_bot = False, None, None
    pause_until, consec_loss = -1, 0
    capital, cash, shares = 1.0, 1.0, 0.0
    pos = None
    exit_pending = False
    trades = []
    equity = np.empty(n)

    def register_exit(px, t, reason):
        nonlocal shares, pos, exit_pending, consec_loss, pause_until, cash
        cash += shares * px * (1 - comm)
        pnl = (px - pos["entry"]) * shares
        trades.append({**pos, "exit_i": t, "exit": px, "reason": reason})
        if pnl < 0:
            consec_loss += 1
            if consec_loss >= 2:
                pause_until = t + pause_days
                consec_loss = 0
        else:
            consec_loss = 0
        shares, pos, exit_pending = 0.0, None, False

    for t in range(n):
        # ---- 盘前：结构状态（仅用 t-1 及之前信息）----
        struct_up = False
        if t > 0:
            hh, ll = last_two("H"), last_two("L")
            if hh and ll and hh[0][1] > hh[1][1] and ll[0][1] > ll[1][1]:
                ladder_on, ladder_high = True, hh[1][1]
            if ladder_on and c[t - 1] > ladder_high:
                ladder_on = False
            if hh and ll and hh[0][1] > hh[1][1] and ll[0][1] < ll[1][1]:
                if not box_on:
                    box_on, box_top, box_bot = True, hh[1][1], ll[1][1]
            if box_on and (c[t - 1] > box_top or c[t - 1] < box_bot):
                box_on = False
            # 结构上升趋势：两个上移高点 + 两个上移低点
            struct_up = bool(hh and ll and hh[0][1] < hh[1][1] and ll[0][1] < ll[1][1])

        # ---- 持仓管理（t > 进场日，T+1）----
        if exit_pending:
            down_sealed = o[t] <= c[t - 1] * (1 - lim + 0.002)
            if not down_sealed:
                register_exit(o[t] * (1 - slip), t, "stop")
        elif pos and t > pos["entry_i"]:
            if trail:
                if not pos["armed"] and h[t] >= pos["tp"]:  # 到达 1R，启动跟踪止盈
                    pos["armed"] = True
                    pos["stop"] = max(pos["stop"], pos["entry"])
                if pos["armed"]:
                    sw = last_swing("L")
                    if sw:
                        pos["stop"] = max(pos["stop"], sw[1] * (1 - stop_buf))
                if c[t] < pos["stop"]:
                    if t == n - 1:
                        register_exit(c[t] * (1 - slip), t, "trail")
                    else:
                        exit_pending = True
            else:
                if not pos["armed"] and h[t] >= pos["tp"] * (1 - be_tol):
                    pos["stop"] = max(pos["stop"], pos["entry"])
                    pos["armed"] = True
                if h[t] >= pos["tp"]:
                    register_exit(pos["tp"] * (1 - slip), t, "tp")
                elif c[t] < pos["stop"]:
                    if t == n - 1:
                        register_exit(c[t] * (1 - slip), t, "stop")
                    else:
                        exit_pending = True

        # ---- 进场 ----
        if pos is None and not exit_pending and ma_n <= t and not np.isnan(ma[t - 1]):
            up_sealed = c[t - 1] >= c[t - 2] * (1 + lim - 0.002) if t >= 2 else False
            if trend_gate == "price":
                trend_ok = c[t - 1] > ma[t - 1]
            elif trend_gate == "none":
                trend_ok = True
            elif trend_gate == "slope":
                trend_ok = t >= 6 and not np.isnan(ma[t - 6]) and ma[t - 1] > ma[t - 6]
            elif trend_gate == "struct":
                trend_ok = struct_up
            else:
                trend_ok = c[t - 1] > ma[t - 1]
            breakout = t >= brk_n and h[t] > np.max(h[t - brk_n:t])
            blocked = ladder_on or box_on or t <= pause_until or up_sealed
            if reg_ok is not None and not reg_ok[t - 1]:
                blocked = True
            if sec_ok is not None and not sec_ok[t - 1]:
                blocked = True
            if (vol_mult is not None and vmed is not None and not np.isnan(vmed[t])
                    and not amt[t] > vol_mult * vmed[t]):
                blocked = True
            if trend_ok and breakout and not blocked:
                level = np.max(h[t - brk_n:t])
                entry = max(o[t], level) * (1 + slip)
                sw = last_swing("L")
                base = sw[1] if sw else np.min(l[max(0, t - 10):t])
                stop = base * (1 - stop_buf)
                risk = entry - stop
                if 0 < risk / entry <= max_risk_dist and cash > 0:
                    sh = min(capital * risk_pct / risk, cash / (entry * (1 + comm)))
                    if sh > 0:
                        cash -= sh * entry * (1 + comm)
                        shares = sh
                        pos = {"entry_i": t, "entry": entry, "stop": stop,
                               "tp": entry + risk * tp_ratio, "shares": sh,
                               "risk_amt": sh * risk, "eq0": capital, "armed": False}
                        if c[t] < stop:   # 当天买入即破位，次日开盘离场（T+1）
                            exit_pending = True

        equity[t] = cash + shares * c[t]
        capital = equity[t]
        if t >= 2:
            zig_update(t)

    if pos:
        px = c[-1] * (1 - slip)
        cash += shares * px * (1 - comm)
        trades.append({**pos, "exit_i": n - 1, "exit": px, "reason": "eod"})
        equity[-1] = cash

    tr = pd.DataFrame(trades)
    if len(tr):
        tr["pnl"] = (tr["exit"] - tr["entry"]) * tr["shares"]
        tr["R"] = tr["pnl"] / tr["risk_amt"]
    return tr, equity

def build_regime3(df, ma_n=60):
    """三态大盘状态：等权指数 vs MA60 + MA60 斜率。
    up=指数>MA60且MA60上行；down=指数<MA60且MA60下行；其余=range。
    用日期 T 收盘计算，T+1 使用，无未来函数。"""
    px = df.pivot_table(index="date", columns="thscode", values="close")
    rets = px.pct_change()
    mkt = (1 + rets.mean(axis=1, skipna=True).fillna(0)).cumprod()
    ma = mkt.rolling(ma_n).mean()
    up = (mkt > ma) & (ma.diff() > 0)
    down = (mkt < ma) & (ma.diff() < 0)
    state = pd.Series("range", index=mkt.index)
    state[up] = "up"; state[down] = "down"
    return state.to_dict()

# ---------- 超跌反弹策略（非上升市的候选打法） ----------
def backtest_mr(o, h, l, c, lim, allow=None, rsi_n=14, rsi_in=30.0,
                ma_exit=10, max_hold=10, stop_pct=0.08,
                risk_pct=0.02, comm=0.0001, slip=0.0005):
    """极简超跌反弹：RSI(14)<30 收盘为信号，次日开盘买；
    收盘站回 MA10 / 持满10日 / 收盘跌破-8%止损 → 次日开盘卖。
    T+1 与涨跌停处理同主引擎。仓位：2%风险/8%止损≈25%本金。"""
    n = len(c)
    if n < 40:
        return [], None
    s = pd.Series(c)
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_n, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss)).to_numpy()
    maxit = s.rolling(ma_exit).mean().to_numpy()

    capital, cash, shares = 1.0, 1.0, 0.0
    pos = None
    exit_pending = False
    trades = []
    equity = np.empty(n)

    def register_exit(px, t, reason):
        nonlocal shares, pos, exit_pending, cash
        cash += shares * px * (1 - comm)
        trades.append({**pos, "exit_i": t, "exit": px, "reason": reason})
        shares, pos, exit_pending = 0.0, None, False

    for t in range(n):
        if exit_pending:
            down_sealed = o[t] <= c[t - 1] * (1 - lim + 0.002)
            if not down_sealed:
                register_exit(o[t] * (1 - slip), t, "mr_exit")
        elif pos and t > pos["entry_i"]:
            if c[t] < pos["stop"] or c[t] > maxit[t] or t - pos["entry_i"] >= max_hold:
                if t == n - 1:
                    register_exit(c[t] * (1 - slip), t, "mr_exit")
                else:
                    exit_pending = True
        if pos is None and not exit_pending and 1 <= t and not np.isnan(rsi[t - 1]):
            open_sealed = o[t] >= c[t - 1] * (1 + lim - 0.002)
            ok = allow[t - 1] if allow is not None else True
            if ok and rsi[t - 1] < rsi_in and not open_sealed:
                entry = o[t] * (1 + slip)
                stop = entry * (1 - stop_pct)
                risk = entry - stop
                sh = min(capital * risk_pct / risk, cash / (entry * (1 + comm)))
                if sh > 0:
                    cash -= sh * entry * (1 + comm)
                    shares = sh
                    pos = {"entry_i": t, "entry": entry, "stop": stop,
                           "tp": np.nan, "shares": sh, "risk_amt": sh * risk,
                           "armed": False}
        equity[t] = cash + shares * c[t]
        capital = equity[t]

    if pos:
        register_exit(c[-1] * (1 - slip), n - 1, "eod")
        equity[-1] = cash
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["pnl"] = (tr["exit"] - tr["entry"]) * tr["shares"]
        tr["R"] = tr["pnl"] / tr["risk_amt"]
    return tr, equity

# ---------- 统计 ----------
def trade_stats(tr):
    if not len(tr):
        return dict(trades=0)
    wins = tr[tr["R"] >= 0.1]
    be = tr[(tr["R"] > -0.1) & (tr["R"] < 0.1)]
    losses = tr[tr["pnl"] < 0]
    gp = tr.loc[tr["pnl"] > 0, "pnl"].sum()
    gl = -tr.loc[tr["pnl"] < 0, "pnl"].sum()
    return dict(
        trades=len(tr), wins=len(wins), be=len(be), losses=int((tr['pnl'] < 0).sum()),
        win_rate=len(wins) / max(1, len(wins) + len(losses)),
        pf=gp / gl if gl > 0 else np.inf,
        avg_R=tr["R"].mean(), sum_R=tr["R"].sum())

def fmt_stats(name, s):
    if s.get("trades", 0) == 0:
        return f"{name}: 无交易"
    return (f"{name}: {s['trades']}笔 | 胜{s['wins']} 负{s['losses']} 保本{s['be']} | "
            f"胜率 {s['win_rate']:.1%} | PF {s['pf']:.2f} | 平均R {s['avg_R']:+.3f} | 总R {s['sum_R']:+.0f}")

# ---------- 样本装载 ----------
def load_bundles(syms):
    df = load_stocks(syms)
    bundles = []
    for sym, g in df.groupby("thscode", sort=False):
        a = g["amount"].to_numpy()
        vmed = pd.Series(a).rolling(20).median().shift(1).to_numpy()
        bundles.append((sym, g["date"], g["open"].to_numpy(), g["high"].to_numpy(),
                        g["low"].to_numpy(), g["close"].to_numpy(), a, vmed))
    return df, bundles

def run_pooled(bundles, regime=None, use_reg=False, sec_strong=None, smap=None,
               use_sec=False, **kw):
    all_trades = []
    for sym, dates, o, h, l, c, a, vmed in bundles:
        reg_ok = None
        if use_reg and regime is not None:
            reg_ok = dates.map(regime).fillna(False).to_numpy(dtype=bool)
        sec_ok = None
        if use_sec and sec_strong is not None:
            ind = (smap or {}).get(sym)
            if ind is not None and ind in sec_strong.columns:
                sec_ok = sec_strong[ind].reindex(dates.values).fillna(False).to_numpy(dtype=bool)
        tr, _ = backtest_symbol(o, h, l, c, limit_pct(sym), amt=a, vmed=vmed,
                                reg_ok=reg_ok, sec_ok=sec_ok, **kw)
        if len(tr):
            tr = tr.copy()
            tr["symbol"] = sym
            tr["exit_date"] = pd.to_datetime(dates.to_numpy()[tr["exit_i"].to_numpy()])
            all_trades.append(tr)
    if not all_trades:
        return pd.DataFrame()
    return pd.concat(all_trades, ignore_index=True)

SWEEPS = [
    ("基线 1:1（书版）",      {}),
    ("盈亏比 1.5:1",          {"tp_ratio": 1.5}),
    ("盈亏比 2:1",            {"tp_ratio": 2.0}),
    ("盈亏比 3:1",            {"tp_ratio": 3.0}),
    ("1R后跟踪止盈",          {"trail": True}),
    ("大盘regime过滤",        {"use_reg": True}),
    ("放量>1.5x确认",         {"vol_mult": 1.5}),
    ("regime+放量",           {"use_reg": True, "vol_mult": 1.5}),
    ("regime+2:1",            {"use_reg": True, "tp_ratio": 2.0}),
    ("regime+放量+跟踪",      {"use_reg": True, "vol_mult": 1.5, "trail": True}),
]

SWEEPS_SECTOR = [
    ("板块top20%",            {"use_sec": True}),
    ("板块+放量",             {"use_sec": True, "vol_mult": 1.5}),
    ("板块+跟踪",             {"use_sec": True, "trail": True}),
    ("板块+放量+跟踪",        {"use_sec": True, "vol_mult": 1.5, "trail": True}),
    ("板块+regime+放量+跟踪", {"use_sec": True, "use_reg": True, "vol_mult": 1.5, "trail": True}),
]

def main():
    args = sys.argv[1:]
    kw = {}
    if "--tp" in args:
        kw["tp_ratio"] = float(args[args.index("--tp") + 1])
    if "--trail" in args:
        kw["trail"] = True
    if "--regime" in args:
        kw["use_reg"] = True
    if "--vol" in args:
        kw["vol_mult"] = float(args[args.index("--vol") + 1])

    if "--symbol" in args:
        sym = args[args.index("--symbol") + 1]
        df = load_stocks([sym])
        if not len(df):
            print(f"{sym} 无数据"); return
        g = df
        a = g["amount"].to_numpy()
        vmed = pd.Series(a).rolling(20).median().shift(1).to_numpy()
        reg_ok = None
        if kw.pop("use_reg", False):
            regime = build_regime(load_stocks(liquid_symbols()))
            reg_ok = g["date"].map(regime).fillna(False).to_numpy(dtype=bool)
        tr, eq = backtest_symbol(g["open"].to_numpy(), g["high"].to_numpy(),
                                 g["low"].to_numpy(), g["close"].to_numpy(),
                                 limit_pct(sym), amt=a, vmed=vmed, reg_ok=reg_ok, **kw)
        print(fmt_stats(sym, trade_stats(tr)))
        if eq is not None and len(eq):
            yrs = max(0.5, (g["date"].iloc[-1] - g["date"].iloc[0]).days / 365.25)
            peak = np.maximum.accumulate(eq)
            print(f"  区间 {g['date'].iloc[0].date()}~{g['date'].iloc[-1].date()} | "
                  f"总收益 {eq[-1]/eq[0]-1:+.1%} | 年化 {(eq[-1]/eq[0])**(1/yrs)-1:+.1%} | "
                  f"最大回撤 {np.min(eq/peak-1):.1%} | 买入持有 {g['close'].iloc[-1]/g['close'].iloc[0]-1:+.1%}")
        if len(tr):
            tr.to_csv(os.path.join(HERE, f"ladder_doublek_trades_{sym}.csv"), index=False)
        return

    if "--etf" in args:
        for sym in ["510300.SH", "510500.SH", "159915.SZ"]:
            df = load_etf(sym)
            if df is None:
                print(f"{sym}: 无缓存数据"); continue
            a = df["amount"].to_numpy()
            vmed = pd.Series(a).rolling(20).median().shift(1).to_numpy()
            tr, eq = backtest_symbol(df["open"].to_numpy(), df["high"].to_numpy(),
                                     df["low"].to_numpy(), df["close"].to_numpy(),
                                     limit_pct(sym), amt=a, vmed=vmed)
            print(fmt_stats(sym, trade_stats(tr)))
            if eq is not None and len(eq):
                yrs = max(0.5, (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25)
                peak = np.maximum.accumulate(eq)
                print(f"  区间 {df['date'].iloc[0].date()}~{df['date'].iloc[-1].date()} | "
                      f"总收益 {eq[-1]/eq[0]-1:+.1%} | 年化 {(eq[-1]/eq[0])**(1/yrs)-1:+.1%} | "
                      f"最大回撤 {np.min(eq/peak-1):.1%} | 买入持有 {df['close'].iloc[-1]/df['close'].iloc[0]-1:+.1%}")
        return

    syms = liquid_symbols()
    print(f"样本: {len(syms)} 只（≥600根K线且日成交额中位数≥1亿）")
    df, bundles = load_bundles(syms)
    print(f"载入 {len(df)} 行日线")

    if "--sweep" in args or "--sweep2" in args:
        regime = build_regime(df)
        print(f"regime 多头状态占比: {np.mean(list(regime.values())):.1%}")
        sec_strong, smap = None, None
        if "--sweep2" in args:
            smap = load_sector_map()
            cov = sum(1 for b in bundles if b[0] in smap)
            print(f"行业映射覆盖: {cov}/{len(bundles)}，构建行业等权指数…")
            sec_strong = build_sector_strength(load_all_closes(), smap)
            print(f"强势行业状态占比: {sec_strong.to_numpy().mean():.1%}")
        configs = SWEEPS_SECTOR if "--sweep2" in args else SWEEPS
        results = []
        for name, cfg in configs:
            cfg = dict(cfg)
            use_reg = cfg.pop("use_reg", False)
            use_sec = cfg.pop("use_sec", False)
            T = run_pooled(bundles, regime=regime, use_reg=use_reg,
                           sec_strong=sec_strong, smap=smap, use_sec=use_sec, **cfg)
            s = trade_stats(T)
            results.append({"config": name, **s})
            print(fmt_stats(name, s), flush=True)
        pd.DataFrame(results).to_csv(
            os.path.join(HERE, "ladder_doublek_sweep_sector.csv" if "--sweep2" in args
                         else "ladder_doublek_sweep.csv"), index=False)
        return

    if "--oos" in args or "--switch" in args:
        regime3 = build_regime3(df)
        states = pd.Series(regime3)
        print("大盘三态占比:", states.value_counts(normalize=True).round(3).to_dict())

        if "--oos" in args:
            configs = [("放量+跟踪", {"vol_mult": 1.5, "trail": True}),
                       ("regime+放量+跟踪", {"vol_mult": 1.5, "trail": True, "use_reg": True})]
            for name, cfg in configs:
                cfg = dict(cfg)
                use_reg = cfg.pop("use_reg", False)
                T = run_pooled(bundles, regime=build_regime(df), use_reg=use_reg, **cfg)
                tr = T[T["exit_date"] <= "2022-12-31"]
                te = T[T["exit_date"] > "2022-12-31"]
                print(fmt_stats(f"{name} | 训练段~2022", trade_stats(tr)))
                print(fmt_stats(f"{name} | 盲测段2023~", trade_stats(te)), flush=True)
            return

        # --switch: 趋势系统只做up态；超跌反弹只做非up态
        def state_arr(dates, want_up):
            m = dates.map(regime3)
            return (m == "up").fillna(False).to_numpy(dtype=bool) if want_up \
                else (m != "up").fillna(False).to_numpy(dtype=bool)

        T_trend, T_mr = [], []
        for sym, dates, o, h, l, c, a, vmed in bundles:
            up_arr = state_arr(dates, True)
            tr, _ = backtest_symbol(o, h, l, c, limit_pct(sym), amt=a, vmed=vmed,
                                    reg_ok=up_arr, vol_mult=1.5, trail=True)
            if len(tr):
                tr = tr.copy(); tr["symbol"] = sym
                tr["exit_date"] = pd.to_datetime(dates.to_numpy()[tr["exit_i"].to_numpy()])
                T_trend.append(tr)
            tr2, _ = backtest_mr(o, h, l, c, limit_pct(sym), allow=state_arr(dates, False))
            if len(tr2):
                tr2 = tr2.copy(); tr2["symbol"] = sym
                tr2["exit_date"] = pd.to_datetime(dates.to_numpy()[tr2["exit_i"].to_numpy()])
                T_mr.append(tr2)
        T_trend = pd.concat(T_trend, ignore_index=True) if T_trend else pd.DataFrame()
        T_mr = pd.concat(T_mr, ignore_index=True) if T_mr else pd.DataFrame()
        print(fmt_stats("趋势突破(仅up态)", trade_stats(T_trend)))
        print(fmt_stats("超跌反弹(仅非up态)", trade_stats(T_mr)))
        both = pd.concat([T_trend, T_mr], ignore_index=True)
        print(fmt_stats("切换组合", trade_stats(both)))
        print("\n切换组合按年：")
        both2 = both.copy(); both2["year"] = both2["exit_date"].dt.year
        for y, g in both2.groupby("year"):
            print(" ", y, fmt_stats("", trade_stats(g)), flush=True)
        T_mr.to_csv(os.path.join(HERE, "mr_trades.csv"), index=False)
        T_trend.to_csv(os.path.join(HERE, "trend_up_only_trades.csv"), index=False)
        return

    # 默认：基线全市场
    T = run_pooled(bundles)
    print(fmt_stats("全部样本合并（基线1:1）", trade_stats(T)))
    T.to_csv(os.path.join(HERE, "ladder_doublek_trades_pooled.csv"), index=False)

if __name__ == "__main__":
    main()
