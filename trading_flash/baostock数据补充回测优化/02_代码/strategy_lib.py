#!/usr/bin/env python3
"""恐慌洗盘反转策略：baostock 面板上的组合回测引擎。

冻结规则（与 2026-10-24 项目一致）：
  市场：主板等权指数 20 日收益 < -5%
  个股：250 日收益 > 0，250 日回撤 < -45%，RSI14 < 35，收盘位置 > 0.45，
        20 日平均成交额 >= 1000 万元，上市满 250 个交易日
  排序：250 日回撤最深优先；每日最多 2 只
  执行：信号日 T 收盘后，T+3 开盘买入；+20% 盘中止盈；持有 15 个交易日收盘卖出
  仓位：2 个半仓；开盘涨停/停牌拒买，不递补；同一天卖出款不用于买入

价格口径：
  信号/收益：baostock 后复权（含分红再投资）
  成交/涨跌停：不复权实际价 + 交易所前收盘价（preclose）
  成本：滑点 0.1%/边，佣金 0.025%（最低 5 元），过户费，卖出印花税
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_数据"
PANEL = DATA / "panel.parquet"
BARS = DATA / "bars.parquet"
STOCKS = ROOT / "05_证据" / "主板股票清单.csv"

PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
SIGNAL_COLS = [
    "code", "date", "open", "high", "low", "close", "preclose", "volume", "amount",
    "turn", "turn20", "turn5", "turn_ratio1_20", "amount20", "amount_ratio1_20",
    "is_st", "pe_ttm", "pb_mrq", "listing_session", "return_250d", "drawdown250",
    "rsi14", "close_location", "return_20d", "return_60d", "return_120d",
    "close_ma20", "close_ma60", "close_ma250", "volatility20",
    "market_return20_all", "market_return20_listed", "market_return",
]


def round_half_up(x: float, ndigits: int = 2) -> float:
    if not np.isfinite(x):
        return np.nan
    m = 10 ** ndigits
    return float(np.floor(x * m + 0.5) / m)


@dataclass
class Config:
    start: str = "2017-10-19"
    end: str = "2026-10-23"
    gate: str = "listed"  # listed | all
    universe: str = "listed"  # listed | all
    slots: int = 2
    position_fraction: float = 0.5
    hold: int = 15
    tp: float = 0.20
    stop_loss: float | None = None
    slippage: float = 0.001
    commission: float = 0.00025
    min_commission: float = 5.0
    transfer: float = 0.00002
    stamp: float = 0.001
    stamp_after: tuple[str, float] = ("2023-08-28", 0.0005)
    initial_capital: float = 100_000.0
    cost_flat: float | None = None  # 若设置，则用固定单边成本率（兼容旧口径）
    extra_filter: str = ""  # SQL 片段，作用在信号筛选上
    label: str = "baseline"


@dataclass
class Position:
    s: int
    code: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_i: int
    shares: int
    entry_px: float
    entry_hfq: float
    entry_factor: float
    target_hfq: float
    value: float
    invested: float
    hfq_last: float
    actual_last: float
    last_px: float
    days: int = 0
    pending: str | None = None


@dataclass
class Result:
    equity: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    stats: dict = field(default_factory=dict)


def load_panel(columns=None) -> pd.DataFrame:
    df = pd.read_parquet(PANEL, columns=columns or SIGNAL_COLS)
    df["date"] = pd.to_datetime(df.date)
    return df


def load_stocks() -> pd.DataFrame:
    df = pd.read_csv(STOCKS)
    df["out_date"] = df["out_date"].fillna("").astype(str)
    return df


def load_matrices() -> dict:
    """把 bars 转成 date × symbol 矩阵，供逐日撮合使用。"""
    df = pd.read_parquet(
        BARS,
        columns=["code", "date", "open", "close", "preclose", "volume",
                 "trade_status", "is_st", "hfq_open", "hfq_high", "hfq_low", "hfq_close"],
    )
    df["date"] = pd.to_datetime(df.date)
    dates = np.array(sorted(df.date.unique()), dtype="datetime64[ns]")
    symbols = np.array(sorted(df.code.unique()))
    di = {d: i for i, d in enumerate(dates)}
    si = {s: i for i, s in enumerate(symbols)}
    shape = (len(dates), len(symbols))
    out = {}
    for name, dtype in [("open", np.float32), ("close", np.float32), ("preclose", np.float32),
                        ("volume", np.float32), ("hfq_open", np.float32), ("hfq_high", np.float32),
                        ("hfq_low", np.float32), ("hfq_close", np.float32)]:
        out[name] = np.full(shape, np.nan, dtype=dtype)
    out["is_st"] = np.zeros(shape, dtype=np.int8)
    out["tradable"] = np.zeros(shape, dtype=bool)
    dd = df.date.map(di).to_numpy()
    ss = df.code.map(si).to_numpy()
    for name in ["open", "close", "preclose", "volume", "hfq_open", "hfq_high", "hfq_low", "hfq_close"]:
        out[name][dd, ss] = df[name].to_numpy(dtype=np.float32)
    out["is_st"][dd, ss] = df.is_st.fillna(0).to_numpy(dtype=np.int8)
    out["tradable"][dd, ss] = (df.trade_status.fillna(0).to_numpy() == 1) & (df.volume.fillna(0).to_numpy() > 0)
    out["dates"] = dates
    out["symbols"] = symbols
    out["di"] = di
    out["si"] = si
    return out


def build_signals(panel: pd.DataFrame, stocks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = panel[(panel.date >= cfg.start) & (panel.date <= cfg.end)].copy()
    df = df[df.listing_session >= 250]
    df = df[df.amount20 >= 10_000_000]
    df = df[df.return_250d > 0]
    df = df[df.drawdown250 < -0.45]
    df = df[df.rsi14 < 35]
    df = df[df.close_location > 0.45]
    gate = "market_return20_all" if cfg.gate == "all" else "market_return20_listed"
    df = df[df[gate] < -0.05]
    if cfg.universe == "listed":
        listed = set(stocks.loc[stocks.out_date == "", "code"])
        df = df[df.code.isin(listed)]
    if cfg.extra_filter:
        df = df.query(cfg.extra_filter)
    df = df.sort_values(["date", "drawdown250", "code"], kind="stable").reset_index(drop=True)
    df["rank"] = df.groupby("date").cumcount() + 1
    return df


def _buy_fees(gross: float, cfg: Config) -> tuple[float, float, float]:
    comm = max(cfg.min_commission, gross * cfg.commission)
    transfer = gross * cfg.transfer
    return comm, transfer, comm + transfer


def _sell_fees(gross: float, date: pd.Timestamp, cfg: Config) -> tuple[float, float, float, float]:
    comm = max(cfg.min_commission, gross * cfg.commission)
    transfer = gross * cfg.transfer
    stamp_rate = cfg.stamp
    if date >= pd.Timestamp(cfg.stamp_after[0]):
        stamp_rate = cfg.stamp_after[1]
    stamp = gross * stamp_rate
    return comm, transfer, stamp, comm + transfer + stamp


def run(signals: pd.DataFrame, M: dict, stocks: pd.DataFrame, cfg: Config) -> Result:
    dates = M["dates"]
    symbols = M["symbols"]
    si = M["si"]
    di = M["di"]
    O, C, PC = M["open"], M["close"], M["preclose"]
    HO, HH, HL, HC = M["hfq_open"], M["hfq_high"], M["hfq_low"], M["hfq_close"]
    ST, TR = M["is_st"], M["tradable"]

    sig = signals[signals["rank"] <= 10].copy()
    sigday: dict[int, list[tuple[int, float]]] = {}
    for d, g in sig.groupby("date"):
        if d not in di:
            continue
        rows = [(si[r.code], float(r.drawdown250)) for r in g.sort_values("rank").itertuples()]
        sigday[di[d]] = rows

    i0 = int(np.searchsorted(dates, np.datetime64(cfg.start)))
    i1 = int(np.searchsorted(dates, np.datetime64(cfg.end), side="right")) - 1
    # 只在有信号的区间运行，净值从第一个信号日开始
    signal_indices = sorted(sigday)
    if not signal_indices:
        return Result(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
    run_start = i0
    out_info = stocks.set_index("code")
    out_date = {code: (pd.Timestamp(v) if v else pd.NaT) for code, v in out_info.out_date.items()}
    delisted = set(out_info.index[out_info.out_date != ""])

    cash = float(cfg.initial_capital)
    prev_equity = float(cfg.initial_capital)
    pos: dict[int, Position] = {}
    trades = []
    orders = []
    equity_rows = []

    def sell(p: Position, i: int, reason: str, price_type: str, raw: float, hfq: float, date: pd.Timestamp):
        nonlocal cash
        ratio = hfq / p.hfq_last if p.hfq_last > 0 else 1.0
        value = p.value * ratio
        gross_actual = p.shares * raw if np.isfinite(raw) and raw > 0 else value
        if cfg.cost_flat is not None:
            proceeds = value * (1 - cfg.cost_flat)
            fees = value - proceeds
        else:
            comm, transfer, stamp, fees = _sell_fees(gross_actual, date, cfg)
            proceeds = value - fees
        cash += proceeds
        ret = proceeds / p.invested - 1 if p.invested > 0 else np.nan
        exit_factor = hfq / raw if np.isfinite(raw) and raw > 0 else np.nan
        split_factor = exit_factor / p.entry_factor if p.entry_factor > 0 and np.isfinite(exit_factor) else np.nan
        trades.append(dict(
            code=p.code, signal_date=p.signal_date, entry_date=p.entry_date, exit_date=date,
            entry_px=p.entry_px, exit_px=raw, shares=p.shares, invested=p.invested,
            proceeds=proceeds, ret=ret, exit_reason=reason, holding_sessions=p.days,
            entry_hfq=p.entry_hfq, exit_hfq=hfq, target_hfq=p.target_hfq,
            entry_factor=p.entry_factor, exit_factor=exit_factor, split_factor=split_factor,
            hfq_ret=hfq / p.entry_hfq - 1 if p.entry_hfq > 0 else np.nan,
            actual_px_ret=raw / p.entry_px - 1 if p.entry_px > 0 and np.isfinite(raw) else np.nan,
            drawdown250_signal=float(sig_by_code.get((p.signal_date, p.code), np.nan)),
            is_st_signal=int(st_sig.get((p.signal_date, p.code), 0)),
            delisted=int(p.code in delisted),
        ))
        orders.append(dict(date=date, code=p.code, side="sell", reason=reason,
                           raw_price=raw, hfq_price=hfq, shares=p.shares,
                           value=value, fees=fees, proceeds=proceeds))
        del pos[p.s]

    # 快速查信号日特征
    feat = signals.set_index(["date", "code"])
    sig_by_code = feat["drawdown250"].to_dict()
    st_sig = feat["is_st"].fillna(0).astype(int).to_dict()

    bought_today: set[int] = set()
    sold_today: set[int] = set()

    for i in range(run_start, i1 + 1):
        date = pd.Timestamp(dates[i])
        cash_start = cash
        buy_budget = cash_start
        bought_today.clear()
        sold_today.clear()
        held_at_open = set(pos)

        # 1) 开盘前：待卖单 / 止盈跳空
        exit_now = []
        for s, p in list(pos.items()):
            if not TR[i, s] or not np.isfinite(O[i, s]) or O[i, s] <= 0:
                continue
            raw = float(O[i, s])
            hfq = float(HO[i, s])
            if p.pending is not None:
                lim = _down_limit(PC[i, s], ST[i, s])
                # 只有“正好等于跌停价”才是跌停无法卖出；低于跌停价说明当日无涨跌幅限制
                # （如退市整理期首日），应按开盘价成交。
                if np.isfinite(lim) and abs(raw - lim) <= 0.011:
                    continue
                exit_now.append((s, "pending_" + p.pending, raw, hfq, date))
                continue
            if i > p.entry_i and hfq >= p.target_hfq:
                exit_now.append((s, "take_profit_open", raw, hfq, date))
        for s, reason, raw, hfq, date in exit_now:
            p = pos[s]
            sell(p, i, reason, "open", raw, hfq, date)
            sold_today.add(s)

        # 2) 开盘：按 T+3 信号买入
        sig_i = i - 3
        if sig_i in sigday:
            cand = [(s, rk) for s, rk in sigday[sig_i]
                    if s not in held_at_open and s not in sold_today and s not in bought_today]
            vacancies = max(0, cfg.slots - len(pos))
            for s, rk in cand[:vacancies]:
                if not TR[i, s] or not np.isfinite(O[i, s]) or O[i, s] <= 0:
                    orders.append(dict(date=date, code=symbols[s], side="buy", reason="reject_no_bar",
                                       raw_price=np.nan, hfq_price=np.nan, shares=0, value=0, fees=0, proceeds=0))
                    continue
                raw = float(O[i, s])
                lim = _up_limit(PC[i, s], ST[i, s])
                if not np.isfinite(lim) or raw >= lim - 1e-9:
                    orders.append(dict(date=date, code=symbols[s], side="buy", reason="reject_limit_up",
                                       raw_price=raw, hfq_price=np.nan, shares=0, value=0, fees=0, proceeds=0))
                    continue
                px = raw * (1 + cfg.slippage)
                budget = min(cash, buy_budget, prev_equity * cfg.position_fraction)
                shares = int(np.floor(budget / (px * 100) + 1e-9)) * 100
                while shares > 0:
                    gross = shares * px
                    comm, transfer, fees = _buy_fees(gross, cfg)
                    total = gross + fees
                    if total <= budget + 1e-9 and total <= cash + 1e-9:
                        break
                    shares -= 100
                if shares <= 0:
                    orders.append(dict(date=date, code=symbols[s], side="buy", reason="reject_budget",
                                       raw_price=raw, hfq_price=np.nan, shares=0, value=0, fees=0, proceeds=0))
                    continue
                gross = shares * raw
                invest = shares * px
                comm, transfer, fees = _buy_fees(invest, cfg)
                invested = invest + fees
                cash -= invested
                p = Position(
                    s=s, code=symbols[s], signal_date=pd.Timestamp(dates[sig_i]), entry_date=date,
                    entry_i=i, shares=shares, entry_px=raw, entry_hfq=float(HO[i, s]),
                    entry_factor=float(HO[i, s]) / raw if raw > 0 else np.nan,
                    target_hfq=float(HO[i, s]) * (1 + cfg.tp), value=gross, invested=invested,
                    hfq_last=float(HO[i, s]), actual_last=raw, last_px=raw, days=0,
                )
                pos[s] = p
                bought_today.add(s)
                orders.append(dict(date=date, code=p.code, side="buy", reason="buy",
                                   raw_price=raw, hfq_price=float(HO[i, s]), shares=shares,
                                   value=gross, fees=fees, proceeds=0))

        # 3) 盘中／收盘：估值、止盈、到期、退市
        exit_close = []
        for s, p in list(pos.items()):
            p.days += 1
            if TR[i, s] and np.isfinite(HC[i, s]) and HC[i, s] > 0:
                if p.hfq_last > 0:
                    p.value *= float(HC[i, s]) / p.hfq_last
                p.hfq_last = float(HC[i, s])
                p.actual_last = float(C[i, s])
                p.last_px = float(C[i, s])
            od = out_date.get(p.code, pd.NaT)
            if pd.notna(od) and date > od and not TR[i, s]:
                exit_close.append((s, "delist_last_close", p.last_px, p.hfq_last, date))
                continue
            if p.pending is not None:
                continue
            if (cfg.stop_loss is not None and i > p.entry_i and TR[i, s]
                    and np.isfinite(HL[i, s]) and HL[i, s] <= p.entry_hfq * (1 - cfg.stop_loss)):
                stop_hfq = p.entry_hfq * (1 - cfg.stop_loss)
                if float(HO[i, s]) <= stop_hfq:
                    lim = _down_limit(PC[i, s], ST[i, s])
                    if np.isfinite(lim) and abs(float(O[i, s]) - lim) <= 0.011:
                        p.pending = "stop_limit_down"
                    else:
                        exit_close.append((s, "stop_open", float(O[i, s]), float(HO[i, s]), date))
                else:
                    raw_t = p.entry_px * (stop_hfq / p.entry_hfq)
                    exit_close.append((s, "stop", raw_t, stop_hfq, date))
                continue
            if i > p.entry_i and TR[i, s] and np.isfinite(HH[i, s]) and HH[i, s] >= p.target_hfq:
                raw = float(O[i, s])
                hfq = float(HO[i, s])
                if hfq >= p.target_hfq:
                    exit_close.append((s, "take_profit_intraday_open", raw, hfq, date))
                else:
                    raw_t = p.entry_px * (p.target_hfq / p.entry_hfq)
                    exit_close.append((s, "take_profit_intraday", raw_t, p.target_hfq, date))
                continue
            if p.days >= cfg.hold:
                if TR[i, s] and np.isfinite(C[i, s]) and C[i, s] > 0:
                    lim = _down_limit(PC[i, s], ST[i, s])
                    if np.isfinite(lim) and abs(float(C[i, s]) - lim) <= 0.011:
                        p.pending = "max_holding_limit_down"
                    else:
                        exit_close.append((s, "max_holding", float(C[i, s]), float(HC[i, s]), date))
                else:
                    p.pending = "max_holding_no_bar"
                continue
        for s, reason, raw, hfq, date in exit_close:
            p = pos[s]
            sell(p, i, reason, "close", raw, hfq, date)
            sold_today.add(s)

        # 4) 收盘权益
        eq = cash + sum(p.value for p in pos.values())
        equity_rows.append((date, eq, cash, len(pos)))
        prev_equity = eq

    # 期末仍持仓：按最后价估值，不强制卖出成本
    for s, p in list(pos.items()):
        ret = p.value / p.invested - 1 if p.invested > 0 else np.nan
        trades.append(dict(
            code=p.code, signal_date=p.signal_date, entry_date=p.entry_date,
            exit_date=pd.Timestamp(dates[i1]), entry_px=p.entry_px, exit_px=p.last_px,
            shares=p.shares, invested=p.invested, proceeds=p.value, ret=ret,
            exit_reason="eod", holding_sessions=p.days, entry_hfq=p.entry_hfq,
            exit_hfq=p.hfq_last, target_hfq=p.target_hfq,
            entry_factor=p.entry_factor,
            exit_factor=(p.hfq_last / p.last_px if p.last_px > 0 else np.nan),
            split_factor=(p.hfq_last / p.last_px / p.entry_factor
                          if p.last_px > 0 and p.entry_factor > 0 else np.nan),
            hfq_ret=(p.hfq_last / p.entry_hfq - 1) if p.entry_hfq > 0 else np.nan,
            actual_px_ret=(p.last_px / p.entry_px - 1) if p.entry_px > 0 else np.nan,
            drawdown250_signal=float(sig_by_code.get((p.signal_date, p.code), np.nan)),
            is_st_signal=int(st_sig.get((p.signal_date, p.code), 0)),
            delisted=int(p.code in delisted),
        ))
        del pos[s]

    eqdf = pd.DataFrame(equity_rows, columns=["date", "equity", "cash", "npos"])
    trdf = pd.DataFrame(trades)
    odf = pd.DataFrame(orders)
    return Result(eqdf, trdf, odf, {})


def _up_limit(preclose: float, is_st: int) -> float:
    if not np.isfinite(preclose) or preclose <= 0:
        return np.nan
    rate = 0.05 if is_st else 0.10
    return round_half_up(preclose * (1 + rate))


def _down_limit(preclose: float, is_st: int) -> float:
    if not np.isfinite(preclose) or preclose <= 0:
        return np.nan
    rate = 0.05 if is_st else 0.10
    return round_half_up(preclose * (1 - rate))


def metrics(eq: pd.DataFrame, tr: pd.DataFrame, risk_free: float = 0.0) -> dict:
    if eq is None or len(eq) < 2 or tr is None or len(tr) == 0:
        return {}
    e_raw = eq.set_index("date").equity
    e = e_raw / e_raw.iloc[0]
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = e.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = (e / e.cummax() - 1).min()
    r = e.pct_change().dropna()
    return dict(
        cagr=cagr, max_dd=dd, calmar=cagr / abs(dd) if dd < 0 else np.nan,
        sharpe=(r.mean() - risk_free / 252) / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        n_trades=len(tr), win=(tr.ret > 0).mean(), avg=tr.ret.mean(),
        median=tr.ret.median(), exposure=(eq.npos > 0).mean(),
        final=e.iloc[-1], final_yuan=e_raw.iloc[-1], years=yrs,
        first_date=e.index[0].date(), last_date=e.index[-1].date(),
    )


def annual(eq: pd.DataFrame) -> pd.Series:
    e = eq.set_index("date").equity
    e = e / e.iloc[0]
    a = e.resample("YE").last().pct_change()
    a.iloc[0] = e.resample("YE").last().iloc[0] - 1
    a.index = a.index.year
    return a


def save_result(res: Result, cfg: Config, outdir: Path, name: str) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    m = metrics(res.equity, res.trades)
    res.equity.to_csv(outdir / f"{name}_每日权益.csv", index=False, encoding="utf-8-sig")
    res.trades.to_csv(outdir / f"{name}_逐笔.csv", index=False, encoding="utf-8-sig")
    res.orders.to_csv(outdir / f"{name}_订单.csv", index=False, encoding="utf-8-sig")
    annual(res.equity).to_csv(outdir / f"{name}_年度.csv", encoding="utf-8-sig")
    meta = {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
            for k, v in cfg.__dict__.items()}
    meta["label"] = name
    meta.update({k: (v.item() if isinstance(v, (np.floating, np.integer)) else v)
                 for k, v in m.items()})
    (outdir / f"{name}_指标.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n")
    return m
