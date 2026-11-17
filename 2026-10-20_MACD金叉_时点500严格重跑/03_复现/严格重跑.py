#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACD 全金叉 + 死叉 / b3 的严格重跑。

修复三类已知的“当时不可知”信息：

1. 原 20 只股票池：用 2026-10-13 的终点信息（活到今天、2025-2026 成交额排名、
   当前非 ST）挑出来，再回测 2018 年。本次改为时点中证 500 成分：
   - 195 期月度快照，2009-12-28 至 2026-10-12；
   - 快照日次日生效，信号日只允许买当时最近一期已公布成分；
   - 行情来自 2010-2025 与 2016-2026 两份后复权缓存，并集覆盖全部 1,679 只
     快照成分，包含退市股，避免幸存者偏差。

2. b3 追踪的同日高低顺序前视：
   原代码先拿当天最高价更新峰值，再检查当天最低价是否触发。日线不知道高低
   先后，这等于假设“先高后低”。本次主口径改为：用前一日收盘时已知的峰值
   设定追踪线，再看当天最低价是否触发；当天的新高只用于下一天。
   另保留“同日高触发”和“收盘确认”两个对照，用于量化该问题。

3. 停牌估值：
   原代码在停牌日按入场价估值。本次改为最后已知收盘价；退市/长期无行情按
   最后收盘价强制退出。

其它口径：T 日收盘确认、T+1 开盘成交；涨停不买、跌停顺延；买入 0.025%、
卖出 0.125%；10 槽、单槽 10%；MACD(12,26,9)；金叉排序用当日 hist 降序。

运行：
    python3 严格重跑.py --mode validate   # 只复现原 20 池
    python3 严格重跑.py --mode pit        # 只跑时点 500 / 前20成交额
    python3 严格重跑.py --mode all        # 两者都跑
"""
from __future__ import annotations

import argparse
import bisect
import os
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import duckdb
import numpy as np
import pandas as pd

# ──────────────────────────── 路径与常量 ────────────────────────────
BASE = Path(__file__).resolve().parents[1]
ROOT = Path("/Users/admin/WorkSpace/trading")
DATA_DIR = ROOT / "2026-10-18_成交量融合_2010-2025" / "01_数据"
CACHE_A = DATA_DIR / "daily_cache_2010_2025_hfq.pkl"      # 1,648 只，含退市，至 2025-12-31
CACHE_B = DATA_DIR / "daily_cache_2016_2026.pkl"          # 1,284 只，至 2026-10-13
SNAP_FILE = DATA_DIR / "时点成分_2010_2026.pkl"           # 195 期时点中证500
POOL_CSV = ROOT / "analysis" / "macd" / "MACD研究股票名单.csv"
DB_PATH = Path(os.path.expanduser(
    "~/Library/Application Support/hithink-finance/data/market.duckdb"
))
OUT = BASE / "02_结果"
OUT.mkdir(parents=True, exist_ok=True)
TWO_OUT = OUT / "2仓对照"
ONE_OUT = OUT / "1仓与零轴"

START = pd.Timestamp("2018-07-01")
END = pd.Timestamp("2026-10-13")
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
FLAT_WEIGHT = 0.10
COST_BUY = 0.00025
COST_SELL = 0.00125
MIN_WARMUP = 150
TRAIL_TRIGGER = 0.10
TRAIL_STOP = 0.08


# ──────────────────────────── 指标与数据 ────────────────────────────
def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    c = out["close"].astype(float)
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    out["dif"] = dif
    out["dea"] = dea
    out["hist"] = dif - dea
    out["golden_cross"] = ((dif > dea) & (dif.shift(1) <= dea.shift(1))).astype(int)
    out["death_cross"] = ((dif < dea) & (dif.shift(1) >= dea.shift(1))).astype(int)
    return out


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


def load_original_pool() -> dict[str, pd.DataFrame]:
    """原 20 池：用原报告相同的 DuckDB 前复权数据，2017-01-01 起。"""
    codes = pd.read_csv(POOL_CSV)["thscode"].tolist()
    sql = """
        SELECT thscode, date, open, high, low, close, volume, amount
        FROM v_daily_qfq
        WHERE thscode IN (SELECT UNNEST(?::VARCHAR[]))
          AND date >= ? AND date <= ?
        ORDER BY thscode, date
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        raw = con.execute(sql, [codes, "2017-01-01", END.strftime("%Y-%m-%d")]).fetchdf()
    raw["date"] = pd.to_datetime(raw["date"])
    data = {}
    for code, g in raw.groupby("thscode"):
        g = _clean_ohlcv(g)
        if len(g) < MIN_WARMUP + 5:
            continue
        data[code] = add_macd(g)
    return data


def load_pit_data() -> dict[str, pd.DataFrame]:
    """时点 500 行情：2010-2025 含退市 + 2016-2026 尾部，后复权。"""
    with open(CACHE_A, "rb") as f:
        a = pickle.load(f)
    with open(CACHE_B, "rb") as f:
        b = pickle.load(f)
    all_codes = set(a["stocks"]) | set(b["stocks"])
    data = {}
    for code in all_codes:
        frames = []
        if code in a["stocks"]:
            da = a["stocks"][code].copy()
            da["date"] = pd.to_datetime(da["date"])
            frames.append(da[da["date"] <= pd.Timestamp("2025-12-31")])
        if code in b["stocks"]:
            db = b["stocks"][code].copy()
            db["date"] = pd.to_datetime(db["date"])
            if code in a["stocks"]:
                db = db[db["date"] > pd.Timestamp("2025-12-31")]
            frames.append(db)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df = _clean_ohlcv(df)
        if len(df) < MIN_WARMUP + 5:
            continue
        data[code] = add_macd(df)
    return data


def load_snapshots() -> dict[pd.Timestamp, set]:
    with open(SNAP_FILE, "rb") as f:
        obj = pickle.load(f)
    return {pd.Timestamp(k): set(v) for k, v in obj["snapshots"].items()}


def load_benchmark() -> pd.DataFrame:
    with open(CACHE_A, "rb") as f:
        a = pickle.load(f)
    with open(CACHE_B, "rb") as f:
        b = pickle.load(f)
    za, zb = a["zz500"].copy(), b["zz500"].copy()
    for z in (za, zb):
        z["date"] = pd.to_datetime(z["date"])
    z = pd.concat([
        za[za["date"] <= pd.Timestamp("2025-12-31")],
        zb[zb["date"] > pd.Timestamp("2025-12-31")],
    ], ignore_index=True)
    z = z.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    z = z[(z["date"] >= START) & (z["date"] <= END)].reset_index(drop=True)
    return z


# ──────────────────────────── 时点股票池 ────────────────────────────
class FixedUniverse:
    """固定股票池：只在开始时点选一次，之后不换池。"""

    def __init__(self, codes: set):
        self.codes = set(codes)

    def snapshot_for(self, date: pd.Timestamp) -> set:
        return self.codes


class Universe:
    """按最近一期已公布快照返回当时成分；可选“过去250日成交额前N”。"""

    def __init__(self, snapshots: dict[pd.Timestamp, set],
                 data: Optional[dict[str, pd.DataFrame]] = None,
                 topn: Optional[int] = None):
        self.snapshots = snapshots
        self.dates = sorted(snapshots)
        self.data = data
        self.topn = topn
        self.cache: dict[pd.Timestamp, set] = {}

    def snapshot_for(self, date: pd.Timestamp) -> set:
        i = bisect.bisect_left(self.dates, date) - 1
        if i < 0:
            return set()
        snap_date = self.dates[i]
        if snap_date not in self.cache:
            members = self.snapshots[snap_date]
            if self.topn is not None:
                members = self._top_by_amount(snap_date, members)
            self.cache[snap_date] = set(members)
        return self.cache[snap_date]

    def _top_by_amount(self, snap_date: pd.Timestamp, members: set) -> set:
        rows = []
        for code in members:
            df = self.data.get(code) if self.data else None
            if df is None:
                continue
            sub = df[df["date"] <= snap_date]
            if len(sub) < 250:
                continue
            amt = float(sub["amount"].tail(250).mean())
            if np.isfinite(amt):
                rows.append((code, amt))
        rows.sort(key=lambda x: (-x[1], x[0]))
        return {c for c, _ in rows[: self.topn]}


# ──────────────────────────── 交易与出场 ────────────────────────────
def limit_pct(code: str, date: pd.Timestamp) -> float:
    """按代码与日期返回涨跌停幅度；ST 未单独处理（见报告限制）。"""
    digits = "".join(ch for ch in code if ch.isdigit())[:6]
    prefix = digits[:3]
    if code.lower().endswith(".bj") or code.lower().startswith("bj."):
        return 0.30
    if prefix == "688":
        return 0.20
    if prefix in ("300", "301"):
        return 0.20 if date >= pd.Timestamp("2020-08-24") else 0.10
    return 0.10


@dataclass
class Pos:
    entry_id: int
    code: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    weight: float
    score: float
    total_invested: float
    highest: float
    peak_close: float
    hold_days: int = 0
    pending_exit: Optional[str] = None
    pending_reason: str = ""


def check_exit(pos: Pos, df: pd.DataFrame, idx: int, strategy: str,
               trail_trigger: float = TRAIL_TRIGGER,
               trail_stop: float = TRAIL_STOP) -> Optional[str]:
    """返回离场原因；None 表示继续持有。"""
    death = int(df["death_cross"].iat[idx]) == 1
    hi = float(df["high"].iat[idx])
    lo = float(df["low"].iat[idx])
    cl = float(df["close"].iat[idx])

    if strategy == "death_cross":
        return "death_cross" if death else None

    if strategy == "b3_same_day":
        # 原口径：先拿当天最高价更新峰值，再检查当天最低价。
        pos.highest = max(pos.highest, hi)
        if pos.highest / pos.entry_price - 1.0 >= trail_trigger:
            if lo <= pos.highest * (1.0 - trail_stop):
                return "b3_trailing"
        return "death_cross" if death else None

    if strategy == "b3_prior_peak":
        # 严格口径：追踪线只使用前一日收盘时已知的峰值，当天新高留到明天。
        if pos.highest / pos.entry_price - 1.0 >= trail_trigger:
            if lo <= pos.highest * (1.0 - trail_stop):
                return "b3_trailing"
        return "death_cross" if death else None

    if strategy == "b3_close_only":
        # 更保守：峰值和触发都用收盘价，完全不需要日内高低顺序。
        if pos.peak_close / pos.entry_price - 1.0 >= trail_trigger:
            if cl <= pos.peak_close * (1.0 - trail_stop):
                return "b3_trailing"
        return "death_cross" if death else None

    raise ValueError(f"未知出场策略: {strategy}")


# ──────────────────────────── 回测引擎 ────────────────────────────
def run_backtest(data: dict[str, pd.DataFrame], dates: list[pd.Timestamp],
                 strategy: str, universe: Optional[Universe] = None,
                 cost_buy: float = COST_BUY, cost_sell: float = COST_SELL,
                 max_positions: int = MAX_POSITIONS,
                 flat_weight: float = FLAT_WEIGHT,
                 zero_axis: bool = False,
                 entry_filter: Optional[Callable[[str, pd.DataFrame, int, pd.Timestamp], bool]] = None,
                 hard_stop: Optional[float] = None,
                 time_stop: Optional[int] = None,
                 trail_trigger: float = TRAIL_TRIGGER,
                 trail_stop: float = TRAIL_STOP,
                 label: str = "",
                 score_fn: Optional[Callable[[str, pd.DataFrame, int, pd.Timestamp], float]] = None,
                 exit_filter: Optional[Callable[[Pos, pd.DataFrame, int], Optional[str]]] = None) -> dict:
    """通用回测。

    entry_filter: 信号日收盘后调用，返回 False 则跳过该信号；只能使用
                  (code, df, idx, date) 中当时已知的数据。
    hard_stop:    收盘价跌破 entry_price*(1-hard_stop) 时，次日开盘卖出。
    time_stop:    持有交易日达到该数时，次日开盘卖出（未触发其它出场时）。
    """
    capital = INITIAL_CAPITAL
    base_equity = INITIAL_CAPITAL
    positions: dict[str, Pos] = {}
    pending: dict[str, tuple[float, float, pd.Timestamp]] = {}
    last_close: dict[str, float] = {}
    date_maps = {c: {d: i for i, d in enumerate(df["date"].tolist())}
                 for c, df in data.items()}
    last_date = {c: df["date"].iloc[-1] for c, df in data.items()}

    equity_records: list[dict] = []
    trades: list[dict] = []
    entries: list[dict] = []
    reasons: Counter = Counter()
    stats: Counter = Counter()
    entry_id = 0

    def close_position(pos: Pos, date: pd.Timestamp, px: float, reason: str,
                       price_date: pd.Timestamp) -> None:
        nonlocal capital
        proceeds = pos.shares * px
        fee = proceeds * cost_sell
        net = proceeds - fee
        capital += net
        pnl = net - pos.total_invested
        trades.append({
            "entry_id": pos.entry_id,
            "code": pos.code,
            "signal_date": pos.signal_date,
            "entry_date": pos.entry_date,
            "entry_price": pos.entry_price,
            "exit_date": date,
            "exit_price": px,
            "exit_price_date": price_date,
            "shares": pos.shares,
            "total_invested": pos.total_invested,
            "net_proceeds": net,
            "total_pnl": pnl,
            "total_ret": pnl / pos.total_invested if pos.total_invested > 0 else 0.0,
            "hold_days": pos.hold_days,
            "reason": reason,
            "peak_profit_pct": pos.highest / pos.entry_price - 1.0,
        })
        reasons[reason] += 1

    for di, date in enumerate(dates):
        # 1) 执行昨日挂起卖出（今日开盘）；跌停顺延，退市按最后收盘。
        for code in list(positions):
            pos = positions[code]
            if pos.pending_exit is None:
                continue
            df = data.get(code)
            idx = date_maps[code].get(date) if df is not None else None
            if idx is None:
                if last_date[code] < date:
                    px = last_close.get(code, pos.entry_price)
                    close_position(pos, date, px, "退市/长期停牌", last_date[code])
                    del positions[code]
                    stats["退市强制卖出"] += 1
                continue
            prev_close = float(df["close"].iat[idx - 1]) if idx > 0 else float(df["open"].iat[idx])
            today_open = float(df["open"].iat[idx])
            lim = limit_pct(code, date)
            if today_open <= prev_close * (1.0 - lim) * 1.001:
                stats["卖出跌停顺延"] += 1
                continue
            close_position(pos, date, today_open, pos.pending_reason, date)
            del positions[code]

        # 2) 执行昨日挂起买入（今日开盘）；涨停跳过。
        for code in list(pending):
            if len(positions) >= max_positions:
                pending.clear()
                break
            if code in positions:
                del pending[code]
                continue
            weight, score, signal_date = pending[code]
            if universe is not None and code not in universe.snapshot_for(date):
                del pending[code]
                stats["买入时已非成分"] += 1
                continue
            df = data.get(code)
            idx = date_maps[code].get(date) if df is not None else None
            if idx is None:
                del pending[code]
                stats["买入日无行情"] += 1
                continue
            prev_close = float(df["close"].iat[idx - 1]) if idx > 0 else float(df["open"].iat[idx])
            today_open = float(df["open"].iat[idx])
            lim = limit_pct(code, date)
            if today_open >= prev_close * (1.0 + lim) * 0.999:
                del pending[code]
                stats["买入涨停跳过"] += 1
                continue
            alloc = min(base_equity * weight, capital)
            if alloc < 100:
                del pending[code]
                stats["买入资金不足"] += 1
                continue
            fee = alloc * cost_buy
            shares = (alloc - fee) / today_open
            if shares <= 0:
                del pending[code]
                continue
            capital -= alloc
            entry_id += 1
            positions[code] = Pos(
                entry_id=entry_id, code=code, signal_date=signal_date,
                entry_date=date, entry_price=today_open, shares=shares,
                weight=weight, score=score, total_invested=alloc,
                highest=today_open, peak_close=today_open,
            )
            entries.append({
                "entry_id": entry_id, "code": code, "signal_date": signal_date,
                "entry_date": date, "entry_price": today_open, "shares": shares,
                "capital_alloc": alloc, "weight": weight, "score": score,
            })
            del pending[code]
            stats["买入成交"] += 1

        # 3) 今日收盘：更新持仓并检查出场。
        for code, pos in list(positions.items()):
            df = data.get(code)
            idx = date_maps[code].get(date) if df is not None else None
            if idx is None:
                if last_date[code] < date:
                    px = last_close.get(code, pos.entry_price)
                    close_position(pos, date, px, "退市/长期停牌", last_date[code])
                    del positions[code]
                    stats["退市强制卖出"] += 1
                continue
            pos.hold_days += 1
            hi = float(df["high"].iat[idx])
            lo = float(df["low"].iat[idx])
            cl = float(df["close"].iat[idx])
            last_close[code] = cl
            if pos.pending_exit is None and hard_stop is not None:
                if cl <= pos.entry_price * (1.0 - hard_stop):
                    pos.pending_exit = "full"
                    pos.pending_reason = "hard_stop"
                    stats["硬止损挂起"] += 1
            if pos.pending_exit is None and time_stop is not None:
                if pos.hold_days >= time_stop:
                    pos.pending_exit = "full"
                    pos.pending_reason = "time_stop"
                    stats["时间止损挂起"] += 1
            if pos.pending_exit is None and exit_filter is not None:
                reason = exit_filter(pos, df, idx)
                if reason is not None:
                    pos.pending_exit = "full"
                    pos.pending_reason = reason
                    stats["提前离场挂起"] += 1
            if pos.pending_exit is None:
                reason = check_exit(pos, df, idx, strategy,
                                    trail_trigger=trail_trigger,
                                    trail_stop=trail_stop)
                if reason is not None:
                    pos.pending_exit = "full"
                    pos.pending_reason = reason
            if strategy != "b3_same_day":
                pos.highest = max(pos.highest, hi)
            pos.peak_close = max(pos.peak_close, cl)

        # 4) 按今日收盘估值；停牌用最后已知收盘，不用入场价。
        market_value = 0.0
        for code, pos in positions.items():
            px = last_close.get(code, pos.entry_price)
            market_value += pos.shares * px
        equity = capital + market_value
        base_equity = equity
        equity_records.append({
            "date": date,
            "equity": equity,
            "n_positions": len(positions),
            "gross_market_value": market_value,
            "market_value_ratio": market_value / equity if equity > 0 else 0.0,
        })

        # 5) 今日收盘生成信号，挂到下一交易日开盘。
        if di + 1 < len(dates):
            free_slots = max_positions - len(positions) - len(pending)
            if free_slots > 0:
                members = universe.snapshot_for(date) if universe is not None else set(data)
                candidates = []
                for code in members:
                    if code in positions or code in pending:
                        continue
                    df = data.get(code)
                    idx = date_maps[code].get(date) if df is not None else None
                    if idx is None or idx < MIN_WARMUP:
                        continue
                    if int(df["golden_cross"].iat[idx]) == 1:
                        if zero_axis and not (
                                float(df["dif"].iat[idx]) > 0
                                and float(df["dea"].iat[idx]) > 0):
                            continue
                        if entry_filter is not None and not entry_filter(
                                code, df, idx, date):
                            continue
                        sc = (float(score_fn(code, df, idx, date))
                              if score_fn is not None
                              else float(df["hist"].iat[idx]))
                        if not np.isfinite(sc):
                            continue
                        candidates.append((code, sc))
                candidates.sort(key=lambda x: (-x[1], x[0]))
                total_w = (sum(p.weight for p in positions.values())
                           + sum(v[0] for v in pending.values()))
                added = 0
                for code, score in candidates:
                    if added >= free_slots:
                        break
                    if total_w + flat_weight > 1.0 + 1e-9:
                        continue
                    pending[code] = (flat_weight, score, date)
                    total_w += flat_weight
                    added += 1
                    stats["挂起买单"] += 1

    n_open = len(positions)
    open_positions = [{
        "code": p.code, "entry_date": p.entry_date, "entry_price": p.entry_price,
        "shares": p.shares, "last_close": last_close.get(c, p.entry_price),
        "unrealized_ret": last_close.get(c, p.entry_price) / p.entry_price - 1.0,
        "pending_exit": p.pending_exit, "pending_reason": p.pending_reason,
    } for c, p in positions.items()]

    eq_df = pd.DataFrame(equity_records)
    tr_df = pd.DataFrame(trades)
    en_df = pd.DataFrame(entries)
    result = {
        "label": label,
        "strategy": strategy,
        "equity_curve": eq_df,
        "trades": tr_df,
        "entries": en_df,
        "open_positions": pd.DataFrame(open_positions),
        "reasons": dict(reasons),
        "stats": dict(stats),
        "n_open": n_open,
        "final_equity": float(eq_df["equity"].iloc[-1]) if len(eq_df) else INITIAL_CAPITAL,
    }
    result["metrics"] = full_metrics(result)
    return result


# ──────────────────────────── 绩效指标 ────────────────────────────
def equity_metrics(eq_df: pd.DataFrame) -> dict:
    if eq_df.empty or len(eq_df) < 2:
        return dict(ann_return=0.0, mdd=0.0, calmar=0.0, sharpe=0.0,
                    total_return=0.0, avg_positions=0.0, avg_exposure=0.0)
    eq = eq_df["equity"].values.astype(float)
    dates = pd.to_datetime(eq_df["date"])
    total_ret = eq[-1] / eq[0] - 1.0
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 0.01)
    ann = (1.0 + total_ret) ** (1.0 / years) - 1.0
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    calmar = ann / abs(mdd) if mdd != 0 else 0.0
    dr = np.diff(eq) / eq[:-1]
    std = float(np.std(dr)) if len(dr) else 0.0
    sharpe = (float(np.mean(dr)) * 250 - 0.02) / (std * np.sqrt(250)) if std > 0 else 0.0
    return dict(
        ann_return=round(ann * 100, 2), mdd=round(mdd * 100, 2),
        calmar=round(calmar, 3), sharpe=round(sharpe, 3),
        total_return=round(total_ret * 100, 2),
        avg_positions=round(float(eq_df["n_positions"].mean()), 3),
        avg_exposure=round(float(eq_df["market_value_ratio"].mean()) * 100, 3),
    )


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return dict(n_trades=0, win_rate=0.0, pf=0.0, pf_pnl=0.0,
                    avg_hold_days=0.0, avg_ret=0.0)
    n = len(trades)
    wins = trades[trades["total_ret"] > 0]
    losses = trades[trades["total_ret"] <= 0]
    gp = float(wins["total_ret"].sum()) if len(wins) else 0.0
    gl = float(abs(losses["total_ret"].sum())) if len(losses) else 1e-12
    gp_pnl = float(wins["total_pnl"].sum()) if len(wins) else 0.0
    gl_pnl = float(abs(losses["total_pnl"].sum())) if len(losses) else 1e-12
    return dict(
        n_trades=n,
        win_rate=round(len(wins) / n * 100, 2),
        pf=round(gp / gl, 3),
        pf_pnl=round(gp_pnl / gl_pnl, 3),
        avg_hold_days=round(float(trades["hold_days"].mean()), 2),
        avg_ret=round(float(trades["total_ret"].mean()) * 100, 3),
    )


def full_metrics(result: dict) -> dict:
    m = equity_metrics(result["equity_curve"])
    m.update(trade_metrics(result["trades"]))
    m["n_open"] = result["n_open"]
    m["final_equity"] = round(result["final_equity"], 2)
    return m


def benchmark_metrics(bench: pd.DataFrame) -> dict:
    b = bench.set_index("date")["close"].astype(float)
    total = b.iloc[-1] / b.iloc[0] - 1.0
    years = max((b.index[-1] - b.index[0]).days / 365.25, 0.01)
    ann = (1.0 + total) ** (1.0 / years) - 1.0
    dd = (b / b.cummax() - 1.0).min()
    return dict(total_return=round(total * 100, 2), ann_return=round(ann * 100, 2),
                mdd=round(float(dd) * 100, 2))


def same_exposure_benchmark(eq_df: pd.DataFrame, bench: pd.DataFrame) -> dict:
    b = bench.set_index("date")["close"].astype(float).reindex(
        pd.to_datetime(eq_df["date"])).ffill()
    ret = b.pct_change().fillna(0.0)
    exposure = eq_df["market_value_ratio"].shift(1).fillna(0.0)
    curve = (1.0 + exposure.values * ret.values).cumprod()
    total = curve[-1] - 1.0
    years = max((pd.to_datetime(eq_df["date"]).iloc[-1]
                 - pd.to_datetime(eq_df["date"]).iloc[0]).days / 365.25, 0.01)
    ann = (1.0 + total) ** (1.0 / years) - 1.0
    dd = float((curve / np.maximum.accumulate(curve) - 1.0).min())
    return dict(same_total=round(total * 100, 2),
                same_ann=round(ann * 100, 2),
                same_mdd=round(dd * 100, 2))


def yearly_table(result: dict, bench: pd.DataFrame) -> pd.DataFrame:
    eq = result["equity_curve"].copy()
    eq["year"] = pd.to_datetime(eq["date"]).dt.year
    b = bench.copy()
    b["year"] = pd.to_datetime(b["date"]).dt.year
    rows = []
    for year, g in eq.groupby("year"):
        g = g.sort_values("date")
        ret = g["equity"].iloc[-1] / g["equity"].iloc[0] - 1.0
        curve = g["equity"].values
        mdd = float((curve / np.maximum.accumulate(curve) - 1.0).min())
        # 基准与策略取同一批日期，避免 2018 年“策略从 7 月开始、基准从 1 月开始”。
        bg = b[(b["year"] == year)
               & (b["date"] >= g["date"].iloc[0])
               & (b["date"] <= g["date"].iloc[-1])]
        bret = (bg["close"].iloc[-1] / bg["close"].iloc[0] - 1.0
                if len(bg) > 1 else np.nan)
        rows.append({
            "年份": int(year), "策略收益": round(ret * 100, 2),
            "策略最大回撤": round(mdd * 100, 2),
            "中证500": round(float(bret) * 100, 2) if np.isfinite(bret) else np.nan,
            "交易笔数": int(((pd.to_datetime(result["trades"]["exit_date"]).dt.year == year).sum())
                            if not result["trades"].empty else 0),
        })
    return pd.DataFrame(rows)


# ──────────────────────────── 输出 ────────────────────────────
def save_result(result: dict, prefix: str, out_dir: Path = OUT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eq = result["equity_curve"].copy().rename(columns={
        "date": "日期", "equity": "权益", "n_positions": "持仓数",
        "gross_market_value": "持仓市值", "market_value_ratio": "仓位比例",
    })
    eq.to_csv(out_dir / f"权益_{prefix}.csv", index=False, encoding="utf-8-sig")

    tr = result["trades"].copy()
    if not tr.empty:
        tr = tr.rename(columns={
            "entry_id": "编号", "code": "代码", "signal_date": "信号日",
            "entry_date": "买入日", "entry_price": "买入价",
            "exit_date": "卖出日", "exit_price": "卖出价",
            "exit_price_date": "计价日", "shares": "股数",
            "total_invested": "投入资金", "net_proceeds": "净回款",
            "total_pnl": "净盈亏", "total_ret": "净收益率",
            "hold_days": "持有交易日", "reason": "离场原因",
            "peak_profit_pct": "峰值浮盈",
        })
        tr.to_csv(out_dir / f"交易_{prefix}.csv", index=False, encoding="utf-8-sig")

    op = result["open_positions"].copy()
    if not op.empty:
        op = op.rename(columns={
            "code": "代码", "entry_date": "买入日", "entry_price": "买入价",
            "shares": "股数", "last_close": "最后收盘", "unrealized_ret": "浮动收益率",
            "pending_exit": "待卖", "pending_reason": "待卖原因",
        })
        op.to_csv(out_dir / f"未平仓_{prefix}.csv", index=False, encoding="utf-8-sig")


def summary_row(result: dict, bench: pd.DataFrame, name: str, pool: str,
                note: str = "") -> dict:
    m = result["metrics"]
    se = same_exposure_benchmark(result["equity_curve"], bench)
    stats = result.get("stats", {})
    return {
        "策略": name, "股票池": pool, "区间": f"{START.date()}~{END.date()}",
        "累计收益": m["total_return"], "年化": m["ann_return"],
        "最大回撤": m["mdd"], "Calmar": m["calmar"], "Sharpe": m["sharpe"],
        "交易笔数": m["n_trades"], "胜率": m["win_rate"],
        "盈亏比": m["pf"], "盈亏比_金额": m["pf_pnl"],
        "平均持仓数": m["avg_positions"], "平均仓位": m["avg_exposure"],
        "期末权益": m["final_equity"], "未平仓": m["n_open"],
        "买入成交": stats.get("买入成交", 0),
        "买入涨停跳过": stats.get("买入涨停跳过", 0),
        "买入时已非成分": stats.get("买入时已非成分", 0),
        "买入日无行情": stats.get("买入日无行情", 0),
        "退市强制卖出": stats.get("退市强制卖出", 0),
        "同暴露中证500": se["same_total"],
        "同暴露年化": se["same_ann"], "同暴露回撤": se["same_mdd"],
        "备注": note,
    }


def print_result(result: dict, name: str) -> None:
    m = result["metrics"]
    print(f"{name:<30} 交易={m['n_trades']:>5} 胜率={m['win_rate']:>6.2f}% "
          f"PF={m['pf']:>5.3f} 年化={m['ann_return']:>7.2f}% "
          f"MDD={m['mdd']:>7.2f}% Calmar={m['calmar']:>6.3f} "
          f"Sharpe={m['sharpe']:>6.3f} 仓位={m['avg_exposure']:>6.2f}% "
          f"未平仓={m['n_open']}")


# ──────────────────────────── 主流程 ────────────────────────────
def run_validate(bench: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    print("\n" + "=" * 88)
    print("一、原 20 池复现校验（同数据、同区间、同成本，只换新引擎）")
    print("=" * 88)
    data = load_original_pool()
    all_dates = sorted({d for df in data.values() for d in df["date"].tolist()})
    dates = [d for d in all_dates if START <= d <= END]
    print(f"有效股票 {len(data)} 只，交易日 {len(dates)} 天，"
          f"{dates[0].date()} ~ {dates[-1].date()}")

    specs = [
        ("death_cross", "原20池_死叉", "原报告口径"),
        ("b3_same_day", "原20池_b3_同日高触发", "原代码口径，日内先高后低"),
        ("b3_prior_peak", "原20池_b3_前日峰值修复", "严格口径"),
        ("b3_close_only", "原20池_b3_收盘确认", "更保守对照"),
    ]
    rows, yearly_frames, results = [], [], []
    for strategy, name, note in specs:
        r = run_backtest(data, dates, strategy, universe=None, label=name)
        print_result(r, name)
        rows.append(summary_row(r, bench, name, "原20只事后龙头", note))
        yearly_frames.append(yearly_table(r, bench).assign(策略=name))
        results.append(r)
        save_result(r, name)
    return pd.DataFrame(rows), pd.concat(yearly_frames, ignore_index=True), results


def run_pit(bench: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    print("\n" + "=" * 88)
    print("二、时点中证 500 重跑（195 期快照，含退市股）")
    print("=" * 88)
    data = load_pit_data()
    snaps = load_snapshots()
    dates = pd.to_datetime(bench["date"]).tolist()
    print(f"行情股票 {len(data)} 只，时点快照 {len(snaps)} 期，"
          f"交易日 {len(dates)} 天，{dates[0].date()} ~ {dates[-1].date()}")

    uni500 = Universe(snaps)
    uni20 = Universe(snaps, data=data, topn=20)

    # 覆盖检查
    all_members = set().union(*snaps.values())
    covered = len(all_members & set(data))
    print(f"时点成分并集 {len(all_members)} 只，有行情 {covered} 只，"
          f"缺失 {len(all_members - set(data))} 只")

    rows, yearly_frames, results = [], [], []
    specs = [
        ("death_cross", "时点500_死叉", uni500, "全部时点500成分"),
        ("b3_prior_peak", "时点500_b3_前日峰值", uni500, "主口径"),
        ("b3_same_day", "时点500_b3_同日高触发", uni500, "原代码口径对照"),
        ("b3_close_only", "时点500_b3_收盘确认", uni500, "保守对照"),
        ("death_cross", "前20成交额_死叉", uni20, "每月按过去250日成交额取前20"),
        ("b3_prior_peak", "前20成交额_b3_前日峰值", uni20, "同上，b3严格口径"),
    ]
    for strategy, name, uni, note in specs:
        r = run_backtest(data, dates, strategy, universe=uni, label=name)
        print_result(r, name)
        yearly = yearly_table(r, bench).assign(策略=name)
        yearly_frames.append(yearly)
        r["yearly"] = yearly
        rows.append(summary_row(r, bench, name, note.split("，")[0], note))
        results.append(r)
        save_result(r, name)

    # 固定时点20：只在 2018-06 选一次，之后不换池；最接近原“固定 20 只”的形态。
    snap_dates = sorted(snaps)
    base_i = bisect.bisect_left(snap_dates, START) - 1
    if base_i < 0:
        raise ValueError("START 之前没有可用快照，固定时点20无法在无前视的前提下选池")
    base_date = snap_dates[base_i]
    fixed_codes = uni20._top_by_amount(base_date, snaps[base_date])
    print(f"\n固定时点20（基准快照 {base_date.date()}，过去250日成交额前20，"
          f"选出 {len(fixed_codes)} 只，之后不换池）")
    fixed_uni = FixedUniverse(fixed_codes)
    for strategy, name in [("death_cross", "时点固定20_死叉"),
                           ("b3_prior_peak", "时点固定20_b3_前日峰值")]:
        r = run_backtest(data, dates, strategy, universe=fixed_uni, label=name)
        print_result(r, name)
        yearly = yearly_table(r, bench).assign(策略=name)
        yearly_frames.append(yearly)
        r["yearly"] = yearly
        rows.append(summary_row(
            r, bench, name, f"固定20（{base_date.date()}时点选池）",
            "2018-06 时点选一次，之后不换池"))
        results.append(r)
        save_result(r, name)

    # 成本压力：买卖各加 0.1 个百分点滑点
    print("\n成本压力（买入 0.125%、卖出 0.225%，往返约 0.35%）")
    for strategy, name in [("death_cross", "时点500_死叉_含滑点"),
                           ("b3_prior_peak", "时点500_b3_前日峰值_含滑点")]:
        r = run_backtest(data, dates, strategy, universe=uni500,
                         cost_buy=0.00125, cost_sell=0.00225, label=name)
        print_result(r, name)
        yearly = yearly_table(r, bench).assign(策略=name)
        yearly_frames.append(yearly)
        r["yearly"] = yearly
        rows.append(summary_row(r, bench, name, "全部时点500成分",
                                "成本压力：买卖各加0.1pp滑点"))
        results.append(r)
        save_result(r, name)

    return (pd.DataFrame(rows), pd.concat(yearly_frames, ignore_index=True), results)


def run_two_slot(bench: pd.DataFrame) -> None:
    """2 槽 × 单槽 50% 的集中组合对照（目标满仓，和 10 槽 ×10% 同口径）。"""
    print("\n" + "=" * 88)
    print("三、2 槽对照（单槽 50%，目标满仓）")
    print("=" * 88)
    TWO_OUT.mkdir(parents=True, exist_ok=True)

    orig_data = load_original_pool()
    orig_dates = sorted({d for df in orig_data.values() for d in df["date"].tolist()})
    orig_dates = [d for d in orig_dates if START <= d <= END]

    pit_data = load_pit_data()
    snaps = load_snapshots()
    pit_dates = pd.to_datetime(bench["date"]).tolist()
    uni500 = Universe(snaps)
    uni20 = Universe(snaps, data=pit_data, topn=20)
    snap_dates = sorted(snaps)
    base_i = bisect.bisect_left(snap_dates, START) - 1
    if base_i < 0:
        raise ValueError("START 之前没有可用快照，2槽固定20无法选池")
    base_date = snap_dates[base_i]
    fixed_codes = uni20._top_by_amount(base_date, snaps[base_date])
    fixed_uni = FixedUniverse(fixed_codes)

    # 十仓对照表（若已生成）
    base_summary = None
    base_path = OUT / "汇总_严格重跑.csv"
    if base_path.exists():
        base_summary = pd.read_csv(base_path).set_index("策略")

    specs = [
        ("原20池", orig_data, orig_dates, None, "原20只事后龙头"),
        ("时点500", pit_data, pit_dates, uni500, "全部时点500成分"),
        ("时点固定20", pit_data, pit_dates, fixed_uni, f"固定20（{base_date.date()}时点选池）"),
        ("时点前20", pit_data, pit_dates, uni20, "每月过去250日成交额前20"),
    ]
    ten_names = {
        ("原20池", "death_cross"): "原20池_死叉",
        ("原20池", "b3_prior_peak"): "原20池_b3_前日峰值修复",
        ("时点500", "death_cross"): "时点500_死叉",
        ("时点500", "b3_prior_peak"): "时点500_b3_前日峰值",
        ("时点固定20", "death_cross"): "时点固定20_死叉",
        ("时点固定20", "b3_prior_peak"): "时点固定20_b3_前日峰值",
        ("时点前20", "death_cross"): "前20成交额_死叉",
        ("时点前20", "b3_prior_peak"): "前20成交额_b3_前日峰值",
    }
    strategy_label = {"death_cross": "死叉", "b3_prior_peak": "b3_前日峰值"}

    rows, yearly_frames, results = [], [], []
    for pool_name, data, dates, uni, pool_note in specs:
        for strategy in ("death_cross", "b3_prior_peak"):
            name = f"{pool_name}_{strategy_label[strategy]}_2仓"
            r = run_backtest(data, dates, strategy, universe=uni,
                             max_positions=2, flat_weight=0.50, label=name)
            print_result(r, name)
            save_result(r, name, out_dir=TWO_OUT)
            yearly_frames.append(yearly_table(r, bench).assign(策略=name))
            results.append(r)

            row = summary_row(r, bench, name, pool_note,
                              "2槽×50%，目标满仓；b3为前日峰值口径")
            ten_name = ten_names.get((pool_name, strategy))
            if base_summary is not None and ten_name in base_summary.index:
                ten = base_summary.loc[ten_name]
                row["十仓年化"] = ten["年化"]
                row["十仓最大回撤"] = ten["最大回撤"]
                row["年化差"] = round(float(row["年化"]) - float(ten["年化"]), 2)
                row["回撤差"] = round(float(row["最大回撤"]) - float(ten["最大回撤"]), 2)
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(TWO_OUT / "汇总_2仓.csv", index=False, encoding="utf-8-sig")
    yearly = pd.concat(yearly_frames, ignore_index=True)
    yearly.to_csv(TWO_OUT / "年度_2仓.csv", index=False, encoding="utf-8-sig")

    reason_rows = []
    for r in results:
        if r["trades"].empty:
            continue
        g = r["trades"].groupby("reason").agg(
            笔数=("total_ret", "size"),
            平均净收益率=("total_ret", "mean"),
            净盈亏合计=("total_pnl", "sum"),
        ).reset_index()
        g["平均净收益率"] = (g["平均净收益率"] * 100).round(3)
        g["净盈亏合计"] = g["净盈亏合计"].round(2)
        g.insert(0, "策略", r["label"])
        reason_rows.append(g)
    if reason_rows:
        pd.concat(reason_rows, ignore_index=True).rename(
            columns={"reason": "离场原因"}).to_csv(
            TWO_OUT / "诊断_2仓_离场原因.csv", index=False, encoding="utf-8-sig")

    print(f"\n2仓汇总已保存：{TWO_OUT / '汇总_2仓.csv'}")
    print(f"2仓年度已保存：{TWO_OUT / '年度_2仓.csv'}")
    print(f"2仓离场原因已保存：{TWO_OUT / '诊断_2仓_离场原因.csv'}")


def run_one_slot_zero_axis(bench: pd.DataFrame) -> None:
    """1 槽 ×100% 与 MACD 零轴之上（DIF>0 且 DEA>0）的对照。"""
    print("\n" + "=" * 88)
    print("四、1 槽与零轴过滤对照")
    print("=" * 88)
    ONE_OUT.mkdir(parents=True, exist_ok=True)

    orig_data = load_original_pool()
    orig_dates = sorted({d for df in orig_data.values() for d in df["date"].tolist()})
    orig_dates = [d for d in orig_dates if START <= d <= END]

    pit_data = load_pit_data()
    snaps = load_snapshots()
    pit_dates = pd.to_datetime(bench["date"]).tolist()
    uni500 = Universe(snaps)
    uni20 = Universe(snaps, data=pit_data, topn=20)
    snap_dates = sorted(snaps)
    base_i = bisect.bisect_left(snap_dates, START) - 1
    if base_i < 0:
        raise ValueError("START 之前没有可用快照，零轴对照无法选池")
    base_date = snap_dates[base_i]
    fixed_codes = uni20._top_by_amount(base_date, snaps[base_date])
    fixed_uni = FixedUniverse(fixed_codes)

    base_summary = None
    base_path = OUT / "汇总_严格重跑.csv"
    if base_path.exists():
        base_summary = pd.read_csv(base_path).set_index("策略")

    pools = [
        ("原20池", orig_data, orig_dates, None, "原20只事后龙头"),
        ("时点固定20", pit_data, pit_dates, fixed_uni,
         f"固定20（{base_date.date()}时点选池）"),
        ("时点500", pit_data, pit_dates, uni500, "全部时点500成分"),
    ]
    configs = [
        ("1槽", dict(max_positions=1, flat_weight=1.0, zero_axis=False),
         "单槽100%，无零轴过滤"),
        ("10槽零轴上", dict(max_positions=10, flat_weight=0.10, zero_axis=True),
         "10槽×10%，DIF>0且DEA>0"),
        ("1槽零轴上", dict(max_positions=1, flat_weight=1.0, zero_axis=True),
         "单槽100%，DIF>0且DEA>0"),
    ]
    strat_labels = {"death_cross": "死叉", "b3_prior_peak": "b3_前日峰值"}
    base_names = {
        ("原20池", "death_cross"): "原20池_死叉",
        ("原20池", "b3_prior_peak"): "原20池_b3_前日峰值修复",
        ("时点固定20", "death_cross"): "时点固定20_死叉",
        ("时点固定20", "b3_prior_peak"): "时点固定20_b3_前日峰值",
        ("时点500", "death_cross"): "时点500_死叉",
        ("时点500", "b3_prior_peak"): "时点500_b3_前日峰值",
    }

    rows, yearly_frames, results = [], [], []
    for pool_name, data, dates, uni, pool_note in pools:
        for strategy in ("death_cross", "b3_prior_peak"):
            for cfg_name, cfg, cfg_note in configs:
                name = f"{pool_name}_{strat_labels[strategy]}_{cfg_name}"
                r = run_backtest(data, dates, strategy, universe=uni,
                                 max_positions=cfg["max_positions"],
                                 flat_weight=cfg["flat_weight"],
                                 zero_axis=cfg["zero_axis"], label=name)
                print_result(r, name)
                save_result(r, name, out_dir=ONE_OUT)
                yearly_frames.append(yearly_table(r, bench).assign(策略=name))
                results.append(r)

                row = summary_row(r, bench, name, pool_note, cfg_note)
                row["配置"] = cfg_name
                row["槽位"] = cfg["max_positions"]
                row["单槽权重"] = f"{cfg['flat_weight'] * 100:.0f}%"
                row["零轴过滤"] = "是" if cfg["zero_axis"] else "否"
                base_name = base_names.get((pool_name, strategy))
                if base_summary is not None and base_name in base_summary.index:
                    ten = base_summary.loc[base_name]
                    row["十槽基准年化"] = ten["年化"]
                    row["十槽基准回撤"] = ten["最大回撤"]
                    row["年化差"] = round(float(row["年化"]) - float(ten["年化"]), 2)
                    row["回撤差"] = round(float(row["最大回撤"]) - float(ten["最大回撤"]), 2)
                rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(ONE_OUT / "汇总_1仓与零轴.csv", index=False, encoding="utf-8-sig")
    yearly = pd.concat(yearly_frames, ignore_index=True)
    yearly.to_csv(ONE_OUT / "年度_1仓与零轴.csv", index=False, encoding="utf-8-sig")

    reason_rows = []
    for r in results:
        if r["trades"].empty:
            continue
        g = r["trades"].groupby("reason").agg(
            笔数=("total_ret", "size"),
            平均净收益率=("total_ret", "mean"),
            净盈亏合计=("total_pnl", "sum"),
        ).reset_index()
        g["平均净收益率"] = (g["平均净收益率"] * 100).round(3)
        g["净盈亏合计"] = g["净盈亏合计"].round(2)
        g.insert(0, "策略", r["label"])
        reason_rows.append(g)
    if reason_rows:
        pd.concat(reason_rows, ignore_index=True).rename(
            columns={"reason": "离场原因"}).to_csv(
            ONE_OUT / "诊断_1仓与零轴_离场原因.csv", index=False, encoding="utf-8-sig")

    print(f"\n1仓/零轴汇总已保存：{ONE_OUT / '汇总_1仓与零轴.csv'}")
    print(f"1仓/零轴年度已保存：{ONE_OUT / '年度_1仓与零轴.csv'}")
    print(f"1仓/零轴离场原因已保存：{ONE_OUT / '诊断_1仓与零轴_离场原因.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "pit", "two", "one", "all"], default="all")
    args = ap.parse_args()

    bench = load_benchmark()
    bm = benchmark_metrics(bench)
    print(f"中证500 基准 {START.date()}~{END.date()}: "
          f"累计 {bm['total_return']:.2f}%、年化 {bm['ann_return']:.2f}%、"
          f"最大回撤 {bm['mdd']:.2f}%")

    if args.mode == "two":
        run_two_slot(bench)
        return
    if args.mode == "one":
        run_one_slot_zero_axis(bench)
        return

    frames, yearly_frames, all_results = [], [], []
    if args.mode in ("validate", "all"):
        s, y, r = run_validate(bench)
        frames.append(s)
        yearly_frames.append(y)
        all_results.extend(r)
    if args.mode in ("pit", "all"):
        s, y, r = run_pit(bench)
        frames.append(s)
        yearly_frames.append(y)
        all_results.extend(r)

    summary = pd.concat(frames, ignore_index=True)
    # 基准单独一行
    summary = pd.concat([
        summary,
        pd.DataFrame([{
            "策略": "中证500买入持有", "股票池": "中证500指数",
            "区间": f"{START.date()}~{END.date()}", "累计收益": bm["total_return"],
            "年化": bm["ann_return"], "最大回撤": bm["mdd"],
            "Calmar": round(bm["ann_return"] / abs(bm["mdd"]), 3),
            "Sharpe": np.nan, "交易笔数": 0, "胜率": np.nan,
            "盈亏比": np.nan, "盈亏比_金额": np.nan,
            "平均持仓数": 0, "平均仓位": 100.0,
            "期末权益": round(INITIAL_CAPITAL * (1 + bm["total_return"] / 100), 2),
            "未平仓": 0, "买入成交": 0, "买入涨停跳过": 0,
            "买入时已非成分": 0, "买入日无行情": 0, "退市强制卖出": 0,
            "同暴露中证500": bm["total_return"],
            "同暴露年化": bm["ann_return"], "同暴露回撤": bm["mdd"],
            "备注": "价格指数，不含分红",
        }])
    ], ignore_index=True)

    summary.to_csv(OUT / "汇总_严格重跑.csv", index=False, encoding="utf-8-sig")
    yearly = pd.concat(yearly_frames, ignore_index=True)
    yearly.to_csv(OUT / "年度_严格重跑.csv", index=False, encoding="utf-8-sig")

    # 离场原因 × 策略
    reason_rows = []
    for r in all_results:
        if r["trades"].empty:
            continue
        g = r["trades"].groupby("reason").agg(
            笔数=("total_ret", "size"),
            平均净收益率=("total_ret", "mean"),
            净盈亏合计=("total_pnl", "sum"),
        ).reset_index()
        g["平均净收益率"] = (g["平均净收益率"] * 100).round(3)
        g["净盈亏合计"] = g["净盈亏合计"].round(2)
        g.insert(0, "策略", r["label"])
        reason_rows.append(g)
    if reason_rows:
        pd.concat(reason_rows, ignore_index=True).rename(
            columns={"reason": "离场原因"}).to_csv(
            OUT / "诊断_离场原因.csv", index=False, encoding="utf-8-sig")

    # 修复分解：以原报告 b3 12.48% 为锚
    key = summary.set_index("策略")
    diag_rows = []
    original_b3 = 12.48
    for name in ["原20池_b3_同日高触发", "原20池_b3_前日峰值修复",
                 "时点固定20_b3_前日峰值", "前20成交额_b3_前日峰值",
                 "时点500_b3_前日峰值"]:
        if name not in key.index:
            continue
        row = key.loc[name]
        diag_rows.append({
            "策略": name,
            "年化": row["年化"], "最大回撤": row["最大回撤"],
            "相对原报告b3年化差": round(float(row["年化"]) - original_b3, 2),
            "相对原报告b3回撤差": round(float(row["最大回撤"]) - (-26.95), 2),
        })
    # 死叉链
    for name in ["原20池_死叉", "时点固定20_死叉", "前20成交额_死叉", "时点500_死叉"]:
        if name not in key.index:
            continue
        row = key.loc[name]
        diag_rows.append({
            "策略": name,
            "年化": row["年化"], "最大回撤": row["最大回撤"],
            "相对原报告b3年化差": round(float(row["年化"]) - original_b3, 2),
            "相对原报告b3回撤差": round(float(row["最大回撤"]) - (-26.95), 2),
        })
    pd.DataFrame(diag_rows).to_csv(
        OUT / "诊断_修复分解.csv", index=False, encoding="utf-8-sig")

    print(f"\n汇总已保存：{OUT / '汇总_严格重跑.csv'}")
    print(f"年度已保存：{OUT / '年度_严格重跑.csv'}")
    print(f"离场原因已保存：{OUT / '诊断_离场原因.csv'}")
    print(f"修复分解已保存：{OUT / '诊断_修复分解.csv'}")


if __name__ == "__main__":
    main()
