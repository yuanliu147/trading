"""核心特征.py — 因果特征与多周期聚合引擎（趋势结构项目，行情冻结 2026-10-26）

因果契约（下游必须遵守，违反即未来函数）
----------------------------------------
1. 每个 bar 的特征在**该 bar 收盘后**生成，available_at = 该 bar 日期。
   日线 available_at = date；周/月线 available_at = period_end（周期内最后一个交易日）。
2. 日期 t 可见的任何值只依赖 <= t 的行情。未走完的周/月周期**不输出**；
   起点不完整的首个周期标记 partial_start 并从 completed 视图剔除。
3. 枢轴在高/低点右侧 k 根**已收盘** bar 后才确认：confirm_idx = pivot_idx + k（按可交易 bar 计）。
   HH/HL/LH/LL、swing 高低点、结构止损只在 confirm 之后可见，绝不回填到枢轴当日。
4. 价格类指标用后复权（baostock adjustflag=1 / DuckDB v_daily_hfq）消除分红送转跳空；
   成交量/成交额/换手为原始口径。结构止损给百分比（复权不变）与原始价换算系数 price_scale。
5. 停牌 bar（trade_status=0，OHLC 平、volume 为空）不参与枢轴/突破/量能统计，
   但保留在日历网格上，使 MA20 仍等于“20 个交易日”。

同价 pivot / 平台处理
---------------------
枢轴定义：high[i] 为枢轴高 ⇔ 左侧 k 根严格低、右侧 k 根不高于它。
平台（连续同价）只在**第一根**产生枢轴，既不重复也不丢失，结果确定。

参数说明
--------
所有窗口取行业常规值（20/50/200 日均线、14 ATR/RSI/ADX、20 日 Donchian、5 根日线枢轴确认），
不按收益调参。参数集中在 FeatureConfig，便于锁定后复现。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform as _platform
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

FREEZE = pd.Timestamp("2026-10-26")
ROOT = Path(__file__).resolve().parents[1]
BAOSTOCK_DB = Path("/Users/admin/WorkSpace/baostock_data/data/baostock.db")
DUCKDB_PATH = Path("/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb")
EVIDENCE = ROOT / "04_证据"


# ---------------------------------------------------------------- 配置

@dataclass(frozen=True)
class FeatureConfig:
    sma_windows: tuple = (5, 10, 20, 50, 200)
    ema_windows: tuple = (10, 21)
    atr_n: int = 14
    atr_short: int = 20
    atr_long: int = 60
    rsi_n: int = 14
    adx_n: int = 14
    don_n: int = 20
    bb_n: int = 20
    bb_rank_n: int = 252
    nr_n: int = 7
    slope_n: int = 20
    pivot_k: int = 5          # 日线枢轴：右侧 5 根已收盘 bar 确认
    pivot_k_w: int = 3
    pivot_k_m: int = 2
    pullback_min: float = 0.236
    pullback_max: float = 0.618
    vol_contract: float = 0.85
    restart_vol: float = 1.2
    restart_lookback: int = 5
    stop_atr_buffer: float = 0.5
    min_hist: int = 250
    drop_partial_start: bool = True

    def pivot_k_for(self, freq: str) -> int:
        return {"D": self.pivot_k, "W": self.pivot_k_w, "M": self.pivot_k_m}[freq]


DEFAULT_CFG = FeatureConfig()


# ---------------------------------------------------------------- 基础工具

def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((b != 0) & np.isfinite(b) & np.isfinite(a), a / b, np.nan)
    return out


def _ffill(a):
    return pd.Series(np.asarray(a, dtype=float)).ffill().to_numpy()


def _round2(x):
    return np.floor(np.asarray(x, dtype=float) * 100.0 + 0.5) / 100.0


def _as_dates(values) -> np.ndarray:
    s = pd.to_datetime(pd.Series(values)).dt.normalize()
    return s.to_numpy(dtype="datetime64[D]")


# ---------------------------------------------------------------- 交易日历

class Calendar:
    """交易日历。所有周期边界由日历决定，不由个股行情决定。"""

    def __init__(self, dates):
        d = np.sort(np.unique(_as_dates(dates)))
        self.dates = d
        self._pos = pd.Series(np.arange(len(d), dtype=np.int64), index=pd.DatetimeIndex(d))
        self._bounds = {}

    @classmethod
    def from_baostock(cls, path=BAOSTOCK_DB, end=None):
        """日历只含交易日，不含任何行情。默认取到数据源已有日历末端，
        以便确认“某一周/月是否真的结束”（末端周期一律视为未确认，见 aggregate）。"""
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            if end is None:
                df = pd.read_sql_query(
                    "select calendar_date from trade_dates where is_trading_day=1 order by 1", con)
            else:
                df = pd.read_sql_query(
                    "select calendar_date from trade_dates where is_trading_day=1 and calendar_date<=? order by 1",
                    con, params=(pd.Timestamp(end).strftime("%Y-%m-%d"),))
        finally:
            con.close()
        return cls(pd.to_datetime(df["calendar_date"]))

    def __len__(self):
        return len(self.dates)

    def pos(self, d):
        return int(self._pos.loc[pd.Timestamp(d).normalize()])

    def has(self, d) -> bool:
        return pd.Timestamp(d).normalize() in self._pos.index

    def prev(self, d):
        p = self.pos(d)
        return self.dates[p - 1] if p > 0 else None

    def next(self, d):
        p = self.pos(d)
        return self.dates[p + 1] if p + 1 < len(self.dates) else None

    def period_key(self, dates, freq: str) -> np.ndarray:
        s = pd.DatetimeIndex(_as_dates(dates))
        if freq == "W":
            iso = s.isocalendar()
            return (iso["year"].to_numpy().astype(np.int64) * 100 + iso["week"].to_numpy().astype(np.int64))
        if freq == "M":
            return s.year.to_numpy().astype(np.int64) * 100 + s.month.to_numpy().astype(np.int64)
        raise ValueError(freq)

    def bounds(self, freq: str) -> pd.DataFrame:
        if freq in self._bounds:
            return self._bounds[freq]
        key = self.period_key(self.dates, freq)
        df = pd.DataFrame({"date": self.dates, "key": key})
        g = df.groupby("key")["date"].agg(period_start="min", period_end="max", n_cal="size")
        g = g.reset_index()
        self._bounds[freq] = g
        return g

    def sessions_between(self, a, b) -> int:
        """交易日间隔（a、b 均须为交易日；返回 b-a 的交易日数）。"""
        return int(self.pos(b) - self.pos(a))


# ---------------------------------------------------------------- 数据源适配器

def to_bs_code(code: str) -> str:
    if "." not in code:
        raise ValueError(code)
    a, b = code.split(".")
    if a.lower() in ("sh", "sz", "bj"):
        return f"{a.lower()}.{b}"
    return f"{b.lower()}.{a}"


def to_ths_code(code: str) -> str:
    a, b = code.split(".")
    if a.lower() in ("sh", "sz", "bj"):
        return f"{b}.{a.upper()}"
    return f"{a}.{b.upper()}"


class BaostockSource:
    """baostock SQLite 只读适配器。adjust='hfq' 取 adjustflag=1，'raw' 取 3。"""

    flag = {"hfq": 1, "qfq": 2, "raw": 3}

    def __init__(self, path=BAOSTOCK_DB):
        self.path = str(path)

    def _con(self):
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def load(self, code, adjust="hfq", start=None, end=FREEZE, with_raw=False) -> pd.DataFrame:
        code = to_bs_code(code)
        flag = self.flag[adjust]
        end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
        start_sql = "and date>=? " if start is not None else ""
        q = ("select date,open,high,low,close,preclose,volume,amount,turn,trade_status,is_st "
             "from stock_kline_d where code=? and adjustflag=? and date<=? " + start_sql + "order by date")
        params = [code, flag, end_s] + ([pd.Timestamp(start).strftime("%Y-%m-%d")] if start is not None else [])
        con = self._con()
        try:
            df = pd.read_sql_query(q, con, params=params)
        finally:
            con.close()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        if with_raw and flag != 3:
            q2 = ("select date,close as close_raw from stock_kline_d "
                  "where code=? and adjustflag=3 and date<=? " + start_sql + "order by date")
            p2 = [code, end_s] + ([pd.Timestamp(start).strftime("%Y-%m-%d")] if start is not None else [])
            con = self._con()
            try:
                raw = pd.read_sql_query(q2, con, params=p2)
            finally:
                con.close()
            raw["date"] = pd.to_datetime(raw["date"])
            df = df.merge(raw, on="date", how="left")
        return df.sort_values("date").reset_index(drop=True)

    def symbols(self, listed_only=False, asof=FREEZE) -> pd.DataFrame:
        con = self._con()
        try:
            df = pd.read_sql_query(
                "select code,code_name,ipo_date,out_date,type,status from stock_basic where type=1", con)
        finally:
            con.close()
        if listed_only:
            a = pd.Timestamp(asof).strftime("%Y-%m-%d")
            df = df[(df["ipo_date"] <= a) & ((df["out_date"].isna()) | (df["out_date"] > a))]
        return df.reset_index(drop=True)

    def index(self, code="sh.000300", start=None, end=FREEZE) -> pd.DataFrame:
        q = "select date,open,high,low,close,preclose,volume,amount from index_kline_d where code=? and date<=?"
        params = [code, pd.Timestamp(end).strftime("%Y-%m-%d")]
        if start is not None:
            q += " and date>=?"
            params.append(pd.Timestamp(start).strftime("%Y-%m-%d"))
        con = self._con()
        try:
            df = pd.read_sql_query(q + " order by date", con, params=params)
        finally:
            con.close()
        df["date"] = pd.to_datetime(df["date"])
        return df


class DuckDBSource:
    """同花顺 DuckDB 只读适配器（2016-10-13 起，无 preclose，需自算）。"""

    def __init__(self, path=DUCKDB_PATH):
        import duckdb
        self.con = duckdb.connect(str(path), read_only=True)

    def load(self, code, adjust="hfq", start=None, end=FREEZE, with_raw=False) -> pd.DataFrame:
        code = to_ths_code(code)
        view = {"hfq": "v_daily_hfq", "qfq": "v_daily_qfq", "raw": "v_daily"}[adjust]
        cols = "thscode,date,open,high,low,close,volume,amount"
        q = f"select {cols} from {view} where thscode=? and date<=?"
        params = [code, pd.Timestamp(end).date()]
        if start is not None:
            q += " and date>=?"
            params.append(pd.Timestamp(start).date())
        df = self.con.execute(q + " order by date", params).df()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        if with_raw and adjust != "raw":
            raw = self.con.execute(
                "select date,close as close_raw from v_daily where thscode=? and date<=? order by date",
                [code, pd.Timestamp(end).date()]).df()
            raw["date"] = pd.to_datetime(raw["date"])
            df = df.merge(raw, on="date", how="left")
        df = df.drop(columns=[c for c in ("thscode",) if c in df])
        return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- BarFrame

class Bars:
    """日历对齐的 bar 序列。traded 掩码区分真实成交与停牌。"""

    def __init__(self, df: pd.DataFrame, code: str = "", freq: str = "D"):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
        for c in ("open", "high", "low", "close", "volume", "amount"):
            if c not in df:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "turn" not in df:
            df["turn"] = np.nan
        if "is_st" not in df:
            df["is_st"] = 0
        if "trade_status" in df:
            ts = pd.to_numeric(df["trade_status"], errors="coerce").fillna(1)
        else:
            ts = pd.Series(1, index=df.index)
        vol = df["volume"]
        traded = (ts.to_numpy() != 0) & vol.notna().to_numpy() & (vol.to_numpy() > 0) & (df["close"].to_numpy() > 0)
        self.df = df
        self.code = code
        self.freq = freq
        self.traded = traded
        self.suspended = ~traded
        if "preclose" in df and df["preclose"].notna().any():
            pc = df["preclose"].to_numpy(dtype=float).copy()
            cs = df["close"].to_numpy(dtype=float)
            pc[np.isnan(pc)] = np.nan
            miss = np.isnan(pc)
            if miss.any():
                shift = np.concatenate([[np.nan], cs[:-1]])
                pc[miss] = shift[miss]
        else:
            cs = df["close"].to_numpy(dtype=float)
            pc = np.concatenate([[np.nan], cs[:-1]])
        self.prev_close = pc
        self.close = df["close"].to_numpy(dtype=float)
        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.volume = vol.to_numpy(dtype=float)
        self.amount = df["amount"].to_numpy(dtype=float)
        self.turn = df["turn"].to_numpy(dtype=float)
        self.is_st = pd.to_numeric(df["is_st"], errors="coerce").fillna(0).to_numpy()
        self.dates = df["date"].to_numpy(dtype="datetime64[D]")

    def __len__(self):
        return len(self.df)

    @property
    def last_date(self):
        return pd.Timestamp(self.dates[-1])

    def subset(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.all():
            return self
        meta = getattr(self, "meta", None)
        if meta is not None:
            m = meta[mask].reset_index(drop=True)
            d = pd.DataFrame({
                "date": m["date"], "open": m["open"], "high": m["high"], "low": m["low"],
                "close": m["close"], "volume": m["volume"], "amount": m["amount"],
                "turn": m["turn"] if "turn" in m else np.nan,
                "trade_status": np.where(m["n_traded"].to_numpy() > 0, 1, 0),
            })
            b = Bars(d, code=self.code, freq=self.freq)
            b.meta = m
            return b
        return Bars(self.df[mask].reset_index(drop=True), code=self.code, freq=self.freq)


def bars_from_arrays(date, open_, high, low, close, volume, amount=None, preclose=None,
                     trade_status=None, turn=None, is_st=None, code="", freq="D") -> Bars:
    n = len(date)
    df = pd.DataFrame({
        "date": pd.to_datetime(date), "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "amount": np.asarray(amount, dtype=float) if amount is not None else np.full(n, np.nan),
        "turn": np.asarray(turn, dtype=float) if turn is not None else np.full(n, np.nan),
    })
    if preclose is not None:
        df["preclose"] = preclose
    if trade_status is not None:
        df["trade_status"] = trade_status
    if is_st is not None:
        df["is_st"] = is_st
    return Bars(df, code=code, freq=freq)


# ---------------------------------------------------------------- 周期聚合

def aggregate(bars: Bars, freq: str, cal: Calendar, cfg: FeatureConfig = DEFAULT_CFG) -> Bars:
    """把日线聚合成**已完成**周/月线。未完成周期不输出；起点不完整周期标 partial_start。"""
    if freq == "D":
        return bars
    df = bars.df.copy()
    key = cal.period_key(df["date"].to_numpy(), freq)
    df["_key"] = key
    df["_traded"] = bars.traded
    df["_vol"] = np.where(bars.traded, bars.volume, 0.0)
    df["_amt"] = np.where(bars.traded, bars.amount, 0.0)
    g = df.groupby("_key", sort=True)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["_vol"].sum(),
        "amount": g["_amt"].sum(),
        "turn": g["turn"].sum(min_count=1),
        "n_days": g.size(),
        "first_date": g["date"].min(),
        "last_date": g["date"].max(),
    })
    traded = df[df["_traded"]].groupby("_key")["date"].max().rename("last_trade_date")
    out = out.join(traded)
    out["n_traded"] = df[df["_traded"]].groupby("_key").size().reindex(out.index).fillna(0).astype(int)
    b = cal.bounds(freq).set_index("key")
    out = out.join(b[["period_start", "period_end", "n_cal"]], how="left")
    last_bar = df["date"].max()
    # 只有“日历末端之后还有下一周期”的周期才被确认为已结束；日历最后一个周期一律未确认。
    confirmed = pd.Index(b.index[:-1])
    out = out[(out["period_end"] <= last_bar) & out.index.isin(confirmed)]
    out = out.reset_index().rename(columns={"_key": "key"})
    out["period_start"] = pd.to_datetime(out["period_start"])
    out["period_end"] = pd.to_datetime(out["period_end"])
    out["first_date"] = pd.to_datetime(out["first_date"])
    out["last_date"] = pd.to_datetime(out["last_date"])
    out["last_trade_date"] = pd.to_datetime(out["last_trade_date"])
    out["coverage"] = out["n_days"] / out["n_cal"].clip(lower=1)
    out["partial_start"] = False
    if len(out):
        out.loc[out.index[0], "partial_start"] = bool(out.loc[out.index[0], "first_date"] > out.loc[out.index[0], "period_start"])
    out["gap_start"] = out["first_date"] > out["period_start"]
    out["stale_end"] = out["last_trade_date"].isna() | (out["last_trade_date"] < out["period_end"])
    out["thin"] = out["n_traded"] < (3 if freq == "W" else 12)
    out["sessions_since_prev"] = out["period_end"].map(
        lambda d: cal.pos(d)).diff().fillna(np.nan)
    dates = out["period_end"].to_numpy(dtype="datetime64[D]")
    frame = out.rename(columns={"period_end": "date"})
    frame = frame.drop(columns=["key"])
    b2 = Bars(pd.DataFrame({
        "date": frame["date"], "open": frame["open"], "high": frame["high"], "low": frame["low"],
        "close": frame["close"], "volume": frame["volume"], "amount": frame["amount"],
        "turn": frame["turn"],
        "trade_status": np.where(frame["n_traded"] > 0, 1, 0),
    }), code=bars.code, freq=freq)
    b2.meta = frame.reset_index(drop=True)
    b2.dates = dates
    return b2


# ---------------------------------------------------------------- 指标（全部因果）

def sma(x, n):
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(n, min_periods=n).mean().to_numpy()


def ema(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if len(x) < n or np.isnan(x[:n]).any():
        return out
    out[n - 1] = np.nanmean(x[:n])
    a = 2.0 / (n + 1.0)
    for i in range(n, len(x)):
        xi = x[i]
        out[i] = a * xi + (1 - a) * out[i - 1] if np.isfinite(xi) else out[i - 1]
    return out


def _wilder(x, n):
    """Wilder 平滑；NaN 处沿用上一值（停牌不制造虚假波动）。"""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    valid = np.isfinite(x)
    if len(x) < n:
        return out
    c = np.convolve(valid.astype(int), np.ones(n, int), "valid")
    idx = np.flatnonzero(c == n)
    if len(idx) == 0:
        return out
    s = int(idx[0])
    out[s + n - 1] = np.nanmean(x[s:s + n])
    for i in range(s + n, len(x)):
        prev = out[i - 1]
        out[i] = (prev * (n - 1) + x[i]) / n if np.isfinite(x[i]) else prev
    return out


def true_range(bars: Bars):
    h, l, pc = bars.high, bars.low, bars.prev_close
    tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))
    tr = np.where(np.isfinite(pc), tr, np.nan)
    return np.where(bars.traded, tr, np.nan)


def atr(bars: Bars, n):
    return _wilder(true_range(bars), n)


def rsi(bars: Bars, n):
    c = bars.close
    d = np.diff(c, prepend=np.nan)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    valid = bars.traded & np.isfinite(d)
    gain = np.where(valid, gain, np.nan)
    loss = np.where(valid, loss, np.nan)
    ag, al = _wilder(gain, n), _wilder(loss, n)
    return 100.0 - 100.0 / (1.0 + _safe_div(ag, al))


def dmi(bars: Bars, n=14):
    h, l, pc = bars.high, bars.low, bars.prev_close
    up = h - np.concatenate([[np.nan], h[:-1]])
    dn = np.concatenate([[np.nan], l[:-1]]) - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    v = bars.traded & np.isfinite(pc)
    tr = np.where(v, true_range(bars), np.nan)
    pdm = np.where(v, pdm, np.nan)
    ndm = np.where(v, ndm, np.nan)
    str_, spdm, sndm = _wilder(tr, n), _wilder(pdm, n), _wilder(ndm, n)
    pdi = 100.0 * _safe_div(spdm, str_)
    ndi = 100.0 * _safe_div(sndm, str_)
    dx = 100.0 * _safe_div(np.abs(pdi - ndi), pdi + ndi)
    adx = _wilder(dx, n)
    return pdi, ndi, adx


def donchian(bars: Bars, n):
    h = pd.Series(bars.high).rolling(n, min_periods=n).max().to_numpy()
    l = pd.Series(bars.low).rolling(n, min_periods=n).min().to_numpy()
    h_prev = pd.Series(bars.high).shift(1).rolling(n, min_periods=n).max().to_numpy()
    l_prev = pd.Series(bars.low).shift(1).rolling(n, min_periods=n).min().to_numpy()
    return h, l, h_prev, l_prev


def bollinger_width(bars: Bars, n):
    c = bars.close
    m = pd.Series(c).rolling(n, min_periods=n).mean().to_numpy()
    s = pd.Series(c).rolling(n, min_periods=n).std(ddof=0).to_numpy()
    return _safe_div(2 * 2 * s, m), m, s


def rolling_pct_rank(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if len(x) < n:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(x, n)
    cur = x[n - 1:]
    out[n - 1:] = np.nanmean(w <= cur[:, None], axis=1)
    return out


def log_slope(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if len(x) < n:
        return out
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    denom = (t * t).sum()
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(np.log(np.where(x > 0, x, np.nan)), n)
    out[n - 1:] = (w * t).sum(axis=1) / denom * 10000.0
    return out


# ---------------------------------------------------------------- 枢轴

def detect_pivots(high, low, k):
    """严格枢轴：左 k 根严格低/高，右 k 根不高于/不低于。平台只认第一根。"""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = len(high)
    empty = np.array([], dtype=int)
    if n < 2 * k + 1:
        return empty, empty
    from numpy.lib.stride_tricks import sliding_window_view
    wh = sliding_window_view(high, 2 * k + 1)
    wl = sliding_window_view(low, 2 * k + 1)
    ch, cl = wh[:, k], wl[:, k]
    fin_h = np.isfinite(wh).all(axis=1) & np.isfinite(ch)
    fin_l = np.isfinite(wl).all(axis=1) & np.isfinite(cl)
    ph = (wh[:, :k] < ch[:, None]).all(axis=1) & (wh[:, k + 1:] <= ch[:, None]).all(axis=1) & fin_h
    pl = (wl[:, :k] > cl[:, None]).all(axis=1) & (wl[:, k + 1:] >= cl[:, None]).all(axis=1) & fin_l
    return np.flatnonzero(ph) + k, np.flatnonzero(pl) + k


def build_pivots(bars: Bars, k: int):
    """在**可交易 bar** 上找枢轴，再映射回日历索引。confirm = 右侧第 k 根可交易 bar。"""
    idx = np.flatnonzero(bars.traded & np.isfinite(bars.high) & np.isfinite(bars.low))
    out = {"high_cal": [], "low_cal": [], "high_confirm": [], "low_confirm": [], "high_px": [], "low_px": []}
    if len(idx) < 2 * k + 1:
        return out
    ph, pl = detect_pivots(bars.high[idx], bars.low[idx], k)
    for i in ph:
        if i + k < len(idx):
            out["high_cal"].append(int(idx[i]))
            out["high_confirm"].append(int(idx[i + k]))
            out["high_px"].append(float(bars.high[idx[i]]))
    for i in pl:
        if i + k < len(idx):
            out["low_cal"].append(int(idx[i]))
            out["low_confirm"].append(int(idx[i + k]))
            out["low_px"].append(float(bars.low[idx[i]]))
    return out


def structure_series(n, pivots):
    """把已确认枢轴转成逐 bar 可见的结构量（在 confirm 处生效并前向填充）。"""
    nan = np.nan
    s = {
        "swing_high": np.full(n, nan), "swing_low": np.full(n, nan),
        "swing_high_idx": np.full(n, nan), "swing_low_idx": np.full(n, nan),
        "h_label": np.full(n, None), "l_label": np.full(n, None),
        "score": np.full(n, nan), "trend": np.full(n, None),
    }
    events = [(c, "H", p, i) for i, c, p in zip(pivots["high_cal"], pivots["high_confirm"], pivots["high_px"])]
    events += [(c, "L", p, i) for i, c, p in zip(pivots["low_cal"], pivots["low_confirm"], pivots["low_px"])]
    events.sort(key=lambda e: (e[0], e[1]))
    lh = ll = nan
    lh_i = ll_i = -1
    hl = llab = None
    hist = []
    for c, kind, px, pi in events:
        if not (0 <= c < n):
            continue
        if kind == "H":
            hl = "H" if not np.isfinite(lh) else ("HH" if px > lh else "LH")
            lh, lh_i = px, pi
            s["h_label"][c] = hl
            hist.append(1 if hl == "HH" else -1)
        else:
            llab = "L" if not np.isfinite(ll) else ("HL" if px > ll else "LL")
            ll, ll_i = px, pi
            s["l_label"][c] = llab
            hist.append(1 if llab == "HL" else -1)
        s["swing_high"][c] = lh
        s["swing_low"][c] = ll
        s["swing_high_idx"][c] = lh_i if lh_i >= 0 else nan
        s["swing_low_idx"][c] = ll_i if ll_i >= 0 else nan
        s["score"][c] = float(sum(hist[-4:]))
        s["h_label"][c] = hl
        s["l_label"][c] = llab
        if hl == "HH" and llab == "HL":
            s["trend"][c] = "up"
        elif hl == "LH" and llab == "LL":
            s["trend"][c] = "down"
        elif hl is None or llab is None:
            s["trend"][c] = "none"
        else:
            s["trend"][c] = "transition"
    for key in ("swing_high", "swing_low", "swing_high_idx", "swing_low_idx", "score"):
        s[key] = _ffill(s[key])
    for key in ("h_label", "l_label", "trend"):
        s[key] = pd.Series(s[key]).ffill().to_numpy()
    return s


# ---------------------------------------------------------------- 结构止损 / 回调

def structure_block(bars: Bars, cfg: FeatureConfig, atr_s):
    n = len(bars)
    piv = build_pivots(bars, cfg.pivot_k_for(bars.freq))
    st = structure_series(n, piv)
    close = bars.close
    sh, sl = st["swing_high"], st["swing_low"]
    hi_idx, lo_idx = st["swing_high_idx"], st["swing_low_idx"]
    up_leg = np.isfinite(hi_idx) & np.isfinite(lo_idx) & (hi_idx > lo_idx) & (sh > sl)
    rng = np.where(up_leg, sh - sl, np.nan)
    retrace = _safe_div(sh - close, rng)
    advance = _safe_div(close - sl, rng)
    stop_px = np.where(np.isfinite(sl) & (sl < close), sl, np.nan)
    stop_pct = _safe_div(close - stop_px, close)
    stop_atr = _safe_div(close - stop_px, atr_s)
    buf_px = np.where(np.isfinite(sl), sl - cfg.stop_atr_buffer * atr_s, np.nan)
    buf_pct = _safe_div(close - buf_px, close)
    broken = np.isfinite(sl) & (close < sl)

    vol20 = pd.Series(np.where(bars.traded, bars.volume, np.nan)).rolling(20, min_periods=10).mean().to_numpy()
    vol5 = pd.Series(np.where(bars.traded, bars.volume, np.nan)).rolling(5, min_periods=3).mean().to_numpy()
    vcont = _safe_div(vol5, vol20) < cfg.vol_contract
    uptrend = (st["trend"] == "up")
    pull = (uptrend & up_leg & (retrace >= cfg.pullback_min) & (retrace <= cfg.pullback_max)
            & (close > sl) & np.isfinite(retrace))
    pull_v = pull & vcont
    recent = pd.Series(pull_v.astype(float)).rolling(cfg.restart_lookback, min_periods=1).max().to_numpy() > 0
    prior_high = pd.Series(bars.high).shift(1).rolling(3, min_periods=2).max().to_numpy()
    vol_day = np.where(bars.traded, bars.volume, np.nan)
    restart = recent & np.isfinite(prior_high) & (close > prior_high) & (vol_day > cfg.restart_vol * vol20)
    return {
        "pivots": piv, "struct": st,
        "retrace": retrace, "advance": advance, "up_leg": up_leg,
        "stop_px": stop_px, "stop_pct": stop_pct, "stop_atr": stop_atr,
        "stop_buf_pct": buf_pct, "broken": broken,
        "pullback": pull, "pullback_vol": pull_v, "restart": restart,
        "vol_ratio5_20": _safe_div(vol5, vol20),
    }


# ---------------------------------------------------------------- 涨跌停

def limit_pct(code: str, is_st, dt=None):
    parts = code.split(".")
    c = parts[1] if parts[0].lower() in ("sh", "sz", "bj") else parts[0]
    if c.startswith(("688", "689")):
        base = 0.20
    elif c.startswith(("300", "301")):
        base = 0.20 if (dt is not None and pd.Timestamp(dt) >= pd.Timestamp("2020-08-24")) else 0.10
    elif c.startswith(("4", "8", "92")):
        base = 0.30
    else:
        base = 0.10
    if is_st and base == 0.10:
        base = 0.05
    return base


def limit_flags(bars: Bars):
    is_st = bars.is_st.astype(bool)
    lim = np.array([limit_pct(bars.code, is_st[i], bars.dates[i]) for i in range(len(bars))])
    up = _round2(bars.prev_close * (1 + lim))
    dn = _round2(bars.prev_close * (1 - lim))
    at_up = bars.traded & np.isfinite(up) & (bars.close >= up - 0.005)
    at_dn = bars.traded & np.isfinite(dn) & (bars.close <= dn + 0.005)
    one_word = at_up & (bars.high <= bars.low + 1e-9)
    return {"limit_pct": lim, "at_limit_up": at_up, "at_limit_down": at_dn, "one_word_up": one_word}


# ---------------------------------------------------------------- 特征矩阵

def feature_matrix(bars: Bars, cfg: FeatureConfig = DEFAULT_CFG, cal: Calendar | None = None,
                   freq: str = "D") -> pd.DataFrame:
    """返回逐 bar 特征表；date 为可用日期（日线=date，周月=period_end）。"""
    if freq != "D" and getattr(bars, "freq", "D") != freq:
        if cal is None:
            raise ValueError("周/月聚合需要 Calendar")
        bars = aggregate(bars, freq, cal, cfg)
    if freq != "D" and getattr(bars, "meta", None) is not None and cfg.drop_partial_start:
        meta = bars.meta
        if "partial_start" in meta and len(meta):
            keep = ~meta["partial_start"].to_numpy(dtype=bool)
            if not keep.all() and keep.any():
                bars = bars.subset(keep)
    n = len(bars)
    out = pd.DataFrame({"date": pd.to_datetime(bars.dates), "close": bars.close,
                        "traded": bars.traded})
    if hasattr(bars, "meta"):
        m = bars.meta
        for c in ("period_start", "n_days", "n_traded", "n_cal", "coverage", "partial_start",
                  "gap_start", "stale_end", "thin", "sessions_since_prev"):
            if c in m:
                out[c] = m[c].to_numpy()
    out["available_at"] = out["date"]

    c = bars.close
    out["ret_1"] = c / np.concatenate([[np.nan], c[:-1]]) - 1
    for w in (5, 20):
        out[f"ret_{w}"] = c / pd.Series(c).shift(w).to_numpy() - 1
    for w in cfg.sma_windows:
        out[f"ma{w}"] = sma(c, w)
    out["ma_stack"] = ((out["ma20"] > out["ma50"]) & (out["ma50"] > out["ma200"])).astype(float)
    out.loc[out["ma200"].isna(), "ma_stack"] = np.nan
    out["px_vs_ma20"] = _safe_div(c, out["ma20"]) - 1
    out["px_vs_ma50"] = _safe_div(c, out["ma50"]) - 1
    for w in cfg.ema_windows:
        out[f"ema{w}"] = ema(c, w)

    a14 = atr(bars, cfg.atr_n)
    a20 = atr(bars, cfg.atr_short)
    a60 = atr(bars, cfg.atr_long)
    out["atr14"] = a14
    out["atr_pct"] = _safe_div(a14, c)
    out["atr_ratio"] = _safe_div(a20, a60)
    out["vol_contract"] = (out["atr_ratio"] < 0.75).astype(float)
    out.loc[out["atr_ratio"].isna(), "vol_contract"] = np.nan
    out["rsi14"] = rsi(bars, cfg.rsi_n)
    pdi, ndi, adx = dmi(bars, cfg.adx_n)
    out["plus_di"], out["minus_di"], out["adx14"] = pdi, ndi, adx
    out["di_spread"] = pdi - ndi

    dh, dl, dhp, dlp = donchian(bars, cfg.don_n)
    out["don_high"], out["don_low"] = dh, dl
    out["don_high_prev"], out["don_low_prev"] = dhp, dlp
    out["don_pos"] = _safe_div(c - dl, dh - dl)
    out["don_break_up"] = (c > dhp).astype(float)
    out.loc[~np.isfinite(dhp), "don_break_up"] = np.nan
    out["don_break_dn"] = (c < dlp).astype(float)
    out.loc[~np.isfinite(dlp), "don_break_dn"] = np.nan
    out["don_break_atr"] = _safe_div(c - dhp, a14)

    bbw, bbm, bbs = bollinger_width(bars, cfg.bb_n)
    out["bb_width"] = bbw
    out["bb_width_rank"] = rolling_pct_rank(bbw, cfg.bb_rank_n)
    rng = bars.high - bars.low
    out["range_pct"] = _safe_div(rng, c)
    out["nr7"] = (rng <= pd.Series(rng).rolling(cfg.nr_n, min_periods=cfg.nr_n).min().to_numpy()).astype(float)

    amt = pd.Series(bars.amount).rolling(20, min_periods=10).mean().to_numpy()
    out["amount"] = bars.amount
    out["amt_ma20"] = amt
    out["amt_ratio20"] = _safe_div(bars.amount, amt)
    la = np.log(np.where(bars.amount > 0, bars.amount, np.nan))
    out["amt_z20"] = _safe_div(la - pd.Series(la).rolling(20, min_periods=10).mean().to_numpy(),
                               pd.Series(la).rolling(20, min_periods=10).std(ddof=0).to_numpy())
    out["turn"] = bars.turn
    out["turn_z20"] = _safe_div(bars.turn - pd.Series(bars.turn).rolling(20, min_periods=10).mean().to_numpy(),
                                pd.Series(bars.turn).rolling(20, min_periods=10).std(ddof=0).to_numpy())
    out["slope20_bp"] = log_slope(c, 20)
    out["slope60_bp"] = log_slope(c, 60)

    lf = limit_flags(bars)
    out["at_limit_up"] = lf["at_limit_up"]
    out["at_limit_down"] = lf["at_limit_down"]
    out["one_word_up"] = lf["one_word_up"]
    out["suspended"] = bars.suspended

    sb = structure_block(bars, cfg, a20)
    st = sb["struct"]
    out["swing_high"] = st["swing_high"]
    out["swing_low"] = st["swing_low"]
    out["bars_since_high"] = np.arange(n) - st["swing_high_idx"]
    out["bars_since_low"] = np.arange(n) - st["swing_low_idx"]
    out["h_label"] = st["h_label"]
    out["l_label"] = st["l_label"]
    out["trend"] = st["trend"]
    out["struct_score"] = st["score"]
    out["retrace"] = sb["retrace"]
    out["advance"] = sb["advance"]
    out["up_leg"] = sb["up_leg"]
    out["pullback"] = sb["pullback"]
    out["pullback_vol"] = sb["pullback_vol"]
    out["restart"] = sb["restart"]
    out["stop_px_adj"] = sb["stop_px"]
    out["stop_pct"] = sb["stop_pct"]
    out["stop_atr"] = sb["stop_atr"]
    out["stop_buf_pct"] = sb["stop_buf_pct"]
    out["struct_broken"] = sb["broken"]
    out["vol_ratio5_20"] = sb["vol_ratio5_20"]
    if "close_raw" in bars.df:
        cr = pd.to_numeric(bars.df["close_raw"], errors="coerce").to_numpy()
        out["price_scale"] = _safe_div(c, cr)
    else:
        out["price_scale"] = np.nan

    piv = sb["pivots"]
    for side in ("high", "low"):
        col = np.full(n, np.nan)
        for cal_i in piv[f"{side}_cal"]:
            if 0 <= cal_i < n:
                col[cal_i] = 1.0
        out[f"pivot_{side}_mark"] = col
        conf = np.full(n, np.nan)
        for cal_i, cc in zip(piv[f"{side}_cal"], piv[f"{side}_confirm"]):
            if 0 <= cc < n:
                conf[cc] = 1.0
        out[f"pivot_{side}_confirm"] = conf
    out["feature_ready"] = ((out["ma200"].notna()) & (out["adx14"].notna())
                            & (out["swing_low"].notna()) & out["atr14"].notna())
    return out


# ---------------------------------------------------------------- 市场宽度 / 指数环境

def breadth_duckdb(start=None, end=FREEZE, con=None, cfg=DEFAULT_CFG) -> pd.DataFrame:
    """同花顺 DuckDB 全市场宽度（2016-10-13 起，仅现存代码，存在幸存者偏差）。"""
    import duckdb
    own = con is None
    if own:
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d") if start is not None else "1900-01-01"
    q = f"""
    with base as (
      select thscode, date, close, amount
      from v_daily where date <= DATE '{end_s}' and date >= DATE '{start_s}' and volume > 0 and close > 0
    ), w as (
      select thscode, date, close, amount,
        avg(close) over (partition by thscode order by date rows between 19 preceding and current row) ma20,
        avg(close) over (partition by thscode order by date rows between 49 preceding and current row) ma50,
        avg(close) over (partition by thscode order by date rows between 199 preceding and current row) ma200,
        max(close) over (partition by thscode order by date rows between 251 preceding and current row) hi252,
        min(close) over (partition by thscode order by date rows between 251 preceding and current row) lo252,
        count(*) over (partition by thscode order by date rows between 19 preceding and current row) n20,
        count(*) over (partition by thscode order by date rows between 49 preceding and current row) n50,
        count(*) over (partition by thscode order by date rows between 199 preceding and current row) n200
      from base
    )
    select date,
      count(*) as n_stocks,
      avg(case when n20 = 20 and close > ma20 then 1.0 else 0.0 end) as pct_above_ma20,
      avg(case when n50 = 50 and close > ma50 then 1.0 else 0.0 end) as pct_above_ma50,
      avg(case when n200 = 200 and close > ma200 then 1.0 else 0.0 end) as pct_above_ma200,
      avg(case when n200 = 200 and close >= hi252 then 1.0 else 0.0 end) as pct_new_high252,
      avg(case when n200 = 200 and close <= lo252 then 1.0 else 0.0 end) as pct_new_low252,
      sum(amount) as total_amount
    from w group by date order by date
    """
    df = con.execute(q).df()
    if own:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    df["breadth_ma20_z"] = _safe_div(df["pct_above_ma20"] - df["pct_above_ma20"].rolling(60, min_periods=20).mean(),
                                     df["pct_above_ma20"].rolling(60, min_periods=20).std(ddof=0))
    return df


def breadth_baostock(start, end=FREEZE, window_sessions=260, cfg=DEFAULT_CFG) -> pd.DataFrame:
    """baostock 全 A 宽度（含退市股，按上市/退市区间取universe）。较慢，按需调用。"""
    con = sqlite3.connect(f"file:{BAOSTOCK_DB}?mode=ro", uri=True)
    try:
        cal = pd.read_sql_query(
            "select calendar_date from trade_dates where is_trading_day=1 and calendar_date<=? order by 1",
            con, params=(pd.Timestamp(end).strftime("%Y-%m-%d"),))
        cal = pd.to_datetime(cal["calendar_date"]).to_numpy(dtype="datetime64[D]")
        s = pd.Timestamp(start).normalize()
        i0 = int(np.searchsorted(cal, np.datetime64(s)))
        look = pd.Timestamp(cal[max(0, i0 - window_sessions)]).strftime("%Y-%m-%d")
        q = """
        with k as (
          select code, date, close, amount, turn, preclose, is_st
          from stock_kline_d
          where adjustflag=3 and trade_status=1 and date between ? and ? and close > 0
        ), w as (
          select code, date, close, amount, is_st, preclose,
            avg(close) over (partition by code order by date rows between 19 preceding and current row) ma20,
            avg(close) over (partition by code order by date rows between 49 preceding and current row) ma50,
            avg(close) over (partition by code order by date rows between 199 preceding and current row) ma200,
            count(*) over (partition by code order by date rows between 199 preceding and current row) n200,
            max(close) over (partition by code order by date rows between 251 preceding and current row) hi252,
            min(close) over (partition by code order by date rows between 251 preceding and current row) lo252
          from k
        )
        select date,
          count(*) n_stocks,
          sum(case when close > ma20 then 1 else 0 end)*1.0/count(*) pct_above_ma20,
          sum(case when n200>=200 and close > ma50 then 1 else 0 end)*1.0/count(*) pct_above_ma50,
          sum(case when n200=200 and close > ma200 then 1 else 0 end)*1.0/count(*) pct_above_ma200,
          sum(case when n200=200 and close >= hi252 then 1 else 0 end)*1.0/count(*) pct_new_high252,
          sum(case when n200=200 and close <= lo252 then 1 else 0 end)*1.0/count(*) pct_new_low252,
          sum(amount) total_amount
        from w where date >= ? group by date order by date
        """
        df = pd.read_sql_query(q, con, params=(look, pd.Timestamp(end).strftime("%Y-%m-%d"),
                                               pd.Timestamp(start).strftime("%Y-%m-%d")))
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def index_context(start=None, end=FREEZE, codes=("sh.000300", "sh.000001", "sz.399006")):
    src = BaostockSource()
    out = []
    for code in codes:
        df = src.index(code, start=start, end=end)
        if df.empty:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = df["close"].to_numpy(dtype=float)
        a = atr(Bars(df.assign(volume=df["volume"].fillna(1.0), trade_status=1, amount=df["amount"].fillna(0.0)),
                     code=code), 14)
        df["ma20"] = sma(c, 20)
        df["ma60"] = sma(c, 60)
        df["ma200"] = sma(c, 200)
        df["atr_pct"] = _safe_div(a, c)
        df["dd252"] = _safe_div(c, pd.Series(c).rolling(252, min_periods=60).max().to_numpy()) - 1
        df["idx_code"] = code
        out.append(df)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 自检工具

def _naive_pivots(high, low, k):
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    n = len(high)
    ph, pl = [], []
    for i in range(k, n - k):
        h = high[i]
        if (np.isfinite(h) and np.all(np.isfinite(high[i - k:i])) and np.all(np.isfinite(high[i + 1:i + k + 1]))
                and np.all(high[i - k:i] < h) and np.all(high[i + 1:i + k + 1] <= h)):
            ph.append(i)
        lo = low[i]
        if (np.isfinite(lo) and np.all(np.isfinite(low[i - k:i])) and np.all(np.isfinite(low[i + 1:i + k + 1]))
                and np.all(low[i - k:i] > lo) and np.all(low[i + 1:i + k + 1] >= lo)):
            pl.append(i)
    return np.array(ph, int), np.array(pl, int)


def _naive_sma(x, n):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.mean(x[i - n + 1:i + 1])
    return out


def _naive_wilder(x, n):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    valid = np.isfinite(x)
    start = None
    for i in range(len(x) - n + 1):
        if valid[i:i + n].all():
            start = i
            break
    if start is None:
        return out
    out[start + n - 1] = np.mean(x[start:start + n])
    for i in range(start + n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n if np.isfinite(x[i]) else out[i - 1]
    return out


def _synth_bars(n=300, seed=0, start="2019-01-02", trend=0.0004, vol=0.02, gaps=False):
    rng = np.random.default_rng(seed)
    ret = rng.normal(trend, vol, n)
    close = 10 * np.exp(np.cumsum(ret))
    open_ = close * np.exp(rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volu = rng.integers(1e5, 5e6, n).astype(float)
    dates = pd.bdate_range(start, periods=n)
    return bars_from_arrays(dates, open_, high, low, close, volu, volu * close)


def feature_checks(cfg: FeatureConfig = DEFAULT_CFG, real_cases: int = 40, seed: int = 7) -> list:
    checks = []
    rng = np.random.default_rng(seed)

    # 1 平台 pivot：同价只认第一根
    n = 40
    high = np.full(n, 10.0)
    high[:10] = np.linspace(9.0, 9.9, 10)
    high[10:15] = 10.0
    high[15:] = np.linspace(9.9, 9.0, n - 15)
    low = high - 0.5
    phil, plow = detect_pivots(high, low, 3)
    ok = len(phil) == 1 and phil[0] == 10
    checks.append({
        "id": "pivot_plateau", "name": "同价平台枢轴唯一性",
        "status": "pass" if ok else "fail",
        "detail": f"40 根、5 根等高的平台，k=3；检出枢轴高 {phil.tolist()}（期望 [10]）",
        "evidence": {"n_pivots": int(len(phil)), "idx": phil.tolist()},
    })

    # 1b 向量化枢轴与逐根实现完全等价（含大量同价/缺失）
    rngp = np.random.default_rng(seed + 9)
    bad_eq = 0
    for _ in range(120):
        nn = int(rngp.integers(10, 150))
        kk = int(rngp.integers(2, 6))
        sh = int(rngp.integers(1, 6))
        h = rngp.integers(0, sh, nn).astype(float)
        l = np.minimum(rngp.integers(0, sh, nn).astype(float), h)
        a1, a2 = detect_pivots(h, l, kk)
        b1, b2 = _naive_pivots(h, l, kk)
        if not (np.array_equal(a1, b1) and np.array_equal(a2, b2)):
            bad_eq += 1
    checks.append({
        "id": "pivot_vector_equiv", "name": "向量化枢轴与逐根实现等价",
        "status": "pass" if bad_eq == 0 else "fail",
        "detail": f"120 组随机序列（取值 1-5 档，含大量同价），k∈[2,5]；不一致 {bad_eq} 组",
        "evidence": {"cases": 120, "mismatch": int(bad_eq)},
    })

    # 2 枢轴确认滞后严格 = k，且 confirm > pivot
    b = _synth_bars(400, seed=1)
    piv = build_pivots(b, 5)
    lags = [c - i for i, c in zip(piv["high_cal"], piv["high_confirm"])] + \
           [c - i for i, c in zip(piv["low_cal"], piv["low_confirm"])]
    ok = len(lags) > 0 and all(x >= 5 for x in lags)
    checks.append({
        "id": "pivot_confirm_lag", "name": "枢轴确认滞后 >= k",
        "status": "pass" if ok else "fail",
        "detail": f"合成 400 根，k=5，枢轴 {len(lags)} 个，最小确认滞后 {min(lags) if lags else None}",
        "evidence": {"n": len(lags), "min_lag": int(min(lags)) if lags else None,
                     "max_lag": int(max(lags)) if lags else None},
    })

    # 3 结构量不回填：某个“创新高”的枢轴在确认前不得出现在 swing_high
    fm = feature_matrix(b, cfg)
    leak = 0
    tested = 0
    for i, c, px in zip(piv["high_cal"], piv["high_confirm"], piv["high_px"]):
        if i <= 0 or i >= c:
            continue
        if px <= np.nanmax(b.high[:i]):
            continue
        tested += 1
        if np.isfinite(fm["swing_high"].iloc[c - 1]) and abs(fm["swing_high"].iloc[c - 1] - px) < 1e-12:
            leak += 1
        if not np.isfinite(fm["swing_high"].iloc[c]) or abs(fm["swing_high"].iloc[c] - px) > 1e-12:
            leak += 1
    checks.append({
        "id": "pivot_no_backfill", "name": "结构量在确认前不回填",
        "status": "pass" if leak == 0 and tested > 0 else "fail",
        "detail": f"检验创新高枢轴 {tested} 个：确认前不可见、确认当根可见；违反 {leak} 次",
        "evidence": {"tested": int(tested), "violations": int(leak),
                     "n_high_pivots": len(piv["high_cal"])},
    })

    # 4 起点不完整周被标记并剔除
    cal = Calendar(pd.bdate_range("2019-01-01", "2019-12-31"))
    d0 = pd.Timestamp("2019-03-06")  # 周三
    sub = b.df[(b.df["date"] >= d0)].head(60).copy()
    sub["date"] = pd.bdate_range(d0, periods=len(sub))
    bsub = Bars(sub, code="000001.SZ")
    w = aggregate(bsub, "W", cal, cfg)
    wm = w.meta
    ok = bool(wm["partial_start"].iloc[0]) and not bool(wm["partial_start"].iloc[1:].any())
    last_ok = pd.Timestamp(wm["date"].iloc[-1]) <= sub["date"].max()
    checks.append({
        "id": "week_partial_start", "name": "起点不完整周标记",
        "status": "pass" if (ok and last_ok) else "fail",
        "detail": f"数据从周三 2019-03-06 起；首周 partial_start={bool(wm['partial_start'].iloc[0])}，"
                  f"后续 partial_start 数={int(wm['partial_start'].iloc[1:].sum())}，末周 period_end<=数据末日={last_ok}",
        "evidence": {"n_weeks": int(len(wm)), "first_period_start": str(wm["period_start"].iloc[0].date()),
                     "first_date": str(wm["date"].iloc[0].date())},
    })

    # 5 未完成周/月不输出
    d_end = pd.Timestamp("2019-06-05")  # 周三
    sub2 = b.df[(b.df["date"] <= d_end)].copy()
    b2 = Bars(sub2, code="000001.SZ")
    w2 = aggregate(b2, "W", cal, cfg).meta
    m2 = aggregate(b2, "M", cal, cfg).meta
    ok = (pd.Timestamp(w2["date"].iloc[-1]) <= d_end) and (pd.Timestamp(m2["date"].iloc[-1]) <= d_end)
    checks.append({
        "id": "no_unfinished_period", "name": "未完成周/月不输出",
        "status": "pass" if ok else "fail",
        "detail": f"截断至周三 {d_end.date()}；末周 {w2['date'].iloc[-1].date()}，末月 {m2['date'].iloc[-1].date()}",
        "evidence": {"n_weeks": int(len(w2)), "n_months": int(len(m2))},
    })

    # 6 指标对照
    x = b.close
    d1 = np.nanmax(np.abs(sma(x, 20) - _naive_sma(x, 20)))
    d2 = np.nanmax(np.abs(_wilder(np.arange(1, 301, dtype=float), 14) -
                          _naive_wilder(np.arange(1, 301, dtype=float), 14)))
    d3 = np.nanmax(np.abs(ema(x, 21) - _naive_ema(x, 21)))
    ok = max(d1, d2, d3) < 1e-9
    checks.append({
        "id": "indicator_reference", "name": "指标对照独立实现",
        "status": "pass" if ok else "fail",
        "detail": f"SMA20 max|diff|={d1:.2e}，Wilder14 max|diff|={d2:.2e}，EMA21 max|diff|={d3:.2e}",
        "evidence": {"sma": float(d1), "wilder": float(d2), "ema": float(d3)},
    })

    # 7 指标预热期
    exp = {"ma20": 19, "ma200": 199, "atr14": 14, "rsi14": 14, "adx14": 27}
    got = {}
    for name, _ in exp.items():
        s = fm[name].to_numpy()
        got[name] = int(np.flatnonzero(np.isfinite(s))[0]) if np.isfinite(s).any() else -1
    ok = all(got[name] == v for name, v in exp.items())
    checks.append({
        "id": "warmup_index", "name": "指标预热起点",
        "status": "pass" if ok else "fail",
        "detail": f"期望 {exp}，实际 {got}",
        "evidence": got,
    })

    # 8 停牌不污染枢轴/量能
    df = _synth_bars(120, seed=3).df
    df.loc[40:52, ["open", "high", "low", "close"]] = df.loc[39, "close"]
    df.loc[40:52, "volume"] = np.nan
    df.loc[40:52, "trade_status"] = 0
    bs = Bars(df, code="000001.SZ")
    pv = build_pivots(bs, 5)
    inside = [i for i in pv["high_cal"] if 40 <= i <= 52] + [i for i in pv["low_cal"] if 40 <= i <= 52]
    fm2 = feature_matrix(bs, cfg)
    ok = len(inside) == 0 and np.isfinite(fm2["atr14"].iloc[80])
    checks.append({
        "id": "suspension_not_pivot", "name": "停牌段不生成枢轴",
        "status": "pass" if ok else "fail",
        "detail": f"人为停牌 13 根（40-52），段内枢轴数 {len(inside)}，停牌后 ATR 有效={bool(np.isfinite(fm2['atr14'].iloc[80]))}",
        "evidence": {"inside": [int(i) for i in inside]},
    })

    # 9 涨跌停价格
    lf = limit_flags(bars_from_arrays(pd.bdate_range("2024-01-01", periods=2),
                                      [10, 11], [10, 11], [10, 11], [10, 11], [1e6, 1e6],
                                      preclose=[10, 10], code="600000.SH"))
    ok = bool(lf["at_limit_up"][1]) and not bool(lf["at_limit_up"][0])
    checks.append({
        "id": "limit_price", "name": "涨停价判定（主板 10%）",
        "status": "pass" if ok else "fail",
        "detail": f"preclose=10 收盘=11 → 涨停 {bool(lf['at_limit_up'][1])}；收盘=10 → 涨停 {bool(lf['at_limit_up'][0])}",
        "evidence": {"lim": lf["limit_pct"].tolist()},
    })

    # 10 真实数据截断重算
    src = BaostockSource()
    syms = src.symbols()
    rng2 = np.random.default_rng(seed + 1)
    pool = syms[(syms["ipo_date"] <= "2010-01-01") & ((syms["out_date"].isna()) | (syms["out_date"] >= "2020-01-01"))]
    pool = pool.sample(min(real_cases, len(pool)), random_state=seed)["code"].tolist()
    maxdiff, ncase, fails, rows = 0.0, 0, 0, []
    cols = ["ma20", "ma200", "atr14", "rsi14", "adx14", "don_high_prev", "stop_pct", "struct_score", "don_pos"]
    for code in pool:
        full = src.load(code, adjust="hfq", with_raw=True)
        if len(full) < 600:
            continue
        f_full = feature_matrix(Bars(full, code=to_ths_code(code)), cfg)
        for _ in range(2):
            j = int(rng2.integers(400, len(full) - 2))
            cut = full.iloc[j]["date"]
            part = full[full["date"] <= cut].reset_index(drop=True)
            f_part = feature_matrix(Bars(part, code=to_ths_code(code)), cfg)
            if len(f_part) != j + 1:
                fails += 1
                continue
            a = f_full.iloc[:j + 1][cols].to_numpy(dtype=float)
            bb = f_part[cols].to_numpy(dtype=float)
            m = np.nanmax(np.abs(np.nan_to_num(a, nan=-9e9) - np.nan_to_num(bb, nan=-9e9))) if a.size else 0
            nanmis = int((np.isnan(a) != np.isnan(bb)).sum())
            maxdiff = max(maxdiff, float(m) if np.isfinite(m) else 0.0)
            ncase += 1
            if nanmis or (np.isfinite(m) and m > 1e-9):
                fails += 1
                rows.append({"code": code, "cut": str(cut.date()), "maxdiff": float(m), "nan_mismatch": nanmis})
    checks.append({
        "id": "truncation_real", "name": "真实数据截断重算一致（日线）",
        "status": "pass" if fails == 0 and ncase > 0 else "fail",
        "detail": f"{len(pool)} 只股票、{ncase} 个随机截断点，比较 {len(cols)} 列；"
                  f"max|diff|={maxdiff:.3e}，失败 {fails}",
        "evidence": {"cases": ncase, "max_abs_diff": maxdiff, "failures": fails, "examples": rows[:5]},
    })

    # 11 未来扰动：截断点之后替换为垃圾数据，截断点特征不变
    pitch = 0
    ncase2 = 0
    for code in pool[:8]:
        full = src.load(code, adjust="hfq")
        if len(full) < 800:
            continue
        j = int(len(full) * 0.7)
        cut = full.iloc[j]["date"]
        base = feature_matrix(Bars(full[full["date"] <= cut], code=to_ths_code(code)), cfg).iloc[-1]
        pert = full.copy()
        fut = pert.index > j
        rngp = np.random.default_rng(11 + j)
        for c in ("open", "high", "low", "close"):
            pert.loc[fut, c] = pert.loc[fut, c] * rngp.uniform(0.3, 3.0, int(fut.sum()))
        pert.loc[fut, "volume"] = rngp.uniform(1e3, 1e9, int(fut.sum()))
        pert.loc[fut, "amount"] = pert.loc[fut, "volume"] * pert.loc[fut, "close"]
        f2 = feature_matrix(Bars(pert, code=to_ths_code(code)), cfg)
        row = f2[f2["date"] == cut].iloc[0]
        for c in cols:
            a, bb = float(base[c]), float(row[c])
            if (np.isnan(a) != np.isnan(bb)) or (np.isfinite(a) and abs(a - bb) > 1e-9):
                pitch += 1
        ncase2 += 1
    checks.append({
        "id": "future_perturbation", "name": "未来数据扰动不影响截断点",
        "status": "pass" if pitch == 0 and ncase2 > 0 else "fail",
        "detail": f"{ncase2} 只股票，把截断点之后的 OHLC 乘以 0.3~3.0 随机倍数并替换成交量；不一致 {pitch} 处",
        "evidence": {"cases": ncase2, "mismatches": int(pitch)},
    })

    # 12 复权桥接：hfq/price_scale 必须还原原始价；并量化“原始价做结构”的失真
    code = "sh.600519"
    hfq = src.load(code, adjust="hfq", with_raw=True)
    raw = src.load(code, adjust="raw")
    bridge, ok_bridge = np.nan, False
    pivot_drift = None
    if len(hfq) and len(raw):
        fh = feature_matrix(Bars(hfq, code="600519.SH"), cfg)
        fr = feature_matrix(Bars(raw, code="600519.SH"), cfg)
        ps = fh["price_scale"].to_numpy()
        recon = fh["close"].to_numpy() / ps
        bridge = float(np.nanmax(np.abs(recon - fr["close"].to_numpy())))
        ok_bridge = bridge < 1e-6
        ph = build_pivots(Bars(hfq, code="600519.SH"), cfg.pivot_k)
        pr = build_pivots(Bars(raw, code="600519.SH"), cfg.pivot_k)
        hs, hr = set(ph["high_cal"]), set(pr["high_cal"])
        pivot_drift = {"hfq_high_pivots": len(hs), "raw_high_pivots": len(hr),
                       "shared": len(hs & hr), "jaccard": round(len(hs & hr) / max(1, len(hs | hr)), 4)}
    checks.append({
        "id": "adjust_bridge", "name": "后复权/原始价桥接与复权失真度",
        "status": "pass" if ok_bridge else "fail",
        "detail": f"{code}: close/price_scale 还原原始价 max|diff|={bridge:.3e}；"
                  f"hfq 与 raw 的枢轴高重合度 Jaccard={pivot_drift['jaccard'] if pivot_drift else None}"
                  "（raw 在除权日产生假跳空，故不要求枢轴一致，仅作证据）",
        "evidence": {"bridge_max_abs_diff": float(bridge) if np.isfinite(bridge) else None,
                     "rows": int(len(hfq)), "pivot_drift": pivot_drift},
    })

    # 13 DuckDB 起点（2016-10-13 周四）不得把首周/首月当成完整周期
    try:
        dsrc = DuckDBSource()
        cal_real = Calendar.from_baostock()
        dd = dsrc.load("600519.SH", adjust="hfq", start="2016-10-13")
        bd = Bars(dd, code="600519.SH")
        wd = aggregate(bd, "W", cal_real, cfg).meta
        md = aggregate(bd, "M", cal_real, cfg).meta
        fw = feature_matrix(bd, cfg, cal_real, "W")
        fm2 = feature_matrix(bd, cfg, cal_real, "M")
        ok = (bool(wd["partial_start"].iloc[0]) and bool(md["partial_start"].iloc[0])
              and pd.Timestamp(fw["date"].iloc[0]) > pd.Timestamp(wd["date"].iloc[0])
              and pd.Timestamp(fm2["date"].iloc[0]) > pd.Timestamp(md["date"].iloc[0]))
        checks.append({
            "id": "duckdb_partial_start", "name": "DuckDB 起点不完整周/月剔除",
            "status": "pass" if ok else "fail",
            "detail": f"起点 2016-10-13(周四)：首周 {wd['date'].iloc[0].date()} partial="
                      f"{bool(wd['partial_start'].iloc[0])} → 周特征首日 {fw['date'].iloc[0].date()}；"
                      f"首月 {md['date'].iloc[0].date()} partial={bool(md['partial_start'].iloc[0])} "
                      f"→ 月特征首日 {fm2['date'].iloc[0].date()}",
            "evidence": {"rows_daily": int(len(bd)), "first_week": str(wd["date"].iloc[0].date()),
                         "first_week_feature": str(fw["date"].iloc[0].date()),
                         "first_month": str(md["date"].iloc[0].date()),
                         "first_month_feature": str(fm2["date"].iloc[0].date())},
        })
    except Exception as e:
        checks.append({"id": "duckdb_partial_start", "name": "DuckDB 起点不完整周/月剔除",
                       "status": "skip", "detail": f"DuckDB 不可用：{e}", "evidence": {}})

    return checks


def _naive_ema(x, n):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    out[n - 1] = np.mean(x[:n])
    a = 2 / (n + 1)
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


# ---------------------------------------------------------------- CLI

def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(description="核心特征因果自检")
    ap.add_argument("--out", default=str(EVIDENCE / "特征因果自检.json"))
    ap.add_argument("--cases", type=int, default=40)
    args = ap.parse_args(argv)
    t0 = time.time()
    cal = Calendar.from_baostock()
    checks = feature_checks(real_cases=args.cases)
    payload = {
        "meta": {
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "freeze": str(FREEZE.date()),
            "scope": "features_only",
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "calendar_sessions": int(len(cal)),
            "calendar_window": [str(pd.Timestamp(cal.dates[0]).date()), str(pd.Timestamp(cal.dates[-1]).date())],
            "files": {"核心特征.py": _sha(Path(__file__))},
            "config": asdict(DEFAULT_CFG),
            "runtime_sec": round(time.time() - t0, 2),
        },
        "checks": checks,
        "summary": {"total": len(checks), "pass": sum(c["status"] == "pass" for c in checks),
                    "fail": sum(c["status"] != "pass" for c in checks)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    for c in checks:
        print(f"[{c['status']}] {c['id']}: {c['detail']}")
    print(f"-> {out}")
    return 0 if payload["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
