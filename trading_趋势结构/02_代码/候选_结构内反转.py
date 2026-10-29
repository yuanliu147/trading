#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""结构内短期温和反转（第二阶段候选族，独立实现）

问题
----
120/250 日长期结构未破坏（TREND）或底部横盘（BASE）时，3/5/10 日的**温和**回撤
（−2% 到 −12%）是否提供可覆盖真实费用的短期反转边际？与“极端接飞刀”
（ret10 ≤ −20%）和“极端赢家”（ret5 ≥ +8%）严格区分。

纪律
----
* 只读 `01_数据/frozen/frozen.duckdb`（冻结 2026-10-26）；面板 2006-01-01..2020-12-31。
* 2021+ 硬护栏：`_assert_window` 在每个收益入口断言日期上界；本文件自建面板，
  SQL 显式 `date <= 2020-12-31`，绩效走 `统计验证.guard_performance`。
* 价格两套：**A 空间** `A_t = factor_t / raw_close_t`（按每股首个有效日归一），
  `adjusted_OHLC = raw_OHLC × A_t`，故 `adjusted_close ≡ factor`，其日收益 == 官方 `pct_chg`；
  **raw** 用于成交/整股/涨跌停/费用。`hfq = raw × factor` 禁用。
* 信号收盘生成、次一交易日开盘成交；T+1、整股、真实费用、滑点、停牌/涨跌停、退市。
* 本文件只写 `02_代码/候选_结构内反转.py` 与 `03_结果/候选_结构内反转/`。

用法
----
python3 02_代码/候选_结构内反转.py selfcheck     # 口径/因果恒等式
python3 02_代码/候选_结构内反转.py events        # 事件层 20 配置 + 分位 + 分层
python3 02_代码/候选_结构内反转.py account       # 账户层 27 配置 × dev/val
python3 02_代码/候选_结构内反转.py stats         # DSR/PBO/聚类汇总
python3 02_代码/候选_结构内反转.py all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "02_代码"
OUT = ROOT / "03_结果" / "候选_结构内反转"
DB = ROOT / "01_数据" / "frozen" / "frozen.duckdb"
PREREG = OUT / "00_预注册_结构内反转.md"

PANEL_START = "2006-01-01"
ANALYSIS_END = "2020-12-31"
HOLDOUT_START = "2021-01-01"
DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")
FULL = ("2008-01-01", "2020-12-31")
WINDOWS = {"dev": DEV, "val": VAL, "full": FULL}

BOARD = "主板"
N_LOT = 100
MIN_PRICE = 2.0
MIN_ADV20 = 3.0e7
MIN_LIST_TD = 120
MIN_LIST_CAL = 250
DELIST_BLOCK_TD = 30
EVENT_NOTIONAL = 100_000.0     # 事件层成本估算的名义下单额（= 100 万 / 10 仓）
MAX_PERSIST_DAYS = 10

# 预注册常量（不得因结果修改）
K_LIST = (3, 5, 10)
K_MAIN = 5
H_MAIN = 10
BAND_MAIN = (-0.12, -0.02)
KNIFE_LEVEL = -0.20
WINNER_LEVEL = 0.08
MAX_DAY_DROP = -0.095
STATES = ("TREND", "BASE", "NOTREND", "BROKEN", "OTHER")


# ---------------------------------------------------------------------------
# 模块加载
# ---------------------------------------------------------------------------
# 必须用标准 import，保证与统一引擎共用同一个模块实例（Bar/FeeConfig 的 isinstance 依赖）
sys.path.insert(0, str(CODE))
import 费用与交易约束 as FEE        # noqa: E402
import 账户撮合 as ACC              # noqa: E402
import 统计验证 as STAT             # noqa: E402
import 统一数据适配 as UDA          # noqa: E402
import 统一回测 as UB               # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prereg_sha() -> str:
    return sha256_file(PREREG) if PREREG.exists() else ""


def _assert_window(start, end, where: str) -> None:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e >= pd.Timestamp(HOLDOUT_START):
        raise RuntimeError(f"[2021+护栏] {where}: 区间 {start}~{end} 触及留出期")
    if s < pd.Timestamp(PANEL_START):
        raise RuntimeError(f"[面板护栏] {where}: 起点 {start} 早于 {PANEL_START}")


def json_dump(obj, path: Path) -> None:
    def conv(x):
        if isinstance(x, (np.floating, float)):
            v = float(x)
            return None if not math.isfinite(v) else v
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, (date, pd.Timestamp)):
            return str(x)
        if isinstance(x, pd.DataFrame):
            return {"__df__": f"{x.shape[0]}x{x.shape[1]}", "cols": [str(c) for c in x.columns]}
        if isinstance(x, pd.Series):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, np.ndarray):
            return conv(x.tolist())
        return str(x)

    def walk(o):
        if isinstance(o, dict):
            return {str(k): walk(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v) for v in o]
        return conv(o)

    path.write_text(json.dumps(walk(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def to_date(x) -> date:
    if isinstance(x, date) and not isinstance(x, pd.Timestamp):
        return x
    return pd.Timestamp(x).date()


# ---------------------------------------------------------------------------
# 面板
# ---------------------------------------------------------------------------
@dataclass
class Panel:
    dates: list
    codes: list
    day_idx: dict
    code_idx: dict
    raw: dict
    a: dict
    flags: dict
    size: np.ndarray
    adv20: np.ndarray
    ca_factor: np.ndarray
    events: object
    ipo_date: list
    delist_date: list
    hs300: np.ndarray
    gate_hs300: np.ndarray
    day_np: np.ndarray

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_codes(self) -> int:
        return len(self.codes)

    def window_index(self, start, end) -> tuple[int, int]:
        import bisect
        s, e = to_date(start), to_date(end)
        i0 = bisect.bisect_left(self.dates, s)
        i1 = bisect.bisect_right(self.dates, e) - 1
        if i0 > i1 or i0 >= self.n_days or i1 < 0:
            raise ValueError(f"区间 {start}~{end} 与面板无交集")
        return i0, i1

    def bars_on(self, i: int, codes) -> dict:
        d = self.dates[i]
        out = {}
        o, h, l, c = (self.raw[x][i] for x in ("open", "high", "low", "close"))
        pc = self.raw["preclose"][i]
        vol, amt = self.raw["volume"][i], self.raw["amount"][i]
        st = self.flags["is_st"][i]
        susp = self.flags["suspended"][i]
        for code in codes:
            j = self.code_idx.get(code)
            if j is None:
                continue
            if not (np.isfinite(o[j]) and np.isfinite(h[j]) and np.isfinite(l[j])
                    and np.isfinite(c[j]) and c[j] > 0):
                continue
            try:
                out[code] = ACC.Bar(
                    code=code, date=d, open=float(o[j]), high=float(h[j]),
                    low=float(l[j]), close=float(c[j]),
                    volume=0.0 if not np.isfinite(vol[j]) else float(vol[j]),
                    amount=None if not np.isfinite(amt[j]) else float(amt[j]),
                    prev_close=None if not np.isfinite(pc[j]) else float(pc[j]),
                    prev_close_is_ref=True, is_st=bool(st[j]),
                    trade_status=0 if susp[j] else 1,
                )
            except Exception:
                continue
        return out


def load_panel(start: str = PANEL_START, end: str = ANALYSIS_END) -> Panel:
    _assert_window(start, end, "load_panel")
    con = duckdb.connect(str(DB), read_only=True)
    try:
        dates = [pd.Timestamp(x).date() for x in con.execute(
            "SELECT date FROM calendar WHERE is_trading_day=1 AND date BETWEEN ? AND ? ORDER BY date",
            [start, end]).df()["date"]]
        cols = ("symbol,date,open,high,low,close,preclose,volume,amount,factor,float_mktcap,"
                "is_suspended,is_st,td_since_list,limit_up_px,limit_down_px")
        df = con.execute(
            f"SELECT {cols} FROM daily WHERE board=? AND date BETWEEN ? AND ?",
            [BOARD, start, end]).df()
        meta = con.execute(
            "SELECT symbol, ipo_date, delist_date FROM symbols WHERE board=?", [BOARD]).df()
        idx = con.execute(
            "SELECT date, close FROM index_daily WHERE symbol='000300.SH' "
            "AND date BETWEEN ? AND ? ORDER BY date", [start, end]).df()
    finally:
        con.close()

    df = df[df["date"] <= pd.Timestamp(end)]
    codes = sorted(df["symbol"].unique())
    shape = (len(dates), len(codes))
    dmap = {d: i for i, d in enumerate(dates)}
    cmap = {c: j for j, c in enumerate(codes)}
    di = df["date"].map(dmap).to_numpy(np.int32)
    ci = df["symbol"].map(cmap).to_numpy(np.int32)
    if len(np.unique(di.astype(np.int64) * len(codes) + ci)) != len(df):
        raise ValueError("daily 存在重复 (date,symbol)")

    def num(c):
        m = np.full(shape, np.nan, dtype=np.float64)
        m[di, ci] = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
        return m

    raw = {c: num(c) for c in ("open", "high", "low", "close", "preclose", "volume",
                               "amount", "factor", "float_mktcap", "td_since_list",
                               "limit_up_px", "limit_down_px")}
    flags = {}
    for c in ("is_suspended", "is_st"):
        m = np.zeros(shape, dtype=bool)
        m[di, ci] = df[c].fillna(False).to_numpy(dtype=bool)
        flags[c] = m

    close = raw["close"]
    listed = np.isfinite(close) & (close > 0)
    flags["listed"] = listed
    susp = flags["is_suspended"] | ~listed
    flags["suspended"] = susp
    prev_susp = np.vstack([np.ones((1, len(codes)), dtype=bool), susp[:-1]])
    prev_listed = np.vstack([np.zeros((1, len(codes)), dtype=bool), listed[:-1]])
    flags["resume"] = prev_susp | ~prev_listed

    # A 空间：A_t = factor_t / raw_close_t，按每股首个有效值归一
    factor = raw["factor"]
    with np.errstate(invalid="ignore", divide="ignore"):
        A = factor / close
    first = np.full(len(codes), np.nan)
    for j in range(len(codes)):
        v = A[:, j]
        k = np.flatnonzero(np.isfinite(v))
        if len(k):
            first[j] = v[k[0]]
    A = A / first[None, :]
    a = {"close": factor / first[None, :]}
    for f in ("open", "high", "low"):
        a[f] = raw[f] * A
    a["ret"] = np.full(shape, np.nan)
    a["ret"][1:] = a["close"][1:] / a["close"][:-1] - 1.0

    adv20 = pd.DataFrame(raw["amount"]).rolling(20, min_periods=10).mean().to_numpy()

    ipo = meta.set_index("symbol")["ipo_date"].reindex(codes)
    dls = meta.set_index("symbol")["delist_date"].reindex(codes)
    ipo_date = [None if pd.isna(x) else pd.Timestamp(x).date() for x in ipo]
    delist_date = [None if pd.isna(x) else pd.Timestamp(x).date() for x in dls]

    days = np.array([(d - date(1990, 1, 1)).days for d in dates])
    ipo_days = np.array([np.nan if x is None else (x - date(1990, 1, 1)).days
                         for x in ipo_date], dtype=np.float64)
    days_since_ipo = days[:, None] - ipo_days[None, :]

    import bisect
    delist_window = np.zeros(shape, dtype=bool)
    for j, dd in enumerate(delist_date):
        if dd is None:
            continue
        k = int(bisect.bisect_left(dates, dd))
        if k < len(dates):
            delist_window[max(0, k - DELIST_BLOCK_TD):k + 1, j] = True
    flags["delist_window"] = delist_window

    elig = (listed & ~susp & ~flags["is_st"]
            & (np.nan_to_num(raw["td_since_list"], nan=0.0) >= MIN_LIST_TD)
            & (np.nan_to_num(close, nan=0.0) >= MIN_PRICE)
            & (np.nan_to_num(adv20, nan=0.0) >= MIN_ADV20)
            & (days_since_ipo >= MIN_LIST_CAL)
            & ~flags["resume"] & ~delist_window
            & np.isfinite(a["close"]))
    flags["elig"] = elig

    hs = pd.Series(np.nan, index=pd.Index(dates, name="date"))
    if len(idx):
        s = pd.Series(idx["close"].to_numpy(float), index=pd.to_datetime(idx["date"]))
        s.index = [t.date() for t in s.index]
        hs = s.reindex(hs.index).ffill()
    hs300 = hs.to_numpy(float)
    ma200 = pd.Series(hs300).rolling(200, min_periods=150).mean().to_numpy()
    gate_hs300 = np.nan_to_num(hs300, nan=0.0) > np.nan_to_num(ma200, nan=np.inf)

    events = UDA.EventBook().load(start, end)
    ca_factor = np.cumprod(events.ratio_frame(dates, codes), axis=0)

    return Panel(dates=dates, codes=codes, day_idx=dmap, code_idx=cmap, raw=raw, a=a,
                 flags=flags, size=raw["float_mktcap"], adv20=adv20,
                 ca_factor=ca_factor, events=events, ipo_date=ipo_date,
                 delist_date=delist_date, hs300=hs300, gate_hs300=gate_hs300,
                 day_np=np.array(dates, dtype="datetime64[D]"))


# ---------------------------------------------------------------------------
# 特征
# ---------------------------------------------------------------------------
def _roll(mat: np.ndarray, win: int, how: str = "mean", minp: int | None = None) -> np.ndarray:
    df = pd.DataFrame(mat)
    r = df.rolling(win, min_periods=minp if minp is not None else win)
    out = getattr(r, how)().to_numpy(dtype=np.float64)
    del df, r
    return out


@dataclass
class Features:
    c: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    ret: np.ndarray
    ma5: np.ndarray
    ma20: np.ndarray
    ma60: np.ndarray
    ma120: np.ndarray
    ma250: np.ndarray
    slope250: np.ndarray
    dd250: np.ndarray
    r250: np.ndarray
    width250: np.ndarray
    pul: dict
    pul5_min3: np.ndarray
    worst_k: dict
    atr14: np.ndarray
    pul5_q: np.ndarray
    size_q: np.ndarray
    liq_q: np.ndarray
    states: dict
    struct: np.ndarray


def build_features(p: Panel) -> Features:
    c, o, h, l = p.a["close"], p.a["open"], p.a["high"], p.a["low"]
    ret = p.a["ret"]
    ma5 = _roll(c, 5)
    ma20 = _roll(c, 20)
    ma60 = _roll(c, 60)
    ma120 = _roll(c, 120)
    ma250 = _roll(c, 250)
    ma250_60 = np.full_like(ma250, np.nan)
    ma250_60[60:] = ma250[:-60]
    with np.errstate(invalid="ignore", divide="ignore"):
        slope250 = ma250 / ma250_60 - 1.0
        days_idx = np.arange(c.shape[0])
        lag = np.clip(days_idx - 250, 0, None)
        r250 = np.where((days_idx >= 250)[:, None], c / c[lag] - 1.0, np.nan)
    rmax250 = _roll(c, 250, "max")
    rmin250 = _roll(c, 250, "min")
    with np.errstate(invalid="ignore", divide="ignore"):
        dd250 = c / rmax250 - 1.0
        width250 = (rmax250 - rmin250) / rmin250
    pul, worst = {}, {}
    for k in K_LIST:
        idx = np.arange(c.shape[0])
        lag = np.clip(idx - k, 0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            pul[k] = np.where((idx >= k)[:, None], c / c[lag] - 1.0, np.nan)
        worst[k] = _roll(ret, k, "min", minp=k)

    prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    atr14 = pd.DataFrame(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()

    finite = (np.isfinite(c) & np.isfinite(ma250) & np.isfinite(ma120)
              & np.isfinite(slope250) & np.isfinite(dd250) & np.isfinite(r250)
              & np.isfinite(width250))
    above = c > ma250
    broken = finite & ((dd250 <= -0.30) | ((~above) & (slope250 < -0.05)))
    trend = finite & above & (slope250 > 0) & (ma120 > ma250) & (dd250 >= -0.20)
    base = finite & (np.abs(r250) <= 0.25) & (np.abs(c / ma250 - 1.0) <= 0.12) \
        & (width250 <= 0.60) & (dd250 >= -0.25) & (slope250 >= -0.05)
    notrend = finite & (~above) & ~broken
    states = {}
    states["BROKEN"] = broken
    states["TREND"] = trend & ~broken
    states["BASE"] = base & ~broken & ~trend
    states["NOTREND"] = notrend & ~broken & ~trend & ~base
    states["OTHER"] = finite & ~(broken | trend | base | notrend)
    struct = states["TREND"] | states["BASE"]

    p5 = pul[5]
    pul5_min3 = p5.copy()
    for sh in (1, 2):
        shifted = np.full_like(p5, np.nan)
        shifted[sh:] = p5[:-sh]
        with np.errstate(invalid="ignore"):
            pul5_min3 = np.fmin(pul5_min3, shifted)

    pul5_q = np.full_like(c, np.nan)
    for i in range(c.shape[0]):
        m = struct[i] & np.isfinite(pul[5][i])
        if m.sum() < 30:
            continue
        v = pul[5][i, m]
        order = np.argsort(np.argsort(v, kind="stable"), kind="stable")
        pul5_q[i, m] = (order + 0.5) / len(v)

    def xs_q(mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.full(mat.shape, -1, dtype=np.int8)
        for i in range(mat.shape[0]):
            m = mask[i] & np.isfinite(mat[i])
            n = int(m.sum())
            if n < 25:
                continue
            v = mat[i, m]
            order = np.argsort(np.argsort(v, kind="stable"), kind="stable")
            out[i, m] = np.minimum((order * 5) // n, 4).astype(np.int8)
        return out

    size_q = xs_q(p.size, p.flags["elig"])
    liq_q = xs_q(p.adv20, p.flags["elig"])

    return Features(c=c, o=o, h=h, l=l, ret=ret, ma5=ma5, ma20=ma20, ma60=ma60,
                    ma120=ma120, ma250=ma250, slope250=slope250, dd250=dd250,
                    r250=r250, width250=width250, pul=pul, pul5_min3=pul5_min3,
                    worst_k=worst,
                    atr14=atr14, pul5_q=pul5_q, size_q=size_q, liq_q=liq_q,
                    states=states, struct=struct)


# ---------------------------------------------------------------------------
# 信号
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    pid: str
    state: str = "STRUCT"
    k: int = K_MAIN
    band: tuple | None = BAND_MAIN
    entry: str = "E1"
    quant: tuple | None = None
    note: str = ""


POLICIES = [
    Policy("P01_STRUCT_K5_E1", "STRUCT", 5, BAND_MAIN, "E1", None, "主候选"),
    Policy("P02_STRUCT_K5_E0", "STRUCT", 5, BAND_MAIN, "E0", None, "不确认"),
    Policy("P03_STRUCT_K5_ANY", "STRUCT", 5, BAND_MAIN, "E_ANY", None, "入场消融"),
    Policy("P04_TREND_K5_E1", "TREND", 5, BAND_MAIN, "E1", None, "仅上行"),
    Policy("P05_BASE_K5_E1", "BASE", 5, BAND_MAIN, "E1", None, "仅底部横盘"),
    Policy("P06_NOTREND_K5_E1", "NOTREND", 5, BAND_MAIN, "E1", None, "非趋势反例"),
    Policy("P07_BROKEN_K5_E1", "BROKEN", 5, BAND_MAIN, "E1", None, "结构破坏反例"),
    Policy("P08_STRUCT_K3_E1", "STRUCT", 3, BAND_MAIN, "E1", None, "k=3"),
    Policy("P09_STRUCT_K10_E1", "STRUCT", 10, BAND_MAIN, "E1", None, "k=10"),
    Policy("P10_STRUCT_K5_E1_B8", "STRUCT", 5, (-0.08, -0.02), "E1", None, "浅带"),
    Policy("P11_STRUCT_K5_E1_B15", "STRUCT", 5, (-0.15, -0.03), "E1", None, "深带"),
    Policy("P12_STRUCT_K5_E1_B20", "STRUCT", 5, (-0.20, -0.02), "E1", None, "深带跨刀"),
    Policy("P13_STRUCT_K5_E1_H3", "STRUCT", 5, BAND_MAIN, "E1", None, "H=3"),
    Policy("P14_STRUCT_K5_E1_H5", "STRUCT", 5, BAND_MAIN, "E1", None, "H=5"),
    Policy("P15_STRUCT_K5_E1_H20", "STRUCT", 5, BAND_MAIN, "E1", None, "H=20"),
    Policy("P16_STRUCT_K5_E1_H40", "STRUCT", 5, BAND_MAIN, "E1", None, "H=40"),
    Policy("P17_STRUCT_KNIFE", "STRUCT", 10, None, "X_KNIFE", None, "极端接飞刀"),
    Policy("P18_TREND_WINNER", "TREND", 5, None, "X_WINNER", None, "极端赢家"),
    Policy("P19_ANYSTATE_K5_E1", "ANYSTATE", 5, BAND_MAIN, "E1", None, "去结构过滤"),
    Policy("P20_STRUCT_Q0525_E1", "STRUCT", 5, None, "E1", (0.05, 0.25), "回撤分位带"),
    Policy("P21_STRUCT_Q0010_E1", "STRUCT", 5, None, "E1", (0.0, 0.10), "追加:深分位"),
]
POLICY_BY_ID = {p.pid: p for p in POLICIES}
POLICY_HOLD = {"P13_STRUCT_K5_E1_H3": 3, "P14_STRUCT_K5_E1_H5": 5,
               "P15_STRUCT_K5_E1_H20": 20, "P16_STRUCT_K5_E1_H40": 40}


def state_mask(f: Features, name: str) -> np.ndarray:
    if name == "STRUCT":
        return f.struct
    if name == "ANYSTATE":
        return np.isfinite(f.ma250)
    return f.states[name]


def _mild(f: Features, k: int, band: tuple) -> np.ndarray:
    lo, hi = band
    return ((f.pul[k] >= lo) & (f.pul[k] <= hi)
            & (np.nan_to_num(f.worst_k[k], nan=-1.0) > MAX_DAY_DROP))


def policy_mask(f: Features, pol: Policy) -> np.ndarray:
    st = state_mask(f, pol.state)
    prev = np.vstack([np.full((1, f.c.shape[1]), np.nan), f.c[:-1]])
    up = (f.c > prev) & (f.c > f.o) & (f.c > f.ma5)
    if pol.entry == "X_KNIFE":
        m = (f.pul[10] <= KNIFE_LEVEL) & st & up
        m = m & np.isfinite(f.ma5) & np.isfinite(f.ma250)
    elif pol.entry == "X_WINNER":
        m = (f.pul[5] >= WINNER_LEVEL) & st & (f.c > prev)
        m = m & np.isfinite(f.ma5) & np.isfinite(f.ma250)
    else:
        if pol.quant is not None:
            lo, hi = pol.quant
            base = st & (f.pul5_q > lo) & (f.pul5_q <= hi)
        else:
            base = _mild(f, pol.k, pol.band) & st
        b1 = np.vstack([np.zeros((1, f.c.shape[1]), dtype=bool), base[:-1]])
        b2 = np.vstack([np.zeros((2, f.c.shape[1]), dtype=bool), base[:-2]])
        if pol.entry == "E0":
            m = base & (f.c < f.ma5) & (f.c < prev)
        elif pol.entry == "E1":
            m = (b1 | b2) & up
        elif pol.entry == "E_ANY":
            m = base | ((b1 | b2) & up)
        else:
            raise ValueError(pol.entry)
        m = m & np.isfinite(f.ma5) & np.isfinite(f.ma250)
        m = m & np.isfinite(f.pul[pol.k if pol.quant is None else 5])
    return np.nan_to_num(m.astype(float), nan=0.0).astype(bool)


# ---------------------------------------------------------------------------
# 成本（事件层估算，与 费用与交易约束 逐项对齐）
# ---------------------------------------------------------------------------
def _stamp_rate(d: date, side: str) -> float:
    rule = FEE._pick_rule(FEE.DEFAULT_STAMP_SCHEDULE, d)
    if rule.sides == "sell" and side != "sell":
        return 0.0
    return float(rule.rate)


def _transfer(d: date, code: str, shares: float, amount: float) -> float:
    rule = FEE._pick_rule(FEE.DEFAULT_TRANSFER_SCHEDULE, d)
    ex = FEE.exchange_of(code)
    if rule.scope == "SH" and ex != "SH":
        return 0.0
    if rule.mode == "par_sh":
        return float(FEE.round_money(shares * rule.rate))
    return float(FEE.round_money(amount * rule.rate))


def _one_cost(d: date, code: str, pi: float, po: float, adv: float,
              stress: int, notional: float) -> float:
    part = notional / adv if (np.isfinite(adv) and adv > 0) else 0.0
    base_slip = 10.0 * stress
    cap = 100.0 if stress == 1 else 200.0
    slip = min(cap, base_slip + 0.02 * math.sqrt(max(part, 0.0)) * 10000.0)
    buy_px = float(FEE.apply_slippage(pi, "buy", slip))
    sell_px = float(FEE.apply_slippage(po, "sell", slip))
    qty = int(math.floor(notional / max(buy_px, 1e-9) / N_LOT) * N_LOT)
    if qty < N_LOT:
        qty = N_LOT          # 一手都买不起时按最小可行手数计费（账户层会直接跳过）
    amt_b, amt_s = qty * buy_px, qty * sell_px
    fee_b = (max(amt_b * 0.00025 * stress, 5.0 * stress)
             + amt_b * _stamp_rate(d, "buy") * stress
             + _transfer(d, code, qty, amt_b) * stress)
    fee_s = (max(amt_s * 0.00025 * stress, 5.0 * stress)
             + amt_s * _stamp_rate(d, "sell") * stress
             + _transfer(d, code, qty, amt_s) * stress)
    # 未扣费、未滑点的 raw 收益 vs 实际到手收益，差额即往返成本（含 tick 取整）
    gross = po / pi - 1.0
    net = (amt_s - fee_s) / (amt_b + fee_b) - 1.0
    return float(gross - net)


def round_trip_cost(dates_in, codes, px_in, px_out, adv, stress: int = 1,
                    notional: float = EVENT_NOTIONAL) -> np.ndarray:
    out = np.full(len(dates_in), np.nan)
    for i in range(len(dates_in)):
        pi, po = float(px_in[i]), float(px_out[i])
        if not (np.isfinite(pi) and np.isfinite(po) and pi > 0 and po > 0):
            continue
        out[i] = _one_cost(to_date(dates_in[i]), str(codes[i]), pi, po,
                           float(adv[i]) if np.isfinite(adv[i]) else 0.0,
                           stress, notional)
    return out


# ---------------------------------------------------------------------------
# 事件层
# ---------------------------------------------------------------------------
@dataclass
class EventData:
    panel: Panel
    feat: Features
    entry_ok: np.ndarray
    sellable: np.ndarray
    nxt_sell: np.ndarray
    last_valid: np.ndarray
    _bench: dict
    _ctrl: dict


def prep_events(p: Panel, f: Features) -> EventData:
    listed = p.flags["listed"]
    lu = p.raw["limit_up_px"]
    ld = p.raw["limit_down_px"]
    o = p.raw["open"]
    with np.errstate(invalid="ignore"):
        lu_open = np.isfinite(lu) & (o >= lu - 0.001)
        ld_open = np.isfinite(ld) & (o <= ld + 0.001)
    sellable = listed & ~p.flags["suspended"] & ~ld_open & np.isfinite(o)
    entry_ok = (listed & ~p.flags["suspended"] & ~p.flags["resume"]
                & ~lu_open & np.isfinite(o) & (o > 0))
    nxt = np.full(listed.shape, -1, dtype=np.int32)
    run = np.full(listed.shape[1], -1, dtype=np.int32)
    lv = np.full(listed.shape, -1, dtype=np.int32)
    runv = np.full(listed.shape[1], -1, dtype=np.int32)
    finite_close = np.isfinite(p.a["close"]) & (p.a["close"] > 0)
    for i in range(listed.shape[0] - 1, -1, -1):
        run = np.where(sellable[i], i, run)
        nxt[i] = run
    for i in range(listed.shape[0]):
        runv = np.where(finite_close[i], i, runv)
        lv[i] = runv
    return EventData(panel=p, feat=f, entry_ok=entry_ok, sellable=sellable,
                     nxt_sell=nxt, last_valid=lv, _bench={}, _ctrl={})


def _forward(ed: EventData, jj: np.ndarray, e: np.ndarray, H: int):
    D = ed.panel.n_days
    tgt = e + H
    tgt_c = np.minimum(tgt, D - 1)
    x = ed.nxt_sell[tgt_c, jj]
    forced = (tgt >= D) | (x < 0) | (x > tgt + MAX_PERSIST_DAYS) | (x >= D)
    xf = np.where(forced, np.minimum(tgt_c + MAX_PERSIST_DAYS, D - 1), x)
    ao, ac = ed.panel.a["open"], ed.panel.a["close"]
    px_open = ao[xf, jj]
    alt = ed.last_valid[xf, jj]
    px_alt = np.where(alt >= 0, ac[np.clip(alt, 0, None), jj], np.nan)
    # 出场日无 bar（退市/长期停牌）时，退化为最后一个有效收盘价
    px_open = np.where(np.isfinite(px_open), px_open, px_alt)
    with np.errstate(invalid="ignore", divide="ignore"):
        gross = px_open / ao[e, jj] - 1.0
        gross_forced = np.where(np.isfinite(px_alt), px_alt, px_open) / ao[e, jj] - 1.0
    gross = np.where(forced, gross_forced, gross)
    return xf, forced, gross


def _bench(ed: EventData, H: int) -> np.ndarray:
    if H in ed._bench:
        return ed._bench[H]
    p = ed.panel
    ao = p.a["open"]
    out = np.full(p.n_days, np.nan)
    for i in range(p.n_days - 1):
        e, x = i + 1, i + 1 + H
        if x >= p.n_days:
            continue
        m = p.flags["elig"][i] & np.isfinite(ao[e]) & np.isfinite(ao[x])
        if m.sum() >= 20:
            out[i] = float(np.mean(ao[x, m] / ao[e, m] - 1.0))
    ed._bench[H] = out
    return out


def _ctrl(ed: EventData, H: int, k: int) -> dict:
    key = (H, k)
    if key in ed._ctrl:
        return ed._ctrl[key]
    p, f = ed.panel, ed.feat
    mild = _mild(f, k, BAND_MAIN)
    ao = p.a["open"]
    out = {s: np.full(p.n_days, np.nan) for s in STATES}
    for i in range(p.n_days - 1):
        e, x = i + 1, i + 1 + H
        if x >= p.n_days:
            continue
        good = np.isfinite(ao[e]) & np.isfinite(ao[x]) & ~mild[i]
        for s in STATES:
            m = good & f.states[s][i]
            if m.sum() >= 10:
                out[s][i] = float(np.mean(ao[x, m] / ao[e, m] - 1.0))
    ed._ctrl[key] = out
    return out


def code_segment(code: str) -> str:
    d = code.split(".")[0]
    if d.startswith("60"):
        return "SH60"
    if d.startswith(("000", "001")):
        return "SZ00"
    if d.startswith(("002", "003")):
        return "SZ002"
    return "OTHER"


def price_bucket(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), "NA", dtype=object)
    out[x < 5] = "<5"
    out[(x >= 5) & (x < 10)] = "5-10"
    out[(x >= 10) & (x < 20)] = "10-20"
    out[(x >= 20) & (x < 50)] = "20-50"
    out[x >= 50] = ">=50"
    out[~np.isfinite(x)] = "NA"
    return out


def extract_events(ed: EventData, pol: Policy, H: int) -> pd.DataFrame:
    p, f = ed.panel, ed.feat
    m = policy_mask(f, pol) & p.flags["elig"]
    ii, jj = np.nonzero(m)
    if len(ii) == 0:
        return pd.DataFrame()
    e = ii + 1
    ok = e < p.n_days - 1
    ok &= ed.entry_ok[np.clip(e, 0, p.n_days - 1), jj]
    ok &= np.isfinite(p.a["close"][ii, jj])
    ii, jj, e = ii[ok], jj[ok], e[ok]
    if len(ii) == 0:
        return pd.DataFrame()
    xf, forced, gross = _forward(ed, jj, e, H)
    o = p.raw["open"]
    with np.errstate(invalid="ignore", divide="ignore"):
        gross_raw = o[xf, jj] / o[e, jj] - 1.0
    cost = round_trip_cost([p.dates[i] for i in e], [p.codes[j] for j in jj],
                           o[e, jj], o[xf, jj], p.adv20[e, jj])
    k_used = pol.k if pol.quant is None else 5
    bench_arr = _bench(ed, H)
    ctrl_map = _ctrl(ed, H, k_used)
    ctrl = np.full(len(ii), np.nan)
    for s in STATES:
        sel = f.states[s][ii, jj]
        if sel.any():
            ctrl[sel] = ctrl_map[s][ii[sel]]
    state_arr = np.full(len(ii), "OTHER", dtype=object)
    for s in STATES:
        state_arr[f.states[s][ii, jj]] = s
    df = pd.DataFrame({
        "policy": pol.pid,
        "state": state_arr,
        "code": [p.codes[j] for j in jj],
        "signal_idx": ii.astype(np.int32),
        "entry_idx": e.astype(np.int32),
        "exit_idx": xf.astype(np.int32),
        "signal_day": p.day_np[ii],
        "entry_day": p.day_np[e],
        "hold_idx": (xf - e).astype(np.int16),
        "forced": forced,
        "H": np.int16(H),
        "k": np.int16(pol.k),
        "pul_k": f.pul[k_used][ii, jj],
        "pul5_q": f.pul5_q[ii, jj],
        "dd250": f.dd250[ii, jj],
        "gross": gross,
        "gross_raw": gross_raw,
        "cost": cost,
        "net": gross - cost,
        "bench": bench_arr[ii],
        "ctrl": ctrl,
        "excess_mkt": gross - bench_arr[ii],
        "excess_state": gross - ctrl,
        "entry_px": o[e, jj],
        "exit_px": o[xf, jj],
        "close_sig": p.raw["close"][ii, jj],
        "adv20": p.adv20[e, jj],
        "size": p.size[ii, jj],
        "size_q": f.size_q[ii, jj],
        "liq_q": f.liq_q[ii, jj],
        "seg": [code_segment(p.codes[j]) for j in jj],
        "price_bucket": price_bucket(p.raw["close"][ii, jj]),
        "mkt_gate": p.gate_hs300[ii],
    })
    return df


# ---------------------------------------------------------------------------
# 账户层
# ---------------------------------------------------------------------------
@dataclass
class ACfg:
    cid: str
    policy: str
    H: int = H_MAIN
    slots: int = 10
    capital: float = 1_000_000.0
    max_new: int = 2
    rank: str = "pul"
    stress: int = 1
    delay: int = 0
    gate: str = "none"
    note: str = ""


def _rank_js(js: np.ndarray, score: np.ndarray, adv: np.ndarray) -> np.ndarray:
    if not len(js):
        return js
    s = np.where(np.isfinite(score[js]), score[js], -np.inf)
    a = np.where(np.isfinite(adv[js]), adv[js], -np.inf)
    return js[np.lexsort((js, -a, -s))]


def _fee_config(stress: int):
    base = UB.make_fee_config(slippage_bps=10.0 * stress, transfer_par_rate=0.0010)
    if stress == 1:
        return base
    return FEE.FeeConfig(
        commission_rate=0.00025 * stress,
        min_commission=5.0 * stress,
        stamp_schedule=tuple(FEE.StampRule(r.effective, r.rate * stress, r.sides)
                             for r in base.stamp_schedule),
        transfer_schedule=tuple(FEE.TransferRule(r.effective, r.mode, r.rate * stress, r.scope)
                                for r in base.transfer_schedule),
        slippage_bps=10.0 * stress, max_participation=0.02, dividend_tax_rate=0.0)


def run_account(ed: EventData, cfg: ACfg, window: str) -> dict:
    start, end = WINDOWS[window]
    _assert_window(start, end, f"run_account:{cfg.cid}:{window}")
    p, f = ed.panel, ed.feat
    i0, i1 = p.window_index(start, end)
    sig = policy_mask(f, POLICY_BY_ID[cfg.policy]) & p.flags["elig"]
    for _ in range(cfg.delay):
        sig = np.vstack([np.zeros((1, sig.shape[1]), dtype=bool), sig[:-1]])

    info = {c: ACC.SymbolInfo(code=c, name="", ipo_date=p.ipo_date[j],
                              out_date=p.delist_date[j])
            for j, c in enumerate(p.codes)}
    acfg = ACC.AccountConfig(
        initial_cash=float(cfg.capital), fee=_fee_config(cfg.stress),
        cash_settle_days=0, enforce_t1=True, allow_buy_downsize=True,
        normalize_buy_orders=True, rights_policy="subscribe", dividend_tax_rate=0.0,
        delisting_missing_days=10_000_000, delisting_haircut=0.0,
        liquidate_on_finalize=False, finalize_haircut=0.0, strict=False)
    acct = UB.UnifiedAccount(acfg, p.dates, freeze_date=p.dates[-1],
                             symbol_info=info, accounting="shares")
    acct._code_pos = p.code_idx

    checks = {"cash_resid_max": 0.0, "nav_resid_max": 0.0, "max_date": "",
              "n_rows_after_2020": 0, "forced": 0}
    prev_cash = float(acct.cash_total)
    trade_ptr = ca_ptr = tax_ptr = 0

    for i in range(i0 - 1, i1 + 1):
        d = p.dates[i]
        held = list(acct.positions.keys())
        active = [o.code for o in acct._orders if o.active]
        ev_today = p.events.on(d)
        needed = set(held) | set(active) | {e.symbol for e in ev_today}
        acct._ca_row = p.ca_factor[i]
        acct._adv = {c: float(p.adv20[i - 1, p.code_idx[c]]) if i > 0 and c in p.code_idx else 0.0
                     for c in set(active) | set(held)}
        acct._resume_today = {p.codes[j] for j in np.flatnonzero(p.flags["resume"][i])}
        rec = acct.on_day(d, bars=p.bars_on(i, needed), events=ev_today)

        forced_now = False
        for o in list(acct._orders):
            if not (o.active and o.side == "sell" and o.filled_qty == 0):
                continue
            try:
                age = i - p.day_idx[o.execute_from]
            except KeyError:
                continue
            if age >= MAX_PERSIST_DAYS:
                pos = acct.positions.get(o.code)
                j = p.code_idx.get(o.code)
                if pos is None or j is None:
                    continue
                px = float(p.raw["close"][i, j])
                if not np.isfinite(px) or px <= 0:
                    px = float(pos.last_price) if pos.last_price else 0.0
                if px > 0:
                    acct._force_sell(o.code, d, px, "sell_persistence_10d")
                    acct._forced_ages.append((o.order_id, int(age)))
                    forced_now = True
        if forced_now:
            rec = acct._mark_and_record(d)

        new_fills = acct.trades[trade_ptr:]
        buys = sum(t.amount + t.fees for t in new_fills if t.side == "buy")
        sells = sum(t.amount - t.fees for t in new_fills if t.side == "sell")
        ca_cash = sum(x["credit"] for x in acct.ca_rows[ca_ptr:])
        tax_cash = sum(x["amount"] for x in acct.tax_ledger[tax_ptr:])
        checks["cash_resid_max"] = max(
            checks["cash_resid_max"],
            abs(float(rec["cash_total"]) - prev_cash - (sells - buys + ca_cash + tax_cash)))
        prev_cash = float(rec["cash_total"])
        mv = 0.0
        for code, pos in acct.positions.items():
            j = p.code_idx.get(code)
            if j is None:
                continue
            px = float(p.raw["close"][i, j])
            if not np.isfinite(px) or px <= 0:
                px = float(pos.last_price) if pos.last_price else 0.0
            mv += pos.shares * px
        checks["nav_resid_max"] = max(checks["nav_resid_max"],
                                      abs(mv - float(rec["market_value"])))
        checks["max_date"] = max(checks["max_date"], d.isoformat())
        trade_ptr = len(acct.trades)
        ca_ptr = len(acct.ca_rows)
        tax_ptr = len(acct.tax_ledger)

        if i >= i1:
            break

        nav = float(rec["nav"])
        cash_avail = float(acct.cash_available)
        free = cfg.slots - sum(1 for x in acct.positions.values() if x.shares > 0) \
            - sum(1 for o in acct._orders if o.active and o.side == "buy")
        if free > 0:
            cand = sig[i].copy()
            for code in list(acct.positions) + active:
                j = p.code_idx.get(code)
                if j is not None:
                    cand[j] = False
            for o in acct._orders:
                if o.active and o.side == "sell":
                    j = p.code_idx.get(o.code)
                    if j is not None:
                        cand[j] = False
            if cfg.gate == "hs300" and not p.gate_hs300[i]:
                cand[:] = False
            js = np.flatnonzero(cand)
            if cfg.rank == "pul":
                js = _rank_js(js, f.pul[5][i], p.adv20[i])
            elif cfg.rank == "trough_deep":
                js = _rank_js(js, -f.pul5_min3[i], p.adv20[i])
            elif cfg.rank == "trough_shallow":
                js = _rank_js(js, f.pul5_min3[i], p.adv20[i])
            elif cfg.rank == "hash":
                rng = np.random.default_rng(1_000_003 + i)
                js = js[rng.permutation(len(js))]
            else:
                js = _rank_js(js, np.zeros(p.n_codes), p.adv20[i])
            n_new = 0
            for j in js:
                if n_new >= min(free, cfg.max_new):
                    break
                price = float(p.raw["close"][i, j])
                if not (np.isfinite(price) and price > 0):
                    continue
                budget = nav / max(cfg.slots, 1)
                shares = int(budget / price / N_LOT) * N_LOT
                cap_sh = int(max(cash_avail, 0.0) * 0.98 / (price * 1.01) / N_LOT) * N_LOT
                shares = int(min(shares, cap_sh))
                if shares < N_LOT:
                    continue
                acct.submit(ACC.Order(code=p.codes[j], side="buy", qty=shares, kind="open",
                                      tif="day", tag="sir_entry"), d)
                cash_avail -= shares * price * 1.01
                n_new += 1

        for code, meta in list(acct.hold_meta.items()):
            pos = acct.positions.get(code)
            if not pos or pos.shares <= 0:
                continue
            if i - int(meta["entry_idx"]) < cfg.H - 1:
                continue
            cur = acct.orders.get(meta.get("exit_order") or "")
            if cur is not None and cur.active:
                continue
            o = acct.submit(ACC.Order(code=code, side="sell", qty=pos.shares,
                                      kind="open", tif="gtc", tag="sir_exit"), d)
            meta["exit_order"] = o.order_id

    acct.finalize(p.dates[i1])
    eq = pd.DataFrame(acct.daily).rename(columns={"cash_total": "cash"})
    eq = eq[(eq["date"] >= start) & (eq["date"] <= end)].copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date").sort_index()
    fills = pd.DataFrame([t.as_dict() for t in acct.trades])
    if len(fills):
        fills["date"] = pd.to_datetime(fills["date"])
        eq["turnover"] = fills.groupby("date")["amount"].sum().reindex(eq.index).fillna(0.0)
        eq["cost"] = fills.groupby("date")["fees"].sum().reindex(eq.index).fillna(0.0)
    else:
        eq["turnover"] = 0.0
        eq["cost"] = 0.0
    trades = pd.DataFrame(acct.round_trips)
    if len(trades):
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])
        trades = trades.sort_values("exit_date").reset_index(drop=True)
    checks["n_rows_after_2020"] = int((eq.index >= pd.Timestamp(HOLDOUT_START)).sum())
    checks["forced"] = int(sum(acct._force_counts.values()))
    checks["max_date"] = str(eq.index.max().date()) if len(eq) else ""
    return {"cid": cfg.cid, "window": window, "cfg": asdict(cfg), "eq": eq,
            "trades": trades, "fills": fills, "checks": checks}


# ---------------------------------------------------------------------------
# 本族基准
# ---------------------------------------------------------------------------
def bench_ew_monthly(p: Panel, i0: int, i1: int, cost_roundtrip: float = 0.0030) -> pd.Series:
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in p.dates[i0:i1 + 1]])
    rets = np.zeros(len(dates))
    prev_w: dict = {}
    active: list = []
    ac = p.a["close"]
    for k in range(1, len(dates)):
        gi = i0 + k
        rebal = (k == len(dates) - 1) or (dates[k].month != dates[k - 1].month)
        if rebal:
            active = np.flatnonzero(p.flags["elig"][gi - 1]).tolist()
            w_new = {p.codes[j]: 1.0 / len(active) for j in active} if active else {}
            names = set(prev_w) | set(w_new)
            turn = 0.5 * sum(abs(w_new.get(n, 0.0) - prev_w.get(n, 0.0)) for n in names)
            rets[k] -= turn * cost_roundtrip
            prev_w = w_new
        if active:
            with np.errstate(invalid="ignore", divide="ignore"):
                r = ac[gi, active] / ac[gi - 1, active] - 1.0
            r = r[np.isfinite(r)]
            if len(r):
                rets[k] += float(np.mean(r))
    return pd.Series(rets, index=dates, name="B1_EW")


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def cmd_selfcheck(p: Panel, f: Features) -> dict:
    t0 = time.time()
    ck = {}
    con = duckdb.connect(str(DB), read_only=True)
    try:
        df = con.execute(
            "SELECT symbol,date,close,factor,pct_chg FROM daily WHERE board=? "
            "AND date BETWEEN ? AND ? AND symbol IN ('600519.SH','000001.SZ','600000.SH','000002.SZ') "
            "ORDER BY symbol,date", [BOARD, PANEL_START, ANALYSIS_END]).df()
    finally:
        con.close()
    dev_lvl, dev_ret = 0.0, 0.0
    for code, g in df.groupby("symbol"):
        j = p.code_idx.get(code)
        if j is None:
            continue
        g = g.reset_index(drop=True)
        ii = [p.day_idx[to_date(x)] for x in g["date"]]
        a = p.a["close"][ii, j]
        fac = g["factor"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(fac)
        if ok.sum() > 1:
            dev_lvl = max(dev_lvl, float(np.max(np.abs(a[ok] / fac[ok] - a[ok][0] / fac[ok][0]))))
        r = np.diff(a) / a[:-1]
        pc = g["pct_chg"].to_numpy(float)[1:] / 100.0
        ok = np.isfinite(r) & np.isfinite(pc)
        if ok.sum():
            dev_ret = max(dev_ret, float(np.max(np.abs(r[ok] - pc[ok]))))
    ck["a_close_vs_factor_level_dev"] = dev_lvl
    ck["adj_ret_vs_pctchg_max"] = dev_ret

    rng = np.random.default_rng(20261028)
    dts, cds, pin, pout, adv = [], [], [], [], []
    for _ in range(300):
        i = int(rng.integers(300, p.n_days - 30))
        j = int(rng.integers(0, p.n_codes))
        if not (np.isfinite(p.raw["open"][i, j]) and np.isfinite(p.raw["open"][i + 10, j])):
            continue
        dts.append(p.dates[i])
        cds.append(p.codes[j])
        pin.append(float(p.raw["open"][i, j]))
        pout.append(float(p.raw["open"][i + 10, j]))
        adv.append(float(p.adv20[i, j]) if np.isfinite(p.adv20[i, j]) else 1e8)
    cvec = round_trip_cost(dts, cds, np.array(pin), np.array(pout), np.array(adv))
    cdir = []
    for d, c, a1, a2, ad in zip(dts, cds, pin, pout, adv):
        cdir.append(_one_cost(d, c, a1, a2, ad, 1, EVENT_NOTIONAL))
    ck["cost_n"] = int(len(cdir))
    ck["cost_mean"] = float(np.nanmean(cvec))
    ck["cost_max_abs_diff_vs_module"] = float(np.max(np.abs(cvec - np.array(cdir))))

    # 费用模块逐笔对照（最少样本，确认印花税/过户费历史分段被用上）
    checks_fee = []
    for d in (date(2008, 3, 3), date(2008, 6, 3), date(2010, 5, 5), date(2016, 5, 5)):
        i = p.day_idx.get(d)
        if i is None:
            continue
        j = int(np.flatnonzero(p.flags["elig"][i])[0])
        px = float(p.raw["close"][i, j])
        q = int(EVENT_NOTIONAL / px // N_LOT) * N_LOT
        fb = FEE.trade_fees(d, p.codes[j], "buy", px, q)
        fs = FEE.trade_fees(d, p.codes[j], "sell", px, q)
        mine_b = (max(q * px * 0.00025, 5.0) + q * px * _stamp_rate(d, "buy")
                  + _transfer(d, p.codes[j], q, q * px))
        mine_s = (max(q * px * 0.00025, 5.0) + q * px * _stamp_rate(d, "sell")
                  + _transfer(d, p.codes[j], q, q * px))
        checks_fee.append({"date": d.isoformat(), "code": p.codes[j],
                           "buy_diff": round(mine_b - fb.total, 4),
                           "sell_diff": round(mine_s - fs.total, 4),
                           "stamp_buy": round(_stamp_rate(d, "buy"), 6),
                           "stamp_sell": round(_stamp_rate(d, "sell"), 6)})
    ck["fee_rule_checks"] = checks_fee

    # 因果截断：主候选信号任抽 120 个，用 ≤i 的子样本重算
    sig = policy_mask(f, POLICY_BY_ID["P01_STRUCT_K5_E1"])
    ii, jj = np.nonzero(sig)
    pick = rng.choice(len(ii), size=min(120, len(ii)), replace=False)
    bad = 0
    for k in pick:
        i, j = int(ii[k]), int(jj[k])
        w = p.a["close"][:i + 1, j]
        if not np.isfinite(w[-1]) or not np.isfinite(w[-6]):
            bad += 1
            continue
        pul5 = w[-1] / w[-6] - 1.0
        win250 = w[-250:] if len(w) >= 250 else None
        ok = abs(pul5 - f.pul[5][i, j]) < 1e-12
        if win250 is not None and np.isfinite(win250).all():
            ok &= abs(float(np.mean(win250)) - f.ma250[i, j]) < 1e-9
            ok &= abs(w[-1] / float(np.max(win250)) - 1.0 - f.dd250[i, j]) < 1e-9
        bad += 0 if ok else 1
    ck["causal_bad"] = int(bad)
    ck["causal_n"] = int(len(pick))

    # 涨跌停对照
    lu, o, pc, st, dtsl = (p.raw["limit_up_px"], p.raw["open"], p.raw["preclose"],
                           p.flags["is_st"], p.raw["td_since_list"])
    mism, n = 0, 0
    for _ in range(400):
        i = int(rng.integers(300, p.n_days - 1))
        j = int(rng.integers(0, p.n_codes))
        if not (np.isfinite(o[i, j]) and np.isfinite(pc[i, j]) and np.isfinite(lu[i, j])):
            continue
        n += 1
        try:
            lim = FEE.limit_prices(pc[i, j], p.codes[j], bool(st[i, j]), p.dates[i],
                                   int(dtsl[i, j]) if np.isfinite(dtsl[i, j]) else None)
        except Exception:
            continue
        if (not lim.no_limit) and abs(lim.up - lu[i, j]) > 0.011:
            mism += 1
    ck["limit_up_mismatch"] = int(mism)
    ck["limit_up_n"] = int(n)

    i0, i1 = p.window_index(*FULL)
    ck["n_dates"] = int(p.n_days)
    ck["n_codes"] = int(p.n_codes)
    ck["n_bars"] = int(np.isfinite(p.raw["close"]).sum())
    ck["avg_elig"] = float(p.flags["elig"][i0:i1 + 1].sum(axis=1).mean())
    ck["n_rows_after_2020"] = int(sum(1 for d in p.dates if d >= to_date(HOLDOUT_START)))
    ck["delisted_in_window"] = int(sum(1 for d in p.delist_date
                                       if d is not None and DEV[0] <= d.isoformat() <= VAL[1]))
    ck["prereg_sha256"] = prereg_sha()
    ck["elapsed_s"] = round(time.time() - t0, 2)
    json_dump(ck, OUT / "01_自检.json")
    log(f"自检: {json.dumps({k: v for k, v in ck.items() if k != 'fee_rule_checks'}, ensure_ascii=False)}")
    return ck


# ---------------------------------------------------------------------------
# 事件层统计
# ---------------------------------------------------------------------------
def _cluster_stats(values: np.ndarray, days: np.ndarray, n_boot: int = 2000,
                   seed: int = 20261028) -> dict:
    """按交易日聚类的均值检验 + 按日重采样 95% CI（向量化，避免逐日 concat）。"""
    y = np.asarray(values, float)
    ok = np.isfinite(y)
    y, days = y[ok], np.asarray(days)[ok]
    if len(y) < 10:
        return {"n": int(len(y)), "error": "样本不足"}
    base = STAT.cluster_mean_test(y, days)
    uniq, inv = np.unique(days, return_inverse=True)
    G = len(uniq)
    if G < 3:
        return base
    sums = np.bincount(inv, weights=y, minlength=G)
    cnts = np.bincount(inv, minlength=G).astype(np.float64)
    rng = np.random.default_rng(seed)
    B = int(n_boot)
    means = np.empty(B)
    step = max(1, 20000 // max(G, 1))
    for s in range(0, B, step):
        e = min(B, s + step)
        idx = rng.integers(0, G, size=(e - s, G))
        means[s:e] = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    base["n_dates"] = int(G)
    base.update({
        "n_boot": B,
        "boot_lo": float(np.quantile(means, 0.025)),
        "boot_hi": float(np.quantile(means, 0.975)),
        "p_boot_le_0": float((means <= 0).mean()),
    })
    return base


def _event_stats(sub: pd.DataFrame, pol: Policy, H: int, win: str) -> dict:
    out = {"policy": pol.pid, "note": pol.note, "state": pol.state, "k": pol.k,
           "entry": pol.entry, "H": H, "window": win, "n": int(len(sub))}
    if len(sub) < 5:
        out["underpowered"] = True
        return out
    gross = sub["gross"].to_numpy(float)
    net = sub["net"].to_numpy(float)
    exc = sub["excess_state"].to_numpy(float)
    days = sub["entry_idx"].to_numpy(np.int64)
    t_gross = _cluster_stats(gross, days)
    t_exc = _cluster_stats(exc, days)
    t_net = _cluster_stats(net, days)
    out.update({
        "n_dates": int(t_gross.get("n_dates", t_gross.get("clusters", 0))),
        "n_codes": int(sub["code"].nunique()),
        "underpowered": bool(len(sub) < 30 or sub["code"].nunique() < 10),
        "gross_mean": float(np.nanmean(gross)),
        "gross_median": float(np.nanmedian(gross)),
        "cost_mean": float(np.nanmean(sub["cost"].to_numpy(float))),
        "net_mean": float(np.nanmean(net)),
        "net_median": float(np.nanmedian(net)),
        "n_net_valid": int(np.isfinite(net).sum()),
        "gross_t": t_gross.get("t"),
        "gross_ci_lo": t_gross.get("ci_lo"),
        "excess_mean": float(np.nanmean(exc)),
        "excess_median": float(np.nanmedian(exc)),
        "excess_t": t_exc.get("t"),
        "excess_ci_lo": t_exc.get("ci_lo"),
        "excess_ci_hi": t_exc.get("ci_hi"),
        "net_t": t_net.get("t"),
        "net_ci_lo": t_net.get("ci_lo"),
        "net_ci_hi": t_net.get("ci_hi"),
        "net_p_boot": t_net.get("p_boot_le_0"),
        "win_rate": float(np.nanmean(gross > 0)),
        "cover_share": float(np.nanmean(net > 0)),
        "forced_share": float(np.mean(sub["forced"].to_numpy(bool))),
    })
    return out


def _decile_table(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if ev.empty:
        return pd.DataFrame()
    d = ev.dropna(subset=["pul5_q"]).copy()
    d["decile"] = np.minimum((d["pul5_q"] * 10).astype(int), 9)
    for win, (ws, we) in {"dev": DEV, "val": VAL, "full": FULL}.items():
        s = d[(d["signal_day"] >= np.datetime64(ws)) & (d["signal_day"] <= np.datetime64(we))]
        for dec, g in s.groupby("decile"):
            rows.append({"window": win, "decile": int(dec), "n": int(len(g)),
                         "pul5_mean": float(np.mean(g["pul_k"])),
                         "gross": float(np.mean(g["gross"])),
                         "cost": float(np.nanmean(g["cost"])),
                         "net": float(np.mean(g["net"])),
                         "win_rate": float(np.mean(g["gross"] > 0)),
                         "net_positive_share": float(np.mean(g["net"] > 0))})
    return pd.DataFrame(rows)


def _layer_table(det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if det.empty:
        return pd.DataFrame()
    keep = ["P01_STRUCT_K5_E1", "P02_STRUCT_K5_E0", "P06_NOTREND_K5_E1",
            "P07_BROKEN_K5_E1", "P17_STRUCT_KNIFE"]
    d = det[det["policy"].isin(keep)]
    for pol, g0 in d.groupby("policy"):
        for win, (ws, we) in {"dev": DEV, "val": VAL}.items():
            g = g0[(g0["signal_day"] >= np.datetime64(ws)) & (g0["signal_day"] <= np.datetime64(we))]
            if len(g) < 20:
                continue
            for key in ("seg", "size_q", "liq_q", "price_bucket", "mkt_gate"):
                for val, gg in g.groupby(key):
                    if len(gg) < 20:
                        continue
                    rows.append({"policy": pol, "window": win, "layer": key,
                                 "bucket": str(val), "n": int(len(gg)),
                                 "gross": float(np.mean(gg["gross"])),
                                 "cost": float(np.nanmean(gg["cost"])),
                                 "net": float(np.mean(gg["net"])),
                                 "win_rate": float(np.mean(gg["gross"] > 0))})
    return pd.DataFrame(rows)


DETAIL_POLICIES = {"P01_STRUCT_K5_E1", "P02_STRUCT_K5_E0", "P06_NOTREND_K5_E1",
                   "P07_BROKEN_K5_E1", "P17_STRUCT_KNIFE", "P18_TREND_WINNER",
                   "P20_STRUCT_Q0525_E1"}


def _rank_layer(det: pd.DataFrame) -> pd.DataFrame:
    """按“同日信号内 pul5 从浅到深”的名次分层：账户只买前 2 名。"""
    rows = []
    main = det[det["policy"] == "P01_STRUCT_K5_E1"]
    if main.empty:
        return pd.DataFrame()
    d = main.copy()
    d["rk"] = d.groupby("signal_idx")["pul_k"].rank(ascending=False, method="first")
    d["rank_bucket"] = np.where(d["rk"] <= 2, "1-2",
                                np.where(d["rk"] <= 5, "3-5",
                                         np.where(d["rk"] <= 10, "6-10", ">10")))
    for win, (ws, we) in {"dev": DEV, "val": VAL, "full": FULL}.items():
        s = d[(d["signal_day"] >= np.datetime64(ws)) & (d["signal_day"] <= np.datetime64(we))]
        for b, g in s.groupby("rank_bucket"):
            rows.append({"window": win, "rank_bucket": b, "n": int(len(g)),
                         "pul_k": float(np.mean(g["pul_k"])),
                         "gross": float(np.mean(g["gross"])),
                         "cost": float(np.nanmean(g["cost"])),
                         "net": float(np.mean(g["net"])),
                         "win_rate": float(np.mean(g["gross"] > 0))})
    return pd.DataFrame(rows)


def cmd_events(p: Panel, f: Features, ed: EventData) -> dict:
    rows, detail = [], []
    for pol in POLICIES:
        H = POLICY_HOLD.get(pol.pid, H_MAIN)
        ev = extract_events(ed, pol, H)
        if ev.empty:
            continue
        ev = ev[(ev["signal_day"] >= np.datetime64(DEV[0]))
                & (ev["signal_day"] <= np.datetime64(VAL[1]))]
        if ev.empty:
            continue
        for win, (ws, we) in WINDOWS.items():
            sub = ev[(ev["signal_day"] >= np.datetime64(ws))
                     & (ev["signal_day"] <= np.datetime64(we))]
            rows.append(_event_stats(sub, pol, H, win))
        if pol.pid in DETAIL_POLICIES:
            detail.append(ev)
        log(f"  {pol.pid}: n={len(ev)}")
        del ev
    det = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    del detail
    if len(det):
        out = det.copy()
        for c, src in (("signal_date", "signal_idx"), ("entry_date", "entry_idx"),
                       ("exit_date", "exit_idx")):
            out[c] = [p.dates[i] for i in out[src].to_numpy()]
        out.drop(columns=["signal_day", "entry_day"], inplace=True)
        out.to_parquet(OUT / "02_事件_明细.parquet", index=False)
        del out
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "03_事件_汇总.csv", index=False, encoding="utf-8-sig")

    main_ev = det[det["policy"] == "P01_STRUCT_K5_E1"] if len(det) else pd.DataFrame()
    dec = _decile_table(main_ev)
    dec.to_csv(OUT / "04_回撤分位_十分位.csv", index=False, encoding="utf-8-sig")
    layers = _layer_table(det)
    layers.to_csv(OUT / "05_分层_板块规模流动性价格.csv", index=False, encoding="utf-8-sig")
    rl = _rank_layer(det)
    rl.to_csv(OUT / "05b_同日名次_分层.csv", index=False, encoding="utf-8-sig")
    costlat = _cost_delay_table(p, ed, main_ev)
    costlat.to_csv(OUT / "06_成本敏感_延迟.csv", index=False, encoding="utf-8-sig")
    log("事件层完成")
    return {"summary": summ, "detail": det}


def _cost_delay_table(p: Panel, ed: EventData, base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if base is None or base.empty:
        return pd.DataFrame()
    ii = base["signal_idx"].to_numpy(np.int64)
    jj = np.array([p.code_idx[c] for c in base["code"]])
    for win, (ws, we) in {"dev": DEV, "val": VAL, "full": FULL}.items():
        m = ((base["signal_day"] >= np.datetime64(ws))
             & (base["signal_day"] <= np.datetime64(we))).to_numpy()
        sub = base[m]
        if len(sub) < 20:
            continue
        for stress in (1, 2, 3):
            cost = round_trip_cost([p.dates[i] for i in ii[m]], list(sub["code"]),
                                   sub["entry_px"].to_numpy(float),
                                   sub["exit_px"].to_numpy(float),
                                   sub["adv20"].to_numpy(float), stress=stress)
            rows.append({"window": win, "scenario": f"cost_{stress}x", "delay": 0,
                         "n": int(len(sub)), "gross": float(np.mean(sub["gross"])),
                         "cost": float(np.nanmean(cost)),
                         "net": float(np.mean(sub["gross"].to_numpy() - cost))})
        for delay in (1, 2):
            e = ii + 1 + delay
            ok = (e + H_MAIN < p.n_days) & ed.entry_ok[np.clip(e, 0, p.n_days - 1), jj] & m
            if ok.sum() < 20:
                continue
            xf, forced, gross = _forward(ed, jj[ok], e[ok], H_MAIN)
            cost = round_trip_cost([p.dates[i] for i in e[ok]],
                                   list(base["code"].to_numpy()[ok]),
                                   p.raw["open"][e[ok], jj[ok]],
                                   p.raw["open"][xf, jj[ok]],
                                   p.adv20[e[ok], jj[ok]])
            rows.append({"window": win, "scenario": "delay", "delay": delay,
                         "n": int(ok.sum()), "gross": float(np.mean(gross)),
                         "cost": float(np.nanmean(cost)),
                         "net": float(np.mean(gross - cost))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 账户层
# ---------------------------------------------------------------------------
ACCOUNT_CFGS = [
    ACfg("A01_MAIN", "P01_STRUCT_K5_E1", 10, 10, 1e6, 2, "pul", 1, 0, "none", "主候选"),
    ACfg("A02_NOCONF", "P02_STRUCT_K5_E0", 10, 10, 1e6, 2, "pul", 1, 0, "none", "不确认"),
    ACfg("A03_ANYENTRY", "P03_STRUCT_K5_ANY", 10, 10, 1e6, 2, "pul", 1, 0, "none", "入场消融"),
    ACfg("A04_TREND", "P04_TREND_K5_E1", 10, 10, 1e6, 2, "pul", 1, 0, "none", "仅上行"),
    ACfg("A05_BASE", "P05_BASE_K5_E1", 10, 10, 1e6, 2, "pul", 1, 0, "none", "仅底部横盘"),
    ACfg("A06_NOTREND", "P06_NOTREND_K5_E1", 10, 10, 1e6, 2, "pul", 1, 0, "none", "反例"),
    ACfg("A07_BROKEN", "P07_BROKEN_K5_E1", 10, 10, 1e6, 2, "pul", 1, 0, "none", "反例"),
    ACfg("A08_KNIFE", "P17_STRUCT_KNIFE", 10, 10, 1e6, 2, "pul", 1, 0, "none", "极端接飞刀"),
    ACfg("A09_WINNER", "P18_TREND_WINNER", 10, 10, 1e6, 2, "pul", 1, 0, "none", "极端赢家"),
    ACfg("A10_NOSTRUCT", "P19_ANYSTATE_K5_E1", 10, 10, 1e6, 2, "pul", 1, 0, "none", "去结构"),
    ACfg("A11_K3", "P08_STRUCT_K3_E1", 10, 10, 1e6, 2, "pul", 1, 0, "none", "k=3"),
    ACfg("A12_K10", "P09_STRUCT_K10_E1", 10, 10, 1e6, 2, "pul", 1, 0, "none", "k=10"),
    ACfg("A13_BAND8", "P10_STRUCT_K5_E1_B8", 10, 10, 1e6, 2, "pul", 1, 0, "none", "浅带"),
    ACfg("A14_BAND15", "P11_STRUCT_K5_E1_B15", 10, 10, 1e6, 2, "pul", 1, 0, "none", "深带"),
    ACfg("A15_H5", "P14_STRUCT_K5_E1_H5", 5, 10, 1e6, 2, "pul", 1, 0, "none", "H=5"),
    ACfg("A16_H20", "P15_STRUCT_K5_E1_H20", 20, 10, 1e6, 2, "pul", 1, 0, "none", "H=20"),
    ACfg("A17_H40", "P16_STRUCT_K5_E1_H40", 40, 10, 1e6, 2, "pul", 1, 0, "none", "H=40"),
    ACfg("A18_H3", "P13_STRUCT_K5_E1_H3", 3, 10, 1e6, 2, "pul", 1, 0, "none", "H=3"),
    ACfg("A19_SLOTS5", "P01_STRUCT_K5_E1", 10, 5, 1e6, 2, "pul", 1, 0, "none", "5 仓"),
    ACfg("A20_SLOTS20", "P01_STRUCT_K5_E1", 10, 20, 1e6, 2, "pul", 1, 0, "none", "20 仓"),
    ACfg("A21_CAP100K", "P01_STRUCT_K5_E1", 10, 10, 1e5, 2, "pul", 1, 0, "none", "10 万"),
    ACfg("A22_CAP20K", "P01_STRUCT_K5_E1", 10, 10, 2e4, 2, "pul", 1, 0, "none", "2 万"),
    ACfg("A23_STRESS2X", "P01_STRUCT_K5_E1", 10, 10, 1e6, 2, "pul", 2, 0, "none", "费用×2"),
    ACfg("A24_DELAY1", "P01_STRUCT_K5_E1", 10, 10, 1e6, 2, "pul", 1, 1, "none", "延迟 1 日"),
    ACfg("A25_RANKADV", "P01_STRUCT_K5_E1", 10, 10, 1e6, 2, "adv", 1, 0, "none", "排序换 ADV"),
    ACfg("A26_GATE", "P01_STRUCT_K5_E1", 10, 10, 1e6, 2, "pul", 1, 0, "hs300", "市场门"),
    ACfg("A27_Q0525", "P20_STRUCT_Q0525_E1", 10, 10, 1e6, 2, "pul", 1, 0, "none", "分位带"),
    ACfg("A28_TROUGH_DEEP", "P01_STRUCT_K5_E1", 10, 10, 1e6, 2, "trough_deep", 1, 0,
         "none", "追加:按谷底深度从深到浅"),
    ACfg("A29_TROUGH_SHALLOW", "P01_STRUCT_K5_E1", 10, 10, 1e6, 2, "trough_shallow", 1, 0,
         "none", "追加:按谷底深度从浅到深"),
    ACfg("A30_Q0010", "P21_STRUCT_Q0010_E1", 10, 10, 1e6, 2, "pul", 1, 0,
         "none", "追加:深分位"),
    ACfg("A31_HASH", "P01_STRUCT_K5_E1", 10, 10, 1e6, 2, "hash", 1, 0,
         "none", "追加:中性哈希选择（v1.0 §5 的排序对照）"),
]


def cmd_account(p: Panel, f: Features, ed: EventData, only=None) -> dict:
    rows, eq_all, tr_all, yr_all = [], [], [], []
    cfgs = [c for c in ACCOUNT_CFGS if (only is None or c.cid in only)]
    for cfg in cfgs:
        for win in ("dev", "val"):
            t0 = time.time()
            res = run_account(ed, cfg, win)
            rows.append(_account_metrics(res, p))
            eq = res["eq"][["nav", "cash", "market_value", "n_positions",
                            "turnover", "cost"]].copy()
            eq["cid"], eq["window"] = cfg.cid, win
            eq_all.append(eq.reset_index())
            if len(res["trades"]):
                tr = res["trades"].copy()
                tr["cid"], tr["window"] = cfg.cid, win
                tr_all.append(tr)
            r = rows[-1]
            log(f"  {cfg.cid}/{win}: n={r['n_trades']} cagr={r['cagr']:.4f} "
                f"mdd={r['mdd']:.4f} net_trade={r['avg_trade_ret']:.4f} "
                f"({time.time()-t0:.1f}s)")
            if cfg.cid == "A01_MAIN":
                try:
                    bt = STAT.BacktestResult(
                        res["eq"][["nav", "market_value", "cash"]].copy(), res["trades"],
                        meta={"initial_cash": cfg.capital})
                    at = STAT.annual_table(bt, initial_cash=cfg.capital)
                    at["window"] = win
                    yr_all.append(at)
                except Exception as exc:
                    log(f"    分年表失败: {exc}")
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "07_账户_汇总.csv", index=False, encoding="utf-8-sig")
    if eq_all:
        pd.concat(eq_all, ignore_index=True).to_parquet(OUT / "08_账户_净值.parquet", index=False)
    if tr_all:
        pd.concat(tr_all, ignore_index=True).to_parquet(OUT / "09_账户_交易.parquet", index=False)
    if yr_all:
        pd.concat(yr_all, ignore_index=True).to_csv(
            OUT / "10_账户_分年.csv", index=False, encoding="utf-8-sig")
    _write_ledger(cfgs)
    log("账户层完成")
    return {"summary": summ, "eq": eq_all, "trades": tr_all}


def _account_metrics(res: dict, p: Panel) -> dict:
    cfg = res["cfg"]
    eq = res["eq"]
    i0, i1 = p.window_index(*WINDOWS[res["window"]])
    bench = bench_ew_monthly(p, i0, i1)
    bnav = (1 + bench).cumprod()
    bnav = bnav.reindex(eq.index).ffill()
    bt = STAT.BacktestResult(
        eq[["nav", "market_value", "cash", "turnover", "cost", "n_positions"]].copy(),
        res["trades"], meta={"initial_cash": cfg["capital"]})
    m = STAT.compute_metrics(bt, initial_cash=cfg["capital"], benchmark=bnav)
    codes = int(res["trades"]["code"].nunique()) if len(res["trades"]) else 0
    out = {
        "cid": res["cid"], "window": res["window"], "policy": cfg["policy"],
        "H": cfg["H"], "slots": cfg["slots"], "capital": cfg["capital"],
        "rank": cfg["rank"], "stress": cfg["stress"], "delay": cfg["delay"],
        "gate": cfg["gate"], "note": cfg["note"],
        "n_trades": int(m.get("n_trades", 0)), "n_codes": codes,
        "cagr": m.get("cagr"), "mdd": m.get("mdd"), "calmar": m.get("calmar"),
        "sharpe": m.get("sharpe"), "vol": m.get("vol"),
        "win_rate": m.get("win_rate"), "avg_trade_ret": m.get("avg_trade_ret"),
        "median_trade_ret": m.get("median_trade_ret"),
        "avg_hold_days": m.get("avg_hold_days"), "exposure": m.get("exposure"),
        "turnover": m.get("turnover"), "cost_paid": m.get("cost_paid"),
        "final_nav": m.get("final_nav"), "bench_cagr": m.get("benchmark_cagr"),
        "excess_cagr": m.get("excess_cagr"), "beta": m.get("beta"),
        "start": m.get("start"), "end": m.get("end"),
        "cash_resid_max": res["checks"]["cash_resid_max"],
        "nav_resid_max": res["checks"]["nav_resid_max"],
        "forced": res["checks"].get("forced", 0),
        "max_date": res["checks"]["max_date"],
        "n_rows_after_2020": res["checks"]["n_rows_after_2020"],
    }
    try:
        boot = STAT.block_bootstrap(eq["nav"].pct_change().dropna(), n_boot=1000,
                                    method="stationary", seed=20261028)
        out["boot_cagr_lo"] = boot.get("cagr_lo")
        out["boot_cagr_hi"] = boot.get("cagr_hi")
        out["boot_sharpe_lo"] = boot.get("sharpe_lo")
        out["boot_p_cagr_le_0"] = boot.get("p_cagr_le_0")
    except Exception as exc:
        out["boot_cagr_lo"] = None
        out["boot_error"] = str(exc)
    return out


def _write_ledger(cfgs) -> None:
    rows = [{"cid": c.cid, "policy": c.policy, "H": c.H, "slots": c.slots,
             "capital": c.capital, "rank": c.rank, "stress": c.stress,
             "delay": c.delay, "gate": c.gate, "n_configs": 1, "note": c.note}
            for c in cfgs]
    pd.DataFrame(rows).to_csv(OUT / "12_试验台账.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def cmd_stats(summ: pd.DataFrame, eq_all: list) -> dict:
    out = {"main": summ[summ["cid"] == "A01_MAIN"].to_dict("records"),
           "n_configs": int(summ["cid"].nunique()),
           "n_runs": int(len(summ))}
    rets = {}
    for eq in eq_all:
        cid = str(eq["cid"].iloc[0])
        win = str(eq["window"].iloc[0])
        s = eq.set_index("date")["nav"].pct_change()
        rets.setdefault(cid, {})[win] = s
    cols, index = {}, None
    for cid, d in rets.items():
        s = pd.concat([d.get("dev", pd.Series(dtype=float)),
                       d.get("val", pd.Series(dtype=float))])
        s = s[~s.index.duplicated()]
        cols[cid] = s
        index = s.index if index is None else index.intersection(s.index)
    if index is not None and len(index) > 100:
        R = pd.DataFrame({k: v.reindex(index) for k, v in cols.items()}).dropna()
        if R.shape[1] >= 3:
            srs = R.mean() / R.std(ddof=1)
            out["trial_sr_daily"] = {k: float(v) for k, v in srs.items()}
            out["n_days_stitched"] = int(len(R))
            try:
                out["dsr_main"] = STAT.deflated_sharpe(
                    R["A01_MAIN"] if "A01_MAIN" in R else R.iloc[:, 0],
                    trial_srs=srs.to_numpy(), n_trials=len(srs))
            except Exception as exc:
                out["dsr_main"] = {"error": str(exc)}
            try:
                out["pbo"] = STAT.pbo_cscv(R.to_numpy(float), blocks=8, max_combinations=5000)
            except Exception as exc:
                out["pbo"] = {"error": str(exc)}
    json_dump(out, OUT / "11_统计.json")
    log(f"统计: DSR={out.get('dsr_main', {}).get('dsr')} "
        f"PBO={out.get('pbo', {}).get('pbo')}")
    return out


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all",
                    choices=["selfcheck", "events", "account", "stats", "all"])
    ap.add_argument("--only", default="", help="account 模式只跑指定 cid，逗号分隔")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if not PREREG.exists():
        raise SystemExit("缺少预注册文件，拒绝运行")
    log(f"预注册 SHA256 = {prereg_sha()}")
    t0 = time.time()
    p = load_panel()
    log(f"面板 {p.n_days} 日 × {p.n_codes} 只，{time.time()-t0:.1f}s")
    f = build_features(p)
    log(f"特征完成 {time.time()-t0:.1f}s")
    ed = prep_events(p, f)
    log(f"事件预处理完成 {time.time()-t0:.1f}s")
    if args.mode in ("selfcheck", "all"):
        cmd_selfcheck(p, f)
    if args.mode in ("events", "all"):
        cmd_events(p, f, ed)
    if args.mode in ("account", "all"):
        only = [x for x in args.only.split(",") if x] or None
        acc = cmd_account(p, f, ed, only)
        if args.mode == "all" and acc["eq"]:
            cmd_stats(acc["summary"], acc["eq"])
    if args.mode == "stats":
        summ = pd.read_csv(OUT / "07_账户_汇总.csv")
        eqdf = pd.read_parquet(OUT / "08_账户_净值.parquet")
        eq_all = [g for _, g in eqdf.groupby(["cid", "window"], sort=False)]
        cmd_stats(summ, eq_all)
    log(f"全部完成，用时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
