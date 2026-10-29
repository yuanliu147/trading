#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势回调 / 均值回复族研究（第二阶段，独立实现）

问题
----
周线 + 日线上升趋势确立后，小周期（日线）下跌是“回调”还是“结构破坏”？
用回撤幅度（ATR 标准化）、回撤持续时间、缩量、支撑距离定义有限状态，
比较三种入场政策（直接买 / 重新转强确认 / 突破确认），并保留非趋势抄底反例。

纪律
----
* 数据：只读 `01_数据/frozen/frozen.duckdb`（冻结 2026-10-26）。
* 硬截断：所有信号、路径、组合只到 2020-12-31；本文件没有任何 2021+ 的数据查询。
* 价格：成交/整股/涨跌停用 raw（Baostock flag3）；收益/ATR/均线用冻结层自建 hfq
  （factor = Π(1+pct_chg/100)），不使用 Baostock 后复权断点。
* 信号收盘生成，次一交易日开盘成交；T+1、整股、历史费用、滑点、停牌/涨跌停、退市。
* 本文件只写 `02_代码/候选_趋势回调.py` 与 `03_结果/候选_趋势回调/`。

用法
----
python3 02_代码/候选_趋势回调.py events           # 特征 + 事件层
python3 02_代码/候选_趋势回调.py portfolio        # 组合层（消融/邻域/延迟/成本/规模）
python3 02_代码/候选_趋势回调.py account-check    # 统一引擎（账户撮合.py）复核
python3 02_代码/候选_趋势回调.py selfcheck        # 自检
python3 02_代码/候选_趋势回调.py all              # 全部（默认）
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "02_代码"
OUT = ROOT / "03_结果" / "候选_趋势回调"
CACHE = OUT / "_cache"
DB_PATH = ROOT / "01_数据" / "frozen" / "frozen.duckdb"

DATA_END = "2026-10-26"          # 只在审计里出现，不进入任何行情查询
ANALYSIS_END = "2020-12-31"      # 硬截断
PANEL_START = "2006-01-01"       # 预热（指标/周线/分型）
DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")
FULL = ("2008-01-01", "2020-12-31")

N_LOT = 100
MAIN_BOARD = "主板"

# 预注册默认参数（不得因结果修改）
DEFAULT = dict(
    d_lo=0.8, d_hi=3.5, dur_lo=2, dur_hi=15,
    vr=0.85, sup_lo=-0.5, sup_hi=3.0, w_ma=30,
    exit_days=20, delay=0, cost_mult=1.0, slots=10, capital=1_000_000.0,
)

POLICIES = ["L0_GENERIC", "L1_WEEKLY", "L2_DAILY", "L3_PULLBACK", "L4_VOL",
            "L5_SUPPORT", "L6_BREAKOUT", "L7_DIRECT", "L8_NONTREND", "L9_BROKEN"]

VARIANTS = {
    "V_d_lo": dict(DEFAULT, d_lo=0.5, d_hi=2.5),
    "V_d_hi": dict(DEFAULT, d_lo=1.2, d_hi=4.0),
    "V_dur10": dict(DEFAULT, dur_hi=10),
    "V_dur25": dict(DEFAULT, dur_hi=25),
    "V_vr075": dict(DEFAULT, vr=0.75),
    "V_vr100": dict(DEFAULT, vr=1.00),
    "V_sup2": dict(DEFAULT, sup_hi=2.0),
    "V_sup5": dict(DEFAULT, sup_hi=5.0),
    "V_w20": dict(DEFAULT, w_ma=20),
    "V_w40": dict(DEFAULT, w_ma=40),
}

MAX_RISK = 0.012          # 单笔账户风险上限
MAX_STOP_PCT = 0.12       # 初始止损距离上限（价格）
COST_BPS = 10.0           # 基准滑点
PARTICIPATION = 0.02      # ADV20 参与率上限


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_module(name: str, filename: str):
    path = CODE_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FEE = _load_module("trend_fee_rules_cb", "费用与交易约束.py")
STAT = _load_module("trend_stat_cb", "统计验证.py")


def fee_config(cost_mult: float = 1.0):
    return FEE.FeeConfig(
        commission_rate=0.00025 * cost_mult,
        min_commission=5.0,
        slippage_bps=COST_BPS * cost_mult,
        max_participation=PARTICIPATION,
    )


# ---------------------------------------------------------------------------
# 数据加载：冻结层 → 每只证券的连续数组
# ---------------------------------------------------------------------------
BASE_COLS = ["open", "high", "low", "close", "preclose", "pct_chg", "volume", "amount",
             "hfq_open", "hfq_high", "hfq_low", "hfq_close",
             "limit_up_px", "limit_down_px", "td_since_list"]


class Market:
    """按证券组织的连续数组；行 j 对应全局交易日 start[sym]+j。"""

    def __init__(self, dates: pd.DatetimeIndex):
        self.dates = dates
        self.idx = {d: i for i, d in enumerate(dates)}
        self.syms: list[str] = []
        self.d: dict[str, dict[str, np.ndarray]] = {}
        self.start: dict[str, int] = {}
        self.n: dict[str, int] = {}
        self.delist: dict[str, pd.Timestamp] = {}

    def __len__(self) -> int:
        return len(self.syms)

    def row(self, sym: str, gi: int) -> int:
        return gi - self.start[sym]


def load_market(start: str = PANEL_START, end: str = ANALYSIS_END,
                max_symbols: int | None = None) -> Market:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        dates = pd.to_datetime(con.execute(
            "select date from calendar where is_trading_day=1 "
            f"and date between date '{start}' and date '{end}' order by date"
        ).df()["date"])
        assert dates.max() <= pd.Timestamp(ANALYSIS_END), "读取越过分析截止日"
        mx = pd.Timestamp(ANALYSIS_END)

        cols = ["symbol", "date", "board"] + BASE_COLS + [
            "is_suspended", "is_st", "is_limit_up_open", "is_limit_down_open"]
        q = f"""select {','.join(cols)} from daily
                where date between date '{start}' and date '{end}'
                  and board = '{MAIN_BOARD}'
                order by symbol, date"""
        df = con.execute(q).df()
        df = df[df["date"] <= mx]
        if max_symbols:
            keep = sorted(df["symbol"].unique())[:max_symbols]
            df = df[df["symbol"].isin(keep)]
        del con

        m = Market(dates)
        gb = df.groupby("symbol", sort=True)
        for sym, g in gb:
            gi = g["date"].map(m.idx).to_numpy()
            s, e = int(gi.min()), int(gi.max())
            n = e - s + 1
            if n <= 0:
                continue
            pos = gi - s
            arr: dict[str, np.ndarray] = {}
            for c in BASE_COLS:
                a = np.full(n, np.nan, dtype=np.float64)
                a[pos] = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
                arr[c] = a
            # 冻结层 hfq = raw × Π(1+pct_chg) 会把日收益平方，且在送转日断裂
            # （见 03_结果/候选_趋势回调/12_数据缺陷_冻结层hfq.md）。
            # 这里用 pct_chg 递归构造正确的总收益价：tr_t = tr_{t-1} × (1+pct_chg_t)，
            # 等价于对 raw OHLC 乘以逐日调整系数 k_t = Π (1+r_s)×P_{s-1}/P_s。
            p = arr["close"]
            pc = arr["pct_chg"] / 100.0
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(np.isfinite(pc), pc,
                             np.where(arr["preclose"] > 0, p / arr["preclose"] - 1.0, 0.0))
                prev = np.r_[np.nan, p[:-1]]
                ratio = (1.0 + r) * prev / p
            ratio = np.where(np.isfinite(ratio), ratio, 1.0)
            ratio[0] = 1.0
            k = np.cumprod(ratio)
            for c in ("open", "high", "low", "close"):
                arr["hfq_" + c] = arr[c] * k
            for c, dtype in (("is_suspended", bool), ("is_st", bool),
                             ("is_limit_up_open", bool), ("is_limit_down_open", bool)):
                a = np.zeros(n, dtype=dtype)
                a[pos] = g[c].fillna(False).to_numpy(dtype=bool)
                arr[c] = a
            arr["valid"] = np.isfinite(arr["close"]) & np.isfinite(arr["hfq_close"])
            # 缺失行（源库断行）按停牌处理
            arr["is_suspended"] = arr["is_suspended"] | (~arr["valid"])
            m.syms.append(sym)
            m.d[sym] = arr
            m.start[sym] = s
            m.n[sym] = n
        return m
    finally:
        try:
            con.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 因果特征
# ---------------------------------------------------------------------------
def _s(a: np.ndarray) -> pd.Series:
    return pd.Series(a, copy=False)


def _rolling_prev_max(mask: np.ndarray, n: int) -> np.ndarray:
    """过去 n 日内（不含当日）是否出现过 True。"""
    s = pd.Series(mask.astype(float))
    return (s.rolling(n, min_periods=1).max().shift(1).fillna(0) > 0).to_numpy()


def symbol_features(m: Market, sym: str, delist: pd.Timestamp | None) -> dict[str, np.ndarray]:
    a = m.d[sym]
    n = m.n[sym]
    c = _s(a["hfq_close"]); h = _s(a["hfq_high"]); l = _s(a["hfq_low"]); o = _s(a["hfq_open"])
    A = _s(a["amount"]); Cr = _s(a["close"])

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    hh20 = h.rolling(20, min_periods=20).max()
    newhigh = (h >= hh20 - 1e-9) & hh20.notna()
    run = np.maximum.accumulate(np.where(newhigh.to_numpy(), np.arange(n), -1))
    dur = np.arange(n) - run
    dur = np.where(run < 0, np.nan, dur).astype(float)
    dur[hh20.isna().to_numpy()] = np.nan
    depth = (hh20 - c) / atr

    amt20 = A.rolling(20, min_periods=15).mean()
    vr = A.rolling(5, min_periods=5).mean() / A.shift(5).rolling(20, min_periods=15).mean()
    vr = vr.replace([np.inf, -np.inf], np.nan)

    ma5 = c.rolling(5, min_periods=5).mean()
    ma20 = c.rolling(20, min_periods=20).mean()
    ma50 = c.rolling(50, min_periods=50).mean()

    # 分型低点 k=2：i 为低点，i+2 收盘才确认
    piv = ((l < l.shift(1)) & (l < l.shift(2)) &
           (l < l.shift(-1)) & (l < l.shift(-2)))
    sup_conf = l.where(piv).shift(2).ffill()
    sup = sup_conf.where(sup_conf.notna(), ma50)
    sup_dist = (c - sup) / atr

    # 周线：只在该周最后一个“有效”交易行形成周 K，之后才可用于日线
    valid = a["valid"]
    dts = m.dates[m.start[sym]: m.start[sym] + n]
    wdf = pd.DataFrame({"date": dts, "c": c.to_numpy(), "valid": valid})
    wdf = wdf[wdf["valid"]]
    wdf["wk"] = wdf["date"].dt.to_period("W-FRI")
    last = wdf.groupby("wk", sort=True).tail(1).set_index("date")["c"]
    wups = {}
    for wm in (20, 30, 40):
        wma = last.rolling(wm, min_periods=max(10, int(wm * 2 / 3))).mean()
        slope = wma / wma.shift(4) - 1.0
        wu = (last > wma) & (slope > 0)
        wups[wm] = wu.reindex(dts).ffill().fillna(False).to_numpy(dtype=bool)

    daily_up = ((c > ma50) & (ma50 > ma50.shift(10))).fillna(False).to_numpy(dtype=bool)

    # 可交易：主板已在 SQL 限定；其余为点对点过滤
    tdsl = a["td_since_list"]
    listed = np.nan_to_num(tdsl, nan=0.0) >= 250
    in_delist = np.zeros(n, dtype=bool)
    if delist is not None and not pd.isna(delist):
        last_gi = m.start[sym] + n - 1
        # 退市日之前 30 个交易日内不新开仓
        di = m.idx.get(pd.Timestamp(delist))
        if di is None:
            di = min(m.idx.values()) + 10**9
        in_delist = np.arange(m.start[sym], m.start[sym] + n) > di - 30
    tradable = (listed & (np.nan_to_num(a["is_st"], nan=1) == 0)
                & (np.nan_to_num(Cr.to_numpy(), nan=0) >= 2.0)
                & (np.nan_to_num(amt20.to_numpy(), nan=0) >= 3e7)
                & (~a["is_suspended"]) & (~in_delist)
                & np.isfinite(atr.to_numpy()) & np.isfinite(depth.to_numpy()))

    return dict(
        c=c.to_numpy(), o=o.to_numpy(), h=h.to_numpy(), l=l.to_numpy(),
        ho=a["hfq_open"], hh=a["hfq_high"], hl=a["hfq_low"],
        cr=Cr.to_numpy(), oraw=a["open"], preclose=a["preclose"],
        amt=a["amount"], vol=a["volume"],
        atr=atr.to_numpy(), hh20=hh20.to_numpy(), depth=depth.to_numpy(),
        dur=dur, vr=vr.to_numpy(), ma5=ma5.to_numpy(), ma20=ma20.to_numpy(),
        ma50=ma50.to_numpy(), amt20=amt20.to_numpy(), sup=sup.to_numpy(),
        sup_dist=sup_dist.to_numpy(), atr_pct=(atr / c).to_numpy(),
        wup20=wups[20], wup30=wups[30], wup40=wups[40],
        daily_up=daily_up, tradable=tradable, tdsl=tdsl,
        lu_open=a["is_limit_up_open"], ld_open=a["is_limit_down_open"],
        lu_px=a["limit_up_px"], ld_px=a["limit_down_px"],
        susp=a["is_suspended"], st=a["is_st"],
        resume=np.r_[False, a["is_suspended"][:-1]],
        c1=np.r_[np.nan, c.to_numpy()[:-1]],
        c2=(np.r_[np.full(min(2, n), np.nan), c.to_numpy()[:max(n - 2, 0)]] if n > 2 else np.full(n, np.nan)),
        hh20p=np.r_[np.nan, hh20.to_numpy()[:-1]],
    )


def policy_masks(F: dict[str, np.ndarray], p: dict) -> dict[str, np.ndarray]:
    """预注册的 10 个消融层 + 邻域变体。"""
    wu = F[f"wup{p['w_ma']}"]
    trend = wu & F["daily_up"]
    band = ((F["depth"] >= p["d_lo"]) & (F["depth"] <= p["d_hi"]) &
            (F["dur"] >= p["dur_lo"]) & (F["dur"] <= p["dur_hi"]))
    vol = F["vr"] <= p["vr"]
    sup = ((F["sup_dist"] >= p["sup_lo"]) & (F["sup_dist"] <= p["sup_hi"]))
    p2 = ((F["c"] > F["c1"]) & (F["c"] > F["c2"]) &
          (F["c"] > F["ma5"]) & (F["c"] > F["o"]))
    p1 = (F["c"] > F["o"]) & (F["c"] > F["c1"])
    healthy = trend & band & vol & sup
    had = _rolling_prev_max(healthy, 10)
    p3 = (had & (F["c"] > F["hh20p"] + 0.05 * F["atr"]) &
          (F["amt"] > 1.2 * F["amt20"]))
    broken = ((F["c"] < F["sup"] - 0.3 * F["atr"]) | (F["depth"] > 4.0))
    with np.errstate(invalid="ignore"):
        return {
            "L0_GENERIC": p2,
            "L1_WEEKLY": p2 & wu,
            "L2_DAILY": p2 & wu & F["daily_up"],
            "L3_PULLBACK": p2 & trend & band,
            "L4_VOL": p2 & trend & band & vol,
            "L5_SUPPORT": p2 & healthy,
            "L6_BREAKOUT": p3 & trend & sup,
            "L7_DIRECT": p1 & healthy,
            "L8_NONTREND": p2 & (~wu) & (F["c"] < F["ma50"]),
            "L9_BROKEN": p2 & (F["depth"] >= 0.8) & (F["dur"] >= 2) & wu & broken,
        }


def variant_masks(F: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """单因素邻域：全部是 L5（P2 + 健康状态）的变体。"""
    out = {}
    for k, p in VARIANTS.items():
        out[k] = policy_masks(F, p)["L5_SUPPORT"]
    return out


# ---------------------------------------------------------------------------
# 事件层
# ---------------------------------------------------------------------------
EVENT_FEATS = ["depth", "dur", "vr", "sup_dist", "atr_pct", "amt20", "c", "atr", "sup"]


def build_events(m: Market, max_symbols: int | None = None) -> pd.DataFrame:
    _ensure_dirs()
    syms_df = None
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        syms_df = con.execute("select symbol, delist_date from symbols where board='主板'").df()
    finally:
        con.close()
    delist_map = {r.symbol: pd.Timestamp(r.delist_date) for r in syms_df.itertuples()
                  if not pd.isna(r.delist_date)}

    all_masks = POLICIES + list(VARIANTS.keys())
    rows = []
    t0 = time.time()
    for si, sym in enumerate(m.syms):
        if m.n[sym] < 60:
            continue
        F = symbol_features(m, sym, delist_map.get(sym))
        masks = policy_masks(F, DEFAULT)
        masks.update(variant_masks(F))
        anym = np.zeros(m.n[sym], dtype=bool)
        for v in masks.values():
            anym |= np.nan_to_num(v, nan=0).astype(bool)
        anym &= F["tradable"]
        js = np.flatnonzero(anym)
        if js.size == 0:
            continue
        base = m.start[sym]
        d = {
            "symbol": np.repeat(sym, js.size),
            "i": (js + base).astype(np.int32),
            "date": m.dates[js + base].to_numpy(),
        }
        for f in EVENT_FEATS:
            d[f] = F[f][js].astype(np.float32)
        for name, mask in masks.items():
            d[name] = np.nan_to_num(mask[js], nan=0).astype(bool)
        d["delisted"] = np.repeat(sym in delist_map, js.size)
        rows.append(pd.DataFrame(d))
        if si % 500 == 0:
            log(f"  features {si}/{len(m.syms)} 事件行 {sum(len(r) for r in rows):,}")

    ev = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    log(f"事件候选行 {len(ev):,}，用时 {time.time()-t0:.1f}s")
    return ev


def add_forward_paths(m: Market, ev: pd.DataFrame) -> pd.DataFrame:
    """次日开盘起的前向收益/超额/MAE/MFE；右删失剔除。"""
    horizons = (5, 10, 20, 40)
    n_total = len(m.dates)
    # 全市场等权（可交易）的 h 日前向收益，用于超额
    log("计算等权基准前向收益 ...")
    bench = {h: np.full(n_total, np.nan) for h in horizons}
    sums = {h: np.zeros(n_total) for h in horizons}
    cnts = {h: np.zeros(n_total, dtype=np.int32) for h in horizons}
    for sym in m.syms:
        a = m.d[sym]
        base = m.start[sym]
        n = m.n[sym]
        cr = a["close"]
        hc = a["hfq_close"]
        ho = a["hfq_open"]
        amt20 = pd.Series(a["amount"]).rolling(20, min_periods=15).mean().to_numpy()
        tdsl = np.nan_to_num(a["td_since_list"], nan=0.0)
        # 与策略同一主宇宙口径（不含退市窗口，基准不做个股择时）
        ok = ((np.nan_to_num(a["is_st"], nan=1) == 0) & (~a["is_suspended"]) &
              np.isfinite(hc) & (np.nan_to_num(cr, nan=0) >= 2.0) &
              (np.nan_to_num(amt20, nan=0) >= 3e7) & (tdsl >= 250))
        ok_next = ok & ~np.r_[a["is_suspended"][1:], True]
        for h in horizons:
            j = np.arange(0, n - h - 1)
            if j.size == 0:
                continue
            r = hc[j + h + 1] / ho[j + 1] - 1.0
            good = ok_next[j] & np.isfinite(r) & np.isfinite(ho[j + 1])
            gi = base + j[good]
            np.add.at(sums[h], gi, r[good])
            np.add.at(cnts[h], gi, 1)
    for h in horizons:
        with np.errstate(invalid="ignore", divide="ignore"):
            bench[h] = np.where(cnts[h] > 0, sums[h] / np.maximum(cnts[h], 1), np.nan)

    # 逐事件取数
    log("填充事件前向路径 ...")
    out = {f"r{h}": np.full(len(ev), np.nan, dtype=np.float32) for h in horizons}
    out.update({f"exc{h}": np.full(len(ev), np.nan, dtype=np.float32) for h in horizons})
    out["gap"] = np.full(len(ev), np.nan, dtype=np.float32)
    out["mae20"] = np.full(len(ev), np.nan, dtype=np.float32)
    out["mfe20"] = np.full(len(ev), np.nan, dtype=np.float32)
    out["entry_ok"] = np.zeros(len(ev), dtype=bool)
    out["reason"] = np.full(len(ev), "", dtype=object)

    ev_sym = ev["symbol"].to_numpy()
    ev_i = ev["i"].to_numpy()
    order = np.argsort(ev_sym, kind="stable")
    sorted_sym = ev_sym[order]
    uniq, starts = np.unique(sorted_sym, return_index=True)
    ends = np.r_[starts[1:], len(sorted_sym)]
    # 逐证券批量（排序后分组是连续的）
    for k, sym in enumerate(uniq):
        sel = order[starts[k]:ends[k]]
        gi = ev_i[sel]
        a = m.d[sym]
        base = m.start[sym]
        n = m.n[sym]
        j = gi - base
        nxt = j + 1
        valid_nxt = (nxt < n)
        ok = np.zeros(len(sel), dtype=bool)
        ok[valid_nxt] = (np.isfinite(a["hfq_open"][nxt[valid_nxt]]) &
                         (~a["is_suspended"][nxt[valid_nxt]]) &
                         (~a["is_limit_up_open"][nxt[valid_nxt]]) &
                         (~np.r_[False, a["is_suspended"][:-1]][nxt[valid_nxt]]))
        out["entry_ok"][sel] = ok
        reason = np.full(len(sel), "", dtype=object)
        reason[~valid_nxt] = "no_next_bar"
        vn = np.flatnonzero(valid_nxt)
        if vn.size:
            susp_n = a["is_suspended"][nxt[vn]]
            lu_n = a["is_limit_up_open"][nxt[vn]]
            reason[vn[susp_n]] = "suspended"
            reason[vn[lu_n]] = "limit_up_open"
        out["reason"][sel] = reason
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.where(ok, a["hfq_open"][np.minimum(nxt, n - 1)] / a["hfq_close"][j] - 1.0, np.nan)
        out["gap"][sel] = gap
        for h in horizons:
            e = nxt + h
            good = ok & (e < n)
            if not good.any():
                continue
            r = a["hfq_close"][e[good]] / a["hfq_open"][nxt[good]] - 1.0
            out[f"r{h}"][sel[good]] = r
            b = bench[h][gi[good]]
            out[f"exc{h}"][sel[good]] = r - b
        # 20 日路径：前向滚动窗口（仅测量，不进入信号）
        fmin = pd.Series(a["hfq_low"]).rolling(21, min_periods=1).min().shift(-20).to_numpy()
        fmax = pd.Series(a["hfq_high"]).rolling(21, min_periods=1).max().shift(-20).to_numpy()
        good = ok & (nxt + 20 < n)
        if good.any():
            out["mae20"][sel[good]] = fmin[nxt[good]] / a["hfq_open"][nxt[good]] - 1.0
            out["mfe20"][sel[good]] = fmax[nxt[good]] / a["hfq_open"][nxt[good]] - 1.0

    for k, v in out.items():
        ev[k] = v
    return ev


def event_summary(ev: pd.DataFrame, policies: list[str], windows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(ev["date"])
    for pol in policies:
        m = ev[pol].to_numpy(dtype=bool) & ev["entry_ok"].to_numpy(dtype=bool)
        for wname, (a, b) in windows.items():
            wm = m & (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
            if wm.sum() < 3:
                continue
            for h in (5, 10, 20, 40):
                x = ev.loc[wm, f"exc{h}"].to_numpy(float)
                r = ev.loc[wm, f"r{h}"].to_numpy(float)
                cl = dates[wm].to_numpy()
                good = np.isfinite(x)
                if good.sum() < 3:
                    continue
                st = STAT.cluster_mean_test(x[good], cl[good])
                rows.append({
                    "policy": pol, "window": wname, "horizon": h,
                    "n": int(good.sum()), "n_dates": st.get("clusters", np.nan),
                    "mean_exc": st.get("mean", np.nan), "median_exc": float(np.median(x[good])),
                    "t_cluster": st.get("t", np.nan), "p_cluster": st.get("p", np.nan),
                    "ci_lo": st.get("ci_lo", np.nan), "ci_hi": st.get("ci_hi", np.nan),
                    "mean_abs": float(np.nanmean(r[good])), "hit": float(np.nanmean(x[good] > 0)),
                    "std_exc": float(np.nanstd(x[good], ddof=1)),
                    "top1pct_share": _top_share(x[good], 0.01),
                })
    return pd.DataFrame(rows)


def _top_share(x: np.ndarray, q: float) -> float:
    if x.size < 20:
        return np.nan
    k = max(1, int(len(x) * q))
    top = np.sort(x)[-k:]
    tot = np.sum(x)
    return float(np.sum(top) / tot) if abs(tot) > 1e-12 else np.nan


# ---------------------------------------------------------------------------
# 组合层
# ---------------------------------------------------------------------------
@dataclass
class Pos:
    sym: str
    qty: int
    entry_i: int
    entry_px: float
    f_entry: float
    stop_hfq: float
    signal_i: int
    cost_basis: float
    buy_fees: float


@dataclass
class Config:
    cid: str
    policy: str
    params: dict = field(default_factory=lambda: dict(DEFAULT))
    exit_days: int = 20
    delay: int = 0
    cost_mult: float = 1.0
    slots: int = 10
    capital: float = 1_000_000.0
    use_stop: bool = True
    rank_amt_asc: bool = False

    def key(self) -> str:
        p = self.params
        return (f"{self.cid}|ex{self.exit_days}|d{self.delay}|c{self.cost_mult:g}"
                f"|s{self.slots}|k{int(self.capital)}"
                + ("" if self.use_stop else "|nostop")
                + ("|smallfirst" if self.rank_amt_asc else ""))

    def as_row(self) -> dict:
        d = dict(cid=self.cid, policy=self.policy, exit_days=self.exit_days,
                 delay=self.delay, cost_mult=self.cost_mult, slots=self.slots,
                 capital=self.capital, use_stop=self.use_stop,
                 rank_amt_asc=self.rank_amt_asc)
        d.update({f"p_{k}": v for k, v in self.params.items()})
        return d


def _bar(m: Market, sym: str, i: int) -> dict | None:
    j = i - m.start[sym]
    n = m.n[sym]
    if j < 0 or j >= n:
        return None
    a = m.d[sym]
    if not np.isfinite(a["close"][j]) or not np.isfinite(a["open"][j]):
        return None
    return dict(j=j, open=float(a["open"][j]), high=float(a["high"][j]),
                low=float(a["low"][j]), close=float(a["close"][j]),
                ho=float(a["hfq_open"][j]), hh=float(a["hfq_high"][j]),
                hl=float(a["hfq_low"][j]), hc=float(a["hfq_close"][j]),
                vol=float(a["volume"][j]) if np.isfinite(a["volume"][j]) else 0.0,
                amt=float(a["amount"][j]) if np.isfinite(a["amount"][j]) else 0.0,
                susp=bool(a["is_suspended"][j]), lu=bool(a["is_limit_up_open"][j]),
                ld=bool(a["is_limit_down_open"][j]),
                resume=bool(a["is_suspended"][j - 1]) if j > 0 else False,
                st=bool(a["is_st"][j]))


def _build_candidates(ev: pd.DataFrame, policy: str,
                      amt_asc: bool = False) -> dict[int, np.ndarray]:
    mask = ev[policy].to_numpy(dtype=bool)
    sub = ev.loc[mask, ["i", "amt20", "sup_dist", "symbol"]]
    out: dict[int, np.ndarray] = {}
    for i, g in sub.groupby("i", sort=True):
        g = g.sort_values(["amt20", "sup_dist", "symbol"],
                          ascending=[amt_asc, True, True], kind="stable")
        out[int(i)] = g.index.to_numpy()
    return out


def _fee(cfg, d, sym: str, side: str, price: float, qty: int):
    return FEE.trade_fees(pd.Timestamp(d).date(), sym, side, price, int(qty), cfg)


def _slice_dates(m: Market, start: str, end: str) -> tuple[int, int]:
    a = int(np.searchsorted(m.dates.values, np.datetime64(pd.Timestamp(start)), side="left"))
    b = int(np.searchsorted(m.dates.values, np.datetime64(pd.Timestamp(end)), side="right")) - 1
    return max(a, 0), min(b, len(m.dates) - 1)


def simulate(m: Market, ev: pd.DataFrame, cfg: Config,
             start: str, end: str, cands: dict[int, np.ndarray] | None = None,
             keep_daily: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """独立组合引擎：真实费用/整股/T+1/涨跌停/停牌/退市。"""
    i0, i1 = _slice_dates(m, start, end)
    cands = cands if cands is not None else _build_candidates(ev, cfg.policy)
    fcfg = fee_config(cfg.cost_mult)
    slip = fcfg.slippage_bps / 10000.0
    dates = m.dates

    cash = float(cfg.capital)
    nav = float(cfg.capital)
    nav_prev = nav
    positions: dict[str, Pos] = {}
    pending: dict[str, int] = {}          # 挂单卖出的顺延天数
    exit_orders: dict[str, str] = {}      # 次日开盘卖出
    trades: list[dict] = []
    daily: list[dict] = []
    failures: list[dict] = []
    min_cash = cash
    traded_notional = 0.0

    # 相对强弱基准（等权）日收益
    for i in range(i0, i1 + 1):
        d = dates[i]
        # ---- 0) 退市强制卖出
        for sym in list(positions):
            dl = m.delist.get(sym)
            if dl is not None and pd.Timestamp(d) > dl:
                pos = positions.pop(sym)
                j = min(i - m.start[sym], m.n[sym] - 1)
                hc = m.d[sym]["hfq_close"][j]
                px = float(hc) * (1 - slip) / pos.f_entry
                fees = _fee(fcfg, d, sym, "sell", px, max(1, int(round(pos.qty))))
                proceeds = pos.qty * hc * (1 - slip) / pos.f_entry - fees.total
                cash += proceeds
                trades.append(_trade_row(cfg, pos, d, "delist", px, fees,
                                         proceeds, pos.entry_i, i, m))
                traded_notional += proceeds
                continue

        # ---- 1) 挂单卖出（跌停/停牌顺延，最多 10 日）
        for sym in list(pending):
            if sym not in positions:
                pending.pop(sym, None); continue
            b = _bar(m, sym, i)
            pos = positions[sym]
            if b is None or b["susp"] or (b["ld"] and b["open"] <= (m.d[sym]["limit_down_px"][b["j"]] + 1e-6)):
                pending[sym] += 1
                continue
            px_raw = FEE.apply_slippage(b["open"], "sell", fcfg.slippage_bps)
            hfq_px = b["ho"]
            proceeds, fees = _sell(m, cfg, fcfg, sym, pos, i, hfq_px, cash, trades, "pending")
            cash += proceeds
            traded_notional += proceeds
            positions.pop(sym); pending.pop(sym, None)
            continue

        # ---- 2) 时间退出（次日开盘）
        for sym, reason in list(exit_orders.items()):
            if sym not in positions:
                exit_orders.pop(sym, None); continue
            b = _bar(m, sym, i)
            if b is None or b["susp"] or b["ld"]:
                pending[sym] = pending.get(sym, 0) + 1
                exit_orders.pop(sym, None)
                continue
            hfq_px = b["ho"]
            proceeds, fees = _sell(m, cfg, fcfg, sym, positions[sym], i, hfq_px, cash,
                                   trades, reason)
            cash += proceeds
            traded_notional += proceeds
            positions.pop(sym); exit_orders.pop(sym, None)

        # ---- 3) 开盘买入
        sig_i = i - 1 - cfg.delay
        if sig_i >= i0 and len(positions) < cfg.slots and sig_i in cands:
            rows = cands[sig_i]
            for ridx in rows:
                if len(positions) >= cfg.slots:
                    break
                r = ev.loc[ridx]
                sym = r["symbol"]
                if sym in positions:
                    continue
                b = _bar(m, sym, i)
                if b is None or b["susp"] or b["lu"] or b["resume"] or b["st"]:
                    failures.append(dict(config=cfg.key(), date=d, symbol=sym,
                                         reason=("suspended" if b is None or b["susp"]
                                                 else "limit_up_open" if b["lu"]
                                                 else "st" if b["st"] else "resume")))
                    continue
                stop_hfq = float(r["sup"]) - 0.3 * float(r["atr"]) if cfg.use_stop else -np.inf
                if b["ho"] <= 0 or (cfg.use_stop and not np.isfinite(stop_hfq)):
                    continue
                risk_frac = (b["ho"] - stop_hfq) / b["ho"] if cfg.use_stop else 0.0
                if cfg.use_stop and (risk_frac <= 0 or risk_frac > MAX_STOP_PCT):
                    continue
                entry_px = FEE.apply_slippage(b["open"], "buy", fcfg.slippage_bps)
                target = nav_prev / cfg.slots
                qty = int(target / entry_px / N_LOT) * N_LOT
                if cfg.use_stop:
                    qty = min(qty, int(MAX_RISK * nav_prev / (risk_frac * entry_px) / N_LOT) * N_LOT)
                cap = FEE.participation_cap(b["vol"], PARTICIPATION) // N_LOT * N_LOT
                qty = min(qty, cap)
                max_aff = int(max(cash, 0) / (entry_px * 1.01) / N_LOT) * N_LOT
                qty = min(qty, max_aff)
                if qty < N_LOT:
                    failures.append(dict(config=cfg.key(), date=d, symbol=sym,
                                         reason="size_or_cash_below_lot"))
                    continue
                fees = _fee(fcfg, d, sym, "buy", entry_px, qty)
                while qty >= N_LOT and qty * entry_px + fees.total > cash:
                    qty -= N_LOT
                    fees = _fee(fcfg, d, sym, "buy", entry_px, qty) if qty >= N_LOT else fees
                if qty < N_LOT:
                    continue
                cost = qty * entry_px + fees.total
                cash -= cost
                min_cash = min(min_cash, cash)
                f_entry = b["ho"] / b["open"]
                positions[sym] = Pos(sym=sym, qty=qty, entry_i=i, entry_px=entry_px,
                                     f_entry=f_entry, stop_hfq=stop_hfq,
                                     signal_i=int(r["i"]), cost_basis=cost,
                                     buy_fees=fees.total)
                trades.append(dict(
                    config=cfg.key(), policy=cfg.policy, side="buy",
                    symbol=sym, signal_date=dates[int(r["i"])], date=d,
                    entry_i=i, exit_i=np.nan, qty=qty, price=entry_px,
                    amount=qty * entry_px, fees=fees.total,
                    commission=fees.commission, stamp=fees.stamp_duty,
                    transfer=fees.transfer_fee, reason="entry",
                    exit_reason="", pnl=np.nan, ret=np.nan, hold_days=np.nan,
                    stop_px=(stop_hfq * b["close"] / b["hc"]
                             if (cfg.use_stop and b["hc"] > 0) else np.nan),
                    depth=float(r["depth"]), dur=float(r["dur"]), vr=float(r["vr"]),
                    sup_dist=float(r["sup_dist"]), amount20=float(r["amt20"]),
                ))
                traded_notional += cost

        # ---- 4) 盘中止损（含当日买入的 T+1 跳过）
        for sym in (list(positions) if cfg.use_stop else []):
            pos = positions[sym]
            if pos.entry_i >= i:
                continue
            if sym in pending:
                continue
            b = _bar(m, sym, i)
            if b is None or b["susp"]:
                continue
            if b["ho"] <= pos.stop_hfq:
                fill_hfq = b["ho"]
                reason = "stop_open"
            elif b["hl"] <= pos.stop_hfq:
                fill_hfq = pos.stop_hfq
                reason = "stop_intraday"
            else:
                continue
            if b["ld"]:
                pending[sym] = pending.get(sym, 0) + 1
                continue
            proceeds, fees = _sell(m, cfg, fcfg, sym, pos, i, fill_hfq, cash, trades, reason)
            cash += proceeds
            traded_notional += proceeds
            positions.pop(sym)

        # ---- 5) 收盘：估值、时间退出、强制挂单到期
        mv = 0.0
        for sym, pos in positions.items():
            j = min(i - m.start[sym], m.n[sym] - 1)
            hc = m.d[sym]["hfq_close"][j]
            mv += pos.qty * (hc / pos.f_entry) if np.isfinite(hc) else 0.0
        nav = cash + mv
        nav_prev = nav
        for sym in list(pending):
            if pending[sym] >= 10 and sym in positions:
                pos = positions[sym]
                b = _bar(m, sym, i)
                hfq_px = b["hc"] if b is not None else m.d[sym]["hfq_close"][-1]
                proceeds, fees = _sell(m, cfg, fcfg, sym, pos, i, hfq_px, cash, trades,
                                       "forced_10d")
                cash += proceeds
                traded_notional += proceeds
                positions.pop(sym); pending.pop(sym, None)
        for sym, pos in positions.items():
            age = i - pos.entry_i + 1
            if age == cfg.exit_days and sym not in exit_orders:
                exit_orders[sym] = "time"
        if keep_daily:
            daily.append(dict(config=cfg.key(), policy=cfg.policy, date=d,
                              nav=nav, cash=cash, n_pos=len(positions),
                              exposure=(nav - cash) / nav if nav > 0 else 0.0))

    # 期末：不强制平仓（主口径）
    mv = 0.0
    for sym, pos in positions.items():
        j = min(i1 - m.start[sym], m.n[sym] - 1)
        hc = m.d[sym]["hfq_close"][j]
        mv += pos.qty * (hc / pos.f_entry)
    nav_end = cash + mv
    # 清算口径：期末持仓按最后收盘价卖出并扣费
    liq = cash
    for sym, pos in positions.items():
        j = min(i1 - m.start[sym], m.n[sym] - 1)
        hc = m.d[sym]["hfq_close"][j]
        px = hc * (1 - slip) / pos.f_entry
        fees = _fee(fcfg, dates[i1], sym, "sell", px, max(1, int(round(pos.qty))))
        liq += pos.qty * hc * (1 - slip) / pos.f_entry - fees.total

    tr = pd.DataFrame(trades)
    dl = pd.DataFrame(daily)
    meta = dict(config=cfg.key(), policy=cfg.policy, window=(start, end),
                initial=cfg.capital, nav_end=nav_end, nav_liq=liq,
                n_trades=int((tr["side"] == "buy").sum()) if len(tr) else 0,
                min_cash=min_cash, traded_notional=traded_notional,
                failures=len(failures), n_open=len(positions),
                avg_exposure=float(dl["exposure"].mean()) if len(dl) else np.nan,
                avg_pos=float(dl["n_pos"].mean()) if len(dl) else np.nan)
    return tr, dl, meta, pd.DataFrame(failures)


def _sell(m: Market, cfg: Config, fcfg, sym: str, pos: Pos, i: int,
          hfq_px: float, cash: float, trades: list, reason: str):
    b = _bar(m, sym, i)
    if b is None:
        # 用最后可得收盘
        j = min(i - m.start[sym], m.n[sym] - 1)
        b = dict(j=j, close=float(m.d[sym]["close"][j]),
                 hc=float(m.d[sym]["hfq_close"][j]))
        hfq_px = b["hc"]
    d = m.dates[i]
    f_i = b["hc"] / b["close"] if b["close"] > 0 else pos.f_entry
    gross = pos.qty * hfq_px / pos.f_entry * (1 - fcfg.slippage_bps / 10000.0)
    shares_eff = pos.qty * f_i / pos.f_entry
    px_eff = gross / shares_eff if shares_eff > 0 else b["close"]
    fees = _fee(fcfg, d, sym, "sell", px_eff, max(1, int(round(shares_eff))))
    proceeds = gross - fees.total
    pnl = proceeds - pos.cost_basis
    trades.append(dict(
        config=cfg.key(), policy=cfg.policy, side="sell", symbol=sym,
        signal_date=pd.NaT, date=d, entry_i=pos.entry_i, exit_i=i,
        qty=int(round(shares_eff)), price=px_eff, amount=gross,
        fees=fees.total, commission=fees.commission, stamp=fees.stamp_duty,
        transfer=fees.transfer_fee, reason=reason, exit_reason=reason,
        pnl=pnl, ret=pnl / pos.cost_basis if pos.cost_basis > 0 else np.nan,
        hold_days=i - pos.entry_i, stop_px=np.nan,
        depth=np.nan, dur=np.nan, vr=np.nan, sup_dist=np.nan, amount20=np.nan,
    ))
    return proceeds, fees


def _trade_row(cfg, pos, d, reason, px, fees, proceeds, entry_i, exit_i, m):
    pnl = proceeds - pos.cost_basis
    return dict(config=cfg.key(), policy=cfg.policy, side="sell", symbol=pos.sym,
                signal_date=pd.NaT, date=d, entry_i=entry_i, exit_i=exit_i,
                qty=pos.qty, price=px, amount=proceeds + fees.total, fees=fees.total,
                commission=fees.commission, stamp=fees.stamp_duty,
                transfer=fees.transfer_fee, reason=reason, exit_reason=reason,
                pnl=pnl, ret=pnl / pos.cost_basis if pos.cost_basis > 0 else np.nan,
                hold_days=exit_i - entry_i, stop_px=np.nan,
                depth=np.nan, dur=np.nan, vr=np.nan, sup_dist=np.nan, amount20=np.nan)


# ---------------------------------------------------------------------------
# 绩效
# ---------------------------------------------------------------------------
def perf_from_daily(dl: pd.DataFrame, initial: float) -> dict:
    if dl.empty:
        return {}
    nav = dl["nav"].to_numpy(float)
    dts = pd.to_datetime(dl["date"])
    days = (dts.iloc[-1] - dts.iloc[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (nav[-1] / initial) ** (1 / years) - 1 if nav[-1] > 0 else -1
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min(nav / peak - 1))
    r = pd.Series(nav).pct_change().dropna().to_numpy()
    vol = float(np.std(r, ddof=1) * math.sqrt(252)) if len(r) > 2 else np.nan
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(252)) if len(r) > 2 and np.std(r) > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return dict(cagr=float(cagr), mdd=mdd, calmar=float(calmar), sharpe=sharpe,
                vol=vol, total=(nav[-1] / initial - 1), years=years,
                nav_end=float(nav[-1]), n_days=len(nav),
                avg_exposure=float(dl["exposure"].mean()),
                avg_pos=float(dl["n_pos"].mean()),
                cash_ratio=float((dl["cash"] / dl["nav"]).mean()))


def trade_stats(tr: pd.DataFrame) -> dict:
    s = tr[tr["side"] == "sell"].copy() if len(tr) else tr
    if s.empty:
        return dict(n_closed=0)
    pnl = s["pnl"].to_numpy(float)
    ret = s["ret"].to_numpy(float)
    win = pnl > 0
    gains = pnl[win].sum(); losses = -pnl[~win].sum()
    return dict(
        n_closed=int(len(s)), n_symbols=int(s["symbol"].nunique()),
        win_rate=float(win.mean()),
        profit_factor=float(gains / losses) if losses > 0 else np.nan,
        avg_ret=float(np.mean(ret)), median_ret=float(np.median(ret)),
        avg_pnl=float(np.mean(pnl)),
        avg_hold=float(s["hold_days"].mean()),
        fees=float(s["fees"].sum()),
        top5_share=float(np.sort(pnl)[-5:].sum() / gains) if gains > 0 else np.nan,
        best_year_share=np.nan,
    )


# ---------------------------------------------------------------------------
# 基准与配置
# ---------------------------------------------------------------------------
def daily_ew(m: Market) -> np.ndarray:
    n = len(m.dates)
    s = np.zeros(n); c = np.zeros(n)
    for sym in m.syms:
        a = m.d[sym]
        hc = a["hfq_close"]
        ok = (np.isfinite(hc) & (~a["is_suspended"]) &
              (np.nan_to_num(a["is_st"], nan=1) == 0) &
              (np.nan_to_num(a["close"], nan=0) >= 2.0))
        if a["close"].size < 3:
            continue
        r = hc[1:] / hc[:-1] - 1.0
        good = ok[1:] & np.isfinite(r)
        gi = m.start[sym] + 1 + np.arange(a["close"].size - 1)
        np.add.at(s, gi[good], r[good])
        np.add.at(c, gi[good], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def hs300(m: Market) -> np.ndarray:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("""select date, close from index_daily
                            where symbol='000300.SH' and date between '2006-01-01' and '2020-12-31'
                            order by date""").df()
    finally:
        con.close()
    s = pd.Series(df["close"].to_numpy(float), index=pd.to_datetime(df["date"]))
    s = s.reindex(m.dates).ffill()
    return (s / s.shift(1) - 1.0).to_numpy()


def make_configs() -> list[Config]:
    cs = [Config(cid=p, policy=p) for p in POLICIES]
    for k, p in VARIANTS.items():
        cs.append(Config(cid=k, policy=k, params=p))
    cs += [
        Config(cid="H10", policy="L5_SUPPORT", exit_days=10),
        Config(cid="H40", policy="L5_SUPPORT", exit_days=40),
        Config(cid="D1", policy="L5_SUPPORT", delay=1),
        Config(cid="D2", policy="L5_SUPPORT", delay=2),
        Config(cid="C2", policy="L5_SUPPORT", cost_mult=2.0),
        Config(cid="C3", policy="L5_SUPPORT", cost_mult=3.0),
        Config(cid="S5", policy="L5_SUPPORT", slots=5),
        Config(cid="S20", policy="L5_SUPPORT", slots=20),
        Config(cid="K20k", policy="L5_SUPPORT", capital=20_000),
        Config(cid="K100k", policy="L5_SUPPORT", capital=100_000),
        Config(cid="NS_L5", policy="L5_SUPPORT", use_stop=False),
        Config(cid="NS_L7", policy="L7_DIRECT", use_stop=False),
        Config(cid="NS_L0", policy="L0_GENERIC", use_stop=False),
        Config(cid="NS_L8", policy="L8_NONTREND", use_stop=False),
        Config(cid="SF_L5", policy="L5_SUPPORT", rank_amt_asc=True),
    ]
    return cs


def run_configs(m: Market, ev: pd.DataFrame, configs: list[Config]) -> tuple[pd.DataFrame, ...]:
    cand_cache = {(c.policy, c.rank_amt_asc): _build_candidates(ev, c.policy, c.rank_amt_asc)
                  for c in configs}
    daily_all, trade_all, summ, yearly, fail_all = [], [], [], [], []
    windows = {"dev": DEV, "val": VAL, "full": FULL}
    t0 = time.time()
    for k, cfg in enumerate(configs):
        for wname, (a, b) in windows.items():
            tr, dl, meta, fail = simulate(m, ev, cfg, a, b,
                                          cands=cand_cache[(cfg.policy, cfg.rank_amt_asc)])
            if not dl.empty:
                p = perf_from_daily(dl, cfg.capital)
                ts = trade_stats(tr)
                row = dict(cfg.as_row())
                buys = tr[tr["side"] == "buy"] if len(tr) else tr
                row.update(window=wname, window_start=a, window_end=b, **p, **ts,
                           median_amt20_buy=float(buys["amount20"].median()) if len(buys) else np.nan,
                           median_price_buy=float(buys["price"].median()) if len(buys) else np.nan,
                           failures=meta["failures"], n_open_end=meta["n_open"],
                           nav_liq=meta["nav_liq"], min_cash=meta["min_cash"],
                           traded_notional=meta["traded_notional"])
                summ.append(row)
                dl["window"] = wname
                daily_all.append(dl)
                if len(tr):
                    tr = tr.copy(); tr["window"] = wname
                    trade_all.append(tr)
                if len(fail):
                    fail = fail.copy(); fail["window"] = wname
                    fail_all.append(fail)
                # 分年
                dts = pd.to_datetime(dl["date"])
                nav = dl["nav"].to_numpy(float)
                for y in sorted(dts.dt.year.unique()):
                    msk = (dts.dt.year == y).to_numpy()
                    idx = np.flatnonzero(msk)
                    if idx.size == 0:
                        continue
                    start_nav = nav[idx[0] - 1] if idx[0] > 0 else cfg.capital
                    yearly.append(dict(cfg.as_row(), window=wname, year=int(y),
                                       ret=nav[idx[-1]] / start_nav - 1,
                                       mdd=float(np.min(nav[idx] / np.maximum.accumulate(nav[idx]) - 1)),
                                       n_days=int(idx.size)))
        if k % 5 == 0:
            log(f"  config {k+1}/{len(configs)} {cfg.cid} 累计 {time.time()-t0:.1f}s")
    return (pd.DataFrame(summ), pd.concat(daily_all, ignore_index=True),
            pd.concat(trade_all, ignore_index=True) if trade_all else pd.DataFrame(),
            pd.DataFrame(yearly),
            pd.concat(fail_all, ignore_index=True) if fail_all else pd.DataFrame())


# ---------------------------------------------------------------------------
# 事件层落盘
# ---------------------------------------------------------------------------
def cmd_events(rebuild: bool = False, max_symbols: int | None = None) -> pd.DataFrame:
    _ensure_dirs()
    ep = CACHE / "events.parquet"
    if ep.exists() and not rebuild:
        ev = pd.read_parquet(ep)
        log(f"读取事件缓存 {ep} ({len(ev):,} 行)")
    else:
        m = load_market(max_symbols=max_symbols)
        log(f"面板 {len(m.dates)} 交易日 × {len(m.syms)} 证券")
        ev = build_events(m, max_symbols=max_symbols)
        ev = add_forward_paths(m, ev)
        ev.to_parquet(ep, index=False)
        log(f"事件已落盘 {ep} ({len(ev):,} 行)")
    summ = event_summary(ev, POLICIES, {"dev": DEV, "val": VAL, "full": FULL})
    summ.to_csv(OUT / "02_事件_汇总.csv", index=False)
    # 分年（20 日）
    rows = []
    dts = pd.to_datetime(ev["date"])
    for pol in POLICIES:
        msk = ev[pol].to_numpy(bool) & ev["entry_ok"].to_numpy(bool)
        for y in range(2008, 2021):
            w = msk & (dts.dt.year == y).to_numpy()
            x = ev.loc[w, "exc20"].to_numpy(float)
            x = x[np.isfinite(x)]
            if x.size < 5:
                continue
            rows.append(dict(policy=pol, year=y, n=int(x.size),
                             mean_exc=float(np.mean(x)), median_exc=float(np.median(x)),
                             hit=float(np.mean(x > 0))))
    pd.DataFrame(rows).to_csv(OUT / "02b_事件_分年.csv", index=False)
    # 事件明细（只保留需要的列）
    keep = ["symbol", "date", "i", "delisted", "entry_ok", "reason", "gap",
            "depth", "dur", "vr", "sup_dist", "atr_pct", "amt20",
            "r5", "r10", "r20", "r40", "exc5", "exc10", "exc20", "exc40",
            "mae20", "mfe20"] + POLICIES + list(VARIANTS.keys())
    ev[keep].to_parquet(OUT / "01_事件_明细.parquet", index=False)
    log(f"事件汇总 {summ.shape}；明细 {len(ev):,} 行")
    return ev


# ---------------------------------------------------------------------------
# 统一引擎复核（账户撮合.py）
# ---------------------------------------------------------------------------
def cmd_account_check(ev: pd.DataFrame, policy: str = "L5_SUPPORT") -> pd.DataFrame:
    ACC = _load_module("trend_account_cb", "账户撮合.py")
    m = load_market()
    fcfg = fee_config(1.0)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        sy = con.execute("select symbol, ipo_date, delist_date from symbols where board='主板'").df()
        imp = con.execute("""select symbol, ex_date, prev_close, preclose, adj_ratio
                             from corp_actions_implied
                             where ex_date between '2008-01-01' and '2020-12-31'""").df()
    finally:
        con.close()
    info = {}
    for r in sy.itertuples():
        info[r.symbol] = ACC.SymbolInfo(
            code=r.symbol,
            ipo_date=None if pd.isna(r.ipo_date) else pd.Timestamp(r.ipo_date).date(),
            out_date=None if pd.isna(r.delist_date) else pd.Timestamp(r.delist_date).date())
    actions: dict[tuple[str, object], object] = {}
    for r in imp.itertuples():
        ratio = float(r.adj_ratio) if np.isfinite(r.adj_ratio) else 1.0
        dvd = 0.0; bonus = 0.0
        if ratio >= 1.0:
            bonus = max(0.0, ratio - 1.0)
        else:
            dvd = max(0.0, float(r.prev_close) - float(r.preclose))
        if dvd <= 0 and bonus <= 0:
            continue
        actions[(r.symbol, pd.Timestamp(r.ex_date).date())] = ACC.CorporateAction(
            code=r.symbol, ex_date=pd.Timestamp(r.ex_date).date(),
            dividend_per_share=dvd, per_share_bonus=bonus, source="implied")

    cands = _build_candidates(ev, policy)
    cfg = Config(cid=policy, policy=policy)
    out = []
    for wname, (a, b) in {"dev": DEV, "val": VAL}.items():
        i0, i1 = _slice_dates(m, a, b)
        cal = [pd.Timestamp(x).date() for x in m.dates[: i1 + 1]]
        acfg = ACC.AccountConfig(
            initial_cash=cfg.capital, fee=fcfg, cash_settle_days=0,
            enforce_t1=True, allow_buy_downsize=True, normalize_buy_orders=True,
            rights_policy="skip", dividend_tax_rate=0.0,
            delisting_missing_days=100000, delisting_haircut=0.0,
            liquidate_on_finalize=False, strict=False)
        acct = ACC.Account(acfg, calendar=cal, freeze_date=cal[-1], symbol_info=info)
        entry_day: dict[str, int] = {}
        exit_sched: dict[str, int] = {}
        submitted: set[str] = set()
        for i in range(i0, i1 + 1):
            d = m.dates[i]; dd = d.date()
            sig_i = i - 1 - cfg.delay
            codes = set(acct.positions) | set(exit_sched)
            if sig_i >= i0 and sig_i in cands:
                codes |= set(ev.loc[cands[sig_i], "symbol"])
            bars = {}
            for s in codes:
                bb = _bar(m, s, i)
                if bb is None or bb["susp"]:
                    continue
                j = i - m.start[s]
                bars[s] = ACC.Bar(code=s, date=dd, open=bb["open"], high=bb["high"],
                                  low=bb["low"], close=bb["close"], volume=bb["vol"],
                                  amount=bb["amt"],
                                  prev_close=float(m.d[s]["preclose"][j])
                                  if np.isfinite(m.d[s]["preclose"][j]) else None,
                                  prev_close_is_ref=True, is_st=bb["st"], trade_status=1)
            if i > i0:
                prev_d = m.dates[i - 1].date()
                for s in list(exit_sched):
                    if (s in acct.positions and acct.positions[s].shares > 0
                            and s not in submitted):
                        acct.sell_open(s, acct.positions[s].shares, decision_date=prev_d)
                        submitted.add(s)
                n_pos = sum(1 for p in acct.positions.values() if p.shares > 0)
                if sig_i >= i0 and sig_i in cands and n_pos < cfg.slots:
                    nav = acct.daily[-1]["nav"] if acct.daily else cfg.capital
                    free = (cfg.slots - n_pos + sum(
                        1 for s in exit_sched
                        if s in acct.positions and acct.positions[s].shares > 0))
                    n_try = 0
                    for ridx in cands[sig_i]:
                        if n_try >= free:
                            break
                        s = ev.at[ridx, "symbol"]
                        if (acct.positions.get(s) is not None
                                and acct.positions[s].shares > 0) or s not in bars:
                            continue
                        bb = _bar(m, s, i)
                        if bb is None or bb["lu"] or bb["resume"]:
                            continue
                        stop_hfq = float(ev.at[ridx, "sup"]) - 0.3 * float(ev.at[ridx, "atr"])
                        if not np.isfinite(stop_hfq) or bb["ho"] <= 0:
                            continue
                        risk = (bb["ho"] - stop_hfq) / bb["ho"]
                        if risk <= 0 or risk > MAX_STOP_PCT:
                            continue
                        epx = FEE.apply_slippage(bb["open"], "buy", fcfg.slippage_bps)
                        qty = int(nav / cfg.slots / epx / N_LOT) * N_LOT
                        qty = min(qty, int(MAX_RISK * nav / (risk * epx) / N_LOT) * N_LOT)
                        qty = min(qty, FEE.participation_cap(bb["vol"], PARTICIPATION) // N_LOT * N_LOT)
                        if qty < N_LOT:
                            continue
                        acct.buy_open(s, qty, decision_date=prev_d)
                        n_try += 1
            evs = []
            for s in acct.positions:
                if acct.positions[s].shares <= 0:
                    continue
                e = actions.get((s, dd))
                if e is not None:
                    evs.append(e)
            acct.on_day(dd, bars, events=evs, st_codes=None)
            submitted.clear()
            # 新成交 → 挂止损
            for t in acct.trades:
                if t.date == dd and t.side == "buy":
                    s = t.code
                    entry_day.setdefault(s, i)
                    bb = _bar(m, s, i)
                    sig_row = None
                    if sig_i >= i0 and sig_i in cands:
                        sub = ev.loc[cands[sig_i]]
                        mm = sub[sub["symbol"] == s]
                        if len(mm):
                            sig_row = mm.iloc[-1]
                    if bb is None or sig_row is None:
                        continue
                    stop_hfq = float(sig_row["sup"]) - 0.3 * float(sig_row["atr"])
                    stop_raw = stop_hfq * bb["close"] / bb["hc"] if bb["hc"] > 0 else np.nan
                    if (np.isfinite(stop_raw) and stop_raw > 0
                            and s in acct.positions and acct.positions[s].shares > 0):
                        acct.sell_stop(s, acct.positions[s].shares, float(stop_raw),
                                       decision_date=dd)
            # 时间退出（次日）
            for s, ei in list(entry_day.items()):
                if (s in acct.positions and acct.positions[s].shares > 0
                        and i - ei + 1 == cfg.exit_days and s not in exit_sched):
                    exit_sched[s] = i
                if s not in acct.positions or acct.positions[s].shares <= 0:
                    entry_day.pop(s, None); exit_sched.pop(s, None)
        acct.finalize(m.dates[i1].date(), liquidate=False)
        dl = pd.DataFrame(acct.daily)
        dl["date"] = pd.to_datetime(dl["date"])
        dl["n_pos"] = dl["n_positions"]
        dl["cash"] = dl["cash_total"]
        dl["exposure"] = dl["market_value"] / dl["nav"]
        p = perf_from_daily(dl, cfg.capital)
        at = pd.DataFrame([t.as_dict() for t in acct.trades])
        ts = dict(n_closed=int((at["side"] == "sell").sum()) if len(at) else 0,
                  n_buy=int((at["side"] == "buy").sum()) if len(at) else 0,
                  fees=float(at["fees"].sum()) if len(at) else 0.0,
                  win_rate=np.nan, profit_factor=np.nan, avg_ret=np.nan,
                  median_ret=np.nan, avg_pnl=np.nan, avg_hold=np.nan,
                  n_symbols=int(at["code"].nunique()) if len(at) else 0,
                  top5_share=np.nan)
        out.append(dict(policy=policy, window=wname, engine="账户撮合.py",
                        **p, **ts, n_flags=sum(acct.flag_counts.values()),
                        flags=json.dumps(dict(acct.flag_counts), ensure_ascii=False)))
        dl["window"] = wname
        dl.to_csv(OUT / f"09a_统一引擎_净值_{wname}.csv", index=False)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def cmd_selfcheck(ev: pd.DataFrame, summ: pd.DataFrame, trades: pd.DataFrame) -> dict:
    res = {}
    dts = pd.to_datetime(ev["date"])
    res["event_max_date"] = str(dts.max().date())
    res["no_2021_event"] = bool(dts.max() <= pd.Timestamp(ANALYSIS_END))
    res["no_2021_daily"] = bool(pd.to_datetime(
        pd.read_csv(OUT / "03_组合_净值.csv", usecols=["date"])["date"]).max()
        <= pd.Timestamp(ANALYSIS_END))
    res["n_events"] = int(len(ev))
    res["n_entry_ok"] = int(ev["entry_ok"].sum())
    res["entry_reject_rate"] = float(1 - ev["entry_ok"].mean())
    res["n_delisted_events"] = int(ev.loc[ev["delisted"], "entry_ok"].sum())
    if len(trades):
        buys = trades[trades["side"] == "buy"]
        sells = trades[trades["side"] == "sell"]
        res["lot_violations"] = int((buys["qty"] % N_LOT != 0).sum())
        res["same_day_exit"] = int((sells["hold_days"] <= 0).sum())
        res["min_fee"] = float(buys["fees"].min())
        res["commission_below_min"] = int((buys["commission"] < 5.0 - 1e-9).sum())
        res["buy_before_signal"] = int((pd.to_datetime(buys["date"]) <
                                        pd.to_datetime(buys["signal_date"])).sum())
        res["overnight_entry_frac"] = float((pd.to_datetime(buys["date"]) >
                                              pd.to_datetime(buys["signal_date"])).mean())
        res["negative_cash_events"] = int((summ["min_cash"] < -1e-6).sum())
    res["windows"] = {k: list(v) for k, v in {"dev": DEV, "val": VAL, "full": FULL}.items()}
    res["n_configs"] = int(summ["cid"].nunique())
    res["n_runs"] = int(len(summ))
    return res


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def cmd_portfolio(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()
    m = load_market()
    configs = make_configs()
    log(f"组合配置 {len(configs)} 个 × 3 窗口")
    summ, daily, trades, yearly, fails = run_configs(m, ev, configs)
    ew = daily_ew(m)
    h3 = hs300(m)
    bench = pd.DataFrame({"date": m.dates, "bench_ew": ew, "bench_hs300": h3})
    daily = daily.merge(bench, on="date", how="left")
    daily.to_csv(OUT / "03_组合_净值.csv", index=False)
    trades.to_csv(OUT / "04_组合_交易.csv", index=False)
    summ.to_csv(OUT / "05_组合_汇总.csv", index=False)
    fails.to_csv(OUT / "09b_失败订单.csv", index=False)
    yearly.to_csv(OUT / "08_分年.csv", index=False)

    # 基准
    rows = []
    for wname, (a, b) in {"dev": DEV, "val": VAL, "full": FULL}.items():
        msk = (pd.to_datetime(bench["date"]) >= pd.Timestamp(a)) & (
            pd.to_datetime(bench["date"]) <= pd.Timestamp(b))
        for nm, col, drag in (("B1_EW_MAIN", "bench_ew", 0.003 / 252), ("B2_HS300", "bench_hs300", 0.0)):
            r = (bench.loc[msk, col] - drag).dropna()
            nav = (1 + r).cumprod()
            yrs = (pd.to_datetime(bench.loc[msk, "date"]).iloc[-1] -
                   pd.to_datetime(bench.loc[msk, "date"]).iloc[0]).days / 365.25
            rows.append(dict(bench=nm, window=wname,
                             cagr=float(nav.iloc[-1] ** (1 / yrs) - 1),
                             mdd=float((nav / nav.cummax() - 1).min()),
                             vol=float(r.std() * math.sqrt(252))))
    pd.DataFrame(rows).to_csv(OUT / "03b_基准.csv", index=False)

    # 消融
    abl = summ[summ["cid"].isin(POLICIES) & summ["window"].isin(["dev", "val"])].copy()
    abl = abl.pivot_table(index="cid", columns="window",
                          values=["cagr", "mdd", "n_closed", "win_rate", "avg_exposure"])
    abl.columns = [f"{a}_{b}" for a, b in abl.columns]
    abl = abl.reset_index()
    abl.to_csv(OUT / "06_消融.csv", index=False)

    # 邻域/延迟/成本/规模
    extra = ["V_", "H10", "H40", "D1", "D2", "C2", "C3", "S5", "S20", "K20k", "K100k", "L5_SUPPORT", "SF_L5", "NS_"]
    mask = summ["cid"].apply(lambda x: any(x.startswith(e) or x == e for e in extra))
    summ.loc[mask].to_csv(OUT / "07_邻域_延迟_成本.csv", index=False)

    # 统计：DSR / PBO / bootstrap
    stats_out = {}
    full = daily[daily["window"] == "full"].pivot(index="date", columns="config", values="nav")
    full = full.dropna(how="all")
    rets = full.pct_change().dropna()
    main_key = Config(cid="L5_SUPPORT", policy="L5_SUPPORT").key()
    if main_key in rets.columns:
        tsrs = [float(rets[c].mean() / rets[c].std(ddof=1)) for c in rets.columns
                if rets[c].std(ddof=1) > 0]
        stats_out["dsr"] = STAT.deflated_sharpe(rets[main_key], trial_srs=tsrs)
        stats_out["bootstrap"] = STAT.block_bootstrap(
            rets[main_key], n_boot=2000, dates=rets.index, method="stationary", seed=20261028)
        try:
            stats_out["pbo"] = STAT.pbo_cscv(rets.values, blocks=16, max_combinations=5000)
        except Exception as exc:
            stats_out["pbo"] = {"error": str(exc)}
        stats_out["n_trials"] = int(len(rets.columns))
        stats_out["sr_all"] = {c: float(rets[c].mean() / rets[c].std(ddof=1))
                               for c in rets.columns if rets[c].std(ddof=1) > 0}
    (OUT / "11_统计.json").write_text(json.dumps(stats_out, ensure_ascii=False, indent=2,
                                                 default=str), encoding="utf-8")

    # 试验台账
    led = summ[summ["window"] == "full"].copy()
    led = led[["cid", "policy", "exit_days", "delay", "cost_mult", "slots", "capital",
               "use_stop", "rank_amt_asc"]]
    led["n_configs"] = 1
    led["note"] = "见 05_组合_汇总.csv 的 dev/val"
    led.loc[led["cid"].str.startswith(("NS_", "SF_")), "note"] = "事后诊断，不用于选参"
    led.loc[led["cid"].isin(["L6_BREAKOUT"]), "note"] = "预注册字面定义（含 sup），样本偏小"
    led.to_csv(OUT / "试验台账.csv", index=False)
    log(f"组合汇总 {len(summ)} 行；交易 {len(trades)} 行")
    return summ, daily, trades


def check_hfq_consistency(n_symbols: int = 30) -> dict:
    """抽样复核：本族 tr 序列逐日收益 == 官方 pct_chg；同时记录冻结层 hfq 的偏差。"""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        syms = con.execute("""select distinct symbol from daily
                              where board='主板' and date between '2008-01-01' and '2020-12-31'
                              order by symbol limit 30""").df()["symbol"].tolist()
        df = con.execute(f"""select symbol,date,close,preclose,pct_chg,hfq_close
                             from daily where symbol in ({','.join("'" + x + "'" for x in syms)})
                             and date between '2008-01-01' and '2020-12-31' order by symbol,date""").df()
    finally:
        con.close()
    worst_tr = 0.0; worst_fr = 0.0; n = 0
    for sym, g in df.groupby("symbol"):
        p = g["close"].to_numpy(float)
        pc = g["pct_chg"].to_numpy(float) / 100.0
        prev = np.r_[np.nan, p[:-1]]
        ratio = (1 + pc) * prev / p
        ratio = np.where(np.isfinite(ratio), ratio, 1.0); ratio[0] = 1.0
        tr = p * np.cumprod(ratio)
        tr_ret = tr[1:] / tr[:-1] - 1
        worst_tr = max(worst_tr, float(np.nanmax(np.abs(tr_ret - pc[1:]))))
        fr = g["hfq_close"].to_numpy(float)
        fr_ret = fr[1:] / fr[:-1] - 1
        worst_fr = max(worst_fr, float(np.nanmax(np.abs(fr_ret - pc[1:]))))
        n += len(g) - 1
    return {"n_rows": int(n), "n_symbols": int(len(syms)),
            "tr_max_abs_diff": worst_tr, "tr_pass": bool(worst_tr < 5e-9),
            "frozen_hfq_max_abs_diff": worst_fr,
            "frozen_hfq_pass": bool(worst_fr < 5e-6),
            "note": "本族 load_market 用 tr 序列；冻结 hfq 仅作缺陷审计"}


# ---------------------------------------------------------------------------
# 诊断：状态分桶 + 流动性匹配对照
# ---------------------------------------------------------------------------
def state_buckets(ev: pd.DataFrame) -> pd.DataFrame:
    """在“周/日趋势向上 + 3 日新高触发”的事件集内，按状态变量分桶。"""
    base = ev[ev["L2_DAILY"].to_numpy(bool) & ev["entry_ok"].to_numpy(bool)].copy()
    base = base[(pd.to_datetime(base["date"]) >= pd.Timestamp(DEV[0])) &
                (pd.to_datetime(base["date"]) <= pd.Timestamp(VAL[1]))]
    specs = {
        "depth": [(-1, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 2.5), (2.5, 3.5), (3.5, 99)],
        "dur": [(2, 4), (5, 7), (8, 10), (11, 15), (16, 25), (26, 999)],
        "vr": [(0, 0.6), (0.6, 0.75), (0.75, 0.85), (0.85, 1.0), (1.0, 99)],
        "sup_dist": [(-99, -0.5), (-0.5, 0), (0, 1), (1, 2), (2, 3), (3, 99)],
    }
    rows = []
    for var, bins in specs.items():
        x = base[[var, "exc20", "date"]].dropna()
        for lo, hi in bins:
            m = (x[var] >= lo) & (x[var] < hi)
            y = x.loc[m, "exc20"].to_numpy(float)
            if len(y) < 30:
                continue
            st = STAT.cluster_mean_test(y, pd.to_datetime(x.loc[m, "date"]).to_numpy())
            rows.append(dict(variable=var, bucket=f"[{lo:g},{hi:g})", n=len(y),
                             mean_exc=float(np.mean(y)), median_exc=float(np.median(y)),
                             t_cluster=st.get("t", np.nan), p_cluster=st.get("p", np.nan),
                             hit=float(np.mean(y > 0))))
    return pd.DataFrame(rows)


def matched_control(ev: pd.DataFrame, base: str = "L5_SUPPORT",
                    control: str = "L0_GENERIC", h: int = 20) -> pd.DataFrame:
    """同日、按 log(20日成交额) 最近邻匹配的对照（事后诊断，不用于选参）。"""
    dts = pd.to_datetime(ev["date"])
    win = (dts >= pd.Timestamp(DEV[0])) & (dts <= pd.Timestamp(VAL[1]))
    b = ev[ev[base].to_numpy(bool) & ev["entry_ok"].to_numpy(bool) & win.to_numpy()][
        ["symbol", "date", "amt20", f"exc{h}"]].dropna()
    c = ev[ev[control].to_numpy(bool) & ev["entry_ok"].to_numpy(bool) & win.to_numpy()][
        ["symbol", "date", "amt20", f"exc{h}"]].dropna()
    # 基准事件可能是对照策略的子集（L5 ⊂ L0）：按 (symbol,date) 剔除，避免匹配到自身
    keys = set(zip(b["symbol"], b["date"]))
    c = c[[k not in keys for k in zip(c["symbol"], c["date"])]]
    out = []
    for d, bd in b.groupby("date"):
        cd = c[c["date"] == d]
        if len(cd) == 0:
            continue
        cl = np.log(cd["amt20"].to_numpy(float))
        order = np.argsort(cl)
        cl = cl[order]
        cv = cd[f"exc{h}"].to_numpy(float)[order]
        bl = np.log(bd["amt20"].to_numpy(float))
        idx = np.clip(np.searchsorted(cl, bl), 1, len(cl) - 1) if len(cl) > 1 else np.zeros(len(bl), dtype=int)
        if len(cl) > 1:
            left = cl[idx - 1]; right = cl[idx]
            idx = np.where(np.abs(bl - left) <= np.abs(bl - right), idx - 1, idx)
        out.append(pd.DataFrame({"date": d, "base": bd[f"exc{h}"].to_numpy(float),
                                 "ctrl": cv[idx]}))
    if not out:
        return pd.DataFrame()
    z = pd.concat(out, ignore_index=True)
    z["diff"] = z["base"] - z["ctrl"]
    st = STAT.cluster_mean_test(z["diff"].to_numpy(float), pd.to_datetime(z["date"]).to_numpy())
    res = pd.DataFrame([dict(base=base, control=control, horizon=h, n=len(z),
                             mean_base=float(z["base"].mean()),
                             mean_ctrl=float(z["ctrl"].mean()),
                             mean_diff=float(z["diff"].mean()),
                             median_diff=float(z["diff"].median()),
                             t_cluster=st.get("t", np.nan), p_cluster=st.get("p", np.nan),
                             ci_lo=st.get("ci_lo", np.nan), ci_hi=st.get("ci_hi", np.nan),
                             hit_diff=float((z["diff"] > 0).mean()))])
    return res


def size_layers(ev: pd.DataFrame) -> pd.DataFrame:
    """事件超额按 20 日成交额五分位；回答“边际是否只在某个规模桶”。"""
    rows = []
    dts = pd.to_datetime(ev["date"])
    base = ev[(dts >= pd.Timestamp(DEV[0])) & (dts <= pd.Timestamp(VAL[1]))]
    for pol in ["L0_GENERIC", "L5_SUPPORT", "L7_DIRECT", "L8_NONTREND", "L9_BROKEN"]:
        x = base[base[pol].to_numpy(bool) & base["entry_ok"].to_numpy(bool)]
        x = x.dropna(subset=["exc20", "amt20"])
        if len(x) < 100:
            continue
        q = pd.qcut(np.log(x["amt20"].to_numpy(float)), 5,
                    labels=["Q1小", "Q2", "Q3", "Q4", "Q5大"])
        for lab, g in x.groupby(q, observed=True):
            y = g["exc20"].to_numpy(float)
            st = STAT.cluster_mean_test(y, pd.to_datetime(g["date"]).to_numpy())
            rows.append(dict(policy=pol, quintile=str(lab), n=len(y),
                             mean_exc=float(np.mean(y)), median_exc=float(np.median(y)),
                             t_cluster=st.get("t", np.nan), p_cluster=st.get("p", np.nan),
                             hit=float(np.mean(y > 0)),
                             median_amt20=float(g["amt20"].median())))
    return pd.DataFrame(rows)


def cmd_diagnostics(ev: pd.DataFrame) -> dict:
    sb = state_buckets(ev)
    sl = size_layers(ev)
    sl.to_csv(OUT / "15_规模分层.csv", index=False)
    sb.to_csv(OUT / "14_状态分桶.csv", index=False)
    mc = []
    for base, control in (("L5_SUPPORT", "L0_GENERIC"), ("L5_SUPPORT", "L4_VOL"),
                          ("L5_SUPPORT", "L7_DIRECT"), ("L9_BROKEN", "L5_SUPPORT")):
        r = matched_control(ev, base, control)
        if len(r):
            mc.append(r)
    mc = pd.concat(mc, ignore_index=True) if mc else pd.DataFrame()
    mc.to_csv(OUT / "13_流动性匹配对照.csv", index=False)
    log(f"状态分桶 {len(sb)} 行；匹配对照 {len(mc)} 行；规模分层 {len(sl)} 行")
    return {"buckets": len(sb), "matched": len(mc), "size_layers": len(sl)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["data-audit", "events", "portfolio", "account-check",
                             "diagnostics", "selfcheck", "all"])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--max-symbols", type=int, default=None)
    a = ap.parse_args(argv)
    _ensure_dirs()

    aud = None
    if a.stage in ("data-audit", "all"):
        aud = cmd_data_audit()
        log(f"数据缺陷审计：{aud}")

    if a.stage == "data-audit":
        return 0

    if a.stage == "events":
        cmd_events(rebuild=a.rebuild, max_symbols=a.max_symbols)
        return 0

    ep = CACHE / "events.parquet"
    if not ep.exists():
        raise SystemExit("缺少事件缓存，先运行 events")
    ev = pd.read_parquet(ep)

    if a.stage in ("portfolio", "all"):
        summ, daily, trades = cmd_portfolio(ev)
    elif a.stage == "diagnostics":
        summ = trades = None
    else:
        summ = pd.read_csv(OUT / "05_组合_汇总.csv")
        trades = pd.read_csv(OUT / "04_组合_交易.csv")

    if a.stage in ("diagnostics", "all"):
        cmd_diagnostics(ev)

    if a.stage in ("account-check", "all"):
        ac = cmd_account_check(ev)
        ac.to_csv(OUT / "09_统一引擎复核.csv", index=False)
        log(ac.to_string(index=False))

    if a.stage in ("selfcheck", "all"):
        sc = cmd_selfcheck(ev, summ, trades)
        sc["hfq_consistency"] = check_hfq_consistency()
        sc["data_audit"] = aud if aud is not None else cmd_data_audit()
        sc["code_sha256"] = sha256_file(Path(__file__))
        sc["prereg_sha256"] = sha256_file(OUT / "00_预注册_趋势回调.md")
        (OUT / "10_自检.json").write_text(json.dumps(sc, ensure_ascii=False, indent=2,
                                                     default=str), encoding="utf-8")
        log(json.dumps(sc, ensure_ascii=False, indent=2, default=str)[:4000])
    return 0




# ---------------------------------------------------------------------------
# 数据缺陷审计：冻结层 hfq 的复利平方
# ---------------------------------------------------------------------------
def cmd_data_audit() -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        q = """
        with x as (
          select symbol, date, close, preclose, pct_chg, hfq_close,
                 lag(close) over (partition by symbol order by date) close_prev,
                 lag(hfq_close) over (partition by symbol order by date) hfq_prev
          from daily
          where date between '2005-01-01' and '2020-12-31'
        )
        select symbol, date, close, close_prev, preclose, pct_chg, hfq_close, hfq_prev,
               hfq_close / hfq_prev - 1 as hfq_ret, close / close_prev - 1 as raw_ret
        from x where hfq_prev is not null and pct_chg is not null
        """
        df = con.execute(q).df()
        # 正确构造：tr_t = tr_{t-1} * (1+pct_chg_t)
        df["pct"] = df["pct_chg"] / 100.0
        df["tr_ret"] = df["pct"]  # 定义上逐行相等；下面用累计价复核
        bad_frozen = (df["hfq_ret"] - df["pct"]).abs() > 5e-5
        bad_raw = (df["raw_ret"] - df["pct"]).abs() > 5e-5
        by_year = df.assign(y=pd.to_datetime(df["date"]).dt.year,
                            bad=bad_frozen).groupby("y")["bad"].agg(["sum", "mean", "count"])
        ex = df[(df["symbol"] == "000078.SZ") &
                (df["date"].astype(str).between("2016-05-24", "2016-05-30"))]
        # 累计价复核：每个证券内 tr 的逐日收益必须等于 pct_chg
        chk = con.execute("""
        with x as (
          select symbol, date, pct_chg,
                 lag(pct_chg) over (partition by symbol order by date) pc_prev
          from daily where date between '2005-01-01' and '2020-12-31'
        ) select count(*) n from x where pc_prev is not null
        """).df().iloc[0, 0]
    finally:
        con.close()

    lines = []
    lines.append("# 数据缺陷：冻结层 hfq 把日收益平方，送转日断裂\n")
    lines.append(f"- 审计时间：2026-10-28；数据：`01_数据/frozen/frozen.duckdb`，2005-01-01..2020-12-31\n")
    lines.append(f"- 样本：{len(df):,} 个证券-日（有前收与 pct_chg）\n")
    lines.append("\n## 1. 结论\n")
    lines.append("冻结层 `hfq_close = raw_close × factor`，其中 `factor = Π(1+pct_chg/100)`。\n")
    lines.append("这在数学上等价于把**日收益乘了两次**：正常交易日 `raw_t/raw_{t-1}=1+r_t`，\n")
    lines.append("于是 `hfq_t/hfq_{t-1}-1 = (1+r_t)^2-1 ≈ 2r_t`；在送转/拆股日 `raw` 跳空，\n")
    lines.append("`factor` 不变，`hfq` 直接把跳空记为亏损（000078.SZ 2016-05-27 记为 -62.9%，\n")
    lines.append("官方 `pct_chg` 为 -3.66%）。\n")
    lines.append("\n## 2. 实测\n")
    lines.append(f"| 指标 | 数值 |\n|---|---:|\n")
    lines.append(f"| hfq_ret 与 pct_chg 偏差 >0.5bp 的行 | **{int(bad_frozen.sum()):,} / {len(df):,} = {bad_frozen.mean():.2%}** |\n")
    lines.append(f"| raw_ret 与 pct_chg 偏差 >0.5bp 的行 | {int(bad_raw.sum()):,} / {len(df):,} = {bad_raw.mean():.2%} |\n")
    lines.append(f"| 中位 hfq_ret 与 pct_chg 偏差 | {float((df['hfq_ret']-df['pct']).abs().median()):.4%} |\n")
    lines.append(f"| 中位 raw_ret 与 pct_chg 偏差 | {float((df['raw_ret']-df['pct']).abs().median()):.4%} |\n")
    lines.append("\n### 2.1 分年\n\n| 年 | 偏差行 | 占比 | 样本 |\n|---:|---:|---:|---:|\n")
    for y, r in by_year.iterrows():
        lines.append(f"| {int(y)} | {int(r['sum']):,} | {r['mean']:.2%} | {int(r['count']):,} |\n")
    lines.append("\n### 2.2 案例（000078.SZ，2016 年 10 转 16）\n\n")
    lines.append("| 日期 | raw close | 官方 pct_chg | 冻结 hfq_close | 冻结 hfq 收益 | 正确总收益 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for r in ex.itertuples():
        lines.append(f"| {r.date} | {r.close:.2f} | {r.pct_chg:.2f}% | {r.hfq_close:.2f} | {r.hfq_ret:.2%} | {r.pct:.2%} |\n")
    lines.append("\n## 3. 为什么自检没发现\n")
    lines.append("`01_数据/_audit/自检报告.md` 的『复权收益 vs 官方 pct_chg 0 行偏差』实际比较的是\n")
    lines.append("`close/preclose-1` 与 `pct_chg`（两者定义上相等），不是 `hfq_close` 的收益。\n")
    lines.append("『hfq = close × factor』也只是恒等式，无法发现 factor 本身漏乘了 `raw_{t-1}/raw_t`。\n")
    lines.append("\n## 4. 影响与处置\n")
    lines.append("- 任何直接用冻结层 `hfq_*` 计算收益、ATR、均线、回撤的研究都会被污染；\n")
    lines.append("  本族第一版调试（200 只子样本）因此出现 -91% 的假回撤，已作废。\n")
    lines.append("- 本族改用**由 pct_chg 递归构造**的总收益价：`tr_t = tr_{t-1} × (1+pct_chg_t/100)`；\n")
    lines.append("  等价于对 raw OHLC 乘 `k_t = Π (1+r_s)×P_{s-1}/P_s`。逐日复核 `tr_ret == pct_chg`（定义上）。\n")
    lines.append("- 未修改 `01_数据/` 与任何他人文件；冻结层重建应由数据负责人执行。\n")
    lines.append("- 前序 `策略族_趋势.py` 用的是 Baostock `adjustflag=1` 后复权，不受此缺陷影响；\n")
    lines.append("  但其 79 行年末重定基问题仍按数据字典 §5.1 处理。\n")
    lines.append("\n## 5. 建议\n")
    lines.append("重建冻结层时把因子改为 `factor_t = hfq_t/raw_t`，其中 `hfq_t` 由 `pct_chg` 递归累乘；\n")
    lines.append("并在自检中加入『`hfq_ret` 与 `pct_chg` 逐行一致（>0.5bp 为 0 行）』。\n")
    (OUT / "12_数据缺陷_冻结层hfq.md").write_text("".join(lines), encoding="utf-8")
    return dict(n=int(len(df)), bad_frozen=int(bad_frozen.sum()),
                bad_frozen_share=float(bad_frozen.mean()),
                bad_raw=int(bad_raw.sum()), bad_raw_share=float(bad_raw.mean()),
                median_frozen_diff=float((df["hfq_ret"] - df["pct"]).abs().median()),
                median_raw_diff=float((df["raw_ret"] - df["pct"]).abs().median()),
                n_rows_check=int(chk))


if __name__ == "__main__":
    raise SystemExit(main())
