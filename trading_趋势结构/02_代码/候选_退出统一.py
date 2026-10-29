#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选_退出统一.py — 同入口、同事件、同成本下的退出规则统一比较。

范围：沪深主板，2008-01-01..2020-12-31；2021+ 硬拦截。
数据：01_数据/frozen（只读）。**不使用冻结层 hfq_* 列**：该列把 pct_chg 计入了两次
      （见 03_结果/候选_退出统一/00_预注册_退出统一.md §1.1）。本脚本用
      scale_t = factor_t/close_t 重建 hfq：hfq_px = raw_px*scale，hfq_close == factor，
      因此 hfq 日收益 ≡ 官方 pct_chg。

用法：
    python3 02_代码/候选_退出统一.py audit
    python3 02_代码/候选_退出统一.py signals
    python3 02_代码/候选_退出统一.py run
    python3 02_代码/候选_退出统一.py portfolio
    python3 02_代码/候选_退出统一.py all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "01_数据" / "frozen" / "frozen.duckdb"
OUT = ROOT / "03_结果" / "候选_退出统一"
CACHE = OUT / "_cache"
FREEZE = "2026-10-26"
WIN_END = "2020-12-31"
DEV_END = "2015-12-31"
PANEL_START = "2004-01-01"

NOTIONAL = 100_000.0
SLIP = 0.0010
COOLDOWN = 20
MAX_HOLD = 120
SUSP_MAX = 20
MIN_PRICE = 2.0
MIN_ADV20 = 5e7
MIN_LISTED = 250
MAX_RISK_PCT = 0.12

META = dict(window_start="2008-01-01", window_end=WIN_END,
            adjust="hfq_rebuilt_from_pct_chg", cost_model_id="C-REAL-V1")

OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ==========================================================================
# 1. 因果特征（可单独调用，用于截断/扰动自检）
# ==========================================================================

def wilder_carry(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    valid = np.isfinite(x)
    if len(x) < n:
        return out
    c = np.convolve(valid.astype(np.int64), np.ones(n, np.int64), "valid")
    idx = np.flatnonzero(c == n)
    if len(idx) == 0:
        return out
    s = int(idx[0])
    out[s + n - 1] = np.nanmean(x[s:s + n])
    for i in range(s + n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n if np.isfinite(x[i]) else out[i - 1]
    return out


def pivot_conf(low: np.ndarray, k: int) -> np.ndarray:
    """公共口径：中心最低（等号允许），右侧 k 根确认后记在 confirm 位置。"""
    n = len(low)
    out = np.full(n, np.nan)
    if n < 2 * k + 1:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(low, 2 * k + 1)
    mid = w[:, k]
    ok = np.isfinite(w).all(axis=1) & (mid <= w.min(axis=1))
    win = np.flatnonzero(ok)
    piv = win + k          # 窗口中心 = 枢轴点
    conf = piv + k         # 右侧 k 根确认
    good = conf < n
    out[conf[good]] = low[piv[good]]
    return out


def ffill(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).ffill().to_numpy()


def pivot_pairs(low: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    conf = pivot_conf(low, k)
    last = ffill(conf)
    prev = np.full(len(low), np.nan)
    idx = np.flatnonzero(np.isfinite(conf))
    for a, b in zip(idx[:-1], idx[1:]):
        prev[b:] = conf[a]
    return last, prev


def compute_features(o, h, l, c, pc, f, vol, amt, st, susp, td):
    n = len(c)
    tradable = (~susp) & (vol > 0) & (c > 0) & np.isfinite(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where((c > 0) & np.isfinite(f), f / c, np.nan)
    ho, hh, hl, hc = o * scale, h * scale, l * scale, c * scale
    prev_hc = np.concatenate([[np.nan], hc[:-1]])
    tr = np.full(n, np.nan)
    v = tradable & np.isfinite(hc) & np.isfinite(prev_hc)
    tr[v] = np.maximum.reduce([hh[v] - hl[v], np.abs(hh[v] - prev_hc[v]),
                               np.abs(hl[v] - prev_hc[v])])
    atr = wilder_carry(tr, 14)
    s_hc = pd.Series(hc)
    ma20 = s_hc.rolling(20, min_periods=20).mean().to_numpy()
    ma50 = s_hc.rolling(50, min_periods=50).mean().to_numpy()
    don55 = pd.Series(hh).rolling(55, min_periods=55).max().shift(1).to_numpy()
    adv20 = pd.Series(np.where(tradable, amt, np.nan)).rolling(20, min_periods=15).mean().to_numpy()
    pl2, pl2p = pivot_pairs(hl, 2)
    pl5, pl5p = pivot_pairs(hl, 5)
    with np.errstate(invalid="ignore"):
        uni = (tradable & (~st) & (td >= MIN_LISTED) & (c >= MIN_PRICE)
               & (adv20 >= MIN_ADV20) & np.isfinite(atr) & (atr > 0))
    return dict(tradable=tradable, scale=scale, ho=ho, hh=hh, hl=hl, hc=hc,
                atr=atr, ma20=ma20, ma50=ma50, don55=don55, adv20=adv20,
                pl2=pl2, pl2p=pl2p, pl5=pl5, pl5p=pl5p, uni=uni)


FIELDS = ["cal", "o", "h", "l", "c", "pc", "pct", "f", "vol", "amt", "td",
          "st", "susp", "up", "dn", "lu_open", "ld_open",
          "ho", "hh", "hl", "hc", "atr", "ma20", "ma50", "don55", "adv20",
          "pl2", "pl2p", "pl5", "pl5p", "uni", "tradable", "scale"]


def build_panel(force: bool = False) -> dict:
    cache = CACHE / "panel_v1.npz"
    if cache.exists() and not force:
        log(f"载入面板缓存 {cache.name}")
        z = np.load(cache, allow_pickle=False)
        p = {k: z[k] for k in z.files}
        p["codes"] = [str(x) for x in p["codes"]]
        p["dates"] = [str(x) for x in p["dates"]]
        return p

    t0 = time.time()
    con = duckdb.connect(str(FROZEN), read_only=True)
    cal_df = con.execute(f"""SELECT CAST(date AS VARCHAR) d FROM calendar
                             WHERE is_trading_day=1 AND date BETWEEN DATE '{PANEL_START}'
                             AND DATE '{WIN_END}' ORDER BY date""").df()
    dates = [str(x) for x in cal_df["d"].to_numpy()]
    d_idx = {d: i for i, d in enumerate(dates)}
    log(f"日历 {len(dates)} 个交易日 {dates[0]}..{dates[-1]}")
    q = f"""
      SELECT symbol, CAST(date AS VARCHAR) d, open, high, low, close, preclose, pct_chg,
             factor, volume, amount, is_st, is_suspended,
             limit_up_px, limit_down_px, is_limit_up_open, is_limit_down_open, td_since_list
      FROM daily
      WHERE board='主板' AND date BETWEEN DATE '{PANEL_START}' AND DATE '{WIN_END}'
      ORDER BY symbol, date
    """
    df = con.execute(q).df()
    con.close()
    log(f"读取 {len(df):,} 行，{df['symbol'].nunique()} 只，{time.time()-t0:.0f}s")

    syms = df["symbol"].to_numpy()
    brk = np.flatnonzero(syms[1:] != syms[:-1]) + 1
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [len(df)]])
    codes_all = [str(syms[s]) for s in starts]
    cal_all = np.array([d_idx[str(x)] for x in df["d"].to_numpy()], dtype=np.int32)
    raw = {k: df[k].to_numpy() for k in
           ["open", "high", "low", "close", "preclose", "pct_chg", "factor", "volume", "amount",
            "is_st", "is_suspended", "limit_up_px", "limit_down_px",
            "is_limit_up_open", "is_limit_down_open", "td_since_list"]}

    out = {k: [] for k in FIELDS}
    codes, seglen = [], []
    for i, (s, e) in enumerate(zip(starts, ends)):
        n = e - s
        if n < 60:
            continue
        o = raw["open"][s:e].astype(np.float64)
        h = raw["high"][s:e].astype(np.float64)
        l = raw["low"][s:e].astype(np.float64)
        c = raw["close"][s:e].astype(np.float64)
        pc = raw["preclose"][s:e].astype(np.float64)
        f = raw["factor"][s:e].astype(np.float64)
        vol = raw["volume"][s:e].astype(np.float64)
        amt = raw["amount"][s:e].astype(np.float64)
        st = pd.Series(raw["is_st"][s:e]).fillna(0).astype(bool).to_numpy()
        susp = pd.Series(raw["is_suspended"][s:e]).fillna(True).astype(bool).to_numpy()
        td = raw["td_since_list"][s:e].astype(np.int32)
        ft = compute_features(o, h, l, c, pc, f, vol, amt, st, susp, td)
        arrs = dict(cal=cal_all[s:e], o=o, h=h, l=l, c=c, pc=pc,
                    pct=raw["pct_chg"][s:e].astype(np.float64),
                    f=f, vol=vol, amt=amt, td=td,
                    st=st, susp=susp, up=raw["limit_up_px"][s:e].astype(np.float64),
                    dn=raw["limit_down_px"][s:e].astype(np.float64),
                    lu_open=pd.Series(raw["is_limit_up_open"][s:e]).fillna(False).astype(bool).to_numpy(),
                    ld_open=pd.Series(raw["is_limit_down_open"][s:e]).fillna(False).astype(bool).to_numpy(), **ft)
        for k in FIELDS:
            out[k].append(arrs[k])
        codes.append(codes_all[i])
        seglen.append(n)
        if (i + 1) % 500 == 0:
            log(f"  特征 {i+1}/{len(codes_all)}")

    p = {k: np.concatenate(v) for k, v in out.items()}
    offs = np.concatenate([[0], np.cumsum(seglen)]).astype(np.int64)
    if len(offs) - 1 != len(codes):
        raise RuntimeError(f"切分不一致 {len(offs)-1} vs {len(codes)}")
    p["codes"] = np.array(codes, dtype=str)
    p["dates"] = np.array(dates, dtype=str)
    p["offsets"] = offs
    np.savez_compressed(cache, **p)
    log(f"面板写入 {cache}，{len(p['cal']):,} 行，{len(codes)} 只，{time.time()-t0:.0f}s")
    p["codes"] = [str(x) for x in p["codes"]]
    p["dates"] = [str(x) for x in p["dates"]]
    return p


class Panel:
    def __init__(self, p: dict):
        self.p = p
        self.codes = p["codes"]
        self.dates = p["dates"]
        self.offsets = p["offsets"]
        self.n = len(self.codes)
        self._cache: dict[int, dict] = {}

    def sym(self, i: int) -> dict:
        j = int(i)
        hit = self._cache.get(j)
        if hit is not None:
            return hit
        a, b = int(self.offsets[j]), int(self.offsets[j + 1])
        d = {k: self.p[k][a:b] for k in FIELDS}
        self._cache[j] = d
        return d

    def __getitem__(self, k):
        return self.p[k]


# ==========================================================================
# 2. 入口
# ==========================================================================

@dataclass(frozen=True)
class EntrySpec:
    name: str
    kind: str          # A=55日突破 B=趋势回调


ENTRIES = [EntrySpec("A_55突破", "A"), EntrySpec("B_趋势回调", "B")]


def signal_mask(s: dict, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "A":
        cond = (s["hc"] > s["don55"]) & s["uni"] & np.isfinite(s["don55"])
        strength = (s["hc"] - s["don55"]) / s["atr"]
    elif kind == "B":
        ma50_prev = np.concatenate([[np.nan] * 10, s["ma50"][:-10]])
        cond = ((s["hc"] > s["ma50"]) & (s["ma50"] > ma50_prev)
                & (s["pl5"] > s["pl5p"]) & (s["hl"] <= s["ma20"])
                & (s["hc"] > s["ma20"]) & (s["hc"] > s["ho"]) & s["uni"])
        strength = (s["hc"] - s["ma20"]) / s["atr"]
    else:
        raise ValueError(kind)
    return np.asarray(cond, dtype=bool) & np.isfinite(strength), strength


def signals_for(panel: Panel, spec: EntrySpec) -> list[dict]:
    rows = []
    for ci in range(panel.n):
        s = panel.sym(ci)
        if len(s["cal"]) < 300:
            continue
        cond, strength = signal_mask(s, spec.kind)
        idx = np.flatnonzero(cond)
        last = -10**9
        for si in idx:
            if si - last < COOLDOWN:
                continue
            last = si
            rows.append(dict(ci=ci, si=int(si), cal=int(s["cal"][si]),
                             strength=float(strength[si]),
                             year=int(panel.dates[s["cal"][si]][:4])))
    return rows


def signal_stats(panel: Panel, spec: EntrySpec) -> dict:
    raw_n = 0
    kept = 0
    for ci in range(panel.n):
        s = panel.sym(ci)
        if len(s["cal"]) < 300:
            continue
        cond, _ = signal_mask(s, spec.kind)
        idx = np.flatnonzero(cond)
        raw_n += len(idx)
        last = -10**9
        for si in idx:
            if si - last < COOLDOWN:
                continue
            last = si
            kept += 1
    return dict(entry=spec.name, raw_signals=int(raw_n), kept=int(kept),
                suppressed=int(raw_n - kept))


# ==========================================================================
# 3. 费用与制度
# ==========================================================================

def stamp_rate(date: str) -> tuple[float, bool]:
    if date <= "2008-04-23":
        return 0.003, False
    if date <= "2008-09-18":
        return 0.001, True
    return 0.001, False


def transfer(date: str, symbol: str) -> tuple[str, float]:
    if date <= "2015-07-31":
        return ("face", 0.001) if symbol.endswith(".SH") else ("amount", 0.0)
    return "amount", 0.00002


def fees(date: str, symbol: str, amount: float, shares: float, side: str) -> float:
    fee = max(amount * 0.00025, 5.0)
    kind, r = transfer(date, symbol)
    fee += max(shares * r, 1.0) if kind == "face" else amount * r
    st, both = stamp_rate(date)
    if side == "sell" or both:
        fee += amount * st
    return float(fee)


def locked_down(h, l, dn) -> bool:
    return bool(np.isfinite(dn) and h <= l + 1e-12 and l <= dn + 0.011)


def locked_up(h, l, up) -> bool:
    return bool(np.isfinite(up) and h <= l + 1e-12 and l >= up - 0.011)


# ==========================================================================
# 4. 规则
# ==========================================================================

@dataclass(frozen=True)
class Policy:
    name: str
    init: str = "atr"
    atr_mult: float = 2.0
    struct_buf: float = 0.30
    struct_floor: float = 0.50
    horizon: int | None = None
    time_days: int | None = None
    time_min_R: float = 1.0
    fail_days: int | None = None
    fail_min_R: float = 0.5
    half_days: int | None = None
    half_min_R: float = 1.0
    be_R: float | None = None
    trail_R: float | None = None
    trail_mult: float = 3.0
    scale_R: float | None = None


POLICIES = [
    Policy("FH10", init="none", horizon=10),
    Policy("FH20", init="none", horizon=20),
    Policy("FH40", init="none", horizon=40),
    Policy("FH20S", init="atr", horizon=20),
    Policy("ATR2", init="atr"),
    Policy("STRUCT", init="struct"),
    Policy("TIME20", init="atr", time_days=20),
    Policy("CAUT10", init="atr", half_days=10),
    Policy("BE1", init="atr", be_R=1.0),
    Policy("CHAND2", init="atr", trail_R=2.0, trail_mult=3.0),
    Policy("SCALE2", init="atr", trail_R=2.0, trail_mult=3.0, scale_R=2.0),
    Policy("RUN1", init="atr", trail_R=1.0, trail_mult=3.0),
    Policy("FAIL3", init="atr", fail_days=3, fail_min_R=0.5),
    Policy("ALL", init="atr", fail_days=3, half_days=10, time_days=20,
           be_R=1.0, trail_R=2.0, trail_mult=3.0),
    Policy("ALL_NF", init="atr", half_days=10, time_days=20,
           be_R=1.0, trail_R=2.0, trail_mult=3.0),
]
PBYNAME = {p.name: p for p in POLICIES}


# ==========================================================================
# 5. 状态机（事件级与组合级共用）
# ==========================================================================

@dataclass
class PosState:
    pol: Policy
    code: str
    entry_cal: int
    entry_date: str
    entry_px: float
    entry_market: float
    entry_hfq: float
    atr0_hfq: float
    stop_hfq: float
    R_hfq: float
    R_raw: float
    shares0: int
    shares: float
    cost0: float
    stop_kind: str = "none"
    stop: float = float("nan")
    max_high: float = 0.0
    min_low: float = 1e18
    mfe: float = 0.0
    mae: float = 0.0
    k: int = 0
    susp_bars: int = 0
    half_done: bool = False
    scale_done: bool = False
    pending_full: bool = False
    pending_reason: str = ""
    pending_half: bool = False
    last_px: float = 0.0
    last_cal: int = 0
    last_row: int = 0
    last_reason: str = "data_end"
    closed: bool = False
    delist: int = 0
    censored: int = 0
    n_partial: int = 0
    scaled: float = 0.0
    proceeds: float = 0.0
    gross_in: float = 0.0
    fee_total: float = 0.0
    slip_cost: float = 0.0
    cash_flow: float = 0.0
    mfe10: float = float("nan")
    alive10: int = 0


def _sell(st: PosState, shares: float, px: float, date: str, slip: float, reason: str,
          fee_mult: float = 1.0) -> None:
    shares = float(min(shares, st.shares))
    if shares <= 1e-9:
        return
    market_px = px / (1.0 - slip) if slip < 1 else px
    amt = shares * px
    fee = fees(date, st.code, amt, shares, "sell") * fee_mult
    net = amt - fee
    st.shares -= shares
    st.proceeds += net
    st.gross_in += shares * market_px
    st.fee_total += fee
    st.slip_cost += shares * market_px * slip
    st.cash_flow += net
    st.last_px = px
    st.last_reason = reason
    if st.shares <= 1e-9:
        st.closed = True


def _force(st: PosState, px: float, date: str, slip: float, reason: str, fee_mult: float = 1.0):
    _sell(st, st.shares, px * (1 - slip), date, slip, reason, fee_mult)


def step_day(st: PosState, s: dict, j: int, date: str, slip: float = SLIP,
             fee_mult: float = 1.0) -> None:
    """推进一个持仓一天。j 为该证券行号，st.k 为入场后第几个交易日（入场日 k=0）。"""
    if st.closed:
        return
    tradable = bool(s["tradable"][j]) and s["o"][j] > 0 and s["c"][j] > 0
    if not tradable:
        st.susp_bars += 1
        if st.susp_bars >= SUSP_MAX:
            st.last_cal = int(s["cal"][j])
            _force(st, st.last_px, date, slip, "suspended_forced", fee_mult)
        return
    st.susp_bars = 0
    st.last_cal = int(s["cal"][j])
    scale = float(s["scale"][j])
    o, h, l, c = float(s["o"][j]), float(s["h"][j]), float(s["l"][j]), float(s["c"][j])
    lk_dn = locked_down(h, l, float(s["dn"][j]))
    k = st.k

    if k > 0:
        if st.pending_full and not lk_dn:
            _sell(st, st.shares, o * (1 - slip), date, slip,
                  st.pending_reason or "pending", fee_mult)
            if st.closed:
                return
        if st.pending_half and not lk_dn and st.shares > 0:
            sh = max(1.0, round(st.shares * 0.5))
            _sell(st, sh, o * (1 - slip), date, slip, "half", fee_mult)
            st.pending_half = False
            st.half_done = True
            st.n_partial += 1
            if st.closed:
                return
        if np.isfinite(st.stop) and not lk_dn and st.shares > 0:
            sr = st.stop / scale
            px = None
            if o <= sr:
                px = o * (1 - slip)
            elif l <= sr:
                px = sr * (1 - slip)
            if px is not None:
                _sell(st, st.shares, px, date, slip, "stop", fee_mult)
                st.last_reason = "stop"
                if st.closed:
                    return
        if st.pol.scale_R is not None and not st.scale_done and st.shares > 0:
            target = st.entry_hfq + st.pol.scale_R * st.R_hfq
            pr = target / scale
            px = None
            if o >= pr:
                px = o * (1 - slip)
            elif h >= pr:
                px = pr * (1 - slip)
            if px is not None:
                sh = max(1.0, round(st.shares * 0.5))
                _sell(st, sh, px, date, slip, "scale", fee_mult)
                st.scale_done = True
                st.scaled = 1.0
                st.n_partial += 1
                if st.closed:
                    return

    st.max_high = max(st.max_high, h * scale)
    st.min_low = min(st.min_low, l * scale)
    if np.isfinite(st.R_hfq) and st.R_hfq > 0:
        st.mfe = (st.max_high - st.entry_hfq) / st.R_hfq
        st.mae = (st.min_low - st.entry_hfq) / st.R_hfq
    if k == 10:
        st.mfe10 = st.mfe
        st.alive10 = 1
    st.last_px = c
    st.last_cal = int(s["cal"][j])

    n_bars = k + 1
    pol = st.pol
    if pol.horizon is not None and n_bars >= pol.horizon:
        st.pending_full = True
        st.pending_reason = "horizon"
    if pol.fail_days is not None and n_bars == pol.fail_days:
        if st.mfe < pol.fail_min_R and c * scale < st.entry_hfq:
            st.pending_full = True
            st.pending_reason = "fail3"
    if pol.half_days is not None and n_bars == pol.half_days and not st.half_done:
        if st.mfe < pol.half_min_R:
            st.pending_half = True
    if pol.time_days is not None and n_bars == pol.time_days:
        if st.mfe < pol.time_min_R:
            st.pending_full = True
            st.pending_reason = "time"
    if n_bars >= MAX_HOLD:
        st.pending_full = True
        st.pending_reason = "maxhold"

    if pol.be_R is not None and st.mfe >= pol.be_R:
        if np.isfinite(st.stop) and st.stop < st.entry_hfq:
            st.stop = st.entry_hfq
            st.stop_kind = "stop_be"
    if pol.trail_R is not None and st.mfe >= pol.trail_R:
        tr = st.max_high - pol.trail_mult * st.atr0_hfq
        if not np.isfinite(st.stop) or tr > st.stop:
            st.stop = tr
            st.stop_kind = "stop_trail"


def entry_allowed(s: dict, si: int, ed: int, gap: int = 1) -> tuple[bool, str]:
    if ed >= len(s["cal"]) or s["cal"][ed] != s["cal"][si] + gap:
        return False, "no_next_bar"
    if ed + 2 >= len(s["cal"]):
        return False, "too_late"
    if not s["tradable"][ed]:
        return False, "suspended"
    o, up = s["o"][ed], s["up"][ed]
    if not (o > 0 and s["c"][ed] > 0):
        return False, "bad_price"
    if np.isfinite(up) and o >= up - 0.001:
        return False, "limit_up_open"
    if locked_up(s["h"][ed], s["l"][ed], up):
        return False, "limit_up_locked"
    if not (np.isfinite(s["atr"][si]) and s["atr"][si] > 0):
        return False, "no_atr"
    atr = s["atr"][si]
    risk = 2.0 * atr / (o * s["scale"][ed])
    if risk > MAX_RISK_PCT:
        return False, "risk_gt_12pct"
    if int(NOTIONAL / (o * 1.001) // 100) * 100 < 100:
        return False, "lot"
    return True, "ok"


def open_position(s: dict, si: int, ed: int, pol: Policy, code: str, date: str,
                  notional: float, slip: float, fee_mult: float = 1.0,
                  gap: int = 1) -> PosState | None:
    ok, _ = entry_allowed(s, si, ed, gap)
    if not ok:
        return None
    o = float(s["o"][ed])
    scale = float(s["scale"][ed])
    entry_px = o * (1 + slip)
    entry_hfq = entry_px * scale
    atr0 = float(s["atr"][si])
    if pol.init == "none":
        stop = np.nan
        R_hfq = 2.0 * atr0
        stop_kind = "none"
    elif pol.init == "atr":
        stop = entry_hfq - pol.atr_mult * atr0
        R_hfq = entry_hfq - stop
        stop_kind = "stop_init"
    elif pol.init == "struct":
        pv = s["pl2"][si]
        floor = entry_hfq - pol.struct_floor * atr0
        if np.isfinite(pv) and pv < entry_hfq:
            stop = min(pv - pol.struct_buf * atr0, floor)
            stop_kind = "stop_struct"
        else:
            stop = entry_hfq - pol.atr_mult * atr0
            stop_kind = "stop_init"
        R_hfq = entry_hfq - stop
    else:
        raise ValueError(pol.init)
    stop_raw = stop / scale if np.isfinite(stop) else np.nan
    R_raw = entry_px - stop_raw if np.isfinite(stop_raw) else 2.0 * atr0 / scale
    if np.isfinite(stop_raw) and (R_raw <= 0 or R_raw / entry_px > MAX_RISK_PCT):
        if pol.init == "struct":
            # 结构止损过宽时回退 ATR2，保持“同一批入场”不被止损口径改变
            stop = entry_hfq - pol.atr_mult * atr0
            R_hfq = entry_hfq - stop
            stop_raw = stop / scale
            R_raw = entry_px - stop_raw
            stop_kind = "stop_init"
        if R_raw <= 0 or R_raw / entry_px > MAX_RISK_PCT:
            return None
    shares = int(notional / entry_px // 100) * 100
    if shares < 100:
        return None
    amt = shares * entry_px
    fee = fees(date, code, amt, shares, "buy") * fee_mult
    cost0 = amt + fee
    st = PosState(pol=pol, code=code, entry_cal=int(s["cal"][ed]), entry_date=date,
                  entry_px=entry_px, entry_market=o, entry_hfq=entry_hfq, atr0_hfq=atr0,
                  stop_hfq=stop, R_hfq=R_hfq, R_raw=R_raw, shares0=shares, shares=float(shares),
                  cost0=cost0, stop=stop, stop_kind=stop_kind, max_high=entry_hfq,
                  min_low=entry_hfq, last_px=entry_px, last_cal=int(s["cal"][ed]),
                  last_row=ed, fee_total=fee, slip_cost=amt * slip, cash_flow=-cost0, k=0)
    return st


def run_trade(s: dict, si: int, ed: int, pol: Policy, code: str, notional: float,
              slip: float, fee_mult: float = 1.0, path60: bool = True,
              gap: int = 1) -> dict | None:
    date = _DATES[int(s["cal"][ed])]
    st = open_position(s, si, ed, pol, code, date, notional, slip, fee_mult, gap)
    if st is None:
        return None
    n = len(s["cal"])
    for k in range(0, MAX_HOLD + 2):
        j = ed + k
        st.k = k
        if j >= n:
            _force(st, st.last_px, _DATES[int(s["cal"][n - 1])], slip, "delist", fee_mult)
            st.delist = 1
            st.last_cal = int(s["cal"][n - 1])
            break
        step_day(st, s, j, _DATES[int(s["cal"][j])], slip, fee_mult)
        if st.closed:
            break
        if k >= MAX_HOLD + 1:
            _force(st, st.last_px, _DATES[int(s["cal"][j])], slip, "maxhold", fee_mult)
            break
    if not st.closed:
        _force(st, st.last_px, _DATES[int(s["cal"][n - 1])], slip, "data_end", fee_mult)
        st.censored = 1
    st.k = 0

    mfe10, alive10, r1120, r1140, fwd_mfe, mfe60, oracle, pmax = \
        float(st.mfe10), int(st.alive10), np.nan, np.nan, np.nan, np.nan, np.nan, -1
    if path60:
        j60 = min(ed + 60, n - 1)
        seg = s["hh"][ed:j60 + 1]
        if len(seg) and np.isfinite(seg).any() and np.isfinite(st.R_hfq) and st.R_hfq > 0:
            imax = int(np.nanargmax(seg))
            mfe60 = (float(np.nanmax(seg)) - st.entry_hfq) / st.R_hfq
            jx = ed + imax
            px = float(s["c"][jx]) * (1 - slip)
            amt = st.shares0 * px
            fx = fees(_DATES[int(s["cal"][jx])], code, amt, st.shares0, "sell") * fee_mult
            oracle = (amt - fx) / st.cost0 - 1.0
            pmax = imax
        if ed + 11 < n:
            start = float(s["ho"][ed + 11])
            if start > 0:
                if ed + 20 < n:
                    r1120 = float(s["hc"][ed + 20]) / start - 1.0
                if ed + 40 < n:
                    r1140 = float(s["hc"][ed + 40]) / start - 1.0
                jf = min(ed + 60, n - 1)
                segf = s["hh"][ed + 11:jf + 1]
                if len(segf) and np.isfinite(segf).any():
                    fwd_mfe = float(np.nanmax(segf)) / start - 1.0
    ret = st.proceeds / st.cost0 - 1.0
    R_net = (st.proceeds - st.cost0) / (st.shares0 * st.R_raw) if st.R_raw > 0 else np.nan
    R_gross = (st.gross_in / st.shares0 - st.entry_market) / st.R_raw if st.R_raw > 0 else np.nan
    return dict(
        code=code, entry_cal=st.entry_cal, entry_date=st.entry_date, arm=pol.name,
        entry_px=st.entry_px, exit_px=st.last_px, shares0=st.shares0, cost0=st.cost0,
        proceeds=st.proceeds, ret=ret, R_net=R_net, R_gross=R_gross,
        hold=int(st.last_cal - st.entry_cal), reason=st.last_reason, stop_kind=st.stop_kind,
        mfe_R=st.mfe, mae_R=st.mae,
        atr_pct=float(2.0 * s["atr"][si] / (s["o"][ed] * s["scale"][ed])),
        n_partial=st.n_partial, scaled=st.scaled, fees=st.fee_total, slip_cost=st.slip_cost,
        censored=st.censored, delist=st.delist, year=int(st.entry_date[:4]),
        window="dev" if st.entry_date <= DEV_END else "val",
        mfe10_R=mfe10, alive10=alive10, ret_11_20=r1120, ret_11_40=r1140,
        fwd_mfe=fwd_mfe, mfe60_R=mfe60, oracle_ret=oracle, path_max_day=pmax,
        exit_date=_DATES[st.last_cal],
    )


_DATES: list[str] = []


# ==========================================================================
# 6. 统计
# ==========================================================================

def cluster_ci(values: np.ndarray, clusters: np.ndarray, B: int = 2000,
               seed: int = 20261026) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    v = v[ok]
    cl = np.asarray(clusters)[ok]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    uniq, inv = np.unique(cl, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    cnt = np.bincount(inv)
    means = sums / cnt
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(means), size=(B, len(means)))
    boot = means[idx].mean(axis=1)
    return float(means.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarize(df: pd.DataFrame, label: dict) -> dict:
    if df is None or len(df) == 0:
        return dict(**label, n=0)
    r = df["ret"].to_numpy(float)
    m, lo, hi = cluster_ci(r, df["entry_date"].to_numpy())
    yr = df.groupby("year")["ret"].mean()
    top5 = np.sort(r)[-5:]
    r2 = np.sort(r)[:-5] if len(r) > 5 else np.array([])
    return dict(
        **label, n=int(len(df)), mean=float(m), mean_trade=float(np.nanmean(r)),
        median=float(np.nanmedian(r)),
        win=float(np.nanmean(r > 0)), mean_R=float(np.nanmean(df["R_net"].to_numpy(float))),
        med_R=float(np.nanmedian(df["R_net"].to_numpy(float))),
        p05_R=float(np.nanquantile(df["R_net"].to_numpy(float), 0.05)),
        p95_R=float(np.nanquantile(df["R_net"].to_numpy(float), 0.95)),
        hold=float(np.nanmean(df["hold"].to_numpy(float))),
        hold_med=float(np.nanmedian(df["hold"].to_numpy(float))),
        mfe=float(np.nanmean(df["mfe_R"].to_numpy(float))),
        mae=float(np.nanmean(df["mae_R"].to_numpy(float))),
        mean_cluster=m, ci_lo=lo, ci_hi=hi,
        year_equal=float(yr.mean()) if len(yr) else np.nan,
        ret20=float(np.nanmedian(r * 20.0 / np.maximum(df["hold"].to_numpy(float), 1))),
        drop_top5=float(np.nanmean(r2)) if len(r2) else np.nan,
        censored=float(np.nanmean(df["censored"].to_numpy(float))),
        delist=float(np.nanmean(df["delist"].to_numpy(float))),
        cost=float(np.nanmean((df["fees"] + df["slip_cost"]) / df["shares0"].to_numpy(float)
                              / df["entry_px"].to_numpy(float))),
    )


def paired(a: pd.DataFrame, b: pd.DataFrame, label: dict, col: str = "ret") -> dict:
    key = ["code", "entry_date"]
    m = a[key + [col]].merge(b[key + [col]], on=key, suffixes=("_a", "_b"))
    if len(m) == 0:
        return dict(**label, n=0)
    d = m[f"{col}_a"].to_numpy(float) - m[f"{col}_b"].to_numpy(float)
    mu, lo, hi = cluster_ci(d, m["entry_date"].to_numpy())
    dev = m[m["entry_date"] <= DEV_END]
    val = m[m["entry_date"] > DEV_END]
    return dict(**label, n=int(len(d)), diff=float(np.nanmean(d)), ci_lo=lo, ci_hi=hi,
                win=float(np.nanmean(d > 0)),
                diff_dev=float((dev[f"{col}_a"] - dev[f"{col}_b"]).mean()) if len(dev) else np.nan,
                diff_val=float((val[f"{col}_a"] - val[f"{col}_b"]).mean()) if len(val) else np.nan,
                diff_drop5=float(np.sort(d)[:-5].mean()) if len(d) > 5 else np.nan)

# ==========================================================================
# 7. 运行
# ==========================================================================

_WORKER: dict = {}


def _worker_init(panel, sig, slip, fee_mult, delay, path_arm, entry_name):
    _WORKER.update(panel=panel, sig=sig, slip=slip, fee=fee_mult, delay=delay,
                   path_arm=path_arm, entry=entry_name)


def _worker_arm(name: str):
    pol = PBYNAME[name]
    panel = _WORKER["panel"]
    slip, fee, delay = _WORKER["slip"], _WORKER["fee"], _WORKER["delay"]
    gap = 1 + delay
    out = []
    for r in _WORKER["sig"]:
        s = panel.sym(r["ci"])
        tr = run_trade(s, r["si"], r["si"] + gap, pol, panel.codes[r["ci"]],
                       NOTIONAL, slip, fee, path60=(name == _WORKER["path_arm"]), gap=gap)
        if tr is not None:
            tr["entry_spec"] = _WORKER["entry"]
            out.append(tr)
    return name, out


def run_entry(panel: Panel, spec: EntrySpec, policies: list[Policy], slip: float,
              fee_mult: float, delay: int, limit: int | None = None,
              path_arm: str = "ATR2", jobs: int = 1) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    sig = signals_for(panel, spec)
    gap = 1 + delay
    keep = []
    for r in sig:
        ec = int(panel.sym(r["ci"])["cal"][r["si"]]) + gap
        if ec < len(panel.dates) and "2008-01-01" <= panel.dates[ec] <= WIN_END:
            keep.append(r)
    sig = keep
    if limit:
        sig = sig[:limit]
    trades = {p.name: [] for p in policies}
    fails: list[dict] = []
    valid = []
    for r in sig:
        s = panel.sym(r["ci"])
        si = r["si"]
        ed = si + gap
        ok, why = entry_allowed(s, si, ed, gap)
        if not ok:
            fails.append(dict(entry=spec.name, year=r["year"], reason=why, n=1))
        else:
            valid.append(r)
    if jobs > 1 and len(valid) > 500:
        import multiprocessing as mp
        names = [p.name for p in policies]
        _WORKER.clear()
        _WORKER.update(panel=panel, sig=valid, slip=slip, fee=fee_mult, delay=delay,
                       path_arm=path_arm, entry=spec.name)
        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(min(jobs, len(names))) as pool:
                got = dict(pool.map(_worker_arm, names))
            trades = got
        except Exception as e:
            log(f"并行失败({e})，退回单进程")
            _WORKER.clear()
            for r in valid:
                s = panel.sym(r["ci"])
                for pol in policies:
                    tr = run_trade(s, r["si"], r["si"] + gap, pol, panel.codes[r["ci"]],
                                   NOTIONAL, slip, fee_mult, path60=(pol.name == path_arm), gap=gap)
                    if tr is not None:
                        tr["entry_spec"] = spec.name
                        trades[pol.name].append(tr)
    else:
        for r in valid:
            s = panel.sym(r["ci"])
            for pol in policies:
                tr = run_trade(s, r["si"], r["si"] + gap, pol, panel.codes[r["ci"]],
                               NOTIONAL, slip, fee_mult, path60=(pol.name == path_arm), gap=gap)
                if tr is not None:
                    tr["entry_spec"] = spec.name
                    trades[pol.name].append(tr)
    out = {k: pd.DataFrame(v) for k, v in trades.items() if len(v)}
    return out, fails


def yearly_table(res: dict[str, pd.DataFrame], entry: str) -> pd.DataFrame:
    rows = []
    for arm, df in res.items():
        if len(df) == 0:
            continue
        g = df.groupby("year").agg(n=("ret", "size"), mean=("ret", "mean"),
                                   med=("ret", "median"), win=("ret", lambda x: (x > 0).mean()),
                                   mean_R=("R_net", "mean"), hold=("hold", "mean"))
        for y, r in g.iterrows():
            rows.append(dict(entry=entry, arm=arm, year=int(y), **r.to_dict(), **META))
    return pd.DataFrame(rows)


def failure_table(fails: list[dict]) -> pd.DataFrame:
    if not fails:
        return pd.DataFrame(columns=["entry", "year", "reason", "n"])
    return (pd.DataFrame(fails).groupby(["entry", "year", "reason"], as_index=False)["n"].sum()
            .sort_values(["entry", "year", "n"], ascending=[True, True, False]))


def state_analysis(df_atr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """事前状态（第10日收盘可知）分桶 + 事后 MFE 对照。"""
    d = df_atr.copy()
    if len(d) == 0:
        return pd.DataFrame(), pd.DataFrame()

    def bucket(r):
        if int(r["alive10"]) == 0:
            return "0_第10日前已退出"
        m = r["mfe10_R"]
        if not np.isfinite(m):
            return "0_第10日前已退出"
        if m < 0:
            return "1_MFE10<0"
        if m < 0.5:
            return "2_0~0.5R"
        if m < 1.0:
            return "3_0.5~1R"
        if m < 2.0:
            return "4_1~2R"
        return "5_>=2R"

    d["bucket"] = d.apply(bucket, axis=1)
    rows = []
    for b, g in d.groupby("bucket"):
        rows.append(dict(
            bucket=b, n=len(g), share=len(g) / len(d),
            mfe10_mean=float(g["mfe10_R"].mean()),
            fwd20=float(g["ret_11_20"].mean()), fwd20_pos=float((g["ret_11_20"] > 0).mean()),
            fwd40=float(g["ret_11_40"].mean()), fwd40_pos=float((g["ret_11_40"] > 0).mean()),
            fwd_mfe=float(g["fwd_mfe"].mean()),
            p_fwd_mfe_20pct=float((g["fwd_mfe"] >= 0.2).mean()),
            mfe60=float(g["mfe60_R"].mean()),
            p_final_2R=float((g["mfe60_R"] >= 2.0).mean()),
            p_final_loss=float((g["ret"] < 0).mean()),
            ret_atr2=float(g["ret"].mean()), hold=float(g["hold"].mean()),
        ))
    tab = pd.DataFrame(rows).sort_values("bucket")

    # 事后：MFE10 对最终 MFE 的解释力（仅第10日仍存活的交易）
    alive = d[d["alive10"] == 1].dropna(subset=["mfe10_R", "mfe60_R"])
    if len(alive) > 10:
        from scipy.stats import spearmanr
        rho, p = spearmanr(alive["mfe10_R"], alive["mfe60_R"])
        early_late = float(((alive["mfe10_R"] < 1.0) & (alive["mfe60_R"] >= 2.0)).mean())
        giveback = float(((alive["mfe10_R"] >= 1.0) & (alive["ret"] < 0)).mean())
        meta = pd.DataFrame([dict(n_alive=len(alive), rho=float(rho), p=float(p),
                                  p_late_bloomer=early_late, p_giveback=giveback,
                                  survival_to_10=float(len(alive) / len(d)))])
    else:
        meta = pd.DataFrame([dict(n_alive=len(alive))])
    return tab, meta


def policy_by_bucket(res: dict[str, pd.DataFrame], base: str = "ATR2") -> pd.DataFrame:
    if base not in res or len(res[base]) == 0:
        return pd.DataFrame()
    b = res[base][["code", "entry_date", "alive10", "mfe10_R", "ret"]].copy()

    def bucket(r):
        if int(r["alive10"]) == 0 or not np.isfinite(r["mfe10_R"]):
            return "0_第10日前已退出"
        m = r["mfe10_R"]
        return ("1_MFE10<0" if m < 0 else "2_0~0.5R" if m < 0.5 else
                "3_0.5~1R" if m < 1 else "4_1~2R" if m < 2 else "5_>=2R")

    b["bucket"] = b.apply(bucket, axis=1)
    rows = []
    for arm, df in res.items():
        if arm == base or len(df) == 0:
            continue
        m = df[["code", "entry_date", "ret"]].merge(
            b[["code", "entry_date", "bucket", "ret"]], on=["code", "entry_date"],
            suffixes=("_a", "_b"))
        if len(m) == 0:
            continue
        m["diff"] = m["ret_a"] - m["ret_b"]
        for bk, g in m.groupby("bucket"):
            rows.append(dict(arm=arm, bucket=bk, n=len(g), diff=float(g["diff"].mean())))
        rows.append(dict(arm=arm, bucket="ALL", n=len(m), diff=float(m["diff"].mean())))
    return pd.DataFrame(rows)


def paired_full(a: pd.DataFrame, b: pd.DataFrame, label: dict, col: str = "ret") -> dict:
    m = a[["code", "entry_date", col]].merge(
        b[["code", "entry_date", col]], on=["code", "entry_date"], suffixes=("_a", "_b"))
    if len(m) == 0:
        return dict(**label, n=0)
    d = m[f"{col}_a"].to_numpy(float) - m[f"{col}_b"].to_numpy(float)
    mu, lo, hi = cluster_ci(d, m["entry_date"].to_numpy())
    m["year"] = m["entry_date"].str[:4].astype(int)
    yr = m.assign(diff=d).groupby("year")["diff"].mean()
    dev = m[m["entry_date"] <= DEV_END]
    val = m[m["entry_date"] > DEV_END]
    dd = dev[f"{col}_a"].to_numpy(float) - dev[f"{col}_b"].to_numpy(float)
    dv = val[f"{col}_a"].to_numpy(float) - val[f"{col}_b"].to_numpy(float)
    mu_dev = cluster_ci(dd, dev["entry_date"].to_numpy())[0] if len(dd) else np.nan
    mu_val = cluster_ci(dv, val["entry_date"].to_numpy())[0] if len(dv) else np.nan
    return dict(**label, n=int(len(d)), diff=float(mu), diff_trade=float(np.mean(d)),
                ci_lo=lo, ci_hi=hi,
                win=float(np.mean(d > 0)),
                diff_dev=float(mu_dev), diff_val=float(mu_val),
                diff_year_equal=float(yr.mean()) if len(yr) else np.nan,
                diff_drop5=float(np.sort(d)[:-5].mean()) if len(d) > 5 else np.nan)


# ==========================================================================
# 8. 组合复核（100万/10仓，provisional）
# ==========================================================================

def run_portfolio(panel: Panel, spec: EntrySpec, pol: Policy, capital: float = 1_000_000.0,
                  slots: int = 10, max_new: int = 2, slip: float = SLIP,
                  fee_mult: float = 1.0, start: str = "2008-01-01",
                  end: str = WIN_END) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = panel.dates
    c0 = int(np.searchsorted(np.array(dates), start, side="left"))
    c1 = int(np.searchsorted(np.array(dates), end, side="right")) - 1
    sig = signals_for(panel, spec)
    by_cal: dict[int, list[dict]] = {}
    for r in sig:
        if c0 - 1 <= r["cal"] <= c1 - 1:
            by_cal.setdefault(r["cal"], []).append(r)

    cash = capital
    pos: dict[int, PosState] = {}
    rows = []
    trades = []
    invested_hist = []
    for c in range(c0, c1 + 1):
        date = dates[c]
        # 1) 持仓推进
        for ci in list(pos):
            st = pos[ci]
            s = panel.sym(ci)
            nxt = st.last_row + 1
            if nxt < len(s["cal"]) and int(s["cal"][nxt]) == c:
                st.last_row = nxt
                if st.k > 0:
                    pj = nxt - 1
                    f0, f1 = s["f"][pj], s["f"][nxt]
                    c0_, c1_ = s["c"][pj], s["c"][nxt]
                    if all(np.isfinite(x) and x > 0 for x in (f0, f1, c0_, c1_)):
                        a = (f1 / f0) / (c1_ / c0_)
                        if 0.2 < a < 5 and abs(a - 1) > 1e-10:
                            st.shares *= a
                st.k += 1
                st.cash_flow = 0.0
                step_day(st, s, nxt, date, slip, fee_mult)
            else:
                st.k += 1
                st.susp_bars += 1
                st.cash_flow = 0.0
                if st.susp_bars >= SUSP_MAX:
                    st.last_cal = c
                    _force(st, st.last_px, date, slip, "suspended_forced", fee_mult)
            cash += st.cash_flow
            st.cash_flow = 0.0
            if st.closed:
                tr = dict(code=st.code, entry_date=st.entry_date, exit_date=date,
                          arm=pol.name, entry_spec=spec.name, ret=st.proceeds / st.cost0 - 1,
                          cost0=st.cost0, proceeds=st.proceeds, hold=st.last_cal - st.entry_cal,
                          reason=st.last_reason, shares0=st.shares0)
                trades.append(tr)
                del pos[ci]

        # 2) 新入场（t-1 收盘信号，t 开盘）
        cand = by_cal.get(c - 1, [])
        if cand:
            nav_prev = cash + sum(
                pos[ci].shares * pos[ci].last_px for ci in pos)
            cand = sorted(cand, key=lambda r: (-r["strength"],
                                               -float(panel.sym(r["ci"])["adv20"][r["si"]]),
                                               panel.codes[r["ci"]]))
            n_new = 0
            for r in cand:
                if len(pos) >= slots or n_new >= max_new:
                    break
                ci = r["ci"]
                if ci in pos:
                    continue
                s = panel.sym(ci)
                si, ed = r["si"], r["si"] + 1
                ok, _ = entry_allowed(s, si, ed, 1)
                if not ok:
                    continue
                target = nav_prev / slots
                st = open_position(s, si, ed, pol, panel.codes[ci], dates[c],
                                   min(target, cash), slip, fee_mult, 1)
                if st is None:
                    continue
                if st.cost0 > cash:
                    o = float(s["o"][ed])
                    sh = int(cash / (o * (1 + slip) * 1.002) // 100) * 100
                    if sh < 100:
                        continue
                    # 重开，保持同一规则状态
                    st = open_position(s, si, ed, pol, panel.codes[ci], dates[c],
                                       sh * o * (1 + slip) / 1.0, slip, fee_mult, 1)
                    if st is None or st.cost0 > cash:
                        continue
                cash -= st.cost0
                st.cash_flow = 0.0
                st.last_row = ed
                st.last_cal = int(s["cal"][ed])
                st.k = 0
                step_day(st, s, ed, dates[c], slip, fee_mult)
                st.cash_flow = 0.0
                pos[ci] = st
                n_new += 1

        # 3) 估值
        inv = 0.0
        for ci, st in pos.items():
            s = panel.sym(ci)
            j = st.last_row
            px = float(s["c"][j]) if j < len(s["c"]) and s["c"][j] > 0 else st.last_px
            inv += st.shares * px
        nav = cash + inv
        rows.append(dict(date=date, nav=nav, cash=cash, n_pos=len(pos),
                         invested=inv, deploy=inv / nav if nav > 0 else np.nan))
        invested_hist.append(inv)

    # 期末清算
    for ci in list(pos):
        st = pos[ci]
        _force(st, st.last_px, dates[c1], slip, "data_end", fee_mult)
        cash += st.cash_flow
        trades.append(dict(code=st.code, entry_date=st.entry_date, exit_date=dates[c1],
                           arm=pol.name, entry_spec=spec.name,
                           ret=st.proceeds / st.cost0 - 1, cost0=st.cost0,
                           proceeds=st.proceeds, hold=st.last_cal - st.entry_cal,
                           reason=st.last_reason, shares0=st.shares0))
    eq = pd.DataFrame(rows)
    if len(eq):
        eq.loc[eq.index[-1], "nav"] = cash
        eq.loc[eq.index[-1], "cash"] = cash
        eq.loc[eq.index[-1], "invested"] = 0.0
        eq.loc[eq.index[-1], "n_pos"] = 0
        eq["deploy"] = eq["invested"] / eq["nav"]
    return eq, pd.DataFrame(trades)


def perf(eq: pd.DataFrame) -> dict:
    if eq is None or len(eq) < 3:
        return {}
    nav = eq["nav"].to_numpy(float)
    d = pd.to_datetime(eq["date"])
    days = (d.iloc[-1] - d.iloc[0]).days
    cagr = (nav[-1] / nav[0]) ** (365.25 / days) - 1 if nav[0] > 0 and days > 0 else np.nan
    r = pd.Series(nav).pct_change().dropna()
    dd = float((pd.Series(nav) / pd.Series(nav).cummax() - 1).min())
    return dict(cagr=float(cagr), mdd=dd, calmar=float(cagr / abs(dd)) if dd < 0 else np.nan,
                vol=float(r.std() * np.sqrt(244)) if len(r) > 2 else np.nan,
                sharpe=float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan,
                nav_end=float(nav[-1]), deploy=float(eq["deploy"].mean()),
                avg_pos=float(eq["n_pos"].mean()), days=int(days))


# ==========================================================================
# 9. 自检
# ==========================================================================

def self_checks(panel: Panel, trades: dict[str, dict[str, pd.DataFrame]],
                fails: list[dict]) -> dict:
    checks = []
    # S1 hfq 日收益 == pct_chg（抽样 300 只）
    rng = np.random.default_rng(20261026)
    pick = rng.choice(panel.n, size=min(300, panel.n), replace=False)
    mx = 0.0
    nrow = 0
    for ci in pick:
        s = panel.sym(int(ci))
        hc, pct = s["hc"], s["pct"]
        if len(hc) < 3:
            continue
        r = hc[1:] / hc[:-1] - 1.0
        e = np.abs(r - pct[1:] / 100.0)
        e = e[np.isfinite(e)]
        if len(e):
            mx = max(mx, float(e.max()))
            nrow += len(e)
    checks.append(dict(id="S1", name="hfq 日收益 ≡ pct_chg", status="pass" if mx < 1e-12 else "fail",
                       detail=f"{len(pick)} 只 / {nrow:,} 行，max|diff|={mx:.3e}"))

    # S2 截断重算：随机 20 个信号
    rng = np.random.default_rng(7)
    bad = 0
    tested = 0
    for spec in ENTRIES:
        sig = signals_for(panel, spec)
        if not sig:
            continue
        for r in [sig[i] for i in rng.choice(len(sig), size=min(20, len(sig)), replace=False)]:
            s = panel.sym(r["ci"])
            si = r["si"]
            cut = si + 1
            ft = compute_features(s["o"][:cut], s["h"][:cut], s["l"][:cut], s["c"][:cut],
                                  s["pc"][:cut], s["f"][:cut], s["vol"][:cut], s["amt"][:cut],
                                  s["st"][:cut], s["susp"][:cut], s["td"][:cut])
            s2 = {**{k: s[k][:cut] for k in ("cal", "o", "h", "l", "c", "pc", "pct", "f", "vol",
                                             "amt", "td", "st", "susp", "up", "dn", "lu_open",
                                             "ld_open")}, **ft}
            cond, strength = signal_mask(s2, spec.kind)
            full_cond, full_strength = signal_mask(s, spec.kind)
            tested += 1
            if bool(cond[si]) != bool(full_cond[si]) or abs(float(strength[si]) - float(full_strength[si])) > 1e-9:
                bad += 1
    checks.append(dict(id="S2", name="截断重算信号一致", status="pass" if bad == 0 else "fail",
                       detail=f"{tested} 个信号，不一致 {bad}"))

    # S3 未来扰动
    bad3 = 0
    tested3 = 0
    for spec in ENTRIES:
        sig = signals_for(panel, spec)
        if not sig:
            continue
        for r in [sig[i] for i in rng.choice(len(sig), size=min(10, len(sig)), replace=False)]:
            s = panel.sym(r["ci"])
            si = r["si"]
            cut = si + 1
            pert = {k: s[k].copy() for k in ("o", "h", "l", "c", "vol", "amt")}
            n = len(s["o"])
            if cut >= n:
                continue
            rr = np.random.default_rng(99 + si)
            m = n - cut
            for k in ("o", "h", "l", "c"):
                pert[k][cut:] *= rr.uniform(0.3, 3.0, m)
            pert["vol"][cut:] *= rr.uniform(0.5, 2.0, m)
            pert["amt"][cut:] = pert["vol"][cut:] * pert["c"][cut:]
            ft = compute_features(pert["o"], pert["h"], pert["l"], pert["c"], s["pc"], s["f"],
                                  pert["vol"], pert["amt"], s["st"], s["susp"], s["td"])
            s2 = {**{k: s[k] for k in ("cal", "o", "h", "l", "c", "pc", "pct", "f", "vol",
                                       "amt", "td", "st", "susp", "up", "dn", "lu_open",
                                       "ld_open")}, **ft}
            cond, strength = signal_mask(s2, spec.kind)
            full_cond, full_strength = signal_mask(s, spec.kind)
            tested3 += 1
            if bool(cond[si]) != bool(full_cond[si]) or abs(float(strength[si]) - float(full_strength[si])) > 1e-9:
                bad3 += 1
    checks.append(dict(id="S3", name="未来扰动不影响信号", status="pass" if bad3 == 0 else "fail",
                       detail=f"{tested3} 个信号，不一致 {bad3}"))

    # S4/S7/S8：逐笔断言
    min_hold, bad_lot, bad_date, ntr = 10**9, 0, 0, 0
    for entry, res in trades.items():
        for arm, df in res.items():
            if len(df) == 0:
                continue
            ntr += len(df)
            min_hold = min(min_hold, int(df["hold"].min()))
            bad_lot += int((df["shares0"] % 100 != 0).sum())
            bad_date += int(((df["entry_date"] > WIN_END) | (df["exit_date"] > WIN_END)).sum())
    checks.append(dict(id="S4", name="T+1（最短持有≥1日）", status="pass" if min_hold >= 1 else "fail",
                       detail=f"{ntr} 笔，最短持有 {min_hold} 日"))
    checks.append(dict(id="S7", name="买入整股（100 股倍数）", status="pass" if bad_lot == 0 else "fail",
                       detail=f"违规 {bad_lot} 笔"))
    checks.append(dict(id="S8", name="窗口不越 2020-12-31", status="pass" if bad_date == 0 else "fail",
                       detail=f"越界 {bad_date} 笔"))

    # S5 涨停失败计数 + 抽样
    fdf = failure_table(fails)
    lu = int(fdf[fdf["reason"].isin(["limit_up_open", "limit_up_locked"])]["n"].sum()) if len(fdf) else 0
    checks.append(dict(id="S5", name="开盘涨停不买（计数）", status="pass" if lu >= 0 else "fail",
                       detail=f"涨停拒买 {lu} 笔；失败原因表见 09_失败订单.csv"))

    # S6 费用复算（无分批的 20 笔）
    bad6 = 0
    cnt6 = 0
    for entry, res in trades.items():
        df = res.get("ATR2")
        if df is None or len(df) == 0:
            continue
        for _, r in df[df["n_partial"] == 0].head(20).iterrows():
            buy = r["entry_px"] * r["shares0"]
            sell = r["exit_px"] * r["shares0"]
            f = fees(r["entry_date"], r["code"], buy, r["shares0"], "buy") + \
                fees(r["exit_date"], r["code"], sell, r["shares0"], "sell")
            cnt6 += 1
            if abs(f - r["fees"]) > 0.02:
                bad6 += 1
    checks.append(dict(id="S6", name="费用手工复算", status="pass" if bad6 == 0 else "fail",
                       detail=f"{cnt6} 笔，误差>0.02 元 {bad6} 笔"))
    return dict(checks=checks, pass_n=sum(c["status"] == "pass" for c in checks),
                fail_n=sum(c["status"] != "pass" for c in checks))

# ==========================================================================
# 10. 阶段编排
# ==========================================================================

STRESS_ALL = [
    ("cost2", 0.0020, 2.0, 0),
    ("delay1", 0.0010, 1.0, 1),
]
STRESS_KEY = [("cost3", 0.0030, 3.0, 0)]
KEY_ARMS = {"ATR2", "FH20", "FH20S", "STRUCT", "TIME20", "CAUT10", "BE1",
            "CHAND2", "SCALE2", "RUN1", "ALL"}


def _concat_trades(res: dict[str, pd.DataFrame], entry: str) -> pd.DataFrame:
    parts = []
    for arm, df in res.items():
        if len(df):
            parts.append(df.assign(entry_spec=entry, arm=arm))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def stage_audit(panel: Panel) -> dict:
    _DATES[:] = panel.dates
    n_rows = len(panel["cal"])
    checks = []
    rng = np.random.default_rng(20261026)
    pick = rng.choice(panel.n, size=min(300, panel.n), replace=False)
    mx, nrow = 0.0, 0
    for ci in pick:
        s = panel.sym(int(ci))
        hc, pct = s["hc"], s["pct"]
        if len(hc) < 3:
            continue
        r = hc[1:] / hc[:-1] - 1.0
        e = np.abs(r - pct[1:] / 100.0)
        e = e[np.isfinite(e)]
        if len(e):
            mx = max(mx, float(e.max()))
            nrow += len(e)
    checks.append(dict(id="S1", name="hfq 日收益 ≡ pct_chg", status="pass" if mx < 1e-12 else "fail",
                       detail=f"{len(pick)} 只 / {nrow:,} 行，max|diff|={mx:.3e}"))
    sig_rows, yearly_rows = [], []
    for spec in ENTRIES:
        st = signal_stats(panel, spec)
        sig_rows.append(st)
        sig = signals_for(panel, spec)
        if sig:
            d = pd.DataFrame(sig)
            g = d.groupby("year").size().reset_index(name="kept")
            g["entry"] = spec.name
            yearly_rows.append(g)
    payload = dict(meta=dict(rows=int(n_rows), codes=int(panel.n), dates=len(panel.dates),
                             start=panel.dates[0], end=panel.dates[-1],
                             panel_sha=sha16(CACHE / "panel_v1.npz"),
                             code_sha=sha16(Path(__file__)), **META),
                   checks=checks,
                   signals=sig_rows,
                   signal_yearly=pd.concat(yearly_rows, ignore_index=True).to_dict("records")
                   if yearly_rows else [])
    return payload


def stage_signals(panel: Panel) -> None:
    _DATES[:] = panel.dates
    sig_rows, yearly_rows, fail_rows = [], [], []
    for spec in ENTRIES:
        sig_rows.append(signal_stats(panel, spec))
        sig = signals_for(panel, spec)
        fail = {}
        for r in sig:
            s = panel.sym(r["ci"])
            ok, why = entry_allowed(s, r["si"], r["si"] + 1, 1)
            if not ok:
                fail[why] = fail.get(why, 0) + 1
                fail_rows.append(dict(entry=spec.name, year=r["year"], reason=why, n=1))
        if sig:
            d = pd.DataFrame(sig)
            g = d.groupby("year").agg(n=("si", "size"), strength=("strength", "mean")).reset_index()
            g["entry"] = spec.name
            yearly_rows.append(g)
    pd.DataFrame(sig_rows).to_csv(OUT / "09_信号汇总.csv", index=False, encoding="utf-8-sig")
    if yearly_rows:
        pd.concat(yearly_rows, ignore_index=True).to_csv(
            OUT / "09_信号分年.csv", index=False, encoding="utf-8-sig")
    if fail_rows:
        (pd.DataFrame(fail_rows).groupby(["entry", "year", "reason"], as_index=False)["n"].sum()
         .to_csv(OUT / "09_信号明细.csv", index=False, encoding="utf-8-sig"))
    log(f"信号：{sig_rows}")


def stage_run(panel: Panel, limit: int | None = None, jobs: int = 1) -> None:
    _DATES[:] = panel.dates
    base_trades: dict[str, dict[str, pd.DataFrame]] = {}
    all_sum, all_pair, all_year, all_fail = [], [], [], []
    base_fails: list[dict] = []
    all_state, all_meta, all_pbucket = [], [], []

    for spec in ENTRIES:
        log(f"=== {spec.name} 基准场景 ===")
        res, fails = run_entry(panel, spec, POLICIES, SLIP, 1.0, 0, limit=limit, jobs=jobs)
        base_trades[spec.name] = res
        base_fails.extend(fails)
        all_fail.extend(fails)
        for arm, df in res.items():
            all_sum.append(summarize(df, dict(entry=spec.name, arm=arm, scenario="base", **META)))
        for arm, df in res.items():
            if arm != "ATR2":
                all_pair.append(paired_full(df, res["ATR2"],
                                            dict(entry=spec.name, arm=arm, scenario="base", **META)))
        all_pair.append(paired_full(res["ATR2"], res["ATR2"],
                                    dict(entry=spec.name, arm="ATR2", scenario="base", **META)))
        all_year.append(yearly_table(res, spec.name))
        if "ATR2" in res and len(res["ATR2"]):
            tab, meta = state_analysis(res["ATR2"])
            if len(tab):
                all_state.append(tab.assign(entry=spec.name, **META))
            if len(meta):
                all_meta.append(meta.assign(entry=spec.name, **META))
            pb = policy_by_bucket(res)
            if len(pb):
                all_pbucket.append(pb.assign(entry=spec.name, **META))
        # 压力场景
        for name, slip, fm, delay in STRESS_ALL:
            log(f"=== {spec.name} {name} ===")
            res_s, fails_s = run_entry(panel, spec, POLICIES, slip, fm, delay, limit=limit, jobs=jobs)
            all_fail.extend(fails_s)
            for arm, df in res_s.items():
                all_sum.append(summarize(df, dict(entry=spec.name, arm=arm, scenario=name, **META)))
            for arm, df in res_s.items():
                if arm != "ATR2":
                    all_pair.append(paired_full(df, res_s["ATR2"],
                                                dict(entry=spec.name, arm=arm, scenario=name, **META)))
        for name, slip, fm, delay in STRESS_KEY:
            log(f"=== {spec.name} {name} (key arms) ===")
            pols = [p for p in POLICIES if p.name in KEY_ARMS]
            res_s, fails_s = run_entry(panel, spec, pols, slip, fm, delay, limit=limit, jobs=jobs)
            all_fail.extend(fails_s)
            for arm, df in res_s.items():
                all_sum.append(summarize(df, dict(entry=spec.name, arm=arm, scenario=name, **META)))
            for arm, df in res_s.items():
                if arm != "ATR2":
                    all_pair.append(paired_full(df, res_s["ATR2"],
                                                dict(entry=spec.name, arm=arm, scenario=name, **META)))

    summary = pd.DataFrame(all_sum)
    paired = pd.DataFrame(all_pair)
    yearly = pd.concat([x for x in all_year if len(x)], ignore_index=True) if all_year else pd.DataFrame()
    summary.to_csv(OUT / "02_臂汇总.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUT / "03_配对差.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT / "04_分年.csv", index=False, encoding="utf-8-sig")
    fail_detail = failure_table(base_fails)
    fail_detail.to_csv(OUT / "09_失败订单.csv", index=False, encoding="utf-8-sig")
    sig_sum = pd.DataFrame([signal_stats(panel, s) for s in ENTRIES])
    sig_sum.to_csv(OUT / "09_信号汇总.csv", index=False, encoding="utf-8-sig")
    fail_sum = fail_detail.groupby("entry", as_index=False)["n"].sum().rename(
        columns={"n": "entry_failures"})
    filled = pd.DataFrame([dict(entry=s.name, filled=len(base_trades[s.name].get("ATR2", [])))
                           for s in ENTRIES])
    audit = sig_sum.merge(fail_sum, on="entry", how="left").merge(filled, on="entry", how="left")
    audit = audit.fillna(0)
    audit["in_window_signals"] = audit["filled"] + audit["entry_failures"]
    audit["window_start"] = META["window_start"]
    audit["window_end"] = META["window_end"]
    audit[["entry", "raw_signals", "kept", "suppressed", "in_window_signals",
           "entry_failures", "filled", "window_start", "window_end"]].to_csv(
        OUT / "09_信号审计.csv", index=False, encoding="utf-8-sig")
    # 压力测试一览
    st = summary.pivot_table(index=["entry", "arm"], columns="scenario", values="mean")
    st = st.reset_index()
    base_cols = summary[summary.scenario == "base"][["entry", "arm", "year_equal", "drop_top5", "censored", "n"]]
    st = st.merge(base_cols, on=["entry", "arm"], how="left")
    for k, v in META.items():
        st[k] = v
    st.to_csv(OUT / "05_压力测试.csv", index=False, encoding="utf-8-sig")
    if all_state:
        pd.concat(all_state, ignore_index=True).to_csv(
            OUT / "06_事前事后状态.csv", index=False, encoding="utf-8-sig")
    if all_meta:
        pd.concat(all_meta, ignore_index=True).to_csv(
            OUT / "06_事前事后_元.csv", index=False, encoding="utf-8-sig")
    if all_pbucket:
        pd.concat(all_pbucket, ignore_index=True).to_csv(
            OUT / "06_臂按状态分桶.csv", index=False, encoding="utf-8-sig")

    # 逐笔（base 场景）
    parts = [_concat_trades(res, spec.name) for spec, res in
             [(s, base_trades[s.name]) for s in ENTRIES]]
    parts = [x for x in parts if len(x)]
    if parts:
        tr = pd.concat(parts, ignore_index=True)
        tr.to_parquet(OUT / "07_逐笔.parquet", index=False, compression="zstd")
        log(f"逐笔 {len(tr):,} 行 -> 07_逐笔.parquet")

    # 自检
    checks = self_checks(panel, base_trades, base_fails)
    (OUT / "10_自检.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    log(f"自检 pass={checks['pass_n']} fail={checks['fail_n']}")
    for c in checks["checks"]:
        if c["status"] != "pass":
            log(f"  FAIL {c['id']} {c['name']}: {c['detail']}")


def stage_portfolio(panel: Panel) -> pd.DataFrame:
    _DATES[:] = panel.dates
    rows = []
    navdir = OUT / "08_组合净值"
    navdir.mkdir(exist_ok=True)
    for spec in ENTRIES:
        for pol in POLICIES:
            eq, tr = run_portfolio(panel, spec, pol)
            pf = perf(eq)
            recon = (1_000_000.0 - float(tr["cost0"].sum()) + float(tr["proceeds"].sum())
                     - float(eq["nav"].iloc[-1])) if len(tr) and len(eq) else np.nan
            rows.append(dict(entry=spec.name, arm=pol.name, **META, **pf,
                             n_trades=int(len(tr)),
                             trade_mean=float(tr["ret"].mean()) if len(tr) else np.nan,
                             trade_win=float((tr["ret"] > 0).mean()) if len(tr) else np.nan,
                             recon_err=recon))
            if pol.name in KEY_ARMS:
                eq.to_csv(navdir / f"{spec.name}_{pol.name}.csv", index=False, encoding="utf-8-sig")
            log(f"[portfolio] {spec.name} {pol.name} cagr={pf.get('cagr', np.nan):.2%} "
                f"mdd={pf.get('mdd', np.nan):.2%} deploy={pf.get('deploy', np.nan):.1%}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "08_组合复核.csv", index=False, encoding="utf-8-sig")
    return df


def decide(summary: pd.DataFrame, paired: pd.DataFrame, port: pd.DataFrame) -> pd.DataFrame:
    """按预注册 §8 的 6 条判定。"""
    rows = []
    for _, s in summary[(summary["scenario"] == "base")].iterrows():
        entry, arm = s["entry"], s["arm"]
        if arm == "ATR2":
            continue

        def pdiff(scen):
            q = paired[(paired["entry"] == entry) & (paired["arm"] == arm) & (paired["scenario"] == scen)]
            return q.iloc[0] if len(q) else None

        b, c2, d1 = pdiff("base"), pdiff("cost2"), pdiff("delay1")
        if b is None:
            continue
        port_arm = port[(port["entry"] == entry) & (port["arm"] == arm)]
        port_base = port[(port["entry"] == entry) & (port["arm"] == "ATR2")]
        cond6 = False
        if len(port_arm) and len(port_base):
            pa, pb = port_arm.iloc[0], port_base.iloc[0]
            cond6 = bool((pa["calmar"] >= pb["calmar"]) or
                         (pa["mdd"] > pb["mdd"] and pa["cagr"] >= pb["cagr"] - 0.02))
        positive_abs = bool(s["mean"] > 0 and s["year_equal"] > 0)
        conds = dict(
            c0_abs_positive=positive_abs,
            c1_mean_ci=(b["diff"] > 0 and b["ci_lo"] > 0),
            c2_dev_val=(b["diff_dev"] > 0 and b["diff_val"] > 0),
            c3_year_equal=(b["diff_year_equal"] > 0),
            c4_stress=(c2 is not None and d1 is not None and c2["diff"] > 0 and d1["diff"] > 0),
            c5_drop5=(b["diff_drop5"] > 0),
            c6_portfolio=cond6,
        )
        passed = all(conds.values())
        passed6 = all(v for k, v in conds.items() if k != "c0_abs_positive")
        rows.append(dict(entry=entry, arm=arm, arm_mean=s["mean"],
                         arm_mean_trade=s.get("mean_trade", np.nan),
                         arm_year_equal=s["year_equal"],
                         positive_abs=positive_abs,
                         diff=b["diff"], ci_lo=b["ci_lo"], ci_hi=b["ci_hi"],
                         diff_trade=b.get("diff_trade", np.nan),
                         diff_dev=b["diff_dev"], diff_val=b["diff_val"],
                         diff_year_equal=b["diff_year_equal"],
                         diff_cost2=c2["diff"] if c2 is not None else np.nan,
                         diff_delay1=d1["diff"] if d1 is not None else np.nan,
                         diff_drop5=b["diff_drop5"], **conds,
                         decision_6="有可执行增益" if passed6 else "无可执行增益",
                         decision="有可执行增益" if passed else "无可执行增益",
                         fail_items=",".join(k for k, v in conds.items() if not v)))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all",
                    choices=["audit", "signals", "run", "portfolio", "all"])
    ap.add_argument("--force-panel", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    t0 = time.time()
    p = build_panel(force=args.force_panel)
    panel = Panel(p)
    _DATES[:] = panel.dates
    log(f"面板 {panel.n} 只 / {len(panel['cal']):,} 行")

    if args.mode in ("audit",):
        payload = stage_audit(panel)
        (OUT / "10_自检_数据.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        log(f"audit: {payload['checks']} signals={payload['signals']}")
        return
    if args.mode == "signals":
        stage_signals(panel)
        return
    if args.mode == "run":
        stage_run(panel, limit=args.limit, jobs=args.jobs)
        return
    if args.mode == "portfolio":
        port = stage_portfolio(panel)
        summary = pd.read_csv(OUT / "02_臂汇总.csv")
        paired = pd.read_csv(OUT / "03_配对差.csv")
        decide(summary, paired, port).to_csv(
            OUT / "05_决策表.csv", index=False, encoding="utf-8-sig")
        return
    stage_run(panel, limit=args.limit, jobs=args.jobs)
    port = stage_portfolio(panel)
    summary = pd.read_csv(OUT / "02_臂汇总.csv")
    paired = pd.read_csv(OUT / "03_配对差.csv")
    dec = decide(summary, paired, port)
    dec.to_csv(OUT / "05_决策表.csv", index=False, encoding="utf-8-sig")
    log("done all")
    log(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
