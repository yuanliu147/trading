#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""突破回踩 vs 立即追涨（第三阶段，独立实现）

问题
----
20/55/120 日已确认突破之后，**不追涨**，等待有限日内回踩突破位（或 EMA20）且结构未破，
再以「直接买」或「转强买」入场，是否优于突破次日立即买？

纪律
----
* 数据：只读 `01_数据/frozen/frozen.duckdb`（冻结 2026-10-26）。
* 硬截断：任何信号/路径/账户都只到 2020-12-31；2021-01-01 之后不查询、不计算。
* 价格：成交/整股/涨跌停/费用用 raw（float64）；突破位/ATR/EMA 用 A 空间
  `a_px = raw_px × factor / raw_close`（`factor = Π(1+pct_chg/100)`，总收益指数）。
  **不用** 冻结层 `hfq = raw × factor`（日收益平方，见 `12_数据口径_验证.md`）。
* 信号收盘生成、次日开盘成交；T+1、整股、历史费用/印花税/过户费、滑点+参与率、
  停牌/涨跌停/复牌、退市、公司行动、红利税，全部进入账户层。
* 事件层与账户层分离：事件层用同一事件集做「同事件配对」，账户层回答资金占用/错过交易。
* 本文件只写 `02_代码/候选_突破回踩.py` 与 `03_结果/候选_突破回踩/`。

用法
----
    python3 02_代码/候选_突破回踩.py panel      # 面板 + 数据口径自检
    python3 02_代码/候选_突破回踩.py events     # 事件层 + 144 格参数平台
    python3 02_代码/候选_突破回踩.py portfolio  # 账户层（独立引擎）+ 稳健性
    python3 02_代码/候选_突破回踩.py unified    # 统一引擎复核
    python3 02_代码/候选_突破回踩.py selfcheck  # 全量自检
    python3 02_代码/候选_突破回踩.py all        # 全部
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "02_代码"
OUT = ROOT / "03_结果" / "候选_突破回踩"
DB_PATH = ROOT / "01_数据" / "frozen" / "frozen.duckdb"
PREREG = OUT / "00_预注册_突破回踩.md"

sys.path.insert(0, str(CODE_DIR))

ANALYSIS_END = "2020-12-31"
PANEL_START = "2006-01-01"
DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")
FULL = ("2008-01-02", "2020-12-31")
WINDOWS = {"dev": DEV, "val": VAL, "full": FULL}

MAIN_BOARD = "主板"
N_LOT = 100
MIN_ADV20 = 3.0e7
MIN_PRICE = 2.0
MIN_LIST_TD = 120
MIN_LIST_CAL = 250
DELIST_BLOCK_TD = 30

HASH_SEED = 20261028
TRADING_DAYS = 243
PATH = 65                     # 事件后 0..64 个交易日路径
EXT_LIMIT = 2.0               # 超延伸：突破收盘高出突破位 > 2 ATR
FAKE_ATR = 1.0                # 假突破：收盘跌破突破位 1 ATR
STOP_ATR = 1.5                # 结构止损：突破位下方 1.5 ATR
MAX_STOP_PCT = 0.12
MAX_RISK = 0.012

NS = (20, 55, 120)
WS = (3, 5, 10, 20)
TOLS = (0.25, 0.5, 1.0)
TARGETS = ("level", "ema20")
MODES = ("direct", "restrength")
ARMS = ("imm", "pull", "pull_rs", "nopull", "fake")

PRIMARY = dict(N=55, W=10, tol=0.5, target="level", mode="direct", exit="FIX20")
ARM_LABEL = {"imm": "立即突破", "pull": "回踩直接买", "pull_rs": "转强买",
             "nopull": "未回踩", "fake": "假突破"}

GRID = [(N, W, tol, tgt, mode)
        for N in NS for W in WS for tol in TOLS for tgt in TARGETS for mode in MODES]

LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def _ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prereg_sha() -> str:
    txt = PREREG.read_text(encoding="utf-8")
    head = txt.split("<!-- PREREG_END -->")[0]
    return hashlib.sha256(head.encode("utf-8")).hexdigest()


import 费用与交易约束 as FEE   # noqa: E402
import 统计验证 as STAT        # noqa: E402
from 统一数据适配 import EventBook, UnifiedEvent  # noqa: E402


# ---------------------------------------------------------------------------
# 面板：raw（float64）+ A 空间 + 点时宇宙
# ---------------------------------------------------------------------------
@dataclass
class Panel:
    dates: pd.DatetimeIndex
    codes: list[str]
    code_idx: dict[str, int]
    mat: dict[str, np.ndarray]          # raw 及标记
    a: dict[str, np.ndarray]            # A 空间
    atr14: np.ndarray
    ema20: np.ndarray
    ema60: np.ndarray
    rmax: dict[int, np.ndarray]
    adv20: np.ndarray
    listed: np.ndarray
    suspended: np.ndarray
    resume: np.ndarray
    elig: np.ndarray
    ipo_date: list
    delist_date: list
    ca_factor: np.ndarray
    events: EventBook
    n_days: int = 0
    n_codes: int = 0
    meta: dict = field(default_factory=dict)

    def window_index(self, start: str, end: str) -> tuple[int, int]:
        import bisect
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        i0 = int(bisect.bisect_left(self.dates, s))
        i1 = int(bisect.bisect_right(self.dates, e) - 1)
        if i0 > i1:
            raise ValueError(f"窗口 {start}~{end} 无交易日")
        return i0, i1


def load_panel(start: str = PANEL_START, end: str = ANALYSIS_END) -> Panel:
    t0 = time.time()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(con.execute(
            "select date from calendar where is_trading_day=1 and date between ? and ? order by date",
            [start, end]).df()["date"]))
        assert dates.max() <= pd.Timestamp(ANALYSIS_END), "读取越过分析截止日"
        q = """select symbol, date, open, high, low, close, preclose, pct_chg, volume, amount,
                      factor, td_since_list, is_st, is_suspended,
                      is_limit_up_open, is_limit_down_open
               from daily where board = ? and date between ? and ?"""
        df = con.execute(q, [MAIN_BOARD, start, end]).df()
        sy = con.execute("select symbol, ipo_date, delist_date from symbols where board = ?",
                         [MAIN_BOARD]).df()
    finally:
        con.close()

    df["date"] = pd.to_datetime(df["date"])
    codes = sorted(df["symbol"].unique().tolist())
    code_idx = {c: j for j, c in enumerate(codes)}
    n_days, n_codes = len(dates), len(codes)
    if df.duplicated(["date", "symbol"]).any():
        raise ValueError("daily 存在重复 (date,symbol)")

    di = np.searchsorted(dates.values, df["date"].values)
    ci = df["symbol"].map(code_idx).to_numpy(np.int32)

    def mat_of(col, dtype=np.float64):
        m = np.full((n_days, n_codes), np.nan if np.issubdtype(dtype, np.floating) else 0,
                    dtype=dtype)
        v = pd.to_numeric(df[col], errors="coerce").astype("float64")
        if np.issubdtype(dtype, np.floating):
            m[di, ci] = v.to_numpy(dtype=dtype)
        else:
            m[di, ci] = v.fillna(0.0).to_numpy(dtype=dtype)
        return m

    mat = {
        "open": mat_of("open"), "high": mat_of("high"), "low": mat_of("low"),
        "close": mat_of("close"), "preclose": mat_of("preclose"),
        "pct_chg": mat_of("pct_chg"),
        "volume": mat_of("volume"), "amount": mat_of("amount"),
        "factor": mat_of("factor"), "td_since_list": mat_of("td_since_list"),
        "is_st": mat_of("is_st", np.int8), "is_suspended": mat_of("is_suspended", np.int8),
        "is_limit_up_open": mat_of("is_limit_up_open", np.int8),
        "is_limit_down_open": mat_of("is_limit_down_open", np.int8),
    }
    n_rows = len(df)
    del df

    close = mat["close"]
    listed = np.isfinite(close) & (close > 0)
    suspended = (mat["is_suspended"] > 0) | ~listed
    prev_susp = np.vstack([np.ones((1, n_codes), dtype=bool), suspended[:-1]])
    prev_listed = np.vstack([np.zeros((1, n_codes), dtype=bool), listed[:-1]])
    resume = prev_susp | ~prev_listed

    with np.errstate(invalid="ignore", divide="ignore"):
        A = mat["factor"] / close                    # A_t = factor_t / raw_close_t
    a = {}
    for c in ("open", "high", "low", "close"):
        a[c] = mat[c] * A
    a["close_ref"] = mat["factor"].copy()            # a_close 归一后即 factor
    a["A"] = A

    idx = pd.Index(dates, name="date")
    a_close = pd.DataFrame(a["close"], index=idx, columns=codes)
    a_high = pd.DataFrame(a["high"], index=idx, columns=codes)
    amt = pd.DataFrame(mat["amount"], index=idx, columns=codes)
    preclose = pd.DataFrame(mat["preclose"], index=idx, columns=codes)
    prev_a = a_close.shift(1)
    tr = pd.DataFrame(np.maximum.reduce([
        (a_high - a_close).to_numpy(),
        (a_high - prev_a).abs().to_numpy(),
        (a_close - prev_a).abs().to_numpy(),
    ]), index=idx, columns=codes)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy(np.float64)
    ema20 = a_close.ewm(span=20, adjust=False, min_periods=20).mean().to_numpy(np.float64)
    ema60 = a_close.ewm(span=60, adjust=False, min_periods=60).mean().to_numpy(np.float64)
    rmax = {}
    for N in NS:
        rmax[N] = a_high.rolling(N, min_periods=N).max().shift(1).to_numpy(np.float64)
    adv20 = amt.rolling(20, min_periods=20).mean().to_numpy(np.float64)
    del a_close, a_high, amt, preclose, tr, prev_a

    days = np.array([(d.date() - date(1990, 1, 1)).days for d in dates], dtype=np.float64)
    ipo_map = {r.symbol: r.ipo_date for r in sy.itertuples()}
    del_map = {r.symbol: r.delist_date for r in sy.itertuples()}
    ipo_date, delist_date = [], []
    ipo_days = np.full(n_codes, np.nan)
    for j, c in enumerate(codes):
        x = ipo_map.get(c)
        y = del_map.get(c)
        ipo_date.append(None if pd.isna(x) else pd.Timestamp(x).date())
        delist_date.append(None if pd.isna(y) else pd.Timestamp(y).date())
        if ipo_date[-1] is not None:
            ipo_days[j] = (ipo_date[-1] - date(1990, 1, 1)).days
    days_since_ipo = days[:, None] - ipo_days[None, :]

    import bisect
    delist_window = np.zeros_like(listed)
    for j, dd in enumerate(delist_date):
        if dd is None:
            continue
        k = int(bisect.bisect_left(dates, pd.Timestamp(dd)))
        if k < n_days:
            delist_window[max(0, k - DELIST_BLOCK_TD):k + 1, j] = True

    elig = (listed & ~suspended & (mat["is_st"] == 0)
            & (mat["td_since_list"] >= MIN_LIST_TD)
            & (mat["close"] >= MIN_PRICE)
            & (adv20 >= MIN_ADV20)
            & (days_since_ipo >= MIN_LIST_CAL)
            & ~resume & ~delist_window
            & np.isfinite(a["close"]) & np.isfinite(atr14) & (atr14 > 0))

    events = EventBook().load(start, end)
    ev_codes = set(codes)
    events.events = events.events[events.events["symbol"].isin(ev_codes)].reset_index(drop=True)
    events.by_date = {}
    for r in events.events.itertuples(index=False):
        events.by_date.setdefault(pd.Timestamp(r.ex_date).date(), []).append(
            UnifiedEvent(r.symbol, pd.Timestamp(r.ex_date).date(), float(r.prev_close),
                         float(r.preclose), float(r.adj_ratio), float(r.dividend),
                         float(r.bonus_ratio), float(r.rights_ratio), float(r.rights_price),
                         r.status, float(r.ths_dividend), float(r.ths_bonus),
                         float(r.ths_rights_ratio), float(r.ths_rights_price)))
    ratio = events.ratio_frame([d.date() for d in dates], codes)
    ca_factor = np.cumprod(ratio, axis=0)
    del ratio

    P = Panel(dates=dates, codes=codes, code_idx=code_idx, mat=mat, a=a,
              atr14=atr14, ema20=ema20, ema60=ema60, rmax=rmax, adv20=adv20,
              listed=listed, suspended=suspended, resume=resume, elig=elig,
              ipo_date=ipo_date, delist_date=delist_date, ca_factor=ca_factor,
              events=events, n_days=n_days, n_codes=n_codes)
    P.meta = {
        "rows": int(n_rows), "n_days": int(n_days), "n_codes": int(n_codes),
        "days": f"{dates[0].date()}..{dates[-1].date()}",
        "n_listed": int(listed.sum()), "n_susp_rows": int((suspended & listed).sum()),
        "n_elig": int(elig.sum()), "n_ca_events": int(len(events.events)),
        "max_date": str(dates[-1].date()),
    }
    log(f"面板 {n_days} 日 × {n_codes} 只，行 {n_rows:,}，事件 {len(events.events):,}，用时 {time.time()-t0:.1f}s")
    return P


def limit_prices(P: Panel, i: int, j: int):
    return FEE.limit_prices(float(P.mat["preclose"][i, j]), P.codes[j],
                            bool(P.mat["is_st"][i, j] > 0), P.dates[i].date(),
                            int(P.mat["td_since_list"][i, j]))


def slip_bps(base: float, notional: float, adv20: float, vol_yuan: float) -> float:
    part = notional / adv20 if adv20 and adv20 > 0 else 0.0
    bps = base + 0.02 * math.sqrt(max(part, 0.0)) * 10000.0
    return min(100.0 if bps <= 100.0 else 200.0, bps)


def fill_price(px: float, side: str, bps: float) -> float:
    return FEE.apply_slippage(float(px), side, float(bps))


# ---------------------------------------------------------------------------
# 事件层
# ---------------------------------------------------------------------------
def _first_true(mask: np.ndarray, default: int = -1) -> np.ndarray:
    """每行第一个 True 的列号；无 True 返回 default。"""
    any_ = mask.any(axis=1)
    idx = mask.argmax(axis=1)
    return np.where(any_, idx, default)


def breakout_mask(P: Panel, N: int) -> tuple[np.ndarray, np.ndarray]:
    """已确认 N 日突破（收盘确认，只用 b 日及之前）。返回 (mask, first_flag)。"""
    close = P.a["close"]
    rmaxN = P.rmax[N]
    with np.errstate(invalid="ignore"):
        brk = (P.elig & np.isfinite(rmaxN) & (close > rmaxN)
               & np.isfinite(P.atr14) & (P.atr14 > 0) & (P.a["A"] > 0))
    # 数据伪跳保护：突破窗口内不得有 |pct_chg|>30
    extreme = (np.abs(P.mat["pct_chg"]) > 30).astype(np.float64)
    extreme = np.where(np.isfinite(P.mat["pct_chg"]), extreme, np.nan)
    roll_union = pd.DataFrame(extreme, index=P.dates, columns=P.codes) \
        .rolling(N + 1, min_periods=N + 1).sum().to_numpy()
    brk &= (roll_union == 0)
    # 首次/连续突破
    lookback = max(N, 60)
    prior = pd.DataFrame(brk.astype(np.float64), index=P.dates, columns=P.codes) \
        .rolling(lookback, min_periods=1).max().shift(1).fillna(0).to_numpy()
    return brk, prior == 0


def build_events(P: Panel, N: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (事件明细, 平台汇总行)。"""
    t0 = time.time()
    close = P.a["close"]
    rmaxN = P.rmax[N]
    brk, first_flag = breakout_mask(P, N)

    # 面板末尾事件保留（只要求能容纳最长入场偏移 22 日）；路径越界处填 NaN
    ii, jj = np.nonzero(brk & (np.arange(P.n_days)[:, None] + 23 < P.n_days))
    n = len(ii)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()
    offs = np.arange(PATH)
    gi = ii[:, None] + offs[None, :]
    gj = np.repeat(jj[:, None], PATH, axis=1)
    inb = gi < P.n_days
    gc = np.minimum(gi, P.n_days - 1)
    def _path(mat, dtype=np.float32):
        out = mat[gc, gj].astype(dtype)
        out[~inb] = np.nan
        return out
    a_o = _path(P.a["open"])
    a_c = _path(P.a["close"])
    a_h = _path(P.a["high"])
    a_l = _path(P.a["low"])
    ema = _path(P.ema20)
    trad = (P.listed[gc, gj] & ~P.suspended[gc, gj]
            & (P.mat["volume"][gc, gj] > 0) & inb)

    row = np.arange(n)
    L = rmaxN[ii, jj]
    atr = P.atr14[ii, jj]
    ext = (close[ii, jj] - L) / atr
    # 入场可执行性（entry+1 开盘）：停牌/复牌/开盘涨停
    def entry_ok_at(entry_off: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = np.clip(entry_off, 1, PATH - 1)
        ei = np.minimum(ii + e, P.n_days - 1)
        ok = ((entry_off > 0) & (ii + e < P.n_days) & trad[row, e]
              & ~P.resume[ei, jj]
              & (P.mat["is_limit_up_open"][ei, jj] == 0))
        fwd = np.full(n, np.nan)
        room = (e + 20) < PATH
        fwd[room] = np.where(ok[room],
                             a_c[row[room], (e + 20)[room]] / a_o[row[room], e[room]] - 1.0,
                             np.nan)
        return ok, fwd

    # 立即臂
    ok_imm, fwd20_imm = entry_ok_at(np.ones(n, dtype=np.int32))

    def fwd_at(e: np.ndarray, h: int) -> np.ndarray:
        out = np.full(n, np.nan)
        good = (e > 0) & (e + h < PATH)
        out[good] = a_c[row[good], (e + h)[good]] / a_o[row[good], e[good]] - 1.0
        return out
    fwd5_imm, fwd10_imm, fwd40_imm = (fwd_at(np.ones(n, dtype=np.int32), h) for h in (5, 10, 40))

    # 失效（假突破）
    invalid = (a_c[:, 1:] < (L - FAKE_ATR * atr)[:, None]) & trad[:, 1:]
    det = {
        "symbol": np.array(P.codes, dtype=object)[jj],
        "code_idx": jj.astype(np.int32),
        "b_idx": ii.astype(np.int32),
        "N": np.full(n, N, dtype=np.int16),
        "date_b": P.dates[ii].values,
        "L": L.astype(np.float64),
        "atr": atr.astype(np.float64),
        "ext": ext.astype(np.float32),
        "first": first_flag[ii, jj],
        "ok_imm": ok_imm,
        "fwd20_imm": fwd20_imm.astype(np.float32),
        "fwd5_imm": fwd5_imm.astype(np.float32),
        "fwd10_imm": fwd10_imm.astype(np.float32),
        "fwd40_imm": fwd40_imm.astype(np.float32),
    }
    rows: list[dict] = []
    for W in WS:
        inv_w = _first_true(invalid[:, :W])
        inv_off = np.where(inv_w >= 0, inv_w + 1, 10 ** 6)
        det[f"inv_off_W{W}"] = np.where(inv_w >= 0, inv_w + 1, 255).astype(np.int16)
        for tol in TOLS:
            lo = (L - tol * atr)[:, None]
            hi = (L + tol * atr)[:, None]
            struct_ok = (a_c[:, 1:W + 1] >= lo) & (a_c[:, 1:W + 1] >= ema[:, 1:W + 1]) \
                & trad[:, 1:W + 1]
            for tgt in TARGETS:
                z = L[:, None] if tgt == "level" else ema[:, 1:W + 1]
                touch = (a_l[:, 1:W + 1] <= z + tol * atr[:, None]) & struct_ok
                ar = np.arange(1, W + 1)[None, :]
                touch &= ar < inv_off[:, None]
                kcol = _first_true(touch)
                k_off = np.where(kcol >= 0, kcol + 1, 255)
                # 转强确认：k 之后、W 之前，收盘 > max(high[k-1], high[k]) 且 > 前一日收盘
                kk = np.where(k_off < 255, k_off, 1).astype(np.int32)
                thr = np.maximum(a_h[row, np.maximum(kk - 1, 0)], a_h[row, kk])
                prev_c = np.concatenate([np.full((n, 1), np.nan, dtype=np.float32), a_c[:, :-1]], axis=1)
                conf = ((a_c[:, 1:W + 1] > thr[:, None])
                        & (a_c[:, 1:W + 1] > prev_c[:, 1:W + 1])
                        & trad[:, 1:W + 1]
                        & (ar > k_off[:, None]) & (ar < inv_off[:, None])
                        & (k_off < 255)[:, None])
                mcol = _first_true(conf)
                m_off = np.where(mcol >= 0, mcol + 1, 255)
                for mode in MODES:
                    base_off = k_off if mode == "direct" else m_off
                    entry_off = np.where(base_off < 255, base_off + 1, 0).astype(np.int32)
                    ok, fwd20 = entry_ok_at(entry_off)
                    key = f"{W}_{tol}_{tgt}_{mode}"
                    det[f"e_{key}"] = entry_off.astype(np.int8 if entry_off.max() < 127 else np.int16)
                    det[f"ok_{key}"] = ok
                    det[f"fwd20_{key}"] = fwd20.astype(np.float32)
                    if (W, tol, tgt, mode) == (PRIMARY["W"], PRIMARY["tol"],
                                               PRIMARY["target"], PRIMARY["mode"]):
                        det["fwd5_pull"] = fwd_at(entry_off, 5).astype(np.float32)
                        det["fwd10_pull"] = fwd_at(entry_off, 10).astype(np.float32)
                        det["fwd40_pull"] = fwd_at(entry_off, 40).astype(np.float32)
                        # MAE / MFE（A 空间，入场日起 20 日）
                        mae = np.full(n, np.nan)
                        mfe = np.full(n, np.nan)
                        for h in range(1, 21):
                            e = entry_off
                            good = (e > 0) & (e + h < PATH)
                            lo_h = a_l[row[good], (e + h)[good]]
                            hi_h = a_h[row[good], (e + h)[good]]
                            base = a_o[row[good], e[good]]
                            cur = mae[good]
                            cur = np.where(np.isnan(cur), lo_h / base - 1.0, np.minimum(cur, lo_h / base - 1.0))
                            mae[good] = cur
                            cur = mfe[good]
                            cur = np.where(np.isnan(cur), hi_h / base - 1.0, np.maximum(cur, hi_h / base - 1.0))
                            mfe[good] = cur
                        det["mae20_pull"] = mae.astype(np.float32)
                        det["mfe20_pull"] = mfe.astype(np.float32)
                    rows.append(dict(N=N, W=W, tol=tol, target=tgt, mode=mode,
                                     n=n, n_touch=int((k_off < 255).sum()),
                                     n_confirm=int((m_off < 255).sum()),
                                     n_entry_ok=int(ok.sum()),
                                     n_ext=int((ext > EXT_LIMIT).sum())))
    detail = pd.DataFrame(det)
    detail["date_b"] = pd.to_datetime(detail["date_b"])
    detail["ym"] = detail["date_b"].dt.to_period("M").astype(str)
    meta = pd.DataFrame(rows)
    log(f"N={N}: 突破事件 {n:,}，用时 {time.time()-t0:.1f}s")
    return detail, meta


def _window_mask(det: pd.DataFrame, window: str) -> np.ndarray:
    d = det["date_b"]
    if window == "dev":
        return ((d >= pd.Timestamp(DEV[0])) & (d <= pd.Timestamp(DEV[1]))).to_numpy()
    if window == "val":
        return ((d >= pd.Timestamp(VAL[0])) & (d <= pd.Timestamp(VAL[1]))).to_numpy()
    return np.ones(len(det), dtype=bool)


def clustered_t(x: np.ndarray, ym: np.ndarray) -> tuple[float, float, int]:
    """按日历月聚类的均值 t：月内取均值，再对月份做单样本 t。"""
    s = pd.Series(np.asarray(x, dtype=float))
    g = s.groupby(pd.Series(ym)).mean().dropna()
    k = len(g)
    if k < 3:
        return float("nan"), float("nan"), k
    m = float(g.mean())
    sd = float(g.std(ddof=1))
    t = m / (sd / math.sqrt(k)) if sd > 0 else float("nan")
    return m, t, k


def bench_h(P: Panel, h: int) -> np.ndarray:
    """U(t) 等权、开-开 h 日收益（A 空间），按日索引。"""
    out = np.full(P.n_days, np.nan)
    a_o, a_c = P.a["open"], P.a["close"]
    for i in range(P.n_days - h):
        cols = np.flatnonzero(P.elig[i])
        if not len(cols):
            continue
        base = a_o[i, cols]
        end = a_c[i + h, cols]
        ok = np.isfinite(base) & np.isfinite(end) & (base > 0)
        if ok.sum() >= 30:
            out[i] = float(np.mean(end[ok] / base[ok] - 1.0))
    return out


# ---------------------------------------------------------------------------
# 事件层统计
# ---------------------------------------------------------------------------
def _arm_vectors(det: pd.DataFrame, W: int, tol: float, tgt: str, mode: str,
                 bench20: np.ndarray) -> dict[str, dict]:
    """返回各臂的 (可执行掩码, 收益, 入场日索引)。"""
    key = f"{W}_{tol}_{tgt}_{mode}"
    b = det["b_idx"].to_numpy()
    ok_imm = det["ok_imm"].to_numpy(bool)
    fwd_imm = det["fwd20_imm"].to_numpy(np.float64)
    e_pull = det[f"e_{key}"].to_numpy(np.int32)
    ok_pull = det[f"ok_{key}"].to_numpy(bool)
    fwd_pull = det[f"fwd20_{key}"].to_numpy(np.float64)
    inv = det[f"inv_off_W{W}"].to_numpy(np.int32)
    touched = e_pull > 0
    fake = (inv < 255) & ~touched
    nopull = (inv == 255) & ~touched
    e_rs_key = f"e_{W}_{tol}_{tgt}_restrength"
    e_rs = det[e_rs_key].to_numpy(np.int32)
    ok_rs = det[f"ok_{W}_{tol}_{tgt}_restrength"].to_numpy(bool)
    fwd_rs = det[f"fwd20_{W}_{tol}_{tgt}_restrength"].to_numpy(np.float64)

    def pack(off, ok, fwd, label):
        ent = np.where(off > 0, b + off, -1)
        good = ok & np.isfinite(fwd) & (ent >= 0)
        ex = np.full(len(det), np.nan)
        ex[good] = fwd[good] - bench20[ent[good]]
        return dict(label=label, ent=ent, ok=ok, fwd=fwd, exc=ex, good=good)

    return {
        "imm": pack(np.ones(len(det), dtype=np.int32), ok_imm, fwd_imm, "立即突破"),
        "pull": pack(e_pull, ok_pull, fwd_pull, "回踩直接买"),
        "pull_rs": pack(e_rs, ok_rs, fwd_rs, "回踩转强买"),
        "nopull": pack(np.ones(len(det), dtype=np.int32), ok_imm & nopull, fwd_imm, "未回踩(立即买)"),
        "fake": pack(np.ones(len(det), dtype=np.int32), ok_imm & fake, fwd_imm, "假突破(立即买)"),
    }, {"touched": touched, "fake": fake, "nopull": nopull}


def summarize_events(det: pd.DataFrame, N: int, W: int, tol: float, tgt: str, mode: str,
                     bench20: np.ndarray) -> list[dict]:
    arms, status = _arm_vectors(det, W, tol, tgt, mode, bench20)
    out: list[dict] = []
    for wname in ("dev", "val", "full"):
        wm = _window_mask(det, wname)
        ym = det["ym"].to_numpy()
        for arm, d in arms.items():
            m = wm & d["good"]
            f = d["fwd"][m]
            x = d["exc"][m]
            n = int(m.sum())
            if n:
                mm, tt, k = clustered_t(f, ym[m])
                mx, tx, _ = clustered_t(x, ym[m])
            else:
                mm = tt = mx = tx = float("nan"); k = 0
            out.append(dict(N=N, W=W, tol=tol, target=tgt, mode=mode, window=wname,
                            arm=arm, n=n,
                            n_symbols=int(det.loc[m, "symbol"].nunique()) if n else 0,
                            mean=float(np.mean(f)) if n else float("nan"),
                            median=float(np.median(f)) if n else float("nan"),
                            win=float((f > 0).mean()) if n else float("nan"),
                            t_month=tt, mean_month=mm, months=k,
                            exc_mean=float(np.mean(x)) if n else float("nan"),
                            exc_t_month=tx,
                            _arr=f, _exc=x, _ym=ym[m]))
        a, b = arms["pull"], arms["imm"]
        m = wm & a["good"] & b["good"]
        n = int(m.sum())
        if n:
            diff = a["fwd"][m] - b["fwd"][m]
            mm, tt, k = clustered_t(diff, ym[m])
            out.append(dict(N=N, W=W, tol=tol, target=tgt, mode=mode, window=wname,
                            arm="paired_pull_minus_imm", n=n,
                            n_symbols=int(det.loc[m, "symbol"].nunique()),
                            mean=float(np.mean(diff)), median=float(np.median(diff)),
                            win=float((diff > 0).mean()), t_month=tt, mean_month=mm,
                            months=k, exc_mean=float("nan"), exc_t_month=float("nan"),
                            _arr=diff, _exc=np.array([]), _ym=ym[m]))
        st = status
        out.append(dict(N=N, W=W, tol=tol, target=tgt, mode=mode, window=wname,
                        arm="status_count", n=int((wm & st["touched"]).sum()),
                        n_symbols=int(det.loc[wm, "symbol"].nunique()),
                        mean=float((wm & st["touched"]).mean()),
                        median=float("nan"), win=float("nan"), t_month=float("nan"),
                        mean_month=float("nan"), months=0, exc_mean=float("nan"),
                        exc_t_month=float("nan"), _arr=np.array([]), _exc=np.array([]),
                        _ym=np.array([])))
        out[-1]["n_nopull"] = int((wm & st["nopull"]).sum())
        out[-1]["n_fake"] = int((wm & st["fake"]).sum())
        out[-1]["n_ext"] = int((wm & (det["ext"].to_numpy() > EXT_LIMIT)).sum())
    return out


def first_split_rows(det: pd.DataFrame, N: int, tol: float = 0.5, target: str = "level",
                     mode: str = "direct") -> list[dict]:
    """首次 vs 连续突破（主口径 tol/target/mode），在 keep 列裁剪之前计算。"""
    key = f"{PRIMARY['W']}_{tol}_{target}_{mode}"
    e = det[f"e_{key}"].to_numpy(np.int32)
    ok_p = det[f"ok_{key}"].to_numpy(bool)
    ok_i = det["ok_imm"].to_numpy(bool)
    f_i = det["fwd20_imm"].to_numpy(np.float64)
    f_p = det[f"fwd20_{key}"].to_numpy(np.float64)
    first = det["first"].to_numpy(bool)
    ym_all = det["ym"].to_numpy()
    rows = []
    for wname in ("dev", "val", "full"):
        wm = _window_mask(det, wname)
        for flag in (True, False):
            m = wm & (first == flag)
            mi = m & ok_i & np.isfinite(f_i)
            mp = m & ok_p & np.isfinite(f_p)
            pair = m & ok_i & ok_p & np.isfinite(f_i) & np.isfinite(f_p)
            diff = f_p[pair] - f_i[pair]
            _mm, tt, _k = clustered_t(diff, ym_all[pair]) if len(diff) else (np.nan, np.nan, 0)
            rows.append(dict(N=N, window=wname, first=bool(flag), n_imm=int(mi.sum()),
                             n_pull=int(mp.sum()),
                             imm_mean=float(np.mean(f_i[mi])) if mi.any() else np.nan,
                             pull_mean=float(np.mean(f_p[mp])) if mp.any() else np.nan,
                             paired_mean=float(np.mean(diff)) if len(diff) else np.nan,
                             paired_t=tt, paired_n=int(len(diff))))
    return rows


def cmd_events(P: Panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    bench = {h: bench_h(P, h) for h in (5, 10, 20, 40)}
    details, metas, sums, fs_rows = [], [], [], []
    for N in NS:
        det, meta = build_events(P, N)
        if det.empty:
            continue
        meta_b = pd.DataFrame({"N": [N], "n_events": [len(det)],
                               "n_first": [int(det["first"].sum())],
                               "n_ext": [int((det["ext"] > EXT_LIMIT).sum())],
                               "n_entry_ok_imm": [int(det["ok_imm"].sum())],
                               "mean_fwd20_imm": [float(np.nanmean(det["fwd20_imm"]))],
                               "median_fwd20_imm": [float(np.nanmedian(det["fwd20_imm"]))]})
        metas.append(meta_b)
        fs_rows.extend(first_split_rows(det, N))
        for (n_, W, tol, tgt, mode) in GRID:
            if n_ != N:
                continue
            sums.extend(summarize_events(det, N, W, tol, tgt, mode, bench[20]))
        keep = ["symbol", "code_idx", "b_idx", "N", "date_b", "ym", "L", "atr", "ext",
                "first", "ok_imm", "fwd20_imm", "fwd5_imm", "fwd10_imm", "fwd40_imm",
                "fwd5_pull", "fwd10_pull", "fwd40_pull", "mae20_pull", "mfe20_pull"]
        keep += [c for c in det.columns if c.startswith(("e_", "ok_", "inv_off_W"))]
        keep = list(dict.fromkeys(keep))
        details.append(det[keep])
    det_all = pd.concat(details, ignore_index=True)
    det_all.to_parquet(OUT / "01_事件_明细.parquet", index=False)
    pd.DataFrame(fs_rows).to_csv(OUT / "16_首次vs连续.csv", index=False)
    meta_all = pd.concat(metas, ignore_index=True)
    meta_all.to_csv(OUT / "01b_事件计数.csv", index=False)

    summ = pd.DataFrame(sums)
    prim = summ[(summ["N"] == PRIMARY["N"]) & (summ["W"] == PRIMARY["W"])
                & (summ["tol"] == PRIMARY["tol"]) & (summ["target"] == PRIMARY["target"])
                & (summ["mode"] == PRIMARY["mode"])]
    for wname in ("dev", "val", "full"):
        sub = prim[prim["window"] == wname]
        for a in ARMS + ("paired_pull_minus_imm",):
            s = sub[sub["arm"] == a]
            if not len(s):
                continue
            log(f"  主配置[{wname}] {a}: n={int(s['n'].iloc[0]):,} "
                f"mean={s['mean'].iloc[0]:+.4%} t={s['t_month'].iloc[0]:+.2f}")
    out_cols = [c for c in summ.columns if not c.startswith("_")]
    summ[out_cols].to_csv(OUT / "02_事件_汇总.csv", index=False)
    log(f"事件汇总 {len(summ):,} 行，明细 {len(det_all):,} 行")
    return det_all, summ


def platform_table(summ: pd.DataFrame) -> pd.DataFrame:
    """144 格平台：主口径行（配对差 + 各臂均值）。"""
    rows = []
    for (N, W, tol, tgt, mode), g in summ.groupby(["N", "W", "tol", "target", "mode"]):
        rec = dict(N=N, W=W, tol=tol, target=tgt, mode=mode)
        for wname in ("dev", "val"):
            for arm in ("imm", "pull", "pull_rs", "nopull", "fake"):
                s = g[(g["window"] == wname) & (g["arm"] == arm)]
                rec[f"{wname}_{arm}_mean"] = float(s["mean"].iloc[0]) if len(s) else np.nan
                rec[f"{wname}_{arm}_n"] = int(s["n"].iloc[0]) if len(s) else 0
            s = g[(g["window"] == wname) & (g["arm"] == "paired_pull_minus_imm")]
            rec[f"{wname}_paired_mean"] = float(s["mean"].iloc[0]) if len(s) else np.nan
            rec[f"{wname}_paired_t"] = float(s["t_month"].iloc[0]) if len(s) else np.nan
            rec[f"{wname}_paired_win"] = float(s["win"].iloc[0]) if len(s) else np.nan
            rec[f"{wname}_paired_n"] = int(s["n"].iloc[0]) if len(s) else 0
            def _g(arm, col="mean"):
                ss = g[(g["window"] == wname) & (g["arm"] == arm)]
                return float(ss[col].iloc[0]) if len(ss) else np.nan
            n_imm, n_pu, n_no, n_fa = (_g("imm", "n"), _g("pull", "n"),
                                       _g("nopull", "n"), _g("fake", "n"))
            if all(np.isfinite([n_imm, n_pu, n_no, n_fa])) and n_imm > 0:
                rec[f"{wname}_policy_gain"] = (
                    n_pu * rec[f"{wname}_paired_mean"]
                    - n_no * _g("nopull") - n_fa * _g("fake")) / n_imm
        s = g[(g["window"] == "full") & (g["arm"] == "status_count")]
        if len(s):
            rec["n_events"] = int(s["n"].iloc[0])
            rec["n_nopull"] = int(s["n_nopull"].iloc[0])
            rec["n_fake"] = int(s["n_fake"].iloc[0])
            rec["n_ext"] = int(s["n_ext"].iloc[0])
        rec["dev_ok"] = bool((rec.get("dev_paired_mean", -1) > 0)
                             and (rec.get("dev_paired_t", 0) >= 2.0)
                             and (rec.get("dev_paired_n", 0) >= 30))
        rec["val_ok"] = bool(rec.get("val_paired_mean", -1) > 0)
        rows.append(rec)
    plat = pd.DataFrame(rows).sort_values(["N", "W", "tol", "target", "mode"])
    plat.to_csv(OUT / "03_平台_参数.csv", index=False)
    return plat


def decomposition(det_all: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """imm = pull + nopull + fake 的收益分解（主口径 tol/target/mode）。"""
    rows = []
    yms = det_all["ym"].to_numpy()
    for N in NS:
        det = det_all[det_all["N"] == N].reset_index(drop=True)
        for W in WS:
            sub = summ[(summ["N"] == N) & (summ["W"] == W) & (summ["tol"] == 0.5)
                       & (summ["target"] == "level") & (summ["mode"] == "direct")]
            for wname in ("dev", "val", "full"):
                for arm in ("imm", "pull", "nopull", "fake"):
                    s = sub[(sub["window"] == wname) & (sub["arm"] == arm)]
                    if not len(s):
                        continue
                    rows.append(dict(N=N, W=W, window=wname, arm=arm, n=int(s["n"].iloc[0]),
                                     mean=float(s["mean"].iloc[0]),
                                     median=float(s["median"].iloc[0]),
                                     win=float(s["win"].iloc[0]),
                                     exc_mean=float(s["exc_mean"].iloc[0]),
                                     t_month=float(s["t_month"].iloc[0])))
                # 加权恒等
                parts = sub[(sub["window"] == wname)
                            & (sub["arm"].isin(["pull", "nopull", "fake"]))]
                if len(parts) == 3 and parts["n"].sum() > 0:
                    wmean = float((parts["mean"] * parts["n"]).sum() / parts["n"].sum())
                    imm = sub[(sub["window"] == wname) & (sub["arm"] == "imm")]
                    rows.append(dict(N=N, W=W, window=wname, arm="decomp_sum", n=int(parts["n"].sum()),
                                     mean=wmean, median=np.nan, win=np.nan, exc_mean=np.nan,
                                     t_month=np.nan))
                    if len(imm):
                        rows.append(dict(N=N, W=W, window=wname, arm="unexplained", n=0,
                                         mean=float(imm["mean"].iloc[0]) - wmean,
                                         median=np.nan, win=np.nan, exc_mean=np.nan,
                                         t_month=np.nan))
                    pair = sub[(sub["window"] == wname)
                               & (sub["arm"] == "paired_pull_minus_imm")]
                    if len(pair) and len(imm):
                        n_pu = float(sub[(sub["window"] == wname) & (sub["arm"] == "pull")]["n"].iloc[0])
                        n_imm = float(imm["n"].iloc[0])
                        gain = (n_pu * float(pair["mean"].iloc[0])
                                - float(parts[parts["arm"] == "nopull"]["n"].iloc[0])
                                * float(parts[parts["arm"] == "nopull"]["mean"].iloc[0])
                                - float(parts[parts["arm"] == "fake"]["n"].iloc[0])
                                * float(parts[parts["arm"] == "fake"]["mean"].iloc[0]))
                        rows.append(dict(N=N, W=W, window=wname, arm="policy_value", n=int(n_imm),
                                         mean=gain / n_imm if n_imm else np.nan,
                                         median=np.nan, win=np.nan, exc_mean=np.nan, t_month=np.nan))
        # 超延伸子样本
        for wname in ("dev", "val", "full"):
            wm = _window_mask(det, wname)
            ext = det["ext"].to_numpy() > EXT_LIMIT
            for label, m in (("ext_events", wm & ext), ("nonext_events", wm & ~ext)):
                f = det["fwd20_imm"].to_numpy(np.float64)[m & det["ok_imm"].to_numpy(bool)]
                n = int(np.isfinite(f).sum())
                rows.append(dict(N=N, W=0, window=wname, arm=label, n=n,
                                 mean=float(np.nanmean(f)) if n else np.nan,
                                 median=float(np.nanmedian(f)) if n else np.nan,
                                 win=float(np.nanmean(f > 0)) if n else np.nan,
                                 exc_mean=np.nan, t_month=np.nan))
    dec = pd.DataFrame(rows)
    dec.to_csv(OUT / "04_对照_分解.csv", index=False)
    return dec


# ---------------------------------------------------------------------------
# 账户层（独立引擎）
# ---------------------------------------------------------------------------
def make_fee(cost_mult: float = 1.0) -> FEE.FeeConfig:
    return FEE.FeeConfig(
        commission_rate=0.00025 * cost_mult,
        min_commission=5.0,
        stamp_schedule=FEE.DEFAULT_STAMP_SCHEDULE,
        transfer_schedule=(
            FEE.TransferRule(date(2005, 1, 1), "par_sh", 0.0010 * cost_mult, "SH"),
            FEE.TransferRule(date(2015, 8, 1), "amount", 0.00002 * cost_mult, "BOTH"),
            FEE.TransferRule(date(2022, 4, 29), "amount", 0.00001 * cost_mult, "BOTH"),
        ),
        slippage_bps=10.0 * cost_mult,
        max_participation=0.02,
        dividend_tax_rate=0.0,
    )


@dataclass
class PortCfg:
    arm: str = "pull"
    N: int = 55
    W: int = 10
    tol: float = 0.5
    target: str = "level"
    mode: str = "direct"
    exit: str = "FIX20"
    window: str = "dev"
    capital: float = 1_000_000.0
    slots: int = 10
    rank: str = "hash"
    cost_mult: float = 1.0
    delay: int = 0
    delist_haircut: float = 0.0
    label: str = ""

    def key(self) -> str:
        base = f"{self.arm}|N{self.N}|W{self.W}|tol{self.tol}|{self.target}|{self.mode}|{self.exit}"
        extra = f"|cap{int(self.capital)}|slots{self.slots}|{self.rank}|c{self.cost_mult}|d{self.delay}"
        return base + extra


@dataclass
class Pos:
    j: int
    symbol: str
    shares: int = 0
    entry_i: int = 0
    entry_px: float = 0.0
    entry_raw: float = 0.0
    cost: float = 0.0            # 含买入费用
    entry_fees: float = 0.0
    stop_a: float = float("nan")
    div_gross: float = 0.0
    ca_credit: float = 0.0
    tax: float = 0.0
    pending: bool = False
    pend_i: int = 0
    pend_age: int = 0
    last_px: float = 0.0
    last_i: int = 0
    b_idx: int = 0


def build_schedule(P: Panel, det: pd.DataFrame, cfg: PortCfg) -> list[dict]:
    """把事件表压成「入场日 → 候选」列表；同 (证券, 入场日) 只保留最早突破。"""
    key = f"{cfg.W}_{cfg.tol}_{cfg.target}_{cfg.mode}"
    d = det[det["N"] == cfg.N]
    b = d["b_idx"].to_numpy(np.int32)
    ci = d["code_idx"].to_numpy(np.int32)
    L = d["L"].to_numpy(np.float64)
    atr = d["atr"].to_numpy(np.float64)
    sym = d["symbol"].to_numpy(object)
    if cfg.arm == "imm":
        off = np.ones(len(d), dtype=np.int32)
        ok = d["ok_imm"].to_numpy(bool)
    elif cfg.arm in ("pull", "pull_rs"):
        off = d[f"e_{key}"].to_numpy(np.int32)
        ok = d[f"ok_{key}"].to_numpy(bool)
    elif cfg.arm in ("nopull", "fake"):
        off = np.ones(len(d), dtype=np.int32)
        e0 = d[f"e_{cfg.W}_{cfg.tol}_{cfg.target}_direct"].to_numpy(np.int32)
        inv = d[f"inv_off_W{cfg.W}"].to_numpy(np.int32)
        touched = e0 > 0
        if cfg.arm == "nopull":
            ok = d["ok_imm"].to_numpy(bool) & (inv == 255) & ~touched
        else:
            ok = d["ok_imm"].to_numpy(bool) & (inv < 255) & ~touched
    else:
        raise ValueError(cfg.arm)
    off = off + cfg.delay
    ent = np.where(off > 0, b + off, -1)
    good = ok & (ent >= 0) & (ent < P.n_days)
    idx = np.flatnonzero(good)
    if not len(idx):
        return []
    ent_i = ent[idx].astype(np.int64)
    jj = ci[idx]
    bb = b[idx].astype(np.int64)
    order = np.lexsort((bb, ent_i))
    ent_i, jj, bb = ent_i[order], jj[order], bb[order]
    keys = ent_i * (P.n_codes + 1) + jj
    _, first = np.unique(keys, return_index=True)
    ent_i, jj, bb = ent_i[first], jj[first], bb[first]
    syms = sym[idx][order][first]
    Ls, atrs = L[idx][order][first], atr[idx][order][first]
    out = [dict(entry_i=int(ent_i[k]), j=int(jj[k]), symbol=str(syms[k]),
                L=float(Ls[k]), atr=float(atrs[k]), b_idx=int(bb[k]))
           for k in range(len(ent_i))]
    return out


_RANK_CACHE: dict[tuple[int, int], int] = {}


def _hash_rank(P: Panel, i: int, j: int) -> int:
    k = (i, j)
    v = _RANK_CACHE.get(k)
    if v is None:
        v = int.from_bytes(hashlib.blake2b(
            f"{HASH_SEED}|{P.dates[i].date()}|{P.codes[j]}".encode(), digest_size=8).digest(), "big")
        if len(_RANK_CACHE) < 3_000_000:
            _RANK_CACHE[k] = v
    return v


def _rank_key(P: Panel, cfg: PortCfg, ev: dict) -> tuple:
    i, j = ev["entry_i"], ev["j"]
    if cfg.rank == "adv":
        return (-float(np.nan_to_num(P.adv20[i, j], nan=0.0)), P.codes[j])
    if cfg.rank == "strength":
        with np.errstate(invalid="ignore"):
            s = (float(P.a["close"][ev["b_idx"], j]) - ev["L"]) / ev["atr"]
        return (-float(np.nan_to_num(s, nan=-1e9)), P.codes[j])
    return (_hash_rank(P, i, j), P.codes[j])


def _sell(P: Panel, cfg: PortCfg, fcfg, p: Pos, i: int, price: float, reason: str,
          res: dict) -> tuple[float, float]:
    """返回 (净现金流入, 成交金额)。"""
    d = P.dates[i].date()
    j = p.j
    bps = min(200.0, fcfg.slippage_bps * 2.0) if P.resume[i, j] else fcfg.slippage_bps
    adv = float(np.nan_to_num(P.adv20[i - 1, j], nan=0.0)) if i > 0 else 0.0
    vol = float(np.nan_to_num(P.mat["volume"][i, j], nan=0.0))
    bps = slip_bps(bps, price * p.shares, adv, price * vol)
    px = fill_price(price, "sell", bps)
    lim = limit_prices(P, i, j)
    if not lim.no_limit:
        px = max(px, lim.down)
    qty = int(p.shares)
    fees = FEE.trade_fees(d, p.symbol, "sell", px, qty, fcfg)
    gross = px * qty
    hold_days = (d - P.dates[p.entry_i].date()).days
    rate = 0.20 if hold_days <= 30 else (0.10 if hold_days <= 365 else 0.0)
    tax = p.div_gross * rate
    proceeds = gross - fees.total - tax
    pnl = proceeds - p.cost + p.ca_credit
    res["trades"].append(dict(
        symbol=p.symbol, entry_date=str(P.dates[p.entry_i].date()), exit_date=str(d),
        entry_px=round(p.entry_px, 4), exit_px=round(px, 4), shares=qty,
        entry_fees=round(p.entry_fees, 2), exit_fees=round(fees.total, 2),
        dividends=round(p.div_gross, 2), tax=round(tax, 2), ca_credit=round(p.ca_credit, 2),
        pnl=round(pnl, 2), ret=float(pnl / p.cost) if p.cost > 0 else np.nan,
        hold_days=int(i - p.entry_i), reason=reason, is_mark=False,
        accounting="shares", window=cfg.window))
    res["fees"] += fees.total
    res["n_sells"] += 1
    res["hold_days"].append(i - p.entry_i)
    res["slip_cost"] += abs(px - price) * qty
    return proceeds, gross


def simulate(P: Panel, det: pd.DataFrame, cfg: PortCfg) -> dict:
    i0, i1 = P.window_index(*WINDOWS[cfg.window])
    fcfg = make_fee(cfg.cost_mult)
    sched = build_schedule(P, det, cfg)
    by_entry: dict[int, list[dict]] = {}
    for ev in sched:
        by_entry.setdefault(ev["entry_i"], []).append(ev)
    for k in by_entry:
        by_entry[k].sort(key=lambda e: _rank_key(P, cfg, e))

    cash = float(cfg.capital)
    pos: dict[int, Pos] = {}
    res = {"trades": [], "fees": 0.0, "n_sells": 0, "hold_days": [], "failed": [],
           "slip_cost": 0.0, "daily": [], "forced": 0, "cash_resid_max": 0.0,
           "cash_resid_bad": 0}
    hold = 20 if cfg.exit == "FIX20" else 40

    for i in range(i0, i1 + 1):
        d = P.dates[i].date()
        cash_prev = cash
        flow = 0.0
        day_turnover = 0.0
        day_cost = 0.0
        # 0) 退市强制
        for j, p in list(pos.items()):
            dd = P.delist_date[j]
            if dd is not None and dd <= d:
                px = p.last_px if p.last_px > 0 else float(P.mat["close"][i, j])
                if np.isfinite(px) and px > 0:
                    if cfg.delist_haircut > 0:
                        px *= (1.0 - cfg.delist_haircut)
                    pr, gross = _sell(P, cfg, fcfg, p, i, px, "delisting_out_date", res)
                    cash += pr
                    flow += pr
                    day_turnover += gross
                    res["forced"] += 1
                    del pos[j]
        # 1) 公司行动（开盘前入账）
        for ev in P.events.on(d):
            j = P.code_idx.get(ev.symbol)
            if j is None or j not in pos:
                continue
            p = pos[j]
            s0 = p.shares
            if s0 <= 0:
                continue
            if ev.status == "verified_ths":
                bonus_sh = int(math.floor(s0 * ev.bonus_ratio + 1e-9))
                rights_sh = int(math.floor(s0 * ev.rights_ratio + 1e-9))
                credit = s0 * ev.prev_close - (s0 + bonus_sh + rights_sh) * ev.preclose
                if rights_sh > 0 and cash + credit < -1e-6:
                    rights_sh = 0
                    credit = s0 * ev.prev_close - (s0 + bonus_sh) * ev.preclose
                p.div_gross += s0 * max(0.0, ev.dividend)
            else:
                bonus_sh = rights_sh = 0
                credit = s0 * ev.prev_close - s0 * ev.preclose
            cash += credit
            flow += credit
            p.shares += bonus_sh + rights_sh
            p.ca_credit += credit
        # 2) 开盘卖
        for j, p in list(pos.items()):
            if not p.pending:
                continue
            lim = limit_prices(P, i, j)
            tradable = bool(P.listed[i, j] and not P.suspended[i, j]
                            and np.nan_to_num(P.mat["volume"][i, j], nan=0.0) > 0)
            px_open = float(P.mat["open"][i, j]) if tradable else np.nan
            blocked = (not tradable) or (px_open <= lim.down + 0.001)
            if blocked:
                p.pend_age += 1
                if p.pend_age >= 10:
                    px = float(P.mat["close"][i, j]) if np.isfinite(P.mat["close"][i, j]) else p.last_px
                    if np.isfinite(px) and px > 0:
                        pr, gross = _sell(P, cfg, fcfg, p, i, px, "sell_persistence_10d", res)
                        cash += pr
                        flow += pr
                        day_turnover += gross
                        res["forced"] += 1
                        del pos[j]
                continue
            pr, gross = _sell(P, cfg, fcfg, p, i, px_open, "scheduled_exit", res)
            cash += pr
            flow += pr
            day_turnover += gross
            del pos[j]
        # 3) 开盘买
        free = cfg.slots - len(pos)
        if free > 0:
            nav_prev = max(res["daily"][-1]["nav"] if res["daily"] else cfg.capital, cash)
            for ev in by_entry.get(i, []):
                if free <= 0:
                    break
                j = ev["j"]
                if j in pos:
                    continue
                lim = limit_prices(P, i, j)
                tradable = bool(P.listed[i, j] and not P.suspended[i, j]
                                and np.nan_to_num(P.mat["volume"][i, j], nan=0.0) > 0)
                if (not tradable) or P.resume[i, j] or P.mat["is_limit_up_open"][i, j] > 0:
                    res["failed"].append(dict(
                        date=str(d), symbol=ev["symbol"], side="buy",
                        reason="suspended" if not tradable else
                        ("resume" if P.resume[i, j] else "limit_up_open"), window=cfg.window))
                    continue
                px_open = float(P.mat["open"][i, j])
                adv = float(np.nan_to_num(P.adv20[i - 1, j], nan=0.0)) if i > 0 else 0.0
                budget = nav_prev / max(cfg.slots, 1)
                px = fill_price(px_open, "buy", slip_bps(fcfg.slippage_bps, budget, adv, 0.0))
                if not lim.no_limit:
                    px = min(px, lim.up)
                if (not np.isfinite(px)) or px <= 0:
                    continue
                qty = int(budget / px / N_LOT) * N_LOT
                qty = min(qty, int(max(cash, 0.0) * 0.995 / (px * 1.002) / N_LOT) * N_LOT)
                vol = float(np.nan_to_num(P.mat["volume"][i, j], nan=0.0))
                cap_notional = min(0.02 * adv, 0.02 * px_open * vol) if adv > 0 else 0.0
                if cap_notional > 0:
                    qty = min(qty, int(cap_notional / px / N_LOT) * N_LOT)
                risk = np.nan
                if cfg.exit == "STOP":
                    A_e = float(P.a["A"][i, j])
                    stop_raw = (ev["L"] - STOP_ATR * ev["atr"]) * A_e if A_e > 0 else np.nan
                    if (not np.isfinite(stop_raw)) or stop_raw <= 0:
                        continue
                    risk = (px_open - stop_raw) / px_open
                    if risk <= 0 or risk > MAX_STOP_PCT:
                        res["failed"].append(dict(date=str(d), symbol=ev["symbol"], side="buy",
                                                  reason="stop_too_far", window=cfg.window))
                        continue
                    qty = min(qty, int(MAX_RISK * nav_prev / (risk * px) / N_LOT) * N_LOT)
                while qty >= N_LOT:
                    fees = FEE.trade_fees(d, ev["symbol"], "buy", px, qty, fcfg)
                    if px * qty + fees.total <= cash + 1e-6:
                        break
                    qty -= N_LOT
                if qty < N_LOT:
                    res["failed"].append(dict(date=str(d), symbol=ev["symbol"], side="buy",
                                              reason="insufficient_cash", window=cfg.window))
                    continue
                fees = FEE.trade_fees(d, ev["symbol"], "buy", px, qty, fcfg)
                cost = px * qty + fees.total
                cash -= cost
                flow -= cost
                day_turnover += px * qty
                day_cost += fees.total
                stop_a = (ev["L"] - STOP_ATR * ev["atr"]) if cfg.exit == "STOP" else float("nan")
                pos[j] = Pos(j=j, symbol=ev["symbol"], shares=qty, entry_i=i, entry_px=px,
                             entry_raw=px_open, cost=cost, entry_fees=fees.total,
                             stop_a=stop_a, last_px=px_open, last_i=i, b_idx=ev["b_idx"])
                res["fees"] += fees.total
                res["slip_cost"] += abs(px - px_open) * qty
                free -= 1
        # 4) 收盘：排程退出
        for j, p in list(pos.items()):
            if p.pending:
                continue
            if cfg.exit in ("FIX20", "FIX40"):
                exit_now = (i - p.entry_i) >= (hold - 1)
            else:
                ac = float(P.a["close"][i, j])
                exit_now = ((np.isfinite(ac) and np.isfinite(p.stop_a) and ac < p.stop_a)
                            or (i - p.entry_i) >= (hold - 1))
            if exit_now:
                p.pending = True
                p.pend_i = i
                p.pend_age = 0
        # 5) 估值
        mv = 0.0
        for j, p in pos.items():
            px = float(P.mat["close"][i, j])
            if (not np.isfinite(px)) or px <= 0:
                px = p.last_px
            else:
                p.last_px = px
                p.last_i = i
            mv += p.shares * px
        nav = cash + mv
        resid = abs((cash - cash_prev) - flow)
        res["cash_resid_max"] = max(res["cash_resid_max"], resid)
        if resid > 0.02:
            res["cash_resid_bad"] += 1
        res["daily"].append(dict(
            date=P.dates[i], cash=round(cash, 4), market_value=round(mv, 4), nav=round(nav, 4),
            n_positions=len(pos), turnover=round(day_turnover, 2),
            cost=round(day_cost, 2), cash_resid=round(resid, 6)))
        if nav <= 0:
            raise RuntimeError(f"{d} NAV 非正")

    # 期末：按最后收盘估值；交易统计用「期末标记」，不产生现金流与费用
    for j, p in list(pos.items()):
        px = p.last_px if p.last_px > 0 else float(P.mat["close"][i1, j])
        gross = px * p.shares
        pnl = gross - p.cost + p.ca_credit
        res["trades"].append(dict(
            symbol=p.symbol, entry_date=str(P.dates[p.entry_i].date()),
            exit_date=str(P.dates[p.last_i].date()), entry_px=round(p.entry_px, 4),
            exit_px=round(px, 4), shares=int(p.shares), entry_fees=round(p.entry_fees, 2),
            exit_fees=0.0, dividends=round(p.div_gross, 2), tax=0.0,
            ca_credit=round(p.ca_credit, 2), pnl=round(pnl, 2),
            ret=float(pnl / p.cost) if p.cost > 0 else np.nan,
            hold_days=int(p.last_i - p.entry_i), reason="end_of_period_mark", is_mark=True,
            accounting="shares", window=cfg.window))
        res["hold_days"].append(p.last_i - p.entry_i)
    res["daily"] = pd.DataFrame(res["daily"])
    res["trades"] = pd.DataFrame(res["trades"])
    res["failed"] = pd.DataFrame(res["failed"])
    res["final_cash"] = float(cash)
    return res


# ---------------------------------------------------------------------------
# 绩效
# ---------------------------------------------------------------------------
def metrics_of(daily: pd.DataFrame, trades: pd.DataFrame, capital: float) -> tuple[dict, pd.DataFrame]:
    tr = pd.DataFrame()
    if len(trades):
        tr = pd.DataFrame({
            "entry_date": pd.to_datetime(trades["entry_date"]),
            "exit_date": pd.to_datetime(trades["exit_date"]),
            "entry_price": trades["entry_px"].astype(float),
            "exit_price": trades["exit_px"].astype(float),
            "shares": trades["shares"].astype(float),
            "pnl": trades["pnl"].astype(float),
            "ret": trades["ret"].astype(float),
            "hold_days": trades["hold_days"].astype(float),
        })
    eq = daily.set_index(pd.DatetimeIndex(pd.to_datetime(daily["date"])))[
        ["nav", "market_value", "cash", "n_positions", "turnover", "cost"]]
    res = STAT.BacktestResult(eq, tr, meta={"initial_cash": capital})
    m = STAT.compute_metrics(res, initial_cash=capital)
    ann = STAT.annual_table(res, initial_cash=capital)
    m["avg_exposure"] = float((eq["market_value"] / eq["nav"]).mean())
    m["avg_pos"] = float(eq["n_positions"].mean())
    m["turnover_x"] = float(eq["turnover"].sum() / eq["nav"].mean() / max(m.get("years", 1), 1e-9))
    m["fees_total"] = float(eq["cost"].sum())
    return m, ann


# ---------------------------------------------------------------------------
# 统一账户复核（账户撮合.UnifiedAccount，与独立引擎互相独立记账）
# ---------------------------------------------------------------------------
def run_unified(P: Panel, det: pd.DataFrame, cfg: PortCfg) -> dict:
    import 账户撮合 as ACC
    import 统一回测 as UB
    i0, i1 = P.window_index(*WINDOWS[cfg.window])
    shim = SimpleNamespace(dates=[x.date() for x in P.dates], codes=P.codes,
                           ipo_date=P.ipo_date, delist_date=P.delist_date)
    rc = UB.RunConfig(variant="BRK", window=cfg.window, accounting="shares",
                      initial_cash=cfg.capital, slots=cfg.slots,
                      max_new_per_day=cfg.slots, max_hold=20, slippage_bps=10.0,
                      transfer_par_rate=0.0010, rank="hash", label=cfg.label)
    acct = UB._make_account(shim, rc)
    acct._code_pos = P.code_idx
    sched = build_schedule(P, det, cfg)
    by_entry: dict[int, list[dict]] = {}
    for ev in sched:
        by_entry.setdefault(ev["entry_i"], []).append(ev)
    for k in by_entry:
        by_entry[k].sort(key=lambda e: _rank_key(P, cfg, e))
    hold = 20 if cfg.exit == "FIX20" else 40
    meta: dict[str, dict] = {}
    rows = []
    n_failed = 0
    for i in range(i0, i1 + 1):
        dd = P.dates[i].date()
        held = list(acct.positions.keys())
        active = [o.code for o in acct._orders if o.active]
        ev_today = P.events.on(dd)
        needed = set(held) | set(active) | {e.symbol for e in ev_today}
        acct._ca_row = P.ca_factor[i]
        acct._adv = {c: float(np.nan_to_num(P.adv20[i - 1, P.code_idx[c]], nan=0.0))
                     for c in (set(active) | set(held)) if i > 0 and c in P.code_idx}
        acct._resume_today = {P.codes[j] for j in np.flatnonzero(P.resume[i])}
        bars = {}
        for c in needed:
            j = P.code_idx.get(c)
            if j is None or not P.listed[i, j]:
                continue
            try:
                bars[c] = ACC.Bar(
                    code=c, date=dd, open=float(P.mat["open"][i, j]),
                    high=float(P.mat["high"][i, j]), low=float(P.mat["low"][i, j]),
                    close=float(P.mat["close"][i, j]),
                    volume=float(np.nan_to_num(P.mat["volume"][i, j], nan=0.0)),
                    amount=float(np.nan_to_num(P.mat["amount"][i, j], nan=0.0)),
                    prev_close=float(P.mat["preclose"][i, j]),
                    prev_close_is_ref=True, is_st=bool(P.mat["is_st"][i, j] > 0),
                    trade_status=0 if P.suspended[i, j] else 1)
            except Exception:
                continue
        rec = acct.on_day(dd, bars=bars, events=ev_today)
        forced = False
        for o in list(acct._orders):
            if not (o.active and o.side == "sell" and o.filled_qty == 0):
                continue
            try:
                age = i - int(np.searchsorted(P.dates.values, np.datetime64(o.execute_from)))
            except KeyError:
                continue
            if age >= 10:
                pos = acct.positions.get(o.code)
                j = P.code_idx.get(o.code)
                if pos is None or j is None:
                    continue
                px = float(P.mat["close"][i, j])
                if not np.isfinite(px) or px <= 0:
                    continue
                acct._force_sell(o.code, dd, px, "sell_persistence_10d")
                forced = True
        if forced:
            rec = acct._mark_and_record(dd)
        rows.append(dict(date=P.dates[i], cash=float(rec["cash_total"]),
                         market_value=float(rec["market_value"]), nav=float(rec["nav"]),
                         n_positions=int(rec["n_positions"]),
                         turnover=0.0, cost=float(sum(t.fees for t in acct.trades
                                                      if t.date == dd))))
        for code, pos in list(acct.positions.items()):
            if pos.shares > 0 and code not in meta:
                meta[code] = dict(entry_i=i, stop_a=np.nan)
        for code in list(meta):
            if code not in acct.positions:
                meta.pop(code, None)
        # 收盘决策
        if i < i1:
            # 卖出
            for code, p in list(acct.positions.items()):
                m = meta.get(code)
                if m is None or p.shares <= 0:
                    continue
                has_sell = any(o.active and o.side == "sell" and o.code == code
                               for o in acct._orders)
                if has_sell:
                    continue
                j = P.code_idx.get(code)
                exit_now = (i - m["entry_i"]) >= (hold - 1)
                if cfg.exit == "STOP" and j is not None:
                    ac = float(P.a["close"][i, j])
                    if np.isfinite(ac) and ac < m["stop_a"]:
                        exit_now = True
                if exit_now:
                    acct.submit(ACC.Order(code=code, side="sell", qty=int(p.shares),
                                          kind="open", tif="gtc", tag="sched_exit"), dd)
            # 买入：明天开盘先卖后买，已挂卖单的仓位当日释放槽位（父预注册 8.7）
            n_hold = sum(1 for p in acct.positions.values() if p.shares > 0)
            n_pending_sell = sum(1 for o in acct._orders if o.active and o.side == "sell")
            free = cfg.slots - n_hold + n_pending_sell
            if free > 0:
                nav = float(rec["nav"])
                for ev in by_entry.get(i + 1, []):
                    if free <= 0:
                        break
                    code = ev["symbol"]
                    if code in acct.positions:
                        continue
                    j = ev["j"]
                    lim = limit_prices(P, i + 1, j)
                    if P.mat["is_limit_up_open"][i + 1, j] > 0 or P.resume[i + 1, j]:
                        n_failed += 1
                        continue
                    px = float(P.mat["close"][i, j])
                    qty = int(nav / cfg.slots / px / N_LOT) * N_LOT
                    if qty < N_LOT:
                        n_failed += 1
                        continue
                    if cfg.exit == "STOP":
                        A = float(P.a["A"][i + 1, j])
                        stop_raw = (ev["L"] - STOP_ATR * ev["atr"]) * A if A > 0 else np.nan
                        if (not np.isfinite(stop_raw)) or (px - stop_raw) / px > MAX_STOP_PCT:
                            n_failed += 1
                            continue
                    acct.submit(ACC.Order(code=code, side="buy", qty=qty, kind="open",
                                          tif="day", tag="brk"), dd)
                    if cfg.exit == "STOP":
                        meta[code] = dict(entry_i=i + 1, stop_a=ev["L"] - STOP_ATR * ev["atr"])
                    free -= 1
    acct.finalize(P.dates[i1].date(), liquidate=False)
    daily = pd.DataFrame(rows)
    trades = pd.DataFrame(acct.round_trips)
    tr = trades.copy()
    if len(tr):
        tr = tr.rename(columns={"entry_price": "entry_px", "exit_price": "exit_px"})
        tr["is_mark"] = False
    m, _ = metrics_of(daily, tr, cfg.capital)
    m["n_failed_orders"] = n_failed
    m["unified_flags"] = json.dumps(dict(acct.flag_counts), ensure_ascii=False)
    m["final_cash"] = float(acct.cash_total)
    return dict(metrics=m, daily=daily, trades=tr)


# ---------------------------------------------------------------------------
# 组合层
# ---------------------------------------------------------------------------


def benchmarks(P: Panel) -> pd.DataFrame:
    """B1 主宇宙等权（月末再平衡，15bp/单位换手）、B2 沪深300、B4 现金。"""
    import duckdb
    rows = []
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        idx = con.execute(
            "select date, close from index_daily where symbol='000300.SH' and date between ? and ? order by date",
            [FULL[0], FULL[1]]).df()
    finally:
        con.close()
    hs = pd.Series(idx["close"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(idx["date"])))
    for wname in ("dev", "val", "full"):
        i0, i1 = P.window_index(*WINDOWS[wname])
        dates = P.dates[i0:i1 + 1]
        rets = np.zeros(len(dates))
        prev_w: dict[int, float] = {}
        active: list[int] = []
        for k in range(1, len(dates)):
            i = i0 + k
            if k == len(dates) - 1 or dates[k + 1].month != dates[k].month:
                cols = np.flatnonzero(P.elig[i])
                active = cols.tolist()
                w_new = {int(j): 1.0 / len(active) for j in active} if active else {}
                names = set(prev_w) | set(w_new)
                turn = 0.5 * sum(abs(w_new.get(n, 0.0) - prev_w.get(n, 0.0)) for n in names)
                rets[k] -= turn * 0.0015
                prev_w = w_new
            if active:
                a0 = P.a["close"][i - 1, active]
                a1 = P.a["close"][i, active]
                ok = np.isfinite(a0) & np.isfinite(a1) & (a0 > 0)
                if ok.sum():
                    rets[k] += float(np.mean(a1[ok] / a0[ok] - 1.0))
        nav = np.cumprod(1.0 + rets)
        yrs = (dates[-1] - dates[0]).days / 365.25
        cagr = float(nav[-1] ** (1 / yrs) - 1)
        dd = float(np.min(nav / np.maximum.accumulate(nav) - 1))
        vol = float(np.std(rets[1:], ddof=1) * math.sqrt(TRADING_DAYS))
        h = hs.reindex(dates).ffill().dropna()
        hy = (h.index[-1] - h.index[0]).days / 365.25
        hcagr = float((h.iloc[-1] / h.iloc[0]) ** (1 / hy) - 1) if len(h) > 2 else np.nan
        hdd = float(np.min(h / np.maximum.accumulate(h) - 1)) if len(h) > 2 else np.nan
        rows.append(dict(window=wname, bench="B1_EW_MAINBOARD", start=str(dates[0].date()),
                         end=str(dates[-1].date()), cagr=cagr, mdd=dd, vol=vol,
                         sharpe=cagr / vol if vol else np.nan))
        rows.append(dict(window=wname, bench="B2_HS300_PRICE", start=str(h.index[0].date()) if len(h) else "",
                         end=str(h.index[-1].date()) if len(h) else "", cagr=hcagr, mdd=hdd,
                         vol=np.nan, sharpe=np.nan))
        rows.append(dict(window=wname, bench="B4_CASH", start=str(dates[0].date()),
                         end=str(dates[-1].date()), cagr=0.0, mdd=0.0, vol=0.0, sharpe=np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "03b_基准.csv", index=False)
    return out

def run_one(P: Panel, det: pd.DataFrame, cfg: PortCfg) -> dict:
    t0 = time.time()
    res = simulate(P, det, cfg)
    m, ann = metrics_of(res["daily"], res["trades"], cfg.capital)
    n_closed = int((~res["trades"]["is_mark"]).sum()) if len(res["trades"]) else 0
    row = dict(label=cfg.label or cfg.key(), arm=cfg.arm, N=cfg.N, W=cfg.W, tol=cfg.tol,
               target=cfg.target, mode=cfg.mode, exit=cfg.exit, window=cfg.window,
               capital=cfg.capital, slots=cfg.slots, rank=cfg.rank, cost_mult=cfg.cost_mult,
               delay=cfg.delay, n_closed=n_closed, n_trades_total=int(len(res["trades"])),
               n_buys=int(len(res["trades"])),
               fees=float(res["fees"]), slip_cost=float(res["slip_cost"]),
               forced=int(res["forced"]), cash_resid_max=float(res["cash_resid_max"]),
               cash_resid_bad=int(res["cash_resid_bad"]), n_failed=len(res["failed"]),
               time_s=round(time.time() - t0, 2))
    for k in ("cagr", "mdd", "calmar", "sharpe", "vol", "total_return", "final_nav",
              "avg_exposure", "avg_pos", "turnover_x", "win_rate", "profit_factor",
              "avg_trade_ret", "median_trade_ret", "n_trades"):
        row[k] = m.get(k, np.nan)
    row["years"] = m.get("years", np.nan)
    return dict(row=row, res=res, metrics=m, annual=ann)


def make_portfolio_plan() -> list[PortCfg]:
    P = PRIMARY
    plan: list[PortCfg] = []

    def add(**kw):
        base = dict(arm="pull", N=P["N"], W=P["W"], tol=P["tol"], target=P["target"],
                    mode=P["mode"], exit="FIX20", window="dev", label="")
        base.update(kw)
        c = PortCfg(**base)
        if not c.label:
            c.label = c.key()
        plan.append(c)

    for w in ("dev", "val"):
        for arm in ("pull", "imm", "pull_rs", "nopull", "fake"):
            mode = "restrength" if arm == "pull_rs" else "direct"
            add(arm=arm, mode=mode, window=w, label=f"主-{ARM_LABEL[arm]}")
    for w in ("dev", "val"):
        for N in (20, 120):
            add(N=N, window=w, label=f"邻域-N{N}")
        for W in (5, 20):
            add(W=W, window=w, label=f"邻域-W{W}")
        for tol in (0.25, 1.0):
            add(tol=tol, window=w, label=f"邻域-tol{tol}")
        add(target="ema20", window=w, label="邻域-ema20")
        add(exit="FIX40", window=w, label="邻域-FIX40")
        add(exit="STOP", window=w, label="邻域-STOP")
        add(cost_mult=2.0, window=w, label="稳健-成本2x")
        add(delay=1, window=w, label="稳健-延迟1日")
        add(capital=20_000, window=w, label="稳健-2万")
        add(capital=100_000, window=w, label="稳健-10万")
        add(slots=5, window=w, label="稳健-5仓")
        add(slots=20, window=w, label="稳健-20仓")
        add(rank="adv", window=w, label="稳健-ADV排序")
        add(rank="strength", window=w, label="稳健-强度排序")
    return plan


def cmd_portfolio(P: Panel, det: pd.DataFrame) -> pd.DataFrame:
    plan = make_portfolio_plan()
    rows, navs, trades_all, annuals, failed_all = [], [], [], [], []
    for cfg in plan:
        out = run_one(P, det, cfg)
        rows.append(out["row"])
        d = out["res"]["daily"].copy()
        d["label"] = cfg.label
        navs.append(d)
        tr = out["res"]["trades"].copy()
        if len(tr):
            tr["label"] = cfg.label
            trades_all.append(tr)
        a = out["annual"].copy()
        if len(a):
            a["label"] = cfg.label
            annuals.append(a)
        f = out["res"]["failed"].copy()
        if len(f):
            f["label"] = cfg.label
            failed_all.append(f)
        log(f"  {cfg.label:22s} {cfg.window:4s} CAGR={out['row']['cagr']:+.2%} "
            f"MDD={out['row']['mdd']:+.1%} n={out['row']['n_trades']} "
            f"t={out['row']['time_s']}s")
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "05_组合_汇总.csv", index=False)
    pd.concat(navs, ignore_index=True).to_csv(OUT / "06_组合_净值.csv", index=False)
    if trades_all:
        pd.concat(trades_all, ignore_index=True).to_csv(OUT / "07_组合_交易.csv", index=False)
    if annuals:
        pd.concat(annuals, ignore_index=True).to_csv(OUT / "08_组合_分年.csv", index=False)
    if failed_all:
        pd.concat(failed_all, ignore_index=True).to_csv(OUT / "14_失败订单.csv", index=False)
    return summ


def cmd_unified_check(P: Panel, det: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in ("dev", "val"):
        for arm, mode, label in (("pull", "direct", "主-回踩直接买"),
                                 ("imm", "direct", "主-立即突破"),
                                 ("pull_rs", "restrength", "主-转强买")):
            cfg = PortCfg(arm=arm, N=PRIMARY["N"], W=PRIMARY["W"], tol=PRIMARY["tol"],
                          target=PRIMARY["target"], mode=mode, exit="FIX20", window=w,
                          label=f"统一-{label}")
            try:
                out = run_unified(P, det, cfg)
                m = out["metrics"]
                mine = summ[(summ["label"] == f"主-{ARM_LABEL[arm]}") & (summ["window"] == w)]
                tr_m = pd.read_csv(OUT / "07_组合_交易.csv", parse_dates=["entry_date", "exit_date"])
                tr_m = tr_m[(tr_m["label"] == f"主-{ARM_LABEL[arm]}") & (tr_m["window"] == w)]
                tu = out["trades"].copy()
                ov = 0
                px_d = np.nan
                ret_d = np.nan
                if len(tu) and len(tr_m):
                    tu["ed"] = pd.to_datetime(tu["entry_date"]).dt.strftime("%Y-%m-%d")
                    km = set(zip(tr_m["entry_date"].dt.strftime("%Y-%m-%d"), tr_m["symbol"]))
                    ku = set(zip(tu["ed"], tu["code"]))
                    ov = len(km & ku)
                    j = tr_m.assign(ed=tr_m["entry_date"].dt.strftime("%Y-%m-%d")).merge(
                        tu, left_on=["ed", "symbol"], right_on=["ed", "code"], suffixes=("_m", "_u"))
                    if len(j):
                        px_d = float(((j["entry_px_m"] / j["entry_px_u"] - 1) * 1e4).median())
                        ret_d = float((j["ret_m"] - j["ret_u"]).mean())
                rows.append(dict(engine="UnifiedAccount", label=label, window=w,
                                 cagr=m.get("cagr"), mdd=m.get("mdd"), calmar=m.get("calmar"),
                                 n_trades=m.get("n_trades"), final_nav=m.get("final_nav"),
                                 fees=m.get("fees_total"), flags=m.get("unified_flags"),
                                 indep_cagr=float(mine["cagr"].iloc[0]) if len(mine) else np.nan,
                                 indep_final_nav=float(mine["final_nav"].iloc[0]) if len(mine) else np.nan,
                                 indep_n_trades=float(mine["n_trades"].iloc[0]) if len(mine) else np.nan,
                                 overlap_entries=int(ov),
                                 overlap_entry_px_median_bp=px_d,
                                 overlap_ret_diff_mean=ret_d))
                log(f"  统一 {label} {w}: CAGR={m.get('cagr'):+.2%} "
                    f"独立={float(mine['cagr'].iloc[0]) if len(mine) else np.nan:+.2%}")
            except Exception as exc:  # noqa: BLE001
                rows.append(dict(engine="UnifiedAccount", label=label, window=w,
                                 cagr=np.nan, error=repr(exc)))
                log(f"  统一 {label} {w} 失败: {exc!r}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "09_统一引擎复核.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _stream_stats(P: Panel) -> dict:
    """A 空间恒等 vs 冻结层 hfq 错误用法（分块，控制内存）。"""
    a_c, pct = P.a["close"], P.mat["pct_chg"] / 100.0
    hfq = P.mat["close"] * P.mat["factor"]
    err_max = 0.0
    err_n = 0
    err_over = 0
    e_max = 0.0
    e_over = 0
    e_n = 0
    sxy = sxx = sy = 0.0
    for i0 in range(1, P.n_days, 200):
        i1 = min(i0 + 200, P.n_days)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = a_c[i0:i1] / a_c[i0 - 1:i1 - 1] - 1.0
            rh = hfq[i0:i1] / hfq[i0 - 1:i1 - 1] - 1.0
        p = pct[i0:i1]
        e = np.abs(r - p)
        f = np.isfinite(e)
        err_n += int(f.sum())
        if f.any():
            err_max = max(err_max, float(e[f].max()))
            err_over += int((e[f] > 1e-5).sum())
        eh = np.abs(rh - p)
        fh = np.isfinite(eh)
        e_n += int(fh.sum())
        if fh.any():
            e_max = max(e_max, float(eh[fh].max()))
            e_over += int((eh[fh] > 1e-5).sum())
            x = p[fh].astype(np.float64)
            y = rh[fh].astype(np.float64)
            sxy += float(np.sum((x - x.mean()) * (y - y.mean())))
            sxx += float(np.sum((x - x.mean()) ** 2))
    return {
        "aspce_identity": {"n": err_n, "max_err": err_max, "n_over_1e-5": err_over},
        "hfq_misuse": {"n": e_n, "max_err": e_max, "n_over_1e-5": e_over,
                       "share_over_1e-5": e_over / max(e_n, 1),
                       "slope_vs_pct": sxy / sxx if sxx > 0 else np.nan},
    }


def _limit_flag_check(P: Panel, n_sample: int = 50_000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P.n_days, n_sample)
    jdx = rng.integers(0, P.n_codes, n_sample)
    bad_u = bad_d = checked = 0
    examples: list[dict] = []
    for i, j in zip(idx, jdx):
        if not P.listed[i, j] or not np.isfinite(P.mat["preclose"][i, j]):
            continue
        try:
            lim = limit_prices(P, int(i), int(j))
        except Exception:
            continue
        checked += 1
        up = (not lim.no_limit) and P.mat["open"][i, j] >= lim.up - 1e-6
        dn = (not lim.no_limit) and P.mat["open"][i, j] <= lim.down + 1e-6
        if up != bool(P.mat["is_limit_up_open"][i, j] > 0):
            bad_u += 1
            if len(examples) < 5:
                examples.append({"date": str(P.dates[i].date()), "code": P.codes[j],
                                 "side": "up", "open": float(P.mat["open"][i, j]),
                                 "preclose": float(P.mat["preclose"][i, j]),
                                 "recomputed_up": float(lim.up),
                                 "frozen_flag": bool(P.mat["is_limit_up_open"][i, j] > 0)})
        if dn != bool(P.mat["is_limit_down_open"][i, j] > 0):
            bad_d += 1
    return {"checked": checked, "mismatch_up": bad_u, "mismatch_down": bad_d,
            "examples": examples}


def _elig_slice(P: Panel, ic: int, cols: np.ndarray) -> np.ndarray:
    sl = slice(0, ic + 1)
    close = P.mat["close"][sl][:, cols]
    listed = P.listed[sl][:, cols]
    susp = P.suspended[sl][:, cols]
    st = P.mat["is_st"][sl][:, cols]
    tds = P.mat["td_since_list"][sl][:, cols]
    amt = pd.DataFrame(P.mat["amount"][sl][:, cols])
    adv20 = amt.rolling(20, min_periods=20).mean().to_numpy()
    days = np.array([(d.date() - date(1990, 1, 1)).days for d in P.dates[sl]], dtype=float)
    ipod = np.array([np.nan if x is None else (x - date(1990, 1, 1)).days
                     for x in np.array(P.ipo_date, dtype=object)[cols]], dtype=float)
    dsi = days[:, None] - ipod[None, :]
    prev_susp = np.vstack([np.ones((1, len(cols)), dtype=bool), susp[:-1]])
    prev_listed = np.vstack([np.zeros((1, len(cols)), dtype=bool), listed[:-1]])
    resume = prev_susp | ~prev_listed
    delw = np.zeros_like(listed)
    for k, j in enumerate(cols):
        dd = P.delist_date[j]
        if dd is None:
            continue
        kk = int(np.searchsorted(P.dates.values, np.datetime64(pd.Timestamp(dd))))
        if kk < ic + 1:
            delw[max(0, kk - DELIST_BLOCK_TD):kk + 1, k] = True
    return (listed & ~susp & (st == 0) & (tds >= MIN_LIST_TD) & (close >= MIN_PRICE)
            & (adv20 >= MIN_ADV20) & (dsi >= MIN_LIST_CAL) & ~resume & ~delw)


def _slice_mask(P: Panel, ic: int, cols: np.ndarray, N: int) -> np.ndarray:
    sl = slice(0, ic + 1)
    a_h = P.a["high"][sl][:, cols]
    a_c = P.a["close"][sl][:, cols]
    h = pd.DataFrame(a_h)
    c = pd.DataFrame(a_c)
    prev = c.shift(1)
    tr = np.maximum(np.maximum(a_h - a_c, np.abs(a_h - prev.to_numpy())),
                    np.abs(a_c - prev.to_numpy()))
    atr = pd.DataFrame(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
    rmax = h.rolling(N, min_periods=N).max().shift(1).to_numpy()
    pct = P.mat["pct_chg"][sl][:, cols]
    extreme = np.where(np.isfinite(pct), (np.abs(pct) > 30).astype(float), np.nan)
    roll = pd.DataFrame(extreme).rolling(N + 1, min_periods=N + 1).sum().to_numpy()
    with np.errstate(invalid="ignore"):
        brk = (_elig_slice(P, ic, cols) & np.isfinite(rmax) & (a_c > rmax)
               & np.isfinite(atr) & (atr > 0))
    return brk & (roll == 0)


def truncation_check(P: Panel, N: int, n_sym: int = 40,
                     cutoffs: Sequence[str] = ("2011-12-30", "2015-12-31", "2019-12-31")) -> dict:
    rng = np.random.default_rng(20261028)
    cols = np.sort(rng.choice(P.n_codes, size=min(n_sym, P.n_codes), replace=False))
    brk_full, _ = breakout_mask(P, N)
    rows = []
    for cut in cutoffs:
        ic = int(np.searchsorted(P.dates.values, np.datetime64(pd.Timestamp(cut))))
        if ic <= 200 or ic >= P.n_days:
            continue
        sl = _slice_mask(P, ic, cols, N)
        ref = brk_full[:ic + 1][:, cols]
        rows.append(dict(cutoff=cut, n_days=ic + 1, mismatch=int((sl != ref).sum()),
                         n_ref=int(ref.sum()), n_slice=int(sl.sum())))
    return {"n_symbols": int(len(cols)), "rows": rows}


def trade_checks(P: Panel, port: pd.DataFrame) -> dict:
    tr = pd.read_csv(OUT / "07_组合_交易.csv", parse_dates=["entry_date", "exit_date"])
    main = tr[tr["label"].isin([f"主-{ARM_LABEL[a]}" for a in ("pull", "imm", "pull_rs")])]
    real = main[~main["is_mark"].astype(bool)]
    out = {"n": int(len(main)), "n_closed": int(len(real))}
    if not len(main):
        return out
    out["t1_violations"] = int((real["exit_date"] <= real["entry_date"]).sum())
    odd = real[real["shares"] % N_LOT != 0]
    out["odd_lot_sells"] = int(len(odd))
    out["odd_lot_without_ca"] = int((odd["ca_credit"].abs() + odd["dividends"].abs() == 0).sum())
    bad_buy = bad_sell = susp = 0
    for r in main.itertuples():
        i = int(np.searchsorted(P.dates.values, np.datetime64(r.entry_date)))
        j = P.code_idx.get(r.symbol)
        if j is not None and 0 <= i < P.n_days:
            bad_buy += int(P.mat["is_limit_up_open"][i, j] > 0)
            susp += int(P.suspended[i, j])
        i2 = int(np.searchsorted(P.dates.values, np.datetime64(r.exit_date)))
        if j is not None and 0 <= i2 < P.n_days and r.reason == "scheduled_exit":
            bad_sell += int(P.mat["is_limit_down_open"][i2, j] > 0)
    out.update(buy_at_limit_up_open=bad_buy, sell_at_limit_down_open=bad_sell,
               fills_on_suspension=susp)
    out["min_hold"] = int(main["hold_days"].min())
    out["median_hold"] = float(main["hold_days"].median())
    return out


def cmd_selfcheck(P: Panel, det: pd.DataFrame, summ: pd.DataFrame,
                  port: pd.DataFrame, uni: pd.DataFrame) -> dict:
    out: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "prereg_sha256": prereg_sha(),
                 "code_sha256": sha256_file(Path(__file__).resolve()),
                 "panel": P.meta}
    out.update(_stream_stats(P))
    out["limit_flags"] = _limit_flag_check(P)
    out["truncation"] = {f"N{N}": truncation_check(P, N) for N in NS}
    out["trade_checks"] = trade_checks(P, port)
    out["no_2021"] = {
        "event_max_date": str(pd.to_datetime(det["date_b"]).max().date()),
        "nav_max_date": str(pd.to_datetime(
            pd.read_csv(OUT / "06_组合_净值.csv", usecols=["date"])["date"]).max().date()),
        "guard_rejects_2021": _guard_check(),
    }
    # 事件分类恒等
    rows = []
    dN = det[det["N"] == PRIMARY["N"]]
    for wname in ("dev", "val", "full"):
        wm = _window_mask(dN, wname)
        total = int(wm.sum())
        s = summ[(summ["N"] == PRIMARY["N"]) & (summ["W"] == PRIMARY["W"])
                 & (summ["tol"] == PRIMARY["tol"]) & (summ["target"] == PRIMARY["target"])
                 & (summ["mode"] == PRIMARY["mode"]) & (summ["window"] == wname)]
        st = s[s["arm"] == "status_count"]
        if not len(st):
            continue
        n_touched = int(st["n"].iloc[0])
        n_no = int(st["n_nopull"].iloc[0])
        n_fa = int(st["n_fake"].iloc[0])
        n_imm = int(s[s["arm"] == "imm"]["n"].iloc[0])
        n_pu = int(s[s["arm"] == "pull"]["n"].iloc[0])
        rows.append(dict(window=wname, n_events_total=total,
                         n_touched=n_touched, n_nopull=n_no, n_fake=n_fa,
                         classify_resid=total - (n_touched + n_no + n_fa),
                         n_exec_imm=n_imm, n_exec_pull=n_pu,
                         exec_resid=n_imm - (n_pu + n_no + n_fa)))
    out["event_identity"] = rows
    # 账户 PnL 恒等（主回踩 dev）
    eq = pd.read_csv(OUT / "06_组合_净值.csv", parse_dates=["date"])
    tr = pd.read_csv(OUT / "07_组合_交易.csv", parse_dates=["entry_date", "exit_date"])
    checks = []
    for label in (f"主-{ARM_LABEL[a]}" for a in ("pull", "imm", "pull_rs")):
        for w in ("dev", "val"):
            e = eq[(eq["label"] == label) & (eq["date"] >= pd.Timestamp(WINDOWS[w][0]))
                   & (eq["date"] <= pd.Timestamp(WINDOWS[w][1]))]
            t = tr[(tr["label"] == label) & (tr["window"] == w)]
            if not len(e) or not len(t):
                continue
            checks.append(dict(label=label, window=w,
                               pnl_sum_minus_nav=float(t["pnl"].sum()
                                                      - (e["nav"].iloc[-1] - 1_000_000.0)),
                               cash_resid=float(port[(port["label"] == label)
                                                     & (port["window"] == w)]["cash_resid_max"].iloc[0])))
    out["account_identity"] = checks
    out["trial_counts"] = {"platform_cells": 144, "account_runs": int(len(port)),
                           "unified_runs": int(len(uni)),
                           "total": 144 + int(len(port)) + int(len(uni))}
    (OUT / "10_自检.json").write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                                 default=float), encoding="utf-8")
    log("自检写入 10_自检.json")
    return out


def _guard_check() -> bool:
    try:
        EventBook().load("2020-01-01", "2021-01-04")
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# 试验台账与数据口径说明
# ---------------------------------------------------------------------------
def write_ledger(plat: pd.DataFrame, port: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    rows = []
    k = 0
    for r in plat.itertuples(index=False):
        k += 1
        rows.append(dict(trial_id=f"BRK-E{k:04d}", kind="experiment", stage="dev+val",
                         family=f"E2-N{r.N}-W{r.W}-tol{r.tol}-{r.target}-{r.mode}",
                         window_start=DEV[0], window_end=VAL[1], n_configs=1,
                         status="done", decision="na",
                         note=f"事件平台；dev配对差 {r.dev_paired_mean:+.4%} (t={r.dev_paired_t:.2f})，"
                              f"val {r.val_paired_mean:+.4%}；[BUDGET] 平台描述性，不用于选参"))
    for r in port.itertuples(index=False):
        k += 1
        rows.append(dict(trial_id=f"BRK-P{k:04d}", kind="experiment",
                         stage=str(r.window), family=f"{r.arm}|N{r.N}|W{r.W}|{r.exit}",
                         window_start=WINDOWS[str(r.window)][0],
                         window_end=WINDOWS[str(r.window)][1], n_configs=1,
                         status="done", decision="na",
                         note=f"{r.label} CAGR {r.cagr:+.2%} MDD {r.mdd:+.1%} n={r.n_trades}"))
    for r in uni.itertuples(index=False):
        k += 1
        rows.append(dict(trial_id=f"BRK-U{k:04d}", kind="audit", stage=str(r.window),
                         family=f"unified|{r.label}", window_start=WINDOWS[str(r.window)][0],
                         window_end=WINDOWS[str(r.window)][1], n_configs=0,
                         status="done" if np.isfinite(getattr(r, "cagr", np.nan)) else "failed",
                         decision="na", note=f"统一引擎复核 {r.label}"))
    led = pd.DataFrame(rows)
    led["trials_cumulative"] = np.arange(1, len(led) + 1)
    led.to_csv(OUT / "11_试验台账.csv", index=False)
    n_exp = int((led["kind"] == "experiment").sum())
    log(f"台账 {len(led)} 行（experiment {n_exp}，audit {len(led)-n_exp}）")
    return led


def write_data_note(sc: dict) -> None:
    a = sc["aspce_identity"]
    h = sc["hfq_misuse"]
    lf = sc["limit_flags"]
    txt = f"""# 数据口径验证：`hfq = raw × factor` 的缺陷与本族口径

> 由 `02_代码/候选_突破回踩.py selfcheck` 生成（{sc['generated']}）。
> 只统计 2006-01-04..2020-12-31 主板行，2021+ 未读取。

## 1. 冻结层 `factor` 是总收益指数，`hfq = raw × factor` 不是价格

`factor_t = Π(1+pct_chg/100)`；`a_close_t = raw_close_t × factor_t / raw_close_t = factor_t`，
所以 A 空间的收盘日收益应当逐日等于官方 `pct_chg`。

| 口径 | 相邻日收益 vs `pct_chg` 最大绝对偏差 | 偏差 >1e-5 的行数 | 占比 |
|---|---:|---:|---:|
| A 空间 `a_close = factor`（本族采用） | **{a['max_err']:.3e}** | {a['n_over_1e-5']} / {a['n']:,} | {a['n_over_1e-5']/max(a['n'],1):.6%} |
| 冻结层 `hfq = raw × factor` | {h['max_err']:.3e} | {h['n_over_1e-5']} / {h['n']:,} | {h['share_over_1e-5']:.2%} |

- `hfq` 的日收益对官方 `pct_chg` 的回归斜率 **{h['slope_vs_pct']:.3f}**（正常日约为 `(1+r)²−1`）。
- 结论：任何跨日使用 `hfq_open/hfq_high/hfq_low/hfq_close` 的指标都会把价格收益算两遍；
  本族只用 `factor` 构造 A 空间，成交/涨跌停全部用 raw float64。

## 2. 涨跌停价：冻结层标记与 `费用与交易约束.limit_prices` 对拍

随机抽样 {lf['checked']:,} 个（日, 证券）单元，用 float64 `preclose` + 半入到分重算：

| 项 | 不一致数 |
|---|---:|
| 开盘涨停标记 | {lf['mismatch_up']} |
| 开盘跌停标记 | {lf['mismatch_down']} |

第一阶段审计 §M3 指出的 float32 `preclose` 四舍五入问题（8.96 → 8.95）在本族**不存在**：
面板全部 float64，成交判定直接调用 `limit_prices`。

## 3. 复牌、停牌、退市

- 停牌行：{sc['panel']['n_susp_rows']:,} 行（有行、零量），不参与信号触发、不可成交。
- 复牌首日不新开仓；持仓可卖，滑点加倍。
- 退市：`delist_date` 到达按最后有效收盘卖出；主口径不折扣。
- **44 次账户运行中退市强制退出 0 笔**（宇宙在 `delist_date` 前 30 个交易日已禁止新开仓，20 日持有很难跨到退市日），因此预注册的「最后价 ×0.7」折价敏感性**无对象、未单独运行**。
- 退市整理期（前 30 个交易日）不新开仓：这是父预注册规定的**未来信息**，照做并披露。

## 4. 截断重算（未来函数防线）

对 40 只随机证券、3 个截断日重算突破掩码，与全样本比较：

| N | 截断日 | 不一致单元 | 参考事件 | 切片事件 |
|---|---|---:|---:|---:|
"""
    for N, res in sc["truncation"].items():
        for r in res["rows"]:
            txt += (f"| {N} | {r['cutoff']} | {r['mismatch']} | {r['n_ref']} | {r['n_slice']} |\n")
    txt += f"""
## 5. 2021+ 护栏

- 事件最大日期：{sc['no_2021']['event_max_date']}；净值最大日期：{sc['no_2021']['nav_max_date']}。
- `EventBook.load(..., '2021-01-04')` 被拒绝：{sc['no_2021']['guard_rejects_2021']}。
- 面板加载函数对 `end >= 2021-01-01` 直接抛错。

## 6. 账户恒等

| 检查 | 结果 |
|---|---|
| `ΣPnL − ΔNAV`（主口径，三臂 × 两窗口） | 见 `10_自检.json: account_identity` |
| 逐日现金流残差上限 | 见 `05_组合_汇总.csv: cash_resid_max` |
| T+1 / 整股 / 涨停买入 / 跌停卖出 | 见 `10_自检.json: trade_checks` |
"""
    (OUT / "12_数据口径_验证.md").write_text(txt, encoding="utf-8")
    log("数据口径说明写入 12_数据口径_验证.md")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def load_events(P: Panel, det: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if det is not None:
        return det, pd.read_csv(OUT / "02_事件_汇总.csv")
    path = OUT / "01_事件_明细.parquet"
    if path.exists():
        return pd.read_parquet(path), pd.read_csv(OUT / "02_事件_汇总.csv")
    return cmd_events(P)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["panel", "events", "portfolio", "unified", "selfcheck", "all"])
    args = ap.parse_args(argv)
    _ensure_dirs()
    t0 = time.time()
    P = load_panel()
    if args.cmd == "panel":
        print(json.dumps(P.meta, ensure_ascii=False, indent=2))
        sc = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "prereg_sha256": prereg_sha(),
              "panel": P.meta, **_stream_stats(P), "limit_flags": _limit_flag_check(P)}
        print(json.dumps(sc, ensure_ascii=False, indent=2, default=float))
        return 0

    det = summ = plat = None
    if args.cmd in ("events", "portfolio", "all"):
        b = benchmarks(P)
        log("基准：" + "；".join(f"{r.bench}/{r.window} {r.cagr:+.2%}" for r in b.itertuples()))
    if args.cmd in ("events", "all"):
        det, summ = cmd_events(P)
        plat = platform_table(summ)
        decomposition(det, summ)
        log(f"平台 {len(plat):,} 格；dev 达标格 {int(plat['dev_ok'].sum())}；"
            f"dev+val 同向格 {int((plat['dev_ok'] & plat['val_ok']).sum())}")
    if args.cmd in ("portfolio", "all", "unified", "selfcheck"):
        det, summ = load_events(P, det)
        if plat is None:
            plat = pd.read_csv(OUT / "03_平台_参数.csv")
    port = uni = None
    if args.cmd in ("portfolio", "all"):
        port = cmd_portfolio(P, det)
    if args.cmd in ("unified", "all"):
        if port is None:
            port = pd.read_csv(OUT / "05_组合_汇总.csv")
        uni = cmd_unified_check(P, det, port)
    if args.cmd in ("selfcheck", "all"):
        if port is None:
            port = pd.read_csv(OUT / "05_组合_汇总.csv")
        if uni is None:
            uni = pd.read_csv(OUT / "09_统一引擎复核.csv")
        sc = cmd_selfcheck(P, det, summ, port, uni)
        write_data_note(sc)
    if args.cmd == "all":
        write_ledger(plat, port, uni)
    log(f"完成，用时 {time.time()-t0:.1f}s")
    (OUT / "_运行日志.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
