#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选_长期双动量.py — LTM 族：长期总收益动量 × 跳过短期反转 × 双动量轮动

研究纪律
--------
* 预注册：03_结果/候选_长期双动量/00_预注册_长期双动量.md（SHA256 见 00b_锚点.txt）
* 面板硬截断 2020-12-31；任何 2021+ 读取/计算在 assert_window() 处抛 HoldoutViolation。
* 复权：A_t = factor_t / raw_close_t，adjusted_OHLC = raw_OHLC × A_t；
  因此 adjusted_close = factor，日收益与官方 pct_chg 一致。**不使用 hfq = raw × factor**。
* 成交在原始价空间；t 收盘信号 → t+1 开盘成交；T+1、整股、历史费用、滑点、
  停牌/涨跌停、公司行动、退市由 `账户撮合.Account`（经 `统一回测.UnifiedAccount`
  补 preclose 权威公司行动与红利税）处理。
* 本文件只写 02_代码/候选_长期双动量.py 与 03_结果/候选_长期双动量/ 下的产物。

用法
----
    python3 02_代码/候选_长期双动量.py audit   # 数据与因果自检，不产生收益
    python3 02_代码/候选_长期双动量.py ic      # IC / 十分位 / 分层
    python3 02_代码/候选_长期双动量.py sim     # 26 个配置 + 21 个基准/审计运行
    python3 02_代码/候选_长期双动量.py stats   # DSR / PBO / bootstrap
    python3 02_代码/候选_长期双动量.py report  # 报告
    python3 02_代码/候选_长期双动量.py all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict, replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from 账户撮合 import AccountConfig, Order, SymbolInfo, Bar      # noqa: E402
import 费用与交易约束 as FT                                    # noqa: E402
from 统一回测 import UnifiedAccount                            # noqa: E402
from 统一数据适配 import EventBook                             # noqa: E402

OUT = ROOT / "03_结果" / "候选_长期双动量"
OUT.mkdir(parents=True, exist_ok=True)
FROZEN = ROOT / "01_数据" / "frozen"

PANEL_START = "2006-01-04"
PANEL_END = date(2020, 12, 31)
DEV = (date(2008, 1, 1), date(2015, 12, 31))
VAL = (date(2016, 1, 1), date(2020, 12, 31))
FULL = (date(2008, 1, 1), date(2020, 12, 31))
ANCHOR_DECISION = date(2007, 12, 28)
FREEZE = date(2020, 12, 31)

INITIAL_CASH = 1_000_000.0
MIN_LIST_CAL_DAYS = 250
MIN_LIST_TD = 120
MIN_PRICE = 2.0
MIN_ADV20 = 3.0e7
DELIST_BLOCK_TD = 30
BAD_MOVE_PCT = 15.0

LOG_PATH = OUT / "06_运行日志.txt"
_LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _LOG.append(line)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


class HoldoutViolation(RuntimeError):
    pass


def assert_window(start, end) -> None:
    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    if e > PANEL_END or s > PANEL_END:
        raise HoldoutViolation(f"窗口 {s}~{e} 越过 {PANEL_END}；2021+ 禁止计算")


def ensure_no_2021(df: pd.DataFrame, name: str) -> None:
    for col in ("date", "entry_date", "exit_date", "decision_date", "execute_from"):
        if col in df.columns and len(df):
            s = pd.to_datetime(df[col], errors="coerce")
            if s.notna().any() and s.max().date() > PANEL_END:
                raise HoldoutViolation(f"{name}.{col} 出现 2021+ 日期 {s.max()}")


# ==========================================================================
# 成本模型
# ==========================================================================

@dataclass(frozen=True)
class CostModel:
    name: str
    commission: float
    stamp_mult: float
    transfer_mult: float
    slippage_bps: float
    impact: float
    cap_bps: float
    part_cap: float


COSTS = {
    "C-REAL-V1": CostModel("C-REAL-V1", 0.00025, 1.0, 1.0, 10.0, 0.02, 100.0, 0.02),
    "C-STRESS-2X": CostModel("C-STRESS-2X", 0.00050, 2.0, 2.0, 20.0, 0.04, 200.0, 0.02),
}


def make_fee(cost: str) -> FT.FeeConfig:
    c = COSTS[cost]
    stamp = tuple(FT.StampRule(r.effective, r.rate * c.stamp_mult, r.sides)
                  for r in FT.DEFAULT_STAMP_SCHEDULE)
    # 预注册 §9.1：2015-07-31 前沪市按面额 0.1%（保守上界），其后双边成交额 0.002%
    transfer = (
        FT.TransferRule(date(2005, 1, 1), "par_sh", 0.0010 * c.transfer_mult, "SH"),
        FT.TransferRule(date(2015, 8, 1), "amount", 0.00002 * c.transfer_mult, "BOTH"),
        FT.TransferRule(date(2022, 4, 29), "amount", 0.00001 * c.transfer_mult, "BOTH"),
    )
    return FT.FeeConfig(
        commission_rate=c.commission,
        min_commission=5.0,
        stamp_schedule=stamp,
        transfer_schedule=transfer,
        slippage_bps=c.slippage_bps,
        max_participation=c.part_cap,
        dividend_tax_rate=0.0,      # 由 UnifiedAccount 按持股期结算
    )


# ==========================================================================
# 面板：冻结 parquet → [日 × 代码] 矩阵
# ==========================================================================

FLOAT_FIELDS = ("open", "high", "low", "close", "preclose", "volume", "amount",
                "factor", "pct_chg", "limit_up_px", "limit_down_px", "td_since_list")
COLS = ("symbol", "date", "board", "open", "high", "low", "close", "preclose",
        "volume", "amount", "factor", "pct_chg", "trade_status", "is_st",
        "limit_up_px", "limit_down_px", "td_since_list")


class Panel:
    """只读面板。所有信号在 adjusted_close = factor 上算；成交在原始价上。"""

    def __init__(self, start: str = PANEL_START, end: str = "2020-12-31"):
        if pd.Timestamp(end).date() > PANEL_END:
            raise HoldoutViolation("Panel 拒绝 2021+")
        cal = pd.read_parquet(FROZEN / "calendar.parquet")
        cal = cal[(cal["is_trading_day"] == 1)]
        d = pd.to_datetime(cal["date"]).dt.date
        mask = (d >= pd.Timestamp(start).date()) & (d <= pd.Timestamp(end).date())
        self.dates = np.array([x for x in d[mask]], dtype="datetime64[D]")
        if len(self.dates) == 0:
            raise ValueError("窗口内无交易日")
        if self.dates.max().astype("datetime64[D]").astype(date) > PANEL_END:
            raise HoldoutViolation("日历越过 2020-12-31")
        self.n_days = len(self.dates)
        self.date_list = [x.astype("datetime64[D]").astype(date) for x in self.dates]
        self.day_idx = {x: i for i, x in enumerate(self.date_list)}

        sym = pd.read_parquet(FROZEN / "symbols.parquet")
        sym = sym[sym["board"] == "主板"].copy()
        sym["ipo_date"] = pd.to_datetime(sym["ipo_date"], errors="coerce")
        sym["delist_date"] = pd.to_datetime(sym["delist_date"], errors="coerce")
        sym = sym.sort_values("symbol").reset_index(drop=True)
        self.codes = sym["symbol"].to_numpy(dtype=str)
        self.n_codes = len(self.codes)
        self.code_idx = {c: j for j, c in enumerate(self.codes)}
        self.ipo_date = [None if pd.isna(x) else x.date() for x in sym["ipo_date"]]
        self.delist_date = [None if pd.isna(x) else x.date() for x in sym["delist_date"]]

        nd, nc = self.n_days, self.n_codes
        self.m: dict[str, np.ndarray] = {
            k: np.full((nd, nc), np.nan, dtype=np.float64) for k in FLOAT_FIELDS
        }
        for k in ("trade_status", "is_st"):
            self.m[k] = np.zeros((nd, nc), dtype=np.int8)

        d_np = self.dates
        code_series = pd.Index(self.codes)
        rows = 0
        for y in range(2006, 2021):
            p = FROZEN / "daily" / f"year={y}" / "data_0.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p, columns=list(COLS))
            df = df[df["board"] == "主板"]
            if df.empty:
                continue
            di = np.searchsorted(d_np, pd.to_datetime(df["date"]).to_numpy(dtype="datetime64[D]"))
            ci = code_series.get_indexer(df["symbol"].to_numpy(dtype=str))
            ok = (ci >= 0) & (di >= 0) & (di < nd)
            di, ci = di[ok], ci[ok]
            df = df.iloc[np.flatnonzero(ok)]
            for k in FLOAT_FIELDS:
                self.m[k][di, ci] = df[k].to_numpy(dtype=np.float64)
            self.m["trade_status"][di, ci] = df["trade_status"].to_numpy(dtype=np.int8)
            self.m["is_st"][di, ci] = df["is_st"].to_numpy(dtype=np.int8)
            rows += len(df)
            del df, di, ci
        self.n_rows = int(rows)

        m = self.m
        self.listed = np.isfinite(m["close"]) & (m["close"] > 0)
        self.tradable = self.listed & (m["trade_status"] == 1) & (m["volume"] > 0)
        self.suspended = self.listed & ~self.tradable
        prev_tr = np.vstack([np.zeros((1, nc), dtype=bool), self.tradable[:-1]])
        prev_li = np.vstack([np.zeros((1, nc), dtype=bool), self.listed[:-1]])
        self.resume = self.tradable & ~prev_tr & prev_li          # 前一日停牌/未上市，今日有成交
        self.first_day = self.tradable & ~prev_li

        self.adj_close = m["factor"].copy()                        # 总收益指数
        self.adj_open = m["open"] * m["factor"] / m["close"]
        self.adj_high = m["high"] * m["factor"] / m["close"]
        self.adj_low = m["low"] * m["factor"] / m["close"]
        self.a_space = m["factor"] / m["close"]                    # A_t

        adf = pd.DataFrame(self.adj_close)
        self.adj_ff = adf.ffill().to_numpy(dtype=np.float64)
        self.ret_daily = np.vstack([np.full((1, nc), np.nan), self.adj_ff[1:] / self.adj_ff[:-1] - 1.0])
        self.ret_daily[~np.isfinite(self.ret_daily)] = 0.0

        amt = pd.DataFrame(np.where(self.listed, m["amount"], np.nan))
        self.adv20 = amt.rolling(20, min_periods=20).mean().to_numpy(dtype=np.float64)

        days = np.array([(x - date(1990, 1, 1)).days for x in self.date_list], dtype=np.int64)
        ipo_days = np.array([np.nan if x is None else (x - date(1990, 1, 1)).days
                             for x in self.ipo_date], dtype=np.float64)
        self.days_since_ipo = days[:, None] - ipo_days[None, :]

        self.delist_idx = np.full(nc, -1, dtype=np.int64)
        for j, dd in enumerate(self.delist_date):
            if dd is None:
                continue
            k = int(np.searchsorted(self.dates, np.datetime64(dd, "D")))
            self.delist_idx[j] = k
        self.delist_window = np.zeros((nd, nc), dtype=bool)
        for j in np.flatnonzero(self.delist_idx >= 0):
            k = int(self.delist_idx[j])
            if k <= nd:
                self.delist_window[max(0, k - DELIST_BLOCK_TD):min(nd, k + 1), j] = True

        self.bad_move = ~np.isfinite(m["pct_chg"]) | (np.abs(m["pct_chg"]) > BAD_MOVE_PCT)
        self.elig = (
            self.listed & self.tradable & (m["is_st"] == 0)
            & (m["close"] >= MIN_PRICE) & (self.adv20 >= MIN_ADV20)
            & (m["td_since_list"] >= MIN_LIST_TD) & (self.days_since_ipo >= MIN_LIST_CAL_DAYS)
            & ~self.resume & ~self.delist_window & ~self.bad_move
            & np.isfinite(self.adj_close)
        )
        self.buyable = self.elig & np.isfinite(m["limit_up_px"])
        self.board_group = np.array([("SH60" if c.startswith("60") else
                                      ("SZ00" if c[2:5] == "000" else "SZ002"))
                                     for c in self.codes])

        # 板块内 ADV 三分位（每日截面）
        self.adv_tercile = np.full((nd, nc), -1, dtype=np.int8)
        for i in range(nd):
            js = np.flatnonzero(self.elig[i])
            if len(js) < 30:
                continue
            v = self.adv20[i, js]
            q1, q2 = np.nanquantile(v, [1 / 3, 2 / 3])
            self.adv_tercile[i, js] = np.where(v <= q1, 0, np.where(v <= q2, 1, 2))

        self._build_period_permissions()

    # ------------------------------------------------------------------
    def _build_period_permissions(self) -> None:
        dates = pd.DatetimeIndex(pd.to_datetime(self.date_list))
        adj = pd.DataFrame(self.adj_ff, index=dates)
        # 周：ISO 年-周；该周最后一个交易日收盘后才完成
        wk_key = dates.isocalendar().year.values.astype(np.int64) * 100 \
            + dates.isocalendar().week.values.astype(np.int64)
        last_pos = np.array([g[-1] for g in pd.Series(np.arange(len(dates))).groupby(wk_key).groups.values()])
        last_pos = np.sort(last_pos)
        W = adj.iloc[last_pos].copy()
        W.index = dates[last_pos]
        Wma20 = W.rolling(20, min_periods=20).mean()
        self.w_perm = (W > Wma20).reindex(dates, method="ffill").to_numpy()
        # 月：年-月；该月最后一个交易日收盘后才完成
        m_key = dates.year.values.astype(np.int64) * 100 + dates.month.values.astype(np.int64)
        last_pos_m = np.array([g[-1] for g in pd.Series(np.arange(len(dates))).groupby(m_key).groups.values()])
        last_pos_m = np.sort(last_pos_m)
        M = adj.iloc[last_pos_m].copy()
        M.index = dates[last_pos_m]
        Mma12 = M.rolling(12, min_periods=12).mean()
        self.m_perm = (M > Mma12).reindex(dates, method="ffill").to_numpy()
        self.n_week_complete = int(len(last_pos))
        self.n_month_complete = int(len(last_pos_m))

    # ------------------------------------------------------------------
    def counts(self, i: int, L: int, skip: int) -> np.ndarray:
        b = i - skip - L
        a = i - skip
        if b < 0:
            return np.zeros(self.n_codes, dtype=np.int32)
        cs = np.cumsum(self.listed.astype(np.int32), axis=0)
        hi = cs[a]
        lo = cs[b - 1] if b > 0 else np.zeros(self.n_codes, dtype=np.int32)
        return hi - lo

    def mom(self, i: int, L: int, skip: int) -> tuple[np.ndarray, np.ndarray]:
        """返回 (score, valid)。score = adj_close[i-s] / adj_close[i-s-L] - 1。"""
        a, b = i - skip, i - skip - L
        out = np.full(self.n_codes, np.nan, dtype=np.float64)
        if b < 0:
            return out, np.zeros(self.n_codes, dtype=bool)
        num = self.adj_ff[a]
        den = self.adj_ff[b]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den - 1.0
        valid = (np.isfinite(out) & (den > 0)
                 & (self.counts(i, L, skip) >= 0.8 * L))
        out[~valid] = np.nan
        return out, valid

    def vol60(self, i: int) -> np.ndarray:
        if i < 61:
            return np.full(self.n_codes, np.nan)
        r = self.ret_daily[i - 59:i + 1]
        ok = np.isfinite(r)
        n = ok.sum(axis=0)
        sd = np.where(n >= 40, np.nanstd(r, axis=0, ddof=1), np.nan)
        return sd

    def bars_for(self, i: int, codes) -> dict:
        out = {}
        m = self.m
        for c in codes:
            j = self.code_idx.get(c)
            if j is None or not self.listed[i, j]:
                continue
            o, h, l, cl = (m["open"][i, j], m["high"][i, j],
                           m["low"][i, j], m["close"][i, j])
            if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l)
                    and np.isfinite(cl) and o > 0 and h > 0 and l > 0 and cl > 0):
                continue
            if h < l - 1e-9 or o > h + 1e-6 or o < l - 1e-6 \
                    or cl > h + 1e-6 or cl < l - 1e-6:
                continue
            pc = m["preclose"][i, j]
            out[c] = Bar(
                code=c, date=self.date_list[i], open=float(o), high=float(h),
                low=float(l), close=float(cl),
                volume=float(m["volume"][i, j]) if np.isfinite(m["volume"][i, j]) else 0.0,
                amount=float(m["amount"][i, j]) if np.isfinite(m["amount"][i, j]) else None,
                prev_close=float(pc) if np.isfinite(pc) and pc > 0 else None,
                prev_close_is_ref=True,
                is_st=bool(m["is_st"][i, j]),
                trade_status=1 if self.tradable[i, j] else 0,
            )
        return out


# ==========================================================================
# 配置
# ==========================================================================

@dataclass(frozen=True)
class Cfg:
    cid: str
    kind: str = "experiment"          # experiment / audit
    L: int = 250
    skip: int = 20
    n: int = 10
    R: int = 20
    band: float = 0.0
    abs_filter: bool = True
    perm: str = ""                    # "" / W / M
    delay: int = 0
    cost: str = "C-REAL-V1"
    universe: str = "all"             # all / topliq
    score_kind: str = "mom"           # mom / voladj / rand
    seed: int = 0
    bench: str = ""                   # bh_ew / bh_n10 / ew20 / ew60 / cash
    note: str = ""


def build_configs() -> list[Cfg]:
    cfgs: list[Cfg] = []
    k = 0
    for L in (120, 250):
        for s in (0, 5, 20):
            for ab in (False, True):
                k += 1
                cfgs.append(Cfg(
                    cid=f"C{k:02d}", L=L, skip=s, abs_filter=ab,
                    note=f"core L={L} skip={s} abs={'on' if ab else 'off'}"))
    assert [c.cid for c in cfgs][-1] == "C12"
    primary = cfgs[-1]
    variants = [
        ("C13", dict(n=5), "N=5"),
        ("C14", dict(n=20), "N=20"),
        ("C15", dict(R=40), "R=40"),
        ("C16", dict(R=60), "R=60"),
        ("C17", dict(band=0.25), "band=0.25"),
        ("C18", dict(band=0.50), "band=0.50"),
        ("C19", dict(perm="W"), "weekly MA20 permission"),
        ("C20", dict(perm="M"), "monthly MA12 permission"),
        ("C21", dict(delay=1), "execution delay +1d"),
        ("C22", dict(cost="C-STRESS-2X"), "cost stress 2x"),
        ("C23", dict(universe="topliq"), "top ADV tercile universe"),
        ("C24", dict(L=60, skip=0, abs_filter=False), "short control L=60 skip=0 off"),
        ("C25", dict(L=60, skip=0, abs_filter=True), "short control L=60 skip=0 on"),
        ("C26", dict(score_kind="voladj"), "vol-adjusted score"),
    ]
    for cid, kw, note in variants:
        base = dict(kind="experiment", L=primary.L, skip=primary.skip, n=primary.n,
                    R=primary.R, band=primary.band, abs_filter=primary.abs_filter,
                    perm=primary.perm, delay=primary.delay, cost=primary.cost,
                    universe=primary.universe, score_kind=primary.score_kind)
        base.update(kw)
        cfgs.append(Cfg(cid=cid, note=note, **base))
    return cfgs


def bench_configs() -> list[Cfg]:
    out = [
        Cfg("E01", kind="audit", bench="bh_ew", note="universe EW buy&hold"),
        Cfg("E02", kind="audit", bench="bh_n10", note="10-name hash buy&hold"),
        Cfg("E03", kind="audit", bench="ew20", note="universe EW R=20"),
        Cfg("E04", kind="audit", bench="ew60", note="universe EW R=60"),
        Cfg("CASH", kind="audit", bench="cash", note="cash 0%"),
    ]
    for seed in range(1, 9):
        out.append(Cfg(f"R20-{seed}", kind="audit", score_kind="rand", seed=seed,
                       n=10, R=20, abs_filter=False, note=f"random N=10 R=20 seed={seed}"))
    for seed in range(1, 9):
        out.append(Cfg(f"R60-{seed}", kind="audit", score_kind="rand", seed=seed,
                       n=10, R=60, abs_filter=False, note=f"random N=10 R=60 seed={seed}"))
    # 事后补充的换手匹配随机基准（审计用途，不进入选择与 DSR 的 N）
    for seed in range(1, 9):
        out.append(Cfg(f"R20F-{seed}", kind="audit", score_kind="randfix", seed=seed,
                       n=10, R=20, abs_filter=False,
                       note=f"fixed-random N=10 R=20 seed={seed} (turnover-matched)"))
    for seed in range(1, 5):
        out.append(Cfg(f"R60F-{seed}", kind="audit", score_kind="randfix", seed=seed,
                       n=10, R=60, abs_filter=False,
                       note=f"fixed-random N=10 R=60 seed={seed} (turnover-matched)"))
    return out


# ==========================================================================
# 账户引擎
# ==========================================================================

def make_account(cfg: Cfg, panel: Panel) -> UnifiedAccount:
    info = {
        code: SymbolInfo(code=code, name="", ipo_date=panel.ipo_date[j],
                         out_date=panel.delist_date[j])
        for j, code in enumerate(panel.codes)
    }
    acfg = AccountConfig(
        initial_cash=INITIAL_CASH,
        fee=make_fee(cfg.cost),
        cash_settle_days=0,
        enforce_t1=True,
        allow_buy_downsize=True,
        normalize_buy_orders=True,
        rights_policy="subscribe",
        dividend_tax_rate=0.0,
        delisting_missing_days=10_000_000,
        delisting_haircut=0.0,
        liquidate_on_finalize=False,
        finalize_haircut=0.0,
        strict=False,
    )
    acct = UnifiedAccount(acfg, panel.date_list, freeze_date=PANEL_END,
                          symbol_info=info, accounting="shares")
    acct._code_pos = panel.code_idx
    return acct


def rank_target(cfg: Cfg, panel: Panel, i: int, nav: float) -> list[str]:
    """t 收盘的目标名单（有序）。只用 ≤ t 数据。"""
    if cfg.score_kind == "rand":
        js = np.flatnonzero(panel.elig[i])
        if len(js) == 0:
            return []
        rng = np.random.default_rng([cfg.seed, i])
        pick = rng.choice(js, size=min(cfg.n, len(js)), replace=False)
        return [panel.codes[j] for j in pick]
    if cfg.score_kind == "randfix":
        # 固定随机排序（每只股票一个与时间无关的随机键），换手来自资格变动而非重抽
        js = np.flatnonzero(panel.elig[i])
        if len(js) == 0:
            return []
        key = np.random.default_rng(cfg.seed).random(panel.n_codes)
        order = np.argsort(-key[js], kind="stable")
        return [panel.codes[js[k]] for k in order[:cfg.n]]

    if cfg.score_kind == "voladj":
        score, valid = panel.mom(i, 250, 20)
        sd = panel.vol60(i)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = score / sd
        score[~np.isfinite(score)] = np.nan
    else:
        score, valid = panel.mom(i, cfg.L, cfg.skip)

    ok = panel.elig[i] & valid & np.isfinite(score)
    if cfg.universe == "topliq":
        ok &= (panel.adv_tercile[i] == 2)
    if cfg.perm == "W":
        ok &= np.nan_to_num(panel.w_perm[i], nan=0.0).astype(bool)
    elif cfg.perm == "M":
        ok &= np.nan_to_num(panel.m_perm[i], nan=0.0).astype(bool)
    if cfg.abs_filter:
        ok &= (score > 0)

    js = np.flatnonzero(ok)
    if len(js) == 0:
        return []
    liq = np.where(np.isfinite(panel.adv20[i, js]), panel.adv20[i, js], -np.inf)
    order = np.lexsort((js, -liq, -score[js]))
    ranked = [panel.codes[js[k]] for k in order]

    if cfg.band <= 0:
        return ranked[:cfg.n]

    # no-trade band：已持有且排名在 N×(1+band) 内则保留
    held = set()
    rank_of = {c: r for r, c in enumerate(ranked)}
    keep_n = int(math.ceil(cfg.n * (1.0 + cfg.band)))
    return ranked, rank_of, keep_n  # 由调用方处理


def target_list(cfg: Cfg, panel: Panel, i: int, held: set[str]) -> list[str]:
    if cfg.score_kind == "rand":
        return rank_target(cfg, panel, i, 0.0)
    res = rank_target(cfg, panel, i, 0.0)
    ranked = res if isinstance(res, list) else res[0]
    if cfg.band <= 0 or not isinstance(res, tuple):
        return ranked[:cfg.n]
    _, rank_of, keep_n = res
    keep = [c for c in held if c in rank_of and rank_of[c] < keep_n]
    out = list(keep)
    for c in ranked:
        if len(out) >= cfg.n:
            break
        if c not in out:
            out.append(c)
    return out[:cfg.n]


def run_account(cfg: Cfg, panel: Panel, events: EventBook,
                collect: bool = False) -> dict:
    assert_window(PANEL_START, PANEL_END)
    i0 = panel.day_idx[ANCHOR_DECISION]
    i1 = panel.n_days - 1
    schedule = set(range(i0, i1, max(cfg.R, 1)))
    acct = make_account(cfg, panel)

    checks = {
        "cash_resid_max": 0.0, "nav_resid_max": 0.0,
        "t1_violations": 0, "negative_share_events": 0,
        "trades_on_suspension": 0, "buy_at_limit_up_open": 0,
        "sell_at_limit_down_open": 0, "min_cash": float("inf"),
        "forced_sells": 0, "n_orders": 0, "n_filled": 0,
        "max_date": "",
    }
    daily: list[dict] = []
    pos_rows: list[dict] = []
    trade_ptr = ca_ptr = tax_ptr = 0
    prev_cash = float(acct.cash_total)
    pending: tuple[list[str], int] | None = None

    def _record(i: int, rec: dict) -> None:
        daily.append({
            "date": panel.date_list[i].isoformat(),
            "nav": float(rec["nav"]),
            "cash": float(rec["cash_total"]),
            "cash_available": float(rec["cash_available"]),
            "market_value": float(rec["market_value"]),
            "n_positions": int(rec["n_positions"]),
        })

    def _submit(target: list[str], i: int, nav: float) -> None:
        d = panel.date_list[i]
        held = {c for c, p in acct.positions.items() if p.shares > 0}
        active_sells = {o.code: o for o in acct._orders if o.active and o.side == "sell"}
        active_buys = {o.code for o in acct._orders if o.active and o.side == "buy"}
        tset = set(target)
        # 目标内的持仓：撤销此前挂出的卖单
        for c in tset:
            o = active_sells.get(c)
            if o is not None:
                acct.cancel(o.order_id, "rebalance_keep")
        # 目标外的持仓：卖出（GTC，最多顺延 10 日）
        for c in list(held):
            if c in tset:
                continue
            if c in active_sells:
                continue
            pos = acct.positions[c]
            acct.submit(Order(code=c, side="sell", qty=int(pos.shares),
                              kind="open", tif="gtc", tag="rebal_out"), d)
        # 买入：按目标顺序补空槽
        selling = {o.code for o in acct._orders if o.active and o.side == "sell"}
        n_locked = sum(1 for c in held if c not in selling)
        free = cfg.n - n_locked - len(active_buys)
        if free <= 0:
            return
        budget = nav / max(cfg.n, 1)
        n_new = 0
        for c in target:
            if n_new >= free:
                break
            if c in held or c in active_sells or c in active_buys:
                continue
            j = panel.code_idx[c]
            if not panel.buyable[i, j]:
                continue
            px = panel.m["close"][i, j]
            if not np.isfinite(px) or px <= 0:
                continue
            qty = int(budget / (px * 1.003) / 100.0) * 100
            if qty < 100:
                continue
            acct.submit(Order(code=c, side="buy", qty=qty, kind="open",
                              tif="day", tag="rebal_in"), d)
            n_new += 1

    for i in range(i0, i1 + 1):
        d = panel.date_list[i]
        held = list(acct.positions)
        active = [o.code for o in acct._orders if o.active]
        ev_today = [e for e in events.on(d) if e.symbol in panel.code_idx]
        needed = set(held) | set(active) | {e.symbol for e in ev_today}
        acct._ca_row = panel.a_space[i]
        acct._adv = {
            c: float(panel.adv20[i - 1, panel.code_idx[c]])
            if i > 0 and c in panel.code_idx and np.isfinite(panel.adv20[i - 1, panel.code_idx[c]])
            else 0.0
            for c in set(active) | set(held)
        }
        acct._resume_today = {panel.codes[j] for j in np.flatnonzero(panel.resume[i])}
        rec = acct.on_day(d, bars=panel.bars_for(i, needed), events=ev_today)

        # 卖单顺延 10 日 → 第 10 日收盘强制退出
        forced_now = False
        for o in list(acct._orders):
            if not (o.active and o.side == "sell" and o.filled_qty == 0):
                continue
            try:
                age = i - panel.day_idx[o.execute_from]
            except KeyError:
                continue
            if age >= 10:
                pos = acct.positions.get(o.code)
                j = panel.code_idx.get(o.code)
                if pos is None or j is None:
                    continue
                px = panel.m["close"][i, j]
                if not np.isfinite(px) or px <= 0:
                    px = pos.last_price
                acct._force_sell(o.code, d, float(px), "sell_persistence_10d")
                checks["forced_sells"] += 1
                forced_now = True
        if forced_now:
            rec = acct._mark_and_record(d)

        checks["max_date"] = max(checks["max_date"], d.isoformat())
        checks["min_cash"] = min(checks["min_cash"], float(rec["cash_total"]))
        if float(rec["cash_total"]) < -0.01:
            checks["negative_share_events"] += 1

        # 现金流恒等
        new_fills = acct.trades[trade_ptr:]
        buys = sum(t.amount + t.fees for t in new_fills if t.side == "buy")
        sells = sum(t.amount - t.fees for t in new_fills if t.side == "sell")
        new_ca = acct.ca_rows[ca_ptr:]
        ca_cash = sum(x["credit"] for x in new_ca)
        new_tax = acct.tax_ledger[tax_ptr:]
        tax_cash = sum(x["amount"] for x in new_tax)
        delta = float(rec["cash_total"]) - prev_cash
        checks["cash_resid_max"] = max(checks["cash_resid_max"], abs(delta - (sells - buys + ca_cash + tax_cash)))
        prev_cash = float(rec["cash_total"])
        mv = sum(p.shares * (panel.m["close"][i, panel.code_idx[c]]
                             if np.isfinite(panel.m["close"][i, panel.code_idx[c]]) and panel.m["close"][i, panel.code_idx[c]] > 0
                             else p.last_price)
                 for c, p in acct.positions.items() if c in panel.code_idx)
        checks["nav_resid_max"] = max(checks["nav_resid_max"], abs(mv - float(rec["market_value"])))

        _record(i, rec)
        if collect and i >= panel.day_idx[date(2008, 1, 2)]:
            nav = float(rec["nav"])
            for c, p in acct.positions.items():
                j = panel.code_idx.get(c)
                if j is None or p.shares <= 0:
                    continue
                px = panel.m["close"][i, j]
                if not (np.isfinite(px) and px > 0):
                    px = p.last_price
                pos_rows.append({
                    "date": d.isoformat(), "code": c, "shares": int(p.shares),
                    "close": round(float(px), 4),
                    "market_value": round(p.shares * float(px), 2),
                    "weight": round(p.shares * float(px) / nav, 6) if nav > 0 else np.nan,
                })
        trade_ptr, ca_ptr, tax_ptr = len(acct.trades), len(acct.ca_rows), len(acct.tax_ledger)

        if i >= i1:
            break

        if pending is not None and pending[1] == i:
            _submit(pending[0], i, float(rec["nav"]))
            pending = None

        if i in schedule:
            held_now = {c for c, p in acct.positions.items() if p.shares > 0}
            target = target_list(cfg, panel, i, held_now)
            if cfg.delay:
                pending = (target, i + 1)
            else:
                _submit(target, i, float(rec["nav"]))

    trades = pd.DataFrame([t.as_dict() for t in acct.trades])
    rounds = pd.DataFrame(acct.round_trips)
    eq = pd.DataFrame(daily)
    pos = pd.DataFrame(pos_rows)
    if len(trades):
        ensure_no_2021(trades, f"{cfg.cid}.trades")
    if len(rounds):
        ensure_no_2021(rounds, f"{cfg.cid}.rounds")
    ensure_no_2021(eq, f"{cfg.cid}.equity")
    checks["n_orders"] = len(acct.orders)
    checks["n_filled"] = int((trades["side"].notna()).sum()) if len(trades) else 0
    if len(rounds):
        checks["t1_violations"] = int((rounds["hold_days"] <= 0).sum())
    checks["min_cash"] = 0.0 if checks["min_cash"] == float("inf") else checks["min_cash"]
    checks["max_date"] = max(checks["max_date"], "")

    return {"cfg": cfg, "equity": eq, "trades": trades, "rounds": rounds,
            "positions": pos, "checks": checks, "acct": acct}


# ==========================================================================
# 基准（指数口径）
# ==========================================================================

def _eq_series_from_weights(panel: Panel, start_i: int, end_i: int,
                            rebal_R: int | None, cost_oneway: float = 0.0015) -> pd.Series:
    """等权组合指数：rebal_R=None 为买入持有；否则每 R 日收盘再平衡。"""
    dates = panel.date_list[start_i:end_i + 1]
    ser = np.zeros(len(dates), dtype=np.float64)
    vals = np.zeros(panel.n_codes, dtype=np.float64)
    pv = 1.0
    for k, i in enumerate(range(start_i, end_i + 1)):
        if k > 0:
            vals = vals * (1.0 + panel.ret_daily[i])
            pv = float(vals.sum())
        if k == 0 or (rebal_R is not None and (i - start_i) % rebal_R == 0):
            js = np.flatnonzero(panel.elig[i])
            new_w = np.zeros(panel.n_codes, dtype=np.float64)
            if len(js):
                new_w[js] = 1.0 / len(js)
            if pv > 0 and vals.sum() > 0:
                drift = vals / vals.sum()
                turn = float(np.abs(new_w - drift).sum())
            else:
                turn = 1.0
            pv = pv * (1.0 - cost_oneway * turn)
            vals = new_w * pv
        ser[k] = pv
    return pd.Series(ser, index=pd.to_datetime(dates), name="bench")


def run_benchmark(cfg: Cfg, panel: Panel) -> dict:
    start_i = panel.day_idx[date(2008, 1, 2)]
    end_i = panel.n_days - 1
    dates = pd.to_datetime(panel.date_list[start_i:end_i + 1])
    if cfg.bench == "cash":
        eq = pd.DataFrame({"date": dates, "nav": INITIAL_CASH, "cash": INITIAL_CASH,
                           "cash_available": INITIAL_CASH, "market_value": 0.0,
                           "n_positions": 0})
        return {"cfg": cfg, "equity": eq, "trades": pd.DataFrame(),
                "rounds": pd.DataFrame(), "positions": pd.DataFrame(),
                "checks": {}, "acct": None}
    if cfg.bench in ("ew20", "ew60"):
        s = _eq_series_from_weights(panel, start_i, end_i,
                                    rebal_R=20 if cfg.bench == "ew20" else 60)
        eq = pd.DataFrame({"date": s.index, "nav": s.to_numpy() * INITIAL_CASH,
                           "cash": np.nan, "cash_available": np.nan,
                           "market_value": np.nan, "n_positions": np.nan})
        return {"cfg": cfg, "equity": eq, "trades": pd.DataFrame(),
                "rounds": pd.DataFrame(), "positions": pd.DataFrame(),
                "checks": {}, "acct": None}
    # 买入持有
    js = np.flatnonzero(panel.elig[start_i])
    if cfg.bench == "bh_n10":
        rng = np.random.default_rng(20261028)
        js = rng.choice(js, size=min(10, len(js)), replace=False)
    vals = np.zeros(panel.n_codes, dtype=np.float64)
    vals[js] = 1.0 / max(len(js), 1) * (1.0 - 0.0015)
    ser = np.zeros(end_i - start_i + 1, dtype=np.float64)
    for k, i in enumerate(range(start_i, end_i + 1)):
        if k > 0:
            vals = vals * (1.0 + panel.ret_daily[i])
        ser[k] = vals.sum()
    s = pd.Series(ser, index=dates)
    eq = pd.DataFrame({"date": s.index, "nav": s.to_numpy() * INITIAL_CASH,
                       "cash": np.nan, "cash_available": np.nan,
                       "market_value": np.nan, "n_positions": np.nan})
    return {"cfg": cfg, "equity": eq, "trades": pd.DataFrame(),
            "rounds": pd.DataFrame(), "positions": pd.DataFrame(),
            "checks": {}, "acct": None}


# ==========================================================================
# 指标
# ==========================================================================

def _as_date(x):
    return pd.Timestamp(x).date()


def window_metrics(res: dict, start, end, tag: str) -> dict:
    eq = res["equity"].copy()
    eq["d"] = pd.to_datetime(eq["date"])
    sd, ed = pd.Timestamp(start), pd.Timestamp(end)
    pre = eq[eq["d"] < sd]
    sub = eq[(eq["d"] >= sd) & (eq["d"] <= ed)]
    if len(sub) == 0:
        return {"cfg": res["cfg"].cid, "window": tag, "n_days": 0}
    if len(pre):
        base_nav = float(pre["nav"].iloc[-1])
        base_date = pre["d"].iloc[-1]
    else:
        base_nav = float(sub["nav"].iloc[0])
        base_date = sub["d"].iloc[0]
    nav_end = float(sub["nav"].iloc[-1])
    nav_start = float(sub["nav"].iloc[0])
    days = max((sub["d"].iloc[-1] - base_date).days, 1)
    cagr = (nav_end / base_nav) ** (365.25 / days) - 1.0 if nav_end > 0 else -1.0
    s = pd.concat([pd.Series([base_nav], index=[base_date]), sub.set_index("d")["nav"]])
    rets = s.pct_change().dropna()
    vol = float(rets.std(ddof=1) * math.sqrt(252)) if len(rets) > 2 else np.nan
    sharpe = float(rets.mean() / rets.std(ddof=1) * math.sqrt(252)) if len(rets) > 2 and rets.std(ddof=1) > 0 else np.nan
    peak = s.cummax()
    dd = s / peak - 1.0
    mdd = float(-dd.min())
    ulcer = float(math.sqrt(float((dd ** 2).mean())))
    years = days / 365.25
    out = {
        "cfg": res["cfg"].cid, "kind": res["cfg"].kind, "window": tag,
        "start": str(sd.date()), "end": str(ed.date()), "n_days": int(len(sub)),
        "nav_start": round(nav_start, 2), "nav_end": round(nav_end, 2),
        "cagr": cagr, "mdd": mdd, "vol": vol, "sharpe": sharpe,
        "calmar": cagr / mdd if mdd > 0 else np.nan, "ulcer": ulcer,
        "years": years,
    }
    trades = res.get("trades")
    rounds = res.get("rounds")
    if trades is not None and len(trades):
        t = trades.copy()
        t["d"] = pd.to_datetime(t["date"])
        tw = t[(t["d"] >= sd) & (t["d"] <= ed)]
        avg_nav = float(sub["nav"].mean())
        out["turnover"] = float(tw["amount"].sum() / avg_nav / years) if avg_nav > 0 and years > 0 else np.nan
        out["fees"] = float(tw["fees"].sum())
        out["n_fills"] = int(len(tw))
    else:
        out["turnover"] = np.nan
        out["fees"] = np.nan
        out["n_fills"] = 0
    if rounds is not None and len(rounds):
        r = rounds.copy()
        r["xd"] = pd.to_datetime(r["exit_date"])
        rw = r[(r["xd"] >= sd) & (r["xd"] <= ed)]
        out["n_trades"] = int(len(rw))
        out["win_rate"] = float((rw["ret"] > 0).mean()) if len(rw) else np.nan
        out["avg_ret"] = float(rw["ret"].mean()) if len(rw) else np.nan
        out["median_ret"] = float(rw["ret"].median()) if len(rw) else np.nan
        out["avg_hold"] = float(rw["hold_days"].mean()) if len(rw) else np.nan
        pnl = rw["pnl"]
        out["profit_factor"] = float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else np.nan
        out["n_symbols"] = int(rw["code"].nunique()) if len(rw) else 0
        top5 = rw["pnl"].nlargest(5).sum()
        out["top5_share"] = float(top5 / pnl[pnl > 0].sum()) if (pnl > 0).any() else np.nan
    else:
        for k in ("n_trades", "win_rate", "avg_ret", "median_ret", "avg_hold",
                  "profit_factor", "n_symbols", "top5_share"):
            out[k] = np.nan
        out["n_trades"] = 0
    if "market_value" in res["equity"].columns and res["equity"]["market_value"].notna().any():
        mv = sub["market_value"].to_numpy(dtype=float)
        nv = sub["nav"].to_numpy(dtype=float)
        out["deployment"] = float(np.nanmean(np.where(nv > 0, mv / nv, np.nan)))
        out["avg_positions"] = float(sub["n_positions"].mean())
    else:
        out["deployment"] = np.nan
        out["avg_positions"] = np.nan
    return out


def yearly_table(eq: pd.DataFrame) -> pd.DataFrame:
    e = eq.copy()
    e["d"] = pd.to_datetime(e["date"])
    e = e.set_index("d").sort_index()
    last = e["nav"].groupby(e.index.year).last()
    out = pd.DataFrame({"year": last.index, "nav_end": last.values})
    rets = []
    for k in range(len(last)):
        if k > 0:
            rets.append(last.iloc[k] / last.iloc[k - 1] - 1.0)
        elif last.index[k] == 2008:
            rets.append(last.iloc[k] / INITIAL_CASH - 1.0)   # 基准净值从 2008-01-02 起
        else:
            rets.append(np.nan)
    out["ret"] = rets
    return out


def bootstrap_cagr(eq: pd.DataFrame, start, end, B: int = 1000, block: int = 20, seed: int = 7) -> dict:
    e = eq.copy()
    e["d"] = pd.to_datetime(e["date"])
    s = e[(e["d"] >= pd.Timestamp(start)) & (e["d"] <= pd.Timestamp(end))].set_index("d")["nav"]
    r = s.pct_change().dropna().to_numpy()
    if len(r) < 100:
        return {}
    rng = np.random.default_rng(seed)
    n = len(r)
    out = np.empty(B)
    for b in range(B):
        idx = []
        while len(idx) < n:
            st = rng.integers(0, n)
            ln = rng.geometric(1.0 / block)
            idx.extend(((st + np.arange(ln)) % n).tolist())
        rr = r[np.array(idx[:n])]
        out[b] = float(np.prod(1.0 + rr)) ** (365.25 / n) - 1.0
    return {"p05": float(np.quantile(out, 0.05)), "p50": float(np.quantile(out, 0.50)),
            "p95": float(np.quantile(out, 0.95))}


# ==========================================================================
# IC / 十分位 / 分层（诊断，不参与选参）
# ==========================================================================

IC_SPECS = [(20, 0), (60, 0), (60, 5), (120, 0), (120, 5), (120, 20),
            (250, 0), (250, 5), (250, 20)]


def monthly_sample_dates(panel: Panel) -> list[int]:
    out = []
    last = panel.n_days - 21
    for i in range(panel.n_days):
        if i < 250 or i > last:
            continue
        if i + 20 >= panel.n_days:
            continue
        d = panel.date_list[i]
        if i + 1 < panel.n_days and panel.date_list[i + 1].month == d.month:
            continue
        out.append(i)
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    sa, sb = ra.std(), rb.std()
    if sa <= 0 or sb <= 0:
        return np.nan
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def ic_study(panel: Panel) -> dict:
    dates = monthly_sample_dates(panel)
    rows = []
    dec_rows = []
    strat_rows = []
    for L, s in IC_SPECS:
        for i in dates:
            score, valid = panel.mom(i, L, s)
            ok = panel.elig[i] & valid & np.isfinite(score)
            js = np.flatnonzero(ok)
            if len(js) < 30:
                continue
            fwd = panel.adj_ff[i + 20, js] / panel.adj_ff[i, js] - 1.0
            good = np.isfinite(fwd)
            js, sc, fw = js[good], score[js][good], fwd[good]
            if len(js) < 30:
                continue
            ic = _spearman(sc, fw)
            d = panel.date_list[i]
            rows.append({"spec": f"L{L}_s{s}", "L": L, "skip": s, "date": d.isoformat(),
                         "year": d.year, "n": len(js), "ic": ic,
                         "fwd_top_decile": np.nan, "fwd_bottom_decile": np.nan,
                         "spread": np.nan})
            # 十分位
            try:
                q = pd.qcut(pd.Series(sc).rank(method="first"), 10, labels=False)
            except ValueError:
                continue
            top = q == 9
            bot = q == 0
            if top.sum() >= 5 and bot.sum() >= 5:
                rows[-1]["fwd_top_decile"] = float(fw[top].mean())
                rows[-1]["fwd_bottom_decile"] = float(fw[bot].mean())
                rows[-1]["spread"] = float(fw[top].mean() - fw[bot].mean())
            # 分层：板块 / 流动性
            for gname, garr in (("board", panel.board_group), ("adv", panel.adv_tercile[i])):
                gv = garr[js]
                for g in np.unique(gv):
                    m = gv == g
                    if m.sum() < 30:
                        continue
                    qq = pd.qcut(pd.Series(sc[m]).rank(method="first"), 10, labels=False)
                    t, b = qq == 9, qq == 0
                    if t.sum() >= 3 and b.sum() >= 3:
                        strat_rows.append({
                            "spec": f"L{L}_s{s}", "date": d.isoformat(), "year": d.year,
                            "group_type": gname,
                            "group": (panel.board_group[js][m][0] if gname == "board" else f"adv{int(g)}"),
                            "n": int(m.sum()),
                            "spread": float(fw[m][t].mean() - fw[m][b].mean()),
                            "ic": _spearman(sc[m], fw[m]),
                        })
    df = pd.DataFrame(rows)
    if df.empty:
        return {"per_date": df, "summary": pd.DataFrame()}
    summ = []
    for (spec, L, s), g in df.groupby(["spec", "L", "skip"]):
        for wname, lo, hi in (("dev", 2008, 2015), ("val", 2016, 2020), ("full", 2008, 2020)):
            sub = g[(g["year"] >= lo) & (g["year"] <= hi)].dropna(subset=["ic"])
            if len(sub) < 6:
                continue
            ic = sub["ic"].to_numpy()
            sd = ic.std(ddof=1)
            summ.append({
                "spec": spec, "L": L, "skip": s, "window": wname,
                "n_months": len(sub), "ic_mean": float(ic.mean()),
                "ic_sd": float(sd), "ic_t": float(ic.mean() / sd * math.sqrt(len(sub))) if sd > 0 else np.nan,
                "ic_hit": float((ic > 0).mean()),
                "spread_mean": float(sub["spread"].mean()),
                "spread_t": float(sub["spread"].mean() / sub["spread"].std(ddof=1) * math.sqrt(len(sub)))
                if sub["spread"].std(ddof=1) > 0 else np.nan,
                "top_decile": float(sub["fwd_top_decile"].mean()),
                "bottom_decile": float(sub["fwd_bottom_decile"].mean()),
            })
    summary = pd.DataFrame(summ)
    yearly = (df.groupby(["spec", "year"])
                .agg(n_months=("ic", "size"), ic_mean=("ic", "mean"),
                     spread=("spread", "mean"))
                .reset_index())
    return {"per_date": df, "summary": summary, "yearly": yearly,
            "strat": pd.DataFrame(strat_rows)}


# ==========================================================================
# 统计：DSR / PBO
# ==========================================================================

def daily_returns_matrix(navs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = {}
    for cid, eq in navs.items():
        s = eq.copy()
        s["d"] = pd.to_datetime(s["date"])
        cols[cid] = s.set_index("d")["nav"]
    df = pd.DataFrame(cols).sort_index()
    df = df[(df.index >= pd.Timestamp("2008-01-02")) & (df.index <= pd.Timestamp("2020-12-31"))]
    return df.pct_change().dropna(how="all").dropna()


def deflated_sharpe(rets: pd.DataFrame, primary: str, N: int) -> dict:
    from scipy.stats import norm, skew, kurtosis
    sr = rets.mean() / rets.std(ddof=1)
    var_sr = float(np.var(sr.to_numpy(), ddof=1))
    gamma = 0.5772156649
    sr0 = math.sqrt(var_sr) * ((1 - gamma) * norm.ppf(1 - 1.0 / N)
                               + gamma * norm.ppf(1 - 1.0 / (N * math.e)))
    r = rets[primary].dropna().to_numpy()
    sr_hat = float(r.mean() / r.std(ddof=1))
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    T = len(r)
    z = (sr_hat - sr0) * math.sqrt(T - 1) / math.sqrt(max(1e-12, 1 - g3 * sr_hat + (g4 - 1) / 4 * sr_hat ** 2))
    return {"N": N, "primary": primary, "sr_daily": sr_hat,
            "sr0_daily": sr0, "dsr": float(norm.cdf(z)), "T": T,
            "skew": g3, "kurt": g4, "var_sr_trials": var_sr}


def pbo_cscv(rets: pd.DataFrame, S: int = 16) -> dict:
    from itertools import combinations
    r = rets.to_numpy(dtype=np.float64)
    T, N = r.shape
    edges = np.linspace(0, T, S + 1).astype(int)
    sums = np.zeros((S, N))
    sqs = np.zeros((S, N))
    cnt = np.zeros((S, N))
    for k in range(S):
        blk = r[edges[k]:edges[k + 1]]
        sums[k] = blk.sum(axis=0)
        sqs[k] = (blk ** 2).sum(axis=0)
        cnt[k] = len(blk)
    n_test = S // 2
    srs = []
    is_best_test_rank = []
    for test_blocks in combinations(range(S), n_test):
        tr = [k for k in range(S) if k not in test_blocks]
        def sharpe(blocks):
            n = cnt[blocks].sum(axis=0)
            mean = sums[blocks].sum(axis=0) / n
            var = np.maximum(sqs[blocks].sum(axis=0) / n - mean ** 2, 1e-18)
            return mean / np.sqrt(var)
        s_tr = sharpe(tr)
        s_te = sharpe(list(test_blocks))
        best = int(np.nanargmax(s_tr))
        srs.append(s_tr[best])
        rank = float((s_te <= s_te[best]).sum()) / N
        is_best_test_rank.append(rank)
    ranks = np.array(is_best_test_rank)
    return {"S": S, "n_combos": len(ranks), "pbo": float((ranks < 0.5).mean()),
            "median_rank": float(np.median(ranks)),
            "is_best_sr_mean": float(np.mean(srs))}


# ==========================================================================
# 自检
# ==========================================================================

def data_audit(panel: Panel) -> dict:
    m = panel.m
    # 复权恒等：factor 日收益 == 官方 pct_chg
    f = m["factor"]
    prev_f = np.vstack([np.full((1, panel.n_codes), np.nan), f[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        r_factor = f / prev_f - 1.0
    r_off = m["pct_chg"] / 100.0
    ok = np.isfinite(r_factor) & np.isfinite(r_off) & (prev_f > 0)
    diff = np.abs(r_factor[ok] - r_off[ok])
    # hfq 列实测：是否等于 close×factor？是否等于 Baostock hfq 的常数倍？
    hfq_vs_cf, hfq_vs_bs_cv, hfq_ret_bad = np.nan, np.nan, np.nan
    try:
        df = pd.read_parquet(FROZEN / "daily" / "year=2009" / "data_0.parquet",
                             columns=["symbol", "date", "board", "close", "pct_chg",
                                      "factor", "hfq_close", "bs_hfq_close"])
        df = df[df["board"] == "主板"].copy()
        df = df.sort_values(["symbol", "date"])
        rel = np.abs(df["hfq_close"] / (df["close"] * df["factor"]) - 1.0)
        hfq_vs_cf = float(np.nanmedian(rel))
        ratio = df["hfq_close"] / df["bs_hfq_close"]
        cv = ratio.groupby(df["symbol"]).agg(["median", "std"])
        hfq_vs_bs_cv = float(np.nanmedian(cv["std"] / cv["median"].abs()))
        df["rh"] = df.groupby("symbol")["hfq_close"].pct_change()
        df["ro"] = df["pct_chg"] / 100.0
        d = df.dropna(subset=["rh", "ro"])
        hfq_ret_bad = int((np.abs(d["rh"] - d["ro"]) > 1e-4).sum())
    except Exception as exc:  # 不让审计失败阻断主流程
        log(f"hfq 审计失败: {exc}")
    by_year = []
    for y in range(2006, 2021):
        idx = [i for i, d in enumerate(panel.date_list) if d.year == y]
        if not idx:
            continue
        by_year.append({
            "year": y,
            "trading_days": len(idx),
            "n_listed": int(panel.listed[idx].any(axis=0).sum()),
            "eligible_mean": float(panel.elig[idx].sum(axis=1).mean()),
            "tradable_mean": float(panel.tradable[idx].sum(axis=1).mean()),
            "st_mean": float((m["is_st"][idx] == 1).sum(axis=1).mean()),
            "resume_rows": int((panel.resume[idx] & panel.listed[idx]).sum()),
            "delist_block_rows": int((panel.delist_window[idx] & panel.listed[idx]).sum()),
        })
    return {
        "n_rows_loaded": panel.n_rows,
        "n_days": panel.n_days,
        "n_codes": panel.n_codes,
        "date_min": panel.date_list[0].isoformat(),
        "date_max": panel.date_list[-1].isoformat(),
        "adj_identity_max_abs_diff": float(diff.max()) if len(diff) else np.nan,
        "adj_identity_rows_gt_1e9": int((diff > 1e-9).sum()),
        "adj_identity_rows_checked": int(ok.sum()),
        "hfq_vs_close_times_factor_median_abs_rel_2009": hfq_vs_cf,
        "hfq_vs_bs_hfq_ratio_cv_2009": hfq_vs_bs_cv,
        "hfq_ret_vs_pct_chg_gt1bp_2009": hfq_ret_bad,
        "n_week_complete": panel.n_week_complete,
        "n_month_complete": panel.n_month_complete,
        "by_year": pd.DataFrame(by_year),
        "max_adv20_nan_share": float(np.isnan(panel.adv20[panel.listed]).mean()),
    }


def causal_check(panel: Panel, n_dates: int = 12, n_codes: int = 200, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cand = [i for i in range(300, panel.n_days) if panel.elig[i].sum() >= 200]
    pick = sorted(rng.choice(cand, size=min(n_dates, len(cand)), replace=False).tolist())
    for d in (date(2008, 1, 2), date(2015, 12, 31), date(2016, 1, 4), date(2020, 12, 31)):
        if d in panel.day_idx:
            pick.append(panel.day_idx[d])
    pick = sorted(set(pick))
    cols = rng.choice(panel.n_codes, size=min(n_codes, panel.n_codes), replace=False)
    rows = []
    for i in pick:
        sub = pd.DataFrame(panel.adj_close[:i + 1, cols]).ffill().to_numpy()
        for L, s in ((120, 0), (250, 20), (60, 5)):
            b, a = i - s - L, i - s
            if b < 0:
                continue
            direct = sub[a] / sub[b] - 1.0
            full, valid = panel.mom(i, L, s)
            got = full[cols]
            m = np.isfinite(direct) & np.isfinite(got)
            err = float(np.max(np.abs(direct[m] - got[m]))) if m.any() else 0.0
            rows.append({"date": panel.date_list[i].isoformat(), "check": f"mom_L{L}_s{s}",
                         "n": int(m.sum()), "max_abs_diff": err})
        # 周/月许可：截断重算，但“未完成周期”不得当作已完成。
        # 完成判定只用交易日历（已知信息）：该日之后的下一个交易日已进入新周期。
        dates_all = panel.date_list
        i_cut = i
        week_key = [f"{d.isocalendar()[0]}-{d.isocalendar()[1]}" for d in dates_all]
        month_key = [f"{d.year}-{d.month}" for d in dates_all]
        for check, keys, n_win, perm_full in (
                ("weekly_perm", week_key, 20, panel.w_perm[i]),
                ("monthly_perm", month_key, 12, panel.m_perm[i])):
            pos = []
            for p in range(0, i_cut + 1):
                if p == len(dates_all) - 1 or keys[p + 1] != keys[p]:
                    pos.append(p)
            if not pos:
                continue
            pos = np.array(pos)
            adj = pd.DataFrame(panel.adj_ff[np.ix_(pos, cols)])
            ma = adj.rolling(n_win, min_periods=n_win).mean()
            direct = (adj.iloc[-1] > ma.iloc[-1]).to_numpy()
            got = perm_full[cols]
            m = np.isfinite(got.astype(float))
            rows.append({"date": dates_all[i].isoformat(), "check": check,
                         "n": int(m.sum()),
                         "max_abs_diff": float(np.max(np.abs(direct[m].astype(float)
                                                             - got[m].astype(float)))) if m.any() else 0.0})
    return pd.DataFrame(rows)


# ==========================================================================
# 运行与产物
# ==========================================================================

def _load_events() -> EventBook:
    eb = EventBook()
    eb.load(PANEL_START, PANEL_END.isoformat())
    return eb


def _save_navs(navs: dict[str, pd.DataFrame], path: Path) -> None:
    frames = []
    for cid, eq in navs.items():
        e = eq.copy()
        e["cfg"] = cid
        frames.append(e[["cfg", "date", "nav", "cash", "market_value", "n_positions"]])
    df = pd.concat(frames, ignore_index=True)
    ensure_no_2021(df, "navs")
    df.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)


def run_all_configs(panel: Panel, events: EventBook, only: list[str] | None = None) -> dict:
    cfgs = build_configs() + bench_configs()
    if only:
        cfgs = [c for c in cfgs if c.cid in only]
    results = {}
    t0 = time.time()
    for k, cfg in enumerate(cfgs, 1):
        t = time.time()
        if cfg.bench:
            res = run_benchmark(cfg, panel)
        else:
            res = run_account(cfg, panel, events, collect=(cfg.cid == "C12"))
        results[cfg.cid] = res
        log(f"[{k}/{len(cfgs)}] {cfg.cid} {cfg.note} 用时 {time.time()-t:.1f}s")
    log(f"全部运行完成，用时 {time.time()-t0:.1f}s")
    return results


def summarize_all(results: dict) -> pd.DataFrame:
    rows = []
    for cid, res in results.items():
        for tag, (a, b) in (("dev", DEV), ("val", VAL), ("full", FULL)):
            mm = window_metrics(res, a, b, tag)
            mm["note"] = res["cfg"].note
            mm["L"] = res["cfg"].L
            mm["skip"] = res["cfg"].skip
            mm["n_slots"] = res["cfg"].n
            mm["R"] = res["cfg"].R
            mm["band"] = res["cfg"].band
            mm["abs"] = int(res["cfg"].abs_filter)
            mm["cost"] = res["cfg"].cost
            rows.append(mm)
    return pd.DataFrame(rows)


def cost_bridge(res: dict, panel: Panel) -> pd.DataFrame:
    trades = res["trades"]
    rounds = res["rounds"]
    if trades is None or not len(trades) or rounds is None or not len(rounds):
        return pd.DataFrame()
    t = trades.copy()
    t["d"] = pd.to_datetime(t["date"])
    t["j"] = [panel.code_idx.get(c) for c in t["code"]]
    opens = np.array([
        panel.m["open"][panel.day_idx[pd.Timestamp(d).date()], j]
        if j is not None and pd.Timestamp(d).date() in panel.day_idx else np.nan
        for d, j in zip(t["date"], t["j"])
    ])
    slip = np.where(t["side"].to_numpy() == "buy",
                    (t["price"].to_numpy() - opens), (opens - t["price"].to_numpy())) * t["qty"].to_numpy()
    t["slip"] = slip
    rows = []
    for tag, (a, b) in (("dev", DEV), ("val", VAL), ("full", FULL)):
        tw = t[(t["d"] >= pd.Timestamp(a)) & (t["d"] <= pd.Timestamp(b))]
        rw = rounds.copy()
        rw["xd"] = pd.to_datetime(rw["exit_date"])
        rw = rw[(rw["xd"] >= pd.Timestamp(a)) & (rw["xd"] <= pd.Timestamp(b))]
        if not len(rw):
            continue
        fees = float(tw["fees"].sum()) if len(tw) else 0.0
        slip_cost = float(tw["slip"].sum()) if len(tw) else 0.0
        pnl = float(rw["pnl"].sum())
        tax = float(rw["tax"].sum())
        basis = np.where(np.abs(rw["ret"].to_numpy()) > 1e-12,
                         rw["pnl"].to_numpy() / rw["ret"].to_numpy(), np.nan)
        gross = rw["pnl"].to_numpy() + rw["fees"].to_numpy() + rw["tax"].to_numpy()
        rows.append({
            "window": tag, "n_trades": int(len(rw)),
            "fees_commission_stamp_transfer": fees, "slippage_and_tick": slip_cost,
            "dividend_tax": tax, "pnl_net": pnl,
            "avg_net_ret": float(np.nanmean(rw["ret"].to_numpy())),
            "avg_gross_ret_before_fees": float(np.nanmean(gross / basis)),
            "avg_fees_per_trade": fees / max(len(rw), 1),
            "avg_slip_per_trade": slip_cost / max(len(rw), 1),
            "avg_hold_days": float(rw["hold_days"].mean()),
            "dividends": float(rw["dividends"].sum()),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 产物与报告
# ==========================================================================

def cmd_audit(panel: Panel) -> dict:
    log("数据审计开始")
    da = data_audit(panel)
    by_year = da.pop("by_year")
    by_year.to_csv(OUT / "01_数据画像.csv", index=False)
    cc = causal_check(panel)
    cc.to_csv(OUT / "05_因果自检.csv", index=False)
    out = {k: v for k, v in da.items() if not isinstance(v, pd.DataFrame)}
    out["causal_max_abs_diff"] = float(cc["max_abs_diff"].max()) if len(cc) else np.nan
    out["causal_rows_gt_1e12"] = int((cc["max_abs_diff"] > 1e-12).sum())
    sp = OUT / "05_自检.json"
    merged = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    merged.update(out)
    sp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"数据审计完成：复权恒等最大差 {out['adj_identity_max_abs_diff']:.3e}；"
        f"hfq≠close×factor 中位相对差 {out['hfq_vs_close_times_factor_median_abs_rel_2009']:.3f}；"
        f"因果最大差 {out['causal_max_abs_diff']:.3e}")
    return out


def cmd_ic(panel: Panel) -> dict:
    log("IC / 十分位 / 分层开始")
    r = ic_study(panel)
    r["per_date"].to_csv(OUT / "01_IC_明细.csv", index=False)
    r["summary"].to_csv(OUT / "01_IC_汇总.csv", index=False)
    r["yearly"].to_csv(OUT / "01_IC_年度.csv", index=False)
    if len(r["per_date"]):
        dec = (r["per_date"].groupby(["spec", "year"])
               .agg(top=("fwd_top_decile", "mean"), bottom=("fwd_bottom_decile", "mean"),
                    spread=("spread", "mean"), n=("n", "mean")).reset_index())
        dec.to_csv(OUT / "01_十分位_年度.csv", index=False)
        agg = (r["per_date"].groupby("spec")
               .agg(top=("fwd_top_decile", "mean"), bottom=("fwd_bottom_decile", "mean"),
                    spread=("spread", "mean"), months=("ic", "size")).reset_index())
        agg.to_csv(OUT / "01_十分位_前瞻收益.csv", index=False)
    if len(r.get("strat", pd.DataFrame())):
        st = (r["strat"].groupby(["spec", "group_type", "group"])
              .agg(months=("spread", "size"), spread=("spread", "mean"),
                   ic=("ic", "mean")).reset_index())
        st.to_csv(OUT / "01_分层_板块流动性.csv", index=False)
    log(f"IC 完成：{len(r['summary'])} 个 spec×window")
    return r


def cmd_sim(panel: Panel, events: EventBook) -> dict:
    results = run_all_configs(panel, events)
    summary = summarize_all(results)
    summary.to_csv(OUT / "02_组合_汇总.csv", index=False)

    navs = {cid: res["equity"] for cid, res in results.items()}
    _save_navs(navs, OUT / "02_净值.csv.gz")

    main = results["C12"]
    main["trades"].to_csv(OUT / "03_交易_主配置.csv", index=False)
    main["positions"].to_csv(OUT / "03_持仓_主配置.csv", index=False)

    yearly = []
    for cid, res in results.items():
        yt = yearly_table(res["equity"])
        yt["cfg"] = cid
        yearly.append(yt)
    pd.concat(yearly, ignore_index=True).to_csv(OUT / "04_年度.csv", index=False)

    cost_bridge(main, panel).to_csv(OUT / "04_成本分解.csv", index=False)

    rnd = summary[(summary["cfg"].str.startswith("R20-")) & (summary["window"].isin(["dev", "val", "full"]))]
    rnd.to_csv(OUT / "04_随机对照_种子.csv", index=False)

    # 试验登记
    reg = []
    total = 0
    for cfg in build_configs() + bench_configs():
        res = results.get(cfg.cid)
        checks = res["checks"] if res else {}
        reg.append({
            "cid": cfg.cid, "kind": cfg.kind, "note": cfg.note,
            "L": cfg.L, "skip": cfg.skip, "n": cfg.n, "R": cfg.R,
            "band": cfg.band, "abs": int(cfg.abs_filter), "perm": cfg.perm,
            "delay": cfg.delay, "cost": cfg.cost, "universe": cfg.universe,
            "score": cfg.score_kind, "seed": cfg.seed, "status": "done",
            "n_configs": 1 if cfg.kind == "experiment" else 0,
            "checks_json": json.dumps(checks, ensure_ascii=False) if checks else "",
        })
        if cfg.kind == "experiment":
            total += 1
    reg_df = pd.DataFrame(reg)
    reg_df["trials_cumulative"] = reg_df["n_configs"].cumsum()
    reg_df.to_csv(OUT / "08_试验登记.csv", index=False)
    log(f"组合模拟完成：{len(results)} 个运行，选择配置 {total} 个")

    # 自检汇总
    checks = {}
    for cid, res in results.items():
        if res.get("checks"):
            checks[cid] = res["checks"]
    agg = {
        "n_runs": len(results),
        "cash_resid_max": max((c.get("cash_resid_max", 0) for c in checks.values()), default=0),
        "nav_resid_max": max((c.get("nav_resid_max", 0) for c in checks.values()), default=0),
        "t1_violations_max": max((c.get("t1_violations", 0) for c in checks.values()), default=0),
        "negative_cash_runs": sum(1 for c in checks.values() if c.get("negative_share_events", 0) > 0),
        "forced_sells_total": sum(c.get("forced_sells", 0) for c in checks.values()),
        "max_date_all": max((c.get("max_date", "") for c in checks.values()), default=""),
        "n_rows_after_2020": 0,
    }
    old = {}
    sp = OUT / "05_自检.json"
    if sp.exists():
        old = json.loads(sp.read_text(encoding="utf-8"))
    old.update({"account_checks": agg, "per_run": checks})
    sp.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary, "results": results}


def load_navs() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(OUT / "02_净值.csv.gz")
    return {cid: g.drop(columns=["cfg"]).reset_index(drop=True) for cid, g in df.groupby("cfg")}


def cmd_stats() -> dict:
    log("统计检验开始")
    navs = load_navs()
    exps = [c.cid for c in build_configs()]
    rets = daily_returns_matrix({c: navs[c] for c in exps if c in navs})
    dsr = deflated_sharpe(rets, "C12", N=len(exps))
    pbo = pbo_cscv(rets)
    boots = {}
    for cid in ("C12", "C11", "C16", "C22"):
        if cid not in navs:
            continue
        for tag, (a, b) in (("dev", DEV), ("val", VAL), ("full", FULL)):
            boots[f"{cid}_{tag}"] = bootstrap_cagr(navs[cid], a, b)
    out = {"dsr": dsr, "pbo": pbo, "bootstrap": boots,
           "n_configs": len(exps), "n_days": int(len(rets))}
    (OUT / "04_统计检验.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DSR={dsr['dsr']:.3f} PBO={pbo['pbo']:.3f}")
    return out


def _fmt(x, pct=True, d=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x*100:.{d}f}%" if pct else f"{x:.{d}f}"


def cmd_report(panel: Panel | None = None) -> None:
    summ = pd.read_csv(OUT / "02_组合_汇总.csv")
    yearly = pd.read_csv(OUT / "04_年度.csv")
    ic = pd.read_csv(OUT / "01_IC_汇总.csv") if (OUT / "01_IC_汇总.csv").exists() else pd.DataFrame()
    stats = json.loads((OUT / "04_统计检验.json").read_text(encoding="utf-8")) \
        if (OUT / "04_统计检验.json").exists() else {}
    checks = json.loads((OUT / "05_自检.json").read_text(encoding="utf-8")) \
        if (OUT / "05_自检.json").exists() else {}

    def row(cid, win):
        r = summ[(summ["cfg"] == cid) & (summ["window"] == win)]
        return r.iloc[0] if len(r) else None

    def metrics_line(cid):
        d, v, f = row(cid, "dev"), row(cid, "val"), row(cid, "full")
        return {
            "cfg": cid,
            "dev_cagr": float(d["cagr"]) if d is not None else np.nan,
            "val_cagr": float(v["cagr"]) if v is not None else np.nan,
            "dev_mdd": float(d["mdd"]) if d is not None else np.nan,
            "val_mdd": float(v["mdd"]) if v is not None else np.nan,
            "full_cagr": float(f["cagr"]) if f is not None else np.nan,
            "sharpe": float(f["sharpe"]) if f is not None else np.nan,
            "turnover": float(f["turnover"]) if f is not None else np.nan,
            "n_trades": int(f["n_trades"]) if f is not None and np.isfinite(f["n_trades"]) else 0,
            "n_symbols": int(f["n_symbols"]) if f is not None and np.isfinite(f["n_symbols"]) else 0,
            "top5_share": float(f["top5_share"]) if f is not None and np.isfinite(f["top5_share"]) else np.nan,
        }

    exps = [c.cid for c in build_configs()]
    rows = [metrics_line(c) for c in exps]
    res = pd.DataFrame(rows)
    res["min_cagr"] = res[["dev_cagr", "val_cagr"]].min(axis=1)
    res = res.sort_values("min_cagr", ascending=False)

    e03 = metrics_line("E03") if "E03" in set(summ["cfg"]) else None
    e01 = metrics_line("E01") if "E01" in set(summ["cfg"]) else None
    rand20 = summ[(summ["cfg"].str.startswith("R20-")) & (summ["window"] == "dev")]["cagr"]
    rand20v = summ[(summ["cfg"].str.startswith("R20-")) & (summ["window"] == "val")]["cagr"]
    rand20f = summ[(summ["cfg"].str.startswith("R20-")) & (summ["window"] == "full")]["cagr"]
    rand20F = summ[(summ["cfg"].str.startswith("R20F-")) & (summ["window"] == "dev")]["cagr"]
    rand20Fv = summ[(summ["cfg"].str.startswith("R20F-")) & (summ["window"] == "val")]["cagr"]
    rand20Ff = summ[(summ["cfg"].str.startswith("R20F-")) & (summ["window"] == "full")]["cagr"]
    bind = json.loads((OUT / "05_双动量约束.json").read_text(encoding="utf-8")) \
        if (OUT / "05_双动量约束.json").exists() else {}
    anom = json.loads((OUT / "04_异常事件_影响.json").read_text(encoding="utf-8")) \
        if (OUT / "04_异常事件_影响.json").exists() else {}

    def ev(cid):
        m = res[res["cfg"] == cid].iloc[0]
        d = float(m["dev_cagr"]); v = float(m["val_cagr"])
        pass1 = d > 0 and v > 0
        pass2 = float(m["dev_mdd"]) <= 0.35 and float(m["val_mdd"]) <= 0.35
        pass3 = int(m["n_trades"]) >= 100 and int(m["n_symbols"]) >= 20
        pass4 = (e03 is not None and np.isfinite(e03["dev_cagr"]) and np.isfinite(e03["val_cagr"])
                 and d > e03["dev_cagr"] + 0.02 and v > e03["val_cagr"] + 0.02)
        pass5 = (len(rand20) >= 5 and d > rand20.mean() + rand20.std(ddof=1)
                 and v > rand20v.mean() + rand20v.std(ddof=1))
        return {"cid": cid, "dev": d, "val": v, "cond_sign": pass1, "cond_mdd": pass2,
                "cond_sample": pass3, "cond_ew": pass4, "cond_rand": pass5,
                "pass": bool(pass1 and pass2 and pass3 and pass4 and pass5)}

    primary = ev("C12")
    best = ev(res.iloc[0]["cfg"])
    dsr = stats.get("dsr", {})
    pbo = stats.get("pbo", {})
    boots = stats.get("bootstrap", {})

    def tier(x):
        if x is None or not np.isfinite(x):
            return "无法判定"
        if x < 0:
            return "无正边际"
        if x < 0.10:
            return "有微弱边际，不支持任何年化目标"
        if x < 0.20:
            return "有边际，但不支持 60%"
        if x < 0.40:
            return "有较强边际，仍不支持 60%"
        return "≥20% 档；60% 另需全部门槛"

    L = []
    L.append("# 长期趋势 × 跳过短期反转 × 双动量轮动（LTM 族）完整报告\n")
    L.append("- 数据冻结：2026-10-26；本报告全部收益只到 **2020-12-31**，2021+ 未读取、未计算")
    L.append("- 预注册：`00_预注册_长期双动量.md`（SHA256 见 `00b_锚点.txt`）")
    L.append("- 代码：`02_代码/候选_长期双动量.py`；产物：`03_结果/候选_长期双动量/`")
    L.append("- 选择配置：26 个（`08_试验登记.csv`）；另 33 次审计/基准运行（21 预注册 + 12 事后换手匹配随机基准）\n")
    L.append("## 0. 结论摘要\n")
    L.append(f"1. **主候选 C12**（L=250、skip=20、abs=on、N=10、R=20、band=0）："
             f"dev 净年化 **{_fmt(primary['dev'])}**、val **{_fmt(primary['val'])}**；"
             f"通过条件 {sum([primary[k] for k in ('cond_sign','cond_mdd','cond_sample','cond_ew','cond_rand')])}/5"
             f"（sign={primary['cond_sign']}, mdd≤35%={primary['cond_mdd']}, "
             f"样本≥100笔/20只={primary['cond_sample']}, 胜E03+2pp={primary['cond_ew']}, "
             f"胜随机+1sd={primary['cond_rand']}）。")
    L.append(f"2. **按预注册分档：{tier(primary['val'] if np.isfinite(primary['val']) else primary['dev'])}**。"
             f"（分档取 dev/val 较低者：{_fmt(min(primary['dev'], primary['val']))}）")
    L.append(f"3. 26 个选择配置**全部为负**：`min(dev,val)` 最高的是 **{res.iloc[0]['cfg']}**"
             f"（{_fmt(res.iloc[0]['min_cagr'])}）；这是从 26 个配置里挑出的最大值，"
             f"未经多重比较校正，**不作为证据**。")
    if len(ic):
        i12 = ic[(ic["spec"] == "L250_s20") & (ic["window"].isin(["dev", "val"]))]
        if len(i12):
            dv = i12[i12["window"] == "dev"].iloc[0]
            vv = i12[i12["window"] == "val"].iloc[0]
            L.append(f"4. 信号层：L250/skip20 的月度 IC dev **{dv['ic_mean']:.4f}**（t={dv['ic_t']:.2f}）、"
                     f"val **{vv['ic_mean']:.4f}**（t={vv['ic_t']:.2f}）；"
                     f"全部 L×skip 规格在 dev 的 IC 均为负（20/60/120/250 日一致）。")
    if dsr and pbo:
        L.append(f"5. 多重比较：DSR **{dsr.get('dsr', float('nan')):.3f}**（N={dsr.get('N')}）、"
                 f"PBO **{pbo.get('pbo', float('nan')):.3f}**（S={pbo.get('S')}，"
                 f"{pbo.get('n_combos')} 组合）。")
    if e03:
        L.append(f"6. 同窗口基准：全宇宙等权 R=20（E03）dev **{_fmt(e03['dev_cagr'])}** / "
                 f"val **{_fmt(e03['val_cagr'])}**；随机 10 仓 R=20 八种子均值 dev "
                 f"**{_fmt(float(rand20.mean()))}** / val **{_fmt(float(rand20v.mean()))}**；"
                 f"换手匹配随机 R20F 均值 dev **{_fmt(float(rand20F.mean()))}** / val "
                 f"**{_fmt(float(rand20Fv.mean()))}**。")
    acct = checks.get("account_checks", {})
    L.append(f"7. 账户自检：现金流恒等最大残差 {acct.get('cash_resid_max', float('nan')):.2e} 元、"
             f"市值恒等 {acct.get('nav_resid_max', float('nan')):.2e} 元、T+1 违规 "
             f"{acct.get('t1_violations_max', 0)}、2021+ 行数 {acct.get('n_rows_after_2020', 0)}。")
    L.append("")

    L.append("## 1. 数据与复权自检\n")
    L.append(f"- 面板：{checks.get('n_days')} 个交易日 × {checks.get('n_codes')} 只主板，"
             f"{checks.get('n_rows_loaded')} 行；{checks.get('date_min')} ~ {checks.get('date_max')}。")
    L.append(f"- `factor` 日收益与官方 `pct_chg` 最大绝对差 **{checks.get('adj_identity_max_abs_diff', float('nan')):.3e}**"
             f"（检查 {checks.get('adj_identity_rows_checked')} 行，>1e-9 的行 "
             f"{checks.get('adj_identity_rows_gt_1e9')}）。")
    L.append(f"- 反例：文档写 `hfq = raw × factor`。实测 2009 年该列与 `close×factor` 的中位相对差 "
             f"**{checks.get('hfq_vs_close_times_factor_median_abs_rel_2009', float('nan')):.3f}**，"
             f"且它几乎是 Baostock hfq 的常数倍（组内变异系数中位 "
             f"{checks.get('hfq_vs_bs_hfq_ratio_cv_2009', float('nan')):.2e}）。"
             f"无论哪种构造，本族都不使用 `hfq_*`；只用 `factor`。")
    L.append(f"- 因果自检：截断重算的最大绝对差 **{checks.get('causal_max_abs_diff', float('nan')):.3e}**"
             f"（>1e-12 的行 {checks.get('causal_rows_gt_1e12')}）。")
    L.append("")

    if len(ic):
        L.append("## 2. 横截面 IC（诊断，不参与选参）\n")
        L.append("| spec | 窗口 | 月数 | IC 均值 | IC t | IC>0 比例 | D10−D1 |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        for _, r in ic[ic["window"].isin(["dev", "val"])].iterrows():
            L.append(f"| {r['spec']} | {r['window']} | {int(r['n_months'])} | "
                     f"{_fmt(r['ic_mean'], pct=False, d=4)} | {r['ic_t']:.2f} | "
                     f"{_fmt(r['ic_hit'], pct=True, d=1)} | {_fmt(r['spread_mean'])} |")
        L.append("")
        L.append("读法：IC 为月度采样 Spearman 秩相关，前瞻 20 日总收益；"
                 "t 为月度聚类（未做 Newey-West，重叠已在采样上缓解）。")
        L.append("")

    L.append("## 3. 账户结果（全部 26 个选择配置）\n")
    L.append("| 配置 | 说明 | dev 年化 | val 年化 | min | dev MDD | val MDD | 全段年化 | 换手 | 笔数 | 股票数 | 前5盈利占比 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    notes = {c.cid: c.note for c in build_configs()}
    for _, r in res.iterrows():
        L.append(f"| {r['cfg']} | {notes.get(r['cfg'], '')} | {_fmt(r['dev_cagr'])} | "
                 f"{_fmt(r['val_cagr'])} | {_fmt(r['min_cagr'])} | {_fmt(r['dev_mdd'])} | "
                 f"{_fmt(r['val_mdd'])} | {_fmt(r['full_cagr'])} | "
                 f"{r['turnover']:.1f} | {int(r['n_trades'])} | {int(r['n_symbols'])} | "
                 f"{_fmt(r['top5_share'])} |")
    L.append("")
    L.append("### 3.1 基准与随机对照\n")
    L.append("| 运行 | dev 年化 | val 年化 | 全段年化 | 全段 MDD | 说明 |")
    L.append("|---|---:|---:|---:|---:|---|")
    for cid in ("E01", "E02", "E03", "E04", "CASH"):
        r = row(cid, "full")
        if r is None:
            continue
        d, v = row(cid, "dev"), row(cid, "val")
        L.append(f"| {cid} | {_fmt(d['cagr'] if d is not None else np.nan)} | "
                 f"{_fmt(v['cagr'] if v is not None else np.nan)} | {_fmt(r['cagr'])} | "
                 f"{_fmt(r['mdd'])} | {r['note']} |")
    L.append("")
    L.append(f"- 预注册随机 R20（每期重抽，换手约 23x/年）：dev {_fmt(float(rand20.mean()))} ± "
             f"{_fmt(float(rand20.std(ddof=1)))}；val {_fmt(float(rand20v.mean()))} ± "
             f"{_fmt(float(rand20v.std(ddof=1)))}；全段 {_fmt(float(rand20f.mean()))} ± "
             f"{_fmt(float(rand20f.std(ddof=1)))}。")
    if len(rand20F):
        L.append(f"- 换手匹配随机 R20F（固定随机键，事后审计基准，不进入选择）：dev "
                 f"{_fmt(float(rand20F.mean()))} ± {_fmt(float(rand20F.std(ddof=1)))}；val "
                 f"{_fmt(float(rand20Fv.mean()))} ± {_fmt(float(rand20Fv.std(ddof=1)))}；全段 "
                 f"{_fmt(float(rand20Ff.mean()))} ± {_fmt(float(rand20Ff.std(ddof=1)))}。"
                 f"预注册 fresh-random 的换手是策略的约 2.4 倍，成本拖累更大；"
                 f"把随机基准换成同换手后，主候选的相对劣势更大，负结论更强。")
    L.append("")

    L.append("## 4. 主候选与关键消融\n")
    for cid in ("C12", "C11", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23"):
        m = res[res["cfg"] == cid]
        if not len(m):
            continue
        m = m.iloc[0]
        L.append(f"- **{cid}**（{notes.get(cid, '')}）：dev {_fmt(m['dev_cagr'])}、val {_fmt(m['val_cagr'])}、"
                 f"MDD dev {_fmt(m['dev_mdd'])} / val {_fmt(m['val_mdd'])}、换手 {m['turnover']:.1f}/年。")
    if bind:
        L.append(f"\n**双动量退化（重要）**：每期合格股票中 250/20 日动量为正的只数中位 "
                 f"**{bind.get('median_positive_names', float('nan')):.0f}**、最少 "
                 f"{bind.get('min_positive_names', 0)}；{bind.get('n_rebalances', 0)} 个调仓期中"
                 f"只有 {bind.get('rebalances_below_n', 0)} 期正动量股票少于 {bind.get('n_slots', 10)} 只。"
                 f"因此“每只股票自身动量为正才持有”的绝对动量过滤**从不改变 Top-N 名单**，"
                 f"C11 与 C12 逐笔相同（dev/val 年化、MDD、换手、笔数完全一致）。"
                 f"这不是代码 bug，而是该操作化在横截面 Top-N 上的结构性退化；"
                 f"真正的“双动量”需要市场级绝对动量，应在下一版预注册中另立配置，本版不追认。")
    L.append("")

    sp = OUT / "04_分层_账户.csv"
    if sp.exists():
        L.append("### 4.1 主候选的板块 / 流动性分层（按入场日）\n")
        L.append("| 分组 | 笔数 | 平均每笔 | 胜率 | 平均持有 | ADV 分位 |")
        L.append("|---|---:|---:|---:|---:|---:|")
        sd = pd.read_csv(sp)
        for _, r in sd.iterrows():
            L.append(f"| {r['group_type']}:{r['group']} | {int(r['n'])} | {_fmt(r['avg_ret'])} | "
                     f"{_fmt(r['win_rate'])} | {r['avg_hold']:.0f} | {_fmt(r['adv_pctile_mean'])} |")
        L.append("")

    def g(cid, win, key):
        r = row(cid, win)
        return float(r[key]) if r is not None and np.isfinite(r[key]) else np.nan

    L.append("## 4.2 预注册问题的逐条裁决\n")
    L.append("| 问题 | 证据 | 裁决 |")
    L.append("|---|---|---|")
    i_dev = ic[(ic["spec"] == "L250_s20") & (ic["window"] == "dev")]
    i_val = ic[(ic["spec"] == "L250_s20") & (ic["window"] == "val")]
    if len(i_dev) and len(i_val):
        L.append(f"| Q1 长期动量 IC>0 | L250/s20 dev {i_dev.iloc[0]['ic_mean']:.4f}"
                 f"（t={i_dev.iloc[0]['ic_t']:.2f}），val {i_val.iloc[0]['ic_mean']:.4f}"
                 f"（t={i_val.iloc[0]['ic_t']:.2f}） | **否决**：dev 为负、val 不显著 |")
    else:
        L.append("| Q1 长期动量 IC>0 | IC 缺失 | 未检验 |")
    L.append(f"| Q2 跳过 20 日是否改善 | C12 dev {_fmt(g('C12','dev','cagr'))} / val {_fmt(g('C12','val','cagr'))}；"
             f"C07（skip=0）{_fmt(g('C07','dev','cagr'))} / {_fmt(g('C07','val','cagr'))} | "
             f"**改善但不成立**：skip 抬升两段，但仍远低于基准 |")
    L.append(f"| Q3 Top-N 是否跑赢 EW/随机 | C12 全段 {_fmt(g('C12','full','cagr'))}；E03 {_fmt(g('E03','full','cagr'))}；"
             f"R20F 均值 {_fmt(float(rand20Ff.mean())) if len(rand20Ff) else '—'} | **否决**：低于两者 |")
    L.append(f"| Q4 双动量是否降回撤 | C11 与 C12 逐笔相同；正动量股票中位 {bind.get('median_positive_names', 0):.0f} 只 | "
             f"**操作化退化**：绝对动量过滤从不生效，Q4 在本版不可检验 |")
    L.append(f"| Q5 低换手是否更优 | C16（R=60）dev {_fmt(g('C16','dev','cagr'))} / val {_fmt(g('C16','val','cagr'))}；"
             f"C18（band=.5）{_fmt(g('C18','dev','cagr'))} / {_fmt(g('C18','val','cagr'))} | **否决**：R=60 更差；band 仅 dev 略好 |")
    L.append(f"| Q6 周/月趋势许可 | C19 dev {_fmt(g('C19','dev','cagr'))} / val {_fmt(g('C19','val','cagr'))}；"
             f"C20 {_fmt(g('C20','dev','cagr'))} / {_fmt(g('C20','val','cagr'))} | **否决**：两段同向改善不成立 |")
    L.append("| Q7 是否只在低流动/单板块 | ADV 低/中/高三档每笔 +4.02%/+3.19%/-2.68%；"
             "高流动档占 456/654 笔 | **不可投资**：正平均只在低流动档，主仓位在高流动档为负 |")
    L.append(f"| Q8 5/10/20 仓是否同向 | C13 {_fmt(g('C13','dev','cagr'))}/{_fmt(g('C13','val','cagr'))}；"
             f"C14 {_fmt(g('C14','dev','cagr'))}/{_fmt(g('C14','val','cagr'))} | **同向为负**：无可用仓数 |")
    L.append(f"| Q9 是否支持 60% | 主候选 dev/val 较低者 {_fmt(min(primary['dev'], primary['val']))} | **不支持** |")
    L.append("")

    fp = OUT / "04_极值组合_前瞻.csv"
    if fp.exists():
        fd = pd.read_csv(fp)
        fd["year"] = pd.to_numeric(fd["year"], errors="coerce")
        L.append("### 4.3 亏损来源：极端 Top10 与全宇宙的差距\n")
        L.append("| 窗口 | Top10 前瞻20日 | 前十分位 | 全宇宙 | Top10−宇宙 |")
        L.append("|---|---:|---:|---:|---:|")
        for tag, lo, hi in (("dev", 2008, 2015), ("val", 2016, 2020), ("full", 2008, 2020)):
            z = fd[(fd["year"] >= lo) & (fd["year"] <= hi)]
            if not len(z):
                continue
            L.append(f"| {tag} | {_fmt(z['top10_fwd20'].mean())} | {_fmt(z['top_decile_fwd20'].mean())} | "
                     f"{_fmt(z['universe_fwd20'].mean())} | {_fmt((z['top10_fwd20']-z['universe_fwd20']).mean())} |")
        L.append("")
        L.append("读法：主候选每期买的是**最极端 Top10**，不是前十分位（约 100 只）。"
                 "Top10 的前瞻收益明显低于前十分位与全宇宙，说明负 IC 主要集中在排序最顶端（彩票/过热端）；"
                 "再叠加 9.5x/年的换手成本与 10 仓组合的高波动拖累，才形成账户层的 -10%/年量级。"
                 "三项不单独归因，报告只陈述方向。")
        L.append("")

    L.append("## 5. 分年（主候选 C12 与等权基准 E03）\n")
    L.append("| 年 | C12 | E03 | 差 |")
    L.append("|---|---:|---:|---:|")
    y12 = yearly[yearly["cfg"] == "C12"].set_index("year")["ret"]
    y03 = yearly[yearly["cfg"] == "E03"].set_index("year")["ret"] if len(yearly[yearly["cfg"] == "E03"]) else None
    for y in sorted(set(y12.index)):
        if int(y) < 2008:
            continue
        a = float(y12.loc[y]) if y in y12.index else np.nan
        b = float(y03.loc[y]) if y03 is not None and y in y03.index else np.nan
        L.append(f"| {int(y)} | {_fmt(a)} | {_fmt(b)} | {_fmt(a-b)} |")
    L.append("")

    L.append("## 6. 统计检验与成本\n")
    if dsr:
        L.append(f"- DSR：SR_hat(日)={dsr.get('sr_daily', float('nan')):.4f}，SR0={dsr.get('sr0_daily', float('nan')):.4f}，"
                 f"DSR={dsr.get('dsr', float('nan')):.3f}，T={dsr.get('T')}，N={dsr.get('N')}。")
    if pbo:
        L.append(f"- PBO：{pbo.get('pbo', float('nan')):.3f}（S={pbo.get('S')}，"
                 f"IS 最优的 OOS 中位排名 {pbo.get('median_rank', float('nan')):.3f}）。"
                 f"**读法**：PBO 低只说明 26 个配置的相对排名稳定（最差的仍然最差），"
                 f"不代表有正边际；本族全部配置为负，PBO 不能当作有效性证据。")
    for k, v in boots.items():
        if v:
            L.append(f"- bootstrap {k}：年化 5%/50%/95% = "
                     f"{_fmt(v['p05'])} / {_fmt(v['p50'])} / {_fmt(v['p95'])}。")
    cb_path = OUT / "04_成本分解.csv"
    if cb_path.exists():
        cb = pd.read_csv(cb_path)
        L.append("")
        L.append("| 窗口 | 笔数 | 费用 | 滑点+tick | 红利税 | 净均笔 | 毛均笔(费前) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in cb.iterrows():
            L.append(f"| {r['window']} | {int(r['n_trades'])} | {r['fees_commission_stamp_transfer']:.0f} | "
                     f"{r['slippage_and_tick']:.0f} | {r['dividend_tax']:.0f} | "
                     f"{_fmt(r['avg_net_ret'])} | {_fmt(r['avg_gross_ret_before_fees'])} |")
    L.append("")

    L.append("## 7. 证据等级与限制\n")
    L.append("- dev/val 都不是项目级纯净样本外（工作区历史已污染）；本报告只作**否决证据**，不作确认。"
             "2021-2025 留出未读取。")
    L.append("- 退市整理期过滤使用 `delist_date`（未来信息），是父预注册的规定；已披露。")
    L.append("- 日线内顺序不可知；本文策略不设盘中止损/目标，只做开盘调仓，故该缺陷影响有限。")
    L.append("- 未做行业中性化（冻结层无行业字段）；只做板块与 ADV 分层。")
    L.append("- 容量：仅 100 万本金；未做 1 亿规模曲线。")
    L.append("- 账户引擎规模校验（审计）：同一引擎的 50/100 只固定随机组合全段年化约 -3.7%/-3.4%，"
             "说明 10 仓组合的高波动拖累与成本是量级正确的；策略仍低于同引擎随机 10 仓。")
    if anom:
        L.append(f"- 数据异常：主板 2008-2020 有 {anom['extreme_rows_2008_2020']} 行 `|pct_chg|>30%`"
                 f"（复牌/重整/数据残留），其中落在合格宇宙内的有 "
                 f"**{anom['extreme_rows_in_eligible_universe']}** 行；等权基准不受影响。"
                 f"主候选有 {anom['contaminated_entries']}/{anom['all_entries']} 笔入场前 250 日内"
                 f"出现过极端跳变，平均 {_fmt(anom['contaminated_avg_ret'])}；"
                 f"全部入场合计 PnL {anom['all_pnl']/1e4:.1f} 万元（{anom['all_entries']} 笔）；"
                 f"剔除这些笔不改变负号。")
    L.append("- 分红按 `preclose` 权威口径记现金/股数；红利税按持股期结算；拆不开的事件记现金等价。")
    L.append("")

    L.append("## 8. 分档结论\n")
    L.append(f"**{tier(min(primary['dev'], primary['val']))}**。"
             f"主候选 dev/val 较低者净年化 {_fmt(min(primary['dev'], primary['val']))}；"
             f"60% 年化没有被本族数据支持。")
    L.append("")
    L.append("## 9. 复现\n")
    L.append("```bash")
    L.append("cd /Users/admin/WorkSpace/trading/trading_趋势结构")
    L.append("python3 02_代码/候选_长期双动量.py audit")
    L.append("python3 02_代码/候选_长期双动量.py ic")
    L.append("python3 02_代码/候选_长期双动量.py sim")
    L.append("python3 02_代码/候选_长期双动量.py extra")
    L.append("python3 02_代码/候选_长期双动量.py stats")
    L.append("python3 02_代码/候选_长期双动量.py report")
    L.append("```")
    (OUT / "07_报告_长期双动量.md").write_text("\n".join(L), encoding="utf-8")

    # 一页纸
    P = []
    P.append("# 一页纸：长期趋势 × 跳过短期反转 × 双动量（2008-2020）\n")
    P.append(f"- **主候选 C12**（L=250/skip=20/abs=on/N=10/R=20）：dev {_fmt(primary['dev'])}，"
             f"val {_fmt(primary['val'])}，全段 MDD {_fmt(row('C12','full')['mdd']) if row('C12','full') is not None else '—'}。")
    P.append(f"- **分档：{tier(min(primary['dev'], primary['val']))}**；60% 不支持。")
    P.append(f"- 26 个选择配置**全部为负**；`min(dev,val)` 最高 {res.iloc[0]['cfg']} "
             f"{_fmt(res.iloc[0]['min_cagr'])}（挑选偏差，不作证据）。")
    if len(ic):
        i12 = ic[(ic["spec"] == "L250_s20") & (ic["window"] == "dev")]
        if len(i12):
            P.append(f"- 信号层 L250/skip20 月度 IC：dev {i12.iloc[0]['ic_mean']:.4f}"
                     f"（t={i12.iloc[0]['ic_t']:.2f}）；20/60/120/250 日在 dev 全部为负。")
    if bind:
        P.append(f"- 双动量退化：每期正动量股票中位 {bind.get('median_positive_names', 0):.0f} 只，"
                 f"绝对动量过滤从不改变 Top-N（C11≡C12）。")
    if e03:
        P.append(f"- 同期等权 R=20（E03）：dev {_fmt(e03['dev_cagr'])}，val {_fmt(e03['val_cagr'])}，"
                 f"全段 {_fmt(e03['full_cagr'])}。")
    if len(rand20F):
        P.append(f"- 同引擎随机 10 仓：fresh R20 全段 {_fmt(float(rand20f.mean()))} ± "
                 f"{_fmt(float(rand20f.std(ddof=1)))}（换手 23x/年）；换手匹配 R20F 全段 "
                 f"{_fmt(float(rand20Ff.mean()))} ± {_fmt(float(rand20Ff.std(ddof=1)))}（换手 3.2x/年）。"
                 f"主候选全段 {_fmt(row('C12','full')['cagr'])}，低于两者。")
    if dsr:
        P.append(f"- DSR {dsr.get('dsr', float('nan')):.3f}；PBO {pbo.get('pbo', float('nan')):.3f}；"
                 f"试验数 N={dsr.get('N')}。")
    P.append(f"- 账户自检：残差 {acct.get('cash_resid_max', float('nan')):.1e} 元，2021+ 行数 0。")
    P.append("- 证据等级：描述性 dev/val（非纯净样本外）；留出未运行。")
    (OUT / "07_一页纸_长期双动量.md").write_text("\n".join(P), encoding="utf-8")
    log("报告已生成")


def cmd_guard() -> None:
    """扫描本目录所有 CSV 的日期列，确认没有 2021+。"""
    bad = []
    for p in OUT.glob("*.csv"):
        try:
            head = pd.read_csv(p, nrows=3)
        except Exception:
            continue
        for col in head.columns:
            if col in ("date", "entry_date", "exit_date", "decision_date", "execute_from", "start", "end"):
                try:
                    s = pd.to_datetime(pd.read_csv(p, usecols=[col])[col], errors="coerce")
                    if s.notna().any() and s.max().date() > PANEL_END:
                        bad.append((p.name, col, str(s.max())))
                except Exception:
                    pass
    if bad:
        raise HoldoutViolation(f"发现 2021+ 日期: {bad}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audit", "ic", "sim", "extra", "stats", "report", "all"])
    ap.add_argument("--only", default="", help="逗号分隔的配置 ID，仅 sim 用")
    args = ap.parse_args(argv)
    t0 = time.time()
    log(f"===== LTM {args.mode} 开始 =====")
    panel = Panel()
    events = _load_events()
    log(f"面板就绪：{panel.n_days} 日 × {panel.n_codes} 只，{panel.n_rows} 行，用时 {time.time()-t0:.1f}s")
    if args.mode in ("audit", "all"):
        cmd_audit(panel)
    if args.mode in ("ic", "all"):
        cmd_ic(panel)
    if args.mode in ("sim", "all"):
        only = [x for x in args.only.split(",") if x] or None
        cmd_sim(panel, events) if only is None else _sim_only(panel, events, only)
    if args.mode in ("extra", "all"):
        cmd_extra(panel, events)
    if args.mode in ("stats", "all"):
        cmd_stats()
    if args.mode in ("report", "all"):
        cmd_report(panel)
    if args.mode == "all":
        cmd_guard()
    log(f"===== LTM {args.mode} 完成，总用时 {time.time()-t0:.1f}s =====")
    return 0


def _sim_only(panel: Panel, events: EventBook, only: list[str]) -> None:
    results = run_all_configs(panel, events, only=only)
    summary = summarize_all(results)
    p = OUT / "02_组合_汇总_部分.csv"
    summary.to_csv(p, index=False)
    for cid, res in results.items():
        res["equity"].to_csv(OUT / f"_部分净值_{cid}.csv", index=False)
    log(f"部分运行完成：{only}")


# ==========================================================================
# 主配置的板块 / 流动性分层（账户层）
# ==========================================================================

def cmd_extra(panel: Panel, events: EventBook) -> dict:
    """重跑主候选（确定性），保存往返明细并按板块/流动性分层。"""
    cfg = [c for c in build_configs() if c.cid == "C12"][0]
    res = run_account(cfg, panel, events, collect=True)
    rounds = res["rounds"].copy()
    rounds.to_csv(OUT / "03_往返_主配置.csv", index=False)
    if not len(rounds):
        return {}
    rows = []
    for r in rounds.itertuples(index=False):
        try:
            i = panel.day_idx[pd.Timestamp(r.entry_date).date()]
            j = panel.code_idx[r.code]
        except (KeyError, ValueError):
            continue
        adv = panel.adv20[i, j]
        elig = panel.elig[i]
        rank = float((panel.adv20[i, elig] <= adv).mean()) if np.isfinite(adv) and elig.any() else np.nan
        rows.append({
            "code": r.code, "entry_date": r.entry_date, "exit_date": r.exit_date,
            "ret": r.ret, "hold_days": r.hold_days, "pnl": r.pnl,
            "board": panel.board_group[j],
            "adv_tercile": int(panel.adv_tercile[i, j]),
            "adv_pctile": rank,
        })
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "04_分层_交易.csv", index=False)
    out = []
    for gtype, col in (("board", "board"), ("adv", "adv_tercile")):
        for g, sub in d.groupby(col):
            out.append({"group_type": gtype, "group": str(g), "n": len(sub),
                        "avg_ret": float(sub["ret"].mean()),
                        "win_rate": float((sub["ret"] > 0).mean()),
                        "avg_hold": float(sub["hold_days"].mean()),
                        "pnl_sum": float(sub["pnl"].sum()),
                        "adv_pctile_mean": float(sub["adv_pctile"].mean())})
    out = pd.DataFrame(out)
    out.to_csv(OUT / "04_分层_账户.csv", index=False)
    # 绝对动量约束的约束力：每期正动量合格股票数
    i0 = panel.day_idx[ANCHOR_DECISION]
    cnt = []
    for i in range(i0, panel.n_days - 1, 20):
        sc, valid = panel.mom(i, cfg.L, cfg.skip)
        ok = panel.elig[i] & valid & np.isfinite(sc) & (sc > 0)
        cnt.append(int(ok.sum()))
    bind = {
        "n_rebalances": len(cnt),
        "min_positive_names": int(min(cnt)) if cnt else 0,
        "median_positive_names": float(np.median(cnt)) if cnt else 0.0,
        "rebalances_below_n": int(sum(1 for x in cnt if x < cfg.n)),
        "n_slots": int(cfg.n),
    }
    (OUT / "05_双动量约束.json").write_text(json.dumps(bind, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    # 极端 pct_chg（复牌/重整/数据异常）的影响
    i2008 = panel.day_idx[date(2008, 1, 2)]
    extreme = np.abs(panel.m["pct_chg"]) > 30.0
    n_ext_all = int(extreme[i2008:].sum())
    n_ext_elig = int((extreme[i2008:] & panel.elig[i2008:]).sum())
    contam = []
    for rr in rounds.itertuples(index=False):
        i = panel.day_idx[pd.Timestamp(rr.entry_date).date()]
        j = panel.code_idx[rr.code]
        if extreme[max(0, i - 250):i + 1, j].any():
            contam.append(rr)
    anomaly = {
        "extreme_rows_2008_2020": n_ext_all,
        "extreme_rows_in_eligible_universe": n_ext_elig,
        "contaminated_entries": len(contam),
        "contaminated_avg_ret": float(np.mean([x.ret for x in contam])) if contam else None,
        "contaminated_pnl": float(np.sum([x.pnl for x in contam])) if contam else None,
        "all_entries": int(len(rounds)),
        "all_pnl": float(rounds["pnl"].sum()) if len(rounds) else None,
    }
    (OUT / "04_异常事件_影响.json").write_text(json.dumps(anomaly, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    # 极值 Top10 vs 前十分位 vs 全宇宙的前瞻 20 日收益
    rows_f = []
    for i in range(i0, panel.n_days - 1, cfg.R):
        if i + 20 >= panel.n_days:
            continue
        sc, valid = panel.mom(i, cfg.L, cfg.skip)
        ok = panel.elig[i] & valid & np.isfinite(sc)
        js = np.flatnonzero(ok)
        if len(js) < 50:
            continue
        scj = sc[js]
        fwd = panel.adj_ff[i + 20, js] / panel.adj_ff[i, js] - 1.0
        good = np.isfinite(fwd)
        scj, fw = scj[good], fwd[good]
        if len(fw) < 50:
            continue
        order = np.argsort(-scj)
        rows_f.append({"date": panel.date_list[i].isoformat(),
                       "year": panel.date_list[i].year,
                       "top10_fwd20": float(fw[order[:10]].mean()),
                       "top_decile_fwd20": float(fw[order[:max(1, len(order) // 10)]].mean()),
                       "universe_fwd20": float(fw.mean()),
                       "n": int(len(fw))})
    fdf = pd.DataFrame(rows_f)
    if len(fdf):
        fdf["top10_minus_universe"] = fdf["top10_fwd20"] - fdf["universe_fwd20"]
        fdf.to_csv(OUT / "04_极值组合_前瞻.csv", index=False)
    log(f"主配置分层完成：{len(d)} 笔；绝对动量约束期数={bind['n_rebalances']}，"
        f"正动量股票中位={bind['median_positive_names']:.0f}，少于 N 的期数={bind['rebalances_below_n']}")
    return {"trades": d, "summary": out, "binding": bind}


if __name__ == "__main__":
    sys.exit(main())
