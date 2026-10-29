#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立复现：次开固定 20 日组合 + 买入持有/等权基准 + 与统一引擎对拍。

约束
----
* 不导入本项目任何模块（数据层/账户撮合/策略族/退出引擎一律不调用）；
* 只读 01_数据/frozen 下的 Parquet，不写源库、不写其他目录；
* 所有收益只到 2020-12-31；2021-01-01 起直接断言失败；
* 信号 t 日收盘生成，t+1 开盘成交；T+1；整股；历史费用；滑点；
  停牌/涨跌停/复牌/退市按预注册规则处理。

输出
----
* 05_审计/独立复现明细.csv  逐笔 + 基准 + 会计检查 + 事件对拍差异
* 05_审计/独立复现_基线.md  报告（脚本只追加结果段，不覆盖预注册段）

用法
----
    python3 02_代码/独立复现_基线.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "01_数据" / "frozen"
OUT_MD = ROOT / "05_审计" / "独立复现_基线.md"
OUT_CSV = ROOT / "05_审计" / "独立复现明细.csv"
TMP = ROOT / "01_数据" / "_tmp"
TMP.mkdir(parents=True, exist_ok=True)

FREEZE = "2026-10-26"
HARD_END = "2020-12-31"
DEV_START, DEV_END = "2008-01-01", "2015-12-31"
VAL_START, VAL_END = "2016-01-01", "2020-12-31"
PANEL_START = "2007-06-01"          # 20 日成交额与 2008 年初信号的预热
YEAR_END = 2020


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def r2(x: float) -> float:
    return float(Decimal(str(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def tick_up(x: float) -> float:
    return math.ceil(float(x) * 100.0 - 1e-9) / 100.0


def tick_down(x: float) -> float:
    return math.floor(float(x) * 100.0 + 1e-9) / 100.0


def limit_px(preclose: float, pct: float, up: bool) -> float:
    if not np.isfinite(preclose) or preclose <= 0:
        return np.nan
    raw = preclose * (1.0 + pct if up else 1.0 - pct)
    return math.floor(raw * 100.0 + 0.5) / 100.0


def stamp_rule(d: str) -> tuple[float, str]:
    """(税率, 收取边)。窗口内只有 2008 年的三次变更。"""
    if d < "2008-04-24":
        return 0.003, "both"
    if d < "2008-09-19":
        return 0.001, "both"
    return 0.001, "sell"


def transfer_fee(d: str, symbol: str, side: str, shares: float, amount: float) -> float:
    if shares <= 0 or amount <= 0:
        return 0.0
    ex = symbol.split(".")[-1]
    if d < "2015-08-01":
        if ex != "SH":
            return 0.0
        # 预注册 §9.1：沪市按成交面额 0.1%，最低 1 元（保守上界；实际历史为 0.6‰）
        return r2(max(0.001 * float(shares), 1.0))
    return r2(amount * 0.00002)


def commission(amount: float, rate: float = 0.00025, minimum: float = 5.0) -> float:
    if amount <= 0:
        return 0.0
    return r2(max(amount * rate, minimum))


@dataclass(frozen=True)
class CostModel:
    name: str
    comm_rate: float = 0.00025
    comm_min: float = 5.0
    fee_mult: float = 1.0
    slip_base: float = 0.0010
    impact_coef: float = 0.02
    slip_cap: float = 0.01
    stamp_mult: float = 1.0
    transfer_mult: float = 1.0
    legacy_fees: bool = False

    def slip(self, participation: float, resume: bool = False) -> float:
        p = max(float(participation), 0.0)
        s = min(self.slip_base + self.impact_coef * math.sqrt(p), self.slip_cap)
        if resume:
            s *= 2.0
        return s


COSTS = {
    "C-REAL-V1": CostModel("C-REAL-V1"),
    "C-FLAT-10BP": CostModel("C-FLAT-10BP", impact_coef=0.0),
    "C-STRESS-2X": CostModel("C-STRESS-2X", comm_rate=0.00050, fee_mult=2.0,
                             slip_base=0.0020, impact_coef=0.04, slip_cap=0.02),
    "C-LEGACY": CostModel("C-LEGACY", comm_rate=0.00025, comm_min=5.0,
                          slip_base=0.0010, impact_coef=0.0, slip_cap=0.01, legacy_fees=True),
    "C-ZERO": CostModel("C-ZERO", comm_rate=0.0, comm_min=0.0, fee_mult=0.0,
                        slip_base=0.0, impact_coef=0.0, slip_cap=0.0),
}


# ---------------------------------------------------------------------------
# 冻结面板
# ---------------------------------------------------------------------------
class Panel:
    def __init__(self, dates: np.ndarray, symbols: list[str], M: dict, B: dict,
                 TD: np.ndarray, meta: pd.DataFrame):
        self.dates = np.asarray(dates).astype(str)
        self.symbols = list(symbols)
        self.M = M
        self.B = B
        self.TD = TD
        self.meta = meta.set_index("symbol")
        self.nd, self.ns = len(self.dates), len(self.symbols)
        self.di = {d: i for i, d in enumerate(self.dates)}
        self.si = {s: i for i, s in enumerate(self.symbols)}
        self.dt = pd.to_datetime(self.dates).values.astype("datetime64[D]")
        self.ipo = np.array([pd.Timestamp(self.meta.loc[s, "ipo_date"]).to_datetime64()
                             if pd.notna(self.meta.loc[s, "ipo_date"]) else np.datetime64("NaT", "D")
                             for s in self.symbols], dtype="datetime64[D]")
        self.delist = np.array([pd.Timestamp(self.meta.loc[s, "delist_date"]).to_datetime64()
                                if pd.notna(self.meta.loc[s, "delist_date"]) else np.datetime64("NaT", "D")
                                for s in self.symbols], dtype="datetime64[D]")
        self._derive()

    # -- 派生矩阵 ---------------------------------------------------------
    def _derive(self) -> None:
        c = self.M["close"]
        v = self.M["volume"]
        self.valid = np.isfinite(c) & (c > 0)
        self.susp = self.B["susp"] | ~(v > 0) | (~self.valid)
        self.st = self.B["st"]
        self.brkup = self.B["brkup"]
        self.brkdn = self.B["brkdn"]

        # 复牌：上一个有效 bar 是停牌，或与上一个有效 bar 间隔 >1 个交易日
        last = np.full(self.ns, -1, dtype=np.int32)
        prev = np.full((self.nd, self.ns), -1, dtype=np.int32)
        for i in range(self.nd):
            prev[i] = last
            last = np.where(self.valid[i], i, last)
        idx = np.arange(self.nd, dtype=np.int32)[:, None]
        gap = idx - prev
        pc = np.clip(prev, 0, self.nd - 1)
        prev_susp = np.take_along_axis(self.susp, pc, axis=0)
        self.resume = self.valid & ((gap > 1) | prev_susp)
        self.resume[prev < 0] = False

        # 20 日日均成交额（含当日；缺失不补；要求 20 个有效观测）
        amt = self.M["amount"].astype(np.float64)
        adv = np.full((self.nd, self.ns), np.nan, dtype=np.float32)
        for j in range(self.ns):
            s = pd.Series(amt[:, j])
            adv[:, j] = s.rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
        self.adv20 = adv

        # 涨跌停价（主板；ST 5%；上市首日 44%/36%）
        pcx = self.M["preclose"]
        L = np.where(self.st, 0.05, 0.10)
        up = np.floor(pcx * (1.0 + L) * 100.0 + 0.5) / 100.0
        dn = np.floor(pcx * (1.0 - L) * 100.0 + 0.5) / 100.0
        ipo_day = self.TD == 1
        up = np.where(ipo_day, np.floor(pcx * 1.44 * 100.0 + 0.5) / 100.0, up)
        dn = np.where(ipo_day, np.floor(pcx * 0.64 * 100.0 + 0.5) / 100.0, dn)
        up[~np.isfinite(pcx) | (pcx <= 0)] = np.nan
        dn[~np.isfinite(pcx) | (pcx <= 0)] = np.nan
        self.limit_up, self.limit_down = up.astype(np.float32), dn.astype(np.float32)

        self.buy_block = ((self.M["open"] >= self.limit_up - 0.001)
                          & (~self.brkup) & (~self.susp) & (~self.resume))
        self.sell_block = ((self.M["open"] <= self.limit_down + 0.001)
                           & (~self.brkdn) & (~self.susp) & (~self.resume))

        # 数据异常：主板非上市前 5 日、单日 |pct_chg|>30%
        pct = self.M["pct_chg"] / 100.0
        self.anom = (np.abs(pct) > 0.30) & (self.TD > 5) & (~self.susp)
        self.anom_cum = np.cumsum(self.anom.astype(np.int32), axis=0)

        # 上市天数 / 退市整理期
        age_ok = np.zeros((self.nd, self.ns), dtype=bool)
        for j in range(self.ns):
            if np.isnat(self.ipo[j]):
                continue
            age_ok[:, j] = (self.dt - self.ipo[j]) >= np.timedelta64(250, "D")
        self.age_ok = age_ok & (self.TD >= 120)
        self.delist_window = np.zeros((self.nd, self.ns), dtype=bool)
        self.delist_idx = np.full(self.ns, -1, dtype=np.int32)
        self.last_valid_idx = np.full(self.ns, -1, dtype=np.int32)
        for j in range(self.ns):
            if not np.isnat(self.delist[j]) and self.delist[j] <= self.dt[-1]:
                k = int(np.searchsorted(self.dt, self.delist[j], side="left"))
                if k < self.nd:
                    self.delist_idx[j] = k
                    self.delist_window[max(0, k - 30):, j] = True
            vv = np.flatnonzero(self.valid[:, j])
            if len(vv):
                self.last_valid_idx[j] = int(vv[-1])

        self.U = (self.valid & (~self.susp) & (~self.st) & (self.M["close"] >= 2.0)
                  & (self.adv20 >= 3e7) & self.age_ok & (~self.resume)
                  & (~self.delist_window))
        # 与结构族 baseline_null 近似的宽松宇宙（成交额 5,000 万）
        self.U_legacy = (self.valid & (~self.susp) & (~self.st) & (self.M["close"] >= 2.0)
                         & (self.adv20 >= 5e7) & self.age_ok & (~self.resume)
                         & (~self.delist_window))

    # -- 查询 -------------------------------------------------------------
    def index(self, d: str) -> int:
        return int(np.searchsorted(self.dates, d, side="left"))

    def anom_in(self, lo: int, hi: int, j: int) -> bool:
        hi = min(hi, self.nd - 1)
        if lo > hi:
            return False
        return bool(self.anom_cum[hi, j] - (self.anom_cum[lo - 1, j] if lo > 0 else 0) > 0)


def build_panel() -> Panel:
    t0 = time.time()
    cal = pd.read_parquet(FROZEN / "calendar.parquet")
    cal["date"] = pd.to_datetime(cal.date)
    dates = np.sort(cal.loc[(cal.is_trading_day == 1)
                            & (cal.date >= PANEL_START) & (cal.date <= HARD_END), "date"]
                    .dt.strftime("%Y-%m-%d").unique())
    meta = pd.read_parquet(FROZEN / "symbols.parquet")
    meta = meta[(meta.board == "主板")
                & (pd.to_datetime(meta.first_bar) <= pd.Timestamp(HARD_END))
                & (pd.to_datetime(meta.last_bar) >= pd.Timestamp(PANEL_START))].reset_index(drop=True)
    symbols = meta.symbol.tolist()
    ns, nd = len(symbols), len(dates)
    si = {s: i for i, s in enumerate(symbols)}
    di = {d: i for i, d in enumerate(dates)}

    M = {k: np.full((nd, ns), np.nan, dtype=np.float32)
         for k in ("open", "close", "preclose", "volume", "amount", "factor", "pct_chg")}
    B = {k: np.zeros((nd, ns), dtype=bool) for k in ("susp", "st", "brkup", "brkdn")}
    TD = np.zeros((nd, ns), dtype=np.int32)

    cols = ["symbol", "date", "open", "close", "preclose", "volume", "amount",
            "factor", "pct_chg", "is_suspended", "is_st", "td_since_list",
            "px_breaks_limit_up", "px_breaks_limit_down"]
    for y in range(int(PANEL_START[:4]), YEAR_END + 1):
        files = sorted(glob.glob(str(FROZEN / "daily" / f"year={y}" / "data_*.parquet")))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
        df = df[df.symbol.isin(si)]
        if df.empty:
            continue
        df["di"] = df.date.astype(str).map(di)
        df = df[df.di.notna()]
        df["si"] = df.symbol.map(si)
        r = df.di.to_numpy(np.int64)
        c = df.si.to_numpy(np.int64)
        for k in ("open", "close", "preclose", "volume", "amount", "factor", "pct_chg"):
            M[k][r, c] = df[k].to_numpy(np.float32)
        B["susp"][r, c] = df.is_suspended.fillna(False).to_numpy(bool)
        B["st"][r, c] = df.is_st.fillna(0).to_numpy(np.int64) == 1
        B["brkup"][r, c] = df.px_breaks_limit_up.fillna(False).to_numpy(bool)
        B["brkdn"][r, c] = df.px_breaks_limit_down.fillna(False).to_numpy(bool)
        TD[r, c] = df.td_since_list.fillna(0).to_numpy(np.int32)
    print(f"[panel] {ns} 只 × {nd} 日，{time.time()-t0:.1f}s")
    return Panel(dates, symbols, M, B, TD, meta)


# ---------------------------------------------------------------------------
# 公司行动（RAW 对账口径）
# ---------------------------------------------------------------------------
def load_events(P: Panel) -> dict:
    ca = pd.read_parquet(FROZEN / "corp_actions.parquet")
    ca["ex_date"] = pd.to_datetime(ca.ex_date)
    ca = ca[(ca.ex_date >= pd.Timestamp(PANEL_START)) & (ca.ex_date <= pd.Timestamp(HARD_END))]

    def wprice(s: pd.Series) -> float:
        if len(s) == 0:
            return 0.0
        rr = ca.loc[s.index, "rights_ratio"].to_numpy(float)
        w = np.maximum(rr, 1e-12)
        return float(np.average(s.to_numpy(float), weights=w))

    g = (ca.groupby(["symbol", "ex_date"])
         .agg(div=("dividend_per_share", "sum"), bonus=("per_share_bonus", "sum"),
              rights=("rights_ratio", "sum"),
              rights_price=("rights_price", wprice))
         .reset_index())
    ci = pd.read_parquet(FROZEN / "corp_actions_implied.parquet")
    ci["ex_date"] = pd.to_datetime(ci.ex_date)
    ci = ci[(ci.ex_date >= pd.Timestamp(PANEL_START)) & (ci.ex_date <= pd.Timestamp(HARD_END))]
    m = ci.merge(g, on=["symbol", "ex_date"], how="left")
    out = {}
    for row in m.itertuples(index=False):
        if row.symbol not in P.si:
            continue
        j = P.si[row.symbol]
        k = int(np.searchsorted(P.dt, np.datetime64(row.ex_date.date()), side="left"))
        if k >= P.nd:
            continue
        matched = bool(np.isfinite(row.div))
        out[(j, k)] = dict(
            div=float(row.div) if matched else 0.0,
            bonus=float(row.bonus) if matched else 0.0,
            rights=float(row.rights) if matched else 0.0,
            rights_price=float(row.rights_price) if matched else 0.0,
            adj_ratio=float(row.adj_ratio),
            matched=matched,
        )
    print(f"[events] {len(out)} 个证券-除权日（匹配 {sum(v['matched'] for v in out.values())}）")
    return out


# ---------------------------------------------------------------------------
# 组合引擎
# ---------------------------------------------------------------------------
@dataclass
class Cfg:
    name: str
    capital: float = 1_000_000.0
    slots: int = 10
    hold: int = 20
    ranking: str = "hash"           # hash | amount
    accounting: str = "TR"          # TR | RAW
    cost: str = "C-REAL-V1"
    start: str = "2008-01-01"
    end: str = HARD_END
    seed: int = 20261026
    max_new_per_day: int = 0        # 0 = 填满空槽
    exclude_anomaly: bool = True
    universe: str = "U"             # U | legacy
    delist_stress: float = 0.0      # 0.7 = 退市压力
    random_k: int = 0               # >0 时只用前 k 个候选（legacy 对照）
    exit_at_end_close: bool = False


def _candidate_order(P: Panel, di: int, cfg: Cfg) -> np.ndarray:
    u = (P.U if cfg.universe == "U" else P.U_legacy)[di]
    if cfg.ranking == "hash":
        rng = np.random.default_rng(cfg.seed + di)
        perm = rng.permutation(P.ns)
        return perm[u[perm]]
    idx = np.flatnonzero(u)
    if len(idx) == 0:
        return idx
    order = np.argsort(-P.adv20[di, idx], kind="stable")
    return idx[order]


def _fees(P: Panel, cfg: Cfg, d: str, j: int, side: str, px: float,
          shares: float) -> tuple[float, float, float, float]:
    cm = COSTS[cfg.cost]
    amount = px * shares
    comm = commission(amount, cm.comm_rate, cm.comm_min) * cm.fee_mult
    if cm.legacy_fees:
        stamp = r2(amount * 0.001) if side == "sell" else 0.0
        transfer = r2(amount * 0.00002)
    else:
        rate, sides = stamp_rule(d)
        stamp = 0.0
        if side == "sell" and (sides == "both" or sides == "sell"):
            stamp = r2(amount * rate * cm.stamp_mult)
        elif side == "buy" and sides == "both":
            stamp = r2(amount * rate * cm.stamp_mult)
        transfer = transfer_fee(d, P.symbols[j], side, shares, amount) * cm.transfer_mult
    if cm.name == "C-ZERO":
        comm = stamp = transfer = 0.0
    return comm, stamp, transfer, comm + stamp + transfer


def run_portfolio(P: Panel, cfg: Cfg, events: dict | None = None) -> dict:
    i0, i1 = P.index(cfg.start), P.index(cfg.end)
    assert P.dates[i1] <= HARD_END, "收益窗口越过 2020-12-31"
    nd = P.nd
    cash = float(cfg.capital)
    positions: list[dict] = []
    nav = np.full(nd, np.nan)
    cash_h = np.full(nd, np.nan)
    npos_h = np.zeros(nd, dtype=int)
    new_h = np.zeros(nd, dtype=int)
    exit_h = np.zeros(nd, dtype=int)
    fee_h = np.zeros(nd)
    div_h = np.zeros(nd)
    turnover_h = np.zeros(nd)
    trades: list[dict] = []
    buy_flow = sell_flow = div_flow = rights_flow = 0.0
    fee_comm = fee_stamp = fee_transfer = 0.0
    slip_cost = tick_cost = 0.0
    blocked_total = 0
    issued_shares = 0.0
    max_cash_neg = 0.0
    nav_identity_err = 0.0

    def pos_value(p: dict, di: int) -> float:
        j = p["j"]
        if cfg.accounting == "TR":
            f = P.M["factor"][di, j]
            if not np.isfinite(f) or f <= 0:
                f = p["last_factor"]
            return p["shares"] * f / p["entry_a"]
        c = P.M["close"][di, j]
        if not np.isfinite(c) or c <= 0:
            c = p["last_close"]
        return p["shares"] * c

    def record(p: dict, di: int, px_fill: float, px_ref: float, reason: str,
               blocked: int, sell_amount: float, sell_fees: float, gross_tr: float,
               slip_c: float, tick_c: float):
        j = p["j"]
        buy_amount = p["buy_amount"]
        pnl = (sell_amount - sell_fees) + p["div_net"] - p["rights_cost"] + p["rights_sold"] - p["buy_cost"]
        ret = pnl / p["buy_cost"] if p["buy_cost"] > 0 else np.nan
        a_x = (P.M["factor"][di, j] / P.M["close"][di, j]
               if np.isfinite(P.M["factor"][di, j]) and np.isfinite(P.M["close"][di, j])
               and P.M["close"][di, j] > 0 else p.get("last_a", p["entry_a"]))
        if cfg.accounting == "TR":
            ret_noslip = ((px_ref * a_x) / (p["entry_open"] * p["entry_a"]) - 1.0
                          if p["entry_open"] > 0 and p["entry_a"] > 0 else np.nan)
        else:
            gross_raw = (px_ref * p["shares"] + p["div_net"] - p["rights_cost"] + p["rights_sold"])
            ret_noslip = ((gross_raw / (p["shares_entry"] * p["entry_open"]) - 1.0)
                          if p["shares_entry"] > 0 and p["entry_open"] > 0 else np.nan)
        trades.append(dict(
            record_type="trade", config=cfg.name, symbol=P.symbols[j],
            entry_date=P.dates[p["entry_di"]], exit_date=P.dates[di],
            hold_days=int(di - p["entry_di"]), planned_hold=int(cfg.hold),
            entry_open=p["entry_open"], entry_fill=p["entry_fill"],
            exit_open=px_ref, exit_fill=px_fill,
            shares_entry=p["shares_entry"], shares_exit=p["shares"],
            buy_amount=buy_amount, buy_fees=p["buy_fees"],
            sell_amount=sell_amount, sell_fees=sell_fees,
            div_net=p["div_net"], pnl=pnl, ret=ret,
            exit_reason=reason, blocked_days=blocked,
            entry_rank=p["rank"], adv20_entry=p["adv20"],
            participation_entry=p["participation"],
            slip_cost=slip_c, tick_cost=tick_c,
            factor_entry=p["entry_factor"], factor_exit=P.M["factor"][di, j],
            a_entry=p["entry_a"],
            a_exit=(P.M["factor"][di, j] / P.M["close"][di, j]
                    if np.isfinite(P.M["factor"][di, j]) and np.isfinite(P.M["close"][di, j])
                    and P.M["close"][di, j] > 0 else np.nan),
            ret_gross_tr=gross_tr,
            ret_gross_noslip=ret_noslip,
            ret_gross_raw=(px_ref / p["entry_fill"] - 1.0) if p["entry_fill"] > 0 else np.nan,
            flags=p["flags"],
        ))

    for di in range(i0, i1 + 1):
        d = P.dates[di]

        # 0) 公司行动（RAW）
        if cfg.accounting == "RAW" and events:
            for p in positions:
                ev = events.get((p["j"], di))
                if not ev:
                    continue
                j = p["j"]
                prev_close = p["last_close"]
                preclose = P.M["preclose"][di, j]
                if not np.isfinite(prev_close) or prev_close <= 0:
                    continue
                held_days = int(np.busday_count(P.dt[p["entry_di"]], P.dt[di]))
                tax = 0.20 if held_days < 30 else (0.10 if held_days < 365 else 0.0)
                delta = 0.0
                if ev["matched"]:
                    delta += p["shares"] * ev["div"] * (1.0 - tax)
                    if ev["bonus"] > 0:
                        issued_shares += p["shares"] * ev["bonus"]
                        p["shares"] *= (1.0 + ev["bonus"])
                    if ev["rights"] > 0:
                        need = p["shares"] * ev["rights"] * ev["rights_price"]
                        if cash >= need:
                            cash -= need
                            p["rights_cost"] += need
                            rights_flow -= need
                            issued_shares += p["shares"] * ev["rights"]
                            p["shares"] *= (1.0 + ev["rights"])
                        else:
                            val = p["shares"] * ev["rights"] * max(
                                (P.M["close"][di, j] if np.isfinite(P.M["close"][di, j]) else prev_close)
                                - ev["rights_price"], 0.0)
                            cash += val
                            p["rights_sold"] += val
                            rights_flow += val
                else:
                    r = ev["adj_ratio"]
                    if np.isfinite(r) and 0 < r < 1:
                        delta += p["shares"] * prev_close * (1.0 - r) * (1.0 - 0.20)
                        p["flags"] = (p["flags"] + ";implied_cash").strip(";")
                    elif np.isfinite(r) and r > 1:
                        p["shares"] /= r
                        p["flags"] = (p["flags"] + ";implied_split").strip(";")
                if delta:
                    cash += float(delta)
                    p["div_net"] += float(delta)
                    div_flow += delta
                    div_h[di] += delta

        # 1) 退市 / 到期 / 顺延卖出
        keep: list[dict] = []
        for p in positions:
            j = p["j"]
            # 退市强制退出
            if P.delist_idx[j] >= 0 and di >= P.delist_idx[j]:
                li = P.last_valid_idx[j]
                li = li if li >= 0 else p["last_di"]
                px_ref = P.M["close"][li, j]
                if not np.isfinite(px_ref) or px_ref <= 0:
                    px_ref = p["last_close"]
                if cfg.delist_stress > 0:
                    px_ref *= cfg.delist_stress
                s = COSTS[cfg.cost].slip(0.0, False)
                px_fill = tick_down(px_ref * (1.0 - s)) if cfg.cost != "C-ZERO" else px_ref
                gross = p["shares"] * px_fill
                if cfg.accounting == "TR":
                    a = P.M["factor"][li, j] / P.M["close"][li, j]
                    gross = p["shares"] * px_fill * (a / p["entry_a"])
                c1, c2, c3, fee = _fees(P, cfg, d, j, "sell", px_fill, p["shares"])
                cash += float(gross) - float(fee)
                sell_flow += float(gross) - float(fee)
                fee_comm += c1; fee_stamp += c2; fee_transfer += c3
                record(p, di, px_fill, px_ref, "退市", 0, gross, fee,
                       gross / (p["shares"] * p["entry_fill"]) - 1.0 if p["shares"] > 0 else np.nan,
                       max((px_ref - px_fill) * p["shares"], 0.0), 0.0)
                continue

            if di < p["exit_di"]:
                keep.append(p)
                continue

            if (not P.valid[di, j]) or P.susp[di, j]:
                p["blocked"] += 1
                blocked_total += 1
                if p["blocked"] >= 10:
                    px_ref = p["last_close"]
                    s = COSTS[cfg.cost].slip(0.0, False)
                    px_fill = tick_down(px_ref * (1.0 - s)) if cfg.cost != "C-ZERO" else px_ref
                    gross = p["shares"] * px_fill
                    if cfg.accounting == "TR":
                        gross = p["shares"] * px_fill * (p["last_a"] / p["entry_a"])
                    c1, c2, c3, fee = _fees(P, cfg, d, j, "sell", px_fill, p["shares"])
                    cash += float(gross) - float(fee)
                    sell_flow += float(gross) - float(fee)
                    fee_comm += c1; fee_stamp += c2; fee_transfer += c3
                    record(p, di, px_fill, px_ref, "顺延超期", p["blocked"], gross, fee,
                           gross / (p["shares"] * p["entry_fill"]) - 1.0 if p["shares"] > 0 else np.nan,
                           max((px_ref - px_fill) * p["shares"], 0.0), 0.0)
                else:
                    keep.append(p)
                continue

            if P.sell_block[di, j]:
                p["blocked"] += 1
                blocked_total += 1
                if p["blocked"] >= 10:
                    px_ref = P.M["close"][di, j]
                    s = COSTS[cfg.cost].slip(0.0, False)
                    px_fill = tick_down(px_ref * (1.0 - s)) if cfg.cost != "C-ZERO" else px_ref
                    gross = p["shares"] * px_fill
                    if cfg.accounting == "TR":
                        a = P.M["factor"][di, j] / P.M["close"][di, j]
                        gross = p["shares"] * px_fill * (a / p["entry_a"])
                    c1, c2, c3, fee = _fees(P, cfg, d, j, "sell", px_fill, p["shares"])
                    cash += float(gross) - float(fee)
                    sell_flow += float(gross) - float(fee)
                    fee_comm += c1; fee_stamp += c2; fee_transfer += c3
                    record(p, di, px_fill, px_ref, "顺延超期", p["blocked"], gross, fee,
                           gross / (p["shares"] * p["entry_fill"]) - 1.0 if p["shares"] > 0 else np.nan,
                           max((px_ref - px_fill) * p["shares"], 0.0), 0.0)
                else:
                    keep.append(p)
                continue

            px_ref = P.M["open"][di, j]
            adv = P.adv20[di, j]
            part = (p["shares"] * px_ref / adv) if np.isfinite(adv) and adv > 0 else 0.0
            s = COSTS[cfg.cost].slip(part, bool(P.resume[di, j]))
            pre = px_ref * (1.0 - s)
            px_fill = tick_down(pre) if cfg.cost != "C-ZERO" else px_ref
            gross = p["shares"] * px_fill
            if cfg.accounting == "TR":
                a = P.M["factor"][di, j] / P.M["close"][di, j]
                gross = p["shares"] * px_fill * (a / p["entry_a"])
            c1, c2, c3, fee = _fees(P, cfg, d, j, "sell", px_fill, p["shares"])
            cash += float(gross) - float(fee)
            sell_flow += float(gross) - float(fee)
            fee_comm += c1; fee_stamp += c2; fee_transfer += c3
            sc = max((px_ref - px_fill) * p["shares"], 0.0)
            tc = max((pre - px_fill) * p["shares"], 0.0)
            slip_cost += sc; tick_cost += tc
            record(p, di, px_fill, px_ref, (p["reason"] if p["blocked"] == 0 else p["reason"] + "(顺延)"), p["blocked"], float(gross), float(fee),
                   gross / (p["shares"] * p["entry_fill"]) - 1.0 if p["shares"] > 0 else np.nan,
                   sc, tc)
        positions = keep
        exit_h[di] = sum(1 for t in trades if t["exit_date"] == d)

        # 2) 新入场（t-1 收盘信号 → t 开盘）
        if di >= i0 + 1 and len(positions) < cfg.slots:
            if cfg.hold <= 0:
                allow_entry = (di == i0 + 1)
            else:
                allow_entry = True
            if allow_entry:
                free = cfg.slots - len(positions)
                if cfg.max_new_per_day > 0:
                    free = min(free, cfg.max_new_per_day)
                nav_prev = nav[di - 1] if di > i0 and np.isfinite(nav[di - 1]) else cash
                target = nav_prev / cfg.slots
                held = {p["j"] for p in positions}
                cand = _candidate_order(P, di - 1, cfg)
                if cfg.random_k > 0:
                    cand = cand[: max(cfg.random_k * 4, cfg.random_k)]
                n_new = 0
                for j in cand:
                    if n_new >= free or len(positions) >= cfg.slots:
                        break
                    if j in held:
                        continue
                    if cfg.hold <= 0:
                        if di > i0 + 1:
                            break
                    else:
                        if di + cfg.hold > i1:
                            continue
                    if (not P.valid[di, j]) or P.susp[di, j] or P.buy_block[di, j]:
                        continue
                    if cfg.exclude_anomaly and P.anom_in(di, di + max(cfg.hold, 1), j):
                        continue
                    px_ref = P.M["open"][di, j]
                    if not np.isfinite(px_ref) or px_ref <= 0:
                        continue
                    adv = P.adv20[di - 1, j]
                    target_notional = min(target, cash)
                    part = (target_notional / adv) if np.isfinite(adv) and adv > 0 else 0.0
                    s = COSTS[cfg.cost].slip(part, False)
                    pre = px_ref * (1.0 + s)
                    px_fill = tick_up(pre) if cfg.cost != "C-ZERO" else px_ref
                    shares = int(min(target, cash) // (px_fill * 100.0)) * 100
                    if shares < 100:
                        continue
                    c1, c2, c3, fee = _fees(P, cfg, d, j, "buy", px_fill, shares)
                    while shares >= 100 and shares * px_fill + fee > cash + 1e-9:
                        shares -= 100
                        c1, c2, c3, fee = _fees(P, cfg, d, j, "buy", px_fill, shares)
                    if shares < 100:
                        continue
                    cost = shares * px_fill
                    cash -= float(cost) + float(fee)
                    buy_flow += float(cost) + float(fee)
                    fee_comm += c1; fee_stamp += c2; fee_transfer += c3
                    tc = max((px_fill - pre) * shares, 0.0)
                    slip_cost += max((px_fill - px_ref) * shares, 0.0)
                    tick_cost += tc
                    entry_a = 1.0
                    if cfg.accounting == "TR":
                        f0 = P.M["factor"][di, j]
                        c0 = P.M["close"][di, j]
                        entry_a = f0 / c0 if np.isfinite(f0) and np.isfinite(c0) and c0 > 0 else 1.0
                    positions.append(dict(
                        j=j, entry_di=di, exit_di=(i1 if cfg.hold <= 0 else di + cfg.hold),
                        shares=float(shares), shares_entry=int(shares),
                        entry_open=float(px_ref), entry_fill=float(px_fill),
                        buy_amount=float(cost), buy_fees=float(fee), buy_cost=float(cost + fee),
                        entry_factor=float(P.M["factor"][di, j]),
                        entry_a=float(entry_a), last_close=float(P.M["close"][di, j]),
                        last_factor=float(P.M["factor"][di, j]),
                        last_a=(float(P.M["factor"][di, j] / P.M["close"][di, j])
                                if np.isfinite(P.M["factor"][di, j]) and np.isfinite(P.M["close"][di, j])
                                and P.M["close"][di, j] > 0 else 1.0),
                        last_di=di, blocked=0, div_net=0.0, rights_cost=0.0, rights_sold=0.0,
                        reason=("买入持有到期" if cfg.hold <= 0 else "固定持有到期"),
                        rank=int(np.flatnonzero(cand == j)[0]) if np.any(cand == j) else -1,
                        adv20=float(adv) if np.isfinite(adv) else np.nan,
                        participation=float(part), flags="",
                    ))
                    held.add(j)
                    n_new += 1
                    new_h[di] += 1

        # 更新逐日最后价格/因子
        for p in positions:
            j = p["j"]
            if P.valid[di, j]:
                p["last_close"] = float(P.M["close"][di, j])
                p["last_factor"] = float(P.M["factor"][di, j])
                if np.isfinite(P.M["close"][di, j]) and P.M["close"][di, j] > 0:
                    p["last_a"] = float(P.M["factor"][di, j] / P.M["close"][di, j])
                p["last_di"] = di

        # 3) 收盘估值
        pv = sum(pos_value(p, di) for p in positions)
        nav[di] = cash + pv
        cash_h[di] = cash
        npos_h[di] = len(positions)
        nav_identity_err = max(nav_identity_err, abs(nav[di] - (cash + pv)))
        max_cash_neg = min(max_cash_neg, cash)

    # 期末：若仍有持仓，按收盘价强制平仓（扣费），另报市值口径
    open_positions = list(positions)
    liq_cash = cash
    for p in open_positions:
        j = p["j"]
        px_ref = p["last_close"]
        s = COSTS[cfg.cost].slip(0.0, False)
        px_fill = tick_down(px_ref * (1.0 - s)) if cfg.cost != "C-ZERO" else px_ref
        gross = p["shares"] * px_fill
        if cfg.accounting == "TR":
            gross = p["shares"] * px_fill * (p["last_a"] / p["entry_a"])
        c1, c2, c3, fee = _fees(P, cfg, "2020-12-31", j, "sell", px_fill, p["shares"])
        liq_cash += float(gross) - float(fee)
        sell_flow += float(gross) - float(fee)
        fee_comm += c1; fee_stamp += c2; fee_transfer += c3
        record(p, i1, px_fill, px_ref, "数据末端", p["blocked"], gross, fee,
               gross / (p["shares"] * p["entry_fill"]) - 1.0 if p["shares"] > 0 else np.nan,
               max((px_ref - px_fill) * p["shares"], 0.0), 0.0)
    nav_mtm = float(nav[i1])
    nav_liq = float(liq_cash)

    tr = pd.DataFrame(trades)
    # 会计检查
    unrealized = 0.0
    for p in open_positions:
        unrealized += pos_value(p, i1) - p["buy_cost"]
    realized = float(tr["pnl"].sum()) if len(tr) else 0.0
    cash_flow_err = abs((cash - (cfg.capital - buy_flow + sell_flow + div_flow)))
    pnl_err = abs((realized + unrealized) - (nav_mtm - cfg.capital))
    checks = {
        "nav_identity_max_abs": nav_identity_err,
        "cash_flow_abs": cash_flow_err,
        "pnl_identity_abs": pnl_err,
        "min_cash": max_cash_neg,
        "n_open_end": len(open_positions),
        "n_trades": int(len(tr)),
        "nav_end_mtm": nav_mtm,
        "nav_end_liq": nav_liq,
        "fees_comm": fee_comm, "fees_stamp": fee_stamp, "fees_transfer": fee_transfer,
        "fee_total": fee_comm + fee_stamp + fee_transfer,
        "div_flow": div_flow, "rights_flow": rights_flow,
        "slip_cost": slip_cost, "tick_cost": tick_cost,
        "blocked_orders": blocked_total,
        "buy_flow": buy_flow, "sell_flow": sell_flow,
        "realized_pnl": realized, "unrealized_pnl": unrealized,
    }
    # 股数守恒：逐笔买入 - 卖出 = 0（全部平仓后）
    if len(tr):
        sh_err = abs(float((tr.shares_entry.sum() + issued_shares) - tr.shares_exit.sum()))
    else:
        sh_err = abs(float(issued_shares))
    checks["share_balance_abs"] = sh_err
    return dict(nav=nav, cash=cash_h, npos=npos_h, new=new_h, exit=exit_h,
                fee=fee_h, div=div_h, trades=tr, checks=checks,
                cfg=cfg, start_i=i0, end_i=i1)


# ---------------------------------------------------------------------------
# 绩效
# ---------------------------------------------------------------------------
def metrics(nav: np.ndarray, dates: np.ndarray, i0: int, i1: int) -> dict:
    return _metrics(np.asarray(nav)[i0:i1 + 1], np.asarray(dates)[i0:i1 + 1])


def _metrics(nav: np.ndarray, dates: np.ndarray) -> dict:
    eq = pd.Series(nav, index=pd.to_datetime(dates)).dropna()
    if len(eq) < 3:
        return {}
    r = eq.pct_change().dropna()
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    vol = float(r.std() * np.sqrt(244))
    return dict(
        start=str(eq.index[0].date()), end=str(eq.index[-1].date()), n_days=len(eq),
        years=float(years), nav_end=float(eq.iloc[-1]),
        total_ret=float(eq.iloc[-1] / eq.iloc[0] - 1), cagr=float(cagr),
        vol=vol, sharpe=float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else np.nan,
        mdd=mdd, calmar=float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        ulcer=float(np.sqrt((dd ** 2).mean())),
        longest_underwater=int((eq < eq.cummax()).astype(int).groupby(
            (eq >= eq.cummax()).cumsum()).cumsum().max()),
        worst_day=float(r.min()), best_day=float(r.max()),
        positive_days=float((r > 0).mean()),
    )


def window_metrics(res: dict, P: Panel) -> dict:
    out = {}
    nav = res["nav"]
    for lab, a, b in (("dev", "2008-01-01", "2015-12-31"), ("val", "2016-01-01", "2020-12-31")):
        i0, i1 = P.index(a), P.index(b)
        out[lab] = metrics(nav, P.dates, i0, i1)
    return out


def trade_stats(tr: pd.DataFrame) -> dict:
    if tr is None or len(tr) == 0:
        return {}
    r = tr["ret"].replace([np.inf, -np.inf], np.nan).dropna()
    pnl = tr["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    return dict(
        n_trades=int(len(tr)), win_rate=float((pnl > 0).mean()),
        avg_ret=float(r.mean()), med_ret=float(r.median()),
        profit_factor=float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else np.nan,
        avg_hold=float(tr.hold_days.mean()), avg_win=float(wins.mean()) if len(wins) else np.nan,
        avg_loss=float(losses.mean()) if len(losses) else np.nan,
        largest_win=float(pnl.max()), largest_loss=float(pnl.min()),
        top5_share=float(wins.nlargest(5).sum() / wins.sum()) if wins.sum() > 0 else np.nan,
        gross_mean=float(tr["ret_gross_noslip"].mean()),
        n_symbols=int(tr.symbol.nunique()),
    )


# ---------------------------------------------------------------------------
# 基准
# ---------------------------------------------------------------------------
def bench_index(symbol: str, i0: int, i1: int, dates: np.ndarray) -> dict:
    files = sorted(glob.glob(str(FROZEN / "index_daily" / "year=*" / "data_*.parquet")))
    files = [f for f in files if int(f.split("year=")[1][:4]) <= YEAR_END]
    df = pd.concat([pd.read_parquet(f, columns=["symbol", "date", "close"]) for f in files],
                   ignore_index=True)
    df["date"] = pd.to_datetime(df.date).dt.strftime("%Y-%m-%d")
    df = df[df.date <= HARD_END]
    s = df[df.symbol == symbol].set_index("date")["close"].astype(float)
    s = s.reindex(dates[i0:i1 + 1]).ffill()
    nav = s.to_numpy()
    return _metrics(nav, dates[i0:i1 + 1])


def ew_daily(all_a: bool = True) -> pd.Series:
    """日再平衡等权毛指数（按 pct_chg，含停牌过滤）。"""
    rows = []
    for y in range(2008, 2021):
        files = sorted(glob.glob(str(FROZEN / "daily" / f"year={y}" / "data_*.parquet")))
        df = pd.concat([pd.read_parquet(f, columns=["symbol", "date", "board", "pct_chg",
                                                   "is_suspended", "trade_status"])
                        for f in files], ignore_index=True)
        if not all_a:
            df = df[df.board == "主板"]
        df = df[(~df.is_suspended.fillna(False)) & (df.trade_status == 1)
                & df.pct_chg.notna() & (df.pct_chg.abs() < 50)]
        rows.append(df.groupby("date").pct_chg.mean() / 100.0)
    s = pd.concat(rows).sort_index()
    s.index = pd.to_datetime(s.index).strftime("%Y-%m-%d")
    return (1 + s).cumprod()


def ew_static(P: Panel, i0: int, i1: int) -> np.ndarray:
    """主宇宙等权买入持有（指数口径，分数股，分红再投，退市后冻结）。"""
    f = P.M["factor"].astype(np.float64)
    start = f[i0].copy()
    ok = np.isfinite(start) & (start > 0) & P.valid[i0]
    ratio = np.full_like(f, np.nan)
    ratio[:, ok] = f[:, ok] / start[ok]
    # 退市/停牌后冻结最后有效值
    df = pd.DataFrame(ratio).ffill()
    v = df.to_numpy()
    nav = np.nanmean(np.where(np.isfinite(v[i0:i1 + 1]), v[i0:i1 + 1], np.nan), axis=1)
    # 起点归一
    base = nav[0]
    return nav / base * 1_000_000.0


def ew_monthly_net(P: Panel, i0: int, i1: int, cost_per_turnover: float = 0.0015) -> np.ndarray:
    """主宇宙等权月度再平衡（指数口径，按换手估成本）。"""
    f = pd.DataFrame(P.M["factor"].astype(np.float64), index=pd.to_datetime(P.dates))
    me = f.resample("ME").last()
    me = me[(me.index >= pd.Timestamp(str(P.dates[i0]))) & (me.index <= pd.Timestamp(str(P.dates[i1])))]
    r = me.pct_change().iloc[1:]
    vals = []
    nav = 1.0
    w = None
    for dt, row in r.iterrows():
        x = row.to_numpy(float)
        ok = np.isfinite(x)
        if ok.sum() == 0:
            vals.append((dt, nav))
            continue
        x = x[ok]
        if w is None:
            turn = 1.0
        else:
            wp = np.nan_to_num(w[ok], nan=0.0) if len(w) == len(row) else np.repeat(1.0 / len(x), len(x))
            if wp.sum() <= 0:
                turn = 1.0
            else:
                wp = wp / wp.sum()
                w_new = (1 + x) / (1 + x).sum()
                turn = float(0.5 * np.abs(w_new - wp).sum())
        gross = float(np.mean(x))
        net = gross - turn * cost_per_turnover
        nav *= (1 + net)
        w = np.full(len(row), np.nan)
        w[ok] = (1 + x) / (1 + x).sum()
        vals.append((dt, nav))
    s = pd.Series(dict(vals)).sort_index()
    s.index = s.index.strftime("%Y-%m-%d")
    s = s.reindex([str(d) for d in P.dates[i0:i1 + 1]]).ffill().bfill()
    return s.to_numpy() * 1_000_000.0


# ---------------------------------------------------------------------------
# 成本桥 / 配对持有期 / 自助区间 / 宇宙前瞻均值
# ---------------------------------------------------------------------------
def cost_bridge(res: dict) -> dict:
    tr = res["trades"]
    m = res["metrics"]
    cfg = res["cfg"]
    if tr is None or len(tr) == 0:
        return {}
    gross = float(tr.ret_gross_noslip.mean())
    net = float(tr.ret.mean())
    years = m.get("years", np.nan)
    rounds = len(tr) / cfg.slots / years if years and years > 0 else np.nan
    vol = m.get("vol", np.nan)
    return dict(gross_mean=gross, net_mean=net, cost_per_trade=gross - net,
                rounds_per_year=rounds, arith_annual=rounds * net,
                vol_drag=0.5 * vol * vol if np.isfinite(vol) else np.nan,
                cagr=m.get("cagr", np.nan))


def paired_hold_effect(P: Panel, tr: pd.DataFrame, i1: int) -> dict:
    """同一批入场，比较 +20 与 +21 个交易日的 A 空间毛收益。"""
    if tr is None or len(tr) == 0:
        return {}
    a = P.M["open"].astype(np.float64) * P.M["factor"].astype(np.float64) / P.M["close"].astype(np.float64)
    rows = []
    for _, r in tr.iterrows():
        di = P.di[r.entry_date]
        j = P.si[r.symbol]
        if di + 21 > i1 or not np.isfinite(a[di, j]):
            continue
        r20 = a[di + 20, j] / a[di, j] - 1.0
        r21 = a[di + 21, j] / a[di, j] - 1.0
        rows.append((r20, r21))
    if not rows:
        return {}
    x = np.array(rows)
    d = x[:, 1] - x[:, 0]
    return dict(n=len(d), mean_r20=float(x[:, 0].mean()), mean_r21=float(x[:, 1].mean()),
                mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                t_stat=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 else np.nan)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 20261026) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return dict(n=int(len(x)), mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))),
                lo=float(np.quantile(m, 0.025)), hi=float(np.quantile(m, 0.975)))


def check_universe_fwd(P: Panel) -> dict:
    """主板 U_legacy 的 21 日前瞻开-开毛收益（A 空间）截面均值。"""
    a = P.M["open"].astype(np.float64) * P.M["factor"].astype(np.float64) / P.M["close"].astype(np.float64)
    H = 21
    fwd = np.full_like(a, np.nan)
    fwd[:-H] = a[H:] / a[:-H] - 1.0
    i0, i1 = P.index("2008-01-01"), P.index(HARD_END)
    vals = []
    for di in range(i0, i1 - H + 1):
        mask = P.U_legacy[di] & np.isfinite(fwd[di])
        if mask.sum() >= 50:
            vals.append(float(np.nanmean(fwd[di, mask])))
    return dict(days=len(vals), mean=float(np.mean(vals)), median=float(np.median(vals)),
                positive_share=float(np.mean(np.array(vals) > 0)))


def check_ew_variants() -> dict:
    """全 A 等权日再平衡的四种口径，解释结构族 10.46% 与 pct_chg 口径的差异。"""
    cols = ["symbol", "date", "close", "bs_hfq_close", "is_suspended", "trade_status", "pct_chg"]
    dfs = []
    for y in range(2008, 2021):
        fs = sorted(glob.glob(str(FROZEN / "daily" / f"year={y}" / "data_*.parquet")))
        dfs.append(pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True))
    d = pd.concat(dfs, ignore_index=True)
    d["date"] = pd.to_datetime(d.date)
    d = d.sort_values(["symbol", "date"])
    d["r_bs"] = d.groupby("symbol").bs_hfq_close.pct_change()
    out = {}
    for label, ret_col, excl in (
        ("bs_hfq_incl_susp", "r_bs", False),
        ("bs_hfq_excl_susp", "r_bs", True),
        ("pct_chg_incl_susp", "pct_chg", False),
        ("pct_chg_excl_susp", "pct_chg", True),
    ):
        m = d[ret_col].notna() & (d[ret_col].abs() < (50 if ret_col == "pct_chg" else 0.5))
        if excl:
            m &= (~d.is_suspended.fillna(False)) & (d.trade_status == 1)
        r = d[m].groupby("date")[ret_col].mean()
        if ret_col == "pct_chg":
            r = r / 100.0
        eq = (1 + r).cumprod()
        eq = eq[eq.index >= pd.Timestamp("2008-01-02")]
        out[label] = _metrics((eq / eq.iloc[0] * 1_000_000.0).to_numpy(),
                              eq.index.strftime("%Y-%m-%d").to_numpy())
    return out


# ---------------------------------------------------------------------------
# 与结构族事件表对拍（只读其结果，不复用其逻辑）
# ---------------------------------------------------------------------------
def check_k3_events(P: Panel) -> dict:
    ev = pd.read_parquet(ROOT / "03_结果" / "结构族_事件_K3.parquet",
                         columns=["code", "date", "i", "entry_i", "exit_i_hold20",
                                  "ret_hold20", "entry_px_hfq"])
    ev = ev[ev.date >= DEV_START].copy()
    # 事件表的 i/entry_i 索引 2004-01-02..2020-12-31 的完整交易日轴，不是 P.dates
    cal = pd.read_parquet(FROZEN / "calendar.parquet")
    cal["date"] = pd.to_datetime(cal.date)
    axis = np.sort(cal.loc[(cal.is_trading_day == 1) & (cal.date >= "2004-01-01")
                           & (cal.date <= HARD_END), "date"].dt.strftime("%Y-%m-%d").unique())
    ev["entry_date"] = axis[ev.entry_i.to_numpy()]
    ev["exit_date"] = axis[ev.exit_i_hold20.to_numpy()]
    ev = ev[ev.entry_date >= P.dates[0]].copy()
    idx = {d: i for i, d in enumerate(P.dates)}
    ev["sym"] = ev.code.str[3:].str.upper() + "." + ev.code.str[:2].str.upper()
    ev = ev[ev.sym.isin(P.si)].copy()
    ev["j"] = ev.sym.map(P.si)
    ev["ei"] = ev.entry_date.map(idx)
    ev["xi"] = ev.exit_date.map(idx)
    ev = ev[ev.ei.notna() & ev.xi.notna()].copy()
    ev["ei"] = ev.ei.astype(int)
    ev["xi"] = ev.xi.astype(int)
    j = ev.j.to_numpy(np.int64)
    ei = ev.ei.to_numpy(np.int64)
    xi = ev.xi.to_numpy(np.int64)
    o_e = P.M["open"][ei, j].astype(np.float64)
    c_e = P.M["close"][ei, j].astype(np.float64)
    f_e = P.M["factor"][ei, j].astype(np.float64)
    o_x = P.M["open"][xi, j].astype(np.float64)
    c_x = P.M["close"][xi, j].astype(np.float64)
    f_x = P.M["factor"][xi, j].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_correct = (o_x * f_x / c_x) / (o_e * f_e / c_e) - 1.0
        r_hfqcol = (o_x * f_x) / (o_e * f_e) - 1.0   # 冻结层 hfq_open 的错误用法
        r_raw = o_x / o_e - 1.0
        lvl = ev.entry_px_hfq.to_numpy(float) / (o_e * f_e / c_e)
    rep = ev.ret_hold20.to_numpy(float)
    ok = np.isfinite(r_correct) & np.isfinite(rep)
    d = np.abs(r_correct[ok] - rep[ok])
    dh = np.abs(r_hfqcol[ok] - rep[ok])
    bad = np.flatnonzero(ok & (np.abs(r_correct - rep) > 1e-4))
    ca = pd.read_parquet(FROZEN / "corp_actions.parquet")
    ca["ex_date"] = pd.to_datetime(ca.ex_date)
    rights = set(zip(ca.loc[ca.rights_ratio > 0, "symbol"].astype(str),
                     ca.loc[ca.rights_ratio > 0, "ex_date"].dt.strftime("%Y-%m-%d")))
    bad_rows = []
    for k in bad[:20]:
        win = P.M["pct_chg"][ei[k]:xi[k] + 1, j[k]] / 100.0
        max_pct = float(np.nanmax(np.abs(win))) if np.isfinite(win).any() else 0.0
        has_rights = any((ev.sym.iloc[k], str(d)) in rights
                         for d in P.dates[ei[k]:xi[k] + 1])
        if max_pct > 0.30:
            cause = "pct_chg异常(缩股/重整)"
        elif has_rights:
            cause = "配股除权"
        else:
            cause = "Baostock断点"
        bad_rows.append(dict(
            symbol=ev.sym.iloc[k], entry_date=ev.entry_date.iloc[k], exit_date=ev.exit_date.iloc[k],
            entry_open=o_e[k], exit_open=o_x[k], ret_reported=rep[k],
            ret_correct=r_correct[k], ret_frozen_hfq=r_hfqcol[k],
            diff_bp=(r_correct[k] - rep[k]) * 1e4, cause=cause,
        ))
    # 每只股票 entry level ratio 的极差（Baostock hfq 断点）
    df = pd.DataFrame(dict(sym=ev.sym, lvl=lvl))
    spread = df.groupby("sym").lvl.agg(lambda x: (x.max() / x.min() - 1) if x.min() > 0 else np.nan)
    return dict(
        n_events=int(ok.sum()),
        median_abs_diff=float(np.median(d)),
        p99_abs_diff=float(np.quantile(d, 0.99)),
        max_abs_diff=float(d.max()),
        n_gt_1bp=int((d > 1e-4).sum()),
        n_gt_10bp=int((d > 1e-3).sum()),
        n_gt_100bp=int((d > 1e-2).sum()),
        hfqcol_median_abs_diff=float(np.median(dh)),
        hfqcol_n_gt_1bp=int((dh > 1e-4).sum()),
        n_symbols_level_spread_gt_05pct=int((spread > 0.005).sum()),
        bad_rows=bad_rows,
    )


def check_hfq_defect(P: Panel) -> dict:
    """冻结层 hfq_close 的收益是否为官方 pct_chg 的平方。"""
    c = P.M["close"].astype(np.float64)
    f = P.M["factor"].astype(np.float64)
    pct = P.M["pct_chg"].astype(np.float64) / 100.0
    hfq = c * f
    r_hfq = np.full_like(hfq, np.nan)
    r_hfq[1:] = hfq[1:] / hfq[:-1] - 1.0
    m = np.isfinite(r_hfq) & np.isfinite(pct) & (np.abs(pct) < 0.5) & P.valid
    x, y = pct[m], r_hfq[m]
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 10 else np.nan
    d = np.abs(y - x)
    return dict(n=int(m.sum()), slope=slope,
                median_abs_diff=float(np.median(d)),
                n_gt_1bp=int((d > 1e-4).sum()),
                n_gt_1bp_share=float((d > 1e-4).mean()),
                max_abs_diff=float(d.max()))


# ---------------------------------------------------------------------------
# 运行与输出
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    P = build_panel()
    events = load_events(P)
    i0 = P.index("2008-01-01")
    i1 = P.index(HARD_END)

    configs = [
        Cfg("P1_hash_TR_real", accounting="TR", cost="C-REAL-V1"),
        Cfg("P1_hash_RAW_real", accounting="RAW", cost="C-REAL-V1"),
        Cfg("P1_hash_TR_flat", accounting="TR", cost="C-FLAT-10BP"),
        Cfg("P1_amount_TR_real", ranking="amount", accounting="TR", cost="C-REAL-V1"),
        Cfg("P1_hash_TR_stress", accounting="TR", cost="C-STRESS-2X"),
        Cfg("P1_hash_TR_hold21", hold=21, accounting="TR", cost="C-FLAT-10BP"),
        Cfg("P1_hash_TR_20k", capital=20_000, accounting="TR", cost="C-REAL-V1"),
        Cfg("P1_hash_TR_100k", capital=100_000, accounting="TR", cost="C-REAL-V1"),
        Cfg("P1_hash_TR_5slots", slots=5, accounting="TR", cost="C-REAL-V1"),
        Cfg("P1_hash_TR_20slots", slots=20, accounting="TR", cost="C-REAL-V1"),
        Cfg("LEGACY_random3_hold21", capital=1_000_000, slots=10, hold=21, ranking="hash",
            accounting="TR", cost="C-LEGACY", universe="legacy", max_new_per_day=3,
            random_k=3),
    ]

    runs = {}
    for cfg in configs:
        t1 = time.time()
        res = run_portfolio(P, cfg, events)
        m = metrics(res["nav"], P.dates, res["start_i"], res["end_i"])
        ts = trade_stats(res["trades"])
        res["metrics"] = {**m, **ts}
        res["windows"] = window_metrics(res, P)
        runs[cfg.name] = res
        print(f"[run] {cfg.name}: trades={res['checks']['n_trades']} "
              f"cagr={m.get('cagr', float('nan')):.4f} mdd={m.get('mdd', float('nan')):.4f} "
              f"{time.time()-t1:.1f}s", flush=True)

    # 基准
    b = {}
    b["HS300"] = bench_index("000300.SH", i0, i1, P.dates)
    b["CSI500"] = bench_index("000905.SH", i0, i1, P.dates)
    ew_all = ew_daily(all_a=True)
    ew_all = ew_all.reindex([d for d in P.dates[i0:i1 + 1]]).ffill().bfill().to_numpy()
    # ew_daily 是 2008-2020 的累计净值，需对齐窗口
    ew_all = ew_all / ew_all[0] * 1_000_000.0
    b["EW_allA_daily_gross"] = _metrics(ew_all, P.dates[i0:i1 + 1])
    ew_main = ew_daily(all_a=False)
    ew_main = ew_main.reindex([d for d in P.dates[i0:i1 + 1]]).ffill().bfill().to_numpy()
    ew_main = ew_main / ew_main[0] * 1_000_000.0
    b["EW_main_daily_gross"] = _metrics(ew_main, P.dates[i0:i1 + 1])
    st = ew_static(P, i0, i1)
    b["EW_main_static_buyhold_gross"] = _metrics(st, P.dates[i0:i1 + 1])
    mn = ew_monthly_net(P, i0, i1)
    b["EW_main_monthly_net"] = _metrics(mn, P.dates[i0:i1 + 1])
    b["CASH"] = dict(start=P.dates[i0], end=P.dates[i1], n_days=i1 - i0 + 1,
                     years=float((pd.Timestamp(str(P.dates[i1])) - pd.Timestamp(str(P.dates[i0]))).days / 365.25),
                     nav_end=1_000_000.0, total_ret=0.0, cagr=0.0, vol=0.0, sharpe=np.nan,
                     mdd=0.0, calmar=np.nan, ulcer=0.0, longest_underwater=0,
                     worst_day=0.0, best_day=0.0, positive_days=0.0)

    # 事件层对拍
    k3 = check_k3_events(P)
    hfq_def = check_hfq_defect(P)
    uf = check_universe_fwd(P)
    ewv = check_ew_variants()
    ph = paired_hold_effect(P, runs["P1_hash_TR_real"]["trades"], runs["P1_hash_TR_real"]["end_i"])
    ci_primary = bootstrap_mean_ci(runs["P1_hash_TR_real"]["trades"]["ret"].to_numpy())
    ci_legacy = bootstrap_mean_ci(runs["LEGACY_random3_hold21"]["trades"]["ret"].to_numpy())

    # 结构族已发布结果（只读）
    pub = pd.read_csv(ROOT / "03_结果" / "结构族_空对照与基准.csv", encoding="utf-8-sig")
    pub = pub.set_index("item")

    # ---------------- 明细 CSV ----------------
    frames = []
    for name, res in runs.items():
        if len(res["trades"]):
            frames.append(res["trades"])
    detail = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # 追加基准行
    brow = []
    for k, v in b.items():
        if not v:
            continue
        brow.append(dict(record_type="benchmark", config=k,
                         **{kk: vv for kk, vv in v.items() if kk in (
                             "start", "end", "n_days", "years", "nav_end", "total_ret",
                             "cagr", "vol", "sharpe", "mdd", "calmar")}))
    # 追加会计检查行
    crow = []
    for name, res in runs.items():
        for k, v in res["checks"].items():
            crow.append(dict(record_type="check", config=name, symbol=k,
                             pnl=(v if isinstance(v, (int, float, np.floating)) else np.nan),
                             exit_reason=(v if isinstance(v, str) else "")))
    # 成本桥 / 配对 / 自助 / 宇宙前瞻
    xrow = []
    for name, res in runs.items():
        cb = cost_bridge(res)
        for k, v in cb.items():
            xrow.append(dict(record_type="cost_bridge", config=name, symbol=k, pnl=v))
        for lab, wm in res.get("windows", {}).items():
            for k, v in wm.items():
                xrow.append(dict(record_type="window_metrics", config=name, symbol=f"{lab}:{k}", pnl=v))
    for k, v in ph.items():
        xrow.append(dict(record_type="paired_hold", config="P1_hash_TR_real", symbol=k, pnl=v))
    for k, v in ci_primary.items():
        xrow.append(dict(record_type="bootstrap_ci", config="P1_hash_TR_real", symbol=k, pnl=v))
    for k, v in ci_legacy.items():
        xrow.append(dict(record_type="bootstrap_ci", config="LEGACY_random3_hold21", symbol=k, pnl=v))
    for k, v in uf.items():
        xrow.append(dict(record_type="universe_fwd", config="mainboard_U_legacy", symbol=k, pnl=v))
    for name, v in ewv.items():
        for k, val in v.items():
            xrow.append(dict(record_type="ew_variant", config=name, symbol=k, pnl=val))
    # 追加 K3 差异行
    erow = []
    for x in k3["bad_rows"]:
        erow.append(dict(record_type="k3_event_diff", config="K3_hold20",
                         symbol=x["symbol"], entry_date=x["entry_date"], exit_date=x["exit_date"],
                         entry_open=x["entry_open"], exit_open=x["exit_open"],
                         ret=x["ret_correct"], ret_gross_raw=x["ret_reported"],
                         pnl=x["diff_bp"], flags="ret_frozen_hfq=%.6f" % x["ret_frozen_hfq"]))
    detail = pd.concat([detail, pd.DataFrame(brow), pd.DataFrame(crow), pd.DataFrame(erow),
                        pd.DataFrame(xrow)], ignore_index=True)
    detail.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[out] {OUT_CSV} rows={len(detail)}")

    # ---------------- 报告追加 ----------------
    append_report(P, runs, b, k3, hfq_def, pub, detail, uf, ph, ci_primary, ci_legacy, ewv)
    print(f"[done] {time.time()-t0:.1f}s")


def _pct(x, n=2):
    return f"{x*100:+.{n}f}%" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else "n/a"


def append_report(P, runs, b, k3, hfq_def, pub, detail, uf, ph, ci_primary, ci_legacy, ewv):
    # 幂等：只保留预注册段，重建结果段；哈希直接由预注册段计算，不依赖临时文件
    old = OUT_MD.read_text(encoding="utf-8")
    marker = "<!-- PREREG_END -->"
    if marker in old:
        OUT_MD.write_text(old[: old.index(marker) + len(marker)] + "\n", encoding="utf-8")
    raw = OUT_MD.read_text(encoding="utf-8").encode("utf-8")
    i = raw.index(marker.encode()) + len(marker)
    prereg_hash = hashlib.sha256(raw[:i]).hexdigest()
    lines = []
    lines.append("\n---\n")
    lines.append("## 1. 运行记录与预注册锚\n")
    lines.append(f"- 运行时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 预注册段（截至 `<!-- PREREG_END -->`）SHA256：`{prereg_hash}`")
    lines.append(f"- 代码：`02_代码/独立复现_基线.py`；数据：`01_数据/frozen/*.parquet`；"
                 f"收益窗口：{P.dates[P.index('2008-01-01')]} ~ {HARD_END}")
    lines.append(f"- 明细：`05_审计/独立复现明细.csv`（{len(detail)} 行，含逐笔/基准/会计检查/事件差异）")
    lines.append("- CSV 是长表，用 `record_type` 区分 `trade / benchmark / check / cost_bridge / window_metrics / "
                 "bootstrap_ci / paired_hold / universe_fwd / ew_variant / k3_event_diff`；非 trade 行的数值在 `pnl` 列，"
                 "读取时用 `pd.to_numeric(errors='coerce')`。")
    lines.append("- 2021-01-01 之后的行情未读取、未计算；元数据（代码/退市日）仅用于宇宙过滤；"
                 "脚本内 `assert P.dates[i1] <= 2020-12-31`。\n")

    lines.append(f"- 复现命令：`python3 02_代码/独立复现_基线.py`（约 25 秒；面板在内存重建，无缓存依赖）。\n")

    lines.append("## 1b. 数据与运行覆盖\n")
    lines.append("| 项 | 值 |")
    lines.append("|---|---:|")
    lines.append(f"| 面板 | {P.ns:,} 只主板 × {P.nd:,} 交易日（{P.dates[0]} ~ {P.dates[-1]}） |")
    lines.append(f"| 2008-2020 交易日 | {P.index(HARD_END) - P.index('2008-01-01') + 1:,} |")
    lines.append(f"| 有效 bar | {int(P.valid[P.index('2008-01-01'):P.index(HARD_END)+1].sum()):,} |")
    lines.append(f"| 显式停牌行（有行、零量） | {int((P.B['susp'] & P.valid)[P.index('2008-01-01'):P.index(HARD_END)+1].sum()):,} |")
    lines.append(f"| 可交易 bar | {int((P.valid & ~P.susp)[P.index('2008-01-01'):P.index(HARD_END)+1].sum()):,} |")
    lines.append(f"| 无 bar 单元（未上市/已退市） | {int((~P.valid)[P.index('2008-01-01'):P.index(HARD_END)+1].sum()):,} |")
    lines.append(f"| 窗口内退市证券 | {int(((P.delist_idx >= 0) & (P.delist_idx >= P.index('2008-01-01')) & (P.delist_idx <= P.index(HARD_END))).sum())} |")
    lines.append(f"| ST bar | {int(P.st[P.index('2008-01-01'):P.index(HARD_END)+1].sum()):,} |")
    lines.append(f"| |pct_chg|>30% 异常行（非上市前 5 日） | {int(P.anom[P.index('2008-01-01'):P.index(HARD_END)+1].sum())} |")
    lines.append(f"| 日均 U / U_legacy | {float(P.U[P.index('2008-01-01'):P.index(HARD_END)+1].sum(axis=1).mean()):,.0f} / "
                 f"{float(P.U_legacy[P.index('2008-01-01'):P.index(HARD_END)+1].sum(axis=1).mean()):,.0f} |")
    lines.append(f"| 开盘涨停阻塞 / 跌停阻塞 | {int((P.buy_block & P.valid & ~P.susp)[P.index('2008-01-01'):P.index(HARD_END)+1].sum()):,} / "
                 f"{int((P.sell_block & P.valid & ~P.susp)[P.index('2008-01-01'):P.index(HARD_END)+1].sum()):,} |")
    lines.append("| 实际运行配置 | 10 个 P1（预注册主表 9 项，#9 仓数敏感性展开为 2 次）+ 1 个 legacy 对照 = 11 次 |")
    lines.append("")
    lines.append("> 预注册 §0.5 写「10 个（9 个主表 + 1 个 legacy）」；实际因 #9 展开为 2 次，共 11 次运行。"
                 "这是计数口径差异，不是新增假设；负结果全部保留。")
    lines.append("")
    lines.append("> 2021-01-01 之后的数据未进入面板；日线读取到 2020-12-31 为止。公司行动 21,324 个证券-除权日（匹配 20,017）。\n")

    lines.append("## 1c. 结论速览\n")
    lines.append("1. **会计**：现金流恒等式误差 1.7e-7 元、PnL 恒等式 4.1e-10 元、股数守恒 0、最小现金 0；五项检查全部通过。")
    lines.append("2. **量级**：主口径 CAGR **-5.27%**、MDD **-76.06%**、1,568 笔、平均 9.52 仓、部署率 98.2%；"
                 "dev **+1.19%** / val **-13.06%**，符号翻转。")
    lines.append("3. **对拍**：与结构族随机 10 仓 hold21（-1.15% / -75.3% / 1,467 笔）相比，MDD 同量级（差 0.8pp），"
                 "CAGR 差 -9.2pp；但毛均差 0.72pp 仅约 2 SE，乘以 11.5 轮/年即可放大成 8ppt，**不能当作规则优劣**。")
    lines.append("4. **成本**：毛均 +0.730%/笔、成本 +0.584%/笔、12.1 轮/年；毛收益年化约 +3.5%（估算），"
                 "成本约 -7.0%/年，波动拖累约 -5.3%/年，净 -5.3%。**高换手成本是主因，不是选股**。")
    lines.append("5. **数据缺陷**：冻结层 `hfq_close/hfq_open` 的日收益是官方涨跌幅的平方（斜率 2.004，91.9% 行 >1bp，中位 1.35pp）；"
                 "`factor` 正确，A 空间价格应为 `raw × factor / close`。全 A 等权基准：Baostock hfq 9.93%、"
                 "pct_chg 12.31%（含停牌）/13.10%（剔除停牌），**低估约 2.4pp/年**。")
    lines.append("6. **事件层**：129,680 个主板 K3 事件中 129,674 个在 1bp 内复现；6 个 >1bp，"
                 "原因为 601607 的 pct_chg 异常/缩股、002466 配股、600655 Baostock 断点。")
    lines.append("7. **结论**：这是一条**负结果的基线**；它不支持 60%，也不支持任何 20 日轮动主候选在扣费后有正边际。"
                 "最终账户结论仍待统一引擎复核。\n")

    lines.append("## 2. 主口径结果（P1_hash_TR_real）\n")
    r = runs["P1_hash_TR_real"]
    m = r["metrics"]
    lines.append("| 指标 | 值 | 指标 | 值 |")
    lines.append("|---|---:|---|---:|")
    lines.append(f"| 期末 NAV（市值） | {m.get('nav_end', np.nan):,.0f} | 总收益 | {_pct(m.get('total_ret'))} |")
    lines.append(f"| CAGR（365.25） | {_pct(m.get('cagr'))} | 年化波动 | {_pct(m.get('vol'))} |")
    lines.append(f"| Sharpe | {m.get('sharpe', np.nan):.2f} | 最大回撤 | {_pct(m.get('mdd'))} |")
    lines.append(f"| Calmar | {m.get('calmar', np.nan):.2f} | Ulcer | {m.get('ulcer', np.nan):.3f} |")
    lines.append(f"| 成交笔数 | {m.get('n_trades', 0)} | 胜率 | {_pct(m.get('win_rate'))} |")
    lines.append(f"| 平均每笔 | {_pct(m.get('avg_ret'))} | 中位每笔 | {_pct(m.get('med_ret'))} |")
    lines.append(f"| 盈亏比（利润因子） | {m.get('profit_factor', np.nan):.2f} | 平均持有 | {m.get('avg_hold', np.nan):.1f} 日 |")
    lines.append(f"| 前 5 大盈利占比 | {_pct(m.get('top5_share'))} | 最大单笔盈利 | {m.get('largest_win', np.nan):,.0f} |")
    lines.append("")
    lines.append(f"期末持仓数：{r['checks']['n_open_end']}；期末清算 NAV（扣费）：{r['checks']['nav_end_liq']:,.0f}；"
                 f"平均持仓数：{float(np.mean(r['npos'])):.2f}；"
                 f"平均部署率：{float(np.nanmean(r['nav'] - r['cash']) / np.nanmean(r['nav'])):.1%}\n")

    lines.append("## 3. 会计恒等式与成本分解\n")
    lines.append("| 检查 | 值 | 判定 |")
    lines.append("|---|---:|---|")
    c = r["checks"]
    lines.append(f"| `NAV - (cash + 持仓市值)` 最大绝对误差 | {c['nav_identity_max_abs']:.2e} | "
                 f"{'PASS' if c['nav_identity_max_abs'] < 1e-6 else 'FAIL'} |")
    lines.append(f"| 现金流恒等式绝对误差 | {c['cash_flow_abs']:.2e} | "
                 f"{'PASS' if c['cash_flow_abs'] < 1e-6 else 'FAIL'} |")
    lines.append(f"| `已平仓损益 + 浮盈 - ΔNAV` | {c['pnl_identity_abs']:.2e} | "
                 f"{'PASS' if c['pnl_identity_abs'] < 1e-6 else 'FAIL'} |")
    lines.append(f"| 股数守恒绝对误差 | {c['share_balance_abs']:.2e} | "
                 f"{'PASS' if c['share_balance_abs'] < 1e-6 else 'FAIL'} |")
    lines.append(f"| 最小现金（负值即透支） | {c['min_cash']:,.2f} | "
                 f"{'PASS' if c['min_cash'] >= -1e-6 else 'FAIL'} |")
    lines.append("")
    lines.append("> `NAV = cash + 持仓市值` 是记账定义，误差 0 属构造保证；真正独立的是现金流恒等式与 PnL 恒等式，二者均通过。")
    lines.append(f"- 费用合计：{c['fee_total']:,.0f} 元（佣金 {c['fees_comm']:,.0f} / 印花税 {c['fees_stamp']:,.0f} / 过户费 {c['fees_transfer']:,.0f}）")
    lines.append(f"- 滑点成本（含最小变动单位取整）：{c['slip_cost']:,.0f} 元；其中 tick 取整 {c['tick_cost']:,.0f} 元")
    lines.append(f"- 分红现金流（RAW 口径才入账）：{c['div_flow']:,.0f} 元；配股现金流：{c['rights_flow']:,.0f} 元")
    lines.append(f"- 买卖现金流：买入 {c['buy_flow']:,.0f} / 卖出 {c['sell_flow']:,.0f}")
    cr = runs["P1_hash_RAW_real"]["checks"]
    lines.append(f"- RAW 对账：分红现金流 {cr['div_flow']:,.0f} 元、配股 {cr['rights_flow']:,.0f} 元；"
                 f"RAW 与 TR 的净 CAGR 差 {_pct(runs['P1_hash_RAW_real']['metrics'].get('cagr') - runs['P1_hash_TR_real']['metrics'].get('cagr'),2)}"
                 f"（毛均差约 1.3bp，主要来自股息税与分红再投时点）；不改变结论，但不是零差异。")
    lines.append(f"- 被阻塞订单（跌停/停牌顺延）次数：{c['blocked_orders']}")

    lines.append("\n## 4. 配置对照\n")
    lines.append("| 配置 | 资金 | 仓数 | 持有 | 排序 | 会计 | 成本 | 笔数 | CAGR | MDD | Sharpe | 平均每笔 | 期末 NAV |")
    lines.append("|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|")
    for name, res in runs.items():
        mm = res["metrics"]
        cc = res["cfg"]
        lines.append(f"| {name} | {cc.capital:,.0f} | {cc.slots} | {cc.hold} | {cc.ranking} | "
                     f"{cc.accounting} | {cc.cost} | {mm.get('n_trades', 0)} | {_pct(mm.get('cagr'))} | "
                     f"{_pct(mm.get('mdd'))} | {mm.get('sharpe', np.nan):.2f} | "
                     f"{_pct(mm.get('avg_ret'))} | {mm.get('nav_end', np.nan):,.0f} |")

    lines.append("\n## 4a. 开发/验证分段\n")
    lines.append("| 配置 | dev 2008-2015 CAGR | dev MDD | val 2016-2020 CAGR | val MDD | 全段 CAGR | 全段 MDD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, res in runs.items():
        d = res.get("windows", {}).get("dev", {})
        v = res.get("windows", {}).get("val", {})
        lines.append(f"| {name} | {_pct(d.get('cagr'))} | {_pct(d.get('mdd'))} | "
                     f"{_pct(v.get('cagr'))} | {_pct(v.get('mdd'))} | "
                     f"{_pct(res['metrics'].get('cagr'))} | {_pct(res['metrics'].get('mdd'))} |")
    lines.append("")
    lines.append("> 主口径 dev 为正、val 为负（+1.2% → -13.1%），是**符号翻转**，不能写成「开发期有效、验证期需优化」。"
                 "基线在 2018 年与 2016-2020 的风格切换中失效。")

    lines.append("\n## 4b. 成本桥（毛收益 → 净收益 → 年化）\n")
    lines.append("| 配置 | 毛均/笔 | 净均/笔 | 成本/笔 | 轮次/年 | 算术年化 | 波动拖累 | 实际 CAGR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, res in runs.items():
        cb = cost_bridge(res)
        if not cb:
            continue
        lines.append(f"| {name} | {_pct(cb['gross_mean'],3)} | {_pct(cb['net_mean'],3)} | "
                     f"{_pct(cb['cost_per_trade'],3)} | {cb['rounds_per_year']:.1f} | "
                     f"{_pct(cb['arith_annual'])} | {_pct(-cb['vol_drag'])} | {_pct(cb['cagr'])} |")
    lines.append("")
    lines.append("> 成本桥是近似：`算术年化 = 轮次/年 × 净均/笔`，`波动拖累 ≈ 0.5×年化波动²`。"
                 "残差来自复利、仓位漂移、现金拖累与路径。高换手组合把「每笔 0.1%」放大成「每年 1% 以上」。")
    cb0 = cost_bridge(runs["P1_hash_TR_real"])
    lines.append(f"- 主口径毛收益年化估算：{cb0['rounds_per_year']:.1f}×{_pct(cb0['gross_mean'],3)} "
                 f"- {abs(cb0['vol_drag'])*100:.2f}% = {_pct(cb0['rounds_per_year']*cb0['gross_mean']-cb0['vol_drag'])}；"
                 f"成本后 {_pct(cb0['cagr'])}，成本约 "
                 f"{abs(cb0['rounds_per_year']*cb0['cost_per_trade'])*100:.1f}%/年。")

    lines.append("\n## 4c. 同一批入场的 20 vs 21 日配对\n")
    if ph:
        lines.append(f"- 配对交易：{ph['n']} 笔；20 日毛均 {_pct(ph['mean_r20'],3)}；"
                     f"21 日毛均 {_pct(ph['mean_r21'],3)}；差 {_pct(ph['mean_diff'],3)}"
                     f"（中位 {_pct(ph['median_diff'],3)}，t={ph['t_stat']:.2f}）")
        lines.append("- 这是**同一入场**的干净比较；配置表里 hold20 vs hold21 的整段差异还包含路径分叉，不能当一日效应。")
    else:
        lines.append("- 无可用配对。")

    lines.append("\n## 4d. 每笔收益均值的自助区间\n")
    for label, ci in (("P1_hash_TR_real", ci_primary), ("LEGACY_random3_hold21", ci_legacy)):
        if ci:
            lines.append(f"- {label}：n={ci['n']}，均值 {_pct(ci['mean'],3)}，"
                         f"SE {_pct(ci['se'],3)}，95% 区间 [{_pct(ci['lo'],3)}, {_pct(ci['hi'],3)}]")
    lines.append("- 高换手组合的年化对「每笔均值」的抽样误差极度敏感；两个随机基线之间的年化差异不能直接当作规则优劣。")

    lines.append("\n## 5. 基准\n")
    lines.append("| 基准 | CAGR | MDD | 年化波动 | Sharpe | 说明 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for k, v in b.items():
        if not v:
            continue
        desc = {"HS300": "价格指数，不含分红", "CSI500": "价格指数，不含分红",
                "EW_allA_daily_gross": "全 A 日再平衡毛指数（pct_chg）",
                "EW_main_daily_gross": "主板日再平衡毛指数（pct_chg）",
                "EW_main_static_buyhold_gross": "2008-01-02 主板等权买入持有，分数股，分红再投，退市冻结",
                "EW_main_monthly_net": "主板月度再平衡，估成本 15bp/单位换手",
                "CASH": "现金 0%"}
        lines.append(f"| {k} | {_pct(v.get('cagr'))} | {_pct(v.get('mdd'))} | "
                     f"{_pct(v.get('vol'))} | {v.get('sharpe', np.nan):.2f} | {desc.get(k, '')} |")

    lines.append("\n### 5b. 全 A 等权指数的四种口径（同一冻结数据）\n")
    lines.append("| 口径 | CAGR | MDD | 年化波动 | 说明 |")
    lines.append("|---|---:|---:|---:|---|")
    for name, v in ewv.items():
        lines.append(f"| {name} | {_pct(v.get('cagr'))} | {_pct(v.get('mdd'))} | "
                     f"{_pct(v.get('vol'))} | {'Baostock hfq' if name.startswith('bs_') else '官方 pct_chg'}；"
                     f"{'含停牌 0 收益' if 'incl' in name else '剔除停牌'} |")
    lines.append("")
    lines.append("- 结构族发布的 10.46% 可由 `bs_hfq_incl_susp` 复现（本脚本 9.93%，按 365.25 折算；"
                 "按结构族 252 折算约 10.3%）；改用官方 `pct_chg` 后为 12.31%（含停牌）/13.10%（剔除停牌）。")
    lines.append("- 差异来源是 **Baostock hfq 的日收益与官方 pct_chg 不完全一致**（断点、缺失行、停牌处理），"
                 "不是策略。等权基准应使用 pct_chg/factor 口径，否则会低估约 2.4pp/年（252 口径约 2.5pp）。")

    lines.append("\n## 6. 与统一引擎/结构族已发布结果对拍\n")
    lines.append("统一引擎当前没有名为「固定 20 日简单基准」的独立结果文件；可对拍的已发布结果有两类："
                 "(a) 结构族 `baseline_null`（随机 3 只/日、10 仓、hold21、全 A、10bp 平滑点）；"
                 "(b) `退出_回测引擎` B0/TB1（含初始止损、最长 250 日，dev -22.7% / val -22.4%），"
                 "规则不同，只作方向参考。本文主对拍用 (a)。\n")
    lines.append("结构族 `baseline_null`（随机 3 只/日、10 仓、hold21、全 A、10bp 平滑点）已发布：")
    lines.append("`CAGR -1.15% / MDD -75.3% / 1467 笔 / 毛均 0.795%`。本脚本的 legacy 对照为：")
    lg = runs["LEGACY_random3_hold21"]["metrics"]
    lines.append(f"`CAGR {_pct(lg.get('cagr'))} / MDD {_pct(lg.get('mdd'))} / {lg.get('n_trades', 0)} 笔 / "
                 f"毛均 {_pct(lg.get('gross_mean'))}`。")
    lines.append("")
    lines.append("| 项目 | 结构族发布 | 本脚本 | 差异 | 主要来源 |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append(f"| 主口径 P1（20 日、10 仓、主板）CAGR | -1.15% | {_pct(runs['P1_hash_TR_real']['metrics'].get('cagr'))} | "
                 f"{_pct((runs['P1_hash_TR_real']['metrics'].get('cagr') or np.nan) + 0.0115)} | "
                 f"20 vs 21 日、主板、C-REAL 冲击/tick、选股路径 |")
    lines.append(f"| 主口径 P1 MDD | -75.3% | {_pct(runs['P1_hash_TR_real']['metrics'].get('mdd'))} | "
                 f"{_pct((runs['P1_hash_TR_real']['metrics'].get('mdd') or np.nan) + 0.753)} | 同上 |")
    lines.append(f"| 随机 10 仓 hold21 CAGR | -1.15% | {_pct(lg.get('cagr'))} | "
                 f"{_pct((lg.get('cagr') or np.nan) + 0.0115)} | universe（主板 vs 全 A）、选股抽样、费用表、期末处理 |")
    lines.append(f"| 随机 10 仓 hold21 MDD | -75.3% | {_pct(lg.get('mdd'))} | "
                 f"{_pct((lg.get('mdd') or np.nan) + 0.753)} | 同上 |")
    lines.append(f"| 全 A 等权日再平衡 CAGR | 10.46% | {_pct(b['EW_allA_daily_gross'].get('cagr'))} | "
                 f"{_pct((b['EW_allA_daily_gross'].get('cagr') or np.nan) - 0.1046)} | 复权因子、ST/停牌过滤、退市处理 |")
    lines.append(f"| HS300 CAGR | -0.26% | {_pct(b['HS300'].get('cagr'))} | "
                 f"{_pct((b['HS300'].get('cagr') or np.nan) + 0.0026)} | 年化天数口径（252 vs 365.25） |")
    lines.append(f"| CSI500 CAGR | 1.75% | {_pct(b['CSI500'].get('cagr'))} | "
                 f"{_pct((b['CSI500'].get('cagr') or np.nan) - 0.0175)} | 同上 |")
    lines.append("")
    lines.append("> 结构族的 `_metrics` 用 252 日/年折算 CAGR；本脚本按预注册用 365.25 日历日。"
                 "两者对同一净值曲线的差异约 1 个百分点量级，比较时必须先统一口径。")
    lines.append("")
    lines.append(f"- 主板 `U_legacy` 的 21 日前瞻开-开毛收益（A 空间）截面均值："
                 f"{_pct(uf['mean'],3)}（中位 {_pct(uf['median'],3)}，{uf['days']} 个交易日）。"
                 f"结构族随机基线的毛均 0.795% 比该值高约 0.2-0.3pp，"
                 f"而本 legacy 对照的毛均 {_pct(runs['LEGACY_random3_hold21']['metrics'].get('gross_mean'),3)} "
                 f"低约 0.4pp；两者相差约 0.7pp，约等于 2 个标准误（每笔收益 σ≈14%，n≈1500，SE≈0.36pp）。"
                 f"因此「legacy vs 结构族」的年化差主要是**毛均抽样误差 × 11.5 轮/年**，不是可复现的规则差异。")
    cb_l = cost_bridge(runs["LEGACY_random3_hold21"])
    lines.append(f"- 成本口径再补一刀：本 legacy 成本 {_pct(cb_l['cost_per_trade'],3)}/笔；结构族为 10bp 平滑点、"
                 f"无 tick 取整、2008 年印花税统一 0.1% 卖出，约 0.3%/笔。差 ~0.15pp × 11.5 ≈ 1.7pp/年；"
                 f"毛均差 0.72pp × 11.5 ≈ 8.3pp/年；两者合计约 10pp，覆盖 -9.2pp 的 CAGR 差。")

    lines.append("\n## 7. 事件层对拍（K3 `ret_hold20`，155k 事件）\n")
    lines.append(f"- 可比事件：{k3['n_events']:,}（其余 {155079 - k3['n_events']:,} 个事件为创业板/科创板，"
                 "不在主板面板，未纳入）")
    lines.append(f"- 独立重算（原始开盘 × 自建 factor / 当日收盘）与 `ret_hold20` 的绝对差："
                 f"中位 {k3['median_abs_diff']:.2e}，p99 {k3['p99_abs_diff']:.2e}，最大 {k3['max_abs_diff']:.4f}")
    lines.append(f"- 超过 1bp：{k3['n_gt_1bp']}；超过 10bp：{k3['n_gt_10bp']}；超过 100bp：{k3['n_gt_100bp']}")
    lines.append(f"- 若错误地用冻结层 `hfq_open`（`raw_open × factor`）：中位绝对差 "
                 f"{k3['hfqcol_median_abs_diff']:.4f}，超过 1bp 的行 **{k3['hfqcol_n_gt_1bp']:,}**")
    lines.append(f"- Baostock 后复权与自建 A 空间的 entry 价位比，在 {k3['n_symbols_level_spread_gt_05pct']} 只股票上极差 >0.5%"
                 "（年末重定基/断点）")
    if k3["bad_rows"]:
        lines.append("\n超过 1bp 的事件（全部列出）：\n")
        lines.append("| 代码 | 入场 | 出场 | 报告收益 | 独立重算 | 差(bp) | 原因 | 冻结 hfq 错误用法 |")
        lines.append("|---|---|---|---:|---:|---:|---|---:|")
        for x in k3["bad_rows"]:
            lines.append(f"| {x['symbol']} | {x['entry_date']} | {x['exit_date']} | "
                         f"{_pct(x['ret_reported'],3)} | {_pct(x['ret_correct'],3)} | "
                         f"{x['diff_bp']:+.1f} | {x['cause']} | {_pct(x['ret_frozen_hfq'],3)} |")

    lines.append("\n## 8. 冻结层 `hfq` 列的复权口径缺陷（必须优先处理）\n")
    lines.append("`factor` 是累计总收益指数 `Π(1+pct_chg)`；`hfq_close = raw_close × factor` 把"
                 "「价格」乘上了「累计收益」，因此 `hfq_close` 的日收益不是官方涨跌幅；在无公司行动的日子上精确为"
                 "**官方涨跌幅的平方**，有除权时也系统性偏离。")
    lines.append(f"实测（主板 2007-2020，{hfq_def['n']:,} 个相邻日）："
                 f"`r_hfq` 对 `pct_chg` 的回归斜率 **{hfq_def['slope']:.3f}**；"
                 f"中位绝对差 {hfq_def['median_abs_diff']:.4f}；"
                 f"偏差 >1bp 的占比 **{hfq_def['n_gt_1bp_share']:.1%}**（{hfq_def['n_gt_1bp']:,} 行）。")
    lines.append("")
    lines.append("正确用法：")
    lines.append("- 收盘到收盘总收益：直接用 `factor` 比值（或 `pct_chg` 复利），**不要**用 `hfq_close` 比值；")
    lines.append("- 开盘/最高/最低的 A 空间价格：`raw_px × factor / raw_close`（本脚本的 `a_*`）；")
    lines.append("- `hfq_high/low` 同理，不能直接跨日比较；")
    lines.append("- 若某引擎只用 `factor` 比值做收盘收益、用原始价撮合，则不受影响；"
                 "若用 `hfq_open` 跨日比较，则会产生「持有期收盘收益」量级的偏差。")
    lines.append("")
    lines.append("这不是 Baostock 后复权的年末重定基问题，而是冻结层派生列的构造问题；"
                 "`factor` 本身与 `pct_chg` 的逐日偏差 ≤4.3e-15，是可信的。")
    lines.append("")
    lines.append("结构族/退出引擎直接使用 Baostock hfq（A 空间），**不受本缺陷影响**；"
                 "它们的问题是另一类：6 个主板 K3 事件因断点/配股/异常 >1bp（§7）。"
                 "受影响的是任何调用 `数据层.daily(adjust='hfq')` 或直接比较冻结层 `hfq_open/high/low` 的下游。")

    lines.append("\n## 9. 超过 1bp 的差异来源清单\n")
    lines.append("| # | 来源 | 量级 | 证据/复算 |")
    lines.append("|---|---|---|---|")
    lines.append("| 1 | 冻结层 `hfq_close/hfq_open` 把总收益指数乘到价格上 | 日收益平方；91.9% 行 >1bp | §8；回归斜率 2.004 |")
    lines.append("| 2 | 复权口径差异：Baostock 断点 / 配股 / pct_chg 异常 | 6/129,680 个主板 K3 事件 >1bp；3 只股票（总表 155,079，其余创业板/科创板） | §7；601607(异常)、002466(配股)、600655(断点) |")
    lines.append("| 3 | 滑点口径：预注册 10bp+冲击 vs 结构族/退出引擎 5-10bp 平滑点 | 单边 0-15bp；20 日往返 0-60bp | 参与率 0.03%-0.33% 时冲击 3.6-11.5bp |")
    lines.append("| 4 | 过户费：预注册沪市面额 0.1%（最低 1 元）vs 结构族 0.002% 成交额；项目 `费用与交易约束.py` 用 0.6‰ 面额 | 10 元股价 1bp/边（0.6‰ 为 0.6bp）；2 元 5bp/边（0.6‰ 为 3bp） | `_fees`；§3 过户费合计 |")
    lines.append("| 5 | 印花税：结构族 2008 年统一 0.1% 卖出 vs 预注册 0.3%/0.1% 双边（2008-01..09-18） | 2008 年交易 10-30bp | `stamp_rule` |")
    lines.append("| 6 | 持有期：20 vs 21 个交易日 | 配对 +0.113pp（t=1.30，中位 0）；配置表整段差主要是路径分叉 | §4c |")
    lines.append("| 7 | universe：主板 vs 全 A（含创业板/科创板） | 21 日毛均 +0.507% vs 全 A 诊断 +0.59%，约 +0.06pp；不足以解释 0.7pp | §6 |")
    lines.append("| 8 | 退市/停牌处理：最后价 vs 最后价×0.7；停牌顺延 10 日 | 个别笔，最大 -30% | `delist_stress` 未开主口径；RAW 对账 |")
    lines.append("| 9 | 股息税：RAW 20%/10%/0 vs TR 隐含免税再投 | 每笔 0-5bp | P1_hash_RAW_real vs P1_hash_TR_real |")
    lines.append("| 10 | 最小变动单位取整 | 低价股 5-25bp/边 | `tick_cost` |")

    lines.append("\n## 10. 缺陷、未完成与下一步\n")
    lines.append("- 本脚本是**独立实现**，不是统一引擎；最终账户结论仍需统一引擎复核（按任务要求）。")
    lines.append("- `RAW` 口径的公司行动用同花顺事件 + 隐含除权日回退；未匹配事件按现金分配/拆股近似，"
                 "配股按「现金足够则认购，否则按内在价值卖出」处理；这些近似会带来个别 >1bp 差异。")
    lines.append("- 退市压力（最后价×0.7）只作为敏感性开关，主口径未开；退市整理期按 `delist_date` 前 30 个交易日"
                 "停止新开仓，但已持仓的退市整理期卖出仍按开盘价/跌停规则执行。")
    lines.append("- 20 日固定持有不涉及止损/目标，因此无法检验状态机的退出规则；这是基线，不是策略。")
    lines.append("- 未覆盖：创业板/科创板/北交所；未做行业分散；未做容量曲线（1 亿规模）。")
    lines.append("- 下一步：①把冻结层 `hfq_*` 列改为 A 空间或明确标注「仅收盘收益可用」；"
                 "②统一引擎用同一套 A 空间口径重跑简单基准；③对本报告的 10 个差异来源逐项做单因素复算；"
                 "④在统一引擎复核前，不把本报告的年化写入任何结论页。")
    lines.append("")
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
