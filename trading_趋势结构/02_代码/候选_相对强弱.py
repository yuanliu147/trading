#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选_相对强弱.py — RS-T 族：横截面相对强弱 × 周趋势许可 × 日回调/再启动

研究纪律
--------
* 预注册：03_结果/候选_相对强弱/00_预注册_相对强弱.md（SHA256 见 00b_锚点.txt）
* 面板硬截断 2020-12-31；任何 2021+ 计算在 assert_window() 处抛 HoldoutViolation。
* 信号 t 收盘生成、t+1 开盘成交；T+1、整股、历史费用、滑点、停牌/涨跌停、退市均按项目口径。
* 本文件只写 02_代码/候选_相对强弱.py 与 03_结果/候选_相对强弱/ 下的产物。

用法
----
    python3 02_代码/候选_相对强弱.py audit     # 只做数据与因果审计，不产生收益
    python3 02_代码/候选_相对强弱.py ic        # 横截面 IC
    python3 02_代码/候选_相对强弱.py events    # 事件研究
    python3 02_代码/候选_相对强弱.py sim       # 40 个配置的组合模拟
    python3 02_代码/候选_相对强弱.py all       # 全部 + 报告
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
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from 退出_数据层 import Panel                      # noqa: E402
import 费用与交易约束 as FT                        # noqa: E402
import 统计验证 as SV                              # noqa: E402

OUT = ROOT / "03_结果" / "候选_相对强弱"
OUT.mkdir(parents=True, exist_ok=True)

FREEZE = "2026-10-26"
PANEL_END = "2020-12-31"
DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")
WARM = ("2006-01-04", "2007-12-31")
FULL = ("2008-01-01", "2020-12-31")
SEED = 20261028

LOG_PATH = OUT / "06_运行日志.txt"
_LOG = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _LOG.append(line)
    LOG_PATH.write_text("\n".join(_LOG) + "\n", encoding="utf-8")


class HoldoutViolation(RuntimeError):
    pass


def assert_window(start: str, end: str) -> None:
    if pd.Timestamp(end) > pd.Timestamp(PANEL_END):
        raise HoldoutViolation(
            f"窗口 {start}~{end} 越过 {PANEL_END}；2021+ 留出禁止计算（预注册 §2.3）")


# ==========================================================================
# 配置
# ==========================================================================

@dataclass(frozen=True)
class CostModel:
    name: str = "C-REAL-V1"
    mult: float = 1.0
    base_bps: float = 10.0
    impact: float = 0.02
    cap_bps: float = 100.0
    part_cap: float = 0.02
    commission: float = 0.00025
    stamp_mult: float = 1.0


COSTS = {
    "C-REAL-V1": CostModel(),
    "C-STRESS-2X": CostModel(name="C-STRESS-2X", mult=2.0, base_bps=20.0,
                             impact=0.04, cap_bps=200.0, commission=0.0005,
                             stamp_mult=2.0),
}


def fee_config(cost: CostModel) -> FT.FeeConfig:
    if cost.stamp_mult == 1.0:
        return FT.FeeConfig(commission_rate=cost.commission)
    sched = tuple(FT.StampRule(r.effective, r.rate * cost.stamp_mult, r.sides)
                  for r in FT.DEFAULT_STAMP_SCHEDULE)
    return FT.FeeConfig(commission_rate=cost.commission, stamp_schedule=sched)


FEES = {k: fee_config(v) for k, v in COSTS.items()}


@dataclass(frozen=True)
class Cfg:
    name: str
    family: str          # ROT / EVT / RAND_ROT / RAND_EVT
    score: str = "A"
    n: int = 10
    perm: int = 1        # 1=W1, 2=W2, 0=off
    rebal: int = 10
    hold: int = 20
    thr: float = 0.60
    stop_atr: float = 2.0
    delay: int = 1
    universe: str = "main"
    bucket: int | None = None
    cost: str = "C-REAL-V1"
    capital: float = 1_000_000.0
    lookbacks: tuple = (20, 60, 120)
    ma_weeks: int = 20
    haircut: float = 1.0
    desc: str = ""


def build_configs() -> list[Cfg]:
    C: list[Cfg] = []
    A = dict(score="A")
    # ---- 核心 12
    C.append(Cfg("C01", "ROT", n=10, perm=1, rebal=10, desc="P-ROT 主候选"))
    C.append(Cfg("C02", "ROT", n=10, perm=1, rebal=5, desc="周频调仓"))
    C.append(Cfg("C03", "ROT", n=10, perm=1, rebal=20, desc="月频调仓"))
    C.append(Cfg("C04", "ROT", n=10, perm=0, rebal=10, desc="去周许可"))
    C.append(Cfg("C05", "ROT", score="B", n=10, perm=1, rebal=10, desc="波动调整动量"))
    C.append(Cfg("C06", "ROT", score="C", n=10, perm=1, rebal=10, desc="单60日"))
    C.append(Cfg("C07", "EVT", n=10, perm=1, hold=20, desc="P-EVT 主候选"))
    C.append(Cfg("C08", "EVT", score="B", n=10, perm=1, hold=20, desc="波动调整动量"))
    C.append(Cfg("C09", "EVT", score="C", n=10, perm=1, hold=20, desc="单60日"))
    C.append(Cfg("C10", "EVT", n=10, perm=0, hold=20, desc="去周许可"))
    C.append(Cfg("C11", "EVT", n=10, perm=1, hold=20, thr=0.0, desc="去RS过滤"))
    C.append(Cfg("C12", "EVT", n=10, perm=1, hold=20, delay=2, desc="排名延迟1日"))
    # ---- 敏感性 14
    C.append(Cfg("S01", "ROT", n=5, perm=1, rebal=10, desc="5仓"))
    C.append(Cfg("S02", "ROT", n=20, perm=1, rebal=10, desc="20仓"))
    C.append(Cfg("S03", "EVT", n=5, perm=1, hold=20, desc="5仓"))
    C.append(Cfg("S04", "EVT", n=20, perm=1, hold=20, desc="20仓"))
    C.append(Cfg("S05", "EVT", n=10, perm=1, hold=10, desc="持有10日"))
    C.append(Cfg("S06", "EVT", n=10, perm=1, hold=40, desc="持有40日"))
    C.append(Cfg("S07", "EVT", n=10, perm=1, hold=20, stop_atr=1.5, desc="止损1.5ATR"))
    C.append(Cfg("S08", "EVT", n=10, perm=1, hold=20, stop_atr=3.0, desc="止损3.0ATR"))
    C.append(Cfg("S09", "EVT", n=10, perm=1, hold=20, thr=0.50, desc="门槛50%"))
    C.append(Cfg("S10", "EVT", n=10, perm=1, hold=20, thr=0.70, desc="门槛70%"))
    C.append(Cfg("S11", "ROT", n=10, perm=1, rebal=10, lookbacks=(10, 40, 80), desc="短窗口"))
    C.append(Cfg("S12", "ROT", n=10, perm=1, rebal=10, lookbacks=(30, 90, 180), desc="长窗口"))
    C.append(Cfg("S13", "ROT", n=10, perm=2, rebal=10, ma_weeks=10, desc="周MA10许可"))
    C.append(Cfg("S14", "ROT", n=10, perm=1, rebal=10, delay=2, desc="排名延迟1日"))
    # ---- 分层 5
    C.append(Cfg("L01", "ROT", n=10, perm=1, rebal=10, bucket=0, desc="ADV低1/3"))
    C.append(Cfg("L02", "ROT", n=10, perm=1, rebal=10, bucket=1, desc="ADV中1/3"))
    C.append(Cfg("L03", "ROT", n=10, perm=1, rebal=10, bucket=2, desc="ADV高1/3"))
    C.append(Cfg("B01", "ROT", n=10, perm=1, rebal=10, universe="cyb", desc="创业板扩展"))
    C.append(Cfg("B02", "EVT", n=10, perm=1, hold=20, universe="cyb", desc="创业板扩展"))
    # ---- 成本/容量 5
    C.append(Cfg("K01", "ROT", n=10, perm=1, rebal=10, cost="C-STRESS-2X", desc="成本2x"))
    C.append(Cfg("K02", "EVT", n=10, perm=1, hold=20, cost="C-STRESS-2X", desc="成本2x"))
    C.append(Cfg("K03", "ROT", n=10, perm=1, rebal=10, capital=200_000, desc="20万"))
    C.append(Cfg("K04", "ROT", n=10, perm=1, rebal=10, capital=10_000_000, desc="1000万"))
    C.append(Cfg("K05", "EVT", n=10, perm=1, hold=20, capital=10_000_000, desc="1000万"))
    # ---- 对照 2
    C.append(Cfg("N01", "RAND_ROT", n=10, perm=1, rebal=10, desc="随机选股轮动"))
    C.append(Cfg("N02", "RAND_EVT", n=10, perm=1, hold=20, desc="随机选股事件"))
    # ---- 退市压力 2
    C.append(Cfg("D01", "ROT", n=10, perm=1, rebal=10, haircut=0.7, desc="退市回收0.7"))
    C.append(Cfg("D02", "EVT", n=10, perm=1, hold=20, haircut=0.7, desc="退市回收0.7"))
    return C


CONFIGS = {c.name: c for c in build_configs()}
CORE = [f"C{i:02d}" for i in range(1, 13)]
PRIMARY = ["C01", "C07"]


# ==========================================================================
# 矩阵工具
# ==========================================================================

def _shift(M: np.ndarray, n: int) -> np.ndarray:
    """沿时间轴（axis=0）向后平移 n 个交易日；前 n 行为 NaN。"""
    out = np.full_like(M, np.nan, dtype=np.float32)
    if n <= 0:
        out[:] = M
    elif n < M.shape[0]:
        out[n:, :] = M[:-n, :]
    return out


def _ffill(M: np.ndarray) -> np.ndarray:
    out = np.asarray(M, dtype=np.float32).copy()
    idx = np.where(np.isfinite(out), np.arange(out.shape[0], dtype=np.int32)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    rows = np.clip(idx, 0, None)
    res = np.take_along_axis(out, rows, axis=0)
    res[idx < 0] = np.nan
    return res


def _ffill_bool_at(rows_mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """把只在 rows_mask 行有值的布尔矩阵按行前向填充。"""
    n, c = values.shape
    base = np.where(rows_mask, np.arange(n, dtype=np.int32), -1)
    idx = np.broadcast_to(base[:, None], (n, c)).copy()
    np.maximum.accumulate(idx, axis=0, out=idx)
    res = np.take_along_axis(values, np.clip(idx, 0, None), axis=0)
    res[idx < 0] = False
    return res


def _chunk(M: np.ndarray, fn, chunk: int = 384) -> np.ndarray:
    out = np.empty(M.shape, dtype=np.float32)
    for j in range(0, M.shape[1], chunk):
        df = pd.DataFrame(M[:, j:j + chunk])
        out[:, j:j + chunk] = fn(df).to_numpy(dtype=np.float32)
    return out


def _roll_mean(M, n, minp):
    return _chunk(M, lambda d: d.rolling(n, min_periods=minp).mean())


def _roll_std(M, n, minp):
    return _chunk(M, lambda d: d.rolling(n, min_periods=minp).std(ddof=0))


def _roll_min(M, n, minp):
    return _chunk(M, lambda d: d.rolling(n, min_periods=minp).min())


def _roll_max(M, n, minp):
    return _chunk(M, lambda d: d.rolling(n, min_periods=minp).max())


def _cs_rank(mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(mat.shape, np.nan, dtype=np.float32)
    for i in range(0, mat.shape[0], 256):
        b = np.where(mask[i:i + 256], mat[i:i + 256], np.nan)
        r = pd.DataFrame(b).rank(axis=1, pct=True)
        out[i:i + 256] = r.to_numpy(dtype=np.float32)
    return out


_WEEK_ENDS = None


def global_week_ends() -> np.ndarray:
    """全局交易日历上的“每周最后一个交易日”（静态日历，不含价格）。

    用它而不是“面板里每周最后一行”，否则截断到周中时会把未完成的周当作完成周。
    """
    global _WEEK_ENDS
    if _WEEK_ENDS is None:
        cal = pd.read_parquet(ROOT / "01_数据" / "frozen" / "calendar.parquet")
        d = pd.to_datetime(cal.loc[cal["is_trading_day"] == 1, "date"])
        d = d[d <= pd.Timestamp(PANEL_END)]
        per = d.dt.to_period("W")
        _WEEK_ENDS = np.array([str(x.date()) for x in d.groupby(per).max()])
    return _WEEK_ENDS


def _round2(x: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(x, dtype=np.float64) * 100.0 + 0.5) / 100.0


def snap(dates: np.ndarray, start: str, end: str) -> tuple[int, int]:
    idx = pd.Index(dates)
    i0 = int(idx.searchsorted(start, side="left"))
    i1 = int(idx.searchsorted(end, side="right")) - 1
    if i0 > i1 or i0 >= len(dates) or i1 < 0:
        raise ValueError(f"窗口 {start}~{end} 与面板无交集")
    return i0, i1


# ==========================================================================
# 市场面板
# ==========================================================================

class Market:
    def __init__(self, p: Panel):
        self.p = p
        self.dates = p.dates
        self.dates_dt = pd.to_datetime(self.dates)
        self.date_obj = [d.date() for d in self.dates_dt]
        self.codes = p.code_str
        self.n_days = p.n_days
        self.n_codes = p.n_codes
        # Panel 的矩阵是 (n_codes, n_days)；本模块统一转成 (n_days, n_codes)
        self.F = p["f"].T.astype(np.float32)
        self.Craw = p["c"].T.astype(np.float32)
        self.Oraw = p["o"].T.astype(np.float32)
        self.Hraw = p["h"].T.astype(np.float32)
        self.Lraw = p["l"].T.astype(np.float32)
        self.pc = p["pc"].T.astype(np.float32)
        self.amt = p["amt"].T.astype(np.float32)
        self.tradable = np.ascontiguousarray(p.tradable.T)
        self.is_st = np.ascontiguousarray(p.is_st.T)
        self.listed = np.ascontiguousarray(p.listed.T)

        C0 = np.where(np.isfinite(p["ch"].T), p["ch"].T, np.nan).astype(np.float32)
        Cff = _ffill(C0)
        prevC0 = _shift(Cff, 1)
        prev_tr0 = np.zeros_like(self.tradable)
        prev_tr0[1:, :] = self.tradable[:-1, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            r0 = np.where(self.tradable & prev_tr0 & np.isfinite(prevC0) & np.isfinite(Cff),
                          Cff / prevC0 - 1.0, np.nan)
        # 数据伪影：|单日 hfq 收益|>25% 只可能来自未调整 preclose 的缩股/重整或新股首日；
        # 统一记 0，避免把 10 倍价格跳变当成收益（数据字典 §5.1 已披露）。
        self.artifact = np.isfinite(r0) & (np.abs(r0) > 0.25)
        r0c = np.where(self.artifact, 0.0, np.nan_to_num(r0, nan=0.0)).astype(np.float64)
        g = np.cumprod(1.0 + r0c, axis=0)
        first = np.where(np.isfinite(C0).any(axis=0),
                         np.argmax(np.isfinite(C0), axis=0), 0)
        anchor = Cff[first, np.arange(self.n_codes)]
        self.C = (g * anchor[None, :]).astype(np.float32)
        self.C[~np.isfinite(Cff)] = np.nan
        self.C = _ffill(self.C)
        with np.errstate(invalid="ignore", divide="ignore"):
            Fc = np.where(self.Craw > 0, self.C / self.Craw, np.nan)
        self.F = _ffill(Fc).astype(np.float32)
        self.Oh = (self.Oraw * self.F).astype(np.float32)
        self.Hh = (self.Hraw * self.F).astype(np.float32)
        self.Lh = (self.Lraw * self.F).astype(np.float32)

        prev_tr = np.zeros_like(self.tradable)
        prev_tr[1:, :] = self.tradable[:-1, :]
        self.resume = self.tradable & (~prev_tr)
        self.anomaly = np.zeros_like(self.tradable)
        with np.errstate(invalid="ignore", divide="ignore"):
            chg = self.Craw / self.pc - 1.0
        self.anomaly = np.isfinite(chg) & (np.abs(chg) > 0.15)

        self.ma20 = _roll_mean(self.C, 20, 20)
        prevC = _shift(self.C, 1)
        tr = np.maximum(self.Hh - self.Lh,
                        np.maximum(np.abs(self.Hh - prevC), np.abs(self.Lh - prevC)))
        self.atr20 = _roll_mean(tr, 20, 15)
        self.adv20 = _roll_mean(self.amt, 20, 15)

        # 涨跌停（规则推导；NaN 表示无参考价）
        pre = np.array([str(c)[3:6] for c in self.codes])
        self.is_main = np.isin(pre, ["600", "601", "603", "605", "000", "001", "002", "003"])
        self.is_cyb = np.isin(pre, ["300", "301"])
        self.is_star = np.isin(pre, ["688", "689"])
        self.day20 = np.asarray(self.dates_dt >= pd.Timestamp("2020-08-24"))
        pct = np.where(self.is_cyb[None, :] & self.day20[:, None], 0.20,
                       np.where(self.is_st, 0.05, 0.10)).astype(np.float32)
        pct = np.where(self.is_star[None, :], 0.20, pct).astype(np.float32)
        with np.errstate(invalid="ignore"):
            self.lu = _round2(self.pc * (1.0 + pct)).astype(np.float32)
            self.ld = _round2(self.pc * (1.0 - pct)).astype(np.float32)
        self.lu[~np.isfinite(self.pc)] = np.nan
        self.ld[~np.isfinite(self.pc)] = np.nan

        prev_tr_f = np.zeros_like(self.F, dtype=bool)
        prev_tr_f[1:, :] = self.tradable[:-1, :]
        self.ret1 = np.where(self.tradable & prev_tr_f & np.isfinite(prevC) & np.isfinite(self.C),
                             self.C / prevC - 1.0, np.nan).astype(np.float32)

        self.elig_base = (
            self.tradable
            & (~self.is_st)
            & (self.listed >= 250)
            & np.isfinite(self.Craw)
            & (self.Craw >= 2.0)
            & np.isfinite(self.adv20)
            & (self.adv20 >= 3e7)
            & (~self.resume)
            & (~self.anomaly)
        )

        # ADV 三分位（在主板宇宙内按日计算）
        self.bucket = np.full(self.elig_base.shape, -1, dtype=np.int8)
        base = self.elig_base & self.is_main[None, :]
        for di in range(self.n_days):
            cols = np.where(base[di, :])[0]
            if len(cols) < 30:
                continue
            a = self.adv20[di, cols]
            q1, q2 = np.quantile(a, [1 / 3, 2 / 3])
            b = np.where(a <= q1, 0, np.where(a <= q2, 1, 2))
            self.bucket[di, cols] = b.astype(np.int8)

        # 每只票最后一个有效行情日
        valid = np.isfinite(self.Craw)
        self.last_valid = np.where(valid.any(axis=0),
                                   self.n_days - 1 - np.argmax(valid[::-1, :], axis=0), -1)
        self._score_cache: dict = {}
        self._rank_cache: dict = {}
        self._perm_cache: dict = {}
        self._event = None

    # ---------------- 宇宙 / 分数 / 许可 ----------------
    def mask_for(self, cfg: Cfg) -> tuple[np.ndarray, str]:
        if cfg.universe == "cyb":
            m = self.elig_base & self.is_cyb[None, :]
            name = "cyb"
        else:
            m = self.elig_base & self.is_main[None, :]
            name = "main"
        if cfg.bucket is not None:
            m = m & (self.bucket == cfg.bucket)
            name = f"{name}_b{cfg.bucket}"
        return m, name

    def score(self, cfg: Cfg, mask: np.ndarray, mask_name: str) -> np.ndarray:
        key = (cfg.score, cfg.lookbacks, mask_name)
        if key in self._score_cache:
            return self._score_cache[key]
        lb = cfg.lookbacks
        moms, vas = {}, {}
        for n in lb:
            mom = (self.C / _shift(self.C, n) - 1.0).astype(np.float32)
            vol = _roll_std(self.ret1, n, max(5, int(0.6 * n)))
            va = (mom / (vol * math.sqrt(n))).astype(np.float32)
            moms[n], vas[n] = mom, va
        if cfg.score in ("A", "B"):
            src = moms if cfg.score == "A" else vas
            acc = np.zeros(self.C.shape, dtype=np.float32)
            cnt = np.zeros(self.C.shape, dtype=np.int8)
            for n in lb:
                r = _cs_rank(src[n], mask)
                acc += np.nan_to_num(r, nan=0.0)
                cnt += np.isfinite(r).astype(np.int8)
                del r
            with np.errstate(invalid="ignore", divide="ignore"):
                s = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
            del acc, cnt
        elif cfg.score == "C":
            s = _cs_rank(vas[lb[1]], mask)
        else:
            raise ValueError(cfg.score)
        s[~mask] = np.nan
        self._score_cache[key] = s
        return s

    def perm(self, weeks: int) -> np.ndarray:
        if weeks in self._perm_cache:
            return self._perm_cache[weeks]
        dmap = {str(x): i for i, x in enumerate(self.dates)}
        last = np.array([dmap[d] for d in global_week_ends() if d in dmap], dtype=int)
        we = np.zeros(self.n_days, dtype=bool)
        we[last] = True
        Wc = self.C[last, :]
        df = pd.DataFrame(Wc)
        ma = df.rolling(weeks, min_periods=max(5, int(weeks * 0.8))).mean()
        slope = ma / ma.shift(4) - 1.0
        ok = ((df > ma) & (slope > 0)).to_numpy()
        out = np.zeros((self.n_days, self.n_codes), dtype=bool)
        out[last] = ok
        res = _ffill_bool_at(we, out)
        self._perm_cache[weeks] = res
        return res

    def events(self) -> np.ndarray:
        if self._event is not None:
            return self._event
        lo10 = _roll_min(self.Lh, 10, 7)
        hi10 = _roll_max(self.Hh, 10, 7)
        hi3 = _roll_max(self.Hh, 3, 3)
        near = (lo10 <= self.ma20 * 1.02) & (lo10 >= self.ma20 * 0.95) & (hi10 >= self.ma20 * 1.05)
        ma_up = self.ma20 > _shift(self.ma20, 10)
        restart = (self.C > _shift(hi3, 1)) & (self.C > _shift(self.C, 1))
        ev = near & (self.C > self.ma20) & ma_up & restart & np.isfinite(self.ma20)
        self._event = ev
        return ev

    def perm_for(self, cfg: Cfg) -> np.ndarray | None:
        if cfg.perm == 0:
            return None
        return self.perm(cfg.ma_weeks if cfg.perm == 2 else 20)


# ==========================================================================
# 组合模拟
# ==========================================================================

@dataclass
class Pos:
    ci: int
    shares: int
    entry_di: int
    entry_raw: float
    entry_f: float
    entry_hfq: float
    cost: float          # 当前剩余仓位的成本（含买入费用）
    stop_hfq: float
    exit_due: int
    init_risk: float
    cost_ps: float       # 每股份成本
    risk_ps: float       # 每股份初始风险
    u: float             # Σ(qty_lot / f_lot)；持仓市值 = u × C


def _slip_bps(notional: float, adv: float, cost: CostModel) -> float:
    if not (adv > 0) or not (notional > 0):
        return cost.base_bps
    part = min(notional / adv, cost.part_cap)
    return min(cost.base_bps + cost.impact * math.sqrt(part) * 1e4, cost.cap_bps)


def simulate(m: Market, cfg: Cfg, window: tuple[str, str],
             score: np.ndarray, perm: np.ndarray | None, event: np.ndarray,
             rank: np.ndarray, mask: np.ndarray, rng_seed: int | None = None,
             trace: set | None = None):
    assert_window(*window)
    i0, i1 = snap(m.dates, *window)
    cost = COSTS[cfg.cost]
    fee_cfg = FEES[cfg.cost]
    lag = max(cfg.delay - 1, 0)
    N = cfg.n
    R = cfg.rebal
    H = cfg.hold
    dates = m.date_obj
    cash = cfg.capital
    nav_ref = cash
    positions: dict[int, Pos] = {}
    pending: dict[int, list] = {}
    buy_orders: list[int] = []
    trades: list[dict] = []
    navs, cashs, nposs = [], [], []
    last_close: dict[int, float] = {}
    overfill = 0
    rng = np.random.default_rng(SEED if rng_seed is None else rng_seed)
    ev_count = 0
    dbg = dict(orders=0, bought=0, skip_susp=0, skip_resume=0, skip_anom=0, skip_limitup=0,
               skip_atr=0, skip_lot=0, skip_cash=0, sell_fail=0)

    def slip(notional, adv, side):
        bps = _slip_bps(notional, adv, cost)
        return bps

    def close_pos(ci: int, qty: int, px_raw: float, reason: str, di: int):
        nonlocal cash
        pos = positions[ci]
        qty = int(min(qty, pos.shares))
        if qty <= 0:
            return
        fees = FT.trade_fees(dates[di], m.codes[ci], "sell", px_raw, qty, fee_cfg)
        frac = qty / pos.shares
        gross = frac * pos.u * float(m.C[di, ci])
        net = gross - fees.total
        cost_slice = pos.cost_ps * qty
        risk_slice = pos.risk_ps * qty
        cash += net
        trades.append(dict(
            code=m.codes[ci], entry_date=str(dates[pos.entry_di]),
            exit_date=str(dates[di]), entry_px=pos.entry_raw, exit_px=px_raw,
            shares=qty, initial_shares=qty,
            cost=cost_slice, proceeds=net, ret=net / cost_slice - 1.0,
            hold_days=di - pos.entry_di, reason=reason,
            init_risk=risk_slice,
            R_mult=(net - cost_slice) / risk_slice if risk_slice > 0 else np.nan,
        ))
        pos.u -= frac * pos.u
        pos.shares -= qty
        pos.cost -= cost_slice
        if pos.shares <= 0:
            del positions[ci]
            pending.pop(ci, None)

    for di in range(i0, i1 + 1):
        d = dates[di]
        # ---------------- 1) 开盘卖出 ----------------
        for ci in list(pending.keys()):
            pos = positions.get(ci)
            if pos is None:
                pending.pop(ci, None)
                continue
            reason, deadline, qsell = pending[ci]
            can = bool(m.tradable[di, ci])
            px = float(m.Oraw[di, ci]) if can else np.nan
            ld = float(m.ld[di, ci])
            if can and np.isfinite(px) and px > 0 and not (np.isfinite(ld) and px <= ld + 0.005):
                adv = float(m.adv20[di, ci])
                qty = pos.shares if qsell is None else min(int(qsell), pos.shares)
                bps = slip(qty * px, adv, "sell")
                fill = FT.apply_slippage(px, "sell", bps)
                close_pos(ci, qty, fill, reason, di)
                if pos.shares > 0:
                    pending.pop(ci, None)
                continue
            if di >= deadline:
                lc = last_close.get(ci, pos.entry_raw)
                bps = slip(pos.shares * lc, float(m.adv20[di, ci]), "sell")
                fill = FT.apply_slippage(lc, "sell", bps)
                close_pos(ci, pos.shares, fill, reason + "|顺延超期", di)

        # ---------------- 2) 开盘买入 ----------------
        for ci, tgt_notional in buy_orders:
            dbg["orders"] += 1
            is_topup = ci in positions
            if (not is_topup) and len(positions) >= N:
                dbg["skip_cash"] += 1
                break
            if not bool(m.tradable[di, ci]):
                dbg["skip_susp"] += 1
                continue
            if bool(m.resume[di, ci]):
                dbg["skip_resume"] += 1
                continue
            if bool(m.anomaly[di, ci]):
                dbg["skip_anom"] += 1
                continue
            o = float(m.Oraw[di, ci])
            lu = float(m.lu[di, ci])
            if not (o > 0):
                dbg["skip_susp"] += 1
                continue
            if np.isfinite(lu) and o >= lu - 0.005:
                dbg["skip_limitup"] += 1
                continue
            adv = float(m.adv20[di, ci])
            if not (adv > 0):
                continue
            f = float(m.F[di, ci])
            atr = float(m.atr20[di, ci])
            ch = float(m.C[di, ci])
            if not (np.isfinite(atr) and np.isfinite(ch) and ch > 0):
                dbg["skip_atr"] += 1
                continue
            atr_pct = atr / ch
            stop_pct = min(cfg.stop_atr * atr_pct, 0.12)
            if not (0.001 < stop_pct < 0.5):
                dbg["skip_atr"] += 1
                continue
            target = tgt_notional
            bps = slip(target, adv, "buy")
            fill = FT.apply_slippage(o, "buy", bps)
            qty = int(target / fill / 100.0) * 100
            qcap = int(cost.part_cap * adv / fill / 100.0) * 100
            qty = min(qty, qcap)
            if qty < 100:
                dbg["skip_lot"] += 1
                continue
            fees = FT.trade_fees(dates[di], m.codes[ci], "buy", fill, qty, fee_cfg)
            need = qty * fill + fees.total
            if need > cash:
                qty = int(cash * 0.998 / fill / 100.0) * 100
                if qty < 100:
                    dbg["skip_cash"] += 1
                    continue
                fees = FT.trade_fees(dates[di], m.codes[ci], "buy", fill, qty, fee_cfg)
                need = qty * fill + fees.total
                if need > cash:
                    dbg["skip_cash"] += 1
                    continue
            cash -= need
            dbg["bought"] += 1
            entry_hfq = fill * f
            stop_hfq = entry_hfq * (1.0 - stop_pct)
            if is_topup:
                po = positions[ci]
                po.shares += qty
                po.cost += need
                po.cost_ps = po.cost / po.shares
                po.u += qty / f
                po.risk_ps = po.init_risk / po.shares if po.shares else po.risk_ps
            else:
                positions[ci] = Pos(ci, qty, di, fill, f, entry_hfq, need, stop_hfq,
                                    di + H, qty * (entry_hfq - stop_hfq), need / qty,
                                    (entry_hfq - stop_hfq), qty / f)

        # ---------------- 3) 盘中止损（仅 EVT 族；ROT 为纯轮动，无止损） ----------------
        for ci, pos in (list(positions.items()) if cfg.family in ("EVT", "RAND_EVT") else []):
            if di <= pos.entry_di or ci in pending:
                continue
            if not bool(m.tradable[di, ci]):
                continue
            lo = float(m.Lh[di, ci])
            if np.isfinite(lo) and lo <= pos.stop_hfq:
                f = float(m.F[di, ci])
                stop_raw = pos.stop_hfq / f
                o = float(m.Oraw[di, ci])
                px = min(o, stop_raw) if o > 0 else stop_raw
                bps = slip(pos.shares * px, float(m.adv20[di, ci]), "sell")
                fill = FT.apply_slippage(px, "sell", bps)
                close_pos(ci, pos.shares, fill, "止损", di)

        # ---------------- 4) 收盘估值 ----------------
        for ci in list(positions.keys()):
            c = float(m.C[di, ci])
            if np.isfinite(c):
                last_close[ci] = c / float(m.F[di, ci]) if m.F[di, ci] > 0 else c
            if di > m.last_valid[ci] and m.last_valid[ci] >= 0:
                pos = positions[ci]
                lc = last_close.get(ci, pos.entry_raw)
                px = lc * cfg.haircut
                bps = slip(pos.shares * px, float(m.adv20[di, ci]), "sell")
                fill = FT.apply_slippage(max(px, 0.01), "sell", bps)
                close_pos(ci, pos.shares, fill, "退市/数据终止", di)

        val = 0.0
        for ci, pos in positions.items():
            c = float(m.C[di, ci])
            if not np.isfinite(c):
                c = pos.entry_raw * pos.entry_f
            val += pos.u * c
        nav = cash + val
        navs.append(nav)
        cashs.append(cash)
        nposs.append(len(positions))
        if len(positions) > N:
            overfill = max(overfill, len(positions))

        if trace and di in trace:
            print(f"TRACE {m.dates[di]} pos={len(positions)} cash={cash:,.0f} nav={nav:,.0f} "
                  f"pending={len(pending)} orders={len(buy_orders)}")
            for ci, po in list(positions.items())[:12]:
                v = po.u * float(m.C[di, ci])
                print(f"    {m.codes[ci]} sh={po.shares} cost={po.cost:,.0f} val={v:,.0f} "
                      f"entry_f={po.entry_f:.4f} C={float(m.C[di,ci]):.3f} raw={float(m.Craw[di,ci]):.2f}")
        nav_ref = nav
        if di >= i1:
            break

        # ---------------- 5) 生成次日订单 ----------------
        dj = di - lag
        if dj < 0:
            buy_orders = []
            continue
        perm_col = np.ones(m.n_codes, dtype=bool) if perm is None else perm[dj, :]
        sc = score[dj, :]
        adv_col = np.nan_to_num(m.adv20[dj, :], nan=0.0)
        code_idx = np.arange(m.n_codes)
        held = set(positions.keys())
        topups: dict[int, float] = {}

        if cfg.family in ("ROT", "RAND_ROT"):
            if (di - i0) % R != 0:
                buy_orders = []
                continue
            cand = np.where(mask[dj, :] & perm_col & np.isfinite(sc))[0]
            if len(cand) == 0:
                buy_orders = []
                continue
            if cfg.family == "RAND_ROT":
                pick = rng.choice(cand, size=min(N, len(cand)), replace=False)
            else:
                order = np.lexsort((code_idx[cand], -adv_col[cand], -sc[cand]))
                pick = cand[order[:N]]
            target = set(int(x) for x in pick)
            for ci in list(positions.keys()):
                if ci not in target:
                    if ci not in pending:
                        pending[ci] = ["轮动", di + 10, None]
                else:
                    pos = positions[ci]
                    cur = pos.u * float(m.C[dj, ci])
                    tgt = nav_ref / N
                    if cur > 1.30 * tgt and ci not in pending:
                        q = int((cur - tgt) / max(float(m.Craw[dj, ci]), 0.01) / 100.0) * 100
                        if q >= 100:
                            pending[ci] = ["再平衡减仓", di + 10, q]
                        if trace and di in trace:
                            print(f"    TRIM {m.codes[ci]} cur={cur:,.0f} tgt={tgt:,.0f} q={q} sh={pos.shares}")
                    elif cur < 0.70 * tgt:
                        topups[ci] = tgt - cur
            full_exits = sum(1 for ci in pending
                             if ci in positions and (pending[ci][2] is None
                                                     or pending[ci][2] >= positions[ci].shares))
            free = N - len(positions) + full_exits
            buy_orders = [(ci, nav_ref / N) for ci in pick if ci not in held][:max(free, 0)]
            buy_orders += [(ci, amt) for ci, amt in topups.items()
                           if ci not in pending and amt > 0.05 * nav_ref / N]
            if trace and di in trace:
                print(f"  DECIDE {m.dates[di]} pick={len(pick)} held={len(held)} "
                      f"pending={len(pending)} free={free} buys={len(buy_orders)} cash={cash:,.0f}")

        elif cfg.family in ("EVT", "RAND_EVT"):
            cond = event[dj, :] & mask[dj, :] & perm_col
            if cfg.thr > 0:
                cond &= np.isfinite(rank[dj, :]) & (rank[dj, :] >= cfg.thr)
            cand = np.where(cond)[0]
            ev_count += len(cand)
            if cfg.family == "RAND_EVT":
                pool = np.where(mask[dj, :] & perm_col & np.isfinite(sc))[0]
                n_pick = min(len(cand), len(pool))
                buy_orders = ([(int(ci), nav_ref / N) for ci in rng.choice(pool, size=n_pick, replace=False)]
                              if n_pick else [])
            else:
                cand = np.array([ci for ci in cand if ci not in held], dtype=int)
                if len(cand):
                    order = np.lexsort((code_idx[cand], -adv_col[cand], -sc[cand]))
                    cand = cand[order]
                buy_orders = [(int(ci), nav_ref / N)
                              for ci in cand[:max(N - len(positions) + len(pending), 0)]]
            # 时间退出
            for ci, pos in list(positions.items()):
                if ci not in pending and di == pos.exit_due - 1:
                    pending[ci] = ["时间", di + 10, None]

    nav = pd.Series(navs, index=pd.DatetimeIndex(pd.to_datetime(m.dates[i0:i1 + 1]), name="date"),
                    dtype=float)
    df = pd.DataFrame(trades)
    diag = dict(overfill=overfill, events=ev_count, **dbg,
                avg_pos=float(np.mean(nposs)) if nposs else 0.0,
                cash_frac=float(np.mean(np.asarray(cashs) / np.maximum(np.asarray(navs), 1e-9))))
    return nav, df, diag


def run_window(config_id: str, start: str, end: str,
               market: Market | None = None) -> tuple[pd.Series, pd.DataFrame, dict]:
    """对外接口：在指定窗口运行一个已注册配置。

    2021-01-01 及以后一律抛 HoldoutViolation（预注册 §2.3）。统一引擎复核时，
    只需替换 market 的构造（或传入统一 DataLayer 构建的 Market），规则与成本口径不变。
    """
    assert_window(start, end)
    cid = config_id.upper()
    if cid not in CONFIGS:
        raise KeyError(f"未注册配置 {config_id}；可选：{sorted(CONFIGS)}")
    cfg = CONFIGS[cid]
    m = market or Market(Panel())
    mask, mname = m.mask_for(cfg)
    score, rank = _scores_for(m, cfg, mask, mname)
    perm = m.perm_for(cfg)
    return simulate(m, cfg, (start, end), score, perm, m.events(), rank, mask)


# ==========================================================================
# 基准
# ==========================================================================

def benchmark_b1(m: Market, window: tuple[str, str], cost: CostModel | None = None,
                daily_rebal: bool = False) -> pd.Series:
    """主宇宙等权基准。

    daily_rebal=False（主口径）：月初等权买入，月内买入持有（权重随价格漂移），月末换仓；
    换手按名单变化 × 单边成本 ×2 计费。daily_rebal=True：每日再平衡的 EW（对照口径，
    会把横截面反转的“再平衡红利”计入基准，不能当作可交易基准）。
    """
    assert_window(*window)
    i0, i1 = snap(m.dates, *window)
    cost = cost or COSTS["C-REAL-V1"]
    mask = m.elig_base & m.is_main[None, :]
    ret = np.where(np.isfinite(m.ret1), m.ret1, 0.0).astype(np.float64)
    per = m.dates_dt[i0:i1 + 1].to_period("M")
    starts = pd.Series(np.arange(i0, i1 + 1)).groupby(per.values).min().to_numpy()
    ends = pd.Series(np.arange(i0, i1 + 1)).groupby(per.values).max().to_numpy()
    nav = 1.0
    out = np.empty(i1 - i0 + 1)
    prev: set[int] = set()
    one_way = cost.commission + cost.base_bps / 1e4 + 0.0005
    for k, ms in enumerate(starts):
        me = int(ends[k])
        names = np.where(mask[ms, :])[0]
        if len(names) == 0:
            for t in range(ms, me + 1):
                out[t - i0] = nav
            continue
        turn = 1.0 - (len(prev & set(names.tolist())) / max(len(prev | set(names.tolist())), 1))
        nav *= (1.0 - turn * 2.0 * one_way)
        prev = set(names.tolist())
        if daily_rebal:
            for t in range(ms, me + 1):
                if t > ms:
                    nav *= (1.0 + float(np.mean(ret[t, names])))
                out[t - i0] = nav
        else:
            cum = np.ones(len(names))
            for t in range(ms, me + 1):
                if t > ms:
                    cum *= (1.0 + ret[t, names])
                    nav *= float(np.mean(cum)) / float(np.mean(cum / (1.0 + ret[t, names])))
                out[t - i0] = nav
    return pd.Series(out, index=pd.DatetimeIndex(pd.to_datetime(m.dates[i0:i1 + 1]), name="date"))


# ==========================================================================
# 绩效与统计
# ==========================================================================

def perf(nav: pd.Series) -> dict:
    nav = nav.dropna()
    if len(nav) < 5:
        return {}
    r = nav.pct_change().dropna()
    days = (nav.index[-1] - nav.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    dd = float((nav / nav.cummax() - 1).min())
    vol = float(r.std() * math.sqrt(244))
    return dict(
        cagr=float(cagr), mdd=dd,
        calmar=float(cagr / abs(dd)) if dd < 0 else np.nan,
        sharpe=float(r.mean() / r.std() * math.sqrt(244)) if r.std() > 0 else np.nan,
        vol=vol, total=float(nav.iloc[-1] / nav.iloc[0] - 1),
        ulcer=float(np.sqrt(np.mean((nav / nav.cummax() - 1) ** 2))),
        years=years,
    )


def yearly(nav: pd.Series) -> pd.DataFrame:
    r = nav.pct_change()
    g = r.groupby(r.index.year)
    out = pd.DataFrame({
        "ret": g.apply(lambda x: (1 + x).prod() - 1),
        "mdd": g.apply(lambda x: float(((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min())),
        "days": g.count(),
    })
    out.index.name = "year"
    return out.reset_index()


def bootstrap_cagr(nav: pd.Series, n_boot: int = 2000, block: float = 20.0, seed: int = SEED) -> dict:
    r = nav.pct_change().dropna().to_numpy(float)
    n = len(r)
    if n < 60:
        return {}
    rng = np.random.default_rng(seed)
    p = 1.0 / block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            L = rng.geometric(p)
            for k in range(L):
                idx[i] = (start + k) % n
                i += 1
                if i >= n:
                    break
        path = np.cumprod(1.0 + r[idx])
        out[b] = path[-1] ** (244.0 / n) - 1.0
    return dict(p05=float(np.quantile(out, 0.05)), p50=float(np.quantile(out, 0.50)),
                p95=float(np.quantile(out, 0.95)), n=n)


def trade_stats(tr: pd.DataFrame) -> dict:
    if tr is None or len(tr) == 0:
        return dict(n_trades=0)
    ret = tr["ret"].to_numpy(float)
    win = ret > 0
    gp = ret[win].sum()
    gl = -ret[~win].sum()
    profits = tr["proceeds"] - tr["cost"]
    top5 = float(profits.nlargest(5).sum() / profits[profits > 0].sum()) if (profits > 0).any() else np.nan
    return dict(
        n_trades=int(len(tr)), win=float(win.mean()),
        avg_ret=float(np.mean(ret)), med_ret=float(np.median(ret)),
        avg_hold=float(tr["hold_days"].mean()), med_hold=float(tr["hold_days"].median()),
        profit_factor=float(gp / gl) if gl > 0 else np.nan,
        avg_R=float(tr["R_mult"].mean()) if "R_mult" in tr else np.nan,
        top5_profit_share=top5,
        n_codes=int(tr["code"].nunique()),
    )


def turnover_of(nav: pd.Series, tr: pd.DataFrame) -> float:
    if len(tr) == 0:
        return 0.0
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    traded = float((tr["cost"] + tr["proceeds"]).sum())
    return traded / float(nav.mean()) / max(years, 1e-9)


# ==========================================================================
# IC 与事件研究
# ==========================================================================

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    sa, sb = ra.std(), rb.std()
    if sa <= 0 or sb <= 0:
        return np.nan
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def forward_returns(m: Market, h: int) -> np.ndarray:
    """t 日信号 → t+1 开盘买入 → t+h 收盘卖出；不可买处为 NaN。"""
    n = m.n_days
    out = np.full((n, m.n_codes), np.nan, dtype=np.float32)
    if h + 1 >= n:
        return out
    buyable = m.tradable.copy()
    lu = m.lu
    with np.errstate(invalid="ignore"):
        blocked = np.isfinite(lu) & (m.Oraw >= lu - 0.005)
    buyable &= ~blocked
    buyable &= ~m.resume
    o = m.Oh
    c = m.C
    # t 从 0..n-h-2
    out[: n - h - 1] = c[1 + h: n, :] / o[1: n - h, :] - 1.0
    # 信号 t 的买入发生在 t+1 开盘：可执行性必须用 t+1 的状态
    ok = np.zeros_like(buyable)
    ok[: n - 1, :] = buyable[1:, :]
    out[~ok] = np.nan
    return out


def ic_table(m: Market, score: np.ndarray, mask: np.ndarray, window, hs=(5, 10, 20, 60),
             label: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    i0, i1 = snap(m.dates, *window)
    rows = []
    for h in hs:
        fwd = forward_returns(m, h)
        vals, dates_ = [], []
        for di in range(i0, i1 + 1):
            ok = mask[di, :] & np.isfinite(fwd[di, :]) & np.isfinite(score[di, :])
            if ok.sum() < 30:
                continue
            ic = _spearman(score[di, ok], fwd[di, ok])
            if np.isfinite(ic):
                vals.append(ic)
                dates_.append(m.dates[di])
        s = pd.Series(vals, index=pd.to_datetime(dates_))
        monthly = s.groupby([s.index.year, s.index.month]).mean()
        if len(monthly) >= 3:
            t = float(monthly.mean() / (monthly.std(ddof=1) / math.sqrt(len(monthly))))
        else:
            t = np.nan
        rows.append(dict(label=label, h=h, n_days=len(s), ic_mean=float(s.mean()),
                         ic_std=float(s.std()), t_month=t,
                         hit=float((s > 0).mean()) if len(s) else np.nan,
                         ic_p05=float(s.quantile(0.05)), ic_p95=float(s.quantile(0.95))))
    df = pd.DataFrame(rows)
    # 年度
    yr = []
    for h in hs:
        fwd = forward_returns(m, h)
        recs = []
        for di in range(i0, i1 + 1):
            ok = mask[di, :] & np.isfinite(fwd[di, :]) & np.isfinite(score[di, :])
            if ok.sum() < 30:
                continue
            recs.append((m.dates[di], _spearman(score[di, ok], fwd[di, ok])))
        s = pd.DataFrame(recs, columns=["date", "ic"]).dropna()
        if len(s) == 0:
            continue
        s["year"] = pd.to_datetime(s["date"]).dt.year
        g = s.groupby("year")["ic"].agg(["mean", "count"]).reset_index()
        g["h"] = h
        g["label"] = label
        yr.append(g)
    return df, pd.concat(yr, ignore_index=True) if yr else pd.DataFrame()


def event_study(m: Market, event: np.ndarray, perm: np.ndarray | None,
                rank: np.ndarray, thr: float, mask: np.ndarray, window,
                hs=(5, 10, 20, 40, 60), label: str = "EVT") -> tuple[pd.DataFrame, pd.DataFrame]:
    i0, i1 = snap(m.dates, *window)
    ev = event & mask
    if perm is not None:
        ev = ev & perm
    if thr > 0:
        ev = ev & np.isfinite(rank) & (rank >= thr)
    idx = np.argwhere(ev[i0:i1 + 1, :])
    recs = []
    fwd_cache = {h: forward_returns(m, h) for h in hs}
    bucket = m.bucket
    # 基准：同日同宇宙、同 ADV 分位
    bm_all = {h: np.full(m.n_days, np.nan) for h in hs}
    bm_q = {h: np.full((m.n_days, 3), np.nan) for h in hs}
    for h in hs:
        f = fwd_cache[h]
        for di in range(i0, i1 + 1):
            ok = mask[di, :] & np.isfinite(f[di, :])
            if ok.sum() == 0:
                continue
            bm_all[h][di] = float(np.mean(f[di, ok]))
            for b in range(3):
                o2 = ok & (bucket[di, :] == b)
                if o2.sum() >= 5:
                    bm_q[h][di, b] = float(np.mean(f[di, o2]))
    for drow, ci in idx:
        di = drow + i0
        row = dict(code=m.codes[ci], date=str(m.dates[di]), bucket=int(bucket[di, ci]))
        ok_any = True
        for h in hs:
            r = fwd_cache[h][di, ci]
            row[f"ret{h}"] = float(r) if np.isfinite(r) else np.nan
            row[f"exc{h}"] = (float(r) - bm_all[h][di]) if np.isfinite(r) else np.nan
            b = bucket[di, ci]
            q = bm_q[h][di, b] if b >= 0 else np.nan
            row[f"exc_q{h}"] = (float(r) - q) if np.isfinite(r) and np.isfinite(q) else np.nan
            if not np.isfinite(r):
                ok_any = False
        row["executable"] = ok_any
        recs.append(row)
    df = pd.DataFrame(recs)
    summ = []
    if len(df):
        for h in hs:
            for col in (f"exc{h}", f"exc_q{h}", f"ret{h}"):
                x = df.loc[df["executable"], col].dropna()
                if len(x) < 5:
                    continue
                months = pd.to_datetime(df.loc[x.index, "date"]).dt.to_period("M").astype(str)
                ct = SV.cluster_mean_test(x.to_numpy(), months.to_numpy())
                summ.append(dict(label=label, h=h, kind=col, n=len(x),
                                 mean=float(x.mean()), median=float(x.median()),
                                 winsor=float(x.clip(x.quantile(0.01), x.quantile(0.99)).mean()),
                                 hit=float((x > 0).mean()), t_month=ct.get("t"),
                                 p_month=ct.get("p")))
    return df, pd.DataFrame(summ)


# ==========================================================================
# 运行
# ==========================================================================

def _scores_for(m: Market, cfg: Cfg, mask: np.ndarray, name: str):
    key = (cfg.score, cfg.lookbacks, name)
    s = m.score(cfg, mask, name)
    if key not in m._rank_cache:
        m._rank_cache[key] = _cs_rank(s, mask)
    return s, m._rank_cache[key]


def run_config(m: Market, cid: str, windows=(("dev", DEV), ("val", VAL))):
    cfg = CONFIGS[cid]
    mask, mname = m.mask_for(cfg)
    score, rank = _scores_for(m, cfg, mask, mname)
    perm = m.perm_for(cfg)
    ev = m.events()
    out = {}
    for wname, win in windows:
        t0 = time.time()
        if cfg.family in ("RAND_ROT", "RAND_EVT"):
            navs, trs, diags, seed_rows = [], [], [], []
            for k in range(5):
                nv, tt, dg = simulate(m, cfg, win, score, perm, ev, rank, mask,
                                      rng_seed=SEED + 1000 * k + (0 if wname == "dev" else 7))
                navs.append(nv)
                trs.append(tt)
                diags.append(dg)
                seed_rows.append(dict(config=cid, window=wname, seed=k,
                                      cagr=perf(nv).get("cagr"), n_trades=len(tt)))
            nav = pd.concat(navs, axis=1).mean(axis=1)
            tr = pd.concat(trs, ignore_index=True)
            diag = {k: float(np.mean([d[k] for d in diags])) for k in diags[0]}
            seed_path = OUT / "03_随机对照_种子.csv"
            head = not seed_path.exists()
            pd.DataFrame(seed_rows).to_csv(seed_path, mode="a", header=head, index=False,
                                           encoding="utf-8-sig")
        else:
            nav, tr, diag = simulate(m, cfg, win, score, perm, ev, rank, mask,
                                     rng_seed=SEED + (0 if wname == "dev" else 1))
        p = perf(nav)
        ts = trade_stats(tr)
        n_seeds = 5 if cfg.family in ("RAND_ROT", "RAND_EVT") else 1
        if n_seeds > 1 and "n_trades" in ts:
            ts["n_trades"] = int(round(ts["n_trades"] / n_seeds))
        p.update(ts)
        p.update(diag)
        p["turnover"] = turnover_of(nav, tr) / n_seeds
        p["config"] = cid
        p["window"] = wname
        p["family"] = cfg.family
        p["desc"] = cfg.desc
        out[wname] = dict(nav=nav, trades=tr, perf=p)
        log(f"  {cid}/{wname}: CAGR {p.get('cagr', np.nan):.2%} MDD {p.get('mdd', np.nan):.2%} "
            f"n={p.get('n_trades', 0)} t={time.time()-t0:.1f}s")
    return out


def run_all():
    t_start = time.time()
    for f in ("03_随机对照_种子.csv",):
        fp = OUT / f
        if fp.exists():
            fp.unlink()
    log("载入面板 …")
    p = Panel()
    m = Market(p)
    log(f"面板 {m.n_codes} 只 × {m.n_days} 日；主板宇宙日均 "
        f"{float((m.elig_base & m.is_main[None,:]).sum(axis=0).mean()):.0f} 只")

    # ---------- 审计 ----------
    audit = {
        "panel_dates": [str(m.dates[0]), str(m.dates[-1]), int(m.n_days)],
        "codes": int(m.n_codes),
        "main_codes": int(m.is_main.sum()), "cyb_codes": int(m.is_cyb.sum()),
        "eligible_main_2008": int((m.elig_base & m.is_main[None, :])[snap(m.dates, *DEV)[0], :].sum()),
        "eligible_main_2020": int((m.elig_base & m.is_main[None, :])[snap(m.dates, *VAL)[1], :].sum()),
        "tradable_cells": int(m.tradable.sum()),
        "suspended_cells": int((np.isfinite(m.Craw) & ~m.tradable).sum()),
        "st_cells": int((m.is_st & np.isfinite(m.Craw)).sum()),
        "resume_cells": int(m.resume.sum()),
        "anomaly_cells": int((m.anomaly & np.isfinite(m.Craw)).sum()),
        "event_count_2008_2020": int(m.events()[:, snap(m.dates, *DEV)[0]:snap(m.dates, *VAL)[1] + 1].sum()),
    }
    (OUT / "05_自检.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"审计：{audit}")

    # ---------- IC ----------
    ic_all, ic_year = [], []
    for score_kind in ("A", "B", "C"):
        cfg0 = Cfg("IC" + score_kind, "ROT", score=score_kind)
        mask = m.elig_base & m.is_main[None, :]
        s, _ = _scores_for(m, cfg0, mask, "main")
        for wname, win in (("dev", DEV), ("val", VAL)):
            t, y = ic_table(m, s, mask, win, label=f"{score_kind}_{wname}")
            ic_all.append(t)
            ic_year.append(y)
    ic_df = pd.concat(ic_all, ignore_index=True)
    ic_df.to_csv(OUT / "01_IC_汇总.csv", index=False, encoding="utf-8-sig")
    pd.concat(ic_year, ignore_index=True).to_csv(OUT / "01_IC_年度.csv", index=False, encoding="utf-8-sig")
    log(f"IC 完成，{len(ic_df)} 行")

    # ---------- 事件研究 ----------
    cfgE = CONFIGS["C07"]
    maskE = m.elig_base & m.is_main[None, :]
    scoreA, rankA = _scores_for(m, cfgE, maskE, "main")
    perm1 = m.perm(20)
    ev = m.events()
    ev_dfs, ev_sum = [], []
    for label, perm_, thr in (("EVT_W1_thr60", perm1, 0.60),
                              ("EVT_W0_thr60", None, 0.60),
                              ("EVT_W1_thr0", perm1, 0.0)):
        d, s = event_study(m, ev, perm_, rankA, thr, maskE, FULL, label=label)
        if len(d):
            d["variant"] = label
            ev_dfs.append(d)
        s["variant"] = label
        ev_sum.append(s)
        log(f"  事件研究 {label}: {len(d)} 条")
    ev_all = pd.concat(ev_dfs, ignore_index=True)
    ev_all.to_parquet(OUT / "02_事件明细.parquet", index=False)
    es = pd.concat(ev_sum, ignore_index=True)
    es.to_csv(OUT / "02_事件研究_汇总.csv", index=False, encoding="utf-8-sig")
    # 年度
    if len(ev_all):
        ev_all["year"] = pd.to_datetime(ev_all["date"]).dt.year
        ev_all.groupby(["variant", "year"])[["exc20", "exc_q20"]].agg(["mean", "count"]).to_csv(
            OUT / "02_事件研究_年度.csv", encoding="utf-8-sig")

    # ---------- 组合 ----------
    perfs, nav_rows, nav_store = [], [], {}
    for cid in CONFIGS:
        try:
            res = run_config(m, cid)
        except Exception as e:
            log(f"  {cid} 失败: {e}\n{traceback.format_exc()}")
            continue
        for wname, r in res.items():
            perfs.append(r["perf"])
            nav_store[(cid, wname)] = r["nav"]
            nav_rows.append(pd.DataFrame({"config": cid, "window": wname,
                                          "date": r["nav"].index, "nav": r["nav"].values}))
            if cid in PRIMARY + ["N01", "N02"]:
                r["trades"].to_csv(OUT / f"03_交易_{cid}_{wname}.csv", index=False,
                                   encoding="utf-8-sig")
        log(f"完成 {cid} ({CONFIGS[cid].desc})")

    perf_df = pd.DataFrame(perfs)
    # 基准
    for wname, win in (("dev", DEV), ("val", VAL)):
        b1 = benchmark_b1(m, win)
        nav_store[("B1", wname)] = b1
        nav_rows.append(pd.DataFrame({"config": "B1", "window": wname,
                                      "date": b1.index, "nav": b1.values}))
        perf_df = pd.concat([perf_df, pd.DataFrame([{
            "config": "B1", "window": wname, "family": "BENCH", "desc": "主宇宙等权月度再平衡",
            **perf(b1), "n_trades": 0}])], ignore_index=True)
        try:
            import duckdb
            con = duckdb.connect(str(ROOT / "01_数据" / "exit_cache.duckdb"), read_only=True)
            idx = con.execute("select date, close from idx where code='sh.000300' "
                              f"and date between '{win[0]}' and '{win[1]}' order by date").df()
            con.close()
            b2 = pd.Series(idx["close"].to_numpy(), index=pd.to_datetime(idx["date"]))
            nav_store[("B2", wname)] = b2
            nav_rows.append(pd.DataFrame({"config": "B2", "window": wname,
                                          "date": b2.index, "nav": b2.values}))
            perf_df = pd.concat([perf_df, pd.DataFrame([{
                "config": "B2", "window": wname, "family": "BENCH", "desc": "沪深300",
                **perf(b2), "n_trades": 0}])], ignore_index=True)
        except Exception as e:
            log(f"B2 读取失败: {e}")
    perf_df.to_csv(OUT / "03_组合_汇总.csv", index=False, encoding="utf-8-sig")
    nav_all = pd.concat(nav_rows, ignore_index=True)
    nav_all.to_csv(OUT / "03_净值.csv.gz", index=False, compression="gzip")
    log(f"组合完成：{len(perf_df)} 行绩效")

    # ---------- 年度 ----------
    yrs = []
    for (cid, wname), nav in nav_store.items():
        y = yearly(nav)
        y["config"] = cid
        y["window"] = wname
        yrs.append(y)
    pd.concat(yrs, ignore_index=True).to_csv(OUT / "03_年度.csv", index=False, encoding="utf-8-sig")

    # ---------- 统计 ----------
    stats = {}
    try:
        R = {}
        for wname in ("dev", "val"):
            wide = nav_all[nav_all["window"] == wname].pivot(index="date", columns="config", values="nav")
            R[wname] = wide.pct_change().dropna(how="all").fillna(0.0)
        for cid in PRIMARY:
            stats[cid] = {}
            for wname in ("dev", "val"):
                rets = R[wname]
                if cid not in rets.columns:
                    continue
                r = rets[cid].dropna()
                srs = []
                for c in rets.columns:
                    x = rets[c].dropna()
                    if len(x) > 30 and x.std() > 0:
                        srs.append(float(x.mean() / x.std()))
                dsr = SV.deflated_sharpe(r.to_numpy(), trial_srs=np.array(srs),
                                         n_trials=len(CONFIGS))
                nav = nav_store[(cid, wname)]
                bs = bootstrap_cagr(nav)
                stats[cid][wname] = dict(dsr=dsr, bootstrap=bs)
        core = [c for c in CORE if c in R["dev"].columns]
        Rc = R["dev"][core].fillna(0.0).to_numpy()
        pbo = SV.pbo_cscv(Rc, blocks=8, max_combinations=5000) if Rc.shape[1] >= 2 else {}
        stats["pbo_core_dev"] = {k: v for k, v in pbo.items() if k != "detail"}
        if "detail" in pbo:
            pbo["detail"].to_csv(OUT / "04_PBO明细.csv", index=False, encoding="utf-8-sig")
        (OUT / "04_统计检验.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        log("统计完成")
    except Exception as e:
        log(f"统计失败: {e}\n{traceback.format_exc()}")

    # ---------- 试验登记 ----------
    rows = []
    for cid, cfg in CONFIGS.items():
        sub = perf_df[perf_df["config"] == cid]
        for _, r in sub.iterrows():
            rows.append(dict(trial_id=cid, window=r["window"], family=cfg.family,
                             config=json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                                for k, v in asdict(cfg).items()},
                                               ensure_ascii=False),
                             n_trades=r.get("n_trades"), cagr=r.get("cagr"), mdd=r.get("mdd"),
                             calmar=r.get("calmar"), turnover=r.get("turnover"), status="done"))
    pd.DataFrame(rows).to_csv(OUT / "03_试验登记.csv", index=False, encoding="utf-8-sig")
    log(f"全部完成，用时 {time.time()-t_start:.0f}s；配置 {len(CONFIGS)} 个")


# ==========================================================================
# 补充诊断：十分位、成本分解、持仓流动性、子区间
# ==========================================================================

def decile_table(m: Market, window, h: int = 20, q: int = 10) -> pd.DataFrame:
    i0, i1 = snap(m.dates, *window)
    mask = m.elig_base & m.is_main[None, :]
    cfg = Cfg("DEC", "ROT", score="A")
    s, _ = _scores_for(m, cfg, mask, "main")
    f = forward_returns(m, h)
    rows = []
    for di in range(i0, i1 + 1):
        ok = np.where(mask[di, :] & np.isfinite(f[di, :]) & np.isfinite(s[di, :]))[0]
        if len(ok) < q * 20:
            continue
        sc = s[di, ok]
        edges = np.quantile(sc, np.linspace(0, 1, q + 1))
        grp = np.clip(np.searchsorted(edges[1:-1], sc, side="right"), 0, q - 1)
        for g in range(q):
            x = f[di, ok[grp == g]]
            x = x[np.isfinite(x)]
            if len(x):
                rows.append(dict(date=str(m.dates[di]), decile=g + 1, ret=float(np.mean(x)), n=len(x)))
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["date"])
    out = []
    for g, sub in df.groupby("decile"):
        by_month = sub.groupby(sub["dt"].dt.to_period("M"))["ret"].mean()
        t = float(by_month.mean() / (by_month.std(ddof=1) / math.sqrt(len(by_month)))) if len(by_month) > 2 else np.nan
        out.append(dict(window=window[0][:4], h=h, decile=int(g), days=len(sub),
                        mean=float(sub["ret"].mean()), median=float(sub["ret"].median()),
                        hit=float((sub["ret"] > 0).mean()), t_month=t))
    return pd.DataFrame(out)


def cost_breakdown(m: Market, cid: str, window) -> dict:
    cfg = CONFIGS[cid]
    cost = COSTS[cfg.cost]
    mask, mname = m.mask_for(cfg)
    score, rank = _scores_for(m, cfg, mask, mname)
    perm = m.perm_for(cfg)
    nav, tr, _ = simulate(m, cfg, window, score, perm, m.events(), rank, mask)
    if len(tr) == 0:
        return dict(config=cid, window=window[0][:4], n=0)
    d_idx = {str(x): i for i, x in enumerate(m.dates)}
    fees, slip_in, slip_out, gross = 0.0, 0.0, 0.0, 0.0
    for _, t in tr.iterrows():
        a = d_idx[str(t.entry_date)]
        b = d_idx[str(t.exit_date)]
        ci = m.codes.index(t.code)
        sh = float(t["initial_shares"])
        oa = float(m.Oraw[a, ci])
        ob = float(m.Oraw[b, ci])
        fa, fb = float(m.F[a, ci]), float(m.F[b, ci])
        if not (np.isfinite(fa) and fa > 0 and np.isfinite(fb) and fb > 0):
            continue
        fr = fb / fa
        fees += float(t["cost"]) - sh * float(t["entry_px"])
        fees += sh * float(t["exit_px"]) * fr - float(t["proceeds"])
        adv_a = float(m.adv20[a, ci])
        adv_b = float(m.adv20[b, ci])
        slip_in += sh * float(t["entry_px"]) * _slip_bps(sh * float(t["entry_px"]), adv_a, cost) / 1e4
        slip_out += sh * float(t["exit_px"]) * fr * _slip_bps(sh * float(t["exit_px"]), adv_b, cost) / 1e4
        gross += sh * (float(t["exit_px"]) * fr - float(t["entry_px"]))
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    mean_nav = float(nav.mean())
    return dict(config=cid, window=window[0][:4], n=int(len(tr)),
                fees_pct_nav=fees / mean_nav / yrs,
                slip_in_pct_nav=slip_in / mean_nav / yrs,
                slip_out_pct_nav=slip_out / mean_nav / yrs,
                gross_pct_nav=gross / mean_nav / yrs,
                net_cagr=perf(nav)["cagr"])


def holdings_liquidity(m: Market, cid: str) -> pd.DataFrame:
    """持仓的 ADV 分位与价格分布（只对主候选做，验证是否只是小盘/低价）。"""
    cfg = CONFIGS[cid]
    mask, mname = m.mask_for(cfg)
    score, rank = _scores_for(m, cfg, mask, mname)
    perm = m.perm_for(cfg)
    rows = []
    for wname, win in (("dev", DEV), ("val", VAL)):
        nav, tr, _ = simulate(m, cfg, win, score, perm, m.events(), rank, mask)
        d_idx = {str(x): i for i, x in enumerate(m.dates)}
        for _, t in tr.iterrows():
            a = d_idx[str(t.entry_date)]
            ci = m.codes.index(t.code)
            base = np.where(mask[a, :])[0]
            adv = m.adv20[a, base]
            pct = float((adv < m.adv20[a, ci]).mean()) if len(adv) else np.nan
            rows.append(dict(config=cid, window=wname, code=t.code, date=t.entry_date,
                             adv_pct=pct, adv_yi=float(m.adv20[a, ci]) / 1e8,
                             price=float(m.Craw[a, ci]), ret=float(t["ret"])))
    return pd.DataFrame(rows)


def subperiod_table() -> pd.DataFrame:
    """主候选的分段年化（含剔除 2014-07..2015-12 杠杆牛）。"""
    rows = []
    nav_all = pd.read_csv(OUT / "03_净值.csv.gz")
    for cid in ("C01", "C07", "C05", "C09", "N02", "B1"):
        for wname, win in (("dev", DEV), ("val", VAL)):
            sub = nav_all[(nav_all.config == cid) & (nav_all.window == wname)]
            if sub.empty:
                continue
            nav = pd.Series(sub["nav"].to_numpy(), index=pd.to_datetime(sub["date"]))
            def cagr_of(x):
                if len(x) < 5:
                    return np.nan
                yrs = (x.index[-1] - x.index[0]).days / 365.25
                return float((x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1)
            mask_ex = ~((nav.index >= "2014-07-01") & (nav.index <= "2015-12-31"))
            spliced = nav[mask_ex]
            rows.append(dict(config=cid, window=wname, full=cagr_of(nav),
                             ex_2014_2015=cagr_of(spliced),
                             y2008_2013=cagr_of(nav["2008":"2013"]),
                             y2016_2020=cagr_of(nav["2016":"2020"])))
    return pd.DataFrame(rows)


def ic_posthoc(m: Market, window, h: int = 20) -> pd.DataFrame:
    """事后分解（不属于预注册假设，只用于解释负 IC 的来源，不产生任何规则）。"""
    i0, i1 = snap(m.dates, *window)
    mask = m.elig_base & m.is_main[None, :]
    f = forward_returns(m, h)
    c = m.C
    mats = {
        "mom20_full": c / _shift(c, 20) - 1.0,
        "mom20_lag5": _shift(c, 5) / _shift(c, 25) - 1.0,
        "ret5_last": c / _shift(c, 5) - 1.0,
        "mom60_lag5": _shift(c, 5) / _shift(c, 65) - 1.0,
    }
    out = []
    for name, mat in mats.items():
        vals, dts = [], []
        for di in range(i0, i1 + 1):
            ok = mask[di, :] & np.isfinite(f[di, :]) & np.isfinite(mat[di, :])
            if ok.sum() < 30:
                continue
            ic = _spearman(mat[di, ok], f[di, ok])
            if np.isfinite(ic):
                vals.append(ic); dts.append(m.dates[di])
        s = pd.Series(vals, index=pd.to_datetime(dts))
        monthly = s.groupby([s.index.year, s.index.month]).mean()
        t = float(monthly.mean() / (monthly.std(ddof=1) / math.sqrt(len(monthly)))) if len(monthly) > 2 else np.nan
        out.append(dict(window=window[0][:4], h=h, variant=name, n=len(s),
                        ic_mean=float(s.mean()), t_month=t, hit=float((s > 0).mean())))
    return pd.DataFrame(out)


def run_diag():
    p = Panel()
    m = Market(p)
    dec = []
    for wname, win in (("dev", DEV), ("val", VAL)):
        for h in (5, 20, 60):
            dec.append(decile_table(m, win, h=h))
    pd.concat(dec, ignore_index=True).to_csv(OUT / "01_十分位_前瞻收益.csv", index=False,
                                             encoding="utf-8-sig")
    log("十分位完成")
    cb = []
    for cid in ("C01", "C07", "C05", "C03", "N01"):
        for win in (DEV, VAL):
            cb.append(cost_breakdown(m, cid, win))
    pd.DataFrame(cb).to_csv(OUT / "03_成本分解.csv", index=False, encoding="utf-8-sig")
    log("成本分解完成")
    hl = pd.concat([holdings_liquidity(m, c) for c in ("C01", "C07")], ignore_index=True)
    hl.to_csv(OUT / "03_持仓流动性.csv", index=False, encoding="utf-8-sig")
    log("持仓流动性完成")
    subperiod_table().to_csv(OUT / "03_子区间.csv", index=False, encoding="utf-8-sig")
    log("子区间完成")
    ph = []
    for win in (DEV, VAL):
        for h in (5, 20):
            ph.append(ic_posthoc(m, win, h=h))
    pd.concat(ph, ignore_index=True).to_csv(OUT / "01_IC_事后分解.csv", index=False,
                                           encoding="utf-8-sig")
    log("事后 IC 分解完成")


# ==========================================================================
# 因果自检（截断重算）
# ==========================================================================

def causality_check(n_dates: int = 10):
    log("因果自检：截断重算 …")
    p = Panel()
    m_full = Market(p)
    score_full = m_full.score(CONFIGS["C01"], m_full.elig_base & m_full.is_main[None, :], "main")
    perm_full = m_full.perm(20)
    ev_full = m_full.events()
    i0, i1 = snap(m_full.dates, *DEV)
    rng = np.random.default_rng(SEED)
    picks = sorted(set([i0, i1, snap(m_full.dates, *VAL)[0]] +
                       list(rng.integers(i0, i1, size=n_dates))))
    res = []
    for di in picks:
        pt = p.truncate(str(m_full.dates[di]))
        mt = Market(pt)
        s = mt.score(CONFIGS["C01"], mt.elig_base & mt.is_main[None, :], "main")
        a = s[di, :]
        b = score_full[di, :]
        ok = np.isfinite(a) & np.isfinite(b)
        diff = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0
        perm_eq = bool(np.array_equal(mt.perm(20)[di, :], perm_full[di, :]))
        ev_eq = bool(np.array_equal(mt.events()[di, :], ev_full[di, :]))
        res.append(dict(date=str(m_full.dates[di]), n=int(ok.sum()), score_maxdiff=diff,
                        perm_equal=perm_eq, event_equal=ev_eq,
                        pass_=bool(diff < 1e-6 and perm_eq and ev_eq)))
    df = pd.DataFrame(res)
    df.to_csv(OUT / "05_因果自检.csv", index=False, encoding="utf-8-sig")
    log(f"因果自检：{int(df['pass_'].sum())}/{len(df)} 通过")
    if not df["pass_"].all():
        log(df.to_string())
    return df


# ==========================================================================
# CLI
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audit", "ic", "events", "sim", "all", "causal", "diag"],
                    default="all", nargs="?")
    a = ap.parse_args()
    if a.stage == "causal":
        causality_check()
        return
    if a.stage == "diag":
        run_diag()
        return
    if a.stage in ("audit",):
        p = Panel()
        m = Market(p)
        i0, i1 = snap(m.dates, *DEV)
        print(f"面板 {m.n_codes}×{m.n_days}；主板可交易宇宙 2008: "
              f"{int((m.elig_base & m.is_main[None,:])[i0, :].sum())}，2020: "
              f"{int((m.elig_base & m.is_main[None,:])[i1, :].sum())}")
        print(f"事件 2008-2020: {int(m.events()[i0:i1+1, :].sum())}")
        return
    run_all()


if __name__ == "__main__":
    main()
