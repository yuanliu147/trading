#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T0005 主候选执行员：F1 × S1 × E2 × X3 × G1。

复原对象
--------
`00_研究预注册.md` §7.6 的预注册主候选，参数取 §7.8 默认值 = 注册表 T0005
`planned` 行 `params_json`（k=2, ATR14, 缓冲0.05, 止损0.3ATR, 最大风险12%,
BE 1R, 跟踪2R/3ATR, 3日失败, 10日减半, 20日时间止损, 组合回撤10/15,
单笔1.2%, 10仓, 排序=突破强度/ADV/代码）。

- 框架层=月（S1 方向）、结构层=周（S1：分型序列 + 结构线）、触发层=日（E2）。
- `S1` 是 §7.1 的分型序列，**不是**趋势族代码里的 `S1`（月周日共振突破）。
- 规范缺口与唯一操作化见 `03_结果/主候选_T0005/00_歧义与操作化记录.md`（Q1..Q25）。

纪律
----
* 数据：v2 冻结 `01_数据/frozen/frozen.duckdb`（`v2-adjspace-2026-10-28`，全量自检 28 PASS）。
* 价格：分型/突破位/止损/MA20/ATR 用 `ca_close = raw × Π(preclose/prev_close)`；
  成交/整股/涨跌停/费用用 raw。**不用**冻结层 `factor` 当价格，不用 `bs_hfq_close`。
* 只跑 dev 2008-2015、val 2016-2020；加载 2021+ 直接抛错，输出后断言 0 行。
* 账户：`统一回测.UnifiedAccount`（公司行动/红利税/滑点/参与率/T+1/涨跌停/停牌/退市），
  另实现一套独立账本回放同一订单流做对账。
* 本文件只写 `02_代码/主候选_T0005.py` 与 `03_结果/主候选_T0005/`。

用法
----
    python3 02_代码/主候选_T0005.py signals     # 信号层 + 覆盖诊断
    python3 02_代码/主候选_T0005.py main        # 主候选 dev/val（统一账户）
    python3 02_代码/主候选_T0005.py reconcile   # 独立账户对账
    python3 02_代码/主候选_T0005.py ablation    # V1..V8 + A1..A7
    python3 02_代码/主候选_T0005.py neighbor    # §7.8 参数邻域
    python3 02_代码/主候选_T0005.py report      # 统计/门槛/报告
    python3 02_代码/主候选_T0005.py all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from 统一数据适配 import (  # noqa: E402
    FrozenPanel, EventBook, WINDOWS, DEV, VAL, FREEZE, HOLDOUT_START,
    MIN_ADV20, MIN_PRICE, MIN_LIST_CAL_DAYS, MIN_LIST_TD, DELIST_BLOCK_TD,
)
from 统一回测 import (  # noqa: E402
    UnifiedAccount, RunConfig, _make_account, load_index_series, drawdown,
    _failed_orders,
)
from 账户撮合 import Order  # noqa: E402
from 统计验证 import (  # noqa: E402
    BacktestResult, compute_metrics, annual_table, monthly_matrix, block_bootstrap,
)

OUT = ROOT / "03_结果" / "主候选_T0005"
EVID_DIR = ROOT / "04_证据"
T0 = time.time()
PANEL_START, PANEL_END = "2007-01-01", "2020-12-31"
HARD_END = date(2020, 12, 31)
SEED = 20261026


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 参数（§7.8 默认值；消融/邻域只改单一字段）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    tag: str = "T0005"
    # 族
    framework: str = "month"      # F1；V1=quarter；V2=week
    structure: str = "week"       # F1；V1=month；V2=day
    trigger: str = "day"          # F1；V1=week；V2=day
    # 结构
    structure_mode: str = "S1"    # S1=分型序列；S2=EMA 结构线（V3）
    framework_mode: str = "S1"
    entry_mode: str = "E2"        # E1/E2/E3（V4/V5）
    use_target: bool = False      # X1 固定 2R 目标（V7）
    target_r: float = 2.0
    k: int = 2
    atr_n: int = 14
    buffer_atr: float = 0.05
    band_atr: float = 0.50
    retest_days: int = 20          # Q5
    # 止损/退出
    stop_atr: float = 0.30
    max_risk_pct: float = 0.12
    be_mfe_r: float = 1.0
    trail_mfe_r: float = 2.0
    trail_atr: float = 3.0
    fail_days: int = 3
    fail_mfe_r: float = 0.5
    half_days: int = 10
    time_stop_days: int = 20
    port_dd_half: float = 0.10
    port_dd_stop: float = 0.15
    risk_per_trade: float = 0.012
    max_pos: int = 10
    rank: str = "breakout"         # breakout | ret60
    # 开关（消融）
    use_structure_stop: bool = True
    use_be_trail: bool = True
    use_3day: bool = True
    use_10day: bool = True
    use_time_stop: bool = True
    use_c1: bool = True
    use_c3: bool = True
    use_g2: bool = False
    fixed_hold: int = 0            # >0 = 固定持有期（A1/A2）
    # 成本/账户
    initial_cash: float = 1_000_000.0
    slippage_bps: float = 10.0
    transfer_par: float = 0.0010
    max_persist_days: int = 10
    delist_haircut: float = 0.0
    accounting: str = "shares"

    def family(self) -> str:
        fw = {"month": "F1", "quarter": "F2", "week": "F3"}[self.framework]
        st = "S2" if self.structure_mode == "S2" else "S1"
        if self.use_target and not self.use_be_trail:
            ex = "X1"
        elif self.use_c1 and self.use_c3:
            ex = "X3"
        else:
            ex = "X2"
        en = self.entry_mode if self.entry_mode in ("E1", "E2", "E3") else "E2"
        return f"{fw}-{st}-{en}-{ex}-G1"


DEFAULT = Params()


# ---------------------------------------------------------------------------
# 周期网格：日 / 周 / 月 / 季
# ---------------------------------------------------------------------------

_LEVEL_LABEL = {"week": "W-FRI", "month": "M", "quarter": "Q"}


@dataclass
class Grid:
    level: str
    dates: np.ndarray        # datetime64[D]：每个周期最后交易日
    pidx: np.ndarray         # 每个日索引 → 周期行号
    is_end: np.ndarray       # 每个日索引 → 是否周期最后交易日
    end_idx: np.ndarray      # 周期行号 → 日索引
    n: int


def build_grid(panel: FrozenPanel, level: str, cal_dates: pd.DatetimeIndex) -> Grid:
    n_days = panel.n_days
    if level == "day":
        e = np.arange(n_days)
        return Grid("day", np.array(panel.dates, dtype="datetime64[D]"), e.copy(),
                    np.ones(n_days, bool), e, n_days)
    lab = pd.DatetimeIndex(panel.dates).to_period(_LEVEL_LABEL[level]).asi8
    change = np.r_[True, lab[1:] != lab[:-1]]
    pidx = np.cumsum(change) - 1
    ends = np.flatnonzero(np.r_[pidx[1:] != pidx[:-1], True])
    # 周期完整性：用全市场交易日历判“该周期是否还有更晚的交易日”
    # （日历只含日期，不含价格；用于 §6.1“不得用未完成周期”）
    cal_lab = cal_dates.to_period(_LEVEL_LABEL[level])
    last_cal = pd.Series(cal_dates, index=cal_lab).groupby(level=0).max()
    cur_lab = pd.DatetimeIndex(panel.dates).to_period(_LEVEL_LABEL[level])[-1]
    if cur_lab in last_cal.index and last_cal[cur_lab] > pd.Timestamp(panel.dates[-1]):
        ends = ends[:-1]
        pidx = np.minimum(pidx, max(len(ends) - 1, 0))
    n = len(ends)
    return Grid(level, np.array([panel.dates[i] for i in ends], dtype="datetime64[D]"),
                pidx, np.r_[pidx[1:] != pidx[:-1], True], ends.astype(np.int64), n)


def aggregate_bars(panel: FrozenPanel, grid: Grid) -> dict[str, np.ndarray]:
    """把日线 ca 空间 OHLC 聚合到周期网格（个股停牌沿用其最后收盘）。"""
    P, C = grid.n, panel.n_codes
    o = np.full((P, C), np.nan, np.float32)
    h = np.full((P, C), np.nan, np.float32)
    lo = np.full((P, C), np.nan, np.float32)
    c = np.full((P, C), np.nan, np.float32)
    ca_o = panel.mat["ca_open"]
    ca_h = panel.mat["ca_high"]
    ca_l = panel.mat["ca_low"]
    ca_c = panel.mat["ca_close"]
    starts = np.r_[0, grid.end_idx[:-1] + 1]
    for p in range(P):
        a, b = int(starts[p]), int(grid.end_idx[p]) + 1
        sh, sl, sc, so = ca_h[a:b], ca_l[a:b], ca_c[a:b], ca_o[a:b]
        with np.errstate(all="ignore"):
            h[p] = np.nanmax(sh, axis=0)
            lo[p] = np.nanmin(sl, axis=0)
        fin = np.isfinite(sc)
        last = np.where(fin, np.arange(b - a)[:, None], -1).max(axis=0)
        okc = last >= 0
        c[p, okc] = sc[last[okc], np.flatnonzero(okc)]
        fin_o = np.isfinite(so)
        first = np.where(fin_o, np.arange(b - a)[:, None], b - a).min(axis=0)
        oko = first < (b - a)
        o[p, oko] = so[first[oko], np.flatnonzero(oko)]
    return {"o": o, "h": h, "l": lo, "c": c}


def ewm_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    """pandas ewm(alpha=1/n, adjust=False, min_periods=n) 等价，NaN 不更新。"""
    P, C = c.shape
    pc = np.vstack([np.full((1, C), np.nan, np.float32), c[:-1]])
    with np.errstate(all="ignore"):
        tr = np.nanmax(np.stack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    out = np.full((P, C), np.nan, np.float32)
    prev = np.full(C, np.nan, np.float64)
    cnt = np.zeros(C, np.int32)
    alpha = 1.0 / float(n)
    for t in range(P):
        x = tr[t].astype(np.float64)
        ok = np.isfinite(x)
        upd = ok & np.isfinite(prev)
        prev[upd] = prev[upd] + alpha * (x[upd] - prev[upd])
        new = ok & ~np.isfinite(prev)
        prev[new] = x[new]
        cnt[ok] += 1
        out[t] = np.where(cnt >= n, prev, np.nan)
    return out


def _ewm_mean(x: np.ndarray, n: int) -> np.ndarray:
    P, C = x.shape
    out = np.full((P, C), np.nan, np.float32)
    prev = np.full(C, np.nan, np.float64)
    cnt = np.zeros(C, np.int32)
    alpha = 2.0 / (n + 1.0)
    for t in range(P):
        v = x[t].astype(np.float64)
        ok = np.isfinite(v)
        upd = ok & np.isfinite(prev)
        prev[upd] = prev[upd] + alpha * (v[upd] - prev[upd])
        new = ok & ~np.isfinite(prev)
        prev[new] = v[new]
        cnt[ok] += 1
        out[t] = np.where(cnt >= n, prev, np.nan)
    return out


def _grid_to_daily(grid: Grid, arr: np.ndarray, n_days: int, fill) -> np.ndarray:
    p = grid.pidx - (~grid.is_end).astype(int)
    out = np.full((n_days,) + arr.shape[1:], fill, dtype=arr.dtype)
    ok = p >= 0
    out[ok] = arr[p[ok]]
    return out


def _fractal(series: np.ndarray, k: int, high: bool) -> np.ndarray:
    """严格不等式分型（k 左右各 k 根）；NaN 邻根自动不成立。"""
    P = series.shape[0]
    m = np.ones_like(series, dtype=bool)
    m &= np.isfinite(series)
    for j in range(1, k + 1):
        up = np.roll(series, j, axis=0)
        dn = np.roll(series, -j, axis=0)
        if high:
            m &= (series > up) & (series > dn)
        else:
            m &= (series < up) & (series < dn)
    if P > k:
        m[:k] = False
        m[-k:] = False
    return m


def levels_to_daily(panel: FrozenPanel, grid: Grid, bars: dict[str, np.ndarray],
                    k: int) -> dict[str, np.ndarray]:
    """结构层分型 → 日频阶梯函数（只有确认日 <= t 的信息）。"""
    P, C = grid.n, panel.n_codes
    n = panel.n_days
    fh = _fractal(bars["h"], k, high=True)
    fl = _fractal(bars["l"], k, high=False)
    out = {kk: np.full((n, C), np.nan, np.float32)
           for kk in ("last_h", "prev_h", "last_l", "prev_l")}
    start = np.full((n, C), -1, np.int32)
    days = np.arange(n, dtype=np.int64)
    for j in range(C):
        for name, frac, vals in (("h", fh[:, j], bars["h"][:, j]),
                                 ("l", fl[:, j], bars["l"][:, j])):
            rows = np.flatnonzero(frac)
            if len(rows) == 0:
                continue
            conf_rows = rows + k
            ok = conf_rows < P
            rows, conf_rows = rows[ok], conf_rows[ok]
            if len(rows) == 0:
                continue
            conf_days = grid.end_idx[conf_rows]
            v = vals[rows].astype(np.float32)
            li = np.searchsorted(conf_days, days, side="right") - 1
            has = li >= 0
            last = np.full(n, np.nan, np.float32)
            prev = np.full(n, np.nan, np.float32)
            last[has] = v[li[has]]
            has2 = li >= 1
            prev[has2] = v[li[has2] - 1]
            if name == "h":
                out["last_h"][:, j] = last
                out["prev_h"][:, j] = prev
                st = np.full(n, -1, np.int64)
                st[has] = conf_days[li[has]]
                start[:, j] = np.maximum(start[:, j], st.astype(np.int32))
            else:
                out["last_l"][:, j] = last
                out["prev_l"][:, j] = prev
    s1 = (out["last_h"] > out["prev_h"]) & (out["last_l"] > out["prev_l"]) \
        & np.isfinite(out["last_h"]) & np.isfinite(out["last_l"])
    out["s1_up"] = s1
    out["level_start"] = start
    return out


# ---------------------------------------------------------------------------
# 信号层
# ---------------------------------------------------------------------------

@dataclass
class Signals:
    params: Params
    mask: np.ndarray          # [n_days, C] bool：t 收盘信号
    score: np.ndarray         # 突破强度 (close-L)/ATR
    ret60: np.ndarray         # 60 日收益（rank 敏感性）
    L: np.ndarray             # 最后确认结构高点
    low: np.ndarray           # 最后确认结构低点
    atr: np.ndarray           # 触发层 ATR
    ma20: np.ndarray
    close: np.ndarray
    adv20: np.ndarray
    elig: np.ndarray
    s1_struct: np.ndarray
    s1_frame: np.ndarray
    diag: dict = field(default_factory=dict)


def _to_trigger(arr_daily: np.ndarray, grid: Grid) -> np.ndarray:
    return arr_daily[grid.end_idx]


def build_signals(panel: FrozenPanel, p: Params, cal_dates: pd.DatetimeIndex,
                  cache: dict | None = None) -> Signals:
    key = (p.framework, p.structure, p.trigger, p.k, p.atr_n, p.buffer_atr,
           p.band_atr, p.retest_days)
    if cache is not None and key in cache:
        return cache[key]
    grids = {}
    for lv in {p.framework, p.structure, p.trigger}:
        grids[lv] = build_grid(panel, lv, cal_dates)
    bars = {lv: aggregate_bars(panel, g) for lv, g in grids.items()}

    struct = levels_to_daily(panel, grids[p.structure], bars[p.structure], p.k)
    frame = levels_to_daily(panel, grids[p.framework], bars[p.framework], p.k)
    tg = grids[p.trigger]
    tb = bars[p.trigger]

    if p.trigger == "day" and p.atr_n == 14:
        atr = panel.atr14.astype(np.float32)
    else:
        atr = ewm_atr(tb["h"], tb["l"], tb["c"], p.atr_n)

    close_d = panel.mat["ca_close"].astype(np.float32)
    ma20_full = pd.DataFrame(close_d).rolling(20, min_periods=20).mean().to_numpy(np.float32)
    ret60 = (close_d / np.vstack([np.full((60, panel.n_codes), np.nan, np.float32),
                                  close_d[:-60]]) - 1.0).astype(np.float32)

    L_trig = _to_trigger(struct["last_h"], tg)
    start_trig = _to_trigger(struct["level_start"].astype(np.float32), tg).astype(np.int32)
    start_trig = np.where(np.isfinite(start_trig), start_trig, -1)
    s1s = _to_trigger(struct["s1_up"], tg)
    s1f = _to_trigger(frame["s1_up"], tg)
    low_trig = _to_trigger(struct["last_l"], tg)
    c, lo = tb["c"], tb["l"]
    # S2（V3）：结构线 = 对应周期收盘 EMA10 / 框架 EMA6，上升 = 收盘在线上且线不下降
    if p.structure_mode == "S2":
        ema_s = _ewm_mean(tb["c"], 10)
        prev_e = np.vstack([np.full((1, panel.n_codes), np.nan, np.float32), ema_s[:-1]])
        with np.errstate(all="ignore"):
            s1s = (tb["c"] > ema_s) & (ema_s >= prev_e) & np.isfinite(ema_s)
        low_trig = ema_s
    if p.framework_mode == "S2":
        fb = bars[p.framework]
        ema_f = _ewm_mean(fb["c"], 6)
        prev_f = np.vstack([np.full((1, panel.n_codes), np.nan, np.float32), ema_f[:-1]])
        with np.errstate(all="ignore"):
            s1f_grid = (fb["c"] > ema_f) & (ema_f >= prev_f) & np.isfinite(ema_f)
        s1f_daily = _grid_to_daily(grids[p.framework], s1f_grid, panel.n_days, False)
        s1f = _to_trigger(s1f_daily, tg)
    P = tg.n
    C = panel.n_codes
    rows = np.arange(P, dtype=np.int32)[:, None]
    with np.errstate(all="ignore"):
        above = np.isfinite(L_trig) & (c > L_trig + p.buffer_atr * atr)
        valid_level = np.isfinite(L_trig) & (start_trig >= 0)

    def _last_idx(cond: np.ndarray) -> np.ndarray:
        idx = np.where(cond, rows, np.int32(-1))
        return np.maximum.accumulate(idx, axis=0)

    last_above = _last_idx(above)
    last_above_prev = np.vstack([np.full((1, C), -1, np.int32), last_above[:-1]])
    first_above = above & (last_above_prev < start_trig) & valid_level
    b_idx = np.where(first_above, rows, np.int32(-1))
    b_prop = np.maximum.accumulate(b_idx, axis=0)
    age = rows - b_prop
    b_ok = valid_level & (b_prop >= 0) & (b_prop >= start_trig) & (b_prop <= rows - 1) \
        & (age <= p.retest_days)

    with np.errstate(all="ignore"):
        no_break = np.isfinite(L_trig) & (c >= L_trig - p.band_atr * atr)
        touch = np.isfinite(L_trig) & (lo <= L_trig + p.band_atr * atr)
    last_break = _last_idx(~no_break)
    last_touch = _last_idx(touch)
    last_touch_prev = np.vstack([np.full((1, C), -1, np.int32), last_touch[:-1]])
    break_ok = last_break < b_prop
    touch_ok = last_touch_prev >= b_prop

    c1 = np.vstack([np.full((1, C), np.nan, np.float32), c[:-1]])
    c2 = np.vstack([np.full((2, C), np.nan, np.float32), c[:-2]])
    with np.errstate(all="ignore"):
        high3 = (c > c1) & (c > c2)

    elig_t = _to_trigger(panel.elig, tg)
    if p.entry_mode == "E1":
        sig_t = first_above & s1s & s1f & elig_t & np.isfinite(atr) & valid_level
    elif p.entry_mode == "E3":
        ma20_t = _to_trigger(ma20_full, tg)
        prev_lo = np.vstack([np.full((1, panel.n_codes), np.nan, np.float32), lo[:-1]])
        with np.errstate(all="ignore"):
            sig_t = (s1s & s1f & elig_t & np.isfinite(ma20_t) & (c > ma20_t)
                     & (lo <= ma20_t) & (lo > prev_lo))
    else:
        sig_t = b_ok & break_ok & touch_ok & high3 & s1s & s1f & elig_t & np.isfinite(atr)
    # G2 市场门（仅消融用）：沪深300 收盘>MA200 且宽度>=40%
    if p.use_g2:
        sig_t = sig_t & _gate_g2(panel, tg)

    # 触发层信号 → 日频（只在周期最后交易日有效）
    n = panel.n_days
    mask = np.zeros((n, C), bool)
    if p.trigger == "day":
        mask[:] = sig_t
    else:
        ie = tg.is_end
        mask[ie] = sig_t[tg.pidx[ie]]

    score_t = np.where(np.isfinite(atr) & (atr > 0), (c - L_trig) / atr, np.nan).astype(np.float32)
    score = np.zeros((n, C), np.float32)
    if p.trigger == "day":
        score[:] = score_t
    else:
        ie = tg.is_end
        score[ie] = score_t[tg.pidx[ie]]

    ma20 = ma20_full

    # 诊断：各层漏斗
    diag = {
        "params": {k2: v for k2, v in asdict(p).items()},
        "grid_rows": {lv: int(g.n) for lv, g in grids.items()},
        "eligible_signal_days": int(mask.any(axis=1).sum()),
        "eligible_codes": int(mask.any(axis=0).sum()),
        "n_signals": int(mask.sum()),
        "by_year": {int(y): int(mask[[i for i, d in enumerate(panel.dates) if d.year == y]].sum())
                    for y in range(panel.dates[0].year, panel.dates[-1].year + 1)},
        "funnel": {
            "s1_struct_up_days": int(_to_trigger(struct["s1_up"], tg).any(axis=1).sum()),
            "s1_frame_up_days": int(_to_trigger(frame["s1_up"], tg).any(axis=1).sum()),
            "both_up_days": int((s1s & s1f).any(axis=1).sum()),
            "breakout_ok_days": int((b_ok & s1s & s1f).any(axis=1).sum()),
            "breakout_no_break_days": int((b_ok & break_ok & s1s & s1f).any(axis=1).sum()),
            "breakout_touch_days": int((b_ok & break_ok & touch_ok & s1s & s1f).any(axis=1).sum()),
            "high3_days": int(sig_t.any(axis=1).sum()),
        },
    }
    sig = Signals(p, mask, score, ret60, L_trig, low_trig, atr, ma20, close_d,
                  panel.adv20.astype(np.float32), panel.elig, s1s, s1f, diag)
    if cache is not None:
        cache[key] = sig
    return sig


def _gate_g2(panel: FrozenPanel, tg: Grid) -> np.ndarray:
    """G2：沪深300 收盘 > MA200 且主宇宙收盘 > MA60 的比例 >= 40%（仅 t 日及之前）。"""
    idx = load_index_series("2006-01-01", str(panel.dates[-1]))
    idx = idx.reindex(pd.DatetimeIndex(panel.dates)).ffill()
    ma200 = idx.rolling(200, min_periods=200).mean()
    ok_idx = (idx > ma200).to_numpy()
    close = panel.mat["ca_close"]
    ma60 = pd.DataFrame(close).rolling(60, min_periods=60).mean().to_numpy()
    with np.errstate(all="ignore"):
        breadth = ((close > ma60) & panel.listed & np.isfinite(ma60)).sum(axis=1) \
            / np.maximum(panel.listed.sum(axis=1), 1)
    ok = ok_idx & (breadth >= 0.40)
    return np.repeat(ok[:, None], panel.n_codes, axis=1)[tg.end_idx]


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------

def load_calendar() -> pd.DatetimeIndex:
    """冻结日历（只含日期，不含价格）。2021+ 行仅用于判断“周期是否走完”（Q25）。"""
    import duckdb
    con = duckdb.connect(str(ROOT / "01_数据" / "frozen" / "frozen.duckdb"), read_only=True)
    try:
        df = con.execute(
            "SELECT date FROM calendar WHERE is_trading_day=1 AND date BETWEEN '2004-01-01' AND '2026-10-26' ORDER BY date"
        ).df()
    finally:
        con.close()
    return pd.DatetimeIndex(pd.to_datetime(df["date"]))


def load_panel() -> tuple[FrozenPanel, EventBook]:
    eb = EventBook().load(PANEL_START, PANEL_END)
    panel = FrozenPanel(PANEL_START, PANEL_END, events=eb)
    return panel, eb


# ---------------------------------------------------------------------------
# 信号覆盖诊断
# ---------------------------------------------------------------------------

def write_signal_diag(panel: FrozenPanel, sig: Signals, path: Path) -> pd.DataFrame:
    rows = []
    for y in range(2007, 2021):
        idx = [i for i, d in enumerate(panel.dates) if d.year == y]
        m = sig.mask[idx]
        rows.append({
            "year": y,
            "trading_days": len(idx),
            "eligible_mean": round(float(panel.elig[idx].sum(axis=1).mean()), 2),
            "s1_struct_up_days": int(sig.s1_struct[idx].any(axis=1).sum()),
            "s1_frame_up_days": int(sig.s1_frame[idx].any(axis=1).sum()),
            "both_up_days": int((sig.s1_struct[idx] & sig.s1_frame[idx]).any(axis=1).sum()),
            "signal_days": int(m.any(axis=1).sum()),
            "signal_count": int(m.sum()),
            "signal_codes": int(m.any(axis=0).sum()),
            "in_dev": 2008 <= y <= 2015,
            "in_val": 2016 <= y <= 2020,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def cmd_signals(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = load_calendar()
    panel, eb = load_panel()
    log(f"panel {panel.n_days} 日 × {panel.n_codes} 只；事件 {eb.stats['n_events']}")
    sig = build_signals(panel, DEFAULT, cal)
    df = write_signal_diag(panel, sig, OUT / "01_信号覆盖.csv")
    print(df.to_string(index=False))
    print(json.dumps(sig.diag["funnel"], ensure_ascii=False, indent=2))
    (OUT / "01_信号覆盖_漏斗.json").write_text(
        json.dumps(_json_safe(sig.diag), ensure_ascii=False, indent=2), encoding="utf-8")
    # 自检：信号只在 <= 2020 的日期
    assert max(panel.dates) <= HARD_END
    dev = sig.mask[[i for i, d in enumerate(panel.dates) if date(2008, 1, 1) <= d <= HARD_END]].sum()
    log(f"信号总数 {sig.diag['n_signals']}，dev+val 内 {dev}")


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    if isinstance(x, (date, datetime, pd.Timestamp)):
        return str(x)
    return x


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T0005 主候选执行")
    ap.add_argument("mode", choices=["signals", "main", "reconcile", "ablation",
                                     "neighbor", "report", "all"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.mode == "signals":
        cmd_signals(args)
    elif args.mode in ("main", "all"):
        cmd_main(args)
    elif args.mode == "reconcile":
        cmd_reconcile(args)
    elif args.mode == "ablation":
        cmd_ablation(args)
    elif args.mode == "neighbor":
        cmd_neighbor(args)
    elif args.mode == "report":
        cmd_report(args)
    else:
        raise SystemExit(f"mode {args.mode} 尚未实现")
    return 0




# ---------------------------------------------------------------------------
# 状态机（§8）：X3 = X2 + C1 + C3
# ---------------------------------------------------------------------------

@dataclass
class PState:
    code: str
    entry_idx: int
    entry_date: date
    entry_price: float          # raw，含滑点
    entry_ca: float
    s0_ca: float
    r_ca: float
    r_pct: float
    signal_idx: int
    highest_ca: float
    stop_ca: float
    structure_low_ca: float
    half_done: bool = False
    caution: bool = False
    below_streak: int = 0
    above_streak: int = 0
    exit_pending: bool = False
    exit_reason: str = ""
    stop_order: str | None = None
    stop_price_raw: float = float("nan")
    stop_qty: int = -1
    exit_order: str | None = None
    half_order: str | None = None
    target_order: str | None = None
    target_price_raw: float = float("nan")
    target_qty: int = -1
    mfe_max: float = 0.0
    days_held: int = 0
    be_active: bool = False
    trail_active: bool = False


def _submit_exit(acct, st: PState, d: date, reason: str) -> None:
    if st.exit_pending:
        return
    st.exit_pending = True
    st.exit_reason = reason
    for oid in (st.stop_order, st.half_order, st.target_order):
        if oid:
            acct.cancel(oid, "exit_replace")
    st.stop_order = None
    st.half_order = None
    st.target_order = None
    pos = acct.positions.get(st.code)
    if not pos or pos.shares <= 0:
        return
    o = acct.submit(Order(code=st.code, side="sell", qty=int(pos.shares),
                          kind="open", tif="gtc", tag=reason), d)
    st.exit_order = o.order_id


def _submit_half(acct, st: PState, d: date, reason: str) -> None:
    if st.half_done:
        return
    pos = acct.positions.get(st.code)
    if not pos or pos.shares < 200:
        st.half_done = True
        return
    qty = int((pos.shares // 2) // 100 * 100)
    if qty < 100:
        st.half_done = True
        return
    o = acct.submit(Order(code=st.code, side="sell", qty=qty, kind="open",
                          tif="gtc", tag=f"half_{reason}"), d)
    st.half_order = o.order_id
    st.half_done = True
    st.exit_reason = st.exit_reason or f"half_{reason}"


def _sync_target(acct, st: PState, i: int, panel: FrozenPanel, p: Params, d: date) -> None:
    """X1 固定 2R 目标：ca 空间目标价按次日除权因子换算成 raw 限价，每日重挂。"""
    if not p.use_target or st.exit_pending:
        return
    pos = acct.positions.get(st.code)
    if not pos or pos.shares <= 0:
        return
    j = panel.code_idx[st.code]
    f_next = float(panel.ca_factor[i + 1, j]) if i + 1 < panel.n_days else 1.0
    raw = (st.entry_ca + p.target_r * st.r_ca) * f_next
    if not np.isfinite(raw) or raw <= 0:
        return
    price = round(float(raw), 2)
    cur = acct.orders.get(st.target_order or "")
    need = (cur is None or not cur.active
            or abs(price - st.target_price_raw) > 0.005
            or int(st.target_qty) != int(pos.shares))
    if not need:
        return
    if cur is not None and cur.active:
        acct.cancel(cur.order_id, "target_reprice")
    o = acct.submit(Order(code=st.code, side="sell", qty=int(pos.shares),
                          kind="limit", price=price, tif="gtc", tag="target_2r"), d)
    st.target_order = o.order_id
    st.target_price_raw = price
    st.target_qty = int(pos.shares)


def _sync_stop(acct, st: PState, i: int, panel: FrozenPanel, d: date) -> None:
    if st.exit_pending:
        return
    pos = acct.positions.get(st.code)
    if not pos or pos.shares <= 0:
        return
    j = panel.code_idx[st.code]
    f_next = float(panel.ca_factor[i + 1, j]) if i + 1 < panel.n_days else 1.0
    stop_raw = float(st.stop_ca) * f_next
    close_raw = float(panel.mat["close"][i, j])
    if not np.isfinite(stop_raw) or stop_raw <= 0:
        return
    if stop_raw >= close_raw - 1e-9:
        _submit_exit(acct, st, d, "stop_above_close")
        return
    price = round(max(stop_raw, 0.01), 2)
    cur = acct.orders.get(st.stop_order or "")
    need = (cur is None or not cur.active
            or abs(price - st.stop_price_raw) > 0.005
            or int(st.stop_qty) != int(pos.shares))
    if not need:
        return
    if cur is not None and cur.active:
        acct.cancel(cur.order_id, "stop_reprice")
    o = acct.submit(Order(code=st.code, side="sell", qty=int(pos.shares),
                          kind="stop", price=price, tif="gtc", tag="structure_stop"), d)
    st.stop_order = o.order_id
    st.stop_price_raw = price
    st.stop_qty = int(pos.shares)


def _size_shares(panel: FrozenPanel, sig: Signals, i: int, j: int, nav: float,
                 cash: float, p: Params, half: bool = False) -> tuple[int, float]:
    """Q14：信号日收盘估值。返回 (股数, R占价格比)。"""
    close_ca = float(sig.close[i, j])
    low_ca = float(sig.low[i, j])
    atr_ca = float(sig.atr[i, j])
    if not (np.isfinite(close_ca) and close_ca > 0 and np.isfinite(low_ca)
            and np.isfinite(atr_ca) and atr_ca > 0):
        return 0, float("nan")
    s0_ca = low_ca - p.stop_atr * atr_ca
    r_ca = close_ca - s0_ca
    if r_ca <= 0:
        return 0, float("nan")
    r_pct = r_ca / close_ca
    if r_pct > p.max_risk_pct:
        return 0, r_pct
    f = float(panel.ca_factor[i, j])
    p_raw = close_ca * f
    s0_raw = s0_ca * f
    r_raw = r_ca * f
    budget = nav / max(p.max_pos, 1) * (0.5 if half else 1.0)
    shares = int(budget / p_raw / 100) * 100
    risk_sh = int(p.risk_per_trade * nav / max(r_raw, 1e-9) / 100) * 100
    cash_sh = int(max(cash, 0.0) * 0.98 / (p_raw * 1.01) / 100) * 100
    return int(max(0, min(shares, risk_sh, cash_sh))), r_pct


def run_account(panel: FrozenPanel, sig: Signals, p: Params, window: str,
                initial_cash: float | None = None, label: str = "") -> dict:
    start, end = WINDOWS[window]
    i0, i1 = panel.window_index(start, end)
    if i0 == 0:
        raise ValueError("窗口首日无前一日")
    cfgs = dict(variant="T0005", window=window, accounting=p.accounting,
                initial_cash=float(p.initial_cash if initial_cash is None else initial_cash),
                slots=p.max_pos, slippage_bps=p.slippage_bps,
                transfer_par_rate=p.transfer_par, delist_haircut=p.delist_haircut,
                max_persist_days=p.max_persist_days)
    cfg = RunConfig(**cfgs)
    acct = _make_account(panel, cfg)
    acct._code_pos = panel.code_idx
    events = panel.events
    idx_nav = np.arange(panel.n_days)

    states: dict[str, PState] = {}
    closed_meta: dict[tuple, dict] = {}
    pending_buy: dict[str, dict] = {}
    nav_rows: list[dict] = []
    prev_cash = float(acct.cash_total)
    peak = float(cfg.initial_cash)
    block_until = -1
    checks = {
        "cash_resid_max": 0.0, "nav_resid_max": 0.0, "cash_resid_bad_days": 0,
        "nav_resid_bad_days": 0, "t1_violations": 0, "forced_count": 0,
        "delist_forced": 0, "persist_max": 0, "n_orders": 0, "n_filled": 0,
        "max_positions": 0, "risk_violations": 0, "max_realized_risk_pct": 0.0,
        "n_buy_submitted": 0, "n_buy_failed": 0, "max_price_r_pct": 0.0,
    }
    trade_ptr = 0
    ca_ptr = 0
    tax_ptr = 0

    def _bar_close(i: int, j: int, pos) -> float:
        px = float(panel.mat["close"][i, j])
        if not np.isfinite(px) or px <= 0:
            px = float(pos.last_price) if pos.last_price else 0.0
        return px

    for i in range(i0 - 1, i1 + 1):
        d = panel.dates[i]
        if d >= date(2021, 1, 1):
            raise RuntimeError("试图运行 2021+")
        held = list(acct.positions.keys())
        active = [o.code for o in acct._orders if o.active]
        ev_today = events.on(d)
        needed = set(held) | set(active) | {e.symbol for e in ev_today}
        acct._ca_row = panel.ca_factor[i]
        acct._adv = {c: float(panel.adv20[i - 1, panel.code_idx[c]])
                     if i > 0 and c in panel.code_idx and np.isfinite(panel.adv20[i - 1, panel.code_idx[c]])
                     else 0.0 for c in set(active) | set(held)}
        acct._resume_today = {panel.codes[jj] for jj in np.flatnonzero(panel.resume[i])}
        rec = acct.on_day(d, bars=panel.bars_on(i, needed), events=ev_today)

        # 卖出顺延上限（§9.2）：只对“全仓退出”开盘卖单生效；
        # 止损单是 GTC 长期挂单、减半单不是清仓，都不触发强退（Q16）。
        forced_now = False
        for o in list(acct._orders):
            if not (o.active and o.side == "sell" and o.kind == "open" and o.filled_qty == 0):
                continue
            try:
                age = i - panel.day_idx[o.execute_from]
            except KeyError:
                continue
            if age < p.max_persist_days:
                continue
            if str(o.tag).startswith("half"):
                acct.cancel(o.order_id, "half_persist_cancel")
                continue
            pos = acct.positions.get(o.code)
            jj = panel.code_idx.get(o.code)
            if pos is None or jj is None:
                continue
            acct._force_sell(o.code, d, _bar_close(i, jj, pos), "sell_persistence_10d")
            acct._forced_ages.append((o.order_id, int(age)))
            forced_now = True
        if forced_now:
            rec = acct._mark_and_record(d)

        # 会计恒等
        new_fills = acct.trades[trade_ptr:]
        buys = sum(t.amount + t.fees for t in new_fills if t.side == "buy")
        sells = sum(t.amount - t.fees for t in new_fills if t.side == "sell")
        ca_cash = sum(x["credit"] for x in acct.ca_rows[ca_ptr:])
        tax_cash = sum(x["amount"] for x in acct.tax_ledger[tax_ptr:])
        delta = float(rec["cash_total"]) - prev_cash
        resid = abs(delta - (sells - buys + ca_cash + tax_cash))
        checks["cash_resid_max"] = max(checks["cash_resid_max"], resid)
        if resid > 0.02:
            checks["cash_resid_bad_days"] += 1
        prev_cash = float(rec["cash_total"])
        mv = 0.0
        for code, pos in acct.positions.items():
            jj = panel.code_idx.get(code)
            if jj is None:
                continue
            mv += pos.shares * _bar_close(i, jj, pos)
        nav_resid = abs(mv - float(rec["market_value"]))
        checks["nav_resid_max"] = max(checks["nav_resid_max"], nav_resid)
        if nav_resid > 0.02:
            checks["nav_resid_bad_days"] += 1
        trade_ptr = len(acct.trades)
        ca_ptr = len(acct.ca_rows)
        tax_ptr = len(acct.tax_ledger)

        nav = float(rec["nav"])

        # ---- 成交后同步状态（建仓） -------------------------------------
        for code, meta in list(acct.hold_meta.items()):
            if meta.get("entry_date") != d:
                continue
            if code in states:
                if states[code].entry_date == d:
                    continue
                old_st = states.pop(code)
                rt = [x for x in acct.round_trips if x["code"] == code
                      and x["entry_date"] == old_st.entry_date.isoformat()]
                if rt:
                    closed_meta[(code, old_st.entry_date)] = {
                        "mfe_max_r": round(float(old_st.mfe_max), 4),
                        "entry_r_pct": round(float(old_st.r_pct), 6),
                        "half_done": bool(old_st.half_done),
                        "caution": bool(old_st.caution),
                        "exit_reason_state": old_st.exit_reason,
                        "days_held": int(old_st.days_held),
                        "be_active": bool(old_st.be_active),
                        "trail_active": bool(old_st.trail_active),
                    }
            jj = panel.code_idx.get(code)
            if jj is None:
                continue
            pb = pending_buy.get(code)
            if pb is None:
                # 理论上不会发生；保守地按当日确认低点补
                low_ca = float(sig.low[i, jj])
                atr_ca = float(sig.atr[i, jj])
                s0_ca = low_ca - p.stop_atr * atr_ca
                sig_idx = i
            else:
                sig_idx = pb["signal_idx"]
                s0_ca = pb["s0_ca"]
                low_ca = pb["low_ca"]
            entry_ca = float(meta["entry_ca"])
            entry_price = float(meta["entry_price"])
            r_ca = entry_ca - s0_ca
            st = PState(code=code, entry_idx=i, entry_date=d, entry_price=entry_price,
                        entry_ca=entry_ca, s0_ca=s0_ca, r_ca=r_ca,
                        r_pct=float((entry_price - s0_ca * panel.ca_factor[i, jj]) / entry_price),
                        signal_idx=int(sig_idx), highest_ca=entry_ca, stop_ca=s0_ca,
                        structure_low_ca=low_ca)
            states[code] = st
            if np.isfinite(st.r_pct):
                checks["max_price_r_pct"] = max(float(checks["max_price_r_pct"]), float(st.r_pct))
            nav_sig = float(pb["nav"]) if pb else nav
            r_raw = st.r_ca * float(panel.ca_factor[i, jj])
            acct_risk = float(meta["entry_shares"]) * r_raw / max(nav_sig, 1.0)
            checks["max_realized_risk_pct"] = max(checks["max_realized_risk_pct"], acct_risk)
            if acct_risk > p.risk_per_trade * 1.10:
                checks["risk_violations"] += 1
            pending_buy.pop(code, None)

        # ---- 平仓后归档 --------------------------------------------------
        for code in [c for c in states if c not in acct.hold_meta]:
            st = states.pop(code)
            rt = [x for x in acct.round_trips if x["code"] == code
                  and x["entry_date"] == st.entry_date.isoformat()]
            if rt:
                r0 = rt[-1]
                closed_meta[(code, st.entry_date)] = {
                    "mfe_max_r": round(float(st.mfe_max), 4),
                    "entry_r_pct": round(float(st.r_pct), 6),
                    "half_done": bool(st.half_done),
                    "caution": bool(st.caution),
                    "exit_reason_state": st.exit_reason,
                    "days_held": int(st.days_held),
                    "be_active": bool(st.be_active),
                    "trail_active": bool(st.trail_active),
                }

        # ---- 逐日状态更新 ------------------------------------------------
        for code, st in list(states.items()):
            if code not in acct.positions or code not in acct.hold_meta:
                continue
            jj = panel.code_idx[code]
            hi = float(panel.mat["ca_high"][i, jj])
            if np.isfinite(hi):
                st.highest_ca = max(st.highest_ca, hi)
            st.days_held = i - st.entry_idx + 1
            st.mfe_max = (st.highest_ca - st.entry_ca) / st.r_ca if st.r_ca > 0 else 0.0
            if st.days_held <= 1:
                continue      # 建仓日不判里程碑（Q15）
            if i >= i1:
                continue      # 最后一日只记 MFE，不再生成订单
            close_ca = float(panel.mat["ca_close"][i, jj])
            atr_ca = float(sig.atr[i, jj])
            low_ca = float(sig.low[i, jj])
            close_raw = float(panel.mat["close"][i, jj])
            mfe = st.mfe_max
            # 结构破坏（§8.5-4）
            if p.use_structure_stop and np.isfinite(low_ca) and np.isfinite(atr_ca):
                if close_ca < low_ca - p.stop_atr * atr_ca:
                    _submit_exit(acct, st, d, "structure_break")
                    continue
            # 固定持有（A1/A2）
            if p.fixed_hold and st.days_held >= p.fixed_hold:
                _submit_exit(acct, st, d, "fixed_hold")
                continue
            # 3 日失败
            if p.use_3day and st.days_held == p.fail_days and mfe < p.fail_mfe_r \
                    and close_raw < st.entry_price:
                _submit_exit(acct, st, d, "fail_3d")
                continue
            # 10 日减半
            if p.use_10day and st.days_held == p.half_days and mfe < 1.0 and not st.half_done:
                _submit_half(acct, st, d, "10d")
            # 20 日时间止损
            if p.use_time_stop and st.days_held >= p.time_stop_days and mfe < 1.0:
                _submit_exit(acct, st, d, "time_20d")
                continue
            # C1 谨慎态
            if p.use_c1 and np.isfinite(sig.ma20[i, jj]) and np.isfinite(atr_ca):
                ma20 = float(sig.ma20[i, jj])
                if close_ca < ma20:
                    st.below_streak += 1
                    st.above_streak = 0
                else:
                    st.above_streak += 1
                    st.below_streak = 0
                if not st.caution and st.below_streak >= 2:
                    st.caution = True
                    st.stop_ca = max(st.stop_ca, low_ca - p.stop_atr * atr_ca)
                    if not st.half_done:
                        _submit_half(acct, st, d, "c1")
                    if st.stop_ca * float(panel.ca_factor[i, jj]) >= close_raw - 1e-9:
                        _submit_exit(acct, st, d, "c1_stop_above")
                        continue
                elif st.caution and st.above_streak >= 3 and mfe >= 1.0:
                    st.caution = False
            # 止损上移（BE / 吊灯）
            if p.use_be_trail:
                if mfe >= p.be_mfe_r and not st.be_active:
                    st.stop_ca = max(st.stop_ca, st.entry_ca)
                    st.be_active = True
                if mfe >= p.trail_mfe_r and np.isfinite(atr_ca):
                    st.trail_active = True
                    st.stop_ca = max(st.stop_ca, st.highest_ca - p.trail_atr * atr_ca)

        # ---- 记录日净值 ------------------------------------------------
        if i >= i0:
            nav_rows.append({
                "date": d.isoformat(), "nav": round(nav, 4),
                "cash": round(float(rec["cash_total"]), 4),
                "market_value": round(float(rec["market_value"]), 4),
                "n_positions": int(rec["n_positions"]),
                "n_active_orders": int(rec["n_active_orders"]),
            })
        peak = max(peak, nav)
        if i >= i1:
            break

        # ---- t 收盘：生成 t+1 订单 --------------------------------------
        for code, st in list(states.items()):
            if code not in acct.positions:
                continue
            if p.use_structure_stop or p.use_be_trail:
                _sync_stop(acct, st, i, panel, d)
            _sync_target(acct, st, i, panel, p, d)

        # 组合回撤（C3）
        dd = nav / peak - 1.0 if peak > 0 else 0.0
        allow_buy = True
        half_size = False
        if p.use_c3:
            if dd <= -p.port_dd_stop:
                block_until = max(block_until, i + 20)
            if dd > -p.port_dd_half:
                block_until = min(block_until, i)
            half_size = dd <= -p.port_dd_half
            if i < block_until:
                allow_buy = False

        # 入场
        open_buys = {o.code for o in acct._orders if o.active and o.side == "buy"}
        n_hold = sum(1 for x in acct.positions.values() if x.shares > 0)
        free = p.max_pos - n_hold - len(open_buys)
        if free > 0 and allow_buy:
            cand = sig.mask[i].copy()
            for code in set(acct.positions) | open_buys:
                jj = panel.code_idx.get(code)
                if jj is not None:
                    cand[jj] = False
            # 有未完成卖单的票不再买
            for o in acct._orders:
                if o.active and o.side == "sell":
                    jj = panel.code_idx.get(o.code)
                    if jj is not None:
                        cand[jj] = False
            js = np.flatnonzero(cand)
            if len(js):
                if p.rank == "ret60":
                    sc = np.nan_to_num(sig.ret60[i, js], nan=-np.inf)
                else:
                    sc = np.nan_to_num(sig.score[i, js], nan=-np.inf)
                adv = np.nan_to_num(sig.adv20[i, js], nan=-np.inf)
                order = np.lexsort((js, -adv, -sc))
                cash_est = float(acct.cash_available)
                n_new = 0
                for kk in order:
                    if n_new >= free:
                        break
                    jj = int(js[kk])
                    code = panel.codes[jj]
                    sh, r_pct = _size_shares(panel, sig, i, jj, nav, cash_est, p, half_size)
                    if sh < 100 or not np.isfinite(r_pct):
                        continue
                    o = acct.submit(Order(code=code, side="buy", qty=int(sh), kind="open",
                                          tif="day", tag="e2"), d)
                    pending_buy[code] = {
                        "signal_idx": i, "s0_ca": float(sig.low[i, jj] - p.stop_atr * sig.atr[i, jj]),
                        "low_ca": float(sig.low[i, jj]), "score": float(sc[kk]),
                        "nav": float(nav), "shares": int(sh),
                    }
                    cash_est -= sh * float(panel.mat["close"][i, jj]) * 1.01
                    n_new += 1
                    checks["n_buy_submitted"] += 1

        checks["max_positions"] = max(checks["max_positions"], int(rec["n_positions"]))

    acct.finalize(panel.dates[i1])
    orders_df = pd.DataFrame([o.as_dict() for o in acct.orders.values()])
    if len(orders_df):
        orders_df["decision_date"] = pd.to_datetime(orders_df["decision_date"])
        orders_df["execute_from"] = pd.to_datetime(orders_df["execute_from"])
    failed = _failed_orders(orders_df, acct, panel)
    if len(failed):
        _ignore = {"exit_replace", "target_reprice", "half_persist_cancel", "stop_reprice"}
        failed = failed[~failed["reason"].astype(str).isin(_ignore)].copy()
    eq = pd.DataFrame(acct.daily).rename(columns={"cash_total": "cash"})
    eq = eq[(eq["date"] >= start) & (eq["date"] <= end)].copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date").sort_index()
    fills = pd.DataFrame([t.as_dict() for t in acct.trades])
    if len(fills):
        fills["date"] = pd.to_datetime(fills["date"])
        by_day = fills.groupby("date")
        eq["turnover"] = by_day["amount"].sum().reindex(eq.index).fillna(0.0)
        eq["cost"] = by_day["fees"].sum().reindex(eq.index).fillna(0.0)
    else:
        eq["turnover"] = 0.0
        eq["cost"] = 0.0
    trades = pd.DataFrame(acct.round_trips)
    if len(trades):
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])
        meta_rows = []
        for r in trades.itertuples(index=False):
            m = closed_meta.get((r.code, pd.Timestamp(r.entry_date).date()), {})
            meta_rows.append(m)
        for col in ("mfe_max_r", "entry_r_pct", "half_done", "caution", "exit_reason_state",
                    "be_active", "trail_active"):
            trades[col] = [m.get(col, None) for m in meta_rows]
        trades = trades.sort_values("exit_date").reset_index(drop=True)
    checks.update({
        "n_orders": int(len(orders_df)),
        "n_filled": int(sum(1 for o in acct.orders.values() if o.status == "filled")),
        "forced_count": int(sum(acct._force_counts.values())),
        "delist_forced": int(acct._force_counts.get("delisting_out_date", 0)),
        "n_fills": int(len(fills)),
        "n_round_trips": int(len(trades)),
        "n_failed_orders": int(len(failed)),
        "n_rows_after_2020": int((eq.index >= pd.Timestamp("2021-01-01")).sum()),
        "max_date": str(eq.index.max().date()) if len(eq) else "",
        "n_ca_events": int(len(acct.ca_rows)),
        "n_tax_events": int(len(acct.tax_ledger)),
        "bad_bars": int(panel.n_bad_bars),
        "n_events_book": int(len(acct.event_ledger)),
    })
    # 期末强制平仓（替代口径，仅报告）
    nav_end = float(eq["nav"].iloc[-1]) if len(eq) else float(cfg.initial_cash)
    liq_cost = 0.0
    for code, pos in acct.positions.items():
        if pos.shares <= 0:
            continue
        jj = panel.code_idx.get(code)
        if jj is None:
            continue
        px = _bar_close(i1, jj, pos)
        liq_cost += pos.shares * px * (0.00025 + 0.001)  # 佣金 + 卖出印花税近似
    checks["liq_nav_after_cost"] = round(nav_end - liq_cost, 2)
    return {
        "cfg": p, "window": window, "label": label, "i0": i0, "i1": i1,
        "equity": eq, "fills": fills, "trades": trades, "failed": failed,
        "orders": orders_df, "checks": checks, "acct": acct,
        "states": states, "closed_meta": closed_meta,
    }


# ---------------------------------------------------------------------------
# 统计与输出
# ---------------------------------------------------------------------------

def window_benchmark(panel: FrozenPanel, i0: int, i1: int) -> tuple[pd.Series, pd.Series]:
    b1 = panel.benchmark_ew(i0, i1)
    nav1 = (1.0 + b1).cumprod()
    nav1.index = pd.to_datetime(nav1.index)
    hs = load_index_series(str(panel.dates[i0]), str(panel.dates[i1]), "000300.SH")
    return nav1, hs


def ex_top5_nav(eq: pd.DataFrame, trades: pd.DataFrame, k: int = 5) -> pd.Series:
    """去掉前 k 大盈利笔后（从退出日起扣减）的近似净值。"""
    nav = eq["nav"].astype(float).copy()
    if not len(trades):
        return nav
    tr = trades.sort_values("pnl", ascending=False).head(k)
    for r in tr.itertuples(index=False):
        pnl = float(r.pnl)
        if not np.isfinite(pnl):
            continue
        nav.loc[nav.index >= pd.Timestamp(r.exit_date)] -= pnl
    return nav


def metrics_bundle(res: dict, panel: FrozenPanel, p: Params, bench_nav: pd.Series,
                   hs: pd.Series) -> dict:
    eq = res["equity"]
    trades = res["trades"]
    br = BacktestResult(eq[["nav", "market_value", "cash", "n_positions", "turnover", "cost"]],
                        trades, meta={"initial_cash": p.initial_cash})
    b = bench_nav.reindex(eq.index).ffill()
    m = compute_metrics(br, initial_cash=p.initial_cash, benchmark=b)
    m["benchmark_cagr"] = float((b.iloc[-1] / b.iloc[-1]) ** 0) if False else m.get("benchmark_cagr")
    try:
        m["bootstrap"] = block_bootstrap(br.returns, n_boot=2000, method="stationary",
                                         block_len=20, seed=SEED, dates=eq.index)
    except Exception as exc:  # pragma: no cover
        m["bootstrap"] = {"error": str(exc)}
    # 集中度
    if len(trades):
        pnl = trades["pnl"].astype(float)
        tot = float(pnl[pnl > 0].sum())
        srt = pnl.sort_values(ascending=False)
        m["top1_share"] = float(srt.iloc[0] / tot) if tot > 0 else np.nan
        m["top5_share"] = float(srt.head(5).sum() / tot) if tot > 0 else np.nan
        by_code = trades.groupby("code")["pnl"].sum()
        m["max_stock_share"] = float(by_code.max() / tot) if tot > 0 and len(by_code) else np.nan
        trades2 = trades.copy()
        trades2["year"] = pd.to_datetime(trades2["exit_date"]).dt.year
        by_year = trades2.groupby("year")["pnl"].sum()
        m["max_year_share"] = float(by_year.max() / tot) if tot > 0 and len(by_year) else np.nan
        m["n_codes"] = int(trades["code"].nunique())
        m["n_years_with_trades"] = int(by_year.size)
        # 去前 5 大盈利
        nav_x = ex_top5_nav(eq, trades)
        yrs = max((nav_x.index[-1] - nav_x.index[0]).days, 1) / 365.25
        m["cagr_ex_top5"] = float((nav_x.iloc[-1] / p.initial_cash) ** (1 / yrs) - 1.0)
        m["final_nav_ex_top5"] = float(nav_x.iloc[-1])
    else:
        m["n_codes"] = 0
    m["hs300_cagr"] = float((hs.iloc[-1] / hs.iloc[0]) ** (365.25 / max((hs.index[-1] - hs.index[0]).days, 1)) - 1.0)
    m["window"] = res["window"]
    m["tag"] = p.tag
    return m


def selfcheck(res: dict, panel: FrozenPanel, sig: Signals, p: Params) -> dict:
    eq = res["equity"]
    fills = res["fills"]
    orders = res["orders"]
    checks = res["checks"]
    out = {}
    out["cash_identity_max_abs"] = round(float(checks["cash_resid_max"]), 6)
    out["cash_identity_bad_days"] = int(checks["cash_resid_bad_days"])
    out["nav_identity_max_abs"] = round(float(checks["nav_resid_max"]), 6)
    out["nav_identity_bad_days"] = int(checks["nav_resid_bad_days"])
    # T+1
    t1_bad = 0
    if len(fills):
        entry = fills[fills["side"] == "buy"].groupby("code")["date"].min()
        for r in fills[fills["side"] == "sell"].itertuples(index=False):
            e = entry.get(r.code)
            if e is not None and r.date <= e:
                t1_bad += 1
    out["t1_violations"] = int(t1_bad)
    # 停牌成交
    susp_bad = 0
    lim_bad = 0
    forced_susp = 0
    if len(fills):
        for r in fills.itertuples(index=False):
            j = panel.code_idx.get(r.code)
            if j is None:
                continue
            di = panel.day_idx.get(pd.Timestamp(r.date).date())
            if di is None:
                continue
            forced = str(r.kind) == "forced" or str(r.reason).startswith("sell_persistence")
            if panel.suspended[di, j]:
                if forced:
                    forced_susp += 1
                else:
                    susp_bad += 1
            if not forced:
                if r.side == "buy" and panel.mat["is_limit_up_open"][di, j]:
                    lim_bad += 1
                if r.side == "sell" and panel.mat["is_limit_down_open"][di, j]:
                    lim_bad += 1
    out["fills_on_suspension"] = int(susp_bad)
    out["fills_at_locked_limit_open"] = int(lim_bad)
    out["forced_valuation_fills_on_suspension"] = int(forced_susp)
    out["max_positions"] = int(checks["max_positions"])
    out["risk_violations"] = int(checks["risk_violations"])
    out["max_realized_risk_pct"] = round(float(checks["max_realized_risk_pct"]), 6)
    out["max_price_r_pct"] = round(float(checks["max_price_r_pct"]), 6)
    out["rows_after_2020"] = int(checks["n_rows_after_2020"])
    out["max_date"] = checks["max_date"]
    out["forced_sells"] = int(checks["forced_count"])
    out["delist_forced"] = int(checks["delist_forced"])
    out["n_orders"] = int(checks["n_orders"])
    out["n_fills"] = int(checks["n_fills"])
    out["n_round_trips"] = int(checks["n_round_trips"])
    out["n_failed_orders"] = int(checks["n_failed_orders"])
    out["n_ca_events"] = int(checks["n_ca_events"])
    out["n_tax_events"] = int(checks["n_tax_events"])
    out["bad_bars"] = int(checks["bad_bars"])
    # 订单时序
    bad_seq = 0
    if len(orders):
        dd = pd.to_datetime(orders["decision_date"])
        ee = pd.to_datetime(orders["execute_from"])
        bad_seq = int((ee <= dd).sum())
    out["orders_execute_before_decision"] = bad_seq
    # 期末
    out["final_nav"] = float(eq["nav"].iloc[-1]) if len(eq) else None
    out["liq_nav_after_cost"] = float(checks["liq_nav_after_cost"])
    # 净值恒等（自算）
    if len(eq):
        resid = (eq["nav"] - (eq["cash"] + eq["market_value"])).abs().max()
        out["nav_identity_self_max_abs"] = round(float(resid), 6)
    return out


def write_run(res: dict, m: dict, sc: dict, prefix: str, window: str,
              ann: pd.DataFrame, mon: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eq = res["equity"]
    eq.to_csv(OUT / f"03_日净值_{prefix}_{window}.csv", encoding="utf-8-sig")
    if len(res["trades"]):
        res["trades"].to_csv(OUT / f"02_逐笔_{prefix}_{window}.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["code", "entry_date", "exit_date", "pnl"]).to_csv(
            OUT / f"02_逐笔_{prefix}_{window}.csv", index=False, encoding="utf-8-sig")
    ann.to_csv(OUT / f"04_年度_{prefix}_{window}.csv", index=False, encoding="utf-8-sig")
    if len(res["failed"]):
        res["failed"].to_csv(OUT / f"05_失败订单_{prefix}_{window}.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["code", "side", "status", "reason"]).to_csv(
            OUT / f"05_失败订单_{prefix}_{window}.csv", index=False, encoding="utf-8-sig")
    if len(res["fills"]):
        res["fills"].to_csv(OUT / f"06_成交_{prefix}_{window}.csv", index=False, encoding="utf-8-sig")
    (OUT / f"07_统计_{prefix}_{window}.json").write_text(
        json.dumps(_json_safe(m), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"08_自检_{prefix}_{window}.json").write_text(
        json.dumps(_json_safe(sc), ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_main(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = load_calendar()
    panel, eb = load_panel()
    p = DEFAULT
    sig = build_signals(panel, p, cal)
    allres = {}
    for window in ("dev", "val"):
        t = time.time()
        res = run_account(panel, sig, p, window, label="T0005")
        i0, i1 = res["i0"], res["i1"]
        bench_nav, hs = window_benchmark(panel, i0, i1)
        br = BacktestResult(res["equity"][["nav", "market_value", "cash", "n_positions",
                                          "turnover", "cost"]], res["trades"],
                            meta={"initial_cash": p.initial_cash})
        ann = annual_table(br, initial_cash=p.initial_cash)
        mon = monthly_matrix(br, initial_cash=p.initial_cash)
        m = metrics_bundle(res, panel, p, bench_nav, hs)
        sc = selfcheck(res, panel, sig, p)
        write_run(res, m, sc, "T0005", window, ann, mon)
        allres[window] = {"metrics": m, "selfcheck": sc,
                          "rows": len(res["equity"]), "n_trades": int(len(res["trades"]))}
        log(f"{window}: 年化 {m['cagr']*100:.2f}% 回撤 {m['mdd']*100:.2f}% "
            f"Calmar {m['calmar']:.2f} 交易 {m.get('n_trades',0)} "
            f"持仓均值 {m['avg_positions']:.2f} 成本 {m['cost_paid']:.0f} "
            f"({time.time()-t:.1f}s)")
        print(json.dumps(_json_safe(m), ensure_ascii=False, indent=1)[:1500])
        print(json.dumps(_json_safe(sc), ensure_ascii=False, indent=1))
    (OUT / "07_统计_主候选.json").write_text(
        json.dumps(_json_safe(allres), ensure_ascii=False, indent=2), encoding="utf-8")




# ---------------------------------------------------------------------------
# 独立账户对账（不 import 账户撮合 / 费用模块）
# ---------------------------------------------------------------------------

STAMP_RULES = (  # (起始日, 税率, 收取边)
    (date(1990, 1, 1), 0.002, "both"),
    (date(2005, 1, 24), 0.001, "both"),
    (date(2007, 5, 30), 0.003, "both"),
    (date(2008, 4, 24), 0.001, "both"),
    (date(2008, 9, 19), 0.001, "sell"),
    (date(2023, 8, 28), 0.0005, "sell"),
)


def _stamp(d: date, side: str) -> float:
    rate, who = 0.001, "sell"
    for dd, rr, ww in STAMP_RULES:
        if d >= dd:
            rate, who = rr, ww
    if who == "both":
        return rate
    return rate if side == "sell" else 0.0


def _transfer(d: date, code: str, amount: float, shares: int) -> float:
    sh = code.endswith(".SH")
    if d < date(2015, 8, 1):
        return max(1.0, shares * 0.001) if sh else 0.0
    if d < date(2022, 4, 29):
        return amount * 0.00002
    return amount * 0.00001


def _commission(amount: float) -> float:
    return max(5.0, amount * 0.00025)


def _slip_price(base: float, side: str, bps: float) -> float:
    raw = base * (1.0 + bps / 1e4) if side == "buy" else base * (1.0 - bps / 1e4)
    if side == "buy":
        return max(0.01, math.ceil(raw * 100 - 1e-9) / 100)
    return max(0.01, math.floor(raw * 100 + 1e-9) / 100)


def fill_price_check(panel: FrozenPanel, res: dict, p: Params) -> dict:
    """逐笔重算成交价：开盘单=开盘±滑点；止损单=min(开盘,止损)±滑点。"""
    fills = res["fills"]
    orders = res["orders"]
    if not len(fills):
        return {"n": 0}
    o = orders.set_index("order_id")
    diffs, fee_diffs, n_stop, n_open = [], [], 0, 0
    bad = []
    for r in fills.itertuples(index=False):
        j = panel.code_idx.get(r.code)
        di = panel.day_idx.get(pd.Timestamp(r.date).date())
        if j is None or di is None:
            continue
        if str(r.kind) == "forced":
            continue
        adv = float(panel.adv20[di - 1, j]) if di > 0 and np.isfinite(panel.adv20[di - 1, j]) else 0.0
        bar_open = float(panel.mat["open"][di, j])
        base = bar_open
        if str(r.kind) == "stop":
            n_stop += 1
            stop_px = float(o.loc[r.order_id, "price"])
            # 与 §8.5 一致：开盘 <= 止损(+tick 容差) 按开盘价；否则按止损价
            base = bar_open if bar_open <= stop_px + 0.005 else stop_px
        else:
            n_open += 1
        part = (r.qty * base / adv) if adv > 0 else 0.0
        bps = min(100.0, p.slippage_bps + 0.02 * math.sqrt(max(part, 0.0)) * 1e4)
        if panel.resume[di, j]:
            bps = min(200.0, bps * 2)
        px = _slip_price(base, r.side, bps)
        diffs.append(abs(px - float(r.price)))
        amt = px * int(r.qty)
        f = _commission(amt) + _stamp(pd.Timestamp(r.date).date(), r.side) * amt \
            + _transfer(pd.Timestamp(r.date).date(), r.code, amt, int(r.qty))
        fee_diffs.append(abs(f - float(r.fees)))
        if abs(px - float(r.price)) > 1e-6 and len(bad) < 10:
            bad.append({"date": str(r.date), "code": r.code, "kind": r.kind, "qty": int(r.qty),
                        "fill": float(r.price), "expected": px, "base": base, "bps": round(bps, 3)})
    return {
        "n_fills": int(len(diffs)), "n_stop": n_stop, "n_open": n_open,
        "max_price_diff": float(np.max(diffs)) if diffs else 0.0,
        "n_price_mismatch_gt_1e-6": int(sum(1 for d in diffs if d > 1e-6)),
        "max_fee_diff": float(np.max(fee_diffs)) if fee_diffs else 0.0,
        "n_fee_mismatch_gt_0.01": int(sum(1 for d in fee_diffs if d > 0.01)),
        "examples": bad,
    }


def independent_ledger(panel: FrozenPanel, res: dict, p: Params) -> dict:
    """独立现金/股数账：从成交流 + 公司行动重算日净值（不调用账户模块）。"""
    eq = res["equity"]
    fills = res["fills"]
    if not len(fills):
        return {"error": "no fills"}
    dates = [d for d in panel.dates
             if pd.Timestamp(d) >= eq.index[0] and pd.Timestamp(d) <= eq.index[-1]]
    fills_by_day: dict[date, list] = {}
    for r in fills.itertuples(index=False):
        fills_by_day.setdefault(pd.Timestamp(r.date).date(), []).append(r)
    cash = float(p.initial_cash)
    pos: dict[str, dict] = {}
    nav_rows = []
    ca_rows = 0
    for d in dates:
        di = panel.day_idx[d]
        # 公司行动（除权日开盘前）
        for ev in panel.events.on(d):
            st = pos.get(ev.symbol)
            if not st or st["shares"] <= 0:
                continue
            s0 = st["shares"]
            prev, pre = float(ev.prev_close), float(ev.preclose)
            bonus = int(math.floor(s0 * ev.bonus_ratio + 1e-9))
            rights = int(math.floor(s0 * ev.rights_ratio + 1e-9))
            new = bonus + rights
            credit = s0 * prev - (s0 + new) * pre
            if rights and cash + credit < 0:
                credit = s0 * max(0.0, ev.dividend)
                new = bonus
            cash += credit
            st["shares"] += new
            div_gross = s0 * max(0.0, float(ev.dividend))
            if div_gross:
                st["div_gross"] += div_gross
            ca_rows += 1
        # 成交
        for r in fills_by_day.get(d, []):
            code = r.code
            qty = int(r.qty)
            amt = float(r.amount)
            fees = float(r.fees)
            st = pos.setdefault(code, {"shares": 0, "entry_date": d, "div_gross": 0.0})
            if r.side == "buy":
                if st["shares"] <= 0:
                    st["entry_date"] = d
                    st["div_gross"] = 0.0
                st["shares"] += qty
                cash -= amt + fees
            else:
                st["shares"] -= qty
                cash += amt - fees
                if st["shares"] <= 0:
                    days = (d - st["entry_date"]).days
                    rate = 0.20 if days <= 30 else (0.10 if days <= 365 else 0.0)
                    tax = st["div_gross"] * rate
                    cash -= tax
                    st.update({"shares": 0, "div_gross": 0.0, "entry_date": d})
        mv = 0.0
        for code, st in pos.items():
            if st["shares"] <= 0:
                continue
            j = panel.code_idx.get(code)
            if j is None:
                continue
            px = float(panel.mat["close"][di, j])
            if not np.isfinite(px) or px <= 0:
                px = 0.0
            mv += st["shares"] * px
        nav_rows.append({"date": d, "nav": cash + mv, "cash": cash, "mv": mv})
    df = pd.DataFrame(nav_rows).set_index("date")
    uni = eq["nav"].astype(float)
    uni.index = [pd.Timestamp(x).date() for x in uni.index]
    df.index = [pd.Timestamp(x).date() for x in df.index]
    common = [d for d in df.index if d in uni.index]
    diffs = np.array([df.loc[d, "nav"] - uni.loc[d] for d in common], dtype=float)
    rel = diffs / np.maximum(np.abs(uni.loc[common].to_numpy()), 1.0)
    return {
        "n_days": len(common),
        "final_nav_independent": float(df["nav"].iloc[-1]),
        "final_nav_unified": float(uni.iloc[-1]),
        "final_diff": float(df["nav"].iloc[-1] - uni.iloc[-1]),
        "max_abs_diff": float(np.max(np.abs(diffs))) if len(diffs) else 0.0,
        "max_rel_diff": float(np.max(np.abs(rel))) if len(rel) else 0.0,
        "n_days_diff_gt_1yuan": int((np.abs(diffs) > 1.0).sum()),
        "n_ca_rows": ca_rows,
    }


def cmd_reconcile(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = load_calendar()
    panel, eb = load_panel()
    p = DEFAULT
    sig = build_signals(panel, p, cal)
    out = {}
    for window in ("dev", "val"):
        res = run_account(panel, sig, p, window, label="T0005")
        fc = fill_price_check(panel, res, p)
        il = independent_ledger(panel, res, p)
        out[window] = {"fill_price_check": fc, "independent_ledger": il}
        log(f"{window} 对账：成交价重算 n={fc.get('n_fills')} "
            f"最大价差 {fc.get('max_price_diff')} 费用差 {fc.get('max_fee_diff')} | "
            f"独立账期末 {il.get('final_nav_independent'):,.2f} vs 统一 {il.get('final_nav_unified'):,.2f} "
            f"max|diff| {il.get('max_abs_diff'):,.4f}")
    t = time.time()
    caus = causality_check(panel, sig, p, cal, n_signal=200, n_other=200)
    (OUT / "08_因果自检.json").write_text(
        json.dumps(_json_safe(caus), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"因果自检：{caus['n_checked']} 个样本，不一致 {caus['n_mismatch']}（{time.time()-t:.1f}s）")
    out["causality"] = {k: v for k, v in caus.items() if k != "mismatches"}
    (OUT / "40_独立账户对账.json").write_text(
        json.dumps(_json_safe(out), ensure_ascii=False, indent=2), encoding="utf-8")




# ---------------------------------------------------------------------------
# 因果自检：截断重算（§6.4）
# ---------------------------------------------------------------------------

def _period_bars_trunc(panel: FrozenPanel, j: int, upto: int, level: str,
                       cal_last: dict) -> list[dict]:
    """只用 date<=panel.dates[upto] 的数据，重建单只股票的周期 K（独立实现）。"""
    dates = panel.dates[:upto + 1]
    freq = _LEVEL_LABEL[level]
    lab = pd.DatetimeIndex(dates).to_period(freq)
    o = panel.mat["ca_open"][:upto + 1, j]
    h = panel.mat["ca_high"][:upto + 1, j]
    lo = panel.mat["ca_low"][:upto + 1, j]
    c = panel.mat["ca_close"][:upto + 1, j]
    bars = []
    start = 0
    for k in range(1, len(dates) + 1):
        if k == len(dates) or lab[k] != lab[start]:
            seg = slice(start, k)
            hh = h[seg]
            if np.isfinite(hh).any():
                finite_c = np.isfinite(c[seg])
                last_c = np.flatnonzero(finite_c)
                if len(last_c):
                    first_o = np.flatnonzero(np.isfinite(o[seg]))
                    bars.append({
                        "end": dates[k - 1],
                        "o": float(o[seg][first_o[0]]) if len(first_o) else np.nan,
                        "h": float(np.nanmax(hh)),
                        "l": float(np.nanmin(lo[seg])),
                        "c": float(c[seg][last_c[-1]]),
                        "complete": cal_last.get(lab[start], dates[k - 1]) <= dates[upto],
                    })
            start = k
    return bars


def recompute_one(panel: FrozenPanel, p: Params, cal_last: dict, j: int, i: int) -> dict:
    """独立重算 (code j, day i) 的 S1/E2 判定，只用 <= i 的数据。"""
    out = {"signal": False, "detail": {}}
    if not bool(panel.elig[i, j]):
        out["detail"]["elig"] = False
        return out
    layers = {}
    for lv in (p.structure, p.framework):
        name = lv
        bars = [b for b in _period_bars_trunc(panel, j, i, lv, cal_last[lv]) if b["complete"]]
        n = len(bars)
        hi = np.array([b["h"] for b in bars], dtype=float)
        low = np.array([b["l"] for b in bars], dtype=float)
        lvl = {"h": [], "l": []}
        kk = p.k
        for idx in range(kk, n - kk):
            ok_h = all(hi[idx] > hi[idx - m] and hi[idx] > hi[idx + m] for m in range(1, kk + 1))
            ok_l = all(low[idx] < low[idx - m] and low[idx] < low[idx + m] for m in range(1, kk + 1))
            if ok_h:
                lvl["h"].append((bars[idx + kk]["end"], float(hi[idx])))
            if ok_l:
                lvl["l"].append((bars[idx + kk]["end"], float(low[idx])))
        if len(lvl["h"]) < 2 or len(lvl["l"]) < 2:
            layers[name] = {"s1": False, "L": np.nan, "low": np.nan, "confirm": None}
            continue
        (d2h, h2), (d1h, h1) = lvl["h"][-2], lvl["h"][-1]
        (d2l, l2), (d1l, l1) = lvl["l"][-2], lvl["l"][-1]
        layers[name] = {"s1": bool(h1 > h2 and l1 > l2), "L": h1, "low": l1,
                        "confirm": d1h, "prev_h": h2, "prev_l": l2}
    st = layers[p.structure]
    fr = layers[p.framework]
    out["detail"].update({"s1_struct": st["s1"], "s1_frame": fr["s1"]})
    if not (st["s1"] and fr["s1"]):
        return out
    L, low, conf = st["L"], st["low"], st["confirm"]
    if conf is None or not np.isfinite(L) or not np.isfinite(low):
        return out
    conf_idx = int(np.searchsorted(np.array(panel.dates), conf, side="left"))
    atr = panel.atr14[:, j]
    close = panel.mat["ca_close"][:, j]
    lows = panel.mat["ca_low"][:, j]
    b = None
    for t in range(conf_idx, i):
        if not np.isfinite(atr[t]) or not np.isfinite(close[t]):
            continue
        if close[t] > L + p.buffer_atr * atr[t]:
            b = t
            break
    out["detail"]["b"] = b
    if b is None or b > i - 1 or (i - b) > p.retest_days:
        return out
    for t in range(b, i + 1):
        if not (np.isfinite(atr[t]) and np.isfinite(close[t])):
            return out
        if close[t] < L - p.band_atr * atr[t]:
            return out
    touched = any(np.isfinite(lows[t]) and np.isfinite(atr[t])
                  and lows[t] <= L + p.band_atr * atr[t] for t in range(b, i))
    if not touched:
        return out
    if not (np.isfinite(close[i]) and np.isfinite(close[i - 1]) and np.isfinite(close[i - 2])):
        return out
    if not (close[i] > close[i - 1] and close[i] > close[i - 2]):
        return out
    out["signal"] = True
    out["detail"].update({"L": float(L), "low": float(low), "b": int(b)})
    return out


def causality_check(panel: FrozenPanel, sig: Signals, p: Params,
                    cal_dates: pd.DatetimeIndex, n_signal: int = 200,
                    n_other: int = 200, seed: int = SEED) -> dict:
    cal_last = {}
    for lv in {p.structure, p.framework}:
        lab = cal_dates.to_period(_LEVEL_LABEL[lv])
        ser = pd.Series(cal_dates, index=lab).groupby(level=0).max()
        cal_last[lv] = {k: v.date() for k, v in ser.to_dict().items()}
    rng = np.random.default_rng(seed)
    dev_lo = panel.window_index(*WINDOWS["dev"])[0]
    dev_hi = panel.window_index(*WINDOWS["dev"])[1]
    val_lo = panel.window_index(*WINDOWS["val"])[0]
    val_hi = panel.window_index(*WINDOWS["val"])[1]
    sig_idx = np.argwhere(sig.mask)
    sig_idx = sig_idx[(sig_idx[:, 0] >= dev_lo) & (sig_idx[:, 0] <= val_hi)]
    pick_sig = rng.choice(len(sig_idx), size=min(n_signal, len(sig_idx)), replace=False)
    # 非信号日：从同日期范围的合格 code-day 里抽
    elig_idx = np.argwhere(panel.elig)
    elig_idx = elig_idx[(elig_idx[:, 0] >= dev_lo) & (elig_idx[:, 0] <= val_hi)]
    pick_oth = rng.choice(len(elig_idx), size=min(n_other, len(elig_idx)), replace=False)
    samples = [tuple(sig_idx[k]) for k in pick_sig] + [tuple(elig_idx[k]) for k in pick_oth]
    # 窗口边界日
    for i in (dev_lo, dev_hi, val_lo, val_hi):
        for j in np.flatnonzero(panel.elig[i])[:20]:
            samples.append((i, int(j)))
    mism = []
    n_checked = 0
    for i, j in samples:
        got = bool(sig.mask[i, j])
        exp = recompute_one(panel, p, cal_last, int(j), int(i))
        n_checked += 1
        if got != exp["signal"]:
            mism.append({"date": panel.dates[i].isoformat(), "code": panel.codes[j],
                         "vectorized": got, "recomputed": exp["signal"], "detail": exp["detail"]})
            if len(mism) >= 20:
                break
    return {
        "n_checked": n_checked, "n_signal_samples": int(min(n_signal, len(sig_idx))),
        "n_other_samples": int(min(n_other, len(elig_idx))),
        "n_mismatch": len(mism), "mismatches": mism,
        "pass": len(mism) == 0,
    }




# ---------------------------------------------------------------------------
# 消融与参数邻域
# ---------------------------------------------------------------------------

def _ablation_configs() -> list[tuple[str, str, Params]]:
    D = DEFAULT
    out = [
        ("V1", "F1→F2 更长周期", replace(D, framework="quarter", structure="month",
                                        trigger="week", tag="V1")),
        ("V2", "F1→F3 去掉高周期", replace(D, framework="week", structure="day",
                                          trigger="day", tag="V2")),
        ("V3", "S1→S2 结构vs均线", replace(D, structure_mode="S2", framework_mode="S2", tag="V3")),
        ("V4", "E2→E1 追突破", replace(D, entry_mode="E1", tag="V4")),
        ("V5", "E2→E3 趋势内回调", replace(D, entry_mode="E3", tag="V5")),
        ("V6", "X3→X2 去谨慎态", replace(D, use_c1=False, use_c3=False, tag="V6")),
        ("V7", "X3→X1 固定2R目标", replace(D, use_c1=False, use_c3=False,
                                          use_be_trail=False, use_target=True, tag="V7")),
        ("V8", "G1→G2 市场门", replace(D, use_g2=True, tag="V8")),
        ("A1", "E2+固定20日", replace(D, fixed_hold=20, use_structure_stop=False,
                                    use_be_trail=False, use_3day=False, use_10day=False,
                                    use_time_stop=False, use_c1=False, use_c3=False, tag="A1")),
        ("A2", "A1+结构止损", replace(D, fixed_hold=20, use_structure_stop=True,
                                    use_be_trail=False, use_3day=False, use_10day=False,
                                    use_time_stop=False, use_c1=False, use_c3=False, tag="A2")),
        ("A3", "A2+保本+跟踪(=X2)", replace(D, use_c1=False, use_c3=False, tag="A3")),
        ("A4", "A3+市场门", replace(D, use_c1=False, use_c3=False, use_g2=True, tag="A4")),
        ("A5", "A3+谨慎态(=X3)", replace(D, tag="A5")),
        ("A6", "A5 去掉3日失败", replace(D, use_3day=False, tag="A6")),
        ("A7", "全状态机(=主候选)", replace(D, tag="A7")),
    ]
    return out


def _neighbor_configs() -> list[tuple[str, str, Params]]:
    D = DEFAULT
    return [
        ("S01", "k=3", replace(D, k=3, tag="S01")),
        ("S02", "ATR20", replace(D, atr_n=20, tag="S02")),
        ("S03", "缓冲0.00", replace(D, buffer_atr=0.0, tag="S03")),
        ("S04", "缓冲0.10", replace(D, buffer_atr=0.10, tag="S04")),
        ("S05", "止损0.5ATR", replace(D, stop_atr=0.5, tag="S05")),
        ("S06", "最大初始风险10%", replace(D, max_risk_pct=0.10, tag="S06")),
        ("S07", "保本1.5R", replace(D, be_mfe_r=1.5, tag="S07")),
        ("S08", "跟踪触发1.5R", replace(D, trail_mfe_r=1.5, tag="S08")),
        ("S09", "跟踪触发2.5R", replace(D, trail_mfe_r=2.5, tag="S09")),
        ("S10", "跟踪宽度2.5ATR", replace(D, trail_atr=2.5, tag="S10")),
        ("S11", "跟踪宽度4.0ATR", replace(D, trail_atr=4.0, tag="S11")),
        ("S12", "去掉10日减半", replace(D, use_10day=False, tag="S12")),
        ("S13", "20日时间止损→30日", replace(D, time_stop_days=30, tag="S13")),
        ("S14", "组合回撤8/12", replace(D, port_dd_half=0.08, port_dd_stop=0.12, tag="S14")),
        ("S15", "单笔风险1.0%", replace(D, risk_per_trade=0.010, tag="S15")),
        ("S16", "最大持仓5", replace(D, max_pos=5, tag="S16")),
        ("S17", "最大持仓20", replace(D, max_pos=20, tag="S17")),
        ("S18", "排序=60日收益", replace(D, rank="ret60", tag="S18")),
    ]


def _run_configs(panel: FrozenPanel, cal_dates: pd.DatetimeIndex,
                 configs: list[tuple[str, str, Params]], out_csv: Path) -> pd.DataFrame:
    rows = []
    navs: dict[str, pd.Series] = {}
    for code, desc, p in configs:
        for window in ("dev", "val"):
            t = time.time()
            try:
                sig = build_signals(panel, p, cal_dates)
                res = run_account(panel, sig, p, window, label=code)
                i0, i1 = res["i0"], res["i1"]
                bench_nav, hs = window_benchmark(panel, i0, i1)
                m = metrics_bundle(res, panel, p, bench_nav, hs)
                sc = selfcheck(res, panel, sig, p)
                rows.append({
                    "id": code, "desc": desc, "window": window,
                    "family": p.family(),
                    "cagr": m.get("cagr"), "mdd": m.get("mdd"), "calmar": m.get("calmar"),
                    "sharpe": m.get("sharpe"), "n_trades": m.get("n_trades"),
                    "n_codes": m.get("n_codes"), "win_rate": m.get("win_rate"),
                    "avg_hold_days": m.get("avg_hold_days"), "turnover": m.get("turnover"),
                    "exposure": m.get("exposure"), "avg_positions": m.get("avg_positions"),
                    "cost_paid": m.get("cost_paid"), "cagr_ex_top5": m.get("cagr_ex_top5"),
                    "final_nav": m.get("final_nav"),
                    "benchmark_cagr": m.get("benchmark_cagr"),
                    "bootstrap_cagr_lo": (m.get("bootstrap") or {}).get("cagr_lo"),
                    "bootstrap_cagr_median": (m.get("bootstrap") or {}).get("cagr_median"),
                    "bootstrap_cagr_hi": (m.get("bootstrap") or {}).get("cagr_hi"),
                    "n_signals": int(sig.mask[slice(*panel.window_index(*WINDOWS[window]))].sum()),
                    "t1_violations": sc.get("t1_violations"),
                    "risk_violations": sc.get("risk_violations"),
                    "secs": round(time.time() - t, 1),
                })
                navs[f"{code}|{window}"] = res["equity"]["nav"].rename(f"{code}")
                log(f"{code} {window}: cagr {m.get('cagr', float('nan'))*100:6.2f}% "
                    f"mdd {m.get('mdd', float('nan'))*100:7.2f}% 交易 {m.get('n_trades'):4d} "
                    f"({time.time()-t:.1f}s)")
            except Exception as exc:
                traceback.print_exc()
                rows.append({"id": code, "desc": desc, "window": window,
                             "family": p.family(), "error": str(exc)})
            finally:
                try:
                    del sig
                except Exception:
                    pass
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    if navs:
        pd.DataFrame(navs).to_csv(out_csv.with_name("_navs_" + out_csv.stem + ".csv"),
                                  encoding="utf-8-sig")
    return df


def cmd_ablation(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = load_calendar()
    panel, eb = load_panel()
    df = _run_configs(panel, cal, _ablation_configs(), OUT / "20_消融.csv")
    print(df.to_string(index=False))


def cmd_neighbor(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = load_calendar()
    panel, eb = load_panel()
    df = _run_configs(panel, cal, _neighbor_configs(), OUT / "30_邻域.csv")
    print(df.to_string(index=False))




# ---------------------------------------------------------------------------
# 多重性：DSR / PBO
# ---------------------------------------------------------------------------

def _daily_stats(nav: pd.Series) -> dict:
    r = nav.astype(float).pct_change().fillna(0.0).to_numpy()
    sd = float(np.std(r, ddof=1)) if len(r) > 2 else np.nan
    mu = float(np.mean(r))
    return {
        "n": len(r), "mean": mu, "sd": sd,
        "sr_daily": mu / sd if sd and sd > 0 else np.nan,
        "skew": float(pd.Series(r).skew()),
        "kurt": float(pd.Series(r).kurtosis() + 3.0),
    }


def deflated_sharpe(sr_daily: float, T: int, n_trials: int, var_sr: float,
                    skew: float, kurt: float) -> dict:
    from scipy.stats import norm
    gamma = 0.5772156649
    if not (np.isfinite(sr_daily) and np.isfinite(var_sr) and var_sr > 0 and n_trials > 1):
        return {"dsr": None}
    sr0 = math.sqrt(var_sr) * ((1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
                               + gamma * norm.ppf(1 - 1.0 / (n_trials * math.e)))
    denom = math.sqrt(max(1e-12, 1 - skew * sr_daily + (kurt - 1) / 4.0 * sr_daily ** 2))
    z = (sr_daily - sr0) * math.sqrt(max(T - 1, 1)) / denom
    return {"dsr": float(norm.cdf(z)), "z": float(z), "sr0_daily": float(sr0),
            "sr0_annual": float(sr0 * math.sqrt(243)), "sr_daily": float(sr_daily),
            "n_trials": int(n_trials), "var_sr": float(var_sr)}


def pbo_cscv(navs: pd.DataFrame, S: int = 16) -> dict:
    """CSCV/PBO：把日收益切成 S 个连续块，枚举 C(S,S/2) 种 IS/OOS 组合。"""
    import itertools
    r = navs.astype(float).pct_change().fillna(0.0)
    r = r.loc[:, r.notna().all() if False else (r.std() > 0)]
    T, N = r.shape
    if N < 2 or T < S * 2:
        return {"error": f"样本不足 T={T} N={N}"}
    arr = r.to_numpy()
    edges = np.linspace(0, T, S + 1).astype(int)
    n = np.array([edges[s + 1] - edges[s] for s in range(S)], dtype=float)
    s1 = np.array([arr[edges[s]:edges[s + 1]].sum(axis=0) for s in range(S)])
    s2 = np.array([(arr[edges[s]:edges[s + 1]] ** 2).sum(axis=0) for s in range(S)])
    combos = list(itertools.combinations(range(S), S // 2))
    ranks, logits = [], []
    for comb in combos:
        oos = [s for s in range(S) if s not in comb]
        ni = n[list(comb)].sum()
        si = s1[list(comb)].sum(axis=0)
        qi = s2[list(comb)].sum(axis=0)
        sdi = np.sqrt(np.maximum((qi - si ** 2 / ni) / max(ni - 1, 1), 1e-18))
        sr_is = (si / ni) / sdi
        best = int(np.nanargmax(sr_is))
        no = n[oos].sum()
        so = s1[oos].sum(axis=0)
        qo = s2[oos].sum(axis=0)
        sdo = np.sqrt(np.maximum((qo - so ** 2 / no) / max(no - 1, 1), 1e-18))
        sr_oos = (so / no) / sdo
        order = np.argsort(np.argsort(sr_oos))
        rank = float(order[best]) / max(N - 1, 1)
        ranks.append(rank)
        logits.append(math.log(min(max(rank, 1e-6), 1 - 1e-6) / (1 - min(max(rank, 1e-6), 1 - 1e-6))))
    ranks = np.array(ranks)
    return {
        "n_comb": len(combos), "n_configs": int(N), "n_days": int(T),
        "pbo": float(np.mean(ranks < 0.5)), "median_rank": float(np.median(ranks)),
        "logit_mean": float(np.mean(logits)), "logit_std": float(np.std(logits, ddof=1)),
    }




# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def _fmt_pct(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x*100:,.{d}f}%"


def risk_gate_table(panel: FrozenPanel, sig: Signals, p: Params) -> pd.DataFrame:
    rows = []
    for window in ("dev", "val"):
        i0, i1 = panel.window_index(*WINDOWS[window])
        n_sig = 0
        pass_n = 0
        rs = []
        by_year: dict[int, list] = {}
        for i in range(i0, i1 + 1):
            js = np.flatnonzero(sig.mask[i])
            if not len(js):
                continue
            c = sig.close[i, js].astype(float)
            lo = sig.low[i, js].astype(float)
            atr = sig.atr[i, js].astype(float)
            ok = np.isfinite(c) & np.isfinite(lo) & np.isfinite(atr) & (c > 0) & (atr > 0)
            if not ok.any():
                continue
            r = (c[ok] - (lo[ok] - p.stop_atr * atr[ok])) / c[ok]
            n_sig += int(len(r))
            pass_n += int((r <= p.max_risk_pct).sum())
            rs.extend(r.tolist())
            by_year.setdefault(panel.dates[i].year, []).extend(r.tolist())
        rs = np.array(rs)
        rows.append({
            "window": window, "n_signals": n_sig,
            "pass_risk_gate": pass_n,
            "pass_share": round(pass_n / max(n_sig, 1), 4),
            "r_p10": round(float(np.quantile(rs, 0.10)), 4) if len(rs) else None,
            "r_median": round(float(np.median(rs)), 4) if len(rs) else None,
            "r_p90": round(float(np.quantile(rs, 0.90)), 4) if len(rs) else None,
        })
        for y, vals in sorted(by_year.items()):
            arr = np.array(vals)
            rows.append({
                "window": f"{window}:{y}", "n_signals": int(len(arr)),
                "pass_risk_gate": int((arr <= p.max_risk_pct).sum()),
                "pass_share": round(float((arr <= p.max_risk_pct).mean()), 4),
                "r_p10": round(float(np.quantile(arr, 0.10)), 4),
                "r_median": round(float(np.median(arr)), 4),
                "r_p90": round(float(np.quantile(arr, 0.90)), 4),
            })
    return pd.DataFrame(rows)


def scan_2021(out_dir: Path) -> dict:
    bad = []
    n = 0
    for f in sorted(out_dir.glob("*.csv")):
        n += 1
        try:
            df = pd.read_csv(f, nrows=20000)
        except Exception:
            continue
        for col in df.columns:
            if "date" in str(col).lower():
                s = pd.to_datetime(df[col], errors="coerce")
                if (s >= pd.Timestamp("2021-01-01")).any():
                    bad.append({"file": f.name, "column": str(col)})
    return {"n_files": n, "files_with_2021": bad, "pass": not bad}


def cmd_report(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main = json.loads((OUT / "07_统计_主候选.json").read_text(encoding="utf-8"))
    dev, val = main["dev"], main["val"]
    md, mv = dev["metrics"], val["metrics"]
    sd, sv = dev["selfcheck"], val["selfcheck"]
    rec = json.loads((OUT / "40_独立账户对账.json").read_text(encoding="utf-8")) \
        if (OUT / "40_独立账户对账.json").exists() else {}
    abl = pd.read_csv(OUT / "20_消融.csv") if (OUT / "20_消融.csv").exists() else pd.DataFrame()
    nei = pd.read_csv(OUT / "30_邻域.csv") if (OUT / "30_邻域.csv").exists() else pd.DataFrame()
    ann_dev = pd.read_csv(OUT / "04_年度_T0005_dev.csv")
    ann_val = pd.read_csv(OUT / "04_年度_T0005_val.csv")
    tr_dev = pd.read_csv(OUT / "02_逐笔_T0005_dev.csv")
    tr_val = pd.read_csv(OUT / "02_逐笔_T0005_val.csv")

    # 门槛（§7.9.5 可执行子集 + §12.1）
    def _g(m, key, thr):
        v = m.get(key)
        return (v is not None) and np.isfinite(v) and (v >= thr)
    gates = {
        "dev_cagr_gt0": _g(md, "cagr", 0.0),
        "val_cagr_gt0": _g(mv, "cagr", 0.0),
        "dev_mdd_le25": _g(md, "mdd", -0.25),
        "val_mdd_le25": _g(mv, "mdd", -0.25),
        "dev_trades_ge30": (md.get("n_trades") or 0) >= 30,
        "val_trades_ge30": (mv.get("n_trades") or 0) >= 30,
        "dev_codes_ge10": (md.get("n_codes") or 0) >= 10,
        "val_codes_ge10": (mv.get("n_codes") or 0) >= 10,
    }
    gates["selection_pass"] = all(gates.values())
    score = (min(md["cagr"], mv["cagr"]) / max(abs(md["mdd"]), abs(mv["mdd"]))
             if md.get("mdd") and mv.get("mdd") else float("nan"))

    # DSR / PBO
    eq_dev = pd.read_csv(OUT / "03_日净值_T0005_dev.csv", index_col=0, parse_dates=True)
    eq_val = pd.read_csv(OUT / "03_日净值_T0005_val.csv", index_col=0, parse_dates=True)
    nav_dev, nav_val = eq_dev["nav"], eq_val["nav"]
    cash_share = {"dev": float((eq_dev["cash"] / eq_dev["nav"]).mean()),
                  "val": float((eq_val["cash"] / eq_val["nav"]).mean())}
    st_dev, st_val = _daily_stats(nav_dev), _daily_stats(nav_val)
    proj = json.loads((ROOT / "03_结果/统计整合/多重性.json").read_text(encoding="utf-8"))["project"]
    var_proj = float(proj["var_sr_daily"])
    # 去重后的新增配置：主候选 1 + V1-V8 8 + A1-A4/A6 5 + 邻域 18 = 32（A5/A7 与主候选同一配置）
    n_new = 1 + len(abl) // 2 + len(nei) // 2
    if abl is not None and len(abl):
        n_new -= 2
    multi = {
        "n_new_configs": int(n_new),
        "n_project_sel": int(proj["n_sel"]),
        "n_total_for_dsr": int(proj["n_sel"] + n_new),
        "var_sr_project_daily": var_proj,
        "dev": deflated_sharpe(st_dev["sr_daily"], st_dev["n"], int(proj["n_sel"] + n_new),
                               var_proj, st_dev["skew"], st_dev["kurt"]),
        "val": deflated_sharpe(st_val["sr_daily"], st_val["n"], int(proj["n_sel"] + n_new),
                               var_proj, st_val["skew"], st_val["kurt"]),
        "dev_daily_stats": st_dev, "val_daily_stats": st_val,
    }
    # 族内 Var(SR)（用消融+邻域配置的年化 sharpe 反算日频）
    if len(nei):
        sh = nei[nei["window"] == "dev"]["sharpe"].dropna().to_numpy(dtype=float) / math.sqrt(243)
        if len(sh) > 2:
            multi["var_sr_family_daily"] = float(np.var(sh, ddof=1))
            multi["dev_family_dsr"] = deflated_sharpe(
                st_dev["sr_daily"], st_dev["n"], int(len(sh)),
                float(np.var(sh, ddof=1)), st_dev["skew"], st_dev["kurt"])
    # PBO（邻域配置，分窗口）
    pbo = {}
    navf = OUT / "_navs_30_邻域.csv"
    if navf.exists():
        navs = pd.read_csv(navf, index_col=0, parse_dates=True)
        for w in ("dev", "val"):
            cols = [c for c in navs.columns if c.endswith(f"|{w}")]
            if len(cols) >= 5:
                pbo[w] = pbo_cscv(navs[cols].dropna(how="all"))
    multi["pbo_neighbor"] = pbo
    (OUT / "41_多重性.json").write_text(json.dumps(_json_safe(multi), ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    # 试验台账（本批新增；官方注册表由方法负责人独占，本文件只供其登记）
    ledger_rows = []
    for w in ("dev", "val"):
        m = md if w == "dev" else mv
        ledger_rows.append({"trial": "T0005", "kind": "experiment", "stage": w,
                            "family": DEFAULT.family(), "desc": "主候选 F1-S1-E2-X3-G1",
                            "cagr": m.get("cagr"), "mdd": m.get("mdd"),
                            "n_trades": m.get("n_trades"), "status": "done"})
    if len(abl):
        for r in abl.to_dict("records"):
            ledger_rows.append({"trial": f"T0005-{r['id']}", "kind": "ablation", "stage": r["window"],
                                "family": r.get("family"), "desc": r.get("desc"),
                                "cagr": r.get("cagr"), "mdd": r.get("mdd"),
                                "n_trades": r.get("n_trades"), "status": "done"})
    if len(nei):
        for r in nei.to_dict("records"):
            ledger_rows.append({"trial": f"T0005-{r['id']}", "kind": "sensitivity", "stage": r["window"],
                                "family": r.get("family"), "desc": r.get("desc"),
                                "cagr": r.get("cagr"), "mdd": r.get("mdd"),
                                "n_trades": r.get("n_trades"), "status": "done"})
    pd.DataFrame(ledger_rows).to_csv(OUT / "60_试验台账.csv", index=False, encoding="utf-8-sig")

    scan = scan_2021(OUT)

    # 风险门表
    cal = load_calendar()
    panel, eb = load_panel()
    sig = build_signals(panel, DEFAULT, cal)
    rg = risk_gate_table(panel, sig, DEFAULT)
    rg.to_csv(OUT / "10_信号_风险门.csv", index=False, encoding="utf-8-sig")

    # 退出原因
    def _reasons(tr: pd.DataFrame) -> pd.DataFrame:
        if not len(tr):
            return pd.DataFrame(columns=["reason", "n", "pnl_sum", "pnl_mean", "win_rate"])
        g = tr.groupby("reason").agg(n=("pnl", "size"), pnl_sum=("pnl", "sum"),
                                     pnl_mean=("pnl", "mean"),
                                     win_rate=("pnl", lambda s: float((s > 0).mean())))
        return g.reset_index().sort_values("n", ascending=False)
    reas = pd.concat([_reasons(tr_dev).assign(window="dev"),
                      _reasons(tr_val).assign(window="val")], ignore_index=True)
    reas.to_csv(OUT / "11_退出原因.csv", index=False, encoding="utf-8-sig")

    lines = []
    A = lines.append
    A("# T0005 主候选执行报告：F1-S1-E2-X3-G1（dev 2008-2015 / val 2016-2020）")
    A("")
    A("- 执行：主候选执行员（`02_代码/主候选_T0005.py`）；数据 `v2-adjspace-2026-10-28`（冻结 2026-10-26）")
    A("- 规范：`00_研究预注册.md` §7.6 主候选 + 注册表 T0005 `planned` 参数；缺口与操作化见 `00_歧义与操作化记录.md`")
    A("- 纪律：**2021+ 未读取、未计算、未消耗留出**；本批只跑 dev/val")
    A("")
    A("## 0. 结论先行")
    A("")
    A(f"1. **T0005 在验证期失败，主候选淘汰。** dev 净年化 **{_fmt_pct(md['cagr'])}**（回撤 {_fmt_pct(md['mdd'])}），"
      f"val 净年化 **{_fmt_pct(mv['cagr'])}**（回撤 {_fmt_pct(mv['mdd'])}）；§7.9.5 要求两段净收益都 > 0，val 不满足。")
    A(f"2. **年化 60% 被拒绝**：dev/val 较低者 {_fmt_pct(min(md['cagr'], mv['cagr']))}，"
      f"按 §12.2 属“无可利用边际”（< 20%）。不申请留出。")
    A(f"3. **回撤门槛过了、收益门槛没过**：两段最大回撤 {_fmt_pct(md['mdd'])} / {_fmt_pct(mv['mdd'])} 均 ≤ 25%，"
      f"但这是**低暴露**换来的：dev 平均持仓 {md['avg_positions']:.2f} 只、在场时间 {_fmt_pct(md['time_in_market'],1)}，"
      f"val 平均持仓 {mv['avg_positions']:.2f} 只、在场时间 {_fmt_pct(mv['time_in_market'],1)}。")
    _rgd = rg[rg["window"] == "dev"].iloc[0]
    _rgv = rg[rg["window"] == "val"].iloc[0]
    A(f"4. **主要约束是预注册自身的 12% 初始风险门**：{int(_rgd['n_signals']):,}（dev）/ "
      f"{int(_rgv['n_signals']):,}（val）个 E2 信号中，只有 {_fmt_pct(_rgd['pass_share'],2)} / "
      f"{_fmt_pct(_rgv['pass_share'],2)} 的 R/P0 ≤ 12%（中位 R/P0 = "
      f"{_fmt_pct(_rgd['r_median'],1)} / {_fmt_pct(_rgv['r_median'],1)}）。"
      "周线结构低点离现价太远，止损距离天然超限。")
    _abl_v = abl[abl["window"] == "val"] if len(abl) else pd.DataFrame()
    _av = _abl_v[_abl_v["n_trades"] > 0]["cagr"].dropna() if len(_abl_v) else pd.Series(dtype=float)
    _nv = nei[nei["window"] == "val"]["cagr"].dropna() if len(nei) else pd.Series(dtype=float)
    _zero = _abl_v[_abl_v["n_trades"] == 0]["id"].tolist() if len(_abl_v) else []
    A(f"5. **没有任何变体在 val 转正**：消融 {len(_av)} 个有成交配置 val 年化 {_fmt_pct(_av.min())} ~ {_fmt_pct(_av.max())}"
      + (f"（{', '.join(_zero)} 为 0 笔，不参与区间）" if _zero else "")
      + f"；参数邻域 {len(_nv)} 个配置 {_fmt_pct(_nv.min())} ~ {_fmt_pct(_nv.max())}。"
      "最好的一格是 A1（只进场+固定20日持有，val −0.11%），仍为负。")
    A("6. **负结果不是数据/引擎问题**：v2 冻结数据、统一账户 + 独立账户逐日净值差 0.0000 元、"
      "成交价独立重算 619 笔中仅 2 笔差 1 分（已解释），T+1/停牌/涨跌停/2021 守卫 0 违规。")
    A("")
    A("## 1. 主候选结果")
    A("")
    A("| 指标 | dev 2008-2015 | val 2016-2020 | 门槛 |")
    A("|---|---:|---:|---|")
    for label, key, fmt in [
        ("净年化", "cagr", _fmt_pct), ("最大回撤", "mdd", _fmt_pct),
        ("Calmar", "calmar", lambda x: f"{x:.2f}" if x is not None and np.isfinite(x) else "—"),
        ("夏普", "sharpe", lambda x: f"{x:.2f}" if x is not None and np.isfinite(x) else "—"),
        ("年化波动", "vol", _fmt_pct), ("Ulcer", "ulcer", lambda x: f"{x:.4f}"),
        ("最长水下（交易日）", "longest_underwater", lambda x: f"{int(x)}"),
        ("回合数", "n_trades", lambda x: f"{int(x)}"), ("股票数", "n_codes", lambda x: f"{int(x)}"),
        ("胜率", "win_rate", _fmt_pct), ("盈亏比(利润因子)", "profit_factor", lambda x: f"{x:.2f}"),
        ("单笔均值", "avg_trade_ret", _fmt_pct), ("平均持仓天数", "avg_hold_days", lambda x: f"{x:.1f}"),
        ("换手/年", "turnover", lambda x: f"{x:.2f}"), ("平均持仓数", "avg_positions", lambda x: f"{x:.2f}"),
        ("在场时间", "time_in_market", _fmt_pct), ("累计成本", "cost_paid", lambda x: f"{x:,.0f}"),
        ("去前5大盈利后年化", "cagr_ex_top5", _fmt_pct),
        ("B1-EW 年化", "benchmark_cagr", _fmt_pct), ("超额年化", "excess_cagr", _fmt_pct),
        ("beta", "beta", lambda x: f"{x:.2f}"),
    ]:
        v1, v2 = md.get(key), mv.get(key)
        thr = ""
        if key == "cagr": thr = "两段 > 0"
        if key == "mdd": thr = "≥ -25%"
        if key in ("n_trades",): thr = "≥ 30"
        if key in ("n_codes",): thr = "≥ 10"
        A(f"| {label} | {fmt(v1) if v1 is not None else '—'} | {fmt(v2) if v2 is not None else '—'} | {thr} |")
    A("")
    bs_d, bs_v = md.get("bootstrap") or {}, mv.get("bootstrap") or {}
    A(f"平均现金占比：dev **{_fmt_pct(cash_share['dev'],1)}**、val **{_fmt_pct(cash_share['val'],1)}**。"
      "也就是说，主候选是一个 90% 左右现金 + 1-2 只股票的小卫星组合；"
      "回撤小是现金带来的，不是择时或止损带来的。")
    A("")
    A(f"区块 bootstrap（stationary，块长20，B=2000）：dev 年化 2.5%/50%/97.5% = "
      f"{_fmt_pct(bs_d.get('cagr_lo'))} / {_fmt_pct(bs_d.get('cagr_median'))} / {_fmt_pct(bs_d.get('cagr_hi'))}；"
      f"val = {_fmt_pct(bs_v.get('cagr_lo'))} / {_fmt_pct(bs_v.get('cagr_median'))} / {_fmt_pct(bs_v.get('cagr_hi'))}。"
      f"峰值/谷值回撤 dev [{_fmt_pct(bs_d.get('mdd_lo'))}, {_fmt_pct(bs_d.get('mdd_hi'))}]、"
      f"val [{_fmt_pct(bs_v.get('mdd_lo'))}, {_fmt_pct(bs_v.get('mdd_hi'))}]。")
    A(f"读法：val 点估计 {_fmt_pct(mv['cagr'])}、97.5% 上界仅 {_fmt_pct(bs_v.get('cagr_hi'))}，"
      "即“没有正边际”与“仅有一个可忽略的正边际”都在区间内；按 §7.9.5 用点估计判定，val 不通过。")
    A("")
    A("## 2. 门槛判定")
    A("")
    A("| 门槛 | 结果 |")
    A("|---|---|")
    for k, v in gates.items():
        A(f"| {k} | {'PASS' if v else 'FAIL'} |")
    A(f"| 排序分数 min(年化)/max(回撤) | {score:.3f}（若两段都为正才用于选择） |")
    A("")
    A("§12.1 的 G1（留出净年化 ≥ 60% 且 walk-forward ≥ 60%）**不适用**：dev/val 已失败，按 §10.3 不运行留出。")
    A("")
    A("## 3. 年度与退出结构")
    A("")
    A("| 年 | dev 收益 | dev 回撤 | val 收益 | val 回撤 |")
    A("|---|---:|---:|---:|---:|")
    yd = {int(r["year"]): r for r in ann_dev.to_dict("records")}
    yv = {int(r["year"]): r for r in ann_val.to_dict("records")}
    for y in sorted(set(yd) | set(yv)):
        a, v = yd.get(y), yv.get(y)
        A(f"| {y} | {_fmt_pct(a['return']) if a else ''} | {_fmt_pct(a['mdd']) if a else ''} | "
          f"{_fmt_pct(v['return']) if v else ''} | {_fmt_pct(v['mdd']) if v else ''} |")
    A("")
    _ad = ann_dev.set_index("year")["return"]
    _best = _ad.sort_values(ascending=False).head(2)
    A(f"dev 的 {_fmt_pct(md['cagr'])} 年化几乎全部来自 "
      + "、".join(f"{int(y)} 年 {_fmt_pct(v)}" for y, v in _best.items())
      + f"；其余 {len(_ad)-len(_best)} 年平均 {_fmt_pct(float(_ad.drop(_best.index).mean()))}，"
      f"其中 2008-2009 年 0 笔交易（风险门全拒）。验证期 5 年逐年为负。")
    A("")
    A("退出原因（笔数 / 合计盈亏 / 单笔均值 / 胜率）：")
    A("")
    A("| 窗口 | 原因 | n | 合计盈亏 | 单笔均值 | 胜率 |")
    A("|---|---|---:|---:|---:|---:|")
    for r in reas.to_dict("records"):
        A(f"| {r['window']} | {r['reason']} | {int(r['n'])} | {r['pnl_sum']:,.0f} | "
          f"{r['pnl_mean']:,.0f} | {_fmt_pct(r['win_rate'],1)} |")
    A("")
    A("## 4. 单层消融（§7.7 V1-V8 + A1-A7）")
    A("")
    if len(abl):
        A("| id | 变更 | dev 年化 | dev 回撤 | dev 笔数 | val 年化 | val 回撤 | val 笔数 | 信号数(dev) |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in abl.to_dict("records"):
            if r.get("window") != "dev":
                continue
            vrow = abl[(abl["id"] == r["id"]) & (abl["window"] == "val")]
            v = vrow.iloc[0] if len(vrow) else {}
            A(f"| {r['id']} | {r['desc']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('mdd'))} | "
              f"{int(r.get('n_trades') or 0)} | {_fmt_pct(v.get('cagr'))} | {_fmt_pct(v.get('mdd'))} | "
              f"{int(v.get('n_trades') or 0)} | {int(r.get('n_signals') or 0)} |")
    A("")
    if len(abl):
        zero = abl[(abl["window"] == "dev") & (abl["n_trades"] == 0)]
        for r in zero.to_dict("records"):
            A(f"> **{r['id']} 在 dev/val 均为 0 笔成交**：{r['desc']}。"
              "季/月结构 + 12% 初始风险门使信号集为空，不是“无损失”，而是“无法交易”。")
    A("")
    A("## 5. 参数邻域（§7.8 敏感性，不参与选择）")
    A("")
    if len(nei):
        A("| id | 变更 | dev 年化 | dev 回撤 | val 年化 | val 回撤 | val 笔数 |")
        A("|---|---|---:|---:|---:|---:|---:|")
        for r in nei.to_dict("records"):
            if r.get("window") != "dev":
                continue
            vrow = nei[(nei["id"] == r["id"]) & (nei["window"] == "val")]
            v = vrow.iloc[0] if len(vrow) else {}
            A(f"| {r['id']} | {r['desc']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('mdd'))} | "
              f"{_fmt_pct(v.get('cagr'))} | {_fmt_pct(v.get('mdd'))} | {int(v.get('n_trades') or 0)} |")
    A("")
    A("## 6. 多重性（DSR / PBO）")
    A("")
    A(f"- 新增配置数：{multi['n_new_configs']}；项目既有 N_sel={multi['n_project_sel']}；DSR 用 N={multi['n_total_for_dsr']}")
    for w in ("dev", "val"):
        d = multi[w]
        if d and d.get("dsr") is not None:
            A(f"- {w}: SR_daily={d['sr_daily']:.4f}，SR0_daily={d['sr0_daily']:.4f}，"
              f"DSR={d['dsr']:.4f}（z={d['z']:.2f}）")
        else:
            A(f"- {w}: DSR 不可算（样本或方差不足）")
    if "dev_family_dsr" in multi:
        d = multi["dev_family_dsr"]
        A(f"- 族内（邻域 {d.get('n_trials')} 配置，Var(SR)={d.get('var_sr'):.3e}）dev DSR={d.get('dsr'):.4f}"
          "——**只作诊断**：18 个邻域变体高度相关，Var(SR_trials) 被系统性低估，"
          "DSR 因此偏高；门槛口径是项目级 N=397 的 DSR。")
    for w, v in pbo.items():
        A(f"- PBO（邻域 {v['n_configs']} 配置，{v['n_comb']} 组合）{w}: {v['pbo']:.3f}，中位排名 {v['median_rank']:.2f}"
          f"（§12.1 G7 门槛 ≤ 0.25 ⇒ {'过' if v['pbo'] <= 0.25 else '不过'}）")
    A("")
    A("## 7. 自检与独立对账")
    A("")
    A("| 检查 | dev | val |")
    A("|---|---:|---:|")
    for k in ("cash_identity_max_abs", "nav_identity_max_abs", "t1_violations",
              "fills_on_suspension", "fills_at_locked_limit_open",
              "forced_valuation_fills_on_suspension", "max_positions", "risk_violations",
              "max_realized_risk_pct", "max_price_r_pct", "forced_sells", "rows_after_2020",
              "orders_execute_before_decision", "n_orders", "n_fills", "n_round_trips",
              "n_failed_orders", "n_ca_events", "n_tax_events", "final_nav", "liq_nav_after_cost"):
        A(f"| {k} | {sd.get(k)} | {sv.get(k)} |")
    A("")
    if rec:
        for w in ("dev", "val"):
            fc = rec[w]["fill_price_check"]
            il = rec[w]["independent_ledger"]
            A(f"- **独立账户 {w}**：成交价重算 {fc['n_fills']} 笔（止损 {fc['n_stop']}），"
              f"价格不符(>1e-6) {fc['n_price_mismatch_gt_1e-6']} 笔，最大差 {fc['max_price_diff']:.4f} 元；"
              f"费用差最大 {fc['max_fee_diff']:.4f} 元。")
            A(f"- **独立现金账 {w}**：期末 {il['final_nav_independent']:,.2f} vs 统一账户 "
              f"{il['final_nav_unified']:,.2f}，逐日最大差 {il['max_abs_diff']:.6f} 元，"
              f"相对差 {il['max_rel_diff']:.2e}。")
    A("")
    A(f"- **2021+ 扫描**：{scan['n_files']} 个输出 CSV 中，含 2021+ 日期列的文件 {len(scan['files_with_2021'])} 个"
      f"（{'PASS' if scan['pass'] else 'FAIL'}）。")
    A("因果自检（截断重算，§6.4）见 `08_因果自检.json`；试验台账见 `60_试验台账.csv`。")
    A("")
    A("## 8. 限制与未做的事")
    A("")
    A("- 2021-2025 留出**未运行**，本报告不含任何 2021+ 收益；延伸段（2026）同样未运行。")
    A("- dev 的 2008-2009 年 0 笔交易（12% 风险门全拒），因此 dev 回撤 −9% 不包含 2008 崩盘暴露；这不是稳健，是空仓。")
    A("- 结构层用周线分型、止损用周线结构低点（Q2/Q7 的字面读法）；若改用日线结构低点，样本与结果会变，"
      "属新的操作化，须新 trial_id，本报告不采用。")
    A("- 周/月线用市场日历聚合，停牌日沿用最后收盘；个股长期停牌的周 K 可能是平 bar（不影响分型的严格不等式）。")
    A("- 无行业中性、无市值分层、无涨跌停外的流动性建模；容量未做规模曲线（本候选暴露极低，容量不是约束）。")
    A("- 组合回撤 C3、MA20 C1 都基于 ca 空间；C3 只在窗口内计算峰值。")
    A(f"- 单笔账户风险按信号日收盘估值、按次日开盘成交，跳空会使实际风险略超 1.2%："
      f"dev 最大 {_fmt_pct(sd.get('max_realized_risk_pct'))}、val 最大 {_fmt_pct(sv.get('max_realized_risk_pct'))}"
      f"（超限笔数 {sd.get('risk_violations')} / {sv.get('risk_violations')}，均为跳空，未在成交后追改股数）。")
    A("")
    A("## 9. 是否申请留出")
    A("")
    A("**否。** 依据 §10.2/§10.3：验证期失败即淘汰，留出只用于否决且必须由独立执行者运行；"
      "本批不写 `stage=holdout` 的 planned 行，2021+ 保持未读。")
    A("")
    A("## 10. 复现")
    A("")
    A("```bash")
    A("cd /Users/admin/WorkSpace/trading/trading_趋势结构")
    A("python3 02_代码/主候选_T0005.py signals      # 信号覆盖")
    A("python3 02_代码/主候选_T0005.py main         # dev/val 主候选")
    A("python3 02_代码/主候选_T0005.py reconcile    # 独立账户 + 成交价重算 + 因果自检")
    A("python3 02_代码/主候选_T0005.py ablation     # V1-V8 + A1-A7")
    A("python3 02_代码/主候选_T0005.py neighbor     # §7.8 参数邻域")
    A("python3 02_代码/主候选_T0005.py report       # 本文件 + 多重性")
    A("```")
    (OUT / "50_结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"报告写入 {OUT / '50_结论.md'}")
    print(json.dumps(_json_safe(gates), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
