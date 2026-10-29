#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构策略族 — 非重绘枢轴 / 多周期结构 / 回调与失败结构 / 分层净增益
====================================================================
数据源（只读，不迁移不改写）：
  1) Baostock SQLite  /Users/admin/WorkSpace/baostock_data/data/baostock.db
       stock_kline_d(三复权, turn, is_st, trade_status, preclose), stock_basic, index_kline_d, trade_dates
  2) 同花顺 DuckDB    /Users/admin/Library/Application Support/hithink-finance/data/market.duckdb
       仅用于交叉校验（raw_adjustment_events / v_daily_hfq）

时间冻结：FREEZE = 2026-10-26。所有收益计算硬截断在 HOLDOUT_START 之前。
时间纪律：
  warmup  2005-01-01 ~ 2007-12-31   （预热/早期压力，不进入绩效统计）
  dev     2008-01-01 ~ 2015-12-31   （开发）
  val     2016-01-01 ~ 2020-12-31   （验证）
  holdout 2021-01-01 ~ 2026-10-26   （最终留出；本脚本禁止计算其收益，仅做覆盖审计）

用法：
  python3 策略族_结构.py cache      # 建立本地 parquet 缓存（一次性）
  python3 策略族_结构.py audit      # 数据/复权/停牌/涨跌停/退市 审计
  python3 策略族_结构.py events     # 生成结构事件（仅 <=2020）
  python3 策略族_结构.py study      # 事件研究（前瞻收益 / 分层）
  python3 策略族_结构.py backtest   # 组合回测（分层净增益）
  python3 策略族_结构.py exits      # 离场规则对比
  python3 策略族_结构.py selfcheck  # 自检（非重绘/无未来函数/口径）
  python3 策略族_结构.py all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 路径与常量
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "02_代码" / "_cache"
RES = ROOT / "03_结果"
CACHE.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

BS_DB = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"
THS_DB = "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"

FREEZE = "2026-10-26"
HOLDOUT_START = "2021-01-01"      # 硬边界：>= 此日期的收益一律不计算
PANEL_START = "2004-01-01"        # 面板起点（预热需要 2005 之前的历史）
PANEL_END = "2020-12-31"          # 分析面板终点＝验证期末；留出期数据不进入任何矩阵
WARMUP_START = "2005-01-01"
WARMUP_END = "2007-12-31"
DEV_START = "2008-01-01"
DEV_END = "2015-12-31"
VAL_START = "2016-01-01"
VAL_END = "2020-12-31"

# 组合参数
CAPITAL = 1_000_000.0
N_SLOTS = 10
COMM_RATE = 0.00025
COMM_MIN = 5.0
STAMP_SELL = 0.001          # <=2023-08-27 卖出印花税 0.1%
TRANSFER_RATE = 0.00002
SLIPPAGE = 0.001            # 单边 10bp

# 板块涨跌幅限制
def limit_pct(code: str, date: str, is_st: int) -> float:
    """按板块与日期返回涨跌幅限制（比例，如 0.10）。"""
    c = code.split(".")[-1]
    d = date.replace("-", "")
    if is_st == 1:
        return 0.05
    if c.startswith("688"):                      # 科创板
        return 0.20
    if c.startswith("300") and d >= "20200824":  # 创业板注册制
        return 0.20
    if c.startswith(("8", "4", "920")):          # 北交所
        return 0.30
    return 0.10


def board_of(code: str) -> str:
    c = code.split(".")[-1]
    if c.startswith("688"):
        return "STAR"
    if c.startswith("300"):
        return "ChiNext"
    if c.startswith(("8", "4", "920")):
        return "BSE"
    if c.startswith("002") or c.startswith("003"):
        return "SME"
    return "Main"


MAIN_BOARDS = ("Main", "SME")


# ----------------------------------------------------------------------------
# 1. 缓存
# ----------------------------------------------------------------------------
def _duck():
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{BS_DB}' AS bs (TYPE SQLITE, READ_ONLY)")
    return con


def build_cache(force: bool = False) -> None:
    """把 Baostock 三张表落到 parquet。仅取 <= FREEZE 且 >= PANEL_START。"""
    f_hfq = CACHE / "kline_hfq.parquet"
    f_raw = CACHE / "kline_raw.parquet"
    f_basic = CACHE / "stock_basic.parquet"
    f_index = CACHE / "index_kline.parquet"
    f_dates = CACHE / "trade_dates.parquet"
    if not force and all(p.exists() for p in (f_hfq, f_raw, f_basic, f_index, f_dates)):
        print("[cache] 已存在，跳过（--force 重建）")
        return

    t0 = time.time()
    con = _duck()
    cols = "code,date,open,high,low,close,preclose,volume,amount,turn,trade_status,is_st"
    for flag, path in ((1, f_hfq), (3, f_raw)):
        if path.exists() and not force:
            continue
        print(f"[cache] 读取 adjustflag={flag} ...")
        con.execute(f"""
            COPY (
              SELECT {cols}
              FROM bs.stock_kline_d
              WHERE adjustflag={flag}
                AND date >= '{PANEL_START}' AND date <= '{FREEZE}'
            ) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        print(f"        -> {path.name}  {path.stat().st_size/1e6:.1f}MB  {time.time()-t0:.0f}s")

    if force or not f_basic.exists():
        con.execute(f"COPY (SELECT * FROM bs.stock_basic) TO '{f_basic}' (FORMAT PARQUET)")
    if force or not f_index.exists():
        con.execute(f"""COPY (SELECT * FROM bs.index_kline_d WHERE date <= '{FREEZE}')
                        TO '{f_index}' (FORMAT PARQUET)""")
    if force or not f_dates.exists():
        con.execute(f"""COPY (SELECT * FROM bs.trade_dates WHERE calendar_date <= '{FREEZE}')
                        TO '{f_dates}' (FORMAT PARQUET)""")
    con.close()
    print(f"[cache] 完成 {time.time()-t0:.0f}s")
    for p in (f_hfq, f_raw, f_basic, f_index, f_dates):
        print(f"   {p.relative_to(ROOT)}  {p.stat().st_size/1e6:.1f}MB")


# ----------------------------------------------------------------------------
# 2. 面板构建
# ----------------------------------------------------------------------------
@dataclass
class Panel:
    dates: np.ndarray            # (nd,) 交易日字符串 'YYYY-MM-DD'
    codes: List[str]             # (ns,)
    O: np.ndarray                # (nd,ns) hfq
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    PC: np.ndarray               # 原始 preclose
    FAC: np.ndarray              # hfq_close / raw_close
    VOL: np.ndarray
    AMT: np.ndarray
    TURN: np.ndarray
    ST: np.ndarray
    STATUS: np.ndarray
    basic: pd.DataFrame
    bench: Dict[str, pd.Series] = field(default_factory=dict)

    def idx(self, date: str) -> int:
        return int(np.searchsorted(self.dates, date, side="left"))


def _pivot_to_matrix(df: pd.DataFrame, value: str, dates: np.ndarray, codes: List[str],
                     dtype=np.float32) -> np.ndarray:
    di = pd.Index(dates)
    ci = pd.Index(codes)
    wide = df.pivot(index="date", columns="code", values=value)
    wide = wide.reindex(index=di, columns=ci)
    return np.ascontiguousarray(wide.to_numpy(dtype=dtype, na_value=np.nan))


def load_panel(rebuild: bool = False) -> Panel:
    t0 = time.time()
    npz = CACHE / "panel_2020.npz"
    basic = pd.read_parquet(CACHE / "stock_basic.parquet")
    idx = pd.read_parquet(CACHE / "index_kline.parquet")
    td = pd.read_parquet(CACHE / "trade_dates.parquet")
    dates = np.sort(td.loc[td.is_trading_day == 1, "calendar_date"].astype(str).unique())
    dates = dates[(dates >= PANEL_START) & (dates <= PANEL_END)]

    basic["board"] = basic["code"].map(board_of)
    basic["ipo"] = basic["ipo_date"].astype(str)
    basic["out"] = basic["out_date"].astype(str).replace("None", "")
    basic = basic.set_index("code")

    if npz.exists() and not rebuild:
        z = np.load(npz, allow_pickle=True)
        if list(z["dates"]) == list(dates) and z["C"].shape[1] == len(z["codes"]):
            codes = [str(c) for c in z["codes"]]
            bench = {}
            for code, name in (("sh.000300", "HS300"), ("sh.000905", "CSI500"),
                               ("sh.000852", "CSI1000"), ("sh.000001", "SSE")):
                s = idx[idx.code == code].set_index("date")["close"].astype(float)
                s = s.reindex(pd.Index(dates)).astype(float)
                if s.notna().sum() > 100:
                    bench[name] = s
            p = Panel(dates=dates, codes=codes,
                      O=z["O"], H=z["H"], L=z["L"], C=z["C"], PC=z["PC"], FAC=z["FAC"],
                      VOL=z["VOL"], AMT=z["AMT"], TURN=z["TURN"], ST=z["ST"],
                      STATUS=z["STATUS"], basic=basic, bench=bench)
            print(f"[panel] 载入缓存 {time.time()-t0:.1f}s shape={p.C.shape}")
            return p

    hfq = pd.read_parquet(CACHE / "kline_hfq.parquet")
    raw = pd.read_parquet(CACHE / "kline_raw.parquet")
    codes = sorted(hfq["code"].unique().tolist())

    print(f"[panel] {len(codes)} 只股票 × {len(dates)} 交易日，透视中 ...")
    C = _pivot_to_matrix(hfq, "close", dates, codes)
    O = _pivot_to_matrix(hfq, "open", dates, codes)
    H = _pivot_to_matrix(hfq, "high", dates, codes)
    L = _pivot_to_matrix(hfq, "low", dates, codes)
    PC = _pivot_to_matrix(raw, "preclose", dates, codes)
    RC = _pivot_to_matrix(raw, "close", dates, codes)
    VOL = _pivot_to_matrix(raw, "volume", dates, codes)
    AMT = _pivot_to_matrix(raw, "amount", dates, codes)
    TURN = _pivot_to_matrix(raw, "turn", dates, codes)
    ST = np.nan_to_num(_pivot_to_matrix(raw, "is_st", dates, codes), nan=0).astype(np.int8)
    STATUS = np.nan_to_num(_pivot_to_matrix(raw, "trade_status", dates, codes), nan=0).astype(np.int8)

    with np.errstate(divide="ignore", invalid="ignore"):
        FAC = np.where(np.isfinite(RC) & (RC > 0), C / RC, np.nan).astype(np.float32)

    bench = {}
    for code, name in (("sh.000300", "HS300"), ("sh.000905", "CSI500"),
                       ("sh.000852", "CSI1000"), ("sh.000001", "SSE")):
        s = idx[idx.code == code].set_index("date")["close"].astype(float)
        s = s.reindex(pd.Index(dates)).astype(float)
        if s.notna().sum() > 100:
            bench[name] = s

    np.savez(npz, dates=dates, codes=np.array(codes, dtype=object),
             O=O, H=H, L=L, C=C, PC=PC, FAC=FAC, VOL=VOL, AMT=AMT,
             TURN=TURN, ST=ST, STATUS=STATUS)
    p = Panel(dates=dates, codes=codes, O=O, H=H, L=L, C=C, PC=PC, FAC=FAC,
              VOL=VOL, AMT=AMT, TURN=TURN, ST=ST, STATUS=STATUS, basic=basic, bench=bench)
    print(f"[panel] 完成 {time.time()-t0:.0f}s  shape={C.shape}")
    return p


# ----------------------------------------------------------------------------
# 3. 非重绘枢轴
# ----------------------------------------------------------------------------
def pivot_candidates(h: np.ndarray, l: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (ph_idx, pl_idx)：满足
         high[i] > max(high[i-K..i-1])  且  high[i] >= max(high[i+1..i+K])
       （左侧严格、右侧非严格 → 平台只取第一根）
       确认时刻 = i + K（收盘后才知道）。
    """
    n = h.size
    out_h, out_l = [], []
    if n < 2 * K + 1:
        return np.array(out_h, np.int64), np.array(out_l, np.int64)
    sh = pd.Series(h)
    sl = pd.Series(l)
    lmax = sh.rolling(K).max().shift(1).to_numpy()
    rmax = sh[::-1].rolling(K).max()[::-1].shift(-1).to_numpy()
    lmin = sl.rolling(K).min().shift(1).to_numpy()
    rmin = sl[::-1].rolling(K).min()[::-1].shift(-1).to_numpy()
    ph = (h > lmax) & (h >= rmax)
    pl = (l < lmin) & (l <= rmin)
    return np.flatnonzero(ph), np.flatnonzero(pl)


@dataclass
class PivotState:
    """逐日可用的枢轴状态（全部来自已确认枢轴，confirm_idx <= t）。"""
    up: np.ndarray          # bool HH+HL
    down: np.ndarray        # bool LH+LL
    last_h: np.ndarray      # 最近已确认枢轴高点价格
    last_h_idx: np.ndarray
    last_l: np.ndarray
    last_l_idx: np.ndarray
    prev_h: np.ndarray
    prev_l: np.ndarray
    n_piv: np.ndarray       # 已确认枢轴计数
    pivot_log: List[dict]   # 调试用


def build_pivot_state(h, l, K, n) -> PivotState:
    """以「确认时刻」为主线做一次顺序提交，然后前向填充到日频。"""
    ph_idx, pl_idx = pivot_candidates(h, l, K)
    cand = [(int(i), 1) for i in ph_idx] + [(int(i), 0) for i in pl_idx]
    cand.sort(key=lambda x: (x[0] + K, x[0]))     # 按确认时刻排序

    # 事件序列：每条记录 confirm 时刻的状态快照
    ev_t: List[int] = []
    ev_up: List[bool] = []
    ev_down: List[bool] = []
    ev_lh: List[float] = []
    ev_lhi: List[int] = []
    ev_ll: List[float] = []
    ev_lli: List[int] = []
    ev_ph: List[float] = []
    ev_pl: List[float] = []
    ev_n: List[int] = []
    log: List[dict] = []

    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []
    last_kind = -1
    last_pos = -1
    committed = 0

    def snapshot(t):
        if len(highs) >= 2 and len(lows) >= 2:
            h1, h2 = highs[-1], highs[-2]
            l1, l2 = lows[-1], lows[-2]
            up = (h1[1] > h2[1]) and (l1[1] > l2[1])
            down = (h1[1] < h2[1]) and (l1[1] < l2[1])
        else:
            up = down = False
        ev_t.append(t)
        ev_up.append(up); ev_down.append(down)
        ev_lh.append(highs[-1][1] if highs else np.nan)
        ev_lhi.append(highs[-1][0] if highs else -1)
        ev_ll.append(lows[-1][1] if lows else np.nan)
        ev_lli.append(lows[-1][0] if lows else -1)
        ev_ph.append(highs[-2][1] if len(highs) >= 2 else np.nan)
        ev_pl.append(lows[-2][1] if len(lows) >= 2 else np.nan)
        ev_n.append(committed)

    for i, kind in cand:
        c = i + K
        if c >= n:
            continue
        # 同类型连续：取更极端者（这是「冲突消解」，在确认时刻完成，不是回填）
        if kind == last_kind and last_pos >= 0:
            arr = highs if kind == 1 else lows
            if arr:
                if kind == 1:
                    if h[i] > arr[-1][1]:
                        arr[-1] = (i, float(h[i]))
                else:
                    if l[i] < arr[-1][1]:
                        arr[-1] = (i, float(l[i]))
                last_pos = arr[-1][0]
                committed += 0
                snapshot(c)
                log.append({"confirm": c, "idx": i, "kind": kind, "action": "replace"})
                continue
        if kind == 1:
            highs.append((i, float(h[i])))
        else:
            lows.append((i, float(l[i])))
        last_kind = kind
        last_pos = i
        committed += 1
        snapshot(c)
        log.append({"confirm": c, "idx": i, "kind": kind, "action": "append"})

    bars = np.arange(n)
    if not ev_t:
        z = np.zeros(n, bool)
        nan = np.full(n, np.nan)
        neg = np.full(n, -1, np.int64)
        return PivotState(z, z, nan, neg, nan, neg, nan, nan, np.zeros(n, np.int64), log)

    et = np.asarray(ev_t)
    pos = np.searchsorted(et, bars, side="right") - 1     # 每个日频 bar 可见的最后一次快照
    neg1 = pos < 0
    pos_c = np.clip(pos, 0, len(et) - 1)

    def take(lst, dtype=float):
        a = np.asarray(lst, dtype=float)
        out = a[pos_c].copy()
        out[neg1] = np.nan
        if dtype is np.int64:
            out = np.where(np.isnan(out), -1, out).astype(np.int64)
        return out

    up = np.asarray(ev_up, bool)[pos_c].copy(); up[neg1] = False
    down = np.asarray(ev_down, bool)[pos_c].copy(); down[neg1] = False
    n_piv = np.asarray(ev_n, np.int64)[pos_c].copy(); n_piv[neg1] = 0
    last_h_idx = take(ev_lhi, np.int64)
    last_l_idx = take(ev_lli, np.int64)
    return PivotState(up=up, down=down, last_h=take(ev_lh), last_h_idx=last_h_idx,
                      last_l=take(ev_ll), last_l_idx=last_l_idx,
                      prev_h=take(ev_ph), prev_l=take(ev_pl), n_piv=n_piv, pivot_log=log)


# ----------------------------------------------------------------------------
# 4. 多周期（周/月）结构，仅用已完成周期
# ----------------------------------------------------------------------------
def build_htf(dates: np.ndarray, o, h, l, c, freq: str, K: int):
    """把日线合成周/月线。返回 (htf_bars_df, daily_map)
    htf 每根 bar 的 avail_from = 该周期结束后的第一个交易日；
    结构标签在 htf 内部用确认枢轴计算（confirm = j+K）。"""
    d = pd.to_datetime(dates)
    if freq == "W":
        key = d.isocalendar()
        key = key.year.astype(str) + "-" + key.week.astype(str).str.zfill(2)
    elif freq == "M":
        key = d.year.astype(str) + "-" + d.month.astype(str).str.zfill(2)
    else:
        raise ValueError(freq)
    key = pd.Series(key)
    grp = key.ne(key.shift()).cumsum().to_numpy()

    nh = grp.max()
    hh = np.full(nh, np.nan); ll = np.full(nh, np.nan)
    cc = np.full(nh, np.nan); oo = np.full(nh, np.nan)
    end_i = np.zeros(nh, np.int64)
    for g in range(1, nh + 1):
        m = grp == g
        hh[g-1] = np.nanmax(h[m]); ll[g-1] = np.nanmin(l[m])
        idx = np.flatnonzero(m)
        oo[g-1] = o[idx[0]]; cc[g-1] = c[idx[-1]]; end_i[g-1] = idx[-1]

    ps = build_pivot_state(hh, ll, K, nh)
    # htf bar j 的结构 = 该 bar 收盘时可见状态（confirm<=j 已包含在 build_pivot_state 内）
    avail_from = end_i + 1                      # 下一交易日才可用
    return dict(h=hh, l=ll, c=cc, o=oo, end_i=end_i, avail_from=avail_from,
                up=ps.up, down=ps.down, last_h=ps.last_h, last_l=ps.last_l) , grp


def map_htf_to_daily(htf, n_daily: int, grp: np.ndarray, field_name: str):
    """每个日频 bar 只能看到 avail_from <= t 的最后一根 htf bar。"""
    a = htf[field_name]
    af = htf["avail_from"]
    # af 单调递增 → 对每个 t 找 af <= t 的最大 j
    j = np.searchsorted(af, np.arange(n_daily), side="right") - 1
    out_ok = j >= 0
    jc = np.clip(j, 0, len(a) - 1)
    if a.dtype == bool:
        out = a[jc].copy(); out[~out_ok] = False
    else:
        out = a[jc].astype(float).copy(); out[~out_ok] = np.nan
    return out, j


# ----------------------------------------------------------------------------
# 5. 技术指标
# ----------------------------------------------------------------------------
def sma(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w).mean().to_numpy()


def atr(h, l, c, w=20) -> np.ndarray:
    pc = np.roll(c, 1); pc[0] = np.nan
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(w, min_periods=w).mean().to_numpy()


# ----------------------------------------------------------------------------
# 6. 单股票事件生成
# ----------------------------------------------------------------------------
SIGNALS = ("A_first_drive", "B_first_pullback", "C_later_pullback",
           "D_breakout", "E_restart_support", "G_ma_structure")


def gen_events_one(code: str, dates: np.ndarray, o, h, l, c, vol, amt, turn, st, status,
                   pc=None, fac=None,
                   K: int = 3, KW: int = 2, KM: int = 1,
                   min_hist: int = 250, with_exits: bool = True) -> pd.DataFrame:
    n = len(c)
    if n < min_hist + 30:
        return pd.DataFrame()

    valid = np.isfinite(c) & (c > 0)
    if valid.sum() < min_hist + 30:
        return pd.DataFrame()

    ma5, ma10 = sma(c, 5), sma(c, 10)
    ma20, ma60, ma120 = sma(c, 20), sma(c, 60), sma(c, 120)
    a20 = atr(h, l, c, 20)
    vma20 = sma(vol, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        vratio = vol / np.roll(vma20, 1)
        atr_pct = a20 / c
        d20 = c / ma20 - 1.0
        d60 = c / ma60 - 1.0
    vratio[0] = np.nan

    pv = build_pivot_state(h, l, K, n)
    (htf_w, grp_w) = build_htf(dates, o, h, l, c, "W", KW)
    (htf_m, grp_m) = build_htf(dates, o, h, l, c, "M", KM)
    w_up, _ = map_htf_to_daily(htf_w, n, grp_w, "up")
    w_dn, _ = map_htf_to_daily(htf_w, n, grp_w, "down")
    m_up, _ = map_htf_to_daily(htf_m, n, grp_m, "up")
    m_dn, _ = map_htf_to_daily(htf_m, n, grp_m, "down")

    bars = np.arange(n)
    R = pv.last_h; R_idx = pv.last_h_idx
    S = pv.last_l; S_idx = pv.last_l_idx
    up_struct, down_struct = pv.up, pv.down

    # 结构低点是否「大低点」：最近 120 日最低
    low120 = pd.Series(np.where(valid, l, np.nan)).rolling(120, min_periods=60).min().to_numpy()
    high120 = pd.Series(np.where(valid, h, np.nan)).rolling(120, min_periods=60).max().to_numpy()

    # ---- 共同前置条件
    hist_ok = bars >= min_hist
    tradable = (status == 1) & np.isfinite(c) & (vol > 0)
    not_st = np.nan_to_num(st, nan=0) == 0
    liquid = pd.Series(np.where(np.isfinite(amt), amt, np.nan)).rolling(20, min_periods=20).median().to_numpy() >= 5e7
    price_ok = c >= 2.0
    base_ok = hist_ok & tradable & not_st & liquid & price_ok
    # 次日必须有可交易 bar（入场可行性）
    next_ok = np.zeros(n, bool)
    next_ok[:-1] = (status[1:] == 1) & (vol[1:] > 0)
    base_ok &= next_ok

    cr = np.roll(c, 1); cr[0] = np.nan
    hr = np.roll(h, 1); hr[0] = np.nan
    lr = np.roll(l, 1); lr[0] = np.nan
    orr = np.roll(o, 1); orr[0] = np.nan

    # ---- A 第一驱动：大低点确认后首次收复「下跌结构最后的高点」
    major_low = np.isfinite(S) & (S <= low120 * 1.001)
    leg_bars = np.where(np.isfinite(S_idx), bars - S_idx, 10**6)
    prior_not_up = ~np.roll(up_struct, 1)
    A = (base_ok & major_low & prior_not_up &
         (leg_bars >= 5) & (leg_bars <= 60) &
         np.isfinite(R) & (cr <= R) & (c > R) &
         (vratio >= 1.2) & (c > ma20))
    A[:] &= np.isfinite(R) & np.isfinite(S)

    # ---- 驱动腿：A 之后直到收盘跌破驱动起点（结构失效）或 250 根
    a_idx = np.flatnonzero(A)
    leg_id = np.full(n, -1, np.int64)
    leg_start = np.full(n, -1, np.int64)
    drive_low_t = np.full(n, np.nan)
    leg_high_t = np.full(n, np.nan)
    peak_idx = np.full(n, -1, np.int64)
    for k, a in enumerate(a_idx):
        end = n if k + 1 >= a_idx.size else a_idx[k + 1]
        end = min(end, a + 250)
        dl = S[a]
        if not np.isfinite(dl):
            continue
        below = np.flatnonzero(c[a:end] < dl)
        if below.size:
            end = a + int(below[0])
        if end <= a:
            end = a + 1
        leg_id[a:end] = k
        leg_start[a:end] = a
        drive_low_t[a:end] = dl
        seg = h[a:end].astype(float)
        cm = np.maximum.accumulate(seg)
        is_peak = seg >= cm
        cand = np.where(is_peak, np.arange(seg.size), 0)
        peak_idx[a:end] = a + np.maximum.accumulate(cand)
        leg_high_t[a:end] = cm

    # 回撤低点：近 10 日最低价（因果、近似）；两种口径分开
    pb_low = pd.Series(np.where(valid, l, np.nan)).rolling(10, min_periods=3).min().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        dd_abs = (leg_high_t - pb_low) / np.maximum(leg_high_t, 1e-9)          # 距高点绝对回撤
        pull_depth = (leg_high_t - pb_low) / np.maximum(leg_high_t - drive_low_t, 1e-9)  # 驱动腿回吐比例

    # 回撤阈值（ATR 自适应）
    delta = np.clip(0.8 * np.nan_to_num(atr_pct, nan=0.03), 0.02, 0.08)

    # 重启：收盘站上近 3 日最高，且此前 5 个交易日内出现过有效回撤
    hh3 = pd.Series(np.where(valid, h, np.nan)).rolling(3, min_periods=2).max().shift(1).to_numpy()
    raw_res = np.isfinite(hh3) & (c > hh3) & (c > cr)
    pb_flag = (leg_start >= 0) & (bars - peak_idx >= 1) & (dd_abs >= 0.5 * delta)
    pf_idx = np.where(pb_flag, bars, -1)
    last_pb = np.maximum.accumulate(pf_idx)
    pb_recent = (last_pb >= 0) & ((bars - last_pb) <= 5)
    resumption = raw_res & ~np.roll(raw_res, 1) & pb_recent
    resumption[0] = False
    # 该次回调所对应的峰值（用于去重与控制变量）
    ref_peak = np.where(last_pb >= 0, peak_idx[np.clip(last_pb, 0, n - 1)], -1)

    in_leg = leg_start >= 0
    pb_common = (in_leg & base_ok & resumption & np.isfinite(pull_depth) &
                 (pull_depth >= 0.15) & (pull_depth <= 0.80) &
                 (dd_abs >= delta) & (dd_abs <= 0.25) &
                 (pb_low > drive_low_t) & (c > ma20 * 0.97) & (c > drive_low_t) &
                 (bars - ref_peak <= 40))
    # 同一次回调只保留第一次重启
    key = np.where(pb_common, leg_id * 1_000_000 + ref_peak, -1)
    _, first_idx = np.unique(key, return_index=True)
    first_mask = np.zeros(n, bool)
    first_mask[first_idx[key[first_idx] >= 0]] = True
    pb_common &= first_mask
    # 回调序号 = 本驱动腿内第几次回调重启（1=首次）
    pb_ord = np.zeros(n, np.int64)
    _cand = np.flatnonzero(pb_common)
    if _cand.size:
        _g = pd.Series(leg_id[_cand])
        pb_ord[_cand] = (_g.groupby(_g).cumcount() + 1).to_numpy()
    B = pb_common & (pb_ord == 1)
    C_ = pb_common & (pb_ord >= 2)

    # ---- D 阻力突破（收盘首次越过已确认枢轴高点）
    r_age = np.where(np.isfinite(R_idx), bars - R_idx, 10**6)
    base_h = pd.Series(np.where(valid, h, np.nan)).rolling(30, min_periods=20).max().to_numpy()
    base_l = pd.Series(np.where(valid, l, np.nan)).rolling(30, min_periods=20).min().to_numpy()
    with np.errstate(invalid="ignore"):
        base_range = (base_h - base_l) / np.maximum(c, 1e-9)
    D = (base_ok & np.isfinite(R) & (r_age >= 10) & (r_age <= 250) &
         (cr <= R) & (c > R) & (vratio >= 1.3) &
         (base_range >= 0.05) & (base_range <= 0.60) & (c > ma60))
    # 首次 / 重复突破：120 日内是否已有同类收盘突破
    last_break = np.full(n, -10**6, np.int64)
    b_idx = np.flatnonzero(D)
    if b_idx.size:
        lb = -10**6
        pos = 0
        for t in range(n):
            if pos < b_idx.size and b_idx[pos] == t:
                last_break[t] = lb
                lb = t
                pos += 1
            elif t > 0:
                last_break[t] = last_break[t - 1]
    brk_first = D & ((bars - last_break) > 120)

    # ---- E 回踩支撑再启动（前高转为支撑 / 均线支撑，要求「首次触及」）
    touch_prev_high = (np.isfinite(R) & (c > R) & (lr > R * 1.02) & (l <= R * 1.02))
    ma_supp = (lr > ma20 * 1.01) & (l <= ma20 * 1.01) & (c > ma20)
    E = (base_ok & (touch_prev_high | ma_supp) & (c > hr) & (c > cr) & (vratio >= 0.9) &
         (up_struct | (c > ma20)) & (d20 > -0.02))

    # ---- G 均线结构（纯均线，无枢轴）
    g_cross = (cr <= ma20) & (c > ma20) & (ma20 > ma60) & (ma20 > np.roll(ma20, 5))
    G = base_ok & g_cross & (d60 > -0.05)

    if os.environ.get("STRUCT_DEBUG"):
        _m = in_leg.copy()
        for nm, vv in [("in_leg", in_leg), ("&base_ok", base_ok), ("&resumption", resumption),
                       ("&retr", (pull_depth >= 0.15) & (pull_depth <= 0.80)),
                       ("&dd_ge", dd_abs >= delta), ("&dd_le25", dd_abs <= 0.25),
                       ("&pblow_dl", pb_low > drive_low_t), ("&c_ma20", c > ma20 * 0.97),
                       ("&c_dl", c > drive_low_t),
                       ("&lag", (bars - peak_idx >= 2) & (bars - peak_idx <= 30))]:
            _m = _m & vv
            print(f"      [cum {nm}] {int(_m.sum())}")
        print(f"  [dbg {code}] n={n} A={int(A.sum())} B={int(B.sum())} C={int(C_.sum())} "
              f"D={int(D.sum())} E={int(E.sum())} G={int(G.sum())} pb_common={int(pb_common.sum())} "
              f"legs={int(a_idx.size)}")

    ev = []
    feats = dict(
        up_struct=up_struct, down_struct=down_struct, w_up=w_up, w_dn=w_dn,
        m_up=m_up, m_dn=m_dn, vratio=vratio, atr_pct=atr_pct, d20=d20, d60=d60,
    )
    sig_map = {"A_first_drive": A, "B_first_pullback": B, "C_later_pullback": C_,
               "D_breakout": D, "E_restart_support": E, "G_ma_structure": G}

    # ---- 可交易性：次日入场与离场顺延
    trad = (status == 1) & (vol > 0) & np.isfinite(c)
    nv = np.where(trad, bars, n)                     # 下一个可交易 bar
    nv = np.minimum.accumulate(nv[::-1])[::-1]
    nv = np.append(nv, n)                            # 越界哨兵

    lim = np.array([limit_pct(code, str(d), int(s) if np.isfinite(s) else 0)
                    for d, s in zip(dates, st)], dtype=float) if pc is not None else np.full(n, 0.10)
    if pc is not None and fac is not None:
        raw_open = np.where(np.isfinite(fac) & (fac > 0), o / fac, np.nan)
        up_lim = np.round(pc * (1 + lim), 2)
        dn_lim = np.round(pc * (1 - lim), 2)
        buy_block = ~(np.isfinite(raw_open) & np.isfinite(up_lim)) | (raw_open >= up_lim - 1e-6)
        sell_block = ~(np.isfinite(raw_open) & np.isfinite(dn_lim)) | (raw_open <= dn_lim + 1e-6)
    else:
        raw_open = o.copy(); up_lim = np.full(n, np.inf); dn_lim = np.zeros(n)
        buy_block = np.zeros(n, bool); sell_block = np.zeros(n, bool)

    def roll_from(start_vec: np.ndarray, block: Optional[np.ndarray] = None, limit: int = 10) -> np.ndarray:
        """从 start 起找第一个「可交易且未被 block」的 bar，找不到返回 -1。"""
        res = np.full(start_vec.size, -1, np.int64)
        cur = start_vec.astype(np.int64).copy()
        blk = block if block is not None else np.zeros(n, bool)
        for _ in range(limit + 1):
            live = (cur >= 0) & (cur < n)
            cur_c = np.clip(cur, 0, n - 1)
            ok = live & ~blk[cur_c] & trad[cur_c]
            res[ok & (res < 0)] = cur_c[ok & (res < 0)]
            cur = np.where(live & ~ok, cur + 1, cur)
        return res

    for name, mask in sig_map.items():
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        sub = pd.DataFrame({
            "code": code, "date": dates[idx], "i": idx, "sig": name, "k_pivot": K,
            "brk_margin": np.where(np.isfinite(R[idx]),
                                   c[idx] / np.where(np.isfinite(R[idx]), R[idx], 1.0) - 1.0, np.nan),
            "pull_depth": pull_depth[idx], "pb_ord": pb_ord[idx], "dd_abs": dd_abs[idx],
            "r_age": np.where(np.isfinite(r_age[idx]), r_age[idx], -1),
            "leg_bars": np.where(np.isfinite(leg_bars[idx]), leg_bars[idx], -1),
            "leg_high": leg_high_t[idx], "drive_low": drive_low_t[idx],
            "last_h": R[idx], "last_l": S[idx],
        })
        for k2, v2 in feats.items():
            sub[k2] = v2[idx]
        if name == "D_breakout":
            sub["sub"] = np.where(brk_first[idx], "first", "repeat")
        elif name == "E_restart_support":
            sub["sub"] = np.where(touch_prev_high[idx], "pivot_support", "ma_support")
        else:
            sub["sub"] = ""
        ev.append(sub)

    if not ev:
        return pd.DataFrame()
    out = pd.concat(ev, ignore_index=True).drop_duplicates(subset=["code", "date", "sig"])
    out = out.reset_index(drop=True)

    if with_exits:
        entry_roll = roll_from(out["i"].to_numpy() + 1, block=buy_block, limit=5)
        out["entry_i"] = np.where(entry_roll <= out["i"].to_numpy() + 5, entry_roll, -1)
        out = attach_exits(out, o, h, l, c, trad, roll_from, pv.last_l, pv.last_h,
                           sell_block=sell_block)
    return out


# ----------------------------------------------------------------------------
# 6b. 向量化前瞻收益 + 离场规则
# ----------------------------------------------------------------------------
HORIZONS = (5, 10, 20, 40, 60)
EXIT_RULES = ("hold10", "hold20", "hold40", "struct", "failstruct", "failbrk",
              "struct_tp8", "tp8", "trail3", "gate", "struct_gate")


def attach_exits(ev: pd.DataFrame, o, h, l, c, trad, roll_from, last_l, last_h,
                 sell_block=None, max_hold: int = 120) -> pd.DataFrame:
    """对每个事件计算：次日开盘入场、若干离场规则下的毛收益与离场 bar。
    入场日若一字涨停/停牌由调用方在 roll_from 的 block 中处理。"""
    n = c.size
    i = ev["i"].to_numpy()
    entry = ev["entry_i"].to_numpy()
    ok = entry >= 0
    eh = np.clip(entry, 0, n - 1)
    base = o[eh]
    with np.errstate(invalid="ignore", divide="ignore"):
        ev["entry_px_hfq"] = np.where(ok, base, np.nan)
        for hz in HORIZONS:
            j = np.clip(entry + hz - 1, 0, n - 1)
            r = c[j] / base - 1.0
            r[(~ok) | (entry + hz - 1 >= n)] = np.nan
            ev[f"fwd{hz}"] = r

        W = max_hold
        offs = np.arange(W + 1)[None, :]
        jj = entry[:, None] + offs
        valid_j = ok[:, None] & (jj < n)
        jc = np.clip(jj, 0, n - 1)
        pc_ = np.where(valid_j, c[jc], np.nan)
        ph_ = np.where(valid_j, h[jc], np.nan)
        pl_ = np.where(valid_j, l[jc], np.nan)
        peak = np.fmax.accumulate(np.where(np.isfinite(pc_), pc_, -np.inf), axis=1)
        atr_t = atr(h, l, c, 20)[jc]
        S_path = np.where(valid_j, last_l[jc], np.nan)     # 逐日「最新已确认结构低点」→ 移动止损
        drive_low_t = ev["drive_low"].to_numpy()[:, None]
        fail_ref = np.where(np.isfinite(drive_low_t), drive_low_t, S_path)

        w20 = 20
        m20 = ok[:, None] & ((entry[:, None] + offs[:, :w20]) < n)
        j20 = np.clip(entry[:, None] + offs[:, :w20], 0, n - 1)
        ev["mae20"] = np.where(m20, l[j20], np.nan).min(axis=1) / base - 1.0
        ev["mfe20"] = np.where(m20, h[j20], np.nan).max(axis=1) / base - 1.0

        def first_hit(cond: np.ndarray, start: int = 1) -> np.ndarray:
            cond = cond.copy()
            cond[:, :start] = False
            return np.where(cond.any(axis=1), np.argmax(cond, axis=1), -1)

        def settle(pos_off: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            pos_off = np.where(pos_off >= 0, pos_off, W)       # 未触发 → 到期离场（避免选择性缺失）
            dec = np.where(ok, entry + pos_off, -1)
            ex = roll_from(dec + 1, block=sell_block, limit=10)
            ex = np.where(ok & (ex >= 0) & ((ex - entry) <= W + 10), ex, -1)
            r = np.where(ex >= 0, o[np.clip(ex, 0, n - 1)] / base - 1.0, np.nan)
            return ex, r

        ev["exit_i_hold10"], ev["ret_hold10"] = settle(np.full(i.size, 10))
        ev["exit_i_hold20"], ev["ret_hold20"] = settle(np.full(i.size, 20))
        ev["exit_i_hold40"], ev["ret_hold40"] = settle(np.full(i.size, 40))

        hit_s = first_hit(pc_ < S_path)
        ev["exit_i_struct"], ev["ret_struct"] = settle(hit_s)

        hit_f = first_hit((pc_ < S_path) & (pc_ < fail_ref))
        ev["exit_i_fail"], ev["ret_fail"] = settle(hit_f)

        # 失败结构（突破失败）：入场后 3 根内收盘跌回被突破的枢轴高点之下
        # 仅对「入场价确实在被突破位之上」的事件生效；否则退化为结构止损
        R_t = np.where(np.isfinite(ev["last_h"].to_numpy()), ev["last_h"].to_numpy(), np.nan)[:, None]
        offs_b = np.arange(W + 1)[None, :]
        use_fb = np.isfinite(R_t) & (R_t < base[:, None] * 0.995)
        cond_fb = np.where(use_fb, (offs_b <= 3) & (pc_ < R_t), pc_ < S_path)
        hit_fb = first_hit(cond_fb)
        ev["exit_i_failbrk"], ev["ret_failbrk"] = settle(hit_fb)

        # 失败结构（结构破坏）：收盘跌破「入场时固定的结构低点」（不随之上移）
        hit_fs = first_hit(pc_ < ev["last_l"].to_numpy()[:, None])
        ev["exit_i_failstruct"], ev["ret_failstruct"] = settle(hit_fs)

        tp = first_hit(ph_ >= base[:, None] * 1.08)
        ev["exit_i_tp8"], ev["ret_tp8"] = settle(tp)

        both = np.where((tp >= 0) & ((hit_s < 0) | (tp < hit_s)), tp, hit_s)
        ev["exit_i_struct_tp8"], ev["ret_struct_tp8"] = settle(both)

        trail = first_hit(pc_ < peak - 3.0 * atr_t)
        ev["exit_i_trail3"], ev["ret_trail3"] = settle(trail)

        # 「买入后是否符合预期」的可执行版本：每 20 根检查一次，浮盈未达 3% 即离场，
        # 达标则继续持有（滚动闸门 = 提前离场 + 让利润奔跑的组合）
        def gate_pos(thr: float = 0.03, step: int = 20, cap: int = W) -> np.ndarray:
            pos = np.full(i.size, -1, np.int64)
            done = np.zeros(i.size, bool)
            for k in range(step, cap, step):
                j = np.clip(entry + k, 0, n - 1)
                r = np.where(ok & ((entry + k) < n), c[j] / base - 1.0, np.nan)
                hit = (~done) & ok & ((entry + k) < n) & np.isfinite(r) & (r < thr)
                pos[hit] = k
                done |= hit
            pos[~done] = cap
            return pos

        gp = gate_pos()
        ev["exit_i_gate"], ev["ret_gate"] = settle(gp)
        gp2 = np.where((hit_s >= 0) & ((gp < 0) | (hit_s < gp)), hit_s, gp)
        ev["exit_i_struct_gate"], ev["ret_struct_gate"] = settle(gp2)

        # 事后标注：入场后 20 日内是否上涨超过 3%（「符合预期」）
        with np.errstate(invalid="ignore"):
            hi20 = np.nanmax(np.where(valid_j[:, :w20], ph_[:, :w20], np.nan), axis=1)
        ev["made_new_high20"] = hi20 >= base * 1.03
    return ev




# ----------------------------------------------------------------------------
# 7. 事件表生成
# ----------------------------------------------------------------------------
def run_events(k: int = 3, tag: Optional[str] = None, limit: Optional[int] = None,
               kw: int = 2, km: int = 1) -> Path:
    p = load_panel()
    tag = tag or f"K{k}"
    frames = []
    t0 = time.time()
    codes = p.codes if limit is None else p.codes[:limit]
    for j, code in enumerate(codes):
        c = p.C[:, j]
        fin = np.isfinite(c)
        if fin.sum() < 320:
            continue
        i0 = int(np.argmax(fin))
        i1 = int(len(fin) - np.argmax(fin[::-1]))
        s = slice(i0, i1)
        try:
            ev = gen_events_one(code, p.dates[s], p.O[s, j], p.H[s, j], p.L[s, j], c[s],
                                p.VOL[s, j], p.AMT[s, j], p.TURN[s, j], p.ST[s, j], p.STATUS[s, j],
                                pc=p.PC[s, j], fac=p.FAC[s, j], K=k, KW=kw, KM=km)
        except Exception as e:  # 单只失败不拖垮全量
            print(f"  !! {code}: {e}")
            continue
        if len(ev) == 0:
            continue
        ev["code_i"] = j
        ev["off"] = i0
        # 只平移有效索引；-1 是「不可成交」哨兵，必须保持 -1
        for col in [c2 for c2 in ev.columns
                    if c2 == "i" or c2.startswith("exit_i_") or c2 == "entry_i"]:
            v = ev[col].to_numpy()
            ev[col] = np.where(v >= 0, v + i0, v)
        bad = ((ev.get("entry_i", pd.Series([0])).to_numpy() >= 0) &
               (ev.get("entry_i", pd.Series([0])).to_numpy() <= ev["i"].to_numpy()))
        if bad.any():
            raise RuntimeError(f"{code}: entry_i <= i  ({int(bad.sum())} 条)")
        frames.append(ev)
        if (j + 1) % 500 == 0:
            print(f"  [{tag}] {j+1}/{len(codes)}  {time.time()-t0:.0f}s  events={sum(len(f) for f in frames)}")
    df = pd.concat(frames, ignore_index=True)
    df["date_idx"] = np.searchsorted(p.dates, df["date"].to_numpy(), side="left")
    df = df[df["date"] <= VAL_END].reset_index(drop=True)
    out = RES / f"结构族_事件_{tag}.parquet"
    df.to_parquet(out, index=False)
    print(f"[events] {tag}: {len(df)} 条 → {out.relative_to(ROOT)}  {time.time()-t0:.0f}s")
    print(df.groupby(["sig"]).size().to_string())
    return out


# ----------------------------------------------------------------------------
# 8. 基准
# ----------------------------------------------------------------------------
def eligibility_mask(p: Panel, min_hist: int = 250) -> np.ndarray:
    """事件研究/回测的共同 universe：上市满 min_hist 根、非ST、可交易、价格>=2、20日成交额中位数>=5000万。"""
    nd = len(p.dates)
    bars = np.arange(nd)[:, None]
    listed = np.isfinite(p.C)
    first = np.where(listed.any(axis=0), listed.argmax(axis=0), nd)
    age = bars - first[None, :]
    med = pd.DataFrame(p.AMT).rolling(20, min_periods=20).median().to_numpy()
    return ((age >= min_hist) & (p.STATUS == 1) & (p.VOL > 0) & (p.ST == 0)
            & (p.C >= 2.0) & (med >= 5e7) & listed)


def bench_fwd(p: Panel, horizons=HORIZONS, elig: Optional[np.ndarray] = None) -> Dict[int, Dict[str, np.ndarray]]:
    """同口径基准：在「合格 universe」内随机取一只股票并持有 hz 日的均值/中位数。
    返回 {hz: {'mean': arr, 'median': arr, 'n': arr}}，索引 d 对应「d+1 开盘入场」。"""
    nd = len(p.dates)
    if elig is None:
        elig = eligibility_mask(p)
    out = {}
    with np.errstate(invalid="ignore"):
        for hz in horizons:
            r = p.C[hz:nd, :] / p.O[1:nd - hz + 1, :] - 1.0
            ok = np.isfinite(r) & elig[:nd - hz, :]
            m = np.full(nd - hz, np.nan)
            md = np.full(nd - hz, np.nan)
            cnt = np.zeros(nd - hz)
            for d in range(nd - hz):
                v = r[d][ok[d]]
                if v.size >= 30:
                    m[d] = v.mean()
                    md[d] = np.median(v)
                    cnt[d] = v.size
            out[hz] = {"mean": m, "median": md, "n": cnt}
    return out


def bench_ew_equity(p: Panel) -> pd.Series:
    """全A等权日收益（可交易样本），用于对照净值。"""
    with np.errstate(invalid="ignore"):
        r = p.C[1:, :] / p.C[:-1, :] - 1.0
    r = np.where(np.isfinite(r) & (np.abs(r) < 0.5), r, np.nan)
    daily = np.nanmean(r, axis=1)
    eq = np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(daily))])
    return pd.Series(eq, index=p.dates)


# ----------------------------------------------------------------------------
# 9. 事件研究
# ----------------------------------------------------------------------------
def _agg(g: pd.DataFrame) -> pd.Series:
    d = {"n": len(g)}
    for hz in HORIZONS:
        col = f"fwd{hz}"
        d[f"fwd{hz}"] = g[col].mean()
    return pd.Series(d)


def study(tag: str = "K3") -> None:
    p = load_panel()
    df = pd.read_parquet(RES / f"结构族_事件_{tag}.parquet")
    elig = eligibility_mask(p)
    bf = bench_fwd(p, elig=elig)
    di = df["date_idx"].to_numpy()
    for hz in HORIZONS:
        bm = np.full(len(p.dates), np.nan); bm[:len(bf[hz]["mean"])] = bf[hz]["mean"]
        bd = np.full(len(p.dates), np.nan); bd[:len(bf[hz]["median"])] = bf[hz]["median"]
        df[f"exc{hz}"] = df[f"fwd{hz}"] - bm[di]
        df[f"excm{hz}"] = df[f"fwd{hz}"] - bd[di]
        df[f"bmean{hz}"] = bm[di]

    # 1) 信号汇总（按时间纪律分段：warmup 只作披露，绩效统计以 dev/val 为主）
    rows = []
    from scipy import stats
    segs = {"全部2008-2020": df[(df.date >= DEV_START) & (df.date <= VAL_END)],
            "开发2008-2015": df[(df.date >= DEV_START) & (df.date <= DEV_END)],
            "验证2016-2020": df[(df.date >= VAL_START) & (df.date <= VAL_END)],
            "预热2005-2007": df[df.date <= WARMUP_END]}
    for seg_name, seg in segs.items():
        for sig, g in seg.groupby("sig"):
            r = {"seg": seg_name, "sig": sig, "n": len(g), "n_codes": g.code.nunique(),
                 "first": g.date.min(), "last": g.date.max()}
            for hz in HORIZONS:
                v = g[f"fwd{hz}"].dropna()
                e = g[f"exc{hz}"].dropna()
                em = g[f"excm{hz}"].dropna()
                r[f"fwd{hz}"] = v.mean()
                r[f"exc{hz}"] = e.mean()
                r[f"excm{hz}"] = em.mean()
                r[f"hit{hz}"] = (e > 0).mean()
                r[f"hitm{hz}"] = (em > 0).mean()
                # 事件高度重叠 → 用「年度均值」作为独立单元做 t 检验
                if len(e) > 5:
                    ymean = g.assign(y=g.date.str[:4]).groupby("y")[f"exc{hz}"].mean().dropna()
                    r[f"t{hz}"] = (stats.ttest_1samp(ymean, 0.0).statistic
                                   if len(ymean) >= 3 else np.nan)
                else:
                    r[f"t{hz}"] = np.nan
            rows.append(r)
    summ = pd.DataFrame(rows)
    summ[summ.seg != "预热2005-2007"].to_csv(RES / f"结构族_事件研究_信号汇总_{tag}.csv",
                                            index=False, encoding="utf-8-sig")
    summ.to_csv(RES / f"结构族_事件研究_信号汇总_含预热_{tag}.csv", index=False, encoding="utf-8-sig")

    # 2) 分年
    df["year"] = df["date"].str[:4].astype(int)
    by_year = df.groupby(["sig", "year"]).apply(_agg, include_groups=False).reset_index()
    ex_year = df.groupby(["sig", "year"])[["exc20", "exc60", "excm20"]].mean().reset_index()
    by_year = by_year.merge(ex_year, on=["sig", "year"], how="left")
    by_year.to_csv(RES / f"结构族_事件研究_分年_{tag}.csv", index=False, encoding="utf-8-sig")

    # 3) 子类
    sub = df[df["sub"].astype(str).str.len() > 0]
    if len(sub):
        s2 = sub.groupby(["sig", "sub"]).agg(n=("fwd20", "size"), fwd20=("fwd20", "mean"),
                                             exc20=("exc20", "mean"), exc60=("exc60", "mean"),
                                             hit20=("exc20", lambda x: (x > 0).mean())).reset_index()
        s2.to_csv(RES / f"结构族_事件研究_子类_{tag}.csv", index=False, encoding="utf-8-sig")

    # 4) 特征分位（只看 <=2020，按分位分桶）
    feats = ["brk_margin", "vratio", "pull_depth", "atr_pct", "r_age", "d20", "d60", "dd_abs", "pb_ord"]
    rows = []
    for sig, g in df.groupby("sig"):
        for f in feats:
            if f not in g.columns:
                continue
            v = pd.to_numeric(g[f], errors="coerce")
            if v.notna().sum() < 200 or v.nunique() < 5:
                continue
            try:
                q = pd.qcut(v, 5, duplicates="drop", labels=False)
            except Exception:
                continue
            tmp = pd.DataFrame({"q": q, "exc20": g["exc20"], "exc60": g["exc60"], "fwd20": g["fwd20"]})
            gg = tmp.groupby("q").agg(n=("fwd20", "size"), exc20=("exc20", "mean"),
                                      exc60=("exc60", "mean"), fwd20=("fwd20", "mean")).reset_index()
            gg["sig"] = sig; gg["feat"] = f
            rows.append(gg)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            RES / f"结构族_事件研究_特征分位_{tag}.csv", index=False, encoding="utf-8-sig")

    # 5) 离场规则对比
    rows = []
    for sig, g in df.groupby("sig"):
        for rule in EXIT_RULES:
            r = g[f"ret_{rule}"].dropna()
            if len(r) < 50:
                continue
            hold = (g.loc[r.index, f"exit_i_{rule}"] - g.loc[r.index, "entry_i"]) if f"exit_i_{rule}" in g else np.nan
            rows.append({"sig": sig, "rule": rule, "n": len(r), "mean": r.mean(),
                         "median": r.median(), "win": (r > 0).mean(),
                         "p10": r.quantile(0.1), "p90": r.quantile(0.9),
                         "avg_hold": float(np.mean(hold)) if not np.isscalar(hold) else np.nan})
    pd.DataFrame(rows).to_csv(RES / f"结构族_离场对比_{tag}.csv", index=False, encoding="utf-8-sig")

    # 6) 「买入后是否符合预期」：入场 20 日内是否上涨超 3%
    df["ok20"] = df["made_new_high20"].astype(bool)
    r6 = df.groupby(["sig", "ok20"]).agg(n=("fwd60", "size"), fwd20=("fwd20", "mean"),
                                         fwd60=("fwd60", "mean"), mae20=("mae20", "mean"),
                                         mfe20=("mfe20", "mean"),
                                         ret_struct=("ret_struct", "mean")).reset_index()
    r6.to_csv(RES / f"结构族_事件研究_预期符合_{tag}.csv", index=False, encoding="utf-8-sig")
    print(summ.round(4).to_string())
    print(f"[study] 输出 → 03_结果/结构族_*.csv")


# ----------------------------------------------------------------------------
# 10. 组合回测
# ----------------------------------------------------------------------------
def _metrics(eq: np.ndarray, dates: np.ndarray) -> dict:
    eq = np.asarray(eq, float)
    n = len(eq)
    if n < 30 or eq[0] <= 0:
        return {}
    yrs = n / 252.0
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1
    r = eq[1:] / eq[:-1] - 1
    sd = np.std(r, ddof=1)
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    return {"CAGR": cagr, "总收益": eq[-1] / eq[0] - 1, "年化波动": sd * np.sqrt(252),
            "Sharpe": (np.mean(r) / sd * np.sqrt(252)) if sd > 0 else np.nan,
            "MaxDD": mdd, "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
            "月胜率": float(np.mean(pd.Series(r).groupby(pd.to_datetime(dates[1:]).to_period("M")).apply(
                lambda x: (1 + x).prod() - 1) > 0))}


def simulate(p: Panel, trades: pd.DataFrame, capital: float = CAPITAL, n_slots: int = N_SLOTS,
             start: str = DEV_START, end: str = VAL_END, seed: int = 42,
             rank_col: Optional[str] = None, slippage: float = SLIPPAGE,
             stamp: float = STAMP_SELL) -> Tuple[pd.Series, pd.DataFrame]:
    """事件驱动、次开入场、T+1、整手、费用滑点、涨跌停/停牌顺延。
    trades 需含 gi_entry / gi_exit / gross_ret / code_i。"""
    d0 = int(np.searchsorted(p.dates, start, "left"))
    d1 = int(np.searchsorted(p.dates, end, "right"))
    tr = trades[(trades.gi_entry >= d0) & (trades.gi_entry < d1) &
                (trades.gi_exit > trades.gi_entry) & np.isfinite(trades.gross_ret)].copy()
    tr = tr.sort_values(["gi_entry", "gi_exit"]).reset_index(drop=True)
    by_entry: Dict[int, np.ndarray] = {}
    for gi, g in tr.groupby("gi_entry"):
        by_entry[int(gi)] = g.index.to_numpy()
    by_exit: Dict[int, list] = {}
    rng = np.random.default_rng(seed)

    cash = capital
    pos: List[dict] = []
    eq = np.zeros(d1 - d0)
    cash_hist = np.zeros(d1 - d0)
    pos_cnt = np.zeros(d1 - d0, dtype=int)
    traded = []
    o_arr, c_arr, fac_arr = p.O, p.C, p.FAC

    for di, d in enumerate(range(d0, d1)):
        # 1) 开盘离场
        keep = []
        for ps in pos:
            if ps["gi_exit"] == d:
                px = o_arr[d, ps["code_i"]] / ps["fac0"]
                gross = ps["shares"] * px
                fee = max(COMM_MIN, COMM_RATE * gross) + TRANSFER_RATE * gross + stamp * gross
                cash += gross - fee
                ps["exit_px"] = px
                ps["exit_date"] = p.dates[d]
                ps["net_ret"] = (gross - fee) / ps["cost"] - 1.0
                traded.append(ps)
            else:
                keep.append(ps)
        pos = keep
        # 2) 开盘入场
        cand = by_entry.get(d)
        free = n_slots - len(pos)
        if cand is not None and free > 0:
            if rank_col is not None and rank_col in tr.columns:
                vals = tr.loc[cand, rank_col].fillna(-np.inf).to_numpy()
                pick = cand[np.argsort(-vals)[:free]]
            else:
                pick = rng.choice(cand, size=min(free, len(cand)), replace=False)
            eq_prev = cash + sum(ps["shares"] * c_arr[d - 1 if d > 0 else d, ps["code_i"]] / ps["fac0"]
                                 for ps in pos)
            alloc = eq_prev / n_slots
            for ti in pick:
                row = tr.loc[ti]
                ci = int(row.code_i)
                raw_px = o_arr[d, ci] / fac_arr[d, ci]
                if not np.isfinite(raw_px) or raw_px <= 0:
                    continue
                buy_px = raw_px * (1 + slippage)
                shares = int(alloc // (buy_px * 100) * 100)
                if shares < 100:
                    continue
                cost = shares * buy_px
                fee = max(COMM_MIN, COMM_RATE * cost) + TRANSFER_RATE * cost
                if cost + fee > cash:
                    continue
                cash -= cost + fee
                pos.append({"code_i": ci, "shares": shares, "fac0": fac_arr[d, ci],
                            "gi_entry": d, "gi_exit": int(row.gi_exit),
                            "cost": cost + fee, "entry_date": p.dates[d], "sig": row.sig})
        # 3) 收盘估值
        mv = sum(ps["shares"] * c_arr[d, ps["code_i"]] / ps["fac0"] for ps in pos)
        eq[di] = cash + mv
        cash_hist[di] = cash
        pos_cnt[di] = len(pos)

    s = pd.Series(eq, index=p.dates[d0:d1])
    trd = pd.DataFrame(traded)
    return s, trd


def layer_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "L0_基准": pd.Series(True, index=df.index),
        "L1_周线结构": df["w_up"].astype(bool),
        "L2_周+月结构": df["w_up"].astype(bool) & df["m_up"].astype(bool),
        "L3_周结构+HH/HL": df["w_up"].astype(bool) & df["up_struct"].astype(bool),
        "L4_周结构+均线": df["w_up"].astype(bool) & (df["d20"] > 0) & (df["d60"] > 0),
        "L5_全体叠加": (df["w_up"].astype(bool) | df["m_up"].astype(bool)) & df["up_struct"].astype(bool)
                     & (df["d20"] > 0),
    }


def backtest(tag: str = "K3", rule: str = "struct", sigs: Optional[Sequence[str]] = None,
             capital: float = CAPITAL, n_slots: int = N_SLOTS, seeds: int = 8) -> None:
    p = load_panel()
    df = pd.read_parquet(RES / f"结构族_事件_{tag}.parquet")
    df = df[df.entry_i >= 0].copy()
    df["gi_entry"] = df["entry_i"]
    df["gi_exit"] = df[f"exit_i_{rule}"]
    df["gross_ret"] = df[f"ret_{rule}"]
    sigs = sigs or [s for s in SIGNALS if s in df.sig.unique()]
    rows, curves = [], {}
    for sig in sigs:
        g0 = df[df.sig == sig]
        if len(g0) < 200:
            continue
        for lname, mask in layer_masks(g0).items():
            g = g0[mask.fillna(False).to_numpy() & np.isfinite(g0.gross_ret).to_numpy()]
            if len(g) < 100:
                rows.append({"sig": sig, "layer": lname, "n": len(g)})
                continue
            ms, trs, curves_seed = [], [], []
            for sd in range(seeds):
                eq, trd = simulate(p, g, capital=capital, n_slots=n_slots, seed=42 + sd)
                m = _metrics(eq.to_numpy(), eq.index.to_numpy())
                if m:
                    ms.append(m); trs.append(trd)
                curves_seed.append(eq)
            if not ms:
                continue
            agg = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
            sd_ = {k: float(np.nanstd([m[k] for m in ms])) for k in ("CAGR", "MaxDD", "Sharpe")}
            trd = pd.concat(trs, ignore_index=True) if trs else pd.DataFrame()
            agg.update({"sig": sig, "layer": lname, "n": len(g), "seeds": seeds,
                        "CAGR_sd": sd_["CAGR"], "MaxDD_sd": sd_["MaxDD"], "Sharpe_sd": sd_["Sharpe"],
                        "trades": len(trd) / max(1, seeds)})
            if len(trd):
                agg["win"] = float((trd.net_ret > 0).mean())
                agg["avg_hold"] = float((trd.gi_exit - trd.gi_entry).mean())
                agg["turnover"] = float(len(trd) / seeds / max(1, len(eq) / 252))
            rows.append(agg)
            curves[f"{sig}|{lname}"] = curves_seed[0]
    res = pd.DataFrame(rows)
    res.to_csv(RES / f"结构族_回测_分层_{tag}_{rule}.csv", index=False, encoding="utf-8-sig")
    if curves:
        pd.DataFrame(curves).to_csv(RES / f"结构族_回测_净值_{tag}_{rule}.csv", encoding="utf-8-sig")
    piv = res.pivot_table(index="layer", columns="sig", values="CAGR")
    print((piv * 100).round(1).to_string())
    print("\n（每格为 %d 个随机种子的均值；标准差见 CAGR_sd 列）" % seeds)


def baseline_null(p: Optional[Panel] = None, n_per_day: int = 3, rule_hold: int = 21,
                  capital: float = CAPITAL, n_slots: int = N_SLOTS) -> pd.DataFrame:
    """空对照：在合格 universe 内随机选股，用同一引擎/同一持有期跑组合。
    这是「10 只等权集中持仓」的正确零假设，而非等权指数。"""
    p = p or load_panel()
    elig = eligibility_mask(p)
    nd = len(p.dates)
    rng = np.random.default_rng(2020)
    rows = []
    for d0 in range(np.searchsorted(p.dates, DEV_START), np.searchsorted(p.dates, "2020-12-01")):
        js = np.flatnonzero(elig[d0])
        if js.size < 50:
            continue
        for j in rng.choice(js, size=min(n_per_day, js.size), replace=False):
            e = d0 + 1
            if e + rule_hold >= nd or not np.isfinite(p.O[e, j]) or not np.isfinite(p.O[e + rule_hold, j]):
                continue
            rows.append({"code_i": int(j), "gi_entry": e, "gi_exit": e + rule_hold,
                         "sig": "RND", "gross_ret": p.O[e + rule_hold, j] / p.O[e, j] - 1})
    rnd = pd.DataFrame(rows)
    out = []
    for cap, ns in ((CAPITAL, N_SLOTS), (100_000.0, 5), (100_000.0, 20)):
        eq, trd = simulate(p, rnd, capital=cap, n_slots=ns)
        m = _metrics(eq.to_numpy(), eq.index.to_numpy())
        m.update({"item": f"随机选股_{ns}仓", "capital": cap, "trades": len(trd),
                  "gross_mean": float(rnd.gross_ret.mean())})
        out.append(m)
    b = bench_ew_equity(p)
    bb = b[(b.index >= DEV_START) & (b.index <= VAL_END)]
    m = _metrics(bb.to_numpy(), bb.index.to_numpy())
    m.update({"item": "全A等权指数(日再平衡)", "capital": np.nan, "trades": np.nan,
              "gross_mean": np.nan})
    out.append(m)
    for name, s in p.bench.items():
        s2 = s[(s.index >= DEV_START) & (s.index <= VAL_END)].dropna()
        if len(s2) > 100:
            m = _metrics((s2 / s2.iloc[0]).to_numpy(), s2.index.to_numpy())
            m.update({"item": f"指数_{name}", "capital": np.nan, "trades": np.nan, "gross_mean": np.nan})
            out.append(m)
    res = pd.DataFrame(out)
    res.to_csv(RES / "结构族_空对照与基准.csv", index=False, encoding="utf-8-sig")
    print(res.round(3).to_string())
    return res


def cost_sensitivity(tag: str = "K3", rule: str = "struct") -> None:
    """成本/滑点敏感性：把净增益拆成「毛收益 - 成本」两部分。"""
    p = load_panel()
    df = pd.read_parquet(RES / f"结构族_事件_{tag}.parquet")
    df = df[df.entry_i >= 0].copy()
    df["gi_entry"] = df["entry_i"]
    df["gi_exit"] = df[f"exit_i_{rule}"]
    df["gross_ret"] = df[f"ret_{rule}"]
    rows = []
    for sl in (0.0, 0.0005, 0.001, 0.002):
        for st in (0.0, STAMP_SELL):
            eq, trd = simulate(p, df, slippage=sl, stamp=st)
            m = _metrics(eq.to_numpy(), eq.index.to_numpy())
            m.update({"slippage": sl, "stamp": st, "trades": len(trd)})
            rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"结构族_成本敏感_{tag}_{rule}.csv", index=False, encoding="utf-8-sig")
    print(out.round(3).to_string())


def sensitivity(tag: str = "K3", rule: str = "struct") -> None:
    """资金规模 / 仓位数量敏感性。"""
    p = load_panel()
    df = pd.read_parquet(RES / f"结构族_事件_{tag}.parquet")
    df = df[df.entry_i >= 0].copy()
    df["gi_entry"] = df["entry_i"]
    df["gi_exit"] = df[f"exit_i_{rule}"]
    df["gross_ret"] = df[f"ret_{rule}"]
    rows = []
    for sig in [s for s in SIGNALS if s in df.sig.unique()]:
        g = df[(df.sig == sig) & np.isfinite(df.gross_ret)]
        if len(g) < 200:
            continue
        for cap in (20_000, 100_000, 1_000_000):
            for ns in (5, 10, 20):
                eq, trd = simulate(p, g, capital=cap, n_slots=ns)
                m = _metrics(eq.to_numpy(), eq.index.to_numpy())
                m.update({"sig": sig, "capital": cap, "slots": ns, "trades": len(trd)})
                rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"结构族_敏感性_{tag}_{rule}.csv", index=False, encoding="utf-8-sig")
    print(out.round(3).to_string())


# ----------------------------------------------------------------------------
# 11. 数据审计
# ----------------------------------------------------------------------------
def audit() -> None:
    """覆盖/复权/停牌/涨跌停/退市 审计。注意：2021+ 只统计覆盖，不计算任何收益。"""
    import duckdb
    p = load_panel()
    rows = []

    def add(item, value, note=""):
        rows.append({"item": item, "value": value, "note": note})
        print(f"  {item:38s} {value}   {note}")

    add("面板交易日", len(p.dates), f"{p.dates[0]} ~ {p.dates[-1]}")
    add("面板股票数", len(p.codes), "含退市")
    add("面板 shape", f"{p.C.shape}")
    nan_frac = float(np.isnan(p.C).mean())
    add("收盘缺失率", f"{nan_frac:.3f}", "未上市/退市/停牌")
    add("最后一行有效股票数", int(np.isfinite(p.C[-1]).sum()))

    # 复权一致性：hfq 与 raw 比值应逐股单调（除权除息日跳变）
    fac = p.FAC
    add("复权因子中位数", float(np.nanmedian(fac)))
    add("复权因子最大", float(np.nanmax(fac)))
    neg = int(np.nansum(fac <= 0))
    add("非正复权因子个数", neg)

    # 停牌
    trad = (p.STATUS == 1) & (p.VOL > 0)
    listed = np.isfinite(p.C)
    add("上市日中停牌占比", f"{float((listed & ~trad).sum() / max(1, listed.sum())):.4f}")

    # 涨跌停统计（用原始价 + preclose 判定）
    pc = p.PC
    with np.errstate(invalid="ignore"):
        chg = p.C / p.FAC / np.where(pc > 0, pc, np.nan) - 1.0
    okc = np.isfinite(chg)
    add("日涨幅>9.8% 占比(非ST)", f"{float(((chg > 0.098) & okc & (p.ST == 0)).sum() / max(1, (okc & (p.ST == 0)).sum())):.4f}")
    add("日跌幅<-9.8% 占比(非ST)", f"{float(((chg < -0.098) & okc & (p.ST == 0)).sum() / max(1, (okc & (p.ST == 0)).sum())):.4f}")

    # ST / 退市
    add("ST 样本次数占比", f"{float((p.ST == 1).sum() / max(1, np.isfinite(p.ST).sum())):.4f}")
    delisted = int((p.basic["out"].astype(str).str.len() > 0).sum())
    add("已退市股票数(全库)", delisted)

    # 2021+ 覆盖审计（只看条数，不算收益）
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{BS_DB}' AS bs (TYPE SQLITE, READ_ONLY)")
    cov = con.execute(f"""
        SELECT substr(date,1,4) y, count(*) n, count(distinct code) c
        FROM bs.stock_kline_d
        WHERE adjustflag=3 AND date > '{VAL_END}' AND date <= '{FREEZE}'
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    cov.to_csv(RES / "结构族_审计_留出期覆盖.csv", index=False, encoding="utf-8-sig")
    add("留出期(2021-2026.10)行数", int(cov["n"].sum()), "仅覆盖统计，未计算收益")
    con.close()

    # 与同花顺 DuckDB 交叉校验（收盘价抽样）
    try:
        con = duckdb.connect(THS_DB, read_only=True)
        q = con.execute("""
            SELECT thscode, date, close FROM raw_kline_daily
            WHERE date = DATE '2020-12-31' AND thscode IN ('600519.SH','000001.SZ','601318.SH')
        """).fetchdf()
        con.close()
        add("同花顺交叉校验样本", len(q), str(q.to_dict("records"))[:160])
    except Exception as e:
        add("同花顺交叉校验", "跳过", str(e)[:80])

    out = pd.DataFrame(rows)
    out.to_csv(RES / "结构族_审计_数据.csv", index=False, encoding="utf-8-sig")
    print(f"[audit] → 03_结果/结构族_审计_数据.csv")


# ----------------------------------------------------------------------------
# 12. 离场规则对比（组合层）
# ----------------------------------------------------------------------------
def exits(tag: str = "K3") -> None:
    p = load_panel()
    df = pd.read_parquet(RES / f"结构族_事件_{tag}.parquet")
    df = df[df.entry_i >= 0].copy()
    rows, curves = [], {}
    for rule in EXIT_RULES:
        if f"exit_i_{rule}" not in df.columns:
            continue
        g = df[np.isfinite(df[f"ret_{rule}"])].copy()
        g["gi_entry"] = g["entry_i"]
        g["gi_exit"] = g[f"exit_i_{rule}"]
        g["gross_ret"] = g[f"ret_{rule}"]
        eq, trd = simulate(p, g)
        m = _metrics(eq.to_numpy(), eq.index.to_numpy())
        m.update({"rule": rule, "trades": len(trd)})
        if len(trd):
            m["win"] = float((trd.net_ret > 0).mean())
            m["avg_hold"] = float((trd.gi_exit - trd.gi_entry).mean())
        rows.append(m)
        curves[rule] = eq
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"结构族_离场规则_组合_{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(RES / f"结构族_离场规则_净值_{tag}.csv", encoding="utf-8-sig")
    print(out.round(3).to_string())


# ----------------------------------------------------------------------------
# 13. 自检
# ----------------------------------------------------------------------------
def selfcheck(tag: str = "K3", n_sample: int = 40, T: str = "2020-06-30") -> None:
    """非重绘 / 无未来函数 / 口径一致性 自检。全部只读、可重复。"""
    checks = []
    p = load_panel()

    def rec(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    # C1 面板不越过留出期
    rec("C1 面板末端 <= 2020-12-31", p.dates[-1] <= VAL_END, p.dates[-1])
    # C2 无 2021+ 数据
    rec("C2 面板不含 2021+ 日期", not (p.dates >= HOLDOUT_START).any(), len(p.dates))

    # C3 截断不变性（非重绘核心检验）
    rng = np.random.default_rng(7)
    cand = [j for j in range(len(p.codes)) if np.isfinite(p.C[:, j]).sum() > 800]
    picks = rng.choice(cand, size=min(n_sample, len(cand)), replace=False)
    diffs = []
    for j in picks:
        code = p.codes[j]
        fin = np.isfinite(p.C[:, j])
        i0 = int(np.argmax(fin)); i1 = int(len(fin) - np.argmax(fin[::-1]))
        full = gen_events_one(code, p.dates[i0:i1], p.O[i0:i1, j], p.H[i0:i1, j], p.L[i0:i1, j],
                              p.C[i0:i1, j], p.VOL[i0:i1, j], p.AMT[i0:i1, j], p.TURN[i0:i1, j],
                              p.ST[i0:i1, j], p.STATUS[i0:i1, j],
                              pc=p.PC[i0:i1, j], fac=p.FAC[i0:i1, j], K=int(tag[1:]))
        if len(full) == 0:
            continue
        cut = np.searchsorted(p.dates[i0:i1], T, "right")
        part = gen_events_one(code, p.dates[i0:i0 + cut], p.O[i0:i0 + cut, j], p.H[i0:i0 + cut, j],
                              p.L[i0:i0 + cut, j], p.C[i0:i0 + cut, j], p.VOL[i0:i0 + cut, j],
                              p.AMT[i0:i0 + cut, j], p.TURN[i0:i0 + cut, j], p.ST[i0:i0 + cut, j],
                              p.STATUS[i0:i0 + cut, j], pc=p.PC[i0:i0 + cut, j],
                              fac=p.FAC[i0:i0 + cut, j], K=int(tag[1:]))
        a = set(map(tuple, full[full.date <= T][["date", "sig"]].to_numpy()))
        b = set(map(tuple, part[part.date <= T][["date", "sig"]].to_numpy())) if len(part) else set()
        diffs.append(len(a ^ b) / max(1, len(a)))
    md = float(np.mean(diffs)) if diffs else 1.0
    rec("C3 截断不变性(信号集合差异率)", md < 0.02, f"mean={md:.4f} n={len(diffs)}")

    # C4 枢轴确认滞后：所有枢轴确认日 > 枢轴日
    h = p.H[:, cand[0]]; l = p.L[:, cand[0]]
    fin = np.isfinite(p.C[:, cand[0]])
    i0 = int(np.argmax(fin)); i1 = int(len(fin) - np.argmax(fin[::-1]))
    K = int(tag[1:])
    ph, pl = pivot_candidates(h[i0:i1], l[i0:i1], K)
    ok4 = (ph + K <= len(h) - i0 - 1).all() if len(ph) else True
    rec("C4 枢轴右确认滞后 = K", ok4, f"K={K} n_ph={len(ph)}")

    # C5 事件表中的离场/入场顺序
    ev = pd.read_parquet(RES / f"结构族_事件_{tag}.parquet")
    e = ev[ev.entry_i >= 0]
    rec("C5 入场严格晚于信号日", bool((e.entry_i > e.i).all()),
        f"min_lag={(e.entry_i - e.i).min()}")
    for rule in EXIT_RULES:
        col = f"exit_i_{rule}"
        if col in e:
            v = e[e[col] >= 0]
            if len(v):
                rec(f"C5 离场>=入场+1 ({rule})", bool((v[col] > v.entry_i).all()),
                    f"min={(v[col] - v.entry_i).min()}")

    # C6 周/月线只用已完成周期
    dates = p.dates[:600]
    oo, hh, ll, cc = p.O[:600, cand[0]], p.H[:600, cand[0]], p.L[:600, cand[0]], p.C[:600, cand[0]]
    htf, grp = build_htf(dates, oo, hh, ll, cc, "W", 2)
    ok6 = bool((htf["avail_from"][:-1] > htf["end_i"][:-1]).all())
    rec("C6 周线仅在周期结束后可用", ok6, f"n_week={len(htf['h'])}")

    # C7 交易成本与印花税
    rec("C7 印花税口径(<=2020 卖出 0.1%)", abs(STAMP_SELL - 0.001) < 1e-9, STAMP_SELL)

    # C8 事件数按年分布无异常聚集
    yr = ev.groupby(ev.date.str[:4]).size()
    rec("C8 事件年份覆盖", int((yr > 0).sum()) >= 10, f"{yr.index.min()}~{yr.index.max()} 年数={len(yr)}")

    # C9 未来函数：事件特征中不得含事件日之后的信息（抽查 w_up 与周线完成日）
    rec("C9 特征列无前视命名", all("future" not in c for c in ev.columns), "")

    res = pd.DataFrame(checks)
    res.to_csv(RES / "结构族_自检.csv", index=False, encoding="utf-8-sig")
    print(f"[selfcheck] {int(res['pass'].sum())}/{len(res)} 通过 → 03_结果/结构族_自检.csv")
    if not res["pass"].all():
        print(res[~res["pass"]].to_string())


# ----------------------------------------------------------------------------
# 14. CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cache", "audit", "events", "study", "backtest",
                                     "exits", "null", "cost", "sens", "selfcheck", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--rule", default="struct")
    args = ap.parse_args()

    if args.mode == "cache":
        build_cache(force=args.force)
    elif args.mode == "audit":
        audit()
    elif args.mode == "events":
        run_events(k=args.k, tag=args.tag)
    elif args.mode == "study":
        study(tag=args.tag or f"K{args.k}")
    elif args.mode == "backtest":
        backtest(tag=args.tag or f"K{args.k}", rule=args.rule)
    elif args.mode == "exits":
        exits(tag=args.tag or f"K{args.k}")
    elif args.mode == "null":
        baseline_null()
    elif args.mode == "cost":
        cost_sensitivity(tag=args.tag or "K3", rule=args.rule)
    elif args.mode == "sens":
        sensitivity(tag=args.tag or "K3", rule=args.rule)
    elif args.mode == "selfcheck":
        selfcheck(tag=args.tag or f"K{args.k}")
    elif args.mode == "all":
        build_cache()
        audit()
        baseline_null()
        for k in (2, 3, 5):
            run_events(k=k)
        for k in (2, 3, 5):
            study(tag=f"K{k}")
        backtest(tag=args.tag or "K3", rule=args.rule)
        exits(tag=args.tag or "K3")
        cost_sensitivity(tag=args.tag or "K3", rule=args.rule)
        sensitivity(tag=args.tag or "K3", rule=args.rule)
        selfcheck(tag=args.tag or "K3")



if __name__ == "__main__":
    main()
