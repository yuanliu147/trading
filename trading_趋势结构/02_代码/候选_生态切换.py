#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选族：截面生态 / 反转×趋势（连续风险预算，无硬多空门）。

研究问题
--------
在连续风险预算 g(状态) ∈ [0.25, 1.0] 下：
  REV = 强趋势中的短期反转（趋势池内 r5 最低 20%）
  BRK = 强突破（收盘创 60 日新高 + 量能确认）
哪一个在不同截面生态（宽度 B / 离散度 D / 波动 V）下有净增益？
用单 sleeve、固定双 sleeve、状态倾斜双 sleeve、槽位竞争组织，哪个更稳？

纪律
----
* 数据只读冻结层；SQL 硬截断 2020-12-31，2021+ 不加载。
* 信号 t 收盘生成，t+1 开盘成交；T+1、整股、涨跌停、停牌、历史费用、滑点。
* 预注册：03_结果/候选_生态切换/00_预注册_生态切换.md（写于任何收益之前）。
* 试验清单：03_结果/候选_生态切换/07_试验清单.csv（本族 14 个配置）。

用法
----
    python3 02_代码/候选_生态切换.py panel       # 构建/复用面板缓存
    python3 02_代码/候选_生态切换.py features    # 特征 + 状态
    python3 02_代码/候选_生态切换.py events      # 事件研究与条件边际
    python3 02_代码/候选_生态切换.py portfolio   # 10 个配置 + 4 个敏感性
    python3 02_代码/候选_生态切换.py account     # 统一账户引擎复核
    python3 02_代码/候选_生态切换.py all         # 全流程
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "03_结果" / "候选_生态切换"
CACHE = OUT / "_cache"
DB = ROOT / "01_数据" / "frozen" / "frozen.duckdb"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FREEZE = "2026-10-26"
PANEL_START = "2004-01-01"
ANALYSIS_END = "2020-12-31"
DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")

# 先验参数（预注册 §4-§6）
N_SLOTS = 10
H_DEFAULT = 10
COST_ROUNDTRIP = 0.0035          # 事件级：费用 0.15% + 滑点 0.20%
SLIP_BPS = 10.0                  # 组合级：单边 10bp
ADV_MIN = 3e7
PRICE_MIN = 2.0
LIST_DAYS_MIN = 120
DISP_WINSOR = (0.01, 0.99)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1. 面板
# --------------------------------------------------------------------------- #

@dataclass
class Panel:
    dates: np.ndarray                 # 'U10'
    codes: np.ndarray                 # 'U12'
    m: dict[str, np.ndarray]          # float32 (n_codes, n_days)
    tradable: np.ndarray
    is_st: np.ndarray
    lim_up_open: np.ndarray
    lim_dn_open: np.ndarray
    delist_i: np.ndarray              # (n_codes,) 首个 >= delist_date 的索引；无退市=n_days
    ipo_date: dict[str, date] = field(default_factory=dict)
    out_date: dict[str, date] = field(default_factory=dict)

    @property
    def n_codes(self) -> int:
        return len(self.codes)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def di(self, d: str) -> int:
        return int(np.searchsorted(self.dates, d, side="left"))


def build_panel(force: bool = False) -> Panel:
    import duckdb

    CACHE.mkdir(parents=True, exist_ok=True)
    npz = CACHE / "panel.npz"
    if npz.exists() and not force:
        z = np.load(npz, allow_pickle=False)
        p = Panel(
            dates=z["dates"].astype("U10"),
            codes=z["codes"].astype("U12"),
            m={k: z[k] for k in z.files if k.startswith("m_")} | {},
            tradable=z["tradable"], is_st=z["is_st"],
            lim_up_open=z["lim_up_open"], lim_dn_open=z["lim_dn_open"],
            delist_i=z["delist_i"],
        )
        p.m = {k[2:]: v for k, v in p.m.items()}
        meta = json.loads((CACHE / "panel_meta.json").read_text())
        p.ipo_date = {k: date.fromisoformat(str(v)[:10]) for k, v in meta["ipo"].items()}
        p.out_date = {k: date.fromisoformat(str(v)[:10]) for k, v in meta["out"].items()}
        log(f"面板缓存：{p.n_codes} 只 × {p.n_days} 日（{p.dates[0]}..{p.dates[-1]}）")
        return p

    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET memory_limit='5GB'")
    dates = con.execute(
        f"SELECT DISTINCT date FROM daily WHERE board='主板' AND date BETWEEN "
        f"DATE '{PANEL_START}' AND DATE '{ANALYSIS_END}' ORDER BY 1"
    ).df()["date"].astype(str).to_numpy()
    codes = con.execute(
        f"""SELECT symbol, count(*) n FROM daily
            WHERE board='主板' AND date BETWEEN DATE '{PANEL_START}' AND DATE '{ANALYSIS_END}'
            GROUP BY 1 HAVING count(*) >= 260 ORDER BY 1"""
    ).df()["symbol"].astype(str).to_numpy()
    sym = con.execute("SELECT symbol, ipo_date, delist_date FROM symbols").df()
    ipo = {r.symbol: r.ipo_date for r in sym.itertuples() if pd.notna(r.ipo_date)}
    out = {r.symbol: r.delist_date for r in sym.itertuples() if pd.notna(r.delist_date)}

    nd, nc = len(dates), len(codes)
    log(f"面板 {nc} 只 × {nd} 日；读取冻结层 ...")
    fields = ["o", "h", "l", "c", "pc", "vol", "amt", "f", "td", "pct", "fmc"]
    m = {k: np.full((nc, nd), np.nan, dtype=np.float32) for k in fields}
    tradable = np.zeros((nc, nd), dtype=bool)
    is_st = np.zeros((nc, nd), dtype=bool)
    lim_up_open = np.zeros((nc, nd), dtype=bool)
    lim_dn_open = np.zeros((nc, nd), dtype=bool)

    d_idx = {d: i for i, d in enumerate(dates)}
    c_idx = {c: i for i, c in enumerate(codes)}
    q = f"""
        SELECT symbol, date, open, high, low, close, preclose, volume, amount,
               trade_status, is_st, td_since_list, pct_chg, factor,
               is_limit_up_open, is_limit_down_open, float_mktcap
        FROM daily
        WHERE board='主板' AND date BETWEEN DATE '{PANEL_START}' AND DATE '{ANALYSIS_END}'
    """
    cur = con.execute(q)
    n_read = 0
    t0 = time.time()
    while True:
        chunk = cur.fetch_df_chunk(64)
        if len(chunk) == 0:
            break
        ci = chunk["symbol"].map(c_idx)
        di = chunk["date"].astype(str).map(d_idx)
        ok = ci.notna() & di.notna()
        if not ok.any():
            continue
        ci = ci[ok].to_numpy(np.int64)
        di = di[ok].to_numpy(np.int64)
        df = chunk[ok]
        for name, col in [("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"),
                          ("pc", "preclose"), ("vol", "volume"), ("amt", "amount"),
                          ("f", "factor"), ("td", "td_since_list"), ("pct", "pct_chg"), ("fmc", "float_mktcap")]:
            m[name][ci, di] = df[col].to_numpy(np.float32)
        vol = df["volume"].to_numpy(np.float64)
        tradable[ci, di] = (df["trade_status"].to_numpy() == 1) & (vol > 0)
        is_st[ci, di] = df["is_st"].to_numpy() == 1
        lim_up_open[ci, di] = df["is_limit_up_open"].fillna(False).to_numpy(bool)
        lim_dn_open[ci, di] = df["is_limit_down_open"].fillna(False).to_numpy(bool)
        n_read += len(df)
    con.close()
    log(f"读入 {n_read:,} 行，{time.time()-t0:.1f}s")

    # 复权因子：缺失处前向填充（停牌/首日）
    f = m["f"]
    for i in range(nc):
        row = pd.Series(f[i]).ffill().bfill().to_numpy(np.float32).copy()
        row[~np.isfinite(row)] = 1.0
        f[i] = row
    m["f"] = f

    delist_i = np.full(nc, nd, dtype=np.int64)
    for i, c in enumerate(codes):
        od = out.get(c)
        if od is not None:
            j = int(np.searchsorted(dates, str(od), side="left"))
            delist_i[i] = min(j, nd)

    np.savez(
        npz, dates=dates.astype("U10"), codes=codes.astype("U12"),
        tradable=tradable, is_st=is_st, lim_up_open=lim_up_open, lim_dn_open=lim_dn_open,
        delist_i=delist_i,
        **{f"m_{k}": v for k, v in m.items()},
    )
    (CACHE / "panel_meta.json").write_text(json.dumps({
        "ipo": {k: v.isoformat() for k, v in ipo.items()},
        "out": {k: v.isoformat() for k, v in out.items()},
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(n_read),
    }, ensure_ascii=False))
    log(f"面板已写入 {npz}")
    p = Panel(dates=dates, codes=codes, m=m, tradable=tradable, is_st=is_st,
              lim_up_open=lim_up_open, lim_dn_open=lim_dn_open, delist_i=delist_i)
    p.ipo_date = ipo
    p.out_date = out
    return p


# --------------------------------------------------------------------------- #
# 2. 特征与状态
# --------------------------------------------------------------------------- #

def _roll_mean(a: np.ndarray, n: int, minp: int | None = None) -> np.ndarray:
    """沿时间轴（axis=1）的 NaN 感知滚动均值，要求窗口内有 minp 个有效值。"""
    minp = n if minp is None else minp
    v = np.isfinite(a)
    x = np.where(v, a, 0.0).astype(np.float64)
    cs = np.cumsum(x, axis=1)
    cn = np.cumsum(v, axis=1)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    out[:, n - 1:] = cs[:, n - 1:]
    out[:, n:] -= cs[:, :-n]
    cnt = np.empty(a.shape, dtype=np.int32)
    cnt[:, n - 1:] = cn[:, n - 1:]
    cnt[:, n:] -= cn[:, :-n]
    out[cnt < minp] = np.nan
    out /= n
    return out.astype(np.float32)


def _roll_std(a: np.ndarray, n: int, minp: int | None = None) -> np.ndarray:
    minp = n if minp is None else minp
    v = np.isfinite(a)
    x = np.where(v, a, 0.0).astype(np.float64)
    cs = np.cumsum(x, axis=1)
    cs2 = np.cumsum(x * x, axis=1)
    cn = np.cumsum(v, axis=1)
    s = np.full(a.shape, np.nan)
    s2 = np.full(a.shape, np.nan)
    c = np.full(a.shape, np.nan)
    s[:, n - 1:] = cs[:, n - 1:]
    s[:, n:] -= cs[:, :-n]
    s2[:, n - 1:] = cs2[:, n - 1:]
    s2[:, n:] -= cs2[:, :-n]
    c[:, n - 1:] = cn[:, n - 1:]
    c[:, n:] -= cn[:, :-n]
    mean = s / c
    var = np.maximum(s2 / c - mean * mean, 0.0)
    out = np.sqrt(var)
    out[c < minp] = np.nan
    return out.astype(np.float32)


def _roll_max(a: np.ndarray, n: int) -> np.ndarray:
    df = pd.DataFrame(a.T)
    return df.rolling(n, min_periods=n).max().to_numpy(np.float32).T


def _shift(a: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    out = np.full_like(a, fill, dtype=np.float32)
    if n == 0:
        return a.copy()
    out[:, n:] = a[:, :-n]
    return out


def _wilder_atr(hh, ll, cc, n=14):
    df_tr = pd.DataFrame(np.maximum.reduce([
        hh - ll, np.abs(hh - _shift(cc, 1)), np.abs(ll - _shift(cc, 1))]).T)
    atr = df_tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return atr.to_numpy(np.float32).T.copy()


def compute_features(p: Panel) -> dict:
    cache = CACHE / "features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        log("特征缓存已载入")
        return {k: z[k] for k in z.files}
    log("计算特征 ...")
    c, o, h, l, pc, amt, td = p.m["c"], p.m["o"], p.m["h"], p.m["l"], p.m["pc"], p.m["amt"], p.m["td"]
    f = p.m["f"]
    # 关键口径：冻结层 factor = Π(1+pct_chg) 是总收益指数；而 hfq_close = close×factor
    # 会重复计入涨跌幅（实测 hfq 日收益 ≈ 2×pct_chg）。本族一律用 factor 本身作为
    # 复权价格序列（同股常数缩放不影响均线/突破/ATR 的比值），日内价格按 factor×px/close 换算。
    hc = f.copy()
    hh = f * (h / c)
    ll = f * (l / c)
    pch = f * (pc / c)
    r1 = hc / _shift(hc, 1) - 1
    out = {
        "hc": hc,
        "oo": f * (o / c),
        "r1": r1,
        "r5": hc / _shift(hc, 5) - 1,
        "r20": hc / _shift(hc, 20) - 1,
        "r60": hc / _shift(hc, 60) - 1,
        "ma20": _roll_mean(hc, 20),
        "ma60": _roll_mean(hc, 60),
        "ma120": _roll_mean(hc, 120),
        "atr14": _wilder_atr(hh, ll, hc, 14),
        "vol20": _roll_std(r1, 20),
        "prior_high60": _shift(_roll_max(hh, 60), 1),
        "adv20": _roll_mean(np.nan_to_num(amt, nan=0.0), 20, minp=10),
    }
    # ewm 在 NaN 之后可能保留末值；ATR 必须与当日价格同域
    out["atr14"][~np.isfinite(out["hc"]) | (out["atr14"] <= 0)] = np.nan
    out["adv20"][~np.isfinite(out["hc"])] = np.nan
    np.savez(cache, **out)
    log("特征已缓存")
    return out


def expanding_pct(x: np.ndarray, minp: int = 250) -> np.ndarray:
    import bisect
    out = np.full(len(x), np.nan)
    hist: list[float] = []
    for i, v in enumerate(x):
        if not np.isfinite(v):
            continue
        bisect.insort(hist, float(v))
        if len(hist) >= minp:
            out[i] = bisect.bisect_right(hist, float(v)) / len(hist)
    return out


def build_state(p: Panel, feats: dict) -> pd.DataFrame:
    cache = OUT / "01_状态序列.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        log(f"状态缓存：{len(df)} 行")
        return df
    log("构建生态状态 ...")
    c, td = p.m["c"], p.m["td"]
    valid = (np.isfinite(c) & (td >= LIST_DAYS_MIN) & (~p.is_st)).astype(bool)
    above20 = (feats["hc"] > feats["ma20"]) & np.isfinite(feats["ma20"])
    above60 = (feats["hc"] > feats["ma60"]) & np.isfinite(feats["ma60"])
    r20 = feats["r20"]
    nd = p.n_days
    breadth20 = np.full(nd, np.nan)
    breadth60 = np.full(nd, np.nan)
    disp20 = np.full(nd, np.nan)
    medvol = np.full(nd, np.nan)
    for j in range(nd):
        m = valid[:, j]
        if m.sum() < 50:
            continue
        breadth20[j] = above20[m, j].mean()
        breadth60[j] = above60[m, j].mean()
        v = r20[m, j]
        v = v[np.isfinite(v)]
        if len(v) >= 50:
            lo, hi = np.quantile(v, DISP_WINSOR)
            vv = np.clip(v, lo, hi)
            disp20[j] = float(np.std(vv))
        vv = feats["vol20"][m, j]
        vv = vv[np.isfinite(vv)]
        if len(vv) >= 50:
            medvol[j] = float(np.median(vv))
    B = 0.5 * breadth20 + 0.5 * breadth60
    pB = expanding_pct(B, 250)
    pD = expanding_pct(disp20, 250)
    pV = expanding_pct(medvol, 250)

    def ema(x, hl=5):
        return pd.Series(x).ewm(halflife=hl, adjust=False).mean().to_numpy()

    sB, sD, sV = ema(pB), ema(pD), ema(pV)
    g = np.clip(0.55 + 0.30 * (2 * sB - 1) - 0.20 * (2 * sV - 1), 0.25, 1.00)
    reg = np.where(~np.isfinite(sB), np.nan, np.where(sB <= 1/3, 0, np.where(sB <= 2/3, 1, 2)))
    age = np.full(nd, np.nan)
    run = 0
    prev = None
    for j in range(nd):
        if not np.isfinite(reg[j]):
            continue
        if prev is None or reg[j] != prev:
            run = 0
        else:
            run += 1
        age[j] = run
        prev = reg[j]

    # 沪深300 20 日波动（2006 起）
    hs_vol = np.full(nd, np.nan)
    try:
        import duckdb
        con = duckdb.connect(str(DB), read_only=True)
        idx = con.execute(
            f"SELECT date, close FROM index_daily WHERE symbol='000300.SH' "
            f"AND date BETWEEN DATE '{PANEL_START}' AND DATE '{ANALYSIS_END}' ORDER BY 1"
        ).df()
        con.close()
        idx["date"] = idx["date"].astype(str)
        s = pd.Series(idx["close"].to_numpy(), index=idx["date"])
        s = s.reindex(pd.Index(p.dates))
        rr = s.pct_change()
        hs_vol = (rr.rolling(20, min_periods=20).std() * math.sqrt(243)).to_numpy()
    except Exception as exc:
        log(f"指数波动跳过：{exc}")

    df = pd.DataFrame({
        "date": p.dates, "breadth20": breadth20, "breadth60": breadth60, "B": B,
        "disp20": disp20, "medvol20": medvol, "hs300_vol20": hs_vol,
        "pB": pB, "pD": pD, "pV": pV, "sB": sB, "sD": sD, "sV": sV,
        "g": g, "regimeB": reg, "ageB": age,
    })
    for L in (5, 10):
        for k in ("sB", "sD", "sV"):
            df[f"{k}_L{L}"] = np.r_[np.full(L, np.nan), df[k].to_numpy()[:-L]]
    df.to_csv(cache, index=False)
    log(f"状态已写入 {cache}（{len(df)} 行）")
    return df


# --------------------------------------------------------------------------- #
# 3. 信号
# --------------------------------------------------------------------------- #

def universe_mask(p: Panel, feats: dict) -> np.ndarray:
    c, td = p.m["c"], p.m["td"]
    valid = (np.isfinite(c) & (td >= LIST_DAYS_MIN) & (~p.is_st) & (c >= PRICE_MIN)
             & (feats["adv20"] >= ADV_MIN) & p.tradable)
    prev_tr = np.zeros_like(p.tradable)
    prev_tr[:, 1:] = p.tradable[:, :-1]
    resume = p.tradable & (~prev_tr)
    valid &= ~resume
    di = np.arange(p.n_days)[None, :]
    valid &= ~((p.delist_i[:, None] - di) <= 30)
    valid &= ~(np.abs(p.m["pct"]) > 25)
    return valid


def pct_rank_within(mask: np.ndarray, vals: np.ndarray, min_n: int = 10) -> np.ndarray:
    out = np.full(vals.shape, np.nan, dtype=np.float32)
    for j in range(vals.shape[1]):
        idx = np.where(mask[:, j])[0]
        if len(idx) < min_n:
            continue
        v = vals[idx, j]
        ok = np.isfinite(v)
        if ok.sum() < min_n:
            continue
        k = int(ok.sum())
        order = np.argsort(v[ok], kind="stable")
        pr = np.empty(k, dtype=np.float32)
        pr[order] = np.arange(1, k + 1, dtype=np.float32) / k
        out[idx[ok], j] = pr
    return out


def make_rank(mask: np.ndarray, key1: np.ndarray, key2: np.ndarray, cap: int = 60) -> np.ndarray:
    rank = np.zeros(mask.shape, dtype=np.int16)
    for j in range(mask.shape[1]):
        idx = np.where(mask[:, j])[0]
        if len(idx) == 0:
            continue
        order = np.lexsort((idx, -key2[idx, j], key1[idx, j]))
        sel = idx[order[:cap]]
        rank[sel, j] = np.arange(1, len(sel) + 1, dtype=np.int16)
    return rank


def build_signals(p: Panel, feats: dict) -> dict:
    cache = CACHE / "signals.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        log("信号缓存已载入")
        return {k: z[k] for k in z.files}
    log("构建信号 ...")
    valid = universe_mask(p, feats)
    hc = feats["hc"]          # factor 空间（总收益指数），不能与原始价混用
    trend = valid & (hc > feats["ma60"]) & (feats["ma20"] > feats["ma60"]) & (feats["r60"] > 0)
    r5 = feats["r5"]
    rev_pr = pct_rank_within(trend, r5)
    rev_mask = trend & np.isfinite(r5) & (r5 < 0) & (rev_pr <= 0.20)
    rev_rank = make_rank(rev_mask, r5, feats["adv20"])

    all_pr = pct_rank_within(valid, r5)
    revall_mask = valid & np.isfinite(r5) & (r5 < 0) & (all_pr <= 0.20)
    revall_rank = make_rank(revall_mask, r5, feats["adv20"])

    strength = (hc - feats["prior_high60"]) / np.maximum(feats["atr14"], 1e-6)
    brk_mask = valid & (hc > feats["prior_high60"]) & (p.m["amt"] >= 1.2 * feats["adv20"])
    brk_rank = make_rank(brk_mask, -strength, feats["adv20"])

    brk_novol_mask = valid & (hc > feats["prior_high60"])
    brk_novol_rank = make_rank(brk_novol_mask, -strength, feats["adv20"])

    out = {
        "valid": valid, "trend": trend,
        "rev_mask": rev_mask, "rev_rank": rev_rank,
        "revall_mask": revall_mask, "revall_rank": revall_rank,
        "brk_mask": brk_mask, "brk_rank": brk_rank,
        "brk_novol_mask": brk_novol_mask, "brk_novol_rank": brk_novol_rank,
        "strength": strength, "rev_pr": rev_pr,
    }
    np.savez(cache, **out)
    log("信号已缓存")
    return out


# --------------------------------------------------------------------------- #
# 4. 事件研究
# --------------------------------------------------------------------------- #

def forward_returns(p: Panel, feats: dict, H: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (gross[n_codes,n_days], exit_days[n_codes,n_days])：
    信号日 di 的 t+1 开盘买入，t+1+H 开盘卖出；退市/停牌/跌停顺延 ≤10 日。
    gross 只在 di 有定义（其余 NaN）。价格用 factor 换算的总收益开盘价。
    """
    o, f = p.m["o"], p.m["f"]
    hfq_o = feats["oo"]
    nc, nd = p.n_codes, p.n_days
    gross = np.full((nc, nd), np.nan, dtype=np.float32)
    exit_idx = np.full((nc, nd), -1, dtype=np.int16)
    for di in range(nd - 1):
        entry = di + 1
        base = entry + H
        if base >= nd:
            break
        rows = np.arange(nc)
        ok_entry = p.tradable[:, entry] & (~p.lim_up_open[:, entry]) & np.isfinite(hfq_o[:, entry])
        # 卖出日顺延
        found = np.full(nc, -1, dtype=np.int16)
        for k in range(11):
            j = base + k
            if j >= nd:
                break
            ok = (found < 0) & p.tradable[:, j] & (~p.lim_dn_open[:, j]) & np.isfinite(hfq_o[:, j])
            found[ok] = j
        exit_px = np.where(found >= 0,
                           hfq_o[rows, np.maximum(found, 0)],
                           np.where(base + 10 < nd, f[:, min(base + 10, nd - 1)], np.nan))
        ret = exit_px / hfq_o[:, entry] - 1
        ret = np.where(found < 0, ret - 0.10, ret)   # 无法卖出：按最后价 ×0.9
        ret = np.where(ok_entry, ret, np.nan)
        gross[:, di] = ret
        exit_idx[:, di] = found
    return gross, exit_idx


def benchmark_forward(p: Panel, gross: np.ndarray) -> np.ndarray:
    valid = np.isfinite(gross) & p.tradable
    with np.errstate(invalid="ignore"):
        out = np.nanmean(np.where(valid, gross, np.nan), axis=0)
    return out


def size_bench(p: Panel, gross: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """同日、同流通市值五分位的等权前向收益（规模控制的诊断基准）。"""
    fmc = p.m["fmc"]
    c, td = p.m["c"], p.m["td"]
    elig = np.isfinite(c) & (td >= LIST_DAYS_MIN) & (~p.is_st) & p.tradable & np.isfinite(fmc)
    nb = 5
    bench = np.full((p.n_days, nb), np.nan, dtype=np.float32)
    bucket = np.full(fmc.shape, -1, dtype=np.int8)
    for di in range(p.n_days):
        idx = np.where(elig[:, di] & np.isfinite(gross[:, di]))[0]
        if len(idx) < 100:
            continue
        v = fmc[idx, di]
        edges = np.quantile(v, [0.2, 0.4, 0.6, 0.8])
        b = np.searchsorted(edges, v, side="right")
        bucket[idx, di] = b
        for k in range(nb):
            m = b == k
            if m.sum() >= 10:
                bench[di, k] = gross[idx[m], di].mean()
    return bucket, bench


def event_study(p: Panel, feats: dict, sig: dict, state: pd.DataFrame, H: int = H_DEFAULT) -> pd.DataFrame:
    sig_path = OUT / "02_事件明细.parquet"
    if sig_path.exists():
        ev = pd.read_parquet(sig_path)
        log(f"事件缓存：{len(ev)} 行")
        return ev
    log("事件研究 ...")
    gross, _ = forward_returns(p, feats, H)
    bench = benchmark_forward(p, gross)
    size_bucket, size_b = size_bench(p, gross)
    # 可投资域内等权（与信号同一 U-MAINBOARD-V1），是“选股边际”的最直接机会成本
    valid_u = sig["valid"]
    bench_u = np.full(p.n_days, np.nan, dtype=np.float32)
    for di in range(p.n_days):
        m = valid_u[:, di] & np.isfinite(gross[:, di])
        if m.sum() >= 10:
            bench_u[di] = gross[m, di].mean()
    nd = p.n_days
    i0, i1 = p.di(DEV[0]), p.di(ANALYSIS_END) + 1
    s = state.set_index("date")
    rows = []
    for sleeve, rankm in (("REV", sig["rev_rank"]), ("BRK", sig["brk_rank"])):
        for di in range(i0, i1):
            idx = np.where(rankm[:, di] > 0)[0]
            if len(idx) == 0:
                continue
            rk = rankm[idx, di]
            keep = rk <= 60
            idx, rk = idx[keep], rk[keep]
            g = gross[idx, di]
            ok = np.isfinite(g)
            idx, rk, g = idx[ok], rk[ok], g[ok]
            if len(idx) == 0:
                continue
            date_s = str(p.dates[di])
            excess = g - COST_ROUNDTRIP - bench[di]
            sel = s.loc[date_s]
            for ci, r, gg, ex in zip(idx, rk, g, excess):
                bsb = size_b[di, size_bucket[ci, di]] if size_bucket[ci, di] >= 0 else np.nan
                rows.append((
                    date_s, di, ci, str(p.codes[ci]), sleeve, int(r),
                    float(feats["r5"][ci, di]), float(feats["r20"][ci, di]), float(feats["r60"][ci, di]),
                    float(gg), float(gg - COST_ROUNDTRIP), float(bench[di]), float(ex),
                    float(bsb), float(gg - COST_ROUNDTRIP - bsb),
                    float(bench_u[di]), float(gg - COST_ROUNDTRIP - bench_u[di]),
                    float(sel["sB"]), float(sel["sD"]), float(sel["sV"]),
                    float(sel["sB_L5"]), float(sel["sD_L5"]), float(sel["sV_L5"]),
                    float(sel["sB_L10"]), float(sel["sD_L10"]), float(sel["sV_L10"]),
                    float(sel["regimeB"]), float(sel["ageB"]),
                ))
    ev = pd.DataFrame(rows, columns=[
        "date", "di", "ci", "code", "sleeve", "rank", "r5", "r20", "r60",
        "gross", "net", "bench", "excess", "bench_size", "excess_size",
        "bench_u", "excess_u", "sB", "sD", "sV",
        "sB_L5", "sD_L5", "sV_L5", "sB_L10", "sD_L10", "sV_L10", "regimeB", "ageB"])
    ev["window"] = np.where((ev["date"] >= DEV[0]) & (ev["date"] <= DEV[1]), "dev", "val")
    ev.to_parquet(sig_path, index=False)
    log(f"事件已写入 {sig_path}（{len(ev)} 行）")
    return ev


def bucket3(x: pd.Series) -> pd.Series:
    return np.where(x <= 1/3, 0, np.where(x <= 2/3, 1, 2))


def clustered_t(df: pd.DataFrame, col: str = "excess") -> tuple[float, int]:
    g = df.groupby("date")[col].mean()
    n = len(g)
    if n < 3:
        return float("nan"), n
    sd = g.std(ddof=1)
    if sd == 0:
        return float("nan"), n
    return float(g.mean() / (sd / math.sqrt(n))), n


def month_boot_diff(ev: pd.DataFrame, dim: str, bucket: int, b: int = 2000, seed: int = 7) -> tuple[float, float]:
    sub = ev[(ev["rank"] <= 10) & (bucket3(ev[dim]) == bucket)]
    sub = sub.assign(month=sub["date"].str[:7])
    piv = sub.groupby(["month", "sleeve"])["excess"].mean().unstack()
    piv = piv.dropna()
    if len(piv) < 12:
        return float("nan"), float("nan")
    d = (piv.get("REV", pd.Series(index=piv.index)) - piv.get("BRK", pd.Series(index=piv.index))).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(d, len(d), replace=True)) for _ in range(b)])
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def conditional_tables(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dim in ("sB", "sD", "sV"):
        ev[f"bk_{dim}"] = bucket3(ev[dim])
        for sleeve in ("REV", "BRK"):
            for bk in (0, 1, 2):
                for win in ("dev", "val"):
                    sub = ev[(ev.sleeve == sleeve) & (ev[f"bk_{dim}"] == bk)
                             & (ev.window == win) & (ev["rank"] <= 10)]
                    if len(sub) < 20:
                        continue
                    t, ndates = clustered_t(sub)
                    ts, _ = clustered_t(sub, "excess_size")
                    tu, _ = clustered_t(sub, "excess_u")
                    rows.append(dict(sleeve=sleeve, dim=dim, bucket=bk, window=win,
                                     n=len(sub), n_dates=ndates,
                                     mean_gross=sub.gross.mean(), mean_net=sub.net.mean(),
                                     median_net=sub.net.median(), win_rate=(sub.net > 0).mean(),
                                     mean_excess=sub.excess.mean(), t_excess=t,
                                     mean_bench=sub.bench.mean(),
                                     mean_excess_size=sub.excess_size.mean(), t_excess_size=ts,
                                     mean_bench_size=sub.bench_size.mean(),
                                     mean_excess_u=sub.excess_u.mean(), t_excess_u=tu,
                                     mean_bench_u=sub.bench_u.mean()))
    cond = pd.DataFrame(rows)

    # 邻域：滞后 5/10 日 + 转换年龄
    rows2 = []
    for lag in (0, 5, 10):
        col = "sB" if lag == 0 else f"sB_L{lag}"
        bk = bucket3(ev[col])
        for sleeve in ("REV", "BRK"):
            for b in (0, 1, 2):
                sub = ev[(ev.sleeve == sleeve) & (bk == b) & (ev["rank"] <= 10)]
                if len(sub) < 20:
                    continue
                rows2.append(dict(kind=f"lag{lag}", sleeve=sleeve, bucket=b,
                                  n=len(sub), mean_net=sub.net.mean(), mean_excess=sub.excess.mean()))
    for b in range(4):
        lo, hi = [(0, 5), (6, 20), (21, 60), (61, 10**9)][b]
        sub_age = ev[(ev.ageB >= lo) & (ev.ageB <= hi) & (ev["rank"] <= 10)]
        for sleeve in ("REV", "BRK"):
            sub = sub_age[sub_age.sleeve == sleeve]
            if len(sub) < 20:
                continue
            rows2.append(dict(kind="age", sleeve=sleeve, bucket=b, n=len(sub),
                              mean_net=sub.net.mean(), mean_excess=sub.excess.mean()))
    return cond, pd.DataFrame(rows2)


# --------------------------------------------------------------------------- #
# 5. 组合模拟（快速，因子比例记账）
# --------------------------------------------------------------------------- #

@dataclass
class PConfig:
    name: str
    signal_rev: str | None
    signal_brk: str | None
    H: int = 10
    alloc: str = "fixed"          # single_rev / single_brk / fixed / tilt / compete
    exposure: str = "state"       # state / full
    topn: int = N_SLOTS
    cost_mult: float = 1.0
    cash: float = 1_000_000.0
    note: str = ""


CONFIGS: dict[str, PConfig] = {
    "C1": PConfig("C1", "rev", None, alloc="single_rev", note="REV 单 sleeve Top-10"),
    "C2": PConfig("C2", None, "brk", alloc="single_brk", note="BRK 单 sleeve Top-10"),
    "C3": PConfig("C3", "rev", "brk", alloc="fixed", note="双 sleeve 固定 5/5"),
    "C4": PConfig("C4", "rev", "brk", alloc="tilt", note="双 sleeve 状态倾斜"),
    "C5": PConfig("C5", "rev", "brk", alloc="compete", note="双 sleeve 槽位竞争"),
    "C6": PConfig("C6", "rev", "brk", alloc="fixed", exposure="full", note="固定 5/5 恒定满仓"),
    "C7": PConfig("C7", "revall", None, alloc="single_rev", note="REV 去掉趋势条件"),
    "C8": PConfig("C8", None, "brk_novol", alloc="single_brk", note="BRK 去掉量能确认"),
    "C9": PConfig("C9", "rev", "brk", alloc="fixed", H=5, note="C3 但 H=5"),
    "C10": PConfig("C10", "rev", "brk", alloc="fixed", H=20, note="C3 但 H=20"),
    "S1": PConfig("S1", "rev", "brk", alloc="fixed", cost_mult=2.0, note="C3 成本压力 2x"),
    "S2": PConfig("S2", "rev", "brk", alloc="fixed", topn=5, note="C3 Top-5"),
    "S3": PConfig("S3", "rev", "brk", alloc="fixed", topn=20, note="C3 Top-20"),
    "S4": PConfig("S4", "rev", "brk", alloc="fixed", cash=200_000, note="C3 初始资金 20 万"),
}
SELECTION = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
SENSITIVITY = ["S1", "S2", "S3", "S4"]


@dataclass
class Pos:
    ci: int
    shares: int
    entry_di: int
    entry_px: float
    tr_entry: float
    exit_due: int
    sleeve: str
    cost: float
    last_px: float
    fail_days: int = 0


def _fees(price: float, qty: int, side: str, d: date, mult: float = 1.0):
    import 费用与交易约束 as fee
    cfg = fee.FeeConfig(commission_rate=0.00025 * mult, min_commission=5.0 * mult)
    return fee.trade_fees(d, "600000.SH", side, price, qty, cfg)


def _slip(px: float, side: str, mult: float = 1.0) -> float:
    import 费用与交易约束 as fee
    return fee.apply_slippage(px, side, SLIP_BPS * mult)


def run_portfolio(p: Panel, feats: dict, sig: dict, state: pd.DataFrame, cfg: PConfig) -> dict:
    dates = p.dates
    start_i = p.di(DEV[0])
    end_i = p.di(ANALYSIS_END)
    g_arr = (np.ones(p.n_days) if cfg.exposure == "full" else state["g"].to_numpy())
    rank_rev = sig[f"{cfg.signal_rev}_rank"] if cfg.signal_rev else None
    rank_brk = sig[f"{cfg.signal_brk}_rank"] if cfg.signal_brk else None
    o, c, f, tradable = p.m["o"], p.m["c"], p.m["f"], p.tradable
    lim_up, lim_dn = p.lim_up_open, p.lim_dn_open
    cash = float(cfg.cash)
    positions: dict[int, Pos] = {}
    queue: list[tuple[int, str]] = []
    nav_prev = float(cfg.cash)
    g_prev = float(g_arr[start_i])
    nav_rows, trades, fails = [], [], []
    d_objs = [date.fromisoformat(str(x)) for x in dates]

    def caps_for(di: int) -> dict:
        if cfg.alloc == "single_rev":
            return {"rev": cfg.topn}
        if cfg.alloc == "single_brk":
            return {"brk": cfg.topn}
        if cfg.alloc == "tilt":
            sD = state["sD"].iloc[di]
            w = float(np.clip(0.5 + 0.30 * (2 * sD - 1), 0.2, 0.8))
            n = int(round(cfg.topn * w))
            n = max(1, min(cfg.topn - 1, n))
            return {"rev": n, "brk": cfg.topn - n}
        return {"rev": cfg.topn // 2, "brk": cfg.topn - cfg.topn // 2}

    def candidates(di: int, sleeve: str) -> list[int]:
        rm = rank_rev if sleeve == "rev" else rank_brk
        if rm is None:
            return []
        idx = np.where(rm[:, di] > 0)[0]
        if len(idx) == 0:
            return []
        return idx[np.argsort(rm[idx, di], kind="stable")].tolist()

    for di in range(start_i, end_i + 1):
        # --- 开盘：卖出（先）
        for ci in list(positions):
            pos = positions[ci]
            if pos.exit_due > di:
                continue
            ok = bool(tradable[ci, di]) and not bool(lim_dn[ci, di]) and np.isfinite(o[ci, di])
            if ok:
                px = _slip(float(o[ci, di]), "sell", cfg.cost_mult)
                fees = _fees(px, pos.shares, "sell", d_objs[di], cfg.cost_mult).total
                tr_exit = float(f[ci, di]) * float(o[ci, di]) / float(c[ci, di])
                proceeds = pos.shares * pos.entry_px * (tr_exit / pos.tr_entry) - fees
                cash += proceeds
                trades.append(dict(code=str(p.codes[ci]), sleeve=pos.sleeve,
                                   entry_di=pos.entry_di, exit_di=di,
                                   entry_date=str(dates[pos.entry_di]), exit_date=str(dates[di]),
                                   entry_px=pos.entry_px, exit_px=px, shares=pos.shares,
                                   cost=pos.cost, proceeds=proceeds,
                                   pnl=proceeds - pos.cost,
                                   ret=proceeds / pos.cost - 1,
                                   holding=di - pos.entry_di, reason="horizon"))
                del positions[ci]
            else:
                pos.fail_days += 1
                if pos.fail_days > 10:
                    px = float(c[ci, di]) * (1 - 0.10)
                    fees = _fees(px, pos.shares, "sell", d_objs[di], cfg.cost_mult).total
                    tr_force = float(f[ci, di]) * 0.90
                    proceeds = pos.shares * pos.entry_px * (tr_force / pos.tr_entry) - fees
                    cash += proceeds
                    trades.append(dict(code=str(p.codes[ci]), sleeve=pos.sleeve,
                                       entry_di=pos.entry_di, exit_di=di,
                                       entry_date=str(dates[pos.entry_di]), exit_date=str(dates[di]),
                                       entry_px=pos.entry_px, exit_px=px, shares=pos.shares,
                                       cost=pos.cost, proceeds=proceeds, pnl=proceeds - pos.cost,
                                       ret=proceeds / pos.cost - 1,
                                       holding=di - pos.entry_di, reason="forced"))
                    del positions[ci]

        # --- 开盘：买入
        for ci, sleeve in queue:
            if ci in positions:
                continue
            if len(positions) >= cfg.topn:
                break
            if not bool(tradable[ci, di]) or bool(lim_up[ci, di]) or not np.isfinite(o[ci, di]):
                fails.append(dict(date=str(dates[di]), code=str(p.codes[ci]), sleeve=sleeve,
                                  reason="limit_up_open" if bool(lim_up[ci, di]) else "suspended"))
                continue
            px = _slip(float(o[ci, di]), "buy", cfg.cost_mult)
            budget = nav_prev * g_prev / cfg.topn
            qty = int(budget / px / 100) * 100
            if qty < 100:
                continue
            fees = _fees(px, qty, "buy", d_objs[di], cfg.cost_mult).total
            while qty >= 100 and qty * px + fees > cash:
                qty -= 100
                if qty >= 100:
                    fees = _fees(px, qty, "buy", d_objs[di], cfg.cost_mult).total
            if qty < 100:
                fails.append(dict(date=str(dates[di]), code=str(p.codes[ci]), sleeve=sleeve,
                                  reason="insufficient_cash"))
                continue
            cost = qty * px + fees
            cash -= cost
            tr_entry = float(f[ci, di]) * float(o[ci, di]) / float(c[ci, di])
            positions[ci] = Pos(ci, qty, di, px, tr_entry, di + cfg.H, sleeve, cost,
                                float(c[ci, di]))
        queue = []

        # --- 收盘估值
        mv = 0.0
        for ci, pos in positions.items():
            px = float(c[ci, di]) if np.isfinite(c[ci, di]) else pos.last_px
            pos.last_px = px
            mv += pos.shares * pos.entry_px * (float(f[ci, di]) / pos.tr_entry)
        nav = cash + mv
        nav_rows.append(dict(date=str(dates[di]), nav=nav, cash=cash, n_pos=len(positions),
                             mv=mv, g=float(g_arr[di])))

        # --- 收盘：为 t+1 生成队列
        if di >= end_i:
            break
        n_hold_next = sum(1 for x in positions.values() if x.exit_due > di + 1)
        held_s = {"rev": 0, "brk": 0}
        for x in positions.values():
            if x.exit_due > di + 1:
                held_s[x.sleeve] = held_s.get(x.sleeve, 0) + 1
        caps = caps_for(di)
        free = cfg.topn - n_hold_next
        if free <= 0:
            g_prev, nav_prev = float(g_arr[di]), nav
            continue
        picked: list[tuple[int, str]] = []
        picked_ci = set(positions)
        for sleeve in ("rev", "brk"):
            cap = caps.get(sleeve, 0) - held_s.get(sleeve, 0)
            if cap <= 0:
                continue
            for ci in candidates(di, sleeve):
                if cap <= 0 or len(picked) >= free:
                    break
                if ci in picked_ci:
                    continue
                picked.append((ci, sleeve))
                picked_ci.add(ci)
                cap -= 1
        if cfg.alloc == "compete" and len(picked) < free:
            for sleeve in ("rev", "brk"):
                for ci in candidates(di, sleeve):
                    if len(picked) >= free:
                        break
                    if ci in picked_ci:
                        continue
                    picked.append((ci, sleeve))
                    picked_ci.add(ci)
        queue = picked
        g_prev, nav_prev = float(g_arr[di]), nav

    nav_df = pd.DataFrame(nav_rows)
    tr = pd.DataFrame(trades)
    fl = pd.DataFrame(fails)
    return {"nav": nav_df, "trades": tr, "fails": fl, "cfg": cfg}


def perf(nav: pd.Series, years: float | None = None) -> dict:
    nav = nav.astype(float)
    if len(nav) < 2:
        return {}
    days = (pd.Timestamp(nav.index[-1]) - pd.Timestamp(nav.index[0])).days
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / days) - 1 if days > 0 else np.nan
    mdd = float((1 - nav / nav.cummax()).max())
    r = nav.pct_change().dropna()
    vol = float(r.std(ddof=1) * math.sqrt(243)) if len(r) > 2 else np.nan
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(243)) if len(r) > 2 and r.std() > 0 else np.nan
    return dict(cagr=cagr, mdd=mdd, calmar=(cagr / mdd if mdd > 0 else np.nan),
                vol=vol, sharpe=sharpe, n_days=len(nav))


def portfolio_metrics(res: dict) -> dict:
    nav = res["nav"].set_index("date")["nav"]
    m = perf(nav)
    tr = res["trades"]
    n_tr = len(tr)
    win = float((tr["pnl"] > 0).mean()) if n_tr else np.nan
    avg_win = float(tr.loc[tr.pnl > 0, "pnl"].mean()) if (tr.pnl > 0).any() else np.nan
    avg_loss = float(tr.loc[tr.pnl < 0, "pnl"].mean()) if (tr.pnl < 0).any() else np.nan
    pl_ratio = abs(avg_win / avg_loss) if avg_loss and np.isfinite(avg_loss) and avg_loss != 0 else np.nan
    total_profit = tr.loc[tr.pnl > 0, "pnl"].sum() if n_tr else 0.0
    top5 = tr.nlargest(5, "pnl")["pnl"].sum() if n_tr else 0.0
    years = max((pd.Timestamp(nav.index[-1]) - pd.Timestamp(nav.index[0])).days / 365.25, 1e-9)
    turn = (tr["cost"].sum() / nav.mean() / years) if n_tr else 0.0
    return {**m, "n_trades": n_tr, "win_rate": win, "pl_ratio": pl_ratio,
            "top5_profit_share": float(top5 / total_profit) if total_profit > 0 else np.nan,
            "turnover": turn, "avg_pos": float(res["nav"]["n_pos"].mean()),
            "cash_ratio": float((res["nav"]["cash"] / res["nav"]["nav"]).mean()),
            "n_fail": len(res["fails"])}


def yearly(nav: pd.Series) -> pd.DataFrame:
    nav = nav.astype(float)
    idx = pd.to_datetime(nav.index)
    s = pd.Series(nav.to_numpy(), index=idx)
    prev = s.iloc[0]
    rows = []
    for y, g in s.groupby(s.index.year):
        end = g.iloc[-1]
        rows.append((int(y), float(end / prev - 1)))
        prev = end
    return pd.DataFrame(rows, columns=["year", "ret"])


def stationary_bootstrap(nav: pd.Series, B: int = 2000, block: int = 20, seed: int = 7) -> dict:
    r = nav.astype(float).pct_change().dropna().to_numpy()
    n = len(r)
    if n < 100:
        return {}
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    nblocks = int(math.ceil(n / block))
    for b in range(B):
        starts = rng.integers(0, n, nblocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        out[b] = np.prod(1 + r[idx]) ** (243 / n) - 1
    return dict(p05=float(np.quantile(out, 0.05)), p50=float(np.quantile(out, 0.5)),
                p95=float(np.quantile(out, 0.95)))


def dsr(returns: np.ndarray, sr_trials: np.ndarray, n_trials: int) -> float:
    from scipy.stats import norm, skew, kurtosis
    r = returns[np.isfinite(returns)]
    if len(r) < 100:
        return float("nan")
    sr = float(np.mean(r) / np.std(r, ddof=1))
    T = len(r)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))
    var_sr = float(np.var([x for x in sr_trials if np.isfinite(x)], ddof=1)) if len(sr_trials) > 1 else 0.5
    gamma = 0.5772156649
    sr0 = math.sqrt(max(var_sr, 1e-12)) * (
        (1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(1 - 1 / (n_trials * math.e)))
    den = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr * sr, 1e-9))
    return float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / den))


def pbo(returns: pd.DataFrame, S: int = 16) -> dict:
    R = returns.dropna().to_numpy()
    T, M = R.shape
    if T < S * 10 or M < 2:
        return {}
    blocks = np.array_split(np.arange(T), S)
    means = np.array([[R[b, m].mean() for b in blocks] for m in range(M)])  # M×S
    stds = np.array([[R[b, m].std(ddof=1) for b in blocks] for m in range(M)])
    splits = list(combinations(range(S), S // 2))
    ranks = []
    for tr in splits:
        te = [i for i in range(S) if i not in tr]
        sr_tr = means[:, tr].sum(1) / np.maximum(stds[:, tr].sum(1), 1e-12)
        sr_te = means[:, te].sum(1) / np.maximum(stds[:, te].sum(1), 1e-12)
        best = int(np.argmax(sr_tr))
        ranks.append((sr_te < sr_te[best]).mean())
    ranks = np.array(ranks)
    return dict(pbo=float((ranks > 0.5).mean()), is_oos_median_rank=float(np.median(ranks)))


# --------------------------------------------------------------------------- #
# 6. 统一账户引擎复核
# --------------------------------------------------------------------------- #

def run_account(p: Panel, sig: dict, state: pd.DataFrame, cfg: PConfig) -> dict:
    import duckdb
    import 账户撮合 as acct_mod
    from 费用与交易约束 import FeeConfig
    import 费用与交易约束 as fee

    dates = p.dates
    start_i, end_i = p.di(DEV[0]), p.di(ANALYSIS_END)
    cal = [date.fromisoformat(str(x)) for x in dates[start_i:end_i + 1]]
    sym_info = {str(c): acct_mod.SymbolInfo(str(c), ipo_date=p.ipo_date.get(str(c)),
                                            out_date=p.out_date.get(str(c)))
                for c in p.codes}
    acfg = acct_mod.AccountConfig(
        initial_cash=cfg.cash,
        fee=FeeConfig(slippage_bps=SLIP_BPS * cfg.cost_mult,
                      commission_rate=0.00025 * cfg.cost_mult,
                      min_commission=5.0 * cfg.cost_mult),
        cash_settle_days=0, enforce_t1=True, allow_buy_downsize=True,
        liquidate_on_finalize=False, strict=False,
    )
    acct = acct_mod.Account(acfg, cal, freeze_date=cal[-1], symbol_info=sym_info)

    con = duckdb.connect(str(DB), read_only=True)
    ca = con.execute(
        f"SELECT * FROM corp_actions WHERE ex_date BETWEEN DATE '{DEV[0]}' AND DATE '{ANALYSIS_END}'"
    ).df()
    con.close()
    ca["ex_date"] = ca["ex_date"].astype(str)
    events_by_day: dict[str, list] = {}
    for r in ca.itertuples():
        ev = acct_mod.CorporateAction(r.symbol, date.fromisoformat(r.ex_date),
                                      float(r.dividend_per_share or 0), float(r.per_share_bonus or 0),
                                      float(r.rights_ratio or 0), float(r.rights_price or 0))
        if not ev.is_empty():
            events_by_day.setdefault(r.ex_date, []).append(ev)

    g_arr = (np.ones(p.n_days) if cfg.exposure == "full" else state["g"].to_numpy())
    rank_rev = sig[f"{cfg.signal_rev}_rank"] if cfg.signal_rev else None
    rank_brk = sig[f"{cfg.signal_brk}_rank"] if cfg.signal_brk else None
    o, c, f, tradable = p.m["o"], p.m["c"], p.m["f"], p.tradable
    lim_up, lim_dn = p.lim_up_open, p.lim_dn_open

    exit_due: dict[str, int] = {}
    sleeve_of: dict[str, str] = {}
    last_fill_n = 0
    nav_rows = []
    need_next: set[str] = set()

    def caps_for(di: int) -> dict:
        if cfg.alloc == "single_rev":
            return {"rev": cfg.topn}
        if cfg.alloc == "single_brk":
            return {"brk": cfg.topn}
        if cfg.alloc == "tilt":
            w = float(np.clip(0.5 + 0.30 * (2 * state["sD"].iloc[di] - 1), 0.2, 0.8))
            n = max(1, min(cfg.topn - 1, int(round(cfg.topn * w))))
            return {"rev": n, "brk": cfg.topn - n}
        return {"rev": cfg.topn // 2, "brk": cfg.topn - cfg.topn // 2}

    def candidates(di: int, sleeve: str) -> list[int]:
        rm = rank_rev if sleeve == "rev" else rank_brk
        if rm is None:
            return []
        idx = np.where(rm[:, di] > 0)[0]
        return idx[np.argsort(rm[idx, di], kind="stable")].tolist() if len(idx) else []

    def bar_for(ci: int, di: int):
        if not np.isfinite(o[ci, di]) or not np.isfinite(c[ci, di]):
            return None
        tr = bool(tradable[ci, di])
        return acct_mod.Bar(
            code=str(p.codes[ci]), date=date.fromisoformat(str(dates[di])),
            open=float(o[ci, di]), high=float(p.m["h"][ci, di]), low=float(p.m["l"][ci, di]),
            close=float(c[ci, di]), volume=float(p.m["vol"][ci, di]) if tr else 0.0,
            amount=float(p.m["amt"][ci, di]) if np.isfinite(p.m["amt"][ci, di]) else None,
            prev_close=float(p.m["pc"][ci, di]) if np.isfinite(p.m["pc"][ci, di]) else None,
            prev_close_is_ref=True, is_st=bool(p.is_st[ci, di]),
            trade_status=1 if tr else 0,
        )

    c_idx = {str(x): i for i, x in enumerate(p.codes)}
    for k, di in enumerate(range(start_i, end_i + 1)):
        d = cal[k]
        codes_need = {str(x) for x in acct.positions} | need_next
        bars = {}
        for code in codes_need:
            ci = c_idx[code]
            b = bar_for(ci, di)
            if b is not None:
                bars[code] = b
        rec = acct.on_day(d, bars, events=events_by_day.get(str(dates[di]), []))
        nav_rows.append(rec)

        # 新成交 -> 记录 exit_due；清理已清仓的 0 股持仓（避免账户内部 O(n^2) 扫描）
        for tr in acct.trades[last_fill_n:]:
            if tr.side == "buy":
                exit_due[tr.code] = di + cfg.H
        last_fill_n = len(acct.trades)
        for code in [c for c, pp in acct.positions.items() if pp.shares <= 0]:
            for od in acct.orders.values():
                if od.active and od.code == code:
                    od.status = "cancelled"
                    od.reason = "position_zero"
            del acct.positions[code]

        if di >= end_i:
            break
        need_next = set()

        # 卖出：到期或逾期的持仓，收盘提交次日开盘卖
        active_sells = {x.code for x in acct.orders.values() if x.active and x.side == "sell"}
        for code, pos in list(acct.positions.items()):
            if pos.shares <= 0:
                continue
            if exit_due.get(code, 10**9) <= di + 1:
                if code not in active_sells:
                    acct.sell_open(code, int(pos.shares), d, tag="horizon")
                need_next.add(code)

        # 买入：按槽位
        held = {"rev": 0, "brk": 0}
        for code, pos in acct.positions.items():
            if pos.shares <= 0:
                continue
            sl = sleeve_of.get(code, "brk")
            held[sl] = held.get(sl, 0) + 1
        n_hold_next = sum(1 for code, pos in acct.positions.items()
                          if pos.shares > 0 and exit_due.get(code, 10**9) > di + 1)
        free = cfg.topn - n_hold_next
        if free > 0:
            caps = caps_for(di)
            nav_now = float(acct.daily[-1]["nav"])
            g_now = float(g_arr[di])
            pending_codes = {x.code for x in acct.orders.values() if x.active and x.side == "buy"}
            picked = []
            picked_ci = set(c_idx.get(x, -1) for x in acct.positions) | set(c_idx.get(x, -1) for x in pending_codes)
            for sleeve in ("rev", "brk"):
                cap = caps.get(sleeve, 0) - held.get(sleeve, 0) - sum(
                    1 for x in pending_codes if sleeve_of.get(x) == sleeve)
                for ci in candidates(di, sleeve):
                    if cap <= 0 or len(picked) >= free:
                        break
                    if ci in picked_ci:
                        continue
                    code = str(p.codes[ci])
                    px = float(c[ci, di])
                    if not np.isfinite(px) or px <= 0:
                        continue
                    qty = int(nav_now * g_now / cfg.topn / px / 100) * 100
                    if qty < 100:
                        continue
                    acct.buy_open(code, qty, d, tag=sleeve)
                    sleeve_of[code] = sleeve
                    picked.append(code)
                    picked_ci.add(ci)
                    need_next.add(code)
                    cap -= 1
            if cfg.alloc == "compete" and len(picked) < free:
                for sleeve in ("rev", "brk"):
                    for ci in candidates(di, sleeve):
                        if len(picked) >= free:
                            break
                        if ci in picked_ci:
                            continue
                        code = str(p.codes[ci])
                        px = float(c[ci, di])
                        if not np.isfinite(px) or px <= 0:
                            continue
                        qty = int(nav_now * g_now / cfg.topn / px / 100) * 100
                        if qty < 100:
                            continue
                        acct.buy_open(code, qty, d, tag=sleeve)
                        sleeve_of[code] = sleeve
                        picked.append(code)
                        picked_ci.add(ci)
                        need_next.add(code)

    acct.finalize(cal[-1], liquidate=False)
    nav = pd.DataFrame(nav_rows)
    nav["date"] = pd.to_datetime(nav["date"])
    return {"nav": nav, "account": acct}


# --------------------------------------------------------------------------- #
# 7. 输出
# --------------------------------------------------------------------------- #

def write_ledger(prereg_sha: str, code_sha: str, extra: dict) -> None:
    path = OUT / "07_试验清单.csv"
    header = ["generated_at", "prereg_sha256", "code_sha256", "family", "configs_selection",
              "configs_sensitivity", "n_selection", "n_sensitivity", "N_for_DSR",
              "window", "universe", "cost_model", "global_registry_rows", "notes"]
    row = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prereg_sha256": prereg_sha, "code_sha256": code_sha,
        "family": "ECOLOGY-REV-BRK", "configs_selection": "|".join(SELECTION),
        "configs_sensitivity": "|".join(SENSITIVITY),
        "n_selection": len(SELECTION), "n_sensitivity": len(SENSITIVITY),
        "N_for_DSR": len(SELECTION) + len(SENSITIVITY),
        "window": f"{DEV[0]}..{VAL[1]}", "universe": "U-MAINBOARD-V1",
        "cost_model": "C-REAL-V1(10bp slip, historical fees)",
        "global_registry_rows": extra.get("registry_rows", ""),
        "notes": extra.get("notes", ""),
    }
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["panel", "features", "check", "events", "portfolio", "account", "all"],
                    default="all", nargs="?")
    ap.add_argument("--force-panel", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    p = build_panel(force=args.force_panel)
    if str(p.dates[-1]) > ANALYSIS_END:
        raise RuntimeError(f"面板越界：{p.dates[-1]} > {ANALYSIS_END}，拒绝计算")
    if args.stage == "panel":
        print(p.n_codes, p.n_days, p.dates[0], p.dates[-1])
        return
    feats = compute_features(p)
    if args.stage == "features":
        print({k: int(np.isfinite(v).sum()) for k, v in feats.items()})
        return
    state = build_state(p, feats)
    sig = build_signals(p, feats)

    prereg = OUT / "00_预注册_生态切换.md"
    code_sha = sha256_file(Path(__file__))
    prereg_sha = sha256_file(prereg)

    if args.stage in ("check", "all"):
        cc = causal_check(p, sig)
        log(f"因果自检：{len(cc)} 日，REV 掩码差中位 {cc.rev_mask_diff.median():.0f}、"
            f"BRK 掩码差中位 {cc.brk_mask_diff.median():.0f}、REV 前10重合中位 {cc.rev_top10_same.median():.0f}")

    if args.stage == "check":
        out = causal_check(p, sig)
        print(out.describe().round(3).to_string())
        print(out.head(10).to_string(index=False))
        return

    if args.stage == "events":
        ev = event_study(p, feats, sig, state)
        cond, neigh = conditional_tables(ev)
        cond.to_csv(OUT / "03_事件条件边际.csv", index=False)
        neigh.to_csv(OUT / "04_事件状态邻域.csv", index=False)
        print(cond.to_string(index=False))
        print(neigh.to_string(index=False))
        return

    if args.stage in ("portfolio", "all"):
        ev = event_study(p, feats, sig, state)
        cond, neigh = conditional_tables(ev)
        cond.to_csv(OUT / "03_事件条件边际.csv", index=False)
        neigh.to_csv(OUT / "04_事件状态邻域.csv", index=False)

        results, perf_rows, nav_all, trade_all = {}, [], {}, []
        for name in SELECTION + SENSITIVITY:
            cfg = CONFIGS[name]
            t0 = time.time()
            res = run_portfolio(p, feats, sig, state, cfg)
            m = portfolio_metrics(res)
            m.update(config=name, note=cfg.note, seconds=round(time.time() - t0, 1))
            perf_rows.append(m)
            results[name] = res
            nav_all[name] = res["nav"].set_index("date")["nav"]
            tr = res["trades"].copy()
            if len(tr):
                tr["config"] = name
                trade_all.append(tr)
            log(f"{name}: CAGR={m['cagr']:.2%} MDD={m['mdd']:.2%} trades={m['n_trades']}")
        perf_df = pd.DataFrame(perf_rows)
        perf_df.to_csv(OUT / "05_组合绩效.csv", index=False)
        diagnostic_quintiles(p, feats, sig)
        rb = ev.copy()
        rb["rank_bucket"] = pd.cut(rb["rank"], [0, 3, 10, 30, 60],
                                   labels=["1-3", "4-10", "11-30", "31-60"])
        rb.groupby(["sleeve", "window", "rank_bucket"], observed=True).agg(
            n=("gross", "size"), mean_gross=("gross", "mean"), mean_net=("net", "mean"),
            mean_excess_u=("excess_u", "mean")).reset_index().to_csv(
            OUT / "18_排名分位.csv", index=False)
        nav_df = pd.DataFrame(nav_all)
        nav_df.index.name = "date"
        nav_df.to_csv(OUT / "06_组合净值.csv")
        pd.concat(trade_all, ignore_index=True).to_csv(OUT / "08_组合交易.csv", index=False)

        yr_rows = []
        for name, nav in nav_all.items():
            y = yearly(nav)
            y["config"] = name
            yr_rows.append(y)
        pd.concat(yr_rows, ignore_index=True).to_csv(OUT / "09_分年收益.csv", index=False)

        # 基准
        bench = build_benchmarks(p, state)
        bench.to_csv(OUT / "10_基准.csv")

        # 分窗口
        win_rows = []
        for name, nav in nav_all.items():
            for w, (a, b) in (("dev", DEV), ("val", VAL)):
                sub = nav[(nav.index >= a) & (nav.index <= b)]
                if len(sub) > 2:
                    m = perf(sub)
                    m.update(config=name, window=w)
                    win_rows.append(m)
        pd.DataFrame(win_rows).to_csv(OUT / "11_分窗口绩效.csv", index=False)

        # 多重性
        rets = pd.DataFrame({k: v.pct_change() for k, v in nav_all.items()}).dropna()
        sharpes = rets.mean() / rets.std(ddof=1)
        sr_trials = sharpes.to_numpy()
        dsr_rows = []
        for name in SELECTION:
            d = dsr(rets[name].to_numpy(), sr_trials, n_trials=len(SELECTION) + len(SENSITIVITY))
            dsr_rows.append(dict(config=name, dsr_n14=d,
                                 dsr_n32=dsr(rets[name].to_numpy(), sr_trials, n_trials=32)))
        pd.DataFrame(dsr_rows).to_csv(OUT / "12_DSR.csv", index=False)
        pb = pbo(rets[SELECTION])
        pd.DataFrame([pb]).to_csv(OUT / "13_PBO.csv", index=False)

        boot = {}
        for name in ("C3", "C4", "C5"):
            if name in nav_all:
                boot[name] = stationary_bootstrap(nav_all[name])
        pd.DataFrame(boot).T.to_csv(OUT / "14_自助法.csv")

        write_ledger(prereg_sha, code_sha, {
            "registry_rows": count_registry(),
            "notes": "本族不写中央注册表（由方法负责人独占）；N=14，事件条件格另计 N=32 上界",
        })

    if args.stage in ("account", "all"):
        targets = ["C2", "C3", "C5", "C8"]
        rows = []
        for name in targets:
            cfg = CONFIGS[name]
            log(f"账户复核 {name} ...")
            acc = run_account(p, sig, state, cfg)
            pd.DataFrame(acc["account"].daily).to_csv(OUT / f"15_账户日明细_{name}.csv", index=False)
            nav = acc["nav"].set_index("date")["nav"]
            m = perf(nav)
            m.update(config=name, n_trades=len(acc["account"].trades),
                     n_flags=sum(acc["account"].flag_counts.values()))
            rows.append(m)
            nav.rename(name).to_csv(OUT / f"15_账户净值_{name}.csv")
        pd.DataFrame(rows).to_csv(OUT / "16_账户复核.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))


def causal_check(p: Panel, sig: dict, n: int = 200, seed: int = 20261026) -> pd.DataFrame:
    """截断重算：只用 <=t 的原始切片重算 REV/BRK 掩码与前 10，与缓存对照。"""
    rng = np.random.default_rng(seed)
    days = rng.choice(np.arange(p.di(DEV[0]), p.di(ANALYSIS_END) + 1), size=min(n, 500), replace=False)
    f, h, c, amt, td = p.m["f"], p.m["h"], p.m["c"], p.m["amt"], p.m["td"]
    hh = f * (h / c)
    rows = []
    for di in days:
        di = int(di)
        if di < 260:
            continue
        hc = f[:, di]
        ma20 = np.nanmean(f[:, di - 19:di + 1], axis=1)
        ma60 = np.nanmean(f[:, di - 59:di + 1], axis=1)
        r5 = hc / f[:, di - 5] - 1
        r20 = hc / f[:, di - 20] - 1
        r60 = hc / f[:, di - 60] - 1
        prior = np.nanmax(hh[:, di - 60:di], axis=1)
        adv = np.nanmean(np.nan_to_num(amt[:, di - 19:di + 1], nan=0.0), axis=1)
        prev_tr = p.tradable[:, di - 1]
        resume = p.tradable[:, di] & (~prev_tr)
        valid = (np.isfinite(c[:, di]) & (td[:, di] >= LIST_DAYS_MIN) & (~p.is_st[:, di])
                 & (c[:, di] >= PRICE_MIN) & (adv >= ADV_MIN) & p.tradable[:, di]
                 & (~resume) & ~((p.delist_i - di) <= 30)
                 & ~(np.abs(p.m["pct"][:, di]) > 25))
        trend = valid & (hc > ma60) & (ma20 > ma60) & (r60 > 0)
        pr = pct_rank_within(trend[:, None], r5[:, None])[:, 0]
        rev_mask = trend & np.isfinite(r5) & (r5 < 0) & (pr <= 0.20)
        brk_mask = valid & (hc > prior) & (amt[:, di] >= 1.2 * adv)
        # 与缓存对照
        rev_cached = sig["rev_mask"][:, di]
        brk_cached = sig["brk_mask"][:, di]
        rev_rank = sig["rev_rank"][:, di]
        brk_rank = sig["brk_rank"][:, di]
        rev_top = set(np.where(rev_rank > 0)[0][np.argsort(rev_rank[rev_rank > 0])[:10]])
        brk_top = set(np.where(brk_rank > 0)[0][np.argsort(brk_rank[brk_rank > 0])[:10]])
        idx_rev = np.where(rev_mask)[0]
        idx_brk = np.where(brk_mask)[0]
        if len(idx_rev) >= 10:
            order = np.lexsort((idx_rev, -adv[idx_rev], r5[idx_rev]))
            rev_top2 = set(idx_rev[order[:10]])
        else:
            rev_top2 = set(idx_rev)
        brk_top2 = set()   # BRK 排序键含 ATR；本检查只验证掩码（ATR 为纯滚动、另由公式审计）
        rows.append(dict(date=str(p.dates[di]),
                         rev_mask_diff=int((rev_mask != rev_cached).sum()),
                         brk_mask_diff=int((brk_mask != brk_cached).sum()),
                         rev_top10_same=len(rev_top & rev_top2)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "24_因果自检.csv", index=False)
    return out


def diagnostic_quintiles(p: Panel, feats: dict, sig: dict) -> pd.DataFrame:
    """无条件 5 日收益五分位的前瞻 10 日超额：检验“反转”是否存在于正确的尾部。"""
    gross, _ = forward_returns(p, feats, 10)
    bench = benchmark_forward(p, gross)
    valid, r5 = sig["valid"], feats["r5"]
    rows = []
    for di in range(p.di(DEV[0]), p.di(ANALYSIS_END) + 1):
        mask = valid[:, di] & np.isfinite(r5[:, di]) & np.isfinite(gross[:, di])
        idx = np.where(mask)[0]
        if len(idx) < 200:
            continue
        v = r5[idx, di]
        e = np.quantile(v, [0.2, 0.4, 0.6, 0.8])
        q = np.searchsorted(e, v, side="right")
        for k in range(5):
            gg = gross[idx[q == k], di]
            rows.append((str(p.dates[di]), k, len(gg), gg.mean() - bench[di]))
    df = pd.DataFrame(rows, columns=["date", "quintile", "n", "excess"])
    df["window"] = np.where(df.date <= DEV[1], "dev", "val")
    out = df.groupby(["window", "quintile"]).agg(
        n=("n", "sum"), mean_excess=("excess", "mean"),
        t=("excess", lambda x: x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))),
    ).reset_index()
    out.to_csv(OUT / "17_反转分位诊断.csv", index=False)
    return out


def count_registry() -> int:
    try:
        path = ROOT / "03_结果" / "实验注册表.csv"
        with open(path, encoding="utf-8") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except Exception:
        return -1


def build_benchmarks(p: Panel, state: pd.DataFrame) -> pd.DataFrame:
    """B1 主板等权（月度再平衡，10bp 单边）、B2 沪深300、B4 现金。"""
    valid = universe_mask(p, dict(p.m) | {"adv20": _roll_mean(np.nan_to_num(p.m["amt"], nan=0.0), 20, minp=10)})
    r1 = p.m["f"] / _shift(p.m["f"], 1) - 1
    ew = np.nanmean(np.where(valid & np.isfinite(r1), r1, np.nan), axis=0)
    dates = pd.to_datetime(p.dates)
    ser = pd.Series(ew, index=dates).fillna(0.0)
    # 指数
    import duckdb
    con = duckdb.connect(str(DB), read_only=True)
    idx = con.execute(
        f"SELECT date, close FROM index_daily WHERE symbol='000300.SH' "
        f"AND date BETWEEN DATE '{PANEL_START}' AND DATE '{ANALYSIS_END}' ORDER BY 1"
    ).df()
    con.close()
    idx["date"] = pd.to_datetime(idx["date"])
    hs = idx.set_index("date")["close"].reindex(dates).ffill().pct_change().fillna(0.0)
    # 月度再平衡成本
    month = pd.Series(dates.strftime("%Y-%m")).to_numpy()
    cost = np.where(np.r_[True, month[1:] != month[:-1]], 0.001, 0.0)
    b1 = (1 + ser - cost).cumprod()
    b2 = (1 + hs).cumprod()
    out = pd.DataFrame({"B1_EW": b1, "B2_HS300": b2, "B4_cash": 1.0}, index=dates)
    out.index.name = "date"
    return out


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
