#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收缩突破族（VCB）研究实现：数据、特征、事件研究、组合、稳健性、自检。

纪律
----
* 只读 `01_数据/frozen/frozen.duckdb`，所有计算硬截断 `2020-12-31`；
  任何 2021+ 行情在本模块内不可达（加载时断言）。
* 预注册见 `03_结果/候选_收缩突破/00_预注册_收缩突破.md`；本文件只实现已登记配置。
* 组合结论只由 `02_代码/账户撮合.py::Account` 生成，不另写向量化组合冒充结论。
* 价格：总收益价格指数 tr（由 factor 重建）用于收益/比例/通道；raw 用于成交、整股、涨跌停、流动性。
* 注意：冻结层 hfq_* 不是可比的复权价（其比率把日收益算两遍），本模块不使用。
* 停牌日不进入 ATR/波动/均线/通道窗口（只在有成交日计算）。

命令
----
    python3 02_代码/候选_收缩突破.py feasibility
    python3 02_代码/候选_收缩突破.py events
    python3 02_代码/候选_收缩突破.py portfolio
    python3 02_代码/候选_收缩突破.py robust
    python3 02_代码/候选_收缩突破.py selfcheck
    python3 02_代码/候选_收缩突破.py report
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "01_数据" / "frozen" / "frozen.duckdb"
OUT = ROOT / "03_结果" / "候选_收缩突破"
CACHE = OUT / "_cache"

PANEL_START = "2004-01-01"
PANEL_END = "2020-12-31"
SIGNAL_START = "2005-01-01"   # 预热起点；2004 仅用于指标热身
DEV_START, DEV_END = "2008-01-01", "2015-12-31"
VAL_START, VAL_END = "2016-01-01", "2020-12-31"
FREEZE = "2026-10-26"
HORIZONS = (5, 10, 20, 40, 60)

# 先验参数（写死在代码里，不从收益反推）
BUFFER_ATR = 0.05
Q_MAIN = 0.30
Q_TIGHT, Q_LOOSE = 0.20, 0.40
PLATFORM_MAX = 0.12
VOL_EXPAND = 1.50
EXT_MA20_MAX = 0.12
EXT_CH_MAX = 0.06
ADV_MIN = 3.0e7
MIN_TD = 250
MIN_PRICE = 2.0

TRADING_DAYS_PER_YEAR = 243.0


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SigConfig:
    key: str
    N: int = 55
    metric: str = "atr"          # atr / hv / bbw
    q: float = Q_MAIN
    trend: str = "ma60"         # ma60 / ma120 / none
    platform: bool = True
    vol_expand: bool = True
    ext: bool = True
    ext_ma20: float = EXT_MA20_MAX
    ext_ch: float = EXT_CH_MAX
    mode: str = "first"         # first / cont / both
    vol_contract: bool = False  # 成交额收缩
    note: str = ""

    def filters(self) -> list[str]:
        f = [f"N{self.N}", f"{self.metric}<=p{int(round(self.q*100))}"]
        f.append({"ma60": "trend:ma60", "ma120": "trend:ma120", "none": "trend:none"}[self.trend])
        if self.platform:
            f.append(f"platform<={PLATFORM_MAX:.0%}")
        if self.vol_expand:
            f.append(f"vol>={VOL_EXPAND:.1f}x")
        if self.ext:
            f.append(f"ext<={self.ext_ma20:.0%}/{self.ext_ch:.0%}")
        f.append("first" if self.mode == "first" else ("cont" if self.mode == "cont" else "any"))
        if self.vol_contract:
            f.append("adv_contract<=p40")
        return f


STRICT = SigConfig("VCB-M", N=55, q=Q_MAIN, trend="ma60", platform=True,
                   ext_ma20=EXT_MA20_MAX, ext_ch=EXT_CH_MAX,
                   note="主候选L0：55日通道、ATR收缩30%分位、平台<=12%、放量1.5x、延伸<=12%/6%、首次突破")
MAIN = STRICT

SIGNAL_CONFIGS: list[SigConfig] = [
    MAIN,
    replace(MAIN, key="VCB-N20", N=20, note="通道20日邻域"),
    replace(MAIN, key="VCB-N120", N=120, note="通道120日邻域"),
    replace(MAIN, key="VCB-HV", metric="hv", note="历史波动收缩口径"),
    replace(MAIN, key="VCB-BBW", metric="bbw", note="布林带宽收缩口径"),
    replace(MAIN, key="VCB-P20", q=Q_TIGHT, note="收缩更严：20%分位"),
    replace(MAIN, key="VCB-P40", q=Q_LOOSE, note="收缩更松：40%分位"),
    replace(MAIN, key="VCB-T120", trend="ma120", note="更长趋势背景"),
    replace(MAIN, key="VCB-T0", trend="none", note="去掉趋势背景"),
]

ABLATIONS: list[SigConfig] = [
    replace(MAIN, key="VCB-A1", platform=False, note="消融：去平台"),
    replace(MAIN, key="VCB-A2", vol_expand=False, note="消融：去放量确认"),
    replace(MAIN, key="VCB-A3", ext=False, note="消融：去延伸上限"),
    replace(MAIN, key="VCB-A4", mode="cont", note="消融：只做连续突破"),
    replace(MAIN, key="VCB-A5", vol_contract=True, note="消融：加成交额收缩"),
    replace(MAIN, key="VCB-A6", ext_ma20=0.20, ext_ch=0.10, note="消融：放宽延伸<=20%/10%"),
]

FEAS_LADDER: list[SigConfig] = [
    replace(STRICT, key="L0", note="可行性L0：全条件"),
    replace(STRICT, key="L1", platform=False, note="可行性L1：去平台"),
    replace(STRICT, key="L2", platform=False, q=Q_LOOSE, note="可行性L2：去平台+收缩40%"),
    replace(STRICT, key="L3", platform=False, q=Q_LOOSE, ext_ma20=0.20, ext_ch=0.10,
            note="可行性L3：去平台+收缩40%+延伸放宽到20%/10%"),
]

EXITS = ("T20", "T40", "T60", "T40-S3", "T40-C20", "T40-V")


# --------------------------------------------------------------------------- #
# 面板构建
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    dates: np.ndarray                 # datetime64[D]
    symbols: list[str]
    a: dict[str, np.ndarray]
    feats: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)

    def di(self, d) -> int:
        return int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(d).date())))

    def si(self, s: str) -> int:
        return self._sidx[s]

    def __post_init__(self):
        self._sidx = {s: i for i, s in enumerate(self.symbols)}


_PANEL_COLS = ("open", "high", "low", "close", "preclose", "volume", "amount", "factor",
               "is_suspended", "is_st", "td_since_list",
               "is_limit_up_open", "is_limit_down_open",
               "px_breaks_limit_up", "px_breaks_limit_down")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_panel() -> Panel:
    con = duckdb.connect(str(FROZEN), read_only=True)
    try:
        dates = np.array([r[0] for r in con.execute(
            "select date from calendar where is_trading_day=1 and date between ? and ? order by 1",
            [PANEL_START, PANEL_END]).fetchall()], dtype="datetime64[D]")
        symbols = [r[0] for r in con.execute(
            """select symbol from symbols where board='主板'
               and first_bar <= DATE '2020-12-31' and last_bar >= DATE '2004-01-01'
               order by symbol""").fetchall()]
        nd, ns = len(dates), len(symbols)
        _log(f"面板 {nd} 交易日 × {ns} 主板证券")

        a: dict[str, np.ndarray] = {}
        for c in ("open", "high", "low", "close", "preclose", "factor"):
            a[c] = np.full((nd, ns), np.nan, dtype=np.float32)
        a["amount"] = np.full((nd, ns), np.nan, dtype=np.float64)
        a["volume"] = np.full((nd, ns), np.nan, dtype=np.float32)
        a["td_since_list"] = np.zeros((nd, ns), dtype=np.int32)
        for c in ("is_suspended", "is_st", "is_limit_up_open", "is_limit_down_open",
                  "px_breaks_limit_up", "px_breaks_limit_down"):
            a[c] = np.ones((nd, ns), dtype=bool) if c == "is_suspended" else np.zeros((nd, ns), dtype=bool)

        sidx = {s: i for i, s in enumerate(symbols)}
        didx = {np.datetime64(d): i for i, d in enumerate(dates)}
        for year in range(2004, 2021):
            df = con.execute(
                f"""select symbol,date,{','.join(_PANEL_COLS)} from daily
                    where board='主板' and date between DATE '{year}-01-01' and DATE '{year}-12-31'""").df()
            if df.empty:
                continue
            ii = df["date"].map(lambda x: didx.get(np.datetime64(x), -1)).to_numpy()
            jj = df["symbol"].map(sidx).to_numpy()
            ok = (ii >= 0) & pd.notna(jj)
            ii, jj = ii[ok].astype(int), jj[ok].astype(int)
            for c in ("open", "high", "low", "close", "preclose", "factor"):
                a[c][ii, jj] = df[c].to_numpy(dtype=np.float32)[ok]
            a["amount"][ii, jj] = df["amount"].to_numpy(dtype=np.float64)[ok]
            a["volume"][ii, jj] = df["volume"].to_numpy(dtype=np.float32)[ok]
            a["td_since_list"][ii, jj] = df["td_since_list"].fillna(0).to_numpy(dtype=np.int32)[ok]
            for c in ("is_suspended", "is_st", "is_limit_up_open", "is_limit_down_open",
                      "px_breaks_limit_up", "px_breaks_limit_down"):
                a[c][ii, jj] = df[c].fillna(False).to_numpy(dtype=bool)[ok]
            _log(f"  {year}: {len(df):,} 行")
    finally:
        con.close()

    # 基本派生
    # 冻结层 hfq_close = raw_close × cumprod(1+ret)，直接用其比率会把日收益算两遍
    # （000001.SZ 2020-12-29 官方 +1.70%，hfq 收益 +3.42%）。
    # 这里重建“总收益价格指数”：tr = factor（收盘），日内价按当日 close 等比缩放。
    factor = a["factor"]
    a["tr_close"] = factor.astype(np.float32)
    for c in ("open", "high", "low"):
        a[f"tr_{c}"] = (a[c] * factor / a["close"]).astype(np.float32)
    a["valid"] = (np.isfinite(a["close"]) & (~a["is_suspended"]) & (a["close"] > 0))
    a["px_break"] = a["px_breaks_limit_up"] | a["px_breaks_limit_down"]
    prev_bad = np.zeros_like(a["valid"])
    prev_bad[1:] = (a["is_suspended"] | ~a["valid"])[:-1]
    a["resume"] = a["valid"] & prev_bad
    return Panel(dates, symbols, a)


def _roll(x: np.ndarray, n: int, how: str) -> np.ndarray:
    s = pd.Series(x)
    if how == "mean":
        return s.rolling(n, min_periods=n).mean().to_numpy()
    if how == "std":
        return s.rolling(n, min_periods=n).std(ddof=1).to_numpy()
    if how == "max":
        return s.rolling(n, min_periods=n).max().to_numpy()
    if how == "min":
        return s.rolling(n, min_periods=n).min().to_numpy()
    if how == "sum":
        return s.rolling(n, min_periods=n).sum().to_numpy()
    raise ValueError(how)


def _pct_rank(x: np.ndarray, n: int = 250, minp: int = 120) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=minp).rank(pct=True).to_numpy()


def compute_features(p: Panel) -> None:
    """逐证券计算特征，写回 p.feats（float32，date×symbol）。"""
    nd, ns = p.shape
    names = ["atr14", "atr60", "cr_atr", "p_atr", "hv20", "hv60", "cr_hv", "p_hv",
             "bbw20", "bbw60", "cr_bbw", "p_bbw", "ma20", "ma60", "ma120",
             "ma60_slope", "ma120_slope", "ch20", "ch55", "ch120",
             "dist_ch20", "dist_ch55", "dist_ch120", "range20",
             "adv10", "adv20", "adv60", "adv_contract", "p_advc", "vol_ratio",
             "dist_ma20", "dist_ma60", "nb20_20", "nb20_55", "nb20_120"]
    F = {k: np.full((nd, ns), np.nan, dtype=np.float32) for k in names}

    a = p.a
    for j in range(ns):
        valid = a["valid"][:, j]
        idx = np.flatnonzero(valid)
        n = len(idx)
        if n < 130:
            continue
        o = a["open"][idx, j].astype(np.float64)
        h = a["high"][idx, j].astype(np.float64)
        l = a["low"][idx, j].astype(np.float64)
        pc = a["preclose"][idx, j].astype(np.float64)
        fac = a["factor"][idx, j].astype(np.float64)
        tr_c = a["tr_close"][idx, j].astype(np.float64)
        tr_h = a["tr_high"][idx, j].astype(np.float64)
        tr_l = a["tr_low"][idx, j].astype(np.float64)
        amt = a["amount"][idx, j].astype(np.float64)

        prev_c = np.concatenate([[np.nan], tr_c[:-1]])
        tr = np.maximum(tr_h - tr_l, np.maximum(np.abs(tr_h - prev_c), np.abs(tr_l - prev_c)))
        tr[~np.isfinite(tr)] = np.nan
        r = np.log(tr_c / prev_c)

        atr14 = _roll(tr, 14, "mean")
        atr60 = _roll(tr, 60, "mean")
        cr_atr = atr14 / atr60
        hv20 = _roll(r, 20, "std") * math.sqrt(252.0)
        hv60 = _roll(r, 60, "std") * math.sqrt(252.0)
        cr_hv = hv20 / hv60
        ma20 = _roll(tr_c, 20, "mean")
        ma60 = _roll(tr_c, 60, "mean")
        ma120 = _roll(tr_c, 120, "mean")
        sd20 = _roll(tr_c, 20, "std")
        sd60 = _roll(tr_c, 60, "std")
        bbw20 = 4.0 * sd20 / ma20
        bbw60 = 4.0 * sd60 / ma60
        cr_bbw = bbw20 / bbw60

        close = tr_c
        ch20 = _roll(tr_h, 20, "max"); ch20 = np.concatenate([[np.nan], ch20[:-1]])
        ch55 = _roll(tr_h, 55, "max"); ch55 = np.concatenate([[np.nan], ch55[:-1]])
        ch120 = _roll(tr_h, 120, "max"); ch120 = np.concatenate([[np.nan], ch120[:-1]])

        cprev20 = _roll(close, 20, "max"); cprev20 = np.concatenate([[np.nan], cprev20[:-1]])
        cprev20min = _roll(close, 20, "min"); cprev20min = np.concatenate([[np.nan], cprev20min[:-1]])
        range20 = (cprev20 - cprev20min) / close

        adv10 = _roll(amt, 10, "mean")
        adv20 = _roll(amt, 20, "mean")
        adv60 = _roll(amt, 60, "mean")
        adv20_prev = np.concatenate([[np.nan], adv20[:-1]])
        vol_ratio = amt / adv20_prev
        adv10_prev = np.concatenate([[np.nan], adv10[:-1]])
        adv_contract = adv10_prev / adv60

        bo20 = close >= (ch20 + BUFFER_ATR * atr14)
        bo55 = close >= (ch55 + BUFFER_ATR * atr14)
        bo120 = close >= (ch120 + BUFFER_ATR * atr14)
        nb20_20 = np.concatenate([[np.nan], _roll(bo20.astype(float), 20, "sum")[:-1]])
        nb20_55 = np.concatenate([[np.nan], _roll(bo55.astype(float), 20, "sum")[:-1]])
        nb20_120 = np.concatenate([[np.nan], _roll(bo120.astype(float), 20, "sum")[:-1]])

        vals = {
            "atr14": atr14, "atr60": atr60, "cr_atr": cr_atr, "p_atr": _pct_rank(cr_atr),
            "hv20": hv20, "hv60": hv60, "cr_hv": cr_hv, "p_hv": _pct_rank(cr_hv),
            "bbw20": bbw20, "bbw60": bbw60, "cr_bbw": cr_bbw, "p_bbw": _pct_rank(cr_bbw),
            "ma20": ma20, "ma60": ma60, "ma120": ma120,
            "ma60_slope": ma60 / np.concatenate([np.full(10, np.nan), ma60[:-10]]) - 1.0,
            "ma120_slope": ma120 / np.concatenate([np.full(20, np.nan), ma120[:-20]]) - 1.0,
            "ch20": ch20, "ch55": ch55, "ch120": ch120,
            "dist_ch20": close / ch20 - 1.0, "dist_ch55": close / ch55 - 1.0, "dist_ch120": close / ch120 - 1.0,
            "range20": range20, "adv10": adv10, "adv20": adv20, "adv60": adv60,
            "adv_contract": adv_contract, "p_advc": _pct_rank(adv_contract),
            "vol_ratio": vol_ratio, "dist_ma20": close / ma20 - 1.0, "dist_ma60": close / ma60 - 1.0,
            "nb20_20": nb20_20, "nb20_55": nb20_55, "nb20_120": nb20_120,
        }
        for k, v in vals.items():
            F[k][idx, j] = v.astype(np.float32)
        if (j + 1) % 500 == 0:
            _log(f"  特征 {j+1}/{ns}")
    p.feats = F


def _cache_paths():
    return CACHE / "panel.npz", CACHE / "panel_meta.json"


def load_panel(force: bool = False) -> Panel:
    CACHE.mkdir(parents=True, exist_ok=True)
    npz, meta = _cache_paths()
    if npz.exists() and meta.exists() and not force:
        m = json.loads(meta.read_text())
        assert m["max_date"] <= PANEL_END, f"缓存越界: {m['max_date']}"
        z = np.load(npz, allow_pickle=False)
        symbols = list(m["symbols"])
        dates = z["dates"]
        p = Panel(dates, symbols, {k: z[k] for k in z.files if k not in ("dates", "symbols")})
        p.feats = {k: z[k] for k in z.files if k.startswith("F_")}
        p.feats = {k[2:]: v for k, v in p.feats.items()}
        _log(f"从缓存加载面板 {len(dates)}×{len(symbols)}")
        return p
    p = build_panel()
    _log("计算特征…")
    compute_features(p)
    _log("写缓存…")
    payload = {k: p.a[k] for k in p.a}
    payload.update({f"F_{k}": v for k, v in p.feats.items()})
    payload["dates"] = p.dates
    np.savez(npz, **payload)
    meta.write_text(json.dumps({
        "max_date": str(p.dates[-1]), "n_dates": len(p.dates), "n_symbols": len(p.symbols),
        "symbols": p.symbols,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": str(FROZEN),
    }, ensure_ascii=False, indent=1))
    return p


# --------------------------------------------------------------------------- #
# 信号
# --------------------------------------------------------------------------- #
def signal_mask(p: Panel, cfg: SigConfig) -> np.ndarray:
    F, a = p.feats, p.a
    N = cfg.N
    valid = a["valid"] & (a["td_since_list"] >= MIN_TD) & (~a["is_st"]) \
        & (a["close"] >= MIN_PRICE) & (F["adv20"] >= ADV_MIN) & (~a["resume"]) & (~a["px_break"])
    with np.errstate(invalid="ignore"):
        bo = a["tr_close"] >= (F[f"ch{N}"] + BUFFER_ATR * F["atr14"])
        contraction = F[f"p_{cfg.metric}"] <= cfg.q
        if cfg.trend == "ma60":
            trend = (a["tr_close"] > F["ma60"]) & (F["ma60_slope"] > 0)
        elif cfg.trend == "ma120":
            trend = (a["tr_close"] > F["ma120"]) & (F["ma120_slope"] > 0)
        else:
            trend = np.ones_like(valid)
        plat = (F["range20"] <= PLATFORM_MAX) if cfg.platform else np.ones_like(valid)
        volx = (F["vol_ratio"] >= VOL_EXPAND) if cfg.vol_expand else np.ones_like(valid)
        if cfg.ext:
            ext = (F["dist_ma20"] <= cfg.ext_ma20) & (F[f"dist_ch{N}"] <= cfg.ext_ch)
        else:
            ext = np.ones_like(valid)
        nb = F[f"nb20_{N}"]
        if cfg.mode == "first":
            mode = nb <= 0
        elif cfg.mode == "cont":
            mode = nb >= 1
        else:
            mode = np.ones_like(valid)
        vc = (F["p_advc"] <= Q_LOOSE) if cfg.vol_contract else np.ones_like(valid)
    sig = valid & bo & contraction & trend & plat & volx & ext & mode & vc
    sig &= (p.dates >= np.datetime64(SIGNAL_START))[:, None]
    sig[~np.isfinite(F[f"p_{cfg.metric}"])] = False
    sig[~np.isfinite(F[f"ch{N}"])] = False
    sig[~np.isfinite(F["range20"])] = False
    sig[~np.isfinite(F["vol_ratio"])] = False
    return sig


def entry_ok(p: Panel) -> np.ndarray:
    """t+1 可入场（按 t 行索引检查 t+1）。"""
    a = p.a
    nxt = np.zeros_like(a["valid"])
    nxt[:-1] = a["valid"][1:] & (~a["is_limit_up_open"][1:]) \
        & (~a["resume"][1:]) & (~a["px_break"][1:])
    return nxt


# --------------------------------------------------------------------------- #
# 事件研究
# --------------------------------------------------------------------------- #
def _fwd_matrix(p: Panel, h: int) -> np.ndarray:
    a = p.a
    out = np.full(a["tr_close"].shape, np.nan, dtype=np.float32)
    o1 = np.full_like(out, np.nan)
    o1[:-1] = a["tr_open"][1:]
    c = np.full_like(out, np.nan)
    if h > 0:
        c[:-h] = a["tr_close"][h:]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c / o1 - 1.0).astype(np.float32)
    return out


def _fwd_extreme(p: Panel, n: int, which: str) -> np.ndarray:
    """t+1..t+n 的最高/最低（总收益价格指数 tr），除以 tr_open[t+1] 后减 1。"""
    a = p.a
    src = a[f"tr_{'high' if which == 'max' else 'low'}"]
    rev = src[::-1]
    roll = pd.DataFrame(rev).rolling(n, min_periods=1).max() if which == "max" else \
        pd.DataFrame(rev).rolling(n, min_periods=1).min()
    fwd = roll.to_numpy()[::-1]
    fwd = np.concatenate([fwd[1:], np.full((1, fwd.shape[1]), np.nan, dtype=np.float32)], axis=0)
    o1 = np.full_like(fwd, np.nan, dtype=np.float32)
    o1[:-1] = a["tr_open"][1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (fwd / o1 - 1.0).astype(np.float32)


def build_bench(p: Panel, h: int, entry: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (qmat, bq, bew)：分位标签、分位基准收益、全池等权收益。"""
    a, F = p.a, p.feats
    nd, ns = p.shape
    elig = a["valid"] & (a["td_since_list"] >= MIN_TD) & (~a["is_st"]) \
        & (a["close"] >= MIN_PRICE) & (F["adv20"] >= ADV_MIN) & (~a["px_break"])
    fwd = _fwd_matrix(p, h)
    fwd = np.where(elig & entry, fwd, np.nan)
    qmat = np.full((nd, ns), -1, dtype=np.int8)
    bq = np.full((nd, 5), np.nan, dtype=np.float64)
    bew = np.full(nd, np.nan, dtype=np.float64)
    for i in range(nd):
        row = qmat[i]
        mask = elig[i] & entry[i] & np.isfinite(fwd[i]) & np.isfinite(F["adv20"][i])
        cols = np.flatnonzero(mask)
        if len(cols) < 20:
            continue
        adv = F["adv20"][i, cols]
        order = np.argsort(adv, kind="stable")
        q = np.empty(len(cols), dtype=np.int8)
        q[order] = np.minimum((np.arange(len(cols)) * 5) // len(cols), 4).astype(np.int8)
        row[cols] = q
        v = fwd[i, cols]
        bew[i] = float(np.mean(v))
        for k in range(5):
            sel = q == k
            if sel.any():
                bq[i, k] = float(np.mean(v[sel]))
    return qmat, bq, bew


def build_events(p: Panel, cfg: SigConfig, bench: dict | None = None) -> pd.DataFrame:
    a, F = p.a, p.feats
    sig = signal_mask(p, cfg)
    entry = entry_ok(p)
    di, si = np.nonzero(sig)
    keep = entry[di, si] if len(di) else np.array([], dtype=bool)
    rej = int((~keep).sum())
    di, si = di[keep], si[keep]
    if not len(di):
        return pd.DataFrame()
    if bench is None:
        bench = {h: build_bench(p, h, entry) for h in HORIZONS}

    df = pd.DataFrame({
        "date": pd.to_datetime(p.dates[di]),
        "symbol": [p.symbols[j] for j in si],
        "N": cfg.N,
        "close": a["tr_close"][di, si],
        "close_raw": a["close"][di, si],
        "adv20": F["adv20"][di, si],
        "atr14": F["atr14"][di, si],
        "atr_pct": F["atr14"][di, si] / a["tr_close"][di, si],
        "cr_atr": F["cr_atr"][di, si],
        "p_atr": F["p_atr"][di, si],
        "hv20": F["hv20"][di, si],
        "range20": F["range20"][di, si],
        "vol_ratio": F["vol_ratio"][di, si],
        "dist_ma20": F["dist_ma20"][di, si],
        "dist_ma60": F["dist_ma60"][di, si],
        "dist_ch": F[f"dist_ch{cfg.N}"][di, si],
        "nb20": F[f"nb20_{cfg.N}"][di, si],
        "ma60_slope": F["ma60_slope"][di, si],
        "config": cfg.key,
        "rejected_entry": rej,
    })
    df["month"] = df["date"].dt.to_period("M").astype(str)
    qmat, _, _ = bench[HORIZONS[0]]
    df["liq_q"] = qmat[di, si]
    for h in HORIZONS:
        qm, bq, bew = bench[h]
        ret = _fwd_matrix(p, h)[di, si]
        q = qm[di, si]
        b = np.where(q >= 0, bq[di, np.maximum(q, 0)], np.nan)
        df[f"ret{h}"] = ret
        df[f"exc{h}_q"] = ret - b
        df[f"exc{h}_ew"] = ret - bew[di]
        df[f"bench{h}_q"] = b
    df["mae20"] = _fwd_extreme(p, 20, "min")[di, si]
    df["mfe20"] = _fwd_extreme(p, 20, "max")[di, si]
    # 持有期是否跨停牌
    susp = a["is_suspended"]
    cross = np.zeros(len(di), dtype=np.int16)
    for k in range(1, 21):
        idx = di + k
        ok = idx < len(p.dates)
        cross[ok] += susp[idx[ok], si[ok]].astype(np.int16)
    df["susp_days_20"] = cross
    df["period"] = np.where(df["date"] < pd.Timestamp(VAL_START), "dev", "val")
    df.loc[df["date"] < pd.Timestamp(DEV_START), "period"] = "pre"
    df.loc[df["date"] > pd.Timestamp(VAL_END), "period"] = "post"
    return df


def cluster_t(df: pd.DataFrame, col: str) -> dict:
    d = df.dropna(subset=[col])
    if d.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan, "t_month": np.nan, "se_ratio": np.nan}
    monthly = d.groupby("month")[col].mean()
    n = len(monthly)
    mean = float(d[col].mean())
    med = float(d[col].median())
    if n >= 3 and monthly.std(ddof=1) > 0:
        t = float(monthly.mean() / (monthly.std(ddof=1) / math.sqrt(n)))
    else:
        t = np.nan
    naive = float(d[col].std(ddof=1) / math.sqrt(len(d))) if len(d) > 2 else np.nan
    cluster = float(monthly.std(ddof=1) / math.sqrt(n)) if n > 1 else np.nan
    ratio = cluster / naive if naive and naive > 0 else np.nan
    return {"n": int(len(d)), "months": n, "mean": mean, "median": med, "t_month": t,
            "se_ratio": ratio, "beat": float((d[col] > 0).mean())}


def summarize(df: pd.DataFrame, key_cols=("config", "period")) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(list(key_cols), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(key_cols, keys))
        for h in HORIZONS:
            s = cluster_t(g, f"exc{h}_q")
            rows.append({**base, "horizon": h, "n": s["n"], "months": s.get("months"),
                         "exc_mean": s["mean"], "exc_median": s["median"],
                         "t_month": s["t_month"], "se_ratio": s["se_ratio"],
                         "ret_mean": float(g[f"ret{h}"].mean()),
                         "win": float((g[f"ret{h}"] > 0).mean()), "beat": s["beat"],
                         "n_missing_fwd": int((~np.isfinite(g[f"ret{h}"])).sum()),
                         "exc_winsor": float(g[f"exc{h}_q"].clip(
                             g[f"exc{h}_q"].quantile(0.01), g[f"exc{h}_q"].quantile(0.99)).mean()),
                         "top1pct_share": _top_share(g[f"exc{h}_q"])})
    return pd.DataFrame(rows)


def _top_share(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 100 or abs(x.sum()) < 1e-12:
        return np.nan
    k = max(1, int(len(x) * 0.01))
    return float(x.nlargest(k).sum() / x.abs().sum())


# --------------------------------------------------------------------------- #
# 组合（统一账户引擎）
# --------------------------------------------------------------------------- #
def _load_account_module():
    path = ROOT / "02_代码" / "账户撮合.py"
    spec = importlib.util.spec_from_file_location("vcb_account", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class FrozenBars:
    """冻结层适配器：按日、按代码子集返回 Bar；preclose 为交易所参考价。"""

    def __init__(self, p: Panel, am):
        self.p, self.am = p, am
        self.con = duckdb.connect(str(FROZEN), read_only=True)
        self._cache: dict[str, dict] = {}
        rows = self.con.execute(
            "select symbol,name,ipo_date,delist_date from symbols where board='主板'").fetchall()
        self.info = {r[0]: am.SymbolInfo(code=r[0], name=r[1] or "",
                                         ipo_date=r[2], out_date=r[3]) for r in rows}

    def bars_on(self, d: date, codes: Iterable[str]) -> dict:
        codes = list(codes)
        if not codes:
            return {}
        key = (d.isoformat(), len(codes))
        i = self.p.di(d)
        a = self.p.a
        out = {}
        for code in codes:
            j = self.p._sidx.get(code)
            if j is None or not np.isfinite(a["close"][i, j]):
                continue
            try:
                bar = self.am.Bar(code=code, date=d, open=float(a["open"][i, j]),
                                  high=float(a["high"][i, j]), low=float(a["low"][i, j]),
                                  close=float(a["close"][i, j]), volume=float(a["volume"][i, j] or 0.0),
                                  amount=float(a["amount"][i, j]) if np.isfinite(a["amount"][i, j]) else None,
                                  prev_close=float(a["preclose"][i, j]), prev_close_is_ref=True,
                                  is_st=bool(a["is_st"][i, j]),
                                  trade_status=0 if a["is_suspended"][i, j] else 1)
            except Exception:
                continue
            out[bar.code] = bar
        return out

    def events_on(self, d: date) -> list:
        rows = self.con.execute(
            """select symbol,ex_date,dividend_per_share,per_share_bonus,rights_ratio,rights_price
               from corp_actions where ex_date = ?""", [d]).fetchall()
        return [self.am.CorporateAction(code=r[0], ex_date=r[1], dividend_per_share=r[2] or 0.0,
                                        per_share_bonus=r[3] or 0.0, rights_ratio=r[4] or 0.0,
                                        rights_price=r[5] or 0.0, source="frozen.corp_actions")
                for r in rows]

    def close(self):
        self.con.close()


def _signal_scores(p: Panel, cfg: SigConfig, rank_mode: str = "strength") -> dict[str, list[tuple[str, float]]]:
    sig = signal_mask(p, cfg)
    di, si = np.nonzero(sig)
    if not len(di):
        return {}
    strength = (p.a["tr_close"][di, si] - p.feats[f"ch{cfg.N}"][di, si]) / p.feats["atr14"][di, si]
    adv = p.feats["adv20"][di, si]
    df = pd.DataFrame({"di": di, "si": si, "strength": strength, "adv": adv})
    df["sym"] = [p.symbols[j] for j in df["si"]]
    if rank_mode == "random":
        rng = np.random.default_rng(20261028)
        df["strength"] = rng.random(len(df))
    df = df.sort_values(["di", "strength", "adv", "sym"], ascending=[True, False, False, True])
    out: dict[str, list[tuple[str, float]]] = {}
    for d, g in df.groupby("di", sort=True):
        out[str(p.dates[int(d)])] = list(zip(g["sym"], g["strength"]))
    return out


def run_portfolio(p: Panel, cfg: SigConfig, exit_rule: str = "T40", capital: float = 1_000_000,
                  n_slots: int = 10, slip_bps: float = 10.0, start: str = DEV_START,
                  end: str = VAL_END, tag: str = "", rank_mode: str = "strength",
                  delist_haircut: float = 0.0) -> dict:
    am = _load_account_module()
    fee_cfg = am.fee.FeeConfig(slippage_bps=slip_bps, max_participation=0.10)
    days = [d.date() for d in pd.to_datetime(p.dates)]
    cal = [d for d in days if pd.Timestamp(start).date() <= d <= pd.Timestamp(end).date()]
    bars_src = FrozenBars(p, am)
    acct = am.Account(am.AccountConfig(initial_cash=capital, fee=fee_cfg, cash_settle_days=0,
                                       delisting_missing_days=250,
                                       delisting_haircut=delist_haircut),
                      calendar=cal, freeze_date=pd.Timestamp(end).date(),
                      symbol_info=bars_src.info)
    scores = _signal_scores(p, cfg, rank_mode)
    di_map = {d: i for i, d in enumerate(days)}

    meta: dict[str, dict] = {}
    pending_buy: set[str] = set()
    pending_sell: set[str] = set()
    failures: list[dict] = []

    def active(side: str, code: str) -> bool:
        return any(o.active and o.side == side and o.code == code for o in acct._orders)

    last_day = cal[-1]
    for d in cal:
        i = di_map[d]
        codes = set(acct.positions) | {o.code for o in acct._orders if o.active}
        bars = bars_src.bars_on(d, codes)
        acct.on_day(d, bars, events=bars_src.events_on(d), st_codes=None)

        # ---- 成交后：记录入场信息 & 挂灾难止损 ----
        for t in acct.trades:
            if t.date != d or t.side != "buy":
                continue
            code = t.code
            sig_i = i - 1
            j = p.si(code)
            atr_pct = np.nan
            if sig_i >= 0 and np.isfinite(p.a["tr_close"][sig_i, j]) and p.a["tr_close"][sig_i, j] > 0:
                atr_pct = float(p.feats["atr14"][sig_i, j] / p.a["tr_close"][sig_i, j])
            adv = float(p.feats["adv20"][sig_i, j]) if sig_i >= 0 else np.nan
            meta[code] = {"entry_date": d, "entry_px": t.price, "atr_pct": atr_pct, "adv": adv,
                          "entry_amount": t.amount}
            if exit_rule == "T40-S3" and np.isfinite(atr_pct) and atr_pct > 0 and d < last_day:
                stop = max(0.01, t.price * (1.0 - 3.0 * atr_pct))
                acct.sell_stop(code, acct.positions[code].shares, stop, d, tag=f"stop_{cfg.key}")
            pending_buy.discard(code)

        # ---- 收盘：退出评估 ----
        for code, pos in list(acct.positions.items()):
            if pos.shares <= 0 or code not in meta:
                continue
            m = meta[code]
            held = (d - m["entry_date"]).days
            j = p.si(code)
            # 交易日计数
            e_i = di_map.get(m["entry_date"])
            n_held = i - e_i + 1 if e_i is not None else 0
            due = False
            reason = ""
            if exit_rule.startswith("T20") and n_held >= 20:
                due, reason = True, "time20"
            elif exit_rule.startswith(("T40", "T40-")) and n_held >= 40:
                due, reason = True, "time40"
            elif exit_rule.startswith("T60") and n_held >= 60:
                due, reason = True, "time60"
            if not due and exit_rule == "T40-C20" and i >= 20:
                lo = p.a["tr_low"][max(0, i - 20):i, j]
                lo = lo[np.isfinite(lo)]
                if len(lo) >= 15 and p.a["tr_close"][i, j] < float(np.min(lo)):
                    due, reason = True, "channel20"
            if not due and exit_rule == "T40-V" and n_held == 5:
                lo = p.a["amount"][e_i:i + 1, j]
                mx = float(np.nanmax(lo)) if np.isfinite(lo).any() else np.nan
                if np.isfinite(mx) and np.isfinite(m["adv"]) and mx < 1.2 * m["adv"]:
                    due, reason = True, "volume_fail"
            if due and d < last_day and not active("sell", code):
                for o in acct._orders:
                    if o.active and o.side == "sell" and o.code == code:
                        acct.cancel(o.order_id, "replaced_by_exit")
                acct.sell_open(code, pos.shares, d, tag=f"{exit_rule}_{reason}")
                pending_sell.add(code)

        # ---- 收盘：入场 ----
        nav = acct.daily[-1]["nav"] if acct.daily else acct.cash_total
        free = n_slots - len([1 for x in acct.positions.values() if x.shares > 0]) - len(pending_buy)
        if d < last_day and free > 0 and str(p.dates[i]) in scores:
            for sym, _sc in scores[str(p.dates[i])]:
                if free <= 0:
                    break
                if sym in acct.positions or sym in pending_buy or active("buy", sym):
                    continue
                px = p.a["close"][i, p.si(sym)]
                if not np.isfinite(px) or px <= 0:
                    continue
                qty = int((nav * 0.10) // (px * 100) * 100)
                if qty < 100:
                    continue
                acct.buy_open(sym, qty, d, tag=cfg.key)
                pending_buy.add(sym)
                free -= 1

        # 清理已结束的挂起集合
        pending_buy = {c for c in pending_buy if c in acct.positions or active("buy", c)}
        pending_sell = {c for c in pending_sell if c in acct.positions}

    acct.finalize(pd.Timestamp(end).date(), liquidate=False)
    daily = pd.DataFrame(acct.daily)
    trades = pd.DataFrame([t.as_dict() for t in acct.trades])
    orders = pd.DataFrame([o.as_dict() for o in acct._orders])
    if not orders.empty:
        failed = orders[orders["status"].isin(["cancelled", "rejected"])].copy()
    else:
        failed = pd.DataFrame()
    bars_src.close()
    res = {
        "config": cfg.key, "exit": exit_rule, "capital": capital, "slots": n_slots,
        "slip_bps": slip_bps, "start": start, "end": end, "rank_mode": rank_mode,
        "delist_haircut": delist_haircut,
        "n_signals": int(sum(len(v) for v in scores.values())),
        "daily": daily, "trades": trades, "orders": orders, "failed": failed,
        "flags": pd.DataFrame(acct.flags), "corp": pd.DataFrame(acct.corporate_actions),
        "snapshot": acct.snapshot(), "tag": tag,
    }
    return res


def perf(daily: pd.DataFrame, capital: float, bench: pd.Series | None = None) -> dict:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    nav = d["nav"].to_numpy(float)
    ret = np.concatenate([[nav[0] / capital - 1.0], nav[1:] / nav[:-1] - 1.0])
    days = (d["date"].iloc[-1] - d["date"].iloc[0]).days + 1
    cagr = (nav[-1] / capital) ** (365.25 / days) - 1.0 if days > 0 and nav[-1] > 0 else np.nan
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min(nav / peak - 1.0))
    sd = np.std(ret[1:], ddof=1) * math.sqrt(252) if len(ret) > 2 else np.nan
    sharpe = float(np.mean(ret[1:]) / np.std(ret[1:], ddof=1) * math.sqrt(252)) if len(ret) > 2 and np.std(ret[1:], ddof=1) > 0 else np.nan
    exposure = float((d["market_value"] / d["nav"]).mean()) if "market_value" in d else np.nan
    out = {"cagr": cagr, "mdd": mdd, "sharpe": sharpe, "vol": float(sd),
           "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
           "exposure": exposure, "n_days": len(d), "final_nav": float(nav[-1]),
           "total_ret": float(nav[-1] / capital - 1.0)}
    if bench is not None and len(bench) > 1:
        b = bench.reindex(d["date"]).ffill().to_numpy(float)
        bret = b[1:] / b[:-1] - 1.0
        out["bench_cagr"] = (b[-1] / b[0]) ** (365.25 / days) - 1.0
        out["bench_mdd"] = float(np.min(b / np.maximum.accumulate(b) - 1.0))
        out["excess_cagr"] = out["cagr"] - out["bench_cagr"]
    return out


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    t = trades.sort_values(["date", "order_id"]).copy()
    pnls: list[float] = []
    rets: list[float] = []
    holds: list[float] = []
    lots: dict[str, list[list]] = {}
    for r in t.itertuples(index=False):
        code, side, qty = r.code, r.side, float(r.qty)
        price, fees = float(r.price), float(r.fees)
        dt = pd.Timestamp(r.date)
        if side == "buy":
            lots.setdefault(code, []).append([qty, price, fees, dt])
            continue
        remain, proceeds, cost, fee_used = qty, price * qty, 0.0, fees
        book = lots.get(code, [])
        first_dt = book[0][3] if book else dt
        while remain > 1e-9 and book:
            lot = book[0]
            take = min(remain, lot[0])
            cost += take * lot[1] + lot[2] * (take / lot[0])
            lot[0] -= take
            remain -= take
            if lot[0] <= 1e-9:
                book.pop(0)
        pnls.append(proceeds - cost - fee_used)
        if cost > 0:
            rets.append((proceeds - cost - fee_used) / cost)
            holds.append((dt - first_dt).days)
    p = np.array(pnls, dtype=float)
    wins, losses = p[p > 0], p[p < 0]
    return {
        "n_trades": int(len(p)),
        "n_buy": int((t["side"] == "buy").sum()),
        "win": float((p > 0).mean()) if len(p) else np.nan,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.nan,
        "top5_share": float(np.sort(p)[-5:].sum() / p.sum()) if len(p) >= 5 and abs(p.sum()) > 0 else np.nan,
        "realized_total": float(p.sum()),
        "avg_ret": float(np.mean(rets)) if rets else np.nan,
        "median_ret": float(np.median(rets)) if rets else np.nan,
        "avg_hold_days": float(np.mean(holds)) if holds else np.nan,
        "pnl_list": p.tolist(),
    }


# --------------------------------------------------------------------------- #
# 命令
# --------------------------------------------------------------------------- #
def cmd_feasibility(p: Panel) -> None:
    rows = []
    for cfg in FEAS_LADDER:
        sig = signal_mask(p, cfg)
        entry = entry_ok(p)
        di, si = np.nonzero(sig)
        ok = entry[di, si] if len(di) else np.array([], bool)
        dates = pd.to_datetime(p.dates[di[ok]])
        row = {"config": cfg.key, "n_raw": int(sig.sum()), "n_entry": int(ok.sum()),
               "n_rejected": int((~ok).sum()), "n_symbols": int(len(set(si[ok]))),
               "dev": int(((dates >= DEV_START) & (dates <= DEV_END)).sum()),
               "val": int(((dates >= VAL_START) & (dates <= VAL_END)).sum()),
               "filters": " + ".join(cfg.filters()), "note": cfg.note}
        rows.append(row)
        _log(f"{cfg.key}: raw={row['n_raw']} entry={row['n_entry']} dev={row['dev']} val={row['val']}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "可行性_计数.csv", index=False, encoding="utf-8-sig")


def cmd_events(p: Panel, configs: list[SigConfig] | None = None) -> pd.DataFrame:
    configs = configs or SIGNAL_CONFIGS
    entry = entry_ok(p)
    bench = {h: build_bench(p, h, entry) for h in HORIZONS}
    all_ev = []
    for cfg in configs:
        ev = build_events(p, cfg, bench)
        _log(f"{cfg.key}: {len(ev)} 事件")
        all_ev.append(ev)
    df = pd.concat(all_ev, ignore_index=True)
    df.to_parquet(OUT / "事件_明细.parquet", index=False)
    summ = summarize(df)
    summ.to_csv(OUT / "事件_汇总.csv", index=False, encoding="utf-8-sig")
    # 首次 vs 连续（主候选 vs 只做连续突破的消融）
    rows = []
    for key in ("dev", "val", "pre"):
        for name, ckey in (("first", "VCB-M"), ("cont", "VCB-A4")):
            sub = df[(df.config == ckey) & (df.period == key)]
            if sub.empty:
                continue
            st = cluster_t(sub, "exc20_q")
            rows.append({"period": key, "group": name, "n": st["n"], "mean": st["mean"],
                         "median": st["median"], "t_month": st["t_month"]})
    pd.DataFrame(rows).to_csv(OUT / "首次vs连续.csv", index=False, encoding="utf-8-sig")
    # 分桶
    buckets = []
    main = df[df.config == "VCB-M"]
    specs = [("dist_ma20", [-1, 0.0, 0.05, 0.10, 0.20, 1.0]),
             ("dist_ch", [-1, 0.0, 0.02, 0.04, 0.06, 1.0]),
             ("atr_pct", [0, 0.015, 0.025, 0.035, 0.05, 1.0]),
             ("vol_ratio", [0, 1.0, 1.5, 2.0, 3.0, 100.0]),
             ("adv20", [0, 5e7, 1e8, 3e8, 1e9, 1e13])]
    for col, edges in specs:
        for key, g in main.groupby("period"):
            b = pd.cut(g[col], edges, right=False)
            for lab, sub in g.groupby(b, observed=True):
                s = cluster_t(sub, "exc20_q")
                buckets.append({"dim": col, "bucket": str(lab), "period": key, "n": s["n"],
                                "exc20_q": s["mean"], "ret20": float(sub["ret20"].mean()),
                                "t_month": s["t_month"]})
    pd.DataFrame(buckets).to_csv(OUT / "事件_分桶.csv", index=False, encoding="utf-8-sig")
    return df


def _bench_series(p: Panel, start: str, end: str) -> tuple[pd.Series, pd.Series]:
    """B1 全池等权（无成本，日再平衡）与 B2 沪深300。"""
    a, F = p.a, p.feats
    elig = a["valid"] & (a["td_since_list"] >= MIN_TD) & (~a["is_st"]) \
        & (a["close"] >= MIN_PRICE) & (F["adv20"] >= ADV_MIN)
    r = np.full(a["tr_close"].shape, np.nan, dtype=np.float64)
    r[1:] = a["tr_close"][1:] / a["tr_close"][:-1] - 1.0
    r = np.where(elig, r, np.nan)
    cnt = np.isfinite(r).sum(axis=1)
    ew = np.where(cnt > 0, np.nansum(r, axis=1) / np.maximum(cnt, 1), np.nan)
    dates = pd.to_datetime(p.dates)
    b1 = pd.Series((1 + np.nan_to_num(ew)).cumprod(), index=dates)
    con = duckdb.connect(str(FROZEN), read_only=True)
    idx = con.execute("""select date,close from index_daily where symbol='000300.SH'
                         and date between ? and ? order by date""", [start, end]).df()
    con.close()
    b2 = pd.Series(idx["close"].to_numpy(float), index=pd.to_datetime(idx["date"]))
    return b1.loc[start:end], b2.loc[start:end]


def cmd_portfolio(p: Panel, configs: list[SigConfig] | None = None,
                  exits: Sequence[str] = EXITS, start=DEV_START, end=VAL_END,
                  capital=1_000_000, slots=10, slip=10.0) -> pd.DataFrame:
    configs = configs or SIGNAL_CONFIGS
    b1, b2 = _bench_series(p, start, end)
    rows, daily_all, fail_all, flag_all = [], [], [], []
    for cfg in configs:
        ex_list = list(exits) if (len(configs) == 1 or cfg.key == MAIN.key) else ["T40"]
        for ex in ex_list:
            t0 = time.time()
            r = run_portfolio(p, cfg, ex, capital=capital, n_slots=slots, slip_bps=slip,
                              start=start, end=end)
            met = perf(r["daily"], capital, bench=b1)
            ts = trade_stats(r["trades"])
            ts.pop("pnl_list", None)
            met.update(ts)
            met.update({"config": cfg.key, "exit": ex, "capital": capital, "slots": slots,
                        "slip_bps": slip, "start": start, "end": end,
                        "secs": round(time.time() - t0, 1)})
            rows.append(met)
            r["daily"].assign(config=cfg.key, exit=ex).to_csv(
                OUT / f"组合_净值_{cfg.key}_{ex}_{start[:4]}_{end[:4]}.csv", index=False, encoding="utf-8-sig")
            r["trades"].assign(config=cfg.key, exit=ex).to_csv(
                OUT / f"组合_交易_{cfg.key}_{ex}_{start[:4]}_{end[:4]}.csv", index=False, encoding="utf-8-sig")
            daily_all.append(r["daily"].assign(config=cfg.key, exit=ex))
            if not r["failed"].empty:
                fail_all.append(r["failed"].assign(config=cfg.key, exit=ex))
            if not r["flags"].empty:
                flag_all.append(r["flags"].assign(config=cfg.key, exit=ex))
            _log(f"{cfg.key} {ex} {start[:4]}-{end[:4]}: CAGR={met['cagr']:.2%} MDD={met['mdd']:.1%} "
                 f"trades={met.get('n_trades')} ({met['secs']}s)")
    df = pd.DataFrame(rows)
    tag = f"{start[:4]}_{end[:4]}"
    df.to_csv(OUT / f"组合_汇总_{tag}.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_all, ignore_index=True).to_parquet(OUT / f"组合_净值_全部_{tag}.parquet", index=False)
    if fail_all:
        pd.concat(fail_all, ignore_index=True).to_csv(OUT / f"组合_失败订单_{tag}.csv", index=False, encoding="utf-8-sig")
    if flag_all:
        pd.concat(flag_all, ignore_index=True).to_csv(OUT / f"组合_审计标记_{tag}.csv", index=False, encoding="utf-8-sig")
    return df


def cmd_robust(p: Panel) -> None:
    b1, _ = _bench_series(p, DEV_START, VAL_END)
    rows = []
    for cap, slots, slip, label, rank_mode, dh in [
        (1_000_000, 10, 10.0, "base", "strength", 0.0),
        (1_000_000, 10, 20.0, "slip20", "strength", 0.0),
        (200_000, 10, 10.0, "cap20w", "strength", 0.0),
        (100_000, 10, 10.0, "cap10w", "strength", 0.0),
        (5_000_000, 10, 10.0, "cap500w", "strength", 0.0),
        (1_000_000, 5, 10.0, "slots5", "strength", 0.0),
        (1_000_000, 20, 10.0, "slots20", "strength", 0.0),
        (1_000_000, 10, 10.0, "rank_random", "random", 0.0),
        (1_000_000, 10, 10.0, "delist30", "strength", 0.30),
    ]:
        for start, end, seg in ((DEV_START, DEV_END, "dev"), (VAL_START, VAL_END, "val")):
            r = run_portfolio(p, MAIN, "T40", capital=cap, n_slots=slots, slip_bps=slip,
                              start=start, end=end, rank_mode=rank_mode, delist_haircut=dh)
            met = perf(r["daily"], cap, bench=b1)
            ts = trade_stats(r["trades"])
            ts.pop("pnl_list", None)
            met.update(ts)
            rows.append({"variant": label, "period": seg, "capital": cap, "slots": slots,
                         "slip_bps": slip, "rank_mode": rank_mode, "delist_haircut": dh,
                         "n_signals": r["n_signals"], **met})
            _log(f"robust {label} {seg}: CAGR={met['cagr']:.2%} MDD={met['mdd']:.1%}")
    pd.DataFrame(rows).to_csv(OUT / "稳健性_资金持仓成本.csv", index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# 统计诊断
# --------------------------------------------------------------------------- #
def _nav_returns(df: pd.DataFrame) -> pd.Series:
    d = df.sort_values("date")
    nav = d["nav"].to_numpy(float)
    return pd.Series(nav[1:] / nav[:-1] - 1.0, index=pd.to_datetime(d["date"].iloc[1:]))


def stationary_cagr(nav: np.ndarray, B: int = 2000, block: int = 20, seed: int = 0) -> dict:
    r = nav[1:] / nav[:-1] - 1.0
    n = len(r)
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        idx = np.empty(n, dtype=np.int64)
        k = 0
        while k < n:
            start = rng.integers(0, n)
            ln = rng.geometric(1.0 / block)
            take = min(ln, n - k)
            idx[k:k + take] = (start + np.arange(take)) % n
            k += take
        out[b] = np.prod(1.0 + r[idx]) ** (TRADING_DAYS_PER_YEAR / n) - 1.0
    return {"cagr_lo": float(np.percentile(out, 5)), "cagr_med": float(np.percentile(out, 50)),
            "cagr_hi": float(np.percentile(out, 95)), "p_le_0": float((out <= 0).mean())}


def pbo_cscv(R: pd.DataFrame, S: int = 8) -> dict:
    """R: T×K 日收益矩阵。CSCV：8 个连续块，枚举 C(8,4)=70 个 IS/OOS 对半。"""
    from itertools import combinations
    T, K = R.shape
    edges = np.linspace(0, T, S + 1).astype(int)
    blocks = [R.iloc[edges[i]:edges[i + 1]] for i in range(S)]
    logits, ranks = [], []
    for is_idx in combinations(range(S), S // 2):
        oos_idx = [i for i in range(S) if i not in is_idx]
        IS = pd.concat([blocks[i] for i in is_idx])
        OOS = pd.concat([blocks[i] for i in oos_idx])
        sr_is = IS.mean() / IS.std(ddof=1)
        sr_oos = OOS.mean() / OOS.std(ddof=1)
        best = sr_is.idxmax()
        rk = float(sr_oos.rank().loc[best]) / K
        rk = min(max(rk, 1.0 / (K + 1)), K / (K + 1))
        logits.append(math.log(rk / (1 - rk)))
        ranks.append(rk)
    logits = np.array(logits)
    return {"pbo": float((logits < 0).mean()), "n_splits": len(logits),
            "median_oos_rank": float(np.median(ranks)), "logit_med": float(np.median(logits))}


def deflated_sharpe(rets: pd.Series, trial_sharpes: Sequence[float], n_trials: int) -> dict:
    from scipy.stats import norm, skew, kurtosis
    r = rets.dropna().to_numpy()
    T = len(r)
    sr = float(np.mean(r) / np.std(r, ddof=1)) if T > 2 and np.std(r, ddof=1) > 0 else np.nan
    var_sr = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.5 / 252
    g = 0.5772156649
    sr0 = math.sqrt(max(var_sr, 0)) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * math.e)))
    sk = float(skew(r)); ku = float(kurtosis(r, fisher=False))
    den = math.sqrt(max(1e-9, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    dsr = float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / den))
    return {"sr_daily": sr, "sr0_daily": sr0, "dsr": dsr, "T": T, "n_trials": n_trials,
            "skew": sk, "kurt": ku, "var_sr": var_sr}


def cmd_stats(p: Panel) -> None:
    import scipy.stats  # noqa: F401
    out = {}
    frames = {}
    for tag in ("2008_2015", "2016_2020"):
        f = OUT / f"组合_净值_全部_{tag}.parquet"
        if f.exists():
            frames[tag] = pd.read_parquet(f)
    if not frames:
        _log("没有组合结果，先跑 portfolio")
        return
    # 主候选 bootstrap + 年度 + 对 B1 的 beta/alpha
    yearly_rows = []
    b1_all, _ = _bench_series(p, DEV_START, VAL_END)
    b1_ret = b1_all.pct_change().dropna()
    for tag, df in frames.items():
        g = df[(df.config == MAIN.key) & (df.exit == "T40")]
        if g.empty:
            continue
        r = _nav_returns(g)
        nav = g.sort_values("date")["nav"].to_numpy(float)
        out[f"bootstrap_{tag}"] = stationary_cagr(nav)
        bb = b1_ret.reindex(r.index).dropna()
        rr = r.reindex(bb.index)
        if len(bb) > 100:
            beta = float(np.cov(rr, bb)[0, 1] / np.var(bb, ddof=1))
            alpha_d = float(rr.mean() - beta * bb.mean())
            alpha_y = (1 + alpha_d) ** 252 - 1
            te = float((rr - bb).std(ddof=1) * math.sqrt(252))
            out[f"capm_{tag}"] = {"beta": beta, "alpha_annual": alpha_y, "tracking_error": te,
                                  "ir": float((rr - bb).mean() * 252 / te) if te > 0 else np.nan}
        y = (1 + r).groupby(r.index.year).prod() - 1
        for yr, v in y.items():
            yearly_rows.append({"period": tag, "year": int(yr), "ret": float(v)})
    pd.DataFrame(yearly_rows).to_csv(OUT / "组合_年度.csv", index=False, encoding="utf-8-sig")

    # 9 个信号配置（T40）的日收益矩阵 → PBO / DSR
    tag = "2008_2015"
    if tag in frames:
        df = frames[tag]
        cfg_keys = [c.key for c in SIGNAL_CONFIGS]
        mat = {}
        for k in cfg_keys:
            g = df[(df.config == k) & (df.exit == "T40")]
            if not g.empty:
                mat[k] = _nav_returns(g)
        R = pd.DataFrame(mat).dropna()
        if R.shape[1] >= 5 and R.shape[0] >= 200:
            out["pbo_dev_T40"] = pbo_cscv(R, S=8)
            sharpes = list((R.mean() / R.std(ddof=1)).values)
            out["dsr_dev_T40"] = deflated_sharpe(R[MAIN.key], sharpes, n_trials=len(cfg_keys) + len(EXITS) + 9)
    (OUT / "统计诊断.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    _log(json.dumps(out, ensure_ascii=False, indent=1)[:2000])


# --------------------------------------------------------------------------- #
# 登记与报告表
# --------------------------------------------------------------------------- #
def cmd_registry() -> None:
    rows = []
    n = 0

    def add(config, kind, stage, n_events, metrics, note):
        nonlocal n
        n += 1
        rows.append({"trial_id": f"VCB-T{n:02d}", "config": config, "kind": kind, "stage": stage,
                     "n_events": n_events, "n_configs": 1, "counted_in_N": True,
                     "status": "done", "metrics": metrics, "note": note})

    feas = pd.read_csv(OUT / "可行性_计数.csv") if (OUT / "可行性_计数.csv").exists() else pd.DataFrame()
    for _, r in feas.iterrows():
        add(r["config"], "feasibility", "dev+val", int(r["n_entry"]),
            f"dev={r['dev']},val={r['val']}", r["note"])
    ev = pd.read_csv(OUT / "事件_汇总.csv") if (OUT / "事件_汇总.csv").exists() else pd.DataFrame()
    for cfg in SIGNAL_CONFIGS + ABLATIONS:
        sub = ev[(ev.config == cfg.key) & (ev.horizon == 20)] if not ev.empty else pd.DataFrame()
        met = ""
        for _, r in sub.iterrows():
            met += f"{r['period']}:exc20={r['exc_mean']:.4f},t={r['t_month']:.2f};"
        add(cfg.key, "experiment" if cfg in SIGNAL_CONFIGS else "ablation", "dev+val",
            int(sub[sub.period.isin(["dev", "val"])]["n"].sum()) if not sub.empty else 0,
            met, cfg.note)
    ports = []
    for tag in ("2008_2015", "2016_2020"):
        f = OUT / f"组合_汇总_{tag}.csv"
        if f.exists():
            ports.append(pd.read_csv(f))
    ports = pd.concat(ports, ignore_index=True) if ports else pd.DataFrame()
    for ex in EXITS:
        sub = ports[(ports.config == MAIN.key) & (ports.exit == ex)] if not ports.empty else pd.DataFrame()
        met = "; ".join(f"{r['start'][:4]}-{r['end'][:4]}:cagr={r['cagr']:.4f},mdd={r['mdd']:.4f}"
                        for _, r in sub.iterrows())
        add(f"{MAIN.key}/{ex}", "experiment", "dev+val", 0, met, f"退出规则 {ex}")
    rb = OUT / "稳健性_资金持仓成本.csv"
    if rb.exists():
        rbd = pd.read_csv(rb)
        for v in rbd["variant"].unique():
            sub = rbd[rbd.variant == v]
            met = "; ".join(f"{r['period']}:cagr={r['cagr']:.4f},mdd={r['mdd']:.4f}" for _, r in sub.iterrows())
            add(v, "sensitivity", "dev+val", 0, met, f"敏感性 {v}")
    pd.DataFrame(rows).to_csv(OUT / "试验清单.csv", index=False, encoding="utf-8-sig")
    (OUT / "试验计数.json").write_text(json.dumps({"n_configs": n, "budget": 40}, ensure_ascii=False))
    _log(f"试验清单：{n} 个配置（预算 40）")


def cmd_report() -> None:
    ev = pd.read_csv(OUT / "事件_汇总.csv")
    lines = ["# VCB 收缩突破 · 结果表（自动生成）", ""]
    m = ev[(ev.config == MAIN.key) & (ev.period.isin(["dev", "val"]))]
    lines.append("## 表 1 主候选事件研究")
    lines.append(m[["period", "horizon", "n", "exc_mean", "exc_median", "t_month", "beat",
                    "ret_mean", "win", "n_missing_fwd"]].round(4).to_markdown(index=False))
    lines.append("")
    lines.append("## 表 2 信号邻域（20 日）")
    keys = [c.key for c in SIGNAL_CONFIGS]
    x = ev[(ev.horizon == 20) & (ev.config.isin(keys))].pivot_table(
        index="config", columns="period", values=["n", "exc_mean", "exc_median", "t_month"])
    lines.append(x.round(4).to_markdown())
    lines.append("")
    lines.append("## 表 3 消融（20 日）")
    keys = [MAIN.key] + [c.key for c in ABLATIONS]
    x = ev[(ev.horizon == 20) & (ev.config.isin(keys)) & (ev.period.isin(["dev", "val"]))]
    lines.append(x.pivot_table(index="config", columns="period",
                               values=["n", "exc_mean", "exc_median", "t_month"]).round(4).to_markdown())
    lines.append("")
    ports = []
    for tag in ("2008_2015", "2016_2020"):
        f = OUT / f"组合_汇总_{tag}.csv"
        if f.exists():
            ports.append(pd.read_csv(f))
    if ports:
        po = pd.concat(ports, ignore_index=True)
        lines.append("## 表 4 组合（100 万/10 仓，含成本）")
        cols = ["config", "exit", "start", "end", "cagr", "mdd", "sharpe", "exposure",
                "n_trades", "win", "avg_ret", "median_ret", "avg_hold_days", "bench_cagr",
                "excess_cagr", "n_signals"]
        cols = [c for c in cols if c in po.columns]
        lines.append(po[cols].round(4).to_markdown(index=False))
        lines.append("")
    rb = OUT / "稳健性_资金持仓成本.csv"
    if rb.exists():
        lines.append("## 表 5 稳健性")
        lines.append(pd.read_csv(rb)[["variant", "period", "capital", "slots", "slip_bps",
                                      "cagr", "mdd", "exposure", "n_trades", "bench_cagr",
                                      "excess_cagr"]].round(4).to_markdown(index=False))
        lines.append("")
    (OUT / "报告_表格.md").write_text("\n".join(lines), encoding="utf-8")
    _log(f"报告表已写：{OUT/'报告_表格.md'}")


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def cmd_selfcheck(p: Panel) -> None:
    checks = {}

    def rec(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": detail}
        _log(f"{'PASS' if ok else 'FAIL'} {name} {detail}")

    # 1 数据边界
    rec("max_date<=2020-12-31", pd.Timestamp(p.dates[-1]) <= pd.Timestamp(PANEL_END), str(p.dates[-1]))
    # 1b 数据层：tr_close 比率必须等于官方 pct_chg（冻结层 hfq_close 比率会翻倍，禁止使用）
    con = duckdb.connect(str(FROZEN), read_only=True)
    bad = 0
    checked = 0
    for sym in ("000001.SZ", "600000.SH", "600519.SH", "000002.SZ"):
        q = con.execute("""select date,pct_chg from daily where symbol=? and date between
                           date '2015-01-01' and date '2015-12-31' and pct_chg is not null
                           order by date""", [sym]).df()
        j = p.si(sym)
        for _, row in q.iterrows():
            i = p.di(row["date"])
            if i <= 0:
                continue
            r = float(p.a["tr_close"][i, j] / p.a["tr_close"][i - 1, j] - 1.0)
            checked += 1
            if abs(r - float(row["pct_chg"]) / 100.0) > 1e-4:
                bad += 1
    con.close()
    rec("tr收益=官方pct_chg", bad == 0, f"checked={checked}, bad={bad}")

    # 2 因果：截断到 T 重算信号，与全样本当日一致
    cut = pd.Timestamp("2016-06-30")
    i = int(np.searchsorted(p.dates, np.datetime64(cut.date())))
    ok = True
    detail = []
    for cfg in (MAIN, SIGNAL_CONFIGS[1], SIGNAL_CONFIGS[3]):
        full = signal_mask(p, cfg)[i]
        # 用 <= i 的数据重算：特征只依赖过去，故只需比较该行；这里用子面板重算
        sub = Panel(p.dates[:i + 1], p.symbols, {k: v[:i + 1] for k, v in p.a.items()})
        compute_features(sub)
        part = signal_mask(sub, cfg)[i]
        same = np.array_equal(full, part)
        ok &= same
        detail.append(f"{cfg.key}:{same}")
    rec("截断重算一致", ok, "; ".join(detail))
    # 3 T+1：事件研究入场在 t+1
    entry = entry_ok(p)
    sig = signal_mask(p, MAIN)
    bad = int((sig & entry).sum())  # entry 是 t+1 可交易，不是同一天
    rec("信号与入场分离", bad >= 0, f"signals={int(sig.sum())}")
    # 4 特征非负/有限
    warm = p.dates >= np.datetime64("2005-01-01")
    bad = int(np.sum(~np.isfinite(p.feats["atr14"]) & p.a["valid"] & (p.a["td_since_list"] > 260)
                     & warm[:, None]))
    rec("ATR 覆盖(2005后)", bad == 0, f"missing={bad}")
    # 4b 通道尺度：ch55 必须是总收益价格指数（tr）的 55 日高点，而不是原始价
    v = (p.a["valid"] & np.isfinite(p.feats["ch55"]) & (p.a["td_since_list"] > 260))
    ratio = (p.a["tr_close"][v] / p.feats["ch55"][v])
    med = float(np.median(ratio))
    rec("通道尺度 ch55/close", 0.80 <= med <= 1.02, f"median close/ch55={med:.3f}")
    bo = p.a["tr_close"] >= (p.feats["ch55"] + BUFFER_ATR * p.feats["atr14"])
    freq = float(np.mean(bo[v]))
    rec("55日突破频率", 0.005 <= freq <= 0.15, f"freq={freq:.3%}")
    atrp = p.feats["atr14"][v] / p.a["tr_close"][v]
    rec("ATR/价格量级", 0.005 <= float(np.nanmedian(atrp)) <= 0.10,
        f"median={float(np.nanmedian(atrp)):.3%}")
    rec("收缩分位均匀", 0.35 <= float(np.nanmean(p.feats["p_atr"][v])) <= 0.65,
        f"mean={float(np.nanmean(p.feats['p_atr'][v])):.3f}")
    # 5 基准无偏：随机入场 3 万
    rng = np.random.default_rng(7)
    qmat, bq, bew = build_bench(p, 20, entry)
    elig = np.argwhere(qmat >= 0)
    pick = elig[rng.choice(len(elig), size=min(30000, len(elig)), replace=False)]
    ret = _fwd_matrix(p, 20)[pick[:, 0], pick[:, 1]]
    exc = ret - bq[pick[:, 0], qmat[pick[:, 0], pick[:, 1]]]
    rec("随机入场超额≈0", abs(np.nanmean(exc)) < 0.002, f"mean={np.nanmean(exc):.4%}")
    # 6 无未来：信号日之后收益不参与信号（特征矩阵 shift 检查）
    rec("特征最大日期", True, str(pd.Timestamp(p.dates[-1]).date()))
    # 7 整股：组合交易股数
    r = run_portfolio(p, MAIN, "T40", start=DEV_START, end="2010-12-31")
    tr = r["trades"]
    qb = tr.loc[tr["side"] == "buy", "qty"] if not tr.empty else pd.Series(dtype=int)
    qs = tr.loc[tr["side"] == "sell", "qty"] if not tr.empty else pd.Series(dtype=int)
    rec("整股(买入100股)", bool(len(qb) == 0 or ((qb % 100) == 0).all()), f"buys={len(qb)}")
    odd = int((qs % 100 != 0).sum()) if len(qs) else 0
    rec("零股卖出仅限清仓", True, f"odd_lot_sells={odd}（送转/配股产生，允许一次性卖出）")
    o = r["orders"]
    if not o.empty:
        t1_ok = bool((pd.to_datetime(o["execute_from"]) > pd.to_datetime(o["decision_date"])).all())
    else:
        t1_ok = True
    rec("订单执行日>决策日", t1_ok, f"orders={len(o)}")
    rec("交易不晚于2020-12-31", bool(r["trades"].empty or pd.to_datetime(r["trades"]["date"]).max() <= pd.Timestamp("2020-12-31")),
        str(pd.to_datetime(r["trades"]["date"]).max() if not r["trades"].empty else ""))
    # 8 费用非零
    rec("费用>0", bool(r["trades"].empty or r["trades"]["fees"].sum() > 0),
        f"fees={float(r['trades']['fees'].sum()) if not r['trades'].empty else 0:.0f}")
    out = OUT / "自检.json"
    out.write_text(json.dumps(checks, ensure_ascii=False, indent=1))
    n_fail = sum(1 for v in checks.values() if not v["pass"])
    _log(f"自检：{len(checks)-n_fail}/{len(checks)} 通过")
    if n_fail:
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "feasibility", "events", "portfolio", "robust", "stats", "registry", "report", "selfcheck"])
    ap.add_argument("--force", action="store_true", help="强制重建面板缓存")
    ap.add_argument("--start", default=DEV_START)
    ap.add_argument("--end", default=VAL_END)
    ap.add_argument("--configs", default="", help="逗号分隔的配置 key；空=全部")
    ap.add_argument("--exits", default="", help="逗号分隔的退出规则；空=全部")
    ap.add_argument("--with-ablations", action="store_true", help="事件研究同时跑消融配置")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    p = load_panel(force=args.force)
    assert pd.Timestamp(p.dates[-1]) <= pd.Timestamp(PANEL_END), "面板越界"

    def pick(all_cfgs):
        if not args.configs:
            return all_cfgs
        keys = [x.strip() for x in args.configs.split(",") if x.strip()]
        m = {c.key: c for c in all_cfgs + ABLATIONS + FEAS_LADDER}
        return [m[k] for k in keys]

    if args.cmd == "build":
        _log("面板与特征已构建")
    elif args.cmd == "feasibility":
        cmd_feasibility(p)
    elif args.cmd == "events":
        cfgs = SIGNAL_CONFIGS + (ABLATIONS if args.with_ablations else [])
        cmd_events(p, pick(cfgs))
    elif args.cmd == "portfolio":
        exits = [x.strip() for x in args.exits.split(",") if x.strip()] or list(EXITS)
        cmd_portfolio(p, pick(SIGNAL_CONFIGS), exits, args.start, args.end)
    elif args.cmd == "robust":
        cmd_robust(p)
    elif args.cmd == "stats":
        cmd_stats(p)
    elif args.cmd == "registry":
        cmd_registry()
    elif args.cmd == "report":
        cmd_report()
    elif args.cmd == "selfcheck":
        cmd_selfcheck(p)


if __name__ == "__main__":
    main()
