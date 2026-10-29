#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小规模低波趋势与容量研究（第二阶段候选，独立文件）

问题
----
前期“按流通规模排序”曾解释约 21pp 的组合收益差，但仍未转正。本文件要回答：
那 21pp 是**规模**、**流动性（成交额）**、**市场暴露（beta）**，还是**趋势 alpha**？
并把 ADV 参与率与冲击成本放进主结果，而不是放在脚注里。

纪律
----
* 数据：只读 `01_数据/frozen/frozen.duckdb`（冻结 2026-10-26）。任何收益只到 2020-12-31。
* 价格：成交/整股/涨跌停/费用用 **raw**；指标与事件收益用 **A 空间**
  `A_t = factor_t / raw_close_t`，`adjusted_OHLC = raw_OHLC × A_t`。
  由此 `adjusted_close = factor`，其日收益等于官方 `pct_chg`；**不使用** `hfq = raw×factor`
  （那会把价格收益乘两次），也不使用 Baostock 后复权列。
* 规模：只用 t 日及之前的 `turn` 反推流通股（`fs = volume/(turn/100)`，20 日中位），
  `size_t = raw_close_t × fs_t`；**不使用** 冻结层按“当年中位数”反推的 `float_mktcap`
  （它含当年未来换手信息），只用于对账。
* 执行：t 收盘信号 → t+1 开盘成交；T+1；整股；历史费用；10bp + 0.02√参与率 滑点，
  2% ADV20 上限；停牌/涨跌停/复牌/退市/公司行动由 `统一回测.UnifiedAccount` 处理。
* 本文件只写 `02_代码/候选_小盘低波趋势.py` 与 `03_结果/候选_小盘低波趋势/`。

用法
----
    python3 02_代码/候选_小盘低波趋势.py counts      # 只数信号/覆盖，不算收益
    python3 02_代码/候选_小盘低波趋势.py events      # 事件层规模/流动性/暴露归因
    python3 02_代码/候选_小盘低波趋势.py portfolio   # 组合层网格（dev/val）
    python3 02_代码/候选_小盘低波趋势.py stress      # 压力/邻域/容量
    python3 02_代码/候选_小盘低波趋势.py report      # 汇总报告（读已有 CSV）
    python3 02_代码/候选_小盘低波趋势.py all
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "02_代码"
sys.path.insert(0, str(CODE_DIR))

OUT = ROOT / "03_结果" / "候选_小盘低波趋势"
DB_PATH = ROOT / "01_数据" / "frozen" / "frozen.duckdb"

PANEL_START = "2006-01-01"
PANEL_END = "2020-12-31"
DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")
FULL = ("2008-01-01", "2020-12-31")
WINDOWS = {"dev": DEV, "val": VAL, "full": FULL}

N_LOT = 100
IMPACT_COEF = 0.02           # 父预注册 9.1：0.02×sqrt(参与率)
MAX_PARTICIPATION = 0.02     # 2% ADV20
BASE_SLIP_BPS = 10.0

# 预注册参数（不得因结果修改；敏感性只在 07 表报告）
PREREG = dict(
    vol_window=60, vol_min=50,
    mom_short=20, mom_mid=120, mom_long=250,
    ma_fast=20, ma_slow=120,
    trend_ma_rise=20,
    pullback_dd_lo=-0.15, pullback_dd_hi=-0.03,
    pullback_ma_tol=1.01,
    size_q=1.0 / 3.0, vol_q=1.0 / 3.0,
    slots=(5, 10, 20),
    holds=(20, 40, 60),
    primary_slots=10, primary_hold=40,
    max_new_per_day=2,
    max_persist_days=10,
    capital=1_000_000.0,
)

VARIANTS = {
    "RAND": "U 内中性哈希（零假设：无规模/低波/趋势选择）",
    "SIZE": "U 内按 PIT 流通市值升序（纯规模，最小优先）",
    "LOWVOL": "U 内按 60 日日收益波动升序（纯低波）",
    "TREND": "U ∩ 长期趋势（收盘>MA120、MA120 上行、120 日动量>0），按 250 日动量降序",
    "PULLBACK": "U ∩ 长期趋势 ∩ 回调触发，按 120 日动量降序",
    "COMBO": "U ∩ 规模下 1/3 ∩ 波动下 1/3 ∩ 长期趋势，按规模升序",
    "ADVQ12": "U ∩ ADV 下 2 个五分位（流动性匹配对照），中性哈希",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _json_safe(x):
    if isinstance(x, (date,)):
        return x.isoformat()
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# 上下文：面板 + A 空间特征 + PIT 规模
# ---------------------------------------------------------------------------

@dataclass
class Context:
    panel: object
    events: object
    codes: list
    dates: list
    n_days: int
    n_codes: int
    feats: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _rolling_median_past(df: pd.DataFrame, win: int, minp: int) -> pd.DataFrame:
    return df.rolling(win, min_periods=minp).median()


def build_context(verbose: bool = True) -> Context:
    from 统一数据适配 import FrozenPanel, EventBook

    t0 = time.time()
    eb = EventBook().load(PANEL_START, PANEL_END)
    panel = FrozenPanel(PANEL_START, PANEL_END, events=eb)
    if verbose:
        log(f"面板 {panel.n_days} 日 × {panel.n_codes} 只，{time.time()-t0:.1f}s，"
            f"公司行动 {len(eb.events):,}")

    dates = panel.dates
    codes = panel.codes
    n_days, n_codes = panel.n_days, panel.n_codes

    # ---- 换手率矩阵（PIT 流通股用；面板不带 turn） -----------------------
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        turn = np.full((n_days, n_codes), np.nan, dtype=np.float32)
        d_np = np.array(dates, dtype="datetime64[D]")
        y0, y1 = dates[0].year, dates[-1].year
        for y in range(y0, y1 + 1):
            a = max(PANEL_START, f"{y}-01-01")
            b = min(PANEL_END, f"{y}-12-31")
            chunk = con.execute(
                "SELECT symbol, date, turn FROM daily WHERE board='主板' AND date BETWEEN ? AND ?",
                [a, b]).df()
            if chunk.empty:
                continue
            di = np.searchsorted(d_np, chunk["date"].to_numpy(dtype="datetime64[D]"))
            ci = pd.Categorical(chunk["symbol"], categories=codes).codes.astype(np.int32)
            turn[di, ci] = chunk["turn"].to_numpy(dtype=np.float32)
            del chunk, di, ci
    finally:
        con.close()

    # ---- A 空间：A_t = factor_t / close_t；adjusted = raw × A ------------
    close = panel.mat["close"].astype(np.float64)
    factor = panel.mat["factor"].astype(np.float64)
    A = np.divide(factor, close, out=np.full_like(factor, np.nan), where=(close > 0))
    a_close = (close * A)                     # == factor，数值上等于总收益指数
    a_open = (panel.mat["open"].astype(np.float64) * A)
    a_high = (panel.mat["high"].astype(np.float64) * A)
    a_low = (panel.mat["low"].astype(np.float64) * A)

    idx = pd.Index(dates, name="date")
    cols = pd.Index(codes)
    df_close = pd.DataFrame(a_close, index=idx, columns=cols)
    df_open = pd.DataFrame(a_open, index=idx, columns=cols)
    df_high = pd.DataFrame(a_high, index=idx, columns=cols)
    df_low = pd.DataFrame(a_low, index=idx, columns=cols)
    ret1 = df_close.pct_change(fill_method=None)

    vol60 = ret1.rolling(PREREG["vol_window"], min_periods=PREREG["vol_min"]).std()
    ma20 = df_close.rolling(PREREG["ma_fast"], min_periods=PREREG["ma_fast"]).mean()
    ma120 = df_close.rolling(PREREG["ma_slow"], min_periods=PREREG["ma_slow"]).mean()
    ma120_rise = ma120 > ma120.shift(PREREG["trend_ma_rise"])
    rmax20 = df_close.rolling(20, min_periods=20).max()
    rmax60 = df_close.rolling(60, min_periods=60).max()
    mom20 = df_close / df_close.shift(PREREG["mom_short"]) - 1.0
    mom120 = df_close / df_close.shift(PREREG["mom_mid"]) - 1.0
    mom250 = df_close / df_close.shift(PREREG["mom_long"]) - 1.0
    dd60 = df_close / rmax60 - 1.0
    dd20 = df_close / rmax20 - 1.0

    trend = (df_close > ma120) & ma120_rise & (mom120 > 0)
    pullback = (trend
                & (df_close <= ma20 * PREREG["pullback_ma_tol"])
                & (dd60 >= PREREG["pullback_dd_lo"])
                & (dd60 <= PREREG["pullback_dd_hi"]))

    # ---- PIT 流通规模：fs = volume/(turn/100)，20 日中位（只用 ≤t） ------
    vol = panel.mat["volume"].astype(np.float64)
    fs = np.divide(vol, turn.astype(np.float64) / 100.0,
                   out=np.full_like(vol, np.nan), where=(np.isfinite(turn) & (turn > 0)))
    fs[fs <= 0] = np.nan
    fs_df = pd.DataFrame(fs, index=idx, columns=cols)
    fs_pit = _rolling_median_past(fs_df, 20, 5)
    size_pit = df_close * fs_pit / 1e8            # A 空间价格 × 流通股（亿元）
    # 对账用：raw close × fs（与 A 空间只差一个累计分红系数）
    raw_close = pd.DataFrame(close, index=idx, columns=cols)
    size_raw = raw_close * fs_pit / 1e8
    del fs

    feats = dict(
        a_close=df_close.to_numpy(dtype=np.float32),
        a_open=df_open.to_numpy(dtype=np.float32),
        ret1=ret1.to_numpy(dtype=np.float32),
        vol60=vol60.to_numpy(dtype=np.float32),
        ma20=ma20.to_numpy(dtype=np.float32),
        ma120=ma120.to_numpy(dtype=np.float32),
        mom20=mom20.to_numpy(dtype=np.float32),
        mom120=mom120.to_numpy(dtype=np.float32),
        mom250=mom250.to_numpy(dtype=np.float32),
        dd60=dd60.to_numpy(dtype=np.float32),
        dd20=dd20.to_numpy(dtype=np.float32),
        trend=trend.to_numpy(dtype=bool),
        pullback=pullback.to_numpy(dtype=bool),
        size_pit=size_pit.to_numpy(dtype=np.float32),
        size_raw=size_raw.to_numpy(dtype=np.float32),
        n_vol=vol60.notna().to_numpy(dtype=bool),
    )

    # ---- 中性哈希键（只与代码有关，与收益/流动性无关） ------------------
    hcode = np.zeros(n_codes, dtype=np.uint64)
    for j, c in enumerate(codes):
        hcode[j] = int.from_bytes(hashlib.blake2b(c.encode(), digest_size=4).digest(), "little")
    feats["hcode"] = hcode

    # ---- 市场代理：主板可交易等权日收益（毛，不扣费） -------------------
    listed = panel.listed & ~panel.suspended
    r1 = feats["ret1"]
    rm = np.full(n_days, np.nan)
    for i in range(n_days):
        v = r1[i][listed[i]]
        v = v[np.isfinite(v)]
        if len(v) >= 50:
            rm[i] = float(v.mean())
    feats["rm"] = rm.astype(np.float32)

    # 规模估计的对账：与冻结层 float_mktcap（当年中位反推）比较
    fm = panel.mat["float_mktcap"].astype(np.float64) / 1e8
    mask = np.isfinite(fm) & (fm > 0) & np.isfinite(size_raw.to_numpy())
    ratio = np.where(mask, size_raw.to_numpy() / fm, np.nan)
    meta = {
        "n_days": n_days, "n_codes": n_codes,
        "size_vs_float_mktcap_median_ratio": float(np.nanmedian(ratio)),
        "size_vs_float_mktcap_p10": float(np.nanpercentile(ratio[mask], 10)),
        "size_vs_float_mktcap_p90": float(np.nanpercentile(ratio[mask], 90)),
        "turn_missing_share": float(np.mean(~np.isfinite(turn))),
        "fs_pit_missing_share": float(np.mean(~np.isfinite(fs_pit.to_numpy()))),
        "n_corp_actions": int(len(eb.events)),
        "corp_action_status": eb.stats.get("by_status", {}),
    }
    bench = {}
    for wname, (a, b) in WINDOWS.items():
        ii0, ii1 = panel.window_index(a, b)
        bs = panel.benchmark_ew(ii0, ii1)
        bs.index = pd.to_datetime(bs.index)
        bench[wname] = bs
    meta["bench_cagr"] = {k: float((1 + v).prod() ** (365.25 / max((v.index[-1]-v.index[0]).days, 1)) - 1)
                          for k, v in bench.items()}
    feats["bench"] = bench
    if verbose:
        log(f"特征完成 {time.time()-t0:.1f}s；规模对账中位比 "
            f"{meta['size_vs_float_mktcap_median_ratio']:.4f}；fs 缺失 "
            f"{meta['fs_pit_missing_share']:.3%}")
    return Context(panel=panel, events=eb, codes=codes, dates=dates,
                   n_days=n_days, n_codes=n_codes, feats=feats, meta=meta)


def eligible_mask(ctx: Context, i: int) -> np.ndarray:
    """U：父预注册主宇宙 + 特征可用。"""
    p, f = ctx.panel, ctx.feats
    m = (p.elig[i] & f["n_vol"][i] & np.isfinite(f["size_pit"][i])
         & np.isfinite(f["mom120"][i]) & np.isfinite(f["mom250"][i]))
    return m


def adv_quintile_bounds(adv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = adv[mask]
    v = v[np.isfinite(v)]
    if len(v) < 50:
        return np.full(6, np.nan)
    return np.quantile(v, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


# ---------------------------------------------------------------------------
# 只数信号：不做任何收益计算
# ---------------------------------------------------------------------------

def cmd_counts(ctx: Context) -> dict:
    p, f = ctx.panel, ctx.feats
    i0, _ = p.window_index(*FULL)
    rows = []
    by_year = {}
    for i in range(i0, p.n_days):
        d = p.dates[i]
        u = np.flatnonzero(eligible_mask(ctx, i))
        if not len(u):
            continue
        adv = p.adv20[i]
        size = f["size_pit"][i]
        vol = f["vol60"][i]
        tr = f["trend"][i]
        pb = f["pullback"][i]
        q_s = np.nanquantile(size[u], PREREG["size_q"])
        q_v = np.nanquantile(vol[u], PREREG["vol_q"])
        aq = adv_quintile_bounds(adv, eligible_mask(ctx, i))
        adv_lo = aq[2] if np.isfinite(aq).all() else np.nan
        sets = {
            "RAND": u,
            "SIZE": u,
            "LOWVOL": u,
            "TREND": u[tr[u]],
            "PULLBACK": u[pb[u]],
            "COMBO": u[(size[u] <= q_s) & (vol[u] <= q_v) & tr[u]],
            "ADVQ12": u[np.isfinite(adv[u]) & (adv[u] <= adv_lo)],
        }
        rec = {"date": d.isoformat(), "n_U": len(u),
               "q_size": float(q_s), "q_vol": float(q_v), "adv20_med": float(np.nanmedian(adv[u])),
               "size_med": float(np.nanmedian(size[u]))}
        for k, v in sets.items():
            rec[f"n_{k}"] = int(len(v))
        rows.append(rec)
        y = d.year
        by_year.setdefault(y, []).append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "00a_每日候选数.csv", index=False)
    yr = []
    for y, rs in sorted(by_year.items()):
        r = {"year": y}
        for k in ["n_U"] + [f"n_{v}" for v in VARIANTS]:
            r[k] = round(float(np.mean([x[k] for x in rs])), 2)
        yr.append(r)
    yd = pd.DataFrame(yr)
    yd.to_csv(OUT / "00b_年度候选数.csv", index=False)
    out = {
        "n_days": int(len(df)),
        "mean_n_U": float(df["n_U"].mean()),
        "mean_counts": {v: float(df[f"n_{v}"].mean()) for v in VARIANTS},
        "median_counts": {v: float(df[f"n_{v}"].median()) for v in VARIANTS},
        "zero_day_share": {v: float((df[f"n_{v}"] == 0).mean()) for v in VARIANTS},
        "p10_counts": {v: float(df[f"n_{v}"].quantile(0.1)) for v in VARIANTS},
        "meta": ctx.meta,
    }
    (OUT / "00c_候选数_汇总.json").write_text(
        json.dumps(_json_safe(out), ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(_json_safe(out["mean_counts"]), ensure_ascii=False))
    log(json.dumps(_json_safe(out["zero_day_share"]), ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# 事件层：规模 / 流动性 / 暴露 / 趋势
# ---------------------------------------------------------------------------

HORIZONS = (20, 40, 60)


def month_end_indices(dates, start_i: int, end_i: int) -> list[int]:
    out = []
    for i in range(start_i, end_i + 1):
        if i + 1 > end_i or dates[i].month != dates[i + 1].month:
            out.append(i)
    return out


def beta_matrix_at(ctx: Context, i: int, cols: np.ndarray) -> np.ndarray:
    """过去 250 日 trailing beta（对主板等权毛收益），winsorize [−0.5, 3]。"""
    f = ctx.feats
    lo = max(0, i - PREREG["mom_long"] + 1)
    R = f["ret1"][lo:i + 1][:, cols].astype(np.float64)
    rm = f["rm"][lo:i + 1].astype(np.float64)[:, None]
    ok = np.isfinite(R) & np.isfinite(rm)
    n = ok.sum(axis=0)
    Rz = np.where(ok, R, 0.0)
    Mz = np.where(ok, rm, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mr = Rz.sum(0) / n
        mm = Mz.sum(0) / n
        cov = (Rz * Mz).sum(0) / n - mr * mm
        var = (Mz * Mz).sum(0) / n - mm * mm
        beta = np.where((n >= 200) & (var > 1e-12), cov / var, np.nan)
    return np.clip(beta, -0.5, 3.0)


def _forward_returns(a_open: np.ndarray, i: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """t+1 开盘 → t+1+h 开盘（A 空间）。缺行（退市）用窗口内最后有效价，标记 censored。"""
    n = a_open.shape[1]
    lo = i + 1
    hi = min(i + 1 + h, a_open.shape[0] - 1)
    sub = a_open[lo:hi + 1]
    valid = np.isfinite(sub) & (sub > 0)
    has = valid.any(axis=0)
    rev = np.argmax(valid[::-1], axis=0)
    last = np.where(has, (sub.shape[0] - 1) - rev, -1)
    base = sub[0]
    out = np.full(n, np.nan)
    idx = np.flatnonzero(has & np.isfinite(base) & (base > 0) & (last >= 0))
    out[idx] = sub[last[idx], idx] / base[idx] - 1.0
    censored = has & (last < sub.shape[0] - 1)
    return out, censored


def _clustered(by_date: pd.DataFrame, col: str = "value") -> dict:
    g = by_date.groupby("date")[col].agg(["mean", "count"])
    g = g.dropna()
    n = len(g)
    if n < 3:
        return {"n_dates": int(n), "mean": None, "t": None, "p": None}
    m = float(g["mean"].mean())
    sd = float(g["mean"].std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else np.nan
    t = m / se if se and se > 0 else np.nan
    p = None
    if np.isfinite(t):
        try:
            from scipy import stats as _st
            p = float(2 * (1 - _st.t.cdf(abs(t), df=n - 1)))
        except Exception:
            p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
    return {"n_dates": int(n), "mean": m, "t": float(t) if np.isfinite(t) else None,
            "p": p, "se": float(se) if np.isfinite(se) else None}


def build_event_sample(ctx: Context) -> pd.DataFrame:
    p, f = ctx.panel, ctx.feats
    i0, i1 = p.window_index(*FULL)
    ends = [i for i in month_end_indices(p.dates, i0, i1) if i + 2 < p.n_days]
    adv = p.adv20
    rows = []
    for i in ends:
        m_all = eligible_mask(ctx, i)
        # 次一交易日可执行：不停牌、非复牌、非开盘涨停
        i2 = i + 1
        exec_ok = (p.listed[i2] & ~p.suspended[i2] & ~p.resume[i2]
                   & ~p.mat["is_limit_up_open"][i2])
        m = m_all & exec_ok
        cols = np.flatnonzero(m)
        if len(cols) < 100:
            continue
        size = f["size_pit"][i][cols].astype(np.float64)
        advv = adv[i][cols].astype(np.float64)
        vol = f["vol60"][i][cols].astype(np.float64)
        mom120 = f["mom120"][i][cols].astype(np.float64)
        trend = f["trend"][i][cols]
        pb = f["pullback"][i][cols]
        beta = beta_matrix_at(ctx, i, cols)
        d = p.dates[i]
        rec = {"date": d, "code": np.array(ctx.codes, dtype=object)[cols],
               "size": size, "adv": advv, "vol": vol, "mom120": mom120,
               "trend": trend, "pullback": pb, "beta": beta}
        ok_any = np.ones(len(cols), dtype=bool)
        for h in HORIZONS:
            fr, cen = _forward_returns(f["a_open"], i, h)
            rec[f"fwd{h}"] = fr[cols]
            rec[f"cens{h}"] = cen[cols]
            ok_any &= np.isfinite(fr[cols])
        rec["ok"] = ok_any
        df = pd.DataFrame(rec)
        # 同日 U 内分位（只用于分桶；阈值随截面变化）
        df["size_dec"] = pd.qcut(df["size"].rank(method="first"), 10, labels=False) + 1
        df["size_ter"] = pd.qcut(df["size"].rank(method="first"), 3, labels=False) + 1
        df["adv_quin"] = pd.qcut(df["adv"].rank(method="first"), 5, labels=False) + 1
        df["vol_ter"] = pd.qcut(df["vol"].rank(method="first"), 3, labels=False) + 1
        # ADV 层内规模中位切分（分层估计用：同流动性内比大小）
        med = df.groupby("adv_quin")["size"].transform("median")
        df["size_hi_in_adv"] = df["size"] > med
        med2 = df.groupby("size_ter")["adv"].transform("median")
        df["adv_hi_in_size"] = df["adv"] > med2
        for h in HORIZONS:
            mkt = df[f"fwd{h}"].mean()
            df[f"mkt{h}"] = mkt
            df[f"exc{h}"] = df[f"fwd{h}"] - mkt
            df[f"alpha{h}"] = df[f"fwd{h}"] - df["beta"].fillna(1.0) * mkt
        rows.append(df)
    ev = pd.concat(rows, ignore_index=True)
    ev = ev[ev["ok"]].copy()
    log(f"事件样本 {len(ev):,} 行 / {ev['date'].nunique()} 个月末截面 / "
        f"{ev['code'].nunique()} 只；censored20={int(ev['cens20'].sum())}")
    return ev


def event_attribution(ev: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    ev = ev.copy()
    ev["year"] = pd.to_datetime(ev["date"]).dt.year

    def table(key: str, horizons=HORIZONS) -> pd.DataFrame:
        rows = []
        for k, g in ev.groupby(key):
            r = {key: k, "n": len(g), "n_dates": g["date"].nunique(),
                 "size_med": float(g["size"].median()), "adv_med": float(g["adv"].median()),
                 "vol_med": float(g["vol"].median()), "beta": float(g["beta"].mean()),
                 "trend_share": float(g["trend"].mean())}
            for h in horizons:
                r[f"fwd{h}"] = float(g[f"fwd{h}"].mean())
                st = _clustered(g.rename(columns={f"exc{h}": "value"})[["date", "value"]])
                r[f"exc{h}"] = st["mean"]
                r[f"t{h}"] = st["t"]
                r[f"alpha{h}"] = float(g[f"alpha{h}"].mean())
                ast = _clustered(g.rename(columns={f"alpha{h}": "value"})[["date", "value"]])
                r[f"t_alpha{h}"] = ast["t"]
            rows.append(r)
        return pd.DataFrame(rows)

    out["by_size_decile"] = table("size_dec")
    out["by_adv_quintile"] = table("adv_quin")
    out["by_vol_tercile"] = table("vol_ter")
    out["by_trend"] = table("trend")

    # ---- 价差：按日截面计算，再按日聚类 --------------------------------
    def spread_table(df: pd.DataFrame, mask_lo, mask_hi, name: str,
                     horizons=HORIZONS, group: str | None = None) -> pd.DataFrame:
        rows = []
        groups = [(None, df)] if group is None else list(df.groupby(group))
        for gname, g in groups:
            lo = g[mask_lo(g)]
            hi = g[mask_hi(g)]
            if len(lo) < 30 or len(hi) < 30:
                continue
            r = {"scope": name, group or "all": gname}
            for h in horizons:
                a = lo.groupby("date")[f"fwd{h}"].mean()
                b = hi.groupby("date")[f"fwd{h}"].mean()
                diff = (a - b).dropna()
                st = _clustered(pd.DataFrame({"date": diff.index, "value": diff.to_numpy()}))
                r[f"spread{h}"] = st["mean"]
                r[f"t{h}"] = st["t"]
                r[f"p{h}"] = st["p"]
                r[f"n_dates{h}"] = st["n_dates"]
            rows.append(r)
        return pd.DataFrame(rows)

    small = lambda g: g["size_ter"] == 1
    big = lambda g: g["size_ter"] == 3
    lowvol = lambda g: g["vol_ter"] == 1
    highvol = lambda g: g["vol_ter"] == 3
    tr = lambda g: g["trend"]
    ntr = lambda g: ~g["trend"]

    out["spread_size"] = pd.concat([
        spread_table(ev, small, big, "size_T1_minus_T3"),
        spread_table(ev, lambda g: g["size_dec"] == 1, lambda g: g["size_dec"] == 10,
                     "size_D1_minus_D10"),
    ], ignore_index=True)
    out["spread_size_by_adv"] = spread_table(
        ev, small, big, "size_T1_minus_T3_within_ADV", group="adv_quin")
    # 分层估计：每个 ADV 五分位内按规模中位切分，等权平均（H2 的判据）
    st_rows = []
    for q, g in ev.groupby("adv_quin"):
        lo = g[~g["size_hi_in_adv"]]
        hi = g[g["size_hi_in_adv"]]
        if len(lo) < 100 or len(hi) < 100:
            continue
        r = {"adv_quin": int(q), "n_small": len(lo), "n_big": len(hi),
             "size_med_small": float(lo["size"].median()),
             "size_med_big": float(hi["size"].median())}
        for h in HORIZONS:
            a = lo.groupby("date")[f"fwd{h}"].mean()
            b = hi.groupby("date")[f"fwd{h}"].mean()
            d = (a - b).dropna()
            st = _clustered(pd.DataFrame({"date": d.index, "value": d.to_numpy()}))
            r[f"spread{h}"] = st["mean"]
            r[f"t{h}"] = st["t"]
        st_rows.append(r)
    out["spread_size_stratified"] = pd.DataFrame(st_rows)
    # 等权分层估计（H2 判据）：每日 5 个层内价差的等权平均，再按日聚类
    per = []
    for q, g in ev.groupby("adv_quin"):
        lo = g[~g["size_hi_in_adv"]].groupby("date")["fwd20"].mean()
        hi = g[g["size_hi_in_adv"]].groupby("date")["fwd20"].mean()
        per.append((lo - hi).rename(q))
    if per:
        avg = pd.concat(per, axis=1).mean(axis=1).dropna()
        st = _clustered(pd.DataFrame({"date": avg.index, "value": avg.to_numpy()}))
        out["spread_size_stratified_avg"] = pd.DataFrame([{
            "scope": "size_within_ADV_median_split_equal_weight",
            "spread20": st["mean"], "t20": st["t"], "p20": st["p"],
            "n_dates": st["n_dates"]}])
    out["spread_adv_by_size"] = spread_table(
        ev, lambda g: g["adv_quin"] == 1, lambda g: g["adv_quin"] == 5,
        "ADV_Q1_minus_Q5_within_size", group="size_ter")
    out["spread_vol_by_size"] = spread_table(
        ev, lowvol, highvol, "lowvol_minus_highvol_within_size", group="size_ter")
    out["spread_trend_by_size"] = spread_table(
        ev, tr, ntr, "trend_minus_notrend_within_size", group="size_ter")
    out["spread_trend_by_adv"] = spread_table(
        ev, tr, ntr, "trend_minus_notrend_within_ADV", group="adv_quin")

    # ---- 分年：规模价差、趋势价差 ---------------------------------------
    yr = []
    for y, g in ev.groupby("year"):
        r = {"year": y, "n": len(g)}
        for h in (20, 60):
            lo = g[small(g)].groupby("date")[f"fwd{h}"].mean()
            hi = g[big(g)].groupby("date")[f"fwd{h}"].mean()
            d = (lo - hi).dropna()
            r[f"size_spread{h}"] = float(d.mean()) if len(d) else np.nan
            lt = g[tr(g)].groupby("date")[f"fwd{h}"].mean()
            nt = g[ntr(g)].groupby("date")[f"fwd{h}"].mean()
            d2 = (lt - nt).dropna()
            r[f"trend_spread{h}"] = float(d2.mean()) if len(d2) else np.nan
        yr.append(r)
    out["by_year"] = pd.DataFrame(yr)

    # ---- 规模 × ADV 的二维均值（残余规模效应） ---------------------------
    rows = []
    for s in (1, 2, 3):
        for a in (1, 2, 3, 4, 5):
            g = ev[(ev["size_ter"] == s) & (ev["adv_quin"] == a)]
            if len(g) < 50:
                continue
            rows.append({"size_ter": s, "adv_quin": a, "n": len(g),
                         "exc20": float(g["exc20"].mean()),
                         "exc60": float(g["exc60"].mean()),
                         "beta": float(g["beta"].mean())})
    out["size_x_adv"] = pd.DataFrame(rows)

    # ---- 规模内趋势 × 低波（2×2） ---------------------------------------
    rows = []
    for s in (1, 2, 3):
        for tflag in (True, False):
            for vflag in ("low", "high"):
                g = ev[(ev["size_ter"] == s) & (ev["trend"] == tflag)]
                g = g[lowvol(g)] if vflag == "low" else g[highvol(g)]
                if len(g) < 30:
                    continue
                rows.append({"size_ter": s, "trend": tflag, "vol": vflag, "n": len(g),
                             "exc20": float(g["exc20"].mean()),
                             "exc60": float(g["exc60"].mean()),
                             "fwd20": float(g["fwd20"].mean()),
                             "beta": float(g["beta"].mean()),
                             "size_med": float(g["size"].median())})
    out["size_trend_vol"] = pd.DataFrame(rows)

    # ---- 长-空年化换算（诊断，不是可执行策略） --------------------------
    rows = []
    for h, turns in ((20, 243 / 20), (60, 243 / 60)):
        for name, df in (("size_T1_minus_T3", out["spread_size"].iloc[0:1]),
                         ("size_D1_minus_D10", out["spread_size"].iloc[1:2])):
            v = float(df[f"spread{h}"].iloc[0]) if len(df) else np.nan
            rows.append({"horizon": h, "spread": v, "turns_per_year": turns,
                         "ann_arith": v * turns if np.isfinite(v) else np.nan,
                         "name": name})
    out["long_short_ann"] = pd.DataFrame(rows)
    return out


def cmd_events(ctx: Context) -> dict:
    t0 = time.time()
    ev = build_event_sample(ctx)
    ev_out = ev.copy()
    ev_out["date"] = pd.to_datetime(ev_out["date"])
    ev_out.to_parquet(OUT / "09_事件_样本.parquet", index=False)
    tabs = event_attribution(ev)
    for k, df in tabs.items():
        df.to_csv(OUT / f"10_事件_{k}.csv", index=False, float_format="%.6g")
    # 分年表单独命名
    tabs["by_year"].to_csv(OUT / "15_事件_分年.csv", index=False, float_format="%.6g")
    summ = {k: df.to_dict(orient="records") for k, df in tabs.items()}
    (OUT / "10_事件_归因.json").write_text(
        json.dumps(_json_safe(summ), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"事件层完成 {time.time()-t0:.1f}s；表 {list(tabs)}")
    return summ


# ---------------------------------------------------------------------------
# 组合层：统一账户引擎（T+1 / 涨跌停 / 停牌 / 公司行动 / 参与率 / 冲击）
# ---------------------------------------------------------------------------

COST_MODELS = {
    "base": dict(comm=0.00025, slip=10.0, impact=0.02, cap=0.02),
    "stress2": dict(comm=0.00050, slip=20.0, impact=0.04, cap=0.02),
    "cap05": dict(comm=0.00025, slip=10.0, impact=0.02, cap=0.005),
}


@dataclass
class PCfg:
    variant: str
    slots: int = 10
    hold: int = 40
    capital: float = 1_000_000.0
    window: str = "dev"
    seed: int = 0
    delay: int = 0
    cost: str = "base"
    delist_haircut: float = 0.0
    max_persist: int = 10
    max_new_per_day: int = 2
    size_q: float = PREREG["size_q"]
    vol_q: float = PREREG["vol_q"]
    label: str = ""

    def key(self) -> str:
        return (f"{self.variant}_s{self.slots}_h{self.hold}_c{int(self.capital)}"
                f"_{self.window}_d{self.delay}_{self.cost}_x{self.delist_haircut}"
                f"_p{self.max_persist}_sd{self.seed}"
                f"_q{self.size_q:.3f}_{self.vol_q:.3f}")


def _fee_config(cost: str):
    import 费用与交易约束 as fm
    c = COST_MODELS[cost]
    transfer = (
        fm.TransferRule(date(2005, 1, 1), "par_sh", 0.0010, "SH"),
        fm.TransferRule(date(2015, 8, 1), "amount", 0.00002, "BOTH"),
        fm.TransferRule(date(2022, 4, 29), "amount", 0.00001, "BOTH"),
    )
    return fm.FeeConfig(
        commission_rate=c["comm"], min_commission=5.0,
        stamp_schedule=fm.DEFAULT_STAMP_SCHEDULE, transfer_schedule=transfer,
        slippage_bps=c["slip"], max_participation=c["cap"], dividend_tax_rate=0.0)


def _make_account(ctx: Context, cfg: PCfg):
    """与统一引擎同源的账户；唯一差异是冲击系数/参与率上限可切换（压力用）。"""
    import 统一回测 as U
    from 账户撮合 import Account, AccountConfig, SymbolInfo

    coef = COST_MODELS[cfg.cost]

    class CandAccount(U.UnifiedAccount):
        impact_coef = coef["impact"]

        def _fill(self, order, d, bar, base_price, execution, reason):
            old = self.config.fee
            slip = float(old.slippage_bps)
            cap = float(old.max_participation)
            adv = float(self._adv.get(order.code, 0.0))
            if adv > 0 and bar.volume > 0:
                notional = float(base_price) * max(int(order.remaining), 1)
                part = notional / adv
                slip = min(100.0, slip + self.impact_coef * math.sqrt(max(part, 0.0)) * 1e4)
                cap = min(cap, 0.02 * adv / float(bar.volume))
            if order.code in self._resume_today:
                slip = min(200.0, slip * 2.0)
            self.config.fee = dataclasses.replace(
                old, slippage_bps=slip, max_participation=max(cap, 1e-9))
            pos = self.positions.get(order.code)
            s0 = int(pos.shares) if pos else 0
            cost0 = float(pos.cost_total) if pos else 0.0
            try:
                ok = Account._fill(self, order, d, bar, base_price, execution, reason)
            finally:
                self.config.fee = old
            if not ok:
                return False
            t = self.trades[-1]
            pos = self.positions.get(order.code)
            if order.side == "buy":
                if s0 == 0 and pos is not None and pos.shares > 0:
                    f = self._ca_factor_of(order.code)
                    entry_ca = float(t.price) / f if f > 0 else float(t.price)
                    self.hold_meta[order.code] = {
                        "entry_date": d,
                        "entry_idx": bisect.bisect_left(self.calendar, d),
                        "entry_price": float(t.price), "entry_ca": entry_ca,
                        "entry_shares": int(t.qty), "cost_basis": float(t.amount) + float(t.fees),
                        "entry_fees": float(t.fees), "pending_div": 0.0,
                        "highest_ca": entry_ca, "stop_ca": float("nan"), "stop_order": None,
                        "stop_price_raw": float("nan"), "exit_order": None,
                        "sell_gross": 0.0, "sell_fees": 0.0, "sell_qty": 0,
                        "tax": 0.0, "exit_reason": "",
                    }
            else:
                meta = self.hold_meta.get(order.code)
                if meta is not None:
                    meta["sell_gross"] += float(t.amount)
                    meta["sell_fees"] += float(t.fees)
                    meta["sell_qty"] += int(t.qty)
                    meta["exit_reason"] = t.reason
                if pos is not None and pos.shares == 0 and order.code in self.hold_meta:
                    tax = self._settle_tax(order.code, d)
                    self._finalize_round_trip(order.code, d, tax)
            return True

    info = {code: SymbolInfo(code=code, name="", ipo_date=ctx.panel.ipo_date[j],
                             out_date=ctx.panel.delist_date[j])
            for j, code in enumerate(ctx.codes)}
    acfg = AccountConfig(
        initial_cash=float(cfg.capital), fee=_fee_config(cfg.cost),
        cash_settle_days=0, enforce_t1=True, allow_buy_downsize=True,
        normalize_buy_orders=True, rights_policy="subscribe", dividend_tax_rate=0.0,
        delisting_missing_days=10_000_000, delisting_haircut=float(cfg.delist_haircut),
        liquidate_on_finalize=False, finalize_haircut=0.0, strict=False)
    return CandAccount(acfg, ctx.panel.dates, freeze_date=ctx.panel.dates[-1],
                       symbol_info=info, accounting="shares")


def _hash_keys(ctx: Context, seed: int, i: int) -> np.ndarray:
    """中性哈希：与收益/流动性/价格无关，但每天重新充分混合。

    早期版本用 `h*2_654_435_761 + i*40_503`，日期项相对哈希乘积太小，
    排序在时间上几乎固定，零假设退化成“固定的 10 只篮子”（实测 5 年只有 52 只）。
    这里改用 splitmix64 型混合，使每天的顺序近似独立。
    """
    M = np.uint64
    h = ctx.feats["hcode"].astype(np.uint64)
    x = (h * M(0x9E3779B97F4A7C15) + M(i) * M(0xBF58476D1CE4E5B9)
         + M(seed + 1) * M(0x94D049BB133111EB))
    x ^= (x >> M(30)); x *= M(0xBF58476D1CE4E5B9)
    x ^= (x >> M(27)); x *= M(0x94D049BB133111EB)
    x ^= (x >> M(31))
    return (x % M(2 ** 32)).astype(np.float64)


def rank_candidates(ctx: Context, i: int, variant: str, seed: int = 0,
                    size_q: float | None = None,
                    vol_q: float | None = None) -> np.ndarray:
    p, f = ctx.panel, ctx.feats
    m = eligible_mask(ctx, i)
    if i + 1 < p.n_days:
        # resume[i+1] 只依赖 t 日状态（susp[i] | ~listed[i]），是 t 日可知的；
        # 不用 listed[i+1]（那是 t+1 的信息），成交可行性由引擎在 t+1 判定。
        m = m & ~p.resume[i + 1]
    else:
        m = np.zeros_like(m)
    if not m.any():
        return np.empty(0, dtype=int)
    js = np.flatnonzero(m)
    adv = np.where(np.isfinite(p.adv20[i][js]), p.adv20[i][js], -np.inf)
    size = f["size_pit"][i][js].astype(np.float64)
    vol = f["vol60"][i][js].astype(np.float64)
    tr = f["trend"][i][js]
    pb = f["pullback"][i][js]
    hs = _hash_keys(ctx, seed, i)[js]

    if variant in ("RAND",):
        key, sel = hs, np.ones(len(js), bool)
    elif variant == "ADVQ12":
        q = np.nanquantile(p.adv20[i][m], 0.4)
        key, sel = hs, np.isfinite(p.adv20[i][js]) & (p.adv20[i][js] <= q)
    elif variant == "SIZE":
        key, sel = size, np.ones(len(js), bool)
    elif variant == "LOWVOL":
        key, sel = vol, np.ones(len(js), bool)
    elif variant == "TREND":
        key, sel = -f["mom250"][i][js].astype(np.float64), tr
    elif variant == "PULLBACK":
        key, sel = -f["mom120"][i][js].astype(np.float64), pb
    elif variant == "COMBO":
        q_s = np.nanquantile(f["size_pit"][i][m],
                             PREREG["size_q"] if size_q is None else size_q)
        q_v = np.nanquantile(f["vol60"][i][m],
                             PREREG["vol_q"] if vol_q is None else vol_q)
        key = size
        sel = tr & (size <= q_s) & (vol <= q_v)
    else:
        raise KeyError(variant)
    idx = np.flatnonzero(sel & np.isfinite(key))
    if not len(idx):
        return idx
    order = np.lexsort((js[idx], -adv[idx], key[idx]))
    return js[idx][order]


def run_portfolio(ctx: Context, cfg: PCfg, collect: bool = False) -> dict:
    from 账户撮合 import Order
    p = ctx.panel
    i0, i1 = p.window_index(*WINDOWS[cfg.window])
    acct = _make_account(ctx, cfg)
    acct._code_pos = p.code_idx
    start, end = WINDOWS[cfg.window]

    eq_rows, fill_rows = [], []
    trade_ptr = 0
    ca_ptr = 0
    tax_ptr = 0
    prev_cash = float(acct.cash_total)
    cash_resid_max = 0.0
    n_bad_cash = 0
    n_orders_submitted = 0

    for i in range(i0 - 1, i1 + 1):
        d = p.dates[i]
        if d >= date(2021, 1, 1):
            raise RuntimeError("试图运行 2021+")
        held = list(acct.positions.keys())
        active = [o.code for o in acct._orders if o.active]
        ev_today = ctx.events.on(d)
        needed = set(held) | set(active) | {e.symbol for e in ev_today}
        acct._ca_row = p.ca_factor[i]
        acct._adv = {c: float(p.adv20[i - 1, p.code_idx[c]])
                     if i > 0 and c in p.code_idx and np.isfinite(p.adv20[i - 1, p.code_idx[c]])
                     else 0.0 for c in set(active) | set(held)}
        acct._resume_today = {p.codes[j] for j in np.flatnonzero(p.resume[i])}
        rec = acct.on_day(d, bars=p.bars_on(i, needed), events=ev_today)

        # 卖出顺延上限（与统一引擎一致：第 N 个交易日收盘强制退出）
        forced_now = False
        for o in list(acct._orders):
            if not (o.active and o.side == "sell" and o.filled_qty == 0):
                continue
            try:
                age = i - p.day_idx[o.execute_from]
            except KeyError:
                continue
            if age >= cfg.max_persist:
                pos = acct.positions.get(o.code)
                j = p.code_idx.get(o.code)
                if pos is None or j is None:
                    continue
                px = float(p.mat["close"][i, j])
                if not np.isfinite(px) or px <= 0:
                    px = float(pos.last_price) if pos.last_price else 0.0
                acct._force_sell(o.code, d, px, "sell_persistence")
                forced_now = True
        if forced_now:
            rec = acct._mark_and_record(d)

        # 逐笔参与率/滑点（新增成交）
        new_fills = acct.trades[trade_ptr:]
        for t in new_fills:
            j = p.code_idx.get(t.code)
            adv_prev = float(p.adv20[i - 1, j]) if i > 0 and j is not None else 0.0
            part = (t.amount / adv_prev) if adv_prev > 0 else np.nan
            o = float(p.mat["open"][i, j]) if j is not None else np.nan
            slip_cost = ((t.price - o) if t.side == "buy" else (o - t.price)) * t.qty \
                if np.isfinite(o) else np.nan
            fill_rows.append({
                "date": d.isoformat(), "code": t.code, "side": t.side, "qty": t.qty,
                "price": t.price, "amount": t.amount, "fees": t.fees,
                "commission": t.commission, "stamp": t.stamp_duty, "transfer": t.transfer_fee,
                "participation": part,
                "impact_bps": (min(100.0, COST_MODELS[cfg.cost]["impact"] * math.sqrt(part) * 1e4)
                               if np.isfinite(part) and part > 0 else 0.0),
                "slip_cost": slip_cost, "reason": t.reason, "tag": t.tag,
            })
        # 现金流恒等
        buys = sum(x.amount + x.fees for x in new_fills if x.side == "buy")
        sells = sum(x.amount - x.fees for x in new_fills if x.side == "sell")
        ca_cash = sum(float(x["credit"]) for x in acct.ca_rows[ca_ptr:])
        ca_ptr = len(acct.ca_rows)
        tax_cash = sum(float(x["amount"]) for x in acct.tax_ledger[tax_ptr:])
        tax_ptr = len(acct.tax_ledger)
        flows = sells - buys + ca_cash + tax_cash
        delta = float(rec["cash_total"]) - prev_cash
        cash_resid_max = max(cash_resid_max, abs(delta - flows))
        if abs(delta - flows) > 0.02:
            n_bad_cash += 1
        prev_cash = float(rec["cash_total"])
        trade_ptr = len(acct.trades)

        if i >= i0:
            nav = float(rec["nav"])
            eq_rows.append({"date": d.isoformat(), "nav": nav,
                            "cash": float(rec["cash_total"]),
                            "market_value": float(rec["market_value"]),
                            "n_positions": int(rec["n_positions"]),
                            "exposure": (float(rec["market_value"]) / nav) if nav > 0 else 0.0,
                            "n_fills": len(new_fills)})
        if i >= i1:
            break

        # ---- t 收盘：先排卖出，再排名买入 ----
        sig_i = i - cfg.delay
        # 到期卖出（entry_idx + hold - 1 的收盘下单，次日开盘卖）
        for code, meta in list(acct.hold_meta.items()):
            pos = acct.positions.get(code)
            if not pos or pos.shares <= 0:
                continue
            if i - int(meta["entry_idx"]) < cfg.hold - 1:
                continue
            acts = [o for o in acct._orders if o.active and o.side == "sell" and o.code == code]
            if acts:
                continue
            # GTC：跌停/停牌时保持挂单，由 max_persist 触发第 N 日收盘强制退出
            acct.submit(Order(code=code, side="sell", qty=int(pos.shares),
                              kind="open", tif="gtc", tag="time_exit"), d)
            n_orders_submitted += 1

        # 买入
        open_buys = {o.code for o in acct._orders if o.active and o.side == "buy"}
        n_hold = sum(1 for x in acct.positions.values() if x.shares > 0)
        free = cfg.slots - n_hold - len(open_buys)
        if free > 0 and sig_i >= 0:
            ranked = rank_candidates(ctx, sig_i, cfg.variant, cfg.seed,
                                     cfg.size_q, cfg.vol_q)
            nav = float(rec["nav"])
            cash_est = float(acct.cash_available)
            n_new = 0
            sell_busy = {o.code for o in acct._orders if o.active and o.side == "sell"}
            for j in ranked:
                if n_new >= min(free, cfg.max_new_per_day):
                    break
                code = p.codes[j]
                if code in acct.positions or code in open_buys or code in sell_busy:
                    continue
                px = float(p.mat["close"][i, j])
                if not np.isfinite(px) or px <= 0:
                    continue
                budget = nav / max(cfg.slots, 1)
                sh = int(budget / (px * 1.01) / N_LOT) * N_LOT
                sh = min(sh, int(max(cash_est, 0.0) * 0.98 / (px * 1.01) / N_LOT) * N_LOT)
                if sh < N_LOT:
                    continue
                acct.submit(Order(code=code, side="buy", qty=sh, kind="open",
                                  tif="day", tag=cfg.variant), d)
                n_orders_submitted += 1
                cash_est -= sh * px * 1.01
                n_new += 1

    acct.finalize(p.dates[i1])

    eq = pd.DataFrame(eq_rows)
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date").sort_index()
    fills = pd.DataFrame(fill_rows)
    trips = pd.DataFrame(acct.round_trips)
    if len(trips):
        trips["entry_date"] = pd.to_datetime(trips["entry_date"])
        trips["exit_date"] = pd.to_datetime(trips["exit_date"])
        # 毛收益：A 空间开盘到开盘（同一持有期）
        a_open = ctx.feats["a_open"]
        entry_i = np.array([p.day_idx.get(x.date()) for x in trips["entry_date"]])
        exit_i = np.array([p.day_idx.get(x.date()) for x in trips["exit_date"]])
        jj = np.array([p.code_idx.get(c, -1) for c in trips["code"]])
        okj = (jj >= 0) & (entry_i >= 0) & (exit_i >= 0)
        gross = np.full(len(trips), np.nan)
        for k in np.flatnonzero(okj):
            a0 = a_open[entry_i[k], jj[k]]
            a1 = a_open[exit_i[k], jj[k]]
            if np.isfinite(a0) and np.isfinite(a1) and a0 > 0:
                gross[k] = a1 / a0 - 1.0
        trips["gross_ret"] = gross
        trips["cost_ret"] = trips["gross_ret"] - trips["ret"]
        trips["window"] = cfg.window
        trips["variant"] = cfg.variant
        trips["slots"] = cfg.slots
        trips["hold_target"] = cfg.hold

    met = perf_metrics(eq, trips, cfg, ctx, fills)
    met.update({"window": cfg.window, "variant": cfg.variant, "slots": cfg.slots,
                "hold": cfg.hold, "capital": cfg.capital, "seed": cfg.seed,
                "delay": cfg.delay, "cost": cfg.cost,
                "delist_haircut": cfg.delist_haircut,
                "max_persist": cfg.max_persist,
                "cash_resid_max": cash_resid_max, "bad_cash_days": n_bad_cash,
                "n_orders_submitted": n_orders_submitted,
                "force_counts": dict(acct._force_counts),
                "n_tax_events": len(acct.tax_ledger),
                "n_ca_rows": len(acct.ca_rows)})
    out = {"metrics": met, "equity": eq, "fills": fills, "trips": trips,
           "cfg": asdict(cfg)}
    if collect:
        out["orders"] = pd.DataFrame([o.as_dict() for o in acct.orders.values()])
    return out


def perf_metrics(eq: pd.DataFrame, trips: pd.DataFrame, cfg: PCfg,
                 ctx: Context, fills: pd.DataFrame) -> dict:
    if len(eq) < 2:
        return {"cagr": np.nan}
    nav = eq["nav"]
    days = (eq.index[-1] - eq.index[0]).days
    yrs = max(days / 365.25, 1e-9)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1 if nav.iloc[0] > 0 else np.nan
    r = nav.pct_change().dropna()
    vol = float(r.std() * math.sqrt(243))
    sharpe = float(r.mean() / r.std() * math.sqrt(243)) if r.std() > 0 else np.nan
    peak = nav.cummax()
    dd = nav / peak - 1.0
    mdd = float(dd.min())
    out = {
        "cagr": float(cagr), "mdd": mdd, "vol": vol, "sharpe": sharpe,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        "total": float(nav.iloc[-1] / nav.iloc[0] - 1), "years": float(yrs),
        "nav_end": float(nav.iloc[-1]),
        "avg_exposure": float(eq["exposure"].mean()),
        "avg_pos": float(eq["n_positions"].mean()),
        "cash_ratio": float((eq["cash"] / eq["nav"]).mean()),
        "n_days": int(len(eq)),
    }
    # 换手
    if len(fills):
        traded = float(fills["amount"].sum())
        out["turnover_ann"] = traded / float(eq["nav"].mean()) / yrs
        out["fees_ann_pct"] = float(fills["fees"].sum()) / float(eq["nav"].mean()) / yrs
        out["slip_cost_ann_pct"] = float(np.nansum(fills["slip_cost"])) / float(eq["nav"].mean()) / yrs
        part = fills["participation"].dropna()
        out["n_fills"] = int(len(fills))
        out["part_mean"] = float(part.mean()) if len(part) else np.nan
        out["part_median"] = float(part.median()) if len(part) else np.nan
        out["part_p95"] = float(part.quantile(0.95)) if len(part) else np.nan
        out["part_max"] = float(part.max()) if len(part) else np.nan
        out["share_part_gt_cap"] = float((part > COST_MODELS[cfg.cost]["cap"]).mean()) if len(part) else np.nan
        out["impact_bps_mean"] = float(fills["impact_bps"].mean()) if len(fills) else np.nan
        out["impact_cost_ann_pct"] = (float((fills["amount"] * fills["impact_bps"] / 1e4).sum())
                                      / float(eq["nav"].mean()) / yrs)
    if len(trips):
        out.update({
            "n_closed": int(len(trips)),
            "n_symbols": int(trips["code"].nunique()),
            "win_rate": float((trips["ret"] > 0).mean()),
            "profit_factor": (float(trips.loc[trips["ret"] > 0, "pnl"].sum() /
                                    abs(trips.loc[trips["ret"] < 0, "pnl"].sum()))
                              if (trips["ret"] < 0).any() else np.nan),
            "avg_ret": float(trips["ret"].mean()),
            "median_ret": float(trips["ret"].median()),
            "avg_gross": float(trips["gross_ret"].mean()),
            "avg_cost": float(trips["cost_ret"].mean()),
            "avg_hold": float(trips["hold_days"].mean()),
            "tail_share_top5": (float(trips["pnl"].nlargest(5).sum() / trips["pnl"].sum())
                                if trips["pnl"].sum() > 0 else np.nan),
            "n_forced": int(trips["reason"].astype(str).str.contains("persistence|delist").sum()),
        })
        # 去前 5 赢家（近似：把该笔已实现盈利从平仓日起从净值中扣除）
        top5 = trips.nlargest(5, "pnl")
        nav2 = nav.copy()
        for r0 in top5.itertuples(index=False):
            if r0.pnl > 0:
                nav2.loc[nav2.index >= r0.exit_date] -= float(r0.pnl)
        if nav2.iloc[0] > 0 and nav2.min() > 0:
            out["cagr_ex_top5"] = float((nav2.iloc[-1] / nav2.iloc[0]) ** (1 / yrs) - 1)
            peak2 = nav2.cummax()
            out["mdd_ex_top5"] = float((nav2 / peak2 - 1).min())
        # 年度：按日收益链式复合（含跨年首日），不用“年内首日→末日”近似
        rr = nav.pct_change().fillna(0.0)
        yearly = (1.0 + rr).groupby(rr.index.year).prod() - 1.0
        out["worst_year"] = float(yearly.min())
        out["best_year"] = float(yearly.max())
    # 相对 B1 的暴露
    b1 = ctx.feats["bench"][cfg.window]
    common = r.index.intersection(b1.index)
    if len(common) > 100:
        rr = r.reindex(common).to_numpy()
        bb = b1.reindex(common).to_numpy()
        var = np.var(bb)
        out["beta_to_B1"] = float(np.cov(rr, bb)[0, 1] / var) if var > 0 else np.nan
        out["corr_B1"] = float(np.corrcoef(rr, bb)[0, 1])
        out["excess_cagr_vs_B1"] = (float(cagr) -
                                    float((1 + b1).prod() ** (1 / yrs) - 1))
    return out


def cmd_portfolio(ctx: Context) -> pd.DataFrame:
    t0 = time.time()
    rows, eqs, trips_all, fills_all = [], [], [], []
    n_run = 0
    for variant in VARIANTS:
        for slots in PREREG["slots"]:
            for hold in PREREG["holds"]:
                for window in ("dev", "val"):
                    cfg = PCfg(variant=variant, slots=slots, hold=hold, window=window)
                    res = run_portfolio(ctx, cfg)
                    m = res["metrics"]
                    m["cid"] = cfg.key()
                    m["is_primary"] = (variant == "COMBO" and slots == 10 and hold == 40)
                    rows.append(m)
                    n_run += 1
                    if slots == 10 and hold == 40:
                        eq = res["equity"].copy()
                        eq["variant"] = variant
                        eq["window"] = window
                        eqs.append(eq.reset_index())
                        if len(res["trips"]):
                            trips_all.append(res["trips"].copy())
                        if len(res["fills"]):
                            fills_all.append(res["fills"].copy())
                log(f"  {variant} slots={slots} hold={hold} 完成，累计 {n_run} 次运行")
    grid = pd.DataFrame(rows)
    if "force_counts" in grid:
        grid["force_counts"] = grid["force_counts"].apply(
            lambda x: json.dumps(_json_safe(x), ensure_ascii=False))
    grid.to_csv(OUT / "10_组合_网格.csv", index=False, float_format="%.6g")
    if eqs:
        pd.concat(eqs, ignore_index=True).to_csv(OUT / "13_组合_净值_主单元.csv", index=False)
    if trips_all:
        pd.concat(trips_all, ignore_index=True).to_csv(
            OUT / "12_组合_交易_主单元.csv", index=False, float_format="%.6g")
    if fills_all:
        pd.concat(fills_all, ignore_index=True).to_csv(
            OUT / "12b_组合_成交_主单元.csv", index=False, float_format="%.6g")
    # 分年
    yr = []
    for e in eqs:
        nav = e.set_index("date")["nav"]
        rr = nav.pct_change().fillna(0.0)
        g = (1.0 + rr).groupby(rr.index.year).prod() - 1.0
        for y, v in g.items():
            yr.append({"variant": e["variant"].iloc[0], "window": e["window"].iloc[0],
                       "year": int(y), "ret": float(v)})
    pd.DataFrame(yr).to_csv(OUT / "11_组合_分年.csv", index=False, float_format="%.6g")
    log(f"组合网格完成 {time.time()-t0:.1f}s，{n_run} 次运行")
    return grid


# ---------------------------------------------------------------------------
# 压力 / 邻域 / 容量 / 零假设分布
# ---------------------------------------------------------------------------

def _suspension_profile(ctx: Context, trips: pd.DataFrame) -> dict:
    """持仓期内停牌/复牌/退市暴露：描述性压力，不是可执行交易。"""
    if not len(trips):
        return {}
    p = ctx.panel
    susp_days, censored, delisted = [], 0, 0
    rets_with, rets_without = [], []
    for r in trips.itertuples(index=False):
        j = p.code_idx.get(r.code)
        e = p.day_idx.get(r.entry_date.date())
        x = p.day_idx.get(r.exit_date.date())
        if j is None or e is None or x is None:
            continue
        n = int(p.suspended[e:x + 1, j].sum()) if x > e else 0
        susp_days.append(n)
        if n > 0:
            rets_with.append(float(r.ret))
        else:
            rets_without.append(float(r.ret))
        if "delist" in str(r.reason):
            delisted += 1
    return {
        "n_trips": int(len(trips)),
        "share_with_suspension": float(np.mean([d > 0 for d in susp_days])) if susp_days else None,
        "median_susp_days": float(np.median(susp_days)) if susp_days else None,
        "p95_susp_days": float(np.percentile(susp_days, 95)) if susp_days else None,
        "max_susp_days": int(np.max(susp_days)) if susp_days else 0,
        "ret_with_susp": float(np.mean(rets_with)) if rets_with else None,
        "ret_without_susp": float(np.mean(rets_without)) if rets_without else None,
        "n_delisted_trips": int(delisted),
    }


def cmd_stress(ctx: Context) -> pd.DataFrame:
    t0 = time.time()
    rows = []
    profiles = {}

    def one(cfg: PCfg, group: str, tag: str) -> dict:
        res = run_portfolio(ctx, cfg, collect=False)
        m = res["metrics"]
        m.update({"group": group, "tag": tag, "cid": cfg.key()})
        return res, m

    # 1) 压力：主单元 10/40，full
    settings = [
        ("S0_base", dict(cost="base", delist_haircut=0.0, delay=0, max_persist=10)),
        ("S1_delist30", dict(cost="base", delist_haircut=0.3, delay=0, max_persist=10)),
        ("S2_cost2x", dict(cost="stress2", delist_haircut=0.0, delay=0, max_persist=10)),
        ("S3_delay1", dict(cost="base", delist_haircut=0.0, delay=1, max_persist=10)),
        ("S4_persist3", dict(cost="base", delist_haircut=0.0, delay=0, max_persist=3)),
        ("S5_cap05", dict(cost="cap05", delist_haircut=0.0, delay=0, max_persist=10)),
    ]
    for v in VARIANTS:
        for tag, kw in settings:
            cfg = PCfg(variant=v, slots=10, hold=40, window="full", **kw)
            res, m = one(cfg, "stress", tag)
            rows.append(m)
            if tag == "S0_base":
                profiles[v] = _suspension_profile(ctx, res["trips"])
        log(f"  压力 {v} 完成")

    # 2) 容量：COMBO / SIZE × 10万/100万/1000万/1亿
    for v in ("COMBO", "SIZE", "RAND"):
        for cap in (1e5, 1e6, 1e7, 1e8):
            cfg = PCfg(variant=v, slots=10, hold=40, window="full", capital=cap)
            res, m = one(cfg, "capacity", f"cap{int(cap/1e4)}wan")
            rows.append(m)
        log(f"  容量 {v} 完成")

    # 3) 邻域：COMBO 的 size_q × vol_q（报告稳健性，不用于选择）
    for sq in (1 / 3, 0.5):
        for vq in (1 / 3, 0.5):
            for window in ("dev", "val"):
                cfg = PCfg(variant="COMBO", slots=10, hold=40, window=window,
                           size_q=sq, vol_q=vq)
                res, m = one(cfg, "neighborhood", f"sq{sq:.2f}_vq{vq:.2f}")
                rows.append(m)
    log("  邻域完成")

    # 4) 零假设分布：RAND 种子 0-4（主单元，full/dev/val）
    for window in ("full", "dev", "val"):
        for seed in ((0, 1, 2, 3, 4) if window == "full" else (1, 2, 3)):
            cfg = PCfg(variant="RAND", slots=10, hold=40, window=window, seed=seed)
            res, m = one(cfg, "rand_seed", f"seed{seed}")
            rows.append(m)
    log("  RAND 种子完成")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "14_压力_邻域_容量.csv", index=False, float_format="%.6g")
    (OUT / "14b_停牌退市暴露.json").write_text(
        json.dumps(_json_safe(profiles), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"压力/邻域/容量完成 {time.time()-t0:.1f}s，{len(out)} 次运行")
    return out


# ---------------------------------------------------------------------------
# 自检 + 台账
# ---------------------------------------------------------------------------

def cmd_selfcheck(ctx: Context) -> dict:
    p, f = ctx.panel, ctx.feats
    grid = pd.read_csv(OUT / "10_组合_网格.csv")
    fills_path = OUT / "12b_组合_成交_主单元.csv"
    fills = pd.read_csv(fills_path) if fills_path.exists() else pd.DataFrame()
    trips = pd.read_csv(OUT / "12_组合_交易_主单元.csv") \
        if (OUT / "12_组合_交易_主单元.csv").exists() else pd.DataFrame()
    out = {}
    out["cash_identity_max"] = float(grid["cash_resid_max"].max())
    out["cash_identity_bad_days"] = int(grid["bad_cash_days"].sum())
    out["max_date_in_grid"] = str(pd.to_datetime(grid["window"].map(
        {"dev": "2015-12-31", "val": "2020-12-31", "full": "2020-12-31"})).max().date())
    # A 空间：adjusted_close 日收益 vs 官方 pct_chg
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        smp = con.execute("""
            SELECT symbol, date, pct_chg FROM daily
            WHERE board='主板' AND date BETWEEN '2010-01-01' AND '2020-12-31'
            USING SAMPLE 200000 ROWS""").df()
    finally:
        con.close()
    dev = []
    for r in smp.itertuples(index=False):
        j = p.code_idx.get(r.symbol)
        i = p.day_idx.get(pd.Timestamp(r.date).date())
        if j is None or i is None or i == 0:
            continue
        a0 = f["a_close"][i - 1, j]
        a1 = f["a_close"][i, j]
        if (np.isfinite(a0) and a0 > 0 and np.isfinite(a1)
                and np.isfinite(r.pct_chg)):
            dev.append(abs((a1 / a0 - 1) - float(r.pct_chg) / 100))
    out["a_space_vs_pctchg_n"] = len(dev)
    out["a_space_vs_pctchg_max_abs"] = float(np.max(dev)) if dev else None
    out["a_space_vs_pctchg_p999"] = float(np.percentile(dev, 99.9)) if dev else None
    # 成交可执行性
    if len(fills):
        bad_buy, bad_sell, susp, forced = 0, 0, 0, 0
        for r in fills.itertuples(index=False):
            if "persistence" in str(r.reason) or "delist" in str(r.reason):
                forced += 1
                continue
            j = p.code_idx.get(r.code)
            i = p.day_idx.get(pd.Timestamp(r.date).date())
            if j is None or i is None:
                continue
            if p.mat["is_limit_up_open"][i, j] and r.side == "buy":
                bad_buy += 1
            if p.mat["is_limit_down_open"][i, j] and r.side == "sell":
                bad_sell += 1
            if p.suspended[i, j]:
                susp += 1
        out.update({"n_fills_checked": int(len(fills)),
                    "n_forced_excluded": int(forced),
                    "buy_at_limit_up_open": int(bad_buy),
                    "sell_at_limit_down_open": int(bad_sell),
                    "fills_on_suspension": int(susp)})
    if len(trips):
        out["min_hold_days"] = int(trips["hold_days"].min())
        out["t1_violations"] = int((trips["hold_days"] < 1).sum())
        out["n_trips_checked"] = int(len(trips))
    # 复现性：同一配置跑两次
    r1 = run_portfolio(ctx, PCfg(variant="COMBO", slots=10, hold=40, window="dev"))
    r2 = run_portfolio(ctx, PCfg(variant="COMBO", slots=10, hold=40, window="dev"))
    out["repro_max_nav_diff"] = float(np.max(np.abs(r1["equity"]["nav"] - r2["equity"]["nav"])))
    out["key_metrics_first"] = {k: r1["metrics"].get(k) for k in
                                ("cagr", "mdd", "n_closed", "turnover_ann", "part_mean")}
    out["all_pass"] = bool(
        out["cash_identity_max"] <= 1.0
        and out.get("buy_at_limit_up_open", 0) == 0
        and out.get("sell_at_limit_down_open", 0) == 0
        and out.get("fills_on_suspension", 0) == 0
        and out.get("min_hold_days", 99) >= 1)
    (OUT / "16_自检.json").write_text(
        json.dumps(_json_safe(out), ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(_json_safe(out), ensure_ascii=False)[:600])
    return out


def write_ledger() -> pd.DataFrame:
    rows = []
    grid = pd.read_csv(OUT / "10_组合_网格.csv")
    for r in grid.itertuples(index=False):
        rows.append({"group": "grid", "cid": r.cid, "variant": r.variant,
                     "slots": r.slots, "hold": r.hold, "window": r.window,
                     "capital": r.capital, "cost": r.cost, "delay": r.delay,
                     "seed": r.seed, "n_configs": 1,
                     "note": "7 变体 × 3 仓 × 3 持有 × 2 窗口"})
    st_path = OUT / "14_压力_邻域_容量.csv"
    if st_path.exists():
        st = pd.read_csv(st_path)
        for r in st.itertuples(index=False):
            rows.append({"group": r.group, "cid": r.cid, "variant": r.variant,
                         "slots": r.slots, "hold": r.hold, "window": r.window,
                         "capital": r.capital, "cost": r.cost, "delay": r.delay,
                         "seed": r.seed, "n_configs": 1,
                         "note": getattr(r, "tag", "")})
    # 去重：同一 (variant, slots, hold, window, capital, delay, cost, seed, 分位) 只算一次
    led = pd.DataFrame(rows).drop_duplicates("cid").reset_index(drop=True)
    led["trials_cumulative"] = np.arange(1, len(led) + 1)
    led.to_csv(OUT / "试验台账.csv", index=False)
    log(f"台账 {len(led)} 个配置")
    return led


def _stratified_index(ctx: Context) -> list[dict]:
    """指数级分层收益（等权、月度再平衡、毛）。分位用全窗口定义，仅作描述，不是可交易规则。"""
    p = ctx.panel
    i0, i1 = p.window_index(*FULL)
    hr = p.hfq_ret.astype(np.float64)
    elig = p.elig
    listed = p.listed & ~p.suspended
    adv = p.adv20
    sz = ctx.feats["size_pit"]

    def ew(mask: np.ndarray) -> float:
        nav, cols = 1.0, None
        for i in range(i0, i1 + 1):
            if i == i0 or p.dates[i].month != p.dates[i - 1].month:
                cols = np.flatnonzero(mask[i - 1])
            if cols is None or not len(cols):
                continue
            r = hr[i][cols]
            r = r[np.isfinite(r)]
            if len(r):
                nav *= 1 + r.mean()
        yrs = (p.dates[i1] - p.dates[i0]).days / 365.25
        return float(nav ** (1 / yrs) - 1)

    rows = []
    for name, base in (("all_listed", listed), ("elig", elig)):
        v = adv[i0:i1][base[i0:i1]]
        qs = np.nanquantile(v, [0, .2, .4, .6, .8, 1.0])
        for k in range(5):
            m2 = base & (adv >= qs[k]) & (adv <= qs[k + 1] + 1e-6)
            rows.append({"scope": name, "bucket": f"ADV_Q{k+1}", "cagr_gross_ew_monthly": ew(m2),
                         "adv_cut_yi": float(qs[k] / 1e8), "n_mean": int(m2[i0:i1].sum(1).mean())})
    v = sz[i0:i1][elig[i0:i1]]
    q3 = np.nanquantile(v, [1 / 3, 2 / 3])
    for k, m2 in enumerate([elig & (sz <= q3[0]),
                            elig & (sz > q3[0]) & (sz <= q3[1]),
                            elig & (sz > q3[1])]):
        rows.append({"scope": "elig", "bucket": f"SIZE_T{k+1}", "cagr_gross_ew_monthly": ew(m2),
                     "adv_cut_yi": None, "n_mean": int(m2[i0:i1].sum(1).mean())})
    rows.append({"scope": "elig", "bucket": "ALL", "cagr_gross_ew_monthly": ew(elig),
                 "adv_cut_yi": None, "n_mean": int(elig[i0:i1].sum(1).mean())})
    pd.DataFrame(rows).to_csv(OUT / "00d_分层指数.csv", index=False, float_format="%.6g")
    return rows


def _dsr(sr_trials_ann: np.ndarray, sr_hat_daily: float, ret: pd.Series,
         n_trials: int) -> dict:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio（诊断，不是认证）。"""
    try:
        from scipy import stats as st
    except Exception:
        return {"error": "scipy 不可用"}
    gamma = 0.5772156649
    sr = np.asarray(sr_trials_ann, dtype=float) / math.sqrt(243.0)
    sr = sr[np.isfinite(sr)]
    if len(sr) < 2:
        return {"error": "试验夏普不足"}
    var = float(np.var(sr, ddof=1))
    N = int(max(n_trials, 2))
    z1 = st.norm.ppf(1 - 1 / N)
    z2 = st.norm.ppf(1 - 1 / (N * math.e))
    sr0 = math.sqrt(var) * ((1 - gamma) * z1 + gamma * z2)
    r = ret.dropna()
    T = len(r)
    g3 = float(st.skew(r))
    g4 = float(st.kurtosis(r, fisher=False))
    den = math.sqrt(max(1e-12, 1 - g3 * sr_hat_daily + (g4 - 1) / 4 * sr_hat_daily ** 2))
    dsr = float(st.norm.cdf((sr_hat_daily - sr0) * math.sqrt(max(T - 1, 1)) / den))
    return {"sr_hat_daily": sr_hat_daily, "sr0_daily": sr0, "N": N,
            "var_sr_trials_daily": var, "T": T, "skew": g3, "kurt": g4, "DSR": dsr}


def cmd_report(ctx: Context) -> dict:
    """汇总关键数字（不重新回测）；报告正文由研究者据此撰写。"""
    grid = pd.read_csv(OUT / "10_组合_网格.csv")
    tabs = {}
    for name in ["by_size_decile", "by_adv_quintile", "spread_size", "spread_size_by_adv",
                 "spread_adv_by_size", "spread_vol_by_size", "spread_trend_by_size",
                 "spread_trend_by_adv", "spread_size_stratified_avg", "size_x_adv",
                 "size_trend_vol", "long_short_ann"]:
        fp = OUT / f"10_事件_{name}.csv"
        if fp.exists():
            tabs[name] = pd.read_csv(fp)
    # 归属判定（预注册 §5.1）
    raw = float(tabs["spread_size"].iloc[0]["spread20"])
    strat = tabs["spread_size_stratified_avg"]
    matched_w = float(strat.iloc[0]["spread20"])
    strat_t = float(strat.iloc[0]["t20"])
    ratio = matched_w / raw if raw else np.nan
    dec = tabs["by_size_decile"]
    beta_lo = float(dec.iloc[0]["beta"])
    beta_hi = float(dec.iloc[-1]["beta"])
    alpha_t = float(dec.iloc[0]["t_alpha20"])
    ts = tabs["spread_trend_by_size"]
    t_trend_max = float(ts["t20"].abs().max())
    vs = tabs["spread_vol_by_size"]
    verdict = {
        "raw_size_spread20": raw,
        "adv_matched_size_spread20_stratified": matched_w,
        "adv_matched_t20": strat_t,
        "matched_retention_ratio": ratio,
        "beta_small_minus_big": beta_lo - beta_hi,
        "alpha20_t_small": alpha_t,
        "trend_spread_within_size_max_abs_t": t_trend_max,
        "lowvol_spread_within_size": vs[["size_ter", "spread20", "t20", "spread60", "t60"]]
            .to_dict(orient="records"),
        "H1_size": bool(raw > 0 and float(tabs["spread_size"].iloc[0]["t20"]) >= 2),
        "H2_liquidity": bool(ratio < 1 / 3 or abs(strat_t) < 1),
        "H3_exposure": bool((beta_lo - beta_hi) >= 0.10 and abs(alpha_t) < 2),
        "H4_trend_lowvol": bool(t_trend_max >= 2),
        "bench_cagr": ctx.meta.get("bench_cagr"),
    }
    key = {"verdict": verdict, "meta": ctx.meta, "bench_cagr": ctx.meta.get("bench_cagr")}
    key["stratified_index"] = _stratified_index(ctx)
    # 主单元 bootstrap / DSR / 滚动 12 个月
    eq = pd.read_csv(OUT / "13_组合_净值_主单元.csv")
    eq["date"] = pd.to_datetime(eq["date"])
    from 统计验证 import block_bootstrap
    boot = {}
    for w in ("dev", "val"):
        sub = eq[(eq["variant"] == "COMBO") & (eq["window"] == w)].sort_values("date")
        nav = sub.set_index("date")["nav"]
        r = nav.pct_change().dropna()
        boot[w] = block_bootstrap(r, n_boot=2000, method="stationary",
                                  seed=20261026, dates=nav.index)
        roll = (nav / nav.shift(243) - 1).dropna()
        boot[w]["worst_roll12m"] = float(roll.min()) if len(roll) else None
        boot[w]["cagr_actual"] = float((nav.iloc[-1] / nav.iloc[0]) **
                                       (365.25 / (nav.index[-1] - nav.index[0]).days) - 1)
        sr_d = float(r.mean() / r.std())
        boot[w]["dsr63"] = _dsr(grid["sharpe"].to_numpy(), sr_d, r, 63)
        boot[w]["dsr112"] = _dsr(grid["sharpe"].to_numpy(), sr_d, r, 112)
    key["primary_bootstrap"] = boot
    # 规模溢价的可复现幅度：SIZE − RAND（组合层）与事件层长-空
    sp = []
    for (slots, hold), g in grid[grid["variant"].isin(["SIZE", "RAND", "COMBO", "ADVQ12"])].groupby(["slots", "hold"]):
        for w in ("dev", "val"):
            sub = g[g["window"] == w]
            if len(sub) < 4:
                continue
            v = sub.set_index("variant")["cagr"]
            sp.append({"slots": slots, "hold": hold, "window": w,
                       "RAND": v.get("RAND"), "SIZE_minus_RAND": v.get("SIZE", np.nan) - v.get("RAND", np.nan),
                       "COMBO_minus_RAND": v.get("COMBO", np.nan) - v.get("RAND", np.nan),
                       "ADVQ12_minus_RAND": v.get("ADVQ12", np.nan) - v.get("RAND", np.nan)})
    key["size_premium_portfolio"] = sp
    pd.DataFrame(sp).to_csv(OUT / "02b_规模溢价_组合层.csv", index=False, float_format="%.6g")
    # 暴露分解：beta × B1 年化 = 可被市场解释的部分
    bc = ctx.meta.get("bench_cagr", {})
    beta_rows = []
    for v in VARIANTS:
        for w in ("dev", "val", "full"):
            sub = grid[(grid["variant"] == v) & (grid["window"] == w) &
                       (grid["slots"] == 10) & (grid["hold"] == 40)]
            if not len(sub):
                continue
            b = float(sub["beta_to_B1"].iloc[0])
            c = float(sub["cagr"].iloc[0])
            mkt = float(bc.get(w, np.nan))
            beta_rows.append({"variant": v, "window": w, "cagr": c, "beta": b,
                              "bench_cagr": mkt, "beta_expected": b * mkt,
                              "alpha_vs_beta": c - b * mkt})
    key["beta_decomposition"] = beta_rows
    pd.DataFrame(beta_rows).to_csv(OUT / "02c_暴露分解.csv", index=False, float_format="%.6g")
    prim = grid[grid["is_primary"] == True]
    key["primary_combo"] = prim[["window", "cagr", "mdd", "sharpe", "n_closed",
                                 "turnover_ann", "part_mean", "part_p95",
                                 "impact_cost_ann_pct", "avg_gross", "avg_cost",
                                 "cagr_ex_top5", "beta_to_B1"]].to_dict(orient="records")
    (OUT / "17_关键数字.json").write_text(
        json.dumps(_json_safe(key), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([verdict]).to_csv(OUT / "02_规模流动性暴露归因.csv", index=False,
                                   float_format="%.6g")
    log(json.dumps(_json_safe(verdict), ensure_ascii=False)[:900])
    return key


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="counts",
                    choices=["counts", "events", "portfolio", "stress", "selfcheck",
                             "ledger", "report", "all"])
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_cache").mkdir(exist_ok=True)
    ctx = build_context()
    if args.mode in ("counts", "all"):
        cmd_counts(ctx)
    if args.mode in ("events", "all"):
        cmd_events(ctx)
    if args.mode in ("portfolio", "all"):
        cmd_portfolio(ctx)
    if args.mode in ("stress", "all"):
        cmd_stress(ctx)
    if args.mode in ("selfcheck", "all"):
        cmd_selfcheck(ctx)
    if args.mode in ("ledger", "all"):
        write_ledger()
    if args.mode in ("report", "all"):
        cmd_report(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
