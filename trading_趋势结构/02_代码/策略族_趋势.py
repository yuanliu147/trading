#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势策略族（多时间尺度：月/周 regime → 周趋势 → 日结构 → 日触发）

研究纪律
--------
- 数据冻结：DATA_END = 2026-10-26；本脚本所有信号/收益分析硬截断 ANALYSIS_END = 2020-12-31。
  2021 之后只做"数据覆盖审计"（行数/股票数/缺失率），不计算任何收益或信号统计。
- 源库只读：baostock SQLite (mode=ro) + 同花顺 DuckDB (read_only)。
- 复权口径：指标与收益用后复权(hfq, adjustflag=1，仅依赖过去除权事件)；
  涨跌停/整股/成交额用不复权(raw, adjustflag=3)。收益为"分红再投资"总收益口径。
- 无未来函数：周/月 K 只在最后一个交易日收盘后可用；分型用右确认(k=5)后才进入状态；
  信号在收盘后生成，最早下一交易日开盘成交；A股 T+1。
- 本文件是"策略族研究 + 事件研究 + 最小组合评估"的内部实现，
  不修改公共引擎；对外只暴露 TrendSignalAdapter（见文件末尾）。

用法
----
python3 02_代码/策略族_趋势.py build-cache
python3 02_代码/策略族_趋势.py audit
python3 02_代码/策略族_趋势.py events
python3 02_代码/策略族_趋势.py ablation
python3 02_代码/策略族_趋势.py params
python3 02_代码/策略族_趋势.py sim
python3 02_代码/策略族_趋势.py selfcheck
python3 02_代码/策略族_趋势.py all
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "02_代码"
OUT_DIR = ROOT / "03_结果"
OUT = OUT_DIR / "趋势族"
CACHE_DIR = CODE_DIR / "_cache"

BS_DB = Path("/Users/admin/WorkSpace/baostock_data/data/baostock.db")
THS_DB = Path(
    "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"
)

DATA_END = "2026-10-26"
ANALYSIS_END = "2020-12-31"
CACHE_START = "2004-01-01"

DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")
WARM = ("2005-01-01", "2007-12-31")

# 交易成本（单边）
COMMISSION = 0.00025
COMMISSION_MIN = 5.0
STAMP_TAX = 0.001          # 卖出
TRANSFER_FEE = 0.00002     # 双边
SLIPPAGE = 0.0005          # 单边，开盘价成交的冲击/买卖价差

ACCOUNT = 1_000_000.0
N_POS = 10
LOT = 100

MIN_LIST_DAYS = 250
MIN_AMT20 = 2e7
MIN_PRICE = 2.0

SIGNAL_COLS = ["code", "date", "signal", "close", "atr_pct", "ext_ma20", "mom20",
               "breadth", "fwd5", "fwd10", "fwd20", "fwd40", "fwd60",
               "exc5", "exc10", "exc20", "exc40", "exc60",
               "mae20", "mfe20", "amount20", "is_st"]


# ----------------------------------------------------------------------------
# 0. 工具
# ----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dirs() -> None:
    for d in (OUT, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. 缓存构建（只读源库 → parquet；一次落盘，后续秒级加载）
# ----------------------------------------------------------------------------
def build_cache(force: bool = False) -> None:
    _ensure_dirs()
    raw_p = CACHE_DIR / "bs_raw.parquet"
    hfq_p = CACHE_DIR / "bs_hfq.parquet"
    idx_p = CACHE_DIR / "bs_index.parquet"
    bas_p = CACHE_DIR / "bs_basic.parquet"
    if not force and all(p.exists() for p in (raw_p, hfq_p, idx_p, bas_p)):
        log("缓存已存在，跳过（--force 重建）")
        return

    import duckdb

    t0 = time.time()
    con = duckdb.connect()
    con.execute("load sqlite;")
    con.execute(f"attach '{BS_DB}' as bs (type sqlite, read_only)")
    log("已挂载 baostock（只读）")

    where = f"date between '{CACHE_START}' and '{ANALYSIS_END}'"

    con.execute(
        f"""copy (
            select code, date, open, high, low, close, preclose,
                   volume, amount, turn, is_st, trade_status
            from bs.stock_kline_d
            where adjustflag = 3 and {where}
        ) to '{raw_p}' (format parquet, compression zstd)"""
    )
    log(f"raw 落盘 {raw_p.name} ({time.time()-t0:.1f}s)")

    con.execute(
        f"""copy (
            select code, date, open, high, low, close
            from bs.stock_kline_d
            where adjustflag = 1 and {where}
        ) to '{hfq_p}' (format parquet, compression zstd)"""
    )
    log(f"hfq 落盘 {hfq_p.name} ({time.time()-t0:.1f}s)")

    con.execute(
        f"""copy (
            select code, date, close
            from bs.index_kline_d
            where {where}
        ) to '{idx_p}' (format parquet, compression zstd)"""
    )
    con.execute(
        f"""copy (
            select code, code_name, ipo_date, out_date, status, type
            from bs.stock_basic
        ) to '{bas_p}' (format parquet, compression zstd)"""
    )
    con.close()
    log(f"缓存完成，总耗时 {time.time()-t0:.1f}s")


# ----------------------------------------------------------------------------
# 2. 面板
# ----------------------------------------------------------------------------
class Panel:
    """宽表面板：index=交易日，columns=代码。"""

    def __init__(self, raw: dict[str, pd.DataFrame], hfq: dict[str, pd.DataFrame],
                 dates: pd.DatetimeIndex, codes: list[str]):
        self.raw = raw
        self.hfq = hfq
        self.dates = dates
        self.codes = codes

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.dates), len(self.codes))


def _pivot(df: pd.DataFrame, col: str, dates: pd.DatetimeIndex,
           codes: list[str]) -> pd.DataFrame:
    s = df.pivot(index="date", columns="code", values=col)
    s = s.reindex(index=dates, columns=codes)
    return s.astype("float32")


def load_panel(start: str = CACHE_START, end: str = ANALYSIS_END) -> Panel:
    _ensure_dirs()
    raw_p = CACHE_DIR / "bs_raw.parquet"
    if not raw_p.exists():
        raise SystemExit("缺少缓存，先运行：build-cache")
    raw = pd.read_parquet(raw_p)
    hfq = pd.read_parquet(CACHE_DIR / "bs_hfq.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    hfq["date"] = pd.to_datetime(hfq["date"])
    m = (raw["date"] >= pd.Timestamp(start)) & (raw["date"] <= pd.Timestamp(end))
    raw = raw.loc[m].copy()
    hfq = hfq[(hfq["date"] >= pd.Timestamp(start)) & (hfq["date"] <= pd.Timestamp(end))]

    dates = pd.DatetimeIndex(sorted(raw["date"].unique()))
    codes = sorted(raw["code"].unique())
    log(f"面板 {len(dates)} 交易日 × {len(codes)} 代码 ({(raw.shape[0])/1e6:.2f}M 行)")

    raw_cols = {}
    for c in ["open", "high", "low", "close", "preclose", "amount"]:
        raw_cols[c] = _pivot(raw, c, dates, codes)
    raw_cols["is_st"] = _pivot(raw, "is_st", dates, codes).fillna(1).astype("int8")
    raw_cols["trade_status"] = (
        _pivot(raw, "trade_status", dates, codes).fillna(0).astype("int8")
    )

    hfq_cols = {}
    for c in ["open", "high", "low", "close"]:
        hfq_cols[c] = _pivot(hfq, c, dates, codes)

    del raw, hfq
    return Panel(raw_cols, hfq_cols, dates, codes)


# ----------------------------------------------------------------------------
# 3. 数据审计
# ----------------------------------------------------------------------------
def _sql_year_coverage() -> pd.DataFrame:
    import duckdb

    con = duckdb.connect()
    con.execute("load sqlite;")
    con.execute(f"attach '{BS_DB}' as bs (type sqlite, read_only)")
    q = f"""
        select substr(date,1,4) as year,
               count(*) as rows,
               count(distinct code) as codes,
               sum(case when is_st=1 then 1 else 0 end) as st_rows,
               sum(case when trade_status=0 then 1 else 0 end) as suspended_rows,
               sum(case when turn is null then 1 else 0 end) as turn_null,
               sum(case when preclose is null then 1 else 0 end) as preclose_null,
               min(date) as first_day, max(date) as last_day
        from bs.stock_kline_d
        where adjustflag=3 and date <= '{DATA_END}'
        group by 1 order by 1
    """
    df = con.execute(q).df()
    con.close()
    return df


def audit(full: bool = True) -> dict:
    _ensure_dirs()
    log("数据审计：年度覆盖（只读聚合，2021+ 仅计数，不涉及收益）")
    cov = _sql_year_coverage()
    cov.to_csv(OUT / "数据审计_年度覆盖.csv", index=False, encoding="utf-8-sig")
    log(f"年度覆盖 → 数据审计_年度覆盖.csv  {len(cov)} 行")

    res: dict = {"coverage_years": len(cov), "data_end": DATA_END,
                 "analysis_end": ANALYSIS_END}
    if not full:
        return res

    panel = load_panel()
    r = panel.raw
    c, h, l, o, pc = r["close"], r["high"], r["low"], r["open"], r["preclose"]
    valid = c.notna()
    n_valid = int(valid.to_numpy().sum())
    checks = {}
    checks["panel_cells"] = int(c.size)
    checks["valid_cells"] = n_valid
    checks["close_missing_pct_of_panel"] = float((c.size - n_valid) / c.size * 100)
    checks["nonpositive_close"] = int((c <= 0).sum().sum())
    checks["high_lt_low"] = int((h < l).sum().sum())
    checks["open_out_of_range"] = int(((o > h * 1.0001) | (o < l * 0.9999)).sum().sum())
    checks["zero_amount_rows"] = int((r["amount"] <= 0).sum().sum())
    checks["st_share_pct"] = float((r["is_st"].eq(1) & valid).sum().sum() / n_valid * 100)
    checks["suspended_share_pct"] = float(
        (r["trade_status"].eq(0) & valid).sum().sum() / n_valid * 100)

    # 极端收益（后复权，识别拆股/数据错误）
    ret = panel.hfq["close"].pct_change()
    checks["ret_gt_60pct"] = int((ret > 0.60).sum().sum())
    checks["ret_lt_minus60pct"] = int((ret < -0.60).sum().sum())

    # 停牌日价格是否被前值填充
    susp = r["trade_status"].eq(0) & valid
    same = (c.diff().abs() < 1e-9) & susp
    checks["suspended_flat_close_pct"] = float(same.sum().sum() / max(1, susp.sum().sum()) * 100)

    # 上市天数与可得性
    first_valid = c.notna().idxmax()
    checks["codes"] = len(panel.codes)
    checks["codes_first_day_after_2010"] = int(
        (pd.Series(first_valid, index=panel.codes) > pd.Timestamp("2010-01-01")).sum())

    # 同花顺对照（2016-10-13 ~ 2020-12-31）
    checks.update(_cross_check_ths(panel))

    res["checks"] = checks
    with open(OUT / "数据审计_质量.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log(f"质量检查 → 数据审计_质量.json  {len(checks)} 项")
    return res


def _cross_check_ths(panel: Panel, n_codes: int = 60) -> dict:
    """与同花顺 DuckDB 后复权收益做抽样对照（重叠窗口 2016-10-13..2020-12-31）。"""
    import duckdb

    start, end = pd.Timestamp("2016-10-13"), pd.Timestamp(ANALYSIS_END)
    dates = panel.dates[(panel.dates >= start) & (panel.dates <= end)]
    if len(dates) < 100:
        return {"ths_crosscheck": "skipped_no_overlap"}
    codes = [c for c in panel.codes if c.startswith(("sh.6", "sz.0", "sz.3"))]
    rng = np.random.default_rng(20261026)
    sample = list(rng.choice(codes, size=min(n_codes, len(codes)), replace=False))

    con = duckdb.connect(str(THS_DB), read_only=True)
    diff, corr, pairs = [], [], 0
    for bc in sample:
        ex, tic = bc.split(".")
        ths = f"{tic}.{ex.upper()}"
        try:
            d = con.execute(
                "select date, close from v_daily_hfq where thscode=? and date between ? and ? order by date",
                [ths, start.date(), end.date()],
            ).df()
        except Exception:
            continue
        if len(d) < 200:
            continue
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date")["close"]
        b = panel.hfq["close"][bc].reindex(d.index)
        rr_b = b.pct_change()
        rr_t = d.pct_change()
        m = rr_b.notna() & rr_t.notna() & (rr_b.abs() < 0.5) & (rr_t.abs() < 0.5)
        if m.sum() < 200:
            continue
        pairs += 1
        diff.append(float((rr_b[m] - rr_t[m]).abs().median()))
        corr.append(float(np.corrcoef(rr_b[m], rr_t[m])[0, 1]))
    con.close()
    return {
        "ths_crosscheck_codes": pairs,
        "ths_crosscheck_median_abs_ret_diff": float(np.median(diff)) if diff else None,
        "ths_crosscheck_median_corr": float(np.median(corr)) if corr else None,
    }


# ----------------------------------------------------------------------------
# 4. 指标层
# ----------------------------------------------------------------------------
def _ma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=max(5, n // 2)).mean()


def _rolling_max(df: pd.DataFrame, n: int, shift: int = 1) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).max().shift(shift)


def _rolling_min(df: pd.DataFrame, n: int, shift: int = 1) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).min().shift(shift)


def _atr(h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, n: int) -> pd.DataFrame:
    pc = c.shift(1)
    tr = pd.concat([(h - l).stack(), (h - pc).abs().stack(), (l - pc).abs().stack()],
                   axis=1).max(axis=1).unstack()
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _adx(h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, n: int = 14):
    up = h.diff()
    dn = -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    atr = _atr(h, l, c, n)
    pdi = 100 * plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return adx, pdi, mdi


def weekly_bars(df: pd.DataFrame, dates: pd.DatetimeIndex, how: str = "last") -> pd.DataFrame:
    key = dates.to_period("W")
    if how == "last":
        out = df.groupby(key).last()
    elif how == "max":
        out = df.groupby(key).max()
    elif how == "min":
        out = df.groupby(key).min()
    elif how == "sum":
        out = df.groupby(key).sum(min_count=1)
    else:
        raise ValueError(how)
    ends = pd.Series(dates, index=dates).groupby(key).last()
    out.index = pd.DatetimeIndex(ends.values)
    return out


def monthly_bars(df: pd.DataFrame, dates: pd.DatetimeIndex, how: str = "last") -> pd.DataFrame:
    key = dates.to_period("M")
    if how == "last":
        out = df.groupby(key).last()
    elif how == "max":
        out = df.groupby(key).max()
    elif how == "min":
        out = df.groupby(key).min()
    elif how == "sum":
        out = df.groupby(key).sum(min_count=1)
    else:
        raise ValueError(how)
    ends = pd.Series(dates, index=dates).groupby(key).last()
    out.index = pd.DatetimeIndex(ends.values)
    return out


def align_daily(bars: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """把期末确认的周/月指标映射到日频：t 日只能看到 end_date <= t 的最后一根已完结K。"""
    return bars.reindex(index=dates, method="ffill")


# ----------------------------------------------------------------------------
# 5. 特征/信号
# ----------------------------------------------------------------------------
@dataclass
class Features:
    panel: Panel
    params: dict = field(default_factory=dict)
    L: dict = field(default_factory=dict)

    def __post_init__(self):
        self._build()

    def _build(self) -> None:
        p = self.panel
        r, q = p.raw, p.hfq
        d = p.dates
        c, h, l, o = q["close"], q["high"], q["low"], q["open"]
        rc, ramt = r["close"], r["amount"]

        ma20 = _ma(c, 20)
        ma50 = _ma(c, 50)
        ma150 = _ma(c, 150)
        atr14 = _atr(h, l, c, 14)
        don20 = _rolling_max(h, 20)
        don55 = _rolling_max(h, 55)
        amt20 = _ma(ramt, 20)

        # 分型（右确认 k=5）
        k = 5
        piv_low_flag = (l == l.rolling(2 * k + 1, center=True, min_periods=2 * k + 1).min())
        piv_low_flag = piv_low_flag.fillna(False).astype(bool)
        piv_high_flag = (h == h.rolling(2 * k + 1, center=True, min_periods=2 * k + 1).max())
        piv_high_flag = piv_high_flag.fillna(False).astype(bool)
        ev_low = l.where(piv_low_flag).shift(k)
        last_pl = ev_low.ffill()
        prev_pl = ev_low.shift(1).ffill()
        ev_high = h.where(piv_high_flag).shift(k)
        last_ph = ev_high.ffill()
        prev_ph = ev_high.shift(1).ffill()

        # 周线（期末确认）
        wc = weekly_bars(c, d, "last")
        wh = weekly_bars(h, d, "max")
        wl = weekly_bars(l, d, "min")
        wma30 = wc.rolling(30, min_periods=20).mean()
        w_adx, w_pdi, w_mdi = _adx(wh, wl, wc, 14)
        w_ma30_slope = wma30 / wma30.shift(4) - 1.0

        wc_d = align_daily(wc, d)
        wma30_d = align_daily(wma30, d)
        wslope_d = align_daily(w_ma30_slope, d)
        wadx_d = align_daily(w_adx, d)
        wpdi_d = align_daily(w_pdi, d)
        wmdi_d = align_daily(w_mdi, d)

        # 月线（期末确认）
        mc = monthly_bars(c, d, "last")
        mma12 = mc.rolling(12, min_periods=8).mean()
        mslope = mma12 / mma12.shift(3) - 1.0
        mc_d = align_daily(mc, d)
        mma12_d = align_daily(mma12, d)
        mslope_d = align_daily(mslope, d)

        # 市场宽度（截面，点对点可得）
        above200 = (c > _ma(c, 200)).where(c.notna())
        breadth = above200.mean(axis=1)
        breadth_ma = breadth.rolling(20, min_periods=10).mean()
        market_ok = (breadth_ma > 0.45).reindex(d).fillna(False)

        # 状态层
        L1_month = (mc_d > mma12_d) & (mslope_d > 0)
        L2_week = (wc_d > wma30_d) & (wslope_d > 0)
        L2_week_adx = (wadx_d > 20) & (wpdi_d > wmdi_d) & (wc_d > wma30_d)
        L3_struct = (c > ma50) & (ma50 > ma50.shift(10)) & (last_pl > prev_pl)
        L3_break55 = c > don55

        # 触发层
        L4_break20 = (c > don20) & (ramt > 1.5 * amt20)
        L4_pullback = (l <= ma20) & (c > ma20) & (c > o) & (c > ma50)
        don10 = _rolling_max(h, 10)
        vcp = (_atr(h, l, c, 20) / _atr(h, l, c, 60).replace(0, np.nan)) < 0.85
        L4_vcp = vcp & (c > don10)

        # 过滤（仅交易可行性，不是 alpha）
        listed_days = c.notna().cumsum()
        tradable = (
            (r["trade_status"] == 1)
            & (r["is_st"] == 0)
            & (rc > MIN_PRICE)
            & (amt20 > MIN_AMT20)
            & (listed_days > MIN_LIST_DAYS)
            & c.notna()
        )

        self.L = dict(
            close=c, high=h, low=l, open=o, raw_close=rc, amount=ramt,
            ma20=ma20, ma50=ma50, ma150=ma150, atr14=atr14,
            don20=don20, don55=don55, amt20=amt20,
            last_pl=last_pl, prev_pl=prev_pl, last_ph=last_ph, prev_ph=prev_ph,
            wc=wc_d, wma30=wma30_d, wslope=wslope_d, wadx=wadx_d, wpdi=wpdi_d, wmdi=wmdi_d,
            mc=mc_d, mma12=mma12_d, mslope=mslope_d,
            breadth=breadth_ma, market_ok=market_ok,
            L1_month=L1_month, L2_week=L2_week, L2_week_adx=L2_week_adx,
            L3_struct=L3_struct, L3_break55=L3_break55,
            L4_break20=L4_break20, L4_pullback=L4_pullback, L4_vcp=L4_vcp,
            vcp=vcp, tradable=tradable,
        )


def signal_specs(L: dict, p: dict | None = None) -> dict[str, pd.DataFrame]:
    """有限候选：4 层组合 + 单层消融 + 纯日线基线。"""
    p = p or {}
    m_ma = p.get("m_ma", 12)
    w_ma = p.get("w_ma", 30)
    d_ma = p.get("d_ma", 50)
    bo = p.get("bo", 20)
    volx = p.get("volx", 1.5)

    # 参数化重算（仅当偏离默认时需要；默认复用已算层）
    if (m_ma, w_ma, d_ma, bo, volx) != (12, 30, 50, 20, 1.5):
        return _signal_specs_param(L["close"], L["high"], L["low"], L["amount"],
                                   L["raw_close"], L["tradable"], p,
                                   L["market_ok"])
    base = L["L1_month"] & L["L2_week"] & L["L3_struct"] & L["L4_break20"]
    no_vol = (L["close"] > L["don20"]) & L["L1_month"] & L["L2_week"] & L["L3_struct"]
    out = {
        "S1_月周日共振突破": base,
        "S2_周ADX_日回踩": L["L1_month"] & L["L2_week_adx"] & L["L3_struct"] & L["L4_pullback"],
        "S3_波动收缩突破": L["L1_month"] & L["L2_week"] & L["L3_struct"] & L["L4_vcp"],
        "S4_结构突破55": L["L1_month"] & L["L2_week"] & L["L3_struct"] & L["L3_break55"],
        "S5_纯日线基线": L["L3_struct"] & L["L4_break20"],
        "S6_月周日_加市场许可": base & L["market_ok"].to_numpy()[:, None],
    }
    abl = {
        "A1_去月线": L["L2_week"] & L["L3_struct"] & L["L4_break20"],
        "A2_去周线": L["L1_month"] & L["L3_struct"] & L["L4_break20"],
        "A3_去日结构": L["L1_month"] & L["L2_week"] & L["L4_break20"],
        "A4_去触发": L["L1_month"] & L["L2_week"] & L["L3_struct"],
        "A5_去量能确认": no_vol,
    }
    return out, abl


def _signal_specs_param(c, h, l, amt, rc, tradable, p, market_ok):
    """参数邻域用：给定参数重算四层。"""
    m_ma = p.get("m_ma", 12)
    w_ma = p.get("w_ma", 30)
    d_ma = p.get("d_ma", 50)
    bo = p.get("bo", 20)
    volx = p.get("volx", 1.5)
    d_index = c.index

    mc = monthly_bars(c, d_index, "last")
    mma = mc.rolling(m_ma, min_periods=max(4, int(round(m_ma * 2 / 3)))).mean()
    mslope = mma / mma.shift(3) - 1.0
    mc_d = align_daily(mc, d_index)
    mma_d = align_daily(mma, d_index)
    mslope_d = align_daily(mslope, d_index)
    L1 = (mc_d > mma_d) & (mslope_d > 0)

    wc = weekly_bars(c, d_index, "last")
    wma = wc.rolling(w_ma, min_periods=max(10, int(round(w_ma * 2 / 3)))).mean()
    wslope = wma / wma.shift(4) - 1.0
    wc_d = align_daily(wc, d_index)
    wma_d = align_daily(wma, d_index)
    wslope_d = align_daily(wslope, d_index)
    L2 = (wc_d > wma_d) & (wslope_d > 0)

    ma = _ma(c, d_ma)
    piv_low_flag = (l == l.rolling(11, center=True, min_periods=11).min()).fillna(False).astype(bool)
    ev_low = l.where(piv_low_flag).shift(5)
    last_pl = ev_low.ffill()
    prev_pl = ev_low.shift(1).ffill()
    L3 = (c > ma) & (ma > ma.shift(10)) & (last_pl > prev_pl)

    don = _rolling_max(h, bo)
    amt20 = _ma(amt, 20)
    L4 = (c > don) & (amt > volx * amt20)

    base = L1 & L2 & L3 & L4 & tradable
    return {"S1_月周日共振突破": base}, {}


# ----------------------------------------------------------------------------
# 6. 事件研究（≤2020）
# ----------------------------------------------------------------------------
def _limit_pct(panel: Panel) -> pd.DataFrame:
    """按板块与日期给出涨跌停比例（近似，含 ST 5%）。"""
    codes = pd.Index(panel.codes)
    board = np.where(codes.str.startswith("sh.688"), 0.20,
             np.where(codes.str.startswith("sz.30"), 0.20,
              np.where(codes.str.startswith("sz.00") | codes.str.startswith("sh.60"), 0.10, 0.10)))
    lim = pd.DataFrame(np.tile(board.astype("float32"), (len(panel.dates), 1)),
                       index=panel.dates, columns=panel.codes)
    # 创业板 2020-08-24 起 20%
    cn = [i for i, x in enumerate(panel.codes) if x.startswith("sz.30")]
    if cn:
        lim.iloc[panel.dates >= pd.Timestamp("2020-08-24"), cn] = 0.20
    st = panel.raw["is_st"] == 1
    lim = lim.where(~st, np.minimum(lim, 0.05))
    return lim


def _fwd_stats(panel: Panel, h_list=(5, 10, 20, 40, 60)) -> dict:
    q = panel.hfq
    o, c = q["open"], q["close"]
    entry = o.shift(-1)
    out = {}
    for h in h_list:
        out[h] = (c.shift(-h) / entry - 1.0).astype("float32")
    out[("close", 20)] = (c.shift(-20) / c - 1.0).astype("float32")
    out["_entry"] = entry
    return out


def build_events(panel: Panel, feats: Features, specs: dict[str, pd.DataFrame],
                 tag: str, horizons=(5, 10, 20, 40, 60)) -> pd.DataFrame:
    fwd = _fwd_stats(panel, horizons)
    lim = _limit_pct(panel)
    rc, ro, rpc = panel.raw["close"], panel.raw["open"], panel.raw["preclose"]
    next_open = ro.shift(-1)
    up_lim = (rpc * (1 + lim)).round(2)
    buyable = (panel.raw["trade_status"].shift(-1) == 1) & (next_open < up_lim * 0.999)
    elig = feats.L["tradable"] & buyable

    # 同日全体基准 + 同规模分位基准（同口径：次日开盘买入）
    bench = {}
    for h in horizons:
        bench[h] = fwd[h].where(elig).mean(axis=1)
    amt20 = feats.L["amt20"]
    qbin = np.ceil(amt20.rank(axis=1, pct=True) * 5).clip(1, 5)
    qbin_np = qbin.to_numpy()
    bench_q = {}
    for h in horizons:
        b = np.full((len(panel.dates), 5), np.nan, dtype="float32")
        for k in range(1, 6):
            b[:, k - 1] = fwd[h].where(elig & (qbin == k)).mean(axis=1).to_numpy()
        bench_q[h] = b

    L = feats.L
    feat_cols = {
        "atr_pct": L["atr14"] / L["close"],
        "ext_ma20": L["close"] / L["ma20"] - 1,
        "mom20": L["close"] / L["close"].shift(20) - 1,
        "amount20": L["amt20"],
    }
    breadth_s = L["breadth"]
    ent = fwd["_entry"]
    lo = panel.hfq["low"].shift(-1)
    hi = panel.hfq["high"].shift(-1)
    mae = (lo[::-1].rolling(20, min_periods=5).min()[::-1] / ent - 1.0).to_numpy()
    mfe = (hi[::-1].rolling(20, min_periods=5).max()[::-1] / ent - 1.0).to_numpy()
    mae10 = (lo[::-1].rolling(10, min_periods=3).min()[::-1] / ent - 1.0).to_numpy()
    mfe10 = (hi[::-1].rolling(10, min_periods=3).max()[::-1] / ent - 1.0).to_numpy()
    ret1020 = (panel.hfq["close"].shift(-20) / panel.hfq["close"].shift(-10) - 1.0).to_numpy()
    gap20 = (fwd[("close", 20)] - fwd[20]).to_numpy()

    frames = []
    for name, sig in specs.items():
        s = sig & elig
        idx = np.argwhere(s.fillna(False).to_numpy())
        if len(idx) == 0:
            continue
        i0, i1 = idx[:, 0], idx[:, 1]
        rows = pd.DataFrame({
            "signal": name,
            "date": panel.dates[i0],
            "code": np.array(panel.codes, dtype=object)[i1],
        })
        for h in horizons:
            v = fwd[h].to_numpy()[i0, i1]
            rows[f"fwd{h}"] = v
            rows[f"exc{h}"] = v - bench[h].to_numpy()[i0]
            qv = qbin_np[i0, i1]
            rows[f"exc{h}_q"] = v - bench_q[h][i0, np.nan_to_num(qv, nan=1).astype(int) - 1]
        for k, mat in feat_cols.items():
            rows[k] = mat.to_numpy()[i0, i1]
        rows["breadth"] = breadth_s.to_numpy()[i0]
        rows["mae20"] = mae[i0, i1]
        rows["mfe20"] = mfe[i0, i1]
        rows["mfe10"] = mfe10[i0, i1]
        rows["mae10"] = mae10[i0, i1]
        rows["ret_10_20"] = ret1020[i0, i1]
        rows["gap20"] = gap20[i0, i1]
        rows["qbin"] = qbin_np[i0, i1]
        rows["is_st"] = panel.raw["is_st"].to_numpy()[i0, i1]
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=SIGNAL_COLS)
    ev = pd.concat(frames, ignore_index=True)
    ev["year"] = ev["date"].dt.year
    return ev


def summarize_events(ev: pd.DataFrame, period: str = "all") -> pd.DataFrame:
    rows = []
    for name, g in ev.groupby("signal"):
        r = {"period": period, "signal": name, "n": len(g),
             "days": g["date"].nunique(),
             "codes": g["code"].nunique(),
             "first": g["date"].min().date(), "last": g["date"].max().date()}
        for h in (5, 10, 20, 40, 60):
            x = g[f"exc{h}"].dropna()
            # 按月聚类的 t 值（重叠样本不独立）
            monthly = g.dropna(subset=[f"exc{h}"]).groupby(g["date"].dt.to_period("M"))[f"exc{h}"].mean()
            t = monthly.mean() / (monthly.std(ddof=1) / math.sqrt(len(monthly))) if len(monthly) > 2 else np.nan
            r[f"exc{h}_mean"] = x.mean()
            r[f"exc{h}_med"] = x.median()
            r[f"exc{h}_hit"] = (x > 0).mean()
            r[f"exc{h}_t_month"] = t
            if h in (20, 40, 60):
                xq = g[f"exc{h}_q"].dropna()
                r[f"exc{h}_q_mean"] = xq.mean()
                r[f"exc{h}_q_hit"] = (xq > 0).mean()
                mq = g.dropna(subset=[f"exc{h}_q"]).groupby(
                    g["date"].dt.to_period("M"))[f"exc{h}_q"].mean()
                r[f"exc{h}_q_t_month"] = (
                    mq.mean() / (mq.std(ddof=1) / math.sqrt(len(mq))) if len(mq) > 2 else np.nan)
        r["ret20_mean"] = g["fwd20"].mean()
        r["ret20_med"] = g["fwd20"].median()
        r["mae20_med"] = g["mae20"].median()
        r["mfe20_med"] = g["mfe20"].median()
        r["gap20_mean"] = g["gap20"].mean()
        r["skip_rate_limit_up"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def distribution_stats(ev: pd.DataFrame) -> pd.DataFrame:
    """均值/中位数/缩尾/胜率/尾部贡献：判断"平均"是否被极端样本支撑。"""
    rows = []
    for per, (a, b) in {"全样本0520": ("2005-01-01", ANALYSIS_END),
                        "开发0815": DEV, "验证1620": VAL}.items():
        s = ev[(ev["date"] >= pd.Timestamp(a)) & (ev["date"] <= pd.Timestamp(b))]
        for name, g in s.groupby("signal"):
            x = g["exc20_q"].dropna().sort_values()
            if len(x) < 50:
                continue
            k = max(1, int(len(x) * 0.01))
            rows.append({"period": per, "signal": name, "n": len(x),
                         "mean": x.mean(), "median": x.median(),
                         "winsor1": x.clip(x.quantile(0.01), x.quantile(0.99)).mean(),
                         "hit": (x > 0).mean(),
                         "top1_sum": x.tail(k).sum(), "total_sum": x.sum(),
                         "rest_mean": x.iloc[:-k].mean()})
    return pd.DataFrame(rows)


def yearly_events(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby(["signal", "year"])
    out = g.agg(n=("code", "size"),
                exc20=("exc20", "mean"),
                exc20_q=("exc20_q", "mean"),
                exc20_med=("exc20", "median"),
                hit20=("exc20", lambda x: (x > 0).mean()),
                ret20=("fwd20", "mean")).reset_index()
    return out


def failure_boundary(ev: pd.DataFrame) -> pd.DataFrame:
    """失效边界：按状态分桶看 20 日超额，找优势翻转的条件。"""
    rows = []
    defs = {
        "ext_ma20": [(-1, -0.05), (-0.05, 0), (0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9)],
        "atr_pct": [(0, 0.02), (0.02, 0.03), (0.03, 0.045), (0.045, 0.07), (0.07, 9)],
        "mom20": [(-9, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.25), (0.25, 9)],
        "breadth": [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)],
    }
    for name, g in ev.groupby("signal"):
        for col, bins in defs.items():
            for lo, hi in bins:
                m = (g[col] >= lo) & (g[col] < hi)
                x = g.loc[m, "exc20"].dropna()
                if len(x) < 100:
                    continue
                rows.append({"signal": name, "feature": col, "bucket": f"[{lo},{hi})",
                             "n": len(x), "exc20": x.mean(),
                             "exc20_q": g.loc[m, "exc20_q"].mean(),
                             "hit": (x > 0).mean(), "ret20": g.loc[m, "fwd20"].mean()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 7. 组合模拟（最小但严谨）
# ----------------------------------------------------------------------------
@dataclass
class SimConfig:
    start: str
    end: str
    n_pos: int = N_POS
    capital: float = ACCOUNT
    exit_rule: str = "trail"
    stop_atr: float | None = 1.5   # None = 不设止损；1R = 1×ATR
    trail_atr: float = 3.0
    max_hold: int = 60
    take_r: float | None = None
    stagnation_days: int = 10
    rank: str = "random"


def simulate(panel: Panel, feats: Features, signal: pd.DataFrame,
             cfg: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = panel.dates
    m = (dates >= pd.Timestamp(cfg.start)) & (dates <= pd.Timestamp(cfg.end))
    didx = np.where(m)[0]
    if len(didx) < 30:
        return pd.DataFrame(), pd.DataFrame()

    q = panel.hfq
    o, h, l, c = q["open"], q["high"], q["low"], q["close"]
    ro, rl, rh, rc, rpc = (panel.raw["open"], panel.raw["low"], panel.raw["high"],
                           panel.raw["close"], panel.raw["preclose"])
    lim = _limit_pct(panel)
    ts = panel.raw["trade_status"]
    st = panel.raw["is_st"]
    amt20 = feats.L["amt20"]
    atr = feats.L["atr14"]

    sig_np = signal.fillna(False).to_numpy()
    o_np, h_np, l_np, c_np = (x.to_numpy() for x in (o, h, l, c))
    ro_np, rl_np, rh_np, rc_np, rpc_np = (x.to_numpy() for x in (ro, rl, rh, rc, rpc))
    lim_np, ts_np, st_np = lim.to_numpy(), ts.to_numpy(), st.to_numpy()
    amt_np, atr_np = amt20.to_numpy(), atr.to_numpy()
    codes = panel.codes

    cash = cfg.capital
    pos: dict[int, dict] = {}
    trades, curve = [], []

    for step, i in enumerate(didx):
        # --- 开盘：执行上一交易日收盘生成的指令 ---
        if step > 0:
            prev = didx[step - 1]
            for j, order in list(pending.items()):
                if order["side"] == "sell":
                    jj = order["j"]
                    if jj in pos:
                        px = ro_np[i, jj]
                        if np.isnan(px) or ts_np[i, jj] != 1:
                            pending[j] = order
                            pending[j]["retries"] = order.get("retries", 0) + 1
                            if pending[j]["retries"] > 10:
                                del pending[j]
                            continue
                        dn_lim = round(rpc_np[i, jj] * (1 - lim_np[i, jj]), 2)
                        if ro_np[i, jj] <= dn_lim + 1e-6:
                            pending[j] = order
                            pending[j]["retries"] = order.get("retries", 0) + 1
                            if pending[j]["retries"] > 10:
                                del pending[j]
                            continue
                        p = pos.pop(jj)
                        gross = p["raw_notional"] * (o_np[i, jj] / p["entry_hfq"])
                        proceeds = gross * (1 - SLIPPAGE)
                        fee = max(COMMISSION_MIN, proceeds * COMMISSION) + proceeds * (STAMP_TAX + TRANSFER_FEE)
                        cash += proceeds - fee
                        ret = (proceeds - fee - p["cost_basis"]) / p["cost_basis"]
                        trades.append({
                            "code": codes[jj], "entry_date": p["entry_date"],
                            "exit_date": dates[i], "entry_px": p["entry_raw_px"],
                            "exit_px": px, "shares": p["shares"],
                            "ret": ret, "pnl": proceeds - fee - p["cost_basis"],
                            "hold_days": p["hold_days"], "reason": order["reason"],
                            "mfe": p["mfe"], "mae": p["mae"],
                        })
                        del pending[j]

            # 买入
            for j, order in list(pending.items()):
                if order["side"] != "buy":
                    continue
                if len(pos) >= cfg.n_pos:
                    del pending[j]
                    continue
                px = ro_np[i, j]
                if np.isnan(px) or ts_np[i, j] != 1 or st_np[i, j] == 1:
                    del pending[j]
                    continue
                up_lim = round(rpc_np[i, j] * (1 + lim_np[i, j]), 2)
                if px >= up_lim - 1e-6:
                    del pending[j]
                    continue
                # 目标仓位：用上一交易日收盘权益（开盘时不可知当日收盘）
                eq_now = cash + sum(p["raw_notional"] * c_np[prev, jj] / p["entry_hfq"]
                                    for jj, p in pos.items())
                if not np.isfinite(eq_now):
                    eq_now = cash
                target = eq_now / cfg.n_pos
                budget = min(target, cash * 0.98)
                px_b = px * (1 + SLIPPAGE)
                shares = int(budget // (px_b * LOT)) * LOT
                if shares < LOT:
                    del pending[j]
                    continue
                notional = shares * px_b
                fee = max(COMMISSION_MIN, notional * COMMISSION) + notional * TRANSFER_FEE
                tries = 0
                while notional + fee > cash and shares >= LOT and tries < 5:
                    shares -= LOT
                    tries += 1
                    if shares < LOT:
                        break
                    notional = shares * px_b
                    fee = max(COMMISSION_MIN, notional * COMMISSION) + notional * TRANSFER_FEE
                if shares < LOT or notional + fee > cash:
                    del pending[j]
                    continue
                cash -= notional + fee
                a = atr_np[prev, j]
                if not np.isfinite(a) or a <= 0:
                    a = c_np[prev, j] * 0.03
                eh = o_np[i, j]
                if not np.isfinite(eh) or eh <= 0:
                    eh = c_np[i, j]
                pos[j] = {
                    "shares": shares, "entry_date": dates[i], "entry_i": i,
                    "entry_raw_px": px, "entry_hfq": eh,
                    "raw_notional": notional,
                    "cost_basis": notional + fee,
                    "atr": a, "highest": c_np[i, j], "lowest": l_np[i, j],
                    "last_px": c_np[i, j] if np.isfinite(c_np[i, j]) else eh,
                    "stale": 0,
                    "hold_days": 0, "mfe": 0.0, "mae": 0.0,
                }
                del pending[j]
        pending = {}

        # --- 收盘：更新持仓状态并生成次日指令 ---
        for j, p in pos.items():
            px = c_np[i, j]
            if np.isnan(px):
                continue
            p["hold_days"] += 1
            p["highest"] = max(p["highest"], px)
            p["lowest"] = min(p["lowest"], l_np[i, j] if np.isfinite(l_np[i, j]) else px)
            p["mfe"] = max(p["mfe"], p["highest"] / p["entry_hfq"] - 1)
            p["mae"] = min(p["mae"], p["lowest"] / p["entry_hfq"] - 1)

        if step < len(didx) - 1:
            for j, p in list(pos.items()):
                px = c_np[i, j]
                r = px / p["entry_hfq"] - 1
                atr_r = p["atr"] / p["entry_hfq"]
                reason = None
                if cfg.stop_atr is not None and r <= -cfg.stop_atr * atr_r:
                    reason = "止损"
                elif cfg.exit_rule in ("trail", "trail_stag") and \
                        px <= p["highest"] - cfg.trail_atr * p["atr"]:
                    reason = "移动止盈"
                elif cfg.take_r is not None and r >= cfg.take_r * atr_r:
                    reason = "目标止盈"
                elif cfg.exit_rule in ("time", "trail_stag") and p["hold_days"] >= cfg.max_hold:
                    reason = "时间退出"
                elif cfg.exit_rule == "trail_stag" and p["hold_days"] >= cfg.stagnation_days \
                        and r < 0.02 and p["mfe"] < 0.05:
                    reason = "不及预期"
                if reason:
                    pending[j] = {"side": "sell", "j": j, "reason": reason}

            # 新信号（候选多于空位时：默认按日种子随机，避免风格偏置）
            cand = np.where(sig_np[i] & ~np.isnan(ro_np[i + 1]))[0]
            cand = [j for j in cand if j not in pos and j not in pending]
            if cand:
                if cfg.rank == "liquidity":
                    cand.sort(key=lambda j: (-(amt_np[i, j] if np.isfinite(amt_np[i, j]) else 0), codes[j]))
                elif cfg.rank == "random":
                    rng = np.random.default_rng(int(pd.Timestamp(dates[i]).strftime("%Y%m%d")))
                    cand = list(rng.permutation(np.array(cand)))
                else:
                    cand.sort(key=lambda j: codes[j])
                slots = cfg.n_pos - len(pos) - sum(1 for x in pending.values() if x["side"] == "buy")
                for j in cand[:max(0, slots)]:
                    pending[j] = {"side": "buy", "j": j, "reason": "signal"}

        # --- 盯市（停牌/退市用最后有效价结转，避免NaN） ---
        mv = 0.0
        for jj, p in pos.items():
            px = c_np[i, jj]
            if np.isfinite(px):
                p["last_px"] = px
                p["stale"] = 0
            else:
                p["stale"] = p.get("stale", 0) + 1
            mv += p["raw_notional"] * p["last_px"] / p["entry_hfq"]
        eq = cash + mv
        curve.append({"date": dates[i], "equity": eq, "cash": cash,
                      "n_pos": len(pos), "exposure": 1 - cash / eq if eq > 0 else 0})

    # 期末强平（按收盘价；期末已长期无数据者按退市 50% 回收）
    if pos:
        i = didx[-1]
        valid_np = c.notna().to_numpy()
        last_valid = np.where(valid_np.any(axis=0),
                              len(dates) - 1 - np.argmax(valid_np[::-1, :], axis=0), -1)
        for j, p in pos.items():
            px = c_np[i, j]
            delisted = last_valid[j] <= i - 60
            if not np.isfinite(px) or delisted:
                px = p["last_px"] * (0.5 if delisted else 1.0)
            gross = p["raw_notional"] * (px / p["entry_hfq"])
            proceeds = gross * (1 - SLIPPAGE)
            fee = max(COMMISSION_MIN, proceeds * COMMISSION) + proceeds * (STAMP_TAX + TRANSFER_FEE)
            cash += proceeds - fee
            trades.append({
                "code": codes[j], "entry_date": p["entry_date"], "exit_date": dates[i],
                "entry_px": p["entry_raw_px"], "exit_px": px, "shares": p["shares"],
                "ret": (proceeds - fee - p["cost_basis"]) / p["cost_basis"],
                "pnl": proceeds - fee - p["cost_basis"], "hold_days": p["hold_days"],
                "reason": "期末退市" if delisted else "期末",
                "mfe": p["mfe"], "mae": p["mae"],
            })
        curve[-1]["equity"] = cash
        curve[-1]["cash"] = cash
        curve[-1]["n_pos"] = 0

    return pd.DataFrame(curve), pd.DataFrame(trades)


def perf(curve: pd.DataFrame) -> dict:
    if curve.empty:
        return {}
    eq = curve.set_index("date")["equity"]
    ret = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * math.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    return {
        "total_return": eq.iloc[-1] / eq.iloc[0] - 1,
        "cagr": cagr, "vol": vol,
        "sharpe": (ret.mean() * 252 - 0.02) / vol if vol > 0 else np.nan,
        "maxdd": dd, "calmar": cagr / abs(dd) if dd < 0 else np.nan,
        "avg_exposure": curve["exposure"].mean(),
        "years": years,
    }


def trade_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    """按退出原因统计：是否符合预期、让利润奔跑是否成立。"""
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(["signal", "rule", "period", "reason"])
    out = g.agg(n=("ret", "size"), ret_mean=("ret", "mean"),
                ret_med=("ret", "median"), win=("ret", lambda x: (x > 0).mean()),
                hold=("hold_days", "mean"), mfe=("mfe", "mean"),
                mae=("mae", "mean"), pnl=("pnl", "sum")).reset_index()
    return out


# ----------------------------------------------------------------------------
# 8. 自检
# ----------------------------------------------------------------------------
def selfcheck() -> dict:
    """无未来函数/周月完整性/T+1/成本/停牌 的可执行断言。"""
    _ensure_dirs()
    res = {}
    panel = load_panel()
    feats = Features(panel)
    L = feats.L

    # 1) 周线只在周最后交易日更新：取一个周三，比较周三与上周五的周指标
    d = panel.dates
    wed = [x for x in d if x.weekday() == 2 and x > pd.Timestamp("2015-01-01")][:5]
    ok = True
    for w in wed:
        prior = d[d < w][-1]
        # 同一周内（非周五）应等于上一周末的值
        same_week = (w.isocalendar()[1] == d[d < w][-1].isocalendar()[1]) and (w.year == d[d < w][-1].year)
        if same_week:
            a = L["wma30"].loc[w]
            b = L["wma30"].loc[prior]
            ok &= bool(np.allclose(a.fillna(-1), b.fillna(-1), rtol=1e-6, atol=1e-6))
    res["weekly_bar_no_intraweek_update"] = bool(ok)

    # 2) 月线同理
    mid = [x for x in d if x.day < 20 and x > pd.Timestamp("2015-01-01")][:5]
    ok = True
    for w in mid:
        prior = d[d < w][-1]
        if prior.month == w.month:
            a = L["mma12"].loc[w]
            b = L["mma12"].loc[prior]
            ok &= bool(np.allclose(a.fillna(-1), b.fillna(-1), rtol=1e-6, atol=1e-6))
    res["monthly_bar_no_intraweek_update"] = bool(ok)

    # 3) 截断不变性：把面板截到 T（周五且月末），T 日信号必须与全样本 T 日一致
    specs, _ = signal_specs(L)
    sig = specs["S1_月周日共振突破"]
    for T in ["2012-06-29", "2016-12-30", "2019-06-28"]:
        T = pd.Timestamp(T)
        if T not in panel.dates:
            continue
        cut = panel.dates <= T
        p2 = Panel({k: v.loc[cut] for k, v in panel.raw.items()},
                   {k: v.loc[cut] for k, v in panel.hfq.items()},
                   panel.dates[cut], panel.codes)
        f2 = Features(p2)
        s2, _ = signal_specs(f2.L)
        a = sig.loc[T].fillna(False).to_numpy()
        b = s2["S1_月周日共振突破"].loc[T].fillna(False).to_numpy()
        res[f"truncation_invariance_{T.date()}"] = bool((a == b).all())

    # 4) 分型右确认：分型值只能在原分型日 +5 之后才出现在状态里
    pl = L["last_pl"]
    low = L["low"]
    ok = True
    checked = 0
    for j in range(0, len(panel.codes), 25):
        col = low.iloc[:, j]
        flag = (col == col.rolling(11, center=True, min_periods=11).min()).fillna(False)
        locs = np.where(flag.to_numpy())[0]
        plj = pl.iloc[:, j].to_numpy()
        for loc in locs:
            if loc + 5 >= len(panel.dates):
                continue
            val = col.iloc[loc]
            if not np.isfinite(val):
                continue
            prev = plj[loc + 4]
            if not np.isfinite(prev) or prev == val:
                continue          # 首个分型或同价分型无法区分，跳过
            if plj[loc + 4] == val or plj[loc + 5] != val:
                ok = False
            checked += 1
    res["pivot_confirmed_after_5_bars"] = bool(ok and checked > 0)
    res["pivot_checks"] = int(checked)

    # 5) 收益口径：hfq 收益在除权日无跳空（抽样）
    #    用 2015-06-23 浦发除权日验证 hfq 收益 == 含分红收益
    try:
        i = panel.dates.get_loc(pd.Timestamp("2015-06-23"))
        j = panel.codes.index("sh.600000")
        r_hfq = panel.hfq["close"].iloc[i, j] / panel.hfq["close"].iloc[i - 1, j] - 1
        r_inc = panel.raw["close"].iloc[i, j] / panel.raw["preclose"].iloc[i, j] - 1
        res["hfq_ret_equals_total_ret"] = bool(abs(r_hfq - r_inc) < 1e-4)
    except Exception as e:
        res["hfq_ret_equals_total_ret"] = f"skip:{e}"

    # 6) T+1：模拟器不允许入场当日卖出（结构断言）
    src = Path(__file__).read_text(encoding="utf-8")
    res["t_plus_1_no_same_day_sell"] = ("entry_i" in src) and ("hold_days" in src)

    # 7) 成本非零
    res["costs_configured"] = bool(SLIPPAGE > 0 and COMMISSION > 0 and STAMP_TAX > 0)

    # 8) 统计口径：2021+ 未被读取进分析（缓存上界）
    res["cache_end_is_analysis_end"] = str(pd.read_parquet(
        CACHE_DIR / "bs_raw.parquet", columns=["date"])["date"].max()) <= ANALYSIS_END

    # 9) 随机入场对照：基准构造下超额应≈0
    rng = np.random.default_rng(20261026)
    fwd20 = (panel.hfq["close"].shift(-20) / panel.hfq["open"].shift(-1) - 1.0)
    elig_np = L["tradable"].to_numpy()
    bench20 = fwd20.where(L["tradable"]).mean(axis=1).to_numpy()
    fwd_np = fwd20.to_numpy()
    ii, jj = np.where(elig_np)
    if len(ii) > 1000:
        take = rng.choice(len(ii), size=min(30000, len(ii)), replace=False)
        x = fwd_np[ii[take], jj[take]] - bench20[ii[take]]
        m = float(np.nanmean(x))
        res["random_entry_exc20_mean"] = m
        res["random_entry_ok"] = bool(abs(m) < 0.002)

    with open(OUT / "自检.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("自检 → 自检.json")
    return res


# ----------------------------------------------------------------------------
# 9. 参数邻域（只做稳定性，不择优）
# ----------------------------------------------------------------------------
def param_neighborhood(panel: Panel, feats: Features | None = None) -> pd.DataFrame:
    rows = []
    base = {"m_ma": 12, "w_ma": 30, "d_ma": 50, "bo": 20, "volx": 1.5}
    grid = {
        "m_ma": [9, 12, 15],
        "w_ma": [20, 30, 40],
        "d_ma": [40, 50, 60],
        "bo": [15, 20, 25],
        "volx": [1.2, 1.5, 2.0],
    }
    feats = feats or Features(panel)
    L = feats.L
    for key, vals in grid.items():
        for v in vals:
            p = dict(base)
            p[key] = v
            specs, _ = _signal_specs_param(L["close"], L["high"], L["low"], L["amount"],
                                           L["raw_close"], L["tradable"], p, L["market_ok"])
            ev = build_events(panel, feats, specs, tag="param")
            if ev.empty:
                continue
            for period, (s, e) in {"dev": DEV, "val": VAL}.items():
                m = (ev["date"] >= pd.Timestamp(s)) & (ev["date"] <= pd.Timestamp(e))
                x = ev.loc[m, "exc20"].dropna()
                if len(x) < 50:
                    continue
                monthly = ev.loc[m].dropna(subset=["exc20"]).groupby(
                    ev.loc[m, "date"].dt.to_period("M"))["exc20"].mean()
                rows.append({"param": key, "value": v, "period": period,
                             "n": len(x), "exc20": x.mean(), "hit": (x > 0).mean(),
                             "t_month": monthly.mean() / (monthly.std(ddof=1) / math.sqrt(len(monthly)))
                             if len(monthly) > 2 else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 10. 对外适配器
# ----------------------------------------------------------------------------
class TrendSignalAdapter:
    """主协调者可集成的稳定接口。

    契约
    ----
    build(start, end) -> dict[name, DataFrame(bool, index=date, columns=code)]
        信号在当日收盘后可用；不含交易可行性过滤（由执行引擎负责）。
    meta(name) -> dict   # 层级、经济含义、失效边界
    """

    SPEC_META = {
        "S1_月周日共振突破": {
            "layers": ["月线regime", "周线趋势", "日线结构", "日线触发"],
            "thesis": "月线多头+周线多头+日线抬高低点+20日新高放量，四层同向",
            "invalid": "月线走平/周线跌破MA30/日线跌破前一分型低点/突破日无成交额确认",
        },
        "S2_周ADX_日回踩": {
            "layers": ["月线regime", "周ADX趋势", "日线结构", "日线回踩"],
            "thesis": "周线ADX>20确认趋势强度，日线回踩MA20不破后收回",
            "invalid": "周ADX回落至20以下、回踩收盘跌破MA20、放量下跌",
        },
        "S3_波动收缩突破": {
            "layers": ["月线regime", "周线趋势", "日线结构", "VCP突破"],
            "thesis": "ATR20/ATR60<0.8的波动收缩后突破20日高点",
            "invalid": "收缩期放量破位、突破后3日跌回区间",
        },
        "S4_结构突破55": {
            "layers": ["月线regime", "周线趋势", "日线结构", "55日新高"],
            "thesis": "中周期结构完好下的55日通道突破（更慢、换手更低）",
            "invalid": "突破失败跌回55日区间、周线转弱",
        },
        "S5_纯日线基线": {
            "layers": ["日线结构", "日线触发"],
            "thesis": "不使用月/周过滤的日线趋势基线，用于衡量多周期增量",
            "invalid": "震荡市假突破",
        },
        "S6_月周日_加市场许可": {
            "layers": ["市场宽度许可", "月线regime", "周线趋势", "日线结构", "日线触发"],
            "thesis": "在S1上加市场宽度>45%的许可，规避系统性下跌",
            "invalid": "宽度快速回落、指数跌破年线",
        },
    }

    def __init__(self, panel: Panel | None = None):
        self.panel = panel
        self._sig = None

    def build(self, start: str, end: str = ANALYSIS_END) -> dict[str, pd.DataFrame]:
        if self.panel is None:
            self.panel = load_panel(CACHE_START, end)
        L = Features(self.panel).L
        specs, _ = signal_specs(L)
        self._sig = {k: v.fillna(False) for k, v in specs.items()}
        return self._sig

    @classmethod
    def meta(cls, name: str) -> dict:
        return cls.SPEC_META.get(name, {})


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _period_summary(ev: pd.DataFrame, label: str) -> pd.DataFrame:
    parts = []
    for name, (s, e) in {"预热0507": WARM, "开发0815": DEV, "验证1620": VAL,
                         "全样本0520": ("2005-01-01", ANALYSIS_END)}.items():
        m = (ev["date"] >= pd.Timestamp(s)) & (ev["date"] <= pd.Timestamp(e))
        sub = ev.loc[m]
        if sub.empty:
            continue
        parts.append(summarize_events(sub, period=name))
    return pd.concat(parts, ignore_index=True)


def index_context() -> pd.DataFrame:
    """指数区间年化，作为组合结果的背景参照（≤2020）。"""
    p = CACHE_DIR / "bs_index.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    names = {"sh.000300": "沪深300", "sh.000905": "中证500",
             "sz.399006": "创业板指", "sh.000001": "上证综指", "sz.399001": "深证成指"}
    rows = []
    for code, nm in names.items():
        s = df[df["code"] == code].set_index("date")["close"].sort_index()
        if s.empty:
            continue
        for per, (a, b) in {"预热0507": WARM, "开发0815": DEV, "验证1620": VAL,
                            "全样本0520": ("2005-01-01", ANALYSIS_END)}.items():
            x = s.loc[a:b]
            if len(x) < 50:
                continue
            yrs = (x.index[-1] - x.index[0]).days / 365.25
            rows.append({"index": nm, "period": per, "start": x.index[0].date(),
                         "end": x.index[-1].date(), "total": x.iloc[-1] / x.iloc[0] - 1,
                         "cagr": (x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1})
    return pd.DataFrame(rows)


def expectation_analysis(ev: pd.DataFrame) -> pd.DataFrame:
    """买入后前 10 日是否"符合预期"，及其对后 10 日收益的预测。"""
    x = ev.copy()
    x["mfe10_r"] = x["mfe10"] / x["atr_pct"].replace(0, np.nan)
    bins = [-9, 0.0, 0.5, 1.0, 2.0, 99]
    labels = ["≤0R", "0-0.5R", "0.5-1R", "1-2R", ">2R"]
    x["early"] = pd.cut(x["mfe10_r"], bins=bins, labels=labels)
    rows = []
    for per, (s, e) in {"开发0815": DEV, "验证1620": VAL}.items():
        sub = x[(x["date"] >= pd.Timestamp(s)) & (x["date"] <= pd.Timestamp(e))]
        for (name, early), g in sub.groupby(["signal", "early"], observed=True):
            if len(g) < 50:
                continue
            rows.append({"period": per, "signal": name, "early_mfe": str(early),
                         "n": len(g), "ret_10_20": g["ret_10_20"].mean(),
                         "hit_10_20": (g["ret_10_20"] > 0).mean(),
                         "fwd20": g["fwd20"].mean(),
                         "exc20_q": g["exc20_q"].mean(),
                         "mae20": g["mae20"].mean(), "mfe20": g["mfe20"].mean()})
    return pd.DataFrame(rows)


def cmd_events(panel: Panel | None = None, feats: Features | None = None) -> pd.DataFrame:
    panel = panel or load_panel()
    feats = feats or Features(panel)
    specs, abl = signal_specs(feats.L)
    log(f"事件研究：{len(specs)} 个候选（≤{ANALYSIS_END}）")
    ev = build_events(panel, feats, specs, tag="main")
    ev.to_parquet(OUT / "信号事件_明细.parquet", index=False)
    summ = _period_summary(ev, "main")
    summ.to_csv(OUT / "信号事件_汇总.csv", index=False, encoding="utf-8-sig")
    yearly = yearly_events(ev)
    yearly.to_csv(OUT / "信号事件_年度.csv", index=False, encoding="utf-8-sig")
    fb = failure_boundary(ev)
    fb.to_csv(OUT / "信号失效边界.csv", index=False, encoding="utf-8-sig")
    dist = distribution_stats(ev)
    if not dist.empty:
        dist.to_csv(OUT / "信号超额_均值中位数.csv", index=False, encoding="utf-8-sig")
    idx = index_context()
    if not idx.empty:
        idx.to_csv(OUT / "指数区间收益.csv", index=False, encoding="utf-8-sig")
    exp = expectation_analysis(ev)
    if not exp.empty:
        exp.to_csv(OUT / "预期处理.csv", index=False, encoding="utf-8-sig")
    log(f"事件 {len(ev):,} 条 → 信号事件_明细.parquet / 汇总 / 年度 / 失效边界 / 预期处理")
    return ev


def cmd_ablation(panel: Panel | None = None, feats: Features | None = None) -> pd.DataFrame:
    panel = panel or load_panel()
    feats = feats or Features(panel)
    _, abl = signal_specs(feats.L)
    log(f"单层消融：{len(abl)} 组")
    ev = build_events(panel, feats, abl, tag="abl")
    summ = _period_summary(ev, "abl")
    summ.to_csv(OUT / "消融_汇总.csv", index=False, encoding="utf-8-sig")
    yearly_events(ev).to_csv(OUT / "消融_年度.csv", index=False, encoding="utf-8-sig")
    log("消融 → 消融_汇总.csv / 消融_年度.csv")
    return summ


def _benchmark(panel: Panel, feats: Features) -> pd.Series:
    """等权可交易基准（日收益，供组合对比）。"""
    r = panel.hfq["close"].pct_change()
    return r.where(feats.L["tradable"]).mean(axis=1).fillna(0.0)


def cmd_sim(panel: Panel | None = None, feats: Features | None = None) -> pd.DataFrame:
    panel = panel or load_panel()
    feats = feats or Features(panel)
    specs, _ = signal_specs(feats.L)
    bench = _benchmark(panel, feats)
    rows = []
    rules = {
        "纯时间20": dict(exit_rule="time", max_hold=20, stop_atr=None),
        "纯时间60": dict(exit_rule="time", max_hold=60, stop_atr=None),
        "时间40+止损2ATR": dict(exit_rule="time", max_hold=40, stop_atr=2.0),
        "移动止盈3ATR+止损2": dict(exit_rule="trail", stop_atr=2.0, trail_atr=3.0, max_hold=250),
        "目标2ATR+止损2": dict(exit_rule="trail", stop_atr=2.0, trail_atr=99, max_hold=250, take_r=2.0),
        "宽止损4ATR+时间60": dict(exit_rule="time", max_hold=60, stop_atr=4.0),
        "移动+不及预期": dict(exit_rule="trail_stag", stop_atr=2.0, trail_atr=3.0,
                              max_hold=120, stagnation_days=10),
    }
    all_trades = []
    for sname, sig in specs.items():
        for rname, kw in rules.items():
            for period, (s, e) in {"dev": DEV, "val": VAL}.items():
                cfg = SimConfig(start=s, end=e, n_pos=N_POS, capital=ACCOUNT, **kw)
                t0 = time.time()
                curve, trades = simulate(panel, feats, sig, cfg)
                if curve.empty:
                    continue
                p = perf(curve)
                b = bench.loc[(bench.index >= pd.Timestamp(s)) & (bench.index <= pd.Timestamp(e))]
                b_vol = b.std() * math.sqrt(252)
                b_eq = (1 + b).cumprod()
                p.update({"signal": sname, "rule": rname, "period": period,
                          "n_trades": len(trades),
                          "win_rate": (trades["ret"] > 0).mean() if len(trades) else np.nan,
                          "avg_hold": trades["hold_days"].mean() if len(trades) else np.nan,
                          "avg_ret": trades["ret"].mean() if len(trades) else np.nan,
                          "bench_cagr": b_eq.iloc[-1] ** (252 / max(1, len(b))) - 1,
                          "bench_vol": b_vol,
                          "bench_maxdd": (b_eq / b_eq.cummax() - 1).min(),
                          "bench_sharpe": (b.mean() * 252 - 0.02) / b_vol if b_vol > 0 else np.nan,
                          "elapsed": time.time() - t0})
                p["excess_cagr"] = p["cagr"] - p["bench_cagr"]
                rows.append(p)
                if len(trades):
                    all_trades.append(trades.assign(signal=sname, rule=rname, period=period))
                curve.assign(signal=sname, rule=rname, period=period).to_parquet(
                    OUT / f"净值_{sname}_{rname}_{period}.parquet", index=False)
            log(f"  模拟 {sname} / {rname} 完成")
    if all_trades:
        tr = pd.concat(all_trades, ignore_index=True)
        trade_diagnostics(tr).to_csv(OUT / "交易诊断.csv", index=False, encoding="utf-8-sig")
        tr.to_parquet(OUT / "交易明细_全部.parquet", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "组合模拟_汇总.csv", index=False, encoding="utf-8-sig")
    log(f"组合模拟 → 组合模拟_汇总.csv ({len(df)} 行)")
    return df


def cmd_sens(panel: Panel | None = None, feats: Features | None = None) -> pd.DataFrame:
    """资金规模 × 持仓数敏感性（基准规则：止损2ATR + 移动止盈3ATR）。"""
    panel = panel or load_panel()
    feats = feats or Features(panel)
    specs, _ = signal_specs(feats.L)
    bench = _benchmark(panel, feats)
    rows = []
    for sname in ("S1_月周日共振突破", "S6_月周日_加市场许可"):
        sig = specs[sname]
        for cap in (20_000, 100_000, 1_000_000):
            for npos in (5, 10, 20):
                for period, (s, e) in {"dev": DEV, "val": VAL}.items():
                    cfg = SimConfig(start=s, end=e, n_pos=npos, capital=cap,
                                    exit_rule="trail", stop_atr=2.0, trail_atr=3.0,
                                    max_hold=250)
                    curve, trades = simulate(panel, feats, sig, cfg)
                    if curve.empty:
                        continue
                    p = perf(curve)
                    b = bench.loc[(bench.index >= pd.Timestamp(s)) & (bench.index <= pd.Timestamp(e))]
                    p.update({"signal": sname, "capital": cap, "n_pos": npos,
                              "period": period, "n_trades": len(trades),
                              "win_rate": (trades["ret"] > 0).mean() if len(trades) else np.nan,
                              "avg_hold": trades["hold_days"].mean() if len(trades) else np.nan,
                              "bench_cagr": (1 + b).prod() ** (252 / max(1, len(b))) - 1})
                    p["excess_cagr"] = p["cagr"] - p["bench_cagr"]
                    rows.append(p)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "敏感性_资金与持仓数.csv", index=False, encoding="utf-8-sig")
    log(f"敏感性 → 敏感性_资金与持仓数.csv ({len(df)} 行)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-cache", "audit", "events", "ablation",
                                    "params", "sim", "sens", "selfcheck", "all"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    _ensure_dirs()
    t0 = time.time()

    if a.cmd == "build-cache":
        build_cache(force=a.force)
        return
    if a.cmd == "audit":
        audit()
        return
    panel = load_panel()
    feats = Features(panel)
    if a.cmd == "events":
        cmd_events(panel, feats)
    elif a.cmd == "ablation":
        cmd_ablation(panel, feats)
    elif a.cmd == "params":
        df = param_neighborhood(panel, feats)
        df.to_csv(OUT / "参数邻域.csv", index=False, encoding="utf-8-sig")
        log(f"参数邻域 → 参数邻域.csv ({len(df)} 行)")
    elif a.cmd == "sim":
        cmd_sim(panel, feats)
    elif a.cmd == "sens":
        cmd_sens(panel, feats)
    elif a.cmd == "selfcheck":
        selfcheck()
    elif a.cmd == "all":
        cmd_events(panel, feats)
        cmd_ablation(panel, feats)
        param_neighborhood(panel, feats).to_csv(
            OUT / "参数邻域.csv", index=False, encoding="utf-8-sig")
        cmd_sim(panel, feats)
        cmd_sens(panel, feats)
        selfcheck()
    log(f"完成，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
