"""多周期.py — 日/周/月三周期对齐引擎与多周期自检

对齐规则（因果）
----------------
t 日收盘时可见的周线 = period_end <= t 的**最后一个已结束周**；月线同理。
周五收盘时当周周线即已结束（available_at = 当日收盘），因此 period_end == t 时可用，
但绝不使用 period_end > t 的周期。未走完的周/月不进入面板。

输出
----
`MultiTimeframe.build()` 返回逐日面板：日线特征沿用核心特征列名，周/月特征加 `w_`/`m_` 前缀，
每行附 `w_period_end`、`m_period_end`、`w_age_sessions`、`m_age_sessions`，
并生成 `align_score`（+3..-3）、`resonance`、`phase_*`、`screen_pass`。

`post_entry_path()` 只提供入场后的**状态描述**（MFE/MAE/回吐/结构是否完好/趋势是否翻转），
不计算策略收益，供策略层决定“符合预期持有 / 不符合预期离场”。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("core_features", _HERE / "核心特征.py")
cf = importlib.util.module_from_spec(_spec)
sys.modules["core_features"] = cf
_spec.loader.exec_module(cf)

FREEZE = cf.FREEZE
ROOT = _HERE.parent
EVIDENCE = ROOT / "04_证据"

W_NUM = ["close", "ma20", "ma50", "ma200", "atr_pct", "atr_ratio", "adx14", "rsi14",
         "don_pos", "don_high_prev", "bb_width_rank", "slope20_bp", "slope60_bp",
         "stop_pct", "stop_atr", "retrace", "advance", "struct_score", "vol_ratio5_20",
         "amt_ratio20", "atr14", "n_days", "coverage"]
W_CAT = ["trend", "h_label", "l_label"]
W_FLAG = ["pullback", "restart", "struct_broken", "don_break_up", "up_leg"]


@dataclass(frozen=True)
class ScreenConfig:
    """示例筛选门槛：业内常规值，**未经收益优化**。策略层可整体替换。"""
    stop_min: float = 0.02
    stop_max: float = 0.12
    min_amt_ma20: float = 5e7
    require_w_trend: str = "up"
    allow_d_trend: tuple = ("up", "transition")


def _align(dates, pf: pd.DataFrame, suffix: str, nums, cats, flags):
    pe = pd.to_datetime(pf["date"]).to_numpy(dtype="datetime64[D]")
    pos = np.searchsorted(pe, dates, side="right") - 1
    valid = pos >= 0
    idx = np.clip(pos, 0, None)
    out = {}
    for c in nums:
        v = pd.to_numeric(pf[c], errors="coerce").to_numpy(dtype=float)
        out[f"{suffix}_{c}"] = np.where(valid, v[idx], np.nan)
    for c in cats + flags:
        v = pf[c].to_numpy(dtype=object)
        out[f"{suffix}_{c}"] = np.where(valid, v[idx], None)
    w_end = np.full(len(dates), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    w_end[valid] = pe[idx][valid]
    out[f"{suffix}_period_end"] = w_end
    dpos = np.searchsorted(dates, w_end, side="right") - 1
    age = np.arange(len(dates)) - dpos
    out[f"{suffix}_age_sessions"] = np.where(valid & (dpos >= 0), age.astype(float), np.nan)
    return out


def _phase(trend, don_pos):
    t = np.asarray(trend, dtype=object)
    dp = np.asarray(don_pos, dtype=float)
    out = np.full(len(t), "unknown", dtype=object)
    out[(t == "up") & (dp >= 0.7)] = "up_strong"
    out[(t == "up") & (dp < 0.7)] = "up_pullback"
    out[(t == "down") & (dp <= 0.3)] = "down_strong"
    out[(t == "down") & (dp > 0.3)] = "down_bounce"
    out[(t == "transition")] = "transition"
    out[(t == "none")] = "none"
    out[~np.isfinite(dp)] = "unknown"
    return out


class MultiTimeframe:
    def __init__(self, source=None, cal=None, cfg=cf.DEFAULT_CFG, screen=ScreenConfig()):
        self.src = source or cf.BaostockSource()
        self.cal = cal or cf.Calendar.from_baostock()
        self.cfg = cfg
        self.screen = screen

    def build(self, symbol, start=None, end=FREEZE, market=False, breadth=None, index_ctx=None) -> pd.DataFrame:
        daily = self.src.load(symbol, adjust="hfq", start=start, end=end, with_raw=True)
        if daily is None or len(daily) < 30:
            return pd.DataFrame()
        bd = cf.Bars(daily, code=cf.to_ths_code(symbol))
        fd = cf.feature_matrix(bd, self.cfg)
        bw = cf.aggregate(bd, "W", self.cal, self.cfg)
        bm = cf.aggregate(bd, "M", self.cal, self.cfg)
        fw = cf.feature_matrix(bw, self.cfg, freq="W") if len(bw) else pd.DataFrame()
        fm = cf.feature_matrix(bm, self.cfg, freq="M") if len(bm) else pd.DataFrame()

        panel = fd.copy()
        panel["open"] = bd.open
        panel["high"] = bd.high
        panel["low"] = bd.low
        panel["volume"] = bd.volume
        dates = panel["date"].to_numpy(dtype="datetime64[D]")
        if len(fw):
            panel = panel.assign(**_align(dates, fw, "w", W_NUM, W_CAT, W_FLAG))
        else:
            for c in W_NUM:
                panel[f"w_{c}"] = np.nan
            for c in W_CAT + W_FLAG:
                panel[f"w_{c}"] = None
            panel["w_period_end"] = np.datetime64("NaT")
            panel["w_age_sessions"] = np.nan
        if len(fm):
            panel = panel.assign(**_align(dates, fm, "m", W_NUM, W_CAT, W_FLAG))
        else:
            for c in W_NUM:
                panel[f"m_{c}"] = np.nan
            for c in W_CAT + W_FLAG:
                panel[f"m_{c}"] = None
            panel["m_period_end"] = np.datetime64("NaT")
            panel["m_age_sessions"] = np.nan

        d_up = (panel["trend"] == "up").astype(int)
        d_dn = (panel["trend"] == "down").astype(int)
        w_up = (panel["w_trend"] == "up").astype(int)
        w_dn = (panel["w_trend"] == "down").astype(int)
        m_up = (panel["m_trend"] == "up").astype(int)
        m_dn = (panel["m_trend"] == "down").astype(int)
        panel["align_score"] = d_up + w_up + m_up - d_dn - w_dn - m_dn
        panel["resonance"] = np.select(
            [panel["align_score"] >= 3, panel["align_score"] <= -3], ["full_up", "full_down"], "mixed")
        panel.loc[panel["w_trend"].isna() | panel["m_trend"].isna(), "resonance"] = "unknown"
        panel["phase_d"] = _phase(panel["trend"], panel["don_pos"])
        panel["phase_w"] = _phase(panel["w_trend"], panel["w_don_pos"])
        panel["phase_m"] = _phase(panel["m_trend"], panel["m_don_pos"])

        if market:
            panel = self.attach_market(panel, breadth=breadth, index_ctx=index_ctx)
        panel["screen_pass"] = self.screen_pass(panel)
        self.validate(panel)
        return panel

    def screen_pass(self, panel: pd.DataFrame) -> pd.Series:
        s = self.screen
        ok = ((panel["w_trend"] == s.require_w_trend)
              & panel["trend"].isin(s.allow_d_trend)
              & (~panel["struct_broken"].astype(bool))
              & panel["up_leg"].astype(bool)
              & panel["stop_pct"].between(s.stop_min, s.stop_max)
              & (panel["amt_ma20"] >= s.min_amt_ma20)
              & panel["traded"].astype(bool)
              & (~panel["one_word_up"].astype(bool))
              & (~panel["suspended"].astype(bool)))
        return ok.fillna(False)

    def validate(self, panel: pd.DataFrame):
        if panel.empty:
            return True
        d = panel["date"].to_numpy(dtype="datetime64[D]")
        for tf in ("w", "m"):
            pe = panel[f"{tf}_period_end"].to_numpy(dtype="datetime64[D]")
            ok = np.isnat(pe) | (pe <= d)
            if not ok.all():
                bad = int((~ok).sum())
                raise AssertionError(f"{tf} 周期在 {bad} 行出现 period_end > date（未来函数）")
        if (panel["available_at"].to_numpy(dtype="datetime64[D]") > d).any():
            raise AssertionError("available_at > date")
        return True

    def attach_market(self, panel, breadth=None, index_ctx=None):
        if breadth is None:
            breadth = breadth_duckdb_cached()
        if breadth is not None and len(breadth):
            b = breadth.copy()
            b["date"] = pd.to_datetime(b["date"])
            panel = panel.merge(
                b[["date", "n_stocks", "pct_above_ma20", "pct_above_ma50", "pct_above_ma200",
                   "pct_new_high252", "pct_new_low252", "total_amount", "breadth_ma20_z"]],
                on="date", how="left")
            panel = panel.rename(columns={
                "pct_above_ma20": "mkt_pct_above_ma20", "pct_above_ma50": "mkt_pct_above_ma50",
                "pct_above_ma200": "mkt_pct_above_ma200", "pct_new_high252": "mkt_pct_new_high252",
                "pct_new_low252": "mkt_pct_new_low252", "total_amount": "mkt_total_amount",
                "n_stocks": "mkt_n_stocks", "breadth_ma20_z": "mkt_breadth_z"})
        else:
            for c in ("mkt_pct_above_ma20", "mkt_pct_above_ma50", "mkt_pct_above_ma200",
                      "mkt_pct_new_high252", "mkt_pct_new_low252", "mkt_total_amount",
                      "mkt_n_stocks", "mkt_breadth_z"):
                panel[c] = np.nan
        if index_ctx is None:
            index_ctx = index_cached()
        if index_ctx is not None and len(index_ctx):
            ix = index_ctx[index_ctx["idx_code"] == "sh.000300"][["date", "close", "ma200", "atr_pct", "dd252"]]
            ix = ix.rename(columns={"close": "idx_close", "ma200": "idx_ma200",
                                    "atr_pct": "idx_atr_pct", "dd252": "idx_dd252"})
            panel = panel.merge(ix, on="date", how="left")
        else:
            for c in ("idx_close", "idx_ma200", "idx_atr_pct", "idx_dd252"):
                panel[c] = np.nan
        cond_on = (panel["idx_close"] > panel["idx_ma200"]) & (panel["mkt_pct_above_ma50"] >= 0.5)
        cond_off = (panel["idx_close"] < panel["idx_ma200"]) & (panel["mkt_pct_above_ma50"] < 0.5)
        panel["mkt_regime"] = np.select([cond_on, cond_off], ["risk_on", "risk_off"], "neutral")
        panel.loc[panel["idx_ma200"].isna(), "mkt_regime"] = "unknown"
        return panel

    def build_panel(self, symbols, start=None, end=FREEZE, market=False, columns=None,
                    progress_every=200, errors=None) -> pd.DataFrame:
        """批量构建长表（symbol + 逐日面板）。columns 可只保留需要的列以控制内存。"""
        frames = []
        for i, s in enumerate(symbols, 1):
            try:
                p = self.build(s, start=start, end=end, market=market)
            except Exception as e:
                if errors is not None:
                    errors.append({"symbol": s, "error": repr(e)})
                continue
            if not len(p):
                continue
            p.insert(0, "symbol", s)
            if columns is not None:
                keep = ["symbol"] + [c for c in columns if c in p.columns]
                p = p[keep]
            frames.append(p)
            if progress_every and i % progress_every == 0:
                print(f"  built {i}/{len(symbols)}", file=sys.stderr)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def snapshot(panel: pd.DataFrame, i: int) -> dict:
        """单日决策快照（全部为第 i 行收盘时已知量）。"""
        r = panel.iloc[i]
        keys = ["date", "close", "trend", "w_trend", "m_trend", "align_score", "resonance",
                "phase_d", "phase_w", "phase_m", "struct_score", "w_struct_score",
                "retrace", "pullback", "restart", "struct_broken", "don_pos", "w_don_pos",
                "stop_pct", "stop_atr", "stop_px_adj", "price_scale", "atr_pct", "amt_ma20",
                "w_period_end", "m_period_end", "w_age_sessions", "m_age_sessions",
                "w_coverage", "screen_pass", "traded", "at_limit_up", "one_word_up", "suspended"]
        out = {}
        for k in keys:
            if k not in panel.columns:
                continue
            v = r[k]
            if isinstance(v, (pd.Timestamp, np.datetime64)):
                out[k] = None if pd.isna(v) else str(pd.Timestamp(v).date())
            elif isinstance(v, (np.bool_, bool)):
                out[k] = bool(v)
            elif isinstance(v, (np.floating, float)):
                out[k] = None if not np.isfinite(v) else float(v)
            elif isinstance(v, (np.integer, int)):
                out[k] = int(v)
            else:
                out[k] = None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else str(v)
        return out


_BREADTH = {}
_INDEX = {}


def breadth_duckdb_cached(start="2016-01-01"):
    key = ("duck", start)
    if key not in _BREADTH:
        try:
            _BREADTH[key] = cf.breadth_duckdb(start=start)
        except Exception as e:  # DuckDB 不可用时降级为无市场层
            _BREADTH[key] = pd.DataFrame()
            print(f"[warn] breadth_duckdb 失败: {e}", file=sys.stderr)
    return _BREADTH[key]


def index_cached(start="2016-01-01"):
    if "idx" not in _INDEX:
        try:
            _INDEX["idx"] = cf.index_context(start=start)
        except Exception as e:
            _INDEX["idx"] = pd.DataFrame()
            print(f"[warn] index_context 失败: {e}", file=sys.stderr)
    return _INDEX["idx"]


# ---------------------------------------------------------------- 入场后状态

def post_entry_path(panel: pd.DataFrame, i: int, max_bars: int | None = None,
                    exec_price: str = "next_open") -> dict:
    """信号在第 i 行收盘生成，默认第 i+1 行开盘成交。只描述路径，不算策略收益。"""
    if i + 1 >= len(panel):
        return {}
    if exec_price == "next_open":
        px = float(panel["open"].iloc[i + 1])
    else:
        px = float(panel["close"].iloc[i])
    if not np.isfinite(px) or px <= 0:
        return {}
    lo = i + 1
    hi = len(panel) if max_bars is None else min(len(panel), i + 1 + max_bars)
    sub = panel.iloc[lo:hi]
    if sub.empty:
        return {}
    high = sub["high"].to_numpy(dtype=float)
    low = sub["low"].to_numpy(dtype=float)
    close = sub["close"].to_numpy(dtype=float)
    stop0 = float(panel["stop_px_adj"].iloc[i]) if np.isfinite(panel["stop_px_adj"].iloc[i]) else np.nan
    stop_hit = np.flatnonzero(np.isfinite(stop0) & (low <= stop0))
    return {
        "entry_date": str(pd.Timestamp(panel["date"].iloc[i + 1]).date()),
        "entry_px": px,
        "bars_held": int(len(sub)),
        "mfe_pct": float(np.nanmax(high) / px - 1) if len(high) else np.nan,
        "mae_pct": float(np.nanmin(low) / px - 1) if len(low) else np.nan,
        "last_pct": float(close[-1] / px - 1),
        "giveback_from_mfe": float(close[-1] / np.nanmax(high) - 1),
        "stop0": stop0,
        "stop_hit": bool(len(stop_hit) > 0),
        "stop_hit_bars": int(stop_hit[0] + 1) if len(stop_hit) else None,
        "stop_now": float(sub["stop_px_adj"].iloc[-1]) if np.isfinite(sub["stop_px_adj"].iloc[-1]) else np.nan,
        "struct_broken_now": bool(sub["struct_broken"].iloc[-1]),
        "trend_d_now": str(sub["trend"].iloc[-1]),
        "trend_w_now": str(sub["w_trend"].iloc[-1]),
        "trend_m_now": str(sub["m_trend"].iloc[-1]),
        "don_break_dn_count": int(sub["don_break_dn"].fillna(0).sum()),
        "max_retrace_in_trade": float(np.nanmax(sub["retrace"].to_numpy(dtype=float)))
        if np.isfinite(sub["retrace"].to_numpy(dtype=float)).any() else np.nan,
        "w_flip_to_down": bool((sub["w_trend"] == "down").any()),
        "m_flip_to_down": bool((sub["m_trend"] == "down").any()),
    }


# ---------------------------------------------------------------- 自检

def _sample_symbols(n=24, seed=3, min_ipo="2006-01-01", era_end="2020-12-31"):
    src = cf.BaostockSource()
    s = src.symbols()
    s = s[s["ipo_date"] <= min_ipo]
    pool = s[(s["out_date"].isna()) | (s["out_date"] > era_end)]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return pool.iloc[idx]["code"].tolist()


def mp_checks(cfg=cf.DEFAULT_CFG, n_sym=12, seed=5) -> list:
    checks = []
    mt = MultiTimeframe(cfg=cfg)
    syms = _sample_symbols(n=n_sym, seed=seed)
    rng = np.random.default_rng(seed)

    panels = {}
    fail_build = []
    for s in syms:
        try:
            p = mt.build(s, start="2006-01-01")
            if len(p) > 300:
                panels[s] = p
        except Exception as e:
            fail_build.append({"symbol": s, "error": repr(e)})
    checks.append({
        "id": "mp_build", "name": "面板构建",
        "status": "pass" if len(panels) >= max(1, n_sym // 2) and not fail_build else "fail",
        "detail": f"尝试 {len(syms)} 只，成功 {len(panels)} 只；失败 {fail_build[:3]}",
        "evidence": {"ok": len(panels), "failed": len(fail_build),
                     "rows_total": int(sum(len(p) for p in panels.values()))},
    })

    # 0 末端周期完整性：最后一行只能用已结束的周/月，且不会把未走完的周期当成完整周期
    viol, detail = 0, []
    for s, p in panels.items():
        last_d = pd.Timestamp(p["date"].iloc[-1])
        wmax = pd.Timestamp(p["w_period_end"].dropna().max()) if p["w_period_end"].notna().any() else pd.NaT
        mmax = pd.Timestamp(p["m_period_end"].dropna().max()) if p["m_period_end"].notna().any() else pd.NaT
        if pd.isna(wmax) or wmax > last_d or (last_d - wmax).days > 7:
            viol += 1
        if pd.isna(mmax) or mmax > last_d or (last_d - mmax).days > 31:
            viol += 1
        detail.append({"symbol": s, "last": str(last_d.date()),
                       "w_end": str(wmax.date()) if pd.notna(wmax) else None,
                       "m_end": str(mmax.date()) if pd.notna(mmax) else None})
    # 直接检验聚合层：末周/末月不得是日历末端周期
    cal = mt.cal
    tail_bad = 0
    for s in list(panels)[:3]:
        daily = mt.src.load(s, adjust="hfq", start="2006-01-01")
        bd = cf.Bars(daily, code=cf.to_ths_code(s))
        for f, lim in (("W", 7), ("M", 31)):
            pb = cf.aggregate(bd, f, cal, cfg)
            if len(pb):
                pe = pd.Timestamp(pb.dates[-1])
                if (pd.Timestamp(bd.dates[-1]) - pe).days > lim:
                    tail_bad += 1
                if pe > pd.Timestamp(bd.dates[-1]):
                    tail_bad += 1
    checks.append({
        "id": "mp_no_partial_tail", "name": "末端未完成周/月被剔除",
        "status": "pass" if viol == 0 and tail_bad == 0 else "fail",
        "detail": f"面板末行的最近周/月不得晚于当日且不过期（检查 {len(panels)} 只，异常 {viol}）；"
                  f"聚合层末端周期异常 {tail_bad}",
        "evidence": {"violations": viol, "aggregate_tail_bad": tail_bad, "examples": detail[:5],
                     "calendar_tail": str(pd.Timestamp(cal.dates[-1]).date())},
    })

    # 1 周期边界：w/m period_end <= date，且是最新可用周期
    viol = 0
    for s, p in panels.items():
        d = p["date"].to_numpy(dtype="datetime64[D]")
        for tf in ("w", "m"):
            pe = p[f"{tf}_period_end"].to_numpy(dtype="datetime64[D]")
            viol += int(((~np.isnat(pe)) & (pe > d)).sum())
    checks.append({
        "id": "mp_period_not_future", "name": "周/月周期不越界",
        "status": "pass" if viol == 0 else "fail",
        "detail": f"{len(panels)} 只股票逐行检查 w/m period_end <= date，越界 {viol} 行",
        "evidence": {"violations": viol},
    })

    # 2 最新可用周期：w_period_end 应等于 <= date 的最大周结束日
    bad = 0
    for s, p in panels.items():
        wends = np.sort(np.unique(p["w_period_end"].dropna().to_numpy(dtype="datetime64[D]")))
        d = p["date"].to_numpy(dtype="datetime64[D]")
        expect = wends[np.searchsorted(wends, d, side="right") - 1]
        got = p["w_period_end"].to_numpy(dtype="datetime64[D]")
        m = ~np.isnat(got)
        bad += int((got[m] != expect[m]).sum())
    checks.append({
        "id": "mp_latest_period", "name": "对齐到最近已结束周期",
        "status": "pass" if bad == 0 else "fail",
        "detail": f"逐行比对 w_period_end 与 searchsorted 期望值，不一致 {bad} 行",
        "evidence": {"mismatch": bad},
    })

    # 3 周线 OHLC 与手工 groupby 一致
    mism, ncmp = 0, 0
    for s in list(panels)[:6]:
        daily = cf.BaostockSource().load(s, adjust="hfq", start="2006-01-01")
        bd = cf.Bars(daily, code=cf.to_ths_code(s))
        dk = pd.Series(bd.dates).dt.isocalendar()
        key = (dk["year"].astype(int) * 100 + dk["week"].astype(int)).to_numpy()
        df = pd.DataFrame({"key": key, "o": bd.open, "h": bd.high, "l": bd.low, "c": bd.close,
                           "v": np.where(bd.traded, bd.volume, 0.0), "date": bd.dates})
        man = df.groupby("key").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                                    c=("c", "last"), v=("v", "sum")).reset_index()
        fw = cf.aggregate(bd, "W", mt.cal, cfg).meta
        got = fw[["date", "open", "high", "low", "close", "volume"]].copy()
        got["key"] = (pd.DatetimeIndex(got["date"]).isocalendar()["year"].astype(int) * 100
                      + pd.DatetimeIndex(got["date"]).isocalendar()["week"].astype(int)).to_numpy()
        j = got.merge(man, on="key", how="inner")
        ncmp += len(j)
        for a, b in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("volume", "v")):
            mism += int((np.abs(j[a] - j[b]) > 1e-9).sum())
    checks.append({
        "id": "mp_week_bar_manual", "name": "周线 OHLCV 与手工聚合一致",
        "status": "pass" if mism == 0 and ncmp > 0 else "fail",
        "detail": f"比对 {ncmp} 个周 bar 的 O/H/L/C/V，不一致 {mism} 项",
        "evidence": {"bars": ncmp, "mismatch": mism},
    })

    # 4 多周期截断重算一致
    cols = ["w_trend", "w_struct_score", "w_stop_pct", "w_don_pos", "w_ma20", "w_atr_pct",
            "m_trend", "m_struct_score", "m_stop_pct", "m_don_pos", "m_ma20",
            "w_period_end", "m_period_end", "align_score", "resonance"]
    maxdiff, fails, cases = 0.0, 0, 0
    for s in list(panels)[:6]:
        p = panels[s]
        for _ in range(2):
            j = int(rng.integers(300, len(p) - 2))
            cut = p["date"].iloc[j]
            part = mt.build(s, start="2006-01-01", end=cut)
            if part.empty:
                fails += 1
                continue
            row_full = p.iloc[j]
            row_part = part.iloc[-1]
            cases += 1
            if pd.Timestamp(row_part["date"]) != pd.Timestamp(cut):
                fails += 1
                continue
            for c in cols:
                a, b = row_full[c], row_part[c]
                if pd.isna(a) and pd.isna(b):
                    continue
                if isinstance(a, (pd.Timestamp, np.datetime64)) or isinstance(b, (pd.Timestamp, np.datetime64)):
                    if pd.Timestamp(a) != pd.Timestamp(b):
                        fails += 1
                    continue
                if isinstance(a, str) or isinstance(b, str):
                    if str(a) != str(b):
                        fails += 1
                    continue
                fa, fb = float(a), float(b)
                if np.isnan(fa) != np.isnan(fb):
                    fails += 1
                elif np.isfinite(fa) and abs(fa - fb) > 1e-9:
                    fails += 1
                    maxdiff = max(maxdiff, abs(fa - fb))
    checks.append({
        "id": "mp_truncation", "name": "多周期面板截断重算一致",
        "status": "pass" if fails == 0 and cases > 0 else "fail",
        "detail": f"{cases} 个随机截断日，比较 {len(cols)} 个日/周/月列；失败 {fails}，max|diff|={maxdiff:.3e}",
        "evidence": {"cases": cases, "failures": fails, "max_abs_diff": maxdiff},
    })

    # 5 共振一致性
    bad = 0
    for s, p in panels.items():
        sc = ((p["trend"] == "up").astype(int) + (p["w_trend"] == "up").astype(int)
              + (p["m_trend"] == "up").astype(int) - (p["trend"] == "down").astype(int)
              - (p["w_trend"] == "down").astype(int) - (p["m_trend"] == "down").astype(int))
        bad += int((sc != p["align_score"]).sum())
        e = np.select([p["align_score"] >= 3, p["align_score"] <= -3], ["full_up", "full_down"], "mixed")
        e = pd.Series(e)
        e[p["w_trend"].isna().to_numpy() | p["m_trend"].isna().to_numpy()] = "unknown"
        bad += int((e != p["resonance"].reset_index(drop=True)).sum())
    checks.append({
        "id": "mp_resonance", "name": "日/周/月共振列自洽",
        "status": "pass" if bad == 0 else "fail",
        "detail": f"align_score 与 resonance 由分项重算，不一致 {bad} 行",
        "evidence": {"mismatch": bad},
    })

    # 6 入场执行价 = 次日开盘，且不受入场后数据影响
    viol, tested = 0, 0
    for s, p in list(panels.items())[:6]:
        cand = np.flatnonzero(p["screen_pass"].to_numpy()[:len(p) - 5])
        if len(cand) == 0:
            continue
        i = int(cand[len(cand) // 2])
        path = post_entry_path(p, i, max_bars=10)
        if not path:
            continue
        tested += 1
        if abs(path["entry_px"] - float(p["open"].iloc[i + 1])) > 1e-9:
            viol += 1
        p2 = p.copy()
        p2.loc[i + 2:, ["open", "high", "low", "close"]] *= 3.0
        path2 = post_entry_path(p2, i, max_bars=1)
        if abs(path2["entry_px"] - path["entry_px"]) > 1e-9:
            viol += 1
    checks.append({
        "id": "mp_t_plus_1", "name": "信号收盘生成、次日开盘成交",
        "status": "pass" if viol == 0 and tested > 0 else "fail",
        "detail": f"{tested} 个入场点：entry_px 必须等于次日开盘，且不随后续数据变化；违反 {viol}",
        "evidence": {"tested": tested, "violations": viol},
    })

    # 7 入场后路径描述的一致性（截断到 10 根 vs 全量前 10 根）
    bad, tested = 0, 0
    for s, p in list(panels.items())[:6]:
        cand = np.flatnonzero(p["screen_pass"].to_numpy()[:len(p) - 12])
        if len(cand) == 0:
            continue
        i = int(cand[len(cand) // 2])
        a = post_entry_path(p, i, max_bars=10)
        cut = p["date"].iloc[i + 10]
        pp = p[p["date"] <= cut]
        b = post_entry_path(pp, i, max_bars=10)
        if not a or not b:
            continue
        tested += 1
        for k in ("bars_held", "mfe_pct", "mae_pct", "last_pct", "stop_hit"):
            if a[k] != b[k] and not (isinstance(a[k], float) and abs(a[k] - b[k]) < 1e-9):
                bad += 1
    checks.append({
        "id": "mp_post_entry_causal", "name": "入场后路径描述因果一致",
        "status": "pass" if bad == 0 and tested > 0 else "fail",
        "detail": f"{tested} 个入场点，截断面板与全量面板的 MFE/MAE/持有根数/止损触发一致；不一致 {bad}",
        "evidence": {"tested": tested, "mismatch": bad},
    })

    # 8 市场宽度因果性（DuckDB 全量 vs 截止到 t）
    bfull = None
    try:
        bfull = cf.breadth_duckdb(start="2016-01-01")
    except Exception:
        pass
    if bfull is not None and len(bfull):
        sample_dates = bfull["date"].iloc[[-1, len(bfull) // 2, len(bfull) // 3]].tolist()
        worst, fails = 0.0, 0
        for d in sample_dates:
            bt = cf.breadth_duckdb(start="2016-01-01", end=d)
            a = bfull[bfull["date"] == d].iloc[0]
            bb = bt[bt["date"] == d].iloc[0]
            for c in ("n_stocks", "pct_above_ma20", "pct_above_ma50", "pct_above_ma200",
                      "pct_new_high252", "pct_new_low252", "total_amount"):
                x, y = float(a[c]), float(bb[c])
                tol = 1e-9 * max(1.0, abs(x))
                if abs(x - y) > tol:
                    fails += 1
                    worst = max(worst, abs(x - y))
        checks.append({
            "id": "mp_breadth_causal", "name": "市场宽度因果性",
            "status": "pass" if fails == 0 else "fail",
            "detail": f"{len(sample_dates)} 个日期：全量计算 vs 截止当日计算，不一致 {fails}，max|diff|={worst:.3e}",
            "evidence": {"dates": [str(pd.Timestamp(d).date()) for d in sample_dates],
                         "mismatch": fails, "max_abs_diff": worst},
        })
        # 幸存者偏差量化：审计日取 <=2020-12-31，避免在留出期做任何观察
        audit_pool = bfull[bfull["date"] <= pd.Timestamp("2020-12-31")]
        audit_date = audit_pool["date"].iloc[-1] if len(audit_pool) else sample_dates[0]
        try:
            bt = cf.breadth_baostock(start=str(pd.Timestamp(audit_date).date()),
                                     end=str(pd.Timestamp(audit_date).date()))
            if len(bt):
                du = bfull[bfull["date"] == audit_date].iloc[0]
                checks.append({
                    "id": "breadth_survivorship", "name": "宽度口径差（DuckDB 现存 vs baostock 含退市）",
                    "status": "pass",
                    "detail": f"{pd.Timestamp(audit_date).date()}: DuckDB n={int(du['n_stocks'])} "
                              f"ma200上方={du['pct_above_ma200']:.3f}；baostock n={int(bt['n_stocks'].iloc[0])} "
                              f"ma200上方={bt['pct_above_ma200'].iloc[0]:.3f}",
                    "evidence": {"date": str(pd.Timestamp(audit_date).date()),
                                 "duckdb": {k: float(du[k]) for k in ("n_stocks", "pct_above_ma50", "pct_above_ma200")},
                                 "baostock": {k: float(bt[k].iloc[0]) for k in ("n_stocks", "pct_above_ma50", "pct_above_ma200")}},
                })
        except Exception as e:
            checks.append({"id": "breadth_survivorship", "name": "宽度口径差", "status": "skip",
                           "detail": f"baostock 宽度失败：{e}", "evidence": {}})
    return checks


def descriptive_stats(panels: dict, cfg=cf.DEFAULT_CFG) -> dict:
    """结构性描述统计。2021 起只报数据覆盖，不看行为分布。"""
    dev_end = pd.Timestamp("2020-12-31")
    rows = []
    for s, p in panels.items():
        p = p[p["date"] <= dev_end]
        if len(p) < 300:
            continue
        rows.append({
            "symbol": s, "rows": len(p),
            "first": str(p["date"].iloc[0].date()), "last": str(p["date"].iloc[-1].date()),
            "w_bars": int(p["w_period_end"].notna().sum()),
            "m_bars": int(p["m_period_end"].notna().sum()),
            "share_up_d": float((p["trend"] == "up").mean()),
            "share_up_w": float((p["w_trend"] == "up").mean()),
            "share_up_m": float((p["m_trend"] == "up").mean()),
            "share_full_up": float((p["resonance"] == "full_up").mean()),
            "pullback_rate": float(p["pullback"].fillna(0).mean()),
            "restart_rate": float(p["restart"].fillna(0).mean()),
            "broken_rate": float(p["struct_broken"].fillna(0).mean()),
            "median_stop_pct": float(p["stop_pct"].median()),
            "median_atr_pct": float(p["atr_pct"].median()),
            "mean_coverage_w": float(p["w_coverage"].mean()),
            "screen_rate": float(p["screen_pass"].mean()),
        })
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    cov = [{"symbol": s, "rows": int(len(p)),
            "first": str(p["date"].iloc[0].date()), "last": str(p["date"].iloc[-1].date()),
            "nan_share_w": float(p["w_trend"].isna().mean())} for s, p in panels.items()]
    return {
        "window": ["2006-01-01", str(dev_end.date())],
        "note": "行为分布仅统计 <=2020-12-31（开发/验证期）；2021+ 只做覆盖审计",
        "n_symbols": int(len(df)),
        "rows_total": int(df["rows"].sum()),
        "median": {k: float(df[k].median()) for k in
                   ("share_up_d", "share_up_w", "share_up_m", "share_full_up", "pullback_rate",
                    "restart_rate", "broken_rate", "median_stop_pct", "median_atr_pct",
                    "mean_coverage_w", "screen_rate")},
        "p10_p90": {k: [float(df[k].quantile(0.1)), float(df[k].quantile(0.9))] for k in
                    ("median_stop_pct", "median_atr_pct", "share_full_up", "screen_rate")},
        "coverage_2021_plus": cov[:10],
    }


def coverage_audit(freeze=FREEZE) -> dict:
    """数据覆盖审计（只查行数/日期，不看收益）。2021+ 仅用于此处。"""
    out = {"freeze": str(pd.Timestamp(freeze).date())}
    con = sqlite3.connect(f"file:{cf.BAOSTOCK_DB}?mode=ro", uri=True)
    try:
        f = pd.Timestamp(freeze).strftime("%Y-%m-%d")
        out["baostock"] = {
            "stock_basic_type1": int(pd.read_sql_query(
                "select count(*) n from stock_basic where type=1", con)["n"].iloc[0]),
            "delisted_type1_by_freeze": int(pd.read_sql_query(
                "select count(*) n from stock_basic where type=1 and out_date is not null and out_date<=?",
                con, params=(f,))["n"].iloc[0]),
            "trade_days": int(pd.read_sql_query(
                "select count(*) n from trade_dates where is_trading_day=1 and calendar_date<=?",
                con, params=(f,))["n"].iloc[0]),
            "index_window": pd.read_sql_query(
                "select code, min(date) first, max(date) last, count(*) rows from index_kline_d "
                "where date<=? group by code order by code", con, params=(f,)).to_dict("records"),
        }
    finally:
        con.close()
    try:
        import duckdb
        d = duckdb.connect(str(cf.DUCKDB_PATH), read_only=True)
        r = d.execute("select count(distinct thscode) n, min(date) d0, max(date) d1, count(*) nrows "
                      "from raw_kline_daily where date<=?", [pd.Timestamp(freeze).date()]).fetchone()
        out["duckdb"] = {"symbols": int(r[0]), "first": str(r[1]), "last": str(r[2]), "rows": int(r[3])}
        d.close()
    except Exception as e:
        out["duckdb"] = {"error": repr(e)}
    out["note"] = ("2021-01-01 起仅做覆盖审计；本文件不含任何 2021+ 收益/行为分布统计。"
                   "日历读数取到数据源已有末端，仅用于确认周期是否结束。")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="多周期引擎自检")
    ap.add_argument("--out", default=str(EVIDENCE / "特征因果自检.json"))
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--symbols", type=int, default=12)
    ap.add_argument("--stats-symbols", type=int, default=40)
    args = ap.parse_args(argv)
    t0 = time.time()
    cal = cf.Calendar.from_baostock()
    feat = cf.feature_checks(real_cases=args.cases)
    mpc = mp_checks(n_sym=args.symbols)
    stats = {}
    try:
        mt = MultiTimeframe(cal=cal)
        panels = {}
        for s in _sample_symbols(n=args.stats_symbols, seed=11):
            p = mt.build(s, start="2006-01-01")
            if len(p):
                panels[s] = p
        stats = descriptive_stats(panels)
    except Exception as e:
        stats = {"error": repr(e)}

    checks = feat + mpc
    payload = {
        "meta": {
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "freeze": str(FREEZE.date()),
            "scope": "features + multi_timeframe",
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "calendar_sessions": int(len(cal)),
            "calendar_window": [str(pd.Timestamp(cal.dates[0]).date()), str(pd.Timestamp(cal.dates[-1]).date())],
            "files": {"核心特征.py": cf._sha(_HERE / "核心特征.py"),
                      "多周期.py": cf._sha(Path(__file__))},
            "config": asdict(cf.DEFAULT_CFG),
            "screen_config": asdict(ScreenConfig()),
            "runtime_sec": round(time.time() - t0, 2),
            "adjustment": "价格类后复权(baostock adjustflag=1)，量额换手原始口径；宽度用不复权收盘算均线",
            "sources": {"baostock": str(cf.BAOSTOCK_DB), "duckdb": str(cf.DUCKDB_PATH)},
        },
        "checks": checks,
        "summary": {"total": len(checks), "pass": sum(c["status"] == "pass" for c in checks),
                    "fail": sum(c["status"] == "fail" for c in checks),
                    "skip": sum(c["status"] == "skip" for c in checks)},
        "dataset_coverage": coverage_audit(),
        "descriptive_stats": stats,
        "limitations": [
            "DuckDB 宽度仅含现存代码，2016+ 有幸存者偏差；baostock 宽度含退市但需全表扫描（约 40s/次）。",
            "周/月线在停牌跨周期时只有实际交易日，n_days/coverage 已标注，未做任何回填。",
            "枢轴确认按可交易 bar 计 k 根；停牌期间不确认，恢复交易后继续计时。",
            "结构止损为百分比口径，落地委托价需乘 price_scale 转原始价。",
            "screen_pass 的门槛是常规值，未经收益优化，不作为结论。",
            "2021-01-01 起不产出任何收益或行为分布；descriptive_stats 窗口截止 2020-12-31。",
            "因果自检（截断重算、未来扰动）会机械地跑过 2021+ 行情以验证无前视，不产生任何策略结论。",
            "交易日历读到数据源末个交易日（2026-10-28），仅用于判断周期是否结束，不含价格；行情严格冻结在 2026-10-26。",
            "本模块不算收益、不做组合回测；MFE/MAE 仅是单笔路径描述原语，供策略层使用。",
            "源库为 WAL 模式且可能被外部同步/其他进程写入（本次运行期间 baostock.db 的 mtime 与行数确有变化）；"
            "本产物的所有查询均带 date<=2026-10-26 过滤且只读打开，复现时应以同一冻结快照为准。",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    for c in checks:
        print(f"[{c['status']}] {c['id']}: {c['detail']}")
    print(f"-> {out}  ({payload['meta']['runtime_sec']}s)")
    return 0 if payload["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
