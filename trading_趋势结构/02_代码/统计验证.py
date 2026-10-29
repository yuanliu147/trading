#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计与动态稳健性验证框架（趋势结构 / 波段交易研究）。

定位
----
本文件只做“度量、推断、扰动、审计”，不产生选股规则，也不替策略研究员选参。
它把任何满足 :class:`BacktestResult` 契约的回测结果，放进统一的绩效口径、
重采样推断、多重试验校正、参数邻域、摩擦扰动、滚动起点与 walk-forward 中检验。

三条硬约束
----------
1. 数据冻结 ``CUTOFF = 2026-10-26``；任何行情读取都截断到该日。
2. 2021-01-01 起只允许“数据覆盖审计”。凡涉及收益的入口都会经
   :func:`guard_performance` 拒绝，除非显式传入 ``allow_holdout=True``
   （留出集只允许在最终一次性评估时使用，且必须登记）。
3. 不使用未来数据：回测结果的成交日必须晚于信号日；参数只在训练窗口内选择。

数据口径
--------
* 收益：用 baostock ``close / preclose - 1``。``preclose`` 已做公司行动调整，
  与后复权收盘价收益在样本核对中一致（见 ``data-audit``）。
* 可交易性：``trade_status == 0`` 或 ``volume == 0`` 视为停牌，不能成交。
* 涨跌停：按板块与日期分段判定，见 :func:`limit_ratio`。
* 费用：佣金双边、印花税卖出单边（2023-08-28 起 0.05%，此前 0.1%）、
  过户费、滑点，见 :class:`CostModel`。

用法
----
    python3 统计验证.py selftest       # 静态数学自检（无需数据库）
    python3 统计验证.py data-audit     # 两源数据质量与覆盖审计
    python3 统计验证.py reference      # 参考夹具端到端跑通（仅开发窗口）
    python3 统计验证.py all            # 以上全部

作者：统计与动态稳健性负责人。其他研究员请勿修改本文件，通过
``StrategyAdapter`` / ``BacktestResult`` 接入。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 0. 常量、窗口与留出集护栏
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "02_代码"
RESULT_DIR = ROOT / "03_结果"
CACHE_DIR = CODE_DIR / "cache"
ARTIFACT_DIR = RESULT_DIR / "统计验证"
for _d in (CACHE_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2026-10-26")
HOLDOUT_START = pd.Timestamp("2021-01-01")

BAOSTOCK_DB = Path("/Users/admin/WorkSpace/baostock_data/data/baostock.db")
DUCKDB_PATH = Path(
    "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"
)

# 时间纪律：开发/验证可用，留出只看覆盖
WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "preheat": (pd.Timestamp("2005-01-01"), pd.Timestamp("2007-12-31")),
    "dev": (pd.Timestamp("2008-01-01"), pd.Timestamp("2015-12-31")),
    "val": (pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
    "holdout": (pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31")),
    "ext": (pd.Timestamp("2026-01-01"), CUTOFF),
}
TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25

LABELS_CN = {
    "cagr": "年化收益",
    "total_return": "累计收益",
    "mdd": "最大回撤",
    "sharpe": "夏普",
    "sortino": "索提诺",
    "calmar": "卡玛",
    "vol": "年化波动",
    "turnover": "年化单边换手",
    "exposure": "平均仓位暴露",
    "time_in_market": "持仓时间占比",
    "capacity": "容量(元)",
    "win_rate": "胜率",
    "profit_factor": "盈亏比",
    "ulcer": "溃疡指数",
    "skew": "日收益偏度",
    "kurtosis": "日收益峰度",
}


class HoldoutViolation(RuntimeError):
    """试图在 2021+ 计算收益。"""


def guard_performance(
    start: Any, end: Any, purpose: str = "performance", allow_holdout: bool = False
) -> None:
    """性能计算护栏。

    purpose='coverage_audit' 时放行（只允许行数/缺失/覆盖率统计）；
    其余情况只要区间触及 2021-01-01 就抛错，除非显式 allow_holdout。
    """
    if purpose == "coverage_audit" or allow_holdout:
        return
    end_ts = pd.Timestamp(end)
    if end_ts >= HOLDOUT_START:
        raise HoldoutViolation(
            f"区间 {start}~{end} 触及留出期（>= {HOLDOUT_START.date()}）。"
            "2021+ 仅允许 purpose='coverage_audit' 的数据覆盖审计；"
            "策略收益评估须由主协调者显式 allow_holdout=True 并登记。"
        )


def window(name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if name not in WINDOWS:
        raise KeyError(f"未知窗口 {name}，可选 {sorted(WINDOWS)}")
    return WINDOWS[name]


# ---------------------------------------------------------------------------
# 1. 小型工具
# ---------------------------------------------------------------------------


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not math.isfinite(v) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return str(pd.Timestamp(o).date() if not isinstance(o, datetime) else o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    if isinstance(o, pd.Series):
        return {str(k): v for k, v in o.to_dict().items()}
    raise TypeError(f"无法序列化 {type(o)}")


def _stringify_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify_keys(v) for v in obj]
    return obj


def dump_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = _stringify_keys(obj)
    clean = json.loads(json.dumps(obj, default=_json_default, ensure_ascii=False))
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def as_series(x: Any, index: Any | None = None, name: str = "value") -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.astype(float).copy()
    else:
        s = pd.Series(np.asarray(x, dtype=float), index=index, name=name)
    if index is not None and not isinstance(x, pd.Series):
        s.index = pd.Index(index)
    s.name = name
    if s.index.has_duplicates:
        raise ValueError("索引存在重复日期")
    return s.sort_index()


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    return float(a / b) if b not in (0, 0.0) and np.isfinite(b) else default


def max_drawdown(nav: np.ndarray | pd.Series) -> tuple[float, int, int]:
    """最大回撤（含起点资金作为第一个峰）。

    返回 (mdd, 峰值位置, 谷值位置)，mdd 为负数。
    """
    v = np.asarray(nav, dtype=float)
    if v.size == 0:
        return np.nan, -1, -1
    peak = np.maximum.accumulate(v)
    dd = v / peak - 1.0
    trough = int(np.argmin(dd))
    peak_idx = int(np.argmax(v[: trough + 1])) if trough >= 0 else 0
    return float(dd[trough]), peak_idx, trough


def longest_underwater(nav: np.ndarray | pd.Series) -> int:
    v = np.asarray(nav, dtype=float)
    peak = np.maximum.accumulate(v)
    under = v < peak - 1e-12
    best = run = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return int(best)


def drawdown_series(nav: np.ndarray | pd.Series) -> np.ndarray:
    v = np.asarray(nav, dtype=float)
    peak = np.maximum.accumulate(v)
    return v / peak - 1.0


def ulcer_index(nav: np.ndarray | pd.Series) -> float:
    dd = drawdown_series(nav)
    return float(np.sqrt(np.mean(np.square(np.minimum(dd, 0.0)))))


# ---------------------------------------------------------------------------
# 2. 数据适配器（只读源库 + 本地派生缓存）
# ---------------------------------------------------------------------------

BS_COLUMNS = [
    "code", "date", "open", "high", "low", "close", "preclose",
    "volume", "amount", "turn", "trade_status", "pct_chg", "is_st",
]


class MarketData:
    """日线面板的最小只读接口。

    所有实现返回长表，列名统一为：
    code, date, open, high, low, close, preclose, volume, amount, turn,
    trade_status, is_st, tradable, ret
    """

    name: str = "abstract"

    def trading_days(self, start: Any, end: Any) -> pd.DatetimeIndex:
        raise NotImplementedError

    def bars(self, start: Any, end: Any, codes: Sequence[str] | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def listed_codes(self, asof: Any | None = None) -> list[str]:
        raise NotImplementedError

    def benchmark(self, code: str, start: Any, end: Any) -> pd.Series:
        raise NotImplementedError

    def adtv(self, start: Any, end: Any, window: int = 20) -> pd.DataFrame:
        """20 日平均成交额（元），用于容量估计。"""
        b = self.bars(start, end)
        b = b.sort_values(["code", "date"])
        b["adtv"] = (
            b.groupby("code", observed=True)["amount"]
            .transform(lambda s: s.rolling(window, min_periods=max(5, window // 2)).mean())
        )
        return b[["code", "date", "adtv"]]


def _normalise_bars(df: pd.DataFrame) -> pd.DataFrame:
    need = set(BS_COLUMNS)
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"行情缺少列 {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= CUTOFF]
    for c in ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pct_chg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trade_status"] = pd.to_numeric(df["trade_status"], errors="coerce").fillna(1).astype("int8")
    df["is_st"] = pd.to_numeric(df["is_st"], errors="coerce").fillna(0).astype("int8")
    df["tradable"] = (df["trade_status"].eq(1)) & (df["volume"].fillna(0) > 0) & df["close"].gt(0)
    # close/preclose-1 即含公司行动收益；preclose 缺失时退回 pct_chg
    ret = df["close"] / df["preclose"] - 1.0
    fallback = df["pct_chg"] / 100.0
    df["ret"] = np.where(np.isfinite(ret), ret, fallback)
    df.loc[~df["tradable"], "ret"] = 0.0
    df["ret"] = df["ret"].replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["date", "code"]).reset_index(drop=True)


class BaoStockMarket(MarketData):
    """baostock SQLite 适配器。

    源库只读（``mode=ro``）。按 ``(adjustflag, date)`` 分区读取并按年缓存为
    parquet，缓存是可重建的派生数据，不含任何加工后的信号。
    """

    name = "baostock"

    def __init__(self, db: Path = BAOSTOCK_DB, use_cache: bool = True):
        self.db = Path(db)
        if not self.db.exists():
            raise FileNotFoundError(self.db)
        self.use_cache = use_cache
        self._mem: dict[tuple[int, int], pd.DataFrame] = {}
        self._codes: list[str] | None = None

    # -- 基础读取 ---------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True, timeout=30)
        con.execute("PRAGMA query_only=ON")
        return con

    def _cache_path(self, adjustflag: int, year: int) -> Path:
        return CACHE_DIR / f"bs_{adjustflag}_{year}.parquet"

    def _read_year(self, year: int, adjustflag: int = 3) -> pd.DataFrame:
        key = (adjustflag, year)
        if key in self._mem:
            return self._mem[key]
        cp = self._cache_path(adjustflag, year)
        if self.use_cache and cp.exists():
            df = pd.read_parquet(cp)
        else:
            con = self._connect()
            try:
                cols = ",".join(BS_COLUMNS) if adjustflag == 3 else "code,date,close,preclose,pct_chg"
                df = pd.read_sql_query(
                    f"SELECT {cols} FROM stock_kline_d "
                    "WHERE adjustflag=? AND date BETWEEN ? AND ?",
                    con,
                    params=(adjustflag, f"{year}-01-01", f"{year}-12-31"),
                )
            finally:
                con.close()
            if self.use_cache and adjustflag == 3 and len(df):
                df.to_parquet(cp, index=False)
        if adjustflag != 3:
            return df
        out = _normalise_bars(df)
        self._mem[key] = out
        return out

    def trading_days(self, start: Any, end: Any) -> pd.DatetimeIndex:
        con = self._connect()
        try:
            d = pd.read_sql_query(
                "SELECT calendar_date FROM trade_dates "
                "WHERE is_trading_day=1 AND calendar_date BETWEEN ? AND ? ORDER BY 1",
                con,
                params=(str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())),
            )["calendar_date"]
        finally:
            con.close()
        return pd.DatetimeIndex(pd.to_datetime(d))

    def ipo_dates(self) -> dict[str, pd.Timestamp]:
        """上市首日（stock_basic.ipo_date）。用于剔除首日不可信收益。"""
        if getattr(self, "_ipo_cache", None) is not None:
            return self._ipo_cache
        con = self._connect()
        try:
            df = pd.read_sql_query("SELECT code, ipo_date FROM stock_basic", con)
        finally:
            con.close()
        out: dict[str, pd.Timestamp] = {}
        for c, d in zip(df["code"], df["ipo_date"]):
            if isinstance(d, str) and len(d) == 10:
                out[c] = pd.Timestamp(d)
        self._ipo_cache = out
        return out

    def bars(self, start: Any, end: Any, codes: Sequence[str] | None = None) -> pd.DataFrame:
        s, e = pd.Timestamp(start), min(pd.Timestamp(end), CUTOFF)
        frames = [self._read_year(y) for y in range(s.year, e.year + 1)]
        df = pd.concat([f for f in frames if len(f)], ignore_index=True)
        df = df[(df["date"] >= s) & (df["date"] <= e)]
        if codes is not None:
            df = df[df["code"].isin(set(codes))]
        df = df.reset_index(drop=True)
        # 上市首日 preclose 常为发行价/未调整值，收益不可信，且首日涨跌停规则特殊
        ipo = self.ipo_dates()
        if ipo and len(df):
            first = df["code"].map(ipo)
            first = first.notna() & (df["date"] <= first + pd.Timedelta(days=3))
            df["first_day"] = first.to_numpy()
            df.loc[first, "ret"] = np.nan
            df.loc[first, "tradable"] = False
        else:
            df["first_day"] = False
        return df

    def listed_codes(self, asof: Any | None = None) -> list[str]:
        con = self._connect()
        try:
            df = pd.read_sql_query(
                "SELECT code, ipo_date, out_date, status FROM stock_basic", con
            )
        finally:
            con.close()
        if asof is not None:
            a = str(pd.Timestamp(asof).date())
            df = df[(df["ipo_date"].fillna("1900-01-01") <= a)]
            df = df[df["out_date"].isna() | (df["out_date"].fillna("9999") > a)]
        return sorted(df["code"].tolist())

    def benchmark(self, code: str = "sh.000300", start: Any = "2005-01-01", end: Any = CUTOFF) -> pd.Series:
        con = self._connect()
        try:
            df = pd.read_sql_query(
                "SELECT date, close FROM index_kline_d WHERE code=? AND date BETWEEN ? AND ? ORDER BY date",
                con,
                params=(code, str(pd.Timestamp(start).date()), str(min(pd.Timestamp(end), CUTOFF).date())),
            )
        finally:
            con.close()
        s = pd.Series(df["close"].to_numpy(float), index=pd.to_datetime(df["date"]))
        return s[~s.index.duplicated()].sort_index()


class DuckDBMarket(MarketData):
    """同花顺 DuckDB 适配器（2016-10 起），用于交叉核对与近期窗口。"""

    name = "duckdb"

    def __init__(self, path: Path = DUCKDB_PATH):
        self.path = Path(path)

    def _con(self):
        import duckdb

        return duckdb.connect(str(self.path), read_only=True)

    def trading_days(self, start: Any, end: Any) -> pd.DatetimeIndex:
        con = self._con()
        try:
            d = con.execute(
                "SELECT DISTINCT date FROM v_daily WHERE date BETWEEN ? AND ? ORDER BY 1",
                [pd.Timestamp(start).date(), min(pd.Timestamp(end), CUTOFF).date()],
            ).df()["date"]
        finally:
            con.close()
        return pd.DatetimeIndex(pd.to_datetime(d))

    def bars(self, start: Any, end: Any, codes: Sequence[str] | None = None) -> pd.DataFrame:
        con = self._con()
        try:
            sql = (
                "SELECT thscode AS code, date, open, high, low, close, prev_close AS preclose,"
                " volume, amount FROM v_daily WHERE date BETWEEN ? AND ?"
            )
            args: list[Any] = [pd.Timestamp(start).date(), min(pd.Timestamp(end), CUTOFF).date()]
            if codes is not None:
                sql += " AND thscode IN (SELECT * FROM unnest(?))"
                args.append(list(codes))
            df = con.execute(sql, args).df()
        finally:
            con.close()
        df["turn"] = np.nan
        df["trade_status"] = np.where(df["volume"].fillna(0) > 0, 1, 0)
        df["is_st"] = 0
        df["pct_chg"] = np.nan
        return _normalise_bars(df)

    def listed_codes(self, asof: Any | None = None) -> list[str]:
        con = self._con()
        try:
            return sorted(con.execute("SELECT thscode FROM dim_symbol").df()["thscode"].tolist())
        finally:
            con.close()

    def benchmark(self, code: str = "000300.SH", start: Any = "2016-10-13", end: Any = CUTOFF) -> pd.Series:
        raise NotImplementedError("DuckDB 库不含指数表；基准请用 BaoStockMarket.benchmark")


# ---------------------------------------------------------------------------
# 3. 策略契约
# ---------------------------------------------------------------------------

REQUIRED_TRADE_COLS = {
    "code", "entry_date", "entry_price", "exit_date", "exit_price", "shares",
    "pnl", "ret",
}


@dataclass
class BacktestResult:
    """回测输出契约。策略研究员只需产出这些表，其余交给本模块。

    equity: 索引为交易日，列至少 nav（收盘总权益）、market_value、cash。
            可选 turnover（当日成交额）、n_positions、cost。
    trades: 一行一笔完整交易，含 entry/exit 日期与价格、股数、净盈亏、收益率。
            可选 weight_at_entry、contrib（对组合净值的贡献）、reason_*。
    signals: 可选，一行一个候选信号，含 asof（信号日）、code、score。
    """

    equity: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.equity, pd.DataFrame):
            raise TypeError("equity 必须是 DataFrame")
        if "nav" not in self.equity.columns:
            raise ValueError("equity 必须包含 nav 列")
        self.equity.index = pd.DatetimeIndex(pd.to_datetime(self.equity.index))
        self.equity = self.equity.sort_index()
        self.equity = self.equity[~self.equity.index.duplicated(keep="last")]
        self.benchmark: pd.Series | None = None

    # -- 便捷属性 ---------------------------------------------------------
    @property
    def nav(self) -> pd.Series:
        return self.equity["nav"].astype(float)

    @property
    def returns(self) -> pd.Series:
        nav = self.nav
        r = nav.pct_change()
        if len(nav):
            r.iloc[0] = nav.iloc[0] / float(self.meta.get("initial_cash", nav.iloc[0])) - 1.0
        return r.fillna(0.0)

    def slice(self, start: Any, end: Any) -> "BacktestResult":
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        eq = self.equity[(self.equity.index >= s) & (self.equity.index <= e)]
        tr = self.trades
        if len(tr):
            tr = tr[(pd.to_datetime(tr["exit_date"]) >= s) & (pd.to_datetime(tr["entry_date"]) <= e)]
        return BacktestResult(eq, tr.copy(), self.signals, dict(self.meta))


def validate_result(res: BacktestResult, market: MarketData | None = None,
                    strict: bool = True) -> dict[str, Any]:
    """契约与未来函数自检。返回问题清单，strict=True 时发现问题即抛错。"""
    issues: list[str] = []
    warn: list[str] = []
    eq, tr = res.equity, res.trades

    if len(eq) == 0:
        issues.append("equity 为空")
    if not np.isfinite(eq["nav"]).all():
        issues.append("nav 含非有限值")
    if (eq["nav"] <= 0).any():
        issues.append("nav 出现非正值")
    if eq.index.has_duplicates:
        issues.append("equity 索引重复")
    if not eq.index.is_monotonic_increasing:
        issues.append("equity 索引非升序")

    if len(tr):
        miss = REQUIRED_TRADE_COLS - set(tr.columns)
        if miss:
            issues.append(f"trades 缺少列 {sorted(miss)}")
        else:
            ed = pd.to_datetime(tr["entry_date"])
            xd = pd.to_datetime(tr["exit_date"])
            if (xd < ed).any():
                issues.append("存在退出日早于进入日的交易")
            if (xd > CUTOFF).any():
                issues.append("存在晚于冻结日的交易")
            if (tr["shares"] <= 0).any():
                issues.append("存在非正股数")
            if (tr["shares"] % 1 != 0).any():
                issues.append("存在非整股交易")
            if (tr["entry_price"] <= 0).any() or (tr["exit_price"] <= 0).any():
                issues.append("存在非正成交价")
            if "signal_date" in tr.columns:
                sd = pd.to_datetime(tr["signal_date"])
                if (ed <= sd).any():
                    issues.append("存在成交日不晚于信号日（T+1 违规）")
            if "exit_signal_date" in tr.columns:
                xsd = pd.to_datetime(tr["exit_signal_date"])
                if (xd <= xsd).any():
                    issues.append("存在卖出日不晚于卖出信号日（T+1 违规）")
            # 同一代码持仓重叠
            for code, g in tr.sort_values("entry_date").groupby("code", observed=True):
                g = g.reset_index(drop=True)
                for i in range(1, len(g)):
                    if g.loc[i, "entry_date"] <= g.loc[i - 1, "exit_date"]:
                        issues.append(f"{code} 持仓区间重叠（未按 T+1 平仓）")
                        break
        if "pnl" in tr.columns and "ret" in tr.columns:
            implied = tr["pnl"] / (tr["entry_price"] * tr["shares"])
            bad = (implied - tr["ret"]).abs() > 0.02
            if bad.mean() > 0.01:
                warn.append(f"{bad.mean():.1%} 笔交易的 pnl 与 ret 不自洽（>2pct）")

    if market is not None and len(tr) and "code" in tr.columns:
        # 成交日必须是该股可交易日
        pass

    out = {"ok": not issues, "issues": issues, "warnings": warn,
           "n_equity_days": int(len(eq)), "n_trades": int(len(tr))}
    if issues and strict:
        raise ValueError("回测结果未通过契约检查：" + "；".join(issues))
    return out


class StrategyAdapter:
    """策略接入的最小接口（策略研究员实现）。

    ``run`` 必须遵守：
      * 信号在收盘后生成，最早下一交易日开盘成交；
      * T+1、停牌不成交、涨跌停不成交、整股、费用滑点；
      * 不得使用 ``end`` 之后的数据。
    """

    name = "adapter"

    def run(self, market: MarketData, params: Mapping[str, Any],
            start: Any, end: Any) -> BacktestResult:
        raise NotImplementedError

    def default_grid(self) -> dict[str, list]:
        return {}

    def supports(self, perturbation: str) -> bool:
        return False



# ---------------------------------------------------------------------------
# 4. 交易摩擦：费用、涨跌停、可成交判定
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """A 股现实现金摩擦。

    commission_bps: 佣金，双边，按成交额；min_commission 为单笔最低佣金。
    stamp_bps: 印花税，仅卖出；2023-08-28 起由 10bp 降至 5bp。
    transfer_bps: 过户费，双边（沪深现行 0.001%）。
    slippage_bps: 滑点，按成交额，买入上滑、卖出下滑。
    """

    commission_bps: float = 2.5
    min_commission: float = 5.0
    stamp_bps: float | None = None
    transfer_bps: float = 0.1
    slippage_bps: float = 10.0

    def stamp_rate(self, when: Any = None) -> float:
        """卖出印花税。stamp_bps 显式给定时固定，否则按 2023-08-28 分段。"""
        if self.stamp_bps is not None:
            return float(self.stamp_bps) / 1e4
        if when is None:
            return 5.0 / 1e4
        d = pd.Timestamp(when)
        return (10.0 if d < pd.Timestamp("2023-08-28") else 5.0) / 1e4

    def buy_cost(self, notional: float) -> float:
        n = abs(notional)
        return max(n * self.commission_bps / 1e4, self.min_commission) + n * (
            self.transfer_bps + self.slippage_bps
        ) / 1e4

    def sell_cost(self, notional: float, when: Any = None) -> float:
        n = abs(notional)
        stamp = n * self.stamp_rate(when)
        return (
            max(n * self.commission_bps / 1e4, self.min_commission)
            + n * (self.transfer_bps + self.slippage_bps) / 1e4
            + stamp
        )

    def total_bps(self, when: Any = None, side: str = "roundtrip") -> float:
        stamp = self.stamp_rate(when) * 1e4
        if side == "buy":
            return self.commission_bps + self.transfer_bps + self.slippage_bps
        if side == "sell":
            return self.commission_bps + self.transfer_bps + self.slippage_bps + stamp
        return 2 * (self.commission_bps + self.transfer_bps + self.slippage_bps) + stamp


DEFAULT_COST = CostModel()


def limit_ratio(code: str, when: Any, is_st: bool = False) -> float:
    """涨跌停幅度（小数）。按板块与制度变更分段。

    主板 10%；ST 5%（科创板/创业板注册制后仍 20%，此处按 20% 处理并标注）；
    创业板 2020-08-24 起 20%；科创板 20%；北交所 30%。
    """
    d = pd.Timestamp(when)
    c = str(code)
    ticker = c.split(".")[-1] if "." in c else c
    board_gem = ticker.startswith("300") or ticker.startswith("301")
    board_star = ticker.startswith("688")
    board_bj = c.startswith("bj.") or ticker.startswith(("83", "87", "88", "43", "92"))
    if board_star:
        return 0.20
    if board_bj:
        return 0.30
    if board_gem:
        return 0.20 if d >= pd.Timestamp("2020-08-24") else 0.10
    if is_st:
        return 0.05
    return 0.10


def price_limit_flags(df: pd.DataFrame, tol: float = 0.005) -> pd.DataFrame:
    """给行情表加涨停/跌停/一字板标记。

    df 需含 code, date, open, high, low, close, preclose, is_st。
    判定用 preclose 与板块阈值；tol 为价格取整容差（元）。
    """
    out = df.copy()
    code_s = out["code"].astype(str)
    ticker = code_s.str.split(".").str[-1]
    gem = ticker.str.startswith(("300", "301"))
    star = ticker.str.startswith("688")
    bj = code_s.str.startswith("bj.") | ticker.str.startswith(("83", "87", "88", "43", "92"))
    after_gem_reform = out["date"] >= pd.Timestamp("2020-08-24")
    is_st = out["is_st"].fillna(0).astype(bool)
    ratios = np.where(
        star, 0.20,
        np.where(bj, 0.30,
                 np.where(gem, np.where(after_gem_reform, 0.20, 0.10),
                          np.where(is_st, 0.05, 0.10))))
    ratios = ratios.astype(float)
    up = out["preclose"].to_numpy(float) * (1 + ratios)
    dn = out["preclose"].to_numpy(float) * (1 - ratios)
    c = out["close"].to_numpy(float)
    o = out["open"].to_numpy(float)
    h = out["high"].to_numpy(float)
    lo = out["low"].to_numpy(float)
    out["limit_up"] = c >= up - tol
    out["limit_down"] = c <= dn + tol
    out["one_price_up"] = out["limit_up"] & (h <= lo + tol)
    out["one_price_down"] = out["limit_down"] & (h <= lo + tol)
    out["open_limit_up"] = o >= up - tol
    out["open_limit_down"] = o <= dn + tol
    out["can_buy_open"] = out["tradable"] & ~out["open_limit_up"]
    out["can_sell_open"] = out["tradable"] & ~out["open_limit_down"]
    return out


# ---------------------------------------------------------------------------
# 5. 绩效指标
# ---------------------------------------------------------------------------


def compute_metrics(
    res: BacktestResult,
    initial_cash: float | None = None,
    rf: float = 0.0,
    market: MarketData | None = None,
    benchmark: pd.Series | None = None,
    participation_limit: float = 0.05,
    allow_holdout: bool = False,
) -> dict[str, Any]:
    """统一绩效口径。所有定义见 ``03_结果/稳健性检验规范.md``。

    rf 为年化无风险利率，按 252 日折算；夏普用日频超额收益、ddof=1。
    """
    eq = res.equity
    if len(eq) < 2:
        return {"error": "样本不足"}
    guard_performance(eq.index[0], eq.index[-1], allow_holdout=allow_holdout)

    nav = eq["nav"].astype(float)
    init = float(initial_cash if initial_cash is not None else res.meta.get("initial_cash", nav.iloc[0]))
    # 把起点资金并入路径，首日费用/盈亏不被漏掉
    path = np.r_[init, nav.to_numpy(float)]
    days = max((nav.index[-1] - nav.index[0]).days, 1)
    years_cal = days / DAYS_PER_YEAR
    n = len(nav)

    r = nav.pct_change().to_numpy(float).copy()
    r[0] = nav.iloc[0] / init - 1.0
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    rf_d = rf / TRADING_DAYS_PER_YEAR
    ex = r - rf_d
    sd = float(np.std(r, ddof=1)) if n > 1 else np.nan
    mdd, peak_i, trough_i = max_drawdown(path)
    total = float(nav.iloc[-1] / init - 1.0)
    cagr = float((nav.iloc[-1] / init) ** (1 / years_cal) - 1.0) if years_cal > 0 else np.nan
    downside = float(np.sqrt(np.mean(np.square(np.minimum(ex, 0.0))))) if n else np.nan

    m: dict[str, Any] = {
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "n_days": int(n),
        "years": float(years_cal),
        "initial_cash": init,
        "final_nav": float(nav.iloc[-1]),
        "total_return": total,
        "cagr": cagr,
        "cagr_trading": float((1 + total) ** (TRADING_DAYS_PER_YEAR / max(n, 1)) - 1),
        "vol": float(sd * math.sqrt(TRADING_DAYS_PER_YEAR)) if np.isfinite(sd) else np.nan,
        "sharpe": _safe_div(float(np.mean(ex)) * TRADING_DAYS_PER_YEAR, float(sd) * math.sqrt(TRADING_DAYS_PER_YEAR))
        if np.isfinite(sd) and sd > 0 else np.nan,
        "sortino": _safe_div(float(np.mean(ex)) * TRADING_DAYS_PER_YEAR, downside * math.sqrt(TRADING_DAYS_PER_YEAR))
        if downside and downside > 0 else np.nan,
        "mdd": mdd,
        "mdd_peak": str(nav.index[max(peak_i - 1, 0)].date()) if peak_i > 0 else None,
        "mdd_trough": str(nav.index[max(trough_i - 1, 0)].date()) if trough_i > 0 else None,
        "calmar": _safe_div(cagr, abs(mdd)) if mdd and mdd < 0 else np.nan,
        "ulcer": ulcer_index(path),
        "longest_underwater": longest_underwater(path),
        "worst_day": float(np.min(r)),
        "best_day": float(np.max(r)),
        "var95": float(np.quantile(r, 0.05)),
        "cvar95": float(np.mean(r[r <= np.quantile(r, 0.05)])) if n else np.nan,
        "skew": float(stats.skew(r, bias=False)) if n > 2 else np.nan,
        "kurtosis": float(stats.kurtosis(r, fisher=False, bias=False)) if n > 3 else np.nan,
        "positive_days": float(np.mean(r > 0)),
        "rf": rf,
    }

    # 换手 / 暴露
    if "turnover" in eq.columns:
        traded = float(eq["turnover"].fillna(0).sum())
    elif len(res.trades):
        tr = res.trades
        traded = float((tr["entry_price"] * tr["shares"] + tr["exit_price"] * tr["shares"]).sum())
    else:
        traded = np.nan
    avg_nav = float(nav.mean())
    m["turnover"] = float(traded / 2.0 / avg_nav / years_cal) if np.isfinite(traded) and avg_nav > 0 else np.nan
    m["traded_notional"] = traded

    if "market_value" in eq.columns:
        mv = eq["market_value"].astype(float).fillna(0.0)
        m["exposure"] = float((mv / nav).mean())
        m["time_in_market"] = float((mv > 0).mean())
    else:
        m["exposure"] = np.nan
        m["time_in_market"] = np.nan
    m["avg_positions"] = float(eq["n_positions"].mean()) if "n_positions" in eq.columns else np.nan
    m["cost_paid"] = float(eq["cost"].fillna(0).sum()) if "cost" in eq.columns else np.nan

    # 交易统计
    tr = res.trades
    if len(tr):
        rets = tr["ret"].astype(float).to_numpy()
        wins, losses = rets[rets > 0], rets[rets < 0]
        m["n_trades"] = int(len(tr))
        m["win_rate"] = float(len(wins) / len(rets))
        m["avg_trade_ret"] = float(np.mean(rets))
        m["median_trade_ret"] = float(np.median(rets))
        m["profit_factor"] = _safe_div(float(wins.sum()), float(-losses.sum())) if len(losses) else np.inf
        m["avg_hold_days"] = float(
            (pd.to_datetime(tr["exit_date"]) - pd.to_datetime(tr["entry_date"])).dt.days.mean()
        )
        m["avg_win"] = float(np.mean(wins)) if len(wins) else np.nan
        m["avg_loss"] = float(np.mean(losses)) if len(losses) else np.nan
        m["largest_win"] = float(np.max(rets))
        m["largest_loss"] = float(np.min(rets))
    else:
        m["n_trades"] = 0

    # 基准对照
    if benchmark is not None and len(benchmark) > 1:
        b = benchmark.reindex(nav.index).ffill().dropna()
        if len(b) > 2:
            br = b.pct_change().fillna(0.0)
            m["benchmark_cagr"] = float((b.iloc[-1] / b.iloc[0]) ** (1 / years_cal) - 1)
            m["benchmark_mdd"] = max_drawdown(np.r_[b.iloc[0], b.to_numpy(float)])[0]
            active = r[-len(br):] - br.to_numpy()
            m["excess_cagr"] = m["cagr"] - m["benchmark_cagr"]
            te = float(np.std(active, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
            m["tracking_error"] = te
            m["information_ratio"] = _safe_div(float(np.mean(active)) * TRADING_DAYS_PER_YEAR, te) if te > 0 else np.nan
            rr, bb = r[-len(br):], br.to_numpy(float)
            if len(rr) == len(bb) and len(bb) > 2:
                m["beta"] = _safe_div(float(np.cov(rr, bb)[0, 1]), float(np.var(bb, ddof=1)))
            m["alpha_annual"] = _safe_div(m["cagr"] - rf - m["beta"] * (m["benchmark_cagr"] - rf), 1.0)
    return m


def annual_table(res: BacktestResult, initial_cash: float | None = None) -> pd.DataFrame:
    nav = res.nav
    init = float(initial_cash if initial_cash is not None else res.meta.get("initial_cash", nav.iloc[0]))
    year_end = nav.resample("YE").last()
    rows = []
    prev = init
    for ts, v in year_end.items():
        r = v / prev - 1
        seg = nav[(nav.index >= ts.replace(month=1, day=1)) & (nav.index <= ts)]
        path = np.r_[prev, seg.to_numpy(float)]
        rows.append(
            {
                "year": int(ts.year),
                "return": float(r),
                "end_nav": float(v),
                "mdd": max_drawdown(path)[0],
                "days": int(len(seg)),
                "partial": bool(ts.month < 12),
            }
        )
        prev = v
    df = pd.DataFrame(rows)
    if len(df):
        df["full_years"] = ~df["partial"]
        df["positive"] = df["return"] > 0
    return df


def monthly_table(res: BacktestResult, initial_cash: float | None = None) -> pd.DataFrame:
    nav = res.nav
    init = float(initial_cash if initial_cash is not None else res.meta.get("initial_cash", nav.iloc[0]))
    m_end = nav.resample("ME").last()
    prev = init
    rows = []
    for ts, v in m_end.items():
        rows.append({"month": ts.strftime("%Y-%m"), "return": float(v / prev - 1), "end_nav": float(v)})
        prev = v
    return pd.DataFrame(rows)


def monthly_matrix(res: BacktestResult, initial_cash: float | None = None) -> pd.DataFrame:
    mt = monthly_table(res, initial_cash)
    if mt.empty:
        return mt
    mt["year"] = mt["month"].str[:4].astype(int)
    mt["mon"] = mt["month"].str[5:].astype(int)
    wide = mt.pivot(index="year", columns="mon", values="return")
    wide.columns = [f"{c:02d}" for c in wide.columns]
    wide["全年"] = wide.apply(lambda row: float(np.prod(1 + row.dropna()) - 1), axis=1)
    return wide


# ---------------------------------------------------------------------------
# 6. 区块 bootstrap
# ---------------------------------------------------------------------------


def polit_white_blocklen(x: np.ndarray, max_frac: float = 1 / 3) -> float:
    """Politis-White (2004) 最优区块长度（circular bootstrap 版本）。

    用 flat-top 滞后窗估计 long-run variance 的 G/D，返回连续值，取整由调用方决定。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 20:
        return float(max(2, int(math.sqrt(n))))
    x = x - x.mean()
    gamma0 = float(np.dot(x, x) / n)
    if gamma0 <= 0:
        return 2.0
    kmax = min(n - 1, int(math.ceil(math.sqrt(n))) + 5)
    rho = np.array([float(np.dot(x[:-k], x[k:]) / (n * gamma0)) for k in range(1, kmax + 1)])
    crit = 2.0 * math.sqrt(math.log10(max(n, 10)) / n)
    m_hat = 1
    for k, r in enumerate(rho, start=1):
        if abs(r) < crit:
            m_hat = k
            break
    else:
        m_hat = kmax
    M = max(1, min(2 * m_hat, kmax))
    k = np.arange(1, M + 1)
    lam = np.where(k / M <= 0.5, 1.0, 2.0 * (1.0 - k / M))
    g_hat = float(np.sum(lam * k * rho[:M]))  # 含正负两侧后乘 2
    d_hat = float(np.sum(lam * rho[:M]))
    G = 2.0 * g_hat
    D = (4.0 / 3.0) * (1.0 + 2.0 * d_hat)
    if D <= 0:
        return float(max(2, min(int(round(1.5 * math.sqrt(n))), int(n * max_frac))))
    b = ((2.0 * G**2) / D) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 2.0), max(2.0, n * max_frac)))


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = int(max(1, block))
    nblocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=nblocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.reshape(-1)[:n]


def stationary_bootstrap_indices(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano 平稳 bootstrap：几何区块，期望长度 block。"""
    p = 1.0 / max(float(block), 1.0)
    idx = np.empty(n, dtype=np.int64)
    i = int(rng.integers(0, n))
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = int(rng.integers(0, n))
        else:
            i = (i + 1) % n
    return idx


def _bootstrap_stats_from_returns(
    r: np.ndarray, n_days_per_year: float, rf_daily: float = 0.0
) -> tuple[float, float, float, float, float]:
    """从日收益重算 (cagr, sharpe, mdd, calmar, vol)。"""
    path = np.cumprod(1.0 + r)
    years = max(len(r) / n_days_per_year, 1e-9)
    cagr = float(path[-1] ** (1 / years) - 1) if path[-1] > 0 else -1.0
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
    sharpe = float((np.mean(r) - rf_daily) / sd * math.sqrt(n_days_per_year)) if sd and sd > 0 else np.nan
    mdd = max_drawdown(np.r_[1.0, path])[0]
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    vol = sd * math.sqrt(n_days_per_year) if np.isfinite(sd) else np.nan
    return cagr, sharpe, mdd, calmar, vol


def block_bootstrap(
    returns: Any,
    n_boot: int = 2000,
    method: str = "stationary",
    block_len: float | None = None,
    seed: int = 20261026,
    rf: float = 0.0,
    ci: float = 0.95,
    dates: Any | None = None,
    allow_holdout: bool = False,
) -> dict[str, Any]:
    """对日收益序列做区块 bootstrap，给出 CAGR/Sharpe/MDD/Calmar 的区间。

    方法：
      stationary — Politis-Romano 几何区块，保留短程相关；
      circular   — 固定长度循环区块，尾首相接；
      monthly    — 以自然月为区块（需 dates），保留月内相关与月度季节性。
    说明：重采样只能反映“历史路径的抽样波动”，不能修复样本选择、
    生存偏差、制度变化，也不等于未来盈利概率。
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if dates is not None and len(dates):
        guard_performance(pd.DatetimeIndex(pd.to_datetime(dates))[0],
                          pd.DatetimeIndex(pd.to_datetime(dates))[-1],
                          allow_holdout=allow_holdout)
    if len(r) < 30:
        raise ValueError("bootstrap 需要至少 30 个有效日收益")
    if np.any(r <= -1):
        raise ValueError("日收益不能小于等于 -100%")
    n = len(r)
    rf_d = rf / TRADING_DAYS_PER_YEAR
    if block_len is None:
        block_len = polit_white_blocklen(r)
    block = max(2, int(round(block_len)))
    rng = np.random.default_rng(seed)

    if method == "monthly":
        if dates is None:
            raise ValueError("monthly 方法需要 dates")
        dt = pd.DatetimeIndex(pd.to_datetime(dates))[: len(r)]
        keys = pd.Series(dt).dt.to_period("M").astype(str).to_numpy()
        uniq = np.unique(keys)
        groups = {k: np.where(keys == k)[0] for k in uniq}
        order = sorted(groups)
        out = np.empty((n_boot, 5))
        for b in range(n_boot):
            pick = rng.choice(len(order), size=len(order), replace=True)
            idx = np.concatenate([groups[order[i]] for i in pick])
            out[b] = _bootstrap_stats_from_returns(r[idx], TRADING_DAYS_PER_YEAR, rf_d)
        used_block = float(np.mean([len(groups[k]) for k in order]))
    else:
        out = np.empty((n_boot, 5))
        for b in range(n_boot):
            if method == "stationary":
                idx = stationary_bootstrap_indices(n, block, rng)
            elif method == "circular":
                idx = circular_block_indices(n, block, rng)
            else:
                raise ValueError(f"未知 bootstrap 方法 {method}")
            out[b] = _bootstrap_stats_from_returns(r[idx], TRADING_DAYS_PER_YEAR, rf_d)
        used_block = float(block)

    cagr0, sharpe0, mdd0, calmar0, vol0 = _bootstrap_stats_from_returns(r, TRADING_DAYS_PER_YEAR, rf_d)
    alpha = (1 - ci) / 2
    names = ["cagr", "sharpe", "mdd", "calmar", "vol"]
    point = {"cagr": cagr0, "sharpe": sharpe0, "mdd": mdd0, "calmar": calmar0, "vol": vol0}
    res: dict[str, Any] = {
        "method": method,
        "block_len": used_block,
        "n_boot": n_boot,
        "n_obs": n,
        "seed": seed,
        "note": "对历史日收益的区块重采样区间，不是未来收益概率；不修复生存/选择偏差",
    }
    for i, nm in enumerate(names):
        col = out[:, i]
        col = col[np.isfinite(col)]
        res[f"{nm}_point"] = float(point[nm])
        res[f"{nm}_lo"] = float(np.quantile(col, alpha)) if len(col) else np.nan
        res[f"{nm}_hi"] = float(np.quantile(col, 1 - alpha)) if len(col) else np.nan
        res[f"{nm}_median"] = float(np.median(col)) if len(col) else np.nan
    res["p_cagr_le_0"] = float((out[:, 0] <= 0).mean())
    res["p_sharpe_le_0"] = float((out[:, 1] <= 0).mean())
    res["p_mdd_worse_than_-20pct"] = float((out[:, 2] <= -0.20).mean())
    return res


def rolling_bootstrap(returns: Any, window: int = 504, step: int = 63, **kw) -> pd.DataFrame:
    """滚动窗口 bootstrap：检验区间结论是否依赖单一起点。"""
    s = as_series(returns)
    n_boot = min(int(kw.pop("n_boot", 600)), 600)
    rows = []
    for end_i in range(window, len(s) + 1, step):
        seg = s.iloc[end_i - window : end_i]
        try:
            b = block_bootstrap(seg, n_boot=n_boot, **kw)
        except (ValueError, HoldoutViolation):
            continue
        rows.append(
            {
                "start": str(seg.index[0].date()),
                "end": str(seg.index[-1].date()),
                "cagr": b["cagr_point"],
                "cagr_lo": b["cagr_lo"],
                "sharpe": b["sharpe_point"],
                "sharpe_lo": b["sharpe_lo"],
                "mdd": b["mdd_point"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. 按日期聚类的推断
# ---------------------------------------------------------------------------


def cluster_mean_test(values: Any, clusters: Any, min_clusters: int = 2) -> dict[str, Any]:
    """单样本均值检验，按 cluster 聚类（CR1 方差 + t 分布）。

    用于“同一天多个事件/多只股票”的情形：同日事件高度相关，
    朴素 SE 会把有效样本量高估。cluster 通常是交易日。
    """
    x = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    ok = np.isfinite(x)
    x, c = x[ok], c[ok]
    n = len(x)
    if n < 3:
        return {"n": n, "error": "样本不足"}
    G = len(np.unique(c))
    if G < min_clusters:
        return {"n": n, "clusters": G, "error": "聚类数不足"}
    mean = float(np.mean(x))
    e = x - mean
    naive_se = float(np.std(x, ddof=1) / math.sqrt(n))
    df = pd.DataFrame({"e": e, "c": c})
    sums = df.groupby("c", observed=True)["e"].sum().to_numpy()
    cr1 = float((G / (G - 1)) * (n / max(n - 1, 1)) * np.sum(sums**2)) / (n**2)
    se = math.sqrt(max(cr1, 0.0))
    t = mean / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=G - 1)) if np.isfinite(t) else np.nan
    return {
        "n": n,
        "clusters": G,
        "mean": mean,
        "naive_se": naive_se,
        "cluster_se": se,
        "se_ratio": se / naive_se if naive_se > 0 else np.nan,
        "t": float(t),
        "p": p,
        "ci_lo": mean - stats.t.ppf(0.975, G - 1) * se,
        "ci_hi": mean + stats.t.ppf(0.975, G - 1) * se,
        "df": G - 1,
    }


def cluster_diff_test(a: Any, b: Any, clusters: Any | None = None) -> dict[str, Any]:
    """配对差值检验，可按 cluster 聚类。a、b 须同日历对齐。"""
    sa, sb = as_series(a), as_series(b)
    idx = sa.index.intersection(sb.index)
    d = (sa.loc[idx] - sb.loc[idx]).to_numpy(float)
    if clusters is None:
        clusters = idx.to_numpy()
    c = np.asarray(clusters)
    if len(c) != len(d):
        raise ValueError("clusters 长度与样本不一致")
    base = cluster_mean_test(d, c)
    # 补充：HAC(Newey-West) 方差，处理日期序列相关
    n = len(d)
    maxlag = int(4 * (n / 100) ** (2 / 9)) if n > 10 else 1
    dm = d - d.mean()
    s = float(np.dot(dm, dm) / n)
    for lag in range(1, maxlag + 1):
        w = 1 - lag / (maxlag + 1)
        s += 2 * w * float(np.dot(dm[:-lag], dm[lag:]) / n)
    nw_se = math.sqrt(max(s, 0) / n)
    base["newey_west_se"] = nw_se
    base["newey_west_t"] = float(d.mean() / nw_se) if nw_se > 0 else np.nan
    base["mean_diff"] = float(d.mean())
    return base


def event_cluster_test(
    forward_returns: Any,
    event_dates: Any,
    benchmark_returns: Any | None = None,
    n_boot: int = 2000,
    seed: int = 20261026,
    allow_holdout: bool = False,
) -> dict[str, Any]:
    """事件研究：同日多事件按日期聚类，给出均值、聚类 SE 与日期重采样 CI。

    forward_returns 为每个事件的持有期收益；event_dates 为其信号日。
    benchmark_returns 若给出，则先减去同期基准收益（超额事件收益）。
    """
    y = np.asarray(forward_returns, dtype=float)
    d = pd.DatetimeIndex(pd.to_datetime(event_dates))
    if len(d):
        guard_performance(d.min(), d.max(), allow_holdout=allow_holdout)
    ok = np.isfinite(y)
    y, d = y[ok], d[ok]
    if benchmark_returns is not None:
        b = as_series(benchmark_returns)
        b = b.reindex(d).ffill().fillna(0.0).to_numpy(float)
        y = y - b
    res = cluster_mean_test(y, d.to_numpy())
    uniq = np.unique(d)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    by_date = {k: y[d == k] for k in uniq}
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        means[i] = np.concatenate([by_date[k] for k in pick]).mean()
    res.update(
        {
            "boot_lo": float(np.quantile(means, 0.025)),
            "boot_hi": float(np.quantile(means, 0.975)),
            "p_boot_le_0": float((means <= 0).mean()),
            "n_boot": n_boot,
            "n_dates": int(len(uniq)),
        }
    )
    return res


# ---------------------------------------------------------------------------
# 8. 多重试验校正
# ---------------------------------------------------------------------------


def p_adjust(pvalues: Any, method: str = "bh") -> np.ndarray:
    """p 值校正：bonferroni / holm / bh（Benjamini-Hochberg FDR）。"""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * ranked)
        adj = np.minimum(adj, 1.0)
        out = np.empty(m)
        out[order] = adj
        return out
    if method == "bh":
        adj = np.minimum.accumulate(((m / (np.arange(m) + 1)) * ranked)[::-1])[::-1]
        adj = np.minimum(adj, 1.0)
        out = np.empty(m)
        out[order] = adj
        return out
    raise ValueError(f"未知校正方法 {method}")


def _bootstrap_t_matrix(
    R: np.ndarray,
    n_boot: int,
    block: float,
    seed: int,
    recenter: np.ndarray,
) -> np.ndarray:
    """对每列做平稳 bootstrap，返回 T×K 的 bootstrap t 统计矩阵。"""
    T, K = R.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, K))
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(T, block, rng)
        x = R[idx] - recenter
        mu = x.mean(axis=0)
        sd = x.std(axis=0, ddof=1)
        out[b] = np.where(sd > 0, math.sqrt(T) * mu / sd, 0.0)
    return out


def reality_check(
    R: Any,
    n_boot: int = 2000,
    block: float | None = None,
    seed: int = 20261026,
    studentized: bool = True,
) -> dict[str, Any]:
    """White (2000) Reality Check / Hansen (2005) SPA。

    H0：所有候选的期望超额收益都 <= 0。R 为 T×K 的超额日收益矩阵。
    SPA 用 max(0, mean) 重新中心化，剔除“明显差”的候选对分布的污染。
    """
    a = np.asarray(R, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if not np.isfinite(a).all():
        raise ValueError("收益矩阵含非有限值")
    T, K = a.shape
    if block is None:
        block = polit_white_blocklen(a[:, int(np.argmax(a.mean(axis=0)))])
    mu = a.mean(axis=0)
    sd = a.std(axis=0, ddof=1)
    t_obs = np.where(sd > 0, math.sqrt(T) * mu / sd, 0.0) if studentized else math.sqrt(T) * mu
    # SPA: 用 max(0, mu) 中心化；RC: 用 mu 中心化
    recenter = np.maximum(mu, 0.0)
    rc_center = mu
    boot_rc = _bootstrap_t_matrix(a, n_boot, block, seed, rc_center)
    boot_spa = _bootstrap_t_matrix(a, n_boot, block, seed + 1, recenter)
    stat_obs = float(np.max(t_obs)) if studentized else float(np.max(math.sqrt(T) * mu))
    rc = float((boot_rc.max(axis=1) >= stat_obs).mean())
    spa = float((boot_spa.max(axis=1) >= stat_obs).mean())
    return {
        "n_obs": T,
        "n_candidates": K,
        "block_len": float(block),
        "n_boot": n_boot,
        "best_mean_daily": float(np.max(mu)),
        "best_annualized": float(np.max(mu) * TRADING_DAYS_PER_YEAR),
        "observed_stat": stat_obs,
        "p_reality_check": float((rc * n_boot + 1) / (n_boot + 1)),
        "p_spa": float((spa * n_boot + 1) / (n_boot + 1)),
        "studentized": bool(studentized),
        "note": "零基准、单侧最大值检验；不能用它证明某候选未来有效",
    }


def romano_wolf(
    R: Any,
    n_boot: int = 2000,
    block: float | None = None,
    seed: int = 20261026,
) -> pd.DataFrame:
    """Romano-Wolf (2005) stepdown 最大 t 检验，控制族错误率。

    返回按 t 值降序的明细，含逐步 p 值与 BH 对照。
    """
    a = np.asarray(R, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    T, K = a.shape
    if block is None:
        block = polit_white_blocklen(a[:, int(np.argmax(a.mean(axis=0)))])
    mu = a.mean(axis=0)
    sd = a.std(axis=0, ddof=1)
    t_obs = np.where(sd > 0, math.sqrt(T) * mu / sd, 0.0)
    boot = _bootstrap_t_matrix(a, n_boot, block, seed, mu)
    order = np.argsort(-t_obs)
    p_step = np.empty(K)
    remaining = list(order)
    for j, k in enumerate(order):
        cols = remaining[j:]
        p_step[k] = float((boot[:, cols].max(axis=1) >= t_obs[k]).mean())
    # 单调化
    for j in range(1, K):
        p_step[order[j]] = max(p_step[order[j]], p_step[order[j - 1]])
    p_step = np.minimum(p_step, 1.0)
    df = pd.DataFrame(
        {
            "candidate": np.arange(K),
            "mean_daily": mu,
            "t": t_obs,
            "p_stepdown": p_step,
            "p_bh": p_adjust(np.clip(2 * stats.t.sf(np.abs(t_obs), df=T - 1), 0, 1), "bh"),
        }
    ).sort_values("t", ascending=False)
    df["reject_5pct"] = df["p_stepdown"] < 0.05
    df.attrs["block_len"] = float(block)
    df.attrs["n_boot"] = n_boot
    return df


def deflated_sharpe(
    returns: Any,
    trial_srs: Any | None = None,
    n_trials: int | None = None,
    sr_variance: float | None = None,
    rf: float = 0.0,
) -> dict[str, Any]:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio。

    输入的 SR 为**日频**（非年化）。trial_srs 为全部已尝试策略的日频 SR 向量，
    用于估计 SR 的横截面方差与试验数。缺省时须显式给 n_trials 与 sr_variance，
    否则 DSR 不可计算——不允许用 1 次试验冒充。
    """
    r = as_series(returns).to_numpy(float)
    T = len(r)
    if T < 30:
        return {"dsr": np.nan, "error": "样本不足 30"}
    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        return {"dsr": np.nan, "error": "零波动"}
    sr = float(np.mean(r) - rf / TRADING_DAYS_PER_YEAR) / sd
    if trial_srs is not None:
        ts = np.asarray(trial_srs, dtype=float)
        ts = ts[np.isfinite(ts)]
        N = max(int(n_trials or len(ts)), len(ts))
        var_sr = float(np.var(ts, ddof=1)) if len(ts) > 1 else 0.0
    else:
        if n_trials is None or sr_variance is None:
            raise ValueError("必须给 trial_srs，或同时给 n_trials 与 sr_variance")
        N = int(n_trials)
        var_sr = float(sr_variance)
    gamma = 0.5772156649015329
    if N <= 1 or var_sr <= 0:
        sr0 = 0.0
    else:
        sr0 = math.sqrt(var_sr) * (
            (1 - gamma) * stats.norm.ppf(1 - 1.0 / N)
            + gamma * stats.norm.ppf(1 - 1.0 / (N * math.e))
        )
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=False, bias=False))
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) * sr**2 / 4, 1e-15))
    z = (sr - sr0) * math.sqrt(T - 1) / denom
    dsr = float(stats.norm.cdf(z))
    # 最小可验证 track record（年数）
    if sr > sr0 and denom > 0:
        mintrl_days = 1 + (1 - g3 * sr + (g4 - 1) * sr**2 / 4) * (stats.norm.ppf(0.95) / (sr - sr0)) ** 2
    else:
        mintrl_days = np.inf
    return {
        "dsr": dsr,
        "sr_daily": sr,
        "sr_annual": sr * math.sqrt(TRADING_DAYS_PER_YEAR),
        "sr0_daily": sr0,
        "sr0_annual": sr0 * math.sqrt(TRADING_DAYS_PER_YEAR),
        "n_trials": N,
        "sr_variance": var_sr,
        "skew": g3,
        "kurtosis": g4,
        "z": float(z),
        "min_track_record_years": float(mintrl_days / TRADING_DAYS_PER_YEAR) if np.isfinite(mintrl_days) else np.inf,
        "note": "DSR 校正的是试验次数与收益非正态，不校正序列相关；须与区块 bootstrap 同看",
    }


def pbo_cscv(R: Any, blocks: int = 8, max_combinations: int = 5000) -> dict[str, Any]:
    """Bailey-Borwein-López de Prado-Zhu (2017) CSCV 过拟合概率 PBO。

    R 为 T×K 的候选日收益矩阵（列=参数组合）。把时间轴切成 S 个连续块，
    枚举一半做 IS、一半做 OOS；在 IS 选夏普最高者，看其 OOS 排名。
    PBO = 入选者 OOS 相对排名低于中位的组合占比。
    """
    a = np.asarray(R, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("需要 T×K，K>=2")
    T, K = a.shape
    if blocks % 2 or blocks < 4:
        raise ValueError("blocks 必须为 >=4 的偶数")
    if T < blocks * 10:
        raise ValueError("样本太短")
    chunks = np.array_split(np.arange(T), blocks)
    combos = list(itertools.combinations(range(blocks), blocks // 2))
    if len(combos) > max_combinations:
        rng = np.random.default_rng(20261026)
        sel = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in sorted(sel)]
        sampled = True
    else:
        sampled = False
    rows = []
    for comb in combos:
        tr = np.concatenate([chunks[k] for k in comb])
        te = np.concatenate([chunks[k] for k in range(blocks) if k not in comb])
        z, y = a[tr], a[te]
        sdz = z.std(axis=0, ddof=1)
        sdy = y.std(axis=0, ddof=1)
        sr_is = np.where(sdz > 0, z.mean(axis=0) / sdz, -np.inf)
        sr_oos = np.where(sdy > 0, y.mean(axis=0) / sdy, -np.inf)
        pick = int(np.argmax(sr_is))
        rank = float(stats.rankdata(sr_oos, method="average")[pick] / (K + 1))
        logit = math.log(rank / (1 - rank)) if 0 < rank < 1 else 0.0
        rows.append(
            {
                "is_blocks": "/".join(str(k + 1) for k in comb),
                "pick": pick,
                "oos_rank_pct": rank,
                "logit": logit,
                "sr_is": float(sr_is[pick]),
                "sr_oos": float(sr_oos[pick]),
            }
        )
    df = pd.DataFrame(rows)
    valid = df[~np.isclose(df["logit"], 0.0)]
    return {
        "pbo": float((valid["logit"] < 0).mean()) if len(valid) else np.nan,
        "n_combinations": int(len(df)),
        "n_valid": int(len(valid)),
        "n_sampled_combinations": sampled,
        "blocks": blocks,
        "n_candidates": K,
        "n_obs": T,
        "detail": df,
        "note": "PBO 是选择流程的诊断，不是策略未来亏损概率；并列组合已剔除",
    }


def multiple_testing_summary(
    R: Any,
    candidate_names: Sequence[str] | None = None,
    n_boot: int = 1000,
    seed: int = 20261026,
) -> dict[str, Any]:
    """把 RC/SPA、Romano-Wolf、BH 汇总成一份可落盘的多重试验报告。"""
    a = np.asarray(R, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    K = a.shape[1]
    names = list(candidate_names) if candidate_names is not None else [f"c{i}" for i in range(K)]
    rc = reality_check(a, n_boot=n_boot, seed=seed)
    rw = romano_wolf(a, n_boot=n_boot, seed=seed)
    rw["candidate"] = [names[i] for i in rw["candidate"]]
    return {"reality_check": rc, "romano_wolf": rw, "n_candidates": K}


# ---------------------------------------------------------------------------
# 9. 参数邻域与曲面
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "cagr", "sharpe", "mdd", "calmar", "vol", "turnover",
    "exposure", "win_rate", "profit_factor", "n_trades", "avg_hold_days",
]


def run_grid(
    adapter: StrategyAdapter,
    market: MarketData,
    grid: Mapping[str, Sequence[Any]],
    start: Any,
    end: Any,
    allow_holdout: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """在给定窗口跑参数网格。只在允许的窗口跑，结果不得用于未来选参。"""
    guard_performance(start, end, allow_holdout=allow_holdout)
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    rows = []
    t0 = time.time()
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        try:
            res = adapter.run(market, params, start, end)
            m = compute_metrics(res, allow_holdout=allow_holdout)
        except Exception as e:  # 单点失败不拖垮整张曲面
            m = {"error": f"{type(e).__name__}: {e}"}
        rows.append({**params, **{k: m.get(k, np.nan) for k in METRIC_COLS}, "error": m.get("error")})
        if verbose and (i == 1 or i % 10 == 0 or i == len(combos)):
            print(f"  grid {i}/{len(combos)} {params} -> "
                  f"cagr={m.get('cagr', float('nan')):.3f} sharpe={m.get('sharpe', float('nan')):.2f} "
                  f"[{time.time()-t0:.1f}s]", flush=True)
    return pd.DataFrame(rows)


def _param_codes(df: pd.DataFrame, param_cols: Sequence[str]) -> np.ndarray:
    codes = np.column_stack([pd.factorize(df[c], sort=True)[0] for c in param_cols])
    return codes.astype(int)


def surface_diagnostics(
    df: pd.DataFrame, param_cols: Sequence[str], metric: str = "sharpe"
) -> dict[str, Any]:
    """参数曲面诊断：平台度、峰孤立度、邻域衰减、正收益占比。

    关键量：
      best              网格最优
      median            网格中位（代表“随便选一组”的典型水平）
      plateau_ratio     中位/最优；接近 1 说明是平台而非尖峰
      peak_isolation    最优/邻域中位；>1.5 提示尖峰，过拟合风险高
      neighbor_drop     最优相对邻域中位的提升幅度
      positive_share    指标为正的参数占比
      rank_consistency  每点与邻域均值的 Spearman 相关（曲面平滑度）
    """
    d = df.dropna(subset=[metric]).reset_index(drop=True)
    if len(d) < 3:
        return {"error": "有效参数点不足 3 个"}
    codes = _param_codes(d, param_cols)
    y = d[metric].to_numpy(float)
    best = float(np.max(y))
    median = float(np.median(y))
    neighbor_med = np.full(len(d), np.nan)
    n_neighbors = np.zeros(len(d), dtype=int)
    for i in range(len(d)):
        dist = np.max(np.abs(codes - codes[i]), axis=1)
        mask = (dist <= 1) & (dist > 0)
        n_neighbors[i] = int(mask.sum())
        if mask.any():
            neighbor_med[i] = float(np.median(y[mask]))
    valid = np.isfinite(neighbor_med)
    best_i = int(np.argmax(y))
    iso_at_best = (
        float(best / neighbor_med[best_i])
        if np.isfinite(neighbor_med[best_i]) and neighbor_med[best_i] > 0 else np.nan
    )
    if valid.sum() >= 3 and np.std(neighbor_med[valid]) > 0 and np.std(y[valid]) > 0:
        rho = float(stats.spearmanr(y[valid], neighbor_med[valid]).statistic)
        drop = float((best - np.median(neighbor_med[valid])) / abs(best)) if best != 0 else np.nan
    else:
        rho = drop = np.nan
    return {
        "metric": metric,
        "n_points": int(len(d)),
        "best": best,
        "median": median,
        "worst": float(np.min(y)),
        "p25": float(np.quantile(y, 0.25)),
        "p75": float(np.quantile(y, 0.75)),
        "plateau_ratio": float(median / best) if best > 0 else np.nan,
        "best_minus_median": float(best - median),
        "frac_within_20pct_of_best": (
            float(np.mean(y >= best - 0.2 * abs(best))) if best != 0 else np.nan
        ),
        "peak_isolation": iso_at_best,
        "neighbor_drop": drop,
        "rank_consistency": rho,
        "positive_share": float(np.mean(y > 0)),
        "best_params": {c: d.loc[int(np.argmax(y)), c] for c in param_cols},
        "n_neighbors_median": float(np.median(n_neighbors)),
    }


def surface_rank_consistency(
    df_a: pd.DataFrame, df_b: pd.DataFrame, param_cols: Sequence[str], metric: str = "sharpe"
) -> dict[str, Any]:
    """两个窗口的曲面秩一致性：参数好坏排序是否稳定。

    注意：窗口 b 必须是 <=2020 的验证窗口；本函数不做窗口选择，只做比较。
    """
    a = df_a.set_index(list(param_cols))[metric]
    b = df_b.set_index(list(param_cols))[metric]
    common = a.index.intersection(b.index)
    if len(common) < 4:
        return {"error": "共同参数点不足"}
    x, y = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    rho = float(stats.spearmanr(x, y).statistic)
    ka, kb = int(np.argmax(x)), int(np.argmax(y))
    top_a = set(np.argsort(-x)[: max(1, len(x) // 5)])
    top_b = set(np.argsort(-y)[: max(1, len(y) // 5)])
    return {
        "n_common": int(len(common)),
        "spearman": rho,
        "best_match": bool(ka == kb),
        "top20_overlap": float(len(top_a & top_b) / max(len(top_a), 1)),
        "best_a": {c: str(v) for c, v in zip(param_cols, np.atleast_1d(common[ka]))},
        "best_b": {c: str(v) for c, v in zip(param_cols, np.atleast_1d(common[kb]))},
        "note": "仅衡量排序稳定性，不构成跨窗口选参许可",
    }


# ---------------------------------------------------------------------------
# 10. 扰动检验
# ---------------------------------------------------------------------------


def _trade_notional(tr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    en = tr["entry_price"].to_numpy(float) * tr["shares"].to_numpy(float)
    ex = tr["exit_price"].to_numpy(float) * tr["shares"].to_numpy(float)
    return en, ex


def cost_sensitivity(
    res: BacktestResult,
    extra_bps: Sequence[float] = (0, 5, 10, 20, 40, 80),
    initial_cash: float | None = None,
) -> pd.DataFrame:
    """在基准成本之上再叠加双边成本 bps，重算净值与指标。

    额外成本按成交额计，买入加到入场日、卖出加到出场日，直接从净值扣除。
    """
    tr = res.trades
    nav = res.nav
    init = float(initial_cash if initial_cash is not None else res.meta.get("initial_cash", nav.iloc[0]))
    if len(tr) == 0:
        return pd.DataFrame()
    en, ex = _trade_notional(tr)
    ed = pd.to_datetime(tr["entry_date"])
    xd = pd.to_datetime(tr["exit_date"])
    rows = []
    for bps in extra_bps:
        delta = pd.Series(0.0, index=nav.index)
        for d, v in zip(ed, en):
            if d in delta.index:
                delta.loc[d] += v * bps / 1e4
        for d, v in zip(xd, ex):
            if d in delta.index:
                delta.loc[d] += v * bps / 1e4
        adj_nav = nav - delta.cumsum()
        adj_nav = adj_nav.clip(lower=1e-6)
        sub = BacktestResult(
            pd.DataFrame({"nav": adj_nav}, index=nav.index), tr, None, dict(res.meta)
        )
        m = compute_metrics(sub, initial_cash=init)
        rows.append({"extra_bps_each_side": bps, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "mdd": m["mdd"], "calmar": m["calmar"], "final_nav": m["final_nav"]})
    return pd.DataFrame(rows)


def rebuild_equity_excluding(res: BacktestResult, mask_exclude: np.ndarray) -> pd.Series:
    """剔除指定交易后重算净值：把该笔已实现盈亏从出场日净值中扣除。

    这是“利润集中度”诊断，不是可执行组合；未重估替代仓位。
    """
    nav = res.nav.copy()
    tr = res.trades
    if len(tr) == 0:
        return nav
    ex = pd.to_datetime(tr["exit_date"]).to_numpy()
    pnl = tr["pnl"].to_numpy(float)
    delta = pd.Series(0.0, index=nav.index)
    for d, p in zip(ex[mask_exclude], pnl[mask_exclude]):
        if d in delta.index:
            delta.loc[d] += p
    adj = nav - delta.cumsum()
    return adj.clip(lower=1e-6)


def drop_top_winners(
    res: BacktestResult, ks: Sequence[int] = (1, 3, 5, 10), initial_cash: float | None = None
) -> pd.DataFrame:
    """剔除盈利最大的 k 笔交易后重算指标，检验收益是否靠少数赢家。"""
    tr = res.trades
    if len(tr) == 0:
        return pd.DataFrame()
    init = float(initial_cash if initial_cash is not None else res.meta.get("initial_cash", res.nav.iloc[0]))
    order = np.argsort(-tr["pnl"].to_numpy(float))
    rows = [{"k": 0, **{k: v for k, v in compute_metrics(res, initial_cash=init).items()
                        if k in ("cagr", "sharpe", "mdd", "calmar", "total_return")}}]
    for k in ks:
        if k >= len(tr):
            break
        mask = np.zeros(len(tr), dtype=bool)
        mask[order[:k]] = True
        nav = rebuild_equity_excluding(res, mask)
        sub = BacktestResult(pd.DataFrame({"nav": nav}), tr[~mask], None, dict(res.meta))
        m = compute_metrics(sub, initial_cash=init)
        rows.append({"k": k, **{kk: m[kk] for kk in ("cagr", "sharpe", "mdd", "calmar", "total_return")}})
    return pd.DataFrame(rows)


def delay_reprice(
    res: BacktestResult, market: MarketData, delays: Sequence[int] = (1, 2, 3, 5)
) -> pd.DataFrame:
    """把成交推迟 k 个交易日，按当日开盘价重估每笔交易。

    这是交易层近似：保持股数不变、按新开盘价重算盈亏与成本，
    不重建替代仓位的资金路径。精确结论须由策略端以 delay 参数重跑。
    """
    tr = res.trades
    if len(tr) == 0:
        return pd.DataFrame()
    codes = sorted(tr["code"].unique())
    lo = pd.to_datetime(tr["entry_date"]).min() - pd.Timedelta(days=10)
    hi = min(pd.to_datetime(tr["exit_date"]).max() + pd.Timedelta(days=30), CUTOFF)
    bars = market.bars(lo, hi, codes)
    bars = price_limit_flags(bars)
    px = bars.set_index(["code", "date"])
    cal = market.trading_days(lo, hi)
    pos = {d: i for i, d in enumerate(cal)}

    def shift(d, k):
        i = pos.get(pd.Timestamp(d))
        if i is None:
            return pd.NaT
        j = min(i + k, len(cal) - 1)
        return cal[j]

    rows = []
    for k in delays:
        recs = []
        for _, t in tr.iterrows():
            nd_en, nd_ex = shift(t["entry_date"], k), shift(t["exit_date"], k)
            try:
                pe = px.loc[(t["code"], nd_en)]
                pxx = px.loc[(t["code"], nd_ex)]
            except KeyError:
                continue
            if not bool(pe["tradable"]) or not bool(pxx["tradable"]):
                continue
            en_px = float(pe["open"]) if np.isfinite(pe["open"]) else float(pe["close"])
            ex_px = float(pxx["open"]) if np.isfinite(pxx["open"]) else float(pxx["close"])
            sh = float(t["shares"])
            bc = DEFAULT_COST.buy_cost(en_px * sh)
            sc = DEFAULT_COST.sell_cost(ex_px * sh, nd_ex)
            pnl = (ex_px - en_px) * sh - bc - sc
            recs.append({"ret": pnl / (en_px * sh + bc), "pnl": pnl})
        if recs:
            d = pd.DataFrame(recs)
            rows.append({"delay_days": k, "n_trades": len(d), "avg_ret": float(d["ret"].mean()),
                         "win_rate": float((d["ret"] > 0).mean()),
                         "total_pnl": float(d["pnl"].sum()),
                         "median_ret": float(d["ret"].median())})
    return pd.DataFrame(rows)


def ranking_perturbation(
    adapter: StrategyAdapter,
    market: MarketData,
    base_params: Mapping[str, Any],
    sigmas: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    n_draws: int = 10,
    start: Any = None,
    end: Any = None,
    seed: int = 20261026,
) -> pd.DataFrame:
    """排序扰动：给打分加 N(0, sigma*std(score)) 噪声后重跑。

    需要策略端支持 ``score_noise`` 参数；否则无法诚实模拟，直接报错而非伪造。
    """
    if not adapter.supports("score_noise"):
        raise NotImplementedError("该策略未实现 score_noise 扰动；请在适配器中支持后重跑")
    rng = np.random.default_rng(seed)
    rows = []
    for sg in sigmas:
        reps = 1 if sg == 0 else n_draws
        vals = []
        for i in range(reps):
            p = dict(base_params)
            p["score_noise"] = float(sg)
            p["score_seed"] = int(rng.integers(0, 2**31 - 1))
            res = adapter.run(market, p, start, end)
            vals.append(compute_metrics(res))
        d = pd.DataFrame([{k: v[k] for k in ("cagr", "sharpe", "mdd", "calmar")} for v in vals])
        rows.append({"sigma": sg, "n_draws": reps, "cagr_mean": d["cagr"].mean(),
                     "cagr_p10": d["cagr"].quantile(0.1), "cagr_p90": d["cagr"].quantile(0.9),
                     "sharpe_mean": d["sharpe"].mean(), "sharpe_p10": d["sharpe"].quantile(0.1),
                     "mdd_mean": d["mdd"].mean()})
    return pd.DataFrame(rows)


def rolling_start_table(
    res: BacktestResult, start_years: Sequence[int] | None = None,
    horizons: Sequence[int] = (1, 2, 3, 5), initial_cash: float | None = None,
) -> pd.DataFrame:
    """滚动起点：每个起点年、每个持有年限单独评估，检验结论是否依赖起点。"""
    nav = res.nav
    init = float(initial_cash if initial_cash is not None else res.meta.get("initial_cash", nav.iloc[0]))
    if start_years is None:
        start_years = list(range(nav.index[0].year, nav.index[-1].year + 1))
    rows = []
    for y in start_years:
        s = nav[nav.index >= pd.Timestamp(f"{y}-01-01")]
        if len(s) < 30:
            continue
        base = float(s.iloc[0])
        for h in horizons:
            e = s[s.index < pd.Timestamp(f"{y + h}-01-01")]
            if len(e) < 30 or e.index[-1] >= CUTOFF:
                continue
            seg = e
            path = np.r_[base, seg.to_numpy(float)]
            yrs = max((seg.index[-1] - seg.index[0]).days, 1) / DAYS_PER_YEAR
            tot = seg.iloc[-1] / base - 1
            r = seg.pct_change().to_numpy(float).copy()
            r[0] = seg.iloc[0] / base - 1
            sd = np.std(r, ddof=1)
            rows.append({
                "start_year": y, "horizon_years": h,
                "start": str(seg.index[0].date()), "end": str(seg.index[-1].date()),
                "total_return": float(tot),
                "cagr": float((1 + tot) ** (1 / yrs) - 1),
                "mdd": max_drawdown(path)[0],
                "sharpe": float(np.mean(r) / sd * math.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def subperiod_table(
    res: BacktestResult, breaks: Sequence[str] = ("2008-01-01", "2011-01-01", "2014-01-01", "2017-01-01", "2020-01-01"),
    initial_cash: float | None = None,
) -> pd.DataFrame:
    """按制度/年份切子区间，各自独立评估。"""
    nav = res.nav
    init = float(initial_cash if initial_cash is not None else res.meta.get("initial_cash", nav.iloc[0]))
    edges = [pd.Timestamp(b) for b in breaks] + [nav.index[-1] + pd.Timedelta(days=1)]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = nav[(nav.index >= a) & (nav.index < b)]
        if len(seg) < 30:
            continue
        prev = nav[nav.index < seg.index[0]]
        base = float(prev.iloc[-1]) if len(prev) else init
        path = np.r_[base, seg.to_numpy(float)]
        yrs = max((seg.index[-1] - seg.index[0]).days, 1) / DAYS_PER_YEAR
        tot = seg.iloc[-1] / base - 1
        r = seg.pct_change().to_numpy(float).copy()
        r[0] = seg.iloc[0] / base - 1
        sd = np.std(r, ddof=1)
        rows.append({
            "start": str(seg.index[0].date()), "end": str(seg.index[-1].date()),
            "total_return": float(tot), "cagr": float((1 + tot) ** (1 / yrs) - 1),
            "mdd": max_drawdown(path)[0],
            "sharpe": float(np.mean(r) / sd * math.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 10b. 容量与一键稳健性报告
# ---------------------------------------------------------------------------


def capacity_estimate(
    res: BacktestResult,
    market: MarketData,
    participation: float = 0.05,
    lookback: int = 20,
) -> dict[str, Any]:
    """容量估计：把每笔交易的入场名义额换算成“达到参与率上限所需的资金”。

    单笔容量 Cap_i = ADTV_i(入场前 lookback 日均额) * participation / w_i，
    w_i 为该笔入场时的仓位权重。组合容量取 10% 分位（最紧约束）。
    这是线性近似上界，不含冲击成本二次项与涨跌停时的流动性真空。
    """
    tr = res.trades
    if len(tr) == 0:
        return {"error": "无交易"}
    nav = res.nav
    codes = sorted(tr["code"].unique())
    lo = pd.to_datetime(tr["entry_date"]).min() - pd.Timedelta(days=lookback * 3 + 10)
    hi = min(pd.to_datetime(tr["exit_date"]).max() + pd.Timedelta(days=5), CUTOFF)
    bars = market.bars(lo, hi, codes).sort_values(["code", "date"])
    bars["adtv"] = bars.groupby("code", observed=True)["amount"].transform(
        lambda x: x.rolling(lookback, min_periods=max(3, lookback // 4)).mean()
    )
    key = bars.set_index(["code", "date"])["adtv"]
    ed = pd.to_datetime(tr["entry_date"])
    adtv = key.reindex(pd.MultiIndex.from_arrays([tr["code"].to_numpy(), ed.to_numpy()])).to_numpy(float)
    notional = (tr["entry_price"] * tr["shares"]).to_numpy(float)
    nav_at = nav.reindex(ed).to_numpy(float)
    w = np.where(np.isfinite(nav_at) & (nav_at > 0), notional / nav_at, np.nan)
    ok = np.isfinite(adtv) & np.isfinite(w) & (w > 0) & (adtv > 0)
    if ok.sum() == 0:
        return {"error": "无有效 ADTV 匹配"}
    cap = adtv[ok] * participation / w[ok]
    part_now = notional[ok] / adtv[ok]
    return {
        "n_trades_used": int(ok.sum()),
        "participation_limit": participation,
        "lookback_days": lookback,
        "capital_at_limit_p10": float(np.quantile(cap, 0.10)),
        "capital_at_limit_median": float(np.median(cap)),
        "capital_at_limit_p90": float(np.quantile(cap, 0.90)),
        "entry_participation_median": float(np.median(part_now)),
        "entry_participation_p90": float(np.quantile(part_now, 0.90)),
        "note": "线性近似；未计冲击成本平方项，未计涨跌停/停牌时的流动性真空",
    }


def robustness_report(
    res: BacktestResult,
    market: MarketData,
    benchmark: pd.Series | None = None,
    adapter: StrategyAdapter | None = None,
    grid: Mapping[str, Sequence[Any]] | None = None,
    grid_window: tuple[Any, Any] | None = None,
    n_boot: int = 2000,
    allow_holdout: bool = False,
    wf_train_years: int = 3,
    wf_test_years: int = 1,
) -> dict[str, Any]:
    """对一个回测结果跑完整套稳健性检查，返回可直接落盘的字典。

    adapter/grid 给出时，额外跑参数曲面与 walk-forward（仅限 <=2020）。
    """
    guard_performance(res.nav.index[0], res.nav.index[-1], allow_holdout=allow_holdout)
    out: dict[str, Any] = {
        "contract": validate_result(res, strict=False),
        "metrics": compute_metrics(res, market=market, benchmark=benchmark,
                                   allow_holdout=allow_holdout),
        "annual": annual_table(res).to_dict("records"),
        "monthly": monthly_table(res).to_dict("records"),
        "bootstrap": {
            "stationary": block_bootstrap(res.returns, n_boot=n_boot, method="stationary",
                                          dates=res.nav.index),
            "circular": block_bootstrap(res.returns, n_boot=n_boot, method="circular",
                                        dates=res.nav.index),
            "monthly": block_bootstrap(res.returns, n_boot=max(n_boot // 2, 200),
                                       method="monthly", dates=res.nav.index),
        },
        "drop_top_winners": drop_top_winners(res, ks=(1, 3, 5, 10)).to_dict("records"),
        "cost_sensitivity": cost_sensitivity(res).to_dict("records"),
        "delay": delay_reprice(res, market).to_dict("records"),
        "rolling_start": rolling_start_table(res).to_dict("records"),
        "subperiod": subperiod_table(res).to_dict("records"),
        "capacity": capacity_estimate(res, market),
        "deflated_sharpe": deflated_sharpe(
            res.returns,
            n_trials=max(int(res.meta.get("n_trials", 1)), 1),
            sr_variance=float(res.meta.get("sr_variance", 0.0)),
        ) if res.meta.get("n_trials") else {"dsr": np.nan, "error": "未登记试验数，DSR 不可计算"},
    }
    if adapter is not None and grid and grid_window is not None:
        gs, ge = grid_window
        guard_performance(gs, ge, allow_holdout=allow_holdout)
        surf = run_grid(adapter, market, grid, gs, ge, allow_holdout=allow_holdout, verbose=False)
        out["surface"] = {**surface_diagnostics(surf, list(grid)), "n_rows": int(len(surf))}
        cand = []
        for _, row in surf.iterrows():
            params = dict(row[list(grid)])
            cand.append(adapter.run(market, params, gs, ge).returns)
        Rm = pd.concat(cand, axis=1).fillna(0.0)
        mt = multiple_testing_summary(Rm, n_boot=min(n_boot, 1000))
        mt["romano_wolf"] = mt["romano_wolf"].to_dict("records")
        out["multiple_testing"] = mt
        pb = pbo_cscv(Rm, blocks=8)
        out["pbo"] = {k: v for k, v in pb.items() if k != "detail"}
        span_years = (pd.Timestamp(ge) - pd.Timestamp(gs)).days / 365.25
        if span_years > wf_train_years + wf_test_years:
            wf = walk_forward(adapter, market, grid, gs, ge,
                              train_years=wf_train_years, test_years=wf_test_years,
                              allow_holdout=allow_holdout, verbose=False)
            out["walk_forward"] = {k: v for k, v in wf.items() if k != "oos_returns"}
        else:
            out["walk_forward"] = {"error": "窗口短于训练+测试跨度，跳过 walk-forward"}
    return out


# ---------------------------------------------------------------------------
# 11. 动态 walk-forward
# ---------------------------------------------------------------------------


def walk_forward(
    adapter: StrategyAdapter,
    market: MarketData,
    grid: Mapping[str, Sequence[Any]],
    start: Any,
    end: Any,
    train_years: int = 3,
    test_years: int = 1,
    mode: str = "expanding",
    embargo_days: int = 5,
    metric: str = "sharpe",
    min_train_days: int = 400,
    allow_holdout: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """动态 walk-forward：每折只用过去数据选参，再在紧接的测试段执行。

    mode='expanding' 训练窗从 start 起扩张；'rolling' 固定 train_years。
    embargo_days 在训练段尾部剔除，避免标签/持仓重叠泄漏。
    返回拼接后的 OOS 日收益与逐折选择记录。测试段之间不重叠、不留缝。
    """
    guard_performance(start, end, allow_holdout=allow_holdout)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    folds = []
    train_start = s
    test_start = s + pd.DateOffset(years=train_years)
    while test_start < e:
        test_end = min(test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1), e)
        tr_start = train_start if mode == "expanding" else max(
            train_start, test_start - pd.DateOffset(years=train_years)
        )
        # embargo：训练段在测试开始前截断
        tr_end = test_start - pd.Timedelta(days=1)
        cal = market.trading_days(tr_start, tr_end)
        if len(cal) > embargo_days:
            tr_end = cal[-(embargo_days + 1)]
        if len(market.trading_days(tr_start, tr_end)) < min_train_days:
            test_start = test_start + pd.DateOffset(years=test_years)
            continue
        folds.append((tr_start, tr_end, test_start, test_end))
        test_start = test_end + pd.Timedelta(days=1)

    if not folds:
        return {"error": "无法形成有效折", "folds": pd.DataFrame()}

    oos_parts: list[pd.Series] = []
    recs = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, 1):
        g = run_grid(adapter, market, grid, tr_s, tr_e, allow_holdout=allow_holdout, verbose=False)
        g = g.dropna(subset=[metric])
        if g.empty:
            continue
        best = g.loc[g[metric].idxmax()]
        params = {k: best[k] for k in grid}
        res = adapter.run(market, params, te_s, te_e)
        m = compute_metrics(res, allow_holdout=allow_holdout)
        if "cagr" not in m or len(res.equity) < 2:
            continue
        oos_parts.append(res.returns)
        recs.append({
            "fold": i, "train_start": str(pd.Timestamp(tr_s).date()), "train_end": str(pd.Timestamp(tr_e).date()),
            "test_start": str(pd.Timestamp(te_s).date()), "test_end": str(pd.Timestamp(te_e).date()),
            **{f"p_{k}": params[k] for k in grid},
            "train_metric": float(best[metric]), "test_cagr": m["cagr"],
            "test_sharpe": m["sharpe"], "test_mdd": m["mdd"],
        })
        if verbose:
            pretty = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
            print(f"  fold {i}: {pd.Timestamp(tr_s).date()}~{pd.Timestamp(tr_e).date()} -> "
                  f"{pd.Timestamp(te_s).date()}~{pd.Timestamp(te_e).date()} "
                  f"{pretty} test_cagr={m['cagr']:.3f}", flush=True)

    if not oos_parts:
        return {"error": "所有折均失败", "folds": pd.DataFrame(recs)}
    oos = pd.concat(oos_parts).sort_index()
    oos = oos[~oos.index.duplicated(keep="first")]
    path = np.cumprod(1 + oos.to_numpy(float))
    years = max(len(oos) / TRADING_DAYS_PER_YEAR, 1e-9)
    sd = float(np.std(oos, ddof=1))
    mdd = max_drawdown(np.r_[1.0, path])[0]
    cagr = float(path[-1] ** (1 / years) - 1)
    sel = pd.DataFrame(recs)
    param_cols = [f"p_{k}" for k in grid]
    switches = int((sel[param_cols].astype(str).apply(tuple, axis=1)
                    != sel[param_cols].astype(str).apply(tuple, axis=1).shift()).sum() - 1) if len(sel) > 1 else 0
    return {
        "folds": sel,
        "oos_returns": oos,
        "oos_metrics": {
            "cagr": cagr, "sharpe": float(oos.mean() / sd * math.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else np.nan,
            "mdd": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
            "n_days": int(len(oos)), "years": years,
        },
        "n_folds": len(sel),
        "param_switches": switches,
        "note": "逐折重选参数的真实 OOS 拼接；仍受限于参数网格本身的历史选择",
    }


# ---------------------------------------------------------------------------
# 12. 参考夹具引擎（仅用于把验证管线跑通，不是策略建议）
# ---------------------------------------------------------------------------


class TrendReferenceEngine(StrategyAdapter):
    """市场状态 + 流动性篮子的参考实现。

    **定位**：测试夹具。它存在的唯一目的是让契约、费用、T+1、涨跌停、
    容量与全部稳健性函数在一份真实数据上端到端跑通。它的收益数字不构成
    任何策略结论，也不得据此调参。

    规则（全部在收盘后计算，次日开盘执行）：
      * 准入：非 ST、上市满 250 日、收盘 > 2 元、20 日均额 >= min_amt；
      * 市场状态：沪深300 收盘 > MA(regime_ma) 时持仓，否则空仓；
      * 选股：按 20 日均额取前 n_pos 只（流动性排序，非收益排序）；
      * 调仓：每 rebalance_every 个交易日等权再平衡，跌出篮子即卖；
      * 仅在 -30% 处设灾难止损，不设个股趋势止盈止损。
    """

    name = "trend_reference_fixture"

    def __init__(self, cost: CostModel = DEFAULT_COST,
                 data_bounds: tuple[str, str] | None = None):
        self.cost = cost
        self._panel_cache: dict[tuple[str, str], dict[str, Any]] = {}
        # 固定数据区间后，多次网格/walk-forward 只加载与透视一次
        self.data_bounds = data_bounds

    def supports(self, perturbation: str) -> bool:
        return perturbation in ("score_noise", "delay")

    def default_grid(self) -> dict[str, list]:
        return {"regime_ma": [100, 150, 200, 250], "n_pos": [10, 20, 30]}

    # -- 数据准备 ---------------------------------------------------------
    def _panel(self, market: MarketData, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
        if self.data_bounds is not None:
            load_start, load_end = pd.Timestamp(self.data_bounds[0]), pd.Timestamp(self.data_bounds[1])
            key = ("bounds", self.data_bounds[0], self.data_bounds[1])
        else:
            load_start = start - pd.Timedelta(days=420)
            load_end = end
            key = (str(start.date()), str(end.date()))
        if key in self._panel_cache:
            return self._panel_cache[key]
        bars = market.bars(load_start, min(load_end, CUTOFF))
        bars = bars[bars["date"] <= CUTOFF]
        bars = price_limit_flags(bars)
        codes = np.array(sorted(bars["code"].unique()))
        dates = np.array(sorted(bars["date"].unique()))
        ci = {c: i for i, c in enumerate(codes)}
        di = {d: i for i, d in enumerate(dates)}
        shape = (len(dates), len(codes))

        def mat(col, dtype=float):
            m = np.full(shape, np.nan, dtype=dtype)
            r = bars["date"].map(di).to_numpy()
            c = bars["code"].map(ci).to_numpy()
            m[r, c] = bars[col].to_numpy(dtype=dtype) if dtype is float else bars[col].to_numpy()
            return m

        close = mat("close")
        open_ = mat("open")
        preclose = mat("preclose")
        amount = np.nan_to_num(mat("amount"), nan=0.0)
        tradable = mat("tradable", dtype=bool)
        can_buy = mat("can_buy_open", dtype=bool)
        can_sell = mat("can_sell_open", dtype=bool)
        is_st = mat("is_st", dtype=float)

        amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
        listed = np.cumsum(tradable.astype(np.int32), axis=0)
        st_f = pd.DataFrame(is_st).ffill().fillna(0).to_numpy()

        out = dict(codes=codes, dates=pd.DatetimeIndex(dates), close=close, open=open_,
                   preclose=preclose, amount=amount, tradable=tradable, can_buy=can_buy,
                   can_sell=can_sell, amt20=amt20, listed=listed, is_st=st_f,
                   di=di)
        self._panel_cache[key] = out
        return out

    # -- 主循环 -----------------------------------------------------------
    def run(self, market: MarketData, params: Mapping[str, Any],
            start: Any, end: Any) -> BacktestResult:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        p = {
            "n_pos": 20, "regime_ma": 200, "rebalance_every": 20,
            "stop_loss": -0.30, "min_amt": 2e7, "min_price": 2.0,
            "min_listed": 250, "initial_cash": 1_000_000.0,
            "score_noise": 0.0, "score_seed": 0, "delay_days": 1,
            "use_market_filter": True,
        }
        p.update({k: v for k, v in params.items() if v is not None})
        pan = self._panel(market, start, end)
        dates, codes = pan["dates"], pan["codes"]
        di = pan["di"]
        i0 = int(np.searchsorted(dates, np.datetime64(start)))
        i1 = int(np.searchsorted(dates, np.datetime64(end), side="right")) - 1
        if i1 - i0 < 20:
            raise ValueError("回测窗口过短")


        # 市场状态：基准收盘 > 基准 MA(regime_ma)。基准缺失时视为开启。
        regime = np.ones(len(dates), dtype=bool)
        try:
            if not p["use_market_filter"]:
                raise ValueError("filter disabled")
            bench = market.benchmark("sh.000300", dates[0], dates[-1])
            bench = bench.reindex(pd.DatetimeIndex(dates)).ffill()
            rma = bench.rolling(int(p["regime_ma"]), min_periods=int(p["regime_ma"]) // 2).mean()
            regime = (bench > rma).fillna(False).to_numpy()
        except Exception:
            regime = np.ones(len(dates), dtype=bool)

        rng = np.random.default_rng(int(p["score_seed"]))
        cash = float(p["initial_cash"])
        positions: dict[int, dict[str, Any]] = {}
        trades: list[dict[str, Any]] = []
        eq_rows: list[dict[str, Any]] = []
        pending_sell: list[int] = []
        pending_buy: list[int] = []
        pending_buy_signal: dict[int, pd.Timestamp] = {}
        pending_signal: dict[int, pd.Timestamp] = {}
        last_reb: int | None = None
        signals: pd.DataFrame | None = None
        turnover_today = 0.0
        cost_today = 0.0

        def mark_nav(i: int, price_key: str = "close") -> float:
            px = pan[price_key][i]
            mv = 0.0
            for c, pos in positions.items():
                v = px[c]
                if not np.isfinite(v):
                    v = pos["last_px"]
                pos["last_px"] = float(v)
                mv += pos["shares"] * float(v)
            return cash + mv

        for i in range(i0, i1 + 1):
            # --- 开盘执行（订单来自 i-1 收盘） ---
            turnover_today = 0.0
            cost_today = 0.0
            if pending_sell:
                for c in pending_sell:
                    pos = positions.get(c)
                    if pos is None:
                        continue
                    if not pan["can_sell"][i, c]:
                        continue
                    px = pan["open"][i, c]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    notional = pos["shares"] * float(px)
                    fee = self.cost.sell_cost(notional, dates[i])
                    proceeds = notional - fee
                    cash += proceeds
                    cost_today += fee
                    turnover_today += notional
                    pnl = proceeds - pos["cost_basis"]
                    trades.append({
                        "code": str(codes[c]),
                        "signal_date": pos["signal_date"],
                        "exit_signal_date": pending_signal.pop(c, dates[i - 1] if i else dates[i]),
                        "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                        "exit_date": dates[i], "exit_price": float(px),
                        "shares": float(pos["shares"]), "pnl": float(pnl),
                        "ret": float(pnl / pos["cost_basis"]) if pos["cost_basis"] > 0 else np.nan,
                        "weight_at_entry": pos["weight_at_entry"],
                        "hold_days": int(i - pos["entry_i"]),
                        "reason_exit": pos["reason"],
                        "cost": float(pos["entry_fee"] + fee),
                    })
                    del positions[c]
                pending_sell = []

            if pending_buy:
                nav_now = mark_nav(i, "open")
                budget_each = nav_now / max(int(p["n_pos"]), 1)
                for c in pending_buy:
                    if c in positions or not pan["can_buy"][i, c]:
                        continue
                    px = pan["open"][i, c]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    budget = min(budget_each, cash * 0.98)
                    shares = int(budget / (float(px) * (1 + self.cost.total_bps(dates[i], "buy") / 1e4)) // 100 * 100)
                    if shares <= 0:
                        continue
                    notional = shares * float(px)
                    fee = self.cost.buy_cost(notional)
                    if notional + fee > cash:
                        shares -= 100
                        if shares <= 0:
                            continue
                        notional = shares * float(px)
                        fee = self.cost.buy_cost(notional)
                    cash -= notional + fee
                    cost_today += fee
                    turnover_today += notional
                    positions[c] = {
                        "shares": shares, "entry_price": float(px),
                        "entry_date": dates[i], "entry_i": i,
                        "cost_basis": notional + fee, "entry_fee": fee,
                        "weight_at_entry": notional / nav_now,
                        "last_px": float(pan["close"][i, c]) if np.isfinite(pan["close"][i, c]) else float(px),
                        "reason": "target",
                        "signal_date": pending_buy_signal.get(c, dates[i - 1] if i else dates[i]),
                    }
                pending_buy = []

            nav = mark_nav(i)
            eq_rows.append({
                "date": dates[i], "nav": nav, "cash": cash,
                "market_value": nav - cash, "n_positions": len(positions),
                "turnover": turnover_today, "cost": cost_today,
            })

            if i >= i1:
                break
            # --- 收盘生成下一日订单 ---
            px = pan["close"][i]
            elig = (
                pan["tradable"][i] & np.isfinite(px) & (px > p["min_price"])
                & (pan["listed"][i] >= p["min_listed"])
                & (pan["is_st"][i] == 0)
                & (pan["amt20"][i] >= p["min_amt"])
            )
            idx = np.flatnonzero(elig)
            held = set(positions)

            reb_every = max(int(p["rebalance_every"]), 1)
            if last_reb is None or (i - last_reb) >= reb_every:
                last_reb = i
                target: list[int] = []
                if regime[i] and len(idx):
                    s_score = pan["amt20"][i, idx].astype(float).copy()
                    if p["score_noise"] and p["score_noise"] > 0:
                        sd = float(np.std(s_score))
                        s_score = s_score + rng.normal(0, float(p["score_noise"]) * sd, size=len(s_score))
                    order = idx[np.argsort(-s_score, kind="stable")]
                    target = list(order[: int(p["n_pos"])])
                active_target: list[int] | None = target
            else:
                active_target = None

            sells: list[int] = []
            for c in held:
                pos = positions[c]
                reason = None
                if not regime[i] and active_target is None:
                    reason = "regime_off"
                elif px[c] / pos["entry_price"] - 1 <= p["stop_loss"]:
                    reason = "stop_loss"
                elif active_target is not None and c not in set(active_target):
                    reason = "left_basket"
                if reason:
                    positions[c]["reason"] = reason
                    sells.append(c)

            buys = [c for c in (active_target or []) if c not in held and c not in set(sells)]

            pending_sell = sells
            pending_buy = buys
            pending_signal = {c: dates[i] for c in sells}
            pending_buy_signal = {c: dates[i] for c in buys}

        eq = pd.DataFrame(eq_rows).set_index("date")
        tr = pd.DataFrame(trades)
        if len(tr):
            tr["hold_days"] = (pd.to_datetime(tr["exit_date"]) - pd.to_datetime(tr["entry_date"])).dt.days
        meta = {"initial_cash": float(p["initial_cash"]), "params": p,
                "engine": self.name, "cost_model": asdict(self.cost)}
        return BacktestResult(eq, tr, signals, meta)


# ---------------------------------------------------------------------------
# 13. 数据审计
# ---------------------------------------------------------------------------


def coverage_by_year(bs: BaoStockMarket, start: Any = "2005-01-01", end: Any = CUTOFF) -> pd.DataFrame:
    """逐年覆盖：只统计行数/标的数/可交易占比，不计算任何收益。"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for y in range(s.year, e.year + 1):
        df = bs._read_year(y)
        if not len(df):
            continue
        df = df[(df["date"] >= s) & (df["date"] <= e)]
        if not len(df):
            continue
        rows.append({
            "year": y,
            "symbols": int(df["code"].nunique()),
            "rows": int(len(df)),
            "trading_days": int(df["date"].nunique()),
            "tradable_share": float(df["tradable"].mean()),
            "suspended_rows": int((~df["tradable"]).sum()),
            "st_share": float((df["is_st"] == 1).mean()),
            "null_close": int(df["close"].isna().sum()),
            "null_preclose": int(df["preclose"].isna().sum()),
            "null_turn": int(df["turn"].isna().sum()),
            "first_date": str(df["date"].min().date()),
            "last_date": str(df["date"].max().date()),
        })
    return pd.DataFrame(rows)


def audit_quality(bs: BaoStockMarket, start: Any = "2005-01-01", end: Any = "2020-12-31") -> dict[str, Any]:
    """2005-2020 的价格/口径质量检查（留出期不做收益相关计算）。"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    guard_performance(s, e)
    stats_acc: dict[str, float] = {}
    dup = 0
    ohlc_bad = 0
    ret_diff_max = 0.0
    ret_diff_n = 0
    n = 0
    extreme = 0
    extreme_ex_first = 0
    first_day_rows = 0
    limit_up = 0
    limit_dn = 0
    suspended = 0
    for y in range(s.year, e.year + 1):
        df = bs.bars(max(s, pd.Timestamp(f"{y}-01-01")), min(e, pd.Timestamp(f"{y}-12-31")))
        if not len(df):
            continue
        n += len(df)
        first_day_rows += int(df["first_day"].sum())
        valid = df["ret"].notna()
        extreme_ex_first += int((df.loc[valid & ~df["first_day"], "ret"].abs() > 0.55).sum())
        dup += int(df.duplicated(["code", "date"]).sum())
        ohlc_bad += int(
            ((df["high"] + 1e-9 < df[["open", "close"]].max(axis=1))
             | (df["low"] - 1e-9 > df[["open", "close"]].min(axis=1))).sum()
        )
        m = (df["pct_chg"].notna() & df["preclose"].notna() & (df["preclose"] > 0)
             & ~df["first_day"])
        if m.any():
            d = (df.loc[m, "close"] / df.loc[m, "preclose"] - 1 - df.loc[m, "pct_chg"] / 100).abs()
            ret_diff_max = max(ret_diff_max, float(d.max()))
            ret_diff_n += int((d > 1e-4).sum())
        r = df["ret"].to_numpy(float)
        extreme += int((np.abs(r) > 0.55).sum())
        suspended += int((~df["tradable"]).sum())
        fl = price_limit_flags(df)
        limit_up += int(fl["limit_up"].sum())
        limit_dn += int(fl["limit_down"].sum())

    return {
        "window": [str(s.date()), str(e.date())],
        "rows": int(n),
        "duplicate_keys": dup,
        "ohlc_violations": ohlc_bad,
        "ret_vs_pctchg_max_abs_diff": ret_diff_max,
        "ret_vs_pctchg_mismatch_gt_1e-4": int(ret_diff_n),
        "ret_vs_pctchg_mismatch_share": float(ret_diff_n / n) if n else np.nan,
        "extreme_abs_ret_gt_55pct": int(extreme),
        "extreme_ex_first_day": int(extreme_ex_first),
        "first_day_rows": int(first_day_rows),
        "suspended_rows": int(suspended),
        "suspended_share": float(suspended / n) if n else np.nan,
        "limit_up_rows": int(limit_up),
        "limit_down_rows": int(limit_dn),
        "limit_up_share": float(limit_up / n) if n else np.nan,
        "limit_down_share": float(limit_dn / n) if n else np.nan,
        "verdict": "通过" if (
            dup == 0 and ohlc_bad <= max(1, n * 1e-6) and ret_diff_n / max(n, 1) < 1e-5
        ) else "有问题",
        "note": "首日收益已置空；极端收益多为停牌复牌/重组复牌的真实缺口，未剔除但已单列",
    }


def audit_survivorship(bs: BaoStockMarket) -> dict[str, Any]:
    con = bs._connect()
    try:
        basic = pd.read_sql_query("SELECT code, code_name, ipo_date, out_date, status FROM stock_basic", con)
        cal = pd.read_sql_query(
            "SELECT calendar_date FROM trade_dates WHERE is_trading_day=1 "
            "AND calendar_date BETWEEN '2005-01-01' AND '2026-10-26'", con
        )
    finally:
        con.close()
    delisted = basic[basic["out_date"].notna() & (basic["out_date"] != "")]
    return {
        "n_basic": int(len(basic)),
        "n_delisted": int(len(delisted)),
        "n_listed": int((basic["status"] == 1).sum()),
        "delisted_with_out_date": int(len(delisted)),
        "delisted_share": float(len(delisted) / len(basic)),
        "trading_days": int(len(cal)),
        "note": "含退市股是生存偏差可控的前提；回测准入必须按当日可交易池，不得用当前上市名单",
    }


def audit_cutoff(bs: BaoStockMarket) -> dict[str, Any]:
    con = bs._connect()
    try:
        r = pd.read_sql_query(
            "SELECT adjustflag, MIN(date) mn, MAX(date) mx, COUNT(*) n, COUNT(DISTINCT code) codes "
            "FROM stock_kline_d GROUP BY adjustflag", con
        )
    finally:
        con.close()
    out = {"table": "stock_kline_d", "cutoff": str(CUTOFF.date()), "by_adjustflag": r.to_dict("records")}
    out["violates_cutoff"] = bool((pd.to_datetime(r["mx"]) > CUTOFF).any())
    out["verdict"] = "通过（读取端已截断）" if out["violates_cutoff"] else "通过"
    return out


def audit_cross_source(bs: BaoStockMarket, duck: DuckDBMarket, year: int = 2020,
                       sample: int = 300, seed: int = 20261026) -> dict[str, Any]:
    """同一交易日在两源间比对收盘价（仅 2020，避免留出期）。"""
    if year >= 2021:
        raise ValueError("交叉核对只允许 <=2020")
    s, e = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    b = bs.bars(s, e)
    d = duck.bars(s, e)
    d["code"] = d["code"].str.lower().str.replace(r"^(\d{6})\.(sh|sz|bj)$", r"\2.\1", regex=True)
    common = sorted(set(b["code"]) & set(d["code"]))
    rng = np.random.default_rng(seed)
    pick = list(rng.choice(common, size=min(sample, len(common)), replace=False)) if common else []
    m = b[b["code"].isin(pick)].merge(d[d["code"].isin(pick)], on=["code", "date"], suffixes=("_bs", "_th"))
    if not len(m):
        return {"error": "无重叠", "common_codes": len(common)}
    diff = (m["close_bs"] - m["close_th"]).abs()
    rel = diff / m["close_th"].abs().replace(0, np.nan)
    return {
        "year": year,
        "common_codes": len(common),
        "sampled_codes": len(pick),
        "matched_rows": int(len(m)),
        "close_max_abs_diff": float(diff.max()),
        "close_median_rel_diff": float(rel.median()),
        "share_rel_diff_gt_1pct": float((rel > 0.01).mean()),
        "verdict": "一致" if rel.median() < 1e-4 else "存在差异，需按复权口径复核",
    }


def run_data_audit(out_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    t0 = time.time()
    bs = BaoStockMarket()
    print("[audit] 覆盖统计 2005-2026 ...", flush=True)
    cov = coverage_by_year(bs)
    print(f"[audit] 覆盖完成 {len(cov)} 年 [{time.time()-t0:.0f}s]", flush=True)
    print("[audit] 质量检查 2005-2020 ...", flush=True)
    qual = audit_quality(bs)
    print(f"[audit] 质量完成 [{time.time()-t0:.0f}s]", flush=True)
    surv = audit_survivorship(bs)
    cut = audit_cutoff(bs)
    cross: dict[str, Any]
    try:
        cross = audit_cross_source(bs, DuckDBMarket())
    except Exception as exc:  # 交叉源不可用不应阻断主审计
        cross = {"error": f"{type(exc).__name__}: {exc}"}
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cutoff": str(CUTOFF.date()),
        "source_baostock": str(BAOSTOCK_DB),
        "source_duckdb": str(DUCKDB_PATH),
        "coverage": cov.to_dict("records"),
        "quality": qual,
        "survivorship": surv,
        "cutoff_check": cut,
        "cross_source": cross,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    dump_json(report, out_dir / "数据审计.json")
    cov.to_csv(out_dir / "数据覆盖_逐年.csv", index=False, encoding="utf-8-sig")
    print(f"[audit] 完成 -> {out_dir/'数据审计.json'}")
    return report


# ---------------------------------------------------------------------------
# 14. 静态自检
# ---------------------------------------------------------------------------


class SyntheticMarket(MarketData):
    """几何布朗运动合成行情，仅用于自检与推断校准。"""

    name = "synthetic"

    def __init__(self, n_codes: int = 40, start: str = "2010-01-01", years: int = 4,
                 drift: float = 0.0002, vol: float = 0.02, seed: int = 7):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start, periods=int(TRADING_DAYS_PER_YEAR * years))
        r = rng.normal(drift, vol, size=(len(dates), n_codes))
        px = 10 * np.exp(np.cumsum(r, axis=0))
        self.dates = dates
        self.codes = [f"sh.{600000+i:06d}" for i in range(n_codes)]
        self.close = pd.DataFrame(px, index=dates, columns=self.codes)
        self.ret = pd.DataFrame(r, index=dates, columns=self.codes)

    def trading_days(self, start: Any, end: Any) -> pd.DatetimeIndex:
        return self.dates[(self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))]

    def bars(self, start: Any, end: Any, codes: Sequence[str] | None = None) -> pd.DataFrame:
        c = list(codes) if codes is not None else self.codes
        sub = self.close.loc[(self.close.index >= pd.Timestamp(start)) & (self.close.index <= pd.Timestamp(end)), c]
        long = sub.reset_index().melt(id_vars="index", var_name="code", value_name="close")
        long = long.rename(columns={"index": "date"})
        long["open"] = long["close"]
        long["high"] = long["close"]
        long["low"] = long["close"]
        long["preclose"] = long.groupby("code", observed=True)["close"].shift(1)
        long["volume"] = 1e7
        long["amount"] = long["close"] * 1e7
        long["turn"] = 1.0
        long["trade_status"] = 1
        long["pct_chg"] = long["close"] / long["preclose"] - 1
        long["is_st"] = 0
        return _normalise_bars(long)

    def listed_codes(self, asof: Any | None = None) -> list[str]:
        return list(self.codes)

    def benchmark(self, code: str = "SYN", start: Any = "2005-01-01", end: Any = CUTOFF) -> pd.Series:
        r = self.ret.mean(axis=1)
        return (1 + r).cumprod()


def _check(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    t0 = time.time()
    try:
        detail = fn()
        return {"name": name, "passed": True, "detail": detail, "sec": round(time.time() - t0, 2)}
    except Exception as exc:
        return {"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3), "sec": round(time.time() - t0, 2)}


def run_selftest(out_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def t_drawdown():
        nav = np.array([100.0, 120.0, 90.0, 110.0, 130.0])
        mdd, pk, tr = max_drawdown(nav)
        assert abs(mdd - (90 / 120 - 1)) < 1e-12, mdd
        assert longest_underwater(nav) == 2
        return {"mdd": mdd, "peak": pk, "trough": tr}

    def t_cagr():
        r = 0.001
        n = 504
        nav = 1e6 * (1 + r) ** n
        idx = pd.bdate_range("2015-01-01", periods=n)
        res = BacktestResult(pd.DataFrame({"nav": nav * np.ones(n)}, index=idx), pd.DataFrame(),
                             None, {"initial_cash": 1e6})
        # 恒定净值序列没有日收益，年化应按首日跳升计算
        m = compute_metrics(res, initial_cash=1e6)
        assert abs(m["total_return"] - (nav / 1e6 - 1)) < 1e-9
        assert np.isfinite(m["cagr_trading"])
        return {"total_return": m["total_return"], "cagr": m["cagr"]}

    def t_sharpe():
        rng = np.random.default_rng(3)
        r = rng.normal(0.0005, 0.01, 1000)
        nav = 1e6 * np.cumprod(1 + r)
        idx = pd.bdate_range("2010-01-04", periods=1000)
        res = BacktestResult(pd.DataFrame({"nav": nav}, index=idx), pd.DataFrame(), None, {"initial_cash": 1e6})
        m = compute_metrics(res, initial_cash=1e6)
        manual = r.mean() / r.std(ddof=1) * math.sqrt(252)
        assert abs(m["sharpe"] - manual) < 0.02, (m["sharpe"], manual)
        return {"sharpe": m["sharpe"], "manual": manual}

    def t_guard():
        try:
            guard_performance("2020-01-01", "2021-06-30")
        except HoldoutViolation:
            guard_performance("2020-01-01", "2021-06-30", purpose="coverage_audit")
            return {"raised": True}
        raise AssertionError("未触发留出期护栏")

    def t_cost():
        c = CostModel()
        assert abs(c.stamp_rate("2023-08-27") - 0.001) < 1e-12
        assert abs(c.stamp_rate("2023-08-28") - 0.0005) < 1e-12
        b = c.buy_cost(1_000_000)
        s = c.sell_cost(1_000_000, "2023-08-28")
        assert s > b
        return {"buy_bps": b / 1e6 * 1e4, "sell_bps": s / 1e6 * 1e4}

    def t_limits():
        assert limit_ratio("sz.300001", "2020-08-21") == 0.10
        assert limit_ratio("sz.300001", "2020-08-24") == 0.20
        assert limit_ratio("sh.600001", "2010-01-04", is_st=True) == 0.05
        assert limit_ratio("sh.688001", "2020-01-02") == 0.20
        assert limit_ratio("bj.830001", "2022-01-04") == 0.30
        return {"ok": True}

    def t_padjust():
        p = np.array([0.01, 0.04, 0.03])
        bh = p_adjust(p, "bh")
        assert np.allclose(bh, [0.03, 0.04, 0.04]), bh
        holm = p_adjust(p, "holm")
        assert np.allclose(holm, [0.03, 0.06, 0.06]), holm
        bonf = p_adjust(p, "bonferroni")
        assert np.allclose(bonf, [0.03, 0.12, 0.09])
        assert np.all(p_adjust(p, "bh") >= p - 1e-12)
        return {"bh": bh.tolist(), "holm": holm.tolist()}

    def t_dsr():
        rng = np.random.default_rng(11)
        r = rng.normal(0.0006, 0.01, 1500)
        d1 = deflated_sharpe(r, n_trials=1, sr_variance=1e-6)
        d100 = deflated_sharpe(r, n_trials=100, sr_variance=1e-4)
        assert d1["dsr"] > d100["dsr"], (d1["dsr"], d100["dsr"])
        assert 0 <= d100["dsr"] <= 1
        return {"dsr_1": d1["dsr"], "dsr_100": d100["dsr"], "srn0": d100["sr0_annual"]}

    def t_pbo():
        rng = np.random.default_rng(5)
        noise = rng.normal(0, 0.01, size=(1000, 10))
        p0 = pbo_cscv(noise, blocks=8)["pbo"]
        sig = noise.copy()
        sig[:, 0] += 0.0008
        p1 = pbo_cscv(sig, blocks=8)["pbo"]
        assert 0.25 <= p0 <= 0.75, p0
        assert p1 < p0, (p0, p1)
        return {"pbo_noise": p0, "pbo_signal": p1}

    def t_cluster():
        rng = np.random.default_rng(9)
        n_dates, per = 150, 8
        date_eff = rng.normal(0, 0.03, n_dates)
        y = np.repeat(date_eff, per) + rng.normal(0, 0.01, n_dates * per)
        c = np.repeat(np.arange(n_dates), per)
        r = cluster_mean_test(y, c)
        assert r["se_ratio"] > 1.5, r
        assert r["p"] > 0.01
        return {"se_ratio": r["se_ratio"], "p": r["p"]}

    def t_bootstrap_coverage():
        rng = np.random.default_rng(21)
        cover = 0
        reps = 60
        for _ in range(reps):
            x = rng.normal(0.0004, 0.012, 400)
            b = block_bootstrap(x, n_boot=200, seed=int(rng.integers(1e6)))
            cover += int(b["cagr_lo"] <= b["cagr_point"] <= b["cagr_hi"])
        rate = cover / reps
        assert 0.80 <= rate <= 1.0, rate
        return {"coverage": rate, "reps": reps}

    def t_validate():
        idx = pd.bdate_range("2015-01-05", periods=5)
        eq = pd.DataFrame({"nav": np.linspace(1e6, 1.01e6, 5)}, index=idx)
        bad = pd.DataFrame({
            "code": ["sh.600000"], "signal_date": pd.to_datetime(["2015-01-05"]),
            "entry_date": pd.to_datetime(["2015-01-05"]), "entry_price": [10.0],
            "exit_date": pd.to_datetime(["2015-01-07"]), "exit_price": [11.0],
            "shares": [100.0], "pnl": [95.0], "ret": [0.095],
        })
        try:
            validate_result(BacktestResult(eq, bad))
        except ValueError as e:
            assert "T+1" in str(e)
            return {"caught": str(e)[:80]}
        raise AssertionError("未捕获 T+1 违规")

    def t_surface():
        grid = pd.DataFrame({
            "a": np.repeat([1, 2, 3], 3), "b": np.tile([1, 2, 3], 3),
            "sharpe": [1.0] * 9,
        })
        s1 = surface_diagnostics(grid, ["a", "b"])
        assert abs(s1["plateau_ratio"] - 1.0) < 1e-12
        spike = grid.copy()
        spike.loc[4, "sharpe"] = 3.0
        s2 = surface_diagnostics(spike, ["a", "b"])
        assert s2["peak_isolation"] > 2.0, s2
        return {"plateau": s1["plateau_ratio"], "isolation": s2["peak_isolation"]}

    def t_bootstrap_blocklen():
        rng = np.random.default_rng(4)
        n = 1200
        e = rng.normal(0, 1, n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + e[i]
        bl = polit_white_blocklen(x)
        assert 2 <= bl <= n / 3, bl
        return {"block_len": bl}

    def t_walkforward():
        syn = SyntheticMarket(n_codes=20, years=7, seed=13)
        class Dummy(StrategyAdapter):
            def run(self, market, params, start, end):
                d = market.trading_days(start, end)
                r = np.full(len(d), 0.0002 * float(params["k"]))
                nav = 1e6 * np.cumprod(1 + r)
                return BacktestResult(pd.DataFrame({"nav": nav}, index=d), pd.DataFrame(), None,
                                      {"initial_cash": 1e6})
        wf = walk_forward(Dummy(), syn, {"k": [1, 2, 3]}, "2012-01-01", "2015-12-31",
                          train_years=2, test_years=1, min_train_days=200, verbose=False)
        assert "error" not in wf, wf
        f = wf["folds"]
        assert (pd.to_datetime(f["train_end"]) < pd.to_datetime(f["test_start"])).all()
        return {"n_folds": wf["n_folds"], "oos_cagr": wf["oos_metrics"]["cagr"]}

    def t_delay():
        syn = SyntheticMarket(n_codes=5, years=1, seed=2)
        eng = TrendReferenceEngine()
        res = eng.run(syn, {"n_pos": 2, "min_amt": 0, "min_listed": 50,
                            "use_market_filter": False},
                      "2010-03-01", "2010-10-01")
        if len(res.trades) == 0:
            raise AssertionError("合成数据未产生交易")
        df = delay_reprice(res, syn, delays=(1, 2))
        assert len(df) >= 1
        return {"rows": len(df), "delays": df["delay_days"].tolist()}

    def t_drop_winners():
        rng = np.random.default_rng(6)
        n = 300
        idx = pd.bdate_range("2012-01-02", periods=n)
        nav = 1e6 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
        tr = pd.DataFrame({
            "code": ["sh.600000"] * 3,
            "entry_date": [idx[10], idx[100], idx[200]],
            "exit_date": [idx[20], idx[110], idx[210]],
            "entry_price": [10.0] * 3, "exit_price": [11.0, 9.0, 12.0],
            "shares": [1000.0] * 3, "pnl": [990.0, -1010.0, 1990.0], "ret": [0.099, -0.101, 0.199],
        })
        res = BacktestResult(pd.DataFrame({"nav": nav}, index=idx), tr, None, {"initial_cash": 1e6})
        d = drop_top_winners(res, ks=(1,))
        assert len(d) == 2
        return {"rows": len(d)}

    def t_power_detect():
        """功效：真实年化夏普约 1.0 的序列应被判为显著。"""
        rng = np.random.default_rng(31)
        r = rng.normal(0.0008, 0.0127, 1500)  # 日SR≈0.063 → 年化≈1.0
        b = block_bootstrap(r, n_boot=800, seed=7)
        d = deflated_sharpe(r, n_trials=1, sr_variance=1e-6)
        assert b["cagr_lo"] > 0, b["cagr_lo"]
        assert b["p_cagr_le_0"] < 0.05, b["p_cagr_le_0"]
        assert d["dsr"] > 0.90, d["dsr"]
        return {"cagr_lo": b["cagr_lo"], "p_le_0": b["p_cagr_le_0"], "dsr": d["dsr"]}

    def t_size_false_positive():
        """尺寸：零均值序列不应被判为显著。"""
        rng = np.random.default_rng(32)
        r = rng.normal(0.0, 0.01, 1500)
        b = block_bootstrap(r, n_boot=800, seed=8)
        d = deflated_sharpe(r, n_trials=20, sr_variance=1e-4)
        assert b["p_cagr_le_0"] > 0.20, b["p_cagr_le_0"]
        assert d["dsr"] < 0.95, d["dsr"]
        return {"p_le_0": b["p_cagr_le_0"], "dsr": d["dsr"]}

    def t_spa_power():
        """SPA：9 个噪声候选 + 1 个真实候选，应拒绝零假设。"""
        rng = np.random.default_rng(33)
        R = rng.normal(0, 0.01, size=(900, 10))
        R[:, 0] += 0.0012
        rc = reality_check(R, n_boot=600, seed=9)
        assert rc["p_spa"] < 0.10, rc["p_spa"]
        return {"p_spa": rc["p_spa"], "p_rc": rc["p_reality_check"]}

    for name, fn in [
        ("最大回撤与水下期", t_drawdown),
        ("累计收益口径", t_cagr),
        ("夏普手算一致", t_sharpe),
        ("留出期护栏", t_guard),
        ("费用与印花税分段", t_cost),
        ("涨跌停分段", t_limits),
        ("p值校正", t_padjust),
        ("DSR随试验数递减", t_dsr),
        ("PBO噪声≈0.5且信号更低", t_pbo),
        ("聚类方差膨胀", t_cluster),
        ("区块bootstrap覆盖率", t_bootstrap_coverage),
        ("T+1契约拦截", t_validate),
        ("曲面平台/尖峰判别", t_surface),
        ("最优区块长度", t_bootstrap_blocklen),
        ("walk-forward折不重叠", t_walkforward),
        ("延迟重估", t_delay),
        ("剔除最大赢家", t_drop_winners),
        ("功效：真实夏普1.0可检出", t_power_detect),
        ("尺寸：零均值不误判", t_size_false_positive),
        ("SPA：真实候选可检出", t_spa_power),
    ]:
        r = _check(name, fn)
        checks.append(r)
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {name} ({r['sec']}s)"
              + ("" if r["passed"] else f" -> {r.get('error')}"), flush=True)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_checks": len(checks),
        "n_passed": sum(c["passed"] for c in checks),
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
    dump_json(summary, out_dir / "自检_结果.json")
    return summary


# ---------------------------------------------------------------------------
# 15. 参考夹具端到端运行
# ---------------------------------------------------------------------------

BASE_PARAMS = {
    "n_pos": 20, "regime_ma": 200, "rebalance_every": 20,
    "stop_loss": -0.30, "min_amt": 2e7, "initial_cash": 1_000_000.0,
}


def run_reference(out_dir: Path = ARTIFACT_DIR, fast: bool = False) -> dict[str, Any]:
    """在开发窗口跑通参考夹具与全部验证函数，落盘所有产物。"""
    t0 = time.time()
    guard_performance(*WINDOWS["dev"])
    bs = BaoStockMarket()
    eng = TrendReferenceEngine(data_bounds=("2007-01-01", "2015-12-31"))
    s, e = WINDOWS["dev"]

    print("[ref] 加载并回测开发窗口 2008-2015 ...", flush=True)
    res = eng.run(bs, BASE_PARAMS, s, e)
    v = validate_result(res)
    m = compute_metrics(res, market=bs, benchmark=bs.benchmark("sh.000300", s, e))
    ann = annual_table(res)
    mon = monthly_table(res)
    mat = monthly_matrix(res)
    print(f"[ref] 交易 {m['n_trades']} 笔, CAGR {m['cagr']:.2%}, MDD {m['mdd']:.2%} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    res.equity.to_csv(out_dir / "参考夹具_净值.csv", encoding="utf-8-sig")
    res.trades.to_csv(out_dir / "参考夹具_交易.csv", index=False, encoding="utf-8-sig")
    ann.to_csv(out_dir / "参考夹具_年度.csv", index=False, encoding="utf-8-sig")
    mat.to_csv(out_dir / "参考夹具_月度矩阵.csv", encoding="utf-8-sig")

    print("[ref] 区块 bootstrap ...", flush=True)
    rets = res.returns
    boot = {
        "stationary": block_bootstrap(rets, n_boot=2000, method="stationary", dates=res.nav.index),
        "circular": block_bootstrap(rets, n_boot=2000, method="circular", dates=res.nav.index),
        "monthly": block_bootstrap(rets, n_boot=1000, method="monthly", dates=res.nav.index),
    }
    roll = rolling_bootstrap(rets, window=504, step=126, n_boot=300, method="stationary")

    print("[ref] 扰动检验 ...", flush=True)
    drop = drop_top_winners(res, ks=(1, 3, 5, 10))
    cost = cost_sensitivity(res, extra_bps=(0, 5, 10, 20, 40, 80))
    delay = delay_reprice(res, bs, delays=(1, 2, 3, 5))
    rs = rolling_start_table(res, horizons=(1, 2, 3))
    sub = subperiod_table(res)
    cap = capacity_estimate(res, bs)
    dump_json(cap, out_dir / "容量估计.json")
    drop.to_csv(out_dir / "扰动_剔除赢家.csv", index=False, encoding="utf-8-sig")
    cost.to_csv(out_dir / "扰动_费用.csv", index=False, encoding="utf-8-sig")
    delay.to_csv(out_dir / "扰动_延迟.csv", index=False, encoding="utf-8-sig")
    rs.to_csv(out_dir / "滚动起点.csv", index=False, encoding="utf-8-sig")
    sub.to_csv(out_dir / "子区间.csv", index=False, encoding="utf-8-sig")
    roll.to_csv(out_dir / "滚动bootstrap.csv", index=False, encoding="utf-8-sig")

    # 参数曲面：只用开发窗口的前段，避免把整段开发期都用于挑参
    gs, ge = pd.Timestamp("2008-01-01"), pd.Timestamp("2012-12-31")
    print(f"[ref] 参数曲面 {gs.date()}~{ge.date()} ...", flush=True)
    grid = {"regime_ma": [150, 200, 250], "n_pos": [10, 20, 30]}
    surf = run_grid(eng, bs, grid, gs, ge, verbose=True)
    surf.to_csv(out_dir / "参数曲面_开发前段.csv", index=False, encoding="utf-8-sig")
    diag = surface_diagnostics(surf, list(grid), "sharpe")

    # 参数曲面在验证段（2013-2015）的排序一致性；仍属开发窗口内部
    print("[ref] 曲面排序一致性 ...", flush=True)
    surf_b = run_grid(eng, bs, grid, pd.Timestamp("2013-01-01"), pd.Timestamp("2015-12-31"), verbose=False)
    surf_b.to_csv(out_dir / "参数曲面_开发后段.csv", index=False, encoding="utf-8-sig")
    rank_cons = surface_rank_consistency(surf, surf_b, list(grid), "sharpe")

    print("[ref] 买入事件按日期聚类检验 ...", flush=True)
    try:
        bench = bs.benchmark("sh.000300", s, e)
        ed = pd.to_datetime(res.trades["entry_date"])
        xd = pd.to_datetime(res.trades["exit_date"])
        b0 = bench.reindex(ed).ffill().to_numpy()
        b1 = bench.reindex(xd).ffill().to_numpy()
        excess = res.trades["ret"].to_numpy() - (b1 / b0 - 1)
        ev = event_cluster_test(excess, ed.to_numpy())
        dump_json(ev, out_dir / "买入事件_日期聚类.json")
    except Exception as exc:
        ev = {"error": f"{type(exc).__name__}: {exc}"}

    print("[ref] walk-forward 2009-2015 ...", flush=True)
    wf = walk_forward(eng, bs, grid, "2009-01-01", "2015-12-31",
                      train_years=3, test_years=1, mode="expanding",
                      embargo_days=10, metric="sharpe", verbose=True)
    if "folds" in wf and len(wf.get("folds", [])):
        wf["folds"].to_csv(out_dir / "walkforward_逐折.csv", index=False, encoding="utf-8-sig")
        wf["oos_returns"].to_csv(out_dir / "walkforward_OOS日收益.csv", encoding="utf-8-sig")

    # 多重试验：把参数网格当候选集，检验“最好参数”是否只是选择偏差
    print("[ref] 多重试验校正（用网格日收益近似）...", flush=True)
    try:
        cand = []
        for _, row in surf.iterrows():
            rr = eng.run(bs, {**BASE_PARAMS, "regime_ma": int(row["regime_ma"]),
                              "n_pos": int(row["n_pos"])}, gs, ge)
            cand.append(rr.returns)
        Rm = pd.concat(cand, axis=1).fillna(0.0)
        Rm.columns = [f"reg{int(a)}_n{int(b)}" for a, b in zip(surf["regime_ma"], surf["n_pos"])]
        mt = multiple_testing_summary(Rm, n_boot=1000)
        pbo = pbo_cscv(Rm, blocks=8)
        dsr = deflated_sharpe(Rm.iloc[:, int(np.argmax(Rm.mean()))], trial_srs=Rm.mean() / Rm.std(ddof=1))
        mt["romano_wolf"] = mt["romano_wolf"].to_dict("records")
        dump_json({"multiple_testing": mt, "pbo": {k: v for k, v in pbo.items() if k != "detail"},
                   "deflated_sharpe": dsr}, out_dir / "多重试验.json")
        Rm.to_csv(out_dir / "候选日收益矩阵.csv", encoding="utf-8-sig")
        pbo["detail"].to_csv(out_dir / "PBO_组合明细.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        mt, pbo, dsr = {}, {}, {}
        dump_json({"error": f"{type(exc).__name__}: {exc}"}, out_dir / "多重试验.json")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "windows": {k: [str(a.date()), str(b.date())] for k, (a, b) in WINDOWS.items()},
        "fixture_notice": "参考夹具仅用于验证管线，其收益不构成策略结论，禁止据此调参",
        "contract_check": v,
        "capacity": cap,
        "metrics": m,
        "bootstrap": boot,
        "surface": diag,
        "surface_rank_consistency": rank_cons,
        "walk_forward": {k: v for k, v in wf.items() if k not in ("oos_returns", "folds")},
        "walk_forward_folds": wf.get("folds"),
        "entry_event_study": ev,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    dump_json(summary, out_dir / "参考夹具_汇总.json")
    print(f"[ref] 完成，用时 {time.time()-t0:.0f}s -> {out_dir}")
    return summary


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "—"


def write_selftest_markdown(summary: dict[str, Any], path: Path) -> Path:
    lines = [
        "# 统计验证自检结果",
        "",
        f"- 生成时间：{summary['generated_at']}",
        f"- 通过：{summary['n_passed']}/{summary['n_checks']}",
        f"- 全通过：{'是' if summary['all_passed'] else '否'}",
        "",
        "| 检查 | 结果 | 用时(s) | 细节 |",
        "|---|---|---|---|",
    ]
    for c in summary["checks"]:
        detail = c.get("detail", c.get("error", ""))
        detail = json.dumps(detail, ensure_ascii=False, default=_json_default) if not isinstance(detail, str) else detail
        lines.append(f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['sec']} | {detail[:160]} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="统计与动态稳健性验证")
    ap.add_argument("cmd", choices=["selftest", "data-audit", "reference", "all"])
    ap.add_argument("--fast", action="store_true", help="缩短窗口/减少重采样（仅调试）")
    args = ap.parse_args(argv)

    if args.cmd in ("selftest", "all"):
        print("== 静态自检 ==")
        s = run_selftest()
        write_selftest_markdown(s, ARTIFACT_DIR / "自检_结果.md")
        print(f"自检通过 {s['n_passed']}/{s['n_checks']}")
        if not s["all_passed"]:
            print("存在失败项，详见 03_结果/统计验证/自检_结果.json")
            if args.cmd == "selftest":
                return 1
    if args.cmd in ("data-audit", "all"):
        print("== 数据审计 ==")
        run_data_audit()
    if args.cmd in ("reference", "all"):
        print("== 参考夹具端到端 ==")
        run_reference(fast=args.fast)
    return 0


if __name__ == "__main__":
    sys.exit(main())
