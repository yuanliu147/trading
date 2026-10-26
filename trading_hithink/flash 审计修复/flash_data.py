from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "初始研究"
FLASH = ROOT.parent / "trading_flash"
PANEL = FLASH / "2026-10-24_主板恐慌反转_年化30" / "01_数据" / "research_panel.parquet"
CORE = INITIAL / "data" / "work" / "core.parquet"
EVENTS = INITIAL / "data" / "snapshot" / "adjustment_events.parquet"
BAR_COLUMNS = [
    "open", "high", "low", "close",
    "hfq_open", "hfq_high", "hfq_low", "hfq_close",
    "volume", "reference_close", "previous_actual_close", "backward_factor",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    return json.loads((ROOT / "flash 审计修复" / "config" / "strategy.json").read_text())


def literal(path: Path) -> str:
    return str(path).replace("'", "''")


@dataclass(frozen=True)
class Candidate:
    symbol: int
    signal_index: int
    rank: int
    score: float
    amount20: float


@dataclass(frozen=True)
class CorporateAction:
    dividend_per_share: float
    per_share_bonus: float
    rights_ratio: float
    rights_price: float


@dataclass(frozen=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    hfq_open: float
    hfq_high: float
    hfq_low: float
    hfq_close: float
    volume: float
    reference_close: float
    previous_actual_close: float
    backward_factor: float
    factor_only_adjustment: bool


@dataclass
class FlashData:
    dates: pd.DatetimeIndex
    symbols: list[str]
    values: dict[str, np.ndarray]
    candidates: dict[int, list[Candidate]]
    events: dict[tuple[int, int], CorporateAction]
    source: dict[str, Any]

    def bar(self, date_index: int, symbol_index: int) -> Bar | None:
        close = self.values["close"][date_index, symbol_index]
        if not np.isfinite(close) or close <= 0:
            return None
        return Bar(
            **{name: float(self.values[name][date_index, symbol_index]) for name in BAR_COLUMNS},
            factor_only_adjustment=bool(self.values["factor_only_adjustment"][date_index, symbol_index]),
        )


def data_from_frames(
    daily: pd.DataFrame,
    candidates: pd.DataFrame,
    events: pd.DataFrame | None = None,
    sessions: pd.DatetimeIndex | None = None,
    source: dict[str, Any] | None = None,
) -> FlashData:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["thscode"] = frame["thscode"].astype(str)
    if frame.duplicated(["date", "thscode"]).any():
        raise ValueError("Duplicate date/thscode bars")
    candidate_frame = candidates.copy()
    candidate_frame["date"] = pd.to_datetime(candidate_frame["date"], errors="raise").dt.normalize()
    candidate_frame["thscode"] = candidate_frame["thscode"].astype(str)
    if candidate_frame.duplicated(["date", "thscode"]).any():
        raise ValueError("Duplicate candidate date/thscode")
    symbols = sorted(set(frame["thscode"]) | set(candidate_frame["thscode"]))
    if sessions is None:
        sessions = pd.DatetimeIndex(sorted(set(frame["date"]) | set(candidate_frame["date"])))
    else:
        sessions = pd.DatetimeIndex(sessions).normalize().sort_values()
    if sessions.empty:
        raise ValueError("No market sessions")
    di = {date: index for index, date in enumerate(sessions)}
    si = {symbol: index for index, symbol in enumerate(symbols)}
    shape = (len(sessions), len(symbols))
    values = {name: np.full(shape, np.nan, dtype=np.float64) for name in BAR_COLUMNS}
    values["factor_only_adjustment"] = np.zeros(shape, dtype=bool)
    for name in BAR_COLUMNS:
        if name not in frame:
            frame[name] = np.nan
    for name in ["hfq_open", "hfq_high", "hfq_low", "hfq_close"]:
        raw = name.removeprefix("hfq_")
        frame[name] = frame[name].where(frame[name].notna(), frame[raw])
    if "factor_only_adjustment" not in frame:
        frame["factor_only_adjustment"] = False
    frame = frame.sort_values(["thscode", "date"])
    if frame["reference_close"].isna().any():
        previous = frame.groupby("thscode")["close"].shift(1)
        frame["reference_close"] = frame["reference_close"].fillna(previous)
    if frame["previous_actual_close"].isna().any():
        previous = frame.groupby("thscode")["close"].shift(1)
        frame["previous_actual_close"] = frame["previous_actual_close"].fillna(previous)
    if frame["backward_factor"].isna().any():
        factor = frame["hfq_close"] / frame["close"]
        frame["backward_factor"] = frame["backward_factor"].fillna(factor)
    for name in BAR_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    has_bar = frame["close"].notna()
    if has_bar.any():
        located = frame.loc[has_bar]
        if (located[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("Nonpositive OHLC")
        if (located["high"] < located[["open", "close", "low"]].max(axis=1)).any():
            raise ValueError("Invalid high")
        if (located["low"] > located[["open", "close", "high"]].min(axis=1)).any():
            raise ValueError("Invalid low")
        if (located["backward_factor"] <= 0).any():
            raise ValueError("Invalid adjustment factor")
    if (frame["volume"].dropna() < 0).any():
        raise ValueError("Negative volume")
    unknown_dates = set(frame["date"]) - set(sessions)
    if unknown_dates:
        raise ValueError(f"Bars outside session calendar: {sorted(unknown_dates)[:3]}")
    d = frame["date"].map(di)
    if d.isna().any():
        raise ValueError("Unmapped bar date")
    s = frame["thscode"].map(si).astype(int)
    di_values = d.astype(int).to_numpy()
    si_values = s.to_numpy()
    for name in BAR_COLUMNS:
        values[name][di_values, si_values] = frame[name].to_numpy(dtype=float)
    values["factor_only_adjustment"][di_values, si_values] = frame["factor_only_adjustment"].fillna(False).astype(bool)
    candidate_lookup: dict[int, list[Candidate]] = {}
    if not candidate_frame.empty:
        if "rank" not in candidate_frame:
            if "score" not in candidate_frame:
                raise ValueError("Candidates need rank or score")
            candidate_frame["score"] = pd.to_numeric(candidate_frame["score"], errors="coerce")
            candidate_frame = candidate_frame.sort_values(["date", "score", "thscode"], ascending=[True, False, True])
            candidate_frame["rank"] = candidate_frame.groupby("date").cumcount() + 1
        candidate_frame["rank"] = candidate_frame["rank"].astype(int)
        candidate_frame = candidate_frame[candidate_frame["rank"] <= 10]
        candidate_frame["amount20"] = pd.to_numeric(candidate_frame["amount20"], errors="coerce")
        if not np.isfinite(candidate_frame["amount20"]).all() or (candidate_frame["amount20"] < 0).any():
            raise ValueError("Invalid candidate amount20")
        if "score" not in candidate_frame:
            candidate_frame["score"] = 0.0
        for date, group in candidate_frame.groupby("date", sort=True):
            if date not in di:
                raise ValueError(f"Candidate date outside market calendar: {date}")
            records = []
            for row in group.sort_values(["rank", "thscode"]).itertuples():
                if not np.isfinite(float(row.score)):
                    raise ValueError("Invalid candidate score")
                records.append(Candidate(si[row.thscode], di[date], int(row.rank), float(row.score), float(row.amount20)))
            candidate_lookup[di[date]] = records
    event_lookup: dict[tuple[int, int], CorporateAction] = {}
    if events is not None and not events.empty:
        event_frame = events.copy()
        event_frame["ex_date"] = pd.to_datetime(event_frame["ex_date"], errors="raise").dt.normalize()
        event_frame["thscode"] = event_frame["thscode"].astype(str)
        for name in ["dividend_per_share", "per_share_bonus", "rights_ratio", "rights_price"]:
            if name not in event_frame:
                event_frame[name] = 0.0
            event_frame[name] = pd.to_numeric(event_frame[name], errors="coerce").fillna(0.0)
            if (event_frame[name] < 0).any():
                raise ValueError(f"Negative corporate action field: {name}")
        grouped = event_frame.groupby(["ex_date", "thscode"], as_index=False)[
            ["dividend_per_share", "per_share_bonus", "rights_ratio", "rights_price"]
        ].sum()
        for row in grouped.itertuples(index=False):
            if row.thscode not in si or row.ex_date not in di:
                continue
            event_lookup[(di[row.ex_date], si[row.thscode])] = CorporateAction(
                float(row.dividend_per_share), float(row.per_share_bonus), float(row.rights_ratio), float(row.rights_price)
            )
    return FlashData(sessions, symbols, values, candidate_lookup, event_lookup, source or {})


def load_flash_data(config: dict[str, Any]) -> FlashData:
    settings = config["data"]
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit TO '4GB'")
    start, end = settings["start"], settings["end"]
    panel_sql = f"""
        WITH eligible AS (
            SELECT thscode, date, drawdown250, return_250d, rsi14, close_location,
                   market_return20, amount20
            FROM read_parquet('{literal(PANEL)}')
            WHERE date BETWEEN DATE '{start}' AND DATE '{end}'
              AND amount20 >= {config['signal']['minimum_amount20']}
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY date ORDER BY drawdown250, thscode
            ) AS rank, count(*) OVER () AS signal_count
            FROM eligible
            WHERE drawdown250 < {config['signal']['drawdown250_lt']}
              AND return_250d > {config['signal']['return_250d_gt']}
              AND rsi14 < {config['signal']['rsi14_lt']}
              AND close_location > {config['signal']['close_location_gt']}
              AND market_return20 < {config['signal']['market_return20_lt']}
        )
        SELECT thscode, date, -drawdown250 AS score, amount20, rank, signal_count
        FROM ranked WHERE rank <= {config['signal']['candidates_per_day']}
        ORDER BY date, rank, thscode
    """
    candidates = con.execute(panel_sql).fetchdf()
    candidates["date"] = pd.to_datetime(candidates["date"])
    stats = con.execute(
        f"""SELECT count(*) AS rows, count(DISTINCT thscode) AS symbols,
                   count(DISTINCT date) AS sessions, min(date) AS first_date, max(date) AS last_date
            FROM read_parquet('{literal(PANEL)}') WHERE date BETWEEN DATE '{start}' AND DATE '{end}'"""
    ).fetchdf().iloc[0].to_dict()
    expected = {
        "rows": settings["expected_panel_rows"],
        "symbols": settings["expected_panel_symbols"],
        "sessions": settings["expected_sessions"],
    }
    for key, value in expected.items():
        if int(stats[key]) != value:
            raise AssertionError(f"Panel {key}: {stats[key]} != {value}")
    if str(pd.Timestamp(stats["first_date"]).date()) != settings["expected_first_panel_date"] or str(pd.Timestamp(stats["last_date"]).date()) != end:
        raise AssertionError("Panel date boundary changed")
    if len(candidates) != settings["expected_candidate_rows"]:
        raise AssertionError("Candidate count changed")
    if int(candidates["signal_count"].iloc[0]) != settings["expected_signal_count"]:
        raise AssertionError("Signal count changed")
    codes = candidates[["thscode"]].drop_duplicates()
    if len(codes) != settings["expected_candidate_symbols"]:
        raise AssertionError("Candidate symbol count changed")
    con.register("selected_codes", codes)
    bars = con.execute(
        f"""SELECT p.thscode, p.date,
                   p.open, p.high, p.low, p.close,
                   p.hfq_open, p.hfq_high, p.hfq_low, p.hfq_close,
                   c.volume, c.reference_close, c.previous_actual_close, c.backward_factor,
                   c.factor_only_adjustment
            FROM read_parquet('{literal(PANEL)}') p
            JOIN read_parquet('{literal(CORE)}') c USING (thscode, date)
            JOIN selected_codes s ON s.thscode = p.thscode
            WHERE p.date BETWEEN DATE '{start}' AND DATE '{end}'
            ORDER BY p.date, p.thscode"""
    ).fetchdf()
    sessions = con.execute(
        f"SELECT DISTINCT date FROM read_parquet('{literal(PANEL)}') WHERE date BETWEEN DATE '{start}' AND DATE '{end}' ORDER BY date"
    ).fetchdf()
    factor_error = con.execute(
        f"""SELECT max(abs(reference_close - previous_actual_close * previous_backward_factor / backward_factor))
            FROM read_parquet('{literal(CORE)}') JOIN selected_codes s USING (thscode)
            WHERE previous_backward_factor IS NOT NULL AND backward_factor > 0"""
    ).fetchone()[0]
    con.close()
    if factor_error is not None and factor_error > 1e-8:
        raise AssertionError(f"Reference close mismatch: {factor_error}")
    symbols = sorted(codes["thscode"])
    event_frame = pd.read_parquet(EVENTS, filters=[("thscode", "in", symbols)])
    source = {
        "panel": str(PANEL),
        "core": str(CORE),
        "events": str(EVENTS),
        "panel_rows": int(stats["rows"]),
        "panel_symbols": int(stats["symbols"]),
        "signal_count": int(candidates["signal_count"].iloc[0]),
        "candidate_rows": int(len(candidates)),
        "candidate_symbols": int(len(codes)),
        "joined_bar_rows": int(len(bars)),
        "reference_price_max_error": float(factor_error or 0.0),
    }
    result = data_from_frames(
        bars,
        candidates,
        event_frame,
        sessions=pd.DatetimeIndex(sessions["date"]),
        source=source,
    )
    if len(result.symbols) != len(codes):
        raise AssertionError("Candidate bars missing")
    return result
