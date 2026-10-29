#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一数据适配层：冻结 Parquet → 点时面板 / 逐日 Bar / 公司行动账。

本文件只读 `01_数据/frozen`，不写源库、不改第一阶段文件。设计要点：

1. **价格两套口径**
   - `raw`：交易所真实价，用于成交、整股、涨跌停、费用；
   - `ca_adj = raw / ca_factor`：`ca_factor = Π(preclose/prev_close)` 只在该证券
     除权日跳变，因此 `ca_adj` 是“连续价格水平”，用于突破/MA/ATR/止损。
   - **不使用**冻结层的 `factor` 作为价格水平：它是逐日复利 `pct_chg` 的总收益指数，
     每天都被收益放大，当价格用会把累计分红漂移带进均线和止损。
   - `factor = Π(1+pct_chg)` 本身是总收益指数，`close × factor` 会重复计入每日收益；
     本层只把 `factor` 当总收益指数用（`tr_index`），不构造 `close×factor` 价格。

2. **公司行动以 preclose 为权威**
   - `corp_actions_implied` 给出每个除权日的 `adj_ratio = preclose/prev_close`；
   - 同花顺事件表只用于拆分现金分红/送转/配股，且仅当事件公式复原的参考价与
     `preclose` 相对误差 ≤0.5% 时采信；
   - 拆不开时按“全部现金”的总回报等价处理（股数不变），净值精确连续；
   - 复牌日 `preclose` 滞后的行不会产生 `adj_ratio != 1`，因此不会伪造公司行动。

3. **点时宇宙** `U-MAINBOARD-V1`
   - 主板、上市≥250自然日且≥120交易日、非ST、收盘≥2、ADV20≥3000万、
     非停牌、非复牌首日、退市整理期前30交易日不新开。
   - 退市日 `delist_date` 来自 `symbols.parquet`，用于两点：退市后强制退出、
     退市整理期过滤。后者是父预注册的规定，但使用了未来信息，必须在报告中披露。

4. **逐日 Bar 不在内存里常驻**
   - 面板用 [交易日 × 代码] 矩阵保存；`bars_on(i)` 现场构造 `Bar`，全窗口 700 万行
     的构造成本约 15 秒，避免 400 万级对象常驻。

对外接口
--------
    from 统一数据适配 import FrozenPanel, EventBook, WINDOWS
    eb = EventBook(); eb.load("2007-01-01", "2020-12-31")
    panel = FrozenPanel("2007-01-01", "2020-12-31", events=eb)
    bars = panel.bars_on(panel.index_of(date(2010, 1, 4)))
    elig = panel.eligible(panel.index_of(date(2010, 1, 4)))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "01_数据" / "frozen"
DB_PATH = FROZEN / "frozen.duckdb"

FREEZE = "2026-10-26"
WARM = ("2007-01-01", "2007-12-31")
DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")
HOLDOUT_START = "2021-01-01"
WINDOWS = {"preheat": WARM, "dev": DEV, "val": VAL}

BOARD = "主板"
MIN_LIST_CAL_DAYS = 250
MIN_LIST_TD = 120
MIN_PRICE = 2.0
MIN_ADV20 = 3.0e7
DELIST_BLOCK_TD = 30

# 面板字段（[day, code] 矩阵）
NUM_FIELDS = (
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "factor", "td_since_list", "limit_up_px", "limit_down_px", "float_mktcap",
    "bs_hfq_close",
)
BOOL_FIELDS = (
    "is_suspended", "is_st", "is_limit_up_open", "is_limit_down_open",
)


def _to_date(x) -> date:
    if isinstance(x, date):
        return x
    return pd.Timestamp(x).date()


@dataclass(frozen=True)
class UnifiedEvent:
    """一个已对账的除权日。金额单位：元/股；比例：每股。"""

    symbol: str
    ex_date: date
    prev_close: float
    preclose: float
    adj_ratio: float
    dividend: float
    bonus_ratio: float
    rights_ratio: float
    rights_price: float
    status: str          # verified_ths / ths_mismatch_cash / no_ths_cash / invalid
    ths_dividend: float = 0.0
    ths_bonus: float = 0.0
    ths_rights_ratio: float = 0.0
    ths_rights_price: float = 0.0

    @property
    def share_multiplier(self) -> float:
        return 1.0 + self.bonus_ratio + self.rights_ratio

    @property
    def is_pure_cash(self) -> bool:
        return self.status.endswith("_cash") or self.status == "invalid"

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "ex_date": self.ex_date.isoformat(),
            "prev_close": self.prev_close,
            "preclose": self.preclose,
            "adj_ratio": round(self.adj_ratio, 8),
            "dividend": round(self.dividend, 6),
            "bonus_ratio": self.bonus_ratio,
            "rights_ratio": self.rights_ratio,
            "rights_price": self.rights_price,
            "status": self.status,
            "ths_dividend": self.ths_dividend,
            "ths_bonus": self.ths_bonus,
            "ths_rights_ratio": self.ths_rights_ratio,
            "ths_rights_price": self.ths_rights_price,
        }


class EventBook:
    """除权日对账账本，以交易所 preclose 为权威。"""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = str(db_path)
        self.events: pd.DataFrame = pd.DataFrame()
        self.by_date: dict[date, list[UnifiedEvent]] = {}
        self.ths_only: pd.DataFrame = pd.DataFrame()
        self.stats: dict = {}

    # ------------------------------------------------------------------
    def load(self, start: str, end: str, con: duckdb.DuckDBPyConnection | None = None) -> "EventBook":
        start, end = str(start), str(end)
        if _to_date(end) >= _to_date(HOLDOUT_START):
            raise ValueError(f"EventBook 拒绝加载 2021+：{end}")
        own = con is None
        con = con or duckdb.connect(self.db_path, read_only=True)
        try:
            imp = con.execute(
                """
                SELECT symbol, ex_date, d_prev, prev_close, preclose, adj_ratio, has_th_event
                FROM corp_actions_implied
                WHERE ex_date BETWEEN ? AND ?
                ORDER BY symbol, ex_date
                """, [start, end]).df()
            act = con.execute(
                """
                SELECT symbol, ex_date,
                       sum(coalesce(dividend_per_share,0)) AS dividend,
                       sum(coalesce(per_share_bonus,0))   AS bonus,
                       sum(coalesce(rights_ratio,0))      AS rights_ratio,
                       CASE WHEN sum(coalesce(rights_ratio,0)) > 0
                            THEN sum(coalesce(rights_ratio,0)*coalesce(rights_price,0))
                                 / sum(coalesce(rights_ratio,0))
                            ELSE 0 END                     AS rights_price
                FROM corp_actions
                WHERE ex_date BETWEEN ? AND ?
                GROUP BY 1,2
                """, [start, end]).df()
            all_act = act.copy()
        finally:
            if own:
                con.close()

        m = imp.merge(act, on=["symbol", "ex_date"], how="left", indicator=True)
        m = m.rename(columns={"_merge": "merge_flag"})
        rows: list[UnifiedEvent] = []
        for r in m.itertuples(index=False):
            prev = float(r.prev_close) if pd.notna(r.prev_close) else np.nan
            pre = float(r.preclose) if pd.notna(r.preclose) else np.nan
            if not (np.isfinite(prev) and np.isfinite(pre) and prev > 0 and pre > 0):
                continue
            ratio = pre / prev
            exd = pd.Timestamp(r.ex_date).date()
            if not (0.02 <= ratio <= 50.0):
                rows.append(UnifiedEvent(r.symbol, exd, prev, pre, ratio,
                                         0.0, 0.0, 0.0, 0.0, "invalid"))
                continue
            has_act = r.merge_flag == "both"
            div = float(r.dividend) if has_act and pd.notna(r.dividend) else 0.0
            bon = float(r.bonus) if has_act and pd.notna(r.bonus) else 0.0
            rr = float(r.rights_ratio) if has_act and pd.notna(r.rights_ratio) else 0.0
            rp = float(r.rights_price) if has_act and pd.notna(r.rights_price) else 0.0
            status = "no_ths_cash"
            if has_act:
                denom = 1.0 + bon + rr
                ref = (prev - div + rp * rr) / denom if denom > 0 else np.nan
                tol = max(0.01, 0.005 * pre)
                if np.isfinite(ref) and abs(ref - pre) <= tol and denom > 0:
                    status = "verified_ths"
                else:
                    status = "ths_mismatch_cash"
            if status == "verified_ths":
                use_div, use_bon, use_rr, use_rp = div, bon, rr, rp
            else:
                use_div, use_bon, use_rr, use_rp = prev - pre, 0.0, 0.0, 0.0
            rows.append(UnifiedEvent(
                r.symbol, exd, prev, pre, ratio,
                use_div, use_bon, use_rr, use_rp, status,
                div, bon, rr, rp,
            ))

        ev = pd.DataFrame([x.as_dict() for x in rows])
        if len(ev):
            ev["ex_date"] = pd.to_datetime(ev["ex_date"]).dt.date
        self.events = ev
        self.by_date = {}
        for e in rows:
            self.by_date.setdefault(e.ex_date, []).append(e)

        # 事件表有、价格未调整：不应用，只统计。
        if len(all_act):
            n_both = int((m["merge_flag"] == "both").sum())
            left = all_act.merge(
                imp[["symbol", "ex_date"]].drop_duplicates(),
                on=["symbol", "ex_date"], how="left", indicator=True)
            self.ths_only = left[left["_merge"] == "left_only"].drop(columns=["_merge"]).copy()
            self._n_ths_matched = n_both
        else:
            self.ths_only = pd.DataFrame()
            self._n_ths_matched = 0
        self.stats = self._stats()
        return self

    # ------------------------------------------------------------------
    def _stats(self) -> dict:
        if not len(self.events):
            return {"n_events": 0}
        ev = self.events
        out = {
            "n_events": int(len(ev)),
            "n_symbols": int(ev["symbol"].nunique()),
            "by_status": ev["status"].value_counts().to_dict(),
            "by_year": ev.assign(y=pd.to_datetime(ev["ex_date"]).dt.year)
                          .groupby("y").size().to_dict(),
            "n_ths_only_not_applied": int(len(self.ths_only)),
            "n_ths_matched": int(getattr(self, "_n_ths_matched", 0)),
            "n_invalid": int((ev["status"] == "invalid").sum()),
        }
        for k in ("verified_ths", "ths_mismatch_cash", "no_ths_cash"):
            sub = ev[ev["status"] == k]
            out[f"{k}_share"] = round(len(sub) / len(ev), 6)
        return out

    # ------------------------------------------------------------------
    def on(self, d: date) -> list[UnifiedEvent]:
        return self.by_date.get(_to_date(d), [])

    def ratio_frame(self, dates: Sequence[date], codes: Sequence[str]) -> np.ndarray:
        """[day, code] 的除权比例矩阵（无事件=1），供 ca_factor 累乘。"""
        mat = np.ones((len(dates), len(codes)), dtype=np.float64)
        if not len(self.events):
            return mat
        di = {_to_date(d): i for i, d in enumerate(dates)}
        ci = {c: j for j, c in enumerate(codes)}
        ev = self.events[self.events["status"] != "invalid"]
        for r in ev.itertuples(index=False):
            i = di.get(r.ex_date)
            j = ci.get(r.symbol)
            if i is not None and j is not None:
                mat[i, j] *= float(r.adj_ratio)
        return mat

    def frame(self) -> pd.DataFrame:
        return self.events.copy()


class FrozenPanel:
    """[交易日 × 代码] 的只读面板 + 逐日 Bar 构造。"""

    def __init__(
        self,
        start: str,
        end: str,
        events: EventBook | None = None,
        board: str = BOARD,
        db_path: Path | str = DB_PATH,
    ):
        if _to_date(end) >= _to_date(HOLDOUT_START):
            raise ValueError(f"FrozenPanel 拒绝 2021+：{end}")
        self.start, self.end = str(start), str(end)
        self.board = board
        self.db_path = str(db_path)
        self.events = events or EventBook(db_path).load(start, end)
        self._load()
        self._derive()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            cal = con.execute(
                "SELECT date FROM calendar WHERE is_trading_day=1 AND date BETWEEN ? AND ? ORDER BY date",
                [self.start, self.end]).df()
            self.dates = [pd.Timestamp(x).date() for x in cal["date"]]
            if not self.dates:
                raise ValueError("窗口内没有交易日")
            self.n_days = len(self.dates)
            self.day_idx = {d: i for i, d in enumerate(self.dates)}
            d_np = np.array(self.dates, dtype="datetime64[D]")

            codes_df = con.execute(
                """SELECT DISTINCT symbol FROM daily
                   WHERE board = ? AND date BETWEEN ? AND ? ORDER BY 1""",
                [self.board, self.start, self.end]).df()
            self.codes = [str(x) for x in codes_df["symbol"]]
            self.code_idx = {c: j for j, c in enumerate(self.codes)}
            self.n_codes = len(self.codes)
            if not self.n_codes:
                raise ValueError("窗口内没有日线")

            self.mat: dict[str, np.ndarray] = {}
            for c in NUM_FIELDS:
                self.mat[c] = np.full((self.n_days, self.n_codes), np.nan, dtype=np.float32)
            for c in BOOL_FIELDS:
                self.mat[c] = np.zeros((self.n_days, self.n_codes), dtype=bool)

            n_rows = 0
            y0, y1 = self.dates[0].year, self.dates[-1].year
            for y in range(y0, y1 + 1):
                a = max(self.start, f"{y}-01-01")
                b = min(self.end, f"{y}-12-31")
                chunk = con.execute(
                    f"""SELECT symbol, date, {', '.join(NUM_FIELDS + BOOL_FIELDS)}
                        FROM daily
                        WHERE board = ? AND date BETWEEN ? AND ?""",
                    [self.board, a, b]).df()
                if chunk.empty:
                    continue
                if chunk.duplicated(["date", "symbol"]).any():
                    raise ValueError(f"{y} 年 daily 存在重复 (date,symbol)")
                di = np.searchsorted(d_np, chunk["date"].to_numpy(dtype="datetime64[D]"))
                ci = pd.Categorical(chunk["symbol"], categories=self.codes).codes.astype(np.int32)
                for c in NUM_FIELDS:
                    self.mat[c][di, ci] = chunk[c].to_numpy(dtype=np.float32)
                for c in BOOL_FIELDS:
                    self.mat[c][di, ci] = chunk[c].fillna(False).to_numpy(dtype=bool)
                n_rows += len(chunk)
                del chunk, di, ci

            meta = con.execute(
                "SELECT symbol, ipo_date, delist_date FROM symbols WHERE board = ?",
                [self.board]).df()
        finally:
            con.close()
        meta = meta.set_index("symbol").reindex(self.codes)
        self.ipo_date = [None if pd.isna(x) else pd.Timestamp(x).date() for x in meta["ipo_date"]]
        self.delist_date = [None if pd.isna(x) else pd.Timestamp(x).date() for x in meta["delist_date"]]
        self.n_rows_raw = int(n_rows)
        self.n_bad_bars = 0
        self.bad_bar_examples: list[dict] = []

    # ------------------------------------------------------------------
    def _derive(self) -> None:
        m = self.mat
        self.listed = np.isfinite(m["close"]) & (m["close"] > 0)
        susp = m["is_suspended"] | ~self.listed
        self.suspended = susp
        # 复牌首日：前一交易日停牌/未上市
        prev_susp = np.vstack([np.ones((1, self.n_codes), dtype=bool), susp[:-1]])
        prev_listed = np.vstack([np.zeros((1, self.n_codes), dtype=bool), self.listed[:-1]])
        self.resume = prev_susp | ~prev_listed

        # 除权因子：只在除权日跳变；ca_adj = raw / ca_factor 是连续价格水平。
        # 注意：冻结层 factor = Π(1+pct_chg) 已是总收益指数，close*factor 会重复计入
        # 每日收益（数据字典 5.1 的 hfq=raw*factor 只适用于“首条=1”的展示，
        # 其收益率不是总收益）。本层不复制该错误。
        ratio = self.events.ratio_frame(self.dates, self.codes)
        self.ca_factor = np.cumprod(ratio, axis=0)
        for f in ("open", "high", "low", "close"):
            self.mat[f"ca_{f}"] = m[f] / self.ca_factor.astype(np.float32)
        self.mat["tr_index"] = m["factor"]
        self.mat["hfq_close"] = m["factor"]
        self.mat["hfq_open"] = m["factor"]
        self.mat["hfq_high"] = m["factor"]
        self.mat["hfq_low"] = m["factor"]

        # 上市自然日数
        days = np.array([(d - date(1990, 1, 1)).days for d in self.dates], dtype=np.int32)
        ipo_days = np.array([
            np.nan if x is None else (x - date(1990, 1, 1)).days for x in self.ipo_date
        ], dtype=np.float64)
        self.days_since_ipo = days[:, None] - ipo_days[None, :]

        # 退市日对应的面板索引（用于退市整理期）
        self.delist_idx = np.full(self.n_codes, -1, dtype=np.int32)
        import bisect
        for j, dd in enumerate(self.delist_date):
            if dd is None:
                continue
            self.delist_idx[j] = int(bisect.bisect_left(self.dates, dd))
        self.delist_window = np.zeros_like(self.listed)
        for j in np.flatnonzero(self.delist_idx >= 0):
            k = int(self.delist_idx[j])
            if k < self.n_days:
                self.delist_window[max(0, k - DELIST_BLOCK_TD):k + 1, j] = True

        # 指标（ca_adj 空间）
        ca = pd.DataFrame(self.mat["ca_close"], index=pd.Index(self.dates, name="date"),
                          columns=self.codes)
        amt = pd.DataFrame(self.mat["amount"], index=ca.index, columns=self.codes)
        raw_close = pd.DataFrame(self.mat["close"], index=ca.index, columns=self.codes)
        hfq_close = pd.DataFrame(self.mat["hfq_close"], index=ca.index, columns=self.codes)
        self.adv20 = amt.rolling(20, min_periods=10).mean().to_numpy(dtype=np.float32)
        self.ma120 = ca.rolling(120, min_periods=120).mean().to_numpy(dtype=np.float32)
        self.rmax60 = ca.rolling(60, min_periods=60).max().to_numpy(dtype=np.float32)
        self.mom20 = (ca / ca.shift(20) - 1.0).to_numpy(dtype=np.float32)
        ca_high = pd.DataFrame(self.mat["ca_high"], index=ca.index, columns=self.codes)
        ca_low = pd.DataFrame(self.mat["ca_low"], index=ca.index, columns=self.codes)
        prev_ca = ca.shift(1)
        tr = pd.DataFrame(
            np.maximum.reduce([
                (ca_high - ca_low).to_numpy(dtype=np.float64),
                (ca_high - prev_ca).abs().to_numpy(dtype=np.float64),
                (ca_low - prev_ca).abs().to_numpy(dtype=np.float64),
            ]),
            index=ca.index, columns=self.codes,
        )
        self.atr14 = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy(dtype=np.float32)
        self.atr_raw = (self.atr14 * self.ca_factor).astype(np.float32)
        self.r_close = raw_close.to_numpy(dtype=np.float32)
        self.tr_index = m["factor"].astype(np.float32)
        self.bs_hfq = m["bs_hfq_close"].astype(np.float32)
        self.hfq = self.tr_index
        self.hfq_close = self.tr_index
        self.hfq_open = self.tr_index
        self.hfq_high = self.tr_index
        self.hfq_low = self.tr_index
        self.hfq_ret = pd.DataFrame(self.tr_index, index=ca.index,
                                    columns=self.codes).pct_change().to_numpy(dtype=np.float32)

        # 宇宙
        self.elig = (
            self.listed & ~self.suspended & ~self.mat["is_st"]
            & (self.mat["td_since_list"] >= MIN_LIST_TD)
            & (self.mat["close"] >= MIN_PRICE)
            & (self.adv20 >= MIN_ADV20)
            & (self.days_since_ipo >= MIN_LIST_CAL_DAYS)
            & ~self.resume & ~self.delist_window
            & np.isfinite(self.mat["ca_close"])
        )
        self.signal = (
            self.elig
            & (self.mat["ca_close"] >= self.rmax60 - 1e-6)
            & (self.mat["ca_close"] > self.ma120)
            & np.isfinite(self.atr14)
        )

    # ------------------------------------------------------------------
    def index_of(self, d) -> int:
        dd = _to_date(d)
        if dd not in self.day_idx:
            raise KeyError(f"{dd} 不在面板交易日")
        return self.day_idx[dd]

    def window_index(self, start, end) -> tuple[int, int]:
        """把自然日区间吸附到面板交易日：start 向后、end 向前。"""
        import bisect
        s, e = _to_date(start), _to_date(end)
        i0 = bisect.bisect_left(self.dates, s)
        i1 = bisect.bisect_right(self.dates, e) - 1
        if i0 > i1 or i0 >= self.n_days or i1 < 0:
            raise ValueError(f"区间 {start}~{end} 与面板无交集")
        return i0, i1

    def bars_on(self, i: int, codes: Iterable[str] | None = None) -> dict:
        """构造第 i 个交易日的 Bar 字典。

        传入 codes 时只构造这些代码（持仓/订单/当日除权），避免每天为全市场
        2,000 只证券创建对象；未持有的证券不需要 Bar，指标走矩阵。
        """
        from 账户撮合 import Bar

        d = self.dates[i]
        if codes is None:
            cols = np.flatnonzero(self.listed[i])
        else:
            js = [self.code_idx.get(c, -1) for c in codes]
            cols = np.array([j for j in js if j >= 0 and self.listed[i, j]], dtype=int)
        o = self.mat["open"][i]
        h = self.mat["high"][i]
        low = self.mat["low"][i]
        c = self.mat["close"][i]
        pc = self.mat["preclose"][i]
        vol = self.mat["volume"][i]
        amt = self.mat["amount"][i]
        st = self.mat["is_st"][i]
        susp = self.suspended[i]
        out = {}
        for j in cols:
            code = self.codes[j]
            try:
                out[code] = Bar(
                    code=code, date=d, open=float(o[j]), high=float(h[j]),
                    low=float(low[j]), close=float(c[j]),
                    volume=0.0 if not np.isfinite(vol[j]) else float(vol[j]),
                    amount=None if not np.isfinite(amt[j]) else float(amt[j]),
                    prev_close=None if not np.isfinite(pc[j]) else float(pc[j]),
                    prev_close_is_ref=True, is_st=bool(st[j]),
                    trade_status=0 if susp[j] else 1,
                )
            except Exception as exc:
                self.n_bad_bars += 1
                if len(self.bad_bar_examples) < 20:
                    self.bad_bar_examples.append({
                        "date": d.isoformat(), "code": code, "error": str(exc),
                        "ohlc": [float(o[j]), float(h[j]), float(low[j]), float(c[j])],
                    })
        return out

    # ------------------------------------------------------------------
    def universe_row(self, i: int) -> dict:
        return {
            "date": self.dates[i].isoformat(),
            "n_listed": int(self.listed[i].sum()),
            "n_tradable": int((self.listed[i] & ~self.suspended[i]).sum()),
            "n_st": int((self.listed[i] & self.mat["is_st"][i]).sum()),
            "n_resume": int((self.listed[i] & self.resume[i]).sum()),
            "n_eligible": int(self.elig[i].sum()),
            "n_signal": int(self.signal[i].sum()),
            "adv20_median_yi": round(float(np.nanmedian(self.adv20[i][self.listed[i]])) / 1e8, 4),
        }

    def coverage_table(self) -> pd.DataFrame:
        rows = []
        for y in sorted({d.year for d in self.dates}):
            idx = [i for i, d in enumerate(self.dates) if d.year == y]
            rows.append({
                "year": y,
                "trading_days": len(idx),
                "n_listed": int(self.listed[idx].any(axis=0).sum()),
                "eligible_mean": round(float(self.elig[idx].sum(axis=1).mean()), 1),
                "signal_mean": round(float(self.signal[idx].sum(axis=1).mean()), 2),
                "signal_days": int((self.signal[idx].sum(axis=1) > 0).sum()),
                "st_share": round(float((self.mat["is_st"][idx] & self.listed[idx]).sum()
                                        / max(1, self.listed[idx].sum())), 4),
                "susp_share": round(float((self.suspended[idx] & self.listed[idx]).sum()
                                          / max(1, self.listed[idx].sum())), 4),
                "resume_rows": int((self.resume[idx] & self.listed[idx]).sum()),
            })
        return pd.DataFrame(rows)

    def benchmark_ew(self, i0: int, i1: int, cost_roundtrip: float = 0.0040) -> pd.Series:
        """B1-EW：主宇宙等权、月末再平衡，按换手扣成本。返回日收益（含成本）。"""
        dates = self.dates[i0:i1 + 1]
        rets = np.zeros(len(dates), dtype=float)
        prev_w: dict[str, float] = {}
        rebal_days = {}
        for k, d in enumerate(dates):
            if k == len(dates) - 1 or (dates[k + 1].month != d.month):
                rebal_days[k] = True
        hr = self.hfq_ret
        active: list[int] = []
        for k in range(1, len(dates)):
            gi = i0 + k
            if k - 1 in rebal_days:
                cols = np.flatnonzero(self.elig[gi - 1])
                active = cols.tolist()
                w_new = {self.codes[j]: 1.0 / len(active) for j in active} if active else {}
                names = set(prev_w) | set(w_new)
                turnover = 0.5 * sum(abs(w_new.get(n, 0.0) - prev_w.get(n, 0.0)) for n in names)
                rets[k] -= turnover * cost_roundtrip
                prev_w = w_new
            if active:
                r = hr[gi, active]
                r = r[np.isfinite(r)]
                if len(r):
                    rets[k] += float(np.mean(r))
        return pd.Series(rets, index=pd.Index(dates, name="date"), name="B1_EW")


def window_bounds(name: str) -> tuple[str, str]:
    if name not in WINDOWS:
        raise KeyError(f"未知窗口 {name}")
    return WINDOWS[name]


if __name__ == "__main__":
    import time
    t0 = time.time()
    eb = EventBook().load("2007-01-01", "2020-12-31")
    print("EventBook:", eb.stats)
    panel = FrozenPanel("2007-01-01", "2020-12-31", events=eb)
    print(f"Panel {panel.n_days} 日 × {panel.n_codes} 只，原始行 {panel.n_rows_raw:,}，"
          f"耗时 {time.time()-t0:.1f}s")
    print(panel.coverage_table().to_string(index=False))
    d = date(2010, 1, 4)
    i = panel.index_of(d)
    print("bars", len(panel.bars_on(i)), "bad", panel.n_bad_bars)
    print(panel.universe_row(i))
