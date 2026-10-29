#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结数据层加载接口（沪深A股，2004-01-01 至 2026-10-26）。

设计原则
--------
* 只读：只读 01_数据/frozen 下的 Parquet / DuckDB，绝不回写源库。
* 无未来函数：所有接口都受 FREEZE 约束；`check_no_future` 可校验任意 DataFrame。
* 价格为"原始价"与"复权价"两套口径：
    - raw  : 交易所真实价格，用于价格档位、涨跌停、整股与费用计算
    - hfq  : A 空间后复权价，`raw_px_t × factor_t / close_t × close_first`
             （首条 hfq_close = 原始收盘）。日收益严格等于官方 `pct_chg`；
             开盘跨日收益等于 `open_t/preclose_t - 1`；OHLC 同乘一个逐日乘子，
             因此高低开收比例与原始价一致。**不要**再用 `raw × factor`。
    - qfq  : `raw_px_t × factor_t / close_t × close_last / factor_last`，末日
             qfq_close = 原始收盘；仅用于展示/横向比较，价格水平随数据更新而变。
* `factor` 是总收益指数 Π(1+pct_chg/100)；需要指数点位/累计收益时直接用 factor，
  不要把 hfq_close 当总收益指数。
* 金额单位：元；成交量：股；换手率：百分数（%）；pct_chg：百分数（%）。

典型用法
--------
    from 数据层 import DataLayer
    dl = DataLayer()
    px = dl.daily("2024-01-01", "2024-12-31", columns=["symbol","date","close","hfq_close","is_suspended"])
    u  = dl.universe("2024-06-28", min_avg_amount=5e7, min_td_since_list=120)
    cal = dl.calendar("2024-01-01", "2024-12-31", trading_only=True)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pandas as pd

FREEZE = "2026-10-26"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "01_数据"
FROZEN = DATA_DIR / "frozen"
DB_PATH = FROZEN / "frozen.duckdb"

DEFAULT_COLS = [
    "symbol", "date", "name", "board", "open", "high", "low", "close", "preclose",
    "volume", "amount", "turn", "is_suspended", "is_st", "pct_chg",
    "factor", "hfq_close", "limit_up_px", "limit_down_px",
    "is_limit_up_close", "is_limit_down_close", "is_limit_up_open", "is_limit_down_open",
    "td_since_list",
    "float_mktcap", "pe_ttm", "pb_mrq",
]

INDEX_NAMES = {
    "000001.SH": "上证综指", "000300.SH": "沪深300", "000905.SH": "中证500",
    "000852.SH": "中证1000", "000016.SH": "上证50", "000015.SH": "上证红利",
    "399001.SZ": "深证成指", "399006.SZ": "创业板指", "399106.SZ": "深证综指",
}


def _q(v) -> str:
    """SQL 字面量（仅用于内部受控值）。"""
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return str(v)


class DataLayer:
    def __init__(self, db_path: Path | str = DB_PATH, freeze: str = FREEZE):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"未找到 {self.db_path}；请先运行 02_代码/构建冻结数据.py")
        self.freeze = pd.Timestamp(freeze).date()
        self.con = duckdb.connect(str(self.db_path), read_only=True)
        self.con.execute("SET memory_limit='6GB'")

    # ------------------------------------------------------------------ #
    # 基础
    # ------------------------------------------------------------------ #
    def query(self, sql: str, params: Sequence | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params).df() if params else self.con.execute(sql).df()

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ------------------------------------------------------------------ #
    # 日线
    # ------------------------------------------------------------------ #
    def daily(
        self,
        start: str | None = None,
        end: str | None = None,
        symbols: Iterable[str] | None = None,
        columns: Sequence[str] | None = None,
        adjust: str = "raw",
        boards: Iterable[str] | None = None,
        include_suspended: bool = True,
        order_by: str | None = "symbol, date",
    ) -> pd.DataFrame:
        """读取日线。adjust ∈ {'raw','hfq','qfq'}。

        hfq 为 A 空间后复权价（首日原始价锚定）：hfq_close 的日收益 = 官方 pct_chg；
        hfq_open/hfq_close 的跨日收益 = open/preclose - 1；OHLC 比例与原始价一致。
        qfq 为冻结日锚定的前复权价，仅展示用。两者都不是总收益指数（那是 factor）。
        """
        start = str(start or "2004-01-01")
        end = str(min(pd.Timestamp(end or self.freeze), pd.Timestamp(self.freeze)).date())

        if adjust == "raw":
            src, cols = "daily", list(columns or DEFAULT_COLS)
        elif adjust in ("hfq", "qfq"):
            pre = adjust
            src = "daily_qfq" if adjust == "qfq" else "daily"
            base = list(columns or ["symbol", "date", "open", "high", "low", "close",
                                    "preclose", "volume", "amount", "turn", "is_suspended",
                                    "is_st", "pct_chg", "factor", "td_since_list",
                                    "limit_up_px", "limit_down_px", "float_mktcap"])
            cols = []
            ohlc = {"open", "high", "low", "close"}
            for c in base:
                if c in ohlc:
                    cols.append(f"{pre}_{c} AS {c}")
                    if c == "close":
                        cols.append(f"{'close'} AS close_raw")
                elif c == "close_raw":
                    continue
                else:
                    cols.append(c)
        else:
            raise ValueError("adjust 必须为 raw / hfq / qfq")

        if symbols is not None:
            symbols = list(symbols)
            if not symbols:
                return pd.DataFrame(columns=[c.split(" AS ")[-1] for c in cols])
            sym_f = f"AND symbol IN ({','.join(_q(s) for s in symbols)})"
        else:
            sym_f = ""
        brd_f = f"AND board IN ({','.join(_q(b) for b in boards)})" if boards else ""
        sus_f = "" if include_suspended else "AND NOT is_suspended"

        sql = f"""
        SELECT {', '.join(cols)} FROM {src}
        WHERE date BETWEEN DATE {_q(start)} AND DATE {_q(end)} {sym_f} {brd_f} {sus_f}
        """
        if order_by:
            sql += f" ORDER BY {order_by}"
        return self.query(sql)

    def bars(self, *a, **kw) -> pd.DataFrame:
        return self.daily(*a, **kw)

    def prices_wide(self, field: str = "close", start: str | None = None,
                    end: str | None = None, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        """宽表：index=date, columns=symbol。默认原始价。"""
        df = self.daily(start, end, symbols, columns=["symbol", "date", field])
        return df.pivot(index="date", columns="symbol", values=field).sort_index()

    # ------------------------------------------------------------------ #
    # 指数 / 日历 / 证券
    # ------------------------------------------------------------------ #
    def index(self, symbols: Iterable[str] | None = None, start: str | None = None,
              end: str | None = None) -> pd.DataFrame:
        start = str(start or "2006-01-01")
        end = str(min(pd.Timestamp(end or self.freeze), pd.Timestamp(self.freeze)).date())
        f = f"AND symbol IN ({','.join(_q(s) for s in symbols)})" if symbols else ""
        df = self.query(f"""SELECT * FROM index_daily
                            WHERE date BETWEEN DATE {_q(start)} AND DATE {_q(end)} {f}
                            ORDER BY symbol, date""")
        df["name"] = df["symbol"].map(INDEX_NAMES)
        return df

    def calendar(self, start: str | None = None, end: str | None = None,
                 trading_only: bool = False) -> pd.DataFrame:
        start = str(start or "1990-12-19")
        end = str(min(pd.Timestamp(end or self.freeze), pd.Timestamp(self.freeze)).date())
        f = "AND is_trading_day = 1" if trading_only else ""
        return self.query(f"""SELECT * FROM calendar
                             WHERE date BETWEEN DATE {_q(start)} AND DATE {_q(end)} {f}
                             ORDER BY date""")

    def trading_days(self, start: str | None = None, end: str | None = None) -> list:
        cal = self.calendar(start, end, trading_only=True)
        return [d.date() for d in pd.to_datetime(cal["date"])]

    def next_trading_day(self, d: str):
        """d 之后的下一个交易日（严格大于 d）。"""
        r = self.query(f"""SELECT min(date) d FROM calendar
                           WHERE is_trading_day = 1 AND date > DATE {_q(str(d))}""").iloc[0, 0]
        return None if pd.isna(r) else pd.Timestamp(r).date()

    def prev_trading_day(self, d: str):
        """d 之前的上一个交易日（严格小于 d）。"""
        r = self.query(f"""SELECT max(date) d FROM calendar
                           WHERE is_trading_day = 1 AND date < DATE {_q(str(d))}""").iloc[0, 0]
        return None if pd.isna(r) else pd.Timestamp(r).date()

    def shift_trading_day(self, d: str, n: int):
        """按交易日平移：n>0 向后，n<0 向前；越界返回 None。"""
        if n == 0:
            return pd.Timestamp(d).date()
        df = self.query(f"""
          WITH t AS (SELECT date, row_number() OVER (ORDER BY date) - 1 AS i
                     FROM calendar WHERE is_trading_day = 1),
          c AS (SELECT i FROM t WHERE date = DATE {_q(str(d))})
          SELECT date FROM t, c WHERE t.i = c.i + {int(n)}""")
        return None if df.empty else pd.Timestamp(df.iloc[0, 0]).date()

    def symbols(self, boards: Iterable[str] | None = None) -> pd.DataFrame:
        f = f"WHERE board IN ({','.join(_q(b) for b in boards)})" if boards else ""
        return self.query(f"SELECT * FROM symbols {f} ORDER BY symbol")

    def market(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        start = str(start or "2004-01-01")
        end = str(min(pd.Timestamp(end or self.freeze), pd.Timestamp(self.freeze)).date())
        return self.query(f"""SELECT * FROM market_daily
                             WHERE date BETWEEN DATE {_q(start)} AND DATE {_q(end)}
                             ORDER BY date""")

    def corp_actions(self, symbols: Iterable[str] | None = None, start: str | None = None,
                     end: str | None = None) -> pd.DataFrame:
        f = f"AND symbol IN ({','.join(_q(s) for s in symbols)})" if symbols else ""
        start = str(start or "1990-01-01")
        end = str(min(pd.Timestamp(end or self.freeze), pd.Timestamp(self.freeze)).date())
        return self.query(f"""SELECT * FROM corp_actions
                             WHERE ex_date BETWEEN DATE {_q(start)} AND DATE {_q(end)} {f}
                             ORDER BY symbol, ex_date""")

    # ------------------------------------------------------------------ #
    # 可交易域
    # ------------------------------------------------------------------ #
    def universe(
        self,
        date: str,
        amount_window: int = 20,
        min_avg_amount: float | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        min_float_mktcap: float | None = None,
        exclude_st: bool = True,
        min_td_since_list: int = 60,
        boards: Iterable[str] | None = None,
        exclude_suspended: bool = True,
        exclude_limit_up: bool = False,
        exclude_limit_down: bool = False,
    ) -> pd.DataFrame:
        """某交易日的可交易股票快照。

        流动性用过去 amount_window 个交易日（含当日）的成交额均值，
        只用 <= date 的数据，无未来信息。
        """
        d = str(min(pd.Timestamp(date), pd.Timestamp(self.freeze)).date())
        w = int(amount_window)
        conds = [f"date = DATE {_q(d)}"]
        if exclude_suspended:
            conds.append("NOT is_suspended")
        if exclude_st:
            conds.append("coalesce(is_st,0) = 0")
        if min_td_since_list:
            conds.append(f"td_since_list >= {int(min_td_since_list)}")
        if min_price is not None:
            conds.append(f"close >= {float(min_price)}")
        if max_price is not None:
            conds.append(f"close <= {float(max_price)}")
        if min_float_mktcap is not None:
            conds.append(f"float_mktcap >= {float(min_float_mktcap)}")
        if boards:
            conds.append(f"board IN ({','.join(_q(b) for b in boards)})")
        if exclude_limit_up:
            conds.append("coalesce(is_limit_up_close,false) = false")
        if exclude_limit_down:
            conds.append("coalesce(is_limit_down_close,false) = false")
        if min_avg_amount is not None:
            conds.append(f"avg_amount_{w} >= {float(min_avg_amount)}")
        return self.query(f"""
        WITH w AS (
          SELECT symbol, date, close, name, board, amount, float_mktcap, is_st,
                 is_suspended, td_since_list, limit_up_px, limit_down_px,
                 is_limit_up_close, is_limit_down_close,
                 is_limit_up_open, is_limit_down_open,
                 avg(amount) OVER (PARTITION BY symbol ORDER BY date
                                   ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS avg_amount_{w},
                 count(*)   OVER (PARTITION BY symbol ORDER BY date
                                   ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS n_{w}
          FROM daily
          WHERE date BETWEEN DATE {_q(d)} - INTERVAL '{w*2} days' AND DATE {_q(d)}
        )
        SELECT * FROM w WHERE {' AND '.join(conds)} AND n_{w} >= {max(1, w//2)}
        ORDER BY symbol
        """)

    # ------------------------------------------------------------------ #
    # 校验
    # ------------------------------------------------------------------ #
    def check_no_future(self, df: pd.DataFrame, date_col: str = "date") -> None:
        if df.empty:
            return
        mx = pd.to_datetime(df[date_col]).max().date()
        if mx > self.freeze:
            raise ValueError(f"数据包含冻结日之后的行：{mx} > {self.freeze}")

    def describe(self) -> pd.DataFrame:
        rows = []
        for t in ("daily", "index_daily", "calendar", "symbols", "market_daily",
                  "corp_actions", "corp_actions_implied", "bj_daily"):
            try:
                n = self.query(f"SELECT count(*) n FROM {t}").iloc[0, 0]
                rows.append((t, int(n)))
            except Exception as e:  # 表可能未构建
                rows.append((t, f"ERR: {e}"))
        return pd.DataFrame(rows, columns=["table", "rows"])


if __name__ == "__main__":
    with DataLayer() as dl:
        print(dl.describe().to_string(index=False))
        print("\n最近 3 个交易日：", dl.trading_days("2026-10-20", "2026-10-26"))
        s = dl.daily("2026-10-20", "2026-10-26", columns=["symbol", "date", "close", "hfq_close"])
        print(s.head(3).to_string(index=False))
        u = dl.universe("2026-10-26", min_avg_amount=5e7)
        print(f"\n2026-10-26 可交易域（20日均额>5000万）：{len(u)} 只，前3：")
        print(u.head(3).to_string(index=False))
        chk = dl.query("""
        WITH r AS (
          SELECT symbol, date, pct_chg, hfq_close,
                 lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) prev
          FROM daily
        )
        SELECT count(*) FILTER (WHERE prev > 0) n,
               count(*) FILTER (WHERE prev > 0 AND abs(hfq_close/prev - 1 - pct_chg/100) > 5e-5) bad,
               max(abs(hfq_close/prev - 1 - pct_chg/100)) FILTER (WHERE prev > 0) mx
        FROM r""")
        print(f"\nhfq_close 相邻日收益 vs pct_chg：{int(chk.n[0]):,} 对，"
              f">0.5bp 的 {int(chk.bad[0])} 行，最大误差 {chk.mx[0]:.2e}")
