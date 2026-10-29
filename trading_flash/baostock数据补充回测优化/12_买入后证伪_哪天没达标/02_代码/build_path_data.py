#!/usr/bin/env python3
"""构建“买入后逐日路径”数据集。

样本：
  1) signal_top2：C 池（含退市、信号日排除风险警示）每个信号日前二，
     T+3 开盘可买才计入；不限总仓位。
  2) actual：现有 C 池组合实际成交 219 笔，直接使用逐笔记录里的买入日。

口径：
  - 收益一律后复权；买入价=买入日开盘；第 d 天=买入日起第 d 个交易日（d=0 为买入日）。
  - 15 个交易日到期；+20% 止盈从买入次日（d>=1）起算，触价即按 +20% 计。
  - 路径统计不强制止盈，另给“实际规则”结果。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "02_代码"
sys.path.insert(0, str(CODE))

from strategy_lib import Config, build_signals, load_panel, load_stocks, round_half_up  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "03_结果"
OUT.mkdir(parents=True, exist_ok=True)
BARS = ROOT / "01_数据" / "bars.parquet"
TRADES = ROOT / "03_结果" / "指标优化" / "2005-2026" / "C_排除ST_逐笔.csv"

START, END = "2005-01-01", "2026-10-23"
HOLD = 15
TP = 0.20

SIGNAL_COLS = [
    "code", "date", "is_st", "listing_session", "amount20", "return_250d",
    "drawdown250", "rsi14", "close_location", "market_return20_all", "market_return",
]
BAR_COLS = [
    "code", "date", "open", "high", "low", "close", "preclose", "volume",
    "trade_status", "hfq_open", "hfq_high", "hfq_low", "hfq_close",
]


def load_signals() -> pd.DataFrame:
    panel = load_panel(columns=SIGNAL_COLS)
    stocks = load_stocks()
    cfg = Config(start=START, end=END, universe="all", gate="all",
                 extra_filter="is_st == 0", label="post_entry")
    full = build_signals(panel, stocks, cfg)
    counts = full.groupby("date").size().rename("候选数").reset_index()
    counts = counts.rename(columns={"date": "信号日"})
    counts.to_csv(OUT / "信号日候选数.csv", index=False, encoding="utf-8-sig")
    sig = full[full["rank"] <= 2].copy()
    sig["signal_date"] = pd.to_datetime(sig["date"])
    return sig[["signal_date", "code", "rank", "drawdown250", "rsi14",
                "return_250d", "close_location", "market_return20_all", "market_return"]].copy()


def load_bars(codes: set[str]) -> pd.DataFrame:
    df = pd.read_parquet(BARS, columns=BAR_COLS)
    df = df[df["code"].isin(codes)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code", sort=False)
    df["ma5"] = g["hfq_close"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    df["ma10"] = g["hfq_close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    df["vol_ma20_prev"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    df["ret1"] = g["hfq_close"].transform(lambda s: s.pct_change())
    rng = df["high"] - df["low"]
    df["close_loc"] = np.where(rng > 0, (df["close"] - df["low"]) / rng, 0.5)
    df["upper_shadow"] = np.where(df["close"] > 0,
                                  (df["high"] - np.maximum(df["open"], df["close"])) / df["close"], np.nan)
    df["body"] = np.where(df["close"] > 0,
                          (df["close"] - df["open"]).abs() / df["close"], np.nan)
    df["gap"] = np.where(df["preclose"] > 0, df["open"] / df["preclose"] - 1, np.nan)
    df["tradable"] = (df["trade_status"].fillna(0) == 1) & (df["volume"].fillna(0) > 0)
    df["limit_up_px"] = np.floor(df["preclose"] * 1.10 * 100 + 0.5) / 100
    df["limit_up"] = df["tradable"] & (df["open"] >= df["limit_up_px"] - 0.001)
    return df


def make_window(bars: pd.DataFrame, entry_date: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.DataFrame | None:
    idx = calendar.get_indexer([entry_date])[0]
    if idx < 0:
        return None
    dates = calendar[idx:min(idx + HOLD, len(calendar))]
    if len(dates) == 0:
        return None
    w = bars.set_index("date").reindex(dates)
    if not np.isfinite(w.loc[entry_date, "hfq_open"]):
        return None
    # 停牌/缺行：价格前填，成交量 0
    for c in ["open", "high", "low", "close", "preclose", "hfq_open", "hfq_high", "hfq_low", "hfq_close"]:
        w[c] = w[c].ffill()
    w["volume"] = w["volume"].fillna(0)
    w["trade_status"] = w["trade_status"].fillna(0)
    w["tradable"] = w["tradable"].fillna(False)
    w["date"] = w.index
    w["d"] = np.arange(len(w))
    return w


def quantify(w: pd.DataFrame, entry_date: pd.Timestamp, mkt: pd.Series) -> pd.DataFrame:
    entry_open = float(w.loc[entry_date, "hfq_open"])
    entry_high = float(w.loc[entry_date, "hfq_high"])
    entry_low = float(w.loc[entry_date, "hfq_low"])
    entry_close = float(w.loc[entry_date, "hfq_close"])
    w = w.copy()
    w["entry_open"] = entry_open
    w["entry_high"] = entry_high
    w["entry_low"] = entry_low
    w["entry_close"] = entry_close
    w["ret"] = w["hfq_close"] / entry_open - 1
    w["ret_prev"] = w["hfq_close"] / w["hfq_close"].shift(1) - 1
    w.loc[w["d"] == 0, "ret_prev"] = w.loc[w["d"] == 0, "hfq_close"] / entry_open - 1
    w["runmax_high"] = w["hfq_high"].cummax()
    prev_runmax = w["runmax_high"].shift(1)
    w["new_high"] = w["hfq_high"] > prev_runmax
    w.loc[w["d"] == 0, "new_high"] = True
    w["no_new_high_since_entry"] = w["runmax_high"] <= entry_high + 1e-12
    w["runmin_close"] = w["hfq_close"].cummin()
    w["runmin_low"] = w["hfq_low"].cummin()
    w["break_entry_low_close"] = w["runmin_close"] < entry_low - 1e-12
    w["break_entry_low_intraday"] = w["runmin_low"] < entry_low - 1e-12
    w["below_entry"] = w["hfq_close"] < entry_open - 1e-12
    w["below_ma5"] = w["hfq_close"] < w["ma5"] - 1e-12
    w["below_ma10"] = w["hfq_close"] < w["ma10"] - 1e-12
    w["ma5_down"] = w["ma5"] < w["ma5"].shift(1) - 1e-12
    w["lower_high_low"] = (w["hfq_high"] < w["hfq_high"].shift(1) - 1e-12) & (w["hfq_low"] < w["hfq_low"].shift(1) - 1e-12)
    w["two_lower"] = w["lower_high_low"] & w["lower_high_low"].shift(1, fill_value=False)
    w["vol_ratio"] = w["volume"] / w["vol_ma20_prev"]
    w["heavy_down"] = (w["ret_prev"] < -0.03) & (w["vol_ratio"] > 1.5)
    w["weak_close"] = w["close_loc"] < 0.35
    w["gap_fade"] = (w["gap"] > 0.01) & (w["close"] < w["open"])
    w["long_upper"] = (w["upper_shadow"] > w["body"]) & (w["upper_shadow"] > 0.01)
    # 相对市场：从买入日收盘到 d 收盘
    mkt_dates = w["date"]
    mkt_cum = (1 + mkt.reindex(mkt_dates).fillna(0)).cumprod() - 1
    w["mkt_cum"] = mkt_cum.to_numpy()
    w["rel_ret"] = (w["hfq_close"] / entry_close - 1) - w["mkt_cum"]
    # 5 日量价背离：上涨日平均量 < 下跌日平均量，且区间仍为正
    up_vol, dn_vol, up_n, dn_n = [], [], [], []
    for i in range(len(w)):
        sub = w.iloc[: i + 1]
        up = sub[sub["ret_prev"] > 0]
        dn = sub[sub["ret_prev"] < 0]
        up_vol.append(up["volume"].mean() if len(up) else np.nan)
        dn_vol.append(dn["volume"].mean() if len(dn) else np.nan)
    w["up_vol"] = up_vol
    w["dn_vol"] = dn_vol
    w["vol_price_div"] = (w["ret"] > 0) & (w["up_vol"] < w["dn_vol"])
    n_days = len(w)
    w["n_days"] = n_days
    w["partial"] = n_days < HOLD
    w["mtm_ret"] = w["ret"].iloc[-1]
    w["hold15_ret"] = w["ret"].iloc[-1] if n_days >= HOLD else np.nan
    # 止盈：从 d>=1 起，盘中触及 +20%
    tp_mask = (w["d"] >= 1) & (w["hfq_high"] >= entry_open * (1 + TP) - 1e-12)
    w["tp_hit_so_far"] = tp_mask.cummax()
    w["tp_day"] = np.where(tp_mask, w["d"], np.nan)
    w["actual_ret"] = np.where(w["tp_hit_so_far"], TP, w["hold15_ret"])
    w["final_loss"] = w["actual_ret"] < 0
    w["hold15_loss"] = w["hold15_ret"] < 0
    # 每笔交易的最终结果（与 d 无关）：曾止盈按 +20%，否则 15 日收盘；未满 15 日且未止盈为 NaN
    any_tp = bool(w["tp_hit_so_far"].iloc[-1])
    final_ret = TP if any_tp else (w["hold15_ret"].iloc[-1] if n_days >= HOLD else np.nan)
    w["tp_any"] = any_tp
    w["final_ret_sim"] = final_ret
    w["final_loss_sim"] = (final_ret < 0) if np.isfinite(final_ret) else np.nan
    # 之后：从 d 收盘到 15 日收盘；以及早退（d+1 开盘）
    w["forward_ret"] = (1 + w["hold15_ret"]) / (1 + w["ret"]) - 1
    fwd = []
    for i in range(len(w)):
        tail = w["hfq_close"].iloc[i + 1:]
        fwd.append(tail.min() / w["hfq_close"].iloc[i] - 1 if len(tail) else np.nan)
    w["future_min_close"] = fwd
    w["future_decline_5"] = w["future_min_close"] <= -0.05
    w["exit_next_open"] = w["hfq_open"].shift(-1) / entry_open - 1
    w["early_exit_adv"] = w["exit_next_open"] - w["actual_ret"]
    return w


def main() -> None:
    sig = load_signals()
    print("信号级前二：", len(sig), "笔，", sig.signal_date.nunique(), "个信号日")
    bars = load_bars(set(sig.code.unique()))
    calendar = pd.DatetimeIndex(sorted(bars.date.unique()))
    panel_mkt = load_panel(columns=["date", "market_return"])
    mkt = panel_mkt.drop_duplicates("date").set_index("date")["market_return"].sort_index()
    mkt.index = pd.to_datetime(mkt.index)
    cal_idx = {d: i for i, d in enumerate(calendar)}

    actual = pd.read_csv(TRADES, parse_dates=["signal_date", "entry_date"])
    actual_keys = set(zip(actual.signal_date, actual.code))
    actual_entry = dict(zip(zip(actual.signal_date, actual.code), actual.entry_date))

    rows = []
    skipped = {"no_entry_bar": 0, "limit_up": 0, "not_enough": 0}
    for r in sig.itertuples(index=False):
        key = (r.signal_date, r.code)
        if key in actual_entry:
            entry_date = pd.Timestamp(actual_entry[key])
        else:
            si = cal_idx.get(r.signal_date)
            if si is None or si + 3 >= len(calendar):
                skipped["not_enough"] += 1
                continue
            entry_date = calendar[si + 3]
        b = bars[bars.code == r.code].set_index("date")
        if entry_date not in b.index:
            skipped["no_entry_bar"] += 1
            continue
        er = b.loc[entry_date]
        if not bool(er["tradable"]):
            skipped["no_entry_bar"] += 1
            continue
        if bool(er["limit_up"]):
            skipped["limit_up"] += 1
            continue
        w = make_window(b.reset_index(), entry_date, calendar)
        if w is None:
            skipped["not_enough"] += 1
            continue
        w = quantify(w, entry_date, mkt)
        w["signal_date"] = r.signal_date
        w["code"] = r.code
        w["rank"] = r.rank
        w["drawdown250_signal"] = r.drawdown250
        w["is_actual"] = key in actual_keys
        w["sample"] = np.where(w["is_actual"], "actual", "signal_top2")
        rows.append(w)
    path = pd.concat(rows, ignore_index=True)
    # 实际成交：用逐笔记录里的净收益/退出原因覆盖模拟结果（含跳空止盈、跌停顺延、退市）
    path = path.merge(
        actual[["signal_date", "code", "ret", "exit_reason", "holding_sessions", "delisted"]]
        .rename(columns={"ret": "logged_ret", "exit_reason": "logged_exit_reason",
                         "holding_sessions": "logged_holding", "delisted": "logged_delisted"}),
        on=["signal_date", "code"], how="left",
    )
    path["final_loss_logged"] = path["logged_ret"] < 0
    path.to_parquet(OUT / "买入后逐日路径.parquet", index=False)
    print("路径行数：", len(path), "交易数：", path.groupby(["signal_date", "code"]).ngroups)
    print("跳过：", skipped)
    print(path.groupby("sample").agg(trades=("code", "size"), final_loss=("final_loss", "mean")).to_string())
    # 对账：actual 与逐笔记录
    tro = path[path["is_actual"]].groupby(["signal_date", "code"]).agg(
        entry=("entry_open", "first"), actual_ret=("actual_ret", "last"), hold15=("hold15_ret", "last"),
        logged_ret=("logged_ret", "last"),
    ).reset_index()
    tro["diff"] = tro["actual_ret"] - tro["logged_ret"]
    print("actual 对账：", len(tro), "最大差", tro["diff"].abs().max(), "中位差", tro["diff"].median())
    print(tro.sort_values("diff", key=lambda s: s.abs(), ascending=False).head(8).to_string())


if __name__ == "__main__":
    main()
