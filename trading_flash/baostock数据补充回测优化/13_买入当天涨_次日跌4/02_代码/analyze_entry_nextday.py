#!/usr/bin/env python3
"""买入当天涨、第二天跌 4 个点：有没有，最终盈利如何。

样本：12_买入后证伪_哪天没达标/03_结果/逐笔摘要.csv（1,256 笔）
  - 实际成交 219 笔（组合真实成交）
  - 其余信号前二 1,037 笔（每个信号日前二，未受仓位约束）
数据：baostock 后复权日线（adjustflag=1），与 12 分册同源。

口径：
  D+0 = 买入日（T+3 开盘买入）；D+1 = 买入后的下一个交易日。
  买入当天涨：
    主口径 d0_up_open = D+0 收盘 / 买入价（开盘） - 1 > 0
    对照   d0_up_prev = D+0 收盘 / D+0 前收 - 1 > 0（K 线收红）
  第二天跌 4 个点：
    主口径 d1_ret = D+1 收盘 / D+0 收盘 - 1 <= -4%
    对照   d1_low_ret = D+1 最低 / D+0 收盘 - 1 <= -4%（盘中触及）
  最终收益：
    实际成交用“实际净收益”；其余信号用“模拟最终收益”（+20% 止盈 / 15 日到期）。
    评估天数不足（是否部分=True）的 5 笔不计入最终收益统计。

输出：
  03_结果/逐笔_买入日次日.csv   每笔的 D+0/D+1 指标与最终收益（同时作为缓存）
  03_结果/条件汇总.csv         各口径、各样本、各期间的组合统计
  03_结果/个案_实际成交.csv     命中的实际成交逐笔清单
  03_结果/个案_全部信号.csv     命中的全部信号逐笔清单

复现：python3 -B 02_代码/analyze_entry_nextday.py
      如已存在逐笔缓存则直接分析，不重复拉取 baostock。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT.parent / ".pylibs"
sys.path.insert(0, str(LIB))

HERE = Path(__file__).resolve().parents[1]
SUMMARY = HERE.parent / "12_买入后证伪_哪天没达标" / "03_结果" / "逐笔摘要.csv"
OUT = HERE / "03_结果"
CACHE = OUT / "逐笔_买入日次日.csv"
OUT.mkdir(parents=True, exist_ok=True)

START, END = "2004-12-01", "2026-11-10"


def to_bs_code(code: str) -> str:
    market, ticker = code.split(".")
    return f"{market}.{ticker}"


def fetch_calendar(bs) -> list[str]:
    rs = bs.query_trade_dates(start_date=START, end_date=END)
    dates: list[str] = []
    while rs.next():
        row = rs.get_row_data()
        if row[1] == "1":
            dates.append(row[0])
    return dates


def fetch_bars(bs, code: str, start: str, end: str) -> pd.DataFrame:
    fields = "date,open,high,low,close,preclose,volume,tradestatus"
    rs = bs.query_history_k_data_plus(
        to_bs_code(code), fields,
        start_date=start, end_date=end,
        frequency="d", adjustflag="1",
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=fields.split(","))
    if df.empty:
        return df
    for c in ["open", "high", "low", "close", "preclose", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce")
    return df


def fetch_trades(trades: pd.DataFrame) -> pd.DataFrame:
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    calendar = fetch_calendar(bs)
    cal_idx = {d: i for i, d in enumerate(calendar)}
    print("交易日历：", calendar[0], "→", calendar[-1], len(calendar))

    records = []
    missing = []
    codes = sorted(trades["代码"].unique())
    for n, code in enumerate(codes, 1):
        sub = trades[trades["代码"] == code]
        idxs = [cal_idx.get(d) for d in sub["entry"]]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            missing.append((code, "entry_not_in_calendar"))
            continue
        start = calendar[max(0, min(idxs) - 5)]
        end = calendar[min(len(calendar) - 1, max(idxs) + 5)]
        bars = fetch_bars(bs, code, start, end)
        if bars.empty:
            missing.append((code, "no_bars"))
            continue
        bars["date"] = pd.to_datetime(bars["date"])
        bars = bars.set_index("date").sort_index()
        # 只在查询区间内按自然日重采样；停牌日价格前填，量 0
        rng = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
        bars = bars.reindex(rng)
        for c in ["open", "high", "low", "close", "preclose"]:
            bars[c] = bars[c].ffill()
        bars["volume"] = bars["volume"].fillna(0)
        bars["tradestatus"] = bars["tradestatus"].fillna(0)
        for r in sub.itertuples(index=False):
            i = cal_idx.get(r.entry)
            if i is None or i + 1 >= len(calendar):
                missing.append((code, f"no_d1:{r.entry}"))
                continue
            d1 = calendar[i + 1]
            try:
                b0 = bars.loc[pd.Timestamp(r.entry)]
                b1 = bars.loc[pd.Timestamp(d1)]
            except KeyError:
                missing.append((code, f"no_row:{r.entry}"))
                continue
            if not np.isfinite(b0["open"]) or not np.isfinite(b0["close"]):
                missing.append((code, f"no_price:{r.entry}"))
                continue
            prev = b0["preclose"] if np.isfinite(b0["preclose"]) else np.nan
            d0_up_open = b0["close"] / b0["open"] - 1
            d0_up_prev = b0["close"] / prev - 1 if np.isfinite(prev) and prev > 0 else np.nan
            records.append({
                "样本": r.样本,
                "信号日": r.信号日,
                "代码": r.代码,
                "买入日": r.买入日,
                "期间": r.期间,
                "排名": r.排名,
                "D1日期": pd.Timestamp(d1),
                "D0开盘": b0["open"],
                "D0收盘": b0["close"],
                "D0前收": prev,
                "D1开盘": b1["open"],
                "D1收盘": b1["close"],
                "D1最低": b1["low"],
                "d0_ret_open": d0_up_open,
                "d0_ret_prev": d0_up_prev,
                "d1_ret": b1["close"] / b0["close"] - 1,
                "d1_open_ret": b1["open"] / b0["close"] - 1,
                "d1_low_ret": b1["low"] / b0["close"] - 1,
                "d1_cum_ret": b1["close"] / b0["open"] - 1,
                "d1_volume": b1["volume"],
                "d1_tradestatus": b1["tradestatus"],
                "最终收益": r.outcome,
                "是否部分": bool(r.是否部分),
                "是否止盈": bool(r.是否止盈),
                "评估天数": r.评估天数,
            })
        if n % 50 == 0:
            print(f"  {n}/{len(codes)} codes, {len(records)} rows", flush=True)
        time.sleep(0.02)
    bs.logout()
    print("逐笔输出：", len(records), "笔；缺失：", len(missing))
    if missing:
        print("缺失样例：", missing[:10])
    return pd.DataFrame(records)


def add_flags(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["d0_up_open"] = out["d0_ret_open"] > 0
    out["d0_up_prev"] = out["d0_ret_prev"] > 0
    out["d1_drop4"] = out["d1_ret"] <= -0.04
    out["d1_low_drop4"] = out["d1_low_ret"] <= -0.04
    out["combo_open"] = out["d0_up_open"] & out["d1_drop4"]
    out["combo_prev"] = out["d0_up_prev"] & out["d1_drop4"]
    out["combo_low"] = out["d0_up_open"] & out["d1_low_drop4"]
    out["complete"] = ~out["是否部分"].astype(bool)
    return out


def summarize(sub: pd.DataFrame, mask: pd.Series, label: dict) -> dict:
    ok = sub["complete"] & sub["最终收益"].notna()
    e = sub[mask & ok]
    allx = sub[ok]
    r = dict(label)
    r.update({
        "N": len(e),
        "占样本": len(e) / len(allx) if len(allx) else np.nan,
        "胜率": (e["最终收益"] > 0).mean() if len(e) else np.nan,
        "亏损率": (e["最终收益"] < 0).mean() if len(e) else np.nan,
        "止盈率": e["是否止盈"].mean() if len(e) else np.nan,
        "均值": e["最终收益"].mean() if len(e) else np.nan,
        "中位": e["最终收益"].median() if len(e) else np.nan,
        "≤-10%": (e["最终收益"] <= -0.10).mean() if len(e) else np.nan,
        "≤-15%": (e["最终收益"] <= -0.15).mean() if len(e) else np.nan,
        "最小": e["最终收益"].min() if len(e) else np.nan,
        "最大": e["最终收益"].max() if len(e) else np.nan,
        "D0均值": e["d0_ret_open"].mean() if len(e) else np.nan,
        "D1均值": e["d1_ret"].mean() if len(e) else np.nan,
        "基准N": len(allx),
        "基准均值": allx["最终收益"].mean() if len(allx) else np.nan,
        "基准胜率": (allx["最终收益"] > 0).mean() if len(allx) else np.nan,
        "基准亏损率": (allx["最终收益"] < 0).mean() if len(allx) else np.nan,
    })
    return r


def analyze(out: pd.DataFrame, trades: pd.DataFrame) -> None:
    out = add_flags(out)
    # 对账：d1_cum_ret 应等于 12 分册的“第1天收益”
    chk = out.merge(trades[["信号日", "代码", "第1天收益"]], on=["信号日", "代码"], how="left")
    diff = (chk["d1_cum_ret"] - chk["第1天收益"]).abs()
    print("D+1 累计收益对账：最大差", diff.max(), "中位差", diff.median())

    rows = []
    for scope, sub in [("全部信号前二", out), ("实际成交", out[out["样本"] == "实际成交"])]:
        for period in ["2005-2016", "2017-2026", "全期"]:
            s = sub if period == "全期" else sub[sub["期间"] == period]
            if not len(s):
                continue
            for key, name in [
                ("combo_open", "D0收盘>买入价 且 D1≤-4%"),
                ("combo_prev", "D0收红 且 D1≤-4%"),
                ("combo_low", "D0收盘>买入价 且 D1最低≤-4%"),
            ]:
                rows.append(summarize(s, s[key], {"样本": scope, "期间": period, "条件": name}))
            base = s[s["complete"] & s["最终收益"].notna()]
            rows.append({
                "样本": scope, "期间": period, "条件": "基准（全部）",
                "N": len(base), "占样本": 1.0,
                "胜率": (base["最终收益"] > 0).mean(), "亏损率": (base["最终收益"] < 0).mean(),
                "止盈率": base["是否止盈"].mean(), "均值": base["最终收益"].mean(),
                "中位": base["最终收益"].median(), "≤-10%": (base["最终收益"] <= -0.10).mean(),
                "≤-15%": (base["最终收益"] <= -0.15).mean(),
                "最小": base["最终收益"].min(), "最大": base["最终收益"].max(),
                "D0均值": base["d0_ret_open"].mean(), "D1均值": base["d1_ret"].mean(),
                "基准N": len(base), "基准均值": base["最终收益"].mean(),
                "基准胜率": (base["最终收益"] > 0).mean(), "基准亏损率": (base["最终收益"] < 0).mean(),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "条件汇总.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    cols = ["信号日", "代码", "买入日", "期间", "排名", "D0开盘", "D0收盘", "D1日期", "D1收盘",
            "d0_ret_open", "d0_ret_prev", "d1_ret", "d1_open_ret", "d1_low_ret", "d1_cum_ret",
            "最终收益", "是否止盈", "评估天数", "是否部分"]
    out[(out["样本"] == "实际成交") & out["combo_open"]].sort_values("买入日")[cols].to_csv(
        OUT / "个案_实际成交.csv", index=False, encoding="utf-8-sig")
    out[out["combo_open"]].sort_values(["买入日", "代码"])[cols + ["样本"]].to_csv(
        OUT / "个案_全部信号.csv", index=False, encoding="utf-8-sig")
    print("完成。")


def main() -> None:
    trades = pd.read_csv(SUMMARY, parse_dates=["信号日", "买入日"])
    trades["entry"] = trades["买入日"].dt.strftime("%Y-%m-%d")
    trades["outcome"] = np.where(trades["样本"].eq("实际成交"), trades["实际净收益"], trades["模拟最终收益"])
    print("样本：", len(trades), "笔；代码：", trades["代码"].nunique())
    if CACHE.exists():
        print("使用缓存：", CACHE)
        out = pd.read_csv(CACHE, parse_dates=["信号日", "买入日", "D1日期"])
    else:
        out = fetch_trades(trades)
        out.to_csv(CACHE, index=False, encoding="utf-8-sig")
    analyze(out, trades)


if __name__ == "__main__":
    main()
