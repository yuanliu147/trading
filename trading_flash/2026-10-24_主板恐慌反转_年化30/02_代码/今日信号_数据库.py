"""直接从本地 DuckDB 计算主板恐慌洗盘反转的当日信号。

默认保留原回测的实际股票池，不代表已识别历史 ST。可选的 5% 样态过滤单独输出，不与基准混用。

用法：
  python3 02_代码/今日信号_数据库.py
  python3 02_代码/今日信号_数据库.py 2026-10-23
  python3 02_代码/今日信号_数据库.py 2026-10-23 --exclude-5pct-sample
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB = "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"
PREFIX = ("600", "601", "603", "605", "000", "001", "002", "003")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "03_结果"


def load(start="2024-10-01"):
    con = duckdb.connect(DB, read_only=True)
    prefixes = ",".join(f"'{p}'" for p in PREFIX)
    sql = f"""
        WITH base AS (
            SELECT h.thscode, h.date, h.high, h.low, h.close,
                   d.close AS actual_close, d.high AS actual_high, d.low AS actual_low,
                   d.amount, f.backward_factor,
                   row_number() OVER (PARTITION BY h.thscode ORDER BY h.date) AS listing_session
            FROM v_daily_hfq h
            JOIN v_daily d USING (thscode, date)
            JOIN calc_adjust_factor_daily f USING (thscode, date)
            WHERE substr(h.thscode, 1, 3) IN ({prefixes})
        )
        SELECT * FROM base WHERE date >= DATE '{start}' ORDER BY thscode, date
    """
    df = con.execute(sql).fetchdf()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def features(df):
    out = []
    for _, g in df.groupby("thscode", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ag = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        al = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        g["ret1"] = g["close"].pct_change()
        g["rsi14"] = 100 - 100 / (1 + ag / al)
        g["ma20"] = g["close"].rolling(20, min_periods=20).mean()
        g["amount20"] = g["amount"].rolling(20, min_periods=20).mean()
        g["return_250d"] = g["close"] / g["close"].shift(250) - 1
        g["drawdown250"] = g["close"] / g["high"].rolling(250, min_periods=250).max() - 1
        g["close_location"] = (g["actual_close"] - g["actual_low"]) / (g["actual_high"] - g["actual_low"])
        g["ref_close"] = g["actual_close"].shift(1) * g["backward_factor"].shift(1) / g["backward_factor"]
        g["limit_up_05"] = g["actual_close"] >= np.round(g["ref_close"] * 1.05, 2) - 1e-4
        g["limit_up_10"] = g["actual_close"] >= np.round(g["ref_close"] * 1.10, 2) - 1e-4
        out.append(g)
    df = pd.concat(out, ignore_index=True)

    mkt = df.groupby("date")["ret1"].mean().sort_index()
    index = (1 + mkt.fillna(0)).cumprod()
    market = pd.DataFrame({
        "market_return": mkt,
        "market_return20": index.pct_change(20, fill_method=None),
        "breadth20": df[df["listing_session"] >= 20].assign(above=lambda x: x["close"] > x["ma20"]).groupby("date")["above"].mean(),
    })
    return df, market


def run(day=None, exclude_5pct_sample=False):
    df, market = features(load())
    if day is None:
        day = df["date"].max()
    day = pd.Timestamp(day)
    m = market.loc[day]
    d = df[df["date"] == day].copy()
    d = d[(d["listing_session"] >= 250) & (d["amount20"] >= 10_000_000)]
    if exclude_5pct_sample:
        d = d[d["limit_up_05"] != 1]
    d["market_return20"] = m["market_return20"]
    d["breadth20"] = m["breadth20"]

    d["cond_dd"] = d["drawdown250"] < -0.45
    d["cond_r250"] = d["return_250d"] > 0
    d["cond_rsi"] = d["rsi14"] < 35
    d["cond_cl"] = d["close_location"] > 0.45
    gate = m["market_return20"] < -0.05
    d["cond_all"] = d[["cond_dd", "cond_r250", "cond_rsi", "cond_cl"]].all(axis=1) & gate
    d["ncond"] = d[["cond_dd", "cond_r250", "cond_rsi", "cond_cl"]].sum(axis=1).astype(int)

    sig = d[d["cond_all"]].sort_values("drawdown250")
    sig["rank"] = range(1, len(sig) + 1)
    cols = ["date", "rank", "thscode", "actual_close", "return_250d", "drawdown250", "rsi14",
            "close_location", "amount20", "limit_up_05", "market_return20", "breadth20"]
    near = d[d["ncond"] >= 3].sort_values(["ncond", "drawdown250"], ascending=[False, True])

    print(f"日期 {day.date()}  主板闸门 {'开' if gate else '关'}  "
          f"market_return20={m['market_return20']:.2%}  breadth20={m['breadth20']:.1%}")
    filter_note = "剔除5%涨停样态（非历史ST识别）" if exclude_5pct_sample else "原回测实际股票池（未识别历史ST）"
    print(f"样本 {len(d)} 只（上市≥250日、20日均额≥1000万；{filter_note}）")
    print("条件通过数：" + str({k: int(v) for k, v in
          {"回撤<-45%": d.cond_dd.sum(), "250日收益>0": d.cond_r250.sum(),
           "RSI<35": d.cond_rsi.sum(), "收盘位置>0.45": d.cond_cl.sum(),
           "四条件全满足": d[["cond_dd", "cond_r250", "cond_rsi", "cond_cl"]].all(axis=1).sum(),
           "含闸门后信号": len(sig)}.items()}))
    if sig.empty:
        print("\n今日无信号。")
    else:
        show = sig[cols].copy()
        for c in ["return_250d", "drawdown250", "close_location", "market_return20", "breadth20"]:
            show[c] = (show[c] * 100).round(2)
        show["amount20"] = (show["amount20"] / 1e8).round(2)
        show["rsi14"] = show["rsi14"].round(1)
        show.columns = ["日期", "排名", "代码", "收盘", "250日收益%", "250日回撤%", "RSI14",
                        "收盘位置", "20日均额(亿)", "收盘≥+5%", "市场20日%", "宽度20%"]
        print("\n信号（按250日回撤升序）：")
        print(show.head(10).to_string(index=False))

    OUT.mkdir(exist_ok=True)
    tag = day.strftime("%Y-%m-%d") + ("_exclude5pct" if exclude_5pct_sample else "")
    sig[cols].to_csv(OUT / f"当前信号_{tag}.csv", index=False)
    near_out = near[[c for c in cols if c != "rank"] + ["ncond"]].copy()
    near_out.to_csv(OUT / f"近信号_{tag}.csv", index=False)
    print(f"\n已写出 {OUT / f'当前信号_{tag}.csv'}（{len(sig)} 行）")
    print(f"已写出 {OUT / f'近信号_{tag}.csv'}（{len(near)} 行）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("day", nargs="?")
    parser.add_argument("--exclude-5pct-sample", action="store_true")
    args = parser.parse_args()
    run(args.day, args.exclude_5pct_sample)
