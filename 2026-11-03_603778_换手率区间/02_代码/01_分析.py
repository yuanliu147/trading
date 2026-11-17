# -*- coding: utf-8 -*-
"""603778（国晟科技，原乾景园林）换手率区间研究。

数据：
- baostock 本地 SQLite：2015-12-31 至 2026-10-29，不复权价 + 官方换手率
- 同花顺 DuckDB：补 2026-10-30、2026-11-02 两根日线；换手率按流通股本 6.428562 亿股估算
收益统一用 baostock 后复权价（adjustflag=1），避免除权跳空污染。
"""
import os
import sqlite3
import duckdb
import numpy as np
import pandas as pd

ROOT = "/Users/admin/WorkSpace/trading/2026-11-03_603778_换手率区间"
BAO = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"
HIT = "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"
FLOAT_SHARES = 642_856_200  # 2021-06 起稳定，2025-2026 逐日反推均为 6.4286 亿股

BINS = [0, 0.5, 1, 2, 3.5, 5, 8, 12, 20, 1e9]
LABELS = ["<0.5", "0.5-1", "1-2", "2-3.5", "3.5-5", "5-8", "8-12", "12-20", ">20"]


def load_data() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{BAO}?mode=ro", uri=True)
    raw = pd.read_sql(
        """select date,open,high,low,close,preclose,volume,amount,turn,pct_chg,
                  pe_ttm,pb_mrq,is_st
           from stock_kline_d
           where code='sh.603778' and adjustflag=3 order by date""",
        con, parse_dates=["date"])
    hfq = pd.read_sql(
        """select date,open,high,low,close from stock_kline_d
           where code='sh.603778' and adjustflag=1 order by date""",
        con, parse_dates=["date"])
    con.close()

    df = raw.merge(
        hfq.rename(columns={c: f"{c}_h" for c in ["open", "high", "low", "close"]}),
        on="date", how="left")
    df = df.dropna(subset=["turn"]).reset_index(drop=True)

    # 所有均线/位置/动量都用后复权价，避免 10 转 15 等除权造成的假跳空
    c, h, l = df["close_h"], df["high_h"], df["low_h"]
    for n in (5, 20, 60, 250):
        df[f"ma{n}"] = c.rolling(n).mean()
    df["ret5"] = c.pct_change(5) * 100
    df["ret20"] = c.pct_change(20) * 100
    df["ret60"] = c.pct_change(60) * 100
    df["hi250"] = h.rolling(250).max()
    df["lo250"] = l.rolling(250).min()
    df["pos250"] = (c - df.lo250) / (df.hi250 - df.lo250) * 100
    df["dd250"] = (c / df.hi250 - 1) * 100
    df["above_ma20"] = (c > df.ma20).astype(int)
    df["above_ma250"] = (c > df.ma250).astype(int)
    df["vol20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100
    df["limit_up"] = (df.pct_chg >= 9.7).astype(int)
    df["limit_dn"] = (df.pct_chg <= -9.7).astype(int)
    df["oneword"] = ((df.limit_up == 1) & (df.turn < 1)).astype(int)

    # T+1 开盘买入，持有 N 个交易日到收盘（后复权）
    o1 = df.open_h.shift(-1)
    for n in (1, 3, 5, 10, 20, 60):
        df[f"f{n}"] = (df.close_h.shift(-n) / o1 - 1) * 100
    fut_hi = pd.concat([df.high_h.shift(-i) for i in range(1, 21)], axis=1).max(axis=1)
    fut_lo = pd.concat([df.low_h.shift(-i) for i in range(1, 21)], axis=1).min(axis=1)
    df["mfe20"] = (fut_hi / o1 - 1) * 100
    df["mae20"] = (fut_lo / o1 - 1) * 100

    df["year"] = df.date.dt.year
    df["band"] = pd.cut(df.turn, bins=BINS, labels=LABELS, right=False)
    df["turn_pct250"] = df.turn.rolling(250, min_periods=120).apply(
        lambda s: (s.iloc[-1] >= s).mean() * 100, raw=False)
    df["phase"] = df.date.map(phase_of)
    return df


def phase_of(d):
    if d < pd.Timestamp("2017-01-01"):
        return "1_2016新股高送转"
    if d < pd.Timestamp("2018-01-01"):
        return "2_2017高送转除权下跌"
    if d < pd.Timestamp("2021-01-01"):
        return "3_2018-20地量熊市"
    if d < pd.Timestamp("2023-01-01"):
        return "4_2021-22题材波动"
    if d < pd.Timestamp("2025-07-01"):
        return "5_2023-24转型HJT"
    if d <= pd.Timestamp("2026-04-13"):
        return "6_2025-07至2026-04主升"
    return "7_2026-04-14后崩塌"


def append_latest(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(HIT, read_only=True)
    new = con.sql(
        """select date,open,high,low,close,volume,amount from v_daily
           where thscode='603778.SH' and date>'2026-10-29' order by date""").df()
    con.close()
    if new.empty:
        return df
    new["preclose"] = [df.close.iloc[-1], *new.close.iloc[:-1].tolist()]
    new["pct_chg"] = (new.close / new.preclose - 1) * 100
    new["turn"] = new.volume / FLOAT_SHARES * 100
    new["year"] = new.date.dt.year
    new["is_latest"] = True
    keep = ["date", "open", "high", "low", "close", "preclose", "volume",
            "amount", "turn", "pct_chg", "year", "is_latest"]
    out = pd.concat([df, new[keep]], ignore_index=True)
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    # 2021-06 后无除权，最新两根用 10/29 的后复权/不复权比例换算
    factor = df.close_h.iloc[-1] / df.close.iloc[-1]
    latest_mask = out["date"] > df.date.iloc[-1]
    for col in ("open", "high", "low", "close"):
        out.loc[latest_mask, f"{col}_h"] = out.loc[latest_mask, col] * factor
    # 最新两根补齐上下文（只用于展示，不进入历史统计）
    c = out.close_h
    for n in (5, 20, 60, 250):
        out[f"ma{n}"] = c.rolling(n).mean()
    out["hi250"] = out.high_h.rolling(250).max()
    out["lo250"] = out.low_h.rolling(250).min()
    out["pos250"] = (c - out.lo250) / (out.hi250 - out.lo250) * 100
    out["dd250"] = (c / out.hi250 - 1) * 100
    out["ret20"] = c.pct_change(20) * 100
    out["band"] = pd.cut(out.turn, BINS, labels=LABELS, right=False)
    out["phase"] = out.date.map(phase_of)
    out["turn_pct250"] = out.turn.rolling(250, min_periods=120).apply(
        lambda s: (s.iloc[-1] >= s).mean() * 100, raw=False)
    out["is_latest"] = out["is_latest"].fillna(False).astype(bool) if "is_latest" in out else False
    return out


def band_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("band", observed=True)
    top_years = g.year.apply(lambda s: "、".join(f"{y}" for y in s.value_counts().head(3).index))
    t = pd.DataFrame({
        "区间(%)": g.size().index.astype(str),
        "天数": g.size().values,
        "占比(%)": (g.size() / len(df) * 100).round(1).values,
        "换手中位(%)": g.turn.median().round(2).values,
        "成交额中位(亿)": (g.amount.median() / 1e8).round(2).values,
        "前20日涨跌中位(%)": g.ret20.median().round(1).values,
        "250日位置中位(%)": g.pos250.median().round(0).values,
        "站上均线20(%)": (g.above_ma20.mean() * 100).round(0).values,
        "站上均线250(%)": (g.above_ma250.mean() * 100).round(0).values,
        "20日波动中位(%)": g.vol20.median().round(0).values,
        "涨停频率(%)": (g.limit_up.mean() * 100).round(1).values,
        "跌停频率(%)": (g.limit_dn.mean() * 100).round(1).values,
        "T+1中位(%)": g.f1.median().round(2).values,
        "T+5中位(%)": g.f5.median().round(2).values,
        "T+20中位(%)": g.f20.median().round(2).values,
        "T+20均值(%)": g.f20.mean().round(2).values,
        "T+20胜率(%)": (g.f20.apply(lambda s: (s > 0).mean()) * 100).round(0).values,
        "20日最大浮盈中位(%)": g.mfe20.median().round(1).values,
        "20日最大浮亏中位(%)": g.mae20.median().round(1).values,
        "主要年份": top_years.values,
    })
    return t


def yearly_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("year")
    t = pd.DataFrame({
        "年份": g.size().index,
        "交易日": g.size().values,
        "换手中位(%)": g.turn.median().round(1).values,
        "换手均值(%)": g.turn.mean().round(1).values,
        "换手P90(%)": g.turn.quantile(.9).round(1).values,
        "换手最大(%)": g.turn.max().round(1).values,
        "收盘中位": g.close.median().round(2).values,
        "成交额中位(亿)": (g.amount.median() / 1e8).round(2).values,
        "全年涨跌(%)": ((g.close_h.last() / g.close_h.first() - 1) * 100).round(0).values,
    })
    return t


def phase_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("phase", observed=True)
    t = pd.DataFrame({
        "阶段": g.size().index,
        "交易日": g.size().values,
        "换手中位(%)": g.turn.median().round(1).values,
        "换手P90(%)": g.turn.quantile(.9).round(1).values,
        "换手最大(%)": g.turn.max().round(1).values,
        "起点收盘": g.close.first().round(2).values,
        "终点收盘": g.close.last().round(2).values,
        "区间涨跌(%)": ((g.close_h.last() / g.close_h.first() - 1) * 100).round(0).values,
        "成交额中位(亿)": (g.amount.median() / 1e8).round(2).values,
        "20日波动中位(%)": g.vol20.median().round(0).values,
    })
    return t


def band_phase_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["phase", "band"], observed=True)
    t = pd.DataFrame({
        "交易日": g.size().values,
        "T+20中位(%)": g.f20.median().round(1).values,
        "T+20均值(%)": g.f20.mean().round(1).values,
        "T+20胜率(%)": (g.f20.apply(lambda s: (s > 0).mean()) * 100).round(0).values,
        "20日最大浮亏中位(%)": g.mae20.median().round(1).values,
        "20日最大浮盈中位(%)": g.mfe20.median().round(1).values,
    }, index=g.size().index)
    t.index.names = ["阶段", "换手区间(%)"]
    return t.reset_index()


def high_turn_events(df: pd.DataFrame) -> pd.DataFrame:
    hot = df[df.turn >= 20].copy()  # 保留原始索引，后面要用 df 的前后行
    hot["gap"] = hot.date.diff().dt.days.fillna(999)
    hot["cluster"] = (hot.gap > 10).cumsum()
    rows = []
    for _, s in hot.groupby("cluster"):
        i0, i1 = s.index[0], s.index[-1]
        prev_close = df.close_h.iloc[i0 - 1] if i0 > 0 else s.close_h.iloc[0]
        end_close = s.close_h.iloc[-1]
        f20 = (df.close_h.iloc[i1 + 20] / end_close - 1) * 100 if i1 + 20 < len(df) else np.nan
        f60 = (df.close_h.iloc[i1 + 60] / end_close - 1) * 100 if i1 + 60 < len(df) else np.nan
        rows.append({
            "起始日": s.date.iloc[0].date(),
            "结束日": s.date.iloc[-1].date(),
            "交易日": len(s),
            "换手中位(%)": round(s.turn.median(), 1),
            "换手最高(%)": round(s.turn.max(), 1),
            "起始收盘": round(s.close.iloc[0], 2),
            "结束收盘": round(s.close.iloc[-1], 2),
            "期间涨跌(%)": round((end_close / prev_close - 1) * 100, 0),
            "前20日涨跌(%)": round(df.ret20.iloc[i0], 0) if pd.notna(df.ret20.iloc[i0]) else np.nan,
            "结束250日位置(%)": round(df.pos250.iloc[i1], 0) if pd.notna(df.pos250.iloc[i1]) else np.nan,
            "后20日(%)": round(f20, 0) if pd.notna(f20) else np.nan,
            "后60日(%)": round(f60, 0) if pd.notna(f60) else np.nan,
        })
    return pd.DataFrame(rows)


def current_analogs(df: pd.DataFrame) -> pd.DataFrame:
    conds = {
        "全部交易日": pd.Series(True, index=df.index),
        "深度回撤>60% + 换手<5%": (df.dd250 < -60) & (df.turn < 5),
        "深度回撤>60% + 换手<5% + 站上均线20": (df.dd250 < -60) & (df.turn < 5) & (df.close_h > df.ma20),
        "深度回撤>60% + 换手<5% + 跌破均线20": (df.dd250 < -60) & (df.turn < 5) & (df.close_h <= df.ma20),
        "深度回撤>60% + 换手5-8%": (df.dd250 < -60) & (df.turn >= 5) & (df.turn < 8),
        "250日低位<30 + 换手3.5-5%": (df.pos250 < 30) & (df.turn >= 3.5) & (df.turn < 5),
        "250日低位<30 + 换手5-8%": (df.pos250 < 30) & (df.turn >= 5) & (df.turn < 8),
        "近一年换手分位<10%": df.turn_pct250 < 10,
    }
    rows = []
    for name, m in conds.items():
        s = df[m]
        rows.append({
            "条件": name,
            "样本天数": len(s),
            "T+5中位(%)": round(s.f5.median(), 1) if len(s) else np.nan,
            "T+20中位(%)": round(s.f20.median(), 1) if len(s) else np.nan,
            "T+20均值(%)": round(s.f20.mean(), 1) if len(s) else np.nan,
            "T+20胜率(%)": round((s.f20 > 0).mean() * 100) if len(s) else np.nan,
            "T+60中位(%)": round(s.f60.median(), 1) if len(s) else np.nan,
            "T+60胜率(%)": round((s.f60 > 0).mean() * 100) if len(s) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(f"{ROOT}/03_结果", exist_ok=True)
    df = load_data()
    hist = df.copy()                      # 历史统计样本（至 2026-10-29）
    full = append_latest(df)              # 展示用，补 10/30、11/02

    band_table(hist).to_csv(f"{ROOT}/03_结果/换手区间统计.csv", index=False, encoding="utf-8-sig")
    yearly_table(hist).to_csv(f"{ROOT}/03_结果/年度换手.csv", index=False, encoding="utf-8-sig")
    phase_table(hist).to_csv(f"{ROOT}/03_结果/阶段换手.csv", index=False, encoding="utf-8-sig")
    band_phase_table(hist).to_csv(f"{ROOT}/03_结果/区间与阶段.csv", index=False, encoding="utf-8-sig")
    high_turn_events(hist).to_csv(f"{ROOT}/03_结果/极端换手事件.csv", index=False, encoding="utf-8-sig")
    current_analogs(hist).to_csv(f"{ROOT}/03_结果/当前同类样本.csv", index=False, encoding="utf-8-sig")

    cols = ["date", "open", "high", "low", "close", "preclose", "volume", "amount",
            "turn", "pct_chg", "ret20", "ma20", "ma60", "pos250", "dd250", "vol20",
            "f5", "f20", "f60", "mfe20", "mae20", "band", "phase", "turn_pct250"]
    out = full.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out["is_latest"] = out.get("is_latest", False)
    out[cols + ["is_latest"]].to_csv(f"{ROOT}/03_结果/日度明细.csv", index=False, encoding="utf-8-sig")

    # 供图表脚本使用
    full.to_pickle(f"{ROOT}/01_数据/603778_full.pkl")
    hist.to_pickle(f"{ROOT}/01_数据/603778_hist.pkl")

    print("最新交易:", full.tail(3)[["date", "close", "turn", "ma20", "pos250", "dd250"]].round(2).to_string(index=False))
    print("\n区间统计:\n", band_table(hist).to_string(index=False))
    print("\n阶段统计:\n", phase_table(hist).to_string(index=False))
    print("\n当前同类样本:\n", current_analogs(hist).to_string(index=False))


if __name__ == "__main__":
    main()
